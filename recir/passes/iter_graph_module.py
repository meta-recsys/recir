# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import inspect
import logging
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Set, Tuple, Type

import torch.nn as nn
from recir.utils.graph import clone_subgraph, get_output, is_leaf_subgraph
from torch import fx
from torch.fx.graph import _PyTreeCodeGen, PythonCode
from torch.fx.node import Argument
from torch.profiler import record_function
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten


logger: logging.Logger = logging.getLogger("IterGraphModule")


class IterGraph(fx.Graph):
    """``IterGraph`` is used to perform cross-iteration optimization.

    ``IterGraph`` keeps track of the 3 graphs, self (the original graph), setup graph, and
    cleanup graph. The 3 graphs should be identical copies of a ``fx.Graph``.

    IterGraph subclass fx.Graph to override the necessary APIs that will be used
    when constructing a optimization, e.g., communication fusion. IterGraph also
    provides APIs that originally belong to fx.Node and all these APIs will have
    ``node_`` prefix. For example, ``IterGraph.node_prepend`` is the equivalence
    of ``fx.Node.prepend``. Note that all the optimizations must be constructed
    using these APIs.
    """

    _codegen: Any

    def __init__(
        self,
        orig_graph: fx.Graph,
        setup_graph: fx.Graph,
        cleanup_graph: fx.Graph,
        owning_module: Optional[fx.GraphModule] = None,
        tracer_cls: Optional[Type["fx.Tracer"]] = None,
        tracer_extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(owning_module, tracer_cls, tracer_extras)

        output_vals = self.graph_copy(orig_graph, {}, return_output_node=True)
        # TODO: if we do ``deepcopy(_codegen)`` and the input argument contains
        # a dictionary with the form of Dict[torch.Tensor, Any], the
        # torch.fx._pytree.treen_flatten_spec will not be able to flatten the
        # dict -- the torch.Tensor will be duplicated because the _input_spec
        # will save the ``keys`` of a dictionary (the values are not saved).
        self._codegen = copy.deepcopy(orig_graph._codegen)
        assert isinstance(output_vals, tuple)
        output_val, old_output_val = output_vals
        super().output(output_val, type_expr=getattr(old_output_val, "type", None))

        self.setup_graph = setup_graph
        self.cleanup_graph = cleanup_graph
        self._all_graphs: Tuple[fx.Graph, ...] = (
            self.setup_graph,
            self.cleanup_graph,
            cast(fx.Graph, super()),  # pyre-ignore
        )

        self._setup_mapping: Dict[fx.Node, fx.Node] = {}
        self._cleanup_mapping: Dict[fx.Node, fx.Node] = {}
        self._freeze_cross_iter_movement = False
        self._cross_iter_block_count = 0

        for node, setup_node, cleanup_node in zip(
            self.nodes, self.setup_graph.nodes, self.cleanup_graph.nodes
        ):
            self._setup_mapping[node] = setup_node
            self._cleanup_mapping[node] = cleanup_node

        self.num_extra_output = 0

    def _lookup_node(self, node: fx.Node, graph: fx.Graph) -> Optional[fx.Node]:
        if graph == self.setup_graph:
            return self._setup_mapping.get(node, None)
        elif graph == self.cleanup_graph:
            return self._cleanup_mapping.get(node, None)
        return node

    def _fx_graph_call(
        self, graph: fx.Graph, func: str, *args: Any, **kwargs: Any
    ) -> Any:
        fx_graph: fx.Graph = (
            graph if graph != self else cast(fx.Graph, super())  # pyre-ignore
        )
        return getattr(fx_graph, func)(*args, **kwargs)

    def _insert_context(self, func: str, node: Optional[fx.Node]) -> Any:
        class _InsertPoint:
            def __init__(self, insert_points: List[Any]) -> None:
                self.insert_points = insert_points

            def __enter__(self) -> None:
                pass

            def __exit__(self, type: Any, value: Any, tb: Any) -> None:
                for insert_point in self.insert_points:
                    insert_point.__exit__(type, value, tb)

        insert_points = []
        for graph in self._all_graphs:
            if node:
                actual_node = self._lookup_node(node, graph)
                assert actual_node is not None, "Cannot handle None case now."
            else:
                actual_node = node
            insert_points.append(getattr(graph, func)(actual_node))

        return _InsertPoint(insert_points)

    # pyrefly: ignore [bad-override]
    def inserting_after(self, n: Optional[fx.Node]) -> Any:
        if self._freeze_cross_iter_movement:
            return super().inserting_after(n)
        return self._insert_context("inserting_after", n)

    # pyrefly: ignore [bad-override]
    def inserting_before(self, n: Optional[fx.Node]) -> Any:
        if self._freeze_cross_iter_movement:
            return super().inserting_before(n)
        return self._insert_context("inserting_before", n)

    def _forward_subgraph_inputs(  # noqa: C901 -- graph boundary rewrite transaction
        self, subgraph: List[fx.Node], graph: fx.Graph, erase_node: bool
    ) -> int:
        """Turn the inputs of a subgraph into the extra output of the entire graph.

        If ``erase_node`` is True, the subgraph will be erased from the graph -- essentially forward the inputs
        of the subgraph to the output of the graph.
        """
        output = get_output(graph)
        inputs = []
        all_nodes: Set[fx.Node] = set(subgraph)

        for node in subgraph:
            node_inputs = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            for _input in node_inputs:
                if not isinstance(_input, fx.Node):
                    continue
                if _input in all_nodes:
                    continue
                inputs.append(_input)

        if erase_node:
            # We have to remove the node in the reversed order to ensure the
            # node has zero users.
            erased = set()
            for node in reversed(subgraph):
                if len(node.users) == 1:
                    key = next(iter(node.users.keys()))
                    if key == output:
                        flatten_args, spec = tree_flatten((output.args, output.kwargs))
                        if node not in flatten_args:
                            # This optimizer node from the legacy _SPMD tracing.
                            node.users.clear()
                        elif str(node.target).startswith("aten.copy_"):
                            # This is the case where the optimizer is
                            # functionalized with copy_.
                            for i in range(len(flatten_args)):
                                if flatten_args[i] == node:
                                    flatten_args[i] = node.args[0]
                        else:
                            # We have not figured out semantics of forwarding
                            # all diff ops.
                            raise RuntimeError(
                                f"IterGraph does not how to forward the output of {node}"
                            )
                        output.args, output.kwargs = tree_unflatten(flatten_args, spec)

                # This is the step case where there is a virtual data dependency
                # (in-place update) between step and optimizer. And
                # functionalize_optim add this dependency
                for user in list(node.users.keys()):
                    if user in erased:
                        node.users.pop(user)
                if node.users:
                    raise RuntimeError(
                        "IterGraph has not supported moving the nodes that "
                        "produce users output result. "
                        f"Error node: {node}."
                    )
                self._fx_graph_call(graph, "erase_node", node)
                erased.add(node)

        # Add all the extra output nodes into a list and append the list to
        # the original output.args[0].
        if self.num_extra_output:
            # If the extra-output list already exist, just use it.
            cast(List[fx.Node], output.args[0][-1]).extend(inputs)  # type: ignore[index]
            new_output = output.args[0]
        else:
            # When adding the extra-output list, out_spec of _PyTreeCodeGen
            # must be updated accordingly.
            if isinstance(graph._codegen, _PyTreeCodeGen):
                codegen = graph._codegen
                new_output = list(output.args[0])  # type: ignore[arg-type]
                # pyrefly: ignore [bad-argument-type]
                new_output.append(inputs)
                assert codegen.pytree_info.out_spec is not None
                original_tree_out = tree_unflatten(
                    cast(List[Any], output.args[0]),
                    codegen.pytree_info.out_spec,  # pyre-ignore[6]
                )
                # Use None as a placeholder. If we use the extra-output list
                # the list will be flatten as well and put into out_spec.
                _, out_spec = tree_flatten((original_tree_out, None))
                codegen.pytree_info = codegen.pytree_info._replace(out_spec=out_spec)
            else:
                new_output = (output.args[0], inputs)
        self._fx_graph_call(graph, "erase_node", output)
        self._fx_graph_call(graph, "output", new_output)

        logger.info("Extended outputs from the subgraph inputs: %s", str(inputs))
        return len(inputs)

    def _forward_inputs_to_subgraph(
        self, subgraph: List[fx.Node], graph: fx.Graph, extra_input: int
    ) -> None:
        """Create extra input nodes and forward the input nodes to the ``subgraph``.

        The external input nodes of ``subgraph`` (nodes that are not in ``subgraph``) will replaced by the newly
        created input nodes.
        """
        placeholders = [node for node in graph.nodes if str(node.op) == "placeholder"]
        assert placeholders, "No placeholders are found"
        # Append the extra input nodes to the current input nodes.
        with self._fx_graph_call(graph, "inserting_after", placeholders[-1]):
            new_input_nodes = list(
                reversed(
                    [
                        self._fx_graph_call(
                            graph,
                            "placeholder",
                            f"cross_iter_input_{self._cross_iter_block_count}_{i}",
                        )
                        for i in reversed(range(extra_input))
                    ]
                )
            )

        # Update the inputs of subgraph to use the newly created input nodes.
        all_nodes = set(subgraph)
        new_input_index = 0
        for node in subgraph:
            node_inputs, spec = tree_flatten((node.args, node.kwargs))
            new_node_inputs = []
            for input_node in node_inputs:
                if not isinstance(input_node, fx.Node) or input_node in all_nodes:
                    new_node_inputs.append(input_node)
                else:
                    new_node_inputs.append(new_input_nodes[new_input_index])
                    new_input_index += 1
            node.args, node.kwargs = tree_unflatten(new_node_inputs, spec)
        assert new_input_index == len(new_input_nodes), (
            f"More inputs than needed {len(new_input_nodes)} > {new_input_index}"
        )

        # Update the in_spec of _PyTreeCodeGen if in_spec is not None (the new
        # SPMD makes in_spec as None).
        if (
            isinstance(graph._codegen, _PyTreeCodeGen)
            and graph._codegen.pytree_info.in_spec is not None
        ):
            codegen = graph._codegen
            original_tree_in = tree_unflatten(placeholders, codegen.pytree_info.in_spec)
            _, in_spec = tree_flatten(tuple(list(original_tree_in) + new_input_nodes))
            codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)
            for new_input in new_input_nodes:
                codegen.pytree_info.orig_args.append(new_input.name)
            codegen.pytree_info = codegen.pytree_info._replace(in_spec=in_spec)

    def move_to_next_iter_before(
        self, subgraph: List[fx.Node], target_node: fx.Node
    ) -> None:
        """Move the ``subgraph`` to the next iteration before ``target_node``.

        The ``subgraph`` is a list of fx.Node and must satisfy the following
        restrictions:
            1. The order of the nodes in ``subgraph`` must obey the topological
               sort order.
            2. The users of the node in ``subgraph`` must be one of the following:
                a.) the user is also a node in ``subgraph``.
                b.) the user is the output of the full graph.
                c.) the node has users (side effect node).
        """
        if self._freeze_cross_iter_movement:
            raise RuntimeError(
                "The cross-iteration movement has been frozen for the given IterGraph."
            )

        if not is_leaf_subgraph(self, subgraph):
            raise ValueError(
                "The target nodes for ``move_to_next_iter_before`` must "
                "satisfy one of the following conditions: 1) the user of the "
                "node is in the target nodes, 2) the user is the output of the "
                "graph, 3) there are no users -- the node is a side-effect node. "
            )

        self._cross_iter_block_count += 1
        # The main graph must be the last one to be modified. Otherwise, the
        # mapping may change and hence introduce incorrect mapping for setup
        # and cleanup graphs.

        # For the setup graph, no additional input is needed but additional
        # outputs will be created. The additional output represents the input of
        # the action to be moved to the next iteration -- main graph.
        setup_subgraph: List[fx.Node] = []
        for node in subgraph:
            mapped_node = self._lookup_node(node, self.setup_graph)
            assert mapped_node is not None
            setup_subgraph.append(mapped_node)
        setup_extra_input = self._forward_subgraph_inputs(
            subgraph=setup_subgraph,
            graph=self.setup_graph,
            erase_node=True,
        )

        # For the cleanup graph, additional input is required to get the output
        # from the last iteration -- main graph. Additional nodes are also
        # needed to perform the action moved from the last iteration.
        target_cleanup_node = self._lookup_node(target_node, self.cleanup_graph)
        assert target_cleanup_node is not None, "The target_cleanup_node is None."
        cleanup_subgraph: List[fx.Node] = []
        for node in subgraph:
            mapped_node = self._lookup_node(node, self.cleanup_graph)
            assert mapped_node is not None
            cleanup_subgraph.append(mapped_node)
        cloned_subgraph = clone_subgraph(
            self.cleanup_graph,
            cleanup_subgraph,
            target=target_cleanup_node,
        )
        self._forward_inputs_to_subgraph(
            cloned_subgraph, self.cleanup_graph, setup_extra_input
        )

        # For the main graph, additional input will be created to represent
        # the output from the last iteration -- main graph or setup graph.
        # Additional output will also be generated to represent the input for
        # the next iteration -- the main graph or the cleanup graph.
        main_extra_input = self._forward_subgraph_inputs(
            subgraph=subgraph, graph=self, erase_node=False
        )
        assert main_extra_input == setup_extra_input
        for node in subgraph:
            target_node.prepend(node)
        self._forward_inputs_to_subgraph(subgraph, self, main_extra_input)

        # TODO: This is a temporary solution. We are going to remove DCE usage
        # or have something to replace fx DCE.
        for node in self.cleanup_graph.nodes:
            if len(node.users) == 0:
                node.users["__hold__"] = None  # type: ignore[index]
        for node in self.nodes:
            if len(node.users) == 0:
                node.users["__hold__"] = None  # type: ignore[index]
        self.num_extra_output += main_extra_input

    def move_before(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        for graph in self._all_graphs:
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None
            for actual_node in actual_nodes:
                actual_target_node.prepend(actual_node)  # pyre-ignore[6]

    def move_after(self, nodes: List[fx.Node], target_node: fx.Node) -> None:
        for graph in self._all_graphs:
            actual_nodes = [self._lookup_node(node, graph) for node in nodes]
            actual_target_node = self._lookup_node(target_node, graph)
            for actual_node in actual_nodes:
                assert actual_target_node is not None
                actual_target_node.append(actual_node)  # pyre-ignore[6]
                actual_target_node = actual_node

    # pyre-ignore
    def call_function(
        self,
        the_function: Callable[..., Any],
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        type_expr: Optional[Any] = None,
    ) -> fx.Node:
        if self._freeze_cross_iter_movement:
            return super().call_function(the_function, args, kwargs, type_expr)

        setup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        setup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.setup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )
        cleanup_args = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            args,
        )
        cleanup_kwargs = tree_map(
            lambda arg: self._lookup_node(arg, self.cleanup_graph)
            if isinstance(arg, fx.Node)
            else arg,
            kwargs,
        )

        setup_node = self.setup_graph.call_function(
            the_function, setup_args, setup_kwargs, type_expr
        )
        main_node = super().call_function(the_function, args, kwargs, type_expr)
        cleanup_node = self.cleanup_graph.call_function(
            the_function, cleanup_args, cleanup_kwargs, type_expr
        )
        self._setup_mapping[main_node] = setup_node
        self._cleanup_mapping[main_node] = cleanup_node
        return main_node

    def erase_node(
        self,
        to_erase: fx.Node,
        *,
        cascade_to_mirrors: bool = True,
        unsafe_drop_users: bool = False,
    ) -> None:
        """Erase ``to_erase`` from the main graph and (optionally) its mirrors.

        Args:
            to_erase: The node to erase from the main graph.
            cascade_to_mirrors: When ``True`` (default), also erase the
                setup/cleanup mirror nodes -- the legacy three-graph
                "stay in sync" behavior. When ``False``, only the main
                graph is mutated; the mirror parents keep running their
                original computation (the right choice when the caller
                is rewriting only the main graph, e.g.
                ``lower_subgraph_nodes`` splicing a ``call_module``
                replacement into main but leaving setup/cleanup to
                execute the original unlowered ops on CPU).
            unsafe_drop_users: Only relevant when
                ``cascade_to_mirrors=True``. Allows the cascade to call
                ``replace_all_uses_with(None)`` on a mirror that still
                has live users. Safe only when those user-mirrors are
                themselves about to be erased in the same sweep
                (profiler ``enter``/``exit`` markers, dead-output
                nodes). Outside those cases the substituted ``None``
                survives into the recompiled mirror forward and
                crashes downstream consumers -- the runtime regression
                that motivated commit 18 (see
                ``recir/lowering/xla_device_state.py:_copy_state_into``).

        Background:
            Commit 16 generalized commit 15's profiler-marker fix by
            always doing ``replace_all_uses_with(None)`` before mirror
            erase. This worked for marker-style ops but corrupted
            mirror parents whose output depended on the erased node
            (e.g. Adam's ``addcdiv_`` flowing to ``output``). Commit
            18 splits the policy: ``unsafe_drop_users`` is the
            explicit opt-in for the marker case, and
            ``cascade_to_mirrors=False`` is the opt-out for callers
            that should leave the mirror parents intact.
        """
        if self._freeze_cross_iter_movement:
            return super().erase_node(to_erase)

        if not cascade_to_mirrors:
            super().erase_node(to_erase)
            return

        # Nodes added directly to the main graph (e.g. via
        # ``super().call_function`` while ``_freeze_cross_iter_movement`` was
        # toggled) have no mirror entry in ``_setup_mapping`` /
        # ``_cleanup_mapping``. Treat a missing mirror as a no-op rather than
        # asserting -- the main-graph erase below is still meaningful, and
        # there is nothing to clean up in the sub-graphs.
        setup_node = self._lookup_node(to_erase, self.setup_graph)
        cleanup_node = self._lookup_node(to_erase, self.cleanup_graph)
        if setup_node is not None:
            self._erase_mirror_node(
                setup_node,
                self.setup_graph,
                "setup",
                unsafe_drop_users=unsafe_drop_users,
            )
        super().erase_node(to_erase)
        if cleanup_node is not None:
            self._erase_mirror_node(
                cleanup_node,
                self.cleanup_graph,
                "cleanup",
                unsafe_drop_users=unsafe_drop_users,
            )

    def _erase_mirror_node(
        self,
        mirror_node: fx.Node,
        mirror_graph: fx.Graph,
        graph_label: str,
        *,
        unsafe_drop_users: bool,
    ) -> None:
        """Erase ``mirror_node`` from ``mirror_graph`` after handling its users.

        See ``IterGraph.erase_node`` for the motivating bug. The split
        keeps the policy (raise vs drop) in one place and avoids
        duplicating it between the setup and cleanup branches.
        """
        if mirror_node.users:
            if unsafe_drop_users:
                # Caller asserts the dropped users are about to be
                # erased themselves (profiler-marker pattern) or
                # otherwise tolerate ``None`` substitution.
                #
                # Correctness verification of ``replace_all_uses_with(None)``:
                # the FX implementation
                # (caffe2/torch/fx/node.py:replace_all_uses_with) calls
                # ``use_node._replace_input_with(self, replace_with)`` for
                # each user, which simply rewrites the user's
                # ``args``/``kwargs`` tuple to substitute the replacement.
                # ``None`` is a valid leaf value in FX arg trees (op
                # signatures accept ``Optional[Tensor]`` and pytree leaves
                # may include ``None``), so the rewrite itself is safe;
                # ``None`` only becomes a problem when the user op runs
                # and the op cannot tolerate ``None``.  The type signature
                # claims ``Node`` so the ``pyre-ignore[6]`` is required.
                #
                # Callers that pass ``unsafe_drop_users=True`` are
                # responsible for either
                # (a) erasing the now-``None``-fed users in the same sweep
                # (profiler enter/exit, dead-output trees), or
                # (b) confirming the user op tolerates a ``None`` argument
                # at runtime (matches the original commit 15/16 fix path
                # for marker ops whose forward is a no-op).
                # Outside those cases the substituted ``None`` survives
                # into the recompiled mirror forward; the assertion in the
                # docstring above is enforced at call sites, not here.
                mirror_node.replace_all_uses_with(None)  # pyre-ignore[6]
            else:
                user_names = [u.name for u in mirror_node.users]
                raise RuntimeError(
                    f"IterGraph.erase_node: {graph_label}-graph mirror of "
                    f"{mirror_node.name} still has users {user_names}. "
                    "Erasing without rewriting these users would substitute "
                    "None in the mirror parent's recompiled forward, which "
                    "crashes at runtime when the consumer is the ``output`` "
                    "node or otherwise expects a Tensor. "
                    "Options: "
                    "(a) pass ``cascade_to_mirrors=False`` to leave the "
                    "mirror nodes intact -- correct when the caller is "
                    "only rewriting the main graph (e.g. "
                    "``lower_subgraph_nodes``); "
                    "(b) perform the output/user rewrite cascade in the "
                    f"{graph_label}_graph parent BEFORE calling erase_node; "
                    "(c) pass ``unsafe_drop_users=True`` if the user-mirror "
                    "is also about to be erased in the same sweep and "
                    "``None`` substitution is safe (e.g. profiler "
                    "enter/exit markers)."
                )
        mirror_graph.erase_node(mirror_node)

    def placeholder(
        self,
        name: str,
        type_expr: Optional[Any] = None,
        default_value: Any = inspect.Signature.empty,
    ) -> fx.Node:
        if self._freeze_cross_iter_movement:
            return super().placeholder(name, type_expr, default_value)

        main_placeholder = super().placeholder(name, type_expr, default_value)
        setup_placeholder = self.setup_graph.placeholder(name, type_expr, default_value)
        cleanup_placeholder = self.cleanup_graph.placeholder(
            name, type_expr, default_value
        )
        self._setup_mapping[main_placeholder] = setup_placeholder
        self._cleanup_mapping[main_placeholder] = cleanup_placeholder
        return main_placeholder

    def output(self, result: Argument, type_expr: Optional[Any] = None) -> fx.Node:
        if self._freeze_cross_iter_movement:
            return super().output(result, type_expr)

        main_output = super().output(result, type_expr)
        setup_result = tree_map(
            lambda _result: self._lookup_node(_result, self.setup_graph)
            if isinstance(_result, fx.Node)
            else _result,
            result,
        )
        cleanup_result = tree_map(
            lambda _result: self._lookup_node(_result, self.cleanup_graph)
            if isinstance(_result, fx.Node)
            else _result,
            result,
        )
        self.setup_graph.output(setup_result, type_expr)
        self.cleanup_graph.output(cleanup_result, type_expr)

        return main_output

    def lint(self) -> None:
        self.setup_graph.lint()
        super().lint()
        self.cleanup_graph.lint()

    def node_prepend(self, target_node: fx.Node, node: fx.Node) -> None:
        """Prepend node to target_node."""
        if self._freeze_cross_iter_movement:
            target_node.prepend(node)
            return

        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, "The node is None"
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None, "The target node is None"
            actual_target_node.prepend(actual_node)

    def node_append(self, target_node: fx.Node, node: fx.Node) -> None:
        """Append node to target_node."""
        if self._freeze_cross_iter_movement:
            target_node.append(node)
            return

        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            assert actual_node is not None, f"The actual node is None, {node}."
            actual_target_node = self._lookup_node(target_node, graph)
            assert actual_target_node is not None, (
                f"The actual target node is None, {target_node}."
            )
            actual_target_node.append(actual_node)

    def node_set_args(self, node: fx.Node, args: Tuple[Argument, ...]) -> None:
        if self._freeze_cross_iter_movement:
            node.args = args
            return

        setup_args = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), args
        )
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None
        setup_node.args = setup_args
        cleanup_args = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), args
        )
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None
        cleanup_node.args = cleanup_args
        node.args = args

    def node_set_kwargs(self, node: fx.Node, kwargs: Dict[str, Argument]) -> None:
        if self._freeze_cross_iter_movement:
            node.kwargs = kwargs
            return

        setup_kwargs = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.setup_graph), kwargs
        )
        setup_node = self._lookup_node(node, self.setup_graph)
        assert setup_node is not None
        setup_node.kwargs = setup_kwargs
        cleanup_kwargs = tree_map_only(
            fx.Node, lambda _arg: self._lookup_node(_arg, self.cleanup_graph), kwargs
        )
        cleanup_node = self._lookup_node(node, self.cleanup_graph)
        assert cleanup_node is not None
        cleanup_node.kwargs = cleanup_kwargs
        node.kwargs = kwargs

    def node_replace_all_uses_with(
        self,
        node: fx.Node,
        replace_with: fx.Node,
        delete_user_cb: Callable[[fx.Node], bool] = lambda user: True,
        *,
        propagate_meta: bool = False,
    ) -> List[fx.Node]:
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            actual_replace_with = self._lookup_node(replace_with, graph)
            assert actual_node is not None
            ret = actual_node.replace_all_uses_with(
                actual_replace_with,  # pyre-ignore[6]
                delete_user_cb,
                propagate_meta=propagate_meta,
            )
        return ret  # type: ignore[possibly-undefined]

    def node_add_user(self, node: fx.Node, user: Any) -> None:
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            actual_node.users[actual_user_node] = None  # type: ignore[index]

    def node_remove_user(self, node: fx.Node, user: Any) -> None:
        for graph in self._all_graphs:
            actual_node = self._lookup_node(node, graph)
            if isinstance(user, fx.Node):
                actual_user_node = self._lookup_node(user, graph)
            else:
                actual_user_node = user
            assert actual_node is not None
            del actual_node.users[actual_user_node]  # type: ignore[arg-type]

    def keep_unused_nodes(self) -> None:
        for node in self.nodes:
            if len(node.users) == 0 and str(node.op) != "output":
                self.node_add_user(node, "__hold__")

    def functionalize_optim(self) -> None:
        # IterGraph can only support full graph (fwd+bwd+optim). As optimizer
        # is not a functional call (it is inplace op), this method adds the of
        # the optimizer call. This method has strong assumption of the optimizer
        # and may not always be working. This method is intended be a temporary
        # solution only.

        # TODO: remove this API after DCE is removed
        for node in reversed(self.nodes):
            if node.name.startswith("output"):
                output_node = node
            elif node.name.startswith(
                "_fused_adam_",
            ):
                optim_node = node
            elif node.name.startswith(
                "_foreach_add_",
            ):
                step_node = node
                self.node_add_user(optim_node, output_node)  # type: ignore[possibly-undefined]
                self.node_add_user(step_node, optim_node)  # type: ignore[possibly-undefined]

    def defunctionalize_optim(self) -> None:
        # TODO: remove this API after DCE is not used with IterGraph
        for graph in self._all_graphs:
            for node in reversed(graph.nodes):
                if node.name.startswith("output"):
                    output_node = node
                elif node.name.startswith(
                    "_fused_adam_",
                ):
                    optim_node = node
                elif node.name.startswith(
                    "_foreach_add_",
                ):
                    step_node = node
                    optim_node.users.pop(output_node, None)  # type: ignore[possibly-undefined]
                    step_node.users.pop(optim_node, None)  # type: ignore[possibly-undefined]

    def freeze_cross_iter_movement(self) -> None:
        self._freeze_cross_iter_movement = True


class IterGraphModule(nn.Module):
    """``IterGraphModule`` provides the ability to do cross-iteration optimization.

    Given a ``fx.GraphModule``, main_gm, ``IterGraphModule`` internally
    duplicate it to 3 copies and redirect the ``forward`` request to a different
    ``fx.GraphModule`` based on the iteration count. This allows users to do
    graph optimizations that across iterations (e.g., moving collective wait in
    the backward to the forward of the next iteration).

    Note that users must call the APIs provided by ``IterGraphModule`` or
    ``IterGraph`` to rewrite the graph so that ``IterGraphModule`` can keep the
    data dependency for all 3 graphs.
    """

    setup_gm: fx.GraphModule
    cleanup_gm: fx.GraphModule
    main_gm: fx.GraphModule

    def __init__(
        self,
        main_gm: fx.GraphModule,
        max_iters: int = -1,
    ) -> None:
        super().__init__()

        def _copy_gm(src: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
            gm = fx.GraphModule(src, graph)
            gm.meta = getattr(graph, "meta", {})
            return gm

        self.setup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.cleanup_gm = _copy_gm(main_gm, copy.deepcopy(main_gm.graph))
        self.main_gm = _copy_gm(
            main_gm,
            IterGraph(main_gm.graph, self.setup_gm.graph, self.cleanup_gm.graph),
        )

        self._iter = 0
        self._max_iters = max_iters
        self._previous_output: Tuple[Any, ...] = ()
        self._num_extra_output = 0
        self._is_frozen = False

    def finalize_setup(self) -> None:
        """Freeze cross-iteration movement and prepare internal states.

        This method must be called after all graph optimizations are done
        and before forward() is called.
        """
        if not self._is_frozen:
            self.graph.freeze_cross_iter_movement()
            self._num_extra_output = self.graph.num_extra_output
            self._is_frozen = True

        self._iter = 0

    def _run(
        self, gm: fx.GraphModule, last_iter: bool, *args: Any, **kwargs: Any
    ) -> Any:
        if self._num_extra_output > 0:
            new_args = args + (self._previous_output)
            output = gm(*new_args, **kwargs)
            if not last_iter:
                assert len(output) == 2
                self._previous_output = tuple(output[-1])
                assert len(self._previous_output) > 0, (
                    "There should be at least one extra output."
                )
                output = output[0]
        else:
            # No cross-iteration optimization is done. Simply call the
            # GraphModule.
            output = gm(*args, **kwargs)
        return output

    def forward(self, *args: Any, last_iter: bool = False, **kwargs: Any) -> Any:
        self._iter += 1
        last_iter = last_iter or self._iter == self._max_iters
        if last_iter:
            logger.info("Using the cleanup graph")
            gm = self.cleanup_gm
            profiler_string = "## IterGraphModule: Cleanup Graph ##"
            self._iter = 0
        elif self._iter == 1:
            logger.info("Using the setup graph")
            gm = self.setup_gm
            profiler_string = "## IterGraphModule: Setup Graph ##"
        else:
            gm = self.main_gm
            if self._iter == 2:
                logger.info("Using the main graph")
                profiler_string = "## IterGraphModule -- Maybe Compiling ##"
            else:
                profiler_string = "## IterGraphModule ##"

        with record_function(profiler_string):
            return self._run(gm, last_iter, *args, **kwargs)

    @property
    def graph(self) -> IterGraph:
        return cast(IterGraph, self.main_gm.graph)

    def recompile(self) -> PythonCode:
        self.setup_gm.recompile()
        self.cleanup_gm.recompile()
        return self.main_gm.recompile()

    def freeze_cross_iter_movement(self) -> None:
        # TODO: remove this API once it is not used.
        self.graph.freeze_cross_iter_movement()
        self._num_extra_output = self.graph.num_extra_output

    def print_readable(self, print_output: bool = True) -> str:
        return self.main_gm.print_readable(print_output)

    def print_all_graphs(self) -> None:
        logger.info("Printing the three fx.Graph:")
        logger.info("1. Setup fx.Graph:")
        logger.info("%s", self.setup_gm.graph)
        logger.info("2. Main fx.Graph:")
        logger.info("%s", self.main_gm.graph)
        logger.info("3. Cleanup fx.Graph:")
        logger.info("%s", self.cleanup_gm.graph)

    def print_all_graph_modules(self) -> None:
        logger.info("Printing the three fx gm:")
        logger.info("1. Setup fx.GraphModule:")
        logger.info("%s", self.setup_gm.print_readable(False))
        logger.info("2. Main fx.GraphModule:")
        logger.info("%s", self.main_gm.print_readable(False))
        logger.info("3. Cleanup fx.GraphModule:")
        logger.info("%s", self.cleanup_gm.print_readable(False))

    def add_subgraph_module(self, name: str, module: nn.Module) -> None:
        """Register ``module`` as a submodule on all three sub-GraphModules.

        ``lower_subgraph_nodes`` (and other graph-surgery callers) splice a
        ``call_module(target=name)`` node into the main IterGraph. The
        post-surgery ``fx.Graph.lint`` walks the call site's
        ``owning_module`` (the ``main_gm``) and raises::

            RuntimeError: Node <name> target <name> references nonexistent
            attribute <name>

        whenever the submodule attribute is only installed on the outer
        ``IterGraphModule`` (a plain ``setattr(iter_gm, name, module)``
        registers it on the ``nn.Module`` parent but *not* on the inner
        ``fx.GraphModule`` whose ``owning_module`` lint actually walks).

        We propagate to all three sub-GMs (main / setup / cleanup) so the
        attribute is reachable regardless of which graph the runtime
        ``forward`` dispatches into for a given iteration. The
        ``call_module`` node itself currently only exists in the main
        IterGraph (``IterGraph.create_node`` is not overridden to mirror
        into setup/cleanup), but registering on all three keeps the
        sub-GM submodule registries consistent and future-proof.
        """
        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            sub_gm = getattr(self, gm_name, None)
            if sub_gm is None:
                continue
            setattr(sub_gm, name, module)

    def named_xla_modules(self) -> Iterator[Tuple[str, str, Any]]:
        """Yield ``(gm_name, sub_path, _XLAModule)`` for every ``_XLAModule``
        reachable across all three sub-graphs.

        ``gm_name`` is one of ``"setup_gm"`` / ``"main_gm"`` / ``"cleanup_gm"``;
        ``sub_path`` is the dotted attribute path returned by
        ``named_modules()``. Callers (e.g. ``recir.lowering.aot``) use the
        ``(gm_name, sub_path)`` tuple as a stable artifact key.

        The ``_XLAModule`` class is imported lazily to avoid a circular
        import — ``recir.lowering.xla_lower`` imports ``IterGraphModule``.
        Pyre cannot follow that lazy import without a BUCK dep that would
        recreate the cycle, so the import and the resulting ``isinstance``
        check are suppressed locally.
        """
        # pyre-ignore[21]: Lazy import to break a build-time circular dep
        # with recir.lowering.xla_lower; see docstring.
        from recir.lowering.xla_lower import _XLAModule

        for gm_name in ("setup_gm", "main_gm", "cleanup_gm"):
            sub_gm = getattr(self, gm_name, None)
            if sub_gm is None:
                continue
            for sub_path, mod in sub_gm.named_modules():
                if isinstance(mod, _XLAModule):  # pyre-ignore[16]
                    yield gm_name, sub_path, mod
