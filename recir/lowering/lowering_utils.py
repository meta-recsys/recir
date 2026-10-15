# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from typing import Any, Callable, List, Set, Tuple

import torch


MIN_ATEN_OPS_TO_LOWER = 10

logger: logging.Logger = logging.getLogger(__name__)


def has_meta_kernel(op: torch._ops.OpOverload) -> bool:
    """Whether ``op`` can run under FakeTensor/meta, as FakeTensorMode decides.

    Uses the computed dispatch table, so composite ops (e.g. ``aten.detach``,
    ``aten.zeros``) whose Meta behavior comes from a CompositeExplicitAutograd
    kernel count; ``OpOverload.has_kernel_for_dispatch_key(Meta)`` only sees a
    direct Meta registration and rejects them.
    """
    return torch._C._dispatch_has_computed_kernel_for_dispatch_key(op.name(), "Meta")


def strip_profiler_nodes(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove profiler ops before backend segmentation.

    Their opaque handles cannot cross lowered subgraph boundaries.
    """
    from recir.passes.iter_graph_module import IterGraph

    changed = False
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target.name().startswith("profiler::"):
                node.replace_all_uses_with(None)  # type: ignore[arg-type]
                if isinstance(gm.graph, IterGraph):
                    gm.graph.erase_node(node, unsafe_drop_users=True)
                else:
                    gm.graph.erase_node(node)
                changed = True
    if changed:
        gm.graph.eliminate_dead_code()
        gm.recompile()
    return gm


def create_subgraph_module(
    inputs: List[torch.fx.Node],
    body: List[torch.fx.Node],
    outputs: List[torch.fx.Node],
) -> torch.fx.GraphModule:
    """Create a standalone fx.GraphModule from a list of nodes.

    Builds a new graph with placeholder inputs, copies the body nodes,
    and sets the output to the specified output nodes.
    """
    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_to_subgraph_node = {}
    for idx, inp in enumerate(inputs):
        subgraph_inp = subgraph.placeholder(name=f"arg_{idx}")
        subgraph_inp.meta = inp.meta
        node_to_subgraph_node[inp] = subgraph_inp

    for node in body:
        subgraph_node = subgraph.node_copy(
            node, arg_transform=lambda x: node_to_subgraph_node[x]
        )
        # Output placement resolved during compilation is a boundary contract.
        # Share it with the parent node consumed by downstream subgraphs.
        if node in outputs:
            subgraph_node.meta = node.meta
        node_to_subgraph_node[node] = subgraph_node

    subgraph.output(result=tuple(node_to_subgraph_node[x] for x in outputs))
    subgraph.eliminate_dead_code()
    subgraph.lint()
    return torch.fx.GraphModule(root={}, graph=subgraph)


def is_container_node(node: torch.fx.Node) -> bool:
    """Check if a node produces a container (tuple/list) consumed via getitem."""
    if any(user.target == operator.getitem for user in node.users):
        assert all(user.target == operator.getitem for user in node.users), (
            "Malformed graph: a container node is used as input for non-getitem nodes."
            "\nNode: {fmt_node}\nUsers: {fmt_users}".format(
                fmt_node=node.format_node(),
                fmt_users="\n".join(u.format_node() for u in node.users),  # pyre-ignore
            )
        )
        return True
    return False


def default_subgraph_predicate(nodes: List[torch.fx.Node]) -> bool:
    """Filter out subgraphs with too few aten ops to be worth lowering."""
    num_aten_ops = len([n for n in nodes if str(n.target).startswith("aten.")])
    return num_aten_ops >= MIN_ATEN_OPS_TO_LOWER


def lower_subgraph_nodes(
    gm: torch.fx.GraphModule,
    subgraph_name: str,
    subgraph_nodes: List[torch.fx.Node],
    module_factory: Callable[[torch.fx.GraphModule], torch.nn.Module],
    dumper: Callable[[str], str],
    all_lowered_nodes: Set[torch.fx.Node],
) -> None:
    """Extract subgraph_nodes from gm, wrap via module_factory, and splice back.

    This is the shared graph-surgery logic used by both the Inductor and
    XLA lowering paths.  The caller provides a ``module_factory`` that
    takes the extracted subgraph ``fx.GraphModule`` and returns the
    ``nn.Module`` wrapper (e.g. ``_InductorModule`` or ``_XLAModule``).

    All graph inputs and outputs are required to be tensors.  Container
    nodes (producing tuples consumed via ``operator.getitem``) are
    automatically unpacked into leaf tensor nodes.
    """
    prologue: List[torch.fx.Node] = []
    inputs: List[torch.fx.Node] = []
    body: List[torch.fx.Node] = []
    visible: Set[torch.fx.Node] = set()

    def add_input(arg: torch.fx.Node) -> None:
        stack = [arg]
        while len(stack) != 0:
            node = stack.pop()
            if is_container_node(node):
                prologue.extend(user for user in node.users if user in subgraph_nodes)
                stack.extend(node.users)
            else:
                if node not in visible:
                    inputs.append(node)
                    visible.add(node)

    for node in subgraph_nodes:
        if node.op == "get_attr":
            inputs.append(node)
            visible.add(node)
            continue

        for arg in node.all_input_nodes:
            if arg not in visible:
                add_input(arg)

        if node not in prologue:
            body.append(node)
            visible.add(node)

    outputs: List[torch.fx.Node] = []

    def add_output(output: torch.fx.Node) -> None:
        stack = [output]
        while len(stack) != 0:
            node = stack.pop()
            if is_container_node(node):
                body.extend(node.users)
                stack.extend(node.users)
            elif not all(user in visible for user in node.users):
                if node not in outputs:
                    outputs.append(node)

    for node in body:
        if not all(user in visible for user in node.users):
            add_output(node)

    assert len(inputs) == len(set(inputs))
    assert len(outputs) == len(set(outputs))

    lowered = visible | all_lowered_nodes
    torch_output_indices = {
        idx
        for idx, output in enumerate(outputs)
        if any(u not in lowered and u.op != "output" for u in output.users)
    }
    final_output_indices = {
        idx
        for idx, output in enumerate(outputs)
        if any(u.op == "output" for u in output.users)
    }
    subgraph_module = create_subgraph_module(inputs, body, outputs)
    readable_tag = dumper(str(subgraph_module.graph))
    module = module_factory(subgraph_module)
    module.final_output_indices = final_output_indices  # pyre-ignore[16]
    if torch_output_indices:
        module.torch_output_indices = torch_output_indices  # pyre-ignore[16]
    # Register the lowered subgraph as a submodule attribute.
    #
    # For a plain ``fx.GraphModule`` a bare ``setattr`` is sufficient: the
    # call_module node we insert below targets the parent gm directly so
    # ``getattr(gm, subgraph_name)`` resolves and ``gm.graph.lint`` passes.
    #
    # For an ``IterGraphModule`` the parent is an ``nn.Module`` wrapper
    # around three ``fx.GraphModule`` sub-graphs (setup / main / cleanup).
    # ``setattr(iter_gm, subgraph_name, module)`` installs on the wrapper
    # only, but ``IterGraph.lint`` walks the *main* IterGraph whose
    # ``owning_module`` is ``iter_gm.main_gm`` (a sub-GM) -- producing::
    #
    #     RuntimeError: Node <subgraph_name> target <subgraph_name>
    #     references nonexistent attribute <subgraph_name>
    #
    # See ``IterGraphModule.add_subgraph_module`` for the helper that
    # propagates the registration to all three sub-GMs so the call_module
    # node we splice in below resolves under any of them.
    from recir.passes.iter_graph_module import IterGraph, IterGraphModule

    if isinstance(gm, IterGraphModule):
        gm.add_subgraph_module(subgraph_name, module)
    else:
        setattr(gm, subgraph_name, module)

    insertion_point = subgraph_nodes[-1].next
    for node in prologue:
        insertion_point.prepend(node)

    # When the parent is an ``IterGraphModule``, the main IterGraph
    # mirrors every mutation into ``setup_gm.graph`` /
    # ``cleanup_gm.graph`` by default. That mirroring is the WRONG
    # semantics here: ``segment_and_lower`` only splices the lowered
    # ``call_module`` into the MAIN IterGraph, so cascading the body
    # erase into setup/cleanup would strip the original ops from
    # mirror parents while leaving them with no replacement -- the
    # mirror parents are supposed to keep running the original
    # unlowered ops on CPU. We therefore pass
    # ``cascade_to_mirrors=False`` so each erase only touches the
    # main IterGraph and leaves the setup/cleanup mirror parents
    # functionally intact.
    #
    # This is the architecturally clean replacement for commit 16's
    # silent ``replace_all_uses_with(None)`` cascade, which papered
    # over the original ``RuntimeError`` by substituting ``None``
    # into the mirror parents' output -- corrupting the recompiled
    # forward and crashing downstream consumers at runtime (see
    # commit 18 / canaries).
    is_iter_graph = isinstance(gm.graph, IterGraph)
    erase_kwargs: dict[str, Any] = (
        {"cascade_to_mirrors": False} if is_iter_graph else {}
    )

    with gm.graph.inserting_before(insertion_point):
        subgraph_call = gm.graph.create_node(
            op="call_module",
            target=subgraph_name,
            args=tuple(inputs),
            kwargs={"tag": readable_tag},
        )
        for idx, output in enumerate(outputs):
            new_output = gm.graph.create_node(
                op="call_function",
                target=operator.getitem,
                args=(subgraph_call, idx),
            )
            new_output.meta = output.meta
            output.replace_all_uses_with(new_output)

    for node in reversed(body + outputs):
        if len(node.users) == 0:
            gm.graph.erase_node(node, **erase_kwargs)


def segment_and_lower(
    gm: torch.fx.GraphModule,
    compatibility_fn: Callable[[torch.fx.Node], Tuple[bool, str]],
    module_factory: Callable[[torch.fx.GraphModule], torch.nn.Module],
    subgraph_prefix: str = "subgraph",
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """Segment a graph into compatible subgraphs and lower each one.

    This is the shared algorithm used by both ``partial_lower()`` (Inductor)
    and ``partial_lower_xla()`` (XLA).

    1. Walk the graph and segment contiguous runs of compatible ops.
    2. Filter subgraphs by size (``MIN_ATEN_OPS_TO_LOWER``) and user predicate.
    3. Extract each subgraph into a standalone ``fx.GraphModule``.
    4. Wrap each in a module produced by ``module_factory``.
    5. Splice the wrapped module back into the parent graph.

    Args:
        gm: The fx.GraphModule to partially lower.
        compatibility_fn: Per-node check returning (is_compatible, reason).
        module_factory: Given a subgraph GraphModule, returns the nn.Module
            wrapper (e.g. _InductorModule or _XLAModule).
        subgraph_prefix: Name prefix for extracted subgraph modules.
        node_predicate: Additional user predicate for node filtering.
        subgraph_predicate: Additional user predicate for subgraph filtering.
        dumper: Callback producing a human-readable label per subgraph.
    """
    nodes_per_subgraph: List[List[torch.fx.Node]] = [[]]
    ptr = next(iter(gm.graph.nodes))

    def _node_predicate(node: torch.fx.Node) -> Tuple[bool, str]:
        should_lower, reason = compatibility_fn(node)
        if not should_lower:
            return should_lower, reason
        if not node_predicate(node):
            return False, "user predicate"
        return True, ""

    while ptr.op != "output":
        if ptr.op == "placeholder":
            ptr = ptr.next
            continue
        should_lower, reason = _node_predicate(ptr)
        if should_lower:
            nodes_per_subgraph[-1].append(ptr)
        else:
            if len(nodes_per_subgraph[-1]) > 0:
                logger.warning(
                    "partial_lower: graph break at %s. Reason: %s",
                    str(ptr),
                    reason,
                )
            nodes_per_subgraph.append([])
        ptr = ptr.next

    nodes_per_subgraph = [
        nodes
        for nodes in nodes_per_subgraph
        if subgraph_predicate(nodes) and default_subgraph_predicate(nodes)
    ]

    all_lowered_nodes: Set[torch.fx.Node] = set()
    for nodes in nodes_per_subgraph:
        all_lowered_nodes.update(nodes)
    for idx, subgraph_nodes in enumerate(nodes_per_subgraph):
        subgraph_name = f"{subgraph_prefix}_{idx}"
        lower_subgraph_nodes(
            gm, subgraph_name, subgraph_nodes, module_factory, dumper, all_lowered_nodes
        )

    gm.graph.lint()
    gm.recompile()
    return gm
