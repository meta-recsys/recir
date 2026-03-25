# pyre-strict
import logging
import operator
from typing import Callable, List, Set, Tuple

import torch


MIN_ATEN_OPS_TO_LOWER = 10

logger: logging.Logger = logging.getLogger(__name__)


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

    subgraph_module = create_subgraph_module(inputs, body, outputs)
    readable_tag = dumper(str(subgraph_module.graph))
    setattr(gm, subgraph_name, module_factory(subgraph_module))

    insertion_point = subgraph_nodes[-1].next
    for node in prologue:
        insertion_point.prepend(node)

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
            gm.graph.erase_node(node)


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

    for idx, subgraph_nodes in enumerate(nodes_per_subgraph):
        subgraph_name = f"{subgraph_prefix}_{idx}"
        lower_subgraph_nodes(gm, subgraph_name, subgraph_nodes, module_factory, dumper)

    gm.graph.lint()
    gm.recompile()
    return gm
