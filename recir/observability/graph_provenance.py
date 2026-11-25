# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Training-node provenance and JSON artifacts for the published RecIR graph.

Every node of the captured inference graph keeps the name of the training-step
node it came from (``recir_train_node``); nodes a publish pass adds record the
pass instead (``recir_inserted_by``). With ``RECIR_GRAPH_ARTIFACT_DIR`` set, the
training step, the captured inference graph and the final publish graph are
written there as JSON, so a served node can be traced back to the training node
and module that produced it.
"""

import difflib
import json
import logging
import os
import re
from typing import Any

import torch

logger: logging.Logger = logging.getLogger(__name__)

TRAIN_NODE_KEY: str = "recir_train_node"
INSERTED_BY_KEY: str = "recir_inserted_by"
ARTIFACT_DIR_ENV: str = "RECIR_GRAPH_ARTIFACT_DIR"

_XAR_PREFIX: re.Pattern[str] = re.compile(r"/mnt/xarfuse/[^/]+/[^/]+/")


def _graph_modules(module: torch.nn.Module) -> list[torch.fx.GraphModule]:
    return [m for m in module.modules() if isinstance(m, torch.fx.GraphModule)]


def stamp_training_provenance(gm: torch.fx.GraphModule) -> None:
    """Record each node's own name as its training node."""
    for node in gm.graph.nodes:
        node.meta.setdefault(TRAIN_NODE_KEY, node.name)


def mark_inserted_nodes(module: torch.nn.Module, pass_name: str) -> int:
    """Attribute nodes with no provenance yet to ``pass_name``. Returns the count."""
    marked = 0
    for gm in _graph_modules(module):
        for node in gm.graph.nodes:
            if TRAIN_NODE_KEY in node.meta or INSERTED_BY_KEY in node.meta:
                continue
            node.meta[INSERTED_BY_KEY] = pass_name
            marked += 1
    return marked


def _target_name(target: object) -> str:
    if isinstance(target, str):
        return target
    if hasattr(target, "_schema") or isinstance(target, torch._ops.OpOverloadPacket):
        return str(target)
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    return f"{module}.{name}" if module and name else str(target)


def _value_summary(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": [str(s) for s in value.shape]}
    if isinstance(value, (list, tuple)):
        return [_value_summary(v) for v in value]
    if isinstance(value, (torch.SymInt, int, float, bool)):
        return str(value)
    return None


def _stack_summary(stack_trace: str | None, frames: int = 3) -> list[str]:
    if not stack_trace:
        return []
    lines = [
        ln.strip() for ln in stack_trace.splitlines() if ln.strip().startswith("File")
    ]
    return [_XAR_PREFIX.sub("", ln) for ln in lines[-frames:]]


def node_record(node: torch.fx.Node) -> dict[str, Any]:
    module_stack = node.meta.get("nn_module_stack") or {}
    return {
        "name": node.name,
        "op": node.op,
        "target": _target_name(node.target),
        "args": str(node.args)[:300],
        "kwargs": str(node.kwargs)[:200],
        "val": _value_summary(node.meta.get("val")),
        TRAIN_NODE_KEY: node.meta.get(TRAIN_NODE_KEY),
        INSERTED_BY_KEY: node.meta.get(INSERTED_BY_KEY),
        "module_stack": [str(path) for path, _ in module_stack.values()][-3:],
        "stack": _stack_summary(node.stack_trace),
    }


def write_graph_artifact(module: torch.nn.Module, name: str) -> str | None:
    """Write every GraphModule in ``module`` to ``$RECIR_GRAPH_ARTIFACT_DIR/<name>.json``,
    keyed by submodule path. Returns the path, or None when the env var is unset.
    Diagnostics never fail the publish."""
    directory = os.environ.get(ARTIFACT_DIR_ENV)
    if not directory:
        return None
    path = os.path.join(directory, f"{name}.json")
    try:
        os.makedirs(directory, exist_ok=True)
        graphs = {
            path_name or "<root>": [node_record(n) for n in child.graph.nodes]
            for path_name, child in module.named_modules()
            if isinstance(child, torch.fx.GraphModule)
        }
        with open(path, "w") as f:
            json.dump(graphs, f)
    except (OSError, TypeError, ValueError) as e:
        logger.warning("[recir-graph-artifact] could not write %s: %s", path, e)
        return None
    logger.info(
        "[recir-graph-artifact] wrote %s (%d graph(s), %d nodes)",
        path,
        len(graphs),
        sum(len(nodes) for nodes in graphs.values()),
    )
    return path


_OVERLOAD_SUFFIX: re.Pattern[str] = re.compile(
    r"\.(default|Tensor|Scalar|int|dim|self|using_ints|correction|Tensor_mode"
    r"|dim_IntList|start|start_step|memory_format|low)$"
)


_MODEL_PREFIX: re.Pattern[str] = re.compile(r"^(model\.)+")


def _alignment_key(record: dict[str, Any]) -> str:
    op = str(record["op"])
    target = str(record["target"]).replace("torch.ops.", "")
    target = _OVERLOAD_SUFFIX.sub("", target.replace("_operator.", "operator."))
    if op in ("call_module", "get_attr"):
        target = target.split(".")[-1]
    return f"{op}:{target}"


def align_served_nodes(
    publish_records: list[dict[str, Any]], served_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Map served graph node names to ``publish_graph`` records.

    A serving compiler that retraces the captured graph replays it in order but
    renames its nodes and drops node.meta, so nodes are matched by the longest
    common subsequence of (op, target). Nodes the serving side adds (such as a
    serving wrapper) stay unmatched."""
    matcher = difflib.SequenceMatcher(
        a=[_alignment_key(r) for r in publish_records],
        b=[_alignment_key(r) for r in served_records],
        autojunk=False,
    )
    mapping: dict[str, dict[str, Any]] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[served_records[block.b + k]["name"]] = publish_records[block.a + k]
    # Retracing can hoist weight reads to different positions, so match the get_attrs
    # the sequence missed by parameter path; serving wrappers add "model." prefixes.
    by_path = {
        _MODEL_PREFIX.sub("", str(r["target"])): r
        for r in publish_records
        if r["op"] == "get_attr"
    }
    for record in served_records:
        if record["op"] != "get_attr" or record["name"] in mapping:
            continue
        match = by_path.get(_MODEL_PREFIX.sub("", str(record["target"])))
        if match is not None:
            mapping[record["name"]] = match
    return mapping
