# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""What a snapshot records about an FX graph."""

from __future__ import annotations

import collections
import enum
import functools
import hashlib
import json
import math
from typing import Any

import torch
from recir.observability.guard import contained
from recir.pipeline import PIPELINE_PARTITION_META, PIPELINE_REGION_META
from torch import fx
from torch.fx.node import _get_qualified_name

# RecIR's node.meta keys shown in the graph text, and each one's field in a node row.
_RECIR_META: dict[str, str] = {
    PIPELINE_REGION_META: "region",
    PIPELINE_PARTITION_META: "stage",
}
_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.int64: "i64",
    torch.int32: "i32",
    torch.int16: "i16",
    torch.int8: "i8",
    torch.uint8: "u8",
    torch.bool: "b8",
}
# First schema arguments through which RecIR ops receive a module's identity;
# it embeds a heap address that differs every process.
_IDENTITY_ARGS = ("handle", "instance_id")
# Higher-order ops that run their subgraph exactly once per call, by __name__;
# any other HOP may run a subgraph never or many times.
_RUN_ONCE_HOPS: frozenset[str] = frozenset(
    {
        "invoke_subgraph",
        "wrap",
        "wrap_with_set_grad_enabled",
        "wrap_with_autocast",
        "wrap_activation_checkpoint",
        "tag_activation_checkpoint",
        "dynamo_bypassing_wrapper",
        "hints_wrapper",
        "strict_mode",
        "local_map_hop",
        "invoke_quant",
        "invoke_quant_packed",
    }
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _type_name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _target(node: fx.Node) -> str:
    target = node.target
    if isinstance(target, str):
        return target
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    try:
        return _get_qualified_name(target)
    except Exception:
        return _type_name(target)


def _dtype(dtype: torch.dtype) -> str:
    return _DTYPES.get(dtype) or str(dtype).replace("torch.", "")


@functools.lru_cache(maxsize=4096)
def _expr_text(expr: Any) -> str:
    # Printing a sympy expression is slow, and a graph repeats a few of them.
    return str(expr)


def _sym(value: torch.SymInt | torch.SymFloat | torch.SymBool) -> str:
    return _expr_text(value.node.expr)


def _dim(d: int | torch.SymInt) -> int | str:
    # int() on a SymInt would specialize it and install a guard.
    if isinstance(d, torch.SymInt):
        const = d.node.maybe_as_int()
        return const if const is not None else _sym(d)
    return int(d)


def _float(value: float) -> float | dict[str, str]:
    return value if math.isfinite(value) else {"float": str(value)}


def _out(value: Any) -> dict[str, Any] | None:
    if isinstance(value, torch.Tensor):
        return {"shape": [_dim(d) for d in value.shape], "dtype": _dtype(value.dtype)}
    if isinstance(value, torch.SymInt):
        const = value.node.maybe_as_int()
        return {"sym": const if const is not None else _sym(value)}
    if isinstance(value, torch.SymFloat):
        const = value.node.maybe_as_float()
        return {"sym": _float(const) if const is not None else _sym(value)}
    if isinstance(value, torch.SymBool):
        const = value.node.maybe_as_bool()
        return {"sym": const if const is not None else _sym(value)}
    if isinstance(value, (bool, int)):
        return {"sym": value}
    if isinstance(value, float):
        return {"sym": _float(value)}
    return None


def _arg(a: Any) -> Any:
    if isinstance(a, fx.Node):
        return "%" + a.name
    if isinstance(a, str):
        # A string literal starting with % is escaped, so it never reads as a node.
        return "%" + a if a.startswith("%") else a
    if a is None or isinstance(a, (bool, int)):
        return a
    if isinstance(a, float):
        return _float(a)
    if isinstance(a, (list, tuple)):
        return [_arg(x) for x in a]
    if isinstance(a, dict):
        return {"dict": {str(k): _arg(v) for k, v in a.items()}}
    if isinstance(
        a,
        (
            torch.device,
            torch.dtype,
            torch.layout,
            torch.memory_format,
            torch.SymInt,
            torch.SymFloat,
            torch.SymBool,
            enum.Enum,
        ),
    ):
        return {"repr": str(a)}
    if isinstance(a, torch.Tensor):
        return {"type": "torch.Tensor", **(_out(a) or {})}
    return {"type": _type_name(a)}


def _identity_arg(node: fx.Node) -> str | None:
    target = node.target
    if not isinstance(target, torch._ops.OpOverload) or not target.namespace.startswith(
        "recir"
    ):
        return None
    schema_args = target._schema.arguments
    if (
        schema_args
        and schema_args[0].name in _IDENTITY_ARGS
        and node.args
        and isinstance(node.args[0], (int, str))
    ):
        return schema_args[0].name
    return None


def _node_row(node: fx.Node) -> dict[str, Any]:
    row: dict[str, Any] = {"name": node.name, "op": node.op}
    if node.op != "output":
        row["target"] = _target(node)
    if node.args:
        args = _arg(list(node.args))
        identity = _identity_arg(node) if node.op == "call_function" else None
        if identity is not None:
            args[0] = {"repr": identity}
        row["args"] = args
    if node.kwargs:
        row["kwargs"] = _arg(dict(node.kwargs))["dict"]
    val = node.meta.get("val")
    if isinstance(val, (list, tuple)):
        row["outs"] = [_out(v) for v in val]
    elif val is not None:
        row.update(_out(val) or {})
    if node.op == "placeholder":
        fqn = node.meta.get("fqn")
        if isinstance(fqn, str):
            row["fqn"] = fqn
        batch_dim = node.meta.get("batch_dim")
        if node.meta.get("is_batch_input") and isinstance(batch_dim, int):
            row["batch_dim"] = batch_dim
    for key, field in _RECIR_META.items():
        if key in node.meta:
            row[field] = node.meta[key]
    return row


def _is_ref(a: Any) -> bool:
    return isinstance(a, str) and a.startswith("%") and not a.startswith("%%")


def _strip_literals(a: Any) -> Any:
    if isinstance(a, list):
        return [_strip_literals(x) for x in a]
    if isinstance(a, dict):
        if "repr" in a or "type" in a or "float" in a:
            return "L"
        return {k: _strip_literals(v) for k, v in a.items()}
    return a if _is_ref(a) else "L"


def _table_hashes(rows: list[dict[str, Any]]) -> dict[str, str]:
    # Topology hashes node references by position, so it ignores names and literals.
    index = {row["name"]: i for i, row in enumerate(rows)}

    def ref(a: Any) -> Any:
        if isinstance(a, list):
            return [ref(x) for x in a]
        if isinstance(a, dict):
            return {k: ref(v) for k, v in a.items()}
        return index.get(a[1:], a) if _is_ref(a) else a

    topology = [
        (
            row["op"],
            None if row["op"] in ("placeholder", "get_attr") else row.get("target"),
            ref(_strip_literals(row.get("args", []))),
            ref(_strip_literals(row.get("kwargs", {}))),
        )
        for row in rows
    ]
    shape = [
        (row.get("shape"), row.get("dtype"), row.get("sym"), row.get("outs"))
        for row in rows
    ]
    return {
        "topology": _digest(topology),
        "shape": _digest(shape),
        "content": _digest(rows),
    }


def _subgraph_modules(
    owner: fx.GraphModule, node: fx.Node
) -> list[tuple[str, fx.GraphModule]]:
    # A HOP takes its subgraphs as get_attr args, alone or in a list (switch's branches).
    args = [x for a in node.args for x in (a if isinstance(a, (list, tuple)) else (a,))]
    subgraphs = []
    for a in args:
        if isinstance(a, fx.Node) and a.op == "get_attr" and isinstance(a.target, str):
            obj: Any = owner
            try:
                for part in a.target.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if isinstance(obj, fx.GraphModule):
                subgraphs.append((a.target.split(".")[-1], obj))
    return subgraphs


def _region(tag: Any) -> str | None:
    # A pipeline region tag may be any Hashable; a manifest holds it as text.
    return None if tag is None else str(tag)


def _collective_op(node: fx.Node) -> str | None:
    # The functional collective a node issues, such as all_reduce; None if it issues none.
    if not isinstance(node.target, torch._ops.OpOverload):
        return None
    target = str(node.target)
    if target.startswith("_c10d_functional.") and "wait_tensor" not in target:
        return target.split(".")[1]
    return None


def _collectives(
    gm: fx.GraphModule,
) -> tuple[dict[str, int], str | None, list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []

    def walk(
        owner: fx.GraphModule, path: str, enclosing: str | None
    ) -> tuple[dict[str, int], list[Any]]:
        census: collections.Counter[str] = collections.Counter()
        order: list[Any] = []
        for node in owner.graph.nodes:
            # An untagged node in a HOP's subgraph runs in the HOP's region.
            region = _region(node.meta.get(PIPELINE_REGION_META))
            if region is None:
                region = enclosing
            if node.op == "call_function" and isinstance(
                node.target, torch._ops.HigherOrderOperator
            ):
                op = getattr(node.target, "__name__", str(node.target))
                subgraphs = {
                    name: walk(module, f"{path}{node.name}/{name}/", region)
                    for name, module in _subgraph_modules(owner, node)
                }
                if op in _RUN_ONCE_HOPS:
                    for sub_census, sub_order in subgraphs.values():
                        census.update(sub_census)
                        order.extend(sub_order)
                    continue
                counted = {name: c for name, (c, _) in subgraphs.items() if c}
                if counted:
                    kept.append(
                        {
                            "node": f"{path}{node.name}",
                            "op": op,
                            "region": region,
                            "subgraphs": counted,
                        }
                    )
                orders = {name: o for name, (_, o) in subgraphs.items() if o}
                if orders:
                    order.append([op, region, orders])
            elif node.op == "call_function" and (op := _collective_op(node)):
                census[op] += 1
                last = node.args[-1] if node.args else None
                order.append([op, region, last if isinstance(last, str) else None])
        return dict(census), order

    census, order = walk(gm, "", None)
    return census, (_digest(order) if order else None), kept


class GraphDump:
    """
    One FX graph being recorded: the one place a snapshot's views of it are computed.

    This class is responsible for every view a snapshot takes of the graph:
    the ``print_readable`` text tlparse shows, its node counts, its in-memory
    node rows and their hashes, its collective census, and the graph's block
    in a manifest. It never talks to tlparse; the compile writing the graph
    decides what is written.
    """

    def __init__(self, gm: fx.GraphModule) -> None:
        self.gm = gm

    @property
    def counts(self) -> dict[str, int]:
        """Node counts: all nodes, and placeholders, ``call_function`` and ``call_module`` nodes."""
        counts = {"nodes": 0, "placeholder": 0, "call_function": 0, "call_module": 0}
        for node in self.gm.graph.nodes:
            counts["nodes"] += 1
            if node.op in counts:
                counts[node.op] += 1
        return counts

    def text(self) -> str:
        """The graph's ``print_readable`` text, with strides, devices, and RecIR's region and partition tags."""
        return self.gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            additional_meta=list(_RECIR_META),
        )

    def rows(self) -> list[dict[str, Any]]:
        """
        One in-memory row per node, in graph order, with each per-process
        module identity masked; never written.
        """
        return [_node_row(node) for node in self.gm.graph.nodes]

    def collectives(self) -> tuple[dict[str, int], str | None, list[dict[str, Any]]]:
        """
        The collective census, a hash of the collectives' order, and the
        collectives kept per subgraph. A higher-order op's subgraph
        collectives count in the census only for a HOP that runs its subgraph
        once; any other HOP's, such as ``torch.cond``'s branches, are kept
        per subgraph.
        """
        return _collectives(self.gm)

    def fields(self, collectives: bool = False) -> dict[str, Any]:
        """
        The graph's block in a manifest: its counts and the topology, shape
        and content hashes of its rows, and with ``collectives`` the
        collective census.
        """
        rows = None
        with contained("node rows not built") as fault:
            rows = self.rows()
        out: dict[str, Any] = {
            "counts": self.counts,
            "hashes": _table_hashes(rows) if rows is not None else None,
        }
        fault.note(out, "rows")
        if collectives:
            census, order, kept = self.collectives()
            out["collectives"] = census
            out["subgraph_collectives"] = kept
            out["collective_order"] = order
        return out
