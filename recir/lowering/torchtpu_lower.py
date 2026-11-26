# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Partial-lowering of FX subgraphs onto the torch_tpu runtime.

Mirrors :mod:`recir.lowering.xla_lower` but lowers each compatible subgraph
via torch_tpu's ``TpuBackend`` (the same backend ``torch.compile(backend="tpu")``
uses) instead of torchax/JAX. ``TpuBackend`` runs ``aot_autograd`` ->
``StaticCompiler`` -> ``fx_to_mlir`` -> ``compile_mlir`` -> a PJRT executable.

This module deliberately does NOT import torchax/JAX so the torch_tpu path
carries no JAX dependency.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import fx
import torch.distributed as dist
import torch.utils._pytree as pytree
from recir.lowering.lowering_utils import segment_and_lower, strip_profiler_nodes
from recir.ops.decompositions import SPMD_DECOMP_TABLE
from torch._decomp import get_decompositions

logger: logging.Logger = logging.getLogger(__name__)

# torch_tpu lacks eager kernels for some ATen ops the HSTU multitask loss emits
# (binary_cross_entropy_with_logits, whose TPU path hits log_sigmoid_forward).
# Register their decompositions into recir's SPMD_DECOMP_TABLE — which make_fx
# consults at trace time (recir.api) — so they lower to supported ops instead of
# hitting the eager TPU dispatch. Registered at import of this module, i.e.
# whenever the torch_tpu backend is imported, which is before any trace.
SPMD_DECOMP_TABLE.update(
    get_decompositions(
        [
            torch.ops.aten.binary_cross_entropy_with_logits,
            torch.ops.aten.log_sigmoid_forward,
            torch.ops.aten.log_sigmoid_backward,
        ]
    )
)


_TPU_FBGEMM_LIB: Optional[torch.library.Library] = None


def _register_tpu_fbgemm_kernels() -> List[str]:
    """Give TorchTPU kernels to the fbgemm ops a RecIR graph may still hold.

    RecIR's trace decomposes fbgemm ops through SPMD_DECOMP_TABLE, but ops that
    custom ops call from their autograd (e.g. fbgemm::jagged_to_padded_dense_backward)
    reach the graph as they are. torchax lowers them itself; torch_tpu has no
    kernels for them. Each gets the table's pure-PyTorch implementation as its
    PrivateUse1 kernel, which torch_tpu's exporter records as ATen ops. Returns
    the ops registered by this call.
    """
    global _TPU_FBGEMM_LIB
    import recir.lowering.fbgemm_fallback  # noqa: F401 -- fills SPMD_DECOMP_TABLE

    registered = []
    for op, fn in list(SPMD_DECOMP_TABLE.items()):
        if not isinstance(op, torch._ops.OpOverload) or op.namespace != "fbgemm":
            continue
        overload = op._overloadname
        name = op._schema.name.split("::", 1)[1]
        if overload != "default":
            name = f"{name}.{overload}"
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            f"fbgemm::{name}", "PrivateUse1"
        ):
            continue
        if _TPU_FBGEMM_LIB is None:
            _TPU_FBGEMM_LIB = torch.library.Library("fbgemm", "IMPL")
        impl_fn = fn
        if len(op._schema.returns) == 2 and isinstance(
            op._schema.returns[1].type, torch._C.ListType
        ):

            def _wrap_tensor_list(
                *args: Any, _fn: Callable[..., Any] = fn, **kwargs: Any
            ) -> Any:
                out = _fn(*args, **kwargs)
                if (
                    isinstance(out, tuple)
                    and len(out) == 2
                    and isinstance(out[1], torch.Tensor)
                ):
                    return out[0], [out[1]]
                return out

            impl_fn = _wrap_tensor_list
            SPMD_DECOMP_TABLE[op] = impl_fn
        _TPU_FBGEMM_LIB.impl(name, impl_fn, "PrivateUse1")
        registered.append(f"fbgemm::{name}")
    return registered


def _is_torchtpu_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    """Per-node check: can this node be absorbed into a torch_tpu subgraph?

    torch_tpu lowers a subgraph by re-running it op-by-op on the tpu device
    (``fx_to_mlir``), so we let it attempt to lower EVERY node except the
    embedding graph-break markers (``xla_graph_break``), which must stay eager
    (they call back into the eager torchrec/SparseCore embedding). Any op
    torch_tpu cannot lower then surfaces as a loud ``fx_to_mlir``/compile error
    rather than being silently kept eager.

    This is intentionally more permissive than the XLA path's
    ``_is_xla_compatible``, which conservatively gates on meta-kernel
    registration and op allowlists. If a class of ops needs to stay eager on
    torch_tpu (e.g. ``_c10d_functional`` collectives), add a targeted reject
    here. ``profiler::`` ops never reach this check: ``partial_lower_torchtpu``
    strips them first.
    """
    # Override-dispatch (embedding graph-break) nodes must never be absorbed.
    if node.meta.get("xla_graph_break", False):
        return False, "graph break (override dispatch)"

    return True, ""


def _fresh_empty_tensors(gm: torch.fx.GraphModule) -> int:
    """Route every use of an empty intermediate through a fresh ``new_empty``.

    TorchTPU hands a view across its collective partitions as the whole
    underlying buffer, which an empty view (e.g. a zero-length slice of a
    padded batch tensor) does not match. An empty tensor holds no data, so a
    fresh one of the same shape is equivalent. Returns the number replaced.
    """
    replaced = 0
    for node in list(gm.graph.nodes):
        val = node.meta.get("val")
        if (
            node.op != "call_function"
            or not isinstance(val, torch.Tensor)
            or val.numel() != 0
            or not node.users
        ):
            continue
        with gm.graph.inserting_after(node):
            fresh = gm.graph.call_function(
                torch.ops.aten.new_empty.default, (node, list(val.shape))
            )
        fresh.meta["val"] = val
        node.replace_all_uses_with(
            fresh, delete_user_cb=lambda user, fresh=fresh: user is not fresh
        )
        replaced += 1
    if replaced:
        gm.recompile()
    return replaced


def _shape_mismatches(gm: torch.fx.GraphModule, args: List[Any]) -> List[str]:
    """Inputs whose shape differs from the shape traced for their placeholder.

    ``TpuBackend`` compiles from the first call's inputs, but sizes recorded
    in the graph (views, ``select_backward``'s ``input_sizes``) come from the
    trace, so the two must agree.
    """
    mismatches = []
    for node, arg in zip(gm.graph.find_nodes(op="placeholder"), args):
        val = node.meta.get("val")
        if (
            isinstance(arg, torch.Tensor)
            and isinstance(val, torch.Tensor)
            and arg.shape != val.shape
        ):
            mismatches.append(
                f"{node.name}: traced {tuple(val.shape)}, got {tuple(arg.shape)}"
            )
    return mismatches


# By default each subgraph is compiled by TpuBackend with its collectives marked
# SPMD-safe; "static" compiles the functionalized graph with StaticCompiler.
COMPILER_ENV = "RECIR_TORCHTPU_COMPILER"
# The node annotation torch.tpu.distributed.spmd_safe sets under compilation.
# TpuBackend's split compiler otherwise gives every collective a program of its
# own, so that ranks whose graphs differ around a collective cannot deadlock.
_SPMD_SAFE_METADATA_KEY = "spmd_safe"


def _compiler_mode() -> str:
    return os.environ.get(COMPILER_ENV, "donate")


def _uses_donated_static_compiler() -> bool:
    return _compiler_mode() == "donate"


def _uses_static_compiler() -> bool:
    return _compiler_mode() == "static"


def _mark_collectives_spmd_safe(gm: torch.fx.GraphModule) -> int:
    """Mark ``gm`` SPMD-safe when it contains collectives; returns collective count."""
    marked = sum(1 for node in gm.graph.nodes if _is_collective(node))
    if not marked:
        return 0
    for node in gm.graph.nodes:
        if node.op not in ("placeholder", "get_attr", "output"):
            node.meta.setdefault("custom", {})[_SPMD_SAFE_METADATA_KEY] = True
    return marked


def _is_collective(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and str(node.target).startswith("_c10d_functional.")
        and node.target is not torch.ops._c10d_functional.wait_tensor.default
    )


def _graph_fingerprint(gm: torch.fx.GraphModule) -> Tuple[int, Counter[str]]:
    """A hash of ``gm``'s structure, and its collectives counted by kind.

    The hash covers each node's op and output shapes and dtypes, and each
    collective's non-tensor arguments (reduce op, group): what a compiled
    collective depends on. Other literal arguments are left out, so a value
    that differs per rank without changing the program raises no alarm.
    """
    lines = []
    collectives: Counter[str] = Counter()
    for node in gm.graph.nodes:
        val = pytree.tree_map_only(
            torch.Tensor, lambda t: (str(t.dtype), tuple(t.shape)), node.meta.get("val")
        )
        line = f"{node.op} {node.target} {val}"
        if _is_collective(node):
            collectives[str(node.target)] += 1
            line += f" {[a for a in node.args if not isinstance(a, torch.fx.Node)]}"
        lines.append(line)
    digest = hashlib.sha256("\n".join(lines).encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=True), collectives


# Safety net for marking collectives SPMD-safe: without it, ranks that traced
# different graphs deadlock silently inside the TPU program. Runs once per
# subgraph compile (one all_gather of a hash), never per step.
def _check_same_graph_on_all_ranks(gm: torch.fx.GraphModule, tag: str) -> None:
    """Raise unless every rank is about to compile the same program as this one.

    A collective compiled into the program waits for the matching collective
    on every other rank, so a rank that traced a different graph hangs the job
    inside the TPU program. ``TpuBackend`` splits at collectives (or, with
    ``TORCH_TPU_SPMD_HANDSHAKE_STAGE``, compares graphs first); with collectives
    marked SPMD-safe, or under StaticCompiler, it does neither, so the ranks
    compare fingerprints over the host process group.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    digest, collectives = _graph_fingerprint(gm)
    logger.info(
        "torch_tpu subgraph (%s) fingerprint %016x: %d nodes, collectives %s",
        tag,
        digest & ((1 << 64) - 1),
        len(gm.graph.nodes),
        dict(collectives),
    )
    try:
        pg_backend = str(dist.get_backend())
    except Exception:
        pg_backend = "cpu"
    dev = "tpu" if pg_backend in ("tpu", "tpu_dist") else "cpu"
    mine = torch.tensor([digest], dtype=torch.int64, device=dev)
    gathered = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, mine)
    ranks_by_digest: Dict[int, List[int]] = defaultdict(list)
    for rank, value in enumerate(gathered):
        ranks_by_digest[int(value)].append(rank)
    if len(ranks_by_digest) > 1:
        groups = "; ".join(
            f"ranks {ranks}: {value & ((1 << 64) - 1):016x}"
            for value, ranks in ranks_by_digest.items()
        )
        raise RuntimeError(
            f"torch_tpu subgraph ({tag}) differs across ranks ({groups}). Its "
            "collectives would never match; compare the ranks' fingerprint log "
            "lines for this subgraph."
        )


def _functional_graph(
    gm: torch.fx.GraphModule,
    args: List[Any],
    decompositions: Optional[Dict[Any, Callable[..., Any]]] = None,
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """``gm`` without mutations, plus the inputs whose new values it returns.

    RecIR graphs update parameters and optimizer state in place. The functional
    graph returns ``gm``'s outputs flattened, followed by the new value of each
    mutated input; the second result lists those inputs' positions in order.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    def run(*inputs: Any) -> List[Any]:
        return pytree.tree_leaves(torch.fx.Interpreter(gm).run(*inputs))

    fgm = make_fx(
        torch.func.functionalize(run, remove="mutations"),
        decomposition_table=decompositions,
        tracing_mode="fake",
    )(*args)
    placeholders = {
        node: i for i, node in enumerate(fgm.graph.find_nodes(op="placeholder"))
    }
    # Functionalization writes each mutated input back with a trailing copy_.
    mutated: Dict[int, torch.fx.Node] = {}
    for node in list(fgm.graph.nodes):
        if node.target is torch.ops.aten.copy_.default and node.args[0] in placeholders:
            new_value = node.args[1]
            assert isinstance(new_value, torch.fx.Node)
            mutated[placeholders[node.args[0]]] = new_value
            node.replace_all_uses_with(new_value)
            fgm.graph.erase_node(node)
    output = next(iter(fgm.graph.find_nodes(op="output")))
    output.args = (list(output.args[0]) + list(mutated.values()),)
    fgm.graph.lint()
    fgm.recompile()
    return fgm, list(mutated)


def _unwrap_aliased_base(node: fx.Node) -> fx.Node:
    """Follow pure non-mutating view ops back to their underlying FX node."""
    cur = node
    while (
        cur.op == "call_function"
        and isinstance(cur.target, torch._ops.OpOverload)
        and cur.args
        and isinstance(cur.args[0], fx.Node)
    ):
        schema = cur.target._schema
        if (
            not schema.returns
            or schema.returns[0].alias_info is None
            or schema.returns[0].alias_info.is_write
            or not schema.arguments
            or schema.arguments[0].alias_info is None
            or schema.arguments[0].alias_info.is_write
        ):
            break
        cur = cur.args[0]
    return cur


def _collect_mutated_placeholders(gm: fx.GraphModule) -> Set[fx.Node]:
    """Return placeholder nodes mutated in-place inside ``gm``."""
    mutated: Set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            continue
        schema = node.target._schema
        for idx, schema_arg in enumerate(schema.arguments):
            alias = schema_arg.alias_info
            if alias is None or not alias.is_write:
                continue
            val = (
                node.args[idx]
                if idx < len(node.args)
                else node.kwargs.get(schema_arg.name)
            )
            candidates = val if isinstance(val, (list, tuple)) else (val,)
            for cand in candidates:
                if isinstance(cand, fx.Node):
                    base = _unwrap_aliased_base(cand)
                    if base.op == "placeholder":
                        mutated.add(base)
    return mutated


def _materialize_mutated_placeholder_outputs(gm: fx.GraphModule) -> None:
    """Insert trailing ``aten.alias`` nodes for mutated placeholders."""
    mutated = _collect_mutated_placeholders(gm)
    if not mutated:
        return
    output_node = next(
        (n for n in reversed(gm.graph.nodes) if n.op == "output"), None
    )
    if output_node is None:
        return
    alias_map: Dict[fx.Node, fx.Node] = {}
    with gm.graph.inserting_before(output_node):
        for p in list(gm.graph.nodes):
            if p.op == "placeholder" and p in mutated:
                alias_node = gm.graph.call_function(
                    torch.ops.aten.alias.default, (p,)
                )
                alias_node.meta = dict(p.meta)
                alias_map[p] = alias_node
    if not alias_map:
        return
    output_node.args = torch.fx.node.map_arg(
        output_node.args,
        lambda n: alias_map.get(n, n),
    )
    gm.recompile()


def _find_donatable_input_indices(
    gm: fx.GraphModule, args: Sequence[Any]
) -> List[int]:
    """Identify tensor-argument indices of mutated state placeholders safe to donate."""
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(args):
        return []
    mutated = _collect_mutated_placeholders(gm)
    if not mutated:
        return []
    ptr_counts = Counter(
        int(a.untyped_storage().data_ptr())
        for a in args
        if isinstance(a, torch.Tensor)
    )
    donated: List[int] = []
    tensor_idx = 0
    for placeholder, arg in zip(placeholders, args):
        if not isinstance(arg, torch.Tensor):
            continue
        if (
            ptr_counts[int(arg.untyped_storage().data_ptr())] == 1
            and arg.device.type == "tpu"
            and placeholder.meta.get("is_state_input", False)
            and placeholder in mutated
        ):
            donated.append(tensor_idx)
        tensor_idx += 1
    return donated


class _TorchTPUModule(torch.nn.Module):
    """Wraps an ``fx.GraphModule`` subgraph for lazy TorchTPU lowering."""

    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled: Optional[Callable[..., Any]] = None
        self.donated_inputs: Tuple[int, ...] = ()
        self._mutated: Optional[List[int]] = None
        self._out_spec: Optional[pytree.TreeSpec] = None
        logger.info(
            "torch_tpu subgraph: %d empty intermediates made fresh",
            _fresh_empty_tensors(gm),
        )

    @staticmethod
    def _to_tpu(args: Sequence[Any]) -> List[Any]:
        out: List[Any] = []
        for a in args:
            if isinstance(a, torch.Tensor) and a.numel() == 0:
                out.append(torch.empty(a.shape, dtype=a.dtype, device="tpu"))
            elif isinstance(a, torch.Tensor) and a.device.type != "tpu":
                out.append(a.to("tpu"))
            else:
                out.append(a)
        return out

    def _compile_donated_static(self, tpu_args: List[Any], tag: str = "") -> None:
        # pyre-ignore[21]: torch_tpu is a TPU-runtime-only package.
        from torch_tpu._internal.compile import compiler as tpu_compiler

        for a in tpu_args:
            if isinstance(a, torch.Tensor):
                layout = tpu_compiler.get_non_default_device_layout(a)
                if layout is not None:
                    a._tpu_device_layout = layout
        os.environ.setdefault(
            "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "0"
        )
        _check_same_graph_on_all_ranks(self.gm, tag)
        donated = _find_donatable_input_indices(self.gm, tpu_args)
        self.donated_inputs = tuple(donated)
        logger.info(
            "Compiling subgraph (%s) via StaticCompiler (donated_inputs=%d)...",
            tag,
            len(donated),
        )
        self.compiled = tpu_compiler.StaticCompiler()(
            self.gm,
            list(tpu_args),
            is_fwd=True,
            donated_inputs=donated,
            module_name=tag or "recir_torchtpu_step",
        )

    def _compile_static(self, tpu_args: List[Any], tag: str) -> None:
        # pyre-ignore[21]: torch_tpu is a TPU-runtime-only package.
        from torch_tpu._internal.compile._backend import _TPU_DECOMPOSITIONS

        # pyre-ignore[21]: torch_tpu is a TPU-runtime-only package.
        from torch_tpu._internal.compile.compiler import StaticCompiler

        output = next(iter(self.gm.graph.find_nodes(op="output")))
        self._out_spec = pytree.tree_flatten(output.args[0])[1]
        fgm, self._mutated = _functional_graph(self.gm, tpu_args, _TPU_DECOMPOSITIONS)
        _check_same_graph_on_all_ranks(fgm, tag)
        self.compiled = StaticCompiler()(fgm, tpu_args)

    def _ensure_compiled(
        self, tpu_args: List[Any], raw_args: Optional[Sequence[Any]] = None, tag: str = ""
    ) -> None:
        if self.compiled is not None:
            return
        for a in tpu_args:
            if isinstance(a, torch.Tensor):
                assert not a.is_meta, (
                    "torch_tpu backend requires real (non-fake) tensors at "
                    "compile time; got a meta/fake tensor."
                )
        check_args = list(raw_args) if raw_args is not None else list(tpu_args)
        mismatches = _shape_mismatches(self.gm, check_args)
        if mismatches:
            raise ValueError(
                f"torch_tpu subgraph ({tag}) inputs differ from their traced "
                f"shapes:\n" + "\n".join(mismatches)
            )
        if _uses_donated_static_compiler():
            self._compile_donated_static(tpu_args, tag=tag)
        elif _uses_static_compiler():
            self._compile_static(tpu_args, tag=tag)
        else:
            # pyre-ignore[21]: torch_tpu is a TPU-runtime-only package.
            from torch_tpu._internal.compile._backend import TpuBackend

            logger.info(
                "spmd_safe: marked %d collectives in subgraph (%s)",
                _mark_collectives_spmd_safe(self.gm),
                tag,
            )
            _check_same_graph_on_all_ranks(self.gm, tag)
            self.compiled = TpuBackend()(self.gm, list(tpu_args))
        logger.info("Completed torch_tpu compilation for subgraph (%s)", tag)

    def run_tpu(self, tpu_args: List[Any], tag: str = "") -> Any:
        if self.compiled is None:
            self._ensure_compiled(tpu_args, tag=tag)
        assert self.compiled is not None
        return self.compiled(*tpu_args)

    def forward(self, *args: Any, tag: str = "") -> Any:
        tpu_args = self._to_tpu(args)
        if self.compiled is None:
            self._ensure_compiled(tpu_args, raw_args=args, tag=tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            outputs = self.compiled(*tpu_args)
            if self._mutated is None:
                return outputs
            num_outputs = len(outputs) - len(self._mutated)
            for index, value in zip(self._mutated, outputs[num_outputs:]):
                args[index].copy_(value)
            assert self._out_spec is not None
            return pytree.tree_unflatten(list(outputs[:num_outputs]), self._out_spec)


def partial_lower_torchtpu(
    gm: torch.fx.GraphModule,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """Segment ``gm`` and lower each compatible subgraph."""
    gm = strip_profiler_nodes(gm)
    registered = _register_tpu_fbgemm_kernels()
    if registered:
        logger.info("TorchTPU kernels for fbgemm ops: %s", registered)
    if _uses_donated_static_compiler():
        _materialize_mutated_placeholder_outputs(gm)
    return segment_and_lower(
        gm,
        compatibility_fn=_is_torchtpu_compatible,
        module_factory=_TorchTPUModule,
        subgraph_prefix="torchtpu_subgraph",
        node_predicate=node_predicate,
        subgraph_predicate=subgraph_predicate,
        dumper=dumper,
    )
