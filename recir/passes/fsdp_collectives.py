# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Generic FSDP shard_map collective optimization passes.

These stage-6 graph passes rewrite the flat FX graph emitted by a
``DataParallel("fully_shard")`` + :class:`~recir.backends.XLAShardMapBackend`
job (make_fx + AOTAutograd, so activation checkpointing is already inlined and
the ``_c10d_functional`` collectives are explicit nodes):

- :func:`merge_backward_gathers` -- drop each weight's redundant 3rd backward
  all-gather (3/weight -> 2/weight).
- :func:`bucket_grad_reduce_scatters` -- fuse consecutive per-weight grad
  reduce_scatters into fewer, larger collectives.
- :func:`serialize_backward_gathers` -- chain backward full-weight gathers
  through an optimization barrier so at most ``window`` weights co-reside.

They share :func:`reshard_fsdp_seq <torch.ops.recir_fsdp.reshard_fsdp_seq>`, a
sequencing optimization-barrier op that lowers to
``jax.lax.optimization_barrier`` via torchax.
"""

from __future__ import annotations

import logging
import operator
from collections import defaultdict, deque
from typing import Any, Callable, Optional

import jax
import torch
from recir.distributed.sharding_spec import get_state_layout
from recir.torchax.ops.decorators import op


logger: logging.Logger = logging.getLogger(__name__)


# --- FSDP sequencing optimization-barrier op --------------------------------
#
# shard_map bakes EXPLICIT all-gathers into the graph; each FSDP weight is
# gathered up to 3x (forward, backward-recompute, backward-dx). During the
# backward, XLA schedules all layers' re-gathers in PARALLEL (they depend only
# on the saved /N shard, not on the reverse-flowing activation gradient), so
# ~all layers' full weights co-reside -> a large batch-independent HBM base.
#
# reshard_fsdp_seq(x, dep) lowers to jax.lax.optimization_barrier((x, dep))[0]:
# it returns x but makes x depend on `dep`, letting serialize_backward_gathers
# chain a backward gather's small /N shard on a PRIOR gather's consuming-matmul
# output. optimization_barrier is the same primitive jax.remat uses to break
# XLA's CSE, so the fwd/bwd gathers stay distinct values and the re-gathers
# serialize in reverse-layer order -> bounded co-resident weight peak.
_RESHARD_LIB_DEF = torch.library.Library("recir_fsdp", "DEF")
_RESHARD_LIB_DEF.define("reshard_fsdp_seq(Tensor x, Tensor dep) -> Tensor")
_RESHARD_LIB_IMPL = torch.library.Library("recir_fsdp", "IMPL")
_RESHARD_LIB_IMPL.impl(
    "reshard_fsdp_seq", lambda x, dep: x, "CompositeExplicitAutograd"
)
# Explicit Meta kernel so the op is XLA-compatible (lowering_utils.has_meta_kernel).
_RESHARD_LIB_IMPL.impl("reshard_fsdp_seq", lambda x, dep: x, "Meta")
# Side-effectful so DCE/CSE keeps each per-consumer barrier distinct.
torch.fx.node._side_effectful_functions.add(
    torch.ops.recir_fsdp.reshard_fsdp_seq.default
)


@op(torch.ops.recir_fsdp.reshard_fsdp_seq.default)  # pyre-ignore[56]
def _reshard_fsdp_seq_lowering(x: Any, dep: Any) -> Any:
    # optimization_barrier is the primitive jax.remat uses to break XLA's CSE; the
    # tuple form makes the returned weight depend on `dep`, so the weight's backward
    # all-gather cannot start until `dep` (an activation gradient from a later
    # layer's backward) is ready -> serialized re-gathers, bounded peak HBM.
    logger.debug(
        "RESHARD_SEQ_FIRED x=%s dep=%s",
        getattr(x, "shape", "?"),
        getattr(dep, "shape", "?"),
    )
    x_out, _ = jax.lax.optimization_barrier((x, dep))
    return x_out


# Matmuls trigger the full-weight all-gather; the non-weight input is the
# activation gradient we serialize the backward re-gather on.
_MATMUL_TARGETS: frozenset[Any] = frozenset(
    {
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.addmm.default,
    }
)


# View-like ops the weight passes through (detach/transpose/reshape) before its
# backward matmul; we BFS across these to reach the gather-triggering matmul.
_VIEW_OPS: frozenset[Any] = frozenset(
    {
        torch.ops.aten.detach.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
    }
)


def _find_fsdp_placeholder(node: torch.fx.Node) -> Optional[str]:
    """BFS back through inputs to the first FSDP-sharded placeholder name.

    Used to group all-gathers by their source FSDP weight (each weight is gathered
    3x: forward, backward-recompute, backward-dx)."""
    seen: set[torch.fx.Node] = set()
    q: deque[torch.fx.Node] = deque([node])
    while q:
        n = q.popleft()
        if n in seen:
            continue
        seen.add(n)
        if n.op == "placeholder":
            # Typed-meta contract only: rewrite collectives for placeholders the
            # producer tagged FSDP-sharded (recir_state_layout), never a plain
            # is_weight placeholder (a replicated weight would be mis-rewritten).
            if get_state_layout(n) is not None:
                return n.name
            continue
        for inp in n.all_input_nodes:
            q.append(inp)
    return None


# View/reshape ops the gathered weight passes through before its matmul. Beyond
# _VIEW_OPS this MUST include the Shard(0)->Replicate redistribution ops
# (slice/narrow/select/split/getitem) that appear between wait_tensor and the
# matmul on the REAL model (e.g. wait -> slice -> view -> mm). Missing `slice`
# silently leaves every real backward gather unchained (chained=0) while a tiny
# model happens to have none.
_SERIALIZE_VIEW_THROUGH: frozenset[Any] = (
    frozenset(
        {
            torch.ops._c10d_functional.wait_tensor.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.narrow.default,
            torch.ops.aten.select.int,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.default,
            torch.ops.aten._to_copy.default,
            operator.getitem,
        }
    )
    | _VIEW_OPS
)


def _all_matmul_consumers(
    gather: torch.fx.Node, order: dict[torch.fx.Node, int]
) -> list[torch.fx.Node]:
    """Forward-BFS from a gather through wait/view ops; return ALL matmuls that
    consume the gathered full weight, sorted by topo order. Some weights feed 2
    matmuls (measured: 35/159), so serializing on the LAST use (not the first)
    is required to guarantee the full weight is dead before the next gather."""
    seen: set[torch.fx.Node] = set()
    frontier: deque[torch.fx.Node] = deque([gather])
    matmuls: list[torch.fx.Node] = []
    while frontier:
        n = frontier.popleft()
        for u in n.users:
            if u in seen:
                continue
            seen.add(u)
            if u.target in _MATMUL_TARGETS:
                matmuls.append(u)
            elif u.target in _SERIALIZE_VIEW_THROUGH:
                frontier.append(u)
    return sorted(matmuls, key=lambda m: order.get(m, 1 << 30))


def merge_backward_gathers(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Merge each FSDP weight's redundant BACKWARD all-gathers into one.

    recir's collectives_insertion (a per-access parametrization) + activation
    checkpointing + _clone_param_chains_for_backward emit **3 all-gathers/weight**:
    forward, backward-recompute, backward-dx. The two backward gathers reconstruct
    the *identical* full weight (both are all_gather of the same /N shard). The JAX
    baseline only gathers twice (dx reuses the remat-recomputed weight); the extra
    3rd gather is redundant collective volume that hurts 256-device weak-scaling.

    This rewires the later backward gathers' wait_tensor consumers to the FIRST
    backward gather's wait, then DCEs the duplicate gathers -> 3/weight -> 2/weight.
    The shared weight lives across one layer's recompute+dx (bounded; the
    serialization pass keeps cross-layer peak in check). Graph is flat at stage-6
    (make_fx already inlined the checkpoint), so this is a plain value rewrite.
    """
    graph = gm.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    ag_target = torch.ops._c10d_functional.all_gather_into_tensor.default
    wait_target = torch.ops._c10d_functional.wait_tensor.default

    def _wait_of(g: torch.fx.Node) -> Optional[torch.fx.Node]:
        return next((u for u in g.users if u.target is wait_target), None)

    by_src: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for n in graph.nodes:
        if n.op == "call_function" and n.target is ag_target:
            src = _find_fsdp_placeholder(n)
            if src is not None:
                by_src[src].append(n)
    merged = 0
    for gs in by_src.values():
        gs.sort(key=lambda n: order[n])
        bwd = gs[1:]  # skip the forward gather
        if len(bwd) < 2:
            continue
        keep_wait = _wait_of(bwd[0])
        if keep_wait is None:
            continue
        for dup in bwd[1:]:
            dup_wait = _wait_of(dup)
            # Only merge same-shape (=> identical) gathers; keep_wait precedes dup.
            if dup_wait is None or order[keep_wait] >= order[dup_wait]:
                continue
            kv = keep_wait.meta.get("val")
            dv = dup_wait.meta.get("val")
            if not (
                hasattr(kv, "shape")
                and hasattr(dv, "shape")
                and tuple(kv.shape) == tuple(dv.shape)
            ):
                continue
            dup_wait.replace_all_uses_with(keep_wait)
            # Functional collectives are impure to FX's DCE, so erase the now-dead
            # duplicate all_gather + wait explicitly (else they linger in the graph
            # and XLA may still emit the redundant collective).
            graph.erase_node(dup_wait)
            if not dup.users:
                graph.erase_node(dup)
            merged += 1
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    logger.info("MERGE_BWD_GATHERS merged=%d", merged)
    return gm


# Op targets for the grad reduce_scatter bucketing pass. IMPORTANT: these are all
# ops WITH a meta kernel (view/cat/squeeze.dim/slice.Tensor); see
# bucket_grad_reduce_scatters for why aten.reshape.default must NOT be used here.
_REDUCE_SCATTER_T: Any = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT_T: Any = torch.ops._c10d_functional.wait_tensor.default
_DETACH_T: Any = torch.ops.aten.detach.default
_CAT_T: Any = torch.ops.aten.cat.default
_VIEW_T: Any = torch.ops.aten.view.default
_SQUEEZE_DIM_T: Any = torch.ops.aten.squeeze.dim
_SLICE_T: Any = torch.ops.aten.slice.Tensor

# One bucket member: (reduce_scatter, wait, grad_input, cols, out_shape).
_BucketMember = tuple[torch.fx.Node, torch.fx.Node, Any, int, list[int]]


def _group_consecutive(
    items: list[Any], key_fn: Callable[[Any], Any]
) -> list[list[Any]]:
    """Split ``items`` into maximal consecutive runs sharing an equal key."""
    groups: list[list[Any]] = []
    for it in items:
        if groups and key_fn(groups[-1][0]) == key_fn(it):
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _grad_rows(n: torch.fx.Node) -> Optional[int]:
    """Leading (row) dim of a reduce_scatter's grad input, or None."""
    v = n.args[0].meta.get("val") if isinstance(n.args[0], torch.fx.Node) else None
    return int(v.shape[0]) if hasattr(v, "shape") and len(v.shape) >= 1 else None


def _grad_dtype(n: torch.fx.Node) -> Any:
    """Dtype of a reduce_scatter's grad input, or None."""
    v = n.args[0].meta.get("val") if isinstance(n.args[0], torch.fx.Node) else None
    return getattr(v, "dtype", None)


def _rs_bucket_key(n: torch.fx.Node) -> tuple[Any, Any, Any, Any, Any]:
    # Only fuse reduce_scatters that share a leading dim, dtype, AND collective
    # params (reduceOp, group_size, group_name). dtype must be part of the key
    # because _emit_bucket torch.cat's the grads together and torch.cat rejects
    # mixed dtypes (e.g. bf16 + fp32 params interleaved under mixed precision).
    return (_grad_rows(n), _grad_dtype(n), n.args[1], n.args[2], n.args[3])


def _rs_wait(rs: torch.fx.Node) -> Optional[torch.fx.Node]:
    return next((u for u in rs.users if u.target is _WAIT_T), None)


def _grad_leaf(wait: torch.fx.Node) -> torch.fx.Node:
    """Follow the wait -> detach chain to the grad leaf feeding the SGD update."""
    n = wait
    while len(n.users) == 1:
        u = next(iter(n.users))
        if u.target is _DETACH_T:
            n = u
        else:
            break
    return n


def _collect_bucket_members(
    grp: list[torch.fx.Node], rows: Optional[int]
) -> list[_BucketMember]:
    """Fuse-able (reduce_scatter, wait, grad_in, cols, out_shape) for a group."""
    members: list[_BucketMember] = []
    for rs in grp:
        w = _rs_wait(rs)
        gin = rs.args[0]
        gv = gin.meta.get("val") if isinstance(gin, torch.fx.Node) else None
        ov = rs.meta.get("val")
        if w is None or gv is None or ov is None or rows is None or rows == 0:
            continue
        # slice-direct/squeeze reconstruction only handles 1D/2D grads; weight
        # grads are always <=2D (linears 2D, norms 1D).
        if getattr(gv, "ndim", 0) > 2 or len(ov.shape) > 2:
            continue
        members.append((rs, w, gin, int(gv.numel() // rows), list(ov.shape)))
    return members


def _bucket_feasible(
    leaves: list[tuple[_BucketMember, torch.fx.Node]],
    order: dict[torch.fx.Node, int],
    anchor_ord: int,
) -> bool:
    """Every grad leaf's consumers must be AFTER the bucket point, and the leaf
    shape must equal the /N grad shape (so the rewire is valid)."""
    for (_rs, _w, _g, _c, out_shape), leaf in leaves:
        lv = leaf.meta.get("val")
        if any(order.get(u, 1 << 30) <= anchor_ord for u in leaf.users):
            return False
        if not (hasattr(lv, "shape") and list(lv.shape) == out_shape):
            return False
    return True


def _emit_bucket(
    graph: torch.fx.Graph,
    anchor: torch.fx.Node,
    members: list[_BucketMember],
    leaves: list[tuple[_BucketMember, torch.fx.Node]],
    rows: int,
    reduceop: Any,
    gsize: Any,
    gname: Any,
) -> None:
    """View each grad to [rows, cols], cat on dim 1, ONE reduce_scatter, slice the
    /N result back per weight, and rewire each grad leaf to its slice."""
    fmode = members[0][2].meta["val"].fake_mode
    total_cols = sum(c for (_r, _w, _g, c, _o) in members)
    with graph.inserting_before(anchor):
        flats = []
        for _rs, _w, gin, cols, _o in members:
            # gin is a contiguous grad ([rows] or [rows, cols]); view to
            # [rows, cols] so all bucket members are 2D for the cat.
            fn = graph.call_function(_VIEW_T, (gin, [rows, cols]))
            with fmode:
                fn.meta["val"] = gin.meta["val"].reshape([rows, cols])
            flats.append(fn)
        concat = graph.call_function(_CAT_T, (flats, 1))
        with fmode:
            concat.meta["val"] = torch.cat([f.meta["val"] for f in flats], dim=1)
        fused_rs = graph.call_function(
            _REDUCE_SCATTER_T, (concat, reduceop, gsize, gname)
        )
        fused_wait = graph.call_function(_WAIT_T, (fused_rs,))
        with fmode:
            rsv = concat.meta["val"].new_empty([rows // gsize, total_cols])
        fused_rs.meta["val"] = rsv
        fused_wait.meta["val"] = rsv
        off = 0
        for (_rs, _w, _g, cols, out_shape), (_m, leaf) in zip(members, leaves):
            col = graph.call_function(_SLICE_T, (fused_wait, 1, off, off + cols))
            with fmode:
                col.meta["val"] = fused_wait.meta["val"][:, off : off + cols]
            # 2D grad: the [rows/N, cols] slice already IS the /N grad. 1D (norm)
            # grad: squeeze the trailing size-1 dim (a valid view regardless of
            # the slice's non-contiguity).
            if len(out_shape) == 1:
                shaped = graph.call_function(_SQUEEZE_DIM_T, (col, 1))
                with fmode:
                    shaped.meta["val"] = col.meta["val"].squeeze(1)
            else:
                shaped = col
            # Inserted nodes inherit the leaf's non-shape meta (is_batch_input,
            # is_fsdp_sharded, ...) so downstream tagging + subgraph sharding stay
            # consistent.
            for _mk, _mv in leaf.meta.items():
                if _mk != "val":
                    shaped.meta.setdefault(_mk, _mv)
            leaf.replace_all_uses_with(shaped)
            off += cols


def _try_bucket_group(
    graph: torch.fx.Graph,
    grp: list[torch.fx.Node],
    order: dict[torch.fx.Node, int],
) -> bool:
    """Fuse one consecutive group's grad reduce_scatters; return True if fused."""
    if len(grp) < 2:
        return False
    rows = _grad_rows(grp[0])
    reduceop, gsize, gname = grp[0].args[1], grp[0].args[2], grp[0].args[3]
    members = _collect_bucket_members(grp, rows)
    if len(members) < 2:
        return False
    anchor = grp[-1]
    leaves = [(m, _grad_leaf(m[1])) for m in members]
    if not _bucket_feasible(leaves, order, order[anchor]):
        return False
    assert rows is not None  # guaranteed: rows None -> members filtered to < 2
    _emit_bucket(graph, anchor, members, leaves, rows, reduceop, gsize, gname)
    return True


def _erase_dead_reduce_scatters(
    graph: torch.fx.Graph, groups: list[list[torch.fx.Node]]
) -> None:
    """Erase now-dead per-weight reduce_scatter + wait nodes. Functional
    collectives are impure to FX's DCE, so erase them explicitly."""
    for grp in groups:
        for rs in grp:
            w = _rs_wait(rs)
            if w is not None and not w.users:
                graph.erase_node(w)
            if not rs.users:
                graph.erase_node(rs)


def bucket_grad_reduce_scatters(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Bucket per-weight grad reduce_scatters into fewer, larger collectives.

    recir emits one reduce_scatter per weight-grad (~530 for Gemma-4-31B); at 256
    devices these small collectives (and their derived inter-node all-reduces)
    dominate the weak-scaling gap. This fuses CONSECUTIVE reduce_scatters that
    share a leading/row dim + collective params: view each grad to [rows, cols],
    concat on dim 1, ONE reduce_scatter over dim 0, slice the /N result back per
    weight. Bit-exact per weight (reduce_scatter is linear). Rewires each grad
    LEAF (end of the wait->detach chain feeding the SGD _foreach_add) so the
    bucketed result at the layer's end is consumed topologically-after.

    @1536/128ch (6/dev, +serialize+AC+einsum): 761.72 -> 790.97 TFLOPS/chip
    (+3.8%), fusing 530 reduce-scatters -> 240.

    IMPORTANT: emit only ops WITH meta kernels (view/squeeze/slice). aten.reshape
    .default has none -> _is_xla_compatible rejects it -> segment_and_lower splits
    the joint graph and batched activations crossing the new boundary become
    replicated placeholders -> shard_map (4,128,1) vs (1,128,256) broadcast error.
    """
    graph = gm.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    rs_all = sorted(
        (
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target is _REDUCE_SCATTER_T
        ),
        key=lambda n: order[n],
    )
    groups = _group_consecutive(rs_all, _rs_bucket_key)
    bucketed = 0
    for grp in groups:
        if _try_bucket_group(graph, grp, order):
            bucketed += 1

    graph.eliminate_dead_code()
    _erase_dead_reduce_scatters(graph, groups)
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    logger.info(
        "BUCKET_GRAD_RS groups=%d buckets_fused=%d rs_before=%d",
        len(groups),
        bucketed,
        len(rs_all),
    )
    return gm


def serialize_backward_gathers(
    gm: torch.fx.GraphModule,
    *,
    window: int,
    min_gib: float,
) -> torch.fx.GraphModule:
    """Serialize backward full-weight gathers to bound co-resident peak.

    shard_map bakes EXPLICIT all-gathers into the graph; each FSDP weight is
    gathered 3x (forward, backward-recompute, backward-dx). During the backward,
    XLA schedules all layers' re-gathers in PARALLEL (they only depend on the saved
    layer input, not on the reverse-flowing activation gradient), so ~all layers'
    full weights co-reside -> the batch-independent base that caps shard_map. The
    JAX baseline avoids this structurally: jax.checkpoint + jax.grad give a SERIAL
    backward where each layer's weight all-gather is born, used, and freed inside
    that layer's remat before the next -> ~1 layer's weight live.

    We reproduce that serial lifetime by chaining each backward gather's /N SHARD
    input (small) on a PREVIOUS backward gather's consuming-matmul OUTPUT (an
    activation gradient that is alive anyway) via ``reshard_fsdp_seq`` ->
    optimization_barrier((shard, dep))[0], which pins only the small shard + the
    activation, NOT the full gathered weight. Because gather[i] waits for
    gather[i-window]'s matmul to consume its weight, gather[i-window]'s full weight
    is freed before gather[i]'s is materialized -> at most ~``window`` weights live.

    Args:
        window: allow ``window`` backward full-weight gathers in flight before
            forcing a wait (must be >= 1). ``1`` = strict serial (~1 weight live,
            lowest peak, least overlap); larger = looser (~``window`` weights
            live, more compute/collective overlap, higher peak). Tune to the
            largest that still fits the target per-device batch.
        min_gib: only serialize gathers whose FULL weight is >= this many GiB
            (``0`` = all). The co-held base is ~all model weights; freeing just
            the largest weights drops the peak with far fewer barriers than
            serializing every gather.
    """
    # Clamp: window < 1 would make anchors[j - window] index from the tail and
    # chain gathers in the wrong order (old code: max(1, int(os.environ.get(
    # "SM_SERIALIZE_WINDOW", "1")))).
    window = max(1, int(window))

    graph = gm.graph
    order = {n: i for i, n in enumerate(graph.nodes)}
    ag_target = torch.ops._c10d_functional.all_gather_into_tensor.default
    all_gathers = [
        n for n in graph.nodes if n.op == "call_function" and n.target is ag_target
    ]
    # Group gathers by their FSDP source weight; each weight is gathered 3x (fwd,
    # backward-recompute, backward-dx). The FIRST (lowest order) gather per weight
    # is the forward gather (leave it — freed after the forward); the rest are
    # backward gathers to serialize. Per-weight grouping is more robust than a
    # "after the first reduce_scatter" split.
    by_src: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for g in all_gathers:
        src = _find_fsdp_placeholder(g)
        if src is not None:
            by_src[src].append(g)
    bwd_set: set[torch.fx.Node] = set()
    for gs in by_src.values():
        gs.sort(key=lambda n: order[n])
        bwd_set.update(gs[1:])  # skip the forward gather
    bwd = sorted(bwd_set, key=lambda n: order[n])

    min_bytes = int(min_gib * (1024**3))
    # anchors[j] = LAST matmul consumer (max topo order) of the j-th serialize-able
    # backward gather. Gather j waits on anchors[j-window] so exactly window weights
    # overlap.
    anchors: list[torch.fx.Node] = []
    chained = skipped = too_small = 0
    for g in bwd:
        mms = _all_matmul_consumers(g, order)
        if not mms:
            skipped += 1  # norm-weight gather (feeds rms_norm, tiny) -> ignore
            continue
        if min_bytes:
            val = g.meta.get("val")
            gbytes = (
                val.numel() * val.element_size()
                if hasattr(val, "numel") and hasattr(val, "element_size")
                else 1 << 62
            )
            if gbytes < min_bytes:
                too_small += 1  # leave small weights free to overlap
                continue
        j = len(anchors)
        if j >= window:
            dep = anchors[j - window]
            # Topology guard: barrier goes before g, so dep must precede g.
            if order.get(dep, 1 << 30) < order[g]:
                shard = g.args[0]
                with graph.inserting_before(g):
                    barrier = graph.call_function(
                        torch.ops.recir_fsdp.reshard_fsdp_seq.default,
                        args=(shard, dep),
                    )
                # reshard_fsdp_seq is a passthrough barrier on its first arg; carry
                # the shard's fake val so downstream passes/lowering can read it.
                barrier.meta["val"] = (
                    shard.meta.get("val") if isinstance(shard, torch.fx.Node) else None
                )
                g.replace_input_with(shard, barrier)  # pyre-ignore[6]
                chained += 1
        anchors.append(mms[-1])  # last (topo-max) matmul use of this weight
    graph.lint()
    gm.recompile()
    logger.info(
        "SERIALIZE_BWD_GATHERS backward=%d serializable=%d chained=%d "
        "window=%d min_gib=%s skipped_norm=%d too_small=%d",
        len(bwd),
        len(anchors),
        chained,
        window,
        min_gib,
        skipped,
        too_small,
    )
    return gm
