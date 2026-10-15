# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""PaddingRegionTransform — input padding via custom ops at the forward boundary.

Implements :meth:`RegionTransform.transform_inputs` so that, when the
outer ``_local_compile`` ``make_fx`` enters ``model(*args, **kwargs)``,
the positional args are routed through ``recir_padding::pad_to_static``,
``pad_jagged_lengths``, ``pad_jagged_values``, and ``pad_count_to_sum``.
Each call dispatches under the active tracer and is captured as a
``call_function`` node in the final compiled graph — these are graph
breaks: they run eagerly on CPU while XLA compiles the static-shaped
subgraphs between them.

Handles both batch-divisibility padding and jagged-tensor static-shape
padding.The user specifies budgets (per-key targets for jagged tensors),
a batch-size reference tensor, and optional ``count_arg`` /
``batch_size_arg`` for each jagged pair.

The custom ops have FakeTensor kernels that return the target static
shape, so ``make_fx`` traces them naturally (no opacity mechanism
needed).  At runtime each op pads dynamically.  Backward uses paired
``recir_padding::unpad_grad`` / ``unpad_jagged_grad`` custom ops, also
graph breaks.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from recir.region_transforms.base import RegionTransform

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global registry: forward stores original_size, backward reads it.
# make_fx linearizes the autograd graph into the FX graph, so by runtime
# the forward and backward ops are independent call_function nodes with
# no autograd ctx linking them.  A side channel keyed by pad_id is the only
# way to pass original_size from forward to backward.
#
# pad_ids are allocated per PaddingRegionTransform instance via
# ``_INSTANCE_ID_COUNTER`` and offset by ``_PAD_ID_STRIDE`` so that
# multiple instances in the same train step never collide on dict keys.
# ---------------------------------------------------------------------------

_PAD_ORIGINAL_SIZES: Dict[int, int] = {}
_PAD_JAGGED_SIZES: Dict[int, Tuple[int, List[int], int]] = {}

_INSTANCE_ID_COUNTER: itertools.count[int] = itertools.count()
_PAD_ID_STRIDE: int = 1_000_000


# ---------------------------------------------------------------------------
# Public helper for compile-time consumers (override builders, override-batch
# resizers) that need the same post-pad sizes that ``_discover`` will pick at
# trace time.  Centralizing the formula keeps those call sites from drifting.
# ---------------------------------------------------------------------------


def compute_padded_shape(
    size: int,
    num_devices: int,
    *,
    must_be_strictly_greater: bool = True,
) -> int:
    """Round *size* up to a multiple of *num_devices*.

    When ``must_be_strictly_greater`` is True (default), bumps the result by
    ``num_devices`` if *size* is already aligned, guaranteeing at least one
    slot of headroom.  ``pad_jagged_lengths`` and ``pad_count_to_sum`` need
    at least one dummy slot to absorb budget deficits, so the jagged path
    always requires strict-greater.

    Callers that need an additional divisibility constraint (e.g. SparseCore
    CSR sharding requires per-key budget to be a multiple of
    ``local_dev × num_sc`` per host = total-SC count globally) should LCM
    the SC constraint into ``num_devices`` themselves before calling, e.g.::

        effective = math.lcm(num_ranks, sc_modulus * num_processes)
        compute_padded_shape(size, effective)
    """
    padded = int(math.ceil(size / num_devices)) * num_devices
    if padded == size and must_be_strictly_greater:
        padded += num_devices
    return padded


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


def runtime_num_local() -> int:
    """Local devices per host = the per-host -> per-device ``np.split`` degree."""
    try:
        import jax

        return max(1, int(jax.local_device_count()))
    except Exception:
        return 1


def trim_per_chip_to_size(
    x: torch.Tensor,
    target_size: int,
    num_local: int,
    dim: int = 0,
) -> torch.Tensor:
    """Remove trailing padding independently from each local-device block."""
    current = x.shape[dim]
    assert target_size <= current
    if current == target_size:
        return x
    if current % num_local != 0 or target_size % num_local != 0:
        return x.narrow(dim, 0, target_size)
    current_per_chip = current // num_local
    target_per_chip = target_size // num_local
    return (
        x.unflatten(dim, (num_local, current_per_chip))
        .narrow(dim + 1, 0, target_per_chip)
        .flatten(dim, dim + 1)
    )


@torch.library.custom_op("recir_padding::pad_to_static", mutates_args=())
def pad_to_static(
    x: torch.Tensor,
    target_size: int,
    dim: int = 0,
    pad_id: int = -1,
    pad_value: float = 0.0,
    num_processes: int = 1,
) -> torch.Tensor:
    """Pad *x* along *dim* to ``target_size // num_processes``.

    ``target_size`` is the **global** padded size as seen by FX trace
    (matching what ``_to_sharded_jax`` will produce by expanding the
    per-host runtime shape by ``num_global // num_local`` chips).  At
    runtime each host only holds its own data shard, so the actual pad
    target is ``target_size // num_processes``.  The fake kernel returns
    the global shape so the FX trace and JAX retrace agree.
    """
    per_host_target = target_size // num_processes
    if pad_id >= 0:
        _PAD_ORIGINAL_SIZES[pad_id] = x.shape[dim]
    current = x.shape[dim]
    if current == per_host_target:
        return x.clone()
    num_local = runtime_num_local()
    assert current % num_local == 0 and per_host_target % num_local == 0, (
        f"pad_to_static: current={current} / target={per_host_target} not "
        f"divisible by num_local={num_local}"
    )
    cur_pb = current // num_local
    target_pb = per_host_target // num_local
    x2 = x.unflatten(dim, (num_local, cur_pb))
    if current > per_host_target:
        return x2.narrow(dim + 1, 0, target_pb).flatten(dim, dim + 1).clone()
    # Pad PER-CHIP so the per-host -> per-device np.split(num_local) lands each
    # device a uniform block (real rows + its own dummy rows). num_local == 1
    # degenerates to a single tail-padded block.
    pw = [0] * (2 * x2.ndim)
    pad_axis = dim + 1
    pw[2 * (x2.ndim - 1 - pad_axis) + 1] = target_pb - cur_pb
    x2 = F.pad(x2, pw, value=pad_value)
    return x2.flatten(dim, dim + 1)


@pad_to_static.register_fake  # pyre-ignore[56]
def _pad_fake(
    x: torch.Tensor,
    target_size: int,
    dim: int = 0,
    pad_id: int = -1,
    pad_value: float = 0.0,
    num_processes: int = 1,
) -> torch.Tensor:
    # Fake kernel uses the GLOBAL target so the FX trace shape matches
    # what JAX sees after _to_sharded_jax expansion.
    if pad_id >= 0:
        _PAD_ORIGINAL_SIZES[pad_id] = x.shape[dim]
    shape = list(x.shape)
    shape[dim] = target_size
    return torch.empty(shape, dtype=x.dtype, device=x.device)


@torch.library.custom_op("recir_padding::pad_count_to_sum", mutates_args=())
def pad_count_to_sum(
    x: torch.Tensor,
    target_size: int,
    target_sum: int,
    pad_id: int = -1,
    num_processes: int = 1,
) -> torch.Tensor:
    """Pad a per-item count tensor (e.g. ``num_candidates``) so that:

    * ``shape[0] == target_size // num_processes``
    * ``sum() == target_sum // num_processes``

    Strategy: append ``num_dummy = per_host_target_size - current`` new
    entries and distribute the sum deficit evenly across them
    (``deficit // num_dummy`` each, with the first ``deficit %
    num_dummy`` getting one extra).  This keeps dummy entries roughly
    on the same scale as real per-sample counts, matching the dummy
    distribution used by ``pad_jagged_lengths``.
    """
    per_host_target_size = target_size // num_processes
    per_host_target_sum = target_sum // num_processes
    if pad_id >= 0:
        _PAD_ORIGINAL_SIZES[pad_id] = x.shape[0]
    current = x.shape[0]
    # Distribute the count PER-CHIP: split into num_local blocks, pad each block
    # to per_host_target_size//num_local entries summing to
    # per_host_target_sum//num_local, so each device's candidate count + sum is
    # UNIFORM (the candidate-axis even-split then coincides with the sample axis).
    # num_local == 1 degenerates to the single tail-append block.
    num_local = runtime_num_local()
    assert (
        current % num_local == 0
        and per_host_target_size % num_local == 0
        and per_host_target_sum % num_local == 0
    ), (
        f"pad_count_to_sum: current={current} / target_size={per_host_target_size}"
        f" / target_sum={per_host_target_sum} not divisible by num_local={num_local}"
    )
    cur_pb = current // num_local
    target_pb = per_host_target_size // num_local
    sum_pb = per_host_target_sum // num_local
    x_blocks = x.reshape(num_local, cur_pb)
    num_dummy = target_pb - cur_pb
    parts: List[torch.Tensor] = []
    for blk in range(num_local):
        xb = x_blocks[blk]
        if num_dummy <= 0:
            xb = xb.narrow(0, 0, target_pb).clone()
            d = sum_pb - int(xb.sum().item())
            if d != 0:
                xb[-1] = xb[-1] + d
            parts.append(xb)
            continue
        deficit = sum_pb - int(xb.sum().item())
        per_d = deficit // num_dummy
        rem = deficit - per_d * num_dummy
        dummy = torch.full((num_dummy,), per_d, dtype=x.dtype, device=x.device)
        if rem > 0:
            dummy[:rem] = dummy[:rem] + 1
        parts.append(torch.cat([xb, dummy], dim=0))
    return torch.cat(parts, dim=0)


@pad_count_to_sum.register_fake  # pyre-ignore[56]
def _pad_count_to_sum_fake(
    x: torch.Tensor,
    target_size: int,
    target_sum: int,
    pad_id: int = -1,
    num_processes: int = 1,
) -> torch.Tensor:
    if pad_id >= 0:
        _PAD_ORIGINAL_SIZES[pad_id] = x.shape[0]
    shape = list(x.shape)
    shape[0] = target_size
    return torch.empty(shape, dtype=x.dtype, device=x.device)


@torch.library.custom_op("recir_padding::unpad_grad", mutates_args=())
def unpad_grad(
    grad: torch.Tensor,
    pad_id: int,
    dim: int = 0,
    num_processes: int = 1,
) -> torch.Tensor:
    """Restore *grad* to the original size stored by ``pad_to_static``.

    The original size in the registry is whatever was seen in the
    matching forward call: per-host original at runtime, per-process
    original at fake trace.  Both are correct in their own context.
    ``num_processes`` is unused at runtime here (kept for symmetry with
    the forward op).
    """
    original_size = _PAD_ORIGINAL_SIZES[pad_id]
    current = grad.shape[dim]
    if current == original_size:
        return grad.clone()
    # Reverse the PER-CHIP pad_to_static: per block, take the first
    # original//num_local (real) of each current//num_local slot (or zero-pad if
    # the forward narrowed). num_local == 1 degenerates to the plain narrow/pad.
    num_local = runtime_num_local()
    cur_pb = original_size // num_local
    target_pb = current // num_local
    g2 = grad.unflatten(dim, (num_local, target_pb))
    if target_pb >= cur_pb:
        g2 = g2.narrow(dim + 1, 0, cur_pb)
    else:
        pw = [0] * (2 * g2.ndim)
        pw[2 * (g2.ndim - 1 - (dim + 1)) + 1] = cur_pb - target_pb
        g2 = F.pad(g2, pw)
    return g2.flatten(dim, dim + 1).clone()


@unpad_grad.register_fake  # pyre-ignore[56]
def _unpad_fake(
    grad: torch.Tensor,
    pad_id: int,
    dim: int = 0,
    num_processes: int = 1,
) -> torch.Tensor:
    shape = list(grad.shape)
    shape[dim] = _PAD_ORIGINAL_SIZES[pad_id]
    return grad.new_empty(shape)


def _trim_lengths(lengths: torch.Tensor, budget: int) -> torch.Tensor:
    """Trim *lengths* so they sum to at most *budget*, preserving order."""
    capped = lengths.cumsum(0).clamp(max=budget)
    return torch.diff(
        capped, prepend=torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    )


def _per_chip_real_counts(
    batch: int, num_local: int, count: Optional[torch.Tensor]
) -> List[int]:
    """Real element count owned by each of the ``num_local`` chip blocks along a
    jagged pair's batch (row) axis.

    ``count is None`` — the batch axis is already chip-major uniform (e.g. the RO
    sample axis), so each chip owns ``batch // num_local`` rows.

    ``count is not None`` — the batch axis is a *candidate* axis whose rows follow
    their parent samples, so it is NOT uniform across chips. ``count`` is the
    per-sample count (sample-granularity, divisible by ``num_local``); chip ``c``
    owns ``sum(count[c's samples])`` candidates. This mirrors how
    ``pad_count_to_sum`` distributes the count itself, so every NRO tensor's
    per-chip candidate boundary is identical.
    """
    if count is None:
        assert batch % num_local == 0, (
            f"_per_chip_real_counts: batch={batch} not divisible by "
            f"num_local={num_local} (no count tensor to define per-chip split)"
        )
        pb = batch // num_local
        return [pb] * num_local
    num_samples = count.shape[0]
    assert num_samples % num_local == 0, (
        f"_per_chip_real_counts: num_samples={num_samples} not divisible by "
        f"num_local={num_local}"
    )
    spb = num_samples // num_local
    return [int(x) for x in count.reshape(num_local, spb).sum(1).tolist()]


@torch.library.custom_op("recir_padding::pad_jagged_lengths", mutates_args=())
def pad_jagged_lengths(
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    target_batch: int,
    num_processes: int = 1,
    count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pad a jagged lengths tensor with dummy entries so that each key's
    total equals its budget and the output has
    ``target_batch * len(per_key_budgets)`` entries.

    ``per_key_budgets`` and ``target_batch`` are GLOBAL sizes (matching
    the FX trace shape).  At runtime each host applies ``// num_processes``
    so its local lengths tensor is padded only to its per-host shard size.

    ``count`` is the per-sample count (e.g. ``num_candidates``) when the batch
    (row) axis is a *candidate* axis: chip ``c`` then owns ``sum(count[c's
    samples])`` real rows rather than a uniform ``B_real // num_local`` (the real
    candidates per chip vary with the jagged data).  ``count is None`` keeps the
    uniform per-chip split used by sample-axis pairs.
    """
    per_host_budgets = [b // num_processes for b in per_key_budgets]
    per_host_target_batch = target_batch // num_processes
    num_keys = len(per_host_budgets)
    B_real = lengths.shape[0] // num_keys

    # Lay out each key PER-CHIP as [chip0: real + dummy | chip1: real + dummy | ...]
    # so the per-host -> per-device np.split(num_local) gives each device its own
    # per-chip block (real rows + its own dummy slots, deficit absorbed per
    # block). num_local == 1 degenerates to a single [real | dummy] block. When
    # count is set the real-row count per chip is data-dependent (rc_c below),
    # matching pad_count_to_sum's candidate distribution.
    num_local = runtime_num_local()
    assert per_host_target_batch % num_local == 0, (
        f"pad_jagged_lengths: per_host_target_batch={per_host_target_batch} "
        f"not divisible by num_local={num_local}"
    )
    tpb = per_host_target_batch // num_local
    rc = _per_chip_real_counts(B_real, num_local, count)

    key_lengths = lengths.reshape(num_keys, B_real)
    parts: List[torch.Tensor] = []
    for k in range(num_keys):
        budget_pb = per_host_budgets[k] // num_local
        offset = 0
        blocks: List[torch.Tensor] = []
        for c in range(num_local):
            rc_c = rc[c]
            bl = key_lengths[k][offset : offset + rc_c]
            offset += rc_c
            num_dummy_c = tpb - rc_c
            block_total = int(bl.sum().item())
            if block_total > budget_pb:
                bl = _trim_lengths(bl, budget_pb)
                block_total = budget_pb
            block_pad = budget_pb - block_total
            if block_pad > 0 and num_dummy_c > 0:
                per_d = block_pad // num_dummy_c
                rem = block_pad % num_dummy_c
                dummy_b = torch.full(
                    (num_dummy_c,), per_d, dtype=lengths.dtype, device=lengths.device
                )
                if rem > 0:
                    dummy_b[:rem] = dummy_b[:rem] + 1
            else:
                dummy_b = torch.zeros(
                    num_dummy_c, dtype=lengths.dtype, device=lengths.device
                )
            blocks.append(torch.cat([bl, dummy_b]))
        parts.append(torch.cat(blocks))
    return torch.cat(parts)


@pad_jagged_lengths.register_fake  # pyre-ignore[56]
def _pad_lengths_fake(
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    target_batch: int,
    num_processes: int = 1,
    count: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Fake kernel uses the GLOBAL target so FX trace and JAX retrace agree.
    return torch.empty(
        target_batch * len(per_key_budgets),
        dtype=lengths.dtype,
        device=lengths.device,
    )


@torch.library.custom_op("recir_padding::pad_jagged_values", mutates_args=())
def pad_jagged_values(
    values: torch.Tensor,
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    dim: int = 0,
    pad_id: int = -1,
    num_processes: int = 1,
    count: Optional[torch.Tensor] = None,
    tile_pad: bool = False,
) -> torch.Tensor:
    """Pad jagged values per-key so each key's segment reaches its budget.

    ``per_key_budgets`` are GLOBAL budgets; runtime divides by
    ``num_processes`` to get per-host budgets.

    ``count`` (per-sample count, e.g. ``num_candidates``) is passed when the
    jagged *row* axis is a candidate axis: chip ``c`` then owns ``sum(count[c's
    samples])`` real rows (rc_c), whose values are narrowed contiguously. When a
    dense candidate-axis tensor is padded, pass ``count`` as ``lengths`` with a
    single-key budget: each row is one candidate, so the per-chip value total
    equals rc_c directly.  ``count is None`` keeps the uniform per-chip split.

    When ``tile_pad`` is set, each per-chip block's deficit is filled by tiling
    that block's own values (``vals[arange(deficit) % current]``) instead of
    zeros. This is for embedding id-lists whose pad positions are masked out
    downstream: it keeps the pad tokens' block-owner distribution uniform
    (matching the real ids) instead of collapsing every pad to id 0 ->
    block-owner 0, which otherwise skews the ring-shift all-to-all. Real values
    stay at the front of each block so the backward (which slices the real
    prefix) is unchanged.
    """
    per_host_budgets = [b // num_processes for b in per_key_budgets]
    num_keys = len(per_host_budgets)
    B = lengths.shape[0] // num_keys
    # Pad PER-CHIP to match pad_jagged_lengths: each key is laid out
    # [chip0: real values padded to budget/num_local | chip1: ...]. Blocks are
    # contiguous chunks of the real per-key values, so np.split(num_local) lands
    # each device its own block. num_local == 1 degenerates to the per-host
    # layout. rc_c is the real row count per chip (data-dependent when count set).
    num_local = runtime_num_local()
    rc = _per_chip_real_counts(B, num_local, count)
    assert lengths.shape[0] == num_keys * B, (
        f"pad_jagged_values: lengths.shape[0]={lengths.shape[0]} is not "
        f"num_keys={num_keys} * B={B}"
    )
    csum = lengths.cumsum(0).tolist()
    chip_bounds = list(itertools.accumulate(rc, initial=0))
    block_sizes: List[int] = []
    parts: List[torch.Tensor] = []
    offset = 0
    for k in range(num_keys):
        budget_pb = per_host_budgets[k] // num_local
        base = k * B
        for c in range(num_local):
            start, end = base + chip_bounds[c], base + chip_bounds[c + 1]
            block_total = int(csum[end - 1] if end > 0 else 0) - int(
                csum[start - 1] if start > 0 else 0
            )
            block_sizes.append(block_total)
            block_vals = values.narrow(dim, offset, block_total)
            offset += block_total
            current = block_vals.shape[dim]
            if current == budget_pb:
                parts.append(block_vals.clone())
            elif current > budget_pb:
                parts.append(block_vals.narrow(dim, 0, budget_pb).clone())
            elif tile_pad and current > 0:
                # Fill this block's deficit by tiling its own real ids instead of
                # zero-filling. For embedding id-lists (pad positions masked out
                # downstream) this keeps the pad tokens' block-owner distribution
                # uniform -- matching the real ids -- instead of routing every pad
                # to id 0 -> block-owner 0, which skews the ring-shift all-to-all.
                # Real ids stay at the block front so backward/fake are unchanged.
                reps = [1] * values.ndim
                reps[dim] = -(-budget_pb // current)
                parts.append(block_vals.repeat(reps).narrow(dim, 0, budget_pb))
            else:
                pw = [0] * (2 * values.ndim)
                pw[2 * (values.ndim - 1 - dim) + 1] = budget_pb - current
                parts.append(F.pad(block_vals, pw))
    if pad_id >= 0:
        _PAD_JAGGED_SIZES[pad_id] = (values.shape[dim], block_sizes, num_local)
    return torch.cat(parts, dim=dim)


@pad_jagged_values.register_fake  # pyre-ignore[56]
def _pad_jagged_values_fake(
    values: torch.Tensor,
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    dim: int = 0,
    pad_id: int = -1,
    num_processes: int = 1,
    count: Optional[torch.Tensor] = None,
    tile_pad: bool = False,
) -> torch.Tensor:
    if pad_id >= 0:
        _PAD_JAGGED_SIZES[pad_id] = (values.shape[dim], [], 1)
    shape = list(values.shape)
    shape[dim] = sum(per_key_budgets)  # global budget for FX shape
    return torch.empty(shape, dtype=values.dtype, device=values.device)


def pad_jagged_pair_to_static(
    values: torch.Tensor,
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    target_batch: int,
    *,
    dim: int = 0,
    pad_id: int = -1,
    num_processes: int = 1,
    count: Optional[torch.Tensor] = None,
    tile_pad: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a jagged values/lengths pair to shared static per-chip targets."""
    padded_values = pad_jagged_values(
        values,
        lengths,
        per_key_budgets,
        dim,
        pad_id,
        num_processes,
        count,
        tile_pad,
    )
    padded_lengths = pad_jagged_lengths(
        lengths,
        per_key_budgets,
        target_batch,
        num_processes,
        count,
    )
    return padded_values, padded_lengths


def pad_candidate_axis_to_static(
    values: torch.Tensor,
    count: torch.Tensor,
    target_size: int,
    *,
    dim: int = 0,
    pad_id: int = -1,
    num_processes: int = 1,
) -> torch.Tensor:
    """Pad one-value-per-candidate data using its per-sample count boundaries."""
    return pad_jagged_values(
        values,
        count,
        [target_size],
        dim,
        pad_id,
        num_processes,
    )


@torch.library.custom_op("recir_padding::unpad_jagged_grad", mutates_args=())
def unpad_jagged_grad(
    grad: torch.Tensor,
    pad_id: int,
    per_key_budgets: List[int],
    dim: int = 0,
    num_processes: int = 1,
) -> torch.Tensor:
    """Restore per-key gradients to original sizes.

    Per key: narrow if budget >= original, zero-pad if budget < original.
    Uses per-host budgets at runtime to match the forward.
    """
    per_host_budgets = [b // num_processes for b in per_key_budgets]
    _, block_sizes, num_local = _PAD_JAGGED_SIZES[pad_id]
    num_keys = len(per_host_budgets)
    parts: List[torch.Tensor] = []
    offset = 0
    # Mirror the per-chip forward: per key, per block, take the first (real)
    # block_size of each budget/num_local slot; concat back to [B_real per key]
    # in real order (blocks are contiguous chunks of the real values).
    for k in range(num_keys):
        budget_pb = per_host_budgets[k] // num_local
        for blk in range(num_local):
            orig = block_sizes[k * num_local + blk]
            if budget_pb >= orig:
                parts.append(grad.narrow(dim, offset, orig).clone())
            else:
                block_grad = grad.narrow(dim, offset, budget_pb).clone()
                pw = [0] * (2 * grad.ndim)
                pw[2 * (grad.ndim - 1 - dim) + 1] = orig - budget_pb
                parts.append(F.pad(block_grad, pw))
            offset += budget_pb
    return torch.cat(parts, dim=dim)


@unpad_jagged_grad.register_fake  # pyre-ignore[56]
def _unpad_jagged_grad_fake(
    grad: torch.Tensor,
    pad_id: int,
    per_key_budgets: List[int],
    dim: int = 0,
    num_processes: int = 1,
) -> torch.Tensor:
    total_original, _, _ = _PAD_JAGGED_SIZES[pad_id]
    shape = list(grad.shape)
    shape[dim] = total_original
    return torch.empty(shape, dtype=grad.dtype, device=grad.device)


# -- autograd for pad_to_static -------------------------------------------


def _pad_backward(
    ctx: Any, grad: torch.Tensor
) -> Tuple[torch.Tensor, None, None, None, None, None]:
    return (
        torch.ops.recir_padding.unpad_grad(
            grad, ctx.pad_id, ctx.dim, ctx.num_processes
        ),
        None,
        None,
        None,
        None,
        None,
    )


def _pad_setup_context(
    ctx: Any,
    inputs: Tuple[torch.Tensor, int, int, int, float, int],
    output: torch.Tensor,
) -> None:
    _, _, dim, pad_id, _, num_processes = inputs
    ctx.pad_id = pad_id
    ctx.dim = dim
    ctx.num_processes = num_processes


pad_to_static.register_autograd(_pad_backward, setup_context=_pad_setup_context)


# -- autograd for pad_jagged_values ----------------------------------------


def _pad_jagged_values_backward(
    ctx: Any, grad: torch.Tensor
) -> Tuple[torch.Tensor, None, None, None, None, None, None, None]:
    return (
        torch.ops.recir_padding.unpad_jagged_grad(
            grad, ctx.pad_id, ctx.per_key_budgets, ctx.dim, ctx.num_processes
        ),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _pad_jagged_values_setup_context(
    ctx: Any,
    inputs: Tuple[
        torch.Tensor,
        torch.Tensor,
        List[int],
        int,
        int,
        int,
        Optional[torch.Tensor],
        bool,
    ],
    output: torch.Tensor,
) -> None:
    _, _, per_key_budgets, dim, pad_id, num_processes, _count, _tile_pad = inputs
    ctx.pad_id = pad_id
    ctx.per_key_budgets = per_key_budgets
    ctx.dim = dim
    ctx.num_processes = num_processes


pad_jagged_values.register_autograd(
    _pad_jagged_values_backward,
    setup_context=_pad_jagged_values_setup_context,
)


# ---------------------------------------------------------------------------
# Post-tracing pass: mark padding ops as graph breaks
# ---------------------------------------------------------------------------

_PADDING_OP_TARGETS: set[Any] = {
    torch.ops.recir_padding.pad_to_static.default,
    torch.ops.recir_padding.pad_count_to_sum.default,
    torch.ops.recir_padding.unpad_grad.default,
    torch.ops.recir_padding.pad_jagged_lengths.default,
    torch.ops.recir_padding.pad_jagged_values.default,
    torch.ops.recir_padding.unpad_jagged_grad.default,
}


def mark_padding_graph_breaks(gm: torch.fx.GraphModule) -> None:
    """Mark all padding custom-op nodes as XLA graph breaks."""
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in _PADDING_OP_TARGETS:
            node.meta["xla_graph_break"] = True


# ---------------------------------------------------------------------------
# Auto-discovery dataclass
# ---------------------------------------------------------------------------


@dataclass
class JaggedPairConfig:
    """Configuration for one jagged (values, lengths) pair.

    Args:
        lengths: Name of the lengths tensor in the train_step signature.
        per_sample_budgets: Per-sample budget for each key.  ``int`` applies
            the same budget to every key; ``list[int]`` sets one budget per
            key (length must equal the inferred ``num_keys``).
            At runtime, the total per-key budget is
            ``per_sample_budget * padded_batch_size``.
        batch_size_arg: Name of a tensor whose ``shape[batch_dim]`` is the
            batch size for *this* pair.  Defaults to the top-level
            ``batch_size_arg`` on the transform.  Use this when the pair
            operates at a different granularity (e.g. candidate-level NRO
            vs. user-level RO).
        count_arg: Name of the per-sample count tensor (e.g. ``num_candidates``)
            that links this pair's batch to the main batch.  Required when
            ``batch_size_arg`` is set: ``PaddingRegionTransform`` uses it to
            pad the count so ``sum(count) == padded_pair_bs``.
        max_per_sample_count: Upper bound on each entry of ``count_arg``
            (e.g. the maximum number of candidates per sample).  When set,
            ``_discover`` derives ``pair_bs = main_batch_size * max_per_sample_count``
            instead of reading the (possibly jagged) trace-time tensor shape.
            Required when ``batch_size_arg`` / ``count_arg`` are set so the
            padded shape is stable across uniform/jagged trace batches and
            matches what override-builders compute via ``compute_padded_shape``.
    """

    lengths: str
    per_sample_budgets: Union[int, List[int]]
    batch_size_arg: Optional[str] = None
    count_arg: Optional[str] = None
    max_per_sample_count: Optional[int] = None


class _JaggedPair:
    __slots__ = (
        "values_idx",
        "lengths_idx",
        "per_key_budgets",
        "padded_bs",
        "bs_idx",
        "count_idx",
    )

    def __init__(
        self,
        values_idx: int,
        lengths_idx: int,
        per_key_budgets: List[int],
        padded_bs: int,
        bs_idx: Optional[int] = None,
        count_idx: Optional[int] = None,
    ) -> None:
        self.values_idx = values_idx
        self.lengths_idx = lengths_idx
        self.per_key_budgets = per_key_budgets
        self.padded_bs = padded_bs
        self.bs_idx = bs_idx
        self.count_idx = count_idx

    @property
    def total_budget(self) -> int:
        return sum(self.per_key_budgets)


class _Layout:
    """Result of one-time shape-based discovery."""

    def __init__(self) -> None:
        self.batch_size: int = 0
        self.padded_bs: int = 0
        self.batch_indices: List[int] = []
        self.jagged_pairs: List[_JaggedPair] = []
        self.inferred_pad_targets: Dict[int, int] = {}
        # Subset of ``inferred_pad_targets`` whose dim0 is a jagged *candidate*
        # axis, mapping tensor_idx to the
        # ``count_idx`` whose values define its per-sample segment lengths. These
        # must be padded count-aware (via pad_jagged_values) rather than by the
        # count-blind pad_to_static, which assumes a chip-uniform dim0.
        self.candidate_count_source_indices: Dict[int, int] = {}


# ---------------------------------------------------------------------------
# PaddingRegionTransform
# ---------------------------------------------------------------------------


class PaddingRegionTransform(RegionTransform):
    """Region transform that pads forward inputs to static shapes.

    Args:
        jagged_pairs: Mapping from values tensor name to a
            :class:`JaggedPairConfig` describing the lengths tensor,
            per-sample budgets, and optional batch-size reference.
        batch_size_arg: Name of a tensor whose
            ``shape[batch_dim]`` is the default batch size (used for
            batch-dim padding and as the default for jagged pairs
            that don't specify their own ``batch_size_arg``).
        num_devices: GLOBAL device count (= jax.device_count() for full
            GSPMD over all hosts), used as the divisor in
            ``compute_padded_shape`` to ensure the padded shape can be
            sharded evenly across all devices. Callers that also need a
            downstream divisibility constraint (e.g. SparseCore CSR shard
            width per host × num hosts) should LCM that into ``num_devices``
            at the call site.
        batch_dim: Which dimension is the batch dimension (default 0).

    Example::

        transform = PaddingRegionTransform(
            jagged_pairs={
                "ro_values": JaggedPairConfig(
                    "ro_lengths",
                    per_sample_budgets=[1, 1, 2, 1, 3554, 3554, ...],
                ),
                "nro_values": JaggedPairConfig(
                    "nro_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float",
                ),
            },
            batch_size_arg="ro_float",
            num_devices=4,
        )

        @recir.compile(region_transforms=[transform])
        def train_step(mod, opt, ro_float, nro_float, ro_values, ...):
            ...
    """

    def __init__(
        self,
        jagged_pairs: Dict[str, JaggedPairConfig],
        batch_size_arg: str,
        num_devices: int = 1,
        num_processes: int = 1,
        batch_dim: int = 0,
        *,
        skip_args: Optional[Set[str]] = None,
    ) -> None:
        self._raw_jagged_pairs: Dict[str, JaggedPairConfig] = jagged_pairs
        self._raw_batch_size_arg: str = batch_size_arg
        self._pad_id_base: int = next(_INSTANCE_ID_COUNTER) * _PAD_ID_STRIDE
        # num_devices is the GLOBAL sharding modulus the padded shape must
        # divide. For GSPMD over JAX devices this is jax.device_count().
        # Callers that need a downstream consumer (e.g. SparseCore CSR
        # sharding) to also divide the result must LCM that constraint into
        # ``num_devices`` themselves:
        #   num_devices = math.lcm(num_ranks, sc_modulus * num_processes)
        self.num_devices: int = num_devices
        # num_processes is the number of OS processes (= num_hosts) the
        # GSPMD pipeline runs across. Pad ops use it to map FX-trace global
        # target shapes to per-host runtime shapes (target // num_processes).
        # See pad_to_static for details.
        self.num_processes: int = num_processes
        self.batch_dim: int = batch_dim
        self.batch_size_arg_idx: int = 0
        # Train-step args opted out of padding here.  Useful when an Override
        # downstream handles the static-shape conversion for these args.
        self._skip_args: Set[str] = set(skip_args or set())
        self._skip_indices: Set[int] = set()  # populated by bind_signature
        self._resolved_pairs: List[
            Tuple[
                int,
                int,
                Optional[int],
                Union[int, List[int]],
                Optional[int],
                Optional[int],
            ]
        ] = []

    def bind_signature(self, func: Any) -> None:
        """Resolve the configured arg *names* into positional *indices* in ``func``.

        The transform is configured declaratively by name (``batch_size_arg``,
        ``jagged_pairs``, ``skip_args``), but at runtime ``transform_inputs``
        receives the forward inputs as a positional tuple and pads them by index.
        This runs once (pre-trace) to freeze those names into indices, skipping the
        leading non-batch args (``mod`` / ``opt`` / module / optimizer params) so
        the batch indices are not shifted.

        Example:
            For a traced step::

                def train_step(mod, opt, emb_opt, ro_float_features,
                               unary_values, unary_lengths, num_candidates): ...

            configured with ``batch_size_arg="ro_float_features"`` and
            ``jagged_pairs={"unary_values": JaggedPairConfig("unary_lengths",
            count_arg="num_candidates", ...)}``, the leading ``mod`` / ``opt`` /
            ``emb_opt`` are skipped, leaving the batch args::

                ["ro_float_features", "unary_values", "unary_lengths", "num_candidates"]
                #         0                  1                2                3

            so ``self.batch_size_arg_idx == 0`` and the pair resolves to
            ``v_idx=1, l_idx=2, count_idx=3``.
        """
        sig = inspect.signature(func)
        names: List[str] = []
        for name, param in sig.parameters.items():
            if name in ("mod", "opt", "model", "optimizer"):
                continue
            ann = param.annotation
            # Skip nn.Module / torch.optim.Optimizer params so they are not
            # mistaken for batch tensors. With `from __future__ import annotations`
            # the annotation is a STRING (e.g. a dedicated embedding optimizer
            # typed ``Optional[torch.optim.Optimizer]``), so match the resolved
            # type when available and fall back to the string form otherwise.
            if isinstance(ann, type) and issubclass(
                ann, (nn.Module, torch.optim.Optimizer)
            ):
                continue
            if isinstance(ann, str) and ("Optimizer" in ann or "nn.Module" in ann):
                continue
            names.append(name)
        self._skip_indices = {i for i, n in enumerate(names) if n in self._skip_args}
        self.batch_size_arg_idx = names.index(self._raw_batch_size_arg)
        self._resolved_pairs = []
        for v_name, config in self._raw_jagged_pairs.items():
            if v_name in self._skip_args:
                continue
            v_idx = names.index(v_name)
            l_idx = names.index(config.lengths)
            bs_idx = (
                names.index(config.batch_size_arg)
                if config.batch_size_arg is not None
                else None
            )
            count_idx = (
                names.index(config.count_arg) if config.count_arg is not None else None
            )
            if bs_idx is not None and count_idx is None:
                raise ValueError(
                    f"JaggedPairConfig for {v_name!r}: batch_size_arg="
                    f"{config.batch_size_arg!r} is set but count_arg is not. "
                    "When the jagged pair has its own batch dimension "
                    "(B_pair != B_main), the per-sample count tensor that "
                    "links them (e.g. num_candidates) must be tracked via "
                    "count_arg=... so PaddingRegionTransform can pad it with "
                    "pad_count_to_sum and keep sum(count) == padded_B_pair."
                )
            if bs_idx is not None and config.max_per_sample_count is None:
                raise ValueError(
                    f"JaggedPairConfig for {v_name!r}: batch_size_arg="
                    f"{config.batch_size_arg!r} is set but max_per_sample_count "
                    "is not. _discover needs an upper bound on the per-sample "
                    "count to derive pair_bs = main_batch_size * "
                    "max_per_sample_count; otherwise it would have to read "
                    "the jagged trace-time tensor shape and silently disagree "
                    "with override-builders that use the upper-bound formula."
                )
            self._resolved_pairs.append(
                (
                    v_idx,
                    l_idx,
                    bs_idx,
                    config.per_sample_budgets,
                    count_idx,
                    config.max_per_sample_count,
                )
            )

    # ------------------------------------------------------------------
    # Auto-discovery
    # ------------------------------------------------------------------

    def _discover(self, tensors: List[torch.Tensor]) -> _Layout:
        dim = self.batch_dim
        layout = _Layout()
        nd = self.num_devices

        batch_size = tensors[self.batch_size_arg_idx].shape[dim]
        layout.batch_size = batch_size
        padded_bs = compute_padded_shape(
            batch_size, nd, must_be_strictly_greater=bool(self._resolved_pairs)
        )
        layout.padded_bs = padded_bs

        jagged_indices: set[int] = set()
        for (
            v_idx,
            l_idx,
            bs_idx,
            per_sample,
            count_idx,
            max_count,
        ) in self._resolved_pairs:
            if bs_idx is not None:
                assert max_count is not None  # enforced in bind_signature
                pair_bs = batch_size * max_count
                pair_padded_bs = padded_bs * max_count
            else:
                pair_bs = batch_size
                pair_padded_bs = padded_bs
            # num_keys is derived from the actual lengths tensor shape at trace time
            actual_pair_bs = (
                tensors[bs_idx].shape[dim] if bs_idx is not None else batch_size
            )
            num_keys = tensors[l_idx].shape[0] // actual_pair_bs
            if num_keys == 0:
                logger.error(
                    "num_keys=0 for jagged pair (v_idx=%d, l_idx=%d): "
                    "lengths_shape=%s, pair_bs=%d, batch_size=%d, "
                    "num_user_tensors=%d, all_shapes=%s",
                    v_idx,
                    l_idx,
                    tensors[l_idx].shape,
                    pair_bs,
                    batch_size,
                    len(tensors),
                    [t.shape for t in tensors],
                )

            if isinstance(per_sample, int):
                per_key = [per_sample * pair_padded_bs] * num_keys
            else:
                assert len(per_sample) == num_keys, (
                    f"per_sample_budgets length {len(per_sample)} != "
                    f"num_keys {num_keys}"
                )
                per_key = [ps * pair_padded_bs for ps in per_sample]

            layout.jagged_pairs.append(
                _JaggedPair(v_idx, l_idx, per_key, pair_padded_bs, bs_idx, count_idx)
            )
            jagged_indices.add(v_idx)
            jagged_indices.add(l_idx)

        count_indices: set[int] = {
            jp.count_idx for jp in layout.jagged_pairs if jp.count_idx is not None
        }
        for i, t in enumerate(tensors):
            if i in self._skip_indices:
                continue
            if i in jagged_indices:
                continue
            if i in count_indices:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if t.ndim > dim and t.shape[dim] == batch_size:
                layout.batch_indices.append(i)

        budgeted_dim0: Dict[int, int] = {}
        total_budget_by_vidx: Dict[int, int] = {}
        pair_bs_to_padded: Dict[int, int] = {}
        pair_bs_to_count: Dict[int, int] = {}
        bs_idx_set: set[int] = set()
        for jp in layout.jagged_pairs:
            budgeted_dim0[jp.values_idx] = tensors[jp.values_idx].shape[0]
            total_budget_by_vidx[jp.values_idx] = jp.total_budget
            if jp.bs_idx is not None:
                pair_bs_val = tensors[jp.bs_idx].shape[dim]
                pair_bs_to_padded[pair_bs_val] = jp.padded_bs
                if jp.count_idx is not None:
                    pair_bs_to_count[pair_bs_val] = jp.count_idx
                bs_idx_set.add(jp.bs_idx)  # pyre-ignore

        for i, t in enumerate(tensors):
            if i in self._skip_indices:
                continue
            if i in jagged_indices or i in bs_idx_set:
                continue
            if i in layout.batch_indices:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if t.ndim >= 1:
                matched = False
                # Candidate-axis (a jagged pair's batch ref) match takes PRIORITY
                # over the values-axis (budgeted_dim0) match: when a pair is
                # 1-value-per-candidate its values dim0 collides with the
                # candidate count, and such tensors (e.g. supervision_label_t)
                # MUST be padded count-aware — the count-blind pad_to_static
                # per-chip split is undefined on the non-uniform candidate axis.
                for pb, ppb in pair_bs_to_padded.items():
                    if t.shape[0] == pb:
                        layout.inferred_pad_targets[i] = ppb
                        cidx = pair_bs_to_count.get(pb)
                        if cidx is not None:
                            layout.candidate_count_source_indices[i] = cidx
                        matched = True
                        break
                if not matched:
                    for v_idx, v_dim0 in budgeted_dim0.items():
                        if t.shape[0] == v_dim0:
                            layout.inferred_pad_targets[i] = total_budget_by_vidx[v_idx]
                            matched = True
                            break

        self._warn_orphans(tensors, layout, jagged_indices, bs_idx_set, count_indices)
        return layout

    def _warn_orphans(
        self,
        tensors: List[torch.Tensor],
        layout: _Layout,
        jagged_indices: Set[int],
        bs_idx_set: Set[int],
        count_indices: Set[int],
    ) -> None:
        """Log any tensor input not claimed by a batch/jagged/count/derived rule.

        Orphans keep their input_transform-expanded shape and are a likely cause
        of downstream broadcast errors, so surface them loudly.
        """
        recognized: Set[int] = set()
        recognized.update(jagged_indices)
        recognized.update(layout.batch_indices)
        recognized.update(bs_idx_set)
        recognized.update(layout.inferred_pad_targets.keys())
        recognized.update(count_indices)
        recognized.update(self._skip_indices)
        orphans: List[Tuple[int, List[int]]] = []
        for i, t in enumerate(tensors):
            if i in recognized or not isinstance(t, torch.Tensor):
                continue
            orphans.append((i, list(t.shape)))
        if not orphans:
            return
        logger.warning(
            "[PAD discover] %d orphan tensor input(s) NOT recognized "
            "(kept input_transform-expanded shape, likely cause of "
            "downstream broadcast errors):",
            len(orphans),
        )
        for i, shape in orphans:
            logger.warning("  arg[%d] shape=%s", i, shape)

    # ------------------------------------------------------------------
    # RegionTransform interface: input pre-transform
    # ------------------------------------------------------------------

    def transform_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        tensors = list(args)
        layout = self._discover(tensors)
        dim = self.batch_dim
        pid = self._pad_id_base
        np_ = self.num_processes
        # Snapshot each pair's ORIGINAL (pre-pad) per-sample count tensor before
        # the pairs loop below rebinds tensors[count_idx] via pad_count_to_sum.
        # Candidate-axis dense tensors (the NRO batch-ref bs_idx tensor and any
        # candidate-axis derived/dict tensors) are padded with these counts as
        # 1-key jagged lengths, so each chip receives its own samples' real
        # candidates padded to the per-chip budget (matching pad_count_to_sum).
        original_counts: Dict[int, torch.Tensor] = {}
        for jp in layout.jagged_pairs:
            cidx = jp.count_idx
            if cidx is not None:
                original_counts[cidx] = tensors[cidx]
        # Pad batch tensors (shape[batch_dim] == batch_size) up to
        # padded_bs.  Per-sample count tensors (e.g. num_candidates) are
        # NOT in batch_indices — they are handled by ``pad_count_to_sum``
        # below via their owning pair's ``count_idx``, which guarantees
        # both shape AND sum alignment.  bind_signature enforces that
        # any pair with bs_idx also has count_idx, so non-trivial count
        # tensors never reach this loop.
        if layout.padded_bs != layout.batch_size:
            for i in layout.batch_indices:
                tensors[i] = pad_to_static(
                    tensors[i], layout.padded_bs, dim, pid, 0.0, np_
                )
                pid += 1
        padded_bs_indices: set[int] = set()
        for jp in layout.jagged_pairs:
            if jp.bs_idx is not None and jp.bs_idx not in padded_bs_indices:
                # The NRO batch-ref tensor is a candidate-axis dense tensor: its
                # dim0 is the REAL jagged candidate sum, not a multiple of
                # num_local, so the count-blind pad_to_static per-chip split is
                # undefined. Pad it via pad_jagged_values with num_candidates as
                # a single-key length vector.
                count_idx = jp.count_idx
                assert count_idx is not None  # guaranteed by bind_signature
                tensors[jp.bs_idx] = pad_candidate_axis_to_static(  # pyre-ignore
                    tensors[jp.bs_idx],
                    original_counts[count_idx],
                    jp.padded_bs,
                    dim=dim,
                    pad_id=pid,
                    num_processes=np_,
                )
                padded_bs_indices.add(jp.bs_idx)  # pyre-ignore
                pid += 1
            if jp.count_idx is not None:
                tensors[jp.count_idx] = pad_count_to_sum(  # pyre-ignore
                    tensors[jp.count_idx],
                    layout.padded_bs,
                    jp.padded_bs,
                    pid,
                    np_,
                )
                pid += 1
            # When the pair's batch axis is a candidate axis (bs_idx set), its
            # rows aren't chip-uniform: pass the pre-pad count so the per-chip
            # split follows num_candidates (matching pad_count_to_sum). Sample-
            # axis pairs pass count=None and keep the uniform split.
            pair_count = (
                original_counts[jp.count_idx] if jp.count_idx is not None else None
            )
            padded_values, padded_lengths = pad_jagged_pair_to_static(
                tensors[jp.values_idx],
                tensors[jp.lengths_idx],
                jp.per_key_budgets,
                jp.padded_bs,
                dim=dim,
                pad_id=pid,
                num_processes=np_,
                count=pair_count,
            )
            tensors[jp.values_idx] = padded_values
            tensors[jp.lengths_idx] = padded_lengths
            pid += 1
        for i, target in layout.inferred_pad_targets.items():
            cidx = layout.candidate_count_source_indices.get(i)
            if cidx is not None:
                # Candidate-axis derived tensor: pad count-aware, same as bs_idx.
                tensors[i] = pad_candidate_axis_to_static(
                    tensors[i],
                    original_counts[cidx],
                    target,
                    dim=dim,
                    pad_id=pid,
                    num_processes=np_,
                )
            else:
                tensors[i] = pad_to_static(tensors[i], target, 0, pid, 0.0, np_)
            pid += 1
        self._pad_dict_tensors(tensors, layout, args, original_counts, pid)
        return tuple(tensors), kwargs

    def _pad_dict_tensors(
        self,
        tensors: List[Any],
        layout: _Layout,
        args: Tuple[Any, ...],
        original_counts: Dict[int, torch.Tensor],
        pid: int,
    ) -> None:
        """Pad tensor values inside dict args (e.g. ``all_labels``) in place.

        Match against both the jagged total-values dim and the pair batch dim,
        since dict values typically have B_NRO (pair_bs), not the total values
        count.  Candidate-axis dict values (keyed by a pair's pre-pad batch
        size) must be padded count-aware; values-axis dict values (keyed by a
        pair's total values dim) can use the plain ``pad_to_static``.  Keyed
        separately because with ``per_sample_budgets == 1`` a pair's
        total_budget equals its padded_bs, so keying by target value alone
        would be ambiguous.
        """
        dim = self.batch_dim
        np_ = self.num_processes
        cand_axis_dict: Dict[int, Tuple[int, int]] = {}
        val_axis_dict: Dict[int, int] = {}
        for jp in layout.jagged_pairs:
            val_axis_dict[args[jp.values_idx].shape[0]] = jp.total_budget
            if jp.bs_idx is not None and jp.count_idx is not None:
                cand_axis_dict[args[jp.bs_idx].shape[dim]] = (
                    jp.padded_bs,
                    jp.count_idx,
                )
        for i, item in enumerate(tensors):
            if not isinstance(item, dict) or i in self._skip_indices:
                continue
            padded_dict = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    if v.shape[0] in cand_axis_dict:
                        tgt, cidx = cand_axis_dict[v.shape[0]]
                        if tgt != v.shape[0]:
                            v = pad_candidate_axis_to_static(
                                v,
                                original_counts[cidx],
                                tgt,
                                pad_id=pid,
                                num_processes=np_,
                            )
                            pid += 1
                    else:
                        target = val_axis_dict.get(v.shape[0])
                        if target is not None and target != v.shape[0]:
                            v = pad_to_static(v, target, 0, pid, 0.0, np_)
                            pid += 1
                padded_dict[k] = v
            tensors[i] = padded_dict

    # ------------------------------------------------------------------
    # RegionTransform interface: post-compile finalization
    # ------------------------------------------------------------------

    def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
        mark_padding_graph_breaks(gm)
