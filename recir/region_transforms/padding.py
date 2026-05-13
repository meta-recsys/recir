# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

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
from typing import Any, Dict, List, Optional, Tuple, Union

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
_PAD_JAGGED_SIZES: Dict[int, Tuple[int, List[int]]] = {}

_INSTANCE_ID_COUNTER: itertools.count[int] = itertools.count()
_PAD_ID_STRIDE: int = 1_000_000


# ---------------------------------------------------------------------------
# Custom ops
# ---------------------------------------------------------------------------


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
    if current > per_host_target:
        return x.narrow(dim, 0, per_host_target).clone()
    pw = [0] * (2 * x.ndim)
    pw[2 * (x.ndim - 1 - dim) + 1] = per_host_target - current
    return F.pad(x, pw, value=pad_value)


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
    if current >= per_host_target_size:
        # Edge case: input already at/over target shape.  No dummy slots
        # to distribute into; bump the last entry to absorb any remaining
        # sum delta.  Unreachable in the standard flow (pair_padded_bs is
        # always strictly greater than pair_bs).
        x = x.narrow(0, 0, per_host_target_size).clone()
        deficit = per_host_target_sum - int(x.sum().item())
        if deficit != 0:
            x[-1] = x[-1] + deficit
        return x
    num_dummy = per_host_target_size - current
    deficit = per_host_target_sum - int(x.sum().item())
    per_d = deficit // num_dummy
    rem = deficit - per_d * num_dummy  # 0 .. num_dummy - 1 when deficit >= 0
    dummy = torch.full((num_dummy,), per_d, dtype=x.dtype, device=x.device)
    if rem > 0:
        dummy[:rem] = dummy[:rem] + 1
    return torch.cat([x, dummy], dim=0)


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
    if current >= original_size:
        return grad.narrow(dim, 0, original_size).clone()
    pw = [0] * (2 * grad.ndim)
    pw[2 * (grad.ndim - 1 - dim) + 1] = original_size - current
    return F.pad(grad, pw)


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


@torch.library.custom_op("recir_padding::pad_jagged_lengths", mutates_args=())
def pad_jagged_lengths(
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    target_batch: int,
    num_processes: int = 1,
) -> torch.Tensor:
    """Pad a jagged lengths tensor with dummy entries so that each key's
    total equals its budget and the output has
    ``target_batch * len(per_key_budgets)`` entries.

    ``per_key_budgets`` and ``target_batch`` are GLOBAL sizes (matching
    the FX trace shape).  At runtime each host applies ``// num_processes``
    so its local lengths tensor is padded only to its per-host shard size.
    """
    per_host_budgets = [b // num_processes for b in per_key_budgets]
    per_host_target_batch = target_batch // num_processes
    num_keys = len(per_host_budgets)
    B_real = lengths.shape[0] // num_keys
    num_dummy = per_host_target_batch - B_real

    key_lengths = lengths.reshape(num_keys, B_real)
    parts: List[torch.Tensor] = []
    for k in range(num_keys):
        budget = per_host_budgets[k]
        kl = key_lengths[k]
        key_total = int(kl.sum().item())
        if key_total > budget:
            kl = _trim_lengths(kl, budget)
            key_total = budget
        key_pad = budget - key_total
        if key_pad > 0 and num_dummy > 0:
            per_d = key_pad // num_dummy
            rem = key_pad % num_dummy
            dummy_k = torch.full(
                (num_dummy,), per_d, dtype=lengths.dtype, device=lengths.device
            )
            if rem > 0:
                dummy_k[:rem] = dummy_k[:rem] + 1
        else:
            dummy_k = torch.zeros(num_dummy, dtype=lengths.dtype, device=lengths.device)
        parts.append(torch.cat([kl, dummy_k]))
    return torch.cat(parts)


@pad_jagged_lengths.register_fake  # pyre-ignore[56]
def _pad_lengths_fake(
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    target_batch: int,
    num_processes: int = 1,
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
) -> torch.Tensor:
    """Pad jagged values per-key so each key's segment reaches its budget.

    ``per_key_budgets`` are GLOBAL budgets; runtime divides by
    ``num_processes`` to get per-host budgets.
    """
    per_host_budgets = [b // num_processes for b in per_key_budgets]
    num_keys = len(per_host_budgets)
    B = lengths.shape[0] // num_keys
    key_lengths = lengths.reshape(num_keys, B)
    per_key_sizes: List[int] = []
    parts: List[torch.Tensor] = []
    offset = 0
    for k in range(num_keys):
        budget = per_host_budgets[k]
        key_total = int(key_lengths[k].sum().item())
        per_key_sizes.append(key_total)
        key_vals = values.narrow(dim, offset, key_total)
        current = key_vals.shape[dim]
        if current == budget:
            parts.append(key_vals.clone())
        elif current > budget:
            parts.append(key_vals.narrow(dim, 0, budget).clone())
        else:
            pw = [0] * (2 * values.ndim)
            pw[2 * (values.ndim - 1 - dim) + 1] = budget - current
            parts.append(F.pad(key_vals, pw))
        offset += key_total
    if pad_id >= 0:
        _PAD_JAGGED_SIZES[pad_id] = (values.shape[dim], per_key_sizes)
    return torch.cat(parts, dim=dim)


@pad_jagged_values.register_fake  # pyre-ignore[56]
def _pad_jagged_values_fake(
    values: torch.Tensor,
    lengths: torch.Tensor,
    per_key_budgets: List[int],
    dim: int = 0,
    pad_id: int = -1,
    num_processes: int = 1,
) -> torch.Tensor:
    if pad_id >= 0:
        _PAD_JAGGED_SIZES[pad_id] = (values.shape[dim], [])
    shape = list(values.shape)
    shape[dim] = sum(per_key_budgets)  # global budget for FX shape
    return torch.empty(shape, dtype=values.dtype, device=values.device)


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
    _, per_key_sizes = _PAD_JAGGED_SIZES[pad_id]
    parts: List[torch.Tensor] = []
    offset = 0
    for k in range(len(per_host_budgets)):
        budget = per_host_budgets[k]
        orig = per_key_sizes[k]
        if budget >= orig:
            parts.append(grad.narrow(dim, offset, orig).clone())
        else:
            key_grad = grad.narrow(dim, offset, budget).clone()
            pw = [0] * (2 * grad.ndim)
            pw[2 * (grad.ndim - 1 - dim) + 1] = orig - budget
            parts.append(F.pad(key_grad, pw))
        offset += budget
    return torch.cat(parts, dim=dim)


@unpad_jagged_grad.register_fake  # pyre-ignore[56]
def _unpad_jagged_grad_fake(
    grad: torch.Tensor,
    pad_id: int,
    per_key_budgets: List[int],
    dim: int = 0,
    num_processes: int = 1,
) -> torch.Tensor:
    total_original, _ = _PAD_JAGGED_SIZES[pad_id]
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
) -> Tuple[torch.Tensor, None, None, None, None, None]:
    return (
        torch.ops.recir_padding.unpad_jagged_grad(
            grad, ctx.pad_id, ctx.per_key_budgets, ctx.dim, ctx.num_processes
        ),
        None,
        None,
        None,
        None,
        None,
    )


def _pad_jagged_values_setup_context(
    ctx: Any,
    inputs: Tuple[torch.Tensor, torch.Tensor, List[int], int, int, int],
    output: torch.Tensor,
) -> None:
    _, _, per_key_budgets, dim, pad_id, num_processes = inputs
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
        self.derived: Dict[int, int] = {}


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
        num_devices: Local device count.
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
    ) -> None:
        self._raw_jagged_pairs: Dict[str, JaggedPairConfig] = jagged_pairs
        self._raw_batch_size_arg: str = batch_size_arg
        self._pad_id_base: int = next(_INSTANCE_ID_COUNTER) * _PAD_ID_STRIDE
        self.num_devices: int = num_devices
        # num_processes is the number of OS processes (= num_hosts) the
        # GSPMD pipeline runs across. Pad ops use it to map FX-trace global
        # target shapes to per-host runtime shapes (target // num_processes).
        # See pad_to_static for details.
        self.num_processes: int = num_processes
        self.batch_dim: int = batch_dim
        self.batch_size_arg_idx: int = 0
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
        """Resolve string names to positional indices in the signature."""
        sig = inspect.signature(func)
        names: List[str] = []
        for name, param in sig.parameters.items():
            ann = param.annotation
            if ann is not inspect.Parameter.empty and isinstance(ann, type):
                if issubclass(ann, (nn.Module, torch.optim.Optimizer)):
                    continue
            elif name in ("mod", "opt", "model", "optimizer"):
                continue
            names.append(name)
        self.batch_size_arg_idx = names.index(self._raw_batch_size_arg)
        self._resolved_pairs = []
        for v_name, config in self._raw_jagged_pairs.items():
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
        padded_bs = int(math.ceil(batch_size / nd)) * nd
        if padded_bs == batch_size and self._resolved_pairs:
            padded_bs += nd
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
            else:
                pair_bs = batch_size
            pair_padded_bs = int(math.ceil(pair_bs / nd)) * nd
            if pair_padded_bs == pair_bs:
                pair_padded_bs += nd
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
        bs_idx_set: set[int] = set()
        for jp in layout.jagged_pairs:
            budgeted_dim0[jp.values_idx] = tensors[jp.values_idx].shape[0]
            total_budget_by_vidx[jp.values_idx] = jp.total_budget
            if jp.bs_idx is not None:
                pair_bs_val = tensors[jp.bs_idx].shape[dim]
                pair_bs_to_padded[pair_bs_val] = jp.padded_bs
                bs_idx_set.add(jp.bs_idx)  # pyre-ignore

        for i, t in enumerate(tensors):
            if i in jagged_indices or i in bs_idx_set:
                continue
            if i in layout.batch_indices:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            if t.ndim >= 1:
                matched = False
                for v_idx, v_dim0 in budgeted_dim0.items():
                    if t.shape[0] == v_dim0:
                        layout.derived[i] = total_budget_by_vidx[v_idx]
                        matched = True
                        break
                if not matched:
                    for pb, ppb in pair_bs_to_padded.items():
                        if t.shape[0] == pb:
                            layout.derived[i] = ppb
                            break

        recognized: set[int] = set()
        recognized.update(jagged_indices)
        recognized.update(layout.batch_indices)
        recognized.update(bs_idx_set)
        recognized.update(layout.derived.keys())
        recognized.update(count_indices)
        orphans: List[Tuple[int, List[int]]] = []
        for i, t in enumerate(tensors):
            if i in recognized:
                continue
            if not isinstance(t, torch.Tensor):
                continue
            orphans.append((i, list(t.shape)))
        if orphans:
            logger.warning(
                "[PAD discover] %d orphan tensor input(s) NOT recognized "
                "(kept input_transform-expanded shape, likely cause of "
                "downstream broadcast errors):",
                len(orphans),
            )
            for i, shape in orphans:
                logger.warning("  arg[%d] shape=%s", i, shape)

        return layout

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
                tensors[jp.bs_idx] = pad_to_static(  # pyre-ignore
                    tensors[jp.bs_idx], jp.padded_bs, dim, pid, 0.0, np_
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
            original_lengths = tensors[jp.lengths_idx]
            tensors[jp.lengths_idx] = pad_jagged_lengths(
                tensors[jp.lengths_idx],
                jp.per_key_budgets,
                jp.padded_bs,
                np_,
            )
            tensors[jp.values_idx] = pad_jagged_values(
                tensors[jp.values_idx],
                original_lengths,
                jp.per_key_budgets,
                dim,
                pid,
                np_,
            )
            pid += 1
        for i, target in layout.derived.items():
            tensors[i] = pad_to_static(tensors[i], target, 0, pid, 0.0, np_)
            pid += 1
        # Pad tensor values inside dicts (e.g. all_labels).
        # Match against both jagged total-values dim and pair batch
        # dim, since dict values typically have B_NRO (pair_bs), not
        # the total values count.
        dim0_to_target: Dict[int, int] = {}
        for jp in layout.jagged_pairs:
            dim0_to_target[args[jp.values_idx].shape[0]] = jp.total_budget
            if jp.bs_idx is not None:
                dim0_to_target[args[jp.bs_idx].shape[dim]] = jp.padded_bs
        for i, item in enumerate(tensors):
            if not isinstance(item, dict):
                continue
            padded_dict = {}
            for k, v in item.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1:
                    target = dim0_to_target.get(v.shape[0])
                    if target is not None and target != v.shape[0]:
                        v = pad_to_static(v, target, 0, pid, 0.0, np_)
                        pid += 1
                padded_dict[k] = v
            tensors[i] = padded_dict
        return tuple(tensors), kwargs

    # ------------------------------------------------------------------
    # RegionTransform interface: post-compile finalization
    # ------------------------------------------------------------------

    def finalize_compiled_graph(self, gm: fx.GraphModule) -> None:
        mark_padding_graph_breaks(gm)
