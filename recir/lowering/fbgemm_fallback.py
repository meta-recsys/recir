# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Pure PyTorch fallbacks for fbgemm ops used by generative_recommenders.

When fbgemm_gpu is not available (e.g., on TPU), importing this module
registers pure PyTorch implementations of the fbgemm ops that
generative_recommenders uses internally (jagged_to_padded_dense,
dense_to_jagged, etc.).

The ops are registered with CompositeImplicitAutograd dispatch and also
added to recir's SPMD_DECOMP_TABLE so that make_fx(tracing_mode="fake")
decomposes them into standard aten ops that torchax can lower to JAX/XLA.

Usage:
    import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers ops as side effect
"""

import logging
from typing import Any, List, Optional, Sequence, Tuple

import torch
from recir.decompositions import SPMD_DECOMP_TABLE

logger: logging.Logger = logging.getLogger(__name__)


# jagged_impl.py was eliminated: route the base jagged<->dense gathers through
# the recir custom ops so they get the shard-local shard_map lowering on TPU
# (and a single torch reference impl in recir/ops/jagged_ops.py). These thin
# wrappers keep the composite ops + fbgemm-absent registrations below working.
def _jagged_to_padded_dense_impl(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: Sequence[int],
    padding_value: float = 0.0,
) -> torch.Tensor:
    import recir.ops.jagged_ops  # noqa: F401

    return torch.ops.recir.jagged_to_padded_dense(
        values, offsets, list(max_lengths), padding_value
    )


def _dense_to_jagged_forward_impl(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
) -> torch.Tensor:
    import recir.ops.jagged_ops  # noqa: F401

    return torch.ops.recir.dense_to_jagged(dense, offsets, total_L)[0]


def _is_fbgemm_available() -> bool:
    try:
        torch.ops.fbgemm.jagged_to_padded_dense
        return True
    except AttributeError:
        return False


# ── Pure PyTorch implementations ────────────────────────────────────────


def _batched_complete_cumsum_impl(values: torch.Tensor) -> torch.Tensor:
    # values: (B, N, ...) -> output: (B, N+1, ...)
    # Batched version of asynchronous_complete_cumsum: prepend zero slice on dim=1.
    # Use zeros_like(narrow) instead of zeros(shape) so the FX graph doesn't
    # hardcode concrete shape literals that break when XLA runtime shapes differ.
    cumsum = torch.cumsum(values, dim=1)
    zeros = torch.zeros_like(values.narrow(1, 0, 1))
    return torch.cat([zeros, cumsum], dim=1)


def _dense_to_jagged_impl(
    dense: torch.Tensor,
    offsets: List[torch.Tensor],
    total_L: Optional[int] = None,
):
    result = _dense_to_jagged_forward_impl(dense, offsets, total_L)
    return result, offsets[0]


def _asynchronous_complete_cumsum_impl(lengths: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(lengths, dim=0)
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, cumsum], dim=0)


def _jagged_dense_elementwise_add_jagged_output_impl(
    jagged: torch.Tensor,
    offsets: List[torch.Tensor],
    dense: torch.Tensor,
):
    seq_offsets = offsets[0]
    max_len = dense.shape[1]

    padded = _jagged_to_padded_dense_impl(jagged, offsets, [max_len], 0.0)
    summed = padded + dense
    result, _ = _dense_to_jagged_impl(summed, offsets, jagged.shape[0])
    return result, seq_offsets


def _jagged_index_select_impl(
    values: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
):
    offsets = torch.zeros(
        lengths.numel() + 1, dtype=lengths.dtype, device=lengths.device
    )
    offsets[1:] = torch.cumsum(lengths, dim=0)

    selected_lengths = lengths[indices]
    out_offsets = torch.zeros(
        selected_lengths.numel() + 1,
        dtype=selected_lengths.dtype,
        device=selected_lengths.device,
    )
    out_offsets[1:] = torch.cumsum(selected_lengths, dim=0)

    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    N = lengths.numel()

    if max_len == 0 or N == 0:
        result = torch.empty(
            0, *values.shape[1:], dtype=values.dtype, device=values.device
        )
        return result, out_offsets

    padded = _jagged_to_padded_dense_impl(values, [offsets], [max_len], 0.0)
    selected_padded = padded[indices]
    result, _ = _dense_to_jagged_impl(
        selected_padded, [out_offsets], int(out_offsets[-1].item())
    )
    return result, out_offsets


# ── Registration ────────────────────────────────────────────────────────

# Keep the Library object alive at module level. If it is garbage-collected,
# the op schemas it defines are unregistered while the implementations remain
# dangling, causing "schema_.has_value() INTERNAL ASSERT FAILED" on any call.
_fbgemm_lib: Optional[torch.library.Library] = None


def register_fbgemm_fallbacks() -> bool:
    """Register pure PyTorch fallbacks for fbgemm ops.

    When fbgemm is not available, registers pure PyTorch op implementations.
    Always populates SPMD_DECOMP_TABLE so that make_fx decomposes fbgemm ops
    into standard aten ops for XLA tracing — even when fbgemm is installed,
    its CUDA kernels cannot run during fake-tensor tracing on TPU.

    Returns True if op fallbacks were registered, False if fbgemm was already
    available (only SPMD_DECOMP_TABLE entries were added).
    """
    global _fbgemm_lib

    registered_ops = False
    if not _is_fbgemm_available():
        lib = torch.library.Library("fbgemm", "DEF")
        _fbgemm_lib = lib

        # Define op schemas
        lib.define(
            "jagged_to_padded_dense(Tensor values, Tensor[] offsets, "
            "int[] max_lengths, float padding_value=0.0) -> Tensor"
        )
        lib.define(
            "dense_to_jagged(Tensor dense, Tensor[] offsets, "
            "int? total_L=None) -> (Tensor, Tensor)"
        )
        lib.define(
            "dense_to_jagged_forward(Tensor dense, Tensor[] offsets, "
            "int? total_L=None) -> Tensor"
        )
        lib.define(
            "jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, "
            "int[] max_lengths, float padding_value=0.0) -> Tensor"
        )
        lib.define(
            "jagged_to_padded_dense_backward(Tensor grad_output, "
            "Tensor[] offsets, int total_L) -> Tensor"
        )
        lib.define("asynchronous_complete_cumsum(Tensor lengths) -> Tensor")
        lib.define(
            "jagged_dense_elementwise_add_jagged_output("
            "Tensor jagged, Tensor[] offsets, Tensor dense"
            ") -> (Tensor, Tensor)"
        )
        lib.define(
            "jagged_index_select(Tensor values, Tensor lengths, Tensor indices"
            ") -> (Tensor, Tensor)"
        )

        # Register implementations with CompositeImplicitAutograd
        torch.library.impl(lib, "jagged_to_padded_dense", "CompositeImplicitAutograd")(
            _jagged_to_padded_dense_impl
        )
        torch.library.impl(lib, "dense_to_jagged", "CompositeImplicitAutograd")(
            _dense_to_jagged_impl
        )
        torch.library.impl(lib, "dense_to_jagged_forward", "CompositeImplicitAutograd")(
            _dense_to_jagged_forward_impl
        )
        torch.library.impl(
            lib, "jagged_to_padded_dense_forward", "CompositeImplicitAutograd"
        )(_jagged_to_padded_dense_impl)
        torch.library.impl(
            lib, "jagged_to_padded_dense_backward", "CompositeImplicitAutograd"
        )(_dense_to_jagged_forward_impl)
        torch.library.impl(
            lib, "asynchronous_complete_cumsum", "CompositeImplicitAutograd"
        )(_asynchronous_complete_cumsum_impl)
        torch.library.impl(
            lib,
            "jagged_dense_elementwise_add_jagged_output",
            "CompositeImplicitAutograd",
        )(_jagged_dense_elementwise_add_jagged_output_impl)
        torch.library.impl(lib, "jagged_index_select", "CompositeImplicitAutograd")(
            _jagged_index_select_impl
        )
        registered_ops = True

    # Always add to SPMD_DECOMP_TABLE so make_fx(tracing_mode="fake")
    # decomposes these ops into standard aten ops for XLA/JAX lowering.
    # This is needed even when fbgemm is installed because the real
    # fbgemm CUDA kernels segfault during fake-tensor tracing on TPU.
    #
    # For jagged<->dense conversions, route to recir custom ops that have
    # shard_map lowerings (keeps activations batch-sharded on TPU/GSPMD).
    # The import is done lazily inside the routing functions to avoid
    # circular dependency at module load time.

    def _route_to_recir_jagged_to_padded_dense(
        values: torch.Tensor,
        offsets: List[torch.Tensor],
        max_lengths: Sequence[int],
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        import recir.ops.jagged_ops  # noqa: F401

        return torch.ops.recir.jagged_to_padded_dense(
            values, offsets, list(max_lengths), padding_value
        )

    def _route_to_recir_dense_to_jagged(
        dense: torch.Tensor,
        offsets: List[torch.Tensor],
        total_L: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import recir.ops.jagged_ops  # noqa: F401

        return torch.ops.recir.dense_to_jagged(dense, offsets, total_L)

    def _route_to_recir_dense_to_jagged_forward(
        dense: torch.Tensor,
        offsets: List[torch.Tensor],
        total_L: Optional[int] = None,
    ) -> torch.Tensor:
        # dense_to_jagged_forward / jagged_to_padded_dense_backward are
        # single-Tensor ops; reuse the tuple route and drop the offsets output.
        return _route_to_recir_dense_to_jagged(dense, offsets, total_L)[0]

    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_to_padded_dense.default] = (
        _route_to_recir_jagged_to_padded_dense
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.dense_to_jagged.default] = (
        _route_to_recir_dense_to_jagged
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.dense_to_jagged_forward.default] = (
        _route_to_recir_dense_to_jagged_forward
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_to_padded_dense_forward.default] = (
        _route_to_recir_jagged_to_padded_dense
    )
    # Backward of jagged_to_padded_dense is dense_to_jagged. Route it to the
    # shard-local recir custom op too (NOT the inline _impl), otherwise the
    # backward pass does a global gather over [total_L, D] -> replicated 65GB
    # buffer -> HBM OOM (forwards route to recir but the backward did not).
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_to_padded_dense_backward.default] = (
        _route_to_recir_dense_to_jagged_forward
    )
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.asynchronous_complete_cumsum.default] = (
        _asynchronous_complete_cumsum_impl
    )
    SPMD_DECOMP_TABLE[
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output.default
    ] = _jagged_dense_elementwise_add_jagged_output_impl
    SPMD_DECOMP_TABLE[torch.ops.fbgemm.jagged_index_select.default] = (
        _jagged_index_select_impl
    )

    return registered_ops


# Auto-register on import
register_fbgemm_fallbacks()


# ── Route jagged concat/split to recir shard-local custom ops ──────────
# The generative_recommenders / hammer 2D-jagged concat+split helpers use
# data-dependent masking that (a) fails make_fx tracing and (b) on TPU/GSPMD
# materializes a replicated [total_L, D] buffer (OOM). Route them to the recir
# custom ops, which have shard_map lowerings that keep activations
# batch-sharded. The recir ops are each other's backward (register_autograd in
# recir/ops/jagged_ops.py), so loss.backward() differentiates them correctly.

# Kept at module scope (not inside the routing fn) so autodeps keeps the
# gen_rec dep; the routing fn references it.
from generative_recommenders.ops.pytorch import (  # noqa: E402
    pt_jagged_tensors as _pt_jt,
)


def _route_jagged_to_recir() -> None:
    """Monkey-patch the generative_recommenders and hammer 2D-jagged concat/split
    leaves to the recir custom ops -- centralized here instead of importing recir
    inside the model code.

    Patch targets are the leaf functions the dispatchers resolve as module-globals
    at call time (verified call graph), so the patch actually takes effect:
      - DlrmV3Dense (gen_rec) -> pt_jagged_tensors._concat_2D_jagged_jagged /
        pytorch_split_2D_jagged
      - RankFM (hammer) -> hammer.ops.jagged.pytorch_concat_2D_jagged_jagged /
        hammer.ops.pytorch.jagged._split_2d_jagged_no_prefix

    The hammer side uses a dynamic import (module name as a variable) so autodeps
    does NOT add a static build dependency on hammer -- the dlrmv3 image is
    OSS-only (no hammer) yet still imports this module; in the RankFM image hammer
    is present at runtime and the patches apply.
    """

    # ---- generative_recommenders (DlrmV3Dense) ----
    _orig_split_2D_jagged_jagged = _pt_jt._split_2D_jagged_jagged

    def _recir_concat_2D_jagged_jagged(
        values_left: torch.Tensor,
        values_right: torch.Tensor,
        max_len_left: int,
        max_len_right: int,
        offsets_left: torch.Tensor,
        offsets_right: torch.Tensor,
    ) -> torch.Tensor:
        import recir.ops.jagged_ops  # noqa: F401

        return torch.ops.recir.concat_2d_jagged(
            values_left, offsets_left, values_right, offsets_right
        )

    _pt_jt._concat_2D_jagged_jagged = _recir_concat_2D_jagged_jagged  # type: ignore[assignment]

    def _recir_pytorch_split_2D_jagged(
        max_seq_len: int,
        values: torch.Tensor,
        max_len_left: Optional[int],
        max_len_right: Optional[int],
        offsets_left: Optional[torch.Tensor],
        offsets_right: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if offsets_left is None:
            assert max_len_left is not None
            assert offsets_right is not None
            offsets_left = max_len_left * torch.arange(
                offsets_right.shape[0], device=values.device
            )
        if offsets_right is None:
            assert max_len_right is not None
            assert offsets_left is not None
            offsets_right = max_len_right * torch.arange(
                offsets_left.shape[0], device=values.device
            )
        if max_len_left is not None and max_len_right is not None:
            import recir.ops.jagged_ops  # noqa: F401

            # Static output rows. For the uniform path total left/right rows
            # equal B * max_len; matches the prior static-shape swap's shapes.
            B = offsets_left.shape[0] - 1
            return torch.ops.recir.split_2d_jagged(
                values,
                offsets_left,
                offsets_right,
                B * max_len_left,
                B * max_len_right,
            )
        return _orig_split_2D_jagged_jagged(
            max_seq_len=max_seq_len,
            values=values,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )

    # No torch.fx.wrap: the recir tracer is make_fx, which records the recir op
    # regardless (and ignores fx.wrap); fx.wrap only matters for symbolic_trace,
    # which does not run on the recir/TPU path that imports this module.
    _pt_jt.pytorch_split_2D_jagged = _recir_pytorch_split_2D_jagged  # type: ignore[assignment]

    # ---- hammer (RankFM), best-effort ----
    def _opt_import(name: str) -> Any:
        import importlib

        try:
            return importlib.import_module(name)
        except ImportError:
            return None

    _h_jagged = _opt_import("hammer.ops.jagged")
    _h_pt_jagged = _opt_import("hammer.ops.pytorch.jagged")
    if _h_jagged is None or _h_pt_jagged is None:
        return

    _orig_concat = _h_jagged.pytorch_concat_2D_jagged_jagged

    def _recir_pytorch_concat_2D_jagged_jagged(
        max_seq_len_left: int,
        offsets_left: torch.Tensor,
        values_left: torch.Tensor,
        max_seq_len_right: int,
        offsets_right: torch.Tensor,
        values_right: torch.Tensor,
        is_replace: bool = False,
        n_prefix_from_right: int = 0,
    ) -> torch.Tensor:
        # recir concat handles the plain interleave only; defer the replace /
        # prefix variants to the original implementation.
        if is_replace or n_prefix_from_right != 0:
            return _orig_concat(
                max_seq_len_left,
                offsets_left,
                values_left,
                max_seq_len_right,
                offsets_right,
                values_right,
                is_replace,
                n_prefix_from_right,
            )
        import recir.ops.jagged_ops  # noqa: F401

        return torch.ops.recir.concat_2d_jagged(
            values_left, offsets_left, values_right, offsets_right
        )

    _h_jagged.pytorch_concat_2D_jagged_jagged = _recir_pytorch_concat_2D_jagged_jagged

    def _recir_split_2d_jagged_no_prefix(
        values: torch.Tensor,
        offsets_left: torch.Tensor,
        offsets_right: torch.Tensor,
        seq_len_a: Optional[int] = None,
        seq_len_b: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import recir.ops.jagged_ops  # noqa: F401

        if seq_len_a is None:
            seq_len_a = int(offsets_left[-1].item())
        if seq_len_b is None:
            seq_len_b = int(values.shape[0]) - seq_len_a
        return torch.ops.recir.split_2d_jagged(
            values, offsets_left, offsets_right, seq_len_a, seq_len_b
        )

    _h_pt_jagged._split_2d_jagged_no_prefix = _recir_split_2d_jagged_no_prefix


_route_jagged_to_recir()
