# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""FakeTensor (Meta dispatch) wrapper for hammer v3 ``MoTPreprocessor``.

``MoTPreprocessor.forward`` is full of patterns that ``make_fx(tracing_mode="fake")``
cannot handle directly:
  * ``num_nro_candidates.sum().item()`` (line 503 of mot_preprocessor.py)
  * ``ro_timestamp.repeat_interleave(repeats=num_nro_candidates, ...)`` (line 504)
  * ``fx_infer_max_len`` / ``fx_infer_sum`` helpers that internally call
    ``.max().item()`` / ``.sum().item()``
  * KJT operations whose internals call ``.tolist()`` / ``.item()``

Strategy: replace the entire ``MoTPreprocessor`` instance with a thin
wrapper (``TraceableMoTPreprocessor``) that flattens the input
``FeatureBundle`` into a list of tensors, invokes a single opaque
``torch.library.custom_op``, then reassembles the output ``FeatureBundle``
from the op's outputs.

The custom op's ``register_fake`` mints fresh ``SymInt``-sized placeholder
tensors via ``ctx.new_dynamic_size()`` for jagged-dim outputs (and marks
them via ``torch._check_is_size`` so downstream size-oblivious predicates
work).  The op's real impl runs the original preprocessor with concrete
runtime tensors, where ``.item()`` etc. are fine.

Usage:
    from recir.ops.fake.mot_preprocessor import patch_mot_preprocessor_for_tracing
    patch_mot_preprocessor_for_tracing(model, sample_input)
    # Now make_fx(tracing_mode="fake") can trace through MoTPreprocessor.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from hammer.v3.modules.core.features.builder import FeatureBundleBuilder
from hammer.v3.modules.core.features.bundle import FeatureBundle
from hammer.v3.modules.core.features.input_features import InputFeatures
from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# Module-global registry: instance_id (str) -> wrapper.  Both real impl and
# fake kernel of the custom op need access to per-instance metadata
# (output structure, KJT keys, etc.) — passing them through the op
# signature would require variable-length args + closures, which custom_op
# does not support.
_INSTANCE_REGISTRY: Dict[str, "TraceableMoTPreprocessor"] = {}

_op_registered: bool = False


# ---------------------------------------------------------------------------
# Output spec ("recipe" describing one output of the preprocessor)
# ---------------------------------------------------------------------------
#
# Each entry in self._output_spec is one of:
#   ("kjt", desc, kjt_keys: List[str], stride: int)
#       -> 2 tensors in the output list: values, lengths
#       -> reassembled as KJT with the given keys + stride from input
#   ("jagged_1d", desc, dtype)
#       -> 1 tensor with SymInt dim 0 (per-batch jagged total)
#   ("batch_dense_2d", desc, B: int, dtype)
#       -> 1 tensor with shape (B, SymInt)  (e.g. PAST_TIMESTAMPS)
#   ("batch_dense_1d", desc, shape: Tuple[int, ...], dtype)
#       -> 1D tensor with shape[0] == ro_kjt_stride (per-batch length tensor)
#   ("high_dim_tensor", desc, shape: Tuple[int, ...], dtype)
#       -> fallback for >=3D tensors; full concrete shape from sample
#   ("dynamic_int", desc)
#       -> 1 entry encoded as int64 tensor whose shape[0] is a SymInt
#       -> stored as nontensor int in FeatureBundle
#   ("batch_int", desc, value: int)
#       -> 1 entry encoded as int64 tensor whose shape[0] = stride
#       -> only used for BATCH_SIZE (known to equal ro_kjt.stride())
#   ("dict_jagged_1d", desc, dict_keys: List[str], dict_dtypes: List[dtype])
#       -> N tensors (one per dict key), each jagged 1D
#       -> reassembled as Dict[str, Tensor]
# ---------------------------------------------------------------------------


def _register_op() -> None:
    """Register the custom op (idempotent)."""
    global _op_registered
    if _op_registered:
        return
    _op_registered = True

    @torch.library.custom_op("recir_preproc::mot_preprocessor_forward", mutates_args=())
    def _mot_preprocessor_forward_op(
        instance_id: str,
        input_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Real impl: reconstruct FeatureBundle, call real preprocessor,
        flatten outputs into a single tensor list.

        ``custom_op`` schema does not support ``List[int]`` returns, so int
        outputs are encoded as 1D int64 tensors whose ``shape[0]`` carries
        the value.  The wrapper extracts via ``t.shape[0]``; this works
        both at runtime (concrete int) and at trace (SymInt).
        """
        wrapper = _INSTANCE_REGISTRY[instance_id]
        return wrapper._real_call(input_tensors)

    @_mot_preprocessor_forward_op.register_fake
    def _(
        instance_id: str,
        input_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Fake kernel: emit SymInt-shaped placeholders per output_spec."""
        wrapper = _INSTANCE_REGISTRY[instance_id]
        return wrapper._fake_outputs(input_tensors)


# ---------------------------------------------------------------------------
# TraceableMoTPreprocessor
# ---------------------------------------------------------------------------


class TraceableMoTPreprocessor(nn.Module):
    """Drop-in replacement for ``MoTPreprocessor`` that runs the original
    behind an opaque custom op with a SymInt-aware fake kernel.
    """

    def __init__(
        self,
        real_preprocessor: MoTPreprocessor,
        sample_input: FeatureBundle,
    ) -> None:
        super().__init__()
        # Hold the real preprocessor without registering as submodule
        # (so it's not double-counted in named_modules / named_parameters).
        object.__setattr__(self, "_real", real_preprocessor)

        # Run the real preprocessor on the sample input ONCE to capture
        # the output FeatureBundle's structure (descriptor list, types,
        # KJT keys, etc.).  This drives both the input-decompose and
        # output-reassemble logic.
        with torch.no_grad():
            sample_output: FeatureBundle = real_preprocessor(sample_input)

        # Build input spec: which descriptors to extract, in which order,
        # and how (kjt vs tensor vs dict).  Hard-coded for the
        # MoTPreprocessor.forward input contract:
        #   InputFeatures.UIH_ID_LIST          -> KJT (values, lengths)
        #   InputFeatures.CANDIDATE_ID_LIST    -> KJT (values, lengths)
        #   InputFeatures.NUM_CANDIDATES       -> tensor
        #   InputFeatures.UIH_FLOAT            -> Optional[tensor]
        #   InputFeatures.SUPERVISION_WEIGHT   -> Optional[tensor]
        #   InputFeatures.ALL_LABELS           -> Dict[str, tensor]
        ro_kjt = sample_input.require_kjt(InputFeatures.UIH_ID_LIST)
        nro_kjt = sample_input.require_kjt(InputFeatures.CANDIDATE_ID_LIST)
        all_labels = sample_input.require(InputFeatures.ALL_LABELS)
        assert isinstance(all_labels, dict)

        self._ro_kjt_keys: List[str] = ro_kjt.keys()
        self._nro_kjt_keys: List[str] = nro_kjt.keys()
        self._ro_kjt_stride: int = ro_kjt.stride()
        self._nro_kjt_stride: int = nro_kjt.stride()
        self._all_label_keys: List[str] = sorted(all_labels.keys())

        # Sample input also tells us if UIH_FLOAT / SUPERVISION_WEIGHT
        # are present.
        self._has_uih_float: bool = (
            sample_input.get_tensor(InputFeatures.UIH_FLOAT) is not None
        )
        self._has_supervision_weight: bool = (
            sample_input.get_tensor(InputFeatures.SUPERVISION_WEIGHT) is not None
        )

        self._output_spec = self._build_output_spec(
            sample_input, sample_output, self._ro_kjt_stride
        )
        self._export_spec = self._build_export_spec(sample_output)

        # Register in global registry.
        self._instance_id: str = f"mot_preproc_{id(self)}"
        _INSTANCE_REGISTRY[self._instance_id] = self

    def __del__(self) -> None:
        _INSTANCE_REGISTRY.pop(self._instance_id, None)

    @staticmethod
    def _build_output_spec(
        sample_input: FeatureBundle,
        sample_output: FeatureBundle,
        ro_kjt_stride: int,
    ) -> List[Tuple[str, Any, ...]]:
        """Introspect sample_output once to build the recipe used by
        _decompose_output, _fake_outputs, and _reassemble_output.

        Each entry tells those three methods how to extract / synthesize /
        re-set one descriptor's value.  See module-level docs for the
        spec kinds.

        Descriptors already in sample_input are skipped: they pass through
        via input_features.clone() in _reassemble_output and don't need to
        round-trip through the custom op.
        """
        spec: List[Tuple[str, Any, ...]] = []
        for desc in sample_output:
            val = sample_output.get(desc)
            if val is None:
                continue
            if desc in sample_input:
                continue
            if isinstance(val, KeyedJaggedTensor):
                spec.append(("kjt", desc, val.keys(), val.stride()))
            elif isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    if val.shape[0] == ro_kjt_stride:
                        # 1D per-batch length tensor (e.g. SEQ_LENGTHS).
                        spec.append(
                            ("batch_dense_1d", desc, tuple(val.shape), val.dtype)
                        )
                    else:
                        # Jagged 1D (e.g. PAST_WEIGHTS, jagged timestamps).
                        spec.append(("jagged_1d", desc, val.dtype))
                elif val.dim() == 2:
                    # (B, runtime_max_seq_len) - dim 1 dynamic, dim 0 = batch.
                    spec.append(("batch_dense_2d", desc, val.shape[0], val.dtype))
                else:
                    # Fallback: assume fully fixed shape for >=3D tensors.
                    spec.append(("high_dim_tensor", desc, tuple(val.shape), val.dtype))
            elif isinstance(val, int):
                # BATCH_SIZE special case: known to equal ro_kjt.stride().
                # All other int outputs are data-dependent at runtime.
                if str(getattr(desc, "name", "")).endswith("BATCH_SIZE"):
                    spec.append(("batch_int", desc, ro_kjt_stride))
                else:
                    spec.append(("dynamic_int", desc))
            elif isinstance(val, dict):
                # Dict of tensors (e.g. additional embedding values keyed
                # by feature name).  All entries assumed jagged 1D.
                # Capture per-key dtypes so the fake kernel can emit
                # placeholders with the correct dtype.
                d_keys = sorted(val.keys())
                d_dtypes = [val[k].dtype for k in d_keys]
                spec.append(("dict_jagged_1d", desc, d_keys, d_dtypes))
            else:
                logger.warning(
                    "Skipping unknown output type %s for descriptor %s",
                    type(val),
                    desc,
                )
        return spec

    @staticmethod
    def _build_export_spec(
        sample_output: FeatureBundle,
    ) -> List[Tuple[str, Tuple[int, ...], torch.dtype]]:
        """Capture the bundle's _exports separately.  FeatureBundle.__iter__
        does NOT yield string-keyed exports, so the descriptor-based
        _output_spec loop misses them entirely.
        """
        spec: List[Tuple[str, Tuple[int, ...], torch.dtype]] = []
        for export_key, export_val in sample_output.exports.items():
            if isinstance(export_val, torch.Tensor):
                spec.append((export_key, tuple(export_val.shape), export_val.dtype))
            else:
                logger.warning(
                    "Skipping non-tensor export %s of type %s",
                    export_key,
                    type(export_val),
                )
        return spec

    # ----- public entry point ----------------------------------------------

    def forward(self, features: FeatureBundle) -> FeatureBundle:
        input_tensors = self._decompose_input(features)
        output_tensors = torch.ops.recir_preproc.mot_preprocessor_forward(
            self._instance_id, input_tensors
        )
        return self._reassemble_output(features, output_tensors)

    # ----- input decomposition (FeatureBundle -> List[Tensor]) -------------

    def _decompose_input(self, features: FeatureBundle) -> List[torch.Tensor]:
        """Extract input tensors in a fixed order matching _real_call's
        reconstruction logic.
        """
        ro_kjt = features.require_kjt(InputFeatures.UIH_ID_LIST)
        nro_kjt = features.require_kjt(InputFeatures.CANDIDATE_ID_LIST)
        num_candidates = features.require_tensor(InputFeatures.NUM_CANDIDATES)
        all_labels = features.require(InputFeatures.ALL_LABELS)
        ro_float = features.get_tensor(InputFeatures.UIH_FLOAT)
        sup_weight = features.get_tensor(InputFeatures.SUPERVISION_WEIGHT)

        out: List[torch.Tensor] = [
            ro_kjt.values(),
            ro_kjt.lengths(),
            nro_kjt.values(),
            nro_kjt.lengths(),
            num_candidates,
        ]
        # Optional inputs: pass empty sentinel tensor if absent so the op
        # signature stays fixed.  Real-impl checks `numel() == 0`.
        if self._has_uih_float:
            assert ro_float is not None
            out.append(ro_float)
        else:
            out.append(num_candidates.new_empty(0))
        if self._has_supervision_weight:
            assert sup_weight is not None
            out.append(sup_weight)
        else:
            out.append(num_candidates.new_empty(0))
        # Labels in sorted-key order.
        for k in self._all_label_keys:
            out.append(all_labels[k])
        return out

    def _reconstruct_input_bundle(
        self, input_tensors: List[torch.Tensor]
    ) -> FeatureBundle:
        """Inverse of _decompose_input: rebuild a FeatureBundle from the
        flat tensor list.  Used only inside _real_call (runtime, not trace)."""
        idx = 0
        ro_values = input_tensors[idx]
        idx += 1
        ro_lengths = input_tensors[idx]
        idx += 1
        nro_values = input_tensors[idx]
        idx += 1
        nro_lengths = input_tensors[idx]
        idx += 1
        num_candidates = input_tensors[idx]
        idx += 1
        ro_float = input_tensors[idx]
        idx += 1
        sup_weight = input_tensors[idx]
        idx += 1
        labels: Dict[str, torch.Tensor] = {}
        for k in self._all_label_keys:
            labels[k] = input_tensors[idx]
            idx += 1

        ro_kjt = KeyedJaggedTensor(
            keys=self._ro_kjt_keys,
            values=ro_values,
            lengths=ro_lengths,
        )
        nro_kjt = KeyedJaggedTensor(
            keys=self._nro_kjt_keys,
            values=nro_values,
            lengths=nro_lengths,
        )

        builder = FeatureBundleBuilder()
        builder.set(InputFeatures.UIH_ID_LIST, ro_kjt)
        builder.set(InputFeatures.CANDIDATE_ID_LIST, nro_kjt)
        builder.set(InputFeatures.NUM_CANDIDATES, num_candidates)
        if self._has_uih_float and ro_float.numel() > 0:
            builder.set(InputFeatures.UIH_FLOAT, ro_float)
        if self._has_supervision_weight and sup_weight.numel() > 0:
            builder.set(InputFeatures.SUPERVISION_WEIGHT, sup_weight)
        builder.set(InputFeatures.ALL_LABELS, labels)
        return builder.build()

    # ----- runtime path (real preprocessor execution) ----------------------

    def _real_call(self, input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Reconstruct FeatureBundle, run real preprocessor, decompose output."""
        bundle = self._reconstruct_input_bundle(input_tensors)
        out_bundle = self._real(bundle)
        return self._decompose_output(out_bundle, input_tensors[0])

    def _decompose_output(
        self, bundle: FeatureBundle, ref_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """Extract outputs from the FeatureBundle into a flat tensor list,
        in the order described by self._output_spec.

        Nontensor int outputs are encoded as 1D int64 tensors whose
        ``shape[0]`` carries the value (since custom_op schema doesn't
        accept ``List[int]``).  ``ref_tensor`` is used to inherit device.

        All real-tensor outputs are ``.clone()``-d before return.  The
        ``custom_op`` runtime forbids any output from aliasing an input
        or another output; the real ``MoTPreprocessor`` is free to pass
        input tensors (e.g. unchanged lengths or labels) through into
        its output bundle, so we conservatively clone every extracted
        tensor.  Cost is dominated by the preprocessor's own work.
        """
        out: List[torch.Tensor] = []
        for spec in self._output_spec:
            kind = spec[0]
            desc = spec[1]
            val = bundle.get(desc)
            if kind == "kjt":
                assert isinstance(val, KeyedJaggedTensor)
                out.append(val.values().clone())
                out.append(val.lengths().clone())
            elif kind in (
                "jagged_1d",
                "batch_dense_2d",
                "batch_dense_1d",
                "high_dim_tensor",
            ):
                assert isinstance(val, torch.Tensor)
                out.append(val.clone())
            elif kind in ("dynamic_int", "batch_int"):
                # Encode the int as a 1D int64 tensor of size = the value.
                # Wrapper extracts via t.shape[0] (concrete int at runtime).
                int_val = int(val) if kind == "dynamic_int" else int(spec[2])
                out.append(
                    torch.empty(int_val, dtype=torch.int64, device=ref_tensor.device)
                )
            elif kind == "dict_jagged_1d":
                d_keys = spec[2]
                assert isinstance(val, dict)
                for k in d_keys:
                    out.append(val[k].clone())
            else:
                raise RuntimeError(f"Unknown output kind: {kind}")
        # Exports (string-keyed _exports dict, not iterated by the regular
        # _output_spec loop above).
        for export_key, _shape, _dtype in self._export_spec:
            export_val = bundle.get_export(export_key)
            assert isinstance(export_val, torch.Tensor), (
                f"Export {export_key!r} missing or not a tensor at runtime"
            )
            out.append(export_val.clone())
        return out

    # ----- trace path (fake kernel emits SymInt-shaped placeholders) -------

    def _fake_outputs(self, input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        ctx = torch.library.get_ctx()
        out: List[torch.Tensor] = []
        # Reference inputs for device/dtype inheritance.
        ro_values = input_tensors[0]
        ro_lengths = input_tensors[1]
        for spec in self._output_spec:
            kind = spec[0]
            if kind == "kjt":
                _, _, kjt_keys, kjt_stride = spec
                K = len(kjt_keys)
                # Per-feature jagged total -> SymInt.
                values_size = ctx.new_dynamic_size()
                torch._check_is_size(values_size)
                out.append(ro_values.new_empty(values_size))
                # Lengths shape is K * stride - fixed.
                out.append(ro_lengths.new_empty(K * kjt_stride))
            elif kind == "jagged_1d":
                _, _, dtype = spec
                sz = ctx.new_dynamic_size()
                torch._check_is_size(sz)
                out.append(torch.empty(sz, dtype=dtype, device=ro_values.device))
            elif kind == "batch_dense_2d":
                _, _, dim0, dtype = spec
                sz = ctx.new_dynamic_size()
                torch._check_is_size(sz)
                out.append(torch.empty(dim0, sz, dtype=dtype, device=ro_values.device))
            elif kind in ("batch_dense_1d", "high_dim_tensor"):
                # Both store full concrete shape from sample (1D per-batch
                # vs >=3D fallback) -- same handling here.
                _, _, shape, dtype = spec
                out.append(torch.empty(shape, dtype=dtype, device=ro_values.device))
            elif kind == "dynamic_int":
                # Emit a 1D int64 tensor whose shape[0] is a SymInt.
                # Wrapper extracts via t.shape[0] -> SymInt.
                sz = ctx.new_dynamic_size()
                torch._check_is_size(sz)
                out.append(torch.empty(sz, dtype=torch.int64, device=ro_values.device))
            elif kind == "batch_int":
                _, _, value = spec
                out.append(
                    torch.empty(int(value), dtype=torch.int64, device=ro_values.device)
                )
            elif kind == "dict_jagged_1d":
                _, _, d_keys, d_dtypes = spec
                for dtype in d_dtypes:
                    sz = ctx.new_dynamic_size()
                    torch._check_is_size(sz)
                    out.append(torch.empty(sz, dtype=dtype, device=ro_values.device))
            else:
                raise RuntimeError(f"Unknown output kind: {kind}")
        # Exports: assume fixed shape captured at __init__ time.
        for _key, shape, dtype in self._export_spec:
            out.append(torch.empty(shape, dtype=dtype, device=ro_values.device))
        return out

    # ----- output reassembly (List[Tensor] -> FeatureBundle) ---------------

    def _reassemble_output(
        self,
        input_features: FeatureBundle,
        output_tensors: List[torch.Tensor],
    ) -> FeatureBundle:
        """Inverse of _decompose_output: build a FeatureBundle from the
        flat tensor list, plus the input bundle (since FeatureBundle clone
        semantics carry input descriptors over).

        Nontensor int outputs are extracted via ``tensor.shape[0]`` -- this
        works both at runtime (concrete int) and at trace (SymInt) without
        ``.item()``.
        """
        out = input_features.clone()
        t_idx = 0
        for spec in self._output_spec:
            kind = spec[0]
            desc = spec[1]
            if kind == "kjt":
                _, _, kjt_keys, _ = spec
                values = output_tensors[t_idx]
                t_idx += 1
                lengths = output_tensors[t_idx]
                t_idx += 1
                kjt = KeyedJaggedTensor(keys=kjt_keys, values=values, lengths=lengths)
                out.set(desc, kjt, validate=False)
            elif kind in (
                "jagged_1d",
                "batch_dense_2d",
                "batch_dense_1d",
                "high_dim_tensor",
            ):
                out.set(desc, output_tensors[t_idx], validate=False)
                t_idx += 1
            elif kind in ("dynamic_int", "batch_int"):
                # Extract int value from tensor's shape[0] (SymInt at trace).
                int_val = output_tensors[t_idx].shape[0]
                t_idx += 1
                out.set(desc, int_val, validate=False)
            elif kind == "dict_jagged_1d":
                d_keys = spec[2]
                d: Dict[str, torch.Tensor] = {}
                for k in d_keys:
                    d[k] = output_tensors[t_idx]
                    t_idx += 1
                out.set(desc, d, validate=False)
            else:
                raise RuntimeError(f"Unknown output kind: {kind}")
        # Exports go into the parallel _exports dict (not yielded by
        # FeatureBundle.__iter__ -- string-keyed).
        for export_key, _shape, _dtype in self._export_spec:
            out.set_export(export_key, output_tensors[t_idx])
            t_idx += 1
        return out


# ---------------------------------------------------------------------------
# Module patcher
# ---------------------------------------------------------------------------


def patch_mot_preprocessor_for_tracing(
    model: nn.Module,
    sample_input: FeatureBundle,
) -> int:
    """Replace each ``MoTPreprocessor`` instance in ``model`` with a
    ``TraceableMoTPreprocessor`` wrapper.

    Returns the number of preprocessors patched.  Idempotent (safe to
    call multiple times; already-wrapped preprocessors are skipped).
    """
    _register_op()

    accessor = NamedMemberAccessor(model)
    patched = 0
    # Collect targets first to avoid mutating the iterator.
    to_patch: List[Tuple[str, MoTPreprocessor]] = []
    for fqn, mod in model.named_modules():
        if not isinstance(mod, MoTPreprocessor):
            continue
        if isinstance(mod, TraceableMoTPreprocessor):
            continue  # already wrapped
        to_patch.append((fqn, mod))

    for fqn, mod in to_patch:
        wrapped = TraceableMoTPreprocessor(mod, sample_input)
        accessor.swap_submodule(fqn, wrapped)
        patched += 1
        logger.info("Patched MoTPreprocessor at %s for tracing", fqn)
    return patched
