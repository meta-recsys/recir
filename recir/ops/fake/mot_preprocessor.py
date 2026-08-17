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
from hammer.v3.modules.mot.features import MetadataFeatures
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
#   ("kjt", desc, keys, values_dtype, lengths_dtype)
#   ("jagged_1d", desc, dtype, group)
#   ("batch_dense_2d", desc, dtype)
#   ("batch_dense_1d", desc, dtype)
#   ("nro_dense_1d", desc, dtype)
#   ("scalar_tensor", desc, dtype)
#   ("dynamic_int", desc)
#   ("batch_int", desc)
#   ("nro_int", desc)
#
# Batch dimensions always come from live input KJT lengths. Jagged groups share
# one unbacked SymInt. Exports use a separate explicit policy: scalar tensors
# stay scalar, and `unsampled_inputs` is one-dimensional over the live NRO batch.
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
        self._all_label_keys: List[str] = sorted(all_labels.keys())

        # Sample input also tells us if UIH_FLOAT / SUPERVISION_WEIGHT
        # are present.
        self._has_uih_float: bool = (
            sample_input.get_tensor(InputFeatures.UIH_FLOAT) is not None
        )
        self._has_supervision_weight: bool = (
            sample_input.get_tensor(InputFeatures.SUPERVISION_WEIGHT) is not None
        )

        self._output_spec = self._build_output_spec(sample_input, sample_output)
        self._export_spec = self._build_export_spec(sample_output)

        # Register in global registry.
        self._instance_id: str = f"mot_preproc_{id(self)}"
        _INSTANCE_REGISTRY[self._instance_id] = self

    def __del__(self) -> None:
        instance_id = getattr(self, "_instance_id", None)
        registry = globals().get("_INSTANCE_REGISTRY")
        if instance_id is not None and registry is not None:
            registry.pop(instance_id, None)

    @staticmethod
    def _one_dim_output_spec(
        desc: Any, desc_name: str, dtype: torch.dtype
    ) -> Tuple[str, Any, ...]:
        is_batch_dense = desc_name.endswith(
            ("/seq_lengths", "/lengths_with_contextual", "/num_targets")
        ) or ("/additional_emb_input/" in desc_name and desc_name.endswith("/lengths"))
        if is_batch_dense:
            return ("batch_dense_1d", desc, dtype)
        if desc_name.endswith(("/query_time", ".nro_padding_mask")):
            return ("nro_dense_1d", desc, dtype)
        parallel_suffixes = (
            "/past_weights",
            "/past_watch_times",
            "/past_vpvd_deltas",
            "/past_comment_vpvd",
            "/past_video_lengths",
            "/past_timestamps_jagged",
        )
        if desc_name.endswith(parallel_suffixes):
            return ("jagged_1d", desc, dtype, desc_name.rsplit("/", 1)[0])
        if "/additional_emb_input/" in desc_name and desc_name.endswith("/values"):
            return ("jagged_1d", desc, dtype, desc_name)
        raise NotImplementedError(
            f"MoT fake output {desc} has unsupported 1D policy; "
            "add an explicit shape policy"
        )

    @staticmethod
    def _tensor_output_spec(desc: Any, value: torch.Tensor) -> Tuple[str, Any, ...]:
        desc_name = str(getattr(desc, "name", ""))
        if value.dim() == 0 and desc_name.endswith(
            ("/runtime_max_seq_len_t", "/l2_runtime_max_seq_len_t")
        ):
            return ("scalar_tensor", desc, value.dtype)
        if value.dim() == 1:
            return TraceableMoTPreprocessor._one_dim_output_spec(
                desc, desc_name, value.dtype
            )
        if value.dim() == 2 and desc_name.endswith("/past_timestamps"):
            return ("batch_dense_2d", desc, value.dtype)
        raise NotImplementedError(
            f"MoT fake output {desc} has unsupported shape {tuple(value.shape)}; "
            "add an explicit shape policy"
        )

    @staticmethod
    def _build_output_spec(
        sample_input: FeatureBundle,
        sample_output: FeatureBundle,
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
        # Known parallel per-event arrays (PAST_WEIGHTS / PAST_WATCH_TIMES /
        # PAST_VIDEO_LENGTHS / ...) get one group id, so the fake kernel can
        # give the whole group a single unbacked SymInt. Sharing one symbol makes
        # their elementwise broadcast provable at trace time. _decompose_output
        # re-checks the descriptor policy on every batch.
        for desc in sample_output:
            val = sample_output.get(desc)
            if val is None:
                continue
            if desc in sample_input:
                continue
            if isinstance(val, KeyedJaggedTensor):
                spec.append(
                    (
                        "kjt",
                        desc,
                        val.keys(),
                        val.values().dtype,
                        val.lengths().dtype,
                    )
                )
            elif isinstance(val, torch.Tensor):
                spec.append(TraceableMoTPreprocessor._tensor_output_spec(desc, val))
            elif isinstance(val, int):
                desc_name = str(getattr(desc, "name", ""))
                if desc_name.endswith(".batch_size"):
                    spec.append(("batch_int", desc))
                elif desc == MetadataFeatures.TOTAL_NUM_TARGETS:
                    spec.append(("nro_int", desc))
                else:
                    spec.append(("dynamic_int", desc))
            else:
                raise NotImplementedError(
                    f"MoT fake output {desc} has unsupported type {type(val)}; "
                    "add an explicit output policy"
                )
        return spec

    @staticmethod
    def _build_export_spec(
        sample_output: FeatureBundle,
    ) -> List[Tuple[str, str, torch.dtype]]:
        """Build explicit shape policies for string-keyed exports."""
        spec: List[Tuple[str, str, torch.dtype]] = []
        for export_key, export_val in sample_output.exports.items():
            if not isinstance(export_val, torch.Tensor):
                raise NotImplementedError(
                    f"MoT export {export_key!r} has unsupported type {type(export_val)}"
                )
            if export_val.dim() == 0:
                spec.append(("scalar", export_key, export_val.dtype))
            elif export_key == "unsampled_inputs" and export_val.dim() == 1:
                spec.append(("nro_1d", export_key, export_val.dtype))
            else:
                raise NotImplementedError(
                    f"MoT export {export_key!r} has unsupported shape "
                    f"{tuple(export_val.shape)}; add an explicit shape policy"
                )
        return spec

    # ----- public entry point ----------------------------------------------

    def forward(self, features: FeatureBundle) -> FeatureBundle:
        input_tensors = self._decompose_input(features)
        self._constrain_input_batch_shapes(input_tensors)
        output_tensors = torch.ops.recir_preproc.mot_preprocessor_forward(
            self._instance_id, input_tensors
        )
        return self._reassemble_output(features, output_tensors)

    def _constrain_input_batch_shapes(self, input_tensors: List[torch.Tensor]) -> None:
        """Tell ShapeEnv which independently dynamic input dimensions agree."""
        ro_lengths = input_tensors[1]
        nro_lengths = input_tensors[3]
        num_candidates = input_tensors[4]
        if self._ro_kjt_keys:
            torch._check(
                ro_lengths.shape[0] == len(self._ro_kjt_keys) * num_candidates.shape[0]
            )

        nro_batch = None
        if self._all_label_keys:
            nro_batch = input_tensors[7].shape[0]
        elif self._has_supervision_weight:
            nro_batch = input_tensors[6].shape[1]
        if nro_batch is None:
            return

        if self._nro_kjt_keys:
            torch._check(nro_lengths.shape[0] == len(self._nro_kjt_keys) * nro_batch)
        for label in input_tensors[7:]:
            torch._check(label.shape[0] == nro_batch)

    def _nro_batch_size(self, input_tensors: List[torch.Tensor]) -> Any:
        n_nro_keys = len(self._nro_kjt_keys)
        if not n_nro_keys:
            raise RuntimeError("Cannot infer NRO batch size without NRO KJT keys")
        return input_tensors[3].shape[0] // n_nro_keys

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
            stride=num_candidates.shape[0],
        )
        nro_kjt = KeyedJaggedTensor(
            keys=self._nro_kjt_keys,
            values=nro_values,
            lengths=nro_lengths,
            stride=self._nro_batch_size(input_tensors),
        )

        builder = FeatureBundleBuilder()
        builder.set(InputFeatures.UIH_ID_LIST, ro_kjt)
        builder.set(InputFeatures.CANDIDATE_ID_LIST, nro_kjt)
        builder.set(InputFeatures.NUM_CANDIDATES, num_candidates.to(torch.int32))
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

    @staticmethod
    def _tensor_shape_matches(
        kind: str, value: torch.Tensor, ro_batch: int, nro_batch: int
    ) -> bool:
        if kind == "scalar_tensor":
            return value.dim() == 0
        if kind == "jagged_1d":
            return value.dim() == 1
        if kind == "batch_dense_1d":
            return tuple(value.shape) == (ro_batch,)
        if kind == "nro_dense_1d":
            return tuple(value.shape) == (nro_batch,)
        return (
            kind == "batch_dense_2d" and value.dim() == 2 and value.shape[0] == ro_batch
        )

    @staticmethod
    def _validate_jagged_group(
        spec: Tuple[str, Any, ...],
        value: torch.Tensor,
        seen_by_group: Dict[Any, Tuple[Any, int]],
    ) -> None:
        desc = spec[1]
        group = spec[3]
        length = int(value.shape[0])
        seen = seen_by_group.get(group)
        if seen is None:
            seen_by_group[group] = (desc, length)
            return
        if seen[1] != length:
            raise RuntimeError(
                f"MoT preprocessor jagged group {group} is not parallel at "
                f"runtime: {seen[0]} has length {seen[1]} but {desc} has "
                f"length {length}. The descriptors share an explicit sequence "
                f"group, and the traced graph uses one SymInt for that group."
            )

    @staticmethod
    def _validate_kjt_output(
        spec: Tuple[str, Any, ...], value: KeyedJaggedTensor, ro_batch: int
    ) -> List[torch.Tensor]:
        _, desc, keys, values_dtype, lengths_dtype = spec
        if (
            value.keys() != keys
            or value.stride() != ro_batch
            or value.values().dtype != values_dtype
            or value.lengths().dtype != lengths_dtype
        ):
            raise RuntimeError(f"MoT output KJT {desc} changed its traced contract")
        return [value.values().clone(), value.lengths().clone()]

    @classmethod
    def _validate_tensor_output(
        cls,
        spec: Tuple[str, Any, ...],
        value: torch.Tensor,
        ro_batch: int,
        nro_batch: int,
        seen_by_group: Dict[Any, Tuple[Any, int]],
    ) -> torch.Tensor:
        kind, desc, expected_dtype = spec[:3]
        if kind == "jagged_1d":
            cls._validate_jagged_group(spec, value, seen_by_group)
        if value.dtype != expected_dtype or not cls._tensor_shape_matches(
            kind, value, ro_batch, nro_batch
        ):
            raise RuntimeError(
                f"MoT output {desc} changed its traced {kind} contract: "
                f"expected dtype {expected_dtype} with RO batch {ro_batch} and "
                f"NRO batch {nro_batch}, got dtype {value.dtype} and shape "
                f"{tuple(value.shape)}"
            )
        return value.clone()

    @classmethod
    def _decompose_output_value(
        cls,
        spec: Tuple[str, Any, ...],
        value: Any,
        ref_tensor: torch.Tensor,
        ro_batch: int,
        nro_batch: int,
        seen_by_group: Dict[Any, Tuple[Any, int]],
    ) -> List[torch.Tensor]:
        kind = spec[0]
        if kind == "kjt":
            assert isinstance(value, KeyedJaggedTensor)
            return cls._validate_kjt_output(spec, value, ro_batch)
        if kind in (
            "jagged_1d",
            "batch_dense_2d",
            "nro_dense_1d",
            "scalar_tensor",
            "batch_dense_1d",
        ):
            assert isinstance(value, torch.Tensor)
            return [
                cls._validate_tensor_output(
                    spec, value, ro_batch, nro_batch, seen_by_group
                )
            ]
        if kind in ("dynamic_int", "batch_int", "nro_int"):
            size = int(value)
            if kind == "nro_int" and size != nro_batch:
                raise RuntimeError(
                    f"MoT output {spec[1]} must equal the NRO batch "
                    f"{nro_batch}; got {size}"
                )
            return [torch.zeros(size, dtype=torch.int64, device=ref_tensor.device)]
        raise RuntimeError(f"Unknown output kind: {kind}")

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
        # The traced graph carries ONE symbol per jagged group, so members that
        # diverge at runtime would contradict shapes Inductor already baked in.
        # Check the descriptor policy here; this runs inside the opaque op and
        # therefore adds no nodes to the traced graph.
        ro_batch = bundle.require_kjt(InputFeatures.UIH_ID_LIST).stride()
        nro_batch = bundle.require_kjt(InputFeatures.CANDIDATE_ID_LIST).stride()
        jagged_group_seen: Dict[Any, Tuple[Any, int]] = {}
        for spec in self._output_spec:
            val = bundle.get(spec[1])
            out.extend(
                self._decompose_output_value(
                    spec,
                    val,
                    ref_tensor,
                    ro_batch,
                    nro_batch,
                    jagged_group_seen,
                )
            )
        # Exports (string-keyed _exports dict, not iterated by the regular
        # _output_spec loop above).
        for export_kind, export_key, dtype in self._export_spec:
            export_val = bundle.get_export(export_key)
            if not isinstance(export_val, torch.Tensor):
                raise RuntimeError(f"MoT export {export_key!r} is missing at runtime")
            if export_val.dtype != dtype:
                raise RuntimeError(
                    f"MoT export {export_key!r} changed dtype from {dtype} "
                    f"to {export_val.dtype}"
                )
            if export_kind == "scalar" and export_val.dim() != 0:
                raise RuntimeError(f"MoT export {export_key!r} must remain scalar")
            if export_kind == "nro_1d" and tuple(export_val.shape) != (nro_batch,):
                raise RuntimeError(
                    f"MoT export {export_key!r} must have NRO shape {(nro_batch,)}; "
                    f"got {tuple(export_val.shape)}"
                )
            out.append(export_val.clone())
        return out

    # ----- trace path (fake kernel emits SymInt-shaped placeholders) -------

    @staticmethod
    def _jagged_group_size(group: Any, cache: Dict[Any, Any]) -> Any:
        """One unbacked SymInt shared by a jagged group (see _build_output_spec)."""
        size = cache.get(group)
        if size is None:
            size = torch.library.get_ctx().new_dynamic_size()
            torch._check_is_size(size)
            cache[group] = size
        return size

    def _fake_output_for_spec(
        self,
        spec: Tuple[str, Any, ...],
        ro_values: torch.Tensor,
        ro_lengths: torch.Tensor,
        batch: Any,
        nro_batch: Any,
        jagged_group_sizes: Dict[Any, Any],
    ) -> List[torch.Tensor]:
        kind = spec[0]
        if kind == "kjt":
            _, _, keys, values_dtype, lengths_dtype = spec
            values_size = torch.library.get_ctx().new_dynamic_size()
            torch._check_is_size(values_size)
            return [
                torch.empty(values_size, dtype=values_dtype, device=ro_values.device),
                torch.empty(
                    len(keys) * batch,
                    dtype=lengths_dtype,
                    device=ro_lengths.device,
                ),
            ]
        if kind == "jagged_1d":
            _, _, dtype, group = spec
            size = self._jagged_group_size(group, jagged_group_sizes)
            return [torch.empty(size, dtype=dtype, device=ro_values.device)]
        if kind == "batch_dense_2d":
            _, _, dtype = spec
            width = torch.library.get_ctx().new_dynamic_size()
            torch._check_is_size(width)
            return [torch.empty(batch, width, dtype=dtype, device=ro_values.device)]
        if kind == "scalar_tensor":
            return [torch.empty((), dtype=spec[2], device=ro_values.device)]
        if kind == "nro_dense_1d":
            return [torch.empty(nro_batch, dtype=spec[2], device=ro_values.device)]
        if kind == "batch_dense_1d":
            return [torch.empty(batch, dtype=spec[2], device=ro_values.device)]
        if kind == "dynamic_int":
            size = torch.library.get_ctx().new_dynamic_size()
            torch._check_is_size(size)
            return [torch.empty(size, dtype=torch.int64, device=ro_values.device)]
        if kind == "batch_int":
            return [torch.empty(batch, dtype=torch.int64, device=ro_values.device)]
        if kind == "nro_int":
            return [torch.empty(nro_batch, dtype=torch.int64, device=ro_values.device)]
        raise RuntimeError(f"Unknown output kind: {kind}")

    def _fake_outputs(self, input_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        # Reference inputs for device/dtype inheritance.
        ro_values = input_tensors[0]
        ro_lengths = input_tensors[1]
        # KJT lengths carry K_ro * B, which loses its affine equality to the
        # canonical per-request input when partial lowering builds a fresh
        # ShapeEnv. Keep every RO-shaped output on the num_candidates symbol.
        batch = input_tensors[4].shape[0]
        # Keep data-dependent NRO outputs unbacked during the top-level trace.
        nro_batch = torch.library.get_ctx().new_dynamic_size()
        torch._check_is_size(nro_batch)
        jagged_group_sizes: Dict[Any, Any] = {}
        for spec in self._output_spec:
            out.extend(
                self._fake_output_for_spec(
                    spec,
                    ro_values,
                    ro_lengths,
                    batch,
                    nro_batch,
                    jagged_group_sizes,
                )
            )
        for export_kind, _key, dtype in self._export_spec:
            if export_kind == "scalar":
                out.append(torch.empty((), dtype=dtype, device=ro_values.device))
            elif export_kind == "nro_1d":
                out.append(torch.empty(nro_batch, dtype=dtype, device=ro_values.device))
            else:
                raise RuntimeError(f"Unknown MoT export kind: {export_kind}")
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
                _, _, kjt_keys, _values_dtype, _lengths_dtype = spec
                values = output_tensors[t_idx]
                t_idx += 1
                lengths = output_tensors[t_idx]
                t_idx += 1
                kjt = KeyedJaggedTensor(keys=kjt_keys, values=values, lengths=lengths)
                out.set(desc, kjt, validate=False)
            elif kind in (
                "jagged_1d",
                "batch_dense_2d",
                "nro_dense_1d",
                "scalar_tensor",
                "batch_dense_1d",
            ):
                out.set(desc, output_tensors[t_idx], validate=False)
                t_idx += 1
            elif kind in ("dynamic_int", "batch_int", "nro_int"):
                # Extract int value from tensor's shape[0] (SymInt at trace).
                int_val = output_tensors[t_idx].shape[0]
                t_idx += 1
                out.set(desc, int_val, validate=False)
            else:
                raise RuntimeError(f"Unknown output kind: {kind}")
        # Exports go into the parallel _exports dict (not yielded by
        # FeatureBundle.__iter__ -- string-keyed).
        for _kind, export_key, _dtype in self._export_spec:
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
