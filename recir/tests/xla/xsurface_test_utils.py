#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Shared test utilities for xsurface V3 model tracing and XLA compilation.

Contains dummy modules, override classes, model/batch builders, and helper
functions used by both the fake-tracing-compat test and the XLA compile test.
"""

import contextlib
import unittest
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hammer.ops.bf16_train_utils import set_autocast_factory
from hammer.utils import HammerKernel
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import (
    BlueReelsVDDModuleConfig,
    ConfigProvider,
)
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import (
    BlueReelsVDDTrainV3,
    build_recblock_from_xsurface_config,
)
from hammer.v3.modules.core.features.bundle import FeatureBundle
from hammer.v3.modules.core.features.descriptor import FeatureDescriptor
from hammer.v3.modules.core.features.input_features import InputFeatures
from hammer.v3.modules.core.preprocessors.preprocessor_stack import PreprocessorStack
from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
from hammer.v3.modules.mot.tokenizers.mot_sparse_arch import ECSparseArch
from hammer.v3.modules.recblock import _inputs_to_feature_bundle, RecBlock
from minimal_viable_ai.core.dataloader.dataloader import RandomDataPipe
from minimal_viable_ai.core.dataloader.dataloader_configs import (
    RandomDataSourceConfig,
    RandomROODataSourceConfig,
)
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig
from minimal_viable_ai.core.ranking_common.configs import MTMLROOTrainBatch
from recir.eager_dispatch import (
    install_graph_break_dispatch,
    mark_input,
    mark_output_with_bwd,
)
from recir.torchax.amp import hammer_autocast_factory
from torchrec import JaggedTensor, KeyedJaggedTensor


XSURFACE_PREPROCESSOR_MARKER_PREFIX = "recir.examples.rankfm.preprocessor"
XSURFACE_SPARSE_MARKER_PREFIX = "recir.examples.rankfm.sparse_arch"


def _xsurface_preprocessor_prefix(override_id: int) -> str:
    return f"{XSURFACE_PREPROCESSOR_MARKER_PREFIX}::{override_id}"


def _xsurface_sparse_prefix(override_id: int) -> str:
    return f"{XSURFACE_SPARSE_MARKER_PREFIX}::{override_id}"


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

TEST_UC_RID_HASH_SIZE: int = 1_000_000
TEST_BATCH_SIZE: int = 4
TEST_MAX_UIH_SEQ_LEN: int = 32
TEST_NUM_CANDIDATES: int = 8

# Sequential RO key used as the basis for deriving SEQ_LENGTHS at the
# override boundary: SEQ_LENGTHS[i] = HSTU_ID_LIST[i][SEQ_LENGTHS_BASIS_KEY].lengths
# + num_candidates.
SEQ_LENGTHS_BASIS_KEY: str = "datafm_fb_trait_post_id"


@contextlib.contextmanager
def _disable_sampling(real_stack: PreprocessorStack) -> Generator[None, None, None]:
    """Temporarily null out ``_uih_sampling_module`` on every
    ``FeaturePreprocessor`` under ``real_stack.mot_preprocessor`` so the
    cached output build is deterministic and represents the upper bound.
    """
    # pyre-ignore[16]: dynamic attr from PreprocessorStack setattr
    preprocessors = list(real_stack.mot_preprocessor._hstu_preprocessor_list)
    saved: List[Any] = []
    for hp in preprocessors:
        fp = hp._feature_preprocessor
        saved.append(fp._uih_sampling_module)
        fp._uih_sampling_module = None
    try:
        yield
    finally:
        for hp, s in zip(preprocessors, saved):
            hp._feature_preprocessor._uih_sampling_module = s


def _fit_to_shape(
    t: torch.Tensor, target_shape: torch.Size | Tuple[int, ...]
) -> torch.Tensor:
    """End-pad with zeros (or narrow) each dim of *t* to ``target_shape``."""
    target = tuple(int(s) for s in target_shape)
    if tuple(t.shape) == target:
        return t
    pad_widths: List[int] = []
    for actual, tgt in zip(reversed(t.shape), reversed(target)):
        pad_widths += [0, max(0, int(tgt) - int(actual))]
    out = F.pad(t, pad_widths) if any(p > 0 for p in pad_widths) else t
    for d, (actual, tgt) in enumerate(zip(out.shape, target)):
        if int(actual) > int(tgt):
            out = out.narrow(d, 0, int(tgt))
    return out


def _pad_kjt_to_cached(
    current_values: torch.Tensor,
    current_lengths: torch.Tensor,
    num_keys: int,
    cached_lengths_buf: torch.Tensor,
    num_processes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad an output KJT to the per-host cached shape."""

    global_stride = int(cached_lengths_buf.shape[0] // num_keys)  # global batch_size
    per_host_stride = global_stride // num_processes  # local batch_size
    cached_per_key_global = (
        cached_lengths_buf.view(num_keys, global_stride).sum(dim=1).tolist()
    )  # global cached length sum for each key
    per_host_per_key_budgets = [int(b // num_processes) for b in cached_per_key_global]
    # Values first — uses current_lengths to slide current per-key segments.
    # tile_pad=True: fill each key's deficit by tiling its own real ids instead
    # of zero-filling. These KJTs are embedding id-lists whose pad positions are
    # masked out downstream, so tiling keeps the pad tokens' block-owner
    # distribution uniform and avoids routing every pad (id 0) to ring-shift
    # block-owner 0 (which caused the period-8 all-to-all hot wave).
    new_values = torch.ops.recir_padding.pad_jagged_values(
        current_values, current_lengths, per_host_per_key_budgets, 0, -1, 1, True
    )
    # Lengths next — extends stride; deficit dummies absorb the growth from
    # current per-key sums to per-host cached budgets so sum(new_lengths)
    # == sum(per_host_per_key_budgets) == len(new_values).
    new_lengths = torch.ops.recir_padding.pad_jagged_lengths(
        current_lengths, per_host_per_key_budgets, per_host_stride, 1
    )
    return new_values, new_lengths


# ---------------------------------------------------------------------------
# Config provider
# ---------------------------------------------------------------------------


class TestConfigProvider(ConfigProvider):
    """ConfigProvider subclass with smaller hash sizes for testing.

    Caps ``embedding_tables`` num_embeddings post-construction so tests
    run with smaller embedding tables.
    """

    def create_config(self) -> RecModelFamilyConfig:
        config = super().create_config()
        module_config = cast(BlueReelsVDDModuleConfig, config.module_config)
        for table in module_config.embedding_tables.values():
            table.num_embeddings = min(table.num_embeddings, TEST_UC_RID_HASH_SIZE)
        return config


# ---------------------------------------------------------------------------
# Dummy modules
# ---------------------------------------------------------------------------


class DummyPreprocessorStack(nn.Module):
    """Replacement for PreprocessorStack that returns fixed-shape outputs.

    Runs the real preprocessor eagerly at construction time to capture
    input/output shapes.  Wraps inputs/outputs in
    :func:`recir.eager_dispatch.mark_input` / ``mark_output_with_bwd`` so the
    post-trace orchestrator can find the markers and wire runtime dispatch to
    the real :class:`PreprocessorStack`.

    A _grad_anchor parameter maintains autograd graph connectivity through
    the dummy so that make_fx traces backward correctly. A _batch_dep term
    adds data dependency on inputs so GSPMD can infer sharding.
    """

    def __init__(
        self,
        real_stack: PreprocessorStack,
        device: torch.device,
        batch_size: int = TEST_BATCH_SIZE,
        max_uih_seq_len: int = TEST_MAX_UIH_SEQ_LEN,
        num_candidates: int = TEST_NUM_CANDIDATES,
        override_id: int = 0,
        sample_feature_bundle: FeatureBundle | None = None,
        precomputed_output: FeatureBundle | None = None,
    ) -> None:
        super().__init__()
        self._override_id: int = override_id
        self._grad_anchor = nn.Parameter(torch.zeros(1))

        if sample_feature_bundle is not None:
            input_features = sample_feature_bundle
        else:
            # pyre-ignore
            preprocessor: MoTPreprocessor = real_stack.mot_preprocessor
            sample_bundle: FeatureBundle = preprocessor.sample_inputs(
                batch_size=batch_size,
                device=device,
                max_uih_seq_len=max_uih_seq_len,
                num_candidates=num_candidates,
            )
            ro_id_list = sample_bundle.require_kjt(InputFeatures.UIH_ID_LIST)
            nro_id_list = sample_bundle.require_kjt(InputFeatures.CANDIDATE_ID_LIST)
            num_cand = sample_bundle.require_tensor(InputFeatures.NUM_CANDIDATES)
            # pyrefly: ignore [bad-assignment]
            all_labels: Dict[str, torch.Tensor] = sample_bundle.require(
                InputFeatures.ALL_LABELS
            )
            input_features = _inputs_to_feature_bundle(
                ro_id_list, None, nro_id_list, num_cand, all_labels
            )
        if precomputed_output is not None:
            cached_output = precomputed_output
        else:
            # Sampling off — see _disable_sampling doc-string for why.
            with _disable_sampling(real_stack):
                cached_output: FeatureBundle = real_stack(input_features)

        # Capture INPUT structure for fwd_fn reconstruction at runtime.
        entries: List[Tuple[FeatureDescriptor, str, Any]] = []
        nontensor_map: Dict[FeatureDescriptor, Any] = {}
        for desc in input_features:
            val = input_features.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                entries.append((desc, "kjt", val.keys()))
            elif isinstance(val, torch.Tensor):
                entries.append((desc, "tensor", None))
            elif isinstance(val, dict):
                entries.append((desc, "dict", sorted(val.keys())))
            else:
                nontensor_map[desc] = val
        self._input_entries: List[Tuple[FeatureDescriptor, str, Any]] = entries
        self._input_nontensor_map: Dict[FeatureDescriptor, Any] = nontensor_map

        # Capture OUTPUT structure and register tensors as buffers.
        self._tensor_descs: List[FeatureDescriptor] = []
        self._kjt_descs: List[FeatureDescriptor] = []
        self._kjt_keys: List[List[str]] = []
        self._nontensor_map: Dict[FeatureDescriptor, Any] = {}
        tensor_idx = 0
        kjt_idx = 0

        for desc in cached_output:
            val = cached_output.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                self._kjt_descs.append(desc)
                self._kjt_keys.append(val.keys())
                self.register_buffer(f"_kjt_values_{kjt_idx}", val.values())
                self.register_buffer(f"_kjt_lengths_{kjt_idx}", val.lengths())
                kjt_idx += 1
            elif isinstance(val, torch.Tensor):
                self._tensor_descs.append(desc)
                self.register_buffer(f"_tensor_{tensor_idx}", val)
                tensor_idx += 1
            else:
                self._nontensor_map[desc] = val

        self._num_tensors: int = tensor_idx
        self._num_kjts: int = kjt_idx

    def forward(self, features: FeatureBundle) -> FeatureBundle:
        oid = self._override_id
        prefix = _xsurface_preprocessor_prefix(oid)

        # Mark input tensor leaves with input markers.  Iterate the actual
        # features bundle (not the sample-captured lists) so the ordering
        # matches what the pre-hook captures at trace time.  This ensures
        # the marker indices align with fwd_fn's reconstruction order.
        input_idx = 0
        anchored_inputs: List[torch.Tensor] = []
        for desc in features:
            val = features.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                anchored_inputs.append(mark_input(val.values(), prefix, input_idx))
                input_idx += 1
                anchored_inputs.append(mark_input(val.lengths(), prefix, input_idx))
                input_idx += 1
            elif isinstance(val, torch.Tensor):
                anchored_inputs.append(mark_input(val, prefix, input_idx))
                input_idx += 1
            elif isinstance(val, dict):
                for key in sorted(val.keys()):
                    if isinstance(val[key], torch.Tensor):
                        anchored_inputs.append(mark_input(val[key], prefix, input_idx))
                        input_idx += 1
        output_tensors: List[torch.Tensor] = []
        for i in range(self._num_tensors):
            buf = getattr(self, f"_tensor_{i}")
            output_tensors.append(torch.zeros_like(buf))
        for i in range(self._num_kjts):
            vbuf = getattr(self, f"_kjt_values_{i}")
            lbuf = getattr(self, f"_kjt_lengths_{i}")
            output_tensors.append(torch.zeros_like(vbuf))
            output_tensors.append(lbuf.clone())

        deps: List[torch.Tensor] = anchored_inputs + [self._grad_anchor]
        anchored = [
            mark_output_with_bwd(t, prefix, idx, deps=deps)
            for idx, t in enumerate(output_tensors)
        ]

        # Reconstruct FeatureBundle from anchored tensors.
        bundle = FeatureBundle()
        out_idx = 0
        for desc in self._tensor_descs:
            bundle.set(desc, anchored[out_idx], validate=False)
            out_idx += 1
        for i, desc in enumerate(self._kjt_descs):
            kjt = KeyedJaggedTensor(
                keys=self._kjt_keys[i],
                values=anchored[out_idx],
                lengths=anchored[out_idx + 1],
            )
            bundle.set(desc, kjt, validate=False)
            out_idx += 2
        for desc, v in self._nontensor_map.items():
            bundle.set(desc, v, validate=False)
        return bundle


class DummySparseArch(nn.Module):
    """Replacement for ECSparseArch that returns cached embeddings.

    Wraps input KJT components in :func:`recir.eager_dispatch.mark_input` and
    outputs in :func:`mark_output_with_bwd` so the post-trace orchestrator can
    find the markers and wire runtime dispatch to the real
    :class:`ECSparseArch`.

    A _grad_anchor parameter maintains autograd graph connectivity through
    the dummy so that make_fx traces backward correctly. A _batch_dep term
    adds data dependency on inputs so GSPMD can infer sharding.
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        override_id: int = 1,
    ) -> None:
        super().__init__()
        self._override_id: int = override_id
        self._grad_anchor = nn.Parameter(torch.zeros(1))
        self._feature_names: List[str] = list(cached_output.keys())
        self._input_kjt_keys: List[str] = input_kjt_keys

        for i, (_name, jt) in enumerate(cached_output.items()):
            self.register_buffer(f"_jt_values_{i}", jt.values())
            self.register_buffer(f"_jt_lengths_{i}", jt.lengths())

    def forward(
        self,
        id_list_features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        oid = self._override_id
        prefix = _xsurface_sparse_prefix(oid)

        # Mark the input KJT components so SparseArchDispatcher.install_graph
        # can rediscover them via the marker prefix and use them as dispatch
        # inputs (instead of falling back to all batch placeholders).
        m_v = mark_input(id_list_features.values(), prefix, 0)
        m_l = mark_input(id_list_features.lengths(), prefix, 1)

        # Zero-valued anchor wired into every output's computation so FX
        # trace DCE keeps the m_v / m_l input markers alive end-to-end.
        # The deps=[m_v, m_l] hint to mark_output_with_bwd alone is not
        # enough; without a real torch-op dependency the markers get pruned
        # and install_graph_break_dispatch falls back to passing all
        # train_step placeholders to fwd_fn.
        anchor = (m_v.to(torch.int64).sum() * 0) + (m_l.to(torch.int64).sum() * 0)

        deps: List[torch.Tensor] = [m_v, m_l, self._grad_anchor]

        out_idx = 0
        result: Dict[str, JaggedTensor] = {}
        for i, name in enumerate(self._feature_names):
            vbuf = getattr(self, f"_jt_values_{i}")
            lbuf = getattr(self, f"_jt_lengths_{i}")
            v_z = torch.zeros_like(vbuf) + anchor.to(vbuf.dtype)
            l_z = lbuf.clone() + anchor.to(lbuf.dtype)
            v = mark_output_with_bwd(v_z, prefix, out_idx, deps=deps)
            lengths = mark_output_with_bwd(l_z, prefix, out_idx + 1, deps=deps)
            result[name] = JaggedTensor(values=v, lengths=lengths)
            out_idx += 2
        return result


# ---------------------------------------------------------------------------
# Dispatcher classes (orchestrator-style overrides)
# ---------------------------------------------------------------------------


class PreprocessorDispatcher:
    """Orchestrator-style replacement for the legacy ``PreprocessorOverride``.

    ``install_eager`` swaps :class:`PreprocessorStack` for
    :class:`DummyPreprocessorStack` so tracing sees stable shapes (the dummy
    plants input + forward/backward markers).  ``install_graph`` finds those
    markers and installs a graph-break dispatch back to the captured real
    :class:`PreprocessorStack`.

    The input FeatureBundle structure is captured at trace time via the
    dummy's stored ``_input_entries``, ensuring the captured structure
    matches what RecBlock actually passes (including supervision_weight and
    the correct KJT keys from the batch, not the sample).
    """

    def __init__(
        self,
        device: torch.device,
        batch_size: int = TEST_BATCH_SIZE,
        max_uih_seq_len: int = TEST_MAX_UIH_SEQ_LEN,
        num_candidates: int = TEST_NUM_CANDIDATES,
        override_id: int = 0,
        num_ranks: int = 1,
        sample_feature_bundle: FeatureBundle | None = None,
        precomputed_output: FeatureBundle | None = None,
        num_processes: int = 1,
    ) -> None:
        self._device = device
        self._batch_size = batch_size
        self._max_uih_seq_len = max_uih_seq_len
        self._num_candidates = num_candidates
        self._override_id = override_id
        self._num_ranks = num_ranks
        self._orig_stack: nn.Module | None = None
        self._dummy: DummyPreprocessorStack | None = None
        self._sample_feature_bundle = sample_feature_bundle
        self._precomputed_output = precomputed_output
        # Per-host fitting: cached buffers are at GLOBAL stride (override-
        # build sample is global), runtime outputs are per-host
        # (= cached.shape with dim 0 // num_processes) so _to_sharded_jax
        # can expand back to global at the XLA boundary.
        self._num_processes: int = num_processes

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, PreprocessorStack):
                self._orig_stack = mod
                self._dummy = DummyPreprocessorStack(
                    real_stack=mod,
                    device=self._device,
                    batch_size=self._batch_size,
                    max_uih_seq_len=self._max_uih_seq_len,
                    num_candidates=self._num_candidates,
                    override_id=self._override_id,
                    sample_feature_bundle=self._sample_feature_bundle,
                    precomputed_output=self._precomputed_output,
                )
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if self._orig_stack is None or self._dummy is None:
            return gm, flat_state

        real_stack: nn.Module = self._orig_stack  # pyre-ignore[8]
        dummy: DummyPreprocessorStack = self._dummy  # pyre-ignore[8]
        num_ranks: int = self._num_ranks
        num_processes_fit: int = self._num_processes

        # Input structure captured at construction time from sample_feature_bundle.
        entries: List[Tuple[FeatureDescriptor, str, Any]] = dummy._input_entries
        input_nontensor_map: Dict[FeatureDescriptor, Any] = dummy._input_nontensor_map

        output_tensor_descs: List[FeatureDescriptor] = dummy._tensor_descs
        output_kjt_descs: List[FeatureDescriptor] = dummy._kjt_descs

        def fwd_fn(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            bundle = FeatureBundle()
            idx = 0
            for desc, type_tag, extra in entries:
                if type_tag == "kjt":
                    kjt = KeyedJaggedTensor(
                        keys=extra,
                        values=input_tensors[idx],
                        lengths=input_tensors[idx + 1],
                    )
                    bundle.set(desc, kjt, validate=False)
                    idx += 2
                elif type_tag == "tensor":
                    bundle.set(desc, input_tensors[idx], validate=False)
                    idx += 1
                elif type_tag == "dict":
                    d: Dict[str, torch.Tensor] = {}
                    for key in extra:
                        d[key] = input_tensors[idx]
                        idx += 1
                    bundle.set(desc, d, validate=False)
            for desc, v in input_nontensor_map.items():
                bundle.set(desc, v, validate=False)

            with torch.enable_grad():
                output = real_stack(bundle)

            # Fit each runtime output to the PER-HOST shape (cached shape
            # with dim 0 // num_processes).  Cached buffers are at global
            # (override-build sample is at global) but real_stack at runtime
            # processes per-host data and produces per-host outputs.  We
            # return per-host so _to_sharded_jax can expand to global —
            # matching the per-host behavior of PRT pad ops at runtime.
            def _per_host_target(cached_shape: torch.Size) -> Tuple[int, ...]:
                out = list(int(s) for s in cached_shape)
                if len(out) > 0:
                    out[0] = out[0] // num_processes_fit
                return tuple(out)

            num_nro_candidates_padded: Optional[torch.Tensor] = output.get_tensor(
                InputFeatures.NUM_CANDIDATES
            )
            seq_basis_lengths: Optional[torch.Tensor] = None
            flat_kjts: List[torch.Tensor] = []
            kjt_i = 0
            for desc in output_kjt_descs:
                cached_v = getattr(dummy, f"_kjt_values_{kjt_i}")
                cached_l = getattr(dummy, f"_kjt_lengths_{kjt_i}")
                target_v = _per_host_target(cached_v.shape)
                target_l = _per_host_target(cached_l.shape)
                num_keys = len(dummy._kjt_keys[kjt_i])
                val = output.get(desc)
                if val is not None and isinstance(val, KeyedJaggedTensor):
                    fitted_v, fitted_l = _pad_kjt_to_cached(
                        val.values(),
                        val.lengths(),
                        num_keys,
                        cached_l,
                        num_processes_fit,
                    )
                    if (
                        desc.name == "merged"
                        and SEQ_LENGTHS_BASIS_KEY in dummy._kjt_keys[kjt_i]
                    ):
                        basis_key_idx = dummy._kjt_keys[kjt_i].index(
                            SEQ_LENGTHS_BASIS_KEY
                        )
                        per_host_stride = fitted_l.shape[0] // num_keys
                        seq_basis_lengths = fitted_l.view(num_keys, per_host_stride)[
                            basis_key_idx
                        ]
                    flat_kjts.append(fitted_v)
                    flat_kjts.append(fitted_l)
                else:
                    flat_kjts.append(
                        torch.zeros(
                            target_v, dtype=cached_v.dtype, device=cached_v.device
                        )
                    )
                    flat_kjts.append(
                        torch.zeros(
                            target_l, dtype=cached_l.dtype, device=cached_l.device
                        )
                    )
                kjt_i += 1

            # Now process tensor outputs; SEQ_LENGTHS gets derived inline
            # as first_key_lengths(HSTU_ID_LIST[i]) + num_candidates.
            flat_tensors: List[torch.Tensor] = []
            tensor_i = 0
            for desc in output_tensor_descs:
                cached_t = getattr(dummy, f"_tensor_{tensor_i}")
                target_shape = _per_host_target(cached_t.shape)
                val = output.get(desc)
                if val is not None and isinstance(val, torch.Tensor):
                    if (
                        desc.name.endswith("/seq_lengths")
                        and seq_basis_lengths is not None
                        and num_nro_candidates_padded is not None
                    ):
                        fitted = (seq_basis_lengths + num_nro_candidates_padded).to(
                            cached_t.dtype
                        )
                    else:
                        fitted = _fit_to_shape(val, target_shape)
                    flat_tensors.append(fitted)
                else:
                    flat_tensors.append(
                        torch.zeros(
                            target_shape,
                            dtype=cached_t.dtype,
                            device=cached_t.device,
                        )
                    )
                tensor_i += 1

            # FX trace order: all tensors first, then KJT (values, lengths) pairs.
            flat: List[torch.Tensor] = flat_tensors + flat_kjts
            # Return raw (grad-preserving) tensors; the install helper wraps
            # this callable with save-for-backward + detach.
            return tuple(flat)

        install_graph_break_dispatch(
            gm,
            _xsurface_preprocessor_prefix(self._override_id),
            fwd_fn,
            num_state_tensors=len(flat_state),
            num_ranks=num_ranks,
        )
        return gm, flat_state


class SparseArchDispatcher:
    """Orchestrator-style replacement for the legacy ``SparseArchOverride``.

    ``install_eager`` swaps :class:`ECSparseArch` for :class:`DummySparseArch`
    so tracing sees stable shapes (the dummy plants input + forward/backward
    markers).  ``install_graph`` finds those markers and installs a graph-
    break dispatch back to the captured real :class:`ECSparseArch`.
    """

    def __init__(
        self,
        cached_output: Dict[str, JaggedTensor],
        input_kjt_keys: List[str],
        override_id: int = 1,
        num_ranks: int = 1,
    ) -> None:
        self._cached_output = cached_output
        self._input_kjt_keys = input_kjt_keys
        self._override_id = override_id
        self._num_ranks = num_ranks
        self._orig_sparse_arch: nn.Module | None = None
        self._dummy: DummySparseArch | None = None

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, ECSparseArch):
                self._orig_sparse_arch = mod
                self._dummy = DummySparseArch(
                    self._cached_output,
                    input_kjt_keys=self._input_kjt_keys,
                    override_id=self._override_id,
                )
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

    def install_graph(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if self._orig_sparse_arch is None or self._dummy is None:
            return gm, flat_state

        real_arch: nn.Module = self._orig_sparse_arch
        input_kjt_keys: List[str] = self._input_kjt_keys
        feature_names: List[str] = self._dummy._feature_names
        num_ranks: int = self._num_ranks

        def fwd_fn(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            # input_tensors == (values, lengths) thanks to mark_input markers
            # in DummySparseArch.forward (kept alive through trace by the
            # zero-valued anchor that depends on them).
            values = input_tensors[0]
            lengths = input_tensors[1]
            total = int(lengths.sum().item())
            if values.shape[0] < total:
                pad_size = total - values.shape[0]
                values = torch.nn.functional.pad(
                    values, [0] * (2 * values.ndim - 1) + [pad_size]
                )
            elif values.shape[0] > total:
                values = values.narrow(0, 0, total)
            kjt = KeyedJaggedTensor(
                keys=input_kjt_keys,
                values=values,
                lengths=lengths,
            )
            with torch.enable_grad():
                output = real_arch(kjt)

            flat: List[torch.Tensor] = []
            for name in feature_names:
                jt = output[name]
                flat.append(jt.values())
                flat.append(jt.lengths())
            # Return raw (grad-preserving) tensors; the install helper wraps
            # this callable with save-for-backward + detach.
            return tuple(flat)

        install_graph_break_dispatch(
            gm,
            _xsurface_sparse_prefix(self._override_id),
            fwd_fn,
            num_state_tensors=len(flat_state),
            num_ranks=num_ranks,
        )
        return gm, flat_state


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_sample_input(
    recblock: RecBlock,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
) -> Tuple[FeatureBundle, List[str], List[str]]:
    """Create a sample input FeatureBundle for RecBlock's preprocessor_stack.

    Returns (feature_bundle, ro_kjt_keys, nro_kjt_keys) so callers can
    match KJT keys between the sample and the batch.
    """
    preprocessor: MoTPreprocessor = (  # pyre-ignore[9]
        # pyrefly: ignore [bad-assignment]
        recblock.preprocessor_stack.mot_preprocessor
    )
    sample_bundle: FeatureBundle = preprocessor.sample_inputs(
        batch_size=batch_size,
        device=device,
        max_uih_seq_len=TEST_MAX_UIH_SEQ_LEN,
        num_candidates=TEST_NUM_CANDIDATES,
    )
    ro_id_list = sample_bundle.require_kjt(InputFeatures.UIH_ID_LIST)
    nro_id_list = sample_bundle.require_kjt(InputFeatures.CANDIDATE_ID_LIST)
    num_candidates = sample_bundle.require_tensor(InputFeatures.NUM_CANDIDATES)
    # pyrefly: ignore [bad-assignment]
    all_labels: Dict[str, torch.Tensor] = sample_bundle.require(
        InputFeatures.ALL_LABELS
    )
    return (
        _inputs_to_feature_bundle(
            ro_id_list, None, nro_id_list, num_candidates, all_labels
        ),
        ro_id_list.keys(),
        nro_id_list.keys(),
    )


def _verify_traced_graph(
    test_case: unittest.TestCase,
    gm: torch.fx.GraphModule,
) -> None:
    """Common verification for a traced FX graph."""
    test_case.assertIsNotNone(gm)
    test_case.assertIsNotNone(gm.graph)

    nodes = list(gm.graph.nodes)
    test_case.assertGreater(
        len(nodes), 0, "Traced graph should contain at least one node"
    )
    gm.graph.lint()
    gm.recompile()


def _build_overrides(
    recblock: RecBlock,
    sample_input: FeatureBundle,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
    num_processes: int = 1,
) -> List[Any]:
    """Build dispatcher instances for tracing, capturing sparse arch I/O via hooks.

    Runs the preprocessor + tokenizer eagerly once to capture the ECSparseArch
    input (via forward pre-hook) and output (via forward hook), then returns
    dispatcher instances (PreprocessorDispatcher + SparseArchDispatcher) for
    both the preprocessor and the sparse arch.  Each dispatcher exposes
    ``install_eager(model)`` and ``install_graph(gm, flat_state)``.
    """
    # pyre-ignore[16]: dynamic attr from setattr in TokenizerStack
    sparse_arch = recblock.tokenizer_stack.mot_sparse_arch._sparse_arch
    captured: Dict[str, Any] = {}
    input_captured: Dict[str, Any] = {}

    def pre_hook(
        _module: nn.Module,
        input_args: Tuple[Any, ...],
    ) -> None:
        input_captured["kjt"] = input_args[0]

    def hook(
        _module: nn.Module,
        _input: Any,
        output: Dict[str, JaggedTensor],
    ) -> None:
        captured["output"] = output

    pre_handle = sparse_arch.register_forward_pre_hook(pre_hook)
    handle = sparse_arch.register_forward_hook(hook)
    with torch.no_grad(), _disable_sampling(recblock.preprocessor_stack):
        preproc_output = recblock.preprocessor_stack(sample_input)
        try:
            recblock.tokenizer_stack(preproc_output)
        except (RuntimeError, AttributeError):
            # We only need the sparse arch I/O from the hooks.
            # Downstream errors (e.g., 0-dim MLP reshape) are expected
            # since some config features are disabled for testing.
            pass
    pre_handle.remove()
    handle.remove()
    assert "output" in captured, "Sparse arch forward hook did not fire"
    assert "kjt" in input_captured, "Sparse arch pre-hook did not fire"

    input_kjt_keys: List[str] = input_captured["kjt"].keys()

    # Pass the precomputed preprocessor output to the DummyPreprocessorStack
    # so it uses the SAME output that was used to capture sparse arch I/O.
    # Running the preprocessor twice can produce different outputs due to
    # module state mutation (e.g., running stats).
    return [
        PreprocessorDispatcher(
            device,
            override_id=0,
            batch_size=batch_size,
            sample_feature_bundle=sample_input,
            precomputed_output=preproc_output,
            num_processes=num_processes,
        ),
        SparseArchDispatcher(
            cached_output=captured["output"],
            input_kjt_keys=input_kjt_keys,
            override_id=1,
        ),
    ]


@contextlib.contextmanager
def _ec_sparse_arch_on(device: torch.device) -> Generator[None, None, None]:
    """Force ``ECSparseArch`` to build its embedding tables on ``device``.

    ``ECSparseArch.__init__`` defaults to ``torch.device("meta")`` when no
    ``device=`` is passed by the caller. Letting it stay on meta would force
    every test to call ``model.to_empty(device)`` afterwards to materialize
    the tables, which in turn blows away every parameter and non-persistent
    buffer in the model.

    Patching the default at construction time lets the entire model build
    correctly on CPU in one pass, so no post-hoc re-init is needed.
    """
    original_init: Callable[..., None] = ECSparseArch.__init__

    def patched_init(  # pyre-ignore[2,3]
        self: ECSparseArch, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault("device", device)
        original_init(self, *args, **kwargs)

    ECSparseArch.__init__ = patched_init  # pyre-ignore[8]
    try:
        yield
    finally:
        ECSparseArch.__init__ = original_init  # pyre-ignore[8]


def build_model_from_config(
    module_config: BlueReelsVDDModuleConfig,
    device: torch.device,
    bf16_training: bool = False,
    embeddings_on_meta: bool = False,
) -> BlueReelsVDDTrainV3:
    """Build the V3 xsurface model from a preconfigured ``module_config``.

    ``embeddings_on_meta``: when True, the ``ECSparseArch`` embedding tables are
    built on the ``meta`` device (no host allocation / init) while the dense
    modules stay on ``device``. Use this for the in_graph + on_device_init path,
    where ``InGraphEmbeddingBackend.setup()`` replaces the entire embedding
    collection with fresh on-device (meta -> HBM) tables, so the vanilla host
    tables are never read. Without this, an uncapped hash size (e.g. 100M rows)
    would allocate + ``normal_()``-init ~100GB per table on the host,
    single-threaded, stalling model construction before compile.
    """
    for stack in module_config.stu_stack_configs:
        for stu_config in stack.stu_configs:
            stu_config.attn_config = stu_config.attn_config._replace(
                use_cuda_attention=False
            )

    ec_device = torch.device("meta") if embeddings_on_meta else device
    with _ec_sparse_arch_on(ec_device):
        recblock = build_recblock_from_xsurface_config(
            module_config=module_config,
            is_train=True,
            custom_kernel=False,
        )
        model = BlueReelsVDDTrainV3(
            module_config=module_config,
            recblock=recblock,
            is_train=True,
        )
    if not embeddings_on_meta:
        model.to(device)
    # When embeddings_on_meta: the dense modules are already built on ``device``
    # (torch default) and the embedding tables are intentionally left on meta --
    # calling model.to(device) here would raise "Cannot copy out of meta tensor".
    # The in_graph backend materializes real on-device tables in setup().

    # Force pure PyTorch kernels (Triton kernels lack TPU support).
    model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    # Route the shared hammer.autocast() call sites to torchax AMP on this
    # TPU/JAX build (torch.autocast("cuda", ...) is inert without CUDA tensors).
    # Installed unconditionally here; the per-site enabled gate (_bf16_training)
    # is what controls whether bf16 actually engages.
    set_autocast_factory(hammer_autocast_factory)
    # bf16: default False keeps the historical fp32 TPU behavior. When True, the
    # HSTU L1/L2 blocks + composer + loss run bf16 (the explicit activation casts
    # plus hammer.autocast -> torchax AMP at the same sites the GPU uses).
    model.recursive_setattr("_bf16_training", bf16_training)
    # Zero out all remaining dropout. torchax's aten.bernoulli uses a PRNG
    # key stored in the Environment as mutable state; on the second jit call
    # the leaked tracer reference causes UnexpectedTracerError.
    model.recursive_setattr("_dropout_ratio", 0.0)
    model.recursive_setattr("_dropout_rate", 0.0)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    return model


def _build_full_model(
    device: torch.device,
    disable_seq_len_sampling: bool = False,
) -> Tuple[BlueReelsVDDTrainV3, BlueReelsVDDFeatureConfig]:
    """Build the full V3 xsurface model with test-sized configs.

    Thin wrapper over :func:`build_model_from_config`: constructs the
    module/feature configs via TestConfigProvider, trims STU stacks to
    a single layer to keep tests cheap, then delegates the model build
    + FakeTensor-tracing-safe wiring to ``build_model_from_config``.
    """
    config = TestConfigProvider().create_config()
    module_config = cast(BlueReelsVDDModuleConfig, config.module_config)
    feature_config = cast(BlueReelsVDDFeatureConfig, config.feature_config)
    if disable_seq_len_sampling:
        module_config.hstu_seq_len_sampling_configs = None
    # Trim each STU stack to a single layer to keep the test cheap.
    for stack in module_config.stu_stack_configs:
        stack.stu_configs = stack.stu_configs[:1]
    model = build_model_from_config(module_config, device)
    return model, feature_config


def _resize_nro_tensor(t: torch.Tensor, target_dim0: int) -> torch.Tensor:
    """Pad or trim tensor's first dimension to target_dim0."""
    if t.shape[0] == target_dim0:
        return t
    new_shape = (target_dim0, *t.shape[1:])
    out = torch.zeros(new_shape, dtype=t.dtype)
    copy_rows = min(t.shape[0], target_dim0)
    out[:copy_rows] = t[:copy_rows]
    return out


def _tile_dim0(t: torch.Tensor, target: int) -> torch.Tensor:
    """Cyclically tile *t* along dim 0 to length ``target`` (or trim)."""
    cur = t.shape[0]
    if cur == target:
        return t
    if cur > target:
        return t[:target].contiguous()
    if cur == 0:
        # Nothing to tile from — fall back to zeros.
        return torch.zeros(target, *t.shape[1:], dtype=t.dtype, device=t.device)
    n_full = target // cur
    n_extra = target - n_full * cur
    parts = [t] * n_full
    if n_extra > 0:
        parts.append(t[:n_extra])
    return torch.cat(parts, dim=0)


def _tile_last_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    """Cyclically tile *t* along the last dim to length ``target`` (or trim)."""
    cur = t.shape[-1]
    if cur == target:
        return t
    if cur > target:
        return t[..., :target].contiguous()
    if cur == 0:
        return torch.zeros(*t.shape[:-1], target, dtype=t.dtype, device=t.device)
    n_full = target // cur
    n_extra = target - n_full * cur
    parts = [t] * n_full
    if n_extra > 0:
        parts.append(t[..., :n_extra])
    return torch.cat(parts, dim=-1)


def resize_kjt_b(kjt: KeyedJaggedTensor, target_b: int) -> KeyedJaggedTensor:
    """Resize a per-key-major KJT to have ``target_b`` entries per key.

    Cyclically tiles each key's per-sample lengths and the corresponding
    per-sample value chunks.  Tiling (rather than zero-padding) keeps the
    synthesized batch "valid" enough to flow through the preprocessor
    and tokenizer in :func:`_build_overrides` — zero-padded entries
    short-circuit those paths and the sparse arch hook never fires.
    """
    keys = kjt.keys()
    K = len(keys)
    cur_lengths = kjt.lengths()
    cur_b = cur_lengths.shape[0] // K
    if cur_b == target_b:
        return kjt
    cur_values = kjt.values()

    new_lengths_parts: List[torch.Tensor] = []
    new_values_parts: List[torch.Tensor] = []
    cur_offset = 0

    for k in range(K):
        key_lengths = cur_lengths[k * cur_b : (k + 1) * cur_b]
        key_total = int(key_lengths.sum().item())
        key_values = cur_values[cur_offset : cur_offset + key_total]
        cur_offset += key_total

        if target_b < cur_b:
            tiled_lengths = key_lengths[:target_b]
            cum = torch.cat(
                [
                    torch.zeros(1, dtype=key_lengths.dtype, device=key_lengths.device),
                    key_lengths.cumsum(0),
                ]
            )
            tiled_values = key_values[: int(cum[target_b].item())]
        elif cur_b == 0:
            # Nothing to tile from — empty key segment.
            tiled_lengths = torch.zeros(
                target_b, dtype=key_lengths.dtype, device=key_lengths.device
            )
            tiled_values = torch.empty(
                0, dtype=key_values.dtype, device=key_values.device
            )
        else:
            n_full = target_b // cur_b
            n_extra = target_b - n_full * cur_b
            length_parts = [key_lengths] * n_full
            value_parts = [key_values] * n_full
            if n_extra > 0:
                length_parts.append(key_lengths[:n_extra])
                cum = torch.cat(
                    [
                        torch.zeros(
                            1, dtype=key_lengths.dtype, device=key_lengths.device
                        ),
                        key_lengths.cumsum(0),
                    ]
                )
                value_parts.append(key_values[: int(cum[n_extra].item())])
            tiled_lengths = torch.cat(length_parts)
            tiled_values = torch.cat(value_parts) if value_parts else key_values

        new_lengths_parts.append(tiled_lengths)
        new_values_parts.append(tiled_values)

    new_lengths = torch.cat(new_lengths_parts)
    new_values = (
        torch.cat(new_values_parts)
        if new_values_parts
        else torch.empty(0, dtype=cur_values.dtype, device=cur_values.device)
    )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=new_values,
        lengths=new_lengths,
    )


def resize_kjt_per_key_lengths(
    kjt: KeyedJaggedTensor,
    per_key_per_sample_budgets: Dict[str, int],
    batch_size: int,
) -> KeyedJaggedTensor:
    """Resize each key's per-sample length to its configured budget."""
    keys = kjt.keys()
    K = len(keys)
    cur_lengths = kjt.lengths()
    cur_values = kjt.values()
    assert cur_lengths.shape[0] == K * batch_size, (
        f"KJT lengths size {cur_lengths.shape[0]} != "
        f"num_keys ({K}) * batch_size ({batch_size})"
    )

    new_lengths_parts: List[torch.Tensor] = []
    new_values_parts: List[torch.Tensor] = []
    cur_offset = 0

    for k_idx, key in enumerate(keys):
        key_lengths = cur_lengths[k_idx * batch_size : (k_idx + 1) * batch_size]
        key_total = int(key_lengths.sum().item())
        key_values = cur_values[cur_offset : cur_offset + key_total]
        cur_offset += key_total

        if key not in per_key_per_sample_budgets:
            new_lengths_parts.append(key_lengths)
            new_values_parts.append(key_values)
            continue

        budget = int(per_key_per_sample_budgets[key])
        new_key_lengths = torch.full(
            (batch_size,),
            budget,
            dtype=key_lengths.dtype,
            device=key_lengths.device,
        )

        if budget == 0:
            new_key_values = torch.empty(
                0, dtype=key_values.dtype, device=key_values.device
            )
        elif key_total == 0:
            new_key_values = torch.zeros(
                (budget * batch_size,),
                dtype=key_values.dtype,
                device=key_values.device,
            )
        else:
            cum = torch.cat(
                [
                    torch.zeros(1, dtype=key_lengths.dtype, device=key_lengths.device),
                    key_lengths.cumsum(0),
                ]
            )
            sample_parts: List[torch.Tensor] = []
            for b in range(batch_size):
                start = int(cum[b].item())
                end = int(cum[b + 1].item())
                sample_data = key_values[start:end]
                sample_len = end - start
                if sample_len >= budget:
                    sample_parts.append(sample_data[:budget])
                elif sample_len == 0:
                    sample_parts.append(
                        torch.zeros(
                            (budget,),
                            dtype=key_values.dtype,
                            device=key_values.device,
                        )
                    )
                else:
                    n_full = budget // sample_len
                    n_extra = budget - n_full * sample_len
                    parts = [sample_data] * n_full
                    if n_extra > 0:
                        parts.append(sample_data[:n_extra])
                    sample_parts.append(torch.cat(parts))
            new_key_values = torch.cat(sample_parts)

        new_lengths_parts.append(new_key_lengths)
        new_values_parts.append(new_key_values)

    new_lengths = torch.cat(new_lengths_parts)
    new_values = (
        torch.cat(new_values_parts)
        if new_values_parts
        else torch.empty(0, dtype=cur_values.dtype, device=cur_values.device)
    )
    return KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=new_values,
        lengths=new_lengths,
    )


def resize_batch_inplace(
    batch: MTMLROOTrainBatch, target_b_ro: int, target_b_nro: int
) -> None:
    """Resize *batch* in place to ``(target_b_ro, target_b_nro)``.

    Synthesizes the override-build batch from a small trace-time batch
    so we don't need a second :class:`RandomDataPipe` call at large
    sizes (that path can SIGABRT inside the data pipe's C++ allocator).

    Strategy: cyclically tile each tensor / KJT to the target size so
    the synthesized batch contains real (repeated) samples — never zeros.
    Zero-padded entries (or a 0 in ``num_candidates``) short-circuit the
    tokenizer's call to the sparse arch in :func:`_build_overrides`, and
    the forward hook then never fires, leaving the cached output empty.

    ``num_candidates`` is distributed as ``target_b_nro // target_b_ro``
    floor with first ``remainder`` samples getting +1, so
    ``sum(num_candidates) == target_b_nro`` exactly -- mirrors PRT's
    ``pad_count_to_sum`` runtime behavior.
    """
    if batch.ro_float_features.shape[0] != target_b_ro:
        batch.ro_float_features = _tile_dim0(batch.ro_float_features, target_b_ro)
        batch.ro_id_list_features = resize_kjt_b(batch.ro_id_list_features, target_b_ro)

    if batch.nro_float_features.shape[0] != target_b_nro:
        batch.nro_float_features = _tile_dim0(batch.nro_float_features, target_b_nro)
        batch.nro_id_list_features = resize_kjt_b(
            batch.nro_id_list_features, target_b_nro
        )
        if batch.unary_features is not None:
            batch.unary_features = resize_kjt_b(batch.unary_features, target_b_nro)
        batch.supervision_label = _tile_last_dim(batch.supervision_label, target_b_nro)
        batch.supervision_weight = _tile_last_dim(
            batch.supervision_weight, target_b_nro
        )
        all_labels = batch.all_labels
        if all_labels is not None:
            for key in list(all_labels.keys()):
                all_labels[key] = _tile_dim0(all_labels[key], target_b_nro)

    per_sample = target_b_nro // target_b_ro
    remainder = target_b_nro - per_sample * target_b_ro
    nc = torch.full(
        (target_b_ro,),
        per_sample,
        dtype=batch.num_candidates.dtype,
        device=batch.num_candidates.device,
    )
    if remainder > 0:
        nc[:remainder] = nc[:remainder] + 1
    batch.num_candidates = nc


def _create_random_train_batch(
    feature_config: BlueReelsVDDFeatureConfig,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
    num_candidates: int = -1,
    pin_memory: bool = False,
) -> MTMLROOTrainBatch:
    """Generate a random MTMLROOTrainBatch via RandomROODataSourceConfig.

    All keys, shapes, and per-key lengths come from the feature config
    (``_RO_HISTORY_FEATURES``, ``_RO_CONTEXTUAL_FEATURES``,
    ``_NRO_FEATURES``).  ``randgen`` produces jagged num_candidates (up to
    ``num_candidates``) and variable RO history lengths when given a
    RecROORandGenContext.

    ``num_candidates`` overrides the per-request candidate count; when left at
    the ``-1`` sentinel it falls back to ``TEST_NUM_CANDIDATES`` (preserving the
    default used by the xsurface unit tests).

    ``pin_memory`` pins the generated host buffers for faster async H2D transfer.
    """
    B = batch_size
    C = num_candidates if num_candidates > 0 else TEST_NUM_CANDIDATES

    data_source_config = RandomROODataSourceConfig(
        batch_size=B,
        num_batches=1,
        num_generate=1,
        num_candidates=C,
        pin_memory=pin_memory,
    )
    data_pipe = RandomDataPipe(
        feature_config=feature_config,
        data_source_config=data_source_config,
    )
    batch = cast(MTMLROOTrainBatch, next(iter(data_pipe)))
    return batch.to(device, non_blocking=False)  # pyre-ignore


def _create_uniform_train_batch(
    feature_config: BlueReelsVDDFeatureConfig,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
) -> MTMLROOTrainBatch:
    """Generate a uniform MTMLROOTrainBatch via RandomDataSourceConfig.

    Uses the non-ROO data source so every sample has exactly
    ``feature_config.num_candidates_per_request`` candidates and
    ``feature_config.history_len`` history items, giving a deterministic
    ``B_NRO = batch_size * num_candidates_per_request``. Callers that need
    ``B_NRO`` to divide evenly across ``num_local`` (e.g., when there is no
    ``PaddingRegionTransform`` to align jagged totals) should use this
    instead of :func:`_create_random_train_batch`. The caller is responsible
    for mutating ``feature_config.num_candidates_per_request`` /
    ``history_len`` first if they need shapes other than the dataloader
    defaults.
    """
    data_pipe = RandomDataPipe(
        feature_config=feature_config,
        data_source_config=RandomDataSourceConfig(
            batch_size=batch_size,
            num_batches=1,
            num_generate=1,
            pin_memory=False,
        ),
    )
    batch = cast(MTMLROOTrainBatch, next(iter(data_pipe)))
    return batch.to(device, non_blocking=False)  # pyre-ignore
