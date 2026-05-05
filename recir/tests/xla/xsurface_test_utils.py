#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Shared test utilities for xsurface V3 model tracing and XLA compilation.

Contains dummy modules, override classes, model/batch builders, and helper
functions used by both the fake-tracing-compat test and the XLA compile test.
"""

import unittest
from typing import Any, cast, Dict, List, Tuple

import torch
import torch.nn as nn
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
from minimal_viable_ai.core.dataloader.dataloader_configs import RandomDataSourceConfig
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig
from minimal_viable_ai.core.ranking_common.configs import MTMLROOTrainBatch
from recir.api import Override
from recir.override_ops import anchor_input, anchor_output, replace_override_anchors
from torchrec import JaggedTensor, KeyedJaggedTensor


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

TEST_UC_RID_HASH_SIZE: int = 1_000_000
TEST_BATCH_SIZE: int = 4
TEST_MAX_UIH_SEQ_LEN: int = 32
TEST_NUM_CANDIDATES: int = 8


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
    input/output shapes. Uses anchor_input on inputs and anchor_output on
    outputs so replace_override_anchors can wire up runtime dispatch to
    the real PreprocessorStack.

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
            all_labels: Dict[str, torch.Tensor] = sample_bundle.require(
                InputFeatures.ALL_LABELS
            )
            input_features = _inputs_to_feature_bundle(
                ro_id_list, None, nro_id_list, num_cand, all_labels
            )
        if precomputed_output is not None:
            cached_output = precomputed_output
        else:
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

        # Mark input tensor leaves with anchor_input. Iterate the actual
        # features bundle (not the sample-captured lists) so the ordering
        # matches what the pre-hook captures at trace time. This ensures
        # the anchor indices align with fwd_fn's reconstruction order.
        input_idx = 0
        anchored_inputs: List[torch.Tensor] = []
        for desc in features:
            val = features.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                anchored_inputs.append(anchor_input(val.values(), oid, input_idx))
                input_idx += 1
                anchored_inputs.append(anchor_input(val.lengths(), oid, input_idx))
                input_idx += 1
            elif isinstance(val, torch.Tensor):
                anchored_inputs.append(anchor_input(val, oid, input_idx))
                input_idx += 1
            elif isinstance(val, dict):
                for key in sorted(val.keys()):
                    if isinstance(val[key], torch.Tensor):
                        anchored_inputs.append(anchor_input(val[key], oid, input_idx))
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
            anchor_output(t, oid, idx, deps=deps)
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

    Uses anchor_input on the input KJT components and anchor_output on
    outputs so replace_override_anchors can wire up runtime dispatch to
    the real ECSparseArch.

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

        # Anchor the input KJT components so SparseArchOverride.transform can
        # rediscover them via replace_override_anchors and use them as
        # dispatch inputs (instead of falling back to all batch placeholders).
        m_v = anchor_input(id_list_features.values(), oid, 0)
        m_l = anchor_input(id_list_features.lengths(), oid, 1)

        deps: List[torch.Tensor] = [m_v, m_l, self._grad_anchor]

        out_idx = 0
        result: Dict[str, JaggedTensor] = {}
        for i, name in enumerate(self._feature_names):
            vbuf = getattr(self, f"_jt_values_{i}")
            lbuf = getattr(self, f"_jt_lengths_{i}")
            v_z = torch.zeros_like(vbuf)
            l_z = lbuf.clone()
            v = anchor_output(v_z, oid, out_idx, deps=deps)
            l = anchor_output(l_z, oid, out_idx + 1, deps=deps)
            result[name] = JaggedTensor(values=v, lengths=l)
            out_idx += 2
        return result


# ---------------------------------------------------------------------------
# Override classes
# ---------------------------------------------------------------------------


class PreprocessorOverride(Override):
    """Override that replaces PreprocessorStack with DummyPreprocessorStack.

    replacement() swaps the module before tracing.
    transform() wires up runtime dispatch to the real PreprocessorStack
    via replace_override_anchors, using input_anchor nodes to identify
    the correct dispatch inputs.

    The input FeatureBundle structure is captured at trace time via a
    forward_pre_hook on the dummy, ensuring the captured structure matches
    what RecBlock actually passes (including supervision_weight and the
    correct KJT keys from the batch, not the sample).
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

    def replacement(self, fqn: str, orig_submodule: nn.Module) -> nn.Module:
        if isinstance(orig_submodule, PreprocessorStack):
            self._orig_stack = orig_submodule
            self._dummy = DummyPreprocessorStack(
                real_stack=orig_submodule,
                device=self._device,
                batch_size=self._batch_size,
                max_uih_seq_len=self._max_uih_seq_len,
                num_candidates=self._num_candidates,
                override_id=self._override_id,
                sample_feature_bundle=self._sample_feature_bundle,
                precomputed_output=self._precomputed_output,
            )
            return self._dummy  # pyre-ignore[7]
        return orig_submodule

    def transform(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> torch.fx.GraphModule:
        if self._orig_stack is None or self._dummy is None:
            return gm

        real_stack: nn.Module = self._orig_stack  # pyre-ignore[8]
        dummy: DummyPreprocessorStack = self._dummy  # pyre-ignore[8]
        num_ranks: int = self._num_ranks

        # Input structure captured at construction time from sample_feature_bundle.
        entries: List[Tuple[FeatureDescriptor, str, Any]] = dummy._input_entries
        input_nontensor_map: Dict[FeatureDescriptor, Any] = dummy._input_nontensor_map

        output_tensor_descs: List[FeatureDescriptor] = dummy._tensor_descs
        output_kjt_descs: List[FeatureDescriptor] = dummy._kjt_descs

        state: Dict[str, Any] = {"saved": None}

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

            flat: List[torch.Tensor] = []
            tensor_i = 0
            for desc in output_tensor_descs:
                val = output.get(desc)
                if val is not None and isinstance(val, torch.Tensor):
                    flat.append(val)
                else:
                    flat.append(getattr(dummy, f"_tensor_{tensor_i}"))
                tensor_i += 1
            kjt_i = 0
            for desc in output_kjt_descs:
                val = output.get(desc)
                if val is not None and isinstance(val, KeyedJaggedTensor):
                    flat.append(val.values())
                    flat.append(val.lengths())
                else:
                    flat.append(getattr(dummy, f"_kjt_values_{kjt_i}"))
                    flat.append(getattr(dummy, f"_kjt_lengths_{kjt_i}"))
                kjt_i += 1

            state["saved"] = [t for t in flat if t.requires_grad]
            return tuple(t.detach() for t in flat)

        def bwd_fn(grad: torch.Tensor, grad_idx: int = 0) -> None:
            saved = state["saved"]
            if saved is not None and grad_idx < len(saved):
                s = saved[grad_idx]
                is_last = grad_idx == len(saved) - 1
                torch.autograd.backward(
                    [s], [grad / num_ranks], retain_graph=not is_last
                )
                if is_last:
                    state["saved"] = None

        return replace_override_anchors(
            gm,
            override_id=self._override_id,
            fwd_fn=fwd_fn,
            bwd_fn=bwd_fn,
            num_state_tensors=len(flat_state),
        )


class SparseArchOverride(Override):
    """Override that replaces ECSparseArch with DummySparseArch.

    replacement() swaps the module before tracing.
    transform() wires up runtime dispatch to the real ECSparseArch
    via replace_override_anchors, using input_anchor nodes to identify
    the correct dispatch inputs (KJT values and lengths).
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

    def replacement(self, fqn: str, orig_submodule: nn.Module) -> nn.Module:
        if isinstance(orig_submodule, ECSparseArch):
            self._orig_sparse_arch = orig_submodule
            self._dummy = DummySparseArch(
                self._cached_output,
                input_kjt_keys=self._input_kjt_keys,
                override_id=self._override_id,
            )
            return self._dummy
        return orig_submodule

    def transform(
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> torch.fx.GraphModule:
        if self._orig_sparse_arch is None or self._dummy is None:
            return gm

        real_arch: nn.Module = self._orig_sparse_arch
        input_kjt_keys: List[str] = self._input_kjt_keys
        feature_names: List[str] = self._dummy._feature_names
        num_ranks: int = self._num_ranks

        state: Dict[str, Any] = {"saved": None}

        def fwd_fn(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            kjt = KeyedJaggedTensor(
                keys=input_kjt_keys,
                values=input_tensors[0],
                lengths=input_tensors[1],
            )
            with torch.enable_grad():
                output = real_arch(kjt)

            flat: List[torch.Tensor] = []
            for name in feature_names:
                jt = output[name]
                flat.append(jt.values())
                flat.append(jt.lengths())

            state["saved"] = [t for t in flat if t.requires_grad]
            return tuple(t.detach() for t in flat)

        def bwd_fn(grad: torch.Tensor, grad_idx: int = 0) -> None:
            saved = state["saved"]
            if saved is not None and grad_idx < len(saved):
                s = saved[grad_idx]
                is_last = grad_idx == len(saved) - 1
                torch.autograd.backward(
                    [s], [grad / num_ranks], retain_graph=not is_last
                )
                if is_last:
                    state["saved"] = None

        return replace_override_anchors(
            gm,
            override_id=self._override_id,
            fwd_fn=fwd_fn,
            bwd_fn=bwd_fn,
            num_state_tensors=len(flat_state),
        )


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
) -> List[Override]:
    """Build Override instances for tracing, capturing sparse arch I/O via hooks.

    Runs the preprocessor + tokenizer eagerly once to capture the ECSparseArch
    input (via forward pre-hook) and output (via forward hook), then returns
    overrides for both the preprocessor and the sparse arch.
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
    with torch.no_grad():
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
        PreprocessorOverride(
            device,
            override_id=0,
            batch_size=batch_size,
            sample_feature_bundle=sample_input,
            precomputed_output=preproc_output,
        ),
        SparseArchOverride(
            cached_output=captured["output"],
            input_kjt_keys=input_kjt_keys,
            override_id=1,
        ),
    ]


def _build_full_model(
    device: torch.device,
) -> Tuple[BlueReelsVDDTrainV3, BlueReelsVDDFeatureConfig]:
    """Build the full V3 xsurface model with test-sized configs.

    The post-refactor model class takes a pre-built ``RecBlock`` and a
    duck-typed ``module_config``. We construct the ``RecBlock`` externally via
    ``build_recblock_from_xsurface_config`` from ``udd_fm_train_v3``.
    """
    config_provider = TestConfigProvider()
    config = config_provider.create_config()
    module_config = cast(BlueReelsVDDModuleConfig, config.module_config)
    feature_config = cast(BlueReelsVDDFeatureConfig, config.feature_config)

    # Disable UIH sequence-length sampling. The config field is
    # ``hstu_seq_len_sampling_configs`` (with the hstu_ prefix); setting
    # ``seq_len_sampling_configs`` would silently create a new attribute
    # and leave the real one at its TS_WEIGHTED default, which randomly
    # drops tokens at preprocess time and produces non-deterministic
    # post-preprocessor jagged totals (breaks the GSPMD shard-divisibility
    # check on override outputs in the XLA compile path).
    module_config.hstu_seq_len_sampling_configs = None
    # Trim each STU stack to a single layer to keep the test cheap.
    # Also disable ``use_cuda_attention`` on every attn config so the kernel
    # selection in ``hstu_cross_mha`` falls back to the pure-PyTorch path.
    # The CUDA path calls ``torch.ops.hstu.hstu_mha_{fwd,bwd}`` which lacks
    # Meta kernel registration and breaks make_fx FakeTensor tracing.
    for stack in module_config.stu_stack_configs:
        stack.stu_configs = stack.stu_configs[:1]
        for stu_config in stack.stu_configs:
            # ``STUSelfAttnConfig`` is NamedTuple-based; use ``_replace``.
            stu_config.attn_config = stu_config.attn_config._replace(
                use_cuda_attention=False
            )

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
    # Force pure PyTorch kernels (Triton kernels lack FakeTensor support).
    model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)

    # Disable bf16 training: XLA's GSPMD collective-permute disallows
    # mixed precision, and the model's explicit .to(bf16) casts create
    # bf16 intermediates alongside f32 ECSparseArch outputs.
    model.recursive_setattr("_bf16_training", False)

    # Zero out all remaining dropout. torchax's aten.bernoulli uses a PRNG
    # key stored in the Environment as mutable state; on the second jit call
    # the leaked tracer reference causes UnexpectedTracerError.
    model.recursive_setattr("_dropout_ratio", 0.0)
    model.recursive_setattr("_dropout_rate", 0.0)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    # The post-refactor ``UnaryEmbeddingBagBatched`` is already a pure PyTorch
    # implementation (see hammer/v3/models/udd_fm_v3/modules.py), so the
    # legacy fbgemm -> pytorch swap is no longer needed.

    model.to_empty(device=device)
    # Use f32 for all floating-point params/buffers.  XLA's GSPMD
    # collective-permute disallows mixed precision (f32 + bf16), and
    # ECSparseArch's EmbeddingCollection calls .float() on lookup results,
    # so using bf16 params creates a dtype mismatch.
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(torch.float32)
    for _buf_name, buf in model.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(torch.float32)
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


def _create_random_train_batch(
    feature_config: BlueReelsVDDFeatureConfig,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
    ro_kjt_keys: List[str] | None = None,
    nro_kjt_keys: List[str] | None = None,
) -> MTMLROOTrainBatch:
    """Generate a random MTMLROOTrainBatch with uniform num_candidates.

    RandomDataPipe produces jagged num_candidates per sample (via
    torch.randint), but the DummyPreprocessorStack caches outputs with
    uniform num_candidates=TEST_NUM_CANDIDATES. To avoid dim0 mismatches
    between nro_float_features and RecBlock's user_embeddings, we
    regenerate the batch with uniform candidate counts.
    """
    B = batch_size
    C = TEST_NUM_CANDIDATES
    B_NRO = B * C

    # TODO: Enable and test ROO data
    # data_source_config = RandomROODataSourceConfig(
    #     batch_size=B,
    #     num_batches=1,
    #     num_generate=1,
    #     num_candidates=C,
    #     pin_memory=False,
    # )
    data_source_config = RandomDataSourceConfig(
        batch_size=B,
        num_generate=1,
    )
    data_pipe = RandomDataPipe(
        feature_config=feature_config,
        data_source_config=data_source_config,
    )
    batch = cast(MTMLROOTrainBatch, next(iter(data_pipe)))

    # Fix num_candidates to be uniform (RandomDataPipe generates jagged).
    batch.num_candidates = torch.full((B,), C, dtype=batch.num_candidates.dtype)

    # Rebuild ro_id_list_features KJT with uniform UIH lengths.
    ro_kjt = batch.ro_id_list_features
    ro_keys = ro_kjt_keys if ro_kjt_keys is not None else ro_kjt.keys()
    num_ro_keys = len(ro_keys)
    batch.ro_id_list_features = KeyedJaggedTensor(
        keys=ro_keys,
        values=torch.randint(
            0,
            50,
            (B * TEST_MAX_UIH_SEQ_LEN * num_ro_keys,),
            dtype=ro_kjt.values().dtype,
        ),
        lengths=torch.full((B * num_ro_keys,), TEST_MAX_UIH_SEQ_LEN, dtype=torch.int64),
    )

    # Same for NRO KJT — use only the model's expected keys.
    nro_kjt = batch.nro_id_list_features
    nro_keys = nro_kjt_keys if nro_kjt_keys is not None else nro_kjt.keys()
    num_nro_keys = len(nro_keys)
    batch.nro_id_list_features = KeyedJaggedTensor(
        keys=nro_keys,
        values=torch.randint(
            0, 50, (B * C * num_nro_keys,), dtype=nro_kjt.values().dtype
        ),
        lengths=torch.full((B * num_nro_keys,), C, dtype=torch.int64),
    )

    # Rebuild all_labels to match sample_inputs shapes.
    UIH = TEST_MAX_UIH_SEQ_LEN
    all_labels = batch.all_labels
    assert all_labels is not None
    for key in list(all_labels.keys()):
        all_labels[key] = torch.randint(0, 256, (B * UIH,), dtype=all_labels[key].dtype)

    # Resize NRO-side tensors to match B_NRO = B * C.
    batch.nro_float_features = _resize_nro_tensor(batch.nro_float_features, B_NRO)

    # Rebuild unary_features KJT to B_NRO entries.
    if batch.unary_features is not None:
        unary_kjt = batch.unary_features
        num_keys = len(unary_kjt.keys())
        total_values = B_NRO * num_keys
        batch.unary_features = KeyedJaggedTensor(
            keys=unary_kjt.keys(),
            values=torch.randint(0, 100, (total_values,)),
            lengths=torch.ones(total_values, dtype=torch.int64),
        )

    # supervision_label / supervision_weight are (T, B_NRO): resize the LAST
    # dim (B_NRO), not dim 0 (T = num_main_tasks). _resize_nro_tensor would
    # mangle dim 0, producing a (B_NRO, original_B_NRO) tensor and breaking
    # the [:cls_end] slice in BCE loss.
    for attr in ("supervision_label", "supervision_weight"):
        t = getattr(batch, attr)
        if t.dim() >= 1 and t.shape[-1] != B_NRO:
            new_t = torch.zeros(*t.shape[:-1], B_NRO, dtype=t.dtype)
            copy_cols = min(t.shape[-1], B_NRO)
            new_t[..., :copy_cols] = t[..., :copy_cols]
            setattr(batch, attr, new_t)

    return batch.to(device, non_blocking=False)  # pyre-ignore
