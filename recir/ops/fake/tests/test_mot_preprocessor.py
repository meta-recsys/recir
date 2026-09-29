#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from typing import cast
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F
from hammer.v3.modules.core.features.bundle import FeatureBundle
from hammer.v3.modules.core.features.descriptor import FeatureDescriptor
from hammer.v3.modules.core.features.input_features import InputFeatures
from hammer.v3.modules.core.features.types import FeatureType
from hammer.v3.modules.mot.features import InputActionFeatures, MetadataFeatures
from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.ops.fake.mot_preprocessor import (
    patch_mot_preprocessor_for_tracing,
    TraceableMoTPreprocessor,
)
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


_HIGH_DIM = FeatureDescriptor(
    name="test.high_dim",
    feature_type=FeatureType.METADATA,
    dtype=torch.float32,
)
_NRO_PADDING_MASK = FeatureDescriptor(
    name="test.nro_padding_mask",
    feature_type=FeatureType.METADATA,
    dtype=torch.float32,
)


class _StubMoTPreprocessor(MoTPreprocessor):
    def __init__(self, emit_high_dim: bool = False) -> None:
        nn.Module.__init__(self)
        self._emit_high_dim = emit_high_dim

    def forward(self, features: FeatureBundle) -> FeatureBundle:
        output = features.clone()
        ro_batch = features.require_kjt(InputFeatures.UIH_ID_LIST).stride()
        nro_batch = features.require_kjt(InputFeatures.CANDIDATE_ID_LIST).stride()
        output.set_tensor(
            MetadataFeatures.SEQ_LENGTHS[0],
            torch.ones(ro_batch, dtype=torch.int64),
            validate=False,
        )
        output.set_tensor(
            _NRO_PADDING_MASK,
            torch.ones(nro_batch, dtype=torch.float32),
            validate=False,
        )
        output.set_tensor(
            InputActionFeatures.QUERY_TIME[0],
            torch.ones(nro_batch, dtype=torch.int64),
            validate=False,
        )
        output.set_nontensor(MetadataFeatures.TOTAL_NUM_TARGETS, nro_batch)
        # A whole-batch reduction with keepdim=True: shape (1,) at every batch.
        output.set_tensor(
            MetadataFeatures.MERGED_RUNTIME_MAX_SEQ_LEN,
            torch.ones(1, dtype=torch.int32),
            validate=False,
        )
        output.set_export("metric", torch.tensor(1.0))
        supervision_weight = features.require_tensor(InputFeatures.SUPERVISION_WEIGHT)
        output.set_export("unsampled_inputs", torch.ones_like(supervision_weight[0]))
        if self._emit_high_dim:
            output.set_tensor(
                _HIGH_DIM,
                torch.ones(ro_batch, 2, 3),
                validate=False,
            )
        return output


class _Model(nn.Module):
    def __init__(self, preprocessor: MoTPreprocessor) -> None:
        super().__init__()
        self.preprocessor = preprocessor


class _RawAdapter(nn.Module):
    def __init__(self, preprocessor: TraceableMoTPreprocessor) -> None:
        super().__init__()
        self.preprocessor = preprocessor

    def forward(
        self,
        ro_values: torch.Tensor,
        ro_lengths: torch.Tensor,
        nro_values: torch.Tensor,
        nro_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
        supervision_weight: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        label = torch.ones_like(supervision_weight[0])
        bundle = _inputs_to_feature_bundle(
            KeyedJaggedTensor(keys=["ro"], values=ro_values, lengths=ro_lengths),
            None,
            KeyedJaggedTensor(keys=["nro"], values=nro_values, lengths=nro_lengths),
            num_candidates,
            {"label": label},
            supervision_weight=supervision_weight,
        )
        output = self.preprocessor(bundle)
        nro_mask = output.require_tensor(_NRO_PADDING_MASK)
        return (
            output.require_tensor(MetadataFeatures.SEQ_LENGTHS[0]),
            nro_mask,
            output.require_tensor(InputActionFeatures.QUERY_TIME[0]),
            output.get_export("metric"),
            output.get_export("unsampled_inputs"),
            output.require(InputFeatures.ALL_LABELS)["label"],
            output.require_tensor(MetadataFeatures.MERGED_RUNTIME_MAX_SEQ_LEN),
        )


class _NROConsumerAdapter(nn.Module):
    def __init__(self, preprocessor: TraceableMoTPreprocessor) -> None:
        super().__init__()
        self._raw_adapter = _RawAdapter(preprocessor)

    def forward(
        self,
        ro_values: torch.Tensor,
        ro_lengths: torch.Tensor,
        nro_values: torch.Tensor,
        nro_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
        supervision_weight: torch.Tensor,
    ) -> torch.Tensor:
        _, nro_mask, _, _, _, _, _ = self._raw_adapter(
            ro_values,
            ro_lengths,
            nro_values,
            nro_lengths,
            num_candidates,
            supervision_weight,
        )
        return F.dropout(nro_mask, p=0.5, training=True)


def _make_bundle(batch_size: int, nro_batch: int) -> FeatureBundle:
    ro_kjt = KeyedJaggedTensor(
        keys=["ro"],
        values=torch.arange(batch_size),
        lengths=torch.ones(batch_size, dtype=torch.int64),
    )
    nro_kjt = KeyedJaggedTensor(
        keys=["nro"],
        values=torch.arange(nro_batch),
        lengths=torch.ones(nro_batch, dtype=torch.int64),
    )
    num_candidates = torch.full(
        (batch_size,), nro_batch // batch_size, dtype=torch.int32
    )
    return _inputs_to_feature_bundle(
        ro_kjt,
        None,
        nro_kjt,
        num_candidates,
        {"label": torch.ones(nro_batch)},
        supervision_weight=torch.ones(1, nro_batch),
    )


def _raw_args(bundle: FeatureBundle) -> tuple[torch.Tensor, ...]:
    ro_kjt = bundle.require_kjt(InputFeatures.UIH_ID_LIST)
    nro_kjt = bundle.require_kjt(InputFeatures.CANDIDATE_ID_LIST)
    return (
        ro_kjt.values(),
        ro_kjt.lengths(),
        nro_kjt.values(),
        nro_kjt.lengths(),
        bundle.require_tensor(InputFeatures.NUM_CANDIDATES),
        bundle.require_tensor(InputFeatures.SUPERVISION_WEIGHT),
    )


def _dynamic_fake_args(
    args: tuple[torch.Tensor, ...],
) -> tuple[FakeTensorMode, tuple[torch.Tensor, ...]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    fake_args = []
    for index, arg in enumerate(args):
        dynamic_sizes = [DimDynamic.STATIC, DimDynamic.DYNAMIC]
        if index != 5:
            dynamic_sizes = [DimDynamic.DYNAMIC] + [DimDynamic.STATIC] * (arg.ndim - 1)
        symbolic_context = StatelessSymbolicContext(dynamic_sizes=dynamic_sizes)
        fake_args.append(
            fake_mode.from_tensor(
                arg,
                static_shapes=False,
                source=LocalSource(f"test_input_{index}"),
                symbolic_context=symbolic_context,
            )
        )
    return fake_mode, tuple(fake_args)


def _static_fake_args(
    args: tuple[torch.Tensor, ...],
) -> tuple[FakeTensorMode, tuple[torch.Tensor, ...]]:
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv())
    return fake_mode, tuple(
        fake_mode.from_tensor(arg, static_shapes=True) for arg in args
    )


class TraceableMoTPreprocessorTest(TestCase):
    def test_custom_op_fake_contract(self) -> None:
        sample = _make_bundle(batch_size=2, nro_batch=4)
        model = _Model(_StubMoTPreprocessor())
        self.assertEqual(patch_mot_preprocessor_for_tracing(model, sample), 1)
        wrapper = cast(TraceableMoTPreprocessor, model.preprocessor)

        # Dynamic opcheck perturbs relational KJT tensors independently. The
        # next test covers dynamic execution with valid batch relationships.
        result = torch.library.opcheck(
            torch.ops.recir_preproc.mot_preprocessor_forward.default,
            (wrapper._instance_id, wrapper._decompose_input(sample)),
            test_utils=(
                "test_schema",
                "test_faketensor",
            ),
        )

        self.assertEqual(
            result,
            {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"},
        )

    def test_trace_does_not_confuse_equal_ro_and_nro_sample_batches(self) -> None:
        sample = _make_bundle(batch_size=2, nro_batch=2)
        model = _Model(_StubMoTPreprocessor())
        self.assertEqual(patch_mot_preprocessor_for_tracing(model, sample), 1)
        adapter = _RawAdapter(cast(TraceableMoTPreprocessor, model.preprocessor))
        fake_mode, fake_args = _dynamic_fake_args(_raw_args(sample))
        with fake_mode:
            traced = make_fx(
                adapter,
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*fake_args)

        runtime = _make_bundle(batch_size=3, nro_batch=6)
        (
            seq_lengths,
            nro_mask,
            query_time,
            metric,
            unsampled,
            label,
            merged_max_seq_len,
        ) = traced(*_raw_args(runtime))

        self.assertEqual(seq_lengths.shape, (3,))
        self.assertEqual(nro_mask.shape, (6,))
        self.assertEqual(query_time.shape, (6,))
        self.assertEqual(metric.shape, ())
        self.assertEqual(unsampled.shape, (6,))
        self.assertEqual(label.shape, (6,))
        # Traced at batch 2, run at batch 3: a batch-shaped policy would have
        # baked (2,) in here.
        self.assertEqual(merged_max_seq_len.shape, (1,))

    def test_trace_allows_downstream_ops_on_dynamic_nro_outputs(self) -> None:
        sample = _make_bundle(batch_size=2, nro_batch=4)
        model = _Model(_StubMoTPreprocessor())
        self.assertEqual(patch_mot_preprocessor_for_tracing(model, sample), 1)
        adapter = _NROConsumerAdapter(
            cast(TraceableMoTPreprocessor, model.preprocessor)
        )
        fake_mode, fake_args = _dynamic_fake_args(_raw_args(sample))
        with fake_mode:
            traced = make_fx(
                adapter,
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*fake_args)

        runtime = _make_bundle(batch_size=3, nro_batch=6)
        self.assertEqual(traced(*_raw_args(runtime)).shape, (6,))

    def test_trace_preserves_data_dependent_nro_batch_from_static_sample(
        self,
    ) -> None:
        sample = _make_bundle(batch_size=2, nro_batch=4)
        model = _Model(_StubMoTPreprocessor())
        self.assertEqual(patch_mot_preprocessor_for_tracing(model, sample), 1)
        adapter = _RawAdapter(cast(TraceableMoTPreprocessor, model.preprocessor))
        fake_mode, fake_args = _static_fake_args(_raw_args(sample))
        with fake_mode:
            traced = make_fx(
                adapter,
                tracing_mode="fake",
                _allow_non_fake_inputs=False,
            )(*fake_args)

        runtime = _make_bundle(batch_size=2, nro_batch=6)
        _, nro_mask, query_time, _, unsampled, label, _ = traced(*_raw_args(runtime))

        self.assertEqual(nro_mask.shape, (6,))
        self.assertEqual(query_time.shape, (6,))
        self.assertEqual(unsampled.shape, (6,))
        self.assertEqual(label.shape, (6,))

    def test_query_time_uses_nro_dense_policy(self) -> None:
        desc = FeatureDescriptor(
            name="hstu_v3.seq_0/query_time",
            feature_type=FeatureType.PAYLOAD,
            dtype=torch.int64,
        )

        spec = TraceableMoTPreprocessor._one_dim_output_spec(
            desc, desc.name, torch.int64
        )

        self.assertEqual(spec[0], "nro_dense_1d")

    def test_num_targets_uses_batch_dense_policy(self) -> None:
        desc = FeatureDescriptor(
            name="hstu_v3.seq_0/num_targets",
            feature_type=FeatureType.METADATA,
            dtype=torch.int64,
        )

        spec = TraceableMoTPreprocessor._one_dim_output_spec(
            desc, desc.name, torch.int64
        )

        self.assertEqual(spec[0], "batch_dense_1d")

    def test_merged_runtime_max_seq_len_uses_singleton_policy(self) -> None:
        desc = MetadataFeatures.MERGED_RUNTIME_MAX_SEQ_LEN

        spec = TraceableMoTPreprocessor._one_dim_output_spec(
            desc, desc.name, torch.int32
        )

        self.assertEqual(spec[0], "singleton_1d")

    def test_high_dim_output_requires_explicit_policy(self) -> None:
        model = _Model(_StubMoTPreprocessor(emit_high_dim=True))

        with self.assertRaisesRegex(NotImplementedError, "explicit shape policy"):
            patch_mot_preprocessor_for_tracing(
                model, _make_bundle(batch_size=2, nro_batch=4)
            )
