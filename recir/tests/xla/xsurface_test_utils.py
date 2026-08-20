#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Test-only fixtures for xsurface V3 tracing and XLA compilation."""

import unittest
from typing import cast, Dict, List, Tuple

import torch
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import (
    BlueReelsVDDModuleConfig,
    ConfigProvider,
)
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.core.features.bundle import FeatureBundle
from hammer.v3.modules.core.features.input_features import InputFeatures
from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
from hammer.v3.modules.recblock import _inputs_to_feature_bundle, RecBlock
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig
from recir.examples.rankfm_mvai.rankfm_dispatchers import (
    TEST_BATCH_SIZE,
    TEST_MAX_UIH_SEQ_LEN,
    TEST_NUM_CANDIDATES,
)
from recir.examples.rankfm_mvai.rankfm_model_utils import build_model_from_config
# ---------------------------------------------------------------------------
# Config provider
# ---------------------------------------------------------------------------

TEST_UC_RID_HASH_SIZE: int = 1_000_000


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
# Helper functions
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
