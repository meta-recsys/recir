#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CLI + config builders for the mvai_cloud-driven RankFM TPU example.

This mirrors recir/examples/rankfm/config.py (the bespoke-loop config),
reusing its CLI dataclasses (RankFMTrainerConfig + RankFMModelArgs),
parse_cli, and build_module_config verbatim by import. On top of that it
produces the two mvai_cloud config objects that RecTrainer consumes:

* :func:`build_model_family_config` -- a
  ``minimal_viable_ai.core.model_family_api.configs.RecModelFamilyConfig``
  built from the hammer udd_fm ``ConfigProvider().create_config()`` with the
  RankFMModelArgs mutations applied (via the shared ``build_module_config``).
* :func:`build_trainer_config` -- a
  ``minimal_viable_ai.core.trainer.trainer_config_def.RecTrainerConfig`` with
  ``SystemConfig(use_gpu=False)`` so ``RecTrainer.__init__`` takes its
  CPU/gloo device path; recir lowers the dense graph to TPU via
  ``jax.devices()`` regardless of the torch device.

The RankFM* CLI dataclasses are NOT redefined here -- they are re-exported
from the sibling rankfm package so the two examples stay byte-identical on
the model/trainer knobs.
"""

from __future__ import annotations

from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.udd_fm_config import (
    BlueReelsVDDModuleConfig,
    ConfigProvider,
)
from minimal_viable_ai.core.dataloader.dataloader_configs import RandomDataSourceConfig
from minimal_viable_ai.core.model_family_api.configs import RecModelFamilyConfig
from minimal_viable_ai.core.trainer.trainer_config_def import (
    MetricsConfig,
    RecTrainerConfig,
    SystemConfig,
)

# Reuse the rankfm CLI surface verbatim (do NOT copy): same dataclasses,
# same parse_cli, same module-config mutation logic. The two examples must
# agree exactly on model identity and trainer knobs.
from recir.examples.rankfm.config import (  # noqa: F401
    build_module_config,
    parse_cli,
    RankFMModelArgs,
    RankFMTrainerConfig,
)

__all__ = [
    "RankFMTrainerConfig",
    "RankFMModelArgs",
    "parse_cli",
    "build_module_config",
    "build_model_family_config",
    "build_trainer_config",
]


# ---------------------------------------------------------------------------
# mvai_cloud RecModelFamilyConfig builder
# ---------------------------------------------------------------------------


def build_model_family_config(
    trainer_cfg: RankFMTrainerConfig,
    model_args: RankFMModelArgs,
) -> RecModelFamilyConfig:
    """CLI args -> a RecModelFamilyConfig carrying the mutated module/feature config.

    Starts from the hammer udd_fm ``ConfigProvider().create_config()`` (so the
    optimizer_config, qcomms_config, and embedding_tables defaults match the
    canonical udd_fm RecTrainer entrypoint), then overwrites module_config and
    feature_config with the RankFMModelArgs-mutated pair produced by the shared
    ``build_module_config``. The mutated module_config is what
    BlueReelsVDDModelFamily.create_vanilla_train_module reads to build the
    byte-identical BlueReelsVDDTrainV3, and the mutated feature_config is what
    the (reused) rankfm data path consumes.
    """
    base_config: RecModelFamilyConfig = ConfigProvider().create_config()

    module_config: BlueReelsVDDModuleConfig
    feature_config: BlueReelsVDDFeatureConfig
    module_config, feature_config = build_module_config(
        model_args,
        enable_padding=trainer_cfg.enable_padding,
    )

    base_config.module_config = module_config
    base_config.feature_config = feature_config
    # HSTU owns its own EmbeddingCollection (recir backend handles sharding),
    # so leave the legacy single-sharder path (use_distributed_sharding=False,
    # embedding_tables={}) exactly as ConfigProvider().create_config() sets it.
    return base_config


# ---------------------------------------------------------------------------
# mvai_cloud RecTrainerConfig builder
# ---------------------------------------------------------------------------


def build_trainer_config(
    trainer_cfg: RankFMTrainerConfig,
) -> RecTrainerConfig:
    """CLI trainer knobs -> a RecTrainerConfig for the TPU/recir path.

    Key choices:
      * ``SystemConfig(use_gpu=False)`` forces ``RecTrainer.__init__`` onto its
        CPU/gloo device branch (trainer.py:81-83). recir places dense compute
        on TPU via ``jax.devices()`` regardless of the torch device being cpu
        -- this matches rankfm/train.py, which builds the model on
        torch.device("cpu") and lets recir/JAX place compute on TPU.
      * ``data_source_config`` is a ``RandomDataSourceConfig`` purely so the
        QPS metric in RecTrainer._create_metrics_reporter has a positive
        ``batch_size`` (= local_batch_size). The actual training data is
        produced by RankFMMvaiModelFamily.create_train_datapipe, which reuses
        rankfm's synthetic / barnfs-replay path.
      * ``max_steps`` is enforced by the data pipe (RecTrainer loops until the
        dataiter raises StopIteration); see model_family.create_train_datapipe.
    """
    data_source_config = RandomDataSourceConfig(
        batch_size=trainer_cfg.local_batch_size,
        num_batches=trainer_cfg.max_steps,
    )

    return RecTrainerConfig(
        data_source_config=data_source_config,
        system_config=SystemConfig(use_gpu=False, dist_backend="gloo"),
        metrics_config=MetricsConfig(report_interval=1, tensorboard_logger=True),
        model_config_module=("hammer.v3.models.udd_fm_v3.udd_fm_config"),
        checkpoint_config=None,
    )
