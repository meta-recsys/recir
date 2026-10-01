# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Smoke-test CoScaling config for the recir (XLA) path.

The prod ``fm_coscaling_shrunk_3k`` config, translated with the same function
the GPU v3 wrapper uses. On top:

* every table (sequence and dense-token) is cut to ``max_rows``, so the model
  fits on one host and the batch's ids, clamped to the same ``max_rows``, are
  valid against all of them;
* the encoder's static bounds, which let it trace without device reads.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

from hammer.v3.models.udd_fm_v3.coscaling.coscaling_config import CoScalingModuleConfig
from hammer.v3.models.udd_fm_v3.launch.coscaling.train_module_wrapper import (
    _translate_coscaling_config,
)
from hammer.v3.models.udd_fm_v3.tests.parity.parity_common import (
    _shrink_embedding_tables,
)
from hammer.v3.modules.mot.sequence_context import SequenceContext
from minimal_viable_ai.core.ranking_common.configs import MTMLROOFeatureConfig
from minimal_viable_ai.models.blue_reels_vdd.standalone.coscaling.fm_coscaling_shrunk_3k import (
    ConfigProvider,
)

logger: logging.Logger = logging.getLogger(__name__)


def build_coscaling_config(
    max_rows: int,
    max_uih_len: int | None = None,
) -> Tuple[CoScalingModuleConfig, MTMLROOFeatureConfig, Any]:
    """Return ``(module_config, feature_config, prod_config)``.

    ``max_uih_len`` caps the per-sample history the bound is sized for. Leave it
    None to size for prod's ``hstu_max_seq_len``; pass the batch's truncation
    length on CPU, where padding every request to ~3k tokens is only cost.
    """
    prod: Any = ConfigProvider().create_config()
    _shrink_embedding_tables(prod.embedding_tables, max_rows)
    for emb in (prod.module_config.additional_embeddings or {}).values():
        if getattr(emb, "num_embeddings", 0) > 0:
            emb.num_embeddings = max_rows

    module_config = _translate_coscaling_config(
        prod_module_config=prod.module_config,
        feature_config=prod.feature_config,
        embedding_tables=prod.embedding_tables,
    )

    uih_len = int(module_config.hstu_max_seq_len)
    if max_uih_len is not None:
        uih_len = min(uih_len, max_uih_len)
    # Same per-sample bound MoTPreprocessor uses for its static runtime max:
    # history + contextual tokens, then candidates on top.
    contextual = SequenceContext(config=module_config, seq_index=0)
    max_ro_len = uih_len + contextual.max_sum_contextual_seq_len
    max_num_targets = int(module_config.hstu_max_output_len)
    # Assigned, not dataclasses.replace: replace re-runs __post_init__ on the
    # whole udd config, which is not guaranteed idempotent.
    module_config.coscaling_static_max_ro_len = max_ro_len
    module_config.coscaling_static_max_num_targets = max_num_targets
    logger.info(
        "[coscaling] tables=%d dense_token_tables=%d max_rows=%d "
        "static_max_ro_len=%d static_max_num_targets=%d",
        len(module_config.embedding_tables),
        len(module_config.dense_token_embedding_tables),
        max_rows,
        max_ro_len,
        max_num_targets,
    )
    return module_config, prod.feature_config, prod
