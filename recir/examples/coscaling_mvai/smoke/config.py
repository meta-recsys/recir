# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Smoke-test CoScaling config and batch, built on the devserver.

The prod ``fm_coscaling_shrunk_3k`` config, translated with the same function
the GPU v3 wrapper uses. On top:

* every table (sequence and dense-token) is cut to ``max_rows``, so the model
  fits on one host; ``build_batch`` clamps the batch's ids to the same
  ``max_rows``, so they are valid against all of them;
* the encoder's static bounds, which let it trace without device reads;
* rankfm's static-shape mode (per-sample RO / NRO token budgets), which the
  MoT preprocessor reads and the padding (PaddingRegionTransform) relies on.

Devserver only: this needs the internal minimal_viable_ai and Configerator.
The TPU image gets the same objects from ``artifact.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import torch
from hammer.v3.models.udd_fm_v3.coscaling.coscaling_config import CoScalingModuleConfig
from hammer.v3.models.udd_fm_v3.launch.coscaling.train_module_wrapper import (
    _translate_coscaling_config,
)
from hammer.v3.models.udd_fm_v3.tests.parity.parity_common import (
    _make_fixed_batch,
    _shrink_embedding_tables,
)
from hammer.v3.modules.mot.sequence_context import SequenceContext
from minimal_viable_ai.models.blue_reels_vdd.standalone.coscaling.fm_coscaling_shrunk_3k import (
    ConfigProvider,
)
from recir.examples.rankfm_mvai.config import _build_static_ro_budgets
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_PATH: str = "/tmp/coscaling_roo_batch.pt"
# _make_fixed_batch always truncates rows; this bound makes that a no-op.
_NO_TRUNCATION: int = 1 << 31


def build_coscaling_config(
    max_rows: int,
    max_uih_len: Optional[int] = None,
) -> Tuple[CoScalingModuleConfig, Any, Any]:
    """Return ``(module_config, feature_config, prod_config)``.

    ``max_uih_len`` caps the per-sample history the bound is sized for. Leave it
    None to size for prod's ``hstu_max_seq_len``; pass the batch's truncation
    length when the batch is truncated, where padding every request to ~3k
    tokens is only cost.
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
    # Static-shape mode, as rankfm's _apply_static_shape_mode (prod-faithful):
    # NRO budget = hstu_max_output_len; RO budget = the same history length the
    # encoder bound is sized for, so PRT never pads a row past that bound.
    module_config.static_shape_mode = True
    module_config.static_nro_per_sample_budget = max_num_targets
    module_config.static_ro_per_sample_budgets = _build_static_ro_budgets(
        module_config, uih_len
    )
    logger.info(
        "[coscaling] tables=%d dense_token_tables=%d max_rows=%d "
        "static_max_ro_len=%d static_max_num_targets=%d ro_budgets=%s",
        len(module_config.embedding_tables),
        len(module_config.dense_token_embedding_tables),
        max_rows,
        max_ro_len,
        max_num_targets,
        module_config.static_ro_per_sample_budgets,
    )
    return module_config, prod.feature_config, prod


def build_batch(
    prod_config: Any, batch_path: str, max_rows: int, max_seq: Optional[int]
) -> Any:
    if not os.path.exists(batch_path):
        raise FileNotFoundError(
            f"no dumped batch at {batch_path}; run "
            "hammer/v3/models/udd_fm_v3/launch/coscaling/dump_batch.sh on a devgpu"
        )
    # _make_fixed_batch reads the batch path and row truncation from the
    # environment; it also dedups keys and clamps ids to the shrunk tables.
    os.environ["PARITY_BATCH_PATH"] = batch_path
    os.environ["PARITY_MAX_SEQ"] = str(_NO_TRUNCATION if max_seq is None else max_seq)
    batch: Any = _make_fixed_batch(prod_config, torch.device("cpu"), max_rows)
    batch.unary_features = padding_anchor(int(batch.nro_float_features.size(0)))
    return batch


# The unary jagged pair is how PaddingRegionTransform finds the candidate axis:
# it pads nro_float_features, the supervision tensors and num_candidates by
# matching them to that pair's batch reference (rankfm's make_padding_region_
# transforms). CoScaling reads no unary feature, so give it one key with one id
# per candidate -- enough to anchor the axis, inert to the model.
PADDING_ANCHOR_KEY: str = "coscaling_padding_anchor"


def padding_anchor(num_candidates: int) -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=[PADDING_ANCHOR_KEY],
        values=torch.zeros(num_candidates, dtype=torch.int64),
        lengths=torch.ones(num_candidates, dtype=torch.int32),
    )
