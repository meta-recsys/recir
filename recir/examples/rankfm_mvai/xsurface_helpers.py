#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared bespoke-compile helpers for the rankfm_mvai example.

These functions were the reusable core of the original bespoke ``train.py``
loop. They are imported by:
  * ``model_family.py`` (``_resolve_nro_per_sample``,
    ``make_padding_region_transforms``, ``prepare_override_build_sample``),
  * ``parity_forward_loss.py`` (``_compile_with_xsurface_dispatchers``,
    ``_extract_train_step_args``),
  * ``train_cpu_eager.py`` (``_build_data_pipe``, ``_build_replay_iter``,
    ``_extract_train_step_args``, ``_log_batch_info``,
    ``_resolve_nro_per_sample``, ``_setup_logging``),
  * ``train.py`` (``_is_jax_profiling_enabled``, ``_log_hbm_memory`` for the
    profiling hook).

Keeping them in one module guarantees the recir-compiled training paths and the
parity check share ONE definition (a drift in ``skip_args`` / ``jagged_pairs``
or override sizing would silently break parity).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax
import recir
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP_TABLE entries
import torch
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.utils import world_candidate_sum
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.region_transforms.padding import (
    compute_padded_shape,
    JaggedPairConfig,
    PaddingRegionTransform,
)
from recir.tests.xla.xsurface_test_utils import (
    resize_batch_inplace,
    resize_kjt_per_key_lengths,
)

from .config import RankFMTrainerConfig
from .data_pipe import BatchReplayPipe, replay_batch_iter

# Bootstrap gr::batched_complete_cumsum if the C++ library failed to load
# (e.g. OSS Docker container where hammer/oss/generative_recommenders/ops/cpp
# is not compiled into a .so). The recir SPMD decomp table replaces this op
# with _batched_complete_cumsum_impl at XLA lowering time, but the registration
# below requires torch.ops.gr.batched_complete_cumsum.default to exist as a
# valid OpOverload first.
try:
    _cumsum_op: object = torch.ops.gr.batched_complete_cumsum.default  # noqa: B018
except AttributeError:
    torch.library.define("gr::batched_complete_cumsum", "(Tensor values) -> Tensor")
    torch.library.impl(
        "gr::batched_complete_cumsum",
        "CompositeImplicitAutograd",
        _batched_complete_cumsum_impl,
    )

# Register gr.batched_complete_cumsum decomposition for XLA lowering.
SPMD_DECOMP_TABLE[torch.ops.gr.batched_complete_cumsum.default] = (
    _batched_complete_cumsum_impl
)

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_nro_per_sample(
    module_config: BlueReelsVDDModuleConfig,
    enable_padding: bool,
) -> int:
    """NRO budget is the source of truth for padding-related shapes when
    padding is enabled — PRT's ``max_per_sample_count`` and the override-
    build sample's ``padded_nro_bs`` must compute the same ``pair_padded_bs``,
    so they both have to read this value. Falls back to ``hstu_max_output_len``
    when padding is off (no budget needed).
    """
    if enable_padding:
        budget = module_config.static_nro_per_sample_budget
        assert budget is not None, (
            "static_nro_per_sample_budget must be set when enable_padding=True"
        )
        return budget
    return int(module_config.hstu_max_output_len)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def _build_data_pipe(
    trainer_cfg: RankFMTrainerConfig,
    gloo_rank: int,
    gloo_world: int,
) -> Optional[BatchReplayPipe]:
    """Construct the replay pipe iff ``--gcs_data_path`` was given.

    Direct-read template path skips the 58-min ``ls /mnt/barnfs`` enumeration
    when ``--data_file_template`` is set; one file per epoch is enough since
    the rotating window in ``replay_batch_iter`` provides per-step diversity.
    """
    if not trainer_cfg.gcs_data_path:
        logger.info("Using synthetic data (no --gcs_data_path)")
        return None
    explicit_files: Optional[List[str]] = None
    if trainer_cfg.data_file_template:
        explicit_files = [trainer_cfg.data_file_template.format(i=1)]
        logger.info(
            f"Direct-read mode: file_list={explicit_files} (template "
            f"'{trainer_cfg.data_file_template}')"
        )
    pipe = BatchReplayPipe(
        path=trainer_cfg.gcs_data_path,
        rank=gloo_rank,
        world_size=gloo_world,
        replicate_across_ranks=trainer_cfg.data_replicate_across_ranks,
        file_list=explicit_files,
    )
    logger.info(
        f"Replay pipe: {len(pipe)} batches for rank {gloo_rank} "
        f"(replicate={trainer_cfg.data_replicate_across_ranks})"
    )
    return pipe


def make_padding_region_transforms(
    num_candidates_per_sample: int,
    padded_num_devices: int,
    num_processes: int,
) -> List[Any]:
    """Build the rankfm train-step PaddingRegionTransform list.

    Shared by the recir-compiled train cell (``model_family.py``) and the
    parity test (``parity_forward_loss``) so a drift in ``skip_args`` /
    ``jagged_pairs`` between them cannot silently break parity.
    """
    return [
        PaddingRegionTransform(
            jagged_pairs={
                # RO/NRO id-list pairs are opted out below -- they only
                # flow into PreprocessorDispatcher (a CPU graph break)
                # and stay at local stride throughout the preprocessor.
                # The dispatcher handles the padding.
                "unary_values": JaggedPairConfig(
                    "unary_lengths",
                    per_sample_budgets=1,
                    batch_size_arg="nro_float_features",
                    count_arg="num_candidates",
                    max_per_sample_count=num_candidates_per_sample,
                ),
            },
            batch_size_arg="ro_float_features",
            # Align to lcm(GSPMD devices, total SC count) so PRT's runtime
            # delivery matches the override-build sample's padded shape
            # (which uses the same modulus above).
            num_devices=padded_num_devices,
            num_processes=num_processes,
            skip_args={
                "ro_id_list_values",
                "ro_id_list_lengths",
                "nro_id_list_values",
                "nro_id_list_lengths",
                "all_labels",
                "candidate_sum",
                "world_candidate_sum",
            },
        ),
    ]


def prepare_override_build_sample(
    batch: Any,
    module_config: BlueReelsVDDModuleConfig,
    global_bs: int,
    num_candidates_per_sample: int,
    padded_num_devices: int,
    enable_padding: bool,
) -> Tuple[Any, int]:
    """Resize ``batch`` to the override-build size; return ``(batch_sample,
    override_bs)``.

    The override-build sample is sized at GLOBAL (padded when padding is on) so
    the override cached_output shapes match the FX trace + PRT pad outputs.
    Shared by the train cell and the parity test so the override sizing cannot
    drift from the PaddingRegionTransform sizing.
    """
    if enable_padding:
        # Override-build sample uses GLOBAL (padded) sizes -- matches PRT's
        # pad_to_static fake kernel which also returns the global target_size,
        # so FX trace shapes are consistent across override outputs and PRT pad
        # outputs. At runtime fwd_fn returns PER-HOST data and _to_sharded_jax
        # expands per-host -> global, matching PRT runtime.
        padded_ro_bs = compute_padded_shape(global_bs, padded_num_devices)
        # padded_nro_bs = padded_ro_bs * per-sample NRO budget: each RO sample
        # contributes exactly static_nro_per_sample_budget NRO entries.
        padded_nro_bs = padded_ro_bs * num_candidates_per_sample
        expanded_batch = copy.deepcopy(batch)
        resize_batch_inplace(
            expanded_batch,
            target_b_ro=padded_ro_bs,
            target_b_nro=padded_nro_bs,
        )
        # Force each RO key's per-sample length to its configured budget.
        if module_config.static_ro_per_sample_budgets:
            expanded_batch.ro_id_list_features = resize_kjt_per_key_lengths(
                expanded_batch.ro_id_list_features,
                module_config.static_ro_per_sample_budgets,
                batch_size=padded_ro_bs,
            )
        override_bs = padded_ro_bs
    else:
        expanded_batch = copy.deepcopy(batch)
        resize_batch_inplace(
            expanded_batch,
            target_b_ro=global_bs,
            target_b_nro=global_bs * num_candidates_per_sample,
        )
        override_bs = global_bs

    all_labels = expanded_batch.all_labels
    assert all_labels is not None
    batch_sample = _inputs_to_feature_bundle(
        expanded_batch.ro_id_list_features,
        None,
        expanded_batch.nro_id_list_features,
        expanded_batch.num_candidates,
        all_labels,
    )
    return batch_sample, override_bs


def _compile_with_xsurface_dispatchers(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    emb_opt: Optional[torch.optim.Optimizer],
    batch: Any,
    dispatchers: List[Any],
    callable_: Any,
    *,
    parallel_mode: GSPMDParallel,
    backend: XLAGSPMDBackend,
    region_transforms: List[Any],
) -> Any:
    """Drive ``recir.trace`` / ``recir.compile_from_gm`` with the rankfm
    PreprocessorDispatcher + SparseArchDispatcher pair.

    Traces both the dense optimizer ``opt`` and the optional embedding optimizer
    ``emb_opt`` (non-None only for the in_graph backend) into the single
    compiled GSPMD program.
    """
    from recir.api import make_compiled_runner
    from torchrec.distributed.model_parallel import DistributedModelParallel

    # Stage 0
    model_to_train = (
        model.module if isinstance(model, DistributedModelParallel) else model
    )
    for dispatcher in dispatchers:
        dispatcher.install_eager(model_to_train)

    # Stage 2
    gm, flat_state = recir.trace(
        callable_,
        model_to_train,
        opt,
        emb_opt,
        *_extract_train_step_args(batch),
        parallel_mode=parallel_mode,
        region_transforms=region_transforms,
    )

    # Stage 3
    for dispatcher in dispatchers:
        gm, flat_state = dispatcher.install_graph(gm, flat_state)

    # Stages 4-10
    compiled_obj: Any = recir.compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
        region_transforms=region_transforms,
    )

    return make_compiled_runner(compiled_obj)


def _is_jax_profiling_enabled(trainer_cfg: RankFMTrainerConfig) -> bool:
    """Check if JAX profiling should be enabled (TPU present + start step > 0)."""
    try:
        is_tpu = len(jax.devices()) > 0 and jax.devices()[0].platform == "tpu"
        return is_tpu and trainer_cfg.profile_start_step > 0
    except (RuntimeError, IndexError):
        logger.warning("Could not detect JAX TPU platform, profiling disabled")
        return False


def _log_hbm_memory() -> None:
    """Log per-device HBM memory usage."""
    for i, dev in enumerate(jax.local_devices()):
        stats = dev.memory_stats()
        if stats:
            used = stats.get("bytes_in_use", 0)
            limit = stats.get("bytes_limit", 0)
            logger.info(
                f"  Device {i} HBM: {used / 1e9:.2f}/{limit / 1e9:.2f} GB "
                f"({100 * used / max(limit, 1):.1f}%)"
            )


def _extract_train_step_args(
    batch: Any,
) -> Tuple[Any, ...]:
    """Extract raw tensor args from a batch for the train step."""
    ro_kjt = batch.ro_id_list_features
    nro_kjt = batch.nro_id_list_features
    unary_kjt = batch.unary_features
    assert unary_kjt is not None
    return (
        batch.ro_float_features,
        batch.nro_float_features,
        ro_kjt.values(),
        ro_kjt.lengths(),
        nro_kjt.values(),
        nro_kjt.lengths(),
        unary_kjt.values(),
        unary_kjt.lengths(),
        batch.num_candidates,
        batch.supervision_label.t().contiguous(),
        batch.supervision_weight.t().contiguous(),
        batch.all_labels,
        world_candidate_sum(batch.num_candidates),
    )


def _log_per_key_stats(name: str, kjt: Any, B: int) -> str:
    """Format per-key length stats for a KJT."""
    keys = kjt.keys()
    lengths = kjt.lengths()
    num_keys = len(keys)
    per_key = lengths.reshape(num_keys, B)
    parts: List[str] = []
    for i, key in enumerate(keys):
        kl = per_key[i]
        total = int(kl.sum())
        mn = int(kl.min())
        mx = int(kl.max())
        if mn == mx:
            parts.append(f"{key}:{mn}x{B}={total}")
        else:
            parts.append(f"{key}:[{mn}-{mx}]sum={total}")
    return f"{name}({num_keys} keys): " + ", ".join(parts)


def _log_batch_info(step: int, batch: Any) -> None:
    """Log per-step batch shapes and per-key length stats."""
    nc = batch.num_candidates
    B = batch.ro_float_features.shape[0]
    logger.info(
        f"[BATCH step={step}] B={B} "
        f"ro_float={list(batch.ro_float_features.shape)} "
        f"nro_float={list(batch.nro_float_features.shape)} "
        f"num_candidates sum={int(nc.sum())} min={int(nc.min())} max={int(nc.max())}"
    )
    logger.info(
        f"[BATCH step={step}] %s",
        _log_per_key_stats("RO", batch.ro_id_list_features, B),
    )
    B_NRO = int(nc.sum())
    logger.info(
        f"[BATCH step={step}] %s",
        _log_per_key_stats("NRO", batch.nro_id_list_features, B_NRO),
    )


def _build_replay_iter(
    trainer_cfg: RankFMTrainerConfig,
    num_candidates: int,
    data_pipe: Optional[BatchReplayPipe],
    key_to_size: Optional[Dict[str, int]],
    ro_per_key_budgets: Optional[List[int]] = None,
) -> Optional[Iterator[Any]]:
    if data_pipe is None:
        return None
    if key_to_size is None:
        logger.warning(
            "Replay mode but key_to_size is None; loaded IDs will not be "
            "clamped against model table sizes (may IndexError)."
        )
    return replay_batch_iter(
        data_pipe,
        key_to_size or {},
        max_steps=trainer_cfg.max_steps,
        local_batch_size=trainer_cfg.local_batch_size,
        num_candidates=num_candidates,
        ro_per_key_budgets=ro_per_key_budgets,
    )
