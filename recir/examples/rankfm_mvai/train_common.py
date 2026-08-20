#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Shared bootstrap + helpers for the rankfm_mvai entrypoints.

Both ``train.py`` (recir XLA-GSPMD TPU run) and ``train_cpu_eager.py`` (the
matched-config CPU-eager reference) import from here so they agree EXACTLY on:
  * the gr::batched_complete_cumsum bootstrap (module-level, runs on import),
  * dist-env resolution (RANK/WORLD_SIZE/MASTER_*) for the gloo PG,
  * the CPU barrier shim for the metric all-gather, and
  * the cls/reg task partition for metric reporting.

Keeping these in one place guarantees the CPU baseline and the TPU run differ
only where they must (compile vs eager, jax.distributed.initialize), not in
incidental setup.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, List, Optional, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP_TABLE entries
import torch
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl

logger: logging.Logger = logging.getLogger(__name__)

# Bootstrap gr::batched_complete_cumsum if the C++ library failed to load
# (shared gr-op bootstrap). recir's SPMD decomp table replaces this op
# with _batched_complete_cumsum_impl at XLA lowering time, but the op must
# exist as a valid OpOverload first.
try:
    _cumsum_op: object = torch.ops.gr.batched_complete_cumsum.default  # noqa: B018
except AttributeError:
    torch.library.define("gr::batched_complete_cumsum", "(Tensor values) -> Tensor")
    torch.library.impl(
        "gr::batched_complete_cumsum",
        "CompositeImplicitAutograd",
        _batched_complete_cumsum_impl,
    )

SPMD_DECOMP_TABLE[torch.ops.gr.batched_complete_cumsum.default] = (
    _batched_complete_cumsum_impl
)


def build_cls_reg_partition(
    module_config: BlueReelsVDDModuleConfig,
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Partition the udd_fm_v3 tasks into cls/reg buckets for metric reporting.

    Mirrors mvai_cloud/.../models/udd_fm/train.py:226-242 exactly. rankfm_mvai
    runs the SAME model (hammer udd_fm_v3 ``BlueReelsVDDTrainV3``) so the same
    task structure (num_main_tasks / num_main_reg_tasks / has_unpersonalized /
    num_unpersonalized_tasks) applies. The naive split using only
    num_main_reg_tasks is wrong when has_unpersonalized=True because the
    unpersonalized tasks live AFTER the regression block and are trained as
    classification (sigmoid+BCE); without this account, metric reporting would
    compute classification NE on regression heads and vice versa.
    """
    all_task_names = list(module_config.task_names)
    n_total = module_config.num_main_tasks
    n_reg = module_config.num_main_reg_tasks
    if module_config.has_unpersonalized:
        n_unpers = module_config.num_unpersonalized_tasks
        cls_end = n_total - n_reg - n_unpers
        reg_end = n_total - n_unpers
        cls_indices = list(range(cls_end)) + list(range(reg_end, n_total))
        reg_indices = list(range(cls_end, reg_end))
    else:
        cls_end = n_total - n_reg
        cls_indices = list(range(cls_end))
        reg_indices = list(range(cls_end, n_total))
    cls_task_names = [all_task_names[i] for i in cls_indices]
    reg_task_names = [all_task_names[i] for i in reg_indices]
    return cls_task_names, cls_indices, reg_task_names, reg_indices


def setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def resolve_dist_env() -> None:
    """Populate RANK / WORLD_SIZE / LOCAL_RANK from the JobSet env for gloo.

    RecTrainer.__init__ reads RANK / WORLD_SIZE / LOCAL_RANK (trainer.py:65-67)
    and, on the use_gpu=False branch, calls init_process_group(backend="gloo")
    which torch resolves against RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
    On GKE TPU the JobSet exposes JOB_COMPLETION_INDEX (the pod rank) and
    TPU_PROCESS_ADDRESSES (the comma-separated worker list); map those onto the
    torch env if RANK / WORLD_SIZE are not already set. Single-host runs fall
    back to RANK=0 / WORLD_SIZE=1, giving RecTrainer a trivial 1-process gloo PG.
    """
    tpu_addrs = os.environ.get("TPU_PROCESS_ADDRESSES", "")
    addr_list = [a.strip() for a in tpu_addrs.split(",") if a.strip()]
    if "RANK" not in os.environ:
        os.environ["RANK"] = os.environ.get("JOB_COMPLETION_INDEX", "0")
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(len(addr_list)) if len(addr_list) > 1 else "1"
    os.environ.setdefault("MASTER_PORT", "29500")
    if int(os.environ["WORLD_SIZE"]) > 1:
        # Multi-host: the gloo coordinator must be the rank-0 pod, NOT
        # 127.0.0.1 (which makes every pod rendezvous with itself and hang
        # "waiting for all ranks to join"). Prefer an explicit MASTER_ADDR
        # (FQDN) from the JobSet; else the first TPU_PROCESS_ADDRESSES host.
        # Mirrors the embedding cpu_backend init_pg.
        if "MASTER_ADDR" not in os.environ and addr_list:
            os.environ["MASTER_ADDR"] = addr_list[0].split(":")[0]
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    logger.info(
        "[DIST] resolved RANK=%s WORLD_SIZE=%s MASTER_ADDR=%s MASTER_PORT=%s "
        "TPU_PROCESS_ADDRESSES=%r",
        os.environ["RANK"],
        os.environ["WORLD_SIZE"],
        os.environ.get("MASTER_ADDR"),
        os.environ["MASTER_PORT"],
        tpu_addrs,
    )


def ensure_gloo_process_group() -> Tuple[int, int]:
    """Initialize the default Gloo group from the resolved dist environment."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(
        "[DIST] initializing Gloo rank=%d world=%d via %s:%s",
        rank,
        world_size,
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"],
    )
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )
    return dist.get_rank(), dist.get_world_size()


def force_cpu_barrier() -> None:
    """Route torch.distributed.barrier() onto CPU instead of the TPU accelerator.

    The udd_fm metric set (NE/CTR/MSE) syncs cross-rank at report time:
    torchmetrics Metric.compute() -> gather_all_tensors() -> torch.distributed
    .barrier(group). Recent torch makes the barrier device-aware via
    torch._C._get_accelerator(), which on this image returns the
    torchax/jax-tpu-embedding PrivateUse1 accelerator -- unregistered for c10d --
    raising "Please register PrivateUse1HooksInterface ...". The process group is
    gloo and all metric state is on CPU, so replace barrier with a CPU all_reduce:
    a full cross-rank sync that never queries the accelerator. The all_gather of
    the (CPU) metric tensors that follows still runs, so the reported NE/CTR/MSE
    stay GLOBALLY correct across all ranks. No mvai_cloud / torchmetrics edits.
    """
    import torch.distributed.distributed_c10d as c10d

    def _cpu_barrier(
        group: Optional[Any] = None,
        async_op: bool = False,
        device_ids: Optional[List[int]] = None,
        timeout: Optional[Any] = None,
    ) -> Optional[Any]:
        if not (dist.is_available() and dist.is_initialized()):
            return None
        pg = group if group is not None else c10d._get_default_group()
        work = dist.all_reduce(torch.zeros(1), group=pg, async_op=async_op)
        return work if async_op else None

    c10d.barrier = _cpu_barrier
    dist.barrier = _cpu_barrier
