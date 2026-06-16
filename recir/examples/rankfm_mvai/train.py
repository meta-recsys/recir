#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""RankFM (xsurface V3) TPU training, driven by the mvai_cloud RecTrainer.

Runs the SAME model as recir/examples/rankfm (the hammer/v3 udd_fm
``BlueReelsVDDTrainV3``) on TPU via recir XLA-GSPMD, but orchestrated by the
mvai_cloud training framework (RecTrainer + RecModelFamily + RecTrainPipeline)
instead of rankfm's bespoke train.py loop.

The TPU path is achieved WITHOUT editing mvai_cloud:
  * ``jax.distributed.initialize()`` is called here, before RecTrainer is
    constructed (mirrors rankfm/train.py:682).
  * ``SystemConfig(use_gpu=False)`` makes RecTrainer take its CPU/gloo device
    branch (trainer.py:81-83); recir lowers the dense graph to TPU via
    ``jax.devices()`` regardless of the torch device being cpu.
  * All TPU/recir logic lives in RankFMMvaiModelFamily (model + recir embedding
    backend + optimizer) and RecirXLATrainPipeline (recir compile + step).

Usage:
    python3 -u -m recir.examples.rankfm_mvai.train --max_steps 10
    python3 -u -m recir.examples.rankfm_mvai.train --max_steps 5 \\
        --embedding_backend tpu
"""

from __future__ import annotations

import logging
import os

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP_TABLE entries
import torch
import torch.distributed as dist
from minimal_viable_ai.core.trainer.job_resolver import JobResolver
from minimal_viable_ai.core.trainer.trainer import RecTrainer
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl

from .config import build_model_family_config, build_trainer_config, parse_cli
from .model_family import RankFMMvaiModelFamily
from .train_pipeline import RecirXLATrainPipelineProvider

# Bootstrap gr::batched_complete_cumsum if the C++ library failed to load
# (same as rankfm/train.py:74-89). recir's SPMD decomp table replaces this op
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

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def _resolve_dist_env() -> None:
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
        # Mirrors recir/examples/rankfm/embedding/cpu_backend.py::init_pg.
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


def main() -> None:
    # Must be called before any JAX op. On GKE TPU the PJRT plugin reads the
    # coordinator address + process index from TPU runtime metadata; without
    # this each pod is jax.process_count()==1 and the SparseCore collective
    # (which spans all chips) deadlocks. Mirrors rankfm/train.py:682.
    jax.distributed.initialize()
    _setup_logging()
    logger.info(
        f"[DIST] jax.distributed.initialize() done: "
        f"process_count={jax.process_count()}, "
        f"device_count={jax.device_count()}, "
        f"local_device_count={jax.local_device_count()}, "
        f"process_index={jax.process_index()}"
    )

    trainer_cfg, model_args = parse_cli()
    logger.info(f"Trainer config: {trainer_cfg}")
    logger.info(f"Model args: {model_args}")

    _resolve_dist_env()

    # Inject the rankfm CLI knobs onto the family class (the base RecModelFamily
    # ctor signature -- config/pg/device -- can't carry them).
    RankFMMvaiModelFamily.configure(trainer_cfg, model_args)

    model_config = build_model_family_config(trainer_cfg, model_args)
    trainer_config = build_trainer_config(trainer_cfg)

    RecTrainer(
        model_family_clazz=RankFMMvaiModelFamily,
        trainer_config=trainer_config,
        model_config=model_config,
        train_pipeline_provider=RecirXLATrainPipelineProvider(),
        job_resolver=JobResolver(),
    ).train()

    # Cleanup. JAX's PJRT runtime + libtpu workers + any gloo collective thread
    # keep the process alive after train() returns, so the JobSet pod never
    # reaches Completed. Shut both down, then force-exit (mirrors
    # rankfm/train.py:787-798).
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        jax.distributed.shutdown()
    except Exception as e:  # noqa: BLE001 -- best-effort cleanup
        logger.warning(f"jax.distributed.shutdown() raised (non-fatal): {e}")
    logger.info("Cleanup complete; exiting.")
    os._exit(0)


if __name__ == "__main__":
    main()
