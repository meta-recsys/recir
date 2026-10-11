# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DLRMv3 training on TPU with recir (GSPMD + XLA through torchax).

torch runs on the host CPU and JAX owns the TPUs. The whole train step is
traced once and lowered to a single GSPMD program over all chips; the batch is
sharded on dim 0 and dense parameters are replicated. Embedding tables are
parameters inside the same program, row-wise sharded over all chips
(``recir.embedding`` in_graph backend). ``--embedding_backend cpu`` instead
keeps them in torchrec on the host, for debugging without TPUs.

Single host::

    python -m recir.examples.dlrmv3.tpu.train --dataset kuairand-1k --data_root /data

Multi-host GKE JobSets (see manifests/) set TPU_PROCESS_ADDRESSES /
JOB_COMPLETION_INDEX; ``--batch_size`` is per chip.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# torch_tpu (if installed) would claim PrivateUse1 before torchax registers it.
if __name__ == "__main__" and os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.execvpe(sys.orig_argv[0], sys.orig_argv, os.environ)

# 64-bit JAX: recir.training's in-graph warmup compares the
# int64 iteration counter against int64 stage bounds.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from generative_recommenders.common import HammerKernel, set_dev_mode
from recir.embedding.backend import EmbeddingBackend
from recir.embedding.in_graph import InGraphEmbeddingBackend, SHARDED_TABLE_PREFIX
from recir.examples.dlrmv3.common.config import DLRMv3TrainConfig, parse_config
from recir.examples.dlrmv3.common.data import (
    make_dataset,
    make_train_eval_loaders,
    TokenBudgetBatch,
    TokenBudgetCollate,
)
from recir.examples.dlrmv3.common.metrics import (
    log_metrics,
    make_metrics,
    update_metrics,
)
from recir.examples.dlrmv3.common.model import DlrmV3TrainModule
from recir.examples.dlrmv3.common.steps import eval_step, flatten_batch
from recir.examples.dlrmv3.tpu.embedding_backends import (
    BACKENDS,
    BatchGeometry,
    build_embedding_backend,
    make_dispatchers,
    prepare_backend,
)
from recir.region_transforms.dropout_prng import DropoutPRNGTransform
from recir.training import tpu_runtime  # noqa: F401 -- also registers gr/fbgemm ops
from recir.training.compile import (
    compile_gspmd_step,
    LR_ITER_BUFFER,
    RecirCompileSpec,
    RecirMode,
)
from recir.training.compiled_module import RecirGSPMDCompiledModuleBase
from recir.training.optim import WarmupController
from recir.training.setup import setup_dense_optimizer_and_warmup
from torch.utils.data import DataLoader

logger: logging.Logger = logging.getLogger(__name__)


class DLRMv3TPUModule(RecirGSPMDCompiledModuleBase):
    """The compiled DLRMv3 train step, plus the eval step compiled together with
    it (sharing the model's parameters and device state). Host syncs,
    checkpoint restore and the DCP view (``make_checkpointable``) come from
    ``RecirGSPMDCompiledModuleBase``."""

    def __init__(
        self,
        model: nn.Module,
        dense_opt: torch.optim.Optimizer,
        emb_opt: Optional[torch.optim.Optimizer],
        warmup: WarmupController,
        dispatchers: List[Any],
        spec: RecirCompileSpec,
        trace_batch: TokenBudgetBatch,
        num_processes: int,
        fused_opt: Any,
        sparse_lr_buffer: Optional[torch.Tensor],
        train_step_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._dense_opt = dense_opt
        self._emb_opt = emb_opt
        self._warmup = warmup
        self._dispatchers = dispatchers
        self._spec = spec
        self._trace_batch = trace_batch
        self._num_processes = num_processes
        self._fused_opt = fused_opt
        self._sparse_lr_buffer = sparse_lr_buffer
        self._train_step_fn = train_step_fn
        self._eval_runner: Any = None

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def compiled_step(self) -> Any:
        return self._get_or_build_compiled_step(self._trace_batch)

    def _build_compiled_step(self, first_batch: Any) -> Any:
        train_runner, self._eval_runner = compile_gspmd_step(
            self._model,
            self._dense_opt,
            self._trace_batch,
            self._dispatchers,
            self._spec,
            train_step_fn=self._train_step_fn,
            flatten_batch=flatten_batch,
            sparse_opt=self._emb_opt,
            eval_step_fn=eval_step,
        )
        return train_runner

    def evaluate(self, batch: TokenBudgetBatch) -> Dict[str, torch.Tensor]:
        """Forward-only step on the current (device-resident) parameters."""
        self.compiled_step  # noqa: B018 -- compiles both steps on first use
        _, named = self._eval_runner(self._model, *batch.flat())
        return named

    def _step_args(self, batch: Any) -> Tuple[Any, ...]:
        opts = (
            (self._dense_opt,)
            if self._emb_opt is None
            else (self._dense_opt, self._emb_opt)
        )
        return (self._model, *opts, *flatten_batch(batch, global_loss_denom=True))

    def _run_step(
        self, compiled_step: Any, batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._push_sparse_lr()
        return super()._run_step(compiled_step, batch)

    def _push_sparse_lr(self) -> None:
        """Send the current sparse LR into the CPU backend's fused optimizer."""
        if self._fused_opt is None or self._sparse_lr_buffer is None:
            return
        lr = float(self._sparse_lr_buffer.item())
        for group in self._fused_opt.param_groups:
            group["lr"] = lr
        self._fused_opt.zero_grad()


def _parse(
    argv: Optional[Sequence[str]],
) -> Tuple[DLRMv3TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embedding_backend", choices=BACKENDS, default="in_graph")
    parser.add_argument("--profile_dir", default="")
    parser.add_argument("--profile_start_step", type=int, default=0)
    parser.add_argument("--profile_end_step", type=int, default=0)
    return parse_config(argv, parser)


class TPUTrainer:
    def __init__(
        self, cfg: DLRMv3TrainConfig, args: argparse.Namespace, world: int
    ) -> None:
        self.cfg = cfg
        self.args = args
        self.hstu = cfg.hstu_config()
        self.tables = cfg.embedding_tables()
        self.layout = cfg.layout(self.hstu)
        self.geom = BatchGeometry(
            real_rows=cfg.batch_size,
            local_chips=jax.local_device_count(),
            num_ranks=jax.device_count(),
        )
        self.world = world
        self.task_names = [t.task_name for t in self.hstu.multitask_configs]
        self.backend: EmbeddingBackend = build_embedding_backend(
            args.embedding_backend, self.tables
        )
        set_dev_mode(True)
        self.module = DlrmV3TrainModule(
            self.hstu,
            self.tables,
            self.layout,
            kernel=HammerKernel.PYTORCH,
        )

    def build(self, trace_batch: TokenBudgetBatch) -> DLRMv3TPUModule:
        cpu = torch.device("cpu")
        prepare_backend(self.backend, self.module, self.geom)
        setup = self.backend.setup(
            self.module,
            list(self.tables.values()),
            self.cfg.sparse_optimizer(),
            cpu,
            jax.device_count(),
        )
        assert isinstance(setup.model, DlrmV3TrainModule)
        model = setup.model
        sparse_rt = setup.sparse_runtime
        # One warmup controller for both sides: the in_graph
        # sparse optimizer reads its own LR buffer; the cpu backend's fused
        # optimizer gets a plain one that DLRMv3TPUModule pushes before each step.
        sparse_lr_buffer = sparse_rt.lr_buffer
        if sparse_lr_buffer is None and sparse_rt.fused_optimizer is not None:
            sparse_lr_buffer = torch.zeros((), dtype=torch.float32, device=cpu)
        dense_opt, warmup = setup_dense_optimizer_and_warmup(
            model,
            list(setup.dense_named_parameters.values()),
            self.cfg.dense_optimizer(),
            cpu,
            sparse_cfg=sparse_rt.config if sparse_lr_buffer is not None else None,
            sparse_lr_buffer=sparse_lr_buffer,
        )
        dispatchers = make_dispatchers(self.backend, model, self.geom)
        model.train()
        spec = RecirCompileSpec(
            mode=RecirMode.GSPMD_TPU,
            device=cpu,
            num_ranks=jax.device_count(),
            num_processes=self.world,
            loss_reduction="mean",
            # Dropout draws its key from the in-graph warmup counter because
            # jax.random cannot run inside the traced step.
            region_transforms=[DropoutPRNGTransform(counter_fqn=LR_ITER_BUFFER)],
            # in_graph tables are row-sharded over all chips; nothing else is.
            fsdp_table_substr=(
                SHARDED_TABLE_PREFIX
                if isinstance(self.backend, InGraphEmbeddingBackend)
                else None
            ),
        )
        return DLRMv3TPUModule(
            model,
            dense_opt,
            sparse_rt.optimizer,
            warmup,
            dispatchers,
            spec,
            trace_batch,
            self.world,
            sparse_rt.fused_optimizer,
            sparse_lr_buffer,
        )


def _evaluate(
    trainer: TPUTrainer,
    train: DLRMv3TPUModule,
    loader: DataLoader,
    metrics: Any,
    num_batches: Optional[int],
) -> Dict[str, float]:
    metrics.reset(mode="eval")
    for i, batch in enumerate(loader):
        named = train.evaluate(batch)
        update_metrics(metrics, named, trainer.layout.num_candidates, "eval")
        if num_batches is not None and i + 1 >= num_batches:
            break
    return log_metrics(metrics, "eval")


def _save(train: DLRMv3TPUModule, path: str) -> None:
    dcp.save({"train": train.make_checkpointable()}, checkpoint_id=path)
    logger.info("[ckpt] saved %s", path)


def _load(train: DLRMv3TPUModule, path: str) -> int:
    """Restore in place and return the warmup iteration."""
    train.prepare_for_restore()
    dcp.load({"train": train.make_checkpointable()}, checkpoint_id=path)
    train.finalize_restore()
    logger.info("[ckpt] loaded %s (step %d)", path, train._warmup.iter)
    return train._warmup.iter


def main(argv: Optional[Sequence[str]] = None) -> None:  # noqa: C901
    tpu_runtime.setup_logging()
    cfg, args = _parse(argv)
    tpu_runtime.resolve_dist_env()
    tpu_runtime.init_jax_distributed()
    rank, world = tpu_runtime.ensure_gloo_process_group()
    # Metric all-gathers run on the gloo PG; keep barrier off the TPU accelerator.
    tpu_runtime.force_cpu_barrier()
    torch.manual_seed(cfg.seed)
    trainer = TPUTrainer(cfg, args, world)
    if (cfg.ckpt_dir or cfg.resume_from) and not isinstance(
        trainer.backend, InGraphEmbeddingBackend
    ):
        # RecirCheckpointable covers the in_graph backend;
        # the cpu backend's torchrec fused optimizer state is not part of it.
        raise NotImplementedError("checkpointing needs --embedding_backend in_graph")

    dataset = make_dataset(cfg.dataset, cfg.data_root, trainer.hstu, trainer.tables)
    geom = trainer.geom
    collate = TokenBudgetCollate(
        trainer.layout, trainer.hstu, geom.real_rows, geom.local_chips
    )
    train_dl, eval_dl = make_train_eval_loaders(
        dataset,
        collate,
        geom.real_rows * geom.local_chips,
        cfg.train_split,
        cfg.num_workers,
        world,
        rank,
        cfg.seed,
        pin_memory=False,
    )
    trace_batch: TokenBudgetBatch = next(iter(train_dl))
    train = trainer.build(trace_batch)
    t0 = time.perf_counter()
    train.compiled_step  # noqa: B018 -- compile the train and eval steps now
    logger.info("train step compiled in %.1fs", time.perf_counter() - t0)
    metrics = make_metrics(
        trainer.hstu,
        trainer.layout.rows(geom.real_rows)
        * geom.local_chips
        * trainer.layout.num_candidates,
        torch.device("cpu"),
        rank,
        cfg.tb_dir if rank == 0 else "",
    )
    step = _load(train, cfg.resume_from) if cfg.resume_from else 0

    def maybe_eval(force: bool = False) -> None:
        if len(eval_dl) == 0:
            return
        if force or (cfg.eval_every and step % cfg.eval_every == 0):
            _evaluate(trainer, train, eval_dl, metrics, cfg.num_eval_batches)

    profiling = False
    t_last = time.perf_counter()
    done = False
    for epoch in range(cfg.num_epochs):
        train_dl.sampler.set_epoch(epoch)  # pyre-ignore[16]
        for batch in train_dl:
            if args.profile_dir and step == args.profile_start_step:
                jax.profiler.start_trace(args.profile_dir)
                profiling = True
            losses, named = train(batch)
            step += 1
            if profiling and step == args.profile_end_step:
                jax.profiler.stop_trace()
                profiling = False
            update_metrics(metrics, named, trainer.layout.num_candidates, "train")
            if step % cfg.log_every == 0:
                log_metrics(metrics, "train", losses, trainer.task_names)
                now = time.perf_counter()
                if rank == 0:
                    logger.info(
                        "epoch %d step %d loss %.5f (%.1f ms/step)",
                        epoch,
                        step,
                        float(losses.sum()),
                        1000.0 * (now - t_last) / cfg.log_every,
                    )
                t_last = now
            if cfg.ckpt_dir and cfg.ckpt_every and step % cfg.ckpt_every == 0:
                _save(train, os.path.join(cfg.ckpt_dir, f"step_{step}"))
            maybe_eval()
            done = cfg.num_train_batches is not None and step >= cfg.num_train_batches
            if done:
                break
        if done:
            break
    if profiling:
        jax.profiler.stop_trace()
    maybe_eval(force=True)
    if cfg.ckpt_dir:
        _save(train, os.path.join(cfg.ckpt_dir, f"step_{step}"))
    if metrics.tb_logger is not None:
        metrics.tb_logger.flush()
    tpu_runtime.shutdown()


if __name__ == "__main__":
    main()
