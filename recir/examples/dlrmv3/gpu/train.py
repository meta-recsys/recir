# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DLRMv3 training on GPUs with recir (HybridParallel + Inductor + FBGEMM TBE).

One process per GPU, launched with torchrun::

    torchrun --nproc_per_node=8 -m recir.examples.dlrmv3.gpu.train \\
        --dataset kuairand-1k --data_root /path/to/data

Each step (forward, backward, dense Adam, sparse RowWiseAdagrad, LR warmup) is
traced into one FX graph. Large embedding tables are row-wise sharded with
all-to-all routing in the graph and lowered to FBGEMM TBE; dense parameters
are replicated with an in-graph gradient all-reduce.

Eval runs a forward-only step compiled together with the train step, so it
reads the training model's sharded parameters directly.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import fbgemm_gpu.sparse_ops  # noqa: F401 -- fake impls of the fbgemm jagged ops
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from generative_recommenders.common import HammerKernel
from recir.distributed.data_parallel import materialize_embedding_weight
from recir.distributed.sharding_plan import EmbeddingShardingPlan
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
from recir.examples.dlrmv3.common.model import (
    DlrmV3TrainModule,
    EMBEDDING_COLLECTION_FQN,
)
from recir.examples.dlrmv3.common.steps import eval_step, flatten_batch
from recir.pipeline import is_pipeline_pending, PipelineConfig
from recir.pipeline_strategies import (
    embedding_steady_state_pipeline_config,
    sdd_pipeline_config,
)
from recir.training.compile import (
    compile_hdp_step,
    create_tbe_process_group,
    RecirCompileSpec,
    RecirMode,
    TBESparseConfig,
)
from recir.training.compiled_module import has_pending_work
from recir.training.embedding_plan import _safe_for_row_wise, build_embedding_plan
from recir.training.gpu_checkpoint import RecirCheckpointAdapter
from recir.training.materialization import (
    materialize_recir_buffers,
    materialize_recir_parameters,
)
from recir.training.setup import RecirOptimizerBundle, setup_dual_optimizers_and_warmup
from recir.utils.meta_tensor_init import default_embedding_init_bound
from torch.utils.data import DataLoader
from torchrec.distributed.embedding_types import OptimType
from torchrec.distributed.types import ShardingType

logger: logging.Logger = logging.getLogger(__name__)

TABLE_PREFIX = f"{EMBEDDING_COLLECTION_FQN}.embeddings."
# Module TBE lowering installs on the compiled module; its rank-local kernel
# state is not checkpointed (the tables stay exposed at their original FQNs).
_TBE_ROOT_PREFIX = "_impl._recir_tbe_sparse_groups."


def _parse(
    argv: Optional[Sequence[str]],
) -> Tuple[DLRMv3TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--row_wise_min_rows", type=int, default=100_000)
    parser.add_argument("--no_tbe", action="store_true", help="plain row-wise lookups")
    parser.add_argument("--pipeline", choices=["none", "sdd", "steady"], default="none")
    return parse_config(argv, parser)


class _Location(Enum):
    HBM = "HBM"


@dataclass
class _PlanTable:
    """Table placement for ``recir.training.embedding_plan.build_embedding_plan``."""

    num_embeddings: int
    embedding_dim: int
    sharding: ShardingType
    location: _Location = _Location.HBM


def _pipeline_config(name: str, world_size: int) -> PipelineConfig:
    if name == "none" or world_size == 1:
        return PipelineConfig.none()
    if name == "sdd":
        return sdd_pipeline_config()
    return embedding_steady_state_pipeline_config()


@dataclass
class _Compiled:
    module: DlrmV3TrainModule
    runner: Callable[..., Any]
    optimizers: Tuple[torch.optim.Optimizer, ...]


class _Checkpoint:
    """DCP save / load through ``recir.training``'s ``RecirCheckpointAdapter``
    plan-sharded tables and their sparse optimizer state are
    saved as ShardedTensors, everything else replicated."""

    def __init__(
        self,
        module: DlrmV3TrainModule,
        plan: EmbeddingShardingPlan,
        device: torch.device,
        world_size: int,
        rank: int,
        cfg: DLRMv3TrainConfig,
        bundle: RecirOptimizerBundle,
    ) -> None:
        # The adapter keys model state as ``_impl.<fqn>`` of the root it hooks.
        self._root = nn.Module()
        self._root._impl = module
        self._adapter = RecirCheckpointAdapter(
            self._root,
            module,
            plan,
            device,
            world_size,
            rank,
            cfg.optimizer_configs(),
        )
        self._adapter.bind_optimizers(
            bundle.dense_opt, bundle.sparse_opt, bundle.warmup
        )
        self._step = torch.zeros((), dtype=torch.int64)

    def _state(self) -> Dict[str, Any]:
        model = {
            k: v
            for k, v in self._root.state_dict().items()
            if not k.startswith(_TBE_ROOT_PREFIX)
        }
        return {
            "model": model,
            "optim": self._adapter.optimizer_state_dict(),
            "step": self._step,
        }

    def save(self, path: str, step: int) -> None:
        self._step.fill_(step)
        dcp.save(self._state(), checkpoint_id=path)
        logger.info("[ckpt] saved %s (step %d)", path, step)

    def load(self, path: str) -> int:
        """Restore in place; returns the saved step."""
        state = self._state()
        dcp.load(state, checkpoint_id=path)
        result = self._root.load_state_dict(state["model"], strict=False)
        unexpected = list(result.unexpected_keys)
        missing = [k for k in result.missing_keys if not k.startswith(_TBE_ROOT_PREFIX)]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: {missing=} {unexpected=}")
        self._adapter.load_optimizer_state_dict(state["optim"])
        step = int(self._step)
        logger.info("[ckpt] loaded %s (step %d)", path, step)
        return step


class GPUTrainer:
    def __init__(self, cfg: DLRMv3TrainConfig, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        self.hstu = cfg.hstu_config()
        self.tables = cfg.embedding_tables()
        self.layout = cfg.layout(self.hstu)
        self.use_tbe = not args.no_tbe
        if (
            self.use_tbe
            and cfg.sparse_optimizer().optim_type != OptimType.ROWWISE_ADAGRAD
        ):
            raise ValueError("FBGEMM TBE supports only --sparse_optim rowwise_adagrad")
        self.tbe_config: Optional[TBESparseConfig] = (
            # Re-trace the whole module around the lookups: its inputs are plain
            # tensors, while the EmbeddingCollection takes a KJT whose static
            # per-key sizes would not survive the re-trace's pytree round trip.
            TBESparseConfig(use_separate_process_group=True, forward_root_fqn="")
            if self.use_tbe
            else None
        )
        self._tbe_pg: Optional[dist.ProcessGroup] = create_tbe_process_group(
            self.tbe_config, self.world_size
        )
        self.task_names = [t.task_name for t in self.hstu.multitask_configs]

    def _new_module(self) -> Tuple[DlrmV3TrainModule, EmbeddingShardingPlan]:
        module = DlrmV3TrainModule(
            self.hstu, self.tables, self.layout, kernel=HammerKernel.TRITON
        )
        plan = build_embedding_plan(
            module,
            set(self.tables),
            self.world_size,
            embedding_tables={
                name: _PlanTable(
                    t.num_embeddings,
                    t.embedding_dim,
                    (
                        ShardingType.ROW_WISE
                        if t.num_embeddings >= self.args.row_wise_min_rows
                        and _safe_for_row_wise(t.num_embeddings, self.world_size)
                        else ShardingType.DATA_PARALLEL
                    ),
                )
                for name, t in self.tables.items()
            },
        )

        # Plan tables stay on meta until the sharding layer materializes each
        # rank's shard; replicated tables get torchrec's default uniform init.
        def materialize_meta_parameter(
            fqn: str, param: nn.Parameter
        ) -> Optional[torch.Tensor]:
            if not fqn.startswith(TABLE_PREFIX):
                return None
            return materialize_embedding_weight(
                param.shape,
                param.dtype,
                self.device,
                init_kind="uniform",
                init_bound=default_embedding_init_bound(param.shape[0]),
            )

        uninitialized = materialize_recir_parameters(
            module, self.device, plan.param_fqns, materialize_meta_parameter
        )
        uninitialized += materialize_recir_buffers(module, self.device)
        if uninitialized:
            raise RuntimeError(f"uninitialized meta tensors: {uninitialized}")
        return module, plan

    def build(
        self, trace_batch: TokenBudgetBatch
    ) -> Tuple[_Compiled, _Compiled, _Checkpoint]:
        """Train and eval steps compiled together: eval reads the train module's
        (sharded, TBE-lowered) parameters directly."""
        module, plan = self._new_module()
        module.train()
        bundle: RecirOptimizerBundle = setup_dual_optimizers_and_warmup(
            module,
            self.cfg.optimizer_configs(),
            plan,
            self.device,
            use_tbe_sparse=self.use_tbe,
        )
        # Before compile: the adapter records each plan table's global shape.
        ckpt = _Checkpoint(
            module,
            plan,
            self.device,
            self.world_size,
            self.rank,
            self.cfg,
            bundle,
        )
        spec = RecirCompileSpec(
            mode=RecirMode.HDP_GPU,
            device=self.device,
            world_size=self.world_size,
            rank=self.rank,
            pipeline=_pipeline_config(self.args.pipeline, self.world_size),
            embedding_plan=plan,
            tbe_sparse_config=self.tbe_config,
            tbe_process_group=self._tbe_pg,
        )
        t0 = time.perf_counter()
        train_runner, eval_runner = compile_hdp_step(
            module,
            bundle.dense_opt,
            bundle.sparse_opt,
            trace_batch.to(self.device),
            spec,
            flatten_batch=flatten_batch,
            eval_step_fn=eval_step,
        )
        logger.info(
            "[rank %d] train + eval steps compiled in %.1fs",
            self.rank,
            time.perf_counter() - t0,
        )
        train = _Compiled(module, train_runner, (bundle.dense_opt, bundle.sparse_opt))
        return train, _Compiled(module, eval_runner, ()), ckpt


def _batches(
    loader: DataLoader, device: torch.device
) -> Iterator[Tuple[torch.Tensor, ...]]:
    for batch in loader:
        yield batch.to(device, non_blocking=True).flat()


def _run_steps(
    compiled: _Compiled, batches: Iterator[Tuple[torch.Tensor, ...]]
) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Run the compiled step over ``batches``, yielding retired step results.

    With cross-batch pipelining a call may return a pending marker; the tail is
    drained with ``last_train_step=True``.
    """
    step = compiled.runner
    args = (compiled.module, *compiled.optimizers)
    for flat in batches:
        result = step(*args, *flat)
        if not is_pipeline_pending(result):
            yield result
    while has_pending_work(step):
        result = step(*args, last_train_step=True)
        if not is_pipeline_pending(result):
            yield result


def evaluate(
    trainer: GPUTrainer,
    evaluator: _Compiled,
    loader: DataLoader,
    metrics: Any,
    num_batches: Optional[int],
) -> Dict[str, float]:
    metrics.reset(mode="eval")
    for i, (_, named) in enumerate(
        _run_steps(evaluator, _batches(loader, trainer.device))
    ):
        update_metrics(metrics, named, trainer.layout.num_candidates, "eval")
        if num_batches is not None and i + 1 >= num_batches:
            break
    return log_metrics(metrics, "eval")


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    cfg, args = _parse(argv)
    dist.init_process_group("nccl")
    trainer = GPUTrainer(cfg, args)
    torch.cuda.set_device(trainer.device)
    torch.manual_seed(cfg.seed)

    dataset = make_dataset(cfg.dataset, cfg.data_root, trainer.hstu, trainer.tables)
    collate = TokenBudgetCollate(trainer.layout, trainer.hstu, cfg.batch_size)
    train_dl, eval_dl = make_train_eval_loaders(
        dataset,
        collate,
        cfg.batch_size,
        cfg.train_split,
        cfg.num_workers,
        trainer.world_size,
        trainer.rank,
        cfg.seed,
        # Citrine C0: pin_memory for faster host-to-device copies.
        pin_memory=True,
    )
    trace_batch: TokenBudgetBatch = next(iter(train_dl))
    train, evaluator, ckpt = trainer.build(trace_batch)
    metrics = make_metrics(
        trainer.hstu,
        trainer.layout.rows(cfg.batch_size) * trainer.layout.num_candidates,
        trainer.device,
        trainer.rank,
        cfg.tb_dir,
    )
    step = ckpt.load(cfg.resume_from) if cfg.resume_from else 0
    if cfg.ckpt_every and _pipeline_config(args.pipeline, trainer.world_size).enabled:
        # A pipelined graph holds in-flight steps between retired ones.
        raise ValueError(
            "--ckpt_every needs --pipeline none; the final checkpoint is taken "
            "after the pipeline drains"
        )

    def maybe_eval(force: bool = False) -> None:
        if len(eval_dl) == 0:
            return
        if force or (cfg.eval_every and step % cfg.eval_every == 0):
            evaluate(trainer, evaluator, eval_dl, metrics, cfg.num_eval_batches)

    t_last = time.perf_counter()
    for epoch in range(cfg.num_epochs):
        train_dl.sampler.set_epoch(epoch)  # pyre-ignore[16]
        batches = _batches(train_dl, trainer.device)
        if cfg.num_train_batches is not None:
            # Stop feeding (rather than break) so _run_steps drains the pipeline.
            batches = itertools.islice(batches, max(cfg.num_train_batches - step, 0))
        for losses, named in _run_steps(train, batches):
            step += 1
            update_metrics(metrics, named, trainer.layout.num_candidates, "train")
            if step % cfg.log_every == 0:
                log_metrics(metrics, "train", losses, trainer.task_names)
                now = time.perf_counter()
                if trainer.rank == 0:
                    logger.info(
                        "epoch %d step %d loss %.5f (%.1f ms/step)",
                        epoch,
                        step,
                        float(losses.sum()),
                        1000.0 * (now - t_last) / cfg.log_every,
                    )
                t_last = now
            if cfg.ckpt_dir and cfg.ckpt_every and step % cfg.ckpt_every == 0:
                ckpt.save(os.path.join(cfg.ckpt_dir, f"step_{step}"), step)
            maybe_eval()
        if cfg.num_train_batches is not None and step >= cfg.num_train_batches:
            break
    maybe_eval(force=True)
    if cfg.ckpt_dir:
        ckpt.save(os.path.join(cfg.ckpt_dir, f"step_{step}"), step)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
