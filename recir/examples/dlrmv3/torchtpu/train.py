# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DLRMv3 training on TorchTPU with recir (DataParallel + TorchTPUBackend).

One process per TPU device, as on GPU: each process runs the dense model on its
own device and gradients are averaged with explicit collectives. The embedding
tables sit behind a graph break: by default in torchrec on the host
(``recir.embedding`` cpu backend, row-wise sharded over the processes with
gloo), or with ``--embedding_backend sparsecore`` on the TPUs' SparseCores
(``recir.embedding.torchtpu_sparsecore``).

    python -m recir.examples.dlrmv3.torchtpu.train --dataset debug --num_train_batches 10

Started without torchrun, the module relaunches itself under torchrun with
``--nproc_per_node`` processes (default 8: every device of a v7x host).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# TorchTPU's launch environment has to be set before its runtime loads, so torch
# must not autoload it.
if __name__ == "__main__" and os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.execvpe(sys.orig_argv[0], sys.orig_argv, os.environ)

# JAX is imported for the gr op bootstrap but must leave the TPU to TorchTPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree as pytree
from generative_recommenders.common import HammerKernel, set_dev_mode
from recir.backends.torchtpu import TorchTPUBackend
from recir.distributed.parallel_mode import DataParallel
from recir.embedding.backend import EmbeddingBackend
from recir.embedding.cpu import CPUSparseArchDispatcher
from recir.embedding.torchtpu_sparsecore import TorchTPUSparseCoreBackend
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
from recir.examples.dlrmv3.common.steps import flatten_batch
from recir.examples.dlrmv3.tpu.embedding_backends import (
    build_embedding_backend,
    SPARSE_ARCH_OVERRIDE_ID,
)
from recir.training import (  # tpu_runtime also registers gr/fbgemm ops
    torchtpu_runtime,
    tpu_runtime,
)
from recir.training.compile import compile_step, recir_train_step
from recir.training.compiled_module import RecirCompiledModuleBase
from recir.training.setup import setup_dense_optimizer_and_warmup

logger: logging.Logger = logging.getLogger(__name__)

_MODULE = "recir.examples.dlrmv3.torchtpu.train"


def _place_dense(model: nn.Module, device: torch.device, host_prefix: str) -> None:
    """Move every parameter and buffer outside ``host_prefix`` to ``device``.

    A TorchTPU tensor cannot be swapped into a CPU parameter (``param.data =``),
    so each parameter is replaced by a new one; ``moved`` keeps shared
    parameters shared.
    """
    moved: Dict[int, nn.Parameter] = {}
    for name, module in model.named_modules():
        if name == host_prefix or name.startswith(host_prefix + "."):
            continue
        for key, param in list(module._parameters.items()):
            if param is None:
                continue
            if id(param) not in moved:
                moved[id(param)] = nn.Parameter(
                    param.detach().to(device), requires_grad=param.requires_grad
                )
            module._parameters[key] = moved[id(param)]
        for key, buf in list(module._buffers.items()):
            if buf is not None:
                module._buffers[key] = buf.to(device)


class DLRMv3TorchTPUModule(RecirCompiledModuleBase):
    """The compiled DLRMv3 train step on one TorchTPU device per process."""

    def __init__(
        self,
        model: nn.Module,
        dense_opt: torch.optim.Optimizer,
        dispatchers: List[Any],
        parallel_mode: DataParallel,
        trace_batch: TokenBudgetBatch,
        fused_opt: Any,
        sparse_lr_buffer: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self._model = model
        self._dense_opt = dense_opt
        self._dispatchers = dispatchers
        self._parallel_mode = parallel_mode
        self._trace_batch = trace_batch
        self._fused_opt = fused_opt
        self._sparse_lr_buffer = sparse_lr_buffer

    def _build_compiled_step(self, first_batch: Any) -> Any:
        return compile_step(
            recir_train_step,
            self._model,
            (self._dense_opt,),
            self._trace_batch,
            parallel_mode=self._parallel_mode,
            backend=TorchTPUBackend(),
            dispatchers=self._dispatchers,
            flatten_batch=flatten_batch,
        )

    def _run_step(
        self, compiled_step: Any, batch: TokenBudgetBatch
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._push_sparse_lr()
        result = compiled_step(self._model, self._dense_opt, *batch.flat())
        # The host metric pipeline reads CPU tensors.
        return pytree.tree_map_only(torch.Tensor, lambda t: t.detach().cpu(), result)

    def _push_sparse_lr(self) -> None:
        """Send the current sparse LR into the host tables' fused optimizer."""
        if self._fused_opt is None or self._sparse_lr_buffer is None:
            return
        lr = float(self._sparse_lr_buffer.item())
        for group in self._fused_opt.param_groups:
            group["lr"] = lr
        self._fused_opt.zero_grad()


class TorchTPUTrainer:
    def __init__(
        self,
        cfg: DLRMv3TrainConfig,
        world: int,
        device: torch.device,
        embedding_backend: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.world = world
        self.device = device
        self.hstu = cfg.hstu_config()
        self.tables = cfg.embedding_tables()
        self.layout = cfg.layout(self.hstu)
        self.task_names = [t.task_name for t in self.hstu.multitask_configs]
        self.backend: EmbeddingBackend
        # Where the backend keeps the tables: torchrec on the host, or SparseCore.
        self.table_device: torch.device
        if embedding_backend == "sparsecore":
            self.backend = TorchTPUSparseCoreBackend(EMBEDDING_COLLECTION_FQN)
            self.table_device = device
        else:
            self.backend = build_embedding_backend("cpu", self.tables)
            self.table_device = torch.device("cpu")
        set_dev_mode(True)
        self.module = DlrmV3TrainModule(
            self.hstu,
            self.tables,
            self.layout,
            kernel=HammerKernel.PYTORCH,
        )

    def build(self, trace_batch: TokenBudgetBatch) -> DLRMv3TorchTPUModule:
        setup = self.backend.setup(
            self.module,
            list(self.tables.values()),
            self.cfg.sparse_optimizer(),
            self.table_device,
            self.world,
        )
        assert isinstance(setup.model, DlrmV3TrainModule)
        model = setup.model
        _place_dense(model, self.device, EMBEDDING_COLLECTION_FQN)
        sparse_rt = setup.sparse_runtime
        # The warmup schedule runs in the graph; DLRMv3TorchTPUModule copies
        # the sparse LR it computes into the host tables' fused optimizer.
        sparse_lr_buffer = (
            torch.zeros((), dtype=torch.float32, device=self.device)
            if sparse_rt.fused_optimizer is not None
            else None
        )
        # Placement replaced the dense parameters; the optimizer takes the new ones.
        placed = dict(model.named_parameters())
        dense_opt, _ = setup_dense_optimizer_and_warmup(
            model,
            [placed[name] for name in setup.dense_named_parameters],
            self.cfg.dense_optimizer(),
            self.device,
            sparse_cfg=sparse_rt.config if sparse_lr_buffer is not None else None,
            sparse_lr_buffer=sparse_lr_buffer,
        )
        keys, cached = model.sparse_io_shapes(self.cfg.batch_size)
        dispatchers = [
            CPUSparseArchDispatcher(
                cached_output=cached,
                input_kjt_keys=keys,
                embedding_collection_fqn=EMBEDDING_COLLECTION_FQN,
                override_id=SPARSE_ARCH_OVERRIDE_ID,
                trace_device=self.device,
            )
        ]
        model.train()
        # DlrmV3TrainModule normalizes its losses per row, so each process's
        # gradient is a local mean and the processes' gradients are averaged
        # ("sum" loss reduction -> "avg" gradient reduction).
        parallel_mode = DataParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=self.world,
            num_processes=self.world,
            device_type="tpu",
        )
        return DLRMv3TorchTPUModule(
            model,
            dense_opt,
            dispatchers,
            parallel_mode,
            trace_batch.to(self.device),
            sparse_rt.fused_optimizer,
            sparse_lr_buffer,
        )


def _parse(
    argv: Optional[Sequence[str]],
) -> Tuple[DLRMv3TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument(
        "--embedding_backend", choices=("cpu", "sparsecore"), default="cpu"
    )
    return parse_config(argv, parser)


def main(argv: Optional[Sequence[str]] = None) -> None:
    tpu_runtime.setup_logging()
    cli = list(sys.argv[1:] if argv is None else argv)
    cfg, args = _parse(cli)
    if not torchtpu_runtime.under_torchrun():
        torchtpu_runtime.relaunch_under_torchrun(_MODULE, cli, args.nproc_per_node)
        return
    if cfg.ckpt_dir or cfg.resume_from or cfg.eval_every:
        raise NotImplementedError(
            "checkpointing and evaluation are not wired up for TorchTPU yet"
        )
    torchtpu_runtime.load_runtime()
    dist.init_process_group("cpu:gloo,tpu:tpu")
    rank, world = dist.get_rank(), dist.get_world_size()
    # Metric all-gathers run on gloo; keep barrier off the TPU.
    tpu_runtime.force_cpu_barrier()
    device = torch.device("tpu")
    torch.manual_seed(cfg.seed)
    trainer = TorchTPUTrainer(cfg, world, device, args.embedding_backend)

    dataset = make_dataset(cfg.dataset, cfg.data_root, trainer.hstu, trainer.tables)
    collate = TokenBudgetCollate(trainer.layout, trainer.hstu, cfg.batch_size, 1)
    train_dl, _ = make_train_eval_loaders(
        dataset,
        collate,
        cfg.batch_size,
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
    train._get_or_build_compiled_step(trace_batch)
    logger.info(
        "[rank %d] train step compiled in %.1fs", rank, time.perf_counter() - t0
    )
    # The random dataset draws its batches from the global RNG, which setting up
    # the embedding backend uses differently per backend; reseeding gives every
    # backend the same training batches.
    torch.manual_seed(cfg.seed)
    metrics = make_metrics(
        trainer.hstu,
        trainer.layout.rows(cfg.batch_size) * trainer.layout.num_candidates,
        torch.device("cpu"),
        rank,
        cfg.tb_dir if rank == 0 else "",
    )

    step = 0
    t_last = time.perf_counter()
    done = False
    for epoch in range(cfg.num_epochs):
        train_dl.sampler.set_epoch(epoch)  # pyrefly: ignore[missing-attribute]
        for batch in train_dl:
            losses, named = train(batch.to(device))
            step += 1
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
            done = cfg.num_train_batches is not None and step >= cfg.num_train_batches
            if done:
                break
        if done:
            break
    if metrics.tb_logger is not None:
        metrics.tb_logger.flush()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
