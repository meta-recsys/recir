# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""DLRMv3 training on TPU with recir, torch_tpu, and SparseCore embeddings."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib
import inspect
import logging
import os
import queue
import threading
import time
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.distributed as dist
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.dlrm_v3.utils import MetricsLogger
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from recir.embedding.torchtpu_sparsecore import (
    TorchTPUSparseCoreBackedEC,
    _ensure_oss_torchrec_tpu_shims,
    install_sparsecore_embeddings,
)
from recir.examples.dlrmv3.common import model as dlrmv3_model
from recir.examples.dlrmv3.common.config import DLRMv3TrainConfig, parse_config
from recir.examples.dlrmv3.common.data import (
    TokenBudgetBatch,
    TokenBudgetCollate,
    make_dataset,
    make_train_eval_loaders,
)
from recir.examples.dlrmv3.common.metrics import make_metrics, update_metrics
from recir.examples.dlrmv3.common.model import DlrmV3TrainModule, FeatureLayout
from recir.examples.dlrmv3.common.steps import eval_step
from recir.training import tpu_runtime
from recir.training.compile import compile_torchtpu_step, current_world_rank
from recir.training.optim import prime_optimizer_state
from recir.training.setup import setup_dense_optimizer_and_warmup
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


def _load_torchrec_tpu_train_types() -> Tuple[Any, Any, Any]:
    """Lazily import torchrec.experimental.torch_tpu train helpers."""
    _ensure_oss_torchrec_tpu_shims()
    dl_mod = importlib.import_module(
        "torchrec.experimental.torch_tpu.datasets.dataloader"
    )
    inp_mod = importlib.import_module(
        "torchrec.experimental.torch_tpu.datasets.input_preprocessing"
    )
    sc_utils = importlib.import_module(
        "torchrec.experimental.torch_tpu.modules.utils"
    )
    return (
        dl_mod.SparseCoreBatch,
        inp_mod.KeyedSparseCorePreprocessedInput,
        sc_utils,
    )


def _tpu_all_gather_dim(
    tensor: torch.Tensor, dim: int, group: Optional[Any] = None
) -> torch.Tensor:
    return getattr(torch, "tpu").ops.all_gather_dim(tensor, dim, group)


def _make_dlrmv3_dataset(
    cfg: DLRMv3TrainConfig,
    hstu: DlrmHSTUConfig,
    tables: Any,
    layout: FeatureLayout,
) -> Any:
    fn: Any = make_dataset
    if len(inspect.signature(make_dataset).parameters) >= 5:
        return fn(cfg.dataset, cfg.data_root, hstu, tables, layout)
    return fn(cfg.dataset, cfg.data_root, hstu, tables)


@dataclasses.dataclass(frozen=True)
class _AlignedFeatureLayout(FeatureLayout):
    row_multiple: int = 1

    def dummy_rows(self, real_rows: int) -> int:
        base = super().dummy_rows(real_rows)
        rem = (real_rows + base) % self.row_multiple
        return base + (self.row_multiple - rem) % self.row_multiple


class _DlrmV3SparseCoreModule(DlrmV3TrainModule):
    """DlrmV3TrainModule binding trailing SparseCore CSR tensors in forward."""

    def forward(
        self,
        uih_seq: torch.Tensor,
        uih_lengths: torch.Tensor,
        ctx: torch.Tensor,
        cand: torch.Tensor,
        cand_weight: torch.Tensor,
        *sc_tensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sc_ec: Any = self.get_submodule(dlrmv3_model.EMBEDDING_COLLECTION_FQN)
        with sc_ec.bind_step_inputs(sc_tensors):
            return super().forward(
                uih_seq, uih_lengths, ctx, cand, cand_weight
            )


def _resolve_hstu_config(cfg: DLRMv3TrainConfig) -> DlrmHSTUConfig:
    hstu = cfg.hstu_config()
    if cfg.num_candidates:
        hstu.max_num_candidates = cfg.num_candidates
    if cfg.uih_len:
        contextual = sum(
            dict(hstu.contextual_feature_to_max_length or {}).values()
        )
        hstu.max_seq_len = cfg.uih_len + hstu.max_num_candidates + contextual
    return hstu


def preprocess_sc_batch(
    batch: TokenBudgetBatch,
    module: DlrmV3TrainModule,
    sc_ec: TorchTPUSparseCoreBackedEC,
) -> Any:
    """Preprocess a CPU TokenBudgetBatch into a SparseCoreBatch."""
    SparseCoreBatch, _, _ = _load_torchrec_tpu_train_types()
    kjt = KeyedJaggedTensor.concat(
        list(
            module._kjts(
                batch.uih_seq, batch.uih_lengths, batch.ctx, batch.cand
            )
        )
    )
    sc_inputs = sc_ec.preprocessor(kjt)
    for t in sc_inputs.table_tensors.values():
        t.lengths = {}
    return SparseCoreBatch(
        dense_features=batch,
        sparse_features=sc_inputs,
        labels=None,
    )


def _batch_gather_tpu_tensors(
    tensors: Sequence[torch.Tensor],
    group: Optional[Any] = None,
) -> List[List[torch.Tensor]]:
    """Coalesce uniform-dtype metric state gathers into one TPU all_gather."""
    if not tensors:
        return []
    pg = group if group is not None else dist.group.WORLD
    if dist.get_world_size(pg) <= 1:
        return [[t] for t in tensors]
    if any(t.dtype != tensors[0].dtype for t in tensors):
        return [
            list(
                _tpu_all_gather_dim(t.to("tpu").unsqueeze(0), 0, pg)
                .to(t.device)
                .unbind(0)
            )
            for t in tensors
        ]
    flat_tpu = torch.cat([t.reshape(-1) for t in tensors]).to("tpu")
    gathered_cpu = _tpu_all_gather_dim(flat_tpu.unsqueeze(0), 0, pg).to("cpu")
    world_size = gathered_cpu.shape[0]
    sizes = [t.numel() for t in tensors]
    per_rank = [torch.split(gathered_cpu[r], sizes) for r in range(world_size)]
    return [
        [
            per_rank[r][i].reshape(orig.shape).to(orig.device)
            for r in range(world_size)
        ]
        for i, orig in enumerate(tensors)
    ]


def log_torchtpu_metrics(
    metrics: MetricsLogger,
    mode: str,
    losses: Optional[torch.Tensor] = None,
    task_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Sync and compute global TorchMetrics over ProcessGroupTpu."""
    computed: Dict[str, float] = {}
    mode_metrics = metrics.all_metrics[mode]
    use_batch_tpu_gather = (
        dist.is_initialized()
        and dist.get_world_size() > 1
        and str(dist.get_backend()) in ("tpu", "tpu_dist")
        and any(
            m._to_sync and m.dist_sync_fn is not None
            for m in mode_metrics
        )
    )

    gathered_iter = None
    if use_batch_tpu_gather:
        collected: List[torch.Tensor] = []

        def _collect_state(
            t: torch.Tensor, group: Optional[Any] = None
        ) -> List[torch.Tensor]:
            del group
            collected.append(t)
            return [t]

        for metric in mode_metrics:
            if not (metric._to_sync and metric.dist_sync_fn is not None):
                continue
            with metric.sync_context(
                dist_sync_fn=_collect_state,
                should_sync=True,
                should_unsync=True,
            ):
                pass
        if collected:
            gathered_iter = iter(_batch_gather_tpu_tensors(collected))

    def _replay_state(
        t: torch.Tensor, group: Optional[Any] = None
    ) -> List[torch.Tensor]:
        del t, group
        assert gathered_iter is not None
        return next(gathered_iter)

    for metric in mode_metrics:
        use_replay = (
            gathered_iter is not None
            and metric._to_sync
            and metric.dist_sync_fn is not None
        )
        sync_fn = _replay_state if use_replay else metric.dist_sync_fn
        with metric.sync_context(
            dist_sync_fn=sync_fn,
            should_sync=metric._to_sync,
            should_unsync=metric._should_unsync,
        ):
            reports = (
                metric.local_compute()
                if (metric._my_rank == 0 or metric._compute_on_all_ranks)
                else []
            )
            for report in reports:
                vals = report.value.detach().cpu().clone()
                for i, task_name in enumerate(metrics.task_names):
                    key = f"metric/{report.metric_prefix}{report.name}/{task_name}"
                    computed[key] = float(vals[i].item())
    if metrics.tb_logger is not None:
        step = metrics.global_step[mode]
        for k, v in computed.items():
            metrics.tb_logger.add_scalar(f"{mode}/{k}", v, global_step=step)
        if losses is not None and task_names is not None:
            for i, name in enumerate(task_names):
                metrics.tb_logger.add_scalar(
                    f"{mode}/losses/{name}",
                    float(losses[i].detach().cpu().item()),
                    global_step=step,
                )
    return computed


class DLRMv3PrefetchLoader:
    """Background host SparseCore CSR preprocessing + async pinned H2D staging."""

    def __init__(
        self,
        dataloader: Iterable[TokenBudgetBatch],
        module: DlrmV3TrainModule,
        sc_ec: TorchTPUSparseCoreBackedEC,
        device: torch.device,
        max_steps: Optional[int] = None,
        queue_size: int = 3,
    ) -> None:
        self._dataloader = dataloader
        self._module = module
        self._sc_ec = sc_ec
        self._device = device
        self._max_steps = max_steps
        self._queue_size = queue_size

    def __iter__(self) -> Generator[Any, None, None]:
        SparseCoreBatch, KeyedSparseCorePreprocessedInput, _ = (
            _load_torchrec_tpu_train_types()
        )
        q: queue.Queue[Any] = queue.Queue(maxsize=self._queue_size)
        stop = threading.Event()
        ring_depth = max(8, self._queue_size + 5)

        def _worker() -> None:
            try:
                if self._device.type == "tpu":
                    torch.accelerator.set_device_index(self._device.index or 0)
                pinned_ring: List[Optional[List[torch.Tensor]]] = [
                    None
                ] * ring_depth
                for idx, batch in enumerate(self._dataloader):
                    if (
                        self._max_steps is not None and idx >= self._max_steps
                    ) or stop.is_set():
                        break
                    prepared = preprocess_sc_batch(
                        batch, self._module, self._sc_ec
                    )
                    if self._device.type == "tpu":
                        prep_dense: Any = prepared.dense_features
                        cpu_flat = (
                            *prep_dense.flat(),
                            *self._sc_ec.flatten_preprocessed(
                                prepared.sparse_features
                            ),
                        )
                        slot = idx % ring_depth
                        slot_bufs = pinned_ring[slot]
                        if slot_bufs is None or len(slot_bufs) != len(cpu_flat):
                            slot_bufs = [
                                torch.empty_like(t).pin_memory(self._device)
                                if t.numel() > 0
                                else torch.empty_like(t)
                                for t in cpu_flat
                            ]
                            pinned_ring[slot] = slot_bufs
                        for dst, src in zip(slot_bufs, cpu_flat):
                            if src.numel() > 0:
                                dst.copy_(src)
                        tpu_flat = [
                            dst.to(self._device, non_blocking=True)
                            for dst in slot_bufs
                        ]
                        table_tensors = {}
                        offset = 5
                        for s in self._sc_ec.sc_ec.stacked_configs:
                            orig = prepared.sparse_features.table_tensors[
                                s.stack_name
                            ]
                            table_tensors[s.stack_name] = dataclasses.replace(
                                orig,
                                row_pointers=tpu_flat[offset],
                                embedding_ids=tpu_flat[offset + 1],
                                sample_ids=tpu_flat[offset + 2],
                                gains=tpu_flat[offset + 3],
                            )
                            offset += 4
                        prepared = SparseCoreBatch(
                            dense_features=TokenBudgetBatch(*tpu_flat[:5]),
                            sparse_features=KeyedSparseCorePreprocessedInput(
                                table_tensors
                            ),
                            labels=None,
                        )
                    while not stop.is_set():
                        try:
                            q.put(prepared, timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except Exception as exc:
                while not stop.is_set():
                    try:
                        q.put(exc, timeout=0.1)
                        break
                    except queue.Full:
                        continue
            finally:
                q.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                dense: Any = item.dense_features
                yield (
                    item
                    if dense.uih_seq.device == self._device
                    else item.to(self._device, non_blocking=True)
                )
        finally:
            stop.set()
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            thread.join(timeout=5.0)


class DLRMv3TPUTrainer:
    """Builds and compiles DlrmV3TrainModule with SparseCore embeddings."""

    def __init__(
        self,
        cfg: DLRMv3TrainConfig,
        rank: int = 0,
        world: int = 1,
    ) -> None:
        _, _, sc_utils = _load_torchrec_tpu_train_types()
        if cfg.dataset == "debug" and cfg.uih_len == 0:
            cfg = dataclasses.replace(
                cfg,
                uih_len=256,
                uih_budget=cfg.uih_budget or 128,
            )
        self.cfg = cfg
        self.rank = rank
        self.world = world
        self.device = torch.device("tpu")
        self.hstu = _resolve_hstu_config(cfg)
        self.tables = cfg.embedding_tables()
        self.layout: FeatureLayout = _AlignedFeatureLayout(
            **dataclasses.asdict(cfg.layout(self.hstu)),
            row_multiple=sc_utils.get_num_sc_per_device() or 2,
        )
        self.task_names = [t.task_name for t in self.hstu.multitask_configs]
        set_dev_mode(True)
        torch.manual_seed(cfg.seed)
        self.module = _DlrmV3SparseCoreModule(
            self.hstu,
            self.tables,
            self.layout,
            kernel=HammerKernel.PYTORCH,
        )
        self.sc_module: Optional[TorchTPUSparseCoreBackedEC] = None
        self.dense_opt: Optional[torch.optim.Optimizer] = None
        self._trace_batch: Optional[Any] = None
        self._train_runner: Any = None
        self._eval_runner: Any = None

    def compile(self, raw_trace_batch: TokenBudgetBatch) -> Any:
        """Install SparseCore tables, move dense state to TPU, and compile steps."""
        fqn = dlrmv3_model.EMBEDDING_COLLECTION_FQN
        sparse_cfg = self.cfg.sparse_optimizer()
        sc_mod = install_sparsecore_embeddings(
            self.module,
            fqn,
            list(self.tables.values()),
            sparse_cfg,
            rows=self.layout.rows(self.cfg.batch_size),
            feature_totals=self.layout.feature_totals(self.cfg.batch_size),
            world_size=self.world,
            device=self.device,
        )
        self.sc_module = sc_mod
        sc_mod.update_preprocessing_parameters(
            KeyedJaggedTensor.concat(
                list(
                    self.module._kjts(
                        raw_trace_batch.uih_seq,
                        raw_trace_batch.uih_lengths,
                        raw_trace_batch.ctx,
                        raw_trace_batch.cand,
                    )
                )
            )
        )
        dense_params = [
            p
            for name, p in self.module.named_parameters()
            if not name.startswith(f"{fqn}.")
        ]
        self.dense_opt, _ = setup_dense_optimizer_and_warmup(
            self.module,
            dense_params,
            self.cfg.dense_optimizer(),
            self.device,
            sparse_cfg=sparse_cfg,
            sparse_lr_buffer=sc_mod.lr_buffer,
        )
        self.module.train()
        prime_optimizer_state(self.dense_opt)
        self._trace_batch = preprocess_sc_batch(
            raw_trace_batch, self.module, sc_mod
        ).to(self.device)
        self._train_runner, self._eval_runner = compile_torchtpu_step(
            self.module,
            self.dense_opt,
            self._trace_batch,
            flatten_batch=lambda b, *, global_loss_denom=False: (
                *b.dense_features.flat(),
                *sc_mod.flatten_preprocessed(b.sparse_features),
            ),
            device=self.device,
            world_size=self.world,
            rank=self.rank,
            eval_step_fn=eval_step,
        )
        return self._train_runner

    def __call__(
        self, batch: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (
            self._train_runner is not None
            and self.dense_opt is not None
            and self.sc_module is not None
        )
        dense: Any = batch.dense_features
        dev_batch = (
            batch
            if dense.uih_seq.device == self.device
            else batch.to(self.device)
        )
        dev_dense: Any = dev_batch.dense_features
        return self._train_runner(
            self.module,
            self.dense_opt,
            *dev_dense.flat(),
            *self.sc_module.flatten_preprocessed(dev_batch.sparse_features),
        )

    def evaluate(self, batch: Any) -> Dict[str, torch.Tensor]:
        assert self._eval_runner is not None and self.sc_module is not None
        dense: Any = batch.dense_features
        dev_batch = (
            batch
            if dense.uih_seq.device == self.device
            else batch.to(self.device)
        )
        dev_dense: Any = dev_batch.dense_features
        _, named = self._eval_runner(
            self.module,
            *dev_dense.flat(),
            *self.sc_module.flatten_preprocessed(dev_batch.sparse_features),
        )
        return named


def _evaluate(
    trainer: DLRMv3TPUTrainer,
    loader: Iterable[TokenBudgetBatch],
    metrics: MetricsLogger,
    num_batches: Optional[int],
) -> Dict[str, float]:
    """Run compiled evaluation batches and sync metrics across TPU ranks."""
    assert trainer.sc_module is not None
    metrics.reset(mode="eval")
    prefetch = DLRMv3PrefetchLoader(
        loader,
        trainer.module,
        trainer.sc_module,
        trainer.device,
        max_steps=num_batches,
    )
    for i, batch in enumerate(prefetch):
        named = trainer.evaluate(batch)
        cpu_named = {k: v.to("cpu") for k, v in named.items()}
        update_metrics(metrics, cpu_named, trainer.layout.num_candidates, "eval")
        if num_batches is not None and i + 1 >= num_batches:
            break
    return log_torchtpu_metrics(metrics, "eval")


def _parse(
    argv: Optional[Sequence[str]],
) -> Tuple[DLRMv3TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc_per_node", type=int, default=8)
    parser.add_argument(
        "--embedding_backend",
        choices=("cpu", "sparsecore"),
        default="sparsecore",
    )
    return parse_config(argv, parser)


def run(
    cfg: DLRMv3TrainConfig,
    *,
    step_hook: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Run end-to-end DLRMv3 compilation, training, and evaluation on TPU."""
    if not dist.is_initialized() and "RANK" in os.environ:
        dist.init_process_group(backend="tpu")
    world, rank = current_world_rank()
    _ = torch.zeros(1, device="tpu")
    trainer = DLRMv3TPUTrainer(cfg, rank=rank, world=world)
    dataset = _make_dlrmv3_dataset(
        cfg, trainer.hstu, trainer.tables, trainer.layout
    )
    collate = TokenBudgetCollate(trainer.layout, trainer.hstu, cfg.batch_size, 1)
    train_dl, eval_dl = make_train_eval_loaders(
        dataset,
        collate,
        cfg.batch_size,
        cfg.train_split,
        0,
        world,
        rank,
        cfg.seed,
        pin_memory=False,
    )
    t0 = time.perf_counter()
    trainer.compile(next(iter(train_dl)))
    assert trainer._trace_batch is not None and trainer.sc_module is not None
    _ = trainer(trainer._trace_batch)
    torch.accelerator.synchronize()
    logger.info(
        "[Rank %d/%d] compiled + warmed up DLRMv3 train step in %.2fs",
        rank,
        world,
        time.perf_counter() - t0,
    )
    torch.manual_seed(cfg.seed)
    metrics = make_metrics(
        trainer.hstu,
        trainer.layout.rows(cfg.batch_size) * trainer.layout.num_candidates,
        torch.device("cpu"),
        rank,
        cfg.tb_dir if rank == 0 else "",
    )
    if dist.is_initialized() and str(dist.get_backend()) in ("tpu", "tpu_dist"):

        def _sync_tpu_metric(
            r: torch.Tensor, group: Optional[Any] = None
        ) -> List[torch.Tensor]:
            g = _tpu_all_gather_dim(r.to("tpu").unsqueeze(0), 0, group)
            return list(g.to(r.device).unbind(0))

        for mode_metrics in metrics.all_metrics.values():
            for m in mode_metrics:
                m.dist_sync_fn = _sync_tpu_metric

    step = 0
    batches_per_epoch = max(1, len(train_dl))
    num_epochs = cfg.num_epochs
    if cfg.num_train_batches is not None:
        req = (cfg.num_train_batches + batches_per_epoch - 1) // batches_per_epoch
        num_epochs = max(num_epochs, req)
    total_steps = cfg.num_train_batches or (num_epochs * batches_per_epoch)
    pending_logs: Deque[
        Tuple[int, int, torch.Tensor, Dict[str, torch.Tensor]]
    ] = collections.deque()
    pinned_named_bufs: Dict[str, torch.Tensor] = {}
    t_last = time.perf_counter()

    def _flush_log(
        logged_step: int,
        logged_epoch: int,
        det_losses: torch.Tensor,
        det_named: Dict[str, torch.Tensor],
    ) -> None:
        nonlocal t_last
        for k, v in det_named.items():
            buf = pinned_named_bufs.get(k)
            if buf is None or buf.shape != v.shape or buf.dtype != v.dtype:
                cpu_base = torch.empty(v.shape, dtype=v.dtype, device="cpu")
                buf = (
                    cpu_base.pin_memory(trainer.device)
                    if v.device.type == "tpu" and v.numel() > 0
                    else cpu_base
                )
                pinned_named_bufs[k] = buf
            buf.copy_(v, non_blocking=True)
        cpu_losses = det_losses.to("cpu")
        cpu_named = {k: pinned_named_bufs[k].clone() for k in det_named}
        update_metrics(metrics, cpu_named, trainer.layout.num_candidates, "train")
        logged = log_torchtpu_metrics(
            metrics, "train", cpu_losses, trainer.task_names
        )
        now = time.perf_counter()
        if rank == 0:
            logger.info(
                "[DLRMv3-TorchTPU] epoch %d step %d/%d loss=%.5f "
                "(%.2f ms/step) metrics=%s",
                logged_epoch,
                logged_step,
                total_steps,
                float(cpu_losses.sum()),
                1000.0 * (now - t_last) / max(1, cfg.log_every),
                {k: round(float(v), 4) for k, v in logged.items()},
            )
        t_last = now

    done = False
    for epoch in range(num_epochs):
        train_dl.sampler.set_epoch(epoch)  # pyrefly: ignore[missing-attribute]
        prefetch = DLRMv3PrefetchLoader(
            train_dl,
            trainer.module,
            trainer.sc_module,
            trainer.device,
            max_steps=(
                max(0, cfg.num_train_batches - step)
                if cfg.num_train_batches is not None
                else None
            ),
        )
        for batch in prefetch:
            while pending_logs and step - pending_logs[0][0] >= 2:
                _flush_log(*pending_logs.popleft())
            if step_hook is not None:
                step_hook(step, total_steps)
            losses, named = trainer(batch)
            step += 1
            if step % cfg.log_every == 0 or step == total_steps:
                pending_logs.append(
                    (
                        step,
                        epoch,
                        losses.detach(),
                        {k: v.detach() for k, v in named.items()},
                    )
                )
            if cfg.eval_every and step % cfg.eval_every == 0 and len(eval_dl) > 0:
                while pending_logs:
                    _flush_log(*pending_logs.popleft())
                _evaluate(trainer, eval_dl, metrics, cfg.num_eval_batches)
            done = (
                cfg.num_train_batches is not None and step >= cfg.num_train_batches
            )
            if done:
                break
        if done:
            break
    while pending_logs:
        _flush_log(*pending_logs.popleft())
    if step_hook is not None:
        step_hook(step, total_steps)
    if metrics.tb_logger is not None:
        metrics.tb_logger.flush()
    torch.accelerator.synchronize()
    if dist.is_initialized():
        dist.destroy_process_group()


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    step_hook: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Parse CLI arguments and run SparseCore DLRMv3 training on TPU."""
    tpu_runtime.setup_logging()
    cfg, _ = _parse(argv)
    run(cfg, step_hook=step_hook)


if __name__ == "__main__":
    main()
