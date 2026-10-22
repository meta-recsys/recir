# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity between eager CPU training and RecIR GSPMD training."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

# Optimizer state includes int64 step counters and bounds. Configure JAX before
# importing it so CPU and TPU lowering preserve those values.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import torch
from recir.embedding.backend import TableOptimizer, TablePlan, TableSharding
from recir.embedding.in_graph import InGraphEmbeddingBackend
from recir.examples.dlrmv3.common.config import DLRMv3TrainConfig
from recir.examples.dlrmv3.common.data import (
    make_dataset,
    TokenBudgetBatch,
    TokenBudgetCollate,
)
from recir.examples.dlrmv3.common.model import EMBEDDING_COLLECTION_FQN
from recir.examples.dlrmv3.tpu.train import DLRMv3TPUModule, TPUTrainer
from recir.training.compile import WARMUP_ATTR
from recir.training.optim import prime_optimizer_state
from recir.training.setup import build_sparse_lr_buffered_optimizer

logger: logging.Logger = logging.getLogger(__name__)


class _DeterministicConfig(DLRMv3TrainConfig):
    def hstu_config(self) -> Any:
        config = super().hstu_config()
        config.hstu_input_dropout_ratio = 0.0
        config.hstu_linear_dropout_rate = 0.0
        return config


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="debug")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument(
        "--sharding",
        choices=("rowwise", "fsdp"),
        default="rowwise",
        help="Use fsdp for CPU-XLA; rowwise exercises the TPU ring-shift kernel.",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--state_atol", type=float)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--batch_file", default="")
    parser.add_argument("--export_batches", default="")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--timestamp", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--target_chips", type=int, default=0)
    return parser.parse_args(argv)


def _backend(trainer: TPUTrainer, sharding: TableSharding) -> InGraphEmbeddingBackend:
    return InGraphEmbeddingBackend(
        embedding_collection_fqn=EMBEDDING_COLLECTION_FQN,
        sparse_optimizer_builder=build_sparse_lr_buffered_optimizer,
        table_plans={
            name: TablePlan(
                sharding=sharding,
                optimizer=TableOptimizer.SPARSE,
            )
            for name in trainer.tables
        },
        on_device_init=False,
    )


def _trainer(
    config: DLRMv3TrainConfig,
    args: argparse.Namespace,
    sharding: TableSharding,
) -> TPUTrainer:
    trainer_args = argparse.Namespace(
        embedding_backend="in_graph",
        profile_dir="",
        profile_start_step=0,
        profile_end_step=0,
    )
    torch.manual_seed(args.seed)
    trainer = TPUTrainer(config, trainer_args, world=1)
    trainer.backend = _backend(trainer, sharding)
    trainer.module.model._bf16_training = not args.fp32
    return trainer


def _batch(trainer: TPUTrainer, seed: int) -> TokenBudgetBatch:
    layout = trainer.layout
    chips = trainer.geom.local_chips
    real_rows = trainer.geom.real_rows
    dummy_rows = layout.dummy_rows(real_rows)
    rows_per_chip = real_rows + dummy_rows
    generator = torch.Generator().manual_seed(seed)

    def ids(*shape: int) -> torch.Tensor:
        return torch.randint(1, trainer.cfg.hash_size, shape, generator=generator)

    lengths_per_chip = torch.tensor(
        [layout.uih_len] * real_rows + [0] * dummy_rows,
        dtype=torch.int64,
    )
    uih_lengths = lengths_per_chip.repeat(chips)
    uih_seq = ids(
        layout.tokens(real_rows) * chips,
        len(layout.seq_keys),
    )
    ctx = ids(
        rows_per_chip * chips,
        sum(layout.contextual_lens),
    )
    cand = ids(
        rows_per_chip * chips,
        len(layout.cand_keys) * layout.num_candidates,
    )
    weights_per_chip = torch.cat(
        [
            torch.ones(real_rows, layout.num_candidates),
            torch.zeros(dummy_rows, layout.num_candidates),
        ]
    )
    return TokenBudgetBatch(
        uih_seq,
        uih_lengths,
        ctx,
        cand,
        weights_per_chip.repeat(chips, 1),
    )


def _remap_embedding_ids(
    batch: TokenBudgetBatch,
    trainer: TPUTrainer,
) -> TokenBudgetBatch:
    feature_rows = {
        feature: table.num_embeddings
        for table in trainer.tables.values()
        for feature in table.feature_names
    }
    layout = trainer.layout
    uih_seq = batch.uih_seq.clone()
    for column, key in enumerate(layout.seq_keys):
        if key in feature_rows:
            uih_seq[:, column].remainder_(feature_rows[key])
    ctx = batch.ctx.clone()
    start = 0
    for key, width in zip(layout.contextual_keys, layout.contextual_lens):
        if key in feature_rows:
            ctx[:, start : start + width].remainder_(feature_rows[key])
        start += width
    cand = batch.cand.clone()
    candidates = layout.num_candidates
    for column, key in enumerate(layout.cand_keys):
        if key in feature_rows:
            cand[:, column * candidates : (column + 1) * candidates].remainder_(
                feature_rows[key]
            )
    return TokenBudgetBatch(
        uih_seq,
        batch.uih_lengths.clone(),
        ctx,
        cand,
        batch.cand_weight.clone(),
    )


def _real_batches(
    trainer: TPUTrainer,
    args: argparse.Namespace,
) -> list[TokenBudgetBatch]:
    dataset = make_dataset(
        trainer.cfg.dataset,
        args.data_root,
        trainer.hstu,
        trainer.tables,
    )
    streaming = dataset.dataset
    set_timestamp = getattr(streaming, "set_ts", None)
    if callable(set_timestamp):
        set_timestamp(args.timestamp)
    elif args.timestamp != 0:
        raise ValueError(f"{trainer.cfg.dataset!r} does not support --timestamp")
    chips = args.target_chips or trainer.geom.local_chips
    samples_per_step = trainer.cfg.batch_size * chips
    needed = samples_per_step * args.steps
    end = args.sample_offset + needed
    if len(dataset) < end:
        raise RuntimeError(
            f"timestamp {args.timestamp} has {len(dataset)} rows, need {end}"
        )
    samples = dataset.__getitems__(list(range(args.sample_offset, end)))
    collate = TokenBudgetCollate(
        trainer.layout,
        trainer.hstu,
        trainer.cfg.batch_size,
        num_blocks=chips,
    )
    return [
        _remap_embedding_ids(
            collate(samples[offset : offset + samples_per_step]),
            trainer,
        )
        for offset in range(0, needed, samples_per_step)
    ]


def _save_batches(path: str, batches: Sequence[TokenBudgetBatch]) -> None:
    torch.save(
        [
            {
                "uih_seq": batch.uih_seq,
                "uih_lengths": batch.uih_lengths,
                "ctx": batch.ctx,
                "cand": batch.cand,
                "cand_weight": batch.cand_weight,
            }
            for batch in batches
        ],
        path,
    )


def _load_batches(path: str) -> list[TokenBudgetBatch]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return [TokenBudgetBatch(**batch) for batch in payload]


def _optimizer_state_by_fqn(
    module: torch.nn.Module,
    optimizers: Iterable[torch.optim.Optimizer],
) -> Dict[Tuple[str, str], Any]:
    result: Dict[Tuple[str, str], Any] = {}
    for fqn, parameter in module.named_parameters():
        for optimizer in optimizers:
            for key, value in optimizer.state.get(parameter, {}).items():
                result[(fqn, str(key))] = value
    return result


def _assert_tree_close(
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
    name: str,
) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{name}: expected a tensor, got {type(actual)}")
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from error
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or actual.keys() != expected.keys():
            raise AssertionError(f"{name}: mapping keys differ")
        for key in expected:
            _assert_tree_close(
                actual[key],
                expected[key],
                atol=atol,
                rtol=rtol,
                name=f"{name}.{key}",
            )
        return
    if actual != expected:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}")


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value


def _training_state(module: DLRMv3TPUModule) -> Dict[str, Any]:
    optimizers = [module._dense_opt, module._emb_opt]
    return {
        "model": _clone_tree(module.model.state_dict()),
        "optimizer": _clone_tree(
            _optimizer_state_by_fqn(
                module.model,
                [optimizer for optimizer in optimizers if optimizer is not None],
            )
        ),
    }


def _replace_eager_adam(compiled_optimizer: torch.optim.Optimizer) -> torch.optim.Adam:
    group = compiled_optimizer.param_groups[0]
    learning_rate = group["lr"]
    if isinstance(learning_rate, torch.Tensor):
        learning_rate = float(learning_rate.item())
    optimizer = torch.optim.Adam(
        group["params"],
        lr=learning_rate,
        betas=group["betas"],
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        foreach=False,
    )
    prime_optimizer_state(optimizer)
    return optimizer


def _train_step_with_gradient_probe(
    module: Any,
    *optimizers_and_batch: Any,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    optimizers = []
    for value in optimizers_and_batch:
        if not isinstance(value, torch.optim.Optimizer):
            break
        optimizers.append(value)
    batch = optimizers_and_batch[len(optimizers) :]
    losses, named = module(*batch)
    losses.sum().backward()
    probe_parameter = module.model._multitask_module._prediction_module[0].weight
    if probe_parameter.grad is None:
        raise RuntimeError("prediction gradient probe was not produced")
    outputs = {key: value.detach() for key, value in named.items()}
    outputs["prediction_weight_gradient"] = probe_parameter.grad.detach()
    for optimizer in optimizers:
        optimizer.step()
    warmup = getattr(module, WARMUP_ATTR, None)
    if warmup is not None:
        warmup.step()
    return losses.detach(), outputs


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse(argv)
    sharding = (
        TableSharding.ROWWISE if args.sharding == "rowwise" else TableSharding.FSDP
    )
    state_atol = args.state_atol
    if state_atol is None:
        state_atol = 1e-4 if args.fp32 else 1e-3
    uses_dataset_batches = args.dataset != "debug"
    config = _DeterministicConfig(
        dataset=args.dataset,
        uih_len=256 if uses_dataset_batches else 16,
        uih_budget=128 if uses_dataset_batches else 0,
        num_candidates=32 if uses_dataset_batches else 4,
        batch_size=2,
        dense_optim="adam",
        sparse_optim="rowwise_adagrad",
        embedding_dim=64,
        num_layers=2 if uses_dataset_batches else 1,
        hash_size=4096 if uses_dataset_batches else 256,
        seed=args.seed,
    )
    compiled_trainer = _trainer(config, args, sharding)
    if args.export_batches:
        batches = _real_batches(compiled_trainer, args)
        _save_batches(args.export_batches, batches)
        logger.info(
            "exported %d real-data batches to %s",
            len(batches),
            args.export_batches,
        )
        return
    eager_trainer = _trainer(config, args, sharding)
    if args.batch_file:
        batches = _load_batches(args.batch_file)
    elif uses_dataset_batches:
        batches = _real_batches(compiled_trainer, args)
    else:
        batches = [_batch(compiled_trainer, args.seed + 1)] * args.steps
    if len(batches) != args.steps:
        raise ValueError(f"batch file has {len(batches)} batches, need {args.steps}")

    torch.manual_seed(args.seed + 2)
    compiled = compiled_trainer.build(batches[0])
    compiled._train_step_fn = _train_step_with_gradient_probe
    torch.manual_seed(args.seed + 2)
    eager = eager_trainer.build(batches[0])
    eager._dense_opt = _replace_eager_adam(eager._dense_opt)
    _assert_tree_close(
        _training_state(compiled),
        _training_state(eager),
        atol=0.0,
        rtol=0.0,
        name="initial state",
    )

    expected_steps = []
    for batch in batches:
        eager._dense_opt.zero_grad(set_to_none=True)
        if eager._emb_opt is not None:
            eager._emb_opt.zero_grad(set_to_none=True)
        expected_losses, expected_named = _train_step_with_gradient_probe(
            eager.model,
            eager._dense_opt,
            eager._emb_opt,
            *batch.flat(),
        )
        expected_steps.append(
            (
                _clone_tree(expected_losses),
                _clone_tree(expected_named),
                _training_state(eager),
            )
        )

    for step, (batch, expected) in enumerate(zip(batches, expected_steps)):
        expected_losses, expected_named, expected_state = expected
        actual_losses, actual_named = compiled(batch)
        compiled.flush_host_state(full=True)
        expected_gradient = expected_named["prediction_weight_gradient"].float()
        actual_gradient = actual_named["prediction_weight_gradient"].float()
        gradient_gap = (actual_gradient - expected_gradient).abs()
        sign_flips = torch.count_nonzero(
            (actual_gradient.sign() != expected_gradient.sign())
            & (actual_gradient != 0)
            & (expected_gradient != 0)
        )
        logger.info(
            "step %d gradient max_abs=%g mean_abs=%g sign_flips=%d/%d",
            step,
            gradient_gap.max().tolist(),
            gradient_gap.mean().tolist(),
            sign_flips.tolist(),
            gradient_gap.numel(),
        )
        loss_gap = (actual_losses.float() - expected_losses.float()).abs()
        logger.info(
            "step %d loss max_abs=%g mean_abs=%g",
            step,
            loss_gap.max().tolist(),
            loss_gap.mean().tolist(),
        )
        for name, expected in expected_named.items():
            gap = (actual_named[name].float() - expected.float()).abs()
            logger.info(
                "step %d output[%s] max_abs=%g mean_abs=%g",
                step,
                name,
                gap.max().tolist(),
                gap.mean().tolist(),
            )
        _assert_tree_close(
            actual_losses,
            expected_losses,
            atol=args.atol,
            rtol=args.rtol,
            name=f"step {step} losses",
        )
        _assert_tree_close(
            actual_named,
            expected_named,
            atol=args.atol,
            rtol=args.rtol,
            name=f"step {step} output",
        )
        _assert_tree_close(
            _training_state(compiled),
            expected_state,
            atol=state_atol,
            rtol=args.rtol,
            name=f"step {step} state",
        )
        logger.info(
            "step %d eager_losses=%s recir_losses=%s",
            step,
            expected_losses.tolist(),
            actual_losses.tolist(),
        )

    logger.info(
        "TPU numerical parity passed: devices=%d sharding=%s steps=%d dtype=%s",
        jax.device_count(),
        args.sharding,
        args.steps,
        "fp32" if args.fp32 else "bf16",
    )
