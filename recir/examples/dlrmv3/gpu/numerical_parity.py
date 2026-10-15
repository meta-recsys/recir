# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Numerical parity between eager GPU training and RecIR+Inductor."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.distributed as dist
from recir.examples.dlrmv3.common.config import DLRMv3TrainConfig
from recir.examples.dlrmv3.common.data import (
    make_dataset,
    TokenBudgetBatch,
    TokenBudgetCollate,
)
from recir.examples.dlrmv3.common.steps import flatten_batch
from recir.examples.dlrmv3.gpu.train import GPUTrainer
from recir.pipeline import PipelineConfig
from recir.training.compile import (
    compile_hdp_step,
    recir_train_step,
    RecirCompileSpec,
    RecirMode,
    TBESparseConfig,
)
from recir.training.materialization import materialize_recir_parameters
from recir.training.setup import RecirOptimizerBundle, setup_dual_optimizers_and_warmup

logger: logging.Logger = logging.getLogger(__name__)


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="debug")
    parser.add_argument("--data_root", default=os.path.expanduser("~"))
    parser.add_argument("--timestamp", type=int, default=0)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--uih_len", type=int, default=16)
    parser.add_argument("--uih_budget", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--hash_size", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--loss_atol", type=float, default=2e-5)
    parser.add_argument("--loss_rtol", type=float, default=2e-5)
    parser.add_argument("--output_atol", type=float, default=2e-5)
    parser.add_argument("--output_rtol", type=float, default=2e-5)
    parser.add_argument("--state_atol", type=float, default=1e-4)
    parser.add_argument("--state_rtol", type=float, default=1e-2)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--eager_control", action="store_true")
    parser.add_argument("--report_only", action="store_true")
    parser.add_argument("--rank_consistency", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--tbe", action="store_true")
    return parser.parse_args(argv)


def _remap_embedding_ids(
    batch: TokenBudgetBatch,
    layout: Any,
    tables: Mapping[str, Any],
) -> TokenBudgetBatch:
    """Map only embedding-backed feature columns into the scaled table ranges."""
    feature_rows = {
        feature: table.num_embeddings
        for table in tables.values()
        for feature in table.feature_names
    }
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


def _fixed_batches(
    cfg: DLRMv3TrainConfig,
    hstu: Any,
    tables: Mapping[str, Any],
    layout: Any,
    timestamp: int,
    sample_offset: int,
    steps: int,
) -> list[TokenBudgetBatch]:
    dataset = make_dataset(cfg.dataset, cfg.data_root, hstu, dict(tables))
    streaming = dataset.dataset
    set_timestamp = getattr(streaming, "set_ts", None)
    if callable(set_timestamp):
        set_timestamp(timestamp)
    elif timestamp != 0:
        raise ValueError(f"{cfg.dataset!r} does not support --timestamp")
    needed = cfg.batch_size * steps
    end = sample_offset + needed
    if len(dataset) < end:
        raise RuntimeError(f"timestamp {timestamp} has {len(dataset)} rows, need {end}")
    samples = dataset.__getitems__(list(range(sample_offset, end)))
    collate = TokenBudgetCollate(layout, hstu, cfg.batch_size)
    return [
        _remap_embedding_ids(collate(samples[i : i + cfg.batch_size]), layout, tables)
        for i in range(0, needed, cfg.batch_size)
    ]


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
        assert isinstance(actual, torch.Tensor), f"{name}: expected tensor"
        _assert_tensor_close(actual, expected, atol=atol, rtol=rtol, name=name)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{name}: expected dict"
        assert set(actual) == set(expected), f"{name}: keys differ"
        for key in expected:
            _assert_tree_close(
                actual[key], expected[key], atol=atol, rtol=rtol, name=f"{name}.{key}"
            )
        return
    assert actual == expected, f"{name}: {actual!r} != {expected!r}"


def _assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"{name}: {error}") from error


def _assert_training_state_close(
    actual_module: torch.nn.Module,
    expected_module: torch.nn.Module,
    actual_bundle: RecirOptimizerBundle,
    expected_bundle: RecirOptimizerBundle,
    *,
    compare_sparse_state: bool,
    atol: float,
    rtol: float,
) -> None:
    actual_optimizers = [actual_bundle.dense_opt]
    expected_optimizers = [expected_bundle.dense_opt]
    if compare_sparse_state:
        actual_optimizers.append(actual_bundle.sparse_opt)
        expected_optimizers.append(expected_bundle.sparse_opt)
    actual_state = _optimizer_state_by_fqn(actual_module, actual_optimizers)
    expected_state = _optimizer_state_by_fqn(expected_module, expected_optimizers)
    assert set(actual_state) == set(expected_state)
    for key, expected in expected_state.items():
        _assert_tree_close(
            actual_state[key],
            expected,
            atol=atol,
            rtol=rtol,
            name=f"optimizer.{key[0]}.{key[1]}",
        )

    actual_params = dict(actual_module.named_parameters())
    expected_params = dict(expected_module.named_parameters())
    assert set(expected_params) <= set(actual_params)
    for fqn, expected in expected_params.items():
        _assert_tensor_close(
            actual_params[fqn],
            expected,
            atol=atol,
            rtol=rtol,
            name=fqn,
        )


def _run_rank_consistency(
    runner: Any,
    module: torch.nn.Module,
    bundle: RecirOptimizerBundle,
    batches: Sequence[TokenBudgetBatch],
    device: torch.device,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    history: list[float] = []
    for step, host_batch in enumerate(batches):
        batch = host_batch.to(device)
        losses, _named = runner(
            module,
            bundle.dense_opt,
            bundle.sparse_opt,
            *batch.flat(),
        )
        gathered = [torch.empty_like(losses) for _ in range(world_size)]
        dist.all_gather(gathered, losses)
        for other_rank, other_losses in enumerate(gathered[1:], start=1):
            _assert_tensor_close(
                other_losses,
                gathered[0],
                atol=2e-5,
                rtol=2e-5,
                name=f"step {step} rank {other_rank} losses",
            )
        loss_values = losses.tolist()
        history.append(sum(loss_values))
        if rank == 0:
            logger.info("step %d rank-consistent losses=%s", step, loss_values)

    dense_digest = torch.stack(
        [parameter.detach().float().sum() for parameter in bundle.dense_params]
    )
    gathered_digest = [torch.empty_like(dense_digest) for _ in range(world_size)]
    dist.all_gather(gathered_digest, dense_digest)
    for other_rank, other_digest in enumerate(gathered_digest[1:], start=1):
        _assert_tensor_close(
            other_digest,
            gathered_digest[0],
            atol=1e-5,
            rtol=1e-5,
            name=f"rank {other_rank} dense state",
        )
    if rank == 0:
        window = min(10, len(history))
        logger.info(
            "rank-consistency summary ranks=%d steps=%d first=%g last=%g",
            world_size,
            len(history),
            sum(history[:window]) / window,
            sum(history[-window:]) / window,
        )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _parse(argv)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)

    cfg = DLRMv3TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        uih_len=args.uih_len,
        num_candidates=args.num_candidates,
        uih_budget=args.uih_budget,
        batch_size=args.batch_size,
        dense_optim="adam",
        dense_lr=1e-3,
        dense_betas=[0.95, 0.999],
        sparse_optim="rowwise_adagrad",
        sparse_lr=1e-3,
        eps=1e-8,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        hash_size=args.hash_size,
        seed=args.seed,
    )
    trainer_args = argparse.Namespace(
        row_wise_min_rows=1,
        no_tbe=not args.tbe,
        pipeline="none",
    )
    trainer = GPUTrainer(cfg, trainer_args)
    trainer.hstu.hstu_input_dropout_ratio = 0.0
    trainer.hstu.hstu_linear_dropout_rate = 0.0
    batches = _fixed_batches(
        cfg,
        trainer.hstu,
        trainer.tables,
        trainer.layout,
        args.timestamp,
        args.sample_offset,
        args.steps,
    )

    compiled_module, plan = trainer._new_module()
    if args.fp32:
        compiled_module.model._bf16_training = False
    compiled_bundle = setup_dual_optimizers_and_warmup(
        compiled_module,
        cfg.optimizer_configs(),
        plan,
        device,
        use_tbe_sparse=args.tbe,
    )
    if args.tbe:
        for group in compiled_bundle.sparse_opt.param_groups:
            group["stochastic_rounding"] = False

    spec = RecirCompileSpec(
        mode=RecirMode.HDP_GPU,
        device=device,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        pipeline=PipelineConfig.none(),
        embedding_plan=plan,
        use_legacy_compile=args.legacy,
        tbe_sparse_config=(
            TBESparseConfig(use_separate_process_group=False, forward_root_fqn="")
            if args.tbe
            else None
        ),
    )
    trace_batch = batches[0].to(device)
    runner = (
        recir_train_step
        if args.eager_control
        else compile_hdp_step(
            compiled_module,
            compiled_bundle.dense_opt,
            compiled_bundle.sparse_opt,
            trace_batch,
            spec,
            flatten_batch=flatten_batch,
        )
    )

    if args.rank_consistency:
        _run_rank_consistency(
            runner,
            compiled_module,
            compiled_bundle,
            batches,
            device,
        )
        dist.destroy_process_group()
        return

    eager_module, eager_plan = trainer._new_module()
    if args.fp32:
        eager_module.model._bf16_training = False
    uninitialized = materialize_recir_parameters(
        eager_module,
        device,
        (),
        lambda _fqn, parameter: torch.empty_like(parameter, device=device),
    )
    assert not uninitialized
    eager_state = eager_module.state_dict()
    compiled_state = compiled_module.state_dict()
    eager_module.load_state_dict(
        {key: value for key, value in compiled_state.items() if key in eager_state}
    )
    eager_bundle = setup_dual_optimizers_and_warmup(
        eager_module,
        cfg.optimizer_configs(),
        eager_plan,
        device,
        use_tbe_sparse=False,
    )
    _assert_training_state_close(
        compiled_module,
        eager_module,
        compiled_bundle,
        eager_bundle,
        compare_sparse_state=not args.tbe,
        atol=0.0,
        rtol=0.0,
    )

    eager_history: list[float] = []
    recir_history: list[float] = []
    for step, host_batch in enumerate(batches):
        batch = host_batch.to(device)
        eager_bundle.dense_opt.zero_grad(set_to_none=True)
        eager_bundle.sparse_opt.zero_grad(set_to_none=True)
        expected_losses, expected_named = recir_train_step(
            eager_module,
            eager_bundle.dense_opt,
            eager_bundle.sparse_opt,
            *batch.flat(),
        )
        actual_losses, actual_named = runner(
            compiled_module,
            compiled_bundle.dense_opt,
            compiled_bundle.sparse_opt,
            *batch.flat(),
        )
        if not args.report_only:
            _assert_tensor_close(
                actual_losses,
                expected_losses,
                atol=args.loss_atol,
                rtol=args.loss_rtol,
                name=f"step {step} losses",
            )
            _assert_tree_close(
                actual_named,
                expected_named,
                atol=args.output_atol,
                rtol=args.output_rtol,
                name=f"step {step} output",
            )
            _assert_training_state_close(
                compiled_module,
                eager_module,
                compiled_bundle,
                eager_bundle,
                compare_sparse_state=not args.tbe,
                atol=args.state_atol,
                rtol=args.state_rtol,
            )
        eager_loss_values = expected_losses.tolist()
        recir_loss_values = actual_losses.tolist()
        eager_history.append(sum(eager_loss_values))
        recir_history.append(sum(recir_loss_values))
        logger.info(
            "step %d eager_losses=%s recir_losses=%s",
            step,
            eager_loss_values,
            recir_loss_values,
        )

    window = min(10, len(eager_history))
    gaps = [
        abs(eager_loss - recir_loss)
        for eager_loss, recir_loss in zip(eager_history, recir_history)
    ]
    logger.info(
        "summary steps=%d eager_first=%g eager_last=%g recir_first=%g "
        "recir_last=%g mean_abs_gap=%g max_abs_gap=%g",
        len(eager_history),
        sum(eager_history[:window]) / window,
        sum(eager_history[-window:]) / window,
        sum(recir_history[:window]) / window,
        sum(recir_history[-window:]) / window,
        sum(gaps) / len(gaps),
        max(gaps),
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
