# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""CPU smoke run: compile and step CoScaling through recir on JAX CPU.

Compiles the train step with rankfm's ``compile_gspmd_step`` (cpu embedding
backend, no padding), runs a few steps on one dumped batch, and checks that the
losses are finite, step 0 matches an eager forward with the encoder's static
bounds off, and the loss moves.

The batch comes from ``hammer/v3/models/udd_fm_v3/launch/coscaling/dump_batch.sh``
(devgpu); a RandomDataPipe batch breaks the MoT preprocessor's KJT merge.

    buck2 run @fbcode//mode/opt fbcode//recir/examples/coscaling_mvai/smoke:cpu_smoke

Knobs: COSCALING_BATCH_PATH (/tmp/coscaling_roo_batch.pt), COSCALING_STEPS (5),
COSCALING_MAX_ROWS (4096), COSCALING_MAX_SEQ (64; 0 disables truncation and
sizes the encoder's static bound for prod).
"""

import os

# Must precede the first jax import (pulled in by train_common).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# As in the rankfm prod launch config: int64 ids, and the warmup schedule's
# terminal stage compares the step counter against int64 max.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import logging
from typing import Any, cast, List, Optional

import torch
import torch.nn as nn

# Import for its side effects: fbgemm op registration, the recir fbgemm
# lowerings and the gr::batched_complete_cumsum decomposition.
from recir.examples.rankfm_mvai import train_common  # usort: skip

from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    LR_ITER_BUFFER,
    RecirCompileSpec,
    RecirMode,
    run_step_and_sync,
    setup_dense_optimizer_and_warmup,
)
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import (
    flatten_batch,
    make_preprocessor_input_from_batch,
)
from hammer.v3.models.udd_fm_v3.tests.parity.parity_common import _make_fixed_batch
from recir.examples.coscaling_mvai.backends import (
    build_cpu_backend,
    coscaling_sparse_arch,
)
from recir.examples.coscaling_mvai.model import build_model, CoScalingRecirModule
from recir.examples.coscaling_mvai.smoke.config import build_coscaling_config
from recir.region_transforms.dropout_prng import DropoutPRNGTransform
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_PATH: str = "/tmp/coscaling_roo_batch.pt"
# _make_fixed_batch always truncates rows; this bound makes that a no-op.
_NO_TRUNCATION: int = 1 << 31


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _build_batch(
    prod_config: Any, batch_path: str, max_rows: int, max_seq: Optional[int]
) -> Any:
    if not os.path.exists(batch_path):
        raise FileNotFoundError(
            f"no dumped batch at {batch_path}; run "
            "hammer/v3/models/udd_fm_v3/launch/coscaling/dump_batch.sh on a devgpu "
            "or point COSCALING_BATCH_PATH at one"
        )
    # _make_fixed_batch reads the batch path and row truncation from the
    # environment; it also dedups keys and clamps ids to the shrunk tables.
    os.environ["PARITY_BATCH_PATH"] = batch_path
    os.environ["PARITY_MAX_SEQ"] = str(_NO_TRUNCATION if max_seq is None else max_seq)
    batch: Any = _make_fixed_batch(prod_config, torch.device("cpu"), max_rows)
    if batch.unary_features is None:
        # flatten_batch always emits the unary pair; CoScaling ignores it.
        batch.unary_features = KeyedJaggedTensor(
            keys=[],
            values=torch.empty(0, dtype=torch.int64),
            lengths=torch.empty(0, dtype=torch.int32),
        )
    return batch


def _eager_reference(model: CoScalingRecirModule, batch: Any) -> torch.Tensor:
    """Per-task losses from an eager forward with the static bounds off.

    Also checks the real per-request maxima against the bounds, which the
    compiled step relies on without being able to check them.
    """
    block: Any = model.pipeline.compute_block_stack.coscaling_encoder
    bounds = (block._static_max_ro_len, block._static_max_num_targets)
    observed: dict[str, int] = {}

    def record(_mod: nn.Module, _args: Any, kwargs: dict[str, Any]) -> None:
        num_targets = kwargs["num_targets"]
        observed["max_ro_len"] = int(
            (kwargs["combined_past_lengths"] - num_targets).max()
        )
        observed["max_num_targets"] = int(num_targets.max())

    handle = block._impl.register_forward_pre_hook(record, with_kwargs=True)
    block._static_max_ro_len, block._static_max_num_targets = None, None
    try:
        with torch.no_grad():
            losses, _ = model(*flatten_batch(batch))
    finally:
        block._static_max_ro_len, block._static_max_num_targets = bounds
        handle.remove()

    if observed["max_ro_len"] > bounds[0] or observed["max_num_targets"] > bounds[1]:
        raise RuntimeError(
            f"static bounds {bounds} do not cover the batch: observed {observed}"
        )
    logger.info("[coscaling] eager reference: bounds=%s observed=%s", bounds, observed)
    return losses


def _disable_dropout(model: nn.Module) -> int:
    # The compiled step draws dropout masks from a counter-keyed PRNG, so any
    # live dropout would make the eager comparison meaningless.
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p > 0:
            module.p = 0.0
            n += 1
    return n


def run(
    batch_path: str, num_steps: int, max_rows: int, max_seq: Optional[int]
) -> List[torch.Tensor]:
    train_common.resolve_dist_env()
    train_common.ensure_gloo_process_group()
    device = torch.device("cpu")

    module_config, feature_config, prod = build_coscaling_config(
        max_rows, max_uih_len=max_seq
    )
    torch.manual_seed(0)
    model = build_model(module_config, feature_config)
    batch = _build_batch(prod, batch_path, max_rows, max_seq)
    batch_size = int(batch.num_candidates.numel())
    logger.info(
        "[coscaling] batch: B=%d B_NRO=%d dropout_disabled=%d",
        batch_size,
        int(batch.num_candidates.sum()),
        _disable_dropout(model),
    )

    backend = build_cpu_backend(module_config.embedding_tables)
    # Read before setup: DMP replaces the collection and its configs with it.
    sparse_arch: Any = coscaling_sparse_arch(model)
    table_configs = sparse_arch._embedding_collection.embedding_configs()
    dispatchers = backend.get_override(
        model.pipeline,
        make_preprocessor_input_from_batch(batch),
        device,
        batch_size,
        1,
        1,
    )
    setup = backend.setup(
        model, table_configs, prod.optimizer_config.sparse.default, device, 1
    )
    model = cast(Any, setup.model)
    model.set_recir_kjt_keys(
        batch.ro_id_list_features.keys(), batch.nro_id_list_features.keys()
    )

    # After DMP (which may re-initialize the tables), before any update.
    ref_losses = _eager_reference(model, batch)

    dense_opt, _warmup = setup_dense_optimizer_and_warmup(
        model,
        list(setup.dense_named_parameters.values()),
        prod.optimizer_config.dense.default,
        device,
    )
    spec = RecirCompileSpec(
        mode=RecirMode.GSPMD_TPU,
        device=device,
        num_ranks=1,
        num_processes=1,
        loss_reduction="sum",
        region_transforms=[DropoutPRNGTransform(counter_fqn=LR_ITER_BUFFER)],
    )
    runner = compile_gspmd_step(model, dense_opt, batch, dispatchers, spec)

    step_losses: List[torch.Tensor] = []
    for step in range(num_steps):
        losses, _named = run_step_and_sync(runner, model, dense_opt, batch)
        losses = losses.detach().clone()
        logger.info("[coscaling] step %d loss=%.6f", step, float(losses.sum()))
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError(f"step {step}: non-finite loss {losses}")
        step_losses.append(losses)

    diff = (step_losses[0] - ref_losses).abs()
    logger.info(
        "[coscaling] step 0 compiled vs eager: max|diff|=%g at task %d",
        float(diff.max()),
        int(diff.argmax()),
    )
    torch.testing.assert_close(
        step_losses[0],
        ref_losses,
        rtol=1e-4,
        atol=1e-5,
        msg=lambda m: f"compiled step 0 vs eager forward: {m}",
    )
    if num_steps > 1 and torch.equal(step_losses[0], step_losses[-1]):
        raise RuntimeError("loss did not change across steps; no update was applied")
    logger.info("[coscaling] CPU smoke passed (%d steps)", num_steps)
    return step_losses


def main() -> None:
    train_common.setup_logging()
    max_seq = _env_int("COSCALING_MAX_SEQ", 64)
    run(
        batch_path=os.environ.get("COSCALING_BATCH_PATH", DEFAULT_BATCH_PATH),
        num_steps=_env_int("COSCALING_STEPS", 5),
        max_rows=_env_int("COSCALING_MAX_ROWS", 4096),
        max_seq=max_seq if max_seq > 0 else None,
    )


if __name__ == "__main__":
    main()
