# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""The CoScaling recir smoke run, shared by the CPU and TPU entry points."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, cast, Dict, List, Tuple

import jax  # @manual
import torch
import torch.nn as nn

# Import for its side effects: fbgemm op registration, the recir fbgemm
# lowerings and the gr::batched_complete_cumsum decomposition.
from recir.examples.rankfm_mvai import train_common  # usort: skip

from hammer.v3.models.udd_fm_v3.coscaling.coscaling_config import CoScalingModuleConfig
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
from recir.examples.coscaling_mvai.backends import (
    build_cpu_backend,
    coscaling_sparse_arch,
)
from recir.examples.coscaling_mvai.model import build_model, CoScalingRecirModule
from recir.examples.rankfm_mvai.xsurface_helpers import (
    make_padding_region_transforms,
    prepare_override_build_sample,
)
from recir.region_transforms.dropout_prng import DropoutPRNGTransform
from recir.region_transforms.padding import compute_padded_shape
from recir.training.compile import (
    compile_gspmd_step,
    LR_ITER_BUFFER,
    recir_train_step,
    RecirCompileSpec,
    RecirMode,
    run_step_and_sync,
)
from recir.training.setup import setup_dense_optimizer_and_warmup

logger: logging.Logger = logging.getLogger(__name__)


def _coscaling_train_step(
    mod: torch.nn.Module,
    opt: torch.optim.Optimizer,
    ro_float_features: torch.Tensor,
    nro_float_features: torch.Tensor,
    ro_id_list_values: torch.Tensor,
    ro_id_list_lengths: torch.Tensor,
    nro_id_list_values: torch.Tensor,
    nro_id_list_lengths: torch.Tensor,
    unary_values: torch.Tensor,
    unary_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    supervision_label_t: torch.Tensor,
    supervision_weight_t: torch.Tensor,
    all_labels: Dict[str, torch.Tensor],
    loss_denom_source: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # PaddingRegionTransform.bind_signature finds its inputs by parameter NAME,
    # and recir_train_step is variadic. Same wrapper as rankfm's
    # model_family._udd_fm_v3_train_step.
    return recir_train_step(
        mod,
        opt,
        ro_float_features,
        nro_float_features,
        ro_id_list_values,
        ro_id_list_lengths,
        nro_id_list_values,
        nro_id_list_lengths,
        unary_values,
        unary_lengths,
        num_candidates,
        supervision_label_t,
        supervision_weight_t,
        all_labels,
        loss_denom_source,
    )


def _eager_reference(model: CoScalingRecirModule, batch: Any) -> torch.Tensor:
    """Per-task losses from an eager forward on the real batch, static shapes off."""
    pipeline: Any = model.pipeline
    block: Any = pipeline.compute_block_stack.coscaling_encoder
    bounds = (block._static_max_ro_len, block._static_max_num_targets)
    preprocessors = [m for m in model.modules() if isinstance(m, MoTPreprocessor)]
    observed: Dict[str, int] = {}

    def record(_mod: nn.Module, _args: Any, kwargs: Dict[str, Any]) -> None:
        num_targets = kwargs["num_targets"]
        observed["max_ro_len"] = int(
            (kwargs["combined_past_lengths"] - num_targets).max()
        )
        observed["max_num_targets"] = int(num_targets.max())

    handle = block._impl.register_forward_pre_hook(record, with_kwargs=True)
    block._static_max_ro_len, block._static_max_num_targets = None, None
    static_modes = [p._static_shape_mode for p in preprocessors]
    for p in preprocessors:
        p._static_shape_mode = False
    try:
        with torch.no_grad():
            losses, _ = model(*flatten_batch(batch))
    finally:
        block._static_max_ro_len, block._static_max_num_targets = bounds
        for p, mode in zip(preprocessors, static_modes):
            p._static_shape_mode = mode
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


def run_smoke(
    module_config: CoScalingModuleConfig,
    feature_config: Any,
    optimizer_config: Any,
    batch: Any,
    num_steps: int,
) -> List[torch.Tensor]:
    """Compile and step the model on ``batch``; raise if a check fails."""
    if not module_config.static_shape_mode:
        raise ValueError(
            "the smoke compiles with padding, which needs static_shape_mode; "
            "regenerate the artifact"
        )
    # Exact fp32 matmuls, as rankfm's parity harness uses: TPU's default runs
    # fp32 matmuls in bf16 passes (~1e-3 relative), which would swamp the eager
    # comparison below.
    precision = os.environ.get("COSCALING_MATMUL_PRECISION", "highest")
    jax.config.update("jax_default_matmul_precision", precision)
    train_common.resolve_dist_env()
    train_common.ensure_gloo_process_group()
    device = torch.device("cpu")
    # One process; the compiled step spans every JAX device (recir builds its
    # mesh from jax.devices()), e.g. 8 on a 2x2x1 tpu7x host.
    num_ranks = jax.device_count()
    num_processes = 1

    torch.manual_seed(0)
    model = build_model(module_config, feature_config)
    batch_size = int(batch.num_candidates.numel())
    backend = build_cpu_backend(module_config.embedding_tables)

    # Padding sizes, exactly as rankfm's model_family computes them.
    nro_budget = module_config.static_nro_per_sample_budget
    if nro_budget is None:
        raise ValueError("static_shape_mode needs static_nro_per_sample_budget")
    nro_per_sample = int(nro_budget)
    padded_num_devices = math.lcm(num_ranks, backend.sc_modulus() * num_processes)
    global_bs = batch_size * num_processes
    padded_bs = compute_padded_shape(global_bs, padded_num_devices)
    logger.info(
        "[coscaling] devices=%d (%s, matmul_precision=%s) batch: B=%d B_NRO=%d -> "
        "padded B=%d B_NRO=%d dropout_disabled=%d",
        num_ranks,
        jax.devices()[0].platform,
        precision,
        batch_size,
        int(batch.num_candidates.sum()),
        padded_bs,
        padded_bs * nro_per_sample,
        _disable_dropout(model),
    )

    # Read before setup: DMP replaces the collection and its configs with it.
    sparse_arch: Any = coscaling_sparse_arch(model)
    table_configs = sparse_arch._embedding_collection.embedding_configs()
    # The override is built on a sample expanded to the PADDED global sizes, so
    # its cached shapes match what PaddingRegionTransform produces.
    batch_sample, override_bs = prepare_override_build_sample(
        batch,
        module_config,
        global_bs,
        nro_per_sample,
        padded_num_devices,
        enable_padding=True,
    )
    dispatchers = backend.get_override(
        model.pipeline, batch_sample, device, override_bs, num_ranks, num_processes
    )
    setup = backend.setup(
        model, table_configs, optimizer_config.sparse.default, device, num_ranks
    )
    model = cast(CoScalingRecirModule, setup.model)
    model.set_recir_kjt_keys(
        batch.ro_id_list_features.keys(),
        batch.nro_id_list_features.keys(),
        batch.unary_features.keys(),
    )

    # After DMP (which may re-initialize the tables), before any update.
    ref_losses = _eager_reference(model, batch)

    dense_opt, _warmup = setup_dense_optimizer_and_warmup(
        model,
        list(setup.dense_named_parameters.values()),
        optimizer_config.dense.default,
        device,
    )
    region_transforms: List[Any] = make_padding_region_transforms(
        nro_per_sample, padded_num_devices, num_processes
    )
    region_transforms.append(DropoutPRNGTransform(counter_fqn=LR_ITER_BUFFER))
    spec = RecirCompileSpec(
        mode=RecirMode.GSPMD_TPU,
        device=device,
        num_ranks=num_ranks,
        num_processes=num_processes,
        # As rankfm: the loss is normalized by the global real candidate count
        # in-graph, so GSPMD must not divide by the world size again.
        loss_reduction="mean",
        region_transforms=region_transforms,
    )
    runner = compile_gspmd_step(
        model,
        dense_opt,
        batch,
        dispatchers,
        spec,
        train_step_fn=_coscaling_train_step,
        flatten_batch=flatten_batch,
    )

    step_losses: List[torch.Tensor] = []
    for step in range(num_steps):
        losses, _named = run_step_and_sync(
            runner, model, dense_opt, batch, flatten_batch=flatten_batch
        )
        losses = losses.detach().clone()
        logger.info("[coscaling] step %d loss=%.6f", step, float(losses.sum()))
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError(f"step {step}: non-finite loss {losses}")
        step_losses.append(losses)

    diff = (step_losses[0] - ref_losses).abs()
    logger.info(
        "[coscaling] step 0 compiled (padded) vs eager: max|diff|=%g at task %d",
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
    logger.info("[coscaling] smoke passed (%d steps)", num_steps)
    return step_losses
