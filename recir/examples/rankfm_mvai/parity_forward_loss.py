#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Compare consecutive CPU-eager and TPU-compiled RankFM optimizer steps.

Both cells start from the same seeded model and consume the same replay batches.
The model shape, precision, embedding backend, optimizer configuration, and static
shape settings match the ``rankfm-prod`` launch config; only execution differs.

Required environment:
    PARITY_DEVICE=tpu
    PARITY_GCS_DATA_PATH=gs://<bucket>/rankfm_1shard_data

Optional overrides include ``PARITY_BF16``, ``PARITY_HASH_SIZE_CAP``, ``RANKFM_S``,
``RANKFM_MVAI_ATTENTION_BACKEND`` (``baseline`` or ``helion``), replay
file/index/batch/step settings, matmul precision, and comparison tolerances.
"""

from __future__ import annotations

import os

# Device selection must happen before JAX is imported.
_PARITY_DEVICE: str = os.environ.get("PARITY_DEVICE", "tpu").lower()

import contextlib
import importlib
import logging
import math
from typing import Any, cast, Dict, List, Optional, Tuple

importlib.import_module("fbgemm_gpu")  # force C++ op registration before fallback
import jax  # @manual
import torch
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.parity.step_digest import arm
from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    RecirCompileSpec,
    RecirMode,
    run_step_and_sync,
    setup_dense_optimizer_and_warmup,
)
from hammer.v3.models.udd_fm_v3.recir_core.compile import compile_step
from hammer.v3.models.udd_fm_v3.recir_core.optim import prime_optimizer_state
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.embedding.in_graph import SHARDED_TABLE_PREFIX
from recir.examples.rankfm_mvai.config import (
    build_module_config,
    build_optimizer_config,
    RankFMModelArgs,
)
from recir.examples.rankfm_mvai.model_family import (
    _udd_fm_v3_train_step,
    _udd_fm_v3_train_step_two_opt,
)
from recir.examples.rankfm_mvai.rankfm_batch_utils import resize_kjt_batch, tile_dim0
from recir.examples.rankfm_mvai.rankfm_embedding_backends import (
    build_embedding_backend,
    RANKFM_EMBEDDING_PARAM_PREFIX,
)
from recir.examples.rankfm_mvai.rankfm_model_utils import (
    AttentionBackend,
    build_model_from_config,
    resolve_attention_backend,
)
from recir.examples.rankfm_mvai.train_common import (
    ensure_gloo_process_group,
    force_cpu_barrier,
    resolve_dist_env,
)
from recir.examples.rankfm_mvai.xsurface_helpers import (
    _resolve_nro_per_sample,
    make_padding_region_transforms,
    prepare_override_build_sample,
)
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl

logger: logging.Logger = logging.getLogger(__name__)

# Bootstrap gr::batched_complete_cumsum if the C++ library didn't load and
# register its XLA-lowering decomposition (mirrors train.py).
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


# Embedding-lookup backend under test. "cpu" (torchrec DMP) and "tpu" (JAX
# SparseCore) keep the embedding update OFF the compiled graph -> single dense
# optimizer. "in_graph" lowers the ring-shift / FSDP tables INTO the GSPMD
# program as plain params, so it adds a dedicated embedding optimizer and uses
# the two-optimizer compile path (mirrors model_family._build_compiled_step).
_PARITY_EMB_BACKEND: str = os.environ.get("PARITY_EMB_BACKEND", "in_graph").lower()
_PARITY_ON_DEVICE_INIT: bool = os.environ.get(
    "PARITY_ON_DEVICE_INIT", "0"
).lower() not in ("0", "false", "no", "")
# Precision regime for BOTH eager and compiled cells. bf16 (default) mirrors the
# bench's --bf16_training. Set PARITY_BF16=0 for fp32 EVERYWHERE to disambiguate
# whether an eager-vs-compiled loss gap is real or just bf16 reduction-order
# noise (fp32 should match near-exactly). fp32 also forces jax matmul
# precision=highest (see main) unless PARITY_PRECISION is set explicitly.
_PARITY_BF16: bool = os.environ.get("PARITY_BF16", "1").lower() not in (
    "0",
    "false",
    "no",
    "",
)
# Parity currently supports Helion because its adapter's eager fallback remains
# the Aten reference. The Pallas adapters need separate compiled-only installation
# before they can preserve that comparison contract.
_PARITY_ATTENTION_BACKEND: AttentionBackend = resolve_attention_backend()
if _PARITY_ATTENTION_BACKEND not in ("baseline", "helion"):
    raise ValueError(
        "RankFM parity supports only baseline or helion attention; "
        "the CPU-eager cell must remain the Aten reference"
    )


# ---------------------------------------------------------------------------
# Static-shape padding (so runtime shapes match the trace-time fake batch and
# downstream XLA submodules don't see data-dependent dims). This is the
# eager equivalent of PaddingRegionTransform, applied per-rank before yield.
# ---------------------------------------------------------------------------


def _load_e2e_batches(
    model: BlueReelsVDDTrainV3,
    module_config: Any,
    gcs_data_path: str,
) -> List[Any]:
    """Load consecutive real batches from the e2e dump, as the trainer does.

    Mirrors model_family.create_train_module_and_optimizer's trace-batch pull:
    compute_kjt_key_to_size (pre-DMP) -> BatchReplayPipe (single file, replicate)
    -> replay_batch_iter(max_steps=PARITY_STEPS). Both parity cells consume the
    same real jagged batches in order.
    Knobs: PARITY_DATA_FILE_TEMPLATE (default
    rank_0_batch_{i}.pt.zst), PARITY_DATA_FILE_INDEX (default 1),
    PARITY_LOCAL_BATCH_SIZE (default 32, matching the 1shard yaml).
    """
    from .data_pipe import BatchReplayPipe, compute_kjt_key_to_size, replay_batch_iter

    file_tmpl = os.environ.get("PARITY_DATA_FILE_TEMPLATE", "rank_0_batch_{i}.pt.zst")
    file_idx = int(os.environ.get("PARITY_DATA_FILE_INDEX", "1"))
    local_bs = int(os.environ.get("PARITY_LOCAL_BATCH_SIZE", "32"))
    steps = int(os.environ.get("PARITY_STEPS", "3"))
    if steps <= 0:
        raise ValueError(f"PARITY_STEPS must be positive, got {steps}")
    num_candidates_per_sample = _resolve_nro_per_sample(module_config, True)
    # MUST run before any backend.setup / DMP wrap: post-DMP the EC is wrapped and
    # embedding_configs() disappears. _compiled_step / _eager_step call
    # backend.setup later, so here (right after the raw build) is the right point.
    key_to_size = compute_kjt_key_to_size(model)
    data_pipe = BatchReplayPipe(
        path=gcs_data_path,
        rank=0,
        world_size=1,
        replicate_across_ranks=True,
        file_list=[file_tmpl.format(i=file_idx)],
    )
    # Always pull the RAW jagged batch, mirroring the e2e trace-batch pull;
    # slice+remap only.
    batches = list(
        replay_batch_iter(
            data_pipe,
            key_to_size,
            max_steps=steps,
            local_batch_size=local_bs,
            num_candidates=num_candidates_per_sample,
        )
    )
    if len(batches) != steps:
        raise RuntimeError(f"Requested {steps} parity batches, loaded {len(batches)}")
    return batches


def _dense_params(backend: Any, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot the dense (over-arch) params the dense optimizer actually steps.

    ``backend.dense_named_parameters`` is the canonical dense set (embeddings
    excluded). Both cells go through the same backend, so the two snapshots share
    names and intersect cleanly.
    """
    return {
        n: p.detach().clone() for n, p in backend.dense_named_parameters(model).items()
    }


_WRAPPER_PREFIXES: Tuple[str, ...] = (
    "module.",
    "_orig_mod.",
    "_fsdp_wrapped_module.",
    "_checkpoint_wrapped_module.",
    "_dmp_wrapped_module.",
)


def _norm_param_name(name: str) -> str:
    """Strip DMP/wrapper prefixes so eager (DMP-wrapped) and compiled (unwrapped in
    compile_step) param names align for the gradient comparison. Strips any
    leading wrapper segment repeatedly (they can nest)."""
    changed = True
    while changed:
        changed = False
        for pfx in _WRAPPER_PREFIXES:
            if name.startswith(pfx):
                name = name[len(pfx) :]
                changed = True
    return name


def _helion_pre_compile() -> None:
    """Clear stale Helion attention shape sigs before the trace (no-op unless gated).

    The compile runs recir's eager make_fx trace synchronously, during which the
    HSTU fwd/bwd register_fake kernels record their attention shapes. Clear stale
    sigs BEFORE the build; warm AFTER it, before the first (lazy) lower.
    """
    if _PARITY_ATTENTION_BACKEND != "helion":
        return
    from recir.ops.helion.jagged_hstu_attention import clear_traced_sigs

    clear_traced_sigs()


def _helion_post_compile(real_bs: int, padded_bs: int) -> None:
    """Latch the PRT pad-row count and warm the Helion kernels (no-op unless gated).

    ``padded_bs`` is the override-build size (compute_padded_shape of the global RO
    batch), the same quantity model_family passes as padded_ro_bs, so the derived
    pad-row count matches what _drop_pad_rows slices off. len(jax.devices()) matches
    the lowering's world_size (recir/torchax/ops/jhstu_attention.py).
    """
    if _PARITY_ATTENTION_BACKEND != "helion":
        return
    from recir.ops.helion.jagged_hstu_attention import (
        set_pad_seqs_per_device,
        warm_traced_hstu_kernels,
    )

    world_size = len(jax.devices())
    set_pad_seqs_per_device(real_bs=real_bs, padded_bs=padded_bs, world_size=world_size)
    warm_traced_hstu_kernels(world_size=world_size)


def _compiled_step(
    model: BlueReelsVDDTrainV3,
    batches: List[Any],
    num_ranks: int,
    num_processes: int,
    global_bs: int,
    padding: bool,
    emb_backend: str = _PARITY_EMB_BACKEND,
) -> Tuple[
    Dict[str, torch.Tensor],
    List[float],
    Dict[str, Any],
    Dict[str, torch.Tensor],
]:
    """Run consecutive compiled training steps and return params, loss, exports, and grads.

    Mirrors RecirXLACompiledModule (rankfm_mvai/model_family.py): build the recir
    embedding backend, expand the override sample, build dispatchers, set up
    embedding storage, build the unified TensorLRAdamW dense optimizer, wire the
    PaddingRegionTransform, then compile once and run consecutive steps with
    carried state. DCN is included (the
    model was built with use_inter_arch=True).
    """
    device = torch.device("cpu")
    model.train(True)
    module_config = model.module_config
    trace_batch = batches[0]
    # global_bs is passed in (the GLOBAL batch); `batch` here is PER-HOST
    # (num_local identical device blocks), so it can no longer be derived from
    # batch.ro_float_features.shape[0].

    backend = build_embedding_backend(
        emb_backend,
        on_device_init=_PARITY_ON_DEVICE_INIT,
        embedding_tables=module_config.embedding_tables,
    )
    # cpu/torchrec DMP needs a gloo PG; mirror the family (reuse if already init).
    if not (dist.is_available() and dist.is_initialized()):
        backend.init_pg()
    num_candidates_per_sample = _resolve_nro_per_sample(module_config, True)
    padded_num_devices = math.lcm(num_ranks, backend.sc_modulus() * num_processes)

    # Override-build sample. PADDED: resize to the global padded shape via the SAME
    # helper prod/train.py uses. JAGGED: build from the RAW batch so the override's
    # NRO matches the actual jagged candidate count (prepare_override_build_sample's
    # non-padding branch rectangular-resizes NRO to global_bs * num_candidates,
    # which would mismatch the jagged dense path -- the 5120-vs-2808 cat error).
    if padding:
        batch_sample, override_bs = prepare_override_build_sample(
            trace_batch,
            module_config,
            global_bs,
            num_candidates_per_sample,
            padded_num_devices,
            enable_padding=True,
        )
    else:
        all_labels = trace_batch.all_labels
        assert all_labels is not None
        # get_override derives each feature's per-chip token total as
        # (GLOBAL token total) / jax.device_count(), so the override-build sample
        # must be GLOBAL. The train batch is PER-HOST (num_local identical
        # blocks), so tile its jagged id-lists up by num_processes to reconstruct
        # the global sample -- preserving the ACTUAL jagged candidate counts
        # (unlike prepare_override_build_sample's rectangular NRO resize, which
        # mismatches the jagged dense path). num_processes == 1 -> no tiling.
        g_ro = trace_batch.ro_float_features.shape[0] * num_processes
        g_nro = trace_batch.nro_float_features.shape[0] * num_processes
        batch_sample = _inputs_to_feature_bundle(
            resize_kjt_batch(trace_batch.ro_id_list_features, g_ro),
            None,
            resize_kjt_batch(trace_batch.nro_id_list_features, g_nro),
            tile_dim0(trace_batch.num_candidates, g_ro),
            {k: tile_dim0(v, g_nro) for k, v in all_labels.items()},
        )
        override_bs = global_bs

    # pyre-ignore[16]: dynamic submodule traversal through the hammer v3 union.
    ec = model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection
    table_configs = ec.embedding_configs()
    dispatchers = backend.get_override(  # pyre-ignore[16]
        model.recblock,
        batch_sample,
        device,
        override_bs,
        num_ranks,
        num_processes,
    )
    embedding_setup = backend.setup(
        model,
        table_configs,
        build_optimizer_config().sparse.default,
        device,
        num_ranks,
    )
    model = cast(BlueReelsVDDTrainV3, embedding_setup.model)
    unary_kjt = trace_batch.unary_features
    assert unary_kjt is not None
    model.set_recir_kjt_keys(
        trace_batch.ro_id_list_features.keys(),
        trace_batch.nro_id_list_features.keys(),
        unary_kjt.keys(),
    )

    dense_named = embedding_setup.dense_named_parameters
    dense_params = list(dense_named.values())
    logger.info(
        f"[PARITY][DENSE-SET] COMPILED-{emb_backend} dist_init={dist.is_initialized()} "
        f"dense_named={len(dense_named)} "
        f"has_EC={any('_embedding_collection' in n for n in dense_named)} "
        f"has_hstu_posw={any('_pos_weights' in n for n in dense_named)}"
    )
    sparse_runtime = embedding_setup.sparse_runtime
    dense_opt, _ = setup_dense_optimizer_and_warmup(
        model,
        dense_params,
        build_optimizer_config().dense.default,
        device,
        sparse_cfg=(
            sparse_runtime.config if sparse_runtime.lr_buffer is not None else None
        ),
        sparse_lr_buffer=sparse_runtime.lr_buffer,
    )
    emb_opt = sparse_runtime.optimizer

    region_transforms = (
        make_padding_region_transforms(
            num_candidates_per_sample, padded_num_devices, num_processes
        )
        if padding
        else []
    )
    spec = RecirCompileSpec(
        mode=RecirMode.GSPMD_TPU,
        device=device,
        num_ranks=num_ranks,
        num_processes=num_processes,
        loss_reduction="mean",
        region_transforms=region_transforms,
    )
    _helion_pre_compile()
    if emb_opt is None:
        runner = compile_gspmd_step(
            model,
            dense_opt,
            trace_batch,
            dispatchers,
            spec,
            # Armed, so the step exports ``grad/<param>`` for the diff
            # below. Inert unless PARITY_DIGEST_KINDS includes "grad".
            train_step_fn=arm(_udd_fm_v3_train_step),
        )

        def _run_one(run_batch: Any) -> Tuple[torch.Tensor, Any]:
            return run_step_and_sync(runner, model, dense_opt, run_batch)

    else:
        parallel_mode = GSPMDParallel(
            "replicate",
            # The global batch is sharded across the GSPMD mesh. The loss is
            # already normalized by the global denominator, so replicated
            # parameter gradients must be summed across ranks.
            loss_reduction="mean",
            num_ranks=num_ranks,
            num_processes=num_processes,
            fsdp_table_substr=SHARDED_TABLE_PREFIX,
        )
        runner = compile_step(
            arm(_udd_fm_v3_train_step_two_opt),
            model,
            (dense_opt, emb_opt),
            trace_batch,
            parallel_mode=parallel_mode,
            backend=XLAGSPMDBackend(),
            dispatchers=dispatchers,
            region_transforms=region_transforms,
        )

        # pyre-ignore
        def _run_one(run_batch: Any) -> Tuple[torch.Tensor, Any]:
            result = runner(model, dense_opt, emb_opt, *flatten_batch(run_batch))
            sync_device_state_to_torch(runner)
            return result

    _helion_post_compile(real_bs=global_bs, padded_bs=override_bs)

    # Reuse the compiled runner so model and optimizer state carry across steps.
    loss_values: List[float] = []
    exports: Dict[str, Any] = {}
    compiled_grads: Dict[str, torch.Tensor] = {}
    # Embeddings are trained by the in-backward DMP-FUSED RowWiseAdagrad (dist is
    # always initialized in main()), so there is nothing extra to step here.
    for step, run_batch in enumerate(batches):
        losses, named = _run_one(run_batch)
        loss_value = float(losses.detach().to(torch.float64).sum())
        loss_values.append(loss_value)
        logger.info("[PARITY][COMPILED][step=%d] loss=%.6f", step + 1, loss_value)
        if isinstance(named, dict):
            for k in (
                "loss",
                "loss_scale",
                "llw_weights",
                "num_candidates",
                "pre_llw_losses",
                "tb:user_export/user_embedding",
                "tb:user_export/item_embedding",
            ):
                t = named.get(k)
                if isinstance(t, torch.Tensor):
                    exports[k] = t.detach().to(torch.float64).cpu()
            # Per-task reductions of the big per-sample tensors (weight_sum is
            # the key masking check: weight-0 dummies match jagged).
            w = named.get("weight")
            lab = named.get("label")
            pred = named.get("prediction")
            if (
                isinstance(w, torch.Tensor)
                and isinstance(lab, torch.Tensor)
                and isinstance(pred, torch.Tensor)
            ):
                wf = w.detach().to(torch.float64)
                labf = lab.detach().to(torch.float64)
                predf = pred.detach().to(torch.float64)
                # The in-graph weight/label/pred tensors are REPLICATED across
                # the num_processes host mesh axis; materializing to cpu sums
                # those replicas, so a raw sum(dim=-1) is num_processes x the
                # true global. Divide it out: the loss (numerator / global
                # world_candidate_sum) cancels this internally, but the
                # un-normalized raw exports do not -- leaving raw_ratio(tpu/cpu)
                # at num_processes^2 instead of the honest num_processes (eager
                # exports a single process's shard). Scale-free gates
                # (label_wmean/pred_wmean) are unaffected (both terms carry the
                # same factor).
                _p = float(num_processes)
                exports["weight_sum"] = wf.sum(dim=-1).cpu() / _p
                exports["label_wsum"] = (wf * labf).sum(dim=-1).cpu() / _p
                exports["pred_wsum"] = (wf * predf).sum(dim=-1).cpu() / _p
                exports["n_samples"] = torch.tensor([float(wf.shape[-1]) / _p])
            compiled_grads = {
                _norm_param_name(k[len("grad/") :]): v.detach().to(torch.float64).cpu()
                for k, v in named.items()
                if isinstance(k, str)
                and k.startswith("grad/")
                and isinstance(v, torch.Tensor)
            }
    return _dense_params(backend, model), loss_values, exports, compiled_grads


def _eager_step(
    model: BlueReelsVDDTrainV3,
    batches: List[Any],
    num_processes: int,
    num_ranks: int = 1,
    exports_out: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    List[float],
    Dict[str, torch.Tensor],
]:
    """Eager ground-truth: consecutive fwd+bwd+opt steps in plain PyTorch, NO recir.

    Runs the SAME train-step logic as the compiled path (``recir_train_step``,
    called DIRECTLY instead of traced/compiled) with NO override/dispatchers, NO
    PaddingRegionTransform, NO GSPMD -- the real preprocessor / sparse arch / DCN
    all run eagerly. Embeddings use the eager torchrec DMP ("cpu") backend
    (in_graph table lowering only exists inside the compiled graph). This is the
    reference the compiled JAGGED run should reproduce; the clean signals are the
    dense params (embeddings are excluded from _dense_params, and their update
    mechanism differs by design) and the scalar loss.

    NUMERIC-ALIGNMENT TODOs before this can gate: the compiled path rebinds
    group['lr'] to the model's LR buffer (warmup) and bf16 reduction order
    differs eager-vs-XLA, so the param-delta compare is only approximate for now.
    The scalar loss (pre-step, forward-only) is the tightest first signal.
    """
    device = torch.device("cpu")
    model.train(True)
    trace_batch = batches[0]
    # Eager embedding path = torchrec DMP; needs a gloo PG (reuse if already up).
    backend = build_embedding_backend("cpu")
    if not (dist.is_available() and dist.is_initialized()):
        backend.init_pg()
    # pyre-ignore[16]: dynamic submodule traversal through the hammer v3 union.
    ec = model.recblock.tokenizer_stack.mot_sparse_arch._sparse_arch._embedding_collection
    table_configs = ec.embedding_configs()
    embedding_setup = backend.setup(
        model,
        table_configs,
        build_optimizer_config().sparse.default,
        device,
        num_ranks,
    )
    model = cast(BlueReelsVDDTrainV3, embedding_setup.model)
    unary_kjt = trace_batch.unary_features
    assert unary_kjt is not None
    model.set_recir_kjt_keys(
        trace_batch.ro_id_list_features.keys(),
        trace_batch.nro_id_list_features.keys(),
        unary_kjt.keys(),
    )
    # Eager ALWAYS consumes the batch as ONE un-sharded global batch, so its loss
    # must run at num_local=1: a plain global seq_offsets + a single global NRO
    # mask. The lazy default is jax.local_device_count() (=8 on the TPU pod), which
    # would make the loss module read the un-sharded batch as 8 chip-major blocks
    # and mis-compute the per-chip NRO padding mask -- dropping real candidates
    # (eager weight_sum < compiled) on the candidate-level tasks. Pin it always,
    # the compiled side is genuinely
    # sharded and masks per-chip correctly.
    # pyre-ignore[16]: dynamic submodule traversal through the hammer v3 union.
    model.recblock.loss_module._num_local = 1
    dense_named = embedding_setup.dense_named_parameters
    dense_params = list(dense_named.values())
    logger.info(
        f"[PARITY][DENSE-SET] EAGER dist_init={dist.is_initialized()} "
        f"dense_named={len(dense_named)} "
        f"has_EC={any('_embedding_collection' in n for n in dense_named)} "
        f"has_hstu_posw={any('_pos_weights' in n for n in dense_named)}"
    )
    # Build the prod optimizer only to LIFT its exact hyperparameters (incl. the
    # warmup-initialized LR), then rebuild as a PLAIN eager AdamW. The prod
    # TensorLRAdamW hardcodes capturable=True + a 0-d tensor LR for XLA graph
    # capture, which asserts on CPU ("capturable=True requires cuda/xla/..."); the
    # eager-equivalent is capturable=False + scalar LR. Each step copies the
    # current tensor LR into eager AdamW and advances the same warmup controller
    # as compiled.
    xla_opt, warmup = setup_dense_optimizer_and_warmup(
        model, dense_params, build_optimizer_config().dense.default, device
    )
    g0 = xla_opt.param_groups[0]
    # Split the optimizer to MATCH the compiled path: dense -> AdamW (in-graph
    # TensorLRAdamW on the compiled side), embeddings -> DMP-FUSED RowWiseAdagrad
    # at the SPARSE optimizer's LR (in-backward on both sides). The old single
    # AdamW over ALL of dense_named stepped embeddings with AdamW, which never
    # matched the compiled/prod RowWiseAdagrad embedding update.
    dense_only = [
        p
        for n, p in dense_named.items()
        if not n.startswith(RANKFM_EMBEDDING_PARAM_PREFIX)
    ]
    dense_opt = torch.optim.AdamW(
        dense_only,
        lr=float(g0["lr"]),
        betas=g0["betas"],
        eps=g0["eps"],
        weight_decay=g0["weight_decay"],
        foreach=True,
    )
    # Embeddings are trained by the DMP-FUSED optimizer in loss.backward() (dist is
    # always initialized in main()), so there is no separate eager Adagrad.
    # Prime with the SAME helper the compiled path uses, so both sides materialize
    # state for the identical requires_grad=True set (no-op on numerics; a grad-less
    # but requires_grad param like _negative_action_embeddings gets zero state that
    # is never stepped, matching compiled).
    prime_optimizer_state(dense_opt)
    max_grad = getattr(xla_opt, "_max_grad", None)
    # Reuse the eager model and optimizer so state carries across batches.
    loss_values: List[float] = []
    eager_grads: Dict[str, torch.Tensor] = {}
    for step, batch in enumerate(batches):
        dense_opt.zero_grad(set_to_none=True)
        # bf16: a bf16 model casts activations to bf16 but keeps fp32 master
        # weights, relying on autocast to cast Linear weights at matmul time.
        # Compiled uses torchax AMP inside the trace; in pure eager that is inert,
        # so wrap forward in NATIVE torch.autocast (else "mat1 BFloat16 vs mat2
        # Float"). fp32 needs none. backward stays OUTSIDE autocast.
        autocast_ctx: Any = (
            torch.autocast(device_type="cpu", dtype=torch.bfloat16)
            if _PARITY_BF16
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            # flatten_batch's trailing slot is now the PRE-PAD per-sample
            # candidate counts, reduced by .sum() inside the model
            # (trace_sample.py). Eager has no device mesh, so that sum is
            # already the LOCAL per-host count -- which is what this leg needs,
            # since it consumes ONLY the per-host batch (loss_module._num_local
            # =1) and its numerator is per-host too. The explicit override below
            # is kept as a guard: pairing a per-host numerator with a global
            # (num_processes x) denom leaves the leaf params that feed ONLY the
            # world-normalized main heads (mlp_task_arch._bias, debias_arch
            # final layer) exactly num_processes x off in backward -- invisible
            # single-host, a hard fail on 32-host.
            _eager_args = list(flatten_batch(batch))
            _eager_args[-1] = batch.num_candidates.sum()
            all_losses, _named = model(*_eager_args)
            loss = all_losses.sum()
        loss.backward()
        loss_value = float(all_losses.detach().to(torch.float64).sum())
        loss_values.append(loss_value)
        logger.info("[PARITY][EAGER][step=%d] loss=%.6f", step + 1, loss_value)
        if step == len(batches) - 1:
            # Capture raw grads before value clipping. With mean loss reduction,
            # the compiled all-reduce SUM already produces this global-mean scale.
            # Names are normalized to align with the compiled dict.
            eager_grads = {
                _norm_param_name(n): p.grad.detach().to(torch.float64).cpu()
                for n, p in dense_named.items()
                if p.grad is not None
            }
            grad_sq = sum(float(g.pow(2).sum()) for g in eager_grads.values())
            logger.info(f"[PARITY][EAGER] dense grad global norm={grad_sq**0.5:.6f}")
            # Optional forward-export capture (mirrors _compiled_step's exports) so
            # callers can diff per-task weight_sum / label_wsum / pred_wsum against
            # the compiled cell. Non-capturing by default (exports_out=None).
            if exports_out is not None and isinstance(_named, dict):
                for _k in (
                    "loss",
                    "loss_scale",
                    "llw_weights",
                    "num_candidates",
                    "pre_llw_losses",
                    "tb:user_export/user_embedding",
                    "tb:user_export/item_embedding",
                ):
                    _t = _named.get(_k)
                    if isinstance(_t, torch.Tensor):
                        exports_out[_k] = _t.detach().to(torch.float64).cpu()
                _w = _named.get("weight")
                _lab = _named.get("label")
                _pred = _named.get("prediction")
                if (
                    isinstance(_w, torch.Tensor)
                    and isinstance(_lab, torch.Tensor)
                    and isinstance(_pred, torch.Tensor)
                ):
                    _wf = _w.detach().to(torch.float64)
                    _labf = _lab.detach().to(torch.float64)
                    _predf = _pred.detach().to(torch.float64)
                    exports_out["weight_sum"] = _wf.sum(dim=-1).cpu()
                    exports_out["label_wsum"] = (_wf * _labf).sum(dim=-1).cpu()
                    exports_out["pred_wsum"] = (_wf * _predf).sum(dim=-1).cpu()
                    exports_out["n_samples"] = torch.tensor([float(_wf.shape[-1])])
        # Mean loss reduction selects SUM gradient reduction in the compiled path:
        # backward has already normalized by global numel, so the all-reduce must
        # not divide again. Apply only the same dense value clipping before AdamW.
        if max_grad is not None:
            torch.nn.utils.clip_grad_value_(dense_only, max_grad)
        current_lr = float(xla_opt.param_groups[0]["lr"])
        for group in dense_opt.param_groups:
            group["lr"] = current_lr
        dense_opt.step()
        warmup.step()
    return _dense_params(backend, model), loss_values, eager_grads


def _unpad_candidate_embeddings(
    value: torch.Tensor, eager_num_candidates: torch.Tensor
) -> Optional[torch.Tensor]:
    devices = jax.devices()
    process_index = jax.process_index()
    local_chip_indices = [
        index
        for index, device in enumerate(devices)
        if device.process_index == process_index
    ]
    if (
        value.ndim != 2
        or not devices
        or value.shape[0] % len(devices) != 0
        or not local_chip_indices
        or eager_num_candidates.numel() % len(local_chip_indices) != 0
    ):
        return None
    candidates_per_chip = value.shape[0] // len(devices)
    chip_grid = value.reshape(len(devices), candidates_per_chip, value.shape[1])
    eager_counts_by_chip = eager_num_candidates.reshape(len(local_chip_indices), -1)
    return torch.cat(
        [
            chip_grid[global_chip, : int(eager_counts_by_chip[local_chip].sum())]
            for local_chip, global_chip in enumerate(local_chip_indices)
        ]
    )


def _log_embedding_parity(
    eager: Dict[str, torch.Tensor], compiled: Dict[str, Any]
) -> None:
    eager_counts = eager.get("num_candidates")
    if not isinstance(eager_counts, torch.Tensor):
        return
    for label in ("user", "item"):
        key = f"tb:user_export/{label}_embedding"
        eager_value = eager.get(key)
        compiled_value = compiled.get(key)
        if not isinstance(eager_value, torch.Tensor) or not isinstance(
            compiled_value, torch.Tensor
        ):
            logger.info("[PARITY][EMBEDDING-UNPAD][%s] missing", label)
            continue
        aligned = _unpad_candidate_embeddings(compiled_value, eager_counts)
        if aligned is None or aligned.shape != eager_value.shape:
            logger.info(
                "[PARITY][EMBEDDING-UNPAD][%s] shape mismatch eager=%s compiled=%s",
                label,
                tuple(eager_value.shape),
                tuple(compiled_value.shape),
            )
            continue
        diff = (eager_value - aligned).abs()
        logger.info(
            "[PARITY][EMBEDDING-UNPAD][%s] shape=%s max_abs=%.6e mean_abs=%.6e",
            label,
            tuple(aligned.shape),
            float(diff.max()),
            float(diff.mean()),
        )


def _compare(
    ref: Dict[str, torch.Tensor],
    test: Dict[str, torch.Tensor],
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Per-param weight agreement, reported as a DISTRIBUTION (not just the max).

    Pass criterion per param: ``max|a - b| <= atol + rtol * max|a|`` (the
    assert_close form). ``atol`` tolerates near-zero-init params (LayerNorm
    biases, freshly-init DCN layers) that a pure relative metric blows up. Since
    the two cells share the init, ``a - b`` IS the per-step update disagreement.
    """
    common = sorted(set(ref) & set(test))
    rows = []  # (name, abs_diff, rel_diff, ref, ok)
    shape_mismatch = 0
    mismatch_names: List[Tuple[str, Any, Any]] = []
    for k in common:
        a, b = ref[k], test[k]
        if a.numel() == 0 or b.numel() == 0:
            continue
        if a.shape != b.shape:
            # e.g. GSPMD-sharded compiled EC m/v (R/n, D) vs full eager (R, D):
            # same optimizer, different sharded representation -> not element-
            # comparable here. Count + skip rather than crash on (a - b).
            shape_mismatch += 1
            mismatch_names.append((k, tuple(a.shape), tuple(b.shape)))
            continue
        ref_mag = float(a.abs().max())
        abs_diff = float((a - b).abs().max())
        rel_diff = abs_diff / max(ref_mag, 1e-12)
        ok = abs_diff <= atol + rtol * ref_mag
        rows.append((k, abs_diff, rel_diff, ref_mag, ok))

    def _pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0

    abs_sorted = sorted(r[1] for r in rows)
    rel_sorted = sorted(r[2] for r in rows)
    worst = sorted(rows, key=lambda r: r[1], reverse=True)[:5]  # by ABS diff
    # Over-tolerance param names, worst REL first: a sign/direction gradient bug
    # shows as large REL diff (opposite-sign values), so ranking by rel surfaces
    # the seed params before abs-dominated (large-magnitude) ones.
    failed_names = [
        r[0]
        for r in sorted(
            (row for row in rows if not row[4]), key=lambda row: row[2], reverse=True
        )
    ]
    return {
        "num_params_compared": len(rows),
        "num_shape_mismatch": shape_mismatch,
        "mismatch_names": mismatch_names,
        "failed_names": failed_names,
        "num_failed": sum(1 for r in rows if not r[4]),
        "max_abs_diff": abs_sorted[-1] if abs_sorted else 0.0,
        "median_abs_diff": _pct(abs_sorted, 0.5),
        "p90_abs_diff": _pct(abs_sorted, 0.9),
        "max_rel_diff": rel_sorted[-1] if rel_sorted else 0.0,
        "median_rel_diff": _pct(rel_sorted, 0.5),
        "p90_rel_diff": _pct(rel_sorted, 0.9),
        "worst": [
            {"name": k, "abs_diff": d, "rel_diff": r, "ref": ref_mag, "pass": ok}
            for (k, d, r, ref_mag, ok) in worst
        ],
    }


def _log_grad_fqn_breakdown(
    ref: Dict[str, torch.Tensor],
    test: Dict[str, torch.Tensor],
    rtol: float,
    atol: float,
    top_n: int = 20,
) -> None:
    """Report eager-vs-compiled grad parity. Prints a single 'all match' line when
    clean, else the ``top_n`` worst FAILING params by |diff| (exact fqn). Pass
    criterion matches ``_compare`` (``abs <= atol + rtol*|ref|``).
    """
    rows: List[Tuple[str, float, float, float, bool]] = []
    for k in sorted(set(ref) & set(test)):
        a, b = ref[k], test[k]
        if a.numel() == 0 or b.numel() == 0:
            continue
        ref_mag = float(a.abs().max())
        abs_d = float((a - b).abs().max())
        rel_d = abs_d / max(ref_mag, 1e-12)
        rows.append((k, abs_d, rel_d, ref_mag, abs_d <= atol + rtol * ref_mag))

    fails = [r for r in rows if not r[4]]
    if not fails:
        logger.info(
            "[PARITY][GRAD-FQN] all %d grads MATCH (rtol=%.2g atol=%.2g)",
            len(rows),
            rtol,
            atol,
        )
        return
    logger.info(
        "[PARITY][GRAD-FQN] %d/%d grads fail (rtol=%.2g atol=%.2g):",
        len(fails),
        len(rows),
        rtol,
        atol,
    )
    for k, abs_d, rel_d, ref_mag, _ok in sorted(
        fails, key=lambda r: r[1], reverse=True
    )[:top_n]:
        logger.info(
            "[PARITY][GRAD-FQN]   %s  abs=%.2e rel=%.2e ref=%.2e",
            k,
            abs_d,
            rel_d,
            ref_mag,
        )


def _add_scale_free_metrics(exp_j: Dict[str, Any], exp_p: Dict[str, Any]) -> float:
    """Add scale-FREE per-example ratios to BOTH export dicts and return the
    n_samples ratio K (compiled/eager), for logging.

    ``label_wmean`` = label_wsum/weight_sum and ``pred_wmean`` = pred_wsum/
    weight_sum are the trustworthy multi-host correctness+masking gates: each is
    built per side from that side's OWN sums, so any common inflation (the compiled
    cell runs the num_processes-inflated GLOBAL batch) cancels -- they need NO P
    factor and hold for distinct data too.
    """

    def _vec(d: Dict[str, Any], k: str) -> Optional[torch.Tensor]:
        v = d.get(k)
        return v.flatten().to(torch.float64) if isinstance(v, torch.Tensor) else None

    _ns_j = _vec(exp_j, "n_samples")
    _ns_p = _vec(exp_p, "n_samples")
    K = (
        float(_ns_p[0]) / float(_ns_j[0])
        if _ns_j is not None and _ns_p is not None and float(_ns_j[0]) != 0.0
        else 1.0
    )
    for _d in (exp_j, exp_p):
        _ws = _vec(_d, "weight_sum")
        _lw = _vec(_d, "label_wsum")
        _pw = _vec(_d, "pred_wsum")
        if _ws is not None and _lw is not None:
            _d["label_wmean"] = _lw / _ws.clamp(min=1e-9)
        if _ws is not None and _pw is not None:
            _d["pred_wmean"] = _pw / _ws.clamp(min=1e-9)
    return K


def _log_worst_entries(
    af: torch.Tensor,
    bf: torch.Tensor,
    diff: torch.Tensor,
    eps: float,
    num_main_tasks: int,
    task_names: Optional[List[str]],
) -> None:
    """For a DIFF'd per-task export, log the MAIN-vs-MT-aux split (MAIN ~0 but MT-aux
    large => mt-only leak; MAIN also off => shared NRO rows misaligned upstream) and
    the top-5 worst entries by |diff| with task name."""

    def _name(i: int) -> str:
        if task_names is not None and i < len(task_names):
            return task_names[i]
        if num_main_tasks and i >= num_main_tasks:
            return f"mt_aux[{i - num_main_tasks}]"
        return f"#{i}"

    if num_main_tasks and af.numel() > num_main_tasks:
        main_d = float(diff[:num_main_tasks].max())
        mt_d = float(diff[num_main_tasks:].max())
        logger.info(
            f"[PARITY]       main-range max|d|={main_d:.2e}  "
            f"mt_aux-range max|d|={mt_d:.2e}"
        )
    for i in torch.argsort(diff, descending=True)[: min(5, af.numel())]:
        if float(diff[i]) <= eps:
            break
        ii = int(i)
        logger.info(
            f"[PARITY]       [{ii:>3} {_name(ii)}] cpu={float(af[ii]):>11.4f} "
            f"tpu={float(bf[ii]):>11.4f} d={float(bf[ii] - af[ii]):>+11.4f}"
        )


def _compare_one_export(
    key: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    is_rel: bool,
    is_gate: bool,
    is_raw: bool,
    rtol: float,
    eps: float,
    num_main_tasks: int,
    task_names: Optional[List[str]],
) -> Optional[bool]:
    """Compare one per-task export (eager ``a`` vs compiled ``b``), log its summary
    line (+ worst entries on a gated DIFF), and return the match verdict for a GATE
    key (else None -- info-only). ``is_raw`` appends the raw tpu/cpu ratio; a gate
    shows match/DIFF, a non-gate shows info."""
    af = a.flatten()
    bf = b.flatten()
    if af.shape != bf.shape:
        logger.info(
            f"[PARITY]   {key:<15} SHAPE DIFFERS  cpu={tuple(a.shape)} "
            f"tpu={tuple(b.shape)}  sum cpu={float(af.sum()):.2f} "
            f"tpu={float(bf.sum()):.2f}"
        )
        return False if is_gate else None
    diff = (af - bf).abs()
    max_d = float(diff.max())
    if is_rel:
        # Pass on relative OR a small absolute floor: near-zero per-task entries
        # make a pure relative check spuriously large.
        max_rel = float((diff / af.abs().clamp(min=1e-9)).max())
        ok = max_rel <= rtol or max_d <= 5e-3
        extra = f" max_rel={max_rel:.2e}"
    else:
        ok = max_d <= eps
        extra = ""
    # Raw compiled/eager ratio for the P-factored sums/loss (info, not divided out).
    note = ""
    if is_raw:
        cpu = float(af.sum()) or 1e-9
        note = f"  raw_ratio(tpu/cpu)={float(bf.sum()) / cpu:.1f}"
    status = "info "
    result: Optional[bool] = None
    if is_gate:
        result = ok
        status = "match" if ok else "DIFF "
    logger.info(
        f"[PARITY]   {key:<15} {status}  shape={tuple(af.shape)}  "
        f"sum cpu={float(af.sum()):.2f} tpu={float(bf.sum()):.2f}  "
        f"max|d|={max_d:.2e}{extra}{note}"
    )
    if result is False and af.numel() > 1:
        _log_worst_entries(af, bf, diff, eps, num_main_tasks, task_names)
    return result


def _log_export_breakdown(
    exp_j: Dict[str, Any],
    exp_p: Dict[str, Any],
    rtol: float = 2e-2,
    task_names: Optional[List[str]] = None,
    num_main_tasks: int = 0,
    num_processes: int = 1,
) -> Dict[str, bool]:
    """Compact jagged-vs-padded breakdown of the model's per-task loss /
    normalization / prediction exports, to localize a discrepancy.

    One summary line per export -- ``match`` or ``DIFF`` with shape, sums, and
    max diff. For exports that DIFFER, a few indented lines name the worst
    entries by |diff| (with the TASK NAME / main-vs-MT-aux tag). Exports that
    agree stay a single line; a shape mismatch (e.g. ``num_candidates`` length)
    is flagged on its own.

    ``pred_wsum`` (per-task Sum(weight*prediction)) is compared with a RELATIVE
    tolerance -- it is a large per-task sum where an absolute eps is meaningless.
    A pred_wsum DIFF means the REAL (weight!=0) predictions differ padded-vs-
    jagged (weight_sum/label_wsum being exact rules out data/weight drift), i.e.
    padding is leaking into the real forward. Returns ``{key: matched}`` for the
    GATE keys: scale-free ``label_wmean``/``pred_wmean`` at any num_processes, plus
    the raw ``weight_sum``/``label_wsum``/``pred_wsum``/``loss`` only at single-
    process (at multi-host those carry a benign replication/padding factor and are
    logged info-only, so they are absent from the returned dict).
    """
    eps = 1e-6
    # Scale-free ratios (label_wmean/pred_wmean) are the real multi-host gates: they
    # cancel the replication/padding factor and must match at any num_processes. Raw
    # SUMS and loss only compare exactly at num_processes==1; at multi-host the
    # compiled cell runs the num_processes-inflated GLOBAL batch, so they carry a
    # benign factor -- shown as raw_ratio=tpu/cpu, NOT divided out (a wrong divisor
    # would mask real signal).
    K = _add_scale_free_metrics(exp_j, exp_p)

    # rel-diff (not abs) keys: large per-task values / ratios where an absolute eps
    # is meaningless; residual is the benign XLA-vs-eager fp floor.
    rel_keys = {
        "loss",
        "pre_llw_losses",
        "pred_wsum",
        "label_wmean",
        "pred_wmean",
    }
    # Real correctness gates (must match at any P). Raw sums/loss join the gate set
    # only at single-process; at multi-host they are info-only (benign P factor).
    raw_keys = {"loss", "weight_sum", "label_wsum", "pred_wsum"}
    gate_keys = {"label_wmean", "pred_wmean"}
    if num_processes <= 1:
        gate_keys = gate_keys | raw_keys
    matched: Dict[str, bool] = {}
    if num_processes > 1:
        logger.info(
            "[PARITY] multi-host: gating on scale-free label_wmean/pred_wmean; raw "
            "sums/loss are info-only (tpu carries a benign factor; n_samples ratio "
            "K=%.1f incl. padding).",
            K,
        )
    logger.info("[PARITY] export breakdown (cpu=eager, tpu=compiled):")
    for key in (
        "label_wmean",
        "pred_wmean",
        "loss",
        "weight_sum",
        "label_wsum",
        "pred_wsum",
    ):
        a = exp_j.get(key)
        b = exp_p.get(key)
        if a is None or b is None:
            continue
        result = _compare_one_export(
            key,
            a,
            b,
            is_rel=key in rel_keys,
            is_gate=key in gate_keys,
            is_raw=key in raw_keys,
            rtol=rtol,
            eps=eps,
            num_main_tasks=num_main_tasks,
            task_names=task_names,
        )
        if result is not None:
            matched[key] = result
    return matched


def _build_cpu_tpu_model(padding: bool) -> Tuple[BlueReelsVDDTrainV3, Any]:
    """Seeded build of the 64-chip training model for CPU-vs-TPU parity:
    Returns (model, module_config).

    64-chip training config with seq-len sampling OFF (deterministic). Re-seeds
    identically so the compiled (padding=True) and eager (padding=False) builds
    share the same dense init. ``padding`` sets the model's enable_padding /
    static_shape_mode only -- the BATCH (loaded separately) is always raw jagged.
    Distinct from ``_build_model_and_batch`` (the padded-vs-jagged flow): this uses
    the e2e 1shard dims and builds no synthetic batch.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    seq = int(os.environ.get("RANKFM_S", "3072"))
    model_args = RankFMModelArgs(
        hash_size_cap=int(os.environ.get("PARITY_HASH_SIZE_CAP", "1000000")),
        l1_num_layers=3,
        l2_num_layers=11,
        max_uih_seq_len=seq,
        l2_max_seq_len=512,
        num_candidates=20,
        disable_seq_len_sampling=True,  # OFF: deterministic sequences for parity
        seq_ro_per_sample_budget=seq,
    )
    module_config, _feature_config = build_module_config(
        model_args, enable_padding=padding
    )
    # This comparison reads tb:user_export/{user,item}_embedding, and they are
    # off by default because training copies 168 MB/step host-ward for nothing.
    # Parity is the one caller that needs them.
    module_config.export_user_item_embedding = True
    model = build_model_from_config(
        module_config,
        device,
        bf16_training=_PARITY_BF16,
        attention_backend=_PARITY_ATTENTION_BACKEND,
    )
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    return model, module_config


def _load_raw_jagged_batches(
    model: BlueReelsVDDTrainV3, module_config: Any
) -> List[Any]:
    """Load the RAW JAGGED e2e batches that BOTH cpu_tpu cells consume.

    ``_load_e2e_batches(..., padding=True)`` returns the real jagged batch (no
    pad_batch_to_static): the compiled model PRTs it in-graph, the eager model
    (enable_padding=False, static_shape_mode off) consumes it natively -- matching
    the e2e, which also feeds eager real jagged input. MUST use the PADDING
    (enable_padding=True) module_config: padding=True routes through
    _resolve_nro_per_sample, which requires static_nro_per_sample_budget (absent
    from the eager enable_padding=False config). The batch is model-agnostic real
    data, so the same jagged tensors feed the eager model fine.
    """
    gcs = os.environ.get("PARITY_GCS_DATA_PATH", "")
    if not gcs:
        raise ValueError(
            "CPU-vs-TPU parity requires real data: set PARITY_GCS_DATA_PATH "
            "(e.g. gs://cq-mrs-core-shared_mrs_tpu_y317h/rankfm_1shard_data)."
        )
    return _load_e2e_batches(model, module_config, gcs)


def _run_cpu_tpu_parity(rtol: float, atol: float) -> None:
    """Multi-step CPU-eager vs TPU-compiled parity on one real e2e file.

    Consecutive training steps, two ways, on the SAME real e2e data file:
      * TPU-compiled -- enable_padding=True (static_shape_mode + PRT), dense lowered
        to TPU via recir GSPMD, embeddings via the selected PARITY_EMB_BACKEND. = train.py.
      * CPU-eager    -- enable_padding=False, plain PyTorch fwd+bwd+opt, no recir.
        = train_cpu_eager.

    Both cells read the SAME raw jagged dump file (BatchReplayPipe): the compiled
    cell PRTs it in-graph, the eager cell consumes it natively. Reports loss, dense
    grads, dense weights, and the per-task forward exports (weight_sum /
    label_wsum / pred_wsum). Reuses this module's proven compiled/eager/compare
    internals; the shared setup (jax init, dist env, precision) runs in main().
    """
    num_ranks = len(jax.devices())
    num_processes = jax.process_count()

    # 1) TPU-compiled cell FIRST -- eager's torchax env would corrupt the XLA
    #    lowering if it ran first. Use padding=True (PRT), the selected embedding
    #    backend, and gradient capture. The raw jagged batch is loaded via the
    #    padding config and reused by BOTH cells.
    model_c, mc_c = _build_cpu_tpu_model(padding=True)
    batches = _load_raw_jagged_batches(model_c, mc_c)
    global_bs = int(batches[0].ro_float_features.shape[0]) * num_processes
    comp_params, comp_loss, comp_exp, comp_grads = _compiled_step(
        model_c,
        batches,
        num_ranks,
        num_processes,
        global_bs,
        padding=True,
        emb_backend=_PARITY_EMB_BACKEND,
    )

    # 2) CPU-eager cell LAST. enable_padding=False (static_shape_mode off), consumes
    #    the SAME raw jagged batch natively. exports_out captures the forward
    #    weight_sum / label_wsum / pred_wsum.
    model_e, _mc_e = _build_cpu_tpu_model(padding=False)
    eager_exp: Dict[str, torch.Tensor] = {}
    eager_params, eager_loss, eager_grads = _eager_step(
        model_e,
        batches,
        num_processes,
        num_ranks=num_ranks,
        exports_out=eager_exp,
    )

    _log_embedding_parity(eager_exp, comp_exp)
    # 3) Compare. Mean loss reduction selects SUM gradient reduction, so compiled
    #    and raw eager gradients have the same global-mean scale.
    loss_failures = 0
    loss_abs_by_step: List[float] = []
    for step, (eager_step_loss, comp_step_loss) in enumerate(
        zip(eager_loss, comp_loss), start=1
    ):
        loss_abs = abs(eager_step_loss - comp_step_loss)
        loss_abs_by_step.append(loss_abs)
        if not math.isclose(
            eager_step_loss, comp_step_loss, rel_tol=rtol, abs_tol=atol
        ):
            loss_failures += 1
        logger.info(
            "[PARITY] LOSS step=%d eager=%.6f tpu_compiled=%.6f abs=%.3e",
            step,
            eager_step_loss,
            comp_step_loss,
            loss_abs,
        )
    max_loss_abs = max(loss_abs_by_step, default=0.0)
    cmp_w = _compare(eager_params, comp_params, rtol, atol)
    logger.info(
        "[PARITY] WEIGHTS compared=%d failed=%d median_abs=%.2e max_abs=%.2e "
        "max_rel=%.2e",
        cmp_w["num_params_compared"],
        cmp_w["num_failed"],
        cmp_w["median_abs_diff"],
        cmp_w["max_abs_diff"],
        cmp_w["max_rel_diff"],
    )
    # Dense-grad parity uses a RELAXED tolerance vs weights/loss: the residual
    # per-param grad diffs are the benign XLA-vs-eager fp floor (reduction order +
    # transcendentals in the HSTU attention / TPM tree head) accumulated into a few
    # small bias grads. The learnable_loss_weights module is EXCLUDED from the grad
    # check -- its s-grad is captured at different in-place clip_grad_value stages
    # eager-vs-compiled (a harness capture artifact, not a real diff). Override via
    # PARITY_GRAD_RTOL/ATOL.
    grad_rtol = float(os.environ.get("PARITY_GRAD_RTOL", "0.15"))
    grad_atol = float(os.environ.get("PARITY_GRAD_ATOL", "2e-3"))
    _grad_exclude: Tuple[str, ...] = ("learnable_loss_weights_module",)

    def _keep(name: str) -> bool:
        return not any(x in name for x in _grad_exclude)

    eager_grads_f = {k: v for k, v in eager_grads.items() if _keep(k)}
    comp_grads_f = {k: v for k, v in comp_grads.items() if _keep(k)}
    cmp_g = _compare(eager_grads_f, comp_grads_f, grad_rtol, grad_atol)
    logger.info(
        "[PARITY] GRADS compared=%d failed=%d median_abs=%.2e "
        "max_abs=%.2e max_rel=%.2e",
        cmp_g["num_params_compared"],
        cmp_g["num_failed"],
        cmp_g["median_abs_diff"],
        cmp_g["max_abs_diff"],
        cmp_g["max_rel_diff"],
    )
    _log_grad_fqn_breakdown(eager_grads_f, comp_grads_f, grad_rtol, grad_atol)
    # Forward exports: cpu=eager(jagged), tpu=compiled(padded). Logs
    # loss / weight_sum / label_wsum / pred_wsum match/DIFF, normalized to the
    # eager per-host basis (compiled runs the num_processes-inflated global batch).
    _log_export_breakdown(eager_exp, comp_exp, rtol, num_processes=num_processes)
    logger.info(
        "[PARITY] VERDICT weights_failed=%d grads_failed=%d "
        "losses_failed=%d max_loss_abs=%.3e",
        cmp_w["num_failed"],
        cmp_g["num_failed"],
        loss_failures,
        max_loss_abs,
    )


def main() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    # On GKE TPU, initialize the JAX distributed runtime BEFORE any jax op, like
    # the trainer (train.py:77): the PJRT plugin reads the coordinator + process
    # info from the TPU runtime metadata. The CPU-spoof path must NOT call this.
    if _PARITY_DEVICE == "tpu":
        jax.distributed.initialize()
        logger.info(
            f"[PARITY] jax.distributed.initialize() done: "
            f"process_count={jax.process_count()} device_count={jax.device_count()}"
        )
    # Mirror train.py's pre-trainer setup: resolve the dist env (RANK/WORLD_SIZE/
    # MASTER_*) and route torch.distributed barriers to CPU. RecTrainer does this
    # before building the family; replicate it so nothing differs from the bench.
    resolve_dist_env()
    ensure_gloo_process_group()
    force_cpu_barrier()
    # Match the bench (train.py): do NOT force matmul precision unless explicit.
    # Exception: the fp32 disambiguation run (PARITY_BF16=0) forces highest so XLA
    # uses exact fp32 matmuls -- else tf32-like fast math would reintroduce the
    # very noise we are trying to rule out, and eager (fp32) vs compiled would not
    # match near-exactly.
    prec = os.environ.get("PARITY_PRECISION")
    if prec:
        jax.config.update("jax_default_matmul_precision", prec)
    elif not _PARITY_BF16:
        jax.config.update("jax_default_matmul_precision", "highest")
    rtol = float(os.environ.get("PARITY_RTOL", "2e-2"))
    atol = float(os.environ.get("PARITY_ATOL", "1e-3"))

    devices = jax.devices()
    logger.info(
        f"[PARITY] device={_PARITY_DEVICE} jax_platform={devices[0].platform} "
        f"num_devices={len(devices)}"
    )
    if _PARITY_DEVICE != "tpu":
        raise RuntimeError(
            "CPU-eager vs TPU-compiled parity requires PARITY_DEVICE=tpu"
        )
    logger.info("[PARITY] CPU-eager vs TPU-compiled on real replay data")
    _run_cpu_tpu_parity(rtol, atol)
    return


if __name__ == "__main__":
    main()
