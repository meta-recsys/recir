#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Single-step padding-inertness parity for RankFM (udd_fm_v3), mvai path.

Compiled-PADDED vs compiled-JAGGED, varying ONLY padding. From the SAME seeded
init and SAME batch, take one compiled training step two ways -- once with
static-shape padding (PaddingRegionTransform + dummy candidates), once jagged --
and diff the resulting dense (over-arch) weights. If TPU padding is inert, the
padded dummy samples do not participate in loss / gradient / dense-weight
updates, so the two runs match.

This is the DCN-capable port of recir/examples/rankfm/parity_forward_loss.py. It
runs the SAME model (BlueReelsVDDTrainV3) through the SAME mvai_cloud compiled
pipeline the rankfm_mvai trainer uses (build_embedding_backend -> get_override ->
backend.setup -> setup_dense_optimizer_and_warmup -> compile_gspmd_step ->
run_step_and_sync, with the unified TensorLRAdamW dense optimizer). Unlike the
OSS rankfm image, the rankfm_mvai image stages the mrs/hpc DCN files, so the
DCN inter-arch stays ON -- this is the environment where DCN padding-inertness
is actually testable.

Both sides run the SAME compiled framework + optimizer; only padding differs --
so the result is a clean inertness verdict (optimizer math cancels). The jagged
(unpadded) run shards across a multi-device mesh by building num_ranks identical
per-device blocks (see _tile_batch_to_devices), so each device gets a jagged
block with equal total tokens; set PARITY_CPU_DEVICES>1 (or run on a multi-chip
mesh) to exercise it.

Run (single-device CPU; no TPU needed -- same graph, CPU XLA compiles it):
    PARITY_DEVICE=cpu PARITY_CPU_DEVICES=1 PARITY_BATCH_RANKS=8 \\
        python3 -u -m recir.examples.rankfm_mvai.parity_forward_loss
Multi-device CPU (2-device mesh, balanced jagged blocks):
    PARITY_DEVICE=cpu PARITY_CPU_DEVICES=2 PARITY_BATCH_RANKS=8 \\
        python3 -u -m recir.examples.rankfm_mvai.parity_forward_loss

Knobs (env vars):
    PARITY_DEVICE       "cpu" (default) | "tpu" (single chip)
    PARITY_EMB_BACKEND  embedding-lookup backend under test: "in_graph"
                        (default; ring-shift / FSDP tables lowered into the
                        compiled GSPMD program; the dedicated embedding optimizer
                        is traced in, so this uses
                        the two-optimizer compile path)
    PARITY_RINGSHIFT_MIN_ROWS / PARITY_FSDP_MIN_ROWS / PARITY_ON_DEVICE_INIT
                        in_graph table-sharding knobs (defaults 4096 / -1 / 1);
                        ignored by the cpu / tpu backends
    PARITY_CPU_DEVICES  CPU-spoof device count (default "1"; CC is single-device)
    PARITY_BATCH_RANKS  global_bs = TEST_BATCH_SIZE * (PARITY_BATCH_RANKS or
                        num_ranks). Default "8" so the single-device batch is big
                        enough that static_shape_mode's L2-split math stays
                        non-degenerate.
    PARITY_INPUT        "random" (default; jagged num_candidates -- padding is
                        meaningful) | "uniform"
    PARITY_NUM_CANDIDATES  NRO candidates per request + static budget (default 32)
    PARITY_RTOL / PARITY_ATOL  weight-compare tolerances (default 2e-2 / 1e-3)
    PARITY_GCS_DIR      gs:// dir to upload parity_result.json (default: tenant
                        bucket on TPU); unset disables upload
    PARITY_LOCAL_JSON   local result path (default /tmp/parity_result.json)
"""

from __future__ import annotations

import os

# Device selection MUST happen before jax is imported. CC is single-device: a
# 1-device mesh lets recir's np.split shard the UNPADDED (jagged) run trivially
# (it cannot on a multi-device mesh, where padding is required for an equal,
# sample-aligned split).
_PARITY_DEVICE: str = os.environ.get("PARITY_DEVICE", "cpu").lower()
if _PARITY_DEVICE == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    _cpu_devices: str = os.environ.get("PARITY_CPU_DEVICES", "1")
    _existing_xla_flags: str = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _existing_xla_flags:
        os.environ["XLA_FLAGS"] = (
            _existing_xla_flags
            + f" --xla_force_host_platform_device_count={_cpu_devices}"
        ).strip()

import json
import logging
import math
import sys
import time
from typing import Any, cast, Dict, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax  # @manual
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP entries
import torch
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    RecirCompileSpec,
    RecirMode,
    run_step_and_sync,
    setup_dense_optimizer_and_warmup,
)
from hammer.v3.models.udd_fm_v3.recir_core.compile import compile_step
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.examples.rankfm_mvai.config import (
    build_module_config,
    RankFMModelArgs,
    RankFMTrainerConfig,
)
from recir.examples.rankfm_mvai.embedding import build_embedding_backend
from recir.examples.rankfm_mvai.embedding.in_graph_embedding_backend import (
    SHARDED_TABLE_PREFIX,
)
from recir.examples.rankfm_mvai.model_family import (
    _udd_fm_v3_train_step,
    _udd_fm_v3_train_step_two_opt,
)
from recir.examples.rankfm_mvai.train_common import force_cpu_barrier, resolve_dist_env
from recir.examples.rankfm_mvai.xsurface_helpers import (
    _resolve_nro_per_sample,
    make_padding_region_transforms,
    prepare_override_build_sample,
)
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    _tile_dim0,
    _tile_last_dim,
    build_model_from_config,
    resize_kjt_b,
    TEST_MAX_UIH_SEQ_LEN,
)

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

_PARITY_INPUT: str = os.environ.get("PARITY_INPUT", "random").lower()
# Per-key RO history token budget for static_shape_mode -- full test history so
# real RO history is never truncated (isolates the experiment to NRO dummy-sample
# padding).
_PARITY_SEQ_RO_BUDGET: int = TEST_MAX_UIH_SEQ_LEN
# NRO candidates per request: drives BOTH the generated count AND the static
# budget (static_nro_per_sample_budget == num_candidates, mirroring prod).
_PARITY_NUM_CANDIDATES: int = int(os.environ.get("PARITY_NUM_CANDIDATES", "32"))
# Decouples global batch size from device count so a 1-device mesh still gets a
# non-degenerate batch. 0 => use num_ranks.
_PARITY_BATCH_RANKS: int = int(os.environ.get("PARITY_BATCH_RANKS", "8"))

# Embedding-lookup backend under test. "cpu" (torchrec DMP) and "tpu" (JAX
# SparseCore) keep the embedding update OFF the compiled graph -> single dense
# optimizer. "in_graph" lowers the ring-shift / FSDP tables INTO the GSPMD
# program as plain params, so it adds a dedicated embedding optimizer and uses
# the two-optimizer compile path (mirrors model_family._build_compiled_step).
_PARITY_EMB_BACKEND: str = os.environ.get("PARITY_EMB_BACKEND", "in_graph").lower()
# in_graph table-sharding thresholds (per-table row counts); ignored by cpu/tpu.
_PARITY_RINGSHIFT_MIN_ROWS: int = int(
    os.environ.get("PARITY_RINGSHIFT_MIN_ROWS", "4096")
)
_PARITY_FSDP_MIN_ROWS: int = int(os.environ.get("PARITY_FSDP_MIN_ROWS", "-1"))
_PARITY_ON_DEVICE_INIT: bool = os.environ.get(
    "PARITY_ON_DEVICE_INIT", "1"
).lower() not in ("0", "false", "no", "")


def _tile_batch_to_devices(batch: Any, num_ranks: int) -> Any:
    """Replicate a per-device batch into ``num_ranks`` identical contiguous blocks.

    The batch is built at the per-device sample count; tiling it W times along the
    batch axis yields W identical jagged blocks. recir shards every flat batch
    tensor with a per-tensor ``np.split(axis=0, W)`` (recir/lowering/xla_lower.py),
    so identical blocks make every cut land exactly on a device boundary -- equal
    samples AND equal total tokens per device -- even though per-sample lengths
    still vary within a block. This is the multi-device generalization of the
    jagged run (no per-sample padding). For num_ranks == 1 it is a no-op.

    Field handling mirrors ``_construct_override_build_sample`` (ro sized by the
    request batch, nro/labels/unary sized by the candidate batch); num_candidates
    is tiled (not recomputed) so sum(num_candidates) stays == the nro batch.
    """
    if num_ranks <= 1:
        return batch
    g_ro = batch.ro_float_features.shape[0] * num_ranks
    g_nro = batch.nro_float_features.shape[0] * num_ranks
    batch.ro_float_features = _tile_dim0(batch.ro_float_features, g_ro)
    batch.ro_id_list_features = resize_kjt_b(batch.ro_id_list_features, g_ro)
    batch.nro_float_features = _tile_dim0(batch.nro_float_features, g_nro)
    batch.nro_id_list_features = resize_kjt_b(batch.nro_id_list_features, g_nro)
    if batch.unary_features is not None:
        batch.unary_features = resize_kjt_b(batch.unary_features, g_nro)
    batch.num_candidates = _tile_dim0(batch.num_candidates, g_ro)
    batch.supervision_label = _tile_last_dim(batch.supervision_label, g_nro)
    batch.supervision_weight = _tile_last_dim(batch.supervision_weight, g_nro)
    all_labels = batch.all_labels
    if all_labels is not None:
        for k in list(all_labels.keys()):
            all_labels[k] = _tile_dim0(all_labels[k], g_nro)
    return batch


def _build_model_and_batch(
    padding: bool,
) -> Tuple[BlueReelsVDDTrainV3, Any, int, int]:
    """Seeded build of the model (DCN ON) + a deterministic global batch.

    Returns ``(model, batch, num_ranks, num_processes)``. Re-seeds identically on
    each call so the padded and jagged builds share the same dense init + batch.

    DCN inter-arch stays ON (``use_inter_arch`` keeps its default ``True``): the
    rankfm_mvai image stages the mrs/hpc DCN files, so ``_make_dcn`` imports
    cleanly here -- unlike the OSS rankfm image, where parity had to disable it.
    """
    num_ranks = len(jax.devices())
    num_processes = 1
    torch.manual_seed(42)
    device = torch.device("cpu")
    global_bs = int(os.environ.get("PARITY_GLOBAL_BS", "512"))
    seq = int(os.environ.get("RANKFM_S", "4096"))
    model_args = RankFMModelArgs(
        hash_size_cap=int(os.environ.get("PARITY_HASH_SIZE_CAP", "131072")),
        l1_num_layers=8,
        l2_num_layers=10,
        max_uih_seq_len=seq,
        l2_max_seq_len=512,
        num_candidates=10,
        disable_seq_len_sampling=True,
        seq_ro_per_sample_budget=seq,
    )
    module_config, feature_config = build_module_config(
        model_args, enable_padding=padding
    )
    model = build_model_from_config(module_config, device, bf16_training=True)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    if global_bs % num_ranks != 0:
        raise ValueError(
            f"global batch {global_bs} not divisible by num_ranks {num_ranks}."
        )
    bs = global_bs // num_ranks
    if _PARITY_INPUT == "random":
        batch = _create_random_train_batch(
            feature_config,
            device,
            batch_size=bs,
            num_candidates=model_args.num_candidates,
        )
    else:
        batch = _create_uniform_train_batch(feature_config, device, batch_size=bs)
    batch = _tile_batch_to_devices(batch, num_ranks)
    return model, batch, num_ranks, num_processes


def _dense_params(backend: Any, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot the dense (over-arch) params the dense optimizer actually steps.

    ``backend.dense_named_parameters`` is the canonical dense set (embeddings
    excluded). Both cells go through the same backend, so the two snapshots share
    names and intersect cleanly.
    """
    return {
        n: p.detach().clone()
        for n, p in backend.dense_named_parameters(model).items()  # pyre-ignore[16]
    }


def _count_all_reduce(runner: Any) -> int:
    """Best-effort count of 'all-reduce' collectives in the compiled HLO.

    On a single-device mesh this is expected to be 0. Diagnostic only -- never
    fail the run if the compiled object isn't reachable on this runner shape.
    """
    try:
        compiled_obj = runner.__dict__.get("_compiled_obj", runner)
        n = 0
        for mod in compiled_obj.gm.modules():
            if isinstance(mod, _XLAModule) and mod.xla_hlo_text:
                n += mod.xla_hlo_text.count("all-reduce")
        return n
    except Exception:  # noqa: BLE001 -- diagnostic only, must not fail the run
        return 0


def _compiled_step(
    model: BlueReelsVDDTrainV3,
    batch: Any,
    num_ranks: int,
    num_processes: int,
    padding: bool,
) -> Tuple[Dict[str, torch.Tensor], int, float, Dict[str, Any]]:
    """One compiled step via the mvai_cloud pipeline; return (dense params,
    all-reduce count, total scalar loss, small diagnostic exports).

    Mirrors RecirXLACompiledModule (rankfm_mvai/model_family.py): build the recir
    embedding backend, expand the override sample, build dispatchers, set up
    embedding storage, build the unified TensorLRAdamW dense optimizer, wire the
    PaddingRegionTransform, then compile + run one step. DCN is included (the
    model was built with use_inter_arch=True).
    """
    device = torch.device("cpu")
    model.train(True)
    module_config = model.module_config
    # Derive the global batch from the actual batch passed in.
    global_bs = batch.ro_float_features.shape[0]

    trainer_cfg = RankFMTrainerConfig()
    backend = build_embedding_backend(
        _PARITY_EMB_BACKEND,
        ringshift_min_rows=_PARITY_RINGSHIFT_MIN_ROWS,
        fsdp_min_rows=_PARITY_FSDP_MIN_ROWS,
        on_device_init=_PARITY_ON_DEVICE_INIT,
    )
    # cpu/torchrec DMP needs a gloo PG; mirror the family (reuse if already init).
    if not (dist.is_available() and dist.is_initialized()):
        backend.init_pg()
    num_candidates_per_sample = _resolve_nro_per_sample(module_config, padding)
    padded_num_devices = math.lcm(num_ranks, backend.sc_modulus() * num_processes)

    # Override-build sample. PADDED: resize to the global padded shape via the SAME
    # helper prod/train.py uses. JAGGED: build from the RAW batch so the override's
    # NRO matches the actual jagged candidate count (prepare_override_build_sample's
    # non-padding branch rectangular-resizes NRO to global_bs * num_candidates,
    # which would mismatch the jagged dense path -- the 5120-vs-2808 cat error).
    if padding:
        batch_sample, override_bs = prepare_override_build_sample(
            batch,
            module_config,
            global_bs,
            num_candidates_per_sample,
            padded_num_devices,
            enable_padding=True,
        )
    else:
        all_labels = batch.all_labels
        assert all_labels is not None
        batch_sample = _inputs_to_feature_bundle(
            batch.ro_id_list_features,
            None,
            batch.nro_id_list_features,
            batch.num_candidates,
            all_labels,
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
    model = cast(
        BlueReelsVDDTrainV3,
        backend.setup(model, table_configs, trainer_cfg.embedding_lr, num_processes),
    )
    unary_kjt = batch.unary_features
    assert unary_kjt is not None
    model.set_recir_kjt_keys(
        batch.ro_id_list_features.keys(),
        batch.nro_id_list_features.keys(),
        unary_kjt.keys(),
    )

    dense_params = list(
        backend.dense_named_parameters(model).values()
    )  # pyre-ignore[16]
    dense_opt, _warmup = setup_dense_optimizer_and_warmup(
        model, dense_params, trainer_cfg.dense_optimizer_config, device
    )
    # in_graph backend only: dedicated SGD opt over the in-graph table params.
    emb_named_fn = getattr(backend, "embedding_named_parameters", None)
    emb_params = emb_named_fn(model) if emb_named_fn is not None else {}
    emb_opt: torch.optim.Optimizer | None = None
    if emb_params:
        emb_opt = torch.optim.SGD(
            list(emb_params.values()), lr=trainer_cfg.embedding_lr, foreach=True
        )

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
        loss_reduction="sum",
        region_transforms=region_transforms,
    )
    if emb_opt is None:
        runner = compile_gspmd_step(
            model,
            dense_opt,
            batch,
            dispatchers,
            spec,
            train_step_fn=_udd_fm_v3_train_step,
        )
        losses, named = run_step_and_sync(runner, model, dense_opt, batch)
    else:
        parallel_mode = GSPMDParallel(
            "replicate",
            loss_reduction=spec.loss_reduction,
            num_ranks=num_ranks,
            num_processes=num_processes,
            fsdp_table_substr=SHARDED_TABLE_PREFIX,
        )
        runner = compile_step(
            _udd_fm_v3_train_step_two_opt,
            model,
            (dense_opt, emb_opt),
            batch,
            parallel_mode=parallel_mode,
            backend=XLAGSPMDBackend(),
            dispatchers=dispatchers,
            region_transforms=region_transforms,
        )
        losses, named = runner(model, dense_opt, emb_opt, *flatten_batch(batch))
        sync_device_state_to_torch(runner)
    # Capture the total scalar loss for the loss-value inertness check (.sum() is
    # robust whether the runner hands back a scalar total or a per-task vector).
    loss_value = float(losses.detach().to(torch.float64).sum())
    all_reduce = _count_all_reduce(runner)
    logger.info(
        f"[PARITY] compiled HLO all-reduce ops: {all_reduce} loss={loss_value:.6f}"
    )
    # Capture small per-task / normalization exports for the padded-vs-jagged
    # breakdown (which task's loss diverges; whether the loss_scale /
    # num_candidates normalization denominators differ). The big per-sample
    # tensors (prediction/label/weight) are left for a future metric compare.
    exports: Dict[str, Any] = {}
    if isinstance(named, dict):
        logger.info(f"[PARITY] step export keys: {sorted(named.keys())}")
        for k in ("loss", "loss_scale", "llw_weights", "num_candidates"):
            t = named.get(k)
            if isinstance(t, torch.Tensor):
                exports[k] = t.detach().to(torch.float64).cpu()
        # Per-task reductions of the big per-sample tensors. Their sample axis
        # differs padded-vs-jagged (1024 vs ~514), so only sample-dim sums are
        # comparable. weight_sum is the key masking check: correctly weight-0
        # dummy candidates make the total effective weight match jagged.
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
            exports["weight_sum"] = wf.sum(dim=-1).cpu()
            exports["label_wsum"] = (wf * labf).sum(dim=-1).cpu()
            exports["pred_wsum"] = (wf * predf).sum(dim=-1).cpu()
            exports["n_samples"] = torch.tensor([float(wf.shape[-1])])
    return _dense_params(backend, model), all_reduce, loss_value, exports


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
    for k in common:
        a, b = ref[k], test[k]
        if a.numel() == 0 or b.numel() == 0:
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
    return {
        "num_params_compared": len(rows),
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


def _persist_result(result: Dict[str, Any]) -> None:
    """Write parity_result.json locally and best-effort upload to GCS."""
    local_path = os.environ.get("PARITY_LOCAL_JSON", "/tmp/parity_result.json")
    with open(local_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(f"[PARITY] wrote result -> {local_path}")

    gcs_dir = os.environ.get("PARITY_GCS_DIR")
    if not gcs_dir and _PARITY_DEVICE == "tpu":
        gcs_dir = "gs://cq-mrs-core-shared_mrs_tpu_y317h/parity_runs/rankfm_mvai"
    if not gcs_dir:
        return
    gs_path = (
        f"{gcs_dir.rstrip('/')}/{time.strftime('%Y%m%d-%H%M%S')}/parity_result.json"
    )
    try:
        from google.cloud import storage  # lazy: only needed when uploading

        bucket = gs_path.split("/")[2]
        blob = "/".join(gs_path.split("/")[3:])
        storage.Client().bucket(bucket).blob(blob).upload_from_filename(local_path)
        logger.info(f"[PARITY] uploaded result -> {gs_path}")
    except Exception as e:  # noqa: BLE001 -- upload is best-effort, must not fail the run
        logger.warning(f"[PARITY] GCS upload failed (non-fatal): {e}")


def _log_export_breakdown(exp_j: Dict[str, Any], exp_p: Dict[str, Any]) -> None:
    """Compact jagged-vs-padded breakdown of the model's per-task loss /
    normalization exports, to localize a loss discrepancy.

    One summary line per export -- ``match`` or ``DIFF`` with shape, sums, and
    max abs diff. For exports that DIFFER, a few indented lines name the worst
    entries by |diff| (e.g. the offending task index). Exports that agree stay
    a single line; a shape mismatch (e.g. ``num_candidates`` length) is flagged
    on its own, since it points straight at a normalization-shape difference.
    """
    eps = 1e-6
    logger.info("[PARITY] export breakdown (j=jagged, p=padded):")
    for key in (
        "loss",
        "loss_scale",
        "llw_weights",
        "num_candidates",
        "weight_sum",
        "label_wsum",
        "pred_wsum",
        "n_samples",
    ):
        a = exp_j.get(key)
        b = exp_p.get(key)
        if a is None or b is None:
            logger.info(f"[PARITY]   {key:<15} not exported")
            continue
        af = a.flatten()
        bf = b.flatten()
        if af.shape != bf.shape:
            logger.info(
                f"[PARITY]   {key:<15} SHAPE DIFFERS  j={tuple(a.shape)} "
                f"p={tuple(b.shape)}  sum j={float(af.sum()):.2f} p={float(bf.sum()):.2f}"
            )
            continue
        diff = (af - bf).abs()
        max_d = float(diff.max())
        status = "match" if max_d <= eps else "DIFF "
        logger.info(
            f"[PARITY]   {key:<15} {status}  shape={tuple(a.shape)}  "
            f"sum j={float(af.sum()):.2f} p={float(bf.sum()):.2f}  max|d|={max_d:.2e}"
        )
        if max_d > eps and af.numel() > 1:
            for i in torch.argsort(diff, descending=True)[: min(5, af.numel())]:
                if float(diff[i]) <= eps:
                    break
                logger.info(
                    f"[PARITY]       [{int(i):>3}] j={float(af[i]):>11.4f} "
                    f"p={float(bf[i]):>11.4f} d={float(bf[i] - af[i]):>+11.4f}"
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
    force_cpu_barrier()
    rtol = float(os.environ.get("PARITY_RTOL", "2e-2"))
    atol = float(os.environ.get("PARITY_ATOL", "1e-3"))

    devices = jax.devices()
    logger.info(
        f"[PARITY] device={_PARITY_DEVICE} jax_platform={devices[0].platform} "
        f"num_devices={len(devices)}"
    )
    if _PARITY_EMB_BACKEND == "tpu" and _PARITY_DEVICE != "tpu":
        raise RuntimeError(
            "PARITY_EMB_BACKEND=tpu requires a real TPU chip (JAX SparseCore)."
        )
    logger.info(f"[PARITY] embedding_backend={_PARITY_EMB_BACKEND}")

    # Padded-vs-jagged parity: both cells on the compiled path; only padding (the
    # static_shape_mode + PaddingRegionTransform) differs. ref = jagged ground
    # truth; test = the padded path under test. WEIGHT parity is the verdict.
    model_p, batch_p, num_ranks, num_processes = _build_model_and_batch(padding=True)
    test_params, all_reduce, loss_padded, exp_p = _compiled_step(
        model_p, batch_p, num_ranks, num_processes, padding=True
    )
    model_j, batch_j, _, _ = _build_model_and_batch(padding=False)
    ref_params, all_reduce_j, loss_jagged, exp_j = _compiled_step(
        model_j, batch_j, num_ranks, num_processes, padding=False
    )
    logger.info(f"[PARITY] all-reduce ops: padded={all_reduce} jagged={all_reduce_j}")

    cmp = _compare(ref_params, test_params, rtol, atol)
    loss_abs = abs(loss_padded - loss_jagged)
    loss_rel = loss_abs / max(abs(loss_jagged), 1e-12)
    loss_ok = loss_abs <= atol + rtol * abs(loss_jagged)
    passed = cmp["num_failed"] == 0
    logger.info(
        f"[PARITY] compared {cmp['num_params_compared']} dense params; "
        f"{cmp['num_failed']} outside tol"
    )
    logger.info(
        f"[PARITY] abs_diff  median={cmp['median_abs_diff']:.2e} "
        f"p90={cmp['p90_abs_diff']:.2e} max={cmp['max_abs_diff']:.2e}"
    )
    logger.info(
        f"[PARITY] loss (diagnostic, non-gating): jagged={loss_jagged:.6f} "
        f"padded={loss_padded:.6f} abs={loss_abs:.2e} rel={loss_rel:.2e} "
        f"{'ok' if loss_ok else 'DIFF'}"
    )
    _log_export_breakdown(exp_j, exp_p)
    logger.info(
        f"[PARITY] {'PASS' if passed else 'FAIL'} (gated on WEIGHT parity) | "
        f"loss {'ok' if loss_ok else 'DIFF -- diagnostic, non-gating'}"
    )
    _persist_result(
        {
            "device": _PARITY_DEVICE,
            "jax_platform": devices[0].platform,
            "num_devices": len(devices),
            "num_ranks": num_ranks,
            "num_processes": num_processes,
            "embedding_backend": _PARITY_EMB_BACKEND,
            "rtol": rtol,
            "atol": atol,
            "passed": passed,
            "all_reduce_ops": all_reduce,
            "loss_padded": loss_padded,
            "loss_jagged": loss_jagged,
            "loss_abs_diff": loss_abs,
            "loss_passed": loss_ok,
            **cmp,
        }
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
