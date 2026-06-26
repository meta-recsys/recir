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
(unpadded) run cannot shard across a multi-device mesh, so this is SINGLE-DEVICE
only.

Run (single-device CPU; no TPU needed -- same graph, CPU XLA compiles it):
    PARITY_DEVICE=cpu PARITY_CPU_DEVICES=1 PARITY_BATCH_RANKS=8 \\
        python3 -u -m recir.examples.rankfm_mvai.parity_forward_loss

Knobs (env vars):
    PARITY_DEVICE       "cpu" (default) | "tpu" (single chip)
    PARITY_CPU_DEVICES  CPU-spoof device count (default "1"; CC is single-device)
    PARITY_BATCH_RANKS  global_bs = TEST_BATCH_SIZE * (PARITY_BATCH_RANKS or
                        num_ranks). Default "8" so the single-device batch is big
                        enough that static_shape_mode's L2-split math stays
                        non-degenerate.
    PARITY_INPUT        "random" (default; jagged num_candidates -- padding is
                        meaningful) | "uniform"
    PARITY_NUM_CANDIDATES  NRO candidates per request + static budget (default 32)
    PARITY_PRECISION    jax matmul precision: "highest" (fp32, default) | "default"
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
from hammer.v3.models.udd_fm_v3.batch_loader import BlueReelsVDDFeatureConfig
from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    RecirCompileSpec,
    RecirMode,
    run_step_and_sync,
    setup_dense_optimizer_and_warmup,
)
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.examples.rankfm.config import _build_static_ro_budgets, RankFMTrainerConfig
from recir.examples.rankfm.embedding import build_embedding_backend
from recir.examples.rankfm.train import (
    _resolve_nro_per_sample,
    make_padding_region_transforms,
    prepare_override_build_sample,
)
from recir.examples.rankfm_mvai.model_family import _udd_fm_v3_train_step
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.xsurface_test_utils import (
    _create_random_train_batch,
    _create_uniform_train_batch,
    build_model_from_config,
    TEST_BATCH_SIZE,
    TEST_MAX_UIH_SEQ_LEN,
    TestConfigProvider,
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
    global_bs = TEST_BATCH_SIZE * (_PARITY_BATCH_RANKS or num_ranks)

    config = TestConfigProvider().create_config()
    module_config = cast(BlueReelsVDDModuleConfig, config.module_config)
    feature_config = cast(BlueReelsVDDFeatureConfig, config.feature_config)
    # disable_seq_len_sampling=True
    module_config.hstu_seq_len_sampling_configs = None
    # Mirror prod's build_module_config sequence-shape coupling (config.py): the
    # HSTU history bound, L2 window, and output length must all be sized to the
    # test's actual sequence, or static_shape_mode's L2-split math goes negative
    # (l2_left_total = GROUP_SIZE - batch*(hstu_l2_max_seq_len + budget) -> < 0 ->
    # torch.arange(<0)). Output length must equal the candidate count (one output
    # token per candidate). Set unconditionally so both cells match.
    module_config.hstu_max_seq_len = TEST_MAX_UIH_SEQ_LEN
    module_config.hstu_l2_max_seq_len = max(1, TEST_MAX_UIH_SEQ_LEN // 4)
    module_config.hstu_max_output_len = _PARITY_NUM_CANDIDATES
    if padding:
        module_config.static_shape_mode = True
        module_config.static_nro_per_sample_budget = _PARITY_NUM_CANDIDATES
        module_config.static_ro_per_sample_budgets = _build_static_ro_budgets(
            module_config, _PARITY_SEQ_RO_BUDGET
        )

    model = build_model_from_config(module_config, device)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    feature_config.num_candidates_per_request = _PARITY_NUM_CANDIDATES
    feature_config.history_len = TEST_MAX_UIH_SEQ_LEN

    if _PARITY_INPUT == "random":
        batch = _create_random_train_batch(
            feature_config,
            device,
            batch_size=global_bs,
            num_candidates=_PARITY_NUM_CANDIDATES,
        )
    else:
        batch = _create_uniform_train_batch(
            feature_config, device, batch_size=global_bs
        )
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
) -> Tuple[Dict[str, torch.Tensor], int]:
    """One compiled step via the mvai_cloud pipeline; return (dense params, all-reduce).

    Mirrors RecirXLACompiledModule (rankfm_mvai/model_family.py): build the recir
    embedding backend, expand the override sample, build dispatchers, set up
    embedding storage, build the unified TensorLRAdamW dense optimizer, wire the
    PaddingRegionTransform, then compile + run one step. DCN is included (the
    model was built with use_inter_arch=True).
    """
    device = torch.device("cpu")
    model.train(True)
    trainer_cfg = RankFMTrainerConfig()
    module_config = model.module_config
    global_bs = TEST_BATCH_SIZE * (_PARITY_BATCH_RANKS or num_ranks)

    backend = build_embedding_backend("cpu")
    num_candidates_per_sample = _resolve_nro_per_sample(module_config, padding)
    padded_num_devices = math.lcm(num_ranks, backend.sc_modulus() * num_processes)

    # Override-build sample. PADDED: resize to the global padded shape via the
    # SAME helper prod/train.py uses (so sizing cannot drift). JAGGED: build from
    # the RAW batch -- the override's NRO must match the actual jagged candidate
    # count so the DCN inter-arch cat (r_dense vs sparse_embeddings) aligns.
    # prepare_override_build_sample's non-padding branch rectangular-resizes NRO
    # to global_bs * num_candidates, which would feed DCN a rectangular sparse
    # tensor against a jagged dense one.
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

    # Table configs from the EC BEFORE backend.setup mutates the model.
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
    # Embedding storage only (DMP for cpu; skipped single-host). Dense optimizer
    # is the unified one below, not the backend's. backend.setup returns Module;
    # for single-host CPU it returns the model unchanged, so cast back.
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
    runner = compile_gspmd_step(
        model, dense_opt, batch, dispatchers, spec, train_step_fn=_udd_fm_v3_train_step
    )
    run_step_and_sync(runner, model, dense_opt, batch)
    all_reduce = _count_all_reduce(runner)
    logger.info(f"[PARITY] compiled HLO all-reduce ops: {all_reduce}")
    return _dense_params(backend, model), all_reduce


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


def main() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    # Default to fp32 (highest) so the MXU matmuls are deterministic across cells.
    precision = os.environ.get("PARITY_PRECISION", "highest")
    jax.config.update("jax_default_matmul_precision", precision)
    rtol = float(os.environ.get("PARITY_RTOL", "2e-2"))
    atol = float(os.environ.get("PARITY_ATOL", "1e-3"))

    devices = jax.devices()
    logger.info(
        f"[PARITY] device={_PARITY_DEVICE} jax_platform={devices[0].platform} "
        f"num_devices={len(devices)} matmul_precision={precision}"
    )

    # Compiled-padded-vs-jagged is SINGLE-DEVICE only: the unpadded/jagged run
    # cannot shard across a multi-device mesh (recir's np.split needs an equal,
    # sample-aligned batch split -- which padding provides). Fail fast instead of
    # a cryptic np.split error deep in the jagged step.
    if len(devices) > 1:
        raise RuntimeError(
            f"compiled-padded-vs-jagged parity requires a single-device mesh "
            f"(got num_devices={len(devices)}). Use PARITY_DEVICE=cpu "
            f"PARITY_CPU_DEVICES=1 PARITY_BATCH_RANKS=8."
        )

    logger.info(
        f"[PARITY] input={_PARITY_INPUT} num_candidates={_PARITY_NUM_CANDIDATES} "
        f"seq_ro_budget={_PARITY_SEQ_RO_BUDGET}"
    )

    # Vary ONLY padding, both sides on the compiled path. Each build re-seeds
    # identically, so the dense init + batch are shared; only static_shape_mode +
    # the PaddingRegionTransform differ.
    model_p, batch_p, num_ranks, num_processes = _build_model_and_batch(padding=True)
    test_params, all_reduce = _compiled_step(
        model_p, batch_p, num_ranks, num_processes, padding=True
    )
    model_j, batch_j, _, _ = _build_model_and_batch(padding=False)
    ref_params, all_reduce_j = _compiled_step(
        model_j, batch_j, num_ranks, num_processes, padding=False
    )
    logger.info(f"[PARITY] all-reduce ops: padded={all_reduce} jagged={all_reduce_j}")

    # ref = jagged (unpadded) ground truth; test = the padded path under test.
    cmp = _compare(ref_params, test_params, rtol, atol)
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
        f"[PARITY] rel_diff  median={cmp['median_rel_diff']:.2e} "
        f"p90={cmp['p90_rel_diff']:.2e} max={cmp['max_rel_diff']:.2e}"
    )
    logger.info(
        f"[PARITY] tolerance (per param): |a-b| <= atol={atol:.1e} + rtol={rtol:.1e}*|a|"
    )
    for w in cmp["worst"]:
        logger.info(
            f"[PARITY]   worst: {w['name']} abs={w['abs_diff']:.2e} "
            f"rel={w['rel_diff']:.2e} ref={w['ref']:.2e} {'ok' if w['pass'] else 'FAIL'}"
        )
    logger.info(f"[PARITY] {'PASS' if passed else 'FAIL'}")

    _persist_result(
        {
            "device": _PARITY_DEVICE,
            "jax_platform": devices[0].platform,
            "num_devices": len(devices),
            "num_ranks": num_ranks,
            "num_processes": num_processes,
            "input": _PARITY_INPUT,
            "num_candidates": _PARITY_NUM_CANDIDATES,
            "matmul_precision": precision,
            "rtol": rtol,
            "atol": atol,
            "passed": passed,
            "all_reduce_ops": all_reduce,
            **cmp,
        }
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
