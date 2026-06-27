#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Single-step WEIGHT parity: CPU eager (fp32) vs compiled (XLA GSPMD on TPU).

Takes ONE training step (forward + backward + SGD) two ways, from the SAME
seeded init and SAME batch, then compares the resulting dense (over-arch)
weights:

  (A) EAGER baseline -- the real preprocessor + sparse arch + dense HSTU, run as
      eager PyTorch on CPU in fp32 (the MoT dtype validator is bypassed for this
      forward, exactly as the compiled path skips it via is_fx_tracing). The
      numerical ground truth.
  (B) COMPILED -- recir lowers the dense path to the accelerator and runs one
      step, using train.py's PROVEN explicit pipeline
      (``_compile_with_xsurface_dispatchers`` = recir.trace + install_graph +
      compile_from_gm + make_compiled_runner), NOT the ``compile()`` decorator
      whose lazy-lowering torchax-env handoff breaks on TPU in this container.

Run (validated path = single-host TPU via the rankfm_tpu image; see
rankfm-parity-4chip.yaml for the 2x2x1 / 4-chip JobSet):
    PARITY_DEVICE=tpu python3 -u -m recir.examples.rankfm.parity_forward_loss

    # Local CPU-spoof (2 fake devices) is the code default but is NOT verified
    # end-to-end: a `buck2 run` currently hits a third-party ffmpeg link diamond
    # under the default build mode, and recir's compiled *execution* on CPU is
    # unproven (only GPU/TPU are exercised). Left as the default for local dev.
    buck2 run //recir/examples/rankfm:parity_forward_loss

Knobs (env vars):
    PARITY_DEVICE     "cpu" (spoof, default) | "tpu"
    PARITY_MULTIHOST  "1" to call jax.distributed.initialize() (multi-host TPU
                      only; a single 2x2x1 host does not need it)
    PARITY_PRECISION  jax matmul precision: "highest" (fp32, default) | "default" (bf16)
    PARITY_RTOL       relative tolerance for the weight compare (default 2e-2)
    PARITY_ATOL       absolute tolerance (default 1e-3)
    PARITY_GCS_DIR    gs:// dir to upload parity_result.json (default: tenant
                      bucket on TPU); unset disables upload
    PARITY_LOCAL_JSON local result path (default /tmp/parity_result.json)
"""

from __future__ import annotations

import os

# Device selection MUST happen before jax is imported. Default to CPU-spoof
# with >= 2 logical devices so the GSPMD (replicate) path runs locally.
_PARITY_DEVICE: str = os.environ.get("PARITY_DEVICE", "cpu").lower()
if _PARITY_DEVICE == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    _existing_xla_flags: str = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in _existing_xla_flags:
        os.environ["XLA_FLAGS"] = (
            _existing_xla_flags + " --xla_force_host_platform_device_count=2"
        ).strip()

import copy
import json
import logging
import sys
import time
from typing import Any, Dict, Optional, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import hammer.v3.modules.core.features.bundle as feature_bundle
import jax  # @manual
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP entries
import torch
import torch.nn as nn
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.examples.rankfm.train import (
    _compile_with_xsurface_dispatchers,
    _extract_train_step_args,
)
from recir.lowering.fbgemm_fallback import _batched_complete_cumsum_impl
from recir.lowering.xla_lower import _XLAModule
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _build_overrides,
    _create_uniform_train_batch,
    DummyPreprocessorStack,
    DummySparseArch,
    TEST_BATCH_SIZE,
    TEST_MAX_UIH_SEQ_LEN,
    TEST_NUM_CANDIDATES,
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

_LR = 0.01


def _build_model_and_batch() -> Tuple[BlueReelsVDDTrainV3, Any, int]:
    """Seeded build of the model + a deterministic uniform global batch."""
    num_ranks = len(jax.devices())
    torch.manual_seed(42)
    device = torch.device("cpu")
    global_bs = TEST_BATCH_SIZE * num_ranks
    model, feature_config = _build_full_model(device, disable_seq_len_sampling=True)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    feature_config.num_candidates_per_request = TEST_NUM_CANDIDATES
    feature_config.history_len = TEST_MAX_UIH_SEQ_LEN
    batch = _create_uniform_train_batch(feature_config, device, batch_size=global_bs)

    ro_kjt = batch.ro_id_list_features
    nro_kjt = batch.nro_id_list_features
    unary_kjt = batch.unary_features
    assert unary_kjt is not None
    model.set_recir_kjt_keys(ro_kjt.keys(), nro_kjt.keys(), unary_kjt.keys())
    return model, batch, num_ranks


def _dense_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Trainable params EXCLUDING the override scaffolding (Dummy* modules).

    On the compiled model the preprocessor/sparse-arch are swapped for dummies;
    skipping them leaves the dense over-arch params, which have the same names as
    in the (un-overridden) eager baseline, so the two dicts intersect cleanly.
    """
    skip = tuple(
        name + "."
        for name, mod in model.named_modules()
        if isinstance(mod, (DummyPreprocessorStack, DummySparseArch))
    )
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not n.startswith(skip)
    }


def _count_all_reduce(compiled_obj: Any) -> int:
    """Count 'all-reduce' collective ops across the compiled XLA HLO text.

    0 means the GSPMD gradient all-reduce is absent from the lowered graph
    (i.e. grads are NOT being summed across the replicas). Mirrors how
    gspmd_test_utils.assert_hlo_has_collective inspects ``_XLAModule.xla_hlo_text``.
    """
    n = 0
    for mod in compiled_obj.gm.modules():
        if isinstance(mod, _XLAModule):
            hlo = mod.xla_hlo_text
            if hlo:
                n += hlo.count("all-reduce")
    return n


# pyre-ignore -- raw-tensor train-step signature mirrors train.py exactly
def _train_step_fn(
    mod: nn.Module,
    opt: torch.optim.Optimizer,
    emb_opt: Optional[torch.optim.Optimizer],
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
) -> None:
    all_losses, _ = mod(
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
    )
    loss: torch.Tensor = all_losses.sum()
    loss.backward()
    opt.step()


def _eager_step(
    model: BlueReelsVDDTrainV3, batch: Any, num_ranks: int
) -> Dict[str, torch.Tensor]:
    """One eager SGD step on CPU in fp32; return dense params.

    Real modules (no overrides). We run a true fp32 forward (bf16 embeddings
    promote through fp32 weights) to match the compiled path's fp32 dense flow.
    The MoT bundle's runtime dtype validator would reject fp32 in a bf16-declared
    slot -- but the COMPILED path skips that exact check via is_fx_tracing(), so
    we opt out here too (process-local, restored in finally). ``loss / num_ranks``
    mirrors test_dlrmv3_numerical_parity: the compiled GSPMD path sums grads
    across the num_ranks replicas.
    """
    model.train(True)
    opt = torch.optim.SGD(model.parameters(), lr=_LR, foreach=False)
    args = _extract_train_step_args(batch)
    orig_validate = feature_bundle._validate_dtype
    feature_bundle._validate_dtype = lambda *a, **k: None  # pyre-ignore[8]
    try:
        all_losses, _ = model(*args)
        loss = all_losses.sum()
        (loss / num_ranks).backward()
        opt.step()
    finally:
        feature_bundle._validate_dtype = orig_validate
    return _dense_params(model)


def _compiled_step(
    model: BlueReelsVDDTrainV3, batch: Any, num_ranks: int
) -> Tuple[Dict[str, torch.Tensor], int]:
    """One compiled SGD step on the accelerator; return (dense params, all-reduce count).

    Uses train.py's PROVEN explicit pipeline (recir.trace + install_graph +
    compile_from_gm + make_compiled_runner) so recir manages its torchax env
    during lowering -- the path that runs on TPU in this image.
    """
    model.train(True)
    global_bs = TEST_BATCH_SIZE * num_ranks
    all_labels = batch.all_labels
    assert all_labels is not None
    batch_sample = _inputs_to_feature_bundle(
        batch.ro_id_list_features,
        None,
        batch.nro_id_list_features,
        batch.num_candidates,
        all_labels,
    )
    overrides = _build_overrides(
        model.recblock, batch_sample, torch.device("cpu"), batch_size=global_bs
    )
    opt = torch.optim.SGD(model.parameters(), lr=_LR, foreach=False)

    train_step = _compile_with_xsurface_dispatchers(
        model,
        opt,
        None,  # no dedicated embedding optimizer in the parity harness
        batch,
        overrides,
        _train_step_fn,
        parallel_mode=GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_ranks,
            num_processes=1,
        ),
        backend=XLAGSPMDBackend(),
        region_transforms=[],
    )
    train_step(model, opt, None, *_extract_train_step_args(batch))
    sync_device_state_to_torch(train_step)
    compiled_obj = train_step.__dict__["_compiled_obj"]
    all_reduce = _count_all_reduce(compiled_obj)
    logger.info(f"[PARITY] compiled HLO all-reduce ops: {all_reduce}")
    return _dense_params(model), all_reduce


def _compare(
    eager: Dict[str, torch.Tensor],
    compiled: Dict[str, torch.Tensor],
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    """Per-param weight agreement, reported as a DISTRIBUTION (not just the max).

    Pass criterion is ``torch.testing.assert_close`` style, per param:
        max|a - b| <= atol + rtol * max|a|
    This correctly tolerates near-zero-init params (e.g. LayerNorm biases) via
    ``atol`` -- a pure relative metric divides their tiny values into noise and
    blows up to ~1.0. Note ``a - b == da - db`` (the shared init cancels), so
    ``abs_diff`` IS the per-step update disagreement, just with a robust
    (weight-scale + atol) denominator instead of the noisy update magnitude.
    """
    common = sorted(set(eager) & set(compiled))
    rows = []  # (name, abs_diff, rel_diff, ref, ok)
    for k in common:
        a, b = eager[k], compiled[k]
        # Skip zero-element params -- .max() throws on numel()==0.
        if a.numel() == 0 or b.numel() == 0:
            continue
        ref = float(a.abs().max())
        abs_diff = float((a - b).abs().max())
        rel_diff = abs_diff / max(ref, 1e-12)
        ok = abs_diff <= atol + rtol * ref
        rows.append((k, abs_diff, rel_diff, ref, ok))

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
            {"name": k, "abs_diff": d, "rel_diff": r, "ref": ref, "pass": ok}
            for (k, d, r, ref, ok) in worst
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
        gcs_dir = "gs://cq-mrs-core-shared_mrs_tpu_y317h/parity_runs/rankfm"
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
    # Default to fp32 (highest) so the MXU matmuls match the fp32 eager path.
    precision = os.environ.get("PARITY_PRECISION", "highest")
    jax.config.update("jax_default_matmul_precision", precision)
    rtol = float(os.environ.get("PARITY_RTOL", "2e-2"))
    atol = float(os.environ.get("PARITY_ATOL", "1e-3"))

    devices = jax.devices()
    logger.info(
        f"[PARITY] device={_PARITY_DEVICE} jax_platform={devices[0].platform} "
        f"num_devices={len(devices)} matmul_precision={precision}"
    )

    model, batch, num_ranks = _build_model_and_batch()
    baseline = copy.deepcopy(model)  # clean reference before overrides mutate `model`

    compiled_params, all_reduce = _compiled_step(model, batch, num_ranks)
    eager_params = _eager_step(baseline, batch, num_ranks)

    cmp = _compare(eager_params, compiled_params, rtol, atol)
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
    logger.info(f"[PARITY] all-reduce ops in compiled HLO = {all_reduce}")
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
