#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Forward-only latency/throughput benchmark for RankFM (udd_fm_v3).

Times the forward(+loss) of BlueReelsVDDTrainV3 through the same recir compiled
GSPMD pipeline the trainer uses, but with a forward-only step (no backward /
opt.step). Padded static-shape path, same as prod/train.py. Config via FWD_* env
vars (see the _FWD_* block below). FWD_SWEEP runs a grid: one or more
';'-separated axes, each "axis=v1,v2,.." (multiple axes -> cartesian product),
e.g. FWD_SWEEP="global_bs=128,256;seq=1024,2048".

  FWD_DEVICE=cpu python3 -u -m recir.examples.rankfm_mvai.forward_benchmark
  FWD_DEVICE=tpu FWD_EMB_BACKEND=in_graph python3 -u -m ...forward_benchmark
"""

from __future__ import annotations

import os

# Device selection MUST happen before jax is imported.
_FWD_DEVICE: str = os.environ.get("FWD_DEVICE", "cpu").lower()
if _FWD_DEVICE == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import itertools
import logging
import math
import statistics
import threading
import time
from typing import Any, cast, Dict, List, Tuple

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax  # @manual
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers SPMD_DECOMP entries
import torch
import torch.distributed as dist
from hammer.v3.models.udd_fm_v3.recir_core import (
    compile_gspmd_step,
    RecirCompileSpec,
    RecirMode,
    setup_dense_optimizer_and_warmup,
)
from hammer.v3.models.udd_fm_v3.recir_core.compile import compile_step
from hammer.v3.models.udd_fm_v3.recir_core.trace_sample import flatten_batch
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from recir.backends import XLAGSPMDBackend
from recir.decompositions import SPMD_DECOMP_TABLE
from recir.distributed.parallel_mode import GSPMDParallel
from recir.embedding.in_graph import SHARDED_TABLE_PREFIX
from recir.examples.rankfm_mvai.config import (
    build_module_config,
    build_optimizer_config,
    RankFMModelArgs,
)
from recir.examples.rankfm_mvai.rankfm_batch_utils import (
    create_random_train_batch,
    resize_kjt_batch,
    tile_dim0,
    tile_last_dim,
)
from recir.examples.rankfm_mvai.rankfm_embedding_backends import build_embedding_backend
from recir.examples.rankfm_mvai.rankfm_model_utils import build_model_from_config
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

# Register the gr::batched_complete_cumsum decomposition if the C++ lib didn't load.
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

_FWD_EMB_BACKEND: str = os.environ.get(
    "FWD_EMB_BACKEND", "cpu"
).lower()  # cpu|in_graph|tpu
_FWD_GLOBAL_BS: int = int(os.environ.get("FWD_GLOBAL_BS", "64"))
_FWD_NUM_CANDIDATES: int = int(os.environ.get("FWD_NUM_CANDIDATES", "32"))
_FWD_SEQ: int = int(os.environ.get("FWD_SEQ", "4096"))  # max UIH seq / RO budget
_FWD_L2_SEQ: int = int(os.environ.get("FWD_L2_SEQ", "512"))
_FWD_L1_LAYERS: int = int(os.environ.get("FWD_L1_LAYERS", "8"))
_FWD_L2_LAYERS: int = int(os.environ.get("FWD_L2_LAYERS", "10"))
_FWD_HASH_SIZE_CAP: int = int(os.environ.get("FWD_HASH_SIZE_CAP", "131072"))
_FWD_WARMUP: int = int(os.environ.get("FWD_WARMUP", "5"))
_FWD_STEPS: int = int(os.environ.get("FWD_STEPS", "20"))
# tpu backend only: drop SparseCore partition-overflow ids instead of aborting.
_FWD_ALLOW_ID_DROPPING: bool = os.environ.get(
    "FWD_ALLOW_ID_DROPPING", "1"
).lower() not in ("0", "false", "no", "")
# in_graph on-device meta init (default True, matching the prod config). True:
# sharded tables are meta params generated on-device via jax.random (each chip
# makes its own shard) -- REQUIRED when the table is too big to host-allocate,
# and the intended path on multi-host. False: init a real host tensor, copied to
# device once + cached -- use only for a SINGLE-host benchmark of a table that
# fits host RAM; it avoids meta weights being re-materialized inside the timed
# loop (forward-only doesn't persist them across steps like training does).
_FWD_ON_DEVICE_INIT: bool = os.environ.get("FWD_ON_DEVICE_INIT", "1").lower() not in (
    "0",
    "false",
    "no",
    "",
)
# Profiling (TPU only). mode "trace" = start_trace/stop_trace (recir-native,
# default); "server" = start_server + xprof remote capture (D101763366 fallback).
_FWD_PROFILE: bool = os.environ.get("FWD_PROFILE", "0").lower() not in (
    "0",
    "false",
    "no",
    "",
)
_FWD_PROFILE_MODE: str = os.environ.get("FWD_PROFILE_MODE", "trace").lower()
_FWD_PROFILE_DIR: str = os.environ.get("FWD_PROFILE_DIR", "/tmp/rankfm_fwd_profile")
_FWD_PROFILER_PORT: int = int(os.environ.get("FWD_PROFILER_PORT", "9999"))
_FWD_PROFILE_DURATION_MS: int = int(os.environ.get("FWD_PROFILE_DURATION_MS", "30000"))
# trace mode: how many steps to actually capture (a short window inside the timed
# loop). Tracing all steps floods the ~1M-event cap (esp. with this image's
# hardware-counter dump) and evicts the device XLA-op rows -- the 64-chip bench
# profiles only ~5 steps for the same reason.
_FWD_PROFILE_STEPS: int = int(os.environ.get("FWD_PROFILE_STEPS", "3"))

# FWD_SWEEP="axis=v1,v2;axis2=.." -> cartesian grid; maps axis name -> _FWD_* global.
_SWEEP_AXES: Dict[str, str] = {
    "seq": "_FWD_SEQ",
    "global_bs": "_FWD_GLOBAL_BS",
    "num_candidates": "_FWD_NUM_CANDIDATES",
    "l1_layers": "_FWD_L1_LAYERS",
    "l2_layers": "_FWD_L2_LAYERS",
    "l2_seq": "_FWD_L2_SEQ",
    "hash_size_cap": "_FWD_HASH_SIZE_CAP",
    "emb_backend": "_FWD_EMB_BACKEND",
}


def _parse_sweep() -> List[Tuple[str, List[str]]] | None:
    """Parse FWD_SWEEP into [(axis, values), ...]; None if unset. Axes are
    ';'-separated, each 'axis=v1,v2,...'; >1 axis -> cartesian grid."""
    raw = os.environ.get("FWD_SWEEP", "").strip()
    if not raw:
        return None
    specs: List[Tuple[str, List[str]]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"FWD_SWEEP spec must be 'axis=v1,v2,...'; got {part!r}")
        axis, vals = part.split("=", 1)
        axis = axis.strip()
        if axis not in _SWEEP_AXES:
            raise ValueError(f"FWD_SWEEP axis {axis!r} not in {sorted(_SWEEP_AXES)}")
        values = [v.strip() for v in vals.split(",") if v.strip()]
        if not values:
            raise ValueError(f"FWD_SWEEP axis {axis!r} has no values")
        specs.append((axis, values))
    if not specs:
        raise ValueError(f"FWD_SWEEP has no axes: {raw!r}")
    return specs


def _set_axis(axis: str, value: str) -> None:
    name = _SWEEP_AXES[axis]
    globals()[name] = value if isinstance(globals()[name], str) else int(value)


def _udd_fm_v3_forward_step(
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
    candidate_sum: torch.Tensor,
    loss_denom_source: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """recir_train_step minus backward()/opt.step() (opt unused). Same named
    signature as _udd_fm_v3_train_step so PaddingRegionTransform can bind args."""
    all_losses, named_tensors = mod(
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
        candidate_sum,
        loss_denom_source,
    )
    detached_named: Dict[str, torch.Tensor] = {
        k: (v.detach() if isinstance(v, torch.Tensor) else v)
        for k, v in named_tensors.items()
    }
    return all_losses.detach(), detached_named


def _tile_batch_to_devices(batch: Any, num_ranks: int) -> Any:
    """Replicate the per-device batch into num_ranks identical blocks so recir's
    per-tensor np.split(axis=0) lands on device boundaries. No-op if num_ranks==1."""
    if num_ranks <= 1:
        return batch
    g_ro = batch.ro_float_features.shape[0] * num_ranks
    g_nro = batch.nro_float_features.shape[0] * num_ranks
    batch.ro_float_features = tile_dim0(batch.ro_float_features, g_ro)
    batch.ro_id_list_features = resize_kjt_batch(batch.ro_id_list_features, g_ro)
    batch.nro_float_features = tile_dim0(batch.nro_float_features, g_nro)
    batch.nro_id_list_features = resize_kjt_batch(batch.nro_id_list_features, g_nro)
    if batch.unary_features is not None:
        batch.unary_features = resize_kjt_batch(batch.unary_features, g_nro)
    batch.num_candidates = tile_dim0(batch.num_candidates, g_ro)
    batch.supervision_label = tile_last_dim(batch.supervision_label, g_nro)
    batch.supervision_weight = tile_last_dim(batch.supervision_weight, g_nro)
    all_labels = batch.all_labels
    if all_labels is not None:
        for k in list(all_labels.keys()):
            all_labels[k] = tile_dim0(all_labels[k], g_nro)
    return batch


def _build_model_and_batch(
    num_ranks: int,
    num_processes: int,
) -> Tuple[BlueReelsVDDTrainV3, Any, Any, int]:
    """Seeded model (padding ON) + this process's PER-PROCESS batch. recir feeds
    per-process batches (global_bs / num_processes) and inflates dim0 by
    num_processes to form the global GSPMD array. Returns (model, batch,
    module_config, global_bs) where global_bs is the TRUE global batch."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    model_args = RankFMModelArgs(
        hash_size_cap=_FWD_HASH_SIZE_CAP,
        l1_num_layers=_FWD_L1_LAYERS,
        l2_num_layers=_FWD_L2_LAYERS,
        max_uih_seq_len=_FWD_SEQ,
        l2_max_seq_len=_FWD_L2_SEQ,
        num_candidates=_FWD_NUM_CANDIDATES,
        disable_seq_len_sampling=True,
        seq_ro_per_sample_budget=_FWD_SEQ,
    )
    module_config, feature_config = build_module_config(model_args, enable_padding=True)
    model = build_model_from_config(module_config, device, bf16_training=True)
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    if _FWD_GLOBAL_BS % num_ranks != 0:
        raise ValueError(
            f"FWD_GLOBAL_BS={_FWD_GLOBAL_BS} not divisible by device count {num_ranks}."
        )
    # Per-process batch = global_bs / num_processes, tiled across this process's
    # local devices (num_ranks // num_processes). Single-host: num_local ==
    # num_ranks, so this is the global batch (unchanged).
    num_local = num_ranks // num_processes
    batch = create_random_train_batch(
        feature_config,
        device,
        batch_size=_FWD_GLOBAL_BS // num_ranks,
        num_candidates=model_args.num_candidates,
    )
    batch = _tile_batch_to_devices(batch, num_local)
    return model, batch, module_config, _FWD_GLOBAL_BS


def _compile_forward_runner(
    model: BlueReelsVDDTrainV3,
    batch: Any,
    module_config: Any,
    num_ranks: int,
    num_processes: int,
) -> Any:
    """Build backend + override + dense opt, then compile the forward-only step
    (mirrors parity_forward_loss._compiled_step, padded path)."""
    device = torch.device("cpu")
    model.train(True)
    # TRUE global batch (the fed `batch` is this process's per-process shard); the
    # override + padding transform must be sized at global, like the trainer.
    global_bs = _FWD_GLOBAL_BS

    if _FWD_EMB_BACKEND == "in_graph":
        backend = build_embedding_backend(
            _FWD_EMB_BACKEND,
            on_device_init=_FWD_ON_DEVICE_INIT,
            embedding_tables=module_config.embedding_tables,
        )
    else:
        backend = build_embedding_backend(_FWD_EMB_BACKEND)
    # cpu/torchrec DMP needs a gloo PG; reuse if already initialized.
    if not (dist.is_available() and dist.is_initialized()):
        backend.init_pg()

    num_candidates_per_sample = _resolve_nro_per_sample(
        module_config, enable_padding=True
    )
    padded_num_devices = math.lcm(num_ranks, backend.sc_modulus() * num_processes)

    batch_sample, override_bs = prepare_override_build_sample(
        batch,
        module_config,
        global_bs,
        num_candidates_per_sample,
        padded_num_devices,
        enable_padding=True,
    )

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
    # tpu backend: flip id-dropping on the SparseCoreLookup post-setup (reads it
    # per-call); no-op for cpu / in_graph (no _jax_ec).
    if _FWD_ALLOW_ID_DROPPING:
        jax_ec = getattr(backend, "_jax_ec", None)
        lookup = getattr(jax_ec, "_lookup", None) if jax_ec is not None else None
        if lookup is not None:
            lookup._allow_id_dropping = True
    unary_kjt = batch.unary_features
    assert unary_kjt is not None
    model.set_recir_kjt_keys(
        batch.ro_id_list_features.keys(),
        batch.nro_id_list_features.keys(),
        unary_kjt.keys(),
    )

    # Dense opt is required by the compile but never stepped (forward-only).
    dense_params = list(embedding_setup.dense_named_parameters.values())
    dense_opt, _warmup = setup_dense_optimizer_and_warmup(
        model, dense_params, build_optimizer_config().dense.default, device
    )

    region_transforms = make_padding_region_transforms(
        num_candidates_per_sample, padded_num_devices, num_processes
    )
    if _FWD_EMB_BACKEND == "in_graph":
        # in_graph tables need fsdp_table_substr sharding, which compile_gspmd_step
        # doesn't set -> drive compile_step directly (dense opt only, unstepped).
        parallel_mode = GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_ranks,
            num_processes=num_processes,
            fsdp_table_substr=SHARDED_TABLE_PREFIX,
        )
        runner = compile_step(
            _udd_fm_v3_forward_step,
            model,
            (dense_opt,),
            batch,
            parallel_mode=parallel_mode,
            backend=XLAGSPMDBackend(),
            dispatchers=dispatchers,
            region_transforms=region_transforms,
        )
    else:
        spec = RecirCompileSpec(
            mode=RecirMode.GSPMD_TPU,
            device=device,
            num_ranks=num_ranks,
            num_processes=num_processes,
            loss_reduction="sum",
            region_transforms=region_transforms,
        )
        runner = compile_gspmd_step(
            model,
            dense_opt,
            batch,
            dispatchers,
            spec,
            train_step_fn=_udd_fm_v3_forward_step,
        )
    return runner, model, dense_opt


def _capture_xprof() -> None:
    """server-mode: xprof remote capture against the local profiler server. Blocks
    for FWD_PROFILE_DURATION_MS; run on a thread while the loop drives the device."""
    try:
        from xprof.convert._pywrap_profiler_plugin import (  # @manual  # pyre-ignore[21]
            trace,
        )

        # (service_addr, logdir, worker_list, include_dataset_ops, duration_ms,
        #  num_tracing_attempts, options)
        trace(
            f"localhost:{_FWD_PROFILER_PORT}",
            _FWD_PROFILE_DIR,
            "",
            False,
            _FWD_PROFILE_DURATION_MS,
            5,
            {},
        )
        logger.info(f"[FWD] xprof capture complete -> {_FWD_PROFILE_DIR}")
    except Exception as e:  # noqa: BLE001 -- profiling must never fail the run
        logger.error(f"[FWD] xprof capture failed: {e}")


def _run_once(num_ranks: int, num_processes: int, profile: bool) -> Dict[str, Any]:
    """Build + compile + warmup + time ONE config (reads the current _FWD_*
    globals). Returns a metrics dict; ``profile`` gates the trace capture."""
    model, batch, module_config, global_bs = _build_model_and_batch(
        num_ranks, num_processes
    )
    runner, model, dense_opt = _compile_forward_runner(
        model, batch, module_config, num_ranks, num_processes
    )
    flat = flatten_batch(batch)

    logger.info(f"[FWD] warmup: {_FWD_WARMUP} steps (first triggers XLA compile)...")
    t_warmup = time.perf_counter()
    for _ in range(_FWD_WARMUP):
        runner(model, dense_opt, *flat)
    logger.info(f"[FWD] warmup done in {time.perf_counter() - t_warmup:.2f}s")

    if profile and not _FWD_PROFILE_DIR.startswith("gs://"):
        os.makedirs(_FWD_PROFILE_DIR, exist_ok=True)
    capture_thread: threading.Thread | None = None
    if profile and _FWD_PROFILE_MODE == "server":
        capture_thread = threading.Thread(target=_capture_xprof, daemon=True)
        capture_thread.start()
        time.sleep(2.0)  # let the server accept the capture before stepping
        logger.info(f"[FWD] xprof capture started ({_FWD_PROFILE_DURATION_MS}ms)")
    # trace mode: capture only a short step window inside the loop (not all steps),
    # so the trace stays under the ~1M-event cap and keeps the device XLA-op rows.
    trace_mode = profile and _FWD_PROFILE_MODE != "server"
    prof_start = min(2, max(0, _FWD_STEPS - _FWD_PROFILE_STEPS))
    prof_end = min(prof_start + _FWD_PROFILE_STEPS, _FWD_STEPS)  # exclusive

    logger.info(f"[FWD] timing: {_FWD_STEPS} forward steps...")
    latencies: List[float] = []
    for i in range(_FWD_STEPS):
        if trace_mode and i == prof_start:
            jax.profiler.start_trace(_FWD_PROFILE_DIR)
            logger.info(
                f"[FWD] start_trace steps [{prof_start},{prof_end}) -> {_FWD_PROFILE_DIR}"
            )
        t0 = time.perf_counter()
        # Bind as Any: jax's stub doesn't type StepTraceAnnotation as a ctx mgr.
        annot: Any = jax.profiler.StepTraceAnnotation("fwd_step", step_num=i)
        with annot:
            runner(model, dense_opt, *flat)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if trace_mode and i == prof_end - 1:
            jax.profiler.stop_trace()
            logger.info(f"[FWD] stop_trace -> {_FWD_PROFILE_DIR} (open in xprof)")

    if capture_thread is not None:
        # keep the device busy until the capture window closes, then flush.
        while capture_thread.is_alive():
            annot_extra: Any = jax.profiler.StepTraceAnnotation("fwd_step_extra")
            with annot_extra:
                runner(model, dense_opt, *flat)
        capture_thread.join()
        logger.info(f"[FWD] trace saved to {_FWD_PROFILE_DIR}")

    latencies.sort()
    mean = statistics.mean(latencies)
    return {
        "global_bs": global_bs,
        "seq": _FWD_SEQ,
        "l2_seq": _FWD_L2_SEQ,
        "l1": _FWD_L1_LAYERS,
        "l2": _FWD_L2_LAYERS,
        "cand": _FWD_NUM_CANDIDATES,
        "emb_backend": _FWD_EMB_BACKEND,
        "mean": mean,
        "min": latencies[0],
        "std": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        "p50": latencies[len(latencies) // 2],
        "p95": latencies[int(len(latencies) * 0.95)],
        "p99": latencies[int(len(latencies) * 0.99)],
        "throughput": global_bs / (mean / 1000.0),
    }


def _print_single(r: Dict[str, Any], platform: str, num_ranks: int) -> None:
    print("\n" + "=" * 72)
    print("  RankFM (udd_fm_v3) Forward-Only Benchmark")
    print(
        f"  device={_FWD_DEVICE} platform={platform} "
        f"devices={num_ranks} emb_backend={r['emb_backend']}"
    )
    print(
        f"  global_bs={r['global_bs']} seq={r['seq']} l2_seq={r['l2_seq']} "
        f"l1_layers={r['l1']} l2_layers={r['l2']} num_candidates={r['cand']}"
    )
    print("-" * 72)
    print(
        "  Mean(ms) | Min(ms) | Std(ms) | P50(ms) | P95(ms) | P99(ms) | "
        "Throughput(samples/s)"
    )
    print(
        f"  {r['mean']:>8.2f} | {r['min']:>7.2f} | {r['std']:>7.2f} | "
        f"{r['p50']:>7.2f} | {r['p95']:>7.2f} | {r['p99']:>7.2f} | "
        f"{r['throughput']:>10.0f}"
    )
    print("=" * 72)


def _print_sweep(
    axes: List[str],
    rows: List[Tuple[Tuple[str, ...], Dict[str, Any]]],
    platform: str,
    num_ranks: int,
) -> None:
    width = 102 + max(0, len(axes) - 1) * 15
    print("\n" + "=" * width)
    print(f"  RankFM (udd_fm_v3) Forward-Only Benchmark -- SWEEP {'x'.join(axes)}")
    print(
        f"  device={_FWD_DEVICE} platform={platform} devices={num_ranks} "
        f"emb_backend={_FWD_EMB_BACKEND}"
    )
    print("-" * width)
    hdr = " | ".join(f"{a:>12}" for a in axes)
    print(
        f"  {hdr} | Mean(ms) | Min(ms) | Std(ms) | P50(ms) | P95(ms) | P99(ms) | "
        "Throughput(s/s)"
    )
    for combo, r in rows:
        cols = " | ".join(f"{v:>12}" for v in combo)
        print(
            f"  {cols} | {r['mean']:>8.2f} | {r['min']:>7.2f} | {r['std']:>7.2f} "
            f"| {r['p50']:>7.2f} | {r['p95']:>7.2f} | {r['p99']:>7.2f} "
            f"| {r['throughput']:>11.0f}"
        )
    print("=" * width)


def main() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    if _FWD_DEVICE == "tpu":
        jax.distributed.initialize()
    resolve_dist_env()
    ensure_gloo_process_group()
    force_cpu_barrier()

    devices = jax.devices()
    num_ranks = len(devices)
    num_processes = jax.process_count()  # 1 single-host; #hosts on a multi-host slice
    platform = str(devices[0].platform)
    if _FWD_EMB_BACKEND == "tpu" and _FWD_DEVICE != "tpu":
        raise RuntimeError("FWD_EMB_BACKEND=tpu requires FWD_DEVICE=tpu (SparseCore).")

    sweep = _parse_sweep()
    if sweep is None:
        profile_on = _FWD_PROFILE and platform == "tpu"
        if profile_on and _FWD_PROFILE_MODE == "server":
            jax.profiler.start_server(_FWD_PROFILER_PORT)
        _print_single(
            _run_once(num_ranks, num_processes, profile_on), platform, num_ranks
        )
        return

    # Sweep: cartesian grid over the axes; profiling disabled.
    axes = [a for a, _ in sweep]
    value_lists = [vs for _, vs in sweep]
    logger.info(f"[FWD] SWEEP {axes} grid={value_lists} (profiling disabled)")
    rows: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []
    for combo in itertools.product(*value_lists):
        for axis, val in zip(axes, combo):
            _set_axis(axis, val)
        logger.info(f"[FWD] === sweep {dict(zip(axes, combo))} ===")
        rows.append((combo, _run_once(num_ranks, num_processes, False)))
    _print_sweep(axes, rows, platform, num_ranks)


if __name__ == "__main__":
    main()
