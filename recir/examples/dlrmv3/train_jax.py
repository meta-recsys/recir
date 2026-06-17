#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DLRMv3 E2E training on GKE TPU v7x — torchax + SparseCore via shard_map, no D2H for embeddings.

Per-step flow:
  CPU:  preprocess_csr()     — raw IDs → KJT → numpy → globally-sharded PreprocessedInput
  CPU:  preprocess_payload() — payload_features via _ZeroEC (no CPU embedding lookup)
  TPU:  tpu_sparse_dense_matmul(csr, emb_vars) → sharded activations (stay on device)
  TPU:  torchax DlrmV3Dense + jax.value_and_grad → loss + (dense_grads, act_grads)
  TPU:  make_array_from_single_device_arrays — lifts sharded act_grads to global 256-device array
  TPU:  shard_map tpu_sparse_dense_matmul_grad(act_grads, csr, emb_vars) → new emb_vars

Embedding activations and dense weight updates never touch the CPU: SC forward output
(sharded JAX arrays) flows directly into the dense jit, which fuses forward, backward,
and SGD update in one compiled unit. act_grads are resharded on-device via
make_array_from_single_device_arrays (per-device jax.Array, no np.asarray).

No recir @compile, no Override, no graph break, no gloo PG.
JAX PJRT handles multi-host coordination for both SparseCore and dense (replicated).

Usage:
    python -u -m recir.examples.dlrmv3.train_jax --preset small --max_steps 10
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any

import fbgemm_gpu  # noqa: F401 -- force C++ op registration before fallback
import jax
import jax.numpy as jnp
import numpy as np
import recir.examples.dlrmv3.jfbgemm  # noqa: F401 -- registers fbgemm→JAX lowerings for torchax dispatch
import recir.lowering.fbgemm_fallback  # noqa: F401 -- registers fbgemm torch fallbacks + monkey-patches concat_2D_jagged
import recir.torchax as torchax
import torch
import torch.nn as nn
from recir.tests.xla.dlrmv3_test_utils import (
    create_dlrmv3_wrapper,
    create_static_batch,
    DlrmV3Preprocess,
    get_hstu_config,
)
from torchrec.modules.embedding_configs import EmbeddingConfig

from .config import DLRMv3Config
from .embedding_jax import (
    _lazy_jte,
    _num_sc_per_device,
    activations_to_emb_values,
    build_sparsecore_tables,
    make_global_csr,
    preprocess_csr,
    preprocess_payload,
)

logger: logging.Logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def create_model_and_data(
    cfg: DLRMv3Config, global_bs: int
) -> tuple[nn.Module, list[EmbeddingConfig], list[str], tuple[torch.Tensor, ...], Any]:
    """Create DLRMv3 model, table configs, and a synthetic static batch.

    Uses get_hstu_config() with dimensions from cfg, so medium/large presets
    produce correspondingly larger models.
    """
    table_configs = [
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="post_id",
            feature_names=[
                "uih_post_id",
                "item_post_id",
                "uih_owner_id",
                "item_owner_id",
            ],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="viewer_id",
            feature_names=["viewer_id"],
        ),
        EmbeddingConfig(
            num_embeddings=cfg.hash_size,
            embedding_dim=cfg.embedding_dim,
            name="dummy_contexual",
            feature_names=["dummy_contexual"],
        ),
    ]

    hstu_cfg = get_hstu_config(
        embedding_dim=cfg.embedding_dim,
        hidden_dim=cfg.hidden_dim,
        transducer_dim=cfg.transducer_dim,
        attn_dim=cfg.attn_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        uih_len=cfg.uih_len,
        num_candidates=cfg.num_candidates,
    )
    wrapper, hstu_config, _, emb_keys = create_dlrmv3_wrapper(
        global_bs, hstu_config=hstu_cfg, table_configs=table_configs
    )
    hstu_config.max_num_candidates = cfg.num_candidates
    hstu_config.max_seq_len = cfg.uih_len + cfg.num_candidates + 16

    rank = jax.process_index()
    batch = create_static_batch(
        hstu_config,
        cfg.local_batch_size,
        seed=42 + rank,
        uih_len_override=cfg.uih_len,
        num_candidates_override=cfg.num_candidates,
        hash_size_override=cfg.hash_size,
    )

    param_count = sum(p.numel() for p in wrapper.parameters())
    emb_count = sum(t.num_embeddings * t.embedding_dim for t in table_configs)
    logger.info(
        f"Model: {param_count:,} total params, {emb_count:,} embedding params, "
        f"dense: {param_count - emb_count:,}"
    )
    return wrapper, table_configs, emb_keys, batch, hstu_config


def build_training_fns(
    dense_mod: nn.Module,
    feature_specs: dict[str, Any],
    emb_keys: list[str],
    seq_lens: dict[str, int],
    num_sc: int,
    dense_lr: float,
) -> tuple[dict[str, Any], Any]:
    """Extract the dense model as a JAX callable and build per-step training functions.

    Returns:
      dense_states: dict of JAX arrays (dense model weights), threaded as
                    functional state through every training step.
      training_fns: 4-tuple (sc_forward, dense_step, build_sc_backward, make_global_act_grads)
                    consumed by _run_training_loop.

    NOTE: DlrmV3Dense uses fbgemm ops (asynchronous_complete_cumsum, concat_2D_jagged).
    These need torchax JAX implementations registered via recir.examples.dlrmv3.jfbgemm
    (imported at module level in train.py).
    """
    jte_embedding: Any
    jte_embedding, _, _, _ = _lazy_jte()
    _kernel_num_dev: int = jax.local_device_count()
    _fspecs: dict[str, Any] = feature_specs
    _nsc: int = num_sc
    _strat: str = "MOD"
    _emb_keys: list[str] = emb_keys
    _lr: float = dense_lr

    # Replace _embedding_collection with nn.Identity() before extract_jax so
    # dense_states does not include EC parameters (those live in emb_vars on SC).
    # DlrmV3Dense.forward() never calls _embedding_collection — it receives
    # pre-computed emb_values — so this is safe.
    orig_ec = dense_mod.model._embedding_collection  # pyre-ignore[16]
    dense_mod.model._embedding_collection = nn.Identity()  # pyre-ignore[8, 16]
    dense_states: dict[str, Any]
    dense_jax_fn: Any
    dense_states, dense_jax_fn = torchax.extract_jax(dense_mod)
    dense_mod.model._embedding_collection = orig_ec  # pyre-ignore[16]

    logger.info(f"extract_jax: {len(dense_states)} dense state tensors extracted")

    # SC forward: plain jax.jit + PJRT coordination (proven working).
    @jax.jit
    def _sc_forward(csr_fwd: Any, emb_vars: dict[str, Any]) -> dict[str, jax.Array]:
        return jte_embedding.tpu_sparse_dense_matmul(
            csr_fwd,
            emb_vars,
            _fspecs,
            global_device_count=_kernel_num_dev,
            sharding_strategy=_strat,
            num_sc_per_device=_nsc,
        )

    # SC backward via shard_map (per jax_tpu_embedding example).
    # Pre-shard ALL inputs with the right PartitionSpec, then @jax.shard_map
    # with matching in_specs — each device calls tpu_sparse_dense_matmul_grad
    # on its local shard [200, 64] act_grads / [4,...] CSR / [117216, 64] emb_vars.
    # PJRT coordinates all 32 pods so the SC hardware collective spans 256 devices.
    # global_device_count is inferred from mesh.size=256 by the kernel.
    _global_mesh_bwd: jax.sharding.Mesh = jax.sharding.Mesh(
        np.array(jax.devices()), ("dev",)
    )
    _P: type[jax.sharding.PartitionSpec] = jax.sharding.PartitionSpec
    _act_grad_sharding: jax.sharding.NamedSharding = jax.sharding.NamedSharding(
        _global_mesh_bwd, _P("dev", None)
    )
    # emb_vars out_spec: P("dev", None) — same sharding as emb_vars input
    _emb_out_spec: jax.sharding.PartitionSpec = _P("dev", None)
    # Local 8-device sharding used to normalise act_grads before global assembly.
    # After value_and_grad some features (e.g. seq_len=1) come back replicated;
    # device_put with this sharding slices each replica to its row range (D2H-free).
    _local_act_sharding: jax.sharding.NamedSharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(jax.local_devices()), ("lx",)),
        _P("lx", None),
    )

    def _build_sc_backward(csr_global: Any, csr_in_specs: Any) -> Any:
        """Build shard_map'd backward using dynamic in_specs from CSR pytree."""

        @jax.jit
        @functools.partial(
            jax.shard_map,
            mesh=_global_mesh_bwd,
            in_specs=(
                {k: _P("dev", None) for k in _emb_keys},  # act_grads per feature
                csr_in_specs,  # CSR: P("dev") for arrays, P() for scalars
                _P("dev", None),  # emb_vars
            ),
            out_specs=_emb_out_spec,
            check_vma=False,
        )
        def _sc_bwd_fn(
            act_grads_local: dict[str, jax.Array],
            csr_local: Any,
            emb_vars_local: dict[str, Any],
        ) -> dict[str, Any]:
            return jte_embedding.tpu_sparse_dense_matmul_grad(
                act_grads_local,
                csr_local,
                emb_vars_local,
                _fspecs,
                sharding_strategy=_strat,
                num_sc_per_device=_nsc,
            )

        return _sc_bwd_fn

    def _make_global_act_grads(
        act_grads_list: list[jax.Array],
    ) -> dict[str, jax.Array]:
        """Lift per-pod act_grads to globally-sharded arrays without D2H.

        After value_and_grad, some features (e.g. seq_len=1, viewer_id) have
        act_grads replicated across local devices — each device holds the full
        (local_tokens, D) rather than a 1/8 slice. We normalise with device_put
        (D2H-free on-device reshard via XLA slice) before assembling globally.

        Per-device shape after reshard: (local_tokens / local_dev, emb_dim).
        32 pods × 8 devices = 256 total shards → global (global_tokens, D).
        """
        num_procs = jax.process_count()
        result: dict[str, jax.Array] = {}
        for i, key in enumerate(_emb_keys):
            local_arr = act_grads_list[i]
            local_shape = local_arr.shape  # (local_tokens, emb_dim)
            # Normalise sharding: reshard to local P("lx", None) so every
            # addressable shard holds (local_tokens // local_dev, emb_dim).
            # jax.device_put with a Sharding uses XLA collectives — no CPU copy.
            local_arr = jax.device_put(local_arr, _local_act_sharding)
            global_shape = (local_shape[0] * num_procs, local_shape[1])
            device_arrs = [s.data for s in local_arr.addressable_shards]
            result[key] = jax.make_array_from_single_device_arrays(
                global_shape, _act_grad_sharding, device_arrs
            )
        return result

    # Dense forward + backward + SGD update (single jit, argnums 6-9 are Python ints).
    # Fusing the SGD update here means dense_grads never surface to Python:
    # XLA can overlap weight update with gradient computation in one compiled unit.
    @functools.partial(jax.jit, static_argnums=(6, 7, 8, 9))
    def _dense_step(
        emb_values_jax: list[jax.Array],
        emb_lengths_jax: list[jax.Array],
        dense_states: dict[str, Any],
        payload_jax: dict[str, jax.Array],
        uih_seq_lengths: jax.Array,
        num_candidates: jax.Array,
        max_uih_len: int,
        max_num_candidates: int,
        total_targets: int,
        total_uih_len: int,
    ) -> tuple[jax.Array, tuple[dict[str, Any], list[jax.Array]]]:
        def compute_loss(ds: dict[str, Any], ev: list[jax.Array]) -> jax.Array:
            aux: dict[str, jax.Array] = dense_jax_fn(
                ds,
                args=(
                    emb_lengths_jax,
                    ev,
                    payload_jax,
                    max_uih_len,
                    uih_seq_lengths,
                    max_num_candidates,
                    num_candidates,
                    total_targets,
                    total_uih_len,
                ),
            )
            return jnp.sum(jnp.stack([jnp.sum(v) for v in aux.values()]))

        loss, (dense_grads, act_grads_list) = jax.value_and_grad(
            compute_loss, argnums=(0, 1)
        )(dense_states, emb_values_jax)
        new_dense_states = jax.tree.map(
            lambda s, g: s - _lr * g, dense_states, dense_grads
        )
        return loss, (new_dense_states, act_grads_list)

    # Return all the pieces needed by _run_training_loop.
    # training_fns are wired in _run_training_loop to allow lazy shard_map build.
    return dense_states, (
        _sc_forward,
        _dense_step,
        _build_sc_backward,
        _make_global_act_grads,
    )


def _to_jax(t: torch.Tensor) -> jax.Array:
    return jnp.array(t.detach().cpu().numpy())


def _payload_to_jax(
    payload_features: dict[str, torch.Tensor],
) -> dict[str, jax.Array]:
    return {k: _to_jax(v) for k, v in payload_features.items()}


def _run_training_loop(
    cfg: DLRMv3Config,
    wrapper: nn.Module,
    table_configs: list[EmbeddingConfig],
    emb_vars: dict[str, Any],
    feature_specs: dict[str, Any],
    dense_states: dict[str, Any],
    training_fns: Any,
    local_batch: tuple[torch.Tensor, ...],
    hstu_config: Any,
    emb_keys: list[str],
    seq_lens: dict[str, int],
    local_bs: int,
    global_bs: int,
    is_main: bool,
) -> None:
    uih_values, uih_lengths, cand_values, cand_lengths = local_batch

    # KJT key lists (for CSR preprocessing)
    contextual = list(hstu_config.contextual_feature_to_max_length.keys())
    uih_seq_feats = [
        k for k in hstu_config.hstu_uih_feature_names if k not in contextual
    ]
    uih_kjt_keys: list[str] = contextual + uih_seq_feats
    cand_kjt_keys: list[str] = hstu_config.hstu_candidate_feature_names

    local_dev = jax.local_device_count()
    global_dev = jax.device_count()
    num_sc = _num_sc_per_device()

    # Profiling setup
    profiling_active = False
    jax_profiling = False
    try:
        is_tpu = jax.devices()[0].platform == "tpu"
        jax_profiling = is_tpu and cfg.profile_start_step > 0
    except (RuntimeError, IndexError):
        logger.warning("Could not detect TPU platform; profiling disabled")

    sc_forward_fn, dense_step_fn, build_sc_backward_fn, make_global_act_grads_fn = (
        training_fns
    )
    sc_backward_fn: Any = None  # built lazily on first step from csr_in_specs

    # emb_lengths is loop-invariant (seq_lens and local_bs never change).
    emb_lengths_jax: list[jax.Array] = [
        jnp.full((local_bs,), seq_lens.get(k, 1), dtype=jnp.int32) for k in emb_keys
    ]

    logger.info(f"Starting training: {cfg.max_steps} steps, global_bs={global_bs}")
    start_time = time.time()

    for step in range(1, cfg.max_steps + 1):
        step_start = time.time()

        # --- CPU: CSR preprocessing ---
        csr_fwd = preprocess_csr(
            uih_values,
            uih_lengths,
            cand_values,
            cand_lengths,
            uih_kjt_keys=uih_kjt_keys,
            cand_kjt_keys=cand_kjt_keys,
            feature_specs=feature_specs,
            seq_lens=seq_lens,
            local_dev=local_dev,
            global_dev=global_dev,
            num_sc=num_sc,
        )
        # Globally shard CSR: each process puts its [32,...] on 8 local devices.
        # Globally: 32 processes × [32,...] = [1024,...] across 256 devices = [4,...]/dev ✓
        csr_global, csr_in_specs = make_global_csr(csr_fwd)

        # Build shard_map'd backward once (CSR in_specs same every step).
        if sc_backward_fn is None:
            sc_backward_fn = build_sc_backward_fn(csr_global, csr_in_specs)
            logger.info("[INIT] SC backward shard_map built successfully")

        # --- CPU: payload preprocessing (no embedding lookup) ---
        preprocess_mod: DlrmV3Preprocess = wrapper.preprocess_mod  # pyre-ignore[16, 9]
        (
            _,
            _,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
            total_targets,
            total_uih_len,
        ) = preprocess_payload(
            preprocess_mod,
            table_configs,
            seq_lens,
            local_bs,
            uih_values,
            uih_lengths,
            cand_values,
            cand_lengths,
        )

        payload_jax = _payload_to_jax(payload_features)
        uih_seq_lengths_jax = _to_jax(uih_seq_lengths)
        num_candidates_jax = _to_jax(num_candidates)

        # --- TPU/SparseCore train step ---
        if jax_profiling and step == cfg.profile_start_step:
            if not cfg.profile_dir.startswith("gs://"):
                os.makedirs(cfg.profile_dir, exist_ok=True)
            jax.profiler.start_trace(cfg.profile_dir)
            profiling_active = True
            logger.info(f"[PROFILE] Started trace at step {step}")

        # 1. SC forward (plain jit + PJRT). XLA data dependencies ensure dense
        # step waits for activations; no explicit host barrier needed.
        activations = sc_forward_fn(csr_fwd, emb_vars)
        emb_values_jax = activations_to_emb_values(activations, emb_keys)

        # 2. Dense forward + backward + SGD update (fused in one jit).
        # Returns updated dense_states directly; dense_grads stay on device.
        loss, (dense_states, act_grads_list) = dense_step_fn(
            emb_values_jax,
            emb_lengths_jax,
            dense_states,
            payload_jax,
            uih_seq_lengths_jax,
            num_candidates_jax,
            max_uih_len,
            max_num_candidates,
            total_targets,
            total_uih_len,
        )

        # 3. SC backward via shard_map — inputs pre-sharded per user's pattern.
        # act_grads globally sharded: 32 pods × [1600,64] = [51200,64].
        # Per device: [200,64]; 51200/(256×4 SC)=50 rows/SC ← INTEGER ✓.
        act_grads_global = make_global_act_grads_fn(act_grads_list)
        emb_vars = sc_backward_fn(act_grads_global, csr_global, emb_vars)

        # Only rank 0 reads the loss scalar (D2H); other 31 ranks skip it.
        # block_until_ready on emb_vars ensures the SC backward (last async op)
        # completes before we measure step_time.
        jax.block_until_ready(emb_vars)

        if jax_profiling and step == cfg.profile_end_step and profiling_active:
            jax.profiler.stop_trace()
            profiling_active = False
            logger.info(f"[PROFILE] Stopped trace at step {step}")

        step_time = time.time() - step_start
        if is_main:
            loss_val = float(loss)
            logger.info(
                f"Step {step}/{cfg.max_steps} | loss={loss_val:.4f} | "
                f"time={step_time:.2f}s"
            )

        if jax_profiling and cfg.enable_memory_log and is_main:
            for i, dev in enumerate(jax.local_devices()):
                stats = dev.memory_stats()
                if stats:
                    used = stats.get("bytes_in_use", 0)
                    limit = stats.get("bytes_limit", 0)
                    logger.info(
                        f"  Device {i} HBM: {used / 1e9:.2f}/{limit / 1e9:.2f} GB "
                        f"({100 * used / max(limit, 1):.1f}%)"
                    )

    if profiling_active and jax_profiling:
        jax.profiler.stop_trace()

    total_time = time.time() - start_time
    if is_main:
        logger.info(
            f"Training complete: {cfg.max_steps} steps in {total_time:.1f}s "
            f"({total_time / cfg.max_steps:.2f}s/step)"
        )


def main() -> None:
    # Must be first: PJRT reads coordinator address from TPU runtime metadata.
    # Without this, each pod sees jax.process_count()=1 and the SparseCore
    # hardware collective deadlocks.
    jax.distributed.initialize()
    _setup_logging()

    rank = jax.process_index()
    world = jax.process_count()
    logger.info(
        f"[DIST] rank={rank}/{world}, "
        f"device_count={jax.device_count()}, "
        f"local_device_count={jax.local_device_count()}"
    )

    cfg = DLRMv3Config.from_args()
    logger.info(f"Config: {cfg}")

    is_main = rank == 0
    global_bs = cfg.local_batch_size * world
    local_bs = cfg.local_batch_size

    # Sequence lengths for SparseCore FeatureSpec shapes and token-as-sample trick
    seq_lens: dict[str, int] = {
        "uih_post_id": cfg.uih_len,
        "uih_owner_id": cfg.uih_len,
        "item_post_id": cfg.num_candidates,
        "item_owner_id": cfg.num_candidates,
        # viewer_id, dummy_contexual are pooled (seq_len=1, default)
    }

    torch.manual_seed(0)
    wrapper, table_configs, emb_keys, local_batch, hstu_config = create_model_and_data(
        cfg, global_bs
    )

    # Initialize SparseCore embedding tables
    logger.info("[INIT] build_sparsecore_tables ...")
    emb_vars, feature_specs = build_sparsecore_tables(
        table_configs=table_configs,
        seq_lens=seq_lens,
        global_batch_size=global_bs,
        world_size=world,
        embedding_lr=cfg.embedding_lr,
    )
    logger.info(
        "[INIT] build_sparsecore_tables done, stacks=%s emb_var_shards=%s",
        list(emb_vars.keys()),
        {k: v.table.sharding for k, v in emb_vars.items()},
    )

    # Extract DlrmV3Dense as a JAX callable and build per-step training functions.
    logger.info("[INIT] build_training_fns ...")
    dense_mod: nn.Module = wrapper.dense_mod  # pyre-ignore[16, 9]
    dense_states, training_fns = build_training_fns(
        dense_mod=dense_mod,
        feature_specs=feature_specs,
        emb_keys=emb_keys,
        seq_lens=seq_lens,
        num_sc=_num_sc_per_device(),
        dense_lr=cfg.dense_lr,
    )

    logger.info(
        "[INIT] build_training_fns done, dense_states keys=%d", len(dense_states)
    )
    _run_training_loop(
        cfg=cfg,
        wrapper=wrapper,
        table_configs=table_configs,
        emb_vars=emb_vars,
        feature_specs=feature_specs,
        dense_states=dense_states,
        training_fns=training_fns,
        local_batch=local_batch,
        hstu_config=hstu_config,
        emb_keys=emb_keys,
        seq_lens=seq_lens,
        local_bs=local_bs,
        global_bs=global_bs,
        is_main=is_main,
    )


if __name__ == "__main__":
    main()
