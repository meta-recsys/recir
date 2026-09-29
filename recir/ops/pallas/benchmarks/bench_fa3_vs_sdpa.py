# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


# Benchmark: Pallas FA3 vs splash_attention vs JAX SDPA
# Includes forward and backward pass benchmarks.
#
# Run on TPU:
#   python bench_fa3_vs_sdpa.py

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
import time

# Ensure the repo root is importable when running standalone on a TPU VM.
_FBCODE_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")
)
if _FBCODE_ROOT not in sys.path:
    sys.path.insert(0, _FBCODE_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu import splash_attention
from recir.ops.pallas.flash_attention import pallas_flash_attention_batched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flush(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def bench_fn(fn, warmup: int = 5, iters: int = 30) -> float:
    """Benchmark on TPU. Returns median time in ms."""
    for i in range(warmup):
        result = fn()
        result.block_until_ready()
        if i == 0:
            _flush("    [warmup 1 done]")

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn()
        result.block_until_ready()
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    return times[len(times) // 2]


def build_target_attention_mask_np(
    prefix_len: int,
    num_items: int,
    item_len: int,
) -> np.ndarray:
    """Build a boolean target attention mask [seq_len, seq_len] as numpy."""
    total_len = prefix_len + num_items * item_len
    mask = np.zeros((total_len, total_len), dtype=np.bool_)

    for i in range(prefix_len):
        mask[i, : i + 1] = True

    for item_idx in range(num_items):
        item_start = prefix_len + item_idx * item_len
        for i in range(item_start, item_start + item_len):
            mask[i, :prefix_len] = True
            mask[i, item_start : i + 1] = True

    return mask


def repeat_kv_jax(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Expand KV heads to match query heads for GQA."""
    if n_rep == 1:
        return hidden_states
    B, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = jnp.repeat(hidden_states[:, :, None, :, :], n_rep, axis=2)
    return hidden_states.reshape(B, num_kv_heads * n_rep, seq_len, head_dim)


# ---------------------------------------------------------------------------
# Forward pass benchmark
# ---------------------------------------------------------------------------


def bench_forward() -> None:
    """Benchmark forward pass: Pallas FA3 vs Splash Attention vs JAX SDPA."""
    try:
        tpu_devices = jax.devices("tpu")
    except RuntimeError:
        tpu_devices = []
    if not tpu_devices:
        print("ERROR: TPU is required to run benchmarks.")
        sys.exit(1)

    _flush(f"TPU: {tpu_devices[0]}")
    _flush(f"JAX: {jax.__version__}")
    _flush()

    # Qwen3-0.6B model config
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    n_rep = num_heads // num_kv_heads
    dtype = jnp.bfloat16
    num_items = 10

    _flush("=" * 90)
    _flush("FORWARD PASS: Pallas FA3 vs Splash Attention vs JAX SDPA")
    _flush(
        f"  Qwen3-0.6B: num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}"
    )
    _flush(f"  num_items={num_items}, dtype=bf16")
    _flush("=" * 90)

    batch_sizes = [1, 16, 32]
    prefix_lens = [256, 512, 1024, 2048, 4096]
    item_lens = [32, 64]

    header = (
        f"{'batch':>5} | {'prefix':>6} | {'ilen':>4} | "
        f"{'tokens':>8} | {'pallas_ms':>9} | {'splash_ms':>9} | "
        f"{'sdpa_ms':>9} | {'p/splash':>8}"
    )
    _flush(header)
    _flush("-" * len(header))

    for batch_size in batch_sizes:
        for prefix_len in prefix_lens:
            for item_len in item_lens:
                seq_len = prefix_len + num_items * item_len
                total_tokens = batch_size * seq_len

                key = jax.random.PRNGKey(42)
                k1, k2, k3 = jax.random.split(key, 3)
                scale = 1.0 / math.sqrt(head_dim)

                # ---- 1. Pallas FA3 (batched, JIT) ----
                q_batched = jax.random.normal(
                    k1, (batch_size, seq_len, num_heads, head_dim), dtype=dtype
                )
                k_batched = jax.random.normal(
                    k2, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype
                )
                v_batched = jax.random.normal(
                    k3, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype
                )

                fa3_jit = jax.jit(
                    functools.partial(
                        pallas_flash_attention_batched,
                        prefix_len=prefix_len,
                        tokens_per_item=item_len,
                        sm_scale=scale,
                    )
                )

                def run_fa3(q=q_batched, k=k_batched, v=v_batched, _fn=fa3_jit):
                    return _fn(q, k, v)

                _flush(
                    f"  [{batch_size}x{prefix_len}+{num_items}x{item_len}] "
                    f"Pallas FA3 ({total_tokens} tok)..."
                )
                fa3_ms = bench_fn(run_fa3)

                # ---- 2. Splash Attention (Google's optimized Pallas) ----
                # Splash requires seq_len % 128 == 0; pad mask and inputs.
                block = 128
                splash_seq = seq_len + (-seq_len % block)  # round up
                mask_2d = build_target_attention_mask_np(
                    prefix_len, num_items, item_len
                )
                if splash_seq > seq_len:
                    pad_w = splash_seq - seq_len
                    mask_2d_padded = np.pad(
                        mask_2d, ((0, pad_w), (0, pad_w)), constant_values=False
                    )
                else:
                    mask_2d_padded = mask_2d
                # Same mask for all heads
                mask_3d = np.broadcast_to(
                    mask_2d_padded[None, :, :],
                    (num_heads, splash_seq, splash_seq),
                ).copy()

                splash_kernel = splash_attention.make_splash_mha_single_device(
                    mask=mask_3d,
                    head_shards=1,
                    q_seq_shards=1,
                )

                # splash expects [num_heads, seq_len, head_dim]
                q_splash = jax.random.normal(
                    k1, (num_heads, splash_seq, head_dim), dtype=dtype
                )
                k_splash = jax.random.normal(
                    k2, (num_kv_heads, splash_seq, head_dim), dtype=dtype
                )
                v_splash = jax.random.normal(
                    k3, (num_kv_heads, splash_seq, head_dim), dtype=dtype
                )

                # vmap over batch for splash
                @jax.jit
                def splash_fn(q, k, v):
                    return jax.vmap(splash_kernel)(q, k, v)

                # Expand to [B, H, S, D] and [B, Hkv, S, D]
                q_splash_b = jnp.broadcast_to(
                    q_splash[None],
                    (batch_size, num_heads, splash_seq, head_dim),
                ).copy()
                k_splash_b = jnp.broadcast_to(
                    k_splash[None],
                    (batch_size, num_kv_heads, splash_seq, head_dim),
                ).copy()
                v_splash_b = jnp.broadcast_to(
                    v_splash[None],
                    (batch_size, num_kv_heads, splash_seq, head_dim),
                ).copy()

                def run_splash(q=q_splash_b, k=k_splash_b, v=v_splash_b, _fn=splash_fn):
                    return _fn(q, k, v)

                _flush("    Splash Attention...")
                try:
                    splash_ms = bench_fn(run_splash)
                except Exception as e:
                    _flush(f"    Splash FAILED: {e}")
                    splash_ms = float("inf")

                # ---- 3. JAX SDPA (4D with explicit mask, JIT) ----
                query_4d = jax.random.normal(
                    k1, (batch_size, num_heads, seq_len, head_dim), dtype=dtype
                )
                kv_4d = jax.random.normal(
                    k2, (batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype
                )
                value_4d = jax.random.normal(
                    k3, (batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype
                )
                kv_expanded = repeat_kv_jax(kv_4d, n_rep)
                value_expanded = repeat_kv_jax(value_4d, n_rep)
                mask_jnp = jnp.array(mask_2d[None, None, :, :])

                @jax.jit
                def sdpa_jit(q, k, v, m):
                    q_t = jnp.transpose(q, (0, 2, 1, 3))
                    k_t = jnp.transpose(k, (0, 2, 1, 3))
                    v_t = jnp.transpose(v, (0, 2, 1, 3))
                    out = jax.nn.dot_product_attention(q_t, k_t, v_t, mask=m)
                    return jnp.transpose(out, (0, 2, 1, 3))

                def run_sdpa(
                    query_4d=query_4d,
                    kv_expanded=kv_expanded,
                    value_expanded=value_expanded,
                    mask=mask_jnp,
                    _fn=sdpa_jit,
                ):
                    return _fn(query_4d, kv_expanded, value_expanded, mask)

                _flush("    JAX SDPA...")
                try:
                    sdpa_ms = bench_fn(run_sdpa)
                except Exception as e:
                    _flush(f"    SDPA FAILED: {e}")
                    sdpa_ms = float("inf")

                ratio = fa3_ms / splash_ms if splash_ms != float("inf") else 0

                _flush(
                    f"{batch_size:>5} | {prefix_len:>6} | {item_len:>4} | "
                    f"{total_tokens:>8} | {fa3_ms:>9.3f} | "
                    f"{'OOM' if splash_ms == float('inf') else f'{splash_ms:>9.3f}'} | "
                    f"{'OOM' if sdpa_ms == float('inf') else f'{sdpa_ms:>9.3f}'} | "
                    f"{ratio:>7.2f}x"
                )

    _flush()


# ---------------------------------------------------------------------------
# Backward pass benchmark
# ---------------------------------------------------------------------------


def bench_backward() -> None:
    """Benchmark backward pass: Pallas FA3 vs JAX SDPA."""
    try:
        tpu_devices = jax.devices("tpu")
    except RuntimeError:
        tpu_devices = []
    if not tpu_devices:
        print("ERROR: TPU is required to run benchmarks.")
        sys.exit(1)

    # Qwen3-0.6B model config
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    n_rep = num_heads // num_kv_heads
    dtype = jnp.bfloat16
    num_items = 10

    _flush()
    _flush("=" * 90)
    _flush("BACKWARD PASS: Pallas FA3 vs JAX SDPA")
    _flush(
        f"  Qwen3-0.6B: num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}"
    )
    _flush(f"  num_items={num_items}, dtype=bf16")
    _flush("=" * 90)

    batch_sizes = [1, 16, 32]
    prefix_lens = [256, 512, 1024, 2048, 4096]
    item_lens = [32, 64]

    header = (
        f"{'batch':>5} | {'prefix':>6} | {'ilen':>4} | "
        f"{'tokens':>8} | {'pallas_ms':>9} | "
        f"{'sdpa_ms':>9} | {'speedup':>8}"
    )
    _flush(header)
    _flush("-" * len(header))

    for batch_size in batch_sizes:
        for prefix_len in prefix_lens:
            for item_len in item_lens:
                seq_len = prefix_len + num_items * item_len
                total_tokens = batch_size * seq_len
                scale = 1.0 / math.sqrt(head_dim)

                key = jax.random.PRNGKey(42)
                k1, k2, k3 = jax.random.split(key, 3)

                # ---- 1. Pallas FA3 backward ----
                q_batched = jax.random.normal(
                    k1, (batch_size, seq_len, num_heads, head_dim), dtype=dtype
                )
                k_batched = jax.random.normal(
                    k2, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype
                )
                v_batched = jax.random.normal(
                    k3, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype
                )

                @jax.jit
                def fa3_grad_fn(q, k, v):
                    def loss(q, k, v):
                        out = pallas_flash_attention_batched(
                            q,
                            k,
                            v,
                            prefix_len=prefix_len,
                            tokens_per_item=item_len,
                            sm_scale=scale,
                        )
                        return jnp.sum(out)

                    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
                    return dq

                def run_fa3_bwd(q=q_batched, k=k_batched, v=v_batched, _fn=fa3_grad_fn):
                    return _fn(q, k, v)

                _flush(
                    f"  [{batch_size}x{prefix_len}+{num_items}x{item_len}] "
                    f"Pallas FA3 bwd ({total_tokens} tok)..."
                )
                try:
                    fa3_bwd_ms = bench_fn(run_fa3_bwd)
                except Exception as e:
                    _flush(f"    Pallas FA3 bwd FAILED: {e}")
                    fa3_bwd_ms = float("inf")

                # ---- 2. JAX SDPA backward ----
                mask_bool = jnp.array(
                    build_target_attention_mask_np(prefix_len, num_items, item_len)
                )[None, None, :, :]  # [1, 1, S, S]

                query_4d = jax.random.normal(
                    k1, (batch_size, num_heads, seq_len, head_dim), dtype=dtype
                )
                kv_4d = jax.random.normal(
                    k2, (batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype
                )
                value_4d = jax.random.normal(
                    k3, (batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype
                )
                kv_expanded = repeat_kv_jax(kv_4d, n_rep)
                value_expanded = repeat_kv_jax(value_4d, n_rep)

                @jax.jit
                def sdpa_grad_fn(q, k, v, m):
                    def loss(q, k, v):
                        q_t = jnp.transpose(q, (0, 2, 1, 3))
                        k_t = jnp.transpose(k, (0, 2, 1, 3))
                        v_t = jnp.transpose(v, (0, 2, 1, 3))
                        out = jax.nn.dot_product_attention(q_t, k_t, v_t, mask=m)
                        return jnp.sum(out)

                    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
                    return dq

                def run_sdpa_bwd(
                    q=query_4d,
                    k=kv_expanded,
                    v=value_expanded,
                    m=mask_bool,
                    _fn=sdpa_grad_fn,
                ):
                    return _fn(q, k, v, m)

                _flush("    JAX SDPA bwd...")
                try:
                    sdpa_bwd_ms = bench_fn(run_sdpa_bwd)
                except Exception as e:
                    _flush(f"    SDPA bwd FAILED: {e}")
                    sdpa_bwd_ms = float("inf")

                speedup = sdpa_bwd_ms / fa3_bwd_ms if fa3_bwd_ms != float("inf") else 0

                _flush(
                    f"{batch_size:>5} | {prefix_len:>6} | {item_len:>4} | "
                    f"{total_tokens:>8} | "
                    f"{'OOM' if fa3_bwd_ms == float('inf') else f'{fa3_bwd_ms:>9.3f}'} | "
                    f"{'OOM' if sdpa_bwd_ms == float('inf') else f'{sdpa_bwd_ms:>9.3f}'} | "
                    f"{speedup:>7.2f}x"
                )

    _flush()


def bench_backward_block_sweep() -> None:
    """Sweep backward block sizes on a representative config."""
    try:
        tpu_devices = jax.devices("tpu")
    except RuntimeError:
        tpu_devices = []
    if not tpu_devices:
        print("ERROR: TPU is required to run benchmarks.")
        sys.exit(1)

    # Qwen3-0.6B model config
    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    dtype = jnp.bfloat16
    num_items = 10

    # Representative config: B=16, prefix=1024, item_len=64 (26624 tokens)
    batch_size = 16
    prefix_len = 1024
    item_len = 64
    seq_len = prefix_len + num_items * item_len
    total_tokens = batch_size * seq_len
    scale = 1.0 / math.sqrt(head_dim)

    _flush()
    _flush("=" * 90)
    _flush("BACKWARD BLOCK SIZE SWEEP")
    _flush(
        f"  Config: B={batch_size}, prefix={prefix_len}, item_len={item_len}, "
        f"tokens={total_tokens}"
    )
    _flush(
        f"  Qwen3-0.6B: num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}"
    )
    _flush("=" * 90)

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    q_batched = jax.random.normal(
        k1, (batch_size, seq_len, num_heads, head_dim), dtype=dtype
    )
    k_batched = jax.random.normal(
        k2, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype
    )
    v_batched = jax.random.normal(
        k3, (batch_size, seq_len, num_kv_heads, head_dim), dtype=dtype
    )

    block_sizes = [64, 128, 256]

    header = f"{'bq_bwd':>6} | {'bk_bwd':>6} | {'pallas_ms':>9} | {'vs_default':>10}"
    _flush(header)
    _flush("-" * len(header))

    baseline_ms = None

    for bq in block_sizes:
        for bk in block_sizes:

            @jax.jit
            def fa3_grad_fn(q, k, v, _bq=bq, _bk=bk):
                def loss(q, k, v):
                    out = pallas_flash_attention_batched(
                        q,
                        k,
                        v,
                        prefix_len=prefix_len,
                        tokens_per_item=item_len,
                        sm_scale=scale,
                        block_q_bwd=_bq,
                        block_k_bwd=_bk,
                    )
                    return jnp.sum(out)

                dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
                return dq

            def run_fn(q=q_batched, k=k_batched, v=v_batched, _fn=fa3_grad_fn):
                return _fn(q, k, v)

            _flush(f"  block_q_bwd={bq}, block_k_bwd={bk} ...")
            try:
                ms = bench_fn(run_fn)
                if baseline_ms is None:
                    baseline_ms = ms
                ratio = f"{baseline_ms / ms:.2f}x"
            except Exception as e:
                _flush(f"    FAILED: {e}")
                ms = float("inf")
                ratio = "FAIL"

            _flush(
                f"{bq:>6} | {bk:>6} | "
                f"{'OOM' if ms == float('inf') else f'{ms:>9.3f}'} | "
                f"{ratio:>10}"
            )

    _flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Pallas FA3 vs splash_attention vs JAX SDPA"
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Run only the forward pass benchmark",
    )
    parser.add_argument(
        "--backward-only",
        action="store_true",
        help="Run only the backward pass benchmark",
    )
    parser.add_argument(
        "--sweep-block-sizes",
        action="store_true",
        help="Sweep backward block sizes (block_q_bwd, block_k_bwd) on a representative config",
    )
    args = parser.parse_args()

    if args.sweep_block_sizes:
        bench_backward_block_sweep()
    else:
        run_fwd = not args.backward_only
        run_bwd = not args.forward_only

        if run_fwd:
            bench_forward()
        if run_bwd:
            bench_backward()
