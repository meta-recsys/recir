# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""RMSNorm on one TPU device: XLA lowering vs the Pallas kernels.

Run on a TPU host:
  python3 -m recir.ops.pallas.benchmarks.bench_rmsnorm --rows 1048576
"""

import argparse
import itertools
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
from recir.ops.pallas import rmsnorm as kernel
from recir.torchax.ops import jrmsnorm

# CPU runs the kernels interpreted, as a smoke test of this script only.
_INTERPRET: bool = jax.default_backend() == "cpu"


def _time_ms(fn: Callable[..., Any], *args: Any, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - start) * 1e3)
    return sorted(times)[len(times) // 2]


def _xla_fwd(x: Any, w: Any, silu: bool) -> Any:
    return jrmsnorm.xla_rmsnorm_fwd(x, w, 1e-6, silu)


def _xla_bwd(g: Any, x: Any, w: Any, rstd: Any, silu: bool) -> Any:
    return jrmsnorm.xla_rmsnorm_backward(g, x, w, rstd, silu)


def _kernel_fwd(x: Any, w: Any, silu: bool) -> Any:
    return kernel.rmsnorm_fwd(x, w, 1e-6, silu, interpret=_INTERPRET)


def _kernel_bwd(g: Any, x: Any, w: Any, rstd: Any, silu: bool) -> Any:
    return kernel.rmsnorm_backward(g, x, w, rstd, silu, interpret=_INTERPRET)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1 << 20)
    parser.add_argument("--dims", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    args, _ = parser.parse_known_args()

    device = jax.devices()[0]
    print(f"device={device.device_kind} rows={args.rows}", flush=True)
    print(
        "dim,dtype,weight,silu,xla_fwd_ms,kernel_fwd_ms,xla_bwd_ms,kernel_bwd_ms,"
        "kernel_fwd_GBps,kernel_bwd_GBps",
        flush=True,
    )
    key = jax.random.PRNGKey(0)
    for dim, dtype, weighted, silu in itertools.product(
        args.dims, (jnp.bfloat16, jnp.float32), (False, True), (False, True)
    ):
        kx, kg, kw = jax.random.split(key, 3)
        x = jax.device_put(jax.random.normal(kx, (args.rows, dim), dtype), device)
        g = jax.device_put(jax.random.normal(kg, (args.rows, dim), dtype), device)
        w = (
            jax.device_put(1 + 0.1 * jax.random.normal(kw, (dim,), dtype), device)
            if weighted
            else None
        )
        times = []
        for fwd, bwd in ((_xla_fwd, _xla_bwd), (_kernel_fwd, _kernel_bwd)):
            fwd_jit = jax.jit(fwd, static_argnums=(2,))
            bwd_jit = jax.jit(bwd, static_argnums=(4,))
            rstd = fwd_jit(x, w, silu)[1]
            times.append(
                (
                    _time_ms(fwd_jit, x, w, silu, warmup=args.warmup, iters=args.iters),
                    _time_ms(
                        bwd_jit,
                        g,
                        x,
                        w,
                        rstd,
                        silu,
                        warmup=args.warmup,
                        iters=args.iters,
                    ),
                )
            )
        itemsize = jnp.dtype(dtype).itemsize
        fwd_bytes = args.rows * (2 * dim * itemsize + 4)
        bwd_bytes = args.rows * (3 * dim * itemsize + 4)
        (xf, xb), (kf, kb) = times
        print(
            f"{dim},{jnp.dtype(dtype).name},{int(weighted)},{int(silu)},"
            f"{xf:.3f},{kf:.3f},{xb:.3f},{kb:.3f},"
            f"{fwd_bytes / kf / 1e6:.0f},{bwd_bytes / kb / 1e6:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
