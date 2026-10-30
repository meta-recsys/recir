# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from hammer.ops.searchsorted import searchsorted_partitioned
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from recir.torchax import interop
from recir.torchax.gspmd_context import gspmd_mesh_context
from recir.torchax.ops.jaten import (
    _jsearchsorted_partitioned,
    _searchsorted as _generic_searchsorted,
    _searchsorted_local_bins,
    _searchsorted_partitioned,
)
from torch.fx.experimental.proxy_tensor import make_fx


class _PartitionedSearch(torch.nn.Module):
    def __init__(self, right: bool, out_int32: bool) -> None:
        super().__init__()
        self.right = right
        self.out_int32 = out_int32

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return searchsorted_partitioned(
            a, b, right=self.right, out_int32=self.out_int32
        )


class LocalROSearchsortedTests(unittest.TestCase):
    def setUp(self) -> None:
        mesh = Mesh(np.asarray(jax.devices()), ("batch",))
        self.enterContext(gspmd_mesh_context(mesh, "batch", 8, False))

    def test_explicit_op_eager_trace_and_torchax_dispatch(self) -> None:
        import recir.torchax

        environment = recir.torchax.default_env()
        import recir.fb.examples.fm_v3_common.xsurface_helpers  # noqa: F401

        self.assertIs(
            environment._get_op_or_decomp(
                torch.ops.hammer.searchsorted_partitioned.default
            ).func,
            _jsearchsorted_partitioned,
        )
        bins = torch.tensor([0, 2, 2, 5], dtype=torch.int64)
        queries = torch.tensor([-1, 0, 2, 3, 5, 6], dtype=torch.int64)
        for right in (False, True):
            for out_int32 in (False, True):
                search = _PartitionedSearch(right, out_int32)
                expected = torch.searchsorted(
                    bins, queries, right=right, out_int32=out_int32
                )
                torch.testing.assert_close(search(bins, queries), expected)
                graph = make_fx(search, tracing_mode="fake")(bins, queries)
                targets = [
                    n.target for n in graph.graph.nodes if n.op == "call_function"
                ]
                self.assertEqual(
                    targets, [torch.ops.hammer.searchsorted_partitioned.default]
                )
                with jax.enable_x64(True):
                    actual = jax.jit(
                        lambda a, b, graph=graph: interop.call_torch(graph, a, b)
                    )(jnp.asarray(bins.numpy()), jnp.asarray(queries.numpy()))
                    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
                    self.assertEqual(str(actual.dtype), str(expected.numpy().dtype))

    def test_context_mesh_and_hybrid_batch_resource(self) -> None:
        devices = np.asarray(jax.devices())[::2][::-1].reshape(2, 2)
        mesh = Mesh(devices, ("data", "fsdp"))
        resource = ("data", "fsdp")
        seq = np.arange(20000, dtype=np.int32)
        queries = np.concatenate([np.arange(16) + chip * 5000 for chip in range(4)])
        with (
            gspmd_mesh_context(mesh, resource, 2, True),
            patch("jax.devices", side_effect=AssertionError("raw devices")),
            patch("jax.device_count", side_effect=AssertionError("raw count")),
            patch.dict(os.environ, {"RECIR_LOCAL_RO_SEARCHSORTED": "0"}),
        ):
            sharding = NamedSharding(mesh, PartitionSpec(resource))
            actual = jax.jit(
                lambda a, b: _jsearchsorted_partitioned(
                    a, b, right=True, out_int32=True
                ),
                out_shardings=sharding,
            )(jnp.asarray(seq), jnp.asarray(queries))
        np.testing.assert_array_equal(
            np.asarray(actual), np.searchsorted(seq, queries, side="right")
        )
        self.assertEqual(actual.sharding, sharding)

    def test_generic_search_does_not_use_partitioned_lowering(self) -> None:
        bins, queries = jnp.arange(20000), jnp.arange(32)
        with (
            patch.dict(os.environ, {"RECIR_LOCAL_RO_SEARCHSORTED": "1"}),
            patch("jax.shard_map", side_effect=AssertionError("unexpected shard_map")),
        ):
            result = jax.jit(lambda: _generic_searchsorted(bins, queries, "right"))()
        np.testing.assert_array_equal(np.asarray(result), np.arange(32) + 1)

    def test_owned_queries_sentinels_duplicate_boundaries_and_fallback(self) -> None:
        self.assertEqual(jax.device_count(), 8)
        n = 128 * 257
        per_chip = n // 8
        g = np.random.default_rng(19)
        counts = g.integers(0, 16, size=n, dtype=np.dtype(np.int64))
        for c in range(1, 8):
            counts[c * per_chip - 4 : c * per_chip + 4] = 0
        for dtype, shift in [
            (np.int64, 0),
            (np.int64, 2**40),
            (np.uint64, 2**40),
            (np.float64, 0),
        ]:
            seq = (np.cumsum(counts) + shift).astype(dtype)
            qs = []
            for c in range(8):
                lo = seq[max(c * per_chip - 1, 0)]
                hi = seq[(c + 1) * per_chip - 1]
                interior = np.linspace(float(lo), float(hi), 128).astype(dtype)
                interior[:4] = [0, seq[-1], lo, hi]
                qs.append(interior)
            for outlier in (False, True):
                v = np.concatenate(qs)
                v[4] = seq[0] - 1 if dtype != np.uint64 else 0
                v[5] = seq[-1] + 100
                v[-1] = 1 if shift == 0 else seq[0] - 1
                if outlier:
                    v[6] = seq[-256]  # nonowned query forces safe global fallback
                for side in ("left", "right"):
                    with (
                        jax.enable_x64(True),
                        patch.dict(
                            os.environ,
                            {
                                "RECIR_LOCAL_RO_SEARCHSORTED": "1",
                                "RECIR_PARTITIONED_SEARCHSORTED_IMPL": "binary",
                                "RECIR_SEARCHSORTED_METHOD": "scan_unrolled",
                            },
                        ),
                    ):
                        mesh = Mesh(np.asarray(jax.devices()), ("chips",))
                        sh = NamedSharding(mesh, PartitionSpec("chips"))
                        result = jax.jit(
                            lambda a, b, side=side: _searchsorted_local_bins(
                                a, b, side
                            ),
                            out_shardings=sh,
                        )(jnp.asarray(seq), jnp.asarray(v))
                        np.testing.assert_array_equal(
                            np.asarray(result),
                            np.searchsorted(seq, v, side=side).astype(np.int32),
                        )

    def test_empty_bins_and_nan_fallback(self) -> None:
        n = 128 * 257
        for nan in (False, True):
            seq = np.zeros(n, dtype=np.float64)
            if nan:
                seq[-1] = np.nan
            v = np.zeros(8 * 32, dtype=np.float64)
            if nan:
                v[-1] = np.nan
            for side in ("left", "right"):
                with (
                    jax.enable_x64(True),
                    patch.dict(
                        os.environ,
                        {
                            "RECIR_LOCAL_RO_SEARCHSORTED": "1",
                        },
                    ),
                ):
                    result = jax.jit(
                        lambda a, b, side=side: _searchsorted_partitioned(a, b, side)
                    )(jnp.asarray(seq), jnp.asarray(v))
                    np.testing.assert_array_equal(
                        np.asarray(result),
                        np.searchsorted(seq, v, side=side).astype(np.int32),
                    )

    def test_uneven_query_shards_preserve_original(self) -> None:
        a = jnp.arange(20000)
        b = jnp.array([-1, 0, 1, 12, 19999, 20000])
        for enabled in ("0", "1"):
            with patch.dict(
                os.environ,
                {
                    "RECIR_LOCAL_RO_SEARCHSORTED": enabled,
                },
            ):
                for side in ("left", "right"):
                    self.assertIsNone(_searchsorted_local_bins(a, b, side))
                    np.testing.assert_array_equal(
                        np.asarray(
                            jax.jit(
                                lambda a=a, b=b, side=side: _searchsorted_partitioned(
                                    a, b, side
                                )
                            )()
                        ),
                        np.searchsorted(np.arange(20000), np.asarray(b), side=side),
                    )

    def test_uneven_bin_shards_preserve_original(self) -> None:
        seq = np.arange(20001, dtype=np.int32)
        queries = np.array([-1, 0, 1, 12, 19999, 20000, 20001, 20002])
        with patch.dict(os.environ, {"RECIR_LOCAL_RO_SEARCHSORTED": "1"}):
            a, b = jnp.asarray(seq), jnp.asarray(queries)
            for side in ("left", "right"):
                self.assertIsNone(_searchsorted_local_bins(a, b, side))
                result = jax.jit(
                    lambda a, b, side=side: _searchsorted_partitioned(a, b, side)
                )(a, b)
                np.testing.assert_array_equal(
                    np.asarray(result), np.searchsorted(seq, queries, side=side)
                )

    def test_empty_and_nonvector_bins_decline_local_search(self) -> None:
        queries = jnp.arange(8)
        with patch.dict(os.environ, {"RECIR_LOCAL_RO_SEARCHSORTED": "1"}):
            for seq in (jnp.empty(0), jnp.asarray(0), jnp.zeros((8, 8))):
                self.assertIsNone(_searchsorted_local_bins(seq, queries, "right"))

    def test_actual_shape_ignores_legacy_geometry_settings(self) -> None:
        seq = np.arange(20000, dtype=np.int32)
        queries = np.concatenate(
            [np.array([0, 1, 2, 2499]) + chip * 2500 for chip in range(8)]
        )
        env = {
            "RECIR_LOCAL_RO_SEARCHSORTED": "1",
            "RECIR_LOCAL_RO_SEARCHSORTED_GLOBAL_ROS": "invalid",
            "RECIR_LOCAL_RO_SEARCHSORTED_PER_CHIP": "invalid",
            "RECIR_NRO_DROPOUT_GLOBAL_ROS": "invalid",
        }
        for impl in ("binary", "compare"):
            env["RECIR_PARTITIONED_SEARCHSORTED_IMPL"] = impl
            with patch.dict(os.environ, env):
                for side in ("left", "right"):
                    result = jax.jit(
                        lambda a, b, side=side: _searchsorted_local_bins(a, b, side)
                    )(jnp.asarray(seq), jnp.asarray(queries))
                    np.testing.assert_array_equal(
                        np.asarray(result), np.searchsorted(seq, queries, side=side)
                    )

    def test_default_compare_exact_and_gather_gradient_matches(self) -> None:
        n = 128 * 257
        per_chip = n // 8
        rng = np.random.default_rng(29)
        counts = rng.integers(0, 8, size=n, dtype=np.dtype(np.int64))
        counts[per_chip - 4 : per_chip + 8] = 0
        seq = np.cumsum(counts)
        queries = []
        for chip in range(8):
            lower = seq[chip * per_chip - 1] if chip else 0
            upper = seq[(chip + 1) * per_chip - 1]
            q = rng.integers(lower, upper + 1, size=64, dtype=np.dtype(np.int64))
            q[:4] = [0, seq[-1], lower, upper]
            queries.append(q)
        queries = np.concatenate(queries)
        env = {
            "RECIR_LOCAL_RO_SEARCHSORTED": "1",
        }
        features = jnp.asarray(rng.normal(size=(n + 1, 4)), dtype=jnp.float32)
        with jax.enable_x64(True), patch.dict(os.environ, env):
            os.environ.pop("RECIR_PARTITIONED_SEARCHSORTED_IMPL", None)
            a, q = jnp.asarray(seq), jnp.asarray(queries)
            for side in ("left", "right"):
                expected = np.searchsorted(seq, queries, side=side).astype(np.int32)
                actual = jax.jit(
                    lambda a=a, q=q, side=side: _searchsorted_partitioned(a, q, side)
                )()
                np.testing.assert_array_equal(np.asarray(actual), expected)

                def candidate(f: jax.Array, a=a, q=q, side=side) -> jax.Array:
                    indices = _searchsorted_partitioned(a, q, side)
                    return jnp.sum(f[indices] * f[indices])

                def reference(f: jax.Array, a=a, q=q, side=side) -> jax.Array:
                    indices = jnp.searchsorted(a, q, side=side, method="scan_unrolled")
                    return jnp.sum(f[indices] * f[indices])

                loss, grad = jax.jit(jax.value_and_grad(candidate))(features)
                ref_loss, ref_grad = jax.jit(jax.value_and_grad(reference))(features)
                # Identical indices can still alter the float32 sum layout.
                # Keep an explicit six-decimal relative reduction tolerance;
                # The same tolerance covers scatter reduction reassociation
                # in the gradient. Search insertion indices remain exact.
                np.testing.assert_allclose(
                    np.asarray(loss), np.asarray(ref_loss), rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    np.asarray(grad), np.asarray(ref_grad), rtol=1e-6, atol=1e-6
                )
                print(
                    "LOCAL_SEARCH_NUMERICS",
                    side,
                    "loss_absolute_difference",
                    float(abs(loss - ref_loss)),
                    "loss_relative_difference",
                    float(abs((loss - ref_loss) / ref_loss)),
                    "gradient_max_absolute_difference",
                    float(jnp.max(jnp.abs(grad - ref_grad))),
                    flush=True,
                )
