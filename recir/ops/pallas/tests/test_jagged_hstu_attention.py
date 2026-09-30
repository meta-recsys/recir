# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import importlib
import math
import unittest
from collections.abc import Iterator
from typing import Any, cast
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.jagged_hstu_attention import (
    jagged_hstu_attention_reference,
    pallas_jagged_hstu_attention,
    pallas_jagged_hstu_attention_backward,
)


def _walk_nested_jaxprs(value: Any) -> Iterator[Any]:
    jaxpr = getattr(value, "jaxpr", value)
    if hasattr(jaxpr, "eqns") and hasattr(jaxpr, "invars"):
        yield jaxpr
        for equation in jaxpr.eqns:
            for parameter in equation.params.values():
                yield from _walk_nested_jaxprs(parameter)
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_nested_jaxprs(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_nested_jaxprs(child)


@contextlib.contextmanager
def _mock_tpu_generation() -> Iterator[None]:
    pipeline_module = importlib.import_module("jax._src.pallas.mosaic.pipeline")
    with contextlib.ExitStack() as stack:
        if hasattr(pipeline_module, "_get_tpu_generation"):
            stack.enter_context(
                mock.patch.object(
                    pipeline_module,
                    "_get_tpu_generation",
                    return_value=5,
                )
            )
        else:
            tpu_info = importlib.import_module("jax._src.pallas.mosaic.tpu_info")
            stack.enter_context(
                mock.patch.object(
                    tpu_info,
                    "get_device_kind",
                    return_value="TPU v5p",
                )
            )
            stack.enter_context(
                mock.patch.object(
                    tpu_info,
                    "get_num_device_cores",
                    return_value=2,
                )
            )
        yield


def _pack_dense_output(
    dense: jax.Array,
    lengths: list[int],
) -> jax.Array:
    return jnp.concatenate(
        tuple(
            dense[index, :, :length, :].transpose(1, 0, 2)
            for index, length in enumerate(lengths)
        ),
        axis=0,
    )


class JaggedHSTUAttentionTest(unittest.TestCase):
    def _inputs(
        self,
        lengths: list[int],
        *,
        heads: int = 2,
        dqk: int = 8,
        dv: int = 6,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        total_length = sum(lengths)
        q_key, k_key, v_key = jax.random.split(jax.random.key(0), 3)
        q = jax.random.normal(q_key, (total_length, heads, dqk))
        k = jax.random.normal(k_key, (total_length, heads, dqk))
        v = jax.random.normal(v_key, (total_length, heads, dv))
        offsets = jnp.asarray([0, *np.cumsum(lengths)], dtype=jnp.int32)
        return q, k, v, offsets

    def _assert_parity(
        self,
        lengths: list[int],
        num_targets: list[int],
        *,
        max_seq_len: int,
        inv_n: float = 0.17,
    ) -> None:
        q, k, v, offsets = self._inputs(lengths)
        targets = jnp.asarray(num_targets, dtype=jnp.int32)
        scale = 1.0 / math.sqrt(q.shape[-1])

        actual = pallas_jagged_hstu_attention(
            q,
            k,
            v,
            offsets,
            targets,
            max_seq_len=max_seq_len,
            scale=scale,
            inv_n=inv_n,
            interpret=True,
            block_q=4,
            block_k=4,
        )
        expected = _pack_dense_output(
            jagged_hstu_attention_reference(
                q,
                k,
                v,
                offsets,
                targets,
                max_seq_len,
                scale,
                inv_n,
            ),
            lengths,
        )

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(actual.shape, (sum(lengths), 2, 6))

    def test_nonuniform_causal_sequences(self) -> None:
        self._assert_parity([7, 3, 9], [0, 0, 0], max_seq_len=9)

    def test_target_attention_and_empty_sequences(self) -> None:
        self._assert_parity([0, 6, 2, 0], [0, 2, 1, 0], max_seq_len=8)

    def test_start_offset_residues(self) -> None:
        for residue in range(8):
            with self.subTest(residue=residue):
                self._assert_parity(
                    [residue, 5],
                    [0, 2],
                    max_seq_len=8,
                )

    def test_block_size_invariance(self) -> None:
        q, k, v, offsets = self._inputs([5, 9])
        targets = jnp.asarray([1, 3], dtype=jnp.int32)
        kwargs = {
            "max_seq_len": 9,
            "scale": 0.31,
            "inv_n": 0.07,
            "interpret": True,
        }

        small = pallas_jagged_hstu_attention(
            q, k, v, offsets, targets, block_q=4, block_k=4, **kwargs
        )
        mixed = pallas_jagged_hstu_attention(
            q, k, v, offsets, targets, block_q=8, block_k=4, **kwargs
        )

        np.testing.assert_allclose(small, mixed, rtol=2e-5, atol=2e-5)

    def test_backward_parity(self) -> None:
        for lengths, num_targets, max_seq_len in (
            ([7, 3, 9], [0, 0, 0], 9),
            ([0, 6, 2, 0], [0, 2, 1, 0], 8),
            ([3, 5], [0, 2], 8),
            ([1, 1, 1, 1, 1, 1, 1, 1, 2], [0] * 9, 2),
        ):
            with self.subTest(lengths=lengths, num_targets=num_targets):
                q, k, v, offsets = self._inputs(lengths)
                targets = jnp.asarray(num_targets, dtype=jnp.int32)
                alpha = 1.0 / math.sqrt(q.shape[-1])
                inv_n = 0.17
                output = _pack_dense_output(
                    jagged_hstu_attention_reference(
                        q,
                        k,
                        v,
                        offsets,
                        targets,
                        max_seq_len,
                        alpha,
                        inv_n,
                    ),
                    lengths,
                )
                grad_out = jax.random.normal(jax.random.key(7), output.shape)

                _, pullback = jax.vjp(
                    lambda q_,
                    k_,
                    v_,
                    offsets=offsets,
                    targets=targets,
                    max_seq_len=max_seq_len,
                    alpha=alpha,
                    inv_n=inv_n,
                    lengths=lengths: _pack_dense_output(
                        jagged_hstu_attention_reference(
                            q_,
                            k_,
                            v_,
                            offsets,
                            targets,
                            max_seq_len,
                            alpha,
                            inv_n,
                        ),
                        lengths,
                    ),
                    q,
                    k,
                    v,
                )
                expected = pullback(grad_out)
                actual = pallas_jagged_hstu_attention_backward(
                    grad_out,
                    q,
                    k,
                    v,
                    offsets,
                    targets,
                    jnp.asarray(inv_n, dtype=jnp.float32),
                    max_seq_len=max_seq_len,
                    alpha=alpha,
                    interpret=True,
                    block_q=4,
                    block_k=4,
                )

                for actual_grad, expected_grad in zip(actual, expected):
                    np.testing.assert_allclose(
                        actual_grad,
                        expected_grad,
                        rtol=2e-4,
                        atol=2e-4,
                    )

    def _trace_native_forward(
        self,
        *,
        max_seq_len: int,
        block_q: int,
        block_k: int,
        query_tiles_per_program: int | None = None,
    ) -> tuple[Any, list[Any], Any]:
        qkv = jax.ShapeDtypeStruct((max_seq_len, 1, 128), jnp.bfloat16)
        offsets = jnp.asarray([0, max_seq_len], dtype=jnp.int32)
        targets = jnp.zeros((1,), dtype=jnp.int32)
        kwargs: dict[str, int] = {}
        if query_tiles_per_program is not None:
            kwargs["query_tiles_per_program"] = query_tiles_per_program

        with _mock_tpu_generation():
            traced = jax.make_jaxpr(
                lambda q_, k_, v_: pallas_jagged_hstu_attention(
                    q_,
                    k_,
                    v_,
                    offsets,
                    targets,
                    max_seq_len=max_seq_len,
                    scale=1.0 / math.sqrt(128),
                    inv_n=1.0 / max_seq_len,
                    interpret=False,
                    block_q=block_q,
                    block_k=block_k,
                    **kwargs,
                )
            )(qkv, qkv, qkv)

        nested_jaxprs = list(_walk_nested_jaxprs(traced))
        pallas_calls = [
            equation
            for jaxpr in nested_jaxprs
            for equation in jaxpr.eqns
            if equation.primitive.name == "pallas_call"
        ]
        self.assertEqual(len(pallas_calls), 1)
        return traced, nested_jaxprs, pallas_calls[0]

    def test_native_forward_traces_double_buffered_pipeline(self) -> None:
        traced, nested_jaxprs, pallas_call = self._trace_native_forward(
            max_seq_len=3072,
            block_q=256,
            block_k=128,
        )
        self.assertEqual(pallas_call.params["grid_mapping"].grid, (1, 3))
        self.assertEqual(pallas_call.outvars[0].aval.shape, (3079, 1, 128))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (3072, 1, 128))

        kernel_jaxpr = cast(Any, pallas_call.params["jaxpr"])
        kernel_jaxpr = cast(Any, getattr(kernel_jaxpr, "jaxpr", kernel_jaxpr))
        kernel_vmem_avals = [
            variable.aval
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval)
        ]
        self.assertEqual(
            sum(aval.shape == (1032, 1, 128) for aval in kernel_vmem_avals),
            1,
        )
        output_scratch = [
            aval for aval in kernel_vmem_avals if aval.shape == (1, 1024, 128)
        ]
        self.assertEqual(len(output_scratch), 1)
        self.assertEqual(
            {aval.dtype for aval in output_scratch},
            {jnp.dtype(jnp.float32)},
        )

        buffered_shape = (2, 136, 1, 128)
        pipeline_scopes = []
        for jaxpr in nested_jaxprs:
            input_avals = [variable.aval for variable in jaxpr.invars]
            buffered_refs = [
                aval
                for aval in input_avals
                if tuple(getattr(aval, "shape", ())) == buffered_shape
            ]
            primitive_names = {equation.primitive.name for equation in jaxpr.eqns}
            if len(buffered_refs) == 2 and "while" in primitive_names:
                pipeline_scopes.append(jaxpr)

        self.assertEqual(len(pipeline_scopes), 1)
        pipeline_scope = pipeline_scopes[0]
        semaphore_refs = [
            variable.aval
            for variable in pipeline_scope.invars
            if tuple(getattr(variable.aval, "shape", ())) == (2,)
            and "dma_sem" in str(variable.aval)
        ]
        self.assertEqual(len(semaphore_refs), 2)

        pipeline_primitives = {
            equation.primitive.name
            for jaxpr in _walk_nested_jaxprs(pipeline_scope)
            for equation in jaxpr.eqns
        }
        self.assertTrue(
            {"dma_start", "dma_wait", "while", "dot_general"} <= pipeline_primitives
        )

        vmem_shapes = {
            tuple(getattr(variable.aval, "shape", ()))
            for jaxpr in nested_jaxprs
            for variable in jaxpr.invars
            if "vmem" in str(variable.aval)
        }
        self.assertIn((1032, 1, 128), vmem_shapes)
        self.assertIn((1, 1024, 128), vmem_shapes)
        self.assertIn(buffered_shape, vmem_shapes)
        self.assertNotIn((3072, 1, 128), vmem_shapes)

        dot_operand_shapes = {
            tuple(variable.aval.shape)
            for jaxpr in nested_jaxprs
            for equation in jaxpr.eqns
            if equation.primitive.name == "dot_general"
            for variable in equation.invars
        }
        self.assertIn((1, 256, 128), dot_operand_shapes)
        self.assertNotIn((1, 1024, 128), dot_operand_shapes)

    def test_native_forward_groups_partial_asymmetric_query_tiles(self) -> None:
        traced, nested_jaxprs, pallas_call = self._trace_native_forward(
            max_seq_len=513,
            block_q=128,
            block_k=256,
            query_tiles_per_program=2,
        )
        self.assertEqual(pallas_call.params["grid_mapping"].grid, (1, 3))
        self.assertEqual(pallas_call.outvars[0].aval.shape, (520, 1, 128))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (513, 1, 128))

        kernel_jaxpr = cast(Any, pallas_call.params["jaxpr"])
        kernel_jaxpr = cast(Any, getattr(kernel_jaxpr, "jaxpr", kernel_jaxpr))
        kernel_vmem_avals = [
            variable.aval
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval)
        ]
        self.assertEqual(
            sum(aval.shape == (264, 1, 128) for aval in kernel_vmem_avals),
            1,
        )
        output_scratch = [
            aval for aval in kernel_vmem_avals if aval.shape == (1, 256, 128)
        ]
        self.assertEqual(len(output_scratch), 1)
        self.assertEqual(
            {aval.dtype for aval in output_scratch},
            {jnp.dtype(jnp.float32)},
        )

        buffered_shape = (2, 264, 1, 128)
        pipeline_scopes = [
            jaxpr
            for jaxpr in nested_jaxprs
            if sum(variable.aval.shape == buffered_shape for variable in jaxpr.invars)
            == 2
            and any(equation.primitive.name == "while" for equation in jaxpr.eqns)
        ]
        self.assertEqual(len(pipeline_scopes), 1)
        pipeline_scope = pipeline_scopes[0]
        buffered_refs = [
            variable.aval
            for variable in pipeline_scope.invars
            if variable.aval.shape == buffered_shape
        ]
        self.assertEqual(len(buffered_refs), 2)
        self.assertEqual(
            {aval.dtype for aval in buffered_refs},
            {jnp.dtype(jnp.bfloat16)},
        )
        pipeline_primitives = {
            equation.primitive.name
            for jaxpr in _walk_nested_jaxprs(pipeline_scope)
            for equation in jaxpr.eqns
        }
        self.assertTrue(
            {"dma_start", "dma_wait", "while", "dot_general"} <= pipeline_primitives
        )

    def test_native_forward_limits_group_to_sequence(self) -> None:
        traced, nested_jaxprs, pallas_call = self._trace_native_forward(
            max_seq_len=512,
            block_q=256,
            block_k=128,
        )
        self.assertEqual(pallas_call.params["grid_mapping"].grid, (1, 1))
        self.assertEqual(pallas_call.outvars[0].aval.shape, (519, 1, 128))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (512, 1, 128))

        kernel_jaxpr = cast(Any, pallas_call.params["jaxpr"])
        kernel_jaxpr = cast(Any, getattr(kernel_jaxpr, "jaxpr", kernel_jaxpr))
        vmem_shapes = {
            tuple(getattr(variable.aval, "shape", ()))
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval)
        }
        self.assertIn((520, 1, 128), vmem_shapes)
        self.assertIn((1, 512, 128), vmem_shapes)
        self.assertNotIn((1032, 1, 128), vmem_shapes)

    def _trace_native_backward(
        self,
        *,
        max_seq_len: int,
        block_q: int,
        block_k: int,
        query_tiles_per_program: int,
    ) -> tuple[Any, list[Any], Any, Any]:
        qkv = jax.ShapeDtypeStruct((max_seq_len, 1, 128), jnp.bfloat16)
        grad_out = jax.ShapeDtypeStruct((max_seq_len, 1, 128), jnp.bfloat16)
        offsets = jnp.asarray([0, max_seq_len], dtype=jnp.int32)
        targets = jnp.zeros((1,), dtype=jnp.int32)
        inv_n = jnp.asarray(1.0 / max_seq_len, dtype=jnp.float32)

        with _mock_tpu_generation():
            traced = jax.make_jaxpr(
                lambda do_, q_, k_, v_: pallas_jagged_hstu_attention_backward(
                    do_,
                    q_,
                    k_,
                    v_,
                    offsets,
                    targets,
                    inv_n,
                    max_seq_len=max_seq_len,
                    alpha=1.0 / math.sqrt(128),
                    interpret=False,
                    block_q=block_q,
                    block_k=block_k,
                    query_tiles_per_program=query_tiles_per_program,
                )
            )(grad_out, qkv, qkv, qkv)

        nested_jaxprs = list(_walk_nested_jaxprs(traced))
        pallas_calls = [
            equation
            for jaxpr in nested_jaxprs
            for equation in jaxpr.eqns
            if equation.primitive.name == "pallas_call"
        ]
        self.assertEqual(len(pallas_calls), 2)
        dq_calls = [equation for equation in pallas_calls if len(equation.outvars) == 1]
        dkv_calls = [
            equation for equation in pallas_calls if len(equation.outvars) == 2
        ]
        self.assertEqual(len(dq_calls), 1)
        self.assertEqual(len(dkv_calls), 1)
        return traced, nested_jaxprs, dq_calls[0], dkv_calls[0]

    def test_native_backward_traces_double_buffered_dq(self) -> None:
        traced, _, dq_call, _ = self._trace_native_backward(
            max_seq_len=3072,
            block_q=256,
            block_k=128,
            query_tiles_per_program=4,
        )
        self.assertEqual(dq_call.params["grid_mapping"].grid, (1, 3))
        self.assertEqual(dq_call.outvars[0].aval.shape, (3079, 1, 128))
        self.assertEqual(
            [variable.aval.shape for variable in traced.jaxpr.outvars],
            [(3072, 1, 128)] * 3,
        )

        dq_jaxprs = list(_walk_nested_jaxprs(dq_call.params["jaxpr"]))
        kernel_jaxpr = dq_jaxprs[0]
        kernel_vmem_avals = [
            variable.aval
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval)
        ]
        self.assertEqual(
            sum(aval.shape == (1032, 1, 128) for aval in kernel_vmem_avals),
            2,
        )
        group_scratch = [
            aval for aval in kernel_vmem_avals if aval.shape == (1, 1024, 128)
        ]
        self.assertEqual(len(group_scratch), 1)
        self.assertEqual(
            sum(aval.dtype == jnp.dtype(jnp.float32) for aval in group_scratch),
            1,
        )

        buffered_shape = (2, 136, 1, 128)
        pipeline_scopes = [
            jaxpr
            for jaxpr in dq_jaxprs
            if sum(variable.aval.shape == buffered_shape for variable in jaxpr.invars)
            == 2
            and any(equation.primitive.name == "while" for equation in jaxpr.eqns)
        ]
        self.assertEqual(len(pipeline_scopes), 1)
        pipeline_primitives = {
            equation.primitive.name
            for jaxpr in _walk_nested_jaxprs(pipeline_scopes[0])
            for equation in jaxpr.eqns
        }
        self.assertTrue(
            {"dma_start", "dma_wait", "while", "dot_general"} <= pipeline_primitives
        )

        dot_operand_shapes = {
            tuple(variable.aval.shape)
            for jaxpr in dq_jaxprs
            for equation in jaxpr.eqns
            if equation.primitive.name == "dot_general"
            for variable in equation.invars
        }
        self.assertIn((1, 256, 128), dot_operand_shapes)
        self.assertNotIn((1, 1024, 128), dot_operand_shapes)
        self.assertNotIn((3072, 1, 128), {aval.shape for aval in kernel_vmem_avals})

        data_gather_shapes = [
            tuple(variable.aval.shape)
            for jaxpr in _walk_nested_jaxprs(traced)
            for equation in jaxpr.eqns
            if equation.primitive.name == "gather"
            for variable in equation.outvars
            if variable.aval.dtype == jnp.dtype(jnp.bfloat16)
            and len(variable.aval.shape) == 3
        ]
        self.assertEqual(data_gather_shapes, [(3072, 1, 384)])

    def test_native_backward_traces_double_buffered_dkv(self) -> None:
        _, _, _, dkv_call = self._trace_native_backward(
            max_seq_len=3072,
            block_q=256,
            block_k=128,
            query_tiles_per_program=4,
        )
        self.assertEqual(dkv_call.params["grid_mapping"].grid, (1, 24))

        dkv_jaxprs = list(_walk_nested_jaxprs(dkv_call.params["jaxpr"]))
        kernel_jaxpr = dkv_jaxprs[0]
        kernel_vmem_avals = [
            variable.aval
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval)
        ]
        self.assertEqual(
            sum(aval.shape == (136, 1, 128) for aval in kernel_vmem_avals),
            2,
        )
        self.assertEqual(
            sum(
                aval.shape == (1, 128, 128) and aval.dtype == jnp.dtype(jnp.float32)
                for aval in kernel_vmem_avals
            ),
            2,
        )

        buffered_shape = (2, 264, 1, 128)
        pipeline_scopes = [
            jaxpr
            for jaxpr in dkv_jaxprs
            if sum(variable.aval.shape == buffered_shape for variable in jaxpr.invars)
            == 2
            and any(equation.primitive.name == "while" for equation in jaxpr.eqns)
        ]
        self.assertEqual(len(pipeline_scopes), 1)
        pipeline_primitives = {
            equation.primitive.name
            for jaxpr in _walk_nested_jaxprs(pipeline_scopes[0])
            for equation in jaxpr.eqns
        }
        self.assertTrue(
            {"dma_start", "dma_wait", "while", "dot_general"} <= pipeline_primitives
        )
        self.assertNotIn((3072, 1, 128), {aval.shape for aval in kernel_vmem_avals})

    def test_native_backward_groups_partial_asymmetric_query_tiles(self) -> None:
        traced, _, dq_call, dkv_call = self._trace_native_backward(
            max_seq_len=769,
            block_q=128,
            block_k=384,
            query_tiles_per_program=2,
        )
        self.assertEqual(dq_call.params["grid_mapping"].grid, (1, 4))
        self.assertEqual(dq_call.outvars[0].aval.shape, (776, 1, 128))
        self.assertEqual(dkv_call.params["grid_mapping"].grid, (1, 3))
        self.assertEqual(
            [variable.aval.shape for variable in dkv_call.outvars],
            [(776, 1, 128)] * 2,
        )
        self.assertEqual(
            [variable.aval.shape for variable in traced.jaxpr.outvars],
            [(769, 1, 128)] * 3,
        )

        dq_jaxprs = list(_walk_nested_jaxprs(dq_call.params["jaxpr"]))
        vmem_shapes = {
            tuple(variable.aval.shape)
            for jaxpr in dq_jaxprs
            for variable in jaxpr.invars
            if "vmem" in str(variable.aval)
        }
        self.assertIn((264, 1, 128), vmem_shapes)
        self.assertIn((1, 256, 128), vmem_shapes)
        self.assertIn((2, 392, 1, 128), vmem_shapes)

    def test_rejects_invalid_shapes(self) -> None:
        q, k, v, offsets = self._inputs([3, 2])
        with self.assertRaisesRegex(ValueError, "shape \\[B\\]"):
            pallas_jagged_hstu_attention(
                q,
                k,
                v,
                offsets,
                jnp.zeros((3,), dtype=jnp.int32),
                max_seq_len=4,
                interpret=True,
            )

    def test_backward_rejects_dense_grad_out(self) -> None:
        q, k, v, offsets = self._inputs([3, 2])
        targets = jnp.zeros((2,), dtype=jnp.int32)
        grad_out = jnp.zeros((2, 2, 4, 6), dtype=v.dtype)

        with self.assertRaisesRegex(ValueError, "packed shape"):
            pallas_jagged_hstu_attention_backward(
                grad_out,
                q,
                k,
                v,
                offsets,
                targets,
                jnp.asarray(0.25, dtype=jnp.float32),
                max_seq_len=4,
                alpha=0.5,
                interpret=True,
                block_q=4,
                block_k=4,
            )
