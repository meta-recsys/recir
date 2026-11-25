# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import importlib
import math
import os
import unittest
from collections.abc import Iterator, Sequence
from typing import Any, cast
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from recir.ops.pallas.jagged_hstu_attention import (
    _ALIGNMENT_ROWS,
    _native_qkv_inputs,
    _single_block_fits,
    _single_block_stage,
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


def _kernel_name(equation: Any) -> str:
    """``pallas_call(name=...)`` lands on the kernel jaxpr's debug info."""
    jaxpr = equation.params.get("jaxpr")
    debug_info = getattr(getattr(jaxpr, "jaxpr", jaxpr), "debug_info", None)
    return str(getattr(debug_info, "func_name", ""))


def _attention_pallas_calls(nested_jaxprs: list[Any]) -> list[Any]:
    """The attention kernels, excluding the alignment compaction kernel."""
    return [
        equation
        for jaxpr in nested_jaxprs
        for equation in jaxpr.eqns
        if equation.primitive.name == "pallas_call"
        and "hstu_attention" in _kernel_name(equation)
    ]


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
        block: int = 4,
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
            block_q=block,
            block_k=block,
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

    def test_a_negative_target_count_cannot_reach_the_next_sequence(self) -> None:
        """num_targets is a count. A negative one puts max_id past length, and
        _keys_precede_queries reads that as the next sequence's keys being this
        sequence's history, so it skips the mask that would have excluded them.
        Clamping is defensive: no shape tried in interpret mode writes a
        different answer either way, and the output windows may be what stops
        it. This pins the invariant the skip's docstring already claims."""
        lengths = [4, 12]
        q, k, v, offsets = self._inputs(lengths)
        scale = 1.0 / math.sqrt(q.shape[-1])

        def run(num_targets: list[int]) -> Any:
            return np.asarray(
                pallas_jagged_hstu_attention(
                    q,
                    k,
                    v,
                    offsets,
                    jnp.asarray(num_targets, dtype=jnp.int32),
                    max_seq_len=16,
                    scale=scale,
                    inv_n=0.17,
                    interpret=True,
                    block_q=4,
                    block_k=4,
                )
            )

        # The second sequence's rows cannot depend on the first's target count.
        np.testing.assert_array_equal(run([-8, 0])[4:], run([0, 0])[4:])

    def test_sequences_within_max_seq_len_are_accepted(self) -> None:
        self._assert_parity([12, 3], [0, 0], max_seq_len=12)

    def test_a_sequence_near_the_end_matches_the_reference(self) -> None:
        # rows=400 against a 256 block: the second sequence starts at 200, past
        # the 400 - 256 - 8 = 136 the stage start used to be clamped to.
        self._assert_parity([200, 200], [0, 0], max_seq_len=200, block=256)

    def test_the_stage_starts_at_the_8_aligned_row_before_the_sequence(self) -> None:
        """So the sequence's first row is stage row ``start % 8``, and a sequence
        of up to ``block - 7`` rows fits the ``block``-row stage."""
        offsets = jnp.asarray([0, 200, 401], dtype=jnp.int32)

        for seq in (0, 1):
            with self.subTest(seq=seq):
                start, length, prefix = jax.jit(
                    lambda s=seq: _single_block_stage(offsets, s)
                )()
                self.assertEqual(int(start) % _ALIGNMENT_ROWS, 0)
                self.assertLess(int(prefix), _ALIGNMENT_ROWS)
                self.assertEqual(int(start) + int(prefix), int(offsets[seq]))
                self.assertEqual(int(length), int(offsets[seq + 1] - offsets[seq]))

    def test_single_block_fits_counts_the_rows_a_sequence_can_occupy(self) -> None:
        # A sequence can reach stage row max_seq_len + 7, rounded up to whole
        # 8-row groups, so a block not a multiple of 8 can lose up to 7 rows.
        self.assertTrue(_single_block_fits(max_seq_len=93, block_q=104, block_k=104))
        self.assertFalse(_single_block_fits(max_seq_len=98, block_q=104, block_k=104))
        self.assertFalse(_single_block_fits(max_seq_len=93, block_q=100, block_k=100))

    def test_packed_rows_are_not_padded_for_alignment(self) -> None:
        """Padding here fires a concatenate over the whole packed tensor."""
        q, k, v, offsets = self._inputs([200, 201], heads=1, dqk=8, dv=8)
        targets = jnp.zeros((2,), jnp.int32)

        _, hbm_args = jax.eval_shape(
            lambda q_, k_, v_: _native_qkv_inputs(
                q_, k_, v_, offsets, targets, 0.17, tail_rows=264
            ),
            q,
            k,
            v,
        )

        for arg in hbm_args:
            self.assertEqual(arg.shape[0], 401)

    def test_unaligned_packed_rows_match_the_reference(self) -> None:
        self._assert_parity([200, 201], [0, 0], max_seq_len=201, block=256)

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
        pallas_calls = _attention_pallas_calls(nested_jaxprs)
        self.assertEqual(len(pallas_calls), 1)
        return traced, nested_jaxprs, pallas_calls[0]

    def _kernel_vmem_avals(self, pallas_call: Any) -> list[Any]:
        kernel_jaxpr = cast(Any, pallas_call.params["jaxpr"])
        kernel_jaxpr = cast(Any, getattr(kernel_jaxpr, "jaxpr", kernel_jaxpr))
        return [
            variable.aval
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval)
        ]

    def _pipeline_scopes(
        self, nested_jaxprs: list[Any], buffered_shape: tuple[int, ...], count: int
    ) -> list[Any]:
        """Pipeline loops over at least ``count`` double-buffered refs of
        ``buffered_shape``."""
        return [
            jaxpr
            for jaxpr in nested_jaxprs
            if sum(
                tuple(getattr(variable.aval, "shape", ())) == buffered_shape
                for variable in jaxpr.invars
            )
            >= count
            and any(equation.primitive.name == "while" for equation in jaxpr.eqns)
        ]

    def test_native_forward_traces_double_buffered_pipeline(self) -> None:
        traced, nested_jaxprs, pallas_call = self._trace_native_forward(
            max_seq_len=3072,
            block_q=256,
            block_k=128,
            query_tiles_per_program=4,
        )
        # A sequence lies in the first 3,079 rows of its stage: four 1,024-row
        # query groups.
        self.assertEqual(pallas_call.params["grid_mapping"].grid, (1, 4))
        self.assertEqual(pallas_call.outvars[0].aval.shape, (3072, 128))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (3072, 1, 128))

        output_scratch = [
            aval
            for aval in self._kernel_vmem_avals(pallas_call)
            if aval.shape == (1, 1024, 128)
        ]
        self.assertEqual(
            [aval.dtype for aval in output_scratch], [jnp.dtype(jnp.float32)]
        )

        # K and V stream through one pipeline as double-buffered 2-D blocks.
        pipeline_scopes = self._pipeline_scopes(nested_jaxprs, (2, 128, 128), 2)
        self.assertEqual(len(pipeline_scopes), 1)
        pipeline_primitives = {
            equation.primitive.name
            for jaxpr in _walk_nested_jaxprs(pipeline_scopes[0])
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
        self.assertIn((2, 1024, 128), vmem_shapes)
        self.assertNotIn((3072, 128), vmem_shapes)

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
        self.assertEqual(pallas_call.outvars[0].aval.shape, (513, 128))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (513, 1, 128))

        output_scratch = [
            aval
            for aval in self._kernel_vmem_avals(pallas_call)
            if aval.shape == (1, 256, 128)
        ]
        self.assertEqual(
            [aval.dtype for aval in output_scratch], [jnp.dtype(jnp.float32)]
        )

        # Q is staged in a buffer of the same shape as the K/V blocks.
        pipeline_scopes = self._pipeline_scopes(nested_jaxprs, (2, 256, 128), 2)
        self.assertTrue(pipeline_scopes)
        buffered_refs = [
            variable.aval
            for variable in pipeline_scopes[0].invars
            if variable.aval.shape == (2, 256, 128)
        ]
        self.assertEqual(
            {aval.dtype for aval in buffered_refs}, {jnp.dtype(jnp.bfloat16)}
        )
        pipeline_primitives = {
            equation.primitive.name
            for jaxpr in _walk_nested_jaxprs(pipeline_scopes[0])
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
        # Three 256-row query tiles cover the sequence's 519 stage rows, not the
        # default four.
        self.assertEqual(pallas_call.params["grid_mapping"].grid, (1, 1))
        self.assertEqual(traced.jaxpr.outvars[0].aval.shape, (512, 1, 128))
        vmem_shapes = {aval.shape for aval in self._kernel_vmem_avals(pallas_call)}
        self.assertIn((1, 768, 128), vmem_shapes)
        self.assertNotIn((1, 1024, 128), vmem_shapes)

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
        pallas_calls = _attention_pallas_calls(nested_jaxprs)
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
        self.assertEqual(dq_call.outvars[0].aval.shape, (3072, 1, 128))
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
            sum(aval.shape == (2064, 1, 128) for aval in kernel_vmem_avals),
            2,
        )
        # The f32 dq accumulator plus the two bf16 q/dO transpose scratches.
        group_scratch = [
            aval for aval in kernel_vmem_avals if aval.shape == (1, 1024, 128)
        ]
        self.assertEqual(len(group_scratch), 3)
        self.assertEqual(
            sum(aval.dtype == jnp.dtype(jnp.float32) for aval in group_scratch),
            1,
        )
        self.assertEqual(
            sum(aval.dtype == jnp.dtype(jnp.bfloat16) for aval in group_scratch),
            2,
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
        # The kernels write packed, so no alignment conversion -- and so no
        # row gather -- reaches the graph at all.
        self.assertEqual(data_gather_shapes, [])

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
            sum(aval.shape == (272, 1, 128) for aval in kernel_vmem_avals),
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
        self.assertEqual(dq_call.outvars[0].aval.shape, (769, 1, 128))
        self.assertEqual(dkv_call.params["grid_mapping"].grid, (1, 3))
        self.assertEqual(
            [variable.aval.shape for variable in dkv_call.outvars],
            [(769, 1, 128)] * 2,
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
        self.assertIn((528, 1, 128), vmem_shapes)
        self.assertIn((1, 256, 128), vmem_shapes)
        self.assertIn((2, 392, 1, 128), vmem_shapes)

    def test_native_backward_zeroes_each_gradient_tail(self) -> None:
        """Each gradient's rows past ``seq_offsets[-1]`` are written by no grid
        program, and are zero only because ``_zero_unvisited_tail`` clears them.

        The kernel aliases its input into its output rather than allocating, so
        dropping the call leaves whatever the buffer held. Executing it to see
        that needs a TPU -- ``interpret=True`` takes the dense fallback, which
        has no tail kernel at all -- so this checks the native trace instead.
        """
        _, nested_jaxprs, _, _ = self._trace_native_backward(
            max_seq_len=3072,
            block_q=256,
            block_k=128,
            query_tiles_per_program=1,
        )
        tail_calls = [
            equation
            for jaxpr in nested_jaxprs
            for equation in jaxpr.eqns
            if equation.primitive.name == "pallas_call"
            and "tail_zero" in _kernel_name(equation)
        ]
        self.assertEqual(len(tail_calls), 3, "one per gradient: dq, dk and dv")
        for call in tail_calls:
            with self.subTest(kernel=_kernel_name(call)):
                # Aliasing the input is what makes the call load-bearing: the
                # output is not a fresh zeroed buffer.
                self.assertEqual(call.params["input_output_aliases"], ((1, 0),))

    def _trace_fused_backward(
        self, *, max_seq_len: int, block: int, dv: int = 128
    ) -> Any:
        qk = jax.ShapeDtypeStruct((max_seq_len, 1, 128), jnp.bfloat16)
        v = jax.ShapeDtypeStruct((max_seq_len, 1, dv), jnp.bfloat16)
        offsets = jnp.asarray([0, max_seq_len], dtype=jnp.int32)
        targets = jnp.zeros((1,), dtype=jnp.int32)
        inv_n = jnp.asarray(1.0 / max_seq_len, dtype=jnp.float32)
        with (
            _mock_tpu_generation(),
            mock.patch.dict(os.environ, {"RECIR_HSTU_FUSED_BWD": "1"}),
        ):
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
                    block_q=block,
                    block_k=block,
                    query_tiles_per_program=1,
                )
            )(v, qk, qk, v)
        pallas_calls = _attention_pallas_calls(list(_walk_nested_jaxprs(traced)))
        # dQ, dK and dV all come from one program.
        self.assertEqual(len(pallas_calls), 1)
        dq_shape, dv_shape = (max_seq_len, 1, 128), (max_seq_len, 1, dv)
        self.assertEqual(
            [variable.aval.shape for variable in traced.jaxpr.outvars],
            [dq_shape, dq_shape, dv_shape],
        )
        return pallas_calls[0]

    def test_fused_backward_handles_a_v_narrower_than_q(self) -> None:
        """Every other fused test uses D_v == D_qk. Interpret mode takes the
        dense fallback before the flag is read, so this checks the native trace
        of both fused kernels."""
        for max_seq_len in (200, 600):
            with self.subTest(max_seq_len=max_seq_len):
                call = self._trace_fused_backward(
                    max_seq_len=max_seq_len, block=256, dv=64
                )
                vmem_widths = {
                    variable.aval.shape[-1]
                    for jaxpr in _walk_nested_jaxprs(call.params["jaxpr"])
                    for variable in jaxpr.invars
                    if "vmem" in str(variable.aval)
                }
                self.assertIn(64, vmem_widths)

    def test_fused_backward_falls_back_when_dq_does_not_fit_vmem(self) -> None:
        """The fused dQ accumulator grows with the layer; past the VMEM limit
        the backward runs the separate dQ and dK/dV programs."""
        with mock.patch.dict(os.environ, {"RECIR_HSTU_FUSED_BWD": "1"}):
            _, _, dq_call, dkv_call = self._trace_native_backward(
                max_seq_len=131072,
                block_q=256,
                block_k=256,
                query_tiles_per_program=1,
            )
        self.assertNotIn("fused", _kernel_name(dq_call))
        self.assertNotIn("fused", _kernel_name(dkv_call))

    def _pipelines_with_buffers(
        self, call: Any, shapes: Sequence[tuple[int, ...]]
    ) -> list[Any]:
        """The kernel's pipeline loops over double-buffered refs of ``shapes``."""
        return [
            jaxpr
            for jaxpr in _walk_nested_jaxprs(call.params["jaxpr"])
            if any(equation.primitive.name == "while" for equation in jaxpr.eqns)
            and sorted(
                tuple(variable.aval.shape)
                for variable in jaxpr.invars
                if len(getattr(variable.aval, "shape", ())) >= 3
                and variable.aval.shape[0] == 2
            )
            == sorted(shapes)
        ]

    def test_fused_backward_pipelines_single_block_sequences(self) -> None:
        call = self._trace_fused_backward(max_seq_len=200, block=256)
        self.assertEqual(call.params["grid_mapping"].grid, (1,))
        # One pipeline over the sequences stages q, k, v and dO as 2-D blocks
        # and writes dK, dV and dQ as 2-D blocks with a spare row group. The
        # inputs are padded to a block and a row group.
        self.assertTrue(
            self._pipelines_with_buffers(
                call, [(2, 256, 128)] * 4 + [(2, 264, 128)] * 3
            )
        )
        self.assertEqual(
            [variable.aval.shape for variable in call.outvars], [(264, 128)] * 3
        )

    def test_native_forward_pipelines_single_block_sequences(self) -> None:
        _, _, call = self._trace_native_forward(
            max_seq_len=200, block_q=256, block_k=256
        )
        self.assertIn("fwd_single_block", _kernel_name(call))
        self.assertEqual(call.params["grid_mapping"].grid, (1,))
        self.assertTrue(
            self._pipelines_with_buffers(call, [(2, 256, 128)] * 3 + [(2, 264, 128)])
        )
        self.assertEqual(
            [variable.aval.shape for variable in call.outvars], [(264, 128)]
        )

    def _dot_output_shapes(self, call: Any) -> set[tuple[int, ...]]:
        return {
            tuple(equation.outvars[0].aval.shape)
            for jaxpr in _walk_nested_jaxprs(call.params["jaxpr"])
            for equation in jaxpr.eqns
            if equation.primitive.name == "dot_general"
        }

    def test_single_block_tiles_are_computed_in_causal_query_strips(self) -> None:
        # A 200-row sequence lies in the first 208 rows of its 256-row stage: a
        # 128-row strip sees 128 keys and the remaining 80 rows see 256; nothing
        # computes the full 256 x 256 scores.
        _, _, forward = self._trace_native_forward(
            max_seq_len=200, block_q=256, block_k=256
        )
        backward = self._trace_fused_backward(max_seq_len=200, block=256)
        for call in (forward, backward):
            shapes = self._dot_output_shapes(call)
            self.assertIn((1, 128, 128), shapes)
            self.assertIn((1, 80, 256), shapes)
            self.assertNotIn((1, 128, 256), shapes)
            self.assertNotIn((1, 256, 256), shapes)

    def test_fused_backward_computes_diagonal_blocks_in_query_strips(self) -> None:
        shapes = self._dot_output_shapes(
            self._trace_fused_backward(max_seq_len=600, block=256)
        )
        # Off-diagonal blocks are whole; the diagonal one is in strips.
        self.assertIn((1, 256, 256), shapes)
        self.assertIn((1, 128, 128), shapes)
        self.assertIn((1, 128, 256), shapes)

    def test_fused_backward_accumulates_dq_over_kv_blocks(self) -> None:
        call = self._trace_fused_backward(max_seq_len=600, block=256)
        self.assertEqual(call.params["grid_mapping"].grid, (1, 3))
        kernel_jaxpr = next(_walk_nested_jaxprs(call.params["jaxpr"]))
        accumulators = [
            variable.aval
            for variable in kernel_jaxpr.invars
            if "vmem" in str(variable.aval) and variable.aval.shape == (1, 768, 128)
        ]
        self.assertEqual(
            [aval.dtype for aval in accumulators], [jnp.dtype(jnp.float32)]
        )

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
