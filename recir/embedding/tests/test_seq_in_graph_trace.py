# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU FX-trace check for InGraphSparseArch.

Answers (without TPU): does the in-graph module trace cleanly -- i.e. does the
KeyedJaggedTensor flow through, and does the lookup appear as the expected node
per table SHARDING -- ROWWISE -> opaque ``torch.ops.recir.ringshift_embedding``
(lowered via the ringshift path); FSDP -> ``recir.fsdp_embedding``; REPLICATED
-> ``embedding`` (lowered via F.embedding).
"""

from __future__ import annotations

import dataclasses
import unittest

import torch
from recir.embedding.backend import (
    table_plans_from_embedding_tables,
    TableOptimizer,
    TablePlan,
    TableSharding,
)
from recir.embedding.in_graph import (
    _seed_in_graph_tables_from_ec,
    DP_TABLE_PREFIX,
    InGraphSparseArch,
    SHARDED_TABLE_PREFIX,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@dataclasses.dataclass
class _FakeTable:
    """The one field ``table_plans_from_embedding_tables`` reads.

    Not the real ``EmbeddingTable``: constructing that pulls the whole MVAI
    config chain, and the mapping only ever touches ``.sharding`` -- including
    the cases the real dataclass cannot express, like a bare string left behind
    by a bad deserialization.
    """

    sharding: object


_SPARSE_ROWWISE: TablePlan = TablePlan(TableSharding.ROWWISE, TableOptimizer.SPARSE)
_SPARSE_FSDP: TablePlan = TablePlan(TableSharding.FSDP, TableOptimizer.SPARSE)
_SPARSE_REPLICATED: TablePlan = TablePlan(
    TableSharding.REPLICATED, TableOptimizer.SPARSE
)


def _build_module_and_kjt(
    big: TablePlan = _SPARSE_ROWWISE,
    small: TablePlan = _SPARSE_REPLICATED,
    on_device_init: bool = False,
):
    """Build a 2-table module + a matching KJT.

    big table = 256 rows (f0,f1, S=4), small table = 16 rows (c0, S=1). Each
    table's plan is passed in: nothing is derived from the row counts, so the
    two sizes here only exercise the padding, not the mode.
    """
    world_size = 4
    batch = 4
    table_configs = [
        EmbeddingConfig(
            name="big",
            num_embeddings=256,
            embedding_dim=8,
            feature_names=["f0", "f1"],
        ),
        EmbeddingConfig(
            name="small",
            num_embeddings=16,
            embedding_dim=8,
            feature_names=["c0"],
        ),
    ]
    feature_seqlens = {"f0": 4, "f1": 4, "c0": 1}
    module = InGraphSparseArch(
        table_configs,
        feature_seqlens,
        world_size=world_size,
        table_plans={"big": big, "small": small},
        on_device_init=on_device_init,
    )
    # KJT: keys [f0,f1,c0]; per-key value block = batch*S = 16, 16, 4 -> 36 values.
    # ids valid for both the big (256) and small (16) tables.
    values = torch.randint(0, 16, (36,), dtype=torch.int64)
    lengths = torch.tensor([4] * batch + [4] * batch + [1] * batch, dtype=torch.int64)
    kjt = KeyedJaggedTensor(keys=["f0", "f1", "c0"], values=values, lengths=lengths)
    return module, kjt, batch


def _big_feature_ref(module, kjt, batch):
    """Yield (key, module_output, reference F.embedding) for the big table's
    features, using whichever dict currently holds the big weight."""
    weight = (
        module._sharded_tables["big"]
        if "big" in module._sharded_tables
        else module._dp_tables["big"]
    )
    out = module(kjt)
    seqlens = {"f0": 4, "f1": 4, "c0": 1}
    values = kjt.values()
    offset = 0
    for key in kjt.keys():
        n = batch * seqlens[key]
        if key in ("f0", "f1"):
            ids = values[offset : offset + n].reshape(batch * seqlens[key]).long()
            yield key, out[key].values(), torch.nn.functional.embedding(ids, weight)
        offset += n


class SeqInGraphTraceTest(unittest.TestCase):
    def test_ringshift_op_preserves_bf16_padding_gradient_semantics(self) -> None:
        import recir.ops.seq_embedding_op  # noqa: F401

        ids = torch.tensor([[0, 3, 0], [2, 3, 1]], dtype=torch.int64)
        lengths = torch.full((2,), 3, dtype=torch.int32)
        values = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
        actual_weight = values.clone().requires_grad_(True)
        reference_weight = values.clone().requires_grad_(True)
        output_gradient = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0], [3.0] * 4],
                [[2.0] * 4, [4.0] * 4, [5.0] * 4],
            ],
            dtype=torch.bfloat16,
        )

        actual = torch.ops.recir.ringshift_embedding(
            ids,
            lengths,
            actual_weight,
            0,
        )
        reference = torch.nn.functional.embedding(
            ids,
            reference_weight,
            padding_idx=0,
        )
        actual.backward(output_gradient)
        reference.backward(output_gradient)

        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        actual_gradient = actual_weight.grad
        self.assertIsNotNone(actual_gradient)
        assert actual_gradient is not None
        torch.testing.assert_close(
            actual_gradient,
            reference_weight.grad,
            atol=0,
            rtol=0,
        )
        self.assertEqual(0, int(torch.count_nonzero(actual_gradient[0])))

    def test_eager_shapes(self) -> None:
        module, kjt, batch = _build_module_and_kjt()
        out = module(kjt)
        self.assertEqual(set(out.keys()), {"f0", "f1", "c0"})
        self.assertEqual(out["f0"].values().shape, (batch * 4, 8))
        self.assertEqual(out["c0"].values().shape, (batch * 1, 8))

    def test_presliced_matches_flat(self) -> None:
        """The per-feature presliced path (what the input-break wrapper feeds
        in-graph) must be numerically identical to the flat-KJT forward.

        This is the core invariant of the eager per-feature slicing fix: slicing
        the feature-major merged values into per-feature [B, S_f] blocks eagerly
        and embedding those blocks gives the same result as the in-graph flat
        slice -- only the SHARDING of the injected ids changes (aligned vs not).
        """
        module, kjt, batch = _build_module_and_kjt()
        keys = kjt.keys()
        values = kjt.values()
        lengths = kjt.lengths()
        seqlens = {"f0": 4, "f1": 4, "c0": 1}

        # Mirror _slice_per_feature / the wrapper's eager fwd_fn.
        ids_2d = []
        lengths_1d = []
        offset = 0
        for i, key in enumerate(keys):
            n = batch * seqlens[key]
            ids_2d.append(values[offset : offset + n].reshape(batch, seqlens[key]))
            lengths_1d.append(lengths[i * batch : (i + 1) * batch])
            offset += n

        out_flat = module(kjt)
        out_presliced = module._forward_presliced(keys, ids_2d, lengths_1d)

        self.assertEqual(set(out_flat.keys()), set(out_presliced.keys()))
        for key in out_flat:
            torch.testing.assert_close(
                out_flat[key].values(), out_presliced[key].values()
            )
            torch.testing.assert_close(
                out_flat[key].lengths(), out_presliced[key].lengths()
            )

    def test_fx_trace_has_opaque_op(self) -> None:
        module, kjt, _ = _build_module_and_kjt()
        keys = kjt.keys()
        values = kjt.values()
        lengths = kjt.lengths()

        # Trace over the (values, lengths) TENSORS and build the KJT inside, which
        # mirrors recir's real flow (the KJT is an intermediate built from traced
        # tensors, not a placeholder-substituted KJT input -- the latter breaks
        # make_fx on KJT.__init__'s offsets validation).
        def fn(values: torch.Tensor, lengths: torch.Tensor):
            k = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
            return module(k)

        # pre_dispatch=True keeps custom ops (recir.ringshift_embedding) opaque (traced
        # before CompositeExplicitAutograd decomposition); real tensors avoid the
        # fake/real param mix. ids in [0,16) are valid for both tables.
        gm = make_fx(fn, pre_dispatch=True)(values, lengths)
        graph_str = gm.graph.python_code("self").src
        print("\n===== FX GRAPH (InGraphSparseArch) =====\n" + graph_str)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        seq_nodes = [t for t in targets if t.endswith("ringshift_embedding.default")]
        emb_nodes = [t for t in targets if t.endswith("aten.embedding.default")]
        # f0 + f1 share the "big" ROWWISE table -> batched into ONE ringshift_embedding.
        self.assertEqual(
            len(seq_nodes),
            1,
            f"expected 1 batched ringshift_embedding node, got {targets}",
        )
        # small DP table -> a plain aten.embedding (F.embedding) node.
        self.assertGreaterEqual(
            len(emb_nodes), 1, f"no aten.embedding node in {targets}"
        )

    def test_op_output_on_compute_device_with_meta_weight(self) -> None:
        """The meta (on-device-init) table must NOT leak device=meta into the
        activation. With a meta weight + cpu ids, the op's fake kernel must put the
        output on the compute (ids) device. This is the exact fix for the
        'FakeTensor Device Propagation ... cpu, meta' trace failure (w9pn2).
        """
        import recir.ops.seq_embedding_op  # noqa: F401  (register the op)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            ids = torch.zeros(4, 4, dtype=torch.int64)  # cpu
            lengths = torch.full((4,), 4, dtype=torch.int64)  # cpu
            weight = torch.empty(256, 8, device="meta")  # meta (on-device init)
            out = torch.ops.recir.ringshift_embedding(ids, lengths, weight)
            self.assertEqual(
                str(out.device), "cpu", f"op output on {out.device}, expected cpu"
            )
            self.assertEqual(tuple(out.shape), (4, 4, 8))
            self.assertEqual(out.dtype, weight.dtype)

    def test_seed_skips_meta_source_weights(self) -> None:
        class OldEC(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = torch.nn.ModuleDict()
                for name, rows in (("big", 256), ("small", 16)):
                    table = torch.nn.Module()
                    table.register_parameter(
                        "weight",
                        torch.nn.Parameter(torch.empty(rows, 8, device="meta")),
                    )
                    self.embeddings[name] = table

        module, _, _ = _build_module_and_kjt(on_device_init=False)

        _seed_in_graph_tables_from_ec(OldEC(), module)

        self.assertFalse(module._dp_tables["small"].is_meta)
        self.assertFalse(module._sharded_tables["big"].is_meta)

    # ---- table sharding routing (ROWWISE / FSDP / REPLICATED) ----

    def test_default_plans(self) -> None:
        """The plan, not the row count, decides where a table is stored."""
        module, _, _ = _build_module_and_kjt()
        self.assertEqual(module._table_sharding["big"], TableSharding.ROWWISE)
        self.assertEqual(module._table_sharding["small"], TableSharding.REPLICATED)
        self.assertIn("big", module._sharded_tables)
        self.assertIn("small", module._dp_tables)
        names = [n for n, _ in module.named_parameters()]
        self.assertTrue(any(SHARDED_TABLE_PREFIX in n and "big" in n for n in names))
        self.assertTrue(any(DP_TABLE_PREFIX in n and "small" in n for n in names))

    def test_row_count_does_not_decide_sharding(self) -> None:
        """The 256-row table is REPLICATED and the 16-row one is ROWWISE purely
        because the plans say so. This is the whole point of the explicit spec:
        the old code derived both from two row thresholds, so a table could not
        be placed against its size."""
        module, _, _ = _build_module_and_kjt(
            big=_SPARSE_REPLICATED, small=_SPARSE_ROWWISE
        )
        self.assertIn("big", module._dp_tables)
        self.assertIn("small", module._sharded_tables)

    def test_missing_plan_is_rejected(self) -> None:
        """A table with no plan must fail loudly. Defaulting it would pick an
        optimizer for it, and the wrong one is invisible in every shape and key."""
        with self.assertRaisesRegex(ValueError, "no TablePlan for \\['small'\\]"):
            InGraphSparseArch(
                [
                    EmbeddingConfig(
                        name="big",
                        num_embeddings=256,
                        embedding_dim=8,
                        feature_names=["f0"],
                    ),
                    EmbeddingConfig(
                        name="small",
                        num_embeddings=16,
                        embedding_dim=8,
                        feature_names=["c0"],
                    ),
                ],
                {"f0": 4, "c0": 1},
                world_size=4,
                table_plans={"big": _SPARSE_ROWWISE},
            )

    def test_replicated_plan_routes_big(self) -> None:
        """REPLICATED: big table replicated + F.embedding, NOT sharded (DP prefix,
        so the FSDP filter leaves it alone)."""
        module, _, _ = _build_module_and_kjt(big=_SPARSE_REPLICATED)
        self.assertEqual(module._table_sharding["big"], TableSharding.REPLICATED)
        self.assertIn("big", module._dp_tables)
        self.assertNotIn("big", module._sharded_tables)
        names = [n for n, _ in module.named_parameters()]
        self.assertTrue(any(DP_TABLE_PREFIX in n and "big" in n for n in names))
        self.assertFalse(any(SHARDED_TABLE_PREFIX in n and "big" in n for n in names))

    def test_fsdp_plan_routes_big(self) -> None:
        """FSDP: big table dim-0 sharded (SHARDED_TABLE_PREFIX, so the FSDP filter
        shards it) but looked up via recir.fsdp_embedding, not the ring-shift
        op."""
        module, _, _ = _build_module_and_kjt(big=_SPARSE_FSDP)
        self.assertEqual(module._table_sharding["big"], TableSharding.FSDP)
        self.assertIn("big", module._sharded_tables)
        self.assertNotIn("big", module._dp_tables)
        names = [n for n, _ in module.named_parameters()]
        self.assertTrue(any(SHARDED_TABLE_PREFIX in n and "big" in n for n in names))

    def _assert_no_seq_op_uses_op(self, module, kjt, op_suffix: str) -> None:
        keys = kjt.keys()
        values = kjt.values()
        lengths = kjt.lengths()

        def fn(values: torch.Tensor, lengths: torch.Tensor):
            k = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
            return module(k)

        gm = make_fx(fn, pre_dispatch=True)(values, lengths)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        seq_nodes = [t for t in targets if t.endswith("ringshift_embedding.default")]
        big_nodes = [t for t in targets if t.endswith(op_suffix)]
        self.assertEqual(
            len(seq_nodes),
            0,
            f"FSDP/REPLICATED path must emit no ringshift_embedding: {targets}",
        )
        # big table feature(s) -> the expected lookup op (no ring-shift).
        self.assertGreaterEqual(
            len(big_nodes), 1, f"expected a {op_suffix} node, got {targets}"
        )

    def test_replicated_fx_uses_aten_embedding(self) -> None:
        module, kjt, _ = _build_module_and_kjt(big=_SPARSE_REPLICATED)
        self._assert_no_seq_op_uses_op(module, kjt, "aten.embedding.default")

    def test_fsdp_fx_uses_fsdp_embedding(self) -> None:
        module, kjt, _ = _build_module_and_kjt(big=_SPARSE_FSDP)
        self._assert_no_seq_op_uses_op(module, kjt, "fsdp_embedding.default")

    def test_replicated_matches_reference_embedding(self) -> None:
        """REPLICATED big-table output equals a plain F.embedding of the same
        weight (shard_batch0 is identity on CPU)."""
        module, kjt, batch = _build_module_and_kjt(big=_SPARSE_REPLICATED)
        for _key, got, ref in _big_feature_ref(module, kjt, batch):
            torch.testing.assert_close(got, ref)

    def test_fsdp_matches_reference_embedding(self) -> None:
        """FSDP big-table output equals a plain F.embedding of the same weight
        (on CPU there is no real sharding; this checks the lookup semantics)."""
        module, kjt, batch = _build_module_and_kjt(big=_SPARSE_FSDP)
        for _key, got, ref in _big_feature_ref(module, kjt, batch):
            torch.testing.assert_close(got, ref)

    def test_fsdp_meta_init_traces(self) -> None:
        """Regression: an FSDP table with on_device_init=True (meta weight) must
        TRACE cleanly. Plain aten.embedding raised 'FakeTensor Device Propagation
        ... meta, cpu'; routing the lookup through
        recir.fsdp_embedding (meta-tolerant fake) fixes it. The weight stays meta
        (on-device init preserved for FSDP)."""
        module, kjt, _ = _build_module_and_kjt(big=_SPARSE_FSDP, on_device_init=True)
        self.assertEqual(module._table_sharding["big"], TableSharding.FSDP)
        self.assertTrue(
            module._sharded_tables["big"].is_meta, "FSDP weight should be meta"
        )
        keys = kjt.keys()
        values = kjt.values()
        lengths = kjt.lengths()

        def fn(values: torch.Tensor, lengths: torch.Tensor):
            k = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
            return module(k)

        # Must not raise the meta-vs-cpu FakeTensor device error.
        gm = make_fx(fn, pre_dispatch=True)(values, lengths)
        targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(
            any(t.endswith("fsdp_embedding.default") for t in targets),
            f"expected a fsdp_embedding node, got {targets}",
        )

    def test_fsdp_embedding_op_meta_tolerant(self) -> None:
        """fsdp_embedding places the activation on the ids (compute) device even
        with a meta weight (on-device init)."""
        import recir.ops.seq_embedding_op  # noqa: F401  (register the ops)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            ids = torch.zeros(16, dtype=torch.int64)  # cpu
            weight = torch.empty(256, 8, device="meta")  # meta (on-device init)
            out = torch.ops.recir.fsdp_embedding(ids, weight)
            self.assertEqual(str(out.device), "cpu", f"out on {out.device}")
            self.assertEqual(tuple(out.shape), (16, 8))
            self.assertEqual(out.dtype, weight.dtype)

    def test_fsdp_embedding_backward_grad_follows_weight_device(self) -> None:
        """fsdp_embedding_backward puts the table grad on the WEIGHT's device (meta
        here), so it matches the meta param in the optimizer's _foreach_add (the
        meta/cpu crash, jobs 78zf7/bxdb5). Mirrors ringshift_embedding_backward."""
        import recir.ops.seq_embedding_op  # noqa: F401  (register the ops)
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            grad_out = torch.zeros(16, 8)  # cpu
            ids = torch.zeros(16, dtype=torch.int64)  # cpu
            weight = torch.empty(256, 8, device="meta")  # meta (on-device init)
            grad = torch.ops.recir.fsdp_embedding_backward(grad_out, ids, weight)
            self.assertEqual(str(grad.device), "meta", f"grad on {grad.device}")
            self.assertEqual(tuple(grad.shape), (256, 8))

    def test_fsdp_embedding_autograd_matches_aten(self) -> None:
        """CPU autograd through recir.fsdp_embedding gives the same table grad as
        aten."""
        import recir.ops.seq_embedding_op  # noqa: F401  (register the ops)

        torch.manual_seed(0)
        weight = torch.randn(16, 4, requires_grad=True)
        ids = torch.randint(0, 16, (10,), dtype=torch.int64)
        torch.ops.recir.fsdp_embedding(ids, weight).sum().backward()
        ref = torch.ops.aten.embedding_dense_backward(
            torch.ones(10, 4), ids, 16, -1, False
        )
        torch.testing.assert_close(weight.grad, ref)


class TablePlansFromEmbeddingTablesTest(unittest.TestCase):
    """The config -> plan mapping, including everything it refuses to guess."""

    def test_row_wise_and_data_parallel_map_to_both_axes(self) -> None:
        plans = table_plans_from_embedding_tables(
            {
                "big": _FakeTable(ShardingType.ROW_WISE),
                "small": _FakeTable(ShardingType.DATA_PARALLEL),
            }
        )
        self.assertEqual(
            plans["big"], TablePlan(TableSharding.ROWWISE, TableOptimizer.SPARSE)
        )
        self.assertEqual(
            plans["small"],
            TablePlan(TableSharding.REPLICATED, TableOptimizer.DENSE),
        )

    def test_missing_sharding_raises(self) -> None:
        """None used to fall through to a row-count heuristic. There is nowhere
        left to fall through to, and guessing picks a learning rate."""
        with self.assertRaisesRegex(ValueError, "table 'big' has no sharding"):
            table_plans_from_embedding_tables({"big": _FakeTable(None)})

    def test_bare_string_sharding_raises(self) -> None:
        """The artifact round-tripped this enum by NAME once: non-None, unequal
        to every member, so every reader saw 'no placement' and said nothing."""
        with self.assertRaisesRegex(TypeError, "as a plain string"):
            table_plans_from_embedding_tables({"big": _FakeTable("ROW_WISE")})

    def test_unsupported_sharding_raises(self) -> None:
        """in_graph implements ROW_WISE and DATA_PARALLEL only; anything else has
        no lookup and must not be silently bucketed into one of them."""
        with self.assertRaisesRegex(ValueError, "no lookup for"):
            table_plans_from_embedding_tables(
                {"big": _FakeTable(ShardingType.TABLE_WISE)}
            )


if __name__ == "__main__":
    unittest.main()


class TableStorageDtypeTest(unittest.TestCase):
    """``EmbeddingConfig.data_type`` is storage precision, and it is honoured.

    This backend used to allocate every table float32 whatever the config said,
    while TorchRec allocated them narrow. That is not a layout detail: for a
    production table stored FP16, the representable step at a weight
    of 0.06 is 3.05e-5 and one optimizer step moves 2.98e-5, so the update does
    not fit and round-to-nearest drops or doubles it. The eager leg's largest
    single-step update on four such tables came back as exactly 2^-15
    (3.051757812500e-05) -- the grid, not a value -- against this backend's
    2.98e-5, and their weights diverged 20-30% after one step.

    Compute stays float32: only the stored weight is narrow.
    """

    @staticmethod
    def _module(data_type: DataType) -> InGraphSparseArch:
        return InGraphSparseArch(
            [
                EmbeddingConfig(
                    name="t",
                    num_embeddings=16,
                    embedding_dim=8,
                    feature_names=["f"],
                    data_type=data_type,
                )
            ],
            {"f": 2},
            world_size=2,
            table_plans={"t": _SPARSE_REPLICATED},
        )

    def test_fp16_is_stored_narrow(self) -> None:
        m = self._module(DataType.FP16)
        self.assertEqual(m._dp_tables["t"].dtype, torch.float16)

    def test_fp32_is_unchanged(self) -> None:
        m = self._module(DataType.FP32)
        self.assertEqual(m._dp_tables["t"].dtype, torch.float32)

    def test_the_lookup_comes_back_float32(self) -> None:
        """A narrow table must not push its dtype into HSTU and the loss --
        that would be a different model, not a different layout."""
        m = self._module(DataType.FP16)
        kjt = KeyedJaggedTensor(
            keys=["f"],
            values=torch.randint(0, 16, (4,), dtype=torch.int64),
            lengths=torch.tensor([2, 2], dtype=torch.int64),
        )
        out = m(kjt)
        self.assertEqual(out["f"].values().dtype, torch.float32)


class RowWiseKeepsFloat32Test(unittest.TestCase):
    """Narrow storage is NOT applied to row-wise tables.

    They go through the ring-shift Pallas kernel, which asserts a float32 table
    (ops/pallas/embeddings/ringshift.py:2028): a narrow one does not degrade,
    it takes the run down at the first lookup.

    The exemption costs nothing numerically. Row-wise is where the sparse
    optimizer lives, and its per-step update sits two orders of magnitude above
    the FP16 grid. The tables that need
    narrow storage are the REPLICATED ones on the dense optimizer, whose update
    lands at roughly one ULP.
    """

    @staticmethod
    def _table(sharding: TablePlan) -> InGraphSparseArch:
        return InGraphSparseArch(
            [
                EmbeddingConfig(
                    name="t",
                    num_embeddings=16,
                    embedding_dim=8,
                    feature_names=["f"],
                    data_type=DataType.FP16,
                )
            ],
            {"f": 2},
            world_size=2,
            table_plans={"t": sharding},
        )

    def test_row_wise_stays_float32(self) -> None:
        m = self._table(_SPARSE_ROWWISE)
        self.assertEqual(m._sharded_tables["t"].dtype, torch.float32)

    def test_replicated_still_goes_narrow(self) -> None:
        """The case the narrowing exists for must not be caught by the guard."""
        m = self._table(_SPARSE_REPLICATED)
        self.assertEqual(m._dp_tables["t"].dtype, torch.float16)

    def test_fsdp_still_goes_narrow(self) -> None:
        """FSDP uses recir.fsdp_embedding, not the ring-shift kernel."""
        m = self._table(_SPARSE_FSDP)
        self.assertEqual(m._sharded_tables["t"].dtype, torch.float16)
