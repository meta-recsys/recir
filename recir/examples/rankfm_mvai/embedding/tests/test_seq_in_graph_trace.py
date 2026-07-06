# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""CPU FX-trace check for InGraphSparseArch.

Answers (without TPU): does the in-graph module trace cleanly -- i.e. does the
KeyedJaggedTensor flow through, and does the lookup appear as the expected node
per table MODE -- ROWWISE -> opaque ``torch.ops.recir.ringshift_embedding`` (lowered
via the ringshift path); FSDP / DP -> ``embedding`` (lowered via F.embedding).
"""

from __future__ import annotations

import unittest

import torch
from recir.examples.rankfm_mvai.embedding.in_graph_embedding_backend import (
    DP_TABLE_PREFIX,
    InGraphSparseArch,
    SHARDED_TABLE_PREFIX,
    TableMode,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _build_module_and_kjt(
    ringshift_min_rows: int = 128,
    fsdp_min_rows: int | None = None,
    on_device_init: bool = False,
):
    """Build a 2-table module + a matching KJT.

    big table = 256 rows (f0,f1, S=4), small table = 16 rows (c0, S=1). The two
    thresholds select the big table's mode: rows>=ringshift_min_rows -> ROWWISE;
    fsdp_min_rows<=rows<ringshift_min_rows -> FSDP; else DP.
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
        ringshift_min_rows=ringshift_min_rows,
        on_device_init=on_device_init,
        fsdp_min_rows=fsdp_min_rows,
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

    # ---- table-mode routing (ROWWISE / FSDP / DP) ----

    def test_default_modes(self) -> None:
        """Default thresholds: big(256) >= 128 -> ROWWISE; small(16) -> DP."""
        module, _, _ = _build_module_and_kjt()
        self.assertEqual(module._table_mode["big"], TableMode.ROWWISE)
        self.assertEqual(module._table_mode["small"], TableMode.DP)
        self.assertIn("big", module._sharded_tables)
        self.assertIn("small", module._dp_tables)
        names = [n for n, _ in module.named_parameters()]
        self.assertTrue(any(SHARDED_TABLE_PREFIX in n and "big" in n for n in names))
        self.assertTrue(any(DP_TABLE_PREFIX in n and "small" in n for n in names))

    def test_dp_mode_routes_big(self) -> None:
        """rows < fsdp_min -> DP: big table replicated + F.embedding, NOT sharded
        (DP prefix, still embedding-lr; FSDP filter leaves it alone)."""
        module, _, _ = _build_module_and_kjt(ringshift_min_rows=512, fsdp_min_rows=512)
        self.assertEqual(module._table_mode["big"], TableMode.DP)
        self.assertIn("big", module._dp_tables)
        self.assertNotIn("big", module._sharded_tables)
        names = [n for n, _ in module.named_parameters()]
        self.assertTrue(any(DP_TABLE_PREFIX in n and "big" in n for n in names))
        self.assertFalse(any(SHARDED_TABLE_PREFIX in n and "big" in n for n in names))

    def test_fsdp_mode_routes_big(self) -> None:
        """fsdp_min <= rows < rowwise_min -> FSDP: big table dim-0 sharded
        (SHARDED_TABLE_PREFIX, so the FSDP filter shards it) but looked up via
        F.embedding (not the ring-shift op)."""
        module, _, _ = _build_module_and_kjt(ringshift_min_rows=512, fsdp_min_rows=128)
        self.assertEqual(module._table_mode["big"], TableMode.FSDP)
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
            f"FSDP/DP path must emit no ringshift_embedding: {targets}",
        )
        # big table feature(s) -> the expected lookup op (no ring-shift).
        self.assertGreaterEqual(
            len(big_nodes), 1, f"expected a {op_suffix} node, got {targets}"
        )

    def test_dp_fx_uses_aten_embedding(self) -> None:
        module, kjt, _ = _build_module_and_kjt(
            ringshift_min_rows=512, fsdp_min_rows=512
        )
        self._assert_no_seq_op_uses_op(module, kjt, "aten.embedding.default")

    def test_fsdp_fx_uses_fsdp_embedding(self) -> None:
        module, kjt, _ = _build_module_and_kjt(
            ringshift_min_rows=512, fsdp_min_rows=128
        )
        self._assert_no_seq_op_uses_op(module, kjt, "fsdp_embedding.default")

    def test_dp_matches_reference_embedding(self) -> None:
        """DP big-table output equals a plain F.embedding of the same weight
        (shard_batch0 is identity on CPU)."""
        module, kjt, batch = _build_module_and_kjt(
            ringshift_min_rows=512, fsdp_min_rows=512
        )
        for _key, got, ref in _big_feature_ref(module, kjt, batch):
            torch.testing.assert_close(got, ref)

    def test_fsdp_matches_reference_embedding(self) -> None:
        """FSDP big-table output equals a plain F.embedding of the same weight
        (on CPU there is no real sharding; this checks the lookup semantics)."""
        module, kjt, batch = _build_module_and_kjt(
            ringshift_min_rows=512, fsdp_min_rows=128
        )
        for _key, got, ref in _big_feature_ref(module, kjt, batch):
            torch.testing.assert_close(got, ref)

    def test_fsdp_meta_init_traces(self) -> None:
        """Regression: an FSDP table with on_device_init=True (meta weight) must
        TRACE cleanly. Plain aten.embedding raised 'FakeTensor Device Propagation
        ... meta, cpu' (job rankfm-64chip-bench-78zf7); routing the lookup through
        recir.fsdp_embedding (meta-tolerant fake) fixes it. The weight stays meta
        (on-device init preserved for FSDP)."""
        module, kjt, _ = _build_module_and_kjt(
            ringshift_min_rows=512, fsdp_min_rows=128, on_device_init=True
        )
        self.assertEqual(module._table_mode["big"], TableMode.FSDP)
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


if __name__ == "__main__":
    unittest.main()
