# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""The SPARSECORE placement's graph wiring, against plain torch.

Traces a train step through InGraphSparseArch (one SPARSECORE table with two
features, one REPLICATED table), installs the dispatches with the real
SparseArchInputBreakDispatcher, runs the graph eagerly and after
``partial_lower_xla``, and compares loss, dense params and the SparseCore table
after one step with the same model in plain torch (row-wise Adagrad on the
SparseCore table, SGD on everything else). The SparseCore tables are a numpy
stand-in that keeps the real gradient collection
(``SparseCoreRowwiseTables.backward``).
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest import mock

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from functorch import make_fx
from jax.sharding import NamedSharding, PartitionSpec as P
from recir.backends.types import BackendType
from recir.compiler.tracing import _rematerialize_optimizer
from recir.embedding.backend import (
    SparseUpdatePlacement,
    TableOptimizer,
    TablePlan,
    TableSharding,
)
from recir.embedding.in_graph import (
    InGraphEmbeddingBackend,
    InGraphSparseArch,
    SparseArchInputBreakDispatcher,
)
from recir.embedding.sparsecore_rowwise import SparseCoreRowwiseTables
from recir.lowering.xla_lower import _gspmd_mesh, _XLAModule, partial_lower_xla
from recir.training.optim import warmup_lr
from recir.utils.compile import _enable_compile
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn.utils import stateless
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.warmup import WarmupPolicy, WarmupStage
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

_LR, _EPS, _D, _CHIPS = 0.05, 0.001, 4, 8
_KEYS = ["f0", "f1", "f2"]
_SEQ = {"f0": 3, "f1": 2, "f2": 1}


class _NumpySparseCore(SparseCoreRowwiseTables):
    """Row-wise Adagrad in numpy; ``backward`` is the real gradient collection."""

    def __init__(
        self, weights: Dict[str, np.ndarray], feature_table: Dict[str, str]
    ) -> None:
        self.w = {t: w.astype(np.float64) for t, w in weights.items()}
        self.m = {t: np.zeros(w.shape[0]) for t, w in weights.items()}
        self._feature_table = feature_table
        self._pending, self._step_inputs, self._expected = {}, None, None
        self.applied: List[List[int]] = []
        self.grad_types: set = set()
        self.grad_replicated: set = set()
        self.jax_sharding: Any = None
        self._host_ids: Dict[str, np.ndarray] = {}

    def lookup(self, host_ids: Dict[str, np.ndarray]) -> List[Any]:
        self._step_inputs = ({f: (a.size, a.size) for f, a in host_ids.items()},)
        self._host_ids = host_ids
        self._pending = {}
        rows = [
            self.w[self._feature_table[f]][a].astype(np.float32)
            for f, a in host_ids.items()
        ]
        if self.jax_sharding is None:
            return [torch.from_numpy(r) for r in rows]
        return [jax.device_put(r, self.jax_sharding) for r in rows]

    def _apply(self, pending: Dict[int, Any]) -> None:
        self.applied.append(sorted(pending))
        self.grad_types |= {type(g).__name__ for g in pending.values()}
        self.grad_replicated |= {
            bool(getattr(g, "is_fully_replicated", False)) for g in pending.values()
        }
        for t, w in self.w.items():
            g = np.zeros_like(w)
            for i, (f, ids) in enumerate(self._host_ids.items()):
                if self._feature_table[f] == t and i in pending:
                    np.add.at(g, ids, np.asarray(pending[i], np.float64))
            rows = np.flatnonzero(np.any(g != 0, axis=1))
            self.m[t][rows] += np.mean(g[rows] ** 2, axis=1)
            w[rows] -= _LR * g[rows] / (np.sqrt(self.m[t][rows])[:, None] + _EPS)


class _Wrapper(nn.Module):
    def __init__(self, arch: InGraphSparseArch, used: List[str]) -> None:
        super().__init__()
        self.arch, self.lin, self.used = arch, nn.Linear(_D, _CHIPS), used

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out = self.arch(KeyedJaggedTensor(keys=_KEYS, values=values, lengths=lengths))
        return torch.stack(
            [self.lin(out[k].values()).sum() * (i + 1) for i, k in enumerate(self.used)]
        ).sum()


def _trace(
    wrapper: nn.Module,
    opt: torch.optim.Optimizer,
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[Any, List[Any]]:
    fake = FakeTensorMode()
    params = dict(wrapper.named_parameters(remove_duplicate=False))
    buffers = dict(wrapper.named_buffers(remove_duplicate=False))

    def fn(params, buffers, states, values, lengths):
        with (
            stateless._reparametrize_module(wrapper, {**params, **buffers}),
            _rematerialize_optimizer(opt, states, params),
        ):
            loss = wrapper(values, lengths)
            loss.backward()
            opt.step()
            return loss, params, buffers, states

    with fake, _enable_compile():
        gm = make_fx(fn, tracing_mode="fake")(
            params, buffers, {}, fake.from_tensor(values), fake.from_tensor(lengths)
        )
    return gm, pytree.tree_leaves([params, buffers, {}])


def _run(used: List[str], lower: bool) -> Dict[str, Any]:
    torch.manual_seed(0)
    tables = [
        EmbeddingConfig(
            num_embeddings=50,
            embedding_dim=_D,
            name="big",
            feature_names=["f0", "f1"],
            data_type=DataType.FP32,
        ),
        EmbeddingConfig(
            num_embeddings=16,
            embedding_dim=_D,
            name="rep",
            feature_names=["f2"],
            data_type=DataType.FP32,
        ),
    ]
    plans = {
        "big": TablePlan(TableSharding.SPARSECORE, TableOptimizer.SPARSE),
        "rep": TablePlan(TableSharding.REPLICATED, TableOptimizer.DENSE),
    }
    arch = InGraphSparseArch(
        tables, _SEQ, world_size=_CHIPS, table_plans=plans, on_device_init=False
    )
    w0 = arch._sparsecore_host_weights["big"][:50].numpy().copy()
    sparsecore = _NumpySparseCore({"big": w0}, {"f0": "big", "f1": "big"})
    arch.sparsecore = sparsecore
    wrapper = _Wrapper(arch, used)
    ref = {n: p.detach().clone() for n, p in wrapper.named_parameters()}
    # Citrine C2: foreach=True for the multi-tensor optimizer step
    opt = torch.optim.SGD(wrapper.parameters(), lr=_LR, foreach=True)
    rng = np.random.RandomState(1)
    ids = {
        "f0": rng.randint(0, 50, _CHIPS * 3),
        "f1": rng.randint(0, 50, _CHIPS * 2),
        "f2": rng.randint(0, 16, _CHIPS),
    }
    ids["f1"][0] = ids["f0"][0]  # one row shared by both features
    values = torch.from_numpy(np.concatenate([ids[k] for k in _KEYS])).long()
    lengths = torch.tensor([3] * _CHIPS + [2] * _CHIPS + [1] * _CHIPS)
    dispatcher = SparseArchInputBreakDispatcher(
        input_kjt_keys=_KEYS, override_id=1, num_ranks=1, feature_seqlens=_SEQ
    )
    dispatcher.install_eager(wrapper)
    gm, flat_state = _trace(wrapper, opt, values, lengths)
    gm, _ = dispatcher.install_graph(gm, flat_state)
    if lower:
        mesh, axis, _ = _gspmd_mesh(jax.devices(), len(jax.devices()))
        sparsecore.jax_sharding = NamedSharding(mesh, P(axis, None))
        gm.meta["gspmd_fsdp_axis_size"] = len(jax.devices())
        gm = partial_lower_xla(gm, backend_type=BackendType.XLA_GSPMD)
        for m in gm.modules():
            if isinstance(m, _XLAModule):
                m.keep_jax_output = True
    with torch.no_grad():
        loss, new_params, _, _ = gm(
            dict(wrapper.named_parameters()),
            dict(wrapper.named_buffers()),
            {},
            values,
            lengths,
        )
    new_params = {
        n: v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v))
        for n, v in new_params.items()
    }

    big = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    p = {n: t.double().requires_grad_() for n, t in ref.items()}
    acts = {
        "f0": big[torch.from_numpy(ids["f0"])],
        "f1": big[torch.from_numpy(ids["f1"])],
        "f2": p["arch._dp_tables.rep"][torch.from_numpy(ids["f2"])],
    }
    ref_loss = torch.stack(
        [
            (acts[k] @ p["lin.weight"].T + p["lin.bias"]).sum() * (i + 1)
            for i, k in enumerate(used)
        ]
    ).sum()
    ref_loss.backward()
    assert big.grad is not None
    g = big.grad.numpy()
    ref_params = {n: t if t.grad is None else t - _LR * t.grad for n, t in p.items()}
    rows = np.flatnonzero(np.any(g != 0, axis=1))
    m = np.zeros(50)
    m[rows] += np.mean(g[rows] ** 2, axis=1)
    ref_big = w0.astype(np.float64)
    ref_big[rows] -= _LR * g[rows] / (np.sqrt(m[rows])[:, None] + _EPS)
    return {
        "loss_err": abs(float(np.asarray(loss)) - float(ref_loss)),
        "dense_err": max(
            float((new_params[n].double() - ref_params[n]).abs().max()) for n in p
        ),
        "table_err": float(np.abs(sparsecore.w["big"] - ref_big).max()),
        "applied": sparsecore.applied,
        "grad_types": sparsecore.grad_types,
        "grad_replicated": sparsecore.grad_replicated,
    }


class SparseCoreInGraphTest(unittest.TestCase):
    def _check(self, out: Dict[str, Any], applied: List[List[int]]) -> None:
        self.assertLess(out["loss_err"], 1e-4)
        self.assertLess(out["dense_err"], 1e-5)
        self.assertLess(out["table_err"], 1e-6)
        self.assertEqual(applied, out["applied"])

    def test_eager_step_matches_torch(self) -> None:
        self._check(_run(["f0", "f1", "f2"], lower=False), [[0, 1]])

    def test_eager_unused_feature_counts_as_zero_gradient(self) -> None:
        self._check(_run(["f0", "f2"], lower=False), [[0]])

    def test_lowered_step_matches_torch_and_keeps_gradients_on_device(self) -> None:
        out = _run(["f0", "f1", "f2"], lower=True)
        self._check(out, [[0, 1]])
        self.assertEqual({"ArrayImpl"}, out["grad_types"])
        self.assertEqual({False}, out["grad_replicated"])

    def test_lowered_unused_feature_counts_as_zero_gradient(self) -> None:
        out = _run(["f0", "f2"], lower=True)
        self._check(out, [[0]])
        self.assertEqual({"ArrayImpl"}, out["grad_types"])

    def test_fp16_table_keeps_fp16_values(self) -> None:
        arch = InGraphSparseArch(
            [
                EmbeddingConfig(
                    num_embeddings=50,
                    embedding_dim=_D,
                    name="big",
                    feature_names=["f0"],
                    data_type=DataType.FP16,
                )
            ],
            {"f0": 3},
            world_size=_CHIPS,
            table_plans={
                "big": TablePlan(TableSharding.SPARSECORE, TableOptimizer.SPARSE)
            },
            on_device_init=False,
        )

        self.assertTrue(arch._sparsecore_tables[0].fp16)
        self.assertEqual(torch.float16, arch._sparsecore_host_weights["big"].dtype)

    @staticmethod
    def _setup_all_on_sparsecore(schedule: Any) -> Tuple[nn.Module, Any, Any]:
        tables = [
            EmbeddingConfig(
                num_embeddings=50,
                embedding_dim=_D,
                name="big",
                feature_names=["f0"],
                data_type=DataType.FP32,
            )
        ]
        model = nn.Module()
        model.collection = EmbeddingCollection(tables=tables)

        def builder(*args: Any) -> Any:
            raise AssertionError("no table is left for the sparse optimizer")

        backend = InGraphEmbeddingBackend(
            "collection",
            builder,
            {"big": TablePlan(TableSharding.ROWWISE, TableOptimizer.SPARSE)},
            on_device_init=False,
        )
        backend.set_feature_seqlens({"f0": 3})
        with (
            mock.patch.dict(os.environ, {"RECIR_SPARSECORE_TABLES": "all"}),
            mock.patch.object(InGraphSparseArch, "init_sparsecore") as init,
        ):
            setup = backend.setup(
                model,
                tables,
                # pyrefly: ignore[bad-argument-type]
                SimpleNamespace(lr=_LR, eps=_EPS, lr_warmup_schedule=schedule),
                torch.device("cpu"),
                _CHIPS,
            )
        return model, setup.sparse_runtime, init.call_args.kwargs["lr"]

    def test_all_sparse_tables_on_sparsecore_need_no_sparse_optimizer(self) -> None:
        model, runtime, _ = self._setup_all_on_sparsecore(None)

        self.assertEqual(SparseUpdatePlacement.BACKEND_FUSED, runtime.placement)
        self.assertIsNone(runtime.optimizer)
        self.assertIs(model.get_buffer("_lr_sparse_buffer"), runtime.lr_buffer)

    def test_sparsecore_lr_follows_the_warmup_from_the_restored_iteration(
        self,
    ) -> None:
        schedule = [WarmupStage(policy=WarmupPolicy.LINEAR, max_iters=10, value=0.1)]
        model, _, lr = self._setup_all_on_sparsecore(schedule)
        cfg = SimpleNamespace(lr=_LR, lr_warmup_schedule=schedule)
        model.register_buffer("_lr_iter_buffer", torch.tensor(5), persistent=False)

        rates = [lr(), lr()]
        model.get_buffer("_lr_iter_buffer").fill_(100)
        rates.append(lr())

        # pyrefly: ignore[bad-argument-type]
        expected = [warmup_lr(cfg, i) for i in (5, 6, 7)]
        for got, want in zip(rates, expected):
            self.assertAlmostEqual(want, got)

    def test_all_selects_unpooled_rowwise_tables_on_the_sparse_optimizer(
        self,
    ) -> None:
        rowwise_sparse = TablePlan(TableSharding.ROWWISE, TableOptimizer.SPARSE)
        tables = [
            EmbeddingConfig(
                num_embeddings=50, embedding_dim=_D, name="a", feature_names=["fa"]
            ),
            EmbeddingConfig(
                num_embeddings=50, embedding_dim=_D, name="b", feature_names=["fb"]
            ),
            EmbeddingConfig(
                num_embeddings=16, embedding_dim=_D, name="rep", feature_names=["fr"]
            ),
            EmbeddingBagConfig(
                num_embeddings=50, embedding_dim=_D, name="pooled", feature_names=["fp"]
            ),
        ]
        plans = {
            "a": rowwise_sparse,
            "b": rowwise_sparse,
            "rep": TablePlan(TableSharding.REPLICATED, TableOptimizer.DENSE),
            "pooled": rowwise_sparse,
        }
        backend = InGraphEmbeddingBackend(
            "collection", mock.Mock(), plans, on_device_init=False
        )

        with mock.patch.dict(os.environ, {"RECIR_SPARSECORE_TABLES": "all"}):
            backend._plan_sparsecore(tables)

        sparsecore = TablePlan(TableSharding.SPARSECORE, TableOptimizer.SPARSE)
        self.assertEqual(
            {
                "a": sparsecore,
                "b": sparsecore,
                "rep": plans["rep"],
                "pooled": rowwise_sparse,
            },
            backend._table_plans,
        )
