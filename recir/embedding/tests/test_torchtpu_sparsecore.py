# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for recir TorchTPU SparseCore helpers."""

from __future__ import annotations

import dataclasses
import unittest
from typing import Any, Optional, Tuple
from unittest import mock

import torch

from recir.compiler.tracing import trace
from recir.distributed.parallel_mode import DataParallel
from torch import nn
from torch.distributed.tensor import DeviceMesh

_SPARSECORE_IMPORT_ERROR: Optional[Exception] = None
try:
    from recir.embedding.torchtpu_sparsecore import (
        _unwrap_dtensors,
        install_sparsecore_embeddings,
        TorchTPUSparseCoreBackedEC,
    )
    from torchrec.modules.embedding_configs import EmbeddingConfig
    from torchrec.modules.embedding_modules import EmbeddingCollection
except ModuleNotFoundError as exc:
    _SPARSECORE_IMPORT_ERROR = exc


@dataclasses.dataclass
class _FakeTableConfig:
    max_ids_per_partition: int = 256
    max_unique_ids_per_partition: int = 256


@dataclasses.dataclass
class _FakeStackConfig:
    stack_name: str
    feature_names: list[str]
    tables: list[_FakeTableConfig] = dataclasses.field(
        default_factory=lambda: [_FakeTableConfig()]
    )
    max_seq_len: int = 2
    total_features: int = 1
    max_ids_per_partition: int = 256
    max_unique_ids_per_partition: int = 256


class _FakeSCEC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stacked_configs = [_FakeStackConfig("s0", ["f0"])]
        sub = nn.Module()
        sub.register_parameter("weight", nn.Parameter(torch.ones(4, 8)))
        self.embeddings = nn.ModuleDict({"s0": sub})
        self.accumulators = nn.ParameterDict({"s0": nn.Parameter(torch.zeros(4, 1))})
        self.momentums = nn.ParameterDict()
        self.velocities = nn.ParameterDict()
        self.learning_rates: Any = nn.ParameterDict(
            {
                "s0": nn.Parameter(
                    torch.tensor(0.01, dtype=torch.float32), requires_grad=False
                )
            }
        )
        self._optimizer_type = torch.optim.Adagrad


class _InPlaceLookupOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight, idx)
        return weight[idx]

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        weight, idx = ctx.saved_tensors
        weight.index_add_(0, idx, -0.01 * grad_out)
        return None, None


class TorchTPUSparseCoreTest(unittest.TestCase):
    @unittest.skipIf(
        _SPARSECORE_IMPORT_ERROR is not None,
        f"SparseCore deps unavailable: {_SPARSECORE_IMPORT_ERROR}",
    )
    def test_install_validates_binding_and_geometry(self) -> None:
        class _OwnerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.owner = nn.Module()
                self.owner.add_module(
                    "collection",
                    EmbeddingCollection(
                        tables=[
                            EmbeddingConfig(
                                name="t0",
                                embedding_dim=8,
                                num_embeddings=16,
                                feature_names=["f0"],
                            )
                        ],
                        device=torch.device("meta"),
                    ),
                )
                self.tower = nn.Linear(4, 1)

        with self.assertRaisesRegex(TypeError, "owner is a"):
            install_sparsecore_embeddings(
                _OwnerModel(),
                "owner",
                table_configs=[],
                sparse_config=mock.Mock(),
                rows=4,
                feature_totals={"f0": 8},
                world_size=1,
                device=torch.device("cpu"),
            )

        with self.assertRaisesRegex(ValueError, "rows must be positive"):
            install_sparsecore_embeddings(
                _OwnerModel(),
                "owner.collection",
                table_configs=[],
                sparse_config=mock.Mock(),
                rows=0,
                feature_totals={},
                world_size=1,
                device=torch.device("cpu"),
            )

    @unittest.skipIf(
        _SPARSECORE_IMPORT_ERROR is not None,
        f"SparseCore deps unavailable: {_SPARSECORE_IMPORT_ERROR}",
    )
    def test_unwrap_dtensors_binding_and_fdo_calibration(self) -> None:
        fake_sc = _FakeSCEC()
        _unwrap_dtensors(fake_sc)
        self.assertEqual(fake_sc.learning_rates, {})
        self.assertNotIn("learning_rates", dict(fake_sc.named_modules()))
        self.assertFalse(
            any("learning_rates" in n for n, _ in fake_sc.named_parameters())
        )

        preprocessor = mock.Mock()
        preprocessor._fdo_headroom = 1.5
        preprocessor._allow_id_dropping = True
        preprocessor.return_value.stats = mock.Mock(table_stats={})

        def _fake_update(
            unused_stats: Any, *, scale_factor: float, bucket_step_up: bool
        ) -> bool:
            del unused_stats
            self.assertEqual(scale_factor, 1.5)
            self.assertTrue(bucket_step_up)
            fake_sc.stacked_configs[0].tables[0].max_ids_per_partition = 1024
            fake_sc.stacked_configs[0].tables[0].max_unique_ids_per_partition = 512
            return True

        preprocessor.update_preprocessing_parameters.side_effect = _fake_update

        backed = TorchTPUSparseCoreBackedEC(
            fake_sc,
            preprocessor=preprocessor,
            rows=4,
            feature_totals={"f0": 8},
            embedding_lr=0.05,
            device=torch.device("cpu"),
        )
        self.assertEqual(backed.num_step_tensors, 4)

        sc_args = tuple(torch.zeros(2) for _ in range(4))
        with backed.bind_step_inputs(sc_args):
            self.assertEqual(len(backed._bound_tensors or ()), 4)
        self.assertIsNone(backed._bound_tensors)

        backed.update_preprocessing_parameters(mock.Mock())
        self.assertEqual(fake_sc.stacked_configs[0].max_ids_per_partition, 1024)
        self.assertEqual(fake_sc.stacked_configs[0].max_unique_ids_per_partition, 512)
        self.assertFalse(preprocessor._allow_id_dropping)
        preprocessor._rebuild_backend.assert_called_once()

    def test_data_parallel_trace_skips_collectives_for_none_grad_params(
        self,
    ) -> None:
        class _MiniModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sc_weight = nn.Parameter(torch.ones(8, 4))
                self.fc = nn.Linear(4, 1)

            def forward(self, idx: torch.Tensor) -> torch.Tensor:
                return self.fc(_InPlaceLookupOp.apply(self.sc_weight, idx)).sum()

        model = _MiniModel()
        opt = torch.optim.SGD([model.fc.weight, model.fc.bias], lr=0.1)
        mesh = DeviceMesh("cpu", torch.arange(1), _init_backend=False, _rank=0)
        dp_mode = DataParallel(
            "replicate",
            num_ranks=1,
            num_processes=1,
            device_type="cpu",
        )
        dp_mode._device_mesh = mesh
        dp_mode._process_mesh = mesh

        def _step(
            mod: nn.Module, optimizer: torch.optim.Optimizer, idx: torch.Tensor
        ) -> torch.Tensor:
            loss = mod(idx)
            loss.backward()
            optimizer.step()
            return loss.detach()

        gm, flat_state = trace(
            _step, model, opt, torch.tensor([1, 2]), parallel_mode=dp_mode
        )
        self.assertGreaterEqual(len(flat_state), 3)
        self.assertIsNotNone(gm)


if __name__ == "__main__":
    unittest.main()
