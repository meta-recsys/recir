# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""SparseCore sequence embeddings for recir via torchrec.experimental.torch_tpu."""

from __future__ import annotations

import contextlib
import enum
import importlib
import importlib.util
import logging
import math
import pathlib
import sys
import types
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import torch
import torch.distributed as dist
import torch.distributed.tensor as dt
from recir.embedding.backend import (
    EmbeddingBackend,
    EmbeddingSetupResult,
    SparseOptimizerRuntime,
    SparseUpdatePlacement,
    check_binding,
    swap_submodule,
)
from recir.training.optim import OptimConfig
from recir.utils.meta_tensor_init import fill_table_shard, table_init_bounds
from torch import nn
from torchrec.distributed.embedding_types import OptimType
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    DataType,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.keyed import KeyedOptimizer
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

NUM_SC: int = 2
_INIT_BLOCK_ROWS: int = 1 << 16

_OPTIMIZER_MAP: Dict[OptimType, Type[torch.optim.Optimizer]] = {
    OptimType.ROWWISE_ADAGRAD: RowWiseAdagrad,
    OptimType.ADAGRAD: torch.optim.Adagrad,
    OptimType.ADAM: torch.optim.Adam,
    OptimType.PARTIAL_ROWWISE_ADAM: torch.optim.Adam,
    OptimType.SGD: torch.optim.SGD,
}


def _round_up(x: int, m: int) -> int:
    return x + (-x) % m


def _resolve_optimizer_type(optim_type: OptimType) -> Type[torch.optim.Optimizer]:
    opt_cls = _OPTIMIZER_MAP.get(optim_type)
    if opt_cls is None:
        raise NotImplementedError(
            f"Unsupported SparseCore optimizer {optim_type!r}; "
            f"supported: {sorted(k.value for k in _OPTIMIZER_MAP)}"
        )
    return opt_cls


def _validate_table_configs(table_configs: Sequence[BaseEmbeddingConfig]) -> None:
    """Validate table configs against SparseCore layout, dtype, and init constraints."""
    for cfg in table_configs:
        if not isinstance(cfg, EmbeddingConfig):
            raise ValueError(
                f"SparseCore sequence backend expects EmbeddingConfig, got {cfg.name}"
            )
        if cfg.embedding_dim % 8 != 0:
            raise ValueError(
                f"{cfg.name}: embedding_dim ({cfg.embedding_dim}) must divide by 8"
            )
        if cfg.num_embeddings > 2**31:
            raise ValueError(
                f"{cfg.name}: {cfg.num_embeddings} rows, but SparseCore ids are int32"
            )
        if cfg.data_type != DataType.FP32:
            raise ValueError(
                f"{cfg.name}: SparseCore tables train in FP32, not {cfg.data_type}"
            )
        table_init_bounds(cfg)


class _FallbackMinibatchingMode(enum.IntEnum):
    DISABLED = 0
    HOST = 1
    DEVICE = 2


def _compute_coo_buffer_size_per_device(
    max_ids_per_partition: int,
    suggested_size: int,
    global_device_count: int,
    num_sc_per_device: int,
    hbm_word_size_in_4b: int,
    enable_minibatching: bool,
) -> int:
    num_scs = global_device_count * num_sc_per_device
    max_ids_rounded = _round_up(max(1, max_ids_per_partition), hbm_word_size_in_4b)
    buckets = 64 if enable_minibatching else 1
    theoretical_max = max_ids_rounded * num_sc_per_device * num_scs * buckets
    if suggested_size <= 0:
        return theoretical_max
    alignment = hbm_word_size_in_4b * num_sc_per_device
    aligned = _round_up(suggested_size, alignment)
    if aligned > theoretical_max:
        raise ValueError(
            f"suggested_coo_buffer_size_per_device ({aligned}) exceeds "
            f"theoretical_max ({theoretical_max})"
        )
    return aligned


class _TorchTpuOpPreprocessorBackend:
    """SparseCorePreprocessorBackend fallback using torch.ops.tpu."""

    def __init__(
        self,
        tables_metadata: Sequence[Mapping[str, Any]],
        batch_size: int,
        local_device_count: int,
        global_device_count: int,
        num_sc_per_device: int,
        sc_simd_width: int,
        hbm_word_size_in_4b: int,
        allow_id_dropping: bool,
        minibatching_mode: int = 0,
    ) -> None:
        del batch_size
        self._tables_metadata = [dict(t) for t in tables_metadata]
        self._local_device_count = int(local_device_count)
        self._global_device_count = int(global_device_count)
        self._num_sc_per_device = int(num_sc_per_device)
        self._sc_simd_width = int(sc_simd_width)
        self._hbm_word_size_in_4b = int(hbm_word_size_in_4b)
        self._allow_id_dropping = bool(allow_id_dropping)
        self._enable_minibatching = int(minibatching_mode) != 0
        self._buffer_sizes: Dict[str, int] = {
            str(t["name"]): _compute_coo_buffer_size_per_device(
                int(t["max_ids_per_partition"]),
                int(t.get("suggested_coo_buffer_size_per_device") or 0),
                self._global_device_count,
                self._num_sc_per_device,
                self._hbm_word_size_in_4b,
                self._enable_minibatching,
            )
            for t in self._tables_metadata
        }

    def preprocess(
        self,
        input_indices: Mapping[str, Sequence[torch.Tensor]],
        input_offsets: Mapping[str, Sequence[torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        op = torch.ops.tpu.preprocess_sparse_dense_matmul_input
        schema_names = [a.name for a in op.default._schema.arguments]
        num_scs = self._global_device_count * self._num_sc_per_device
        outputs: Dict[str, Dict[str, torch.Tensor]] = {}

        for table_meta in self._tables_metadata:
            name = str(table_meta["name"])
            features = list(table_meta["features"])
            idx_list = input_indices[name]
            off_list = input_offsets[name]
            remapped: List[torch.Tensor] = []
            for feat_meta, idx_t in zip(features, idx_list):
                col_offset = int(feat_meta["col_offset"])
                col_shift = int(feat_meta["col_shift"])
                if col_offset == 0 and col_shift == 0:
                    remapped.append(idx_t.to(torch.int32).contiguous())
                else:
                    idx64 = idx_t.to(torch.int64)
                    mapped = (
                        (idx64 // num_scs) * num_scs
                        + (idx64 + col_shift) % num_scs
                        + col_offset
                    )
                    remapped.append(mapped.to(torch.int32).contiguous())

            feat_batch = int(features[0]["batch_size"])
            combiner = str(features[0]["combiner"])
            if len(features) == 1:
                comb_idx = remapped[0]
                comb_off = off_list[0].to(torch.int32).contiguous()
                total_batch = feat_batch
            else:
                per_sc = feat_batch // self._num_sc_per_device
                val_chunks: List[torch.Tensor] = []
                len_chunks: List[torch.Tensor] = []
                for c in range(self._num_sc_per_device):
                    lo = c * per_sc
                    hi = (c + 1) * per_sc
                    for r_idx, r_off in zip(remapped, off_list):
                        off_slice = r_off[lo : hi + 1].to(torch.int64)
                        s = int(off_slice[0].item())
                        e = int(off_slice[-1].item())
                        val_chunks.append(r_idx[s:e])
                        len_chunks.append(off_slice[1:] - off_slice[:-1])
                comb_idx = torch.cat(val_chunks, dim=0).to(torch.int32).contiguous()
                all_lens = torch.cat(len_chunks, dim=0)
                comb_off = torch.zeros(all_lens.numel() + 1, dtype=torch.int32)
                comb_off[1:] = all_lens.cumsum(0).to(torch.int32)
                total_batch = feat_batch * len(features)

            arg_pool: Dict[str, Any] = {
                "input_indices": comb_idx,
                "input_offsets": comb_off,
                "max_ids_per_partition": int(table_meta["max_ids_per_partition"]),
                "max_unique_ids_per_partition": int(
                    table_meta["max_unique_ids_per_partition"]
                ),
                "suggested_coo_buffer_size": self._buffer_sizes[name],
                "batch_size": total_batch,
                "combiner": combiner,
                "local_device_count": self._local_device_count,
                "global_device_count": self._global_device_count,
                "num_sc_per_device": self._num_sc_per_device,
                "sc_simd_width": self._sc_simd_width,
                "hbm_word_size_in_4b": self._hbm_word_size_in_4b,
                "allow_id_dropping": self._allow_id_dropping,
                "table_name": name,
                "enable_minibatching": self._enable_minibatching,
            }
            res = op(*[arg_pool[arg_name] for arg_name in schema_names])
            dropped = (
                int(res["dropped_id_count"].sum().item())
                if "dropped_id_count" in res
                else 0
            )
            max_ids_stat = res.get("max_ids_per_partition")
            obs_max_ids = (
                int(max_ids_stat.max().item())
                if max_ids_stat is not None and max_ids_stat.numel() > 0
                else 0
            )
            max_uniq_stat = res.get("max_unique_ids_per_partition")
            obs_max_uniq = (
                int(max_uniq_stat.max().item())
                if max_uniq_stat is not None and max_uniq_stat.numel() > 0
                else 0
            )
            outputs[name] = {
                "row_pointers": res["row_pointers"],
                "embedding_ids": res["embedding_ids"],
                "sample_ids": res["sample_ids"],
                "gains": res["gains"],
                "dropped_count": torch.tensor(dropped, dtype=torch.int32),
                "observed_max_ids": torch.tensor(obs_max_ids, dtype=torch.int32),
                "observed_max_unique_ids": torch.tensor(
                    obs_max_uniq, dtype=torch.int32
                ),
            }
        return outputs


def _ensure_oss_torchrec_tpu_shims() -> None:
    """Install etils and pybind_input_preprocessing shims for OSS pip."""
    if "etils" not in sys.modules and importlib.util.find_spec("etils") is None:
        etils_mod = types.ModuleType("etils")
        epath_mod = types.ModuleType("etils.epath")
        setattr(epath_mod, "Path", pathlib.Path)
        setattr(etils_mod, "epath", epath_mod)
        sys.modules["etils"] = etils_mod
        sys.modules["etils.epath"] = epath_mod

    pybind_fqn = "torchrec.experimental.torch_tpu.datasets.pybind_input_preprocessing"
    if pybind_fqn not in sys.modules:
        try:
            spec = importlib.util.find_spec(pybind_fqn)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            shim = types.ModuleType(pybind_fqn)
            setattr(shim, "MinibatchingMode", _FallbackMinibatchingMode)
            setattr(
                shim,
                "SparseCorePreprocessorBackend",
                _TorchTpuOpPreprocessorBackend,
            )
            sys.modules[pybind_fqn] = shim
            datasets_pkg = sys.modules.get("torchrec.experimental.torch_tpu.datasets")
            if datasets_pkg is not None:
                setattr(datasets_pkg, "pybind_input_preprocessing", shim)


def _load_torchrec_tpu() -> Tuple[Any, ...]:
    """Lazily import torchrec.experimental.torch_tpu (not a static Buck dependency)."""
    _ensure_oss_torchrec_tpu_shims()
    fdo_mod = importlib.import_module(
        "torchrec.experimental.torch_tpu.datasets.fdo.fdo_client"
    )
    inp_mod = importlib.import_module(
        "torchrec.experimental.torch_tpu.datasets.input_preprocessing"
    )
    sc_utils = importlib.import_module(
        "torchrec.experimental.torch_tpu.modules.utils"
    )
    cfg_mod = importlib.import_module(
        "torchrec.experimental.torch_tpu.modules.embedding_configs"
    )
    fused_mod = importlib.import_module(
        "torchrec.experimental.torch_tpu.modules.fused_embedding_modules"
    )
    return (
        fdo_mod.KeyedSparseCoreInputStats,
        fdo_mod.SparseCoreInputStats,
        inp_mod.KeyedSparseCorePreprocessedInput,
        inp_mod.MinibatchingMode,
        inp_mod.SparseCoreInputPreprocessor,
        inp_mod.SparseCorePreprocessedInput,
        sc_utils,
        cfg_mod.SparseCoreEmbeddingConfig,
        fused_mod.SparseCoreFusedEmbeddingCollection,
        fused_mod.SparseCoreFusedOptimizer,
    )


def _seed_local_stack_shard(
    stack_config: Any,
    owner: int,
    world_size: int,
    num_sc_per_device: int = NUM_SC,
) -> torch.Tensor:
    """Build rank ``owner``'s stacked & MOD-sharded weight shard in blocks."""
    num_shards = world_size * num_sc_per_device
    rows_per_sc = stack_config.stack_num_embeddings // num_shards
    stack_dim = stack_config.stack_embedding_dim
    local_3d = torch.zeros(
        (num_sc_per_device, rows_per_sc, stack_dim), dtype=torch.float32
    )

    for table in stack_config.tables:
        cfg = getattr(table, "config", table)
        num_embeddings = cfg.num_embeddings
        dim = cfg.embedding_dim
        row_offset = stack_config.row_offsets_in_shard[table.name]
        rot = stack_config.shard_rotations[table.name] % num_shards
        table_shape = (num_embeddings, dim)
        low, high = table_init_bounds(cfg)

        for start in range(0, num_embeddings, _INIT_BLOCK_ROWS):
            end = min(start + _INIT_BLOCK_ROWS, num_embeddings)
            block = torch.empty((end - start, dim), dtype=torch.float32)
            fill_table_shard(
                block,
                cfg.name,
                table_shape,
                (start, 0),
                low,
                high,
            )
            for c in range(num_sc_per_device):
                phys_sc = owner * num_sc_per_device + c
                logical_mod = (phys_sc - rot) % num_shards
                first = start + (logical_mod - start) % num_shards
                if first < end:
                    picked = block[first - start :: num_shards]
                    at = row_offset + first // num_shards
                    local_3d[c, at : at + picked.shape[0], :dim] = picked

    return local_3d.reshape(-1, stack_dim)


def _unwrap_dtensors(sc_ec: nn.Module) -> None:
    """Unwrap SparseCore DTensor shards for DataParallel make_fx tracing."""
    for mod in sc_ec.modules():
        for name, param in list(mod._parameters.items()):
            if isinstance(param, dt.DTensor) or hasattr(param, "to_local"):
                local = nn.Parameter(
                    param.to_local(), requires_grad=param.requires_grad
                )
                local.__dict__.update(param.__dict__)
                mod._parameters[name] = local
    if isinstance(sc_ec.learning_rates, nn.Module):
        delattr(sc_ec, "learning_rates")
    setattr(sc_ec, "learning_rates", {})


def _seed_stacked_shards(
    sc_ec: Any,
    *,
    owner: int,
    world_size: int,
    num_sc_per_device: int,
    device: torch.device,
    sc_utils: Any,
    fused_opt_cls: Any,
) -> None:
    """Initialize unwrapped SparseCore stacked weights from RecIR's seeded init."""
    for stack_config in sc_ec.stacked_configs:
        local_cpu = _seed_local_stack_shard(
            stack_config, owner, world_size, num_sc_per_device
        )
        tpu_w = sc_utils.to_device_in_sparsecore_layout(local_cpu, device)
        sub = sc_ec.embeddings[stack_config.stack_name]
        old_w = sub.weight
        new_w = nn.Parameter(tpu_w, requires_grad=old_w.requires_grad)
        new_w.__dict__.update(old_w.__dict__)
        sub.register_parameter("weight", new_w)
    sc_ec._fused_optimizer = fused_opt_cls(sc_ec)


class TorchTPUSparseCoreBackedEC(nn.Module):
    """EmbeddingCollection backed by SparseCoreFusedEmbeddingCollection."""

    def __init__(
        self,
        sc_ec: Any,
        preprocessor: Any,
        *,
        rows: int,
        feature_totals: Mapping[str, int],
        embedding_lr: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.sc_ec = sc_ec
        self.preprocessor = preprocessor
        self._rows = rows
        self._feature_totals: Dict[str, int] = dict(feature_totals)
        self._bound_tensors: Optional[Sequence[torch.Tensor]] = None
        self._calibrated: bool = False
        self._device = device
        self.register_buffer(
            "lr_buffer",
            torch.tensor(embedding_lr, dtype=torch.float32, device=device),
            persistent=False,
        )
        self._anchor = torch.zeros((), requires_grad=True)
        self.optimizer = KeyedOptimizer(
            params={"lr_anchor": self._anchor},
            state={},
            param_groups=[{"params": [self._anchor], "lr": embedding_lr}],
        )

    @property
    def num_step_tensors(self) -> int:
        return 4 * len(self.sc_ec.stacked_configs)

    def update_preprocessing_parameters(
        self, features: KeyedJaggedTensor
    ) -> None:
        """Update per-partition SparseCore capacities from a batch via FDO."""
        stats = self.preprocessor(features).stats
        assert stats is not None
        if dist.is_initialized() and dist.get_world_size() > 1:
            KeyedSparseCoreInputStats, SparseCoreInputStats, *_ = _load_torchrec_tpu()
            names = list(stats.table_stats.keys())
            try:
                pg_backend = str(dist.get_backend())
            except Exception:
                pg_backend = "cpu"
            red_dev = (
                self.lr_buffer.device
                if pg_backend in ("tpu", "tpu_dist")
                else torch.device("cpu")
            )
            counts = torch.tensor(
                [
                    [
                        stats.table_stats[n].dropped_count,
                        stats.table_stats[n].observed_max_ids,
                        stats.table_stats[n].observed_max_unique_ids,
                    ]
                    for n in names
                ],
                dtype=torch.int32,
                device=red_dev,
            )
            dist.all_reduce(counts, op=dist.ReduceOp.MAX)
            stats = KeyedSparseCoreInputStats(
                {
                    n: SparseCoreInputStats(*row)
                    for n, row in zip(names, counts.to("cpu").tolist())
                }
            )
        self.preprocessor.update_preprocessing_parameters(
            stats,
            scale_factor=self.preprocessor._fdo_headroom,
            bucket_step_up=True,
        )
        for s in self.sc_ec.stacked_configs:
            s.max_ids_per_partition = max(
                t.max_ids_per_partition for t in s.tables
            )
            s.max_unique_ids_per_partition = max(
                t.max_unique_ids_per_partition for t in s.tables
            )
        self.preprocessor._allow_id_dropping = False
        self.preprocessor._rebuild_backend()
        self._calibrated = True

    def flatten_preprocessed(self, sc_inputs: Any) -> Tuple[torch.Tensor, ...]:
        out: List[torch.Tensor] = []
        for s in self.sc_ec.stacked_configs:
            t = sc_inputs.table_tensors[s.stack_name]
            out.extend((t.row_pointers, t.embedding_ids, t.sample_ids, t.gains))
        return tuple(out)

    @contextlib.contextmanager
    def bind_step_inputs(
        self, flat_tensors: Sequence[torch.Tensor]
    ) -> Generator[None, None, None]:
        if len(flat_tensors) != self.num_step_tensors:
            raise ValueError(
                f"Expected {self.num_step_tensors} SparseCore step tensors, got "
                f"{len(flat_tensors)}"
            )
        prev = self._bound_tensors
        self._bound_tensors = flat_tensors
        try:
            yield
        finally:
            self._bound_tensors = prev

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        (
            _,
            _,
            KeyedSparseCorePreprocessedInput,
            MinibatchingMode,
            _,
            SparseCorePreprocessedInput,
            *_,
        ) = _load_torchrec_tpu()

        if self._bound_tensors is not None:
            all_lengths = features.lengths()
            key_lengths = {
                k: all_lengths[i * self._rows : (i + 1) * self._rows]
                for i, k in enumerate(features.keys())
            }
            table_tensors = {
                s.stack_name: SparseCorePreprocessedInput(
                    row_pointers=self._bound_tensors[4 * i],
                    embedding_ids=self._bound_tensors[4 * i + 1],
                    sample_ids=self._bound_tensors[4 * i + 2],
                    gains=self._bound_tensors[4 * i + 3],
                    lengths={f: key_lengths[f] for f in s.feature_names},
                    actual_num_ids={
                        f: self._feature_totals[f] for f in s.feature_names
                    },
                    enable_minibatching=(
                        self.preprocessor._minibatching_mode
                        != MinibatchingMode.DISABLED
                    ),
                    max_ids_per_partition=s.max_ids_per_partition,
                    max_unique_ids_per_partition=s.max_unique_ids_per_partition,
                )
                for i, s in enumerate(self.sc_ec.stacked_configs)
            }
            sc_features = KeyedSparseCorePreprocessedInput(table_tensors)
            self.sc_ec.learning_rates = {
                s.stack_name: self.lr_buffer for s in self.sc_ec.stacked_configs
            }
            return self.sc_ec(sc_features)

        features_cpu = (
            features.to(torch.device("cpu"))
            if features.values().device.type != "cpu"
            else features
        )
        if not self._calibrated:
            self.update_preprocessing_parameters(features_cpu)
        sc_inputs = self.preprocessor(features_cpu)
        sc_tpu = sc_inputs.to(self._device)
        lr_val = float(next(iter(self.optimizer.param_groups))["lr"])
        self.lr_buffer.fill_(lr_val)
        self.sc_ec.learning_rates = {
            s.stack_name: self.lr_buffer for s in self.sc_ec.stacked_configs
        }
        return self.sc_ec(sc_tpu)


def install_sparsecore_embeddings(
    model: nn.Module,
    embedding_collection_fqn: str,
    table_configs: Sequence[BaseEmbeddingConfig],
    sparse_config: OptimConfig,
    *,
    rows: int,
    feature_totals: Mapping[str, int],
    world_size: int,
    device: torch.device,
) -> TorchTPUSparseCoreBackedEC:
    """Move dense weights to device and swap EmbeddingCollection with SparseCore."""
    check_binding(model, embedding_collection_fqn)
    if not feature_totals or rows <= 0:
        raise ValueError("rows must be positive and feature_totals non-empty")
    _validate_table_configs(table_configs)
    optimizer_type = _resolve_optimizer_type(sparse_config.optim_type)
    (
        _,
        _,
        _,
        _,
        SparseCoreInputPreprocessor,
        _,
        sc_utils,
        SparseCoreEmbeddingConfig,
        SparseCoreFusedEmbeddingCollection,
        SparseCoreFusedOptimizer,
    ) = _load_torchrec_tpu()

    num_sc_per_device = sc_utils.get_num_sc_per_device() or NUM_SC
    if rows % num_sc_per_device != 0:
        raise ValueError(
            f"Per-device rows ({rows}) must be divisible by "
            f"num_sc_per_device ({num_sc_per_device})"
        )

    sc_tables: List[Any] = []
    for tc in table_configs:
        assert isinstance(tc, EmbeddingConfig)
        max_feat_tokens = max(feature_totals[f] for f in tc.feature_names)
        max_seq_len = max(1, math.ceil(max_feat_tokens / rows))
        sc_tables.append(
            SparseCoreEmbeddingConfig(
                config=tc,
                max_seq_len=max_seq_len,
            )
        )

    auto_stack = all(tc.embedding_dim % 8 == 0 for tc in table_configs)
    raw_sc_ec = SparseCoreFusedEmbeddingCollection(
        tables=sc_tables,
        optimizer_type=optimizer_type,
        optimizer_kwargs={
            "lr": sparse_config.lr,
            "eps": sparse_config.eps,
            "beta1": getattr(sparse_config, "beta1", 0.9),
            "beta2": getattr(sparse_config, "beta2", 0.999),
        },
        batch_size=rows,
        global_device_count=world_size,
        num_sc_per_device=num_sc_per_device,
        auto_stack=auto_stack,
    )
    _unwrap_dtensors(raw_sc_ec)
    owner = dist.get_rank() if world_size > 1 and dist.is_initialized() else 0
    _seed_stacked_shards(
        raw_sc_ec,
        owner=owner,
        world_size=world_size,
        num_sc_per_device=num_sc_per_device,
        device=device,
        sc_utils=sc_utils,
        fused_opt_cls=SparseCoreFusedOptimizer,
    )
    preprocessor = SparseCoreInputPreprocessor(
        tables=sc_tables,
        batch_size=rows,
        global_device_count=world_size,
        num_sc_per_device=num_sc_per_device,
        auto_stack=auto_stack,
        allow_id_dropping=True,
    )
    sc_ec = TorchTPUSparseCoreBackedEC(
        raw_sc_ec,
        preprocessor,
        rows=rows,
        feature_totals=feature_totals,
        embedding_lr=sparse_config.lr,
        device=device,
    )
    swap_submodule(model, embedding_collection_fqn, nn.Identity())
    model.to(device)
    swap_submodule(model, embedding_collection_fqn, sc_ec)
    logger.info(
        "Initialized %d SparseCore stack(s) (%s): %s",
        len(raw_sc_ec.stacked_configs),
        optimizer_type.__name__,
        [s.stack_name for s in raw_sc_ec.stacked_configs],
    )
    return sc_ec


class SparseCoreEmbeddingCollection(nn.Module):
    """Lazy graph-break wrapper delegating directly to TorchTPUSparseCoreBackedEC."""

    def __init__(
        self,
        table_configs: Sequence[BaseEmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        _validate_table_configs(table_configs)
        _resolve_optimizer_type(sparse_config.optim_type)
        self._table_configs = list(table_configs)
        self._sparse_config = sparse_config
        self._device = device
        self._world_size = world_size
        self._backed: Optional[TorchTPUSparseCoreBackedEC] = None
        self._anchor = torch.zeros((), requires_grad=True)
        self.optimizer = KeyedOptimizer(
            params={"lr_anchor": self._anchor},
            state={},
            param_groups=[{"params": [self._anchor], "lr": sparse_config.lr}],
        )

    def _ensure_backed(
        self, features: KeyedJaggedTensor
    ) -> TorchTPUSparseCoreBackedEC:
        if self._backed is not None:
            return self._backed
        keys = features.keys()
        rows = features.stride()
        jagged = features.to_dict()
        feature_totals = {k: int(jagged[k].values().numel()) for k in keys}
        holder = nn.Module()
        holder.ec = EmbeddingCollection(
            tables=[
                tc for tc in self._table_configs if isinstance(tc, EmbeddingConfig)
            ],
            device=torch.device("meta"),
        )
        self._backed = install_sparsecore_embeddings(
            holder,
            "ec",
            self._table_configs,
            self._sparse_config,
            rows=rows,
            feature_totals=feature_totals,
            world_size=self._world_size,
            device=self._device,
        )
        self._backed.optimizer = self.optimizer
        return self._backed

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        backed = self._ensure_backed(features)
        return backed(features)


class TorchTPUSparseCoreBackend(EmbeddingBackend):
    """EmbeddingBackend backed by torchrec.experimental.torch_tpu SparseCore modules."""

    def init_pg(self) -> Tuple[int, int]:
        return 0, 1

    def dense_named_parameters(
        self, model: nn.Module
    ) -> Dict[str, nn.Parameter]:
        prefix = f"{self.embedding_collection_fqn}."
        return {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad and not name.startswith(prefix)
        }

    def setup(
        self,
        model: nn.Module,
        table_configs: Sequence[BaseEmbeddingConfig],
        sparse_config: OptimConfig,
        device: torch.device,
        world_size: int,
    ) -> EmbeddingSetupResult:
        check_binding(model, self.embedding_collection_fqn)
        _resolve_optimizer_type(sparse_config.optim_type)
        if world_size > 1 and dist.is_initialized():
            if dist.get_world_size() != world_size:
                raise ValueError(
                    f"one device per process: world_size {world_size} != "
                    f"{dist.get_world_size()} processes"
                )
        collection = model.get_submodule(self.embedding_collection_fqn)
        configs = (
            list(table_configs)
            if table_configs
            else list(collection.embedding_configs())
        )
        sparsecore = SparseCoreEmbeddingCollection(
            configs, sparse_config, device, world_size
        )
        swap_submodule(model, self.embedding_collection_fqn, sparsecore)
        return EmbeddingSetupResult(
            model=model,
            dense_named_parameters=self.dense_named_parameters(model),
            sparse_runtime=SparseOptimizerRuntime(
                placement=SparseUpdatePlacement.BACKEND_FUSED,
                config=sparse_config,
                fused_optimizer=sparsecore.optimizer,
            ),
        )
