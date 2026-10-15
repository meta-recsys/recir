# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""GPU checkpoint adapter for RecIR HybridParallel/TBE embedding state."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from recir.distributed.data_parallel import materialize_embedding_weight
from recir.distributed.sharding_plan import EmbeddingShardingPlan
from recir.training.optim import WarmupController
from recir.training.setup import RecOptimizerConfig
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
    ShardMetadata,
)

_DENSE_WARMUP_KEY = "__dense.__warmup"
_SPARSE_WARMUP_KEY = "__sparse_fused.__sparse_fused.__warmup"


@dataclass(frozen=True)
class _ParameterMetadata:
    global_shape: Tuple[int, ...]
    dtype: torch.dtype


class RecirCheckpointAdapter:
    """Presents embedding parameters as correctly sharded checkpoint state."""

    def __init__(
        self,
        module: torch.nn.Module,
        model: torch.nn.Module,
        embedding_plan: EmbeddingShardingPlan,
        device: torch.device,
        world_size: int,
        rank: int,
        optimizer_config: Optional[RecOptimizerConfig] = None,
    ) -> None:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        invalid_owners = {
            fqn: owner_rank
            for fqn, owner_rank in embedding_plan.table_wise_rank_map.items()
            if not 0 <= owner_rank < world_size
        }
        if invalid_owners:
            raise ValueError(
                "Table-wise checkpoint owners must be inside the world: "
                f"{invalid_owners}"
            )
        self._model = model
        self._embedding_plan = embedding_plan
        self._parameter_metadata: Dict[str, _ParameterMetadata] = {
            fqn: _ParameterMetadata(
                global_shape=tuple(self._parameter(fqn).shape),
                dtype=self._parameter(fqn).dtype,
            )
            for fqn in sorted(embedding_plan.param_fqns)
        }
        self._device = device
        self._world_size = world_size
        self._rank = rank
        self._optimizer_config = optimizer_config
        self._dense_opt: Optional[torch.optim.Optimizer] = None
        self._sparse_opt: Optional[torch.optim.Optimizer] = None
        self._warmup: Optional[WarmupController] = None

        def state_dict_hook(
            hooked_module: torch.nn.Module,
            state_dict: Dict[str, Any],
            prefix: str,
            local_metadata: Dict[str, Any],
        ) -> None:
            self._state_dict_post_hook(state_dict, prefix)

        def load_state_dict_hook(
            hooked_module: torch.nn.Module,
            state_dict: Dict[str, Any],
            prefix: str,
            local_metadata: Dict[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            self._load_state_dict_pre_hook(state_dict, prefix)

        module.register_state_dict_post_hook(state_dict_hook)
        module.register_load_state_dict_pre_hook(load_state_dict_hook)

    @staticmethod
    def local_shard_shape_and_offset(
        global_shape: Tuple[int, ...], world_size: int, rank: int
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        rows = global_shape[0]
        chunk_rows = (rows + world_size - 1) // world_size
        row_offset = min(chunk_rows * rank, rows)
        local_rows = min(chunk_rows, rows - row_offset)
        local_shape = (local_rows, *global_shape[1:])
        offsets = (row_offset, *(0 for _ in global_shape[1:]))
        return local_shape, offsets

    def _parameter(self, fqn: str) -> torch.nn.Parameter:
        value: Any = self._model
        for atom in fqn.split("."):
            value = getattr(value, atom)
        if not isinstance(value, torch.nn.Parameter):
            raise RuntimeError(f"Expected parameter at {fqn}, found {type(value)}")
        return value

    def _as_sharded_tensor(
        self,
        fqn: str,
        local_tensor: torch.Tensor,
        global_shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> ShardedTensor:
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed must be initialized for RecIR checkpointing"
            )
        metadata = self._parameter_metadata[fqn]
        global_shape = metadata.global_shape if global_shape is None else global_shape
        expected_dtype = metadata.dtype if dtype is None else dtype
        if local_tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"Checkpoint shard for {fqn} has dtype {local_tensor.dtype}, "
                f"expected {expected_dtype}"
            )
        local_shard = self._local_shard_shape_and_offset(fqn, global_shape)
        if local_shard is None:
            expected_shape = (0, *global_shape[1:])
            if tuple(local_tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Checkpoint shard for {fqn} has shape "
                    f"{tuple(local_tensor.shape)}, expected {expected_shape}"
                )
            return init_from_local_shards([], list(global_shape))
        local_shape, offsets = local_shard
        if tuple(local_tensor.shape) != local_shape:
            raise RuntimeError(
                f"Checkpoint shard for {fqn} has shape {tuple(local_tensor.shape)}, "
                f"expected {local_shape}"
            )
        metadata = ShardMetadata(
            shard_offsets=list(offsets),
            shard_sizes=list(local_shape),
            placement=f"rank:{self._rank}/{local_tensor.device}",
        )
        return init_from_local_shards(
            [Shard(local_tensor, metadata)], list(global_shape)
        )

    def _local_shard_shape_and_offset(
        self,
        fqn: str,
        global_shape: Tuple[int, ...],
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        if self._embedding_plan.is_table_wise(fqn):
            if self._rank != self._embedding_plan.table_wise_owner(fqn):
                return None
            return global_shape, tuple(0 for _ in global_shape)
        return self.local_shard_shape_and_offset(
            global_shape, self._world_size, self._rank
        )

    def _local_shard_tensor(
        self,
        fqn: str,
        value: ShardedTensor,
        global_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if value.dtype != dtype:
            raise RuntimeError(
                f"Checkpoint shard for {fqn} has dtype {value.dtype}, expected {dtype}"
            )
        local_shards = value.local_shards()
        expected = self._local_shard_shape_and_offset(fqn, global_shape)
        expected_shard_count = int(expected is not None)
        if len(local_shards) != expected_shard_count:
            raise RuntimeError(
                f"Checkpoint shard for {fqn} has {len(local_shards)} local "
                f"shards, expected {expected_shard_count}"
            )
        if expected is None:
            return None
        expected_shape, expected_offsets = expected
        shard = local_shards[0]
        if (
            tuple(shard.tensor.shape) != expected_shape
            or tuple(shard.metadata.shard_sizes) != expected_shape
            or tuple(shard.metadata.shard_offsets) != expected_offsets
        ):
            raise RuntimeError(
                f"Checkpoint shard for {fqn} has tensor shape "
                f"{tuple(shard.tensor.shape)}, shard sizes "
                f"{tuple(shard.metadata.shard_sizes)}, and offsets "
                f"{tuple(shard.metadata.shard_offsets)}; expected shape "
                f"{expected_shape} and offsets {expected_offsets}"
            )
        return shard.tensor

    def _state_dict_post_hook(self, state_dict: Dict[str, Any], prefix: str) -> None:
        for fqn in self._parameter_metadata:
            key = f"{prefix}_impl.{fqn}"
            parameter = self._parameter(fqn)
            if parameter.device.type == "meta":
                raise RuntimeError(
                    "RecIR checkpoint state requires prepare_for_restore() before "
                    f"accessing sharded parameter {fqn}"
                )
            local_tensor = parameter.detach()
            state_dict[key] = self._as_sharded_tensor(fqn, local_tensor)

    def _load_state_dict_pre_hook(
        self, state_dict: Dict[str, Any], prefix: str
    ) -> None:
        for fqn, metadata in self._parameter_metadata.items():
            key = f"{prefix}_impl.{fqn}"
            value = state_dict.get(key)
            if not isinstance(value, ShardedTensor):
                continue
            if tuple(value.size()) != metadata.global_shape:
                raise RuntimeError(
                    f"Checkpoint parameter {key} has global shape "
                    f"{tuple(value.size())}, expected {metadata.global_shape}"
                )
            parameter = self._parameter(fqn)
            if parameter.device.type == "meta":
                raise RuntimeError(
                    "RecIR checkpoint restore requires prepare_for_restore() before "
                    f"loading sharded parameter {fqn}"
                )
            local_tensor = self._local_shard_tensor(
                fqn,
                value,
                metadata.global_shape,
                metadata.dtype,
            )
            if local_tensor is None:
                state_dict[key] = parameter.detach()
                continue
            state_dict[key] = local_tensor

    def _checkpoint_key(self, fqn: str) -> str:
        return f"_impl.{fqn}"

    def _optimizer_checkpoint_key(self, fqn: str) -> str:
        prefix = "__sparse_fused" if fqn in self._parameter_metadata else "__dense"
        return f"{prefix}.{self._checkpoint_key(fqn)}"

    def _sparse_state_key(self, fqn: str) -> str:
        table_name = fqn.split(".")[-2]
        return f"{table_name}.momentum1"

    def _find_optimizer_state(
        self, state: Mapping[str, Any], fqn: str
    ) -> Optional[Mapping[str, Any]]:
        candidates = [
            self._optimizer_checkpoint_key(fqn),
            self._checkpoint_key(fqn),
            fqn,
        ]
        for key in candidates:
            value = state.get(key)
            if isinstance(value, Mapping):
                return value
        suffix = f".{fqn}"
        matches = [
            value
            for key, value in state.items()
            if key.endswith(suffix) and isinstance(value, Mapping)
        ]
        if len(matches) > 1:
            raise RuntimeError(f"Multiple optimizer states match parameter {fqn}")
        return matches[0] if matches else None

    def _sharded_checkpoint_tensor(
        self, fqn: str, state: Mapping[str, Any]
    ) -> torch.Tensor:
        metadata = self._parameter_metadata[fqn]
        state_key = self._sparse_state_key(fqn)
        value = state.get(state_key)
        if value is None:
            for fallback in ("momentum1", "sum", f"{fqn.split('.')[-2]}.sum"):
                value = state.get(fallback)
                if value is not None:
                    break
        if isinstance(value, ShardedTensor):
            if tuple(value.size()) != (metadata.global_shape[0],):
                raise RuntimeError(
                    f"Optimizer state for {fqn} has global shape "
                    f"{tuple(value.size())}, expected {(metadata.global_shape[0],)}"
                )
            local_shard = self._local_shard_tensor(
                fqn,
                value,
                (metadata.global_shape[0],),
                torch.float32,
            )
            local = (
                local_shard
                if local_shard is not None
                else torch.empty(0, dtype=torch.float32, device=self._device)
            )
        elif isinstance(value, torch.Tensor):
            if value.dtype != torch.float32:
                raise RuntimeError(
                    f"Optimizer state for {fqn} has dtype {value.dtype}, "
                    "expected torch.float32"
                )
            local = value.reshape(-1)
        else:
            raise RuntimeError(f"Checkpoint has no sharded optimizer state for {fqn}")

        expected = self._local_shard_shape_and_offset(fqn, (metadata.global_shape[0],))
        local_rows = expected[0][0] if expected is not None else 0
        if expected is not None and local.numel() == metadata.global_shape[0]:
            _, offsets = expected
            local = local.reshape(-1).narrow(0, offsets[0], local_rows)
        else:
            local = local.reshape(-1)
        if local.shape != (local_rows,):
            raise RuntimeError(
                f"Optimizer shard for {fqn} has shape {tuple(local.shape)}, "
                f"expected {(local_rows,)}"
            )

        target = materialize_embedding_weight(
            (local_rows, 1),
            torch.float32,
            self._device,
            location=self._embedding_plan.location_map.get(fqn, "HBM"),
        )
        target.copy_(local.reshape(local_rows, 1).to(target))
        return target

    def optimizer_state_dict(self) -> Dict[str, Any]:
        if self._dense_opt is None or self._sparse_opt is None or self._warmup is None:
            raise RuntimeError(
                "RecIR optimizer state requires prepare_for_restore() before save"
            )
        return self._current_optimizer_state_dict()

    def load_optimizer_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        state = state_dict.get("state")
        if not isinstance(state, Mapping):
            raise RuntimeError("Optimizer checkpoint must contain a state mapping")
        if self._dense_opt is None or self._sparse_opt is None or self._warmup is None:
            raise RuntimeError(
                "RecIR optimizer restore requires prepare_for_restore() before load"
            )
        self._apply_optimizer_state(state)

    def bind_optimizers(
        self,
        dense_opt: torch.optim.Optimizer,
        sparse_opt: torch.optim.Optimizer,
        warmup: WarmupController,
    ) -> None:
        self._dense_opt = dense_opt
        self._sparse_opt = sparse_opt
        self._warmup = warmup

    def _apply_optimizer_state(self, state: Mapping[str, Any]) -> None:
        dense_opt = self._dense_opt
        sparse_opt = self._sparse_opt
        warmup = self._warmup
        assert dense_opt is not None and sparse_opt is not None and warmup is not None

        for fqn, parameter in self._model.named_parameters():
            if fqn in self._parameter_metadata:
                loaded = self._find_optimizer_state(state, fqn)
                if loaded is None:
                    raise RuntimeError(f"Checkpoint has no optimizer state for {fqn}")
                local_sum = self._sharded_checkpoint_tensor(fqn, loaded)
                target_state = sparse_opt.state.get(parameter)
                if target_state is None:
                    if (
                        self._embedding_plan.is_table_wise(fqn)
                        and self._rank != self._embedding_plan.table_wise_owner(fqn)
                        and local_sum.numel() == 0
                    ):
                        continue
                    raise RuntimeError(
                        f"Sparse optimizer state for {fqn} is not initialized"
                    )
                target_sum = target_state.get("sum")
                if not isinstance(target_sum, torch.Tensor):
                    raise RuntimeError(f"Sparse optimizer state for {fqn} lacks sum")
                if target_sum.shape != local_sum.shape:
                    raise RuntimeError(
                        f"Sparse optimizer state for {fqn} has shape "
                        f"{tuple(target_sum.shape)}, expected "
                        f"{tuple(local_sum.shape)}"
                    )
                target_sum.copy_(local_sum)
            elif parameter in dense_opt.state:
                loaded = self._find_optimizer_state(state, fqn)
                if loaded is None:
                    raise RuntimeError(f"Checkpoint has no optimizer state for {fqn}")
                self._copy_dense_optimizer_state(
                    fqn, dense_opt.state[parameter], loaded
                )

        self._load_warmup_state(state, warmup)

    def _copy_dense_optimizer_state(
        self,
        fqn: str,
        target_state: Dict[str, Any],
        loaded_state: Mapping[str, Any],
    ) -> None:
        for key, target in target_state.items():
            if key not in loaded_state:
                raise RuntimeError(f"Checkpoint optimizer state for {fqn} lacks {key}")
            loaded = loaded_state[key]
            if isinstance(target, torch.Tensor):
                if not isinstance(loaded, torch.Tensor):
                    raise RuntimeError(
                        f"Checkpoint optimizer state {fqn}.{key} is not a tensor"
                    )
                if loaded.shape != target.shape:
                    raise RuntimeError(
                        f"Checkpoint optimizer state {fqn}.{key} has shape "
                        f"{tuple(loaded.shape)}, expected {tuple(target.shape)}"
                    )
                target.copy_(loaded.to(target))
            else:
                target_state[key] = loaded

    def _load_warmup_state(
        self, state: Mapping[str, Any], warmup: WarmupController
    ) -> None:
        dense_iter = self._find_warmup_state(state, sparse=False)
        sparse_iter = self._find_warmup_state(state, sparse=True)
        if dense_iter is not None and sparse_iter is not None:
            if dense_iter != sparse_iter:
                raise RuntimeError(
                    f"Dense and sparse warmup iterations differ: "
                    f"{dense_iter} vs {sparse_iter}"
                )
        iteration = dense_iter if dense_iter is not None else sparse_iter
        if iteration is None:
            return
        warmup.load_state_dict({"iter": iteration})

    def _find_warmup_state(
        self, state: Mapping[str, Any], *, sparse: bool
    ) -> Optional[int]:
        for key, value in state.items():
            if not key.endswith("__warmup") or ("sparse" in key) != sparse:
                continue
            if not isinstance(value, Mapping):
                continue
            warmup = value.get("warmup")
            if isinstance(warmup, torch.Tensor) and warmup.numel() >= 1:
                return int(warmup.reshape(-1)[0])
        return None

    def _current_optimizer_state_dict(self) -> Dict[str, Any]:
        dense_opt = self._dense_opt
        sparse_opt = self._sparse_opt
        warmup = self._warmup
        config = self._optimizer_config
        assert dense_opt is not None and sparse_opt is not None and warmup is not None

        state: Dict[str, Any] = {}
        for fqn, parameter in sorted(self._model.named_parameters()):
            key = self._optimizer_checkpoint_key(fqn)
            if fqn in self._parameter_metadata:
                metadata = self._parameter_metadata[fqn]
                optimizer_state = sparse_opt.state.get(parameter)
                sum_state = (
                    optimizer_state.get("sum")
                    if isinstance(optimizer_state, Mapping)
                    else None
                )
                if not isinstance(sum_state, torch.Tensor):
                    if self._embedding_plan.is_table_wise(
                        fqn
                    ) and self._rank != self._embedding_plan.table_wise_owner(fqn):
                        local = torch.empty(
                            0, dtype=torch.float32, device=parameter.device
                        )
                    else:
                        raise RuntimeError(
                            f"Sparse optimizer state for {fqn} lacks sum"
                        )
                else:
                    local = sum_state.reshape(-1)
                state[key] = {
                    self._sparse_state_key(fqn): self._as_sharded_tensor(
                        fqn,
                        local,
                        (metadata.global_shape[0],),
                        torch.float32,
                    )
                }
            elif parameter in dense_opt.state:
                state[key] = {
                    name: value.detach() if isinstance(value, torch.Tensor) else value
                    for name, value in dense_opt.state[parameter].items()
                }

        warmup_state = warmup.state_dict()
        if config is not None and config.dense.default.lr_warmup_schedule:
            state[_DENSE_WARMUP_KEY] = {
                "warmup": torch.tensor([warmup_state["iter"]], dtype=torch.long)
            }
        if config is not None and config.sparse.default.lr_warmup_schedule:
            state[_SPARSE_WARMUP_KEY] = {
                "warmup": torch.tensor([warmup_state["iter"]], dtype=torch.long)
            }
        return {"state": state}
