# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared training state across compiled graphs."""

from typing import Any, Collection, Sequence


def graph_local_state_indices(
    state_fqns: Sequence[str], local_fqns: Collection[str]
) -> set[int]:
    """Locate declared model leaves and their optimizer state by exact keypath."""
    model_paths = {f"[0][{name!r}]" for name in local_fqns}
    optimizer_paths = tuple(f"[1][{name!r}][" for name in local_fqns)
    return {
        index
        for index, path in enumerate(state_fqns)
        if path in model_paths or path.startswith(optimizer_paths)
    }


class SharedGraphState:
    """Validate shared training state and bind each graph to one device store."""

    def __init__(
        self, compiled: Sequence[Any], local_indices: Sequence[Collection[int]]
    ) -> None:
        if len(compiled) != len(local_indices):
            raise ValueError("Each capacity graph needs its own local-state indices")
        self.compiled: tuple[Any, ...] = tuple(compiled)
        self.store: dict[str, Any] = {}
        self.indices: list[dict[str, int]] = []
        for obj, local in zip(compiled, local_indices):
            names = obj.state_manager.state_fqns
            if not names:
                raise ValueError("Graph switching requires named training state")
            if any(index < 0 or index >= len(obj.flat_state) for index in local):
                raise ValueError("Graph-local state index is out of range")
            indices = {
                name: index for index, name in enumerate(names) if index not in local
            }
            if len(indices) != sum(index not in local for index in range(len(names))):
                raise ValueError("Duplicate training-state names")
            self.indices.append(indices)
        self._bind_host_state()
        for obj, local in zip(compiled, local_indices):
            obj.state_manager.share_device_state(
                self.store, obj.gm, obj.flat_state, local_indices=local
            )

    def _bind_host_state(self) -> None:
        compiled = self.compiled
        reference = self.indices[-1]
        original_inputs = [
            n for n in compiled[-1].gm.graph.nodes if n.op == "placeholder"
        ]
        for obj, indices in zip(compiled[:-1], self.indices[:-1]):
            if indices.keys() != reference.keys():
                raise ValueError("Capacity graphs have different training-state names")
            other_inputs = [n for n in obj.gm.graph.nodes if n.op == "placeholder"]
            for name, index in indices.items():
                original = compiled[-1].flat_state[reference[name]]
                other = obj.flat_state[index]
                if original.shape != other.shape or original.dtype != other.dtype:
                    raise ValueError(f"Capacity graphs have incompatible state: {name}")
                for flag in ("is_fsdp_sharded", "fsdp_shard_dim"):
                    if original_inputs[reference[name]].meta.get(flag) != other_inputs[
                        index
                    ].meta.get(flag):
                        raise ValueError(
                            f"Capacity graphs have incompatible state layout: {name}"
                        )
                # Host state also shares storage; checkpoints need only one copy.
                obj.flat_state[index] = original

    def refresh_host_state(self, source: int) -> None:
        for target, obj in enumerate(self.compiled):
            for name, index in self.indices[target].items():
                obj.flat_state[index] = self.compiled[source].flat_state[
                    self.indices[source][name]
                ]

    def check_host_state(self, source: int) -> None:
        """Check each graph's shared host binding without changing it."""
        for target, obj in enumerate(self.compiled):
            for name, index in self.indices[target].items():
                expected = self.compiled[source].flat_state[self.indices[source][name]]
                actual = obj.flat_state[index]
                if (
                    actual.is_meta
                    or actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                    or actual.data_ptr() != expected.data_ptr()
                ):
                    raise RuntimeError(
                        f"Capacity graph {target} has a stale host-state binding: {name}"
                    )

    def handoff(self, source: int, target: int) -> None:
        if source == target:
            return
        previous, following = self.compiled[source], self.compiled[target]
        following.state_manager.keep_output_jax = previous.state_manager.keep_output_jax

    def invalidate(self) -> None:
        for obj in self.compiled:
            obj.state_manager.invalidate_device_state()
