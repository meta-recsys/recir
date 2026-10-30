# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Shared training state across compiled graphs."""

from typing import Any, Sequence


class SharedGraphState:
    """Move authoritative state between graphs by name, keeping constants local.

    The previous graph relinquishes device handles after a switch: executable
    donation can invalidate them. Graph-local dummy buffers and unnamed transform
    constants retain their own shapes. Model/optimizer state must match exactly.
    """

    def __init__(self, compiled: Sequence[Any], local_prefixes: Sequence[str]) -> None:
        self.compiled: tuple[Any, ...] = tuple(compiled)
        self.indices: list[dict[str, int]] = []
        for obj in compiled:
            names = obj.state_manager.state_fqns
            if not names:
                raise ValueError("Graph switching requires named training state")
            indices = {
                name: index
                for index, name in enumerate(names)
                if not any(prefix in name for prefix in local_prefixes)
            }
            if len(indices) != sum(
                not any(prefix in name for prefix in local_prefixes) for name in names
            ):
                raise ValueError("Duplicate training-state names")
            self.indices.append(indices)
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

    def handoff(self, source: int, target: int) -> None:
        if source == target:
            return
        previous, following = self.compiled[source], self.compiled[target]
        state = previous.state_manager.device_state
        if state is None:
            following.state_manager.device_state = None
            return
        updated = list(following.flat_state)
        for name, destination in self.indices[target].items():
            updated[destination] = state[self.indices[source][name]]
        following.state_manager.device_state = updated
        following.state_manager.keep_output_jax = previous.state_manager.keep_output_jax
        previous.state_manager.device_state = None

    def invalidate(self) -> None:
        for obj in self.compiled:
            obj.state_manager.device_state = None
