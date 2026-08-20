# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Dict, Optional, Set

import torch


logger: logging.Logger = logging.getLogger(__name__)


class OpSwapRegistry:
    """Registry mapping source op targets to replacement ops.

    The replacement ops must:
      1. Be registered as ``torch.library.custom_op`` with a Meta/FakeTensor
         kernel for shape inference (so ``_is_xla_compatible`` passes them).
      2. If use XLA lowering, provide a torchax/JAX implementation so that
         ``torchax.interop.call_torch`` can dispatch them into a
         ``jax.experimental.pallas.pallas_call(...)``.

    Usage::
        registry = OpSwapRegistry()
        registry.register(
            torch.ops.mylib.triton_flash_attn,
            torch.ops.mylib.pallas_flash_attn,
        )
        # Apply the swap before XLA lowering:
        swapped_gm = swap_ops(gm, registry)
        lowered_gm = partial_lower_xla(swapped_gm)
    """

    def __init__(self) -> None:
        self._swap_map: Dict[object, Callable[..., torch.Tensor]] = {}

    def register(
        self,
        source_op: object,
        replace_op: Callable[..., torch.Tensor],
    ) -> None:
        if source_op in self._swap_map:
            logger.warning("Overwriting existing swap mapping for %s", source_op)
        self._swap_map[source_op] = replace_op
        logger.info("Registered op swap: %s -> %s", source_op, replace_op)

    def unregister(self, source_op: object) -> None:
        self._swap_map.pop(source_op, None)

    def get(self, source_op: object) -> Optional[Callable[..., torch.Tensor]]:
        return self._swap_map.get(source_op, None)

    def has(self, source_op: object) -> bool:
        return source_op in self._swap_map

    @property
    def swap_map(self) -> Dict[object, Callable[..., torch.Tensor]]:
        return dict(self._swap_map)

    @property
    def registered_targets(self) -> Set[object]:
        return set(self._swap_map.keys())

    def clear(self) -> None:
        self._swap_map.clear()

    def __len__(self) -> int:
        return len(self._swap_map)

    def __repr__(self) -> str:
        entries = ", ".join(f"{k} -> {v}" for k, v in self._swap_map.items())
        return f"OpSwapRegistry({entries})"


def swap_ops(
    gm: torch.fx.GraphModule, registry: OpSwapRegistry
) -> torch.fx.GraphModule:
    swap_map = registry.swap_map
    if not swap_map:
        logger.info("swap_ops: empty swap map — no-op")
        return gm

    swap_count = 0
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target in swap_map:
            old_target = node.target
            new_target = swap_map[old_target]
            node.target = new_target
            swap_count += 1
    if swap_count:
        gm.recompile()
        logger.info("swap_ops: performed %d swap(s)", swap_count)

    return gm
