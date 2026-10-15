# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

from recir.backends.types import BackendType
from recir.lowering.device_state import DeviceStateManager
from recir.lowering.partial_lower import partial_lower
from recir.lowering.xla_device_state import XLADeviceStateManager
from recir.lowering.xla_lower import partial_lower_xla
from recir.passes import (
    comm_fusion_with_concat,
    enable_graph_optimization_dump,
    insert_weight_free_nodes,
    remove_copy_from_optimizer,
    schedule_comm_wait,
)
from recir.passes.iter_graph_module import IterGraphModule
from recir.utils.graph import dump_graphs_to_files
from torch import fx


def lower_gm(gm: fx.GraphModule, backend: Optional[BackendType]) -> fx.GraphModule:
    """Lower a single graph module for the given backend."""
    if backend is None:
        return gm

    if backend == BackendType.INDUCTOR:
        return partial_lower(gm)
    elif backend in (
        BackendType.XLA_DYNAMO_BRIDGE,
        BackendType.XLA_SHARD_MAP,
        BackendType.XLA_GSPMD,
    ):
        return partial_lower_xla(gm, backend_type=backend)
    return gm


def lower_iter_gm(iter_gm: IterGraphModule, backend: Optional[BackendType]) -> None:
    """Lower all graph modules in an IterGraphModule for the given backend.

    Each graph module (setup, main, cleanup) is lowered independently
    since they may have diverged after cross-iteration optimizations.
    """
    if backend is None:
        return

    iter_gm.setup_gm = lower_gm(iter_gm.setup_gm, backend)
    iter_gm.main_gm = lower_gm(iter_gm.main_gm, backend)
    iter_gm.cleanup_gm = lower_gm(iter_gm.cleanup_gm, backend)


class GraphModuleTransformation:
    def __init__(
        self,
        *,
        enable_graph_optimization: bool = False,
        enable_weight_free: bool = False,
        backend: Optional[BackendType] = None,
        dump_graphs: bool = False,
        enable_device_state: bool = False,
        enable_iter_graph: bool = True,
    ) -> None:
        if enable_graph_optimization and not enable_iter_graph:
            raise ValueError(
                "enable_graph_optimization=True requires enable_iter_graph=True "
                "because cross-iteration optimization passes operate on IterGraphModule."
            )
        self.enable_graph_optimization = enable_graph_optimization
        self.enable_weight_free = enable_weight_free
        self.backend = backend
        self.dump_graphs = dump_graphs
        self.enable_device_state = enable_device_state
        self.enable_iter_graph = enable_iter_graph

    @property
    def device_state_manager(self) -> DeviceStateManager:
        if self.enable_device_state and self.backend in (
            BackendType.XLA_SHARD_MAP,
            BackendType.XLA_GSPMD,
            BackendType.XLA_DYNAMO_BRIDGE,
        ):
            return XLADeviceStateManager()
        return DeviceStateManager()

    def __call__(self, gm: fx.GraphModule) -> Union[IterGraphModule, fx.GraphModule]:
        if self.dump_graphs:
            graph_folder = dump_graphs_to_files(
                {"before_transformation_gm": gm.print_readable(False)}
            )
            enable_graph_optimization_dump(graph_folder)

        if self.enable_weight_free:
            insert_weight_free_nodes(gm)

        if not self.enable_iter_graph:
            # Skip IterGraphModule wrapping to avoid 3X compilation overhead
            # from lowering setup/main/cleanup graphs separately.
            gm = lower_gm(gm, self.backend)

            if self.dump_graphs:
                dump_graphs_to_files(
                    {"transformed_gm": gm.print_readable(False)},
                    graph_folder,  # type: ignore[possibly-undefined]
                )
            return gm

        iter_gm = IterGraphModule(gm)
        if self.enable_graph_optimization:
            comm_fusion_with_concat(iter_gm, 100)
            schedule_comm_wait(iter_gm)
            remove_copy_from_optimizer(iter_gm)
        iter_gm.finalize_setup()
        lower_iter_gm(iter_gm, self.backend)

        if self.dump_graphs:
            dump_graphs_to_files(
                {
                    "iter_graph_setup_gm": iter_gm.setup_gm.print_readable(False),
                    "iter_graph_main_gm": iter_gm.main_gm.print_readable(False),
                    "iter_graph_cleanup_gm": iter_gm.cleanup_gm.print_readable(False),
                },
                graph_folder,  # type: ignore[possibly-undefined]
            )

        return iter_gm
