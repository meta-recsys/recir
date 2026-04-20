# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Optional

from recir.backend import BackendType
from recir.graph_utils import dump_graphs_to_files
from recir.iter_graph_module import IterGraphModule
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
from torch import fx


def lower_iter_gm(
    iter_gm: IterGraphModule,
    backend: Optional[BackendType],
    parallel_style: str = "replicate",
) -> None:
    """Lower all graph modules in an IterGraphModule for the given backend.

    Each graph module (setup, main, cleanup) is lowered independently
    since they may have diverged after cross-iteration optimizations.
    """
    if backend is None:
        return

    if backend == BackendType.INDUCTOR:
        iter_gm.setup_gm = partial_lower(iter_gm.setup_gm)
        iter_gm.main_gm = partial_lower(iter_gm.main_gm)
        iter_gm.cleanup_gm = partial_lower(iter_gm.cleanup_gm)
    elif backend in (
        BackendType.XLA_DYNAMO_BRIDGE,
        BackendType.XLA_SHARD_MAP,
        BackendType.XLA_GSPMD,
    ):
        iter_gm.setup_gm = partial_lower_xla(iter_gm.setup_gm, backend_type=backend, parallel_style=parallel_style)
        iter_gm.main_gm = partial_lower_xla(iter_gm.main_gm, backend_type=backend, parallel_style=parallel_style)
        iter_gm.cleanup_gm = partial_lower_xla(iter_gm.cleanup_gm, backend_type=backend, parallel_style=parallel_style)


class GraphModuleTransformation:
    def __init__(
        self,
        *,
        enable_graph_optimization: bool = False,
        enable_weight_free: bool = False,
        backend: Optional[BackendType] = None,
        dump_graphs: bool = False,
        enable_device_state: bool = False,
        parallel_style: str = "replicate",
    ) -> None:
        self.enable_graph_optimization = enable_graph_optimization
        self.enable_weight_free = enable_weight_free
        self.backend = backend
        self.dump_graphs = dump_graphs
        self.enable_device_state = enable_device_state
        self.parallel_style = parallel_style

    @property
    def device_state_manager(self) -> DeviceStateManager:
        if self.enable_device_state and self.backend in (
            BackendType.XLA_SHARD_MAP,
            BackendType.XLA_GSPMD,
            BackendType.XLA_DYNAMO_BRIDGE,
        ):
            return XLADeviceStateManager()
        return DeviceStateManager()

    def __call__(self, gm: fx.GraphModule) -> IterGraphModule:
        if self.dump_graphs:
            graph_folder = dump_graphs_to_files(
                {"before_transformation_gm": gm.print_readable(False)}
            )
            enable_graph_optimization_dump(graph_folder)

        if self.enable_weight_free:
            insert_weight_free_nodes(gm)

        iter_gm = IterGraphModule(gm)
        if self.enable_graph_optimization:
            comm_fusion_with_concat(iter_gm, 100)
            schedule_comm_wait(iter_gm)
            remove_copy_from_optimizer(iter_gm)
        iter_gm.finalize_setup()
        lower_iter_gm(iter_gm, self.backend, self.parallel_style)

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
