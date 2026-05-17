# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Optional, Union

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


def dispatch_gm(gm: fx.GraphModule, backend: Optional[BackendType]) -> fx.GraphModule:
    """Dispatch a single graph module to the given backend.

    Walks the FX graph, segments contiguous backend-compatible regions, and
    wraps each region in a backend dispatch module (``_InductorModule`` /
    ``_XLAModule``). NO backend IR is produced here — that happens lazily
    on first ``forward`` of each dispatch module.
    """
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


def dispatch_iter_gm(iter_gm: IterGraphModule, backend: Optional[BackendType]) -> None:
    """Dispatch every sub-GM in an IterGraphModule to the given backend.

    Each sub-GM (setup, main, cleanup) is dispatched independently
    since they may have diverged after cross-iteration optimizations.
    """
    if backend is None:
        return

    iter_gm.setup_gm = dispatch_gm(iter_gm.setup_gm, backend)
    iter_gm.main_gm = dispatch_gm(iter_gm.main_gm, backend)
    iter_gm.cleanup_gm = dispatch_gm(iter_gm.cleanup_gm, backend)


class GraphPipeline:
    """Runs the per-graph recir pipeline: weight-free → IterGraphModule
    formation → cross-iteration optimization → backend dispatch.

    Despite the historical name (`GraphModuleTransformation`, kept as a
    deprecated alias below), this class does NOT do region-semantic
    transforms. The semantics-changing steps live in `RegionTransform`
    subclasses; the per-mode optimization passes live in
    `ParallelMode.refine_graph`. This class is the umbrella that runs the
    remaining pipeline.
    """

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
            # from dispatching setup/main/cleanup graphs separately.
            gm = dispatch_gm(gm, self.backend)

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
        dispatch_iter_gm(iter_gm, self.backend)

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


# Deprecated alias for the old name. Internal callers should prefer
# `GraphPipeline`. Kept until OSS users migrate.
GraphModuleTransformation = GraphPipeline
