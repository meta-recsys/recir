# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""What a snapshot records about an FX graph."""

from __future__ import annotations

from recir.pipeline import PIPELINE_PARTITION_META, PIPELINE_REGION_META
from torch import fx


class GraphDump:
    """
    One FX graph being recorded: the one place a snapshot's views of it are computed.

    This class is responsible for every view a snapshot takes of the graph:
    the ``print_readable`` text tlparse shows, and its node counts. It never
    talks to tlparse; the compile writing the graph decides what is written.
    """

    def __init__(self, gm: fx.GraphModule) -> None:
        self.gm = gm

    @property
    def counts(self) -> dict[str, int]:
        """Node counts: all nodes, and placeholders, ``call_function`` and ``call_module`` nodes."""
        counts = {"nodes": 0, "placeholder": 0, "call_function": 0, "call_module": 0}
        for node in self.gm.graph.nodes:
            counts["nodes"] += 1
            if node.op in counts:
                counts[node.op] += 1
        return counts

    def text(self) -> str:
        """The graph's ``print_readable`` text, with strides, devices, and RecIR's region and partition tags."""
        return self.gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            additional_meta=[PIPELINE_REGION_META, PIPELINE_PARTITION_META],
        )
