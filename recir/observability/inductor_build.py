# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor's subgraph builds."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack

from recir.observability.builds import SubgraphBuild


def _without_bundled_autotune_cache() -> AbstractContextManager[object]:
    # Inductor reads and writes its bundled autotune cache only inside a compile
    # context, which a build has only because it is snapshotted. The patch is
    # process-global, so this assumes one build runs at a time.
    import torch._inductor.config as inductor_config

    return inductor_config.patch(bundled_autotune_remote_cache=False)


class InductorBuild(SubgraphBuild):
    """
    One carved subgraph's Inductor build.

    This class is responsible for what an Inductor build adds to the neutral
    build: keeping Inductor's bundled autotune cache off while the build's
    compile id is pushed, so the build autotunes as it would without
    snapshots.
    """

    def _setup(self, stack: ExitStack) -> None:
        super()._setup(stack)
        stack.enter_context(_without_bundled_autotune_cache())
