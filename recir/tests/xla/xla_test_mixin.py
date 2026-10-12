# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import os
from contextlib import contextmanager

from recir.backends.types import BackendType
from recir.passes.gm_transformation import GraphModuleTransformation


@contextmanager
def jax_distributed(port_offset=1000):
    """Context manager for JAX distributed runtime.

    Initializes JAX distributed on entry and shuts it down on exit.
    Uses PyTorch distributed env vars to bootstrap coordination.

    Args:
        port_offset: Offset from MASTER_PORT to avoid colliding with
            PyTorch's NCCL rendezvous. Default 1000.

    Usage::

        @with_comms
        def test_something(self):
            with jax_distributed():
                mesh = Mesh(jax.devices(), ...)
                ...
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        yield
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size <= 1:
        yield
        return

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    jax_coordinator_port = master_port + port_offset
    coordinator_address = f"{master_addr}:{jax_coordinator_port}"
    local_device_id = rank % torch.cuda.device_count()

    # Prevent JAX from initializing the CPU backend (which uses Gloo and
    # may fail to bind on CI hostnames).  Use PyTorch to detect the
    # accelerator so this works on both GPU and TPU environments.
    if "JAX_PLATFORMS" not in os.environ and torch.cuda.is_available():
        os.environ["JAX_PLATFORMS"] = "cuda"

    import jax  # @manual

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=world_size,
        process_id=rank,
        local_device_ids=[local_device_id],
    )
    try:
        yield
    finally:
        jax.distributed.shutdown()


class XLABackendMixin:
    """Mixin that adds JAX/XLA backend to DTensorTestBase tests.

    Uses ``jax_distributed()`` context manager in setUp/tearDown to
    manage the JAX distributed lifecycle cleanly per test method.
    """

    def setUp(self):
        super().setUp()
        self._jax_dist_cm = jax_distributed()
        self._jax_dist_cm.__enter__()

    def tearDown(self):
        self._jax_dist_cm.__exit__(None, None, None)
        super().tearDown()

    @property
    def gm_transformation(self):
        return GraphModuleTransformation(backend=BackendType.XLA_SHARD_MAP)

    @property
    def compilation_backend(self):
        return BackendType.XLA_SHARD_MAP
