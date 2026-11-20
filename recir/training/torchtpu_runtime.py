# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Launch helpers for recir entrypoints on TorchTPU.

TorchTPU runs one process per TPU device under torchrun. Its launch
environment (slice-builder addresses and topology) is decided once per host,
before any process loads the runtime, and every worker inherits it.

    if not under_torchrun():
        relaunch_under_torchrun("my.train.module", sys.argv[1:], nproc_per_node=8)
    load_runtime()
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import List, Optional, Sequence

logger: logging.Logger = logging.getLogger(__name__)

LAUNCH_ENV: tuple[str, ...] = (
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
)


def slice_topology() -> Optional[str]:
    """The whole slice's TorchTPU topology on a multi-host GKE job, or None.

    Left unset, TorchTPU fills in the host's own topology while addressing every
    device of every host, and libtpu rejects the mismatch ("Expected 8 worker
    addresses, got 16"). GKE gives the slice's chip grid as TPU_TOPOLOGY (e.g.
    2x2x2); TorchTPU's v7 topology also counts cores per chip, which the
    host-local topology carries last.
    """
    chips = os.environ.get("TPU_TOPOLOGY")
    if not chips:
        return None
    hardware = importlib.import_module("torch_tpu._internal.utils.hardware")
    host = hardware.get_tpu_topology() or ""
    cores = host.split(",")[3:]
    return ",".join(chips.split("x") + cores)


def set_launch_env(nproc_per_node: int, nnodes: Optional[int] = None) -> None:
    """Describe the job to TorchTPU before its runtime starts.

    JobSet sets JOB_COMPLETION_INDEX on every pod, which TorchTPU reads as a
    multi-host job, so on one host it is hidden for the call and then restored.
    Workers inherit the variables from their launcher; computing them again in
    a worker would describe one device, which v7x has none of.
    """
    if all(name in os.environ for name in LAUNCH_ENV):
        return
    distributed = importlib.import_module("torch_tpu._internal.distributed.distributed")
    single_host = nnodes is None or nnodes == 1
    completion_index = (
        os.environ.pop("JOB_COMPLETION_INDEX", None) if single_host else None
    )
    try:
        if single_host:
            distributed.set_tpu_launch_env(nproc_per_node=nproc_per_node)
            return
        topology = slice_topology()
        if topology is not None:
            os.environ.setdefault("TORCH_TPU_TOPOLOGY", topology)
        # Given the slice topology, TorchTPU ranks hosts by device coordinates
        # but takes MASTER_ADDR and the slice-builder address order from
        # TPU_WORKER_HOSTNAMES, which GKE orders by worker id.
        worker_id = os.environ.get("TPU_WORKER_ID") or os.environ.get(
            "JOB_COMPLETION_INDEX"
        )
        if worker_id is not None:
            os.environ.setdefault("NODE_RANK", worker_id)
        distributed.set_tpu_launch_env(nnodes=nnodes, nproc_per_node=nproc_per_node)
    finally:
        if completion_index is not None:
            os.environ["JOB_COMPLETION_INDEX"] = completion_index


def load_runtime() -> None:
    """Load TorchTPU in a worker, which drives exactly one device."""
    set_launch_env(nproc_per_node=1)
    importlib.import_module("torch_tpu._loader").load()


def under_torchrun() -> bool:
    """True in a worker torchrun started (its agent sets the run id)."""
    return "TORCHELASTIC_RUN_ID" in os.environ


def worker_hostnames() -> List[str]:
    hostnames = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    return [host.strip() for host in hostnames.split(",") if host.strip()]


def torchrun_argv(
    module: str, args: Sequence[str], nproc_per_node: int, nnodes: int
) -> List[str]:
    """torchrun command for this host; read after the launch env is set."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nnodes}",
        f"--node_rank={os.environ.get('NODE_RANK', '0')}",
        f"--nproc_per_node={nproc_per_node}",
        f"--master_addr={os.environ.get('MASTER_ADDR', '127.0.0.1')}",
        f"--master_port={os.environ.get('MASTER_PORT', '29500')}",
        "-m",
        module,
        *args,
    ]


def relaunch_under_torchrun(
    module: str, args: Sequence[str], nproc_per_node: int
) -> None:
    """Replace this process with torchrun running ``module`` on every device."""
    nnodes = max(1, len(worker_hostnames()))
    set_launch_env(nproc_per_node=nproc_per_node, nnodes=nnodes)
    command = torchrun_argv(module, args, nproc_per_node, nnodes)
    logger.info("[DIST] TorchTPU launch: %s", " ".join(command))
    os.execv(sys.executable, command)
