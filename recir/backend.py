# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class BackendType(Enum):
    """Compilation backend for lowering fx.GraphModule subgraphs.

    INDUCTOR: Lower to Inductor (Triton kernels for CUDA GPUs).
    XLA_DYNAMO_BRIDGE: Lower to XLA via torch_xla.core.dynamo_bridge.
    XLA_SHARD_MAP: Lower to XLA via torchax/JAX with shard_map for explicit
        collectives. Maps _c10d_functional ops to JAX collectives via shard_map.
    XLA_GSPMD: Lower to XLA via torchax/JAX with GSPMD auto-parallelism.
    TORCHTPU: Lower to TPU via torch_tpu (StableHLO/PJRT, no JAX). Each dense
        subgraph is compiled with torch_tpu's TpuBackend (aot_autograd ->
        StaticCompiler -> fx_to_mlir -> compile_mlir).
    """

    INDUCTOR = "inductor"
    XLA_DYNAMO_BRIDGE = "xla_dynamo_bridge"
    XLA_SHARD_MAP = "xla_shard_map"
    XLA_GSPMD = "xla_gspmd"
    TORCHTPU = "torchtpu"
