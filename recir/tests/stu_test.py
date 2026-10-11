# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3


import os
import statistics
import time
import unittest
from copy import deepcopy
from typing import Callable, List, Tuple

import torch
from functorch import make_fx
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from recir.api import compile, COMPILED_OBJECT_KEY, sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import DataParallel, GSPMDParallel


def _has_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _has_jax() -> bool:
    try:
        import jax  # @manual

        return len(jax.devices()) >= 1
    except ImportError:
        return False


def _has_accelerator() -> bool:
    return torch.cuda.is_available() or _has_jax()


def _get_test_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_test_kernel() -> HammerKernel:
    return HammerKernel.TRITON if torch.cuda.is_available() else HammerKernel.PYTORCH


def build_stu_model(kernel: HammerKernel = HammerKernel.TRITON) -> STUStack:
    stu_layers: List[STU] = [
        STULayer(
            config=STULayerConfig(
                embedding_dim=16,
                num_heads=2,
                hidden_dim=16,
                attention_dim=16,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                max_attn_len=None,
                attn_alpha=None,
                use_group_norm=False,
                recompute_normed_x=True,
                recompute_uvqk=True,
                recompute_y=True,
                sort_by_length=False,
                contextual_seq_len=0,
            ),
            is_inference=False,
        )
        for _ in range(1)
    ]
    stu = STUStack(
        stu_list=stu_layers,
        is_inference=False,
    )
    stu.recursive_setattr("_hammer_kernel", kernel)
    return stu


def build_inputs(
    num_devices: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    max_uih_len = 64
    batch_size = max(8, num_devices)
    if batch_size % num_devices != 0:
        batch_size = num_devices * ((batch_size + num_devices - 1) // num_devices)
    contextual_seq_len = 0
    embedding_dim = 16
    x_lengths = torch.randint(max_uih_len + 1, (batch_size,))
    x_lengths = x_lengths + contextual_seq_len
    max_seq_len = max_uih_len + contextual_seq_len
    max_targets = 20
    num_targets = torch.randint(1, max_targets, size=(batch_size,))
    x_lengths = x_lengths + num_targets
    max_seq_len = max_seq_len + max_targets
    x_offsets = torch.zeros(x_lengths.numel() + 1, dtype=x_lengths.dtype)
    x_offsets[1:] = torch.cumsum(x_lengths, dim=0)
    total_seq_len = int(x_offsets[-1].cpu().item())
    x = torch.randn(int(total_seq_len), embedding_dim, dtype=torch.bfloat16)
    # Pad the jagged value rows up to a multiple of num_devices so GSPMD's
    # batch-sharding split (np.split along dim 0 in
    # recir.lowering.xla_lower._to_sharded_jax) divides evenly across the
    # local devices. total_seq_len is the sum of random per-sample lengths,
    # so it is generally not divisible. The trailing pad rows fall beyond
    # x_offsets[-1], so the model never indexes them. No-op when
    # num_devices == 1 (the default for the single-device / DTensor tests).
    pad_rows = (-total_seq_len) % num_devices
    if pad_rows:
        x = torch.cat([x, torch.zeros(pad_rows, embedding_dim, dtype=x.dtype)], dim=0)
    return x, x_lengths, x_offsets, max_seq_len, num_targets


class Model(torch.nn.Module):
    def __init__(self, kernel: HammerKernel = HammerKernel.TRITON) -> None:
        super().__init__()
        self.stu: STUStack = build_stu_model(kernel)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(
            self.stu(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )
        )


class STUInferenceTest(unittest.TestCase):
    """Test STU inference by tracing the forward pass into an FX graph via make_fx.

    This mirrors how recir.api.compile works internally: it uses functorch.make_fx
    to trace a stateless version of the model into an fx.GraphModule. Here we trace
    only the forward pass (no backward/optimizer) and verify the traced graph
    produces output identical to eager execution.
    """

    def setUp(self) -> None:
        # Register fbgemm CPU fallbacks so these tests can run on TPU/CPU
        try:
            import recir.lowering.fbgemm_fallback  # noqa: F401
        except ImportError:
            pass

    @unittest.skipUnless(_has_accelerator(), "Requires CUDA or TPU")  # pyre-ignore[56]
    def test_fx_traced_forward(self) -> None:
        """Trace STU forward into an FX graph and verify it matches eager output."""
        torch.manual_seed(42)
        set_dev_mode(True)

        device = _get_test_device()
        kernel = _get_test_kernel()

        model: STUStack = build_stu_model()
        model.recursive_setattr("_hammer_kernel", kernel)
        model.to(device)
        model.eval()

        max_seq_len: int
        x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
        x = x.to(device)
        x_lengths = x_lengths.to(device)
        x_offsets = x_offsets.to(device)
        num_targets = num_targets.to(device)

        # Lift params/buffers as function args (same pattern as recir._local_compile)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        def forward_fn(
            params: dict[str, torch.nn.Parameter],
            buffers: dict[str, torch.Tensor],
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            x_offsets: torch.Tensor,
            num_targets: torch.Tensor,
        ) -> torch.Tensor:
            return torch.func.functional_call(
                model,
                {**params, **buffers},
                args=(),
                kwargs={
                    "x": x,
                    "x_lengths": x_lengths,
                    "x_offsets": x_offsets,
                    "max_seq_len": max_seq_len,
                    "num_targets": num_targets,
                },
            )

        # Trace into FX graph (same primitive recir uses internally)
        gm = make_fx(forward_fn)(params, buffers, x, x_lengths, x_offsets, num_targets)

        # Eager reference
        with torch.no_grad():
            eager_out = model(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )

        # Run traced FX graph
        with torch.no_grad():
            traced_out = gm(params, buffers, x, x_lengths, x_offsets, num_targets)

        self.assertEqual(eager_out.shape, traced_out.shape)
        self.assertEqual(eager_out.dtype, traced_out.dtype)
        self.assertFalse(torch.isnan(traced_out).any().item())
        self.assertFalse(torch.isinf(traced_out).any().item())
        torch.testing.assert_close(eager_out, traced_out)

    @unittest.skipUnless(_has_accelerator(), "Requires CUDA or TPU")  # pyre-ignore[56]
    def test_training_graph_to_inference(self) -> None:
        """Show how RecIR's training graph relates to the inference graph.

        RecIR's @compile() uses make_fx to trace both training and inference.
        The forward computation is shared -- training adds loss reduction and
        backward on top. This test:
        1. Traces a training loss fn (forward + sum) via make_fx
        2. Traces an inference fn (forward only) via make_fx
        3. Verifies inference ops are a subset of training ops
        4. Verifies both produce consistent forward outputs
        """
        torch.manual_seed(42)
        set_dev_mode(True)

        device = _get_test_device()
        kernel = _get_test_kernel()

        model: STUStack = build_stu_model()
        model.recursive_setattr("_hammer_kernel", kernel)
        model.to(device)
        model.eval()

        max_seq_len: int
        x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
        x = x.to(device)
        x_lengths = x_lengths.to(device)
        x_offsets = x_offsets.to(device)
        num_targets = num_targets.to(device)

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        # Shared forward pass — used by both training and inference graphs.
        # This is the key RecIR insight: one forward graph, reused everywhere.
        def forward_fn(
            params: dict[str, torch.nn.Parameter],
            buffers: dict[str, torch.Tensor],
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            x_offsets: torch.Tensor,
            num_targets: torch.Tensor,
        ) -> torch.Tensor:
            return torch.func.functional_call(
                model,
                {**params, **buffers},
                args=(),
                kwargs={
                    "x": x,
                    "x_lengths": x_lengths,
                    "x_offsets": x_offsets,
                    "max_seq_len": max_seq_len,
                    "num_targets": num_targets,
                },
            )

        # Training = shared forward + loss reduction
        def train_loss_fn(
            params: dict[str, torch.nn.Parameter],
            buffers: dict[str, torch.Tensor],
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            x_offsets: torch.Tensor,
            num_targets: torch.Tensor,
        ) -> torch.Tensor:
            return forward_fn(
                params, buffers, x, x_lengths, x_offsets, num_targets
            ).sum()

        inp = (params, buffers, x, x_lengths, x_offsets, num_targets)
        train_gm = make_fx(train_loss_fn)(*inp)
        # Inference = shared forward only
        inference_gm = make_fx(forward_fn)(*inp)

        # --- Graph structure: inference ops ⊆ training ops ---
        def _op_targets(gm: torch.fx.GraphModule) -> set[str]:
            return {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}

        train_ops = _op_targets(train_gm)
        infer_ops = _op_targets(inference_gm)

        self.assertTrue(
            infer_ops.issubset(train_ops),
            f"Extra inference ops: {infer_ops - train_ops}",
        )
        train_only = train_ops - infer_ops
        self.assertTrue(
            len(train_only) > 0,
            "Training graph should have ops not in inference",
        )

        train_n: int = len(list(train_gm.graph.nodes))
        infer_n: int = len(list(inference_gm.graph.nodes))
        self.assertGreater(train_n, infer_n)

        # --- Full training graph with backward ---
        # NOTE: torch.func.grad is not compatible with the custom Triton
        # autograd.Functions in STU (they lack setup_context). RecIR handles
        # this by tracing .backward() directly inside make_fx with
        # FakeTensorMode (see recir.api._local_compile), which does not
        # require functorch-compatible backward. The forward graph sharing
        # demonstrated above is the key insight: both training and inference
        # use the same make_fx primitive and share the forward subgraph.

        # --- Forward output parity ---
        with torch.no_grad():
            eager_out = model(
                x=x,
                x_lengths=x_lengths,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )
            traced_out = inference_gm(*inp)

        self.assertEqual(eager_out.shape, traced_out.shape)
        torch.testing.assert_close(eager_out, traced_out)


# Guard GPU-only imports to avoid import errors on TPU-only machines.
if _has_gpu():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
    from torch.testing._internal.distributed._tensor.common_dtensor import (
        DTensorTestBase,
        with_comms,
    )
else:
    # Stubs so STUDTensorTest can be defined on non-GPU machines.
    DTensorTestBase = unittest.TestCase  # type: ignore[misc]

    def skip_if_lt_x_gpu(n: int) -> Callable:  # type: ignore[assignment]
        return unittest.skip("No GPU")

    def with_comms(fn: Callable) -> Callable:  # type: ignore[assignment]
        return fn


@unittest.skipUnless(_has_gpu(), "Requires at least 2 GPUs")
# pyrefly: ignore [invalid-inheritance]
class STUDTensorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_train_step(
        self,
        train_step: Callable[..., object],
        mod: torch.nn.Module,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        data_parallel_mode: str = "replicate",
    ) -> None:
        ref_mod = (
            FSDP(deepcopy(mod), device_id=self.rank)
            if data_parallel_mode == "fully_shard"
            else DDP(deepcopy(mod), device_ids=[self.rank])
        )
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        ref_opt = torch.optim.SGD(ref_mod.parameters(), lr=0.01, foreach=True)
        ref_x = deepcopy(x)
        ref_x_lengths = deepcopy(x_lengths)
        ref_x_offsets = deepcopy(x_offsets)
        ref_num_targets = deepcopy(num_targets)
        # warm up optimizers using the compiled train_step
        train_step(mod, opt, x, x_lengths, x_offsets, max_seq_len, num_targets)
        opt.zero_grad()
        ref_mod(
            ref_x, ref_x_lengths, ref_x_offsets, max_seq_len, ref_num_targets
        ).sum().backward()
        ref_opt.step()
        ref_opt.zero_grad()
        # run one more compiled step and one reference step
        train_step(mod, opt, x, x_lengths, x_offsets, max_seq_len, num_targets)
        ref_mod(
            ref_x, ref_x_lengths, ref_x_offsets, max_seq_len, ref_num_targets
        ).sum().backward()
        ref_opt.step()
        # test parameter parity
        if data_parallel_mode == "fully_shard":
            full_p1_list = []
            for p1 in mod.parameters():
                shards = [torch.empty_like(p1) for _ in range(self.world_size)]
                torch.distributed.all_gather(shards, p1)
                full_p1_list.append(torch.cat(shards, dim=0))

            with FSDP.summon_full_params(ref_mod):
                for full_p1, p2 in zip(full_p1_list, ref_mod.parameters()):
                    self.assertEqual(full_p1, p2)
        else:
            for p1, p2 in zip(mod.parameters(), ref_mod.parameters()):
                self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_compile_inference(self) -> None:
        """Use @compile() for inference — no backward, no optimizer.

        This is the RecIR inference pattern from the doc:
          @compile()
          def inference_step(model, input_data):
              return model(input_data)
        The decorator traces the forward-only step into an FX graph,
        caches it, and reuses it on subsequent calls.
        """
        torch.manual_seed(0)
        set_dev_mode(True)

        @compile(parallel_mode=DataParallel("replicate"))
        def inference_step(
            mod: torch.nn.Module,
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            x_offsets: torch.Tensor,
            max_seq_len: int,
            num_targets: torch.Tensor,
        ) -> torch.Tensor:
            return mod(x, x_lengths, x_offsets, max_seq_len, num_targets)

        mod = Model().cuda(self.rank)
        x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
        x = x.cuda(self.rank)
        x_lengths = x_lengths.cuda(self.rank)
        x_offsets = x_offsets.cuda(self.rank)
        num_targets = num_targets.cuda(self.rank)

        # First call triggers compilation + caching
        output = inference_step(mod, x, x_lengths, x_offsets, max_seq_len, num_targets)
        self.assertFalse(torch.isnan(output).any().item())

        # Extract compiled FX graph
        compiled_obj = inference_step.__dict__[COMPILED_OBJECT_KEY]
        gm = compiled_obj.gm
        self.assertIsNotNone(gm)
        graph_nodes = list(gm.graph.nodes)
        self.assertGreater(len(graph_nodes), 0)

        # No backward ops in the graph (inference only)
        op_names = [str(n.target) for n in graph_nodes if n.op == "call_function"]
        backward_ops = [op for op in op_names if "backward" in op.lower()]
        self.assertEqual(
            len(backward_ops), 0, f"Unexpected backward ops: {backward_ops}"
        )

        # Second call reuses cached graph (no recompilation)
        output2 = inference_step(mod, x, x_lengths, x_offsets, max_seq_len, num_targets)
        self.assertIs(compiled_obj, inference_step.__dict__[COMPILED_OBJECT_KEY])
        self.assertEqual(output.shape, output2.shape)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_stu(self) -> None:
        seed = torch.random.initial_seed()
        torch.manual_seed(0)
        set_dev_mode(True)

        for data_parallel_mode in ["replicate", "fully_shard"]:

            @compile(parallel_mode=DataParallel(data_parallel_mode))
            def train_step(
                mod: torch.nn.Module,
                opt: torch.optim.Optimizer,
                x: torch.Tensor,
                x_lengths: torch.Tensor,
                x_offsets: torch.Tensor,
                max_seq_len: int,
                num_targets: torch.Tensor,
            ) -> None:
                mod(x, x_lengths, x_offsets, max_seq_len, num_targets).sum().backward()
                opt.step()

            mod = Model().cuda(self.rank)
            torch.manual_seed(seed)
            x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
            x = x.cuda(self.rank)
            x_lengths = x_lengths.cuda(self.rank)
            x_offsets = x_offsets.cuda(self.rank)
            num_targets = num_targets.cuda(self.rank)
            self._test_train_step(
                train_step,
                mod,
                x,
                x_lengths,
                x_offsets,
                max_seq_len,
                num_targets,
                data_parallel_mode,
            )


# ---------------------------------------------------------------------------
# XLA/TPU tests (single-process multi-device, requires JAX)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_jax(), "Requires JAX with at least 1 device")
class STUXLATest(unittest.TestCase):
    """Single-process STU smoke tests for XLA (TPU or GPU via JAX).

    Runs in a single process — no torchrun needed. JAX sees all local
    devices and handles placement. Uses PYTORCH kernel (no Triton on TPU).
    """

    def setUp(self) -> None:
        import jax  # @manual

        # Register fbgemm fallbacks if available
        try:
            import recir.lowering.fbgemm_fallback  # noqa: F401
        except ImportError:
            pass

        self.devices = jax.devices()
        if len(self.devices) < 1:
            self.skipTest("No JAX devices available")
        # On machines with TPUs, verify JAX is actually using them.
        platform = self.devices[0].platform
        if platform != "tpu":
            try:
                tpu_devices = jax.devices("tpu")
                if tpu_devices:
                    self.fail(
                        "TPU devices available but JAX defaulted to "
                        f"{platform}: {self.devices}"
                    )
            except RuntimeError:
                pass

    def test_stu_train(self) -> None:
        """Test STU forward+backward on XLA via GSPMD single-process."""
        torch.manual_seed(42)
        set_dev_mode(True)

        num_devices = len(self.devices)
        mod = Model(kernel=HammerKernel.PYTORCH).cpu()
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)
        # x_offsets is intentionally not passed to the compiled step: an
        # offsets tensor has shape [B + 1], which is never divisible by
        # num_devices and so cannot be batch-sharded by GSPMD. It is
        # recomputed on-device from the (batch-sharded) lengths below.
        x, x_lengths, _x_offsets, max_seq_len, num_targets = build_inputs(num_devices)

        @compile(
            parallel_mode=GSPMDParallel(
                "replicate",
                loss_reduction="sum",
                num_ranks=num_devices,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )
        def train_step(
            mod: torch.nn.Module,
            opt: torch.optim.Optimizer,
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            max_seq_len: int,
            num_targets: torch.Tensor,
        ) -> None:
            # Build jagged offsets from the lengths inside the graph rather
            # than passing a global [B + 1] offsets input (which is not
            # batch-shardable). GSPMD computes this cumsum over the logical
            # global lengths and reshards as needed for the HSTU jagged ops.
            x_offsets = torch.cat(
                [x_lengths.new_zeros(1), torch.cumsum(x_lengths, dim=0)]
            )
            mod(x, x_lengths, x_offsets, max_seq_len, num_targets).sum().backward()
            opt.step()

        train_step(mod, opt, x, x_lengths, max_seq_len, num_targets)

        # Verify parameters were updated and are valid
        for p in mod.parameters():
            self.assertFalse(torch.isnan(p).any().item())
            self.assertFalse(torch.isinf(p).any().item())


def _has_tpu() -> bool:
    """Check if JAX has TPU devices available."""
    try:
        import jax  # @manual

        devices = jax.devices()
        return len(devices) >= 1 and devices[0].platform == "tpu"
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "Requires JAX with TPU devices")
class STUBenchmark(unittest.TestCase):
    """Latency / throughput benchmark for the STU/HSTU operator on XLA (TPU).

    Skipped unless ``STU_BENCH=1`` so it never runs during the normal
    ``python -m unittest recir.tests.stu_test`` suite. It mirrors the dlrmv3
    ``infer_benchmark`` structure (warmup -> ``perf_counter`` measured loop ->
    p50/p95/p99 + throughput) but exercises the same compiled STU step as
    ``STUXLATest.test_stu_train``, reusing ``build_inputs`` / ``Model``.

    Inputs are sharded across all local devices exactly like ``test_stu_train``:
    ``build_inputs`` pads the jagged values to a multiple of ``num_devices`` and
    the step rebuilds the ``[B + 1]`` offsets on-device from the batch-sharded
    lengths (the global offsets tensor is not batch-shardable).

    Env knobs:
      ``STU_BENCH=1``                enable (required)
      ``STU_BENCH_MODE=train|forward``  fwd+bwd+opt vs fwd-only (default train)
      ``STU_BENCH_WARMUP=<int>``     warmup steps incl. XLA compile (default 5)
      ``STU_BENCH_STEPS=<int>``      measured steps (default 50)
      ``STU_BENCH_PROFILE_DIR=<dir>``  if set, emit a JAX xplane trace there
                                     (use a ``gs://`` path on TPU to retrieve it)
      ``STU_BENCH_DUMP_COST=1``      log compiled ``cost_analysis()`` once

    Run on a TPU slice by invoking the benchmark test in a suitable JobSet::

        STU_BENCH=1 STU_BENCH_STEPS=50 \\
            python3 -u -m unittest recir.tests.stu_test.STUBenchmark.test_stu_latency
    """

    def setUp(self) -> None:
        import jax  # @manual

        try:
            import recir.lowering.fbgemm_fallback  # noqa: F401
        except ImportError:
            pass

        self.devices = jax.devices()
        if len(self.devices) < 1:
            self.skipTest("No JAX devices available")
        # Verify we're running on TPU (not GPU) - this benchmark is TPU-specific
        if self.devices[0].platform != "tpu":
            self.skipTest(f"Benchmark requires TPU, found {self.devices[0].platform}")

    @unittest.skipUnless(  # pyre-ignore[56]
        os.environ.get("STU_BENCH") == "1", "set STU_BENCH=1 to run the benchmark"
    )
    def test_stu_latency(self) -> None:
        """Benchmark STU forward (or forward+backward) latency on XLA GSPMD."""
        torch.manual_seed(42)
        set_dev_mode(True)

        mode: str = os.environ.get("STU_BENCH_MODE", "train")
        self.assertIn(mode, ("train", "forward"))
        num_warmup = int(os.environ.get("STU_BENCH_WARMUP", "5"))
        num_steps = int(os.environ.get("STU_BENCH_STEPS", "50"))
        profile_dir = os.environ.get("STU_BENCH_PROFILE_DIR")

        num_devices = len(self.devices)
        mod = Model(kernel=HammerKernel.PYTORCH).cpu()
        x, x_lengths, _x_offsets, max_seq_len, num_targets = build_inputs(num_devices)

        step: Callable[..., object]
        run_step: Callable[[], object]
        step, run_step = self._build_step(
            mode, num_devices, mod, x, x_lengths, max_seq_len, num_targets
        )

        def run_and_barrier() -> None:
            # Force the asynchronously-dispatched device work to complete before
            # the timer stops. In ``forward`` mode ``run_step`` returns a host
            # torch tensor (the runner's jax->torch conversion already forced a
            # device->host sync). In ``train`` mode the step returns None, so we
            # wait on the in-place weight update via sync_device_state_to_torch.
            run_step()
            if mode == "train":
                sync_device_state_to_torch(step)

        # Warmup (the first call triggers lazy XLA compilation).
        for _ in range(num_warmup):
            run_and_barrier()

        if os.environ.get("STU_BENCH_DUMP_COST") == "1":
            self._log_compiled_cost(step)

        latencies = self._measure(run_and_barrier, num_steps, profile_dir)
        self.assertGreater(len(latencies), 0)
        self._report(
            latencies, mode, num_devices, x_lengths.numel(), max_seq_len, num_warmup
        )

    def _build_step(
        self,
        mode: str,
        num_devices: int,
        mod: torch.nn.Module,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> Tuple[Callable[..., object], Callable[[], object]]:
        """Compile the STU step and return ``(compiled_step, run_step)``.

        ``run_step`` is a zero-arg closure over the (host) inputs so the
        measured loop stays free of arg-marshalling. The compiled step rebuilds
        the ``[B + 1]`` offsets on-device from the batch-sharded lengths.
        """
        parallel_mode = GSPMDParallel(
            "replicate",
            loss_reduction="sum",
            num_ranks=num_devices,
            num_processes=1,
        )
        if mode == "train":
            # Citrine C2: foreach=False is required by the XLA GSPMD lowering
            # here and mirrors STUXLATest.test_stu_train (the step under test).
            opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)

            @compile(parallel_mode=parallel_mode, backend=XLAGSPMDBackend())
            def train_step(
                mod: torch.nn.Module,
                opt: torch.optim.Optimizer,
                x: torch.Tensor,
                x_lengths: torch.Tensor,
                max_seq_len: int,
                num_targets: torch.Tensor,
            ) -> None:
                x_offsets = torch.cat(
                    [x_lengths.new_zeros(1), torch.cumsum(x_lengths, dim=0)]
                )
                mod(x, x_lengths, x_offsets, max_seq_len, num_targets).sum().backward()
                opt.step()

            return train_step, lambda: train_step(
                mod, opt, x, x_lengths, max_seq_len, num_targets
            )

        # forward-only (inference operator latency)
        mod.eval()

        @compile(parallel_mode=parallel_mode, backend=XLAGSPMDBackend())
        def forward_step(
            mod: torch.nn.Module,
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            max_seq_len: int,
            num_targets: torch.Tensor,
        ) -> torch.Tensor:
            x_offsets = torch.cat(
                [x_lengths.new_zeros(1), torch.cumsum(x_lengths, dim=0)]
            )
            return mod(x, x_lengths, x_offsets, max_seq_len, num_targets)

        return forward_step, lambda: forward_step(
            mod, x, x_lengths, max_seq_len, num_targets
        )

    def _measure(
        self,
        run_and_barrier: Callable[[], None],
        num_steps: int,
        profile_dir: str | None,
    ) -> List[float]:
        """Time ``num_steps`` calls; optionally emit a JAX xplane trace."""
        import jax  # @manual

        latencies: List[float] = []
        if profile_dir:
            jax.profiler.start_trace(profile_dir)
        try:
            for _ in range(num_steps):
                t0 = time.perf_counter()
                run_and_barrier()
                latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
        finally:
            if profile_dir:
                jax.profiler.stop_trace()
        return latencies

    def _report(
        self,
        latencies: List[float],
        mode: str,
        num_devices: int,
        batch_size: int,
        max_seq_len: int,
        num_warmup: int,
    ) -> None:
        """Compute p50/p95/p99 + throughput and print a summary table."""
        latencies.sort()
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[min(int(n * 0.95), n - 1)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        mean = statistics.mean(latencies)
        throughput = batch_size / (mean / 1000.0)

        platform = self.devices[0].platform
        print("\n" + "=" * 72)
        print("  STU/HSTU XLA latency benchmark")
        print("=" * 72)
        print(
            f"  platform={platform} devices={num_devices} mode={mode} "
            f"batch={batch_size} max_seq_len={max_seq_len}"
        )
        print(f"  warmup={num_warmup} steps  measured={n} steps")
        print("-" * 72)
        print(
            f"  mean={mean:.3f}ms  p50={p50:.3f}ms  p95={p95:.3f}ms  "
            f"p99={p99:.3f}ms  throughput={throughput:.0f} samples/s"
        )
        print("=" * 72)

    def _log_compiled_cost(self, step: Callable[..., object]) -> None:
        """Best-effort log of XLA static cost for each compiled subgraph."""
        compiled_obj = step.__dict__.get(COMPILED_OBJECT_KEY)
        if compiled_obj is None:
            return
        for name, module in compiled_obj.gm.named_modules():
            jax_compiled = getattr(module, "_jax_compiled", None)
            if jax_compiled is None:
                continue
            try:
                cost = jax_compiled.cost_analysis()
            except Exception as e:  # noqa: BLE001 - diagnostics only
                print(f"[cost] {name or '<root>'}: cost_analysis failed: {e}")
                continue
            print(f"[cost] {name or '<root>'}: {cost}")
