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

# pyre-strict


import unittest
from copy import deepcopy
from typing import Callable, List, Tuple

import torch
from functorch import make_fx
from generative_recommenders.common import HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from recir.api import compile, COMPILED_OBJECT_KEY
from recir.distributed.parallel_mode import DataParallel
from recir.graph_utils import get_output, rebuild_graph
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def build_stu_model() -> STUStack:
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
    stu.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
    return stu


def build_inputs() -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor
]:
    max_uih_len = 64
    batch_size = 8
    contextual_seq_len = 0
    embedding_dim = 16
    x_lengths = torch.randint(max_uih_len + 1, (batch_size,))
    x_lengths = x_lengths + contextual_seq_len
    max_seq_len = max_uih_len + contextual_seq_len
    max_targets = 20
    num_targets = torch.randint(1, max_targets, size=(batch_size,))
    x_lengths = x_lengths + num_targets
    max_seq_len = max_seq_len + max_targets
    x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
    total_seq_len = int(x_offsets[-1].cpu().item())
    x = torch.randn(int(total_seq_len), embedding_dim, dtype=torch.bfloat16)
    return x, x_lengths, x_offsets, max_seq_len, num_targets


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stu: STUStack = build_stu_model()

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")  # pyre-ignore[56]
    def test_fx_traced_forward(self) -> None:
        """Trace STU forward into an FX graph and verify it matches eager output."""
        torch.manual_seed(42)
        set_dev_mode(True)

        model: STUStack = build_stu_model()
        model.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        model.cuda()
        model.eval()

        max_seq_len: int
        x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
        x = x.cuda()
        x_lengths = x_lengths.cuda()
        x_offsets = x_offsets.cuda()
        num_targets = num_targets.cuda()

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")  # pyre-ignore[56]
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

        model: STUStack = build_stu_model()
        model.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        model.cuda()
        model.eval()

        max_seq_len: int
        x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
        x = x.cuda()
        x_lengths = x_lengths.cuda()
        x_offsets = x_offsets.cuda()
        num_targets = num_targets.cuda()

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")  # pyre-ignore[56]
    def test_detach_inference_from_training(self) -> None:
        """Detach the inference graph from a training graph via DCE.

        Given a training graph that returns (loss, forward_output), modify
        the output to only keep forward_output and run dead code elimination.
        The loss reduction ops (aten.sum) become unreachable and get removed,
        producing a clean inference graph from the training graph.
        """
        torch.manual_seed(42)
        set_dev_mode(True)

        model: STUStack = build_stu_model()
        model.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        model.cuda()
        model.eval()

        max_seq_len: int
        x, x_lengths, x_offsets, max_seq_len, num_targets = build_inputs()
        x = x.cuda()
        x_lengths = x_lengths.cuda()
        x_offsets = x_offsets.cuda()
        num_targets = num_targets.cuda()

        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())

        # Shared forward
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

        # Training returns BOTH loss and forward output
        def train_with_output_fn(
            params: dict[str, torch.nn.Parameter],
            buffers: dict[str, torch.Tensor],
            x: torch.Tensor,
            x_lengths: torch.Tensor,
            x_offsets: torch.Tensor,
            num_targets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            out = forward_fn(params, buffers, x, x_lengths, x_offsets, num_targets)
            return out.sum(), out

        inp = (params, buffers, x, x_lengths, x_offsets, num_targets)

        # Trace training graph (returns (loss, forward_output))
        train_gm = make_fx(train_with_output_fn)(*inp)
        train_node_count: int = len(list(train_gm.graph.nodes))

        # --- Detach inference graph via output modification + DCE ---
        infer_gm = deepcopy(train_gm)
        output_node = get_output(infer_gm.graph)
        # output_node.args[0] is [loss, forward_output]
        # Replace loss with forward_output so both slots return the same tensor.
        # This preserves the pytree output spec (still a 2-tuple) while making
        # the loss node unreachable for DCE to remove.
        original_output = output_node.args[0]
        assert isinstance(original_output, (list, tuple))
        self.assertEqual(len(original_output), 2)
        forward_output = original_output[1]
        output_node.args = ([forward_output, forward_output],)
        rebuild_graph(infer_gm)

        infer_node_count: int = len(list(infer_gm.graph.nodes))
        self.assertLess(
            infer_node_count,
            train_node_count,
            "DCE should have removed training-only ops (aten.sum)",
        )

        # Verify no aten.sum in the detached graph
        detached_ops = {
            str(n.target) for n in infer_gm.graph.nodes if n.op == "call_function"
        }
        sum_ops = {op for op in detached_ops if "sum" in op}
        self.assertEqual(len(sum_ops), 0, f"aten.sum should be removed: {sum_ops}")

        # --- Verify numerical parity with separately traced inference graph ---
        ref_gm = make_fx(forward_fn)(*inp)

        with torch.no_grad():
            detached_out = infer_gm(*inp)[0]  # both tuple elements are identical
            ref_out = ref_gm(*inp)

        self.assertEqual(detached_out.shape, ref_out.shape)
        torch.testing.assert_close(detached_out, ref_out)


class STUTest(DTensorTestBase):
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
            for use_parametrization in [False, True]:

                @compile(
                    parallel_mode=DataParallel(
                        data_parallel_mode,
                        use_parametrization=use_parametrization,
                    )
                )
                def train_step(
                    mod: torch.nn.Module,
                    opt: torch.optim.Optimizer,
                    x: torch.Tensor,
                    x_lengths: torch.Tensor,
                    x_offsets: torch.Tensor,
                    max_seq_len: int,
                    num_targets: torch.Tensor,
                ) -> None:
                    mod(
                        x, x_lengths, x_offsets, max_seq_len, num_targets
                    ).sum().backward()
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
