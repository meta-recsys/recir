# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""Train and eval GSPMD graphs compiled together share one set of parameters."""

import unittest

import torch
from recir.api import compile, sync_device_state_to_torch
from recir.backends import XLAGSPMDBackend
from recir.distributed.parallel_mode import GSPMDParallel
from recir.tests.xla.gspmd_test_utils import DIM, LOCAL_BS


def _mlp() -> torch.nn.Module:
    # Deep enough that the eval graph clears partial lowering's minimum size and
    # runs on the XLA devices.
    layers = []
    for _ in range(4):
        layers += [torch.nn.Linear(DIM, DIM), torch.nn.ReLU()]
    return torch.nn.Sequential(*layers)


def _train_step(mod, opt, x):
    torch.sum(mod(x)).backward()
    opt.step()


def _eval_step(mod, x):
    with torch.no_grad():
        return torch.sum(mod(x))


class GSPMDSharedEvalTest(unittest.TestCase):
    def test_train_and_eval_graphs_share_fsdp_parameters(self) -> None:
        self._run_train_and_eval(eval_first=False)

    def test_eval_graph_listed_first_shares_fsdp_parameters(self) -> None:
        self._run_train_and_eval(eval_first=True)

    def _run_train_and_eval(self, eval_first: bool) -> None:
        import jax  # @manual

        num_ranks = jax.device_count()
        if num_ranks < 2:
            self.skipTest("Need >= 2 JAX devices")
        torch.manual_seed(200)
        x = torch.randn(LOCAL_BS * num_ranks, DIM)
        torch.manual_seed(42)
        model = _mlp()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        torch.manual_seed(42)
        reference = _mlp()
        reference_opt = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)

        functions = [_train_step, _eval_step]
        example_args = [(model, opt, x), (model, x)]
        order = [1, 0] if eval_first else [0, 1]
        runners = compile(
            parallel_mode=GSPMDParallel(
                "fully_shard",
                loss_reduction="sum",
                num_ranks=num_ranks,
                num_processes=1,
            ),
            backend=XLAGSPMDBackend(),
        )(*(functions[i] for i in order), example_args=[example_args[i] for i in order])
        train_step, eval_step = (runners[order.index(i)] for i in (0, 1))
        params = {name: id(p) for name, p in model.named_parameters()}

        def train() -> None:
            train_step(model, opt, x)
            reference_opt.zero_grad()
            # GSPMD sums the per-device gradients of a "sum" loss over the
            # global batch, i.e. the reference loss divided by the device count.
            (reference(x).sum() / num_ranks).backward()
            reference_opt.step()

        def evaluate() -> None:
            with torch.no_grad():
                expected = reference(x).sum()
            torch.testing.assert_close(
                eval_step(model, x), expected, atol=1e-4, rtol=1e-4
            )

        # Steps after the first stay on device, so the host tensors are stale
        # and a correct eval proves it reads the train graph's device state.
        train()
        train()
        evaluate()
        train()
        evaluate()
        self.assertEqual({name: id(p) for name, p in model.named_parameters()}, params)
        sync_device_state_to_torch(train_step)
        reference_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter, reference_params[name], atol=1e-5, rtol=1e-5, msg=name
            )


if __name__ == "__main__":
    unittest.main()
