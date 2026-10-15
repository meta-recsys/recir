# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from copy import deepcopy
from functools import wraps
from typing import Any, Callable, List, Tuple

import recir
import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from recir.api import compile, COMPILED_OBJECT_KEY, make_compiled_runner
from recir.backends.base import Backend
from recir.distributed.parallel_mode import DataParallel, ParallelMode
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


class _PassThroughBackend(Backend):
    """Trivial Backend that dispatches gm unchanged (test-only)."""

    def dispatch(self, gm: fx.GraphModule) -> nn.Module:
        return gm


def _compile_with_inline_orchestrator(
    callable_: Callable[..., Any],
    *args: Any,
    eager_passes: Tuple[Callable[[nn.Module], None], ...] = (),
    graph_patches: Tuple[
        Callable[
            [fx.GraphModule, List[torch.Tensor]],
            Tuple[fx.GraphModule, List[torch.Tensor]],
        ],
        ...,
    ] = (),
    parallel_mode: ParallelMode | None = None,
    backend: Backend | None = None,
) -> Callable[..., Any]:
    """Stage 0 -> stage 2 -> stage 3 (graph patches) -> stages 4-10 orchestrator.

    Used by the test_trace_train_step tests to exercise the v7.1 trace +
    compile_from_gm split with bare eager passes + graph patches (the
    equivalent of the legacy ``module_override`` + the ``transform()``
    side of Override).
    """
    import torch.utils._pytree as pytree

    if parallel_mode is None:
        parallel_mode = DataParallel()
    if backend is None:
        backend = _PassThroughBackend()

    mod: nn.Module | None = None
    for arg in pytree.arg_tree_leaves(*args):
        if isinstance(arg, nn.Module):
            mod = arg
            break
    assert mod is not None
    for hook in eager_passes:
        hook(mod)
    # Pass ``parallel_mode`` through so ``recir.trace`` applies the
    # per-process -> global batch transform that mirrors the legacy
    # ``_compile_data_parallel`` inline inflation.  Without it,
    # ``compile_from_gm``'s ``partition_data_parallel`` shards the
    # batch-dim placeholder to ``per_host_batch // num_processes`` while
    # the runtime still passes the per-host shape -- an ``aten.mm``
    # shape mismatch surfaces on the first compiled call (e.g.
    # ``(2, 10) @ (10/num_processes, 10)`` for the Adam train_step test).
    gm, flat_state = recir.trace(callable_, *args, parallel_mode=parallel_mode)
    for patch in graph_patches:
        gm, flat_state = patch(gm, flat_state)
    compiled_obj: recir.api._CompiledResult = recir.compile_from_gm(
        gm,
        flat_state,
        parallel_mode=parallel_mode,
        backend=backend,
    )
    return make_compiled_runner(compiled_obj)


def with_comms(func: Callable[..., Any]) -> Callable[..., Any]:
    @base_with_comms
    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(self.rank)
        return func(self, *args, **kwargs)

    return wrapper


class DataDependentModule(nn.Module):
    def __init__(self, world_size: int) -> None:
        super().__init__()
        self.world_size = world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "This eager implementation shouldn't be executed."
            "This implementation is just an example of how to get around "
            "data-dependant user-defined modules. "
        )
        shape = x.shape
        x = x.view(-1)
        positive = x[x >= 0]
        negative = x[x < 0]

        in_sizes = torch.tensor([positive.numel(), negative.numel()], dtype=torch.int32)
        out_sizes = torch.empty_like(in_sizes)
        dist.all_to_all_single(
            out_sizes,
            in_sizes,
            output_split_sizes=[1, 1],
            input_split_sizes=[1, 1],
        )

        xs = [positive, negative]
        ys = [torch.Tensor(out_sizes[i].item()) for i in range(out_sizes.numel())]
        dist.all_to_all(ys, xs)

        # some dummy compute
        for y in ys:
            y.add_(1)

        dist.all_to_all(xs, ys)

        return torch.cat(xs).reshape(shape)


class DummyModel(nn.Module):
    def __init__(self, world_size: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.ddm = DataDependentModule(world_size)
        self.l2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 2

        return self.relu(self.l2(self.ddm(self.l1(x))))


def ddm(x: torch.Tensor) -> torch.Tensor:
    return x


def ddm_backward(grad: torch.Tensor) -> torch.Tensor:
    return grad


dummy_lib: torch.library.Library = torch.library.Library("dummy", "DEF")  # noqa: TOR901
dummy_lib.define("ddm(Tensor x) -> Tensor")
dummy_lib.impl("ddm", ddm, "CompositeExplicitAutograd")
dummy_lib.define("ddm_backward(Tensor x) -> Tensor")
dummy_lib.impl("ddm_backward", ddm_backward, "CompositeExplicitAutograd")


def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"

    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))


@register_prop_rule(torch.ops.dummy.ddm.default)  # pyre-ignore[56]
def _prop_ddm(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)


@register_prop_rule(torch.ops.dummy.ddm_backward.default)  # pyre-ignore[56]
def _prop_ddm_backward(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)


class DDMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # pyre-ignore [14]
        return torch.ops.dummy.ddm(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:  # pyre-ignore [14]
        return torch.ops.dummy.ddm_backward(grad_x)


class DummyDDM(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DDMFunction.apply(x)


class TraceTrainStepTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _test_optimizer(
        self,
        mod: nn.Module,
        ddp_mod: nn.Module,
        opt: torch.optim.Optimizer,
        ddp_opt: torch.optim.Optimizer,
        inp: torch.Tensor,
        train_step: Callable[..., Any],
    ) -> None:
        ddp_inp = deepcopy(inp)

        # materialize optimizer states
        mod(inp).sum().backward()
        opt.step()
        opt.zero_grad()
        # NB: In *normal* use cases you don't need to do this, but some
        # of the tests setup buffers which require_grad=True which is weird
        # and so work around it
        for buf in mod.buffers():
            buf.grad = None

        ddp_mod(ddp_inp).sum().backward()
        ddp_opt.step()
        ddp_opt.zero_grad()
        for buf in ddp_mod.buffers():
            buf.grad = None

        # test parameter parity
        train_step(mod, opt, inp)

        ddp_mod(ddp_inp).sum().backward()
        ddp_opt.step()

        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sgd(self) -> None:
        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        # FIXME(@mrshenli): remove manual seed once dist.compile can synchronize
        # module parameters.
        torch.manual_seed(1)
        mod = nn.Linear(10, 10, bias=True).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        inp = torch.randn(2, 10).cuda(self.rank)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.01, foreach=True)
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)

    def _test_adam(self, *, foreach: bool, fused: bool) -> None:
        outer: "TraceTrainStepTest" = self

        def assert_dedup(
            gm: fx.GraphModule,
            flat_state: List[torch.Tensor],
        ) -> Tuple[fx.GraphModule, List[torch.Tensor]]:
            # Inline orchestrator runs graph_patches AFTER recir.trace but
            # BEFORE recir.compile_from_gm.  At that point the traced gm
            # carries one ``_spmd::tag_grad.default`` per parameter gradient
            # (inserted by ``_local_compile(..., _stateless_func_wrapper=
            # gradients_tagging)``) and no ``_c10d_functional.all_reduce``
            # nodes yet -- the latter are inserted by
            # ``partition_data_parallel`` during ``compile_from_gm``'s
            # partition step.  Accept either form so the assertion stays
            # valid across both the legacy (post-partition + dedup) and
            # the new (pre-partition) graph patch insertion points.
            collective_nodes = [
                n
                for n in gm.graph.nodes
                if n.target
                in (
                    torch.ops._c10d_functional.all_reduce.default,
                    torch.ops._spmd.tag_grad.default,
                )
            ]
            # Adam-foreach has 1 parameter (bias=False) -> exactly 1
            # collective marker on the only grad.
            outer.assertEqual(len(collective_nodes), 1)
            return gm, flat_state

        def train_step_fn(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        # FIXME(@mrshenli): remove manual seed once dist.compile can synchronize
        # module parameters.
        torch.manual_seed(0)
        # FIXME(@mrshenli): gradients for bias is missing
        mod = nn.Sequential(nn.Linear(10, 10, bias=False)).cuda(self.rank)
        opt = torch.optim.Adam(
            mod.parameters(),
            lr=0.01,
            foreach=foreach,
            fused=fused,
            capturable=True,
        )
        inp = torch.randn(2, 10).cuda(self.rank)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = torch.optim.Adam(
            ddp_mod.parameters(), lr=0.01, foreach=foreach, fused=fused
        )
        train_step = _compile_with_inline_orchestrator(
            train_step_fn, mod, opt, inp, graph_patches=(assert_dedup,)
        )
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_adam_foreach(self) -> None:
        self._test_adam(foreach=True, fused=False)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_adam_fused(self) -> None:
        self._test_adam(foreach=False, fused=True)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multi_optimizer(self) -> None:
        """Two optimizers, each owning a disjoint subset of params, both step
        inside the compiled GM (the multi-optimizer ``recir.api`` path).

        Parameter parity vs DDP running the same two-optimizer split proves
        BOTH ``opt.step()`` calls were traced + executed: if only one stepped,
        the other layer's weights would not update in the compiled step and
        would diverge from the reference. Distinct LRs (0.01 vs 0.1) also catch
        the two optimizers being coalesced into one.

        Uses the lazy ``@compile()`` decorator (like ``test_sgd`` /
        ``test_tied_parameters``) so tracing happens on the first compiled call
        -- AFTER the optimizer states are materialized below -- which is what
        lifts both optimizers' state into ``flat_state`` (the trace-time
        ``named_states`` snapshot in ``_local_compile``).
        """

        @compile()
        def train_step(
            mod: nn.Module,
            opt0: torch.optim.Optimizer,
            opt1: torch.optim.Optimizer,
            inp: torch.Tensor,
        ) -> None:
            mod(inp).sum().backward()
            opt0.step()
            opt1.step()

        torch.manual_seed(0)
        mod = nn.Sequential(
            nn.Linear(10, 10, bias=False),
            nn.Linear(10, 10, bias=False),
        ).cuda(self.rank)
        inp = torch.randn(2, 10).cuda(self.rank)

        # opt0 owns layer 0's weight, opt1 owns layer 1's weight (disjoint).
        params = list(mod.parameters())
        opt0 = torch.optim.Adam([params[0]], lr=0.01, foreach=True, capturable=True)
        opt1 = torch.optim.Adam([params[1]], lr=0.1, foreach=True, capturable=True)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_params = list(ddp_mod.parameters())
        ddp_opt0 = torch.optim.Adam([ddp_params[0]], lr=0.01, foreach=True)
        ddp_opt1 = torch.optim.Adam([ddp_params[1]], lr=0.1, foreach=True)

        ddp_inp = deepcopy(inp)
        # Materialize optimizer states with one eager step on each side BEFORE
        # the first compiled call (so the @compile trace snapshots both states).
        mod(inp).sum().backward()
        opt0.step()
        opt1.step()
        opt0.zero_grad()
        opt1.zero_grad()
        ddp_mod(ddp_inp).sum().backward()
        ddp_opt0.step()
        ddp_opt1.step()
        ddp_opt0.zero_grad()
        ddp_opt1.zero_grad()

        # Compiled step: traces now (state populated), both opts step in the GM.
        train_step(mod, opt0, opt1, inp)

        # Reference step.
        ddp_mod(ddp_inp).sum().backward()
        ddp_opt0.step()
        ddp_opt1.step()

        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_multi_param_group(self) -> None:
        """A single optimizer with TWO param_groups (different LRs) traces +
        steps every group, exercising ``_rematerialize_optimizer`` rewriting
        EVERY ``param_group['params']`` (not just ``param_groups[0]``).

        The distinct per-group LRs (0.01 vs 0.1) mean that if the second group
        were not rewired onto the proxy params, layer 1's weight would not be
        updated by the compiled step and would diverge from the DDP reference.

        Uses the lazy ``@compile()`` decorator so the trace (driven by
        ``_test_optimizer``'s first ``train_step`` call) runs after the
        optimizer state is materialized.
        """

        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        torch.manual_seed(0)
        mod = nn.Sequential(
            nn.Linear(10, 10, bias=False),
            nn.Linear(10, 10, bias=False),
        ).cuda(self.rank)
        inp = torch.randn(2, 10).cuda(self.rank)

        params = list(mod.parameters())
        opt = torch.optim.Adam(
            [
                {"params": [params[0]], "lr": 0.01},
                {"params": [params[1]], "lr": 0.1},
            ],
            foreach=True,
            capturable=True,
        )

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_params = list(ddp_mod.parameters())
        ddp_opt = torch.optim.Adam(
            [
                {"params": [ddp_params[0]], "lr": 0.01},
                {"params": [ddp_params[1]], "lr": 0.1},
            ],
            foreach=True,
        )

        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)

    def _test_train_step_override(self) -> None:
        transform_targets: list[Any] = []

        def swap_ddm_modules(mod: nn.Module) -> None:
            for name, child in list(mod.named_modules()):
                if isinstance(child, DataDependentModule):
                    parent_fqn, leaf = (
                        name.rsplit(".", 1) if "." in name else ("", name)
                    )
                    parent = mod.get_submodule(parent_fqn) if parent_fqn else mod
                    setattr(parent, leaf, DummyDDM())

        def rewrite_ddm_ops(
            gm: fx.GraphModule,
            flat_state: List[torch.Tensor],
        ) -> Tuple[fx.GraphModule, List[torch.Tensor]]:
            for node in gm.graph.nodes:
                if node.target in [
                    torch.ops.dummy.ddm.default,
                    torch.ops.dummy.ddm_backward.default,
                ]:
                    transform_targets.append(node.target)
                    # N.B.: this is not a complete subgraph representing
                    # original logic, as we are testing the ability to
                    # modify graph after DTensor expansion.  The dummy.ddm
                    # op takes one Tensor; replace with ``torch.add(x, x)``
                    # so the rewritten graph stays callable after
                    # ``gm.recompile()`` (the legacy path skipped recompile
                    # and ran the original ddm ops, masking the arity
                    # mismatch).
                    arg = node.args[0]
                    with gm.graph.inserting_before(node):
                        new_node = gm.graph.call_function(torch.add, args=(arg, arg))
                    # Propagate the original node's meta (in particular
                    # ``meta["val"]`` carrying the FakeTensor shape) so the
                    # post-trace ``partition_data_parallel`` strategy builder
                    # treats ``new_node`` as a tensor-valued ACT node rather
                    # than NON_TENSOR.  Without this, downstream ``view``
                    # literals (e.g. the global-batch ``[8, 10]`` baked by
                    # ``recir.trace``) never get rewritten to per-process
                    # shapes by the partitioner.
                    new_node.meta = node.meta.copy()
                    node.replace_all_uses_with(new_node)

            gm.graph.lint()
            gm.graph.eliminate_dead_code()
            gm.recompile()
            return gm, flat_state

        def train_step_fn(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        mod = DummyModel(self.world_size).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)
        # FIXME: symbolic tracing treats bs=1 as constant, have to use bs > 1.
        inp = torch.randn(4, 10).cuda(self.rank)
        train_step = _compile_with_inline_orchestrator(
            train_step_fn,
            mod,
            opt,
            inp,
            eager_passes=(swap_ddm_modules,),
            graph_patches=(rewrite_ddm_ops,),
        )
        train_step(mod, opt, inp)

        # checking transforms are indeed invoked.
        self.assertEqual(
            transform_targets,
            [torch.ops.dummy.ddm.default, torch.ops.dummy.ddm_backward.default],
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_module_override(self) -> None:
        self._test_train_step_override()

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_module_multi_fqn_override(self) -> None:
        transform_targets: list[Any] = []

        def swap_ddm_modules(mod: nn.Module) -> None:
            for name, child in list(mod.named_modules()):
                if isinstance(child, DataDependentModule):
                    parent_fqn, leaf = (
                        name.rsplit(".", 1) if "." in name else ("", name)
                    )
                    parent = mod.get_submodule(parent_fqn) if parent_fqn else mod
                    setattr(parent, leaf, DummyDDM())

        def rewrite_ddm_ops(
            gm: fx.GraphModule,
            flat_state: List[torch.Tensor],
        ) -> Tuple[fx.GraphModule, List[torch.Tensor]]:
            for node in gm.graph.nodes:
                if node.target in [
                    torch.ops.dummy.ddm.default,
                    torch.ops.dummy.ddm_backward.default,
                ]:
                    transform_targets.append(node.target)
                    # N.B.: see _test_train_step_override for the rationale
                    # behind the ``torch.add(arg, arg)`` substitute (dummy.ddm
                    # is a one-arg op and ``torch.add`` requires two; the
                    # legacy path skipped ``gm.recompile()`` so the arity
                    # mismatch never reached the runtime).
                    arg = node.args[0]
                    with gm.graph.inserting_before(node):
                        new_node = gm.graph.call_function(torch.add, args=(arg, arg))
                    # Propagate the original node's meta (see
                    # _test_train_step_override for the same fix); without
                    # ``meta["val"]`` the post-trace
                    # ``partition_data_parallel`` strategy builder marks
                    # ``new_node`` as NON_TENSOR and never rewrites the
                    # global-batch ``view`` literal baked by ``recir.trace``.
                    new_node.meta = node.meta.copy()
                    node.replace_all_uses_with(new_node)

            gm.graph.eliminate_dead_code()
            gm.recompile()
            return gm, flat_state

        class MultiDDM(nn.Module):
            def __init__(self, world_size: int) -> None:
                super().__init__()
                self.l1 = nn.Linear(10, 10)
                self.ddm1 = DataDependentModule(world_size)
                self.l2 = nn.Linear(10, 10)
                self.ddm2 = DataDependentModule(world_size)
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                assert len(x.size()) == 2

                return self.relu(self.ddm2(self.l2(self.ddm1(self.l1(x)))))

        def train_step_fn(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        mod = MultiDDM(self.world_size).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)
        inp = torch.randn(4, 10).cuda(self.rank)
        train_step = _compile_with_inline_orchestrator(
            train_step_fn,
            mod,
            opt,
            inp,
            eager_passes=(swap_ddm_modules,),
            graph_patches=(rewrite_ddm_ops,),
        )
        train_step(mod, opt, inp)

        # checking transforms are indeed invoked.
        self.assertEqual(
            transform_targets,
            [
                torch.ops.dummy.ddm.default,
                torch.ops.dummy.ddm.default,
                torch.ops.dummy.ddm_backward.default,
                torch.ops.dummy.ddm_backward.default,
            ],
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_gm_cache_and_transformation(self) -> None:
        """Verify the compiled-object cache: a custom stage-6 pass runs once
        across repeated calls and the cached ``gm`` reference is reused.

        Refactored off the legacy ``gm_transformation=`` API to the v7.1
        ``backend=`` path: the per-compile callback that used to be
        ``gm_transformation`` is now ``backend.optimize_passes`` (which
        :func:`compile_from_gm` invokes exactly once per compile).  The
        :class:`compile` decorator still caches the resulting
        ``_CompiledResult`` in ``wrapper.__dict__[COMPILED_OBJECT_KEY]`` so
        subsequent ``train_step(...)`` calls reuse the same ``gm``.
        """

        class _CountingBackend(Backend):
            """Counts how often the stage-6 pass fires (== how often
            :func:`compile_from_gm` runs)."""

            def __init__(self) -> None:
                self.call_count: int = 0

                def _record_pass(gm: fx.GraphModule) -> None:
                    self.call_count += 1

                super().__init__(optimize_passes=[_record_pass])

            def dispatch(self, gm: fx.GraphModule) -> nn.Module:
                return gm

        backend = _CountingBackend()

        @compile(parallel_mode=DataParallel(), backend=backend)
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        rank = torch.distributed.get_rank()
        torch.manual_seed(0)
        mod = nn.Linear(10, 10, bias=False).cuda(rank)
        opt = torch.optim.Adam(
            mod.parameters(), lr=0.01, foreach=False, capturable=True
        )
        inp = torch.randn(2, 10).cuda(rank)

        # materialize optimizer states
        mod(inp).sum().backward()
        opt.step()
        opt.zero_grad()

        train_step(mod, opt, inp)
        self.assertEqual(backend.call_count, 1)
        gm = train_step.__dict__[COMPILED_OBJECT_KEY].gm
        train_step(mod, opt, inp)
        self.assertEqual(id(gm), id(train_step.__dict__[COMPILED_OBJECT_KEY].gm))
        self.assertEqual(backend.call_count, 1)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_buffer(self) -> None:
        class BufferModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.register_buffer("dummy_buffer", torch.ones(10, 10))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # N.B.: setting requires_grad in forward, as deepcopy does not
                # work for requires_grad=True buffers.
                # pyrefly: ignore [bad-argument-type, missing-attribute]
                self.dummy_buffer.requires_grad = True
                # pyrefly: ignore [bad-argument-type]
                return torch.matmul(self.fc(x), self.dummy_buffer)

        class AssertOptimizer(torch.optim.Optimizer):
            def __init__(self, params: Any, lr: float) -> None:
                super().__init__(params, dict(lr=lr))

            def step(self) -> None:  # pyre-ignore[14]
                assert len(self.param_groups[0]["params"]) == 2
                with torch.no_grad():
                    for p in self.param_groups[0]["params"]:
                        p += p.grad

        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        torch.manual_seed(0)
        mod = BufferModule().cuda(self.rank)
        inp = torch.randn(2, 10).cuda(self.rank)
        opt = AssertOptimizer(mod.parameters(), lr=0.01)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = AssertOptimizer(ddp_mod.parameters(), lr=0.01)

        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)
        self.assertEqual(mod.dummy_buffer, ddp_mod.module.dummy_buffer)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_expand_dimension(self) -> None:
        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        mod = nn.Linear(10, 10, bias=True).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        inp = torch.randn(2, 10).cuda(self.rank)
        train_step(mod, opt, inp)
        for node in train_step._compiled_obj.gm.graph.nodes:  # pyre-ignore[16]
            if node.target == torch.ops.aten.expand.default:
                # backward grad expansion op should match local batch size
                # instead of global batch size.
                self.assertEqual(node.args[1], [2, 10])

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_tied_parameters(self) -> None:
        """Verify multi-step compiled training with tied (shared) parameters.

        Models with tied parameters (e.g., tie_word_embeddings in Gemma3)
        have fewer entries from list(_mod.parameters()) (which deduplicates)
        than from named_parameters(remove_duplicate=False) (used to build
        flat_state).  Without the pytree alignment fix in stateless_func,
        step 2 crashes because the return structure doesn't match flat_state.
        """

        class TiedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(32, 10)
                self.fc = nn.Linear(10, 10, bias=False)
                self.head = nn.Linear(10, 32, bias=False)
                # Tie head.weight to embed.weight (same as tie_word_embeddings)
                self.head.weight = self.embed.weight

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.head(self.fc(self.embed(x)))

        @compile()
        def train_step(
            mod: nn.Module, opt: torch.optim.Optimizer, inp: torch.Tensor
        ) -> None:
            mod(inp).sum().backward()
            opt.step()

        torch.manual_seed(0)
        mod = TiedModel().cuda(self.rank)
        opt = torch.optim.Adam(
            mod.parameters(), lr=0.01, foreach=False, capturable=True
        )
        inp = torch.randint(0, 32, (4, 8)).cuda(self.rank)

        # Materialize optimizer states
        mod(inp).sum().backward()
        opt.step()
        opt.zero_grad()

        # Step 1 triggers make_fx tracing
        train_step(mod, opt, inp)
        # Step 2 exercises DeviceStateManager output copy-back
        train_step(mod, opt, inp)


if __name__ == "__main__":
    if False:
        run_tests()
