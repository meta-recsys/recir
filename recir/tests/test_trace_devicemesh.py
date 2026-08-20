# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


from functools import wraps
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._collective_utils import mesh_broadcast, mesh_scatter
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


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


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor: torch.Tensor) -> None:
        mesh: DeviceMesh = DeviceMesh(self.device_type, mesh_tensor)  # pyre-ignore[16]
        local_tensor = (
            torch.ones(3, 3, device=self.device_type) * self.rank  # pyre-ignore[16]
        )

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor) -> torch.Tensor:  # pyre-ignore[53]
                tensor = funcol.all_reduce(tensor, "sum", group=(mesh, dim))
                # multiply with 1 to trigger wait on read during tracing.
                return tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor.clone())
            res_num = sum(global_ranks)
            self.assertEqual(  # pyre-ignore[16]
                reduced_tensor, torch.ones(3, 3) * res_num
            )

    def _test_broadcast_nd(self, mesh_tensor: torch.Tensor) -> None:
        mesh: DeviceMesh = DeviceMesh(self.device_type, mesh_tensor)  # pyre-ignore[16]

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor) -> torch.Tensor:  # pyre-ignore[53]
                received_tensor = tensor.clone()
                mesh_broadcast(received_tensor, mesh, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            local_tensor = (
                torch.ones(3, 3, device=self.device_type) * self.rank  # pyre-ignore[16]
            )
            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            received_tensor = traced_fn(local_tensor)
            res_num = global_ranks[0]
            self.assertEqual(  # pyre-ignore[16]
                received_tensor, torch.ones(3, 3) * res_num
            )

    def _test_scatter_nd(self, mesh_tensor: torch.Tensor) -> None:
        mesh: DeviceMesh = DeviceMesh(self.device_type, mesh_tensor)  # pyre-ignore[16]

        # check all dim groups
        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                # pyrefly: ignore [missing-attribute]
                torch.ones(3, 3, device=self.device_type)
                * global_rank  # pyre-ignore[16]
                for global_rank in global_ranks
            ]

            def fn(  # pyre-ignore[53]
                to_receive: torch.Tensor, to_scatter: List[torch.Tensor]
            ) -> torch.Tensor:
                mesh_scatter(to_receive, to_scatter, mesh, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return to_receive * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            coordinate: Optional[Tuple[int, ...]] = mesh.get_coordinate()
            assert coordinate is not None
            to_receive = torch.empty_like(scattered_tensors[coordinate[dim]])
            traced_fn = make_fx(fn)(to_receive, [t + 1 for t in scattered_tensors])

            received_tensor = traced_fn(to_receive, scattered_tensors)
            self.assertEqual(  # pyre-ignore[16]
                received_tensor,
                torch.ones(3, 3) * self.rank,  # pyre-ignore[16]
            )

    def _test_all_gather_nd(self, mesh_tensor: torch.Tensor) -> None:
        mesh: DeviceMesh = DeviceMesh(self.device_type, mesh_tensor)  # pyre-ignore[16]
        # each rank have its own tensor, all_gather gives a big tensor
        local_tensor = (
            torch.ones(3, 3, device=self.device_type) * self.rank  # pyre-ignore[16]
        )

        dim_to_subgroups = mesh.get_all_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor) -> list[torch.Tensor]:  # pyre-ignore[53]
                big_tensor = funcol.all_gather_tensor(
                    tensor, gather_dim=0, group=(mesh, dim)
                )
                return list(torch.chunk(big_tensor, dim_group_size))

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)
            gathered_list = traced_fn(local_tensor)

            self.assertEqual(len(gathered_list), dim_group_size)  # pyre-ignore[16]
            for idx, gathered_tensor in enumerate(gathered_list):
                self.assertEqual(  # pyre-ignore[16]
                    gathered_tensor, torch.ones(3, 3) * global_ranks[idx]
                )


class TraceDeviceMesh3DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self) -> int:
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self) -> None:
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_broadcast_nd(self) -> None:
        self._test_broadcast_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_scatter_nd(self) -> None:
        self._test_scatter_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_all_gather_nd(self) -> None:
        self._test_all_gather_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self) -> None:
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_broadcast_nd(self) -> None:
        self._test_broadcast_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_scatter_nd(self) -> None:
        self._test_scatter_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_all_gather_nd(self) -> None:
        self._test_all_gather_nd(torch.arange(4).reshape(2, 2))


if __name__ == "__main__":
    if False:
        run_tests()
