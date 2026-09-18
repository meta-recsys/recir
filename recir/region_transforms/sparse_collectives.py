# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch


def declare_split_invariants(split_list: list[int], total: int | None = None) -> None:
    """Constrain split sizes materialized from tensors during tracing."""
    for split in split_list:
        torch._check_is_size(split)
    if total is not None:
        torch._check(sum(split_list) == total)


def materialize_splits(
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
) -> tuple[list[int], list[int], torch.Tensor]:
    split_count = input_splits.shape[0]
    # Citrine C6: one batched D2H copy supplies all later sparse routing stages.
    cpu_splits = torch.cat((input_splits, output_splits)).to(device="cpu", copy=True)
    return (
        cpu_splits[:split_count].tolist(),
        cpu_splits[split_count:].tolist(),
        cpu_splits,
    )


def split_lists(
    cpu_splits: torch.Tensor, world_size: int
) -> tuple[list[int], list[int]]:
    return (
        cpu_splits[:world_size].tolist(),
        cpu_splits[world_size:].tolist(),
    )


def table_wise_send_counts(
    local_count: int,
    owner_rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    destinations = torch.arange(world_size, dtype=torch.int64, device=device)
    return (destinations == owner_rank).to(torch.int64) * local_count


def table_wise_split_metadata(
    local_count: int,
    owner_rank: int,
    group_name: str,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    send_counts = table_wise_send_counts(
        local_count,
        owner_rank,
        world_size,
        device,
    )
    received_counts = torch.ops._c10d_functional.all_to_all_single(
        send_counts,
        [1] * world_size,
        [1] * world_size,
        group_name,
    )
    received_counts = torch.ops._c10d_functional.wait_tensor(received_counts)
    _input_splits, _output_splits, cpu_splits = materialize_splits(
        send_counts,
        received_counts,
    )
    return cpu_splits


def route_table_wise_vector(
    values: torch.Tensor,
    cpu_splits: torch.Tensor,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    source_splits, owner_splits = split_lists(cpu_splits, world_size)
    declare_split_invariants(source_splits, values.shape[0])
    declare_split_invariants(owner_splits)
    received = torch.ops._c10d_functional.all_to_all_single(
        values.contiguous(),
        owner_splits,
        source_splits,
        group_name,
    )
    return torch.ops._c10d_functional.wait_tensor(received)


def route_table_wise_jagged(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    owner_rank: int,
    group_name: str,
    world_size: int,
    *,
    segments_per_bag: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route complete jagged bags to one owner in source-major order."""
    local_bag_count = lengths.shape[0] // segments_per_bag
    send_bag_counts = table_wise_send_counts(
        local_bag_count,
        owner_rank,
        world_size,
        lengths.device,
    )
    received_bag_counts = torch.ops._c10d_functional.all_to_all_single(
        send_bag_counts,
        [1] * world_size,
        [1] * world_size,
        group_name,
    )
    received_bag_counts = torch.ops._c10d_functional.wait_tensor(received_bag_counts)
    _send_bag_splits, _received_bag_splits, cpu_bag_splits = materialize_splits(
        send_bag_counts,
        received_bag_counts,
    )
    _send_bag_splits, received_bag_splits = split_lists(cpu_bag_splits, world_size)
    cpu_segment_splits = torch.cat(
        (
            cpu_bag_splits[:world_size] * segments_per_bag,
            cpu_bag_splits[world_size:] * segments_per_bag,
        )
    )
    owner_lengths = route_table_wise_vector(
        lengths,
        cpu_segment_splits,
        group_name,
        world_size,
    )
    received_segment_splits = [size * segments_per_bag for size in received_bag_splits]
    send_id_counts = table_wise_send_counts(
        indices.shape[0], owner_rank, world_size, indices.device
    )
    received_id_counts = torch.stack(
        tuple(
            source_lengths.sum()
            for source_lengths in torch.split(
                owner_lengths,
                received_segment_splits,
            )
        )
    )
    _send_id_splits, _received_id_splits, cpu_id_splits = materialize_splits(
        send_id_counts,
        received_id_counts,
    )
    owner_indices = route_table_wise_vector(
        indices,
        cpu_id_splits,
        group_name,
        world_size,
    )
    return owner_indices, owner_lengths, cpu_bag_splits, received_bag_counts


def route_table_wise_rows(
    rows: torch.Tensor,
    cpu_splits: torch.Tensor,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    source_splits, owner_splits = split_lists(cpu_splits, world_size)
    declare_split_invariants(source_splits, rows.shape[0])
    declare_split_invariants(owner_splits)
    width = rows.shape[1]
    contiguous_rows = torch.ops.aten.clone.default(
        rows,
        memory_format=torch.contiguous_format,
    )
    received = torch.ops._c10d_functional.all_to_all_single(
        contiguous_rows.view(-1),
        [size * width for size in owner_splits],
        [size * width for size in source_splits],
        group_name,
    )
    return torch.ops._c10d_functional.wait_tensor(received).reshape(-1, width)


def return_table_wise_rows(
    owner_rows: torch.Tensor,
    cpu_splits: torch.Tensor,
    local_row_count: int,
    group_name: str,
    world_size: int,
) -> torch.Tensor:
    source_splits, owner_splits = split_lists(cpu_splits, world_size)
    declare_split_invariants(owner_splits, owner_rows.shape[0])
    declare_split_invariants(source_splits, local_row_count)
    width = owner_rows.shape[1]
    returned = torch.ops._c10d_functional.all_to_all_single(
        owner_rows.contiguous().view(-1),
        [size * width for size in source_splits],
        [size * width for size in owner_splits],
        group_name,
    )
    returned = torch.ops._c10d_functional.wait_tensor(returned)
    return returned.reshape(local_row_count, width)
