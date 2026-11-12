# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import logging
import zlib
from typing import Any, Sequence

import torch

logger: logging.Logger = logging.getLogger(__name__)

_META_INIT_KIND_ATTR: str = "_recir_meta_init_kind"
_META_INIT_BOUND_ATTR: str = "_recir_meta_init_bound"
_META_INIT_STREAM_ID_ATTR: str = "_recir_meta_init_stream_id"
# Rows per independently seeded block of a table (see fill_table_shard). Fixed,
# so a row's initial value depends on its table and index, never on the sharding.
_INIT_BLOCK_ROWS: int = 1 << 16


def default_embedding_init_bound(num_embeddings: int) -> float:
    """Return the default TorchRec uniform-init bound for an embedding table."""
    return (1.0 / max(int(num_embeddings), 1)) ** 0.5


def initialize_tensor_from_meta_init(
    tensor: torch.Tensor,
    kind: str,
    bound: float,
) -> torch.Tensor:
    """Initialize a materialized tensor from a recir meta-init specification."""
    if kind == "zeros":
        tensor.zero_()
    elif kind == "uniform":
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"Unsupported recir meta init kind: {kind}")
    return tensor


def table_init_stream_id(name: str) -> int:
    """Stable per-table PRNG stream identifier for on-device init."""
    return zlib.crc32(name.encode("utf-8"))


def fill_table_shard(
    shard: torch.Tensor,
    table_name: str,
    table_shape: Sequence[int],
    offsets: Sequence[int],
    low: float,
    high: float,
) -> torch.Tensor:
    """Fill ``shard``, at ``offsets`` in the table, with its part of the table's
    uniform ``[low, high)`` init.

    The table is drawn in fixed blocks of rows, each from a generator seeded by
    the table's stream id and the block index, so every split of a table over
    any number of ranks assembles the same table; a shard draws at most one
    extra partial block at each end. Values are drawn on the CPU, whose
    generator gives the same stream on every host, then copied to ``shard``.
    """
    if shard.numel() == 0:
        return shard
    first, last = offsets[0], offsets[0] + shard.shape[0]
    columns = tuple(slice(o, o + n) for o, n in zip(offsets[1:], shard.shape[1:]))
    stream = table_init_stream_id(table_name)
    for block in range(first // _INIT_BLOCK_ROWS, -(-last // _INIT_BLOCK_ROWS)):
        start = block * _INIT_BLOCK_ROWS
        end = min(start + _INIT_BLOCK_ROWS, table_shape[0])
        # The CPU generator keeps only a seed's low 32 bits, so the table and
        # block are mixed into one 32-bit value rather than packed side by side.
        seed = zlib.crc32(block.to_bytes(8, "little"), stream)
        generator = torch.Generator().manual_seed(seed)
        values = torch.empty((end - start, *table_shape[1:])).uniform_(
            low, high, generator=generator
        )
        lo, hi = max(start, first), min(end, last)
        shard[lo - first : hi - first].copy_(
            values[(slice(lo - start, hi - start), *columns)]
        )
    return shard


def mark_meta_uniform_init(tensor: Any, bound: float, stream_id: int) -> None:
    setattr(tensor, _META_INIT_KIND_ATTR, "uniform")
    setattr(tensor, _META_INIT_BOUND_ATTR, bound)
    setattr(tensor, _META_INIT_STREAM_ID_ATTR, int(stream_id))


def mark_meta_zero_init(tensor: Any) -> None:
    if tensor.is_meta:
        setattr(tensor, _META_INIT_KIND_ATTR, "zeros")


def copy_meta_init_attrs(dst: Any, src: Any) -> None:
    if not (dst.is_meta and src.is_meta):
        return
    for attr in (
        _META_INIT_KIND_ATTR,
        _META_INIT_BOUND_ATTR,
        _META_INIT_STREAM_ID_ATTR,
    ):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def get_meta_init_stream_id(tensor: Any) -> int:
    return int(getattr(tensor, _META_INIT_STREAM_ID_ATTR, 0))


def get_meta_init_spec(tensor: Any, default_bound: float) -> tuple[str, float]:
    kind = getattr(tensor, _META_INIT_KIND_ATTR, "uniform")
    raw_bound = getattr(tensor, _META_INIT_BOUND_ATTR, None)
    if kind == "uniform" and raw_bound is None:
        # The bound is normally populated via ``mark_meta_uniform_init``. If a
        # caller forgets to mark the tensor, the fallback default (derived from
        # the padded/global row count) may not match the intended TorchRec init
        # scale (which uses the pre-padding ``num_embeddings``). Log so the
        # mismatch doesn't go unnoticed.
        logger.warning(
            "meta tensor missing %s; falling back to default uniform bound "
            "%g, which may not match the intended init scale. Mark the tensor "
            "via mark_meta_uniform_init to set the bound explicitly.",
            _META_INIT_BOUND_ATTR,
            default_bound,
        )
    return kind, float(default_bound if raw_bound is None else raw_bound)
