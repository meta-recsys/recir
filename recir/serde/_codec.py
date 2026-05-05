# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Encode and decode :class:`torch._subclasses.fake_tensor.FakeTensor` for JSON.

The encoding strategy is a top-level *storage table*.  Every distinct
``UntypedStorage`` shared by FakeTensors that decorate the graph's nodes
becomes one entry keyed by an opaque ``storage_id``.  A FakeTensor is encoded
as ``(storage_id, storage_offset, shape, stride, dtype, fake_device,
requires_grad)`` -- enough to materialise an alias of the appropriate base
storage on the decode side via :func:`torch.as_strided`.

Sharing the storage table across all node-meta FakeTensors preserves
view-aliasing through the round-trip, which matters when downstream consumers
reason about storage identity (DCP sharding, in-graph optimisers, ...).
"""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from .schema import FakeTensorSpec, SchemaViolationError, StorageEntry


# Explicit dtype table (NOT torch introspection) so the wire format remains
# stable across torch versions: a new dtype won't silently change the schema
# until it's added here.
_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.uint16: "uint16",
    torch.uint32: "uint32",
    torch.uint64: "uint64",
    torch.bool: "bool",
    torch.complex32: "complex32",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
    torch.float8_e4m3fnuz: "float8_e4m3fnuz",
    torch.float8_e5m2fnuz: "float8_e5m2fnuz",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


def encode_dtype(dtype: torch.dtype) -> str:
    """Map a ``torch.dtype`` to its short JSON-friendly name."""
    if dtype not in _DTYPE_TO_STR:
        raise ValueError(f"Unsupported dtype for serialisation: {dtype}")
    return _DTYPE_TO_STR[dtype]


def decode_dtype(name: str) -> torch.dtype:
    """Inverse of :func:`encode_dtype`."""
    if name not in _STR_TO_DTYPE:
        raise SchemaViolationError(f"Unknown dtype name in payload: {name!r}")
    return _STR_TO_DTYPE[name]


def encode_device(device: torch.device) -> str:
    """Encode ``torch.device`` as ``"<type>"`` or ``"<type>:<index>"``."""
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def decode_device(s: str) -> torch.device:
    """Inverse of :func:`encode_device`."""
    return torch.device(s)


class _StorageRegistry:
    """Helper that assigns stable IDs to ``UntypedStorage`` instances.

    Encoding walks every FakeTensor in the graph and asks the registry for
    an ID for that tensor's storage.  IDs are minted in encounter order
    (``"s0"``, ``"s1"``, ...).
    """

    def __init__(self) -> None:
        self._by_storage_key: dict[int, str] = {}
        self._entries: dict[str, StorageEntry] = {}

    def storage_id_for(self, t: FakeTensor) -> str:
        # ``UntypedStorage`` is reference-equality based; we use the Python
        # id() of the storage object as a key into our memo table.  This
        # is safe because the FakeTensorMode keeps storages alive for the
        # lifetime of the trace.
        storage = t.untyped_storage()
        key = id(storage)
        if key in self._by_storage_key:
            return self._by_storage_key[key]
        sid = f"s{len(self._by_storage_key)}"
        self._by_storage_key[key] = sid
        self._entries[sid] = StorageEntry(
            storage_id=sid,
            nbytes=storage.nbytes(),
            fake_device=encode_device(t.fake_device),
            base_dtype=encode_dtype(t.dtype),
        )
        return sid

    def entries(self) -> dict[str, StorageEntry]:
        return self._entries


def encode_fake_tensor(t: FakeTensor, registry: _StorageRegistry) -> FakeTensorSpec:
    """Encode a single ``FakeTensor`` against a storage registry."""
    return FakeTensorSpec(
        kind="fake_tensor",
        storage_id=registry.storage_id_for(t),
        storage_offset=int(t.storage_offset()),
        shape=list(t.shape),
        stride=list(t.stride()),
        dtype=encode_dtype(t.dtype),
        fake_device=encode_device(t.fake_device),
        requires_grad=bool(t.requires_grad),
    )


class FakeTensorDecoder:
    """Stateful decoder that materialises FakeTensors against a fresh
    ``FakeTensorMode``.

    A single decoder instance is used for the entire graph so that all
    FakeTensors with the same ``storage_id`` share one underlying base
    tensor (and therefore one ``UntypedStorage``).
    """

    def __init__(self, fake_mode: FakeTensorMode | None = None) -> None:
        self._fake_mode: FakeTensorMode = fake_mode or FakeTensorMode(
            allow_non_fake_inputs=True
        )
        self._bases: dict[str, FakeTensor] = {}

    @property
    def fake_mode(self) -> FakeTensorMode:
        return self._fake_mode

    def _get_or_create_base(
        self, sid: str, table: dict[str, StorageEntry]
    ) -> FakeTensor:
        if sid in self._bases:
            return self._bases[sid]
        if sid not in table:
            raise SchemaViolationError(f"storage_id {sid!r} not found in storage_table")
        entry = table[sid]
        device = decode_device(entry.fake_device)
        base_dtype = decode_dtype(entry.base_dtype)
        elem_size = base_dtype.itemsize
        if elem_size == 0:
            raise ValueError(f"dtype {base_dtype} has zero itemsize")
        # Round up so the base storage is at least as large as ``nbytes``.
        numel = (entry.nbytes + elem_size - 1) // elem_size
        with self._fake_mode:
            base = torch.empty_strided((numel,), (1,), dtype=base_dtype, device=device)
        if not isinstance(base, FakeTensor):
            raise AssertionError(
                f"FakeTensorMode failed to produce a FakeTensor; got {type(base)!r}"
            )
        self._bases[sid] = base
        return base

    def decode(
        self, spec: FakeTensorSpec, table: dict[str, StorageEntry]
    ) -> FakeTensor:
        """Materialise a FakeTensor whose storage matches ``spec.storage_id``."""
        base = self._get_or_create_base(spec.storage_id, table)
        with self._fake_mode:
            view = torch.as_strided(
                base,
                size=tuple(spec.shape),
                stride=tuple(spec.stride),
                storage_offset=int(spec.storage_offset),
            )
            target_dtype = decode_dtype(spec.dtype)
            if view.dtype != target_dtype:
                view = view.view(target_dtype)
        if not isinstance(view, FakeTensor):
            raise AssertionError(
                f"as_strided in FakeTensorMode produced {type(view)!r}, "
                f"expected FakeTensor"
            )
        view.requires_grad_(bool(spec.requires_grad))
        return view
