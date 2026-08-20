# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Load tensors from safetensors files without the safetensors package."""

from __future__ import annotations

import json
import logging
import struct
from typing import Any

import torch

logger: logging.Logger = logging.getLogger(__name__)

_SAFETENSORS_DTYPE_MAP: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F64": torch.float64,
}


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load tensors from a safetensors file without the safetensors package.

    The safetensors format is: 8-byte header size (LE uint64) + JSON header
    + raw tensor data.  The header maps tensor names to ``{dtype, shape,
    data_offsets: [start, end]}`` relative to the data section start.

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        Dict mapping tensor names to CPU tensors.
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header: dict[str, Any] = json.loads(f.read(header_size))
        data_start = 8 + header_size

        tensors: dict[str, torch.Tensor] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype_str: str = info["dtype"]
            shape: list[int] = info["shape"]
            start: int = info["data_offsets"][0]
            end: int = info["data_offsets"][1]

            torch_dtype = _SAFETENSORS_DTYPE_MAP.get(dtype_str)
            if torch_dtype is None:
                logger.warning(f"Unknown dtype {dtype_str} for {name}, skipping")
                continue

            f.seek(data_start + start)
            raw = f.read(end - start)
            tensor = torch.frombuffer(bytearray(raw), dtype=torch_dtype)
            tensors[name] = tensor.reshape(shape)

    return tensors
