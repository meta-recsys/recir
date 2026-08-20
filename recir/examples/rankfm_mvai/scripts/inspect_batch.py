#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""Introspect one batch file from a PreprocOutput dump.

Scratch tool for the recir-vs-light parity bring-up: confirms the concrete
batch class, KJT key sets, and tensor shapes of a dump written by
PreprocOutputDumpDataLoader, so the recir data path can be validated against
it before any model is built.

    buck2 run //recir/examples/rankfm_mvai/scripts:inspect_batch -- /tmp/b0.pt.zst
"""

import io
import sys

import torch
import zstandard

# Imported for its unpickling side effect: the dump is a torch.save of this
# dataclass, so the module must be in the link tree for torch.load to resolve
# it. Also used for the concrete-type assertion below.
from minimal_viable_ai.core.ranking_common.configs import MTMLROOTrainBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _describe_kjt(name: str, kjt: KeyedJaggedTensor) -> None:
    keys = kjt.keys()
    values = kjt.values()
    lengths = kjt.lengths()
    print(f"  {name}: {len(keys)} keys, stride={kjt.stride()}")
    print(f"    values: shape={tuple(values.shape)} dtype={values.dtype}")
    print(f"    lengths: shape={tuple(lengths.shape)} sum={int(lengths.sum())}")
    print(f"    max_value={int(values.max()) if values.numel() else 'n/a'}")
    # Per-key ranges, not just the global max. The global max is dominated by
    # the *_unhashed features and says nothing about whether the features that
    # actually index an embedding table are in range -- which is what decides
    # if a shrunk hash size will trip
    # jagged_unique_indices' device-side assert.
    jt_dict = kjt.to_dict()
    print(f"    {'key':<48} {'n':>10} {'max':>22}")
    for key in keys:
        key_values = jt_dict[key].values()
        n = int(key_values.numel())
        key_max = int(key_values.max()) if n else -1
        print(f"      - {key:<46} {n:>10} {key_max:>22}")


def main() -> None:
    path = sys.argv[1]
    with open(path, "rb") as fh:
        raw = zstandard.ZstdDecompressor().decompress(fh.read())
    batch = torch.load(io.BytesIO(raw), weights_only=False)

    cls = type(batch)
    print(f"class: {cls.__module__}.{cls.__qualname__}")
    print(f"mro: {[c.__name__ for c in cls.__mro__]}")
    print(f"is MTMLROOTrainBatch: {isinstance(batch, MTMLROOTrainBatch)}")
    print()

    for field in sorted(vars(batch)):
        value = getattr(batch, field)
        if isinstance(value, KeyedJaggedTensor):
            _describe_kjt(field, value)
        elif isinstance(value, torch.Tensor):
            print(f"  {field}: Tensor shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"  {field}: dict with {len(value)} keys -> {sorted(value)[:8]}")
        elif isinstance(value, list):
            print(f"  {field}: list len={len(value)}")
        else:
            print(f"  {field}: {type(value).__name__} = {value!r:.120}")


if __name__ == "__main__":
    main()
