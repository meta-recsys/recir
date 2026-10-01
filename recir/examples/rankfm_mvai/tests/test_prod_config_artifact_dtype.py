# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""The prod-config transport carries torch.dtype values.

CoScaling's config holds one (``TwoStageCacheConfig.inference_export_feature_dtype``),
which the transport used to reject as "cannot carry as plain data".
"""

import dataclasses
import pickle
import pickletools
import unittest

import torch
from recir.examples.rankfm_mvai.prod_config_artifact import _decode, _encode


@dataclasses.dataclass
class _HoldsDtype:
    dtype: torch.dtype = torch.float32
    name: str = "x"


class TorchDtypeTransportTest(unittest.TestCase):
    def test_round_trips_every_common_dtype(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int64):
            with self.subTest(dtype=dtype):
                self.assertIs(_decode(_encode(dtype, "dtype")), dtype)

    def test_encoded_form_is_plain_data(self) -> None:
        encoded = _encode(torch.float16, "dtype")
        self.assertEqual(encoded, {"__torch_dtype__": "float16"})
        # Plain builtins only: the pickle stream imports no class at all.
        opcodes = {op.name for op, _, _ in pickletools.genops(pickle.dumps(encoded))}
        self.assertFalse(opcodes & {"GLOBAL", "STACK_GLOBAL"})

    def test_nested_in_a_dataclass(self) -> None:
        decoded = _decode(_encode(_HoldsDtype(dtype=torch.bfloat16), "cfg"))
        self.assertEqual(decoded, _HoldsDtype(dtype=torch.bfloat16))
