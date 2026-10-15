# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Public exports for the recir QAT pass.

Quantization-aware training for embedding tables (dense QAT coming later).
The joint fwd+bwd+opt graph traced by ``functorch.make_fx`` captures both the
fake-quant forward and the STE backward when QAT is injected pre-trace via
:func:`install_qat` + :func:`qat_fake_quant`.

Example::

    from recir.quant import QATConfig, install_qat, qat_fake_quant

    cfg = QATConfig(target_table_names={"table_a"}, n_bits=8)
    recir.compile(
        parallel_mode=parallel_mode,
        backend=backend,
        eager_preproc=[install_qat(cfg, optimizer)],
        _parametrize_fns=[qat_fake_quant(cfg)],
    )(train_step)
"""

from recir.quant.config import QATConfig
from recir.quant.eager_preproc import (
    finalize_qat_calibration,
    install_qat,
    set_qat_calibration,
)
from recir.quant.fake_quant import fake_quantize_ste, STEFakeQuant
from recir.quant.parametrize import qat_fake_quant

__all__ = [
    "QATConfig",
    "STEFakeQuant",
    "fake_quantize_ste",
    "finalize_qat_calibration",
    "install_qat",
    "qat_fake_quant",
    "set_qat_calibration",
]
