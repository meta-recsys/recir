# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""Runtime patches for the pip-installed torchrec used by the TPU bench image.

apply_seq_a2a_int64_patch: fix an int32 offset overflow in torchrec's ROW_WISE
sequence-embedding output all-to-all.

The crash (multi-host 64-chip run, host-side SIGSEGV) is in
``All2All_Seq_Req.forward`` -> ``torch.ops.fbgemm.permute_2D_sparse_data``. That
fbgemm CPU kernel accumulates the per-segment offsets in the *lengths* dtype,
which is int32 for torchrec KJTs. When the flattened sequence-embedding tensor
exceeds 2**31 elements (reached at the large global batch of the 64-chip run, but
not the 512-batch 4-chip run), the int32 offset wraps negative and the kernel
dereferences an out-of-bounds pointer -> SEGV_MAPERR.

Fix: cast ``a2ai.lengths_after_sparse_data_all2all`` to int64 before the permute
so the kernel uses int64 offsets. The same ``a2ai`` is reused by the backward, so
the bwd permute is covered too. Also logs the element count (to confirm the
overflow threshold). Container uses pip torchrec, so this is applied at runtime
rather than as a source edit.
"""

import logging

import torch

logger: logging.Logger = logging.getLogger(__name__)

_INT32_MAX: int = 2**31
_PATCHED: bool = False
_LOG_BUDGET: int = 12


def apply_seq_a2a_int64_patch() -> None:
    """Monkeypatch torchrec ``All2All_Seq_Req.forward`` to use int64 offsets."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import torchrec.distributed.comm_ops as comm_ops
    except Exception as e:  # pragma: no cover - torchrec always present in image
        logger.warning("seq-a2a int64 patch skipped (torchrec import failed): %s", e)
        return

    cls = getattr(comm_ops, "All2All_Seq_Req", None)
    if cls is None or not hasattr(cls, "forward"):
        logger.warning("seq-a2a int64 patch skipped (All2All_Seq_Req.forward missing)")
        return

    orig_forward = cls.forward

    def patched_forward(ctx, pg, myreq, a2ai, sharded_input_embeddings):  # noqa: ANN001
        global _LOG_BUDGET
        numel = sharded_input_embeddings.numel()
        lengths = getattr(a2ai, "lengths_after_sparse_data_all2all", None)
        prev_dtype = None if lengths is None else lengths.dtype
        if lengths is not None and lengths.dtype != torch.int64:
            # Cast to int64 so fbgemm.permute_2D_sparse_data accumulates offsets
            # in int64 (the kernel's index_t follows the lengths dtype).
            a2ai.lengths_after_sparse_data_all2all = lengths.to(torch.int64)
        # Rate-limited: log the first few calls (the int32-overflow evidence),
        # plus always log any call that would have overflowed int32.
        if _LOG_BUDGET > 0 or numel >= _INT32_MAX:
            _LOG_BUDGET -= 1
            logger.warning(
                "[seq-a2a] embs.numel=%d exceeds_int32=%s lengths.dtype=%s->int64",
                numel,
                numel >= _INT32_MAX,
                prev_dtype,
            )
        return orig_forward(ctx, pg, myreq, a2ai, sharded_input_embeddings)

    cls.forward = staticmethod(patched_forward)
    _PATCHED = True
    logger.warning("Applied torchrec All2All_Seq_Req int64-lengths patch")
