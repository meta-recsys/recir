#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""RankFM model construction for eager and Recir-compiled execution."""

import contextlib
import logging
import os
from typing import Any, Callable, Generator

import torch
from hammer.ops.bf16_train_utils import set_autocast_factory
from hammer.utils import HammerKernel
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import (
    BlueReelsVDDTrainV3,
    build_recblock_from_xsurface_config,
)
from hammer.v3.modules.mot.tokenizers.mot_sparse_arch import ECSparseArch
from recir.torchax.amp import hammer_autocast_factory

logger: logging.Logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _ec_sparse_arch_on(device: torch.device) -> Generator[None, None, None]:
    """Force ``ECSparseArch`` to build its embedding tables on ``device``.

    ``ECSparseArch.__init__`` defaults to ``torch.device("meta")`` when no
    ``device=`` is passed by the caller. Letting it stay on meta would force
    every test to call ``model.to_empty(device)`` afterwards to materialize
    the tables, which in turn blows away every parameter and non-persistent
    buffer in the model.

    Patching the default at construction time lets the entire model build
    correctly on CPU in one pass, so no post-hoc re-init is needed.
    """
    original_init: Callable[..., None] = ECSparseArch.__init__

    def patched_init(self: ECSparseArch, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("device", device)
        original_init(self, *args, **kwargs)

    ECSparseArch.__init__ = patched_init  # pyre-ignore[8]
    try:
        yield
    finally:
        ECSparseArch.__init__ = original_init


def build_model_from_config(
    module_config: BlueReelsVDDModuleConfig,
    device: torch.device,
    bf16_training: bool = False,
    embeddings_on_meta: bool = False,
    use_helion: bool = False,
) -> BlueReelsVDDTrainV3:
    """Build the V3 xsurface model from a preconfigured ``module_config``.

    When ``use_helion`` is True, the HSTU attention leaf is rebound to the recir
    Helion adapter (``install_helion_hstu_attention_swap()``) so the STU traces
    attention as the single ``recir::helion_ragged_attention_fwd`` node the torchax
    lowering runs on the Helion kernels. Eager numerics are unchanged -- the op's
    fallback is the captured original attention.

    ``embeddings_on_meta``: when True, the ``ECSparseArch`` embedding tables are
    built on the ``meta`` device (no host allocation / init) while the dense
    modules stay on ``device``. Use this for the in_graph + on_device_init path,
    where ``InGraphEmbeddingBackend.setup()`` replaces the entire embedding
    collection with fresh on-device (meta -> HBM) tables, so the vanilla host
    tables are never read. Without this, an uncapped hash size (e.g. 100M rows)
    would allocate + ``normal_()``-init ~100GB per table on the host,
    single-threaded, stalling model construction before compile.
    """
    for stack in module_config.stu_stack_configs:
        for stu_config in stack.stu_configs:
            stu_config.attn_config = stu_config.attn_config._replace(
                use_cuda_attention=False
            )

    # Sequence-length sampling off, unconditionally. NOT a choice -- the recir
    # trace does not survive it, so it does not belong on the CLI where it reads
    # like one. Prod turns the L1 sampler on and derives
    # output_segments=["unsampled_inputs"] from it as a matched pair
    # (..._xsurface_config.py:1664-1665), but mot_preprocessor derives its flags
    # PER HSTU STACK (:800-807) against a single global output_segments, and
    # prod's L2 sampler is None. The L2 stack therefore wants the segment while
    # nothing produces it, its `elif supervision_weight is not None` fallback
    # (:987-997) does not fire, and the FX trace dies in
    # udd_fm_train_v3._post_processing with KeyError('unsampled_inputs').
    #
    # BOTH fields, together: clearing only the sampler leaves the segment
    # requested with no producer, which is the same crash.
    #
    # Mutated here rather than in config.build_module_config because it has to
    # be true for every caller that traces -- and because it belongs next to the
    # dropout zeroing below, as the other thing the compiled path cannot do yet.
    # TODO: need to fix later
    module_config.hstu_seq_len_sampling_configs = None
    module_config.output_segments = None

    ec_device = torch.device("meta") if embeddings_on_meta else device
    with _ec_sparse_arch_on(ec_device):
        recblock = build_recblock_from_xsurface_config(
            module_config=module_config,
            is_train=True,
            custom_kernel=False,
        )
        model = BlueReelsVDDTrainV3(
            module_config=module_config,
            recblock=recblock,
            is_train=True,
        )
    if not embeddings_on_meta:
        model.to(device)
    # When embeddings_on_meta: the dense modules are already built on ``device``
    # (torch default) and the embedding tables are intentionally left on meta --
    # calling model.to(device) here would raise "Cannot copy out of meta tensor".
    # The in_graph backend materializes real on-device tables in setup().

    # Install the RecIR operation behind Hammer's PyTorch attention seam.
    if os.environ.get("RECIR_HSTU_PALLAS", "0") == "1":
        import recir.torchax.ops.jhstu_attention  # noqa: F401
        from recir.ops.hstu_attention_op import install_hstu_attention_swap

        install_hstu_attention_swap()
    # Force pure PyTorch kernels (Triton kernels lack TPU support).
    model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    resolved_hammer_kernels: set[str] = set()
    for module in model.modules():
        hammer_kernel = getattr(module, "hammer_kernel", None)
        if callable(hammer_kernel):
            resolved_hammer_kernels.add(str(hammer_kernel()))
    logger.info(
        "[HSTU_KERNEL_SELECT] RECIR_HSTU_PALLAS=%r; resolved hammer_kernel()=%r",
        os.environ.get("RECIR_HSTU_PALLAS"),
        sorted(resolved_hammer_kernels),
    )
    if use_helion:
        from recir.ops.helion_hstu_attention_op import (
            install_helion_hstu_attention_swap,
        )

        install_helion_hstu_attention_swap()
    # Route the shared hammer.autocast() call sites to torchax AMP on this
    # TPU/JAX build (torch.autocast("cuda", ...) is inert without CUDA tensors).
    # Installed unconditionally here; the per-site enabled gate (_bf16_training)
    # is what controls whether bf16 actually engages.
    set_autocast_factory(hammer_autocast_factory)
    # bf16: default False keeps the historical fp32 TPU behavior. When True, the
    # HSTU L1/L2 blocks + composer + loss run bf16 (the explicit activation casts
    # plus hammer.autocast -> torchax AMP at the same sites the GPU uses).
    model.recursive_setattr("_bf16_training", bf16_training)
    return model
