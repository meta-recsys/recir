#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""RankFM model construction for eager and Recir-compiled execution."""

import contextlib
import logging
import os
from typing import Any, Callable, cast, Generator, Literal

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

AttentionBackend = Literal[
    "baseline",
    "pallas_padded",
    "pallas_jagged",
    "helion",
]
_ATTENTION_BACKEND_ENV = "RANKFM_MVAI_ATTENTION_BACKEND"
_ATTENTION_BACKENDS: frozenset[str] = frozenset(
    ("baseline", "pallas_padded", "pallas_jagged", "helion")
)


def resolve_attention_backend() -> AttentionBackend:
    """Resolve one deterministic HSTU implementation from the environment."""
    backend = os.environ.get(_ATTENTION_BACKEND_ENV, "")
    if not backend:
        return "baseline"
    if backend not in _ATTENTION_BACKENDS:
        supported = ", ".join(sorted(_ATTENTION_BACKENDS))
        raise ValueError(
            f"{_ATTENTION_BACKEND_ENV} must be one of {supported}; got {backend!r}"
        )
    return cast(AttentionBackend, backend)


def install_attention_backend(backend: AttentionBackend) -> None:
    """Restore Hammer's baseline attention, then install exactly one adapter."""
    import hammer.v2.ops.hstu_preprocess_and_attention as hstu_ops
    from hammer.ops.pytorch.ragged_hstu_attention import pytorch_hstu_mha

    if backend not in _ATTENTION_BACKENDS:
        supported = ", ".join(sorted(_ATTENTION_BACKENDS))
        raise ValueError(
            f"attention backend must be one of {supported}; got {backend!r}"
        )

    hstu_ops.pytorch_hstu_mha = pytorch_hstu_mha
    if backend == "pallas_padded":
        from recir.ops.hstu_attention_op import install_hstu_attention_swap

        install_hstu_attention_swap()
    elif backend == "pallas_jagged":
        from recir.ops.pallas_jagged_hstu_attention_op import (
            install_pallas_jagged_hstu_attention_swap,
        )

        install_pallas_jagged_hstu_attention_swap()
    elif backend == "helion":
        from recir.ops.helion_hstu_attention_op import (
            install_helion_hstu_attention_swap,
        )

        install_helion_hstu_attention_swap()


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
    attention_backend: AttentionBackend | None = None,
) -> BlueReelsVDDTrainV3:
    """Build the V3 xsurface model from a preconfigured ``module_config``.

    ``attention_backend`` selects the HSTU implementation explicitly. When omitted,
    ``RANKFM_MVAI_ATTENTION_BACKEND`` is resolved from the environment.

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

    backend = (
        resolve_attention_backend() if attention_backend is None else attention_backend
    )
    install_attention_backend(backend)
    if backend != "baseline":
        import recir.torchax.ops.jhstu_attention  # noqa: F401

    # Force pure PyTorch kernels (Triton kernels lack TPU support).
    model.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
    resolved_hammer_kernels: set[str] = set()
    for module in model.modules():
        hammer_kernel = getattr(module, "hammer_kernel", None)
        if callable(hammer_kernel):
            resolved_hammer_kernels.add(str(hammer_kernel()))
    logger.info(
        "[HSTU_KERNEL_SELECT] backend=%s RANKFM_MVAI_ATTENTION_BACKEND=%r "
        "hammer_kernel()=%r",
        backend,
        os.environ.get(_ATTENTION_BACKEND_ENV),
        sorted(resolved_hammer_kernels),
    )
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
