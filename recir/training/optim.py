# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.


"""Tensor-LR optimizers + warmup controller for the recir-compiled train step.

Both optimizers read their LR from a 0-d Tensor instead of a Python float so
external scheduler code can mutate the tensor in place between compiled-step
calls without triggering a recompile. AdamW does this via PyTorch's
capturable=True path; Adagrad implements the update with tensor ops because
addcdiv_(value=...) is Scalar-only.

The optimizer instances are discovered by recir's compile path via pytree
leaves and threaded through the parallel mode's param sharding; their step()s
are captured inside the compiled GM.

`WarmupController` mirrors torchrec.optim.warmup.WarmupOptimizer's per-step LR
mutation against the 0-d LR buffers, without the KeyedOptimizer layout.

Depends only on torch, recir and torchrec. Optimizer configs are read through
the ``OptimConfig`` protocol.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch
from recir.utils.meta_tensor_init import mark_meta_zero_init
from torchrec.distributed.embedding_types import OptimType
from torchrec.optim.clipping import GradientClipping
from torchrec.optim.warmup import _lr_stages, WarmupPolicy, WarmupStage

logger: logging.Logger = logging.getLogger(__name__)


class OptimConfig(Protocol):
    """The optimizer-config fields read by this package."""

    @property
    def optim_type(self) -> OptimType: ...

    @property
    def lr(self) -> float: ...

    @property
    def eps(self) -> float: ...

    @property
    def beta1(self) -> float: ...

    @property
    def beta2(self) -> float: ...

    @property
    def weight_decay(self) -> float: ...

    @property
    def gradient_clipping(self) -> GradientClipping: ...

    @property
    def max_gradient(self) -> float: ...

    @property
    def lr_warmup_schedule(self) -> Sequence[WarmupStage]: ...


class TensorLRAdamW(torch.optim.AdamW):
    """torch.optim.AdamW with a runtime-mutable tensor LR and optional
    prod-equivalent value-mode gradient clipping.

    Tensor LR: ``lr_tensor`` is a 0-d tensor whose IDENTITY is captured by the
    GM (recir treats it as a runtime placeholder). External scheduler code
    mutates LR via ``lr_tensor.fill_(new)``. Requires ``capturable=True`` so
    AdamW's step math handles tensor lr/step end-to-end.

    NOT ``fused=True``: with capturable=True the dispatcher picks
    ``aten._fused_adamw_.tensor_lr`` which has no upstream fake kernel and can't
    be Python-registered (aten's TORCH_LIBRARY lacks set_python_module). Unblock
    by landing a fake kernel upstream, then re-enable fused=True.

    Value clip: when ``max_grad`` is set, runs clip_grad_value_ over all owned
    grads before the standard step (mirrors GradientClippingOptimizer.step).
    """

    def __init__(
        self,
        params: Any,
        *,
        lr_tensor: torch.Tensor,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        max_grad: Optional[float] = None,
    ) -> None:
        assert lr_tensor.dim() == 0, (
            f"lr_tensor must be 0-d (got shape {tuple(lr_tensor.shape)})"
        )
        # AdamW.__init__ asserts 0.0 <= lr against the value passed, so it must
        # be a Python number at construction; we overwrite with the tensor below
        # so step() sees a tensor. capturable=True but NOT fused=True.
        super().__init__(
            params,
            lr=float(lr_tensor.item()),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            capturable=True,
        )
        for group in self.param_groups:
            group["lr"] = lr_tensor
        self._max_grad: Optional[float] = max_grad

    def step(self, closure: Any = None) -> Any:
        if self._max_grad is not None:
            all_params: List[torch.nn.Parameter] = [
                p for group in self.param_groups for p in group["params"]
            ]
            torch.nn.utils.clip_grad_value_(all_params, self._max_grad)
        return super().step(closure)


class TensorLRAdam(torch.optim.Adam):
    """torch.optim.Adam with a runtime-mutable tensor LR and optional
    prod-equivalent value-mode gradient clipping.

    The Adam counterpart of :class:`TensorLRAdamW`, identical to it in every
    respect except the base class -- and NOT interchangeable with it at nonzero
    weight_decay, because Adam folds weight decay into the gradient while AdamW
    decouples it.

    See :class:`TensorLRAdamW` for why capturable=True and not fused=True.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr_tensor: torch.Tensor,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        max_grad: Optional[float] = None,
    ) -> None:
        assert lr_tensor.dim() == 0, (
            f"lr_tensor must be 0-d (got shape {tuple(lr_tensor.shape)})"
        )
        # Adam.__init__ validates lr as a number, so pass the value and swap in
        # the tensor afterwards -- same dance as TensorLRAdamW.
        super().__init__(
            params,
            lr=float(lr_tensor.item()),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            capturable=True,
        )
        for group in self.param_groups:
            group["lr"] = lr_tensor
        self._max_grad: Optional[float] = max_grad

    def step(self, closure: Any = None) -> Any:
        if self._max_grad is not None:
            all_params: List[torch.nn.Parameter] = [
                p for group in self.param_groups for p in group["params"]
            ]
            torch.nn.utils.clip_grad_value_(all_params, self._max_grad)
        return super().step(closure)


class TensorLRRowWiseAdagrad(torch.optim.Optimizer):
    """Per-row Adagrad with runtime-mutable tensor LR. Mirrors
    torchrec.optim.RowWiseAdagrad's per-row state shape ``(num_rows, 1)``.

    Update rule (2-D embedding weights):
        state['sum'] += (grad ** 2).mean(axis=1, keepdim=True)  # (rows, 1)
        std = state['sum'].sqrt() + eps                          # (rows, 1)
        param -= (grad / std) * lr_tensor                        # broadcast

    Differences from torchrec.optim.RowWiseAdagrad: drops lr_decay, the step
    counter (avoids a host sync that would graph-break under make_fx), and
    maximize; replaces addcdiv_(value=-clr) with explicit tensor arithmetic
    (value= is Scalar-only, can't take a tensor LR). Restricted to 2-D params.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr_tensor: torch.Tensor,
        eps: float,
        weight_decay: float,
    ) -> None:
        assert lr_tensor.dim() == 0, (
            f"lr_tensor must be 0-d (got shape {tuple(lr_tensor.shape)})"
        )
        defaults: Dict[str, Any] = {
            "lr": lr_tensor,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr_t: torch.Tensor = group["lr"]
            eps: float = group["eps"]
            wd: float = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise ValueError(
                        "TensorLRRowWiseAdagrad expects 2-D embedding params; "
                        f"got param of shape {tuple(p.shape)}. Route 1-D / "
                        "non-embedding params through a per-element optimizer."
                    )
                grad = p.grad
                state = self.state[p]
                if "sum" not in state:
                    # Per-row state: (num_rows, 1). Matches prod RowWiseAdagrad.
                    state["sum"] = torch.zeros(
                        (p.shape[0], 1), dtype=torch.float32, device=p.device
                    )
                    mark_meta_zero_init(state["sum"])
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                state["sum"] += grad.pow(2).mean(axis=1, keepdim=True)
                std = state["sum"].sqrt().add_(eps)
                # Cannot use addcdiv_(value=lr_t) (Scalar-only); explicit math.
                p.sub_(grad / std * lr_t)
        return loss


def _run_adam_init_step(opt: torch.optim.Optimizer) -> bool:
    """Run the zero-gradient step used by ``KeyedOptimizer.init_state``.

    Returns ``False`` when parameters are meta and only zero-decay state can be
    materialized structurally. Nonzero weight decay must update the parameter,
    which is impossible before it has storage.
    """
    groups = list(opt.param_groups)
    parameters = [p for group in groups for p in group["params"] if p.requires_grad]
    if any(p.is_meta for p in parameters):
        if any(float(group.get("weight_decay", 0.0)) != 0.0 for group in groups):
            raise RuntimeError(
                "Adam state priming with nonzero weight decay requires "
                "materialized parameters so the production dummy-step update "
                "can be applied"
            )
        return False

    saved_groups: List[Tuple[Dict[str, Any], Any, Any]] = []
    saved_grads = [(p, p.grad) for p in parameters]
    try:
        for group in groups:
            lr = group["lr"]
            saved_groups.append((group, lr, group.get("capturable")))
            group["lr"] = float(lr.item()) if isinstance(lr, torch.Tensor) else lr
            if "capturable" in group:
                group["capturable"] = False
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        opt.step()
    finally:
        for parameter, grad in saved_grads:
            parameter.grad = grad
        for group, lr, capturable in saved_groups:
            group["lr"] = lr
            if capturable is not None:
                group["capturable"] = capturable

    for group in groups:
        if not group.get("capturable", False):
            continue
        for parameter in group["params"]:
            state = opt.state.get(parameter)
            if state and isinstance(state.get("step"), torch.Tensor):
                state["step"] = state["step"].to(parameter.device)
    return True


def prime_optimizer_state(opt: torch.optim.Optimizer) -> None:
    """Pre-allocate per-param optimizer state before the recir trace.

    Torch optimizers create their state lazily inside the first ``step()``. In
    the recir path that step runs inside the traced graph, while recir snapshots
    ``opt.state`` at trace time. State created during the trace would therefore
    be a discarded graph intermediate instead of compiled carried state.

    torchrec materializes the same state by calling
    ``KeyedOptimizer.init_state()``, which assigns zero gradients and runs a
    real ``step()``. We run that step too: besides advancing the counter, it
    applies configured weight decay (and for Adam/L2, updates the moments).
    With zero weight decay the moments stay zero and the counter is the only
    difference from an unprimed optimizer; its bias corrections differ by

        step 1:  m_hat/sqrt(v_hat) = 1.000
        step 2:  0.1/(1-0.9^2) / sqrt(0.001/(1-0.999^2)) = 0.744

    so every dense parameter took a first update 34% larger than production's.
    Measured against the eager leg at 0.743955 against 0.744137 predicted.
    """
    try:
        from torch.optim.optimizer import _get_scalar_dtype

        scalar_dtype = _get_scalar_dtype()
    except Exception:  # pragma: no cover - defensive across torch versions
        scalar_dtype = torch.float32

    is_adam = isinstance(opt, (torch.optim.Adam, torch.optim.AdamW))
    is_rowwise = isinstance(opt, TensorLRRowWiseAdagrad)
    if not (is_adam or is_rowwise):
        return

    if is_adam and _run_adam_init_step(opt):
        return

    primed = 0
    for group in opt.param_groups:
        capturable = bool(group.get("capturable", False))
        amsgrad = bool(group.get("amsgrad", False))
        for p in group["params"]:
            state = opt.state[p]
            if state:
                continue
            if is_adam:
                step_device = p.device if capturable else torch.device("cpu")
                state["step"] = torch.ones((), dtype=scalar_dtype, device=step_device)
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            else:
                if p.dim() != 2:
                    continue
                state["sum"] = torch.zeros(
                    (p.shape[0], 1), dtype=torch.float32, device=p.device
                )
                mark_meta_zero_init(state["sum"])
            primed += 1
    logger.info(
        "[recir optimizer] primed %s state for %d params (pre-trace)",
        type(opt).__name__,
        primed,
    )


def build_torch_optimizer(
    params: List[torch.nn.Parameter],
    cfg: OptimConfig,
    lr_tensor: torch.Tensor,
) -> torch.optim.Optimizer:
    """Construct a torch.optim instance from MVAI's OptimConfig.

    Postcondition, no exceptions: every ``param_group["lr"]`` of the returned
    optimizer IS ``lr_tensor`` -- the same 0-d tensor object, not a copy and
    not a float. That is what lets the WarmupController drive the LR by
    mutating the buffer in place, and what lets make_fx trace against a lifted
    proxy instead of a baked constant. ``recir_train_step`` asserts this before
    tracing, so any branch added here has to preserve it.

    ``lr_tensor``'s initial value must equal ``cfg.lr``.

    Value-mode grad clipping (GradientClipping.VALUE + max_gradient) is honored
    for the Adam/AdamW branch. Other clipping modes and FBGEMM-only knobs are
    dropped.
    """
    if cfg.optim_type == OptimType.SGD:
        raise NotImplementedError(
            "OptimType.SGD is not supported on the recir path: torch's SGD "
            "cannot take a tensor LR in eager, so it cannot honor the "
            "tensor-LR contract build_torch_optimizer guarantees. Use ADAMW, "
            "ADAM or ROWWISE_ADAGRAD, or add a TensorLRSGD that binds "
            "lr_tensor into every param_group."
        )
    if cfg.optim_type in (OptimType.ADAMW, OptimType.ADAM):
        name = cfg.optim_type.name
        max_grad: Optional[float] = None
        if cfg.gradient_clipping == GradientClipping.VALUE:
            max_grad = cfg.max_gradient
            logger.info(
                "[recir optimizer] %s with GradientClipping.VALUE @ "
                "max_gradient=%g (prod-parity).",
                name,
                cfg.max_gradient,
            )
        elif cfg.gradient_clipping != GradientClipping.NONE:
            logger.warning(
                "[recir optimizer] %s gradient_clipping=%s is not yet "
                "supported in the recir path; running without clipping.",
                name,
                cfg.gradient_clipping,
            )
        # Adam and AdamW differ in how weight decay enters the update (L2 vs
        # decoupled), so the config's choice is honoured rather than collapsed
        # onto one of them.
        cls = TensorLRAdamW if cfg.optim_type == OptimType.ADAMW else TensorLRAdam
        opt = cls(
            params,
            lr_tensor=lr_tensor,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            max_grad=max_grad,
        )
        prime_optimizer_state(opt)
        return opt
    if cfg.optim_type == OptimType.ROWWISE_ADAGRAD:
        if cfg.gradient_clipping != GradientClipping.NONE:
            logger.warning(
                "[recir optimizer] ROWWISE_ADAGRAD gradient_clipping=%s is not "
                "yet supported in the recir path; running without clipping.",
                cfg.gradient_clipping,
            )
        logger.info(
            "[recir optimizer] ROWWISE_ADAGRAD: using TensorLRRowWiseAdagrad "
            "with per-row state shape (num_rows, 1) - prod-parity for sparse "
            "embeddings."
        )
        opt = TensorLRRowWiseAdagrad(
            params,
            lr_tensor=lr_tensor,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
        prime_optimizer_state(opt)
        return opt
    raise NotImplementedError(
        f"OptimType {cfg.optim_type} is not yet mapped to a torch.optim class. "
        "Add a case to build_torch_optimizer."
    )


_TRACEABLE_POLICIES: Tuple[WarmupPolicy, ...] = (
    WarmupPolicy.NONE,
    WarmupPolicy.LINEAR,
)


def _assert_traceable_stages(stages: List[WarmupStage]) -> None:
    """Reject schedules this module cannot compute as tensor ops."""
    for stage in stages:
        if stage.policy not in _TRACEABLE_POLICIES:
            raise NotImplementedError(
                f"WarmupPolicy.{stage.policy.name} has no tensor implementation. "
                f"The warmup schedule runs inside the compiled step, so every "
                f"stage must be expressible as tensor arithmetic on the 0-d "
                f"iteration counter. Supported: "
                f"{[p.name for p in _TRACEABLE_POLICIES]}."
            )


def _stage_multiplier(stages: List[WarmupStage], iter_t: torch.Tensor) -> torch.Tensor:
    """The LR multiplier for ``iter_t``, as a traceable 0-d tensor.

    Mirrors ``torchrec.optim.warmup._get_multiplier``: every stage contributes,
    including NONE, and every result is scaled by ``stage.lr_scale``. Skipping
    a policy would leave a later stage's ``iter_t <= max_iters`` branch to cover
    its window, applying the wrong curve there.
    """
    it = iter_t.to(torch.float32)
    mult = torch.ones_like(it)
    # ``_lr_stages`` orders stages by increasing ``max_iters`` and appends a
    # terminal NONE. Walking backwards means the earliest stage whose bound
    # still covers ``iter_t`` writes last, i.e. wins.
    for stage in reversed(stages):
        if stage.policy == WarmupPolicy.LINEAR:
            here = stage.value + (1.0 - stage.value) * it / float(stage.max_iters)
        else:
            here = torch.ones_like(it)
        here = here * float(stage.lr_scale)
        # The terminal stage carries ``max_iters = 1 << 63``, one past int64,
        # so comparing against it directly would overflow.
        bound = min(int(stage.max_iters), torch.iinfo(torch.int64).max)
        mult = torch.where(iter_t <= bound, here, mult)
    return mult


class WarmupController:
    """Drives runtime-mutable LR over warmup stages for the dense + sparse opts.

    Owns the warmup iteration counter and per-bucket stage cursors, and writes
    the current-iter LR into the two 0-d LR buffers in place (so the compiled GM
    picks it up on the next invocation without a recompile). Mirrors
    torchrec.optim.warmup.WarmupOptimizer semantics; ``_lr_stages`` appends a
    terminal NONE-policy stage so reads past the last stage hold at base_lr.

    Lifecycle: construct, call :meth:`apply` once to seed iter 0, then call
    :meth:`step` after each compiled step to advance to the next iter's LR.
    The 0-d buffers are owned by the caller (registered on the module) and
    passed in by reference so fill_() preserves the traced identity.
    """

    def __init__(
        self,
        *,
        dense_base_lr: float,
        dense_lr_warmup_schedule: Optional[List[Any]],
        dense_lr_buffer: torch.Tensor,
        iter_buffer: torch.Tensor,
        sparse_base_lr: Optional[float] = None,
        sparse_lr_warmup_schedule: Optional[List[Any]] = None,
        sparse_lr_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        # Sparse side is optional: the TPU (GSPMD) cell has no in-GM sparse torch
        # optimizer (the embedding backend owns the sparse update off-GM), so it
        # constructs a dense-only controller (no sparse buffer / schedule).
        self._dense_base_lr = dense_base_lr
        self._sparse_base_lr: float = (
            sparse_base_lr if sparse_base_lr is not None else 0.0
        )
        self._dense_stages: List[WarmupStage] = _lr_stages(
            list(dense_lr_warmup_schedule or [])
        )
        self._sparse_stages: List[WarmupStage] = _lr_stages(
            list(sparse_lr_warmup_schedule or [])
        )
        self._dense_lr_buffer = dense_lr_buffer
        self._sparse_lr_buffer: Optional[torch.Tensor] = sparse_lr_buffer
        self._iter_buffer: torch.Tensor = iter_buffer
        _assert_traceable_stages(self._dense_stages)
        _assert_traceable_stages(self._sparse_stages)

    @property
    def iter(self) -> int:
        return int(self._iter_buffer)

    @property
    def num_dense_stages(self) -> int:
        return len(self._dense_stages)

    @property
    def num_sparse_stages(self) -> int:
        return len(self._sparse_stages)

    def state_dict(self) -> Dict[str, int]:
        return {"iter": self.iter}

    def load_state_dict(self, state_dict: Mapping[str, int]) -> None:
        iteration = state_dict["iter"]
        if iteration < 0:
            raise ValueError(f"Warmup iteration must be non-negative, got {iteration}")
        self.set_iter(iteration)

    def set_iter(self, iteration: int) -> None:
        self._iter_buffer.fill_(iteration)
        self.apply()

    def apply(self) -> None:
        self._dense_lr_buffer.copy_(
            self._dense_base_lr
            * _stage_multiplier(self._dense_stages, self._iter_buffer)
        )
        # Sparse side only when a sparse buffer was supplied (GPU). Dense-only
        # controllers (TPU) skip it entirely.
        sparse_lr_buffer = self._sparse_lr_buffer
        if sparse_lr_buffer is None:
            return
        sparse_lr_buffer.copy_(
            self._sparse_base_lr
            * _stage_multiplier(self._sparse_stages, self._iter_buffer)
        )

    def step(self) -> None:
        self._iter_buffer.add_(1)
        self.apply()
