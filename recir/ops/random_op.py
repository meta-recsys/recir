# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""``recir::bernoulli_keyed`` -- dropout's Bernoulli draw with the PRNG key
derived from an explicit tensor argument.

torchax's ``aten.bernoulli`` reads its key off the ``Environment``
(``recir/torchax/tensor.py``) and rotates it in place. The
``Environment`` is neither an argument nor a return value of the traced
function, so nothing hung on it can become a graph placeholder.

``stochastic_round_fp16`` takes an FP32 update plus tensor seed/counter and
returns FP16 weights. Values within the finite FP16 range round to either
adjacent value in proportion to their distance. Out-of-range values and
NaN/Inf keep the ordinary FP16 conversion result.

This op takes the PRNG source as a **tensor**, a
registered buffer rides the ``stateless_func`` loop (``recir/api.py``)
-- placeholder in, graph output out, copied back by index -- so a tensor
argument is a real graph parameter. The data dependency is also what forces the
counter into the same lowered subgraph as the dropout, which matters because the
step is not reliably a single subgraph.

Contract::

    bernoulli_keyed(self, p, counter, site, base_seed) -> Tensor
      self:      the noise tensor (an ``empty_like`` of the dropout input);
                 supplies shape and dtype only, contents are ignored
      p:         KEEP probability, i.e. ``1 - dropout_rate``, matching
                 ``aten.bernoulli_.float``'s argument as emitted by
                 ``at::native::_dropout_impl``
      counter:   0-d integer tensor advanced once per step in-graph
      site:      static per-call-site index, so two dropouts in one step draw
                 independent masks
      base_seed: fixed, so a rerun reproduces the same masks

Returns a tensor of ``self``'s dtype (not bool): the caller is
``noise.div_(1 - p)`` followed by ``input * noise``.

Nodes calling this op are inserted post-trace by ``DropoutPRNGTransform``
(``recir/region_transforms/dropout_prng.py``), which runs on the already-joint
forward+backward graph -- hence no autograd registration. Dropout's backward is
the ``mul`` by the saved noise, which needs no gradient through the draw itself.
"""

import torch

_LIB = torch.library.Library("recir", "FRAGMENT")

_SCHEMA = (
    "bernoulli_keyed(Tensor self, float p, Tensor counter, int site, "
    "int base_seed) -> Tensor"
)
_LIB.define(_SCHEMA)


def _bernoulli_keyed_fallback(
    self: torch.Tensor,
    p: float,
    counter: torch.Tensor,
    site: int,
    base_seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=self.device)
    gen.manual_seed(
        (base_seed * 0x9E3779B1 + int(counter) * 0x85EBCA6B + site) % (2**63)
    )
    return torch.empty_like(self).bernoulli_(p, generator=gen)


_LIB.impl("bernoulli_keyed", _bernoulli_keyed_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::bernoulli_keyed")
def _bernoulli_keyed_meta(
    self: torch.Tensor,
    p: float,
    counter: torch.Tensor,
    site: int,
    base_seed: int,
) -> torch.Tensor:
    return torch.empty_like(self)


_LIB.define(
    "stochastic_round_fp16(Tensor value, Tensor counter, Tensor seed) -> Tensor"
)


def _stochastic_round_fp16_fallback(
    value: torch.Tensor, counter: torch.Tensor, seed: torch.Tensor
) -> torch.Tensor:
    """Round finite, in-range FP32 values to adjacent FP16 values without bias."""
    nearest = value.to(torch.float16)
    nearest_f32 = nearest.float()
    direction = torch.where(
        value > nearest_f32,
        torch.full_like(nearest, float("inf")),
        torch.full_like(nearest, -float("inf")),
    )
    neighbor = torch.nextafter(nearest, direction).float()
    gap = (neighbor - nearest_f32).abs()
    probability = (value - nearest_f32).abs() / torch.where(gap > 0, gap, 1)
    gen = torch.Generator(device=value.device)
    gen.manual_seed((int(seed) * 0x9E3779B1 + int(counter) * 0x85EBCA6B) % (2**63))
    uniform = torch.rand(
        value.shape, dtype=torch.float32, device=value.device, generator=gen
    )
    use_neighbor = (
        (uniform < probability) & torch.isfinite(value) & torch.isfinite(neighbor)
    )
    return torch.where(use_neighbor, neighbor, nearest_f32).to(torch.float16)


_LIB.impl(
    "stochastic_round_fp16",
    _stochastic_round_fp16_fallback,
    "CompositeExplicitAutograd",
)


@torch.library.register_fake("recir::stochastic_round_fp16")
def _stochastic_round_fp16_meta(
    value: torch.Tensor, counter: torch.Tensor, seed: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(value, dtype=torch.float16)
