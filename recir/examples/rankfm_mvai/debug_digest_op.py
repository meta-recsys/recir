# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""``recir::debug_digest`` -- emit a parity digest from INSIDE the compiled graph.

An identity passthrough whose torchax lowering prints one digest
line per device from inside the compiled program, in the format
``parse_digest`` already reads. Same transport as ``recir::debug_ids``
(``recir/ops/debug_op.py``), no graph outputs, no extra host round trip beyond
the digest itself.

    x = torch.ops.recir.debug_digest(x, "weight", name, sharded)

Three things this file is careful about, each of which has a way of being
silently wrong:

1. **The emit must sit inside a ``jax.shard_map`` body.** recir's op-splice
   drops effects at the top level of an ``@op`` lowering, so a plain
   ``jax.debug.print`` / callback there is removed and the digest simply never
   appears (``recir/torchax/ops/jdebug.py:12-19``).

2. **The reductions run on the HOST, through ``digest_line``.** Not for
   convenience: ``jax_enable_x64`` is off in performance mode
   (``recir/torchax/__init__.py:157-160``), so an on-device ``astype(float64)``
   silently yields float32 and the leg's own l2 error lands in the comparison as
   a model difference. Calling the eager legs' emitter means the format and the
   float64 arithmetic are literally the same code, not a reimplementation of it.

3. **Nothing is reduced across the mesh.** Under ``GSPMDParallel("replicate")``
   the dense params, the loss and the activations are identical on every device;
   a ``psum`` would report N times the true value. Each device emits its own
   line under its own axis index, which is exactly the shape of light's
   per-DDP-rank digests, so ``parse_digest``'s ``(rank, kind, name)`` key pairs
   them directly.

``sharded`` tells the lowering which it is holding, because that cannot be
recovered here: a replicated tensor fed through ``P(axis)`` gets CUT, and each
device would digest 1/N of it while reporting a full-looking line.
"""

import hashlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import jax
import numpy as np
import torch
from hammer.v3.models.udd_fm_v3.parity.digest import digest_line, norm_param_name
from jax.sharding import Mesh, PartitionSpec as P
from recir.embedding.in_graph import SHARDED_TABLE_PREFIX
from recir.torchax.ops.decorators import op

_AXIS = "torch_dist"

logger: logging.Logger = logging.getLogger(__name__)

#: Modules whose call signature has already been logged (once each).
_ARGS_LOGGED: Set[str] = set()

_LIB = torch.library.Library("recir", "FRAGMENT")

_LIB.define(
    "debug_digest(Tensor x, str kind, str name, bool sharded, Tensor? offsets=None,"
    " int max_len=0, Tensor? num_candidates=None, Tensor? ro_lengths=None, int[] padded_sizes=[],"
    " int num_devices=0) -> Tensor"
)


def _drop_jagged_padding(
    x: torch.Tensor, offsets: Optional[torch.Tensor], max_len: int = 0
) -> torch.Tensor:
    """Keep only the rows belonging to REAL samples.

    Jagged layout: variable-length samples concatenated, delimited by
    ``offsets``. The slack-absorbing dummies are identified by LENGTH -- they
    are filled far past the per-sample bound every real sample is built to.
    """
    if offsets is None or x.ndim == 0 or offsets.numel() < 2 or max_len <= 0:
        return x
    offsets_tensor = offsets.reshape(-1).to(torch.int64)
    lengths = offsets_tensor[1:] - offsets_tensor[:-1]
    real = lengths <= max_len
    if bool(real.all()):
        return x
    indices = torch.cat(
        [
            torch.arange(int(offsets_tensor[sample]), int(offsets_tensor[sample + 1]))
            for sample in range(lengths.numel())
            if bool(real[sample])
        ]
    )
    return x[indices] if indices.numel() else x


def _history_tokens_per_sample(
    ro_lengths: torch.Tensor, num_candidates: torch.Tensor, padded_sizes: Sequence[int]
) -> torch.Tensor:
    """Per-sample history token counts, from the whole RO lengths tensor."""
    if ro_lengths.numel() == num_candidates.numel():
        raw_lengths = ro_lengths
    elif num_candidates.numel() == 0 or ro_lengths.numel() % num_candidates.numel():
        # Not [num_keys * B]; reshaping would raise INSIDE the host callback,
        # which is what takes a whole run down rather than producing a bad
        # digest. Skip the trim instead -- an untrimmed digest reads as a size
        # mismatch in the comparison, which is visible.
        return ro_lengths
    else:
        per_key = ro_lengths.reshape(-1, num_candidates.numel())
        sums = per_key.sum(dim=1)
        if "ro_lengths" not in _ARGS_LOGGED:
            _ARGS_LOGGED.add("ro_lengths")
            logger.info(
                "[digest] ro_lengths per-key sums (B=%d): %s",
                num_candidates.numel(),
                sums.tolist(),
            )
        raw_lengths = per_key[int(torch.argmax(sums).item())]
    slot = (
        padded_sizes[2] // padded_sizes[0]
        if len(padded_sizes) > 2 and padded_sizes[0]
        else 0
    )
    return torch.clamp(raw_lengths, max=slot) if slot else raw_lengths


def _drop_block_padding(
    x: torch.Tensor,
    num_candidates: Optional[torch.Tensor],
    ro_lengths: Optional[torch.Tensor],
    padded_sizes: Sequence[int],
    num_devices: int,
) -> torch.Tensor:
    """Trim whichever dim of ``x`` matches a known padded size.

    Args:
        x: the activation or gradient to trim.
        num_candidates: per-sample candidate (NRO) counts, shape ``[B]``. From
            the batch; ``batch_args[8]`` on the compiled path.
        ro_lengths: the RO id-list ``lengths``, shape ``[num_keys * B]`` -- the
            user-history sequence lengths, all keys concatenated, NOT the
            history itself. Reduced to per-sample token counts by
            ``_history_tokens_per_sample``.
        padded_sizes: the padded size of each kind of axis, indexed to match
            ``pairs`` below. Built by ``parity/step_digest.padded_extents``,
            which reads them off the model builder rather than the batch.
        num_devices: local chip count (``jax.device_count()``); the axis is cut
            into this many equal blocks.

    A padded axis is ``num_devices`` equal blocks with the real entries at the
    front of each. ``padded_sizes[i]`` is that axis kind's padded size and
    ``pairs[i]`` the per-sample count that says how much of it is real::

        i   padded_sizes[i]        pairs[i]
        0   the sample axis        one per sample
        1   the candidate axis     num_candidates
        2   history tokens         history_tokens
        3   history + candidates   history_tokens + num_candidates
        4   the TOKEN axis         history_tokens + num_candidates + contextual

    Worked example -- ``num_devices=2``, ``num_candidates=[1, 2, 1, 2]``,
    ``padded_sizes[1]=8``, ``x`` of shape ``(8, 2)``. Dim 0 is 8, which equals
    ``padded_sizes[1]``, so it is the candidate axis: 2 devices x 4 slots, 2
    samples each, so device 0 holds ``1+2=3`` real rows and device 1 ``1+2=3``::

        index    0    1    2    3  |  4    5    6    7
        content  r    r    r    .  |  r    r    r    .
                 \\___real___/  pad |  \\___real___/  pad

        -> index_select(0, [0, 1, 2, 4, 5, 6]), shape (6, 2)

    Pinned by ``DropBlockPaddingUnitTest.test_the_worked_example_in_the_docstring``.

    Matched by EXACT size, never by divisibility: a ``520x458752`` activation
    has a feature axis that divides evenly by the block count and exceeds any
    real total, and a heuristic would trim that instead.
    """
    if num_candidates is None or ro_lengths is None or x.ndim == 0 or num_devices <= 0:
        return x
    history_tokens = _history_tokens_per_sample(
        ro_lengths, num_candidates, padded_sizes
    )
    if history_tokens.numel() != num_candidates.numel():
        # _history_tokens_per_sample could not reduce the RO lengths to
        # per-sample counts and handed them back raw. Its whole reason for
        # doing that rather than raising is that an exception in the host
        # callback takes a run down; adding them to num_candidates below would
        # raise on the shape two lines later and undo exactly that. Skip the
        # trim, which reads as a size mismatch in the comparison -- visible,
        # and not fatal.
        return x
    # The contextual count is not passed separately -- it falls out of the two
    # widest padded_sizes, so the op keeps its signature.
    contextual = (
        (padded_sizes[4] - padded_sizes[3]) // padded_sizes[0]
        if len(padded_sizes) > 4 and padded_sizes[0]
        else 0
    )
    pairs = [
        torch.ones_like(num_candidates),
        num_candidates,
        history_tokens,
        history_tokens + num_candidates,
        history_tokens + num_candidates + contextual,
    ]
    for dim, size in enumerate(x.shape):
        for kind_index, counts in enumerate(pairs):
            if (
                kind_index >= len(padded_sizes)
                or size != padded_sizes[kind_index]
                or size % num_devices
            ):
                continue
            if counts.numel() % num_devices:
                continue
            samples_per_device = counts.numel() // num_devices
            slots_per_device = size // num_devices
            real_per_device = [
                int(
                    counts[
                        device * samples_per_device : (device + 1) * samples_per_device
                    ]
                    .sum()
                    .item()
                )
                for device in range(num_devices)
            ]
            if any(count > slots_per_device for count in real_per_device):
                # The counts do not describe this axis after all. Skipping
                # leaves an untrimmed digest, which the comparison reports as a
                # size mismatch; indexing past the block instead raised
                # "index out of range" inside the host callback and took the
                # whole run down.
                continue
            indices = torch.cat(
                [
                    torch.arange(
                        device * slots_per_device,
                        device * slots_per_device + real_per_device[device],
                    )
                    for device in range(num_devices)
                ]
            )
            return x.index_select(dim, indices) if indices.numel() else x
    return x


def _debug_digest_fallback(
    x: torch.Tensor,
    kind: str,
    name: str,
    sharded: bool,
    offsets: Optional[torch.Tensor] = None,
    max_len: int = 0,
    num_candidates: Optional[torch.Tensor] = None,
    ro_lengths: Optional[torch.Tensor] = None,
    padded_sizes: Optional[Sequence[int]] = None,
    num_devices: int = 0,
) -> torch.Tensor:
    """Eager/CPU: emit the digest directly and pass the tensor through."""
    if offsets is not None:
        # The same lines the compiled path emits, in the same order, so an eager
        # run and a compiled one are read the same way. Both are needed: the
        # offsets' own reductions cannot separate the two padded layouts, and
        # the lengths' absmax is what tells them apart.
        print(digest_line(kind, f"{name}.offsets", offsets), flush=True)
        if offsets.numel() > 1:
            lengths = (offsets.reshape(-1)[1:] - offsets.reshape(-1)[:-1]).to(
                torch.float64
            )
            print(digest_line(kind, f"{name}.lengths", lengths), flush=True)
    trimmed = _drop_block_padding(
        _drop_jagged_padding(x, offsets, max_len),
        num_candidates,
        ro_lengths,
        padded_sizes or [],
        num_devices,
    )
    print(digest_line(kind, name, trimmed), flush=True)
    return x


_LIB.impl("debug_digest", _debug_digest_fallback, "CompositeExplicitAutograd")


@torch.library.register_fake("recir::debug_digest")
def _debug_digest_meta(
    x: torch.Tensor,
    kind: str,
    name: str,
    sharded: bool,
    offsets: Optional[torch.Tensor] = None,
    max_len: int = 0,
    num_candidates: Optional[torch.Tensor] = None,
    ro_lengths: Optional[torch.Tensor] = None,
    padded_sizes: Optional[Sequence[int]] = None,
    num_devices: int = 0,
) -> torch.Tensor:
    # No print: this runs during the make_fx trace, where the values are fake.
    return torch.empty_like(x)


def _debug_digest_backward(
    ctx: Any, grad_out: torch.Tensor
) -> Tuple[torch.Tensor, None, None, None, None, None, None, None, None, None]:
    """Identity backward: the gradient passes straight through."""
    return grad_out, None, None, None, None, None, None, None, None, None


torch.library.register_autograd("recir::debug_digest", _debug_digest_backward)


def _device_mesh() -> Mesh:
    # Matches recir's Mesh(_get_global_devices(), ("torch_dist",)).
    return Mesh(jax.devices(), (_AXIS,))


def _to_torch(shard: Any) -> torch.Tensor:
    array = np.ascontiguousarray(np.asarray(shard))
    try:
        return torch.from_numpy(array)
    except TypeError:
        # ml_dtypes (bfloat16 and friends) have no torch equivalent here.
        return torch.from_numpy(array.astype(np.float32))


def _emit_on_host(
    kind: str,
    name: str,
    dtype_name: str,
    rank: Any,
    shard: Any,
    offsets: Any = None,
    max_len: int = 0,
    num_candidates: Any = None,
    ro_lengths: Any = None,
    padded_sizes: Any = None,
    num_devices: int = 0,
) -> None:
    """The host end of ``jax.debug.callback``, run once per device.

    The digest cannot live in the graph: which rows are real is data-dependent
    and XLA's shapes must stay static. So the op ships the shard and its padding
    metadata across, and this is what runs on the other side::

        device array -> torch -> drop padding -> reduce -> one digest line

    ``dtype_name`` is passed in rather than read off the tensor, because
    ``_to_torch`` widens bfloat16 to float32 to get it into numpy at all
    (lossless, and the reductions go to float64 anyway); the name keeps what the
    graph actually computed in.

    ``act`` and ``grad`` emit from rank 0 only. Both are replicated -- the hook
    digests activations whole so they pair with the eager leg's, and gradients
    are all-reduced before it sees them -- measured bit-identical across all
    eight ranks, not assumed. Emitting eightfold cost a run 4 -> 19 minutes
    before killing it, and eight ranks sharing one stream dropped 177 of 2224
    gradient lines unevenly, which reads as "these parameters have no gradient".
    Every other kind stays per-device, which is what makes a stray psum or a
    wrong PartitionSpec visible at all.
    """
    if kind.startswith(("act", "grad")) and int(rank) != 0:
        return
    offsets_tensor = None if offsets is None else _to_torch(offsets)
    if offsets_tensor is not None:
        print(
            digest_line(kind, f"{name}.offsets", offsets_tensor, rank=int(rank)),
            flush=True,
        )
        if offsets_tensor.numel() > 1:
            lengths = (offsets_tensor[1:] - offsets_tensor[:-1]).to(torch.float64)
            print(
                digest_line(kind, f"{name}.lengths", lengths, rank=int(rank)),
                flush=True,
            )
    trimmed = _drop_block_padding(
        _drop_jagged_padding(_to_torch(shard), offsets_tensor, max_len),
        None if num_candidates is None else _to_torch(num_candidates),
        None if ro_lengths is None else _to_torch(ro_lengths),
        list(padded_sizes or []),
        num_devices,
    )
    print(
        digest_line(kind, name, trimmed, rank=int(rank), dtype_name=dtype_name),
        flush=True,
    )


@op(torch.ops.recir.debug_digest)
def _jdebug_digest(
    x: Any,
    kind: str,
    name: str,
    sharded: bool,
    offsets: Any = None,
    max_len: int = 0,
    num_candidates: Any = None,
    ro_lengths: Any = None,
    padded_sizes: Any = None,
    num_devices: int = 0,
) -> Any:
    """Identity passthrough that emits one digest line per device."""
    dtype_name = str(x.dtype)
    # A replicated tensor must NOT be split: pass it whole to every device.
    spec = P(_AXIS, *([None] * (x.ndim - 1))) if sharded else P(*([None] * x.ndim))

    present = [
        tensor for tensor in (offsets, num_candidates, ro_lengths) if tensor is not None
    ]
    if present and sharded:
        raise ValueError(
            "debug_digest got padding metadata for a SHARDED tensor. It indexes "
            "the global row axis, so trimming a per-device shard with it cuts "
            "the wrong rows; digest replicated or drop the metadata."
        )
    flags = (offsets is not None, num_candidates is not None, ro_lengths is not None)

    def _per_device(x_device: Any, *extra_device: Any) -> Any:
        def _callback(rank: Any, shard: Any, *extra: Any) -> None:
            # Unpacked against the same flags the operands were built from.
            # None is never passed: jax treats it as an empty pytree node and
            # drops it, which silently shifts every later operand.
            remaining = list(extra)
            offsets_arg = remaining.pop(0) if flags[0] else None
            num_candidates_arg = remaining.pop(0) if flags[1] else None
            ro_lengths_arg = remaining.pop(0) if flags[2] else None
            _emit_on_host(
                kind,
                name,
                dtype_name,
                rank,
                shard,
                offsets_arg,
                max_len,
                num_candidates_arg,
                ro_lengths_arg,
                list(padded_sizes or []),
                num_devices,
            )

        jax.debug.callback(
            _callback, jax.lax.axis_index(_AXIS), x_device, *extra_device
        )
        return x_device

    in_specs = (spec, *(P(*([None] * tensor.ndim)) for tensor in present))
    return jax.shard_map(
        _per_device,
        mesh=_device_mesh(),
        in_specs=in_specs,
        out_specs=spec,
        check_vma=False,
    )(x, *present)


def debug_digest(
    x: torch.Tensor,
    kind: str,
    name: str,
    *,
    sharded: bool,
    offsets: Optional[torch.Tensor] = None,
    max_len: int = 0,
    num_candidates: Optional[torch.Tensor] = None,
    ro_lengths: Optional[torch.Tensor] = None,
    padded_sizes: Optional[Sequence[int]] = None,
    num_devices: int = 0,
) -> torch.Tensor:
    return torch.ops.recir.debug_digest(
        x,
        kind,
        name,
        sharded,
        offsets,
        max_len,
        num_candidates,
        ro_lengths,
        list(padded_sizes or []),
        num_devices,
    )


def _describe_args(args: Any, kwargs: Any) -> str:
    """One-line summary of what a module was called with."""

    def describe(value: Any) -> str:
        if isinstance(value, torch.Tensor):
            return f"T{list(value.shape)}:{str(value.dtype).replace('torch.', '')}"
        return type(value).__name__

    positional_text = ", ".join(describe(arg) for arg in args)
    kwargs_text = ", ".join(
        f"{key}={describe(value)}" for key, value in (kwargs or {}).items()
    )
    return f"({positional_text}){' {' + kwargs_text + '}' if kwargs_text else ''}"


def _jagged_offsets_index(args: Any, rows: int) -> Optional[int]:
    """The jagged offsets among a module's arguments, if it takes any."""
    for index, candidate in enumerate(args):
        if not isinstance(candidate, torch.Tensor):
            continue
        if candidate.ndim != 1 or candidate.dtype not in (torch.int32, torch.int64):
            continue
        if rows and candidate.shape[0] >= rows:
            continue
        return index
    return None


def param_is_sharded(name: str) -> bool:
    """Whether a parameter is dim-0 sharded across the mesh."""
    return SHARDED_TABLE_PREFIX in name


#: Restrict activation digests to modules whose FQN contains this substring.
#: Empty means every leaf module, which is what the eager legs capture.
#:
#: Exists because activation digests are the expensive kind: each is gathered
#: to the full global tensor (so it pairs with the eager leg's, which is not
#: sharded) and then sent to the host. The eager leg emits 226 of them; doing
#: that for every leaf here would dominate the step. When the question is
#: "where does the L1 stack diverge", "_hstu_opt" answers it for a fraction of
#: the cost.
DIGEST_ACT_SUBSTR_ENV: str = "PARITY_DIGEST_ACT_SUBSTR"


def _bundle_tensors(bundle: Any) -> List[Tuple[str, torch.Tensor]]:
    """(name, tensor) for every embedding in an EmbeddingBundle-like output."""
    embeddings = getattr(bundle, "_embeddings", None)
    if not isinstance(embeddings, dict):
        return []
    pairs: List[Tuple[str, torch.Tensor]] = []
    for descriptor, tensor in embeddings.items():
        if isinstance(tensor, torch.Tensor):
            pairs.append((str(getattr(descriptor, "name", descriptor)), tensor))
    return pairs


def _probe_grad(
    tensor: Any,
    name: str,
    counter: str,
    next_index: Callable[[str], int],
    num_candidates: Optional[torch.Tensor],
    ro_lengths: Optional[torch.Tensor],
    padded_sizes: Optional[Sequence[int]],
    num_devices: int,
) -> None:
    """Digest the gradient of ``tensor`` when it arrives, under ``name``."""
    if not (isinstance(tensor, torch.Tensor) and tensor.requires_grad):
        return
    tensor.register_hook(
        lambda grad_tensor, probe_key=name, call_key=counter: debug_digest(
            grad_tensor,
            f"gradout[{next_index(call_key + ':' + probe_key)}]",
            probe_key,
            sharded=False,
            num_candidates=num_candidates,
            ro_lengths=ro_lengths,
            padded_sizes=padded_sizes,
            num_devices=num_devices,
        )
    )


def digest_activations_in_graph(
    model: torch.nn.Module,
    kinds: Set[str],
    num_candidates: Optional[torch.Tensor] = None,
    ro_lengths: Optional[torch.Tensor] = None,
    padded_sizes: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Digest module inputs and outputs, from inside the compiled graph."""
    log_source_fingerprint()
    if "act" not in kinds and "gradout" not in kinds:
        return []
    num_devices = jax.device_count()
    name_substring = os.environ.get(DIGEST_ACT_SUBSTR_ENV, "")
    handles: List[Any] = []
    seen: Dict[str, int] = {}

    def _next(key: str) -> int:
        arg_index = seen.get(key, 0)
        seen[key] = arg_index + 1
        return arg_index

    def make_pre(name: str) -> Callable[[Any, Any, Any], Any]:
        key = norm_param_name(f"IN:{name}")
        bare_name = norm_param_name(name)

        def pre_hook(_module: Any, args: Any, kwargs: Any) -> Any:
            if "gradout" in kinds and isinstance(args, tuple) and args:
                _probe_grad(
                    args[0],
                    f"GRADIN:{bare_name}",
                    "gi",
                    _next,
                    num_candidates,
                    ro_lengths,
                    padded_sizes,
                    num_devices,
                )
            if "act" not in kinds:
                return None
            if not (isinstance(args, tuple) and args):
                return None
            first = args[0]
            if not isinstance(first, torch.Tensor):
                return None
            if key not in _ARGS_LOGGED:
                _ARGS_LOGGED.add(key)
                logger.info(
                    "[digest] %s called with %s", key, _describe_args(args, kwargs)
                )
            offsets_index = _jagged_offsets_index(args, first.shape[0])
            offsets = None if offsets_index is None else args[offsets_index]
            max_len = next(
                (
                    arg
                    for arg in args[1:]
                    if isinstance(arg, int) and not isinstance(arg, bool)
                ),
                0,
            )
            trailing = list(args[1:])
            for arg_index, arg in enumerate(trailing, start=1):
                if (
                    isinstance(arg, torch.Tensor)
                    and arg.ndim == 0
                    and arg.is_floating_point()
                ):
                    trailing[arg_index - 1] = debug_digest(
                        arg,
                        f"act[{_next(key + f'.arg{arg_index}')}]",
                        f"{key}.arg{arg_index}",
                        sharded=False,
                    )
            digested = debug_digest(
                first,
                f"act[{_next(key)}]",
                key,
                sharded=False,
                offsets=offsets,
                max_len=max_len,
                num_candidates=num_candidates,
                ro_lengths=ro_lengths,
                padded_sizes=padded_sizes,
                num_devices=num_devices,
            )
            return (digested, *trailing), kwargs

        return pre_hook

    def make_post(name: str) -> Callable[[Any, Any, Any], Any]:
        key = norm_param_name(name)

        def post_hook(_module: Any, _args: Any, out: Any) -> Any:
            target = out
            if isinstance(out, (tuple, list)):
                target = next(
                    (element for element in out if isinstance(element, torch.Tensor)),
                    None,
                )
            if "gradout" in kinds:
                _probe_grad(
                    target,
                    key,
                    "go",
                    _next,
                    num_candidates,
                    ro_lengths,
                    padded_sizes,
                    num_devices,
                )
            if "act" not in kinds or not isinstance(target, torch.Tensor):
                return None
            if target is not out:
                digested = debug_digest(
                    target,
                    f"act[{_next(key)}]",
                    key,
                    sharded=False,
                    num_candidates=num_candidates,
                    ro_lengths=ro_lengths,
                    padded_sizes=padded_sizes,
                    num_devices=num_devices,
                )
                items = [digested if element is target else element for element in out]
                if isinstance(out, list):
                    return items
                return type(out)(*items) if hasattr(out, "_fields") else tuple(items)
            return debug_digest(
                out,
                f"act[{_next(key)}]",
                key,
                sharded=False,
                num_candidates=num_candidates,
                ro_lengths=ro_lengths,
                padded_sizes=padded_sizes,
                num_devices=num_devices,
            )

        return post_hook

    for name, module in model.named_modules():
        if not name or list(module.children()):
            continue
        if name_substring and name_substring not in name:
            continue
        handles.append(
            module.register_forward_pre_hook(make_pre(name), with_kwargs=True)
        )
        handles.append(module.register_forward_hook(make_post(name)))

    if "gradout" in kinds:

        def make_bundle(name: str) -> Callable[[Any, Any, Any], None]:
            base = norm_param_name(name)

            def bundle_hook(_module: Any, _args: Any, out: Any) -> None:
                for bundle_name, bundle_tensor in _bundle_tensors(out):
                    _probe_grad(
                        bundle_tensor,
                        f"{base}#{bundle_name}",
                        "gb",
                        _next,
                        num_candidates,
                        ro_lengths,
                        padded_sizes,
                        num_devices,
                    )

            return bundle_hook

        for name, module in model.named_modules():
            if not name or (name_substring and name_substring not in name):
                continue
            handles.append(module.register_forward_hook(make_bundle(name)))
    return handles


def digest_grads_via_hooks(
    module: torch.nn.Module,
    kinds: Set[str],
    sink: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Any]:
    """Digest gradients through tensor hooks. Returns the handles to remove."""
    log_source_fingerprint()
    if "grad" not in kinds:
        return []

    def make(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        sharded = param_is_sharded(name)
        fqn = norm_param_name(name)

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            out = debug_digest(grad, "grad", fqn, sharded=sharded)
            if sink is not None:
                sink[f"grad/{name}"] = out.detach().clone()
            return out

        return hook

    return [
        param.register_hook(make(name))
        for name, param in module.named_parameters()
        if param.requires_grad
    ]


def _source_fingerprint() -> str:
    """First 12 hex of a hash over the files that shape the digest.

    Pod-side code only reaches the cluster through the image, and forgetting to
    rebuild is not an error: the pod pulls the previous image, runs it, and the
    log looks like a clean run of code that was never deployed. Two cluster
    rounds were spent reading such a log before this line existed.

    Covers the STEP as well as this module -- an earlier version hashed only
    this file, so a change to the instrumentation call sites left the
    fingerprint identical and the check blind to exactly the staleness it is
    for. Compare against the devserver with:

        python3 -c "import hashlib;print(hashlib.sha256(b''.join(open(f,'rb')\
.read() for f in FILES)).hexdigest()[:12])"
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    udd_fm_v3_dir = os.path.normpath(
        os.path.join(
            module_dir, "..", "..", "..", "hammer", "v3", "models", "udd_fm_v3"
        )
    )
    hammer = os.path.normpath(os.path.join(module_dir, "..", "..", "..", "hammer"))
    files = [
        os.path.abspath(__file__),
        os.path.join(module_dir, "..", "..", "training", "compile.py"),
        os.path.join(udd_fm_v3_dir, "parity", "digest.py"),
        os.path.join(hammer, "v2", "modules", "hstu_common.py"),
        os.path.join(hammer, "v2", "modules", "hstu_truncate.py"),
        os.path.join(
            hammer, "v3", "modules", "mot", "compute_blocks", "mot_stustack.py"
        ),
        os.path.join(module_dir, "..", "..", "embedding", "in_graph.py"),
        os.path.join(module_dir, "..", "..", "torchax", "tensor.py"),
        os.path.normpath(
            os.path.join(
                module_dir,
                "..",
                "..",
                "..",
                "hammer",
                "v3",
                "modules",
                "mot",
                "tokenizers",
                "embedding_composer_tokenizer.py",
            )
        ),
        os.path.normpath(
            os.path.join(
                module_dir,
                "..",
                "..",
                "..",
                "hammer",
                "v3",
                "modules",
                "mot",
                "preprocessors",
                "mot_preprocessor.py",
            )
        ),
    ]
    hasher = hashlib.sha256()
    for path in files:
        try:
            with open(path, "rb") as file_handle:
                hasher.update(file_handle.read())
        except OSError:
            hasher.update(b"missing")
    return hasher.hexdigest()[:12]


_FINGERPRINT_LOGGED: bool = False


def log_source_fingerprint() -> None:
    """Log the fingerprint once, the first time the digest is armed.

    Not at import: reading these files is real I/O, and this module is imported
    on paths that never arm anything. Latched, because both entry points call
    it and the pod log wants one line, not one per step.
    """
    global _FINGERPRINT_LOGGED
    if _FINGERPRINT_LOGGED:
        return
    _FINGERPRINT_LOGGED = True
    logger.info("[debug_digest_op] source=%s", _source_fingerprint())
