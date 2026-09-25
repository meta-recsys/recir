#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""RankFM xsurface tracing stubs and eager dispatchers."""

import contextlib
import logging
import threading
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hammer.v3.modules.core.features.bundle import FeatureBundle
from hammer.v3.modules.core.features.descriptor import FeatureDescriptor
from hammer.v3.modules.core.features.input_features import InputFeatures
from hammer.v3.modules.core.preprocessors.preprocessor_stack import PreprocessorStack
from hammer.v3.modules.mot.preprocessors.mot_preprocessor import MoTPreprocessor
from hammer.v3.modules.recblock import _inputs_to_feature_bundle, RecBlock
from recir.eager_dispatch import (
    install_graph_break_dispatch,
    mark_input,
    mark_output_with_bwd,
)
from recir.examples.rankfm_mvai.graph_input_replay import (
    build_dispatch_input_replay,
    DispatchInputReplay,
)
from recir.examples.rankfm_mvai.preproc_pipeline import PreprocPipeline
from recir.region_transforms.padding import (
    _per_chip_real_counts,
    pad_candidate_axis_to_static,
    pad_jagged_pair_to_static,
    runtime_num_local,
    trim_per_chip_to_size,
)
from torchrec import JaggedTensor, KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

XSURFACE_PREPROCESSOR_MARKER_PREFIX = "recir.examples.rankfm.preprocessor"


def _xsurface_preprocessor_prefix(override_id: int) -> str:
    return f"{XSURFACE_PREPROCESSOR_MARKER_PREFIX}::{override_id}"


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

TEST_BATCH_SIZE: int = 4
TEST_MAX_UIH_SEQ_LEN: int = 32
TEST_NUM_CANDIDATES: int = 8

# Sequential RO key used as the basis for deriving SEQ_LENGTHS at the
# override boundary: SEQ_LENGTHS[i] = HSTU_ID_LIST[i][SEQ_LENGTHS_BASIS_KEY].lengths
# + num_candidates.
SEQ_LENGTHS_BASIS_KEY: str = "datafm_fb_trait_post_id"


@contextlib.contextmanager
def _disable_sampling(real_stack: PreprocessorStack) -> Generator[None, None, None]:
    """Temporarily null out ``_uih_sampling_module`` on every
    ``FeaturePreprocessor`` under ``real_stack.mot_preprocessor`` so the
    cached output build is deterministic and represents the upper bound.
    """
    # pyre-ignore[16]: dynamic attr from PreprocessorStack setattr
    preprocessors = list(real_stack.mot_preprocessor._hstu_preprocessor_list)
    saved: List[Any] = []
    for hp in preprocessors:
        fp = hp._feature_preprocessor
        saved.append(fp._uih_sampling_module)
        fp._uih_sampling_module = None
    try:
        yield
    finally:
        for hp, s in zip(preprocessors, saved):
            hp._feature_preprocessor._uih_sampling_module = s


def _fit_to_shape(
    t: torch.Tensor, target_shape: torch.Size | Tuple[int, ...]
) -> torch.Tensor:
    """End-pad with zeros (or narrow) each dim of *t* to ``target_shape``."""
    target = tuple(int(s) for s in target_shape)
    if tuple(t.shape) == target:
        return t
    pad_widths: List[int] = []
    for actual, tgt in zip(reversed(t.shape), reversed(target)):
        pad_widths += [0, max(0, int(tgt) - int(actual))]
    out = F.pad(t, pad_widths) if any(p > 0 for p in pad_widths) else t
    for d, (actual, tgt) in enumerate(zip(out.shape, target)):
        if int(actual) > int(tgt):
            out = out.narrow(d, 0, int(tgt))
    return out


def _assert_within_budget(
    current_lengths: torch.Tensor,
    keys: List[str],
    per_key_budgets: List[int],
    num_processes: int,
    kjt_name: str,
    count: Optional[torch.Tensor] = None,
) -> None:
    num_keys = len(keys)
    b_stride = int(current_lengths.shape[0] // num_keys)
    num_local = runtime_num_local()
    rc = _per_chip_real_counts(b_stride, num_local, count)
    key_lengths = current_lengths.view(num_keys, b_stride)
    violations: List[str] = []
    for k, (key, budget_global) in enumerate(zip(keys, per_key_budgets)):
        budget = int(budget_global) // num_processes // num_local
        row_off = 0
        for chip in range(num_local):
            real = int(key_lengths[k][row_off : row_off + rc[chip]].sum().item())
            row_off += rc[chip]
            if real > budget:
                violations.append(
                    f"  {kjt_name}/{key} chip {chip}: {real} real tokens vs "
                    f"{budget} budgeted ({real / max(budget, 1):.1f}x)"
                )
    if violations:
        raise ValueError(
            "[static-shape] token budget exceeded; the padder would drop the "
            "excess by zeroing trailing samples of the chip block, not by "
            "trimming evenly:\n"
            + "\n".join(violations)
            + "\nTurn on --budgeted_seq_len_sampling so the sampler holds the "
            "per-chip total under the budget."
        )


def _pad_kjt_to_cached(
    current_values: torch.Tensor,
    current_lengths: torch.Tensor,
    keys: List[str],
    cached_lengths_buf: torch.Tensor,
    num_processes: int,
    kjt_name: str,
    count: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad an output KJT to the per-host cached shape.

    ``count`` is the per-sample candidate count (``num_candidates``). When the
    KJT's per-host stride equals ``sum(count)`` the KJT is CANDIDATE-axis: its
    rows follow their parent samples and are NOT chip-uniform, so the pad ops
    split the stride per chip by ``count`` (matching PRT's ``pad_count_to_sum``).
    A sample-axis KJT (stride == num samples, divisible by num_local) uses the
    uniform split (``count=None``). Auto-detected so the caller need not know
    each output KJT's granularity.
    """

    num_keys = len(keys)
    global_stride = int(cached_lengths_buf.shape[0] // num_keys)  # global batch_size
    cached_per_key_global = (
        cached_lengths_buf.view(num_keys, global_stride).sum(dim=1).tolist()
    )  # global cached length sum for each key
    per_key_budgets = [int(b) for b in cached_per_key_global]
    # Candidate-axis iff the per-host stride matches the candidate total.
    b_stride = int(current_lengths.shape[0] // num_keys)
    eff_count = (
        count if (count is not None and int(count.sum().item()) == b_stride) else None
    )
    _assert_within_budget(
        current_lengths, keys, per_key_budgets, num_processes, kjt_name, eff_count
    )
    return pad_jagged_pair_to_static(
        current_values,
        current_lengths,
        per_key_budgets,
        global_stride,
        num_processes=num_processes,
        count=eff_count,
        tile_pad=True,
    )


# ---------------------------------------------------------------------------
# Dummy modules
# ---------------------------------------------------------------------------


class DummyPreprocessorStack(nn.Module):
    """Replacement for PreprocessorStack that returns fixed-shape outputs.

    Runs the real preprocessor eagerly at construction time to capture
    input/output shapes.  Wraps inputs/outputs in
    :func:`recir.eager_dispatch.mark_input` / ``mark_output_with_bwd`` so the
    post-trace orchestrator can find the markers and wire runtime dispatch to
    the real :class:`PreprocessorStack`.

    A _grad_anchor parameter maintains autograd graph connectivity through
    the dummy so that make_fx traces backward correctly. A _batch_dep term
    adds data dependency on inputs so GSPMD can infer sharding.
    """

    def __init__(  # noqa: C901 -- one linear capture pass over the sample bundle: inputs, then output tensors / KJTs / exports. The branches are per-value-kind and read as a table; splitting them would scatter one structure-capture across four helpers.
        self,
        real_stack: PreprocessorStack,
        device: torch.device,
        batch_size: int = TEST_BATCH_SIZE,
        max_uih_seq_len: int = TEST_MAX_UIH_SEQ_LEN,
        num_candidates: int = TEST_NUM_CANDIDATES,
        override_id: int = 0,
        sample_feature_bundle: FeatureBundle | None = None,
        precomputed_output: FeatureBundle | None = None,
    ) -> None:
        super().__init__()
        self._override_id: int = override_id
        self._grad_anchor = nn.Parameter(torch.zeros(1))

        if sample_feature_bundle is not None:
            input_features = sample_feature_bundle
        else:
            # pyre-ignore
            preprocessor: MoTPreprocessor = real_stack.mot_preprocessor
            sample_bundle: FeatureBundle = preprocessor.sample_inputs(
                batch_size=batch_size,
                device=device,
                max_uih_seq_len=max_uih_seq_len,
                num_candidates=num_candidates,
            )
            ro_id_list = sample_bundle.require_kjt(InputFeatures.UIH_ID_LIST)
            nro_id_list = sample_bundle.require_kjt(InputFeatures.CANDIDATE_ID_LIST)
            num_cand = sample_bundle.require_tensor(InputFeatures.NUM_CANDIDATES)
            # pyrefly: ignore [bad-assignment]
            all_labels: Dict[str, torch.Tensor] = sample_bundle.require(
                InputFeatures.ALL_LABELS
            )
            input_features = _inputs_to_feature_bundle(
                ro_id_list, None, nro_id_list, num_cand, all_labels
            )
        if precomputed_output is not None:
            cached_output = precomputed_output
        else:
            # Sampling off — see _disable_sampling doc-string for why.
            with _disable_sampling(real_stack):
                cached_output: FeatureBundle = real_stack(input_features)

        # Capture INPUT structure for fwd_fn reconstruction at runtime.
        entries: List[Tuple[FeatureDescriptor, str, Any]] = []
        nontensor_map: Dict[FeatureDescriptor, Any] = {}
        for desc in input_features:
            val = input_features.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                entries.append((desc, "kjt", val.keys()))
            elif isinstance(val, torch.Tensor):
                entries.append((desc, "tensor", None))
            elif isinstance(val, dict):
                entries.append((desc, "dict", sorted(val.keys())))
            else:
                nontensor_map[desc] = val
        self._input_entries: List[Tuple[FeatureDescriptor, str, Any]] = entries
        self._input_nontensor_map: Dict[FeatureDescriptor, Any] = nontensor_map

        # Capture OUTPUT structure and register tensors as buffers.
        self._tensor_descs: List[FeatureDescriptor] = []
        self._kjt_descs: List[FeatureDescriptor] = []
        self._kjt_keys: List[List[str]] = []
        self._nontensor_map: Dict[FeatureDescriptor, Any] = {}
        tensor_idx = 0
        kjt_idx = 0

        for desc in cached_output:
            val = cached_output.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                self._kjt_descs.append(desc)
                self._kjt_keys.append(val.keys())
                self.register_buffer(f"_kjt_values_{kjt_idx}", val.values())
                self.register_buffer(f"_kjt_lengths_{kjt_idx}", val.lengths())
                kjt_idx += 1
            elif isinstance(val, torch.Tensor):
                self._tensor_descs.append(desc)
                self.register_buffer(f"_tensor_{tensor_idx}", val)
                tensor_idx += 1
            else:
                self._nontensor_map[desc] = val

        self._export_keys: List[str] = []
        for e_key, e_val in sorted(cached_output.exports.items()):
            if not isinstance(e_val, torch.Tensor):
                continue
            # Index off the dense key list, not enumerate(): skipping a
            # non-tensor export would otherwise leave a hole that the dense
            # lookup in forward reads past.
            self.register_buffer(f"_export_{len(self._export_keys)}", e_val)
            self._export_keys.append(e_key)

        self._num_tensors: int = tensor_idx
        self._num_kjts: int = kjt_idx

    def forward(self, features: FeatureBundle) -> FeatureBundle:  # noqa: C901
        oid = self._override_id
        prefix = _xsurface_preprocessor_prefix(oid)

        # Mark input tensor leaves with input markers.  Iterate the actual
        # features bundle (not the sample-captured lists) so the ordering
        # matches what the pre-hook captures at trace time.  This ensures
        # the marker indices align with fwd_fn's reconstruction order.
        input_idx = 0
        anchored_inputs: List[torch.Tensor] = []
        for desc in features:
            val = features.get(desc)
            if val is None:
                continue
            if isinstance(val, KeyedJaggedTensor):
                anchored_inputs.append(mark_input(val.values(), prefix, input_idx))
                input_idx += 1
                anchored_inputs.append(mark_input(val.lengths(), prefix, input_idx))
                input_idx += 1
            elif isinstance(val, torch.Tensor):
                anchored_inputs.append(mark_input(val, prefix, input_idx))
                input_idx += 1
            elif isinstance(val, dict):
                for key in sorted(val.keys()):
                    if isinstance(val[key], torch.Tensor):
                        anchored_inputs.append(mark_input(val[key], prefix, input_idx))
                        input_idx += 1
        output_tensors: List[torch.Tensor] = []
        for i in range(self._num_tensors):
            buf = getattr(self, f"_tensor_{i}")
            output_tensors.append(torch.zeros_like(buf))
        for i in range(self._num_kjts):
            vbuf = getattr(self, f"_kjt_values_{i}")
            lbuf = getattr(self, f"_kjt_lengths_{i}")
            output_tensors.append(torch.zeros_like(vbuf))
            output_tensors.append(lbuf.clone())
        for i in range(len(self._export_keys)):
            output_tensors.append(torch.zeros_like(getattr(self, f"_export_{i}")))

        deps: List[torch.Tensor] = anchored_inputs + [self._grad_anchor]
        anchored = [
            mark_output_with_bwd(t, prefix, idx, deps=deps)
            for idx, t in enumerate(output_tensors)
        ]

        # Reconstruct FeatureBundle from anchored tensors.
        bundle = FeatureBundle()
        out_idx = 0
        for desc in self._tensor_descs:
            bundle.set(desc, anchored[out_idx], validate=False)
            out_idx += 1
        for i, desc in enumerate(self._kjt_descs):
            kjt = KeyedJaggedTensor(
                keys=self._kjt_keys[i],
                values=anchored[out_idx],
                lengths=anchored[out_idx + 1],
            )
            bundle.set(desc, kjt, validate=False)
            out_idx += 2
        for desc, v in self._nontensor_map.items():
            bundle.set(desc, v, validate=False)
        for key in self._export_keys:
            bundle.set_export(key, anchored[out_idx])
            out_idx += 1
        return bundle


# ---------------------------------------------------------------------------
# Dispatcher classes (orchestrator-style overrides)
# ---------------------------------------------------------------------------


class PreprocessorDispatcher:
    """Orchestrator-style replacement for the legacy ``PreprocessorOverride``.

    ``install_eager`` swaps :class:`PreprocessorStack` for
    :class:`DummyPreprocessorStack` so tracing sees stable shapes (the dummy
    plants input + forward/backward markers).  ``install_graph`` finds those
    markers and installs a graph-break dispatch back to the captured real
    :class:`PreprocessorStack`.

    The input FeatureBundle structure is captured at trace time via the
    dummy's stored ``_input_entries``, ensuring the captured structure
    matches what RecBlock actually passes (including supervision_weight and
    the correct KJT keys from the batch, not the sample).
    """

    def __init__(
        self,
        device: torch.device,
        batch_size: int = TEST_BATCH_SIZE,
        max_uih_seq_len: int = TEST_MAX_UIH_SEQ_LEN,
        num_candidates: int = TEST_NUM_CANDIDATES,
        override_id: int = 0,
        num_ranks: int = 1,
        sample_feature_bundle: FeatureBundle | None = None,
        precomputed_output: FeatureBundle | None = None,
        num_processes: int = 1,
        skip_backward_break: bool = False,
    ) -> None:
        self._device = device
        self._batch_size = batch_size
        self._max_uih_seq_len = max_uih_seq_len
        self._num_candidates = num_candidates
        self._override_id = override_id
        self._num_ranks = num_ranks
        self._orig_stack: nn.Module | None = None
        self._dummy: DummyPreprocessorStack | None = None
        self._sample_feature_bundle = sample_feature_bundle
        self._precomputed_output = precomputed_output
        # Per-host fitting: cached buffers are at GLOBAL stride (override-
        # build sample is global), runtime outputs are per-host
        # (= cached.shape with dim 0 // num_processes) so _to_sharded_jax
        # can expand back to global at the XLA boundary.
        self._num_processes: int = num_processes
        # Only safe where the embedding backend registers no backward of its
        # own; see the in_graph call site in rankfm_embedding_backends.py.
        self._skip_backward_break: bool = skip_backward_break

        # --- one-step-ahead pipelining (off unless enable_pipelining runs) ---
        self._pipeline: Optional[PreprocPipeline[Tuple[torch.Tensor, ...]]] = None
        self._flat_step_args_fn: Optional[Callable[[Any], List[Any]]] = None
        self._replay: Optional[DispatchInputReplay] = None
        self._produce: Optional[Callable[..., Tuple[torch.Tensor, ...]]] = None
        # (gm, dispatch_id, num_state_tensors) -- what build_dispatch_input_replay takes.
        self._replay_build_args: Optional[Tuple[torch.fx.GraphModule, int, int]] = None
        # Set by begin_step, read by the callback.
        self._step: int = 0
        self._fwd_calls_this_step: int = 0
        self._batch: Optional[Any] = None
        self._next_step: int = 0
        self._next_batch: Optional[Any] = None
        self._consumer: Optional[Callable[[], None]] = None
        # The preprocessor stack is shared, and on a miss the callback runs it
        # on the main thread while the worker may be running it for the next
        # step. Serializing the degraded path costs nothing it was not already
        # paying.
        self._produce_lock: threading.Lock = threading.Lock()

    def enable_pipelining(
        self,
        flat_step_args_fn: Callable[[Any], List[Any]],
    ) -> None:
        self._flat_step_args_fn = flat_step_args_fn
        if self._pipeline is None:
            self._pipeline = PreprocPipeline(name="preproc-producer")
        build_args = self._replay_build_args
        if build_args is not None and self._replay is None:
            self._replay = build_dispatch_input_replay(*build_args)

    def begin_step(self, step: int, batch: Any, next_batch: Optional[Any]) -> None:
        self._step = step
        self._fwd_calls_this_step = 0
        self._batch = batch
        self._next_step = step + 1
        self._next_batch = next_batch

    def set_consumer(self, consumer: Optional[Callable[[], None]]) -> None:
        if consumer is not None and self._consumer is not None:
            raise RuntimeError(
                "the previous step's consume never ran; the graph break did "
                "not fire, and its work would be dropped"
            )
        self._consumer = consumer

    def drain_producer(self) -> None:
        """Block until worker-side work has finished."""
        if self._pipeline is not None:
            self._pipeline.drain()

    def close_pipelining(self) -> None:
        if self._pipeline is not None:
            logger.info(
                f"[PIPE] producer hits={self._pipeline.hits} "
                f"misses={self._pipeline.misses}"
            )
            self._pipeline.close()
            self._pipeline = None

    def install_eager(self, model: nn.Module) -> None:
        for fqn, mod in list(model.named_modules()):
            if isinstance(mod, PreprocessorStack):
                self._orig_stack = mod
                self._dummy = DummyPreprocessorStack(
                    real_stack=mod,
                    device=self._device,
                    batch_size=self._batch_size,
                    max_uih_seq_len=self._max_uih_seq_len,
                    num_candidates=self._num_candidates,
                    override_id=self._override_id,
                    sample_feature_bundle=self._sample_feature_bundle,
                    precomputed_output=self._precomputed_output,
                )
                parent_fqn, leaf = fqn.rsplit(".", 1) if "." in fqn else ("", fqn)
                parent = model.get_submodule(parent_fqn) if parent_fqn else model
                setattr(parent, leaf, self._dummy)

    def install_graph(  # noqa: C901
        self,
        gm: torch.fx.GraphModule,
        flat_state: List[torch.Tensor],
    ) -> Tuple[torch.fx.GraphModule, List[torch.Tensor]]:
        if self._orig_stack is None or self._dummy is None:
            return gm, flat_state

        real_stack: nn.Module = self._orig_stack
        dummy: DummyPreprocessorStack = self._dummy
        num_ranks: int = self._num_ranks
        num_processes_fit: int = self._num_processes
        # Per-host -> per-device split degree at lowering (np.split(num_local)).
        # = local devices per host. Used to lay out jagged-total supervision
        # outputs CHIP-MAJOR so the split lands on per-chip blocks.
        num_local_fit: int = runtime_num_local()

        # Input structure captured at construction time from sample_feature_bundle.
        entries: List[Tuple[FeatureDescriptor, str, Any]] = dummy._input_entries
        input_nontensor_map: Dict[FeatureDescriptor, Any] = dummy._input_nontensor_map

        output_tensor_descs: List[FeatureDescriptor] = dummy._tensor_descs
        output_kjt_descs: List[FeatureDescriptor] = dummy._kjt_descs

        def _produce(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            bundle = FeatureBundle()
            idx = 0
            for desc, type_tag, extra in entries:
                if type_tag == "kjt":
                    kjt = KeyedJaggedTensor(
                        keys=extra,
                        values=input_tensors[idx],
                        lengths=input_tensors[idx + 1],
                    )
                    bundle.set(desc, kjt, validate=False)
                    idx += 2
                elif type_tag == "tensor":
                    bundle.set(desc, input_tensors[idx], validate=False)
                    idx += 1
                elif type_tag == "dict":
                    d: Dict[str, torch.Tensor] = {}
                    for key in extra:
                        d[key] = input_tensors[idx]
                        idx += 1
                    bundle.set(desc, d, validate=False)
            for desc, v in input_nontensor_map.items():
                bundle.set(desc, v, validate=False)

            num_nro_candidates_padded = bundle.get_tensor(InputFeatures.NUM_CANDIDATES)
            real_num_candidates: Optional[torch.Tensor] = None
            if num_nro_candidates_padded is not None:
                ro_stride = bundle.require_kjt(InputFeatures.UIH_ID_LIST).stride()
                real_num_candidates = trim_per_chip_to_size(
                    num_nro_candidates_padded,
                    ro_stride,
                    num_local_fit,
                )
                bundle.set(
                    InputFeatures.LOCAL_NUM_CANDIDATES,
                    real_num_candidates,
                    validate=False,
                )

            with torch.enable_grad():
                output = real_stack(bundle)

            # Fit each runtime output to the PER-HOST shape (cached shape
            # with dim 0 // num_processes).  Cached buffers are at global
            # (override-build sample is at global) but real_stack at runtime
            # processes per-host data and produces per-host outputs.  We
            # return per-host so _to_sharded_jax can expand to global —
            # matching the per-host behavior of PRT pad ops at runtime.
            def _per_host_target(cached_shape: torch.Size) -> Tuple[int, ...]:
                out = [int(s) for s in cached_shape]
                if len(out) > 0:
                    out[0] = out[0] // num_processes_fit
                return tuple(out)

            seq_basis_lengths: Optional[torch.Tensor] = None
            flat_kjts: List[torch.Tensor] = []
            kjt_i = 0
            for desc in output_kjt_descs:
                cached_v = getattr(dummy, f"_kjt_values_{kjt_i}")
                cached_l = getattr(dummy, f"_kjt_lengths_{kjt_i}")
                target_v = _per_host_target(cached_v.shape)
                target_l = _per_host_target(cached_l.shape)
                num_keys = len(dummy._kjt_keys[kjt_i])
                val = output.get(desc)
                if val is not None and isinstance(val, KeyedJaggedTensor):
                    fitted_v, fitted_l = _pad_kjt_to_cached(
                        val.values(),
                        val.lengths(),
                        dummy._kjt_keys[kjt_i],
                        cached_l,
                        num_processes_fit,
                        count=real_num_candidates,
                        kjt_name=desc.name,
                    )
                    if (
                        desc.name == "merged"
                        and SEQ_LENGTHS_BASIS_KEY in dummy._kjt_keys[kjt_i]
                    ):
                        basis_key_idx = dummy._kjt_keys[kjt_i].index(
                            SEQ_LENGTHS_BASIS_KEY
                        )
                        per_host_stride = fitted_l.shape[0] // num_keys
                        seq_basis_lengths = fitted_l.view(num_keys, per_host_stride)[
                            basis_key_idx
                        ]
                    flat_kjts.append(fitted_v)
                    flat_kjts.append(fitted_l)
                else:
                    flat_kjts.append(
                        torch.zeros(
                            target_v, dtype=cached_v.dtype, device=cached_v.device
                        )
                    )
                    flat_kjts.append(
                        torch.zeros(
                            target_l, dtype=cached_l.dtype, device=cached_l.device
                        )
                    )
                kjt_i += 1

            # Additional-embedding (values, lengths) form a JAGGED PAIR; the
            # preprocessor just emits them as two separate tensor outputs rather
            # than a KJT. Pad them through the SAME _pad_kjt_to_cached path the
            # real KJTs use (num_keys=1).
            addemb_fitted: Dict[str, torch.Tensor] = {}
            addemb_pairs: Dict[str, Dict[str, int]] = {}
            for ti, d_desc in enumerate(output_tensor_descs):
                if "/additional_emb_input/" not in d_desc.name:
                    continue
                if d_desc.name.endswith("/values"):
                    addemb_pairs.setdefault(d_desc.name[: -len("/values")], {})["v"] = (
                        ti
                    )
                elif d_desc.name.endswith("/lengths"):
                    addemb_pairs.setdefault(d_desc.name[: -len("/lengths")], {})[
                        "l"
                    ] = ti
            for pair in addemb_pairs.values():
                if "v" not in pair or "l" not in pair:
                    continue
                v_desc = output_tensor_descs[pair["v"]]
                l_desc = output_tensor_descs[pair["l"]]
                v_val = output.get(v_desc)
                l_val = output.get(l_desc)
                if not (
                    isinstance(v_val, torch.Tensor) and isinstance(l_val, torch.Tensor)
                ):
                    continue
                cached_v = getattr(dummy, f"_tensor_{pair['v']}")
                cached_l = getattr(dummy, f"_tensor_{pair['l']}")
                fitted_v, fitted_l = _pad_kjt_to_cached(
                    v_val,
                    l_val,
                    [v_desc.name],
                    cached_l,
                    num_processes_fit,
                    count=real_num_candidates,
                    kjt_name="addemb",
                )
                addemb_fitted[v_desc.name] = fitted_v.to(cached_v.dtype)
                addemb_fitted[l_desc.name] = fitted_l.to(cached_l.dtype)

            # Now process tensor outputs; SEQ_LENGTHS gets derived inline
            # as first_key_lengths(HSTU_ID_LIST[i]) + num_candidates.
            flat_tensors: List[torch.Tensor] = []
            # Local (pre-pad) tensor values by name -- used to derive the
            # max_sum_contextual constant (LWC - SEQ_LENGTHS) when rebuilding
            # LENGTHS_WITH_CONTEXTUAL padded below.
            _local_by_name: Dict[str, Any] = {
                d.name: output.get(d) for d in output_tensor_descs
            }
            tensor_i = 0
            for desc in output_tensor_descs:
                cached_t = getattr(dummy, f"_tensor_{tensor_i}")
                target_shape = _per_host_target(cached_t.shape)
                val = output.get(desc)
                if val is not None and isinstance(val, torch.Tensor):
                    if desc.name in addemb_fitted:
                        # Additional-embedding jagged pair, already fitted above
                        # via the shared KJT path (_pad_kjt_to_cached).
                        fitted = addemb_fitted[desc.name]
                    elif (
                        desc.name.endswith("/seq_lengths")
                        and seq_basis_lengths is not None
                        and num_nro_candidates_padded is not None
                    ):
                        fitted = (seq_basis_lengths + num_nro_candidates_padded).to(
                            cached_t.dtype
                        )
                    elif (
                        desc.name.endswith("/lengths_with_contextual")
                        and seq_basis_lengths is not None
                        and num_nro_candidates_padded is not None
                        and isinstance(
                            _local_by_name.get(
                                desc.name.replace(
                                    "/lengths_with_contextual", "/seq_lengths"
                                )
                            ),
                            torch.Tensor,
                        )
                    ):
                        # Rebuild LWC from padded sequence lengths to keep RO/NRO offsets aligned.
                        _seq_local = _local_by_name[
                            desc.name.replace(
                                "/lengths_with_contextual", "/seq_lengths"
                            )
                        ]
                        _ctx = (val - _seq_local).amax()
                        fitted = (
                            seq_basis_lengths + num_nro_candidates_padded + _ctx
                        ).to(cached_t.dtype)
                    elif desc.name.endswith("nro_padding_mask") and (
                        num_local_fit > 1
                        and real_num_candidates is not None
                        and real_num_candidates.shape[0] % num_local_fit == 0
                        and int(target_shape[0]) % num_local_fit == 0
                    ):
                        # Rebuild the mask per chip so padded NRO candidates are excluded from the loss.
                        real_mask = val.new_ones(
                            (int(real_num_candidates.sum().item()),)
                        )
                        fitted = pad_candidate_axis_to_static(
                            real_mask, real_num_candidates, int(target_shape[0])
                        ).to(cached_t.dtype)
                    elif desc.name.endswith(
                        (
                            "/past_weights",
                            "/past_watch_times",
                            "/past_timestamps_jagged",
                            "/exploded_actions",
                            "/exploded_action_intensity",
                            "/bucket_video_wt",
                            "/bucket_log_video_wt",
                            "/bucket_vvp",
                            "/bucket_log_vvp",
                            "/bucket_vpvd",
                            "/bucket_log_vpvd",
                            "/bucket_comment_vpvd",
                            "/bucket_log_comment_vpvd",
                        )
                    ):
                        # Pad jagged totals by seq_lengths to preserve per-chip boundaries.
                        real_lengths = _local_by_name.get(
                            desc.name.rsplit("/", 1)[0] + "/seq_lengths"
                        )
                        if not isinstance(real_lengths, torch.Tensor):
                            raise RuntimeError(
                                f"Missing seq_lengths for chip-major output {desc.name}"
                            )
                        if int(real_lengths.sum().item()) != int(val.shape[0]):
                            raise RuntimeError(
                                f"seq_lengths sum does not match {desc.name}: "
                                f"{int(real_lengths.sum().item())} != {val.shape[0]}"
                            )
                        target_total = int(target_shape[0])
                        if target_total % num_local_fit != 0:
                            raise RuntimeError(
                                f"Target size for {desc.name} is not divisible by "
                                f"num_local={num_local_fit}: {target_total}"
                            )
                        fitted = torch.ops.recir_padding.pad_jagged_values(
                            val,
                            real_lengths,
                            [target_total],
                            0,
                            -1,
                            1,
                            None,
                            False,
                        )
                    else:
                        fitted = _fit_to_shape(val, target_shape)
                    flat_tensors.append(fitted)
                else:
                    flat_tensors.append(
                        torch.zeros(
                            target_shape,
                            dtype=cached_t.dtype,
                            device=cached_t.device,
                        )
                    )
                tensor_i += 1

            # Exports last, matching the order DummyPreprocessorStack.forward
            # emits them. The dummy carries them so the traced bundle has the
            # output_segments _post_processing indexes; the runtime side has to
            # fill the same slots or the graph indexes past the end of the
            # dispatch result.
            flat_exports: List[torch.Tensor] = []
            for e_i, e_key in enumerate(dummy._export_keys):
                cached_e = getattr(dummy, f"_export_{e_i}")
                val = output.exports.get(e_key)
                if isinstance(val, torch.Tensor):
                    flat_exports.append(
                        _fit_to_shape(val, _per_host_target(cached_e.shape))
                    )
                else:
                    flat_exports.append(
                        torch.zeros(
                            _per_host_target(cached_e.shape),
                            dtype=cached_e.dtype,
                            device=cached_e.device,
                        )
                    )

            # FX trace order: all tensors first, then KJT (values, lengths)
            # pairs, then exports.
            flat: List[torch.Tensor] = flat_tensors + flat_kjts + flat_exports
            # Return raw (grad-preserving) tensors; the install helper wraps
            # this callable with save-for-backward + detach.
            return tuple(flat)

        def fwd_fn(*input_tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            pipeline = self._pipeline
            if pipeline is None:
                return _produce(*input_tensors)
            if self._replay is None:
                raise RuntimeError(
                    "preprocessor pipelining is on but no input replay was "
                    "built for the installed graph break"
                )

            self._fwd_calls_this_step += 1
            if self._fwd_calls_this_step != 1:
                raise RuntimeError(
                    f"the preprocessor graph break fired "
                    f"{self._fwd_calls_this_step} times on step {self._step}; "
                    "the step-keyed handoff requires exactly one"
                )

            result = pipeline.take(self._step)
            if result is None:
                with self._produce_lock:
                    result = _produce(*input_tensors)

            # Everything below is queued, not run: from here the device is busy
            # for the rest of the step, which is the window the worker has to
            # land in.
            self._submit_verify(input_tensors)
            self._submit_producer()
            self._submit_consumer()
            return result

        self._replay = None
        self._replay_build_args = None
        self._produce = _produce

        dispatch_id = install_graph_break_dispatch(
            gm,
            _xsurface_preprocessor_prefix(self._override_id),
            fwd_fn,
            num_state_tensors=len(flat_state),
            num_ranks=num_ranks,
            register_backward=not self._skip_backward_break,
        )
        if dispatch_id is None:
            if self._pipeline is not None:
                raise RuntimeError(
                    "preprocessor pipelining is on but no graph-break dispatch "
                    "was installed; there is nothing to run ahead of"
                )
            return gm, flat_state
        self._replay_build_args = (gm, dispatch_id, len(flat_state))
        if self._pipeline is not None:
            self._replay = build_dispatch_input_replay(*self._replay_build_args)
        return gm, flat_state

    def _submit_verify(self, from_graph: Tuple[torch.Tensor, ...]) -> None:
        """Queue the check that the worker's inputs are the graph's inputs."""
        replay = self._replay
        flat_step_args_fn = self._flat_step_args_fn
        batch = self._batch
        pipeline = self._pipeline
        if replay is None or flat_step_args_fn is None or batch is None:
            raise RuntimeError(
                "pipelining is on but the replay, the arg flattener or the "
                "current batch is missing; enable_pipelining/begin_step were "
                "not wired"
            )

        def check() -> None:
            replay.assert_matches(from_graph, flat_step_args_fn(batch))

        if pipeline is None or not pipeline.queue(check, label="verify"):
            check()

    def _submit_producer(self) -> None:
        batch = self._next_batch
        step = self._next_step
        pipeline = self._pipeline
        replay = self._replay
        flat_step_args_fn = self._flat_step_args_fn
        produce = self._produce
        self._next_batch = None
        if batch is None or pipeline is None or replay is None:
            return
        if flat_step_args_fn is None or produce is None:
            return

        def work() -> Tuple[torch.Tensor, ...]:
            inputs = replay(flat_step_args_fn(batch))
            with self._produce_lock, torch.enable_grad():
                return produce(*inputs)

        # A full queue is fine here and nowhere else: the next step's `take`
        # misses and the callback computes inline, which is exactly the
        # behaviour without pipelining.
        if not pipeline.queue(work, step=step, label="produce"):
            logger.debug("producer still busy; skipping precompute for this step")

    def _submit_consumer(self) -> None:
        consumer = self._consumer
        pipeline = self._pipeline
        if consumer is None:
            return
        self._consumer = None
        if pipeline is None or not pipeline.queue(consumer, label="consumer"):
            consumer()


def capture_sparse_arch_io(
    recblock: RecBlock,
    sample_input: FeatureBundle,
    capture_output: bool = True,
) -> Tuple[KeyedJaggedTensor, Dict[str, JaggedTensor] | None, Any]:
    """Run preprocessor + tokenizer eagerly once, capturing sparse-arch I/O.

    Returns ``(input_kjt, output_or_None, preprocessor_output)``. Downstream
    tokenizer errors are expected and swallowed: the hooks have already fired by
    then, and features disabled in the config produce e.g. 0-dim MLP reshapes.
    """
    # pyre-ignore[16]: dynamic attr from setattr in TokenizerStack
    sparse_arch = recblock.tokenizer_stack.mot_sparse_arch._sparse_arch
    input_captured: Dict[str, Any] = {}
    captured: Dict[str, Any] = {}

    def pre_hook(_module: nn.Module, input_args: Tuple[Any, ...]) -> None:
        input_captured["kjt"] = input_args[0]

    handles = [sparse_arch.register_forward_pre_hook(pre_hook)]
    if capture_output:

        def hook(
            _module: nn.Module,
            _input: Any,
            output: Dict[str, JaggedTensor],
        ) -> None:
            captured["output"] = output

        handles.append(sparse_arch.register_forward_hook(hook))

    with torch.no_grad(), _disable_sampling(recblock.preprocessor_stack):
        preproc_output = recblock.preprocessor_stack(sample_input)
        try:
            recblock.tokenizer_stack(preproc_output)
        except (RuntimeError, AttributeError):
            pass
    for handle in handles:
        handle.remove()

    # Gate on what the preprocessor was actually configured to emit. Prod sets
    # output_segments=["unsampled_inputs"] and _post_processing then hard-indexes
    # it, so a missing export there has to fail here rather than inside the
    # trace. A config that never requests the segment (the xsurface tests) has
    # nothing to check -- and could not produce it anyway, since the capture
    # above runs under _disable_sampling.
    # pyre-ignore[16]: dynamic attr from PreprocessorStack setattr
    needs_unsampled = getattr(
        recblock.preprocessor_stack.mot_preprocessor, "_needs_unsampled_inputs", []
    )
    if any(needs_unsampled):
        assert "unsampled_inputs" in preproc_output.exports, (
            "capture_sparse_arch_io produced exports "
            f"{sorted(preproc_output.exports.keys())} with no 'unsampled_inputs'"
        )
    assert "kjt" in input_captured, "Sparse arch pre-hook did not fire"
    if capture_output:
        assert "output" in captured, "Sparse arch forward hook did not fire"
    return input_captured["kjt"], captured.get("output"), preproc_output
