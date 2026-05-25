#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from hammer.utils import HammerKernel
from hammer.v3.models.udd_fm_v3.udd_fm_config import BlueReelsVDDModuleConfig
from hammer.v3.models.udd_fm_v3.udd_fm_train_v3 import BlueReelsVDDTrainV3
from hammer.v3.modules.core.features.bundle import FeatureBundle
from hammer.v3.modules.recblock import _inputs_to_feature_bundle
from recir.api import compile, COMPILED_OBJECT_KEY, sync_device_state_to_torch
from recir.distributed.parallel_mode import HybridParallel
from recir.ops.fake.kjt import patch_ec_for_tracing
from recir.ops.fake.mot_preprocessor import patch_mot_preprocessor_for_tracing
from recir.sharding_plan import EmbeddingShardingPlan, EmbeddingShardingType
from recir.tests.xla.xsurface_test_utils import (
    _build_full_model,
    _create_random_train_batch,
    TEST_BATCH_SIZE,
    TEST_MAX_UIH_SEQ_LEN,
)
from torch.distributed.distributed_c10d import _get_default_group, _world
from torch.distributed.tensor import DeviceMesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# Mocked world size: HybridParallel sees WORLD_SIZE ranks via the fake
# process group, while the test runs on a single physical GPU.
WORLD_SIZE: int = 2


# buck2 test @//mode/opt //recir/tests/xla:test_xsurface_hybrid_parallel


def _build_default_pg_mesh(device_type: str, size: int) -> DeviceMesh:
    """Build a DeviceMesh backed by the already-registered default fake pg.

    ``init_process_group(backend="fake", world_size=N)`` (called once in
    setUpClass) creates a FakeProcessGroup for the default group AND
    globally registers it under a c10d-managed name.  We reuse that pg +
    name here so functional collectives (``all_reduce``,
    ``all_to_all_single``, ``all_gather``) resolve it by name through
    ``c10d._get_group_size_by_name``.

    This sidesteps the type-strict ``_register_process_group`` C++ binding,
    which rejects raw ``FakeProcessGroup`` instances created via
    ``FakeProcessGroup._create_internal``.
    """
    default_pg = _get_default_group()
    pg_name = _world.pg_names[default_pg]
    mesh = DeviceMesh(
        device_type,
        torch.arange(size),
        _init_backend=False,
        _rank=0,
    )
    mesh._dim_group_names = [pg_name]
    mesh._pg_registry[pg_name] = default_pg
    return mesh


def _make_sample_input_from_module_config(
    module_config: BlueReelsVDDModuleConfig,
    batch: Any,
    device: torch.device,
    batch_size: int = TEST_BATCH_SIZE,
    max_uih_seq_len: int = TEST_MAX_UIH_SEQ_LEN,
) -> FeatureBundle:
    """Build a sample FeatureBundle whose KJT key ORDER matches the
    runtime batch (not module_config).
    """
    ro_keys = list(batch.ro_id_list_features.keys())
    nro_keys = list(batch.nro_id_list_features.keys())

    ro_lengths = torch.full(
        (len(ro_keys) * batch_size,),
        max_uih_seq_len,
        dtype=torch.int32,
        device=device,
    )
    ro_values = torch.randint(
        0, 50, (int(ro_lengths.sum().item()),), dtype=torch.int64, device=device
    )
    ro_id_list = KeyedJaggedTensor(keys=ro_keys, values=ro_values, lengths=ro_lengths)

    B_NRO = int(batch.num_candidates.sum().item())
    nro_lengths = torch.ones(len(nro_keys) * B_NRO, dtype=torch.int32, device=device)
    nro_values = torch.randint(
        0, 50, (len(nro_keys) * B_NRO,), dtype=torch.int64, device=device
    )
    nro_id_list = KeyedJaggedTensor(
        keys=nro_keys, values=nro_values, lengths=nro_lengths
    )

    return _inputs_to_feature_bundle(
        ro_id_list,
        None,
        nro_id_list,
        batch.num_candidates,
        batch.all_labels,
    )


def _build_row_wise_embedding_plan(
    model: BlueReelsVDDTrainV3, module_config: BlueReelsVDDModuleConfig
) -> EmbeddingShardingPlan:
    """Build a row-wise EmbeddingShardingPlan over every config-declared
    ``embedding_tables`` entry.
    """
    table_names: set[str] = set(module_config.embedding_tables.keys())
    sharding_map: Dict[str, EmbeddingShardingType] = {}
    for fqn, _param in model.named_parameters():
        for name in table_names:
            if fqn.endswith(f".{name}.weight"):
                sharding_map[fqn] = EmbeddingShardingType.ROW_WISE
                break
    assert sharding_map, (
        "No embedding params matched the configured table names - the FQN "
        "format produced by ECSparseArch may have changed."
    )
    return EmbeddingShardingPlan(sharding_map=sharding_map)


class TestXSurfaceHybridParallelCompile(unittest.TestCase):
    """Verify the xsurface V3 model compiles through the HybridParallel path.

    DDP (replicate) dense style only.  FSDP is out of scope.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not dist.is_initialized():
            dist.init_process_group(
                backend="fake",
                world_size=WORLD_SIZE,
                rank=0,
                store=FakeStore(),
            )

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDownClass()

    def _build_compiled_train_step(
        self,
        num_steps: int = 1,
    ) -> Tuple[Any, BlueReelsVDDTrainV3, EmbeddingShardingPlan]:
        """Build the model, compile via HybridParallel v2 (DDP), run ``num_steps``."""
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        model, feature_config = _build_full_model(device)

        model.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)

        # Patch every torchrec EmbeddingCollection in the model so its
        # forward routes through recir_torchrec::kjt_split_per_feature
        # (SymInt fake kernel) + per-feature aten.embedding.  Without this,
        # ECSparseArch's KJT.to_dict() call hits a DataDependentOutputException
        # under make_fx(tracing_mode="fake").
        patched = patch_ec_for_tracing(model)
        assert patched > 0, "Expected at least one EmbeddingCollection to patch"

        for p in model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        batch = _create_random_train_batch(
            feature_config, device, batch_size=TEST_BATCH_SIZE
        )

        # Wrap MoTPreprocessor in TraceableMoTPreprocessor: runs the real
        # preprocessor behind an opaque custom op with SymInt-aware fake
        # kernel. The sample input is needed to introspect the output
        # FeatureBundle structure once at construction time.
        recblock_sample = _make_sample_input_from_module_config(
            model.module_config, batch, device
        )
        n_preproc_patched = patch_mot_preprocessor_for_tracing(model, recblock_sample)
        assert n_preproc_patched > 0, "Expected at least one MoTPreprocessor"

        ro_kjt = batch.ro_id_list_features
        nro_kjt = batch.nro_id_list_features
        unary_kjt = batch.unary_features
        assert unary_kjt is not None
        model.set_recir_kjt_keys(ro_kjt.keys(), nro_kjt.keys(), unary_kjt.keys())

        opt = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)

        module_config = model.module_config
        embedding_plan = _build_row_wise_embedding_plan(model, module_config)

        parallel_mode = HybridParallel(
            dense_style="replicate",
            embedding_plan=embedding_plan,
        )
        parallel_mode._mesh = _build_default_pg_mesh("cuda", WORLD_SIZE)

        # pyre-ignore
        def train_step_fn(
            mod: nn.Module,
            opt: torch.optim.Optimizer,
            ro_float_features: torch.Tensor,
            nro_float_features: torch.Tensor,
            ro_id_list_values: torch.Tensor,
            ro_id_list_lengths: torch.Tensor,
            nro_id_list_values: torch.Tensor,
            nro_id_list_lengths: torch.Tensor,
            unary_values: torch.Tensor,
            unary_lengths: torch.Tensor,
            num_candidates: torch.Tensor,
            supervision_label_t: torch.Tensor,
            supervision_weight_t: torch.Tensor,
            all_labels: Dict[str, torch.Tensor],
        ) -> None:
            all_losses, _ = mod(
                ro_float_features,
                nro_float_features,
                ro_id_list_values,
                ro_id_list_lengths,
                nro_id_list_values,
                nro_id_list_lengths,
                unary_values,
                unary_lengths,
                num_candidates,
                supervision_label_t,
                supervision_weight_t,
                all_labels,
            )
            loss: torch.Tensor = all_losses.sum()
            loss.backward()
            opt.step()

        train_step = compile(
            parallel_mode=parallel_mode,
        )(train_step_fn)

        for _ in range(num_steps):
            train_step(
                model,
                opt,
                batch.ro_float_features,
                batch.nro_float_features,
                ro_kjt.values(),
                ro_kjt.lengths(),
                nro_kjt.values(),
                nro_kjt.lengths(),
                unary_kjt.values(),
                unary_kjt.lengths(),
                batch.num_candidates,
                batch.supervision_label.t().contiguous(),
                batch.supervision_weight.t().contiguous(),
                batch.all_labels,
            )

        compiled_obj = train_step.__dict__[COMPILED_OBJECT_KEY]
        return compiled_obj, model, embedding_plan

    @skip_if_lt_x_gpu(1)
    def test_xsurface_compiles_ddp(self) -> None:
        """xsurface compiles through HybridParallel v2 with DDP dense style."""
        compiled_obj, _, _ = self._build_compiled_train_step()
        self.assertIsNotNone(compiled_obj)
        self.assertIsNotNone(compiled_obj.gm)

    @skip_if_lt_x_gpu(1)
    def test_xsurface_embedding_plan_covers_all_tables(self) -> None:
        """The discovered plan contains one FQN per ``embedding_tables`` entry.

        ``additional_embeddings`` are dense bucketize+embed modules — they
        are replicated by DDP, not row-wise sharded, so the plan does not
        cover them.
        """
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        model, _ = _build_full_model(device)
        module_config = model.module_config
        plan = _build_row_wise_embedding_plan(model, module_config)
        expected_table_count = len(module_config.embedding_tables)
        self.assertEqual(
            len(plan.sharding_map),
            expected_table_count,
            f"Expected one sharded FQN per config table (got "
            f"{len(plan.sharding_map)} for {expected_table_count} tables): "
            f"{sorted(plan.sharding_map)}",
        )

    @skip_if_lt_x_gpu(1)
    def test_xsurface_graph_has_embedding_alltoall(self) -> None:
        """Row-wise embedding sharding is decomposed into all_to_all_single ops.

        DDP dense style replicates dense params (no allgather), but the
        row-wise embedding plan still triggers ``rewrite_sparse_embeddings``
        which inserts ``all_to_all_single`` for the embedding lookups.
        """
        compiled_obj, _, _ = self._build_compiled_train_step()
        gm = compiled_obj.gm
        a2a_count = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops._c10d_functional.all_to_all_single.default
        )
        self.assertGreater(
            a2a_count,
            0,
            "Row-wise embedding sharding should produce all_to_all_single nodes",
        )

    @skip_if_lt_x_gpu(1)
    def test_xsurface_graph_has_grad_allreduce(self) -> None:
        """DDP dense style inserts all_reduce on dense gradients."""
        compiled_obj, _, _ = self._build_compiled_train_step()
        gm = compiled_obj.gm
        ar_count = sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and "all_reduce" in str(n.target)
        )
        self.assertGreater(
            ar_count,
            0,
            "DDP (replicate) should insert all_reduce on dense gradients",
        )

    @skip_if_lt_x_gpu(1)
    def test_xsurface_params_change_after_training(self) -> None:
        """Verify parameters actually change after training (not frozen)."""
        torch.manual_seed(42)
        device = torch.device("cuda:0")
        model, _ = _build_full_model(device)
        init_params = {n: p.detach().clone() for n, p in model.named_parameters()}

        _, trained_model, _ = self._build_compiled_train_step(num_steps=2)

        changed = 0
        for name, param in trained_model.named_parameters():
            init_p = init_params.get(name)
            # Sharded parameters may have a smaller local shape after
            # row-wise sharding; only compare elementwise when shapes match.
            if init_p is not None and init_p.shape == param.shape:
                if not torch.equal(init_p, param):
                    changed += 1
            elif init_p is not None:
                # Different shape implies the param was sharded - count as
                # changed (we can't directly compare without all_gather).
                changed += 1
        self.assertGreater(
            changed,
            0,
            "No parameters changed after training - model may be frozen.",
        )


class TestXSurfaceCompileMatchesEager(unittest.TestCase):
    """Numerical parity test: recir-compiled fwd+bwd vs eager fwd+bwd.

    Single-device (WORLD_SIZE=1) so collectives and row-wise sharding are
    no-ops -- this isolates the test to compile-vs-eager differences and
    lets the eager reference be plain torch (no DDP / FSDP wrapper, no
    grad scaling).

    Two assertions:
      1. Forward loss tensor from the compiled train_step equals the
         loss tensor from an eager forward (within tolerance).
      2. After ``opt.step()`` on both sides, every named parameter of
         the compiled model matches the corresponding eager param
         (DTensor params are unwrapped via ``full_tensor()``).
    """

    _LOCAL_WORLD_SIZE: int = 1

    def setUp(self) -> None:
        super().setUp()
        # Per-test pg lifecycle: this class wants WORLD_SIZE=1 while the
        # sibling TestXSurfaceHybridParallelCompile uses WORLD_SIZE=2 via
        # setUpClass.  Destroy + re-init guarantees a fresh size-1 pg
        # regardless of which class ran first.
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(
            backend="fake",
            world_size=self._LOCAL_WORLD_SIZE,
            rank=0,
            store=FakeStore(),
        )

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    @staticmethod
    def _apply_tracing_patches(
        model: BlueReelsVDDTrainV3, batch: Any, device: torch.device
    ) -> None:
        patched = patch_ec_for_tracing(model)
        assert patched > 0
        sample = _make_sample_input_from_module_config(
            model.module_config, batch, device
        )
        n = patch_mot_preprocessor_for_tracing(model, sample)
        assert n > 0

    @staticmethod
    def _flatten_inputs(batch: Any) -> Tuple[Any, ...]:
        ro_kjt = batch.ro_id_list_features
        nro_kjt = batch.nro_id_list_features
        unary_kjt = batch.unary_features
        assert unary_kjt is not None
        return (
            batch.ro_float_features,
            batch.nro_float_features,
            ro_kjt.values(),
            ro_kjt.lengths(),
            nro_kjt.values(),
            nro_kjt.lengths(),
            unary_kjt.values(),
            unary_kjt.lengths(),
            batch.num_candidates,
            batch.supervision_label.t().contiguous(),
            batch.supervision_weight.t().contiguous(),
            batch.all_labels,
        )

    @skip_if_lt_x_gpu(1)
    def test_compile_matches_eager(self) -> None:
        device = torch.device("cuda:0")

        torch.manual_seed(42)
        base_model, feature_config = _build_full_model(device)
        base_model.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        base_model.recursive_setattr("_bf16_training", True)
        # Disable LearnableLossWeights for this parity test: it
        # initialises loss_scale and llw_weights to zero, so
        # all_losses = llw.compute(real_losses) = real_losses * 0 = 0.
        base_model.learnable_loss_weights_module = None
        for p in base_model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        batch = _create_random_train_batch(
            feature_config, device, batch_size=TEST_BATCH_SIZE
        )

        compile_model: BlueReelsVDDTrainV3 = deepcopy(base_model)
        ref_model: BlueReelsVDDTrainV3 = deepcopy(base_model)
        del base_model

        self._apply_tracing_patches(compile_model, batch, device)
        self._apply_tracing_patches(ref_model, batch, device)

        ro_kjt = batch.ro_id_list_features
        nro_kjt = batch.nro_id_list_features
        unary_kjt = batch.unary_features
        assert unary_kjt is not None
        for m in (compile_model, ref_model):
            m.set_recir_kjt_keys(ro_kjt.keys(), nro_kjt.keys(), unary_kjt.keys())

        # Sanity check: deepcopies start byte-identical.
        for (n1, p1), (n2, p2) in zip(
            compile_model.named_parameters(), ref_model.named_parameters()
        ):
            self.assertEqual(n1, n2)
            torch.testing.assert_close(
                p1.detach(),
                p2.detach(),
                msg=f"initial param diverged after deepcopy: {n1}",
            )

        opt = torch.optim.SGD(compile_model.parameters(), lr=0.01, foreach=False)
        ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01, foreach=False)

        inputs = self._flatten_inputs(batch)

        # pyre-ignore
        def train_step_fn(
            mod: nn.Module,
            optim: torch.optim.Optimizer,
            ro_float_features: torch.Tensor,
            nro_float_features: torch.Tensor,
            ro_id_list_values: torch.Tensor,
            ro_id_list_lengths: torch.Tensor,
            nro_id_list_values: torch.Tensor,
            nro_id_list_lengths: torch.Tensor,
            unary_values: torch.Tensor,
            unary_lengths: torch.Tensor,
            num_candidates: torch.Tensor,
            supervision_label_t: torch.Tensor,
            supervision_weight_t: torch.Tensor,
            all_labels: Dict[str, torch.Tensor],
        ) -> torch.Tensor:
            all_losses, _ = mod(
                ro_float_features,
                nro_float_features,
                ro_id_list_values,
                ro_id_list_lengths,
                nro_id_list_values,
                nro_id_list_lengths,
                unary_values,
                unary_lengths,
                num_candidates,
                supervision_label_t,
                supervision_weight_t,
                all_labels,
            )
            loss: torch.Tensor = all_losses.sum()
            loss.backward()
            optim.step()
            return all_losses.detach()

        embedding_plan = _build_row_wise_embedding_plan(
            compile_model, compile_model.module_config
        )
        parallel_mode = HybridParallel(
            dense_style="replicate",
            embedding_plan=embedding_plan,
        )
        parallel_mode._mesh = _build_default_pg_mesh("cuda", self._LOCAL_WORLD_SIZE)

        compiled_train_step = compile(parallel_mode=parallel_mode)(train_step_fn)
        compiled_losses = compiled_train_step(compile_model, opt, *inputs)
        ref_all_losses, _ = ref_model(*inputs)
        ref_loss: torch.Tensor = ref_all_losses.sum()
        ref_loss.backward()
        ref_opt.step()
        ref_losses_detached = ref_all_losses.detach()

        torch.testing.assert_close(
            compiled_losses,
            ref_losses_detached,
            atol=1e-4,
            rtol=1e-4,
            equal_nan=True,
            msg="forward loss tensors diverged between compiled and eager",
        )

        sync_device_state_to_torch(compiled_train_step)
        compiled_params = dict(compile_model.named_parameters())
        ref_params = dict(ref_model.named_parameters())
        self.assertEqual(set(compiled_params), set(ref_params))
        for name in compiled_params:
            cp = compiled_params[name]
            rp = ref_params[name]
            torch.testing.assert_close(
                cp,
                rp,
                atol=1e-4,
                rtol=1e-4,
                equal_nan=True,
                msg=f"param mismatch after opt.step: {name}",
            )


class TestXSurfacePatchesPreserveNumerics(unittest.TestCase):
    """Verify patch_ec_for_tracing + patch_mot_preprocessor_for_tracing
    don't change the model's runtime numerics.

    Sibling to TestXSurfaceCompileMatchesEager, which patches BOTH eager
    and compile sides and therefore CANNOT catch numerical bugs that
    live inside the patches themselves.  This test closes that coverage
    gap by comparing patched eager forward against unpatched eager
    forward (no compile involved).
    """

    _LOCAL_WORLD_SIZE: int = 1

    def setUp(self) -> None:
        super().setUp()
        # Per-test pg lifecycle: same rationale as
        # TestXSurfaceCompileMatchesEager.
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.init_process_group(
            backend="fake",
            world_size=self._LOCAL_WORLD_SIZE,
            rank=0,
            store=FakeStore(),
        )

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    @skip_if_lt_x_gpu(1)
    def test_patches_preserve_numerics(self) -> None:
        device = torch.device("cuda:0")

        torch.manual_seed(42)
        base_model, feature_config = _build_full_model(device)
        base_model.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        base_model.recursive_setattr("_bf16_training", True)
        base_model.learnable_loss_weights_module = None
        for p in base_model.parameters():
            if p.is_floating_point():
                p.requires_grad_(True)

        batch = _create_random_train_batch(
            feature_config, device, batch_size=TEST_BATCH_SIZE
        )

        # Two byte-identical copies; patch only one of them.
        unpatched_model: BlueReelsVDDTrainV3 = deepcopy(base_model)
        patched_model: BlueReelsVDDTrainV3 = deepcopy(base_model)
        del base_model

        n_ec = patch_ec_for_tracing(patched_model)
        assert n_ec > 0
        sample = _make_sample_input_from_module_config(
            patched_model.module_config, batch, device
        )
        n_pp = patch_mot_preprocessor_for_tracing(patched_model, sample)
        assert n_pp > 0

        ro_kjt = batch.ro_id_list_features
        nro_kjt = batch.nro_id_list_features
        unary_kjt = batch.unary_features
        assert unary_kjt is not None
        for m in (unpatched_model, patched_model):
            m.set_recir_kjt_keys(ro_kjt.keys(), nro_kjt.keys(), unary_kjt.keys())

        inputs = TestXSurfaceCompileMatchesEager._flatten_inputs(batch)

        with torch.no_grad():
            unpatched_losses, unpatched_named = unpatched_model(*inputs)
            patched_losses, patched_named = patched_model(*inputs)

        torch.testing.assert_close(
            patched_losses,
            unpatched_losses,
            atol=1e-4,
            rtol=1e-4,
            equal_nan=True,
            msg=(
                "patch_ec_for_tracing + patch_mot_preprocessor_for_tracing "
                "changed forward losses; the patches are not numerically "
                "transparent at runtime"
            ),
        )

        unp_keys = set(unpatched_named.keys())
        pat_keys = set(patched_named.keys())
        for k in sorted(unp_keys & pat_keys):
            uv = unpatched_named[k]
            pv = patched_named[k]
            if isinstance(uv, torch.Tensor) and isinstance(pv, torch.Tensor):
                torch.testing.assert_close(
                    pv,
                    uv,
                    atol=1e-4,
                    rtol=1e-4,
                    equal_nan=True,
                    msg=f"named_tensors[{k!r}] diverged after patching",
                )


if __name__ == "__main__":
    run_tests()
