# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import subprocess
import sys
import weakref
from unittest import TestCase
from unittest.mock import patch

from recir.training.compilation_gc import compilation_gc_scope, freeze_compilation_heap

_GC_POLICY = sys.modules["recir.training.compilation_gc"]


class _Node:
    def __init__(self) -> None:
        self.cycle: _Node | None = None


class TestCompilationHeap(TestCase):
    def _run_isolated_gc(self) -> bool:
        if os.environ.get("RECIR_GC_TEST_CHILD") == "1":
            return False
        test = f"TestCompilationHeap.{self._testMethodName}"
        script = f"""
import gc, runpy, sys, types, unittest
gc.unfreeze()
policy = types.ModuleType({_GC_POLICY.__name__!r})
policy.__dict__.update(runpy.run_path({_GC_POLICY.__file__!r}))
sys.modules[{_GC_POLICY.__name__!r}] = policy
tests = types.ModuleType("isolated_gc_tests")
tests.__dict__.update(runpy.run_path({__file__!r}))
unittest.main(module=tests, defaultTest={test!r})
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(sys.path),
                "RECIR_GC_TEST_CHILD": "1",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return True

    def test_runtime_cycles_collected_and_heap_thaws_on_error(self) -> None:
        if self._run_isolated_gc():
            return
        self.assertTrue(gc.isenabled())
        self.assertEqual(gc.get_freeze_count(), 0)
        persistent = _Node()
        persistent.cycle = persistent
        reference = weakref.ref(persistent)
        with self.assertRaisesRegex(RuntimeError, "exit"):
            with compilation_gc_scope(True):
                freeze_compilation_heap()
                count = gc.get_freeze_count()
                freeze_compilation_heap()
                self.assertEqual(gc.get_freeze_count(), count)
                transient = _Node()
                transient.cycle = transient
                transient_reference = weakref.ref(transient)
                del transient
                gc.collect()
                self.assertIsNone(transient_reference())
                del persistent
                self.assertIsNotNone(reference())
                raise RuntimeError("exit")
        self.assertTrue(gc.isenabled())
        self.assertEqual(gc.get_freeze_count(), 0)
        gc.collect()
        self.assertIsNone(reference())

    def test_existing_freeze_policy_is_preserved(self) -> None:
        if self._run_isolated_gc():
            return
        bootstrap = _Node()
        bootstrap.cycle = bootstrap
        bootstrap_reference = weakref.ref(bootstrap)
        gc.freeze()
        try:
            count = gc.get_freeze_count()
            compiled = _Node()
            compiled.cycle = compiled
            compiled_reference = weakref.ref(compiled)
            with compilation_gc_scope(True):
                freeze_compilation_heap()
                self.assertGreater(gc.get_freeze_count(), count)
                transient = _Node()
                transient.cycle = transient
                transient_reference = weakref.ref(transient)
                del transient
                gc.collect()
                self.assertIsNone(transient_reference())
                del bootstrap, compiled
            gc.collect()
            self.assertIsNotNone(bootstrap_reference())
            self.assertIsNotNone(compiled_reference())
        finally:
            gc.unfreeze()
        gc.collect()
        self.assertIsNone(bootstrap_reference())
        self.assertIsNone(compiled_reference())

    def test_disabled_scope_has_no_global_effect(self) -> None:
        with patch("gc.freeze") as freeze:
            with compilation_gc_scope(False):
                freeze_compilation_heap()
            freeze.assert_not_called()
