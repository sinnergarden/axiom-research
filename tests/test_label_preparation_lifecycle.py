"""Bounded full-GC spacing, cleanup and original normalized output parity."""
import gc
import unittest
from unittest.mock import patch
import weakref

from axiom_research.stock_artifacts import digest
from axiom_research.stock_label_normalization import normalize_forward_labels
from axiom_research.stock_label_preparation import _LabelShardGC
from test_label_normalization import CUTOFF, inputs


class LabelPreparationLifecycleTests(unittest.TestCase):
    def test_spacing_force_and_stage_release(self):
        with patch("axiom_research.stock_label_preparation.gc.collect") as collect:
            with _LabelShardGC() as lifecycle:
                for _ in range(31):
                    lifecycle.after_release()
                self.assertEqual(collect.call_count, 0)
                lifecycle.after_release()
                self.assertEqual(collect.call_count, 1)
                lifecycle.after_release(force=True)
                self.assertEqual(collect.call_count, 2)
            self.assertEqual(collect.call_count, 3)
            lifecycle.close()
            self.assertEqual(collect.call_count, 3)
        with self.assertRaises(RuntimeError):
            lifecycle.after_release()

    def test_invalid_interval_and_force_are_rejected(self):
        for value in (0, 33, True, 1.5):
            with self.assertRaises(ValueError):
                _LabelShardGC(interval=value)
        with _LabelShardGC() as lifecycle:
            with self.assertRaises(ValueError):
                lifecycle.after_release(force=1)

    def test_exception_closes_stage_and_collects_cycle(self):
        class Cycle:
            pass
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            with _LabelShardGC() as lifecycle:
                value = Cycle()
                value.self = value
                ref = weakref.ref(value)
                del value
                lifecycle.after_release()
                raise RuntimeError("worker failed")
        self.assertTrue(lifecycle.closed)
        self.assertIsNone(ref())

    def test_original_core_results_independent_of_collection_spacing(self):
        results = []
        for interval in (1, 32):
            refs = []
            with _LabelShardGC(interval=interval) as lifecycle:
                for _ in range(4):
                    raw, feature = inputs(("A", "B", "C"))
                    normalized = normalize_forward_labels(raw, features=feature, cutoff=CUTOFF)
                    refs.append(digest(normalized))
                    del raw, feature, normalized
                    lifecycle.after_release()
            results.append(refs)
        self.assertEqual(results[0], results[1])
        self.assertTrue(gc.isenabled())
