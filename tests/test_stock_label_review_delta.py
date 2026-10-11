"""Two small counterexamples for the independent review blockers; no Core."""
import ast
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import sys
import unittest

from axiom_research.stock_canonical_json import validate_canonical_chunks
from axiom_research.stock_label_wire import _LabelWireValidator


class LabelReviewDeltaTests(unittest.TestCase):
    def test_deep_coverage_reserves_all_path_tuples_before_growth(self):
        # The reviewer's complete one-security/session wire, only 4KB JSON.
        # Its 1800 coexisting full paths exceed 13MB of pointer containers.
        depth = 1800
        coverage = b'['*depth + b'0' + b']'*depth
        context = (b'{"coverage":' + coverage
            + b',"query":{"fields":["close","open"],"sessions":["2026-01-02"],"symbols":["A"]}}')
        field_meta = (b'{"close":{"by_key":[{"security_id":"A","session":"2026-01-02"}]},'
                      b'"open":{"by_key":[{"security_id":"A","session":"2026-01-02"}]}}')
        records = b'[{"close":2.0,"open":1.0,"security_id":"A","session":"2026-01-02"}]'
        payload = b'{"context":'+context+b',"field_meta":'+field_meta+b',"records":'+records+b'}'
        maximum = 2*1024**2
        validate_canonical_chunks((payload,), maximum_workspace_bytes=maximum)
        observer = _LabelWireValidator(maximum, None, None)
        with self.assertRaisesRegex(ValueError, 'workspace budget exceeded'):
            observer.consume(payload)
        self.assertLess(len(observer.paths), depth)
        self.assertLessEqual(observer.workspace(), maximum)
        self.assertGreaterEqual(observer.path_bytes,
            sum(sys.getsizeof(frame[0]) for frame in observer.paths))
        adequate = _LabelWireValidator(32*1024**2, None, None)
        prefix = b'{"context":{"coverage":'+b'['*depth
        adequate.consume(prefix)
        containers = sys.getsizeof(adequate.paths) + sum(
            sys.getsizeof(frame)+sys.getsizeof(frame[0]) for frame in adequate.paths)
        self.assertLessEqual(containers,adequate.workspace())
        adequate.consume(payload[len(prefix):])
        proof = adequate.finish()
        ref = lambda value: 'sha256:'+sha256(value).hexdigest()
        self.assertEqual(proof['digest'],ref(payload))
        self.assertEqual(proof['component_refs'],{'context':ref(context),
            'field_meta':ref(field_meta),'records':ref(records)})
        self.assertEqual(proof['coverage_identity'],(True,ref(coverage)))
        self.assertEqual(adequate.path_bytes,0)

    def test_staged_and_saved_hit_are_separate_from_initial_admission(self):
        # Extract only the pure observer: importing the real runner would read
        # its external plan and monitor. No prepare/Data/Core is performed.
        path = Path(__file__).resolve().parents[1]/'tools'/'run_saved_fourfold.py'
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == '_FeatureAdmissionPhases')
        namespace = {'contextmanager':contextmanager}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),namespace)
        for closure_phase in ('staged','prepared_saved_batch'):
            observations = {'feature_admission_calls':0}
            phases = namespace['_FeatureAdmissionPhases'](observations)
            phases.completed()
            with phases.scope(closure_phase):
                phases.completed()
                with self.assertRaisesRegex(RuntimeError,'closure failed'):
                    with phases.scope('nested_failure'):
                        raise RuntimeError('closure failed')
                self.assertEqual(phases.phase,closure_phase)
            self.assertEqual(phases.phase,'initial')
            self.assertEqual(phases.initial_calls,1)
            self.assertEqual(observations['feature_admission_calls'],2)
            self.assertEqual(observations['feature_admission_calls_by_phase'],
                             {'initial':1,closure_phase:1})
            self.assertTrue(phases.complete_counts_match(closure_phase=='prepared_saved_batch'))
            self.assertFalse(phases.complete_counts_match(closure_phase=='staged'))
            if closure_phase == 'staged':
                with phases.scope('prepared_saved_batch'):
                    phases.completed()  # Concurrent winner requires public admission.
                self.assertTrue(phases.complete_counts_match(False))
            phases.completed()  # A second initial admission still rejects.
            self.assertFalse(phases.complete_counts_match(closure_phase=='prepared_saved_batch'))


if __name__ == '__main__':
    unittest.main()
