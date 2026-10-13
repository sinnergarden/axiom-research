"""Report-only UI reads retain identity and never reopen frozen ancestors."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (save_stock_signal_evaluation_inputs, evaluate_stock_signal_inputs,
    load_stock_signal_evaluation_summary)
from axiom_research.stock_artifacts import write_json, _read
from test_stock_signal_evaluation import fixture


class SignalEvaluationSummaryTests(unittest.TestCase):
    def test_report_identity_bindings_and_no_ancestor_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); signal,label,scope=fixture(root/'source')
            ref=save_stock_signal_evaluation_inputs({'candidate':signal},raw_label_input=label,
                scope=scope,destination=root/'inputs')
            saved=evaluate_stock_signal_inputs(ref,scope=scope,destination=root/'reports')['candidate']
            original=saved.to_dict()
            with patch('axiom_research.stock_signal_evaluation_projection._load_inputs',
                       side_effect=AssertionError('summary opened ancestors')):
                summary=load_stock_signal_evaluation_summary(saved.path,research_root=root)
            for key in ('contract_version','evidence_ref','content_digest','input_signal_refs','label_ref',
                        'label_spec','sample_mask_ref','statistics_ref','summary','status','limitations'):
                self.assertEqual(summary[key],original[key])
            self.assertFalse(summary['ancestors_audited'])
            self.assertEqual(summary['audit_scope'],'report_and_bindings')
            self.assertNotIn('by_session',summary['coverage']); self.assertNotIn('series',summary['coverage']['native'])
            with self.assertRaises(ValueError):
                load_stock_signal_evaluation_summary(saved.path,research_root=root/'inputs')
            damaged=_read(saved.path); damaged['status']='TAMPERED'; write_json(saved.path,damaged)
            with self.assertRaises(ValueError): load_stock_signal_evaluation_summary(saved.path,research_root=root)
