"""The thin analysis adapter shares one admission with both Core evaluations."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_signal_evaluation_projection import evaluate_stock_signal_input_analysis
from axiom_research.stock_artifacts import file_digest


class SignalGroupAnalysisTests(unittest.TestCase):
    def test_single_admission_shared_pair_identity_and_repeat(self):
        sample = dict(contract_version='signal_statistics_input_v1',
            sessions=['2023-01-03'], signal_keys=['model'], pairs=[
                dict(signal_key='model', security_id=f'S{i:02}', session='2023-01-03',
                     score=float(i), outcome=i/100) for i in range(20)])
        prepared = ('frozen-ref', {}, {'common_input':sample, 'native_input':sample})
        spec = dict(group_count=10, minimum_pairs=20, rank_ties='average',
            group_assignment='average_rank_fraction', time_weighting='equal_valid_sessions',
            return_semantics='raw_label_return', annualization='none')
        module = 'axiom_research.stock_signal_evaluation_projection'
        with tempfile.TemporaryDirectory() as temp, patch(module+'._load_inputs',
                return_value=prepared) as admit, patch(module+'._evaluate_prepared',
                return_value={'model':'saved-IC'}) as ic:
            result = evaluate_stock_signal_input_analysis('frozen-ref', scope={},
                destination=temp, group_spec=spec)
            admit.assert_called_once(); self.assertIs(ic.call_args.args[0][2], prepared[2])
            self.assertEqual(result['groups']['common'], result['groups']['native'])
            path = Path(result['groups']['common']['path']); original = (file_digest(path),path.stat().st_mtime_ns)
            repeated = evaluate_stock_signal_input_analysis('frozen-ref', scope={},
                destination=temp, group_spec=spec)
            self.assertEqual(result, repeated)
            self.assertEqual((file_digest(path),path.stat().st_mtime_ns), original)


if __name__ == '__main__': unittest.main()
