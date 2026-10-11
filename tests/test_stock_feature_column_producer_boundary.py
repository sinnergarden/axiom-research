"""Reject an understated target history before touching physical inputs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from axiom_research.stock_feature_column_producer import build_column_feature_matrix


class ColumnProducerBoundaryTests(unittest.TestCase):
    def test_target_missing_history_is_rejected_before_source_or_data_access(self):
        calendar = ['2023-01-02', '2023-01-03', '2023-01-04']
        catalog = Mock()
        catalog.select.return_value = [{'id': 'ALP101', 'lookback': 3}]
        data = Mock()
        with tempfile.TemporaryDirectory() as temp, \
             patch('axiom_research.stock_feature_inputs._spec', return_value='csi300'), \
             patch('axiom_research.feature_catalog.load_feature_catalog_ref', return_value=catalog), \
             patch('axiom_research.stock_feature_columns.build_feature_column_source') as build_source, \
             patch('axiom_research.stock_feature_columns.load_feature_column_source') as load_source:
            destination = Path(temp) / 'must-not-be-created'
            for declared in ([calendar[-1]], calendar[1:]):
                with self.subTest(read_sessions=declared):
                    spec = {'calendar': calendar, 'feature_sessions': [calendar[-1]],
                        'read_sessions': declared, 'catalog_ref': 'sha256:'+'a'*64,
                        'feature_selection': [{'id': 'ALP101', 'semantic_version': '1'}],
                        'ordered_features': ['ALP101']}
                    with self.assertRaisesRegex(ValueError, 'actual Feature history range mismatch'):
                        build_column_feature_matrix(data, spec=spec, destination=destination,
                            storage_options={'layout': 'matrix_v3'}, progress=None, column_source=None)
            build_source.assert_not_called()
            load_source.assert_not_called()
            data.read.assert_not_called()
            data.members.assert_not_called()
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
