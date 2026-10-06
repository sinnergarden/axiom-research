"""Small proposed OOS handoff shapes; no source Reader, model or statistics."""
from copy import deepcopy
from types import MappingProxyType
import unittest

from axiom_research.stock_artifacts import digest
from axiom_research.stock_signal_evaluation_matrix import _merge_evaluation_targets


def fixture(name, *, days=None):
    calendar = [f'2024-01-{day:02}' for day in range(1, 13)]
    days = calendar[:2] if days is None else days
    universe = ['S00', 'S01', 'S02']
    source_ref = digest(['original-containing-query', name])
    rows = [{'security_id': security, 'feature_session': day,
        'start_session': calendar[calendar.index(day)+1],
        'end_session': calendar[calendar.index(day)+5], 'return': i/100,
        'valid': True, 'invalid_reason': None,
        'label_available_at': calendar[calendar.index(day)+5]+'T20:00:00Z',
        'source_refs': [source_ref]}
        for day in days for i, security in enumerate(universe)]
    raw_ref = digest(['original-raw-build', name])
    raw = {raw_ref: {'raw_input': {'path': '/synthetic/raw-'+name+'.json',
        'file_digest': digest(['raw-file', name]), 'label_ref': raw_ref},
        'contract_version': 'stock_label_build_v1', 'label_ref': raw_ref,
        'label_spec': {'label_id': 'forward_5_session_open_close_v1', 'horizon_sessions': 5,
            'start_session_offset': 1, 'end_session_offset': 5,
            'start_price': 'open', 'end_price': 'close',
            'price_basis': 'common_anchor_adjusted_v1', 'adjustment_anchor': calendar[-1],
            'normalization': 'none'}, 'calendar_ref': digest({
                'contract_version': 'stock_label_calendar_v1', 'sessions': calendar}),
        'source_ref': source_ref, 'source_context': {'snapshot_id': 'saved-snapshot'},
        'sessions': days}}
    evaluation = {'contract_version': 'stock_matrix_evaluation_target_slice_v1',
        'prepared_view_ref': digest('prepared-view'), 'fold_spec_ref': digest(['fold', name]),
        'selector': {'path': '/synthetic/selector-'+name+'.json',
            'file_digest': digest(['selector-file', name]), 'selector_ref': digest(['selector', name])},
        'cutoff': calendar[-1]+'T23:00:00Z', 'rows': rows}
    evaluation['label_ref'] = digest(evaluation)
    leaves = {(r['security_id'], r['feature_session']): digest([
        'actual-selected-endpoint-factor-versions', r['security_id'], r['feature_session']]) for r in rows}
    scope = {'calendar': calendar, 'sessions': calendar[:2], 'universe': universe,
        'evaluation_cutoff': calendar[-1]+'T23:00:00Z'}
    return (evaluation, leaves, raw), scope


def reseal(item):
    item[0]['label_ref'] = digest({k: v for k, v in item[0].items() if k != 'label_ref'})
    return item


class MatrixEvaluationJoinTests(unittest.TestCase):
    def merge(self, items, scope):
        return _merge_evaluation_targets(items, calendar=scope['calendar'],
            universe=scope['universe'], scope=scope)

    def test_distinct_slice_and_query_refs_share_identical_leaves_and_keep_all_lineage(self):
        a, scope = fixture('a'); b, _ = fixture('b')
        originals = deepcopy([a, b])
        value = self.merge([a, b], scope)
        self.assertEqual(len(value['labels']), 6)
        self.assertEqual(len(value['label_inputs']), 2)
        self.assertEqual(len(value['raw_provenance']), 2)
        for key, row in value['labels'].items():
            self.assertEqual(row['source_refs'], sorted([next(iter(a[2].values()))['source_ref'],
                                                      next(iter(b[2].values()))['source_ref']]))
            self.assertEqual(value['label_leaf_bindings'][key], a[1][key])
        self.assertEqual([a, b], originals)
        self.assertEqual(value, self.merge([b, a], scope))

    def test_actual_leaf_revision_conflict_rejected_even_when_return_matches(self):
        a, scope = fixture('a'); b, _ = fixture('b')
        key = next(iter(b[1])); b[1][key] = digest('later-actual-leaf-version')
        with self.assertRaisesRegex(ValueError, 'leaf/value/clock conflict'):
            self.merge([a, b], scope)

    def test_evaluation_horizon_comes_from_target_and_mixed_definitions_rejected(self):
        a, scope = fixture('a'); b, _ = fixture('b')
        for header in b[2].values():
            header['label_spec'].update(label_id='forward_6_session_open_close_v1',
                horizon_sessions=6, end_session_offset=6)
        for row in b[0]['rows']:
            row['end_session'] = scope['calendar'][scope['calendar'].index(row['feature_session'])+6]
            row['label_available_at'] = row['end_session']+'T20:00:00Z'
        reseal(b)
        standalone = self.merge([b], scope)
        self.assertEqual(standalone['label_spec']['horizon_sessions'], 6)
        with self.assertRaisesRegex(ValueError, 'Label definition/version conflict'):
            self.merge([a, b], scope)

    def test_value_clock_and_reason_conflicts_rejected(self):
        for change in ({'return': .99}, {'label_available_at': '2024-01-07T21:00:00Z'},
                       {'return': None, 'valid': False, 'invalid_reason': 'MISSING_PRICE'}):
            with self.subTest(change=change):
                a, scope = fixture('a'); b, _ = fixture('b')
                b[0]['rows'][0].update(change); reseal(b)
                with self.assertRaisesRegex(ValueError, 'leaf/value/clock conflict'):
                    self.merge([a, b], scope)

    def test_null_reason_and_original_sources_preserved(self):
        a, scope = fixture('a')
        row = a[0]['rows'][0]; row.update({'return': None, 'valid': False,
            'invalid_reason': 'LABEL_NOT_MATURE', 'label_available_at': None})
        reseal(a); value = self.merge([a], scope)
        self.assertEqual(value['labels'][row['security_id'], row['feature_session']], row)

    def test_complete_scope_grid_leaves_and_source_provenance_required(self):
        for mutation, expected in (
                (lambda a: a[0]['rows'].pop(), 'complete'),
                (lambda a: a[1].pop(next(iter(a[1]))), 'leaf bindings'),
                (lambda a: a[0]['rows'][0].update(source_refs=[digest('wrong-raw-source')]), 'source outside')):
            with self.subTest(expected=expected):
                a, scope = fixture('a'); mutation(a); reseal(a)
                with self.assertRaisesRegex(ValueError, expected):
                    self.merge([a], scope)
        a, scope = fixture('a', days=['2024-01-01'])
        with self.assertRaisesRegex(ValueError, 'complete matrix evaluation Label scope'):
            self.merge([a], scope)

    def test_source_cutoff_and_endpoint_conflicts_rejected(self):
        a, scope = fixture('a')
        with self.assertRaisesRegex(ValueError, 'source cutoff'):
            self.merge([a], {**scope, 'evaluation_cutoff': '2024-01-12T22:00:00Z'})
        a[0]['rows'][0]['end_session'] = '2024-01-07'; reseal(a)
        with self.assertRaisesRegex(ValueError, 'calendar endpoint'):
            self.merge([a], scope)

    def test_training_normalized_or_malformed_label_definition_cannot_be_evaluation_target(self):
        for change in ({'normalization': 'cs_zscore'}, {'horizon_sessions': True}, None):
            with self.subTest(change=change):
                a, scope = fixture('a'); header = next(iter(a[2].values()))
                if change is None:
                    header['label_spec'] = None
                else:
                    header['label_spec'].update(change)
                with self.assertRaisesRegex(ValueError, 'RawLabel.*required'):
                    self.merge([a], scope)

    def test_immutable_owner_metadata_is_consumed_without_mutation(self):
        a, scope = fixture('a')
        evaluation = {**a[0], 'selector': MappingProxyType(a[0]['selector'])}
        raw = {ref: MappingProxyType({**header, 'raw_input': MappingProxyType(header['raw_input']),
            'sessions': tuple(header['sessions'])}) for ref, header in a[2].items()}
        value = self.merge([(MappingProxyType(evaluation), MappingProxyType(a[1]), MappingProxyType(raw))], scope)
        self.assertEqual(value, self.merge([a], scope))


if __name__ == '__main__':
    unittest.main()
