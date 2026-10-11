"""Direct compact owner stub: 3 folds/2 signals/24 securities across a year.

No historical fixture, Data, normalization, Feature preparation or fit runs.
The stub stands in only for the pending owner lease; frozen consumption is real.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta, datetime
from pathlib import Path
from types import SimpleNamespace
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (save_stock_signal_evaluation_inputs, evaluate_stock_signal_inputs,
    evaluate_stock_signal_input_periods, load_stock_signal_evaluation, audit_stock_signal_evaluation)
from axiom_research.stock_artifacts import digest, file_digest, write_json, _read, _verify_ref
from axiom_research.stock_fold_inputs import seal, file_fingerprint
from axiom_research.stock_batch import StockMLBatchInputs, _TOKEN
from axiom_research.stock_folds import prediction_rows
from axiom_research.stock_compact_controls import grid_ranges
from axiom_research.stock_signal_evaluation_inputs import _select_inputs, _validate_raw_label_header
from axiom_research.stock_signal_evaluation import SPEC
from axiom_research import stock_signal_evaluation_projection as frozen
from axiom_research import stock_signal_evaluation_compact as compact
from axiom_research.stock_signal_evaluation_matrix import _sources_table, _label_projection_ref, _label_day_ref
from axiom_engine.core import evaluate_signal_statistics
from stock_signal_native_wire_fixture import saved_context


class OwnerStub:
    def __init__(self, root, *, missing=True):
        self.root = root; self.active = 0; self.calls = 0; self.leases = {}; self.changed = None
        self.calendar = [(date(2023, 12, 18)+timedelta(days=i)).isoformat() for i in range(45)
            if (date(2023, 12, 18)+timedelta(days=i)).weekday() < 5]
        self.universe = [f'S{i:02d}' for i in range(24)]
        self.days = ['2023-12-27', '2023-12-28', '2023-12-29', '2024-01-02', '2024-01-03', '2024-01-04']
        self.cutoff = '2024-01-31T20:00:00Z'
        self.scope = {'calendar': self.calendar, 'sessions': self.days, 'universe': self.universe,
            'evaluation_cutoff': self.cutoff}
        self.common = {'calendar': self.calendar, 'universe': self.universe, 'snapshot': 'synthetic_compact_v1',
            'pit_policy': 'synthetic_pit_v1'}
        self.raw = {}; self.scores = {}; self.inputs = {'a': [], 'b': []}; folds = []
        root.mkdir(); write_json(root/'prepared.json', seal({'definition': self.common}, 'prepared_view_ref'))
        prepared = {'path': str(root/'prepared.json'), 'file_digest': file_digest(root/'prepared.json'),
            'prepared_view_ref': _read(root/'prepared.json')['prepared_view_ref']}
        endpoints = sorted({self.calendar[-1]} | {self.calendar[self.calendar.index(day)+offset]
            for day in self.days for offset in (1, 5)})
        context = saved_context(snapshot=self.common['snapshot'], pit=self.common['pit_policy'],
            universe=self.universe, sessions=endpoints, anchor=self.calendar[-1], cutoff=self.cutoff)
        from axiom_research.labels import _query_context
        _query_context(context, self.calendar)  # Actual owner validator; no read or preparation.
        price = seal({'contract_version': 'stock_label_price_view_v1', 'context': context,
            'records_ref': digest('synthetic selected prices'), 'field_meta_ref': digest('synthetic field metadata')}, 'price_view_ref')
        for n in range(3):
            days = self.days[n*2:n*2+2]; target_root = root/f'target-{n}'; target_root.mkdir()
            rows = []
            for day in days:
                pos = self.calendar.index(day); j = self.days.index(day)
                for i, security in enumerate(self.universe):
                    valid = i != 4
                    row = {'security_id': security, 'feature_session': day, 'start_session': self.calendar[pos+1],
                        'end_session': self.calendar[pos+5], 'return': (0.25 if j == 1 else (i*i+j*i)/1000) if valid else None,
                        'valid': valid, 'invalid_reason': None if valid else 'SOURCE_MISSING',
                        'label_available_at': self.calendar[pos+5]+'T19:00:00Z' if valid else None,
                        'source_refs': [price['price_view_ref']]}
                    rows.append(row); self.raw[security, day] = deepcopy(row)
            definition = {**{k: self.common[k] for k in ('snapshot', 'universe', 'calendar')}, 'price_view': price,
                'cutoff': self.cutoff, 'sessions': days, 'formula': 'close(f+5) / open(f+1) - 1',
                'horizon_sessions': 5, 'start_session_offset': 1, 'end_session_offset': 5,
                'price_basis': 'common_anchor_adjusted_v1', 'missing_policy': 'invalid_null_preserve_grid',
                'implementation_ref': digest('synthetic Raw operator')}
            types = {'values': ('float64_le', 'd'), 'validity': ('bool_u8', 'B'), 'availability': ('int64_le', 'q'),
                'availability_validity': ('bool_u8', 'B'), 'reason_codes': ('int32_le', 'i'),
                'start_session': ('int32_le', 'i'), 'end_session': ('int32_le', 'i'), 'source_codes': ('int32_le', 'i')}
            buffers = {}
            for name, (dtype, code) in types.items():
                values = {'values': [r['return'] or 0.0 for r in rows], 'validity': [int(r['valid']) for r in rows],
                    'availability': [int(datetime.fromisoformat(r['label_available_at']).timestamp()*1000000)
                        if r['label_available_at'] else 0 for r in rows],
                    'availability_validity': [int(r['label_available_at'] is not None) for r in rows],
                    'reason_codes': [0 if r['valid'] else 1 for r in rows],
                    'start_session': [self.calendar.index(r['start_session']) for r in rows],
                    'end_session': [self.calendar.index(r['end_session']) for r in rows], 'source_codes': [0]*len(rows)}[name]
                path = target_root/(name+'.bin'); path.write_bytes(struct.pack('<'+code*len(rows), *values))
                buffers[name] = {'path': str(path), 'file_digest': file_digest(path), 'dtype': dtype,
                    'shape': [len(rows)], 'buffer_digest': file_digest(path)}
            header = seal({'contract_version': 'stock_compact_raw_v1', 'definition': definition,
                'definition_ref': digest(definition), 'row_count': len(rows), 'reason_dictionary': [None, 'SOURCE_MISSING'],
                'source_dictionary': [[price['price_view_ref']]], 'buffers': buffers, 'core_ref': None, 'cohort': None}, 'target_ref')
            write_json(target_root/'target.json', header)
            target = {'descriptor': {'path': str(target_root/'target.json'), 'file_digest': file_digest(target_root/'target.json'),
                'target_ref': header['target_ref']}, 'header': header, 'rows': rows}
            spec = {'contract_version': 'stock_ml_fold_spec_v2', 'fit_cutoff': '2023-12-20T18:00:00Z',
                'simulated_model_available_at': '2023-12-20T19:00:00Z', 'evaluation_cutoff': self.cutoff,
                'inference_cutoff_by_session': {d: d+'T21:00:00Z' for d in days}}
            def selected(kind, selected_days):
                return seal({'contract_version': 'stock_fold_selector_v1', 'kind': kind,
                    'row_index_ref': digest(['synthetic row index', self.calendar, self.universe]),
                    'ranges': grid_ranges(selected_days, self.calendar, 24), 'target_refs': [header['target_ref']],
                    'selected_count': len(selected_days)*24,
                    'keys_digest': digest([[s, d] for d in selected_days for s in self.universe])}, 'selector_ref')
            train, infer = selected('normalized_valid_rows', self.calendar[:2]), selected('complete_grid', days)
            control = seal({'contract_version': 'stock_ml_fold_control_v1', 'fold_spec': spec,
                'evaluation_parts': [target['descriptor']]}, 'fold_control_ref')
            write_json(root/f'control-{n}.json', control)
            inputs = seal({'contract_version': 'stock_ml_saved_inputs_v4', 'prepared_view': prepared,
                'fold_control': {'path': str(root/f'control-{n}.json'), 'file_digest': file_digest(root/f'control-{n}.json'),
                    'fold_control_ref': control['fold_control_ref']}, 'fold_spec_ref': digest(spec),
                'selectors': {'training': train, 'training_labels': train, 'inference': infer,
                    'evaluation_labels': infer, 'validation': None}, 'core_result_refs': [digest('synthetic training Core')]}, 'input_ref')
            folds.append({'input_manifest': inputs, 'fold_spec': spec})
            feature_rows = [{'security_id': security, 'session': day, 'member': i != 0,
                'knowledge_cutoff': day+'T20:00:00Z', 'availability': [day+'T19:00:00Z'],
                'values': [float(i) if i != 2 else None], 'validity': [i != 2], 'source_refs': [digest(['feature', day])]}
                for day in days for i, security in enumerate(self.universe)]
            features = seal({'contract_version': 'stock_feature_slice_v3', 'input_manifest_ref': digest(inputs),
                'prepared_view_ref': prepared['prepared_view_ref'], 'selectors': inputs['selectors'], 'universe': self.universe,
                'ordered_features': ['synthetic_feature'], 'catalog_ref': digest('synthetic catalog'), 'selection': ['synthetic_feature'],
                'training_sessions': self.calendar[:2], 'prediction_sessions': days,
                'parents_by_session': {d: {'feature_ref': digest(['feature-parent', d]), 'qlib_view_ref': digest('synthetic qlib')} for d in days},
                'rows': feature_rows}, 'feature_ref')
            for name in self.inputs:
                path = root/f'{name}-{n}'; path.mkdir()
                model = seal({'contract_version': 'stock_model_release_v2', 'fit_cutoff': spec['fit_cutoff'],
                    'simulated_available_at': spec['simulated_model_available_at'], 'clock_basis': 'declared_simulation',
                    'feature_ref': features['feature_ref'], 'target_semantics': 'forward_5_session_cs_zscore_prediction',
                    'fixture_model': name}, 'model_ref')
                scores = {(security, day): float((i%6) if name == 'a' else ((i*7+self.days.index(day))%11))
                    for day in days for i, security in enumerate(self.universe) if i not in (0, 2)}
                self.scores[name, n] = scores
                signal = seal({'contract_version': 'stock_prediction_run_v2', 'signal_stage': 'prediction_raw',
                    'score_unit': 'dimensionless', 'score_semantics': model['target_semantics'],
                    'feature_ref': features['feature_ref'], 'model_ref': model['model_ref'], 'fold_spec_ref': digest(spec),
                    'clock_basis': 'declared_simulation', 'universe': self.universe,
                    'rows': prediction_rows(features, spec, model, scores)}, 'signal_run_ref')
                fold = seal({'contract_version': 'stock_ml_fold_v3', 'fold_ref': digest(['synthetic fold', name, n]),
                    'definition': {'input_manifest': inputs, 'fold_spec': spec}, 'signal_run_ref': signal['signal_run_ref']}, 'content_digest')
                documents = {'fold.json': fold, 'model.json': model, 'feature-slice.json': features, 'predictions.json': signal}
                for filename, document in documents.items(): write_json(path/filename, document)
                for filename in ('dataset.json', 'label-slice.json', 'signal-evidence.json'): write_json(path/filename, {'synthetic': filename})
                (path/'booster.txt').write_text('synthetic saved bytes; no model execution\n')
                saved = {'contract_version': 'stock_ml_fold_manifest_v2', 'fold_ref': fold['fold_ref'],
                    'files': {p.name: file_digest(p) for p in sorted(path.iterdir())}}
                write_json(path/'manifest.json', saved); documents = {**documents, 'manifest.json': saved}
                descriptor = {'path': str(path/'predictions.json'), 'file_digest': saved['files']['predictions.json'],
                    'signal_run_ref': signal['signal_run_ref']}
                pins = tuple((str(p), file_digest(p)) for p in sorted([*path.iterdir(), *target_root.iterdir(),
                    root/'prepared.json', root/f'control-{n}.json']))
                self.leases[descriptor['path']] = SimpleNamespace(documents=deepcopy(documents), common=deepcopy(self.common),
                    evaluation_targets=[deepcopy(target)], source_records=pins,
                    source_fingerprints={p: file_fingerprint(p) for p, _ in pins})
                if not (missing and name == 'b' and n == 1): self.inputs[name].append(descriptor)
        batch_definition = {'synthetic': True, 'feature_view': {'spec': {'feature_sessions': self.calendar}}}
        manifest = {'contract_version': 'stock_ml_batch_inputs_v4', 'definition': batch_definition,
            'definition_ref': digest(batch_definition), 'prepared_view': prepared, 'folds': folds, 'status': 'COMPLETE'}
        manifest['batch_ref'] = digest(manifest); self.manifest = seal(manifest, 'content_digest')

    def batch(self):
        return StockMLBatchInputs(_TOKEN, {'identity': self.manifest['batch_ref'], 'manifest': self.manifest,
            'matrix_state': SimpleNamespace(store=SimpleNamespace(check=lambda: None), close=lambda: None), 'closed': False, 'metrics': {}})

    @contextmanager
    def hook(self, descriptor, *, batch):
        self.assert_sources(self.leases[descriptor['path']]); self.calls += 1; self.active += 1
        try:
            yield self.leases[descriptor['path']]
            if self.changed is not None: self.changed()
            self.assert_sources(self.leases[descriptor['path']])
        finally: self.active -= 1

    @staticmethod
    def assert_sources(lease):
        for path, expected in lease.source_records:
            if file_digest(path) != expected or file_fingerprint(path) != lease.source_fingerprints[path]:
                raise ValueError('synthetic owner original source changed: '+path)

    def freeze(self, destination):
        with patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold', self.hook, create=True):
            return save_stock_signal_evaluation_inputs(self.inputs, scope=self.scope, destination=destination, batch=self.batch())


class CompactEvaluationTests(unittest.TestCase):
    def test_scope_filter_index_and_pin_checks_have_linear_operation_counts(self):
        class CountedFolds(list):
            iterations = 0
            visits = 0
            def __iter__(self):
                self.iterations += 1
                for fold in super().__iter__():
                    self.visits += 1
                    yield fold
        with tempfile.TemporaryDirectory() as temp:
            owner = OwnerStub(Path(temp)/'sources')
            scope = {**owner.scope, 'sessions': [owner.days[0], owner.days[3]], 'universe': owner.universe[1:]}
            scans, filters, fold_lists = [], [], []
            read_manifest, join, check = StockMLBatchInputs.to_dict, compact._join, frozen._check_marks
            def manifest(batch):
                wire = read_manifest(batch); wire['folds'] = CountedFolds(wire['folds'])
                fold_lists.append(wire['folds']); return wire
            def observed_join(*args):
                filters.append(args[-1]); return join(*args)
            def observed_check(marks):
                scans.append(set(marks)); return check(marks)
            with patch.object(StockMLBatchInputs, 'to_dict', manifest), \
                 patch.object(compact, '_join', observed_join), \
                 patch.object(frozen, '_check_marks', observed_check), \
                 patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold', owner.hook, create=True):
                admitted, records, marks, _ = compact._admit_compact(owner.inputs, None, scope, owner.batch())
            self.assertEqual(len(fold_lists), 1)
            self.assertEqual((fold_lists[0].iterations, fold_lists[0].visits), (1, len(owner.manifest['folds'])))
            expected = [{Path(path) for path, _ in owner.leases[item['path']].source_records}
                for descriptors in owner.inputs.values() for item in descriptors]
            self.assertEqual(scans, [pins for own in expected for pins in (own, own)] + [set(marks)])
            self.assertEqual(sum(map(len, scans)), 2*sum(map(len, expected))+len(records))
            self.assertEqual(len(filters), len(expected))
            self.assertTrue(all(wanted is filters[0] for wanted in filters))
            self.assertEqual(filters[0], (set(scope['universe']), set(scope['sessions'])))
            self.assertEqual(set(admitted['labels']), {(s, d) for d in scope['sessions'] for s in scope['universe']})
            print(json.dumps({'compact_operations': {'leases': len(expected), 'fold_index_passes': fold_lists[0].iterations,
                'fold_index_visits': fold_lists[0].visits, 'scope_set_sizes': list(map(len, filters[0])),
                'lease_pin_entries': sum(map(len, expected)), 'unique_pin_entries': len(records),
                'consumer_stat_checks': sum(map(len, scans))}}))

    def test_current_lease_pre_post_and_final_released_source_mutations_rejected(self):
        for timing in ('before_consume', 'during_consume', 'after_release'):
            with self.subTest(timing=timing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); owner = OwnerStub(root/'sources')
                first = owner.inputs['a'][0]; original_join = compact._join
                changed_path = Path(first['path']).parent/'booster.txt'
                def observed_join(*args):
                    result = original_join(*args)
                    if timing == 'during_consume': changed_path.write_text('changed during consumption')
                    return result
                @contextmanager
                def before(descriptor, *, batch):
                    lease = owner.leases[descriptor['path']]; owner.assert_sources(lease)
                    changed_path.write_text('changed after owner admission')
                    yield lease
                def after():
                    if owner.calls == 5: changed_path.write_text('changed after first lease closed')
                owner.changed = after if timing == 'after_release' else None
                with patch.object(compact, '_join', side_effect=observed_join) as join, \
                     patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold',
                         before if timing == 'before_consume' else owner.hook, create=True):
                    with self.assertRaisesRegex(ValueError, 'frozen evaluation file changed'):
                        save_stock_signal_evaluation_inputs(owner.inputs, scope=owner.scope,
                            destination=root/'inputs', batch=owner.batch())
                    if timing == 'before_consume': join.assert_not_called()
                self.assertEqual(owner.active, 0); self.assertFalse((root/'inputs').exists())

    def test_exact_all_years_and_single_read_zero_compute_hit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
            self.assertEqual(ref.artifact_contract_version, 'stock_signal_evaluation_inputs_v3')
            self.assertEqual(owner.calls, 5); self.assertEqual(owner.active, 0)
            reads = []; original = frozen._read_checked
            def read(path, *args, **kwargs):
                reads.append(Path(path)); return original(path, *args, **kwargs)
            with patch.object(frozen, '_read_checked', side_effect=read):
                periods = evaluate_stock_signal_input_periods(ref, scope=owner.scope, destination=root/'reports')
            source_files = {Path(ref.uri), *(Path(ref.uri).parent/(d+'.json') for d in owner.days)}
            self.assertTrue(all(reads.count(path) == 1 for path in source_files))
            self.assertEqual(list(periods['by_year']), ['2023', '2024'])
            for label, reports in [('all', periods['all']), *periods['by_year'].items()]:
                days = owner.days if label == 'all' else [d for d in owner.days if d[:4] == label]
                keys = {name: digest({'input_signal_refs': [x['signal_run_ref'] for x in descriptors]})
                    for name, descriptors in owner.inputs.items()}
                for native in (False, True):
                    pairs = []
                    for name in owner.inputs:
                        for day in days:
                            n = owner.days.index(day)//2
                            if n == 1 and (name == 'b' or not native): continue
                            for security in owner.universe:
                                if security in ('S00', 'S02', 'S04'): continue
                                pairs.append({'signal_key': keys[name], 'session': day, 'security_id': security,
                                    'score': owner.scores[name, n][security, day], 'outcome': owner.raw[security, day]['return']})
                    expected = evaluate_signal_statistics({'contract_version': 'signal_statistics_input_v1',
                        'sessions': days, 'signal_keys': list(keys.values()), 'pairs': pairs}, spec=SPEC).to_dict()
                    for name, saved in reports.items():
                        report = saved.to_dict(); actual = report['coverage']['native'] if native else report
                        self.assertEqual(actual['series'], [r for r in expected['series'] if r['signal_key'] == keys[name]])
                        self.assertEqual(actual['summary'], next(r for r in expected['summary'] if r['signal_key'] == keys[name]))
                        self.assertEqual(actual['statistics_input_ref'], expected['input_ref'])
                        self.assertEqual(actual['statistics_ref'], expected['statistics_ref'])
                        self.assertEqual(report['contract_version'], 'stock_signal_evidence_v5')
                        self.assertEqual(report['scope']['evaluation_cutoff'], owner.cutoff)
                        self.assertEqual(report['spec'], SPEC)
                        self.assertEqual(load_stock_signal_evaluation(saved.path).to_dict(), report)
            with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('HIT executed Core')):
                hit = evaluate_stock_signal_input_periods(ref, scope=owner.scope, destination=root/'reports')
            self.assertTrue(all(saved.reused for group in [hit['all'], *hit['by_year'].values()] for saved in group.values()))
            _, _, admitted = frozen._load_inputs(ref, owner.scope)
            wanted = [[s, d] for d in owner.days for s in owner.universe if s not in ('S00', 'S02', 'S04')]
            self.assertEqual(admitted['mask']['native_keys']['a'], wanted)
            self.assertEqual(admitted['mask']['native_keys']['b'], [key for key in wanted if key[1] not in owner.days[2:4]])
            self.assertEqual(admitted['mask']['common_keys'], admitted['mask']['native_keys']['b'])
            self.assertEqual(admitted['native_coverage']['b']['excluded_counts'], {'NOT_MEMBER': 6,
                'SIGNAL_INVALID': 4, 'RAW_LABEL_INVALID': 6, 'SIGNAL_MISSING': 44})

    def test_resealed_native_context_buffer_and_lineage_attacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
            wire = _read(ref.uri); records = _sources_table(wire['admission_receipt']['source_records'])
            original = next(iter(wire['raw_metadata']['sources'].values()))
            attacks = [lambda h: h['definition']['price_view']['context']['derivation']['factor_query'].update(pit_policy='other'),
                lambda h: h['definition']['price_view']['context']['derivation']['price_query'].update(price_basis='adjusted'),
                lambda h: h['definition']['price_view']['context']['derivation']['factor_query'].update(purpose='decision_facts'),
                lambda h: h['definition']['price_view']['context']['derivation']['factor_query'].update(domain='market_daily'),
                lambda h: h['buffers']['values'].pop('buffer_digest'),
                lambda h: h['buffers']['values'].update(buffer_digest=digest('different bytes')),
                lambda h: h['buffers']['values'].update(shape=[float(h['row_count'])]),
                lambda h: h['buffers']['values'].update(byte_length=1)]
            for attack in attacks:
                source = deepcopy(original); header = source['header']; attack(header)
                price = header['definition']['price_view']
                price.update(price_view_ref=digest({k: v for k, v in price.items() if k != 'price_view_ref'}))
                header['definition_ref'] = digest(header['definition'])
                header['target_ref'] = digest({k: v for k, v in header.items() if k != 'target_ref'})
                source['descriptor']['target_ref'] = header['target_ref']
                with self.assertRaises(ValueError): compact._header(source, owner.scope, owner.common, records)
            attacks = [lambda t: t.update(input_ref=digest('foreign fold')),
                lambda t: t.update(fold_spec_ref=digest('foreign spec')),
                lambda t: t['selector'].update(ranges=[[0, 48]]),
                lambda t: t['selector'].update(selected_count=47),
                lambda t: t['selector'].update(keys_digest=digest('foreign keys'))]
            for attack in attacks:
                changed = deepcopy(wire); target = changed['raw_metadata']['label_inputs'][0]; attack(target)
                selector = target['selector']; selector['selector_ref'] = digest({k: v for k, v in selector.items() if k != 'selector_ref'})
                with self.assertRaises(ValueError): compact._verify_raw(changed, owner.scope, records)

    def test_shared_context_preserves_legacy_factor_domain_requirement(self):
        with tempfile.TemporaryDirectory() as temp:
            owner = OwnerStub(Path(temp)/'sources'); target = next(iter(owner.leases.values())).evaluation_targets[0]
            definition = target['header']['definition']; context = deepcopy(definition['price_view']['context'])
            context['derivation']['factor_domain'] = 'adjustment_factors'
            raw = {'contract_version': 'stock_label_build_v1',
                'calendar_ref': digest({'contract_version': 'stock_label_calendar_v1', 'sessions': owner.calendar}),
                'label_spec': {**compact._spec(definition), 'adjustment_anchor': context['query']['adjustment_anchor']},
                'source_evidence': {'context': context}}
            universe, cutoff = _validate_raw_label_header(raw, owner.scope, owner.common['snapshot'], owner.common['pit_policy'])
            self.assertEqual(universe, owner.universe); self.assertEqual(cutoff, datetime.fromisoformat(owner.cutoff))
            context['derivation'].pop('factor_domain')
            with self.assertRaisesRegex(ValueError, 'native source query mismatch'):
                _validate_raw_label_header(raw, owner.scope, owner.common['snapshot'], owner.common['pit_policy'])

    def test_unselected_pinned_raw_source_cannot_enter_frozen_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
            wire = _read(ref.uri); raw = wire['raw_metadata']; receipt = wire['admission_receipt']
            source = deepcopy(next(iter(raw['sources'].values()))); header = source['header']
            price = header['definition']['price_view']; price['records_ref'] = digest('unselected Raw source')
            price['price_view_ref'] = digest({k: v for k, v in price.items() if k != 'price_view_ref'})
            header['source_dictionary'] = [[price['price_view_ref']]]
            header['definition_ref'] = digest(header['definition'])
            header['target_ref'] = digest({k: v for k, v in header.items() if k != 'target_ref'})
            path = Path(source['descriptor']['path']).with_name('unselected-target.json'); write_json(path, header)
            source['descriptor'] = {'path': str(path), 'file_digest': file_digest(path), 'target_ref': header['target_ref']}
            raw['sources'][header['target_ref']] = source
            old_records = receipt['source_records']; records = _sources_table(old_records)
            records[str(path)] = file_digest(path)
            receipt['source_records'] = [{'path': p, 'file_digest': records[p]} for p in sorted(records)]
            indices = {item['path']: n for n, item in enumerate(receipt['source_records'])}
            for closures in receipt['source_closure'].values():
                for closure in closures:
                    closure['source_record_indices'] = [indices[old_records[n]['path']] for n in closure['source_record_indices']]
            # Every header and pin is valid; only its absence from evaluation
            # selectors forbids this otherwise fully resealed row/source rebind.
            compact._header(source, owner.scope, owner.common, records)
            for day in header['definition']['sessions']:
                shard_path = Path(ref.uri).parent/wire['shards'][day]['file']; shard = _read(shard_path)
                for row in shard['rows']:
                    row['label']['source_refs'] = [price['price_view_ref']]
                    row['label_leaf_ref'] = compact._binding(raw['label_spec'], raw['snapshot'], row['label'])
                raw['label_shard_refs'][day] = _label_day_ref(shard)
                write_json(shard_path, shard); wire['shards'][day]['file_digest'] = file_digest(shard_path)
            raw['label_ref'] = _label_projection_ref(raw)
            receipt['receipt_ref'] = digest({k: v for k, v in receipt.items() if k != 'receipt_ref'})
            wire['input_id'] = frozen._root_id(wire); write_json(ref.uri, wire)
            changed = replace(ref, artifact_id=wire['input_id'], content_digest=file_digest(ref.uri))
            with self.assertRaisesRegex(ValueError, 'Raw sources must equal consumed evaluation targets'):
                evaluate_stock_signal_inputs(changed, scope=owner.scope, destination=root/'reports')

    def test_resealed_frozen_prediction_clock_and_member_conflicts(self):
        for case in ('clock', 'membership'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
                wire = _read(ref.uri); day = owner.days[0]; path = Path(ref.uri).parent/(day+'.json'); shard = _read(path)
                prediction = shard['rows'][1]['predictions']['a']
                if case == 'clock': prediction['knowledge_cutoff'] = '2025-01-01T00:00:00Z'
                else: prediction['member'] = False
                write_json(path, shard); wire['shards'][day]['file_digest'] = file_digest(path)
                wire['input_id'] = frozen._root_id(wire); write_json(ref.uri, wire)
                changed = replace(ref, artifact_id=wire['input_id'], content_digest=file_digest(ref.uri))
                with self.assertRaisesRegex(ValueError, 'source clock conflict|key/member'):
                    evaluate_stock_signal_inputs(changed, scope=owner.scope, destination=root/'reports')

    def test_overlap_membership_raw_vintage_and_future_clock_rejected(self):
        for case in ('overlap', 'membership', 'raw', 'clock'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); owner = OwnerStub(root/'sources', missing=False)
                if case == 'overlap': owner.inputs['a'].insert(1, owner.inputs['a'][0])
                elif case == 'membership':
                    owner.leases[owner.inputs['b'][0]['path']].documents['feature-slice.json']['rows'][1]['member'] = False
                elif case == 'raw': owner.leases[owner.inputs['b'][0]['path']].evaluation_targets[0]['rows'][1]['return'] += 1
                else: owner.leases[owner.inputs['a'][0]['path']].documents['predictions.json']['rows'][1]['knowledge_cutoff'] = '2025-01-01T00:00:00Z'
                with self.assertRaises(ValueError): owner.freeze(root/'inputs')
                self.assertEqual(owner.active, 0)

    def test_original_source_move_and_mutation_frozen_load_and_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
            reports = evaluate_stock_signal_inputs(ref, scope=owner.scope, destination=root/'reports')
            with patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold', owner.hook, create=True), \
                 patch('axiom_research.stock_batch.load_stock_ml_batch_inputs', side_effect=lambda *a, **k: owner.batch()):
                audit_stock_signal_evaluation(reports['a'].path)
                (owner.root/'control-0.json').write_text('{}')
                self.assertEqual(load_stock_signal_evaluation(reports['a'].path).identity, reports['a'].identity)
                with self.assertRaises(ValueError): audit_stock_signal_evaluation(reports['a'].path)
            source = owner.root.rename(root/'moved-sources')
            self.assertEqual(load_stock_signal_evaluation(reports['a'].path).identity, reports['a'].identity)
            (Path(ref.uri).parent/(owner.days[0]+'.json')).write_text('{}')
            with self.assertRaises(ValueError): load_stock_signal_evaluation(reports['a'].path)

    def test_source_mutation_during_admission_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources')
            owner.changed = lambda: (owner.root/'control-0.json').write_text('{}')
            with self.assertRaises(ValueError): owner.freeze(root/'inputs')
            self.assertEqual(owner.active, 0); self.assertFalse((root/'inputs').exists())

    def test_original_maturity_reason_order_and_owner_hook_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
            _, _, _, admission = frozen._load_inputs(ref, owner.scope, include_admission=True)
            key = 'S01', owner.days[0]; admission['labels'][key]['label_available_at'] = '2025-01-01T00:00:00Z'
            selected = _select_inputs(admission, owner.scope)
            self.assertEqual(selected['native_coverage']['a']['excluded_counts']['LABEL_NOT_MATURE'], 1)
            self.assertNotIn(list(key), selected['mask']['common_keys'])
            earlier = {**owner.scope, 'evaluation_cutoff': '2023-12-31T20:00:00Z'}
            with self.assertRaisesRegex(ValueError, 'source Snapshot/visibility|source cutoff|precedes frozen'):
                evaluate_stock_signal_inputs(ref, scope=earlier, destination=root/'earlier')
            with patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold', None, create=True):
                with self.assertRaisesRegex(ValueError, 'owner admit_stock_signal_evaluation_fold interface required'):
                    save_stock_signal_evaluation_inputs(owner.inputs, scope=owner.scope, destination=root/'absent', batch=owner.batch())

    def test_stdlib_only_saved_report_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); owner = OwnerStub(root/'sources'); ref = owner.freeze(root/'inputs')
            saved = evaluate_stock_signal_inputs(ref, scope=owner.scope, destination=root/'reports')['a']
            code = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'numpy','pandas','pyarrow','lightgbm','qlib','axiom_data','axiom_engine'}:
   raise ImportError('forbidden numeric/source runtime: '+fullname)
sys.meta_path.insert(0,Block())
from axiom_research import load_stock_signal_evaluation
assert load_stock_signal_evaluation(sys.argv[1]).to_dict()['contract_version']=='stock_signal_evidence_v5'
'''
            result = subprocess.run([sys.executable, '-c', code, str(saved.path)], capture_output=True, text=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr)
