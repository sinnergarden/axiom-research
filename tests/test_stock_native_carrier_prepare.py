"""Small staging and one-fold carrier preparation; no real provider/model.

The existing deterministic public Data boundary and real CS Core are used.
Coverage changes context storage only: queries, values and clocks are retained.
"""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_engine.core import execute_cs_zscore_batch
from axiom_research.labels import build_forward_labels
from axiom_research.stock_artifacts import _read, digest, file_digest
from axiom_research.stock_batch import load_stock_ml_batch_inputs
from axiom_research.stock_feature_inputs import _write_feature_matrix_block
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_prepare import _Publisher
from axiom_research.stock_matrix_reader import VerifiedMatrixStore, _admit_raw_label
from axiom_research.stock_matrix_storage import row_index
from test_stock_feature_inputs import file_state
from test_stock_matrix_prepare import PrepareFeatureFixture, PublicDataFixture, Query, Batch
from test_stock_native_carrier_reader import ABSENT, raw_build
from legacy_stock_matrix_fixture import prepare_saved_v2_fixture


LIMIT = 32 * 1024 * 1024
CARRIER = 'stock_native_json_carrier_v1'


def coverage_for(number):
    if number % 3 == 0:
        return {'complete': True, 'observed': [{'key': '甲𝄞', 'value': None}]}
    if number % 3 == 1:
        return None
    return ABSENT


class CoverageFeatureFixture(PrepareFeatureFixture):
    """The original synthetic Feature proof, resealed after context-only data."""
    def day(self, day):
        rows, proof = super().day(day)
        coverage = coverage_for(self.days.index(day))
        if coverage is ABSENT:
            return rows, proof
        replacements = {key: digest({'original_batch_ref': proof[key],
                                    'coverage': coverage})
                        for key in ('adjusted_input_ref', 'membership_ref')}
        sources = {}
        old_bindings = {b['id']: b for b in proof['core_plan']['sources']}
        bindings = []
        for source_id, source in proof['source_evidence'].items():
            source = deepcopy(source)
            query_before = deepcopy(source['query_context']['query'])
            source['query_context']['coverage'] = deepcopy(coverage)
            source['batch_ref'] = replacements['membership_ref' if
                source['field'] == 'is_member' else 'adjusted_input_ref']
            binding = deepcopy(old_bindings[source_id])
            identity = digest({'field': source['field'], 'batch_ref': source['batch_ref'],
                'qualification': binding['qualification'], 'basis': binding['availability_basis']})
            binding['id'] = identity
            sources[identity] = source
            bindings.append(binding)
            if source['query_context']['query'] != query_before:
                raise AssertionError('coverage fixture changed the Query')
        proof.update(replacements)
        proof['source_evidence'] = sources
        proof['core_plan']['sources'] = bindings
        proof['core_frame_ref'] = digest(['synthetic coverage frame', day, proof['core_plan']])
        for row in rows:
            row['source_refs'] = [proof['core_frame_ref'], digest(proof['core_plan'])]
        return rows, proof


class CoverageDataFixture(PublicDataFixture):
    """Add deterministic coverage to the native context; adjust inherits it."""
    def read(self, *, snapshot, query):
        batch = super().read(snapshot=snapshot, query=query)
        before = deepcopy(batch.context['query'])
        mode = self.spec['calendar'].index(query.sessions[0])
        coverage = coverage_for(mode)
        if coverage is not ABSENT:
            batch.context['coverage'] = deepcopy(coverage)
        if batch.context['query'] != before:
            raise AssertionError('coverage fixture changed native Query')
        return batch


class NativeCarrierPrepareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='axiom-native-prepare-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def publisher(self):
        stage, target = self.root / 'stage', self.root / 'published'
        stage.mkdir()
        publisher = _Publisher(stage, target, maximum_resident_bytes=LIMIT)
        self.addCleanup(lambda: None if publisher.store.closed else publisher.store.close())
        return publisher

    def assert_final_blob_paths(self, wrapper, target):
        self.assertIn(wrapper['contract_version'], (CARRIER,'stock_native_json_carrier_v2'))
        self.assertTrue(wrapper['coverage_slots']+wrapper.get('wire_slots',[]))
        for slot in wrapper['coverage_slots']+wrapper.get('wire_slots',[]):
            self.assertTrue(Path(slot['bytes']['path']).is_relative_to(target))
            self.assertNotIn('/stage/', slot['bytes']['path'])

    def test_publisher_raw_null_absent_dispatch_and_stage_rename(self):
        publisher = self.publisher()
        original = [raw_build(None), raw_build()]
        descriptors = [publisher.part(value, 'label_ref') for value in original]
        null_wrapper = _read(publisher.actual(descriptors[0]))
        self.assert_final_blob_paths(null_wrapper, publisher.target)
        null_blob = null_wrapper['coverage_slots'][0]['bytes']
        self.assertEqual(null_blob['shape'], [4])
        self.assertEqual(publisher.resolve(null_blob['path']).read_bytes(), b'null')
        self.assertEqual(_read(publisher.actual(descriptors[1])), original[1])
        for desc, expected in zip(descriptors, original):
            value = publisher.read_part(desc, 'label_ref')
            self.assertEqual(value['label_ref'], expected['label_ref'])
            self.assertEqual(publisher.native_digest(value, exclude_ref_key='label_ref'),
                             expected['label_ref'])
            publisher.release_part(desc, 'label_ref')
        publisher.finish()
        publisher.stage.rename(publisher.target)
        with VerifiedMatrixStore(maximum_matrix_bytes=LIMIT) as store:
            for desc in descriptors:
                value = store.read_json(desc, 'label_ref')
                self.assertEqual(store.native_digest(value, exclude_ref_key='label_ref'), desc['label_ref'])
                self.assertEqual(file_digest(desc['path']), desc['file_digest'])
                for path in store.source_paths(desc): self.assertTrue(Path(path).is_file())

    def test_publisher_label_metadata_reuses_admitted_coverage_without_dom(self):
        publisher = self.publisher()
        selected = {'context': {'domain': 'market_daily', 'query': {'fields': ['close']},
                                'coverage': {'observed': [1, None]}},
                    'records': [{'security_id': 'A', 'session': '2026-01-02', 'close': 1.0}],
                    'field_meta': {}}
        raw = raw_build(selected['context']['coverage'])
        raw_desc = publisher.part(raw, 'label_ref')
        metadata = seal({'contract_version': 'stock_matrix_label_metadata_v1',
            'role': 'training', 'raw_build': raw_desc, 'rows': [],
            'contents': {digest(selected): selected}, 'core_result_refs': []}, 'metadata_ref')
        desc = publisher.part(metadata, 'metadata_ref')
        admitted = publisher.read_part(desc, 'metadata_ref')
        selected_ref = digest(selected)
        self.assertNotIn('coverage', admitted['contents'][selected_ref]['context'])
        self.assertEqual(publisher.native_digest(admitted['contents'][selected_ref]), selected_ref)
        normalized = dict(admitted)
        normalized['core_result_refs'] = [digest('synthetic saved Core reference')]
        normalized = publisher.seal(normalized, 'metadata_ref')
        normalized_desc = publisher.part(normalized, 'metadata_ref')
        normalized_wrapper = _read(publisher.actual(normalized_desc))
        original_wrapper = _read(publisher.actual(desc))
        self.assertEqual(normalized_wrapper['coverage_slots'], original_wrapper['coverage_slots'])
        self.assertNotEqual(normalized_desc['metadata_ref'], desc['metadata_ref'])
        reread = publisher.read_part(normalized_desc, 'metadata_ref')
        self.assertNotIn('coverage', reread['contents'][selected_ref]['context'])
        self.assertEqual(publisher.native_digest(reread, exclude_ref_key='metadata_ref'),
                         normalized_desc['metadata_ref'])
        self.assertEqual(publisher.store.metrics['label_wire_validation_calls'], 1)
        publisher.finish()
        publisher.stage.rename(publisher.target)
        for saved in (raw_desc, desc, normalized_desc):
            self.assert_final_blob_paths(_read(saved['path']), publisher.target)
        with VerifiedMatrixStore(maximum_matrix_bytes=LIMIT) as store:
            reread = store.read_json(normalized_desc, 'metadata_ref')
            self.assertNotIn('coverage', reread['contents'][selected_ref]['context'])

    def test_feature_writer_preserves_original_proof_refs_with_final_inner_paths(self):
        feature = CoverageFeatureFixture(self.root)
        publisher = self.publisher()
        day = feature.days[0]
        rows, proof = feature.day(day)
        original_rows, original_proof = deepcopy(rows), deepcopy(proof)
        schema = [{'name': name, 'dtype': 'float64', 'unit': 'dimensionless',
                   'stage': 'cross_sectional', 'missing': 'preserve'} for name in feature.columns]
        index = row_index([day], feature.universe)
        parts, selection = _write_feature_matrix_block(publisher.stage, rows=rows, proof=[proof],
            spec=feature.spec, view=feature.view, schema=schema, index_ref=index['row_index_ref'],
            row_offset=0, options={'column_block': 32, 'maximum_resident_bytes': LIMIT},
            universe_id=feature.universe_id, metrics=publisher.metrics,
            descriptor_mapper=publisher._desc, path_resolver=publisher.resolve)
        self.assertEqual(rows, original_rows)
        self.assertEqual(proof, original_proof)
        self.assertEqual(len(parts), 1)
        desc = publisher._desc(parts[0]['metadata'])
        wrapper = _read(publisher.actual(desc))
        self.assert_final_blob_paths(wrapper, publisher.target)
        metadata = publisher.read_part(desc, 'metadata_ref')
        self.assertEqual(metadata['rows'][0]['source_refs'], original_rows[0]['source_refs'])
        self.assertEqual(metadata['input_evidence'][0]['core_plan'], original_proof['core_plan'])
        self.assertEqual(selection[0]['selected_versions_ref'], digest(original_proof['source_evidence']))
        self.assertEqual(publisher.native_digest(metadata['contents'][selection[0]['selected_versions_ref']]),
                         selection[0]['selected_versions_ref'])
        self.assertEqual(publisher.native_digest(metadata, exclude_ref_key='metadata_ref'), desc['metadata_ref'])
        publisher.finish()
        publisher.stage.rename(publisher.target)
        with VerifiedMatrixStore(maximum_matrix_bytes=LIMIT) as store:
            value = store.read_json(desc, 'metadata_ref')
            self.assertEqual(value['original_feature_ref'], metadata['original_feature_ref'])
            self.assertTrue(set(store.source_paths(desc)) <= {str(p) for p in publisher.target.rglob('*') if p.is_file()})

    def test_one_fold_three_holder_chain_actual_cs_saved_read_and_zero_work_hit(self):
        feature = CoverageFeatureFixture(self.root)
        feature_options = {'layout': 'matrix_v1', 'row_block_sessions': 8,
                           'column_block': 32, 'maximum_resident_bytes': LIMIT}
        saved = feature.matrix(options=feature_options)
        self.addCleanup(saved.close)
        data = CoverageDataFixture(feature.spec)
        folds = feature.folds()[:1]
        options = {'row_block_sessions': 32, 'column_block': 32,
                   'maximum_resident_bytes': LIMIT, 'normalization_backend': 'core_cs_batch_v1'}
        module = types.ModuleType('axiom_data')
        module.QuerySpec, module.adjust_prices = Query, data.adjust
        metrics = {}
        with patch.dict(sys.modules, {'axiom_data': module}), \
             patch('axiom_research.stock_ml._implementation', return_value=feature.implementation), \
             patch('axiom_research.stock_ml._environment', return_value=feature.environment), \
             patch('axiom_engine.core.execute_cs_zscore_batch', wraps=execute_cs_zscore_batch) as core, \
             patch('axiom_research.stock_ml._iter_stock_feature_days', side_effect=AssertionError('Feature recomputed')), \
             patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('model executed')):
            batch = prepare_saved_v2_fixture(data, feature_inputs=saved, fold_specs=folds,
                destination=self.root/'prepared', preparation_options=options, metrics=metrics)
            self.assertEqual(core.call_count, 1)
            self.assertEqual(metrics['core_calls'], 1)
            self.assertEqual(metrics['label_core_calls'], 1)
            self.assertEqual(metrics['feature_core_calls'], 0)
            self.assertEqual(metrics['train_calls'], 0)
            self.assertEqual(metrics['predict_calls'], 0)
            self.assertEqual(metrics['account_calls'], 0)
            self.assertEqual(metrics['data_read_calls'], len(data.queries))
            self.assertEqual(len(batch['folds']), 1)
            self.assertEqual(batch['status'], 'COMPLETE')
            view = _read(batch['prepared_view']['path'])
            target = Path(batch['prepared_view']['path']).parents[1]
            original_wires = {digest(w): w for w in data.adjusted}
            carried_raw = {}; all_blobs = set(); holder_kinds = set()
            feature_index = saved.to_dict()
            metadata_descriptors = [p['metadata'] for p in feature_index['partitions']]
            metadata_descriptors += [p['metadata'] for p in view['partitions']]
            raw_descriptors = {}
            for descriptor in metadata_descriptors:
                wrapper = _read(descriptor['path'])
                if wrapper.get('contract_version') in (CARRIER,'stock_native_json_carrier_v2'):
                    holder_kinds.add(wrapper['skeleton']['contract_version'])
                    for slot in wrapper['coverage_slots']+wrapper.get('wire_slots',[]):
                        all_blobs.add(slot['bytes']['path'])
                        self.assertTrue(Path(slot['bytes']['path']).is_file())
                        self.assertNotIn('/.stock-matrix-', slot['bytes']['path'])
                    body = wrapper['skeleton']
                else:
                    body = wrapper
                if 'raw_build' in body:
                    raw_descriptors[body['raw_build']['path']] = body['raw_build']
            for descriptor in raw_descriptors.values():
                wrapper = _read(descriptor['path'])
                if wrapper.get('contract_version') in (CARRIER,'stock_native_json_carrier_v2'):
                    holder_kinds.add(wrapper['skeleton']['contract_version'])
                    self.assert_final_blob_paths(wrapper, target)
                    carried_raw[descriptor['label_ref']] = {
                        'representation': CARRIER, 'carrier': descriptor}
                    all_blobs.update(s['bytes']['path'] for s in wrapper['coverage_slots'])
                with _admit_raw_label(descriptor) as lease:
                    actual = lease.raw
                    original_wire = original_wires[actual['source_ref']]
                    days = sorted({r['feature_session'] for r in actual['rows']})
                    expected = build_forward_labels(Batch(original_wire), calendar=feature.calendar,
                                                    feature_sessions=days)
                    self.assertEqual(actual['label_ref'], expected['label_ref'])
                    self.assertEqual(actual['rows'], expected['rows'])
            self.assertEqual(holder_kinds, {'stock_matrix_feature_metadata_v1',
                'stock_matrix_label_metadata_v1', 'stock_label_build_v1'})
            self.assertTrue(carried_raw)
            self.assertTrue(all_blobs)
            before = file_state(self.root)
            counters = {}
            with patch.object(data, 'read', side_effect=AssertionError('HIT read Data')), \
                 patch('axiom_engine.core.execute_cs_zscore_batch', side_effect=AssertionError('HIT ran Core')):
                with load_stock_ml_batch_inputs(batch) as verified:
                    fold = batch['folds'][0]
                    with verified._matrix_project(fold['input_manifest'], fold['fold_spec']) as projection:
                        self.assertEqual(projection.X.shape, (122, 6))
                        self.assertEqual(projection.X.tolist(), [[float(i) for i in range(1, 7)]] * 122)
                        for i, value in enumerate(projection.y.tolist()):
                            self.assertAlmostEqual(value, -1.0 if i % 2 == 0 else 1.0, places=12)
                    source_records = verified._evaluation_source_records()
                    with verified._project_evaluation(fold['input_manifest'], fold['fold_spec']) as lease:
                        self.assertEqual(lease.fold_binding['raw_provenance_storage'], carried_raw)
                        paths = {source_records[i][0] for i in lease.source_record_indices}
                        self.assertTrue(all_blobs <= paths)
                        self.assertFalse(any('/.stock-matrix-' in path for path in paths))
                        self.assertEqual(len(lease.raw_provenance[0]), 7)
                    self.assertEqual(verified.metrics['data_read_calls'], 0)
                    self.assertEqual(verified.metrics['core_calls'], 0)
                again = prepare_saved_v2_fixture(data, feature_inputs=saved, fold_specs=folds,
                    destination=self.root/'prepared', preparation_options=options, metrics=counters)
            self.assertEqual(again, batch)
            self.assertTrue(counters['cache_hit'])
            for name in ('data_read_calls', 'core_calls', 'feature_core_calls',
                         'train_calls', 'predict_calls', 'account_calls'):
                self.assertEqual(counters[name], 0)
            self.assertEqual(file_state(self.root), before)
            self.assertLessEqual(sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()), LIMIT)
            for query in data.queries:
                self.assertEqual(query['symbols'], feature.universe)
                self.assertEqual(query['purpose'], 'label_outcomes')
                self.assertEqual(query['price_basis'], 'unadjusted')


if __name__ == '__main__':
    unittest.main()
