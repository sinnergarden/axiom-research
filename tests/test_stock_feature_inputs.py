"""Synthetic storage acceptance; no provider reads or numerical executions.

The pipeline hooks emit complete ordinary v1 documents. These tests exercise
saved closure, publication, and resume; numerical equivalence is tested at the
actual Core adapter/iterator boundary elsewhere.
"""
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import load_stock_feature_inputs
from axiom_research.feature_catalog import load_feature_catalog
from axiom_research.stock_artifacts import _read, digest, file_digest, write_json
from axiom_research.stock_feature_inputs import build_stock_feature_inputs
from axiom_research.stock_fold_inputs import seal


class InterruptedPreparation(RuntimeError):
    pass


class NoData:
    """Any accidental use of the provider makes synthetic acceptance fail."""

    def __getattr__(self, name):
        raise AssertionError('synthetic storage test accessed Data: ' + name)


class SyntheticInputs:
    """Actual small files with a fixed scope and complete v1 source proof."""

    def __init__(self, root, *, feature_count=5):
        self.root = root
        self.catalog = load_feature_catalog()
        self.selection = [self.catalog.default_selection[0].to_dict()]
        self.history = self.catalog.select(self.selection)[0]['lookback']
        self.universe = ['A', 'B', 'C']
        self.universe_id = 'synthetic_membership_v1'
        self.snapshot = 's-synthetic-fixed'
        self.pit = 'best_effort_vendor_v1'
        start = date(2026, 1, 2)
        self.calendar = [(start + timedelta(days=2*i)).isoformat()
                         for i in range(self.history - 1 + feature_count)]
        self.days = self.calendar[self.history-1:]
        request = {'snapshot': self.snapshot, 'pit_policy': self.pit,
                   'universe_id': self.universe_id}
        result = {'read_sessions': self.calendar, 'read_symbols': self.universe,
                  'snapshot_id': self.snapshot, 'pit_policy': self.pit}
        source_proof = {'synthetic': True, 'provider_reads': 0}
        scope = seal({'contract_version': 'private_frozen_scope_v1',
            'request': request, 'request_ref': digest(request),
            'result': result, 'result_ref': digest(result),
            'source_proof': source_proof, 'source_proof_ref': digest(source_proof)},
            'scope_bundle_ref')
        write_json(root/'scope.json', scope)
        self.spec = {'contract_version': 'stock_feature_inputs_spec_v1',
            'scope': {'path': str((root/'scope.json').resolve()),
                      'file_digest': file_digest(root/'scope.json'),
                      'scope_bundle_ref': scope['scope_bundle_ref']},
            'snapshot': self.snapshot, 'pit_policy': self.pit,
            'calendar': self.calendar, 'universe': self.universe,
            'catalog_ref': self.catalog.identity, 'feature_selection': self.selection,
            'ordered_features': [self.selection[0]['id']],
            'read_sessions': self.calendar, 'feature_sessions': self.days,
            'cutoff_by_session': {d: d+'T20:30:00+08:00' for d in self.calendar}}
        queries = [self.query('market_daily',
                             ['open', 'high', 'low', 'close', 'amount_cny']),
                   self.query('adjustment_factors', ['factor']),
                   self.query('universe_membership', ['is_member'])]
        body = {'schema_version': 'axiom_qlib_view_v1', 'snapshot_id': self.snapshot,
            'queries': queries[:2], 'fields': ['open', 'high', 'low', 'close', 'amount_cny'],
            'instrument_map': {s: s for s in self.universe}, 'calendar': self.calendar,
            'universe_query': queries[2], 'universe_name': self.universe_id,
            'numeric_format': 'synthetic_storage_only', 'reader_version': 'synthetic/1',
            'exporter_version': 'synthetic/1', 'limitations': ['synthetic; not numerical evidence'],
            'files': {'data.bin': {'size': 4, 'sha256': sha256(b'test').hexdigest()}}}
        self.manifest = {**body, 'view_id': digest(body)[7:]}
        self.view = {k: deepcopy(v) for k, v in self.manifest.items() if k != 'files'}
        self.prepare_calls = 0
        self.computed = []
        self.requested = []
        self.implementation = {'synthetic': digest('fixed source')}
        self.environment = {'synthetic': 'fixed environment'}

    def query(self, domain, fields, *, sessions=None, cutoff=None, adjusted=False):
        sessions = self.calendar if sessions is None else sessions
        query = {'domain': domain, 'fields': list(fields), 'symbols': self.universe,
                 'sessions': list(sessions), 'pit_policy': self.pit,
                 'purpose': 'decision_facts',
                 'price_basis': 'common_anchor_adjusted_v1' if adjusted else 'unadjusted',
                 'policy_by_session': None, 'adjustment_anchor': None, 'universe_id': None,
                 'cutoff_by_session': {d: cutoff or self.spec['cutoff_by_session'][d]
                                       for d in sessions}}
        if domain == 'universe_membership':
            query['universe_id'] = self.universe_id
        if adjusted:
            query['adjustment_anchor'] = cutoff[:10]
        return query

    def prepare(self, data, *, config, destination):
        self.prepare_calls += 1
        if data is not self.data:
            raise AssertionError('provider object changed')
        if config['read_sessions'] != self.calendar:
            raise AssertionError('global Qlib calendar changed')
        if config['universe_id'] != self.universe_id:
            raise AssertionError('scope universe was guessed')
        qlib = destination/'qlib'
        qlib.mkdir(exist_ok=True)
        # Existing files are retained so a corrupt resume cannot heal them.
        if not (qlib/'axiom-qlib.json').exists():
            (qlib/'data.bin').write_bytes(b'test')
            write_json(qlib/'axiom-qlib.json', self.manifest)
        return {'view_reference': deepcopy(self.view), 'qlib_path': qlib,
                'values': {}, 'seconds': 0.0}

    def day(self, day):
        i = self.calendar.index(day)
        history = self.calendar[i-self.history+1:i+1]
        cutoff = self.spec['cutoff_by_session'][day]
        price = self.query('market_daily', ['open', 'high', 'low', 'close', 'amount_cny'], sessions=history,
                           cutoff=cutoff, adjusted=True)
        membership = self.query('universe_membership', ['is_member'],
                                sessions=history, cutoff=cutoff)
        adjusted_ref = digest({'synthetic_adjusted': price})
        membership_ref = digest({'synthetic_membership': membership})
        plan = {'synthetic_session': day, 'reference_members': {day: ['A', 'B']}}
        frame_ref = digest(['synthetic frame', day])
        proof = {'session': day, 'sessions': history,
                 'cutoffs': {d: cutoff for d in history},
                 'core_plan': plan, 'core_frame_ref': frame_ref,
                 'adjusted_input_ref': adjusted_ref, 'membership_ref': membership_ref,
                 'source_evidence': {
                     'price': {'field': 'close', 'batch_ref': adjusted_ref,
                               'query_context': {'snapshot_id': self.snapshot,
                                                 'domain': 'market_daily', 'query': price}},
                     'membership': {'field': 'is_member', 'batch_ref': membership_ref,
                                    'query_context': {'snapshot_id': self.snapshot,
                                                      'domain': 'universe_membership',
                                                      'query': membership}}}}
        rows = [{'security_id': security, 'session': day,
                 'values': [float(100*i+j)], 'validity': [True],
                 'availability': [day+'T20:00:00+08:00'], 'reasons': [[]],
                 'member': security != 'C', 'knowledge_cutoff': cutoff,
                 'source_refs': [frame_ref, digest(plan)]}
                for j, security in enumerate(self.universe)]
        return rows, proof

    def iterate(self, data, *, config, catalog, chosen, qlib_inputs, stats,
                history_sessions, **unused):
        if data is not self.data or history_sessions != self.history:
            raise AssertionError('original provider/history changed')
        self.requested.append(list(config['feature_sessions']))
        for day in config['feature_sessions']:
            self.computed.append(day)
            stats['core_calls'] += 1
            stats['feature_core_calls'] += 1
            yield self.day(day)

    def build(self, *, destination=None, spec=None, shard_sessions=2, progress=None):
        self.data = NoData()
        with patch('axiom_research.stock_ml._implementation', return_value=self.implementation), \
             patch('axiom_research.stock_ml._environment', return_value=self.environment), \
             patch('axiom_research.stock_ml._prepare_stock_qlib', side_effect=self.prepare), \
             patch('axiom_research.stock_ml._iter_stock_feature_days', side_effect=self.iterate):
            return build_stock_feature_inputs(self.data, spec=self.spec if spec is None else spec,
                destination=self.root/'saved' if destination is None else destination,
                shard_sessions=shard_sessions, progress=progress)


def reseal_index(path, value):
    value = {k: v for k, v in value.items() if k != 'content_digest'}
    value['feature_inputs_ref'] = digest({k: value[k] for k in
                                         ('definition_ref', 'qlib_manifest', 'feature_parents')})
    write_json(path/'index.json', seal(value, 'content_digest'))


def reseal_first_parent_proof(saved, original, proof):
    """Rebind all stored hashes, so rejection tests check the source contract."""
    wire = deepcopy(original)
    parent = wire['feature_parents'][0]
    write_json(parent['input_evidence']['path'], proof)
    feature = _read(parent['features']['path'])
    feature.pop('feature_ref')
    feature['input_evidence_ref'] = digest(proof)
    feature = seal(feature, 'feature_ref')
    write_json(parent['features']['path'], feature)
    parent['input_evidence'].update(file_digest=file_digest(parent['input_evidence']['path']),
                                    input_evidence_ref=digest(proof))
    parent['features'].update(file_digest=file_digest(parent['features']['path']),
                              feature_ref=feature['feature_ref'])
    reseal_index(saved.path, wire)


def saved_values(saved):
    rows, proof = [], []
    for parent in saved.feature_parents:
        rows.extend(_read(parent['features']['path'])['rows'])
        proof.extend(_read(parent['input_evidence']['path']))
    return rows, proof


def file_state(root):
    return {str(p.relative_to(root)): (file_digest(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


class StockFeatureInputTests(unittest.TestCase):
    def test_scope_change_after_admission_never_publishes_complete_index(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); fixture.data = NoData()
            def changed_environment():
                scope = Path(fixture.spec['scope']['path'])
                scope.write_text(scope.read_text()+' ')
                return fixture.environment
            with patch('axiom_research.stock_ml._implementation', return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment', side_effect=changed_environment), \
                 patch('axiom_research.stock_ml._prepare_stock_qlib', side_effect=AssertionError('changed scope prepared')):
                with self.assertRaisesRegex(ValueError, 'scope changed after admission'):
                    build_stock_feature_inputs(fixture.data,spec=fixture.spec,destination=Path(temp)/'saved')
            self.assertFalse((Path(temp)/'saved').exists())

    def test_late_qlib_file_change_never_publishes_complete_index(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp))
            def change_after_first_parent(update):
                if update['completed_dates'] == 2:
                    manifest, = (Path(temp)/'saved').glob('*/qlib/axiom-qlib.json')
                    (manifest.parent/'late.bin').write_bytes(b'unlisted')
            with self.assertRaisesRegex(ValueError, 'scope/Qlib changed during preparation'):
                fixture.build(progress=change_after_first_parent)
            target, = (Path(temp)/'saved').iterdir()
            self.assertTrue((target/'checkpoint.json').exists())
            self.assertFalse((target/'index.json').exists())

    def test_two_date_parents_preserve_whole_v1_rows_and_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp))
            sharded = fixture.build()
            whole = fixture.build(shard_sessions=len(fixture.days))
            self.assertFalse(sharded.reused)
            self.assertEqual([p['sessions'] for p in sharded.feature_parents],
                             [fixture.days[:2], fixture.days[2:4], fixture.days[4:]])
            self.assertEqual(saved_values(sharded), saved_values(whole))
            expected = [fixture.day(d) for d in fixture.days]
            self.assertEqual(saved_values(sharded),
                             ([r for rows, _ in expected for r in rows],
                              [p for _, p in expected]))
            for parent in sharded.feature_parents:
                feature = _read(parent['features']['path'])
                self.assertEqual(feature['contract_version'], 'stock_feature_build_v1')
                self.assertEqual(feature['qlib_view'], sharded.to_dict()['qlib_view'])
                self.assertEqual(feature['input_evidence_ref'],
                                 digest(_read(parent['input_evidence']['path'])))
                self.assertEqual(set(Path(parent['features']['path']).parent.iterdir()),
                                 {Path(parent['features']['path']), Path(parent['input_evidence']['path'])})
            self.assertEqual(load_stock_feature_inputs(sharded.path).identity, sharded.identity)

    def test_partial_checkpoint_resume_computes_only_missing_complete_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp))
            def stop_after_first_parent(update):
                self.assertEqual(update['completed_dates'], 2)
                raise InterruptedPreparation('after atomic parent publication')
            with self.assertRaises(InterruptedPreparation):
                fixture.build(progress=stop_after_first_parent)
            target, = (Path(temp)/'saved').iterdir()
            self.assertFalse((target/'index.json').exists())
            checkpoint = _read(target/'checkpoint.json')
            self.assertEqual(checkpoint['contract_version'], 'stock_feature_inputs_checkpoint_v1')
            self.assertEqual(checkpoint['feature_parents'][0]['sessions'], fixture.days[:2])
            before = {name: file_state(Path(desc['path']).parent)
                      for name, desc in checkpoint['feature_parents'][0].items() if name != 'sessions'}
            with self.assertRaises(FileNotFoundError): load_stock_feature_inputs(target)
            resumed = fixture.build()
            self.assertEqual(fixture.computed, fixture.days)
            self.assertEqual(fixture.requested, [fixture.days, fixture.days[2:]])
            self.assertEqual(fixture.prepare_calls, 2)
            self.assertEqual(resumed.path, target)
            self.assertEqual(saved_values(resumed),
                             ([r for d in fixture.days for r in fixture.day(d)[0]],
                              [fixture.day(d)[1] for d in fixture.days]))
            for name, desc in checkpoint['feature_parents'][0].items():
                if name != 'sessions': self.assertEqual(file_state(Path(desc['path']).parent), before[name])
            self.assertFalse(any(p.name.startswith('.feature-') for p in (target/'parents').iterdir()))

    def test_incomplete_first_shard_has_no_checkpoint_or_complete_index(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp))
            original = fixture.iterate
            def interrupt(data, **kwargs):
                iterator = original(data, **kwargs)
                yield next(iterator)
                raise InterruptedPreparation('inside unpublished shard')
            fixture.iterate = interrupt
            with self.assertRaises(InterruptedPreparation): fixture.build()
            target, = (Path(temp)/'saved').iterdir()
            self.assertFalse((target/'index.json').exists())
            self.assertFalse((target/'checkpoint.json').exists())
            self.assertFalse((target/'parents').exists())
            fixture.iterate = original
            saved = fixture.build()
            self.assertEqual(fixture.requested[-1], fixture.days)
            self.assertEqual(len(saved.feature_parents), 3)

    def test_failed_temporary_shard_write_leaves_no_published_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp))
            def fail_proof_write(path, value):
                if Path(path).name == 'feature-inputs.json':
                    raise InterruptedPreparation('before shard publication')
                write_json(path, value)
            with patch('axiom_research.stock_feature_inputs.write_json', side_effect=fail_proof_write):
                with self.assertRaises(InterruptedPreparation): fixture.build()
            target, = (Path(temp)/'saved').iterdir()
            self.assertFalse((target/'index.json').exists())
            self.assertFalse((target/'checkpoint.json').exists())
            self.assertEqual(list((target/'parents').iterdir()), [])
            saved = fixture.build()
            self.assertEqual(len(saved.feature_parents), 3)
            self.assertEqual(fixture.requested, [fixture.days, fixture.days])

    def test_complete_cache_does_not_prepare_read_or_execute(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build()
            before = file_state(Path(temp))
            with patch('axiom_research.stock_ml._prepare_stock_qlib', side_effect=AssertionError('HIT prepared')), \
                 patch('axiom_research.stock_ml._iter_stock_feature_days', side_effect=AssertionError('HIT executed')), \
                 patch('axiom_research.stock_ml._implementation', return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment', return_value=fixture.environment):
                cached = build_stock_feature_inputs(NoData(), spec=fixture.spec,
                                                    destination=Path(temp)/'saved')
            self.assertTrue(cached.reused)
            self.assertEqual(cached.identity, saved.identity)
            self.assertEqual(file_state(Path(temp)), before)
            public = cached.to_dict(); public['feature_parents'].clear()
            parents = cached.feature_parents; parents[0]['sessions'].clear()
            self.assertEqual(cached.identity, saved.identity)
            self.assertEqual(cached.feature_parents, saved.feature_parents)

    def test_resealed_unordered_duplicate_and_missing_parent_dates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build(); original = saved.to_dict()
            for change in ('unordered', 'duplicate', 'missing'):
                with self.subTest(change=change):
                    wire = deepcopy(original)
                    if change == 'unordered': wire['feature_parents'][0]['sessions'].reverse()
                    elif change == 'duplicate':
                        wire['feature_parents'][1]['sessions'] = wire['feature_parents'][0]['sessions'][:]
                    else: wire['feature_parents'].pop()
                    reseal_index(saved.path, wire)
                    with self.assertRaises(ValueError): load_stock_feature_inputs(saved.path)
            write_json(saved.path/'index.json', original)
            self.assertEqual(load_stock_feature_inputs(saved.path).identity, saved.identity)

    def test_resealed_missing_or_duplicate_security_grid_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build(); original = saved.to_dict()
            parent = original['feature_parents'][0]
            original_feature = _read(parent['features']['path'])
            for change in ('missing', 'duplicate'):
                with self.subTest(change=change):
                    feature = {k: deepcopy(v) for k, v in original_feature.items() if k != 'feature_ref'}
                    if change == 'missing': feature['rows'].pop()
                    else: feature['rows'].append(deepcopy(feature['rows'][0]))
                    feature = seal(feature, 'feature_ref'); write_json(parent['features']['path'], feature)
                    wire = deepcopy(original)
                    wire['feature_parents'][0]['features'].update(
                        file_digest=file_digest(parent['features']['path']), feature_ref=feature['feature_ref'])
                    reseal_index(saved.path, wire)
                    with self.assertRaisesRegex(ValueError, 'grid'):
                        load_stock_feature_inputs(saved.path)

    def test_resealed_proof_clock_window_and_membership_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build(); original = saved.to_dict()
            parent = original['feature_parents'][0]
            original_feature = _read(parent['features']['path'])
            original_proof = _read(parent['input_evidence']['path'])
            for change, reason in (('source_clock', 'source query'),
                                   ('window', 'window/clock'), ('membership', 'member/value')):
                with self.subTest(change=change):
                    proof = deepcopy(original_proof)
                    day = proof[0]['session']
                    if change == 'source_clock':
                        source_query = proof[0]['source_evidence']['price']['query_context']['query']
                        source_query['cutoff_by_session'][day] = day+'T21:00:00+08:00'
                    elif change == 'window':
                        proof[0]['sessions'].pop(0)
                    else:
                        proof[0]['core_plan']['reference_members'][day].append('C')
                    write_json(parent['input_evidence']['path'], proof)
                    feature = {k: deepcopy(v) for k, v in original_feature.items() if k != 'feature_ref'}
                    feature['input_evidence_ref'] = digest(proof)
                    for row in feature['rows']:
                        if row['session'] == day:
                            row['source_refs'] = [proof[0]['core_frame_ref'], digest(proof[0]['core_plan'])]
                    feature = seal(feature, 'feature_ref'); write_json(parent['features']['path'], feature)
                    wire = deepcopy(original)
                    wire['feature_parents'][0]['features'].update(
                        file_digest=file_digest(parent['features']['path']), feature_ref=feature['feature_ref'])
                    wire['feature_parents'][0]['input_evidence'].update(
                        file_digest=file_digest(parent['input_evidence']['path']), input_evidence_ref=digest(proof))
                    reseal_index(saved.path, wire)
                    with self.assertRaisesRegex(ValueError, reason): load_stock_feature_inputs(saved.path)

    def test_parent_byte_corruption_prevents_load_and_complete_cache_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build()
            path = Path(saved.feature_parents[0]['input_evidence']['path'])
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'evidence'): load_stock_feature_inputs(saved.path)
            before = fixture.prepare_calls
            with self.assertRaisesRegex(ValueError, 'evidence'): fixture.build()
            self.assertEqual(fixture.prepare_calls, before)
            self.assertEqual(fixture.computed, fixture.days)

    def test_resealed_source_queries_cannot_override_fixed_input_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build(); original = saved.to_dict()
            original_proof = _read(original['feature_parents'][0]['input_evidence']['path'])
            cases = (
                ('price', 'policy_by_session', {fixture.calendar[0]: 'different_pit_v1'}, 'source query'),
                ('price', 'fields', ['close'], 'adjusted ref'),
                ('price', 'universe_id', 'different_universe', 'adjusted ref'),
                ('membership', 'policy_by_session', {fixture.calendar[0]: 'different_pit_v1'}, 'source query'),
                ('membership', 'fields', ['different_field'], 'membership ref'),
                ('membership', 'price_basis', 'common_anchor_adjusted_v1', 'membership ref'),
                ('membership', 'adjustment_anchor', fixture.days[0], 'membership ref'))
            for source_name, field, value, reason in cases:
                with self.subTest(source=source_name, field=field):
                    proof = deepcopy(original_proof)
                    source = proof[0]['source_evidence'][source_name]
                    source['query_context']['query'][field] = value
                    # Preserve the within-proof batch-ref equality as well.
                    source['batch_ref'] = digest(source['query_context'])
                    proof[0]['membership_ref' if source_name == 'membership' else 'adjusted_input_ref'] = source['batch_ref']
                    reseal_first_parent_proof(saved, original, proof)
                    with self.assertRaisesRegex(ValueError, reason): load_stock_feature_inputs(saved.path)

    def test_corrupt_partial_parent_cannot_resume_or_publish_complete_index(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp))
            with self.assertRaises(InterruptedPreparation):
                fixture.build(progress=lambda _: (_ for _ in ()).throw(InterruptedPreparation()))
            target, = (Path(temp)/'saved').iterdir()
            checkpoint = _read(target/'checkpoint.json')
            path = Path(checkpoint['feature_parents'][0]['features']['path'])
            path.write_text(path.read_text()+' ')
            before = list(fixture.computed)
            with self.assertRaisesRegex(ValueError, 'file digest'): fixture.build()
            self.assertEqual(fixture.computed, before)
            self.assertFalse((target/'index.json').exists())

    def test_qlib_actual_file_corruption_and_unlisted_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build()
            qlib = Path(saved.to_dict()['qlib_manifest']['path']).parent
            (qlib/'data.bin').write_bytes(b'fail')
            with self.assertRaisesRegex(ValueError, 'Qlib file mismatch'):
                load_stock_feature_inputs(saved.path)
            before = fixture.prepare_calls
            with self.assertRaisesRegex(ValueError, 'Qlib file mismatch'): fixture.build()
            self.assertEqual(fixture.prepare_calls, before)
            (qlib/'data.bin').write_bytes(b'test'); (qlib/'extra.bin').write_bytes(b'unlisted')
            with self.assertRaisesRegex(ValueError, 'Qlib file coverage'):
                load_stock_feature_inputs(saved.path)

    def test_resealed_qlib_query_cannot_override_original_policy_anchor_or_universe(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build(); original = saved.to_dict()
            path = Path(original['qlib_manifest']['path']); original_manifest = _read(path)
            cases = (('policy_by_session', {fixture.calendar[0]: 'different_pit_v1'}, 'Qlib original query'),
                     ('adjustment_anchor', fixture.days[0], 'Qlib original query'),
                     ('universe_id', 'different_universe', 'Qlib original fields/universe'))
            for field, value, reason in cases:
                with self.subTest(field=field):
                    manifest = deepcopy(original_manifest)
                    manifest['queries'][0][field] = value
                    manifest['view_id'] = digest({k: v for k, v in manifest.items() if k != 'view_id'})[7:]
                    write_json(path, manifest)
                    wire = deepcopy(original)
                    wire['qlib_manifest'].update(file_digest=file_digest(path), view_id=manifest['view_id'])
                    wire['qlib_view'] = {k: v for k, v in manifest.items() if k != 'files'}
                    reseal_index(saved.path, wire)
                    with self.assertRaisesRegex(ValueError, reason): load_stock_feature_inputs(saved.path)

    def test_original_cutoff_and_scope_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); fixture.build()
            wrong_clock = deepcopy(fixture.spec)
            wrong_clock['cutoff_by_session'][fixture.days[0]] = fixture.days[0]+'T21:00:00+08:00'
            before = fixture.prepare_calls
            with self.assertRaisesRegex(ValueError, 'clocks'): fixture.build(spec=wrong_clock)
            scope = Path(fixture.spec['scope']['path']); scope.write_text(scope.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'file digest'): fixture.build()
            self.assertEqual(fixture.prepare_calls, before)

    def test_changed_implementation_or_environment_uses_new_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build()
            old = file_state(saved.path)
            fixture.implementation = {'synthetic': digest('changed source')}
            source_changed = fixture.build()
            self.assertFalse(source_changed.reused); self.assertNotEqual(source_changed.path, saved.path)
            fixture.environment = {'synthetic': 'changed environment'}
            env_changed = fixture.build()
            self.assertFalse(env_changed.reused); self.assertNotEqual(env_changed.path, source_changed.path)
            self.assertEqual(fixture.prepare_calls, 3)
            self.assertEqual(fixture.computed, fixture.days*3)
            self.assertEqual(file_state(saved.path), old)

    def test_positive_integer_shard_sizes_and_exact_byte_limits(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build()
            for bad in (True, False, 0, -1, 1.5):
                with self.subTest(shard=bad), self.assertRaises(ValueError):
                    fixture.build(shard_sessions=bad)
            for limits in ({}, {'maximum_parent_bytes': True}, {'maximum_parent_bytes': 0},
                           {'maximum_parent_bytes': -1}, {'maximum_parent_bytes': 2.5},
                           {'maximum_parent_bytes': 1, 'other': 1}):
                with self.subTest(limits=limits), self.assertRaises(ValueError):
                    load_stock_feature_inputs(saved.path, limits=limits)
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                load_stock_feature_inputs(saved.path, limits={'maximum_parent_bytes': 1})
            largest = max(p.stat().st_size for p in Path(temp).rglob('*.json'))
            self.assertEqual(load_stock_feature_inputs(saved.path,
                limits={'maximum_parent_bytes': largest}).identity, saved.identity)

    def test_fresh_public_loader_blocks_runtime_imports_and_all_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = SyntheticInputs(Path(temp)); saved = fixture.build(); before = file_state(Path(temp))
            script = r'''
import builtins, io, os, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.startswith(('axiom_data', 'axiom_engine', 'qlib', 'lightgbm', 'numpy', 'pandas')):
        raise AssertionError('public saved loader imported runtime: '+name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
def read_only(original):
    def guarded(file, mode='r', *args, **kwargs):
        if any(c in mode for c in 'wax+'):
            raise AssertionError('public saved loader attempted a write')
        return original(file, mode, *args, **kwargs)
    return guarded
builtins.open = read_only(builtins.open)
io.open = read_only(io.open)
original_os_open = os.open
def guarded_os_open(path, flags, *args, **kwargs):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        raise AssertionError('public saved loader attempted os.open write')
    return original_os_open(path, flags, *args, **kwargs)
os.open = guarded_os_open
def no_mutation(*args, **kwargs):
    raise AssertionError('public saved loader attempted filesystem mutation')
for name in ('mkdir', 'makedirs', 'rename', 'replace', 'remove', 'unlink', 'rmdir'):
    setattr(os, name, no_mutation)
from axiom_research import load_stock_feature_inputs
saved = load_stock_feature_inputs(sys.argv[1])
assert saved.identity == sys.argv[2]
assert saved.to_dict()['status'] == 'COMPLETE'
assert len(saved.feature_parents) == 3
'''
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            result = subprocess.run([sys.executable, '-c', script, str(saved.path), saved.identity],
                                    capture_output=True, text=True, env=env, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(file_state(Path(temp)), before)


if __name__ == '__main__': unittest.main()
