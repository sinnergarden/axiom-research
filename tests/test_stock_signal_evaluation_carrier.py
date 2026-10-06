"""Small carrier adapter integration; synthetic Data and model backend only."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import gc
import os
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

from axiom_engine.core import evaluate_signal_statistics
from axiom_research import (prepare_stock_ml_batch_inputs, load_stock_ml_batch_inputs,
    build_stock_ml_fold_from_saved_inputs, save_stock_signal_evaluation_inputs,
    evaluate_stock_signal_inputs, load_stock_signal_evaluation, audit_stock_signal_evaluation)
from axiom_research.stock_artifacts import _read, digest, file_digest
from axiom_research.stock_batch import _data
from axiom_research.stock_native_json import make_native_carrier
from axiom_research import stock_signal_evaluation_matrix as matrix
from axiom_research import stock_signal_evaluation_projection as frozen
from test_stock_matrix_prepare import PrepareFeatureFixture, Query
from test_stock_folds import backend
from test_stock_signal_evaluation_matrix_integration import ExactData, raw_override
from test_stock_signal_evaluation_matrix_public import MockOwner
from test_stock_signal_evaluation_matrix import fixture as target_fixture, reseal
from test_stock_signal_evaluation import raw_variant


class CoverageData(ExactData):
    def read(self, **kwargs):
        batch = super().read(**kwargs)
        batch.context['coverage'] = {'complete': True, 'observed': ['甲𝄞', None, 0.125]}
        return batch


def prepare(root):
    fixture = PrepareFeatureFixture(root)
    with patch('axiom_research.feature_catalog.load_feature_catalog', return_value=fixture.catalog):
        saved = fixture.matrix()
    data = CoverageData(fixture.spec)
    module = types.ModuleType('axiom_data'); module.QuerySpec = Query; module.adjust_prices = data.adjust
    with patch.dict(sys.modules, {'axiom_data': module}), \
         patch('axiom_research.stock_ml._implementation', return_value=fixture.implementation), \
         patch('axiom_research.stock_ml._environment', return_value=fixture.environment):
        manifest = prepare_stock_ml_batch_inputs(data, feature_inputs=saved, fold_specs=fixture.folds()[:2],
            destination=root/'prepared', preparation_options={'row_block_sessions':32, 'column_block':32,
                'maximum_resident_bytes':64*1024**2, 'normalization_backend':'core_cs_batch_v1'})
    saved.close()
    return fixture, data, manifest


class SignalCarrierTests(unittest.TestCase):
    def test_public_default_external_override_exact_math_lifecycle_and_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); fixture, data, manifest = prepare(root)
            with load_stock_ml_batch_inputs(manifest) as batch:
                signals, sessions, original_refs = [], [], set()
                for fold in manifest['folds']:
                    with patch('axiom_research.feature_catalog.load_feature_catalog', return_value=fixture.catalog), \
                         patch('axiom_research.stock_ml._implementation', return_value=fixture.implementation), \
                         patch('axiom_research.stock_ml._environment', return_value=fixture.environment), \
                         patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=backend):
                        run = build_stock_ml_fold_from_saved_inputs(fold['input_manifest'], fold_spec=fold['fold_spec'],
                            destination=root/'folds', batch=batch)
                    signals.append({'path':str(run.path/'predictions.json'),
                        'file_digest':file_digest(run.path/'predictions.json'), 'signal_run_ref':run.predictions()['signal_run_ref']})
                    with batch._project_evaluation(fold['input_manifest'], fold['fold_spec']) as view:
                        sessions.extend(view.features['prediction_sessions'])
                        for raw in view.raw_provenance:
                            original_refs.add(raw['label_ref'])
                            self.assertNotIn('coverage', raw['source_evidence']['context'])
                            binding = view.fold_binding['raw_provenance_storage'][raw['label_ref']]
                            self.assertEqual(binding['carrier'], raw['raw_build'])
                            consumed = {batch._evaluation_source_records()[index][0] for index in view.source_record_indices}
                            self.assertTrue(set(_data(batch)['matrix_state'].store.source_paths(raw['raw_build'])) <= consumed)
                scope = {'sessions':sorted(set(sessions)), 'universe':fixture.universe, 'calendar':fixture.calendar,
                    'evaluation_cutoff':manifest['folds'][0]['fold_spec']['evaluation_cutoff']}
                inputs = {'a':signals, 'b':signals[:1]}
                inline = raw_override(root, fixture, data, scope, horizon=6)
                carried = make_native_carrier(root/'external', _read(inline['path']), 'label_ref',
                    maximum_source_bytes=8*1024**2, maximum_parent_bytes=2*1024**2, maximum_workspace_bytes=64*1024**2)
                self.assertEqual(carried['label_ref'], inline['label_ref'])
                self.assertNotIn(carried['path'], dict(batch._evaluation_source_records()))
                leases = []
                admit = batch._admit_raw_label
                def raw(descriptor):
                    lease = admit(descriptor); leases.append(weakref.ref(lease))
                    return lease
                before = batch.metrics
                with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('freeze statistics')), \
                     patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('model executed')):
                    default = save_stock_signal_evaluation_inputs(inputs, batch=batch, scope=scope, destination=root/'default')
                    direct = save_stock_signal_evaluation_inputs(inputs, batch=batch, raw_label_input=inline,
                        scope=scope, destination=root/'direct')
                    with patch.object(type(batch), '_admit_raw_label', side_effect=raw) as admission:
                        external = save_stock_signal_evaluation_inputs(inputs, batch=batch, raw_label_input=carried,
                            scope=scope, destination=root/'external-frozen')
                    self.assertEqual(admission.call_count, 1)
                gc.collect(); self.assertTrue(all(ref() is None for ref in leases))
                state = _data(batch)['matrix_state']
                self.assertEqual((state.store.borrowers, state.store.lease_bytes), (0, 0))
                self.assertEqual(batch.metrics['common_key_index_builds'], before['common_key_index_builds'])
                frozen_default = _read(default.uri)
                self.assertEqual(set(frozen_default['raw_metadata']['sources']), original_refs)
                for header in frozen_default['raw_metadata']['sources'].values():
                    self.assertEqual(header['storage_binding']['carrier'], header['raw_input'])
                    self.assertNotIn('coverage', header['source_context'])
                root_wire = _read(external.uri)
                header = next(iter(root_wire['raw_metadata']['sources'].values()))
                self.assertEqual(header['storage_binding'], {'representation':'stock_native_json_carrier_v1', 'carrier':carried})
                records = {item['path']:item['file_digest'] for item in root_wire['admission_receipt']['source_records']}
                closure = _read(carried['path'])['coverage_slots']
                self.assertTrue(all(slot['bytes']['path'] in records for slot in closure))
                with patch.object(type(batch), '_admit_raw_label', side_effect=AssertionError('daily native admission')), \
                     patch('axiom_research.stock_native_json.verify_carrier_native', side_effect=AssertionError('daily carrier replay')), \
                     patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as core:
                    plain = evaluate_stock_signal_inputs(direct, scope=scope, destination=root/'plain-reports')
                    reports = evaluate_stock_signal_inputs(external, scope=scope, destination=root/'reports')
                self.assertEqual([call.args for call in core.call_args_list[:2]], [call.args for call in core.call_args_list[2:]])
                for name in inputs:
                    a, b = plain[name].to_dict(), reports[name].to_dict()
                    for field in ('label_ref','label_spec','series','summary','coverage'):
                        self.assertEqual(a[field], b[field], field)
                bad = deepcopy(root_wire)
                next(iter(bad['raw_metadata']['sources'].values()))['storage_binding']['carrier']['file_digest'] = digest('wrong bytes')
                altered = replace(external, artifact_id=frozen._root_id(bad))
                bad['input_id'] = altered.artifact_id
                with self.assertRaisesRegex(ValueError, 'storage descriptor'):
                    matrix._verify_matrix_root(bad, altered, scope)
            audit_stock_signal_evaluation(reports['a'].path)
            blob = Path(closure[0]['bytes']['path']); blob.write_bytes(b'null')
            with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('HIT computed')):
                hit = evaluate_stock_signal_inputs(external, scope=scope, destination=root/'reports')
            self.assertTrue(all(report.reused for report in hit.values()))
            self.assertEqual(load_stock_signal_evaluation(reports['a'].path).identity, reports['a'].identity)
            with self.assertRaises(ValueError): audit_stock_signal_evaluation(reports['a'].path)

    def test_unknown_or_conflicting_storage_map_rejected_without_publication(self):
        for kind in ('unknown', 'descriptor', 'representation'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); owner = MockOwner(root/'source')
                view = next(iter(owner.views.values())); raw = view.raw_provenance[0]
                descriptor = deepcopy(raw['raw_build'])
                binding = {'representation':'stock_native_json_carrier_v1', 'carrier':descriptor}
                ref = raw['label_ref']
                if kind == 'unknown': ref = digest('unrelated raw')
                if kind == 'descriptor': descriptor['file_digest'] = digest('wrong bytes')
                if kind == 'representation': binding['representation'] = 'unrelated carrier'
                view.fold_binding['raw_provenance_storage'] = {ref:binding}
                with owner.patch_owner(), owner.batch() as batch, self.assertRaises(ValueError):
                    save_stock_signal_evaluation_inputs(owner.inputs, batch=batch,
                        scope=owner.scope, destination=root/'frozen')
                self.assertFalse((root/'frozen').exists())

    def test_join_retains_storage_and_rejects_same_ref_storage_conflict(self):
        item, scope = target_fixture('a'); raw = next(iter(item[2].values()))
        binding = {'representation':'stock_native_json_carrier_v1', 'carrier':deepcopy(raw['raw_input'])}
        raw['storage_binding'] = binding
        merged = matrix._merge_evaluation_targets([item], calendar=scope['calendar'], universe=scope['universe'], scope=scope)
        self.assertEqual(merged['raw_provenance'][raw['label_ref']]['storage_binding'], binding)
        another = deepcopy(item); another[0]['fold_spec_ref'] = digest('another fold'); reseal(another)
        next(iter(another[2].values()))['storage_binding']['carrier']['file_digest'] = digest('conflict')
        with self.assertRaisesRegex(ValueError, 'conflicting matrix RawLabel provenance'):
            matrix._merge_evaluation_targets([item, another], calendar=scope['calendar'], universe=scope['universe'], scope=scope)

    def test_fresh_carried_override_report_loader_requires_no_native_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); owner = MockOwner(root/'source', two_signals=True)
            inline = raw_variant(root, owner.label,
                lambda raw: raw['source_evidence']['context'].update(coverage={'fixture':True, 'values':[None, 0.125]}))
            carried = make_native_carrier(root/'carrier', _read(inline['path']), 'label_ref',
                maximum_source_bytes=4*1024**2, maximum_parent_bytes=2*1024**2, maximum_workspace_bytes=32*1024**2)
            with owner.patch_owner(), owner.batch() as batch:
                ref = save_stock_signal_evaluation_inputs(owner.inputs, batch=batch, raw_label_input=carried,
                    scope=owner.scope, destination=root/'frozen')
            reports = evaluate_stock_signal_inputs(ref, scope=owner.scope, destination=root/'reports')
            script = '''import builtins,sys
real=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'axiom_engine','axiom_data','numpy','pandas','pyarrow','qlib','lightgbm'}:
        raise AssertionError('runtime import '+name)
    if name in {'axiom_research.stock_native_json','axiom_research.stock_matrix_reader'}:
        raise AssertionError('native admission import '+name)
    return real(name,*args,**kwargs)
builtins.__import__=guard
from axiom_research import load_stock_signal_evaluation
print(load_stock_signal_evaluation(sys.argv[1]).identity)
'''
            fresh = subprocess.run([sys.executable, '-B', '-c', script, str(reports['a'].path)], env=os.environ,
                capture_output=True, text=True, check=True, timeout=15)
            self.assertEqual(fresh.stdout.strip(), reports['a'].identity)
            reads = []; checked = frozen._read_checked
            def read(path, *args, **kwargs):
                reads.append(str(path)); return checked(path, *args, **kwargs)
            with patch.object(frozen, '_read_checked', new=read), \
                 patch('axiom_research.stock_matrix_reader._admit_raw_label', side_effect=AssertionError('daily native admission')), \
                 patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('HIT statistics')):
                hit = evaluate_stock_signal_inputs(ref, scope=owner.scope, destination=root/'reports')
            self.assertTrue(all(report.reused for report in hit.values()))
            carrier = _read(carried['path'])
            self.assertNotIn(carried['path'], reads)
            self.assertTrue(all(slot['bytes']['path'] not in reads for slot in carrier['coverage_slots']))
