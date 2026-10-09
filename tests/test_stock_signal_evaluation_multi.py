"""Two tiny public owners, synthetic Data columns, original Core statistics.

Fixture setup saves two fake-backend folds once. Every evaluation reuse below
forbids Feature, normalization, Data selection, fit/predict and accounts.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import ModuleType
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (prepare_stock_ml_batch_inputs, load_stock_ml_batch_inputs,
    build_stock_evaluation_label_inputs, build_stock_ml_fold_from_saved_inputs,
    save_stock_signal_evaluation_inputs, evaluate_stock_signal_input_periods,
    load_stock_signal_evaluation, audit_stock_signal_evaluation)
from axiom_research.stock_artifacts import _read, file_digest, digest
from axiom_research.stock_signal_evaluation_projection import _load_inputs, _shard
from axiom_research.stock_batch import _data
from axiom_research.stock_compact_store import _size, OwnedStore
from test_stock_column_asof import ColumnSource
import test_stock_compact_v3 as fixture_sources
from test_stock_matrix_prepare import Query
from test_stock_sequential_windows import backend
from test_stock_target_config_blocks import label
from test_stock_signal_evaluation_owner_vertical import readonly_execution, CrossYearFeature


class MultiOwnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); cls.root = Path(cls.temporary.name).resolve()
        def missing_last_column(day, rows):
            for row in rows:
                if row['security_id'] == 'S003':
                    row['values'][-1] = None; row['validity'][-1] = False
                    row['availability'][-1] = None; row['reasons'][-1] = ['SYNTHETIC_LAST_COLUMN_MISSING']
        cls.feature, table = fixture_sources.CompactV3Tests().fixture(cls.root,
            feature_class=CrossYearFeature, transform=missing_last_column)
        f = cls.feature; f.environment = {'synthetic': 'fixed fake backend', 'packages': {'lightgbm': '4.6.0'}}
        entries={item['id']:item for item in f.catalog.select(f.selection)}
        f.catalog=type('SyntheticSelectedCatalog',(),{'identity':f.catalog.identity,
            'select':lambda _,selection:[deepcopy(entries[item['id']]) for item in selection]})()
        module = ModuleType('axiom_data'); module.QuerySpec = Query
        cls.manifests, cls.signals, cls.bindings, cls.paths = {}, {}, {}, {}
        with patch.dict(sys.modules, {'axiom_data': module}):
            for horizon in (3, 5):
                name = 'h'+str(horizon)
                manifest = prepare_stock_ml_batch_inputs(object(), feature_inputs=table,
                    fold_specs=f.folds()[:1], label_spec=label(horizon), column_source=ColumnSource(f.spec),
                    model_feature_selection=f.selection[:3] if horizon == 3 else None,
                    destination=cls.root/('prepared-'+name), preparation_options={'row_block_sessions':10,
                        'column_block':32, 'maximum_resident_bytes':64*1024**2, 'normalization_backend':'core_cs_batch_v1'})
                with load_stock_ml_batch_inputs(manifest, residency='sequential') as batch, \
                    patch('axiom_research.feature_catalog.load_feature_catalog', return_value=f.catalog), \
                    patch('axiom_research.stock_ml._environment', return_value=f.environment), \
                    patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=backend):
                    fold = manifest['folds'][0]
                    run = build_stock_ml_fold_from_saved_inputs(fold['input_manifest'], fold_spec=fold['fold_spec'],
                        destination=cls.root/('folds-'+name), batch=batch)
                cls.paths[name] = run.path
                path = run.path/'predictions.json'; wire = _read(path)
                descriptor = {'path':str(path), 'file_digest':file_digest(path), 'signal_run_ref':wire['signal_run_ref']}
                cls.manifests[manifest['batch_ref']] = manifest
                cls.signals[name] = descriptor; cls.bindings[wire['signal_run_ref']] = manifest['batch_ref']
            cls.scope = {'calendar':f.calendar, 'universe':f.universe,
                'sessions':sorted(f.folds()[0]['inference_cutoff_by_session']),
                'evaluation_cutoff':f.folds()[0]['evaluation_cutoff']}
            cls.labels = {}
            for horizon in (3, 5):
                cls.labels[horizon] = build_stock_evaluation_label_inputs(object(), snapshot=f.spec['snapshot'],
                    pit_policy=f.spec['pit_policy'], label_spec=label(horizon), scope=cls.scope,
                    column_source=ColumnSource(f.spec), destination=cls.root/'labels', limits=None)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def owners(self, stack):
        return {ref:stack.enter_context(load_stock_ml_batch_inputs(m, residency='sequential'))
            for ref,m in self.manifests.items()}

    def save(self, owners, *, horizon=5, signals=None, bindings=None, destination=None, limits=None):
        return save_stock_signal_evaluation_inputs(signals or self.signals, raw_label_input=self.labels[horizon],
            scope=self.scope, destination=destination or self.root/'multi', owner_batches=owners,
            prediction_owner_refs=self.bindings if bindings is None else bindings, limits=limits)

    def test_two_original_training_labels_share_independent_label_and_cold_evaluation(self):
        with ExitStack() as stack:
            owners = self.owners(stack)
            with readonly_execution(allow_statistics=True):
                saved = self.save(owners)
                same = self.save(owners)
                changed = self.save(owners, horizon=3)
            self.assertEqual(saved.artifact_id, same.artifact_id)
            self.assertNotEqual(saved.artifact_id, changed.artifact_id)
            for owner in owners.values():
                state = _data(owner)['matrix_state']
                self.assertEqual((state.active, state.store.borrowers, state.store.lease_bytes), (0,0,0))
        with readonly_execution(allow_statistics=True):
            _, root, selected, admission = _load_inputs(saved, self.scope, include_admission=True)
            self.assertEqual(root['raw_metadata']['label_ref'], self.labels[5]['label_ref'])
            self.assertEqual(len(root['admission_receipt']['owner_manifests']), 2)
            self.assertEqual(set(root['admission_receipt']['prediction_owner_refs']), set(self.bindings))
            reports = evaluate_stock_signal_input_periods(saved, scope=self.scope, destination=self.root/'reports')
            again = evaluate_stock_signal_input_periods(saved, scope=self.scope, destination=self.root/'reports')
            for name, report in reports['all'].items():
                self.assertEqual(report.to_dict()['contract_version'], 'stock_signal_evidence_v8')
                self.assertEqual(report.to_dict()['status'], 'COMPLETE')
                self.assertEqual(report.to_dict()['label_ref'], self.labels[5]['label_ref'])
                self.assertEqual(report.to_dict()['input_signal_refs'], [self.signals[name]['signal_run_ref']])
                self.assertTrue(again['all'][name].reused)
                with patch('axiom_engine.core.evaluate_signal_statistics', side_effect=AssertionError('cold loader recomputed')):
                    self.assertEqual(load_stock_signal_evaluation(report.path).to_dict(), report.to_dict())
            self.assertGreater(reports['all']['h3'].to_dict()['coverage']['native']['valid_pair_count'],
                reports['all']['h5'].to_dict()['coverage']['native']['valid_pair_count'])
            self.assertEqual(reports['all']['h3'].to_dict()['sample_mask_ref'],
                reports['all']['h5'].to_dict()['sample_mask_ref'])
            audit_stock_signal_evaluation(next(iter(reports['all'].values())).path)
            # Shards use keyed lookup, independent of dictionary insertion order.
            day = self.scope['sessions'][0]; original = _shard(admission_for_shard(admission, root), self.scope, day)
            reordered = admission_for_shard(admission, root)
            for item in reordered['projected'].values():
                for field in item: item[field] = dict(reversed(list(item[field].items())))
            self.assertEqual(_shard(reordered, self.scope, day), original)

    def test_explicit_mapping_missing_extra_wrong_and_mixed_modes_reject(self):
        with ExitStack() as stack:
            owners = self.owners(stack)
            examples = [{}, {**self.bindings, digest('extra'):next(iter(owners))},
                {r:next(b for b in owners if b != owner) for r,owner in self.bindings.items()}]
            for bindings in examples:
                with self.subTest(bindings=bindings), self.assertRaises(ValueError): self.save(owners, bindings=bindings)
            with self.assertRaises(ValueError):
                save_stock_signal_evaluation_inputs(self.signals, raw_label_input=self.labels[5], scope=self.scope,
                    destination=self.root/'mixed', batch=next(iter(owners.values())), owner_batches=owners,
                    prediction_owner_refs=self.bindings)
            for owner in owners.values(): self.assertEqual(_data(owner)['matrix_state'].active, 0)

    def test_fixed_core_combination_preserves_both_original_parents_on_v6_base(self):
        from axiom_research import (SignalPlanSpec,SignalInput,SignalNode,build_stock_derived_signal,
            save_stock_derived_signal_evaluation_inputs)
        from axiom_research.stock_label_contracts import NORMALIZATION_SPEC
        plan = SignalPlanSpec(name='two-original-owners-fixed-weights',key=('security_id','session'),
            inputs=tuple(SignalInput(alias=name,label=label(h),source_stage='raw_prediction',
                score_semantics=_read(self.signals[name]['path'])['score_semantics']) for name,h in (('h3',3),('h5',5))),
            nodes=tuple(SignalNode(name='z'+name,op='daily_zscore',inputs=(name,),input_stages=('raw_prediction',),
                output_stage='daily_zscore',weights=(),reference_universe='saved-fixture-members',
                missing_policy=NORMALIZATION_SPEC['params']['missing'],parameters=NORMALIZATION_SPEC['params'])
                for name in ('h3','h5'))+(SignalNode(name='combined',op='weighted_combine',inputs=('zh3','zh5'),
                input_stages=('daily_zscore','daily_zscore'),output_stage='final',weights=(.25,.75),
                reference_universe='saved-fixture-members',missing_policy='propagate',parameters={}),),
            output='combined',join_policy='inner_on_security_session',score_semantics='fixed-weights-synthetic',
            available_time_semantics='max_original_dependencies')
        prediction = _read(self.signals['h3']['path'])
        context={'calendar_ref':prediction['label_spec']['calendar_ref'],
            'reference_universe':'saved-fixture-members','reference_universe_ref':digest('fixed synthetic membership'),
            'reference_members':{day:[{'security_id':r['security_id'],'member':r['member'],
                'available_at':r['feature_knowledge_cutoff'],'source_refs':[prediction['feature_ref']]}
                for r in prediction['rows'] if r['session']==day] for day in self.scope['sessions']},
            'cutoff_by_session':{d:next(r['knowledge_cutoff'] for r in prediction['rows'] if r['session']==d)
                for d in self.scope['sessions']},'clock_basis':'declared_simulation'}
        # Core creates this one new combination. The saved consumer subsequently
        # forbids normalization or regeneration and copies the original values.
        with patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('combined fit')):
            derived=build_stock_derived_signal(plan,prediction_inputs=self.paths,context=context,destination=self.root/'derived')
        self.assertEqual(derived.to_dict()['parent_signal_refs'],{n:d['signal_run_ref'] for n,d in self.signals.items()})
        with ExitStack() as stack:
            owners=self.owners(stack)
            with readonly_execution(allow_statistics=True):base=self.save(owners)
        with readonly_execution(allow_statistics=True), \
            patch('axiom_engine.core.execute_signal_plan',side_effect=AssertionError('combined signal replay')):
            joint=save_stock_derived_signal_evaluation_inputs(base,derived_inputs={'combined':[str(derived.path)]},
                scope=self.scope,destination=self.root/'joint',maximum_resident_bytes=64*1024**2)
            reports=evaluate_stock_signal_input_periods(joint,scope=self.scope,destination=self.root/'joint-reports')
            self.assertEqual(set(reports['all']),{'h3','h5','combined'})
            for report in reports['all'].values():self.assertEqual(report.to_dict()['label_ref'],self.labels[5]['label_ref'])
            original=derived.to_dict(); combined=reports['all']['combined'].to_dict()
            self.assertEqual(combined['input_signal_refs'],[original['signal_run_ref']])
            self.assertEqual(combined['signal_lineage'][0]['parent_signal_refs'],original['parent_signal_refs'])

    def test_joint_budget_counts_actual_live_owner_stores_and_restores_limits(self):
        from axiom_research.stock_signal_evaluation_multi import _live
        with ExitStack() as stack:
            owners = self.owners(stack); states = {r:_data(o)['matrix_state'] for r,o in owners.items()}
            baselines = {r:(s._fixed_shared_bytes,s._fixed_shared_source_bytes,deepcopy(s.store.limits)) for r,s in states.items()}
            total,_ = _live(states)
            for state in states.values(): self.assertLess(state.store.shared_bytes+state.store.resident_bytes,total)
            with patch('axiom_research.stock_matrix_folds.admit_stock_signal_evaluation_fold',
                side_effect=AssertionError('joint budget admitted a fold')):
                with self.assertRaisesRegex(ValueError,'(?:byte|accounting workspace) budget'):
                    self.save(owners, limits={'maximum_matrix_bytes':max(1,total-1)})
            for ref,state in states.items():
                self.assertEqual((state._fixed_shared_bytes,state._fixed_shared_source_bytes,state.store.limits),baselines[ref])

    def test_scope_calendar_and_incomplete_alias_reject(self):
        with ExitStack() as stack:
            owners = self.owners(stack)
            scope = deepcopy(self.scope); scope['sessions'] = scope['sessions'][:-1]
            with self.assertRaisesRegex(ValueError,'complete declared grid'):
                save_stock_signal_evaluation_inputs(self.signals, raw_label_input=self.labels[5], scope=scope,
                    destination=self.root/'wrong-scope',owner_batches=owners,prediction_owner_refs=self.bindings)
            scope = deepcopy(self.scope); scope['calendar'] = scope['calendar'][:-1]
            with self.assertRaises(ValueError):
                save_stock_signal_evaluation_inputs(self.signals, raw_label_input=self.labels[5], scope=scope,
                    destination=self.root/'wrong-calendar',owner_batches=owners,prediction_owner_refs=self.bindings)

    def test_audit_first_root_respects_each_budget_before_decode(self):
        from axiom_research.stock_signal_evaluation_multi import _audit_multi_input
        with ExitStack() as stack:
            saved = self.save(self.owners(stack))
        for field in ('maximum_matrix_bytes','maximum_source_bytes','maximum_parent_bytes'):
            tiny = replace(saved, metadata={'limits':{**saved.metadata['limits'],field:1}})
            with self.subTest(field=field), patch('axiom_research.stock_compact_store.json.loads',
                side_effect=AssertionError('root decoded before budget rejection')) as decode:
                with self.assertRaisesRegex(ValueError,'byte budget'):
                    _audit_multi_input(tiny)
                decode.assert_not_called()

    def test_audit_next_owner_uses_remaining_live_source_allowance(self):
        from axiom_research.stock_signal_evaluation_multi import _audit_multi_input, _live
        from axiom_research.stock_signal_evaluation_projection import _root_id
        from axiom_research.stock_artifacts import write_json
        import json
        with ExitStack() as stack:
            saved = self.save(self.owners(stack))
        with ExitStack() as stack:
            fresh = self.owners(stack)
            sources = [_live({r:_data(o)['matrix_state']})[1] for r,o in fresh.items()]
        root = _read(saved.uri); budget = {**saved.metadata['limits'],
            'maximum_source_bytes':Path(saved.uri).stat().st_size+max(sources)+1024}
        root['admission_receipt']['limits'] = budget
        root['admission_receipt']['receipt_ref'] = digest({k:v for k,v in
            root['admission_receipt'].items() if k!='receipt_ref'})
        root['input_id'] = _root_id(root)
        path = self.root/'joint-audit-limited.json'; write_json(path,root)
        limited = replace(saved, uri=str(path),artifact_id=root['input_id'],content_digest=file_digest(path),
            metadata={'limits':budget})
        actual_loader = load_stock_ml_batch_inputs; actual_decode = json.loads; calls = []
        def load(manifest, **kwargs):
            calls.append(kwargs['limits'])
            return actual_loader(manifest,**kwargs)
        def decode(*args,**kwargs):
            if len(calls)>=2: raise AssertionError('second owner decoded after joint source allowance exhausted')
            return actual_decode(*args,**kwargs)
        with patch('axiom_research.stock_batch.load_stock_ml_batch_inputs',side_effect=load), \
            patch('axiom_research.stock_compact_store.json.loads',side_effect=decode):
            with self.assertRaisesRegex(ValueError,'source byte budget'):
                _audit_multi_input(limited)
        self.assertEqual(len(calls),2)
        self.assertLess(calls[0]['maximum_matrix_bytes'],budget['maximum_matrix_bytes'])
        self.assertLess(calls[1]['maximum_matrix_bytes'],calls[0]['maximum_matrix_bytes'])
        self.assertLess(calls[1]['maximum_source_bytes'],calls[0]['maximum_source_bytes'])

    def test_piece_charges_are_incremental_and_owner_operation_is_shared(self):
        from axiom_research import stock_signal_evaluation_multi as multi
        from axiom_research import stock_signal_evaluation_compact as compact
        # Repeated aliases stress accounting only, not model diversity. The
        # unchanged two model identities must still reject at final selection.
        signals = {name+'-'+str(i):descriptor for i in range(8) for name,descriptor in self.signals.items()}
        operations = {}; indexes = {}; fixed_graph_visits = 0; piece_count = 0
        original_size = multi._size; original_operation = OwnedStore.operation
        original_admit = compact._admit_compact
        with ExitStack() as stack:
            owners = self.owners(stack); ids = {id(_data(o)['matrix_state'].store) for o in owners.values()}
            @contextmanager
            def operation(store):
                if id(store) in ids and store._operation_depth==0:
                    operations[id(store)] = operations.get(id(store),0)+1
                with original_operation(store): yield store
            def measure(value,**kwargs):
                nonlocal fixed_graph_visits
                if type(value) is list and any(type(v) is dict and
                    {'projected','label_leaf_bindings','scope'}<=set(v) for v in value):
                    fixed_graph_visits+=1
                return original_size(value,**kwargs)
            def admit(*args,**kwargs):
                nonlocal piece_count
                piece_count+=1
                batch=args[3]; key=batch.identity
                indexes.setdefault(key,set()).add(id(kwargs['_fold_index']))
                self.assertTrue(kwargs['_defer_source_checks'])
                return original_admit(*args,**kwargs)
            with patch.object(OwnedStore,'operation',new=operation),patch.object(multi,'_size',side_effect=measure), \
                patch.object(compact,'_admit_compact',side_effect=admit), readonly_execution(allow_statistics=True):
                from axiom_research.stock_signal_evaluation_inputs import _scope
                with self.assertRaisesRegex(ValueError,'duplicate comparison Signal identity'):
                    multi._admit_multi(signals,self.labels[5],_scope(self.scope),owners,self.bindings,
                        multi.owner_limits(None))
            self.assertEqual(piece_count,len(signals))
            self.assertEqual(fixed_graph_visits,1)
            self.assertEqual(operations,{i:1 for i in ids})
            self.assertTrue(all(len(values)==1 for values in indexes.values()))

    def test_source_mutation_after_deferred_piece_cannot_publish(self):
        from axiom_research import stock_signal_evaluation_compact as compact
        path = self.paths['h3']/'booster.txt'; original = path.read_bytes()
        original_admit = compact._admit_compact; changed = False
        def admit(*args,**kwargs):
            nonlocal changed
            result = original_admit(*args,**kwargs)
            if not changed: path.write_bytes(original+b'changed'); changed=True
            return result
        try:
            with ExitStack() as stack, patch.object(compact,'_admit_compact',side_effect=admit):
                owners = self.owners(stack)
                with self.assertRaisesRegex(ValueError,'changed'):
                    self.save(owners,destination=self.root/'changed-after-piece')
            self.assertFalse((self.root/'changed-after-piece').exists())
        finally: path.write_bytes(original)

    def test_frozen_shard_and_original_label_tamper_reject(self):
        with ExitStack() as stack:
            saved = self.save(self.owners(stack), destination=self.root/'tamper')
        root = _read(saved.uri); day = self.scope['sessions'][0]
        path = Path(saved.uri).parent/root['shards'][day]['file']; original = path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with self.assertRaises(ValueError): _load_inputs(saved,self.scope)
        finally:path.write_bytes(original)
        label_manifest = _read(self.labels[5]['path']); target = _read(label_manifest['raw_parts'][0]['path'])
        path = Path(target['buffers']['values']['path']); original = path.read_bytes()
        try:
            path.write_bytes(b'X'*len(original))
            with ExitStack() as stack, self.assertRaises(ValueError): self.save(self.owners(stack))
        finally:path.write_bytes(original)


def admission_for_shard(admission, root):
    # The cold projection already carries its original Feature clocks inline.
    out = deepcopy(admission)
    for item in out['projected'].values():
        item['prediction_features'] = {key:{'knowledge_cutoff':row['feature_knowledge_cutoff'],
            'feature_available_at':row['feature_available_at'],'source_refs':row['feature_source_refs']}
            for key,row in item['rows'].items()}
    out['label_leaf_bindings'] = {key:__import__('axiom_research.stock_signal_evaluation_compact',
        fromlist=['_binding'])._binding(root['raw_metadata']['label_spec'],root['raw_metadata']['snapshot'],row)
        for key,row in out['labels'].items()}
    return out
