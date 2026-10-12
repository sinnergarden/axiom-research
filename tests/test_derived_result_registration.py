"""Tiny frozen-output fixtures; registration and reads never execute mathematics.

The original-parent headers are static loader fixtures, not trained models.
They exercise saved identity/pin linkage and do not claim numerical acceptance.
"""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (ExperimentReader, ExperimentRecordError, SignalPlanSpec,
    SignalInput, SignalNode, begin_stock_result_registration, publish_stock_result,
    load_stock_derived_signal, semantic_identity, to_dict)
from axiom_research.result_registration import _seal
from axiom_research.stock_artifacts import digest, file_digest, write_json
from axiom_research.stock_derived_signal import _artifact, _register_saved
from axiom_research.stock_experiment_config import _transport_defaults
from test_stock_target_config_blocks import label


def plan(weights=(.5, .5)):
    return SignalPlanSpec(name='static-saved-combination', key=('security_id', 'session'),
        inputs=tuple(SignalInput(alias=n, label=label(h), source_stage='raw_prediction',
            score_semantics='static-parent') for n, h in (('a', 3), ('b', 5))),
        nodes=(SignalNode(name='combined', op='weighted_combine', inputs=('a', 'b'),
            input_stages=('raw_prediction', 'raw_prediction'), output_stage='final',
            weights=weights, reference_universe='static-members', missing_policy='propagate', parameters={}),),
        output='combined', join_policy='inner_on_security_session', score_semantics='static-combined',
        available_time_semantics='max_original_dependencies')


def configuration(signal_plan):
    specs={'signal':signal_plan}; refs={'signal':semantic_identity(signal_plan)}
    specification=digest(dict(contract_version='stock_effective_configuration_v1', spec_refs=refs))
    scope=dict(calendar=['2024-01-02', '2024-01-03'], sessions=['2024-01-03'],
        universe=['A', 'B'], evaluation_cutoff='2024-01-04T12:00:00Z')
    dataset=dict(fold_specs=[], scope=scope)
    dataset_ref=digest(dataset)
    return dict(configuration_ref=digest(dict(contract_version='stock_sequential_effective_configuration_v1',
        specification_ref=specification, dataset_ref=dataset_ref)), dataset_ref=dataset_ref,
        dataset=dataset, specification=dict(configuration_ref=specification, spec_refs=refs, specs=specs))


def saved_combination(root, name, signal_plan):
    """Write declared saved values and byte pins directly; call no build owner."""
    from axiom_research.stock_fold_inputs import seal
    target=root/name; target.mkdir(parents=True)
    parents={}; pins={}; feature_refs=[]; model_refs=[]
    for alias in ('a', 'b'):
        base=target/alias; base.mkdir()
        feature=digest([name, alias, 'feature']); feature_refs.append(feature)
        model=seal(dict(contract_version='static_model_header_v1', fixture=name+alias,
            simulated_available_at='2024-01-02T12:00:00Z'), 'model_ref'); model_refs.append(model['model_ref'])
        prediction=seal(dict(contract_version='static_prediction_header_v1', fixture=name+alias,
            feature_ref=feature), 'signal_run_ref')
        spec=dict(contract_version='static_fold_spec_v1', fixture=name+alias)
        fold=seal(dict(contract_version='static_fold_header_v1', fold_ref=digest([name, alias, 'fold']),
            model_ref=model['model_ref'], feature_ref=feature, signal_run_ref=prediction['signal_run_ref'],
            definition=dict(fold_spec=spec)), 'content_digest')
        for filename, wire in (('model.json', model), ('predictions.json', prediction), ('fold.json', fold)):
            write_json(base/filename, wire); pins[str(base/filename)]=file_digest(base/filename)
        spec_ref=_artifact(base/'fold.json', spec, 'StockFoldSpec', digest(spec))
        spec_ref['manifest_uri']+='#definition/fold_spec'
        parents[alias]=dict(kind='raw', fold_ref=fold['fold_ref'], fold_spec_ref=digest(spec),
            model_ref=model['model_ref'], feature_ref=feature, signal_run_ref=prediction['signal_run_ref'],
            fold_spec_artifact=spec_ref,
            model_metadata_artifact=_artifact(base/'model.json', model, 'StockModelRelease', model['model_ref']),
            prediction_artifact=_artifact(base/'predictions.json', prediction, 'StockPredictionRun', prediction['signal_run_ref']))
    context=dict(calendar_ref=digest(configuration(signal_plan)['dataset']['scope']['calendar']),
        cutoff_by_session={'2024-01-03':'2024-01-03T13:00:00Z'}, clock_basis='declared_simulation')
    rows=[dict(security_id=s, session='2024-01-03', knowledge_cutoff='2024-01-03T13:00:00Z',
        available_at='2024-01-03T13:00:00Z', feature_knowledge_cutoff='2024-01-03T12:30:00Z',
        score=v, valid=True, invalid_reason=None, member=True, source_refs=model_refs+feature_refs)
        for s,v in (('A', .25), ('B', -.25))]
    transport=_transport_defaults(to_dict(signal_plan))
    def neutral(v):
        if type(v) is list: return [neutral(i) for i in v]
        if type(v) is dict: return {k:neutral(i) for k,i in v.items() if k!='contract_type'}
        return v
    original={a:p['signal_run_ref'] for a,p in parents.items()}
    signal=seal(dict(contract_version='derived_signal_run_v1', score_ref=digest(rows),
        signal_plan=neutral(transport), signal_plan_ref=semantic_identity(signal_plan),
        parent_signal_refs=original, implementation_ref=digest('static-saved-implementation'),
        signal_stage='final', score_semantics=signal_plan.score_semantics, score_unit='dimensionless',
        universe=['A','B'], rows=rows, context=context, limitations=['Static saved-output fixture; no trained model.']),
        'signal_run_ref')
    write_json(target/'signal.json', signal)
    manifest=seal(dict(contract_version='stock_derived_signal_manifest_v1',
        definition=dict(contract_version='stock_derived_signal_request_v1', signal_plan=transport,
            configuration_spec_ref=semantic_identity(signal_plan), context=context, parent_signal_refs=original),
        signal_file_digest=file_digest(target/'signal.json'),
        signal_artifact=_artifact(target/'signal.json', signal, 'DerivedSignal', signal['signal_run_ref']),
        parent_inputs=parents, parent_files=pins), 'content_digest')
    write_json(target/'manifest.json', manifest)
    return target


class DerivedResultRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name).resolve()
        self.roots={'research':self.root}; self.index=self.root/'index.json'
        self.request=self.root/'attempt.json'; self.plan=plan()
        self.cfg=configuration(self.plan)
        self.context=dict(request_path=self.request, index_path=self.index, roots=self.roots)
        self.first=saved_combination(self.root, 'one', self.plan)
        self.second=saved_combination(self.root, 'two', self.plan)

    def tearDown(self): self.temp.cleanup()

    def begin(self, **options):
        return begin_stock_result_registration(self.cfg, **self.context, required_stages=(), **options)

    def forbidden(self, stack):
        for name in ('execute_signal_plan', 'evaluate_signal_statistics'):
            stack.enter_context(patch('axiom_engine.core.'+name, side_effect=AssertionError('numerical replay')))
        for name in ('run_backtest', 'evaluate_backtest'):
            stack.enter_context(patch('axiom_engine.runtime.'+name, side_effect=AssertionError('account replay')))
        stack.enter_context(patch('axiom_research.stock_training.fit_predict_stock_model',
            side_effect=AssertionError('training replay')))

    def test_public_saved_registration_index_retry_and_missing_leaf(self):
        self.begin()
        before={p:(file_digest(p),p.stat().st_mtime_ns) for folder in (self.first,self.second)
                for p in folder.rglob('*.json')}
        with ExitStack() as stack:
            self.forbidden(stack)
            head=publish_stock_result(**self.context, derived_signals=[self.first,self.second/'signal.json'])
            reader=ExperimentReader(self.index); index=reader.index()
            manifest=reader.result(head['result_id'],roots=self.roots)
            self.assertEqual(head['status'],'COMPLETE'); self.assertEqual(manifest['folds'],[])
            self.assertEqual(len(index['result_heads']),1); self.assertEqual(len(manifest['signals']),2)
            for declaration in manifest['signals']:
                ref=next(a for a in manifest['artifacts'] if a['artifact_key']==declaration['prediction_artifact_key'])
                saved=load_stock_derived_signal(self.root/ref['relative_path'],research_root=self.root)
                self.assertEqual((ref['artifact_type'],ref['artifact_id'],ref['content_digest']),
                    ('StockDerivedSignal',saved.identity,saved.content_digest))
                self.assertEqual(len(saved.to_dict()['parent_signal_refs']),2)
            old=(self.index.read_bytes(),self.index.stat().st_mtime_ns)
            publish_stock_result(**self.context,derived_signals=[self.first,self.second])
            _register_saved(load_stock_derived_signal(self.first),self.context)
            self.assertEqual(old,(self.index.read_bytes(),self.index.stat().st_mtime_ns))
        self.assertEqual(before,{p:(file_digest(p),p.stat().st_mtime_ns) for p in before})
        (self.first/'signal.json').unlink()
        self.assertEqual(ExperimentReader(self.index).result(head['result_id'],roots=self.roots),manifest)
        with self.assertRaises(FileNotFoundError): load_stock_derived_signal(self.first,research_root=self.root)
        self.assertEqual(load_stock_derived_signal(self.second,research_root=self.root).to_dict()['rows'][0]['score'],.25)

    def test_changed_plan_and_external_parent_refuse_before_publish(self):
        self.begin(); publish_stock_result(**self.context,derived_signals=[self.first])
        old=(self.index.read_bytes(),self.index.stat().st_mtime_ns)
        other=saved_combination(self.root,'different-plan',plan((.25,.75)))
        with self.assertRaisesRegex(ExperimentRecordError,'frozen SignalPlan'):
            publish_stock_result(**self.context,derived_signals=[other])
        self.assertEqual(old,(self.index.read_bytes(),self.index.stat().st_mtime_ns))
        import json
        with tempfile.TemporaryDirectory() as external:
            outside=Path(external)/'private.json'; write_json(outside,{'static':'external'})
            manifest=json.loads((self.second/'manifest.json').read_text()); manifest.pop('content_digest')
            manifest['parent_files'][str(outside)]=file_digest(outside)
            write_json(self.second/'manifest.json',_seal(manifest))
            original_open=Path.open
            def guarded(path,*args,**kwargs):
                if path.resolve()==outside.resolve(): raise AssertionError('external bytes were read')
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',guarded),self.assertRaisesRegex(ValueError,'escapes research_root'):
                publish_stock_result(**self.context,derived_signals=[self.second])
        self.assertEqual(old,(self.index.read_bytes(),self.index.stat().st_mtime_ns))

    def test_derived_account_original_binding_and_blocked_state(self):
        """Trusted owner projections isolate registry edges; no synthetic account run."""
        from types import SimpleNamespace
        from axiom_engine.runtime.stock_stream_contracts import logical_ref
        self.begin(); publish_stock_result(**self.context,derived_signals=[self.first])
        saved=load_stock_derived_signal(self.first); binding=saved.engine_input_binding()
        inputs=dict(contract_version='stock_prediction_input_refs_v2',frames=[binding])
        inputs['prediction_ref']=logical_ref(inputs,'prediction_ref')
        engine=self.root/'engine';engine.mkdir();self.roots['engine']=engine
        path=engine/'saved-account.json';write_json(path,{'static':'trusted owner projection'})
        ref=dict(run_id=digest('static-account'),content_digest=digest('static-account-content'),committed_sequence=7)
        run=dict(contract_version='stock_backtest_run_v9',**ref,status='BLOCKED',
            stopped={'reason':'FIXED_SOURCE_MISSING'},signal_ref=inputs['prediction_ref'],
            request_manifest=dict(portfolio_policy={'static':'policy'},prediction_input=inputs))
        projection=SimpleNamespace(input_run_ref=ref,verify=lambda:SimpleNamespace(wire=run))
        with patch('axiom_engine.runtime.load_stock_backtest_projection',return_value=projection):
            head=publish_stock_result(**self.context,engine_handoff={'accounts':[
                dict(run_manifest=str(path),run_ref=ref,evaluations={})]})
            manifest=ExperimentReader(self.index).result(head['result_id'],roots=self.roots)
            self.assertEqual((head['status'],manifest['accounts'][0]['status']),('BLOCKED','BLOCKED'))
            self.assertEqual(manifest['accounts'][0]['reason'],'FIXED_SOURCE_MISSING')
            old=(self.index.read_bytes(),self.index.stat().st_mtime_ns)
            run['request_manifest']['prediction_input']['frames'][0]=deepcopy(binding)
            run['request_manifest']['prediction_input']['frames'][0]['score_ref']=digest('wrong-score')
            with self.assertRaisesRegex(ExperimentRecordError,'original saved prediction'):
                publish_stock_result(**self.context,engine_handoff={'accounts':[
                    dict(run_manifest=str(path),run_ref=ref,evaluations={})]})
            self.assertEqual(old,(self.index.read_bytes(),self.index.stat().st_mtime_ns))
            # A valid saved artifact outside signals is not an admitted account input.
            other=saved_combination(self.root,'unregistered',plan((.25,.75)))
            obj=load_stock_derived_signal(other)
            run['request_manifest']['prediction_input']['frames']=[obj.engine_input_binding()]
            bad=deepcopy(manifest);bad.pop('content_digest');bad['revision']+=1
            bad['artifacts'].append(dict(artifact_key='unregistered-derived',owner='Research',
                artifact_type='StockDerivedSignal',artifact_id=obj.identity,
                artifact_contract_version='derived_signal_run_v1',content_digest=obj.content_digest,
                root_id='research',relative_path=(other/'signal.json').relative_to(self.root).as_posix()))
            from axiom_research import ExperimentStore
            with self.assertRaisesRegex(ExperimentRecordError,'original saved prediction'):
                ExperimentStore(self.index).publish_result(_seal(bad),roots=self.roots)
            self.assertEqual(old,(self.index.read_bytes(),self.index.stat().st_mtime_ns))

    def test_sequential_cannot_omit_derived_binding_declaration(self):
        from axiom_research import StockSavedProjection,load_stock_sequential_result
        saved=load_stock_derived_signal(self.first); frames=[saved.engine_input_binding()]
        base=self.root/'receipt';base.mkdir();batch={'folds':[{'static':'owner'}]}
        result=dict(batch_manifest=batch,prediction_bindings=frames,folds=[{'static':'fold'}],derived_signals=[])
        write_json(base/'experiment-result.json',result);write_json(base/'batch.json',batch)
        write_json(base/'prediction-bindings.json',frames)
        with patch('axiom_research.stock_result_projection.load_stock_ml_batch_manifest',
                return_value=StockSavedProjection(batch,digest('batch'),digest('batch-content'))), \
             patch('axiom_research.stock_result_projection.load_stock_fold_projection',
                side_effect=AssertionError('omission reached fold processing')):
            with self.assertRaisesRegex(ExperimentRecordError,'Derived declarations differ'):
                load_stock_sequential_result(base/'experiment-result.json',research_root=self.root)
        parent=frames[0]['parent_inputs']['a']
        fold=dict(fold_ref=parent['fold_ref'],signal_run_ref=parent['signal_run_ref'],
            definition={'fold_spec_ref':parent['fold_spec_ref']})
        result.update(folds=[dict(path=str(self.first/'a'),fold_ref=parent['fold_ref'],
            signal_run_ref=parent['signal_run_ref'],fold_spec_ref=parent['fold_spec_ref'])],
            derived_signals=[dict(path=str(self.first),signal_run_ref=saved.identity)],
            configuration_ref=self.cfg['configuration_ref'],spec_refs=self.cfg['specification']['spec_refs'],
            signal_evaluation_input={'static':'saved joint input'})
        write_json(base/'experiment-result.json',result)
        with patch('axiom_research.stock_result_projection.load_stock_ml_batch_manifest',
                return_value=StockSavedProjection(batch,digest('batch'),digest('batch-content'))), \
             patch('axiom_research.stock_result_projection.load_stock_fold_projection',
                return_value=StockSavedProjection(fold,parent['fold_ref'],digest('fold-content'))):
            projection=load_stock_sequential_result(base/'experiment-result.json',research_root=self.root)
            self.assertEqual(projection.to_dict()['derived_signals'],result['derived_signals'])
