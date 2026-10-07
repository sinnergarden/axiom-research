"""Compact saved matrix folds; the original native model backend is reused."""
from copy import deepcopy
from pathlib import Path
import errno
import tempfile
import time

from .stock_artifacts import digest, file_digest, write_json, _read, _verify_ref
from .stock_fold_inputs import require, seal, _finite, _instant, feature_available
from .stock_label_contracts import NORMALIZATION_SPEC
from .stock_training import LGBM_PARAMETERS, TREES, TARGET_SEMANTICS

OUTPUTS={'feature-slice.json':'feature_ref','label-slice.json':'label_ref',
    'dataset.json':'dataset_ref','model.json':'model_ref','predictions.json':'signal_run_ref',
    'signal-evidence.json':'evidence_ref'}


def _projection(inputs,spec,batch):
    if batch is not None:
        from .stock_batch import _data
        _data(batch); return batch._matrix_project(inputs,spec)
    if inputs.get('contract_version')=='stock_ml_saved_inputs_v3':
        from .stock_compact_batch import load_compact_projection
        return load_compact_projection(inputs,spec)
    from .stock_matrix_reader import load_matrix_input_projection
    return load_matrix_input_projection(inputs,spec)


def _dataset(projection,inputs,spec):
    return seal({'contract_version':'stock_fold_dataset_v3',
        'prepared_view_ref':inputs['prepared_view']['prepared_view_ref'],
        'feature_ref':projection.features['feature_ref'],'label_ref':projection.labels['label_ref'],
        'raw_label_refs':projection.raw_refs,'fold_spec_ref':digest(spec),'fit_cutoff':spec['fit_cutoff'],
        'ordered_features':projection.common['ordered_features'],'selectors':inputs['selectors'],
        'training_keys_digest':digest(projection.training_keys),
        'training_rows_ref':projection.training_rows_ref,'training_row_count':len(projection.training_keys),
        'excluded':projection.excluded,'target_semantics':TARGET_SEMANTICS,
        'normalization':NORMALIZATION_SPEC,'validation':'none_fixed_parameters_no_early_stopping'},'dataset_ref')


def build_matrix_fold(inputs, *, spec,destination,metrics=None,batch=None):
    from .stock_fold_artifacts import StockMLFold
    from .stock_ml import _implementation,_environment,_signal_evidence
    from .stock_folds import prediction_rows
    from .stock_training import fit_predict_stock_model
    from .feature_catalog import load_feature_catalog
    begin=time.perf_counter(); inputs,spec=deepcopy(inputs),deepcopy(spec)
    with _projection(inputs,spec,batch) as projection:
        common=projection.common; catalog=load_feature_catalog()
        require(common['catalog_ref']==catalog.identity and common['ordered_features']==[
            f['id'] for f in catalog.select(common['feature_selection'])], 'current catalog selection mismatch')
        implementations=_implementation()
        definition={'version':'axiom.stock_ml_fold/3','input_manifest':inputs,'input_manifest_ref':digest(inputs),
            'fold_spec':spec,'fold_spec_ref':digest(spec),'catalog_ref':common['catalog_ref'],
            'parameters':LGBM_PARAMETERS,'num_boost_round':TREES,'target_semantics':TARGET_SEMANTICS,
            'label_normalization':NORMALIZATION_SPEC,'environment':_environment(),
            'implementation_sources':implementations,'implementation_ref':digest(implementations)}
        definition_ref=digest(definition); target=Path(destination)/definition_ref[7:]
        zeros=dict(data_read_calls=0,supplier_calls=0,feature_core_calls=0,label_core_calls=0,
            core_calls=0,account_calls=0,train_calls=0,predict_calls=0)
        if target.exists():
            saved=load_matrix_fold(target,projection=projection,batch=batch)
            require(saved.to_dict()['definition']==definition,'cached matrix fold definition mismatch')
            if metrics is not None: metrics.update(zeros,cache_hit=True,total_seconds=time.perf_counter()-begin)
            return StockMLFold(target,True)
        require(len(projection.training_keys)>=40,'insufficient mature finite training rows')
        dataset=_dataset(projection,inputs,spec); features,labels=projection.features,projection.labels
        stats={**zeros,'cache_hit':False,'saved_input_validation_seconds':time.perf_counter()-begin}
        booster,scores=fit_predict_stock_model(projection.X,projection.y,projection.P,
            ordered_features=common['ordered_features'],parameters=LGBM_PARAMETERS,num_boost_round=TREES,metrics=stats)
        require(len(scores)==len(projection.candidate_keys) and all(_finite(float(s)) for s in scores),
                'invalid native model scores')
        score_map={tuple(key):float(score) for key,score in zip(projection.candidate_keys,scores)}
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.matrix-fold-',dir=target.parent) as temporary:
            stage=Path(temporary)/'complete'; stage.mkdir(); booster.save_model(str(stage/'booster.txt'))
            model=seal({'contract_version':'stock_model_release_v2','dataset_ref':dataset['dataset_ref'],
                'feature_ref':features['feature_ref'],'label_ref':labels['label_ref'],'raw_label_refs':projection.raw_refs,
                'fit_cutoff':spec['fit_cutoff'],'simulated_available_at':spec['simulated_model_available_at'],
                'clock_basis':'declared_simulation','ordered_features':common['ordered_features'],
                'feature_selection':common['feature_selection'],'catalog_ref':common['catalog_ref'],
                'target_semantics':TARGET_SEMANTICS,'parameters':LGBM_PARAMETERS,'num_boost_round':TREES,
                'environment':definition['environment'],'implementation_ref':definition['implementation_ref'],
                'booster_digest':file_digest(stage/'booster.txt'),
                'feature_normalization':'same_date_visible_members_cs_zscore_no_fit',
                'label_normalization':NORMALIZATION_SPEC},'model_ref')
            predictions=seal({'contract_version':'stock_prediction_run_v2','signal_stage':'prediction_raw',
                'score_semantics':TARGET_SEMANTICS,'score_unit':'dimensionless','feature_ref':features['feature_ref'],
                'model_ref':model['model_ref'],'fold_spec_ref':digest(spec),'clock_basis':'declared_simulation',
                'universe':common['universe'],'rows':prediction_rows(features,spec,model,score_map),
                'limitations':['Declared simulation publication clocks are not historical realtime completion evidence.',
                    'Feature inputs retain their original best-effort historical availability and immutable refs.',
                    'Scores are normalized-target predictions, not return percentages.',
                    'Engine validation and Runtime account execution are not performed by this builder.']},'signal_run_ref')
            evidence=_signal_evidence(predictions,projection.evaluation,spec['evaluation_cutoff'])
            refs={key:value[key] for key,value in [('feature_ref',features),('label_ref',labels),
                ('dataset_ref',dataset),('model_ref',model),('signal_run_ref',predictions),('evidence_ref',evidence)]}
            fold=seal({'contract_version':'stock_ml_fold_v3','definition':definition,'definition_ref':definition_ref,
                'status':'COMPLETE',**refs,'fold_ref':digest({'definition_ref':definition_ref,**refs}),
                'engine_admission':{'neutral_validation':'NOT_PERFORMED_BY_BUILDER','runtime':'NOT_PERFORMED_BY_BUILDER'},
                'limitations':['Explicit immutable prepared-view parent paths remain required.',
                    'Only the active fit matrix is projected; native model memory is separately measured.',
                    'No Data, supplier, Feature, label normalization or account execution in this builder.']},'content_digest')
            values={'fold.json':fold,'feature-slice.json':features,'label-slice.json':labels,'dataset.json':dataset,
                'model.json':model,'predictions.json':predictions,'signal-evidence.json':evidence}
            for name,value in values.items(): write_json(stage/name,value)
            write_json(stage/'manifest.json',{'contract_version':'stock_ml_fold_manifest_v2',
                'fold_ref':fold['fold_ref'],'files':{n:file_digest(stage/n) for n in [*values,'booster.txt']}})
            load_matrix_fold(stage,projection=projection,batch=batch)
            try: stage.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
                require(load_matrix_fold(target,projection=projection,batch=batch).identity==fold['fold_ref'],
                        'concurrent matrix fold conflict')
        stats.update(training_rows=len(projection.training_keys),prediction_rows=len(predictions['rows']),
            valid_predictions=len(projection.candidate_keys),
            matrix_bytes=sum(x.nbytes for x in (projection.X,projection.y,projection.P)),
            artifact_bytes=sum(p.stat().st_size for p in target.iterdir()),total_seconds=time.perf_counter()-begin)
        if metrics is not None: metrics.update(stats)
        return StockMLFold(target)


def _predictions(predictions,features,common,spec,model):
    require(set(predictions)=={'contract_version','signal_run_ref','signal_stage','score_semantics','score_unit',
        'feature_ref','model_ref','limitations','universe','rows','fold_spec_ref','clock_basis'} and
        predictions['contract_version']=='stock_prediction_run_v2' and predictions['signal_stage']=='prediction_raw' and
        predictions['score_unit']=='dimensionless' and predictions['clock_basis']=='declared_simulation' and
        predictions['score_semantics']==model['target_semantics'] and predictions['fold_spec_ref']==digest(spec) and
        predictions['feature_ref']==features['feature_ref'] and predictions['model_ref']==model['model_ref'] and
        predictions['universe']==common['universe'],'prediction contract/stage mismatch')
    source={(r['security_id'],r['session']):r for r in features['rows']}
    expected={(s,d) for d in features['prediction_sessions'] for s in common['universe']}; indexed={}
    for row in predictions['rows']:
        require(set(row)=={'security_id','session','knowledge_cutoff','available_at','score','valid','invalid_reason',
            'source_refs','member','feature_knowledge_cutoff','feature_available_at','simulated_model_available_at'},
            'prediction row fields mismatch')
        key=row['security_id'],row['session']; require(key in expected and key not in indexed,'duplicate/unexpected prediction key')
        indexed[key]=row; original=source[key]; day=key[1]; parent=features['parents_by_session'][day]
        require(row['feature_knowledge_cutoff']==original['knowledge_cutoff'] and
            row['feature_available_at']==feature_available(original) and row['member'] is original['member'] and
            row['simulated_model_available_at']==model['simulated_available_at'] and
            row['source_refs']==[features['feature_ref'],parent['feature_ref'],parent['qlib_view_ref'],
                model['model_ref'],common['catalog_ref']],'prediction original source/clock mismatch')
        infer=_instant(spec['inference_cutoff_by_session'][day])
        require(_instant(row['knowledge_cutoff'])==_instant(row['available_at'])==infer and
            _instant(model['simulated_available_at'])<infer and _instant(row['feature_knowledge_cutoff'])<=infer,
            'prediction inference clock mismatch')
        valid=original['member'] and all(original['validity']) and all(_finite(v) for v in original['values']) and feature_available(original) is not None
        reason=None if valid else 'NOT_MEMBER' if not original['member'] else (
            'FEATURE_AVAILABILITY_UNKNOWN' if feature_available(original) is None else 'FEATURE_MISSING')
        require(type(row['valid']) is bool and row['valid']==valid and row['invalid_reason']==reason and
            (_finite(row['score']) if valid else row['score'] is None),'prediction validity/value mismatch')
    require(set(indexed)==expected,'complete prediction union required')


def load_matrix_fold(path, *, projection=None,batch=None):
    """Complete readonly v3 closure; no Data/Core/model execution is imported."""
    from .stock_compact_store import OwnedStore
    with OwnedStore() as ingress:
        return _load_matrix_fold(path,projection=projection,batch=batch,ingress=ingress)


def _load_matrix_fold(path, *, projection,batch,ingress):
    from .stock_fold_artifacts import StockMLFold
    path=Path(path).resolve(); manifest=ingress.read_json({'path':str(path/'manifest.json')})
    require(set(manifest)=={'contract_version','fold_ref','files'} and
        manifest['contract_version']=='stock_ml_fold_manifest_v2' and
        set(manifest['files'])=={*OUTPUTS,'fold.json','booster.txt'},'unexpected compact fold manifest')
    documents={}
    for name,ref in manifest['files'].items():
        descriptor={'path':str(path/name),'file_digest':ref}
        if name=='booster.txt': ingress.read(descriptor)
        else: documents[name]=ingress.read_json(descriptor,key='content_digest' if name=='fold.json' else OUTPUTS[name])
    fold=documents['fold.json']; definition=fold['definition']
    require(fold['contract_version']=='stock_ml_fold_v3' and fold['status']=='COMPLETE' and
        definition['version']=='axiom.stock_ml_fold/3' and definition['parameters']==LGBM_PARAMETERS and
        definition['num_boost_round']==TREES and definition['target_semantics']==TARGET_SEMANTICS and
        definition['label_normalization']==NORMALIZATION_SPEC,'unsupported compact fold profile')
    spec,inputs=definition['fold_spec'],definition['input_manifest']
    require(fold['definition_ref']==digest(definition) and definition['input_manifest_ref']==digest(inputs) and
        definition['fold_spec_ref']==digest(spec) and definition['implementation_ref']==digest(definition['implementation_sources']),
        'compact fold definition mismatch')
    refs={key:fold[key] for key in OUTPUTS.values()}
    require(fold['fold_ref']==digest({'definition_ref':fold['definition_ref'],**refs})==manifest['fold_ref'],'compact fold identity mismatch')
    saved={}
    for name,key in OUTPUTS.items():
        value=documents[name]; require(value[key]==fold[key],'fold stage reference mismatch'); saved[name]=value
    own=projection is None
    if own:
        projection=(batch._project_evaluation(inputs,spec) if batch is not None
                    else _projection(inputs,spec,batch))
    try:
        features,labels=projection.features,projection.labels; common=projection.common
        require(saved['feature-slice.json']==features and saved['label-slice.json']==labels,'saved matrix slice/parent mismatch')
        dataset,model=saved['dataset.json'],saved['model.json']; predictions,evidence=saved['predictions.json'],saved['signal-evidence.json']
        binding=getattr(projection,'fold_binding',None)
        expected_dataset=binding['dataset'] if binding is not None else _dataset(projection,inputs,spec)
        expected_raw_refs=expected_dataset['raw_label_refs']
        require((binding is None or binding['labels']==labels) and dataset==expected_dataset and
                expected_dataset['training_row_count']>=40,'saved compact training selection mismatch')
        require(model['contract_version']=='stock_model_release_v2','unsupported saved model contract')
        for actual,expected in ((model['dataset_ref'],dataset['dataset_ref']),(model['feature_ref'],features['feature_ref']),
            (model['label_ref'],labels['label_ref']),(model['raw_label_refs'],expected_raw_refs),
            (model['fit_cutoff'],spec['fit_cutoff']),(model['simulated_available_at'],spec['simulated_model_available_at']),
            (model['clock_basis'],'declared_simulation'),(model['ordered_features'],common['ordered_features']),
            (model['feature_selection'],common['feature_selection']),(model['catalog_ref'],common['catalog_ref']),
            (model['parameters'],LGBM_PARAMETERS),(model['num_boost_round'],TREES),
            (model['target_semantics'],TARGET_SEMANTICS),(model['label_normalization'],NORMALIZATION_SPEC),
            (model['environment'],definition['environment']),(model['implementation_ref'],definition['implementation_ref']),
            (model['booster_digest'],manifest['files']['booster.txt'])):
            require(actual==expected,'saved matrix model linkage mismatch')
        require(_instant(model['fit_cutoff'])<_instant(model['simulated_available_at']),'model publication precedes fit')
        _predictions(predictions,features,common,spec,model)
        require(evidence['contract_version']=='stock_signal_evidence_v1' and
            evidence['signal_ref']==predictions['signal_run_ref'] and evidence['label_ref']==projection.evaluation['label_ref'] and
            evidence['evaluation_cutoff']==spec['evaluation_cutoff'] and evidence['score_semantics']==predictions['score_semantics'] and
            [r['session'] for r in evidence['series']]==features['prediction_sessions'],'saved evidence input linkage mismatch')
        require(fold['engine_admission']=={'neutral_validation':'NOT_PERFORMED_BY_BUILDER','runtime':'NOT_PERFORMED_BY_BUILDER'},
                'fold cannot claim Engine Runtime admission')
        # A standalone build also borrows a verified projection. A backend
        # changing its immutable parents cannot publish a COMPLETE fold.
        projection._store.check()
        if batch is not None: batch._check_sources()
        ingress.check()
        return StockMLFold(path)
    finally:
        if own: projection.close()
