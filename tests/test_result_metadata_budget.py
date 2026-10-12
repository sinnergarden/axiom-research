"""Static saved-output metadata exercises real readers without numerical replay."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (ExperimentReader, ExperimentRecordError,
    begin_stock_result_registration, load_stock_sequential_result, publish_stock_result,
    semantic_identity)
from axiom_research.result_discovery import read_json
from axiom_research.result_registration import _save_stock_sequential_receipt
from axiom_research.stock_artifacts import digest, file_digest, write_json
from axiom_research.stock_derived_signal import _artifact
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_folds import OUTPUTS, _label_normalization
from axiom_research.stock_target_spec import resolve_stock_label_spec, stock_label_wire, target_semantics
from axiom_research.stock_training import resolve_stock_training_spec
from test_stock_target_config_blocks import label, training


def saved_output_fixture(root):
    """Declare small frozen values and headers; these are not fitted-model evidence."""
    specs={'label':label(), 'model':training()}; refs={k:semantic_identity(v) for k,v in specs.items()}
    target=resolve_stock_label_spec(specs['label']); parameters,rounds,model_binding=resolve_stock_training_spec(specs['model'])
    day='2024-01-03'; calendar=['2024-01-02',day]; cutoff=day+'T13:00:00Z'
    scope=dict(calendar=calendar,sessions=[day],universe=['A'],evaluation_cutoff='2024-02-01T00:00:00Z')
    spec=dict(inference_cutoff_by_session={day:cutoff})
    inputs={'contract_version':'stock_ml_saved_inputs_v5','ancestor':str(root/'unopened-ancestors.json')}
    folder=root/'fold'; folder.mkdir()
    booster=folder/'booster.txt'; booster.write_text('Static saved header; no fitted model.\n')
    catalog=digest('static-catalog'); parent=dict(feature_ref=digest('static-day'),qlib_view_ref=digest('static-view'))
    feature_row=dict(security_id='A',session=day,knowledge_cutoff=day+'T12:30:00Z',
        member=True,values=[1.0],validity=[True],availability=[day+'T12:30:00Z'])
    features=seal(dict(contract_version='stock_feature_slice_v4',universe=['A'],catalog_ref=catalog,
        prediction_sessions=[day],parents_by_session={day:parent},rows=[feature_row]),'feature_ref')
    labels=seal(dict(contract_version='stock_training_label_slice_v3',rows=[]),'label_ref')
    dataset=seal(dict(contract_version='stock_fold_dataset_v3',fixture='static-output-metadata'),'dataset_ref')
    label_wire=stock_label_wire(target,calendar=calendar,anchor=calendar[0])
    model=seal(dict(contract_version='stock_model_release_v3',dataset_ref=dataset['dataset_ref'],
        feature_ref=features['feature_ref'],label_ref=labels['label_ref'],parameters=parameters,
        num_boost_round=rounds,booster_digest=file_digest(booster),target_semantics=target_semantics({'target_spec':target}),
        simulated_available_at='2024-01-02T11:00:00Z',label_spec=label_wire,label_spec_ref=digest(label_wire)),'model_ref')
    row=dict(security_id='A',session=day,knowledge_cutoff=cutoff,available_at=cutoff,score=.25,
        valid=True,invalid_reason=None,member=True,feature_knowledge_cutoff=feature_row['knowledge_cutoff'],
        feature_available_at=feature_row['availability'][0],simulated_model_available_at=model['simulated_available_at'],
        source_refs=[features['feature_ref'],parent['feature_ref'],parent['qlib_view_ref'],model['model_ref'],catalog])
    predictions=seal(dict(contract_version='stock_prediction_run_v3',signal_stage='raw_prediction',
        score_semantics=model['target_semantics'],score_unit='dimensionless',feature_ref=features['feature_ref'],
        model_ref=model['model_ref'],limitations=['Static output fixture'],universe=['A'],rows=[row],
        fold_spec_ref=digest(spec),clock_basis='declared_simulation',label_spec=label_wire,
        label_spec_ref=digest(label_wire),label_normalization=_label_normalization()),'signal_run_ref')
    evidence=seal(dict(contract_version='stock_signal_evidence_v1',fixture='static-output-metadata'),'evidence_ref')
    documents=dict(zip(OUTPUTS,(features,labels,dataset,model,predictions,evidence)))
    definition=dict(input_manifest=inputs,input_manifest_ref=digest(inputs),fold_spec=spec,fold_spec_ref=digest(spec),
        target_spec=target,parameters=parameters,num_boost_round=rounds,**model_binding)
    output_refs={field:documents[name][field] for name,field in OUTPUTS.items()}
    fold_ref=digest(dict(definition_ref=digest(definition),**output_refs))
    fold=seal(dict(contract_version='stock_ml_fold_v4',status='COMPLETE',definition=definition,
        definition_ref=digest(definition),fold_ref=fold_ref,**output_refs),'content_digest')
    for name,value in {**documents,'fold.json':fold}.items(): write_json(folder/name,value)
    write_json(folder/'manifest.json',dict(contract_version='stock_ml_fold_manifest_v2',fold_ref=fold_ref,
        files={name:file_digest(folder/name) for name in [*OUTPUTS,'fold.json','booster.txt']}))
    spec_artifact=_artifact(folder/'fold.json',spec,'StockFoldSpec',digest(spec))
    spec_artifact['manifest_uri']+='#definition/fold_spec'
    binding=dict(kind='raw',fold_ref=fold_ref,fold_spec_ref=digest(spec),model_ref=model['model_ref'],
        feature_ref=features['feature_ref'],signal_run_ref=predictions['signal_run_ref'],fold_spec_artifact=spec_artifact,
        model_metadata_artifact=_artifact(folder/'model.json',model,'StockModelRelease',model['model_ref']),
        prediction_artifact=_artifact(folder/'predictions.json',predictions,'StockPredictionRun',predictions['signal_run_ref']))
    options={'column_block':32}
    definition=dict(fold_specs=[spec],preparation_options={**options,'control_layout':'fold_controls_v2'},
        target_spec=target,feature_view={'spec':{'calendar':calendar,'universe':['A']}})
    batch=dict(contract_version='stock_ml_batch_inputs_v5',status='COMPLETE',definition=definition,
        definition_ref=digest(definition),folds=[dict(fold_spec=spec,input_manifest=inputs)],
        metrics={'static_metadata_padding':'x'*(3*1024**2)})
    batch['batch_ref']=digest(batch); batch=seal(batch,'content_digest')
    dataset_cfg=dict(fold_specs=[spec],preparation_options=options,model_feature_selection=None,scope=scope)
    spec_ref=digest(dict(contract_version='stock_effective_configuration_v1',spec_refs=refs))
    cfg=dict(dataset=dataset_cfg,dataset_ref=digest(dataset_cfg),
        specification=dict(configuration_ref=spec_ref,spec_refs=refs,specs=specs))
    cfg['configuration_ref']=digest(dict(contract_version='stock_sequential_effective_configuration_v1',
        specification_ref=spec_ref,dataset_ref=cfg['dataset_ref']))
    signal_input=root/'signal-input.json'; write_json(signal_input,dict(input_id=digest('static-input'),scope=scope))
    result=dict(configuration_ref=cfg['configuration_ref'],dataset_configuration_ref=cfg['dataset_ref'],
        spec_refs=refs,batch_manifest=batch,prediction_bindings=[binding],derived_signals=[],
        folds=[dict(path=str(folder),fold_ref=fold_ref,fold_spec_ref=digest(spec),signal_run_ref=predictions['signal_run_ref'])],
        raw_signal_evaluation_input={},signal_evaluation_input=dict(uri=str(signal_input),
            artifact_id=digest('static-input'),content_digest=file_digest(signal_input)))
    return cfg,result


class ResultMetadataBudgetTests(unittest.TestCase):
    def test_large_saved_receipt_public_load_and_registration_keep_explicit_budget(self):
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp); cfg,result=saved_output_fixture(root)
            calls=[stack.enter_context(patch(name,side_effect=AssertionError('business execution'))) for name in (
                'axiom_data.Data.read','axiom_data.Data.members','axiom_engine.core.execute_feature_columns',
                'axiom_research.stock_training.fit_predict_stock_model','axiom_engine.runtime.run_stock_backtest')]
            path=_save_stock_sequential_receipt(result,destination=root)
            self.assertGreater(path.stat().st_size,3*1024**2)
            self.assertGreater((path.parent/'batch.json').stat().st_size,3*1024**2)
            before={p:(file_digest(p),p.stat().st_mtime_ns) for p in path.parent.iterdir()}
            self.assertEqual(_save_stock_sequential_receipt(result,destination=root),path)
            with self.assertRaisesRegex(ExperimentRecordError,'saved metadata exceeds byte budget'):
                read_json(path,maximum=2*1024**2)
            saved=load_stock_sequential_result(path,research_root=root)
            self.assertEqual(saved.identity,digest(result))
            context=dict(request_path=root/'request.json',index_path=root/'index.json',roots={'research':root})
            frozen=begin_stock_result_registration(cfg,**context,required_stages=['training'],imported=True)
            self.assertEqual(frozen['registration_mode'],'existing_saved_import')
            head=publish_stock_result(**context,experiment_result=path)
            reader=ExperimentReader(context['index_path']); manifest=reader.result(head['result_id'],roots=context['roots'])
            self.assertEqual(head['status'],'COMPLETE')
            ref=next(a for a in manifest['artifacts'] if a['artifact_type']=='StockSequentialResult')
            self.assertEqual((ref['artifact_id'],ref['content_digest']),(saved.identity,saved.content_digest))
            self.assertEqual(reader.index()['result_heads'],[head])
            self.assertEqual(before,{p:(file_digest(p),p.stat().st_mtime_ns) for p in before})
            for call in calls: call.assert_not_called()


if __name__=='__main__': unittest.main()
