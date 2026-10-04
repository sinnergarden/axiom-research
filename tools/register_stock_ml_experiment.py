"""Register an owner-saved stock ML experiment in the existing Research index."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from axiom_research import ArtifactRef,ExperimentStore,ExperimentReader,load_stock_ml_experiment
from axiom_research.stock_artifacts import digest,file_digest


def register(args):
    saved=load_stock_ml_experiment(args.experiment);doc=saved.to_dict();base=saved.path.resolve()
    definition=doc['definition'];config=definition['config']
    outputs=[]
    for filename,kind,key,version in (
        ('features.json','FeatureBuild','feature_ref','stock_feature_build_v1'),
        ('labels.json','LabelBuild','label_ref','stock_label_bundle_v1'),
        ('dataset.json','TrainingDataset','dataset_ref','stock_training_dataset_v1'),
        ('model.json','ModelRelease','model_ref','stock_model_release_v1'),
        ('predictions.json','SignalRun','signal_run_ref','stock_prediction_run_v1'),
        ('signal-evidence.json','SignalEvidence','evidence_ref','stock_signal_evidence_v1'),
        ('experiment.json','StockMLExperiment','experiment_ref','stock_ml_experiment_v1')):
        outputs.append(ArtifactRef(artifact_type=kind,artifact_id=doc[key],
            artifact_contract_version=version,content_digest=file_digest(base/filename),
            uri=str(base/filename),metadata={'owner_loader':'axiom_research.load_stock_ml_experiment',
                                           'artifact_directory':str(base)}))
    inputs=[ArtifactRef(artifact_type='StockInputConfig',artifact_id=digest(config),
        artifact_contract_version='stock_input_config_v1',content_digest=digest(config),
        uri='inline:fixed_input_config',metadata={'definition':config}),
        ArtifactRef(artifact_type='ImplementationSources',artifact_id=definition['implementation_ref'],
        artifact_contract_version='implementation_sources_v1',content_digest=digest(definition['implementation_sources']),
        uri='inline:source_file_digests',metadata={'definition':definition['implementation_sources']})]
    writer=ExperimentStore(args.index)
    writer.register_saved_experiment(question=dict(question_id=args.question_id,title=args.title,
        description=args.description,hypothesis=args.hypothesis),version=dict(label=args.version_label,
        explanation=args.explanation,parent_version_ref=None,parameters={**config,
            'model_parameters':definition['parameters'],'num_boost_round':definition['num_boost_round'],
            'catalog_ref':definition['catalog_ref']},input_refs=inputs,explicit_changes=args.change),
        run=dict(status='COMPLETE',output_refs=outputs,reason=doc['account_reason'],
            outcome='Model/prediction/label evidence saved; account BLOCKED_PENDING_STOCK_RUNTIME_ADMISSION.',
            backtest_ref=None,evaluation_ref=None))
    writer.update_organization(args.question_id,expected_revision=0,groups=['Stocks'],
                               tags=['qlib','lightgbm','5-session','fixed-oos'],favorite=False,shelved=False)
    return ExperimentReader(args.index).detail(args.question_id)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',required=True,type=Path);p.add_argument('--experiment',required=True,type=Path)
    for name in ('question-id','title','description','hypothesis','version-label','explanation'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--change',action='append',default=[])
    args=p.parse_args();print(json.dumps(register(args),ensure_ascii=False,sort_keys=True))


if __name__=='__main__':main()
