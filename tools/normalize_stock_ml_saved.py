"""Derive one normalized-target model using saved Features/raw Labels only."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import resource
import shutil
import time
from unittest.mock import patch

from axiom_research import load_stock_ml_experiment,predict_stock_model
from axiom_research.stock_artifacts import file_digest,write_json
from axiom_research.stock_ml import build_stock_ml_experiment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--destination',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--model-only',required=True,type=Path)
    args=p.parse_args()
    if args.receipt.exists() or args.model_only.exists(): raise ValueError('new output paths required')
    source=load_stock_ml_experiment(args.source);config=source.to_dict()['definition']['config']
    unchanged={f.name:{'digest':file_digest(f),'mtime_ns':f.stat().st_mtime_ns}
               for f in source.path.iterdir() if f.is_file()}
    class NoFacts:
        def __getattr__(self,name): raise AssertionError('saved build tried Data.'+name)
    metrics={};begin=time.perf_counter()
    with patch('axiom_research.stock_ml._prepare_stock_inputs',side_effect=AssertionError('Feature rerun')):
        run=build_stock_ml_experiment(NoFacts(),config=config,destination=args.destination,
                                     reuse_input_path=source.path,metrics=metrics)
        metrics['build_seconds']=time.perf_counter()-begin
        cache_metrics={};begin=time.perf_counter()
        with patch('lightgbm.train',side_effect=AssertionError('cache trained')), \
             patch('axiom_research.stock_label_normalization.execute_feature_plan',side_effect=AssertionError('cache Core')):
            cached=build_stock_ml_experiment(NoFacts(),config=config,destination=args.destination,
                                           reuse_input_path=source.path,metrics=cache_metrics)
        cache_metrics['seconds']=time.perf_counter()-begin
        if not cached.reused or cached.identity!=run.identity: raise AssertionError('cache identity mismatch')
    args.model_only.mkdir(parents=True)
    for name in ('model.json','booster.txt'): shutil.copyfile(run.path/name,args.model_only/name)
    features=json.loads((run.path/'features.json').read_text());predictions=run.predictions()
    expected={(r['security_id'],r['session']):r['score'] for r in predictions['rows'] if r['valid']}
    selected=[r for r in features['rows'] if (r['security_id'],r['session']) in expected]
    predicted=predict_stock_model(args.model_only,selected,ordered_features=features['ordered_features'],
                                  feature_selection=features['selection'])
    error=max((abs(p-expected[r['security_id'],r['session']]) for p,r in zip(predicted,selected)),default=0)
    if error>1e-12: raise AssertionError('independent native prediction differs')
    labels=json.loads((run.path/'labels.json').read_text());sections=[]
    for section in labels['normalized_training']['sections']:
        values=[r['normalized_target'] for r in labels['normalized_training']['rows']
                if r['feature_session']==section['feature_session'] and r['valid']]
        if values:
            mean=sum(values)/len(values);variance=sum((v-mean)**2 for v in values)/len(values)
            if abs(mean)>1e-10 or abs(variance-1)>1e-10: raise AssertionError('invalid population zscore')
            sections.append({'session':section['feature_session'],'count':len(values),'mean':mean,'variance_ddof0':variance})
    for name,proof in unchanged.items():
        f=source.path/name
        if file_digest(f)!=proof['digest'] or f.stat().st_mtime_ns!=proof['mtime_ns']:
            raise AssertionError('old immutable artifact changed: '+name)
    if run.to_dict()['feature_ref']!=source.to_dict()['feature_ref']: raise AssertionError('Feature identity changed')
    receipt={'status':'NORMALIZED_MODEL_SIGNAL_PASS_ACCOUNT_PENDING',
        'created_at':datetime.now(timezone.utc).isoformat(),'source_experiment_ref':source.identity,
        'experiment_path':str(run.path.resolve()),'experiment':run.to_dict(),'metrics':metrics,
        'cache_reuse':cache_metrics,'independent_model_max_error':error,
        'independent_model_path':str(args.model_only.resolve()),'training_sections':sections,
        'old_artifacts_unchanged':unchanged,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'score_semantics':predictions['score_semantics'],
        'limits':['Saved Feature/raw Label reuse: no Data read or Feature re-execution.',
                  'One fixed January 2024 fold; no full-year or long-history stock execution.',
                  'No stock account admission or return claim.']}
    args.receipt.parent.mkdir(parents=True,exist_ok=True);write_json(args.receipt,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('experiment','training_sections','old_artifacts_unchanged')},ensure_ascii=False))


if __name__=='__main__': main()
