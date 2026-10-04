"""Execute one predeclared, fixed-Snapshot Qlib/LightGBM sample; never collect."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

from axiom_research.stock_artifacts import digest, write_json
from axiom_research.stock_ml import build_stock_ml_experiment,predict_stock_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',required=True,help='saved public-Reader scope/calendar evidence JSON')
    p.add_argument('--destination',required=True)
    p.add_argument('--receipt',required=True,help='new local acceptance path; never overwrite')
    args=p.parse_args(); receipt=Path(args.receipt)
    if receipt.exists(): raise ValueError('acceptance receipt already exists')
    scope=json.loads(Path(args.scope).read_text()); calendar=scope['calendar']; plan=scope['scope']
    if plan['missing_reasons']: raise ValueError('scope is not ready')
    from axiom_data import Data
    from axiom_research import load_feature_catalog
    catalog=load_feature_catalog()
    evaluation_sessions=[s for s in calendar if '2024-01-01'<=s<='2024-01-31']
    pred_sessions=[calendar[calendar.index(s)-1] for s in evaluation_sessions]
    first=calendar.index(min(pred_sessions)); fit_day=calendar[first-1]
    feature_sessions=[s for s in plan['read_sessions'] if '2023-10-01'<=s<=max(pred_sessions)]
    config={'snapshot':scope['snapshot'],'universe_id':plan['universe_id'],
        'symbols':plan['read_symbols'],'calendar':calendar,'read_sessions':plan['read_sessions'],
        'feature_sessions':feature_sessions,'prediction_sessions':sorted(set(pred_sessions)),
        'fit_cutoff':fit_day+'T20:30:00+08:00','pit_policy':'best_effort_vendor_v1',
        'cutoff_by_session':{s:s+'T20:30:00+08:00' for s in plan['read_sessions']},
        'evaluation_cutoff':calendar[-1]+'T20:30:00+08:00',
        'feature_selection':[s.to_dict() for s in catalog.default_selection],
        'scope_ref':digest(plan),'calendar_ref':digest(scope['calendar_evidence'])}
    metrics={};start=time.perf_counter()
    def progress(item):
        if item['completed']==1 or item['completed']%10==0 or item['completed']==item['total']:
            print(json.dumps(item),flush=True)
    experiment=build_stock_ml_experiment(Data(scope['root']),config=config,destination=args.destination,
                                         metrics=metrics,progress=progress)
    metrics['total_seconds']=time.perf_counter()-start
    metrics['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # A second invocation must take the saved-only branch before any execution.
    class NoFacts:
        def __getattr__(self,name): raise AssertionError('cache tried Data.'+name)
    cached_metrics={};cached_start=time.perf_counter()
    cached=build_stock_ml_experiment(NoFacts(),config=config,destination=args.destination,metrics=cached_metrics)
    cached_metrics['seconds']=time.perf_counter()-cached_start
    if cached.identity!=experiment.identity or not cached.reused: raise AssertionError('cache reuse mismatch')
    features=json.loads((experiment.path/'features.json').read_text())
    saved=experiment.predictions(); valid_keys={(r['security_id'],r['session']) for r in saved['rows'] if r['valid']}
    selected=[r for r in features['rows'] if (r['security_id'],r['session']) in valid_keys]
    independent=predict_stock_model(experiment.path,selected)
    expected={(r['security_id'],r['session']):r['score'] for r in saved['rows'] if r['valid']}
    errors=[abs(v-expected[r['security_id'],r['session']]) for r,v in zip(selected,independent)]
    if max(errors,default=0)>1e-12: raise AssertionError('saved native model prediction mismatch')
    result={'status':'MODEL_SIGNAL_PASS_ACCOUNT_PENDING','created_at':datetime.now(timezone.utc).isoformat(),
        'scope_ref':digest(plan),'experiment_path':str(experiment.path),'experiment_ref':experiment.identity,
        'metrics':metrics,'cache_reuse':cached_metrics,'independent_model_max_error':max(errors,default=0),
        'evaluation_sessions':evaluation_sessions,'input_config':config,
        'account_status':experiment.to_dict()['account_status'],
        'limits':['One fixed January 2024 fold; no full-year or long-history execution.',
                  'No stock account return or execution admission claim.']}
    receipt.parent.mkdir(parents=True,exist_ok=True);write_json(receipt,result)
    print(json.dumps({k:v for k,v in result.items() if k!='input_config'},ensure_ascii=False))


if __name__=='__main__':main()
