"""Save fixed stock ML scope from public Reader diagnostics; no collection."""
from __future__ import annotations
import argparse
from datetime import date,timedelta
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--calendar-start',default='2023-08-25');p.add_argument('--calendar-end',default='2024-02-08')
    p.add_argument('--feature-start',default='2023-10-01');p.add_argument('--feature-end',default='2024-01-31')
    p.add_argument('--calendar-probe-security',default='cnstock.000001.SZ.19910403')
    p.add_argument('--universe',default='csi300')
    args=p.parse_args();target=Path(args.output)
    if target.exists(): raise ValueError('scope evidence path exists; never overwrite')
    if args.snapshot in ('','current','latest'): raise ValueError('pin a concrete Snapshot')
    from axiom_data import Data,QuerySpec
    from axiom_research.stock_artifacts import write_json
    data=Data(args.data_root);start=date.fromisoformat(args.calendar_start);end=date.fromisoformat(args.calendar_end)
    if end<start: raise ValueError('calendar range reversed')
    days=tuple((start+timedelta(days=i)).isoformat() for i in range((end-start).days+1))
    q=QuerySpec('market_daily',('close',),(args.calendar_probe_security,),days,'best_effort_vendor_v1',
                {s:s+'T20:30:00+08:00' for s in days})
    begin=time.perf_counter();wire=data.states(snapshot=args.snapshot,query=q).to_json()
    meta={(m['security_id'],m['session']):m for m in wire['field_meta']['market_state']['by_key']}
    if len(wire['records'])!=len(days) or len(meta)!=len(days): raise ValueError('incomplete calendar diagnostics')
    calendar=[]
    for row in wire['records']:
        reason=meta[row['security_id'],row['session']].get('missing_reason') or ''
        if reason.startswith(('calendar_','identity_')): raise ValueError('calendar not proven: '+reason)
        if row['market_state']!='calendar_closed': calendar.append(row['session'])
    calendar.sort()
    outputs=tuple(s for s in calendar if args.feature_start<=s<=args.feature_end)
    scope=data.plan_scope(snapshot=args.snapshot,universe_id=args.universe,output_sessions=outputs,
        cutoff_by_session={s:s+'T20:30:00+08:00' for s in outputs},pit_policy='best_effort_vendor_v1',
        lookback_sessions=20,exchange='SSE')
    result={'root':str(Path(args.data_root).resolve()),'snapshot':args.snapshot,'calendar':calendar,
            'scope':scope,'elapsed_seconds':time.perf_counter()-begin,'calendar_evidence':wire}
    target.parent.mkdir(parents=True,exist_ok=True);write_json(target,result)
    print(json.dumps({'snapshot':args.snapshot,'read_sessions':len(scope['read_sessions']),
        'read_symbols':len(scope['read_symbols']),'missing_reasons':len(scope['missing_reasons']),
        'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
