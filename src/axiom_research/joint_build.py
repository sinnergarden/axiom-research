"""A bounded market + latest visible financial report FeatureBuild, using Data and Core.

Research owns orchestration and durable reuse. Data selects each report revision
at the actual session cutoff; Core alone executes identity, return and asof.
There is no financial forward fill or alternate feature executor here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from axiom_engine.core import FeaturePlan, FactBatch, ExecutionContext, execute_feature_plan, ABI
from . import contracts as c
from .api import to_dict, from_dict, semantic_identity, validate
from .data_adapter import (adapt_decision_batch, AdapterError, _require, _digest,
                           _utc, _value, _versioned, _availability)
from .view_ref import ViewRef


class _Batch:
    def __init__(self, wire):
        self.wire = wire

    def to_json(self):
        return deepcopy(self.wire)


class _DailyHistory:
    """Index once; each Core call receives only its bounded market lookback."""
    def __init__(self, batch):
        wire = batch.to_json()
        self.context = wire['context']
        self.rows = {}
        for row in wire['records']:
            self.rows.setdefault(row['session'], []).append(row)
        self.fields = {}
        for field, spec in wire['field_meta'].items():
            by_session = {}
            for meta in spec['by_key']:
                by_session.setdefault(meta['session'], []).append(meta)
            self.fields[field] = ({k:v for k,v in spec.items() if k!='by_key'}, by_session)

    def slice(self, sessions):
        q = self.context['query']
        return _Batch(dict(
            records=[row for session in sessions for row in self.rows.get(session,[])],
            field_meta={field:{**spec,'by_key':[m for session in sessions for m in index.get(session,[])]}
                        for field,(spec,index) in self.fields.items()},
            context={**self.context,'query':{**q,'sessions':list(sessions),
                'cutoff_by_session':{s:q['cutoff_by_session'][s] for s in sessions}}}))


def _daily_query(q):
    return dict(fields=list(q.fields), symbols=list(q.symbols), sessions=list(q.sessions),
        pit_policy=q.pit_policy, cutoff_by_session={s: str(v.isoformat() if hasattr(v, 'isoformat') else v)
            for s, v in q.cutoff_by_session.items()}, purpose=q.purpose,
        price_basis=q.price_basis, adjustment_anchor=q.adjustment_anchor,
        universe_id=q.universe_id, policy_by_session=dict(q.policy_by_session) if q.policy_by_session else None)


def _ref(kind, value, *, uri='inline:definition'):
    digest = _digest(value)
    return c.ArtifactRef(artifact_type=kind, artifact_id=digest[7:],
        artifact_contract_version='1', content_digest=digest, uri=uri,
        metadata={'definition': value})


_NUMERIC = {'float64','float32','int64','int32','integer','int','float','double'}
_FINANCIAL_DOMAINS = {'financial_events','balance_sheet_events','cash_flow_events','financial_indicator_events'}


def _identity(data, snapshot, markets, membership, financials, lag, implementation):
    from axiom_data.reader import READER_VERSION
    from axiom_data.event_reader import EVENT_READER_VERSION
    validate(implementation, require_resolved=True)
    _require(snapshot not in ('', 'current', 'latest'), 'pin a concrete Snapshot')
    _require(type(lag) is int and lag > 0, 'positive market lag required')
    _require(isinstance(markets, Mapping) and markets and isinstance(financials, Mapping) and financials,
             'named market_queries and financial_queries required')
    aliases=list(markets)+list(financials)
    _require(all(isinstance(a,str) and a.isidentifier() for a in aliases) and len(set(aliases))==len(aliases),
             'unique input aliases required')
    market=next(iter(markets.values()))
    mq, rq = _daily_query(market), _daily_query(membership)
    _require(market.sessions and tuple(market.sessions)==tuple(sorted(set(market.sessions))),
             'ordered unique sessions required')
    _require(market.symbols and len(set(market.symbols))==len(market.symbols), 'unique symbols required')
    _require(membership.domain=='universe_membership' and membership.fields==('is_member',) and
             membership.purpose=='decision_facts' and membership.policy_by_session is None and
             all(mq[k]==rq[k] for k in ('symbols','sessions','pit_policy','cutoff_by_session')),
             'membership scope/cutoff mismatch')
    _require(set(mq['cutoff_by_session'])==set(mq['sessions']), 'complete cutoffs required')
    manifest=data.store.load_snapshot(snapshot)  # metadata only on reuse
    views=[]; definitions=[]; columns=[]
    for alias,q in list(markets.items())+list(financials.items()):
        _require(q.purpose=='decision_facts', 'joint build only accepts decision_facts')
        _require(q.fields and len(set(q.fields))==len(q.fields), 'unique numeric fields required')
        _require(q.pit_policy==market.pit_policy and tuple(q.symbols)==tuple(market.symbols),
                 'input scope/policy mismatch')
        contract=manifest['domains'][q.domain]['contract']
        if alias in markets:
            wire=_daily_query(q)
            _require(q.price_basis=='unadjusted' and q.policy_by_session is None,
                     'joint build requires native market facts and one PIT policy')
            _require(all(wire[k]==mq[k] for k in ('sessions','cutoff_by_session')), 'market cutoff mismatch')
            views.append(ViewRef(snapshot,q.domain,wire,READER_VERSION).as_artifact_ref())
        else:
            _require(q.domain in _FINANCIAL_DOMAINS and q.time_field=='report_period', 'native financial report domain required')
            _require(date.fromisoformat(q.start).isoformat()==q.start and
                     date.fromisoformat(q.end).isoformat()==q.end and q.start<=q.end, 'ordered report range required')
            _require(all(q.filters.get(k) for k in ('endpoint','report_type') if k in contract['logical_key']),
                     'bind financial endpoint/report_type in filters')
            views.append(_ref('FinancialDecisionView',dict(snapshot_id=snapshot,alias=alias,domain=q.domain,
                fields=list(q.fields),symbols=list(q.symbols),start=q.start,end=q.end,time_field=q.time_field,
                filters=dict(q.filters),pit_policy=q.pit_policy,purpose=q.purpose,
                cutoff_by_session=mq['cutoff_by_session'],reader_version=READER_VERSION,
                event_reader_version=EVENT_READER_VERSION,
                report_selection='latest_visible_report_period_not_after_session_v1')))
        for field in q.fields:
            spec=contract['fields'][field]
            _require(str(spec['dtype']).lower() in _NUMERIC and isinstance(spec.get('unit'),str) and spec['unit'],
                     'declared numeric fields and units required')
            name=alias+'__'+field
            outputs=[(name,spec['unit'],'identity',1,0)] if alias in markets else [
                (name,spec['unit'],'Data PIT; latest report_period <= session; Core asof(exact_date,event_order)',0,0)]
            if alias in markets:
                outputs.append((name+'__return','dimensionless',f'pct_change(periods={lag},fill_method=none,zero=missing)',1,lag))
            for output,unit,formula,window,shift in outputs:
                columns.append(c.Column(name=output,dtype='float64',unit=unit,stage='base'))
                definitions.append(c.FeatureDefinition(name=output,business_definition=alias+' / '+field,
                    inputs=(name,),formula=formula,implementation_ref=implementation,window_sessions=window,
                    lag_sessions=shift,missing_policy='preserve with Core reasons',outlier_policy='none',
                    reference_universe_policy='explicit same-Snapshot membership'))
    reference=ViewRef(snapshot,membership.domain,rq,READER_VERSION).as_artifact_ref()
    views.append(reference)
    _require(len({col.name for col in columns})==len(columns),'duplicate output names')
    scope=c.SessionRange(start=market.sessions[0],end=market.sessions[-1])
    req=c.Requirement(name='joint_decision_inputs',required_by=tuple(col.name for col in columns),
        input_semantics='fixed Snapshot, named daily inputs and per-cutoff PIT reports',
        output_semantics='keyed numeric panel with original availability, missing/revision provenance',
        edge_cases=('no visible report','retraction','late receipt','same-announcement correction','multiple domains'),
        reference_fixture='tests/test_joint_build.py',status='DECISION',evidence=tuple(views))
    requirements=c.DataRequirements(requirements=(req,),key=('security_id','session'),scope=scope,
        lookback_sessions=lag+1,label_extension_sessions=0,pit_policy=market.pit_policy,public_view_binding=views[0])
    release=c.FeatureRelease(name='joint_latest_visible_reports_v1',
        business_definition='Named market identities/returns and latest visible financial values; no labels/models',
        plan=c.FeaturePlanSpec(key=('security_id','session'),features=tuple(definitions),ordered_output_schema=tuple(columns),
            data_requirements=requirements,history_policy='partial',pit_policy=market.pit_policy,
            cutoff_policy='explicit per-session cutoff; Core UTC second precision; exact_date asof',execution_abi=ABI))
    return validate(c.FeatureBuildIdentity(data_refs=(_ref('DataSnapshot',manifest),),view_refs=tuple(views),
        feature_release=release,scope=scope,lookback_sessions=lag+1,pit_policy=market.pit_policy,
        cutoff_policy=release.plan.cutoff_policy,reference_universe=reference,implementation_package=implementation))


def _execute(adapted_inputs, event_batches, session, lag):
    first_alias,first=next(iter(adapted_inputs.items()))
    plan,facts=first.plan.to_dict(),first.facts.to_dict()
    plan['input_schema']=[];plan['nodes']=[];plan['sources']=[];plan['event_schema']={}
    facts['schema']=[];facts['sources']=[];facts['event_schema']={};facts['events']=[]
    rows={}
    evidence={}
    def column(name,unit,stage='base'):
        return dict(name=name,dtype='float64',unit=unit,stage=stage,missing='preserve')
    for alias,adapted in adapted_inputs.items():
        incoming=adapted.facts.to_dict()
        names={col['name']:'input__'+alias+'__'+col['name'] for col in incoming['schema']}
        ids={source['id']:alias+':'+source['id'] for source in incoming['sources']}
        for source in incoming['sources']:
            plan['sources'].append({**source,'id':ids[source['id']]})
            evidence[ids[source['id']]]={**adapted.source_evidence[source['id']],'alias':alias}
        for col in incoming['schema']:
            _require(col['dtype']=='float64','numeric market columns required')
            plan['input_schema'].append({**col,'name':names[col['name']]})
            output=alias+'__'+col['name']
            plan['nodes'].extend([
                dict(name=output,op='identity',version='1',inputs=[names[col['name']]],params={},column=column(output,col['unit'])),
                dict(name=output+'__return',op='pct_change',version='1',inputs=[names[col['name']]],
                    params={'periods':lag,'fill_method':'none','zero':'missing'},column=column(output+'__return','dimensionless'))])
        for row in incoming['rows']:
            key=row['security_id'],row['session']
            dest=rows.setdefault(key,dict(security_id=key[0],session=key[1],values=[],availability=[],sources=[],missing_reasons=[]))
            for field in ('values','availability','missing_reasons'):dest[field].extend(row[field])
            dest['sources'].extend([[ids[i] for i in refs] for refs in row['sources']])
    facts['schema']=plan['input_schema'];facts['rows']=[rows[k] for k in sorted(rows)]
    for alias,event_batch in event_batches.items():
        records,metadata,ctx=_versioned(event_batch,'decision_facts')
        _require(ctx['snapshot_id']==first.view_ref.snapshot_id and
                 ctx['query']['pit_policy']==first.view_ref.query['pit_policy'] and
                 _utc(ctx['query']['cutoff'])==first.context.to_dict()['cutoffs'][session],'event binding/cutoff mismatch')
        fields=ctx['query']['fields'];keys=ctx['logical_key']
        specs={field:metadata[field] for field in fields}
        indexes={field:{tuple(str(m[k]) for k in keys):m for m in spec['by_key']} for field,spec in specs.items()}
        expected={tuple(str(r[k]) for k in keys) for r in records}
        _require(len(expected)==len(records) and all(set(index)==expected and len(specs[f]['by_key'])==len(records)
                 for f,index in indexes.items()),'event provenance incomplete or duplicate')
        plan['event_schema'][alias]=[column(field,specs[field]['unit'],'fact') for field in fields]
        # Revision choice belongs to Data. The recipe selects the latest economic
        # report period, including null/retracted cells, never the last receipt.
        selected={}
        for row in records:
            security,period=row['security_id'],row['report_period']
            _require(security in ctx['query']['symbols'],'unexpected event security')
            if period>session:continue
            old=selected.get(security)
            if old is None or period>old['report_period']:selected[security]=row
            elif period==old['report_period']:raise AdapterError('ambiguous latest-report stream')
        for security,row in sorted(selected.items()):
            values=[];availability=[];sources=[];reasons=[]
            for field in fields:
                _require(specs[field]['dtype'] in _NUMERIC,'numeric financial field required')
                meta=indexes[field][tuple(str(row[k]) for k in keys)]
                value=_value(row[field],'float64');reason=meta.get('missing_reason')
                _require(value is None or reason is None,'present event has missing reason')
                available=_availability(meta,_utc(ctx['query']['cutoff']))
                source_id=_digest({'alias':alias,'context':ctx,'field':field,'provenance':meta})
                basis=str(meta.get('availability_basis') or 'missing')
                plan['sources'].append(dict(id=source_id,data_ref=_digest({'snapshot_id':ctx['snapshot_id'],'domain':ctx['domain']}),
                    view_ref=ViewRef.from_batch(event_batch).digest,revision_policy=ctx['query']['pit_policy'],
                    qualification='synthetic' if 'synthetic' in basis else 'best_effort' if 'assumption' in basis
                        else 'verified' if meta.get('evidence_ref') else 'observed',availability_basis=basis))
                evidence[source_id]=dict(alias=alias,field=field,provenance=meta,query_context=ctx,
                    report_selection='latest_visible_report_period_not_after_session_v1')
                values.append(value);availability.append(available);sources.append([source_id])
                reasons.append(str(reason or 'MISSING') if value is None else None)
            facts['events'].append(dict(stream=alias,security_id=security,
                event_id=alias+':'+security+':'+row['report_period'],event_session=min(availability)[:10],
                report_period=row['report_period'],values=values,availability=availability,sources=sources,missing_reasons=reasons))
        for field in fields:
            output=alias+'__'+field
            plan['nodes'].append(dict(name=output,op='asof',version='1',inputs=[],
                params={'stream':alias,'field':field,'match':'exact_date','report_policy':'event_order'},
                column=column(output,specs[field]['unit'])))
    plan['sources']=sorted(plan['sources'],key=lambda s:s['id'])
    plan['outputs']=[{'node':n['name'],'column':n['column']} for n in plan['nodes']]
    facts['sources']=plan['sources'];facts['event_schema']=plan['event_schema']
    context=first.context.to_dict()
    for row in context['reference']:row['source']=first_alias+':'+row['source']
    frame=execute_feature_plan(FeaturePlan.from_dict(plan),FactBatch.from_dict(facts),ExecutionContext.from_dict(context))
    return frame.to_dict(),evidence


def _file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FeatureBuild:
    identity: c.FeatureBuildIdentity
    path: Path
    reused: bool

    def read(self):
        """Read saved numeric panel after checking its committed byte digest."""
        import pyarrow.parquet as pq
        _check(self.path, self.identity)
        return pq.read_table(self.path/'panel.parquet').to_pandas()

    def evidence(self):
        """Read persisted per-cell Core validity and original Data provenance."""
        _check(self.path, self.identity)
        return json.loads((self.path/'evidence.json').read_text())


def load_feature_build(path: str | Path) -> FeatureBuild:
    """Open a persisted build without Data reads or Core execution."""
    path = Path(path)
    identity = from_dict(json.loads((path/'manifest.json').read_text())['identity'])
    _require(isinstance(identity, c.FeatureBuildIdentity), 'expected FeatureBuildIdentity')
    _check(path, identity)
    return FeatureBuild(identity, path, True)


def _check(path, identity):
    manifest = json.loads((path/'manifest.json').read_text())
    _require(manifest.get('schema_version') == 'research_feature_build_v1' and
             semantic_identity(from_dict(manifest['identity'])) == semantic_identity(identity), 'saved build identity mismatch')
    _require(set(manifest['files']) == {'panel.parquet','evidence.json'}, 'invalid saved build files')
    for name,digest in manifest['files'].items():
        _require(_file_digest(path/name)==digest, 'saved FeatureBuild integrity failure: '+name)


def build_joint_features(data: Any, *, snapshot: str, market_queries: Mapping[str, Any],
        membership_query: Any, financial_queries: Mapping[str, Any], destination: str | Path,
        implementation_package: c.ArtifactRef, lag_sessions: int = 1) -> FeatureBuild:
    """Build/reuse named numeric inputs, merging by security_id/session.

    Query mappings assign stable aliases; each query may request multiple fields.
    Market inputs share sessions, symbols, cutoffs and PIT policy. Financial
    queries declare report-period ranges and one endpoint/report_type stream.
    Their cutoff is replaced by each daily cutoff. Data selects revisions;
    each financial stream selects its latest visible report_period <= session,
    including null/retracted values, then Core asof uses original availability.

    Columns are alias__field, plus alias__field__return for each native market
    numeric field. This fixed recipe uses Core identity/pct_change/asof; arbitrary
    FeaturePlans, derived TTM, nonnumeric inputs, labels and training are outside
    this API. Caller provides ordered exchange sessions including lookback.

    Writes only destination; identical identity reuses saved files without fact
    reads or Core execution. Pin a reviewed implementation package covering
    Research and Core. Corruption raises and never overwrites existing builds.
    """
    identity=_identity(data,snapshot,market_queries,membership_query,financial_queries,lag_sessions,implementation_package)
    target=Path(destination)/semantic_identity(identity).split(':')[-1]
    if target.exists():
        _check(target,identity)
        return FeatureBuild(identity,target,True)
    markets={alias:_DailyHistory(data.read(snapshot=snapshot,query=q)) for alias,q in market_queries.items()}
    membership=_DailyHistory(data.members(snapshot=snapshot,query=membership_query))
    template=next(iter(market_queries.values()));sessions=tuple(template.sessions)
    frames=[];evidence={};contexts={};records=[]
    recipe_ref=semantic_identity(identity.feature_release)
    for i,session in enumerate(sessions):
        history=sessions[max(0,i-lag_sessions):i+1]
        outputs=tuple((symbol,session) for symbol in template.symbols)
        inputs={alias:adapt_decision_batch(batch.slice(history),reference=membership.slice(history),
                    recipe_ref=recipe_ref,output_keys=outputs,lag_sessions=lag_sessions) for alias,batch in markets.items()}
        events={alias:data.events(snapshot=snapshot,query=replace(q,cutoff=template.cutoff_by_session[session]))
                for alias,q in financial_queries.items()}
        frame,sources=_execute(inputs,events,session,lag_sessions)
        _require([(x['name'],x['unit']) for x in frame['schema']]==
                 [(x.name,x.unit) for x in identity.feature_release.plan.ordered_output_schema], 'output schema differs from release')
        frames.append(frame)
        for source,record in sources.items():
            ctx=record['query_context']
            context_ref=record['alias']+':'+ctx['domain']+':'+session
            contexts.setdefault(context_ref,ctx)
            evidence[source]={k:v for k,v in record.items() if k!='query_context'}
            evidence[source]['query_context_ref']=context_ref
        for row in frame['rows']:
            records.append({'security_id':row['security_id'],'session':row['session'],
                **{col['name']:value for col,value in zip(frame['schema'],row['values'])}})
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=pa.schema([('security_id',pa.string()),('session',pa.string())]+
        [(col.name,pa.float64()) for col in identity.feature_release.plan.ordered_output_schema])
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.feature-build-',dir=target.parent) as staging:
        stage=Path(staging)/'complete';stage.mkdir()
        pq.write_table(pa.Table.from_pylist(records,schema=schema),stage/'panel.parquet')
        (stage/'evidence.json').write_text(json.dumps({'frames':frames,'source_evidence':evidence,'query_contexts':contexts},
            sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        (stage/'manifest.json').write_text(json.dumps({'schema_version':'research_feature_build_v1',
            'identity':to_dict(identity),'files':{n:_file_digest(stage/n) for n in ('panel.parquet','evidence.json')}},
            sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        try:stage.rename(target)
        except FileExistsError:
            _check(target,identity)
            return FeatureBuild(identity,target,True)
    return FeatureBuild(identity,target,False)
