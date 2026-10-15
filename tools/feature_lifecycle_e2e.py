"""One explicit real-input lifecycle acceptance; never starts full history.

Run only after implementation is frozen. An exclusive start receipt prevents
accidental repetition. Failure is recorded once and is not retried. The eight
output sessions retain the full physical universe and the catalog's warmup.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import traceback


def footprint(root):
    seen=set();result={'unique_logical_bytes':0,'allocated_bytes':0,'private_allocated_bytes':0,
                      'multiply_linked_logical_bytes':0,'files':0}
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        st=path.stat();key=(st.st_dev,st.st_ino)
        if key in seen:continue
        seen.add(key);result['files']+=1
        result['unique_logical_bytes']+=st.st_size
        result['allocated_bytes']+=st.st_blocks*512
        if st.st_nlink>1:result['multiply_linked_logical_bytes']+=st.st_size
        else:result['private_allocated_bytes']+=st.st_blocks*512
    result['scope']='destination only; hardlinks counted once here; APFS shared extents and whole-platform peak unmeasured'
    return result


def _performance_acceptance(cfg,run):
    """Checks are part of this one full run, not separate profiling attempts."""
    from collections import defaultdict
    from dataclasses import replace
    from datetime import datetime,timedelta
    from axiom_data import EventReadOwner,DataBatch,ttm,single_quarter
    from axiom_data.event_reader import read_events
    from axiom_research import financial_feature_columns as financial
    from axiom_research import feature_input_evidence as evidence
    counts=defaultdict(int);recent=defaultdict(list);phase={};validation_seconds=0.0
    original_iter=EventReadOwner.iter_cutoffs
    original_spans=EventReadOwner.iter_security_spans
    original_memo=financial._memoized_results
    original_projection=financial._quarter_projection
    original_compact=evidence.compact_reader_evidence
    original_materialize=evidence.materialize_evidence_rows
    original_receipt=evidence.input_block_receipt
    disk_peak={'private_allocated_bytes':0,'destination_allocated_bytes':0,'sampled':True};last_disk_sample=0.0
    source_clocks={}
    def compare(left,right):
        assert left.frame.equals(right.frame)
        assert left.field_meta==right.field_meta and left.context==right.context
    def projection(batch,rows,tokens,day):
        nonlocal validation_seconds
        projected,eligible,selected_tokens=original_projection(batch,rows,tokens,day)
        security='cnstock.003040.SZ.20210322'
        if batch.context['domain']=='financial_events' and 'revenue' in batch.field_meta and security in tokens and day in ('2017-07-03','2017-07-04'):
            started=time.monotonic()
            original=[r for r in rows if r['security_id']==security]
            irregular=[r for r in original if r['report_period']=='2017-01-31']
            if day=='2017-07-03':
                assert not irregular
                counts['nonquarter_case_before_visibility']+=1
            else:
                assert len(irregular)==1 and irregular[0]['revenue']==81660000.0
                kept=[r for r in eligible if r['security_id']==security]
                anchor=max(r['report_period'] for r in kept)
                assert anchor=='2016-12-31'
                assert next(r for r in kept if r['report_period']==anchor)['revenue']==1163950000.0
                assert all(r['report_period']!='2017-01-31' for r in projected.field_meta['revenue']['by_key'] if r['security_id']==security)
                assert all('2017-01-31' not in key for key in selected_tokens[security])
                counts['nonquarter_case_eligible_anchor']+=1
            validation_seconds+=time.monotonic()-started
        return projected,eligible,selected_tokens
    def spans(owner,*,query,cutoffs,sessions,economic_boundary=True,maximum_group_bytes=None):
        nonlocal validation_seconds
        identity=(query.domain,query.fields)
        for security,start,stop,batch in original_spans(owner,query=query,cutoffs=cutoffs,sessions=sessions,
                economic_boundary=economic_boundary,maximum_group_bytes=maximum_group_bytes):
            started=time.monotonic()
            saved_rows=owner.selected_records;saved_tokens=owner.selected_tokens
            if counts[('span_selector',identity)]==0 and saved_rows:
                probe=replace(query,symbols=(security,))
                field=query.fields[0]
                boundary=datetime.fromisoformat(batch.field_meta[field]['by_key'][-1]['usable_from'])
                for delta in (-1,0,1):
                    cutoff=(boundary+timedelta(microseconds=delta)).isoformat()
                    selected=read_events(owner,owner.snapshot,replace(probe,cutoff=cutoff),
                        _snapshot=owner.manifest,_evidence=owner.evidence[query.domain],_owner=owner,_borrow_context=True)
                    baseline=read_events(owner.store,owner.snapshot,replace(probe,cutoff=cutoff),
                        _snapshot=owner.manifest,_evidence=owner.evidence[query.domain],_borrow_context=True)
                    compare(selected,baseline);counts['real_revision_boundary_cutoffs']+=1
                baseline=read_events(owner.store,owner.snapshot,replace(probe,cutoff=cutoffs[stop-1]),
                    _snapshot=owner.manifest,_evidence=owner.evidence[query.domain],_borrow_context=True)
                assert batch.frame.equals(baseline.frame) and batch.field_meta==baseline.field_meta
                clock=batch.field_meta[field]['by_key'][-1]['usable_from']
                previous=source_clocks.setdefault(security,{})
                if any(domain!=query.domain and value!=clock for domain,value in previous.items()):
                    counts['asynchronous_source_clock_comparisons']+=1
                previous[query.domain]=clock
                counts['unchanged_span_end_comparisons']+=1;counts[('span_selector',identity)]+=1
            if security==query.symbols[0] and start==0:
                counts['stock_block_boundary_visits']+=1
            if query.domain=='financial_events' and 'revenue' in query.fields and security=='cnstock.003040.SZ.20210322':
                if sessions[start]<='2017-07-03'<=sessions[stop-1]:
                    assert not any(r['report_period']=='2017-01-31' for r in saved_rows)
                    counts['nonquarter_case_before_visibility']+=1
            owner.selected_records=saved_rows;owner.selected_tokens=saved_tokens
            validation_seconds+=time.monotonic()-started
            yield security,start,stop,batch
    def cutoffs(owner,*,query,cutoffs,_borrow_context=False):
        nonlocal validation_seconds
        identity=(query.domain,query.fields)
        for batch in original_iter(owner,query=query,cutoffs=cutoffs,_borrow_context=_borrow_context):
            if counts[('events',identity)]==0:
                started=time.monotonic()
                # Real revision availability boundary: original selector and
                # cached grouping at one microsecond before/equal/after.
                candidates=[(datetime.fromisoformat(row['usable_from']),row['security_id'])
                            for row in batch.field_meta[query.fields[0]]['by_key']]
                if candidates:
                    boundary,security=min(candidates,key=lambda item:abs((item[0]-datetime.fromisoformat(batch.context['query']['cutoff'])).total_seconds()))
                    probe=replace(query,symbols=(security,))
                    saved_records=owner.selected_records;saved_tokens=owner.selected_tokens
                    for delta in (-1,0,1):
                        cutoff=(boundary+timedelta(microseconds=delta)).isoformat()
                        selected=read_events(owner,owner.snapshot,replace(probe,cutoff=cutoff),
                            _snapshot=owner.manifest,_evidence=owner.evidence[query.domain],_owner=owner,_borrow_context=True)
                        baseline=read_events(owner.store,owner.snapshot,replace(probe,cutoff=cutoff),
                            _snapshot=owner.manifest,_evidence=owner.evidence[query.domain],_borrow_context=True)
                        compare(selected,baseline)
                        counts['real_revision_boundary_cutoffs']+=1
                    owner.selected_records=saved_records;owner.selected_tokens=saved_tokens
                    counts[('events',identity)]+=1
                validation_seconds+=time.monotonic()-started
            yield batch
    def memo(batch,event_rows,tokens,targets_by_name,outputs,field,query,unit,owner):
        nonlocal validation_seconds
        result=original_memo(batch,event_rows,tokens,targets_by_name,outputs,field,query,unit,owner)
        identity=(query.domain,field)
        if counts[('financial',identity)]<2 and any(result.values()):
            started=time.monotonic()
            for recipe,values in result.items():
                pairs=list(values)[:8]
                if not pairs:continue
                securities=sorted({s for s,p in pairs});periods=sorted({p for s,p in pairs});chosen=set(securities)
                if recipe in ('ttm','single_quarter'):
                    restricted=DataBatch(batch.frame.loc[batch.frame['security_id'].isin(chosen)],
                        {field:{**batch.field_meta[field],'by_key':[r for r in batch.field_meta[field]['by_key'] if r['security_id'] in chosen]}},
                        {**batch.context,'query':{**batch.context['query'],'symbols':securities}})
                    baseline=(ttm if recipe=='ttm' else single_quarter)(restricted,field=field,
                        endpoint=query.filters['endpoint'],report_type=query.filters['report_type'],
                        basis='cumulative_ytd',unit=unit,periods=periods,_borrow_context=True)
                    rows={(r['security_id'],r['report_period']):r for r in baseline.frame.to_dict('records')}
                    meta={(r['security_id'],r['report_period']):r for r in baseline.field_meta[field]['by_key']}
                else:
                    rows={(r['security_id'],r['report_period']):r for r in event_rows};meta={}
                    for item in batch.field_meta[field]['by_key']:
                        pair=(item['security_id'],item['report_period'])
                        meta[pair]={'status':item['status'],'missing_reason':item.get('missing_reason'),
                            'components':[{'security_id':pair[0],'report_period':pair[1],'status':item['status'],'provenance':item}]}
                for pair in pairs:
                    proof=meta.get(pair,{'components':[],'status':'source_missing','missing_reason':'no_visible_report'})
                    value=rows.get(pair,{}).get(field) if proof['status']=='value' else None
                    assert values[pair]==financial._stable_result(value,proof,recipe)
                    counts['financial_reference_cells']+=1
            # Repeat identical dependencies to explicitly exercise memo hits.
            before=owner.metrics['hits']
            assert original_memo(batch,event_rows,tokens,targets_by_name,outputs,field,query,unit,owner)==result
            assert owner.metrics['hits']>before or not any(result.values())
            counts['memo_hit_comparisons']+=1
            if counts['missing_quarter_control']==0:
                for recipe,values in result.items():
                    if recipe!='ttm' or not values:continue
                    pair=next((pair for pair in values if any(r['security_id']==pair[0] and r['report_period']==pair[1] for r in event_rows)),None)
                    if pair is None:continue
                    security,period=pair
                    keep=(batch.frame['security_id']==security)&(batch.frame['report_period']!=period)
                    missing=DataBatch(batch.frame.loc[keep],
                        {field:{**batch.field_meta[field],'by_key':[r for r in batch.field_meta[field]['by_key'] if r['security_id']==security and r['report_period']!=period]}},
                        {**batch.context,'query':{**batch.context['query'],'symbols':[security]}})
                    control=ttm(missing,field=field,endpoint=query.filters['endpoint'],report_type=query.filters['report_type'],basis='cumulative_ytd',unit=unit,periods=[period],_borrow_context=True)
                    proof=control.field_meta[field]['by_key'][0]
                    assert proof['status']=='source_missing' and proof['missing_reason']=='missing_quarter'
                    counts['missing_quarter_control']+=1
                    # Original financial recipe also preserves a selected
                    # retraction; it must not fall back to an older value.
                    from copy import deepcopy
                    retracted=deepcopy(batch.field_meta[field]);frame=batch.frame.loc[batch.frame['security_id']==security].copy()
                    retracted['by_key']=[r for r in retracted['by_key'] if r['security_id']==security]
                    for item in retracted['by_key']:
                        if item['report_period']==period:item.update(status='retracted',missing_reason='retracted')
                    frame.loc[frame['report_period']==period,field]=None
                    control=ttm(DataBatch(frame,{field:retracted},missing.context),field=field,
                        endpoint=query.filters['endpoint'],report_type=query.filters['report_type'],basis='cumulative_ytd',unit=unit,periods=[period],_borrow_context=True)
                    assert control.field_meta[field]['by_key'][0]['status']=='retracted'
                    counts['synthetic_retraction_formula_control']+=1
                    break
            if counts['same_value_revision_invalidation']==0:
                from copy import deepcopy
                for name,targets in targets_by_name.items():
                    recipe=outputs[name]['derivation']
                    if recipe not in ('ttm','single_quarter'):continue
                    found=None
                    for security,period in targets.items():
                        needed=financial._dependencies(period,recipe)
                        found=next(((security,period,row) for row in event_rows
                                    if row['security_id']==security and row['report_period'] in needed),None)
                        if found:break
                    if found:
                        security,period,row=found;native=tuple(str(row[k]) for k in batch.context['logical_key'])
                        changed_tokens={s:dict(v) for s,v in tokens.items()}
                        changed_tokens[security][native]=(*tokens[security][native],'same-value-revision-control')
                        changed_meta=deepcopy(batch.field_meta)
                        for item in changed_meta[field]['by_key']:
                            if tuple(str(item[k]) for k in batch.context['logical_key'])==native:
                                item['revision_id']=str(item['revision_id'])+'-control'
                        changed=DataBatch(batch.frame,changed_meta,batch.context)
                        before=owner.metrics['misses']
                        revised=original_memo(changed,event_rows,changed_tokens,{name:{security:period}},
                                              {name:outputs[name]},field,query,unit,owner)
                        assert owner.metrics['misses']>before
                        assert revised[recipe][security,period][0:2]==result[recipe][security,period][0:2]
                        # A same-value revision invalidates computation; no
                        # per-cell provenance digest is persisted in the new format.
                        counts['same_value_revision_invalidation']+=1
                        break
            counts[('financial',identity)]+=1
            validation_seconds+=time.monotonic()-started
        return result
    def forbidden_proof(*args,**kwargs):
        counts['legacy_proof_calls']+=1
        raise AssertionError('new input path constructed or expanded legacy cell proofs')
    def receipt(**kwargs):
        value=original_receipt(**kwargs)
        assert value['contract_version']=='reader_column_block_v1'
        assert all('by_key' not in meta for meta in value['field_meta'].values())
        size=len(json.dumps(value).encode())
        assert size<64*1024
        counts['block_receipts']+=1
        counts['maximum_block_receipt_bytes']=max(counts['maximum_block_receipt_bytes'],size)
        return value
    def observe(record):
        nonlocal last_disk_sample
        if time.monotonic()-last_disk_sample>=60:
            current=footprint(Path(cfg['destination']))
            disk_peak['private_allocated_bytes']=max(disk_peak['private_allocated_bytes'],current['private_allocated_bytes'])
            disk_peak['destination_allocated_bytes']=max(disk_peak['destination_allocated_bytes'],current['allocated_bytes'])
            last_disk_sample=time.monotonic()
        if record['stage']!='financial_inputs' or 'detail' not in record:return
        detail=record['detail'];bundle=record['bundle'];phase[bundle]=detail
        if detail.get('completed_dates',0)>=8:
            recent[bundle].append(detail['last_block_seconds']*2/cfg.get('input_block_sessions',2))
            recent[bundle]=recent[bundle][-3:]
            if cfg.get('maximum_financial_block_seconds') is not None and len(recent[bundle])==3 and sum(recent[bundle])/3>=cfg['maximum_financial_block_seconds']:
                raise RuntimeError('FINANCIAL_THROUGHPUT_STOP: three two-session blocks exceed declared threshold')
    EventReadOwner.iter_cutoffs=cutoffs;financial._memoized_results=memo;evidence.compact_reader_evidence=forbidden_proof
    financial._quarter_projection=projection
    EventReadOwner.iter_security_spans=spans
    evidence.materialize_evidence_rows=forbidden_proof;evidence.input_block_receipt=receipt
    try:
        result=run(cfg,observer=observe)
        assert counts['real_revision_boundary_cutoffs']>0 and counts['financial_reference_cells']>0
        assert counts['same_value_revision_invalidation']==1 and counts['block_receipts']>0 and counts['legacy_proof_calls']==0
        assert counts['nonquarter_case_before_visibility']>0 and counts['nonquarter_case_eligible_anchor']>0 and counts['missing_quarter_control']==1
        assert counts['unchanged_span_end_comparisons']>0 and counts['stock_block_boundary_visits']>0
        return result,{'checks':{str(k):v for k,v in counts.items()},'validation_seconds':validation_seconds,'financial_phases':phase,'disk_peak':disk_peak}
    finally:
        EventReadOwner.iter_cutoffs=original_iter;financial._memoized_results=original_memo;evidence.compact_reader_evidence=original_compact
        financial._quarter_projection=original_projection
        EventReadOwner.iter_security_spans=original_spans
        evidence.materialize_evidence_rows=original_materialize;evidence.input_block_receipt=original_receipt
        Path(cfg['destination'],'performance-validation.json').write_text(json.dumps({
            'checks':{str(k):v for k,v in counts.items()},'validation_seconds':validation_seconds,
            'financial_phases':phase,'disk_peak':disk_peak,'maximum_financial_block_seconds':cfg['maximum_financial_block_seconds']},indent=2))


def acceptance(cfg):
    import numpy as np
    from prepare_qsys_history import run
    from axiom_engine.feature_library import FeatureLibrary
    from axiom_research.qsys_catalog import load_qsys_feature_catalog
    from axiom_research.stock_feature_library import SavedColumnFeatureSource,CatalogFeatureLibrary
    from axiom_research.stock_feature_library_adapter import load_feature_library_training_view
    from axiom_research.stock_compact_store import (feature_eligibility_columns,training_matrix,
        feature_training_blocks,clear_feature_window)
    from axiom_research.stock_matrix_storage import instant_us
    from axiom_research.stock_compact_batch import TargetRows,ConcatRows,target_eligibility_reasons
    from axiom_data import Data
    root=Path(cfg['destination']);catalog=load_qsys_feature_catalog()
    selection=[s.to_dict() for s in catalog.default_selection]
    assert len(cfg['output_sessions'])==8 and len(selection)==401
    result,performance=_performance_acceptance(cfg,run)
    assert result['dates']==8 and result['columns']==401 and result['securities']==3396
    assert result['warmup']==756 and result['read_dates']==764
    limits={'maximum_source_bytes':cfg['maximum_source_bytes'],
            'maximum_matrix_bytes':cfg['working_bytes'],'maximum_parent_bytes':64*1024**2}
    library=FeatureLibrary(root/'library',maximum_source_bytes=cfg['maximum_source_bytes'],
                           maximum_working_bytes=cfg['working_bytes'])
    # Changed request grouping must query real coverage, with no Core calls.
    before={str(p.relative_to(root/'library')) for p in (root/'library').rglob('*') if p.is_file()}
    with SavedColumnFeatureSource(result['source'],maximum_source_bytes=cfg['maximum_source_bytes'],
                                  maximum_working_bytes=cfg['working_bytes']) as source:
        owner=CatalogFeatureLibrary(library,source,catalog)
        _,warm=owner.ensure(selection=selection,output_sessions=cfg['output_sessions'][1:7],normalized=False)
        assert warm['raw']['core_calls']==0 and warm['raw']['computed_output_cells']==0
        known=source.source.arrays['member_known']
        member=source.source.arrays['member']
        unknown=int((~known).sum())
        assert not bool((member & ~known).any())
    after={str(p.relative_to(root/'library')) for p in (root/'library').rglob('*') if p.is_file()}
    assert not any('/partitions/' in '/'+p or p.startswith('buffers/') for p in after-before)
    # Additional Axiom normalization is an explicit small request; it reuses
    # existing raw columns instead of duplicating all 401 normalized columns.
    with SavedColumnFeatureSource(result['source'],maximum_source_bytes=cfg['maximum_source_bytes'],
                                  maximum_working_bytes=cfg['working_bytes']) as source:
        owner=CatalogFeatureLibrary(library,source,catalog)
        norm_view,norm=owner.ensure(selection=selection[:2],output_sessions=cfg['output_sessions'][-1:],normalized=True)
        assert norm['raw']['core_calls']==0
        with library.open_feature_view(norm_view) as view:
            assert len(view.describe()['columns'])==2
    # A fresh training owner must not open Data or re-admit source ancestry.
    original_reader=Data._reader;original_source=SavedColumnFeatureSource.__init__
    def prohibited(*args,**kwargs):raise AssertionError('native consumer reopened source ancestry')
    Data._reader=prohibited;SavedColumnFeatureSource.__init__=prohibited
    consumer_started=time.monotonic();cells_count=valid_count=training_rows=0
    try:
        with library.open_feature_view(result['feature_view']) as view, \
                load_feature_library_training_view(result['training_view'],limits=limits) as handle:
            cap=view.describe();securities=cap['security_ids'];count=len(securities)
            # Pick a pair with real eligible members on the last day; this is
            # a consumer subset, never a narrower feature preparation universe.
            final=view.project([(s,cfg['output_sessions'][-1]) for s in securities])
            valid_by_column=final['validity'].sum(axis=0)
            chosen=sorted(range(401),key=lambda i:(-int(valid_by_column[i]),i))[:2]
            subset=[selection[i] for i in chosen];names=[s['id'] for s in subset]
            for ordinal,day in enumerate(cfg['output_sessions']):
                clear_feature_window(handle)
                offsets=list(range(ordinal*count,(ordinal+1)*count))
                original=view.project([(s,day) for s in securities])
                available=original['available_at_utc_us'];valid=original['validity']
                assert not bool((valid & (available>instant_us(day+'T20:30:00+08:00'))).any())
                assert not bool((valid & (available==original['null_available_at_utc_us'])).any())
                cells_count+=int(valid.size);valid_count+=int(valid.sum())
                members,complete,knowledge,maximum=feature_eligibility_columns(handle,offsets,model_feature_selection=subset)
                expected=view.project([(s,day) for s in securities],columns=names)
                assert np.array_equal(complete,expected['validity'].all(axis=1))
                eligible=np.flatnonzero(members & complete).tolist()
                if eligible:
                    selected_offsets=[offsets[i] for i in eligible]
                    actual=training_matrix(handle,selected_offsets,day+'T20:30:00+08:00',model_feature_selection=subset)
                    assert np.array_equal(actual,expected['values'][eligible])
                    proofs=feature_training_blocks(handle,selected_offsets,model_feature_selection=subset)
                    assert proofs and all(p['ordered_features']==names for p in proofs)
                    training_rows+=len(eligible)
                    if ordinal==7:
                        # Boundary controls use the existing compact Label
                        # owner against real Feature membership and clocks.
                        cutoff=day+'T20:30:00+08:00';clock=instant_us(cutoff)
                        arrays={'validity':np.ones(4,dtype='u1'),'reason_codes':np.zeros(4,dtype='<i4'),
                                'end_session':np.array([0,0,0,-1],dtype='<i8'),
                                'availability_validity':np.ones(4,dtype='u1'),
                                'availability':np.array([clock-1,clock,clock+1,clock-1],dtype='<i8')}
                        raw=ConcatRows([TargetRows({'row_count':4,'definition':{'calendar':[day]},
                            'reason_dictionary':[None]},arrays)])
                        reasons=target_eligibility_reasons(raw,handle,[selected_offsets[0]]*4,cutoff,
                            {'target_spec':{'label_spec':{'maturity':{'rule':'all_outcome_dependencies_strictly_before_fit_cutoff'}}}},
                            model_feature_selection=subset)
                        assert reasons==[None,'LABEL_NOT_MATURE','LABEL_NOT_MATURE','LABEL_CLOCK_OR_ENDPOINT_UNKNOWN']
            assert training_rows>0 and cells_count==8*3396*401
            consumer_metrics=handle.metrics
    finally:
        Data._reader=original_reader;SavedColumnFeatureSource.__init__=original_source
    staging=[str(p) for p in (root/'inputs').rglob('.feature-input-*')
             if p.name!='.feature-input-checkpoint.json']
    assert not staging,staging[:5]
    manifest=json.loads(Path(result['source']['path']).read_text())
    receipts=[e for c in manifest['chunks'] for e in c['evidence']]
    receipts.extend(e for ext in manifest.get('daily_extensions',[]) for c in ext['chunks'] for e in c['evidence'])
    proof_bytes=0
    for desc in receipts:
        path=Path(desc['path']);proof_bytes+=path.stat().st_size
        proof=json.loads(path.read_text())
        assert proof['contract_version']=='reader_column_block_v1'
        assert all('by_key' not in meta for meta in proof['field_meta'].values())
    assert result['input_reuse']['buffer_copy_bytes']==0 and result['input_reuse']['old_value_recomputations']==0
    return {'status':'PASS','run':result,'performance':performance,'warm_overlap':warm,'requested_normalized_subset':norm,
        'consumer_seconds':time.monotonic()-consumer_started,'consumer_metrics':consumer_metrics,
        'physical_cells':cells_count,'valid_cells':valid_count,'subset_training_rows':training_rows,
        'unknown_membership_cells_preserved':unknown,'unpublished_staging_files':staging,
        'native_consumer_ancestor_calls':0,'input_block_receipt_bytes':proof_bytes,
        'input_block_receipt_count':len(receipts),'saved_cell_proof_rows':0,
        'coverage_owner_metrics':library.coverage_metrics,'label_boundary_controls':4,
        'label_scope':'maturity controls only; no real Label generation/model fit/account replay',
        'footprint':footprint(root),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'full_history_capacity_or_12h_acceptance':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    root=Path(cfg['destination']);root.mkdir(parents=True,exist_ok=True)
    with (root/'e2e-started.json').open('x') as stream:
        json.dump({'config':str(args.config.resolve()),'started_unix':time.time(),'attempt':1},stream)
    started=time.monotonic()
    try:receipt=acceptance(cfg)
    except BaseException as exc:
        receipt={'status':'FAIL','error':type(exc).__name__,'message':str(exc),
                 'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,
                 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'footprint':footprint(root),
                 'retry_performed':False}
        (root/'e2e-result.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
        raise
    receipt['elapsed_seconds']=time.monotonic()-started
    (root/'e2e-result.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
