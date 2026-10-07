"""Test-only constructor for frozen v2 saved-artifact compatibility fixtures.

Copied from Research 486ef862. Production prepare now has one v3 path.
This constructor exercises the historical codec/reader on synthetic test wires;
it is not exported, packaged, or used by any runtime/orchestrator. Both versions
still call the one Research Raw operator and the existing Core.
"""
from array import array
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import errno
import json
import struct
import sys
import tempfile
import time
from axiom_research.stock_artifacts import digest, file_digest, write_json, _read
from axiom_research.stock_fold_inputs import require, seal, validate_spec, file_fingerprint, read_parent
from axiom_research.stock_label_contracts import (_eligible_reason, _instant, NORMALIZATION_SPEC,
    RAW_TARGET_SCHEMA as RAW_SCHEMA, NORMALIZED_TARGET_SCHEMA as NORMALIZED_SCHEMA)
from axiom_research.stock_matrix_storage import (write_buffer, write_part, write_partition,
                                   instant_us)
from axiom_research.stock_matrix_prepare import (_options, _Publisher, _V1FeatureAccess, _v1_partitions, _canonical_buffers, _readonly, _clock_text, _graph_bytes, _price_workgroups, _label_partition, _query_for, _request_ranges, _label_work_plan, _source_contents, _selector_payload, _selector, CORE_TYPES)




def prepare_saved_v2_fixture(data, *, feature_inputs, fold_specs, destination,
                                 preparation_options, metrics=None, progress=None):
    """Prepare saved targets through existing public Data and Core exactly once.

    Date blocks are outermost; related fit cutoffs are consecutive within each
    physical working range. cache0 stays the caller's profile. No supplier,
    Feature execution, model training, predictions or account is performed.
    """
    from axiom_research.stock_feature_inputs import load_stock_feature_inputs
    from axiom_research.stock_matrix_reader import load_feature_matrix_index,VerifiedMatrixStore
    from axiom_research.stock_ml import _implementation, _environment
    from axiom_research.labels import build_forward_labels
    from axiom_data import adjust_prices
    from axiom_engine.core import execute_cs_zscore_batch
    begin=time.perf_counter(); options=_options(preparation_options); read_work=False; publisher=None
    limits={'maximum_matrix_bytes':options['maximum_resident_bytes'],
        'maximum_source_bytes':options.get('maximum_source_bytes',8*1024**3),
        'maximum_parent_bytes':options.get('maximum_parent_bytes',64*1024**2)}
    path=feature_inputs.path if hasattr(feature_inputs,'path') else Path(feature_inputs)
    # A public object is a locator, never a trusted skip-validation marker.
    locator=_read(Path(path)/'index.json')
    feature=(load_feature_matrix_index(path,store=VerifiedMatrixStore(**limits)) if locator.get('contract_version') in ('stock_feature_inputs_v2','stock_feature_inputs_v3')
             else _V1FeatureAccess(load_stock_feature_inputs(path)))
    try:
        index=feature.to_dict(); spec=index['definition']['spec']; row_index=feature._row_index
        require(type(fold_specs) is list and bool(fold_specs),'nonempty ordered fold specs required')
        folds=deepcopy(fold_specs); requests=[]; states={}; previous=None
        for fold in folds:
            training,inference=validate_spec(fold,spec['calendar']); ref=digest(fold)
            require(ref not in states and (previous is None or fold['oos_trade_sessions'][0]>previous),
                    'ordered disjoint OOS folds required')
            previous=fold['oos_trade_sessions'][-1]
            require(set(training+inference)<=set(row_index['sessions']),'Feature input does not cover fold')
            states[ref]={'spec':fold,'training':training,'inference':inference,'chunks':[],
                'arrays':{name:array('d' if name=='values' else 'B' if name in
                    ('value_validity','reference_member') else 'i' if name.endswith('codes') else 'q')
                    for name in list(CORE_TYPES)[:9]},'sources':{},'reasons':[],
                'training_offsets':[],'core_refs':[]}
            requests.extend([(ref,'training',training,fold['fit_cutoff']),
                             (ref,'evaluation',inference,fold['evaluation_cutoff'])])
        # Account every fit, refreshing only the state that changed. Independent
        # per-fit graphs conservatively count shared small strings twice; they
        # avoid repeatedly walking all retained source descriptors per block.
        state_sizes={ref:_graph_bytes(state) for ref,state in states.items()}
        state_mapping_bytes=sys.getsizeof(states)+sys.getsizeof(state_sizes)+sum(sys.getsizeof(ref) for ref in states)
        def retained_fit_bytes(ref):
            state_sizes[ref]=_graph_bytes(states[ref])
            return state_mapping_bytes+sum(state_sizes.values())+sum(sys.getsizeof(size) for size in state_sizes.values())
        implementations=_implementation(); definition={'version':'axiom.stock_ml_batch_inputs/2',
            'feature_inputs':feature.descriptor,'fold_specs':folds,'preparation_options':options,
            'implementation_sources':implementations,'implementation_ref':digest(implementations),
            'environment':_environment()}
        definition_ref=digest(definition); target=(Path(destination)/definition_ref[7:]).resolve()
        if target.exists():
            saved=_read(target/'batch.json')
            require(saved['definition']==definition,'cached matrix batch definition mismatch')
            from axiom_research.stock_batch import load_stock_ml_batch_inputs
            with load_stock_ml_batch_inputs(saved,limits=limits): pass
            if metrics is not None: metrics.update(cache_hit=True,data_read_calls=0,core_calls=0,
                feature_core_calls=0,train_calls=0,predict_calls=0,account_calls=0,total_seconds=time.perf_counter()-begin)
            return saved
        target.parent.mkdir(parents=True,exist_ok=True)
        stats={'cache_hit':False,'data_read_calls':0,'supplier_calls':0,'feature_core_calls':0,
            'label_core_calls':0,'core_calls':0,'train_calls':0,'predict_calls':0,'account_calls':0,
            'date_working_groups':0,'normalization_seconds':0.0,'data_cache_profile':'caller_unchanged',
            'physical_partition_reads':None,'physical_index_hits':None,
            'physical_metrics_reason':'public Data API does not expose physical Reader counters'}
        with tempfile.TemporaryDirectory(prefix='.stock-matrix-',dir=target.parent) as temporary:
            stage=Path(temporary)/'complete'; stage.mkdir()
            publisher=_Publisher(stage,target,maximum_resident_bytes=options['maximum_resident_bytes'],
                maximum_source_bytes=limits['maximum_source_bytes'],maximum_parent_bytes=limits['maximum_parent_bytes'],
                known_source_bytes=getattr(feature,'metrics',{}).get('source_bytes',0),metrics=stats)
            if index['contract_version'] in ('stock_feature_inputs_v2','stock_feature_inputs_v3'):
                partitions=list(index['partitions']); feature_selection=_read(index['source_selection']['path'])['feature_rows']
                feature_schema=index['schema']; row_desc=index['row_index']
            else:
                partitions,feature_selection,feature_schema,row_desc=_v1_partitions(feature,publisher,options)
            label_selection=[]; core_wrappers=[]
            needed=sorted({d for _,_,days,_ in requests for d in days}); width=len(spec['universe'])
            positions={d:i for i,d in enumerate(row_index['sessions'])}
            for start in range(0,len(needed),options['row_block_sessions']):
                working=needed[start:start+options['row_block_sessions']]; stats['date_working_groups']+=1
                active=[]
                for ref,role,cutoff,group in _request_ranges(requests,working,positions):
                    query,anchor=_query_for(spec,cutoff,group)
                    active.append((ref,role,cutoff,group,query,anchor))
                # Fixed symbols/columns across fits. Market is read for all
                # related cutoffs before factor reads advance this working set.
                read_work=True
                for group,retained in _price_workgroups(data,active,snapshot=spec["snapshot"],
                        budget=options["maximum_resident_bytes"],stats=stats):
                    for request,price in group:
                        ref,role,cutoff,days,query,anchor=request
                        factors=data.read(snapshot=spec['snapshot'],query=replace(query,domain='adjustment_factors',fields=('factor',)))
                        stats['data_read_calls']+=1
                        adjusted=adjust_prices(price,factors,fields=('open','close'),anchor_session=anchor,
                                                decision_session=anchor,factor_field='factor')
                        wire=adjusted.to_json(); raw=build_forward_labels(adjusted,calendar=spec['calendar'],feature_sessions=days)
                        require(retained+_graph_bytes(wire)+_graph_bytes(raw)<=options['maximum_resident_bytes'],
                                'adjusted Label working set resident byte budget exceeded')
                        publisher.live_bytes=retained+retained_fit_bytes(ref)+_graph_bytes([
                            wire,raw,factors.field_meta,factors.context,adjusted.field_meta,adjusted.context])
                        publisher.live_bytes+=int(factors.frame.memory_usage(deep=True).sum())+int(
                            adjusted.frame.memory_usage(deep=True).sum())
                        raw_desc=publisher.part(raw,'label_ref'); rows=raw['rows']; offset=positions[days[0]]*width
                        cohort=None
                        if role=='training':
                            offsets=range(offset,offset+len(rows)); features=feature.row_metadata(offsets)
                            reasons=[_eligible_reason(r,f,len(spec['ordered_features']),_instant(cutoff)) for r,f in zip(rows,features)]
                            for reason,f in zip(reasons,features):
                                if reason is None:
                                    require(all(a is None or _instant(a)<=_instant(cutoff) for a in f['availability']) and
                                        any(a is not None for a in f['availability']),'training Feature clock exceeds fit')
                            cohort={'contract_version':'stock_matrix_label_cohort_v1','feature_inputs_ref':feature.identity,
                                'raw_label_ref':raw['label_ref'],'fold_spec_ref':ref,'cutoff':cutoff,
                                'sessions':days,'keys':[[r['security_id'],r['feature_session']] for r in rows],
                                'eligibility_reasons':reasons,'eligible_keys':[[r['security_id'],r['feature_session']]
                                    for r,reason in zip(rows,reasons) if reason is None]}
                        publisher.live_bytes+=_graph_bytes(cohort)
                        contents,qref,vref,cref=_source_contents(wire,raw,cohort,digest_fn=publisher.native_digest)
                        raw_partition=_label_partition(publisher,table='training_raw_labels' if role=='training'
                            else 'evaluation_raw_labels',spec_ref=ref,index_ref=row_index['row_index_ref'],offset=offset,
                            rows=rows,raw_desc=raw_desc,cutoff=cutoff,contents=contents)
                        partitions.append(raw_partition)
                        label_selection.append({'role':role,'fold_spec_ref':ref,'cutoff':cutoff,'sessions':days,
                            'adjustment_anchor':anchor,'query_refs':[qref],'selected_versions_ref':vref,'cohort_ref':cref})
                        if role=='training':
                            state=states[ref]
                            state['chunks'].append((offset,len(rows),raw_desc,cutoff,raw_partition['metadata']))
                        retained_states=retained_fit_bytes(ref)
                        stats['peak_retained_fit_state_bytes']=max(stats.get('peak_retained_fit_state_bytes',0),retained_states)
                        require(retained+_graph_bytes(wire)+_graph_bytes(raw)+retained_states<=options['maximum_resident_bytes'],
                                'retained fit states exceed resident byte budget')
                        del wire,raw,rows,contents,adjusted,factors,price
                        publisher.live_bytes=0
                    group.clear()
                if progress is not None: progress({'stage':'raw_labels','completed':min(start+len(working),len(needed)),
                    'total':len(needed),'logical_query_count':stats['data_read_calls'],'seconds':time.perf_counter()-begin})
            for ref,state in states.items():
                arrays=state['arrays']; normalized_chunks=[]
                # Rehydrate only this fit's packed input. All other fit source
                # graphs remain on disk; no fold-count multiplier of raw panels.
                for offset,count,raw_desc,cutoff,metadata_desc in state['chunks']:
                    publisher.live_bytes=retained_fit_bytes(ref)
                    raw=publisher.read_part(raw_desc,'label_ref'); metadata=publisher.read_part(metadata_desc,'metadata_ref')
                    cohort=next(v for v in metadata['contents'].values() if type(v) is dict and
                                v.get('contract_version')=='stock_matrix_label_cohort_v1')
                    cohort_ref=digest(cohort)
                    # Rehydrate source bindings only for the active fit. Their
                    # exact Raw/cohort refs are unchanged by this lifetime.
                    for day in cohort['sessions']:
                        state['sources'][day]={'bindings':[
                            {'id':'offline_eligibility','data_ref':feature.identity,'view_ref':cohort_ref,
                             'revision_policy':'frozen_feature_membership_and_explicit_outcome_cutoff',
                             'qualification':'observed','availability_basis':'derived_offline_cutoff_selection'},
                            {'id':'raw_labels','data_ref':raw['label_ref'],'view_ref':raw['label_ref'],
                             'revision_policy':'frozen_saved_label_build','qualification':'observed',
                             'availability_basis':'exact_raw_label_available_at'}],
                            'source_sets':[['offline_eligibility'],['raw_labels']]}
                    core_start=len(arrays['values'])
                    arrays['selection_cutoff_utc_us'].extend(instant_us(cutoff) for d in cohort['sessions'])
                    for row,reason in zip(raw['rows'],cohort['eligibility_reasons']):
                        valid=reason is None; arrays['values'].append(float(row['return']) if valid else 0.0)
                        arrays['value_validity'].append(valid); arrays['value_reason_codes'].append(0)
                        arrays['fact_available_at_utc_us'].append(instant_us(row['label_available_at'] if valid else cutoff))
                        arrays['reference_member'].append(valid); arrays['reference_available_at_utc_us'].append(instant_us(cutoff))
                        arrays['fact_source_codes'].append(1 if valid else 0); arrays['reference_source_codes'].append(0)
                    state['reasons'].extend(cohort['eligibility_reasons'])
                    normalized_chunks.append((offset,count,raw_desc,cutoff,metadata_desc,core_start))
                    retained_states=retained_fit_bytes(ref)
                    stats['peak_retained_fit_state_bytes']=max(stats.get('peak_retained_fit_state_bytes',0),retained_states)
                    require(retained_states+_graph_bytes(raw)+_graph_bytes(metadata)<=options['maximum_resident_bytes'],
                            'Core input resident byte budget exceeded')
                    publisher.release_part(raw_desc,'label_ref'); publisher.release_part(metadata_desc,'metadata_ref')
                    del raw,metadata,cohort
                    publisher.live_bytes=0
                reasons=[None,*sorted({r for r in state['reasons'] if r is not None})]
                codes={r:i for i,r in enumerate(reasons)}
                arrays['value_reason_codes']=array('i',(codes[r] for r in state['reasons']))
                native=sum(len(values)*values.itemsize for values in arrays.values())
                # Check before the readonly encoding also allocates copies.
                core_budget=retained_fit_bytes(ref)+3*native+len(arrays['values'])*256
                require(core_budget<=options['maximum_resident_bytes'],'Core working set resident byte budget exceeded')
                carrier={'contract_version':'core_cs_zscore_batch_input_v1',
                    'calendar_ref':digest({'contract_version':'stock_label_calendar_v1','sessions':spec['calendar']}),
                    'schema':RAW_SCHEMA,'output_schema':NORMALIZED_SCHEMA,'sessions':state['training'],
                    'security_ids':spec['universe'],'reason_dictionary':reasons,
                    'source_bindings_by_session':state['sources'],
                    **{name:_readonly(values,name) for name,values in arrays.items()}}
                # Core captures nine buffers, then allocates output lists and
                # five owned outputs. Bound these simultaneous copies before
                # entering its public mathematical executor.
                input_doc,input_buffers=_canonical_buffers(carrier,publisher,arrays)
                core_input=publisher.part(seal({'input':input_doc,'buffers':input_buffers},
                    'core_input_artifact_ref'),'core_input_artifact_ref')
                del input_doc,input_buffers
                clock=time.perf_counter(); result=execute_cs_zscore_batch(carrier,params=NORMALIZATION_SPEC['params'])
                stats['normalization_seconds']+=time.perf_counter()-clock; stats['core_calls']+=1; stats['label_core_calls']+=1
                output_names=['values','value_validity','value_reason_codes','available_at_utc_us','source_codes']
                canonical,physical=_canonical_buffers(result,publisher,output_names)
                wrapper=seal({'contract_version':'stock_matrix_core_result_v1','result':canonical,
                    'buffers':physical},'core_result_artifact_ref')
                core_wrappers.append(publisher.part(wrapper,'core_result_artifact_ref'))
                core_ref=result['metadata']['result_ref']; state['core_refs']=[core_ref]
                for offset,count,raw_desc,cutoff,metadata_desc,core_start in normalized_chunks:
                    publisher.live_bytes=retained_fit_bytes(ref)+_graph_bytes([carrier,result,normalized_chunks,
                        canonical,physical,wrapper,core_input,reasons,codes])
                    raw=publisher.read_part(raw_desc,'label_ref'); metadata=publisher.read_part(metadata_desc,'metadata_ref')
                    contents=metadata['contents']; normalized=[]
                    for local,row in enumerate(raw['rows']):
                        i=core_start+local; valid=bool(result['value_validity'][i]); reason=state['reasons'][i]
                        normalized.append({**row,'normalized_return':float(result['values'][i]) if valid else None,
                            'normalized_valid':valid,'normalized_available_at':_clock_text(result['available_at_utc_us'][i]) if valid else None,
                            'normalization_reason':None if valid else reason or 'NORMALIZATION_UNDEFINED',
                            'normalization_source_refs':sorted(set(row['source_refs']+[raw['label_ref'],feature.identity,
                                state['sources'][row['feature_session']]['bindings'][0]['view_ref'],core_ref]))})
                        if valid: state['training_offsets'].append(offset+local)
                    contents={**contents,digest(core_input):core_input}
                    publisher.live_bytes+=_graph_bytes(normalized)
                    partitions.append(_label_partition(publisher,table='training_normalized_labels',spec_ref=ref,
                        index_ref=row_index['row_index_ref'],offset=offset,rows=normalized,raw_desc=raw_desc,
                        cutoff=cutoff,contents=contents,core_refs=[core_ref]))
                    publisher.release_part(raw_desc,'label_ref'); publisher.release_part(metadata_desc,'metadata_ref')
                    del raw,metadata,contents,normalized
                    publisher.live_bytes=0
                # Write selector bytes now; only their small descriptors wait
                # for the prepared view identity. Do not retain prior fits'
                # Python offset lists, sources or chunk graphs.
                state['selector_payloads']={
                    'training':_selector_payload(publisher,row_index=row_index,offsets=state['training_offsets']),
                    'inference':_selector_payload(publisher,row_index=row_index,
                        offsets=array('Q',(positions[d]*width+i for d in state['inference'] for i in range(width))))}
                for name in ('arrays','reasons','sources','chunks','training_offsets'): state[name].clear()
                retained_fit_bytes(ref)
                stats['maximum_completed_fit_working_bytes']=max(stats.get('maximum_completed_fit_working_bytes',0),
                    sum(_graph_bytes(state[name])-sys.getsizeof(state[name]) for name in
                        ('arrays','reasons','sources','chunks','training_offsets')))
                del carrier,result,normalized_chunks,canonical,physical,wrapper,core_input,reasons,codes
            selection=seal({'contract_version':'stock_matrix_source_selection_v2' if index['contract_version']=='stock_feature_inputs_v3' else 'stock_matrix_source_selection_v1','feature_inputs_ref':feature.identity,
                'feature_rows':feature_selection,'label_rows':label_selection},'source_selection_ref')
            schema={'features':feature_schema,'training_raw_labels':RAW_SCHEMA,
                'training_normalized_labels':NORMALIZED_SCHEMA,'evaluation_raw_labels':RAW_SCHEMA}
            view_definition={k:deepcopy(spec[k]) for k in ('scope','snapshot','pit_policy','calendar','universe',
                'catalog_ref','feature_selection','ordered_features')}
            view_definition.update({k:deepcopy(definition[k]) for k in ('feature_inputs','fold_specs',
                'preparation_options','implementation_sources','implementation_ref','environment')})
            view=seal({'contract_version':'stock_ml_prepared_view_v1','definition':view_definition,
                'definition_ref':digest(view_definition),'schema':schema,'schema_digest':digest(schema),
                'row_index':row_desc,'source_selection':publisher.part(selection,'source_selection_ref'),
                'partitions':partitions,'core_results':core_wrappers},'prepared_view_ref')
            view_desc=publisher.part(view,'prepared_view_ref'); outputs=[]
            for ref,state in states.items():
                selectors={role:None if role=='validation' else _selector(publisher,view_ref=view['prepared_view_ref'],
                    row_index=row_index,schema_ref=view['schema_digest'],role=role,fold_ref=ref,
                    payload=state['selector_payloads']['training' if role in ('training','training_labels') else 'inference'])
                    for role in ('training','validation','inference','training_labels','evaluation_labels')}
                inputs=seal({'contract_version':'stock_ml_saved_inputs_v2','prepared_view':view_desc,
                    'fold_spec_ref':ref,'selectors':selectors,'core_result_refs':state['core_refs']},'input_ref')
                outputs.append({'input_manifest':inputs,'fold_spec':state['spec']})
            batch={'contract_version':'stock_ml_batch_inputs_v2','definition':definition,'definition_ref':definition_ref,
                'prepared_view':view_desc,'folds':outputs,'status':'COMPLETE'}
            batch['batch_ref']=digest(batch); batch=seal(batch,'content_digest'); write_json(stage/'batch.json',batch)
            publisher._desc({'path':str((stage/'batch.json').resolve())})
            from axiom_research.stock_matrix_reader import _validate_staged_matrix_batch
            stats['staged_validation']=_validate_staged_matrix_batch(batch,stage=stage,target=target,
                **limits)
            feature._store.check()
            publisher.finish()
            try: stage.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
                require(_read(target/'batch.json')==batch,'concurrent prepared batch conflict')
                # Equal manifest bytes do not prove a concurrent publisher's
                # children exist or retain their digests. This uncommon path
                # must admit the actual winner, rather than trust our stage.
                from axiom_research.stock_batch import load_stock_ml_batch_inputs
                with load_stock_ml_batch_inputs(batch,limits=limits): pass
        # The complete closure was admitted before rename. Only exact staged
        # bytes are published; repeat full disk admission belongs to the next
        # explicit public batch load or to a future HIT.
        require(_read(target/'batch.json')==batch,'published prepared batch changed')
        stats.update(total_seconds=time.perf_counter()-begin,prepared_view_ref=view['prepared_view_ref'],
                     batch_ref=batch['batch_ref'],logical_query_count=stats['data_read_calls'])
        if metrics is not None: metrics.update(stats)
        return batch
    finally:
        if publisher is not None and not publisher.store.closed: publisher.store.close()
        feature.close()
        if read_work and callable(getattr(data,'clear_cache',None)): data.clear_cache()
