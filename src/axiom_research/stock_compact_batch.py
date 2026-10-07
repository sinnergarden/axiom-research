"""Ordinary compact target admission and the existing fold projection ABI."""
from copy import deepcopy
from pathlib import Path
import math

from .stock_artifacts import digest
from .stock_fold_inputs import require, seal, validate_spec, feature_available
from .stock_label_contracts import _instant, NORMALIZATION_SPEC
from .stock_compact_store import (OwnedStore, _view_data, load_stock_feature_view,
    feature_rows, training_matrix, sealed, fields, limits, _size, reference)


def _clock(us):
    from .stock_compact_labels import _clock
    return _clock(us)


class TargetRows:
    """Virtual target rows over owned bytes; no retained Python Raw panels."""
    def __init__(self,value,arrays,raw_rows=None):
        self.value=value; self.arrays=arrays; self.raw_rows=raw_rows
    def __len__(self): return self.value['row_count']
    def __iter__(self):
        for i in range(len(self)): yield self[i]
    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(len(self)))]
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        value=self.value; definition=value['definition']; a=self.arrays; calendar=definition['calendar']
        universe=definition['universe']; valid=bool(a['validity'][index]); code=int(a['reason_codes'][index])
        normalized=value['contract_version']=='stock_compact_normalized_v1'
        if normalized:
            require(self.raw_rows is not None,'normalized target requires its admitted Raw table')
            row=self.raw_rows[index]
            row.update(raw_return=row['return'],raw_available_at=row['label_available_at'])
        else:
            start,end=int(a['start_session'][index]),int(a['end_session'][index])
            row={'security_id':universe[index%len(universe)],'feature_session':definition['sessions'][index//len(universe)],
                'start_session':calendar[start] if start>=0 else None,'end_session':calendar[end] if end>=0 else None,
                'source_refs':deepcopy(value['source_dictionary'][int(a['source_codes'][index])])}
        row.update(**{'return':float(a['values'][index]) if valid else None},valid=valid,
            label_available_at=_clock(a['availability'][index]) if a['availability_validity'][index] else None,
            invalid_reason=value['reason_dictionary'][code])
        return row


class ConcatRows:
    def __init__(self,parts): self.parts=parts; self.count=sum(len(p) for p in parts)
    def __len__(self): return self.count
    def __iter__(self):
        for part in self.parts: yield from part
    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(len(self)))]
        for part in self.parts:
            if index<len(part): return part[index]
            index-=len(part)
        raise IndexError(index)


def read_target(store,descriptor,expected=None,raw_rows=None):
    value=store.read_json(descriptor,key='target_ref'); definition=value['definition']
    fields(value,{'contract_version','definition','definition_ref','row_count','reason_dictionary',
        'source_dictionary','buffers','core_ref','cohort','target_ref'},'exact compact target required')
    require(value['contract_version'] in ('stock_compact_raw_v1','stock_compact_normalized_v1') and
            value['definition_ref']==digest(definition) and (expected is None or definition==expected),
            'compact target definition mismatch')
    if 'target_ref' in descriptor: require(value['target_ref']==descriptor['target_ref'],'compact target reference mismatch')
    days=definition['sessions']; universe=definition['universe']; count=len(days)*len(universe)
    require(days==sorted(set(days)) and universe==sorted(set(universe)) and value['row_count']==count,
            'compact target grid mismatch')
    root=Path(descriptor['path']).parent.resolve()
    for item in value['buffers'].values():
        require(Path(item['path']).resolve().is_relative_to(root),'compact target buffer escaped its owner directory')
    raw={'values','validity','availability','availability_validity','reason_codes'}
    normalized=value['contract_version']=='stock_compact_normalized_v1'
    expected_types={'values':'float64_le','validity':'bool_u8','availability':'int64_le',
        'availability_validity':'bool_u8','reason_codes':'int32_le'}
    if not normalized: expected_types.update(start_session='int32_le',end_session='int32_le',source_codes='int32_le')
    require(set(value['buffers'])==set(expected_types) and
        all(v['dtype']==expected_types[k] for k,v in value['buffers'].items()),'compact target physical dtypes mismatch')
    arrays={k:store.buffer(v) for k,v in value['buffers'].items()}
    require(set(arrays)==raw|(set() if normalized else {'start_session','end_session','source_codes'}) and
            all(a.shape==(count,) for a in arrays.values()),'compact target columns/shapes mismatch')
    dictionary=value['reason_dictionary']; require(dictionary[0] is None and dictionary[1:]==sorted(set(dictionary[1:])),
                                                 'compact target reason dictionary mismatch')
    calendar=definition['calendar']
    require(type(calendar) is list and calendar==sorted(set(calendar)) and set(days)<=set(calendar) and
            type(value['row_count']) is int and count>0,'compact calendar/row count mismatch')
    if normalized:
        require(reference(value['core_ref']) and type(value['cohort']) is dict and value['source_dictionary']==[],
                'normalized Core/cohort refs required')
    else:
        require(value['core_ref'] is None and value['cohort'] is None and type(value['source_dictionary']) is list and
            bool(value['source_dictionary']) and all(type(s) is list and bool(s) and all(reference(r) for r in s)
                for s in value['source_dictionary']), 'Raw fixed source refs required')
    import numpy as np
    require(bool(np.isfinite(arrays['values']).all()),'compact target physical values must be finite')
    positions={day:i for i,day in enumerate(calendar)}
    from .stock_label_contracts import core_clock
    from .stock_matrix_storage import instant_us
    cutoff=instant_us(core_clock(definition['cutoff']) if normalized else definition['cutoff'])
    cutoff_day=_instant(definition['cutoff']).date().isoformat()
    for i in range(count):
        code=int(arrays['reason_codes'][i]); valid=bool(arrays['validity'][i]); at=bool(arrays['availability_validity'][i])
        require(0<=code<len(dictionary) and valid is (code==0) and (not valid or at),
                'compact target validity/reason/clock mismatch')
        if not normalized:
            start,end=int(arrays['start_session'][i]),int(arrays['end_session'][i])
            pos=positions[days[i//len(universe)]]
            require(start==(pos+1 if pos+1<len(calendar) else -1) and end==(pos+5 if pos+5<len(calendar) else -1) and
                    (not valid or start>=0 and end>=0 and calendar[end]<=cutoff_day),
                    'compact target endpoint offset mismatch')
            require(0<=int(arrays['source_codes'][i])<len(value['source_dictionary']), 'compact target source code mismatch')
        if not valid: require(float(arrays['values'][i])==0.0,'compact null physical value must be zero')
        if not at: require(int(arrays['availability'][i])==0,'compact null physical clock must be zero')
        require(not at or int(arrays['availability'][i])<=cutoff, 'compact target clock exceeds cutoff')
    require(raw_rows is None or len(raw_rows)==count,'normalized Raw key alignment mismatch')
    store.check(); return value,TargetRows(value,arrays,raw_rows)


class CompactState:
    def __init__(self,store,feature,batch,view,targets,own_feature):
        self.store=store; self.feature=feature; self.batch=batch; self.view=view; self.targets=targets
        self.closed=False; self.own_feature=own_feature; self.active=0; self._batch_ref=batch['batch_ref']
        self.selectors={(digest(f['input_manifest']),digest(f['fold_spec'])):f['input_manifest']['selectors'] for f in batch['folds']}
        _view_data(feature)['borrowers']+=1
        self.source_records=tuple(sorted({**_view_data(feature)['store'].hashes,**store.hashes}.items()))

    def check(self):
        require(not self.closed,'compact batch is closed'); self.store.check(); _view_data(self.feature)

    def _account_resident(self,extra_roots=()):
        self.store.reserve(0); return self.store.shared_bytes+self.store.resident_bytes

    def compatible(self,budgets):
        fd=_view_data(self.feature); fstore=fd['store']
        require(fstore.metrics['source_bytes']+self.store.metrics['source_bytes']<=budgets['maximum_source_bytes'] and
                self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes<=budgets['maximum_matrix_bytes'] and
                max(fstore.metrics['largest_parent_bytes'],self.store.metrics['largest_parent_bytes'])<=budgets['maximum_parent_bytes'],
                'admitted compact accounting exceeds requested budget')
        self.store.limits={k:min(v,budgets[k]) for k,v in self.store.limits.items()}

    def close(self):
        require(self.active==0 and self.store.borrowers==0,'compact batch backing still borrowed')
        if self.closed: return
        self.store.close(); self.closed=True; fd=_view_data(self.feature,check=False); fd['borrowers']-=1
        if self.own_feature: self.feature.close()

    def _parts(self,inputs,spec):
        self.check(); key=digest(inputs),digest(spec)
        require(key in self.selectors,'fold outside compact batch definition')
        record=next(r for r in self.view['fold_targets'] if digest(r['fold_spec'])==digest(spec))
        require(inputs['selectors']=={k:None if k=='validation' else record['training_offsets' if k in
            ('training','training_labels') else 'inference_offsets'] for k in inputs['selectors']},
            'compact fold selectors mismatch')
        return record

    def _feature_slice(self,inputs,spec,rows):
        common=self.view['definition']; training,inference=validate_spec(spec,common['calendar'])
        fd=_view_data(self.feature)
        parents={d:deepcopy(fd['parents'][d]) for d in sorted(set(training+inference))}
        return seal({'contract_version':'stock_feature_slice_v3','input_manifest_ref':digest(inputs),
            'prepared_view_ref':self.view['prepared_view_ref'],'selectors':deepcopy(inputs['selectors']),
            'universe':common['universe'],'ordered_features':common['ordered_features'],
            'catalog_ref':common['catalog_ref'],'selection':common['feature_selection'],
            'training_sessions':training,'prediction_sessions':inference,'parents_by_session':parents,'rows':rows},'feature_ref')

    def project(self,inputs,spec):
        return self._project(inputs,spec,training=True)

    def project_evaluation(self,inputs,spec,*,batch_ref):
        require(batch_ref==self._batch_ref,'compact batch identity mismatch')
        return self._project(inputs,spec,training=False)

    def evaluation(self,inputs,spec,*,batch_ref):
        raise ValueError('compact v3 has no legacy signal-evaluation proof projection; use explicit batch audit')

    def _project(self,inputs,spec,*,training):
        from .stock_matrix_reader import MatrixFoldProjection
        from .stock_label_contracts import TARGET_SEMANTICS
        import numpy as np
        record=self._parts(inputs,spec); common=deepcopy(self.view['definition']); normalized=self.targets[record['normalized']['target_ref']]
        self.store.reserve(len(record['inference_offsets'])*(len(common['ordered_features'])*96+2048))
        rows=feature_rows(self.feature,record['inference_offsets']); candidate_keys=[]; candidates=[]
        for row in rows:
            if row['member'] and all(row['validity']) and all(type(v) in (int,float) and math.isfinite(v) for v in row['values']) and feature_available(row) is not None:
                candidates.append(row['values']); candidate_keys.append([row['security_id'],row['session']])
        features=self._feature_slice(inputs,spec,rows); nrows=normalized[1]
        fspec=_view_data(self.feature)['definition']['spec']; width=len(common['universe'])
        keys=[[common['universe'][off%width],fspec['feature_sessions'][off//width]] for off in record['training_offsets']]
        raw_refs=sorted(d['target_ref'] for d in record['raw_parts'])
        excluded={}
        for code,flag in zip(nrows.arrays['reason_codes'],nrows.arrays['validity']):
            if not flag:
                reason=nrows.value['reason_dictionary'][int(code)]; excluded[reason]=excluded.get(reason,0)+1
        labels=seal({'contract_version':'stock_fold_label_slice_v3','input_manifest_ref':digest(inputs),
            'prepared_view_ref':self.view['prepared_view_ref'],'fold_spec_ref':digest(spec),'cutoff':spec['fit_cutoff'],
            'selectors':{k:deepcopy(inputs['selectors'][k]) for k in ('training_labels','evaluation_labels')},
            'core_result_refs':inputs['core_result_refs'],'raw_label_refs':raw_refs,
            'training_sessions':validate_spec(spec,common['calendar'])[0],'training_row_count':len(keys),'excluded':excluded},'label_ref')
        erows=list(self.targets[record['evaluation']['target_ref']][1])
        evaluation=seal({'contract_version':'stock_matrix_evaluation_target_slice_v1',
            'prepared_view_ref':self.view['prepared_view_ref'],'fold_spec_ref':digest(spec),
            'selector':deepcopy(inputs['selectors']['evaluation_labels']),'cutoff':spec['evaluation_cutoff'],'rows':erows},'label_ref')
        bindings=record['training_binding']; training_ref=bindings['training_rows_ref']
        payload={'common':common,'features':features,'labels':labels,'evaluation':evaluation,
            'training_keys':keys,'training_rows_ref':training_ref,'excluded':excluded,'raw_refs':raw_refs,
            'candidate_keys':candidate_keys,'feature_rows':rows,'feature_ref':features['feature_ref'],'label_ref':labels['label_ref']}
        if training:
            self.store.reserve((len(keys)+len(candidates))*len(common['ordered_features'])*8+len(keys)*24)
            X=training_matrix(self.feature,record['training_offsets'],spec['fit_cutoff'])
            y=nrows.arrays['values'][nrows.arrays['validity'].astype(bool)]
            P=np.asarray(candidates,dtype='<f8').reshape(len(candidates),len(common['ordered_features']))
            require(bool(np.isfinite(X).all()) and bool(np.isfinite(y).all()) and bool(np.isfinite(P).all()),'compact active matrix must be finite')
            for a in (X,y,P): a.flags.writeable=False
            payload.update(X=X,y=y,P=P)
            self.store.metrics['training_projection_calls']=self.store.metrics.get('training_projection_calls',0)+1
            self.store.metrics['matrix_allocated_bytes']=self.store.metrics.get('matrix_allocated_bytes',0)+sum(a.nbytes for a in (X,y,P))
        else:
            self.store.metrics['evaluation_projection_calls']=self.store.metrics.get('evaluation_projection_calls',0)+1
            dataset=seal({'contract_version':'stock_fold_dataset_v3','prepared_view_ref':self.view['prepared_view_ref'],
                'feature_ref':features['feature_ref'],'label_ref':labels['label_ref'],'raw_label_refs':raw_refs,
                'fold_spec_ref':digest(spec),'fit_cutoff':spec['fit_cutoff'],'ordered_features':common['ordered_features'],
                'selectors':inputs['selectors'],'training_keys_digest':digest(keys),'training_rows_ref':training_ref,
                'training_row_count':len(keys),'excluded':excluded,'target_semantics':TARGET_SEMANTICS,
                'normalization':NORMALIZATION_SPEC,'validation':'none_fixed_parameters_no_early_stopping'},'dataset_ref')
            payload['fold_binding']={'dataset':dataset,'labels':labels}
        lease=_size(payload,maximum=self.store.maximum_matrix_bytes,
            retained=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes)+sum(
            a.nbytes for k,a in payload.items() if k in ('X','y','P'))+4096
        self.store.reserve(lease); self.check()
        self.store.lease_bytes+=lease; self.store.metrics['fold_projection_calls']+=1
        return MatrixFoldProjection(self.store,**payload,_lease_bytes=lease)


def load_compact_state(manifest,*,feature_inputs=None,limits=None,resolver=None,publication_store=None):
    fields(manifest,{'contract_version','definition','definition_ref','prepared_view','folds','status','batch_ref','content_digest'},
           'exact compact batch manifest required')
    require(manifest['contract_version']=='stock_ml_batch_inputs_v3' and manifest['status']=='COMPLETE',
            'complete compact batch required')
    sealed(manifest,'content_digest'); require(manifest['batch_ref']==digest({k:v for k,v in manifest.items() if k not in
        ('batch_ref','content_digest')}) and manifest['definition_ref']==digest(manifest['definition']),'compact batch identity mismatch')
    budgets=globals()['limits'](limits); definition=manifest['definition']; expected=definition['feature_view']; own=feature_inputs is None
    feature=load_stock_feature_view(Path(expected['source_index']['path']).parent,limits=budgets) if own else feature_inputs
    fd=_view_data(feature); require(feature.to_dict()==expected,'compact Feature view exact identity/cutoff mismatch')
    require(fd['store'].resident_bytes<=budgets['maximum_matrix_bytes'] and
            fd['store'].metrics['source_bytes']<=budgets['maximum_source_bytes'] and
            fd['store'].metrics['largest_parent_bytes']<=budgets['maximum_parent_bytes'],
            'borrowed Feature budget incompatible')
    store=publication_store or OwnedStore(budgets,resolver,shared_bytes=fd['store'].resident_bytes,
                     shared_source_bytes=fd['store'].metrics['source_bytes'])
    require(type(store) is OwnedStore and not store.closed and store.borrowers==0,'active owner byte admission required')
    if publication_store is not None:
        store.shared_bytes=max(store.shared_bytes,fd['store'].resident_bytes)
        store.shared_source_bytes=max(store.shared_source_bytes,fd['store'].metrics['source_bytes'])
        store.reserve(0)
        require(store.shared_source_bytes+store.metrics['source_bytes']<=budgets['maximum_source_bytes'],
                'compact source byte budget exceeded')
    if resolver is not None: store.resolve=resolver
    store.check_hook=fd['store'].check
    state=None
    try:
        view=store.read_json(manifest['prepared_view'],key='prepared_view_ref')
        fields(view,{'contract_version','definition','feature_view','fold_targets','prepared_view_ref'},
               'exact compact prepared view required')
        require(view['contract_version']=='stock_ml_prepared_view_v2' and
            view['prepared_view_ref']==manifest['prepared_view']['prepared_view_ref'] and view['feature_view']==expected,
            'compact prepared view identity mismatch')
        common={k:fd['definition']['spec'][k] for k in ('scope','snapshot','pit_policy','calendar','universe',
            'catalog_ref','feature_selection','ordered_features')}
        require(view['definition']==common and definition['version']=='axiom.stock_ml_batch_inputs/3' and
                [f['fold_spec'] for f in manifest['folds']]==definition['fold_specs'] and
                len(manifest['folds'])==len(view['fold_targets'])>0,'compact common/fold definition mismatch')
        targets={}; previous=None
        for fold,record in zip(manifest['folds'],view['fold_targets']):
            spec,inputs=fold['fold_spec'],fold['input_manifest']; training,inference=validate_spec(spec,view['definition']['calendar'])
            require(spec==record['fold_spec'] and (previous is None or spec['oos_trade_sessions'][0]>previous),
                    'compact fold chronology/definition mismatch'); previous=spec['oos_trade_sessions'][-1]
            fields(inputs,{'contract_version','prepared_view','fold_spec_ref','selectors','core_result_refs','input_ref'},
                   'exact compact saved inputs required'); sealed(inputs,'input_ref')
            require(inputs['contract_version']=='stock_ml_saved_inputs_v3' and inputs['fold_spec_ref']==digest(spec) and
                    inputs['prepared_view']==manifest['prepared_view'],'compact fold input linkage mismatch')
            fields(inputs['selectors'],{'training','training_labels','inference','evaluation_labels','validation'},
                   'exact compact selectors required')
            require(inputs['selectors']=={k:None if k=='validation' else record['training_offsets' if k in
                ('training','training_labels') else 'inference_offsets'] for k in inputs['selectors']} and
                all(type(off) is int for k in ('training_offsets','inference_offsets') for off in record[k]) and
                reference(record['training_binding']['training_rows_ref']) and
                reference(record['training_binding']['training_keys_digest']) and
                record['training_binding']['training_row_count']==len(record['training_offsets']),
                'compact saved selector/training binding mismatch')
            raw_rows=[]
            for desc in record['raw_parts']:
                if desc['target_ref'] not in targets: targets[desc['target_ref']]=read_target(store,desc)
                value,rows=targets[desc['target_ref']]
                require(value['definition']['cutoff']==spec['fit_cutoff'] and value['definition']['snapshot']==view['definition']['snapshot'],
                        'compact Raw cutoff/Snapshot mismatch'); raw_rows.append(rows)
                _raw_binding(value['definition'],common,spec['fit_cutoff'])
            concatenated=ConcatRows(raw_rows)
            require([r['feature_session'] for r in concatenated[::len(view['definition']['universe'])]]==training,
                    'compact Raw training date coverage mismatch')
            for desc in (record['normalized'],record['evaluation']):
                if desc['target_ref'] not in targets:
                    targets[desc['target_ref']]=read_target(store,desc,raw_rows=concatenated if desc is record['normalized'] else None)
            nv,nrows=targets[record['normalized']['target_ref']]; ev,erows=targets[record['evaluation']['target_ref']]
            _raw_binding(ev['definition'],common,spec['evaluation_cutoff'])
            require(nv['definition']['sessions']==training and nv['definition']['cutoff']==spec['fit_cutoff'] and
                    nv['definition']['raw_refs']==[d['target_ref'] for d in record['raw_parts']] and
                    nv['core_ref']==record['core_ref'] and digest(nv['cohort'])==record['cohort_ref'] and
                    nv['cohort']['feature_view_ref']==fd['definition']['feature_view_ref'] and nv['cohort']['cutoff']==spec['fit_cutoff'] and
                    ev['definition']['sessions']==inference and ev['definition']['cutoff']==spec['evaluation_cutoff'],
                    'compact normalized/cohort/evaluation linkage mismatch')
            require(nv['definition']['universe']==common['universe'] and nv['definition']['calendar']==common['calendar'] and
                nv['definition']['normalization_spec']==NORMALIZATION_SPEC and
                nv['definition']['cohort_ref']==record['cohort_ref'] and
                nv['cohort']['sessions']==training and nv['cohort']['universe']==common['universe'] and
                nv['cohort']['raw_refs']==nv['definition']['raw_refs'] and
                len(nv['cohort']['eligibility_reasons'])==len(nrows) and
                nv['cohort']['eligible_keys']==[[r['security_id'],r['feature_session']] for r,reason in
                    zip(concatenated,nv['cohort']['eligibility_reasons']) if reason is None],
                'compact cohort/schema mismatch')
            positions={d:i for i,d in enumerate(fd['definition']['spec']['feature_sessions'])}; universe=view['definition']['universe']
            offsets=[positions[training[i//len(universe)]]*len(universe)+i%len(universe)
                     for i,flag in enumerate(nrows.arrays['validity']) if flag]
            require(offsets==record['training_offsets'] and record['inference_offsets']==[positions[d]*len(universe)+i for d in inference for i in range(len(universe))],
                    'compact fold key selectors mismatch')
            require(inputs['core_result_refs']==[record['core_ref']],'compact fold Core ref mismatch')
            require(record['training_binding']['training_keys_digest']==digest([[universe[o%len(universe)],
                fd['definition']['spec']['feature_sessions'][o//len(universe)]] for o in offsets]),
                'compact training keys digest mismatch')
        require(len(manifest['folds'])==len(view['fold_targets'])==len(definition['fold_specs'])>0,
                'complete compact fold coverage required')
        # Row snapshots are owned compact decoded targets, not source/proof
        # graphs. Account once after ingress, never once per training row.
        state=CompactState(store,feature,deepcopy(manifest),view,targets,own)
        charge=_size([targets,state.batch,state.selectors,state.source_records],maximum=store.maximum_matrix_bytes,
                     retained=store.shared_bytes+store.resident_bytes)
        store.reserve(charge); store.resident_bytes+=charge
        store.metrics['resident_bytes']=store.shared_bytes+store.resident_bytes
        state.check(); return state
    except BaseException:
        if state is not None: state.close()
        else:
            store.close()
            if own: feature.close()
        raise


def _raw_binding(definition,common,cutoff):
    source=definition['price_view']; sealed(source,'price_view_ref'); context=source['context']; query=context['query']
    require(definition['universe']==common['universe'] and definition['calendar']==common['calendar'] and
        definition['snapshot']==common['snapshot'] and definition['cutoff']==cutoff and
        definition['formula']=='close(f+5) / open(f+1) - 1' and definition['horizon_sessions']==5 and
        definition['start_session_offset']==1 and definition['end_session_offset']==5 and
        definition['price_basis']=='common_anchor_adjusted_v1' and
        definition['missing_policy']=='invalid_null_preserve_grid' and
        context['snapshot_id']==common['snapshot'] and query['symbols']==common['universe'] and
        query['purpose']=='label_outcomes' and query['pit_policy']==common['pit_policy'] and
        set(query['fields'])=={'open','close'} and query['price_basis']=='common_anchor_adjusted_v1' and
        set(query['cutoff_by_session'])==set(query['sessions']) and
        all(_instant(v)==_instant(cutoff) for v in query['cutoff_by_session'].values()) and
        reference(source['records_ref']) and reference(source['field_meta_ref']),
        'compact Raw query/calendar/vintage binding mismatch')


def load_compact_projection(inputs,spec,*,evaluation=False):
    # The manifest and view share one actual-byte admission. No preliminary
    # view decode/hash followed by a second independent read.
    descriptor=inputs['prepared_view']; store=OwnedStore()
    root=Path(descriptor['path']).parent
    try:
        manifest=store.read_json({'path':str(root/'batch.json')},key='content_digest')
        state=load_compact_state(manifest,publication_store=store)
    except BaseException: store.close(); raise
    try:
        projection=(state.project_evaluation(inputs,spec,batch_ref=manifest['batch_ref']) if evaluation else state.project(inputs,spec))
        projection._owned_state=state
        return projection
    except BaseException: state.close(); raise
