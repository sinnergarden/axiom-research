"""Ordinary compact target admission and the existing fold projection ABI."""
from copy import deepcopy
from contextlib import contextmanager, ExitStack
from pathlib import Path
import math
import sys

from .stock_artifacts import digest
from .stock_fold_inputs import require, seal, validate_spec, feature_available
from .stock_label_contracts import _instant, NORMALIZATION_SPEC
from .stock_compact_controls import (evaluation_parts, expand_ranges, validate_controls,
    validate_input, price_domain_ranges, validate_price_part)
from .stock_compact_store import (OwnedStore, _view_data, load_stock_feature_view,
    feature_rows, training_matrix, sealed, fields, limits, _size, reference, set_feature_window,
    clear_feature_window)
from .stock_compact_store import model_feature_binding,model_common,model_eligibility_ref


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
        normalized=value['contract_version'] in ('stock_compact_normalized_v1','stock_compact_normalized_v2')
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
    def column(self,name):
        import numpy as np
        if len(self.parts)==1:return self.parts[0].arrays[name]
        return np.concatenate([part.arrays[name] for part in self.parts])
    @property
    def sessions(self):
        return [day for part in self.parts for day in part.value['definition']['sessions']]
    def __iter__(self):
        for part in self.parts: yield from part
    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(len(self)))]
        for part in self.parts:
            if index<len(part): return part[index]
            index-=len(part)
        raise IndexError(index)


def target_eligibility_reasons(raw,feature,offsets,cutoff,common,*,model_feature_selection=None):
    """Apply the scalar admission precedence to admitted typed columns."""
    try:
        import numpy as np
        from .stock_compact_store import feature_eligibility_columns
        from .stock_matrix_storage import instant_us
        member,complete,knowledge,maximum=feature_eligibility_columns(feature,offsets,
            model_feature_selection=model_feature_selection)
        reasons=np.empty(len(raw),dtype=object);reasons[:]=None
        at=raw.column('availability');clock=instant_us(cutoff);first=0
        cutoff_day=_instant(cutoff).date().isoformat()
        for part in raw.parts:
            a=part.arrays;end=first+len(part);r=reasons[first:end]
            valid=a['validity'].view('?');dictionary=part.value['reason_dictionary']
            for code in np.unique(a['reason_codes'][~valid]):
                r[(~valid)&(a['reason_codes']==code)]=dictionary[int(code)] or 'RAW_LABEL_INVALID'
            known=(a['end_session']>=0)&(a['availability_validity']!=0)
            r[(r==None)&~known]='LABEL_CLOCK_OR_ENDPOINT_UNKNOWN'
            calendar=part.value['definition']['calendar']
            boundary=sum(day<=cutoff_day for day in calendar)-1
            r[(r==None)&((a['end_session']>boundary)|(a['availability']>clock))]='LABEL_NOT_MATURE'
            first=end
        reasons[(reasons==None)&~member]='NOT_MEMBER'
        reasons[(reasons==None)&~complete]='FEATURE_MISSING'
        reasons[(reasons==None)&(knowledge>clock)]='FEATURE_NOT_AVAILABLE'
        eligible=reasons==None
        require(bool(((maximum[eligible]!=-(2**63))&(maximum[eligible]<=clock)).all()),
            'training Feature native clock exceeds fit')
        target=common.get('target_spec')
        if target is not None and target['label_spec']['maturity']['rule']=='all_outcome_dependencies_strictly_before_fit_cutoff':
            reasons[eligible&(at>=clock)]='LABEL_NOT_MATURE'
        return reasons.tolist()

    finally:
        raw=feature=member=complete=knowledge=maximum=reasons=at=part=a=r=valid=known=eligible=None


def read_target(store,descriptor,expected=None,raw_rows=None):
    existing=getattr(store,'target_views',None)
    header=store.json.get(descriptor['path'])
    cached=None if existing is None or header is None else existing.get(header.get('target_ref'))
    if cached is not None and (raw_rows is None or cached[1].raw_rows is not None):
        try:
            value,rows=cached
            require(descriptor.get('file_digest',store.hashes[descriptor['path']])==store.hashes[descriptor['path']] and
                descriptor.get('target_ref',value['target_ref'])==value['target_ref'] and
                (expected is None or value['definition']==expected),'borrowed target definition mismatch')
            for path in (descriptor['path'],*(v['path'] for v in value['buffers'].values())):store.check_path(path)
            if raw_rows is not None:
                require(len(raw_rows)==len(rows) and [p.value['target_ref'] for p in raw_rows.parts]==value['definition']['raw_refs'],
                    'borrowed normalized Raw binding mismatch')
                rows.raw_rows=raw_rows
            store.metrics['target_array_borrows']=store.metrics.get('target_array_borrows',0)+1
            return value,rows
        except BaseException:
            cached=header=existing=value=rows=raw_rows=None
            raise
    value=store.read_json(descriptor,key='target_ref'); definition=value['definition']
    fields(value,{'contract_version','definition','definition_ref','row_count','reason_dictionary',
        'source_dictionary','buffers','core_ref','cohort','target_ref'},'exact compact target required')
    require(value['contract_version'] in ('stock_compact_raw_v1','stock_compact_normalized_v1','stock_compact_raw_v2','stock_compact_normalized_v2') and
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
    normalized=value['contract_version'] in ('stock_compact_normalized_v1','stock_compact_normalized_v2')
    expected_types={'values':'float64_le','validity':'bool_u8','availability':'int64_le',
        'availability_validity':'bool_u8','reason_codes':'int32_le'}
    if not normalized: expected_types.update(start_session='int32_le',end_session='int32_le',source_codes='int32_le')
    require(set(value['buffers'])==set(expected_types) and
        all(v['dtype']==expected_types[k] for k,v in value['buffers'].items()),'compact target physical dtypes mismatch')
    arrays=None
    try:
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
            require(digest(value['cohort'])==definition['cohort_ref'] and
                value['cohort']['sessions']==days and value['cohort']['universe']==universe and
                value['cohort']['cutoff']==definition['cutoff'] and value['cohort']['raw_refs']==definition['raw_refs'],
                'normalized cohort differs from its declared identity')
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
        codes=arrays['reason_codes'];valid=arrays['validity'].view('?');at=arrays['availability_validity'].view('?')
        require(bool(((codes>=0)&(codes<len(dictionary))).all()) and bool((valid==(codes==0)).all()) and
            bool((~valid|at).all()),'compact target validity/reason/clock mismatch')
        require(bool((valid|(arrays['values']==0.0)).all()),'compact null physical value must be zero')
        require(bool((at|(arrays['availability']==0)).all()),'compact null physical clock must be zero')
        require(bool((~at|(arrays['availability']<=cutoff)).all()),'compact target clock exceeds cutoff')
        if not normalized:
            h=definition['horizon_sessions'] if value['contract_version']=='stock_compact_raw_v2' else 5
            require(type(h) is int and h>0,'positive compact target horizon required')
            positions_by_row=np.repeat(np.asarray([positions[day] for day in days],dtype='<i4'),len(universe))
            starts=positions_by_row+1;ends=positions_by_row+h
            starts=np.where(starts<len(calendar),starts,-1);ends=np.where(ends<len(calendar),ends,-1)
            boundary=sum(day<=cutoff_day for day in calendar)-1
            require(bool((arrays['start_session']==starts).all()) and bool((arrays['end_session']==ends).all()) and
                bool((~valid|((starts>=0)&(ends>=0)&(ends<=boundary))).all()),'compact target endpoint offset mismatch')
            require(bool(((arrays['source_codes']>=0)&(arrays['source_codes']<len(value['source_dictionary']))).all()),
                'compact target source code mismatch')
        require(raw_rows is None or len(raw_rows)==count,'normalized Raw key alignment mismatch')
        if isinstance(raw_rows,ConcatRows):
            require(raw_rows.sessions==days and [p.value['target_ref'] for p in raw_rows.parts]==definition['raw_refs'],
                'normalized Raw source alignment mismatch')
        if value['contract_version']=='stock_compact_normalized_v2':
            from .stock_column_inputs import ColumnMathReuse
            proof={'contract_version':'stock_normalization_reuse_v1',
                'normalization_spec_ref':digest(NORMALIZATION_SPEC),
                'numerical_origins':definition['numerical_origins']}
            require(definition['normalization_proof_version']==proof['contract_version'] and
                value['core_ref']==digest(proof) and len(proof['numerical_origins'])==len(days),
                'normalized block numerical proof mismatch')
            width=len(universe)
            raw_column=raw_rows.column('values') if isinstance(raw_rows,ConcatRows) else None
            for n,(day,origin) in enumerate(zip(days,proof['numerical_origins'])):
                fields(origin,{'session','input_numeric_ref','output_numeric_ref'},'exact numerical proof required')
                require(origin['session']==day and reference(origin['input_numeric_ref']) and
                    origin['output_numeric_ref']==ColumnMathReuse.vector_ref((
                        arrays['values'][n*width:(n+1)*width],
                        arrays['validity'][n*width:(n+1)*width].view('?'))),
                    'normalized numerical output differs from admitted bytes')
                if raw_rows is not None:
                    members=np.asarray([r is None for r in value['cohort']['eligibility_reasons'][n*width:(n+1)*width]],dtype='?')
                    raw_values=(np.where(members,raw_column[n*width:(n+1)*width],0.0) if raw_column is not None
                        else np.asarray([raw_rows[n*width+j]['return'] if members[j] else 0.0 for j in range(width)],dtype='<f8'))
                    require(origin['input_numeric_ref']==digest({'vectors':ColumnMathReuse.vector_ref((raw_values,members)),
                        'normalization_spec_ref':digest(NORMALIZATION_SPEC)}),'normalized numerical input differs from Raw/cohort')
        store.metrics['target_semantic_admissions']=store.metrics.get('target_semantic_admissions',0)+1
        store.check(); rows=TargetRows(value,arrays,raw_rows)
        if existing is not None:existing[value['target_ref']]=(value,rows)
        return value,rows
    except BaseException:
        # Tracebacks keep frame locals. Drop array views and decoded aliases
        # before the owner can release their backing bytes after failed ingress.
        if arrays is not None: arrays.clear()
        arrays=value=definition=dictionary=calendar=raw_rows=raw_column=raw_values=valid=at=codes=positions_by_row=starts=ends=None
        cached=header=existing=rows=members=proof=origin=None
        raise



class CompactState:
    def __init__(self,store,feature,batch,view,targets,own_feature,*,residency='eager'):
        self.store=store; self.feature=feature; self.batch=batch; self.view=view; self.targets=targets
        store.target_views=self.targets
        store.ancestor_store=_view_data(feature,check=False)['store']
        self.closed=False; self.own_feature=own_feature; self.active=0
        self._batch_ref=batch.get('batch_ref',batch.get('definition_ref'))
        self.incomplete='batch_ref' not in batch
        self.residency=residency; self._pending_release=False; self._active_record=None
        self.model_binding=batch['definition'].get('model_feature_selection')
        self.feature_options=({} if self.model_binding is None else
            {'model_feature_selection':self.model_binding['selection']})
        self._control_charge=0; self._target_charge=0; self._source_charge=0
        self.columnar=batch['contract_version']=='stock_ml_batch_inputs_v5'
        self.compact=batch['contract_version'] in ('stock_ml_batch_inputs_v4','stock_ml_batch_inputs_v5'); self._control_path=None
        self.price_domains=price_domain_ranges([{'fold_spec':s} for s in batch['definition']['fold_specs']],view['definition']['calendar'],
            batch['definition']['preparation_options']['row_block_sessions']) if self.compact and not self.columnar else None
        self.selectors={(digest(f['input_manifest']),digest(f['fold_spec'])):f['input_manifest']['selectors'] for f in batch['folds']}
        self.records={(digest(f['input_manifest']),digest(f['fold_spec'])):(f,r)
            for f,r in zip(batch['folds'],[f['input_manifest']['fold_control'] for f in batch['folds']]
                if self.compact else view['fold_targets'])}
        fd=_view_data(feature); fd['borrowers']+=1
        self._fixed_shared_bytes=max(0,store.shared_bytes-fd['store'].resident_bytes)
        self._fixed_shared_source_bytes=max(0,store.shared_source_bytes-fd['store'].metrics['source_bytes'])
        self.positions={d:i for i,d in enumerate(fd['definition']['spec']['feature_sessions'])}
        self._keep_paths={batch['prepared_view']['path']} | {path for path,value in store.json.items()
            if value.get('contract_version') in ('stock_ml_batch_inputs_v3','stock_ml_batch_inputs_v4','stock_ml_batch_inputs_v5') and value.get('batch_ref')==self._batch_ref}
        self._source_table={}; self.source_records=()
        self._verified_folds=set()
        self._active_selected=None
        self._sync_shared()

    @contextmanager
    def operation(self):
        """Borrow the same Feature/target owners through nested helpers."""
        fd=_view_data(self.feature,check=False)
        with ExitStack() as stack:
            stack.enter_context(fd['store'].operation());stack.enter_context(self.store.operation())
            for path in fd['control_paths']:fd['store'].check_path(path)
            for path in self._keep_paths:self.store.check_path(path)
            yield

    def _sync_shared(self):
        fstore=_view_data(self.feature,check=False)['store']
        self.store.shared_bytes=fstore.resident_bytes+self._fixed_shared_bytes
        self.store.shared_source_bytes=fstore.metrics['source_bytes']+self._fixed_shared_source_bytes
        self.store.metrics.update(residency=self.residency,
            feature_residency=_view_data(self.feature,check=False).get('residency','eager'),
            resident_bytes=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes,
            active_target_count=len(self.targets),active_projection_count=self.active,
            current_window_fold_ref=self._active_record)

    def _controls_size(self,batch):
        try:
            return _size([batch,self.selectors,self.positions,self.price_domains,{key:None for key in self.records},
                          self._keep_paths],maximum=self.store.maximum_matrix_bytes,
                         retained=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes)
        finally:batch=None

    def _account_controls(self):
        charge=self._controls_size(self.batch)
        delta=charge-self._control_charge
        if delta>0:self.store.reserve(delta)
        self.store.resident_bytes+=delta; self._control_charge=charge
        self._remember_sources()

    def _append_checkpoint_fold(self,fold,*,produced_rows=None):
        """Admit one complete control under the original full price-domain plan."""
        require(self.incomplete and self.active==0 and self.store.borrowers==0,
                'idle incomplete checkpoint owner required')
        index=len(self.batch['folds'])
        require(index<len(self.batch['definition']['fold_specs']) and
            fold['fold_spec']==self.batch['definition']['fold_specs'][index], 'checkpoint fold is not the next planned fold')
        validate_input(fold,self.batch,_view_data(self.feature)['row_index'],self.view['definition'])
        key=(digest(fold['input_manifest']),digest(fold['fold_spec']))
        self.batch['folds'].append(fold)
        self.records[key]=(fold,fold['input_manifest']['fold_control'])
        self.selectors[key]=fold['input_manifest']['selectors']
        self._account_controls()
        record=None; admitted=False
        try:
            record=self._parts(fold['input_manifest'],fold['fold_spec'])
            if produced_rows is None:self._activate(fold,record)
            else:self._borrow_produced_fold(fold,record,produced_rows)
            self.check()
            admitted=True
        finally:
            record=produced_rows=None
            if not admitted:self._release_window()

    def _borrow_produced_fold(self,fold,record,rows):
        """Transfer this owner's fully checked producer window, without replay.

        This is a private object handoff, not a saved verified marker. Cold
        loaders always use _admit_fold; only the exact table already admitted
        by the same store can reach this branch.
        """
        try:
            require(self.columnar and self.store._operation_depth and self.active==0,
                'live column producer owner required')
            normalized,owned=self.targets[record['normalized']['target_ref']]
            require(rows is owned and rows.raw_rows is not None and
                normalized['target_ref']==record['normalized']['target_ref'] and
                normalized['core_ref']==record['core_ref'] and
                normalized['definition']['cohort_ref']==record['cohort_ref'] and
                normalized['definition']['raw_refs']==[d['target_ref'] for d in record['raw_parts']] and
                [p.value['target_ref'] for p in rows.raw_rows.parts]==normalized['definition']['raw_refs'],
                'produced window is not this owner\'s admitted target')
            spec=fold['fold_spec'];training,_=validate_spec(spec,self.view['definition']['calendar'])
            require(normalized['definition']['sessions']==training and normalized['definition']['cutoff']==spec['fit_cutoff'],
                'produced window changed its fold')
            for descriptor in [*record['raw_parts'],record['normalized'],*evaluation_parts(record)]:
                read_target(self.store,descriptor)
            width=len(self.view['definition']['universe'])
            selected=[self.positions[training[i//width]]*width+i%width for i,valid in enumerate(rows.arrays['validity']) if valid]
            require(fold['input_manifest']['selectors']['training']['selected_count']==len(selected) and
                record['training_binding']['training_row_count']==len(selected), 'produced training selection mismatch')
            for descriptor in record['training_binding']['feature_blocks']:
                require(self.store.hashes.get(descriptor['path'])==descriptor['file_digest'] and
                    self.store.json[descriptor['path']]['feature_block_ref']==descriptor['feature_block_ref'],
                    'produced Feature proof was not admitted by this writer')
                self.store.check_path(descriptor['path'])
            self.store.validate_boundary()
            self._account_targets();self._remember_sources()
            self._active_record=digest(spec);self._verified_folds.add(self._active_record);self._active_selected=selected
            self.store.metrics['fold_window_admissions']=self.store.metrics.get('fold_window_admissions',0)+1
            self.store.metrics['produced_fold_window_borrows']=self.store.metrics.get('produced_fold_window_borrows',0)+1
            self.store.metrics['verified_fold_count']=len(self._verified_folds)
            self._sync_shared()
        finally:rows=owned=normalized=record=fold=None

    def _complete_checkpoint(self,manifest,*,publish):
        require(self.incomplete and self.active==0 and self.store.borrowers==0 and
            len(self.batch['folds'])==len(self.batch['definition']['fold_specs']), 'complete checkpoint coverage required')
        require(manifest['definition']==self.batch['definition'] and manifest['folds']==self.batch['folds'] and
            manifest['prepared_view']==self.batch['prepared_view'], 'checkpoint final definition changed')
        self.check();charge=self._controls_size(manifest);delta=charge-self._control_charge
        if delta>0:self.store.reserve(delta)
        self.check();self.store.validate_boundary();publish()
        # No fallible admission remains after the public atomic link.
        self.store.resident_bytes+=delta;self._control_charge=charge
        self.batch=manifest;self._batch_ref=manifest['batch_ref'];self.incomplete=False

    def _account_targets(self):
        if not self.targets:
            self.store.resident_bytes-=self._target_charge; self._target_charge=0
            return
        charge=_size(_target_headers(self.targets),maximum=self.store.maximum_matrix_bytes,
                     retained=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes)
        charge+=sys.getsizeof(self.targets)+sum(sys.getsizeof(pair)+sys.getsizeof(vars(rows))+
            (sys.getsizeof(vars(rows.raw_rows)) if rows.raw_rows is not None else 0)
            for pair in self.targets.values() for rows in [pair[1]])
        delta=charge-self._target_charge
        if delta>0: self.store.reserve(delta)
        self.store.resident_bytes+=delta; self._target_charge=charge

    def _remember_sources(self):
        for store in (_view_data(self.feature,check=False)['store'],self.store):
            for path,ref in store.hashes.items():
                require(path not in self._source_table or self._source_table[path]==ref,
                        'conflicting compact admitted source bytes')
                self._source_table[path]=ref
        records=tuple(sorted(self._source_table.items()))
        charge=_size([self._source_table,records],maximum=self.store.maximum_matrix_bytes,
                     retained=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes-self._source_charge)
        delta=charge-self._source_charge
        if delta>0: self.store.reserve(delta)
        self.store.resident_bytes+=delta; self._source_charge=charge; self.source_records=records

    def _activate(self,fold,record):
        if self._active_record==digest(fold['fold_spec']) and self._active_selected is not None:
            for value,_ in self.targets.values():
                for item in value['buffers'].values():self.store.check_path(item['path'])
            self.store.metrics['fold_window_borrows']=self.store.metrics.get('fold_window_borrows',0)+1
            return self._active_selected
        if self.residency=='eager':
            if self.model_binding is None:
                return _admit_fold(self.store,self.feature,self.view,fold,record,self.targets,self.positions,self.price_domains,
                    self.batch['definition']['preparation_options']['row_block_sessions']) if self.compact else record['training_offsets']
            fstore=_view_data(self.feature,check=False)['store']
            previous=(fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes)
            try:
                fstore.limits={k:min(v,self.store.limits[k]) for k,v in fstore.limits.items()}
                fstore.shared_bytes=max(fstore.shared_bytes,self._fixed_shared_bytes+self.store.resident_bytes+self.store.lease_bytes)
                fstore.shared_source_bytes=max(fstore.shared_source_bytes,self._fixed_shared_source_bytes+self.store.metrics['source_bytes'])
                selected=_admit_fold(self.store,self.feature,self.view,fold,record,self.targets,self.positions,self.price_domains,
                    self.batch['definition']['preparation_options']['row_block_sessions'])
            finally: fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes=previous
            self._sync_shared(); self.store.reserve(0)
            return selected
        require(self.active==0 and self.store.borrowers==0,'sequential fold window still borrowed')
        spec=fold['fold_spec']; training,inference=validate_spec(spec,self.view['definition']['calendar'])
        width=len(self.view['definition']['universe'])
        # Admit the complete declared history, not merely eligible training
        # rows or a reduced lookback chosen to fit a memory budget.
        offsets=[self.positions[day]*width+i for day in sorted(set(training+inference)) for i in range(width)]
        try:
            keep_refs={desc['target_ref'] for desc in [*record['raw_parts'],record['normalized'],*evaluation_parts(record)]}
            self._trim_targets(keep_refs)
            fstore=_view_data(self.feature,check=False)['store']
            previous_shared=(fstore.shared_bytes,fstore.shared_source_bytes)
            previous_limits=fstore.limits
            try:
                fstore.limits={k:min(v,self.store.limits[k]) for k,v in previous_limits.items()}
                fstore.shared_bytes=max(fstore.shared_bytes,self._fixed_shared_bytes+self.store.resident_bytes+self.store.lease_bytes)
                fstore.shared_source_bytes=max(fstore.shared_source_bytes,self._fixed_shared_source_bytes+self.store.metrics['source_bytes'])
                set_feature_window(self.feature,offsets,**self.feature_options)
            finally:
                fstore.shared_bytes,fstore.shared_source_bytes=previous_shared
                fstore.limits=previous_limits
            self._sync_shared(); self.store.reserve(0)
            selected=_admit_fold(self.store,self.feature,self.view,fold,record,self.targets,self.positions,self.price_domains,
                self.batch['definition']['preparation_options']['row_block_sessions'])
            self._account_targets(); self._remember_sources()
            self._active_record=digest(spec); self._verified_folds.add(self._active_record)
            self._active_selected=selected
            self.store.metrics['fold_window_admissions']=self.store.metrics.get('fold_window_admissions',0)+1
            self.store.metrics['verified_fold_count']=len(self._verified_folds)
            self._sync_shared(); self.check()
            return selected
        except BaseException:
            # _admit_fold clears local array aliases before this trim, including
            # exception traceback frames. Actual bytes are credited afterwards.
            record=selected=None
            self._release_window()
            raise

    def _projection_finalized(self):
        self.active-=1
        _view_data(self.feature,check=False)['store'].borrowers-=1
        require(self.active>=0,'compact projection borrower underflow')
        if self.residency=='sequential': self._pending_release=True

    def _projection_cleared(self):
        # Keep exactly the current immutable input window, so an exact HIT or
        # the next overlapping fold can reuse bytes. X/y/P have been detached;
        # the next activation trims everything outside its declared closure.
        self._pending_release=False; self._sync_shared()

    def _trim_targets(self,keep_refs=()):
        keep_refs=set(keep_refs)
        for ref in list(self.targets):
            if ref not in keep_refs: del self.targets[ref]
        self._account_targets()
        keep_paths=set(self._keep_paths)
        if self._control_path is not None: keep_paths.add(self._control_path)
        if self.columnar and self._control_path in self.store.json:
            control=self.store.json[self._control_path]
            keep_paths.update(d['path'] for d in control['training_binding']['feature_blocks'])
        for value,_ in self.targets.values():
            keep_paths.update(item['path'] for item in value['buffers'].values())
        for path,value in self.store.json.items():
            if value.get('target_ref') in self.targets: keep_paths.add(path)
        value=None
        return self.store.release_payloads(keep_paths=keep_paths)

    def _release_window(self,*,keep_feature=False):
        if self.residency!='sequential' or self.closed: return
        require(self.active==0 and self.store.borrowers==0,'sequential fold window still borrowed')
        self._control_path=None
        released=self._trim_targets()
        # The immutable window belongs to the shared Feature owner, including
        # the full-column path. Successful prepare must hand it to the builder
        # under the same budget; failed admission still clears the window.
        if not keep_feature: clear_feature_window(self.feature)
        self._active_record=None; self._pending_release=False; self._sync_shared()
        self._active_selected=None
        return released

    def verify_all(self):
        """Validate every saved fold using one source window and no matrices."""
        self.check()
        require(self.active==0 and self.store.borrowers==0,'compact verification window still borrowed')
        if self.residency=='eager': return
        complete=False
        try:
            for fold in self.batch['folds']:
                record=self._parts(fold['input_manifest'],fold['fold_spec'])
                self._activate(fold,record); self.check()
                record=None
            require(len(self._verified_folds)==len(self.batch['folds']),'complete compact fold admission required')
            complete=True
        finally:
            record=None
            self._release_window(keep_feature=complete)

    def check(self):
        require(not self.closed,'compact batch is closed')
        if self._pending_release and self.active==0: self._projection_cleared()
        self._sync_shared(); self.store.check(); _view_data(self.feature)

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
        require(getattr(self,'_build_oos_borrowers',0)==0,'compact batch OOS writer still borrowed')
        require(self.active==0 and self.store.borrowers==0,'compact batch backing still borrowed')
        if self.closed: return
        if self.residency=='sequential': self._release_window(keep_feature=not self.own_feature)
        self.targets.clear(); self.store.close(); self.closed=True; fd=_view_data(self.feature,check=False); fd['borrowers']-=1
        if self.own_feature: self.feature.close()

    def _parts(self,inputs,spec):
        self.check(); key=digest(inputs),digest(spec)
        require(key in self.selectors,'fold outside compact batch definition')
        record=self.records[key][1]
        if self.compact:
            require(self.active==0 and self.store.borrowers==0,'compact control window still borrowed')
            try:
                changed=self._control_path!=record['path']
                self._control_path=record['path']
                record=self.store.read_json(record,key='fold_control_ref')
                validate_controls(self.records[key][0],record,self.view,self.batch,_view_data(self.feature)['row_index'])
                if changed:self._trim_targets(self.targets)
                return record
            except BaseException:
                record=None; self._release_window(); raise
        require(inputs['selectors']=={k:None if k=='validation' else record['training_offsets' if k in
            ('training','training_labels') else 'inference_offsets'] for k in inputs['selectors']},
            'compact fold selectors mismatch')
        return record

    def control_common(self,inputs,spec):
        """Return saved common metadata without admitting typed fold inputs."""
        self._parts(inputs,spec)
        return deepcopy(self.view['definition'])

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

    def evaluation_sources(self,inputs,spec):
        """The actual active fold closure, including reused owned bytes.

        The historical source table is intentionally not used: an eager owner
        or an earlier sequential fold may have admitted unrelated leaves.
        """
        self.check()
        require(self.compact and self.active>0,'active compact OOS projection required')
        record=self.store.json[inputs['fold_control']['path']]
        fd=_view_data(self.feature); training,inference=validate_spec(spec,self.view['definition']['calendar'])
        paths=set(self._keep_paths)|{inputs['fold_control']['path']}
        if self.batch['contract_version']=='stock_ml_batch_inputs_v5':
            paths.update(d['path'] for d in record['training_binding']['feature_blocks'])
        targets=[]
        for descriptor in [*record['raw_parts'],record['normalized'],*evaluation_parts(record)]:
            header,rows=self.targets[descriptor['target_ref']]
            paths.add(descriptor['path']); paths.update(d['path'] for d in header['buffers'].values())
        for descriptor in evaluation_parts(record):
            header,rows=self.targets[descriptor['target_ref']]
            targets.append((descriptor,header,rows))
        feature_paths=set(fd['control_paths']); ordinals=set()
        for day in set(training+inference): ordinals.update(fd['day_admissions'][self.positions[day]])
        for ordinal in ordinals:
            block=fd['blocks'][ordinal]
            require(block['parts'] is not None,'evaluation Feature block not admitted')
            for part in block['descriptors']:
                feature_paths.add(part['metadata']['path'])
                feature_paths.update(d['path'] for d in part['buffers'].values())
        records={}; marks={}
        for store,chosen in ((self.store,paths),(fd['store'],feature_paths)):
            for path in chosen:
                require(path in store.hashes and path in store.marks,'evaluation source not admitted')
                ref=store.hashes[path]
                require(path not in records or records[path]==ref,'conflicting evaluation source pin')
                require(path not in marks or marks[path]==store.marks[path],'conflicting evaluation source fingerprint')
                records[path]=ref; marks[path]=store.marks[path]
        return targets,records,marks

    def evaluation(self,inputs,spec,*,batch_ref):
        raise ValueError('compact v3 has no legacy signal-evaluation proof projection; use explicit batch audit')

    def _project(self,inputs,spec,*,training):
        from .stock_matrix_reader import MatrixFoldProjection
        from .stock_label_contracts import TARGET_SEMANTICS
        import numpy as np
        record=training_offsets=inference_offsets=normalized=nrows=X=y=P=payload=a=None; committed_lease=0; guarded=False
        try:
            record=self._parts(inputs,spec)
            training_offsets=self._activate(self.records[(digest(inputs),digest(spec))][0],record)
            inference_offsets=expand_ranges(inputs['selectors']['inference']) if self.compact else record['inference_offsets']
            common=deepcopy(self.view['definition']); normalized=self.targets[record['normalized']['target_ref']]
            self.store.reserve(len(inference_offsets)*(len(common['ordered_features'])*96+2048))
            rows=feature_rows(self.feature,inference_offsets,**self.feature_options); candidate_keys=[]; candidates=[]
            for row in rows:
                if row['member'] and all(row['validity']) and all(type(v) in (int,float) and math.isfinite(v) for v in row['values']) and feature_available(row) is not None:
                    candidates.append(row['values']); candidate_keys.append([row['security_id'],row['session']])
            features=self._feature_slice(inputs,spec,rows); nrows=normalized[1]
            fspec=_view_data(self.feature)['definition']['spec']; width=len(common['universe'])
            keys=[[common['universe'][off%width],fspec['feature_sessions'][off//width]] for off in training_offsets]
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
            erows=[row for desc in evaluation_parts(record) for row in self.targets[desc['target_ref']][1]]
            evaluation=seal({'contract_version':'stock_matrix_evaluation_target_slice_v1',
                'prepared_view_ref':self.view['prepared_view_ref'],'fold_spec_ref':digest(spec),
                'selector':deepcopy(inputs['selectors']['evaluation_labels']),'cutoff':spec['evaluation_cutoff'],'rows':erows},'label_ref')
            bindings=record['training_binding']; training_ref=bindings['binding_ref' if self.columnar else 'training_rows_ref']
            payload={'common':common,'features':features,'labels':labels,'evaluation':evaluation,
                'training_keys':keys,('training_binding_ref' if self.columnar else 'training_rows_ref'):training_ref,'excluded':excluded,'raw_refs':raw_refs,
                'candidate_keys':candidate_keys,'feature_rows':rows,'feature_ref':features['feature_ref'],'label_ref':labels['label_ref']}
            if training:
                gather_rows=min(1024,len(keys)); feature_width=len(common['ordered_features'])
                workspace=gather_rows*feature_width*8+gather_rows*32+len(nrows)
                self.store.reserve((len(keys)+len(candidates))*feature_width*8+len(keys)*8+workspace)
                X=training_matrix(self.feature,training_offsets,spec['fit_cutoff'],**self.feature_options)
                y=nrows.arrays['values'][nrows.arrays['validity'].astype(bool)]
                P=np.asarray(candidates,dtype='<f8').reshape(len(candidates),len(common['ordered_features']))
                require(bool(np.isfinite(X).all()) and bool(np.isfinite(y).all()) and bool(np.isfinite(P).all()),'compact active matrix must be finite')
                for a in (X,y,P): a.flags.writeable=False
                payload.update(X=X,y=y,P=P)
                self.store.metrics['training_projection_calls']=self.store.metrics.get('training_projection_calls',0)+1
                self.store.metrics['matrix_allocated_bytes']=self.store.metrics.get('matrix_allocated_bytes',0)+sum(a.nbytes for a in (X,y,P))
            else:
                self.store.metrics['evaluation_projection_calls']=self.store.metrics.get('evaluation_projection_calls',0)+1
                from .stock_dataset_binding import dataset_binding
                dataset=dataset_binding(common=common,features=features,labels=labels,raw_refs=raw_refs,
                    keys=keys,training_ref=training_ref,excluded=excluded,inputs=inputs,spec=spec)
                payload['fold_binding']={'dataset':dataset,'labels':labels}
            lease=_size(payload,maximum=self.store.maximum_matrix_bytes,
                retained=self.store.shared_bytes+self.store.resident_bytes+self.store.lease_bytes)+sum(
                a.nbytes for k,a in payload.items() if k in ('X','y','P'))+4096
            self.store.reserve(lease); self.check()
            self.store.lease_bytes+=lease; committed_lease=lease
            self.store.metrics['fold_projection_calls']+=1
            self.active+=1; _view_data(self.feature,check=False)['store'].borrowers+=1
            guarded=True
            # Match the original payload measurement plus explicit native
            # nbytes. Headers/None replacement are conservatively retained.
            matrix_charge=0
            if training:
                import sys
                matrix_charge=max(0,sum(sys.getsizeof(a)+a.nbytes for a in (X,y,P))-64)
            return MatrixFoldProjection(self.store,**payload,_lease_bytes=lease,
                _matrix_lease_bytes=matrix_charge,
                _release_notice=self._projection_finalized,_after_clear=self._projection_cleared,
                _on_failure=self._release_window)
        except BaseException:
            if payload is not None: payload.clear()
            normalized=nrows=X=y=P=payload=a=None
            training_offsets=inference_offsets=record=None
            common=features=labels=evaluation=rows=candidates=keys=erows=row=None
            if guarded:
                self.active-=1; _view_data(self.feature,check=False)['store'].borrowers-=1
            if committed_lease: self.store.lease_bytes-=committed_lease
            self._release_window()
            raise



def _validate_fold_controls(fold,record,view,manifest):
    spec,inputs=fold['fold_spec'],fold['input_manifest']; training,inference=validate_spec(spec,view['definition']['calendar'])
    require(spec==record['fold_spec'],'compact fold definition mismatch')
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
    return training,inference


def _admit_fold(store,feature,view,fold,record,targets,positions=None,price_domains=None,block=None):
    """The same actual-byte and PIT checks serve eager and sequential loads."""
    spec,inputs=fold['fold_spec'],fold['input_manifest']
    common=view['definition']; training,inference=validate_spec(spec,common['calendar'])
    fd=_view_data(feature)
    raw_rows=[]; concatenated=nv=nrows=ev=erows=value=rows=None
    try:
        for desc in record['raw_parts']:
            if desc['target_ref'] not in targets: targets[desc['target_ref']]=read_target(store,desc)
            value,rows=targets[desc['target_ref']]
            require(value['definition']['cutoff']==spec['fit_cutoff'] and value['definition']['snapshot']==view['definition']['snapshot'],
                    'compact Raw cutoff/Snapshot mismatch'); raw_rows.append(rows)
            _raw_binding(value['definition'],common,spec['fit_cutoff'])
            if price_domains is not None: validate_price_part(value['definition'],common,price_domains,block,evaluation=False)
        concatenated=ConcatRows(raw_rows)
        require(concatenated.sessions==training,
                'compact Raw training date coverage mismatch')
        for desc in [record['normalized'],*evaluation_parts(record)]:
            if desc['target_ref'] not in targets:
                targets[desc['target_ref']]=read_target(store,desc,raw_rows=concatenated if desc is record['normalized'] else None)
        nv,nrows=targets[record['normalized']['target_ref']]
        erows=ConcatRows([targets[desc['target_ref']][1] for desc in evaluation_parts(record)])
        for desc in evaluation_parts(record):
            ev=targets[desc['target_ref']][0]; _raw_binding(ev['definition'],common,spec['evaluation_cutoff'])
            if price_domains is not None: validate_price_part(ev['definition'],common,price_domains,block,evaluation=True)
            require(ev['definition']['cutoff']==spec['evaluation_cutoff'],'compact evaluation cutoff mismatch')
        require(nv['definition']['sessions']==training and nv['definition']['cutoff']==spec['fit_cutoff'] and
                nv['definition']['raw_refs']==[d['target_ref'] for d in record['raw_parts']] and
                nv['core_ref']==record['core_ref'] and digest(nv['cohort'])==record['cohort_ref'] and
                nv['cohort']['feature_view_ref']==fd['definition']['feature_view_ref'] and nv['cohort']['cutoff']==spec['fit_cutoff'] and
                erows.sessions==inference,
                'compact normalized/cohort/evaluation linkage mismatch')
        require(nv['definition']['universe']==common['universe'] and nv['definition']['calendar']==common['calendar'] and
            nv['definition']['normalization_spec']==NORMALIZATION_SPEC and
            nv['definition']['cohort_ref']==record['cohort_ref'] and
            nv['cohort']['sessions']==training and nv['cohort']['universe']==common['universe'] and
            nv['cohort']['raw_refs']==nv['definition']['raw_refs'] and
            len(nv['cohort']['eligibility_reasons'])==len(nrows) and
            nv['cohort']['eligible_keys']==[[common['universe'][i%len(common['universe'])],training[i//len(common['universe'])]]
                for i,reason in enumerate(nv['cohort']['eligibility_reasons']) if reason is None],
            'compact cohort/schema mismatch')
        binding=common.get('model_feature_selection')
        require(nv['cohort'].get('model_feature_eligibility_ref')==model_eligibility_ref(binding),
                'compact selected cohort mismatch')
        if binding is not None or inputs['contract_version']=='stock_ml_saved_inputs_v5':
            from .stock_compact_store import iter_feature_eligibility
            from .stock_label_contracts import _eligible_reason
            if positions is None: positions={d:i for i,d in enumerate(fd['definition']['spec']['feature_sessions'])}
            offsets=[positions[d]*len(common['universe'])+i for d in training for i in range(len(common['universe']))]
            reasons=target_eligibility_reasons(concatenated,feature,offsets,spec['fit_cutoff'],common,
                model_feature_selection=None if binding is None else binding['selection'])
            require(reasons==nv['cohort']['eligibility_reasons'],'selected Feature cohort differs from admitted bytes')
        if positions is None: positions={d:i for i,d in enumerate(fd['definition']['spec']['feature_sessions'])}
        universe=view['definition']['universe']
        offsets=[positions[training[i//len(universe)]]*len(universe)+i%len(universe)
                 for i,flag in enumerate(nrows.arrays['validity']) if flag]
        if 'training_offsets' in record:
            require(offsets==record['training_offsets'] and record['inference_offsets']==[positions[d]*len(universe)+i for d in inference for i in range(len(universe))],
                    'compact fold key selectors mismatch')
        else:
            require(len(offsets)==inputs['selectors']['training']['selected_count'],'compact training selected count mismatch')
        require(inputs['core_result_refs']==[record['core_ref']],'compact fold Core ref mismatch')
        require(record['training_binding']['training_keys_digest']==digest([[universe[o%len(universe)],
            fd['definition']['spec']['feature_sessions'][o//len(universe)]] for o in offsets]),
            'compact training keys digest mismatch')
        if inputs['contract_version']=='stock_ml_saved_inputs_v5':
            from .stock_training_blocks import validate_training_block_binding
            validate_training_block_binding(record['training_binding'],feature,offsets,normalized=nv,
                cohort_ref=record['cohort_ref'],selector=inputs['selectors']['training'],store=store,
                model_feature_selection=None if binding is None else binding['selection'])
        return offsets
    except BaseException:
        raw_rows.clear()
        concatenated=nv=nrows=ev=erows=value=rows=None
        raise



def _target_headers(targets):
    # JSON definitions and byte payloads have their own per-file charges. Count
    # only the virtual row wrappers, ndarray headers and container graph here.
    headers=[]
    for ref,(_,rows) in targets.items():
        headers.append([ref,rows,rows.arrays])
        if rows.raw_rows is not None: headers.append([rows.raw_rows,rows.raw_rows.parts])
    return headers


def load_compact_state(manifest,*,feature_inputs=None,limits=None,resolver=None,publication_store=None,residency='eager'):
    require(residency in ('eager','sequential'),'compact residency must be eager or sequential')
    fields(manifest,{'contract_version','definition','definition_ref','prepared_view','folds','status','batch_ref','content_digest'},
           'exact compact batch manifest required')
    require(manifest['contract_version'] in ('stock_ml_batch_inputs_v3','stock_ml_batch_inputs_v4','stock_ml_batch_inputs_v5') and manifest['status']=='COMPLETE',
            'complete compact batch required')
    sealed(manifest,'content_digest'); require(manifest['batch_ref']==digest({k:v for k,v in manifest.items() if k not in
        ('batch_ref','content_digest')}) and manifest['definition_ref']==digest(manifest['definition']),'compact batch identity mismatch')
    budgets=globals()['limits'](limits); definition=manifest['definition']; expected=definition['feature_view']; own=feature_inputs is None
    feature=store=state=None
    try:
        binding=definition.get('model_feature_selection')
        feature=load_stock_feature_view(Path(expected['source_index']['path']).parent,limits=budgets,residency=residency,
            model_feature_selection=None if binding is None else binding['selection']) if own else feature_inputs
        fd=_view_data(feature)
        require(binding==model_feature_binding(feature,None if binding is None else binding['selection']),
                'saved model selection/schema/source mismatch')
        require(residency!='eager' or fd['residency']=='eager',
                'sequential Feature requires sequential batch residency')
        require(feature.to_dict()==expected,'compact Feature view exact identity/cutoff mismatch')
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
        columnar=manifest['contract_version']=='stock_ml_batch_inputs_v5'
        compact=manifest['contract_version'] in ('stock_ml_batch_inputs_v4','stock_ml_batch_inputs_v5')
        view=store.read_json(manifest['prepared_view'],key='prepared_view_ref')
        fields(view,{'contract_version','definition','feature_view','prepared_view_ref'} | (set() if compact else {'fold_targets'}),
               'exact compact prepared view required')
        require(view['contract_version']==('stock_ml_prepared_view_v4' if columnar else 'stock_ml_prepared_view_v3' if compact else 'stock_ml_prepared_view_v2') and
            view['prepared_view_ref']==manifest['prepared_view']['prepared_view_ref'] and view['feature_view']==expected,
            'compact prepared view identity mismatch')
        common=model_common(fd['definition']['spec'],binding)
        if columnar:
            from .stock_target_spec import resolve_stock_label_spec
            target=resolve_stock_label_spec(definition['target_spec']['label_spec'])
            require(target==definition['target_spec'],'saved target profile identity mismatch')
            common.update(target_spec=target,source_contract='data_column_selection_v1')
        require(view['definition']==common and definition['version']==('axiom.stock_ml_batch_inputs/5' if columnar else 'axiom.stock_ml_batch_inputs/4' if compact else 'axiom.stock_ml_batch_inputs/3') and
                [f['fold_spec'] for f in manifest['folds']]==definition['fold_specs'] and
                (compact or len(manifest['folds'])==len(view['fold_targets'])) and bool(manifest['folds']),
                'compact common/fold definition mismatch')
        if compact: require(definition['price_domain_plan']==('column_asof_endpoint_dependencies_v1' if columnar else 'fit_window_evaluation_calendar_blocks_v1'),'compact query plan mismatch')
        previous=None
        for n,fold in enumerate(manifest['folds']):
            if compact: validate_input(fold,manifest,fd['row_index'],common)
            else: _validate_fold_controls(fold,view['fold_targets'][n],view,manifest)
            spec=fold['fold_spec']
            require(previous is None or spec['oos_trade_sessions'][0]>previous,
                    'compact fold chronology/definition mismatch')
            previous=spec['oos_trade_sessions'][-1]
        require(len(manifest['folds'])==len(definition['fold_specs'])>0,
                'complete compact fold coverage required')
        state=CompactState(store,feature,deepcopy(manifest),view,{},own,residency=residency)
        state._account_controls()
        if residency=='eager':
            if binding is not None:
                from .stock_compact_store import _admit_model_columns
                fstore=fd['store']; previous=(fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes)
                try:
                    fstore.limits={k:min(v,budgets[k]) for k,v in fstore.limits.items()}
                    fstore.shared_bytes=max(fstore.shared_bytes,store.resident_bytes+store.lease_bytes)
                    fstore.shared_source_bytes=max(fstore.shared_source_bytes,store.metrics['source_bytes'])
                    for block in fd['blocks']: _admit_model_columns(fd,block,binding)
                finally: fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes=previous
                state._sync_shared(); store.reserve(0)
            for fold in manifest['folds']:
                record=state._parts(fold['input_manifest'],fold['fold_spec'])
                _admit_fold(store,feature,view,fold,record,state.targets,state.positions,state.price_domains,
                    definition['preparation_options']['row_block_sessions'])
            state._account_targets(); state._remember_sources()
        state.check(); return state
    except BaseException:
        if state is not None: state.close()
        else:
            if type(store) is OwnedStore: store.close()
            if own and feature is not None: feature.close()
        raise


def _load_checkpoint_state(definition,view_descriptor,plan_descriptor,folds,*,feature_inputs,limits,publication_store):
    """Private partial owner. The public COMPLETE manifest loader stays strict."""
    state=manifest=view=plan=None
    try:
        fd=_view_data(feature_inputs);store=publication_store
        require(type(store) is OwnedStore and not store.closed and store.borrowers==0,'active checkpoint byte owner required')
        plan=store.read_json(plan_descriptor)
        columnar=definition['version']=='axiom.stock_ml_batch_inputs/5'
        require(plan==definition and definition['version'] in ('axiom.stock_ml_batch_inputs/4','axiom.stock_ml_batch_inputs/5') and
            definition['price_domain_plan']==('column_asof_endpoint_dependencies_v1' if columnar else 'fit_window_evaluation_calendar_blocks_v1') and
            definition['feature_view']==feature_inputs.to_dict(), 'checkpoint final plan/source mismatch')
        binding=definition.get('model_feature_selection')
        require(binding==model_feature_binding(feature_inputs,None if binding is None else binding['selection']),
                'checkpoint model selection mismatch')
        view=store.read_json(view_descriptor,key='prepared_view_ref')
        fields(view,{'contract_version','definition','feature_view','prepared_view_ref'},'exact checkpoint prepared view required')
        common=model_common(fd['definition']['spec'],binding)
        if columnar:
            from .stock_target_spec import resolve_stock_label_spec
            target=resolve_stock_label_spec(definition['target_spec']['label_spec'])
            require(target==definition['target_spec'],'checkpoint target profile mismatch')
            common.update(target_spec=target,source_contract='data_column_selection_v1')
        require(view['contract_version']==('stock_ml_prepared_view_v4' if columnar else 'stock_ml_prepared_view_v3') and view['definition']==common and
            view['feature_view']==definition['feature_view'], 'checkpoint prepared view/source mismatch')
        require(type(folds) is list and len(folds)<=len(definition['fold_specs']) and
            [f['fold_spec'] for f in folds]==definition['fold_specs'][:len(folds)], 'checkpoint ready prefix mismatch')
        manifest={'contract_version':'stock_ml_batch_inputs_v5' if columnar else 'stock_ml_batch_inputs_v4',
            'definition':definition,'definition_ref':digest(definition),
            'prepared_view':view_descriptor,'folds':folds}
        for fold in folds:validate_input(fold,manifest,fd['row_index'],common)
        state=CompactState(store,feature_inputs,manifest,view,{},False,residency='sequential')
        state._keep_paths.add(plan_descriptor['path']);state._account_controls()
        state.verify_all();state.check()
        return state
    except BaseException:
        if state is not None:state.close()
        raise
    finally:
        definition=view_descriptor=plan_descriptor=folds=manifest=view=plan=fd=common=None
        state=store=feature_inputs=publication_store=binding=fold=None


def _raw_binding(definition,common,cutoff):
    if 'target_spec' in common:
        from .stock_column_inputs import validate_column_raw_binding
        return validate_column_raw_binding(definition,common,cutoff)
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


def load_compact_state_from_inputs(inputs):
    # The manifest and view share one actual-byte admission. No preliminary
    # view decode/hash followed by a second independent read.
    descriptor=inputs['prepared_view']; store=OwnedStore()
    root=Path(descriptor['path']).parent
    try:
        manifest=store.read_json({'path':str(root/'batch.json')},key='content_digest')
        state=load_compact_state(manifest,publication_store=store,residency='sequential')
    except BaseException: store.close(); raise
    return state


def load_compact_projection(inputs,spec,*,evaluation=False):
    state=load_compact_state_from_inputs(inputs)
    try:
        projection=(state.project_evaluation(inputs,spec,batch_ref=state._batch_ref) if evaluation else state.project(inputs,spec))
        projection._owned_state=state
        return projection
    except BaseException: state.close(); raise
