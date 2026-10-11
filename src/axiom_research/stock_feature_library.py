"""Research definitions and thin adapters for Engine's incremental Feature store.

Saved source admission stays with its existing Research owner. This adapter
supports the proved CSI300/vendor profile only. It uses original Data adjustment
and Core DAG/CS arithmetic; no Feature formula is implemented here.
"""
from copy import deepcopy
from contextlib import nullcontext
from hashlib import sha256

from axiom_engine.core import FeaturePlan, required_history
from axiom_engine.core.feature_columns import _needed
from axiom_engine.core.plan import CS
from axiom_engine.feature_library import FeatureLibrary, FeatureRequest, feature_definitions
from axiom_engine.feature_library.codec import digest
from .stock_fold_inputs import require
from .stock_matrix_reader import VerifiedMatrixStore
from .stock_matrix_feature_column_metadata import verify_feature_column_source
from .stock_feature_columns import FeatureColumnSource, feature_column_plan


def _array_ref(array):
    return {'dtype':array.dtype.str,'shape':list(array.shape),
            'sha256':sha256(array.tobytes(order='C')).hexdigest()}


class SavedColumnFeatureSource:
    """Admit an immutable source through the original owner, then borrow mmap."""
    def __init__(self, descriptor, *, reference_security_ids=None, maximum_source_bytes=8*1024**3,
                 maximum_working_bytes=512*1024**2):
        self.store=VerifiedMatrixStore(maximum_source_bytes=maximum_source_bytes,
                                      maximum_matrix_bytes=maximum_working_bytes)
        self._leaf_refs={}; self._dependencies={}; self._buffer_refs={}; self._reference_refs={}
        self.metrics={'physical_field_day_hashes':0,'physical_hash_bytes':0}
        manifest,_=verify_feature_column_source(descriptor,store=self.store,
            _dependency_receipts=self._capture_leaf_refs)
        require(manifest['qlib_view']['universe_name']=='csi300', 'unproved source universe profile')
        self.descriptor=deepcopy(descriptor)
        arrays={name:self.store.buffer(desc) for name,desc in manifest['buffers'].items()}
        for name, value in arrays.items():
            if manifest['buffers'][name]['dtype']=='bool_u8': arrays[name]=value.view('bool')
        self.source=FeatureColumnSource(manifest,arrays)
        physical=manifest['spec']['universe']
        self.reference_security_ids=list(physical if reference_security_ids is None else reference_security_ids)
        require(self.reference_security_ids==sorted(set(self.reference_security_ids)) and
                set(self.reference_security_ids)<=set(physical), 'explicit reference subset of admitted source required')
        self._reference_set=set(self.reference_security_ids)

    def _capture_leaf_refs(self, child):
        for field, metadata in child['field_meta'].items():
            grouped={day:[] for day in child['sessions']}
            for row in metadata['by_key']:
                require(row['session'] in grouped, 'source provenance outside declared chunk')
                grouped[row['session']].append(row)
            for day,rows in grouped.items():
                rows=sorted(rows,key=lambda r:r['security_id'])
                symbols=tuple(r['security_id'] for r in rows)
                require(len(symbols)==len(set(symbols)), 'duplicate source provenance key')
                self._leaf_refs[field,day]=(symbols,digest({'dtype':metadata['dtype'],
                    'unit':metadata['unit'],'selected_revisions':rows}))

    def capabilities(self):
        return {'contract_version':'feature_source_capabilities_v1','source_ref':self.source.identity,
            'universe_profile':'csi300','pit_policies':['best_effort_vendor_v1'],
            'physical_security_ids':list(self.source.manifest['spec']['universe']),
            'reference_security_ids':list(self.reference_security_ids),
            'sessions':list(self.source.manifest['spec']['read_sessions']), 'daily_anchor':True}

    def input_semantics(self):
        profiles=self.source.manifest['source_profiles']
        result={}
        for field in ('open','high','low','close','amount_cny'):
            result[field]={'value_projection':'qlib_float32_then_binary64_v1',
                'source_profile':self._field_profile(profiles['market_daily'],field), 'missing':'preserve',
                'clock':'leaf_max_ceil_second_cutoff_floor_second_v1'}
            if field!='amount_cny': result[field].update(adjustment='Data.adjust_prices_output_day_anchor_v1',
                factor_profile=self._field_profile(profiles['adjustment_factors'],'factor'))
        return result

    @staticmethod
    def _field_profile(profile,field):
        contract=profile['contract']
        selected=deepcopy(profile['source_profile'])
        # Canonical security axes and selected revision rows bind identity maps;
        # adding an unrelated security must not change every field's semantics.
        selected.pop('identity_map',None)
        raw_field=selected.get('field_map',{}).get(field)
        if 'field_map' in selected:
            selected['field_map']={k:v for k,v in selected['field_map'].items() if k in (field,'security_id','session')}
        if 'source_units' in selected:
            selected['source_units']={k:v for k,v in selected['source_units'].items() if k==raw_field}
        if 'unit_conversions' in selected:
            selected['unit_conversions']={k:v for k,v in selected['unit_conversions'].items() if k==field}
        return {'source_profile':selected, 'contract_id':contract['contract_id'],
                'logical_key':contract['logical_key'], 'field':contract['fields'][field]}

    def plan(self, catalog, selection, *, normalized=False):
        original=feature_column_plan(self.source,catalog,selection=selection,normalized=normalized).to_dict()
        original['reference_members']={day:{s:g for s,g in members.items() if s in self._reference_set}
                                       for day,members in original['reference_members'].items()}
        original['reference_ref']=digest({'original_reference_ref':original['reference_ref'],
                                         'reference_selection':self.reference_security_ids})
        return FeaturePlan.from_dict(original)

    def _selected_leaf_refs(self, fields, days, securities):
        wanted={(field,day) for field in fields for day in days}
        require(wanted<=set(self._leaf_refs), 'used source leaf evidence absent')
        require(all(self._leaf_refs[key][0]==tuple(securities) for key in wanted), 'source leaf security axes mismatch')
        return {field:{day:self._leaf_refs[field,day][1] for day in sorted(days)} for field in sorted(fields)}

    def _selected_buffers(self, field, indices):
        """One bounded physical field/day fingerprint; rolling windows combine refs."""
        j=('open','high','low','close','amount_cny','factor').index(field)
        rows=[]
        for i in indices:
            key=field,i
            if key not in self._buffer_refs:
                buffers={k:_array_ref(self.source.arrays[k][i,:,j]) for k in
                    ('values','original_values','validity','available','available_validity','reasons')}
                self.metrics['physical_field_day_hashes']+=1
                self.metrics['physical_hash_bytes']+=sum(self.source.arrays[k][i,:,j].nbytes for k in buffers)
                self._buffer_refs[key]=digest(buffers)
            rows.append([self.source.manifest['spec']['read_sessions'][i],self._buffer_refs[key]])
        return {'contract_version':'ordered_field_day_buffers_v1','field':field,'days':rows}

    def _reference_days(self, indices):
        import numpy as np
        source=self.source; days=source.manifest['spec']['read_sessions']; rows=[]
        for i in indices:
            if i not in self._reference_refs:
                member=source.arrays['member'][i].copy()
                member &= np.asarray([s in self._reference_set for s in source.manifest['spec']['universe']],dtype=bool)
                self._reference_refs[i]=digest({'member':_array_ref(member),
                    'available':_array_ref(source.arrays['member_available'][i]),
                    'available_validity':_array_ref(source.arrays['member_available_validity'][i]),
                    'leaf':self._selected_leaf_refs({'is_member'},{days[i]},source.manifest['spec']['universe'])})
            rows.append([days[i],self._reference_refs[i]])
        return {'selection':self.reference_security_ids,'days':rows}

    def dependency_refs(self, definitions, days, securities):
        import numpy as np
        self.store.check(); source=self.source; spec=source.manifest['spec']; positions=source.positions
        require(securities==spec['universe'] and set(days)<=set(spec['feature_sessions']),
                'requested axes outside admitted physical source')
        result={}
        for day in days:
            result[day]={}
            for name, definition in definitions.items():
                key=(definition['semantic_ref'],day)
                if key in self._dependencies:
                    result[day][name]=self._dependencies[key]; continue
                semantic=definition['semantic']; needed=_needed({'input_schema':[v['schema'] for v in semantic['inputs'].values()],
                    'nodes':semantic['nodes'],'outputs':[semantic['output']]},[positions[day]])
                selected={}; field_days={}
                for field in definition['inputs']:
                    ix=sorted(needed[field])
                    field_days.setdefault(field,set()).update(spec['read_sessions'][i] for i in ix)
                    selected[field]=deepcopy(self._selected_buffers(field,ix))
                    if field!='amount_cny':
                        factor_ix=sorted(set(ix)|{positions[day]})
                        field_days.setdefault('factor',set()).update(spec['read_sessions'][i] for i in factor_ix)
                        selected[field]['factor']=self._selected_buffers('factor',factor_ix)
                receipt={'selected_buffers':selected,'leaf_revisions':{
                    field:self._selected_leaf_refs({field},needed_days,securities)[field]
                    for field,needed_days in field_days.items()},
                    'output_session':day,'anchor_session':day,'cutoff':spec['cutoff_by_session'][day],
                    'pit_policy':spec['pit_policy']}
                if definition['reference_dependent']:
                    ref_ix=sorted(set().union(*(needed[n['name']] for n in semantic['nodes'] if n['op'] in CS)))
                    receipt['reference']=self._reference_days(ref_ix)
                result[day][name]=digest(receipt)
                self._dependencies[key]=result[day][name]
        self.store.check()
        return result

    def prepare(self, plan, day, securities, dependencies):
        import numpy as np
        p=plan.to_dict(); history=max(required_history(plan).values())+1
        inputs,_=self.source.core_inputs(day,calendar_ref=p['calendar_ref'],reference_ref=p['reference_ref'],
            schema=p['input_schema'],history_sessions=history)
        inputs['reference_member']=inputs['reference_member'] & np.asarray(
            [s in self._reference_set for s in securities],dtype=bool)[None,:]
        inputs['input_refs'].update(feature_dependency_ref=digest(dependencies),
            reference_selection_ref=digest(self.reference_security_ids))
        self.store.check(); return inputs

    def close(self): self.source.arrays.clear(); self.store.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


class CatalogFeatureLibrary:
    """Definitions stay in Research; stable storage/execution belongs to Engine."""
    def __init__(self, library, source, catalog):
        self.library, self.source, self.catalog=library,source,catalog

    def ensure(self, *, selection, output_sessions, sample_universe=None, normalized=False, progress=None):
        with (self.library.execution_session() if self.library._session is None else nullcontext(self.library)):
            return self._ensure(selection=selection,output_sessions=output_sessions,
                sample_universe=sample_universe,normalized=normalized,progress=progress)

    def _ensure(self, *, selection, output_sessions, sample_universe=None, normalized=False, progress=None):
        require(self.catalog.to_dict()['input_contract']=={
            'price_basis':'common_anchor_adjusted_v1','adjustment_anchor':'feature_session_visible_at_cutoff',
            'amount_unit':'CNY'}, 'unsupported catalog input interpretation')
        plan=self.source.plan(self.catalog,selection,normalized=False)
        physical=tuple(self.source.source.manifest['spec']['universe'])
        request=FeatureRequest(plan,tuple(output_sessions),physical,self.source.input_semantics(),
            None if sample_universe is None else tuple(sample_universe))
        raw, raw_stats=self.library.ensure_features(request,self.source,progress=progress)
        if not normalized: return raw, {'raw':raw_stats,'cross_sectional':{'core_calls':0}}
        with self.library.open_feature_view(raw) as raw_view:
            adapter=_RawCrossSectionSource(raw_view,self.source)
            full=self.source.plan(self.catalog,selection,normalized=True).to_dict()
            raw_definitions=feature_definitions(plan,input_semantics=self.source.input_semantics())
            names=[f['id'] for f in self.catalog.select(selection)]
            full['input_schema']=[{**raw_definitions[n]['schema'],'stage':'fact'} for n in names]
            full['nodes']=[n for n in full['nodes'] if n['name'] in {name+'_cs_zscore' for name in names}]
            # Definition templates bind the current admitted source only. Each
            # actual gap projects its own saved raw lineage before Core compile.
            full['sources']=self.source.source.source_bindings()
            semantics={n:{'derived_feature':raw_definitions[n]['semantic']} for n in names}
            cs_plan=FeaturePlan.from_dict(full)
            cs_request=FeatureRequest(cs_plan,tuple(output_sessions),physical,semantics,
                None if sample_universe is None else tuple(sample_universe))
            normalized_view,stats=self.library.ensure_features(cs_request,adapter,progress=progress)
        return normalized_view,{'raw':raw_stats,'cross_sectional':stats}


class _RawCrossSectionSource:
    def __init__(self, raw_view, source):
        self.raw_view,self.source=raw_view,source
        self._tiles={(t['session'],t['column']):t['partition'] for t in raw_view._view['tiles']}

    def source_bindings(self, day, columns):
        reference=self.source.source.source_bindings()[-1]
        bindings={reference['id']:reference}
        for column in columns:
            for binding in self._tiles[day,column]['source_bindings'][column]:
                require(binding['id'] not in bindings or bindings[binding['id']]==binding, 'conflicting cached raw source ID')
                bindings[binding['id']]=binding
        return sorted(bindings.values(),key=lambda b:b['id'])

    def execution_plan(self, plan, day, securities, dependencies):
        p=plan.to_dict(); p['sources']=self.source_bindings(day,set(dependencies))
        return FeaturePlan.from_dict(p)

    def dependency_refs(self, definitions, days, securities):
        import numpy as np
        refs=self._tiles
        physical=self.source.source; result={}
        for day in days:
            i=physical.positions[day]
            reference=self.source._reference_days([i])
            result[day]={name:digest({'raw_semantic_ref':refs[day,name]['semantic_refs'][name],
                'raw_dependency_ref':refs[day,name]['dependency_refs'][name], 'reference':reference,
                'cutoff':physical.manifest['spec']['cutoff_by_session'][day],'anchor':day}) for name in definitions}
        return result

    def prepare(self, plan, day, securities, dependencies):
        import numpy as np
        p=plan.to_dict(); names=[c['name'] for c in p['input_schema']]
        raw=self.raw_view.project([(s,day) for s in securities],names)
        original=self.source.source
        base,_=original.core_inputs(day,calendar_ref=p['calendar_ref'],reference_ref=p['reference_ref'],
            schema=[{'name':f,'dtype':'float64','unit':'CNY' if f=='amount_cny' else 'CNY/share',
                     'stage':'fact','missing':'preserve'} for f in ('open','high','low','close','amount_cny')],history_sessions=1)
        # Core requires actual original source IDs, including the reference source.
        bindings={b['id']:b for b in raw['source_bindings']}
        sets=deepcopy(raw['source_sets'])
        require(all(sets[int(c)] for c in np.unique(raw['source_codes'])), 'raw constant-only source not supported in CS profile')
        # Drop unused empty zero set and remap actual codes to Core's nonempty sets.
        codes=raw['source_codes']; nonempty=[i for i,s in enumerate(sets) if s]
        remap=np.zeros(len(sets),dtype='int32'); admitted=[sets[i] for i in nonempty]
        for new,old in enumerate(nonempty): remap[old]=new
        reference_set=base['source_sets'][int(base['reference_source_codes'][0,0])]
        for binding in base['source_bindings']:
            if binding['id'] in reference_set: bindings[binding['id']]=binding
        if reference_set not in admitted: admitted.append(reference_set)
        reference_code=admitted.index(reference_set)
        require(np.all(raw['available_at_utc_us'] != raw['null_available_at_utc_us']), 'clockless raw source not supported in CS profile')
        base.update(schema=p['input_schema'],source_bindings=p['sources'],sessions=[day],
            values={name:raw['values'][:,j][None,:] for j,name in enumerate(names)},
            validity={name:raw['validity'][:,j][None,:] for j,name in enumerate(names)},
            reason_codes={name:raw['reason_codes'][:,j][None,:] for j,name in enumerate(names)},
            available_at_utc_us={name:raw['available_at_utc_us'][:,j][None,:] for j,name in enumerate(names)},
            source_codes={name:remap[codes[:,j]][None,:] for j,name in enumerate(names)},
            source_sets=admitted,reason_dictionary=raw['reason_dictionary'])
        expected={b['id']:b for b in p['sources']}
        require(all(expected.get(key)==value for key,value in bindings.items()),
                'raw/reference source bindings differ from frozen CS plan')
        base['reference_member']=base['reference_member'] & np.asarray([s in self.source._reference_set for s in securities],dtype=bool)[None,:]
        base['reference_source_codes']=np.full((1,len(securities)),reference_code,dtype='int32')
        base['input_refs'].update(raw_view_ref=self.raw_view.identity,feature_dependency_ref=digest(dependencies))
        return base
