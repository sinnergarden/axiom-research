"""Research definitions and thin adapters for Engine's incremental Feature store.

Saved source admission stays with its existing Research owner. The configured
universe uses the admitted daily membership and vendor selection policy. This
adapter uses original Data adjustment and Core DAG/CS arithmetic.
"""
from copy import deepcopy
from collections.abc import Mapping
from contextlib import nullcontext
from hashlib import sha256
import json
import sys
import tempfile
import zlib

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


def normalized_feature_dependency_ref(source, day, raw_semantic_ref, raw_dependency_ref):
    physical=source.source
    return digest({'raw_semantic_ref':raw_semantic_ref,'raw_dependency_ref':raw_dependency_ref,
        'reference':source._reference_days([physical.positions[day]]),
        'cutoff':physical.manifest['spec']['cutoff_by_session'][day],'anchor':day})


def admitted_raw_dependency_refs(view, source, definitions, days, securities):
    """Validate each original raw block, then return topology-independent refs."""
    expected=source.dependency_refs(definitions,days,securities)
    scoped={tuple(securities):expected}
    for day in days:
        receipts=view.column_receipts(day,list(definitions))
        for name,definition in definitions.items():
            for block in receipts[name]['blocks']:
                axis=tuple(block['security_ids'])
                if axis not in scoped:
                    scoped[axis]=source.dependency_refs(definitions,days,list(axis))
                actual=scoped[axis][day][name]
                require(block['semantic_ref']==definition['semantic_ref'] and block['dependency_ref']==actual,
                        'Feature/source local dependency or reference mismatch')
    return expected


class _SourceArrays(Mapping):
    """Borrow a physical buffer only when its field is actually requested."""
    _readonly_buffers=True

    def __init__(self, manifest, store):
        self.descriptors=manifest['buffers']; self.store=store; self.cache={}

    def __iter__(self): return iter(self.descriptors)
    def __len__(self): return len(self.descriptors)
    def __getitem__(self, name):
        if name not in self.cache:
            desc=self.descriptors[name]; value=self.store.buffer(desc)
            self.cache[name]=value.view('bool') if desc['dtype']=='bool_u8' else value
        else: self.store.check_path(self.descriptors[name]['path'])
        return self.cache[name]
    def clear(self): self.cache.clear()


class SavedColumnFeatureSource:
    """Admit an immutable source through the original owner, then borrow mmap."""
    def __init__(self, descriptor, *, reference_security_ids=None, maximum_source_bytes=8*1024**3,
                 maximum_working_bytes=512*1024**2):
        self._leaf_cache_bytes=0
        self._leaf_spool=None
        self.store=VerifiedMatrixStore(maximum_source_bytes=maximum_source_bytes,
            maximum_matrix_bytes=maximum_working_bytes,
            _caller_retained_bytes=lambda:self._leaf_cache_bytes)
        self._leaf_refs={}; self._dependencies={}; self._buffer_refs={}; self._reference_refs={}
        self._leaf_subsets={}; self._leaf_axes={}; self._axis_positions={}
        self.metrics={'physical_field_day_hashes':0,'physical_hash_bytes':0}
        manifest,_=verify_feature_column_source(descriptor,store=self.store,
            _evidence_days=set(),_verify_buffers=False)
        from .stock_feature_inputs import source_universe_binding
        binding=source_universe_binding(manifest,reader=self.store.read_json)
        require(manifest.get('universe_binding')==binding or
                manifest.get('universe_binding') is None and manifest['qlib_view'].get('universe_query') is not None,
                'explicit saved source universe binding required; publish a scope-bound descriptor')
        self.universe_id=binding['universe_id']
        self.descriptor=deepcopy(descriptor)
        arrays=_SourceArrays(manifest,self.store)
        self.source=FeatureColumnSource(manifest,arrays)
        physical=manifest['spec']['universe']
        self.reference_security_ids=list(physical if reference_security_ids is None else reference_security_ids)
        require(self.reference_security_ids==sorted(set(self.reference_security_ids)) and
                set(self.reference_security_ids)<=set(physical), 'explicit reference subset of admitted source required')
        self._reference_set=set(self.reference_security_ids)

    def _leaf_axis(self, axis):
        if axis not in self._leaf_axes:
            self._charge_leaf_cache(256+sys.getsizeof(axis)+sum(sys.getsizeof(s) for s in axis))
            self._leaf_axes[axis]=axis
        return self._leaf_axes[axis]

    def _charge_leaf_cache(self, amount):
        require(self._leaf_cache_bytes+amount<=self.store.maximum_matrix_bytes,
                'Feature leaf evidence cache exceeds maximum_working_bytes')
        self._leaf_cache_bytes+=amount

    def _retain_leaf(self, mapping, key, value):
        require(key not in mapping,'duplicate Feature leaf cache entry')
        # Container/key overhead plus direct scalar/packed payloads. Shared
        # security tuples are charged once by _leaf_axis, not once per day.
        amount=512+sum(sys.getsizeof(v) for v in key if type(v) is str)
        amount+=sum(sys.getsizeof(v) for v in value if type(v) in (str,bytes,int)) if type(value) is tuple else sys.getsizeof(value)
        self._charge_leaf_cache(amount);mapping[key]=value

    def _capture_leaf_refs(self, child):
        if child['contract_version']=='reader_column_block_v1':
            manifest=self.source.manifest
            for field,metadata in child['field_meta'].items():
                symbols=tuple(manifest.get('financial_materializations',{}).get(field,{}).get(
                    'coverage_security_ids',manifest['spec']['universe']))
                symbols=self._leaf_axis(symbols)
                for day in child['sessions']:
                    key=(field,day)
                    if key not in self._leaf_refs:
                        self._retain_leaf(self._leaf_refs,key,(symbols,metadata['source_binding'],
                            metadata['dtype'],metadata['unit'],None,0))
            return
        if self._leaf_spool is None:self._leaf_spool=tempfile.TemporaryFile(mode="w+b")
        base=self.store._native_caller_bytes(child)-self._leaf_cache_bytes
        original_bytes=max(self.store.fingerprints[path][2] for path,descriptor in self.store.descriptors.items()
                           if descriptor.get("evidence_ref")==child["evidence_ref"])
        for field, metadata in child['field_meta'].items():
            workspace=1024**2+16*original_bytes+32*len(metadata['by_key'])+256*len(child['sessions'])
            require(base+self._leaf_cache_bytes+workspace<=self.store.maximum_matrix_bytes,
                    'Feature leaf encoding workspace exceeds maximum_working_bytes')
            grouped={day:[] for day in child['sessions']}
            for row in metadata['by_key']:
                require(row['session'] in grouped, 'source provenance outside declared chunk')
                grouped[row['session']].append(row)
            for day,rows in grouped.items():
                rows=sorted(rows,key=lambda r:r['security_id'])
                symbols=tuple(r['security_id'] for r in rows)
                require(len(symbols)==len(set(symbols)), 'duplicate source provenance key')
                symbols=self._leaf_axis(symbols)
                # Keep day-local exact evidence for future subset admission,
                # without retaining the decoded per-security provenance objects.
                # The complete admitted parent byte size bounds this one-day
                # encode, hash copies and zlib workspace before allocation.
                raw=json.dumps(rows,separators=(',',':')).encode()
                raw_bytes=len(raw);packed=zlib.compress(raw);raw=None
                offset=self._leaf_spool.tell();self._leaf_spool.write(packed)
                stored=(offset,len(packed));packed=None
                self._retain_leaf(self._leaf_refs,(field,day),(symbols,digest({'dtype':metadata['dtype'],
                    'unit':metadata['unit'],'selected_revisions':rows}),metadata['dtype'],metadata['unit'],stored,raw_bytes))
                require(base+self._leaf_cache_bytes+workspace<=self.store.maximum_matrix_bytes,
                        'Feature leaf evidence cache and source exceed maximum_working_bytes')

    def capabilities(self):
        return {'contract_version':'feature_source_capabilities_v1','source_ref':self.source.identity,
            'universe_profile':self.universe_id,
            'pit_policies':[self.source.manifest['spec']['pit_policy']],
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
        for field,meta in self.source.manifest.get('daily_fields',{}).items():
            result[field]={'value_projection':'boolean_0_1_binary64_v1' if meta['dtype']=='bool' else 'native_binary64_v1',
                'source_profile':self._field_profile(profiles[meta['domain']],field), 'missing':'preserve',
                'clock':'leaf_max_ceil_second_cutoff_floor_second_v1'}
        result['raw_factor']={'value_projection':'native_binary64_v1','source_profile':self._field_profile(profiles['adjustment_factors'],'factor'),'meaning':'native_supplier_adjustment_multiplier; not anchor-normalized price','missing':'preserve','clock':'leaf_max_ceil_second_cutoff_floor_second_v1'}
        result['session_ordinal']={'projection':'proleptic_gregorian_session_date_ordinal_v1','unit':'days','missing':'preserve'}
        if 'up_limit' in result:
            for field in ('raw_open','raw_close'):
                result[field]={'value_projection':'native_binary64_v1','source_profile':self._field_profile(profiles['market_daily'],field[4:]),'missing':'preserve','adjustment':'none','clock':'leaf_max_ceil_second_cutoff_floor_second_v1'}
        for field in self.source.manifest.get('financial_materializations',{}):
            result[field]['value_projection']='Data_financial_derivation_binary64_v1'
        if 'volume_shares' in result:
            result['vwap']={'formula':'original_amount_cny / volume_shares',
                'value_projection':'native_binary64_v1','missing':'preserve',
                'amount_profile':self._field_profile(profiles['market_daily'],'amount_cny'),
                'volume_profile':result['volume_shares']['source_profile'],
                'adjustment':'Data.adjust_prices_output_day_anchor_v1',
                'factor_profile':self._field_profile(profiles['adjustment_factors'],'factor'),
                'clock':'leaf_max_ceil_second_cutoff_floor_second_v1'}
        return result

    @staticmethod
    def _field_profile(profile,field):
        contract=profile['contract']
        selected=deepcopy(profile['source_profile'])
        # Canonical security axes and selected revision rows bind identity maps;
        # adding an unrelated security must not change every field's semantics.
        selected.pop('identity_map',None)
        if selected.get('id')=='research.financial_materialization.v1':
            projection=selected['projections'][field]
            field_inputs=selected.pop('financial_inputs_by_field',None)
            if field_inputs is not None:
                selected['financial_inputs']=field_inputs[field]
            selected['projections']={field:projection}
            selected['financial_inputs']=SavedColumnFeatureSource._field_profile(
                selected['financial_inputs'],projection['source_field'])
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
        missing=wanted-set(self._leaf_refs)
        if missing:
            verify_feature_column_source(self.descriptor,store=self.store,
                _dependency_receipts=self._capture_leaf_refs,
                _evidence_days={d for _,d in missing},
                _evidence_fields={f for f,_ in missing},_verify_buffers=False)
        require(wanted<=set(self._leaf_refs), 'used source leaf evidence absent')
        axis=tuple(securities); axis=self._leaf_axis(axis)
        result={}
        for field in sorted(fields):
            result[field]={}
            for day in sorted(days):
                symbols,full,dtype,unit,packed,raw_bytes=self._leaf_refs[field,day]
                require(set(securities)<=set(symbols), 'source leaf security axes mismatch')
                if packed is None:
                    result[field][day]=digest({'contract_version':'block_input_dependency_v1',
                        'source_binding':full,'field':field,'dtype':dtype,'unit':unit,
                        'security_ids':securities,'session':day,
                        'cutoff':self.source.manifest['spec']['cutoff_by_session'][day]})
                elif axis==symbols: result[field][day]=full
                else:
                    key=(field,day,axis)
                    if key not in self._leaf_subsets:
                        # Bound inflate output and its decoded flat provenance
                        # graph before allocation; saved rows come from admitted JSON.
                        base=self.store._native_caller_bytes()
                        require(base+1024**2+64*raw_bytes<=self.store.maximum_matrix_bytes,
                                'Feature leaf decode workspace exceeds maximum_working_bytes')
                        offset,length=packed
                        require(base+1024**2+64*raw_bytes+length<=self.store.maximum_matrix_bytes,'Feature leaf spool decode budget exceeded')
                        self._leaf_spool.seek(offset);compressed=self._leaf_spool.read(length)
                        require(len(compressed)==length,'Feature leaf spool truncated')
                        decoder=zlib.decompressobj(); raw=decoder.decompress(compressed,raw_bytes+1)
                        compressed=None
                        require(len(raw)==raw_bytes and decoder.eof and not decoder.unused_data,
                                'Feature leaf compressed length mismatch')
                        selected=set(securities);rows=json.loads(raw)
                        reference=digest({'dtype':dtype,'unit':unit,
                            'selected_revisions':[r for r in rows if r['security_id'] in selected]})
                        rows=raw=decoder=None
                        self._retain_leaf(self._leaf_subsets,key,reference)
                    result[field][day]=self._leaf_subsets[key]
        return result

    def _selected_buffers(self, field, indices, securities):
        """One bounded physical field/day fingerprint; rolling windows combine refs."""
        rows=[]
        axis=tuple(securities); axis=self._leaf_axis(axis)
        if axis not in self._axis_positions:
            self._axis_positions[axis]=[self.source.security_positions[s] for s in securities]
        positions=self._axis_positions[axis]
        for i in indices:
            key=field,i,axis
            if key not in self._buffer_refs:
                buffers={k:_array_ref(self.source.field_array(field,k)[i,positions]) for k in
                    ('values','original_values','validity','available','available_validity','reasons')}
                self.metrics['physical_field_day_hashes']+=1
                self.metrics['physical_hash_bytes']+=sum(self.source.field_array(field,k)[i,positions].nbytes for k in buffers)
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
                    **{k:_array_ref(source.arrays[k][i]) for k in ('member_known','member_pool_complete','member_reasons') if k in source.arrays},
                    'available':_array_ref(source.arrays['member_available'][i]),
                    'available_validity':_array_ref(source.arrays['member_available_validity'][i]),
                    'leaf':self._selected_leaf_refs({'is_member'}|({'industry_code'} if 'industry_code' in source.physical_fields() else set()),{days[i]},source.manifest['spec']['universe'])})
            rows.append([days[i],self._reference_refs[i]])
        return {'selection':self.reference_security_ids,'days':rows}

    def dependency_refs(self, definitions, days, securities):
        import numpy as np
        self.store.check(); source=self.source; spec=source.manifest['spec']; positions=source.positions
        require(bool(securities) and securities==sorted(set(securities)) and
                set(securities)<=set(spec['universe']) and set(days)<=set(spec['feature_sessions']),
                'requested axes outside admitted physical source')
        axis=tuple(securities); axis=self._leaf_axis(axis)
        result={}
        for day in days:
            result[day]={}
            for name, definition in definitions.items():
                require(not definition['reference_dependent'] or securities==spec['universe'],
                        'CS dependencies require complete physical cohort')
                key=(definition['semantic_ref'],day,axis)
                if key in self._dependencies:
                    result[day][name]=self._dependencies[key]; continue
                source.require_coverage(definition['inputs'],securities)
                semantic=definition['semantic']; needed=_needed({'input_schema':[v['schema'] for v in semantic['inputs'].values()],
                    'nodes':semantic['nodes'],'outputs':[semantic['output']]},[positions[day]])
                selected={}; field_days={}
                for field in definition['inputs']:
                    ix=sorted(needed[field])
                    leaves=self.source.leaf_fields(field)
                    # Preserve old receipt structure byte-for-byte for old fields.
                    first=leaves[0]
                    field_days.setdefault(first,set()).update(spec['read_sessions'][i] for i in ix)
                    selected[field]=deepcopy(self._selected_buffers(first,ix,securities))
                    for leaf in leaves[1:]:
                        leaf_ix=sorted(set(ix)|{positions[day]}) if leaf=='factor' else ix
                        field_days.setdefault(leaf,set()).update(spec['read_sessions'][i] for i in leaf_ix)
                        selected[field][leaf]=self._selected_buffers(leaf,leaf_ix,securities)
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
            schema=p['input_schema'],history_sessions=history,securities=securities)
        names=[c['name'] for c in p['input_schema']]
        for field in ('values','validity','reason_codes','available_at_utc_us','source_codes'):
            inputs[field]={name:inputs[field][name] for name in names}
        chosen=sorted(set().union(*(set(int(code) for code in np.unique(inputs['source_codes'][name]))
                                    for name in names),
                                 set(int(code) for code in np.unique(inputs['reference_source_codes']))))
        sets=inputs['source_sets']; remap=np.zeros(len(sets),dtype='int32')
        for code,old in enumerate(chosen): remap[old]=code
        inputs['source_sets']=[sets[i] for i in chosen]
        inputs['source_codes']={name:remap[value] for name,value in inputs['source_codes'].items()}
        inputs['reference_source_codes']=remap[inputs['reference_source_codes']]
        bindings={b['id']:b for b in inputs['source_bindings']}
        used={source for group in inputs['source_sets'] for source in group}
        require(all(any(b==bindings[source] for b in p['sources']) for source in used),
                'prepared fact/reference sources differ from frozen gap plan')
        inputs['source_bindings']=p['sources']
        inputs['reference_member']=inputs['reference_member'] & np.asarray(
            [s in self._reference_set for s in securities],dtype=bool)[None,:]
        inputs['input_refs'].update(feature_dependency_ref=digest(dependencies),
            reference_selection_ref=digest(self.reference_security_ids))
        self.store.check(); return inputs

    def execution_plan(self, plan, day, securities, dependencies):
        p=plan.to_dict(); names={c['name'] for c in p['input_schema']}
        fields=self.source.physical_fields()
        require(names<={c['name'] for c in self.source.input_schema()}, 'source fact fields unavailable')
        selected={'is_member'}|set().union(*(self.source.leaf_fields(name) for name in names))
        p['sources']=sorted((binding for field,binding in zip(fields,self.source.source_bindings())
                             if field in selected),key=lambda b:b['id'])
        return FeaturePlan.from_dict(p)

    def close(self):
        self.source.arrays.clear(); self.store.close()
        if self._leaf_spool is not None:self._leaf_spool.close()
        self._leaf_cache_bytes=0
        for cache in (self._leaf_refs,self._leaf_axes,self._leaf_subsets,self._dependencies,
                      self._buffer_refs,self._reference_refs,self._axis_positions): cache.clear()
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

    def source_bindings(self, day, columns):
        reference=self.source.source.source_bindings()[-1]
        bindings={reference['id']:reference}
        receipts=self.raw_view.column_receipts(day,list(columns))
        for column in columns:
            for binding in receipts[column]['source_bindings']:
                require(binding['id'] not in bindings or bindings[binding['id']]==binding, 'conflicting cached raw source ID')
                bindings[binding['id']]=binding
        return sorted(bindings.values(),key=lambda b:b['id'])

    def execution_plan(self, plan, day, securities, dependencies):
        p=plan.to_dict(); names={c['name'] for c in p['input_schema']}
        require(names==set(dependencies), 'CS gap fact closure differs from output dependencies')
        p['sources']=self.source_bindings(day,names)
        return FeaturePlan.from_dict(p)

    def dependency_refs(self, definitions, days, securities):
        raw_definitions={name:self.raw_view.describe()['definitions'][name] for name in definitions}
        physical=self.source.source.manifest['spec']['universe']
        require(securities==physical,'CS dependencies require complete physical cohort')
        raw=admitted_raw_dependency_refs(self.raw_view,self.source,raw_definitions,days,physical)
        return {day:{name:normalized_feature_dependency_ref(self.source,day,
            raw_definitions[name]['semantic_ref'],raw[day][name]) for name in definitions} for day in days}

    def prepare(self, plan, day, securities, dependencies):
        import numpy as np
        p=plan.to_dict(); names=[c['name'] for c in p['input_schema']]
        require(set(names)==set(dependencies), 'CS raw projection differs from gap fact closure')
        raw=self.raw_view.project([(s,day) for s in securities],names)
        original=self.source.source
        from .stock_matrix_storage import instant_us
        spec=original.manifest['spec']; position=original.positions[day]
        require(securities==spec['universe'],'CS preparation requires complete physical cohort')
        cutoff=instant_us(spec['cutoff_by_session'][day])//1000000*1000000
        reference_binding=original.source_bindings()[-1]
        member_clock=np.where(original.arrays['member_available_validity'][position:position+1],
            (original.arrays['member_available'][position:position+1]+999999)//1000000*1000000,cutoff)
        # CS consumes cached raw cells and membership only. Do not adjust prices
        # again merely to construct the reference input envelope.
        base=dict(contract_version='feature_column_input_v1',calendar_ref=p['calendar_ref'],
            reference_ref=p['reference_ref'],security_ids=list(securities),
            cutoffs_utc_us=np.asarray([cutoff],dtype='<i8'),
            reference_member=original.arrays['member'][position:position+1],
            **{target:original.arrays[key][position:position+1] for key,target in
               (('member_known','reference_member_known'),('member_pool_complete','reference_pool_complete')) if key in original.arrays},
            reference_industry=original.industries([position],[original.security_positions[s] for s in securities]),
            reference_available_at_utc_us=member_clock,
            input_refs={'source_ref':original.identity})
        # Core requires actual original source IDs, including the reference source.
        bindings={b['id']:b for b in raw['source_bindings']}
        sets=deepcopy(raw['source_sets'])
        require(all(sets[int(c)] for c in np.unique(raw['source_codes'])), 'raw constant-only source not supported in CS profile')
        # Drop unused empty zero set and remap actual codes to Core's nonempty sets.
        codes=raw['source_codes']; nonempty=[i for i,s in enumerate(sets) if s]
        remap=np.zeros(len(sets),dtype='int32'); admitted=[sets[i] for i in nonempty]
        for new,old in enumerate(nonempty): remap[old]=new
        reference_set=[reference_binding['id']]
        bindings[reference_binding['id']]=reference_binding
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
        require(expected==bindings,
                'raw/reference source bindings differ from frozen CS plan')
        base['reference_member']=base['reference_member'] & np.asarray([s in self.source._reference_set for s in securities],dtype=bool)[None,:]
        base['reference_source_codes']=np.full((1,len(securities)),reference_code,dtype='int32')
        base['input_refs'].update(raw_view_ref=self.raw_view.identity,feature_dependency_ref=digest(dependencies))
        return base
