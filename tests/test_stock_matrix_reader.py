"""Small immutable v2 fixtures: admission, keyed projection and lifetime."""
from copy import deepcopy
from pathlib import Path
import gc
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest, _read, file_digest, write_json
from axiom_research.stock_batch import load_stock_ml_batch_inputs
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_reader import (load_matrix_input_projection, VerifiedMatrixStore,
                                                _validate_staged_matrix_batch)
from axiom_research.stock_matrix_storage import write_part, write_buffer
from axiom_research.stock_matrix_prepare import prepare_stock_ml_batch_inputs
from axiom_research.stock_feature_inputs import build_stock_feature_inputs
from test_stock_matrix_prepare import PrepareFeatureFixture, PublicDataFixture, Query
from test_stock_feature_inputs import NoData


def sealed(value,key):
    return seal({k:v for k,v in value.items() if k!=key},key)


def rebatch(manifest):
    value={k:v for k,v in manifest.items() if k not in ('batch_ref','content_digest')}
    return seal({**value,'batch_ref':digest(value)},'content_digest')


def saved_core_pair(metadata):
    for key,value in metadata['contents'].items():
        if type(value) is not dict: continue
        if set(value)=={'input','buffers'}: return key,deepcopy(value),False
        if set(value)=={'path','file_digest','core_input_artifact_ref'}:
            child=_read(value['path'])
            return key,{k:deepcopy(child[k]) for k in ('input','buffers')},True
    raise AssertionError('fixture saved Core input missing')


def replace_core_pair(metadata,key,pair,external,temp):
    metadata['contents'].pop(key)
    value=write_part(temp,sealed(pair,'core_input_artifact_ref'),'core_input_artifact_ref') if external else pair
    metadata['contents'][digest(value)]=value


def retained_owned_bytes(value):
    """Independent walk of Reader-owned Python objects, excluding mmap pages."""
    seen=set()
    def visit(item):
        if id(item) in seen: return 0
        seen.add(id(item)); total=sys.getsizeof(item)
        if type(item) is dict: total+=sum(visit(k)+visit(v) for k,v in item.items())
        elif type(item) in (list,tuple,set): total+=sum(visit(v) for v in item)
        elif type(item).__module__=='axiom_research.stock_matrix_reader': total+=visit(vars(item))
        return total
    return visit(value)


class MatrixReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(); cls.root=Path(cls.temporary.name)
        fixture=PrepareFeatureFixture(cls.root); saved=fixture.matrix(); data=PublicDataFixture(fixture.spec)
        module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
        with patch.dict(sys.modules,{'axiom_data':module}), \
             patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
             patch('axiom_research.stock_ml._environment',return_value=fixture.environment):
            cls.manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=fixture.folds(),
                destination=cls.root/'prepared',preparation_options={'row_block_sessions':32,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'})
        saved.close(); cls.fixture=fixture

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_continuous_folds_do_not_hash_decode_or_rebuild_common_index(self):
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            initial=batch.metrics
            self.assertEqual(initial['common_key_index_builds'],1)
            self.assertEqual(initial['unique_descriptor_admissions'],initial['file_hash_calls'])
            self.assertEqual(initial['raw_cache_live_bytes'],0)
            self.assertGreater(initial['raw_cache_peak_bytes'],0)
            self.assertLess(initial['raw_cache_peak_entries'],initial['raw_cache_admissions'])
            self.assertEqual(initial['training_rows_ref_builds'],4)
            self.assertEqual(initial['training_rows_streamed'],4*122)
            # Core input files are consumed once, then only their mapped
            # buffers and fingerprints remain in this verified batch.
            from axiom_research.stock_batch import _data
            state=_data(batch)['matrix_state']
            children=[d for d in state.store.descriptors.values() if 'core_input_artifact_ref' in d]
            self.assertEqual(len(children),4)
            self.assertTrue(all(d['path'] not in state.store.json for d in children))
            for f in self.manifest['folds']:
                with batch._matrix_project(f['input_manifest'],f['fold_spec']) as p:
                    self.assertEqual(p.X.shape,(122,6)); self.assertEqual(p.y.shape,(122,))
                    self.assertEqual(p.P.shape,(6,6)); self.assertEqual(len(p.features['rows']),9)
                    self.assertEqual(p.training_keys,sorted(p.training_keys,key=lambda k:(k[1],k[0])))
                    self.assertEqual(p.X.tolist(),[[float(i+1) for i in range(6)]]*122)
                    self.assertEqual(len(p.evaluation['rows']),9)
                    self.assertEqual(p.features['contract_version'],'stock_feature_slice_v3')
                    self.assertNotIn('training_rows',p.features)
                    self.assertIn('LABEL_NOT_MATURE',p.excluded)
            final=batch.metrics
            for key in ('file_hash_calls','json_decode_calls','unique_descriptor_admissions','mmap_opens','common_key_index_builds'):
                self.assertEqual(final[key],initial[key],key)
            self.assertEqual(final['fold_projection_calls'],4)
            self.assertGreater(final['projection_bytes'],initial['projection_bytes'])

    def test_final_admission_releases_proofs_and_accounts_actual_retained_graph(self):
        from axiom_research.stock_batch import _data
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            value=_data(batch); state=value['matrix_state']; store=state.store
            self.assertFalse(hasattr(state,'core_wrappers'))
            self.assertEqual(store.json,{}); self.assertEqual(store._decoded,{})
            self.assertEqual(store.content,{})
            self.assertTrue(all('_source_selection_rows' not in b for b in state.feature._blocks))
            # Descriptor/fingerprint controls, the required view/index, native
            # compact arrays and batch manifest are all actually retained.
            self.assertTrue(store.descriptors); self.assertTrue(store.fingerprints)
            self.assertTrue(state.view['core_results']); self.assertTrue(state.feature._index['partitions'])
            self.assertGreaterEqual(store.resident_bytes,retained_owned_bytes(value))
            self.assertEqual(batch.metrics['resident_metadata_bytes'],store.resident_bytes)
            initial=batch.metrics; resident=store.resident_bytes
            for fold in self.manifest['folds']:
                with batch._project_evaluation(fold['input_manifest'],fold['fold_spec']) as lease:
                    self.assertEqual(len(lease.evaluation['rows']),9)
                    self.assertGreater(store.lease_bytes,0)
                    self.assertEqual(store.borrowers,1)
                self.assertEqual(store.resident_bytes,resident)
                self.assertEqual(store.lease_bytes,0); self.assertEqual(store.borrowers,0)
            for metric in ('file_hash_calls','json_decode_calls','common_key_index_builds'):
                self.assertEqual(batch.metrics[metric],initial[metric])

    def test_actual_retained_graph_rejects_tight_budget_and_allows_adequate_budget(self):
        from axiom_research.stock_batch import _data
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            actual=retained_owned_bytes(_data(batch)); reported=batch.metrics['resident_metadata_bytes']
        # A previous compact-only ledger admitted a graph larger than its
        # limit. This bound targets the actual post-admission owned graph.
        # Native-carrier preflight may reject before final graph admission.
        with self.assertRaisesRegex(ValueError,'admitted matrix resident byte budget|Legacy JSON parent workspace budget'):
            load_stock_ml_batch_inputs(self.manifest,limits={
                'maximum_source_bytes':16*1024**2,'maximum_matrix_bytes':actual-1024})
        with load_stock_ml_batch_inputs(self.manifest,limits={
                # Admission also reserves temporary native-proof workspace.
                'maximum_source_bytes':16*1024**2,'maximum_matrix_bytes':max(reported+1024**2,64*1024**2)}) as batch:
            state=_data(batch)['matrix_state']; fold=self.manifest['folds'][0]
            self.assertGreaterEqual(state.store.resident_bytes,retained_owned_bytes(_data(batch)))
            with batch._project_evaluation(fold['input_manifest'],fold['fold_spec']) as lease:
                self.assertEqual(lease.features['contract_version'],'stock_feature_slice_v3')
            self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)

    def test_close_rejects_active_projection_and_then_releases(self):
        batch=load_stock_ml_batch_inputs(self.manifest); f=self.manifest['folds'][0]
        p=batch._matrix_project(f['input_manifest'],f['fold_spec'])
        with self.assertRaisesRegex(ValueError,'still borrowed'): batch.close()
        self.assertEqual(p.X.shape,(122,6)); p.close(); p.close()
        for name in ('X','y','P','features','labels','common','feature_rows','training_keys','candidate_keys','evaluation','excluded','raw_refs'):
            self.assertIsNone(getattr(p,name),name)
        batch.close()
        with self.assertRaisesRegex(ValueError,'closed'): batch.metrics

    def test_gc_and_explicit_close_release_the_same_lease_once(self):
        from axiom_research.stock_batch import _data
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            store=_data(batch)['matrix_state'].store; fold=self.manifest['folds'][0]
            projection=batch._matrix_project(fold['input_manifest'],fold['fold_spec'])
            lease=store.lease_bytes
            self.assertGreater(lease,0); self.assertEqual(store.borrowers,1)
            del projection; gc.collect()
            self.assertEqual(store.lease_bytes,0); self.assertEqual(store.borrowers,0)
            store.maximum_matrix_bytes=store.resident_bytes+lease+1024
            projection=batch._matrix_project(fold['input_manifest'],fold['fold_spec'])
            projection.close(); projection.close(); del projection; gc.collect()
            self.assertEqual(store.lease_bytes,0); self.assertEqual(store.borrowers,0)

    def test_independent_fold_projection_matches_verified_batch(self):
        f=self.manifest['folds'][0]
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            with batch._matrix_project(f['input_manifest'],f['fold_spec']) as a, \
                 load_matrix_input_projection(f['input_manifest'],f['fold_spec']) as b:
                self.assertEqual(a.X.tolist(),b.X.tolist()); self.assertEqual(a.y.tolist(),b.y.tolist())
                self.assertEqual(a.P.tolist(),b.P.tolist()); self.assertEqual(a.features,b.features)
                self.assertEqual(a.labels,b.labels); self.assertEqual(a.training_rows_ref,b.training_rows_ref)
        self.assertTrue(b._store.closed)

    def test_selector_transplant_duplicate_and_missing_keys_rejected(self):
        manifest=deepcopy(self.manifest)
        inputs=manifest['folds'][0]['input_manifest']
        inputs['selectors']['training']=manifest['folds'][1]['input_manifest']['selectors']['training']
        manifest['folds'][0]['input_manifest']=sealed(inputs,'input_ref')
        with self.assertRaisesRegex(ValueError,'selector binding'): load_stock_ml_batch_inputs(rebatch(manifest))
        for change in ('duplicate','missing'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                manifest=deepcopy(self.manifest); inputs=manifest['folds'][0]['input_manifest']
                selector=_read(inputs['selectors']['training']['path'])
                import numpy as np
                offsets=np.fromfile(selector['payload']['path'],dtype='<u8').tolist()
                offsets=offsets+[offsets[-1]] if change=='duplicate' else offsets[1:]
                selector['payload']=write_buffer(temp,offsets,dtype='uint64_le',shape=[len(offsets)])
                selector['row_count']=len(offsets)
                if change=='missing':
                    view=_read(manifest['prepared_view']['path']); row_index=_read(view['row_index']['path']); width=len(row_index['security_ids'])
                    selector['keys_digest']=digest([[row_index['security_ids'][i%width],row_index['sessions'][i//width]] for i in offsets])
                inputs['selectors']['training']=write_part(temp,sealed(selector,'selector_ref'),'selector_ref')
                manifest['folds'][0]['input_manifest']=sealed(inputs,'input_ref')
                with self.assertRaisesRegex(ValueError,'ordered selector|identical keys'): load_stock_ml_batch_inputs(rebatch(manifest))

    def _bad_view(self, temp, *, clock=False,source=False):
        manifest=deepcopy(self.manifest); view=_read(manifest['prepared_view']['path'])
        part=next(p for p in view['partitions'] if p['table']=='training_normalized_labels')
        metadata=_read(part['metadata']['path'])
        if clock: metadata['rows'][0]['normalized_available_at']='2099-01-01T00:00:00Z'
        if source: metadata['rows'][0]['normalization_source_refs']=[digest('foreign source')]
        part['metadata']=write_part(temp,sealed(metadata,'metadata_ref'),'metadata_ref')
        part.update(partition_ref=digest({k:v for k,v in part.items() if k!='partition_ref'}))
        descriptor=write_part(temp,sealed(view,'prepared_view_ref'),'prepared_view_ref')
        manifest['prepared_view']=descriptor
        for f in manifest['folds']:
            f['input_manifest']['prepared_view']=descriptor
            f['input_manifest']=sealed(f['input_manifest'],'input_ref')
        return rebatch(manifest)

    def test_resealed_clock_and_source_forgery_rejected(self):
        for kind in ('clock','source'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                bad=self._bad_view(temp,clock=kind=='clock',source=kind=='source')
                with self.assertRaisesRegex(ValueError,'Target/Core|output source'): load_stock_ml_batch_inputs(bad)

    def test_resealed_core_source_cannot_substitute_query_for_exact_cohort(self):
        """All hashes are consistent; a reachable Query still is not eligibility."""
        from axiom_research.stock_matrix_reader import validate_saved_core_result
        with tempfile.TemporaryDirectory() as temp:
            manifest=deepcopy(self.manifest); view=_read(manifest['prepared_view']['path'])
            part=next(p for p in view['partitions'] if p['table']=='training_normalized_labels')
            metadata=_read(part['metadata']['path']); old_ref=metadata['core_result_refs'][0]
            core_desc=next(d for d in view['core_results'] if _read(d['path'])['result']['metadata']['result_ref']==old_ref)
            wrapper=deepcopy(_read(core_desc['path']))
            old_input_key,core_input,external=saved_core_pair(metadata)
            raw=_read(metadata['raw_build']['path']); query_ref=digest(raw['source_evidence']['context']['query'])
            day=metadata['rows'][0]['feature_session']; inp=core_input['input']
            binding=next(b for b in inp['source_bindings_by_session'][day]['bindings'] if b['id']=='offline_eligibility')
            binding['view_ref']=query_ref
            result=wrapper['result']; result['source_bindings_by_session'][day]['bindings']=deepcopy(inp['source_bindings_by_session'][day]['bindings'])
            result['metadata']['input_ref']=digest(inp)
            result['metadata']['source_ref']=digest({'source_bindings_by_session':inp['source_bindings_by_session'],
                'fact_source_codes':inp['fact_source_codes'],'reference_source_codes':inp['reference_source_codes']})
            result_without_ref=deepcopy(result); result_without_ref['metadata'].pop('result_ref')
            result['metadata']['result_ref']=digest(result_without_ref); new_ref=result['metadata']['result_ref']
            wrapper=sealed(wrapper,'core_result_artifact_ref')
            # This is a valid neutral Core closure. Research must separately
            # insist that offline eligibility binds the current admitted cohort.
            with VerifiedMatrixStore() as store: validate_saved_core_result(wrapper,store,core_input=core_input)
            new_desc=write_part(temp,wrapper,'core_result_artifact_ref')
            view['core_results']=[new_desc if d==core_desc else d for d in view['core_results']]
            for p in view['partitions']:
                if p['table']!='training_normalized_labels': continue
                m=_read(p['metadata']['path'])
                if old_ref not in m['core_result_refs']: continue
                m['core_result_refs']=[new_ref if ref==old_ref else ref for ref in m['core_result_refs']]
                replace_core_pair(m,old_input_key,core_input,external,temp)
                for row in m['rows']:
                    row['normalization_source_refs']=sorted(new_ref if r==old_ref else r for r in row['normalization_source_refs'])
                p['metadata']=write_part(temp,sealed(m,'metadata_ref'),'metadata_ref')
                p.update(partition_ref=digest({k:v for k,v in p.items() if k!='partition_ref'}))
            view_desc=write_part(temp,sealed(view,'prepared_view_ref'),'prepared_view_ref'); manifest['prepared_view']=view_desc
            for f in manifest['folds']:
                inputs=f['input_manifest']; inputs['prepared_view']=view_desc
                inputs['core_result_refs']=[new_ref if r==old_ref else r for r in inputs['core_result_refs']]
                for role,desc in inputs['selectors'].items():
                    if desc is None: continue
                    selector=_read(desc['path']); selector['prepared_view_ref']=view_desc['prepared_view_ref']
                    inputs['selectors'][role]=write_part(temp,sealed(selector,'selector_ref'),'selector_ref')
                f['input_manifest']=sealed(inputs,'input_ref')
            with self.assertRaisesRegex(ValueError,'eligibility source differs from current cohort'):
                load_stock_ml_batch_inputs(rebatch(manifest))

    def test_resealed_target_schema_names_and_units_rejected(self):
        for table in ('training_raw_labels','training_normalized_labels','evaluation_raw_labels'):
            for field,value in (('name','unapproved_target'),('unit','CNY/share')):
                with self.subTest(table=table,field=field),tempfile.TemporaryDirectory() as temp:
                    manifest=deepcopy(self.manifest); view=_read(manifest['prepared_view']['path'])
                    view['schema'][table][0][field]=value; view['schema_digest']=digest(view['schema'])
                    for p in view['partitions']:
                        if p['table']!=table: continue
                        p['schema_digest']=digest(view['schema'][table])
                        p['columns']=[view['schema'][table][0]['name']]
                        p.update(partition_ref=digest({k:v for k,v in p.items() if k!='partition_ref'}))
                    descriptor=write_part(temp,sealed(view,'prepared_view_ref'),'prepared_view_ref')
                    manifest['prepared_view']=descriptor
                    for f in manifest['folds']:
                        inputs=f['input_manifest']; inputs['prepared_view']=descriptor
                        for role,desc in inputs['selectors'].items():
                            if desc is None: continue
                            selector=_read(desc['path']); selector['prepared_view_ref']=descriptor['prepared_view_ref']
                            selector['schema_digest']=view['schema_digest']
                            inputs['selectors'][role]=write_part(temp,sealed(selector,'selector_ref'),'selector_ref')
                        f['input_manifest']=sealed(inputs,'input_ref')
                    bad=rebatch(manifest)
                    # Every affected partition, view, selector, input and batch
                    # identity is valid; this fails at the fixed profile boundary.
                    with self.assertRaisesRegex(ValueError,'fixed Target table schema required: '+table):
                        load_stock_ml_batch_inputs(bad)

    def test_resealed_saved_core_target_schema_names_and_units_rejected(self):
        from axiom_research.stock_matrix_reader import validate_saved_core_result
        view=_read(self.manifest['prepared_view']['path'])
        part=next(p for p in view['partitions'] if p['table']=='training_normalized_labels')
        metadata=_read(part['metadata']['path']); ref=metadata['core_result_refs'][0]
        original=_read(next(d['path'] for d in view['core_results'] if _read(d['path'])['result']['metadata']['result_ref']==ref))
        _,saved_input,_=saved_core_pair(metadata)
        for column in ('schema','output_schema'):
            for field,value in (('name','unapproved_target'),('unit','CNY/share')):
                with self.subTest(column=column,field=field):
                    wrapper=deepcopy(original); core_input=deepcopy(saved_input); inp=core_input['input']
                    inp[column][0][field]=value; result=wrapper['result']; meta=result['metadata']
                    result['schema']=deepcopy(inp['output_schema'])
                    meta['input_ref']=digest(inp)
                    meta['schema_ref']=digest({'schema':inp['schema'],'output_schema':inp['output_schema']})
                    meta['numeric_input_ref']=digest({'keys_ref':meta['keys_ref'],'schema_ref':meta['schema_ref'],
                        'spec_ref':meta['spec_ref'],'reason_dictionary':inp['reason_dictionary'],
                        **{k:inp[k] for k in ('values','value_validity','value_reason_codes','reference_member')}})
                    output=deepcopy(result); output['metadata'].pop('result_ref'); meta['result_ref']=digest(output)
                    wrapper=sealed(wrapper,'core_result_artifact_ref')
                    with VerifiedMatrixStore() as store:
                        with self.assertRaisesRegex(ValueError,'fixed (Target Core input/output schemas|normalized Target output schema) required'):
                            validate_saved_core_result(wrapper,store,core_input=core_input)

    def test_fully_sealed_valid_feature_with_unknown_training_clock_rejected(self):
        """A faulty producer can seal a complete DAG; admission still owns PIT."""
        from axiom_research.stock_fold_inputs import require
        from axiom_research.stock_feature_inputs import load_stock_feature_inputs
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture=PrepareFeatureFixture(root); original_day=fixture.day
            first_training=fixture.calendar[fixture.calendar.index(fixture.folds()[0]['fit_session'])-65]
            def unknown_clock(day):
                rows,proof=original_day(day)
                if day==first_training: rows[0]['availability']=[None]*6
                return rows,proof
            fixture.day=unknown_clock; saved=fixture.matrix()
            # The inherited Feature contract intentionally preserves unknown
            # availability even with finite values and explicit validity.
            with load_stock_feature_inputs(saved.path) as feature:
                offset=feature._row_index['sessions'].index(first_training)*3
                row=feature.row_metadata([offset])[0]
                self.assertTrue(all(row['validity'])); self.assertEqual(row['availability'],[None]*6)
            data=PublicDataFixture(fixture.spec); module=types.ModuleType('axiom_data')
            module.QuerySpec=Query; module.adjust_prices=data.adjust
            def faulty_producer(condition,message):
                if message!='training Feature clock exceeds fit': require(condition,message)
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=fixture.environment), \
                 patch('axiom_research.stock_matrix_prepare.require',side_effect=faulty_producer), \
                 patch('axiom_research.stock_matrix_reader._training_feature_clock',return_value=None):
                manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=fixture.folds()[:1],
                    destination=root/'bad-prepared',preparation_options={'row_block_sessions':32,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'})
            saved.close()
            with self.assertRaisesRegex(ValueError,'training Feature clock exceeds fit'):
                load_stock_ml_batch_inputs(manifest)
            with patch('axiom_research.stock_matrix_reader._training_feature_clock',return_value=None):
                batch=load_stock_ml_batch_inputs(manifest)
            with batch:
                f=manifest['folds'][0]
                with self.assertRaisesRegex(ValueError,'training Feature clock exceeds fit'):
                    batch._matrix_project(f['input_manifest'],f['fold_spec'])

    def test_outcome_anchor_cannot_follow_last_native_query_session(self):
        from axiom_research.stock_matrix_reader import _outcome_query
        from axiom_research.labels import _query_context
        view=_read(self.manifest['prepared_view']['path'])
        p=next(p for p in view['partitions'] if p['table']=='training_raw_labels')
        raw=_read(_read(p['metadata']['path'])['raw_build']['path'])
        context=deepcopy(raw['source_evidence']['context']); query=context['query']
        anchor=self.fixture.calendar[-1]
        self.assertGreater(anchor,query['sessions'][-1])
        query['adjustment_anchor']=anchor; context['derivation']['anchor_session']=anchor
        factors=context['derivation']['factor_query']; factors['sessions']=sorted(set(query['sessions'])|{anchor})
        factors['cutoff_by_session']={d:next(iter(query['cutoff_by_session'].values())) for d in factors['sessions']}
        for validator in (_outcome_query,_query_context):
            with self.subTest(validator=validator.__name__),self.assertRaisesRegex(ValueError,'anchor|derivation'):
                validator(context,self.fixture.calendar)

    def test_same_fit_disjoint_folds_share_raw_parent_without_readmission(self):
        from axiom_research.stock_batch import _data
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture=PrepareFeatureFixture(root); saved=fixture.matrix()
            first=fixture.folds()[0]; second=deepcopy(first)
            pos=fixture.calendar.index(first['fit_session'])
            second['oos_trade_sessions']=fixture.calendar[pos+4:pos+7]
            second['inference_cutoff_by_session']={d:d+'T21:00:00+08:00' for d in fixture.calendar[pos+3:pos+6]}
            data=PublicDataFixture(fixture.spec); module=types.ModuleType('axiom_data')
            module.QuerySpec=Query; module.adjust_prices=data.adjust
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=fixture.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=fixture.environment):
                manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=[first,second],
                    destination=root/'same-fit',preparation_options={'row_block_sessions':32,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'})
            saved.close(); view=_read(manifest['prepared_view']['path'])
            raw_paths={}
            for p in view['partitions']:
                if p['table']=='training_raw_labels':
                    raw_paths.setdefault(p['fold_spec_ref'],set()).add(_read(p['metadata']['path'])['raw_build']['path'])
            self.assertTrue(set.intersection(*raw_paths.values()))
            with load_stock_ml_batch_inputs(manifest) as batch:
                initial=batch.metrics; state=_data(batch)['matrix_state']
                for path in set.intersection(*raw_paths.values()):
                    self.assertIn(path,state.store.descriptors); self.assertNotIn(path,state.store.json)
                    self.assertEqual(len([key for key in state.raw_provenance if key[0]==path]),1)
                for f in manifest['folds']:
                    with batch._matrix_project(f['input_manifest'],f['fold_spec']) as p: self.assertEqual(p.X.shape,(122,6))
                for metric in ('file_hash_calls','json_decode_calls','common_key_index_builds'):
                    self.assertEqual(batch.metrics[metric],initial[metric])

    def test_resealed_raw_types_are_checked_before_compact_coercion(self):
        from axiom_research.stock_matrix_reader import _RawAdmission
        from axiom_research.stock_fold_inputs import validate_raw
        view=_read(self.manifest['prepared_view']['path']); common=view['definition']
        partition=next(p for p in view['partitions'] if p['table']=='training_raw_labels')
        metadata=_read(partition['metadata']['path']); original=_read(metadata['raw_build']['path'])
        self.assertIs(original['rows'][0]['valid'],True)
        for field,value,message in (('valid',1,'validity must be bool'),
            ('invalid_reason',1,'reason must be text or null'),('label_available_at',1,'clock/session must be text or null'),
            ('source_refs',original['source_ref'],'source refs must be unique digest list'),
            ('return',True,'return must be finite or null')):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                raw=deepcopy(original); raw['rows'][0][field]=value
                descriptor=write_part(temp,sealed(raw,'label_ref'),'label_ref')
                with VerifiedMatrixStore() as store:
                    saved=store.read_json(descriptor,'label_ref')
                    if field=='valid':
                        # The inherited validator accepts numeric truthiness.
                        # The new compact boundary must not turn it into bool.
                        indexed=validate_raw(saved,common,metadata['cutoff'])
                    else: indexed={(r['security_id'],r['feature_session']):r for r in saved['rows']}
                    with self.assertRaisesRegex(ValueError,message):
                        _RawAdmission(descriptor,saved,indexed,common,store.np)

    def test_original_inline_core_inputs_remain_readable(self):
        from axiom_research.stock_batch import _data
        with tempfile.TemporaryDirectory() as temp:
            manifest=deepcopy(self.manifest); view=_read(manifest['prepared_view']['path'])
            for p in view['partitions']:
                if p['table']!='training_normalized_labels': continue
                metadata=_read(p['metadata']['path']); key,pair,external=saved_core_pair(metadata)
                self.assertTrue(external); replace_core_pair(metadata,key,pair,False,temp)
                p['metadata']=write_part(temp,sealed(metadata,'metadata_ref'),'metadata_ref')
                p.update(partition_ref=digest({k:v for k,v in p.items() if k!='partition_ref'}))
            descriptor=write_part(temp,sealed(view,'prepared_view_ref'),'prepared_view_ref')
            manifest['prepared_view']=descriptor
            for f in manifest['folds']:
                inputs=f['input_manifest']; inputs['prepared_view']=descriptor
                for role,desc in inputs['selectors'].items():
                    if desc is None: continue
                    selector=_read(desc['path']); selector['prepared_view_ref']=descriptor['prepared_view_ref']
                    inputs['selectors'][role]=write_part(temp,sealed(selector,'selector_ref'),'selector_ref')
                f['input_manifest']=sealed(inputs,'input_ref')
            with load_stock_ml_batch_inputs(rebatch(manifest)) as batch:
                self.assertFalse(any('core_input_artifact_ref' in d for d in _data(batch)['matrix_state'].store.descriptors.values()))
                for f in manifest['folds']:
                    with batch._matrix_project(f['input_manifest'],f['fold_spec']) as p:
                        self.assertEqual(p.X.tolist(),[[float(i+1) for i in range(6)]]*122)
                        self.assertEqual(p.P.shape,(6,6)); self.assertEqual(len(p.y),122)

    def test_valid_oos_leaves_require_actual_endpoints_and_native_clocks_without_io(self):
        from axiom_research.stock_matrix_reader import _label_leaves
        view=_read(self.manifest['prepared_view']['path'])
        partition=next(p for p in view['partitions'] if p['table']=='evaluation_raw_labels')
        metadata=_read(partition['metadata']['path']); original=next(r for r in metadata['rows'] if r['valid'])
        selected=next(v for v in metadata['contents'].values() if type(v) is dict and set(v)=={'context','records','field_meta'})
        anchor=selected['context']['query']['adjustment_anchor']
        def project(row,wire,store):
            return _label_leaves([row],wire,raw_ref=metadata['raw_build']['label_ref'],source_ref=digest(wire),
                query_ref=digest(wire['context']['query']),anchor=anchor,store=store,contents={})
        for case in ('start_null','end_null','record_missing','metadata_missing','zero','bool','text',
                     'missing_reason','provenance_key','provenance_clock','label_clock'):
            with self.subTest(case=case),VerifiedMatrixStore() as store:
                row=deepcopy(original); wire=deepcopy(selected); key=row['security_id'],row['start_session']
                record=next(r for r in wire['records'] if (r['security_id'],r['session'])==key)
                item=next(r for r in wire['field_meta']['open']['by_key'] if (r['security_id'],r['session'])==key)
                if case=='start_null': row['start_session']=None
                elif case=='end_null': row['end_session']=None
                elif case=='record_missing': wire['records'].remove(record)
                elif case=='metadata_missing': wire['field_meta']['open']['by_key'].remove(item)
                elif case in ('zero','bool','text'): record['open']={'zero':0,'bool':True,'text':'10.0'}[case]
                elif case=='missing_reason': item['missing_reason']='missing_price'
                elif case=='provenance_key': item['factor_provenance']['security_id']='FOREIGN'
                elif case=='provenance_clock': item['price_provenance']['usable_from']='2099-01-01T00:00:00Z'
                elif case=='label_clock': row['label_available_at']='2000-01-01T00:00:00Z'
                with patch('axiom_research.stock_matrix_reader.file_digest',side_effect=AssertionError('leaf read file')), \
                     patch('axiom_research.stock_matrix_reader._read',side_effect=AssertionError('leaf decoded file')):
                    with self.assertRaisesRegex(ValueError,'valid Label Target'): project(row,wire,store)
                self.assertEqual(store.metrics['file_hash_calls'],0); self.assertEqual(store.metrics['json_decode_calls'],0)
        # Incomplete invalid targets preserve null endpoints and the exact
        # observed leaf, including its original missing/null source metadata.
        with VerifiedMatrixStore() as store:
            row=deepcopy(original); row.update({'valid':False,'return':None,'start_session':None,'invalid_reason':'incomplete'})
            value=project(row,deepcopy(selected),store)
            self.assertIsNone(value[0]['start_open_ref']); self.assertIsNotNone(value[0]['end_close_ref'])

    def test_oos_evaluation_owns_rows_and_never_projects_training(self):
        from axiom_research.stock_batch import _data
        from axiom_research.stock_matrix_folds import _dataset
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            state=_data(batch)['matrix_state']; fold=self.manifest['folds'][0]
            with batch._matrix_project(fold['input_manifest'],fold['fold_spec']) as p:
                expected_features=deepcopy(p.features); expected_evaluation=deepcopy(p.evaluation)
                expected_binding={'dataset':deepcopy(_dataset(p,fold['input_manifest'],fold['fold_spec'])),
                    'labels':deepcopy(p.labels)}
            initial=batch.metrics; expected_offsets=state.selectors[digest(fold['input_manifest']),digest(fold['fold_spec'])]['inference']
            original_project=state.feature.project
            def oos_only(offsets,columns=None):
                self.assertEqual(list(offsets),list(expected_offsets)); return original_project(offsets,columns)
            with patch.object(state,'project',side_effect=AssertionError('training fold projected')), \
                 patch.object(state.feature,'project',side_effect=oos_only), \
                 patch('axiom_research.stock_matrix_reader._stream_ref',side_effect=AssertionError('training rows hashed')):
                value=batch._matrix_evaluation(fold['input_manifest'],fold['fold_spec'])
                self.assertEqual(value['features'],expected_features); self.assertEqual(value['evaluation'],expected_evaluation)
                self.assertEqual(value['saved_fold_binding'],expected_binding)
                self.assertEqual(value['batch_ref'],batch.identity)
                self.assertEqual(value['inputs'],fold['input_manifest']); self.assertEqual(value['spec'],fold['fold_spec'])
                self.assertEqual(set(value),{'contract_version','input_ref','batch_ref','inputs','spec','prepared_view_ref',
                    'fold_spec_ref','common','features','evaluation','raw_provenance','training_raw_provenance',
                    'label_leaf_rows','label_leaf_contents','saved_fold_binding','source_records','source_records_ref',
                    'source_paths','evaluation_input_ref'})
                self.assertEqual(value['evaluation_input_ref'],digest({k:v for k,v in value.items() if k not in ('source_records','evaluation_input_ref')}))
                self.assertEqual(set(value['common']),{'scope','snapshot','pit_policy','calendar','universe',
                    'catalog_ref','feature_selection','ordered_features'})
                self.assertIs(value['source_records'],state.source_records); self.assertIsInstance(value['source_records'],tuple)
                self.assertEqual(value['source_records'],tuple(sorted(state.store._hashes.items())))
                self.assertEqual(value['source_records_ref'],digest(value['source_records']))
                self.assertIs(value['source_paths'],state.source_paths[digest(fold['fold_spec'])])
                self.assertEqual(value['source_paths'],tuple(sorted(set(value['source_paths']))))
                self.assertTrue(set(value['source_paths'])<=set(dict(value['source_records'])))
                for other_part in state.view['partitions']:
                    if other_part['table']!='features' and other_part['fold_spec_ref']!=digest(fold['fold_spec']):
                        self.assertNotIn(other_part['metadata']['path'],value['source_paths'])
                with self.assertRaises(TypeError): value['source_records'][0][1]='caller_mutation'
                self.assertTrue(value['raw_provenance'])
                self.assertEqual({p['label_ref'] for p in value['training_raw_provenance']},set(expected_binding['dataset']['raw_label_refs']))
                self.assertTrue({p['label_ref'] for p in value['training_raw_provenance']}.isdisjoint(
                    {p['label_ref'] for p in value['raw_provenance']}))
                for provenance in value['raw_provenance']:
                    self.assertEqual(set(provenance),{'raw_build','contract_version','label_spec','calendar_ref',
                        'source_ref','source_evidence','label_ref'})
                    self.assertEqual(provenance['label_ref'],provenance['raw_build']['label_ref'])
                    self.assertNotIn('records',provenance['source_evidence'])
                leaf_refs={r[name] for r in value['label_leaf_rows'] for name in ('start_open_ref','end_close_ref') if r[name] is not None}
                self.assertEqual(set(value['label_leaf_contents']),leaf_refs)
                self.assertEqual([(r['security_id'],r['feature_session']) for r in value['label_leaf_rows']],
                    [(r['security_id'],r['feature_session']) for r in value['evaluation']['rows']])
                for ref,leaf in value['label_leaf_contents'].items():
                    self.assertEqual(ref,digest(leaf)); self.assertEqual(set(leaf),{'session','value','field_meta'})
                # Independently reconstruct leaves from the saved actual
                # selected wire, including all three native provenance arms.
                for leaf_row in value['label_leaf_rows']:
                    part=next(p for p in state.view['partitions'] if p['table']=='evaluation_raw_labels' and
                        p['fold_spec_ref']==digest(fold['fold_spec']) and any(r['security_id']==leaf_row['security_id'] and
                        r['feature_session']==leaf_row['feature_session'] for r in _read(p['metadata']['path'])['rows']))
                    metadata=_read(part['metadata']['path'])
                    wire=next(v for v in metadata['contents'].values() if type(v) is dict and set(v)=={'context','records','field_meta'})
                    target=next(r for r in value['evaluation']['rows'] if r['security_id']==leaf_row['security_id'] and
                        r['feature_session']==leaf_row['feature_session'])
                    for field,name,session in (('open','start_open_ref',target['start_session']),('close','end_close_ref',target['end_session'])):
                        if session is None:
                            self.assertIsNone(leaf_row[name]); continue
                        record=next(r for r in wire['records'] if r['security_id']==leaf_row['security_id'] and r['session']==session)
                        item=next(r for r in wire['field_meta'][field]['by_key'] if r['security_id']==leaf_row['security_id'] and r['session']==session)
                        expected={'session':session,'value':record[field],
                            'field_meta':{**{k:v for k,v in wire['field_meta'][field].items() if k!='by_key'},'by_key':[item]}}
                        self.assertEqual(value['label_leaf_contents'][leaf_row[name]],expected)
                value['features']['rows'][0]['values'][0]=987
                value['evaluation']['rows'][0]['return']=987
                value['raw_provenance'][0]['label_spec']['normalization']='caller_mutation'
                value['training_raw_provenance'][0]['label_spec']['normalization']='caller_mutation'
                value['saved_fold_binding']['dataset']['normalization']['operator']='caller_mutation'
                first_leaf=next(iter(value['label_leaf_contents'])); value['label_leaf_contents'][first_leaf]['value']=987
                again=batch._matrix_evaluation(fold['input_manifest'],fold['fold_spec'])
                self.assertEqual(again['features'],expected_features); self.assertEqual(again['evaluation'],expected_evaluation)
                self.assertEqual(again['raw_provenance'][0]['label_spec']['normalization'],'none')
                self.assertEqual(again['saved_fold_binding'],expected_binding)
                self.assertNotEqual(again['label_leaf_contents'][first_leaf]['value'],987)
                self.assertIs(again['source_records'],value['source_records'])
                other=self.manifest['folds'][1]
                expected_offsets=state.selectors[digest(other['input_manifest']),digest(other['fold_spec'])]['inference']
                third=batch._matrix_evaluation(other['input_manifest'],other['fold_spec'])
                self.assertIs(third['source_records'],again['source_records'])
                self.assertNotEqual(third['evaluation_input_ref'],again['evaluation_input_ref'])
                with self.assertRaisesRegex(ValueError,'outside saved batch'):
                    batch._matrix_evaluation(fold['input_manifest'],other['fold_spec'])
            self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)
            self.assertEqual(batch.metrics['fold_projection_calls'],initial['fold_projection_calls'])
            self.assertEqual(batch.metrics['evaluation_projection_calls'],3)
            for metric in ('file_hash_calls','json_decode_calls','common_key_index_builds','training_rows_ref_builds','training_rows_streamed','source_records_ref_builds'):
                self.assertEqual(batch.metrics[metric],initial[metric])
            state.store.maximum_matrix_bytes=state.store.resident_bytes+100
            with patch.object(state.feature,'project',side_effect=AssertionError('allocated before preflight')):
                with self.assertRaisesRegex(ValueError,'budget exceeded'):
                    batch._matrix_evaluation(fold['input_manifest'],fold['fold_spec'])
        self.assertEqual(again['features'],expected_features)
        self.assertEqual(again['evaluation'],expected_evaluation)
        self.assertEqual(again['source_records'][0],value['source_records'][0])
        with self.assertRaisesRegex(ValueError,'closed'):
            batch._matrix_evaluation(fold['input_manifest'],fold['fold_spec'])

    def test_oos_evaluation_checks_source_before_and_after_owned_copy(self):
        from axiom_research.stock_batch import _data
        for during in (False,True):
            with self.subTest(during=during),load_stock_ml_batch_inputs(self.manifest) as batch:
                state=_data(batch)['matrix_state']; fold=self.manifest['folds'][0]
                path=Path(state.view['source_selection']['path']); mark=path.stat()
                def change(): os.utime(path,ns=(mark.st_atime_ns,mark.st_mtime_ns+10**6))
                original_project=state.feature.project
                def change_after_copy(offsets,columns=None):
                    value=original_project(offsets,columns); change(); return value
                try:
                    if not during: change()
                    with patch.object(state.feature,'project',side_effect=change_after_copy if during else original_project):
                        with self.assertRaisesRegex(ValueError,'saved matrix source changed'):
                            batch._matrix_evaluation(fold['input_manifest'],fold['fold_spec'])
                    self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)
                    self.assertEqual(batch.metrics['evaluation_projection_calls'],0)
                finally: os.utime(path,ns=(mark.st_atime_ns,mark.st_mtime_ns))

    def test_owner_evaluation_lease_matches_goldens_and_releases_all_payloads(self):
        from axiom_research.stock_batch import _data
        batch=load_stock_ml_batch_inputs(self.manifest); state=_data(batch)['matrix_state']
        fold=self.manifest['folds'][0]; value=batch._matrix_evaluation(fold['input_manifest'],fold['fold_spec'])
        source_table=batch._evaluation_source_records(); self.assertIs(source_table,state.source_records)
        self.assertIs(batch._evaluation_source_records(),source_table)
        initial=batch.metrics; infer=state.selectors[digest(fold['input_manifest']),digest(fold['fold_spec'])]['inference']
        original_project=state.feature.project
        def oos_only(offsets,columns=None):
            self.assertEqual(list(offsets),list(infer)); return original_project(offsets,columns)
        with patch.object(state,'project',side_effect=AssertionError('projected training')), \
             patch.object(state.feature,'project',side_effect=oos_only), \
             patch('axiom_research.stock_matrix_reader._stream_ref',side_effect=AssertionError('hashed training rows')):
            lease=batch._project_evaluation(fold['input_manifest'],fold['fold_spec'])
        fields={'common','features','labels','evaluation','raw_provenance','label_leaf_bindings','fold_binding','source_record_indices'}
        self.assertEqual({k for k in vars(lease) if not k.startswith('_')},fields)
        self.assertFalse(hasattr(lease,'X')); self.assertFalse(hasattr(lease,'y')); self.assertFalse(hasattr(lease,'P'))
        self.assertEqual(lease.common,value['common']); self.assertEqual(lease.features,value['features'])
        self.assertEqual(lease.labels,value['saved_fold_binding']['labels']); self.assertEqual(lease.evaluation,value['evaluation'])
        self.assertEqual(lease.raw_provenance,value['raw_provenance'])
        self.assertEqual(lease.label_leaf_bindings,{'rows':value['label_leaf_rows'],'contents':value['label_leaf_contents']})
        self.assertEqual(lease.fold_binding,{**value['saved_fold_binding'],'training_raw_provenance':value['training_raw_provenance']})
        self.assertIs(lease.source_record_indices,state.source_record_indices[digest(fold['fold_spec'])])
        self.assertEqual(tuple(source_table[i][0] for i in lease.source_record_indices),value['source_paths'])
        self.assertGreater(state.store.lease_bytes,0); self.assertEqual(state.store.borrowers,1)
        with self.assertRaisesRegex(ValueError,'still borrowed'): batch.close()
        owned_copy=deepcopy(lease.features); lease.close(); lease.close()
        for name in fields: self.assertIsNone(getattr(lease,name),name)
        self.assertEqual(owned_copy,value['features']); self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)
        for metric in ('file_hash_calls','json_decode_calls','common_key_index_builds','training_rows_ref_builds',
                       'training_rows_streamed','source_records_ref_builds','source_record_index_builds','fold_projection_calls'):
            self.assertEqual(batch.metrics[metric],initial[metric])
        kept=[lease]
        for f in self.manifest['folds']:
            projection=batch._project_evaluation(f['input_manifest'],f['fold_spec']); projection.close(); kept.append(projection)
        self.assertTrue(all(p.features is None and p.fold_binding is None for p in kept))
        projection=batch._project_evaluation(fold['input_manifest'],fold['fold_spec']); del projection; gc.collect()
        self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)
        batch.close()
        self.assertEqual(owned_copy,value['features'])
        with self.assertRaisesRegex(ValueError,'closed'): batch._evaluation_source_records()
        with self.assertRaisesRegex(ValueError,'closed'): batch._project_evaluation(fold['input_manifest'],fold['fold_spec'])

    def test_owner_oos_lease_preflight_rejects_before_array_allocation(self):
        from axiom_research.stock_batch import _data
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            state=_data(batch)['matrix_state']; fold=self.manifest['folds'][0]
            state.store.maximum_matrix_bytes=state.store.resident_bytes+100
            with patch.object(state.feature,'project',side_effect=AssertionError('allocated OOS matrix')):
                with self.assertRaisesRegex(ValueError,'OOS evaluation matrix byte budget'):
                    batch._project_evaluation(fold['input_manifest'],fold['fold_spec'])
            self.assertEqual(state.store.borrowers,0); self.assertEqual(state.store.lease_bytes,0)

    def test_global_budget_includes_resident_metadata_before_projection(self):
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            state=__import__('axiom_research.stock_batch',fromlist=['_data'])._data(batch)['matrix_state']
            state.store.maximum_matrix_bytes=state.store.resident_bytes+100
            f=self.manifest['folds'][0]
            with self.assertRaisesRegex(ValueError,'active fold matrix'): batch._matrix_project(f['input_manifest'],f['fold_spec'])
        with self.assertRaisesRegex(ValueError,'compact metadata budget|matrix resident accounting workspace budget'):
            load_stock_ml_batch_inputs(self.manifest,limits={'maximum_source_bytes':16*1024**2,'maximum_matrix_bytes':100})

    def test_public_loader_and_projection_import_no_provider_core_or_model(self):
        path=self.root/'test-batch.json'; write_json(path,self.manifest)
        code='''import builtins,json,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'axiom_data','axiom_engine','qlib','lightgbm','pandas','pyarrow'}:
        raise AssertionError('forbidden read-only import '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
from axiom_research import load_stock_ml_batch_inputs
manifest=json.load(open(sys.argv[1]))
with load_stock_ml_batch_inputs(manifest) as batch:
    value=batch._matrix_evaluation(manifest['folds'][0]['input_manifest'],manifest['folds'][0]['fold_spec'])
    assert value['saved_fold_binding']['dataset']['training_row_count']==122
    assert len(value['label_leaf_rows'])==9
    with batch._project_evaluation(manifest['folds'][0]['input_manifest'],manifest['folds'][0]['fold_spec']) as lease:
        assert lease.fold_binding['dataset']==value['saved_fold_binding']['dataset']
        assert not hasattr(lease,'X')
    with batch._matrix_project(manifest['folds'][0]['input_manifest'],manifest['folds'][0]['fold_spec']) as p:
        print(json.dumps([list(p.X.shape),list(p.P.shape),batch.metrics['core_calls']]))
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(path)],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
            capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),[[122,6],[6,6],0])

    def test_raw_buffer_bytes_shared_across_shapes_hash_once(self):
        with tempfile.TemporaryDirectory() as temp:
            desc=write_buffer(temp,[1.0,2.0,3.0,4.0],dtype='float64_le',shape=[2,2])
            flat={**desc,'shape':[4]}
            with VerifiedMatrixStore() as store:
                self.assertEqual(store.buffer(desc).shape,(2,2)); self.assertEqual(store.buffer(flat).shape,(4,))
                self.assertEqual(store.metrics['file_hash_calls'],1)
                self.assertEqual(store.metrics['unique_descriptor_admissions'],1)

    def test_source_change_rejects_use_but_close_still_releases_mmap(self):
        with tempfile.TemporaryDirectory() as temp:
            desc=write_buffer(temp,[1.0,2.0],dtype='float64_le',shape=[2])
            store=VerifiedMatrixStore(); store.buffer(desc)
            Path(desc['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'source changed'): store.check()
            store.close(); self.assertTrue(store.closed); self.assertFalse(store.arrays)

    def test_staging_admission_validates_final_refs_before_publication(self):
        target=Path(self.manifest['prepared_view']['path']).parents[1]
        stage=target.parent/'reader-test-stage'
        target.rename(stage)
        try:
            self.assertFalse(target.exists())
            metrics=_validate_staged_matrix_batch(self.manifest,stage=stage,target=target)
            self.assertEqual(metrics['common_key_index_builds'],1)
            self.assertEqual(metrics['core_calls'],0)
            self.assertFalse(target.exists())
            view=_read(stage/Path(self.manifest['prepared_view']['path']).relative_to(target))
            buffer=next(p for p in view['partitions'] if p['table']=='training_normalized_labels')['buffers']['values']
            physical=stage/Path(buffer['path']).relative_to(target)
            old=physical.read_bytes(); mark=physical.stat()
            physical.write_bytes(old+b'bad')
            try:
                with self.assertRaisesRegex(ValueError,'file digest mismatch'):
                    _validate_staged_matrix_batch(self.manifest,stage=stage,target=target)
                self.assertFalse(target.exists())
            finally:
                physical.write_bytes(old); os.utime(physical,ns=(mark.st_atime_ns,mark.st_mtime_ns))
        finally: stage.rename(target)

    def test_v1_feature_index_is_exact_ancestry_of_new_prepared_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f=PrepareFeatureFixture(root); f.data=NoData()
            with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=f.catalog), \
                 patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                 patch('axiom_research.stock_ml._prepare_stock_qlib',side_effect=f.prepare), \
                 patch('axiom_research.stock_ml._iter_stock_feature_days',side_effect=f.iterate):
                saved=build_stock_feature_inputs(f.data,spec=f.spec,destination=root/'v1',shard_sessions=2)
            self.assertEqual(saved.to_dict()['contract_version'],'stock_feature_inputs_v1')
            before={str(p):(file_digest(p),p.stat().st_mtime_ns) for p in saved.path.rglob('*') if p.is_file()}
            data=PublicDataFixture(f.spec); module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=f.environment):
                manifest=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=f.folds()[:1],destination=root/'prepared',
                    preparation_options={'row_block_sessions':32,'column_block':32,'maximum_resident_bytes':64*1024**2,
                                         'normalization_backend':'core_cs_batch_v1'})
            self.assertEqual(manifest['definition']['feature_inputs']['feature_inputs_ref'],saved.identity)
            with load_stock_ml_batch_inputs(manifest) as batch:
                fold=manifest['folds'][0]
                with batch._matrix_project(fold['input_manifest'],fold['fold_spec']) as p:
                    self.assertEqual(p.X.shape,(122,6)); self.assertEqual(p.P.shape,(6,6))
                self.assertEqual(batch.metrics['common_key_index_builds'],1)
            self.assertEqual(before,{str(p):(file_digest(p),p.stat().st_mtime_ns) for p in saved.path.rglob('*') if p.is_file()})


if __name__=='__main__': unittest.main()
