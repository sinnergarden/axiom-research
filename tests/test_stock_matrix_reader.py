"""Small immutable v2 fixtures: admission, keyed projection and lifetime."""
from copy import deepcopy
from pathlib import Path
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

    def test_close_rejects_active_projection_and_then_releases(self):
        batch=load_stock_ml_batch_inputs(self.manifest); f=self.manifest['folds'][0]
        p=batch._matrix_project(f['input_manifest'],f['fold_spec'])
        with self.assertRaisesRegex(ValueError,'still borrowed'): batch.close()
        self.assertEqual(p.X.shape,(122,6)); p.close(); p.close(); batch.close()
        with self.assertRaisesRegex(ValueError,'closed'): batch.metrics

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
            old_input_key,core_input=next((k,deepcopy(v)) for k,v in metadata['contents'].items() if type(v) is dict and set(v)=={'input','buffers'})
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
                m['contents'].pop(old_input_key); m['contents'][digest(core_input)]=core_input
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

    def test_global_budget_includes_resident_metadata_before_projection(self):
        with load_stock_ml_batch_inputs(self.manifest) as batch:
            state=__import__('axiom_research.stock_batch',fromlist=['_data'])._data(batch)['matrix_state']
            state.store.maximum_matrix_bytes=state.store.resident_bytes+100
            f=self.manifest['folds'][0]
            with self.assertRaisesRegex(ValueError,'active fold matrix'): batch._matrix_project(f['input_manifest'],f['fold_spec'])
        with self.assertRaisesRegex(ValueError,'compact metadata budget'):
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
