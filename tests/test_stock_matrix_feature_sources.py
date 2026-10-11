"""Small compact-source vertical slice; no real facts, fit, inference or account."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import builtins
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_research import load_stock_feature_inputs, load_stock_ml_batch_inputs
from axiom_research.stock_artifacts import _read, digest, write_json
from axiom_research.stock_fold_inputs import seal, feature_available
from axiom_research.stock_matrix_storage import write_part, write_buffer, write_partition, instant_us
from axiom_research.stock_matrix_feature_sources import _day_ref, _selection
from test_stock_matrix_storage import SyntheticWideInputs
from test_stock_matrix_prepare import PrepareFeatureFixture, PublicDataFixture, Query
from test_stock_feature_inputs import InterruptedPreparation
from legacy_stock_matrix_fixture import prepare_saved_v2_fixture


def original_proof(rows, p):
    p['fact_ref']=digest(['synthetic actual fact document',p['session']])
    p['context_ref']=digest(['synthetic actual context document',p['session']])
    for source in p['source_evidence'].values():
        source['provenance_by_key_ref']=digest(['synthetic keyed provenance',source['field'],p['session']])
        source['query_context']['coverage']={'domain':source['query_context']['domain'],
            'evidence':['same original coverage']*40,'nullable':None}
    return rows,p


class SourceFixture(SyntheticWideInputs):
    def day(self, day): return original_proof(*super().day(day))


class FoldSourceFixture(PrepareFeatureFixture):
    def day(self, day): return original_proof(*super().day(day))


def options(*, rows=1, columns=2):
    return {'layout':'matrix_v2','row_block_sessions':rows,'column_block':columns,
            'maximum_resident_bytes':64*1024**2}


def replace_metadata(saved, meta, contexts=None):
    """Reseal the whole v3 graph, so malformed tests reach semantic admission."""
    index=saved.to_dict(); spec=index['definition']['spec']; view=index['qlib_view']
    if contexts is None: contexts={d['query_context_ref']:_read(d['path']) for d in meta['query_contexts']}
    for p in meta['input_evidence']:
        p.pop('day_evidence_ref',None); p['day_evidence_ref']=digest(p)
    refs={}
    for p,out in zip(meta['input_evidence'],meta['day_outputs']):
        refs[p['session']]={'feature_ref':_day_ref(p,out,spec,view),'qlib_view_ref':digest(view)}
    meta['row_references']=refs; meta.pop('metadata_ref',None); md=write_part(saved.path,seal(meta,'metadata_ref'),'metadata_ref')
    old=_read(index['partitions'][0]['metadata']['path'])['sessions']
    for part in index['partitions']:
        if _read(part['metadata']['path'])['sessions']==old:
            part['metadata']=md; part.pop('partition_ref'); part['partition_ref']=digest(part)
    selection=_read(index['source_selection']['path']); by_day={p['session']:p for p in meta['input_evidence']}
    selection['feature_rows']=[_selection(by_day[r['session']],refs[r['session']]['feature_ref'],contexts)
        if r['session'] in by_day else r for r in selection['feature_rows']]
    selection.pop('source_selection_ref'); index['source_selection']=write_part(saved.path,seal(selection,'source_selection_ref'),'source_selection_ref')
    index.pop('content_digest'); index['feature_inputs_ref']=digest({k:index[k] for k in
        ('contract_version','definition_ref','qlib_manifest','schema_digest','row_index','source_selection','partitions')})
    write_json(saved.path/'index.json',seal(index,'content_digest'))


class CompactSourceTests(unittest.TestCase):
    def test_publication_new_blocks_are_measured_once_without_revisiting_history(self):
        import axiom_research.stock_feature_inputs as owner
        measure=owner._object_upper_bytes; visits={}; updates=[]
        def counted(value):
            if type(value) is dict and set(value)>={'start','count','parts','rows','_source_selection_rows'}:
                visits[id(value)]=visits.get(id(value),0)+1
            return measure(value)
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6)
            def progress(update):
                if update['stage']=='feature_matrix_block':
                    updates.append(deepcopy(visits))
                    update['producer_stats']['core_calls']=999
            with patch.object(owner,'_object_upper_bytes',side_effect=counted):
                saved=f.matrix(options=options(),progress=progress)
            try:
                self.assertEqual([len(u) for u in updates],[1,2,3])
                self.assertEqual(list(visits.values()),[1,1,1])
                self.assertIsNone(saved._store._publication_accounting)
                self.assertIsNone(saved._store._caller_retained_bytes)
                self.assertEqual(saved.row_metadata(list(range(9))),[r for d in f.days for r in f.day(d)[0]])
            finally: saved.close()

    def test_publication_store_charges_private_roots_once_and_releases_last_alias(self):
        import axiom_research.stock_matrix_reader as reader
        maximum=4*1024**2
        with tempfile.TemporaryDirectory() as temp, reader.VerifiedMatrixStore(maximum_matrix_bytes=maximum) as store:
            store._begin_publication_accounting()
            desc=write_part(temp,seal({'contract_version':'synthetic_private_json','rows':['first']*80},'metadata_ref'),'metadata_ref')
            first=store.read_json(desc,'metadata_ref'); before=store._native_caller_bytes()
            visits=[]; getsizeof=sys.getsizeof
            def counted(value,*args):
                if value is first or value is first['rows']: visits.append(id(value))
                return getsizeof(value,*args)
            second_desc=write_part(temp,seal({'contract_version':'synthetic_private_json','rows':['second']*80},'metadata_ref'),'metadata_ref')
            with patch.object(reader.sys,'getsizeof',side_effect=counted):
                for _ in range(3): self.assertEqual(store._native_caller_bytes(first),before)
                second=store.read_json(second_desc,'metadata_ref')
                for _ in range(3): store._native_caller_bytes(second)
            self.assertEqual(visits,[])
            self.assertEqual(store._publication_accounting[1][id(first)][2],3)
            self.assertEqual(store._publication_accounting[1][id(second)][2],3)
            with_both=store._native_caller_bytes()
            roots=[*store._publication_maps(),store._native_verified,store.metrics]
            self.assertGreaterEqual(with_both,reader._resident_size(roots))
            store.drop_json(desc,'metadata_ref')
            self.assertNotIn(id(first),store._publication_accounting[1])
            self.assertLess(store._native_caller_bytes(),with_both)
            store.drop_json(second_desc,'metadata_ref')
            self.assertNotIn(id(second),store._publication_accounting[1])
            self.assertEqual(store.json,{}); self.assertEqual(store._decoded,{}); self.assertEqual(store.content,{})

    def test_publication_incremental_budget_includes_external_state_and_replacement(self):
        from axiom_research.stock_matrix_reader import VerifiedMatrixStore
        with VerifiedMatrixStore(maximum_matrix_bytes=2*1024**2) as store:
            store._begin_publication_accounting()
            value={'private':['x']*100}; store._retain_entry(store.json,'private',value)
            before=store._native_caller_bytes()
            store._caller_retained_bytes=lambda:1234
            self.assertEqual(store._native_caller_bytes(),before+1234)
            store.maximum_matrix_bytes=before+1234+1024
            with self.assertRaises(ValueError): store._retain_entry(store.json,'private',{'private':['new']*200})
            self.assertIs(store.json['private'],value)
            store.maximum_matrix_bytes=2*1024**2
            replacement={'private':['replacement']*200}
            store._retain_entry(store.json,'private',replacement)
            self.assertNotIn(id(value),store._publication_accounting[1])
            self.assertIs(store.json['private'],replacement)
            store._release_entry(store.json,'private')
            self.assertEqual(store._publication_accounting[0],{}); self.assertEqual(store._publication_accounting[1],{})
            self.assertEqual(store._publication_accounting[2],0)
            self.assertGreaterEqual(store._native_caller_bytes(),1234)

    def test_public_memory_producer_preserves_original_core_rows_and_proof(self):
        from test_stock_matrix_feature_producer import MemoryFeatureInputs
        from test_stock_matrix_feature_batch import prepared_inputs, arguments
        from axiom_research.stock_matrix_feature_producer import iter_matrix_feature_days
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f=MemoryFeatureInputs(root); prepared=prepared_inputs(f,root)
            with patch('axiom_research.qlib_adapter.QlibView.read',lambda view,**kwargs:f.native_read(view,**kwargs)):
                old=list(iter_matrix_feature_days(f.data,**arguments(f,prepared,stats={})))
                stats={}; args=arguments(f,prepared,stats=stats)
                args['config']={**f.config,'_compact_source':True}
                with patch('axiom_research.stock_matrix_feature_producer._view_signature',side_effect=AssertionError('diagnostic hot path')):
                    new=list(iter_matrix_feature_days(f.data,**args))
            self.assertEqual(new,old)
            self.assertEqual(stats['feature_source_batch_ref_reuses'],2*len(new))
            self.assertEqual(stats['classifier_status_counts'],{})
            self.assertFalse(f.fact_root.exists())

    def test_cross_column_row_major_exact_values_and_distinct_coverage_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6)
            with patch('axiom_research.stock_feature_inputs._feature_wire',side_effect=AssertionError('legacy whole FeatureBuild')), \
                 patch('axiom_research.stock_native_json.make_native_carrier',side_effect=AssertionError('legacy native metadata')):
                saved=f.matrix(options=options())
            try:
                expected=[r for d in f.days for r in f.day(d)[0]]
                self.assertEqual(saved.row_metadata(list(range(9))),expected)
                self.assertEqual(saved.metrics['feature_coverage_encode_calls'],2)
                self.assertEqual(saved.metrics['coverage_validation_calls'],2)
                self.assertEqual(saved.metrics['feature_context_write_calls'],6)
                index=saved.to_dict(); self.assertEqual(index['contract_version'],'stock_feature_inputs_v3')
                meta=_read(index['partitions'][0]['metadata']['path']); output=meta['day_outputs'][0]
                day_rows=f.day(f.days[0])[0]
                raw=b''.join(struct.pack('<d',v if v is not None else 0.0) for r in day_rows for v in r['values'])
                correct='sha256:'+sha256(raw).hexdigest()
                wrong=b''.join(struct.pack('<d',r['values'][j] if r['values'][j] is not None else 0.0)
                    for start in (0,2,4) for r in day_rows for j in range(start,start+2))
                self.assertEqual(output['buffers']['values']['buffer_digest'],correct)
                self.assertNotEqual(correct,'sha256:'+sha256(wrong).hexdigest())
                with load_stock_feature_inputs(saved.path) as fresh:
                    self.assertEqual(fresh.row_metadata(list(range(9))),expected)
                    self.assertEqual(fresh.identity,saved.identity)
                    self.assertEqual(fresh.metrics['coverage_validation_calls'],2)
                full=f.matrix(options=options(rows=3,columns=6))
                try:
                    fullmeta=_read(full.to_dict()['partitions'][0]['metadata']['path'])
                    self.assertEqual([o['output_ref'] for o in fullmeta['day_outputs']],
                        [_read(p['metadata']['path'])['day_outputs'][0]['output_ref'] for p in index['partitions'][::3]])
                finally: full.close()
            finally: saved.close()

    def test_same_input_hit_fresh_bytes_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6)
            def interrupt(update):
                if update['stage']=='feature_matrix_block': raise InterruptedPreparation('stop after published block')
            with self.assertRaises(InterruptedPreparation): f.matrix(options=options(),progress=interrupt)
            target=next((f.root/'saved').iterdir()); self.assertFalse((target/'index.json').exists())
            self.assertEqual(_read(target/'checkpoint.json')['contract_version'],'stock_feature_inputs_checkpoint_v3')
            saved=f.matrix(options=options())
            try:
                self.assertEqual(f.computed,f.days)
                before=len(f.computed); hit=f.matrix(options=options())
                try:
                    self.assertTrue(hit.reused); self.assertEqual(hit.identity,saved.identity)
                    self.assertEqual(len(f.computed),before)
                    blob=_read(_read(saved.to_dict()['partitions'][0]['metadata']['path'])['query_contexts'][0]['path'])['coverage']['bytes']
                    path=Path(blob['path']); data=path.read_bytes(); path.write_bytes(data[:-1]+b' ')
                    with self.assertRaises(ValueError): f.matrix(options=options())
                finally: hit.close()
            finally: saved.close()

    def test_resealed_future_cutoff_anchor_and_source_binding_rejected(self):
        for field in ('cutoff','anchor','batch_ref','extra_context'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                f=SourceFixture(Path(temp),6); saved=f.matrix(options=options(rows=3))
                meta=_read(saved.to_dict()['partitions'][0]['metadata']['path'])
                first=meta['input_evidence'][0]; sid=next(s for s,v in first['source_evidence'].items() if v['field']=='close')
                binding=first['source_evidence'][sid]; oldref=binding['query_context_ref']
                contexts={d['query_context_ref']:_read(d['path']) for d in meta['query_contexts']}
                if field=='batch_ref': binding['batch_ref']=digest('wrong actual source')
                else:
                    child=deepcopy(contexts[oldref]); child.pop('query_context_ref')
                    if field=='cutoff': child['context']['query']['cutoff_by_session']={d:'2099-01-01T00:00:00Z' for d in first['sessions']}
                    elif field=='anchor': child['context']['query']['adjustment_anchor']=f.days[-1]
                    else: child['context']['reader_version']='unused-new-context'
                    desc=write_part(saved.path,seal(child,'query_context_ref'),'query_context_ref'); newref=desc['query_context_ref']
                    contexts[newref]=_read(desc['path'])
                    if field!='extra_context':
                        for p in meta['input_evidence']:
                            for source in p['source_evidence'].values():
                                if source['query_context_ref']==oldref: source['query_context_ref']=newref
                        meta['query_contexts']=[d for d in meta['query_contexts'] if d['query_context_ref']!=oldref]
                        del contexts[oldref]
                    meta['query_contexts'].append(desc); meta['query_contexts'].sort(key=lambda d:d['query_context_ref'])
                replace_metadata(saved,meta,contexts); saved.close()
                with self.assertRaises(ValueError): load_stock_feature_inputs(saved.path)

    def test_resealed_column_block_concatenation_digest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6); saved=f.matrix(options=options(rows=3))
            meta=_read(saved.to_dict()['partitions'][0]['metadata']['path'])
            raw=b''.join(struct.pack('<d',r['values'][j] if r['values'][j] is not None else 0.0)
                for start in (0,2,4) for r in f.day(f.days[0])[0] for j in range(start,start+2))
            output=meta['day_outputs'][0]; output['buffers']['values']['buffer_digest']='sha256:'+sha256(raw).hexdigest()
            output.pop('output_ref'); output['output_ref']=digest(output)
            replace_metadata(saved,meta); saved.close()
            with self.assertRaisesRegex(ValueError,'daily output bytes'): load_stock_feature_inputs(saved.path)

    def test_interleaved_partition_columns_keep_complete_row_major_order(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6); saved=f.matrix(options=options(rows=3,columns=3))
            index=saved.to_dict(); previous=index['partitions'][0]
            rows=[r for day in f.days for r in f.day(day)[0]]; parts=[]
            for positions in ((0,2,4),(1,3,5)):
                shape=[len(rows),len(positions)]
                buffers={
                    'values':write_buffer(saved.path,(r['values'][j] if r['values'][j] is not None else 0.0
                        for r in rows for j in positions),dtype='float64_le',shape=shape),
                    'value_validity':write_buffer(saved.path,(r['validity'][j] for r in rows for j in positions),
                        dtype='bool_u8',shape=shape),
                    'available_at_utc_us':write_buffer(saved.path,(instant_us(r['availability'][j]) if r['availability'][j] is not None else 0
                        for r in rows for j in positions),dtype='int64_le',shape=shape),
                    'available_at_validity':write_buffer(saved.path,(r['availability'][j] is not None
                        for r in rows for j in positions),dtype='bool_u8',shape=shape)}
                parts.append(write_partition(saved.path,table='features',fold_spec_ref=None,
                    row_index_ref=previous['row_index_ref'],schema_digest=previous['schema_digest'],
                    row_offset=0,row_count=len(rows),columns=[f.columns[j] for j in positions],
                    buffers=buffers,metadata=previous['metadata']))
            index['partitions']=parts; index.pop('content_digest')
            index['feature_inputs_ref']=digest({k:index[k] for k in
                ('contract_version','definition_ref','qlib_manifest','schema_digest','row_index','source_selection','partitions')})
            write_json(saved.path/'index.json',seal(index,'content_digest')); saved.close()
            with load_stock_feature_inputs(saved.path) as fresh: self.assertEqual(fresh.row_metadata(list(range(9))),rows)

    def test_query_hashing_uses_distinct_contexts_without_merging_bindings(self):
        from axiom_research.stock_feature_inputs import _validate_feature_block
        from axiom_research.stock_matrix_feature_sources import _same_json
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6); saved=f.matrix(options=options(rows=3))
            meta=_read(saved.to_dict()['partitions'][0]['metadata']['path'])
            contexts={d['query_context_ref']:_read(d['path']) for d in meta['query_contexts']}
            evidence=meta['input_evidence'][0]; sources=evidence['source_evidence']
            self.assertEqual(len(sources),6)
            with patch('axiom_research.stock_matrix_feature_sources.digest',wraps=digest) as hashes:
                _selection(evidence,meta['row_references'][evidence['session']]['feature_ref'],contexts)
                self.assertEqual(hashes.call_count,2)
            rows=saved.row_metadata(list(range(9))); indexed={(r['security_id'],r['session']):r for r in rows}
            proof={p['session']:p for p in meta['input_evidence']}
            with patch('axiom_research.stock_feature_inputs.digest',wraps=digest) as hashes:
                _validate_feature_block({'qlib_view':f.view},indexed,proof,f.spec,f.view,f.universe_id,
                                        source_contexts=contexts)
                # Per-source qualifications remain checked. Query reference
                # work is once for each actual day/context, not each binding.
                refs=[call.args[0] for call in hashes.call_args_list if type(call.args[0]) is dict and
                      set(call.args[0])=={'snapshot_id','query','reader_version'}]
                self.assertEqual(len(refs),len(contexts))
            bad=deepcopy(proof); sid=next(iter(bad[f.days[0]]['source_evidence']))
            bad[f.days[0]]['core_plan']['sources'][0]['qualification']='transplanted qualification'
            with self.assertRaisesRegex(ValueError,'field/source binding'):
                _validate_feature_block({'qlib_view':f.view},indexed,bad,f.spec,f.view,f.universe_id,
                                        source_contexts=contexts)
            alias={'nested':list(range(100))}
            with patch('axiom_research.stock_matrix_feature_sources._same_json',wraps=_same_json) as compare:
                self.assertTrue(compare(alias,alias)); self.assertEqual(compare.call_count,1)
            saved.close()

    def test_created_coverage_is_fully_admitted_before_checkpoint(self):
        from axiom_research.stock_native_json import _publish
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6)
            def altered(temporary,path):
                created=_publish(temporary,path)
                if created:
                    raw=path.read_bytes(); path.write_bytes(raw[:-1]+b' ')
                return created
            with patch('axiom_research.stock_native_json._publish',side_effect=altered):
                with self.assertRaises(ValueError): f.matrix(options=options())
            target=next((f.root/'saved').iterdir())
            self.assertFalse((target/'checkpoint.json').exists()); self.assertFalse((target/'index.json').exists())

    def test_resealed_bool_shape_is_not_an_integer_dimension(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),1); saved=f.matrix(options=options(rows=3,columns=1))
            meta=_read(saved.to_dict()['partitions'][0]['metadata']['path'])
            output=meta['day_outputs'][0]
            output['buffers']['values']['shape'][1]=True
            output.pop('output_ref'); output['output_ref']=digest(output)
            replace_metadata(saved,meta); saved.close()
            with self.assertRaisesRegex(ValueError,'daily output bytes'): load_stock_feature_inputs(saved.path)

    def test_concurrent_index_does_not_overwrite_a_different_result(self):
        from axiom_research.stock_feature_inputs import _publish_compact_index
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6); saved=f.matrix(options=options())
            path=saved.path/'index.json'; original=path.read_bytes(); index=saved.to_dict()
            _publish_compact_index(path,index)
            index['status']='PARTIAL'
            with self.assertRaisesRegex(ValueError,'concurrent compact Feature index changed'):
                _publish_compact_index(path,index)
            self.assertEqual(path.read_bytes(),original)
            with load_stock_feature_inputs(saved.path) as fresh: self.assertEqual(fresh.identity,saved.identity)
            saved.close()

    def test_actual_buffer_tamper_and_version_pairing_rejected(self):
        for mode in ('buffer','selection'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                f=SourceFixture(Path(temp),6); saved=f.matrix(options=options())
                index=saved.to_dict()
                if mode=='buffer':
                    path=Path(index['partitions'][0]['buffers']['values']['path']); raw=path.read_bytes()
                    path.write_bytes(bytes([raw[0]^1])+raw[1:])
                else:
                    selection=_read(index['source_selection']['path']); selection.pop('source_selection_ref')
                    selection['contract_version']='stock_feature_source_selection_v1'
                    index['source_selection']=write_part(saved.path,seal(selection,'source_selection_ref'),'source_selection_ref')
                    index.pop('content_digest'); index['feature_inputs_ref']=digest({k:index[k] for k in
                        ('contract_version','definition_ref','qlib_manifest','schema_digest','row_index','source_selection','partitions')})
                    write_json(saved.path/'index.json',seal(index,'content_digest'))
                saved.close()
                with self.assertRaises(ValueError): load_stock_feature_inputs(saved.path)

    def test_coverage_absent_null_and_canonical_types_remain_distinct(self):
        from axiom_research.stock_matrix_feature_sources import _FeatureCoverageWriter
        from axiom_research.stock_matrix_reader import VerifiedMatrixStore
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); writer=_FeatureCoverageWriter(); metrics={}; maximum=2*1024**2
            descs=[writer.write(root,v,maximum=maximum,retained=lambda:0,metrics=metrics)
                   for v in (None,0,0.0,-0.0,False,(0,),(False,))]
            self.assertEqual(len({d['buffer_digest'] for d in descs}),7)
            self.assertEqual(Path(descs[0]['path']).read_bytes(),b'null')
            with VerifiedMatrixStore(maximum_matrix_bytes=maximum) as store:
                for desc in descs: store._admit_coverage(desc,parent={})
                self.assertEqual(store.metrics['coverage_validation_calls'],7)
            class AbsentAndNull(SourceFixture):
                def day(self,day):
                    rows,p=super().day(day)
                    for s in p['source_evidence'].values():
                        if s['field']=='is_member': s['query_context']['coverage']=None
                        else: s['query_context'].pop('coverage')
                    return rows,p
            f=AbsentAndNull(root,6); saved=f.matrix(options=options())
            try:
                meta=_read(saved.to_dict()['partitions'][0]['metadata']['path'])
                coverage=[_read(d['path'])['coverage'] for d in meta['query_contexts']]
                self.assertIn({'present':False,'bytes':None},coverage)
                self.assertEqual(saved.metrics['feature_coverage_encode_calls'],1)
                self.assertEqual(saved.metrics['coverage_validation_calls'],1)
                with load_stock_feature_inputs(saved.path) as fresh: self.assertEqual(fresh.identity,saved.identity)
            finally: saved.close()

    def test_budget_and_changed_coverage_are_not_cached_as_trust(self):
        from axiom_research.stock_matrix_feature_sources import _FeatureCoverageWriter
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); writer=_FeatureCoverageWriter(); metrics={}
            with self.assertRaises(ValueError): writer.write(root,{'nested':['x']*100},maximum=8192,retained=lambda:8192,metrics=metrics)
            self.assertFalse((root/'buffers').exists())
            desc=writer.write(root,{'ok':1},maximum=1024**2,retained=lambda:0,metrics=metrics)
            Path(desc['path']).write_bytes(b'{"ok":2}')
            with self.assertRaisesRegex(ValueError,'coverage changed'):
                writer.write(root,{'ok':1},maximum=1024**2,retained=lambda:0,metrics=metrics)
            # A new writer must verify orphan bytes, not trust the CAS name.
            fresh=_FeatureCoverageWriter()
            with self.assertRaisesRegex(ValueError,'coverage mismatch'):
                fresh.write(root,{'ok':1},maximum=1024**2,retained=lambda:0,metrics=metrics)

    def test_public_loader_imports_no_data_core_or_training(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6); saved=f.matrix(options=options()); identity=saved.identity; saved.close()
            original=builtins.__import__
            def blocked(name,*args,**kwargs):
                if name.startswith(('axiom_data','axiom_engine','qlib','lightgbm','pandas')): raise AssertionError(name)
                return original(name,*args,**kwargs)
            with patch('builtins.__import__',side_effect=blocked),load_stock_feature_inputs(saved.path) as fresh:
                self.assertEqual(fresh.identity,identity); self.assertEqual(len(fresh.row_metadata([0,1,2])),3)

    def test_completion_callback_failure_keeps_partial_and_published_child_is_checked(self):
        import axiom_research.stock_feature_inputs as owner
        for mode in ('callback','after_publish'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                f=SourceFixture(Path(temp),6); publish=owner._publish_compact_index
                def callback(update):
                    if update['stage']=='feature_producer_complete': raise InterruptedPreparation('before COMPLETE')
                def altered(path,value):
                    publish(path,value)
                    blob=Path(value['partitions'][0]['buffers']['values']['path']); raw=blob.read_bytes()
                    blob.write_bytes(bytes([raw[0]^1])+raw[1:])
                with patch.object(owner,'_publish_compact_index',side_effect=altered if mode=='after_publish' else publish):
                    with self.assertRaises((InterruptedPreparation,ValueError)):
                        f.matrix(options=options(),progress=callback if mode=='callback' else None)
                target=next((f.root/'saved').iterdir())
                if mode=='callback': self.assertFalse((target/'index.json').exists())
                else:
                    with self.assertRaises(ValueError): load_stock_feature_inputs(target)

    def test_complete_checkpoint_resumes_without_recomputing_and_releases_builder(self):
        with tempfile.TemporaryDirectory() as temp:
            f=SourceFixture(Path(temp),6)
            def fail(update):
                if update['stage']=='feature_producer_complete': raise InterruptedPreparation('before index')
            with self.assertRaises(InterruptedPreparation): f.matrix(options=options(),progress=fail)
            self.assertEqual(f.computed,f.days)
            updates=[]; saved=f.matrix(options=options(),progress=updates.append)
            try:
                self.assertEqual(f.computed,f.days)
                self.assertIsNone(saved._store._caller_retained_bytes)
                self.assertEqual(updates[-1]['stage'],'feature_producer_complete')
                with load_stock_feature_inputs(saved.path) as fresh:
                    self.assertEqual(fresh.identity,saved.identity)
            finally: saved.close()

    def test_prepare_fold_projection_and_prediction_ancestry_without_training(self):

        from axiom_research.stock_folds import prediction_rows
        from axiom_research.stock_matrix_folds import _predictions
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f=FoldSourceFixture(root); saved=f.matrix(options=options(rows=24)); fold=f.folds()[0]
            data=PublicDataFixture(f.spec); module=types.ModuleType('axiom_data'); module.QuerySpec=Query; module.adjust_prices=data.adjust
            metrics={}
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=f.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=f.environment), \
                 patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('training called')):
                batch=prepare_saved_v2_fixture(data,feature_inputs=saved,fold_specs=[fold],destination=root/'prepared',
                    preparation_options={'row_block_sessions':24,'column_block':2,'maximum_resident_bytes':64*1024**2,
                                         'normalization_backend':'core_cs_batch_v1'},metrics=metrics)
            view=_read(batch['prepared_view']['path'])
            self.assertEqual(_read(view['source_selection']['path'])['contract_version'],'stock_matrix_source_selection_v2')
            self.assertEqual([p for p in view['partitions'] if p['table']=='features'],saved.to_dict()['partitions'])
            for name in ('supplier_calls','feature_core_calls','train_calls','predict_calls','account_calls'): self.assertEqual(metrics[name],0)
            with load_stock_ml_batch_inputs(batch) as admitted:
                first=batch['folds'][0]
                with admitted._matrix_project(first['input_manifest'],first['fold_spec']) as projected:
                    features=projected.features; parents=saved._parents_by_session
                    self.assertEqual(features['parents_by_session'],{d:parents[d] for d in sorted(set(features['training_sessions']+features['prediction_sessions']))})
                    model={'model_ref':digest('synthetic saved model, no training'),
                        'simulated_available_at':fold['simulated_model_available_at'],'target_semantics':'synthetic_target'}
                    scores={(r['security_id'],r['session']):0.125 for r in features['rows']
                        if r['member'] and all(r['validity']) and feature_available(r) is not None}
                    prediction=seal({'contract_version':'stock_prediction_run_v2','signal_stage':'prediction_raw',
                        'score_semantics':model['target_semantics'],'score_unit':'dimensionless','feature_ref':features['feature_ref'],
                        'model_ref':model['model_ref'],'limitations':['synthetic; no inference executed'],
                        'universe':f.universe,'rows':prediction_rows(features,fold,model,scores),
                        'fold_spec_ref':digest(fold),'clock_basis':'declared_simulation'},'signal_run_ref')
                    _predictions(prediction,features,f.spec,fold,model)
                    prediction['rows'][0]['source_refs'][1]=digest('transplanted old day ref')
                    with self.assertRaisesRegex(ValueError,'original source'): _predictions(prediction,features,f.spec,fold,model)
                self.assertEqual(admitted.metrics['train_calls'],0)
            self.assertTrue(all(Path(p).exists() for p in saved._store.fingerprints)); saved.close()


if __name__=='__main__': unittest.main()
