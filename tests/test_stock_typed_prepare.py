"""Small typed/streaming acceptance; synthetic widths are not feature catalogs."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import tempfile
import unittest
import weakref
from collections import deque
from unittest.mock import patch

import test_stock_compact_v3 as fixtures
from axiom_research import load_stock_feature_view, load_stock_ml_batch_inputs
from axiom_research.stock_artifacts import digest, digest_array_rows, _read, write_json
from axiom_research.stock_batch import _data
from axiom_research.stock_compact_store import feature_rows, iter_feature_eligibility, training_matrix, _view_data
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_label_contracts import (_eligible_reason, _FeatureEligibility,
    _instant, normalization_section_inputs, NORMALIZATION_SPEC)
from axiom_research.stock_matrix_storage import instant_us, write_part, write_buffer


class TypedPrepareTests(unittest.TestCase):
    def test_streaming_canonical_array_matches_independent_bytes(self):
        rows=[{'z':None,'unicode':'基金\n\"\\é','float':-0.0,'flags':[True,False]},
              {'time':'2024-01-02T12:00:00.123456Z','n':1.2345678901234567,'tiny':5e-324}]
        encoded=json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
        expected='sha256:'+sha256(encoded).hexdigest()
        self.assertEqual(digest_array_rows(iter(rows)),expected)
        self.assertEqual(digest_array_rows(dict(reversed(list(row.items()))) for row in rows),expected)
        self.assertEqual(digest_array_rows(iter(())),'sha256:'+sha256(b'[]').hexdigest())
        self.assertNotEqual(expected,'sha256:'+sha256(encoded+b'\n').hexdigest())
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                digest_array_rows(iter([{'value':value}]))

    def test_streaming_array_does_not_retain_previous_rows(self):
        class Row(dict): pass
        recent=deque(maxlen=4); produced=[]
        def rows():
            for i in range(1000):
                if len(recent)>=2:
                    self.assertIsNone(recent[-2](),'stream retained a completed row')
                row=Row(index=i,value=float(i)); recent.append(weakref.ref(row)); produced.append(i)
                yield row
        expected=digest([{'index':i,'value':float(i)} for i in range(1000)])
        self.assertEqual(digest_array_rows(rows()),expected)
        self.assertEqual(produced,list(range(1000)))
        self.assertTrue(all(ref() is None for ref in recent))

    def test_typed_and_dict_share_conflicting_reason_priority(self):
        cutoff=_instant('2024-01-08T12:30:00.123456Z')
        raw={'valid':True,'return':0.1,'end_session':'2024-01-08',
             'label_available_at':'2024-01-08T12:30:00.123456Z'}
        feature={'member':True,'values':[1.0,2.0],'validity':[True,True],
                 'knowledge_cutoff':'2024-01-08T12:30:00.123456Z'}
        cases=[({}, {}, None),({'valid':False,'invalid_reason':'RAW_CUSTOM'}, {'member':False},'RAW_CUSTOM'),
               ({'return':None},{'member':False},'RAW_LABEL_INVALID'),
               ({'end_session':None},{'member':False},'LABEL_CLOCK_OR_ENDPOINT_UNKNOWN'),
               ({'label_available_at':'2024-01-08T12:30:00.123457Z'},{'member':False},'LABEL_NOT_MATURE'),
               ({'end_session':'2024-01-09'},{'member':False},'LABEL_NOT_MATURE'),
               ({},{'member':False,'values':[None,2.0]},'NOT_MEMBER'),
               ({},{'member':None,'values':[None,2.0]},'MEMBERSHIP_UNKNOWN'),
               ({},{'values':[None,2.0],'validity':[False,True],'knowledge_cutoff':None},'FEATURE_MISSING'),
               ({},{'validity':[False,True],'knowledge_cutoff':None},'FEATURE_INVALID'),
               ({},{'knowledge_cutoff':'2024-01-08T12:30:00.123457Z'},'FEATURE_NOT_AVAILABLE'),
               ({},{'knowledge_cutoff':None},'FEATURE_CLOCK_UNKNOWN')]
        for raw_changes,feature_changes,expected in cases:
            r={**raw,**raw_changes}; f={**feature,**feature_changes}
            typed=_FeatureEligibility(f['member'],all(type(v) is float for v in f['values']),
                all(f['validity']),f['knowledge_cutoff'],instant_us(raw['label_available_at']))
            with self.subTest(raw=raw_changes,feature=feature_changes):
                self.assertEqual(_eligible_reason(r,f,2,cutoff),expected)
                self.assertEqual(_eligible_reason(r,typed,2,cutoff),expected)

    def test_wide_typed_gather_and_binding_match_full_row_oracle(self):
        helper=fixtures.CompactV3Tests()
        for width in (158,300):
            with self.subTest(width=width),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); first=[None]
                def missing(day,rows):
                    if first[0] is None: first[0]=day
                    if day==first[0]:
                        rows[0]['values'][-1]=None; rows[0]['validity'][-1]=False
                        rows[0]['reasons'][-1]=['synthetic_missing']
                f,path=helper.fixture(root,width=width,transform=missing)
                from axiom_engine.core import execute_cs_zscore_batch
                captured=[]
                def core(carrier,**kwargs):
                    captured.append(deepcopy(carrier['source_bindings_by_session']))
                    return execute_cs_zscore_batch(carrier,**kwargs)
                with load_stock_feature_view(path) as view:
                    with patch('axiom_research.stock_compact_store.feature_rows',side_effect=AssertionError('producer expanded Feature panel')), \
                         patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=core):
                        manifest,_=helper.prepare(f,view,root,folds=f.folds()[:1])
                    with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                        state=_data(batch)['matrix_state']; record=state.view['fold_targets'][0]
                        normalized=list(state.targets[record['normalized']['target_ref']][1])
                        raw=[row for part in record['raw_parts'] for row in state.targets[part['target_ref']][1]]
                        spec=_view_data(view)['definition']['spec']; securities=spec['universe']; positions={d:i for i,d in enumerate(spec['feature_sessions'])}
                        offsets=[positions[r['feature_session']]*len(securities)+securities.index(r['security_id']) for r in raw]
                        full=feature_rows(view,offsets)
                        typed=list(iter_feature_eligibility(view,offsets))
                        cutoff=_instant(record['fold_spec']['fit_cutoff'])
                        reasons=[_eligible_reason(r,row,width,cutoff) for r,row in zip(raw,full)]
                        self.assertEqual(reasons,[_eligible_reason(r,row,width,cutoff) for r,row in zip(raw,typed)])
                        self.assertEqual(reasons,state.targets[record['normalized']['target_ref']][0]['cohort']['eligibility_reasons'])
                        self.assertEqual(typed[0].maximum_available_at_utc_us,instant_us(full[0]['availability'][0]))
                        valid=[r for r in normalized if r['valid']]
                        joined=feature_rows(view,record['training_offsets'])
                        for row,target in zip(joined,valid):
                            row.update(label=target['return'],raw_return=target['raw_return'],label_available_at=target['raw_available_at'],normalized_available_at=target['label_available_at'])
                        self.assertEqual(record['training_binding'],{'training_rows_ref':digest(joined),'training_row_count':len(joined),
                            'training_keys_digest':digest([[r['security_id'],r['session']] for r in joined])})
                        common_raw={'label_ref':digest([p['target_ref'] for p in record['raw_parts']]),'calendar_ref':digest(spec['calendar']),
                            'label_spec':{'formula':'close(f+5) / open(f+1) - 1'},'rows':raw}
                        index={(r['security_id'],r['session']):r for r in full}
                        self.assertEqual(len(captured),1)
                        for day in sorted({r['feature_session'] for r in raw}):
                            oracle=normalization_section_inputs(common_raw,feature_ref=view.identity,feature_rows=index,session=day,
                                securities=securities,width=width,cutoff=record['fold_spec']['fit_cutoff'])
                            old_section={'contract_version':'stock_label_section_v1','feature_session':day,
                                'eligible_keys':[[r['security_id'],day] for r,row in zip(raw,full)
                                    if r['feature_session']==day and _eligible_reason(r,row,width,cutoff) is None],
                                'raw_label_ref':common_raw['label_ref'],'feature_ref':view.identity,
                                'cutoff':cutoff.isoformat().replace('+00:00','Z'),'normalization_spec':NORMALIZATION_SPEC}
                            self.assertEqual(oracle['section_ref'],digest(old_section))
                            self.assertEqual(captured[0][day]['bindings'],sorted(oracle['plan']['sources'],key=lambda r:r['id']))
                        with batch._matrix_project(manifest['folds'][0]['input_manifest'],record['fold_spec']) as projection:
                            self.assertEqual(projection.X.shape,(len(joined),width))
                            self.assertEqual(projection.X.tolist(),[r['values'] for r in joined])
                            self.assertEqual(projection.y.tolist(),[r['label'] for r in joined])
                            candidates=[r for r in projection.feature_rows if r['member'] and all(r['validity'])]
                            self.assertEqual(projection.P.tolist(),[r['values'] for r in candidates])
                            self.assertFalse(projection.X.flags.writeable)
                            self.assertEqual(projection.excluded['FEATURE_MISSING'],1)

    def test_saved_prepared_admission_ignores_current_producer_and_environment(self):
        from axiom_research import stock_compact_labels as owner
        helper=fixtures.CompactV3Tests()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=helper.fixture(root)
            with load_stock_feature_view(path) as view:
                with patch.object(owner,'_implementation',return_value=digest('historical-producer')), \
                     patch.object(owner,'_input_environment',return_value=f.environment):
                    manifest,_=helper.prepare(f,view,root,folds=f.folds()[:1])
                with patch.object(owner,'_implementation',side_effect=AssertionError('load required current producer')), \
                     patch.object(owner,'_input_environment',side_effect=AssertionError('load required current environment')):
                    with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                        self.assertEqual(batch.identity,manifest['batch_ref'])
                        item=manifest['folds'][0]
                        with batch._matrix_project(item['input_manifest'],item['fold_spec']) as projection:
                            self.assertEqual(projection.X.shape,(122,6))

    def test_unknown_native_clock_still_rejects_before_core_and_complete(self):
        helper=fixtures.CompactV3Tests()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); first=[None]
            def unknown(day,rows):
                if first[0] is None: first[0]=day
                if day==first[0]: rows[0]['availability']=[None]*6
            f,path=helper.fixture(root,transform=unknown)
            with load_stock_feature_view(path) as view, \
                 patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=AssertionError('Core accepted unknown Feature clocks')):
                with self.assertRaisesRegex(ValueError,'native clock exceeds fit'):
                    helper.prepare(f,view,root,folds=f.folds()[:1])
                self.assertFalse(list((root/'prepared').rglob('batch.json')))
                self.assertEqual(len(list(iter_feature_eligibility(view,[0]))),1)

    def test_independent_metadata_column_parts_preserve_missing_and_reject_gather(self):
        import numpy as np
        helper=fixtures.CompactV3Tests()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=helper.fixture(root); index=_read(path/'index.json')
            original=index['partitions'][0]; metadata=_read(original['metadata']['path']); parts=[]
            dtypes={'values':'<f8','value_validity':'u1','available_at_utc_us':'<i8','available_at_validity':'u1'}
            for name,chosen in [('a',[0,1,2]),('b',[3,4,5])]:
                directory=path/name; directory.mkdir(); part=deepcopy(original); part.pop('partition_ref')
                part['columns']=[f.columns[i] for i in chosen]
                part['metadata']=write_part(directory,metadata,'metadata_ref')
                for field,descriptor in original['buffers'].items():
                    values=np.frombuffer(Path(descriptor['path']).read_bytes(),dtype=dtypes[field]).reshape(descriptor['shape'])[:,chosen].ravel().tolist()
                    if field in ('value_validity','available_at_validity'): values=[bool(v) for v in values]
                    part['buffers'][field]=write_buffer(directory,values,dtype=descriptor['dtype'],shape=[original['row_count'],3])
                parts.append(seal(part,'partition_ref'))
            index['partitions']=parts; write_json(path/'index.json',index)
            with load_stock_feature_view(path) as view:
                self.assertEqual(len(_view_data(view)['blocks']),2)
                full=feature_rows(view,[0])[0]; typed=list(iter_feature_eligibility(view,[0]))[0]
                raw={'valid':True,'return':0.1,'end_session':f.days[0],'label_available_at':f.days[0]+'T00:00:00Z'}
                cutoff=_instant(f.folds()[0]['fit_cutoff'])
                self.assertEqual(full['values'][3:],[None,None,None])
                self.assertFalse(typed.complete_finite)
                self.assertEqual(_eligible_reason(raw,typed,6,cutoff),_eligible_reason(raw,full,6,cutoff))
                self.assertEqual(_eligible_reason(raw,typed,6,cutoff),'FEATURE_MISSING')
                with patch('numpy.empty',side_effect=AssertionError('allocated underfilled matrix')):
                    with self.assertRaisesRegex(ValueError,'missing columns'):
                        training_matrix(view,[0],f.folds()[0]['fit_cutoff'])
                manifest,_=helper.prepare(f,view,root,folds=f.folds()[:1])
                with load_stock_ml_batch_inputs(manifest,feature_inputs=view) as batch:
                    state=_data(batch)['matrix_state']; record=state.view['fold_targets'][0]
                    self.assertEqual(record['training_offsets'],[])
                    reasons=state.targets[record['normalized']['target_ref']][0]['cohort']['eligibility_reasons']
                    self.assertIn('FEATURE_MISSING',reasons); self.assertNotIn(None,reasons)


if __name__=='__main__': unittest.main()
