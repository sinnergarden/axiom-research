"""Bounded synthetic Feature admission: exact clocks, bytes and lifecycle."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import gc
import struct
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from axiom_research import stock_compact_store as owner
from axiom_research.stock_artifacts import digest, write_json
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_storage import instant_us, write_part


class FeatureAdmissionTests(unittest.TestCase):
    def fixture(self,root,*,width=6,column_block=32,mutate=None):
        days=['2026-01-05','2026-01-06']; securities=['SYN001.SH','SYN002.SH']
        columns=[f'SYN{i:03d}' for i in range(width)]
        spec={'scope':'explicit_synthetic','snapshot':'s_admission_fixture','pit_policy':'synthetic_v1',
              'calendar':days,'read_sessions':days,'feature_sessions':days,'universe':securities,
              'ordered_features':columns,'feature_selection':[{'id':c,'semantic_version':'1'} for c in columns],
              'catalog_ref':digest(['synthetic',columns]),'cutoff_by_session':{d:d+'T20:30:00+08:00' for d in days}}
        schema=[{'name':c,'dtype':'float64','unit':'dimensionless','stage':'cross_sectional','missing':'preserve'} for c in columns]
        index=write_part(root,seal({'sessions':days,'security_ids':securities,'row_count':4,'order':'session_security'},'row_index_ref'),'row_index_ref')
        parts=[];expected=[]
        for ordinal,day in enumerate(days):
            rows=[{'security_id':security,'session':day,'member':True,'knowledge_cutoff':spec['cutoff_by_session'][day],
                   'source_refs':[digest(['synthetic',day])],'values':[float(i+j+1) for j in range(width)],
                   'validity':[True]*width,'availability':[day+'T20:00:00.123456+08:00']*width,
                   'reasons':[[] for _ in columns]} for i,security in enumerate(securities)]
            physical={'values':[v for r in rows for v in r['values']],
                      'value_validity':[1]*(2*width),'available_at_validity':[1]*(2*width),
                      'available_at_utc_us':[instant_us(r['availability'][0]) for r in rows for _ in columns]}
            if mutate is not None:mutate(rows,physical)
            projection=deepcopy(rows)
            for i,row in enumerate(projection):
                row['values']=[physical['values'][i*width+j] if row['validity'][j] else None for j in range(width)]
            expected.extend(projection)
            meta=write_part(root,seal({'rows':rows,'row_references':{day:{'feature_ref':digest(['parent',day]),
                      'qlib_view_ref':digest(['qlib',day])}}},'metadata_ref'),'metadata_ref')
            for first in range(0,width,column_block):
                chosen=columns[first:first+column_block];positions=[i*width+j for i in range(2) for j in range(first,first+len(chosen))]
                buffers={}
                for name,dtype,code in [('values','float64_le','d'),('value_validity','bool_u8','B'),
                                       ('available_at_utc_us','int64_le','q'),('available_at_validity','bool_u8','B')]:
                    # Deliberately permits sealed bad physical payloads for ingress tests.
                    payload=struct.pack('<'+code*len(positions),*(physical[name][i] for i in positions))
                    ref='sha256:'+sha256(payload).hexdigest();path=root/(ref[7:]+'.bin');path.write_bytes(payload)
                    buffers[name]={'path':str(path),'file_digest':ref,'buffer_digest':ref,'dtype':dtype,'shape':[2,len(chosen)]}
                parts.append(seal({'table':'features','row_index_ref':index['row_index_ref'],'schema_digest':digest(schema),
                    'row_offset':ordinal*2,'row_count':2,'columns':chosen,'buffers':buffers,'metadata':meta},'partition_ref'))
        write_json(root/'index.json',{'contract_version':'stock_feature_inputs_v3','status':'COMPLETE',
            'definition':{'spec':spec},'schema':schema,'row_index':index,'partitions':parts,
            'feature_inputs_ref':digest(['explicit_synthetic',spec])})
        return expected

    def test_widths_share_successful_exact_clock_parses_across_column_parts(self):
        for width in (6,158,300):
            with self.subTest(width=width),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);expected=self.fixture(root,width=width)
                with owner.load_stock_feature_view(root,residency='sequential') as view:
                    original=owner.instant_us;calls=[]
                    def counted(text):calls.append(text);return original(text)
                    with patch.object(owner,'instant_us',side_effect=counted):owner.set_feature_window(view,list(range(4)))
                    self.assertEqual(len(calls),4)  # Two exact strings per independently admitted date.
                    self.assertEqual(view.metrics['feature_clock_parse_calls'],4)
                    self.assertEqual(view.metrics['feature_clock_cache_max_entries'],2)
                    self.assertEqual(owner.feature_rows(view,list(range(4))),expected)
                    X=owner.training_matrix(view,list(range(4)),'2026-01-06T20:30:00+08:00')
                    golden=np.asarray([r['values'] for r in expected],dtype='<f8')
                    self.assertEqual(X.dtype,golden.dtype);self.assertEqual(X.tobytes(),golden.tobytes())

    def test_null_invalid_reason_and_exact_timezone_spelling_survive(self):
        def mutate(rows,physical):
            rows[0]['validity'][0]=False;rows[0]['reasons'][0]=['SOURCE_MISSING','original detail'];physical['value_validity'][0]=0;physical['values'][0]=-0.0
            rows[0]['availability'][1]=None;physical['available_at_validity'][1]=0;physical['available_at_utc_us'][1]=0
            rows[0]['availability'][2]=rows[0]['session']+'T12:00:00.123456Z'
            rows[0]['availability'][3]='1970-01-01T00:00:00+00:00';physical['available_at_utc_us'][3]=0
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);expected=self.fixture(root,mutate=mutate)
            with owner.load_stock_feature_view(root) as view:
                rows=owner.feature_rows(view,list(range(4)));self.assertEqual(rows,expected)
                facts=list(owner.iter_feature_eligibility(view,list(range(4))))
                self.assertFalse(facts[0].complete_finite)
                self.assertIsNone(rows[0]['availability'][1]);self.assertTrue(rows[0]['validity'][1])
                self.assertTrue(rows[0]['availability'][2].endswith('Z'));self.assertIsNone(rows[0]['values'][0])

    def test_partial_unknown_clock_keeps_existing_training_eligibility(self):
        def mutate(rows,physical):
            rows[0]['availability'][0]=None;physical['available_at_validity'][0]=0;physical['available_at_utc_us'][0]=0
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);expected=self.fixture(root,mutate=mutate)
            with owner.load_stock_feature_view(root) as view:
                self.assertEqual(owner.feature_rows(view,list(range(4))),expected)
                X=owner.training_matrix(view,list(range(4)),'2026-01-06T20:30:00+08:00')
                self.assertEqual(X.tobytes(),np.asarray([r['values'] for r in expected],dtype='<f8').tobytes())

    def test_metadata_typed_and_cell_cutoff_corruptions_are_rejected(self):
        def after_K(rows,p):
            rows[0]['availability'][0]=rows[0]['session']+'T20:30:00.000001+08:00';p['available_at_utc_us'][0]=instant_us(rows[0]['availability'][0])
        mutations={
            'after original cutoff':after_K,
            'wrong typed clock':lambda r,p:p['available_at_utc_us'].__setitem__(0,p['available_at_utc_us'][0]+1),
            'metadata int instead of bool':lambda r,p:r[0]['validity'].__setitem__(0,1),
            'flag mismatch':lambda r,p:r[0]['validity'].__setitem__(0,False),
            'clock presence mismatch':lambda r,p:p['available_at_validity'].__setitem__(0,0),
            'invalid nonzero':lambda r,p:(r[0]['validity'].__setitem__(0,False),p['value_validity'].__setitem__(0,0)),
            'null nonzero clock':lambda r,p:(r[0]['availability'].__setitem__(0,None),p['available_at_validity'].__setitem__(0,0)),
            'nonfinite valid':lambda r,p:p['values'].__setitem__(0,float('nan')),
            'nonfinite invalid':lambda r,p:(r[0]['validity'].__setitem__(0,False),p['value_validity'].__setitem__(0,0),p['values'].__setitem__(0,float('inf'))),
            'invalid boolean physical byte':lambda r,p:p['value_validity'].__setitem__(0,2)}
        for name,mutate in mutations.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.fixture(root,mutate=mutate)
                with owner.load_stock_feature_view(root,residency='sequential') as view:
                    with self.assertRaises(ValueError):owner.set_feature_window(view,[0])
                    state=owner._view_data(view);self.assertEqual(state['store'].arrays,{})
                    self.assertTrue(all(b['parts'] is None for b in state['blocks']))

    def test_cache_is_discarded_on_release_and_bounded_for_diverse_clocks(self):
        def diverse(rows,p):
            for i,row in enumerate(rows):
                for j in range(6):
                    at=row['session']+f'T20:00:00.{i*6+j+1:06d}+08:00'
                    row['availability'][j]=at;p['available_at_utc_us'][i*6+j]=instant_us(at)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,mutate=diverse)
            with owner.load_stock_feature_view(root,residency='sequential') as view,patch.object(owner,'_ADMISSION_CLOCK_CACHE_ENTRIES',2):
                owner.set_feature_window(view,[0]);first=view.metrics['feature_clock_parse_calls']
                self.assertGreater(first,2);self.assertLessEqual(view.metrics['feature_clock_cache_max_entries'],2)
                owner.set_feature_window(view,[]);owner.set_feature_window(view,[0])
                self.assertEqual(view.metrics['feature_clock_parse_calls'],first*2)

    def test_workspace_budget_is_checked_before_scratch_allocation(self):
        store=owner.OwnedStore({'maximum_matrix_bytes':4096,'maximum_source_bytes':1024**2,'maximum_parent_bytes':1024**2})
        with patch.object(np,'empty',side_effect=AssertionError('scratch allocated before budget check')):
            with self.assertRaisesRegex(ValueError,'resident byte budget'):
                owner._validate_feature_cells({'store':store,'column_positions':{'a':0}},[({'columns':['a']},{})],[{}])
        store.close()

    def test_cache_growth_is_budgeted_before_parsing_new_key(self):
        store=owner.OwnedStore({'maximum_matrix_bytes':4400,'maximum_source_bytes':1024**2,'maximum_parent_bytes':1024**2})
        with patch.object(owner,'instant_us',side_effect=AssertionError('clock parsed before cache budget check')):
            with self.assertRaisesRegex(ValueError,'resident byte budget'):
                owner._validate_feature_cells({'store':store,'column_positions':{'a':0}},[({'columns':['a']},{})],
                    [{'knowledge_cutoff':'2026-01-05T20:30:00+08:00'}])
        store.close()

    def test_retained_error_drops_scratch_arrays_cache_and_owner_aliases(self):
        def mutate(rows,p):p['available_at_utc_us'][0]+=1
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root,mutate=mutate);refs=[];original=np.empty
            def track(*args,**kwargs):
                result=original(*args,**kwargs);refs.append(weakref.ref(result));return result
            with owner.load_stock_feature_view(root,residency='sequential') as view:
                caught=None
                try:
                    with patch.object(np,'empty',side_effect=track):owner.set_feature_window(view,[0])
                except ValueError as error:caught=error
                self.assertIsNotNone(caught);gc.collect();self.assertTrue(refs);self.assertTrue(all(r() is None for r in refs))
                trace=caught.__traceback__;found=False
                while trace is not None:
                    frame=trace.tb_frame
                    if frame.f_code.co_name=='_validate_feature_cells':
                        found=True
                        for name in ('value','group','rows','store','cache','cutoffs','meta_flags','meta_present','meta_clock','arrays','part','row'):
                            self.assertIsNone(frame.f_locals[name],name)
                    trace=trace.tb_next
                self.assertTrue(found);self.assertEqual(owner._view_data(view)['store'].arrays,{})


if __name__=='__main__':unittest.main()
