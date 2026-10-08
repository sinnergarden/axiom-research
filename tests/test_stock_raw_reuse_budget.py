"""Small public Raw-reuse budget probes; no providers, fit or accounts."""
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import load_stock_feature_view,load_stock_ml_batch_inputs,prepare_stock_ml_batch_inputs
from axiom_research.stock_batch import _data
from axiom_research.stock_compact_store import OwnedStore,_view_data,clear_feature_window
from axiom_research import stock_compact_labels as producer
from axiom_research.stock_compact_batch import CompactState
import test_stock_compact_v4 as fixtures


class RawReuseBudgetTests(unittest.TestCase):
    def test_a_b_c_operation_lifecycle_cold_hit_and_failed_transfer(self):
        reserve=OwnedStore.reserve; verify=CompactState.verify_all
        for same_handle in (True,False):
            with self.subTest(same_handle=same_handle),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); f,path,manifest,_=fixtures.CompactV4Tests().prepare(root,block=64)
                with load_stock_feature_view(path,residency='eager') as fa, \
                     (nullcontext(fa) if same_handle else load_stock_feature_view(path,residency='sequential')) as fb, \
                     (nullcontext(fb) if same_handle else load_stock_feature_view(path,residency='sequential')) as fc:
                    handles=[]; table=[]
                    try:
                        a=load_stock_ml_batch_inputs(manifest,feature_inputs=fa,residency='eager'); handles.append(a)
                        astate=_data(a)['matrix_state']
                        options=manifest['definition']['preparation_options']
                        def prepare(name,feature,source,selection,caller,caller_source,expect_hit,fail=False):
                            origin=_data(source)['matrix_state']; fstore=_view_data(feature)['store']
                            source_feature=_view_data(origin.feature)['store']; created=[]; checks=[]
                            prior=(origin.store.limits,origin._fixed_shared_bytes,origin._fixed_shared_source_bytes)
                            def create(*args,**kwargs):
                                store=OwnedStore(*args,**kwargs); created.append(store); return store
                            def check_reserve(owner,amount):
                                if created and owner is created[-1]:
                                    extra=origin.store.resident_bytes+origin.store.lease_bytes+prior[1]
                                    if source_feature is not fstore:
                                        extra+=source_feature.resident_bytes+source_feature.lease_bytes
                                    self.assertEqual(owner.shared_bytes,fstore.resident_bytes+caller+extra)
                                    checks.append(extra)
                                return reserve(owner,amount)
                            def check_verify(state):
                                if created and state.store is created[-1] and fail:
                                    raise RuntimeError('synthetic transfer validation failure')
                                return verify(state)
                            metrics={}
                            with patch.object(producer,'OwnedStore',new=create), \
                                 patch.object(OwnedStore,'reserve',new=check_reserve), \
                                 patch.object(CompactState,'verify_all',new=check_verify):
                                kwargs=dict(feature_inputs=feature,fold_specs=f.folds()[:2],destination=root/name,
                                    preparation_options=options,model_feature_selection=selection,reuse_raw_from_batch=source,
                                    _caller_bytes=caller,_caller_source_bytes=caller_source,metrics=metrics)
                                if fail:
                                    with self.assertRaisesRegex(RuntimeError,'transfer validation failure'):
                                        producer.prepare_compact_batch(None,**kwargs)
                                    self.assertTrue(created[-1].closed)
                                    self.assertFalse(_view_data(feature)['prepared'])
                                    result=None
                                else:
                                    result=producer.prepare_compact_batch(None,**kwargs)
                                    self.assertEqual(metrics['cache_hit'],expect_hit)
                                    returned=_view_data(feature)['prepared'][result['batch_ref']]
                                    self.assertEqual((returned._fixed_shared_bytes,returned._fixed_shared_source_bytes),
                                        (caller,caller_source))
                                    self.assertEqual(returned.store.shared_bytes,fstore.resident_bytes+caller)
                                    self.assertEqual(returned.store.shared_source_bytes,fstore.metrics['source_bytes']+caller_source)
                            self.assertTrue(checks and any(extra>0 for extra in checks))
                            self.assertEqual((origin.store.limits,origin._fixed_shared_bytes,origin._fixed_shared_source_bytes),prior)
                            origin.check()
                            table.append({'feature':'same' if same_handle else 'distinct','step':name,
                                'hit':expect_hit,'failed_transfer':fail,'returned_caller_bytes':None if fail else caller,
                                'temporary_raw_charge_removed':not fail,'source_restored':True})
                            return result
                        # A is active for both B paths; B keeps its own caller,
                        # never A's temporary Raw/Feature snapshot.
                        bmanifest=prepare('B',fb,a,f.selection[:1],8192,4096,False)
                        b=load_stock_ml_batch_inputs(bmanifest,feature_inputs=fb,residency='sequential'); handles.append(b)
                        b.close(); handles.remove(b)
                        bmanifest=prepare('B',fb,a,f.selection[:1],8192,4096,True)
                        b=load_stock_ml_batch_inputs(bmanifest,feature_inputs=fb,residency='sequential'); handles.append(b)
                        bstate=_data(b)['matrix_state']
                        a.close(); handles.remove(a)
                        self.assertTrue(astate.closed); self.assertEqual(astate.store.resident_bytes,0)
                        self.assertEqual(bstate._fixed_shared_bytes,8192)
                        # C borrows only active B, then retains its own zero
                        # caller baseline after cold or HIT completion.
                        cmanifest=prepare('C',fc,b,f.selection[:2],0,0,False)
                        c=load_stock_ml_batch_inputs(cmanifest,feature_inputs=fc,residency='sequential'); handles.append(c)
                        c.close(); handles.remove(c)
                        cmanifest=prepare('C',fc,b,f.selection[:2],0,0,True)
                        c=load_stock_ml_batch_inputs(cmanifest,feature_inputs=fc,residency='sequential'); handles.append(c)
                        c.close(); handles.remove(c)
                        # Failure before transfer closes the candidate and
                        # restores B for both a fresh path and an existing HIT.
                        prepare('failed-cold',fc,b,f.selection[:2],0,0,False,fail=True)
                        prepare('C',fc,b,f.selection[:2],0,0,True,fail=True)
                        print('RAW_REUSE_LIFECYCLE '+str(table))
                    finally:
                        for handle in reversed(handles): handle.close()

    def test_default_hit_transfers_the_same_store_to_saved_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path,manifest,_=fixtures.CompactV4Tests().prepare(root,block=64)
            with load_stock_feature_view(path,residency='sequential') as feature:
                metrics={}
                result=prepare_stock_ml_batch_inputs(None,feature_inputs=feature,fold_specs=f.folds()[:2],
                    destination=root/'prepared',preparation_options=manifest['definition']['preparation_options'],metrics=metrics)
                self.assertEqual(result,manifest); self.assertTrue(metrics['cache_hit'])
                state=_view_data(feature)['prepared'][result['batch_ref']]
                self.assertFalse(state.store.closed); state.verify_all()

    def test_live_eager_owner_cold_hit_and_feature_handle_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path,manifest,_=fixtures.CompactV4Tests().prepare(root,block=64)
            with load_stock_feature_view(path,residency='eager') as original, \
                 load_stock_ml_batch_inputs(manifest,feature_inputs=original,residency='eager') as source:
                state=_data(source)['matrix_state']; raw=state.store
                source_feature=_view_data(original)['store']
                # A historical peak deliberately bears no relation to live bytes.
                raw.metrics['peak_resident_bytes']=64*1024**3
                prior=(raw.limits,state._fixed_shared_bytes,state._fixed_shared_source_bytes)
                reserve=OwnedStore.reserve; working=producer._working
                for same_handle in (True,False):
                    context=nullcontext(original) if same_handle else load_stock_feature_view(path,residency='sequential')
                    with self.subTest(same_handle=same_handle),context as feature:
                        feature_store=_view_data(feature)['store']; targets=[]; observations=[]; raw_live=set()
                        def create_store(*args,**kwargs):
                            store=OwnedStore(*args,**kwargs); targets.append(store); return store
                        def ledger():
                            target=targets[-1]
                            stores={feature_store,source_feature,raw,target}
                            live=sum(s.resident_bytes+s.lease_bytes for s in stores)
                            blind=feature_store.resident_bytes+feature_store.lease_bytes+target.resident_bytes+target.lease_bytes
                            return target,stores,live,blind
                        def observe_reserve(owner,amount):
                            if targets:
                                target,stores,live,blind=ledger()
                                if owner in stores:
                                    self.assertEqual(owner.shared_bytes,live-owner.resident_bytes-owner.lease_bytes)
                                    observations.append((blind+amount,live+amount))
                                    raw_live.add(raw.resident_bytes+raw.lease_bytes)
                            return reserve(owner,amount)
                        def observe_working(stats,amount):
                            producer._sync_feature_charge(stats)
                            target,stores,live,blind=ledger()
                            scratch=stats['_retained_raw_bytes']+stats.get('_price_view_bytes',0)+amount
                            self.assertEqual(target.shared_bytes,live-target.resident_bytes-target.lease_bytes)
                            observations.append((blind+scratch,live+scratch))
                            raw_live.add(raw.resident_bytes+raw.lease_bytes)
                            return working(stats,amount)
                        options={'row_block_sessions':64,'column_block':32,'maximum_resident_bytes':64*1024**2,
                            'normalization_backend':'core_cs_batch_v1'}
                        selection=f.selection[:1]; destination=root/('shared' if same_handle else 'separate')
                        with patch.object(producer,'OwnedStore',new=create_store), \
                             patch.object(OwnedStore,'reserve',new=observe_reserve), \
                             patch.object(producer,'_working',new=observe_working):
                            metrics={}
                            result=prepare_stock_ml_batch_inputs(None,feature_inputs=feature,fold_specs=f.folds()[:2],
                                destination=destination,preparation_options=options,metrics=metrics,
                                model_feature_selection=selection,reuse_raw_from_batch=source)
                            self.assertFalse(metrics['cache_hit']); self.assertTrue(observations)
                            self.assertGreater(len(raw_live),1)  # Control ingress changed actual source residency.
                            blind_peak=max(x for x,_ in observations); full_peak=max(x for _,x in observations)
                            self.assertGreater(full_peak,blind_peak)
                            _view_data(feature)['prepared'][result['batch_ref']].close()
                            observations.clear(); hit_metrics={}
                            hit=prepare_stock_ml_batch_inputs(None,feature_inputs=feature,fold_specs=f.folds()[:2],
                                destination=destination,preparation_options=options,metrics=hit_metrics,
                                model_feature_selection=selection,reuse_raw_from_batch=source)
                            self.assertEqual(hit,result); self.assertTrue(hit_metrics['cache_hit'])
                            self.assertEqual(len(targets),2); self.assertTrue(targets[0].closed)
                            self.assertFalse(targets[1].closed); self.assertTrue(observations)
                            _view_data(feature)['prepared'][hit['batch_ref']].close()
                            # Without the live source charge this budget admits
                            # the measured peak. The public reuse path must reject.
                            budget=blind_peak+(full_peak-blind_peak)//2
                            observations.clear()
                            if not same_handle: clear_feature_window(feature)
                            with self.assertRaisesRegex(ValueError,'byte budget exceeded'):
                                prepare_stock_ml_batch_inputs(None,feature_inputs=feature,fold_specs=f.folds()[:2],
                                    destination=destination/'tight',preparation_options={**options,'maximum_resident_bytes':budget},
                                    model_feature_selection=selection,reuse_raw_from_batch=source)
                            self.assertTrue(any(blind<=budget<full for blind,full in observations))
                            self.assertTrue(targets[-1].closed)
                            self.assertFalse((destination/'tight').exists())
                        self.assertEqual((raw.limits,state._fixed_shared_bytes,state._fixed_shared_source_bytes),prior)
                        self.assertFalse(state.closed); state.check()
                        print('RAW_REUSE_LIVE_BUDGET '+str({'same_feature_handle':same_handle,'blind_peak':blind_peak,
                            'full_peak':full_peak,'rejected_budget':budget,'source_control_live_states':len(raw_live),
                            'cold_and_hit_shared_charge_exact':True}))


if __name__=='__main__': unittest.main()
