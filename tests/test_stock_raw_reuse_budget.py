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
import test_stock_compact_v4 as fixtures


class RawReuseBudgetTests(unittest.TestCase):
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
