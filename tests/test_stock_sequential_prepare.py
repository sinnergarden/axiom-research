"""Tiny sequential producer evidence; no real Data, model or account work."""
from collections import defaultdict
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch
import numpy  # Load before the synthetic sys.modules patch takes its snapshot.

import test_stock_compact_v3 as fixtures
from test_stock_matrix_prepare import PublicDataFixture, Query
from axiom_research import load_stock_feature_view, prepare_stock_ml_batch_inputs
from axiom_research import stock_compact_batch as batch_owner
from axiom_research.stock_compact_store import OwnedStore, _view_data
from axiom_research.stock_fold_inputs import validate_spec
from axiom_research.stock_label_contracts import _instant
from axiom_research.stock_artifacts import _read, file_digest


class SequentialPrepareTests(unittest.TestCase):
    def prepare(self, f, feature, root, *, metrics=None, options=None, progress=None):
        data=PublicDataFixture(f.spec); module=types.ModuleType('axiom_data')
        module.QuerySpec=Query; module.adjust_prices=data.adjust
        settings={'row_block_sessions':32,'column_block':32,
            'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'}
        settings.update(options or {})
        with patch.dict(sys.modules,{'axiom_data':module}):
            result=prepare_stock_ml_batch_inputs(data,feature_inputs=feature,fold_specs=f.folds(),
                destination=root/'prepared',preparation_options=settings,metrics=metrics,progress=progress)
        return result,data

    def test_releases_actual_target_buffers_and_preserves_full_query_union(self):
        """Every Raw chunk and normalized fold loses its byte/NumPy owners."""
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            refs=defaultdict(list); releases=[]; stores=[]; target_store_ids=set()
            real_read=batch_owner.read_target; real_release=OwnedStore.release_payloads
            real_init=OwnedStore.__init__

            def init(store,*args,**kwargs):
                real_init(store,*args,**kwargs); stores.append(weakref.ref(store))

            def read(store,descriptor,*args,**kwargs):
                value,rows=real_read(store,descriptor,*args,**kwargs)
                target_store_ids.add(id(store))
                for name,array in rows.arrays.items():
                    refs[(id(store),value['buffers'][name]['path'])].append(weakref.ref(array))
                return value,rows

            def release(store,keep_paths=()):
                before={p:len(v) for p,v in store.arrays.items()}
                before_charge=store.resident_bytes
                result=real_release(store,keep_paths)
                removed={p for p in before if p not in store.arrays}
                amount=sum(before[p] for p in removed)
                # The actual owned byte buffers and every returned NumPy view
                # must be gone, rather than merely discounted in a lease.
                for p in removed:
                    self.assertTrue(all(ref() is None for ref in refs[(id(store),p)]),
                        'released target still has an external NumPy view: '+p)
                self.assertEqual(result['buffer_bytes'],amount)
                self.assertEqual(result['buffer_bytes']+result['json_bytes'],before_charge-store.resident_bytes)
                releases.append((id(store),amount,before_charge-store.resident_bytes))
                return result

            progress=[]; stats={}
            with patch.object(OwnedStore,'__init__',new=init), \
                 patch.object(batch_owner,'read_target',new=read), \
                 patch.object(OwnedStore,'release_payloads',new=release):
                result,data=self.prepare(f,path,root,metrics=stats,progress=progress.append)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(len(progress),4)
            self.assertGreater(stats['label_released_buffer_bytes'],0)
            self.assertEqual(stats['label_released_buffer_bytes'],sum(amount for store_id,amount,_
                in releases if store_id in target_store_ids))
            self.assertTrue(any(amount>0 and charge>=amount for _,amount,charge in releases))
            self.assertTrue(all(ref() is None or ref().closed for ref in stores))
            self.assertTrue(all(ref() is None for group in refs.values() for ref in group))
            self.assertEqual((stats['train_calls'],stats['predict_calls'],stats['account_calls']),(0,0,0))
            self.assertEqual((stats['raw_operator_calls'],stats['core_calls']),(16,4))
            self.assertEqual(len(data.queries),10)

            # Independent full-cutoff union oracle. It does not call the
            # producer's _query helper or derive a sliced DataBatch envelope.
            domains={}
            for fold in f.folds():
                training,inference=validate_spec(fold,f.spec['calendar'])
                for cutoff,days in ((fold['fit_cutoff'],training),
                                    (fold['evaluation_cutoff'],inference)):
                    key=_instant(cutoff)
                    domain=domains.setdefault(key,{'cutoff':cutoff,'days':set()})
                    domain['days'].update(days)
            markets=[query for query in data.queries if query['domain']=='market_daily']
            self.assertEqual(len(markets),len(domains))
            for query,domain in zip(markets,domains.values()):
                cutoff=domain['cutoff']; calendar=f.spec['calendar']
                anchor=max(d for d in calendar if d<=_instant(cutoff).date().isoformat())
                endpoints={anchor}
                for day in domain['days']:
                    index=calendar.index(day)
                    for shift in (1,5):
                        if index+shift<len(calendar) and calendar[index+shift]<=anchor:
                            endpoints.add(calendar[index+shift])
                self.assertEqual(query['sessions'],sorted(endpoints))
                self.assertEqual(query['symbols'],f.spec['universe'])
                self.assertEqual(query['cutoff_by_session'],{d:cutoff for d in sorted(endpoints)})
                self.assertEqual((query['pit_policy'],query['purpose'],query['price_basis']),
                    (f.spec['pit_policy'],'label_outcomes','unadjusted'))
            evaluation=_instant(f.folds()[0]['evaluation_cutoff'])
            self.assertEqual(len(domains[evaluation]['days']),12)

    def test_cached_targets_keep_binding_selectors_and_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            first,_=self.prepare(f,path,root)
            first_view=_read(first['prepared_view']['path'])
            frozen={p:file_digest(p) for p in (root/'prepared/compact-cache').rglob('*') if p.is_file()}
            stats={}; second,_=self.prepare(f,path,root,metrics=stats,options={'column_block':64})
            second_view=_read(second['prepared_view']['path'])
            self.assertNotEqual(first['batch_ref'],second['batch_ref'])
            self.assertEqual((stats['raw_operator_calls'],stats['core_calls']),(0,0))
            self.assertEqual((stats['raw_cache_hits'],stats['normalized_cache_hits']),(16,4))
            self.assertEqual(first_view['fold_targets'],second_view['fold_targets'])
            self.assertEqual([fold['input_manifest']['selectors'] for fold in first['folds']],
                             [fold['input_manifest']['selectors'] for fold in second['folds']])
            self.assertEqual({p:file_digest(p) for p in frozen},frozen)
            hit={}; repeated,data=self.prepare(f,path,root,metrics=hit,options={'column_block':64})
            self.assertEqual(repeated,second); self.assertTrue(hit['cache_hit'])
            self.assertEqual(data.queries,[])
            self.assertEqual((hit['raw_operator_calls'],hit['core_calls']),(0,0))

    def test_core_failure_closes_stores_without_complete_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            stores=[]; real_init=OwnedStore.__init__
            def init(store,*args,**kwargs):
                real_init(store,*args,**kwargs); stores.append(store)
            with patch.object(OwnedStore,'__init__',new=init), \
                 patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=RuntimeError('explicit Core failure')):
                with self.assertRaisesRegex(RuntimeError,'explicit Core failure'):
                    self.prepare(f,path,root)
            self.assertFalse(list((root/'prepared').glob('*/batch.json')))
            self.assertTrue(all(store.closed and not store.arrays and not store.json and
                store.resident_bytes==0 for store in stores))

    def test_setup_failure_releases_owned_feature(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            stores=[]; real_init=OwnedStore.__init__
            def init(store,*args,**kwargs):
                real_init(store,*args,**kwargs); stores.append(store)
            with patch.object(OwnedStore,'__init__',new=init), \
                 patch('axiom_research.stock_compact_labels._input_environment',
                       side_effect=RuntimeError('explicit input dependency failure')):
                with self.assertRaisesRegex(RuntimeError,'explicit input dependency failure'):
                    self.prepare(f,path,root)
            self.assertTrue(stores)
            self.assertTrue(all(store.closed and store.resident_bytes==0 for store in stores))
            self.assertFalse((root/'prepared').exists())

    def test_core_carrier_released_with_retained_exception_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            buffers=[]; failure=None
            def fail(carrier,**kwargs):
                buffers.extend(weakref.ref(value) for value in carrier.values()
                               if type(value) is memoryview)
                carrier=None
                raise RuntimeError('retain explicit Core exception')
            with patch('axiom_engine.core.execute_cs_zscore_batch',new=fail):
                try: self.prepare(f,path,root)
                except RuntimeError as error: failure=error
            self.assertIsNotNone(failure)
            self.assertIsNotNone(failure.__traceback__)
            self.assertEqual(len(buffers),9)
            self.assertTrue(all(ref() is None for ref in buffers),
                            'producer traceback retained the Core input buffers')
            self.assertFalse(list((root/'prepared').glob('*/batch.json')))

    def test_full_closure_failure_prevents_complete_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            stores=[]; real_init=OwnedStore.__init__
            def init(store,*args,**kwargs):
                real_init(store,*args,**kwargs); stores.append(store)
            with patch.object(OwnedStore,'__init__',new=init), \
                 patch.object(batch_owner.CompactState,'verify_all',
                       side_effect=ValueError('explicit saved closure failure')):
                with self.assertRaisesRegex(ValueError,'explicit saved closure failure'):
                    self.prepare(f,path,root)
            self.assertFalse(list((root/'prepared').glob('*/batch.json')))
            self.assertTrue(all(store.closed and store.resident_bytes==0 for store in stores))

    def test_borrowed_eager_feature_choice_survives_prepare(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=fixtures.CompactV3Tests().fixture(root)
            with load_stock_feature_view(path) as feature:
                store=_view_data(feature)['store']; before=dict(store.arrays)
                result,_=self.prepare(f,feature,root)
                self.assertEqual(result['status'],'COMPLETE')
                self.assertEqual(store.arrays,before)
                self.assertFalse(store.closed)


if __name__=='__main__': unittest.main()
