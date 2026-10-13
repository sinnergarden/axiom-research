"""Small synthetic saved sources; no vendor collection or production data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from axiom_engine.feature_library import FeatureLibrary, feature_definitions
from axiom_research import CatalogFeatureLibrary, SavedColumnFeatureSource, load_a01_feature_catalog
from axiom_research.stock_feature_library_adapter import _feature_source_compatibility
from test_stock_feature_library import saved_source


class SavedSecurityBlockTests(unittest.TestCase):
    def test_growth_reuses_old_base_and_batches_new_members_then_cs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); catalog=load_a01_feature_catalog()
            selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP001','ALP003')]
            old=saved_source(root/'old',day_count=4,securities=['A','B'])
            new=saved_source(root/'new',day_count=4,securities=['A','B','C','D'])
            library=FeatureLibrary(root/'library',maximum_working_bytes=8*1024**2)
            with SavedColumnFeatureSource(old,maximum_working_bytes=8*1024**2) as source:
                days=source.source.manifest['spec']['feature_sessions']
                facade=CatalogFeatureLibrary(library,source,catalog)
                raw,_=facade.ensure(selection=selection,output_sessions=days)
                normalized,_=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                with library.open_feature_view(raw) as view:
                    old_receipts={d:view.column_receipts(d) for d in days}
                with library.open_feature_view(normalized) as view: before=view.project()['values'].copy()
            original_bytes={r['partition']['path']:Path(r['partition']['path']).read_bytes()
                            for group in old_receipts.values() for r in group.values()}
            with SavedColumnFeatureSource(new,maximum_working_bytes=8*1024**2) as source:
                facade=CatalogFeatureLibrary(library,source,catalog)
                with patch.object(source,'prepare',wraps=source.prepare) as prepare:
                    grown,stats=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                self.assertEqual((stats['raw']['core_calls'],stats['raw']['computed_output_cells']),(2,8))
                self.assertEqual((stats['cross_sectional']['core_calls'],stats['cross_sectional']['computed_output_cells']),(2,16))
                self.assertEqual([call.args[2] for call in prepare.call_args_list],[['C','D'],['C','D']])
                raw,stats=facade.ensure(selection=selection,output_sessions=days)
                self.assertEqual(stats['raw']['core_calls'],0)
                with library.open_feature_view(raw) as view:
                    for day in days:
                        for name,receipt in view.column_receipts(day).items():
                            self.assertEqual({tuple(b['security_ids']) for b in receipt['blocks']},
                                             {('A','B'),('C','D')})
                            self.assertIn(old_receipts[day][name]['partition'],[b['partition'] for b in receipt['blocks']])
                    _feature_source_compatibility(view,source,selection)
                with library.open_feature_view(grown) as view:
                    actual=view.project(); _feature_source_compatibility(view,source,selection)
                fresh=FeatureLibrary(root/'fresh',maximum_working_bytes=8*1024**2)
                expected_ref,_=CatalogFeatureLibrary(fresh,source,catalog).ensure(
                    selection=selection,output_sessions=days,normalized=True)
                with fresh.open_feature_view(expected_ref) as view: expected=view.project()
                for field in ('values','validity','available_at_utc_us'):
                    np.testing.assert_array_equal(actual[field],expected[field])
                with patch('axiom_data.derived._adjust_columns',side_effect=AssertionError('warm raw must not adjust')):
                    _,warm=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                self.assertEqual((warm['raw']['core_calls'],warm['cross_sectional']['core_calls']),(0,0))
            self.assertTrue(all(Path(p).read_bytes()==data for p,data in original_bytes.items()))
            with library.open_feature_view(normalized) as view:
                np.testing.assert_array_equal(view.project()['values'],before)

    def test_subset_receipts_bind_fields_revisions_validity_clock_and_lookback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); catalog=load_a01_feature_catalog()
            selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP001','ALP019','LQD001')]
            old=saved_source(root/'old',securities=['A','B'])
            with SavedColumnFeatureSource(old,maximum_working_bytes=8*1024**2) as source:
                day=source.source.manifest['spec']['feature_sessions'][-1]
                definitions=feature_definitions(source.plan(catalog,selection),input_semantics=source.input_semantics())
                baseline=source.dependency_refs(definitions,[day],['A','B'])[day]
            def revision(metadata):
                for row in metadata.get('close',{}).get('by_key',[]):
                    if row['security_id']=='A' and row['session']=='2020-01-02': row['revision_id']+=':corrected'
            def anchor_revision(metadata):
                for row in metadata.get('factor',{}).get('by_key',[]):
                    if row['security_id']=='A' and row['session']==day: row['revision_id']+=':corrected'
            cases=[('unchanged',None,None,set()),
                ('new_security',lambda a:a['values'].__setitem__((5,2,3),99),None,set()),
                ('unused_field',lambda a:a['values'].__setitem__((5,0,1),99),None,set()),
                ('outside_window',lambda a:a['values'].__setitem__((0,0,3),99),None,set()),
                ('lookback_value',lambda a:a['values'].__setitem__((1,0,3),99),None,{'ALP019'}),
                ('lookback_valid',lambda a:a['validity'].__setitem__((1,0,3),False),None,{'ALP019'}),
                ('lookback_clock',lambda a:a['available'].__setitem__((1,0,3),a['available'][1,0,3]+1000000),None,{'ALP019'}),
                ('revision_only',None,revision,{'ALP019'}),
                ('anchor_revision',None,anchor_revision,{'ALP001','ALP019'}),
                ('amount',lambda a:a['values'].__setitem__((5,0,4),99),None,{'LQD001'})]
            for name,arrays,evidence,changed in cases:
                with self.subTest(name=name):
                    descriptor=saved_source(root/name,securities=['A','B','C','D'],
                        mutate_arrays=arrays,mutate_evidence=evidence)
                    with SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2) as source:
                        actual=source.dependency_refs(definitions,[day],['A','B'])[day]
                        self.assertEqual({n for n in baseline if baseline[n]!=actual[n]},changed)

    def test_prepare_projects_fields_securities_and_history_before_adjustment(self):
        from axiom_data.derived import _adjust_columns
        from axiom_engine.feature_library.identity import select_security_plan, select_feature_plan
        with tempfile.TemporaryDirectory() as temp:
            def unrelated_future(arrays): arrays['available'][:,:,1]+=10*24*3600*1000000
            descriptor=saved_source(Path(temp)/'source',securities=['A','B','C','D'],mutate_arrays=unrelated_future)
            catalog=load_a01_feature_catalog()
            with SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2) as source:
                day=source.source.manifest['spec']['feature_sessions'][-1]; securities=['C','D']
                for feature,history,fields in [('ALP019',5,{'close'}),('LQD001',1,{'amount_cny'})]:
                    plan=select_security_plan(select_feature_plan(
                        source.plan(catalog,[{'id':feature,'semantic_version':'1.0.0'}]),[feature]),securities)
                    definitions=feature_definitions(plan,input_semantics=source.input_semantics())
                    dependencies=source.dependency_refs(definitions,[day],securities)[day]
                    with patch('axiom_data.derived._adjust_columns',wraps=_adjust_columns) as adjust:
                        inputs=source.prepare(plan,day,securities,dependencies)
                    self.assertEqual(inputs['security_ids'],securities)
                    self.assertEqual(len(inputs['sessions']),history)
                    self.assertEqual(set(inputs['values']),fields)
                    self.assertTrue(all(value.shape==(history,2) for value in inputs['values'].values()))
                    if feature=='ALP019':
                        self.assertEqual(adjust.call_count,1)
                        self.assertEqual(adjust.call_args.args[0].shape,(5,2,1))
                    else: adjust.assert_not_called()

    def test_leaf_cache_and_subset_inflate_share_existing_working_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            descriptor=saved_source(Path(temp)/'source',day_count=4)
            with SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2) as source:
                self.assertGreater(source._leaf_cache_bytes,0)
                self.assertEqual(source.store._caller_retained_bytes(),source._leaf_cache_bytes)
                maximum=source.store.maximum_matrix_bytes
                try:
                    retained=source._leaf_cache_bytes
                    source.store.maximum_matrix_bytes=retained+100
                    with self.assertRaisesRegex(ValueError,'cache exceeds'):
                        source._charge_leaf_cache(101)
                    self.assertEqual(source._leaf_cache_bytes,retained)
                    source.store.maximum_matrix_bytes=maximum
                    base=source.store._native_caller_bytes()
                    source.store.maximum_matrix_bytes=base+512*1024
                    field,day=next(iter(source._leaf_refs))
                    with patch('axiom_research.stock_feature_library.zlib.decompressobj',
                               side_effect=AssertionError('inflate allocated before budget check')):
                        with self.assertRaisesRegex(ValueError,'decode workspace'):
                            source._selected_leaf_refs({field},{day},['A'])
                finally: source.store.maximum_matrix_bytes=maximum

    def test_cohort_only_change_recomputes_cs_without_adjusting_cached_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); descriptor=saved_source(root/'source',day_count=4)
            catalog=load_a01_feature_catalog(); selection=[{'id':'ALP001','semantic_version':'1.0.0'}]
            library=FeatureLibrary(root/'library',maximum_working_bytes=8*1024**2)
            with SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2) as source:
                days=source.source.manifest['spec']['feature_sessions']
                CatalogFeatureLibrary(library,source,catalog).ensure(selection=selection,output_sessions=days,normalized=True)
            with SavedColumnFeatureSource(descriptor,reference_security_ids=['A','B'],maximum_working_bytes=8*1024**2) as source:
                with patch('axiom_data.derived._adjust_columns',side_effect=AssertionError('CS must consume cached raw')):
                    _,stats=CatalogFeatureLibrary(library,source,catalog).ensure(
                        selection=selection,output_sessions=days,normalized=True)
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(0,2))


if __name__=='__main__': unittest.main()
