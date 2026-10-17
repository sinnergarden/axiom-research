"""Small synthetic E2E for saved raw composition; no production data or training."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from axiom_engine.core import compile_feature_columns, execute_feature_columns
from axiom_engine.feature_library import FeatureLibrary
from axiom_research import CatalogFeatureLibrary, SavedColumnFeatureSource, load_a01_feature_catalog
from axiom_research.stock_feature_library_adapter import build_feature_library_training_view, load_feature_library_training_view
from axiom_research.stock_compact_store import iter_feature_rows
from test_stock_feature_library import saved_source


class SavedFeatureSubsetE2E(unittest.TestCase):
    def test_saved_subset_stitch_membership_and_native_consumer(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp);catalog=load_a01_feature_catalog();budget=8*1024**2
            selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP003','ALP001')]
            columns=[s['id'] for s in selection];axis=['A','B','C'];sample=['A','B']
            members={'2020-01-03':['A','B'],'2020-01-04':['B','C'],
                     '2020-01-05':['A','C'],'2020-01-06':['A','B','C']}
            def source(name, **kwargs):
                return stack.enter_context(SavedColumnFeatureSource(saved_source(root/name,**kwargs),maximum_working_bytes=budget))
            first=source('raw-first',securities=['A','B','C','D'],snapshot='raw-1')
            second=source('raw-second',securities=['A','B','C','D'],snapshot='raw-2')
            member=source('member',securities=axis,snapshot='member',members_by_session=members)
            alternate=source('alternate',securities=axis,snapshot='other-members')
            wrong=source('wrong',securities=['A','B','C','D'],snapshot='wrong',value_shift=2)
            days=member.source.manifest['spec']['feature_sessions']
            libraries=[FeatureLibrary(root/name,maximum_working_bytes=budget) for name in ('l1','l2','output')]
            parents=[]
            for lib,owner,dates in zip(libraries[:2],(first,second),(days[:2],days[2:])):
                descriptor,_=CatalogFeatureLibrary(lib,owner,catalog).ensure(selection=selection,output_sessions=dates)
                parents.append(stack.enter_context(lib.open_feature_view(descriptor)))
            output=libraries[2];before=set(root.rglob('*.bin'))
            descriptor=output.compose_views(parents,output_sessions=days,security_ids=axis,columns=columns,sample_universe=sample)
            self.assertEqual(set(root.rglob('*.bin')),before)
            raw=stack.enter_context(output.open_feature_view(descriptor));bindings=list(zip(parents,(first,second)))
            for parent in parents:
                for day in parent.describe()['sessions']:
                    keys=[(s,day) for s in axis];actual=raw.project(keys);expected=parent.project(keys,columns)
                    for name in ('values','validity','available_at_utc_us'):
                        np.testing.assert_array_equal(actual[name],expected[name])
                    for field,dictionary in (('reason_codes','reason_dictionary'),('source_codes','source_sets')):
                        self.assertEqual([actual[dictionary][int(c)] for c in actual[field].flat],
                                         [expected[dictionary][int(c)] for c in expected[field].flat])
                    self.assertEqual(actual['source_bindings'],expected['source_bindings'])
                    self.assertEqual(actual['partition_refs'],expected['partition_refs'])
            with self.assertRaisesRegex(ValueError,'overlapping'):
                output.compose_views(parents+parents[:1],output_sessions=days,security_ids=axis,columns=columns)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                output.compose_views(parents[:1],output_sessions=days,security_ids=axis,columns=columns)
            # Independent full catalog DAG expectations, before disabling every
            # saved source's raw prepare entry. Normalization uses Core itself.
            expected={}
            for owner in (member,alternate):
                plan=owner.plan(catalog,selection,normalized=True)
                expected[id(owner)]=[]
                for day in days:
                    inputs=owner.prepare(plan,day,axis,{n:'synthetic-direct' for n in columns})
                    expected[id(owner)].append(execute_feature_columns(compile_feature_columns(plan),inputs=inputs,
                        output_sessions=[day],max_working_bytes=budget))
            for owner in (first,second,member,alternate,wrong):
                stack.enter_context(patch.object(owner,'prepare',side_effect=AssertionError('raw recomputation forbidden')))
            results=[]
            for owner in (member,alternate):
                facade=CatalogFeatureLibrary(output,owner,catalog)
                normalized,stats=facade.normalize_saved(raw,raw_bindings=bindings,selection=selection)
                self.assertEqual(stats['raw']['core_calls'],0)
                self.assertEqual(stats['cross_sectional']['core_calls'],len(days) if owner is member else 3)
                view=stack.enter_context(output.open_feature_view(normalized));results.append(view.project()['values'].copy())
                for day,expected_day in zip(days,expected[id(owner)]):
                    actual=view.project([(s,day) for s in axis])
                    for key in ('values','validity','available_at_utc_us'):
                        np.testing.assert_array_equal(actual[key],expected_day[key][0])
                _,again=facade.normalize_saved(raw,raw_bindings=bindings,selection=selection)
                self.assertEqual(again['cross_sectional']['core_calls'],0)
                with self.assertRaisesRegex(ValueError,'reference-dependent'):
                    output.compose_views([view],output_sessions=days,security_ids=['A','B'],columns=columns)
                if owner is member:
                    training=build_feature_library_training_view(view,member,selection=selection,destination=root/'training',
                        raw_view=raw,raw_bindings=bindings,maximum_working_bytes=budget)
                    cells=view.project()
                    expected_training=np.where(cells['validity'],cells['values'].astype(object),None)
            self.assertFalse(np.array_equal(*results,equal_nan=True))
            with self.assertRaisesRegex(ValueError,'local dependency mismatch'):
                CatalogFeatureLibrary(output,member,catalog).normalize_saved(raw,selection=selection,
                    raw_bindings=[(parents[0],wrong),bindings[1]])
            # Restore a native research handle after closing every source/view.
            stack.close()
            with load_feature_library_training_view(training,limits={'maximum_matrix_bytes':budget}) as restored:
                rows=list(iter_feature_rows(restored,list(range(len(days)*len(sample)))))
                np.testing.assert_array_equal([r['values'] for r in rows],expected_training)
                self.assertEqual([r['member'] for r in rows],[s in members[d] for d in days for s in sample])


if __name__=='__main__':unittest.main()
