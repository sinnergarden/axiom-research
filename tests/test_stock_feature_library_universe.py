"""Configured pools reuse unchanged facts; daily reference changes invalidate CS."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from axiom_engine.core import compile_feature_columns, execute_feature_columns
from axiom_engine.feature_library import FeatureLibrary
from axiom_research import CatalogFeatureLibrary, SavedColumnFeatureSource, load_a01_feature_catalog
from axiom_research.stock_artifacts import digest
from axiom_research.stock_feature_columns import _source_config, build_feature_column_source
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_storage import write_part
from test_stock_feature_library import saved_source, tiles


class ConfiguredFeatureUniverseTests(unittest.TestCase):
    def test_two_saved_pools_reuse_raw_and_bind_daily_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); catalog=load_a01_feature_catalog()
            selection=[{'id':n,'semantic_version':'1.0.0'} for n in ('ALP001','ALP003')]
            original=saved_source(root/'csi300')
            members={'2020-01-03':['A','B'],'2020-01-04':['B','C'],
                     '2020-01-05':['A','C'],'2020-01-06':['A','B']}
            alternate=saved_source(root/'csi500',universe_name='csi500',members_by_session=members)
            library=FeatureLibrary(root/'library',maximum_working_bytes=8*1024**2)
            with SavedColumnFeatureSource(original,maximum_working_bytes=8*1024**2) as source:
                self.assertEqual(source.capabilities()['universe_profile'],'csi300')
                self.assertEqual(source.capabilities()['pit_policies'],['best_effort_vendor_v1'])
                days=source.source.manifest['spec']['feature_sessions']
                facade=CatalogFeatureLibrary(library,source,catalog)
                raw,_=facade.ensure(selection=selection,output_sessions=days)
                normalized,_=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                with library.open_feature_view(raw) as view: old_raw=tiles(view)
                with library.open_feature_view(normalized) as view:
                    old_cs=tiles(view); before=view.project()['values'].copy()
                    old_semantics={n:d['semantic_ref'] for n,d in view.describe()['definitions'].items()}
            with SavedColumnFeatureSource(alternate,maximum_working_bytes=8*1024**2) as source:
                self.assertEqual(source.capabilities()['universe_profile'],'csi500')
                self.assertEqual(source.capabilities()['pit_policies'],['best_effort_vendor_v1'])
                facade=CatalogFeatureLibrary(library,source,catalog)
                raw,stats=facade.ensure(selection=selection,output_sessions=days)
                self.assertEqual(stats['raw']['core_calls'],0)
                with library.open_feature_view(raw) as view: self.assertEqual(tiles(view),old_raw)
                current,stats=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(0,4))
                plan=source.plan(catalog,selection,normalized=True)
                with library.open_feature_view(current) as view:
                    self.assertEqual({n:d['semantic_ref'] for n,d in view.describe()['definitions'].items()},old_semantics)
                    self.assertTrue(all(a['partition']['partition_ref']!=b['partition']['partition_ref']
                                        for a,b in zip(old_cs,tiles(view))))
                    self.assertFalse(np.array_equal(before,view.project()['values'],equal_nan=True))
                    for day in days:
                        self.assertEqual(set(plan.to_dict()['reference_members'][day]),set(members[day]))
                        inputs=source.prepare(plan,day,['A','B','C'],{n['id']:digest(['direct']) for n in selection})
                        expected=execute_feature_columns(compile_feature_columns(plan),inputs=inputs,
                            output_sessions=[day],max_working_bytes=8*1024**2)
                        actual=view.project([(s,day) for s in ('A','B','C')])
                        for key in ('values','validity','available_at_utc_us'):
                            np.testing.assert_array_equal(actual[key],expected[key][0])
                _,stats=facade.ensure(selection=selection,output_sessions=days,normalized=True)
                self.assertEqual((stats['raw']['core_calls'],stats['cross_sectional']['core_calls']),(0,0))
            with library.open_feature_view(normalized) as view:
                self.assertEqual(tiles(view),old_cs)
                np.testing.assert_array_equal(view.project()['values'],before)

    def test_source_config_keeps_verified_scope_universe(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for universe_name in ('csi300','csi500'):
                saved=saved_source(root/universe_name,universe_name=universe_name)
                spec=deepcopy(json.loads(Path(saved['path']).read_text())['spec'])
                request={'universe_id':universe_name,'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy']}
                result={'read_sessions':spec['calendar'],'read_symbols':spec['universe'],
                        'snapshot_id':spec['snapshot'],'pit_policy':spec['pit_policy']}
                scope=seal({'request':request,'request_ref':digest(request),'result':result,'result_ref':digest(result),
                            'source_proof':{},'source_proof_ref':digest({})},'scope_bundle_ref')
                spec.update(contract_version='stock_feature_inputs_spec_v1',ordered_features=['ALP001'],
                    cutoff_by_session={d:d+'T20:30:00+08:00' for d in spec['read_sessions']},
                    scope=write_part(root/universe_name,scope,'scope_bundle_ref'))
                self.assertEqual(_source_config(spec),{'snapshot':spec['snapshot'],'pit_policy':spec['pit_policy'],
                    'symbols':spec['universe'],'universe_id':universe_name,'calendar':spec['calendar'],
                    'read_sessions':spec['read_sessions'],'feature_sessions':spec['feature_sessions'],
                    'cutoff_by_session':spec['cutoff_by_session'],'feature_selection':spec['feature_selection'],
                    'scope_ref':spec['scope']['scope_bundle_ref']})

    def test_unproved_policy_and_membership_profile_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); saved=saved_source(root/'source',universe_name='csi500')
            manifest=json.loads(Path(saved['path']).read_text())
            unproved=deepcopy(manifest); unproved.pop('source_ref')
            unproved['spec']['pit_policy']='operational_pit_v1'
            descriptor=write_part(root/'unproved',seal(unproved,'source_ref'),'source_ref')
            with self.assertRaisesRegex(ValueError,'unsupported column source selection policy'):
                SavedColumnFeatureSource(descriptor,maximum_working_bytes=8*1024**2)
            domains=deepcopy(manifest['source_profiles'])
            for profile in domains.values():
                profile['source_profile']['availability']={'timezone':'UTC','session_release_time':'12:00:00'}
            data=Mock(); data.store.load_snapshot.return_value={'domains':domains}
            destination=root/'must-not-be-created'
            with self.assertRaisesRegex(ValueError,'membership source needs stable dated-event identities'):
                build_feature_column_source(data,spec=manifest['spec'],destination=destination)
            data.read.assert_not_called(); data.members.assert_not_called(); data.export_qlib.assert_not_called()
            self.assertFalse(destination.exists())


if __name__=='__main__': unittest.main()
