"""Bounded wide-protocol regression: actual QSYS rank + legacy member catalog."""
from contextlib import ExitStack
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from axiom_engine.feature_library import FeatureLibrary
from axiom_research import SavedColumnFeatureSource, CatalogFeatureLibrary, load_a01_feature_catalog
from axiom_research.feature_catalog import FeatureCatalog, load_feature_catalog
from axiom_research.stock_feature_library_adapter import build_feature_library_training_view, load_feature_library_training_view
from axiom_research.stock_compact_store import iter_feature_rows
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_storage import write_part
from test_stock_feature_library import saved_source


class SavedReferenceWideE2E(unittest.TestCase):
    def test_frozen_qsys_rank_with_legacy_member_catalog(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp); budget=8*1024**2
            catalog=load_feature_catalog(files('axiom_research').joinpath('catalogs/qsys_stock_v4.json'))
            selection=[{'id':'RST008','semantic_version':'1.0.0'}]
            with self.assertRaisesRegex(ValueError,'unknown feature'):
                load_a01_feature_catalog().select(selection)
            def missing(a):
                a['values'][3,1,3]=0; a['original_values'][3,1,3]=0
            desc=saved_source(root/'raw-source',securities=['A','B','C','D'],
                              universe_name='csi1800',mutate_arrays=missing)
            owner=stack.enter_context(SavedColumnFeatureSource(desc,maximum_working_bytes=budget))
            forged=stack.enter_context(SavedColumnFeatureSource(desc,reference_security_ids=['A','B','C'],maximum_working_bytes=budget))
            members=[]
            for i,changes in enumerate((None,{'2020-01-03':['A','B']})):
                m=saved_source(root/f'member-{i}',securities=['A','B','C'],snapshot=f'member-{i}',
                               universe_name='csi300',members_by_session=changes,value_shift=100+i)
                members.append(stack.enter_context(SavedColumnFeatureSource(m,maximum_working_bytes=budget)))
            days=owner.source.manifest['spec']['feature_sessions'];axis=['A','B','C']
            library=FeatureLibrary(root/'library',maximum_working_bytes=budget)
            parent_desc,_=CatalogFeatureLibrary(library,owner,catalog).ensure(selection=selection,output_sessions=days)
            parent=stack.enter_context(library.open_feature_view(parent_desc))
            self.assertTrue(parent.describe()['definitions']['RST008']['reference_dependent'])
            before=set(root.rglob('*.bin'))
            with self.assertRaisesRegex(ValueError,'reference-dependent'):
                library.compose_views([parent],output_sessions=days,security_ids=axis,columns=['RST008'])
            selected=library.compose_views([parent],output_sessions=days,security_ids=axis,
                                           columns=['RST008'],preserve_reference=True)
            raw=stack.enter_context(library.open_feature_view(selected))
            self.assertEqual(before,set(root.rglob('*.bin')))
            keys=[(s,d) for d in days for s in axis]
            expected=parent.project(keys);actual=raw.project()
            for key in ('values','validity','available_at_utc_us'):
                np.testing.assert_array_equal(actual[key],expected[key])
            for field,dictionary in (('reason_codes','reason_dictionary'),('source_codes','source_sets')):
                self.assertEqual([actual[dictionary][int(c)] for c in actual[field].flat],
                                 [expected[dictionary][int(c)] for c in expected[field].flat])
            for key in ('source_bindings','partition_refs'):
                self.assertEqual(actual[key],expected[key])
            self.assertFalse(actual['validity'].all())
            for d in days:
                a=raw.column_receipts(d)['RST008']['blocks'][0]
                b=parent.column_receipts(d)['RST008']['blocks'][0]
                self.assertEqual(a['dependency_ref'],b['dependency_ref'])
                self.assertEqual(a['security_ids'],['A','B','C','D'])
                self.assertEqual(a['selected_security_ids'],axis)
            for s in (owner,forged,*members):
                stack.enter_context(patch.object(s,'prepare',side_effect=AssertionError('raw recomputation')))
            original_dependencies=owner.dependency_refs
            def full_only(definitions, sessions, securities):
                self.assertEqual(securities,['A','B','C','D'])
                return original_dependencies(definitions,sessions,securities)
            stack.enter_context(patch.object(owner,'dependency_refs',side_effect=full_only))
            bindings=[(parent,owner)];outputs=[]
            for i,member in enumerate(members):
                normalized,stats=CatalogFeatureLibrary(library,member,catalog).normalize_saved(
                    raw,raw_bindings=bindings,selection=selection)
                self.assertEqual(stats['raw']['core_calls'],0)
                view=stack.enter_context(library.open_feature_view(normalized));outputs.append(view.project())
                semantic=view.describe()['definitions']['RST008']['semantic']
                self.assertEqual([n['op'] for n in semantic['nodes']],['cs_zscore'])
                self.assertEqual(semantic['inputs']['RST008']['semantics']['frozen_reference_feature']['view_ref'],raw.identity)
                for j,d in enumerate(days):
                    values=actual['values'][j*3:(j+1)*3,0]
                    valid=actual['validity'][j*3:(j+1)*3,0]
                    mask=valid & np.array([s in member.source._reference_members[d] for s in axis])
                    result=view.project([(s,d) for s in axis]);wanted=mask & (np.std(values[mask])>1e-12)
                    np.testing.assert_array_equal(result['validity'][:,0],wanted)
                    if wanted.any():
                        np.testing.assert_allclose(result['values'][wanted,0],
                            (values[wanted]-np.mean(values[mask]))/np.std(values[mask]),rtol=1e-14)
                if i==0:
                    training=build_feature_library_training_view(view,member,selection=selection,destination=root/'training',
                        raw_view=raw,raw_bindings=bindings,catalog=catalog,maximum_working_bytes=budget)
                    broken=catalog.to_dict()
                    next(f for f in broken['features'] if f['id']=='RST008')['normalization']['ddof']=1
                    with self.assertRaisesRegex(ValueError,'semantics'):
                        build_feature_library_training_view(view,member,selection=selection,destination=root/'bad',
                            raw_view=raw,raw_bindings=bindings,catalog=FeatureCatalog.from_dict(broken))
            self.assertFalse(np.array_equal(outputs[0]['validity'],outputs[1]['validity']))
            with self.assertRaisesRegex(ValueError,'local dependency mismatch'):
                CatalogFeatureLibrary(library,members[0],catalog).normalize_saved(
                    raw,raw_bindings=[(parent,forged)],selection=selection)
            stack.close()
            with load_feature_library_training_view(training,limits={'maximum_matrix_bytes':budget}) as view:
                rows=list(iter_feature_rows(view,list(range(len(days)*3))))
                for row,value,valid in zip(rows,outputs[0]['values'],outputs[0]['validity']):
                    self.assertEqual(row['values'],[float(value[0]) if valid[0] else None])
            wire=json.loads(Path(training['path']).read_text())
            self.assertEqual(wire['definition']['spec']['catalog_ref'],catalog.identity)
            self.assertEqual(wire['definition']['source_compatibility']['member_catalog_ref'],load_a01_feature_catalog().identity)
            wire['definition']['source_compatibility']['feature_catalog_ref']=load_a01_feature_catalog().identity
            wire.pop('training_adapter_ref'); tampered=write_part(root/'tampered',seal(wire,'training_adapter_ref'),'training_adapter_ref')
            with self.assertRaisesRegex(ValueError,'catalog binding mismatch'):
                load_feature_library_training_view(tampered,limits={'maximum_matrix_bytes':budget})


if __name__=='__main__':unittest.main()
