"""Bounded saved-table lifecycle checks; no Data, Feature execution or model fit."""
from copy import deepcopy
import gc
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

from axiom_research import load_stock_feature_view
from axiom_research.stock_artifacts import _read, write_json
from axiom_research.stock_compact_store import (
    _view_data, clear_feature_window, feature_rows, set_feature_window)
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_matrix_storage import write_buffer
import test_stock_model_feature_selection as selection_tests


class FeatureFactLifecycleTests(unittest.TestCase):
    def fixture(self, root): return selection_tests.ModelFeatureSelectionTests().fixture(root)

    def test_full_and_subset_share_one_fact_owner_and_release_native_json(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path=self.fixture(Path(temp)); selection=deepcopy(f.selection[:3])
            for subset_first in (True,False):
                with self.subTest(subset_first=subset_first), \
                     load_stock_feature_view(path,residency='sequential') as view:
                    original=view.to_dict(); state=_view_data(view); store=state['store']
                    first={'model_feature_selection':selection} if subset_first else {}
                    second={} if subset_first else {'model_feature_selection':selection}
                    set_feature_window(view,[0,1,2],**first)
                    block=state['blocks'][state['day_blocks'][0]]
                    facts=block['model_rows'] if subset_first else block['rows']
                    metadata=block['descriptors'][0]['metadata']
                    self.assertFalse(metadata['path'] in store.json,'native wide JSON was retained')
                    self.assertNotIn(metadata['path'],store.charges)
                    self.assertEqual(store.hashes[metadata['path']],metadata['file_digest'])
                    mark=store.marks[metadata['path']]; decodes=store.metrics['json_decode_calls']
                    if subset_first: self.assertGreater(store.metrics['released_json_bytes'],0)
                    set_feature_window(view,[0,1,2],**second)
                    self.assertIs(block['rows'],facts)
                    self.assertIs(block['model_rows'],facts)
                    self.assertEqual(store.metrics['json_decode_calls'],decodes)
                    self.assertEqual(store.marks[metadata['path']],mark)
                    self.assertEqual(feature_rows(view,[0,1,2]),f.day(f.days[0])[0])
                    self.assertEqual(view.to_dict(),original)
                    self.assertTrue(all(not a.flags.writeable
                        for _,arrays in block['parts'] for a in arrays.values()))
                    # 300 identical cell flags/clocks are shared vectors, not
                    # one independent Python fact for every row and column.
                    self.assertLess(len(set(map(int,facts.availability))),len(facts.member))
                    self.assertEqual(len(set(map(int,facts.validity))),2)
                    refs=[weakref.ref(facts),*(weakref.ref(a)
                        for _,arrays in block['parts'] for a in arrays.values())]
                    resident=store.resident_bytes
                    released=sum(store.charges[p] for p in store.arrays)
                    released+=block['charge']+block['model_charge']+block.get('row_fact_charge',0)
                    facts=block=None
                    clear_feature_window(view); gc.collect()
                    self.assertTrue(all(ref() is None for ref in refs))
                    self.assertEqual(store.resident_bytes,resident-released)
                    self.assertEqual(store.arrays,{})
                    self.assertTrue(all('row_facts' not in b for b in state['blocks']))

    def test_later_part_validates_original_flags_without_json_redecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); f,path=self.fixture(root); index=_read(path/'index.json')
            part=index['partitions'][9]; descriptor=part['buffers']['value_validity']
            flags=[bool(v) for v in Path(descriptor['path']).read_bytes()]
            flags[part['columns'].index(f.columns[299])]=True  # A's original flag is false.
            part['buffers']['value_validity']=write_buffer(path,flags,dtype='bool_u8',shape=descriptor['shape'])
            index['partitions'][9]=seal({k:v for k,v in part.items() if k!='partition_ref'},'partition_ref')
            write_json(path/'index.json',index)
            with load_stock_feature_view(path,residency='sequential') as view:
                set_feature_window(view,[0],model_feature_selection=deepcopy(f.selection[:3]))
                store=_view_data(view)['store']
                with patch.object(store,'read_json',side_effect=AssertionError('native JSON reread')), \
                     self.assertRaisesRegex(ValueError,'buffer/metadata clock or validity'):
                    set_feature_window(view,[0],model_feature_selection=deepcopy(f.selection[299:]))
                self.assertTrue(all('row_facts' not in b for b in _view_data(view)['blocks']))

    def test_released_json_epoch_still_rejects_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path=self.fixture(Path(temp))
            with load_stock_feature_view(path,residency='sequential') as view:
                set_feature_window(view,[0],model_feature_selection=deepcopy(f.selection[:3]))
                state=_view_data(view); store=state['store']; block=state['blocks'][state['day_blocks'][0]]
                descriptor=block['descriptors'][0]['metadata']; before=store.hashes[descriptor['path']]
                self.assertNotIn(descriptor['path'],store.json)
                metadata=Path(descriptor['path']); metadata.write_bytes(metadata.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'source changed'):
                    feature_rows(view,[0],model_feature_selection=deepcopy(f.selection[:3]))
                self.assertEqual(store.hashes[descriptor['path']],before)

    def test_retained_post_conversion_failure_releases_facts_and_buffers(self):
        with tempfile.TemporaryDirectory() as temp:
            f,path=self.fixture(Path(temp))
            with load_stock_feature_view(path,residency='sequential') as view:
                state=_view_data(view); store=state['store']; actual=store.release; refs=[]; failed=False
                def fail_after_release(paths):
                    nonlocal failed
                    result=actual(paths)
                    if not failed and 'row_facts' in state['blocks'][state['day_blocks'][0]]:
                        failed=True
                        facts=state['blocks'][state['day_blocks'][0]]['row_facts']
                        refs.append(weakref.ref(facts)); facts=None
                        raise ValueError('synthetic post-conversion failure')
                    return result
                caught=None
                with patch.object(store,'release',new=fail_after_release):
                    try: set_feature_window(view,[0],model_feature_selection=deepcopy(f.selection[:3]))
                    except ValueError as error: caught=error
                self.assertIsNotNone(caught); gc.collect()
                self.assertTrue(refs and all(ref() is None for ref in refs))
                self.assertEqual(store.arrays,{})
                self.assertTrue(all('row_facts' not in b for b in state['blocks']))
                self.assertGreaterEqual(store.resident_bytes,0)
                set_feature_window(view,[0],model_feature_selection=deepcopy(f.selection[:3]))
                self.assertEqual(len(feature_rows(view,[0],model_feature_selection=deepcopy(f.selection[:3]))[0]['values']),3)
