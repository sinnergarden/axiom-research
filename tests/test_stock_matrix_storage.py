"""Synthetic exact storage/projection checks; provider and Core execution are zero."""
from copy import deepcopy
from pathlib import Path
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research.stock_artifacts import digest, file_digest, _read, write_json
from axiom_research.stock_feature_inputs import build_stock_feature_inputs, load_stock_feature_inputs
from axiom_research.stock_matrix_storage import write_buffer, write_part, instant_us
from axiom_research.stock_fold_inputs import seal
from test_stock_feature_inputs import SyntheticInputs, NoData, InterruptedPreparation, file_state


class SyntheticWideInputs(SyntheticInputs):
    """Same synthetic clock/source proof, only ordered column configuration varies."""
    def __init__(self, root, width):
        super().__init__(root,feature_count=3)
        chosen=deepcopy(self.catalog.select(self.selection)[0])
        self.columns=['SYN'+str(i).zfill(3) for i in range(width)]
        self.selection=[{'id':c,'semantic_version':'1.0.0'} for c in self.columns]
        self.spec.update(ordered_features=self.columns,feature_selection=self.selection)
        identity=self.catalog.identity
        self.catalog=type('SyntheticCatalog',(),{'identity':identity,'select':lambda _,selection:
            [{**deepcopy(chosen),'id':c} for c in self.columns]})()

    def day(self, day):
        rows,proof=super().day(day)
        proof['core_plan']['outputs']=[{'node':c,'column':{'name':c,'dtype':'float64',
            'unit':'dimensionless','stage':'cross_sectional','missing':'preserve'}} for c in self.columns]
        for i,row in enumerate(rows):
            row['values']=[None if (i+j)%7==0 else (-0.0 if j%11==0 else float(i+j)+math.ulp(1.0))
                           for j in range(len(self.columns))]
            row['validity']=[v is not None for v in row['values']]
            row['reasons']=[[] if valid else ['REFERENCE_MISSING','synthetic original reason']
                            for valid in row['validity']]
            row['availability']=[None if j%5==0 else day+'T20:00:00.123456+08:00'
                                 for j in range(len(self.columns))]
            row['source_refs']=[proof['core_frame_ref'],digest(proof['core_plan'])]
        return rows,proof

    def matrix(self, *, progress=None, options=None):
        self.data=NoData()
        with patch('axiom_research.feature_catalog.load_feature_catalog',return_value=self.catalog), \
             patch('axiom_research.stock_ml._implementation',return_value=self.implementation), \
             patch('axiom_research.stock_ml._environment',return_value=self.environment), \
             patch('axiom_research.stock_matrix_feature_producer.prepare_matrix_qlib',side_effect=self.prepare), \
             patch('axiom_research.stock_matrix_feature_producer.iter_matrix_feature_days',side_effect=self.iterate):
            return build_stock_feature_inputs(self.data,spec=self.spec,destination=self.root/'saved',
                progress=progress,storage_options=options or {'layout':'matrix_v1',
                    'row_block_sessions':2,'column_block':32,'maximum_resident_bytes':64*1024**2})


class MatrixStorageTests(unittest.TestCase):
    def _reseal_index(self, path, value):
        value=deepcopy(value); value.pop('content_digest',None)
        value['feature_inputs_ref']=digest({k:value[k] for k in ('contract_version','definition_ref',
            'qlib_manifest','schema_digest','row_index','source_selection','partitions')})
        write_json(path/'index.json',seal(value,'content_digest'))

    def test_binary64_exact_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            values=[0.0,-0.0,math.nextafter(1.0,2.0),1e-308,-1e308]
            first=write_buffer(temp,values,dtype='float64_le',shape=[1,len(values)])
            before=file_state(Path(temp))
            second=write_buffer(temp,values,dtype='float64_le',shape=[len(values)])
            self.assertEqual(Path(first['path']).read_bytes(),struct.pack('<5d',*values))
            self.assertEqual(first['path'],second['path'])
            self.assertEqual(first['buffer_digest'],file_digest(first['path']))
            self.assertEqual(before,file_state(Path(temp)))
            Path(first['path']).write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'existing matrix buffer changed'):
                write_buffer(temp,values,dtype='float64_le',shape=[len(values)])

    def test_strict_codes_shapes_and_clocks(self):
        with tempfile.TemporaryDirectory() as temp:
            for values,dtype,shape in [([True],'int32_le',[1]),([1],'bool_u8',[1]),
                    ([float('nan')],'float64_le',[1]),([1],'int64_le',[2]),
                    ([2**31],'int32_le',[1]),([1],'float64_le',[True])]:
                with self.subTest(dtype=dtype,values=values):
                    with self.assertRaises(ValueError): write_buffer(temp,values,dtype=dtype,shape=shape)
            self.assertEqual(instant_us('1970-01-01T08:00:00.123456+08:00'),123456)
            self.assertEqual(instant_us('1969-12-31T23:59:59.999999Z'),-1)
            part=seal({'contract_version':'synthetic','instant':123456},'metadata_ref')
            desc=write_part(temp,part,'metadata_ref')
            self.assertEqual(_read(desc['path']),part)
            self.assertNotEqual(desc['file_digest'],desc['metadata_ref'])

    def test_same_writer_route_at_six_158_and_300_columns(self):
        for width in (6,158,300):
            with self.subTest(width=width),tempfile.TemporaryDirectory() as temp:
                fixture=SyntheticWideInputs(Path(temp),width)
                saved=fixture.matrix()
                try:
                    wire=saved.to_dict(); self.assertEqual(wire['contract_version'],'stock_feature_inputs_v2')
                    self.assertEqual(len(wire['schema']),width)
                    self.assertNotIn('feature_parents',wire)
                    self.assertEqual(len(wire['partitions']),2*math.ceil(width/32))
                    expected=[r for d in fixture.days for r in fixture.day(d)[0]]
                    rows=saved.row_metadata(list(range(len(expected))))
                    self.assertEqual(rows,expected)
                    for a,b in zip(rows,expected):
                        for actual,old in zip(a['values'],b['values']):
                            if old is not None: self.assertEqual(struct.pack('<d',actual),struct.pack('<d',old))
                    metadata_paths={p['metadata']['path'] for p in wire['partitions']}
                    self.assertEqual(len(metadata_paths),2)
                    before=file_state(saved.path)
                    again=fixture.matrix()
                    try:
                        self.assertEqual(again.identity,saved.identity)
                        self.assertEqual(fixture.computed,fixture.days)
                        self.assertEqual(before,file_state(saved.path))
                    finally: again.close()
                finally: saved.close()

    def test_checkpoint_resumes_only_unfinished_complete_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6)
            def interrupt(update): raise InterruptedPreparation('published block')
            with self.assertRaises(InterruptedPreparation): fixture.matrix(progress=interrupt)
            target,= (Path(temp)/'saved').iterdir()
            checkpoint=_read(target/'checkpoint.json')
            self.assertEqual(checkpoint['contract_version'],'stock_feature_inputs_checkpoint_v2')
            self.assertFalse((target/'index.json').exists())
            saved=fixture.matrix()
            try:
                self.assertEqual(fixture.computed,fixture.days)
                self.assertEqual(saved.row_metadata(list(range(9))),[r for d in fixture.days for r in fixture.day(d)[0]])
            finally: saved.close()

    def test_corrupt_progress_block_is_rejected_before_complete_index(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6)
            def corrupt_final(update):
                if update['completed_dates']!=len(fixture.days): return
                target,=(Path(temp)/'saved').iterdir()
                checkpoint=_read(target/'checkpoint.json')
                source=Path(checkpoint['partitions'][-1]['buffers']['values']['path'])
                raw=source.read_bytes(); source.write_bytes(bytes([raw[0]^1])+raw[1:])
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                fixture.matrix(progress=corrupt_final)
            target,=(Path(temp)/'saved').iterdir()
            self.assertTrue((target/'checkpoint.json').exists())
            self.assertFalse((target/'index.json').exists())

    def test_matrix_options_reject_conflicting_shard_and_bool(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6)
            with self.assertRaisesRegex(ValueError,'shard_sessions conflicts'):
                build_stock_feature_inputs(NoData(),spec=fixture.spec,destination=Path(temp)/'none',
                    shard_sessions=3,storage_options={'layout':'matrix_v1'})
            with self.assertRaisesRegex(ValueError,'positive integer'):
                fixture.matrix(options={'layout':'matrix_v1','row_block_sessions':True})
            self.assertFalse((Path(temp)/'saved').exists())

    def test_suspended_producer_budget_blocks_publish_before_children(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6)
            original=fixture.iterate
            limit=64*1024**2
            observed=[]
            def retained_generator(*args, **kwargs):
                stats=kwargs['stats']; reserve=kwargs['caller_retained_bytes']
                observed.append(reserve())
                for rows,proof in original(*args, **kwargs):
                    # Both the pending writer block and this producer value
                    # fit independently. Their combined publication cannot.
                    stats['yield_live_bytes']=limit-1
                    try:
                        yield rows,proof
                    finally:
                        stats['yield_live_bytes']=0
                    observed.append(reserve())
            fixture.iterate=retained_generator
            with patch('axiom_research.stock_feature_inputs._write_feature_matrix_block',
                       side_effect=AssertionError('overbudget child publication')) as publish:
                with self.assertRaisesRegex(ValueError,'writer/producer combined working set'):
                    fixture.matrix()
                self.assertEqual(publish.call_count,0)
            self.assertGreater(observed[1],observed[0])
            target,=(Path(temp)/'saved').iterdir()
            self.assertFalse((target/'index.json').exists())
            self.assertFalse((target/'checkpoint.json').exists())

    def test_rehashed_cross_day_query_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6); saved=fixture.matrix()
            index=saved.to_dict(); path=saved.path; saved.close()
            selection=_read(index['source_selection']['path']); selection.pop('source_selection_ref')
            selection['feature_rows'][0]['query_refs']=selection['feature_rows'][1]['query_refs']
            index['source_selection']=write_part(path,seal(selection,'source_selection_ref'),'source_selection_ref')
            self._reseal_index(path,index)
            with self.assertRaisesRegex(ValueError,'selection differs from actual'):
                load_stock_feature_inputs(path)

    def test_rehashed_null_negative_zero_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6); saved=fixture.matrix()
            index=saved.to_dict(); path=saved.path; saved.close()
            partition=index['partitions'][0]; old=partition['buffers']['values']
            values=list(struct.unpack('<'+str(math.prod(old['shape']))+'d',Path(old['path']).read_bytes()))
            self.assertEqual(values[0],0.0); values[0]=-0.0
            partition['buffers']['values']=write_buffer(path,values,dtype='float64_le',shape=old['shape'])
            partition.pop('partition_ref'); partition['partition_ref']=digest(partition)
            self._reseal_index(path,index)
            with self.assertRaisesRegex(ValueError,'canonical finite/null'):
                load_stock_feature_inputs(path)

    def test_saved_v2_loader_imports_no_data_core_or_training_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=SyntheticWideInputs(Path(temp),6); saved=fixture.matrix(); path=saved.path; saved.close()
            code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'axiom_data','axiom_engine','qlib','lightgbm','pandas'}:
   raise AssertionError('forbidden loader import: '+name)
sys.meta_path.insert(0,Block())
from axiom_research import load_stock_feature_inputs
with load_stock_feature_inputs(sys.argv[1]) as value:
 assert value.to_dict()['contract_version']=='stock_feature_inputs_v2'
 assert len(value.row_metadata([0]))==1
'''
            result=subprocess.run([sys.executable,'-B','-c',code,str(path)],capture_output=True,text=True,
                                  env=os.environ.copy(),timeout=12)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
