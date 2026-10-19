"""Concentrated synthetic comparisons; never production coverage evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'axiom-engine/tests'),str(ROOT/'axiom-research/tests')]
from test_conformance import setup,node,rolling,cs,col
from test_feature_columns import packed
from test_incremental_feature_library import FixedSource,decoded
from axiom_engine.core import FeaturePlan,compile_feature_columns,execute_feature_columns
from axiom_engine.core.feature_columns import _needed
from axiom_engine.feature_library import FeatureLibrary
from axiom_engine.feature_library.storage import write_buffer
from axiom_engine.feature_library.store import VerifiedMatrixStore
from axiom_research.feature_input_buffers import segmented_buffer,security_segmented_buffer,SegmentedFeatureBuffer


def legacy_module(repo, revision, path, name):
    text=subprocess.check_output(['git','-C',str(ROOT/repo),'show',revision+':'+path],text=True)
    module=types.ModuleType(name);module.__file__=str(ROOT/repo/path)
    sys.modules[name]=module;exec(compile(text,module.__file__,'exec'),module.__dict__)
    return module


def equal(a,b,*,sources=True):
    for name in ('values','validity','available_at_utc_us'):
        np.testing.assert_array_equal(a[name],b[name])
    for name,dictionary in (('reason_codes','reason_dictionary'),('source_codes','source_sets')):
        if name=='source_codes' and not sources:continue
        actual=[a[dictionary][int(v)] for v in a[name].ravel()]
        expected=[b[dictionary][int(v)] for v in b[name].ravel()]
        assert actual==expected,name


def math_checks():
    old=legacy_module('axiom-engine','7d47a28','src/axiom_engine/core/feature_columns.py','axiom_engine.core._baseline_columns')
    nodes=[node('lag','shift',['x'],{'periods':756}),rolling('short','y',2,1),
           cs('rank',src='x'),cs('winsor','cs_winsorize','x'),cs('normalized','cs_zscore','x')]
    values={s:[[float((i+j)%5),None if i==757 and j==1 else float(i+j)] for i in range(760)]
            for j,s in enumerate(('A','B','C'))}
    p,f,c=setup(values,nodes,['lag','short','rank','winsor','normalized'],schema=[col('x',stage='fact'),col('y',stage='fact')])
    f['rows'][-1]['availability'][0]=c['sessions'][-1]+'T23:30:00Z'
    inputs=packed(p,f,c);days=c['sessions'][-2:];plan=FeaturePlan.from_dict(p)
    expected=old.execute_feature_columns(old.compile_feature_columns(plan),inputs=inputs,output_sessions=days,max_working_bytes=16*1024**2)
    dense=execute_feature_columns(compile_feature_columns(plan),inputs=inputs,output_sessions=days,max_working_bytes=16*1024**2)
    equal(dense,expected)
    needed=_needed(p,[758,759]);sparse=copy.deepcopy(inputs)
    sparse['field_rows']={name:sorted(needed[name]) for name in ('x','y')}
    for category in ('values','validity','reason_codes','available_at_utc_us','source_codes'):
        sparse[category]={name:values[sparse['field_rows'][name]] for name,values in inputs[category].items()}
    result=execute_feature_columns(compile_feature_columns(plan),inputs=sparse,output_sessions=days,max_working_bytes=16*1024**2)
    equal(result,expected)
    assert sparse['field_rows']['y']==[757,758,759]
    assert 2 in sparse['field_rows']['x'] and 3 in sparse['field_rows']['x']
    return {'outputs':len(days)*3*5,'field_rows':sparse['field_rows'],'raw_normalized_ties_missing_long_shift_rolling_pit':'exact'}


def adjustment_checks(root):
    from test_stock_feature_library import saved_source
    from axiom_research.stock_feature_columns import FeatureColumnSource
    from axiom_research.stock_feature_library import _SourceArrays
    from axiom_research.stock_matrix_reader import VerifiedMatrixStore as ResearchStore
    old=legacy_module('axiom-research','3138294','src/axiom_research/stock_feature_columns.py','axiom_research._baseline_feature_columns')
    descriptor=saved_source(root/'adjustment');manifest=json.loads(Path(descriptor['path']).read_text())
    with ResearchStore(maximum_source_bytes=2**28,maximum_matrix_bytes=2**26) as store:
        arrays=_SourceArrays(manifest,store);before=old.FeatureColumnSource(manifest,arrays);after=FeatureColumnSource(manifest,arrays)
        days=manifest['spec']['read_sessions']
        for day in days[-2:]:
            count=days.index(day)+1
            expected=before.select(day,history_sessions=count,fields=['close','raw_factor'])
            selected=after.select(day,history_sessions=count,fields=['close','raw_factor'],rows=[0,count-1],include_reference=False)
            for group in ('values','validity','availability','missing_reasons'):
                for field in selected[group]:np.testing.assert_array_equal(selected[group][field],expected[group][field][[0,count-1]])
    return {'output_day_anchors':2,'selected_rows_per_field':2,'price_native_factor_clocks':'exact'}


def increment_checks(root):
    source=FixedSource();days=source.c['sessions'][2:5];columns=['raw_x','mean_x','z_x']
    library=FeatureLibrary(root/'increment',maximum_working_bytes=16*1024**2)
    with library.execution_session():
        _,first=library.ensure_features(source.request(columns,days[:2]),source)
        before={str(p):(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns) for p in (root/'increment/partitions').glob('*.json')}
        _,added_day=library.ensure_features(source.request(columns,days),source)
        assert added_day['core_calls']==1 and added_day['computed_output_cells']==9
        source.add_raw_field()
        descriptor,added_column=library.ensure_features(source.request(columns+['raw_z'],days),source)
        assert added_column['computed_output_cells']==9
        assert all((Path(p).stat().st_ino,Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==mark for p,mark in before.items())
        with library.open_feature_view(descriptor) as view:actual=view.project()
        fresh=FeatureLibrary(root/'one-shot',maximum_working_bytes=16*1024**2)
        expected_descriptor,_=fresh.ensure_features(source.request(columns+['raw_z'],days),source)
        with fresh.open_feature_view(expected_descriptor) as view:expected=view.project()
        equal(actual,expected,sources=False)
        # Expanded fixture changes its source id; reused blocks retain original provenance.
        assert {s for group in actual['source_sets'] for s in group}=={'facts','facts_v2'}
    # Native input append references old files, including exact sparse indexing.
    values=np.arange(30,dtype='<f8').reshape(5,3,2)
    old=write_buffer(root/'slabs',values[:4].reshape(-1),dtype='float64_le',shape=[4,3,2])
    new=write_buffer(root/'slabs',values[4:].reshape(-1),dtype='float64_le',shape=[1,3,2])
    descriptor=segmented_buffer([old,new])
    with VerifiedMatrixStore(maximum_matrix_bytes=2**20) as store:
        array=SegmentedFeatureBuffer(descriptor,store)
        np.testing.assert_array_equal(array[:],values)
        np.testing.assert_array_equal(array[np.ix_([0,4],[0,2],[1])],values[np.ix_([0,4],[0,2],[1])])
        np.testing.assert_array_equal(array.column(1)[4,[0,2]],values[4,[0,2],1])
        left=write_buffer(root/'slabs',values[:,:2].reshape(-1),dtype='float64_le',shape=[5,2,2])
        right=write_buffer(root/'slabs',values[:,2:].reshape(-1),dtype='float64_le',shape=[5,1,2])
        merged=security_segmented_buffer([left,right],[['A','B'],['C']],['A','B','C'])
        np.testing.assert_array_equal(SegmentedFeatureBuffer(merged,store)[:],values)
        appended=segmented_buffer([merged,merged])
        nested=SegmentedFeatureBuffer(appended,store)
        np.testing.assert_array_equal(nested[:],np.concatenate([values,values]))
        np.testing.assert_array_equal(nested.column(1)[[1,6],2],values[[1,1],2,1])
    return {'appended_day':added_day,'added_raw_column':added_column,'old_partition_files_unchanged':len(before),
            'one_shot_comparison':'exact','date_and_security_slabs':'exact','old_input_bytes_copied':0}



def membership_merge_checks(root):
    import pandas as pd
    from types import SimpleNamespace
    from axiom_research.qsys_history_inputs import _merged_pool_complete
    days=['2020-01-02','2020-01-03'];symbols=['A','B','C']
    values=np.array([[True,True,False]]*2,dtype=bool);known=np.ones((2,3),dtype=bool)
    buffers={name:write_buffer(root/'pool',array.astype('u1').reshape(-1),dtype='bool_u8',shape=[2,3])
             for name,array in (('member',values),('member_known',known))}
    spec={'read_sessions':days,'universe':symbols,'snapshot':'fixed','pit_policy':'best_effort_vendor_v1',
          'cutoff_by_session':{d:d+'T12:30:00Z' for d in days}}
    class Data:
        complete=True
        def members(self,*,snapshot,query):
            assert snapshot=='fixed' and tuple(query.symbols)==tuple(symbols)
            frame=pd.DataFrame([{'security_id':s,'session':d,'is_member':s!='C'} for d in query.sessions for s in symbols])
            meta={'by_key':[{'security_id':s,'session':d,'missing_reason':None} for d in query.sessions for s in symbols],
                  'reference_pools':{d:{'complete':self.complete,'member_count':2} for d in query.sessions}}
            return SimpleNamespace(frame=frame,field_meta={'is_member':meta})
    data=Data()
    with VerifiedMatrixStore(maximum_matrix_bytes=2**20) as store:
        result=_merged_pool_complete(data,spec=spec,buffers=buffers,store=store,root=root/'complete',universe_id='test')
        assert bool(store.buffer(result).all())
        data.complete=False
        result=_merged_pool_complete(data,spec=spec,buffers=buffers,store=store,root=root/'unknown',universe_id='test')
        assert not bool(store.buffer(result).any())
    return {'full_pool_rechecked':True,'unknown_preserved':True,'old_values_recomputed':False}


def main():
    destination=Path(sys.argv[1]);destination.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    result={'math':math_checks(),'adjustment':adjustment_checks(destination),'increment':increment_checks(destination),'membership_merge':membership_merge_checks(destination)}
    result.update(status='PASS',elapsed_seconds=time.monotonic()-started)
    (destination/'checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
