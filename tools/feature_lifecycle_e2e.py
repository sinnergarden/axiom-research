"""One explicit real-input lifecycle acceptance; never starts full history.

Run only after implementation is frozen. An exclusive start receipt prevents
accidental repetition. Failure is recorded once and is not retried. The eight
output sessions retain the full physical universe and the catalog's warmup.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import traceback


def footprint(root):
    seen=set();result={'unique_logical_bytes':0,'allocated_bytes':0,
                      'multiply_linked_logical_bytes':0,'files':0}
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        st=path.stat();key=(st.st_dev,st.st_ino)
        if key in seen:continue
        seen.add(key);result['files']+=1
        result['unique_logical_bytes']+=st.st_size
        result['allocated_bytes']+=st.st_blocks*512
        if st.st_nlink>1:result['multiply_linked_logical_bytes']+=st.st_size
    result['scope']='destination only; hardlinks counted once here; APFS shared extents and whole-platform peak unmeasured'
    return result


def acceptance(cfg):
    import numpy as np
    from prepare_qsys_history import run
    from axiom_engine.feature_library import FeatureLibrary
    from axiom_research.qsys_catalog import load_qsys_feature_catalog
    from axiom_research.stock_feature_library import SavedColumnFeatureSource,CatalogFeatureLibrary
    from axiom_research.stock_feature_library_adapter import load_feature_library_training_view
    from axiom_research.stock_compact_store import (feature_eligibility_columns,training_matrix,
        feature_training_blocks,clear_feature_window)
    from axiom_research.stock_matrix_storage import instant_us
    from axiom_research.stock_compact_batch import TargetRows,ConcatRows,target_eligibility_reasons
    from axiom_data import Data
    root=Path(cfg['destination']);catalog=load_qsys_feature_catalog()
    selection=[s.to_dict() for s in catalog.default_selection]
    assert len(cfg['output_sessions'])==8 and len(selection)==401
    result=run(cfg)
    assert result['dates']==8 and result['columns']==401 and result['securities']==3396
    assert result['warmup']==756 and result['read_dates']==764
    limits={'maximum_source_bytes':cfg['maximum_source_bytes'],
            'maximum_matrix_bytes':cfg['working_bytes'],'maximum_parent_bytes':64*1024**2}
    library=FeatureLibrary(root/'library',maximum_source_bytes=cfg['maximum_source_bytes'],
                           maximum_working_bytes=cfg['working_bytes'])
    # Changed request grouping must query real coverage, with no Core calls.
    before={str(p.relative_to(root/'library')) for p in (root/'library').rglob('*') if p.is_file()}
    with SavedColumnFeatureSource(result['source'],maximum_source_bytes=cfg['maximum_source_bytes'],
                                  maximum_working_bytes=cfg['working_bytes']) as source:
        owner=CatalogFeatureLibrary(library,source,catalog)
        _,warm=owner.ensure(selection=selection,output_sessions=cfg['output_sessions'][1:7],normalized=False)
        assert warm['raw']['core_calls']==0 and warm['raw']['computed_output_cells']==0
        known=source.source.arrays['member_known']
        member=source.source.arrays['member']
        unknown=int((~known).sum())
        assert not bool((member & ~known).any())
    after={str(p.relative_to(root/'library')) for p in (root/'library').rglob('*') if p.is_file()}
    assert not any('/partitions/' in '/'+p or p.startswith('buffers/') for p in after-before)
    # Additional Axiom normalization is an explicit small request; it reuses
    # existing raw columns instead of duplicating all 401 normalized columns.
    with SavedColumnFeatureSource(result['source'],maximum_source_bytes=cfg['maximum_source_bytes'],
                                  maximum_working_bytes=cfg['working_bytes']) as source:
        owner=CatalogFeatureLibrary(library,source,catalog)
        norm_view,norm=owner.ensure(selection=selection[:2],output_sessions=cfg['output_sessions'][-1:],normalized=True)
        assert norm['raw']['core_calls']==0
        with library.open_feature_view(norm_view) as view:
            assert len(view.describe()['columns'])==2
    # A fresh training owner must not open Data or re-admit source ancestry.
    original_reader=Data._reader;original_source=SavedColumnFeatureSource.__init__
    def prohibited(*args,**kwargs):raise AssertionError('native consumer reopened source ancestry')
    Data._reader=prohibited;SavedColumnFeatureSource.__init__=prohibited
    consumer_started=time.monotonic();cells_count=valid_count=training_rows=0
    try:
        with library.open_feature_view(result['feature_view']) as view, \
                load_feature_library_training_view(result['training_view'],limits=limits) as handle:
            cap=view.describe();securities=cap['security_ids'];count=len(securities)
            # Pick a pair with real eligible members on the last day; this is
            # a consumer subset, never a narrower feature preparation universe.
            final=view.project([(s,cfg['output_sessions'][-1]) for s in securities])
            valid_by_column=final['validity'].sum(axis=0)
            chosen=sorted(range(401),key=lambda i:(-int(valid_by_column[i]),i))[:2]
            subset=[selection[i] for i in chosen];names=[s['id'] for s in subset]
            for ordinal,day in enumerate(cfg['output_sessions']):
                clear_feature_window(handle)
                offsets=list(range(ordinal*count,(ordinal+1)*count))
                original=view.project([(s,day) for s in securities])
                available=original['available_at_utc_us'];valid=original['validity']
                assert not bool((valid & (available>instant_us(day+'T20:30:00+08:00'))).any())
                assert not bool((valid & (available==original['null_available_at_utc_us'])).any())
                cells_count+=int(valid.size);valid_count+=int(valid.sum())
                members,complete,knowledge,maximum=feature_eligibility_columns(handle,offsets,model_feature_selection=subset)
                expected=view.project([(s,day) for s in securities],columns=names)
                assert np.array_equal(complete,expected['validity'].all(axis=1))
                eligible=np.flatnonzero(members & complete).tolist()
                if eligible:
                    selected_offsets=[offsets[i] for i in eligible]
                    actual=training_matrix(handle,selected_offsets,day+'T20:30:00+08:00',model_feature_selection=subset)
                    assert np.array_equal(actual,expected['values'][eligible])
                    proofs=feature_training_blocks(handle,selected_offsets,model_feature_selection=subset)
                    assert proofs and all(p['ordered_features']==names for p in proofs)
                    training_rows+=len(eligible)
                    if ordinal==7:
                        # Boundary controls use the existing compact Label
                        # owner against real Feature membership and clocks.
                        cutoff=day+'T20:30:00+08:00';clock=instant_us(cutoff)
                        arrays={'validity':np.ones(4,dtype='u1'),'reason_codes':np.zeros(4,dtype='<i4'),
                                'end_session':np.array([0,0,0,-1],dtype='<i8'),
                                'availability_validity':np.ones(4,dtype='u1'),
                                'availability':np.array([clock-1,clock,clock+1,clock-1],dtype='<i8')}
                        raw=ConcatRows([TargetRows({'row_count':4,'definition':{'calendar':[day]},
                            'reason_dictionary':[None]},arrays)])
                        reasons=target_eligibility_reasons(raw,handle,[selected_offsets[0]]*4,cutoff,
                            {'target_spec':{'label_spec':{'maturity':{'rule':'all_outcome_dependencies_strictly_before_fit_cutoff'}}}},
                            model_feature_selection=subset)
                        assert reasons==[None,'LABEL_NOT_MATURE','LABEL_NOT_MATURE','LABEL_CLOCK_OR_ENDPOINT_UNKNOWN']
            assert training_rows>0 and cells_count==8*3396*401
            consumer_metrics=handle.metrics
    finally:
        Data._reader=original_reader;SavedColumnFeatureSource.__init__=original_source
    staging=[str(p) for p in (root/'inputs').rglob('.feature-input-*')
             if p.name!='.feature-input-checkpoint.json']
    assert not staging,staging[:5]
    return {'status':'PASS','run':result,'warm_overlap':warm,'requested_normalized_subset':norm,
        'consumer_seconds':time.monotonic()-consumer_started,'consumer_metrics':consumer_metrics,
        'physical_cells':cells_count,'valid_cells':valid_count,'subset_training_rows':training_rows,
        'unknown_membership_cells_preserved':unknown,'unpublished_staging_files':staging,
        'native_consumer_ancestor_calls':0,'label_boundary_controls':4,
        'label_scope':'maturity controls only; no real Label generation/model fit/account replay',
        'footprint':footprint(root),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'full_history_capacity_or_12h_acceptance':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    root=Path(cfg['destination']);root.mkdir(parents=True,exist_ok=True)
    with (root/'e2e-started.json').open('x') as stream:
        json.dump({'config':str(args.config.resolve()),'started_unix':time.time(),'attempt':1},stream)
    started=time.monotonic()
    try:receipt=acceptance(cfg)
    except BaseException as exc:
        receipt={'status':'FAIL','error':type(exc).__name__,'message':str(exc),
                 'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,
                 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'retry_performed':False}
        (root/'e2e-result.json').write_text(json.dumps(receipt,indent=2))
        print(json.dumps(receipt),flush=True)
        raise
    receipt['elapsed_seconds']=time.monotonic()-started
    (root/'e2e-result.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
