"""Explicit compact label replay. Ordinary loaders never import this module."""
from copy import deepcopy
from pathlib import Path
import tempfile

from .stock_artifacts import digest
from .stock_fold_inputs import require
from .stock_compact_store import _view_data, load_stock_feature_view,limits as byte_limits
from .stock_compact_controls import evaluation_parts


def _check_budget(state):
    state._sync_shared(); state.store.reserve(0)
    require(state.store.shared_source_bytes+state.store.metrics['source_bytes']<=
        state.store.limits['maximum_source_bytes'],'audit combined source byte budget exceeded')


def _compare_fold(saved,replay,left,right):
    va=ra=vb=rb=x=y=None
    try:
        require(left['fold_spec']==right['fold_spec'],'audit fold mismatch')
        for role in ('raw_parts','normalized','evaluation'):
            ld=left[role] if role=='raw_parts' else evaluation_parts(left) if role=='evaluation' else [left[role]]
            rd=right[role] if role=='raw_parts' else evaluation_parts(right) if role=='evaluation' else [right[role]]
            require(len(ld)==len(rd),'audit target count mismatch')
            for a,b in zip(ld,rd):
                va,ra=saved.targets[a['target_ref']]; vb,rb=replay.targets[b['target_ref']]
                require(len(ra)==len(rb),'audit target grid mismatch')
                for x,y in zip(ra,rb):
                    for key in ('security_id','feature_session','start_session','end_session',
                                'valid','invalid_reason','label_available_at','raw_available_at','raw_return','return'):
                        lx,ry=x.get(key),y.get(key)
                        equal=lx.hex()==ry.hex() if type(lx) is type(ry) is float else lx==ry
                        require(equal,'audit saved target differs: '+key)
                if role in ('raw_parts','evaluation'):
                    require(va['definition']['price_view']==vb['definition']['price_view'],
                            'audit selected vintage/source view mismatch')
                if role=='normalized':
                    for key in ('cutoff','feature_view_ref','sessions','universe','eligible_keys','eligibility_reasons'):
                        require(va['cohort'][key]==vb['cohort'][key],'audit cohort differs: '+key)
    finally:
        va=ra=vb=rb=x=y=left=right=None


def audit_stock_ml_batch_inputs(manifest, *, data, feature_inputs=None, limits=None):
    """Replay current reviewed operator against saved fixed Data queries.

    This checks stored output against computation; it is not an independent
    arithmetic oracle and does not certify unknown vendor correction history.
    """
    from .stock_batch import load_stock_ml_batch_inputs, _data
    from .stock_compact_labels import prepare_compact_batch
    expected=manifest['definition']['feature_view']; own=feature_inputs is None
    feature=load_stock_feature_view(Path(expected['source_index']['path']).parent,limits=limits,residency='sequential') if own else feature_inputs
    fd=_view_data(feature); saved_batch=None; replay=None
    stats={}
    try:
        saved_batch=load_stock_ml_batch_inputs(manifest,feature_inputs=feature,limits=limits,residency='sequential')
        saved=_data(saved_batch)['matrix_state']
        options=deepcopy(manifest['definition']['preparation_options']); budgets=byte_limits(limits)
        options['control_layout']='inline_v3' if manifest['contract_version']=='stock_ml_batch_inputs_v3' else 'fold_controls_v1'
        for option,limit in (('maximum_resident_bytes','maximum_matrix_bytes'),
                             ('maximum_source_bytes','maximum_source_bytes'),('maximum_parent_bytes','maximum_parent_bytes')):
            options[option]=min(options.get(option,budgets[limit]),budgets[limit])
        with tempfile.TemporaryDirectory(prefix='axiom-compact-audit-') as temporary:
            candidate=prepare_compact_batch(data,feature_inputs=feature,
                fold_specs=manifest['definition']['fold_specs'],destination=temporary,
                preparation_options=options,metrics=stats,_caller_bytes=saved.store.resident_bytes,
                _caller_source_bytes=saved.store.metrics['source_bytes'],model_feature_selection=
                    manifest['definition'].get('model_feature_selection',{}).get('selection'))
            replay=fd['prepared'].pop(candidate['batch_ref'])
            saved_fixed=(saved._fixed_shared_bytes,saved._fixed_shared_source_bytes)
            replay_external=(max(0,replay._fixed_shared_bytes-saved.store.resident_bytes),
                max(0,replay._fixed_shared_source_bytes-saved.store.metrics['source_bytes']))
            comparisons=[]
            require(len(saved.batch['folds'])==len(replay.batch['folds']),'audit fold count mismatch')
            for lf,rf in zip(saved.batch['folds'],replay.batch['folds']):
                left=right=None
                try:
                    saved._fixed_shared_bytes=saved_fixed[0]+replay.store.resident_bytes+replay.store.lease_bytes
                    saved._fixed_shared_source_bytes=saved_fixed[1]+replay.store.metrics['source_bytes']
                    _check_budget(saved)
                    left=saved._parts(lf['input_manifest'],lf['fold_spec'])
                    saved._activate(lf,left)
                    replay._fixed_shared_bytes=replay_external[0]+saved.store.resident_bytes+saved.store.lease_bytes
                    replay._fixed_shared_source_bytes=replay_external[1]+saved.store.metrics['source_bytes']
                    _check_budget(replay)
                    right=replay._parts(rf['input_manifest'],rf['fold_spec'])
                    replay._activate(rf,right)
                    _compare_fold(saved,replay,left,right)
                    comparisons.append({'fold_spec_ref':digest(lf['fold_spec']),'status':'PASS'})
                finally:
                    left=right=None
                    saved._release_window(); replay._release_window()
                    saved._fixed_shared_bytes,saved._fixed_shared_source_bytes=saved_fixed
                    saved._sync_shared()
                    replay._fixed_shared_bytes=replay_external[0]+saved.store.resident_bytes+saved.store.lease_bytes
                    replay._fixed_shared_source_bytes=replay_external[1]+saved.store.metrics['source_bytes']
                    replay._sync_shared()
            result={'contract_version':'stock_compact_audit_v1','input_batch_ref':manifest['batch_ref'],
                'status':'PASS','folds':comparisons,'replay_calls':deepcopy(stats),
                'limitations':['Replay uses the reviewed Raw operator and existing Core; independent oracle is separate.',
                    'Original best-effort PIT and unknown historical correction coverage remain limited.',
                    'No fit, prediction, supplier or account execution.']}
            result['audit_ref']=digest(result); return result
    finally:
        if replay is not None: replay.close()
        if saved_batch is not None: saved_batch.close()
        if own: feature.close()
