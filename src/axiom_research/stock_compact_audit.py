"""Explicit compact label replay. Ordinary loaders never import this module."""
from copy import deepcopy
from pathlib import Path
import tempfile

from .stock_artifacts import digest
from .stock_fold_inputs import require
from .stock_compact_store import _view_data, load_stock_feature_view,limits as byte_limits


def audit_stock_ml_batch_inputs(manifest, *, data, feature_inputs=None, limits=None):
    """Replay current reviewed operator against saved fixed Data queries.

    This checks stored output against computation; it is not an independent
    arithmetic oracle and does not certify unknown vendor correction history.
    """
    from .stock_batch import load_stock_ml_batch_inputs, _data
    from .stock_compact_labels import prepare_compact_batch
    expected=manifest['definition']['feature_view']; own=feature_inputs is None
    feature=load_stock_feature_view(Path(expected['source_index']['path']).parent,limits=limits) if own else feature_inputs
    fd=_view_data(feature); saved_batch=None; replay=None
    stats={}
    try:
        saved_batch=load_stock_ml_batch_inputs(manifest,feature_inputs=feature,limits=limits)
        saved=_data(saved_batch)['matrix_state']
        options=deepcopy(manifest['definition']['preparation_options']); budgets=byte_limits(limits)
        for option,limit in (('maximum_resident_bytes','maximum_matrix_bytes'),
                             ('maximum_source_bytes','maximum_source_bytes'),('maximum_parent_bytes','maximum_parent_bytes')):
            options[option]=min(options.get(option,budgets[limit]),budgets[limit])
        with tempfile.TemporaryDirectory(prefix='axiom-compact-audit-') as temporary:
            candidate=prepare_compact_batch(data,feature_inputs=feature,
                fold_specs=manifest['definition']['fold_specs'],destination=temporary,
                preparation_options=options,metrics=stats,_caller_bytes=saved.store.resident_bytes,
                _caller_source_bytes=saved.store.metrics['source_bytes'])
            replay=fd['prepared'].pop(candidate['batch_ref'])
            comparisons=[]
            for left,right in zip(saved.view['fold_targets'],replay.view['fold_targets']):
                require(left['fold_spec']==right['fold_spec'],'audit fold mismatch')
                for role in ('raw_parts','normalized','evaluation'):
                    ld=left[role] if role=='raw_parts' else [left[role]]
                    rd=right[role] if role=='raw_parts' else [right[role]]
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
                        if role=='raw_parts':
                            require(va['definition']['price_view']==vb['definition']['price_view'],
                                    'audit selected vintage/source view mismatch')
                        if role=='normalized':
                            for key in ('cutoff','feature_view_ref','sessions','universe','eligible_keys','eligibility_reasons'):
                                require(va['cohort'][key]==vb['cohort'][key],'audit cohort differs: '+key)
                comparisons.append({'fold_spec_ref':digest(left['fold_spec']),'status':'PASS'})
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
