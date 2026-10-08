"""Versioned structural training proof over existing admitted vector blocks."""
from copy import deepcopy
from contextlib import contextmanager

from .stock_artifacts import digest
from .stock_compact_store import (feature_training_blocks, _view_data, fields, reference,
                                  sealed, model_feature_binding,_size)
from .stock_fold_inputs import require, seal
from .stock_matrix_storage import write_part


@contextmanager
def _proof_blocks(feature,offsets,*,store,model_feature_selection=None,metrics=None):
    """The existing two stores share one operation budget during proof work."""
    fstore=_view_data(feature)['store'];blocks=None;temporary=0
    if metrics is not None:
        from .stock_compact_labels import _sync_feature_charge
        _sync_feature_charge(metrics)
    before=(fstore.resident_bytes,fstore.metrics['source_bytes'],store.shared_bytes,store.shared_source_bytes)
    previous=(fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes)
    def sync():
        if metrics is not None:_sync_feature_charge(metrics)
        else:
            store.shared_bytes=before[2]+fstore.resident_bytes-before[0]
            store.shared_source_bytes=before[3]+fstore.metrics['source_bytes']-before[1]
        store.reserve(0)
    try:
        fstore.limits={k:min(v,store.limits[k]) for k,v in previous[0].items()}
        fstore.shared_bytes=max(previous[1],max(0,store.shared_bytes-fstore.resident_bytes)+store.resident_bytes+store.lease_bytes)
        fstore.shared_source_bytes=max(previous[2],max(0,store.shared_source_bytes-fstore.metrics['source_bytes'])+
            store.metrics['source_bytes'])
        blocks=feature_training_blocks(feature,offsets,model_feature_selection=model_feature_selection)
        sync()
        amount=_size(blocks)
        store.reserve(amount);store.lease_bytes+=amount;temporary=amount
        yield blocks
    finally:
        blocks=feature=offsets=None
        if temporary:store.lease_bytes-=temporary
        fstore.limits,fstore.shared_bytes,fstore.shared_source_bytes=previous
        sync()


def training_block_binding(feature, offsets, *, normalized, cohort_ref, selector,
                           store, destination, model_feature_selection=None,metrics=None):
    """Save small immutable proofs without scanning training Feature values."""
    with _proof_blocks(feature,offsets,store=store,model_feature_selection=model_feature_selection,metrics=metrics) as blocks:
        return _training_block_binding(feature,offsets,normalized=normalized,cohort_ref=cohort_ref,selector=selector,
            store=store,destination=destination,model_feature_selection=model_feature_selection,blocks=blocks)


def _training_block_binding(feature, offsets, *, normalized, cohort_ref, selector,
                           store, destination, model_feature_selection,blocks):
    model_binding=model_feature_binding(feature,model_feature_selection)
    ordered_features=(_view_data(feature)['definition']['spec']['ordered_features']
                      if model_binding is None else model_binding['ordered_features'])
    store.reserve(2*_size(blocks)+4096)
    descriptors=[write_part(destination,body,'feature_block_ref') for body in blocks]
    header=store.read_json(normalized,key='target_ref')
    require(header['target_ref']==normalized['target_ref'] and reference(cohort_ref),
            'training target/cohort binding mismatch')
    mask=deepcopy(header['buffers']['validity'])
    return seal({'contract_version':'stock_training_binding_v2',
        'ordered_features':list(ordered_features),
        'feature_blocks':descriptors,'training_selector_ref':selector['selector_ref'],
        'training_row_count':len(offsets),'cohort_ref':cohort_ref,
        'eligibility_mask':mask,'target_refs':[normalized['target_ref']],
        'training_keys_digest':selector['keys_digest'],
        'dependency_ref':digest({'feature_dependencies':[b['dependency_ref'] for b in blocks],
            'ordered_features':list(ordered_features),
            'selector_ref':selector['selector_ref'],'mask_ref':mask['buffer_digest'],
            'target_ref':normalized['target_ref'],'cohort_ref':cohort_ref})},'binding_ref')


def validate_training_block_binding(binding, feature, offsets, *, normalized, cohort_ref,
                                    selector, store, model_feature_selection=None):
    fields(binding,{'contract_version','ordered_features','feature_blocks','training_selector_ref',
        'training_row_count','cohort_ref','eligibility_mask','target_refs',
        'training_keys_digest','dependency_ref','binding_ref'},'exact training block binding required')
    sealed(binding,'binding_ref')
    require(binding['contract_version']=='stock_training_binding_v2' and
        binding['training_row_count']==len(offsets) and binding['cohort_ref']==cohort_ref and
        binding['training_selector_ref']==selector['selector_ref'] and
        binding['training_keys_digest']==selector['keys_digest'] and
        binding['target_refs']==[normalized['target_ref']] and
        binding['eligibility_mask']==normalized['buffers']['validity'],
        'training block target, selector or mask mismatch')
    with _proof_blocks(feature,offsets,store=store,model_feature_selection=model_feature_selection) as blocks:
        _validate_training_block_binding(binding,feature,offsets,normalized=normalized,cohort_ref=cohort_ref,
            selector=selector,store=store,model_feature_selection=model_feature_selection,blocks=blocks)


def _validate_training_block_binding(binding,feature,offsets,*,normalized,cohort_ref,selector,store,
    model_feature_selection,blocks):
    model_binding=model_feature_binding(feature,model_feature_selection)
    ordered_features=(_view_data(feature)['definition']['spec']['ordered_features']
                      if model_binding is None else model_binding['ordered_features'])
    require(len(blocks)==len(binding['feature_blocks']) and
        binding['ordered_features']==list(ordered_features),
        'training block count/order mismatch')
    for actual,descriptor in zip(blocks,binding['feature_blocks']):
        fields(descriptor,{'path','file_digest','feature_block_ref'},'exact Feature block descriptor required')
        saved=store.read_json(descriptor,key='feature_block_ref')
        require(saved==actual and descriptor['feature_block_ref']==actual['feature_block_ref'],
                'training Feature block differs from admitted bytes')
    expected=digest({'feature_dependencies':[b['dependency_ref'] for b in blocks],
        'ordered_features':binding['ordered_features'],'selector_ref':selector['selector_ref'],
        'mask_ref':binding['eligibility_mask']['buffer_digest'],
        'target_ref':normalized['target_ref'],'cohort_ref':cohort_ref})
    require(binding['dependency_ref']==expected,'training dependency closure mismatch')
