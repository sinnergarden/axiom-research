"""Versioned structural training proof over existing admitted vector blocks."""
from copy import deepcopy

from .stock_artifacts import digest
from .stock_compact_store import (feature_training_blocks, _view_data, fields, reference,
                                  sealed, model_feature_binding)
from .stock_fold_inputs import require, seal
from .stock_matrix_storage import write_part


def training_block_binding(feature, offsets, *, normalized, cohort_ref, selector,
                           store, destination, model_feature_selection=None):
    """Save small immutable proofs without scanning training Feature values."""
    blocks=feature_training_blocks(feature,offsets,model_feature_selection=model_feature_selection)
    model_binding=model_feature_binding(feature,model_feature_selection)
    ordered_features=(_view_data(feature)['definition']['spec']['ordered_features']
                      if model_binding is None else model_binding['ordered_features'])
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
    blocks=feature_training_blocks(feature,offsets,model_feature_selection=model_feature_selection)
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
