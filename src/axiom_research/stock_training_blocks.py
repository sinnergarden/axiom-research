"""Published Feature/target selections for rolling model preparation."""
from .stock_artifacts import digest
from .stock_compact_store import _view_data, model_feature_binding
from .stock_fold_inputs import require, seal


def training_block_binding(feature, offsets, *, normalized, cohort_ref, selector,
                           store, destination, model_feature_selection=None,metrics=None):
    """Bind the published Feature view, selected rows and prepared target."""
    value=_view_data(feature);binding=model_feature_binding(feature,model_feature_selection)
    columns=value['definition']['spec']['ordered_features'] if binding is None else binding['ordered_features']
    header=store.read_json(normalized,key='target_ref')
    # This is the fold's actual input selection, not a second proof of every
    # date/column buffer. Existing v3 bindings remain readable.
    return seal({'contract_version':'stock_training_binding_v4',
        'ordered_features':list(columns),'feature_blocks':[],
        'feature_view_ref':value['definition']['feature_view_ref'],
        'training_selector_ref':selector['selector_ref'],'training_row_count':len(offsets),
        'cohort_ref':cohort_ref,'target_validity_mask':header['buffers']['validity'],
        'target_refs':[normalized['target_ref']],'training_keys_digest':selector['keys_digest'],
        'dependency_ref':digest({'feature_view_ref':value['definition']['feature_view_ref'],
            'ordered_features':list(columns),'selector_ref':selector['selector_ref'],
            'target_ref':normalized['target_ref'],'cohort_ref':cohort_ref})},'binding_ref')



def validate_training_block_binding(binding, feature, offsets, *, normalized, cohort_ref,
                                    selector, store, model_feature_selection=None):
    require(binding['training_row_count']==len(offsets) and
            binding['training_selector_ref']==selector['selector_ref'] and
            binding['target_refs']==[normalized['target_ref']], 'training selection differs from prepared target')
