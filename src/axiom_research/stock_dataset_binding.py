"""Pure dataset identity shared by build and readonly fold admission."""
from .stock_artifacts import digest
from .stock_fold_inputs import seal
from .stock_label_contracts import NORMALIZATION_SPEC,TARGET_SEMANTICS
from .stock_target_spec import target_semantics


def dataset_binding(*,common,features,labels,raw_refs,keys,training_ref,excluded,inputs,spec):
    columnar=inputs['contract_version']=='stock_ml_saved_inputs_v5'
    value={'contract_version':'stock_fold_dataset_v4' if columnar else 'stock_fold_dataset_v3',
        'prepared_view_ref':inputs['prepared_view']['prepared_view_ref'],
        'feature_ref':features['feature_ref'],'label_ref':labels['label_ref'],
        'raw_label_refs':raw_refs,'fold_spec_ref':digest(spec),'fit_cutoff':spec['fit_cutoff'],
        'ordered_features':common['ordered_features'],'selectors':inputs['selectors'],
        'training_keys_digest':digest(keys),'training_row_count':len(keys),'excluded':excluded,
        'target_semantics':target_semantics(common) if columnar else TARGET_SEMANTICS,
        'normalization':NORMALIZATION_SPEC,'validation':'none_fixed_parameters_no_early_stopping'}
    value['training_binding_ref' if columnar else 'training_rows_ref']=training_ref
    if columnar:value.update(target_spec=common['target_spec'],label_definition_ref=common['target_spec']['label_definition_ref'])
    return seal(value,'dataset_ref')
