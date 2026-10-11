"""Small saved fold controls; selection remains the original target validity."""
from pathlib import Path
from .stock_artifacts import digest, digest_array_rows
from .stock_fold_inputs import require, seal, validate_spec
from .stock_compact_store import fields, sealed, reference
from .stock_label_contracts import _instant


def grid_ranges(days, sessions, width):
    positions={day:i for i,day in enumerate(sessions)}; ranges=[]
    for day in days:
        start=positions[day]*width; stop=start+width
        if ranges and ranges[-1][1]==start: ranges[-1][1]=stop
        else: ranges.append([start,stop])
    return ranges


def expand_ranges(selector):
    return [offset for start,stop in selector['ranges'] for offset in range(start,stop)]


def validate_input(fold, manifest, row_index, common):
    inputs=fold['input_manifest']; spec=fold['fold_spec']
    binding=common.get('model_feature_selection')
    extra=set() if binding is None else {'model_feature_selection_ref'}
    fields(inputs,{'contract_version','prepared_view','fold_control','fold_spec_ref','selectors',
        'core_result_refs','input_ref'}|extra,'exact compact v4 saved inputs required'); sealed(inputs,'input_ref')
    require(inputs.get('model_feature_selection_ref')==(None if binding is None else binding['model_feature_selection_ref']),
            'compact input model selection mismatch')
    input_version='stock_ml_saved_inputs_v5' if 'target_spec' in common else 'stock_ml_saved_inputs_v4'
    require(inputs['contract_version']==input_version and inputs['fold_spec_ref']==digest(spec) and
        inputs['prepared_view']==manifest['prepared_view'] and type(inputs['core_result_refs']) is list and
        len(inputs['core_result_refs'])==1 and reference(inputs['core_result_refs'][0]),'compact v4 input linkage mismatch')
    desc=inputs['fold_control']; fields(desc,{'path','file_digest','fold_control_ref'},'exact fold control descriptor required')
    require(type(desc['path']) is str and Path(desc['path']).is_absolute() and
        reference(desc['file_digest']) and reference(desc['fold_control_ref']),'invalid fold control descriptor')
    roles=inputs['selectors']; fields(roles,{'training','training_labels','inference','evaluation_labels','validation'},
        'exact compact v4 selectors required')
    require(roles['validation'] is None and roles['training']==roles['training_labels'] and
        roles['inference']==roles['evaluation_labels'],'compact v4 selector roles mismatch')
    training,inference=validate_spec(spec,common['calendar']); width=len(common['universe'])
    for role,days,kind in (('training',training,'normalized_valid_rows'),('inference',inference,'complete_grid')):
        selected=roles[role]
        fields(selected,{'contract_version','kind','row_index_ref','ranges','target_refs','selected_count',
            'keys_digest','selector_ref'},'exact compact selector required'); sealed(selected,'selector_ref')
        require(selected['contract_version']=='stock_fold_selector_v1' and selected['kind']==kind and
            selected['row_index_ref']==row_index['row_index_ref'] and type(selected['selected_count']) is int and
            0<=selected['selected_count']<=len(days)*width and reference(selected['keys_digest']) and
            type(selected['target_refs']) is list and bool(selected['target_refs']) and
            all(reference(ref) for ref in selected['target_refs']) and
            type(selected['ranges']) is list and all(type(pair) is list and len(pair)==2 and
                all(type(value) is int for value in pair) for pair in selected['ranges']) and
            selected['ranges']==grid_ranges(days,row_index['sessions'],width),'compact selector range/binding mismatch')
    return training,inference


def selectors_for(record, spec, row_index, training_offsets):
    training,inference=validate_spec(record['fold_spec'],spec['calendar'])
    sessions=spec['feature_sessions']; universe=spec['universe']; width=len(universe)
    def selector(kind,days,refs,count,keys_ref):
        return seal({'contract_version':'stock_fold_selector_v1','kind':kind,
            'row_index_ref':row_index['row_index_ref'],'ranges':grid_ranges(days,sessions,width),
            'target_refs':refs,'selected_count':count,'keys_digest':keys_ref},'selector_ref')
    train=selector('normalized_valid_rows',training,[record['normalized']['target_ref']],
        len(training_offsets),record['training_binding']['training_keys_digest'])
    inference_keys=digest_array_rows([security,day] for day in inference for security in universe)
    infer=selector('complete_grid',inference,[part['target_ref'] for part in record['evaluation_parts']],
        len(inference)*width,inference_keys)
    return {'training':train,'training_labels':train,'inference':infer,'evaluation_labels':infer,'validation':None}


def validate_controls(fold, record, view, manifest, row_index):
    inputs=fold['input_manifest']; spec=fold['fold_spec']; common=view['definition']
    binding=common.get('model_feature_selection')
    extra=set() if binding is None else {'model_feature_selection_ref'}
    fields(record,{'contract_version','fold_spec','raw_parts','normalized','evaluation_parts',
        'core_ref','cohort_ref','training_binding','fold_control_ref'}|extra,'exact fold control required')
    require(record.get('model_feature_selection_ref')==(None if binding is None else binding['model_feature_selection_ref']),
            'compact control model selection mismatch')
    sealed(record,'fold_control_ref')
    v5=inputs['contract_version']=='stock_ml_saved_inputs_v5'
    require(record['contract_version']==('stock_ml_fold_control_v2' if v5 else 'stock_ml_fold_control_v1') and record['fold_spec']==spec and
        record['fold_control_ref']==inputs['fold_control']['fold_control_ref'],'fold control linkage mismatch')
    training,inference=validate_input(fold,manifest,row_index,common)
    binding=record['training_binding']
    if v5:
        fields(binding,{'contract_version','ordered_features','feature_blocks','training_selector_ref',
            'training_row_count','cohort_ref','eligibility_mask','target_refs',
            'training_keys_digest','dependency_ref','binding_ref'},'exact v5 training block binding required')
        sealed(binding,'binding_ref')
        require(binding['contract_version']=='stock_training_binding_v2' and
            binding['cohort_ref']==record['cohort_ref'] and
            binding['target_refs']==[record['normalized']['target_ref']] and
            binding['training_selector_ref']==inputs['selectors']['training']['selector_ref'] and
            binding['ordered_features']==common['ordered_features'], 'training block control linkage mismatch')
        training_ref=binding['binding_ref']
    else:
        fields(binding,{'training_rows_ref','training_row_count','training_keys_digest'},'exact compact training binding required')
        training_ref=binding['training_rows_ref']
    require(reference(record['core_ref']) and reference(record['cohort_ref']) and
        reference(training_ref) and reference(binding['training_keys_digest']) and
        type(binding['training_row_count']) is int and binding['training_row_count']>=0 and
        type(record['raw_parts']) is list and bool(record['raw_parts']) and
        type(record['evaluation_parts']) is list and bool(record['evaluation_parts']),'invalid compact fold control')
    roles=inputs['selectors']; width=len(common['universe'])
    require(inputs['core_result_refs']==[record['core_ref']] and
        roles['training']['target_refs']==[record['normalized']['target_ref']] and
        roles['inference']['target_refs']==[part['target_ref'] for part in record['evaluation_parts']],
        'compact v4 target/Core linkage mismatch')
    require(roles['training']['selected_count']==binding['training_row_count'] and
        roles['training']['keys_digest']==binding['training_keys_digest'] and
        roles['inference']['selected_count']==len(inference)*width and roles['inference']['keys_digest']==
        digest_array_rows([security,day] for day in inference for security in common['universe']),
        'compact selector count/keys mismatch')
    return training,inference


def evaluation_parts(record):
    return record['evaluation_parts'] if 'evaluation_parts' in record else [record['evaluation']]


def price_domain_ranges(folds, calendar, block):
    """Keep date ranges, never the security-major training selectors."""
    require(type(block) is int and block>0,'positive evaluation block required')
    positions={day:i for i,day in enumerate(calendar)}; domains={}
    for fold in folds:
        spec=fold['fold_spec']; training,inference=validate_spec(spec,calendar)
        key=(_instant(spec['fit_cutoff']).isoformat(),'training')
        domains.setdefault(key,[]).extend(grid_ranges(training,calendar,1))
        for day in inference:
            key=(_instant(spec['evaluation_cutoff']).isoformat(),positions[day]//block)
            domains.setdefault(key,[]).append([positions[day],positions[day]+1])
    for key,ranges in domains.items():
        merged=[]
        for start,stop in sorted(ranges):
            if merged and start<=merged[-1][1]: merged[-1][1]=max(stop,merged[-1][1])
            else: merged.append([start,stop])
        domains[key]=merged
    return domains


def validate_price_part(definition, common, domains, block, *, evaluation):
    from .labels import _query_context
    calendar=common['calendar']; positions={day:i for i,day in enumerate(calendar)}
    cutoff=definition['cutoff']; days=definition['sessions']
    bucket=positions[days[0]]//block if evaluation else 'training'
    if evaluation:
        require(all(positions[day]//block==bucket for day in days),'evaluation target crosses fixed date block')
    ranges=domains[(_instant(cutoff).isoformat(),bucket)]
    anchor=max(day for day in calendar if day<=_instant(cutoff).date().isoformat()); wanted={anchor}
    for start,stop in ranges:
        for pos in range(start,stop):
            for offset in (1,5):
                if pos+offset<len(calendar) and calendar[pos+offset]<=anchor: wanted.add(calendar[pos+offset])
    query,_,actual_anchor,actual_cutoff=_query_context(definition['price_view']['context'],calendar)
    require(query['sessions']==sorted(wanted) and actual_anchor==anchor and _instant(cutoff)==actual_cutoff,
        'compact fixed price domain/anchor mismatch')
