"""Safe YAML transport for existing typed specs, without an expression DSL."""
from pathlib import Path
from copy import deepcopy

from . import contracts
from .api import from_dict, semantic_identity, TYPES
from .stock_artifacts import digest
from .stock_fold_inputs import require

_ROLES = {'feature': contracts.FeaturePlanSpec, 'label': contracts.LabelSpec,
          'dataset': contracts.DatasetSpec, 'model': contracts.TrainingSpec,
          'signal': contracts.SignalPlanSpec,'strategy':contracts.StrategyRecipeDraft}
_MAXIMUM_CONFIG_BYTES = 2 * 1024 * 1024


def _transport_defaults(value):
    if type(value) is list:
        return [_transport_defaults(v) for v in value]
    if type(value) is not dict:
        return value
    out={k:_transport_defaults(v) for k,v in value.items()}
    if out.get('contract_type') in TYPES:
        # Omitted annotations are harmless; required semantic fields and the
        # explicit contract version still go through the original strict API.
        out.setdefault('metadata',{})
    return out


def _read_configuration_yaml(path, *, expected_digest=None):
    """One bounded safe transport parser for specs and concrete schedules."""
    import yaml
    from yaml.events import AliasEvent
    class SpecLoader(yaml.SafeLoader):
        pass
    # Dates are contract strings; require callers to declare all executable
    # values rather than let YAML silently coerce an ISO session into date().
    SpecLoader.yaml_implicit_resolvers={k:[(tag,rx) for tag,rx in entries
        if tag!='tag:yaml.org,2002:timestamp']
        for k,entries in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    def mapping(loader,node,deep=False):
        result={}
        for key_node,value_node in node.value:
            require(key_node.tag!='tag:yaml.org,2002:merge','YAML merge keys are not supported')
            key=loader.construct_object(key_node,deep=deep)
            require(type(key) is str and key not in result,'duplicate or non-string YAML field')
            result[key]=loader.construct_object(value_node,deep=deep)
        return result
    SpecLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
    def read(path):
        path=Path(path)
        require(path.is_file() and path.stat().st_size<=_MAXIMUM_CONFIG_BYTES,
                'configuration file exceeds the fixed ingress byte limit')
        with path.open('rb') as stream: payload=stream.read(_MAXIMUM_CONFIG_BYTES+1)
        require(len(payload)<=_MAXIMUM_CONFIG_BYTES,'configuration file grew past its byte limit')
        if expected_digest is not None:
            from hashlib import sha256
            require(expected_digest=='sha256:'+sha256(payload).hexdigest(), 'configuration source byte digest mismatch')
        text=payload.decode('utf-8')
        require(not any(isinstance(event,AliasEvent) for event in yaml.parse(text,Loader=SpecLoader)),
                'YAML aliases are not supported')
        return yaml.load(text,Loader=SpecLoader)
    return read(path)


def load_stock_experiment_specs(paths):
    """Merge explicitly split files into typed specs and an effective identity.

    Each file is {contract_version: stock_experiment_config_v1, specs: {role:
    <existing typed contract transport>}}. Roles cannot repeat across files.
    One experiment file may instead declare stock_experiment_v1, an explicit
    files list relative to itself, and role-keyed overrides of existing fields.
    Referenced files must be typed-spec files; references cannot recurse.
    Paths, YAML comments and Contract metadata do not enter semantic identity.
    """
    require(type(paths) in (list,tuple) and bool(paths),'explicit configuration paths required')
    paths=[Path(p).resolve() for p in paths];values=[_read_configuration_yaml(p) for p in paths];overrides={}
    if len(values)==1 and type(values[0]) is dict and values[0].get('contract_version')=='stock_experiment_v1':
        experiment=values[0]
        require(set(experiment)=={'contract_version','files','overrides'} and
            type(experiment['files']) is list and bool(experiment['files']) and
            all(type(p) is str and bool(p) for p in experiment['files']) and
            type(experiment['overrides']) is dict,'exact split experiment references/overrides required')
        base=paths[0].parent;paths=[(base/p).resolve() for p in experiment['files']]
        require(len(set(paths))==len(paths),'repeated experiment configuration file')
        values=[_read_configuration_yaml(p) for p in paths];overrides=deepcopy(experiment['overrides'])
    def merge(wire,change):
        require(type(wire) is dict and type(change) is dict and set(change)<=set(wire) and
            not set(change)&{'contract_type','contract_version'},'unknown or protected override field')
        out=deepcopy(wire)
        for key,value in change.items():
            # Parameter keys are backend data, not a field-path DSL. The concrete
            # TrainingSpec resolver rejects unsupported backend parameters.
            if type(value) is dict and type(out[key]) is dict and key not in ('parameters','metadata'):
                out[key]=merge(out[key],value)
            elif key in ('parameters','metadata') and type(value) is dict and type(out[key]) is dict:
                out[key].update(deepcopy(value))
            else:out[key]=deepcopy(value)
        return out
    specs={};wires={}
    for value in values:
        require(type(value) is dict and set(value)=={'contract_version','specs'} and
            value['contract_version']=='stock_experiment_config_v1' and type(value['specs']) is dict and
            bool(value['specs']), 'exact typed-spec configuration envelope required')
        for role,wire in value['specs'].items():
            require(role in _ROLES and role not in wires,'unknown or repeated configuration role')
            wires[role]=wire
    require(set(overrides)<=set(wires),'override names an absent configuration role')
    for role,wire in wires.items():
        if role in overrides:wire=merge(wire,overrides[role])
        spec=from_dict(_transport_defaults(wire))
        require(type(spec) is _ROLES[role],'configuration role has the wrong typed spec')
        specs[role]=spec
    refs={name:semantic_identity(spec) for name,spec in sorted(specs.items())}
    if 'dataset' in specs and 'label' in specs:
        require(semantic_identity(specs['dataset'].label)==refs['label'],'configuration Dataset and Label differ')
    if 'dataset' in specs and 'model' in specs:
        require(specs['model'].dataset_name==specs['dataset'].name,'configuration model names a different Dataset')
    return {'specs':specs,'spec_refs':refs,'configuration_ref':digest({
        'contract_version':'stock_effective_configuration_v1','spec_refs':refs}),
        'configuration_files':[str(p) for p in paths],'explicit_overrides':overrides}


def load_stock_sequential_configuration(path):
    """Resolve split typed specs and the existing concrete fold-v3 schedule.

    The Dataset transport contains explicit folds, scope and owner budgets. It
    does not reinterpret the R0 DatasetSpec's session-count field as calendar
    years, expand a scheduling language, or invent feature/fit/model clocks.
    """
    from .stock_fold_inputs import validate_spec
    from .stock_signal_evaluation_inputs import _scope
    path=Path(path).resolve();wire=_read_configuration_yaml(path)
    require(type(wire) is dict and set(wire)=={'contract_version','specification_files','dataset_file'} and
        wire['contract_version']=='stock_sequential_configuration_v1' and
        type(wire['specification_files']) is list and bool(wire['specification_files']) and
        all(type(p) is str and bool(p) for p in wire['specification_files']) and
        type(wire['dataset_file']) is str and bool(wire['dataset_file']),
        'exact sequential specification and Dataset file references required')
    spec_paths=[(path.parent/p).resolve() for p in wire['specification_files']]
    configuration=load_stock_experiment_specs(spec_paths)
    dataset_path=(path.parent/wire['dataset_file']).resolve();dataset=_read_configuration_yaml(dataset_path)
    compilation=None
    if type(dataset) is dict and dataset.get('contract_version')=='stock_dataset_schedule_v2':
        from .stock_weekly_config import _expand_weekly_dataset
        dataset,compilation=_expand_weekly_dataset(dataset,dataset_path)
    require(type(dataset) is dict and set(dataset)=={'contract_version','fold_specs','scope',
        'preparation_options','model_feature_selection','signal_contexts'} and
        dataset['contract_version']=='stock_dataset_schedule_v1' and
        type(dataset['fold_specs']) is list and bool(dataset['fold_specs']),
        'exact explicit stock Dataset schedule required')
    scope=_scope(dataset['scope']);previous=None;prediction_sessions=[]
    for fold in dataset['fold_specs']:
        require(type(fold) is dict and fold.get('contract_version')=='stock_ml_fold_spec_v3',
            'configured Dataset requires explicit fold-v3 windows and clocks')
        _,sessions=validate_spec(fold,scope['calendar'])
        require(previous is None or fold['oos_trade_sessions'][0]>previous,'configured folds overlap or are unordered')
        previous=fold['oos_trade_sessions'][-1];prediction_sessions.extend(sessions)
    require(prediction_sessions==scope['sessions'],'configured Dataset must cover the exact OOS scope')
    options=dataset['preparation_options'];required={'row_block_sessions','column_block',
        'maximum_resident_bytes','normalization_backend'}
    require(type(options) is dict and required<=set(options)<=required|{
        'maximum_source_bytes','maximum_parent_bytes','control_layout'} and
        options['normalization_backend']=='core_cs_batch_v1' and
        options.get('control_layout','fold_controls_v2')=='fold_controls_v2' and
        all(type(v) is int and v>0 for k,v in options.items() if k not in
            ('normalization_backend','control_layout')),'configured Dataset owner budget/profile mismatch')
    selection=dataset['model_feature_selection']
    require(selection is None or type(selection) is list and bool(selection),'explicit model Feature selection required')
    require(dataset['signal_contexts'] is None or type(dataset['signal_contexts']) is dict,
        'explicit Signal contexts or null required')
    dataset_ref=digest(dataset) if compilation is None else digest({'dataset':dataset,'compiler_ref':compilation['compiler_ref']})
    result={'specification_paths':[str(p) for p in spec_paths],'specification':configuration,
        'dataset':dataset,'dataset_ref':dataset_ref,'configuration_ref':digest({
            'contract_version':'stock_sequential_effective_configuration_v1' if compilation is None else
                'stock_sequential_effective_configuration_v2',
            'specification_ref':configuration['configuration_ref'],'dataset_ref':dataset_ref}),
        'configuration_files':[str(path),str(dataset_path),*configuration['configuration_files']]}
    if compilation is not None:
        result['weekly_compilation']=compilation
        result['configuration_files'].append(compilation['axes_input']['path'])
    return result
