"""Research saves Core's actual DerivedSignal; no signal mathematics here."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import errno
import tempfile

from .api import validate,from_dict,_encode,semantic_identity
from .contracts import SignalPlanSpec
from .stock_artifacts import digest,file_digest,write_json,_read
from .stock_compact_store import OwnedStore,fields,sealed,reference
from .stock_fold_inputs import require,seal
from .stock_experiment_config import _transport_defaults


@dataclass(frozen=True)
class StockDerivedSignal:
    path: Path
    _signal: dict
    _manifest: dict
    reused: bool=False

    def to_dict(self):
        sealed(self._signal,'signal_run_ref');return deepcopy(self._signal)
    @property
    def identity(self):return self.to_dict()['signal_run_ref']
    @property
    def artifact(self):
        sealed(self._manifest,'content_digest');self.to_dict()
        return deepcopy(self._manifest['signal_artifact'])
    def engine_input_binding(self):
        """Neutral Runtime v2 input only; this does not execute an account."""
        sealed(self._manifest,'content_digest');signal=self.to_dict()
        return {'kind':'derived',**{k:signal[k] for k in
            ('signal_run_ref','signal_plan_ref','score_ref','implementation_ref','signal_stage')},
            'signal_artifact':self.artifact,'parent_inputs':deepcopy(self._manifest['parent_inputs'])}


def _artifact(path,wire,kind,identity):
    # Engine's neutral ArtifactRef binds canonical Document content, distinct
    # from the file-byte digests used by the Research manifest below.
    return {'artifact_type':kind,'artifact_id':identity,'contract_version':wire.get('contract_version','1'),
        'manifest_uri':str(Path(path).resolve()),'content_digest':digest(wire)}


def _parent(run):
    fold=run.to_dict();model=run.model();prediction=run.predictions();spec=fold['definition']['fold_spec']
    spec_artifact=_artifact(run.path/'fold.json',spec,'StockFoldSpec',digest(spec))
    spec_artifact['manifest_uri']+='#definition/fold_spec'
    binding={'kind':'raw','fold_ref':fold['fold_ref'],'fold_spec_ref':digest(spec),
        'model_ref':model['model_ref'],'feature_ref':prediction['feature_ref'],
        'signal_run_ref':prediction['signal_run_ref'],'fold_spec_artifact':spec_artifact,
        'model_metadata_artifact':_artifact(run.path/'model.json',model,'StockModelRelease',model['model_ref']),
        'prediction_artifact':_artifact(run.path/'predictions.json',prediction,'StockPredictionRun',prediction['signal_run_ref'])}
    manifest=_read(run.path/'manifest.json')
    require(manifest['fold_ref']==run.identity,'DerivedSignal raw fold manifest changed')
    files={str(run.path/name):ref for name,ref in manifest['files'].items()}
    files[str(run.path/'manifest.json')]=file_digest(run.path/'manifest.json')
    return prediction,binding,files


def build_stock_derived_signal(signal_plan,*,prediction_inputs,context,destination,batch=None):
    """One actual Core SignalPlan call over original fully admitted raw folds.

    prediction_inputs maps each declared alias to a saved fold directory. Use
    one call per common OOS fold, preserving each real model and parent ref.
    A supplied active batch reuses its existing owner; no fit/predict occurs.
    """
    from .stock_fold_artifacts import load_stock_ml_fold
    from axiom_engine.core import execute_signal_plan,StockPredictionFrame
    plan=from_dict(_transport_defaults(signal_plan)) if type(signal_plan) is dict else signal_plan
    require(type(plan) is SignalPlanSpec,'typed SignalPlanSpec required');validate(plan,require_resolved=True)
    wire=_transport_defaults(_encode(plan,True));configuration_ref=semantic_identity(plan)
    require(type(prediction_inputs) is dict and set(prediction_inputs)=={i.alias for i in plan.inputs},
        'exact declared raw prediction aliases required')
    parents={};bindings={};files={}
    for alias,path in prediction_inputs.items():
        run=load_stock_ml_fold(path,batch=batch)
        raw,binding,pins=_parent(run);parents[alias]=StockPredictionFrame.from_dict(raw);bindings[alias]=binding
        for name,ref in pins.items():
            require(name not in files or files[name]==ref,'conflicting DerivedSignal parent pin');files[name]=ref
    definition={'contract_version':'stock_derived_signal_request_v1','signal_plan':wire,
        'configuration_spec_ref':configuration_ref,'context':deepcopy(context),
        'parent_signal_refs':{a:b['signal_run_ref'] for a,b in bindings.items()}}
    target=Path(destination).resolve()/digest(definition)[7:]
    if target.exists():
        saved=load_stock_derived_signal(target)
        require(saved._manifest['definition']==definition,'saved DerivedSignal request mismatch')
        return StockDerivedSignal(saved.path,saved.to_dict(),deepcopy(saved._manifest),True)
    # Core owns CS normalization, weighted multiplication/addition order,
    # stage, validity, clocks and the actual resulting frame identities.
    signal=execute_signal_plan(wire,inputs=parents,context=deepcopy(context)).to_dict()
    sealed(signal,'signal_run_ref')
    require(signal['contract_version']=='derived_signal_run_v1' and
        signal['parent_signal_refs']==definition['parent_signal_refs'] and signal['score_ref']==digest(signal['rows']),
        'Core DerivedSignal original parent/output identity mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.derived-signal-',dir=target.parent) as temporary:
        stage=Path(temporary)/'complete';stage.mkdir();write_json(stage/'signal.json',signal)
        manifest=seal({'contract_version':'stock_derived_signal_manifest_v1','definition':definition,
            'signal_file_digest':file_digest(stage/'signal.json'),
            'signal_artifact':_artifact(target/'signal.json',signal,'DerivedSignal',signal['signal_run_ref']),
            'parent_inputs':bindings,'parent_files':files},'content_digest')
        write_json(stage/'manifest.json',manifest)
        # Verify parent byte pins again after Core; no changed output/input
        # may be made visible as a complete saved signal.
        for name,ref in files.items():require(file_digest(name)==ref,'DerivedSignal parent changed during build')
        try:stage.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY):raise
            saved=load_stock_derived_signal(target)
            require(saved.to_dict()==signal and saved._manifest==manifest,'concurrent DerivedSignal conflict')
            return saved
    return load_stock_derived_signal(target)


def load_stock_derived_signal(path,*,limits=None,_shared_bytes=0):
    """Hash/ref-load frozen Core output and saved parents; never import Core.

    Original raw folds were fully admitted during build. This loader checks
    their immutable saved output pins; it does not redo Data/Feature audit,
    signal arithmetic, training, prediction or an account.
    """
    path=Path(path).resolve()
    with OwnedStore(limits,shared_bytes=_shared_bytes) as store:
        manifest=store.read_json({'path':str(path/'manifest.json')},key='content_digest')
        fields(manifest,{'contract_version','definition','signal_file_digest','signal_artifact','parent_inputs',
            'parent_files','content_digest'},'exact saved DerivedSignal manifest required')
        require(manifest['contract_version']=='stock_derived_signal_manifest_v1','unsupported DerivedSignal manifest')
        definition=manifest['definition']
        fields(definition,{'contract_version','signal_plan','configuration_spec_ref','context','parent_signal_refs'},
            'exact saved DerivedSignal request required')
        require(definition['contract_version']=='stock_derived_signal_request_v1','unsupported DerivedSignal request')
        plan=from_dict(definition['signal_plan'])
        require(type(plan) is SignalPlanSpec,'saved typed SignalPlanSpec required');validate(plan,require_resolved=True)
        require(semantic_identity(plan)==definition['configuration_spec_ref'],
            'saved DerivedSignal typed configuration identity mismatch')
        signal=store.read_json({'path':str(path/'signal.json'),'file_digest':manifest['signal_file_digest']},key='signal_run_ref')
        fields(signal,{'contract_version','signal_run_ref','score_ref','signal_plan','signal_plan_ref','parent_signal_refs',
            'implementation_ref','signal_stage','score_semantics','score_unit','universe','rows','context','limitations'},
            'exact saved Core DerivedSignal required')
        require(signal['contract_version']=='derived_signal_run_v1' and signal['score_ref']==digest(signal['rows']) and
            reference(signal['signal_plan_ref']) and reference(signal['implementation_ref']) and
            signal['signal_plan_ref']==manifest['definition']['configuration_spec_ref'] and
            signal['parent_signal_refs']==manifest['definition']['parent_signal_refs']=={
                a:b['signal_run_ref'] for a,b in manifest['parent_inputs'].items()} and
            signal['context']==manifest['definition']['context'] and
            manifest['signal_artifact']==_artifact(path/'signal.json',signal,'DerivedSignal',signal['signal_run_ref']),
            'saved DerivedSignal original identity/context mismatch')
        def neutral(value):
            if type(value) is list:return [neutral(v) for v in value]
            if type(value) is dict:return {k:neutral(v) for k,v in value.items() if k!='contract_type'}
            return value
        require(signal['signal_plan']==neutral(definition['signal_plan']) and
            set(manifest['parent_inputs'])=={i.alias for i in plan.inputs} and
            signal['signal_stage']==next(n.output_stage for n in plan.nodes if n.name==plan.output) and
            signal['score_semantics']==plan.score_semantics and signal['score_unit']=='dimensionless',
            'saved DerivedSignal plan/stage/semantics mismatch')
        require(type(manifest['parent_files']) is dict and bool(manifest['parent_files']),
            'saved DerivedSignal parent byte pins required')
        for name,ref in manifest['parent_files'].items():
            require(Path(name).is_absolute() and reference(ref),'fixed DerivedSignal parent byte pin required')
            store.read({'path':name,'file_digest':ref})
        for alias,binding in manifest['parent_inputs'].items():
            require(type(alias) is str and binding['kind']=='raw','real raw DerivedSignal parent required')
            for field,key in (('prediction_artifact','signal_run_ref'),('model_metadata_artifact','model_ref')):
                ref=binding[field];name=ref['manifest_uri']
                require(name in manifest['parent_files'],'DerivedSignal original parent outside saved closure')
                parent=store.read_json({'path':name,'file_digest':manifest['parent_files'][name]},key=key)
                require(parent[key]==binding[key] and digest(parent)==ref['content_digest'],
                    'DerivedSignal original model/prediction binding mismatch')
            locator=binding['fold_spec_artifact']['manifest_uri'];name,fragment=locator.rsplit('#',1)
            require(fragment=='definition/fold_spec' and name in manifest['parent_files'],
                'original DerivedSignal fold spec locator required')
            fold=store.read_json({'path':name,'file_digest':manifest['parent_files'][name]},key='content_digest')
            require(fold['fold_ref']==binding['fold_ref'] and digest(fold['definition']['fold_spec'])==
                binding['fold_spec_ref']==binding['fold_spec_artifact']['content_digest'] and
                (fold['model_ref'],fold['signal_run_ref'],fold['feature_ref'])==(
                    binding['model_ref'],binding['signal_run_ref'],binding['feature_ref']),
                'DerivedSignal original fold/model/feature closure mismatch')
        store.check()
        from .stock_compact_store import _size
        store.reserve(_size([signal,manifest],maximum=store.maximum_matrix_bytes,
            retained=store.shared_bytes+store.resident_bytes)*3+4096)
        return StockDerivedSignal(path,deepcopy(signal),deepcopy(manifest))
