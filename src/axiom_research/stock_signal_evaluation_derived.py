"""Opt-in frozen Raw/Derived comparison over the original OOS label owner.

No source selection, label arithmetic, normalization, IC or account code lives
here. Original Derived outputs are admitted once and copied; ordinary saved
reads verify those copies. Reopening every original parent is an explicit audit.
"""
from copy import deepcopy
from pathlib import Path
import os
import tempfile

from .api import semantic_identity,to_dict
from .contracts import ArtifactRef
from .stock_artifacts import digest,file_digest,write_json,_verify_ref
from .stock_compact_store import fields,_size,reference
from .stock_fold_inputs import require
from .stock_label_contracts import _instant,_finite
from .stock_signal_evaluation_inputs import _scope,_select_inputs

VERSION='stock_signal_evaluation_inputs_v5'
_FIELDS={'contract_version','input_id','scope','source_input_ref','signal_order',
    'signal_refs','signal_lineage','derived_inputs','admission_receipt','maximum_resident_bytes'}


def _lineage(signal):
    return {'signal_run_ref':signal['signal_run_ref'],'signal_stage':signal['signal_stage'],
        'signal_plan_ref':signal.get('signal_plan_ref'),
        'parent_signal_refs':deepcopy(signal.get('parent_signal_refs',{}))}


def _identity(root):
    body=deepcopy(root);body.pop('input_id',None)
    body['source_input_ref']=semantic_identity(ArtifactRef(**{
        k:v for k,v in body['source_input_ref'].items() if k!='contract_type'}))
    body['admission_receipt'].pop('receipt_ref',None)
    body['admission_receipt']['source_records']=[{'file_digest':r['file_digest']}
        for r in body['admission_receipt']['source_records']]
    return digest(body)


def _shape(root,ref,scope):
    from .stock_signal_evaluation_projection import _input_ref
    fields(root,_FIELDS,'exact joint Raw/Derived frozen input required')
    require(root['contract_version']==VERSION and root['input_id']==ref.artifact_id==_identity(root),
        'joint frozen input identity mismatch')
    base=_scope(root['scope'])
    require(base==root['scope'] and scope['calendar']==base['calendar'] and
        set(scope['sessions'])<=set(base['sessions']) and set(scope['universe'])<=set(base['universe']),
        'joint evaluation outside the frozen scope')
    source=_input_ref(root['source_input_ref'])
    require(source.artifact_contract_version=='stock_signal_evaluation_inputs_v4',
        'joint input requires the original column OOS label projection')
    require(type(root['maximum_resident_bytes']) is int and root['maximum_resident_bytes']>0,
        'joint input requires a positive owner byte budget')
    require(type(root['signal_order']) is list and bool(root['signal_order']) and
        len(root['signal_order'])==len(set(root['signal_order'])) and
        set(root['signal_order'])==set(root['signal_refs']) and
        type(root['signal_lineage']) is dict and set(root['signal_lineage'])==set(root['signal_order']) and
        type(root['derived_inputs']) is dict and bool(root['derived_inputs']) and
        set(root['derived_inputs'])<set(root['signal_order']), 'joint input axes mismatch')
    receipt=root['admission_receipt'];fields(receipt,{'contract_version','source_records',
        'validation_sources','receipt_ref'},'exact joint admission receipt required')
    _verify_ref(receipt,'receipt_ref')
    require(receipt['contract_version']=='stock_signal_evaluation_admission_v5',
        'joint admission version mismatch')
    from .stock_signal_evaluation_matrix import _sources_table
    _sources_table(receipt['source_records'])
    return source


def _derived_rows(signal,base,scope,raw_admission,seen):
    """Validate frozen stage/parents/keys/clocks, without replaying SignalPlan."""
    try:
        from .stock_compact_store import sealed
        sealed(signal,'signal_run_ref')
        require(signal['contract_version']=='derived_signal_run_v1' and
            signal['score_ref']==digest(signal['rows']) and signal['signal_stage'] in ('daily_zscore','final') and
            signal['score_unit']=='dimensionless' and signal['universe']==base['scope']['universe'] and
            signal['context']['calendar_ref']==base['raw_metadata']['calendar_ref'],
            'joint Derived stage, rows or original calendar mismatch')
        parents={ref:item for items in base['signal_metadata'].values() for item in items
            for ref in [item['signal_run_ref']]}
        require(type(signal['parent_signal_refs']) is dict and bool(signal['parent_signal_refs']) and
            set(signal['parent_signal_refs'].values())<=set(parents),
            'joint Derived must reference the original admitted raw predictions')
        metadata=[parents[ref] for ref in signal['parent_signal_refs'].values()]
        days=metadata[0]['prediction_sessions']
        require(all(item['prediction_sessions']==days for item in metadata) and
            set(days)<=set(base['scope']['sessions']) and set(signal['context']['cutoff_by_session'])==set(days),
            'joint Derived original parent fold/date mismatch')
        require(type(signal['rows']) is list and len(signal['rows'])==len(days)*len(signal['universe']),
            'joint Derived complete original union required')
        model_refs={item['model']['model_ref'] for item in metadata}
        feature_refs={item['feature_ref'] for item in metadata};projected={}
        wanted=set(scope['sessions']);securities=set(scope['universe'])
        for row in signal['rows']:
            fields(row,{'security_id','session','knowledge_cutoff','available_at','score','valid',
                'invalid_reason','source_refs','member','feature_knowledge_cutoff'},'exact Derived row required')
            key=row['security_id'],row['session']
            require(row['session'] in days and row['security_id'] in signal['universe'] and key not in seen,
                'joint Derived duplicate or out-of-fold key')
            seen.add(key)
            knowledge,available,feature=map(_instant,(row['knowledge_cutoff'],row['available_at'],row['feature_knowledge_cutoff']))
            require(feature<=knowledge and available<=knowledge<=_instant(scope['evaluation_cutoff']) and
                row['knowledge_cutoff']==signal['context']['cutoff_by_session'][row['session']] and
                all(_instant(item['model']['simulated_available_at'])<=available for item in metadata),
                'joint Derived original source or inference clock mismatch')
            require(type(row['valid']) is bool and type(row['member']) is bool and
                type(row['source_refs']) is list and all(reference(ref) for ref in row['source_refs']) and
                model_refs|feature_refs<=set(row['source_refs']) and
                ((row['valid'] and _finite(row['score']) and row['invalid_reason'] is None) or
                 (not row['valid'] and row['score'] is None and type(row['invalid_reason']) is str and bool(row['invalid_reason']))),
                'joint Derived value or actual parent dependencies mismatch')
            if row['session'] in wanted and row['security_id'] in securities:
                require(raw_admission['projected'][next(iter(raw_admission['projected']))]['members'][key]['member']==row['member'],
                    'joint Derived differs from frozen historical membership')
                projected[key]=deepcopy(row)
        return projected
    finally:
        signal=saved=base=metadata=parents=row=rows=current=projected=raw_admission=admission=root=selected=closures=lineage=result=scope=derived_inputs=None


def _load_joint_inputs(ref,scope,*,marks=None,include_admission=False,_validate_only=False):
    try:
        from .stock_signal_evaluation_projection import _read_checked,_load_inputs,_check_marks
        marks={} if marks is None else marks
        root,_=_read_checked(ref.uri,ref.content_digest,marks=marks);source=_shape(root,ref,scope)
        source_ref,base,selected,admission=_load_inputs(source,scope,marks=marks,include_admission=True)
        source_ref=selected=None
        resident=_size([root,base,admission],maximum=root['maximum_resident_bytes'])
        require(root['scope']==base['scope'] and
            root['signal_order']==[*base['signal_order'],*root['derived_inputs']] and
            all(root['signal_refs'][name]==base['signal_refs'][name] for name in base['signal_order']),
            'joint original raw axes, scope or refs mismatch')
        require(all(root['signal_lineage'][name]==[_lineage(item) for item in base['signal_metadata'][name]]
            for name in base['signal_order']),'joint original raw stage lineage mismatch')
        for name,descriptors in root['derived_inputs'].items():
            require(type(name) is str and bool(name) and type(descriptors) is list and bool(descriptors),
                'joint named original Derived folds required')
            rows={};seen=set();refs=[];closures=[];lineage=[]
            for descriptor in descriptors:
                fields(descriptor,{'file','file_digest','signal_run_ref'},'exact frozen Derived descriptor required')
                require(reference(descriptor['signal_run_ref']) and descriptor['file']==descriptor['signal_run_ref'][7:]+'.json',
                    'joint frozen Derived locator mismatch')
                signal,_=_read_checked(Path(ref.uri).parent/descriptor['file'],descriptor['file_digest'],marks=marks)
                signal_bytes=_size(signal)
                require(resident+signal_bytes*3+65536<=root['maximum_resident_bytes'],
                    'joint current fold projection exceeds the shared byte budget')
                require(signal['signal_run_ref']==descriptor['signal_run_ref'],'joint frozen Derived identity mismatch')
                current=_derived_rows(signal,base,scope,admission,seen)
                resident+=_size(current)+65536
                rows.update(current);current=None;refs.append(signal['signal_run_ref'])
                lineage.append(_lineage(signal))
                closures.append({'signal_input':deepcopy(descriptor),'parent_signal_refs':deepcopy(signal['parent_signal_refs']),
                    'signal_plan_ref':signal['signal_plan_ref'],'score_ref':signal['score_ref'],'signal_stage':signal['signal_stage']})
            require(refs==root['signal_refs'][name] and lineage==root['signal_lineage'][name],
                'joint frozen Derived ref order or lineage mismatch')
            admission['projected'][name]={'rows':rows,'members':next(iter(admission['projected'].values()))['members']}
            admission['refs'][name]=refs;admission['closures'][name]=closures
            require(resident<=root['maximum_resident_bytes'],'joint cumulative projection exceeds the shared byte budget')
        _check_marks(marks)
        admission['signal_lineage']=deepcopy(root['signal_lineage'])
        if _validate_only:return ref,root,None
        result=(ref,root,_select_inputs(admission,scope))
        return (*result,admission) if include_admission else result
    finally:
        signal=saved=base=metadata=parents=row=rows=current=projected=raw_admission=admission=root=selected=closures=lineage=result=scope=derived_inputs=None


def save_stock_derived_signal_evaluation_inputs(raw_input_ref,*,derived_inputs,scope,destination,
    maximum_resident_bytes=512*1024**2):
    """Freeze original saved Derived frames beside the original frozen OOS refs."""
    try:
        from .stock_signal_evaluation_projection import _input_ref,_read_checked,_verify_root,_check_marks
        from .stock_derived_signal import load_stock_derived_signal
        from axiom_engine.core.stock_signal import validate_derived_signal
        from axiom_engine.core import SignalFrame
        scope=_scope(scope);ref=_input_ref(raw_input_ref);marks={}
        require(ref.artifact_contract_version=='stock_signal_evaluation_inputs_v4','original frozen column OOS input required')
        base,_=_read_checked(ref.uri,ref.content_digest,marks=marks);_verify_root(base,ref,scope)
        require(scope==base['scope'] and type(derived_inputs) is dict and bool(derived_inputs) and
            not set(derived_inputs)&set(base['signal_order']),'joint full frozen scope and distinct Derived names required')
        require(type(maximum_resident_bytes) is int and maximum_resident_bytes>0,'positive joint input budget required')
        destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
        sources={};descriptors={};refs=deepcopy(base['signal_refs'])
        lineage={name:[_lineage(item) for item in items] for name,items in base['signal_metadata'].items()}
        with tempfile.TemporaryDirectory(prefix='.joint-signal-inputs-',dir=destination) as temporary:
            stage=Path(temporary)/'complete';stage.mkdir()
            for name,paths in derived_inputs.items():
                require(type(name) is str and bool(name) and type(paths) is list and bool(paths),'named Derived fold paths required')
                descriptors[name]=[];refs[name]=[];lineage[name]=[]
                for path in paths:
                    saved=load_stock_derived_signal(path);signal=saved.to_dict()
                    _size(signal,maximum=maximum_resident_bytes,retained=_size([base,descriptors,refs,sources]))
                    validate_derived_signal(SignalFrame.from_dict(signal))
                    for source,expected in {**saved._manifest['parent_files'],
                        str(saved.path/'signal.json'):saved._manifest['signal_file_digest'],
                        str(saved.path/'manifest.json'):file_digest(saved.path/'manifest.json')}.items():
                        require(source not in sources or sources[source]==expected,'conflicting joint original source pin')
                        sources[source]=expected
                    filename=saved.identity[7:]+'.json';write_json(stage/filename,signal)
                    descriptors[name].append({'file':filename,'file_digest':file_digest(stage/filename),
                        'signal_run_ref':saved.identity});refs[name].append(saved.identity)
                    lineage[name].append(_lineage(signal))
                    saved=signal=None
            receipt={'contract_version':'stock_signal_evaluation_admission_v5','source_records':[
                {'path':path,'file_digest':sources[path]} for path in sorted(sources)],
                'validation_sources':{Path(__file__).name:file_digest(__file__)}}
            receipt['receipt_ref']=digest(receipt)
            root={'contract_version':VERSION,'scope':scope,'source_input_ref':to_dict(ref),
                'signal_order':[*base['signal_order'],*descriptors],'signal_refs':refs,'derived_inputs':descriptors,
                'signal_lineage':lineage,'admission_receipt':receipt,'maximum_resident_bytes':maximum_resident_bytes}
            root['input_id']=_identity(root);write_json(stage/'manifest.json',root)
            staged=ArtifactRef(artifact_type='StockSignalEvaluationInputs',artifact_id=root['input_id'],
                artifact_contract_version=VERSION,content_digest=file_digest(stage/'manifest.json'),uri=str(stage/'manifest.json'))
            _load_joint_inputs(staged,scope,_validate_only=True)
            for path,expected in sources.items():require(file_digest(path)==expected,'joint original source changed during freezing')
            _check_marks(marks);target=destination/root['input_id'][7:]
            final=ArtifactRef(artifact_type=staged.artifact_type,artifact_id=staged.artifact_id,
                artifact_contract_version=VERSION,content_digest=staged.content_digest,uri=str(target/'manifest.json'))
            if target.exists():_load_joint_inputs(final,scope,_validate_only=True);return final
            try:os.rename(stage,target)
            except OSError:
                if not target.exists():raise
                _load_joint_inputs(final,scope,_validate_only=True)
            return final
    finally:
        signal=saved=base=metadata=parents=row=rows=current=projected=raw_admission=admission=root=selected=closures=lineage=result=scope=derived_inputs=None


def _audit_joint_input(ref):
    from .stock_signal_evaluation_projection import _read_checked,_audit_input
    root,_=_read_checked(ref.uri,ref.content_digest)
    _load_joint_inputs(ref,_scope(root['scope']),_validate_only=True)
    _audit_input(root['source_input_ref'])
    for row in root['admission_receipt']['source_records']:
        require(file_digest(row['path'])==row['file_digest'],'joint original Derived source differs from frozen admission')
    return ref
