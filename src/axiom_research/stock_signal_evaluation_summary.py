"""Bounded report/binding reads for UI; no ancestor admission or numerical work."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .stock_artifacts import digest
from .stock_fold_inputs import require
from .saved_read_root import saved_read_root, checked_saved_path


def load_stock_signal_evaluation_summary(path, *, research_root):
    """Return saved v2-v8 report fields with an explicit report-only audit scope.

    input_signal_refs is the original list of sha256 Signal run identities, not
    model refs. The caller must compare evidence_ref/content_digest/contract to
    its registered ArtifactRef. Missing bindings fail closed; never call full.
    """
    require(research_root is not None, 'explicit research_root required')
    marks={}; hashes={}; remaining=16*1024**2
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,'duplicate report JSON key'); value[key]=item
        return value
    def read(name, expected=None):
        nonlocal remaining
        target=checked_saved_path(name)
        before=target.stat(); size=before.st_size
        require(target.is_file() and 0<size<=remaining,'report-only metadata budget exceeded')
        raw=target.read_bytes(); remaining-=len(raw)
        mark=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(len(raw)==size and mark(before)==mark(target.stat()),'saved report changed during read')
        if expected is not None:
            require('sha256:'+sha256(raw).hexdigest()==expected,'saved report byte binding mismatch')
        marks[target]=mark(before)
        hashes[target]="sha256:"+sha256(raw).hexdigest()
        return json.loads(raw,object_pairs_hook=pairs)
    with saved_read_root(research_root):
        path=checked_saved_path(path)
        require(path.is_file(),'select one saved report file, not an evaluation group')
        report=read(path)
        require(report.get('contract_version') in tuple('stock_signal_evidence_v'+str(i) for i in range(2,9)),
                'unsupported report-only Signal contract')
        require(report['content_digest']==digest({k:v for k,v in report.items() if k!='content_digest'}),
                'saved report content identity mismatch')
        refs=report['input_signal_refs']
        require(type(refs) is list and bool(refs) and len(refs)==len(set(refs)) and
                all(type(ref) is str and len(ref)==71 and ref.startswith('sha256:') and
                    all(c in '0123456789abcdef' for c in ref[7:]) for ref in refs),
                'original Signal run refs required')
        require(report['spec_ref']==digest(report['spec']),'saved report statistics spec mismatch')
        if report['contract_version']=='stock_signal_evidence_v2':
            from .stock_signal_evaluation import _identity
            require(report['evidence_ref']==_identity(report),'saved report evidence identity mismatch')
            manifest_path=path.parent/'manifest.json'
            manifest=read(manifest_path)
            require(manifest['contract_version']=='stock_signal_evaluation_manifest_v1' and
                    manifest['evidence_ref']==report['evidence_ref'],'saved report manifest mismatch')
            require(hashes[path]==manifest['files'][path.name],'saved report byte binding mismatch')
        else:
            from .stock_signal_evaluation_projection import _group_key
            manifest=read(path.parent/'manifest.json')
            require(manifest['contract_version']=='stock_signal_evaluation_manifest_v2', 'saved group manifest required')
            require(path.name==report['evidence_ref'][7:]+'.json' and
                    hashes[path]==manifest['files'][path.name],'saved report byte binding mismatch')
            group=read(path.parent/'group.json',manifest['files']['group.json'])
            require(group['contract_version']=='stock_signal_evaluation_group_v1' and
                    group['evaluation_key']==manifest['evaluation_key']==report['evaluation_key']==_group_key(group),
                    'saved evaluation group identity mismatch')
            evidence=report['input_evidence']; name=evidence['signal_name']
            require(name in group['signal_order'] and
                    report['evidence_ref']==digest({'evaluation_key':group['evaluation_key'],'signal_name':name}) and
                    evidence['input_ref']==group['input_ref'] and evidence['admission_receipt_ref']==group['admission_receipt_ref'] and
                    all(report[k]==group[k] for k in ('scope','spec','spec_ref','sample_mask_ref','implementation_ref')) and
                    report['statistics_input_ref']==group['common_input_ref'] and
                    group['implementation_ref']==digest(group['implementation_sources']),
                    'saved report/group binding mismatch')
        for target,mark in marks.items():
            s=target.stat()
            require(mark==(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns),'report binding changed during read')
        result={k:deepcopy(report[k]) for k in ('contract_version','evidence_ref','content_digest','input_signal_refs',
            'label_ref','label_spec','scope','sample_mask_ref','statistics_ref','summary','coverage','status','limitations')}
        def aggregate(value):
            return {k:aggregate(v) if type(v) is dict else deepcopy(v) for k,v in value.items()
                    if k not in ('series','by_session','common_statistics','native_statistics')}
        result['coverage']=aggregate(result['coverage'])
        result.update(audit_scope='report_and_bindings',ancestors_audited=False)
        return result
