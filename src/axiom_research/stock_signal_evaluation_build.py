"""Private build-to-frozen-OOS lifetime; the existing admission/writer is reused."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from weakref import WeakKeyDictionary
import errno
import os
import tempfile

from .stock_artifacts import digest, file_digest, write_json
from .stock_fold_inputs import require
from .stock_compact_store import _size
from .stock_signal_evaluation_inputs import _scope

_WRITERS=WeakKeyDictionary()


def _writer_for(batch):
    writer=_WRITERS.get(batch) if batch is not None else None
    if writer is not None: writer._check()
    return writer


def _descriptor(lease,path):
    return {'path':str(Path(path)/'predictions.json'),
        'file_digest':dict(lease.source_records)[str(Path(path)/'predictions.json')],
        'signal_run_ref':lease.documents['predictions.json']['signal_run_ref']}


def _publish_build_fold(stage,target,*,projection,batch,fold_ref,writer):
    from .stock_signal_evaluation_lease import _admit_build_fold
    stage=Path(stage).resolve(); target=Path(target).resolve()
    with _admit_build_fold(stage,projection=projection,batch=batch) as (lease,run):
        projection._store.validate_boundary()
        try: stage.rename(target)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST,errno.ENOTEMPTY): raise
        else:
            lease._repath_outputs(stage,target)
            writer._consume(_descriptor(lease,target),lease)
            return run
    # The winning inode requires its own saved-byte admission, still borrowing
    # the original build projection. A discarded stage produces no OOS shard.
    with _admit_build_fold(target,projection=projection,batch=batch) as (lease,run):
        require(run.identity==fold_ref,'concurrent matrix fold conflict')
        writer._consume(_descriptor(lease,target),lease)
        return run


def _consume_saved_fold(target,*,batch,writer):
    from .stock_signal_evaluation_lease import _admit_build_fold
    target=Path(target).resolve()
    # One ordinary OOS admission on an exact HIT; no X/y/P or backend call.
    with _admit_build_fold(target,projection=None,batch=batch) as (lease,run):
        writer._consume(_descriptor(lease,target),lease)
        return run


class _BuildOOSWriter:
    """Only small fold controls/proof metadata survive each temporary shard."""
    def __init__(self,batch,scope,destination,name):
        from .stock_batch import _data
        self.batch=batch; self.state=_data(batch)['matrix_state']
        require(self.state.compact and self.state.active==0,'idle compact v4 owner required')
        require(type(name) is str and bool(name),'named Signal required')
        self.scope=_scope(scope); self.name=name; self.manifest=batch._ready_manifest()
        self._manifest_pending=self.state.incomplete
        common=self.manifest['definition']['feature_view']['spec']
        require(self.scope['calendar']==common['calendar'] and
            set(self.scope['universe'])<=set(common['universe']), 'build OOS scope differs from owner')
        self.destination=Path(destination).resolve(); self.destination.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='.build-oos-',dir=self.destination)
        self.stage=Path(self.temporary.name)/'complete'; self.stage.mkdir()
        self.pid=os.getpid(); self.closed=False; self.finished=False; self.charge=0; self.borrowed_owner=False
        self.signal_inputs={name:[]}; self.metadata={name:[]}; self.refs={name:[]}; self.closures={name:[]}
        self.records={}; self.marks={}; self.raw=None; self.lineages={}; self.shards={}; self.previous=None
        self.metrics={'folds':0,'build_projection_borrows':0,'saved_oos_admissions':0,'shards':0,
                      'maximum_fold_rows':0,'retained_control_bytes':0}
        try:
            self._manifest_charge=_size(self.manifest,maximum=self.state.store.maximum_matrix_bytes,
                retained=self.state.store.shared_bytes+self.state.store.resident_bytes+self.state.store.lease_bytes)
            initial=self._manifest_charge+_size(self.scope,maximum=self.state.store.maximum_matrix_bytes,
                retained=self.state.store.shared_bytes+self.state.store.resident_bytes+self.state.store.lease_bytes)
            self._retain(initial+65536)
            self.state._build_oos_borrowers=getattr(self.state,'_build_oos_borrowers',0)+1
            self.borrowed_owner=True
        except BaseException:
            self.close()
            raise

    def _check(self):
        from .stock_signal_evaluation_projection import _check_marks
        require(self.pid==os.getpid(),'build OOS writer belongs to another process')
        require(not self.closed and not self.finished,'build OOS writer is closed')
        self.batch._check_sources()
        _check_marks(self.marks)

    def _retain(self,amount):
        self.state.store.reserve(amount)
        self.state._fixed_shared_bytes+=amount; self.charge+=amount
        self.state._sync_shared()
        self.metrics['retained_control_bytes']=self.charge

    def _require_inputs(self,inputs,spec):
        self._check()
        self._refresh_manifest()
        require(any(f['input_manifest']==inputs and f['fold_spec']==spec for f in self.manifest['folds']),
                'build input outside frozen owner')
        days=sorted(spec['inference_cutoff_by_session'])
        require(self.previous is None or self.previous<days[0],'build OOS folds must be ordered and disjoint')
        require(set(days)&set(self.scope['sessions']),'build fold outside requested OOS dates')

    def _refresh_manifest(self):
        if not self._manifest_pending:return
        current=None
        try:
            from .stock_batch import _data
            value=_data(self.batch)
            if value.get('incomplete',False) and len(value['manifest']['folds'])==len(self.manifest['folds']):return
            current=self.batch._ready_manifest()
            charge=_size(current,maximum=self.state.store.maximum_matrix_bytes,
                retained=self.state.store.shared_bytes+self.state.store.resident_bytes+self.state.store.lease_bytes)
            self._retain(charge)
            self.manifest=current;current=None
            self.state._fixed_shared_bytes-=self._manifest_charge;self.charge-=self._manifest_charge
            self._manifest_charge=charge;self.state._sync_shared()
            self._manifest_pending=value.get('incomplete',False)
        finally:current=value=None

    def _consume(self,descriptor,lease):
        from .stock_signal_evaluation_compact import _admit_compact
        from .stock_signal_evaluation_projection import _shard,_check_marks
        try:
            self._check()
            require(lease._batch is self.batch,'build lease owner mismatch')
            fold=lease.documents['fold.json']; spec=fold['definition']['fold_spec']
            self._require_inputs(fold['definition']['input_manifest'],spec)
            days=sorted(spec['inference_cutoff_by_session'])
            selected=[d for d in days if d in self.scope['sessions']]
            local={**self.scope,'sessions':selected}
            # Current lease plus one detached OOS child coexist. No training matrix
            # is included in this admission or the emitted date shards.
            self.state.store.reserve(lease._charge*2+len(days)*len(lease.common['universe'])*16384+65536)
            admitted,records,marks,_=_admit_compact({self.name:[descriptor]},None,local,self.batch,
                _borrowed_lease=lease,_manifest=self.manifest)
            raw=admitted['raw']
            if self.raw is None:
                self.raw={k:v for k,v in raw.items() if k not in ('sources','label_inputs','label_shard_refs','label_ref')}
                self.raw.update(sources={},label_inputs=[],label_shard_refs={},label_ref=None)
            require(all(self.raw[k]==raw[k] for k in ('mode','label_spec','calendar_ref','snapshot','pit_policy')),
                    'build OOS Raw definition conflict')
            new_records={p:r for p,r in records.items() if p not in self.records}
            new_sources={r:s for r,s in raw['sources'].items() if r not in self.raw['sources']}
            new_lineages={digest(v):v for v in raw['label_inputs'] if digest(v) not in self.lineages}
            delta=[admitted['metadata'],admitted['refs'],admitted['closures'],descriptor,
                   new_records,{p:marks[Path(p)] for p in new_records},new_sources,new_lineages,
                   raw['label_shard_refs']]
            self._retain(_size(delta,maximum=self.state.store.maximum_matrix_bytes,
                retained=self.state.store.shared_bytes+self.state.store.resident_bytes+self.state.store.lease_bytes)+65536)
            for p,r in records.items():
                require(p not in self.records or self.records[p]==r,'conflicting build OOS source pin')
                require(Path(p) not in self.marks or self.marks[Path(p)]==marks[Path(p)],'build OOS source changed')
            for r,s in raw['sources'].items():
                require(r not in self.raw['sources'] or self.raw['sources'][r]==s,'conflicting build OOS Raw source')
            for day in selected:
                require(day not in self.shards,'duplicate build OOS date')
                name=day+'.json'; write_json(self.stage/name,_shard(admitted,local,day))
                self.shards[day]={'file':name,'file_digest':file_digest(self.stage/name)}
            self.records.update(records); self.marks.update(marks)
            self.raw['sources'].update(new_sources); self.lineages.update(new_lineages)
            self.raw['label_shard_refs'].update(raw['label_shard_refs'])
            self.signal_inputs[self.name].append(deepcopy(descriptor))
            for key in ('metadata','refs','closures'): getattr(self,key)[self.name].extend(admitted[key][self.name])
            self.previous=days[-1]
            self.metrics['folds']+=1; self.metrics['shards']+=len(selected)
            self.metrics['maximum_fold_rows']=max(self.metrics['maximum_fold_rows'],len(days)*len(lease.common['universe']))
            self.metrics['build_projection_borrows' if not lease._owns_projection else 'saved_oos_admissions']+=1
            # Only control metadata is retained. Current grids/masks/rows disappear
            # before the borrowed lease and the enclosing build projection close.
            lease._check_sources(); _check_marks(self.marks)
        finally:
            # Tracebacks may outlive writer.close() and its charge credit.
            fold=spec=days=selected=local=admitted=records=marks=raw=_=None
            new_records=new_sources=new_lineages=delta=p=r=s=day=name=None

    def finish(self):
        from .stock_signal_evaluation_matrix import _matrix_input_root,_publish_matrix_inputs,_label_projection_ref
        try:
            self._check(); require(self.state.active==0,'finish requires released build projection')
            self._refresh_manifest()
            require(not self.state.incomplete,'complete checkpoint batch required before OOS publication')
            require(set(self.shards)==set(self.scope['sessions']),'complete build OOS date scope required')
            self.state.store.reserve(self.charge*4+len(self.scope['universe'])*16384+65536)
            self.raw['sources']={r:self.raw['sources'][r] for r in sorted(self.raw['sources'])}
            self.raw['label_inputs']=[self.lineages[r] for r in sorted(self.lineages)]
            self.raw['label_ref']=_label_projection_ref(self.raw)
            admitted={'metadata':self.metadata,'refs':self.refs,'closures':self.closures,'raw':self.raw}
            root=_matrix_input_root(self.signal_inputs,None,self.scope,admitted,self.records,self.manifest)
            root['shards']=self.shards
            result=_publish_matrix_inputs(root,self.stage,self.destination,self.batch,self.marks)
            self.finished=True
            return result
        finally:
            admitted=root=None

    def close(self):
        if self.closed:return
        require(self.pid==os.getpid(),'build OOS writer belongs to another process')
        self.closed=True
        for name in ('manifest','scope','signal_inputs','metadata','refs','closures','records','marks',
                     'raw','lineages','shards'): setattr(self,name,None)
        self.state._fixed_shared_bytes-=self.charge; self.charge=0
        if self.borrowed_owner:
            self.state._build_oos_borrowers-=1; self.borrowed_owner=False
        self.state._sync_shared(); self.temporary.cleanup()


@contextmanager
def _freeze_build_oos_inputs(batch,*,scope,destination,signal_name):
    require(batch not in _WRITERS,'one build OOS writer per owner')
    writer=_BuildOOSWriter(batch,scope,destination,signal_name); _WRITERS[batch]=writer
    try: yield writer
    finally:
        del _WRITERS[batch]
        writer.close()
