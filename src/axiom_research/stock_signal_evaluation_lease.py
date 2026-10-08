"""Narrow OOS lease over the ordinary compact saved-fold admission."""
from collections.abc import Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import os

from .stock_fold_inputs import require
from .stock_compact_store import OwnedStore, _size, reference


class _TargetRows(Sequence):
    """Readonly virtual rows; each returned dictionary is a detached snapshot."""
    __slots__=('_lease','_rows')

    def __init__(self,lease,rows): self._lease=lease; self._rows=rows
    def __len__(self): self._lease._check_open(); return len(self._rows)
    def __getitem__(self,index):
        self._lease._check_open()
        if isinstance(index,slice): return tuple(self[i] for i in range(*index.indices(len(self))))
        return self._rows[index]
    def _detach(self): self._rows=None


class _FoldLease:
    __slots__=('_state','_batch','_ingress','_projection','_value','_rows','_charge',
               '_shared_charge','_source_charge','_pid','_closed','_owns_projection')

    def __init__(self,state,batch,ingress,*,owns_projection=True):
        self._state=state; self._batch=batch; self._ingress=ingress
        self._projection=None; self._value=None; self._rows=[]; self._charge=0
        self._shared_charge=self._source_charge=0; self._pid=os.getpid(); self._closed=False
        self._owns_projection=owns_projection

    def _check_open(self):
        require(self._pid==os.getpid(),'evaluation lease belongs to another process')
        require(not self._closed and self._projection is not None,'evaluation lease is closed')

    def _prepare(self,ingress):
        # Output JSON is owned by ingress. Charge it alongside the Feature and
        # Label window before the original OOS projection allocates metadata.
        self._shared_charge=ingress.resident_bytes
        self._source_charge=ingress.metrics['source_bytes']
        state=self._state
        state._fixed_shared_bytes+=self._shared_charge
        state._fixed_shared_source_bytes+=self._source_charge
        state._sync_shared(); state.store.reserve(0)

    def _admit(self,manifest,documents,projection,inputs,spec):
        targets=records=marks=selected=metadata=value=rows=view=target=None
        store=self._state.store
        try:
            require(inputs['contract_version']=='stock_ml_saved_inputs_v4','compact v4 evaluation lease required')
            targets,records,marks=self._state.evaluation_sources(inputs,spec)
            for path,ref in self._ingress.hashes.items():
                require(path not in records or records[path]==ref,'conflicting saved evaluation source pin')
                require(path not in marks or marks[path]==self._ingress.marks[path],
                        'conflicting saved evaluation source fingerprint')
                records[path]=ref; marks[path]=self._ingress.marks[path]
            selected={'manifest.json':manifest,**{name:documents[name] for name in
                ('fold.json','model.json','feature-slice.json','predictions.json')}}
            metadata={'documents':selected,'common':{key:projection.common[key] for key in
                ('snapshot','pit_policy','calendar','universe')},
                'evaluation_targets':[{'descriptor':d,'header':h} for d,h,_ in targets],
                'source_records':tuple(sorted(records.items())),'source_fingerprints':marks}
            estimate=_size(metadata,maximum=store.maximum_matrix_bytes,
                           retained=store.shared_bytes+store.resident_bytes+store.lease_bytes)
            # Both source metadata and its detached public copy coexist briefly.
            store.reserve(estimate*2+4096*(len(targets)+1))
            value=deepcopy(metadata)
            charge=_size(value)+4096*(len(targets)+1)
            store.reserve(charge); store.lease_bytes+=charge; self._charge=charge
            for target,(_,_,rows) in zip(value['evaluation_targets'],targets):
                view=_TargetRows(self,rows); self._rows.append(view); target['rows']=view
            self._value=value; self._projection=projection
        except BaseException:
            self._value=None
            for rows in self._rows: rows._detach()
            self._rows.clear(); store.lease_bytes-=self._charge; self._charge=0
            raise
        finally:
            targets=records=marks=metadata=selected=value=rows=view=target=None

    @property
    def documents(self): self._check_open(); return self._value['documents']
    @property
    def common(self): self._check_open(); return self._value['common']
    @property
    def evaluation_targets(self): self._check_open(); return self._value['evaluation_targets']
    @property
    def source_records(self): self._check_open(); return self._value['source_records']
    @property
    def source_fingerprints(self): self._check_open(); return self._value['source_fingerprints']

    def _check_sources(self):
        self._check_open()
        require(self._projection._store is self._state.store, 'evaluation projection store mismatch')
        self._ingress.check()
        self._batch._check_sources()

    def close(self):
        if self._closed: return
        require(self._pid==os.getpid(),'evaluation lease belongs to another process')
        self._closed=True
        for rows in self._rows: rows._detach()
        self._rows.clear(); self._value=None
        projection=self._projection; self._projection=None
        state=self._state
        state.store.lease_bytes-=self._charge; self._charge=0
        try:
            if projection is not None and self._owns_projection: projection.close()
        finally:
            self._ingress.close()
            state._fixed_shared_bytes-=self._shared_charge
            state._fixed_shared_source_bytes-=self._source_charge
            self._shared_charge=self._source_charge=0
            state._sync_shared()
            self._state=self._batch=self._ingress=None

    def _repath_outputs(self,original,destination):
        """Transfer already-hashed file pins after the owned directory rename."""
        from .stock_fold_inputs import file_fingerprint
        self._check_open(); original=Path(original); destination=Path(destination)
        ingress=self._ingress
        pairs=[]
        for path,mark in ingress.marks.items():
            require(Path(path).parent==original,'build output pin outside publication directory')
            target=str(destination/Path(path).name)
            require(file_fingerprint(target)==mark,'build output changed during publication')
            pairs.append((path,target))
        records=dict(self._value['source_records']); marks=self._value['source_fingerprints']
        for path,target in pairs:
            records[target]=records.pop(path); marks[target]=marks.pop(path)
        self._value['source_records']=tuple(sorted(records.items()))
        for name in ('marks','hashes','json','charges','arrays'):
            values=getattr(ingress,name)
            for path,target in pairs:
                if path in values: values[target]=values.pop(path)
        self._check_sources()


@contextmanager
def _admit_build_fold(path,*,projection,batch):
    """Internal borrowed lease from this build's actual saved-byte validator."""
    from .stock_batch import _data
    from .stock_matrix_folds import _load_matrix_fold
    from .stock_fold_artifacts import _owned_fold
    state=_data(batch)['matrix_state']
    require(state.compact and (projection is None or
            projection._store is state.store and not projection._closed),
            'active build projection required')
    ingress=OwnedStore(state.store.limits,
        shared_bytes=state.store.shared_bytes+state.store.resident_bytes+state.store.lease_bytes,
        shared_source_bytes=state.store.shared_source_bytes+state.store.metrics['source_bytes'])
    lease=_FoldLease(state,batch,ingress,owns_projection=projection is None)
    failed=True
    try:
        _load_matrix_fold(path,projection=projection,batch=batch,ingress=ingress,_lease=lease)
        documents={Path(p).name:v for p,v in ingress.json.items() if Path(p).name!='manifest.json'}
        run=_owned_fold(path,documents)
        lease._check_sources()
        yield lease,run
        lease._check_sources()
        failed=False
    finally:
        try: lease.close()
        finally:
            if failed and projection is None: state._release_window()
        state=ingress=lease=documents=run=None


@contextmanager
def admit_fold(prediction_input, *, batch):
    from .stock_batch import _data
    from .stock_matrix_folds import _load_matrix_fold
    require(type(prediction_input) is dict and set(prediction_input)=={'path','file_digest','signal_run_ref'} and
        type(prediction_input['path']) is str and Path(prediction_input['path']).is_absolute() and
        Path(prediction_input['path']).name=='predictions.json' and
        reference(prediction_input['file_digest']) and reference(prediction_input['signal_run_ref']),
        'fixed saved prediction descriptor required')
    value=_data(batch); state=value.get('matrix_state')
    require(state is not None and state.compact,'owner-loaded compact v4 batch required')
    state.check()
    ingress=OwnedStore(state.store.limits,
        shared_bytes=state.store.shared_bytes+state.store.resident_bytes+state.store.lease_bytes,
        shared_source_bytes=state.store.shared_source_bytes+state.store.metrics['source_bytes'])
    lease=_FoldLease(state,batch,ingress)
    failed=True
    try:
        _load_matrix_fold(Path(prediction_input['path']).parent,
            projection=None,batch=batch,ingress=ingress,_lease=lease)
        require(dict(lease.source_records).get(prediction_input['path'])==prediction_input['file_digest'] and
            lease.documents['predictions.json']['signal_run_ref']==prediction_input['signal_run_ref'],
            'saved prediction descriptor binding mismatch')
        lease._check_sources()
        yield lease
        lease._check_sources()
        failed=False
    finally:
        try: lease.close()
        finally:
            if failed: state._release_window()
        # A context-manager traceback must not retain the released owner graph.
        value=state=ingress=lease=None
