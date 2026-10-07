"""Bounded process-local reuse of fully verified saved fold inputs.

The manifest lists prepared per-fit Raw/normalized Label parents. Normalization
continues to use normalize_forward_labels and Feature Core when preparing those
parents. This loader neither reads Data nor creates a second numerical executor.
"""
from copy import deepcopy
from collections import Counter
from pathlib import Path
from weakref import WeakKeyDictionary
import os
import time

from .stock_artifacts import digest
from .stock_fold_inputs import (common_source_identity, load_saved_feature_inputs,
                                _admit_saved_fold_labels, _project_admitted_saved_fold,
                                read_parent, require, validate_spec)

DEFAULT_LIMITS = {'maximum_source_bytes': 8 * 1024**3,
                  'maximum_matrix_bytes': 512 * 1024**2}
_TOKEN = object()
_DATA = WeakKeyDictionary()


def _fingerprint(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _descriptors(inputs):
    yield inputs['scope']
    for parent in inputs['feature_parents']:
        yield parent['features']
        yield parent['input_evidence']
    for parent in inputs['training_labels']:
        yield parent['raw']
        yield parent['normalized']
    yield inputs['evaluation_labels']


def _data(batch):
    require(type(batch) is StockMLBatchInputs and batch in _DATA, 'validated saved batch required')
    value = _DATA[batch]
    require(value['owner_pid']==os.getpid(), 'saved batch belongs to another process')
    require(not value['closed'], 'saved batch is closed')
    return value


class StockMLBatchInputs:
    """An immutable source definition with a releasable shared Feature matrix.

    Instances are created only by load_stock_ml_batch_inputs. Public metadata is
    copied. No mutable parent, matrix or validation flag is exposed to callers.
    Source fingerprints are checked on every fold use. Public saved-fold
    loading admits disk parents by default, or reuses this process's full
    admission while retaining every saved output check.
    """
    __slots__ = ('__weakref__',)

    def __init__(self, token, value):
        require(token is _TOKEN, 'saved batch requires initialization')
        value['owner_pid']=os.getpid()
        _DATA[self] = value

    @property
    def identity(self):
        return _data(self)['identity']

    def to_dict(self):
        return deepcopy(_data(self)['manifest'])

    @property
    def metrics(self):
        value=_data(self); state=value.get('matrix_state')
        if state is None or getattr(state,'residency','eager')=='eager':
            return deepcopy(value['metrics'])
        result=deepcopy(state.store.metrics); feature=state.feature.metrics
        for key in ('file_hash_calls','hash_bytes','source_bytes','json_decode_calls','released_buffer_bytes'):
            result[key]=result.get(key,0)+feature.get(key,0)
        result.update(residency=state.residency,feature_residency=state.store.metrics['feature_residency'],
            resident_bytes=state.store.shared_bytes+state.store.resident_bytes,
            lease_bytes=state.store.lease_bytes)
        return result

    def _check_sources(self):
        value = _data(self)
        if 'matrix_state' in value:
            value['matrix_state'].store.check()
            return
        require(all(_fingerprint(path) == mark for path, mark in value['fingerprints'].items()),
                'saved batch source changed; initialize a fresh batch')

    def _project(self, inputs, spec):
        value = _data(self)
        require('matrix_state' not in value, 'matrix inputs require a bounded fold projection')
        self._check_sources()
        require((digest(inputs), digest(spec)) in value['fold_keys'], 'fold outside saved batch definition')
        key = digest(inputs), digest(spec)
        result = _project_admitted_saved_fold(inputs, spec, feature_inputs=value['features'],
                                             labels=value['labels'][key])
        self._check_sources()
        value['metrics']['fold_projection_calls'] += 1
        return result

    def _matrix_project(self, inputs, spec):
        value = _data(self)
        require('matrix_state' in value, 'saved matrix batch required')
        self._check_sources()
        require((digest(inputs),digest(spec)) in value['fold_keys'], 'fold outside saved batch definition')
        return value['matrix_state'].project(inputs,spec)

    def _matrix_evaluation(self, inputs, spec):
        """Project only admitted OOS evidence; no training matrices or hashes."""
        value = _data(self)
        require('matrix_state' in value, 'saved matrix batch required')
        self._check_sources()
        require((digest(inputs),digest(spec)) in value['fold_keys'], 'fold outside saved batch definition')
        return value['matrix_state'].evaluation(inputs,spec,batch_ref=value['identity'])

    def _project_evaluation(self, inputs, spec):
        """Borrow the admitted OOS evidence lease for a saved matrix fold."""
        value = _data(self)
        require('matrix_state' in value, 'saved matrix batch required')
        self._check_sources()
        require((digest(inputs),digest(spec)) in value['fold_keys'], 'fold outside saved batch definition')
        return value['matrix_state'].project_evaluation(inputs,spec,batch_ref=value['identity'])

    def _evaluation_source_records(self):
        """The one immutable (absolute path, file digest) table for this batch."""
        value = _data(self)
        require('matrix_state' in value, 'saved matrix batch required')
        self._check_sources()
        if getattr(value['matrix_state'],'residency','eager')=='sequential':
            value['matrix_state'].verify_all()
        return value['matrix_state'].source_records

    def _admit_raw_label(self, descriptor):
        """Admit an explicit Raw override through the owner's storage checks."""
        from .stock_matrix_reader import _admit_raw_label
        value=_data(self); self._check_sources()
        state=value.get('matrix_state')
        return _admit_raw_label(descriptor,store=None if state is None else state.store)

    def _matrices(self, training, candidates):
        import numpy as np
        value = _data(self)
        require('matrix_state' not in value, 'matrix inputs require a bounded fold projection')
        def select(rows):
            indexes = [value['key_index'][r['security_id'], r['session']] for r in rows]
            matrix = value['matrix'][indexes]
            require(matrix.shape == (len(rows), len(value['columns'])) and np.isfinite(matrix).all(),
                    'joint training/prediction matrix contains invalid features')
            # The same joined rows select X and y. No independent positional
            # Label file is used to construct the target vector.
            return matrix
        X, P = select(training), select(candidates)
        y = np.asarray([r['label'] for r in training], dtype=np.float64)
        require(np.isfinite(y).all(), 'nonfinite joint training target')
        return X, y, P

    def close(self):
        value = _data(self)
        if 'matrix_state' in value:
            value['matrix_state'].close()
        value.clear()
        value['closed'] = True

    def __enter__(self):
        _data(self)
        return self

    def __exit__(self, *args):
        self.close()


def _compact_batch_handle(state,manifest,begin):
    """One handle construction for public and standalone compact ingress."""
    metrics=deepcopy(state.store.metrics)
    for key in ('file_hash_calls','hash_bytes','source_bytes','json_decode_calls'):
        metrics[key]+=state.feature.metrics.get(key,0)
    metrics.update(initialization_seconds=time.perf_counter()-begin,legacy_ancestor_reads=0,
        legacy_native_hash_calls=0,common_key_index_builds=1)
    value={'identity':manifest['batch_ref'],'manifest':manifest,'matrix_state':state,
        'fold_keys':{(digest(f['input_manifest']),digest(f['fold_spec'])) for f in manifest['folds']},
        'closed':False,'metrics':metrics}
    return StockMLBatchInputs(_TOKEN,value)


def load_stock_ml_batch_inputs(batch_manifest, *, feature_inputs=None, limits=None, residency='eager'):
    """Verify each unique saved parent once and allocate one readonly matrix.

    ``batch_manifest`` is {contract_version: stock_ml_batch_inputs_v1, folds:
    [{input_manifest: stock_ml_saved_inputs_v1, fold_spec: ...}, ...]}. Folds
    must have the same complete common-source definition, be chronological and
    have disjoint OOS trade sessions. Per-fit Label descriptors remain distinct.
    Limits are positive integer byte budgets, not an arbitrary fold-count cap.
    Compact v3 accepts ``residency='sequential'`` for controls-only initial
    admission followed by one complete fold window. Saved identities are
    independent of this owner-local memory choice. No fit/predict/account is
    performed; sequential metrics include subsequent window admissions.
    """
    begin = time.perf_counter()
    require(residency in ('eager','sequential'),'unknown batch residency mode')
    manifest = deepcopy(batch_manifest)
    if type(manifest) is dict and manifest.get('contract_version')=='stock_ml_batch_inputs_v3':
        from .stock_compact_batch import load_compact_state
        from .stock_compact_store import _view_data, sealed, limits as compact_limits
        sealed(manifest,'content_digest')
        state=None
        if feature_inputs is not None:
            fd=_view_data(feature_inputs)
            state=fd['prepared'].get(manifest['batch_ref'])
            if state is not None:
                require(state.batch==manifest and state.feature is feature_inputs,'prepared compact handle mismatch')
                state.check(); state.compatible(compact_limits(limits))
                fd['prepared'].pop(manifest['batch_ref'])
                if state.residency!=residency:
                    state.close(); state=None
        if state is None: state=load_compact_state(manifest,feature_inputs=feature_inputs,limits=limits,residency=residency)
        return _compact_batch_handle(state,manifest,begin)
    require(residency=='eager','sequential residency requires compact v3 inputs')
    require(feature_inputs is None,'Feature handle reuse requires compact v3 inputs')
    if type(manifest) is dict and manifest.get('contract_version') == 'stock_ml_batch_inputs_v2':
        from .stock_matrix_reader import load_matrix_batch_state
        state = load_matrix_batch_state(manifest, limits=limits)
        state.store.metrics['initialization_seconds'] = time.perf_counter()-begin
        value = {'identity':manifest['batch_ref'],'manifest':manifest,'matrix_state':state,
            'fold_keys':{(digest(f['input_manifest']),digest(f['fold_spec'])) for f in manifest['folds']},
            'closed':False,'metrics':state.store.metrics}
        try:
            # Count the public handle's retained manifest/key graph in the
            # same admission budget as the backing, without double counting
            # aliases shared with the verified state.
            state._account_resident(extra_roots=(value,))
        except BaseException:
            state.close()
            raise
        return StockMLBatchInputs(_TOKEN,value)
    require(type(manifest) is dict and set(manifest) == {'contract_version', 'folds'} and
            manifest['contract_version'] == 'stock_ml_batch_inputs_v1' and
            type(manifest['folds']) is list and bool(manifest['folds']), 'unsupported saved batch manifest')
    budgets = deepcopy(DEFAULT_LIMITS if limits is None else limits)
    require(type(budgets) is dict and set(budgets) == set(DEFAULT_LIMITS) and
            all(type(v) is int and v > 0 for v in budgets.values()), 'positive integer batch byte limits required')
    common = None; previous_end = None; fold_keys = set(); descriptors = {}
    for fold in manifest['folds']:
        require(type(fold) is dict and set(fold) == {'input_manifest', 'fold_spec'}, 'explicit saved batch fold required')
        inputs, spec = fold['input_manifest'], fold['fold_spec']
        identity = common_source_identity(inputs)
        common = identity if common is None else common
        require(identity == common, 'batch common source definitions differ')
        validate_spec(spec, inputs['calendar'])
        trades = spec['oos_trade_sessions']
        require(previous_end is None or trades[0] > previous_end, 'batch OOS folds overlap or are unordered')
        previous_end = trades[-1]
        fold_key = digest(inputs), digest(spec)
        require(fold_key not in fold_keys, 'duplicate saved batch fold')
        fold_keys.add(fold_key)
        for desc in _descriptors(inputs):
            require(type(desc) is dict and type(desc.get('path')) is str and Path(desc['path']).is_absolute(),
                    'absolute saved batch parent required')
            path = desc['path']
            require(path not in descriptors or descriptors[path] == desc, 'conflicting saved parent descriptor')
            descriptors[path] = desc
    fingerprints = {p: _fingerprint(p) for p in descriptors}
    source_bytes = sum(mark[2] for mark in fingerprints.values())
    require(source_bytes <= budgets['maximum_source_bytes'], 'saved batch source byte budget exceeded')
    first = manifest['folds'][0]['input_manifest']
    rows = sum(len(p['sessions']) * len(first['universe']) for p in first['feature_parents'])
    matrix_bytes = rows * len(first['ordered_features']) * 8
    require(matrix_bytes <= budgets['maximum_matrix_bytes'], 'saved batch matrix byte budget exceeded')
    features = load_saved_feature_inputs(first)
    cache = {}; read_counts = {}; remaining = Counter()
    for fold in manifest['folds']:
        inputs = fold['input_manifest']
        # Raw admission is shared within one fit; normalized parents and the
        # evaluation parent are each consumed exactly once by that admission.
        remaining.update((p, 'label_ref') for p in {d['raw']['path'] for d in inputs['training_labels']})
        remaining.update((d['normalized']['path'], 'label_ref') for d in inputs['training_labels'])
        remaining[inputs['evaluation_labels']['path'], 'label_ref'] += 1
    def reader(desc, ref_key):
        key = desc['path'], ref_key
        require(descriptors.get(desc['path']) == desc, 'parent outside saved batch definition')
        require(remaining[key] > 0, 'unexpected saved parent admission read')
        if key not in cache:
            cache[key] = read_parent(desc, ref_key)
            read_counts[desc['path']] = read_counts.get(desc['path'], 0) + 1
        value = cache[key]
        remaining[key] -= 1
        if remaining[key] == 0:
            del cache[key]
        return value
    # Each unique parent is still parsed/content-hashed once. Verify every
    # fit's Core/raw/Feature/clock closure now, then release its full objects.
    # Keep invalid rows and the original metadata/order, not four training
    # projections or a second normalization executor.
    labels = {}
    for fold in manifest['folds']:
        inputs, spec = fold['input_manifest'], fold['fold_spec']
        labels[digest(inputs), digest(spec)] = _admit_saved_fold_labels(
            inputs, spec, feature_inputs=features, reader=reader)
    require(not cache and not any(remaining.values()), 'saved parent admission cache not released')
    import numpy as np
    keys = sorted(features._rows, key=lambda k: (k[1], k[0]))
    matrix = np.asarray([features._rows[k]['values'] for k in keys], dtype=np.float64)
    require(matrix.nbytes == matrix_bytes, 'saved Feature matrix size mismatch')
    matrix.flags.writeable = False
    require(all(_fingerprint(p) == mark for p, mark in fingerprints.items()), 'batch source changed during initialization')
    value = {'identity': digest(manifest), 'manifest': manifest, 'features': features,
        'matrix': matrix, 'key_index': {key: i for i, key in enumerate(keys)}, 'columns': first['ordered_features'],
        'labels': labels, 'fingerprints': fingerprints, 'fold_keys': fold_keys, 'closed': False,
        'metrics': {'source_bytes': source_bytes, 'shared_matrix_bytes': matrix.nbytes,
            'shared_matrix_builds': 1, 'common_source_validations': 1,
            'feature_parent_reads': len(first['feature_parents']), 'proof_reads': len(first['feature_parents']),
            'label_parent_reads': read_counts, 'unique_label_parent_reads': sum(read_counts.values()),
            'fold_projection_calls': 0, 'initialization_seconds': time.perf_counter()-begin,
            'data_read_calls': 0, 'supplier_calls': 0, 'core_calls': 0, 'train_calls': 0,
            'predict_calls': 0, 'account_calls': 0}}
    return StockMLBatchInputs(_TOKEN, value)
