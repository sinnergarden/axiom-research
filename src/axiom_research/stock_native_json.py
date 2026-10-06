"""One physical coverage carrier, preserving the existing native JSON hashes.

Only the frozen Raw/Label/Feature coverage positions are supported. Readers
admit canonical blob syntax separately before using the identity-bound bytes
here. No decoded coverage, global digest override or business protocol is added.
"""
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


VERSION = 'stock_native_json_carrier_v1'
BUFFER_FIELDS = {'path', 'file_digest', 'dtype', 'shape', 'buffer_digest'}
REFS = {'stock_label_build_v1': 'label_ref', 'stock_matrix_label_metadata_v1': 'metadata_ref',
        'stock_matrix_feature_metadata_v1': 'metadata_ref'}
_SHA = re.compile(r'^sha256:[0-9a-f]{64}$')
_SCALAR_COUNTERS = ('peak_workspace_bytes', 'peak_combined_workspace_bytes',
    'native_hash_calls', 'native_hash_bytes', 'carrier_hash_calls', 'carrier_hash_bytes',
    'coverage_replay_calls', 'coverage_replay_bytes', 'coverage_encode_calls',
    'coverage_encode_bytes', 'created_coverage_source_bytes')
_REF_COUNTERS = ('native_hash_calls_by_ref', 'carrier_hash_calls_by_ref')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _graph_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_graph_bytes(k, seen)+_graph_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_graph_bytes(v, seen) for v in value)
    return size


def _measure(value, budget):
    """Bound the traversal's seen set before measuring a caller-owned graph."""
    seen, size, temporary = set(), 0, 0
    def visit(current):
        nonlocal size, temporary
        if id(current) in seen:
            return
        # The traversal's ID set is temporary workspace too. Never allocate an
        # unbounded accounting set before rejecting an oversized skeleton.
        budget.check(size+sys.getsizeof(current)+96)
        budget.reserve(96)
        temporary += 96
        seen.add(id(current))
        size += sys.getsizeof(current)
        with budget.hold(256):
            if isinstance(current, dict):
                for key, child in current.items():
                    visit(key); visit(child)
            elif isinstance(current, (list, tuple, set, frozenset)):
                for child in current:
                    visit(child)
    try:
        visit(value)
        return size
    finally:
        budget.owned -= temporary


class _Budget:
    def __init__(self, maximum, caller, metrics):
        _require(type(maximum) is int and maximum > 0, 'positive integer workspace budget required')
        _require(caller is None or callable(caller), 'caller_retained_bytes must be callable')
        self.maximum, self.caller, self.metrics, self.owned = maximum, caller, metrics, 0
        # One scan per budget, rather than one scan per JSON span. The fixed
        # allowance includes this small capacity table, pending scalar keys,
        # old/new scalar integers, and the top metrics dict's resize workspace.
        self.counter_bytes = 2048+3*sys.getsizeof(metrics)+len(metrics)*128
        for key, value in metrics.items():
            self.counter_bytes += sys.getsizeof(key)+sys.getsizeof(value)
            if isinstance(value, dict):
                # Keep room for old/new dict tables at a resize, including an
                # already populated ref map near its next growth threshold.
                self.counter_bytes += 2*sys.getsizeof(value)+len(value)*128+256
                self.counter_bytes += sum(sys.getsizeof(k)+sys.getsizeof(v)
                                          for k, v in value.items())
        self.scalar_capacity = {}
        for name in _SCALAR_COUNTERS:
            value = metrics.get(name, 0)
            _require(type(value) is int and value >= 0, 'native metrics counters must be nonnegative integers')
            capacity = max(64, sys.getsizeof(value)+32)
            self.scalar_capacity[name] = capacity
            self.counter_bytes += capacity-(sys.getsizeof(value) if name in metrics else 0)
            if name not in metrics:
                self.counter_bytes += sys.getsizeof(name)+128
        for name in _REF_COUNTERS:
            if name not in metrics:
                self.counter_bytes += sys.getsizeof(name)+512
        self.check()

    def retained(self):
        retained = 0 if self.caller is None else self.caller()
        _require(type(retained) is int and retained >= 0, 'caller_retained_bytes must return a nonnegative integer')
        return retained

    def check(self, extra=0):
        retained = self.retained()
        total = self.owned+extra+self.counter_bytes
        # The byte high-water counters normally fit the precharged scalar
        # capacity. Keep the guard sound for unusually large caller integers.
        for name, value in (('peak_workspace_bytes', total),
                            ('peak_combined_workspace_bytes', total+retained)):
            required = sys.getsizeof(value)+32
            if required > self.scalar_capacity[name]:
                self.counter_bytes += required-self.scalar_capacity[name]
                self.scalar_capacity[name] = required
        total = self.owned+extra+self.counter_bytes
        self.metrics['peak_workspace_bytes'] = max(self.metrics.get('peak_workspace_bytes', 0), total)
        self.metrics['peak_combined_workspace_bytes'] = max(self.metrics.get('peak_combined_workspace_bytes', 0), total+retained)
        _require(total+retained <= self.maximum, 'native JSON workspace budget exceeded')

    def room(self):
        # Reserving a quarter of the total limit is bounded by a real guard
        # below, including dynamic caller and counter storage.
        self.check()
        return self.maximum-self.owned-self.retained()-self.counter_bytes

    def _charge_counter(self, amount):
        self.check(amount)
        self.counter_bytes += amount

    def increment(self, name, amount=1):
        old = self.metrics.get(name, 0)
        _require(name in self.scalar_capacity and type(old) is int and old >= 0 and
                 type(amount) is int and amount >= 0, 'nonnegative integer native metric increment required')
        # Positive integer addition can add at most one base-2**30 digit to
        # the larger operand. Guard old/new values together before allocating.
        required = max(sys.getsizeof(old), sys.getsizeof(amount))+4
        if required > self.scalar_capacity[name]:
            self._charge_counter(required-self.scalar_capacity[name])
            self.scalar_capacity[name] = required
        self.check(required+sys.getsizeof(amount)+32)
        self.metrics[name] = old+amount

    def hash_ref(self, kind, ref):
        name = kind+'_hash_calls_by_ref'
        _require(name in _REF_COUNTERS, 'unknown native hash-ref counter')
        counts = self.metrics.get(name)
        if counts is None:
            counts = {}
        _require(isinstance(counts, dict), 'native hash-ref counters must be dictionaries')
        if ref not in counts:
            # Charge once before insertion. This deliberately exceeds a dict
            # entry's amortized size and covers transient table growth too.
            self._charge_counter(sys.getsizeof(ref)+64+512)
            counts[ref] = 1
        else:
            old = counts[ref]
            _require(type(old) is int and old >= 0, 'native hash-ref count must be a nonnegative integer')
            self.check(sys.getsizeof(old)+36)
            # The admission scan precharges growth slack; retain a further
            # digit conservatively for each existing-entry integer update.
            self._charge_counter(4)
            counts[ref] = old+1
        self.metrics[name] = counts

    def reserve(self, amount):
        self.check(amount)
        self.owned += amount

    @contextmanager
    def hold(self, amount):
        self.reserve(amount)
        try:
            yield
        finally:
            self.owned -= amount


def _metrics(metrics):
    _require(metrics is None or type(metrics) is dict, 'metrics must be a dictionary')
    return {} if metrics is None else metrics


def _buffer(desc):
    _require(type(desc) is dict and set(desc) == BUFFER_FIELDS and desc['dtype'] == 'uint8',
             'exact uint8 coverage BufferDesc required')
    _require(type(desc['shape']) is list and len(desc['shape']) == 1 and
             type(desc['shape'][0]) is int and desc['shape'][0] > 0, 'positive coverage byte length required')
    _require(type(desc['path']) is str and bool(desc['path']) and
             type(desc['file_digest']) is str and bool(_SHA.fullmatch(desc['file_digest'])) and
             desc['file_digest'] == desc['buffer_digest'], 'coverage byte identity mismatch')


def _stat(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _replay(desc, *, budget, metrics, path_resolver):
    _buffer(desc)
    path = Path(desc['path'] if path_resolver is None else path_resolver(desc['path']))
    _require(not path.is_symlink(), 'coverage path must not be a symlink')
    before = path.stat()
    _require(before.st_size == desc['shape'][0], 'coverage size mismatch')
    h, count = sha256(), 0
    budget.increment('coverage_replay_calls')
    with path.open('rb') as stream:
        _require(_stat(os.fstat(stream.fileno())) == _stat(before), 'coverage changed before replay')
        while True:
            # Leave room for the parser/encoder stack and caller registry.
            size = min(65536, max(1, (budget.room()-64)//4))
            with budget.hold(size+64):
                chunk = stream.read(size)
                if not chunk:
                    break
                count += len(chunk)
                h.update(chunk)
                budget.increment('coverage_replay_bytes', len(chunk))
                yield chunk
        _require(_stat(os.fstat(stream.fileno())) == _stat(before), 'coverage changed during replay')
    _require(_stat(path.stat()) == _stat(before), 'coverage changed after replay')
    _require(count == desc['shape'][0] and 'sha256:'+h.hexdigest() == desc['buffer_digest'],
             'coverage replay digest mismatch')


def _string(value, budget):
    yield b'"'
    for start in range(0, len(value), 1024):
        # Each fragment uses stdlib's original escaping. Concatenating their
        # interiors is exactly the canonical encoding of the original string.
        count = min(1024, len(value)-start)
        with budget.hold(128+count*32):
            encoded = json.dumps(value[start:start+count], ensure_ascii=False,
                                 allow_nan=False, separators=(',', ':'))[1:-1].encode('utf-8')
            yield encoded
    yield b'"'


def _registry(bindings, budget):
    out = {}
    for context, desc in bindings:
        _require(type(context) is dict and 'coverage' not in context, 'binding requires a small context without coverage')
        _buffer(desc)
        previous = out.get(id(context))
        _require(previous is None or previous[0] is context and previous[1] == desc, 'conflicting coverage object binding')
        if previous is None:
            budget.reserve(128+_graph_bytes(desc))
            out[id(context)] = context, desc
    return out


def _json_chunks(value, *, budget, registry, path_resolver, exclude=None, root=True):
    with budget.hold(256):
        if type(value) is dict:
            _require(all(type(key) is str for key in value), 'native JSON object keys must be strings')
            binding = registry.get(id(value))
            _require(binding is None or binding[0] is value, 'coverage object identity mismatch')
            count = len(value)-(1 if root and exclude in value else 0)+(1 if binding is not None else 0)
            with budget.hold(128+count*16):
                names = [key for key in value if not (root and key == exclude)]
                if binding is not None:
                    names.append('coverage')
                names.sort()
                yield b'{'
                for index, key in enumerate(names):
                    if index:
                        yield b','
                    yield from _string(key, budget)
                    yield b':'
                    if key == 'coverage' and binding is not None:
                        yield from _replay(binding[1], budget=budget, metrics=budget.metrics, path_resolver=path_resolver)
                    else:
                        yield from _json_chunks(value[key], budget=budget, registry=registry,
                                                path_resolver=path_resolver, root=False)
                yield b'}'
        elif type(value) in (list, tuple):
            yield b'['
            for index, child in enumerate(value):
                if index:
                    yield b','
                yield from _json_chunks(child, budget=budget, registry=registry, path_resolver=path_resolver, root=False)
            yield b']'
        elif type(value) is str:
            yield from _string(value, budget)
        else:
            _require(value is None or type(value) in (bool, int, float), 'unsupported native JSON scalar')
            _require(type(value) is not float or math.isfinite(value), 'nonfinite native JSON scalar')
            estimate = 128+(value.bit_length()+2 if type(value) is int else 64)
            with budget.hold(estimate):
                yield json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def _hash(value, *, budget, registry, exclude=None, path_resolver=None, kind='native'):
    h = sha256()
    budget.increment(kind+'_hash_calls')
    for chunk in _json_chunks(value, budget=budget, registry=registry, exclude=exclude, path_resolver=path_resolver):
        h.update(chunk)
        budget.increment(kind+'_hash_bytes', len(chunk))
    budget.check(256)
    result = 'sha256:'+h.hexdigest()
    budget.hash_ref(kind, result)
    return result


def native_digest(value, *, coverage_bindings=(), exclude_ref_key=None, maximum_workspace_bytes,
                  caller_retained_bytes=None, metrics=None, path_resolver=None):
    """Restore the original H from ordinary values or admitted coverage bytes."""
    if exclude_ref_key is not None:
        _require(type(value) is dict and REFS.get(value.get('contract_version')) == exclude_ref_key,
                 'unsupported native self-reference exclusion')
    budget = _Budget(maximum_workspace_bytes, caller_retained_bytes, _metrics(metrics))
    registry = _registry(coverage_bindings, budget)
    return _hash(value, budget=budget, registry=registry, exclude=exclude_ref_key, path_resolver=path_resolver)


def _holder(value, ref_key):
    _require(type(value) is dict and REFS.get(value.get('contract_version')) == ref_key,
             'unsupported coverage holder/ref_key')
    _require(type(value.get(ref_key)) is str and bool(_SHA.fullmatch(value[ref_key])), 'native self reference required')


def _contexts(value, budget):
    """Return only the three approved holders' exact business positions."""
    version = value['contract_version']
    if version == 'stock_label_build_v1':
        context = value.get('source_evidence', {}).get('context')
        _require(type(context) is dict, 'Raw source context required')
        yield ('source_evidence', 'context', 'coverage'), context
    elif version == 'stock_matrix_label_metadata_v1':
        _require(type(value.get('contents')) is dict, 'Label contents required')
        for ref, selected in value['contents'].items():
            if type(selected) is dict and set(selected) == {'context', 'records', 'field_meta'}:
                _require(type(ref) is str and bool(_SHA.fullmatch(ref)) and type(selected['context']) is dict and
                         type(selected['records']) is list and type(selected['field_meta']) is dict,
                         'selected Data wire identity required')
                yield ('contents', ref, 'context', 'coverage'), selected['context']
    else:
        proofs, contents = value.get('input_evidence'), value.get('contents')
        _require(type(proofs) is list and type(contents) is dict, 'Feature proof/contents required')
        for index, proof in enumerate(proofs):
            _require(type(proof) is dict and type(proof.get('source_evidence')) is dict and
                     type(proof.get('core_plan')) is dict, 'Feature original proof required')
            versions = proof['source_evidence']
            sources = proof['core_plan'].get('sources')
            _require(type(sources) is list and all(type(source) is dict and type(source.get('id')) is str for source in sources),
                     'Feature Core sources required')
            with budget.hold(256+len(sources)*192+len(contents)*16):
                ids = [source['id'] for source in sources]
                _require(len(ids) == len(set(ids)) and set(ids) == set(versions), 'Feature source IDs differ from original Core')
                matching = [ref for ref, child in contents.items() if type(child) is dict and child == versions]
                _require(bool(matching), 'Feature original SourceVersionMap missing')
                for source_id, source in versions.items():
                    _require(type(source_id) is str and bool(_SHA.fullmatch(source_id)) and type(source) is dict and
                             type(source.get('query_context')) is dict, 'Feature original source context required')
                    yield ('input_evidence', index, 'source_evidence', source_id, 'query_context', 'coverage'), source['query_context']
                    for ref in matching:
                        _require(type(ref) is str and bool(_SHA.fullmatch(ref)), 'SourceVersionMap reference required')
                        yield ('contents', ref, source_id, 'query_context', 'coverage'), contents[ref][source_id]['query_context']


def _path_bytes(path, budget=None):
    _require(type(path) is list and bool(path) and all(type(step) is str and bool(step) or
             type(step) is int and step >= 0 for step in path), 'invalid coverage path')
    if budget is None:
        return json.dumps(path, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    with budget.hold(256+sum(32*len(step) if type(step) is str else 128+step.bit_length() for step in path)):
        return json.dumps(path, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _parent(value, path):
    current = value
    for step in path[:-1]:
        if type(current) is dict:
            _require(type(step) is str and step in current, 'missing coverage path intermediate')
        else:
            _require(type(current) is list and type(step) is int and 0 <= step < len(current), 'coverage path out of bounds')
        current = current[step]
    _require(type(current) is dict and path[-1] == 'coverage' and 'coverage' not in current,
             'coverage skeleton leaf must be omitted')
    return current


def _shape(carrier, ref_key, budget):
    _require(type(carrier) is dict and set(carrier) == {'contract_version', 'native_ref', 'skeleton', 'coverage_slots', 'carrier_ref'}
             and carrier['contract_version'] == VERSION, 'exact native JSON carrier required')
    skeleton = carrier['skeleton']
    _holder(skeleton, ref_key)
    _require(carrier['native_ref'] == skeleton[ref_key], 'carrier/native self reference mismatch')
    _require(carrier['carrier_ref'] == _hash(carrier, budget=budget, registry={}, exclude='carrier_ref', kind='carrier'), 'carrier reference mismatch')
    allowed = set()
    for path, context in _contexts(skeleton, budget):
        if path not in allowed:
            budget.reserve(128+_graph_bytes(path))
            allowed.add(path)
    slots = carrier['coverage_slots']
    _require(type(slots) is list and bool(slots), 'nonempty coverage slots required')
    keys, bindings, paths = [], [], []
    budget.reserve(192+len(slots)*192)
    for slot in slots:
        _require(type(slot) is dict and set(slot) == {'path', 'bytes'}, 'exact coverage slot required')
        key = _path_bytes(slot['path'], budget)
        path = tuple(slot['path'])
        _require(path in allowed, 'coverage slot outside fixed allowlist')
        _require(path not in paths and all(path[:len(old)] != old and old[:len(path)] != path for old in paths),
                 'duplicate or overlapping coverage slots')
        context = _parent(skeleton, slot['path'])
        _buffer(slot['bytes'])
        budget.reserve(_graph_bytes(key)+sys.getsizeof(path))
        paths.append(path); keys.append(key); bindings.append((context, slot['bytes']))
    _require(all(left < right for left, right in zip(keys, keys[1:])), 'coverage slots must be canonically sorted')
    return bindings


def validate_carrier_shape(carrier, ref_key, *, maximum_workspace_bytes,
                           caller_retained_bytes=None, metrics=None):
    """Structure/physical carrier identity only; do not read an unadmitted blob."""
    budget = _Budget(maximum_workspace_bytes, caller_retained_bytes, _metrics(metrics))
    budget.reserve(_measure(carrier, budget))
    return _shape(carrier, ref_key, budget)


def _verify(carrier, ref_key, coverage_bindings, budget, path_resolver):
    expected = _shape(carrier, ref_key, budget)
    supplied = _registry(coverage_bindings, budget)
    _require(all(id(context) in supplied and supplied[id(context)][0] is context and supplied[id(context)][1] == desc
                 for context, desc in expected) and set(supplied) == {id(context) for context, desc in expected},
             'admitted coverage bindings differ from carrier')
    skeleton = carrier['skeleton']
    _require(_hash(skeleton, budget=budget, registry=supplied, exclude=ref_key, path_resolver=path_resolver)
             == carrier['native_ref'], 'restored native reference mismatch')
    budget.reserve(128+len(skeleton.get('contents', {}))*128)
    if skeleton['contract_version'] == 'stock_matrix_label_metadata_v1':
        children = [(ref, child) for ref, child in skeleton['contents'].items()
                    if type(child) is dict and set(child) == {'context', 'records', 'field_meta'}]
    elif skeleton['contract_version'] == 'stock_matrix_feature_metadata_v1':
        children = [(ref, child) for ref, child in skeleton['contents'].items()
                    if any(child == proof['source_evidence'] for proof in skeleton['input_evidence'])]
    else:
        children = []
    for ref, child in children:
        _require(_hash(child, budget=budget, registry=supplied, path_resolver=path_resolver) == ref,
                 'restored original child reference mismatch')


def verify_carrier_native(carrier, ref_key, *, coverage_bindings, maximum_workspace_bytes,
                          caller_retained_bytes=None, metrics=None, path_resolver=None):
    """After Store admission, verify native identity and original child refs."""
    budget = _Budget(maximum_workspace_bytes, caller_retained_bytes, _metrics(metrics))
    budget.reserve(_measure(carrier, budget))
    _verify(carrier, ref_key, coverage_bindings, budget, path_resolver)


def _clone(value, path, omitted, budget):
    if type(value) is dict:
        budget.reserve(128+len(value)*96)
        return {key: _clone(child, path+(key,), omitted, budget) for key, child in value.items() if path+(key,) not in omitted}
    if type(value) is list:
        budget.reserve(64+len(value)*16)
        return [_clone(child, path+(index,), omitted, budget) for index, child in enumerate(value)]
    budget.reserve(sys.getsizeof(value))
    return value


def _publish(temporary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, path)
        return True
    except FileExistsError:
        _require(not path.is_symlink(), 'existing native CAS path is a symlink')
        return False


def _file_hash(path, budget):
    h = sha256()
    path = Path(path)
    _require(not path.is_symlink(), 'native CAS path must not be a symlink')
    before = path.stat()
    with path.open('rb') as stream:
        _require(_stat(os.fstat(stream.fileno())) == _stat(before), 'native CAS changed before hash')
        while True:
            size = min(65536, max(1, (budget.room()-64)//4))
            with budget.hold(size+64):
                chunk = stream.read(size)
                if not chunk:
                    break
                h.update(chunk)
        _require(_stat(os.fstat(stream.fileno())) == _stat(before), 'native CAS changed during hash')
    _require(_stat(path.stat()) == _stat(before), 'native CAS changed after hash')
    return 'sha256:'+h.hexdigest()


def make_native_carrier(root, value, ref_key, *, maximum_source_bytes, maximum_parent_bytes,
                        maximum_workspace_bytes, caller_retained_bytes=None, metrics=None,
                        descriptor_mapper=None, path_resolver=None, coverage_bindings=()):
    """Write immutable canonical coverage CAS and the one physical carrier."""
    _holder(value, ref_key)
    for maximum in (maximum_source_bytes, maximum_parent_bytes):
        _require(type(maximum) is int and maximum > 0, 'positive source/parent budget required')
    metrics = _metrics(metrics)
    budget = _Budget(maximum_workspace_bytes, caller_retained_bytes, metrics)
    existing = _registry(coverage_bindings, budget)
    contexts = []
    for path, context in _contexts(value, budget):
        if 'coverage' in context or id(context) in existing and existing[id(context)][0] is context:
            budget.reserve(192+_graph_bytes(path))
            contexts.append((path, context))
    if not contexts:
        return None
    _require(_hash(value, budget=budget, registry=existing, exclude=ref_key, path_resolver=path_resolver)
             == value[ref_key], 'original native reference mismatch')
    root = Path(root)
    buffers = root/'buffers'; buffers.mkdir(parents=True, exist_ok=True)
    slots, cached, unique = [], {}, {}
    for path, context in contexts:
        if 'coverage' not in context:
            desc = existing[id(context)][1]
            unique[desc['buffer_digest']] = desc['shape'][0]
            _require(sum(unique.values()) <= maximum_source_bytes, 'unique coverage source byte budget exceeded')
            budget.reserve(128+_graph_bytes(list(path)))
            slots.append({'path': list(path), 'bytes': desc})
            continue
        coverage = context['coverage']
        cached_value = cached.get(id(coverage))
        if cached_value is not None and cached_value[0] is coverage:
            desc = cached_value[1]
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix='.coverage-', dir=buffers, delete=False) as stream:
                    temporary = Path(stream.name); h, size = sha256(), 0
                    budget.increment('coverage_encode_calls')
                    for chunk in _json_chunks(coverage, budget=budget, registry={}, path_resolver=None):
                        size += len(chunk)
                        _require(size <= maximum_source_bytes, 'coverage source byte budget exceeded')
                        stream.write(chunk); h.update(chunk)
                        budget.increment('coverage_encode_bytes', len(chunk))
                ref = 'sha256:'+h.hexdigest(); actual = buffers/(ref[7:]+'.bin')
                unique[ref] = size
                _require(sum(unique.values()) <= maximum_source_bytes, 'unique coverage source byte budget exceeded')
                created = _publish(temporary, actual)
                _require(actual.stat().st_size == size and _file_hash(actual, budget) == ref, 'published coverage bytes mismatch')
                if created: budget.increment('created_coverage_source_bytes', size)
                desc = {'path': str(actual.resolve()), 'file_digest': ref, 'dtype': 'uint8', 'shape': [size], 'buffer_digest': ref}
                if descriptor_mapper is not None: desc = descriptor_mapper(desc)
                _buffer(desc)
                cached[id(coverage)] = coverage, desc
                budget.reserve(128+_graph_bytes(desc))
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
        slots.append({'path': list(path), 'bytes': desc})
        budget.reserve(128+_graph_bytes(list(path)))
    budget.check(sum(256+32*sum(len(step) for step in slot['path'] if type(step) is str) for slot in slots))
    slots.sort(key=lambda slot: _path_bytes(slot['path']))
    skeleton = _clone(value, (), {tuple(slot['path']) for slot in slots}, budget)
    carrier = {'contract_version': VERSION, 'native_ref': value[ref_key], 'skeleton': skeleton, 'coverage_slots': slots}
    bindings = [(_parent(skeleton, slot['path']), slot['bytes']) for slot in slots]
    carrier['carrier_ref'] = _hash(carrier, budget=budget, registry={}, kind='carrier')
    _verify(carrier, ref_key, bindings, budget, path_resolver)
    directory = root/'metadata'; directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.carrier-', dir=directory, delete=False) as stream:
            temporary = Path(stream.name); size, h = 0, sha256()
            for chunk in _json_chunks(carrier, budget=budget, registry={}, path_resolver=None):
                size += len(chunk)
                _require(size+1 <= maximum_parent_bytes, 'carrier parent byte budget exceeded')
                stream.write(chunk); h.update(chunk)
            stream.write(b'\n'); h.update(b'\n')
        file_ref = 'sha256:'+h.hexdigest()
        final = directory/(carrier['carrier_ref'][7:]+'.json')
        _publish(temporary, final)
        _require(final.stat().st_size == size+1 and _file_hash(final, budget) == file_ref, 'published carrier bytes mismatch')
        return {'path': str(final.resolve()), 'file_digest': file_ref, ref_key: value[ref_key]}
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
