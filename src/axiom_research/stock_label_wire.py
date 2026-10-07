"""Label-only projection during the existing canonical byte admission.

No whole selected-wire DOM is decoded. The canonical scanner owns syntax;
this observer hashes the three original components, retains small context,
and optionally decodes one OOS endpoint cell at a time under the same budget.
"""
from hashlib import sha256
import json
import sys

from .stock_canonical_json import _Validator
from .stock_parent_json import preflight_parent_json
from .stock_native_json import _graph_bytes


class _LabelWireValidator(_Validator):
    def __init__(self, maximum, retained, endpoint_keys):
        self.paths = []
        self.active = {}
        self.component_refs = {}
        self.capture = None
        self.capture_path = None
        self.context = {}
        self.headers = {}
        self.records = []
        self.metadata = {}
        self.endpoint_keys = endpoint_keys
        self.grids = {}
        self.key_parts = {}
        self.symbols = {}
        self.sessions = {}
        self.expected = None
        self.coverage_ref = None
        self.owned = 4096 + _graph_bytes(endpoint_keys)
        super().__init__(maximum, retained)

    def workspace(self):
        return (super().workspace() + self.owned + len(self.paths)*512
                + (sys.getsizeof(self.capture) if self.capture is not None else 0))

    def _value_start(self):
        if not self.paths:
            path = ()
        else:
            parent, ordinal = self.paths[-1]
            step = self.stack[-1][2] if self.stack[-1][0] == 'object' else ordinal
            path = parent + (step,)
        self.reserve(512)
        self.paths.append([path, 0])
        if len(path) == 1:
            if path[0] not in ('context', 'records', 'field_meta'):
                raise ValueError('exact selected Label wire components required')
            self.active[path] = sha256()
        if path == ('context', 'coverage'):
            self.active[path] = sha256()
        is_record = len(path) == 2 and path[0] == 'records'
        is_meta = len(path) == 4 and path[0] == 'field_meta' and path[2] == 'by_key'
        if is_record or is_meta:
            self.key_parts[path] = {}
        capture = (
            len(path) == 2 and path[0] == 'context' and path[1] != 'coverage'
            or len(path) == 3 and path[0] == 'field_meta' and path[2] != 'by_key'
            or self.endpoint_keys is not None and (is_record or is_meta))
        if capture and self.capture is None:
            self.reserve(128)
            self.capture, self.capture_path = bytearray(), path

    def begin_value(self, byte):
        super().begin_value(byte)
        path = self.paths[-1][0]
        if len(path) == 1 and byte != (91 if path[0] == 'records' else 123):
            raise ValueError('exact selected Label wire container types required')
        if ((len(path)==2 and path[0] in ('records','field_meta')
             or len(path)==4 and path[0]=='field_meta' and path[2]=='by_key') and byte != 123):
            raise ValueError('selected Label cells/field headers must be objects')
        if len(path)==3 and path[0]=='field_meta' and path[2]=='by_key' and byte != 91:
            raise ValueError('selected Label by_key must be an array')

    def _scalar_value(self, value):
        path = self.paths[-1][0]
        if self.endpoint_keys is None and (
            len(path) == 3 and path[0] == 'records' and path[2] in ('security_id','session')
            or len(path) == 5 and path[0] == 'field_meta' and path[2] == 'by_key'
            and path[4] in ('security_id','session')):
            self.reserve(sys.getsizeof(value)+128)
            self.owned += sys.getsizeof(value)+128
            self.key_parts[path[:-1]][path[-1]] = value

    def _span(self, chunk, start, end):
        if start == end:
            return
        view = memoryview(chunk)[start:end]
        for hasher in self.active.values():
            hasher.update(view)
        if self.capture is not None:
            self.reserve(2*(len(self.capture)+len(view))+len(view)+256)
            self.capture.extend(view)

    def _decode_capture(self):
        # Reserve the stdlib text/DOM and its post-check controls before decode.
        payload = bytes(self.capture)
        base = self.workspace() + sys.getsizeof(payload)
        external = self.retained
        def retained():
            return base + (0 if external is None else external())
        admission=preflight_parent_json((payload,), physical_size=len(payload),
            maximum_workspace_bytes=self.maximum, caller_retained_bytes=retained)
        amount=admission['decoded_graph_upper_bytes']+1024
        self.reserve(amount); self.owned+=amount
        return json.loads(payload),amount

    def _retain(self, value):
        amount = _graph_bytes(value) + 256
        self.reserve(amount)
        self.owned += amount

    def _grid(self, field, item):
        if type(item) is not dict:
            raise ValueError('selected Label cell must be an object')
        key = item.get('security_id'), item.get('session')
        if self.expected is not None:
            if key[0] not in self.symbols or key[1] not in self.sessions:
                raise ValueError('selected Label cell outside actual query grid')
            ordinal = self.sessions[key[1]]*len(self.symbols)+self.symbols[key[0]]
            if field not in self.grids:
                self.reserve(self.expected+1024)
                self.grids[field] = bytearray(self.expected)
                self.owned += self.expected+1024
            if self.grids[field][ordinal]:
                raise ValueError('duplicate selected Label cell key')
            self.grids[field][ordinal] = 1
        return key

    def _value_end(self):
        if not self.paths:
            raise ValueError('selected Label observation stack mismatch')
        path, _ = self.paths.pop()
        if self.capture_path == path:
            value,temporary = self._decode_capture()
            if len(path) == 2 and path[0] == 'context':
                self._retain(value); self.context[path[1]] = value
            elif len(path) == 3 and path[0] == 'field_meta':
                self._retain(value); self.headers.setdefault(path[1], {})[path[2]] = value
            elif len(path) == 3 and path[0] == 'records' or len(path) == 5 and path[0] == 'field_meta':
                self.key_parts[path[:-1]][path[-1]] = value
            else:
                field = 'records' if path[0] == 'records' else path[1]
                key = self._grid(field, value)
                if key in self.endpoint_keys:
                    self._retain(value)
                    if field == 'records': self.records.append(value)
                    else: self.metadata.setdefault(field, []).append(value)
            self.capture = None; self.capture_path = None
            self.owned -= temporary
        is_record = len(path) == 2 and path[0] == 'records'
        is_meta = len(path) == 4 and path[0] == 'field_meta' and path[2] == 'by_key'
        if is_record or is_meta:
            keys = self.key_parts.pop(path)
            if self.endpoint_keys is None:
                self._grid('records' if is_record else path[1], keys)
                self.owned -= sum(sys.getsizeof(v)+128 for v in keys.values())
        if path in self.active:
            ref = 'sha256:'+self.active.pop(path).hexdigest()
            if path == ('context', 'coverage'): self.coverage_ref = ref
            else: self.component_refs[path[0]] = ref
        if path == ('context',):
            query = self.context.get('query', {})
            symbols, sessions = query.get('symbols'), query.get('sessions')
            if type(symbols) is list and type(sessions) is list:
                if (not symbols or not sessions or any(type(v) is not str for v in symbols+sessions)
                    or len(set(symbols)) != len(symbols) or len(set(sessions)) != len(sessions)):
                    raise ValueError('unique nonempty selected Label query grid required')
                self.reserve(512*(len(symbols)+len(sessions))+1024)
                self.symbols = {v:i for i,v in enumerate(symbols)}
                self.sessions = {v:i for i,v in enumerate(sessions)}
                self.expected = len(symbols)*len(sessions)
                self.owned += 512*(len(symbols)+len(sessions))+1024
            elif self.endpoint_keys is not None:
                raise ValueError('actual Label query grid required for endpoint projection')
        if self.paths and self.stack and self.stack[-1][0] == 'array':
            self.paths[-1][1] += 1

    def finish(self):
        result = super().finish()
        if set(self.component_refs) != {'context', 'records', 'field_meta'}:
            raise ValueError('exact selected Label wire components required')
        if self.expected is not None:
            query = self.context['query']
            fields = query.get('fields')
            if type(fields) is not list or not fields or len(set(fields)) != len(fields):
                raise ValueError('actual selected Label query fields required')
            if set(self.grids) != {'records', *fields} or any(not all(grid) for grid in self.grids.values()):
                raise ValueError('incomplete selected Label query grid')
        result.update(context=self.context, component_refs=self.component_refs,
            coverage_identity=(self.coverage_ref is not None, self.coverage_ref))
        if self.endpoint_keys is not None:
            self.reserve(1024+len(self.endpoint_keys)*64+len(self.metadata)*512)
            result['endpoint_keys'] = sorted(self.endpoint_keys)
            result['endpoint_wire'] = {'context': self.context, 'records': self.records,
                'field_meta': {field:{**self.headers.get(field, {}), 'by_key':items}
                               for field,items in self.metadata.items()}}
        return result


def validate_label_wire_chunks(chunks, *, maximum_workspace_bytes,
                               caller_retained_bytes=None, endpoint_keys=None):
    """One canonical scan; optional endpoint keys are a private bounded request."""
    if type(maximum_workspace_bytes) is not int or maximum_workspace_bytes <= 0:
        raise ValueError('positive Label wire workspace budget required')
    validator = _LabelWireValidator(maximum_workspace_bytes, caller_retained_bytes, endpoint_keys)
    for chunk in chunks:
        if type(chunk) is not bytes:
            raise ValueError('canonical Label wire chunks must be bytes')
        validator.consume(chunk)
    return validator.finish()
