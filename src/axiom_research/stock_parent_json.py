"""Structural workspace admission for the existing legacy parent decoder.

This pass retains one scalar and a bounded syntax stack. It estimates the
subsequent stdlib whole-text/object/pairs-hook allocations without creating a
parent DOM. Legacy whitespace, key order and scalar encodings remain accepted;
the existing decoder still enforces its duplicate-key rule. This is a workspace
reservation, not an RSS measurement or a replacement business JSON protocol.
"""
import json
import os
import sys

from .stock_canonical_json import _Validator, _reject_constant


_FIXED_DECODE = 16384
_ARRAY_ENTRY = 32
_DICT_ENTRY = 128
_PAIR_ENTRY = 96
_KEY_MEMO_ENTRY = 128
_POSTCHECK_NODE = 256


class _ParentPreflight(_Validator):
    __slots__ = ('physical_size', 'source_kind', 'decoded_bytes', 'pairs_bytes',
        'memo_bytes', 'nodes', 'scalar_scratch', 'maximum_depth', 'decode_peak')

    def __init__(self, maximum, retained, physical_size):
        self.physical_size, self.source_kind = physical_size, 1
        self.decoded_bytes, self.pairs_bytes, self.memo_bytes, self.nodes = 0, 0, 256, 0
        self.scalar_scratch, self.maximum_depth, self.decode_peak = 0, 0, 0
        super().__init__(maximum, retained)

    def decode_workspace(self):
        # CPython UTF-8 conversion starts with byte-length Unicode capacity,
        # can widen it before shrinking, and may hold old/new buffers. Use raw
        # document kind/capacity, not the final string's character count. The
        # same bound covers universal newline old/new Unicode and bytes.
        unicode_capacity = 128+self.source_kind*(self.physical_size+1)
        text = 2*self.physical_size+2*unicode_capacity
        decode = (_FIXED_DECODE+text+self.decoded_bytes+self.pairs_bytes+
                  self.memo_bytes+self.scalar_scratch)
        # The post-decode resident walk has a seen-ID set and recursion frames;
        # unlike decoding, it no longer retains the whole source text/hook lists.
        post = (_FIXED_DECODE+self.decoded_bytes+self.nodes*_POSTCHECK_NODE+
                self.maximum_depth*1024)
        return max(decode, post)

    def reserve(self, extra=0):
        caller = 0 if self.retained is None else self.retained()
        if type(caller) is not int or caller < 0:
            raise ValueError('caller_retained_bytes must return a nonnegative int')
        current = self.workspace()+extra
        future = self.decode_workspace()
        if caller+max(current, future) > self.maximum:
            raise ValueError('Legacy JSON parent workspace budget exceeded')
        self.peak = max(self.peak, current)
        self.decode_peak = max(self.decode_peak, future)

    def push(self, kind):
        # No DOM is allocated here. These are conservative future containers,
        # including spare pointer/hash-table capacity and their empty headers.
        self.decoded_bytes += 256 if kind == 'object' else 64
        self.nodes += 1
        self.maximum_depth = max(self.maximum_depth, len(self.stack)+1)
        super().push(kind)

    def value_done(self):
        if self.stack:
            if self.stack[-1][0] == 'object':
                self.decoded_bytes += _DICT_ENTRY
                # The hook's list/tuple pairs can coexist with its output dict.
                # Summing all pairs is an upper bound across nested objects.
                self.pairs_bytes += _PAIR_ENTRY
            else:
                # Includes list overallocation and old/new pointer arrays.
                self.decoded_bytes += _ARRAY_ENTRY
        super().value_done()

    def finish_token(self):
        # Reuse the existing scanner's admitted token buffer. No canonical
        # re-encoding or sorted-key requirement is imposed on a legacy parent.
        length = len(self.token)
        scratch = 24*length+16384
        self.reserve(scratch)
        self.temporary = scratch
        try:
            try:
                raw = bytes(self.token)
                text = raw.decode('utf-8', 'strict')
                value = json.loads(text, parse_constant=_reject_constant)
                if self.token_kind == 'string':
                    if type(value) is not str:
                        raise ValueError('Invalid JSON string')
                elif type(value) not in (type(None), bool, int, float):
                    raise ValueError('Invalid JSON scalar')
            except (ValueError, UnicodeError, OverflowError):
                raise ValueError('Invalid legacy JSON scalar') from None
            self.reserve(sys.getsizeof(value))
            self.decoded_bytes += sys.getsizeof(value)
            self.nodes += 1
            # The whole-text decoder already has its Unicode source reserved.
            # Allow scalar construction/builder copies, without a canonical
            # encoder's additional output text/bytes used by coverage syntax.
            self.scalar_scratch = max(self.scalar_scratch, 16*length+16384)
            if self.token_is_key:
                self.memo_bytes += _KEY_MEMO_ENTRY
                self.stack[-1][1] = 'colon'
            else:
                self.value_done()
            self.tokens += 1
            self.reserve()
        finally:
            self.temporary = 0
            self.token = None
            self.token_kind = None
            self.token_is_key = False

    def consume(self, chunk):
        # This grammar differs only where legacy JSON permits whitespace.
        # String/bare scanning, token growth and stack allocation are shared.
        self.input_bytes = sys.getsizeof(chunk)
        if chunk:
            largest = max(chunk)
            self.source_kind = max(self.source_kind, 4 if largest >= 240 else 2 if largest >= 196 else 1)
        self.reserve()
        self.size += len(chunk)
        cursor = 0
        while cursor < len(chunk):
            if self.token_kind == 'string':
                cursor = self.scan_string(chunk, cursor, cursor)
                continue
            if self.token_kind == 'bare':
                cursor = self.scan_bare(chunk, cursor)
                continue
            byte = chunk[cursor]
            if byte in b' \t\r\n':
                cursor += 1
                continue
            frame = self.stack[-1] if self.stack else None
            state = self.root if frame is None else frame[1]
            if state == 'done':
                raise ValueError('Bytes after JSON root')
            if frame is not None and frame[0] == 'object':
                if state in ('first', 'key'):
                    if byte == 125 and state == 'first':
                        self.pop()
                    elif byte == 34:
                        self.start_token('string', is_key=True)
                    else:
                        raise ValueError('Expected JSON object key')
                elif state == 'colon':
                    if byte != 58:
                        raise ValueError('Expected JSON colon')
                    frame[1] = 'value'
                elif state == 'after':
                    if byte == 125:
                        self.pop()
                    elif byte == 44:
                        frame[1] = 'key'
                    else:
                        raise ValueError('Expected JSON object separator')
                else:
                    self.begin_value(byte)
            elif frame is not None and state == 'after':
                if byte == 93:
                    self.pop()
                elif byte == 44:
                    frame[1] = 'value'
                else:
                    raise ValueError('Expected JSON array separator')
            elif frame is not None and state == 'first' and byte == 93:
                self.pop()
            else:
                self.begin_value(byte)
            if self.token_kind == 'bare':
                cursor = self.scan_bare(chunk, cursor)
            elif self.token_kind == 'string':
                cursor = self.scan_string(chunk, cursor, cursor+1)
            else:
                cursor += 1
        self.input_bytes = 0

    def finish(self):
        if self.token_kind == 'bare':
            self.finish_token()
        if self.token_kind is not None or self.stack or self.root != 'done':
            raise ValueError('Incomplete JSON document')
        if self.size != self.physical_size:
            raise ValueError('Legacy JSON physical size changed')
        self.reserve(1024)
        return {'size': self.size, 'token_count': self.tokens,
            'decode_workspace_bytes': self.decode_peak,
            'scanner_workspace_peak_bytes': self.peak,
            'decoded_graph_upper_bytes': self.decoded_bytes,
            'pairs_hook_upper_bytes': self.pairs_bytes,
            'key_memo_upper_bytes': self.memo_bytes,
            'postcheck_workspace_upper_bytes': self.nodes*_POSTCHECK_NODE+self.maximum_depth*1024}


def preflight_parent_json(chunks, *, physical_size, maximum_workspace_bytes,
                          caller_retained_bytes=None):
    """Admit future legacy stdlib allocations, without retaining a decoded DOM."""
    if type(physical_size) is not int or physical_size < 0:
        raise ValueError('physical_size must be a nonnegative int')
    if type(maximum_workspace_bytes) is not int or maximum_workspace_bytes <= 0:
        raise ValueError('maximum_workspace_bytes must be a positive int')
    if caller_retained_bytes is not None and not callable(caller_retained_bytes):
        raise ValueError('caller_retained_bytes must be callable')
    caller = 0 if caller_retained_bytes is None else caller_retained_bytes()
    if type(caller) is not int or caller < 0:
        raise ValueError('caller_retained_bytes must return a nonnegative int')
    if caller+8192+1024 > maximum_workspace_bytes:
        raise ValueError('Legacy JSON parent workspace budget exceeded')
    scanner = _ParentPreflight(maximum_workspace_bytes, caller_retained_bytes, physical_size)
    for chunk in chunks:
        if type(chunk) is not bytes:
            raise ValueError('Legacy JSON chunks must be bytes')
        scanner.consume(chunk)
        del chunk
    return scanner.finish()


def stat_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def preflight_parent_file(stream, *, expected_stat, maximum_workspace_bytes,
                           caller_retained_bytes=None, block_size=65536):
    """Preflight one watched descriptor; reject in-place mutation before/after."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError('positive preflight block_size required')
    if stat_identity(os.fstat(stream.fileno())) != expected_stat:
        raise ValueError('JSON parent changed before preflight')
    def chunks():
        while chunk := stream.read(block_size):
            yield chunk
    result = preflight_parent_json(chunks(), physical_size=expected_stat[2],
        maximum_workspace_bytes=maximum_workspace_bytes, caller_retained_bytes=caller_retained_bytes)
    if stat_identity(os.fstat(stream.fileno())) != expected_stat:
        raise ValueError('JSON parent changed during preflight')
    return result
