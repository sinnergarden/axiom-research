"""Bounded, scalar-only validation of the project's canonical JSON bytes.

The workspace counter is a conservative allocation reservation, not an RSS
measurement. Caller-owned iterator/backing storage is supplied separately by
``caller_retained_bytes``. No decoded container or persistent proof is created.
"""

import hashlib
import json
import math
import re
import sys


_BARE_END = re.compile(rb'[\x00-\x20{}\[\],:"]')
_STRING_EVENT = re.compile(rb'["\\\x00-\x1f]')
_FIXED_WORKSPACE = 8192


def _reject_constant(_value):
    raise ValueError("Nonfinite JSON constant")


class _Validator:
    __slots__ = (
        "maximum", "retained", "stack", "frame_bytes", "root", "token",
        "token_kind", "token_is_key", "escaped", "input_bytes", "temporary",
        "peak", "size", "tokens", "hasher",
    )

    def __init__(self, maximum, retained):
        self.maximum = maximum
        self.retained = retained
        self.stack = []
        self.frame_bytes = 0
        self.root = "start"
        self.token = None
        self.token_kind = None
        self.token_is_key = False
        self.escaped = False
        self.input_bytes = 0
        self.temporary = 0
        self.peak = 0
        self.size = 0
        self.tokens = 0
        self.hasher = hashlib.sha256()
        self.reserve()

    def workspace(self):
        return (
            _FIXED_WORKSPACE + sys.getsizeof(self) + sys.getsizeof(self.stack)
            + self.frame_bytes + self.input_bytes + self.temporary
            + (sys.getsizeof(self.token) if self.token is not None else 0)
        )

    def reserve(self, extra=0):
        caller = 0 if self.retained is None else self.retained()
        if type(caller) is not int or caller < 0:
            raise ValueError("caller_retained_bytes must return a nonnegative int")
        workspace = self.workspace() + extra
        if caller + workspace > self.maximum:
            raise ValueError("Canonical JSON workspace budget exceeded")
        self.peak = max(self.peak, workspace)

    def push(self, kind):
        # Include a possible list reallocation before creating either object.
        self.reserve(256 + 16 * (len(self.stack) + 1))
        frame = [kind, "first", None]
        self.stack.append(frame)
        self.frame_bytes += sys.getsizeof(frame)

    def pop(self):
        # list.pop may shrink/reallocate the backing pointer array as well.
        self.reserve(sys.getsizeof(self.stack))
        frame = self.stack.pop()
        self.frame_bytes -= sys.getsizeof(frame)
        if frame[2] is not None:
            self.frame_bytes -= sys.getsizeof(frame[2])
        self.value_done()

    def value_done(self):
        self._value_end()
        if not self.stack:
            self.root = "done"
        else:
            frame = self.stack[-1]
            if frame[1] not in ("value", "first"):
                raise ValueError("Invalid JSON value position")
            frame[1] = "after"

    def _span(self, chunk, start, end):
        """Private observation hook; the canonical grammar still owns admission."""

    def _value_start(self):
        pass

    def _value_end(self):
        pass

    def _scalar_value(self, value):
        pass

    def start_token(self, kind, is_key=False):
        self.reserve(128)
        self.token = bytearray()
        self.token_kind = kind
        self.token_is_key = is_key

    def append(self, chunk, start, end):
        if start == end:
            return
        length = end - start
        # Old and new bytearray allocations may coexist during growth, along
        # with the sliced input span. Reserve before either allocation.
        self.reserve(2 * (len(self.token) + length) + length + 256)
        self.token.extend(chunk[start:end])
        self._span(chunk, start, end)

    def finish_token(self):
        # CPython scalar decode/parse/reencode can hold several Unicode copies,
        # a bytes copy and encoder/decoder scratch simultaneously. This bound
        # deliberately exceeds their payload sizes, including wide Unicode.
        scratch = 24 * len(self.token) + 16384
        self.reserve(scratch)
        self.temporary = scratch
        try:
            try:
                raw = bytes(self.token)
                text = raw.decode("utf-8", "strict")
                value = json.loads(text, parse_constant=_reject_constant)
                if self.token_kind == "string":
                    if type(value) is not str:
                        raise ValueError("Invalid JSON string")
                elif type(value) not in (type(None), bool, int, float):
                    raise ValueError("Invalid JSON scalar")
                if type(value) is float and not math.isfinite(value):
                    raise ValueError("Nonfinite JSON number")
                canonical = json.dumps(
                    value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ).encode("utf-8", "strict")
                if canonical != raw:
                    raise ValueError("Noncanonical JSON scalar")
            except (ValueError, UnicodeError, OverflowError):
                # Keep error messages independent of the input's size/content.
                raise ValueError("Invalid or noncanonical JSON scalar") from None
            if self.token_is_key:
                frame = self.stack[-1]
                previous = frame[2]
                if previous is not None and not previous < value:
                    raise ValueError("JSON object keys must increase strictly")
                self.reserve(sys.getsizeof(value))
                self.frame_bytes += sys.getsizeof(value)
                if previous is not None:
                    self.frame_bytes -= sys.getsizeof(previous)
                frame[2] = value
                frame[1] = "colon"
            else:
                self._scalar_value(value)
                self.value_done()
            self.tokens += 1
        finally:
            self.temporary = 0
            self.token = None
            self.token_kind = None
            self.token_is_key = False

    def scan_string(self, chunk, start, cursor):
        end = len(chunk)
        if self.escaped and cursor < end:
            if chunk[cursor] < 32:
                raise ValueError("Raw control byte in JSON string")
            self.escaped = False
            cursor += 1
        while cursor < end:
            match = _STRING_EVENT.search(chunk, cursor)
            if match is None:
                break
            position = match.start()
            byte = chunk[position]
            if byte == 34:
                self.append(chunk, start, position + 1)
                self.finish_token()
                return position + 1
            if byte < 32:
                raise ValueError("Raw control byte in JSON string")
            cursor = position + 1
            if cursor == end:
                self.escaped = True
                break
            if chunk[cursor] < 32:
                raise ValueError("Raw control byte in JSON string")
            cursor += 1
        self.append(chunk, start, end)
        return end

    def scan_bare(self, chunk, cursor):
        match = _BARE_END.search(chunk, cursor)
        end = len(chunk) if match is None else match.start()
        self.append(chunk, cursor, end)
        if match is not None:
            self.finish_token()
        return end

    def begin_value(self, byte):
        self._value_start()
        if byte == 123:
            self.push("object")
        elif byte == 91:
            self.push("array")
        elif byte == 34:
            self.start_token("string")
        elif byte in b"}],:" or byte <= 32:
            raise ValueError("Invalid JSON value")
        else:
            self.start_token("bare")

    def consume(self, chunk):
        self.input_bytes = sys.getsizeof(chunk)
        self.reserve()
        self.hasher.update(chunk)
        self.size += len(chunk)
        cursor = 0
        while cursor < len(chunk):
            if self.token_kind == "string":
                cursor = self.scan_string(chunk, cursor, cursor)
                continue
            if self.token_kind == "bare":
                cursor = self.scan_bare(chunk, cursor)
                continue
            byte = chunk[cursor]
            if byte <= 32:
                raise ValueError("Whitespace or control outside JSON string")
            frame = self.stack[-1] if self.stack else None
            state = self.root if frame is None else frame[1]
            if state == "done":
                raise ValueError("Bytes after JSON root")
            if frame is not None and frame[0] == "object":
                if state in ("first", "key"):
                    if byte == 125 and state == "first":
                        self._span(chunk, cursor, cursor + 1)
                        self.pop()
                    elif byte == 34:
                        self.start_token("string", is_key=True)
                    else:
                        raise ValueError("Expected JSON object key")
                elif state == "colon":
                    if byte != 58:
                        raise ValueError("Expected JSON colon")
                    frame[1] = "value"
                    self._span(chunk, cursor, cursor + 1)
                elif state == "after":
                    self._span(chunk, cursor, cursor + 1)
                    if byte == 125:
                        self.pop()
                    elif byte == 44:
                        frame[1] = "key"
                    else:
                        raise ValueError("Expected JSON object separator")
                else:
                    self.begin_value(byte)
                    if self.token_kind is None:
                        self._span(chunk, cursor, cursor + 1)
            elif frame is not None and state == "after":
                self._span(chunk, cursor, cursor + 1)
                if byte == 93:
                    self.pop()
                elif byte == 44:
                    frame[1] = "value"
                else:
                    raise ValueError("Expected JSON array separator")
            elif frame is not None and state == "first" and byte == 93:
                self._span(chunk, cursor, cursor + 1)
                self.pop()
            else:
                self.begin_value(byte)
                if self.token_kind is None:
                    self._span(chunk, cursor, cursor + 1)
            if self.token_kind == "bare":
                cursor = self.scan_bare(chunk, cursor)
            elif self.token_kind == "string":
                cursor = self.scan_string(chunk, cursor, cursor + 1)
            else:
                cursor += 1
        self.input_bytes = 0

    def finish(self):
        if self.token_kind == "bare":
            self.finish_token()
        if self.token_kind is not None or self.stack or self.root != "done":
            raise ValueError("Incomplete JSON document")
        self.reserve(512)
        return {
            "digest": "sha256:" + self.hasher.hexdigest(),
            "size": self.size,
            "peak_workspace_bytes": self.peak,
            "token_count": self.tokens,
        }


def validate_canonical_chunks(
    chunks, *, maximum_workspace_bytes, caller_retained_bytes=None
):
    """Validate canonical UTF-8 JSON without materializing its containers.

    ``token_count`` counts completed scalar values and object keys. Container
    punctuation is excluded. The reported peak excludes caller-retained bytes,
    but every reservation checks their combined size against the given limit.
    The finite iterator must yield exact ``bytes`` objects; empty chunks are OK.
    """
    if type(maximum_workspace_bytes) is not int or maximum_workspace_bytes <= 0:
        raise ValueError("maximum_workspace_bytes must be a positive int")
    if caller_retained_bytes is not None and not callable(caller_retained_bytes):
        raise ValueError("caller_retained_bytes must be callable")
    # Admit the fixed state, including its initially empty stack, before it is
    # allocated. Later checks use the actual state/header sizes as well.
    caller = 0 if caller_retained_bytes is None else caller_retained_bytes()
    if type(caller) is not int or caller < 0:
        raise ValueError("caller_retained_bytes must return a nonnegative int")
    if caller + _FIXED_WORKSPACE + 512 > maximum_workspace_bytes:
        raise ValueError("Canonical JSON workspace budget exceeded")
    validator = _Validator(maximum_workspace_bytes, caller_retained_bytes)
    iterator = iter(chunks)
    while True:
        try:
            chunk = next(iterator)
        except StopIteration:
            break
        if type(chunk) is not bytes:
            raise ValueError("Canonical JSON chunks must be bytes")
        validator.consume(chunk)
        del chunk
    return validator.finish()
