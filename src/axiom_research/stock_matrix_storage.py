"""Strict, content-addressed raw buffers for saved stock matrices.

This module is a storage codec, not a numerical executor.  It imports neither
Data, Core nor NumPy.  Values use the same binary64 representation at every
column count; file names are conveniences, never an admission shortcut.
"""
from array import array
from hashlib import sha256
import math
import os
from pathlib import Path
import sys
import tempfile

from .stock_artifacts import digest, file_digest, write_json
from .stock_fold_inputs import require, seal, ordered
from .stock_label_contracts import _instant, _session


DTYPES = {
    'float64_le': ('d', 8), 'bool_u8': ('B', 1),
    'int64_le': ('q', 8), 'uint64_le': ('Q', 8),
    'int32_le': ('i', 4), 'uint8': ('B', 1),
}
BUFFER_FIELDS = {'path', 'file_digest', 'dtype', 'shape', 'buffer_digest'}
PARTITION_FIELDS = {'table', 'fold_spec_ref', 'row_index_ref', 'schema_digest',
    'row_offset', 'row_count', 'columns', 'buffers', 'metadata', 'partition_ref'}
TABLES = {'features', 'training_raw_labels', 'training_normalized_labels',
          'evaluation_raw_labels'}


def shape_size(shape):
    require(type(shape) is list and bool(shape) and all(type(n) is int and n >= 0
            for n in shape), 'explicit nonnegative integer buffer shape required')
    return math.prod(shape)


def _buffer_array(values, dtype):
    require(dtype in DTYPES, 'unsupported matrix buffer dtype')
    # Native column handoff stays columnar. Snapshot before writing; no mutable
    # caller can change the hash's input while publication is in progress.
    np=sys.modules.get('numpy')
    if np is not None and isinstance(values,np.ndarray):
        expected={'float64_le':'<f8','bool_u8':'u1','int32_le':'<i4',
                  'int64_le':'<i8','uint64_le':'<u8','uint8':'u1'}[dtype]
        require(values.ndim==1 and values.dtype==np.dtype(expected),'native matrix dtype/shape mismatch')
        require(bool(np.isfinite(values).all()),'finite native matrix values required')
        if dtype=='bool_u8':require(bool(((values==0)|(values==1)).all()),'native bool values must be0/1')
        return np.frombuffer(values.tobytes(),dtype=values.dtype)
    code, size = DTYPES[dtype]
    out = array(code)
    require(out.itemsize == size, 'platform raw buffer size mismatch')
    for value in values:
        if dtype == 'float64_le':
            require(type(value) in (int, float) and math.isfinite(value),
                    'finite binary64 value required')
        elif dtype == 'bool_u8':
            require(type(value) is bool, 'boolean matrix value required')
        else:
            require(type(value) is int, 'integer matrix value required')
            bits = size * 8
            low = -(1 << (bits-1)) if dtype.startswith('int') else 0
            high = (1 << (bits-1)) if dtype.startswith('int') else (1 << bits)
            require(low <= value < high, 'matrix integer out of range')
        out.append(value)
    if sys.byteorder != 'little' and size > 1:
        out.byteswap()
    return out


def write_buffer(root, values, *, dtype, shape):
    """Publish one finite, explicit raw buffer and return its exact descriptor."""
    try:
        shape = list(shape)
        count = shape_size(shape)
        packed = _buffer_array(values, dtype)
        require(len(packed) == count, 'matrix buffer shape mismatch')
        raw = memoryview(packed).cast('B')
        reference = 'sha256:' + sha256(raw).hexdigest()
        directory = Path(root)/'buffers'; directory.mkdir(parents=True, exist_ok=True)
        path = directory/(reference[7:]+'.bin')
        if path.exists():
            require(not path.is_symlink() and file_digest(path) == reference,
                    'existing matrix buffer changed')
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix='.buffer-', dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                # A concurrent identical publisher is harmless, but corrupt/orphaned
                # data is always verified before it can become a descriptor.
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    require(not path.is_symlink() and file_digest(path) == reference,
                            'concurrent matrix buffer changed')
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
        require(path.stat().st_size == count*DTYPES[dtype][1] and file_digest(path) == reference,
                'published matrix buffer mismatch')
        return {'path': str(path.resolve()), 'file_digest': reference, 'dtype': dtype,
                'shape': shape, 'buffer_digest': reference}
    finally:
        values=packed=raw=None


def write_part(root, value, ref_key):
    """Publish a sealed canonical JSON child; byte and content refs stay separate."""
    require(type(value) is dict, 'matrix JSON object required')
    require(value.get(ref_key) == digest({k:v for k,v in value.items() if k != ref_key}),
            'matrix child content identity mismatch')
    directory = Path(root)/'metadata'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/(value[ref_key][7:]+'.json')
    with tempfile.NamedTemporaryFile(prefix='.metadata-', dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_json(temporary, value)
        expected = file_digest(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(not path.is_symlink() and file_digest(path) == expected,
                    'existing matrix metadata changed')
        require(file_digest(path) == expected, 'published matrix metadata mismatch')
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(path.resolve()), 'file_digest': expected, ref_key: value[ref_key]}


def row_index(sessions, security_ids):
    """The single complete, day-major key grid shared by partitions/selectors."""
    sessions = list(sessions); security_ids = list(security_ids)
    ordered(sessions, 'matrix sessions'); ordered(security_ids, 'matrix securities')
    for day in sessions: _session(day)
    return seal({'contract_version': 'stock_matrix_row_index_v1', 'sessions': sessions,
        'security_ids': security_ids, 'order': 'session_security',
        'row_count': len(sessions)*len(security_ids)}, 'row_index_ref')


def instant_us(value):
    """Exact aware UTC microseconds, without float timestamps or rounding."""
    from datetime import datetime, timezone
    delta = _instant(value)-datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*1000000+delta.microseconds


def write_partition(root, *, table, fold_spec_ref, row_index_ref, schema_digest,
                    row_offset, row_count, columns, buffers, metadata):
    """Seal an internal partition. Caller owns its current lineage metadata."""
    require(table in TABLES and (fold_spec_ref is None) == (table == 'features'),
            'matrix partition table/fold mismatch')
    require(type(row_offset) is int and row_offset >= 0 and type(row_count) is int and row_count > 0,
            'positive partition row range required')
    require(type(columns) is list and bool(columns) and len(set(columns)) == len(columns) and
            all(type(c) is str and bool(c) for c in columns), 'ordered partition columns required')
    require(type(buffers) is dict and bool(buffers) and all(type(v) is dict and
            set(v) == BUFFER_FIELDS for v in buffers.values()), 'explicit partition buffers required')
    require(type(metadata) is dict and set(metadata) == {'path','file_digest','metadata_ref'},
            'explicit partition metadata required')
    return seal({'table': table, 'fold_spec_ref': fold_spec_ref, 'row_index_ref': row_index_ref,
        'schema_digest': schema_digest, 'row_offset': row_offset, 'row_count': row_count,
        'columns': list(columns), 'buffers': buffers, 'metadata': metadata}, 'partition_ref')
