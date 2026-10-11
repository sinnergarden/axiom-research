"""Research partition contracts using Engine's shared immutable buffer codec."""
from axiom_engine.feature_library.storage import (DTYPES, BUFFER_FIELDS, shape_size,
    _buffer_array, write_buffer as _write_buffer, write_part as _write_part,
    _verify_written as _engine_verify_written)
from .stock_artifacts import digest, file_digest
from .stock_fold_inputs import require, seal, ordered
from .stock_label_contracts import _instant, _session

PARTITION_FIELDS = {'table', 'fold_spec_ref', 'row_index_ref', 'schema_digest',
    'row_offset', 'row_count', 'columns', 'buffers', 'metadata', 'partition_ref'}
TABLES = {'features', 'training_raw_labels', 'training_normalized_labels', 'evaluation_raw_labels'}

def _verify_written(path, expected, marks=None):
    return _engine_verify_written(path, expected, marks, _file_hasher=file_digest)

def write_buffer(root, values, *, dtype, shape, _verified_marks=None):
    return _write_buffer(root, values, dtype=dtype, shape=shape,
        _verified_marks=_verified_marks, _file_hasher=file_digest)

def write_part(root, value, ref_key, *, _verified_marks=None):
    return _write_part(root, value, ref_key, _verified_marks=_verified_marks, _file_hasher=file_digest)


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
