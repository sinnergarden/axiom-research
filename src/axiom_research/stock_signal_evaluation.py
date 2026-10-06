"""Saved Raw Label / stock Signal evaluation; loading uses only the stdlib.

Research admits sources, clocks, membership and exact sample keys. The single
optional Core import is inside the evaluator: all numerical statistics belong
to that operator. Loading never imports it or recomputes a correlation or IR.
"""
from copy import deepcopy
from collections import Counter
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
import os
import tempfile

from .stock_artifacts import digest, file_digest, _read, _verify_ref, write_json
from .stock_label_contracts import _finite
from .stock_signal_evaluation_inputs import _require, _scope, _inputs


SPEC = {'minimum_pairs': 20, 'rank_ties': 'average', 'std_ddof': 1,
        'time_weighting': 'equal_valid_sessions', 'annualization': 'none'}
TOP_FIELDS = set(('contract_version evidence_ref content_digest input_signal_refs '
    'input_evidence label_ref label_spec scope spec_ref spec sample_mask_ref '
    'statistics_input_ref statistics_ref series summary coverage status limitations '
    'implementation_ref').split())


def _implementation():
    module = import_module('axiom_engine.core.signal_statistics')
    core = Path(module.__file__).parent
    return {'research': {name: file_digest(Path(__file__).parent/name) for name in
        ('stock_signal_evaluation.py', 'stock_signal_evaluation_inputs.py', 'stock_artifacts.py', 'stock_label_contracts.py',
         'stock_fold_artifacts.py', 'stock_fold_inputs.py', 'stock_training.py')},
        'core': {name: file_digest(core/name) for name in ('signal_statistics.py', 'contracts.py')},
        'operator': 'axiom_engine.core.evaluate_signal_statistics'}


def _identity(report):
    return digest({key: report[key] for key in ('contract_version', 'input_signal_refs', 'label_ref',
        'scope', 'spec_ref', 'sample_mask_ref', 'statistics_input_ref', 'implementation_ref')})


def evaluate_stock_signals(signal_inputs, *, raw_label_input, scope):
    """Evaluate an insertion-ordered Signal group on common and native samples.

    Each input is an explicit saved descriptor or an ordered nonoverlapping
    weekly list. Scope is exactly sessions/universe/evaluation_cutoff/calendar.
    The complete frozen calendar must match original Raw Label calendar_ref.
    """
    admitted = _inputs(signal_inputs, raw_label_input, _scope(scope))
    from axiom_engine.core import evaluate_signal_statistics
    common = evaluate_signal_statistics(admitted['common_input'], spec=SPEC).to_dict()
    native = evaluate_signal_statistics(admitted['native_input'], spec=SPEC).to_dict()
    implementation = _implementation()
    common_counts = Counter(day for _, day in admitted['common_keys'])
    reports = {}
    for name, signal_key in admitted['signal_keys'].items():
        series = [row for row in common['series'] if row['signal_key'] == signal_key]
        summary = next(row for row in common['summary'] if row['signal_key'] == signal_key)
        own_native = {**deepcopy(admitted['native_coverage'][name]),
            'statistics_input_ref': native['input_ref'], 'statistics_ref': native['statistics_ref'],
            'series': [row for row in native['series'] if row['signal_key'] == signal_key],
            'summary': next(row for row in native['summary'] if row['signal_key'] == signal_key)}
        coverage = deepcopy(admitted['native_coverage'][name])
        coverage['valid_pair_count'] = len(admitted['common_keys'])
        coverage['excluded_counts']['OUTSIDE_COMMON_SAMPLE'] = own_native['valid_pair_count'] - coverage['valid_pair_count']
        for row in coverage['by_session']:
            common_count = common_counts[row['session']]
            row['excluded_counts']['OUTSIDE_COMMON_SAMPLE'] = row['valid_pair_count'] - common_count
            row['valid_pair_count'] = common_count
        coverage.update(comparison_mode='common_valid_key_intersection', native=own_native,
            common_statistics=common, native_statistics=native)
        report = {'contract_version': 'stock_signal_evidence_v2',
            'input_signal_refs': admitted['refs'][name],
            'input_evidence': {'signal_inputs': deepcopy(signal_inputs), 'raw_label_input': deepcopy(raw_label_input),
                'signal_name': name, 'source_closure': admitted['closures'], 'sample_mask': admitted['mask'],
                'implementation_sources': implementation},
            'label_ref': admitted['raw']['label_ref'], 'label_spec': deepcopy(admitted['raw']['label_spec']),
            'scope': admitted['scope'], 'spec': deepcopy(SPEC), 'spec_ref': common['spec_ref'],
            'sample_mask_ref': digest(admitted['mask']), 'statistics_input_ref': common['input_ref'],
            'statistics_ref': common['statistics_ref'], 'series': series, 'summary': summary,
            'coverage': coverage, 'status': 'COMPLETE' if summary['valid_ic_session_count'] else 'NO_VALID_SESSIONS',
            'limitations': ['Forward-label statistics are not account returns.',
                'Overlapping label horizons and short-sample IR are descriptive, without a confidence claim.',
                'Saved source proof verifies declared Snapshot/query bindings; it does not requery supplier facts.',
                'Label spec is copied from the saved Raw Label build; absent unit fields remain absent.'],
            'implementation_ref': digest(implementation)}
        report['evidence_ref'] = _identity(report)
        report['content_digest'] = digest(report)
        reports[name] = report
    return reports


def evaluate_stock_signal(signal_input, *, raw_label_input, scope):
    """Evaluate one saved Signal or an ordered disjoint weekly Signal list."""
    return evaluate_stock_signals({'signal': signal_input}, raw_label_input=raw_label_input, scope=scope)['signal']


def _statistics(value, expected_input, spec):
    """Verify saved Core bindings and axes/counts without numerical execution."""
    _require(set(value) == {'contract_version', 'input_ref', 'spec_ref', 'statistics_ref', 'series', 'summary'} and
        value['contract_version'] == 'signal_statistics_v1', 'saved statistics contract mismatch')
    _verify_ref(value, 'statistics_ref')
    _require(value['input_ref'] == digest(expected_input) and value['spec_ref'] == digest(spec),
        'saved statistics exact input/spec mismatch')
    axes = [(key, day) for key in expected_input['signal_keys'] for day in expected_input['sessions']]
    _require([(r['signal_key'], r['session']) for r in value['series']] == axes and
        [r['signal_key'] for r in value['summary']] == expected_input['signal_keys'], 'saved statistics axes mismatch')
    counts = Counter((p['signal_key'], p['session']) for p in expected_input['pairs'])
    valid_sessions = Counter()
    for row in value['series']:
        _require(set(row) == {'signal_key', 'session', 'valid_pair_count', 'ic', 'rank_ic', 'reason'},
            'saved statistics series fields mismatch')
        count = counts[row['signal_key'], row['session']]
        _require(type(row['valid_pair_count']) is int and row['valid_pair_count'] == count, 'saved statistics pair count mismatch')
        for field in ('ic', 'rank_ic'):
            _require(row[field] is None or (_finite(row[field]) and -1 <= row[field] <= 1), 'saved correlation value invalid')
            valid_sessions[row['signal_key'], field] += row[field] is not None
        _require(row['reason'] in (None, 'INSUFFICIENT_PAIRS', 'CONSTANT_CROSS_SECTION', 'NON_FINITE_CORRELATION'),
            'saved correlation reason invalid')
        if count < spec['minimum_pairs']:
            _require(row['ic'] is None and row['rank_ic'] is None and row['reason'] == 'INSUFFICIENT_PAIRS',
                'saved insufficient-pair null/reason mismatch')
        else:
            _require(row['reason'] != 'INSUFFICIENT_PAIRS', 'saved sufficient-pair reason mismatch')
            if row['ic'] is not None and row['rank_ic'] is not None:
                _require(row['reason'] is None, 'saved finite correlation reason mismatch')
            elif row['reason'] == 'CONSTANT_CROSS_SECTION':
                _require(row['ic'] is None and row['rank_ic'] is None, 'saved constant correlation null mismatch')
            else:
                _require(row['reason'] == 'NON_FINITE_CORRELATION', 'saved missing correlation reason mismatch')
    fields = set(('signal_key valid_ic_session_count mean_ic ic_std icir icir_reason '
        'valid_rank_ic_session_count mean_rank_ic rank_ic_std rank_icir rank_icir_reason').split())
    for row in value['summary']:
        _require(set(row) == fields, 'saved statistics summary fields mismatch')
        for count_field, field in (('valid_ic_session_count', 'ic'), ('valid_rank_ic_session_count', 'rank_ic')):
            _require(type(row[count_field]) is int and row[count_field] == valid_sessions[row['signal_key'], field],
                'saved valid session count mismatch')
        for field in ('mean_ic', 'ic_std', 'icir', 'mean_rank_ic', 'rank_ic_std', 'rank_icir'):
            _require(row[field] is None or _finite(row[field]), 'saved summary number invalid')
        for count, mean, std, ir, reason in (
                ('valid_ic_session_count', 'mean_ic', 'ic_std', 'icir', 'icir_reason'),
                ('valid_rank_ic_session_count', 'mean_rank_ic', 'rank_ic_std', 'rank_icir', 'rank_icir_reason')):
            n = row[count]
            if n == 0:
                _require(row[mean] is row[std] is row[ir] is None and row[reason] == 'NO_VALID_SESSIONS',
                    'saved empty summary null/reason mismatch')
            else:
                _require(_finite(row[mean]) and -1 <= row[mean] <= 1, 'saved valid summary mean required')
                if n == 1:
                    _require(row[std] is row[ir] is None and row[reason] == 'INSUFFICIENT_VALID_SESSIONS',
                        'saved single-session summary null/reason mismatch')
                else:
                    _require(_finite(row[std]) and row[std] >= 0, 'saved summary standard deviation invalid')
                    _require((row[std] == 0 and row[ir] is None and row[reason] == 'ZERO_VARIANCE') or
                        (row[std] > 0 and _finite(row[ir]) and row[reason] is None), 'saved IR null/reason mismatch')


def _verify_report(report, *, batch=None):
    _require(type(report) is dict and set(report) == TOP_FIELDS and
        report['contract_version'] == 'stock_signal_evidence_v2', 'unsupported saved Signal evaluation')
    _verify_ref(report, 'content_digest')
    _require(report['evidence_ref'] == _identity(report), 'saved evaluation identity mismatch')
    evidence = report['input_evidence']
    _require(set(evidence) == {'signal_inputs', 'raw_label_input', 'signal_name', 'source_closure',
        'sample_mask', 'implementation_sources'}, 'saved input evidence fields mismatch')
    _require(report['implementation_ref'] == digest(evidence['implementation_sources']) and
        report['spec'] == SPEC and report['spec_ref'] == digest(SPEC) == digest(report['spec']),
        'saved implementation/spec binding mismatch')
    scope = _scope(report['scope'])
    _require(scope == report['scope'], 'saved evaluation scope is not canonical')
    order = [item['name'] for item in evidence['sample_mask']['signals']]
    _require(len(order) == len(set(order)) and set(order) == set(evidence['signal_inputs']),
        'saved comparison order mismatch')
    selected = _inputs({name: evidence['signal_inputs'][name] for name in order}, evidence['raw_label_input'], scope,
                       batch=batch)
    name = evidence['signal_name']; _require(name in selected['signal_keys'], 'saved comparison Signal missing')
    _require(evidence['source_closure'] == selected['closures'] and evidence['sample_mask'] == selected['mask'] and
        report['sample_mask_ref'] == digest(selected['mask']), 'saved source/sample-mask binding mismatch')
    _require(report['input_signal_refs'] == selected['refs'][name] and report['label_ref'] == selected['raw']['label_ref'] and
        report['label_spec'] == selected['raw']['label_spec'], 'saved Signal/Raw Label binding mismatch')
    coverage = report['coverage']; common, native = coverage['common_statistics'], coverage['native_statistics']
    _statistics(common, selected['common_input'], SPEC); _statistics(native, selected['native_input'], SPEC)
    key = selected['signal_keys'][name]
    own_series = [r for r in common['series'] if r['signal_key'] == key]
    own_summary = next(r for r in common['summary'] if r['signal_key'] == key)
    _require(report['statistics_input_ref'] == common['input_ref'] and report['statistics_ref'] == common['statistics_ref'] and
        report['series'] == own_series and report['summary'] == own_summary, 'saved common statistics projection mismatch')
    expected_native = {**selected['native_coverage'][name], 'statistics_input_ref': native['input_ref'],
        'statistics_ref': native['statistics_ref'], 'series': [r for r in native['series'] if r['signal_key'] == key],
        'summary': next(r for r in native['summary'] if r['signal_key'] == key)}
    expected = deepcopy(selected['native_coverage'][name]); expected['valid_pair_count'] = len(selected['common_keys'])
    expected['excluded_counts']['OUTSIDE_COMMON_SAMPLE'] = expected_native['valid_pair_count'] - expected['valid_pair_count']
    common_counts = Counter(day for _, day in selected['common_keys'])
    for row in expected['by_session']:
        count = common_counts[row['session']]
        row['excluded_counts']['OUTSIDE_COMMON_SAMPLE'] = row['valid_pair_count'] - count
        row['valid_pair_count'] = count
    expected.update(comparison_mode='common_valid_key_intersection', native=expected_native,
        common_statistics=common, native_statistics=native)
    _require(coverage == expected, 'saved common/native coverage mismatch')
    _require(report['status'] == ('COMPLETE' if own_summary['valid_ic_session_count'] else 'NO_VALID_SESSIONS'),
        'saved evaluation status mismatch')
    return report


@dataclass(frozen=True)
class StockSignalEvaluation:
    path: Path
    reused: bool = False
    _verified: object = field(default=None, init=False, repr=False, compare=False)

    def to_dict(self):
        if self._verified is not None:
            return deepcopy(self._verified)
        return _read(self.path/'signal-evidence.json' if self.path.is_dir() else self.path)

    @property
    def identity(self):
        return self.to_dict()['evidence_ref']


def _verified_evaluation(path, reused, report):
    saved = StockSignalEvaluation(Path(path), reused)
    object.__setattr__(saved, '_verified', deepcopy(report))
    return saved


def load_stock_signal_evaluation(path, *, batch=None):
    """Read saved values and verify their immutable sources; no runtime imports.

    Direct legacy v1 JSON preserves its original ref-only read behavior. V2
    directories additionally freeze file bytes in the existing records layout.
    A batch reuses only inputs admitted in this process; saved statistics and
    every fold output are still verified without numerical execution.
    """
    path = Path(path)
    if path.is_file():
        # Dispatch is only a hint. V3 hashes/parses its consumed buffer again
        # against the group manifest and returns that verified snapshot.
        if _read(path).get('contract_version') == 'stock_signal_evidence_v3':
            _require(batch is None, 'frozen v3 evaluation does not accept a training batch')
            from .stock_signal_evaluation_projection import _load_v3_report
            return _load_v3_report(path)
    elif path.is_dir() and _read(path/'manifest.json').get('contract_version') == 'stock_signal_evaluation_manifest_v2':
        _require(batch is None, 'frozen v3 evaluation does not accept a training batch')
        from .stock_signal_evaluation_projection import _load_v3_report
        return _load_v3_report(path)
    if batch is not None:
        from .stock_batch import _data
        _data(batch)
        batch._check_sources()
    path = Path(path)
    if path.is_dir():
        manifest = _read(path/'manifest.json')
        _require(set(manifest) == {'contract_version', 'evidence_ref', 'files'} and
            manifest['contract_version'] == 'stock_signal_evaluation_manifest_v1' and
            set(manifest['files']) == {'signal-evidence.json'}, 'saved evaluation manifest mismatch')
        _require(file_digest(path/'signal-evidence.json') == manifest['files']['signal-evidence.json'],
            'saved evaluation file digest mismatch')
        report = _read(path/'signal-evidence.json')
        _require(manifest['evidence_ref'] == report['evidence_ref'], 'saved evaluation manifest identity mismatch')
    else:
        report = _read(path)
    if report.get('contract_version') == 'stock_signal_evidence_v1':
        _require(batch is None, 'saved batch requires v2 evaluation')
        _verify_ref(report, 'evidence_ref')
    else:
        _verify_report(report, batch=batch)
    if batch is not None:
        batch._check_sources()
    return StockSignalEvaluation(path)


def save_stock_signal_evaluation(report, *, destination):
    """Atomically publish immutable signal-evidence.json under evidence_ref."""
    report = deepcopy(report); _verify_report(report)
    target = Path(destination)/report['evidence_ref'][7:]
    def existing():
        saved = load_stock_signal_evaluation(target)
        _require(saved.to_dict() == report, 'immutable saved evaluation collision')
        return StockSignalEvaluation(target, True)
    if target.exists():
        return existing()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.signal-evaluation-', dir=target.parent) as temporary:
        stage = Path(temporary)/'complete'; stage.mkdir()
        write_json(stage/'signal-evidence.json', report)
        write_json(stage/'manifest.json', {'contract_version': 'stock_signal_evaluation_manifest_v1',
            'evidence_ref': report['evidence_ref'], 'files': {'signal-evidence.json': file_digest(stage/'signal-evidence.json')}})
        load_stock_signal_evaluation(stage)
        try:
            os.rename(stage, target)
        except OSError:
            if target.exists():
                return existing()
            raise
    return StockSignalEvaluation(target)
