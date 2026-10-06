"""Synthetic saved Signals, explicit calendar and Raw Labels; no training."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import (evaluate_stock_signal, evaluate_stock_signals,
    save_stock_signal_evaluation, load_stock_signal_evaluation)
from axiom_research.stock_artifacts import digest, file_digest, write_json, _read
from axiom_research.stock_signal_evaluation import _identity


def seal(value, key):
    return {**value, key: digest(value)}


def fixture(root, *, dates=None, invalid=(), reverse=False, constant=False, ties=False):
    root.mkdir(parents=True, exist_ok=True)
    calendar = [(date(2024, 1, 2)+timedelta(days=2*i+(i//3))).isoformat() for i in range(12)]
    days = calendar[:4] if dates is None else dates
    universe = [f'S{i:03}' for i in range(24)]
    cutoff = calendar[-1]+'T23:00:00Z'
    query = {'pit_policy': 'synthetic_pit_v1', 'symbols': universe, 'sessions': calendar,
        'domain': 'market_daily', 'fields': ['open','close'],
        'cutoff_by_session': {d: cutoff for d in calendar}, 'purpose': 'label_outcomes',
        'price_basis': 'common_anchor_adjusted_v1', 'adjustment_anchor': calendar[-1]}
    native_query = {**query, 'price_basis': 'unadjusted', 'adjustment_anchor': None}
    raw = seal({'contract_version': 'stock_label_build_v1', 'calendar_ref': digest({
        'contract_version': 'stock_label_calendar_v1', 'sessions': calendar}),
        'label_spec': {'label_id': 'forward_5_session_open_close_v1', 'horizon_sessions': 5,
            'start_session_offset': 1, 'end_session_offset': 5, 'start_price': 'open', 'end_price': 'close',
            'price_basis': 'common_anchor_adjusted_v1', 'adjustment_anchor': calendar[-1], 'normalization': 'none'},
        'source_ref': digest('synthetic_raw_source'), 'source_evidence': {'context': {
            'snapshot_id': 'synthetic_snapshot_v1', 'domain': 'market_daily', 'query': query, 'derivation': {
                'decision_cutoff': cutoff, 'anchor_session': calendar[-1],
                'recipe_version': 'common_anchor_price_v1', 'formula': 'price_t * factor_t / factor_anchor',
                'decision_session':calendar[-1], 'factor_field':'factor', 'factor_domain':'adjustment_factors',
                'price_query': native_query,
                'factor_query': {**native_query,'domain':'adjustment_factors','fields':['factor']}}}},
        'rows': [{'security_id': s, 'feature_session': d, 'start_session': calendar[calendar.index(d)+1],
            'end_session': calendar[calendar.index(d)+5], 'return': (i % 9)/100 + i/1000,
            'label_available_at': calendar[calendar.index(d)+5]+'T20:00:00Z', 'valid': True,
            'invalid_reason': None, 'source_refs': [digest('synthetic_raw_source')]}
            for d in calendar[:4] for i,s in enumerate(universe)]}, 'label_ref')
    training = seal({**{k:v for k,v in raw.items() if k != 'label_ref'}, 'synthetic_training': True}, 'label_ref')
    labels = seal({'contract_version': 'stock_label_bundle_v1', 'training': training, 'evaluation': raw}, 'label_ref')
    proof = [{'session': d, 'sessions': [d], 'cutoffs': {d:d+'T21:00:00Z'},
        'core_plan': {'synthetic': d,'reference_members':{d:{s:None for s in universe}}}, 'core_frame_ref': digest(['frame',d]),
        'membership_ref': digest(['membership',d]), 'adjusted_input_ref': digest(['adjusted',d]),
        'source_evidence': {'membership': {'field':'is_member','batch_ref':digest(['membership',d]),
            'query_context':{'snapshot_id':'synthetic_snapshot_v1','domain':'universe_membership','query':{
                'pit_policy':'synthetic_pit_v1','symbols':universe,'sessions':[d],
                'cutoff_by_session':{d:d+'T21:00:00Z'},'purpose':'decision_facts','fields':['is_member']}}}}} for d in days]
    decision_query = {'pit_policy': 'synthetic_pit_v1', 'symbols': universe, 'sessions': days,
        'cutoff_by_session': {d:d+'T21:00:00Z' for d in days}, 'purpose': 'decision'}
    features = seal({'contract_version': 'stock_feature_build_v1', 'catalog_ref': digest('catalog'),
        'selection': ['synthetic_feature'], 'ordered_features': ['synthetic_feature'],
        'qlib_view': {'snapshot_id': 'synthetic_snapshot_v1', 'view_id': digest('qlib'),
            'queries': [decision_query], 'universe_query': decision_query}, 'input_evidence_ref': digest(proof),
        'rows': [{'security_id': s, 'session': d, 'values': [i % 7], 'availability': [d+'T20:00:00Z'],
            'validity': [True], 'member': True, 'knowledge_cutoff': d+'T21:00:00Z',
            'source_refs': [digest(['frame',d]), digest({'synthetic':d,'reference_members':{d:{s:None for s in universe}}})]}
            for d in days for i,s in enumerate(universe)]}, 'feature_ref')
    dataset = seal({'feature_ref': features['feature_ref'], 'label_ref': training['label_ref'],
        'ordered_features': ['synthetic_feature'], 'fit_cutoff': '2024-01-01T19:00:00Z'}, 'dataset_ref')
    (root/'booster.txt').write_text('synthetic saved booster; never execute')
    model = seal({'contract_version': 'stock_model_release_v1', 'dataset_ref': dataset['dataset_ref'],
        'target_semantics':'synthetic_prediction',
        'feature_ref': features['feature_ref'], 'label_ref': training['label_ref'],
        'ordered_features': ['synthetic_feature'], 'fit_cutoff': '2024-01-01T19:00:00Z',
        'booster_digest': file_digest(root/'booster.txt')}, 'model_ref')
    rows = []
    for d in days:
        for i,s in enumerate(universe):
            valid = (s,d) not in invalid
            score = 1.0 if constant else float(i//3 if ties else 24-i if reverse else i)
            rows.append({'security_id':s, 'session':d, 'knowledge_cutoff':d+'T21:00:00Z',
                'available_at':d+'T20:00:00Z', 'score':score if valid else None, 'valid':valid,
                'invalid_reason':None if valid else 'SYNTHETIC_INVALID', 'member': True,
                'source_refs':[features['feature_ref'], model['model_ref'], digest('catalog'), digest('qlib')]})
    signal = seal({'contract_version':'stock_prediction_run_v1', 'signal_stage':'prediction_raw',
        'score_semantics':'synthetic_prediction', 'score_unit':'dimensionless', 'model_ref':model['model_ref'],
        'feature_ref':features['feature_ref'], 'universe':universe, 'rows':rows}, 'signal_run_ref')
    evidence = seal({'contract_version':'stock_signal_evidence_v1', 'signal_ref':signal['signal_run_ref'],
        'label_ref':raw['label_ref'], 'series':[]}, 'evidence_ref')
    config = {'calendar':calendar, 'symbols':universe, 'prediction_sessions':days,
        'cutoff_by_session':{d:d+'T21:00:00Z' for d in days},
        'snapshot':'synthetic_snapshot_v1', 'pit_policy':'synthetic_pit_v1'}
    experiment = seal({'definition':{'config':config}, 'feature_ref':features['feature_ref'],
        'label_ref':labels['label_ref'], 'dataset_ref':dataset['dataset_ref'], 'model_ref':model['model_ref'],
        'signal_run_ref':signal['signal_run_ref'], 'evidence_ref':evidence['evidence_ref']}, 'experiment_ref')
    values = {'experiment.json':experiment, 'features.json':features, 'feature-inputs.json':proof,
        'labels.json':labels, 'dataset.json':dataset, 'model.json':model, 'predictions.json':signal,
        'signal-evidence.json':evidence}
    for name,value in values.items():
        write_json(root/name,value)
    write_json(root/'manifest.json', {'contract_version':'stock_ml_manifest_v1',
        'experiment_ref':experiment['experiment_ref'],
        'files':{name:file_digest(root/name) for name in [*values,'booster.txt']}})
    return ({'path':str(root/'predictions.json'), 'file_digest':file_digest(root/'predictions.json'),
             'signal_run_ref':signal['signal_run_ref']},
            {'path':str(root/'labels.json'), 'file_digest':file_digest(root/'labels.json'), 'label_ref':raw['label_ref']},
            {'sessions':calendar[:4], 'universe':universe, 'evaluation_cutoff':cutoff, 'calendar':calendar})


def mutate_signal(descriptor, change):
    path = Path(descriptor['path']); root = path.parent
    signal = _read(path); change(signal)
    signal = seal({k:v for k,v in signal.items() if k != 'signal_run_ref'}, 'signal_run_ref')
    write_json(path,signal)
    evidence = _read(root/'signal-evidence.json'); evidence['signal_ref'] = signal['signal_run_ref']
    evidence = seal({k:v for k,v in evidence.items() if k != 'evidence_ref'}, 'evidence_ref'); write_json(root/'signal-evidence.json',evidence)
    exp = _read(root/'experiment.json'); exp.update(signal_run_ref=signal['signal_run_ref'],evidence_ref=evidence['evidence_ref'])
    exp = seal({k:v for k,v in exp.items() if k != 'experiment_ref'}, 'experiment_ref'); write_json(root/'experiment.json',exp)
    manifest = _read(root/'manifest.json'); manifest['experiment_ref'] = exp['experiment_ref']
    manifest['files'] = {name:file_digest(root/name) for name in manifest['files']}; write_json(root/'manifest.json',manifest)
    descriptor.update(file_digest=file_digest(path), signal_run_ref=signal['signal_run_ref'])


def mutate_feature_proof(descriptor, change):
    root=Path(descriptor['path']).parent
    proof=_read(root/'feature-inputs.json'); change(proof); write_json(root/'feature-inputs.json',proof)
    feature=_read(root/'features.json'); feature['input_evidence_ref']=digest(proof)
    feature=seal({k:v for k,v in feature.items() if k!='feature_ref'},'feature_ref'); write_json(root/'features.json',feature)
    dataset=_read(root/'dataset.json'); dataset['feature_ref']=feature['feature_ref']
    dataset=seal({k:v for k,v in dataset.items() if k!='dataset_ref'},'dataset_ref'); write_json(root/'dataset.json',dataset)
    model=_read(root/'model.json'); model.update(feature_ref=feature['feature_ref'],dataset_ref=dataset['dataset_ref'])
    model=seal({k:v for k,v in model.items() if k!='model_ref'},'model_ref'); write_json(root/'model.json',model)
    signal=_read(root/'predictions.json'); signal.update(feature_ref=feature['feature_ref'],model_ref=model['model_ref'])
    for row in signal['rows']:
        row['source_refs']=[feature['feature_ref'],model['model_ref'],digest('catalog'),digest('qlib')]
    signal=seal({k:v for k,v in signal.items() if k!='signal_run_ref'},'signal_run_ref'); write_json(root/'predictions.json',signal)
    old=_read(root/'signal-evidence.json'); old['signal_ref']=signal['signal_run_ref']
    old=seal({k:v for k,v in old.items() if k!='evidence_ref'},'evidence_ref'); write_json(root/'signal-evidence.json',old)
    experiment=_read(root/'experiment.json'); experiment.update(feature_ref=feature['feature_ref'],dataset_ref=dataset['dataset_ref'],
        model_ref=model['model_ref'],signal_run_ref=signal['signal_run_ref'],evidence_ref=old['evidence_ref'])
    experiment=seal({k:v for k,v in experiment.items() if k!='experiment_ref'},'experiment_ref'); write_json(root/'experiment.json',experiment)
    manifest=_read(root/'manifest.json'); manifest.update(experiment_ref=experiment['experiment_ref'],
        files={name:file_digest(root/name) for name in manifest['files']}); write_json(root/'manifest.json',manifest)
    descriptor.update(file_digest=file_digest(root/'predictions.json'),signal_run_ref=signal['signal_run_ref'])


def reseal(report):
    report['evidence_ref'] = _identity(report)
    report['content_digest'] = digest({k:v for k,v in report.items() if k != 'content_digest'})
    return report


def raw_variant(root, descriptor, change):
    bundle = _read(descriptor['path'])
    raw = deepcopy(bundle['evaluation'] if bundle.get('contract_version') == 'stock_label_bundle_v1' else bundle)
    change(raw)
    raw = seal({k:v for k,v in raw.items() if k != 'label_ref'},'label_ref')
    path = root/'raw-variant.json'; write_json(path,raw)
    return {'path':str(path),'file_digest':file_digest(path),'label_ref':raw['label_ref']}


class SavedSignalEvaluationTests(unittest.TestCase):
    def test_saved_evaluation_hit_never_reexecutes_statistics_and_scope_change_reuses_sources(self):
        from axiom_engine.core import evaluate_signal_statistics
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); signal, label, scope = fixture(root/'sources')
            with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as stats:
                report = evaluate_stock_signal(signal, raw_label_input=label, scope=scope)
                self.assertEqual(stats.call_count, 2)
            saved = save_stock_signal_evaluation(report, destination=root/'reports')
            marks = {p: (file_digest(p), p.stat().st_mtime_ns) for p in saved.path.iterdir()}
            with patch('axiom_engine.core.evaluate_signal_statistics',
                       side_effect=AssertionError('saved HIT recomputed statistics')), \
                 patch('axiom_research.stock_signal_evaluation_inputs.file_digest', wraps=file_digest) as reads:
                hit = save_stock_signal_evaluation(report, destination=root/'reports')
                loaded = load_stock_signal_evaluation(saved.path)
                self.assertTrue(hit.reused)
                self.assertEqual(hit.identity, saved.identity)
                self.assertEqual(loaded.to_dict(), report)
                self.assertGreater(reads.call_count, 0)  # HIT verifies sources, it is not zero-I/O.
            self.assertEqual({p: (file_digest(p), p.stat().st_mtime_ns) for p in marks}, marks)
            shorter = {**scope, 'sessions': scope['sessions'][:3]}
            with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as stats:
                changed = evaluate_stock_signal(signal, raw_label_input=label, scope=shorter)
                self.assertEqual(stats.call_count, 2)
            self.assertEqual(changed['input_signal_refs'], report['input_signal_refs'])
            self.assertEqual(changed['label_ref'], report['label_ref'])
            self.assertNotEqual(changed['evidence_ref'], report['evidence_ref'])
            self.assertEqual(changed['coverage']['valid_pair_count'], 72)

    def test_source_record_hashes_unique_paths_once_and_rejects_conflicting_refs(self):
        from collections import Counter
        from axiom_research.stock_signal_evaluation_inputs import _source_records
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent = root/'raw.json'; write_json(parent, {'synthetic': True})
            descriptor = {'path': str(parent), 'file_digest': file_digest(parent)}
            write_json(root/'manifest.json', {'files': {'fold.json': 'unused'}})
            definition = {'repeated_raw': [deepcopy(descriptor) for _ in range(40)]}
            write_json(root/'fold.json', {'definition': {'input_manifest': definition}})
            with patch('axiom_research.stock_signal_evaluation_inputs.file_digest', wraps=file_digest) as hashes:
                records = _source_records(root)
            counts = Counter(str(c.args[0]) for c in hashes.call_args_list)
            self.assertEqual(counts[str(parent)], 1)
            self.assertEqual(records, [{'path': str(p), 'file_digest': file_digest(p)}
                                       for p in sorted((parent, root/'manifest.json', root/'fold.json'), key=str)])
            definition['repeated_raw'][1]['file_digest'] = digest('conflicting source')
            write_json(root/'fold.json', {'definition': {'input_manifest': definition}})
            with self.assertRaisesRegex(ValueError, 'conflicting saved source'):
                _source_records(root)

    def test_source_record_rejects_mutation_during_unique_hash(self):
        from axiom_research.stock_signal_evaluation_inputs import _source_records
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); parent = root/'aaa-raw.json'; write_json(parent, {'synthetic': True})
            desc = {'path': str(parent), 'file_digest': file_digest(parent)}
            write_json(root/'manifest.json', {'files': {'fold.json': 'unused'}})
            write_json(root/'fold.json', {'definition': {'input_manifest': {'raw': desc}}})
            def altered(path):
                value = file_digest(path)
                # The parent was already hashed: a later file changes it.
                # This must be caught by the global post-fingerprint guard.
                if Path(path) == root/'fold.json':
                    with parent.open('a') as stream: stream.write(' ')
                return value
            with patch('axiom_research.stock_signal_evaluation_inputs.file_digest', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'changed during hashing'):
                    _source_records(root)

    def test_single_common_native_save_exact_hit_and_stdlib_fresh_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); signal,label,scope = fixture(root/'input')
            report = evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            self.assertEqual(len(report['series']),4)
            self.assertEqual(report['coverage']['valid_pair_count'],96)
            self.assertEqual(report['summary'],report['coverage']['native']['summary'])
            self.assertNotIn('unit',report['label_spec'])
            saved = save_stock_signal_evaluation(report,destination=root/'reports')
            before = (saved.path/'signal-evidence.json').stat().st_mtime_ns
            self.assertTrue(save_stock_signal_evaluation(report,destination=root/'reports').reused)
            self.assertEqual((saved.path/'signal-evidence.json').stat().st_mtime_ns,before)
            self.assertEqual(load_stock_signal_evaluation(saved.path).to_dict(),report)
            script = '''import builtins,sys
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'axiom_data','axiom_engine','lightgbm','qlib','pandas','numpy'}:
        raise AssertionError('runtime import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from axiom_research import load_stock_signal_evaluation
result=load_stock_signal_evaluation(sys.argv[1])
print(result.identity)
'''
            run = subprocess.run([sys.executable,'-c',script,str(saved.path)],env=os.environ,
                capture_output=True,text=True,check=True)
            self.assertEqual(run.stdout.strip(),report['evidence_ref'])

    def test_common_intersection_and_natural_coverage_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a,label,scope = fixture(root/'a')
            invalid = [(s,day) for s in scope['universe'][:5] for day in scope['sessions']]
            b,_,_ = fixture(root/'b',invalid=invalid,reverse=True)
            reports = evaluate_stock_signals({'z':a,'a':b},raw_label_input=label,scope=scope)
            for name,report in reports.items():
                self.assertEqual(report['coverage']['valid_pair_count'],76)
                self.assertEqual(report['sample_mask_ref'],reports['z']['sample_mask_ref'])
                self.assertIsNone(report['summary']['mean_ic'])
                save_stock_signal_evaluation(report,destination=root/'reports')
            self.assertEqual(reports['z']['coverage']['native']['valid_pair_count'],96)
            self.assertIsNotNone(reports['z']['coverage']['native']['summary']['mean_ic'])
            self.assertEqual(reports['a']['coverage']['native']['valid_pair_count'],76)

    def test_ordered_disjoint_weekly_sources_keep_original_model_refs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a,label,scope = fixture(root/'a')
            first,_,_ = fixture(root/'first',dates=scope['sessions'][:2])
            last,_,_ = fixture(root/'last',dates=scope['sessions'][2:])
            report = evaluate_stock_signal([first,last],raw_label_input=label,scope=scope)
            self.assertEqual(report['input_signal_refs'],[first['signal_run_ref'],last['signal_run_ref']])
            self.assertEqual(report['coverage']['valid_pair_count'],96)
            self.assertNotIn('model_ref',report)
            with self.assertRaisesRegex(ValueError,'ordered and disjoint'):
                evaluate_stock_signal([last,first],raw_label_input=label,scope=scope)
            with self.assertRaisesRegex(ValueError,'ordered and disjoint'):
                evaluate_stock_signal([first,first],raw_label_input=label,scope=scope)

    def test_null_days_minimum_pairs_constant_ties_and_core_delegation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source',ties=True)
            from axiom_engine.core import evaluate_signal_statistics
            from unittest.mock import patch
            with patch('axiom_engine.core.evaluate_signal_statistics', wraps=evaluate_signal_statistics) as operator:
                report=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            self.assertEqual(operator.call_count,2)
            self.assertEqual(report['spec']['rank_ties'],'average')
            self.assertIsNotNone(report['series'][0]['rank_ic'])
            self.assertEqual(report['summary']['icir_reason'],'ZERO_VARIANCE')
            constant,_,_=fixture(root/'constant',constant=True)
            report=evaluate_stock_signal(constant,raw_label_input=label,scope=scope)
            self.assertEqual([r['reason'] for r in report['series']],['CONSTANT_CROSS_SECTION']*4)
            self.assertEqual(report['summary']['icir_reason'],'NO_VALID_SESSIONS')
            invalid=[(s,d) for s in scope['universe'] for d in scope['sessions'][-1:]]
            null,_,_=fixture(root/'null-day',invalid=invalid)
            report=evaluate_stock_signal(null,raw_label_input=label,scope=scope)
            self.assertEqual([r['session'] for r in report['series']],scope['sessions'])
            self.assertEqual(report['series'][-1]['valid_pair_count'],0)
            self.assertIsNone(report['series'][-1]['ic'])
            self.assertEqual(report['coverage']['native']['by_session'][-1]['excluded_counts'],{'SIGNAL_INVALID':24})
            scope['sessions']=scope['sessions'][:1]
            single=evaluate_stock_signal(null,raw_label_input=label,scope=scope)
            self.assertEqual(single['summary']['icir_reason'],'INSUFFICIENT_VALID_SESSIONS')

    def test_end_maturity_and_both_source_and_availability_clocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            early=deepcopy(scope); early['evaluation_cutoff']=scope['calendar'][6]+'T21:00:00Z'
            with self.assertRaisesRegex(ValueError,'source query exceeds evaluation cutoff'):
                evaluate_stock_signal(signal,raw_label_input=label,scope=early)
            def early_source(raw):
                ctx=raw['source_evidence']['context']; ctx['query']['cutoff_by_session']={d:early['evaluation_cutoff'] for d in ctx['query']['sessions']}
                ctx['derivation']['decision_cutoff']=early['evaluation_cutoff']
                for name in ('price_query','factor_query'):
                    q=ctx['derivation'][name]; q['cutoff_by_session']={d:early['evaluation_cutoff'] for d in q['sessions']}
                for row in raw['rows']:
                    row['label_available_at']=early['evaluation_cutoff']
            earlier=raw_variant(root,label,early_source)
            report=evaluate_stock_signal(signal,raw_label_input=earlier,scope=early)
            self.assertEqual([r['valid_pair_count'] for r in report['series']],[24,24,0,0])
            self.assertEqual(report['coverage']['excluded_counts']['LABEL_NOT_MATURE'],48)
            bad=raw_variant(root,label,lambda raw:raw['rows'][0].update(label_available_at='2099-01-01T00:00:00Z'))
            with self.assertRaisesRegex(ValueError,'availability exceeds saved source query'):
                evaluate_stock_signal(signal,raw_label_input=bad,scope=scope)
            # Equivalent aware UTC offsets are normalized at the evaluation boundary.
            equivalent=deepcopy(scope); equivalent['evaluation_cutoff']=scope['calendar'][-1]+'T18:00:00-05:00'
            self.assertEqual(evaluate_stock_signal(signal,raw_label_input=label,scope=equivalent)['evidence_ref'],
                evaluate_stock_signal(signal,raw_label_input=label,scope=scope)['evidence_ref'])

    def test_duplicate_offset_outside_scope_and_signal_clock_are_rejected(self):
        mutations=[lambda value:value['rows'].append(deepcopy(value['rows'][0])),
            lambda value:value['rows'][0].update(session='2099-01-01'),
            lambda value:value['rows'][0].update(security_id='OUTSIDE'),
            lambda value:value['rows'][0].update(available_at='2099-01-01T00:00:00Z'),
            lambda value:value['rows'][0].update(knowledge_cutoff=value['rows'][0]['session']+'T22:00:00Z'),
            lambda value:value['rows'][0].update(source_refs=[digest('wrong')])]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); signal,label,scope=fixture(root/'source'); mutate_signal(signal,mutation)
                with self.assertRaises(ValueError):
                    evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            wrong=raw_variant(root,label,lambda value:value['rows'][0].update(end_session=scope['calendar'][6]))
            with self.assertRaisesRegex(ValueError,'endpoint conflict'):
                evaluate_stock_signal(signal,raw_label_input=wrong,scope=scope)
            wrong=deepcopy(scope); wrong['calendar'].pop(1)
            with self.assertRaisesRegex(ValueError,'outside calendar'):
                evaluate_stock_signal(signal,raw_label_input=label,scope=wrong)

    def test_resealed_saved_binding_count_and_mask_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            report=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            mutations=[lambda value:value['input_evidence']['source_closure']['signal'][0].update(model_ref=digest('wrong')),
                lambda value:value['input_evidence']['sample_mask']['common_keys'].pop(),
                lambda value:value['coverage']['native'].update(valid_pair_count=95),
                lambda value:value['coverage']['common_statistics'].update(input_ref=digest('wrong-input')),
                lambda value:value.update(statistics_input_ref=digest('wrong-input'))]
            for i,mutation in enumerate(mutations):
                with self.subTest(i=i):
                    bad=deepcopy(report); mutation(bad); reseal(bad)
                    path=root/f'wrong-{i}.json'; write_json(path,bad)
                    with self.assertRaises(ValueError):
                        load_stock_signal_evaluation(path)
            saved=save_stock_signal_evaluation(report,destination=root/'reports')
            (saved.path/'signal-evidence.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'file digest mismatch'):
                load_stock_signal_evaluation(saved.path)

    def test_source_format_and_rows_shuffle_identity_rules_and_collision(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            report=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            saved=save_stock_signal_evaluation(report,destination=root/'reports')
            path=Path(signal['path']); original=_read(path)
            path.write_text(json.dumps(original,indent=2)+'\n')
            manifest=_read(path.parent/'manifest.json'); manifest['files']['predictions.json']=file_digest(path)
            write_json(path.parent/'manifest.json',manifest); signal['file_digest']=file_digest(path)
            equivalent=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            self.assertEqual(equivalent['evidence_ref'],report['evidence_ref'])
            self.assertEqual(equivalent['statistics_ref'],report['statistics_ref'])
            # The same immutable ref with a different source-byte descriptor is a collision.
            with self.assertRaises(ValueError):
                save_stock_signal_evaluation(equivalent,destination=root/'reports')
            mutate_signal(signal,lambda value:value['rows'].reverse())
            shuffled=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            self.assertNotEqual(shuffled['evidence_ref'],report['evidence_ref'])
            self.assertEqual([(r['ic'],r['rank_ic'],r['valid_pair_count']) for r in shuffled['series']],
                [(r['ic'],r['rank_ic'],r['valid_pair_count']) for r in report['series']])
            self.assertEqual(shuffled['input_evidence']['sample_mask']['common_keys'],
                report['input_evidence']['sample_mask']['common_keys'])

    def test_old_v1_read_and_saved_implementation_version_do_not_require_current_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            legacy=load_stock_signal_evaluation(Path(signal['path']).parent/'signal-evidence.json')
            self.assertEqual(legacy.to_dict()['contract_version'],'stock_signal_evidence_v1')
            report=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            report['input_evidence']['implementation_sources']['research']['stock_signal_evaluation.py']=digest('old-version')
            report['implementation_ref']=digest(report['input_evidence']['implementation_sources']); reseal(report)
            path=root/'previous-version.json'; write_json(path,report)
            self.assertEqual(load_stock_signal_evaluation(path).identity,report['evidence_ref'])

    def test_training_target_or_resealed_wrong_raw_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            mutations=[lambda value:value.update(contract_version='stock_normalized_label_build_v1'),
                lambda value:value['label_spec'].update(normalization='cross_section_zscore'),
                lambda value:value['label_spec'].update(start_price='close',end_price='open'),
                lambda value:value['label_spec'].update(label_id='forward_1_session_open_close_v1'),
                lambda value:value['rows'][0].update(source_refs=[digest('wrong-source')]),
                lambda value:value['source_evidence']['context'].update(snapshot_id='different-Snapshot'),
                lambda value:value['source_evidence']['context']['derivation']['factor_query'].update(pit_policy='different-PIT'),
                lambda value:value['source_evidence']['context']['derivation']['factor_query'].update(policy_by_session={scope['calendar'][0]:'different-PIT'})]
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    wrong=raw_variant(root,label,mutation)
                    with self.assertRaises(ValueError):
                        evaluate_stock_signal(signal,raw_label_input=wrong,scope=scope)
            def remove_endpoint(raw):
                ctx=raw['source_evidence']['context']; removed=scope['calendar'][5]
                for q in (ctx['query'],ctx['derivation']['price_query'],ctx['derivation']['factor_query']):
                    q['sessions'].remove(removed); q['cutoff_by_session'].pop(removed)
            wrong=raw_variant(root,label,remove_endpoint)
            with self.assertRaisesRegex(ValueError,'valid value mismatch'):
                evaluate_stock_signal(signal,raw_label_input=wrong,scope=scope)

    def test_resealed_feature_source_queries_cannot_follow_original_knowledge(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            def late(proof):
                for entry in proof:
                    entry['cutoffs']={d:d+'T22:00:00Z' for d in entry['sessions']}
                    for source in entry['source_evidence'].values():
                        source['query_context']['query']['cutoff_by_session']=dict(entry['cutoffs'])
            mutate_feature_proof(signal,late)
            with self.assertRaisesRegex(ValueError,'source query exceeds knowledge cutoff'):
                evaluate_stock_signal(signal,raw_label_input=label,scope=scope)

    def test_resealed_statistics_reasons_must_match_counts_and_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            report=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            for field in ('series','summary'):
                with self.subTest(field=field):
                    bad=deepcopy(report); statistics=bad['coverage']['common_statistics']
                    if field=='series':
                        statistics['series'][0]['reason']='INSUFFICIENT_PAIRS'
                        bad['series']=deepcopy(statistics['series'])
                    else:
                        statistics['summary'][0]['icir_reason']='INSUFFICIENT_VALID_SESSIONS'
                        bad['summary']=deepcopy(statistics['summary'][0])
                    statistics['statistics_ref']=digest({k:v for k,v in statistics.items() if k!='statistics_ref'})
                    bad['statistics_ref']=statistics['statistics_ref']; reseal(bad)
                    path=root/(field+'-reason.json'); write_json(path,bad)
                    with self.assertRaisesRegex(ValueError,'reason mismatch'):
                        load_stock_signal_evaluation(path)
            bad=deepcopy(report); bad['spec']['std_ddof']=True; reseal(bad)
            path=root/'bool-policy.json'; write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'spec binding mismatch'):
                load_stock_signal_evaluation(path)

    def test_atomic_publish_failure_and_race_checks_complete_winner(self):
        from unittest.mock import patch
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); signal,label,scope=fixture(root/'source')
            report=evaluate_stock_signal(signal,raw_label_input=label,scope=scope)
            target=root/'reports'/report['evidence_ref'][7:]
            with patch('axiom_research.stock_signal_evaluation.os.rename',side_effect=OSError('synthetic publication failure')):
                with self.assertRaisesRegex(OSError,'publication failure'):
                    save_stock_signal_evaluation(report,destination=root/'reports')
            self.assertFalse(target.exists())
            self.assertEqual(list((root/'reports').iterdir()),[])
            def race(stage,destination):
                self.assertEqual({p.name for p in stage.iterdir()},{'signal-evidence.json','manifest.json'})
                shutil.copytree(stage,destination)
                raise FileExistsError('synthetic complete winner')
            with patch('axiom_research.stock_signal_evaluation.os.rename',side_effect=race):
                winner=save_stock_signal_evaluation(report,destination=root/'reports')
            self.assertTrue(winner.reused)
            self.assertEqual(winner.to_dict(),report)
            other=deepcopy(report); other['limitations'].append('different saved output'); reseal(other)
            with self.assertRaisesRegex(ValueError,'collision'):
                save_stock_signal_evaluation(other,destination=root/'reports')

    def test_v2_saved_fold_source_closure_and_fresh_loader(self):
        self._saved_fold_source_closure(compact=False)

    def test_compact_v2_saved_fold_source_closure_and_fresh_loader(self):
        self._saved_fold_source_closure(compact=True)

    def _saved_fold_source_closure(self, *, compact):
        # Reuse the existing tiny three-security owner fixture, with an inert
        # backend. This verifies v2 clocks/closure; the 24-ID cases above cover
        # numerical sample admission and the minimum-pair boundary.
        from test_stock_folds import fixture as fold_fixture, backend
        from axiom_research import build_stock_ml_fold_from_saved_inputs
        from axiom_research.stock_label_normalization import normalize_forward_labels
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); parents=root/'parents'; parents.mkdir()
            calendar = ([(date(2021,1,1)+timedelta(days=10*i+(i//7))).isoformat() for i in range(100)]
                        if compact else None)
            fit_index = 80 if compact else 65
            manifest,spec=fold_fixture(parents,projected=False,calendar=calendar,
                                      two_year=compact,fit_index=fit_index)
            proof=_read(parents/'proof.json')
            calendar=manifest['calendar']; future=calendar[-1]
            evaluation_index = fit_index+7
            spec['evaluation_cutoff']=calendar[evaluation_index]+'T20:30:00+08:00'
            proof.append({'session':future,'core_plan':{'synthetic_session':future},'core_frame_ref':digest(['frame',future])})
            manifest['feature_parents'][0]['sessions'].append(future)
            for evidence in proof:
                d=evidence['session']; evidence.update(sessions=[d],cutoffs={d:d+'T20:30:00+08:00'},
                    membership_ref=digest(['membership',d]),adjusted_input_ref=digest(['adjusted',d]))
                evidence['core_plan']['reference_members']={d:{s:None for s in manifest['universe'] if s!='C'}}
                evidence['source_evidence']={'membership':{'field':'is_member','batch_ref':evidence['membership_ref'],
                    'query_context':{'snapshot_id':manifest['snapshot'],'domain':'universe_membership','query':{
                        'symbols':manifest['universe'],'sessions':[d],'pit_policy':manifest['pit_policy'],
                        'cutoff_by_session':evidence['cutoffs'],'purpose':'decision_facts','fields':['is_member']}}}}
            write_json(parents/'proof.json',proof)
            feature=_read(parents/'features.json'); feature['input_evidence_ref']=digest(proof)
            for original in deepcopy(feature['rows'][:3]):
                original.update(session=future,knowledge_cutoff=future+'T20:30:00+08:00',
                    availability=[future+'T20:00:00+08:00'],source_refs=[digest(['frame',future]),None])
                feature['rows'].append(original)
            proof_by_day={p['session']:p for p in proof}
            for row in feature['rows']:
                row['source_refs'][1]=digest(proof_by_day[row['session']]['core_plan'])
            feature=seal({k:v for k,v in feature.items() if k!='feature_ref'},'feature_ref')
            write_json(parents/'features.json',feature)
            normalized=normalize_forward_labels(_read(parents/'raw.json'),features=feature,cutoff=spec['fit_cutoff'])
            write_json(parents/'normalized.json',normalized)
            raw=_read(parents/'evaluation.json'); outcome_days=calendar[:evaluation_index+1]
            raw['source_ref']=digest('raw'); raw['label_spec'].update(start_session_offset=1,end_session_offset=5,
                start_price='open',end_price='close',price_basis='common_anchor_adjusted_v1',adjustment_anchor=outcome_days[-1])
            ctx=raw['source_evidence']['context']; ctx['domain']='market_daily'
            query=ctx['query']; query.update(sessions=outcome_days,fields=['open','close'],
                cutoff_by_session={d:spec['evaluation_cutoff'] for d in outcome_days},
                price_basis='common_anchor_adjusted_v1',adjustment_anchor=outcome_days[-1])
            native={**query,'price_basis':'unadjusted','adjustment_anchor':None}
            ctx['derivation']={'decision_cutoff':spec['evaluation_cutoff'],'anchor_session':outcome_days[-1],
                'recipe_version':'common_anchor_price_v1','formula':'price_t * factor_t / factor_anchor',
                'decision_session':outcome_days[-1],'factor_field':'factor','factor_domain':'adjustment_factors',
                'price_query':native,'factor_query':{**native,'fields':['factor']}}
            raw=seal({k:v for k,v in raw.items() if k!='label_ref'},'label_ref'); write_json(parents/'evaluation.json',raw)
            def desc(name,ref):
                value=_read(parents/name)
                return {'path':str(parents/name),'file_digest':file_digest(parents/name),ref:value[ref]}
            manifest['feature_parents'][0]['features']=desc('features.json','feature_ref')
            manifest['feature_parents'][0]['input_evidence'].update(file_digest=file_digest(parents/'proof.json'),input_evidence_ref=digest(proof))
            manifest['training_labels'][0]['normalized']=desc('normalized.json','label_ref')
            manifest['evaluation_labels']=desc('evaluation.json','label_ref')
            with patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend), \
                 patch('axiom_research.stock_folds._environment',return_value={'synthetic':True}), \
                 patch('axiom_research.stock_folds._implementation',return_value={'synthetic':digest('code')}):
                fold=build_stock_ml_fold_from_saved_inputs(manifest,fold_spec=spec,destination=root/'folds')
            if compact:
                self.assertEqual(fold.to_dict()['contract_version'],'stock_ml_fold_v2')
                self.assertEqual(len(_read(fold.path/'feature-slice.json')['rows']),6)
                self.assertNotIn('rows',_read(fold.path/'label-slice.json'))
            signal=fold.predictions()
            descriptor={'path':str(fold.path/'predictions.json'),'file_digest':file_digest(fold.path/'predictions.json'),
                'signal_run_ref':signal['signal_run_ref']}
            scope={'sessions':sorted({r['session'] for r in signal['rows']}),'universe':manifest['universe'],
                'calendar':calendar,'evaluation_cutoff':spec['evaluation_cutoff']}
            report=evaluate_stock_signal(descriptor,raw_label_input=manifest['evaluation_labels'],scope=scope)
            self.assertEqual(report['coverage']['reference_key_count'],4)
            self.assertEqual(report['coverage']['excluded_counts']['NOT_MEMBER'],2)
            saved=save_stock_signal_evaluation(report,destination=root/'reports')
            from axiom_research import load_stock_ml_batch_inputs, load_stock_ml_fold
            batch_definition={'contract_version':'stock_ml_batch_inputs_v1',
                'folds':[{'input_manifest':manifest,'fold_spec':spec}]}
            with load_stock_ml_batch_inputs(batch_definition) as batch:
                with patch('axiom_research.stock_fold_inputs.load_saved_feature_inputs',
                           side_effect=AssertionError('reader reloaded common Feature')), \
                     patch('axiom_research.stock_fold_inputs._admit_saved_fold_labels',
                           side_effect=AssertionError('reader readmitted Label')):
                    self.assertEqual(load_stock_ml_fold(fold.path,batch=batch).identity,fold.identity)
                    loaded=load_stock_signal_evaluation(saved.path,batch=batch)
                    self.assertEqual(loaded.to_dict(),report)
                self.assertEqual(batch.metrics['fold_projection_calls'],2)
            with self.assertRaisesRegex(ValueError,'closed'):
                load_stock_signal_evaluation(saved.path,batch=batch)
            with self.assertRaisesRegex(ValueError,'validated saved batch'):
                load_stock_signal_evaluation(saved.path,batch=True)
            write_json(root/'reader-batch.json',batch_definition)
            batch_script='''import builtins,json,sys
original=builtins.__import__
def guard(name,*a,**kw):
    if name.split('.')[0] in {'axiom_data','axiom_engine','qlib','lightgbm','pandas','pyarrow'}:
        raise AssertionError('reader imported runtime '+name)
    return original(name,*a,**kw)
builtins.__import__=guard
from axiom_research import load_stock_ml_batch_inputs,load_stock_signal_evaluation
with load_stock_ml_batch_inputs(json.load(open(sys.argv[1]))) as batch:
    print(load_stock_signal_evaluation(sys.argv[2],batch=batch).identity)
'''
            fresh_batch=subprocess.run([sys.executable,'-c',batch_script,str(root/'reader-batch.json'),str(saved.path)],
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,check=True,timeout=15)
            self.assertEqual(fresh_batch.stdout.strip(),report['evidence_ref'])
            script='''import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'axiom_data','axiom_engine','lightgbm','qlib','pandas','numpy'}:
        raise AssertionError('runtime import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from axiom_research import load_stock_signal_evaluation
print(load_stock_signal_evaluation(sys.argv[1]).identity)
'''
            completed=subprocess.run([sys.executable,'-c',script,str(saved.path)],env=os.environ,capture_output=True,text=True,check=True)
            self.assertEqual(completed.stdout.strip(),report['evidence_ref'])
            # An unused parent date after evaluation is admitted above; a
            # source query after a selected Feature's knowledge still fails.
            day=scope['sessions'][0]
            late=deepcopy(proof)
            for entry in late:
                if entry['session']==day:
                    entry['cutoffs']={d:d+'T20:45:00+08:00' for d in entry['sessions']}
                    for source in entry['source_evidence'].values():
                        source['query_context']['query']['cutoff_by_session']=dict(entry['cutoffs'])
            write_json(parents/'proof.json',late)
            feature['input_evidence_ref']=digest(late)
            feature=seal({k:v for k,v in feature.items() if k!='feature_ref'},'feature_ref'); write_json(parents/'features.json',feature)
            normalized=normalize_forward_labels(_read(parents/'raw.json'),features=feature,cutoff=spec['fit_cutoff'])
            write_json(parents/'normalized.json',normalized)
            manifest['feature_parents'][0]['features']=desc('features.json','feature_ref')
            manifest['feature_parents'][0]['input_evidence'].update(file_digest=file_digest(parents/'proof.json'),input_evidence_ref=digest(late))
            manifest['training_labels'][0]['normalized']=desc('normalized.json','label_ref')
            with patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=backend), \
                 patch('axiom_research.stock_folds._environment',return_value={'synthetic':True}), \
                 patch('axiom_research.stock_folds._implementation',return_value={'synthetic':digest('code')}):
                bad_fold=build_stock_ml_fold_from_saved_inputs(manifest,fold_spec=spec,destination=root/'folds')
            descriptor.update(path=str(bad_fold.path/'predictions.json'),file_digest=file_digest(bad_fold.path/'predictions.json'),
                signal_run_ref=bad_fold.predictions()['signal_run_ref'])
            with self.assertRaisesRegex(ValueError,'source query exceeds knowledge cutoff'):
                evaluate_stock_signal(descriptor,raw_label_input=manifest['evaluation_labels'],scope=scope)


if __name__ == '__main__':
    unittest.main()
