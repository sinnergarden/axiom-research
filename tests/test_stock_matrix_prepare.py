"""Bounded four-fit prepare acceptance with real Label/Core mathematics.

Only the public Data boundary is synthetic. No provider, Feature executor,
fit, prediction or account operation is used by these tests.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from axiom_engine.core import execute_cs_zscore_batch
from axiom_research.stock_artifacts import _read, _verify_ref, digest, file_digest
from axiom_research.stock_feature_inputs import _feature_wire
from axiom_research.stock_fold_inputs import seal
from axiom_research.stock_label_contracts import _instant
from axiom_research.stock_label_normalization import normalize_forward_labels
from axiom_research.stock_matrix_prepare import prepare_stock_ml_batch_inputs
from axiom_research.stock_matrix_reader import VerifiedMatrixStore, validate_saved_core_result
from axiom_research.stock_batch import load_stock_ml_batch_inputs
from test_stock_feature_inputs import SyntheticInputs
from test_stock_matrix_storage import SyntheticWideInputs


def fixture_native(descriptor, ref_key):
    """Independent full restoration only for these tiny synthetic oracles."""
    assert file_digest(descriptor['path']) == descriptor['file_digest']
    value=_read(descriptor['path'])
    if value.get('contract_version') in ('stock_native_json_carrier_v1','stock_native_json_carrier_v2'):
        _verify_ref(value,'carrier_ref')
        wrapper=value; value=deepcopy(wrapper['skeleton'])
        for slot in wrapper['coverage_slots']+wrapper.get('wire_slots',[]):
            blob=slot['bytes']
            assert file_digest(blob['path']) == blob['buffer_digest'] == blob['file_digest']
            assert Path(blob['path']).stat().st_size == blob['shape'][0]
            parent=value
            for step in slot['path'][:-1]: parent=parent[step]
            parent[slot['path'][-1]]=_read(blob['path'])
    _verify_ref(value,ref_key)
    assert value[ref_key] == descriptor[ref_key]
    return value


@dataclass(frozen=True)
class Query:
    domain: str
    fields: tuple
    symbols: tuple
    sessions: tuple
    pit_policy: str
    cutoff_by_session: dict
    purpose: str = 'decision_facts'
    price_basis: str = 'unadjusted'
    policy_by_session: dict | None = None
    adjustment_anchor: str | None = None
    universe_id: str | None = None

    def to_dict(self):
        value=asdict(self)
        for key in ('fields','symbols','sessions'): value[key]=list(value[key])
        return value


class Batch:
    def __init__(self,wire):
        self.wire=wire; self.field_meta=wire['field_meta']; self.context=wire['context']
        amount=sum(sys.getsizeof(row)+sum(sys.getsizeof(v) for v in row.values()) for row in wire['records'])
        usage=type('SyntheticFrameUsage',(),{'sum':lambda _:amount})()
        self.frame=type('SyntheticFrame',(),{'memory_usage':lambda _,**kwargs:usage})()
    def to_json(self): return deepcopy(self.wire)


class PublicDataFixture:
    """Public read and adjust boundaries expose deterministic exact source wires."""
    def __init__(self,spec): self.spec=spec; self.queries=[]; self.adjusted=[]; self.cache_clears=0
    def clear_cache(self): self.cache_clears+=1
    def read(self,*,snapshot,query):
        if snapshot != self.spec['snapshot']: raise AssertionError('Snapshot changed')
        self.queries.append(query.to_dict()); rows=[]; meta={field:{'dtype':'float64',
            'unit':'CNY/share' if query.domain=='market_daily' else 'dimensionless','by_key':[]}
            for field in query.fields}
        for day in query.sessions:
            day_index=self.spec['calendar'].index(day)
            for i,security in enumerate(query.symbols):
                row={'security_id':security,'session':day}
                for field in query.fields:
                    row[field]=1.0 if field=='factor' else 10.0+day_index*.01 if field=='open' else 11.0+i*i*.4+day_index*.01
                    meta[field]['by_key'].append({'security_id':security,'session':day,
                        'usable_from':day+'T08:15:00.123456Z','missing_reason':None,
                        'revision_id':digest([query.domain,security,day,query.cutoff_by_session[day]]),
                        'availability_basis':'synthetic_explicit_public_reader'})
                rows.append(row)
        return Batch({'records':rows,'field_meta':meta,'context':{
            'contract_version':'data_batch_v1','snapshot_id':snapshot,'reader_version':'synthetic/1',
            'domain':query.domain,'contract_id':query.domain+'_v1','source_profile_id':'synthetic',
            'limitations':['synthetic source, not investment evidence'],'query':query.to_dict()}})

    def adjust(self,price,factors,*,fields,anchor_session,decision_session,factor_field):
        wire=price.to_json(); factor=factors.to_json(); pq=deepcopy(wire['context']['query'])
        fq=deepcopy(factor['context']['query']); cutoff=pq['cutoff_by_session'][anchor_session]
        wire['context']['query'].update(price_basis='common_anchor_adjusted_v1',adjustment_anchor=anchor_session)
        wire['context']['derivation']={'recipe_version':'common_anchor_price_v1',
            'formula':'price_t * factor_t / factor_anchor','factor_field':factor_field,
            'anchor_session':anchor_session,'decision_session':decision_session,'decision_cutoff':cutoff,
            'price_query':pq,'factor_query':fq}
        factor_meta={(r['security_id'],r['session']):r for r in factor['field_meta']['factor']['by_key']}
        for field in fields:
            wire['field_meta'][field]['recipe_version']='common_anchor_price_v1'
            wire['field_meta'][field]['by_key']=[{'security_id':r['security_id'],'session':r['session'],
                'missing_reason':None,'price_provenance':r,
                'factor_provenance':deepcopy(factor_meta[r['security_id'],r['session']]),
                'anchor_factor_provenance':deepcopy(factor_meta[r['security_id'],anchor_session])}
                for r in wire['field_meta'][field]['by_key']]
        self.adjusted.append(deepcopy(wire)); return Batch(wire)


class PrepareFeatureFixture(SyntheticWideInputs):
    def __init__(self,root):
        SyntheticInputs.__init__(self,root,feature_count=90)
        chosen=deepcopy(self.catalog.select(self.selection)[0]); identity=self.catalog.identity
        self.columns=['SYN'+str(i).zfill(3) for i in range(6)]
        self.selection=[{'id':c,'semantic_version':'1.0.0'} for c in self.columns]
        self.spec.update(ordered_features=self.columns,feature_selection=self.selection)
        self.catalog=type('SyntheticCatalog',(),{'identity':identity,'select':lambda _,selection:
            [{**deepcopy(chosen),'id':c} for c in self.columns]})()

    def day(self,day):
        rows,proof=SyntheticInputs.day(self,day)
        proof['core_plan']['outputs']=[{'node':c,'column':{'name':c,'dtype':'float64',
            'unit':'dimensionless','stage':'cross_sectional','missing':'preserve'}} for c in self.columns]
        for row in rows:
            row['values']=[float(1+i) for i in range(6)]; row['validity']=[True]*6
            row['availability']=[day+'T20:00:00.123456+08:00']*6; row['reasons']=[[] for _ in range(6)]
            row['source_refs']=[proof['core_frame_ref'],digest(proof['core_plan'])]
        return rows,proof

    def folds(self):
        out=[]
        first=self.history-1+65
        for pos in (first,first+3,first+6,first+9):
            fit=self.calendar[pos]; prediction=self.calendar[pos:pos+3]
            out.append({'contract_version':'stock_ml_fold_spec_v1','training_window':{
                'unit':'feature_sessions','length':65,'end':'previous_fit_session'},
                'fit_session':fit,'fit_cutoff':fit+'T20:30:00+08:00',
                'simulated_model_available_at':fit+'T20:45:00+08:00',
                'oos_trade_sessions':self.calendar[pos+1:pos+4],
                'inference_cutoff_by_session':{d:d+'T21:00:00+08:00' for d in prediction},
                'evaluation_cutoff':self.calendar[-1]+'T20:30:00+08:00'})
        return out


class MatrixPrepareTests(unittest.TestCase):
    def test_four_fit_native_selection_and_existing_cs_golden(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); feature=PrepareFeatureFixture(root); saved=feature.matrix()
            data=PublicDataFixture(feature.spec); folds=feature.folds(); module=types.ModuleType('axiom_data')
            module.QuerySpec=Query; module.adjust_prices=data.adjust; metrics={}
            options={'row_block_sessions':32,'column_block':32,'maximum_resident_bytes':64*1024**2,
                     'normalization_backend':'core_cs_batch_v1'}
            with patch.dict(sys.modules,{'axiom_data':module}), \
                 patch('axiom_research.stock_ml._implementation',return_value=feature.implementation), \
                 patch('axiom_research.stock_ml._environment',return_value=feature.environment), \
                 patch('axiom_engine.core.execute_cs_zscore_batch',wraps=execute_cs_zscore_batch) as core, \
                 patch('axiom_research.stock_ml._iter_stock_feature_days',side_effect=AssertionError('Feature executed')), \
                 patch('axiom_research.stock_training.fit_predict_stock_model',side_effect=AssertionError('model executed')):
                batch=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=folds,
                    destination=root/'prepared',preparation_options=options,metrics=metrics)
                self.assertEqual(core.call_count,4)
                self.assertEqual(metrics['core_calls'],4)
                self.assertEqual(metrics['feature_core_calls'],0); self.assertEqual(metrics['train_calls'],0)
                queries=deepcopy(data.queries); view=_read(batch['prepared_view']['path'])
                _verify_ref(batch,'content_digest'); _verify_ref(view,'prepared_view_ref')
                self.assertEqual(batch['batch_ref'],digest({k:v for k,v in batch.items() if k not in ('batch_ref','content_digest')}))
                for desc in view['core_results']:
                    self.assertEqual(file_digest(desc['path']),desc['file_digest'])
                    wrapper=_read(desc['path']); _verify_ref(wrapper,'core_result_artifact_ref')
                    candidates=[]
                    for partition in view['partitions']:
                        if partition['table']!='training_normalized_labels': continue
                        metadata=fixture_native(partition['metadata'],'metadata_ref')
                        if wrapper['result']['metadata']['result_ref'] not in metadata['core_result_refs']: continue
                        for content in metadata['contents'].values():
                            if type(content) is dict and set(content)=={'input','buffers'}:
                                candidates.append(content)
                            elif type(content) is dict and set(content)=={'path','file_digest','core_input_artifact_ref'}:
                                child=_read(content['path']); _verify_ref(child,'core_input_artifact_ref')
                                candidates.append({k:child[k] for k in ('input','buffers')})
                    self.assertTrue(candidates)
                    with VerifiedMatrixStore() as store:
                        validate_saved_core_result(wrapper,store,core_input=candidates[0])
                for partition in view['partitions']:
                    _verify_ref(partition,'partition_ref')
                    for desc in partition['buffers'].values(): self.assertEqual(file_digest(desc['path']),desc['buffer_digest'])
                    if partition['table']!='training_normalized_labels': continue
                    metadata=fixture_native(partition['metadata'],'metadata_ref'); raw=_read(metadata['raw_build']['path'])
                    days=sorted({r['feature_session'] for r in raw['rows']})
                    feature_rows=[r for day in days for r in feature.day(day)[0]]
                    original=_feature_wire(feature_rows,[feature.day(d)[1] for d in days],feature.spec,feature.view)
                    old=normalize_forward_labels(raw,features=original,cutoff=metadata['cutoff'])
                    for actual,expected in zip(metadata['rows'],old['rows']):
                        self.assertEqual((actual['security_id'],actual['feature_session']),
                            (expected['security_id'],expected['feature_session']))
                        self.assertEqual(actual['normalized_return'],expected['normalized_target'])
                        self.assertEqual(actual['normalized_valid'],expected['valid'])
                        self.assertEqual(actual['normalization_reason'],expected['invalid_reason'])
                        if actual['normalized_available_at'] is None: self.assertIsNone(expected['normalized_available_at'])
                        else: self.assertEqual(_instant(actual['normalized_available_at']),_instant(expected['normalized_available_at']))
                children={content['path'] for partition in view['partitions']
                    if partition['table']=='training_normalized_labels'
                    for content in fixture_native(partition['metadata'],'metadata_ref')['contents'].values()
                    if type(content) is dict and set(content)=={'path','file_digest','core_input_artifact_ref'}}
                self.assertEqual(len(children),len(folds))
                self.assertEqual(metrics['maximum_completed_fit_working_bytes'],0)
                hits={}
                with patch.object(data,'read',side_effect=AssertionError('HIT read Data')), \
                     patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=AssertionError('HIT ran Core')):
                    again=prepare_stock_ml_batch_inputs(data,feature_inputs=saved,fold_specs=folds,
                        destination=root/'prepared',preparation_options=options,metrics=hits)
                self.assertEqual(again,batch); self.assertEqual(hits['data_read_calls'],0)
                self.assertEqual(hits['core_calls'],0)
                with patch.object(data,'read',side_effect=AssertionError('saved projection read Data')), \
                     patch('axiom_engine.core.execute_cs_zscore_batch',side_effect=AssertionError('saved projection ran Core')):
                    with load_stock_ml_batch_inputs(batch) as verified:
                        before=verified.metrics
                        self.assertEqual(before['common_key_index_builds'],1)
                        for fold in batch['folds']:
                            with verified._matrix_project(fold['input_manifest'],fold['fold_spec']) as projection:
                                self.assertEqual(projection.X.shape[1],6)
                                self.assertEqual(projection.X.shape[0],projection.y.shape[0])
                                self.assertGreater(projection.X.shape[0],0)
                                selectors=fold['input_manifest']['selectors']
                                train=_read(selectors['training']['path'])
                                self.assertEqual(train['keys_digest'],digest(projection.training_keys))
                                self.assertEqual(train['payload'],_read(selectors['training_labels']['path'])['payload'])
                                infer=_read(selectors['inference']['path'])
                                self.assertEqual(infer['keys_digest'],digest([[r['security_id'],r['session']]
                                    for r in projection.feature_rows]))
                                self.assertEqual(infer['payload'],_read(selectors['evaluation_labels']['path'])['payload'])
                                with self.assertRaisesRegex(ValueError,'backing still borrowed'): verified.close()
                        after=verified.metrics
                        self.assertEqual(after['file_hash_calls'],before['file_hash_calls'])
                        self.assertEqual(after['json_decode_calls'],before['json_decode_calls'])
                        self.assertEqual(after['common_key_index_builds'],1)
            for q in queries:
                self.assertEqual(q['symbols'],feature.universe); self.assertEqual(q['purpose'],'label_outcomes')
                cutoffs=set(q['cutoff_by_session'].values()); self.assertEqual(len(cutoffs),1)
                self.assertIn(next(iter(cutoffs)),{f['fit_cutoff'] for f in folds}|{folds[-1]['evaluation_cutoff']})
                self.assertEqual(q['price_basis'],'unadjusted')
            # Each working group visits related cutoffs using a fixed union.
            self.assertGreater(metrics['date_working_groups'],1)
            self.assertEqual(metrics['data_read_calls'],len(queries)); saved.close()


class PublicMemoryFeatureFixture(PrepareFeatureFixture):
    """Saved synthetic Features; targets use the unmodified public Data reader.

    This fixture deliberately does not validate the Feature producer. Its saved
    rows provide a complete cohort and exact clocks for Label preparation.
    """
    def __init__(self, root):
        from axiom_research.stock_artifacts import write_json
        super().__init__(root)
        self.pit = 'operational_pit_v1'
        self.spec['pit_policy'] = self.pit
        scope = _read(root/'scope.json')
        scope.pop('scope_bundle_ref')
        for name in ('request', 'result'):
            scope[name]['pit_policy'] = self.pit
            scope[name+'_ref'] = digest(scope[name])
        scope = seal(scope, 'scope_bundle_ref')
        write_json(root/'scope.json', scope)
        self.spec['scope'].update(file_digest=file_digest(root/'scope.json'),
                                  scope_bundle_ref=scope['scope_bundle_ref'])
        body = deepcopy(self.manifest)
        body.pop('view_id')
        for query in [*body['queries'], body['universe_query']]:
            query['pit_policy'] = self.pit
        self.manifest = {**body, 'view_id': digest(body)[7:]}
        self.view = {k: deepcopy(v) for k, v in self.manifest.items() if k != 'files'}
        fit_position = self.calendar.index(self.folds()[0]['fit_session'])
        self.invalid_feature_session = self.calendar[fit_position-22]
        self.revised_endpoint = self.calendar[fit_position-15]
        self.revised_feature_session = self.calendar[fit_position-20]

    def day(self, day):
        rows, proof = super().day(day)
        proof['core_plan']['reference_members'] = {day: list(self.universe)}
        for number, row in enumerate(rows):
            row['member'] = True
            row['availability'] = [day+'T20:00:00.'+str(123451+number).zfill(6)+'+08:00']*6
            if day == self.invalid_feature_session and row['security_id'] == 'B':
                row['values'][0] = None
                row['validity'][0] = False
                row['reasons'][0] = ['REFERENCE_MISSING', 'synthetic original Feature reason']
            row['source_refs'] = [proof['core_frame_ref'], digest(proof['core_plan'])]
        return rows, proof


class PublicMemoryStore:
    """Test-only Store protocol, matching Data's own memory-reader fixtures.

    Neither this protocol fixture nor its Arrow tables are a new Data API.
    No physical fact root, supplier, or snapshot builder is involved.
    """
    def __init__(self, feature):
        from datetime import timedelta
        self.snapshot = feature.snapshot
        self.calls = []
        self.partitions = {'market_daily': {}, 'adjustment_factors': {}}
        for day_index, day in enumerate(feature.calendar):
            for number, security in enumerate(feature.universe):
                for domain in self.partitions:
                    row = {'security_id': security, 'session': day,
                        'revision_id': domain+'-'+security+'-'+day+'-original',
                        'revision_sequence': 1, 'first_observed_at': day+'T08:15:00.123456Z',
                        'raw_batch_id': 'memory-'+domain+'-'+day,
                        'source_available_at': None, 'evidence_ref': None}
                    if domain == 'market_daily':
                        row.update(open=10.0+number*.5+day_index*.05,
                                   close=12.0+number*number*.8+day_index*.05)
                    else:
                        row['factor'] = 1.0+(day_index % 7)*.15+number*.05
                    self.partitions[domain].setdefault(day[:7], []).append(row)
        # One microsecond after fit 1: an actual old-key revision is invisible
        # at that fit and visible at the later two cutoffs.
        late = (_instant(feature.folds()[0]['fit_cutoff'])+timedelta(microseconds=1))
        self.revision_available_at = late.isoformat().replace('+00:00', 'Z')
        for domain in self.partitions:
            rows = self.partitions[domain][feature.revised_endpoint[:7]]
            original = next(r for r in rows if r['security_id'] == 'B' and
                            r['session'] == feature.revised_endpoint)
            revised = deepcopy(original)
            revised.update(revision_id='late-'+domain, revision_sequence=2,
                           first_observed_at=self.revision_available_at,
                           raw_batch_id='memory-late-'+domain)
            if domain == 'market_daily': revised['close'] += 3.25
            else: revised['factor'] *= 1.125
            rows.append(revised)
        self.manifest = {'snapshot_id': self.snapshot, 'domains': {}}
        for domain, fields in [('market_daily', {'open': {'dtype': 'float64', 'unit': 'CNY/share'},
                    'close': {'dtype': 'float64', 'unit': 'CNY/share'}}),
                ('adjustment_factors', {'factor': {'dtype': 'float64', 'unit': 'dimensionless'}})]:
            self.manifest['domains'][domain] = {
                'contract': {'contract_id': domain+'_memory_v1',
                             'logical_key': ['security_id', 'session'], 'fields': fields},
                'source_profile': {'id': 'public_reader_memory_fixture', 'availability': {
                    'timezone': 'Asia/Shanghai', 'session_release_time': '17:00:00'}},
                'partitions': [{'partition': month, 'domain': domain}
                               for month in sorted(self.partitions[domain])], 'coverage': {}}

    def load_snapshot(self, snapshot_id):
        if snapshot_id != self.snapshot: raise AssertionError('fixed snapshot changed')
        return deepcopy(self.manifest)

    def verify_partition(self, part):
        self.calls.append(('verify', part['domain'], part['partition']))

    def read_partition(self, part, *, columns=None, symbols=None, sessions=None):
        import pyarrow as pa
        self.calls.append(('read', part['domain'], part['partition'], tuple(columns),
                           tuple(symbols), sessions))
        rows = [row for row in self.partitions[part['domain']][part['partition']]
                if row['security_id'] in symbols and (sessions is None or row['session'] in sessions)]
        return pa.Table.from_pylist([{column: row.get(column) for column in columns} for row in rows])


class PublicMemoryMatrixPrepareTests(unittest.TestCase):
    def _adjusted(self, data, feature, query_wire):
        from dataclasses import replace
        from axiom_data import QuerySpec, adjust_prices
        # Reader's saved query omits domain because its context owns it.
        query = QuerySpec(**{'domain': 'market_daily', **query_wire})
        anchor = max(query.sessions)
        price = data.read(snapshot=feature.snapshot, query=query)
        factors = data.read(snapshot=feature.snapshot,
                            query=replace(query, domain='adjustment_factors', fields=('factor',)))
        return adjust_prices(price, factors, fields=('open', 'close'),
                             anchor_session=anchor, decision_session=anchor, factor_field='factor')

    def test_three_fit_public_reader_adjustment_and_old_path_goldens(self):
        from axiom_data import Data, QuerySpec
        from axiom_research.labels import build_forward_labels
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature = PublicMemoryFeatureFixture(root)
            saved = feature.matrix()
            try:
                physical_root = root/'no-physical-fact-root'
                data = Data(physical_root, cache_bytes=0)
                store = PublicMemoryStore(feature)
                data.store = store
                immutable_facts = digest({'manifest': store.manifest, 'partitions': store.partitions})
                folds = feature.folds()[:3]
                metrics = {}
                options = {'row_block_sessions': 24, 'column_block': 32,
                    'maximum_resident_bytes': 64*1024**2, 'normalization_backend': 'core_cs_batch_v1'}
                with patch('axiom_research.stock_ml._implementation', return_value=feature.implementation), \
                     patch('axiom_research.stock_ml._environment', return_value=feature.environment), \
                     patch.object(data, 'read', wraps=data.read) as public_reads, \
                     patch('axiom_engine.core.execute_cs_zscore_batch', wraps=execute_cs_zscore_batch) as core, \
                     patch('axiom_research.stock_ml._iter_stock_feature_days', side_effect=AssertionError('Feature executed')), \
                     patch('axiom_research.stock_training.fit_predict_stock_model', side_effect=AssertionError('model executed')):
                    batch = prepare_stock_ml_batch_inputs(data, feature_inputs=saved, fold_specs=folds,
                        destination=root/'prepared-public', preparation_options=options, metrics=metrics)
                    self.assertEqual(core.call_count, 3)
                    self.assertEqual(public_reads.call_count, metrics['data_read_calls'])
                    preparation_queries = [call.kwargs['query'] for call in public_reads.call_args_list]
                self.assertEqual(metrics['core_calls'], 3)
                for name in ('supplier_calls', 'feature_core_calls', 'train_calls', 'predict_calls', 'account_calls'):
                    self.assertEqual(metrics[name], 0)
                self.assertFalse(physical_root.exists())
                view = _read(batch['prepared_view']['path'])
                _verify_ref(view, 'prepared_view_ref')
                def original_wire(descriptor, ref_key):
                    # Physical native carriers preserve the original logical
                    # document. Restore its exact saved coverage for this
                    # independent oracle; production admission remains bounded.
                    self.assertEqual(file_digest(descriptor['path']), descriptor['file_digest'])
                    wire = _read(descriptor['path'])
                    if wire.get('contract_version') in ('stock_native_json_carrier_v1','stock_native_json_carrier_v2'):
                        _verify_ref(wire, 'carrier_ref')
                        original = deepcopy(wire['skeleton'])
                        for slot in wire['coverage_slots']+wire.get('wire_slots',[]):
                            blob = slot['bytes']
                            self.assertEqual(file_digest(blob['path']), blob['buffer_digest'])
                            self.assertEqual(Path(blob['path']).stat().st_size, blob['shape'][0])
                            parent = original
                            for key in slot['path'][:-1]: parent = parent[key]
                            parent[slot['path'][-1]] = _read(blob['path'])
                        wire = original
                    _verify_ref(wire, ref_key)
                    self.assertEqual(wire[ref_key], descriptor[ref_key])
                    return wire
                raw_rows, normalized_rows = {}, {}
                for part in view['partitions']:
                    if part['table'] not in ('training_raw_labels', 'training_normalized_labels'):
                        continue
                    metadata = original_wire(part['metadata'], 'metadata_ref')
                    raw = original_wire(metadata['raw_build'], 'label_ref')
                    ref = part['fold_spec_ref']
                    target = raw_rows if part['table'] == 'training_raw_labels' else normalized_rows
                    indexed = target.setdefault(ref, {})
                    for row in metadata['rows']:
                        key = row['security_id'], row['feature_session']
                        self.assertNotIn(key, indexed)
                        indexed[key] = row
                    if part['table'] != 'training_raw_labels': continue
                    context = raw['source_evidence']['context']
                    # Independently replay the precise saved public queries,
                    # not a reconstructed or expanded whole-window batch.
                    actual = self._adjusted(data, feature, context['derivation']['price_query'])
                    wire = actual.to_json()
                    expected_raw = build_forward_labels(actual, calendar=feature.calendar,
                        feature_sessions=sorted({r['feature_session'] for r in raw['rows']}))
                    self.assertEqual(raw, expected_raw)
                    selected = {key: wire[key] for key in ('context', 'records', 'field_meta')}
                    self.assertEqual(metadata['contents'][digest(selected)], selected)
                    self.assertEqual(raw['source_ref'], digest(selected))
                    self.assertEqual(raw['source_evidence']['records_ref'], digest(wire['records']))
                    self.assertEqual(raw['source_evidence']['field_meta_ref'], digest(wire['field_meta']))
                    self.assertEqual(metadata['contents'][digest(wire['context']['query'])], wire['context']['query'])
                    cohort = next(v for v in metadata['contents'].values() if isinstance(v, dict) and
                                  v.get('contract_version') == 'stock_matrix_label_cohort_v1')
                    self.assertEqual(cohort['keys'], [[r['security_id'], r['feature_session']] for r in raw['rows']])
                    self.assertEqual(len(cohort['keys']), len(cohort['sessions'])*3)
                oracle_raw, oracle_normalized, oracle_adjusted = {}, {}, {}
                for fold in folds:
                    ref = digest(fold)
                    days = sorted({key[1] for key in raw_rows[ref]})
                    allowed = [day for day in feature.calendar if day <= _instant(fold['fit_cutoff']).date().isoformat()]
                    query = QuerySpec('market_daily', ('open', 'close'), tuple(feature.universe), tuple(allowed),
                        feature.pit, {day: fold['fit_cutoff'] for day in allowed}, purpose='label_outcomes')
                    # The previous path reads its full native window; its
                    # actual source identity intentionally differs from blocks.
                    query_wire = {'domain': query.domain, 'fields': query.fields, 'symbols': query.symbols,
                        'sessions': query.sessions, 'pit_policy': query.pit_policy,
                        'cutoff_by_session': dict(query.cutoff_by_session), 'purpose': query.purpose}
                    adjusted = self._adjusted(data, feature, query_wire)
                    raw = build_forward_labels(adjusted, calendar=feature.calendar, feature_sessions=days)
                    old_features = _feature_wire([r for day in days for r in feature.day(day)[0]],
                        [feature.day(day)[1] for day in days], feature.spec, feature.view)
                    normalized = normalize_forward_labels(raw, features=old_features, cutoff=fold['fit_cutoff'])
                    oracle_raw[ref] = {(r['security_id'], r['feature_session']): r for r in raw['rows']}
                    oracle_normalized[ref] = {(r['security_id'], r['feature_session']): r for r in normalized['rows']}
                    oracle_adjusted[ref] = adjusted.to_json()
                    self.assertEqual(set(raw_rows[ref]), set(oracle_raw[ref]))
                    self.assertEqual(set(normalized_rows[ref]), set(oracle_normalized[ref]))
                    for key, actual in raw_rows[ref].items():
                        expected = oracle_raw[ref][key]
                        for field in ('return', 'valid', 'invalid_reason', 'start_session', 'end_session', 'label_available_at'):
                            self.assertEqual(actual[field], expected[field], (fold['fit_session'], key, field))
                    for key, actual in normalized_rows[ref].items():
                        expected = oracle_normalized[ref][key]
                        for actual_field, old_field in [('normalized_return', 'normalized_target'),
                                ('normalized_valid', 'valid'), ('normalization_reason', 'invalid_reason')]:
                            self.assertEqual(actual[actual_field], expected[old_field], (fold['fit_session'], key, old_field))
                        if actual['normalized_available_at'] is None:
                            self.assertIsNone(expected['normalized_available_at'])
                        else:
                            self.assertEqual(_instant(actual['normalized_available_at']), _instant(expected['normalized_available_at']))
                    bad_key = 'B', feature.invalid_feature_session
                    self.assertTrue(raw_rows[ref][bad_key]['valid'])
                    self.assertFalse(normalized_rows[ref][bad_key]['normalized_valid'])
                    self.assertEqual(normalized_rows[ref][bad_key]['normalization_reason'], 'FEATURE_MISSING')
                    for security in ('A', 'C'):
                        self.assertTrue(normalized_rows[ref][security, feature.invalid_feature_session]['normalized_valid'])
                    mature = 'A', feature.revised_feature_session
                    self.assertTrue(normalized_rows[ref][mature]['normalized_valid'])
                    self.assertEqual(_instant(normalized_rows[ref][mature]['normalized_available_at']), _instant(fold['fit_cutoff']))
                first, second, third = [digest(f) for f in folds]
                def meta(ref, field, security, day):
                    return next(r for r in oracle_adjusted[ref]['field_meta'][field]['by_key']
                                if (r['security_id'], r['session']) == (security, day))
                def price(ref, security, day):
                    return next(r for r in oracle_adjusted[ref]['records']
                                if (r['security_id'], r['session']) == (security, day))
                for field, domain in [('price_provenance', 'market_daily'), ('factor_provenance', 'adjustment_factors')]:
                    before = meta(first, 'close', 'B', feature.revised_endpoint)[field]
                    self.assertNotEqual(before['revision_id'], 'late-'+domain)
                    for ref in (second, third):
                        after = meta(ref, 'close', 'B', feature.revised_endpoint)[field]
                        self.assertEqual(after['revision_id'], 'late-'+domain)
                        self.assertEqual(_instant(after['usable_from']), _instant(store.revision_available_at))
                revised_key = 'B', feature.revised_feature_session
                self.assertNotEqual(raw_rows[first][revised_key]['return'], raw_rows[second][revised_key]['return'])
                # A has no revision. Its old adjusted price still changes with
                # the explicit fit anchor, and that anchor supplies availability.
                historical = feature.revised_endpoint
                self.assertNotEqual(price(first, 'A', historical)['close'], price(second, 'A', historical)['close'])
                day_index = feature.calendar.index(historical)
                native_close = 12.0+day_index*.05
                historical_factor = 1.0+(day_index % 7)*.15
                anchors = []
                for ref, fold in zip((first, second, third), folds):
                    anchor_index = feature.calendar.index(fold['fit_session'])
                    anchor_factor = 1.0+(anchor_index % 7)*.15
                    self.assertEqual(price(ref, 'A', historical)['close'],
                                     native_close*historical_factor/anchor_factor)
                    provenance = meta(ref, 'close', 'A', historical)['anchor_factor_provenance']
                    anchors.append(provenance['session'])
                    self.assertEqual(provenance['session'], fold['fit_session'])
                    self.assertEqual(_instant(raw_rows[ref]['A', feature.revised_feature_session]['label_available_at']),
                                     _instant(provenance['usable_from']))
                    self.assertEqual(provenance['availability_basis'], 'first_observed_at')
                self.assertEqual(len(set(anchors)), 3)
                self.assertTrue(any(price(ref, 'A', historical)['close'] != native_close
                                    for ref in (first, second, third)))
                self.assertNotEqual(raw_rows[first]['A', feature.revised_feature_session]['label_available_at'],
                                    raw_rows[second]['A', feature.revised_feature_session]['label_available_at'])
                original_clock = feature.day(feature.revised_feature_session)[0][2]['availability'][0]
                offset = feature.days.index(feature.revised_feature_session)*3+2
                self.assertEqual(saved.row_metadata([offset])[0]['availability'][0], original_clock)
                self.assertEqual(_instant(original_clock).microsecond, 123453)
                for position, query in enumerate(preparation_queries):
                    self.assertIsInstance(query, QuerySpec)
                    self.assertEqual(query.symbols, tuple(feature.universe))
                    self.assertEqual(query.pit_policy, 'operational_pit_v1')
                    self.assertEqual(query.purpose, 'label_outcomes')
                    self.assertEqual(query.price_basis, 'unadjusted')
                    self.assertEqual(len(set(query.cutoff_by_session.values())), 1)
                    if query.domain == 'adjustment_factors':
                        previous = [p for p in preparation_queries[:position] if p.domain == 'market_daily'
                                    and p.sessions == query.sessions and p.cutoff_by_session == query.cutoff_by_session]
                        self.assertTrue(previous, 'factor query precedes its corresponding price query')
                hits = {}
                with patch('axiom_research.stock_ml._implementation', return_value=feature.implementation), \
                     patch('axiom_research.stock_ml._environment', return_value=feature.environment), \
                     patch.object(data, 'read', side_effect=AssertionError('HIT read Data')), \
                     patch('axiom_engine.core.execute_cs_zscore_batch', side_effect=AssertionError('HIT ran Core')):
                    again = prepare_stock_ml_batch_inputs(data, feature_inputs=saved, fold_specs=folds,
                        destination=root/'prepared-public', preparation_options=options, metrics=hits)
                self.assertEqual(again, batch)
                self.assertEqual(hits['data_read_calls'], 0)
                self.assertEqual(hits['core_calls'], 0)
                self.assertEqual(immutable_facts, digest({'manifest': store.manifest, 'partitions': store.partitions}))
                self.assertFalse(physical_root.exists())
            finally:
                saved.close()


if __name__=='__main__': unittest.main()
