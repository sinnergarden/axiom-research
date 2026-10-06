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
                        metadata=_read(partition['metadata']['path'])
                        if wrapper['result']['metadata']['result_ref'] not in metadata['core_result_refs']: continue
                        candidates.extend(content for content in metadata['contents'].values()
                            if type(content) is dict and set(content)=={'input','buffers'})
                    self.assertTrue(candidates)
                    with VerifiedMatrixStore() as store:
                        validate_saved_core_result(wrapper,store,core_input=candidates[0])
                for partition in view['partitions']:
                    _verify_ref(partition,'partition_ref')
                    for desc in partition['buffers'].values(): self.assertEqual(file_digest(desc['path']),desc['buffer_digest'])
                    if partition['table']!='training_normalized_labels': continue
                    metadata=_read(partition['metadata']['path']); raw=_read(metadata['raw_build']['path'])
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


if __name__=='__main__': unittest.main()
