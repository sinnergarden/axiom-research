"""Metric forwarding only; no Data, Feature, LightGBM or account execution."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch


class StockEntryProgressTests(TestCase):
    def test_one_owner_forwards_distinct_existing_metrics_without_replaying_folds(self):
        from axiom_research.stock_experiment_entry import _build_stock_sequential_experiment
        items=[{'input_manifest':{'input':i},'fold_spec':{'fold':i}} for i in range(3)]
        owner=SimpleNamespace(batch=object(),operation=nullcontext,
            next_fold=Mock(side_effect=[*items,None]),finish=Mock(return_value={'batch':'saved'}))
        writer=SimpleNamespace(finish=Mock(return_value={'oos':'saved'}))
        events=[]

        def build(inputs, *, metrics, **kwargs):
            i=inputs['input']
            metrics.update(train_seconds=i+1,predict_seconds=0.25,total_seconds=i+2)
            return SimpleNamespace(path=Path('/synthetic/fold')/str(i),identity='saved-fold-'+str(i))

        with patch('axiom_research.stock_sequential_api.open_stock_ml_batch_preparation',
                return_value=nullcontext(owner)) as preparation, \
             patch('axiom_research.stock_sequential_api.open_stock_signal_evaluation_freeze',
                return_value=nullcontext(writer)) as freezing, \
             patch('axiom_research.stock_folds.build_stock_ml_fold_from_saved_inputs',
                side_effect=build) as builder, \
             patch('axiom_research.stock_derived_signal._parent',
                return_value=({'signal_run_ref':'saved-signal'},{'prediction':'saved'},None)), \
             patch('axiom_research.stock_experiment_entry.time.perf_counter',
                side_effect=[10,12,20,22,30,32,40]):
            result=_build_stock_sequential_experiment(None,
                configuration={'specs':{'label':object(),'model':object()},
                    'configuration_ref':'config','spec_refs':{}},
                feature_inputs=object(),fold_specs=[v['fold_spec'] for v in items],
                column_source=object(),preparation_options={},scope={},destination='/synthetic',
                progress=events.append)

        self.assertEqual((preparation.call_count,freezing.call_count,builder.call_count),(1,1,3))
        self.assertEqual([e['completed'] for e in events],[1,2,3])
        self.assertEqual([e['preparation_seconds'] for e in events],[2,2,2])
        self.assertEqual([e['fold_metrics']['train_seconds'] for e in events],[1,2,3])
        self.assertEqual(len({id(c.kwargs['metrics']) for c in builder.call_args_list}),3)
        self.assertEqual(result['prediction_bindings'],[{'prediction':'saved'}]*3)
