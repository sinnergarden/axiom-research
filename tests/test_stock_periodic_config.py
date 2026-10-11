"""Periodic compiler checks; no Feature/Data/model/account execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from axiom_research import compile_stock_periodic_folds, stock_evaluation_sessions
from axiom_research.stock_fold_inputs import validate_spec
from axiom_research.stock_periodic_config import nominal_mature_sessions
from test_stock_weekly_config import artificial_calendar


class PeriodicConfigTests(unittest.TestCase):
    def arguments(self):
        calendar = artificial_calendar('2019-01-01', '2024-03-01')
        return dict(calendar=calendar, feature_sessions=calendar,
            trade_start='2023-01-03', trade_end_exclusive='2023-12-30',
            mature_sessions=504, label_horizon_sessions=40,
            retrain_every_sessions=20, timezone='Asia/Shanghai',
            fit_time='20:30', model_time='20:45', inference_time='21:00',
            evaluation_cutoff='2024-03-01T20:30:00+08:00')

    def test_daily_fresh_predictions_and_twenty_session_fits(self):
        args = self.arguments(); folds = compile_stock_periodic_folds(**args)
        trades = [d for d in args['calendar'] if '2023-01-03' <= d < '2023-12-30']
        self.assertEqual([d for fold in folds for d in fold['oos_trade_sessions']], trades)
        self.assertEqual([len(f['oos_trade_sessions']) for f in folds[:-1]], [20] * (len(folds)-1))
        for fold in folds:
            training, predictions = validate_spec(fold, args['calendar'])
            self.assertEqual(len(training), 543)
            self.assertEqual(len(nominal_mature_sessions(fold, args['calendar'], 40)), 504)
            self.assertEqual(predictions, [args['calendar'][args['calendar'].index(d)-1]
                for d in fold['oos_trade_sessions']])

    def test_evaluation_frequency_selects_existing_dates(self):
        calendar = ['2022-12-30','2023-01-03','2023-01-04','2023-01-05',
                    '2023-01-06','2023-01-09','2023-01-10']
        trades = calendar[1:]
        self.assertEqual(stock_evaluation_sessions(calendar=calendar, trade_sessions=trades,
            frequency='daily'), calendar[:-1])
        self.assertEqual(stock_evaluation_sessions(calendar=calendar, trade_sessions=trades,
            frequency='weekly'), ['2022-12-30','2023-01-06'])
        for changed in (dict(frequency='monthly'), dict(trade_sessions=['2023-01-02'])):
            with self.assertRaises(ValueError):
                stock_evaluation_sessions(**{**dict(calendar=calendar, trade_sessions=trades,
                    frequency='weekly'), **changed})

    def test_scope_counts_clocks_and_history_reject(self):
        args = self.arguments()
        for changed in (dict(mature_sessions=True), dict(label_horizon_sessions=0),
                dict(retrain_every_sessions=0), dict(model_time='21:30'),
                dict(timezone='Invalid/Zone'), dict(feature_sessions=args['calendar'][-10:]),
                dict(trade_end_exclusive='2025-01-01')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                compile_stock_periodic_folds(**{**deepcopy(args), **changed})

    def test_readable_transport_rejects_different_label_horizon(self):
        import yaml
        from axiom_research import load_stock_sequential_configuration, to_dict
        from axiom_research.stock_artifacts import file_digest, write_json
        from test_stock_target_config_blocks import label, training
        args = self.arguments()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            axes = dict(contract_version='stock_feature_inputs_spec_v1',calendar=args['calendar'],
                feature_sessions=args['feature_sessions'],universe=['A','B'],snapshot='s_synthetic',
                pit_policy='best_effort_vendor_v1',scope={},catalog_ref='sha256:'+'0'*64,
                feature_selection=[],ordered_features=[],read_sessions=args['calendar'],
                cutoff_by_session={d:d+'T20:30:00+08:00' for d in args['calendar']})
            write_json(root/'axes.json', axes)
            target = to_dict(label()); target.update(horizon_sessions=40,return_end_offset_sessions=40)
            target['maturity']['lag_sessions'] = 40
            specs = dict(contract_version='stock_experiment_config_v1',specs=dict(label=target,
                model=to_dict(training())))
            rules = {k:v for k,v in args.items() if k not in ('calendar','feature_sessions')}
            dataset = dict(contract_version='stock_dataset_schedule_v3',
                axes_input=dict(path='axes.json',file_digest=file_digest(root/'axes.json')),
                periodic_schedule=rules,preparation_options=dict(row_block_sessions=1,column_block=32,
                    maximum_resident_bytes=64*1024**2,normalization_backend='core_cs_batch_v1'),
                model_feature_selection=None,signal_contexts=None)
            config = dict(contract_version='stock_sequential_configuration_v1',
                specification_files=['specs.yaml'],dataset_file='dataset.yaml')
            for name,wire in (('specs.yaml',specs),('dataset.yaml',dataset),('experiment.yaml',config)):
                (root/name).write_text(yaml.safe_dump(wire))
            saved = load_stock_sequential_configuration(root/'experiment.yaml')
            self.assertEqual(saved['dataset']['fold_specs'],compile_stock_periodic_folds(**args))
            self.assertEqual(set(len(v) for v in saved['periodic_compilation'][
                'nominal_mature_sessions_by_fit'].values()), {504})
            dataset['periodic_schedule']['label_horizon_sessions'] = 20
            (root/'dataset.yaml').write_text(yaml.safe_dump(dataset))
            with self.assertRaisesRegex(ValueError,'training Label horizon differ'):
                load_stock_sequential_configuration(root/'experiment.yaml')


if __name__ == '__main__': unittest.main()
