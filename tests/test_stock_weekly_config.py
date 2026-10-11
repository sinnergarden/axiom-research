"""Pure schedule/compiler checks; no source preparation or model execution."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import compile_stock_weekly_folds, load_stock_sequential_configuration, to_dict
from axiom_research.stock_artifacts import file_digest, write_json
from axiom_research.stock_fold_inputs import validate_spec
from test_stock_target_config_blocks import label, training


def artificial_calendar(first, last):
    day=date.fromisoformat(first); end=date.fromisoformat(last); out=[]
    while day<=end:
        if day.weekday()<5:out.append(day.isoformat())
        day+=timedelta(days=1)
    return out


class WeeklyConfigTests(unittest.TestCase):
    def arguments(self):
        calendar=artificial_calendar('2020-11-01','2021-02-05')
        return {'calendar':calendar,'feature_sessions':calendar,'trade_start':'2020-12-30',
            'trade_end_exclusive':'2021-01-08','training_window':{'unit':'feature_sessions','length':8,
                'end':'previous_fit_session'},'timezone':'Asia/Shanghai','fit_time':'20:30:00',
            'model_time':'20:45:00','inference_time':'21:00:00','evaluation_cutoff':'2021-02-05T20:30:00+08:00'}

    def test_partial_weeks_iso_year_and_strict_previous_sessions(self):
        args=self.arguments();folds=compile_stock_weekly_folds(**args)
        self.assertEqual([f['oos_trade_sessions'] for f in folds],
            [['2020-12-30','2020-12-31','2021-01-01'],['2021-01-04','2021-01-05','2021-01-06','2021-01-07']])
        self.assertEqual([f['fit_session'] for f in folds],['2020-12-29','2021-01-01'])
        self.assertEqual(list(folds[0]['inference_cutoff_by_session']),['2020-12-29','2020-12-30','2020-12-31'])
        self.assertEqual(folds[0]['fit_cutoff'],'2020-12-29T20:30:00+08:00')
        self.assertEqual(folds[0]['evaluation_cutoff'],args['evaluation_cutoff'])
        for fold in folds:
            train,pred=validate_spec(fold,args['calendar'])
            self.assertEqual(len(train),8);self.assertEqual(train[-1],args['calendar'][args['calendar'].index(fold['fit_session'])-1])
        self.assertEqual(date.fromisoformat('2020-12-29').isocalendar()[:2],
            date.fromisoformat('2020-12-30').isocalendar()[:2])

    def test_calendar_year_leap_clamp_and_history_requirement(self):
        # The irregular artificial frozen calendar expressly includes leap day.
        calendar=['2018-02-27','2019-02-27','2019-02-28','2019-03-01','2020-02-28',
            '2020-02-29','2020-03-02','2020-03-03','2020-03-10']
        args={**self.arguments(),'calendar':calendar,'feature_sessions':calendar,'trade_start':'2020-03-02',
            'trade_end_exclusive':'2020-03-04','evaluation_cutoff':'2020-03-10T20:30:00+08:00',
            'training_window':{'unit':'calendar_years','length':1,'end':'previous_fit_session',
                'start':'fit_date_minus_years_inclusive','leap_day':'clamp_feb_28'}}
        folds=compile_stock_weekly_folds(**args)
        train,pred=validate_spec(folds[0],calendar)
        self.assertEqual(train,['2019-02-28','2019-03-01','2020-02-28'])
        self.assertEqual(folds[0]['fit_session'],'2020-02-29')
        with self.assertRaises(ValueError):compile_stock_weekly_folds(**{**args,'calendar':calendar[2:],
            'feature_sessions':calendar[2:]})

    def test_empty_missing_history_feature_coverage_and_bad_clocks_reject(self):
        args=self.arguments();cases=[]
        cases.append({**args,'trade_start':'2021-01-02','trade_end_exclusive':'2021-01-03'})
        cases.append({**args,'trade_start':'2021-01-08'})
        cases.append({**args,'trade_start':args['calendar'][0]})
        cases.append({**args,'feature_sessions':args['calendar'][40:]})
        cases.append({**args,'training_window':{**args['training_window'],'length':True}})
        cases.append({**args,'model_time':'21:30:00'})
        cases.append({**args,'fit_time':'20:30:00+08:00'})
        cases.append({**args,'evaluation_cutoff':'2021-01-01T20:30:00+08:00'})
        cases.append({**args,'evaluation_cutoff':'2021-02-05T20:30:00'})
        cases.append({**args,'timezone':'Invalid/Zone'})
        cases.append({**args,'trade_end_exclusive':'2025-01-01'})
        cases.append({**args,'trade_start':'2010-01-01'})
        for changed in cases:
            with self.subTest(changed=changed),self.assertRaises(ValueError):compile_stock_weekly_folds(**changed)

    def test_explicit_alternative_clocks_and_ambiguous_time_reject(self):
        args={**self.arguments(),'fit_time':'20:50','model_time':'21:00','inference_time':'21:15'}
        folds=compile_stock_weekly_folds(**args)
        self.assertTrue(folds[0]['fit_cutoff'].endswith('20:50:00+08:00'))
        self.assertTrue(folds[0]['simulated_model_available_at'].endswith('21:00:00+08:00'))
        from axiom_research.stock_weekly_config import _local_clock
        from zoneinfo import ZoneInfo
        for day,clock in [('2021-03-14','02:30'),('2021-11-07','01:30')]:
            with self.assertRaises(ValueError):_local_clock(day,clock,ZoneInfo('America/New_York'))

    def test_short_yaml_effective_v1_identity_comments_paths_and_pin_failure(self):
        import yaml
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.arguments()
            source={'contract_version':'stock_feature_inputs_spec_v1','calendar':args['calendar'],
                'feature_sessions':args['feature_sessions'],'universe':['A','B'],
                'snapshot':'s_synthetic','pit_policy':'best_effort_vendor_v1','scope':{},
                'catalog_ref':'sha256:'+'0'*64,'feature_selection':[],'ordered_features':[],
                'read_sessions':args['calendar'],
                'cutoff_by_session':{d:d+'T20:30:00+08:00' for d in args['calendar']}}
            axes=root/'axes.json';write_json(axes,source)
            specs={'contract_version':'stock_experiment_config_v1','specs':{'label':to_dict(label()),'model':to_dict(training())}}
            (root/'specs.yaml').write_text(yaml.safe_dump(specs))
            rules={k:v for k,v in args.items() if k not in ('calendar','feature_sessions')}
            short={'contract_version':'stock_dataset_schedule_v2','axes_input':{'path':'axes.json','file_digest':file_digest(axes)},
                'weekly_schedule':rules,'preparation_options':{'row_block_sessions':10,'column_block':32,
                    'maximum_resident_bytes':64*1024**2,'normalization_backend':'core_cs_batch_v1'},
                'model_feature_selection':None,'signal_contexts':None}
            dataset=root/'dataset.yaml';dataset.write_text(yaml.safe_dump(short))
            config=root/'experiment.yaml';config.write_text(yaml.safe_dump({'contract_version':'stock_sequential_configuration_v1',
                'specification_files':['specs.yaml'],'dataset_file':'dataset.yaml'}))
            result=load_stock_sequential_configuration(config)
            self.assertEqual(result['dataset']['contract_version'],'stock_dataset_schedule_v1')
            self.assertEqual(result['dataset']['fold_specs'],compile_stock_weekly_folds(**args))
            self.assertEqual(result['dataset']['scope']['universe'],['A','B'])
            dataset.write_text('# human comment\n'+yaml.safe_dump(short,sort_keys=False))
            self.assertEqual(load_stock_sequential_configuration(config)['configuration_ref'],result['configuration_ref'])
            copy=root/'same-axes.json';copy.write_bytes(axes.read_bytes());short['axes_input']['path']='same-axes.json'
            dataset.write_text(yaml.safe_dump(short))
            self.assertEqual(load_stock_sequential_configuration(config)['configuration_ref'],result['configuration_ref'])
            short['weekly_schedule']['inference_time']='21:15';dataset.write_text(yaml.safe_dump(short))
            self.assertNotEqual(load_stock_sequential_configuration(config)['configuration_ref'],result['configuration_ref'])
            short['weekly_schedule']['inference_time']='21:00';dataset.write_text(yaml.safe_dump(short))
            copy.write_bytes(copy.read_bytes()+b' ')
            with patch('yaml.load',side_effect=AssertionError('parser consumed unpinned axes')):
                from axiom_research.stock_experiment_config import _read_configuration_yaml
                with self.assertRaisesRegex(ValueError,'byte digest'):
                    _read_configuration_yaml(copy,expected_digest=short['axes_input']['file_digest'])
            with self.assertRaisesRegex(ValueError,'byte digest'):load_stock_sequential_configuration(config)


if __name__=='__main__':unittest.main()
