"""Synthetic saved-input sharing and calendar-year fold boundaries only."""
from copy import copy, deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from axiom_research import stock_fold_inputs
from axiom_research.stock_artifacts import digest
from axiom_research.stock_fold_inputs import (common_source_identity,
    load_saved_feature_inputs, project_saved_fold, validate_spec)
from test_stock_folds import fixture


TWO_YEAR_WINDOW = {'unit': 'calendar_years', 'length': 2,
    'end': 'previous_fit_session', 'start': 'fit_date_minus_years_inclusive',
    'leap_day': 'clamp_feb_28'}


def fold_spec(calendar, fit, trades, *, version='stock_ml_fold_spec_v2'):
    prediction = [calendar[calendar.index(day)-1] for day in trades]
    return {'contract_version': version,
        'training_window': deepcopy(TWO_YEAR_WINDOW) if version == 'stock_ml_fold_spec_v2' else
            {'unit': 'feature_sessions', 'length': 65, 'end': 'previous_fit_session'},
        'fit_session': fit, 'fit_cutoff': fit+'T20:30:00+08:00',
        'simulated_model_available_at': fit+'T20:45:00+08:00',
        'oos_trade_sessions': trades,
        'inference_cutoff_by_session': {day: day+'T21:00:00+08:00' for day in prediction},
        'evaluation_cutoff': calendar[-1]+'T22:00:00+08:00'}


class SharedFeatureInputTests(unittest.TestCase):
    def test_v1_shared_and_standalone_have_identical_six_values(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, spec = fixture(Path(temp), fit_index=69, label_start=0,
                                     labels_include_immature=False)
            expected = project_saved_fold(manifest, spec)
            shared = load_saved_feature_inputs(manifest)
            actual = project_saved_fold(manifest, spec, feature_inputs=shared)
            self.assertEqual(len(actual), 6)
            self.assertEqual(actual, expected)
            self.assertEqual(actual[0]['training_sessions'], manifest['calendar'][4:69])
            self.assertIn(('A', manifest['calendar'][0]), shared.rows)
            self.assertEqual(actual[3]['LABEL_NOT_MATURE'], 12)

    def test_common_identity_excludes_both_label_definitions(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _ = fixture(Path(temp))
            changed = deepcopy(manifest)
            changed['training_labels'] = [{'raw': {'path': '/tmp/another-raw.json'},
                'normalized': {'path': '/tmp/another-normalized.json'},
                'sessions': ['2023-09-01'], 'raw_projection': False}]
            changed['evaluation_labels'] = {'path': '/tmp/another-evaluation.json'}
            self.assertEqual(common_source_identity(changed), common_source_identity(manifest))

    def test_every_common_definition_change_rejects_shared_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, spec = fixture(Path(temp))
            shared = load_saved_feature_inputs(manifest)
            changes = {
                'scope': {**manifest['scope'], 'file_digest': digest('other-scope')},
                'snapshot': 'other-fixed-snapshot', 'pit_policy': 'other-fixed-policy',
                'calendar': manifest['calendar']+['2025-01-01'],
                'universe': manifest['universe']+['D'], 'catalog_ref': digest('other-catalog'),
                'feature_selection': manifest['feature_selection']+[{'id': 'OTHER'}],
                'ordered_features': manifest['ordered_features']+['OTHER'],
                'feature_parents': deepcopy(manifest['feature_parents'])}
            changes['feature_parents'][0]['features']['file_digest'] = digest('other-features')
            for key, value in changes.items():
                with self.subTest(field=key):
                    changed = deepcopy(manifest)
                    changed[key] = value
                    self.assertNotEqual(common_source_identity(changed), shared.common_source_identity)
                    with self.assertRaisesRegex(ValueError, 'common source identity mismatch'):
                        project_saved_fold(changed, spec, feature_inputs=shared)

    def test_public_copies_and_fold_results_cannot_poison_saved_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, spec = fixture(Path(temp))
            shared = load_saved_feature_inputs(manifest)
            expected = project_saved_fold(manifest, spec, feature_inputs=shared)
            day = manifest['calendar'][0]
            key = 'A', day
            feature_ref = shared.parents_by_session[day]['feature_ref']
            exposed_row = shared.rows[key]
            exposed_row['values'][0] = -1000
            exposed_row['validity'][0] = False
            exposed_row['source_refs'].append('changed')
            shared.parents_by_session[day]['feature_ref'] = 'changed'
            shared.feature_build(feature_ref)['rows'][0]['values'][0] = -1000
            shared.feature_builds[feature_ref]['qlib_view']['queries'][0]['symbols'].append('D')
            with self.assertRaises(TypeError):
                shared.rows[key] = exposed_row
            with self.assertRaises(AttributeError):
                shared.common_source_identity = 'changed'
            changed_output = project_saved_fold(manifest, spec, feature_inputs=shared)
            changed_output[0]['rows'][0]['values'][0] = -1000
            changed_output[0]['parents_by_session'][day]['feature_ref'] = 'changed'
            changed_output[2][0]['values'][0] = -1000
            self.assertEqual(project_saved_fold(manifest, spec, feature_inputs=shared), expected)

    def test_unvalidated_booleans_fakes_and_copies_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, spec = fixture(Path(temp))
            shared = load_saved_feature_inputs(manifest)
            impostors = [True, False, {}, object(), object.__new__(type(shared)),
                         copy(shared), deepcopy(shared)]
            for impostor in impostors:
                with self.subTest(impostor=type(impostor).__name__):
                    with self.assertRaisesRegex(ValueError, 'validated saved Feature inputs required'):
                        project_saved_fold(manifest, spec, feature_inputs=impostor)

    def test_shared_projection_reads_common_parents_once_and_standalone_reads_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, spec = fixture(Path(temp))
            common_paths = [manifest['scope']['path'],
                manifest['feature_parents'][0]['features']['path'],
                manifest['feature_parents'][0]['input_evidence']['path']]
            with patch.object(stock_fold_inputs, '_read', wraps=stock_fold_inputs._read) as reads:
                shared = load_saved_feature_inputs(manifest)
                first = project_saved_fold(manifest, spec, feature_inputs=shared)
                second = project_saved_fold(manifest, spec, feature_inputs=shared)
                self.assertEqual(first, second)
                paths = [str(call.args[0]) for call in reads.call_args_list]
                self.assertEqual([paths.count(path) for path in common_paths], [1, 1, 1])
                standalone = project_saved_fold(manifest, spec)
                self.assertEqual(standalone, first)
                paths = [str(call.args[0]) for call in reads.call_args_list]
                self.assertEqual([paths.count(path) for path in common_paths], [2, 2, 2])


class CalendarYearFoldTests(unittest.TestCase):
    def setUp(self):
        # Deliberate holes are frozen calendar facts, including a holiday gap.
        self.calendar = ['2021-12-28', '2021-12-29', '2021-12-30', '2022-01-04',
            '2022-04-07', '2023-12-27', '2023-12-28', '2023-12-29',
            '2024-01-02', '2024-01-03']
        self.spec = fold_spec(self.calendar, '2023-12-29', ['2024-01-02', '2024-01-03'])

    def test_inclusive_start_exclusive_fit_and_last_actual_session(self):
        training, prediction = validate_spec(self.spec, self.calendar)
        self.assertEqual(training, ['2021-12-29', '2021-12-30', '2022-01-04',
            '2022-04-07', '2023-12-27', '2023-12-28'])
        self.assertEqual(training[0], '2021-12-29')
        self.assertEqual(training[-1], '2023-12-28')
        self.assertNotIn(self.spec['fit_session'], training)
        self.assertEqual(prediction, ['2023-12-29', '2024-01-02'])

    def test_holiday_holes_do_not_create_sessions_or_retreat_an_extra_day(self):
        calendar = ['2022-01-07', '2022-01-10', '2022-02-01', '2024-01-04',
                    '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-12']
        spec = fold_spec(calendar, '2024-01-08', ['2024-01-09', '2024-01-12'])
        training, prediction = validate_spec(spec, calendar)
        self.assertEqual(training, ['2022-01-10', '2022-02-01', '2024-01-04', '2024-01-05'])
        self.assertEqual(prediction, ['2024-01-08', '2024-01-09'])
        self.assertNotIn('2022-01-08', training)
        self.assertNotIn('2024-01-11', prediction)

    def test_february_29_clamps_to_inclusive_february_28(self):
        calendar = ['2022-02-25', '2022-02-28', '2022-03-01', '2024-02-27',
                    '2024-02-28', '2024-02-29', '2024-03-01', '2024-03-04']
        spec = fold_spec(calendar, '2024-02-29', ['2024-03-01', '2024-03-04'])
        training, prediction = validate_spec(spec, calendar)
        self.assertEqual(training, ['2022-02-28', '2022-03-01', '2024-02-27', '2024-02-28'])
        self.assertEqual(prediction, ['2024-02-29', '2024-03-01'])

    def test_calendar_must_prove_coverage_before_left_boundary(self):
        for calendar in (self.calendar[1:], self.calendar[2:]):
            with self.subTest(first_session=calendar[0]):
                with self.assertRaisesRegex(ValueError, 'insufficient frozen two-year training calendar/lookback'):
                    validate_spec(self.spec, calendar)
        self.assertEqual(validate_spec(self.spec, self.calendar)[0][0], '2021-12-29')

    def test_v2_rejects_non_integer_lengths_and_window_extensions(self):
        for length in (True, False, 2.0, 1, 3):
            with self.subTest(length=length):
                spec = deepcopy(self.spec)
                spec['training_window']['length'] = length
                with self.assertRaisesRegex(ValueError, 'requires two calendar years'):
                    validate_spec(spec, self.calendar)
        for field, value in (('leap_day', 'roll_to_march'), ('end', 'two_sessions_before_fit'),
                             ('extra_policy', True)):
            with self.subTest(field=field):
                spec = deepcopy(self.spec)
                spec['training_window'][field] = value
                with self.assertRaisesRegex(ValueError, 'requires two calendar years'):
                    validate_spec(spec, self.calendar)

    def test_v2_rejects_duplicate_or_unordered_frozen_calendars(self):
        duplicate = self.calendar[:2]+[self.calendar[1]]+self.calendar[2:]
        unordered = self.calendar[:]
        unordered[1:3] = reversed(unordered[1:3])
        for calendar in (duplicate, unordered):
            with self.subTest(calendar=calendar[:4]):
                with self.assertRaisesRegex(ValueError, 'ordered unique calendar required'):
                    validate_spec(self.spec, calendar)

    def test_both_versions_preserve_stage_clocks_and_timestamp_equivalence(self):
        calendar = ['2021-12-31']+[(date(2022, 1, 5)+timedelta(days=10*i)).isoformat()
            for i in range(65)]+['2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09']
        for version in ('stock_ml_fold_spec_v1', 'stock_ml_fold_spec_v2'):
            with self.subTest(version=version):
                spec = fold_spec(calendar, '2024-01-05', ['2024-01-08', '2024-01-09'], version=version)
                expected = validate_spec(spec, calendar)
                equivalent = deepcopy(spec)
                equivalent['fit_cutoff'] = '2024-01-05T12:30:00Z'
                equivalent['simulated_model_available_at'] = '2024-01-05T12:45:00Z'
                equivalent['inference_cutoff_by_session'] = {
                    day: day+'T13:00:00Z' for day in expected[1]}
                self.assertEqual(validate_spec(equivalent, calendar), expected)
                if version == 'stock_ml_fold_spec_v1':
                    self.assertEqual(expected[0], calendar[calendar.index(spec['fit_session'])-65:
                                                           calendar.index(spec['fit_session'])])
                invalid = [
                    ('fit_cutoff', '2024-01-05T20:31:00+08:00', 'bounded fit/model clock conflict'),
                    ('simulated_model_available_at', '2024-01-05T20:44:00+08:00', 'bounded fit/model clock conflict'),
                    ('inference_cutoff_by_session', {day: day+'T21:01:00+08:00' for day in expected[1]},
                     'fold inference clock conflict'),
                    ('inference_cutoff_by_session', {day: day+'T21:00:00+08:00' for day in spec['oos_trade_sessions']},
                     'strict previous-session inference clocks required'),
                    ('evaluation_cutoff', expected[1][-1]+'T21:00:00+08:00', 'must follow inference')]
                for key, value, message in invalid:
                    with self.subTest(clock=key, value=value):
                        changed = deepcopy(spec)
                        changed[key] = value
                        with self.assertRaisesRegex(ValueError, message):
                            validate_spec(changed, calendar)


if __name__ == '__main__':
    unittest.main()
