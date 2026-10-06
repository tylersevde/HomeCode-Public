"""CPU comparison science and scheduling without device access."""
from collections import Counter
from copy import deepcopy
import itertools
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from efficiency import cpu_compare_spec as spec
from efficiency.refine_spec import POLICIES
from efficiency.refine_worker import environment as policy_environment


def config():
    return dict(fixture_namespace='cpu-compare-unit-fixtures')


def events(c, single=80, batch=1280):
    bycell = {cell['cell_id']: cell for cell in spec.CELLS}
    result = []
    for job in spec.schedule(c, spec.fixtures(c)):
        is_batch = job['label'].startswith('batch')
        duration = (1600 if is_batch else 100) if job['configuration'] == 'baseline' else (batch if is_batch else single)
        call = dict(fixture_id=job['fixture_id'], **spec.arm(bycell[job['cell_id']], job['label']),
                    correct=True, matches_warmup=True, validation_errors=0, unexpected_output_io=False)
        raw = dict(correct=True, requests=[dict(call, request_id=i) for i in range(16)]) if is_batch else call
        result.append(dict(event='measurement', **job, total_ms=duration, response=dict(result=raw)))
    result.append(dict(event='measurement_complete'))
    return result


class DesignTests(unittest.TestCase):
    def test_frozen_scientific_contract_and_defensive_copy(self):
        contract = spec.specification()
        self.assertEqual((contract['blocks'], contract['max_seconds'], contract['repeats'],
                          contract['batch_calls'], contract['bootstrap_samples']), (256, 14400, 3, 16, 10000))
        self.assertEqual(contract['minimum_speedup'], 1.10)
        self.assertEqual(contract['cleanup_seconds'], 120)
        self.assertEqual(contract['pilot_margin'], 1.2)
        self.assertEqual(contract['equivalence_interval'], [.95, 1.05])
        self.assertEqual(contract['configuration_policies']['candidate'], POLICIES[5])
        self.assertEqual(json.loads(json.dumps(contract)), contract)
        contract['cells'][0]['cpu'] = 'changed'
        self.assertNotEqual(spec.specification()['cells'][0]['cpu'], 'changed')

    def test_environment_clears_standard_and_unknown_inherited_overrides(self):
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '99', 'OMP_FUTURE_EXTENSION': 'on',
                                     'GOMP_NEW_OVERRIDE': 'off', 'UNRELATED': 'keep'}):
            for configuration in spec.CONFIGURATIONS:
                overlay = spec.environment(configuration)
                child = dict(os.environ)
                for key, value in overlay.items():
                    if value is None:
                        child.pop(key, None)
                    else:
                        child[key] = value
                actual = {key: value for key, value in child.items() if key.startswith(('OMP_', 'GOMP_'))}
                expected = {} if configuration == 'baseline' else {
                    key: value for key, value in policy_environment(POLICIES[5]).items() if value is not None}
                self.assertEqual(actual, expected)
                self.assertEqual(child['UNRELATED'], 'keep')

    def test_frozen_spec_does_not_depend_on_parent_environment(self):
        original = spec.specification()
        with patch.dict(os.environ, {'OMP_NEW_SETTING': 'different'}):
            self.assertEqual(original, spec.specification())

    def test_unknown_configuration_and_arm_rejected(self):
        with self.assertRaises(ValueError):
            spec.environment('new-policy')
        with self.assertRaises(ValueError):
            spec.arm(spec.CELLS[0], 'gpu')

    def test_production_matrix_fixtures_and_identical_cpu_routing(self):
        c = config()
        fs = spec.fixtures(c)
        jobs = spec.schedule(c, fs)
        self.assertEqual(len(fs), 257 * 6)
        self.assertEqual(len(jobs), 257 * 2 * 6 * 3 * 4)
        self.assertEqual(sum(16 if j['label'].startswith('batch') else 1 for j in jobs if j['section'] == 'main'), 313344)
        self.assertEqual(len({f['seed'] for f in fs}), len(fs))
        self.assertEqual(spec.fixtures(c), fs)
        alternate = dict(c, fixture_namespace='independent-retry')
        self.assertFalse({f['seed'] for f in fs} & {f['seed'] for f in spec.fixtures(alternate)})
        self.assertEqual({f['block'] for f in fs if f['section'] == 'pilot'}, {-1})
        for cell in spec.CELLS:
            for label in spec.CPU_ARMS:
                self.assertEqual(spec.arm(cell, label), dict(arm=label, backend=cell['cpu'], variant=0))

    def test_orders_balance_and_configurations_alternate(self):
        c = config()
        with patch.object(spec, 'BLOCKS', 8):
            jobs = spec.schedule(c, spec.fixtures(c))
        for cell in spec.CELLS:
            orders = [order for block in range(8) for order in spec.orders(c, cell['cell_id'], block, 'main')]
            self.assertEqual(set(orders), set(itertools.permutations(spec.CPU_ARMS)))
            transitions = Counter((a, b) for order in orders for a, b in zip(order, order[1:]))
            self.assertEqual(set(transitions.values()), {6})
        for block in range(8):
            selected = [j for j in jobs if j['section'] == 'main' and j['block'] == block]
            self.assertEqual(selected[0]['configuration'], spec.CONFIGURATIONS[block % 2])
            a, b = [[{k: v for k, v in j.items() if k != 'configuration'} for j in selected
                     if j['configuration'] == configuration] for configuration in spec.CONFIGURATIONS]
            self.assertEqual(a, b)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        # Runtime design remains fixed. Only module globals in these in-process
        # synthetic tests are reduced; manifest/config keys cannot change them.
        override = patch.object(spec, 'BLOCKS', 8)
        override.start()
        self.addCleanup(override.stop)
        self.c = config()

    def analyze(self, rows):
        return spec.analyze(rows, self.c)

    def test_qualifying_both_kinds_produces_no_default_change(self):
        summary = self.analyze(events(self.c))
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['accepted'])
        self.assertEqual(summary['decision'], 'cpu_improved')
        self.assertFalse(summary['defaults_changed'])
        self.assertEqual([metric['kind'] for metric in summary['metrics']], ['single', 'batch'])
        for metric in summary['metrics']:
            self.assertEqual(metric['speedup']['estimate'], 1.25)
            self.assertEqual(metric['speedup']['interval'], [1.25, 1.25])
            self.assertEqual(metric['speedup']['samples'], 10000)
            self.assertEqual(metric['speedup']['blocks'], 8)
            self.assertEqual(set(metric['per_cell_speedup']), {cell['cell_id'] for cell in spec.CELLS})

    def test_equal_speed_and_mixed_results_reject(self):
        for single, batch in ((100, 1600), (80, 1600), (100, 1280)):
            with self.subTest(single=single, batch=batch):
                summary = self.analyze(events(self.c, single, batch))
                self.assertTrue(summary['complete'])
                self.assertFalse(summary['accepted'])

    def test_significant_but_under_ten_percent_gain_rejects(self):
        summary = self.analyze(events(self.c, single=95, batch=1520))
        self.assertGreater(summary['metrics'][0]['speedup']['interval'][0], 1)
        self.assertFalse(summary['accepted'])

    def test_ten_percent_estimate_requires_paired_lower_bound_above_one(self):
        rows = events(self.c)
        for row in rows[:-1]:
            if (row['section'] == 'main' and row['configuration'] == 'candidate'
                    and row['label'].startswith('single')):
                row['total_ms'] = 10 if row['block'] < 4 else 150
        summary = self.analyze(rows)
        self.assertEqual(summary['metrics'][0]['speedup']['estimate'], 1.25)
        self.assertLess(summary['metrics'][0]['speedup']['interval'][0], 1)
        self.assertFalse(summary['accepted'])

    def test_aggregate_gain_cannot_hide_shape_regression(self):
        rows = events(self.c, single=50)
        for row in rows[:-1]:
            if (row['configuration'] == 'candidate' and row['cell_id'] == spec.CELLS[0]['cell_id']
                    and row['label'].startswith('single')):
                row['total_ms'] = 106
        summary = self.analyze(rows)
        self.assertGreater(summary['metrics'][0]['speedup']['estimate'], 1.1)
        self.assertEqual(summary['metrics'][0]['max_shape_slowdown'], 1.06)
        self.assertFalse(summary['accepted'])

    def test_each_configuration_and_kind_require_stable_duplicate_controls(self):
        for configuration in spec.CONFIGURATIONS:
            for kind in ('single', 'batch'):
                rows = events(self.c)
                for row in rows[:-1]:
                    if (row['configuration'] == configuration and row['label'] == kind + '_b'
                            and row['cell_id'] == spec.CELLS[0]['cell_id']):
                        row['total_ms'] *= 1.1
                with self.subTest(configuration=configuration, kind=kind):
                    summary = self.analyze(rows)
                    self.assertFalse(summary['accepted'])
                    self.assertFalse(next(m for m in summary['metrics'] if m['kind'] == kind)['stable'])

    def test_pilot_excluded_from_speed_estimate(self):
        rows = events(self.c)
        for row in rows[:-1]:
            if row['section'] == 'pilot':
                row['total_ms'] *= 1000 if row['configuration'] == 'candidate' else .01
        self.assertEqual(self.analyze(rows), self.analyze(events(self.c)))

    def test_missing_duplicate_reordered_and_wrong_configuration_reject(self):
        original = events(self.c)
        variants = []
        missing = deepcopy(original)
        del missing[0]
        variants.append(missing)
        duplicate = deepcopy(original)
        duplicate[1] = deepcopy(duplicate[0])
        variants.append(duplicate)
        reordered = deepcopy(original)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        variants.append(reordered)
        wrong = deepcopy(original)
        wrong[0]['configuration'] = 'candidate'
        variants.append(wrong)
        variants.append([dict(event='measurement_complete')] + original[:-1])
        variants.append(original + [dict(event='measurement_complete')])
        for rows in variants:
            summary = self.analyze(rows)
            self.assertFalse(summary['complete'])
            self.assertFalse(summary['accepted'])

    def test_missing_pilot_and_completion_reject(self):
        rows = events(self.c)
        for candidate in (rows[:-1], [r for r in rows if r.get('section') != 'pilot']):
            self.assertFalse(self.analyze(candidate)['complete'])

    def test_nonfinite_zero_negative_or_boolean_times_reject(self):
        for value in (float('nan'), float('inf'), 0, -1, True):
            rows = events(self.c)
            rows[0]['total_ms'] = value
            self.assertEqual(self.analyze(rows)['decision'], 'invalid_results')

    def test_incorrect_or_unvalidated_native_output_rejects(self):
        for field, value in (('correct', False), ('matches_warmup', False), ('validation_errors', 1),
                             ('unexpected_output_io', True), ('backend', 'C4'), ('variant', 4)):
            rows = events(self.c)
            row = next(r for r in rows[:-1] if r['label'].startswith('single') and r['section'] == 'main')
            row['response']['result'][field] = value
            with self.subTest(field=field):
                self.assertFalse(self.analyze(rows)['accepted'])

    def test_batch_requires_every_unique_correct_call(self):
        for mutation in ('missing', 'duplicate', 'incorrect', 'wrong_fixture'):
            rows = events(self.c)
            row = next(r for r in rows[:-1] if r['label'].startswith('batch') and r['section'] == 'main')
            calls = row['response']['result']['requests']
            if mutation == 'missing':
                calls.pop()
            elif mutation == 'duplicate':
                calls[-1]['request_id'] = calls[0]['request_id']
            elif mutation == 'incorrect':
                calls[-1]['correct'] = False
            else:
                calls[-1]['fixture_id'] = 'another-fixture'
            with self.subTest(mutation=mutation):
                self.assertFalse(self.analyze(rows)['accepted'])

    def test_config_cannot_reduce_preregistered_coverage_or_thresholds(self):
        rows = events(self.c, single=100, batch=1600)
        c = dict(self.c, blocks=1, minimum_speedup=.5, bootstrap_samples=1)
        self.assertEqual(spec.analyze(rows, c), self.analyze(rows))

    def test_whole_paired_blocks_preserve_common_drift(self):
        rows = events(self.c)
        for row in rows[:-1]:
            if row['section'] == 'main':
                row['total_ms'] *= (1, 50, 2, 100, 3, 200, 4, 1000)[row['block']]
        summary = self.analyze(rows)
        self.assertTrue(summary['accepted'])
        for metric in summary['metrics']:
            np.testing.assert_allclose(metric['speedup']['interval'], [1.25, 1.25])


if __name__ == '__main__':
    unittest.main()
