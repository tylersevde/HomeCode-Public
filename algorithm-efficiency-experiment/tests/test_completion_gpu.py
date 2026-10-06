"""GPU crossover design and reconstructed evidence without GPU/HAT access."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from efficiency import completion_gpu_spec as spec
from efficiency import completion_gpu_protocol as protocol
from efficiency import completion_gpu_audit as auditor
from efficiency.attention_native import fixture, oracle, errors
from efficiency.common import atomic_json, digest_file, read_jsonl
from efficiency.cpu_compare_spec import environment
from efficiency.cpu_policy import NUMERICAL_LIBRARY_THREADS
from efficiency.feedback_spec import input_hash
from efficiency.refine_spec import POLICIES
from efficiency.refine_worker import environment as policy_environment
from efficiency.research_report import seal


def configuration(stage='develop'):
    return dict(profile='completion', track='gpu', stage=stage, fixture_namespace='fresh-'+stage,
                max_seconds=7200, reserved_seconds=7200, phase='cpu', original_governor='ondemand', governor_socket='fake', campaign='fake')


def synthetic(c, gpu_ms=80):
    cells = {cell['cell_id']: cell for cell in spec.CELLS}
    result = []
    for job in spec.schedule(c, spec.fixtures(c)):
        native = dict(fixture_id=job['fixture_id'], **spec.arm(cells[job['cell_id']], job['label']),
                      correct=True, matches_warmup=True, unexpected_output_io=False, validation_errors=0)
        result.append(dict(event='measurement', **job,
                           total_ms={'cpu': 100, 'cpu_duplicate': 100, 'gpu': 300, 'stream64': gpu_ms}[job['label']],
                           response=dict(result=native)))
    return result + [dict(event='measurement_complete')]


class DesignTests(unittest.TestCase):
    def test_matrix_native_bounds_and_stage_counts(self):
        self.assertEqual(len(spec.CELLS), 12)
        self.assertEqual({(c['n'], c['d'], c['b']) for c in spec.CELLS},
                         set(itertools.product((128, 512, 1024), (64, 128), (1, 16))))
        self.assertTrue(all(spec.workspace_bytes(c['n'], c['d'], c['b']) < 512*1024**2 for c in spec.CELLS))
        for stage, blocks in (('develop', 8), ('confirm', 16)):
            c = configuration(stage)
            self.assertEqual(len(spec.schedule(c, spec.fixtures(c))), (blocks+1)*12*3*4)
            self.assertEqual(spec.spec(c)['stage_max_seconds'], 7200)
        self.assertGreater(spec.workspace_bytes(4096, 128, 16), 512*1024**2)

    def test_order_balance_fresh_data_and_thread_routes(self):
        c = configuration()
        orders = [order for block in range(8) for order in spec.orders(c, 'cell', block, 'main')]
        self.assertEqual(set(orders), set(itertools.permutations(spec.LABELS)))
        self.assertEqual(set(Counter((a, b) for order in orders for a, b in zip(order, order[1:])).values()), {6})
        self.assertFalse({f['seed'] for f in spec.fixtures(c)} & {f['seed'] for f in spec.fixtures(configuration('confirm'))})
        for cell in spec.CELLS:
            self.assertEqual(spec.arm(cell, 'cpu')['backend'], 'native1' if cell['b'] == 1 else 'native4')
            self.assertEqual(spec.arm(cell, 'stream64'), dict(arm='stream64', backend='C6', variant=4))

    def test_pilot_reserves_margin_and_rejects_infeasible_complete_matrix(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(protocol.time, 'monotonic', return_value=100.):
            directory = Path(tmp); config = configuration()
            protocol.pilot_gate(directory, config, SimpleNamespace(work_deadline=2000), 0.)
            gate = read_jsonl(directory/spec.EVENTS)[0][-1]
            self.assertEqual(gate['required_seconds'], 100*8*1.2)
            with self.assertRaisesRegex(TimeoutError, 'cannot fit'):
                protocol.pilot_gate(directory, config, SimpleNamespace(work_deadline=1000), 0.)

    def test_invalid_stage_budget_rejected_before_devices(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(protocol, 'controls') as controls:
            for seconds in (120, 7201, float('nan')):
                with self.assertRaisesRegex(ValueError, 'allowance'):
                    protocol.run(Path(tmp), dict(configuration(), max_seconds=seconds))
            controls.assert_not_called()


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        context = patch.dict(spec.BLOCKS, develop=2, confirm=2)
        context.start(); self.addCleanup(context.stop)

    def checked(self, rows, config):
        summary = spec.analyze(rows, config)
        if summary['complete']:
            auditor.check_decisions(rows, config, summary)
        return summary

    def test_development_selects_and_confirmation_uses_frozen_routes(self):
        development = configuration()
        selected = self.checked(synthetic(development), development)
        self.assertFalse(selected['accepted'])
        self.assertTrue(all(route == 'stream64' for route in selected['selected']['routes'].values()))
        confirm = dict(configuration('confirm'), selected=selected['selected'])
        result = self.checked(synthetic(confirm), confirm)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['speedup']['estimate'], 1.25)

    def test_improvement_against_old_gpu_does_not_establish_cpu_crossover(self):
        c = configuration()
        result = self.checked(synthetic(c, gpu_ms=150), c)
        self.assertEqual(result['old_gpu_speedup']['estimate'], 2)
        self.assertIsNone(result['selected'])
        self.assertEqual(set(result['routes'].values()), {'cpu'})

    def test_one_bad_shape_cannot_hide_inside_confirmation_aggregate(self):
        c = dict(configuration('confirm'), selected=dict(routes={cell['cell_id']: 'stream64' for cell in spec.CELLS}))
        rows = synthetic(c, gpu_ms=50)
        for row in rows[:-1]:
            if row['label'] == 'stream64' and row['cell_id'] == spec.CELLS[0]['cell_id']:
                row['total_ms'] = 106
        result = self.checked(rows, c)
        self.assertGreater(result['speedup']['estimate'], 1.1)
        self.assertEqual(result['max_shape_slowdown'], 1.06)
        self.assertFalse(result['accepted'])

    def test_duplicate_controls_and_wrong_outputs_block_selection(self):
        c = configuration()
        for mode in ('unstable', 'incorrect'):
            rows = synthetic(c)
            for row in rows[:-1]:
                if row['label'] == 'cpu_duplicate' and mode == 'unstable':
                    row['total_ms'] = 120
                if row['label'] == 'stream64' and mode == 'incorrect':
                    row['response']['result']['correct'] = False
            self.assertIsNone(self.checked(rows, c)['selected'])

    def test_schedule_tamper_missing_pilot_and_wrong_routes_reject(self):
        c = configuration()
        original = synthetic(c)
        duplicate = deepcopy(original); duplicate[1] = deepcopy(duplicate[0])
        for rows in (original[:-2]+original[-1:], duplicate,
                     [r for r in original if r.get('section') != 'pilot']):
            self.assertFalse(spec.analyze(rows, c)['complete'])
        confirm = dict(configuration('confirm'), selected=dict(routes={'missing': 'stream64'}))
        self.assertEqual(spec.analyze(synthetic(confirm), confirm)['decision'], 'invalid_selected_routes')

    def test_pilot_not_used_in_statistics_and_forged_summary_rejected(self):
        c = configuration(); rows = synthetic(c)
        for row in rows[:-1]:
            if row['section'] == 'pilot':
                row['total_ms'] *= 100
        summary = self.checked(rows, c)
        self.assertEqual(summary, self.checked(synthetic(c), c))
        summary['speedup']['estimate'] = 10
        with self.assertRaisesRegex(ValueError, 'statistical decision'):
            auditor.check_decisions(rows, c, summary)

    def test_uncertain_or_less_than_ten_percent_gain_rejected(self):
        c = configuration()
        self.assertIsNone(self.checked(synthetic(c, gpu_ms=95), c)['selected'])
        rows = synthetic(c)
        for row in rows[:-1]:
            if row['label'] == 'stream64' and row['section'] == 'main':
                row['total_ms'] = 30 if row['block'] == 0 else 130
        result = self.checked(rows, c)
        self.assertIsNone(result['selected'])
        self.assertGreater(result['metrics'][0]['stream64_cpu_speedup']['estimate'], 1.1)
        self.assertLess(result['metrics'][0]['stream64_cpu_speedup']['interval'][0], 1)


class Evidence:
    def __init__(self, root, development=None):
        self.root = root; self.clock = 100.; self.released = False; self.bad_output = False
        self.directory = root/'run'; self.directory.mkdir()
        self.parent = root/'historical'; (self.parent/'native-build').mkdir(parents=True)
        source = {'efficiency/attention_native.py': 'unchanged wrapper', 'native/attention/attention.cpp': 'unchanged native'}
        self.sources = {}
        for name, content in source.items():
            target = self.directory/'source'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(content)
            self.sources[name] = digest_file(target)
        artifacts = {}
        for name in ('libattention.so', 'libaffinity.so', 'project-stream64.spv', 'scores-stream64.spv', 'apply-stream64.spv'):
            path = self.parent/'native-build'/name; path.write_bytes(b'fake-native-never-loaded'); artifacts[name] = digest_file(path)
        atomic_json(self.parent/'native-build/build.json', dict(sources={'native/attention/attention.cpp': self.sources['native/attention/attention.cpp']}, artifacts=artifacts))
        atomic_json(self.parent/'manifest.json', dict(source_sha256=self.sources))
        atomic_json(self.parent/'config.json', dict(reliability_stage='gpu-confirm'))
        atomic_json(self.parent/'summary.json', dict(complete=True, accepted=True, selected=5))
        atomic_json(self.parent/'validation.json', dict(passed=True)); seal(self.parent)
        self.config = dict(configuration('confirm' if development else 'develop'), parents={'gpu-confirmed': str(self.parent)})
        if development:
            self.config['parents']['gpu-develop'] = str(development)
            self.config['selected'] = json.loads((development/'summary.json').read_text())['selected']
        atomic_json(self.directory/'config.json', self.config)
        atomic_json(self.directory/'protocol.json', spec.spec(self.config))
        atomic_json(self.directory/'prior-fixtures.json', dict(seeds=[], hashes=[]))
        atomic_json(self.directory/'manifest.json', dict(source_sha256=self.sources, config=self.config, initial_throttle_flags=0))
        protocol.prepare_source(self.directory, self.config, dict(source_sha256=self.sources))

    def controls(self, directory, budget):
        rows = []
        def save(array):
            path = 'outputs/'+input_hash(array)+'.npy'; np.save(directory/path, array, allow_pickle=False); return path
        for n, d, b in ((1, 1, 1), (17, 7, 2), (33, 17, 4), (65, 65, 2)):
            x, weights = fixture(n, d, b, 17701+n); expected = oracle(x, weights)
            name = f'controls/n{n}-d{d}-b{b}.npz'; np.savez(directory/name, x=x, w=weights, expected=expected)
            output = expected.astype(np.float32)
            for mode, variants in (('stream', (0, 3, 4)), ('prefill', (0, 1, 2))):
                for variant in variants:
                    rows.append(dict(kind='oracle', mode=mode, variant=variant, input_file=name,
                                     output_file=save(output), **errors(output, expected), validation_errors=0))
                    if variant in (3, 4):
                        cut = max(1, n//2); changed = x.copy(); changed[0, cut:] *= -1
                        perturbed = oracle(changed, weights).astype(np.float32)
                        rows.append(dict(kind='semantic', mode=mode, variant=variant, input_file=name, cut=cut, correct=True,
                                         output_files=dict(baseline=save(output), repeated=save(output), perturbed=save(perturbed))))
        atomic_json(directory/'correctness.json', rows)

    def worker(self, architecture, kind, directory, config, check, factory=None, environment=None):
        if architecture != 'process' or kind != 'numeric' or config['numeric_device'] != 'both' or factory is not protocol.CrossoverNumeric:
            raise AssertionError('Wrong numerical owner')
        owner = dict(pid=42, tid=42)
        env = dict(numeric_device='both', cpu_initialization_ms=1, gpu_initialization_ms=2,
                   policy=POLICIES[5], observed=dict(governor='performance'),
                   openmp_environment=policy_environment(POLICIES[5]),
                   openmp_actual={k: v for k, v in environment.items() if v is not None},
                   numerical_library_threads=NUMERICAL_LIBRARY_THREADS,
                   probes=[dict(requested=n, team=n, threads=[dict(mask=1 << i) for i in range(n)]) for n in (1, 4)])
        def call(operation, **payload):
            if operation == 'load':
                self.clock += .001; return dict(result={})
            start = self.clock; label = payload['arm']['arm']; fid = payload['fixture_id']
            duration = .005 if operation == 'warm' else dict(cpu=.1, cpu_duplicate=.1, gpu=.3, stream64=.08)[label]
            self.clock += duration
            with np.load(directory/'fixtures'/(fid+'.npz'), allow_pickle=False) as data:
                expected = data['expected']; output = expected.astype(np.float32)
                if self.bad_output and operation == 'measure':
                    output += 1
                relative = 'outputs/'+input_hash(output)+'.npy'; np.save(directory/relative, output, allow_pickle=False)
                result = dict(fixture_id=fid, **payload['arm'], **errors(output, expected), matches_warmup=True,
                              unexpected_output_io=False, validation_errors=0, output_file=relative, output_sha256=input_hash(output),
                              policy=POLICIES[5], system_before=dict(governor='performance'), system_after=dict(governor='performance'),
                              observer_ms=.01, started=start+.0002, computed=start+duration*.8, ended=start+duration*.9,
                              worker_request_ms=((start+duration*.9)-(start+.0002))*1000, request_ms=duration*700, cpu_seconds=duration*.2)
            return dict(owner=owner, submitted=start+.0001, received=start+.0002, ended=start+duration*.98,
                        delivered=self.clock, result=result)
        def close():
            self.released = True
        return SimpleNamespace(ready=dict(owner=owner, environment=env), call=call, close=close,
                               release=dict(owner=owner, alive=False, forced=False))

    def run(self):
        budget = SimpleNamespace(work_deadline=7200, check=lambda: None)
        with patch.object(protocol, 'Budget', return_value=budget), patch.object(protocol, 'controls', side_effect=self.controls), \
             patch.object(protocol, 'request', return_value=dict(current='performance', original='ondemand')), \
             patch.object(protocol, 'register_fixture'), patch.object(protocol, 'wait_cool'), \
             patch.object(protocol, 'Worker', side_effect=self.worker), patch.object(protocol.time, 'monotonic', side_effect=lambda: self.clock):
            protocol.run(self.directory, self.config)
        events = read_jsonl(self.directory/spec.EVENTS)[0]
        atomic_json(self.directory/'summary.json', spec.analyze(events, self.config))
        atomic_json(self.directory/'outcome.json', dict(status='complete', elapsed_seconds=self.clock-100))
        atomic_json(self.directory/'governor-restoration.json', dict(original='ondemand', current='ondemand', restored=True))
        atomic_json(self.directory/'runtime-finalization.json', dict(started_monotonic=100., finished_monotonic=self.clock,
                    charged_seconds=self.clock-100, reserved_seconds=7200, restored=True, original_governor='ondemand'))
        (self.directory/'telemetry.jsonl').write_text(''.join(json.dumps(dict(event='sample', throttle_flags=0,
            cpu_temp_c=40, available_memory_bytes=4*1024**3, swap_used_bytes=0, hat_max_c=None, hat_sample_age_s=None,
            monotonic=instant))+'\n' for instant in (100., self.clock)))
        seal(self.directory)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.dict(spec.BLOCKS, develop=2, confirm=2))
        cells = [dict(cell_id=f'tiny-{n}', n=n, d=2, b=1, mode='stream', cpu='native1', gpu='C6') for n in (2, 3)]
        self.stack.enter_context(patch.object(spec, 'CELLS', cells))
        # The common budget auditor is covered by its own ledger integration suite.
        # Everything GPU-specific, including seals and provenance, is real here.
        self.stack.enter_context(patch.object(auditor, 'audit_budget', return_value=dict(passed=True)))
        self.evidence = Evidence(Path(temporary)); self.evidence.run()

    def test_fake_protocol_and_independent_sealed_audit_pass(self):
        with patch.object(spec, 'analyze', side_effect=AssertionError('Audit must independently reconstruct decisions')):
            result = auditor.audit(self.evidence.directory)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['measurements'], 3*2*3*4)
        self.assertTrue(self.evidence.released)

    def test_common_budget_failure_blocks_otherwise_qualifying_evidence(self):
        with patch.object(auditor, 'audit_budget', side_effect=ValueError('Budget lineage tampered')):
            result = auditor.audit(self.evidence.directory)
        self.assertFalse(result['passed']); self.assertIn('Budget lineage tampered', result['error'])

    def test_fresh_confirmation_seal_reconstructs_parent_and_fixed_routes(self):
        root = self.evidence.root/'confirmation'; root.mkdir()
        confirmed = Evidence(root, development=self.evidence.directory); confirmed.run()
        result = auditor.audit(confirmed.directory)
        self.assertTrue(result['passed'], result)
        summary = json.loads((confirmed.directory/'summary.json').read_text())
        self.assertTrue(summary['accepted'])
        self.assertEqual(summary['selected'], confirmed.config['selected'])

    def test_resealed_summary_tamper_rejected(self):
        path = self.evidence.directory/'summary.json'; saved = json.loads(path.read_text()); saved['speedup']['estimate'] = 99
        atomic_json(path, saved); seal(self.evidence.directory)
        self.assertIn('statistical decision', auditor.audit(self.evidence.directory)['error'])

    def test_preseal_runtime_and_telemetry_tampering_rejected(self):
        directory = self.evidence.directory; path = directory/'runtime-finalization.json'
        original = json.loads(path.read_text()); altered = dict(original, charged_seconds=original['charged_seconds']-1)
        atomic_json(path, altered)
        self.assertIn('Full GPU attempt', auditor.audit(directory, require_seal=False)['error'])
        atomic_json(path, original)
        path = directory/'telemetry.jsonl'; original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]; rows[0]['monotonic'] = rows[-1]['monotonic']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); seal(directory)
        self.assertIn('telemetry does not cover', auditor.audit(directory)['error'])

    def test_resealed_fixture_tamper_rejected_even_with_updated_fixture_hash(self):
        directory = self.evidence.directory; fs = json.loads((directory/'fixtures.json').read_text())
        path = directory/'fixtures'/(fs[0]['fixture_id']+'.npz')
        with np.load(path, allow_pickle=False) as data:
            x, w, expected = data['x'].copy(), data['w'].copy(), data['expected'].copy()
        x[0, 0, 0] += 1; np.savez(path, x=x, w=w, expected=expected)
        fs[0]['file_sha256'] = digest_file(path); atomic_json(directory/'fixtures.json', fs); seal(directory)
        self.assertIn('Fixture/FP64 oracle changed', auditor.audit(directory)['error'])

    def test_resealed_native_output_tamper_rejected(self):
        directory = self.evidence.directory; rows = read_jsonl(directory/spec.EVENTS)[0]
        event = next(row for row in rows if row['event'] == 'measurement')
        path = directory/event['response']['result']['output_file']; output = np.load(path); output += 1; np.save(path, output)
        seal(directory)
        result = auditor.audit(directory)
        self.assertFalse(result['passed']); self.assertIn('Output/oracle', result['error'])

    def test_resealed_duplicate_measurement_and_governor_drift_rejected(self):
        directory = self.evidence.directory; path = directory/spec.EVENTS; original = path.read_text()
        for change in ('order', 'governor'):
            rows = [json.loads(line) for line in original.splitlines()]
            positions = [i for i, row in enumerate(rows) if row['event'] == 'measurement']
            if change == 'order':
                rows[positions[1]] = deepcopy(rows[positions[0]])
            else:
                rows[positions[0]]['response']['result']['system_after']['governor'] = 'ondemand'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows)); seal(directory)
            self.assertFalse(auditor.audit(directory)['passed'])

    def test_confirmation_cannot_change_selected_routes_or_namespace(self):
        development = self.evidence.directory
        for mutate in ('routes', 'namespace'):
            out = self.evidence.root/('confirm-'+mutate); out.mkdir()
            config = dict(configuration('confirm'), parents={'gpu-confirmed': str(self.evidence.parent), 'gpu-develop': str(development)},
                          selected=json.loads((development/'summary.json').read_text())['selected'])
            if mutate == 'routes':
                config['selected'] = dict(routes={cell['cell_id']: 'cpu' for cell in spec.CELLS})
            else:
                config['fixture_namespace'] = self.evidence.config['fixture_namespace']
            for name, value in (('config.json', config), ('protocol.json', spec.spec(config)), ('prior-fixtures.json', dict(seeds=[], hashes=[]))):
                atomic_json(out/name, value)
            with self.assertRaisesRegex(ValueError, 'routing differs|new fixture namespace'):
                protocol.prepare_source(out, config, dict(source_sha256=self.evidence.sources))

    def test_numerical_failure_releases_owner_without_completing_matrix(self):
        root = self.evidence.root/'failure'; root.mkdir()
        failed = Evidence(root); failed.bad_output = True
        with self.assertRaisesRegex(RuntimeError, 'measured numerical failure'):
            failed.run()
        self.assertTrue(failed.released)
        events = read_jsonl(failed.directory/spec.EVENTS)[0]
        self.assertEqual(events[-1]['event'], 'failure')
        self.assertFalse(any(event['event'] == 'measurement_complete' for event in events))


if __name__ == '__main__':
    unittest.main()
