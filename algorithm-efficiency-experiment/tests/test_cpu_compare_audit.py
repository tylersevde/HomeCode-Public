import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from efficiency import cpu_compare_spec as spec
from efficiency.attention_native import fixture, oracle, errors
from efficiency.common import atomic_json, digest_file
from efficiency.cpu_compare_audit import audit, decisions, NUMERICAL_LIBRARY_THREADS
from efficiency.cpu_compare_report import build_report, plots
from efficiency.feedback_spec import input_hash
from efficiency.research_report import seal


CELLS = [dict(cell_id=f'stream-n{n}-b1', n=n, d=2, b=1, mode='stream', cpu='native1') for n in (2, 3)]


def raw_events(config, ratio=.75):
    events = []
    for job in spec.schedule(config, spec.fixtures(config)):
        row = dict(fixture_id=job['fixture_id'], **spec.arm(next(c for c in spec.CELLS if c['cell_id'] == job['cell_id']), job['label']),
                   correct=True, matches_warmup=True, validation_errors=0, unexpected_output_io=False)
        result = row if job['label'].startswith('single') else dict(correct=True, requests=[dict(row, request_id=i) for i in range(16)])
        duration = (16 if job['label'].startswith('batch') else 1) * (ratio if job['configuration'] == 'candidate' else 1)
        events.append(dict(event='measurement', **job, total_ms=duration, response=dict(result=result)))
    return events + [dict(event='measurement_complete')]


def archive(root):
    """Small, fully reconstructed CPU evidence; no hardware or external artifacts."""
    directory = root / 'run'; parent = root / 'parent'; source = root / 'source-campaign'
    for path in (directory, parent, source):
        path.mkdir()
    for path in (directory / 'fixtures', directory / 'outputs', directory / 'native-build',
                 directory / 'source/efficiency', directory / 'source/native/attention', parent / 'native-build'):
        path.mkdir(parents=True, exist_ok=True)
    (directory / 'source/efficiency/attention_native.py').write_text('archived wrapper\n')
    (directory / 'source/native/attention/attention.cpp').write_text('archived calculation\n')
    (directory / 'native-build/libattention.so').write_bytes(b'archived binary')
    (directory / 'native-build/libaffinity.so').write_bytes(b'archived affinity probe')
    hashes = {name: digest_file(directory / 'source' / name)
              for name in ('efficiency/attention_native.py', 'native/attention/attention.cpp')}
    build = dict(sources={'native/attention/attention.cpp': hashes['native/attention/attention.cpp']},
                 artifacts={name: digest_file(directory / 'native-build' / name) for name in ('libattention.so', 'libaffinity.so')})
    atomic_json(directory / 'native-build/build.json', build); atomic_json(parent / 'native-build/build.json', build)
    atomic_json(parent / 'config.json', dict(reliability_stage='cpu-confirm'))
    atomic_json(parent / 'manifest.json', dict(source_sha256=hashes))
    atomic_json(parent / 'summary.json', dict(complete=True, accepted=True, selected=5))
    atomic_json(parent / 'validation.json', dict(passed=True)); seal(parent)
    atomic_json(source / 'campaign.json', dict(attempts=[dict(stage='cpu-confirm', output=str(parent),
        scientific_complete=True, audit_passed=True, checksums_sha256=digest_file(parent / 'checksums.json'))]))
    atomic_json(source / 'final-validation.json', dict(passed=True)); seal(source)
    config = dict(profile='cpu-compare', phase='cpu', fixture_namespace='fresh-audit-fixtures',
        source_campaign=str(source), campaign=str(root / 'campaign'), original_governor='ondemand',
        baseline_seconds=15, cooldown_seconds=15, cleanup_reserve_seconds=120,
        max_seconds=14400, reserved_seconds=14400)
    atomic_json(directory / 'config.json', config); atomic_json(directory / 'protocol.json', spec.specification())
    atomic_json(directory / 'manifest.json', dict(config=config, source_sha256=hashes, initial_throttle_flags=0,
        cpu_model='Raspberry Pi 5 Model B', python='3.13.5', versions=dict(numpy='2.2.4')))
    atomic_json(directory / 'cpu-runtime.json', dict(version='cpu-policy-v1', model='Raspberry Pi 5 Model B',
        machine='aarch64', cpu_affinity=[0, 1, 2, 3], python='3.13.5', numpy='2.2.4',
        numerical_library_threads=NUMERICAL_LIBRARY_THREADS,
        native_artifacts={name: digest_file(directory / 'native-build' / name) for name in ('libattention.so', 'libaffinity.so', 'build.json')},
        libraries={name: dict(path='/usr/lib/' + name, sha256='a' * 64) for name in ('libgomp.so.1', 'libc.so.6')}))
    atomic_json(directory / 'provenance.json', dict(selected=5,
        source_campaign=dict(path=str(source), checksums_sha256=digest_file(source / 'checksums.json')),
        cpu_confirmation=dict(path=str(parent), checksums_sha256=digest_file(parent / 'checksums.json'))))
    atomic_json(directory / 'freeze.json', dict(source_sha256=hashes,
        config_sha256=digest_file(directory / 'config.json'), protocol_sha256=digest_file(directory / 'protocol.json'),
        provenance_sha256=digest_file(directory / 'provenance.json')))
    fs = []; outputs = {}
    for f in spec.fixtures(config):
        x, w = fixture(f['n'], f['d'], f['b'], f['seed']); expected = oracle(x, w)
        p = directory / 'fixtures' / (f['fixture_id'] + '.npz'); np.savez(p, x=x, w=w, expected=expected)
        fs.append(dict(f, file_sha256=digest_file(p), input_sha256=input_hash(x), weights_sha256=input_hash(w), oracle_sha256=input_hash(expected)))
        output = expected.astype(np.float32); digest = input_hash(output); relative = 'outputs/' + digest + '.npy'
        np.save(directory / relative, output, allow_pickle=False)
        outputs[f['fixture_id']] = dict(output_sha256=digest, output_file=relative, **errors(output, expected))
    jobs = spec.schedule(config, fs); byid = {f['fixture_id']: f for f in fs}
    atomic_json(directory / 'fixtures.json', fs); atomic_json(directory / 'schedule.json', jobs)
    atomic_json(directory / 'prior-fixtures.json', dict(seeds=[], hashes=[]))
    events = [dict(event='fixtures_frozen', fixtures_sha256=digest_file(directory / 'fixtures.json'),
                   schedule_sha256=digest_file(directory / 'schedule.json'))]
    tick = 10.; process = 1
    for section, block, configuration in dict.fromkeys((j['section'], j['block'], j['configuration']) for j in jobs):
        policy = spec.policy(configuration); name = f'{section}-{block}-{configuration}'; owner = dict(pid=process, tid=process); process += 1
        events.append(dict(event='governor_transition', section=section, block=block, configuration=configuration,
                           current=policy['governor'], original='ondemand'))
        env = dict(configuration=configuration, policy=policy, device='CPU', numeric_device='cpu',
            numerical_library_threads=NUMERICAL_LIBRARY_THREADS,
            gpu_initialization_ms=None, cpu_initialization_ms=1, openmp_environment=spec.environment(configuration),
            openmp_actual=spec.specification()['openmp'][configuration], observed=dict(governor=policy['governor']),
            probes=[dict(requested=n, team=n, threads=[dict(mask=(1 << i) if configuration == 'candidate' else 15) for i in range(n)]) for n in (1, 4)])
        events.append(dict(event='worker_ready', instance=name, configuration=configuration, device_kind='cpu', owner=owner, environment=env))
        warmed = set()

        def response(fid, label, started, ended, batch=False):
            count = 16 if batch else 1; rows = []
            for index in range(count):
                start = started + .25 + index * .125
                row = dict(fixture_id=fid, **spec.arm(byid[fid], label), **outputs[fid],
                    matches_warmup=True, unexpected_output_io=False, validation_errors=0,
                    started=start, computed=start + .0625, ended=start + .125,
                    worker_request_ms=125., request_ms=62.5, cpu_seconds=.0625)
                rows.append(dict(row, request_id=index) if batch else row)
            result = dict(requests=rows, correct=True, worker_batch_ms=count * 125.) if batch else rows[0]
            result.update(policy=policy, observer_ms=0., system_before=dict(governor=policy['governor']), system_after=dict(governor=policy['governor']))
            return dict(owner=owner, submitted=started + .125, received=started + .25,
                        ended=ended - .125, delivered=ended - .0625, result=result)

        for job in [j for j in jobs if (j['section'], j['block'], j['configuration']) == (section, block, configuration)]:
            fid = job['fixture_id']
            if fid not in warmed:
                for label in spec.CPU_ARMS:
                    events.append(dict(event='warmup', fixture_id=fid, label=label, instance=name, configuration=configuration,
                                       response=response(fid, label, tick, tick + 1)))
                    tick += 2
                warmed.add(fid)
            batch = job['label'].startswith('batch'); duration = (16. if batch else 1.) * (.75 if configuration == 'candidate' else 1.)
            events.append(dict(event='measurement', **job, instance=name, started=tick, ended=tick + duration,
                total_ms=duration * 1000, response=response(fid, job['label'], tick, tick + duration, batch)))
            tick += duration + 1
        events.append(dict(event='worker_release', instance=name, owner=owner, alive=False, forced=False))
        if section == 'pilot' and configuration == 'candidate':
            events.append(dict(event='pilot_gate', pilot_seconds=600., multiplier=spec.BLOCKS,
                               required_seconds=600. * spec.BLOCKS * 1.2, remaining_work_seconds=10000., fits=True))
    events.append(dict(event='measurement_complete'))
    (directory / spec.EVENTS).write_text(''.join(json.dumps(row) + '\n' for row in events))
    atomic_json(directory / 'outcome.json', dict(status='complete', stop_reason=None, elapsed_seconds=9990.))
    atomic_json(directory / 'runtime-finalization.json', dict(reserved_seconds=14400, charged_seconds=10000.,
        started_monotonic=0., finished_monotonic=10000., original_governor='ondemand', restored=True))
    atomic_json(directory / 'governor-restoration.json', dict(restored=True, current='ondemand'))
    (directory / 'telemetry.jsonl').write_text(''.join(json.dumps(dict(event='sample', cpu_temp_c=50.,
        throttle_flags=0, available_memory_bytes=1024**3, host_cpu_percent=50, phase='cpu', monotonic=t)) + '\n'
        for t in (1., 9999.)))
    atomic_json(directory / 'summary.json', spec.analyze(events, config))
    seal(directory)
    return directory


class DecisionTests(unittest.TestCase):
    def setUp(self):
        for name, value in (('BLOCKS', 4), ('CELLS', CELLS)):
            handle = patch.object(spec, name, value); handle.start(); self.addCleanup(handle.stop)
        self.config = dict(fixture_namespace='decisions')

    def checked(self, events):
        summary = spec.analyze(events, self.config)
        decisions([e for e in events if e.get('section') == 'main'], self.config, summary)
        return summary

    def test_equal_speed_rejected_and_improvement_accepted(self):
        self.assertFalse(self.checked(raw_events(self.config, 1.))['accepted'])
        self.assertTrue(self.checked(raw_events(self.config))['accepted'])

    def test_mixed_endpoints_rejected(self):
        events = raw_events(self.config)
        for row in events[:-1]:
            if row['configuration'] == 'candidate' and row['label'].startswith('batch'):
                row['total_ms'] = 16.
        self.assertFalse(self.checked(events)['accepted'])

    def test_shape_regression_and_unstable_controls_rejected(self):
        for mutation in ('shape', 'controls'):
            events = raw_events(self.config)
            for row in events[:-1]:
                if row['configuration'] != 'candidate':
                    continue
                if mutation == 'shape' and row['cell_id'] == CELLS[0]['cell_id']:
                    row['total_ms'] *= 1.5
                if mutation == 'controls' and row['label'].endswith('_b'):
                    row['total_ms'] *= 1.1
            self.assertFalse(self.checked(events)['accepted'])

    def test_independent_bootstrap_rejects_tampered_summary(self):
        events = raw_events(self.config); summary = spec.analyze(events, self.config)
        summary['metrics'][0]['speedup']['interval'][0] += .01
        with self.assertRaisesRegex(ValueError, 'bootstrap'):
            decisions([e for e in events if e.get('section') == 'main'], self.config, summary)

    def test_nonuniform_floating_point_measurements_agree(self):
        rng = np.random.default_rng(171)
        events = raw_events(self.config)
        for row in events[:-1]:
            row['total_ms'] *= float(rng.uniform(.99, 1.01))
        self.assertTrue(self.checked(events)['accepted'])


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        for name, value in (('BLOCKS', 2), ('CELLS', CELLS)):
            handle = patch.object(spec, name, value); handle.start(); self.addCleanup(handle.stop)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = archive(Path(self.tmp.name))

    def change_events(self, operation):
        path = self.directory / spec.EVENTS
        events = [json.loads(line) for line in path.read_text().splitlines()]
        operation(events)
        path.write_text(''.join(json.dumps(row) + '\n' for row in events)); seal(self.directory)

    def assert_rejected(self, pattern):
        result = audit(self.directory)
        self.assertFalse(result['passed'], result); self.assertRegex(result['error'], pattern)

    def test_complete_valid_archive_and_read_only_audit(self):
        before = digest_file(self.directory / 'checksums.json')
        result = audit(self.directory)
        self.assertTrue(result['passed'], result); self.assertTrue(result['accepted'])
        self.assertEqual(before, digest_file(self.directory / 'checksums.json'))

    def test_resealed_output_tamper_rejected(self):
        file = next((self.directory / 'outputs').iterdir()); output = np.load(file); output[0, 0, 0] += 1.; np.save(file, output)
        seal(self.directory); self.assert_rejected('output/verification')

    def test_resealed_wrong_oracle_rejected(self):
        fs = json.loads((self.directory / 'fixtures.json').read_text()); path = self.directory / 'fixtures' / (fs[0]['fixture_id'] + '.npz')
        with np.load(path) as arrays:
            x, w, expected = arrays['x'], arrays['w'], arrays['expected']
        expected[0, 0, 0] += 1.; np.savez(path, x=x, w=w, expected=expected)
        fs[0]['file_sha256'] = digest_file(path); atomic_json(self.directory / 'fixtures.json', fs); seal(self.directory)
        self.assert_rejected('FP64 fixture')

    def test_environment_leak_rejected_even_when_resealed(self):
        def mutate(events):
            next(e for e in events if e['event'] == 'worker_ready')['environment']['openmp_actual']['OMP_NUM_THREADS'] = '4'
        self.change_events(mutate); self.assert_rejected('environment contamination')

    def test_nested_batch_tamper_rejected_even_when_resealed(self):
        def mutate(events):
            event = next(e for e in events if e['event'] == 'measurement' and e['label'].startswith('batch'))
            event['response']['result']['requests'][7]['output_sha256'] = 'tampered'
        self.change_events(mutate); self.assert_rejected('output/verification')

    def test_missing_measurement_and_wrong_order_rejected(self):
        def mutate(events):
            index = next(i for i, e in enumerate(events) if e['event'] == 'measurement'); events.pop(index)
        self.change_events(mutate); self.assert_rejected('Incomplete measurement')

    def test_gpu_initialization_rejected(self):
        def mutate(events):
            next(e for e in events if e['event'] == 'worker_ready')['environment']['gpu_initialization_ms'] = 1.
        self.change_events(mutate); self.assert_rejected('Non-CPU initialization')

    def test_budget_tamper_rejected(self):
        path = self.directory / 'runtime-finalization.json'; value = json.loads(path.read_text()); value['charged_seconds'] += 1
        atomic_json(path, value); seal(self.directory); self.assert_rejected('attempt budget')

    def test_governor_restoration_failure_rejected(self):
        atomic_json(self.directory / 'governor-restoration.json', dict(restored=False, current='performance'))
        seal(self.directory); self.assert_rejected('Governor restoration failed')

    def test_native_runtime_mismatch_rejected(self):
        path = self.directory / 'cpu-runtime.json'; value = json.loads(path.read_text())
        value['native_artifacts']['libaffinity.so'] = 'a' * 64
        atomic_json(path, value); seal(self.directory); self.assert_rejected('runtime native artifact differs')

    def test_warmup_must_precede_matching_measurement(self):
        def mutate(events):
            index = next(i for i, e in enumerate(events) if e['event'] == 'warmup'); events.pop(index)
        self.change_events(mutate); self.assert_rejected('Warmup/output identity')

    def test_clock_tamper_rejected(self):
        def mutate(events):
            event = next(e for e in events if e['event'] == 'measurement')
            event['response']['delivered'] = event['ended'] + 1
        self.change_events(mutate); self.assert_rejected('Clock/transport ordering')

    def test_reordered_measurements_rejected(self):
        def mutate(events):
            positions = [i for i, e in enumerate(events) if e['event'] == 'measurement'][:2]
            events[positions[0]], events[positions[1]] = events[positions[1]], events[positions[0]]
        self.change_events(mutate); self.assert_rejected('Measurement order')

    def test_resealed_parent_selection_tamper_rejected(self):
        provenance_path = self.directory / 'provenance.json'; provenance = json.loads(provenance_path.read_text())
        parent = Path(provenance['cpu_confirmation']['path']); source = Path(provenance['source_campaign']['path'])
        atomic_json(parent / 'summary.json', dict(complete=True, accepted=True, selected=7)); seal(parent)
        provenance['cpu_confirmation']['checksums_sha256'] = digest_file(parent / 'checksums.json')
        campaign = json.loads((source / 'campaign.json').read_text())
        campaign['attempts'][0]['checksums_sha256'] = provenance['cpu_confirmation']['checksums_sha256']
        atomic_json(source / 'campaign.json', campaign); seal(source)
        provenance['source_campaign']['checksums_sha256'] = digest_file(source / 'checksums.json')
        atomic_json(provenance_path, provenance)
        frozen = json.loads((self.directory / 'freeze.json').read_text()); frozen['provenance_sha256'] = digest_file(provenance_path)
        atomic_json(self.directory / 'freeze.json', frozen); seal(self.directory)
        self.assert_rejected('confirmation did not qualify')

    def test_empty_seal_cannot_bypass_coverage(self):
        atomic_json(self.directory / 'checksums.json', {}); self.assert_rejected('seal coverage')

    def test_report_refuses_sealed_archive_before_writing(self):
        before = digest_file(self.directory / 'summary.json')
        with self.assertRaisesRegex(ValueError, 'immutable'):
            build_report(self.directory)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            plots(self.directory)
        self.assertEqual(before, digest_file(self.directory / 'summary.json'))

    def test_report_contains_raw_calls_and_explicit_plot_status(self):
        (self.directory / 'checksums.json').unlink()
        summary = build_report(self.directory, charts=False)
        self.assertTrue(summary['accepted']); self.assertFalse(summary['charts']['available'])
        csv = (self.directory / 'measurements.csv').read_text().splitlines()
        self.assertEqual(len(csv) - 1, 3 * 2 * len(CELLS) * 3 * 34)
        self.assertIn('Energy was not measured', (self.directory / 'report.html').read_text())
        self.assertTrue(audit(self.directory, require_seal=False)['passed'])

    def test_missing_plot_dependency_keeps_reviewable_report(self):
        (self.directory / 'checksums.json').unlink()
        with patch('efficiency.cpu_compare_report.subprocess.run', side_effect=FileNotFoundError('plot interpreter')):
            summary = build_report(self.directory)
        self.assertTrue(summary['accepted']); self.assertFalse(summary['charts']['available'])
        self.assertIn('Plotting unavailable', (self.directory / 'report.html').read_text())


if __name__ == '__main__':
    unittest.main()
