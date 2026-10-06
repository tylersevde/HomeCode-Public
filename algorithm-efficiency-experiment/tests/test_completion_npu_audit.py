"""A complete simulated archive proves raw evidence gates survive resealing."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficiency import completion_npu_spec as spec
from efficiency.completion_npu_audit import audit
from efficiency.common import atomic_json, digest_file, digest_value, read_jsonl
from efficiency.research_report import seal
from efficiency.study_spec import BINDINGS, MODEL_HASHES, PARAMETERS
from tests.test_completion_npu import TEMPLATE, config, raw_worker


def read(path):
    return json.loads(Path(path).read_text())


def write_events(path, events):
    path.write_text(''.join(json.dumps(e) + '\n' for e in events))


def archived_diagnostic(directory):
    c = config(); c['parents'] = {}; c['model'] = '/verified-model.hef'; c['original_governor'] = 'ondemand'
    atomic_json(directory / 'config.json', c)
    atomic_json(directory / 'protocol.json', spec.specification(c))
    source = directory / 'source/efficiency/frozen.py'
    source.parent.mkdir(parents=True)
    source.write_text('# Fixed test implementation\n')
    source_hashes = {'efficiency/frozen.py': digest_file(source)}
    atomic_json(directory / 'manifest.json', dict(config=c, source_sha256=source_hashes,
        model_sha256=MODEL_HASHES['llama'], initial_throttle_flags=0))
    atomic_json(directory / 'prior-fixtures.json', dict(seeds=[], hashes=[]))
    rationale = directory / 'historical-rationale.txt'; rationale.write_text('Frozen 2x2 rationale\n')
    atomic_json(directory / 'freeze.json', dict(source_sha256=source_hashes,
        config_sha256=digest_file(directory / 'config.json'), protocol_sha256=digest_file(directory / 'protocol.json'),
        prior_fixtures_sha256=digest_file(directory / 'prior-fixtures.json'), model_sha256=MODEL_HASHES['llama'],
        revised_policy_sha256=None, parents={}, historical_rationale=dict(sha256=digest_file(rationale))))
    worker = raw_worker()
    fs, sizes = [], {}
    for entry in spec.fixtures(c):
        f = spec.table(entry, 4); fs.append(f)
        prompt = worker.render(f['initial_messages'])
        ids = list(range(len(prompt) // 4))
        sizes[f['fixture_id']] = dict(count=4, token_ids=ids, counts=[[count, len(ids) if count == 4 else 9999] for count in range(4, 301)])
    atomic_json(directory / 'fixtures.json', fs)
    atomic_json(directory / 'sizing.json', sizes)
    jobs = spec.schedule(c, fs); atomic_json(directory / 'schedule.json', jobs)
    byid = {f['fixture_id']: f for f in fs}
    env = dict(hailort_version='5.1.1', model_sha256=MODEL_HASHES['llama'], model_id='llama', parameters=PARAMETERS,
        template_bindings=BINDINGS, native_stops=spec.NATIVE_STOPS, prompt_template=TEMPLATE,
        template_probe=worker.render([dict(role='system', content='Test'), dict(role='user', content='Test')]),
        recovery_token_ids=[], experiment_limit_tokens=1792, context_policy='full_rebuild_each_turn')
    events = [dict(event='worker_ready', instance='llama-completion', owner=worker.owner, environment=env),
        dict(event='fixtures_frozen', fixtures_sha256=digest_file(directory / 'fixtures.json'),
             schedule_sha256=digest_file(directory / 'schedule.json'), revised_policy_sha256=None)]
    for section in ('pilot', 'main'):
        for job in (j for j in jobs if j['section'] == section):
            arm = {k: job[k] for k in ('label', 'table_mode', 'history_mode', 'condition', 'system', 'diagnostic_only')}
            result = worker.perform(dict(operation='completion_dialogue', arm=arm, fixture=byid[job['fixture_id']], deadline=float('inf')))
            response = dict(owner=worker.owner, submitted=1, received=1, ended=2, delivered=2, result=result)
            events.append(dict(event='measurement', **job, instance='llama-completion', started=1, ended=2, total_ms=1000, response=response))
        if section == 'pilot':
            events.append(dict(event='pilot_gate', multiplier=8, pilot_seconds=1, required_seconds=9.6, remaining_work_seconds=1000, fits=True))
    events.extend([dict(event='native_stops_restored', instance='llama-completion',
                        response=dict(owner=worker.owner, result=dict(stops=spec.NATIVE_STOPS, context_tokens=0))),
                   dict(event='worker_release', instance='llama-completion', owner=worker.owner, forced=False, alive=False),
                   dict(event='measurement_complete')])
    write_events(directory / spec.EVENTS, events)
    atomic_json(directory / 'summary.json', spec.analyze(events, c))
    atomic_json(directory / 'outcome.json', dict(status='complete', elapsed_seconds=2000))
    atomic_json(directory / 'runtime-finalization.json', dict(started_monotonic=0, finished_monotonic=2010,
        charged_seconds=2010, reserved_seconds=c['reserved_seconds'], original_governor='ondemand', restored=True))
    atomic_json(directory / 'governor-restoration.json', dict(current='ondemand', restored=True))
    write_events(directory / 'telemetry.jsonl', [dict(event='sample', phase='hat', stop_reason=None, cpu_temp_c=42,
        monotonic=stamp, throttle_flags=0, available_memory_bytes=1024 ** 3, sensor_alive=True,
        hat_sample_age_s=1, hat_ts0_c=40, hat_ts1_c=40) for stamp in (0.5, 2000)])
    seal(directory)


class ArchiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        archived_diagnostic(self.directory)
        # The shared receipt machinery has separate integration tests; these tests
        # independently exercise all NPU raw evidence and local runtime checks.
        budget = patch('efficiency.completion_npu_audit.audit_budget', return_value=dict(passed=True))
        budget.start(); self.addCleanup(budget.stop)

    def test_complete_diagnostic_archive_has_no_qualification(self):
        result = audit(self.directory)
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['measurements'], 108)
        self.assertFalse(read(self.directory / 'summary.json')['accepted'])

    def test_resealed_raw_answer_or_history_tamper_is_rejected(self):
        path = self.directory / spec.EVENTS
        rows, _ = read_jsonl(path)
        event = next(e for e in rows if e['event'] == 'measurement' and e['label'] == 'full_table_actual_history')
        event['response']['result']['rows'][1]['messages'][2]['content'] = 'CPU corrected this reply'
        write_events(path, rows); seal(self.directory)
        result = audit(self.directory)
        self.assertFalse(result['passed'])
        self.assertIn('Prompt reconstruction', result['error'])

    def test_resealed_omission_cannot_pass(self):
        path = self.directory / spec.EVENTS
        rows, _ = read_jsonl(path)
        rows.remove(next(e for e in rows if e['event'] == 'measurement'))
        write_events(path, rows); seal(self.directory)
        self.assertIn('matrix incomplete', audit(self.directory)['error'])

    def test_resealed_simplified_arm_promotion_is_rejected(self):
        path = self.directory / 'summary.json'; value = read(path)
        value['accepted'] = True; value['scientific_qualification'] = True
        atomic_json(path, value); seal(self.directory)
        self.assertIn('diagnostic role', audit(self.directory)['error'])

    def test_resealed_unsafe_telemetry_or_failed_release_is_rejected(self):
        path = self.directory / spec.EVENTS
        rows, _ = read_jsonl(path)
        next(e for e in rows if e['event'] == 'worker_release')['forced'] = True
        write_events(path, rows); seal(self.directory)
        self.assertIn('Unclean', audit(self.directory)['error'])

    def test_resealed_budget_inflation_cannot_pass(self):
        path = self.directory / 'config.json'; value = read(path)
        value['development_charged_seconds'] = 1000
        atomic_json(path, value)
        freeze = read(self.directory / 'freeze.json'); freeze['config_sha256'] = digest_file(path)
        atomic_json(self.directory / 'freeze.json', freeze)
        manifest = read(self.directory / 'manifest.json'); manifest['config'] = value
        atomic_json(self.directory / 'manifest.json', manifest); seal(self.directory)
        self.assertIn('share the development allowance', audit(self.directory)['error'])

    def test_resealed_inconsistent_runtime_charge_is_rejected(self):
        path = self.directory / 'runtime-finalization.json'; value = read(path)
        value['charged_seconds'] -= 1
        atomic_json(path, value); seal(self.directory)
        self.assertIn('attempt budget/restoration', audit(self.directory)['error'])

    def test_resealed_telemetry_outside_attempt_is_rejected(self):
        path = self.directory / 'telemetry.jsonl'; rows, _ = read_jsonl(path)
        rows[-1]['monotonic'] = 9000
        write_events(path, rows); seal(self.directory)
        self.assertIn('cover measurement interval', audit(self.directory)['error'])


if __name__ == '__main__':
    unittest.main()
