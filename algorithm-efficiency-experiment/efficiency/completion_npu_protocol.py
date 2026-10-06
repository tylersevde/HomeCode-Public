"""Bounded execution; diagnostics never reopen a historical qualification gate."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

from .common import ROOT, atomic_json, digest_file, digest_value, emit, wait_cool
from .coordination_workers import Worker
from .hybrid import verify_artifacts
from .research_campaign import register_fixture
from .study_protocol import Budget
from .study_spec import LLAMA, MODEL_HASHES
from .completion_npu_spec import (EVENTS, BLOCKS, specification, validate_config,
                                  fixtures, schedule, table, policy)
from .completion_npu_worker import CompletionContext


def read(path):
    return json.loads(Path(path).read_text())


def prepare_source(directory, config, inventory):
    directory = Path(directory)
    selected = validate_config(config, budget=True)
    if Path(config['model']).resolve() != LLAMA.resolve() or digest_file(LLAMA) != MODEL_HASHES['llama']:
        raise ValueError('Verified installed Llama identity differs')
    if inventory['model_sha256'] != MODEL_HASHES['llama']:
        raise ValueError('Inventoried NPU model differs')
    if read(directory / 'protocol.json') != specification(config):
        raise ValueError('Protocol must be frozen before NPU source preparation')
    expected = set() if config['stage'] == 'diagnostic' else {'diagnostic'}
    if config['stage'] == 'confirm':
        expected.add('develop')
    if set(config.get('parents', {})) != expected:
        raise ValueError('NPU parent set differs')
    parents = {}
    from .completion_npu_audit import audit
    for stage, name in config.get('parents', {}).items():
        path = Path(name).resolve()
        verify_artifacts(path)
        if not audit(path)['passed']:
            raise ValueError('Independent NPU parent audit failed')
        parent_config, summary = read(path / 'config.json'), read(path / 'summary.json')
        if parent_config['stage'] != stage or not summary['complete']:
            raise ValueError('NPU parent stage differs')
        checksum = digest_file(path / 'checksums.json')
        if stage == 'diagnostic':
            if summary['selected'] is not None or summary['accepted'] or checksum != selected['diagnostic_checksums_sha256']:
                raise ValueError('Policy does not identify this completed diagnostic')
        elif summary['selected'] != selected or not summary['metrics'][1]['qualified']:
            raise ValueError('Qualification development did not select this policy')
        elif read(path / 'manifest.json')['source_sha256'] != inventory['source_sha256']:
            raise ValueError('Source changed after qualification development')
        parents[stage] = dict(path=str(path), checksums_sha256=checksum)
    if selected is not None:
        canonical = ROOT / 'runs/.completion-npu-policy.json'
        if read(canonical) != selected:
            raise ValueError('The single mission policy must be frozen before qualification')
        atomic_json(directory / 'npu-policy.json', selected)
    # Historical rationale is a pinned input, never an editable experiment parent.
    rationale = ROOT / 'campaigns/reliability-20261003T152506Z/NEXT_EXPERIMENT.txt'
    historical = dict(path=str(rationale), sha256=digest_file(rationale))
    (directory / 'historical-rationale.txt').write_bytes(rationale.read_bytes())
    atomic_json(directory / 'freeze.json', dict(version=specification(config)['version'],
        source_sha256=inventory['source_sha256'], config_sha256=digest_file(directory / 'config.json'),
        protocol_sha256=digest_file(directory / 'protocol.json'), parents=parents,
        prior_fixtures_sha256=digest_file(directory / 'prior-fixtures.json'),
        revised_policy_sha256=digest_value(selected) if selected else None,
        model_sha256=MODEL_HASHES['llama'], historical_rationale=historical))


@contextmanager
def worker(directory, config, budget):
    budget.stage = 'llama-completion-load'
    budget.check()
    value = Worker('process', 'hat', directory, dict(config, model_id='llama'), budget.check,
                   factory=CompletionContext)
    emit(directory / EVENTS, 'worker_ready', instance='llama-completion', **value.ready)
    try:
        yield value
    finally:
        try:
            emit(directory / EVENTS, 'native_stops_restored', instance='llama-completion', response=value.call('restore'))
        finally:
            try:
                value.close()
            finally:
                emit(directory / EVENTS, 'worker_release', instance='llama-completion',
                     **getattr(value, 'release', dict(alive=True, forced=True)))


def pilot_gate(directory, config, budget, started):
    seconds = time.monotonic() - started
    remaining = budget.work_deadline - time.monotonic()
    multiplier = BLOCKS[config['stage']]
    required = seconds * multiplier * 1.2
    emit(directory / EVENTS, 'pilot_gate', pilot_seconds=seconds, multiplier=multiplier,
         required_seconds=required, remaining_work_seconds=remaining, fits=required <= remaining)
    if required > remaining:
        raise TimeoutError('Frozen NPU matrix cannot fit remaining allowance; no coverage reduction')


def run(directory, config):
    directory = Path(directory)
    validate_config(config, budget=True)
    budget = Budget(directory, config)
    try:
        with worker(directory, config, budget) as device:
            fs, sized = [], {}
            for spec in fixtures(config):
                budget.stage = 'sizing-' + spec['fixture_id']
                budget.check()
                entry = device.call('size', spec=spec)['result']
                sized[spec['fixture_id']] = entry
                fixture = table(spec, entry['count'])
                register_fixture(config['campaign'], 'npu-' + config['stage'], fixture['fixture_id'],
                                 fixture['seed'], fixture['table_sha256'])
                fs.append(fixture)
            jobs = schedule(config, fs)
            atomic_json(directory / 'sizing.json', sized)
            atomic_json(directory / 'fixtures.json', fs)
            atomic_json(directory / 'schedule.json', jobs)
            emit(directory / EVENTS, 'fixtures_frozen', fixtures_sha256=digest_file(directory / 'fixtures.json'),
                 schedule_sha256=digest_file(directory / 'schedule.json'),
                 revised_policy_sha256=digest_value(policy(config)) if policy(config) else None)
            byid = {f['fixture_id']: f for f in fs}
            for section in ('pilot', 'main'):
                started = time.monotonic()
                for job in (j for j in jobs if j['section'] == section):
                    budget.check()
                    wait_cool(directory)
                    budget.stage = job['fixture_id'] + '-' + job['label']
                    budget.check()
                    arm = {k: job[k] for k in ('label', 'table_mode', 'history_mode', 'condition', 'system', 'diagnostic_only')}
                    tick = time.monotonic()
                    response = device.call('completion_dialogue', fixture=byid[job['fixture_id']], arm=arm,
                                           deadline=budget.work_deadline)
                    ended = time.monotonic()
                    emit(directory / EVENTS, 'measurement', **job, instance='llama-completion',
                         started=tick, ended=ended, total_ms=(ended - tick) * 1000, response=response)
                if section == 'pilot':
                    pilot_gate(directory, config, budget, started)
        emit(directory / EVENTS, 'measurement_complete')
    except Exception as exc:
        emit(directory / EVENTS, 'failure', error=f'{type(exc).__name__}: {exc}')
        raise
