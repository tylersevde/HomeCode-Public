"""Supervised coordination experiment with a frozen design and excluded pilot."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .attention_native import fixture, oracle
from .common import atomic_json, digest_file, digest_value, emit, progress, wait_cool
from .coordination_spec import (EVENTS, advice_cases, fixture_specs, pilot_required_seconds,
                                schedule, specification)
from .coordination_workers import Worker, run_condition
from .feedback_spec import input_hash
from .hybrid import verify_artifacts


def prepare_source(directory, config, inventory):
    source = Path(config['source_run']).resolve()
    verified = verify_artifacts(source)
    read = lambda name: json.loads((source/name).read_text())
    required = ('manifest.json', 'summary.json', 'outcome.json', 'concurrency-spec.json',
                'feedback-environment.json', 'native-build/build.json', 'fixtures.json')
    if not all(n in verified for n in required):
        raise ValueError('Source requires completed, checksummed attention artifacts')
    if not read('summary.json').get('complete') or read('outcome.json')['status']!='complete':
        raise ValueError('Source attention experiment is incomplete')
    old = read('manifest.json')
    for key in ('model_sha256', 'hailort_cli', 'archive_sha256', 'cpu_model'):
        if old[key] != inventory[key]:
            raise ValueError(f'Source identity differs: {key}')
    for relative, checksum in old['source_sha256'].items():
        if relative.startswith('native/attention/') or relative=='efficiency/attention_native.py':
            if inventory['source_sha256'].get(relative) != checksum:
                raise ValueError(f'Numerical implementation differs: {relative}')
    selected = read('concurrency-spec.json')
    if (selected['cpu_backend'], selected['gpu_candidate']) != ('native4', 'C6'):
        raise ValueError('Frozen CPU/GPU controls differ from the plan')
    frozen = directory/'source-run'; frozen.mkdir()
    for name in (*required, 'checksums.json'):
        target = frozen/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    shutil.copytree(source/'native-build', directory/'native-build')
    atomic_json(directory/'source-provenance.json', dict(source_run=str(source),
        verified_artifacts=len(verified), source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={n:digest_file(frozen/n) for n in (*required,'checksums.json')}))
    atomic_json(directory/'protocol.json', specification())
    atomic_json(directory/'advice-cases.json', advice_cases())
    atomic_json(directory/'schedule.json', schedule())


class Budget:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        status = json.loads((self.directory/'status.json').read_text())
        self.deadline = status['monotonic']-status['elapsed_seconds']+config['max_seconds']
        self.work_deadline = self.deadline-45
        self.last_progress = 0
        self.stage = 'fixtures'

    def check(self):
        now = time.monotonic()
        if now >= self.work_deadline:
            raise TimeoutError('Coordination work deadline reached; cleanup reserved')
        if now-self.last_progress >= 1:
            status = json.loads((self.directory/'status.json').read_text())
            if status.get('stop_reason'):
                raise RuntimeError(status['stop_reason'])
            if not 0 <= now-status['monotonic'] <= 8:
                raise RuntimeError('Supervisor telemetry stale')
            progress(self.directory, 'hat', self.stage)
            self.last_progress = now


def freeze_fixtures(directory, budget):
    (directory/'fixtures').mkdir(); (directory/'outputs').mkdir()
    old = json.loads((directory/'source-run/fixtures.json').read_text())
    previous = {r['seed'] for r in old}
    metadata = []
    for row in fixture_specs():
        budget.check()
        if row['seed'] in previous:
            raise ValueError('New fixture reuses a previous seed')
        x, w = fixture(row['n'], row['d'], row['b'], row['seed'])
        expected = oracle(x, w)
        path = directory/'fixtures'/(row['fixture_id']+'.npz')
        np.savez(path, x=x, w=w, expected=expected)
        metadata.append(dict(**row, file_sha256=digest_file(path), input_sha256=input_hash(x),
                             weights_sha256=input_hash(w), oracle_sha256=input_hash(expected)))
    atomic_json(directory/'fixtures.json', metadata)


def verify_environment(directory, environment):
    prior = json.loads((directory/'source-run/feedback-environment.json').read_text())
    keys = ('hailort_version', 'prompt_template', 'stop_tokens', 'capacity_tokens',
            'model_defaults', 'parameters', 'experiment_limit_tokens')
    if any(prior[k] != environment[k] for k in keys):
        raise ValueError('Model generation environment differs from frozen source')
    path = directory/'adviser-environment.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved[k] != environment[k] for k in keys):
            raise ValueError('Model environment changed between workers')
    else:
        atomic_json(path, environment)


def start_worker(stack, directory, config, budget, architecture, kind, block_id):
    worker = Worker(architecture, kind, directory, dict(config, block_id=block_id), budget.check)
    def close():
        try:
            worker.close()
        finally:
            emit(directory/EVENTS, 'worker_release', block_id=block_id,
                 **getattr(worker, 'release', dict(worker=kind, alive=True)))
    stack.callback(close)
    emit(directory/EVENTS, 'worker_ready', block_id=block_id, architecture=architecture,
         worker=kind, **worker.ready)
    if kind=='hat':
        verify_environment(directory, worker.ready['environment'])
    return worker


def adviser_checks(directory, config, budget, cases):
    budget.stage = 'adviser_contract'
    wait_cool(directory)
    with ExitStack() as stack:
        hat = start_worker(stack, directory, config, budget, 'thread', 'hat', 'adviser_contract')
        for split in ('development', 'holdout'):
            if split=='holdout':
                atomic_json(directory/'adviser-freeze.json', dict(
                    cases_sha256=digest_file(directory/'advice-cases.json'),
                    environment_sha256=digest_file(directory/'adviser-environment.json'),
                    prompt_sha256=digest_value(cases[0]['messages'][0]),
                    heldout_started=False, revision_count=0))
                emit(directory/EVENTS, 'adviser_frozen', cases_sha256=digest_file(directory/'advice-cases.json'))
            for case in (c for c in cases if c['split']==split):
                budget.check()
                result = hat.call('advice', case=case)
                emit(directory/EVENTS, 'advice_check', split=split, case_id=case['case_id'], response=result)


def run_block(directory, config, budget, block, cases):
    started = time.monotonic()
    budget.stage = block['block_id']
    budget.check(); wait_cool(directory)
    emit(directory/EVENTS, 'block_start', **block, started=started)
    with ExitStack() as stack:
        numeric = start_worker(stack, directory, config, budget, block['architecture'], 'numeric', block['block_id'])
        hat = start_worker(stack, directory, config, budget, block['architecture'], 'hat', block['block_id'])
        numeric.call('load', fixture_ids=block['fixture_ids'])
        warm = numeric.call('warm', fixture_ids=block['fixture_ids'], block_id=block['block_id'])
        emit(directory/EVENTS, 'numeric_warmup', block_id=block['block_id'], response=warm)
        warm = hat.call('advice', case=cases['pilot-00'])
        emit(directory/EVENTS, 'adviser_warmup', block_id=block['block_id'], response=warm)
        for condition in block['conditions']:
            budget.check(); wait_cool(directory)
            result = run_condition(numeric, hat, condition, block['fixture_ids'], cases[block['case_id']],
                                   budget.work_deadline)
            emit(directory/EVENTS, 'condition', block_id=block['block_id'], stage=block['stage'],
                 repeat=block['repeat'], architecture=block['architecture'], condition=condition,
                 case_id=block['case_id'], **result)
    row = dict(**block, elapsed_seconds=time.monotonic()-started)
    emit(directory/EVENTS, 'block_end', **row)
    return row


def run(directory, config):
    directory = Path(directory)
    budget = Budget(directory, config)
    try:
        freeze_fixtures(directory, budget)
        cases = json.loads((directory/'advice-cases.json').read_text())
        adviser_checks(directory, config, budget, cases)
        case_map = {c['case_id']:c for c in cases}
        blocks = json.loads((directory/'schedule.json').read_text())
        pilot = [run_block(directory, config, budget, b, case_map) for b in blocks if b['stage']=='pilot']
        required = pilot_required_seconds(pilot)
        remaining = budget.deadline-time.monotonic()
        gate = dict(required_seconds=required, remaining_seconds=remaining, fits=required<=remaining,
                    repeats=6, conditions=48, margin=1.20, cleanup_reserve_seconds=45)
        atomic_json(directory/'pilot-gate.json', gate)
        emit(directory/EVENTS, 'pilot_gate', **gate)
        if not gate['fits']:
            raise TimeoutError('Full frozen matrix does not fit remaining budget; no repetitions removed')
        for block in blocks:
            if block['stage']=='main':
                run_block(directory, config, budget, block, case_map)
        emit(directory/EVENTS, 'complete', conditions=48, numerical_requests=1152)
    except BaseException as exc:
        emit(directory/EVENTS, 'incomplete', error=f'{type(exc).__name__}: {exc}')
        raise
