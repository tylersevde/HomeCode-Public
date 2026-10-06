"""Bounded GPU crossover execution using the existing numerical implementations."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .attention_native import fixture, oracle
from .common import atomic_json, digest_file, emit, wait_cool
from .coordination_workers import Worker
from .cpu_compare_spec import environment
from .cpu_policy import NUMERICAL_LIBRARY_THREADS
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .refine_governor import request
from .refine_spec import POLICIES
from .reliability_worker import ReliableNumeric
from .research_campaign import register_fixture
from .study_protocol import Budget, controls
from . import completion_gpu_spec as design


def read(path):
    return json.loads(Path(path).read_text())


def prepare_source(directory, config, inventory):
    directory = Path(directory)
    if not 120 < config['max_seconds'] <= design.MAX_SECONDS:
        raise ValueError('GPU stage exceeds frozen two-hour allowance')
    parent = Path(config['parents']['gpu-confirmed'])
    verify_artifacts(parent)
    summary = read(parent/'summary.json')
    if not (read(parent/'validation.json')['passed'] and summary['complete'] and summary['accepted']
            and read(parent/'config.json')['reliability_stage'] == 'gpu-confirm' and summary['selected'] == 5):
        raise ValueError('Audited historical stream64 confirmation required')
    previous = read(parent/'manifest.json')['source_sha256']
    for name, checksum in previous.items():
        if name.startswith('native/attention/') or name == 'efficiency/attention_native.py':
            if inventory['source_sha256'].get(name) != checksum:
                raise ValueError('Frozen numerical source changed: '+name)
    parents = {'gpu-confirmed': dict(path=str(parent.resolve()), checksums_sha256=digest_file(parent/'checksums.json'))}
    if config['stage'] == 'confirm':
        development = Path(config['parents']['gpu-develop'])
        from .completion_gpu_audit import audit
        if not audit(development)['passed']:
            raise ValueError('Development parent failed independent audit')
        selected = read(development/'summary.json')['selected']
        if selected is None or selected != config.get('selected'):
            raise ValueError('Confirmation routing differs from qualified development')
        if read(development/'manifest.json')['source_sha256'] != inventory['source_sha256']:
            raise ValueError('Implementation changed after development selection')
        if read(development/'config.json')['fixture_namespace'] == config['fixture_namespace']:
            raise ValueError('Confirmation must use a new fixture namespace')
        parents['gpu-develop'] = dict(path=str(development.resolve()), checksums_sha256=digest_file(development/'checksums.json'))
    shutil.copytree(parent/'native-build', directory/'native-build')
    atomic_json(directory/'gpu-provenance.json', parents)
    atomic_json(directory/'freeze.json', dict(source_sha256=inventory['source_sha256'],
        config_sha256=digest_file(directory/'config.json'), protocol_sha256=digest_file(directory/'protocol.json'),
        prior_fixtures_sha256=digest_file(directory/'prior-fixtures.json'),
        provenance_sha256=digest_file(directory/'gpu-provenance.json'),
        native_build_sha256=digest_file(directory/'native-build/build.json')))


class CrossoverNumeric(ReliableNumeric):
    def __init__(self, directory, config, stack):
        actual = {k: v for k, v in os.environ.items() if k.startswith(('OMP_', 'GOMP_'))}
        expected = {k: v for k, v in environment('candidate').items() if v is not None}
        if actual != expected or {k: os.environ.get(k) for k in NUMERICAL_LIBRARY_THREADS} != NUMERICAL_LIBRARY_THREADS:
            raise RuntimeError('GPU comparison numerical environment differs')
        super().__init__(directory, dict(config, numeric_device='both', policy=5), stack)
        self.environment.update(openmp_actual=actual, numerical_library_threads=dict(NUMERICAL_LIBRARY_THREADS))


@contextmanager
def worker(directory, config, budget):
    budget.stage = 'gpu-crossover-owner'; budget.check()
    value = Worker('process', 'numeric', directory, dict(config, block_id='gpu-crossover', numeric_device='both', policy=5),
                   budget.check, factory=CrossoverNumeric, environment=environment('candidate'))
    emit(directory/design.EVENTS, 'worker_ready', instance='gpu-crossover', **value.ready)
    try:
        yield value
    finally:
        try:
            value.close()
        finally:
            emit(directory/design.EVENTS, 'worker_release', instance='gpu-crossover',
                 **getattr(value, 'release', dict(alive=True, forced=True)))


def pilot_gate(directory, config, budget, started):
    elapsed = time.monotonic()-started
    required = elapsed*design.BLOCKS[config['stage']]*1.2
    remaining = budget.work_deadline-time.monotonic()
    emit(directory/design.EVENTS, 'pilot_gate', pilot_seconds=elapsed, multiplier=design.BLOCKS[config['stage']],
         required_seconds=required, remaining_work_seconds=remaining, fits=required <= remaining)
    if required > remaining:
        raise TimeoutError('Frozen GPU matrix cannot fit remaining stage allowance')


def run(directory, config):
    directory = Path(directory)
    design.spec(config)
    if not 120 < config['max_seconds'] <= design.MAX_SECONDS:
        raise ValueError('Invalid GPU stage allowance')
    budget = Budget(directory, config)
    for name in ('fixtures', 'outputs', 'controls'):
        (directory/name).mkdir()
    try:
        budget.check()
        emit(directory/design.EVENTS, 'governor_transition', **request(config['governor_socket'], POLICIES[5]['governor']))
        controls(directory, budget)
        emit(directory/design.EVENTS, 'correctness_complete', checks=len(read(directory/'correctness.json')))
        fs = []
        for item in design.fixtures(config):
            budget.stage = 'fixture-'+item['fixture_id']; budget.check()
            x, weights = fixture(item['n'], item['d'], item['b'], item['seed'])
            expected = oracle(x, weights)
            register_fixture(config['campaign'], 'completion-gpu-'+config['stage'], item['fixture_id'], item['seed'], input_hash(x))
            path = directory/'fixtures'/(item['fixture_id']+'.npz')
            np.savez(path, x=x, w=weights, expected=expected)
            fs.append(dict(item, file_sha256=digest_file(path), input_sha256=input_hash(x),
                           weights_sha256=input_hash(weights), oracle_sha256=input_hash(expected)))
            atomic_json(directory/'fixtures.json', fs)
        jobs = design.schedule(config, fs)
        atomic_json(directory/'schedule.json', jobs)
        emit(directory/design.EVENTS, 'fixtures_frozen', fixtures_sha256=digest_file(directory/'fixtures.json'),
             schedule_sha256=digest_file(directory/'schedule.json'))
        byid = {f['fixture_id']: f for f in fs}
        with worker(directory, config, budget) as device:
            for section in ('pilot', 'main'):
                started = time.monotonic(); current = None
                for job in (j for j in jobs if j['section'] == section):
                    budget.stage = job['fixture_id']; budget.check()
                    fid = job['fixture_id']; cell = byid[fid]
                    if fid != current:
                        wait_cool(directory); budget.check(); current = fid
                        device.call('load', fixture_id=fid)
                        for label in design.LABELS:
                            response = device.call('warm', fixture_id=fid, arm=design.arm(cell, label))
                            emit(directory/design.EVENTS, 'warmup', fixture_id=fid, label=label,
                                 instance='gpu-crossover', response=response)
                            if not response['result']['correct'] or response['result']['validation_errors']:
                                raise RuntimeError('GPU crossover warmup numerical failure')
                    tick = time.monotonic()
                    response = device.call('measure', fixture_id=fid, arm=design.arm(cell, job['label']))
                    end = time.monotonic()
                    emit(directory/design.EVENTS, 'measurement', **job, instance='gpu-crossover',
                         started=tick, ended=end, total_ms=(end-tick)*1000, response=response)
                    if not design.valid_result(dict(job, response=response), cell):
                        raise RuntimeError('GPU crossover measured numerical failure')
                if section == 'pilot':
                    pilot_gate(directory, config, budget, started)
        emit(directory/design.EVENTS, 'measurement_complete')
    except BaseException as exc:
        emit(directory/design.EVENTS, 'failure', error=f'{type(exc).__name__}: {exc}')
        raise
