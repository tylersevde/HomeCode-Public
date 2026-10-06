"""Frozen paired CPU comparison with independent process environments."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .attention_native import fixture, oracle
from .common import ROOT, atomic_json, digest_file, emit, wait_cool
from .coordination_workers import Worker
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .research_campaign import register_fixture
from .refine_governor import request
from .study_protocol import Budget
from .cpu_compare_spec import (EVENTS, BLOCKS, BATCH_CALLS, CPU_ARMS, fixtures,
                               schedule, arm, environment, policy)
from .cpu_compare_worker import CompareNumeric


def read(path):
    return json.loads(Path(path).read_text())


def prepare_source(directory, config, inventory):
    provenance = read(directory / 'provenance.json')
    for name in ('source_campaign', 'cpu_confirmation'):
        value = provenance[name]; parent = Path(value['path'])
        verify_artifacts(parent)
        if digest_file(parent / 'checksums.json') != value['checksums_sha256']:
            raise ValueError('Historical provenance changed')
    parent = Path(provenance['cpu_confirmation']['path'])
    old = read(parent / 'manifest.json')
    for name, checksum in old['source_sha256'].items():
        if name.startswith('native/attention/') or name == 'efficiency/attention_native.py':
            if inventory['source_sha256'].get(name) != checksum:
                raise ValueError('Numerical implementation changed: ' + name)
    shutil.copytree(parent / 'native-build', directory / 'native-build')
    atomic_json(directory / 'freeze.json', dict(
        source_sha256=inventory['source_sha256'], config_sha256=digest_file(directory / 'config.json'),
        protocol_sha256=digest_file(directory / 'protocol.json'),
        provenance_sha256=digest_file(directory / 'provenance.json')))


@contextmanager
def worker(directory, config, budget, name, configuration):
    budget.stage = name; budget.check()
    value = Worker('process', 'numeric', directory,
                   dict(config, block_id=name, configuration=configuration, numeric_device='cpu'),
                   budget.check, factory=CompareNumeric, environment=environment(configuration))
    emit(directory / EVENTS, 'worker_ready', instance=name, configuration=configuration,
         device_kind='cpu', **value.ready)
    try:
        yield value
    finally:
        try:
            value.close()
        finally:
            emit(directory / EVENTS, 'worker_release', instance=name,
                 configuration=configuration,
                 **getattr(value, 'release', dict(alive=True, forced=True)))


def pilot_gate(directory, budget, started):
    seconds = time.monotonic() - started
    required = seconds * BLOCKS * 1.2
    remaining = budget.work_deadline - time.monotonic()
    emit(directory / EVENTS, 'pilot_gate', pilot_seconds=seconds, multiplier=BLOCKS,
         required_seconds=required, remaining_work_seconds=remaining, fits=required <= remaining)
    if required > remaining:
        raise TimeoutError('Frozen comparison cannot fit remaining allowance; pilot deferred main matrix')


def run(directory, config):
    directory = Path(directory); budget = Budget(directory, config)
    for name in ('fixtures', 'outputs'):
        (directory / name).mkdir()
    try:
        fs = []
        for spec in fixtures(config):
            budget.stage = 'numerical-fixtures'; budget.check()
            x, w = fixture(spec['n'], spec['d'], spec['b'], spec['seed'])
            expected = oracle(x, w)
            register_fixture(config['campaign'], 'cpu-compare', spec['fixture_id'],
                             spec['seed'], input_hash(x))
            path = directory / 'fixtures' / (spec['fixture_id'] + '.npz')
            np.savez(path, x=x, w=w, expected=expected)
            fs.append(dict(spec, file_sha256=digest_file(path), input_sha256=input_hash(x),
                           weights_sha256=input_hash(w), oracle_sha256=input_hash(expected)))
        jobs = schedule(config, fs); byid = {f['fixture_id']:f for f in fs}
        atomic_json(directory / 'fixtures.json', fs)
        atomic_json(directory / 'schedule.json', jobs)
        emit(directory / EVENTS, 'fixtures_frozen', fixtures_sha256=digest_file(directory / 'fixtures.json'),
             schedule_sha256=digest_file(directory / 'schedule.json'))
        for section in ('pilot', 'main'):
            started = time.monotonic()
            chosen = [j for j in jobs if j['section'] == section]
            groups = dict.fromkeys((j['block'], j['configuration']) for j in chosen)
            for block, configuration in groups:
                budget.check(); wait_cool(directory); budget.check()
                transition = request(config['governor_socket'], policy(configuration)['governor'])
                emit(directory / EVENTS, 'governor_transition', block=block, section=section,
                     configuration=configuration, **transition)
                name = f'{section}-block{block}-{configuration}'
                with worker(directory, config, budget, name, configuration) as device:
                    current = None
                    for job in (j for j in chosen if (j['block'], j['configuration']) == (block, configuration)):
                        budget.stage = name + '-' + job['fixture_id']; budget.check()
                        fid = job['fixture_id']; cell = byid[fid]
                        if current != fid:
                            wait_cool(directory); budget.check(); current = fid
                            device.call('load', fixture_id=fid)
                            for label in CPU_ARMS:
                                response = device.call('warm', fixture_id=fid, arm=arm(cell, label))
                                emit(directory / EVENTS, 'warmup', fixture_id=fid, configuration=configuration,
                                     label=label, instance=name, response=response)
                                if not response['result']['correct']:
                                    raise RuntimeError('CPU warmup numerical failure')
                        tick = time.monotonic()
                        if job['label'].startswith('batch'):
                            response = device.call('batch', fixture_id=fid, arm=arm(cell, job['label']),
                                                   request_ids=list(range(BATCH_CALLS)), deadline=budget.work_deadline)
                        else:
                            response = device.call('measure', fixture_id=fid, arm=arm(cell, job['label']))
                        end = time.monotonic()
                        emit(directory / EVENTS, 'measurement', **job, instance=name, started=tick,
                             ended=end, total_ms=(end-tick)*1000, response=response)
                        result = response['result']; calls = result.get('requests', [result])
                        if not result['correct'] or any(not r['matches_warmup'] or r['validation_errors'] for r in calls):
                            raise RuntimeError('CPU measured numerical failure')
            if section == 'pilot':
                pilot_gate(directory, budget, started)
        emit(directory / EVENTS, 'measurement_complete')
    except BaseException as exc:
        emit(directory / EVENTS, 'failure', error=f'{type(exc).__name__}: {exc}')
        raise
