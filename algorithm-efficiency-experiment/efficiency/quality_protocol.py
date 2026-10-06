"""Frozen, supervised quality requests, with independent pilot and main data."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time

from .common import ROOT, atomic_json, digest_file, digest_value, emit, wait_cool
from .coordination_workers import Worker
from .hybrid import verify_artifacts
from .quality_spec import EVENTS, MODEL_SHA, fixture_specs, schedule
from .quality_worker import QualityContext
from .research_campaign import register_fixture
from .research_protocol import ResearchBudget, pilot_gate


def read(path):
    return json.loads(Path(path).read_text())


ENVIRONMENT_KEYS = ('hailort_version','model_sha256','parameters','model_defaults','stop_tokens',
                    'prompt_template','capacity_tokens','experiment_limit_tokens')


def prepare_source(directory, config, inventory):
    if inventory['model_sha256'] != MODEL_SHA:
        raise ValueError('Quality study requires the checkpoint model')
    basis = ROOT/'runs/research-context-20261003'
    verify_artifacts(basis)
    old = read(basis/'manifest.json')
    for key in ('model_sha256','hailort_cli','archive_sha256','cpu_model','cpu_governor'):
        if old[key] != inventory[key]:
            raise ValueError(f'Quality baseline environment changed: {key}')
    shutil.copy2(basis/'context-primary-environment.json',directory/'baseline-environment.json')
    parent = None
    if config['parent']:
        p = Path(config['parent'])
        verify_artifacts(p)
        if not read(p/'validation.json')['passed'] or read(p/'summary.json')['selected'] != config['selected']:
            raise ValueError('Confirmation selection or audit changed')
        if read(p/'manifest.json')['source_sha256'] != inventory['source_sha256']:
            raise ValueError('Source changed between development and confirmation')
        parent = dict(path=str(p), checksums_sha256=digest_file(p/'checksums.json'),
                      summary_sha256=digest_file(p/'summary.json'))
        shutil.copy2(p/'summary.json',directory/'development-summary.json')
    atomic_json(directory/'freeze.json',dict(config_sha256=digest_file(directory/'config.json'),
        protocol_sha256=digest_file(directory/'protocol.json'),source_sha256=inventory['source_sha256'],
        parent=parent, baseline_manifest_sha256=digest_file(basis/'checksums.json'),
        baseline_environment_sha256=digest_file(directory/'baseline-environment.json')))


def run(directory, config):
    directory = Path(directory)
    budget = ResearchBudget(directory,config)
    with ExitStack() as stack:
        worker = Worker('process','hat',directory,dict(config,block_id='quality'),budget.check,factory=QualityContext)
        def close():
            try:
                worker.close()
            finally:
                emit(directory/EVENTS,'worker_release',**getattr(worker,'release',dict(alive=True,forced=True)))
        stack.callback(close)
        emit(directory/EVENTS,'worker_ready',**worker.ready)
        environment = worker.ready['environment']
        baseline = read(directory/'baseline-environment.json')
        if any(environment[k] != baseline[k] for k in ENVIRONMENT_KEYS):
            raise ValueError('NPU environment differs from checkpoint baseline')
        atomic_json(directory/'quality-environment.json',environment)
        fixtures = []
        for spec in fixture_specs(config['quality_stage'],config['fixture_namespace']):
            budget.stage='quality_fixtures';budget.check()
            value = worker.call('quality_fixture',spec=spec)['result']
            register_fixture(config['campaign'],config['quality_stage'],value['fixture_id'],value['seed'],value['table_sha256'])
            fixtures.append(value)
            atomic_json(directory/'quality-fixtures.json',fixtures)
        jobs = schedule(fixtures,config['conditions'])
        atomic_json(directory/'quality-schedule.json',jobs)
        emit(directory/EVENTS,'fixtures_frozen',fixtures_sha256=digest_file(directory/'quality-fixtures.json'),
             schedule_sha256=digest_file(directory/'quality-schedule.json'))
        by_id = {f['fixture_id']:f for f in fixtures}
        for section in ('pilot','main'):
            start = time.monotonic()
            for job in (j for j in jobs if j['section']==section):
                budget.stage=f"quality_{job['fixture_id']}_{job['condition']['label']}";budget.check()
                wait_cool(directory);budget.check()
                emit(directory/EVENTS,'dialogue_start',**job)
                tick = time.monotonic()
                response = worker.call('quality_dialogue',fixture=by_id[job['fixture_id']],
                                       condition=job['condition'],deadline=budget.work_deadline)
                elapsed = (time.monotonic()-tick)*1000
                emit(directory/EVENTS,'dialogue',**job,response=response,total_ms=elapsed)
            if section=='pilot':
                pilot_seconds = time.monotonic()-start
                # The shared helper writes its event to research.jsonl; the quality
                # event additionally freezes the inputs for its independent auditor.
                remaining = budget.work_deadline-time.monotonic()
                multiplier = 8 if config['quality_stage']=='develop' else 16
                emit(directory/EVENTS,'pilot_gate',pilot_seconds=pilot_seconds,multiplier=multiplier,
                     remaining_work_seconds=remaining,required_seconds=pilot_seconds*multiplier*1.2,
                     fits=pilot_seconds*multiplier*1.2<=remaining)
                pilot_gate(directory,budget,'quality',pilot_seconds,multiplier)
        emit(directory/EVENTS,'measurement_complete')
