"""Bounded, replayable attention tuning. The HAT proposes catalog IDs, never code."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import itertools
import json
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import time

import numpy as np

from .attention_native import Native, build, errors, fixture, numpy_attention, oracle
from .common import ROOT, atomic_json, digest_file, emit, progress, read_jsonl, wait_cool
from .feedback_spec import (CATALOG, CELLS, CPU_BACKENDS, SEED, STRATEGIES, advisor_messages,
    deterministic_choice, input_hash, median_by_seed, parse_proposal, promotion, specification, split_manifest)
from .hybrid import generate_checked, verify_artifacts
from .diagnostic import renderer
from .state_isolation import parameter_settings, resolve_defaults

EVENTS = 'feedback.jsonl'


def resume_budget(source):
    source = Path(source)
    config = json.loads((source/'config.json').read_text())
    outcome = json.loads((source/'outcome.json').read_text())
    prior = config.get('resume_budget', {})
    cap = prior.get('total_cap_seconds', config['max_seconds'])
    consumed = prior.get('prior_elapsed_seconds', 0)+outcome['elapsed_seconds']
    if config['profile']!='feedback-attention' or not 0 <= consumed < cap <= 3600:
        raise ValueError('No remaining attention experiment budget')
    if cap-consumed <= 65: raise ValueError('Insufficient remaining budget for supervised continuation')
    return dict(total_cap_seconds=cap, prior_elapsed_seconds=consumed, remaining_seconds=cap-consumed)


def prepare_source(directory, config, inventory):
    source = Path(config['source_run']).resolve()
    checks = verify_artifacts(source)
    old = json.loads((source/'manifest.json').read_text())
    if json.loads((source/'outcome.json').read_text())['status'] != 'complete':
        raise ValueError('Source run must be completed')
    if any(old[k] != inventory[k] for k in ('model_sha256', 'hailort_cli', 'archive_sha256')):
        raise ValueError('Source model, runtime or original archive differs')
    frozen = directory/'source-run'; frozen.mkdir()
    for name in ('checksums.json', 'config.json', 'manifest.json', 'summary.json', 'outcome.json'):
        if name != 'checksums.json' and name not in checks: raise ValueError(f'Unverified source {name}')
        shutil.copy2(source/name, frozen/name)
    atomic_json(directory/'source-provenance.json', dict(source_run=str(source), verified_artifacts=len(checks),
        source_checksums_sha256=digest_file(source/'checksums.json'),
        relationship='Original ZIP attention workload; prior run supplies verified hardware/model provenance only.'))
    shutil.copytree(build(), directory/'native-build')
    atomic_json(directory/'protocol.json', specification())
    atomic_json(directory/'splits.json', split_manifest())
    for name in ('README.md', 'EXPERIMENT-NOTES.md'):
        if (ROOT/name).exists(): shutil.copy2(ROOT/name, directory/('initial-'+name))
    result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True, timeout=30)
    (directory/'vulkan-info.txt').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode: raise RuntimeError('Vulkan inventory failed')
    if config.get('resume_run'):
        resume = Path(config['resume_run']); verified = verify_artifacts(resume)
        if resume_budget(resume) != config['resume_budget']: raise ValueError('Resume budget changed')
        prior_rows, errors = read_jsonl(resume/EVENTS)
        if errors or not prior_rows or prior_rows[-1]['event']!='incomplete':
            raise ValueError('Only an explicitly recorded stage timeout can be continued')
        if not json.loads((resume/'correctness.json').read_text()).get('passed'):
            raise ValueError('Cannot resume failed correctness checks')
        if any(not r['correct'] for r in prior_rows if r['event']=='measurement'):
            raise ValueError('Cannot resume numerical failures')
        for name in ('protocol.json', 'splits.json', 'native-build/build.json'):
            if digest_file(directory/name)!=digest_file(resume/name): raise ValueError(f'Resume {name} differs')
        old_manifest = json.loads((resume/'manifest.json').read_text())
        if any(old_manifest[k]!=inventory[k] for k in ('model_sha256','hailort_cli','archive_sha256',
                'versions','blas_library','cpu_governor','cpu_model','python','platform')):
            raise ValueError('Resume hardware/model provenance differs')
        for name in ('efficiency/attention_native.py','efficiency/feedback_spec.py'):
            if old_manifest['source_sha256'][name]!=inventory['source_sha256'][name]:
                raise ValueError(f'Resume numerical implementation/specification differs: {name}')
        folder = directory/'resume-source'; folder.mkdir()
        for name in ('checksums.json','manifest.json','config.json','outcome.json','protocol.json','telemetry.jsonl',
                     'feedback-environment.json','native-environment.json','choices.json','validation.json'):
            if (resume/name).exists(): shutil.copy2(resume/name, folder/name)
        shutil.copytree(resume/'fixtures', directory/'fixtures')
        for name in ('fixtures.json','correctness.json','baseline-policy.json','choices.json','development-history.json',
                     'policies.json','concurrency-spec.json',EVENTS,'execution-budget.json'):
            if (resume/name).exists():
                if name not in verified: raise ValueError(f'Unverified checkpoint {name}')
                shutil.copy2(resume/name, directory/name)
        atomic_json(directory/'resume-provenance.json', dict(source=str(resume), verified_artifacts=len(verified),
            source_checksums_sha256=digest_file(resume/'checksums.json'), **config['resume_budget'],
            amendment='Stage time allowance restarted using only unused total-session budget. Completed measurements and proposals are reused, not repeated. Kernels, inputs, choices and statistical rules are unchanged.'))
        emit(directory/EVENTS, 'resume_boundary', prior_run=str(resume), **config['resume_budget'])
    if config.get('replay_run'):
        replay = Path(config['replay_run']).resolve(); verified = verify_artifacts(replay)
        required = ('protocol.json', 'splits.json', 'fixtures.json', 'choices.json', 'policies.json', 'baseline-policy.json',
                    'manifest.json', 'feedback-environment.json', 'native-build/build.json')
        if not all(n in verified for n in required): raise ValueError('Replay artifacts missing')
        for name in ('protocol.json', 'splits.json', 'native-build/build.json'):
            if json.loads((directory/name).read_text()) != json.loads((replay/name).read_text()):
                raise ValueError(f'Replay {name} differs')
        prior = json.loads((replay/'manifest.json').read_text())
        if any(prior[k] != inventory[k] for k in ('model_sha256', 'hailort_cli', 'versions', 'blas_library',
                'cpu_governor', 'cpu_model', 'python', 'platform')):
            raise ValueError('Replay model/runtime differs')
        if prior['source_sha256']['efficiency/attention_native.py']!=inventory['source_sha256']['efficiency/attention_native.py']:
            raise ValueError('Replay numerical input/implementation source differs')
        folder = directory/'replay-source'; folder.mkdir()
        for name in required:
            target = folder/name; target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(replay/name, target)
        atomic_json(directory/'replay-provenance.json', dict(replay_run=str(replay), new_evidence=False,
            source_checksums_sha256=digest_file(replay/'checksums.json')))


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, directory, config, clock=time.monotonic):
        self.directory, self.clock = Path(directory), clock
        state = json.loads((self.directory/'status.json').read_text())
        self.deadline = state['monotonic']-state['elapsed_seconds']+config['max_seconds']-35
        self.stage, self.stage_end, self.last_progress = 'startup', self.deadline, 0

    def begin(self, stage):
        self.stage = stage
        self.stage_end = min(self.deadline, self.clock()+specification()['stage_seconds'][stage])
        emit(self.directory/EVENTS, 'stage_start', stage=stage, deadline=self.stage_end)
        self.check(force=True)

    def check(self, force=False):
        now = self.clock()
        if now >= self.stage_end: raise BudgetExpired(f'{self.stage} time allowance exhausted; remaining coverage is incomplete')
        if force or now-self.last_progress >= 2:
            state = json.loads((self.directory/'status.json').read_text())
            if state.get('stop_reason'): raise RuntimeError(state['stop_reason'])
            if not 0 <= now-state['monotonic'] <= 8: raise RuntimeError('Supervisor telemetry is stale')
            progress(self.directory, 'hat', self.stage)
            self.last_progress = now


def freeze_fixtures(directory, budget):
    folder = directory/'fixtures'; folder.mkdir()
    metadata = []
    for row in split_manifest():
        budget.check()
        x, w = fixture(row['n'], row['d'], row['b'], row['seed'])
        expected = oracle(x, w)
        path = folder/(row['fixture_id']+'.npz')
        np.savez(path, x=x, w=w, expected=expected)
        metadata.append(dict(**row, input_sha256=input_hash(x), weights_sha256=input_hash(w),
                             oracle_sha256=input_hash(expected), file_sha256=digest_file(path)))
    atomic_json(directory/'fixtures.json', metadata)
    return {r['fixture_id']: r for r in metadata}


class Measurements:
    def __init__(self, directory, budget, cpu, gpu, metadata):
        self.directory, self.budget, self.cpu, self.gpu, self.metadata = directory, budget, cpu, gpu, metadata
        self.serial = 0
        prior, damaged = read_jsonl(directory/EVENTS)
        if damaged: raise ValueError('Damaged measurement checkpoint')
        self.prior = {self.key(r): r for r in prior if r['event']=='measurement'}
        if len(self.prior)!=sum(r['event']=='measurement' for r in prior): raise ValueError('Duplicate checkpoint measurements')

    @staticmethod
    def key(row):
        return tuple(row.get(k) for k in ('stage','strategy','round','cell_id','seed_index','repeat','arm'))

    @lru_cache(maxsize=6)
    def data(self, fixture_id):
        path = self.directory/'fixtures'/(fixture_id+'.npz')
        if digest_file(path) != self.metadata[fixture_id]['file_sha256']: raise RuntimeError('Fixture checksum mismatch')
        with np.load(path, allow_pickle=False) as f: return f['x'], f['w'], f['expected']

    def request(self, x, w, expected, mode, backend):
        native = self.gpu if backend in CATALOG else self.cpu
        before = time.perf_counter()
        if backend != 'numpy': native.configure(x.shape)
        allocation_ms = (time.perf_counter()-before)*1000
        output = np.empty_like(x)
        started = time.monotonic(); cpu_started = time.process_time()
        if backend == 'numpy':
            output = numpy_attention(x, w, mode)
            metrics = dict(request_ms=(time.monotonic()-started)*1000,
                           cpu_seconds=time.process_time()-cpu_started)
        else: metrics = native.run(x, w, output, mode, backend)
        ended = time.monotonic()
        checked = errors(output, expected)
        return dict(**metrics, **checked, output_sha256=input_hash(output),
                    setup_ms=allocation_ms, started=started, ended=ended)

    def block(self, stage, cell, split, repeats, arms, strategy=None, round_number=None):
        count = 8 if split == 'test' else 3
        rows = []; warmed = False
        # Permutations are cycled; pair order differs by seed/repetition, not by measured outcome.
        orders = list(itertools.permutations(arms))
        for seed_index in range(count):
            fixture_id = f'{split}-{cell["cell_id"]}-s{seed_index}'
            def key(repeat, arm): return (stage, strategy, round_number, cell['cell_id'], seed_index, repeat, arm)
            missing = any(key(repeat, arm) not in self.prior for repeat in range(repeats) for arm in arms)
            if not missing:
                for repeat in range(repeats):
                    for arm in orders[(seed_index*repeats+repeat+self.serial) % len(orders)]:
                        row = self.prior[key(repeat, arm)]
                        if row['backend']!=arms[arm] or row['fixture_id']!=fixture_id: raise ValueError('Checkpoint arm/input mismatch')
                        rows.append(row)
                continue
            x, w, expected = self.data(fixture_id)
            if not warmed:
                for backend in dict.fromkeys(arms.values()):
                    self.budget.check()
                    warm = self.request(x, w, expected, cell['mode'], backend)
                    emit(self.directory/EVENTS, 'warmup', stage=stage, cell_id=cell['cell_id'], backend=backend, **warm)
                    if not warm['correct']: raise RuntimeError(f'Warmup numerical failure: {backend}/{cell["cell_id"]}')
                warmed = True
            for repeat in range(repeats):
                order = orders[(seed_index*repeats+repeat+self.serial) % len(orders)]
                for arm in order:
                    if key(repeat, arm) in self.prior:
                        old = self.prior[key(repeat, arm)]
                        if old['backend']!=arms[arm] or old['fixture_id']!=fixture_id: raise ValueError('Checkpoint arm/input mismatch')
                        rows.append(old); continue
                    self.budget.check()
                    row = self.request(x, w, expected, cell['mode'], arms[arm])
                    row = emit(self.directory/EVENTS, 'measurement', stage=stage, strategy=strategy,
                        round=round_number, cell_id=cell['cell_id'], mode=cell['mode'], split=split,
                        fixture_id=fixture_id, seed_index=seed_index, repeat=repeat, arm=arm, backend=arms[arm],
                        input_sha256=self.metadata[fixture_id]['input_sha256'], **row)
                    rows.append(row)
                    if not row['correct']: raise RuntimeError(f'Numerical failure: {arms[arm]}/{cell["cell_id"]}')
        self.serial += 1
        return rows


def correctness(directory, budget, build_dir):
    observations = []
    with Native(build_dir, gpu=True, validation=True) as gpu, Native(build_dir) as cpu:
        for n, d, b in ((1, 1, 1), (2, 8, 4), (17, 16, 2), (19, 65, 2), (64, 64, 4)):
            x, w = fixture(n, d, b, 912+n); expected = oracle(x, w)
            for mode in ('stream', 'prefill'):
                for backend in (*CPU_BACKENDS, *CATALOG):
                    budget.check(); engine = gpu if backend in CATALOG else cpu
                    engine.configure(x.shape)
                    output = np.empty_like(x)
                    if backend == 'numpy': output = numpy_attention(x, w, mode)
                    else: engine.run(x, w, output, mode, backend)
                    check = errors(output, expected)
                    observations.append(dict(n=n, d=d, b=b, mode=mode, backend=backend, **check))
                    emit(directory/EVENTS, 'correctness', **observations[-1])
                    if not check['correct']: raise RuntimeError(f'Correctness failed: {observations[-1]}')
        # Future-input perturbation, independent streams and A/B/A cache reset checks.
        x, w = fixture(17, 64, 2, 456)
        # Sign changes preserve the frozen standard-normal input distribution.
        changed = x.copy(); changed[:, 9:] *= -1; changed[1] *= -1
        for mode in ('stream', 'prefill'):
            for backend in (*CPU_BACKENDS, *CATALOG):
                budget.check(); engine = gpu if backend in CATALOG else cpu; engine.configure(x.shape)
                results = []
                for inputs in (x, changed, x):
                    out = np.empty_like(inputs)
                    if backend == 'numpy': out = numpy_attention(inputs, w, mode)
                    else: engine.run(inputs, w, out, mode, backend)
                    check = errors(out, oracle(inputs, w))
                    emit(directory/EVENTS, 'perturbation', mode=mode, backend=backend, **check)
                    if not check['correct']: raise RuntimeError(f'Changed-input correctness failed: {mode}/{backend}: {check}')
                    results.append(out.copy())
                causal = bool(np.array_equal(results[0][0, :9], results[1][0, :9]))
                reset = bool(np.array_equal(results[0], results[2]))
                isolated = x.copy(); isolated[1] *= -1
                out = np.empty_like(x)
                if backend == 'numpy': out = numpy_attention(isolated, w, mode)
                else: engine.run(isolated, w, out, mode, backend)
                batch_isolated = bool(np.array_equal(out[0], results[0][0]))
                observations.append(dict(mode=mode, backend=backend, causality=causal, cache_reset=reset,
                    batch_isolation=batch_isolated, correct=causal and reset and batch_isolated))
                emit(directory/EVENTS, 'correctness', **observations[-1])
                if not observations[-1]['correct']: raise RuntimeError(f'Semantic check failed: {observations[-1]}')
        identity = gpu.identity
    atomic_json(directory/'correctness.json', dict(passed=True, synchronization_validation=True,
        device=identity, cases=observations, count=len(observations)))


def paired(rows, left, right):
    a, b = median_by_seed(rows, left), median_by_seed(rows, right)
    return [(a[s], b[s]) for s in sorted(a)]


class Adviser:
    def __init__(self, llm, environment):
        self.llm, self.environment = llm, environment
        self.render = renderer(environment['prompt_template'])

    def propose(self, messages, eligible):
        started = time.monotonic()
        result = generate_checked(self.llm, self.render, messages, '', True, self.environment['parameters'],
            self.environment['stop_tokens'], self.environment['experiment_limit_tokens'])
        choice, reason = parse_proposal(result['output'], self.environment['stop_tokens'],
            result['completion_status'], result['token_ledger_valid'], eligible)
        return dict(**result, choice=choice, rejection_reason=reason, started=started,
                    ended=time.monotonic(), total_ms=(time.monotonic()-started)*1000)


def choose(adviser, strategy, history, routes, eligible, replay_choice=None):
    if replay_choice is not None:
        if replay_choice not in eligible: raise ValueError('Replay choice is ineligible or repeated')
        return replay_choice, dict(origin='replay', credited_to_hat=False)
    if strategy == 'coordinate':
        return deterministic_choice(history, eligible), dict(origin='coordinate', credited_to_hat=False)
    result = adviser.propose(advisor_messages(history, routes, eligible), eligible)
    candidate = result['choice'] or deterministic_choice(history, eligible)
    return candidate, dict(origin='hat' if result['choice'] else 'fallback',
        credited_to_hat=result['choice'] is not None, adviser=result)


def concurrency(directory, budget, measurements, adviser, baseline, histories):
    cell = next(c for c in CELLS if c['cell_id']=='prefill-n1024-b4')
    available = [h for history in histories.values() for h in history]
    fastest = min(available, key=lambda h: h['development_cells'][cell['cell_id']][0])['candidate']
    messages = advisor_messages([], baseline, list(CATALOG))
    atomic_json(directory/'concurrency-spec.json', dict(messages=messages, gpu_candidate=fastest,
        cpu_backend=baseline[cell['cell_id']], batches=24, repeats=6, forced_gpu=True,
        interpretation='Independent numeric and adviser jobs; host interval overlap, not proof of simultaneous device execution.'))
    data = [measurements.data(f'development-{cell["cell_id"]}-s{i}') for i in range(3)]
    def numeric(backend):
        start = time.monotonic(); values = []
        for i in range(24):
            budget.check()
            values.append(measurements.request(*data[i % 3], cell['mode'], backend))
        return dict(started=start, ended=time.monotonic(), requests=values,
                    correct=all(v['correct'] for v in values))
    conditions = ('cpu_serial', 'cpu_overlap', 'gpu_serial', 'gpu_overlap')
    old_rows, _ = read_jsonl(directory/EVENTS)
    completed = {(r['repeat'], r['condition']) for r in old_rows if r['event']=='concurrency'}
    with ThreadPoolExecutor(max_workers=1) as pool:
        for repeat in range(6):
            for condition in conditions[repeat % 4:]+conditions[:repeat % 4]:
                if (repeat, condition) in completed: continue
                budget.check(force=True); wait_cool(directory)
                backend = fastest if condition.startswith('gpu') else baseline[cell['cell_id']]
                start = time.monotonic()
                if condition.endswith('overlap'):
                    future = pool.submit(numeric, backend)
                    advised = adviser.propose(messages, list(CATALOG))
                    numerical = future.result()
                else:
                    numerical = numeric(backend); advised = adviser.propose(messages, list(CATALOG))
                end = time.monotonic()
                overlap = max(0, min(numerical['ended'], advised['ended'])-max(numerical['started'], advised['started']))
                emit(directory/EVENTS, 'concurrency', repeat=repeat, condition=condition, backend=backend,
                    total_ms=(end-start)*1000, overlap_ms=overlap*1000, numerical=numerical, adviser=advised)
                if not numerical['correct']: raise RuntimeError('Concurrency numerical failure')


def run(directory, config):
    directory = Path(directory); budget = Budget(directory, config)
    started = time.monotonic(); histories = {s: [] for s in STRATEGIES}; choices = []
    resumed = bool(config.get('resume_run'))
    checkpoint, _ = read_jsonl(directory/EVENTS)
    existing_decisions = {(r['strategy'],r['round'],r['cell_id']): r for r in checkpoint if r['event']=='decision'}
    saved_choices = json.loads((directory/'choices.json').read_text()) if resumed else []
    choices = list(saved_choices)
    try:
        if resumed:
            metadata = {r['fixture_id']: r for r in json.loads((directory/'fixtures.json').read_text())}
        else:
            budget.begin('correctness_baseline')
            metadata = freeze_fixtures(directory, budget)
            if config.get('replay_run'):
                prior = {r['fixture_id']: r for r in json.loads((directory/'replay-source/fixtures.json').read_text())}
                if metadata!=prior: raise ValueError('Regenerated replay fixture fingerprints differ')
            correctness(directory, budget, directory/'native-build')
        with ExitStack() as stack:
            cpu = stack.enter_context(Native(directory/'native-build'))
            gpu = stack.enter_context(Native(directory/'native-build', gpu=True))
            if resumed and gpu.identity!=json.loads((directory/'resume-source/native-environment.json').read_text())['device']:
                raise ValueError('Resume GPU/driver identity differs')
            measurements = Measurements(directory, budget, cpu, gpu, metadata)
            atomic_json(directory/'native-environment.json', dict(device=gpu.identity,
                cpu_initialization_ms=cpu.initialization_ms, gpu_initialization_ms=gpu.initialization_ms,
                gpu_utilization_available=False, timestamps='Diagnostic only; main timing includes host copies and fences.'))
            baseline = {}
            if resumed:
                baseline = json.loads((directory/'baseline-policy.json').read_text())['routes']
                measurements.serial = len(CELLS)
            else:
                for cell in CELLS:
                    budget.check(force=True); wait_cool(directory)
                    rows = measurements.block('baseline', cell, 'development', 3, {b: b for b in CPU_BACKENDS})
                    baseline[cell['cell_id']] = min(CPU_BACKENDS, key=lambda b: sum(median_by_seed(rows, b).values()))
            replay = config.get('replay_run')
            if replay:
                baseline = json.loads((directory/'replay-source/baseline-policy.json').read_text())['routes']
            atomic_json(directory/'baseline-policy.json', dict(routes=baseline, unknown_shape='native1'))
            if not resumed: emit(directory/EVENTS, 'stage_end', stage=budget.stage)
            budget.begin('search')
            from hailo_platform import VDevice, __version__
            from hailo_platform.pyhailort.pyhailort import LLM
            progress(directory, 'hat', 'model_loading')
            loading = time.monotonic()
            device = stack.enter_context(VDevice()); llm = stack.enter_context(LLM(device, config['model']))
            env = dict(hailort_version=__version__, prompt_template=llm.prompt_template(),
                stop_tokens=llm.get_stop_tokens(), capacity_tokens=llm.max_context_capacity(),
                model_defaults=resolve_defaults(llm, __version__), loading_seconds=time.monotonic()-loading)
            env['parameters'] = parameter_settings(env['model_defaults'], 'penalty_1_0')[0]
            env['experiment_limit_tokens'] = min(1792, env['capacity_tokens']-256)
            if resumed:
                previous_env = json.loads((directory/'resume-source/feedback-environment.json').read_text())
                for key in ('hailort_version','prompt_template','stop_tokens','capacity_tokens','model_defaults','parameters','experiment_limit_tokens'):
                    if env[key]!=previous_env[key]: raise ValueError(f'Resume model environment differs: {key}')
                env['this_segment_loading_seconds'] = env['loading_seconds']
                env['loading_seconds'] += previous_env['loading_seconds']
            atomic_json(directory/'feedback-environment.json', env)
            adviser = Adviser(llm, env)
            policies = {s: dict(baseline) for s in STRATEGIES}
            old_choices = json.loads((directory/'replay-source/choices.json').read_text()) if replay else []
            replay_routes = json.loads((directory/'replay-source/policies.json').read_text()) if replay else None
            for round_number in range(1, 4):
                for strategy in (STRATEGIES if round_number % 2 else STRATEGIES[::-1]):
                    budget.check(force=True); wait_cool(directory)
                    tuning_start = time.monotonic()
                    history, routes = histories[strategy], policies[strategy]
                    eligible = sorted(set(CATALOG)-{h['candidate'] for h in history})
                    old = next((c for c in old_choices if c['strategy']==strategy and c['round']==round_number), None)
                    if replay and old is None: raise ValueError('Replay choice coverage incomplete')
                    saved = next((c for c in saved_choices if c['strategy']==strategy and c['round']==round_number), None)
                    if saved:
                        choice = saved; candidate = saved['candidate']
                        if candidate not in eligible: raise ValueError('Checkpoint candidate became ineligible')
                    else:
                        candidate, proposal = choose(adviser, strategy, history, routes, eligible, old['candidate'] if old else None)
                        choice = dict(strategy=strategy, round=round_number, candidate=candidate, **proposal)
                        choices.append(choice); atomic_json(directory/'choices.json', choices)
                        emit(directory/EVENTS, 'proposal', **choice)
                    development = {}
                    for cell in CELLS:
                        rows = measurements.block('development', cell, 'development', 3,
                            dict(incumbent=routes[cell['cell_id']], candidate=candidate), strategy, round_number)
                        development[cell['cell_id']] = [round(statistics.mean(median_by_seed(rows, 'candidate').values()), 6),
                                                       all(r['correct'] for r in rows)]
                    for cell in CELLS:
                        before = routes[cell['cell_id']]
                        rows = measurements.block('validation', cell, f'validation-{round_number}', 5,
                            dict(incumbent=before, candidate=candidate), strategy, round_number)
                        decision = promotion(paired(rows, 'incumbent', 'candidate'), all(r['correct'] for r in rows),
                                             SEED+round_number)
                        if decision['promote'] and not replay: routes[cell['cell_id']] = candidate
                        if replay:
                            routes[cell['cell_id']] = old['routes_after'][cell['cell_id']]
                        previous = existing_decisions.get((strategy,round_number,cell['cell_id']))
                        if previous:
                            if any(previous[k]!=decision[k] for k in decision) or previous['before']!=before or previous['after']!=routes[cell['cell_id']]:
                                raise ValueError('Checkpoint promotion changed')
                        else:
                            emit(directory/EVENTS, 'decision', strategy=strategy, round=round_number,
                                 cell_id=cell['cell_id'], candidate=candidate, before=before,
                                 after=routes[cell['cell_id']], applied=not replay, **decision)
                    history.append(dict(candidate=candidate, development_total_ms=sum(v[0] for v in development.values()),
                                        development_cells=development))
                    choice['routes_after'] = dict(routes)
                    if 'tuning_seconds' not in choice:
                        previous_seconds = 0
                        if saved:
                            proposal_row = next(r for r in checkpoint if r['event']=='proposal' and r['strategy']==strategy and r['round']==round_number)
                            previous_start = proposal_row.get('adviser', {}).get('started', proposal_row['monotonic'])
                            previous_stop = max(r['monotonic'] for r in checkpoint if r['event']=='incomplete')
                            previous_seconds = previous_stop-previous_start
                        choice['tuning_seconds'] = previous_seconds+time.monotonic()-tuning_start
                    atomic_json(directory/'choices.json', choices)
                    atomic_json(directory/'development-history.json', histories)
            if replay:
                policies = {s: replay_routes[s]['routes'] for s in STRATEGIES}
            frozen = {s: dict(routes=policies[s], unknown_shape='native1', opt_in_only=True) for s in STRATEGIES}
            if (directory/'policies.json').exists():
                if json.loads((directory/'policies.json').read_text())!=frozen: raise ValueError('Checkpoint frozen policy changed')
            else:
                atomic_json(directory/'policies.json', frozen)
                emit(directory/EVENTS, 'policies_frozen', sha256=digest_file(directory/'policies.json'))
            emit(directory/EVENTS, 'stage_end', stage=budget.stage)
            budget.begin('final')
            for cell in CELLS:
                budget.check(force=True); wait_cool(directory)
                arms = dict(baseline=baseline[cell['cell_id']], **{s: policies[s][cell['cell_id']] for s in STRATEGIES})
                measurements.block('final', cell, 'test', 5, arms)
            emit(directory/EVENTS, 'stage_end', stage=budget.stage)
            budget.begin('concurrency')
            concurrency(directory, budget, measurements, adviser, baseline, histories)
            emit(directory/EVENTS, 'stage_end', stage=budget.stage)
        emit(directory/EVENTS, 'complete', elapsed_seconds=time.monotonic()-started, replay=bool(config.get('replay_run')))
    except BudgetExpired as exc:
        emit(directory/EVENTS, 'incomplete', reason=str(exc), elapsed_seconds=time.monotonic()-started)
        raise
