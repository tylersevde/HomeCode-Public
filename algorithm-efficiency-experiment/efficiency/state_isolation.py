"""Controlled context/snapshot/process and target-generation parameter study."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time

import psutil

from .common import ROOT, atomic_json, digest_file, digest_value, emit, progress, read_jsonl, wait_cool
from .diagnostic import LOGICAL_END, check_budget, renderer, reset_context, score_answer
from .hat import exact_suffix, visible_text

MODES = ('rebuild', 'live', 'save_live', 'restore_same', 'restore_fresh', 'rebuild_fresh')
PRESETS = ('implicit', 'explicit_defaults', 'penalty_1_0', 'penalty_1_2')
PARAMETERS = ('temperature', 'top_p', 'top_k', 'frequency_penalty',
              'max_generated_tokens', 'do_sample', 'seed')
BASELINE = dict(do_sample=False, seed=12345, max_generated_tokens=16)
EVENT_FILE = 'state_isolation.jsonl'


class Quarantine(RuntimeError):
    """A particular context cannot support the planned comparison."""


def select_cases(rows, source_inputs, environment):
    indexed = {(r['fixture_id'], r['order_index'], r['repeat'], r['arm'], r['turn']): r
               for r in rows if r['event'] == 'measurement' and r.get('study') == 'position'}
    render = renderer(environment['prompt_template'])
    cases = []
    for name, fixture_id, turn in [('small_failure', 'n128-pair06', 1),
                                   ('medium_failure', 'n512-pair03', 1),
                                   ('long_reverse', 'n1536-pair01', 2)]:
        target = indexed[(fixture_id, 0, 0, 'rebuild', turn)]
        other = indexed[(fixture_id, 0, 0, 'retain', turn)]
        conditioning = [indexed[(fixture_id, 0, 0, 'rebuild', t)] for t in range(turn)]
        for t, row in enumerate(conditioning):
            if row['output'] != indexed[(fixture_id, 0, 0, 'retain', t)]['output']:
                raise ValueError('Selected source conditioning histories differ between arms')
        if target['effective_prompt'] != other['effective_prompt']:
            raise ValueError('Selected source target histories differ between arms')
        if render(target['messages']) != target['effective_prompt']:
            raise ValueError('Source messages do not reproduce the exact target prompt')
        prefix = conditioning[-1]['effective_prompt'] + conditioning[-1]['output']
        if not target['effective_prompt'].startswith(prefix):
            raise ValueError('Source target does not extend the recorded history')
        cases.append(dict(case_id=name, fixture_id=fixture_id, turn=turn,
            target_tokens=target['target_tokens'], expected_answer=target['expected_answer'],
            queried_item=target['queried_item'], messages=deepcopy(target['messages']),
            effective_prompt=target['effective_prompt'], prefix=prefix,
            suffix=target['effective_prompt'][len(prefix):],
            source_input_sha256=target['effective_input_sha256'],
            original_outputs={a:indexed[(fixture_id,0,0,a,turn)]['output'] for a in ('rebuild','retain')},
            conditioning=[{k:r[k] for k in ('turn','effective_prompt','output','expected_answer','queried_item')}
                          for r in conditioning]))
    control = deepcopy(cases[0])
    control.update(case_id='same_table_control', queried_item='item002', source_input_sha256=None,
                   original_outputs={})
    fixture = next(f for f in source_inputs['fixtures'] if f['pair_id'] == control['fixture_id'])
    control['expected_answer'] = dict(fixture['table'])['item002']
    control['messages'][-1]['content'] = 'What color is item002?'
    control['effective_prompt'] = render(control['messages'])
    assert control['effective_prompt'].startswith(control['prefix'])
    control['suffix'] = control['effective_prompt'][len(control['prefix']):]
    cases.insert(1, control)
    return cases


def make_schedule(cases, repeats=3, seed=20261002, case_id=None):
    rng = random.Random(seed)
    def job(case, preset, repeat, phase, live_first):
        after = ['rebuild', 'restore_same'] + ([] if live_first else ['live'])
        fresh = ['restore_fresh', 'rebuild_fresh']
        rng.shuffle(after)
        rng.shuffle(fresh)
        return dict(job_id=f'{phase}-{case["case_id"]}-{preset}-r{repeat+1}',
                    case_id=case['case_id'], preset=preset, repeat=repeat, phase=phase,
                    conditioning_turns=len(case['conditioning']),
                    live_first=live_first, after_snapshot=after, fresh_order=fresh)
    validation = [job(c,'implicit',0,'validation',i % 2 == 0) for i,c in enumerate(cases[:2])]
    main = [job(c,p,r,'main',(i+j+r)%2 == 0) for i,c in enumerate(cases)
            for j,p in enumerate(PRESETS) for r in range(repeats)]
    rng.shuffle(main)
    if case_id:
        main = [j for j in main if j['case_id'] == case_id]
        if not main:
            raise ValueError('Selected state-isolation case is unavailable')
    counts = {phase:dict(targets=len(jobs)*len(MODES),
              conditioning=sum(2*len(next(c for c in cases if c['case_id']==j['case_id'])['conditioning'])
                               for j in jobs)) for phase,jobs in [('validation',validation),('main',main)]}
    return dict(validation=validation, main=main, planned=counts)


def prepare_source(directory, config, inventory):
    source = Path(config['source_run']).resolve()
    checksums = json.loads((source/'checksums.json').read_text())
    for relative, expected in checksums.items():
        path = (source/relative).resolve()
        if not path.is_relative_to(source) or digest_file(path) != expected:
            raise ValueError(f'Source checksum mismatch: {relative}')
    names = ('config.json','manifest.json','hat-environment.json','diagnostic-inputs.json',
             'diagnostic.jsonl','summary.json','outcome.json')
    if not all(name in checksums for name in names):
        raise ValueError('Source must be a checksummed complete diagnostic run')
    summary = json.loads((source/'summary.json').read_text())
    if not summary.get('complete') or not summary.get('protocol_completed'):
        raise ValueError('State isolation requires complete source diagnostic coverage')
    manifest = json.loads((source/'manifest.json').read_text())
    for key in ('model_sha256','hailort_cli'):
        if manifest[key] != inventory[key]:
            raise ValueError(f'Source runtime identity differs: {key}')
    frozen = directory/'source-run'
    frozen.mkdir()
    for name in (*names,'checksums.json'):
        shutil.copy2(source/name, frozen/name)
    rows, errors = read_jsonl(frozen/'diagnostic.jsonl')
    if errors:
        raise ValueError('Source diagnostic records are damaged')
    cases = select_cases(rows, json.loads((frozen/'diagnostic-inputs.json').read_text()),
                         json.loads((frozen/'hat-environment.json').read_text()))
    atomic_json(directory/'state-inputs.json', cases)
    atomic_json(directory/'schedule.json', make_schedule(cases,config['state_repeats'],config['schedule_seed'],config.get('state_case')))
    atomic_json(directory/'source-provenance.json', dict(source_run=str(source),
        verified_source_artifacts=len(checksums), source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={name:digest_file(frozen/name) for name in (*names,'checksums.json')}))


def resolve_defaults(llm, version):
    # This adapter deliberately targets the installed, inspected 5.1.1 binding.
    if version != '5.1.1':
        raise RuntimeError('Generation-default adapter requires HailoRT 5.1.1')
    params = llm._llm.create_generator_params()
    values = {key:getattr(params,key) for key in PARAMETERS}
    for key,value in values.items():
        if not isinstance(value,(bool,int,float)) or not math.isfinite(value):
            raise RuntimeError(f'Invalid resolved generation default: {key}')
    return values


def parameter_settings(defaults, preset):
    effective = dict(defaults, **BASELINE)
    if preset == 'implicit':
        return dict(BASELINE), effective
    if preset not in PRESETS:
        raise ValueError(f'Unknown parameter preset: {preset}')
    if preset.startswith('penalty_'):
        effective['frequency_penalty'] = 1.0 if preset == 'penalty_1_0' else 1.2
    return dict(effective), effective


def snapshot_memory_guard(size, available=None):
    available = psutil.virtual_memory().available if available is None else available
    required = 10*size + 1024**3
    if available <= required:
        raise RuntimeError(f'Snapshot load requires more than {required} available bytes; observed {available}')
    return dict(snapshot_bytes=size, available_memory_bytes=available, required_available_bytes=required)


def generation(llm, prompt, kwargs, stops):
    chunks, first = [], None
    cpu = time.process_time()
    started = time.perf_counter()
    with llm.generate(prompt, **kwargs) as gen:
        while str(gen.generation_status).endswith('.GENERATING'):
            chunk = gen.read(timeout_ms=90000)
            chunks.append(chunk)
            if first is None and visible_text(''.join(chunks),stops).strip():
                first = time.perf_counter()
        status = str(gen.generation_status).split('.')[-1]
    return dict(output=''.join(chunks), stream_chunks=chunks, stream_events=len(chunks),
        completion_status=status, request_ms=(time.perf_counter()-started)*1000,
        first_visible_ms=(first-started)*1000 if first is not None else None,
        cpu_seconds=time.process_time()-cpu)


@contextmanager
def open_model(directory, config, job, mode):
    from hailo_platform import VDevice, __version__
    from hailo_platform.pyhailort.pyhailort import LLM
    progress(directory,'hat','model_loading',job_id=job['job_id'],mode=mode)
    began = time.perf_counter()
    with VDevice() as device, LLM(device,config['model']) as llm:
        defaults = resolve_defaults(llm,__version__)
        source = json.loads((directory/'source-run/hat-environment.json').read_text())
        observed = dict(hailort_version=__version__,prompt_template=llm.prompt_template(),
            stop_tokens=llm.get_stop_tokens(),generation_recovery_sequence=llm.get_generation_recovery_sequence(),
            capacity_tokens=llm.max_context_capacity(), model=config['model'])
        for key in ('hailort_version','prompt_template','stop_tokens','generation_recovery_sequence','capacity_tokens'):
            if observed[key] != source[key]:
                raise RuntimeError(f'Runtime metadata differs from frozen source: {key}')
        env_path = directory/'state-environment.json'
        if env_path.exists():
            if json.loads(env_path.read_text())['model_defaults'] != defaults:
                raise RuntimeError('Resolved model defaults changed between model instances')
        else:
            atomic_json(env_path,dict(observed,model_defaults=defaults,
                effective_baseline=parameter_settings(defaults,'implicit')[1],
                experiment_limit_tokens=min(1792,observed['capacity_tokens']-256),
                snapshot_semantics='Opaque serialized state; contents and generated token IDs are not inferred',
                parameter_semantics='1.0 and 1.2 are sensitivity values, not assumed neutral'))
        emit(directory/EVENT_FILE,'model_loaded',job_id=job['job_id'],mode=mode,pid=os.getpid(),
             loading_seconds=time.perf_counter()-began,model_defaults=defaults)
        reset_context(llm)
        yield llm, defaults, observed['stop_tokens'], min(1792,observed['capacity_tokens']-256)
        reset_context(llm)
    emit(directory/EVENT_FILE,'model_released',job_id=job['job_id'],mode=mode,pid=os.getpid())


def observation(llm, directory, job, case, mode, kind, prompt, full, kwargs, effective,
                stops, limit, expected_before, snapshot=None, expected_output=None, conditioning_turn=None):
    progress(directory,'hat','generation',job_id=job['job_id'],mode=mode,kind=kind)
    ids = llm.tokenize(full)
    check_budget(llm,ids,limit)
    before = llm.get_context_usage_size()
    if before != expected_before:
        reset_context(llm)
        raise Quarantine(f'Before-request context mismatch: {before} != {expected_before}')
    if kind == 'target' and case['source_input_sha256'] is not None and digest_value(ids) != case['source_input_sha256']:
        raise RuntimeError('Frozen target token identity differs from the source')
    result = generation(llm,prompt,kwargs,stops)
    after = llm.get_context_usage_size()
    expected_after = len(llm.tokenize(full+result['output']))
    result.update(job_id=job['job_id'],case_id=case['case_id'],phase=job['phase'],repeat=job['repeat'],
        preset=job['preset'],mode=mode,kind=kind,pid=os.getpid(),conditioning_turn=conditioning_turn,
        effective_prompt=full,submitted_prompt=prompt,effective_input_sha256=digest_value(ids),
        effective_input_tokens=len(ids),context_before=before,context_after=after,
        expected_context_after=expected_after,token_ledger_valid=after==expected_after,
        provided_parameters=kwargs,effective_parameters=effective,effective_parameters_sha256=digest_value(effective),
        snapshot_sha256=snapshot['sha256'] if snapshot else None,
        snapshot_origin_job=snapshot['job_id'] if snapshot else None,
        expected_answer=case['expected_answer'],queried_item=case['queried_item'],
        expected_conditioning_output=expected_output,
        conditioning_matches=expected_output==result['output'] if expected_output is not None else None,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    result.update(score_answer(result['output'],case['expected_answer'],case['queried_item'],stops,result['completion_status']))
    emit(directory/EVENT_FILE,'response',**result)
    if after > limit:
        raise RuntimeError('Observed context exceeds the experiment budget')
    if not result['token_ledger_valid']:
        reset_context(llm)
        emit(directory/EVENT_FILE,'context_quarantined',job_id=job['job_id'],mode=mode,kind=kind,
             reason='runtime_context_count_mismatch',reset_verified=True)
    return result


def condition(llm, directory, job, case, mode, defaults, stops, limit):
    reset_context(llm)
    accumulated = ''
    kwargs, effective = parameter_settings(defaults,'implicit')
    for row in case['conditioning']:
        suffix, old, new, full_ids = exact_suffix(row['effective_prompt'],accumulated,llm.tokenize)
        scoring_case = dict(case,expected_answer=row['expected_answer'],queried_item=row['queried_item'])
        result = observation(llm,directory,job,scoring_case,mode,'conditioning',suffix,row['effective_prompt'],
            kwargs,effective,stops,limit,len(old),expected_output=row['output'],conditioning_turn=row['turn'])
        if (not result['token_ledger_valid'] or not result['conditioning_matches']
                or result['completion_status'] != LOGICAL_END):
            reset_context(llm)
            raise Quarantine('Conditioning did not reproduce the frozen history exactly')
        accumulated = row['effective_prompt']+result['output']
    if accumulated != case['prefix']:
        raise RuntimeError('Conditioned text differs from the frozen prefix')
    exact_suffix(case['effective_prompt'],accumulated,llm.tokenize)


def save_snapshot(llm, directory, job, case, model_sha256):
    expected = len(llm.tokenize(case['prefix']))
    before = llm.get_context_usage_size()
    if before != expected:
        reset_context(llm)
        raise Quarantine('Unverified context cannot be snapshotted')
    progress(directory,'hat','save_context',job_id=job['job_id'])
    began = time.perf_counter()
    blob = llm.save_context()
    if llm.get_context_usage_size() != before:
        reset_context(llm)
        raise Quarantine('Saving context changed its reported token count')
    if not blob:
        raise RuntimeError('SDK returned an empty context snapshot')
    guard = snapshot_memory_guard(len(blob))
    if shutil.disk_usage(directory).free <= len(blob)+1024**3:
        raise RuntimeError('Insufficient SSD space for snapshot and 1 GiB reserve')
    path = directory/'snapshots'/f'{job["job_id"]}.bin'
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)
    meta = dict(job_id=job['job_id'],case_id=case['case_id'],snapshot_path=str(path.relative_to(directory)),
        sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),context_tokens=before,
        prefix_sha256=digest_value(case['prefix']),model_sha256=model_sha256,origin_pid=os.getpid(),
        save_ms=(time.perf_counter()-began)*1000,memory_guard=guard)
    atomic_json(path.with_suffix('.json'),meta)
    emit(directory/EVENT_FILE,'snapshot_saved',**meta)
    return meta


def restore_snapshot(llm, directory, job, case, snapshot, model_sha256):
    if (snapshot['case_id'] != case['case_id'] or snapshot['prefix_sha256'] != digest_value(case['prefix'])
            or snapshot['model_sha256'] != model_sha256 or snapshot['job_id'] != job['job_id']):
        raise RuntimeError('Snapshot lineage does not match the requested case/model/job')
    path = (directory/snapshot['snapshot_path']).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise RuntimeError('Snapshot path leaves the run directory')
    if path.stat().st_size != snapshot['bytes'] or digest_file(path) != snapshot['sha256']:
        raise RuntimeError('Snapshot bytes do not match their saved identity')
    guard = snapshot_memory_guard(snapshot['bytes'])
    reset_context(llm)
    progress(directory,'hat','load_context',job_id=job['job_id'])
    began = time.perf_counter()
    blob = path.read_bytes()
    llm.load_context(blob)
    actual = llm.get_context_usage_size()
    emit(directory/EVENT_FILE,'snapshot_restored',job_id=job['job_id'],case_id=case['case_id'],
         pid=os.getpid(),snapshot_sha256=snapshot['sha256'],context_tokens=actual,
         expected_context_tokens=snapshot['context_tokens'],load_ms=(time.perf_counter()-began)*1000,
         memory_guard=guard)
    if actual != snapshot['context_tokens'] or actual != len(llm.tokenize(case['prefix'])):
        reset_context(llm)
        raise Quarantine('Restoring snapshot did not restore the expected context count')


def target(llm, directory, job, case, mode, defaults, stops, limit, snapshot=None):
    rebuilding = mode in ('rebuild','rebuild_fresh')
    if rebuilding:
        reset_context(llm)
    suffix, old, new, ids = exact_suffix(case['effective_prompt'],case['prefix'],llm.tokenize)
    kwargs,effective = parameter_settings(defaults,job['preset'])
    return observation(llm,directory,job,case,mode,'target',case['effective_prompt'] if rebuilding else suffix,
        case['effective_prompt'],kwargs,effective,stops,limit,0 if rebuilding else len(old),snapshot=snapshot)


def skip(directory, job, mode, reason):
    emit(directory/EVENT_FILE,'target_skipped',job_id=job['job_id'],phase=job['phase'],
         case_id=job['case_id'],preset=job['preset'],repeat=job['repeat'],mode=mode,reason=str(reason))


def wait_for_child(process, timeout=180):
    """Child inherits worker process group; outer supervision can stop both."""
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise


def launch_fresh(directory, config, job, case, mode, snapshot):
    folder = directory/'fresh'/f'{job["job_id"]}-{mode}'
    folder.mkdir(parents=True)
    spec = folder/'trial.json'
    atomic_json(spec,dict(job=job,case=case,mode=mode,snapshot=snapshot))
    progress(directory,'hat','fresh_process',job_id=job['job_id'],mode=mode)
    env = dict(os.environ,HAILORT_LOGGER_PATH=str(folder),PYTHONDONTWRITEBYTECODE='1')
    with (folder/'worker.log').open('w') as log:
        # Inherit the supervised worker's process group; never detach this child.
        process = subprocess.Popen([sys.executable,str(ROOT/'experiment.py'),'_worker','--phase','hat',
            '--output',str(directory),'--state-trial',str(spec)],stdout=log,stderr=subprocess.STDOUT,
            cwd=directory,env=env,start_new_session=False)
        code = wait_for_child(process)
    if code:
        raise RuntimeError(f'Fresh-process condition failed ({code}); inspect {folder/"worker.log"}')


def run_fresh(directory, config, trial_path):
    trial_path = trial_path.resolve()
    if not trial_path.is_relative_to(directory.resolve()):
        raise ValueError('Fresh trial specification must belong to this run')
    spec = json.loads(trial_path.read_text())
    job,case,mode,snapshot = (spec[k] for k in ('job','case','mode','snapshot'))
    if mode not in ('restore_fresh','rebuild_fresh'):
        raise ValueError('Fresh worker received a non-fresh condition')
    manifest = json.loads((directory/'manifest.json').read_text())
    wait_cool(directory)
    with open_model(directory,config,job,mode) as (llm,defaults,stops,limit):
        try:
            if mode == 'restore_fresh':
                restore_snapshot(llm,directory,job,case,snapshot,manifest['model_sha256'])
            target(llm,directory,job,case,mode,defaults,stops,limit,snapshot)
        except Quarantine as exc:
            reset_context(llm)
            skip(directory,job,mode,exc)


def run_job(directory, config, job, case, model_sha256):
    emit(directory/EVENT_FILE,'job_start',**job)
    wait_cool(directory)
    snapshot = None
    with open_model(directory,config,job,'original_instance') as (llm,defaults,stops,limit):
        def live():
            try:
                condition(llm,directory,job,case,'live',defaults,stops,limit)
                target(llm,directory,job,case,'live',defaults,stops,limit)
            except Quarantine as exc:
                reset_context(llm)
                skip(directory,job,'live',exc)
        if job['live_first']:
            live()
        try:
            condition(llm,directory,job,case,'snapshot_donor',defaults,stops,limit)
            snapshot = save_snapshot(llm,directory,job,case,model_sha256)
            target(llm,directory,job,case,'save_live',defaults,stops,limit,snapshot)
        except Quarantine as exc:
            reset_context(llm)
            skip(directory,job,'save_live',exc)
        for mode in job['after_snapshot']:
            wait_cool(directory)
            if mode == 'live':
                live()
                continue
            try:
                if mode == 'restore_same':
                    if snapshot is None:
                        raise Quarantine('No verified snapshot was created')
                    restore_snapshot(llm,directory,job,case,snapshot,model_sha256)
                target(llm,directory,job,case,mode,defaults,stops,limit,snapshot if mode=='restore_same' else None)
            except Quarantine as exc:
                reset_context(llm)
                skip(directory,job,mode,exc)
    # All original model/device resources are released before any fresh process.
    for mode in job['fresh_order']:
        if mode == 'restore_fresh' and snapshot is None:
            skip(directory,job,mode,'No verified snapshot was created')
        else:
            launch_fresh(directory,config,job,case,mode,snapshot if mode=='restore_fresh' else None)
    emit(directory/EVENT_FILE,'job_complete',job_id=job['job_id'],phase=job['phase'])


def validate_capabilities(rows):
    selected = [r for r in rows if r.get('phase') == 'validation']
    targets = [r for r in selected if r['event']=='response' and r['kind']=='target']
    conditioning = [r for r in selected if r['event']=='response' and r['kind']=='conditioning']
    valid = (len(targets)==12 and len(conditioning)==4 and
        all(r['token_ledger_valid'] and r['completion_status']==LOGICAL_END for r in targets+conditioning)
        and all(r['conditioning_matches'] for r in conditioning)
        and not any(r['event']=='target_skipped' for r in selected))
    return dict(passed=valid,target_responses=len(targets),conditioning_responses=len(conditioning),
                explanation='Checks restore availability, verified context counts and frozen conditioning; target answer equality is not required.')


def run(directory, config):
    cases = {c['case_id']:c for c in json.loads((directory/'state-inputs.json').read_text())}
    schedule = json.loads((directory/'schedule.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    try:
        for job in schedule['validation']:
            run_job(directory,config,job,cases[job['case_id']],manifest['model_sha256'])
        result = validate_capabilities(read_jsonl(directory/EVENT_FILE)[0])
        atomic_json(directory/'capability-check.json',result)
        if not result['passed']:
            raise RuntimeError('Hardware capability validation failed; main matrix was not started')
        print('Snapshot hardware validation passed: 12 targets and 4 conditioning responses',flush=True)
        for index,job in enumerate(schedule['main'],1):
            run_job(directory,config,job,cases[job['case_id']],manifest['model_sha256'])
            print(f'State trial {index}/{len(schedule["main"])} {job["job_id"]}',flush=True)
    except Exception as exc:
        if not (directory/'capability-check.json').exists():
            atomic_json(directory/'capability-check.json',dict(passed=False,error=f'{type(exc).__name__}: {exc}'))
        raise
    emit(directory/EVENT_FILE,'complete',planned=schedule['planned'])
