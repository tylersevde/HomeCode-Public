"""Fresh-table accuracy and application-request latency under context reuse."""
from copy import deepcopy
import json
from pathlib import Path
import random
import resource
import shutil
import time

from .common import ROOT, atomic_json, digest_file, digest_value, emit, progress, read_jsonl, wait_cool
from .diagnostic import LOGICAL_END, check_budget, renderer, reset_context, score_answer
from .hat import assistant_content, build_fixture, exact_suffix, visible_text
from .state_isolation import parameter_settings, resolve_defaults

PRESETS = ('explicit_defaults', 'penalty_1_0')
ARMS = ('rebuild', 'retain')
POSITIONS = ('beginning', 'middle', 'end', 'beginning')
EVENT_FILE = 'cache_validation.jsonl'


class Quarantine(RuntimeError):
    pass


def prior_fixtures(root):
    """Include preserved runs and their frozen copies; hashes document discovery."""
    seeds, tables, files = set(), set(), []
    paths = sorted(set(root.rglob('fixtures.json')) | set(root.rglob('cache-fixtures.json')) |
                   set(root.rglob('diagnostic-inputs.json')) | set(root.rglob('hybrid-fixtures.json')))
    for path in paths:
        value = json.loads(path.read_text())
        fixtures = value['fixtures'] if isinstance(value, dict) else value
        for fixture in fixtures:
            seeds.add(fixture['seed'])
            tables.add(digest_value(fixture['table']))
        files.append(dict(path=str(path.resolve()), sha256=digest_file(path), fixtures=len(fixtures)))
    return dict(seeds=sorted(seeds), table_sha256=sorted(tables), files=files)


def prepare_source(directory, config, inventory):
    source = Path(config['source_run']).resolve()
    checksums = json.loads((source/'checksums.json').read_text())
    for relative, expected in checksums.items():
        path = (source/relative).resolve()
        if not path.is_relative_to(source) or digest_file(path) != expected:
            raise ValueError(f'Source checksum mismatch: {relative}')
    names = ('config.json', 'manifest.json', 'state-environment.json', 'state-inputs.json',
             'state_isolation.jsonl', 'summary.json', 'outcome.json')
    if not all(name in checksums for name in names):
        raise ValueError('Source must contain the checksummed state-isolation evidence')
    old_config = json.loads((source/'config.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    if old_config['profile'] != 'state-isolation' or not summary.get('complete') or not summary.get('protocol_completed'):
        raise ValueError('Cache validation requires a complete state-isolation source run')
    manifest = json.loads((source/'manifest.json').read_text())
    for key in ('model_sha256', 'hailort_cli'):
        if manifest[key] != inventory[key]:
            raise ValueError(f'Source runtime identity differs: {key}')
    frozen = directory/'source-run'
    frozen.mkdir()
    for name in (*names, 'checksums.json'):
        shutil.copy2(source/name, frozen/name)
    atomic_json(directory/'prior-fixtures.json', prior_fixtures(ROOT/'runs'))
    atomic_json(directory/'source-provenance.json', dict(source_run=str(source),
        verified_source_artifacts=len(checksums), source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={name:digest_file(frozen/name) for name in (*names, 'checksums.json')}))
    if config.get('replay_run'):
        replay = Path(config['replay_run']).resolve()
        replay_checks = json.loads((replay/'checksums.json').read_text())
        for relative,expected in replay_checks.items():
            path = (replay/relative).resolve()
            if not path.is_relative_to(replay) or digest_file(path)!=expected:
                raise ValueError(f'Replay checksum mismatch: {relative}')
        replay_names = ('config.json','manifest.json','cache-fixtures.json','schedule.json','cache-environment.json')
        if not all(name in replay_checks for name in replay_names):
            raise ValueError('Replay requires checksummed cache fixtures, schedule and environment')
        previous = json.loads((replay/'config.json').read_text())
        for key in ('profile','hat_sizes','tables_per_size','cache_repeats','turns','fixture_seed_base','schedule_seed'):
            if previous[key]!=config[key]:
                raise ValueError(f'Replay protocol differs: {key}')
        previous_manifest = json.loads((replay/'manifest.json').read_text())
        if any(previous_manifest[k]!=inventory[k] for k in ('model_sha256','hailort_cli')):
            raise ValueError('Replay model/runtime identity differs')
        folder = directory/'replay-source'
        folder.mkdir()
        for name in (*replay_names,'checksums.json'):
            shutil.copy2(replay/name,folder/name)
        atomic_json(directory/'replay-provenance.json',dict(replay_run=str(replay),
            verified_artifacts=len(replay_checks),fresh_evidence=False,
            copied_sha256={n:digest_file(folder/n) for n in (*replay_names,'checksums.json')}))


def fresh_fixtures(render, tokenize, config, prior, heartbeat=lambda: None):
    seen_seeds, seen_tables = set(prior['seeds']), set(prior['table_sha256'])
    fixtures = []
    for size in config['hat_sizes']:
        for index in range(config['tables_per_size']):
            heartbeat()
            seed = config['fixture_seed_base'] + size*1000 + index
            if seed in seen_seeds:
                raise ValueError(f'Fresh fixture seed collision: {seed}')
            fixture = build_fixture(render, tokenize, size, seed, config['turns'])
            table_hash = digest_value(fixture['table'])
            if table_hash in seen_tables:
                raise ValueError(f'Fresh fixture table collision: {seed}')
            seen_seeds.add(seed)
            seen_tables.add(table_hash)
            fixture.update(fixture_id=f'fresh-n{size}-{index+1:02d}', table_index=index,
                           table_sha256=table_hash, positions=list(POSITIONS))
            fixtures.append(fixture)
    return fixtures


def make_schedule(fixtures, config):
    main = []
    for fixture in fixtures:
        for preset in PRESETS:
            for repeat in range(config['cache_repeats']):
                arms = list(ARMS)
                if (fixture['table_index']+repeat) % 2:
                    arms.reverse()
                main.append(dict(job_id=f'{fixture["fixture_id"]}-{preset}-r{repeat+1}',
                    fixture_id=fixture['fixture_id'], target_tokens=fixture['target_tokens'],
                    preset=preset, repeat=repeat, arms=arms, turns=config['turns'], phase='main'))
    random.Random(config['schedule_seed']).shuffle(main)
    validation = [dict(job_id=f'validation-small-{p}',fixture_id='validation-small',
        target_tokens=128,preset=p,repeat=0,arms=list(ARMS),turns=2,phase='validation') for p in PRESETS]
    return dict(main=main, validation=validation, expected_main_responses=len(main)*2*config['turns'],
                expected_validation_responses=8)


def sentinel(source_cases, source_rows):
    case = next(c for c in source_cases if c['case_id']=='small_failure')
    first = case['conditioning'][0]
    fixture = dict(fixture_id='validation-small', seed=None, target_tokens=128,
        initial_messages=deepcopy(case['messages'][:2]),
        questions=[f'What color is {first["queried_item"]}?',case['messages'][-1]['content']],
        answers=[first['expected_answer'],case['expected_answer']], positions=['beginning','middle'])
    expected = {}
    for preset in PRESETS:
        expected[preset] = {}
        for arm in ARMS:
            candidates = [r for r in source_rows if r['event']=='response' and r['phase']=='main'
                and r['kind']=='target' and r['case_id']=='small_failure' and r['preset']==preset
                and r['mode']==('live' if arm=='retain' else 'rebuild')]
            if len(candidates)!=3 or len({r['output'] for r in candidates})!=1:
                raise ValueError('Source does not contain three agreeing small-case controls')
            expected[preset][arm] = [first['output'], candidates[0]['output']]
    return fixture, expected


def prepare_request(render, messages, accumulated, arm):
    full = render(messages)
    if not full.startswith(accumulated):
        raise Quarantine('Chat rendering does not extend the actual generated transcript')
    return full, full if arm=='rebuild' else full[len(accumulated):]


def timed_request(llm, prepare, arm, parameters, stops):
    """One continuous application interval; no tokenization/ledger RPCs here."""
    chunks, first, clear_ms = [], None, 0.0
    cpu = time.process_time()
    started = time.perf_counter()
    full, submitted = prepare()
    prepared = time.perf_counter()
    if arm=='rebuild':
        clear_start = time.perf_counter()
        llm.clear_context()
        clear_ms = (time.perf_counter()-clear_start)*1000
    generated = time.perf_counter()
    with llm.generate(submitted, **parameters) as gen:
        while str(gen.generation_status).endswith('.GENERATING'):
            chunk = gen.read(timeout_ms=90000)
            chunks.append(chunk)
            if first is None and visible_text(''.join(chunks),stops).strip():
                first = time.perf_counter()
        status = str(gen.generation_status).split('.')[-1]
    ended = time.perf_counter()
    return dict(output=''.join(chunks),stream_chunks=chunks,stream_events=len(chunks),
        completion_status=status,request_ms=(ended-started)*1000,
        first_visible_ms=(first-started)*1000 if first is not None else None,
        prompt_preparation_ms=(prepared-started)*1000,clear_context_ms=clear_ms,
        generation_ms=(ended-generated)*1000,cpu_seconds=time.process_time()-cpu,
        effective_prompt=full,submitted_prompt=submitted)


def quarantine(llm, directory, job, arm, first_skipped, completed, reason):
    reset_context(llm)
    emit(directory/EVENT_FILE,'session_quarantined',job_id=job['job_id'],phase=job['phase'],
         arm=arm,completed_turns=completed,reason=str(reason),reset_verified=True)
    for turn in range(first_skipped,job['turns']):
        emit(directory/EVENT_FILE,'turn_skipped',job_id=job['job_id'],phase=job['phase'],
            fixture_id=job['fixture_id'],target_tokens=job['target_tokens'],preset=job['preset'],
            repeat=job['repeat'],arm=arm,turn=turn,reason=str(reason))


def run_session(llm, render, fixture, job, arm, directory, parameters, stops, limit):
    began = time.perf_counter()
    reset_context(llm)
    setup_ms = (time.perf_counter()-began)*1000
    messages, accumulated, results = deepcopy(fixture['initial_messages']), '', []
    for turn in range(job['turns']):
        progress(directory,'hat','cache_generation',job_id=job['job_id'],arm=arm,turn=turn)
        if turn:
            messages.append(dict(role='user',content=fixture['questions'][turn]))
        prep = lambda: prepare_request(render,messages,accumulated,arm)
        check_start = time.perf_counter()
        try:
            full, submitted = prep()
            try:
                suffix, old, new, ids = exact_suffix(full,accumulated,llm.tokenize)
            except ValueError as exc:
                raise Quarantine(str(exc)) from exc
            check_budget(llm,ids,limit)
            before = llm.get_context_usage_size()
            if before != len(old):
                raise Quarantine(f'Before-request context mismatch: {before} != {len(old)}')
        except Quarantine as exc:
            quarantine(llm,directory,job,arm,turn,len(results),exc)
            break
        precheck_ms = (time.perf_counter()-check_start)*1000
        result = timed_request(llm,prep,arm,parameters,stops)
        post_start = time.perf_counter()
        if result['effective_prompt']!=full or result['submitted_prompt']!=submitted:
            raise RuntimeError('Timed preparation differs from the prevalidated request')
        after = llm.get_context_usage_size()
        expected_after = len(llm.tokenize(full+result['output']))
        postcheck_ms = (time.perf_counter()-post_start)*1000
        result.update(job_id=job['job_id'],phase=job['phase'],fixture_id=fixture['fixture_id'],
            fixture_seed=fixture['seed'],target_tokens=fixture['target_tokens'],preset=job['preset'],
            repeat=job['repeat'],arm=arm,arm_order=job['arms'].index(arm),turn=turn,
            position=fixture['positions'][turn],messages=deepcopy(messages),
            expected_answer=fixture['answers'][turn],queried_item=fixture['questions'][turn].split()[-1].rstrip('?'),
            effective_input_sha256=digest_value(ids),effective_input_tokens=len(ids),
            submitted_input_tokens=len(ids) if arm=='rebuild' else len(new),
            pre_request_context=before,context_before=0 if arm=='rebuild' else before,
            context_after=after,expected_context_after=expected_after,token_ledger_valid=after==expected_after,
            effective_parameters=dict(parameters),effective_parameters_sha256=digest_value(parameters),
            diagnostic_precheck_ms=precheck_ms,diagnostic_postcheck_ms=postcheck_ms,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        result.update(score_answer(result['output'],result['expected_answer'],result['queried_item'],stops,result['completion_status']))
        emit(directory/EVENT_FILE,'response',**result)
        results.append(result)
        if after>limit:
            raise RuntimeError('Observed context exceeds experiment budget')
        if not result['token_ledger_valid'] or result['completion_status']!=LOGICAL_END:
            reason = 'runtime_context_count_mismatch' if not result['token_ledger_valid'] else result['completion_status']
            quarantine(llm,directory,job,arm,turn+1,len(results),reason)
            break
        accumulated = full+result['output']
        try:
            content = assistant_content(result['output'],stops)
        except ValueError as exc:
            quarantine(llm,directory,job,arm,turn+1,len(results),exc)
            break
        messages.append(dict(role='assistant',content=content))
    emit(directory/EVENT_FILE,'session_complete',job_id=job['job_id'],phase=job['phase'],
        fixture_id=fixture['fixture_id'],target_tokens=fixture['target_tokens'],preset=job['preset'],
        repeat=job['repeat'],arm=arm,arm_order=job['arms'].index(arm),completed_turns=len(results),
        expected_turns=job['turns'],session_wall_ms=(time.perf_counter()-began)*1000,
        setup_reset_ms=setup_ms,request_total_ms=sum(r['request_ms'] for r in results),
        diagnostic_total_ms=sum(r['diagnostic_precheck_ms']+r['diagnostic_postcheck_ms'] for r in results))
    return results


def validate_control(rows, expected):
    responses = [r for r in rows if r['event']=='response' and r['phase']=='validation']
    keys = {(r['preset'],r['arm'],r['turn']) for r in responses}
    required = {(p,a,t) for p in PRESETS for a in ARMS for t in range(2)}
    passed = len(responses)==8 and keys==required and all(
        r['token_ledger_valid'] and r['completion_status']==LOGICAL_END and
        r['output']==expected[r['preset']][r['arm']][r['turn']] for r in responses)
    return dict(passed=passed,observed=len(responses),expected=8,expected_outputs=expected)


def run(directory, config):
    from hailo_platform import VDevice, __version__
    from hailo_platform.pyhailort.pyhailort import LLM
    progress(directory,'hat','model_loading')
    started = time.perf_counter()
    with VDevice() as device, LLM(device,config['model']) as llm:
        loading = time.perf_counter()-started
        source = json.loads((directory/'source-run/state-environment.json').read_text())
        env = dict(hailort_version=__version__,prompt_template=llm.prompt_template(),
            stop_tokens=llm.get_stop_tokens(),generation_recovery_sequence=llm.get_generation_recovery_sequence(),
            capacity_tokens=llm.max_context_capacity(),model_defaults=resolve_defaults(llm,__version__))
        for key,value in env.items():
            if source[key]!=value:
                raise RuntimeError(f'Runtime metadata differs from source: {key}')
        settings = {p:parameter_settings(env['model_defaults'],p)[0] for p in PRESETS}
        limit = min(1792,env['capacity_tokens']-256)
        env.update(model=config['model'],loading_seconds=loading,experiment_limit_tokens=limit,settings=settings,
            timing_definition='Continuous rendering/suffix construction, required rebuild clear, generate/read and generator cleanup. Pre/post diagnostic RPCs, scoring, logging and initial verified session reset are outside request timing. Timed preparation must equal prevalidated preparation.')
        atomic_json(directory/'cache-environment.json',env)
        render = renderer(env['prompt_template'])
        prior = dict(seeds=[],table_sha256=[]) if config.get('replay_run') else json.loads((directory/'prior-fixtures.json').read_text())
        fixtures = fresh_fixtures(render,llm.tokenize,config,prior,
                                  lambda:progress(directory,'hat','freeze_fixtures'))
        schedule = make_schedule(fixtures,config)
        if config.get('replay_run'):
            if fixtures!=json.loads((directory/'replay-source/cache-fixtures.json').read_text()) or schedule!=json.loads((directory/'replay-source/schedule.json').read_text()):
                raise RuntimeError('Regenerated replay inputs/schedule differ from frozen evidence')
            if settings!=json.loads((directory/'replay-source/cache-environment.json').read_text())['settings']:
                raise RuntimeError('Replay generation settings differ')
        atomic_json(directory/'cache-fixtures.json',fixtures)
        atomic_json(directory/'schedule.json',schedule)
        source_rows, errors = read_jsonl(directory/'source-run/state_isolation.jsonl')
        if errors:
            raise RuntimeError('Source control observations are damaged')
        control, expected = sentinel(json.loads((directory/'source-run/state-inputs.json').read_text()),source_rows)
        atomic_json(directory/'control-input.json',dict(fixture=control,expected_outputs=expected))
        by_id = {f['fixture_id']:f for f in fixtures}
        by_id[control['fixture_id']] = control
        for phase in ('validation','main'):
            for index,job in enumerate(schedule[phase],1):
                emit(directory/EVENT_FILE,'job_start',**job)
                for arm in job['arms']:
                    wait_cool(directory)
                    run_session(llm,render,by_id[job['fixture_id']],job,arm,directory,
                                settings[job['preset']],env['stop_tokens'],limit)
                reset_context(llm)
                emit(directory/EVENT_FILE,'job_complete',job_id=job['job_id'],phase=phase)
                print(f'Cache validation {phase} {index}/{len(schedule[phase])}: {job["job_id"]}',flush=True)
            if phase=='validation':
                result = validate_control(read_jsonl(directory/EVENT_FILE)[0],expected)
                atomic_json(directory/'control-check.json',result)
                if not result['passed']:
                    raise RuntimeError('Saved small-case controls did not reproduce; main matrix was not started')
        reset_context(llm)
    emit(directory/EVENT_FILE,'complete',expected_main_responses=schedule['expected_main_responses'])
