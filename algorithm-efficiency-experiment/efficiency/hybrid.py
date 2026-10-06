"""Exact-record retrieval, verified HAT answers, and supervised workflow comparison."""
from copy import deepcopy
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
import re
import shutil
import time
from types import MappingProxyType

from .common import ROOT, atomic_json, digest_file, digest_value, emit, progress, read_jsonl, wait_cool
from .cache_validation import Quarantine, fresh_fixtures, prior_fixtures, timed_request
from .diagnostic import LOGICAL_END, check_budget, renderer, reset_context, score_answer
from .hat import COLORS, SYSTEM, assistant_content, exact_suffix, visible_text
from .state_isolation import parameter_settings, resolve_defaults

WORKFLOWS = ('cpu', 'full_table', 'hybrid')
EVENT_FILE = 'hybrid.jsonl'


@dataclass(frozen=True, init=False)
class FactIndex:
    """An immutable, validated index; never accepts benchmark answer labels."""
    records: tuple
    _values: object
    table_sha256: str

    def __init__(self, records):
        if not isinstance(records, list):
            raise ValueError('Facts must be a JSON array of {item, color} records')
        values = {}
        frozen = []
        for row in records:
            if not isinstance(row, dict) or set(row) != {'item', 'color'}:
                raise ValueError('Each record must contain exactly item and color')
            item, color = row['item'], row['color']
            self.validate_item(item)
            if not isinstance(color, str) or color not in COLORS:
                raise ValueError(f'Unsupported color for {item}')
            if item in values:
                raise ValueError(f'Duplicate identifier: {item}')
            values[item] = (color, digest_value(dict(item=item, color=color)))
            frozen.append((item, color))
        object.__setattr__(self, 'records', tuple(frozen))
        object.__setattr__(self, '_values', MappingProxyType(values))
        object.__setattr__(self, 'table_sha256', digest_value(frozen))

    @staticmethod
    def validate_item(item):
        if not isinstance(item, str) or not re.fullmatch(r'item[0-9]+', item):
            raise ValueError('Identifier must match item[0-9]+ exactly')

    def lookup(self, item):
        self.validate_item(item)
        return self._values.get(item)


def index_fixture(fixture):
    return FactIndex([dict(item=k, color=v) for k, v in fixture['table']])


def cpu_answer(index, item):
    start = time.perf_counter()
    evidence = index.lookup(item)
    result = dict(status='ok' if evidence else 'not_found', queried_item=item,
        answer=evidence[0] if evidence else None, answer_source='cpu' if evidence else None,
        evidence_sha256=evidence[1] if evidence else None, fallback_reason=None)
    result['request_ms'] = (time.perf_counter()-start)*1000
    return result


def generate_checked(llm, render, messages, accumulated, clear, parameters, stops, limit):
    """RPC checks are intentionally within the caller's complete request timer."""
    prepare_start = time.perf_counter()
    full = render(messages)
    try:
        submitted, old, new, ids = exact_suffix(full, accumulated, llm.tokenize)
    except ValueError as exc:
        raise Quarantine(str(exc)) from exc
    check_budget(llm, ids, limit)
    prepared_ms = (time.perf_counter()-prepare_start)*1000
    reset_ms = 0.0
    if clear:
        reset_start = time.perf_counter()
        reset_context(llm)
        reset_ms = (time.perf_counter()-reset_start)*1000
    before = llm.get_context_usage_size()
    if before != len(old):
        raise Quarantine(f'Before-request context mismatch: {before} != {len(old)}')
    result = timed_request(llm, lambda: (full, submitted), 'retain', parameters, stops)
    diagnostic_start = time.perf_counter()
    after = llm.get_context_usage_size()
    expected_after = len(llm.tokenize(full+result['output']))
    if after > limit:
        raise RuntimeError('Observed context exceeds experiment budget')
    result.update(messages=deepcopy(messages), effective_input_sha256=digest_value(ids),
        effective_input_tokens=len(ids), submitted_input_tokens=len(new),
        context_before=before, context_after=after, expected_context_after=expected_after,
        token_ledger_valid=after == expected_after, effective_parameters=dict(parameters),
        preparation_check_ms=prepared_ms, verified_reset_ms=reset_ms,
        postcheck_ms=(time.perf_counter()-diagnostic_start)*1000)
    result['generation_request_ms'] = result.pop('request_ms')
    return result


def answer(index, item, llm, render, parameters, stops, limit):
    """Return a verified answer, recording every model correction explicitly."""
    start = time.perf_counter()
    evidence = index.lookup(item)
    retrieved = time.perf_counter()
    if evidence is None:
        return dict(status='not_found', queried_item=item, answer=None, answer_source=None,
            fallback_reason=None, evidence_sha256=None, request_ms=(retrieved-start)*1000)
    color, evidence_hash = evidence
    messages = [dict(role='system', content=SYSTEM), dict(role='user',
        content=f'Reference facts: {item} = {color}. What color is {item}?')]
    result = generate_checked(llm, render, messages, '', True, parameters, stops, limit)
    validation_start = time.perf_counter()
    score = score_answer(result['output'], color, item, stops, result['completion_status'])
    accepted = score['strict_correct'] and result['token_ledger_valid']
    reason = (None if accepted else 'invalid_context_ledger' if not result['token_ledger_valid']
              else score['score_reason'] or ('format_violation' if score['factual_correct'] else 'wrong_fact'))
    # A damaged/incomplete request is isolated before returning even the CPU fallback.
    recovery_ms = 0.0
    if not result['token_ledger_valid'] or result['completion_status'] != LOGICAL_END:
        recovery_start = time.perf_counter()
        reset_context(llm)
        recovery_ms = (time.perf_counter()-recovery_start)*1000
    result.update(score, status='ok', queried_item=item,
        answer=score['parsed_color'] if accepted else color,
        answer_source='hat' if accepted else 'cpu_fallback', fallback_reason=reason,
        evidence_sha256=evidence_hash, retrieval_ms=(retrieved-start)*1000,
        validation_ms=(time.perf_counter()-validation_start)*1000,
        recovery_reset_ms=recovery_ms)
    result['request_ms'] = (time.perf_counter()-start)*1000
    return result


def verify_artifacts(source):
    source = Path(source).resolve()
    checks = json.loads((source/'checksums.json').read_text())
    for relative, expected in checks.items():
        path = (source/relative).resolve()
        if not path.is_relative_to(source) or digest_file(path) != expected:
            raise ValueError(f'Source checksum mismatch: {relative}')
    return checks


def select_controls(fixtures, rows):
    """First table at each size, all four retained turns, two agreeing source repeats."""
    selected = []
    for size in (128, 512, 1536):
        fixture = min((f for f in fixtures if f['target_tokens'] == size), key=lambda f:f['fixture_id'])
        outputs = []
        for turn in range(4):
            matches = [r for r in rows if r['event']=='response' and r['phase']=='main'
                and r['fixture_id']==fixture['fixture_id'] and r['preset']=='penalty_1_0'
                and r['arm']=='retain' and r['turn']==turn]
            if (len(matches)!=2 or {r['repeat'] for r in matches}!={0,1}
                or len({r['output'] for r in matches})!=1
                or not all(r['token_ledger_valid'] and r['completion_status']==LOGICAL_END for r in matches)):
                raise ValueError('Source controls require two complete agreeing repetitions')
            outputs.append(matches[0]['output'])
        selected.append(dict(fixture=fixture, expected_outputs=outputs))
    return selected


def prepare_source(directory, config, inventory):
    source = Path(config['source_run']).resolve()
    checks = verify_artifacts(source)
    names = ('config.json','manifest.json','summary.json','outcome.json','cache-environment.json',
             'cache-fixtures.json','cache_validation.jsonl','schedule.json')
    if not all(n in checks for n in names):
        raise ValueError('Source must include checksummed cache-validation artifacts')
    previous = json.loads((source/'config.json').read_text())
    summary = json.loads((source/'summary.json').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    groups = [g for g in summary.get('groups',[]) if g['preset']=='penalty_1_0']
    if (previous['profile']!='cache-validation' or not summary.get('protocol_completed')
        or {g['target_tokens'] for g in groups}!={128,512,1536}
        or not all(g['matched_coverage_complete'] and g['eligible_paired_turns']==80
                   and g['exact_output_matches']==80 and g['complete_independent_tables']==10 for g in groups)):
        raise ValueError('Source penalty-1.0 subgroup must have complete matched coverage')
    if any(manifest[k]!=inventory[k] for k in ('model_sha256','hailort_cli')):
        raise ValueError('Source model/runtime identity differs')
    rows, errors = read_jsonl(source/'cache_validation.jsonl')
    if errors:
        raise ValueError('Source observations are damaged')
    controls = select_controls(json.loads((source/'cache-fixtures.json').read_text()), rows)
    frozen = directory/'source-run'
    frozen.mkdir()
    for name in (*names,'checksums.json'):
        shutil.copy2(source/name, frozen/name)
    atomic_json(directory/'control-inputs.json', controls)
    atomic_json(directory/'prior-fixtures.json', prior_fixtures(ROOT/'runs'))
    atomic_json(directory/'source-provenance.json', dict(source_run=str(source),
        verified_artifacts=len(checks), source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={n:digest_file(frozen/n) for n in (*names,'checksums.json')}))
    if config.get('replay_run'):
        replay = Path(config['replay_run']).resolve()
        replay_checks = verify_artifacts(replay)
        replay_names = ('config.json','manifest.json','hybrid-fixtures.json','schedule.json','hybrid-environment.json')
        if not all(n in replay_checks for n in replay_names):
            raise ValueError('Replay is missing frozen inputs')
        old = json.loads((replay/'config.json').read_text())
        for key in ('profile','hat_sizes','tables_per_size','cache_repeats','turns','fixture_seed_base','schedule_seed'):
            if old[key]!=config[key]:
                raise ValueError(f'Replay protocol differs: {key}')
        old_manifest = json.loads((replay/'manifest.json').read_text())
        if any(old_manifest[k]!=inventory[k] for k in ('model_sha256','hailort_cli')):
            raise ValueError('Replay runtime differs')
        folder = directory/'replay-source'
        folder.mkdir()
        for name in (*replay_names,'checksums.json'):
            shutil.copy2(replay/name, folder/name)
        atomic_json(directory/'replay-provenance.json',dict(replay_run=str(replay),
            verified_artifacts=len(replay_checks), fresh_evidence=False))


def make_schedule(fixtures, config):
    permutations = list(itertools.permutations(WORKFLOWS))
    permutations = [permutations[i] for i in (0,3,4,1,2,5)]
    jobs = []
    for size in config['hat_sizes']:
        fs = sorted((f for f in fixtures if f['target_tokens']==size), key=lambda f:f['table_index'])
        for f in fs:
            for repeat in range(config['cache_repeats']):
                # Across 20 blocks/size every workflow occupies each slot 6 or 7 times.
                order = permutations[(f['table_index']*config['cache_repeats']+repeat)%6]
                jobs.append(dict(job_id=f'{f["fixture_id"]}-r{repeat+1}', fixture_id=f['fixture_id'],
                    target_tokens=size, repeat=repeat, workflows=list(order), turns=config['turns'], phase='main'))
    random.Random(config['schedule_seed']).shuffle(jobs)
    return dict(main=jobs, expected_main_answers=len(jobs)*3*config['turns'],
                expected_main_generations=len(jobs)*2*config['turns'], expected_control_generations=20)


def metadata(fixture, job, workflow, turn):
    return dict(job_id=job['job_id'], phase=job['phase'], fixture_id=fixture['fixture_id'],
        target_tokens=fixture['target_tokens'], fixture_seed=fixture['seed'], repeat=job['repeat'],
        workflow=workflow, workflow_order=job['workflows'].index(workflow), turn=turn,
        position=fixture['positions'][turn], expected_answer=fixture['answers'][turn],
        table_sha256=digest_value(fixture['table']))


def run_workflow(llm, render, fixture, index, job, workflow, directory, parameters, stops, limit):
    began = time.perf_counter()
    results, messages, accumulated = [], deepcopy(fixture['initial_messages']), ''
    for turn in range(job['turns']):
        progress(directory,'hat','hybrid_generation', job_id=job['job_id'], workflow=workflow, turn=turn)
        item = fixture['questions'][turn].split()[-1].rstrip('?')
        if workflow=='cpu':
            result = cpu_answer(index, item)
        elif workflow=='hybrid':
            result = answer(index, item, llm, render, parameters, stops, limit)
        else:
            start = time.perf_counter()
            if turn:
                messages.append(dict(role='user',content=fixture['questions'][turn]))
            try:
                result = generate_checked(llm,render,messages,accumulated,turn==0,parameters,stops,limit)
            except Quarantine as exc:
                # Prefix/ledger failure before inference quarantines only this conversation.
                reset_context(llm)
                for skipped in range(turn,job['turns']):
                    emit(directory/EVENT_FILE,'turn_skipped',**metadata(fixture,job,workflow,skipped),reason=str(exc))
                break
            result.update(score_answer(result['output'],fixture['answers'][turn],item,stops,result['completion_status']))
            result.update(status='ok', queried_item=item, answer=visible_text(result['output'],stops).strip(),
                answer_source='hat', fallback_reason=None)
            result['request_ms'] = (time.perf_counter()-start)*1000
        result.update(metadata(fixture,job,workflow,turn))
        result['final_correct'] = (result['strict_correct'] if workflow=='full_table'
                                   else result['answer']==fixture['answers'][turn])
        emit(directory/EVENT_FILE,'response',**result)
        results.append(result)
        if workflow=='full_table':
            reason = None
            if not result['token_ledger_valid'] or result['completion_status']!=LOGICAL_END:
                reason = 'invalid_context_ledger' if not result['token_ledger_valid'] else result['completion_status']
            else:
                accumulated = result['effective_prompt']+result['output']
                try:
                    messages.append(dict(role='assistant',content=assistant_content(result['output'],stops)))
                except ValueError as exc:
                    reason = str(exc)
            if reason:
                reset_context(llm)
                for skipped in range(turn+1,job['turns']):
                    emit(directory/EVENT_FILE,'turn_skipped',**metadata(fixture,job,workflow,skipped),reason=reason)
                break
    emit(directory/EVENT_FILE,'session_complete',job_id=job['job_id'],phase=job['phase'],workflow=workflow,
        completed_turns=len(results),expected_turns=job['turns'],request_total_ms=sum(r['request_ms'] for r in results),
        session_wall_ms=(time.perf_counter()-began)*1000)
    return results


def run_controls(llm, render, directory, parameters, stops, limit):
    checks = []
    for control in json.loads((directory/'control-inputs.json').read_text()):
        f = control['fixture']
        job = dict(job_id='control-'+f['fixture_id'],phase='control',repeat=0,workflows=['full_table'],turns=4)
        wait_cool(directory)
        rows = run_workflow(llm,render,f,index_fixture(f),job,'full_table',directory,parameters,stops,limit)
        checks.append(dict(case=job['job_id'],passed=len(rows)==4 and all(
            r['output']==expected and r['token_ledger_valid'] and r['completion_status']==LOGICAL_END
            for r,expected in zip(rows,control['expected_outputs']))))
    for color in COLORS:
        progress(directory,'hat','single_record_control',color=color)
        wait_cool(directory)
        r = answer(FactIndex([dict(item='item001',color=color)]),'item001',llm,render,parameters,stops,limit)
        emit(directory/EVENT_FILE,'response',phase='color_control',workflow='hybrid',expected_answer=color,**r)
        checks.append(dict(case=color,passed=r['answer']==color,raw_strict_correct=r['strict_correct'],
                           answer_source=r['answer_source']))
    result = dict(passed=all(c['passed'] for c in checks),checks=checks,expected_generations=20)
    atomic_json(directory/'control-check.json',result)
    if not result['passed']:
        raise RuntimeError('Frozen baseline controls or verified color controls failed')


def run(directory, config):
    if config['profile']=='hybrid-query':
        started = time.perf_counter()
        index = FactIndex(json.loads((directory/'facts.json').read_text()))
        index_ms = (time.perf_counter()-started)*1000
        if index.lookup(config['item']) is None:
            result = cpu_answer(index,config['item'])
            result['index_build_ms'] = index_ms
            atomic_json(directory/'result.json',result)
            emit(directory/EVENT_FILE,'complete',generation_count=0)
            return
    from hailo_platform import VDevice, __version__
    from hailo_platform.pyhailort.pyhailort import LLM
    progress(directory,'hat','model_loading')
    started = time.perf_counter()
    with VDevice() as device, LLM(device,config['model']) as llm:
        loading_seconds = time.perf_counter()-started
        env = dict(hailort_version=__version__,prompt_template=llm.prompt_template(),
            stop_tokens=llm.get_stop_tokens(),generation_recovery_sequence=llm.get_generation_recovery_sequence(),
            capacity_tokens=llm.max_context_capacity(),model_defaults=resolve_defaults(llm,__version__))
        if config['profile']=='hybrid-validation':
            source_env = json.loads((directory/'source-run/cache-environment.json').read_text())
            if any(env[k]!=source_env[k] for k in env):
                raise RuntimeError('Source runtime metadata differs')
        parameters = parameter_settings(env['model_defaults'],'penalty_1_0')[0]
        limit = min(1792,env['capacity_tokens']-256)
        env.update(parameters=parameters,loading_seconds=loading_seconds,experiment_limit_tokens=limit,
            timing_definition='Continuous request: retrieval/rendering, token/context checks, required verified reset, generation/cleanup, output validation and fallback/recovery. Logging, index construction and model loading are separate. First-visible time is generation-only, not time to verified answer.')
        atomic_json(directory/'hybrid-environment.json',env)
        render, stops = renderer(env['prompt_template']), env['stop_tokens']
        if config['profile']=='hybrid-query':
            wait_cool(directory)
            result = answer(index,config['item'],llm,render,parameters,stops,limit)
            result.update(index_build_ms=index_ms,table_sha256=index.table_sha256)
            atomic_json(directory/'result.json',result)
            emit(directory/EVENT_FILE,'response',phase='query',workflow='hybrid',**result)
        else:
            prior = dict(seeds=[],table_sha256=[]) if config.get('replay_run') else json.loads((directory/'prior-fixtures.json').read_text())
            fixtures = fresh_fixtures(render,llm.tokenize,config,prior,lambda:progress(directory,'hat','freeze_fixtures'))
            # JSON round-trip avoids tuple/list differences when verifying replay artifacts.
            fixtures = json.loads(json.dumps(fixtures))
            schedule = make_schedule(fixtures,config)
            if config.get('replay_run'):
                folder = directory/'replay-source'
                if (fixtures!=json.loads((folder/'hybrid-fixtures.json').read_text())
                    or schedule!=json.loads((folder/'schedule.json').read_text())
                    or parameters!=json.loads((folder/'hybrid-environment.json').read_text())['parameters']):
                    raise RuntimeError('Replay inputs, schedule or parameters differ')
            atomic_json(directory/'hybrid-fixtures.json',fixtures)
            atomic_json(directory/'schedule.json',schedule)
            by_id = {f['fixture_id']:f for f in fixtures}
            indexes = {}
            for f in fixtures:
                start = time.perf_counter()
                indexes[f['fixture_id']] = index_fixture(f)
                emit(directory/EVENT_FILE,'index_built',fixture_id=f['fixture_id'],rows=len(f['table']),
                    index_build_ms=(time.perf_counter()-start)*1000,table_sha256=indexes[f['fixture_id']].table_sha256)
            atomic_json(directory/'example-records.json',[dict(item=k,color=v) for k,v in fixtures[0]['table']])
            run_controls(llm,render,directory,parameters,stops,limit)
            for number, job in enumerate(schedule['main'],1):
                fixture = by_id[job['fixture_id']]
                for workflow in job['workflows']:
                    wait_cool(directory)
                    run_workflow(llm,render,fixture,indexes[job['fixture_id']],job,workflow,directory,parameters,stops,limit)
                reset_context(llm)
                print(f'Hybrid validation {number}/{len(schedule["main"])}: {job["job_id"]}',flush=True)
        reset_context(llm)
    emit(directory/EVENT_FILE,'complete')
