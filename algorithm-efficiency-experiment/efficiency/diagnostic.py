"""Frozen-prompt repeatability and balanced position/turn/context diagnostics."""
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import resource
import shutil
import time

from jinja2 import Environment

from .common import atomic_json, digest_file, digest_value, emit, progress, read_jsonl, wait_cool
from .hat import COLORS, SYSTEM, assistant_content, exact_suffix, generate, normalize_answer

FAILURES = (('n128-pair06', 1), ('n512-pair03', 1), ('n1536-pair01', 2))
ORDERS = (('beginning', 'middle', 'end'), ('middle', 'end', 'beginning'),
          ('end', 'beginning', 'middle'))
LOGICAL_END = 'LOGICAL_END_OF_GENERATION'


def renderer(template):
    compiled = Environment().from_string(template)
    return lambda messages: compiled.render(messages=messages, tools=None, add_generation_prompt=True)


def score_answer(output, expected, queried_item, stop_tokens, status=LOGICAL_END):
    """Accept only complete, unambiguous answers about the exact queried item."""
    result = dict(legacy_strict_correct=normalize_answer(output, stop_tokens) == expected,
                  factual_correct=None, strict_correct=False, format_correct=False,
                  parsed_color=None, answer_category='malformed', score_reason=None)
    if status != LOGICAL_END:
        result.update(answer_category='truncated' if status == 'MAX_TOKENS_REACHED' else 'incomplete',
                      score_reason=status)
        return result
    try:
        body = assistant_content(output, stop_tokens)
    except ValueError:
        result['score_reason'] = 'missing_terminal_token'
        return result
    # Remove ONE terminal token only. Embedded terminators/roles remain malformed.
    if '<|' in body or '|>' in body or any(s in body for s in stop_tokens):
        result['score_reason'] = 'exposed_control_marker'
        return result
    text = ' '.join(body.casefold().split()).strip().strip('.,!?;:').strip()
    colors = '|'.join(COLORS)
    one_word = re.fullmatch(f'({colors})', text)
    patterns = [f'the color of (item[0-9]+) is ({colors})',
                f'(item[0-9]+) is ({colors})', f'(item[0-9]+) = ({colors})']
    color = one_word[1] if one_word else None
    if color is None:
        for pattern in patterns:
            matched = re.fullmatch(pattern, text)
            if matched:
                if matched[1] != queried_item.casefold():
                    result['score_reason'] = 'wrong_identifier'
                    return result
                color = matched[2]
                break
    if color is None:
        matched = re.fullmatch(f'it is ({colors})', text)
        color = matched[1] if matched else None
    if color is None:
        result['score_reason'] = 'unsupported_or_ambiguous_answer'
        return result
    correct = color == expected
    result.update(parsed_color=color, factual_correct=correct, format_correct=bool(one_word),
                  strict_correct=bool(one_word) and correct,
                  answer_category='correct_fact' if correct else 'wrong_fact')
    return result


def reconstruct_messages(fixture, rebuild_rows, turn, stop_tokens):
    messages = deepcopy(fixture['initial_messages'])
    by_turn = {r['turn']: r for r in rebuild_rows}
    for earlier in range(turn):
        messages.append(dict(role='assistant', content=assistant_content(by_turn[earlier]['output'], stop_tokens)))
        messages.append(dict(role='user', content=fixture['questions'][earlier+1]))
    return messages


def select_inputs(fixtures, rows, environment):
    """Deterministic selection independent of any new diagnostic results."""
    indexed = {f['pair_id']: f for f in fixtures}
    measurements = {(r['pair_id'], r['arm'], r['turn']): r for r in rows if r['event'] == 'measurement'}
    render = renderer(environment['prompt_template'])
    cases, selected = [], []
    for pair_id, failure_turn in FAILURES:
        fixture = indexed[pair_id]
        selected.append(dict(deepcopy(fixture), selection='original_divergence'))
        candidates = sorted(f['pair_id'] for f in fixtures
            if f['target_tokens'] == fixture['target_tokens'] and f['pair_id'] != pair_id
            and all(measurements[(f['pair_id'], arm, 0)]['correct'] for arm in ('rebuild', 'retain')))
        if not candidates:
            raise ValueError(f'No distinct first-turn control for {pair_id}')
        selected.append(dict(deepcopy(indexed[candidates[0]]), selection='first_turn_correct_control'))
        rebuild = [r for (p, a, t), r in measurements.items() if p == pair_id and a == 'rebuild']
        for turn in (0, failure_turn):
            original = measurements[(pair_id, 'rebuild', turn)]
            retained = measurements[(pair_id, 'retain', turn)]
            messages = reconstruct_messages(fixture, rebuild, turn, environment['stop_tokens'])
            if render(messages) != original['effective_prompt']:
                raise ValueError(f'Original prompt cannot be reconstructed byte-for-byte: {pair_id}, {turn}')
            if original['effective_input_sha256'] != retained['effective_input_sha256']:
                raise ValueError('Selected original divergence has unequal recorded inputs')
            item = re.fullmatch(r'What color is (item[0-9]+)\?', fixture['questions'][turn])[1]
            cases.append(dict(case_id=f'{pair_id}-turn{turn+1}', source_pair_id=pair_id,
                source_turn=turn, turn=turn, target_tokens=fixture['target_tokens'],
                case_kind='first_turn_control' if turn == 0 else 'original_divergence',
                position=('beginning', 'middle', 'end')[turn], queried_item=item,
                expected_answer=fixture['answers'][turn], messages=messages,
                effective_prompt=original['effective_prompt'],
                effective_input_sha256=original['effective_input_sha256'],
                original_rebuild_output=original['output'], original_retain_output=retained['output']))
    return dict(cases=cases, fixtures=selected)


def make_schedule(inputs, config):
    cases, fixtures = inputs['cases'], inputs['fixtures']
    if config['profile'] == 'diagnostic-smoke':
        cases = cases[:2]
        fixtures = fixtures[:1]
    rng = random.Random(config['schedule_seed'])
    repeats = [dict(case_id=c['case_id'], route=route, repeat=repeat)
               for c in cases for repeat in range(config['repeatability_repeats'])
               for route in ('raw', 'native_chat')]
    rng.shuffle(repeats)
    positions = []
    for fixture in fixtures:
        for order_index, order in enumerate(ORDERS):
            for repeat in range(config['position_repeats']):
                arms = ['rebuild', 'retain']
                rng.shuffle(arms)
                positions.append(dict(fixture_id=fixture['pair_id'], order_index=order_index,
                    positions=list(order), repeat=repeat, arms=arms,
                    pair_id=f'{fixture["pair_id"]}-order{order_index+1}-repeat{repeat+1}'))
    rng.shuffle(positions)
    if config.get('replay_fixture'):
        repeats = []
        positions = [j for j in positions if j['fixture_id'] == config['replay_fixture']
                     and j['order_index'] == config['replay_order']-1]
        if not positions:
            raise ValueError('Replay fixture/order is not among the selected diagnostic cases')
    return dict(repeatability=repeats, position=positions,
                expected_responses=len(repeats)+len(positions)*6)


def prepare_source(directory, config, inventory):
    source = Path(config['source_run'])
    checksums = json.loads((source / 'checksums.json').read_text())
    for relative, expected in checksums.items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or digest_file(path) != expected:
            raise ValueError(f'Source artifact checksum mismatch: {relative}')
    names = ('config.json', 'manifest.json', 'hat-environment.json', 'fixtures.json',
             'hat.jsonl', 'outcome.json')
    for name in names:
        if name not in checksums:
            raise ValueError(f'Source checksum inventory is missing {name}')
    frozen = directory / 'source-run'
    frozen.mkdir()
    for name in (*names, 'checksums.json'):
        shutil.copy2(source / name, frozen / name)
    old_manifest = json.loads((frozen / 'manifest.json').read_text())
    if inventory['model_sha256'] != old_manifest['model_sha256']:
        raise ValueError('Diagnostic requires the original model bytes')
    if inventory['hailort_cli'] != old_manifest['hailort_cli']:
        raise ValueError('Diagnostic requires the original HailoRT version')
    rows, errors = read_jsonl(frozen / 'hat.jsonl')
    if errors:
        raise ValueError('Source HAT observations have damaged JSONL records')
    environment = json.loads((frozen / 'hat-environment.json').read_text())
    inputs = select_inputs(json.loads((frozen / 'fixtures.json').read_text()), rows, environment)
    atomic_json(directory / 'diagnostic-inputs.json', inputs)
    atomic_json(directory / 'schedule.json', make_schedule(inputs, config))
    atomic_json(directory / 'source-provenance.json', dict(source_run=str(source),
        source_checksums_sha256=digest_file(source / 'checksums.json'),
        verified_source_artifacts=len(checksums),
        frozen_sha256={name: digest_file(frozen / name) for name in (*names, 'checksums.json')}))


def reset_context(llm):
    llm.clear_context()
    if llm.get_context_usage_size() != 0:
        raise RuntimeError('Context reset did not establish an empty conversation')


def check_budget(llm, full_ids, limit):
    if len(full_ids) + 16 + len(llm.tokenize(llm.get_generation_recovery_sequence())) > limit:
        raise RuntimeError('Conversation would exceed its reserved context budget')


def record_response(llm, directory, submitted, full, arm, route, stops, metadata, before,
                    full_ids, limit):
    result = generate(llm, submitted, arm, stops, capture_chunks=True)
    after = llm.get_context_usage_size()
    expected_after = len(llm.tokenize(full + result['output']))
    result.update(metadata, route=route, arm=arm, effective_prompt=full,
        submitted_prompt=submitted, effective_input_sha256=digest_value(full_ids),
        effective_input_tokens=len(full_ids), context_before=before,
        context_after=after, expected_context_after=expected_after,
        token_ledger_valid=after == expected_after,
        input_identity_basis='recorded_raw_tokens' if route == 'raw' else 'local_template_only',
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    result.update(score_answer(result['output'], result['expected_answer'], result['queried_item'],
                               stops, result['completion_status']))
    emit(directory / 'diagnostic.jsonl', 'measurement', **result)
    # Native serialization is opaque. A mismatch is evidence to retain, not a
    # claim that local token IDs describe the SDK's hidden effective input.
    if after > limit:
        raise RuntimeError('Observed runtime context exceeds experiment budget')
    return result


def repeat_request(llm, case, job, directory, limit, stops):
    reset_context(llm)
    full = case['effective_prompt']
    ids = llm.tokenize(full)
    check_budget(llm, ids, limit)
    submitted = full if job['route'] == 'raw' else deepcopy(case['messages'])
    metadata = {k: v for k, v in case.items() if k not in
                ('messages', 'effective_prompt', 'effective_input_sha256')}
    metadata.update(study='repeatability', repeat=job['repeat'], case_id=job['case_id'])
    result = record_response(llm, directory, submitted, full, 'rebuild', job['route'], stops,
                             metadata, 0, ids, limit)
    if not result['token_ledger_valid']:
        reset_context(llm)
        emit(directory / 'diagnostic.jsonl', 'request_quarantined', case_id=job['case_id'],
             route=job['route'], repeat=job['repeat'], reason='runtime_context_count_mismatch',
             reset_verified=True)
    return result


def position_fixture(fixture, positions):
    indices = dict(beginning=0, middle=len(fixture['table'])//2, end=len(fixture['table'])-1)
    items = [fixture['table'][indices[p]] for p in positions]
    questions = [f'What color is {key}?' for key, _ in items]
    messages = deepcopy(fixture['initial_messages'])
    original_question = fixture['questions'][0]
    if not messages[-1]['content'].endswith(original_question):
        raise ValueError('Original initial question is not an exact terminal suffix')
    messages[-1]['content'] = messages[-1]['content'][:-len(original_question)] + questions[0]
    return dict(messages=messages, questions=questions, items=items)


def position_session(llm, render, fixture, job, arm, directory, limit, stops):
    reset_context(llm)
    prepared = position_fixture(fixture, job['positions'])
    messages, accumulated, results = prepared['messages'], '', []
    for turn, (item, expected) in enumerate(prepared['items']):
        progress(directory, 'hat', 'position_generation', pair_id=job['pair_id'], arm=arm, turn=turn)
        if turn:
            messages.append(dict(role='user', content=prepared['questions'][turn]))
        full = render(messages)
        suffix, old_ids, new_ids, ids = exact_suffix(full, accumulated, llm.tokenize)
        check_budget(llm, ids, limit)
        before = llm.get_context_usage_size()
        if before != len(old_ids):
            raise RuntimeError(f'Context ledger mismatch before turn: {before} != {len(old_ids)}')
        metadata = dict(study='position', pair_id=job['pair_id'], fixture_id=fixture['pair_id'],
            fixture_selection=fixture['selection'], fixture_seed=fixture['seed'],
            target_tokens=fixture['target_tokens'], repeat=job['repeat'], order_index=job['order_index'],
            question_order=job['positions'], position=job['positions'][turn], turn=turn,
            queried_item=item, expected_answer=expected, messages=deepcopy(messages),
            prefix_suffix_tokens_equal=True, submitted_input_tokens=len(ids) if arm == 'rebuild' else len(new_ids))
        result = record_response(llm, directory, full if arm == 'rebuild' else suffix, full,
                                 arm, 'raw', stops, metadata, before, ids, limit)
        results.append(result)
        accumulated = full + result['output']
        if not result['token_ledger_valid']:
            # A decoded string need not re-encode to the actual generated token
            # sequence. Preserve this evidence and abandon only this context.
            # Never infer hidden token IDs from matching text or keep using an
            # unverified cache. Independent arms/jobs start with a verified reset.
            reset_context(llm)
            emit(directory / 'diagnostic.jsonl', 'session_quarantined', pair_id=job['pair_id'],
                 arm=arm, turn=turn, completed_turns=len(results), skipped_turns=2-turn,
                 reason='runtime_context_count_mismatch', reset_verified=True)
            break
        if result['completion_status'] not in (LOGICAL_END, 'MAX_TOKENS_REACHED'):
            emit(directory / 'diagnostic.jsonl', 'session_incomplete', pair_id=job['pair_id'],
                 arm=arm, completed_turns=len(results), reason=result['completion_status'])
            break
        messages.append(dict(role='assistant', content=assistant_content(result['output'], stops)))
    return results


def compare_position_pair(by_arm, pair_id):
    indexed = {a: {r['turn']: r for r in rows} for a, rows in by_arm.items()}
    comparisons, first = [], None
    for turn in range(3):
        a, b = (indexed.get(arm, {}).get(turn) for arm in ('rebuild', 'retain'))
        inputs_equal = bool(a and b and a['effective_input_sha256'] == b['effective_input_sha256'])
        outputs_equal = bool(a and b and a['output'] == b['output'])
        reason = ('missing_arm' if not a or not b else 'different_recorded_history' if not inputs_equal
                  else 'invalid_ledger' if not (a['token_ledger_valid'] and b['token_ledger_valid'])
                  else 'incomplete_generation' if any(r['completion_status'] != LOGICAL_END for r in (a, b))
                  else None)
        if a and b and not outputs_equal and first is None:
            first = turn
        comparisons.append(dict(pair_id=pair_id, turn=turn, inputs_equal=inputs_equal,
            outputs_equal=outputs_equal, eligible=reason is None, exclusion_reason=reason,
            first_divergence=first == turn,
            factual_disagreement=bool(a and b and a['parsed_color'] != b['parsed_color']),
            rebuild_factual_correct=a['factual_correct'] if a else None,
            retain_factual_correct=b['factual_correct'] if b else None,
            first_visible_speedup=(a['first_visible_ms']/b['first_visible_ms']
                if reason is None and a['first_visible_ms'] and b['first_visible_ms'] else None)))
        record = a or b or {}
        comparisons[-1].update({key:record.get(key) for key in
            ('fixture_id','target_tokens','order_index','repeat','position')})
    return comparisons


def run(directory, config):
    from hailo_platform import VDevice, __version__
    from hailo_platform.pyhailort.pyhailort import LLM

    inputs = json.loads((directory / 'diagnostic-inputs.json').read_text())
    schedule = json.loads((directory / 'schedule.json').read_text())
    source_env = json.loads((directory / 'source-run/hat-environment.json').read_text())
    cases = {c['case_id']: c for c in inputs['cases']}
    fixtures = {f['pair_id']: f for f in inputs['fixtures']}
    progress(directory, 'hat', 'model_loading')
    loading = time.perf_counter()
    with VDevice() as device, LLM(device, config['model']) as llm:
        capacity = llm.max_context_capacity()
        limit = min(1792, capacity-256)
        stops = llm.get_stop_tokens()
        template = llm.prompt_template()
        render = renderer(template)
        if (__version__ != source_env['hailort_version'] or template != source_env['prompt_template']
                or stops != source_env['stop_tokens']
                or llm.get_generation_recovery_sequence() != source_env['generation_recovery_sequence']):
            raise RuntimeError('Runtime, template, stop tokens or recovery sequence differs from the source run')
        for case in cases.values():
            if render(case['messages']) != case['effective_prompt']:
                raise RuntimeError('Frozen messages do not reconstruct the original raw prompt')
            if digest_value(llm.tokenize(case['effective_prompt'])) != case['effective_input_sha256']:
                raise RuntimeError('Frozen prompt token identity changed since the source run')
        atomic_json(directory / 'hat-environment.json', dict(hailort_version=__version__,
            model=config['model'], loading_seconds=time.perf_counter()-loading,
            capacity_tokens=capacity, experiment_limit_tokens=limit, prompt_template=template,
            stop_tokens=stops, generation_recovery_sequence=llm.get_generation_recovery_sequence(),
            decoding=dict(do_sample=False, seed=12345, max_generated_tokens=16),
            reconstructed_prompts_verified=len(cases), native_serialization='SDK internal; local template is an expectation, not observed hidden input'))
        warm = [dict(role='system', content=SYSTEM),
                dict(role='user', content='Reference facts: item001 = blue. What color is item001?')]
        for route in ('raw', 'native_chat'):
            progress(directory, 'hat', 'warmup', route=route)
            reset_context(llm)
            generate(llm, render(warm) if route == 'raw' else warm, 'rebuild', stops)
        for number, job in enumerate(schedule['repeatability'], 1):
            wait_cool(directory)
            progress(directory, 'hat', 'repeatability_generation', **job)
            repeat_request(llm, cases[job['case_id']], job, directory, limit, stops)
            print(f'Repeatability {number}/{len(schedule["repeatability"])} {job["case_id"]} {job["route"]}', flush=True)
        for number, job in enumerate(schedule['position'], 1):
            by_arm = {}
            emit(directory / 'diagnostic.jsonl', 'pair_start', **job)
            for arm in job['arms']:
                wait_cool(directory)
                by_arm[arm] = position_session(llm, render, fixtures[job['fixture_id']], job,
                                              arm, directory, limit, stops)
            comparisons = compare_position_pair(by_arm, job['pair_id'])
            emit(directory / 'diagnostic.jsonl', 'pair_complete', pair_id=job['pair_id'], comparisons=comparisons)
            print(f'Position pair {number}/{len(schedule["position"])} {job["pair_id"]}: '
                  f'{sum(c["inputs_equal"] for c in comparisons)} matched histories', flush=True)
        reset_context(llm)
    emit(directory / 'diagnostic.jsonl', 'complete', expected_responses=schedule['expected_responses'])
