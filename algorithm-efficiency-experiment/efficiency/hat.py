"""Paired Hailo experiments, with exact raw-text and token-history checks."""
import json
import random
import re
import resource
import time

from jinja2 import Environment

from .common import atomic_json, digest_value, emit, progress, wait_cool

SYSTEM = 'Answer using the reference facts. Keep answers to one word.'
COLORS = ('blue', 'green', 'red', 'yellow', 'white', 'black', 'pink', 'brown')


def visible_text(text, stop_tokens):
    for token in stop_tokens:
        text = text.replace(token, '')
    return text


def normalize_answer(text, stop_tokens):
    return visible_text(text, stop_tokens).strip().strip('.,!?;:').strip().casefold()


def mentioned_colors(text, stop_tokens):
    """Supplementary description only; never relax the primary answer rubric."""
    words = re.findall(r'\b[a-z]+\b', visible_text(text, stop_tokens).casefold())
    return sorted(set(words).intersection(COLORS))


def assistant_content(text, stop_tokens):
    for token in sorted(stop_tokens, key=len, reverse=True):
        if text.endswith(token):
            return text[:-len(token)]
    raise ValueError('Completion lacks a supported terminal token')


def exact_suffix(full_prompt, accumulated_text, tokenize):
    """Prove both raw prefix identity and tokenizer compositionality."""
    if not full_prompt.startswith(accumulated_text):
        raise ValueError('Chat template does not extend the exact generated transcript')
    suffix = full_prompt[len(accumulated_text):]
    old_ids = tokenize(accumulated_text) if accumulated_text else []
    new_ids = tokenize(suffix)
    full_ids = tokenize(full_prompt)
    if old_ids + new_ids != full_ids:
        raise ValueError('Concatenated prompt token IDs differ from full replay')
    return suffix, old_ids, new_ids, full_ids


def build_fixture(render, tokenize, target, seed, turns=4):
    rng = random.Random(seed)
    rows = [(f'item{i:03d}', rng.choice(COLORS)) for i in range(1, 301)]

    def first_messages(count):
        table = '; '.join(f'{key} = {value}' for key, value in rows[:count])
        return [dict(role='system', content=SYSTEM), dict(role='user', content=
            f'Reference facts: {table}. What color is {rows[0][0]}?')]

    low, high, best = 4, len(rows), None
    while low <= high:
        count = (low + high) // 2
        prompt = render(first_messages(count))
        size = len(tokenize(prompt))
        if size <= target:
            best = count
            low = count + 1
        else:
            high = count - 1
    if best is None:
        raise ValueError(f'Target {target} is too small for four reference rows')
    indices = [0, best // 2, best-1, 0][:turns]
    return dict(seed=seed, target_tokens=target, table=rows[:best],
                initial_messages=first_messages(best),
                initial_prompt_tokens=len(tokenize(render(first_messages(best)))),
                questions=[f'What color is {rows[index][0]}?' for index in indices],
                answers=[rows[index][1] for index in indices])


def generate(llm, prompt, arm, stop_tokens, max_tokens=16, capture_chunks=False):
    chunks = []
    first_ns = None
    cpu_start = time.process_time()
    started = time.perf_counter_ns()
    clear_ns = 0
    if arm == 'rebuild':
        clear_start = time.perf_counter_ns()
        llm.clear_context()
        clear_ns = time.perf_counter_ns() - clear_start
    with llm.generate(prompt, do_sample=False, seed=12345, max_generated_tokens=max_tokens) as gen:
        while str(gen.generation_status).endswith('.GENERATING'):
            chunk = gen.read(timeout_ms=90000)
            chunks.append(chunk)
            if first_ns is None and visible_text(''.join(chunks), stop_tokens).strip():
                first_ns = time.perf_counter_ns()
        status = str(gen.generation_status).split('.')[-1]
    end_ns = time.perf_counter_ns()
    result = dict(output=''.join(chunks), completion_status=status, stream_events=len(chunks),
                request_ms=(end_ns-started)/1e6,
                first_visible_ms=(first_ns-started)/1e6 if first_ns is not None else None,
                clear_context_ms=clear_ns/1e6, cpu_seconds=time.process_time()-cpu_start)
    if capture_chunks:
        result['stream_chunks'] = chunks
    return result


def run_session(llm, render, fixture, arm, pair_id, directory, context_limit, stop_tokens, turns):
    session_started = time.perf_counter()
    llm.clear_context()
    if llm.get_context_usage_size() != 0:
        raise RuntimeError('Context reset did not establish an empty conversation')
    messages = list(fixture['initial_messages'])
    accumulated = ''
    results = []
    for turn in range(turns):
        progress(directory, 'hat', 'generation', pair_id=pair_id, arm=arm, turn=turn)
        if turn:
            messages.append(dict(role='user', content=fixture['questions'][turn]))
        prep_start = time.perf_counter_ns()
        full = render(messages)
        suffix, old_ids, new_ids, full_ids = exact_suffix(full, accumulated, llm.tokenize)
        # Reserve output and recovery space before requesting any generation.
        if len(full_ids) + 16 + len(llm.tokenize(llm.get_generation_recovery_sequence())) > context_limit:
            raise RuntimeError('Conversation would exceed its reserved context budget')
        before = llm.get_context_usage_size()
        if before != len(old_ids):
            raise RuntimeError(f'Context ledger mismatch before turn: {before} != {len(old_ids)}')
        preparation_ms = (time.perf_counter_ns()-prep_start)/1e6
        result = generate(llm, full if arm == 'rebuild' else suffix, arm, stop_tokens)
        accumulated = full + result['output']
        after = llm.get_context_usage_size()
        expected_after = len(llm.tokenize(accumulated))
        ledger_valid = after == expected_after
        result.update(pair_id=pair_id, arm=arm, turn=turn, target_tokens=fixture['target_tokens'],
                      fixture_seed=fixture['seed'], initial_prompt_tokens=fixture['initial_prompt_tokens'],
                      effective_input_sha256=digest_value(full_ids), effective_input_tokens=len(full_ids),
                      submitted_prompt=full if arm == 'rebuild' else suffix,
                      effective_prompt=full, submitted_input_tokens=len(full_ids) if arm == 'rebuild' else len(new_ids),
                      context_before=before, context_after=after, expected_context_after=expected_after,
                      token_ledger_valid=ledger_valid,
                      context_completion_tokens=after-len(full_ids),
                      expected_answer=fixture['answers'][turn],
                      normalized_answer=normalize_answer(result['output'], stop_tokens),
                      mentioned_colors=mentioned_colors(result['output'], stop_tokens),
                      correct=normalize_answer(result['output'], stop_tokens) == fixture['answers'][turn],
                      preparation_and_validation_ms=preparation_ms,
                      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        emit(directory / 'hat.jsonl', 'measurement', **result)
        results.append(result)
        if not ledger_valid:
            raise RuntimeError(f'Context ledger mismatch after turn: {after} != {expected_after}')
        if result['completion_status'] == 'MAX_TOKENS_REACHED':
            # The validated recovery terminator permits a subsequent turn. This
            # truncated observation remains excluded from completed-response ratios.
            emit(directory / 'hat.jsonl', 'turn_truncated', pair_id=pair_id, arm=arm, turn=turn)
        elif result['completion_status'] != 'LOGICAL_END_OF_GENERATION':
            emit(directory / 'hat.jsonl', 'session_incomplete', pair_id=pair_id, arm=arm,
                 reason=result['completion_status'], completed_turns=len(results))
            break
        messages.append(dict(role='assistant', content=assistant_content(result['output'], stop_tokens)))
    emit(directory / 'hat.jsonl', 'session_complete', pair_id=pair_id, arm=arm,
         completed_turns=len(results), expected_turns=turns,
         session_wall_ms=(time.perf_counter()-session_started)*1000,
         request_total_ms=sum(r['request_ms'] for r in results),
         diagnostic_preparation_total_ms=sum(r['preparation_and_validation_ms'] for r in results))
    return results


def run(directory, config):
    from hailo_platform import VDevice, __version__
    from hailo_platform.pyhailort.pyhailort import LLM

    progress(directory, 'hat', 'model_loading')
    loading = time.perf_counter()
    with VDevice() as device, LLM(device, config['model']) as llm:
        loading_seconds = time.perf_counter()-loading
        stop_tokens = llm.get_stop_tokens()
        template = llm.prompt_template()
        capacity = llm.max_context_capacity()
        limit = min(1792, capacity-256)
        if limit < max(config['hat_sizes']) + 128:
            raise RuntimeError('Compiled model capacity is insufficient for selected fixtures')
        renderer = Environment().from_string(template)

        def render(messages):
            return renderer.render(messages=messages, tools=None, add_generation_prompt=True)

        atomic_json(directory / 'hat-environment.json', dict(hailort_version=__version__,
            model=config['model'], loading_seconds=loading_seconds, capacity_tokens=capacity,
            experiment_limit_tokens=limit, prompt_template=template, stop_tokens=stop_tokens,
            generation_recovery_sequence=llm.get_generation_recovery_sequence(),
            decoding=dict(do_sample=False, seed=12345, max_generated_tokens=16),
            timing_definition='Prompt preparation and diagnostic RPCs precede timing; context clearing, generate/read and generator cleanup are timed.'))
        # Warm both paths with a disposable conversation, never reuse its context.
        progress(directory, 'hat', 'warmup')
        warm = render([dict(role='system', content=SYSTEM),
                       dict(role='user', content='Reference table: item001: blue. What color is item001?')])
        for arm in ('rebuild', 'retain'):
            llm.clear_context()
            generate(llm, warm, arm, stop_tokens)
        llm.clear_context()
        rng = random.Random(20261002)
        jobs = [(size, repetition) for size in config['hat_sizes'] for repetition in range(config['hat_pairs'])]
        rng.shuffle(jobs)
        fixtures = []
        for size, repetition in jobs:
            wait_cool(directory)
            pair_id = f'n{size}-pair{repetition+1:02d}'
            progress(directory, 'hat', 'fixture', pair_id=pair_id)
            fixture_seed_base = 20261002 if config['profile'] == 'pilot' else 17
            fixture = build_fixture(render, llm.tokenize, size, fixture_seed_base+size*100+repetition, config['turns'])
            fixture['pair_id'] = pair_id
            fixtures.append(fixture)
            atomic_json(directory / 'fixtures.json', fixtures)
            order = ['rebuild', 'retain']
            rng.shuffle(order)
            emit(directory / 'hat.jsonl', 'pair_start', pair_id=pair_id,
                 target_tokens=size, order=order, fixture_seed=fixture['seed'])
            by_arm = {}
            for arm in order:
                wait_cool(directory)
                by_arm[arm] = run_session(llm, render, fixture, arm, pair_id, directory,
                                          limit, stop_tokens, config['turns'])
            comparisons = []
            for rebuilt, retained in zip(by_arm['rebuild'], by_arm['retain']):
                comparisons.append(dict(turn=rebuilt['turn'],
                    inputs_equal=rebuilt['effective_input_sha256'] == retained['effective_input_sha256'],
                    outputs_equal=rebuilt['output'] == retained['output'],
                    both_correct=rebuilt['correct'] and retained['correct']))
            emit(directory / 'hat.jsonl', 'pair_complete', pair_id=pair_id,
                 target_tokens=size, comparisons=comparisons)
            print(f'HAT {pair_id}: {len(comparisons)} turns; '+
                  f'{sum(c["both_correct"] for c in comparisons)} both correct; '+
                  f'{sum(c["inputs_equal"] for c in comparisons)} matched histories', flush=True)
            llm.clear_context()
        if llm.get_context_usage_size() != 0:
            raise RuntimeError('Final context reset failed')
    emit(directory / 'hat.jsonl', 'complete')
