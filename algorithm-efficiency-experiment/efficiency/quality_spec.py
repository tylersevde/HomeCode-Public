"""Frozen NPU quality study: fixtures, conditions and decisions without hardware."""
import hashlib
import math
import random
import statistics
from collections import Counter

from .common import digest_value
from .hat import COLORS, SYSTEM
from .research_spec import balanced_order

VERSION = 'npu-quality-v1'
MODEL_SHA = '5310176848638505fbc28add04ba60c97abe345cdb0ec7e3b8ffaa4b0a8c65dd'
SIZES = (128, 512, 1024)
PROMPTS = dict(legacy=SYSTEM, explicit=(
    'Use only the reference facts. Reply with exactly one lowercase color: blue, '
    'green, red, yellow, white, black, pink, or brown. Output no other words.'))
CONDITIONS = [dict(id=f'{p}-{n}', prompt_id=p, prompt=PROMPTS[p], max_generated_tokens=n)
              for p in PROMPTS for n in (16, 32, 64)]
EVENTS = 'quality.jsonl'


def specification(stage):
    if stage not in ('develop', 'confirm'):
        raise ValueError('Unknown quality stage')
    return dict(version=VERSION, stage=stage, model_sha256=MODEL_SHA,
                stage_limit_seconds=7200, campaign_limit_seconds=14400,
                cleanup_reserve_seconds=120, pilot_margin=1.2,
                blocks=8 if stage == 'develop' else 16, sizes=list(SIZES), turns=4,
                conditions=CONDITIONS, prompts=PROMPTS, context_policy='rebuild',
                min_rows=4, max_rows=300, bootstrap_seed=20261003,
                bootstrap_samples=10000, confidence=.95, improvement_points=.05,
                development_min_correct=92, development_min_per_size=29,
                confirmation_min_correct=183, confirmation_min_per_size=58,
                selection='correct descending, complete dialogue mean ascending, tokens ascending, legacy first',
                score='Existing diagnostic strict scorer; valid completed answers only; all planned turns in denominator')


def conditions(stage, selected=None):
    if stage == 'develop':
        return [dict(c, label=c['id']) for c in CONDITIONS]
    if stage != 'confirm' or selected not in [c['id'] for c in CONDITIONS]:
        raise ValueError('Confirmation requires a frozen development selection')
    chosen = next(c for c in CONDITIONS if c['id'] == selected)
    return [dict(CONDITIONS[0], label='control'), dict(chosen, label='candidate')]


def seed_for(namespace, stage, fid):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{namespace}|{stage}|{fid}'.encode()).digest()[:8], 'little')


def fixture_specs(stage, namespace):
    count = 8 if stage == 'develop' else 16
    return [dict(fixture_id=f'{section}-n{size}-b{block}', section=section, block=block,
                 target_tokens=size, first_color=(('brown', 'green', 'blue')[j] if section == 'pilot'
                                                 else COLORS[block % 8]),
                 seed=seed_for(namespace, stage, f'{section}-n{size}-b{block}'))
            for section, blocks in (('pilot', [-1]), ('main', range(count)))
            for block in blocks for j, size in enumerate(SIZES)]


def messages(table, prompt_id):
    facts = '; '.join(f'{key} = {value}' for key, value in table)
    return [dict(role='system', content=PROMPTS[prompt_id]),
            dict(role='user', content=f'Reference facts: {facts}. What color is {table[0][0]}?')]


def build_fixture(render, tokenize, spec):
    rng = random.Random(spec['seed'])
    rows = [[f'item{i:03d}', rng.choice(COLORS)] for i in range(1, 301)]
    rows[0][1] = spec['first_color']
    cache = {}

    def encoded(count):
        if count not in cache:
            cache[count] = {p: list(tokenize(render(messages(rows[:count], p)))) for p in PROMPTS}
        return cache[count]

    low, high, best = 4, 300, None
    while low <= high:
        n = (low + high) // 2
        if max(map(len, encoded(n).values())) <= spec['target_tokens']:
            best, low = n, n + 1
        else:
            high = n - 1
    if best is None:
        raise ValueError('Target cannot fit four rows with both prompt forms')
    table = rows[:best]
    indices = (0, best // 2, best - 1, 0)
    return dict(spec, table=table, table_sha256=digest_value(table),
                initial_token_ids=encoded(best), next_row_token_ids=encoded(best+1) if best < 300 else None,
                questions=[f'What color is {table[i][0]}?' for i in indices],
                answers=[table[i][1] for i in indices])


def schedule(fixtures, arms):
    return [dict(fixture_id=f['fixture_id'], section=f['section'], block=f['block'],
                 target_tokens=f['target_tokens'], condition=condition)
            for i, f in enumerate(fixtures)
            for condition in balanced_order(arms, i if f['section'] == 'pilot' else i-3)]


def percentile(values, fraction):
    return sorted(values)[min(len(values)-1, math.ceil(len(values)*fraction)-1)] if values else None


def aggregate(events, stage, arms):
    blocks = 8 if stage == 'develop' else 16
    result = []
    for arm in arms:
        jobs = [r for r in events if r['event'] == 'dialogue' and r['section'] == 'main'
                and r['condition']['label'] == arm['label']]
        counts, per_size = Counter(), {}
        latencies, request_times = [], []
        for size in SIZES:
            selected = [r for r in jobs if r['target_tokens'] == size]
            rows = [row for job in selected for row in job['response']['result']['rows']]
            valid = sum(bool(r['valid']) for r in rows)
            correct = sum(bool(r['valid'] and r['answer_correct']) for r in rows)
            per_size[str(size)] = dict(planned=blocks*4, observed=len(rows), valid=valid, correct=correct)
        for job in jobs:
            response = job['response']['result']
            counts['contract_failures'] += len(response['contract_failures']) + (not response['reset_verified'])
            complete = len(response['rows']) == 4 and all(r['valid'] for r in response['rows'])
            if complete:
                latencies.append(job['total_ms'])
            for row in response['rows']:
                counts[row['score']['answer_category']] += 1
                counts['format_errors'] += bool(row['valid'] and not row['score']['format_correct'])
                request_times.append(row['request_ms'])
        planned = blocks * 12
        valid = sum(v['valid'] for v in per_size.values())
        correct = sum(v['correct'] for v in per_size.values())
        floor, size_floor = (92, 29) if stage == 'develop' else (183, 58)
        eligible = (len(jobs) == blocks*3 and valid == planned and correct >= floor
                    and all(v['correct'] >= size_floor for v in per_size.values())
                    and counts['contract_failures'] == 0)
        result.append(dict(condition=arm, planned=planned, observed=sum(v['observed'] for v in per_size.values()),
                           valid=valid, correct=correct, accuracy=correct/planned,
                           complete_dialogues=len(latencies), expected_dialogues=blocks*3,
                           per_size=per_size, categories=dict(counts), qualified=eligible,
                           mean_dialogue_ms=statistics.mean(latencies) if latencies else None,
                           p50_dialogue_ms=percentile(latencies,.5), p95_dialogue_ms=percentile(latencies,.95),
                           p50_request_ms=percentile(request_times,.5), p95_request_ms=percentile(request_times,.95)))
    return result


def bootstrap_difference(events):
    differences = []
    for block in range(16):
        total = {}
        for label in ('candidate', 'control'):
            jobs = [r for r in events if r['event']=='dialogue' and r['section']=='main'
                    and r['block']==block and r['condition']['label']==label]
            if len(jobs)!=3:
                return None
            total[label] = sum(bool(x['valid'] and x['answer_correct'])
                               for j in jobs for x in j['response']['result']['rows'])/12
        differences.append(total['candidate']-total['control'])
    rng = random.Random(20261003)
    values = sorted(statistics.mean(rng.choices(differences, k=16)) for _ in range(10000))
    return dict(difference=statistics.mean(differences), interval=[values[250], values[9750]],
                confidence=.95, blocks=16, samples=10000, seed=20261003)


def decide(events, stage, arms):
    metrics = aggregate(events, stage, arms)
    if stage == 'develop':
        eligible = sorted((r for r in metrics if r['qualified']), key=lambda r:(
            -r['correct'], r['mean_dialogue_ms'], r['condition']['max_generated_tokens'],
            r['condition']['prompt_id'] != 'legacy'))
        return dict(metrics=metrics, selected=eligible[0]['condition']['id'] if eligible else None,
                    decision='candidate_selected' if eligible else 'no_qualified_condition')
    candidate = next(r for r in metrics if r['condition']['label']=='candidate')
    changed = candidate['condition']['id'] != 'legacy-16'
    interval = bootstrap_difference(events)
    promoted = bool(changed and candidate['qualified'] and interval
                    and interval['difference'] >= .05 and interval['interval'][0] > 0)
    qualified_control = candidate['qualified'] and not changed
    return dict(metrics=metrics, selected=candidate['condition']['id'], changed=changed,
                interval=interval, accepted=promoted, qualified_unchanged_control=qualified_control,
                decision='quality_improved' if promoted else ('unchanged_control_qualified' if qualified_control
                                                              else 'confirmation_failed'))
