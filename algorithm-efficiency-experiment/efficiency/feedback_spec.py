"""Frozen search space, splits, proposal parsing and deterministic promotion rules."""
import hashlib
import json
import math
import random
import re
import statistics

VERSION = 'attention-feedback-v1'
SEED = 20261006
CPU_BACKENDS = ('numpy', 'native1', 'native4')
STRATEGIES = ('coordinate', 'hat')
CATALOG = {f'C{i}': dict(tile=16 if i&4 else 8, reduction=128 if i&2 else 64,
    projection='cpu' if i&1 else 'gpu') for i in range(8)}
CELLS = [dict(cell_id=f'{mode}-n{n}-b{b}', mode=mode, n=n, d=64, b=b)
         for mode in ('stream', 'prefill') for n in (128, 512, 1024) for b in (1, 4)]
ADVISOR_SYSTEM = '''Select one untested attention implementation to benchmark next. Return only its two-character ID, C0 through C7.
Optimize measured end-to-end time while preserving numerical accuracy. GPU kernel time alone is not request time.
Streaming releases inputs one step at a time; prefill has the full causal batch. CPU validates all results.
Choose only an eligible untested ID. Examples: C0, C5. No explanation or extra text.'''


def specification():
    return dict(version=VERSION, seed=SEED, cells=CELLS, catalog=CATALOG, cpu_backends=CPU_BACKENDS,
        strategies=STRATEGIES, rounds=3, development_seeds=3, development_repeats=3,
        validation_seeds_per_round=3, validation_repeats=5, test_seeds=8, test_repeats=5,
        tolerances=dict(atol=1e-5, rtol=1e-4), promotion_gain=1.10, final_cell_slowdown_limit=1.05,
        bootstrap_samples=10000, advisor_system=ADVISOR_SYSTEM,
        concurrency_repeats=6, concurrency_batches=24, concurrency_cell='prefill-n1024-b4',
        stage_seconds=dict(correctness_baseline=600, search=1500, final=900, concurrency=300),
        expected_measurements=dict(baseline=324, development=1296, validation=2160, final=1440),
        warmup='One untimed-for-comparison request per distinct backend per measurement block; outputs checked.',
        order='Cycle all arm permutations by seed/repetition/block serial; reverse strategy order in round two.',
        final_mixture='Equal frequency across all twelve cells; streaming and prefill also reported separately.',
        concurrency_order='Rotate cpu_serial,cpu_overlap,gpu_serial,gpu_overlap by repeat modulo four.',
        policy='Opt-in only; unknown shapes use native1; final tests never change frozen routes.',
        correctness_domain='Standard-normal inputs and scaled-normal weights. Semantic perturbations change signs, preserving input magnitude/distribution; arbitrary large-magnitude inputs are outside the accuracy claim.',
        gpu_buffer_limit_bytes=512*1024**2)


def fixture_seed(split, cell_id, index):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{SEED}|{split}|{cell_id}|{index}'.encode()).digest()[:8], 'little')


def input_hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def split_manifest():
    rows = []
    for split, count in [('development', 3), ('validation-1', 3), ('validation-2', 3), ('validation-3', 3), ('test', 8)]:
        for cell in CELLS:
            for index in range(count):
                rows.append(dict(**cell, split=split, seed_index=index,
                    seed=fixture_seed(split, cell['cell_id'], index), fixture_id=f'{split}-{cell["cell_id"]}-s{index}'))
    return rows


def deterministic_choice(history, eligible):
    eligible = set(eligible)
    tested = {h['candidate'] for h in history}
    remaining = eligible-tested
    if not remaining: return None
    if not history: return min(remaining)
    center = int(min(history, key=lambda h: (h['development_total_ms'], h['candidate']))['candidate'][1:])
    for bit in (4, 2, 1):
        candidate = f'C{center^bit}'
        if candidate in remaining: return candidate
    return min(remaining)


def advisor_messages(history, routes, eligible):
    # Explicit field allowlist excludes validation/test metrics and all oracle arrays.
    feedback = [dict(candidate=h['candidate'], development_total_ms=round(h['development_total_ms'], 3),
                     cells=h['development_cells']) for h in history]
    payload = dict(catalog=CATALOG, eligible_untested=sorted(eligible), current_routes=routes,
                   measured_development_history=feedback)
    return [dict(role='system', content=ADVISOR_SYSTEM), dict(role='user', content=json.dumps(payload, separators=(',', ':')))]


def parse_proposal(output, stops, completion_status, ledger_valid, eligible):
    if completion_status!='LOGICAL_END_OF_GENERATION' or not ledger_valid: return None, 'incomplete_or_invalid_context'
    ends = [s for s in sorted(stops, key=len, reverse=True) if output.endswith(s)]
    if not ends: return None, 'missing_terminal'
    body = output[:-len(ends[0])].strip()
    if not re.fullmatch(r'C[0-7]', body): return None, 'malformed_proposal'
    if body not in eligible: return None, 'ineligible_or_repeated_proposal'
    return body, None


def paired_interval(pairs, seed, samples=10000):
    """Pairs are independent seed summaries, with both arms retained together."""
    if len(pairs)<2 or any(a<=0 or b<=0 or not math.isfinite(a+b) for a, b in pairs): return None
    ratio = lambda ps: sum(a for a, _ in ps)/sum(b for _, b in ps)
    rng = random.Random(seed); values = sorted(ratio(rng.choices(pairs, k=len(pairs))) for _ in range(samples))
    return dict(ratio=ratio(pairs), ci95=[values[int(.025*samples)], values[int(.975*samples)]],
                independent_seeds=len(pairs), samples=samples, seed=seed)


def promotion(pairs, correct=True, seed=SEED):
    interval = paired_interval(pairs, seed)
    accepted = bool(correct and interval and interval['ratio']>=1.10 and interval['ci95'][0]>1)
    return dict(promote=accepted, inference=interval, correct=bool(correct))


def resolve_route(policy, cell_id, cpu_default='native1'):
    if cell_id not in {c['cell_id'] for c in CELLS}: return cpu_default
    route = policy.get('routes', {}).get(cell_id, cpu_default)
    if route not in (*CPU_BACKENDS, *CATALOG): raise ValueError('Policy contains an unregistered backend')
    return route


def median_by_seed(rows, arm):
    seeds = sorted({r['seed_index'] for r in rows})
    return {s: statistics.median(r['request_ms'] for r in rows if r['seed_index']==s and r['arm']==arm) for s in seeds}
