"""Frozen coordination design; no device access or adaptation to holdout results."""
import hashlib
import json
import random

from .feedback_spec import CATALOG

VERSION = 'coordination-v1'
SEED = 20261007
EVENTS = 'coordination.jsonl'
CONDITIONS = ('cpu_serial', 'cpu_overlap', 'gpu_serial', 'gpu_overlap')
ARCHITECTURES = ('thread', 'process')
SYSTEM = ('Choose one eligible untested attention configuration to benchmark. '
          'Reply with its ID only, such as C2. Never repeat a tested ID. '
          'The times are development measurements in milliseconds; lower is better. '
          'Example: eligible=[C2], tested=[C0,C1], answer=C2. '
          'Example: eligible=[C5], tested=[C4], answer=C5. No explanation.')


def seed_for(split, index):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{SEED}|{split}|{index}'.encode()).digest()[:8], 'little')


def advice_cases():
    cases = []
    for split, count in (('development', 20), ('holdout', 20), ('pilot', 1)):
        for i in range(count):
            rng = random.Random(seed_for(split, i))
            order = list(CATALOG); rng.shuffle(order)
            eligible = sorted(order[:1+i % 7])
            tested = sorted(set(CATALOG)-set(eligible))
            history = {c: round(rng.uniform(30, 1700), 2) for c in tested}
            payload = dict(eligible=eligible, tested=tested, development_ms=history,
                          configurations={c:CATALOG[c] for c in eligible})
            cases.append(dict(case_id=f'{split}-{i:02}', split=split, eligible=eligible,
                messages=[dict(role='system', content=SYSTEM),
                          dict(role='user', content=json.dumps(payload, separators=(',', ':')))]))
    return cases


def fixture_specs():
    return [dict(fixture_id=f'{split}-{r}-{i}', split=split, repeat=r, index=i,
                 n=1024, d=64, b=4, seed=seed_for(f'numeric-{split}', r*3+i))
            for split, repeats in (('pilot', 1), ('main', 6))
            for r in range(repeats) for i in range(3)]


def schedule():
    blocks = []
    for stage, repeats in (('pilot', 1), ('main', 6)):
        for r in range(repeats):
            architectures = ARCHITECTURES if r % 2 == 0 else ARCHITECTURES[::-1]
            order = CONDITIONS[r % 4:]+CONDITIONS[:r % 4]
            for arch in architectures:
                blocks.append(dict(block_id=f'{stage}-{r}-{arch}', stage=stage, repeat=r,
                    architecture=arch, conditions=list(order),
                    fixture_ids=[f'{stage}-{r}-{i}' for i in range(3)],
                    case_id='pilot-00' if stage=='pilot' else f'holdout-{r:02}'))
    return blocks


def specification():
    return dict(version=VERSION, seed=SEED, cpu_backend='native4', gpu_backend='C6',
        batches=24, repeats=6, conditions=CONDITIONS, architectures=ARCHITECTURES,
        expected_main_conditions=48, expected_main_numerical_requests=1152,
        development_cases=20, holdout_cases=20, bootstrap_samples=10000,
        primary_comparison=['thread_cpu_overlap', 'process_cpu_overlap'],
        promotion_ratio=1.10, tolerances=dict(atol=1e-5, rtol=1e-4),
        max_seconds=1800, cleanup_reserve_seconds=45, pilot_margin=1.20,
        prompt_revision='One predeclared shortened prompt; no run-time edits or retries.',
        advice_gate='20/20 held-out and 48/48 main outputs must be complete eligible IDs with valid context ledgers.',
        worker_lifetime='Persistent within four-condition architecture block; creation and warmups excluded from job latency.',
        statistical_unit='Six paired repetition blocks, not the 1152 numerical requests.',
        timing='Submission through both validated results delivered; includes transport. GPU timestamps diagnostic only.',
        invalid_advice='Continue timing diagnostics, prohibit application-benefit recommendation.',
        interpretation='Independent attention and adviser jobs, not model layers split across devices; no energy measurement.')


def pilot_required_seconds(blocks):
    """Both architecture blocks include startup, six warmups, four jobs and cleanup."""
    if len(blocks) != 2 or {b['architecture'] for b in blocks} != set(ARCHITECTURES):
        raise ValueError('Pilot requires both complete architectures')
    return sum(b['elapsed_seconds'] for b in blocks)*6*1.20+45
