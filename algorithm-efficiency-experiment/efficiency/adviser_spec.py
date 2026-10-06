"""Frozen adviser experiment: old cases are development; holdouts are created later."""
import hashlib
import itertools
import json
import random

from .common import digest_value
from .coordination_spec import SYSTEM as BASELINE_SYSTEM
from .feedback_spec import CATALOG

VERSION = 'adviser-validation-v1'
SEED = 20261008
EVENTS = 'adviser.jsonl'
ARMS = ('baseline', 'examples', 'bounded')
SYSTEM = ('Select one untested attention configuration to benchmark next. '
          'Choose only an ID in eligible_options. tested_history_ms contains previously tested IDs; '
          'never select those. Lower measured development time is better. '
          'Return exactly the chosen two-character ID. No punctuation or explanation.')


def user_text(payload):
    return json.dumps(dict(eligible_options=payload['configurations'],
                          tested_history_ms=payload['development_ms']), separators=(',', ':'))


def example(eligible, best):
    return dict(eligible=eligible, tested=sorted(set(CATALOG)-set(eligible)),
        configurations={c:CATALOG[c] for c in eligible},
        development_ms={c:50 if c==best else 100+int(c[1:]) for c in sorted(set(CATALOG)-set(eligible))})


def messages(case, arm):
    if arm=='baseline':
        return case['baseline_messages']
    if arm not in ('examples', 'bounded'):
        raise ValueError('Unknown adviser arm')
    return [dict(role='system',content=SYSTEM),
        dict(role='user',content=user_text(example(['C1','C5'],'C4'))),
        dict(role='assistant',content='C5'),
        dict(role='user',content=user_text(example(['C0','C2'],'C3'))),
        dict(role='assistant',content='C2'),
        dict(role='user',content=user_text(case['payload']))]


def development_cases(previous):
    cases=[]
    for row in previous:
        if row['split'] not in ('development','holdout'):continue
        payload=json.loads(row['messages'][1]['content'])
        cases.append(dict(case_id='dev-'+row['case_id'], split='development',
            source_case_id=row['case_id'], former_split=row['split'], payload=payload,
            baseline_messages=row['messages']))
    if len(cases)!=40:raise ValueError('Exactly forty source adviser cases are required')
    return cases


def fresh_holdout(previous):
    old={digest_value(c['payload']) for c in previous}
    rng=random.Random(SEED)
    sizes=[size for size,count in enumerate((0,6,6,6,6,6,5,4,1)) for _ in range(count)]+[0]*8
    cases=[]
    for index,size in enumerate(sizes):
        order=list(CATALOG);rng.shuffle(order)
        eligible=sorted(order[:size]);tested=sorted(set(CATALOG)-set(eligible))
        payload=dict(eligible=eligible,tested=tested,
            development_ms={c:round(rng.uniform(30,1700),2) for c in tested},
            configurations={c:CATALOG[c] for c in eligible})
        fingerprint=digest_value(payload)
        if fingerprint in old:raise ValueError('Holdout duplicates a previous payload; do not resample after seeing outcomes')
        old.add(fingerprint)
        cases.append(dict(case_id=f'holdout-{index:02}',split='holdout',payload=payload,
            payload_sha256=fingerprint,baseline_messages=[dict(role='system',content=BASELINE_SYSTEM),
                dict(role='user',content=json.dumps(payload,separators=(',',':')))]))
    return cases


def make_schedule(cases, arms, seed):
    order=list(cases);random.Random(seed).shuffle(order)
    permutations=list(itertools.permutations(arms))
    return [dict(case_id=c['case_id'],arms=list(permutations[i%len(permutations)])) for i,c in enumerate(order)]


def choose_challenger(rows, available):
    scores=[]
    for arm in ('examples','bounded'):
        if arm not in available:continue
        selected=[r for r in rows if r['arm']==arm]
        if len(selected)!=40 or len({r['case_id'] for r in selected})!=40:
            raise ValueError('Development coverage incomplete')
        scores.append(dict(arm=arm,accepted=sum(r['decision']['origin'].startswith('hat_') for r in selected),
            exact=sum(r['decision']['origin']=='hat_exact' for r in selected),
            mean_request_ms=sum(r['total_ms'] for r in selected)/40))
    if not scores:raise ValueError('No challenger available')
    ordered=sorted(scores,key=lambda r:(-r['accepted'],-r['exact'],r['mean_request_ms'],ARMS.index(r['arm'])))
    return dict(challenger=ordered[0]['arm'],scores=scores,
        rule='accepted count, exact count, mean full request latency, examples on a remaining tie')


def specification():
    return dict(version=VERSION,seed=SEED,arms=ARMS,max_seconds=1800,
        development_cases=40,holdout_nonempty_cases=40,holdout_empty_cases=8,
        eligibility_size_counts={'1':6,'2':6,'3':6,'4':6,'5':6,'6':5,'7':4,'8':1,'0':8},
        preflight='For C0 through C7: default/custom/default; compare original-setting controls, returned text and ledgers.',
        pilot_cases=['dev-development-00','dev-development-10','dev-development-18'],
        pilot_margin=1.20,cleanup_reserve_seconds=45,
        normalization='Only a completed, eligible C[0-7] followed by one ASCII period; surrounding whitespace allowed.',
        native_stops='Original stops plus all eight candidate IDs, not only eligible IDs. Never infer suppressed text.',
        selection='Development only: accepted, exact, latency, then examples.',
        reliability_gate='40/40 nonempty challenger outputs accepted before fallback; 8/8 empty cases abstain.',
        interface_gate='Every returned candidate eligible and untested; all empty sets abstain without a model call.',
        timing='Controller request through model IPC, validation, verified recovery/restoration and CPU fallback.',
        boundary='Response reliability and cost only; no numerical benchmarks, routing promotion, or tuning-utility claim.')
