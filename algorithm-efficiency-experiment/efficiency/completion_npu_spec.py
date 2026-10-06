"""Frozen NPU diagnosis and one evidence-informed full-table qualification."""
from copy import deepcopy
import hashlib
import math
import re

from .common import digest_value
from .hat import SYSTEM
from .reliability_spec import interval, table, sizing, williams
from .study_spec import BINDINGS, MODEL_HASHES, PARAMETERS

VERSION = 'completion-npu-v1'
EVENTS = 'completion-npu.jsonl'
BLOCKS = {'diagnostic': 8, 'develop': 8, 'confirm': 16}
STAGE_SECONDS = 7200
DIAGNOSTIC_ARMS = (
    'full_table_actual_history', 'full_table_independent',
    'selected_fact_actual_history', 'selected_fact_independent',
)
NATIVE_STOPS = ['<|end_of_text|>', '<|eom_id|>', '<|eot_id|>']


def policy(config):
    value = config.get('revised_policy')
    if config['stage'] == 'diagnostic':
        if value is not None:
            raise ValueError('Diagnostic cannot preselect a revised policy')
        return None
    if not isinstance(value, dict) or set(value) != {
        'version', 'id', 'system', 'stop_condition', 'justification', 'diagnostic_checksums_sha256'
    }:
        raise ValueError('One explicit, evidence-informed revised policy is required')
    if value['version'] != 1 or not isinstance(value['id'], str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', value['id']):
        raise ValueError('Invalid revised policy identity')
    if not isinstance(value['system'], str) or not value['system'].strip() or value['system'].strip() == SYSTEM:
        raise ValueError('Unchanged historical full-table policy cannot be retried')
    if value['stop_condition'] not in ('native', 'punctuation'):
        raise ValueError('Unsupported frozen stop condition')
    if not isinstance(value['justification'], str) or not value['justification'].strip():
        raise ValueError('Diagnostic justification required before qualification data')
    if not isinstance(value['diagnostic_checksums_sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', value['diagnostic_checksums_sha256']):
        raise ValueError('Revised policy must identify sealed diagnostic evidence')
    return deepcopy(value)


def validate_config(config, budget=False):
    if config.get('profile') != 'completion' or config.get('track') != 'npu' or config.get('stage') not in BLOCKS:
        raise ValueError('Unknown completion NPU stage')
    result = policy(config)
    if budget:
        if config.get('phase') != 'hat':
            raise ValueError('NPU requires monitored HAT phase')
        key = 'confirmation_charged_seconds' if config['stage'] == 'confirm' else 'development_charged_seconds'
        prior = config[key]
        reserved = config['reserved_seconds']
        limit = config['max_seconds']
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (prior, reserved, limit)):
            raise ValueError('Invalid NPU stage allowance')
        if not (0 <= prior and 120 < limit <= reserved and prior + reserved <= STAGE_SECONDS + 1e-6):
            raise ValueError('Diagnostic and qualification share the development allowance')
    return result


def specification(config):
    selected = validate_config(config)
    return dict(version=VERSION, track='npu', stage=config['stage'], blocks=BLOCKS[config['stage']],
        development_seconds=STAGE_SECONDS, confirmation_seconds=STAGE_SECONDS,
        diagnostic_charged_to='develop', cleanup_seconds=120, pilot_margin=1.2,
        sizes=[128, 512, 1024], turns=4, diagnostic_arms=list(DIAGNOSTIC_ARMS),
        diagnostic_stop_condition='punctuation', diagnostic_only=config['stage'] == 'diagnostic',
        model_sha256=MODEL_HASHES['llama'], runtime='5.1.1', parameters=PARAMETERS,
        bindings=BINDINGS, original_system=SYSTEM, native_stops=NATIVE_STOPS,
        extra_stops=['.', '\n'], revised_policy=selected,
        sizing='largest original-control first prompt at or below target; identical complete table in every matched arm',
        history='all actual previous user prompts and model assistant bodies; never corrected answers',
        independent='fresh complete question with no earlier user or assistant messages',
        selected_fact='CPU selects only the queried input fact; model answer remains raw diagnostic output',
        context_policy='full_rebuild_each_turn', scoring='unchanged strict whole-answer rubric',
        qualifying_task='full_table_actual_history', candidate_count_limit=1,
        development_gate=dict(planned=96, valid=96, correct=92, per_size=29, contracts=0),
        confirmation_gate=dict(planned=192, valid=192, correct=183, per_size=58, contracts=0, min_gain=.05),
        bootstrap_seed=20261006, bootstrap_samples=10000, bootstrap_unit='paired whole blocks',
        control='unchanged native-stop full-table actual-history dialogue',
        defaults_changed=False)


def seed(config, fid):
    key = f'{VERSION}|{config["fixture_namespace"]}|{config["stage"]}|{fid}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def fixtures(config):
    validate_config(config)
    return [dict(fixture_id=f'{section}-n{n}-block{block}', section=section, block=block,
                 target_tokens=n, cell_id=f'n{n}', seed=seed(config, f'{section}-n{n}-block{block}'))
            for section, blocks in (('pilot', [-1]), ('main', range(BLOCKS[config['stage']])))
            for block in blocks for n in (128, 512, 1024)]


def arms(config):
    if config['stage'] == 'diagnostic':
        return [dict(label=label, table_mode='full_table' if label.startswith('full_table') else 'selected_fact',
                     history_mode='actual_history' if label.endswith('actual_history') else 'independent',
                     condition='punctuation', system=SYSTEM, diagnostic_only=True) for label in DIAGNOSTIC_ARMS]
    selected = policy(config)
    return [dict(label=label, table_mode='full_table', history_mode='actual_history',
                 condition='native' if label == 'native' else selected['stop_condition'],
                 system=SYSTEM if label == 'native' else selected['system'], diagnostic_only=False)
            for label in ('native', 'candidate')]


def schedule(config, fs):
    options = arms(config)
    labels = [a['label'] for a in options]
    bylabel = {a['label']: a for a in options}
    jobs = []
    for f in fs:
        block = max(0, f['block'])
        size_index = (128, 512, 1024).index(f['target_tokens'])
        order = williams(labels, block + size_index) if len(labels) == 4 else (labels if (block + size_index) % 2 == 0 else labels[::-1])
        for label in order:
            jobs.append(dict(fixture_id=f['fixture_id'], section=f['section'], block=f['block'],
                             target_tokens=f['target_tokens'], **bylabel[label]))
    return jobs


def user_message(fixture, turn, table_mode, initial=False):
    """Construct inputs only. Expected answers never influence these prompts."""
    question = fixture['questions'][turn]
    if table_mode == 'selected_fact':
        from .reliability_worker import select_fact
        item, color = select_fact(fixture['table'], question)
        return f'Reference facts: {item} = {color}. {question}'
    if initial:
        facts = '; '.join(f'{key} = {value}' for key, value in fixture['table'])
        return f'Reference facts: {facts}. {question}'
    return question


def planned_messages(fixture, arm, turn, history):
    independent = arm['history_mode'] == 'independent'
    result = [] if independent else deepcopy(history)
    if not result:
        result.append(dict(role='system', content=arm['system']))
    result.append(dict(role='user', content=user_message(fixture, turn, arm['table_mode'], initial=independent or turn == 0)))
    return result


def analyze(events, config):
    validate_config(config)
    result = dict(track='npu', stage=config['stage'], complete=False, selected=None, accepted=False,
                  scientific_qualification=False, diagnostic_only=config['stage'] == 'diagnostic',
                  decision='incomplete', metrics=[], defaults_changed=False)
    jobs = [j for j in schedule(config, fixtures(config)) if j['section'] == 'main']
    rows = [e for e in events if e['event'] == 'measurement' and e['section'] == 'main']
    if not any(e['event'] == 'measurement_complete' for e in events) or len(rows) != len(jobs):
        return result
    if any(any(row.get(k) != v for k, v in job.items()) for row, job in zip(rows, jobs)):
        return result
    blocks = BLOCKS[config['stage']]
    vectors = {}
    for arm in arms(config):
        mine = [r for r in rows if r['label'] == arm['label']]
        answers = [a for r in mine for a in r['response']['result']['rows']]
        valid = sum(bool(a['valid']) for a in answers)
        correct = sum(bool(a['valid'] and a['answer_correct']) for a in answers)
        per = {str(n): sum(bool(a['valid'] and a['answer_correct']) for r in mine if r['target_tokens'] == n
                           for a in r['response']['result']['rows']) for n in (128, 512, 1024)}
        contracts = sum(len(r['response']['result']['contract_failures']) + int(not r['response']['result']['reset_verified']) for r in mine)
        meets = (valid == blocks * 12 and correct >= (183 if config['stage'] == 'confirm' else 92)
                 and min(per.values()) >= (58 if config['stage'] == 'confirm' else 29) and contracts == 0)
        result['metrics'].append(dict(label=arm['label'], planned=blocks * 12, observed=len(answers), valid=valid,
            correct=correct, per_size_correct=per, contracts=contracts, accuracy=correct / (blocks * 12),
            meets_numeric_threshold=meets, qualified=bool(meets and arm['label'] == 'candidate' and not arm['diagnostic_only'])))
        vectors[arm['label']] = [sum(bool(a['valid'] and a['answer_correct']) for r in mine if r['block'] == b
                                     for a in r['response']['result']['rows']) / 12 for b in range(blocks)]
    result['complete'] = True
    if config['stage'] == 'diagnostic':
        result['decision'] = 'diagnostic_complete'
        return result
    gain = interval(vectors['candidate'], vectors['native'], ratio=False)
    result['accuracy_gain'] = gain
    eligible = result['metrics'][1]['qualified'] and (config['stage'] == 'develop' or gain['estimate'] >= .05 and gain['interval'][0] > 0)
    result['metrics'][1]['qualified'] = bool(eligible)
    if eligible:
        result['selected'] = policy(config)
        result['accepted'] = config['stage'] == 'confirm'
        result['scientific_qualification'] = result['accepted']
        result['decision'] = 'confirmed' if result['accepted'] else 'candidate_selected'
    else:
        result['decision'] = 'no_qualified_candidate'
    return result
