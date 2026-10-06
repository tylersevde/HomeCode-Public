#!/usr/bin/env python3
"""Offline public integrity/statistics verifier; imports no experiment modules."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile
import numpy as np

VERSION = 'public-adaptation-v1'
CPU = 'completion-cpu-20261006T120615Z-compare'
GPU = 'completion-gpu-develop-20261006T145314Z-develop'
NPU = ['completion-npu-diagnostic-20261006T112430Z-diagnostic', 'completion-npu-develop-20261006T155952Z-develop']
COLORS = {'red', 'green', 'blue', 'yellow', 'pink', 'brown', 'black', 'white'}
NATIVE = ['<|end_of_text|>', '<|eom_id|>', '<|eot_id|>']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(value):
    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def read(path):
    return loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and str(p) == name and not p.is_absolute()
            and all(x not in ('', '.', '..') for x in p.parts) and '\\' not in name,
            'Unsafe manifest path')
    return name


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def same(actual, expected, path='statistic'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path + ': keys differ')
        for key in expected:
            same(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ': length differs')
        for index, value in enumerate(expected):
            same(actual[index], value, path + '[' + str(index) + ']')
    elif isinstance(expected, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14), path + ': value differs')
    else:
        require(type(actual) is type(expected) and actual == expected, path + ': value differs')


def interval(a, b, seed=20261006, ratio=True):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(a.ndim == 1 and a.shape == b.shape and len(a) > 1, 'Unpaired blocks')
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite block data')
    require(not ratio or ((a > 0).all() and (b > 0).all()), 'Nonpositive timing')
    ix = np.random.Generator(np.random.PCG64(seed)).integers(0, len(a), (10000, len(a)))
    aa, bb = a[ix].mean(1), b[ix].mean(1)
    samples = np.sort(aa / bb if ratio else aa - bb)
    return dict(estimate=float(a.mean() / b.mean() if ratio else a.mean() - b.mean()),
        interval=[float(samples[250]), float(samples[9750])], seed=seed, samples=10000, blocks=len(a))


def stable(value):
    return .95 <= value['interval'][0] <= value['interval'][1] <= 1.05


def frozen_fields(protocol, expected):
    for key, value in expected.items():
        require(key in protocol, 'Missing protocol field: ' + key)
        same(protocol[key], value, 'protocol.' + key)


def call_checks(row, track):
    c = row['checks']
    count = 16 if track == 'cpu' and row['label'].startswith('batch') else 1
    require(type(c['count']) is int and type(c['expected_count']) is int and c['expected_count'] == c['count'] == count, 'Wrong numerical call denominator')
    require(c['request_ids'] == (list(range(16)) if count == 16 else None), 'Request mapping differs')
    for key in ['correct_count', 'warmup_match_count', 'zero_validation_error_count', 'no_unexpected_io_count']:
        require(type(c[key]) is int and 0 <= c[key] <= count, 'Invalid call field: ' + key)
    require(type(c['identities_match']) is bool and type(c['result_correct']) is bool, 'Invalid correctness type')
    require(finite(c['max_absolute_error']) and finite(c['max_tolerance_fraction']), 'Invalid numerical error')
    require(isinstance(c['output_sha256'], list) and 1 <= len(c['output_sha256']) <= count
            and len(set(c['output_sha256'])) == len(c['output_sha256'])
            and all(re.fullmatch('[a-f0-9]{64}', h) for h in c['output_sha256']), 'Invalid output hashes')
    return c['identities_match'] and c['result_correct'] and all(c[k] == count for k in (
        'correct_count', 'warmup_match_count', 'zero_validation_error_count', 'no_unexpected_io_count'))


def check_fixture_coverage(fixtures, name):
    npu = name in NPU
    cells = ([128, 512, 1024] if npu else
             [f'stream-n{n}-b{b}' for n in (128, 512, 1024) for b in (1, 4)] if name == CPU else
             [f'stream-n{n}-d{d}-b{b}' for n in (128, 512, 1024) for d in (64, 128) for b in (1, 16)])
    field = 'target_tokens' if npu else 'cell_id'
    blocks = 256 if name == CPU else 8
    expected = {(section, b, cell) for section, bs in [('pilot', [-1]), ('main', range(blocks))] for b in bs for cell in cells}
    slots = []
    ids = set()
    for f in fixtures:
        require(all(key in f for key in ('fixture_id', 'section', 'block', field)), 'Fixture mapping field missing')
        slot = (f['section'], f['block'], f[field])
        slots.append(slot)
        cell = f'n{f[field]}' if npu else f[field]
        require(f['fixture_id'] == f'{f["section"]}-{cell}-block{f["block"]}', 'Fixture identity does not identify paired slot')
        require(f['fixture_id'] not in ids, 'Duplicate fixture identity')
        ids.add(f['fixture_id'])
    require(len(slots) == len(expected) and len(set(slots)) == len(slots) and set(slots) == expected, 'Fixture pairing matrix differs')


def load_run(root, name):
    d = root / 'evidence' / name
    protocol, summary, schedule, fixtures = [read(d / (f + '.json')) for f in ('protocol', 'summary', 'schedule', 'fixtures')]
    with (d / 'measurements.jsonl').open(encoding='utf-8') as stream:
        rows = [loads(line) for line in stream]
    require(len(rows) == len(schedule), name + ': measurement schedule incomplete')
    check_fixture_coverage(fixtures, name)
    byid = {f['fixture_id']: f for f in fixtures}
    last = 0
    for row, job in zip(rows, schedule):
        require(all(row.get(k) == v for k, v in job.items()), 'Schedule/order substitution')
        require(type(row['source_line']) is int and row['source_line'] > last, 'Source line ordering differs')
        last = row['source_line']
        require(finite(row['total_ms'], True), 'Invalid complete request duration')
        require(row['fixture_id'] in byid, 'Unknown fixture')
        f = byid[row['fixture_id']]
        fields = ('section', 'block', 'target_tokens') if name in NPU else ('section', 'block', 'cell_id')
        require(all(k in row and k in f and row[k] == f[k] for k in fields), 'Fixture block/cell mapping differs')
    require(summary['complete'] is True and summary['defaults_changed'] is False, 'Incomplete comparison/default drift')
    require({r['fixture_id'] for r in rows} == set(byid), 'Fixture coverage differs')
    return protocol, summary, rows, byid


def numerical_array(rows, labels, cells, blocks, configurations=None):
    keys = set()
    shape = ([len(configurations)] if configurations else []) + [blocks, len(cells), 3, len(labels)]
    result = np.full(shape, np.nan)
    pilot = set()
    for r in rows:
        require(r['cell_id'] in cells and r['label'] in labels and type(r['repeat']) is int and 0 <= r['repeat'] < 3, 'Unknown numerical slot')
        require(r['section'] in ('pilot', 'main'), 'Unknown section')
        require(type(r['block']) is int and (r['block'] == -1 if r['section'] == 'pilot' else 0 <= r['block'] < blocks), 'Unknown block')
        ix = ([configurations.index(r['configuration'])] if configurations else []) + [r['block'], cells.index(r['cell_id']), r['repeat'], labels.index(r['label'])]
        key = tuple(ix)
        seen = pilot if r['section'] == 'pilot' else keys
        require(key not in seen, 'Duplicate numerical slot')
        seen.add(key)
        if r['section'] == 'main':
            result[key] = r['total_ms']
    require(np.isfinite(result).all() and len(pilot) == len(cells) * 3 * len(labels) * (len(configurations) if configurations else 1), 'Missing numerical slot/pilot')
    return result


def cpu_statistics(root):
    p, s, rows, fixtures = load_run(root, CPU)
    cells = [f'stream-n{n}-b{b}' for n in (128, 512, 1024) for b in (1, 4)]
    labels, configs = ['single_a', 'single_b', 'batch_a', 'batch_b'], ['baseline', 'candidate']
    require(p['blocks'] == 256 and p['repeats'] == 3 and p['batch_calls'] == 16, 'CPU protocol changed')
    frozen_fields(p, dict(bootstrap_seed=20261006, bootstrap_samples=10000, minimum_speedup=1.10,
        equivalence_interval=[.95, 1.05], max_shape_slowdown=1.05, defaults_changed=False))
    repeated = numerical_array(rows, labels, cells, 256, configs).mean(3)
    correctness = {kind: all([call_checks(r, 'cpu') for r in rows if r['label'].startswith(kind)]) for kind in ('single', 'batch')}
    metrics = []
    for kind, pair in [('single', [0, 1]), ('batch', [2, 3])]:
        checks = {}
        for ci, config in enumerate(configs):
            a, b = repeated[ci, :, :, pair[1]], repeated[ci, :, :, pair[0]]
            checks[config] = {'aggregate': interval(a.mean(1), b.mean(1)),
                **{cell: interval(a[:, i], b[:, i]) for i, cell in enumerate(cells)}}
        byshape = repeated[:, :, :, pair].mean(3)
        byblock = byshape.mean(2)
        speed = interval(*byblock)
        slowdown = max(float(byshape[1, :, i].mean() / byshape[0, :, i].mean()) for i in range(len(cells)))
        stability = all(stable(x) for group in checks.values() for x in group.values())
        qualified = bool(correctness[kind] and stability and speed['estimate'] >= 1.10 and speed['interval'][0] > 1 and slowdown <= 1.05)
        metrics.append(dict(label=kind, kind=kind, speedup=speed,
            mean_ms={config: float(byblock[i].mean()) for i, config in enumerate(configs)},
            per_cell_speedup={cell: interval(byshape[0, :, i], byshape[1, :, i]) for i, cell in enumerate(cells)},
            max_shape_slowdown=slowdown, equivalence=checks, stable=stability, correct=correctness[kind], qualified=qualified))
    same(s['metrics'], metrics, 'CPU')
    accepted = all(m['qualified'] for m in metrics)
    require(s['accepted'] is accepted and s['decision'] == ('cpu_improved' if accepted else 'no_qualified_candidate'), 'CPU disposition differs')
    return dict(measurements=len(rows), main_blocks=256, accepted=accepted, statistics='all endpoint/per-shape/control intervals reproduced')


def gpu_statistics(root):
    p, s, rows, fixtures = load_run(root, GPU)
    cells = [f'stream-n{n}-d{d}-b{b}' for n in (128, 512, 1024) for d in (64, 128) for b in (1, 16)]
    labels = ['cpu', 'cpu_duplicate', 'gpu', 'stream64']
    require(p['blocks'] == 8 and p['repeats'] == 3 and p['stage'] == 'develop', 'GPU protocol changed')
    frozen_fields(p, dict(bootstrap_seed=20261007, bootstrap_samples=10000, minimum_speedup=1.10,
        equivalence_interval=[.95, 1.05], maximum_shape_slowdown=1.05, defaults_changed=False))
    means = numerical_array(rows, labels, cells, 8).mean(2)
    correct = all([call_checks(r, 'gpu') for r in rows])
    boot = lambda a, b: interval(a, b, 20261007)
    baseline = means[:, :, :2].mean(2)
    checks = {'aggregate': boot(means[:, :, 1].mean(1), means[:, :, 0].mean(1)),
              **{cell: boot(means[:, i, 1], means[:, i, 0]) for i, cell in enumerate(cells)}}
    stability = all(stable(x) for x in checks.values())
    per = {cell: boot(baseline[:, i], means[:, i, 3]) for i, cell in enumerate(cells)}
    routes = {cell: 'stream64' if correct and stability and per[cell]['estimate'] >= 1.10 and per[cell]['interval'][0] > 1 else 'cpu' for cell in cells}
    routed = np.stack([means[:, i, 3] if routes[cell] == 'stream64' else baseline[:, i] for i, cell in enumerate(cells)], axis=1)
    speed = boot(baseline.mean(1), routed.mean(1))
    slowdown = max(float(routed[:, i].mean() / baseline[:, i].mean()) for i in range(len(cells)))
    qualified = bool(correct and stability and 'stream64' in routes.values() and speed['estimate'] >= 1.10 and speed['interval'][0] > 1 and slowdown <= 1.05)
    for key, value in dict(correct=correct, stable=stability, equivalence=checks, routes=routes,
            speedup=speed, max_shape_slowdown=slowdown, old_gpu_speedup=boot(means[:, :, 2].mean(1), means[:, :, 3].mean(1)),
            selected=dict(routes=routes) if qualified else None, accepted=False,
            decision='candidate_selected' if qualified else 'no_qualified_candidate').items():
        same(s[key], value, 'GPU.' + key)
    metrics = [dict(cell_id=cell, route=routes[cell], mean_ms={label: float(means[:, i, li].mean()) for li, label in enumerate(labels)},
        cpu_mean_ms=float(baseline[:, i].mean()), stream64_cpu_speedup=per[cell], old_gpu_speedup=boot(means[:, i, 2], means[:, i, 3]),
        routed_slowdown=float(routed[:, i].mean() / baseline[:, i].mean())) for i, cell in enumerate(cells)]
    same(s['metrics'], metrics, 'GPU.metrics')
    return dict(measurements=len(rows), main_blocks=8, accepted=False, statistics='all routes/per-shape/control intervals reproduced')


def semantic_score(output, expected, item, effective, status):
    suffix = next((x for x in sorted(effective, key=len, reverse=True) if output.endswith(x)), None)
    body = output[:-len(suffix)] if suffix else None
    result = dict(factual_correct=None, strict_correct=False, format_correct=False, parsed_color=None,
        answer_category='malformed', score_reason=None, terminal_suffix=suffix, assistant_body=body)
    if status != 'LOGICAL_END_OF_GENERATION':
        result.update(answer_category='truncated' if status == 'MAX_TOKENS_REACHED' else 'incomplete', score_reason=status)
    elif suffix is None:
        result['score_reason'] = 'missing_terminal_token'
    elif not body.strip():
        result['score_reason'] = 'empty_output'
    elif '<|' in body or '|>' in body or any(x in body for x in NATIVE):
        result['score_reason'] = 'exposed_control_marker'
    else:
        text = ' '.join(body.casefold().split()).strip().strip('.,!?;:').strip()
        one = text in COLORS
        color = text if one else None
        if not one:
            text2 = text.removeprefix('the color of ') if text.startswith('the color of ') and ' = ' not in text else text
            match = re.fullmatch(r'(item[0-9]+) (?:is|=) ([a-z]+)', text2)
            if match and match[2] in COLORS:
                if match[1] != item.lower():
                    result['score_reason'] = 'wrong_identifier'
                    return result
                color = match[2]
            elif text.startswith('it is ') and text[6:] in COLORS:
                color = text[6:]
        if color is None:
            result['score_reason'] = 'unsupported_or_ambiguous_answer'
        else:
            result.update(parsed_color=color, factual_correct=color == expected, format_correct=one,
                strict_correct=one and color == expected, answer_category='correct_fact' if color == expected else 'wrong_fact')
    return result


def check_dialogue(row, fixture, parameters):
    require(row['table_mode'] in ('full_table', 'selected_fact') and row['history_mode'] in ('actual_history', 'independent'), 'Unknown dialogue factors')
    require(row['diagnostic_only'] or (row['table_mode'], row['history_mode']) == ('full_table', 'actual_history'), 'Simplified task cannot qualify')
    if row['diagnostic_only']:
        factors = {f'{table}_{history}': (table, history) for table in ('full_table', 'selected_fact')
                   for history in ('actual_history', 'independent')}
        require(row['label'] in factors and (row['table_mode'], row['history_mode']) == factors[row['label']]
                and row['condition'] == 'punctuation', 'Diagnostic label/factor substitution')
    else:
        require(row['label'] in ('native', 'candidate') and (row['label'] != 'native' or row['condition'] == 'native'), 'Qualification label substitution')
    require([a['turn'] for a in row['turns']] == list(range(4)), 'Planned turn denominator changed')
    require(type(row['reset_verified']) is bool, 'Reset flag malformed')
    observed_turns = [a['turn'] for a in row['turns'] if a['observed']]
    failures = row['contract_failures']
    require(isinstance(failures, list), 'Invalid contract failures')
    for failure in failures:
        require(type(failure.get('turn')) is int and 0 <= failure['turn'] < 4
                and failure.get('category') == 'context_contract'
                and isinstance(failure.get('error'), str) and bool(failure['error']), 'Invalid context-contract failure')
    failed_turns = [f['turn'] for f in failures]
    require(len(set(failed_turns)) == len(failed_turns) and not set(failed_turns) & set(observed_turns), 'Failure/observation mapping overlap')
    effective = NATIVE + (['.', '\n'] if row['condition'] == 'punctuation' else [])
    require(row['native_stops'] == NATIVE and row['effective_stops'] == effective, 'Stop policy differs')
    history, absent, invalid = [], [], False
    table = dict(fixture['table'])
    require(len(table) == len(fixture['table']), 'Duplicate table identifier')
    for a in row['turns']:
        turn = a['turn']
        require(type(a['observed']) is bool and a['quarantined'] is (not a['observed']), 'Observation/quarantine overlap')
        if not a['observed']:
            absent.append(turn)
            continue
        independent = row['history_mode'] == 'independent'
        require(independent or (not invalid and not absent), 'Actual history continued after invalid/missing turn')
        q = fixture['questions'][turn]
        match = re.fullmatch(r'What color is (item[0-9]+)\?', q)
        require(match is not None and match[1] in table, 'Question does not map uniquely to table')
        item = match[1]
        require(a['expected_answer'] == fixture['answers'][turn] == table[item], 'Expected answer mapping differs')
        if row['table_mode'] == 'selected_fact':
            user = f'Reference facts: {item} = {table[item]}. {q}'
            require(a['selected_fact'] == [item, table[item]], 'Selected fact substitution')
        else:
            require(a['selected_fact'] is None, 'Full-table input was CPU selected')
            facts = '; '.join(f'{k} = {v}' for k, v in fixture['table'])
            user = f'Reference facts: {facts}. {q}' if independent or turn == 0 else q
        expected_messages = ([] if independent else history.copy())
        if not expected_messages:
            expected_messages.append(dict(role='system', content=row['system']))
        expected_messages.append(dict(role='user', content=user))
        require(a['messages'] == expected_messages, 'Actual assistant history or full-table mapping changed')
        require(a['parameters'] == parameters and a['native_stops'] == NATIVE and a['effective_stops'] == effective and a['condition'] == row['condition'], 'Generation policy drift')
        require(''.join(a['stream_chunks']) == a['output'], 'Output stream changed')
        score = semantic_score(a['output'], table[item], item, effective, a['status'])
        same(a['score'], score, 'NPU score')
        require(a['answer_correct'] is score['strict_correct'] and a['assistant_body'] == score['assistant_body'] and a['terminal_suffix'] == score['terminal_suffix'], 'Whole-answer score changed')
        c = a['token_counts']
        require(all(type(x) is int and x >= 0 for x in c.values()), 'Token counts invalid')
        require(a['context_before'] == 0 and a['submitted_matches_prompt_tokens'] is True
                and a['prompt_tokens'] == c['prompt_token_ids'] == c['submitted_token_ids']
                and a['expected_context_after'] == c['after_token_ids'], 'Full rebuild/count accounting changed')
        ledger = a['context_after'] == c['after_token_ids']
        valid = bool(ledger and score['terminal_suffix'] and score['assistant_body'].strip() and a['status'] == 'LOGICAL_END_OF_GENERATION')
        require(a['ledger_valid'] is ledger and a['valid'] is valid and bool(c['terminal_token_ids']) == bool(score['terminal_suffix']), 'Validity accounting changed')
        require(finite(a['request_ms']), 'Invalid NPU request duration')
        invalid = not valid
        if valid:
            history = expected_messages + [dict(role='assistant', content=score['assistant_body'])]
    require(absent == row['quarantined_turns'], 'Quarantined turn mapping differs')
    if row['history_mode'] == 'actual_history':
        require(observed_turns == list(range(len(observed_turns))) and absent == list(range(len(observed_turns), 4)), 'History gap/quarantine differs')
        require(len(failures) <= 1 and (not failures or failed_turns == [len(observed_turns)]), 'History failure location differs')
        require(len(observed_turns) == 4 or bool(failures) or invalid, 'Valid history stopped without failure')
    else:
        require(sorted(observed_turns + failed_turns) == [0, 1, 2, 3] and absent == failed_turns, 'Independent questions omitted without contract failure')


def npu_statistics(root, name):
    p, s, rows, fixtures = load_run(root, name)
    diagnostic = p['stage'] == 'diagnostic'
    require(p['stage'] == ('diagnostic' if name == NPU[0] else 'develop'), 'NPU stage identity changed')
    labels = ['full_table_actual_history', 'full_table_independent', 'selected_fact_actual_history', 'selected_fact_independent'] if diagnostic else ['native', 'candidate']
    require(p['blocks'] == 8 and p['turns'] == 4 and p['sizes'] == [128, 512, 1024], 'NPU denominator protocol changed')
    frozen_fields(p, dict(bootstrap_seed=20261006, bootstrap_samples=10000, candidate_count_limit=1,
        development_gate=dict(planned=96, valid=96, correct=92, per_size=29, contracts=0),
        confirmation_gate=dict(planned=192, valid=192, correct=183, per_size=58, contracts=0, min_gain=.05),
        defaults_changed=False))
    expected_slots = {(section, b, n, label) for section, blocks in [('pilot', [-1]), ('main', range(8))] for b in blocks for n in (128, 512, 1024) for label in labels}
    actual_slots = [(r['section'], r['block'], r['target_tokens'], r['label']) for r in rows]
    require(len(set(actual_slots)) == len(actual_slots) and set(actual_slots) == expected_slots, 'Missing/duplicate NPU session')
    for row in rows:
        expected_system = p['original_system'] if diagnostic or row['label'] == 'native' else p['revised_policy']['system']
        require(row['system'] == expected_system and row['diagnostic_only'] is diagnostic, 'NPU system/diagnostic role drift')
        expected_condition = 'punctuation' if diagnostic else 'native' if row['label'] == 'native' else p['revised_policy']['stop_condition']
        require(row['condition'] == expected_condition, 'Frozen stop choice differs')
        check_dialogue(row, fixtures[row['fixture_id']], p['parameters'])
    main = [r for r in rows if r['section'] == 'main']
    metrics, vectors = [], {}
    for label in labels:
        mine = [r for r in main if r['label'] == label]
        answers = [a for r in mine for a in r['turns'] if a['observed']]
        valid = sum(a['valid'] for a in answers)
        correct = sum(a['valid'] and a['answer_correct'] for a in answers)
        per = {str(n): sum(a['valid'] and a['answer_correct'] for r in mine if r['target_tokens'] == n for a in r['turns'] if a['observed']) for n in (128, 512, 1024)}
        contracts = sum(len(r['contract_failures']) + int(not r['reset_verified']) for r in mine)
        meets = valid == 96 and correct >= 92 and min(per.values()) >= 29 and contracts == 0
        metrics.append(dict(label=label, planned=96, observed=len(answers), valid=valid, correct=correct,
            per_size_correct=per, contracts=contracts, accuracy=correct / 96, meets_numeric_threshold=meets,
            qualified=bool(meets and label == 'candidate' and not diagnostic)))
        vectors[label] = [sum(a['valid'] and a['answer_correct'] for r in mine if r['block'] == b for a in r['turns'] if a['observed']) / 12 for b in range(8)]
    same(s['metrics'], metrics, 'NPU metrics')
    if not diagnostic:
        same(s['accuracy_gain'], interval(vectors['candidate'], vectors['native'], ratio=False), 'NPU accuracy gain')
    require(s['accepted'] is False and s['scientific_qualification'] is False, 'Development/diagnostic cannot confirm')
    selected = p['revised_policy'] if not diagnostic and metrics[-1]['qualified'] else None
    same(s['selected'], selected, 'NPU selected')
    require(s['decision'] == ('diagnostic_complete' if diagnostic else 'candidate_selected' if selected else 'no_qualified_candidate'), 'NPU decision differs')
    return dict(measurements=len(rows), planned_main_turns=96 * len(labels), metrics=metrics, accepted=False)


def verify_manifest(root, allow_incomplete=False):
    require(root.is_dir() and not root.is_symlink(), 'Invalid public root')
    manifest = read(root / 'PUBLICATION-MANIFEST.json')
    require(manifest['version'] == VERSION and manifest['original_seals_certify_derivatives'] is False, 'Provenance contract changed')
    require(allow_incomplete or manifest['complete'] is True, 'Unfinished publication documents')
    expected = set(manifest['files']) | {'PUBLICATION-MANIFEST.json'}
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Public symlink forbidden')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        else:
            require(path.is_dir(), 'Special public entry forbidden')
    require(actual == expected, 'Unlisted or missing public files')
    for name, row in manifest['files'].items():
        safe_name(name)
        path = root / name
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Public digest differs: ' + name)
        require(row['transform_version'] == VERSION, 'Unknown transformation version')
        if row['transformation'] in ('identity', 'identity-archive-member'):
            require(row['original_sha256'] == row['sha256'], 'Identity transformation changed bytes')
    frozen = manifest['original_source_sha256']
    require(len(frozen) == manifest['frozen_source_files'] == 136, 'Frozen source coverage changed')
    for name, value in frozen.items():
        require(manifest['files'][name]['original_sha256'] == value, 'Original source link changed')
        if name != 'efficiency/common.py':
            require(manifest['files'][name]['sha256'] == value, 'Unapproved public source change')
    evidence = read(root / 'evidence/PUBLIC-EVIDENCE-MANIFEST.json')
    require(set(evidence['sources']) == {CPU, GPU, *NPU}, 'Final evidence run coverage differs')
    for name, source in evidence['sources'].items():
        require(re.fullmatch('[a-f0-9]{64}', source['original_checksums_sha256']) is not None, 'Original seal link malformed')
        for published, entry in manifest['files'].items():
            prefix = f'evidence/{name}/'
            if not published.startswith(prefix):
                continue
            original = entry['source'].removeprefix(f'runs/{name}/')
            require(original in source['files'] and entry['original_sha256'] == source['files'][original], 'Derivative original evidence link differs')
    npu_policy = read(root / 'evidence' / NPU[1] / 'protocol.json')['revised_policy']
    require(npu_policy['diagnostic_checksums_sha256'] == evidence['sources'][NPU[0]]['original_checksums_sha256'], 'Candidate diagnostic provenance differs')
    archive = root / 'original/algorithm_efficiency_experiment.zip'
    require(digest(archive) == '18bebaa5904a2c685198e5ba5688025d3156c284243abc3c414458a8a3f88cec', 'Original ZIP differs')
    with zipfile.ZipFile(archive) as z:
        require(set(z.namelist()) == {'algorithm_efficiency_demo.py', 'algorithm_efficiency_results.json', 'algorithm_efficiency_readme.txt'}, 'Original member mapping differs')
        for name in z.namelist():
            require((root / 'original' / name).read_bytes() == z.read(name), 'Original extracted member differs')
    return manifest


def validate(root, allow_incomplete=False):
    root = Path(root)
    manifest = verify_manifest(root, allow_incomplete)
    result = dict(version=VERSION, passed=True, manifest_sha256=digest(root / 'PUBLICATION-MANIFEST.json'),
        files=len(manifest['files']), cpu=cpu_statistics(root), gpu=gpu_statistics(root),
        npu={name: npu_statistics(root, name) for name in NPU},
        scope='Public integrity and independent retained-statistics/context/scoring verification; no new hardware or full private audit.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path('.'))
    parser.add_argument('--allow-incomplete-docs', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.allow_incomplete_docs), indent=2, allow_nan=False))
