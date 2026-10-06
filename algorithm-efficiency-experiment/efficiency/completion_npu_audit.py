"""Offline reconstruction independent of model scoring and qualification analysis."""
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .common import digest_file, digest_value, read_jsonl
from .completion_evidence import audit_budget
from .hat import SYSTEM
from .hybrid import verify_artifacts
from .monitor import safety_reason
from .refine_audit import require
from .reliability_audit import check_model_row, check_interval
from .study_spec import BINDINGS, MODEL_HASHES, PARAMETERS
from .study_worker import renderer
from .completion_npu_spec import (EVENTS, BLOCKS, NATIVE_STOPS, specification,
                                  fixtures, schedule, table, arms, policy, validate_config)


def read(path):
    return json.loads(Path(path).read_text())


def check_dialogue(result, fixture, arm, env, size):
    require(result['arm'] == arm and result['diagnostic_only'] == arm['diagnostic_only'], 'NPU factor arm changed')
    effective = env['native_stops'] + (['.', '\n'] if arm['condition'] == 'punctuation' else [])
    require(result['native_stops'] == env['native_stops'] and result['effective_stops'] == effective, 'Stop settings changed')
    require(result['reset_verified'], 'NPU reset failed')
    rows, failures = result['rows'], result['contract_failures']
    turns = [r['turn'] for r in rows]
    failed_turns = [f['turn'] for f in failures]
    require(turns == sorted(set(turns)) and all(type(t) is int and 0 <= t < 4 for t in turns), 'Turn order or denominator changed')
    require(len(set(failed_turns)) == len(failed_turns) and not set(turns) & set(failed_turns), 'Contract failure turns changed')
    for failure in failures:
        require(type(failure['turn']) is int and 0 <= failure['turn'] < 4 and failure['category'] == 'context_contract'
                and isinstance(failure['error'], str) and bool(failure['error']), 'Unaccounted context-contract failure')
    history = []
    for row in rows:
        turn = row['turn']
        independent = arm['history_mode'] == 'independent'
        transcript = [] if independent else deepcopy(history)
        if not transcript:
            transcript.append(dict(role='system', content=arm['system']))
        question = fixture['questions'][turn]
        selected = None
        if arm['table_mode'] == 'selected_fact':
            item = re.fullmatch(r'What color is (item[0-9]+)\?', question)[1]
            found = [pair for pair in fixture['table'] if pair[0] == item]
            require(len(found) == 1, 'Ambiguous selected fact')
            selected = found[0]
            content = f'Reference facts: {selected[0]} = {selected[1]}. {question}'
        elif independent or turn == 0:
            facts = '; '.join(f'{key} = {value}' for key, value in fixture['table'])
            content = f'Reference facts: {facts}. {question}'
        else:
            content = question
        transcript.append(dict(role='user', content=content))
        require(row['selected_fact'] == selected and row['selector_ms'] >= 0, 'CPU input selection differs')
        body, valid = check_model_row(row, transcript, fixture['answers'][turn], question, env, arm['condition'])
        if turn == 0 and arm['table_mode'] == 'full_table' and arm['system'] == SYSTEM:
            require(row['prompt_token_ids'] == size['token_ids'], 'Original full-table sizing tokens changed')
        if not independent:
            require(turns == list(range(len(rows))), 'Actual history has skipped turns')
            if not valid:
                require(turn == turns[-1], 'Invalid actual history continued')
            else:
                history = transcript + [dict(role='assistant', content=body)]
    if arm['history_mode'] == 'actual_history':
        require(result['quarantined_turns'] == list(range(len(rows), 4)), 'History quarantine denominator changed')
        require(len(failures) <= 1 and (not failures or failed_turns == [len(rows)]), 'History failure location changed')
        require(len(rows) == 4 or bool(failures) or bool(rows and not rows[-1]['valid']), 'Valid history stopped without failure')
    else:
        require(sorted(turns + failed_turns) == [0, 1, 2, 3], 'Independent questions were omitted')
        require(result['quarantined_turns'] == failed_turns, 'Independent contract failures changed')


def decisions(rows, config, summary):
    """Recount planned answers and recompute the paired interval from raw rows."""
    count = BLOCKS[config['stage']]
    expected_metrics, vectors = [], {}
    for arm in arms(config):
        mine = [r for r in rows if r['label'] == arm['label']]
        observed = [a for r in mine for a in r['response']['result']['rows']]
        valid = sum(bool(a['valid']) for a in observed)
        correct = sum(bool(a['valid'] and a['answer_correct']) for a in observed)
        per = {str(n): sum(bool(a['valid'] and a['answer_correct']) for r in mine if r['target_tokens'] == n
                           for a in r['response']['result']['rows']) for n in (128, 512, 1024)}
        contracts = sum(len(r['response']['result']['contract_failures']) + int(not r['response']['result']['reset_verified']) for r in mine)
        meets = valid == count * 12 and correct >= (183 if config['stage'] == 'confirm' else 92) and min(per.values()) >= (58 if config['stage'] == 'confirm' else 29) and not contracts
        expected_metrics.append(dict(label=arm['label'], planned=count * 12, observed=len(observed), valid=valid,
            correct=correct, per_size_correct=per, contracts=contracts, accuracy=correct / (count * 12),
            meets_numeric_threshold=bool(meets), qualified=bool(meets and arm['label'] == 'candidate' and not arm['diagnostic_only'])))
        vectors[arm['label']] = [sum(bool(a['valid'] and a['answer_correct']) for r in mine if r['block'] == b
                                     for a in r['response']['result']['rows']) / 12 for b in range(count)]
    selected, accepted = None, False
    decision = 'diagnostic_complete'
    if config['stage'] != 'diagnostic':
        gain, ci = check_interval(summary['accuracy_gain'], vectors['candidate'], vectors['native'], False)
        eligible = expected_metrics[1]['qualified'] and (config['stage'] == 'develop' or gain >= .05 and ci[0] > 0)
        expected_metrics[1]['qualified'] = bool(eligible)
        if eligible:
            selected = policy(config)
            accepted = config['stage'] == 'confirm'
        decision = ('confirmed' if accepted else 'candidate_selected') if eligible else 'no_qualified_candidate'
    require(summary['metrics'] == expected_metrics, 'Reconstructed NPU counts/gates differ')
    require(summary['selected'] == selected and summary['accepted'] == accepted and summary['scientific_qualification'] == accepted
            and summary['decision'] == decision and summary['diagnostic_only'] == (config['stage'] == 'diagnostic')
            and summary['defaults_changed'] is False and summary['track'] == 'npu' and summary['stage'] == config['stage'],
            'NPU qualification or diagnostic role changed')


def audit(directory, require_seal=True):
    directory, checks = Path(directory), []
    try:
        if require_seal:
            verify_artifacts(directory)
            checks.append('artifact seal')
        config, manifest = read(directory / 'config.json'), read(directory / 'manifest.json')
        audit_budget(directory, require_final=require_seal)
        checks.append('cumulative shared budget receipt')
        selected = validate_config(config, budget=True)
        require(read(directory / 'protocol.json') == specification(config), 'Frozen NPU protocol changed')
        freeze = read(directory / 'freeze.json')
        require(freeze['source_sha256'] == manifest['source_sha256'], 'NPU source identity differs')
        require(manifest['config'] == config, 'Manifest configuration differs')
        for relative, checksum in manifest['source_sha256'].items():
            require(digest_file(directory / 'source' / relative) == checksum, 'Archived NPU source changed')
        require(freeze['config_sha256'] == digest_file(directory / 'config.json') and
                freeze['protocol_sha256'] == digest_file(directory / 'protocol.json'), 'NPU configuration freeze differs')
        require(freeze['prior_fixtures_sha256'] == digest_file(directory / 'prior-fixtures.json'), 'Prior fixture registry changed')
        require(freeze['model_sha256'] == manifest['model_sha256'] == MODEL_HASHES['llama'], 'Frozen model changed')
        require(digest_file(directory / 'historical-rationale.txt') == freeze['historical_rationale']['sha256'], 'Historical diagnostic rationale changed')
        expected_parents = set() if config['stage'] == 'diagnostic' else {'diagnostic'}
        if config['stage'] == 'confirm':
            expected_parents.add('develop')
        require(set(config.get('parents', {})) == set(freeze['parents']) == expected_parents, 'NPU parent set differs')
        for stage, entry in freeze['parents'].items():
            parent = Path(entry['path'])
            require(parent.resolve() != directory.resolve(), 'Self-referencing NPU parent')
            require(config['parents'][stage] == str(parent), 'NPU parent path changed')
            verify_artifacts(parent)
            require(digest_file(parent / 'checksums.json') == entry['checksums_sha256'], 'NPU parent seal changed')
            pc, ps = read(parent / 'config.json'), read(parent / 'summary.json')
            require(pc['stage'] == stage and pc['track'] == 'npu' and ps['complete'], 'NPU parent stage changed')
            require(read(parent / 'validation.json')['passed'], 'NPU parent audit did not pass')
            require(audit(parent)['passed'], 'Independent NPU parent reconstruction failed')
            if stage == 'diagnostic':
                require(ps['decision'] == 'diagnostic_complete' and ps['selected'] is None and not ps['accepted']
                        and entry['checksums_sha256'] == selected['diagnostic_checksums_sha256'], 'Diagnostic provenance changed')
            else:
                require(ps['selected'] == selected and ps['decision'] == 'candidate_selected', 'Confirmation candidate changed')
                require(read(parent / 'manifest.json')['source_sha256'] == manifest['source_sha256'], 'Source changed since qualification development')
        if selected:
            require(read(directory / 'npu-policy.json') == selected and freeze['revised_policy_sha256'] == digest_value(selected), 'Frozen revised policy changed')
        else:
            require(freeze['revised_policy_sha256'] is None, 'Diagnostic selected a candidate')
        checks.append('source, model, budgets, frozen policy and parent evidence')
        fs, specs, sizes = read(directory / 'fixtures.json'), fixtures(config), read(directory / 'sizing.json')
        prior, seen, hashes = read(directory / 'prior-fixtures.json'), set(), set()
        require(len(fs) == len(specs) and set(sizes) == {f['fixture_id'] for f in fs}, 'Fixture/sizing matrix incomplete')
        for f, spec in zip(fs, specs):
            require(all(f[k] == v for k, v in spec.items()) and f == table(spec, len(f['table'])), 'Fresh table reconstruction differs')
            require(f['seed'] not in prior['seeds'] and f['seed'] not in seen and f['table_sha256'] not in prior['hashes']
                    and f['table_sha256'] not in hashes, 'NPU fixture was reused')
            seen.add(f['seed']); hashes.add(f['table_sha256'])
            entry = sizes[f['fixture_id']]
            require([r[0] for r in entry['counts']] == list(range(4, 301)), 'Largest-fit sizing candidates incomplete')
            best = max(n for n, tokens in entry['counts'] if tokens <= f['target_tokens'])
            require(entry['count'] == len(f['table']) == best and len(entry['token_ids']) == dict(entry['counts'])[best], 'Largest-fit sizing changed')
        jobs = schedule(config, fs)
        require(read(directory / 'schedule.json') == jobs, 'NPU schedule changed')
        events, damaged = read_jsonl(directory / EVENTS)
        require(not damaged, 'Damaged NPU events')
        frozen = [e for e in events if e['event'] == 'fixtures_frozen']
        require(len(frozen) == 1 and frozen[0]['fixtures_sha256'] == digest_file(directory / 'fixtures.json')
                and frozen[0]['schedule_sha256'] == digest_file(directory / 'schedule.json')
                and frozen[0]['revised_policy_sha256'] == freeze['revised_policy_sha256'], 'Fixture/policy freeze differs')
        observations = [e for e in events if e['event'] == 'measurement']
        require(len(observations) == len(jobs), 'NPU measurement matrix incomplete')
        for event, job in zip(observations, jobs):
            require(all(event[k] == value for k, value in job.items()), 'Measurement order or factor arm changed')
            require(event['started'] < event['ended'] and event['total_ms'] == (event['ended'] - event['started']) * 1000, 'NPU timing changed')
        ready = [e for e in events if e['event'] == 'worker_ready']
        releases = [e for e in events if e['event'] == 'worker_release']
        restores = [e for e in events if e['event'] == 'native_stops_restored']
        require(len(ready) == len(releases) == len(restores) == 1, 'NPU ownership lifecycle incomplete')
        start, release, restore = ready[0], releases[0], restores[0]
        env = start['environment']
        require(env['hailort_version'] == '5.1.1' and env['model_sha256'] == MODEL_HASHES['llama'] and env['model_id'] == 'llama'
                and env['parameters'] == PARAMETERS and env['template_bindings'] == BINDINGS
                and set(env['native_stops']) == set(NATIVE_STOPS) and env['context_policy'] == 'full_rebuild_each_turn', 'NPU runtime or template contract changed')
        probe = renderer(env['prompt_template'], 'llama')([dict(role='system', content='Test'), dict(role='user', content='Test')])
        require(probe == env['template_probe'] and probe.startswith(BINDINGS['bos_token']) and probe.count(BINDINGS['bos_token']) == 1, 'Native template probe changed')
        require(start['instance'] == release['instance'] == restore['instance'] == 'llama-completion'
                and release['owner'] == start['owner'] and not release['alive'] and not release['forced'], 'Unclean NPU owner release')
        require(restore['response']['owner'] == start['owner'] and restore['response']['result'] == dict(stops=env['native_stops'], context_tokens=0), 'Native stops/context restoration missing')
        byid = {f['fixture_id']: f for f in fs}
        positions = {id(e): i for i, e in enumerate(events)}
        require(positions[id(start)] < positions[id(frozen[0])] < positions[id(observations[0])]
                <= positions[id(observations[-1])] < positions[id(restore)] < positions[id(release)], 'NPU lifecycle event order differs')
        for event in observations:
            response = event['response']
            require(event['instance'] == start['instance'] and response['owner'] == start['owner'], 'NPU request owner changed')
            require(event['started'] <= response['submitted'] <= response['received'] <= response['ended'] <= response['delivered'] <= event['ended'], 'NPU transport timing changed')
            arm = {k: event[k] for k in ('label', 'table_mode', 'history_mode', 'condition', 'system', 'diagnostic_only')}
            check_dialogue(response['result'], byid[event['fixture_id']], arm, env, sizes[event['fixture_id']])
        checks.append('fresh matched tables, complete schedule, raw answers and actual history')
        gates = [e for e in events if e['event'] == 'pilot_gate']
        require(len(gates) == 1, 'Pilot gate missing')
        gate = gates[0]
        require(gate['multiplier'] == BLOCKS[config['stage']] and gate['required_seconds'] == gate['pilot_seconds'] * BLOCKS[config['stage']] * 1.2
                and gate['fits'] and gate['required_seconds'] <= gate['remaining_work_seconds'], 'Pilot feasibility gate changed')
        main = [e for e in observations if e['section'] == 'main']
        require(positions[id(gate)] < positions[id(main[0])], 'Main data preceded pilot feasibility gate')
        complete = [e for e in events if e['event'] == 'measurement_complete']
        require(len(complete) == 1 and positions[id(complete[0])] > positions[id(release)]
                and not any(e['event'] == 'failure' for e in events), 'NPU stage did not complete cleanly')
        summary = read(directory / 'summary.json')
        require(summary['complete'], 'NPU summary incomplete')
        decisions(main, config, summary)
        outcome = read(directory / 'outcome.json')
        runtime = read(directory / 'runtime-finalization.json')
        finite = lambda value: isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        require(outcome['status'] == 'complete' and not outcome.get('stop_reason') and finite(outcome['elapsed_seconds'])
                and 0 <= outcome['elapsed_seconds'] <= config['max_seconds'], 'Supervised NPU outcome or budget differs')
        require(all(finite(runtime[key]) for key in ('started_monotonic', 'finished_monotonic', 'charged_seconds', 'reserved_seconds'))
                and runtime['reserved_seconds'] == config['reserved_seconds']
                and runtime['charged_seconds'] == runtime['finished_monotonic'] - runtime['started_monotonic']
                and outcome['elapsed_seconds'] <= runtime['charged_seconds'] <= config['reserved_seconds']
                and runtime['original_governor'] == config['original_governor'] and runtime['restored'],
                'Full NPU attempt budget/restoration differs')
        require(runtime['started_monotonic'] <= observations[0]['started'] <= observations[-1]['ended'] <= runtime['finished_monotonic'],
                'NPU measurement outside charged attempt')
        restoration = read(directory / 'governor-restoration.json')
        require(restoration['restored'] and restoration['current'] == config['original_governor'], 'Governor restoration failed')
        telemetry, damaged = read_jsonl(directory / 'telemetry.jsonl')
        require(telemetry and not damaged, 'NPU telemetry missing or damaged')
        for row in telemetry:
            require(not row.get('stop_reason') and safety_reason(row, manifest['initial_throttle_flags'],
                    require_hat=row['phase'] not in ('preflight', 'complete', 'stopped')) is None, 'NPU telemetry safety gate failed')
        require(all(finite(row['monotonic']) for row in telemetry)
                and [row['monotonic'] for row in telemetry] == sorted(row['monotonic'] for row in telemetry)
                and runtime['started_monotonic'] <= telemetry[0]['monotonic'] <= observations[0]['started']
                and observations[-1]['ended'] <= telemetry[-1]['monotonic'] <= runtime['finished_monotonic'],
                'NPU telemetry does not cover measurement interval')
        checks.append('paired whole-block statistics, planned denominators, safety and cleanup')
        return dict(passed=True, checks=checks, measurements=len(observations), fixtures=len(fs),
                    offline_limit='Saved token IDs and counts are checked for consistency; no hardware tokenizer replay.')
    except Exception as exc:
        return dict(passed=False, checks=checks, error=f'{type(exc).__name__}: {exc}')
