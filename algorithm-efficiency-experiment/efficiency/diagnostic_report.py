"""Offline diagnostic analysis; all outcomes contribute to accuracy denominators."""
import base64
from collections import Counter, defaultdict
import html
import json
import shlex
import statistics

from .common import atomic_json, digest_file, read_jsonl, utc
from .diagnostic import compare_position_pair
from .report import table, write_csv


def quality(rows):
    return dict(total=len(rows), factual_correct=sum(r['factual_correct'] is True for r in rows),
        strict_correct=sum(r['strict_correct'] for r in rows),
        format_correct=sum(r['format_correct'] for r in rows),
        wrong_fact=sum(r['answer_category'] == 'wrong_fact' for r in rows),
        unscorable=sum(r['factual_correct'] is None for r in rows),
        categories=dict(Counter(r['answer_category'] for r in rows)))


def grouped_quality(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys, values), **quality(group)) for values, group in sorted(groups.items())]


def analyze(rows, schedule):
    measurements = [r for r in rows if r['event'] == 'measurement']
    repeated = [r for r in measurements if r['study'] == 'repeatability']
    position = [r for r in measurements if r['study'] == 'position']
    quarantines = [r for r in rows if r['event'] in ('request_quarantined','session_quarantined')]
    groups = defaultdict(list)
    for r in repeated:
        groups[(r['case_id'], r['route'])].append(r)
    planned = Counter((j['case_id'], j['route']) for j in schedule['repeatability'])
    repetition = []
    for (case, route), expected in sorted(planned.items()):
        group = groups[(case, route)]
        counts = Counter(r['output'] for r in group)
        total_pairs = len(group)*(len(group)-1)//2
        agreements = sum(n*(n-1)//2 for n in counts.values())
        repetition.append(dict(case_id=case, route=route, expected=expected, **quality(group),
            unique_outputs=len(counts), output_counts=dict(counts),
            unique_recorded_prompts=len({r['effective_input_sha256'] for r in group}),
            context_count_mismatches=sum(not r['token_ledger_valid'] for r in group),
            pairwise_agreement=agreements/total_pairs if total_pairs else None,
            median_request_ms=statistics.median(r['request_ms'] for r in group) if group else None))
    route_comparisons = []
    for case in sorted({key[0] for key in planned}):
        raw, native = groups[(case, 'raw')], groups[(case, 'native_chat')]
        raw_counts, native_counts = Counter(r['output'] for r in raw), Counter(r['output'] for r in native)
        denominator = len(raw)*len(native)
        route_comparisons.append(dict(case_id=case, raw_total=len(raw), native_total=len(native),
            output_histograms_equal=raw_counts == native_counts if denominator else None,
            cross_route_exact_agreement=(sum(v*native_counts[k] for k,v in raw_counts.items())/denominator
                                         if denominator else None)))
    indexed = defaultdict(lambda: defaultdict(list))
    for row in position:
        indexed[row['pair_id']][row['arm']].append(row)
    comparisons = [c for job in schedule['position']
                   for c in compare_position_pair(indexed[job['pair_id']], job['pair_id'])]
    recurring = defaultdict(list)
    for c in comparisons:
        if c['fixture_id'] is not None:
            recurring[(c['fixture_id'],c['order_index'],c['turn'])].append(c)
    recurrence = [dict(fixture_id=k[0],order_index=k[1],turn=k[2],observed_pairs=len(group),
        eligible=sum(c['eligible'] for c in group),
        output_divergences=sum(c['eligible'] and not c['outputs_equal'] for c in group),
        rebuild_correct_retain_wrong=sum(c['eligible'] and c['rebuild_factual_correct'] is True
                                        and c['retain_factual_correct'] is False for c in group),
        retain_correct_rebuild_wrong=sum(c['eligible'] and c['retain_factual_correct'] is True
                                        and c['rebuild_factual_correct'] is False for c in group))
        for k, group in sorted(recurring.items())]
    duplicates = len(measurements) - len({(r['study'], r.get('case_id', r.get('pair_id')),
                                          r['repeat'], r['route'], r['arm'], r['turn']) for r in measurements})
    summary = dict(expected_responses=schedule['expected_responses'], observed_responses=len(measurements),
        context_count_mismatches=sum(not r['token_ledger_valid'] for r in measurements),
        quarantines=quarantines, skipped_dependent_responses=sum(r.get('skipped_turns',0) for r in quarantines),
        duplicate_observations=duplicates,
        repeatability=repetition, route_comparisons=route_comparisons,
        position_quality=quality(position), repeatability_quality=quality(repeated),
        quality_by_route=grouped_quality(repeated, ['route']),
        quality_by_arm=grouped_quality(position, ['arm']),
        quality_by_position_arm=grouped_quality(position, ['position','arm']),
        quality_by_turn_arm=grouped_quality(position, ['turn','arm']),
        quality_by_position_turn_arm=grouped_quality(position, ['position','turn','arm']),
        quality_by_size_position_turn_arm=grouped_quality(position, ['target_tokens','position','turn','arm']),
        divergence_recurrence=recurrence,
        position_comparisons=dict(expected=len(comparisons), eligible=sum(c['eligible'] for c in comparisons),
            equal_recorded_inputs=sum(c['inputs_equal'] for c in comparisons),
            matched_input_output_divergences=sum(c['eligible'] and not c['outputs_equal'] for c in comparisons),
            matched_rebuild_correct_retain_wrong=sum(c['eligible'] and c['rebuild_factual_correct'] is True
                                                    and c['retain_factual_correct'] is False for c in comparisons),
            matched_retain_correct_rebuild_wrong=sum(c['eligible'] and c['retain_factual_correct'] is True
                                                    and c['rebuild_factual_correct'] is False for c in comparisons),
            first_divergences=[c for c in comparisons if c['first_divergence']],
            exclusions=dict(Counter(c['exclusion_reason'] for c in comparisons if c['exclusion_reason']))))
    return summary, comparisons, measurements


def conclusions(summary):
    text = []
    repeat = summary['repeatability']
    variable = [r for r in repeat if r['unique_outputs'] > 1]
    observed_repeats = [r for r in repeat if r['total'] >= 2]
    text.append('Too few repeated requests to assess repeatability.' if not observed_repeats else
                f'{len(variable)} of {len(repeat)} frozen-prompt/route groups produced multiple exact outputs. '
                'Variation under repeated cleared requests limits any claim that a single arm difference is caused by context reuse.'
                if variable else 'All observed repeats within each frozen-prompt/route group were identical; this bounded run found no repeatability variation.')
    routes = [r for r in summary['route_comparisons'] if r['cross_route_exact_agreement'] is not None
              and r['cross_route_exact_agreement'] < 1]
    observed_routes = [r for r in summary['route_comparisons'] if r['cross_route_exact_agreement'] is not None]
    text.append('No completed raw/native route pairs were available.' if not observed_routes else
                f'{len(routes)} frozen prompts had raw/native output disagreement. '
                'Native serialization is internal to the SDK; route disagreement identifies an API/serialization/runtime question, not a proven identical-input model difference.'
                if routes else 'Raw and native-chat routes agreed exactly on all observed frozen-prompt outputs. Matching local renders and context counts support parity but do not expose native internal serialization.')
    c = summary['position_comparisons']
    text.append(f'{c["matched_input_output_divergences"]} of {c["eligible"]} completed comparisons with equal recorded token histories produced different outputs; '
                f'{c["matched_rebuild_correct_retain_wrong"]} had a correct rebuilt answer and a wrong retained fact, and '
                f'{c["matched_retain_correct_rebuild_wrong"]} showed the reverse. '
                'These observations localize differences to execution/context conditions after the checked input boundary; they do not identify a particular kernel, numeric precision or firmware cause.')
    text.append('The three cyclic question orders balance beginning/middle/end across turns by design. Position and turn summaries describe the selected tables; repeated answers are correlated, and these counts are not general-model accuracy estimates. Later turns with different histories remain in accuracy totals but are excluded from matched-input comparisons.')
    text.append(f'{summary["context_count_mismatches"]} responses had runtime context counts different from the re-tokenized transcript. '
                f'{summary["skipped_dependent_responses"]} dependent responses were skipped after a verified context reset. '
                'Invalid ledgers are excluded from matched-input comparisons. Decoded text does not reveal the original generated token IDs, so a count mismatch alone does not establish cache corruption. '
                'Skipped responses reduce observed balance and coverage; they are reported separately from measured accuracy.')
    return text


def save_reproducer(directory, measurements, comparisons):
    """Smallest observed failing conversation, without claiming global minimization."""
    eligible = { (c['pair_id'], c['turn']) for c in comparisons
                if c['eligible'] and c['rebuild_factual_correct'] is True and c['retain_factual_correct'] is False }
    candidates = [r for r in measurements if r['study'] == 'position' and r['arm'] == 'retain'
                  and (r['pair_id'], r['turn']) in eligible]
    if candidates:
        chosen = min(candidates, key=lambda r: (r['effective_input_tokens'], r['turn'], r['pair_id']))
        inputs = json.loads((directory / 'diagnostic-inputs.json').read_text())
        fixture = next(f for f in inputs['fixtures'] if f['pair_id'] == chosen['fixture_id'])
        evidence = [r for r in measurements if r.get('pair_id') == chosen['pair_id'] and r['turn'] <= chosen['turn']]
        payload = dict(kind='matched_history_context_difference', fixture=fixture,
            question_order=chosen['question_order'], failing_turn=chosen['turn'], records=evidence,
            explanation='Smallest observed completed retained-only wrong fact by effective input tokens. The fixture is not globally minimized. Replay both arms from an empty context through this turn.')
    else:
        candidates = [r for r in measurements if r['factual_correct'] is not True]
        if not candidates:
            atomic_json(directory / 'minimal-failing-case.json', dict(kind='none_observed'))
            return
        chosen = min(candidates, key=lambda r: (r['effective_input_tokens'], r['turn']))
        payload = dict(kind='single_answer_failure', record=chosen,
                       explanation='Smallest observed answer failure by recorded input tokens; not globally minimized.')
    payload['model_sha256'] = json.loads((directory / 'manifest.json').read_text())['model_sha256']
    payload['decoding'] = dict(do_sample=False, seed=12345, max_generated_tokens=16)
    config = json.loads((directory / 'config.json').read_text())
    if chosen['study'] == 'position':
        from .common import ROOT
        payload['replay_command'] = shlex.join(['python3', str(ROOT / 'experiment.py'), 'run',
            '--phase', 'hat', '--profile', 'diagnostic', '--source-run', config['source_run'],
            '--replay-fixture', chosen['fixture_id'], '--replay-order', str(chosen['order_index']+1),
            '--output', str(directory.parent / (directory.name+'-replay')), '--max-seconds', '3600'])
        payload['replay'] = 'Runs this one table and order three times in both arms under the same supervisor (18 responses); use a new output directory for every replay.'
    else:
        payload['replay'] = 'Replay the recorded frozen prompt through repeat_request; the full diagnostic CLI repeats all six selected frozen prompts.'
    atomic_json(directory / 'minimal-failing-case.json', payload)


def plot_results(directory, summary, telemetry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for axis, dimension, labels in [(axes[0], 'position', ['beginning','middle','end']),
                                     (axes[1], 'turn', [0,1,2])]:
        rows = summary['quality_by_position_arm' if dimension == 'position' else 'quality_by_turn_arm']
        for arm, offset, color in [('rebuild',-.18,'#286da8'),('retain',.18,'#b7572b')]:
            values = {r[dimension]: r for r in rows if r['arm'] == arm}
            y = [100*values[label]['factual_correct']/values[label]['total'] if label in values else 0 for label in labels]
            axis.bar([i+offset for i in range(3)], y, width=.35, label=arm, color=color)
        axis.set(xticks=range(3), xticklabels=labels if dimension=='position' else ['1','2','3'],
                 ylim=(0,105), ylabel='Factual correctness (% of all responses)',
                 xlabel='Fact position' if dimension=='position' else 'Conversation turn')
        axis.legend()
        axis.grid(axis='y', alpha=.2)
    fig.suptitle('Balanced diagnostic: selected tables, including unscorable answers as failures')
    for ext in ('png','svg'):
        fig.savefig(directory / f'diagnostic-quality.{ext}', dpi=160)
    plt.close(fig)
    if telemetry:
        fig, axis = plt.subplots(figsize=(11,3.5), layout='constrained')
        for field, label in [('cpu_temp_c','Pi CPU'), ('hat_max_c','HAT maximum sensor')]:
            valid = [r for r in telemetry if r.get(field) is not None]
            axis.plot([r['elapsed_seconds']/60 for r in valid], [r[field] for r in valid], label=label)
        axis.set(xlabel='Elapsed minutes', ylabel='Temperature (°C)', title='Observed hardware temperatures')
        axis.legend()
        axis.grid(alpha=.2)
        for ext in ('png','svg'):
            fig.savefig(directory / f'diagnostic-thermal.{ext}', dpi=160)
        plt.close(fig)
    return ['diagnostic-quality'] + (['diagnostic-thermal'] if telemetry else [])


def build_report(directory, charts=True):
    config = json.loads((directory / 'config.json').read_text())
    schedule = json.loads((directory / 'schedule.json').read_text())
    rows, errors = read_jsonl(directory / 'diagnostic.jsonl')
    telemetry, telemetry_errors = read_jsonl(directory / 'telemetry.jsonl')
    outcome = (json.loads((directory / 'outcome.json').read_text()) if (directory / 'outcome.json').exists()
               else dict(status='incomplete', elapsed_seconds=None, stop_reason='No outcome file'))
    summary, comparisons, measurements = analyze(rows, schedule)
    summary.update(generated_utc=utc(), outcome=outcome, jsonl_errors=errors,
        telemetry_jsonl_errors=telemetry_errors,
        complete=outcome['status'] == 'complete' and summary['observed_responses'] == summary['expected_responses']
            and not summary['duplicate_observations'] and not errors,
        protocol_completed=outcome['status'] == 'complete' and any(r['event']=='complete' for r in rows)
            and summary['observed_responses']+summary['skipped_dependent_responses']==summary['expected_responses'],
        thermal=dict(max_cpu_temp_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None), default=None),
                     max_hat_temp_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None), default=None),
                     throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None}),
                     max_context_tokens=max((r['context_after'] for r in measurements), default=0)))
    summary['conclusions'] = conclusions(summary)
    atomic_json(directory / 'summary.json', summary)
    comparison_lookup = {(r['pair_id'], r['turn']):r for r in comparisons}
    exported = []
    for row in measurements:
        row = dict(row)
        comp = comparison_lookup.get((row.get('pair_id'), row['turn']))
        row.update(first_divergence=comp['first_divergence'] if comp else None,
                   comparison_eligible=comp['eligible'] if comp else None)
        exported.append(row)
    write_csv(directory / 'diagnostic-measurements.csv', exported)
    write_csv(directory / 'diagnostic-comparisons.csv', comparisons)
    write_csv(directory / 'diagnostic-repeatability.csv', summary['repeatability'])
    write_csv(directory / 'diagnostic-quality.csv', summary['quality_by_size_position_turn_arm'])
    write_csv(directory / 'diagnostic-recurrence.csv', summary['divergence_recurrence'])
    save_reproducer(directory, measurements, comparisons)
    plots = plot_results(directory, summary, telemetry) if charts else []
    sections = ['<h1>Diagnosing HAT answer errors</h1>',
        f'<p>{summary["observed_responses"]}/{summary["expected_responses"]} measured responses. '
        f'Status: {html.escape(outcome["status"])}. Complete coverage: {summary["complete"]}.</p>',
        f'<p>Dependent requests skipped after context quarantine: {summary["skipped_dependent_responses"]}. '
        f'Protocol completed: {summary["protocol_completed"]}. Context-count mismatches: {summary["context_count_mismatches"]}.</p>',
        '<p>Qwen2.5-1.5B-Instruct, HailoRT 5.1.1, greedy decoding, seed 12345, maximum 16 generated tokens; '
        '1792-token experiment budget. This is a diagnostic of selected pilot failures, not a quality fix.</p>',
        '<h2>Findings and limits</h2>', *['<p>'+html.escape(s)+'</p>' for s in summary['conclusions']],
        '<h2>Repeated frozen requests</h2>',
        table(['Frozen prompt','Route','N / planned','Distinct outputs','Exact agreement','Factual / strict correct','Local context mismatches'],
          [[r['case_id'],r['route'],f'{r["total"]}/{r["expected"]}',r['unique_outputs'],
            f'{r["pairwise_agreement"]:.1%}' if r['pairwise_agreement'] is not None else 'insufficient repeats',
            f'{r["factual_correct"]} / {r["strict_correct"]}',r['context_count_mismatches']] for r in summary['repeatability']]),
        '<p>Exact agreement considers every unordered pair of repeats within a group. These pairs are dependent; no independent-sample confidence interval is claimed.</p>',
        '<h2>Balanced fact position and conversation turn</h2>']
    for title, key, dims in [('By cache arm','quality_by_arm',['arm']),
                             ('By position and arm','quality_by_position_arm',['position','arm']),
                             ('By turn and arm','quality_by_turn_arm',['turn','arm'])]:
        sections += ['<h3>'+title+'</h3>', table(dims+['N','Factual correct','Strict correct','Wrong fact','Unscorable'],
            [[r[d]+1 if d=='turn' else r[d] for d in dims] +
             [r[k] for k in ('total','factual_correct','strict_correct','wrong_fact','unscorable')] for r in summary[key]])]
    sections += ['<p>Six tables: one original divergence and one distinct first-turn-correct control per target size. '
        'Three cyclic orders, three repeats, two cache arms, three turns produce 324 responses; six frozen prompts × two routes × ten repeats produce 120. '
        'The smoke profile uses two prompts once per route and one table with all three orders (22 responses).</p>',
        '<h2>Answer scoring</h2><p>Strict correctness requires the expected color alone, allowing case, whitespace and terminal punctuation. '
        'Factual correctness also accepts “The color of itemNNN is color”, “itemNNN is color”, “itemNNN = color”, and “It is color”. '
        'Identifiers must match exactly. Unsupported prose, multiple answers, embedded role/stop markers, missing terminators and truncated generations are unscorable. '
        'Every response stays in the denominator. Legacy strict scores are exported separately for comparison with the pilot.</p>',
        '<h2>First divergences</h2>', table(['Pair','Turn','Equal input','Eligible','Rebuild fact correct','Retain fact correct'],
          [[c['pair_id'],c['turn']+1,c['inputs_equal'],c['eligible'],c['rebuild_factual_correct'],c['retain_factual_correct']]
           for c in summary['position_comparisons']['first_divergences']]),
        '<h2>Repeated directional factual differences</h2>',
        table(['Fixture','Order','Turn','Eligible repeats','Rebuild correct / retain wrong','Retain correct / rebuild wrong'],
            [[r['fixture_id'],r['order_index']+1,r['turn']+1,r['eligible'],
              r['rebuild_correct_retain_wrong'],r['retain_correct_rebuild_wrong']]
             for r in summary['divergence_recurrence']
             if r['rebuild_correct_retain_wrong'] or r['retain_correct_rebuild_wrong']]),
        '<h2>Hardware and artifacts</h2><pre>'+html.escape(json.dumps(summary['thermal'],indent=2))+'</pre>',
        '<p>Raw outputs, messages, prompts, token hashes, context counts and completion status are in diagnostic.jsonl. '
        'The frozen source evidence and its checksums are in source-run/; diagnostic-inputs.json and schedule.json record selection and execution order. '
        'minimal-failing-case.json contains the smallest observed qualifying failure, without claiming global minimization. '
        'Latency is secondary; energy and device KV-cache bytes were not measured.</p>']
    if outcome.get('stop_reason'):
        sections.append('<p>Stop reason: '+html.escape(outcome['stop_reason'])+'</p>')
    for stem in plots:
        data = base64.b64encode((directory / f'{stem}.png').read_bytes()).decode()
        sections.append(f'<figure><img alt="{stem}" src="data:image/png;base64,{data}"></figure>')
    sections.append('<details><summary>Configuration</summary><pre>'+html.escape(json.dumps(config,indent=2))+'</pre></details>')
    css = 'body{font:16px/1.5 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eff4f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}h2{margin-top:36px}'
    (directory / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>HAT answer diagnostic</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    checksums = {str(p.relative_to(directory)):digest_file(p) for p in sorted(directory.rglob('*'))
                 if p.is_file() and p != directory / 'checksums.json'}
    atomic_json(directory / 'checksums.json', checksums)
    print(json.dumps(dict(report=str(directory / 'report.html'), complete=summary['complete'],
                         responses=summary['observed_responses']), indent=2))
