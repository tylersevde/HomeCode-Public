"""Offline analysis. Experimental units are conversation pairs, not turns."""
import base64
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import random
import statistics

from .common import atomic_json, digest_file, read_jsonl, utc


def median(values):
    return statistics.median(values) if values else None


def bootstrap_median(values, seed=20261002, samples=10000):
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(samples))
    return [estimates[int(.025*samples)], estimates[int(.975*samples)]]


def cpu_summary(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['event'] == 'measurement':
            groups[row['sequence_length']][row['variant']].append(row)
    summaries = []
    for length, variants in sorted(groups.items()):
        medians = {name: median([r['milliseconds'] for r in values]) for name, values in variants.items()}
        required = {'full_prefix_reference', 'latest_query_no_cache', 'cached_stream'}
        result = dict(sequence_length=length, median_ms=medians,
                      sample_counts={k: len(v) for k, v in variants.items()},
                      max_absolute_error=max(r['max_absolute_error'] for values in variants.values() for r in values))
        if required <= variants.keys():
            result['cached_vs_latest_speedup'] = medians['latest_query_no_cache']/medians['cached_stream']
            result['cached_vs_full_speedup'] = medians['full_prefix_reference']/medians['cached_stream']
            result['persistent_kv_cache_bytes'] = variants['cached_stream'][0]['persistent_kv_cache_bytes']
        summaries.append(result)
    return summaries


def compare_hat(rows, config):
    measurements = [r for r in rows if r['event'] == 'measurement']
    grouped = defaultdict(dict)
    for row in measurements:
        grouped[(row['pair_id'], row['turn'])][row['arm']] = row
    comparisons = []
    for (pair, turn), arms in sorted(grouped.items()):
        rebuild, retain = arms.get('rebuild'), arms.get('retain')
        reason = None
        if not rebuild or not retain:
            reason = 'missing_arm'
        elif rebuild['effective_input_sha256'] != retain['effective_input_sha256']:
            reason = 'different_token_histories'
        elif not rebuild['token_ledger_valid'] or not retain['token_ledger_valid']:
            reason = 'invalid_context_ledger'
        elif any(r['completion_status'] != 'LOGICAL_END_OF_GENERATION' for r in (rebuild, retain)):
            reason = 'incomplete_generation'
        elif any(r['first_visible_ms'] is None for r in (rebuild, retain)):
            reason = 'no_visible_output'
        row = dict(pair_id=pair, turn=turn, target_tokens=(rebuild or retain)['target_tokens'],
                   eligible=reason is None, exclusion_reason=reason,
                   outputs_equal=bool(rebuild and retain and rebuild['output'] == retain['output']),
                   both_correct=bool(rebuild and retain and rebuild['correct'] and retain['correct']))
        if reason is None:
            row.update(first_visible_speedup=rebuild['first_visible_ms']/retain['first_visible_ms'],
                       request_speedup=rebuild['request_ms']/retain['request_ms'],
                       rebuild_first_visible_ms=rebuild['first_visible_ms'],
                       retain_first_visible_ms=retain['first_visible_ms'],
                       rebuild_request_ms=rebuild['request_ms'], retain_request_ms=retain['request_ms'])
        comparisons.append(row)
    summaries = []
    for size in sorted(set(r['target_tokens'] for r in measurements)):
        observed = [r for r in measurements if r['target_tokens'] == size]
        compared = [r for r in comparisons if r['target_tokens'] == size]
        eligible = [r for r in compared if r['eligible']]
        follows = [r for r in eligible if r['turn'] > 0]
        controls = [r for r in eligible if r['turn'] == 0]
        pair_groups = defaultdict(list)
        for r in follows:
            pair_groups[r['pair_id']].append(r)
        # Use only pairs with all specified follow-up turns for aggregate inference.
        complete = [v for v in pair_groups.values() if len(v) == config['turns']-1]
        pair_ratios = [median([r['first_visible_speedup'] for r in v]) for v in complete]
        ci = bootstrap_median(pair_ratios)
        accuracy = {}
        for arm in ('rebuild', 'retain'):
            arm_rows = [r for r in observed if r['arm'] == arm]
            accuracy[arm] = dict(correct=sum(r['correct'] for r in arm_rows), total=len(arm_rows),
                                 only_expected_color_mentioned=sum(r.get('mentioned_colors') == [r.get('expected_answer')] for r in arm_rows),
                                 expected=config['hat_pairs']*config['turns'])
        quality = all(a['correct'] == a['expected'] and a['total'] == a['expected'] for a in accuracy.values())
        complete_coverage = len(eligible) == config['hat_pairs']*config['turns']
        sessions = [r for r in rows if r['event']=='session_complete' and
                    r['pair_id'] in {v['pair_id'] for v in observed} and r['completed_turns']==config['turns']]
        summaries.append(dict(target_tokens=size,
            initial_prompt_token_range=[min(r['initial_prompt_tokens'] for r in observed), max(r['initial_prompt_tokens'] for r in observed)],
            accuracy=accuracy, quality_preserved=quality,
            matched_turns=len(eligible), excluded_turns=len(compared)-len(eligible),
            observed_paired_turns=len(compared), expected_paired_turns=config['hat_pairs']*config['turns'],
            exact_output_matches=sum(r['outputs_equal'] for r in compared),
            correct_matched_followups=sum(r['both_correct'] for r in follows),
            complete_followup_pairs=len(complete),
            followup_first_visible_median_ms={arm: median([r[f'{arm}_first_visible_ms'] for r in follows]) for arm in ('rebuild', 'retain')},
            paired_followup_speedup=median(pair_ratios), speedup_ci95=ci,
            first_turn_control_speedup=median([r['first_visible_speedup'] for r in controls]),
            median_conversation_request_ms={arm: median([r['request_total_ms'] for r in sessions if r['arm']==arm]) for arm in ('rebuild','retain')},
            median_conversation_wall_ms={arm: median([r['session_wall_ms'] for r in sessions if r['arm']==arm]) for arm in ('rebuild','retain')},
            pair_speedups=pair_ratios,
            useful_improvement=bool(quality and complete_coverage and ci and ci[0] > 1)))
    return summaries, comparisons


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def charts_for(directory, summary, telemetry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    files = []
    if summary['cpu']:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        xs = np.arange(len(summary['cpu']))
        for i, (key, label, color) in enumerate([
                ('full_prefix_reference', 'Full prefix reference', '#94a3b8'),
                ('latest_query_no_cache', 'Latest query, no cache', '#d97706'),
                ('cached_stream', 'Cached stream', '#0891b2')]):
            ys = [r['median_ms'].get(key, float('nan')) for r in summary['cpu']]
            ax.bar(xs+(i-1)*.24, ys, width=.23, label=label, color=color)
        ax.set_xticks(xs, [str(r['sequence_length']) for r in summary['cpu']])
        ax.set(xlabel='Sequence length (dimension 64, float64)', ylabel='Median milliseconds (log scale)',
               title='CPU attention: fixed inputs and weights', yscale='log')
        ax.legend()
        files.append(('cpu-latency', fig))
    if summary['hat']:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        xs = np.arange(len(summary['hat']))
        for i, (arm, color) in enumerate([('rebuild', '#d97706'), ('retain', '#0891b2')]):
            values = [r['followup_first_visible_median_ms'][arm] for r in summary['hat']]
            ax.bar(xs+(i-.5)*.32, [v if v is not None else float('nan') for v in values],
                   width=.31, label=arm.capitalize()+' history', color=color)
        ax.set_xticks(xs, [str(r['target_tokens']) for r in summary['hat']])
        ax.set(xlabel='Initial prompt target (tokens)', ylabel='Median first visible output (ms)',
               title='HAT follow-up questions with matched token histories')
        ax.legend()
        files.append(('hat-latency', fig))
    if telemetry:
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        for key, label in [('cpu_temp_c', 'Pi CPU'), ('hat_ts0_c', 'HAT sensor 0'), ('hat_ts1_c', 'HAT sensor 1')]:
            valid = [r for r in telemetry if r.get(key) is not None]
            ax.plot([r['elapsed_seconds']/60 for r in valid], [r[key] for r in valid], label=label)
        ax.set(xlabel='Elapsed minutes', ylabel='Temperature (°C)', title='Observed temperatures throughout the run')
        ax.legend()
        files.append(('temperatures', fig))
    for stem, fig in files:
        fig.savefig(directory / f'{stem}.png', dpi=160)
        fig.savefig(directory / f'{stem}.svg')
        plt.close(fig)
    return [stem for stem, _ in files]


def fmt(value, suffix=''):
    return '—' if value is None else f'{value:.3f}{suffix}'


def table(headers, rows):
    return '<table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def build_report(directory, charts=True):
    config = json.loads((directory / 'config.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text()) if (directory / 'manifest.json').exists() else {}
    outcome = json.loads((directory / 'outcome.json').read_text()) if (directory / 'outcome.json').exists() else {'status': 'in_progress'}
    cpu, cpu_errors = read_jsonl(directory / 'cpu.jsonl')
    hat, hat_errors = read_jsonl(directory / 'hat.jsonl')
    telemetry, telemetry_errors = read_jsonl(directory / 'telemetry.jsonl')
    hat_results, comparisons = compare_hat(hat, config)
    cpu_results = cpu_summary(cpu)
    if outcome['status'] != 'complete':
        for result in hat_results:
            result['useful_improvement'] = False
    hat_environment = json.loads((directory / 'hat-environment.json').read_text()) if (directory / 'hat-environment.json').exists() else {}
    thermal = {key: max((r[key] for r in telemetry if r.get(key) is not None), default=None)
               for key in ('cpu_temp_c', 'hat_ts0_c', 'hat_ts1_c', 'worker_rss_bytes')}
    summary = dict(schema_version=1, generated_utc=utc(), profile=config['profile'], outcome=outcome,
        cpu=cpu_results, hat=hat_results, thermal_maxima=thermal,
        model_loading_seconds=hat_environment.get('loading_seconds'),
        maximum_context_tokens=max((r['context_after'] for r in hat if r['event']=='measurement'),default=None),
        damaged_lines=dict(cpu=cpu_errors, hat=hat_errors, telemetry=telemetry_errors),
        observed_cpu_measurements=sum(r['event']=='measurement' for r in cpu),
        expected_cpu_measurements=len(config['cpu_lengths'])*config['cpu_repeats']*3 if config['phase'] in ('cpu','all') else 0,
        observed_hat_measurements=sum(r['event']=='measurement' for r in hat),
        expected_hat_measurements=len(config['hat_sizes'])*config['hat_pairs']*config['turns']*2 if config['phase'] in ('hat','all') else 0,
        inference_policy='All expected answers must be correct in both arms and all histories matched; exploratory 95% pair bootstrap lower bound must exceed 1 to label a useful improvement.')
    atomic_json(directory / 'summary.json', summary)
    write_csv(directory / 'cpu-measurements.csv', [r for r in cpu if r['event']=='measurement'])
    write_csv(directory / 'hat-measurements.csv', [r for r in hat if r['event']=='measurement'])
    write_csv(directory / 'paired-comparisons.csv', comparisons)
    write_csv(directory / 'telemetry.csv', telemetry)
    plots = charts_for(directory, summary, telemetry) if charts else []
    content = [f'<h1>Caching in practice</h1><p class="subtitle">Raspberry Pi 5 · Hailo-10H · {html.escape(config["profile"])} profile</p>',
        '<h2>Run status and coverage</h2>',
        f'<p><strong>{html.escape(outcome["status"])}</strong> · {fmt(outcome.get("elapsed_seconds"))} seconds. '
        f'{html.escape(outcome.get("stop_reason") or "All selected measurement phases completed.")}</p>',
        f'<p>CPU observations: {summary["observed_cpu_measurements"]}/{summary["expected_cpu_measurements"]}. '
        f'HAT observations: {summary["observed_hat_measurements"]}/{summary["expected_hat_measurements"]}. '
        f'Damaged JSONL lines: {len(cpu_errors)+len(hat_errors)+len(telemetry_errors)}.</p>',
        '<h2>CPU attention results</h2>',
        '<p>One synthetic attention layer, float64, dimension 64, one requested BLAS thread. '
        'Every measured output is checked at rtol=atol=1e-10. The primary baseline recomputes keys and values for only the latest query.</p>',
        table(['Length','Latest query ms','Cached ms','Speedup','KV bytes','Maximum error'],
              [[r['sequence_length'],fmt(r['median_ms'].get('latest_query_no_cache')),fmt(r['median_ms'].get('cached_stream')),
                fmt(r.get('cached_vs_latest_speedup'),'×'),r.get('persistent_kv_cache_bytes','—'),f'{r["max_absolute_error"]:.3g}'] for r in cpu_results]),
        '<h2>HAT conversation results</h2>',
        '<p>Same compiled Qwen2.5 1.5B model, fixed greedy generation, maximum 16 output tokens. '
        'Both arms use the runtime’s ordinary decoding cache. The experiment changes whether conversation context survives between requests. '
        'Time to first visible output includes context clearing where required, generation and reading; prompt preparation and diagnostic RPCs are outside this interval. '
        'Stream events are stored separately from token counts.</p>',
        table(['Initial target','Rebuild first output ms','Retain first output ms','Paired speedup [95% CI]','Correct rebuild / retain','Matched turns','Useful improvement'],
              [[r['target_tokens'],fmt(r['followup_first_visible_median_ms']['rebuild']),fmt(r['followup_first_visible_median_ms']['retain']),
                fmt(r['paired_followup_speedup'],'×')+(' ['+', '.join(fmt(v) for v in r['speedup_ci95'])+']' if r['speedup_ci95'] else ' [insufficient pairs]'),
                ' / '.join(f'{r["accuracy"][a]["correct"]}/{r["accuracy"][a]["total"]}' for a in ('rebuild','retain')),
                f'{r["matched_turns"]}/{r["expected_paired_turns"]}', 'Yes' if r['useful_improvement'] else 'Not established'] for r in hat_results]),
        '<p>Speedups use the median of each conversation pair’s follow-up ratios. Bootstrap intervals resample whole pairs, '
        'so multiple turns are not treated as independent experiments. Seven pairs per size provide exploratory evidence. '
        '“Useful improvement” requires complete coverage, matched histories, all answers correct in both arms, and an interval entirely above 1.</p>',
        '<h3>First-turn controls and exclusions</h3>',
        table(['Target','First-turn speedup','Complete follow-up pairs','Excluded paired turns','Exact output matches'],
              [[r['target_tokens'],fmt(r['first_turn_control_speedup'],'×'),r['complete_followup_pairs'],r['excluded_turns'],r['exact_output_matches']] for r in hat_results]),
        '<p>Every attempted observation is retained. Missing arms, divergent token histories, ledger failures, empty output and incomplete generations '
        'are explicitly excluded from matched-input latency comparisons. Accuracy totals include incorrect and incomplete answers.</p>',
        '<h3>Answer format and complete conversations</h3>',
        '<p>The primary rubric requires the color alone (allowing case, whitespace and terminal punctuation). '
        'The supplementary column below counts responses mentioning only the expected color, even if accompanied by extra prose; '
        'that count does not change the primary correctness verdict.</p>',
        table(['Target','Only expected color mentioned: rebuild / retain','Median conversation request ms: rebuild / retain','Wall ms including diagnostics: rebuild / retain'],
              [[r['target_tokens'], ' / '.join(str(r['accuracy'][a]['only_expected_color_mentioned']) for a in ('rebuild','retain')),
                ' / '.join(fmt(r['median_conversation_request_ms'][a]) for a in ('rebuild','retain')),
                ' / '.join(fmt(r['median_conversation_wall_ms'][a]) for a in ('rebuild','retain'))] for r in hat_results]),
        f'<p>Initial model loading: {fmt(summary["model_loading_seconds"])} seconds. '
        f'Maximum observed conversation occupancy: {summary["maximum_context_tokens"]} tokens.</p>',
        '<h2>Hardware observations</h2>',
        table(['Metric','Maximum'],[[k,fmt(v)] for k,v in thermal.items()]),
        '<p>CPU and HAT phases run sequentially. Experiment stop limits: CPU 80°C, HAT 85°C; blocks begin below CPU 65°C and HAT 60°C. '
        'Current or new historical throttle/power flags, stale essential telemetry and runtime failures stop the run. '
        'HAT temperature comes from an isolated sampler. Host RSS does not measure device KV-cache memory. Power and energy were not measured.</p>',
        '<h2>Reproducibility</h2>',
        f'<p>Python {html.escape(manifest.get("python","unknown"))}; BLAS {html.escape(manifest.get("blas_library","unknown"))}. '
        f'Model SHA-256: <code>{html.escape(manifest.get("model_sha256","not used"))}</code>.</p>',
        '<p>The original archive reports x86 timings; this study computes speedups within each Pi workload. CPU and HAT results describe different workloads. '
        'Run manifests include source hashes, a source snapshot, configuration and model identity. Raw JSONL, CSV, exact prompts, outputs, fixtures and telemetry accompany this report.</p>']
    for stem in plots:
        encoded = base64.b64encode((directory / f'{stem}.png').read_bytes()).decode()
        content.append(f'<figure><img alt="{html.escape(stem)}" src="data:image/png;base64,{encoded}"></figure>')
    content.append('<details><summary>Configuration</summary><pre>'+html.escape(json.dumps(config,indent=2))+'</pre></details>')
    css = 'body{font:16px/1.55 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#172033;background:#fafbfc}h1{font-size:38px;margin-bottom:0}h2{margin-top:36px}.subtitle{color:#536176}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #d5dce4}th{background:#eef3f8}code,pre{overflow-wrap:anywhere;white-space:pre-wrap}figure{margin:32px 0}img{width:100%;height:auto}'
    (directory / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Caching in practice</title><style>'+css+'</style><main>'+''.join(content)+'</main></html>')
    # Only ordinary files in this run are hashed; the checksum inventory excludes itself.
    checksums = {str(p.relative_to(directory)): digest_file(p) for p in sorted(directory.rglob('*'))
                 if p.is_file() and p.name != 'checksums.json'}
    atomic_json(directory / 'checksums.json', checksums)
    print(json.dumps({'report':str(directory / 'report.html'),'summary':str(directory / 'summary.json'),
                      'status':outcome['status'],'cpu_measurements':summary['observed_cpu_measurements'],
                      'hat_measurements':summary['observed_hat_measurements']},indent=2))
