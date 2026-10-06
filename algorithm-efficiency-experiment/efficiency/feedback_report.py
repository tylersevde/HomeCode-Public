"""Offline analysis of frozen attention policies and equal-budget searches."""
import base64
from collections import Counter
import html
import json
import math
import statistics
import shutil

from .common import ROOT, atomic_json, digest_file, read_jsonl
from .feedback_spec import CATALOG, CELLS, SEED, STRATEGIES, median_by_seed, paired_interval
from .report import table, write_csv


def expected_measurements():
    expected = set()
    for cell in CELLS:
        cid = cell['cell_id']
        for seed in range(3):
            for repeat in range(3):
                for arm in ('numpy', 'native1', 'native4'):
                    expected.add(('baseline', None, None, cid, seed, repeat, arm))
        for strategy in STRATEGIES:
            for round_number in range(1, 4):
                for stage, repeats in (('development', 3), ('validation', 5)):
                    for seed in range(3):
                        for repeat in range(repeats):
                            for arm in ('incumbent', 'candidate'):
                                expected.add((stage, strategy, round_number, cid, seed, repeat, arm))
        for seed in range(8):
            for repeat in range(5):
                for arm in ('baseline', *STRATEGIES):
                    expected.add(('final', None, None, cid, seed, repeat, arm))
    return expected


def measurement_key(row):
    return tuple(row.get(k) for k in ('stage', 'strategy', 'round', 'cell_id', 'seed_index', 'repeat', 'arm'))


def analyze(rows, outcome, policies, choices, environment, correctness, replay=False):
    measures = [r for r in rows if r['event']=='measurement']
    actual = Counter(measurement_key(r) for r in measures); expected = expected_measurements()
    coverage = set(actual)==expected and all(v==1 for v in actual.values())
    concurrent = [r for r in rows if r['event']=='concurrency']
    expected_concurrency = {(r, c) for r in range(6) for c in ('cpu_serial', 'cpu_overlap', 'gpu_serial', 'gpu_overlap')}
    concurrency_coverage = (len(concurrent)==24 and {(r['repeat'], r['condition']) for r in concurrent}==expected_concurrency)
    protocol_complete = (coverage and concurrency_coverage and len(choices)==6 and correctness.get('passed') is True
                         and any(r['event']=='complete' for r in rows) and outcome.get('status')=='complete')
    all_correct = all(r['correct'] for r in measures) and all(r['numerical']['correct'] for r in concurrent)
    cells, values = [], {}
    for cell in CELLS:
        selected = [r for r in measures if r['stage']=='final' and r['cell_id']==cell['cell_id']]
        if len(selected)!=120: continue
        values[cell['cell_id']] = {arm: median_by_seed(selected, arm) for arm in ('baseline', *STRATEGIES)}
        for strategy in STRATEGIES:
            data = values[cell['cell_id']]
            interval = paired_interval([(data['baseline'][s], data[strategy][s]) for s in range(8)], SEED)
            cells.append(dict(cell_id=cell['cell_id'], mode=cell['mode'], strategy=strategy,
                backend=policies.get(strategy, {}).get('routes', {}).get(cell['cell_id']),
                baseline_ms=statistics.mean(data['baseline'].values()), policy_ms=statistics.mean(data[strategy].values()),
                **interval))
    def aggregate(left, right, selected):
        if any(c['cell_id'] not in values for c in selected): return None
        pairs = [(sum(values[c['cell_id']][left][s] for c in selected),
                  sum(values[c['cell_id']][right][s] for c in selected)) for s in range(8)]
        result = paired_interval(pairs, SEED)
        result.update(left_mean_request_ms=sum(a for a, _ in pairs)/8/len(selected),
                      right_mean_request_ms=sum(b for _, b in pairs)/8/len(selected))
        return result
    comparisons = {}
    for strategy in STRATEGIES:
        comparisons[strategy] = {mode: aggregate('baseline', strategy,
            [c for c in CELLS if mode=='mixture' or c['mode']==mode]) for mode in ('stream', 'prefill', 'mixture')}
    comparisons['hat_vs_coordinate'] = aggregate('coordinate', 'hat', CELLS)
    final_rows = [r for r in measures if r['stage']=='final']
    identical_routes = {strategy: bool(len(final_rows)==1440 and all(
        {r['backend'] for r in final_rows if r['cell_id']==cell['cell_id'] and r['arm']=='baseline'} ==
        {r['backend'] for r in final_rows if r['cell_id']==cell['cell_id'] and r['arm']==strategy}
        for cell in CELLS)) for strategy in STRATEGIES}
    development = []
    for cell in CELLS:
        for candidate in sorted(CATALOG):
            selected = [r for r in measures if r['stage']=='development' and r['strategy']=='coordinate'
                        and r['cell_id']==cell['cell_id']]
            rounds = {r['round'] for r in selected if r['arm']=='candidate' and r['backend']==candidate}
            selected = [r for r in selected if r['round'] in rounds]
            if len(selected)!=18: continue
            cpu_ms = statistics.mean(median_by_seed(selected, 'incumbent').values())
            gpu_ms = statistics.mean(median_by_seed(selected, 'candidate').values())
            development.append(dict(cell_id=cell['cell_id'], mode=cell['mode'], candidate=candidate,
                cpu_ms=cpu_ms, gpu_ms=gpu_ms, gpu_over_cpu=gpu_ms/cpu_ms,
                evidence='Coordinate-search development inputs; descriptive, not the final test.'))
    gates = {}; tuning = {}
    for strategy in STRATEGIES:
        inference = comparisons[strategy]['mixture']
        sc = [c for c in cells if c['strategy']==strategy]
        gpu_cells = sum(b in CATALOG for b in policies.get(strategy, {}).get('routes', {}).values())
        passed = bool(protocol_complete and all_correct and gpu_cells and inference and inference['ratio']>=1.10
            and inference['ci95'][0]>1 and len(sc)==12 and all(c['ratio']>=1/1.05 for c in sc) and not replay)
        gates[strategy] = dict(gpu_benefit=passed, gpu_cells=gpu_cells,
            worst_cell_slowdown=max((1/c['ratio'] for c in sc), default=None))
        seconds = sum(c.get('tuning_seconds', 0) for c in choices if c['strategy']==strategy)
        if strategy=='hat': seconds += environment.get('loading_seconds', 0)
        saved_ms = inference['left_mean_request_ms']-inference['right_mean_request_ms'] if inference else None
        tuning[strategy] = dict(search_and_model_seconds=seconds,
            observed_mean_difference_ms=saved_ms,
            amortization_requests=math.ceil(seconds*1000/saved_ms) if passed and saved_ms and saved_ms>0 else None)
    comparison = comparisons['hat_vs_coordinate']
    valid_advice = sum(c.get('credited_to_hat', False) for c in choices if c['strategy']=='hat')
    different = policies.get('hat', {}).get('routes') != policies.get('coordinate', {}).get('routes')
    hat_benefit = bool(protocol_complete and all_correct and different and valid_advice and comparison
        and comparison['ratio']>=1.10 and comparison['ci95'][0]>1 and not replay)
    saved = comparison['left_mean_request_ms']-comparison['right_mean_request_ms'] if comparison else 0
    extra = max(0, tuning['hat']['search_and_model_seconds']-tuning['coordinate']['search_and_model_seconds'])
    concurrency_summary = {}
    by_condition = {}
    for r in concurrent: by_condition.setdefault(r['condition'], {})[r['repeat']] = r
    for backend in ('cpu', 'gpu'):
        selected = {r['condition']: {} for r in concurrent}
        for r in concurrent: selected[r['condition']][r['repeat']] = r
        serial, overlap = selected.get(backend+'_serial', {}), selected.get(backend+'_overlap', {})
        if set(serial)==set(range(6)) and set(overlap)==set(range(6)):
            valid = all(r['numerical']['correct'] and r['adviser']['choice'] is not None for r in (*serial.values(), *overlap.values()))
            interval = paired_interval([(serial[i]['total_ms'], overlap[i]['total_ms']) for i in range(6)], SEED)
            interval['paired_repetitions'] = interval.pop('independent_seeds')
            concurrency_summary[backend] = dict(**interval, outputs_valid=valid,
                mean_overlap_ms=statistics.mean(r['overlap_ms'] for r in overlap.values()),
                serial_total_ms=statistics.mean(r['total_ms'] for r in serial.values()),
                overlap_total_ms=statistics.mean(r['total_ms'] for r in overlap.values()),
                hat_serial_ms=statistics.mean(r['adviser']['total_ms'] for r in serial.values()),
                hat_overlap_ms=statistics.mean(r['adviser']['total_ms'] for r in overlap.values()))
    cross_backend = {}
    for mode in ('serial', 'overlap'):
        cpu, gpu = by_condition.get('cpu_'+mode, {}), by_condition.get('gpu_'+mode, {})
        if set(cpu)==set(gpu)==set(range(6)):
            cross_backend[mode] = paired_interval([(cpu[i]['total_ms'],gpu[i]['total_ms']) for i in range(6)], SEED)
            cross_backend[mode]['paired_repetitions'] = cross_backend[mode].pop('independent_seeds')
    return dict(complete=protocol_complete, replay=replay, coverage_exact=coverage,
        planned_measurements=len(expected), observed_measurements=len(measures),
        missing_measurements=len(expected-set(actual)), unexpected_measurements=len(set(actual)-expected),
        duplicate_measurements=sum(v-1 for v in actual.values()), concurrency_coverage=concurrency_coverage,
        observed_concurrency=len(concurrent), all_observed_correct=all_correct,
        cells=cells, development=development, comparisons=comparisons, acceptance=gates, tuning=tuning,
        valid_hat_proposals=valid_advice, hat_policy_benefit=hat_benefit,
        policies_identical_to_baseline=identical_routes,
        adviser_generations=sum('adviser' in c for c in choices)+len(concurrent),
        valid_concurrency_advice=sum(r['adviser']['choice'] is not None for r in concurrent),
        hat_incremental_amortization_requests=math.ceil(extra*1000/saved) if hat_benefit and saved>0 else None,
        concurrency=concurrency_summary, concurrency_cpu_over_gpu=cross_backend,
        statistical_unit='Per-input seed median over repetitions. Mixture bootstrap resamples eight seed-index bundles, each containing one independent seed per cell. Concurrency intervals resample six paired repetitions of the same fixed jobs, not six independent input datasets. Exploratory intervals; no population-wide guarantee.')


def build_report(directory, charts=True):
    read = lambda name, default: json.loads((directory/name).read_text()) if (directory/name).exists() else default
    rows, malformed = read_jsonl(directory/'feedback.jsonl')
    outcome = read('outcome.json', {}); policies = read('policies.json', {}); choices = read('choices.json', [])
    summary = analyze(rows, outcome, policies, choices, read('feedback-environment.json', {}),
                      read('correctness.json', {}), bool(read('config.json', {}).get('replay_run')))
    if malformed:
        summary.update(complete=False, damaged_rows=malformed, hat_policy_benefit=False)
        for result in summary['acceptance'].values(): result['gpu_benefit']=False
    telemetry, telemetry_errors = read_jsonl(directory/'telemetry.jsonl')
    prior_telemetry, _ = read_jsonl(directory/'resume-source/telemetry.jsonl')
    telemetry = prior_telemetry+telemetry
    continuation = read('resume-provenance.json', {})
    external_seconds = read('execution-budget.json', {}).get('earlier_stopped_attempt_seconds', 0)
    summary['total_supervised_seconds'] = outcome.get('elapsed_seconds',0)+continuation.get('prior_elapsed_seconds',0)+external_seconds
    summary['continuation'] = continuation
    summary['thermal'] = {k: max((r[k] for r in telemetry if r.get(k) is not None), default=None)
                          for k in ('cpu_temp_c', 'hat_max_c')}
    summary['outcome'] = outcome
    summary['measurement_stages'] = dict(Counter(r['stage'] for r in rows if r['event']=='measurement'))
    atomic_json(directory/'summary.json', summary)
    write_csv(directory/'measurements.csv', [r for r in rows if r['event']=='measurement'])
    write_csv(directory/'final-cells.csv', summary['cells'])
    write_csv(directory/'development-candidates.csv', summary['development'])
    write_csv(directory/'decisions.csv', [r for r in rows if r['event']=='decision'])
    sections = ['<h1>CPU, VideoCore GPU and AI HAT+ 2: bounded attention feedback</h1>',
        f'<p>Completed: {summary["complete"]}. Numerical requests: {summary["observed_measurements"]}/{summary["planned_measurements"]}. '
        f'Concurrency conditions: {summary["observed_concurrency"]}/24. All observed numerical outputs correct: {summary["all_observed_correct"]}.</p>',
        '<p>Original ZIP causal attention with fixed FP32 inputs and weights, verified against its FP64 cached reference. '
        'Streaming exposes one step at a time; prefill exposes the complete causally masked sequence. The HAT selects fixed '
        'Vulkan configurations; it does not execute these attention tensors or change model weights/source code.</p>',
        '<h2>Acceptance</h2><pre>'+html.escape(json.dumps(dict(gpu=summary['acceptance'],
            hat_policy_benefit=summary['hat_policy_benefit'], valid_hat_proposals=summary['valid_hat_proposals']), indent=2))+'</pre>',
        '<h2>Frozen policies on untouched inputs</h2>', table(['Cell', 'Strategy', 'Backend', 'CPU ms', 'Policy ms', 'Speed ratio', '95% interval'],
            [[r['cell_id'], r['strategy'], r['backend'], round(r['baseline_ms'],3), round(r['policy_ms'],3),
              round(r['ratio'],3), r['ci95']] for r in summary['cells']]),
        '<p>Policies identical to the baseline routes: '+html.escape(json.dumps(summary['policies_identical_to_baseline']))+
        '. Where routes are identical, ratios describe repeat-timing variability and are not an algorithmic speedup.</p>',
        '<h2>GPU candidates: development evidence</h2>', table(['Cell','Candidate','CPU ms','GPU ms','GPU / CPU time'],
            [[r['cell_id'],r['candidate'],round(r['cpu_ms'],3),round(r['gpu_ms'],3),round(r['gpu_over_cpu'],2)]
             for r in summary['development']]),
        '<p>These are coordinate-search development inputs, summarized by the mean of three seed medians. '
        'They describe candidate costs; they are not independent final-test evidence.</p>',
        '<h2>Tuning cost and overlap diagnostic</h2><pre>'+html.escape(json.dumps(dict(tuning=summary['tuning'],
            hat_incremental_amortization_requests=summary['hat_incremental_amortization_requests'],
            concurrency=summary['concurrency'], cpu_over_gpu_completion_ratio=summary['concurrency_cpu_over_gpu']), indent=2))+'</pre>',
        '<p>Amortization uses the measured equal-frequency mixture (one request per cell), including model loading for HAT tuning. '
        'Common fixture preparation, correctness checks, baseline selection, compiler/setup and offline analysis are separate costs. '
        'A finite estimate alone is not an acceptance result. No energy measurements were made.</p>',
        '<h2>Methods and limits</h2><p>Eight fixed GPU configurations; three candidate trials per search strategy. '
        'Each development cell uses three seeds × three repetitions; each round validates on three fresh seeds × five repetitions. '
        'Promotion requires correct results, at least 1.10× speed ratio and a paired 95% lower bound above 1. '
        'Final testing uses eight untouched seeds × five repetitions per cell. GPU acceptance also requires no cell more than 5% slower. '
        'Final results cannot change the frozen routes. Invalid model answers use a logged deterministic fallback without model credit.</p>',
        '<p>Primary request time includes CPU/GPU computation, weight/input staging, synchronization and output retrieval. '
        'Persistent native buffer allocation and initialization are separate; NumPy retains its ordinary allocation behavior. '
        'GPU timestamps are diagnostic and GPU utilization is unavailable. Overlap is measured on host intervals and does not '
        'prove simultaneous device execution. Concurrency forces the fastest verified development GPU configuration even when '
        'the attention policy retains CPU. The GPU shares the live desktop; background graphics were not disabled. '
        'It is not an end-to-end Qwen acceleration measurement.</p>',
        '<p>'+html.escape(summary['statistical_unit'])+'</p>',
        '<p>Arbitrary large-magnitude inputs are outside the tested accuracy domain. An earlier amplitude-stress check '
        'exceeded the FP32 NumPy tolerance before benchmarking; the preserved follow-up uses distribution-preserving '
        'sign changes for causality and isolation. The numerical tolerance was not relaxed.</p>',
        '<h2>Execution segments</h2><pre>'+html.escape(json.dumps(dict(total_supervised_seconds=summary['total_supervised_seconds'],
            continuation=continuation), indent=2))+'</pre>',
        '<pre>'+html.escape(json.dumps(dict(thermal=summary['thermal'], outcome=outcome,
            coverage=summary['measurement_stages'], damaged_telemetry=telemetry_errors), indent=2))+'</pre>']
    if charts and summary['cells']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout='constrained')
        for ax, strategy in zip(axes, STRATEGIES):
            data = [r for r in summary['cells'] if r['strategy']==strategy]
            xs = list(range(len(data))); ys = [r['ratio'] for r in data]
            ax.errorbar(xs, ys, yerr=[[r['ratio']-r['ci95'][0] for r in data],
                [r['ci95'][1]-r['ratio'] for r in data]], fmt='o', capsize=3)
            ax.axhline(1, color='gray'); ax.axhline(1.10, color='green', linestyle='--')
            title = strategy+(' — identical CPU routes; repeat-timing control' if summary['policies_identical_to_baseline'][strategy] else '')
            ax.set(xticks=xs, xticklabels=[r['cell_id'] for r in data], ylabel='CPU / policy request time', title=title)
            ax.tick_params(axis='x', rotation=30)
        for extension in ('png', 'svg'): fig.savefig(directory/f'feedback-results.{extension}', dpi=160)
        plt.close(fig)
        encoded = base64.b64encode((directory/'feedback-results.png').read_bytes()).decode()
        sections.append(f'<img alt="Final per-cell speed ratios" src="data:image/png;base64,{encoded}">')
    if charts and summary['development']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,1,figsize=(12,8),layout='constrained')
        candidates = sorted({r['candidate'] for r in summary['development']})
        for ax, mode in zip(axes, ('stream','prefill')):
            cells = [c['cell_id'] for c in CELLS if c['mode']==mode]
            width = .8/len(candidates)
            for index,candidate in enumerate(candidates):
                by_cell = {r['cell_id']:r for r in summary['development'] if r['candidate']==candidate and r['mode']==mode}
                ax.bar([i-.4+width*(index+.5) for i,c in enumerate(cells) if c in by_cell],
                       [by_cell[c]['gpu_over_cpu'] for c in cells if c in by_cell],width=width,label=candidate)
            ax.axhline(1,color='black',linestyle='--',label='CPU incumbent')
            ax.set(xticks=range(len(cells)),xticklabels=cells,yscale='log',ylabel='GPU / CPU request time (log)',
                   title=mode+' — development results; lower is better')
            ax.tick_params(axis='x',rotation=20); ax.legend(ncol=4)
        for extension in ('png','svg'): fig.savefig(directory/f'gpu-development.{extension}',dpi=160)
        plt.close(fig)
        encoded = base64.b64encode((directory/'gpu-development.png').read_bytes()).decode()
        sections.append(f'<img alt="GPU candidate costs compared with CPU incumbents" src="data:image/png;base64,{encoded}">')
    css = 'body{font:16px/1.5 system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#192435}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}img{width:100%}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Attention feedback experiment</title><style>'+css+'</style>'+''.join(sections)+'</html>')
    config = read('config.json', {})
    atomic_json(directory/'reproduce.json', dict(evidence='Replay of frozen decisions, not a new adaptive search',
        argv=['python3', str(directory.parent.parent/'experiment.py'), 'improve', '--source-run', config.get('source_run', ''),
              '--replay-run', str(directory), '--output', str(directory.parent/(directory.name+'-replay')), '--max-seconds', '3600'],
        on_stage_timeout_argv=['python3', str(directory.parent.parent/'experiment.py'), 'improve', '--source-run', config.get('source_run', ''),
              '--resume-run', str(directory.parent/(directory.name+'-replay')), '--output', str(directory.parent/(directory.name+'-replay-continuation'))]))
    analysis_source = directory/'analysis-source'; analysis_source.mkdir(exist_ok=True)
    for name in ('feedback_report.py','feedback_audit.py'):
        shutil.copy2(ROOT/'efficiency'/name, analysis_source/name)
    atomic_json(directory/'checksums.json', {str(p.relative_to(directory)): digest_file(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and p != directory/'checksums.json'})
    print(json.dumps(dict(report=str(directory/'report.html'), complete=summary['complete'], acceptance=summary['acceptance']), indent=2))
