"""Offline report with separate contract and unfamiliar-language evidence."""
import base64
import html
import json
import random
import shlex

from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .language_report import quality
from .report import table, write_csv
from .strict_protocol import ENGINES, STRATA, PROFILE


def family_bootstrap(families, seed, samples=10000):
    if len(families) < 2: return None
    def metrics(fs):
        times = {e: sum(f[e+'_ms'] for f in fs) for e in ENGINES}
        supported = sum(f['supported_planned'] for f in fs)
        values = dict(auto_over_strict_ms=times['auto']/times['strict'],
                      strict_over_cpu_ms=times['strict']/times['cpu'])
        if supported:
            resolved = {e: sum(f[e+'_resolved'] for f in fs) for e in ENGINES}
            values.update(strict_minus_cpu_coverage=(resolved['strict']-resolved['cpu'])/supported,
                          auto_minus_strict_coverage=(resolved['auto']-resolved['strict'])/supported)
        return values
    point = metrics(families); distributions = {k: [] for k in point}; rng = random.Random(seed)
    for _ in range(samples):
        values = metrics(rng.choices(families, k=len(families)))
        for k, v in values.items(): distributions[k].append(v)
    intervals = {}
    for k, values in distributions.items():
        values.sort(); intervals[k] = [values[int(.025*samples)], values[int(.975*samples)]]
    return dict(point=point, ci95=intervals, samples=samples, families=len(families), seed=seed)


def analyze(rows, corpus, schedule, config, regression, protocol_ok):
    observed = [r for r in rows if r['event']=='response' and r['phase']=='main']
    indexed = {(r['job_id'], r['engine']): r for r in observed}
    expected = {(j['job_id'], e) for j in schedule['main'] for e in j['engines']}
    repeats = config['language_repeats']; duplicates = len(observed)-len(indexed)
    complete = len(expected)==960 and set(indexed)==expected and duplicates==0
    groups = {e: quality([r for r in observed if r['engine']==e], 160*repeats, 120*repeats) for e in ENGINES}
    strata = {}
    for stratum in STRATA:
        cs = [c for c in corpus if c['stratum']==stratum]
        planned = (40 if stratum=='negative' else 60)*repeats
        strata[stratum] = {e: quality([r for r in observed if r['stratum']==stratum and r['engine']==e],
            planned, 0 if stratum=='negative' else planned) for e in ENGINES}
    categories = []
    for stratum in STRATA:
        for category in sorted({c['category'] for c in corpus if c['stratum']==stratum}):
            cs = [c for c in corpus if c['stratum']==stratum and c['category']==category]
            for e in ENGINES:
                rs = [r for r in observed if r['stratum']==stratum and r['category']==category and r['engine']==e]
                categories.append(dict(stratum=stratum, category=category, engine=e,
                    **quality(rs, len(cs)*repeats, sum(c['supported'] for c in cs)*repeats)))
    by_case = {c['case_id']: c for c in corpus}; pairs = []
    for job in schedule['main']:
        case = by_case[job['case_id']]
        rs = {e: indexed.get((job['job_id'], e)) for e in ENGINES}
        reason = None
        if not all(rs.values()): reason = 'missing_engine'
        elif len({(r['question'], r['table_sha256']) for r in rs.values()}) != 1: reason = 'different_task'
        elif any(r['request_ms'] <= 0 for r in rs.values()): reason = 'invalid_timing'
        pair = dict(job_id=job['job_id'], case_id=case['case_id'], family_id=case['family_id'],
                    stratum=case['stratum'], repeat=job['repeat'], supported=case['supported'],
                    eligible=reason is None, exclusion_reason=reason)
        if reason is None:
            for e, r in rs.items(): pair.update({e+'_ms': r['request_ms'], e+'_resolved': int(r['supported_resolved'])})
        pairs.append(pair)
    families = []
    for fid in sorted({c['family_id'] for c in corpus}):
        ps = [p for p in pairs if p['family_id']==fid]
        planned = sum(c['family_id']==fid for c in corpus)*repeats
        if len(ps)==planned and all(p['eligible'] for p in ps):
            families.append(dict(family_id=fid, stratum=ps[0]['stratum'], attempts_per_engine=planned,
                supported_planned=sum(p['supported'] for p in ps),
                **{e+'_ms': sum(p[e+'_ms'] for p in ps) for e in ENGINES},
                **{e+'_resolved': sum(p[e+'_resolved'] for p in ps) for e in ENGINES}))
    inference = {s: family_bootstrap([f for f in families if f['stratum']==s], config['schedule_seed']+i)
                 for i, s in enumerate(STRATA)}
    main_calls_match = all(groups[e]['hat_calls']==schedule.get('calls_by_engine', {}).get(e) for e in ENGINES)
    warmups = [r for r in rows if r['event']=='response' and r['phase']=='warmup']
    warmups_match = len(warmups)==6 and {r['warmup'] for r in warmups}==set(range(6)) and all(r['hat_called'] for r in warmups)
    mismatches = sum(r.get('token_ledger_valid') is False for r in (*observed, *warmups))
    gates = dict(complete_measurements=bool(complete and len(families)==40),
        regression_passed=bool(regression.get('passed')),
        documented_grammar_resolved=strata['contract']['strict']['supported_resolved']==60*repeats,
        all_negative_cases_rejected=strata['negative']['strict']['correct_abstentions']==40*repeats,
        no_incorrect_strict_acceptances=groups['strict']['incorrect_accepted']==0,
        planned_call_counts_match=main_calls_match and warmups_match,
        context_ledgers_valid=mismatches==0)
    return dict(planned_attempts=160*repeats*3, observed_attempts=len(observed), coverage_exact=bool(complete),
        duplicate_responses=duplicates, groups=groups, strata=strata, categories=categories, inference=inference,
        success_criteria=gates, strict_gate_passed=bool(protocol_ok and all(gates.values())),
        context_mismatches=mismatches,
        incorrect_acceptances=[dict(case_id=r['case_id'], stratum=r['stratum'], engine=r['engine'],
            repeat=r['repeat'], question=r['question'], expected_command=r['expected_command'],
            operation=r['operation'], argument=r['argument'], answer=r['answer']) for r in observed if r['incorrect_accepted']]), pairs, families


def build_report(directory, charts=True):
    def read(name, default=None):
        return json.loads((directory/name).read_text()) if (directory/name).exists() else default
    config = read('config.json'); rows, errors = read_jsonl(directory/'language_protocol.jsonl')
    telemetry, telemetry_errors = read_jsonl(directory/'telemetry.jsonl')
    outcome = read('outcome.json', dict(status='incomplete'))
    ok = outcome['status']=='complete' and not errors and not telemetry_errors and any(r['event']=='complete' for r in rows)
    regression = read('regression.json', {})
    summary, pairs, families = analyze(rows, read('corpus.json', []), read('schedule.json', dict(main=[])), config, regression, ok)
    summary.update(complete=bool(ok and summary['coverage_exact']), protocol_completed=ok, generated_utc=utc(),
        outcome=outcome, environment=read('language-environment.json', {}),
        new_questions=not bool(config.get('replay_run')), regression={k: v for k, v in regression.items() if k!='cases'},
        total_generations=sum(r['event']=='response' and r['hat_called'] for r in rows),
        warmup_generations=sum(r['event']=='response' and r['phase']=='warmup' and r['hat_called'] for r in rows),
        damaged_lines=dict(observations=errors, telemetry=telemetry_errors),
        thermal=dict(max_cpu_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None), default=None),
            max_hat_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None), default=None),
            throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})))
    atomic_json(directory/'summary.json', summary)
    for name, values in [('responses', [r for r in rows if r['event']=='response']), ('paired-attempts', pairs),
                         ('family-totals', families), ('quality-by-category', summary['categories']),
                         ('indexes', [r for r in rows if r['event']=='index_built']), ('telemetry', telemetry)]:
        write_csv(directory/f'{name}.csv', values)
    atomic_json(directory/'reproduce.json', dict(command=shlex.join(['python3', str(ROOT/'experiment.py'), 'run',
        '--phase', 'hat', '--profile', PROFILE, '--source-run', config['source_run'], '--replay-run', str(directory),
        '--output', str(directory.parent/(directory.name+'-replay')), '--max-seconds', '3600']),
        note='Verified replay of frozen grammar, corpus, prompt, tables and schedule; not new evidence.'))
    plots = []
    if charts and summary['observed_attempts']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = ['#3973ac', '#27826d', '#d17b26']; xs = list(range(3))
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
        for ax, s, title in [(axes[0,0], 'contract', 'Documented grammar combinations'),
                             (axes[0,1], 'unfamiliar', 'Unfamiliar paraphrases')]:
            gs = [summary['strata'][s][e] for e in ENGINES]
            values = [100*g['supported_resolved']/g['supported_planned'] for g in gs]
            ax.bar(xs, values, color=colors)
            ax.set(ylabel='Correct supported queries / planned (%)', ylim=(0, 108), title=title)
            for x, v in zip(xs, values): ax.text(x, v+2, f'{v:.1f}%', ha='center')
        gs = [summary['groups'][e] for e in ENGINES]
        axes[1,0].bar(xs, [g['incorrect_accepted'] for g in gs], color=colors)
        axes[1,0].set(ylabel='Incorrect accepted intents', title='Errors across all strata')
        values = [g['request_median_ms'] for g in gs]
        axes[1,1].bar(xs, values, color=colors)
        axes[1,1].set(yscale='log', ylabel='Median request milliseconds (log scale)', title='Request cost; accuracy reported separately')
        for ax in axes.flat: ax.set(xticks=xs, xticklabels=ENGINES)
        for ext in ('png', 'svg'): fig.savefig(directory/f'strict-results.{ext}', dpi=160)
        plt.close(fig); plots.append('strict-results')
        if telemetry:
            fig, ax = plt.subplots(figsize=(9, 3.5), layout='constrained')
            for key, label in [('cpu_temp_c', 'Pi CPU'), ('hat_max_c', 'HAT maximum')]:
                valid = [r for r in telemetry if r.get(key) is not None]
                ax.plot([r['elapsed_seconds']/60 for r in valid], [r[key] for r in valid], label=label)
            ax.set(xlabel='Elapsed minutes', ylabel='Temperature (°C)', title='Observed temperatures'); ax.legend()
            for ext in ('png', 'svg'): fig.savefig(directory/f'temperatures.{ext}', dpi=160)
            plt.close(fig); plots.append('temperatures')
    fmt = lambda x: '—' if x is None else f'{x:.5g}'
    sections = ['<h1>Strict query handling and CPU/HAT comparison</h1>',
        f'<p>Protocol complete: {ok}. Exact attempt coverage: {summary["coverage_exact"]}. '
        f'Strict acceptance gate: {summary["strict_gate_passed"]}. '
        f'Attempts: {summary["observed_attempts"]}/{summary["planned_attempts"]}; HAT generations: {summary["total_generations"]} '
        f'including {summary["warmup_generations"]} warm-ups.</p>',
        '<p>Strict mode recognizes a closed, versioned grammar and abstains otherwise. The CPU baseline recognizes six forms; '
        'auto uses that baseline then the unchanged HAT interpreter. A CPU can execute a model command correctly even when '
        'that command misreads the question. Such an accepted interpretation is counted as an error.</p>']
    for s in STRATA:
        sections.extend([f'<h2>{s.capitalize()}</h2>',
            table(['Engine', 'Correct supported / planned', 'Required abstentions correct / planned', 'Incorrect accepted',
                   'Raw HAT intents correct / calls', 'HAT calls', 'Total request seconds', 'Median request ms'],
                [[e, f'{g["supported_resolved"]}/{g["supported_planned"]}',
                  f'{g["correct_abstentions"]}/{g["required_abstentions"]}', g['incorrect_accepted'],
                  f'{g["raw_intent_correct"]}/{g["hat_calls"]}', g['hat_calls'], fmt(g['request_total_ms']/1000),
                  fmt(g['request_median_ms'])] for e, g in summary['strata'][s].items()])])
    sections.extend(['<h2>Evidence and limitations</h2><p>The 160 old questions are regression evidence only. The new corpus has '
        '60 combinations of documented grammar, 60 unfamiliar paraphrases and 40 negative requests. Grammar combinations '
        'are deliberately derived from the contract, not independent evidence of general English understanding. Thirty fact '
        'tables are reused. Each question is measured twice; repetitions are not independent language examples. '
        'Grammar was frozen before corpus authoring; both were frozen before new-corpus evaluation. No outcome-based tuning '
        'or question selection is performed. Passing the gate means zero observed strict errors on this corpus, not universal reliability.</p>',
        '<h2>Acceptance and exploratory uncertainty</h2><pre>'+html.escape(json.dumps(dict(regression=summary['regression'],
            criteria=summary['success_criteria'], family_bootstrap_by_stratum=summary['inference']), indent=2))+'</pre>',
        '<p>Paired 10,000-sample bootstrap intervals resample whole template families within each stratum, retaining all argument '
        'variants and repetitions. Strata are not pooled for inference. Timing ratios compare the same requests but may compare '
        'answers with abstentions; they are not speedups at equal language coverage. Raw model correctness, accepted intent, '
        'final answers and abstentions are reported separately.</p>',
        '<h2>Accepted interpretation errors</h2>',
        table(['Case', 'Stratum', 'Engine', 'Repeat', 'Question', 'Expected', 'Interpreted'],
            [[r['case_id'], r['stratum'], r['engine'], r['repeat'], r['question'], r['expected_command'],
              dict(operation=r['operation'], argument=r['argument'])] for r in summary['incorrect_acceptances']]),
        '<h2>Timing and supervision</h2><p>Request timings include parsing/routing, model token checks and verified reset when '
        'applicable, generation/cleanup, validation and CPU execution. Index construction, model loading, file logging and '
        'whole CLI duration are separate. The interleaved experiment has a loaded model, while public strict queries start '
        'neither the model nor HAT sensors. Power and energy are not measured.</p><pre>'+html.escape(json.dumps(dict(
            thermal=summary['thermal'], parameters=summary['environment'].get('parameters'),
            loading_seconds=summary['environment'].get('loading_seconds'), session=outcome), indent=2))+'</pre>'])
    for stem in plots:
        data = base64.b64encode((directory/f'{stem}.png').read_bytes()).decode()
        sections.append(f'<figure><img alt="{stem}" src="data:image/png;base64,{data}"></figure>')
    css = 'body{font:16px/1.5 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Strict query validation</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    atomic_json(directory/'checksums.json', {str(p.relative_to(directory)): digest_file(p) for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'checksums.json'})
    print(json.dumps(dict(report=str(directory/'report.html'), complete=summary['complete'], strict_gate_passed=summary['strict_gate_passed']), indent=2))
