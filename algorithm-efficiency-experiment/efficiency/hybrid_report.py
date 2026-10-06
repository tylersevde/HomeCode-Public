"""Offline accounting of verified answers, raw model failures, and workflow cost."""
import base64
from collections import Counter
import html
import json
import shlex
import statistics

from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .diagnostic import LOGICAL_END
from .hybrid import WORKFLOWS
from .report import bootstrap_median, median, table, write_csv


def analyze(rows, schedule, config, protocol_ok):
    observed = [r for r in rows if r['event']=='response' and r['phase']=='main']
    key = lambda r:(r['job_id'],r['workflow'],r['turn'])
    indexed = {key(r):r for r in observed}
    expected = {(j['job_id'],w,t) for j in schedule['main'] for w in WORKFLOWS for t in range(j['turns'])}
    planned = len(config['hat_sizes'])*config['tables_per_size']*config['cache_repeats']*config['turns']*3
    skipped = [r for r in rows if r['event']=='turn_skipped' and r['phase']=='main']
    duplicates = len(observed)-len(indexed)
    coverage = set(indexed)==expected and len(expected)==planned and not duplicates
    accounted = set(indexed)|{key(r) for r in skipped}
    groups, pairs, tables = [], [], []
    for size in config['hat_sizes']:
        quality = {}
        expected_per_workflow = config['tables_per_size']*config['cache_repeats']*config['turns']
        for workflow in WORKFLOWS:
            rs = [r for r in observed if (r['target_tokens'],r['workflow'])==(size,workflow)]
            raw = [r for r in rs if 'output' in r]
            quality[workflow] = dict(planned=expected_per_workflow, observed=len(rs),
                final_correct=sum(r['final_correct'] for r in rs),
                raw_strict_correct=sum(r['strict_correct'] for r in raw) if workflow!='cpu' else None,
                raw_factual_correct=sum(r['factual_correct'] is True for r in raw) if workflow!='cpu' else None,
                accepted_hat_answers=sum(r['answer_source']=='hat' for r in rs) if workflow=='hybrid' else None,
                fallbacks=sum(r['answer_source']=='cpu_fallback' for r in rs),
                fallback_reasons=dict(Counter(r['fallback_reason'] for r in rs if r.get('fallback_reason'))),
                request_median_ms=median([r['request_ms'] for r in rs]))
        jobs = [j for j in schedule['main'] if j['target_tokens']==size]
        local_pairs = []
        for job in jobs:
            for turn in range(config['turns']):
                group = {w:indexed.get((job['job_id'],w,turn)) for w in WORKFLOWS}
                reason = None
                if not all(group.values()):
                    reason = 'missing_workflow'
                elif len({(r['table_sha256'],r['queried_item'],r['expected_answer']) for r in group.values()})!=1:
                    reason = 'different_task'
                elif any(r.get('status')!='ok' or r['request_ms']<=0 for r in group.values()):
                    reason = 'invalid_request'
                elif not group['full_table']['token_ledger_valid'] or group['full_table']['completion_status']!=LOGICAL_END:
                    reason = 'invalid_full_table_generation'
                p = dict(job_id=job['job_id'],fixture_id=job['fixture_id'],target_tokens=size,
                    repeat=job['repeat'],turn=turn,eligible=reason is None,exclusion_reason=reason)
                if reason is None:
                    p.update({w+'_ms':r['request_ms'] for w,r in group.items()})
                local_pairs.append(p)
        pairs.extend(local_pairs)
        local_tables = []
        for fixture_id in sorted({j['fixture_id'] for j in jobs}):
            ratios, cpu_cost = [], []
            for repeat in range(config['cache_repeats']):
                ps = [p for p in local_pairs if p['fixture_id']==fixture_id and p['repeat']==repeat]
                if len(ps)!=config['turns'] or not all(p['eligible'] for p in ps):
                    break
                hybrid = sum(p['hybrid_ms'] for p in ps)
                ratios.append(sum(p['full_table_ms'] for p in ps)/hybrid)
                cpu_cost.append(hybrid/sum(p['cpu_ms'] for p in ps))
            if len(ratios)==config['cache_repeats']:
                local_tables.append(dict(fixture_id=fixture_id,target_tokens=size,
                    full_over_hybrid_by_repeat=ratios, hybrid_over_cpu_by_repeat=cpu_cost,
                    full_over_hybrid=statistics.median(ratios),hybrid_over_cpu=statistics.median(cpu_cost)))
        tables.extend(local_tables)
        ci = bootstrap_median([t['full_over_hybrid'] for t in local_tables],seed=config['schedule_seed'])
        cpu_ci = bootstrap_median([t['hybrid_over_cpu'] for t in local_tables],seed=config['schedule_seed'])
        verified = all(quality[w]['final_correct']==quality[w]['planned']==quality[w]['observed'] for w in ('cpu','hybrid'))
        groups.append(dict(target_tokens=size,quality=quality,complete_tables=len(local_tables),
            planned_tables=config['tables_per_size'],eligible_queries=sum(p['eligible'] for p in local_pairs),
            exclusions=dict(Counter(p['exclusion_reason'] for p in local_pairs if not p['eligible'])),
            full_over_hybrid=median([t['full_over_hybrid'] for t in local_tables]),full_over_hybrid_ci95=ci,
            hybrid_over_cpu=median([t['hybrid_over_cpu'] for t in local_tables]),hybrid_over_cpu_ci95=cpu_ci,
            verified_pipeline_complete=verified,
            faster_than_full_table=bool(protocol_ok and verified and len(local_tables)==config['tables_per_size'] and ci and ci[0]>1)))
    return dict(planned_answers=planned,observed_answers=len(observed),duplicate_responses=duplicates,
        coverage_exact=bool(coverage),planned_keys_accounted_for=accounted==expected and len(expected)==planned,
        skipped_responses=skipped,groups=groups,
        pipeline_passed=bool(protocol_ok and coverage and all(g['verified_pipeline_complete'] and
            g['complete_tables']==g['planned_tables'] for g in groups)),
        context_mismatches=sum(r.get('token_ledger_valid') is False for r in observed),
        main_generations=sum('output' in r for r in observed)),pairs,tables


def build_report(directory, charts=True):
    def read(name, default=None):
        return json.loads((directory/name).read_text()) if (directory/name).exists() else default
    config = read('config.json')
    rows, errors = read_jsonl(directory/'hybrid.jsonl')
    telemetry, telemetry_errors = read_jsonl(directory/'telemetry.jsonl')
    outcome = read('outcome.json',dict(status='incomplete'))
    control = read('control-check.json',dict(passed=False))
    query = config['profile']=='hybrid-query'
    protocol_ok = bool(outcome['status']=='complete' and (query or control['passed']) and not errors
        and not telemetry_errors and any(r['event']=='complete' for r in rows))
    if query:
        result = read('result.json')
        summary = dict(complete=protocol_ok and result is not None,result=result)
        pairs,tables = [],[]
    else:
        summary,pairs,tables = analyze(rows,read('schedule.json',dict(main=[])),config,protocol_ok)
        summary['complete'] = protocol_ok and summary['coverage_exact']
    summary.update(generated_utc=utc(),protocol_completed=protocol_ok,outcome=outcome,control_check=control,
        environment=read('hybrid-environment.json',{}),fresh_evidence=not bool(config.get('replay_run')),
        total_generations=sum(r['event']=='response' and 'output' in r for r in rows),
        damaged_lines=dict(observations=errors,telemetry=telemetry_errors),
        thermal=dict(max_cpu_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None),default=None),
            max_hat_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None),default=None),
            throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})))
    atomic_json(directory/'summary.json',summary)
    write_csv(directory/'responses.csv',[r for r in rows if r['event']=='response'])
    write_csv(directory/'workflow-comparisons.csv',pairs)
    write_csv(directory/'table-speedups.csv',tables)
    write_csv(directory/'indexes.csv',[r for r in rows if r['event']=='index_built'])
    write_csv(directory/'sessions.csv',[r for r in rows if r['event']=='session_complete'])
    write_csv(directory/'telemetry.csv',telemetry)
    if not query:
        command = ['python3',str(ROOT/'experiment.py'),'run','--phase','hat','--profile','hybrid-validation',
            '--source-run',config['source_run'],'--replay-run',str(directory),
            '--output',str(directory.parent/(directory.name+'-replay')),'--max-seconds','3600']
        atomic_json(directory/'reproduce.json',dict(command=shlex.join(command),
            note='Verified replay of the same tables and schedule; not fresh held-out evidence.'))
    plots = []
    if charts and not query:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        gs = summary['groups']; xs = list(range(len(gs)))
        for workflow,offset in [('cpu',-.25),('full_table',0),('hybrid',.25)]:
            q = [g['quality'][workflow] for g in gs]
            axes[0].bar([x+offset for x in xs],[100*v['final_correct']/v['planned'] for v in q],width=.24,label=workflow)
            axes[1].plot(xs,[v['request_median_ms'] for v in q],marker='o',label=workflow)
        axes[0].plot(xs,[100*g['quality']['hybrid']['raw_strict_correct']/g['quality']['hybrid']['planned'] for g in gs],
                     'kD--',label='hybrid raw HAT',markersize=5)
        for ax in axes:
            ax.set(xticks=xs,xticklabels=[g['target_tokens'] for g in gs],xlabel='Original full-table token target')
            ax.legend(fontsize=8)
        axes[0].set(ylabel='Correct / planned (%)',ylim=(0,105),title='Returned answers and raw model accuracy')
        axes[1].set(ylabel='Median complete request (ms, logarithmic)',yscale='log',title='Cost includes retrieval and validation')
        for ext in ('png','svg'):fig.savefig(directory/f'hybrid-results.{ext}',dpi=160)
        plt.close(fig);plots.append('hybrid-results')
        if telemetry:
            fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
            for key,label in [('cpu_temp_c','Pi CPU'),('hat_max_c','HAT maximum')]:
                valid=[r for r in telemetry if r.get(key) is not None]
                ax.plot([r['elapsed_seconds']/60 for r in valid],[r[key] for r in valid],label=label)
            ax.set(xlabel='Elapsed minutes',ylabel='Temperature (°C)',title='Observed temperatures');ax.legend()
            for ext in ('png','svg'):fig.savefig(directory/f'temperatures.{ext}',dpi=160)
            plt.close(fig);plots.append('temperatures')
    fmt = lambda x:'—' if x is None else f'{x:.4g}'
    sections = ['<h1>Verified Pi–HAT lookup pipeline</h1>',
        f'<p>Outcome: {html.escape(outcome["status"])}. Protocol completed: {protocol_ok}. Complete coverage: {summary["complete"]}.</p>']
    if query:
        sections.append('<pre>'+html.escape(json.dumps(summary['result'],indent=2))+'</pre>')
    else:
        sections.extend([f'<p>Measured {summary["observed_answers"]}/{summary["planned_answers"]} main answers; '
            f'{summary["main_generations"]} main HAT generations; pipeline correctness gate: {summary["pipeline_passed"]}. '
            f'Fresh evidence: {summary["fresh_evidence"]}.</p>',
            '<h2>Accuracy and fallback provenance</h2>',
            table(['Tokens','Workflow','Returned correct / planned','Raw HAT strict correct','Raw HAT factual correct','CPU fallbacks','Median request ms'],
                [[g['target_tokens'],w,f'{q["final_correct"]}/{q["planned"]}',q['raw_strict_correct'],q['raw_factual_correct'],q['fallbacks'],fmt(q['request_median_ms'])]
                 for g in summary['groups'] for w,q in g['quality'].items()]),
            '<p>CPU retrieval supplies the exact record. The HAT sees only that evidence in the hybrid workflow. '
            'A completed, strictly correct answer with a valid context ledger is accepted; other model answers are replaced '
            'with the verified CPU value and labelled cpu_fallback. Corrected answers are never counted as raw model successes. '
            'Direct CPU lookup already solves this exact task, so HAT participation is not itself an added capability.</p>',
            '<h2>Whole-workflow comparison</h2>',
            table(['Tokens','Independent tables','Full-table / hybrid time','95% interval','Hybrid / CPU time','95% interval','Hybrid faster than full-table'],
                [[g['target_tokens'],g['complete_tables'],fmt(g['full_over_hybrid']),g['full_over_hybrid_ci95'],
                  fmt(g['hybrid_over_cpu']),g['hybrid_over_cpu_ci95'],g['faster_than_full_table']] for g in summary['groups']]),
            '<p>These workflows answer the same questions using intentionally different prompts. This is not a cache-only comparison. '
            'Full-table answers may be incorrect; their accuracy is shown above. Every timed request includes preparation, '
            'required context reset, token/context checks, generation and cleanup, validation and any fallback recovery. '
            'Index construction, model loading and file logging are separate. CPU request timings are short Python operations '
            'and include timer/result-construction overhead; ratios describe this implementation, not an intrinsic CPU/NPU ratio.</p>',
            '<p>For each table and repetition, sum all four request times per workflow and take the ratio. '
            'Take the median across the two repetitions, then across tables. Exploratory 95% intervals use 10,000 independent-table '
            'bootstrap samples. A table needs complete eligible coverage in both repeats. Skips remain in accuracy denominators. '
            'The 12 saved baseline controls and eight color controls are excluded from main results.</p>',
            '<h2>Loading and indexing</h2><pre>'+html.escape(json.dumps(dict(
                model_loading_seconds=summary['environment'].get('loading_seconds'),
                index_build_median_ms=median([r['index_build_ms'] for r in rows if r['event']=='index_built']),
                whole_session_seconds=outcome.get('elapsed_seconds')),indent=2))+'</pre>'])
    sections.append('<h2>Supervision and controls</h2><pre>'+html.escape(json.dumps(dict(thermal=summary['thermal'],controls=control,
        stop_reason=outcome.get('stop_reason'),parameters=summary['environment'].get('parameters')),indent=2))+'</pre>')
    sections.append('<p>Reports preserve raw outputs, prompts, streaming chunks, record hashes, source provenance and executed code. '
                    'Runtime counts and re-tokenized text do not expose hidden generated token IDs. Power and energy are not measured.</p>')
    for stem in plots:
        data=base64.b64encode((directory/f'{stem}.png').read_bytes()).decode()
        sections.append(f'<figure><img alt="{stem}" src="data:image/png;base64,{data}"></figure>')
    css='body{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Verified lookup</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    atomic_json(directory/'checksums.json',{str(p.relative_to(directory)):digest_file(p) for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'checksums.json'})
    print(json.dumps(dict(report=str(directory/'report.html'),complete=summary['complete'],total_generations=summary['total_generations']),indent=2))
