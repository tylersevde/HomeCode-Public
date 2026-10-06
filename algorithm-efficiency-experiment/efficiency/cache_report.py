"""Accuracy-gated latency analysis with independent tables as bootstrap units."""
import base64
from collections import Counter, defaultdict
import html
import json
import shlex
import statistics

from .cache_validation import ARMS, PRESETS
from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .diagnostic import LOGICAL_END
from .diagnostic_report import quality
from .report import bootstrap_median, median, table, write_csv


def compare(a, b, **labels):
    reason = ('missing_arm' if a is None or b is None else
        'different_recorded_inputs' if a['effective_input_sha256']!=b['effective_input_sha256'] else
        'different_parameters' if a['effective_parameters']!=b['effective_parameters'] else
        'invalid_context_ledger' if not (a['token_ledger_valid'] and b['token_ledger_valid']) else
        'incomplete_generation' if any(r['completion_status']!=LOGICAL_END for r in (a,b)) else
        'no_visible_output' if any(r['first_visible_ms'] is None for r in (a,b)) else None)
    result = dict(labels,eligible=reason is None,exclusion_reason=reason,
        outputs_equal=a['output']==b['output'] if a and b else None,
        both_strict_correct=bool(a and b and a['strict_correct'] and b['strict_correct']))
    if reason is None:
        result.update(rebuild_request_ms=a['request_ms'],retain_request_ms=b['request_ms'],
            rebuild_first_visible_ms=a['first_visible_ms'],retain_first_visible_ms=b['first_visible_ms'],
            request_speedup=a['request_ms']/b['request_ms'],
            first_visible_speedup=a['first_visible_ms']/b['first_visible_ms'])
    return result


def table_ratios(comparisons, jobs, turns):
    """A table enters inference only with complete follow-ups in every repeat."""
    by_job = defaultdict(list)
    for row in comparisons:
        if row['turn']>0:
            by_job[row['job_id']].append(row)
    by_fixture = defaultdict(list)
    for job in jobs:
        by_fixture[job['fixture_id']].append(job)
    tables = []
    for fixture,jobs_for_fixture in sorted(by_fixture.items()):
        request, first = [], []
        for job in jobs_for_fixture:
            group = by_job[job['job_id']]
            if len(group)!=turns-1 or not all(r['eligible'] for r in group):
                break
            request.append(sum(r['rebuild_request_ms'] for r in group)/sum(r['retain_request_ms'] for r in group))
            first.append(sum(r['rebuild_first_visible_ms'] for r in group)/sum(r['retain_first_visible_ms'] for r in group))
        if len(request)==len(jobs_for_fixture):
            tables.append(dict(fixture_id=fixture,repeats=len(request),request_ratios_by_repeat=request,
                first_visible_ratios_by_repeat=first,request_ratio=statistics.median(request),
                first_visible_ratio=statistics.median(first)))
    return tables


def analyze(rows, schedule, config, protocol_ok):
    observed = [r for r in rows if r['event']=='response' and r['phase']=='main']
    indexed = {(r['job_id'],r['arm'],r['turn']):r for r in observed}
    expected = {(j['job_id'],a,t) for j in schedule['main'] for a in ARMS for t in range(j['turns'])}
    planned_total = len(config['hat_sizes'])*config['tables_per_size']*config['cache_repeats']*2*2*config['turns']
    skipped = [r for r in rows if r['event']=='turn_skipped' and r['phase']=='main']
    skipped_keys = {(r['job_id'],r['arm'],r['turn']) for r in skipped}
    duplicate_count = len(observed)-len(indexed)
    coverage = set(indexed)==expected and len(expected)==planned_total and duplicate_count==0
    comparisons = [compare(indexed.get((j['job_id'],'rebuild',t)),indexed.get((j['job_id'],'retain',t)),
        job_id=j['job_id'],fixture_id=j['fixture_id'],target_tokens=j['target_tokens'],preset=j['preset'],
        repeat=j['repeat'],turn=t) for j in schedule['main'] for t in range(j['turns'])]
    groups, all_tables = [], []
    for size in config['hat_sizes']:
        for preset in PRESETS:
            jobs = [j for j in schedule['main'] if j['target_tokens']==size and j['preset']==preset]
            measured = [r for r in observed if r['target_tokens']==size and r['preset']==preset]
            paired = [r for r in comparisons if r['target_tokens']==size and r['preset']==preset]
            eligible = [r for r in paired if r['eligible']]
            tables = table_ratios(paired,jobs,config['turns'])
            for row in tables:
                row.update(target_tokens=size,preset=preset)
            all_tables.extend(tables)
            ratios = [r['request_ratio'] for r in tables]
            ci = bootstrap_median(ratios,seed=config['schedule_seed'])
            first_ci = bootstrap_median([r['first_visible_ratio'] for r in tables],seed=config['schedule_seed'])
            expected_per_arm = config['tables_per_size']*config['cache_repeats']*config['turns']
            accuracy = {a:dict(quality([r for r in measured if r['arm']==a]),planned=expected_per_arm,
                skipped=sum(r['target_tokens']==size and r['preset']==preset and r['arm']==a for r in skipped)) for a in ARMS}
            all_correct = all(q['strict_correct']==q['planned'] and q['total']==q['planned'] for q in accuracy.values())
            matches = len(eligible)==expected_per_arm and len(paired)==expected_per_arm
            repeats_complete = len(tables)==config['tables_per_size'] and all(t['repeats']==config['cache_repeats'] for t in tables)
            useful = bool(protocol_ok and all_correct and matches and repeats_complete and ci and ci[0]>1)
            sessions = [r for r in rows if r['event']=='session_complete' and r['phase']=='main'
                        and r['target_tokens']==size and r['preset']==preset and r['completed_turns']==config['turns']]
            groups.append(dict(target_tokens=size,preset=preset,accuracy=accuracy,all_answers_strictly_correct=all_correct,
                planned_paired_turns=expected_per_arm,observed_paired_turns=sum(all((j['job_id'],a,t) in indexed for a in ARMS)
                    for j in jobs for t in range(j['turns'])),
                eligible_paired_turns=len(eligible),matched_coverage_complete=matches,
                exact_output_matches=sum(r['outputs_equal'] is True for r in paired),
                exclusions=dict(Counter(r['exclusion_reason'] for r in paired if not r['eligible'])),
                complete_independent_tables=len(tables),planned_independent_tables=config['tables_per_size'],
                followup_request_speedup=median(ratios),request_speedup_ci95=ci,
                followup_first_visible_speedup=median([r['first_visible_ratio'] for r in tables]),
                first_visible_speedup_ci95=first_ci,
                first_turn_control_speedup=median([r['request_speedup'] for r in eligible if r['turn']==0]),
                followup_request_median_ms={a:median([r[f'{a}_request_ms'] for r in eligible if r['turn']>0]) for a in ARMS},
                conversation_request_median_ms={a:median([r['request_total_ms'] for r in sessions if r['arm']==a]) for a in ARMS},
                conversation_wall_median_ms={a:median([r['session_wall_ms'] for r in sessions if r['arm']==a]) for a in ARMS},
                useful_improvement=useful))
    repeat_quality=[]
    for size in config['hat_sizes']:
        for preset in PRESETS:
            for repeat in range(config['cache_repeats']):
                for arm in ARMS:
                    group=[r for r in observed if (r['target_tokens'],r['preset'],r['repeat'],r['arm'])==(size,preset,repeat,arm)]
                    repeat_quality.append(dict(target_tokens=size,preset=preset,repeat=repeat,arm=arm,
                        **quality(group),request_median_ms=median([r['request_ms'] for r in group])))
    position_quality=[]
    for size in config['hat_sizes']:
        for preset in PRESETS:
            for arm in ARMS:
                for turn in range(config['turns']):
                    group=[r for r in observed if (r['target_tokens'],r['preset'],r['arm'],r['turn'])==(size,preset,arm,turn)]
                    position_quality.append(dict(target_tokens=size,preset=preset,arm=arm,turn=turn,
                        planned=config['tables_per_size']*config['cache_repeats'],**quality(group)))
    order_timing=[]
    for size in config['hat_sizes']:
        for preset in PRESETS:
            for arm in ARMS:
                for order in (0,1):
                    group=[r for r in observed if r['turn']>0 and (r['target_tokens'],r['preset'],r['arm'],r['arm_order'])==(size,preset,arm,order)]
                    order_timing.append(dict(target_tokens=size,preset=preset,arm=arm,arm_order=order,
                        observed=len(group),request_median_ms=median([r['request_ms'] for r in group])))
    return dict(expected_responses=planned_total,observed_responses=len(observed),coverage_exact=coverage,
        planned_keys_accounted_for=(set(indexed)|skipped_keys)==expected and len(expected)==planned_total,
        duplicate_responses=duplicate_count,skipped_responses=skipped,
        context_mismatches=sum(not r['token_ledger_valid'] for r in observed),
        incomplete_responses=sum(r['completion_status']!=LOGICAL_END for r in observed),
        groups=groups,repeat_quality=repeat_quality,position_quality=position_quality,order_timing=order_timing,
        quality=quality(observed)),comparisons,all_tables


def build_report(directory, charts=True):
    def read(name, default=None):
        return json.loads((directory/name).read_text()) if (directory/name).exists() else default
    config=read('config.json')
    schedule=read('schedule.json',dict(main=[]))
    rows,errors=read_jsonl(directory/'cache_validation.jsonl')
    telemetry,telemetry_errors=read_jsonl(directory/'telemetry.jsonl')
    outcome=read('outcome.json',dict(status='incomplete'))
    control=read('control-check.json',dict(passed=False))
    protocol_ok=outcome['status']=='complete' and control['passed'] and not errors and not telemetry_errors and any(r['event']=='complete' for r in rows)
    summary,comparisons,tables=analyze(rows,schedule,config,protocol_ok)
    summary.update(generated_utc=utc(),outcome=outcome,control_check=control,environment=read('cache-environment.json',{}),
        fresh_evidence=not bool(config.get('replay_run')),
        protocol_completed=protocol_ok,complete=protocol_ok and summary['coverage_exact'],
        damaged_lines=dict(observations=errors,telemetry=telemetry_errors),
        thermal=dict(max_cpu_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None),default=None),
            max_hat_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None),default=None),
            throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})),
        inference_policy='Strict correctness for all 80 planned answers per arm, complete matched coverage and a table-clustered exploratory 95% interval above 1. Repetitions are combined within each independent table. No universal model-accuracy claim.')
    atomic_json(directory/'summary.json',summary)
    write_csv(directory/'cache-responses.csv',[r for r in rows if r['event']=='response'])
    write_csv(directory/'paired-comparisons.csv',comparisons)
    write_csv(directory/'table-speedups.csv',tables)
    write_csv(directory/'quality-by-turn.csv',summary['position_quality'])
    write_csv(directory/'quality-by-repeat.csv',summary['repeat_quality'])
    write_csv(directory/'timing-by-order.csv',summary['order_timing'])
    write_csv(directory/'sessions.csv',[r for r in rows if r['event']=='session_complete'])
    write_csv(directory/'telemetry.csv',telemetry)
    atomic_json(directory/'reproduce.json',dict(command=shlex.join(['python3',str(ROOT/'experiment.py'),
        'run','--phase','hat','--profile','cache-validation','--source-run',config['source_run'],
        '--replay-run',str(directory),
        '--output',str(directory.parent/(directory.name+'-replay')),'--max-seconds','3600']),
        note='Explicit replay: verifies saved artifacts and requires identical regenerated fixtures, schedule and settings. Repetition of existing tables, not fresh held-out evidence.'))
    plots=[]
    if charts:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
        labels=[f'{g["target_tokens"]}\n{g["preset"]}' for g in summary['groups']]
        xs=list(range(len(labels)))
        for arm,offset in [('rebuild',-.18),('retain',.18)]:
            ys=[100*g['accuracy'][arm]['strict_correct']/g['accuracy'][arm]['planned'] for g in summary['groups']]
            axes[0].bar([x+offset for x in xs],ys,width=.36,label=arm)
        axes[0].set(xticks=xs,xticklabels=labels,ylabel='Strict correct / planned (%)',ylim=(0,105),title='Accuracy includes missing responses')
        axes[0].tick_params(axis='x',labelrotation=35)
        axes[0].legend()
        for x,g in zip(xs,summary['groups']):
            value=g['followup_request_speedup']; ci=g['request_speedup_ci95']
            if value is not None:
                axes[1].plot(x,value,'o',color='#167c80')
                if ci: axes[1].vlines(x,*ci,color='#167c80')
        axes[1].axhline(1,color='#64748b',linestyle='--')
        axes[1].set(xticks=xs,xticklabels=labels,ylabel='Rebuild / retain request time',title='Follow-up speedup; 95% table bootstrap')
        axes[1].tick_params(axis='x',labelrotation=35)
        for ext in ('png','svg'):fig.savefig(directory/f'cache-results.{ext}',dpi=160)
        plt.close(fig)
        plots.append('cache-results')
        if telemetry:
            fig,axis=plt.subplots(figsize=(9,3.5),layout='constrained')
            for key,label in [('cpu_temp_c','Pi CPU'),('hat_max_c','HAT maximum')]:
                valid=[r for r in telemetry if r.get(key) is not None]
                axis.plot([r['elapsed_seconds']/60 for r in valid],[r[key] for r in valid],label=label)
            axis.set(xlabel='Elapsed minutes',ylabel='Temperature (°C)',title='Observed temperatures')
            axis.legend()
            for ext in ('png','svg'):fig.savefig(directory/f'temperatures.{ext}',dpi=160)
            plt.close(fig);plots.append('temperatures')
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    sections=['<h1>Fresh-workload validation of context reuse</h1>',
        f'<p>Outcome: {html.escape(outcome["status"])}. Protocol completed: {summary["protocol_completed"]}. '
        f'Measured {summary["observed_responses"]}/{summary["expected_responses"]} main responses; '
        f'{len(summary["skipped_responses"])} dependent turns skipped. Complete measurement coverage: {summary["complete"]}.</p>',
        f'<p>Fresh evidence: {summary["fresh_evidence"]}. Thirty tables, ten per size, two repeats, two explicit generation settings and two context methods. '
        'Every conversation applies its setting throughout all four turns and uses its own actual previous answers. '
        'Eight saved-case control responses precede the main matrix and are excluded from inference.</p>',
        '<h2>Accuracy and useful improvement</h2>',table(['Tokens','Setting','Strict correct / planned: rebuild','Strict correct / planned: retain','Eligible turns / planned','Independent tables','Request speedup [95% CI]','Useful improvement'],
            [[g['target_tokens'],g['preset'],*[f'{g["accuracy"][a]["strict_correct"]}/{g["accuracy"][a]["planned"]}' for a in ARMS],
              f'{g["eligible_paired_turns"]}/{g["planned_paired_turns"]}',g['complete_independent_tables'],
              fmt(g['followup_request_speedup'])+(' ['+', '.join(fmt(x) for x in g['request_speedup_ci95'])+']' if g['request_speedup_ci95'] else ''),
              'Yes' if g['useful_improvement'] else 'Not established'] for g in summary['groups']]),
        '<p>A useful improvement requires complete coverage, every planned answer strictly correct in both methods, matched recorded histories, '
        'and an exploratory 95% interval for complete-response speedup entirely above 1. Faster incorrect answers do not satisfy this standard.</p>',
        '<h2>Timing and statistical units</h2><p>The primary continuous request timer includes real prompt rendering and suffix construction, '
        'required rebuilding clear, generation, stream reads and generator cleanup. Prevalidation runs before timing; the timed preparation must match it. '
        'Context/token diagnostic RPCs, scoring and logging are outside this interval. Verified session setup and model loading are separate. '
        'Conversation request totals sum timed requests; conversation wall time additionally includes instrumentation.</p>'
        '<p>For each repetition, sum the three follow-up request times in each arm and form their ratio. Take the median across repetitions within each table, '
        'then the median across tables. The 10,000-sample bootstrap resamples independent tables, never individual turns or repetitions. '
        'A table requires all follow-ups from both repetitions to enter timing inference. Accuracy retains all planned answers as its denominator.</p>',
        table(['Tokens','Setting','First-turn control ratio','First-visible follow-up speedup','Conversation request ms: rebuild / retain','Conversation wall ms: rebuild / retain','Exclusions'],
            [[g['target_tokens'],g['preset'],fmt(g['first_turn_control_speedup']),fmt(g['followup_first_visible_speedup']),
              ' / '.join(fmt(g['conversation_request_median_ms'][a]) for a in ARMS),
              ' / '.join(fmt(g['conversation_wall_median_ms'][a]) for a in ARMS),g['exclusions']] for g in summary['groups']]),
        '<h2>Settings and supervision</h2><pre>'+html.escape(json.dumps(dict(settings=summary['environment'].get('settings'),
            controls=control,thermal=summary['thermal'],context_mismatches=summary['context_mismatches'],incomplete_responses=summary['incomplete_responses']),indent=2))+'</pre>',
        '<p>Raw prompts, stream chunks, settings, scores, context counts, paired exclusions, table-level speedups, turn/repetition quality, '
        'order timing, telemetry, provenance, executed sources and checksums accompany this report. Matching re-encoded text and runtime counts '
        'does not expose the actual generated token IDs. Power and energy are not measured. Findings apply to this selected workload and installed model.</p>']
    if outcome.get('stop_reason'):sections.append('<p>Stop reason: '+html.escape(outcome['stop_reason'])+'</p>')
    for stem in plots:
        data=base64.b64encode((directory/f'{stem}.png').read_bytes()).decode()
        sections.append(f'<figure><img alt="{stem}" src="data:image/png;base64,{data}"></figure>')
    css='body{font:16px/1.5 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}h2{margin-top:36px}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Useful context reuse</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    atomic_json(directory/'checksums.json',{str(p.relative_to(directory)):digest_file(p) for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'checksums.json'})
    print(json.dumps(dict(report=str(directory/'report.html'),complete=summary['complete'],responses=summary['observed_responses'],
        useful_improvement_groups=sum(g['useful_improvement'] for g in summary['groups'])),indent=2))
