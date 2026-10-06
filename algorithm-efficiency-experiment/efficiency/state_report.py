"""Offline reporting for the context-state isolation experiment."""
import base64
from collections import Counter, defaultdict
import html
import json
import shlex
import statistics

from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .diagnostic import LOGICAL_END
from .diagnostic_report import quality
from .report import table, write_csv
from .state_isolation import MODES, PRESETS

CONTRASTS = (('rebuild','live'),('live','save_live'),('save_live','restore_same'),
             ('restore_same','restore_fresh'),('rebuild','rebuild_fresh'),('rebuild_fresh','restore_fresh'))


def compare(a, b, **labels):
    reason = ('missing_response' if a is None or b is None else
              'different_recorded_inputs' if a['effective_input_sha256'] != b['effective_input_sha256'] else
              'invalid_ledger' if not(a['token_ledger_valid'] and b['token_ledger_valid']) else
              'incomplete_generation' if any(r['completion_status'] != LOGICAL_END for r in (a,b)) else None)
    return dict(labels,eligible=reason is None,exclusion_reason=reason,
        outputs_equal=(a['output']==b['output']) if a and b else None,
        parameters_equal=(a['effective_parameters']==b['effective_parameters']) if a and b else None,
        left_output=a['output'] if a else None,right_output=b['output'] if b else None,
        left_factual_correct=a['factual_correct'] if a else None,
        right_factual_correct=b['factual_correct'] if b else None)


def analyze(rows, schedule):
    targets = [r for r in rows if r['event']=='response' and r['kind']=='target' and r['phase']=='main']
    conditioning = [r for r in rows if r['event']=='response' and r['kind']=='conditioning' and r['phase']=='main']
    indexed = {(r['case_id'],r['preset'],r['repeat'],r['mode']):r for r in targets}
    groups = defaultdict(list)
    for r in targets:
        groups[(r['case_id'],r['preset'],r['mode'])].append(r)
    counts = Counter((j['case_id'],j['preset'],m) for j in schedule['main'] for m in MODES)
    cells = []
    for key, planned in sorted(counts.items()):
        group = groups[key]
        cells.append(dict(case_id=key[0],preset=key[1],mode=key[2],planned=planned,**quality(group),
            outputs=dict(Counter(r['output'] for r in group)),
            ledger_mismatches=sum(not r['token_ledger_valid'] for r in group),
            median_request_ms=statistics.median(r['request_ms'] for r in group) if group else None))
    state_comparisons = []
    for j in schedule['main']:
        for left,right in CONTRASTS:
            state_comparisons.append(compare(indexed.get((j['case_id'],j['preset'],j['repeat'],left)),
                indexed.get((j['case_id'],j['preset'],j['repeat'],right)),
                case_id=j['case_id'],preset=j['preset'],repeat=j['repeat'],left_mode=left,right_mode=right))
    parameter_comparisons = []
    case_repeats = sorted({(j['case_id'],j['repeat']) for j in schedule['main']})
    for case,repeat in case_repeats:
        for mode in MODES:
            for left,right in [('implicit','explicit_defaults'),('explicit_defaults','penalty_1_0'),
                               ('explicit_defaults','penalty_1_2')]:
                parameter_comparisons.append(compare(indexed.get((case,left,repeat,mode)),indexed.get((case,right,repeat,mode)),
                    case_id=case,repeat=repeat,mode=mode,left_preset=left,right_preset=right))
    def summarize(comparisons, keys):
        grouped = defaultdict(list)
        for c in comparisons:
            grouped[tuple(c[k] for k in keys)].append(c)
        return [dict(zip(keys, key),planned=len(group),eligible=sum(c['eligible'] for c in group),
                     output_differences=sum(c['eligible'] and not c['outputs_equal'] for c in group),
                     equal_parameter_vectors=sum(c['eligible'] and c['parameters_equal'] for c in group))
                for key,group in sorted(grouped.items())]
    skips = [r for r in rows if r['event']=='target_skipped' and r['phase']=='main']
    expected = {(j['case_id'],j['preset'],j['repeat'],m) for j in schedule['main'] for m in MODES}
    skipped_keys = {(r['case_id'],r['preset'],r['repeat'],r['mode']) for r in skips}
    conditioning_keys={(r['job_id'],r['mode'],r['conditioning_turn']) for r in conditioning}
    expected_conditioning={(j['job_id'],m,t) for j in schedule['main']
                           for m in ('live','snapshot_donor') for t in range(j['conditioning_turns'])}
    return dict(planned=schedule['planned'],observed_targets=len(targets),observed_conditioning=len(conditioning),
        conditioning_coverage_exact=conditioning_keys==expected_conditioning and len(conditioning)==len(conditioning_keys),
        duplicate_targets=len(targets)-len(indexed),target_coverage_exact=set(indexed)==expected,
        planned_targets_accounted_for=set(indexed)|skipped_keys==expected,
        skipped_targets=skips,conditioning_failures=sum(r['conditioning_matches'] is not True for r in conditioning),
        context_mismatches=sum(not r['token_ledger_valid'] for r in targets+conditioning),
        quality=quality(targets),cells=cells,
        state_contrasts=summarize(state_comparisons,['preset','left_mode','right_mode']),
        parameter_contrasts=summarize(parameter_comparisons,['left_preset','right_preset']),
        state_exclusions=dict(Counter(c['exclusion_reason'] for c in state_comparisons if not c['eligible']))),state_comparisons,parameter_comparisons


def findings(summary):
    statements = []
    for r in summary['parameter_contrasts']:
        statements.append(f'{r["left_preset"]} versus {r["right_preset"]}: '
            f'{r["output_differences"]}/{r["eligible"]} eligible target comparisons changed output; '
            f'{r["equal_parameter_vectors"]} compared equal resolved parameter vectors.')
    for preset in PRESETS:
        contrasts = [r for r in summary['state_contrasts'] if r['preset']==preset]
        description = '; '.join(f'{r["left_mode"]} → {r["right_mode"]}: {r["output_differences"]}/{r["eligible"]}'
                                for r in contrasts)
        statements.append(f'{preset}: changed-output counts among eligible comparisons — {description}.')
    statements.append('Parameters change only for the target request. Every live conditioning history uses the original settings and must reproduce the frozen source. Penalty values 1.0 and 1.2 are sensitivity conditions; this experiment does not establish the penalty formula or assume either value is neutral.')
    statements.append('A difference after saving raises an observation-effect question; a difference after restoring raises a restoration or unsaved-state question. An error that follows a snapshot into a new process implicates serialized context or associated state. A new client process does not reboot the accelerator. None of these observations alone proves KV-cache corruption, a firmware defect, or a numerical mechanism.')
    return statements


def build_report(directory, charts=True):
    config = json.loads((directory/'config.json').read_text())
    schedule = json.loads((directory/'schedule.json').read_text()) if (directory/'schedule.json').exists() else dict(main=[],validation=[],planned={})
    rows, errors = read_jsonl(directory/'state_isolation.jsonl')
    telemetry, telemetry_errors = read_jsonl(directory/'telemetry.jsonl')
    outcome = json.loads((directory/'outcome.json').read_text()) if (directory/'outcome.json').exists() else dict(status='incomplete')
    env = json.loads((directory/'state-environment.json').read_text()) if (directory/'state-environment.json').exists() else {}
    capability = json.loads((directory/'capability-check.json').read_text()) if (directory/'capability-check.json').exists() else dict(passed=False)
    summary,states,parameters = analyze(rows,schedule)
    summary.update(generated_utc=utc(),outcome=outcome,capability=capability,environment=env,
        damaged_lines=dict(observations=errors,telemetry=telemetry_errors),
        complete=outcome['status']=='complete' and capability['passed'] and summary['target_coverage_exact']
            and summary['conditioning_coverage_exact'] and not summary['conditioning_failures']
            and not summary['duplicate_targets'] and not errors,
        protocol_completed=outcome['status']=='complete' and any(r['event']=='complete' for r in rows),
        snapshots_saved=sum(r['event']=='snapshot_saved' for r in rows),
        snapshot_bytes=sum(r.get('bytes',0) for r in rows if r['event']=='snapshot_saved'),
        successful_restores=sum(r['context_tokens']==r['expected_context_tokens'] for r in rows if r['event']=='snapshot_restored'),
        model_loads=sum(r['event']=='model_loaded' for r in rows),
        thermal=dict(max_cpu_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None),default=None),
            max_hat_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None),default=None),
            throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})))
    summary['findings'] = findings(summary)
    atomic_json(directory/'summary.json',summary)
    observations = [r for r in rows if r['event']=='response']
    write_csv(directory/'state-responses.csv',observations)
    write_csv(directory/'state-comparisons.csv',states)
    write_csv(directory/'parameter-comparisons.csv',parameters)
    write_csv(directory/'state-matrix.csv',summary['cells'])
    write_csv(directory/'snapshot-operations.csv',[r for r in rows if r['event'] in ('snapshot_saved','snapshot_restored')])
    cases = json.loads((directory/'state-inputs.json').read_text()) if (directory/'state-inputs.json').exists() else []
    candidates = [c for c in states if c['eligible'] and not c['outputs_equal'] and c['preset']=='implicit'
                  and c['left_mode']=='rebuild' and c['right_mode']=='live']
    chosen = min(candidates,key=lambda c:(next(x['target_tokens'] for x in cases if x['case_id']==c['case_id']),c['case_id'],c['repeat'])) if candidates else None
    if chosen:
        job = next(j for j in schedule['main'] if j['case_id']==chosen['case_id'] and j['preset']==chosen['preset'] and j['repeat']==chosen['repeat'])
        case = next(c for c in cases if c['case_id']==chosen['case_id'])
        atomic_json(directory/'focused-reproducer.json',dict(case=case,job=job,
            comparisons=[c for c in states if c['case_id']==chosen['case_id'] and c['repeat']==chosen['repeat']],
            snapshots=[r for r in rows if r['event']=='snapshot_saved' and r['job_id']==job['job_id']],
            observed_targets=[r for r in observations if r['kind']=='target' and r['phase']=='main'
                              and r['case_id']==chosen['case_id'] and r['repeat']==chosen['repeat']],
            replay_command=shlex.join(['python3',str(ROOT/'experiment.py'),'run','--phase','hat',
                '--profile','state-isolation','--source-run',config['source_run'],
                '--state-case',chosen['case_id'],'--output',str(directory.parent/(directory.name+'-replay')),
                '--max-seconds','3600']),
            interpretation='Smallest selected case with an observed baseline context difference. Replays all settings and state modes for this case; not a globally minimized prompt.'))
    else:
        atomic_json(directory/'focused-reproducer.json',dict(status='No eligible baseline context difference observed'))
    plots = []
    if charts and summary['cells']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        case_ids = sorted({c['case_id'] for c in summary['cells']})
        fig,axes=plt.subplots(1,len(case_ids),figsize=(4.3*len(case_ids),4.8),squeeze=False,layout='constrained')
        for axis,case_id in zip(axes[0],case_ids):
            values={(c['preset'],c['mode']):c for c in summary['cells'] if c['case_id']==case_id}
            matrix=[[100*values[(p,m)]['factual_correct']/values[(p,m)]['total'] if values[(p,m)]['total'] else float('nan') for m in MODES] for p in PRESETS]
            im=axis.imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
            axis.set(xticks=range(len(MODES)),xticklabels=MODES,yticks=range(len(PRESETS)),yticklabels=PRESETS,title=case_id)
            axis.tick_params(axis='x',labelrotation=60)
            for y,p in enumerate(PRESETS):
                for x,m in enumerate(MODES):
                    c=values[(p,m)]
                    axis.text(x,y,f'{c["factual_correct"]}/{c["total"]}',ha='center',va='center',fontsize=8,color='white' if matrix[y][x]>60 else 'black')
        fig.colorbar(im,ax=axes[0].tolist(),label='Factual correctness (%)',shrink=.7)
        fig.suptitle('State isolation: correct / observed target responses; planned three per cell')
        for ext in ('png','svg'):fig.savefig(directory/f'state-matrix.{ext}',dpi=150)
        plt.close(fig)
        plots.append('state-matrix')
    sections=['<h1>Isolating context-dependent answer changes</h1>',
        f'<p>Outcome: {html.escape(outcome["status"])}. Complete target coverage: {summary["complete"]}. '
        f'{summary["observed_targets"]} main target responses; {summary["observed_conditioning"]} main conditioning responses. '
        f'{len(summary["skipped_targets"])} target conditions skipped.</p>',
        '<p>Four selected cases, six state/process conditions, four parameter presets and three repetitions. '
        'The main matrix plans 288 target and 120 conditioning responses. A separate 16-response hardware validation checks snapshot support. '
        'A case-filtered replay is a subset with its own recorded schedule.</p>',
        '<h2>Resolved model settings</h2><pre>'+html.escape(json.dumps(env.get('model_defaults',{}),indent=2))+'</pre>',
        '<h2>Findings</h2>',*['<p>'+html.escape(s)+'</p>' for s in summary['findings']],
        '<h2>State comparison matrix</h2>',table(['Preset','Left condition','Right condition','Changed / eligible','Planned'],
            [[r['preset'],r['left_mode'],r['right_mode'],f'{r["output_differences"]}/{r["eligible"]}',r['planned']] for r in summary['state_contrasts']]),
        '<h2>Answer quality</h2>',table(['Case','Preset','Condition','Observed / planned','Factual correct','Strict correct','Wrong fact','Unscorable','Ledger mismatches'],
            [[r['case_id'],r['preset'],r['mode'],f'{r["total"]}/{r["planned"]}',r['factual_correct'],r['strict_correct'],r['wrong_fact'],r['unscorable'],r['ledger_mismatches']] for r in summary['cells']]),
        '<h2>Hardware validation and limits</h2><pre>'+html.escape(json.dumps(dict(capability=capability,thermal=summary['thermal'],exclusions=summary['state_exclusions']),indent=2))+'</pre>',
        '<p>Conditioning uses the original implicit generation settings and must exactly reproduce the frozen transcript. '
        'Parameter perturbations apply only to the target. Exact text/tokenized input, runtime counts, output chunks, completion status and scoring are saved. '
        'Decoded text and matching token counts do not expose actual generated token IDs. Invalid ledgers and incomplete responses remain in quality totals and are excluded from eligible comparisons. '
        'Snapshots are opaque; their hashes establish file lineage, not equivalence of hidden state. '
        'Fresh processes run sequentially after releasing the previous model. Request timing excludes model loading, reset and snapshot operations; these operations are logged separately. '
        'This is a selected-case diagnostic, not a general model accuracy or energy benchmark.</p>',
        '<h2>Artifacts</h2><p>state-inputs.json, schedule.json, state_isolation.jsonl, snapshots/*.bin and metadata, '
        'fresh/*/trial.json and logs, CSV exports, focused-reproducer.json, source snapshots and checksums accompany this report.</p>']
    if outcome.get('stop_reason'):sections.append('<p>Stop reason: '+html.escape(outcome['stop_reason'])+'</p>')
    for stem in plots:
        data=base64.b64encode((directory/f'{stem}.png').read_bytes()).decode()
        sections.append(f'<figure><img alt="State isolation response matrix" src="data:image/png;base64,{data}"></figure>')
    css='body{font:16px/1.5 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}h2{margin-top:36px}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Context state isolation</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    atomic_json(directory/'checksums.json',{str(p.relative_to(directory)):digest_file(p) for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'checksums.json'})
    print(json.dumps(dict(report=str(directory/'report.html'),complete=summary['complete'],targets=summary['observed_targets'],conditioning=summary['observed_conditioning']),indent=2))
