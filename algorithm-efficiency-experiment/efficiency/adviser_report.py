"""Offline adviser reliability/cost report with separate model and fallback counts."""
import base64
from collections import Counter
import html
import json
from pathlib import Path
import shutil
import statistics

from .adviser_spec import EVENTS
from .common import ROOT, atomic_json, digest_file, read_jsonl
from .coordination_report import release_snapshot, seal
from .report import table, write_csv


def summarize(rows,outcome,selection,holdout,development_schedule,holdout_schedule):
    measures=[r for r in rows if r['event']=='measurement']
    expected={(stage,j['case_id'],a) for stage,schedule in
              (('development',development_schedule),('holdout',holdout_schedule)) for j in schedule for a in j['arms']}
    actual=Counter((r['stage'],r['case_id'],r['arm']) for r in measures if r['stage'] in ('development','holdout'))
    coverage=bool(expected) and set(actual)==expected and all(n==1 for n in actual.values())
    complete=bool(coverage and len(holdout)==48 and outcome.get('status')=='complete' and
        any(r['event']=='complete' for r in rows) and not any(r['event']=='incomplete' for r in rows))
    case_map={c['case_id']:c for c in holdout};groups=[];interface=True
    for arm in ('baseline',selection.get('challenger'),'cpu'):
        if not arm:continue
        selected=[r for r in measures if r['stage']=='holdout' and r['arm']==arm]
        nonempty=[r for r in selected if case_map[r['case_id']]['payload']['eligible']]
        empty=[r for r in selected if not case_map[r['case_id']]['payload']['eligible']]
        origins=Counter(r['decision']['origin'] for r in nonempty)
        for r in selected:
            p=case_map[r['case_id']]['payload'];d=r['decision']
            interface=interface and ((d['choice'] in p['eligible'] and d['choice'] not in p['tested']) if p['eligible']
                                    else d['choice'] is None and d['origin']=='abstain' and r['response'] is None)
        times=[r['total_ms'] for r in nonempty]
        groups.append(dict(arm=arm,nonempty_cases=len(nonempty),empty_cases=len(empty),
            exact=origins['hat_exact'],canonicalized=origins['hat_canonicalized'],
            native_boundary=sum(r['decision']['origin']=='hat_exact' and r['decision']['completion_boundary']=='candidate_stop' for r in nonempty),
            model_accepted=origins['hat_exact']+origins['hat_canonicalized'],
            cpu_fallback=origins['cpu_fallback'],cpu_direct=origins['cpu'],
            mean_request_ms=statistics.mean(times) if times else None,
            median_request_ms=statistics.median(times) if times else None,
            p95_request_ms=sorted(times)[max(0,int(.95*len(times))-1)] if times else None))
    challenger=next((g for g in groups if g['arm']==selection.get('challenger')),None)
    reliable=bool(complete and interface and challenger and challenger['model_accepted']==40 and
                  challenger['empty_cases']==8 and all(r['state_valid'] for r in measures if r['stage']=='holdout'))
    return dict(complete=complete,coverage_exact=coverage,interface_passed=bool(complete and interface),
        reliability_passed=reliable,challenger=selection.get('challenger'),groups=groups,
        holdout_requests=sum(r['stage']=='holdout' for r in measures),
        holdout_model_requests=sum(r['stage']=='holdout' and r['response'] is not None for r in measures),
        total_model_requests=sum(r['response'] is not None for r in measures),outcome=outcome,
        recommendation='Candidate for a separate equal-budget tuning-utility experiment; defaults unchanged' if reliable
                       else 'Retain deterministic CPU selection; adviser reliability gate failed',
        limitations=['Synthetic eligibility/history cases measure a response contract, not optimal tuning choices.',
            'CPU fallbacks and punctuation corrections are separate from exact NPU outputs.',
            'All previous development and holdout examples are development data in this experiment.',
            'Repeating frozen cases is replication, not new independent evidence. GPU kernels were not exercised.'])


def build_report(directory,charts=True):
    directory=Path(directory)
    read=lambda name,default:json.loads((directory/name).read_text()) if (directory/name).exists() else default
    rows,damaged=read_jsonl(directory/EVENTS)
    summary=summarize(rows,read('outcome.json',{}),read('selection.json',{}),read('holdout-cases.json',[]),
                      read('development-schedule.json',[]),read('holdout-schedule.json',[]))
    if damaged:
        summary.update(complete=False,interface_passed=False,reliability_passed=False,damaged_records=damaged)
    telemetry,_=read_jsonl(directory/'telemetry.jsonl')
    summary['thermal']={k:max((r[k] for r in telemetry if isinstance(r.get(k),(int,float))),default=None)
                        for k in ('cpu_temp_c','hat_max_c')}
    summary['native_stop_preflight']=read('native-stop-preflight.json',{})
    atomic_json(directory/'summary.json',summary)
    write_csv(directory/'adviser-results.csv',summary['groups'])
    if not (directory/'device-release.json').exists():release_snapshot(directory,rows)
    fields=['arm','nonempty_cases','empty_cases','exact','canonicalized','native_boundary','model_accepted','cpu_fallback','cpu_direct','mean_request_ms','p95_request_ms']
    sections=['<h1>NPU adviser reliability with CPU validation</h1>',
        '<p>'+html.escape(summary['recommendation'])+'</p>',
        '<p>Complete: '+str(summary['complete'])+'. Interface passed: '+str(summary['interface_passed'])+
        '. NPU reliability passed: '+str(summary['reliability_passed'])+'.</p>',
        table(fields,[[r[k] for k in fields] for r in summary['groups']]),
        '<p>Request latency includes model communication, parsing, restoration and any CPU fallback. '
        'Exact counts include the native-boundary subset; that subset is not an additional count.</p>']
    if charts and summary['groups']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data=summary['groups'];xs=list(range(len(data)))
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        bottom=[0]*len(data)
        for key,color in (('exact','#346bab'),('canonicalized','#38a383'),('cpu_fallback','#d78a34'),('cpu_direct','#7e7e7e')):
            ys=[r[key] for r in data];axes[0].bar(xs,ys,bottom=bottom,label=key,color=color)
            bottom=[a+b for a,b in zip(bottom,ys)]
        axes[0].set(xticks=xs,xticklabels=[r['arm'] for r in data],ylabel='Nonempty holdout cases',ylim=(0,44),title='Origins of 40 selections')
        axes[0].legend(fontsize=8)
        axes[1].bar(xs,[max(r['mean_request_ms'] or .000001,.000001) for r in data],color='#346bab')
        axes[1].set(xticks=xs,xticklabels=[r['arm'] for r in data],ylabel='Mean full request ms (log scale)',yscale='log',title='Cost including validation and fallback')
        for extension in ('png','svg'):fig.savefig(directory/f'adviser-results.{extension}',dpi=160)
        plt.close(fig)
        encoded=base64.b64encode((directory/'adviser-results.png').read_bytes()).decode()
        sections.append(f'<img alt="Adviser response origins and latency" src="data:image/png;base64,{encoded}">')
    sections+=['<h2>Native stop controls</h2><pre>'+html.escape(json.dumps(summary['native_stop_preflight'],indent=2))+'</pre>',
        '<h2>Development selection</h2><pre>'+html.escape(json.dumps(read('selection.json',{}),indent=2))+'</pre>',
        '<h2>Interpretation</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in summary['limitations'])+'</ul>',
        '<h2>Runtime and temperatures</h2><pre>'+html.escape(json.dumps(dict(outcome=summary['outcome'],thermal=summary['thermal']),indent=2))+'</pre>']
    css='body{font:16px/1.5 system-ui;max-width:1250px;margin:36px auto;padding:0 20px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}img{width:100%}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adviser validation</title><style>'+css+'</style>'+''.join(sections)+'</html>')
    failed=[dict(case_id=r['case_id'],arm=r['arm'],raw=r['response']['result']['raw'],reason=r['decision']['fallback_reason'])
            for r in rows if r['event']=='measurement' and r['stage']=='holdout' and r['decision']['origin']=='cpu_fallback']
    atomic_json(directory/'next-experiment.json',dict(status='planned_not_run',reliability_passed=summary['reliability_passed'],
        next_step='Equal-total-budget comparison of model-guided versus deterministic tuning on new workloads' if summary['reliability_passed']
                  else 'Keep CPU selection; use these failures only as development evidence and require fresh holdouts before another reliability claim',
        failed_cases_for_future_development=failed))
    analysis=directory/'analysis-source';analysis.mkdir(exist_ok=True)
    for name in ('adviser_report.py','adviser_audit.py','adviser_spec.py','adviser_interface.py'):
        shutil.copy2(ROOT/'efficiency'/name,analysis/name)
    seal(directory)
    print(json.dumps(dict(report=str(directory/'report.html'),complete=summary['complete'],
                         interface_passed=summary['interface_passed'],reliability_passed=summary['reliability_passed']),indent=2))
