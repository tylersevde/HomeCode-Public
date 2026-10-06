"""Offline coordination analysis; invalid advice never earns application credit."""
import base64
from collections import Counter
import html
import json
from pathlib import Path
import shutil
import statistics

from .common import ROOT, atomic_json, digest_file, read_jsonl
from .coordination_spec import ARCHITECTURES, CONDITIONS, EVENTS, SEED
from .feedback_spec import paired_interval, parse_proposal
from .report import table, write_csv


def advice_valid(response, case, environment):
    r = response['result']
    ledger = (r.get('token_ledger_valid') is True and r.get('context_before')==0 and
              r.get('context_after')==r.get('expected_context_after'))
    choice, _ = parse_proposal(r['output'], environment['stop_tokens'], r['completion_status'], ledger, case['eligible'])
    return choice is not None and choice==r.get('choice') and r.get('messages')==case['messages']


def analyze(rows, outcome, cases, environment):
    case_map = {c['case_id']:c for c in cases}
    main = [r for r in rows if r['event']=='condition' and r['stage']=='main']
    expected = {(i,a,c) for i in range(6) for a in ARCHITECTURES for c in CONDITIONS}
    counts = Counter((r['repeat'],r['architecture'],r['condition']) for r in main)
    coverage = set(counts)==expected and all(n==1 for n in counts.values())
    checks = [r for r in rows if r['event']=='advice_check']
    check_counts = Counter(r['case_id'] for r in checks)
    expected_checks = {c['case_id'] for c in cases if c['split'] in ('development','holdout')}
    check_coverage = set(check_counts)==expected_checks and all(n==1 for n in check_counts.values()) and len(checks)==40
    valid_checks = {split:sum(advice_valid(r['response'],case_map[r['case_id']],environment)
                   for r in checks if r['split']==split) for split in ('development','holdout')}
    valid_main = sum(advice_valid(r['adviser'],case_map[r['case_id']],environment) for r in main)
    correct = bool(main) and all(len(r['numerical']['result']['requests'])==24 and
        r['numerical']['result']['correct'] and all(v['correct'] and not v['validation_errors']
        for v in r['numerical']['result']['requests']) for r in main)
    complete = (coverage and check_coverage and outcome.get('status')=='complete' and
                any(r['event']=='complete' for r in rows) and not any(r['event']=='incomplete' for r in rows))
    valid = complete and valid_checks['holdout']==20 and valid_main==48
    groups = []
    lookup = {(r['repeat'],r['architecture']+'_'+r['condition']):r for r in main}
    for arch in ARCHITECTURES:
        for condition in CONDITIONS:
            selected = [r for r in main if r['architecture']==arch and r['condition']==condition]
            if not selected: continue
            first = [r['numerical']['result']['requests'][0] for r in selected]
            groups.append(dict(architecture=arch, condition=condition, count=len(selected),
                mean_total_ms=statistics.mean(r['total_ms'] for r in selected),
                mean_first_numeric_ms=statistics.mean(r['request_ms'] for r in first),
                mean_first_gpu_ms=statistics.mean(r['gpu_ms'] for r in first),
                mean_hat_ms=statistics.mean(r['adviser']['result']['total_ms'] for r in selected),
                valid_advice=sum(advice_valid(r['adviser'],case_map[r['case_id']],environment) for r in selected)))
    pairs = [('primary_process_cpu_overlap','thread_cpu_overlap','process_cpu_overlap'),
             ('process_vs_thread_gpu_overlap','thread_gpu_overlap','process_gpu_overlap')]
    for arch in ARCHITECTURES:
        for backend in ('cpu','gpu'):
            pairs.append((f'{arch}_{backend}_overlap',f'{arch}_{backend}_serial',f'{arch}_{backend}_overlap'))
        for mode in ('serial','overlap'):
            pairs.append((f'{arch}_{mode}_gpu_vs_cpu',f'{arch}_cpu_{mode}',f'{arch}_gpu_{mode}'))
    comparisons = {}
    for name, left, right in pairs:
        if all((i,k) in lookup for i in range(6) for k in (left,right)):
            result = paired_interval([(lookup[i,left]['total_ms'],lookup[i,right]['total_ms']) for i in range(6)],SEED)
            comparisons[name] = dict(left=left, right=right, **result,
                timing_gate=result['ratio']>=1.10 and result['ci95'][0]>1,
                application_gate=valid and correct and result['ratio']>=1.10 and result['ci95'][0]>1)
    primary = comparisons.get('primary_process_cpu_overlap',{})
    return dict(complete=complete, coverage_exact=coverage, advice_check_coverage=check_coverage,
        observed_conditions=len(main), expected_conditions=48,
        observed_numerical_requests=sum(len(r['numerical']['result']['requests']) for r in main),
        all_numerical_correct=correct, valid_checks=valid_checks, valid_main_advice=valid_main,
        useful_advice_gate=valid, groups=groups, comparisons=comparisons,
        recommend_process_overlap=bool(primary.get('application_gate')),
        recommendation='Opt-in CPU/process overlap' if primary.get('application_gate') else 'Retain current CPU routing; no validated orchestration promotion',
        statistical_unit='Six paired repetition blocks; intervals describe this session.',
        limitations=['Valid IDs demonstrate an interface contract, not superior tuning decisions.',
            'The 48 main adviser responses repeat six preassigned held-out cases across eight conditions.',
            'Independent attention and adviser jobs; no splitting of the Qwen model across devices.',
            'Host interval overlap does not establish continuous simultaneous device execution or a GIL diagnosis.',
            'GPU shared with the desktop; energy and GPU utilization unmeasured.'], outcome=outcome)


def seal(directory):
    atomic_json(directory/'checksums.json', {str(p.relative_to(directory)):digest_file(p)
        for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'checksums.json'})


def release_snapshot(directory, rows):
    import psutil
    holders, errors = [], []
    for process in psutil.process_iter(['pid','name']):
        fd_dir = Path('/proc')/str(process.pid)/'fd'
        try:
            devices = sorted({str(p.resolve()) for p in fd_dir.iterdir()
                              if str(p.resolve()).startswith(('/dev/hailo','/dev/dri/'))})
            if devices:
                holders.append(dict(**process.info, devices=devices))
        except (OSError, psutil.Error) as exc:
            # Other users' descriptors may be unreadable; preserve that limit.
            if isinstance(exc, PermissionError): errors.append(dict(pid=process.pid,error='permission_denied'))
    releases = [r for r in rows if r['event']=='worker_release']
    atomic_json(directory/'device-release.json', dict(worker_releases=releases,
        all_recorded_workers_stopped=bool(releases) and all(not r['alive'] for r in releases),
        device_holders=holders, inaccessible_processes=errors,
        note='Other desktop GPU owners are unrelated to this experiment.'))


def build_report(directory, charts=True):
    directory = Path(directory)
    read = lambda name, default: json.loads((directory/name).read_text()) if (directory/name).exists() else default
    rows, damaged = read_jsonl(directory/EVENTS)
    summary = analyze(rows, read('outcome.json',{}), read('advice-cases.json',[]), read('adviser-environment.json',{}))
    if damaged:
        summary.update(complete=False, recommend_process_overlap=False, damaged_records=damaged)
    telemetry, _ = read_jsonl(directory/'telemetry.jsonl')
    summary['thermal'] = {k:max((r[k] for r in telemetry if isinstance(r.get(k),(int,float))),default=None)
                          for k in ('cpu_temp_c','hat_max_c')}
    atomic_json(directory/'summary.json',summary)
    write_csv(directory/'coordination-results.csv',summary['groups'])
    if not (directory/'device-release.json').exists():
        release_snapshot(directory,rows)
    sections = ['<h1>CPU / GPU / NPU coordination</h1>',
        '<p>'+html.escape(summary['recommendation'])+'</p>',
        '<p>Complete: '+str(summary['complete'])+'. Valid held-out advice: '+str(summary['valid_checks']['holdout'])+
        '/20. Valid main advice: '+str(summary['valid_main_advice'])+'/48.</p>',
        '<p>Timing gains with invalid advice are diagnostics, not application improvements.</p>',
        table(list(summary['groups'][0]) if summary['groups'] else [], [list(r.values()) for r in summary['groups']]),
        '<h2>Paired comparisons</h2><pre>'+html.escape(json.dumps(summary['comparisons'],indent=2))+'</pre>',
        '<h2>Interpretation</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in summary['limitations'])+'</ul>',
        '<h2>Runtime and temperatures</h2><pre>'+html.escape(json.dumps(dict(outcome=summary['outcome'],thermal=summary['thermal']),indent=2))+'</pre>']
    if charts and summary['groups']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = summary['groups']
        fig, ax = plt.subplots(figsize=(11,5),layout='constrained')
        ax.bar(range(len(data)),[r['mean_total_ms']/1000 for r in data],
               color=['#3264a8' if r['condition'].startswith('cpu') else '#d87735' for r in data])
        ax.set(xticks=range(len(data)),xticklabels=[r['architecture']+'\n'+r['condition'] for r in data],
               ylabel='Mean full-job seconds',title='Coordination: 24 attention requests + one adviser request')
        for suffix in ('png','svg'): fig.savefig(directory/f'coordination-results.{suffix}',dpi=160)
        plt.close(fig)
        encoded=base64.b64encode((directory/'coordination-results.png').read_bytes()).decode()
        sections.insert(3,f'<img alt="Eight coordination conditions" src="data:image/png;base64,{encoded}">')
    css='body{font:16px/1.5 system-ui;max-width:1200px;margin:36px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}img{width:100%}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Device coordination</title><style>'+css+'</style>'+''.join(sections)+'</html>')
    failures=[dict(case_id=r['case_id'],split=r['split'],output=r['response']['result']['output'],
                   rejection_reason=r['response']['result']['rejection_reason']) for r in rows
              if r['event']=='advice_check' and r['response']['result']['choice'] is None]
    atomic_json(directory/'next-experiment.json',dict(status='planned_not_run',
        primary_timing_gate=summary['comparisons'].get('primary_process_cpu_overlap',{}).get('timing_gate',False),
        recommendation=summary['recommendation'],
        next_adviser='Use failed cases only for development; new disjoint holdout required. Compare with deterministic selection before claiming tuning utility.',
        next_gpu='Profile individual kernels before further GPU tuning; compare against CPU with matching orchestration.',
        failed_cases_for_future_development=failures))
    folder=directory/'analysis-source';folder.mkdir(exist_ok=True)
    for name in ('coordination_report.py','coordination_audit.py','coordination_spec.py'):
        shutil.copy2(ROOT/'efficiency'/name,folder/name)
    seal(directory)
    print(json.dumps(dict(report=str(directory/'report.html'),complete=summary['complete'],
                         recommendation=summary['recommendation']),indent=2))
