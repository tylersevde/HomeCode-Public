"""Offline campaign analysis. Correctness, completion and promotion stay separate."""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .common import ROOT,PLOT_PYTHON,atomic_json,digest_file,read_jsonl
from .research_spec import CELLS,interval,accepted,checkpoint_equivalent


def mean(values):return statistics.mean(values) if values else None


def paired(rows,first,second,confidence=.95):
    by=defaultdict(lambda:defaultdict(list))
    for r in rows:by[r['block']][r['arm']].append(r['total_ms'])
    pairs=[(mean(g[first]),mean(g[second])) for _,g in sorted(by.items()) if first in g and second in g]
    return interval(pairs,confidence=confidence)


def groups(rows,keys):
    result=defaultdict(list)
    for r in rows:result[tuple(r[k] for k in keys)].append(r)
    return result


def cache_matches(rows,a,b):
    grouped=groups(rows,('fixture_id','repeat'))
    return bool(grouped) and all(set([a,b])<=set(r['arm'] for r in rs) and checkpoint_equivalent({
        r['arm']:r['response']['result']['rows'] for r in rs if r['arm'] in (a,b)}) for rs in grouped.values())


def combined_matches(rows,arm,baseline):
    grouped=groups(rows,('fixture_id','repeat'))
    for rs in grouped.values():
        selected={r['arm']:r for r in rs if r['arm'] in (arm,baseline)}
        if not {arm,baseline}<=set(selected):return False
        signatures=[]
        for r in selected.values():
            hat=r['responses']['hat']['result']
            if not r['correct'] or not r['npu_valid']:return False
            signatures.append((hat['prompt_sha256'],hat['output'],hat['status'],hat['context_after']))
        if len(set(signatures))!=1:return False
    return bool(grouped)


def analyze(directory):
    directory=Path(directory);read=lambda n:json.loads((directory/n).read_text())
    config=read('config.json');stage=config['research_stage'];parents=read('parents.json')
    rows,damaged=read_jsonl(directory/'research.jsonl');outcome=read('outcome.json')
    main=[r for r in rows if r.get('phase')=='main'];numeric=[r for r in main if r['event']=='numeric']
    contexts=[r for r in main if r['event']=='context'];combined=[r for r in main if r['event']=='combined']
    summary=dict(stage=stage,complete=outcome['status']=='complete' and any(r['event']=='complete' for r in rows) and not damaged,
        hardware_seconds=outcome['elapsed_seconds'],damaged_events=damaged,selection={},numeric=[],context=[],combined=[],
        numeric_requests=len(numeric),context_dialogues=len(contexts),combined_jobs=len(combined),
        measured_model_generations=sum(len(r['response']['result']['rows']) for r in contexts)+sum(not r['responses']['hat']['result'].get('skipped',False) for r in combined),
        skipped_combined_followups=sum(r['responses']['hat']['result'].get('skipped',False) for r in combined),
        unexpected_output_io=sum(r['response']['result']['unexpected_output_io'] for r in numeric),
        future_default='CPU; all selected policies remain opt-in pending independent confirmation.')
    if stage=='confirm':summary['future_default']='CPU attention remains the default; accepted confirmation policies are eligible for opt-in.'
    for (cell,arm),values in sorted(groups(numeric,('cell_id','arm')).items()):
        data=[r['response']['result'] for r in values]
        summary['numeric'].append(dict(cell_id=cell,arm=arm,count=len(values),mean_ms=mean([r['total_ms'] for r in values]),
            median_ms=statistics.median(r['total_ms'] for r in values),native_mean_ms=mean([r['request_ms'] for r in data]),
            correct=all(r['correct'] and r['matches_warmup'] and not r['validation_errors'] for r in data),
            mean_gpu_ms=mean([r['gpu_ms'] for r in data]),mean_cpu_seconds=mean([r['cpu_seconds'] for r in data]),
            stages_ms={k:mean([r['profile']['stages_ms'][k] for r in data]) for k in ('projection','scores','softmax','apply')},
            host_ms={k:mean([r['profile'][k] for r in data]) for k in ('projection_ms','input_copy_ms','output_copy_ms','submit_ms','fence_ms','query_ms')},
            backend=data[0]['backend'],variant=data[0]['variant'],profiling=data[0]['profiling']))
    if stage=='profile' and summary['complete']:
        cpu={};gpu={};overheads=[]
        for cell in (c['cell_id'] for c in CELLS):
            gs=[r for r in summary['numeric'] if r['cell_id']==cell and r['correct']]
            cpu[cell]=min((r for r in gs if r['backend'].startswith('native')),key=lambda r:(r['mean_ms'],r['arm']))['backend']
            gpu[cell]=min((r for r in gs if r['backend'].startswith('C') and not r['profiling']),key=lambda r:(r['mean_ms'],r['arm']))['backend']
            for backend in ('C0','C4','C6','C7'):
                enabled=next(r for r in gs if r['arm']==backend+'-p1');disabled=next(r for r in gs if r['arm']==backend+'-p0')
                ratio=enabled['median_ms']/disabled['median_ms']
                overheads.append(dict(cell_id=cell,backend=backend,ratio=ratio,diagnostic_only=abs(ratio-1)>.05))
        summary['selection']=dict(cpu=cpu,gpu=gpu);summary['profiling_overhead']=overheads
    if stage=='gpu' and summary['complete']:
        routes={};best={};comparisons=[]
        for cell in (c['cell_id'] for c in CELLS):
            gs=[r for r in summary['numeric'] if r['cell_id']==cell and r['correct']]
            winner=min((r for r in gs if r['backend'].startswith('C')),key=lambda r:(r['mean_ms'],r['arm']))
            cpu=parents['profile']['selection']['cpu'][cell]
            values=[r for r in numeric if r['cell_id']==cell]
            comparison=paired(values,cpu,winner['arm']);promote=accepted(comparison,True)
            best[cell]=dict(backend=winner['backend'],variant=winner['variant'])
            routes[cell]=best[cell] if promote else dict(backend=cpu,variant=0)
            original=winner['backend']+'-v0'
            comparisons.append(dict(cell_id=cell,against_cpu=comparison,development_eligible=promote,
                against_original_gpu=paired(values,original,winner['arm'])))
        summary['selection']=dict(routes=routes,gpu_best=best);summary['development_comparisons']=comparisons
    for (model,arm),values in sorted(groups(contexts,('model','arm')).items()):
        answers=[a for r in values for a in r['response']['result']['rows']]
        summary['context'].append(dict(model=model,arm=arm,dialogues=len(values),planned_answers=len(values)*4,
            observed_answers=len(answers),valid_answers=sum(a['valid'] for a in answers),correct_answers=sum(a['answer_correct'] for a in answers),
            contract_failures=sum(len(r['response']['result'].get('contract_failures',[])) for r in values),
            quarantined_turns=sum(len(r['response']['result'].get('quarantined_turns',[])) for r in values),
            mean_dialogue_ms=mean([r['total_ms'] for r in values]),mean_followup_ms=mean([a['request_ms'] for a in answers if a['turn']>0]),
            checkpoint_count=sum(a['checkpoint'] is not None for a in answers),
            equivalent_to_rebuild=cache_matches([r for r in contexts if r['model']==model],arm,'rebuild')))
    if stage=='context' and summary['complete']:
        primary=[r for r in contexts if r['model']=='primary'];candidates=[];comparisons=[]
        for arm in ('retain','checkpoint'):
            match=cache_matches(primary,arm,'rebuild');result=paired(primary,'rebuild',arm)
            comparisons.append(dict(arm=arm,cache_equivalent=match,inference=result))
            if accepted(result,match):candidates.append(arm)
        selected=min(candidates,key=lambda a:(mean([r['total_ms'] for r in primary if r['arm']==a]),a)) if candidates else 'rebuild'
        summary['selection']=dict(cache=selected);summary['development_comparisons']=comparisons
    for (cell,arm),values in sorted(groups(combined,('cell_id','arm')).items()):
        summary['combined'].append(dict(cell_id=cell,arm=arm,jobs=len(values),mean_ms=mean([r['total_ms'] for r in values]),
            p95_ms=sorted(r['total_ms'] for r in values)[min(len(values)-1,int(.95*len(values)))],
            numerical_correct=all(r['correct'] for r in values),npu_valid=sum(r['npu_valid'] for r in values),
            npu_correct_answers=sum(r['responses']['hat']['result']['answer_correct'] for r in values),
            npu_skipped=sum(r['responses']['hat']['result'].get('skipped',False) for r in values)))
    if stage=='combined' and summary['complete']:
        candidates=[];comparisons=[]
        for arm in ('overlap-0','overlap-25','overlap-50','overlap-100'):
            match=combined_matches(combined,arm,'overlap-0');result=paired(combined,'overlap-0',arm)
            comparisons.append(dict(arm=arm,behavior_preserved=match,inference=result))
            if match:candidates.append(arm)
        selected=min(candidates,key=lambda a:(mean([r['total_ms'] for r in combined if r['arm']==a]),a)) if candidates else 'overlap-0'
        summary['selection']=dict(condition=selected);summary['development_comparisons']=comparisons
    if stage=='confirm' and summary['complete']:
        confidence=1-.05/3
        numeric_correct=all(r['correct'] for r in summary['numeric'])
        shape_ratios=[paired([r for r in numeric if r['cell_id']==c['cell_id']],'incumbent','candidate',confidence)['ratio'] for c in CELLS]
        numeric_changed=any(v['backend']!=parents['profile']['selection']['cpu'][k] for k,v in parents['gpu']['selection']['routes'].items())
        cache_changed=parents['context']['selection']['cache']!='rebuild'
        combined_changed=parents['combined']['selection']['condition']!='overlap-0'
        specs=[('attention',numeric,'incumbent','candidate',numeric_correct,numeric_changed,shape_ratios),
            ('context',contexts,'rebuild','candidate',cache_matches(contexts,'rebuild','candidate'),cache_changed,()),
            ('coordination',combined,'incumbent','candidate',combined_matches(combined,'candidate','incumbent'),combined_changed,())]
        summary['confirmation']=[dict(comparison=name,inference=paired(data,a,b,confidence),correct=correct,changed=changed,
            accepted=changed and accepted(paired(data,a,b,confidence),correct,ratios),shape_ratios=ratios)
            for name,data,a,b,correct,changed,ratios in specs]
        summary['selection']=dict(routes=parents['gpu']['selection']['routes'] if summary['confirmation'][0]['accepted'] else
            {c:dict(backend=b,variant=0) for c,b in parents['profile']['selection']['cpu'].items()},
            cache=parents['context']['selection']['cache'] if summary['confirmation'][1]['accepted'] else 'rebuild',
            condition=parents['combined']['selection']['condition'] if summary['confirmation'][2]['accepted'] else 'overlap-0')
    summary['pilot_gates']=[r for r in rows if r['event']=='pilot_gate']
    summary['secondary_deferred']=[r['reason'] for r in rows if r['event']=='secondary_deferred']
    telemetry,_=read_jsonl(directory/'telemetry.jsonl')
    summary['telemetry']=dict(cpu_peak_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None),default=None),
        hat_peak_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None),default=None),
        throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None}))
    summary['telemetry'].update(
        sampled_worker_tree_peak_rss_bytes=max((r['worker_tree_rss_bytes'] for r in telemetry if r.get('worker_tree_rss_bytes') is not None),default=None),
        minimum_host_available_memory_bytes=min((r['available_memory_bytes'] for r in telemetry if r.get('available_memory_bytes') is not None),default=None))
    return summary


def seal(directory):
    directory=Path(directory)
    atomic_json(directory/'checksums.json',{str(p.relative_to(directory)):digest_file(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and p!=directory/'checksums.json'})


def plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);s=json.loads((directory/'summary.json').read_text());labels=[];values=[]
    for key in ('numeric','context','combined'):
        rows=s[key]
        if key in ('numeric','combined'):rows=[r for r in rows if r['cell_id']=='prefill-n1024-b4']
        for r in rows:labels.append(f'{key}: '+r.get('model','')+' '+r['arm']);values.append(r.get('mean_ms',r.get('mean_dialogue_ms')))
    if not labels:return
    fig,ax=plt.subplots(figsize=(10,max(4,len(labels)*.34)));ax.barh(labels,values);ax.invert_yaxis()
    ax.set_xlabel('Mean full request / dialogue / job latency (ms; different workloads shown separately)')
    ax.set_title(f"Attention research: {s['stage']}"+(' — largest prefill cell' if not s['context'] else ' — attention jobs and context dialogue totals'))
    fig.tight_layout();fig.savefig(directory/'latency.png',dpi=160);fig.savefig(directory/'latency.svg');plt.close(fig)


def build_report(directory,charts=True):
    directory=Path(directory);summary=analyze(directory);atomic_json(directory/'summary.json',summary)
    atomic_json(directory/'selection.json',summary['selection'])
    with (directory/'measurements.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('kind','cell_or_model','arm','count','mean_ms','correct_or_equivalent'))
        for kind in ('numeric','context','combined'):
            for r in summary[kind]:writer.writerow((kind,r.get('cell_id',r.get('model')),r['arm'],r.get('count',r.get('dialogues',r.get('jobs'))),
                r.get('mean_ms',r.get('mean_dialogue_ms')),r.get('correct',r.get('equivalent_to_rebuild',r.get('numerical_correct')))))
    if charts and PLOT_PYTHON.exists():
        subprocess.run([str(PLOT_PYTHON),'-m','efficiency.research_report',str(directory),'--charts-only'],cwd=ROOT,check=True,timeout=90)
    body=f'<h1>Attention research: {html.escape(summary["stage"])}</h1><p>Hardware session: {summary["hardware_seconds"]:.3f} seconds. Completed: {summary["complete"]}.</p>'
    body+='<p>GPU stages overlap host wait time. Model answer accuracy and cache equivalence are distinct. Development selections are experimental.</p>'
    if (directory/'latency.png').exists():body+='<img src="latency.png" style="max-width:100%" alt="Latency comparison">'
    body+='<pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'
    (directory/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Attention research</title><body>'+body+'</body></html>')
    seal(directory)
    return summary


if __name__=='__main__':
    if '--charts-only' in sys.argv:plots(Path(sys.argv[1]))
    else:build_report(Path(sys.argv[1]))
