"""Quality outcomes and standalone plots; incomplete work never earns a speedup."""
import csv
import html
import json
from pathlib import Path
import subprocess
import sys

from .common import ROOT, PLOT_PYTHON, atomic_json, read_jsonl
from .quality_spec import EVENTS, decide


def analyze(directory):
    directory=Path(directory)
    config=json.loads((directory/'config.json').read_text())
    events,errors=read_jsonl(directory/EVENTS)
    summary=decide(events,config['quality_stage'],config['conditions'])
    telemetry,_=read_jsonl(directory/'telemetry.jsonl')
    summary.update(stage=config['quality_stage'],campaign_id=config['campaign_id'],event_errors=errors,
                   complete=any(r['event']=='measurement_complete' for r in events),
                   planned_main_answers=sum(r['planned'] for r in summary['metrics']),
                   measured_generations=sum(len(r['response']['result']['rows']) for r in events if r['event']=='dialogue'),
                   context_policy='rebuild',application_defaults_changed=False)
    summary['telemetry']={key:max((r[key] for r in telemetry if r.get(key) is not None),default=None)
                          for key in ('cpu_temp_c','hat_max_c','worker_tree_rss_bytes','worker_tree_cpu_seconds')}
    summary['throttle_flags']=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})
    if not summary['complete'] or errors:
        summary.update(decision='incomplete',selected=None,accepted=False)
    return summary


def plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    rows=summary['metrics'];labels=[r['condition']['label'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].bar(labels,[r['accuracy']*100 for r in rows],label='Strict correctness')
    axes[0].scatter(labels,[r['valid']/r['planned']*100 for r in rows],color='black',label='Completion')
    axes[0].axhline(95,color='grey',linestyle='--');axes[0].set_ylim(0,105)
    axes[0].set_ylabel('Percent of planned answers');axes[0].legend()
    axes[1].bar(labels,[(r['mean_dialogue_ms'] or 0)/1000 for r in rows])
    axes[1].set_ylabel('Mean complete dialogue seconds (incomplete excluded)')
    for ax in axes:ax.tick_params(axis='x',rotation=40)
    fig.suptitle(f"NPU quality: {summary['stage']} — {summary['decision']}")
    fig.tight_layout();fig.savefig(directory/'quality.png',dpi=150);fig.savefig(directory/'quality.svg');plt.close(fig)


def build_report(directory, charts=True):
    directory=Path(directory)
    if (directory/'validation.json').exists():
        raise ValueError('Sealed quality reports are immutable')
    summary=analyze(directory);atomic_json(directory/'summary.json',summary)
    events,_=read_jsonl(directory/EVENTS)
    with (directory/'answers.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('section','fixture','condition','turn','expected','output','status','valid','strict_correct','request_ms'))
        for e in events:
            if e['event']!='dialogue':continue
            for r in e['response']['result']['rows']:
                writer.writerow((e['section'],e['fixture_id'],e['condition']['label'],r['turn'],r['expected_answer'],
                                 r['output'],r['status'],r['valid'],r['answer_correct'],r['request_ms']))
    if charts:
        python=PLOT_PYTHON if PLOT_PYTHON.exists() else Path(sys.executable)
        subprocess.run([str(python),'-B','-m','efficiency.quality_report','--plot',str(directory)],cwd=ROOT,check=True)
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (
        r['condition']['label'],f"{r['correct']}/{r['planned']}",f"{r['valid']}/{r['planned']}",r['qualified'],r['mean_dialogue_ms']))+'</tr>'
        for r in summary['metrics'])
    page=f'''<!doctype html><meta charset="utf-8"><title>NPU quality experiment</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:3em auto;padding:1em}}td,th{{padding:.6em;text-align:left;border-bottom:1px solid #ccc}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>NPU quality: {html.escape(summary['stage'])}</h1><p>{html.escape(summary['decision'])}</p>
<p>Rebuilt context, pinned Qwen2.5 model. All planned answers remain in accuracy denominators.
Latency describes completed dialogues only. No application defaults changed.</p>
<table><tr><th>Condition</th><th>Strict correct</th><th>Valid complete</th><th>Qualified</th><th>Mean complete dialogue ms</th></tr>{rows}</table>
<img src="quality.png" alt="Accuracy, completion and complete-dialogue latency">
<p><a href="answers.csv">Answers</a> · <a href="quality.jsonl">Raw events</a> · <a href="validation.json">Independent audit</a></p>
<pre>{html.escape(json.dumps(summary,indent=2))}</pre>'''
    (directory/'report.html').write_text(page)
    return summary


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--plot':raise SystemExit('Usage: --plot DIRECTORY')
    plots(sys.argv[2])
