"""Saved evidence and standalone figures for the refinement campaign."""
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
from .common import ROOT,PLOT_PYTHON,atomic_json,read_jsonl
from .refine_spec import EVENTS,analyze

def build_report(directory,charts=True):
    directory=Path(directory)
    if (directory/'checksums.json').exists():raise ValueError('Sealed report is immutable')
    config=json.loads((directory/'config.json').read_text());events,errors=read_jsonl(directory/EVENTS)
    summary=analyze(events,config);summary['event_errors']=errors
    telemetry,_=read_jsonl(directory/'telemetry.jsonl')
    summary['telemetry']={k:max((r[k] for r in telemetry if r.get(k) is not None),default=None)
        for k in ('cpu_temp_c','hat_max_c','worker_tree_rss_bytes','gpu_clock_mhz')}
    summary['throttle_flags']=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})
    summary['failures']=[e for e in events if e['event'] in ('failure','compatibility_failure')]
    if errors:summary.update(complete=False,selected=None,accepted=False,decision='incomplete')
    atomic_json(directory/'summary.json',summary)
    with (directory/'measurements.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('section','fixture','policy','arm','repeat_or_turn','total_ms','valid','strict_correct','output'))
        for e in events:
            if e['event']!='measurement':continue
            if config['phase']=='cpu':writer.writerow((e['section'],e['fixture_id'],e['policy'],e['arm']['arm'],e['repeat'],e['total_ms'],e['response']['result']['correct'],'',''))
            else:
                for r in e['response']['result']['rows']:writer.writerow((e['section'],e['fixture_id'],'',e['condition'],r['turn'],r['request_ms'],r['valid'],r['answer_correct'],r['output']))
    if charts:subprocess.run([str(PLOT_PYTHON if PLOT_PYTHON.exists() else Path(sys.executable)),'-B','-m','efficiency.refine_report',str(directory)],cwd=ROOT,check=True)
    (directory/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Refinement experiment</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:3em auto;padding:1em}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(summary['stage'])}</h1><p>{html.escape(summary['decision'])}</p>
<p>All inference uses the existing Llama 3.2 3B package. Numerical GPU measurements are standalone attention kernels; they do not establish LLM GPU acceleration. No application defaults changed.</p>
<img src="refine.png" alt="Measured refinement results"><p><a href="measurements.csv">Measurements</a> · <a href="refine.jsonl">Raw events</a> · <a href="validation.json">Audit</a></p>
<pre>{html.escape(json.dumps(summary,indent=2))}</pre>''')
    return summary

def plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);s=json.loads((directory/'summary.json').read_text());rows=s['metrics']
    fig,ax=plt.subplots(figsize=(11,5))
    if s['stage'].startswith('npu'):
        ax.bar([r['label'] for r in rows],[100*r['accuracy'] for r in rows])
        ax.scatter([r['label'] for r in rows],[100*r['valid']/r['planned'] for r in rows],c='black',label='Valid completion')
        ax.set_ylim(0,105);ax.set_ylabel('Strict correctness (% of all planned answers)');ax.legend()
    else:
        labels=[f"{r['policy']['governor']}\n{r['policy']['binding']} / {r['policy']['waiting']}" for r in rows]
        ax.bar(labels,[r['cpu_mean_ms'] for r in rows],color=['seagreen' if r['stable'] else 'darkorange' for r in rows])
        ax.tick_params(axis='x',labelsize=8);ax.set_ylabel('Mean CPU request ms (IPC and verification included)')
    ax.set_title(f"{s['stage']}: {s['decision']}")
    if not rows:ax.text(.5,.5,'No complete scientific matrix',ha='center',transform=ax.transAxes)
    fig.tight_layout();fig.savefig(directory/'refine.png',dpi=150);fig.savefig(directory/'refine.svg');plt.close(fig)

if __name__=='__main__':plots(sys.argv[1])
