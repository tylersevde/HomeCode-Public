"""Offline study summaries, raw exports and standalone figures."""
import csv
import html
import json
from pathlib import Path
import subprocess
import sys
from .common import ROOT,PLOT_PYTHON,atomic_json,read_jsonl
from .study_spec import EVENTS,analyze


def build_report(directory,charts=True):
    directory=Path(directory)
    if (directory/'validation.json').exists():raise ValueError('Sealed study report is immutable')
    config=json.loads((directory/'config.json').read_text());events,parse_errors=read_jsonl(directory/EVENTS)
    summary=analyze(events,config);summary['event_errors']=parse_errors
    telemetry,_=read_jsonl(directory/'telemetry.jsonl')
    summary['telemetry']={k:max((r[k] for r in telemetry if r.get(k) is not None),default=None)
                          for k in ('cpu_temp_c','hat_max_c','worker_tree_rss_bytes','gpu_clock_mhz')}
    summary['throttle_flags']=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})
    summary['cold_loads']=[dict(instance=e['instance'],model=e['environment'].get('model_id'),
        loading_seconds=e['environment'].get('loading_seconds'),owner=e['owner']) for e in events if e['event']=='worker_ready']
    if parse_errors:summary.update(selected=None,accepted=False,complete=False,decision='incomplete')
    atomic_json(directory/'summary.json',summary)
    with (directory/'measurements.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('section','fixture','arm','repeat_or_turn','total_ms','valid','strict_correct','output'))
        for e in events:
            if e['event']!='measurement':continue
            if config['track']=='gpu-stream':
                writer.writerow((e['section'],e['fixture_id'],e['arm']['arm'],e['repeat'],e['total_ms'],e['response']['result']['correct'],'',''))
            else:
                for r in e['response']['result']['rows']:
                    writer.writerow((e['section'],e['fixture_id'],e['model'],r['turn'],r['request_ms'],r['valid'],r['answer_correct'],r['output']))
    if charts:
        subprocess.run([str(PLOT_PYTHON if PLOT_PYTHON.exists() else Path(sys.executable)),'-B','-m','efficiency.study_report','--plot',str(directory)],cwd=ROOT,check=True)
    desc=('Single-query FP32 attention, equal weight per shape. This does not measure LLM GPU acceleration.' if config['track']=='gpu-stream' else
          'Deployed model packages compared with native templates and tokenizers. All planned turns remain in the accuracy denominator. Latency excludes cold loading, which is reported separately.')
    page=f'''<!doctype html><meta charset="utf-8"><title>CPU GPU NPU study</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:3em auto;padding:1em}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(config['track'])}: {html.escape(config['study_stage'])}</h1><p>{html.escape(summary['decision'])}</p>
<p>{desc} No application defaults changed.</p><img src="study.png" alt="Study results">
<p><a href="measurements.csv">Measurements</a> · <a href="study.jsonl">Raw records</a> · <a href="validation.json">Audit</a></p>
<pre>{html.escape(json.dumps(summary,indent=2))}</pre>'''
    (directory/'report.html').write_text(page);return summary


def plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);s=json.loads((directory/'summary.json').read_text());rows=s['metrics']
    fig,ax=plt.subplots(figsize=(10,5));labels=[r['label'] for r in rows]
    if s['track']=='gpu-stream':
        ax.bar(labels,[(r['mean_ms'] or 0) for r in rows]);ax.set_ylabel('Mean request milliseconds (includes IPC and verification)')
    else:
        ax.bar(labels,[r['accuracy']*100 for r in rows]);ax.scatter(labels,[r['valid']/r['planned']*100 for r in rows],color='black',label='Completion')
        ax.axhline(95,color='grey',linestyle='--');ax.set_ylim(0,105);ax.set_ylabel('Strict correctness, % of planned answers');ax.legend()
    ax.set_title(f"{s['track']} {s['stage']}: {s['decision']}");fig.tight_layout()
    fig.savefig(directory/'study.png',dpi=150);fig.savefig(directory/'study.svg');plt.close(fig)


if __name__=='__main__':plots(sys.argv[2])
