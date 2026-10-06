"""Offline reports and final campaign sealing. Never regenerate sealed artifacts."""
import csv
import html
import json
import math
from pathlib import Path
import subprocess
import sys
from .common import ROOT, PLOT_PYTHON, atomic_json, digest_file, read_jsonl, utc
from .hybrid import verify_artifacts
from .reliability_spec import EVENTS, CAPS, MAX_SECONDS, analyze

def read(p):return json.loads(Path(p).read_text())

def build_report(directory,charts=True):
    directory=Path(directory)
    if (directory/'checksums.json').exists():raise ValueError('Sealed reliability report is immutable')
    config=read(directory/'config.json');events,errors=read_jsonl(directory/EVENTS);summary=analyze(events,config)
    telemetry,_=read_jsonl(directory/'telemetry.jsonl')
    summary.update(event_errors=errors,failures=[e for e in events if e['event']=='failure'],
        telemetry={k:max((r[k] for r in telemetry if r.get(k) is not None),default=None) for k in ('cpu_temp_c','hat_max_c','worker_tree_rss_bytes','gpu_clock_mhz')},
        throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None}))
    if errors:summary.update(complete=False,selected=None,accepted=False,decision='incomplete')
    if not summary['complete'] and any(e['event']=='pilot_gate' and not e['fits'] for e in events):summary['decision']='deferred_by_pilot'
    atomic_json(directory/'summary.json',summary)
    with (directory/'measurements.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(('section','fixture','policy','condition','job_started','repeat','turn_or_call','full_ms','native_ms','verification_ms','observer_ms','valid','strict_correct','output'))
        for e in events:
            if e['event']!='measurement':continue
            prefix=(e['section'],e['fixture_id'],e.get('policy',''),e.get('condition',e.get('label','')),e['started'],e.get('repeat',''))
            if config['reliability_stage'].startswith('npu'):
                for r in e['response']['result']['rows']:writer.writerow((*prefix,r['turn'],e['total_ms'] if r['turn']==0 else '','','','',r['valid'],r['answer_correct'],r['output']))
            elif config['reliability_stage'].startswith('combined'):
                writer.writerow((*prefix,'',e['total_ms'],'','','',e['valid'],all(e['answer_correct']),''))
            else:
                result=e['response']['result'];rows=result.get('requests',[result])
                for i,r in enumerate(rows):writer.writerow((*prefix,i,e['total_ms'] if i==0 else '',r['request_ms'],(r['ended']-r['computed'])*1000,result['observer_ms'] if i==0 else '',r['correct'],'',''))
    if charts:subprocess.run([str(PLOT_PYTHON if PLOT_PYTHON.exists() else Path(sys.executable)),'-B','-m','efficiency.reliability_report',str(directory)],cwd=ROOT,check=True)
    explanation='CPU batch calls are nested measurements, not independent statistical samples. GPU work is standalone numerical attention; no model layers are offloaded.'
    if config['reliability_stage'].startswith(('npu','combined')):
        explanation='The one-fact task is diagnostic only. Qualification requires full-table dialogues. '+explanation
    (directory/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Reliability experiment</title>'+STYLE+
        f'<h1>{html.escape(summary["stage"])}</h1><p>{html.escape(summary["decision"])}</p>'+
        '<p>'+explanation+'</p>'+
        '<img src="reliability.png" alt="Stage results"><p><a href="measurements.csv">Measurements</a> · <a href="reliability.jsonl">Raw evidence</a> · <a href="validation.json">Offline audit</a></p>'+
        '<pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>')
    return summary

STYLE='<style>body{font:17px system-ui;max-width:1100px;margin:3em auto;padding:1em}img{max-width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:.5em;text-align:left;border-bottom:1px solid #ccc}</style>'

def plots(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);s=read(directory/'summary.json');rows=s['metrics']
    fig,ax=plt.subplots(figsize=(11,5))
    if s['stage'].startswith('npu'):
        labels=[r['label'].replace('_',' ') for r in rows]
        ax.bar(labels,[100*r['accuracy'] for r in rows],label='Strictly correct',color=['steelblue' if not r['diagnostic_only'] else 'gray' for r in rows])
        ax.scatter(labels,[100*r['valid']/r['planned'] for r in rows],c='black',label='Valid completion');ax.axhline(95,color='green',ls='--',label='95% reference')
        ax.set_ylim(0,105);ax.set_ylabel('Percentage of all planned answers');ax.legend()
    elif s['stage'].startswith('cpu'):
        labels=[f"{r['policy']['governor']} / {r['policy']['binding']}" for r in rows]
        ax.bar(labels,[r['cpu_mean_ms'] for r in rows],color=['seagreen' if r['qualified'] else 'darkorange' for r in rows])
        ax.set_ylabel('Mean individual request ms, including transport and verification')
    else:
        ax.bar([r['label'] for r in rows],[r.get('mean_ms',{}).get('stream64',0) if isinstance(r.get('mean_ms'),dict) else r['mean_ms'] for r in rows]);ax.set_ylabel('Mean request/job ms')
    if not rows:ax.text(.5,.5,'No complete scientific matrix',ha='center',transform=ax.transAxes)
    ax.set_title(f'{s["stage"]}: {s["decision"]}');fig.tight_layout()
    fig.savefig(directory/'reliability.png',dpi=150);fig.savefig(directory/'reliability.svg');plt.close(fig)
    if s['stage'].startswith('cpu') and rows:
        fig,axes=plt.subplots(len(rows),2,figsize=(12,3.2*len(rows)),squeeze=False)
        for index,row in enumerate(rows):
            for col,kind in enumerate(('single','batch')):
                axis=axes[index,col];group=row['equivalence'][kind]
                for y,(name,value) in enumerate(group.items()):
                    lo,hi=value['interval'];axis.plot([lo,hi],[y,y],color='steelblue');axis.scatter([value['estimate']],[y],color='black')
                axis.axvspan(.95,1.05,color='green',alpha=.15);axis.axvline(1,color='gray',ls=':')
                axis.set_yticks(range(len(group)),list(group));axis.set_title(f'Policy {row["label"]}: {kind} controls');axis.set_xlabel('CPU B/A time ratio, paired 95% interval')
        fig.tight_layout();fig.savefig(directory/'equivalence.png',dpi=150);fig.savefig(directory/'equivalence.svg');plt.close(fig)

def finalize(campaign):
    from .reliability_audit import audit
    from .refine_governor import GOVERNOR
    from .research_report import seal
    from .reliability_campaign import campaign_policy, RECOVERY_PROFILE
    campaign=Path(campaign)
    if (campaign/'checksums.json').exists():raise ValueError('Sealed campaign is immutable')
    ledger=read(campaign/'campaign.json');historical=[];results=[]
    scope=campaign_policy(ledger);maximum=scope['max_seconds'];recovery=scope['profile']==RECOVERY_PROFILE
    if recovery:
        from .reliability_recovery import verify_lineage, verify_diagnostic
        verify_lineage(campaign)
        diagnostic=verify_diagnostic(campaign)
    if any(r['state']=='running' for r in ledger['attempts']):raise ValueError('Unresolved running attempt; preserve campaign for recovery')
    if any(not math.isfinite(r['charged_seconds']) or r['charged_seconds']<0 for r in ledger['attempts']):raise ValueError('Invalid budget ledger')
    for entry in read(campaign/'historical-seals.json'):
        path=ROOT/entry['path']
        if digest_file(path)!=entry['sha256'] or verify_artifacts(path.parent)!=entry['artifacts']:raise ValueError('Historical seal changed')
        historical.append(dict(path=entry['path'],artifacts=len(entry['artifacts']),unchanged=True))
    for attempt in ledger['attempts']:
        p=Path(attempt['output']);checks=verify_artifacts(p)
        if digest_file(p/'checksums.json')!=attempt['checksums_sha256']:raise ValueError('Attempt seal changed')
        if recovery and (p/'config.json').exists():
            config=read(p/'config.json')
            expected=dict(campaign_profile=scope['profile'],campaign_max_seconds=maximum,
                          allowed_stages=scope['allowed_stages'],recovery_from=ledger.get('recovery_from'),
                          recovery_lineage_sha256=ledger.get('recovery_lineage_sha256'))
            if any(config.get(key)!=value for key,value in expected.items()):raise ValueError('Frozen recovery metadata differs')
        summary=read(p/'summary.json') if (p/'summary.json').exists() else dict(
            stage=attempt['stage'],complete=False,selected=None,accepted=False,decision='missing_summary',metrics=[],defaults_changed=False)
        if attempt.get('scientific_complete') and not summary['complete']:raise ValueError('Completed attempt summary is missing or incomplete')
        restoration=read(p/'governor-restoration.json')
        if not restoration['restored']:raise ValueError('Attempt did not restore governor')
        validation=audit(p) if summary['complete'] else dict(passed=False,partial_preserved=True,reason=summary['decision'])
        if summary['complete'] and not validation['passed']:raise ValueError(f'Completed run failed final audit: {validation}')
        results.append(dict(stage=attempt['stage'],output=str(p),charged_seconds=attempt['charged_seconds'],summary=summary,audit=validation,artifacts=len(checks)))
    spent=sum(r['charged_seconds'] for r in ledger['attempts'])
    if spent>maximum or any(sum(r['charged_seconds'] for r in ledger['attempts'] if r['stage']==stage)>cap for stage,cap in CAPS.items()):raise ValueError('Hardware budget exceeded')
    if GOVERNOR.read_text().strip()!=ledger['original_governor']:raise ValueError('Final governor differs')
    validation=read(campaign/'implementation-validation.json')
    if not validation['passed']:raise ValueError('Implementation validation failed')
    negative=read(campaign/'negative-gates.json');tamper=read(campaign/'audit-tamper-tests.json')
    if not negative['passed'] or not tamper['passed']:raise ValueError('Negative audit/gate validation failed')
    summary=dict(utc=utc(),hardware_seconds=spent,max_seconds=maximum,stages=results,
        execution=read(campaign/'execution.json'),defaults_changed=False)
    final=dict(utc=utc(),passed=True,integrity_passed=True,hardware_seconds=spent,max_seconds=maximum,
        historical=historical,stages=[dict(stage=r['stage'],audit=r['audit']) for r in results],
        implementation=validation,negative_gates=negative,tamper_tests=tamper,governor_restored=True)
    if recovery:
        metadata=dict(profile=scope['profile'],allowed_stages=scope['allowed_stages'],
                      recovery_from=ledger.get('recovery_from'),recovery_lineage_sha256=ledger.get('recovery_lineage_sha256'))
        summary.update(metadata);final.update(metadata);final['diagnostic']=diagnostic
    atomic_json(campaign/'summary.json',summary);atomic_json(campaign/'final-validation.json',final)
    title='CPU / GPU reliability recovery' if recovery else 'Reliability-first CPU / GPU / NPU campaign'
    design=('User selected an 8-hour ceiling: CPU development 4h, eligible CPU confirmation 2h, eligible GPU confirmation 2h.'
            if recovery else 'User selected a 16-hour ceiling and original full-table NPU qualification.')
    evidence=('Previous NPU evidence remains historical only. No application defaults changed.' if recovery
              else 'One-fact CPU assistance is diagnostic only. No application defaults changed.')
    lines=[title,utc(),f'Charged runtime: {spent:.3f} / {maximum} seconds',
           f'Governor restored to {ledger["original_governor"]}',f'Historical sealed directories unchanged: {len(historical)}',
           design,evidence,
           'This checkpoint is on the Development SSD; previous external backups remain earlier checkpoints.']
    for r in results:lines.append(f'{r["stage"]}: {r["summary"]["decision"]}; scientific audit={r["audit"]["passed"]}; {r["charged_seconds"]:.3f}s; {r["output"]}')
    (campaign/'CONTEXT.txt').write_text('\n'.join(lines)+'\n')
    links=''.join(f'<li><a href="{html.escape(r["output"])}/report.html">{html.escape(r["stage"])}</a>: {html.escape(r["summary"]["decision"])}</li>' for r in results)
    (campaign/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Reliability campaign</title>'+STYLE+'<h1>'+html.escape(title)+'</h1><pre>'+html.escape('\n'.join(lines))+'</pre><ul>'+links+'</ul><p><a href="final-validation.json">Final validation</a> · <a href="summary.json">Results</a> · <a href="NEXT_EXPERIMENT.txt">Next experiment</a></p>')
    seal(campaign);verify_artifacts(campaign)
    return dict(campaign=str(campaign),hardware_seconds=spent,historical=len(historical),sealed_artifacts=len(read(campaign/'checksums.json')))

if __name__=='__main__':plots(sys.argv[1])
