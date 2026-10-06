"""Independent eight-hour ledger, conditional stages, and governor restoration."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from .common import ROOT,atomic_json,digest_file,utc
from .hybrid import verify_artifacts
from .research_campaign import collect_previous,fixture_namespace
from .refine_spec import VERSION,CAPS,specification
from .refine_governor import GOVERNOR,request
from .study_spec import LLAMA

def read(path):return json.loads(Path(path).read_text())

def allowance(ledger,stage,requested=None):
    cap=CAPS[stage];requested=cap if requested is None else requested
    if not math.isfinite(requested) or not 120<requested<=cap:raise ValueError('Invalid frozen stage cap')
    available=min(requested,28800-sum(r['charged_seconds'] for r in ledger['attempts']),
        cap-sum(r['charged_seconds'] for r in ledger['attempts'] if r['stage']==stage))
    if available<=120:raise ValueError('Refinement budget exhausted')
    return available

def dependency(ledger,stage):
    if any(r['stage']==stage and r.get('scientific_complete') for r in ledger['attempts']):raise ValueError('Completed scientific stage is immutable')
    parent_stage={'cpu-confirm':'cpu-develop','gpu-confirm':'cpu-confirm','npu-confirm':'npu-develop'}.get(stage)
    if not parent_stage:return None,None
    candidates=[r for r in ledger['attempts'] if r['stage']==parent_stage and r.get('scientific_complete') and r.get('audit_passed')]
    if not candidates:raise ValueError('Audited parent stage required; confirmation is closed')
    path=Path(candidates[-1]['output']);verify_artifacts(path);summary=read(path/'summary.json')
    if summary['selected'] is None:raise ValueError('No qualified parent candidate; confirmation is closed')
    return str(path),summary['selected']

def recover(ledger):
    # A missing finalized ledger cannot prove startup/restoration duration: charge the reservation.
    from .common import read_jsonl
    for row in ledger['attempts']:
        if row['state']!='running':continue
        path=Path(row['output']);complete=False;audit_passed=False
        if (path/'refine.jsonl').exists():
            events,_=read_jsonl(path/'refine.jsonl')
            complete=any(e['event']=='measurement_complete' for e in events)
        if (path/'checksums.json').exists():
            verify_artifacts(path)
            complete=read(path/'summary.json')['complete']
            audit_passed=read(path/'validation.json')['passed']
        row.update(state='interrupted',charged_seconds=row['reserved_seconds'],scientific_complete=complete,
            audit_passed=audit_passed,recovery='Full reservation charged; completed matrices cannot be rerun.')


def build():
    from .attention_native import build as native_build
    output=native_build()
    argv=['g++','-std=c++17','-O2','-fopenmp','-fPIC','-shared',str(ROOT/'native/attention/affinity-probe.cpp'),'-o',str(output/'libaffinity.so')]
    subprocess.run(argv,check=True)
    manifest=read(output/'build.json');manifest['artifacts']['libaffinity.so']=digest_file(output/'libaffinity.so')
    manifest['probe_command']=argv;atomic_json(output/'build.json',manifest)

def run(args):
    stage=args.stage;allowance(dict(attempts=[]),stage,args.max_seconds)
    directory=Path(args.output).resolve();campaign=Path(args.campaign).resolve();source=Path(args.source_campaign).resolve()
    if directory.exists():raise ValueError('Output must be a new directory')
    if (campaign/'checksums.json').exists():raise ValueError('Sealed campaign is immutable')
    verify_artifacts(source)
    if not read(source/'final-validation.json')['passed']:raise ValueError('Source campaign audit failed')
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'.lock').open('a') as lock,(ROOT/'runs/.experiment.lock').open('a') as hardware:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(hardware,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=campaign/'campaign.json'
        if path.exists():
            ledger=read(path)
            if ledger['version']!=VERSION or ledger['source_campaign']!=str(source):raise ValueError('Campaign identity differs')
            recover(ledger)
        else:
            ledger=dict(version=VERSION,campaign_id=str(uuid.uuid4()),created_utc=utc(),max_seconds=28800,
                source_campaign=str(source),original_governor=GOVERNOR.read_text().strip(),attempts=[])
            atomic_json(campaign/'registry.json',collect_previous())
        if GOVERNOR.read_text().strip()!=ledger['original_governor']:raise RuntimeError('Governor not restored; halt hardware work')
        parent,selected=dependency(ledger,stage);limit=allowance(ledger,stage,args.max_seconds)
        cpu=not stage.startswith('npu')
        if cpu:build()
        index=sum(r['stage']==stage for r in ledger['attempts'])
        directory.mkdir();atomic_json(directory/'protocol.json',specification(stage))
        atomic_json(directory/'prior-fixtures.json',read(campaign/'registry.json'))
        row=dict(stage=stage,output=str(directory),state='running',started_utc=utc(),pid=os.getpid(),
                 reserved_seconds=limit,charged_seconds=limit)
        ledger['attempts'].append(row);atomic_json(path,ledger)
        helper=None;temporary=None;log=None;socket_path=None;start=time.monotonic();supervisor_started=False
        original=ledger['original_governor'];code=1
        try:
            if cpu:
                temporary=tempfile.TemporaryDirectory(prefix='refine-governor-');socket_path=Path(temporary.name)/'lease.sock'
                log=(directory/'governor-helper.log').open('w')
                helper=subprocess.Popen(['pkexec',sys.executable,str(ROOT/'efficiency/refine_governor.py'),
                    '--socket',str(socket_path),'--pid',str(os.getpid()),'--seconds',str(limit)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                expiry=time.monotonic()+120
                while not socket_path.exists() and helper.poll() is None and time.monotonic()<expiry:time.sleep(.1)
                if not socket_path.exists():raise PermissionError('Desktop administrator authentication unavailable; CPU branch dependency blocked')
                lease=request(socket_path)
                if lease['original']!=original:raise RuntimeError('Governor baseline changed')
            config=dict(profile='refine',phase='cpu' if cpu else 'hat',refine_stage=stage,
                campaign=str(campaign),source_campaign=str(source),campaign_id=ledger['campaign_id'],
                max_seconds=limit-(time.monotonic()-start),reserved_seconds=limit,baseline_seconds=15,cooldown_seconds=15,
                cleanup_reserve_seconds=120,attempt_index=index,
                fixture_namespace=fixture_namespace(ledger['campaign_id'],stage,index),
                model=str(LLAMA),parent=parent,selected=selected,governor_socket=str(socket_path) if cpu else None,
                original_governor=original)
            if config['max_seconds']<=120:raise TimeoutError('Startup exhausted work allowance')
            atomic_json(directory/'config.json',config);os.environ['HAILORT_LOGGER_PATH']=str(directory)
            from .runner import Supervisor
            supervisor_started=True;code=Supervisor(directory,config).execute()
        except Exception as exc:
            atomic_json(directory/'startup-failure.json',dict(error=f'{type(exc).__name__}: {exc}',utc=utc()))
            print(str(exc),flush=True)
        finally:
            restoration_error=None
            if socket_path and socket_path.exists():
                try:request(socket_path,close=True)
                except Exception as exc:restoration_error=str(exc)
            if helper:
                try:helper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    helper.terminate()
                    try:helper.wait(timeout=5)
                    except subprocess.TimeoutExpired:restoration_error='Privileged helper did not exit; watchdog lease remains active'
            if log:log.close()
            if temporary and (helper is None or helper.poll() is not None):temporary.cleanup()
            current=GOVERNOR.read_text().strip()
            restored=current==original and (helper is None or helper.poll() is not None)
            atomic_json(directory/'governor-restoration.json',dict(original=original,current=current,restored=restored,error=restoration_error))
            outcome=read(directory/'outcome.json') if (directory/'outcome.json').exists() else {}
            row.update(state=outcome.get('status','dependency_blocked'),charged_seconds=time.monotonic()-start,restored=restored)
            atomic_json(path,ledger)
        from .research_report import seal
        if supervisor_started and (directory/'config.json').exists():
            from .refine_report import build_report
            from .refine_audit import audit
            summary=build_report(directory);validation=audit(directory,require_seal=False)
        else:
            summary=dict(complete=False,selected=None,accepted=False,decision='dependency_blocked',metrics=[],stage=stage)
            validation=dict(passed=False,error='Administrator dependency unavailable; no hardware matrix started')
            atomic_json(directory/'summary.json',summary)
        atomic_json(directory/'validation.json',validation)
        (directory/'CONTEXT.txt').write_text(f"{utc()}\nStage: {stage}\nDecision: {summary['decision']}\nHardware/startup seconds: {row['charged_seconds']}\nAudit: {validation['passed']}\nGovernor restored: {restored}\nNo application defaults changed.\n")
        seal(directory);row.update(audit_passed=validation['passed'],scientific_complete=summary['complete'],checksums_sha256=digest_file(directory/'checksums.json'))
        atomic_json(path,ledger)
        print(json.dumps(dict(output=str(directory),summary=summary,validation=validation),indent=2),flush=True)
        if not restored:raise RuntimeError('Governor restoration failed; halt further hardware work')
        return code or (0 if validation['passed'] else 1)
