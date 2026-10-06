"""Frozen campaign budgets, conditional stages and governor restoration."""
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
from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .hybrid import verify_artifacts
from .research_campaign import collect_previous, fixture_namespace
from .refine_campaign import build
from .refine_governor import GOVERNOR, request
from .reliability_spec import VERSION, EVENTS, CAPS, MAX_SECONDS, DEPENDENCIES, specification
from .study_spec import LLAMA

def read(p):return json.loads(Path(p).read_text())

RECOVERY_PROFILE = 'cpu-gpu-recovery'
RECOVERY_STAGES = ('cpu-develop','cpu-confirm','gpu-confirm')
RECOVERY_SECONDS = 28800

def campaign_policy(ledger):
    """Validate a campaign's scope without opening hardware or creating files."""
    profile=ledger.get('profile','reliability')
    if profile=='reliability':
        if any(key in ledger for key in ('recovery_from','recovery_lineage_sha256','allowed_stages')):
            raise ValueError('Recovery campaign profile is missing or changed')
        return dict(profile=profile,max_seconds=MAX_SECONDS,allowed_stages=list(CAPS))
    if profile!=RECOVERY_PROFILE:raise ValueError('Unknown frozen campaign profile')
    if ledger.get('max_seconds')!=RECOVERY_SECONDS or ledger.get('allowed_stages')!=list(RECOVERY_STAGES):
        raise ValueError('Recovery campaign scope or budget differs')
    if any(row['stage'] not in RECOVERY_STAGES for row in ledger['attempts']):
        raise ValueError('Attempt is outside the frozen campaign scope')
    return dict(profile=profile,max_seconds=RECOVERY_SECONDS,allowed_stages=list(RECOVERY_STAGES))

def require_stage(ledger,stage):
    policy=campaign_policy(ledger)
    if stage not in policy['allowed_stages']:raise ValueError('Stage is outside the frozen campaign scope')
    return policy

def allowance(ledger,stage,requested=None):
    policy=require_stage(ledger,stage)
    cap=CAPS[stage];requested=cap if requested is None else requested
    if not math.isfinite(requested) or not 120<requested<=cap:raise ValueError('Invalid frozen stage cap')
    for row in ledger['attempts']:
        if not math.isfinite(row['charged_seconds']) or row['charged_seconds']<0:raise ValueError('Invalid budget ledger')
    result=min(requested,policy['max_seconds']-sum(r['charged_seconds'] for r in ledger['attempts']),
        cap-sum(r['charged_seconds'] for r in ledger['attempts'] if r['stage']==stage))
    if result<=120:raise ValueError('Campaign or stage budget exhausted')
    return result

def dependency(ledger,stage):
    require_stage(ledger,stage)
    if any(r['stage']==stage and (r.get('scientific_complete') or r.get('closed')) for r in ledger['attempts']):
        raise ValueError('Completed or deferred scientific stage is immutable')
    parents={};selected=None;policy=None
    for name in DEPENDENCIES.get(stage,()):
        candidates=[r for r in ledger['attempts'] if r['stage']==name and r.get('scientific_complete') and r.get('audit_passed')]
        if not candidates:raise ValueError('Audited parent required; dependent stage is closed')
        p=Path(candidates[-1]['output']);verify_artifacts(p);s=read(p/'summary.json')
        if s['selected'] is None:raise ValueError('No qualified parent; dependent stage is closed')
        parents[name]=str(p)
        if name.startswith('cpu'):policy=s['selected']
        if stage=='combined-confirm':
            selected=s['selected'];policy=read(p/'config.json')['cpu_policy']
        elif stage in ('cpu-confirm','gpu-confirm','npu-confirm'):selected=s['selected']
    return parents,selected,policy

def recover(ledger):
    for row in ledger['attempts']:
        if row['state']!='running':continue
        p=Path(row['output']);events,_=read_jsonl(p/EVENTS)
        complete=any(e['event']=='measurement_complete' for e in events)
        closed=complete or any(e['event']=='pilot_gate' and not e['fits'] for e in events)
        passed=False
        if (p/'checksums.json').exists():
            verify_artifacts(p)
            s=read(p/'summary.json') if (p/'summary.json').exists() else {}
            complete=complete or s.get('complete',False);closed=closed or complete
            passed=read(p/'validation.json').get('passed',False) if (p/'validation.json').exists() else False
            row['checksums_sha256']=digest_file(p/'checksums.json')
        row.update(state='interrupted',charged_seconds=row['reserved_seconds'],scientific_complete=complete,closed=closed,audit_passed=passed,
                   recovery='Full reservation charged; completed/deferred scientific stages remain closed.')

def historical_seals():
    return [dict(path=str(p.relative_to(ROOT)),sha256=digest_file(p),artifacts=verify_artifacts(p.parent))
            for base in ('runs','campaigns') for p in sorted((ROOT/base).glob('*/checksums.json'))]

def run(args):
    stage=args.stage
    directory=Path(args.output).resolve();campaign=Path(args.campaign).resolve();source=Path(args.source_campaign).resolve()
    if directory.exists():raise ValueError('Output must be a new directory')
    if (campaign/'checksums.json').exists():raise ValueError('Sealed campaign is immutable')
    path=campaign/'campaign.json'
    preflight=read(path) if path.exists() else dict(attempts=[])
    require_stage(preflight,stage)
    # Scope rejection happens before source verification, lock files or hardware.
    allowance(preflight,stage,args.max_seconds)
    verify_artifacts(source);validation=read(source/'final-validation.json')
    if not validation.get('passed',validation.get('integrity_passed',False)):raise ValueError('Source audit failed')
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'.lock').open('a') as lock,(ROOT/'runs/.experiment.lock').open('a') as hardware:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(hardware,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            ledger=read(path)
            if ledger['version']!=VERSION or ledger['source_campaign']!=str(source):raise ValueError('Campaign identity differs')
            require_stage(ledger,stage)
            recover(ledger)
            atomic_json(path,ledger)
        else:
            ledger=dict(version=VERSION,campaign_id=str(uuid.uuid4()),created_utc=utc(),max_seconds=MAX_SECONDS,
                        source_campaign=str(source),original_governor=GOVERNOR.read_text().strip(),attempts=[])
            atomic_json(campaign/'historical-seals.json',historical_seals())
            atomic_json(campaign/'registry.json',collect_previous())
        if GOVERNOR.read_text().strip()!=ledger['original_governor']:raise RuntimeError('Governor not restored; halt hardware work')
        parents,selected,policy=dependency(ledger,stage);limit=allowance(ledger,stage,args.max_seconds)
        index=sum(r['stage']==stage for r in ledger['attempts']);needs_governor=not stage.startswith('npu')
        directory.mkdir();atomic_json(directory/'protocol.json',specification(stage));atomic_json(directory/'prior-fixtures.json',read(campaign/'registry.json'))
        row=dict(stage=stage,output=str(directory),state='running',started_utc=utc(),pid=os.getpid(),reserved_seconds=limit,charged_seconds=limit)
        ledger['attempts'].append(row);atomic_json(path,ledger)
        helper=None;temporary=None;log=None;socket_path=None;start=time.monotonic();supervisor_started=False
        original=ledger['original_governor'];code=1
        try:
            if needs_governor:
                build()
                temporary=tempfile.TemporaryDirectory(prefix='reliability-governor-');socket_path=Path(temporary.name)/'lease.sock'
                log=(directory/'governor-helper.log').open('w')
                seconds=limit-(time.monotonic()-start)
                helper=subprocess.Popen(['pkexec',sys.executable,str(ROOT/'efficiency/refine_governor.py'),
                    '--socket',str(socket_path),'--pid',str(os.getpid()),'--seconds',str(seconds)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                expiry=min(start+limit-120,time.monotonic()+120)
                while not socket_path.exists() and helper.poll() is None and time.monotonic()<expiry:time.sleep(.1)
                if not socket_path.exists():raise PermissionError('Desktop administrator authentication unavailable; numerical branch dependency blocked')
                if request(socket_path)['original']!=original:raise RuntimeError('Governor baseline changed')
            config=dict(profile='reliability',phase='hat' if stage.startswith(('npu','combined')) else 'cpu',reliability_stage=stage,
                campaign=str(campaign),source_campaign=str(source),campaign_id=ledger['campaign_id'],max_seconds=limit-(time.monotonic()-start),
                reserved_seconds=limit,baseline_seconds=15,cooldown_seconds=15,cleanup_reserve_seconds=120,attempt_index=index,
                fixture_namespace=fixture_namespace(ledger['campaign_id'],stage,index),model=str(LLAMA),parents=parents,
                selected=selected,cpu_policy=policy,governor_socket=str(socket_path) if needs_governor else None,original_governor=original)
            scope=campaign_policy(ledger)
            if scope['profile']==RECOVERY_PROFILE:
                config.update(campaign_profile=scope['profile'],campaign_max_seconds=scope['max_seconds'],
                              allowed_stages=scope['allowed_stages'],recovery_from=ledger.get('recovery_from'),
                              recovery_lineage_sha256=ledger.get('recovery_lineage_sha256'))
            if config['max_seconds']<=120:raise TimeoutError('Startup exhausted work allowance')
            atomic_json(directory/'config.json',config);os.environ['HAILORT_LOGGER_PATH']=str(directory)
            from .runner import Supervisor
            supervisor_started=True;code=Supervisor(directory,config).execute()
        except Exception as exc:
            atomic_json(directory/'startup-failure.json',dict(error=f'{type(exc).__name__}: {exc}',utc=utc()));print(str(exc),flush=True)
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
                    except subprocess.TimeoutExpired:restoration_error='Privileged helper remains active; watchdog will restore'
            if log:log.close()
            if temporary and (helper is None or helper.poll() is not None):temporary.cleanup()
            current=GOVERNOR.read_text().strip();restored=current==original and (helper is None or helper.poll() is not None)
            atomic_json(directory/'governor-restoration.json',dict(original=original,current=current,restored=restored,error=restoration_error))
            outcome=read(directory/'outcome.json') if (directory/'outcome.json').exists() else {}
            # prepare_source updates the frozen source identity; retain it when finalizing the attempt.
            latest=read(path)
            if 'source_sha256' in latest:ledger['source_sha256']=latest['source_sha256']
            row.update(state=outcome.get('status','dependency_blocked'),charged_seconds=time.monotonic()-start,restored=restored)
            atomic_json(path,ledger)
        from .research_report import seal
        if supervisor_started and (directory/'config.json').exists():
            from .reliability_report import build_report
            from .reliability_audit import audit
            try:
                summary=build_report(directory);validation=audit(directory,require_seal=False)
            except Exception as exc:
                failure=dict(error=f'{type(exc).__name__}: {exc}',utc=utc())
                atomic_json(directory/'report-failure.json',failure)
                summary=read(directory/'summary.json') if (directory/'summary.json').exists() else dict(
                    stage=stage,complete=False,selected=None,accepted=False,decision='incomplete',metrics=[],defaults_changed=False)
                atomic_json(directory/'summary.json',summary)
                validation=dict(passed=False,error='Report finalization failed: '+failure['error']);code=code or 1
        else:
            summary=dict(stage=stage,complete=False,selected=None,accepted=False,decision='dependency_blocked',metrics=[],defaults_changed=False)
            validation=dict(passed=False,error='Startup dependency unavailable; no supervised matrix began')
            atomic_json(directory/'summary.json',summary)
        atomic_json(directory/'validation.json',validation)
        (directory/'CONTEXT.txt').write_text(f'{utc()}\nStage: {stage}\nDecision: {summary["decision"]}\nCharged seconds: {row["charged_seconds"]}\nAudit passed: {validation["passed"]}\nGovernor restored: {restored}\nNo application defaults changed.\n')
        seal(directory)
        events,_=read_jsonl(directory/EVENTS)
        row.update(audit_passed=validation['passed'],scientific_complete=summary['complete'],
                   closed=summary['complete'] or any(e['event']=='pilot_gate' and not e['fits'] for e in events),checksums_sha256=digest_file(directory/'checksums.json'))
        atomic_json(path,ledger)
        print(json.dumps(dict(output=str(directory),summary=summary,validation=validation),indent=2),flush=True)
        if not restored:raise RuntimeError('Governor restoration failed; halt further hardware work')
        return code or (0 if validation['passed'] else 1)
