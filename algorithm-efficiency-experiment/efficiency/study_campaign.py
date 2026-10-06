"""Durable eight-hour ledger for independent GPU and NPU stages."""
import fcntl
import json
import math
import os
from pathlib import Path
import uuid
from .common import ROOT,atomic_json,digest_file,utc
from .hybrid import verify_artifacts
from .research_campaign import collect_previous,fixture_namespace,recover
from .study_spec import VERSION,specification


def read(path):return json.loads(Path(path).read_text())


def allowance(ledger,track,stage,requested):
    specification(track,stage)
    if not math.isfinite(requested) or not 120<requested<=7200:raise ValueError('Stage cap must be >120 and <=7200 seconds')
    rows=ledger['attempts'];available=min(requested,28800-sum(r['charged_seconds'] for r in rows),
        14400-sum(r['charged_seconds'] for r in rows if r['track']==track),
        7200-sum(r['charged_seconds'] for r in rows if r['track']==track and r['stage']==stage))
    if available<=120:raise ValueError('Study budget exhausted')
    return available


def run(args):
    allowance(dict(attempts=[]),args.track,args.stage,args.max_seconds)
    directory=Path(args.output).resolve();campaign=Path(args.campaign).resolve()
    if directory.exists():raise ValueError('Study output must be a new directory')
    if (campaign/'checksums.json').exists():raise ValueError('Sealed campaign is immutable')
    campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'.lock').open('a') as lock,(ROOT/'runs/.experiment.lock').open('a') as hardware:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(hardware,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=campaign/'campaign.json'
        if path.exists():
            ledger=read(path)
            if ledger['version']!=VERSION:raise ValueError('Wrong campaign version')
            recover(ledger)
        else:
            if args.stage!='develop':raise ValueError('Development required first')
            ledger=dict(version=VERSION,campaign_id=str(uuid.uuid4()),created_utc=utc(),max_seconds=28800,attempts=[])
            atomic_json(campaign/'registry.json',collect_previous())
        atomic_json(path,ledger)
        done=[r for r in ledger['attempts'] if r['track']==args.track and r.get('scientific_complete')]
        if any(r['stage']==args.stage for r in done):raise ValueError('Completed scientific stage is immutable')
        parent=None;selected=None
        if args.stage=='confirm':
            eligible=[r for r in done if r['stage']=='develop' and r.get('audit_passed')]
            if not eligible:raise ValueError('Audited development required')
            parent=Path(eligible[-1]['output']);verify_artifacts(parent)
            if not read(parent/'validation.json')['passed']:raise ValueError('Development audit failed')
            selected=read(parent/'summary.json')['selected']
            if selected is None:raise ValueError('No qualified development candidate; confirmation is closed')
        limit=allowance(ledger,args.track,args.stage,args.max_seconds)
        index=sum(r['track']==args.track and r['stage']==args.stage for r in ledger['attempts'])
        config=dict(profile='study',phase='cpu' if args.track=='gpu-stream' else 'hat',track=args.track,study_stage=args.stage,
            campaign=str(campaign),campaign_id=ledger['campaign_id'],max_seconds=limit,baseline_seconds=15,cooldown_seconds=15,
            cleanup_reserve_seconds=120,attempt_index=index,
            fixture_namespace=fixture_namespace(ledger['campaign_id'],args.track+'-'+args.stage,index),
            model=str(Path(args.model).resolve()),challenger_model=str(Path(args.challenger_model).resolve()),
            parent=str(parent) if parent else None,selected=selected)
        # Compilation is software preparation, outside the hardware reservation.
        if args.track=='gpu-stream':
            from .attention_native import build
            build()
        directory.mkdir(parents=True)
        atomic_json(directory/'config.json',config);atomic_json(directory/'protocol.json',specification(args.track,args.stage))
        atomic_json(directory/'prior-fixtures.json',read(campaign/'registry.json'))
        row=dict(track=args.track,stage=args.stage,output=str(directory),state='running',started_utc=utc(),pid=os.getpid(),
                 reserved_seconds=limit,charged_seconds=limit)
        ledger['attempts'].append(row);atomic_json(path,ledger)
        os.environ['HAILORT_LOGGER_PATH']=str(directory)
        from .runner import Supervisor
        try:code=Supervisor(directory,config).execute()
        finally:
            outcome=read(directory/'outcome.json') if (directory/'outcome.json').exists() else {}
            row.update(state=outcome.get('status','interrupted'),charged_seconds=outcome.get('elapsed_seconds',limit))
            atomic_json(path,ledger)
        from .study_report import build_report
        from .study_audit import audit
        from .research_report import seal
        summary=build_report(directory);validation=audit(directory,require_seal=False)
        atomic_json(directory/'validation.json',validation)
        (directory/'CONTEXT.txt').write_text(f"{utc()}\nTrack: {args.track}; stage: {args.stage}\nDecision: {summary['decision']}\nHardware seconds: {row['charged_seconds']}\nAudit: {validation['passed']}\nNo application defaults changed. Read protocol.json, report.html, and validation.json.\n")
        seal(directory)
        row.update(audit_passed=validation['passed'],scientific_complete=summary['complete'],
                   checksums_sha256=digest_file(directory/'checksums.json'))
        atomic_json(path,ledger)
        print(json.dumps(dict(output=str(directory),summary=summary,validation=validation),indent=2),flush=True)
        return code or (0 if validation['passed'] else 1)
