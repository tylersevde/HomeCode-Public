"""Durable, bounded research campaigns; immutable stages and audited parent links."""
import fcntl
import json
import math
import os
from pathlib import Path
import time
import uuid

from .common import ROOT,atomic_json,digest_file,utc
from .research_spec import STAGES,specification


def read(path):return json.loads(Path(path).read_text())


def fixture_namespace(campaign_id,stage,attempt):
    if not isinstance(attempt,int) or attempt<0:raise ValueError('Invalid attempt index')
    return campaign_id if attempt==0 else f'{campaign_id}|{stage}|attempt-{attempt}'


def allowance(ledger,stage,requested):
    if not math.isfinite(requested) or not 120<requested<=7200:raise ValueError('Session limit must be finite, greater than 120 and at most 7200 seconds')
    spent=sum(r['charged_seconds'] for r in ledger['attempts'])
    stage_spent=sum(r['charged_seconds'] for r in ledger['attempts'] if r['stage']==stage)
    available=min(requested,36000-spent,7200-stage_spent)
    if available<=120:raise ValueError('Campaign or stage budget exhausted')
    return available


def collect_previous():
    seeds=set();hashes=set();sources=[]
    def visit(value):
        if isinstance(value,dict):
            if isinstance(value.get('seed'),int):seeds.add(value['seed'])
            for k in ('input_sha256','table_sha256'):
                if isinstance(value.get(k),str):hashes.add(value[k])
            if 'table' in value:
                from .common import digest_value
                hashes.add(digest_value(value['table']))
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    for run in sorted((ROOT/'runs').iterdir()):
        if not run.is_dir():continue
        checksum=run/'checksums.json'
        if checksum.exists():sources.append(dict(path=str(run),checksums_sha256=digest_file(checksum)))
        for pattern in ('*fixtures.json','diagnostic-inputs.json'):
            for path in run.glob(pattern):visit(read(path))
    return dict(seeds=sorted(seeds),hashes=sorted(hashes),historical_sources=sources,registered=[])


def register_fixture(campaign,stage,fid,seed,value_hash):
    path=Path(campaign)/'registry.json';registry=read(path)
    if seed in registry['seeds'] or value_hash in registry['hashes']:raise ValueError('Fixture reuses previously observed data')
    registry['seeds'].append(seed);registry['hashes'].append(value_hash)
    registry['registered'].append(dict(stage=stage,fixture_id=fid,seed=seed,sha256=value_hash))
    atomic_json(path,registry)


def index(campaign,ledger):
    lines=['# Attention research campaign','',f"Campaign: {ledger['campaign_id']}",
        f"Hardware time charged: {sum(r['charged_seconds'] for r in ledger['attempts']):.3f} / 36000 seconds",'',
        '| Stage | Outcome | Audit | Hardware seconds | Run |','| --- | --- | --- | ---: | --- |']
    for r in ledger['attempts']:
        lines.append(f"| {r['stage']} | {r['state']} | {r.get('audit_passed','pending')} | {r['charged_seconds']:.3f} | [{Path(r['output']).name}]({r['output']}/report.html) |")
    lines+=['','Historical next-experiment files are preserved. This ledger records their current descendants.',
        'CPU remains the default until a confirmed policy is explicitly enabled.','']
    (campaign/'INDEX.md').write_text('\n'.join(lines))


def recover(ledger):
    for row in ledger['attempts']:
        if row['state']!='running':continue
        # The exclusive campaign lock proves no live campaign supervisor owns this entry.
        outcome=Path(row['output'])/'outcome.json'
        elapsed=read(outcome)['elapsed_seconds'] if outcome.exists() else row['reserved_seconds']
        if not math.isfinite(elapsed) or elapsed<0:elapsed=row['reserved_seconds']
        row.update(state='interrupted',charged_seconds=max(0,elapsed),audit_passed=False,
                   recovery='Outcome duration if present; otherwise conservatively charge full reservation.')


def run(args):
    campaign=Path(args.campaign).resolve();campaign.mkdir(parents=True,exist_ok=True)
    with (campaign/'.lock').open('a') as lock,(ROOT/'runs/.experiment.lock').open('a') as hardware:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(hardware,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A campaign or experiment is already running')
        path=campaign/'campaign.json'
        if path.exists():ledger=read(path);recover(ledger)
        else:
            ledger=dict(version=1,campaign_id=str(uuid.uuid4()),created_utc=utc(),max_seconds=36000,attempts=[])
            atomic_json(campaign/'registry.json',collect_previous())
        atomic_json(path,ledger)
        if any(r['stage']==args.stage and r.get('audit_passed') for r in ledger['attempts']):
            raise ValueError('Stage already sealed; create a new campaign for replication')
        parents={}
        for stage in STAGES[:STAGES.index(args.stage)]:
            candidates=[r for r in ledger['attempts'] if r['stage']==stage and r.get('audit_passed')]
            if not candidates:raise ValueError(f'An audited {stage} stage is required first')
            parent=Path(candidates[-1]['output'])
            from .hybrid import verify_artifacts
            verify_artifacts(parent)
            if not read(parent/'validation.json')['passed']:raise ValueError('Parent audit failed')
            parents[stage]=str(parent)
        limit=allowance(ledger,args.stage,args.max_seconds)
        directory=Path(args.output).resolve();directory.mkdir(parents=True,exist_ok=False)
        config=dict(profile='research',phase='hat',research_stage=args.stage,campaign=str(campaign),
            campaign_id=ledger['campaign_id'],max_seconds=limit,baseline_seconds=15,cooldown_seconds=15,
            model=str(Path(args.model).resolve()),secondary_model=str(Path(args.secondary_model).resolve()) if args.secondary_model else None,
            parents=parents,cleanup_reserve_seconds=120)
        attempt_index=sum(r['stage']==args.stage for r in ledger['attempts'])
        config.update(attempt_index=attempt_index,fixture_namespace=fixture_namespace(ledger['campaign_id'],args.stage,attempt_index))
        atomic_json(directory/'config.json',config);atomic_json(directory/'protocol.json',specification(args.stage))
        atomic_json(directory/'prior-fixtures.json',read(campaign/'registry.json'))
        attempt=dict(stage=args.stage,output=str(directory),state='running',started_utc=utc(),
            reserved_seconds=limit,charged_seconds=limit,pid=os.getpid())
        ledger['attempts'].append(attempt);atomic_json(path,ledger);index(campaign,ledger)
        os.environ['HAILORT_LOGGER_PATH']=str(directory)
        from .runner import Supervisor
        try:
            code=Supervisor(directory,config).execute()
        finally:
            outcome=read(directory/'outcome.json') if (directory/'outcome.json').exists() else {}
            attempt.update(state=outcome.get('status','interrupted'),charged_seconds=outcome.get('elapsed_seconds',limit))
            atomic_json(path,ledger);index(campaign,ledger)
        from .research_report import build_report,seal
        from .research_audit import audit
        build_report(directory)
        result=audit(directory)
        atomic_json(directory/'validation.json',result);seal(directory)
        attempt.update(audit_passed=result['passed'],checksums_sha256=digest_file(directory/'checksums.json'))
        atomic_json(path,ledger);index(campaign,ledger)
        print(json.dumps(dict(output=str(directory),audit=result,spent_seconds=sum(r['charged_seconds'] for r in ledger['attempts'])),indent=2),flush=True)
        return code if code else (0 if result['passed'] else 1)
