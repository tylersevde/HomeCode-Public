"""Durable two-stage quality campaign with four cumulative hardware hours."""
import fcntl
import json
import math
import os
from pathlib import Path
import uuid

from .common import ROOT, atomic_json, digest_file, utc
from .hybrid import verify_artifacts
from .quality_spec import VERSION, conditions, specification
from .research_campaign import collect_previous, fixture_namespace, recover


def read(path):
    return json.loads(Path(path).read_text())


def allowance(ledger, stage, requested):
    if stage not in ('develop','confirm') or not math.isfinite(requested) or not 120 < requested <= 7200:
        raise ValueError('Quality stage cap must be finite, greater than 120 and at most 7200 seconds')
    available = min(requested, 14400-sum(r['charged_seconds'] for r in ledger['attempts']),
                    7200-sum(r['charged_seconds'] for r in ledger['attempts'] if r['stage']==stage))
    if available <= 120:
        raise ValueError('Quality campaign or stage budget exhausted')
    return available


def run(args):
    # Reject bad input before creating a campaign or touching the HAT.
    allowance(dict(attempts=[]), args.stage, args.max_seconds)
    directory = Path(args.output).resolve()
    if directory.exists():
        raise ValueError('Quality output must be a new directory')
    campaign = Path(args.campaign).resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign/'.lock').open('a') as lock, (ROOT/'runs/.experiment.lock').open('a') as hardware:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(hardware, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = campaign/'campaign.json'
        if path.exists():
            ledger = read(path)
            if ledger['version'] != VERSION:
                raise ValueError('Campaign is not a quality campaign')
            recover(ledger)
        else:
            if args.stage != 'develop':
                raise ValueError('Development must precede confirmation')
            ledger = dict(version=VERSION, campaign_id=str(uuid.uuid4()), created_utc=utc(),
                          max_seconds=14400, attempts=[])
            atomic_json(campaign/'registry.json', collect_previous())
        atomic_json(path, ledger)
        if any(r['stage']==args.stage and r.get('audit_passed') for r in ledger['attempts']):
            raise ValueError('Completed stage is immutable; no retries after a scientific result')
        parent, selected = None, None
        if args.stage == 'confirm':
            eligible = [r for r in ledger['attempts'] if r['stage']=='develop' and r.get('audit_passed')]
            if not eligible:
                raise ValueError('Audited development required')
            parent = Path(eligible[-1]['output'])
            verify_artifacts(parent)
            if not read(parent/'validation.json')['passed']:
                raise ValueError('Development audit failed')
            selected = read(parent/'summary.json')['selected']
            if selected is None:
                raise ValueError('No qualified development condition; confirmation is closed')
        limit = allowance(ledger, args.stage, args.max_seconds)
        index = sum(r['stage']==args.stage for r in ledger['attempts'])
        config = dict(profile='quality', phase='hat', quality_stage=args.stage, campaign=str(campaign),
                      campaign_id=ledger['campaign_id'], max_seconds=limit, baseline_seconds=15,
                      cooldown_seconds=15, cleanup_reserve_seconds=120, attempt_index=index,
                      fixture_namespace=fixture_namespace(ledger['campaign_id'],args.stage,index),
                      model=str(Path(args.model).resolve()), selected=selected,
                      conditions=conditions(args.stage,selected), parent=str(parent) if parent else None)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory/'config.json',config)
        atomic_json(directory/'protocol.json',specification(args.stage))
        atomic_json(directory/'prior-fixtures.json',read(campaign/'registry.json'))
        row = dict(stage=args.stage, output=str(directory), state='running', started_utc=utc(),
                   reserved_seconds=limit, charged_seconds=limit, pid=os.getpid())
        ledger['attempts'].append(row)
        atomic_json(path,ledger)
        os.environ['HAILORT_LOGGER_PATH']=str(directory)
        from .runner import Supervisor
        try:
            code = Supervisor(directory,config).execute()
        finally:
            outcome = read(directory/'outcome.json') if (directory/'outcome.json').exists() else {}
            row.update(state=outcome.get('status','interrupted'),charged_seconds=outcome.get('elapsed_seconds',limit))
            atomic_json(path,ledger)
        from .quality_report import build_report
        from .quality_audit import audit
        from .research_report import seal
        build_report(directory)
        validation = audit(directory, require_seal=False)
        atomic_json(directory/'validation.json',validation)
        seal(directory)
        row.update(audit_passed=validation['passed'],checksums_sha256=digest_file(directory/'checksums.json'))
        atomic_json(path,ledger)
        print(json.dumps(dict(output=str(directory),validation=validation,
                              charged_seconds=sum(r['charged_seconds'] for r in ledger['attempts'])),indent=2))
        return code or (0 if validation['passed'] else 1)
