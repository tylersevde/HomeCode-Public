"""Durable, bounded recovery orchestration independent of the launching terminal.

Public commands: start, status, stop. Internal commands are service entrypoints.
Runtime logs and stop receipts live outside the sealable campaign.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .common import ROOT, atomic_json, digest_file, utc

STAGES = ['cpu-develop', 'cpu-confirm', 'gpu-confirm']
MAX_SECONDS = 28800
SMOKE_SECONDS = 1200
STATE_ROOT = Path.home() / 'Documents' / 'reliability-services'


def read(path):
    return json.loads(Path(path).read_text())


def state_dir(campaign):
    return STATE_ROOT / Path(campaign).name


def unit_name(campaign):
    name = Path(campaign).name
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Campaign name must contain only letters, digits, dashes or underscores')
    return 'algorithm-efficiency-' + name + '.service'


def update(campaign, phase, **values):
    directory = state_dir(campaign)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'status.json', dict(utc=utc(), campaign=str(campaign),
                unit=unit_name(campaign), phase=phase, **values))


def service_properties(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=ActiveState,SubState,Result,MainPID,ControlGroup,InvocationID,ExecMainCode,ExecMainStatus'],
        text=True, capture_output=True)
    if result.returncode:
        return dict(query_error=result.stderr.strip())
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


@contextmanager
def exclusive(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def build_service_command(campaign):
    campaign = Path(campaign).resolve()
    directory = state_dir(campaign)
    stop_command = f'{sys.executable} -B -m efficiency.reliability_service _stopped --campaign {campaign}'
    # Project and campaign paths on this host are fixed, without systemd argument metacharacters.
    if any(c in str(campaign) + str(ROOT) + str(directory) + sys.executable for c in ' \t\n%$"\\'):
        raise ValueError('Service paths contain unsupported argument characters')
    return ['systemd-run', '--user', '--unit=' + unit_name(campaign), '--service-type=exec',
            '--expand-environment=no', '--working-directory=' + str(ROOT),
            '--property=Restart=no', '--property=KillMode=control-group',
            '--property=TimeoutStopSec=120', '--property=RuntimeMaxSec=9h',
            '--property=StandardOutput=append:' + str(directory / 'service.log'),
            '--property=StandardError=append:' + str(directory / 'service.log'),
            '--property=ExecStopPost=' + stop_command,
            '--setenv=PYTHONDONTWRITEBYTECODE=1', sys.executable, '-B', '-m',
            'efficiency.reliability_service', '_run', '--campaign', str(campaign)]


def initialize(campaign, source, previous, validation_file):
    from .hybrid import verify_artifacts
    from .refine_governor import GOVERNOR
    from .reliability_campaign import historical_seals
    from .reliability_recovery import capture_lineage, verify_source
    from .reliability_spec import VERSION
    from .research_campaign import collect_previous
    campaign, source, previous = (Path(p).resolve() for p in (campaign, source, previous))
    if campaign.parent != ROOT / 'campaigns':
        raise ValueError('Recovery campaign must be a new project campaigns directory')
    unit_name(campaign)
    if campaign.exists() or state_dir(campaign).exists():
        raise ValueError('Campaign and service state must be new; automatic retries are disabled')
    validation = read(validation_file)
    verify_source(validation)
    verify_artifacts(source)
    source_audit = read(source / 'final-validation.json')
    if not source_audit.get('passed', source_audit.get('integrity_passed', False)):
        raise ValueError('Source campaign audit failed')
    if not (previous / 'campaign.json').is_file():
        raise ValueError('Recovery source ledger is missing')
    manager = subprocess.run(['systemctl', '--user', 'is-system-running'], text=True, capture_output=True)
    if manager.stdout.strip() not in ('running', 'degraded'):
        raise RuntimeError('User service manager is unavailable: ' + manager.stdout.strip())
    with exclusive(ROOT / 'runs/.reliability-service.lock'), exclusive(ROOT / 'runs/.experiment.lock'):
        campaign.mkdir()
        state_dir(campaign).mkdir(parents=True)
        lineage = capture_lineage(campaign, previous)
        ledger = dict(version=VERSION, campaign_id=str(uuid.uuid4()), created_utc=utc(),
                      profile='cpu-gpu-recovery', max_seconds=MAX_SECONDS, allowed_stages=STAGES,
                      source_campaign=str(source), recovery_from=str(previous),
                      recovery_lineage_sha256=lineage,
                      original_governor=GOVERNOR.read_text().strip(), attempts=[])
        atomic_json(campaign / 'campaign.json', ledger)
        atomic_json(campaign / 'historical-seals.json', historical_seals())
        atomic_json(campaign / 'registry.json', collect_previous())
        atomic_json(campaign / 'implementation-validation.json', validation)
        atomic_json(campaign / 'authorization.json', dict(utc=utc(),
            instruction='Implement the plan: repair, validation, CPU plus eligible GPU',
            stages=STAGES, max_seconds=MAX_SECONDS, diagnostic_max_seconds=SMOKE_SECONDS,
            automatic_retries=False, defaults_changed=False))
        atomic_json(campaign / 'execution.json', [])
        (campaign / 'CONTEXT.txt').write_text(
            'New CPU/GPU recovery campaign. Historical campaign remains unchanged.\n'
            'Diagnostic cap: 20 minutes; CPU development/confirmation/GPU confirmation: 4/2/2 hours.\n'
            'Only audited qualifying parents advance. No NPU or combined work.\n'
            f'Status: {state_dir(campaign) / "status.json"}\n')
        (campaign / 'NEXT_EXPERIMENT.txt').write_text(
            'Review this campaign result before proposing additional experiments.\n'
            'The prior full-table NPU failure remains a closed historical result.\n')
    update(campaign, 'prepared')
    return campaign


def start(args):
    campaign = initialize(args.campaign, args.source_campaign, args.recovery_from, args.validation)
    command = build_service_command(campaign)
    atomic_json(state_dir(campaign) / 'launch.json', dict(utc=utc(), command=command,
                launcher_pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        update(campaign, 'launch_failed', error=str(exc))
        raise
    print(json.dumps(dict(campaign=str(campaign), unit=unit_name(campaign),
                         status=str(state_dir(campaign) / 'status.json'))))


def request_stop(campaign):
    atomic_json(state_dir(campaign) / 'stop-request.json', dict(utc=utc(), requested=True))
    subprocess.run(['systemctl', '--user', 'stop', unit_name(campaign)], check=True)


def check_stop(campaign):
    if (state_dir(campaign) / 'stop-request.json').exists():
        raise InterruptedError('Operator requested stop')


def run_child(campaign, command, phase, output):
    """Only lightweight monitoring occurs while the child performs measurements."""
    output = Path(output)
    log_path = state_dir(campaign) / (phase + '.log')
    with log_path.open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        last = 0
        while child.poll() is None:
            check_stop(campaign)
            if time.monotonic() - last >= 15:
                detail = dict(child_pid=child.pid, output=str(output))
                for name in ('status', 'progress'):
                    path = output / (name + '.json')
                    if path.exists():
                        try:
                            detail[name] = read(path)
                        except (OSError, ValueError):
                            pass
                update(campaign, phase, **detail)
                last = time.monotonic()
            time.sleep(1)
        return child.returncode


def execute_stages(campaign):
    from .refine_governor import GOVERNOR
    from .reliability_campaign import allowance, dependency
    from .reliability_recovery import verify_source
    records = read(campaign / 'execution.json')
    for stage in STAGES:
        check_stop(campaign)
        verify_source(read(campaign / 'implementation-validation.json'))
        ledger = read(campaign / 'campaign.json')
        if GOVERNOR.read_text().strip() != ledger['original_governor']:
            raise RuntimeError('Governor not restored; halt campaign')
        try:
            dependency(ledger, stage)
            allowance(ledger, stage)
        except ValueError as exc:
            records.append(dict(stage=stage, state='closed', reason=str(exc), utc=utc()))
            atomic_json(campaign / 'execution.json', records)
            continue
        output = ROOT / 'runs' / (campaign.name + '-' + stage)
        if output.exists():
            raise ValueError('Run output already exists; no automatic retries')
        command = [sys.executable, '-B', str(ROOT / 'experiment.py'), 'reliability',
                   '--stage', stage, '--source-campaign', ledger['source_campaign'],
                   '--campaign', str(campaign), '--output', str(output)]
        record = dict(stage=stage, state='running', started_utc=utc(), output=str(output), argv=command)
        records.append(record)
        atomic_json(campaign / 'execution.json', records)
        code = run_child(campaign, command, stage, output)
        record.update(state='finished', returncode=code, finished_utc=utc())
        summary = read(output / 'summary.json') if (output / 'summary.json').exists() else {}
        validation = read(output / 'validation.json') if (output / 'validation.json').exists() else {}
        record.update(decision=summary.get('decision', 'missing_summary'), audit_passed=validation.get('passed', False))
        atomic_json(campaign / 'execution.json', records)
        if code or not summary.get('complete') or not validation.get('passed'):
            raise RuntimeError(stage + ': stopped after ' + record['decision'] + '; no automatic retry')


def run_service(campaign):
    campaign = Path(campaign).resolve()
    if not os.environ.get('INVOCATION_ID') or unit_name(campaign) not in Path('/proc/self/cgroup').read_text():
        raise RuntimeError('Hardware workflow must run in its managed user service')
    from .reliability_recovery import verify_diagnostic, verify_lineage, verify_source
    old = {}
    def stopping(signum, _frame):
        atomic_json(state_dir(campaign) / 'stop-request.json', dict(utc=utc(), signal=signum))
        raise InterruptedError(f'Service interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        old[signum] = signal.signal(signum, stopping)
    try:
        with exclusive(ROOT / 'runs/.reliability-service.lock'):
            verify_source(read(campaign / 'implementation-validation.json'))
            verify_lineage(campaign)
            atomic_json(state_dir(campaign) / 'service-identity.json', dict(utc=utc(), pid=os.getpid(),
                        service=service_properties(unit_name(campaign)),
                        cgroup=Path('/proc/self/cgroup').read_text(), invocation_id=os.environ.get('INVOCATION_ID')))
            smoke = ROOT / 'runs' / (campaign.name + '-diagnostic')
            command = [sys.executable, '-B', '-m', 'efficiency.reliability_smoke', '--output', str(smoke)]
            code = run_child(campaign, command, 'diagnostic', smoke)
            result = read(smoke / 'summary.json') if (smoke / 'summary.json').exists() else {}
            diagnostic = dict(output=str(smoke), returncode=code, result=result)
            if (smoke / 'summary.json').exists():
                from .research_report import seal
                seal(smoke)
                diagnostic['checksums_sha256'] = digest_file(smoke / 'checksums.json')
            atomic_json(campaign / 'diagnostic-validation.json', diagnostic)
            if code or not result.get('passed'):
                raise RuntimeError('Diagnostic validation did not pass; scientific stages were not started')
            verify_diagnostic(campaign)
            from .research_campaign import collect_previous
            atomic_json(campaign / 'registry.json', collect_previous())
            check_stop(campaign)
            execute_stages(campaign)
            update(campaign, 'auditing')
            from .reliability_postvalidate import validate
            validate(campaign)
            verify_lineage(campaign)
            verify_diagnostic(campaign)
            verify_source(read(campaign / 'implementation-validation.json'))
            from .reliability_report import finalize
            with exclusive(ROOT / 'runs/.experiment.lock'), exclusive(campaign / '.lock'):
                result = finalize(campaign)
            update(campaign, 'complete', result=result, scientific_outcomes=[
                dict(stage=a['stage'], decision=read(Path(a['output']) / 'summary.json')['decision'])
                for a in read(campaign / 'campaign.json')['attempts']])
    except BaseException as exc:
        update(campaign, 'needs_attention', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)


def stopped(campaign):
    """Post-stop observation; never fabricate missing run outputs or audit success."""
    from .refine_governor import GOVERNOR
    from .reliability_campaign import recover
    campaign = Path(campaign).resolve()
    status_path = state_dir(campaign) / 'status.json'
    current = read(status_path) if status_path.exists() else {}
    ledger = read(campaign / 'campaign.json') if (campaign / 'campaign.json').exists() else {}
    observed = GOVERNOR.read_text().strip()
    # A privileged watchdog may outlive the user's signal briefly; allow its
    # one-second controller check to restore before recording the observation.
    deadline = time.monotonic() + 10
    while ledger.get('original_governor', observed) != observed and time.monotonic() < deadline:
        time.sleep(.2)
        observed = GOVERNOR.read_text().strip()
    receipt = dict(utc=utc(), service_result=os.environ.get('SERVICE_RESULT'),
                   exit_code=os.environ.get('EXIT_CODE'), exit_status=os.environ.get('EXIT_STATUS'),
                   governor=observed, status_before=current)
    if (campaign / 'checksums.json').exists():
        receipt['sealed_campaign_unchanged'] = True
    else:
        try:
            with exclusive(ROOT / 'runs/.experiment.lock'), exclusive(campaign / '.lock'):
                ledger = read(campaign / 'campaign.json')
                recover(ledger)
                atomic_json(campaign / 'campaign.json', ledger)
                receipt['governor_matches_original'] = receipt['governor'] == ledger['original_governor']
                receipt['charged_seconds'] = sum(a['charged_seconds'] for a in ledger['attempts'])
                records = read(campaign / 'execution.json')
                for record in records:
                    if record['state'] == 'running':
                        record.update(state='interrupted', finished_utc=utc(), audit_passed=False)
                atomic_json(campaign / 'execution.json', records)
                atomic_json(campaign / 'interruption-observation.json', receipt)
        except Exception as exc:
            receipt['recovery_error'] = f'{type(exc).__name__}: {exc}'
    atomic_json(state_dir(campaign) / 'stop-receipt.json', receipt)
    if current.get('phase') != 'complete':
        update(campaign, 'needs_attention', error=current.get('error', 'Service stopped without completed final audit'),
               stop_receipt=str(state_dir(campaign) / 'stop-receipt.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'status', 'stop', '_run', '_stopped'):
        command = commands.add_parser(name)
        command.add_argument('--campaign', type=Path, required=True)
        if name == 'start':
            command.add_argument('--source-campaign', type=Path, required=True)
            command.add_argument('--recovery-from', type=Path, required=True)
            command.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    if args.command == 'start':
        start(args)
    elif args.command == 'status':
        print(json.dumps(dict(saved=read(state_dir(args.campaign) / 'status.json'),
                            service=service_properties(unit_name(args.campaign))), indent=2))
    elif args.command == 'stop':
        request_stop(args.campaign)
    elif args.command == '_run':
        run_service(args.campaign)
    else:
        stopped(args.campaign)


if __name__ == '__main__':
    main()
