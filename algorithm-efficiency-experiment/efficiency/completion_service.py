"""One durable, bounded roadmap stage per immutable campaign.

Stages are explicitly launched after review; this service never retries, tunes a
candidate or advances to a dependent stage automatically.
"""
import argparse
from contextlib import contextmanager
import fcntl
import html
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .refine_governor import GOVERNOR, request

STATE_ROOT = Path.home() / 'Documents/completion-services'
BUDGET_PATH = ROOT / 'runs/.completion-budget.json'
CPU_ORIGIN = ROOT / 'campaigns/cpu-baseline-comparison-20261005T155600Z'
POLICY_PATH = ROOT / 'runs/.completion-npu-policy.json'
TRACKS = ('gpu', 'npu')


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def protocol_module(track):
    if track not in TRACKS:
        raise ValueError('Track has no frozen implementation: ' + str(track))
    return 'completion_' + track + '_protocol'


def modules(track):
    protocol_module(track)
    return tuple(importlib.import_module('efficiency.completion_' + track + '_' + part)
                 for part in ('spec', 'protocol', 'audit'))


def specification(config):
    spec, _, _ = modules(config['track'])
    return (getattr(spec, 'spec', None) or spec.specification)(config)


def state_dir(campaign):
    return STATE_ROOT / Path(campaign).name


def unit_name(campaign):
    name = Path(campaign).name
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Campaign name must contain letters, digits, dashes or underscores')
    return 'algorithm-completion-' + name + '.service'


def update(campaign, phase, **values):
    atomic_json(state_dir(campaign) / 'status.json', dict(utc=utc(), phase=phase,
                campaign=str(campaign), unit=unit_name(campaign), **values))


def parent_paths(values):
    parents = {}
    for value in values:
        key, separator, raw = value.partition('=')
        if not separator or not key or key in parents:
            raise ValueError('Parents require unique NAME=PATH values')
        path = Path(raw).resolve(strict=True)
        if path.parent != ROOT / 'runs':
            raise ValueError('Parent must be an original project run directory')
        parents[key] = str(path)
    return parents


def validate_parent(path, *, accepted=True):
    from .hybrid import verify_artifacts
    path = Path(path)
    verify_artifacts(path)
    if not read(path / 'validation.json').get('passed'):
        raise ValueError('Parent independent audit did not pass: ' + str(path))
    summary = read(path / 'summary.json')
    qualified = summary.get('accepted') or (summary.get('stage') == 'develop' and summary.get('selected') is not None)
    if not summary.get('complete') or (accepted and not qualified):
        raise ValueError('Parent scientific prerequisite did not qualify: ' + str(path))
    return summary


def validate_dependencies(config):
    parents = config['parents']
    if config['track'] == 'gpu':
        validate_parent(parents['gpu-confirmed'])
        if config['stage'] == 'confirm':
            summary = validate_parent(parents['gpu-develop'])
            if summary['selected'] != config['selected']:
                raise ValueError('GPU route must equal the frozen development selection')
    elif config['stage'] != 'diagnostic':
        validate_parent(parents['diagnostic'], accepted=False)
        if not POLICY_PATH.is_file() or read(POLICY_PATH) != config.get('revised_policy'):
            raise ValueError('One reviewed NPU policy must be frozen before qualification')
        if config['stage'] == 'confirm':
            summary = validate_parent(parents['develop'])
            if summary['selected'] != config['revised_policy']:
                raise ValueError('NPU policy differs from qualified development')


def stage_config(args):
    from .study_spec import LLAMA
    if args.track not in TRACKS or args.stage not in ('diagnostic', 'develop', 'confirm'):
        raise ValueError('Unknown completion stage')
    if args.stage == 'diagnostic' and args.track != 'npu':
        raise ValueError('Only NPU has a separate diagnostic stage')
    parents = parent_paths(args.parent)
    config = dict(profile='completion', track=args.track, stage=args.stage,
                  phase='cpu' if args.track == 'gpu' else 'hat', model=str(LLAMA),
                  model_id='llama', policy=5, parents=parents, selected=None,
                  baseline_seconds=15, cooldown_seconds=15, cleanup_reserve_seconds=120)
    if args.track == 'gpu' and args.stage == 'confirm':
        config['selected'] = validate_parent(parents['gpu-develop'])['selected']
    if args.track == 'npu' and args.stage != 'diagnostic':
        config['revised_policy'] = read(POLICY_PATH)
        if args.stage == 'confirm':
            config['selected'] = validate_parent(parents['develop'])['selected']
    validate_dependencies(config)
    specification(config)
    return config


def build_service_command(campaign):
    campaign = Path(campaign).resolve()
    directory = state_dir(campaign)
    if any(c in str(campaign) + str(ROOT) + str(directory) + sys.executable for c in ' \t\n%$"\\'):
        raise ValueError('Service paths contain unsupported argument characters')
    post = f'{sys.executable} -B -m efficiency.completion_service _stopped --campaign {campaign}'
    return ['systemd-run', '--user', '--unit=' + unit_name(campaign), '--service-type=exec',
            '--expand-environment=no', '--working-directory=' + str(ROOT),
            '--property=Restart=no', '--property=KillMode=mixed',
            '--property=TimeoutStopSec=120', '--property=RuntimeMaxSec=9000',
            '--property=ExecStopPost=' + post,
            '--property=StandardOutput=append:' + str(directory / 'service.log'),
            '--property=StandardError=append:' + str(directory / 'service.log'),
            '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1',
            '--setenv=MKL_NUM_THREADS=1', '--setenv=VECLIB_MAXIMUM_THREADS=1',
            '--setenv=NUMEXPR_NUM_THREADS=1', sys.executable, '-B', '-m',
            'efficiency.completion_service', '_run', '--campaign', str(campaign)]


def initialize(args):
    from . import completion_budget as budget
    from .reliability_recovery import verify_source
    from .reliability_campaign import historical_seals
    from .research_campaign import collect_previous
    campaign = args.campaign.resolve()
    if campaign.parent != ROOT / 'campaigns':
        raise ValueError('Campaign must be a fresh project campaigns directory')
    unit_name(campaign)
    config = stage_config(args)
    validation = read(args.validation)
    verify_source(validation)
    if GOVERNOR.read_text().strip() != 'ondemand':
        raise RuntimeError('Original ondemand governor must be restored before launch')
    manager = subprocess.run(['systemctl', '--user', 'is-system-running'], capture_output=True, text=True)
    if manager.stdout.strip() not in ('running', 'degraded'):
        raise RuntimeError('User service manager is unavailable')
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Less than 20 GiB free for new evidence; inspect storage before launch')
    with exclusive(ROOT / 'runs/.experiment.lock'), exclusive(ROOT / 'runs/.completion-service.lock'):
        if campaign.exists() or state_dir(campaign).exists():
            raise ValueError('Campaign and service state must be new')
        budget.initialize(BUDGET_PATH, cpu_origin=CPU_ORIGIN)
        before = budget.snapshot(BUDGET_PATH)
        bucket = 'develop' if args.stage == 'diagnostic' else args.stage
        previous = collect_previous()
        seals = historical_seals()
        reservation = budget.reserve(BUDGET_PATH, args.track, bucket, campaign,
                    requested=args.max_seconds, retry_from=args.retry_from,
                    continuation_from=args.continuation_from,
                    purpose='diagnostic' if args.stage == 'diagnostic' else 'qualification')
        # The reservation remains charged conservatively if any later setup fails.
        campaign.mkdir()
        state_dir(campaign).mkdir(parents=True)
        reserved = reservation['reservation']['reserved_seconds']
        identity = str(uuid.uuid4())
        config.update(campaign=str(campaign), campaign_id=identity,
                      budget_reservation_id=reservation['reservation_id'],
                      fixture_namespace=identity + '|' + args.track + '|' + args.stage,
                      max_seconds=reserved, reserved_seconds=reserved,
                      development_charged_seconds=sum(r['charged_seconds'] for r in before['reservations']
                          if r['track'] == args.track and r['stage'] == 'develop'),
                      confirmation_charged_seconds=sum(r['charged_seconds'] for r in before['reservations']
                          if r['track'] == args.track and r['stage'] == 'confirm'))
        ledger = dict(version='completion-v1', created_utc=utc(), campaign_id=identity,
                      track=args.track, stage=args.stage, max_seconds=reserved,
                      original_governor='ondemand', reservation_id=reservation['reservation_id'],
                      source_sha256=validation['source_sha256'], attempts=[])
        for name, value in (('campaign', ledger), ('planned-config', config),
                            ('implementation-validation', validation), ('registry', previous),
                            ('historical-seals', seals), ('budget-reservation', reservation)):
            atomic_json(campaign / (name + '.json'), value)
        atomic_json(campaign / 'authorization.json', dict(utc=utc(), instruction='Implement the completion roadmap',
                    automatic_retries=False, defaults_changed=False, stage_seconds=reserved))
    update(campaign, 'prepared')
    return campaign


def start(args):
    campaign = initialize(args)
    command = build_service_command(campaign)
    atomic_json(state_dir(campaign) / 'launch.json', dict(utc=utc(), command=command))
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        update(campaign, 'launch_failed', error=str(exc))
        stopped(campaign)
        raise
    print(json.dumps(dict(campaign=str(campaign), unit=unit_name(campaign),
                         status=str(state_dir(campaign) / 'status.json'))))
    return 0


def scientific_failure(rows):
    for row in rows:
        if row.get('event') in ('measurement_complete', 'correctness_failure') or (
                row.get('event') == 'pilot_gate' and row.get('fits') is False):
            return True
        if row.get('event') in ('warmup', 'measurement'):
            result = row.get('response', {}).get('result', {})
            for value in [result, *result.get('requests', [])]:
                if (value.get('correct') is False or value.get('matches_warmup') is False
                        or value.get('validation_errors', 0) or value.get('unexpected_output_io') is True):
                    return True
    return False


def evidence_events(directory, event_file):
    rows, damaged = read_jsonl(directory / event_file)
    correctness = directory / 'correctness.json'
    if correctness.exists() and any(row.get('correct') is False or row.get('validation_errors', 0)
                                   for row in read(correctness)):
        rows.append(dict(event='correctness_failure'))
    legacy, _ = read_jsonl(directory / 'refine.jsonl')
    if any(row.get('event') == 'compatibility_failure' for row in legacy):
        rows.append(dict(event='correctness_failure'))
    return rows, damaged


def classify(summary, validation, rows, restored):
    if restored and summary.get('complete') and validation.get('passed'):
        if summary.get('stage') == 'diagnostic':
            return 'diagnostic_complete'
        return 'qualified' if summary.get('accepted') or summary.get('selected') is not None else 'rejected'
    if scientific_failure(rows) or summary.get('complete') or not restored:
        return 'inconclusive'
    return 'infrastructure_failed'


def report(directory, summary, validation, disposition):
    from .completion_report import report as build
    build(directory, summary, validation, disposition)


def execute(campaign):
    from . import completion_budget as budget
    from .reliability_recovery import verify_source
    from .cpu_compare_service import verify_history
    from .research_report import seal
    from .runner import Supervisor
    campaign = Path(campaign).resolve()
    with exclusive(ROOT / 'runs/.experiment.lock'), exclusive(campaign / '.lock'):
        if (campaign / 'checksums.json').exists():
            raise ValueError('Completed campaign is immutable')
        ledger = read(campaign / 'campaign.json')
        if ledger['attempts']:
            raise ValueError('Campaign already attempted; no automatic retries')
        began = time.monotonic()
        verify_source(read(campaign / 'implementation-validation.json'))
        config = read(campaign / 'planned-config.json')
        validate_dependencies(config)
        if GOVERNOR.read_text().strip() != ledger['original_governor']:
            raise RuntimeError('Governor baseline changed')
        if (state_dir(campaign) / 'stop-request.json').exists():
            raise InterruptedError('Stop requested before initialization')
        spec, _, auditor = modules(config['track'])
        directory = ROOT / 'runs' / (campaign.name + '-' + config['stage'])
        directory.mkdir()
        reservation = read(campaign / 'budget-reservation.json')
        started = budget.mark_started(BUDGET_PATH, ledger['reservation_id'])
        live = started['reservation']
        bucket = 'develop' if config['stage'] == 'diagnostic' else config['stage']
        if not (live['id'] == config['budget_reservation_id'] == reservation['reservation_id']
                and live['campaign'] == str(campaign) == config['campaign']
                and live['track'] == config['track'] and live['stage'] == bucket
                and live['reserved_seconds'] == ledger['max_seconds'] == config['reserved_seconds']
                == reservation['reservation']['reserved_seconds']):
            raise ValueError('Live budget differs from frozen stage reservation')
        limit = ledger['max_seconds']
        attempt = dict(stage=config['stage'], state='running', output=str(directory),
                       started_utc=utc(), reserved_seconds=limit, charged_seconds=limit,
                       scientific_complete=False, audit_passed=False)
        ledger['attempts'].append(attempt)
        atomic_json(campaign / 'campaign.json', ledger)
        atomic_json(directory / 'protocol.json', specification(config))
        atomic_json(directory / 'budget-reservation.json', reservation)
        shutil.copy2(campaign / 'registry.json', directory / 'prior-fixtures.json')
        helper = log = temporary = socket_path = None
        restored = False
        restoration_error = None
        code = 1
        cleanup = False

        def interrupt(signum, _frame):
            if not cleanup:
                raise InterruptedError(f'Completion stage interrupted by signal {signum}')

        old_handlers = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            update(campaign, 'initializing', output=str(directory))
            if config['track'] == 'gpu':
                temporary = tempfile.TemporaryDirectory(prefix='completion-governor-')
                socket_path = Path(temporary.name) / 'lease.sock'
                log = (directory / 'governor-helper.log').open('w')
                if limit - (time.monotonic() - began) <= 120:
                    raise TimeoutError('Setup exhausted allowance before authentication')
                helper = subprocess.Popen(['pkexec', sys.executable, str(ROOT / 'efficiency/refine_governor.py'),
                    '--socket', str(socket_path), '--pid', str(os.getpid()),
                    '--seconds', str(limit - (time.monotonic() - began))],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(campaign, 'awaiting_governor_authentication', output=str(directory))
                expiry = min(began + limit - 120, time.monotonic() + 120)
                while not socket_path.exists() and helper.poll() is None and time.monotonic() < expiry:
                    if (state_dir(campaign) / 'stop-request.json').exists():
                        raise InterruptedError('Operator stopped authentication')
                    time.sleep(.1)
                if not socket_path.exists():
                    raise PermissionError('Desktop governor authentication unavailable; no matrix started')
                if request(socket_path)['original'] != ledger['original_governor']:
                    raise RuntimeError('Governor baseline changed')
                config['governor_socket'] = str(socket_path)
            config.update(max_seconds=limit - (time.monotonic() - began),
                          original_governor=ledger['original_governor'])
            if config['max_seconds'] <= 120:
                raise TimeoutError('Initialization exhausted work allowance')
            atomic_json(directory / 'config.json', config)
            update(campaign, 'measuring', output=str(directory))
            code = Supervisor(directory, config).execute()
        except BaseException as exc:
            atomic_json(directory / 'startup-failure.json', dict(utc=utc(), error=f'{type(exc).__name__}: {exc}'))
        finally:
            cleanup = True
            if socket_path and socket_path.exists():
                try:
                    request(socket_path, close=True)
                except Exception as exc:
                    restoration_error = str(exc)
            if helper:
                try:
                    helper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        helper.terminate()
                    except (PermissionError, ProcessLookupError):
                        pass
                    try:
                        helper.wait(timeout=max(0, min(65, began + limit - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        restoration_error = 'Governor watchdog restoration pending'
            if log:
                log.close()
            if temporary and (helper is None or helper.poll() is not None):
                temporary.cleanup()
            current = GOVERNOR.read_text().strip()
            restored = current == ledger['original_governor'] and (helper is None or helper.poll() is not None)
            ended = time.monotonic()
            atomic_json(directory / 'governor-restoration.json', dict(original=ledger['original_governor'],
                        current=current, restored=restored, error=restoration_error))
            atomic_json(directory / 'runtime-finalization.json', dict(started_monotonic=began,
                        finished_monotonic=ended, charged_seconds=ended - began, reserved_seconds=limit,
                        original_governor=ledger['original_governor'], restored=restored))
            attempt.update(charged_seconds=ended - began, restored=restored, state='finished')
            atomic_json(campaign / 'campaign.json', ledger)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        update(campaign, 'auditing', output=str(directory))
        rows, damaged = evidence_events(directory, spec.EVENTS)
        summary = dict(stage=config['stage'], complete=False, accepted=False, selected=None,
                       decision='incomplete', defaults_changed=False)
        validation = dict(passed=False, errors=['Stage did not initialize'])
        if (directory / 'manifest.json').exists():
            try:
                summary = spec.analyze(rows, config)
                atomic_json(directory / 'summary.json', summary)
                validation = auditor.audit(directory, require_seal=False)
            except Exception as exc:
                validation = dict(passed=False, errors=[f'{type(exc).__name__}: {exc}'])
        atomic_json(directory / 'summary.json', summary)
        atomic_json(directory / 'validation.json', validation)
        disposition = classify(dict(summary, stage=config['stage']), validation, rows, restored)
        receipt = budget.finish(BUDGET_PATH, ledger['reservation_id'], attempt['charged_seconds'], disposition)
        atomic_json(directory / 'budget-finalization.json', receipt)
        atomic_json(campaign / 'budget-finalization.json', receipt)
        report(directory, summary, validation, disposition)
        seal(directory)
        if validation.get('passed'):
            validation = auditor.audit(directory, require_seal=True)
            atomic_json(campaign / 'post-seal-validation.json', validation)
            if not validation.get('passed'):
                raise RuntimeError('Sealed independent audit failed; budget remains finalizing')
        history = verify_history(campaign)
        verify_source(read(campaign / 'implementation-validation.json'))
        attempt.update(audit_passed=validation.get('passed', False), scientific_complete=summary.get('complete', False),
                       decision=summary.get('decision'), disposition=disposition,
                       checksums_sha256=digest_file(directory / 'checksums.json'))
        atomic_json(campaign / 'campaign.json', ledger)
        result = dict(passed=bool(validation.get('passed') and restored), disposition=disposition,
                      summary=summary, validation=validation, history=history, output=str(directory),
                      defaults_changed=False, restored=restored)
        atomic_json(campaign / 'final-validation.json', result)
        (campaign / 'CONTEXT.txt').write_text(f'{utc()}\n{config["track"]}/{config["stage"]}: {disposition}\n'
            f'Hardware seconds: {attempt["charged_seconds"]:.3f} / {limit}\n'
            f'Report: {directory / "report.html"}\nDefaults unchanged.\n')
        seal(campaign)
        budget.attach_evidence(BUDGET_PATH, ledger['reservation_id'], campaign)
        update(campaign, 'complete' if result['passed'] else 'needs_attention', result=result)
        return 0 if result['passed'] else (code or 1)


def stopped(campaign):
    from . import completion_budget as budget
    campaign = Path(campaign).resolve()
    ledger = read(campaign / 'campaign.json')
    expiry = time.monotonic() + 10
    while GOVERNOR.read_text().strip() != ledger['original_governor'] and time.monotonic() < expiry:
        time.sleep(.2)
    receipt = dict(utc=utc(), service_result=os.environ.get('SERVICE_RESULT'),
                   governor=GOVERNOR.read_text().strip(), restored=GOVERNOR.read_text().strip() == ledger['original_governor'])
    with exclusive(ROOT / 'runs/.experiment.lock'), exclusive(campaign / '.lock'):
        ledger = read(campaign / 'campaign.json')
        current = budget.snapshot(BUDGET_PATH)
        row = next(r for r in current['reservations'] if r['id'] == ledger['reservation_id'])
        if row['state'] != 'finalized' and (campaign / 'checksums.json').exists():
            budget.attach_evidence(BUDGET_PATH, row['id'], campaign)
        elif row['state'] != 'finalized':
            seal_interrupted(campaign, ledger, receipt)
    if not read(campaign / 'final-validation.json').get('passed'):
        update(campaign, 'needs_attention', error='Service stopped; conservative budget charge and failure evidence sealed')
    atomic_json(state_dir(campaign) / 'stop-receipt.json', receipt)


def seal_interrupted(campaign, ledger, restoration):
    """Preserve partial data and charge the full reservation after interrupted finalization."""
    from . import completion_budget as budget
    from .research_report import seal
    config = read(campaign / 'planned-config.json')
    directory = ROOT / 'runs' / (campaign.name + '-' + config['stage'])
    directory.mkdir(exist_ok=True)
    rows, _ = evidence_events(directory, 'completion-' + config['track'] + '.jsonl')
    scientific = scientific_failure(rows)
    summary = read(directory / 'summary.json') if (directory / 'summary.json').exists() else dict(
        stage=config['stage'], complete=False, accepted=False, selected=None, decision='interrupted')
    disposition = 'inconclusive' if scientific or summary.get('complete') or not restoration['restored'] else 'infrastructure_failed'
    receipt = budget.abandon(BUDGET_PATH, ledger['reservation_id'], disposition)
    validation = dict(passed=False, errors=['Service interrupted; no completed final scientific audit'])
    atomic_json(campaign / 'budget-finalization.json', receipt)
    atomic_json(campaign / 'interruption-observation.json', restoration)
    if not (directory / 'checksums.json').exists():
        atomic_json(directory / 'summary.json', summary)
        atomic_json(directory / 'validation.json', validation)
        atomic_json(directory / 'budget-finalization.json', receipt)
        atomic_json(directory / 'interruption-observation.json', restoration)
        report(directory, summary, validation, disposition)
        seal(directory)
    if not ledger['attempts']:
        ledger['attempts'].append(dict(stage=config['stage'], output=str(directory), reserved_seconds=ledger['max_seconds']))
    ledger['attempts'][-1].update(state='interrupted', charged_seconds=ledger['max_seconds'],
        audit_passed=False, scientific_complete=False, restored=restoration['restored'],
        disposition=disposition, checksums_sha256=digest_file(directory / 'checksums.json'))
    atomic_json(campaign / 'campaign.json', ledger)
    atomic_json(campaign / 'final-validation.json', dict(passed=False, disposition=disposition,
        validation=validation, summary=summary, output=str(directory), restored=restoration['restored'], defaults_changed=False))
    (campaign / 'CONTEXT.txt').write_text(f'{utc()}\nInterrupted completion stage; no scientific qualification.\n'
        f'Full reservation charged: {ledger["max_seconds"]} seconds.\nEvidence: {directory}\n')
    seal(campaign)
    budget.attach_evidence(BUDGET_PATH, ledger['reservation_id'], campaign)


def freeze_policy(policy_file, diagnostic):
    from .completion_npu_spec import policy as validate_policy
    policy = read(policy_file)
    diagnostic = Path(diagnostic).resolve(strict=True)
    validate_parent(diagnostic, accepted=False)
    if read(diagnostic / 'config.json')['stage'] != 'diagnostic':
        raise ValueError('Policy needs the new four-way diagnostic evidence')
    if policy.get('diagnostic_checksums_sha256') != digest_file(diagnostic / 'checksums.json'):
        raise ValueError('Policy diagnostic seal differs')
    validate_policy(dict(stage='develop', revised_policy=policy))
    with exclusive(ROOT / 'runs/.experiment.lock'), exclusive(ROOT / 'runs/.completion-service.lock'):
        if POLICY_PATH.exists():
            if read(POLICY_PATH) != policy:
                raise ValueError('One revised policy already frozen; no second candidate permitted')
        else:
            atomic_json(POLICY_PATH, policy)
    return dict(path=str(POLICY_PATH), sha256=digest_file(POLICY_PATH))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'status', 'stop', '_run', '_stopped'):
        command = commands.add_parser(name)
        command.add_argument('--campaign', type=Path, required=True)
        if name == 'start':
            command.add_argument('--track', choices=TRACKS, required=True)
            command.add_argument('--stage', choices=('diagnostic', 'develop', 'confirm'), required=True)
            command.add_argument('--validation', type=Path, required=True)
            command.add_argument('--parent', action='append', default=[], metavar='NAME=PATH')
            command.add_argument('--max-seconds', type=float)
            command.add_argument('--retry-from', type=Path)
            command.add_argument('--continuation-from', type=Path)
    commands.add_parser('budget')
    freeze = commands.add_parser('freeze-npu-policy')
    freeze.add_argument('--policy', type=Path, required=True)
    freeze.add_argument('--diagnostic', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'budget':
            from .completion_budget import snapshot
            print(json.dumps(snapshot(BUDGET_PATH), indent=2))
            return 0
        if args.command == 'freeze-npu-policy':
            print(json.dumps(freeze_policy(args.policy, args.diagnostic), indent=2))
            return 0
        campaign = args.campaign.resolve()
        if args.command == 'start':
            return start(args)
        if args.command == 'status':
            from .reliability_service import service_properties
            value = read(state_dir(campaign) / 'status.json')
            value['service'] = service_properties(unit_name(campaign))
            if value.get('output') and (Path(value['output']) / 'progress.json').exists():
                value['progress'] = read(Path(value['output']) / 'progress.json')
            print(json.dumps(value, indent=2))
            return 0
        if args.command == 'stop':
            atomic_json(state_dir(campaign) / 'stop-request.json', dict(utc=utc()))
            subprocess.run(['systemctl', '--user', 'stop', unit_name(campaign)], check=True)
            return 0
        if args.command == '_stopped':
            stopped(campaign)
            return 0
        from .completion_evidence import require_managed_service
        require_managed_service(unit_name(campaign))
        return execute(campaign)
    except BaseException as exc:
        if args.command == '_run':
            update(args.campaign.resolve(), 'needs_attention', error=f'{type(exc).__name__}: {exc}')
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
