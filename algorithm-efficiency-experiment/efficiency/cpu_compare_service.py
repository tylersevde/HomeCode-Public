"""Durable CPU comparison: start/status/stop; no automatic retries or promotion."""
import argparse
from contextlib import contextmanager
import fcntl
import html
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .common import ROOT, MODEL, atomic_json, digest_file, read_jsonl, utc
from .cpu_compare_spec import VERSION, MAX_SECONDS, specification
from .refine_governor import GOVERNOR, request
from . import completion_budget as budgets

STATE_ROOT = Path.home() / 'Documents/cpu-compare-services'


def budget_path():
    return ROOT / 'runs/.completion-budget.json'


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle


def state_dir(campaign):
    return STATE_ROOT / Path(campaign).name


def unit_name(campaign):
    name = Path(campaign).name
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Campaign name must contain letters, digits, dashes or underscores')
    return 'algorithm-cpu-compare-' + name + '.service'


def update(campaign, phase, **values):
    atomic_json(state_dir(campaign) / 'status.json', dict(utc=utc(), phase=phase,
                campaign=str(campaign), unit=unit_name(campaign), **values))


def ensure_report(directory, summary, restored):
    """A failed startup still has a working report link and explicit disposition."""
    target = directory / 'report.html'
    if not target.exists():
        target.write_text('<!doctype html><meta charset="utf-8"><title>CPU comparison attempt</title>'
            '<h1>CPU comparison attempt</h1><p>' + html.escape(summary.get('decision', 'incomplete')) +
            '</p><p>Governor restored: ' + str(bool(restored)) +
            '. No performance qualification or preset is claimed for an incomplete attempt.</p>'
            '<p><a href="summary.json">Summary</a> · <a href="validation.json">Validation</a> · '
            '<a href="runtime-finalization.json">Budget and restoration</a></p><pre>' +
            html.escape(json.dumps(summary, indent=2)) + '</pre>')


def scientific_outcome(events, summary=None):
    """Scientific completion, infeasibility or invalid output cannot become a retry."""
    summary = summary or {}
    if summary.get('complete') or summary.get('decision') == 'deferred_by_pilot':
        return True
    for event in events:
        kind = event.get('event')
        if kind == 'measurement_complete' or (kind == 'pilot_gate' and event.get('fits') is False):
            return True
        if kind not in ('warmup', 'measurement'):
            continue
        result = event.get('response', {}).get('result', {})
        for row in [result, *result.get('requests', [])]:
            if (row.get('correct') is False or row.get('validation_errors', 0)
                    or row.get('unexpected_output_io') is True
                    or (kind == 'measurement' and row.get('matches_warmup') is False)):
                return True
    return False


def source_provenance(source):
    from .hybrid import verify_artifacts
    source = Path(source).resolve()
    verify_artifacts(source)
    validation = read(source / 'final-validation.json')
    if not validation.get('passed', validation.get('integrity_passed', False)):
        raise ValueError('Historical campaign audit failed')
    ledger = read(source / 'campaign.json')
    choices = [r for r in ledger['attempts'] if r['stage'] == 'cpu-confirm' and
               r.get('scientific_complete') and r.get('audit_passed')]
    if not choices:
        raise ValueError('No audited historical CPU confirmation')
    parent = Path(choices[-1]['output']).resolve(); verify_artifacts(parent)
    summary = read(parent / 'summary.json')
    if not (summary.get('complete') and summary.get('accepted') and summary.get('selected') == 5
            and read(parent / 'validation.json').get('passed')):
        raise ValueError('Historical CPU policy 5 did not qualify')
    return dict(source_campaign=dict(path=str(source), checksums_sha256=digest_file(source/'checksums.json')),
                cpu_confirmation=dict(path=str(parent), checksums_sha256=digest_file(parent/'checksums.json')),
                selected=5)


def build_service_command(campaign):
    campaign = Path(campaign).resolve(); directory = state_dir(campaign)
    if any(c in str(campaign)+str(ROOT)+str(directory)+sys.executable for c in ' \t\n%$"\\'):
        raise ValueError('Service paths contain unsupported argument characters')
    post = f'{sys.executable} -B -m efficiency.cpu_compare_service _stopped --campaign {campaign}'
    return ['systemd-run', '--user', '--unit='+unit_name(campaign), '--service-type=exec',
            '--expand-environment=no', '--working-directory='+str(ROOT),
            '--property=Restart=no', '--property=KillMode=mixed', '--property=TimeoutStopSec=120',
            '--property=RuntimeMaxSec=16200', '--property=ExecStopPost='+post,
            '--property=StandardOutput=append:'+str(directory/'service.log'),
            '--property=StandardError=append:'+str(directory/'service.log'),
            '--setenv=PYTHONDONTWRITEBYTECODE=1', '--setenv=OPENBLAS_NUM_THREADS=1',
            '--setenv=MKL_NUM_THREADS=1', '--setenv=VECLIB_MAXIMUM_THREADS=1',
            '--setenv=NUMEXPR_NUM_THREADS=1', sys.executable, '-B', '-m',
            'efficiency.cpu_compare_service', '_run', '--campaign', str(campaign)]


def initialize(args):
    from .reliability_recovery import verify_source
    from .reliability_campaign import historical_seals
    from .research_campaign import collect_previous
    campaign = args.campaign.resolve()
    if campaign.parent != ROOT/'campaigns':
        raise ValueError('Campaign must be a new directory under project campaigns')
    unit_name(campaign)
    retry_from = getattr(args, 'retry_from', None)
    if retry_from is None:
        raise ValueError('Use --retry-from with the sealed failed CPU comparison; its allowance cannot be reset')
    retry = budgets.validate_retry_source(retry_from)
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or not 120 < args.max_seconds <= MAX_SECONDS):
        raise ValueError('Hardware ceiling must be greater than 120 and at most 14400 seconds')
    validation = read(args.validation); verify_source(validation)
    provenance = source_provenance(args.source_campaign)
    if str(args.source_campaign.resolve()) != retry['source_campaign']:
        raise ValueError('CPU retry must preserve the original scientific source campaign')
    if GOVERNOR.read_text().strip() != 'ondemand':
        raise ValueError('Expected normal ondemand governor before starting')
    manager = subprocess.run(['systemctl', '--user', 'is-system-running'], text=True, capture_output=True)
    if manager.stdout.strip() not in ('running', 'degraded'):
        raise RuntimeError('User service manager unavailable')
    with exclusive(ROOT/'runs/.cpu-compare-service.lock'), exclusive(ROOT/'runs/.experiment.lock'):
        if campaign.exists() or state_dir(campaign).exists():
            raise ValueError('Campaign and service state must be new; automatic retries are disabled')
        previous = collect_previous(); seals = historical_seals()
        origin = (retry['budget_reservation']['snapshot']['cpu_origin']['campaign']['path']
                  if retry['budget_reservation'] else retry_from)
        budgets.initialize(budget_path(), cpu_origin=origin)
        reservation = budgets.reserve(budget_path(), 'cpu', 'compare', campaign,
                                      requested=args.max_seconds, retry_from=retry_from)
        limit = reservation['reservation']['reserved_seconds']
        campaign.mkdir(); state_dir(campaign).mkdir(parents=True)
        atomic_json(campaign/'campaign.json', dict(version=VERSION, campaign_id=str(uuid.uuid4()),
                    created_utc=utc(), source_campaign=str(args.source_campaign.resolve()),
                    max_seconds=limit, original_governor='ondemand', attempts=[],
                    retry_from=retry['campaign'], budget_reservation=reservation))
        atomic_json(campaign/'budget-reservation.json', reservation)
        atomic_json(campaign/'retry-lineage.json', retry)
        atomic_json(campaign/'registry.json', previous)
        atomic_json(campaign/'historical-seals.json', seals)
        atomic_json(campaign/'provenance.json', provenance)
        atomic_json(campaign/'implementation-validation.json', validation)
        atomic_json(campaign/'authorization.json', dict(utc=utc(), instruction='Implement the approved CPU comparison plan',
                    max_seconds=limit, blocks=256, automatic_retries=False,
                    promote_only_if_both_workloads_qualify=True, defaults_changed=False))
    update(campaign, 'prepared')
    return campaign


def start(args):
    campaign = initialize(args); command = build_service_command(campaign)
    atomic_json(state_dir(campaign)/'launch.json', dict(utc=utc(), command=command))
    try:
        subprocess.run(command, check=True)
    except Exception as exc:
        update(campaign, 'launch_failed', error=str(exc))
        # An uncertain launch never returns its reservation to the budget.
        # The hardware lock proves that recovery cannot race a live controller.
        stopped(campaign)
        raise
    print(json.dumps(dict(campaign=str(campaign), unit=unit_name(campaign),
                         status=str(state_dir(campaign)/'status.json'))))
    return 0


def verify_history(campaign):
    from .hybrid import verify_artifacts
    rows = read(campaign/'historical-seals.json')
    for row in rows:
        path = ROOT/row['path']
        if digest_file(path) != row['sha256']:
            raise ValueError('Historical seal changed: '+row['path'])
        verify_artifacts(path.parent)
    return dict(passed=True, historical_seals=len(rows))


def execute(campaign):
    from .reliability_recovery import verify_source
    from .cpu_policy import runtime_identity
    from .runner import Supervisor
    from .research_report import seal
    campaign = Path(campaign).resolve()
    with exclusive(ROOT/'runs/.experiment.lock'), exclusive(campaign/'.lock'):
        if (campaign/'checksums.json').exists():
            raise ValueError('Completed campaign is immutable')
        ledger = read(campaign/'campaign.json')
        if ledger['attempts']:
            raise ValueError('Campaign already attempted; no automatic retries')
        reservation = ledger.get('budget_reservation')
        if reservation is not None:
            if reservation['ledger_path'] != str(budget_path().resolve()):
                raise ValueError('CPU retry uses a different cumulative allowance ledger')
            state = budgets.mark_started(budget_path(), reservation['reservation_id'])
            if (state['reservation']['campaign'] != str(campaign)
                    or state['reservation']['reserved_seconds'] != ledger['max_seconds']):
                raise ValueError('CPU campaign and reserved allowance differ')
        began = time.monotonic()
        verify_source(read(campaign/'implementation-validation.json'))
        if source_provenance(ledger['source_campaign']) != read(campaign/'provenance.json'):
            raise ValueError('Historical selection provenance changed')
        if GOVERNOR.read_text().strip() != ledger['original_governor']:
            raise RuntimeError('Governor baseline changed')
        if (state_dir(campaign)/'stop-request.json').exists():
            raise InterruptedError('Stop requested before hardware initialization')
        output = ROOT/'runs'/(campaign.name+'-compare')
        output.mkdir()
        limit = ledger['max_seconds']
        attempt = dict(stage='cpu-compare', state='running', output=str(output),
                       started_utc=utc(), reserved_seconds=limit, charged_seconds=limit,
                       scientific_complete=False, audit_passed=False)
        ledger['attempts'].append(attempt); atomic_json(campaign/'campaign.json', ledger)
        update(campaign, 'initializing', output=str(output))
        atomic_json(output/'protocol.json', specification())
        shutil.copy2(campaign/'registry.json', output/'prior-fixtures.json')
        shutil.copy2(campaign/'provenance.json', output/'provenance.json')
        if reservation is not None:
            shutil.copy2(campaign/'budget-reservation.json', output/'budget-reservation.json')
            shutil.copy2(campaign/'retry-lineage.json', output/'retry-lineage.json')
        helper = None; log = None; temporary = None; socket_path = None; code = 1
        original = ledger['original_governor']; restoration_error = None
        signal_state = dict(cleanup=False)
        def interrupt(signum, _frame):
            if not signal_state['cleanup']:
                raise InterruptedError(f'CPU comparison interrupted by signal {signum}')
        old_handlers = {s:signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            atomic_json(output/'cpu-runtime.json', runtime_identity())
            temporary = tempfile.TemporaryDirectory(prefix='cpu-compare-governor-')
            socket_path = Path(temporary.name)/'lease.sock'
            log = (output/'governor-helper.log').open('w')
            if limit-(time.monotonic()-began) <= 120:
                raise TimeoutError('Setup exhausted the work allowance before authentication')
            helper = subprocess.Popen(['pkexec', sys.executable, str(ROOT/'efficiency/refine_governor.py'),
                '--socket', str(socket_path), '--pid', str(os.getpid()),
                '--seconds', str(limit-(time.monotonic()-began))],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(campaign, 'awaiting_governor_authentication', output=str(output))
            expiry = min(began+limit-120, time.monotonic()+120)
            while not socket_path.exists() and helper.poll() is None and time.monotonic() < expiry:
                if (state_dir(campaign)/'stop-request.json').exists():
                    raise InterruptedError('Operator stopped authentication')
                time.sleep(.1)
            if not socket_path.exists():
                raise PermissionError('Desktop governor authentication unavailable; no matrix started')
            if request(socket_path)['original'] != original:
                raise RuntimeError('Governor baseline changed')
            config = dict(profile='cpu-compare', phase='cpu', model=str(MODEL),
                          campaign=str(campaign), source_campaign=ledger['source_campaign'],
                          campaign_id=ledger['campaign_id'], attempt_index=0,
                          fixture_namespace=ledger['campaign_id']+'|cpu-compare|attempt-0',
                          max_seconds=limit-(time.monotonic()-began), reserved_seconds=limit,
                          baseline_seconds=15, cooldown_seconds=15, cleanup_reserve_seconds=120,
                          governor_socket=str(socket_path), original_governor=original)
            if reservation is not None:
                config['budget_reservation_id'] = reservation['reservation_id']
                config['attempt_index'] = len(reservation['snapshot']['reservations'])
                config['fixture_namespace'] = ledger['campaign_id'] + '|cpu-compare|' + reservation['reservation_id']
            if config['max_seconds'] <= 120:
                raise TimeoutError('Startup exhausted allowance')
            atomic_json(output/'config.json', config)
            update(campaign, 'measuring', output=str(output))
            code = Supervisor(output, config).execute()
        except BaseException as exc:
            atomic_json(output/'startup-failure.json', dict(utc=utc(), error=f'{type(exc).__name__}: {exc}'))
        finally:
            signal_state['cleanup'] = True
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
                    # The root-owned helper restores on controller loss, lease
                    # expiry or a missing heartbeat. Unprivileged signaling is
                    # not a prerequisite for cleanup.
                    remaining = max(0, min(65, began+limit-time.monotonic()))
                    try:
                        helper.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        restoration_error = 'Governor helper remains active; watchdog restoration pending'
            if log:
                log.close()
            if temporary and (helper is None or helper.poll() is not None):
                temporary.cleanup()
            current = GOVERNOR.read_text().strip()
            restored = current == original and (helper is None or helper.poll() is not None)
            ended = time.monotonic()
            atomic_json(output/'governor-restoration.json', dict(original=original, current=current,
                        restored=restored, error=restoration_error))
            atomic_json(output/'runtime-finalization.json', dict(started_monotonic=began,
                        finished_monotonic=ended, charged_seconds=ended-began, reserved_seconds=limit,
                        original_governor=original, restored=restored))
            attempt.update(charged_seconds=ended-began, restored=restored, state='finished')
            atomic_json(campaign/'campaign.json', ledger)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        update(campaign, 'auditing', output=str(output))
        summary = dict(complete=False, accepted=False, decision='incomplete', metrics=[], defaults_changed=False)
        validation = dict(passed=False, errors=['Supervised comparison did not initialize'])
        if (output/'manifest.json').exists():
            from .cpu_compare_report import build_report
            from .cpu_compare_audit import audit
            try:
                summary = build_report(output)
                validation = audit(output, require_seal=False)
            except Exception as exc:
                validation = dict(passed=False, errors=[f'{type(exc).__name__}: {exc}'])
                atomic_json(output/'finalization-failure.json', validation)
        atomic_json(output/'summary.json', summary)
        atomic_json(output/'validation.json', validation)
        ensure_report(output, summary, restored)
        if reservation is not None:
            events, _ = read_jsonl(output/'cpu-compare.jsonl')
            disposition = ('qualified' if summary['accepted'] and validation['passed'] and restored else
                           'rejected' if summary['complete'] and validation['passed'] else
                           'inconclusive' if scientific_outcome(events, summary) or not restored else
                           'infrastructure_failed')
            final_budget = budgets.finish(budget_path(), reservation['reservation_id'],
                                          attempt['charged_seconds'], disposition)
            ledger['budget_disposition'] = disposition
            atomic_json(output/'budget-finalization.json', final_budget)
            atomic_json(campaign/'budget-finalization.json', final_budget)
        (output/'CONTEXT.txt').write_text(f'{utc()}\nDecision: {summary["decision"]}\n'
            f'Audit passed: {validation["passed"]}\nGovernor restored: {restored}\nDefaults unchanged.\n')
        seal(output)
        if validation['passed']:
            from .cpu_compare_audit import audit
            post = audit(output)
            atomic_json(campaign/'post-seal-validation.json', post)
            validation = post
        history = verify_history(campaign)
        verify_source(read(campaign/'implementation-validation.json'))
        attempt.update(audit_passed=validation['passed'], scientific_complete=summary['complete'],
                       decision=summary['decision'], checksums_sha256=digest_file(output/'checksums.json'))
        atomic_json(campaign/'campaign.json', ledger)
        result = dict(passed=bool(validation['passed'] and restored), summary=summary,
                      validation=validation, history=history, output=str(output), defaults_changed=False)
        if result['passed'] and summary['accepted']:
            from .cpu_policy import export_preset
            export_preset(output, campaign/'preset.json')
            result['preset'] = str(campaign/'preset.json')
        atomic_json(campaign/'final-validation.json', result)
        (campaign/'CONTEXT.txt').write_text(f'{utc()}\nCPU comparison: {summary["decision"]}\n'
            f'Hardware seconds: {attempt["charged_seconds"]:.3f} / {limit}\n'
            f'Report: {output / "report.html"}\nDefaults unchanged.\n')
        seal(campaign)
        if reservation is not None:
            budgets.attach_evidence(budget_path(), reservation['reservation_id'], campaign)
        update(campaign, 'complete' if result['passed'] else 'needs_attention', result=result)
        return 0 if result['passed'] else (code or 1)


def stopped(campaign):
    campaign = Path(campaign).resolve(); ledger_path = campaign/'campaign.json'
    ledger = read(ledger_path); original = ledger['original_governor']
    expiry = time.monotonic()+10
    while GOVERNOR.read_text().strip() != original and time.monotonic() < expiry:
        time.sleep(.2)
    receipt = dict(utc=utc(), service_result=os.environ.get('SERVICE_RESULT'),
                   governor=GOVERNOR.read_text().strip(), restored=GOVERNOR.read_text().strip()==original)
    if ledger.get('budget_reservation'):
        with exclusive(ROOT/'runs/.experiment.lock'), exclusive(campaign/'.lock'):
            ledger = read(ledger_path)
            reference = ledger['budget_reservation']; identifier = reference['reservation_id']
            current = budgets.snapshot(budget_path())
            row = next(r for r in current['reservations'] if r['id'] == identifier)
            if row['state'] != 'finalized' and (campaign/'checksums.json').exists():
                budgets.attach_evidence(budget_path(), identifier, campaign)
            elif row['state'] != 'finalized':
                _seal_interrupted(campaign, ledger, receipt)
        if not read(campaign/'final-validation.json').get('passed'):
            update(campaign, 'needs_attention', error='Service stopped; reserved allowance conservatively finalized')
    elif not (campaign/'checksums.json').exists():
        with exclusive(ROOT/'runs/.experiment.lock'), exclusive(campaign/'.lock'):
            ledger = read(ledger_path)
            for row in ledger['attempts']:
                if row['state'] == 'running':
                    row.update(state='interrupted', charged_seconds=row['reserved_seconds'],
                               audit_passed=False, restored=receipt['restored'])
            atomic_json(ledger_path, ledger)
        update(campaign, 'needs_attention', error='Service stopped without a sealed final result')
    atomic_json(state_dir(campaign)/'stop-receipt.json', receipt)


def _seal_interrupted(campaign, ledger, restoration):
    """Preserve an interrupted run and consume its reservation without rerunning it."""
    from .research_report import seal
    reference = ledger['budget_reservation']
    output = ROOT/'runs'/(campaign.name+'-compare')
    output.mkdir(exist_ok=True)
    observations, _ = read_jsonl(output/'cpu-compare.jsonl')
    summary = read(output/'summary.json') if (output/'summary.json').exists() else dict(
        complete=False, accepted=False, decision='interrupted', metrics=[], defaults_changed=False)
    disposition = 'inconclusive' if scientific_outcome(observations, summary) or not restoration['restored'] else 'infrastructure_failed'
    final = budgets.abandon(budget_path(), reference['reservation_id'], disposition)
    ledger['budget_disposition'] = disposition
    charge = reference['reservation']['reserved_seconds']
    if not (output/'checksums.json').exists():
        atomic_json(output/'summary.json', summary)
        atomic_json(output/'validation.json', dict(passed=False, errors=['Service interruption; no scientific qualification']))
        atomic_json(output/'governor-restoration.json', dict(original=ledger['original_governor'],
                    current=restoration['governor'], restored=restoration['restored'],
                    error=None if restoration['restored'] else 'Governor restoration is unverified'))
        atomic_json(output/'runtime-finalization.json', dict(started_monotonic=None, finished_monotonic=None,
                    charged_seconds=charge, reserved_seconds=charge, original_governor=ledger['original_governor'],
                    restored=restoration['restored'], charge_policy='conservative_full_reservation'))
        atomic_json(output/'budget-reservation.json', reference)
        atomic_json(output/'budget-finalization.json', final)
        ensure_report(output, summary, restoration['restored'])
        (output/'CONTEXT.txt').write_text('Service interrupted. Full reserved allowance charged. '
            'No successful scientific qualification is claimed. Read the cumulative budget receipt.\n')
        seal(output)
    runtime = read(output/'runtime-finalization.json')
    ledger['attempts'] = [dict(stage='cpu-compare', state='interrupted', output=str(output),
        reserved_seconds=charge, charged_seconds=runtime['charged_seconds'], restored=restoration['restored'],
        scientific_complete=bool(summary.get('complete')), audit_passed=False, decision='interrupted',
        checksums_sha256=digest_file(output/'checksums.json'))]
    atomic_json(campaign/'campaign.json', ledger)
    atomic_json(campaign/'budget-finalization.json', final)
    atomic_json(campaign/'interruption.json', dict(utc=utc(), restoration=restoration,
                conservative_charge_seconds=charge, no_automatic_retry=True))
    atomic_json(campaign/'final-validation.json', dict(passed=False, summary=summary,
                disposition=disposition, output=str(output), defaults_changed=False))
    (campaign/'CONTEXT.txt').write_text('Service interrupted; the full budget reservation was charged. '
        'Historical evidence is preserved and no preset was exported.\n')
    seal(campaign)
    budgets.attach_evidence(budget_path(), reference['reservation_id'], campaign)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('start', 'status', 'stop', '_run', '_stopped'):
        command = commands.add_parser(name); command.add_argument('--campaign', type=Path, required=True)
        if name == 'start':
            command.add_argument('--source-campaign', type=Path, required=True)
            command.add_argument('--validation', type=Path, required=True)
            command.add_argument('--retry-from', type=Path,
                                 help='Sealed infrastructure failure; cumulative original allowance is enforced')
            command.add_argument('--max-seconds', type=float,
                                 help='Optional lower cap; default is the exact remaining original allowance')
    args = parser.parse_args(argv); campaign = args.campaign.resolve()
    try:
        if args.command == 'start':
            return start(args)
        if args.command == 'status':
            from .reliability_service import service_properties
            value = read(state_dir(campaign)/'status.json')
            value['service'] = service_properties(unit_name(campaign))
            if value.get('output') and (Path(value['output'])/'progress.json').exists():
                value['progress'] = read(Path(value['output'])/'progress.json')
            print(json.dumps(value, indent=2)); return 0
        if args.command == 'stop':
            atomic_json(state_dir(campaign)/'stop-request.json', dict(utc=utc()))
            subprocess.run(['systemctl', '--user', 'stop', unit_name(campaign)], check=True); return 0
        if args.command == '_stopped':
            stopped(campaign); return 0
        from .completion_evidence import require_managed_service
        require_managed_service(unit_name(campaign))
        return execute(campaign)
    except BaseException as exc:
        if args.command == '_run':
            update(campaign, 'needs_attention', error=f'{type(exc).__name__}: {exc}')
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
