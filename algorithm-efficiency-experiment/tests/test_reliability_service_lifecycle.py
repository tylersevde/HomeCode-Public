"""Opt-in user-systemd lifecycle checks with file-backed fake governor state.

Run with RUN_SYSTEMD_TESTS=1 python3 -B -m unittest
tests.test_reliability_service_lifecycle. No fixture accesses sysfs or pkexec.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficiency.refine_governor import Lease, controller_start


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def identity(pid):
    return controller_start(Path('/proc') / str(pid) / 'stat')


def alive(record):
    return identity(record['pid']) == record['start_ticks']


def save_identity(state, role):
    record = dict(pid=os.getpid(), start_ticks=identity(os.getpid()),
                  session=os.getsid(0), cgroup=Path('/proc/self/cgroup').read_text())
    write_json(state / (role + '.json'), record)
    return record


def fixture_command(role, state, *extra):
    return [sys.executable, '-B', str(SCRIPT), '--fixture', role,
            '--state', str(state), *extra]


def stop_child(child):
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=2)


def interruption(signum, frame):
    raise SystemExit(128 + signum)


def fixture_worker(state):
    save_identity(state, 'worker')
    signal.signal(signal.SIGTERM, interruption)
    while not (state / 'complete-work').exists():
        time.sleep(.05)
    write_json(state / 'worker-result.json', dict(complete=True))


def fixture_watchdog(state, controller_pid):
    save_identity(state, 'watchdog')
    governor = state / 'fake-governor'
    start_ticks = identity(controller_pid)
    lease = Lease(lambda: governor.read_text(), governor.write_text,
                  lambda: identity(controller_pid) == start_ticks,
                  time.monotonic, 30)
    reason = 'watchdog_exit'

    def stopping(signum, frame):
        nonlocal reason
        reason = 'signal_' + signal.Signals(signum).name
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    (state / 'watchdog-ready').touch()
    try:
        while lease.check():
            time.sleep(.05)
    finally:
        lease.restore(reason)
        write_json(state / 'watchdog-result.json', dict(
            restored=governor.read_text() == 'ondemand',
            reason=lease.restoration_reason, complete=False))


def fixture_controller(state):
    save_identity(state, 'controller')
    signal.signal(signal.SIGTERM, interruption)
    signal.signal(signal.SIGINT, interruption)
    children = []
    complete = False
    try:
        watchdog = subprocess.Popen(fixture_command('watchdog', state,
            '--controller-pid', str(os.getpid())), start_new_session=True)
        children.append(watchdog)
        deadline = time.monotonic() + 5
        while not (state / 'watchdog-ready').exists():
            if watchdog.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('Fake watchdog failed to start')
            time.sleep(.02)
        (state / 'fake-governor').write_text('performance')
        worker = subprocess.Popen(fixture_command('worker', state), start_new_session=True)
        children.append(worker)
        while worker.poll() is None and watchdog.poll() is None:
            time.sleep(.05)
        result = state / 'worker-result.json'
        complete = (worker.poll() == 0 and result.exists()
                    and json.loads(result.read_text()).get('complete') is True)
        if not complete:
            raise RuntimeError('A fixture child exited before completing work')
    finally:
        for child in children:
            stop_child(child)
        write_json(state / 'controller-result.json', dict(complete=complete))
    return 0


def fixture_driver(state):
    save_identity(state, 'driver')
    signal.signal(signal.SIGTERM, interruption)
    signal.signal(signal.SIGINT, interruption)
    child = subprocess.Popen(fixture_command('controller', state))
    complete = False
    code = None
    try:
        code = child.wait()
        result = state / 'controller-result.json'
        complete = (code == 0 and result.exists()
                    and json.loads(result.read_text()).get('complete') is True
                    and (state / 'fake-governor').read_text() == 'ondemand')
        return 0 if complete else 1
    finally:
        stop_child(child)
        write_json(state / 'driver-result.json', dict(complete=complete, returncode=code))


def fixture_stopped(state):
    records = {role: json.loads((state / (role + '.json')).read_text())
               for role in ('driver', 'controller', 'worker', 'watchdog')
               if (state / (role + '.json')).exists()}
    result = state / 'driver-result.json'
    complete = (os.environ.get('SERVICE_RESULT') == 'success' and result.exists()
                and json.loads(result.read_text()).get('complete') is True)
    write_json(state / 'stopped.json', dict(
        service_result=os.environ.get('SERVICE_RESULT'),
        exit_code=os.environ.get('EXIT_CODE'), exit_status=os.environ.get('EXIT_STATUS'),
        governor_restored=(state / 'fake-governor').read_text() == 'ondemand',
        live_roles=[role for role, record in records.items() if alive(record)],
        complete=complete))


def fixture_launch(state, unit):
    # The caller exits immediately; the user manager must own the complete tree.
    stopped = ' '.join(fixture_command('stopped', state))
    command = ['systemd-run', '--user', '--quiet', '--unit=' + unit,
               '--service-type=exec', '--expand-environment=no',
               '--property=Restart=no', '--property=KillMode=control-group',
               '--property=TimeoutStopSec=5', '--property=RuntimeMaxSec=30',
               '--property=StandardOutput=append:' + str(state / 'service.log'),
               '--property=StandardError=append:' + str(state / 'service.log'),
               '--property=ExecStopPost=' + stopped,
               '--setenv=PYTHONDONTWRITEBYTECODE=1', *fixture_command('driver', state)]
    subprocess.run(command, check=True, timeout=8)


@unittest.skipUnless(os.environ.get('RUN_SYSTEMD_TESTS') == '1',
                     'Set RUN_SYSTEMD_TESTS=1 to run fake user-service tests')
class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.deadline = time.monotonic() + 45
        self.temporary = tempfile.TemporaryDirectory(prefix='reliability-service-test-')
        self.state = Path(self.temporary.name)
        self.unit = 'reliability-fake-' + uuid.uuid4().hex + '.service'
        self.addCleanup(self.cleanup)
        (self.state / 'fake-governor').write_text('ondemand')
        manager = self.systemctl('is-system-running', check=False)
        if manager.stdout.strip() not in ('running', 'degraded'):
            self.fail('The explicitly requested systemd tests need an active user manager')

    def remaining_timeout(self, limit):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Fixture exceeded its 45-second test deadline')
        return min(limit, remaining)

    def systemctl(self, *args, check=True, timeout=8):
        return subprocess.run(['systemctl', '--user', *args], check=check,
                              text=True, capture_output=True,
                              timeout=self.remaining_timeout(timeout))

    def cleanup(self):
        try:
            try:
                self.systemctl('stop', self.unit, check=False, timeout=6)
            finally:
                self.systemctl('reset-failed', self.unit, check=False, timeout=3)
        finally:
            self.temporary.cleanup()

    def wait_for(self, predicate, seconds=12):
        # Keep time for bounded stop/reset-failed even when an assertion fails.
        deadline = min(self.deadline - 10, time.monotonic() + seconds)
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.05)
        log = self.state / 'service.log'
        self.fail('Fixture timed out; log: ' + (log.read_text() if log.exists() else '(none)'))

    def launch(self):
        launcher = subprocess.run(fixture_command('launch', self.state, '--unit', self.unit),
                                  text=True, capture_output=True,
                                  timeout=self.remaining_timeout(10))
        self.assertEqual(launcher.returncode, 0, launcher.stderr)
        self.wait_for(lambda: all((self.state / (role + '.json')).exists()
                                 for role in ('driver', 'controller', 'worker', 'watchdog')))
        self.records = {role: json.loads((self.state / (role + '.json')).read_text())
                        for role in ('driver', 'controller', 'worker', 'watchdog')}
        self.assertEqual((self.state / 'fake-governor').read_text(), 'performance')
        for role, record in self.records.items():
            self.assertTrue(alive(record), role)
            self.assertIn(self.unit, record['cgroup'])
        for role in ('worker', 'watchdog'):
            self.assertEqual(self.records[role]['session'], self.records[role]['pid'])

    def kill_role(self, role):
        record = self.records[role]
        self.assertTrue(alive(record), 'Refusing to signal a stale fixture identity')
        os.kill(record['pid'], signal.SIGKILL)

    def assert_stopped_without_success(self):
        self.wait_for(lambda: (self.state / 'stopped.json').exists())
        self.wait_for(lambda: all(not alive(record) for record in self.records.values()))
        receipt = json.loads((self.state / 'stopped.json').read_text())
        self.assertEqual(receipt['live_roles'], [])
        self.assertTrue(receipt['governor_restored'], receipt)
        self.assertFalse(receipt['complete'])
        watchdog = json.loads((self.state / 'watchdog-result.json').read_text())
        self.assertTrue(watchdog['restored'], watchdog)
        self.assertEqual((self.state / 'fake-governor').read_text(), 'ondemand')
        for role in ('driver', 'controller', 'watchdog'):
            path = self.state / (role + '-result.json')
            if path.exists():
                self.assertFalse(json.loads(path.read_text())['complete'])
        self.wait_for(lambda: self.systemctl('show', self.unit, '--property=ActiveState',
                      '--value', check=False).stdout.strip() != 'deactivating')
        active = self.systemctl('show', self.unit, '--property=ActiveState', '--value', check=False)
        self.assertNotIn(active.stdout.strip(), ('active', 'activating', 'deactivating'))
        return receipt

    def test_launcher_exits_service_survives_and_graceful_stop_restores(self):
        self.launch()
        time.sleep(.25)
        active = self.systemctl('show', self.unit, '--property=ActiveState', '--value')
        self.assertEqual(active.stdout.strip(), 'active')
        self.assertTrue(alive(self.records['driver']))
        self.systemctl('stop', self.unit)
        self.assert_stopped_without_success()

    def test_controller_sigkill_stops_new_session_descendants(self):
        self.launch()
        self.kill_role('controller')
        self.assert_stopped_without_success()

    def test_driver_sigkill_stops_tree_and_never_reports_success(self):
        self.launch()
        self.kill_role('driver')
        receipt = self.assert_stopped_without_success()
        self.assertEqual(receipt['service_result'], 'signal')

    def test_worker_sigkill_stops_driver_and_never_reports_success(self):
        self.launch()
        self.kill_role('worker')
        self.assert_stopped_without_success()

    def test_completed_fixture_can_report_success_after_restoration(self):
        self.launch()
        (self.state / 'complete-work').touch()
        self.wait_for(lambda: (self.state / 'stopped.json').exists())
        receipt = json.loads((self.state / 'stopped.json').read_text())
        self.assertTrue(receipt['complete'], receipt)
        self.assertTrue(receipt['governor_restored'], receipt)
        self.assertEqual(receipt['live_roles'], [])


def fixture_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--controller-pid', type=int)
    args = parser.parse_args()
    if args.fixture == 'launch':
        return fixture_launch(args.state, args.unit)
    if args.fixture == 'watchdog':
        return fixture_watchdog(args.state, args.controller_pid)
    return {'driver': fixture_driver, 'controller': fixture_controller,
            'worker': fixture_worker, 'stopped': fixture_stopped}[args.fixture](args.state)


if __name__ == '__main__':
    if '--fixture' in sys.argv:
        sys.exit(fixture_main())
    unittest.main()
