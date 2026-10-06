"""Bounded CPU supervision diagnostic, excluded from all scientific ledgers."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import numpy as np

from .attention_native import fixture, oracle
from .common import ROOT, atomic_json, digest_file, emit, stop_process_group, utc
from .coordination_workers import Worker
from .feedback_spec import input_hash
from .monitor import sample, safety_reason
from .refine_governor import GOVERNOR, request
from .refine_worker import environment
from .reliability_spec import CPU_ARMS, CPU_POLICIES, POLICIES, arm
from .reliability_worker import ReliableNumeric
from .study_spec import CELLS

MAX_SECONDS = 1200
CLEANUP_SECONDS = 120
POLICY_SECONDS = 181
EVENTS = 'diagnostic.jsonl'


class Hardware:
    """Hardware boundary; tests supply a clock and a simulated implementation."""

    now = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def __init__(self, directory):
        self.directory = directory
        self.lock = self.helper = self.build_process = self.active_worker = None
        self.socket = self.temporary = self.helper_log = None
        self.lease_ready = False

    def acquire(self):
        self.lock = (ROOT / 'runs/.experiment.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def governor(self):
        return GOVERNOR.read_text().strip()

    def observe(self):
        return sample(self.directory)

    def authenticate(self, guard):
        # Runtime sockets need a short pathname even when the artifact path is long.
        self.temporary = tempfile.TemporaryDirectory(prefix='cpu-smoke-')
        self.socket = Path(self.temporary.name) / 'lease.sock'
        self.helper_log = (self.directory / 'governor-helper.log').open('w')
        self.helper = subprocess.Popen([
            'pkexec', sys.executable, str(ROOT / 'efficiency/refine_governor.py'),
            '--socket', str(self.socket), '--pid', str(os.getpid()),
            '--seconds', str(guard.deadline - self.now()),
        ], stdout=self.helper_log, stderr=subprocess.STDOUT, start_new_session=True)
        auth_deadline = min(guard.work_deadline, self.now() + 120)
        while not self.socket.exists() and self.helper.poll() is None:
            guard.check()
            if self.now() >= auth_deadline:
                raise PermissionError('Administrator authentication exceeded 120 seconds')
            self.sleep(.1)
        if self.now() >= auth_deadline:
            raise PermissionError('Administrator authentication exceeded 120 seconds')
        if not self.socket.exists():
            raise PermissionError('Administrator authentication unavailable')
        self.lease_ready = True
        value = request(self.socket)
        if value['original'] != guard.original:
            raise RuntimeError('Governor baseline changed before the lease started')

    def heartbeat(self):
        if self.lease_ready:
            if self.helper.poll() is not None:
                raise RuntimeError('Governor helper exited')
            request(self.socket)

    def select_policy(self, policy):
        return request(self.socket, POLICIES[policy]['governor'])

    def prepare(self, guard):
        # Poll the build process so telemetry, STOP and the lease remain live.
        with (self.directory / 'build.log').open('w') as log:
            self.build_process = subprocess.Popen([
                sys.executable, '-B', '-c',
                'from efficiency.refine_campaign import build; build()',
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while self.build_process.poll() is None:
                    guard.check()
                    self.sleep(.1)
                if self.build_process.returncode:
                    raise RuntimeError('Native build failed; see build.log')
            finally:
                if self.build_process.poll() is None:
                    stop_process_group(self.build_process)
        shutil.copytree(ROOT / 'build/attention', self.directory / 'native-build')
        for name in ('fixtures', 'outputs'):
            (self.directory / name).mkdir()
        namespace = 'diagnostic-' + str(uuid.uuid4())
        fixtures = []
        for cell in CELLS:
            guard.check()
            fid = 'diagnostic-' + cell['cell_id']
            seed = int.from_bytes(hashlib.sha256(f'{namespace}|{fid}'.encode()).digest()[:8], 'little')
            x, w = fixture(cell['n'], cell['d'], cell['b'], seed)
            expected = oracle(x, w)
            path = self.directory / 'fixtures' / (fid + '.npz')
            np.savez(path, x=x, w=w, expected=expected)
            fixtures.append(dict(cell, fixture_id=fid, seed=seed,
                                 file_sha256=digest_file(path), input_sha256=input_hash(x),
                                 weights_sha256=input_hash(w), oracle_sha256=input_hash(expected)))
        atomic_json(self.directory / 'fixtures.json', fixtures)
        atomic_json(self.directory / 'fixture-namespace.json', dict(namespace=namespace, diagnostic_only=True))
        return fixtures

    def worker(self, policy, guard):
        value = Worker('process', 'numeric', self.directory,
                       dict(block_id=f'diagnostic-policy{policy}', policy=policy, numeric_device='cpu'),
                       guard.check, factory=ReliableNumeric, environment=environment(POLICIES[policy]))
        self.active_worker = value
        return value

    def release_worker(self, worker):
        worker.close()
        self.active_worker = None
        return worker.release

    def cleanup(self, deadline):
        errors = []
        if self.active_worker is not None:
            try:
                self.release_worker(self.active_worker)
            except Exception as exc:
                errors.append(f'Worker cleanup: {exc}')
        if self.build_process is not None and self.build_process.poll() is None:
            try:
                stop_process_group(self.build_process)
            except Exception as exc:
                errors.append(f'Build cleanup: {exc}')
        if self.socket is not None and self.socket.exists():
            try:
                request(self.socket, close=True)
            except Exception as exc:
                errors.append(f'Lease close: {exc}')
        if self.helper is not None:
            if not self.lease_ready and self.helper.poll() is None:
                try:
                    self.helper.terminate()
                except (OSError, ProcessLookupError) as exc:
                    errors.append(f'Authentication cancellation: {exc}')
            # If close failed, withholding heartbeat lets the 60-second watchdog restore.
            while self.helper.poll() is None and self.now() < deadline:
                self.sleep(min(.2, max(0, deadline - self.now())))
            if self.helper.poll() is None:
                errors.append('Governor helper still active at cleanup deadline')
        inactive = self.helper is None or self.helper.poll() is not None
        if self.helper_log is not None:
            self.helper_log.close()
        if self.temporary is not None and inactive:
            self.temporary.cleanup()
        return dict(helper_exited=inactive, errors=errors)

    def unlock(self):
        if self.lock is not None:
            self.lock.close()


class Guard:
    def __init__(self, directory, hardware, seconds):
        self.directory, self.hardware = directory, hardware
        self.started = hardware.now()
        self.deadline = self.started + seconds
        self.work_deadline = self.deadline - CLEANUP_SECONDS
        self.stop_reason = None
        self.original = self.expected_governor = None
        self.phase = 'preflight'
        self.initial_flags = None
        self.last_sample = self.last_heartbeat = float('-inf')
        self.latest = None
        self.samples = 0

    def stop(self, signum, _frame):
        self.stop_reason = self.stop_reason or f'Interrupted by signal {signum}'

    def check(self):
        now = self.hardware.now()
        if self.stop_reason:
            raise RuntimeError(self.stop_reason)
        if (self.directory / 'STOP').exists():
            raise RuntimeError('Operator STOP file detected')
        if now >= self.work_deadline:
            raise TimeoutError('Diagnostic work deadline reached; 120 seconds reserved for cleanup')
        if now - self.last_heartbeat >= 5:
            self.hardware.heartbeat()
            self.last_heartbeat = self.hardware.now()
        if now - self.last_sample >= 1:
            row = self.hardware.observe()
            if self.initial_flags is None:
                self.initial_flags = row.get('throttle_flags') or 0
            reason = safety_reason(row, self.initial_flags, require_hat=False)
            observed = self.hardware.governor()
            if observed != self.expected_governor:
                reason = reason or 'Governor drift during diagnostic'
            row.update(phase=self.phase, governor=observed, stop_reason=reason,
                       elapsed_seconds=self.hardware.now() - self.started)
            emit(self.directory / 'telemetry.jsonl', 'sample', **row)
            atomic_json(self.directory / 'status.json', row)
            self.latest = row
            self.samples += 1
            self.last_sample = self.hardware.now()
            if reason:
                raise RuntimeError(reason)
        if self.hardware.now() >= self.work_deadline:
            raise TimeoutError('Diagnostic work deadline reached; 120 seconds reserved for cleanup')

    def cool(self):
        while True:
            self.check()
            if self.latest['cpu_temp_c'] < 65:
                return
            self.hardware.sleep(.25)


def verify_response(response, *, batch=False):
    result = response['result']
    rows = result.get('requests', []) if batch else [result]
    if (not result.get('correct') or not rows or (batch and len(rows) != 16)
            or any(not row.get('correct') or not row.get('matches_warmup')
                   or row.get('validation_errors') != 0 for row in rows)):
        raise RuntimeError('Diagnostic numerical correctness or warmup verification failed')
    return len(rows)


def run(output: Path, max_seconds=MAX_SECONDS, *, _hardware=None):
    """Run once in a new directory; never qualify or charge a scientific stage."""
    if (not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds)
            or not CLEANUP_SECONDS < max_seconds <= MAX_SECONDS):
        raise ValueError('Diagnostic allowance must be finite, over 120 and at most 1200 seconds')
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    hardware = _hardware or Hardware(directory)
    guard = Guard(directory, hardware, max_seconds)
    handlers = {}
    if threading.current_thread() is threading.main_thread():
        handlers = {sig: signal.signal(sig, guard.stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    policies, reason, complete = [], None, False
    restoration = dict(original=None, current=None, restored=False, helper_exited=None, errors=[])
    try:
        hardware.acquire()
        guard.original = guard.expected_governor = hardware.governor()
        atomic_json(directory / 'config.json', dict(
            diagnostic_only=True, scientific_qualification=False, scientific_charge_seconds=0,
            max_seconds=max_seconds, cleanup_seconds=CLEANUP_SECONDS,
            minimum_policy_seconds=POLICY_SECONDS, cpu_policies=list(CPU_POLICIES),
            original_governor=guard.original, controller_pid=os.getpid(), started_utc=utc()))
        guard.phase = 'authentication'
        guard.check()
        hardware.authenticate(guard)
        guard.phase = 'build-and-fixtures'
        fixtures = hardware.prepare(guard)
        for policy in CPU_POLICIES:
            guard.phase = f'policy{policy}'
            guard.cool()
            transition = hardware.select_policy(policy)
            guard.expected_governor = POLICIES[policy]['governor']
            emit(directory / EVENTS, 'governor_transition', policy=policy, response=transition)
            guard.last_sample = float('-inf')
            guard.check()
            worker = hardware.worker(policy, guard)
            emit(directory / EVENTS, 'worker_ready', policy=policy, response=worker.ready)
            started = hardware.now()
            numerical_calls, cycles = 0, 0
            try:
                while hardware.now() - started < POLICY_SECONDS or not cycles:
                    for cell in fixtures:
                        guard.cool()
                        fid = cell['fixture_id']
                        worker.call('load', fixture_id=fid)
                        for label in CPU_ARMS:
                            response = worker.call('warm', fixture_id=fid, arm=arm(cell, label))
                            verify_response(response)
                            emit(directory / EVENTS, 'warmup', policy=policy, fixture_id=fid,
                                 label=label, response=response)
                        for label in CPU_ARMS:
                            guard.check()
                            batch = label.startswith('batch')
                            payload = dict(fixture_id=fid, arm=arm(cell, label))
                            if batch:
                                payload.update(request_ids=list(range(16)), deadline=guard.work_deadline)
                            response = worker.call('batch' if batch else 'measure', **payload)
                            numerical_calls += verify_response(response, batch=batch)
                            emit(directory / EVENTS, 'measurement', policy=policy, fixture_id=fid,
                                 label=label, response=response, diagnostic_only=True)
                    cycles += 1
                guard.check()
            finally:
                release = hardware.release_worker(worker)
                emit(directory / EVENTS, 'worker_release', policy=policy, **release)
                if release.get('alive') or release.get('forced'):
                    raise RuntimeError('Diagnostic worker did not shut down cleanly')
            policies.append(dict(policy=policy, healthy_seconds=hardware.now() - started,
                                 numerical_calls=numerical_calls, cycles=cycles))
            atomic_json(directory / 'progress.json', dict(phase=guard.phase, policies=policies, utc=utc()))
        complete = True
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        guard.phase = 'cleanup'
        restoration['original'] = guard.original
        cleanup_deadline = min(guard.deadline, hardware.now() + CLEANUP_SECONDS)
        try:
            restoration.update(hardware.cleanup(cleanup_deadline))
        except BaseException as exc:
            restoration['errors'].append(f'{type(exc).__name__}: {exc}')
        try:
            restoration['current'] = hardware.governor()
        except BaseException as exc:
            restoration['errors'].append(f'Governor readback: {exc}')
        restoration['restored'] = (guard.original is not None
                                   and restoration['current'] == guard.original
                                   and restoration['helper_exited'] is True)
        restoration.update(utc=utc(), elapsed_seconds=hardware.now() - guard.started)
        try:
            atomic_json(directory / 'governor-restoration.json', restoration)
        finally:
            hardware.unlock()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    elapsed = hardware.now() - guard.started
    healthy = sum(row['healthy_seconds'] for row in policies)
    reason = reason or guard.stop_reason
    if (directory / 'STOP').exists():
        reason = reason or 'Operator STOP file detected'
    passed = (complete and reason is None and healthy > 720 and len(policies) == len(CPU_POLICIES)
              and restoration['restored'] and not restoration['errors'] and elapsed <= max_seconds)
    if not passed and reason is None:
        reason = 'Diagnostic incomplete, cleanup failed, or allowance exceeded'
    summary = dict(passed=passed, reason=reason, diagnostic_only=True,
                   scientific_qualification=False, scientific_charge_seconds=0,
                   elapsed_seconds=elapsed, healthy_seconds=healthy, policies=policies,
                   telemetry_samples=guard.samples, restoration=restoration, finished_utc=utc())
    atomic_json(directory / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    args = parser.parse_args()
    result = run(args.output, args.max_seconds)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
