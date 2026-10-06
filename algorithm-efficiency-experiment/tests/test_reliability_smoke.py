import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficiency import reliability_smoke as smoke


class SimulatedWorker:
    def __init__(self, hardware, guard):
        self.hardware, self.guard = hardware, guard
        self.ready = dict(owner=dict(pid=123), environment=dict(numeric_device='cpu'))

    def call(self, operation, **payload):
        self.guard.check()
        self.hardware.advance(1)
        if operation == 'load':
            return dict(result=dict(fixture_id=payload['fixture_id']))
        row = dict(correct=not self.hardware.wrong_answer, matches_warmup=True,
                   validation_errors=0)
        if operation == 'batch':
            result = dict(correct=row['correct'], requests=[dict(row) for _ in range(16)])
        else:
            result = row
        return dict(result=result)


class SimulatedHardware:
    def __init__(self, directory):
        self.directory, self.clock = directory, 0.
        self.current, self.original = 'ondemand', 'ondemand'
        self.heartbeats, self.releases, self.selected = [], [], []
        self.wrong_answer = self.throttled = self.forced = self.fail_restore = False
        self.unlocked = self.cleaned = False
        self.on_prepare = None

    def now(self):
        return self.clock

    def advance(self, seconds):
        self.clock += seconds

    sleep = advance

    def acquire(self):
        pass

    def governor(self):
        return self.current

    def observe(self):
        return dict(monotonic=self.clock, cpu_temp_c=42,
                    throttle_flags=4 if self.throttled else 0,
                    available_memory_bytes=1024**3)

    def heartbeat(self):
        self.heartbeats.append(self.clock)

    def authenticate(self, guard):
        self.advance(2)
        guard.check()

    def prepare(self, guard):
        if self.on_prepare:
            self.on_prepare(guard)
        self.advance(10)
        guard.check()
        return [dict(cell, fixture_id=cell['cell_id']) for cell in smoke.CELLS]

    def select_policy(self, policy):
        self.selected.append(policy)
        self.current = smoke.POLICIES[policy]['governor']
        return dict(current=self.current)

    def worker(self, policy, guard):
        return SimulatedWorker(self, guard)

    def release_worker(self, worker):
        self.releases.append(worker)
        return dict(alive=False, forced=self.forced)

    def cleanup(self, deadline):
        self.cleaned = True
        self.current = 'performance' if self.fail_restore else self.original
        return dict(helper_exited=True, errors=[])

    def unlock(self):
        self.unlocked = True


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / 'diagnostic'
        self.hardware = SimulatedHardware(self.output)

    def run_diagnostic(self, seconds=1200):
        return smoke.run(self.output, seconds, _hardware=self.hardware)

    def assert_receipts(self):
        self.assertTrue(self.hardware.cleaned)
        self.assertTrue(self.hardware.unlocked)
        summary = json.loads((self.output / 'summary.json').read_text())
        receipt = json.loads((self.output / 'governor-restoration.json').read_text())
        self.assertEqual(summary['restoration'], receipt)
        self.assertFalse(summary['scientific_qualification'])
        self.assertEqual(summary['scientific_charge_seconds'], 0)
        self.assertFalse((self.output / 'campaign.json').exists())

    def test_all_policies_over_twelve_minutes_with_checked_calls(self):
        result = self.run_diagnostic()
        self.assertTrue(result['passed'], result)
        self.assertGreater(result['healthy_seconds'], 720)
        self.assertLessEqual(result['elapsed_seconds'], 1200)
        self.assertEqual(self.hardware.selected, list(smoke.CPU_POLICIES))
        self.assertEqual(len(self.hardware.releases), 4)
        self.assertTrue(all(row['numerical_calls'] > 0 and row['healthy_seconds'] >= 181
                            for row in result['policies']))
        self.assertGreater(len(self.hardware.heartbeats), 100)
        self.assert_receipts()

    def test_incorrect_output_blocks_success_and_restores_governor(self):
        self.hardware.wrong_answer = True
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('numerical correctness', result['reason'])
        self.assertEqual(len(self.hardware.releases), 1)
        self.assertTrue(result['restoration']['restored'])
        self.assert_receipts()

    def test_throttling_stops_before_workers(self):
        self.hardware.throttled = True
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('throttling', result['reason'])
        self.assertEqual(self.hardware.selected, [])
        self.assert_receipts()

    def test_stop_during_preparation_still_creates_receipts(self):
        self.hardware.on_prepare = lambda guard: (self.output / 'STOP').touch()
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('STOP', result['reason'])
        self.assertEqual(self.hardware.selected, [])
        self.assert_receipts()

    def test_signal_stop_restores_handlers_and_governor(self):
        previous = signal.getsignal(signal.SIGTERM)
        self.hardware.on_prepare = lambda guard: guard.stop(signal.SIGTERM, None)
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('signal', result['reason'])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)
        self.assert_receipts()

    def test_work_deadline_reserves_cleanup(self):
        result = self.run_diagnostic(130)
        self.assertFalse(result['passed'])
        self.assertIn('120 seconds reserved', result['reason'])
        self.assertLess(result['elapsed_seconds'], 130)
        self.assert_receipts()

    def test_governor_drift_during_build_stops(self):
        self.hardware.on_prepare = lambda guard: setattr(self.hardware, 'current', 'performance')
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('Governor drift', result['reason'])
        self.assert_receipts()

    def test_forced_worker_cleanup_cannot_pass(self):
        self.hardware.forced = True
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('shut down cleanly', result['reason'])
        self.assert_receipts()

    def test_restore_readback_is_required(self):
        self.hardware.fail_restore = True
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertFalse(result['restoration']['restored'])
        self.assert_receipts()

    def test_preparation_exception_is_an_explicit_failure(self):
        def fail(guard):
            raise FileNotFoundError('missing compiler')
        self.hardware.on_prepare = fail
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('missing compiler', result['reason'])
        self.assert_receipts()

    def test_long_build_keeps_lease_and_telemetry_alive(self):
        def build(guard):
            for _ in range(80):
                self.hardware.advance(1)
                guard.check()
        self.hardware.on_prepare = build
        result = self.run_diagnostic()
        self.assertTrue(result['passed'], result)
        preparation_beats = [stamp for stamp in self.hardware.heartbeats if 2 <= stamp <= 82]
        self.assertGreaterEqual(len(preparation_beats), 15)
        self.assertLessEqual(max(b - a for a, b in zip(preparation_beats, preparation_beats[1:])), 5)
        self.assert_receipts()

    def test_stop_during_cleanup_cannot_report_success(self):
        cleanup = self.hardware.cleanup
        def stopped(deadline):
            (self.output / 'STOP').touch()
            return cleanup(deadline)
        self.hardware.cleanup = stopped
        result = self.run_diagnostic()
        self.assertFalse(result['passed'])
        self.assertIn('STOP', result['reason'])
        self.assert_receipts()

    def test_preexisting_output_is_immutable(self):
        self.output.mkdir()
        path = self.output / 'summary.json'
        path.write_text('original')
        with self.assertRaises(FileExistsError):
            self.run_diagnostic()
        self.assertEqual(path.read_text(), 'original')

    def test_invalid_allowance_never_creates_output(self):
        for seconds in (float('nan'), float('inf'), -1, 120, 1201):
            with self.assertRaises(ValueError):
                self.run_diagnostic(seconds)
            self.assertFalse(self.output.exists())


class HardwareBoundaryTests(unittest.TestCase):
    def test_authentication_wait_has_120_second_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            hardware = smoke.Hardware(Path(temporary))
            clock = [0.]
            hardware.now = lambda: clock[0]
            hardware.sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
            guard = SimpleNamespace(deadline=1200, work_deadline=1080, check=lambda: None)
            child = SimpleNamespace(poll=lambda: None)
            try:
                with patch.object(smoke.subprocess, 'Popen', return_value=child):
                    with self.assertRaisesRegex(PermissionError, '120 seconds'):
                        hardware.authenticate(guard)
                self.assertGreaterEqual(clock[0], 120)
                self.assertLess(clock[0], 121)
            finally:
                hardware.helper_log.close()
                hardware.temporary.cleanup()

    def test_missing_incomplete_or_bad_batch_cannot_pass(self):
        good = dict(correct=True, matches_warmup=True, validation_errors=0)
        for result in (dict(correct=True, requests=[]),
                       dict(correct=True, requests=[good] * 15),
                       dict(correct=True, requests=[dict(good, validation_errors=1)] * 16)):
            with self.assertRaisesRegex(RuntimeError, 'numerical correctness'):
                smoke.verify_response(dict(result=result), batch=True)


if __name__ == '__main__':
    unittest.main()
