"""Governor lifecycle diagnostics using fake clocks, processes, and sockets."""
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import signal
import socket
import struct
import unittest
from unittest.mock import MagicMock, patch

from efficiency.refine_governor import Lease, controller_start, serve


class FakeLeaseCase(unittest.TestCase):
    def setUp(self):
        self.clock = 0
        self.value = 'ondemand'
        self.alive = True
        self.writes = []

        def write(value):
            self.value = value
            self.writes.append(value)

        self.lease = Lease(lambda: self.value, write, lambda: self.alive,
                           lambda: self.clock, 7200)


class LeaseLifecycleTests(FakeLeaseCase):
    def test_controller_loss_preserves_reason_through_final_cleanup(self):
        self.lease.request('performance')
        self.clock = 2
        self.alive = False
        self.assertFalse(self.lease.check())
        self.clock = 3
        self.lease.restore('helper_exit')
        self.assertEqual(self.value, 'ondemand')
        self.assertEqual(self.lease.restoration_reason, 'controller_unavailable')
        self.assertEqual(self.lease.restored_monotonic, 2)
        self.assertFalse(self.lease.check())

    def test_heartbeat_boundary_and_refresh(self):
        self.lease.request('performance')
        self.clock = 60
        self.assertTrue(self.lease.check())
        self.lease.request()
        self.clock = 120
        self.assertTrue(self.lease.check())
        self.clock = 120.001
        self.assertFalse(self.lease.check())
        self.assertEqual(self.lease.restoration_reason, 'heartbeat_timeout')
        self.assertEqual(self.value, 'ondemand')

    def test_absolute_deadline_with_recent_heartbeat(self):
        self.lease.request('performance')
        for current in range(59, 7200, 59):
            self.clock = current
            self.lease.request()
        self.clock = 7200
        self.assertFalse(self.lease.check())
        self.assertEqual(self.lease.restoration_reason, 'deadline')
        self.assertEqual(self.lease.deadline, 7200)
        self.assertEqual(self.value, 'ondemand')

    def test_explicit_close_remains_closed(self):
        self.lease.request('performance')
        self.lease.restore()
        self.lease.restore('helper_exit')
        self.assertEqual(self.lease.restoration_reason, 'explicit_close')
        self.assertEqual(self.lease.restored_monotonic, 0)
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            self.lease.request('performance')

    def test_failed_restore_keeps_initiating_reason_on_retry(self):
        self.lease.request('performance')
        write = self.lease.write
        self.lease.write = lambda value: None
        with self.assertRaisesRegex(RuntimeError, 'restoration failed'):
            self.lease.restore('signal_SIGTERM')
        self.assertIsNone(self.lease.restored_monotonic)
        self.lease.write = write
        self.lease.restore('helper_exit')
        self.assertEqual(self.lease.restoration_reason, 'signal_SIGTERM')
        self.assertEqual(self.value, 'ondemand')


class ControllerIdentityTests(unittest.TestCase):
    @staticmethod
    def process(state='S', start=12345):
        process = MagicMock(spec=Path)
        # comm is enclosed by the last ')', even when it contains spaces/parentheses.
        fields = [state] + ['0'] * 18 + [str(start)] + ['0'] * 10
        process.read_text.return_value = '72 (controller ) with spaces) ' + ' '.join(fields)
        return process

    def test_spaced_parenthesized_name(self):
        self.assertEqual(controller_start(self.process()), '12345')

    def test_pid_reuse_changes_identity(self):
        process = self.process()
        original = controller_start(process)
        process.read_text.return_value = self.process(start=12346).read_text.return_value
        self.assertNotEqual(controller_start(process), original)

    def test_zombie_and_dead_states_are_not_alive(self):
        for state in ('Z', 'X', 'x'):
            with self.subTest(state=state):
                self.assertIsNone(controller_start(self.process(state=state)))

    def test_missing_unreadable_and_malformed_stat_fail_closed(self):
        for error in (FileNotFoundError(), PermissionError()):
            process = self.process()
            process.read_text.side_effect = error
            self.assertIsNone(controller_start(process))
        for text in ('', '72 no-comm S 1 2', '72 (name) S 1 2',
                     self.process(start='invalid').read_text.return_value):
            process = self.process()
            process.read_text.return_value = text
            self.assertIsNone(controller_start(process))


class HelperDiagnosticsTests(FakeLeaseCase):
    def run_helper(self, action, expected_exception=None):
        self.lease.request('performance')
        uid = 1234
        controller = dict(pid=72, start_ticks='12345', uid=uid)
        server = MagicMock()
        server.__enter__.return_value = server
        path = MagicMock(spec=Path)
        handlers = {signal.SIGTERM: 'old-term', signal.SIGINT: 'old-int'}
        original_handlers = dict(handlers)

        def install(signum, handler):
            previous = handlers[signum]
            handlers[signum] = handler
            return previous

        def accept():
            return action(handlers, uid)

        server.accept.side_effect = accept
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch('efficiency.refine_governor.socket.socket', return_value=server))
            stack.enter_context(patch('efficiency.refine_governor.os.chown'))
            stack.enter_context(patch('efficiency.refine_governor.os.chmod'))
            stack.enter_context(patch('efficiency.refine_governor.signal.signal', side_effect=install))
            stack.enter_context(contextlib.redirect_stdout(output))
            if expected_exception:
                with self.assertRaises(expected_exception):
                    serve(self.lease, path, uid, controller)
            else:
                serve(self.lease, path, uid, controller)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(handlers, original_handlers)
        path.unlink.assert_called_once_with(missing_ok=True)
        for event in events:
            self.assertEqual(event['controller'], controller)
            self.assertIsNotNone(datetime.fromisoformat(event['utc']).tzinfo)
            self.assertIsInstance(event['monotonic'], float)
        return events

    def test_signal_records_cause_and_restores(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum):
                self.setUp()

                def stop(handlers, uid):
                    handlers[signum](signum, None)

                events = self.run_helper(stop, SystemExit)
                restored = events[-1]
                self.assertEqual(restored['event'], 'restored')
                self.assertEqual(restored['reason'], 'signal_' + signum.name)
                self.assertEqual(restored['current'], 'ondemand')

    def test_controller_loss_records_cause(self):
        def disappear(handlers, uid):
            self.alive = False
            raise socket.timeout()

        events = self.run_helper(disappear)
        self.assertEqual(events[-1]['reason'], 'controller_unavailable')
        self.assertEqual(events[-1]['current'], 'ondemand')

    def test_explicit_close_socket_request_records_cause(self):
        def close(handlers, uid):
            conn = MagicMock()
            conn.__enter__.return_value = conn
            conn.getsockopt.return_value = struct.pack('3i', 999, uid, uid)
            conn.recv.return_value = b'{"close":true}\n'
            return conn, None

        events = self.run_helper(close)
        self.assertEqual([e['event'] for e in events], ['ready', 'request', 'restored'])
        self.assertEqual(events[-1]['reason'], 'explicit_close')
        self.assertEqual(events[-1]['current'], 'ondemand')

    def test_socket_failure_records_cause_and_restores(self):
        def fail(handlers, uid):
            raise OSError('simulated accept failure')

        events = self.run_helper(fail, OSError)
        self.assertEqual(events[-2]['event'], 'error')
        self.assertIn('simulated accept failure', events[-2]['error'])
        self.assertEqual(events[-1]['reason'], 'helper_error')
        self.assertEqual(events[-1]['current'], 'ondemand')

    def test_expiration_causes_are_logged(self):
        for reason in ('heartbeat_timeout', 'deadline'):
            with self.subTest(reason=reason):
                self.setUp()

                def expire(handlers, uid):
                    self.clock = 61 if reason == 'heartbeat_timeout' else 7200
                    if reason == 'deadline':
                        self.lease.last = 7199
                    raise socket.timeout()

                events = self.run_helper(expire)
                self.assertEqual(events[-1]['reason'], reason)
                self.assertEqual(events[-1]['current'], 'ondemand')

    def test_failed_restoration_is_reported_without_claiming_success(self):
        def disappear(handlers, uid):
            self.alive = False
            self.lease.write = lambda value: None
            raise socket.timeout()

        events = self.run_helper(disappear, RuntimeError)
        self.assertEqual(events[-1]['event'], 'restoration_failed')
        self.assertEqual(events[-1]['reason'], 'controller_unavailable')
        self.assertFalse(any(e['event'] == 'restored' for e in events))
        self.assertEqual(self.value, 'performance')


if __name__ == '__main__':
    unittest.main()
