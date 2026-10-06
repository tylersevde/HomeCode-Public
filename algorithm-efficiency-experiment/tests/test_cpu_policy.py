"""CPU preset identity and launcher lifecycle tests; never touch real governor sysfs."""
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from efficiency import cpu_policy as policy
from efficiency.common import atomic_json, digest_file


def runtime():
    return dict(version=policy.VERSION, model='Raspberry Pi 5 Model B Rev 1.0', machine='aarch64',
                kernel='test-kernel', python='3.13.5', python_executable='/usr/bin/python3.13',
                numpy='2.2.0', cpu_affinity=[0, 1, 2, 3],
                numerical_library_threads=dict(policy.NUMERICAL_LIBRARY_THREADS),
                native_artifacts={name: 'a' * 64 for name in ('libattention.so', 'libaffinity.so', 'build.json')},
                libraries={name: dict(path='/test/' + name, sha256='b' * 64)
                           for name in ('libgomp.so.1', 'libc.so.6')})


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'comparison'
        self.run.mkdir()
        (self.root / 'efficiency').mkdir()
        (self.root / 'efficiency/cpu_policy.py').write_text('frozen implementation')
        self.sources = {'efficiency/cpu_policy.py': digest_file(self.root / 'efficiency/cpu_policy.py')}
        atomic_json(self.run / 'manifest.json', dict(source_sha256=self.sources))
        atomic_json(self.run / 'summary.json', dict(complete=True, accepted=True))
        atomic_json(self.run / 'validation.json', dict(passed=True))
        atomic_json(self.run / 'cpu-runtime.json', runtime())
        self.seal()
        self.output = self.root / 'preset.json'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.audit = self.stack.enter_context(patch.object(policy, '_audit', return_value=dict(passed=True)))
        self.stack.enter_context(patch.object(policy, 'ROOT', self.root))
        self.stack.enter_context(patch.object(policy, 'runtime_identity', return_value=runtime()))
        self.stack.enter_context(patch.dict(policy.os.environ, {}, clear=True))

    def seal(self):
        atomic_json(self.run / 'checksums.json',
                    {p.name: digest_file(p) for p in self.run.iterdir() if p.name != 'checksums.json'})

    def export(self):
        return policy.export_preset(self.run, self.output)

    def test_export_requires_independent_read_only_audit_and_pins_evidence(self):
        preset = self.export()
        self.assertEqual(preset['evidence']['checksums_sha256'], digest_file(self.run / 'checksums.json'))
        self.assertEqual(preset['source_sha256'], self.sources)
        self.assertEqual(policy.validate_preset(self.output), preset)
        self.assertEqual(self.audit.call_count, 2)

    def test_rejects_unqualified_incomplete_and_failed_audit(self):
        for field in ('complete', 'accepted'):
            with self.subTest(field=field):
                atomic_json(self.run / 'summary.json', dict(complete=field != 'complete', accepted=field != 'accepted'))
                self.seal()
                with self.assertRaisesRegex(ValueError, 'qualify'):
                    self.export()
        atomic_json(self.run / 'summary.json', dict(complete=True, accepted=True))
        self.seal()
        self.audit.return_value = dict(passed=False)
        with self.assertRaisesRegex(ValueError, 'audit failed'):
            self.export()
        self.assertFalse(self.output.exists())

    def test_rejects_missing_or_unsealed_runtime_evidence(self):
        checks = json.loads((self.run / 'checksums.json').read_text())
        del checks['cpu-runtime.json']
        atomic_json(self.run / 'checksums.json', checks)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.export()

    def test_changed_seal_or_artifact_invalidates_existing_preset(self):
        self.export()
        atomic_json(self.run / 'summary.json', dict(complete=True, accepted=False))
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            policy.validate_preset(self.output)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'seal changed'):
            policy.validate_preset(self.output)

    def test_settings_cannot_be_relaxed_or_injected(self):
        baseline = self.export()
        for key, value in [('governor', 'ondemand'), ('policy_id', 4), ('supported_cells', []),
                           ('environment', dict(policy.ENVIRONMENT, OMP_NUM_THREADS='8'))]:
            with self.subTest(key=key):
                atomic_json(self.output, dict(baseline, **{key: value}))
                with self.assertRaisesRegex(ValueError, 'settings differ'):
                    policy.validate_preset(self.output)

    def test_source_and_runtime_changes_are_rejected_before_any_launch(self):
        self.export()
        with patch.object(policy, 'runtime_identity', return_value=dict(runtime(), kernel='changed')):
            with self.assertRaisesRegex(ValueError, 'native runtime differs'):
                policy.validate_preset(self.output)
        (self.root / 'efficiency/cpu_policy.py').write_text('changed implementation')
        with self.assertRaisesRegex(ValueError, 'Current source differs'):
            policy.validate_preset(self.output)

    def test_export_keeps_sealed_evidence_immutable(self):
        with self.assertRaisesRegex(ValueError, 'outside sealed'):
            policy.export_preset(self.run, self.run / 'preset.json')
        self.export()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.export()

    def test_platform_and_affinity_are_not_portable_claims(self):
        for changed in [dict(runtime(), model='Raspberry Pi 4'), dict(runtime(), cpu_affinity=[0, 1])]:
            atomic_json(self.run / 'cpu-runtime.json', changed)
            self.seal()
            with self.assertRaisesRegex(ValueError, 'Raspberry Pi 5'):
                self.export()

    def test_loader_overrides_cannot_substitute_runtime(self):
        self.export()
        for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
            with patch.dict(policy.os.environ, {key: '/unqualified'}):
                with self.assertRaisesRegex(ValueError, 'loader overrides'):
                    policy.validate_preset(self.output)

    def test_preset_identity_itself_is_bound_to_evidence(self):
        value = self.export()
        value['runtime']['kernel'] = 'different'
        atomic_json(self.output, value)
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            policy.validate_preset(self.output)


class EnvironmentTests(unittest.TestCase):
    def test_all_inherited_openmp_overrides_are_removed_without_mutating_parent(self):
        original = dict(PATH='/bin', OMP_NUM_THREADS='99', OMP_THREAD_LIMIT='1',
                        GOMP_CPU_AFFINITY='0', GOMP_STACKSIZE='99', KEEP='yes', OPENBLAS_NUM_THREADS='64')
        before = original.copy()
        result = policy.policy_environment(original)
        self.assertEqual(original, before)
        self.assertEqual(result, dict(policy.ENVIRONMENT, **policy.NUMERICAL_LIBRARY_THREADS, PATH='/bin', KEEP='yes'))

    def test_missing_numerical_thread_limits_are_filled_without_altering_openmp_routing(self):
        result = policy.policy_environment({})
        self.assertEqual({k: result[k] for k in policy.NUMERICAL_LIBRARY_THREADS},
                         policy.NUMERICAL_LIBRARY_THREADS)
        self.assertNotIn('OMP_NUM_THREADS', result)

    def test_invalid_evidence_never_starts_a_process(self):
        with patch.object(policy.os, 'geteuid', return_value=1000), \
                patch.object(policy, 'validate_preset', side_effect=ValueError('invalid evidence')), \
                patch.object(policy.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'invalid evidence'):
                policy.run('/missing.json', 130, ['/bin/true'])
            popen.assert_not_called()

    def test_limits_and_explicit_command_are_required(self):
        for seconds in (0, 120, 14401, float('inf'), float('nan'), True):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, 'max-seconds'):
                policy.run('/missing.json', seconds, ['/bin/true'])
        with self.assertRaisesRegex(ValueError, 'explicit command'):
            policy.run('/missing.json', 130, [])

    def test_cli_passes_argv_without_shell_parsing(self):
        with patch.object(policy, 'run', return_value=7) as run:
            self.assertEqual(policy.main(['run', '--preset', '/p.json', '--max-seconds', '130', '--',
                                          '/bin/echo', '$(touch /never)']), 7)
            self.assertEqual(run.call_args.args[2], ['/bin/echo', '$(touch /never)'])


class Clock:
    def __init__(self):
        self.now = 0
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()


class Process:
    def __init__(self, clock, duration=None, returncode=0):
        self.clock, self.duration, self.code = clock, duration, returncode
        self.started, self.returncode, self.pid = clock.now, None, 1234

    def poll(self):
        if self.returncode is None and self.duration is not None and self.clock.now - self.started >= self.duration:
            self.returncode = self.code
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.governor = self.root / 'fake-governor'
        self.governor.write_text('ondemand')
        self.clock = Clock()
        self.calls, self.requests, self.handlers = [], [], {}
        self.duration, self.command_code, self.command_failure = 1, 7, None
        self.helper = self.child = None
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        patches = [(policy, 'ROOT', self.root), (policy, 'GOVERNOR', self.governor),
                   (policy.os, 'geteuid', lambda: 1000),
                   (policy, 'validate_preset', lambda _: dict(evidence=dict(path='/evidence', checksums_sha256='a' * 64))),
                   (policy, '_probe', lambda *_: []), (policy.time, 'monotonic', self.clock.monotonic),
                   (policy.time, 'sleep', self.clock.sleep), (policy.subprocess, 'Popen', self.popen),
                   (policy, 'request', self.request), (policy, 'stop_process_group', self.stop_child),
                   (policy.signal, 'getsignal', lambda _: None), (policy.signal, 'signal', self.install_handler)]
        for module, name, value in patches:
            self.stack.enter_context(patch.object(module, name, value))
        self.stack.enter_context(patch.dict(policy.os.environ, dict(OMP_NUM_THREADS='42', KEEP='yes'), clear=True))
        self.stopped_child = False

    def install_handler(self, sig, handler):
        self.handlers[sig] = handler

    def popen(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if argv[0] == 'pkexec':
            self.helper = Process(self.clock)
            Path(argv[argv.index('--socket') + 1]).touch()
            return self.helper
        if self.command_failure:
            raise self.command_failure
        self.child = Process(self.clock, self.duration, self.command_code)
        return self.child

    def request(self, path, desired=None, close=False):
        self.requests.append(dict(desired=desired, close=close, now=self.clock.now))
        if close:
            self.governor.write_text('ondemand')
            self.helper.returncode = 0
            Path(path).unlink(missing_ok=True)
        elif desired:
            self.governor.write_text(desired)
        return dict(original='ondemand', current=self.governor.read_text(), restored=close)

    def stop_child(self, child, grace):
        self.stopped_child = True
        child.terminate()

    def result(self):
        return json.loads(next((self.root / 'runs').glob('cpu-policy-*/result.json')).read_text())

    def execute(self, seconds=140):
        return policy.run('/preset.json', seconds, ['/bin/example', 'literal argument'])

    def test_exit_status_environment_child_ownership_and_restoration(self):
        self.assertEqual(self.execute(), 7)
        command, options = self.calls[1]
        self.assertEqual(command, ['/bin/example', 'literal argument'])
        self.assertFalse(options['shell'])
        self.assertTrue(options['start_new_session'])
        self.assertEqual(options['env'], dict(policy.ENVIRONMENT, **policy.NUMERICAL_LIBRARY_THREADS, KEEP='yes'))
        self.assertTrue(self.stopped_child)
        self.assertTrue(self.result()['restored'])
        self.assertEqual(self.result()['returncode'], 7)
        self.assertEqual(self.governor.read_text(), 'ondemand')
        self.assertEqual(policy.os.environ['OMP_NUM_THREADS'], '42')

    def test_heartbeat_continues_during_long_command(self):
        self.duration = 12
        self.assertEqual(self.execute(), 7)
        heartbeats = [r for r in self.requests if r['desired'] is None and not r['close']]
        self.assertGreaterEqual(len(heartbeats), 3)

    def test_deadline_stops_owned_command_and_restores(self):
        self.duration = None
        self.assertEqual(self.execute(125), 124)
        self.assertTrue(self.stopped_child)
        self.assertTrue(self.result()['restored'])
        self.assertLess(self.clock.now, 125)

    def test_signal_stops_child_restores_governor_and_preserves_signal_status(self):
        self.duration = None

        def interrupt():
            if self.clock.now >= .5:
                self.handlers[signal.SIGTERM](signal.SIGTERM, None)
        self.clock.on_sleep = interrupt
        self.assertEqual(self.execute(), 143)
        self.assertTrue(self.stopped_child)
        self.assertTrue(self.result()['restored'])

    def test_command_start_failure_still_restores(self):
        self.command_failure = FileNotFoundError('missing command')
        with self.assertRaisesRegex(FileNotFoundError, 'missing command'):
            self.execute()
        self.assertTrue(self.result()['restored'])

    def test_probe_failure_does_not_apply_governor_or_spawn_helper(self):
        with patch.object(policy, '_probe', side_effect=ValueError('bad affinity')):
            with self.assertRaisesRegex(ValueError, 'bad affinity'):
                self.execute()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.requests, [])
        self.assertTrue(self.result()['restored'])

    def test_governor_lease_failure_terminates_command(self):
        normal = self.request
        self.duration = 12

        def broken(path, desired=None, close=False):
            if self.clock.now >= 5 and not close:
                raise RuntimeError('lost lease')
            return normal(path, desired, close)
        with patch.object(policy, 'request', broken):
            with self.assertRaisesRegex(RuntimeError, 'lost lease'):
                self.execute()
        self.assertTrue(self.stopped_child)
        self.assertTrue(self.result()['restored'])

    def test_failed_governor_restoration_is_reported_and_bounded(self):
        normal = self.request

        def broken(path, desired=None, close=False):
            if close:
                # Model an inaccessible helper; the launcher waits only within its reserve.
                raise RuntimeError('restoration unavailable')
            return normal(path, desired, close)
        with patch.object(policy, 'request', broken):
            with self.assertRaisesRegex(RuntimeError, 'cleanup could not be verified'):
                self.execute(125)
        self.assertFalse(self.result()['restored'])
        self.assertEqual(self.clock.now, 125)
        self.assertTrue(self.result()['cleanup_errors'])

    def test_authentication_failure_never_starts_command(self):
        normal = self.popen

        def rejected(argv, **kwargs):
            process = normal(argv, **kwargs)
            if argv[0] == 'pkexec':
                Path(argv[argv.index('--socket') + 1]).unlink()
                process.returncode = 126
            return process
        with patch.object(policy.subprocess, 'Popen', rejected):
            with self.assertRaisesRegex(ValueError, 'authentication unavailable'):
                self.execute()
        self.assertIsNone(self.child)
        self.assertTrue(self.result()['restored'])

    def test_child_signal_exit_has_normal_shell_status(self):
        self.command_code = -signal.SIGINT
        self.assertEqual(self.execute(), 130)
        self.assertTrue(self.result()['restored'])

    def test_experiment_lock_blocks_concurrent_use(self):
        (self.root / 'runs').mkdir()
        with (self.root / 'runs/.experiment.lock').open('w') as lock:
            policy.fcntl.flock(lock, policy.fcntl.LOCK_EX | policy.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.execute()
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
