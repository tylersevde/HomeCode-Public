"""CPU comparison lifecycle tests: all governors, processes and devices are fake."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from efficiency.common import atomic_json, digest_file, read_jsonl
from efficiency import cli
from efficiency import cpu_compare_protocol as protocol
from efficiency import cpu_compare_service as service
from efficiency import cpu_compare_spec as spec
from efficiency import cpu_compare_worker as worker


class FakeProbe:
    def __init__(self, bound=False, wrong_team=False, wrong_mask=False):
        self.bound, self.wrong_team, self.wrong_mask = bound, wrong_team, wrong_mask

    def __call__(self, size, values):
        for index in range(size):
            values[index * 4] = index
            values[index * 4 + 1] = index
            values[index * 4 + 2] = 1 << index if self.bound else 15
            values[index * 4 + 3] = index if self.bound else -1
        if self.wrong_mask:
            values[2] = 3
        return size - 1 if self.wrong_team else size


class WorkerTests(unittest.TestCase):
    def test_spawn_entrypoint_preserves_measured_environment(self):
        entrypoint = Path(__file__).resolve().parents[1] / 'experiment.py'
        for configuration in spec.CONFIGURATIONS:
            env = {key: value for key, value in spec.environment(configuration).items() if value is not None}
            env.update(worker.NUMERICAL_LIBRARY_THREADS)
            env['UNRELATED'] = 'preserved'
            with patch.dict(os.environ, env, clear=True), patch.object(cli, 'main') as main:
                runpy.run_path(str(entrypoint), run_name='__mp_main__')
                self.assertEqual(dict(os.environ), env)
                main.assert_not_called()

    def test_main_entrypoint_keeps_original_parent_thread_limits(self):
        entrypoint = Path(__file__).resolve().parents[1] / 'experiment.py'
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '99'}, clear=True), \
             patch.object(cli, 'main', return_value=0) as main:
            with self.assertRaises(SystemExit) as exit_status:
                runpy.run_path(str(entrypoint), run_name='__main__')
            self.assertEqual(exit_status.exception.code, 0)
            main.assert_called_once_with()
            self.assertTrue(all(os.environ[key] == '1' for key in (
                'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')))

    def create(self, configuration, wrong_team=False, wrong_mask=False, inherited=None):
        captured = []
        def numeric_init(value, directory, config, stack):
            captured.append(config)
            value.directory, value.owner, value.environment = Path(directory), worker.owner(), {}
        env = {key: value for key, value in spec.environment(configuration).items() if value is not None}
        env.update(worker.NUMERICAL_LIBRARY_THREADS)
        if inherited:
            env.update(inherited)
        probe = FakeProbe(configuration == 'candidate', wrong_team, wrong_mask)
        with patch.dict(os.environ, env, clear=True), patch.object(worker.Numeric, '__init__', numeric_init), \
             patch.object(worker.ct, 'CDLL', return_value=SimpleNamespace(refine_probe=probe)), \
             patch.object(worker, 'sample', return_value=dict(governor=spec.policy(configuration)['governor'])):
            value = worker.CompareNumeric('/fake', dict(configuration=configuration, numeric_device='both'), ExitStack())
        return value, captured

    def test_default_and_candidate_environments_cpu_only_with_actual_teams(self):
        for configuration in spec.CONFIGURATIONS:
            value, captured = self.create(configuration)
            self.assertEqual(captured[0]['numeric_device'], 'cpu')
            self.assertEqual(value.environment['numerical_library_threads'], worker.NUMERICAL_LIBRARY_THREADS)
            self.assertEqual([probe['team'] for probe in value.environment['probes']], [1, 4])
            masks = [[row['mask'] for row in probe['threads']] for probe in value.environment['probes']]
            self.assertEqual(masks, [[1], [1, 2, 4, 8]] if configuration == 'candidate' else [[15], [15] * 4])
            if configuration == 'baseline':
                self.assertEqual(value.environment['openmp_actual'], {})

    def test_inherited_environment_and_affinity_drift_reject_before_work(self):
        for configuration in spec.CONFIGURATIONS:
            with self.assertRaisesRegex(RuntimeError, 'OpenMP launch environment'):
                self.create(configuration, inherited={'OMP_NUM_THREADS': '99'})
            with self.assertRaisesRegex(RuntimeError, 'team differs'):
                self.create(configuration, wrong_team=True)
            with self.assertRaisesRegex(RuntimeError, 'affinity differs'):
                self.create(configuration, wrong_mask=True)

    def test_numerical_library_threads_must_remain_one_for_both_configurations(self):
        for configuration in spec.CONFIGURATIONS:
            for key in worker.NUMERICAL_LIBRARY_THREADS:
                with self.subTest(configuration=configuration, key=key):
                    with self.assertRaisesRegex(RuntimeError, 'Numerical-library thread limits'):
                        self.create(configuration, inherited={key: '4'})

    def test_batch_preserves_sixteen_native_calls_and_rejects_ids_deadline_and_drift(self):
        value, _ = self.create('baseline')
        task = dict(operation='batch', request_ids=list(range(16)), deadline=time.monotonic() + 60,
                    fixture_id='fixture', arm=dict(arm='batch_a', backend='native1', variant=0))
        native = dict(correct=True, matches_warmup=True, validation_errors=0)
        with patch.object(worker, 'sample', return_value=dict(governor='ondemand')), \
             patch.object(worker.Numeric, 'measure', return_value=native) as measure:
            response = value.perform(task)
            self.assertEqual(measure.call_count, 16)
            self.assertEqual([row['request_id'] for row in response['requests']], list(range(16)))
            self.assertTrue(response['correct'])
            with self.assertRaisesRegex(ValueError, 'sixteen'):
                value.perform(dict(task, request_ids=[0] * 16))
            with self.assertRaisesRegex(TimeoutError, 'deadline'):
                value.perform(dict(task, deadline=0))
        with patch.object(worker, 'sample', side_effect=[dict(governor='ondemand'), dict(governor='performance')]), \
             patch.object(worker.Numeric, 'measure', return_value=native):
            with self.assertRaisesRegex(RuntimeError, 'Governor drift'):
                value.perform(task)


class ProtocolTests(unittest.TestCase):
    def test_pilot_projects_remaining_work_with_margin_and_cleanup_already_reserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            budget = SimpleNamespace(work_deadline=4000.)
            with patch.object(protocol.time, 'monotonic', return_value=10.):
                protocol.pilot_gate(directory, budget, 0.)
            row = read_jsonl(directory / spec.EVENTS)[0][-1]
            self.assertEqual(row['required_seconds'], 10 * 256 * 1.2)
            self.assertTrue(row['fits'])
            with patch.object(protocol.time, 'monotonic', return_value=20.):
                with self.assertRaisesRegex(TimeoutError, 'cannot fit'):
                    protocol.pilot_gate(directory, budget, 0.)
            self.assertFalse(read_jsonl(directory / spec.EVENTS)[0][-1]['fits'])

    def test_source_preparation_preserves_history_and_freezes_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, confirmation, output = [root / name for name in ('history', 'confirmation', 'new')]
            for path in (source, confirmation, output):
                path.mkdir()
            for path in (source, confirmation):
                atomic_json(path / 'checksums.json', {'historic': path.name})
            (confirmation / 'native-build').mkdir()
            (confirmation / 'native-build/libattention.so').write_bytes(b'unchanged-native')
            frozen = {'native/attention/attention.cpp': 'same', 'efficiency/attention_native.py': 'same-wrapper',
                      'efficiency/old.py': 'historical'}
            atomic_json(confirmation / 'manifest.json', dict(source_sha256=frozen))
            provenance = dict(source_campaign=dict(path=str(source), checksums_sha256=digest_file(source / 'checksums.json')),
                              cpu_confirmation=dict(path=str(confirmation), checksums_sha256=digest_file(confirmation / 'checksums.json')))
            atomic_json(output / 'provenance.json', provenance)
            atomic_json(output / 'config.json', {'new': 'comparison'})
            atomic_json(output / 'protocol.json', spec.specification())
            inventory = dict(source_sha256=dict(frozen, **{'efficiency/old.py': 'updated', 'efficiency/cpu_compare_spec.py': 'new'}))
            before = {str(path): path.read_bytes() for folder in (source, confirmation) for path in folder.rglob('*') if path.is_file()}
            with patch.object(protocol, 'verify_artifacts') as verify:
                protocol.prepare_source(output, {}, inventory)
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(service.read(output / 'freeze.json')['source_sha256'], inventory['source_sha256'])
            self.assertEqual((output / 'native-build/libattention.so').read_bytes(), b'unchanged-native')
            self.assertTrue(all(Path(path).read_bytes() == content for path, content in before.items()))
            inventory['source_sha256']['native/attention/attention.cpp'] = 'changed'
            with patch.object(protocol, 'verify_artifacts'):
                with self.assertRaisesRegex(ValueError, 'Numerical implementation changed'):
                    protocol.prepare_source(output, {}, inventory)

    def test_process_release_occurs_even_on_numerical_failure(self):
        fake = SimpleNamespace(ready=dict(owner={'pid': 12}), close=Mock(), release=dict(alive=False, forced=False))
        with tempfile.TemporaryDirectory() as tmp, patch.object(protocol, 'Worker', return_value=fake) as constructor:
            with self.assertRaisesRegex(RuntimeError, 'wrong result'):
                with protocol.worker(Path(tmp), {}, SimpleNamespace(check=lambda: None), 'unit', 'baseline'):
                    raise RuntimeError('wrong result')
            fake.close.assert_called_once()
            self.assertEqual(constructor.call_args.args[0:2], ('process', 'numeric'))
            self.assertEqual(constructor.call_args.args[3]['numeric_device'], 'cpu')
            self.assertIs(constructor.call_args.kwargs['factory'], worker.CompareNumeric)
            self.assertEqual(read_jsonl(Path(tmp) / spec.EVENTS)[0][-1]['event'], 'worker_release')


class SimulatedService:
    """Actual orchestration/protocol with fake process ownership and governor IO."""
    def __init__(self, root):
        self.root, self.campaign = root, root / 'campaigns/unit'
        self.campaign.mkdir(parents=True)
        (root / 'runs').mkdir()
        self.governor = root / 'governor'
        self.governor.write_text('ondemand')
        self.failure, self.interrupt, self.restore_error = None, False, False
        self.release_count, self.transitions, self.created = 0, [], []
        self.helper = SimpleNamespace(poll=lambda: 0, wait=Mock(return_value=0), terminate=Mock())
        self.helper_command = None
        self.summary = dict(complete=True, accepted=False, decision='no_qualified_candidate', metrics=[], defaults_changed=False)
        self.audit_passed = True
        atomic_json(self.campaign / 'campaign.json', dict(version=spec.VERSION, campaign_id='fresh',
                    source_campaign='/fake/history', original_governor='ondemand', max_seconds=14400, attempts=[]))
        for name, value in (('registry.json', {}), ('provenance.json', {}), ('implementation-validation.json', {})):
            atomic_json(self.campaign / name, value)

    def request(self, path, desired=None, close=False):
        if close:
            if self.restore_error:
                raise RuntimeError('simulated restoration failure')
            self.governor.write_text('ondemand')
        elif desired:
            self.transitions.append(desired)
            self.governor.write_text(desired)
        return dict(original='ondemand', current=self.governor.read_text(), restored=close)

    def popen(self, command, **kwargs):
        self.helper_command = command
        Path(command[command.index('--socket') + 1]).touch()
        return self.helper

    def make_worker(self, architecture, kind, directory, config, check, **kwargs):
        assert architecture == 'process' and kind == 'numeric' and config['numeric_device'] == 'cpu'
        assert kwargs['factory'] is worker.CompareNumeric
        self.created.append(config['configuration'])
        ready = dict(owner=dict(pid=100 + len(self.created)), environment=dict(numeric_device='cpu'))
        def call(operation, **payload):
            check()
            if operation == self.failure:
                raise RuntimeError('simulated worker failure')
            if operation == 'load':
                return dict(result=dict(fixture_id=payload['fixture_id']))
            value = dict(fixture_id=payload['fixture_id'], **payload['arm'], correct=True,
                         matches_warmup=True, validation_errors=0, unexpected_output_io=False)
            if operation == 'batch':
                value = dict(correct=True, requests=[dict(value, request_id=i) for i in payload['request_ids']])
            return dict(result=value)
        def close():
            self.release_count += 1
        return SimpleNamespace(ready=ready, call=call, close=close, release=dict(alive=False, forced=False))

    def supervisor(self, directory, config):
        def execute():
            atomic_json(directory / 'manifest.json', dict(config=config))
            atomic_json(directory / 'status.json', dict(monotonic=time.monotonic(), elapsed_seconds=0, phase='cpu'))
            if self.interrupt:
                self.governor.write_text('performance')
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            protocol.run(directory, config)
            return 0
        return SimpleNamespace(execute=execute)

    def run(self):
        cell = dict(cell_id='tiny', n=2, d=2, b=1, mode='stream', cpu='native1')
        with ExitStack() as stack:
            patches = [patch.object(service, 'ROOT', self.root), patch.object(service, 'STATE_ROOT', self.root / 'state'),
                       patch.object(service, 'GOVERNOR', self.governor),
                       patch.object(service, 'source_provenance', return_value={}),
                       patch.object(service, 'verify_history', return_value={'passed': True}),
                       patch('efficiency.reliability_recovery.verify_source', return_value=True),
                       patch('efficiency.cpu_policy.runtime_identity', return_value={'fake': True}),
                       patch.object(service.subprocess, 'Popen', side_effect=self.popen),
                       patch.object(service, 'request', side_effect=self.request),
                       patch.object(protocol, 'request', side_effect=self.request),
                       patch.object(protocol, 'Worker', side_effect=self.make_worker),
                       patch.object(protocol, 'wait_cool'), patch.object(protocol, 'register_fixture'),
                       patch.object(spec, 'BLOCKS', 2), patch.object(protocol, 'BLOCKS', 2), patch.object(spec, 'CELLS', [cell]),
                       patch('efficiency.runner.Supervisor', side_effect=self.supervisor),
                       patch('efficiency.cpu_compare_report.build_report', return_value=self.summary),
                       patch('efficiency.cpu_compare_audit.audit', return_value=dict(passed=self.audit_passed)),
                       patch('efficiency.cpu_policy.export_preset')]
            mocks = [stack.enter_context(p) for p in patches]
            code = service.execute(self.campaign)
        self.export = mocks[-1]
        self.output = self.root / 'runs/unit-compare'
        return code


class ServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.hardware = SimulatedService(Path(directory.name))

    def test_end_to_end_protocol_restores_and_seals_without_qualifying_preset(self):
        self.assertEqual(self.hardware.run(), 0)
        self.assertEqual(self.hardware.transitions, ['ondemand', 'performance', 'ondemand', 'performance', 'performance', 'ondemand'])
        self.assertEqual(self.hardware.release_count, 6)
        self.assertEqual(self.hardware.governor.read_text(), 'ondemand')
        events, damaged = read_jsonl(self.hardware.output / spec.EVENTS)
        self.assertFalse(damaged)
        self.assertEqual(len([e for e in events if e['event'] == 'measurement']), 3 * 2 * 3 * 4)
        self.assertEqual(events[-1]['event'], 'measurement_complete')
        self.assertTrue(service.read(self.hardware.output / 'governor-restoration.json')['restored'])
        self.assertTrue((self.hardware.campaign / 'checksums.json').exists())
        self.hardware.export.assert_not_called()

    def test_worker_failure_still_closes_owner_and_restores_governor(self):
        self.hardware.failure = 'measure'
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='incomplete')
        self.hardware.audit_passed = False
        self.assertNotEqual(self.hardware.run(), 0)
        self.assertEqual(self.hardware.release_count, 1)
        self.assertTrue(service.read(self.hardware.output / 'governor-restoration.json')['restored'])
        self.hardware.export.assert_not_called()
        self.assertFalse(service.read(self.hardware.campaign / 'final-validation.json')['passed'])

    def test_pilot_rejection_never_starts_main_measurements_or_exports(self):
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='incomplete')
        self.hardware.audit_passed = False
        with patch.object(protocol, 'pilot_gate', side_effect=TimeoutError('pilot does not fit')):
            self.assertNotEqual(self.hardware.run(), 0)
        events = read_jsonl(self.hardware.output / spec.EVENTS)[0]
        self.assertEqual({event['section'] for event in events if event['event'] == 'measurement'}, {'pilot'})
        self.assertFalse(any(event['event'] == 'measurement_complete' for event in events))
        self.assertEqual(self.hardware.release_count, 2)
        self.assertTrue(service.read(self.hardware.output / 'governor-restoration.json')['restored'])
        self.hardware.export.assert_not_called()

    def test_stop_file_halts_protocol_before_fixtures_and_restores(self):
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='incomplete')
        self.hardware.audit_passed = False
        original = self.hardware.supervisor
        def stopping_supervisor(directory, config):
            (directory / 'STOP').touch()
            return original(directory, config)
        self.hardware.supervisor = stopping_supervisor
        self.assertNotEqual(self.hardware.run(), 0)
        self.assertEqual(self.hardware.created, [])
        self.assertTrue(service.read(self.hardware.output / 'governor-restoration.json')['restored'])
        self.hardware.export.assert_not_called()

    def test_signal_during_hardware_phase_restores_before_returning_original_handlers(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        self.hardware.interrupt = True
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='incomplete')
        self.hardware.audit_passed = False
        self.assertNotEqual(self.hardware.run(), 0)
        self.assertTrue(service.read(self.hardware.output / 'governor-restoration.json')['restored'])
        self.assertEqual({sig: signal.getsignal(sig) for sig in previous}, previous)
        self.hardware.export.assert_not_called()

    def test_qualifying_result_exports_only_after_audit_and_restoration(self):
        self.hardware.summary = dict(self.hardware.summary, accepted=True, decision='cpu_improved')
        self.assertEqual(self.hardware.run(), 0)
        self.hardware.export.assert_called_once_with(self.hardware.output, self.hardware.campaign / 'preset.json')

    def test_failed_restoration_blocks_preset_even_with_qualifying_statistics(self):
        self.hardware.summary = dict(self.hardware.summary, accepted=True, decision='cpu_improved')
        self.hardware.restore_error = True
        # Finish with candidate governor so a failing close cannot appear restored.
        original = self.hardware.supervisor
        def ending_candidate(directory, config):
            instance = original(directory, config)
            run = instance.execute
            def execute():
                result = run()
                self.hardware.governor.write_text('performance')
                return result
            return SimpleNamespace(execute=execute)
        self.hardware.supervisor = ending_candidate
        self.assertNotEqual(self.hardware.run(), 0)
        self.assertFalse(service.read(self.hardware.output / 'governor-restoration.json')['restored'])
        self.hardware.export.assert_not_called()

    def test_unprivileged_helper_signal_failure_still_writes_restoration_receipt(self):
        self.hardware.helper.wait.side_effect = [subprocess.TimeoutExpired('fake-pkexec', 5), 0]
        self.hardware.helper.terminate.side_effect = PermissionError('root-owned helper')
        self.assertEqual(self.hardware.run(), 0)
        self.hardware.helper.terminate.assert_called_once()
        waits = self.hardware.helper.wait.call_args_list
        self.assertEqual(waits[0].kwargs['timeout'], 5)
        self.assertGreaterEqual(waits[1].kwargs['timeout'], 0)
        self.assertLessEqual(waits[1].kwargs['timeout'], 65)
        self.assertTrue(service.read(self.hardware.output / 'governor-restoration.json')['restored'])


class CommandTests(unittest.TestCase):
    def test_public_start_status_and_stop_route_to_durable_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'campaigns/unit'
            campaign.mkdir(parents=True)
            output = root / 'output'
            atomic_json(output / 'progress.json', {'operation': 'measurement'})
            with patch.object(service, 'ROOT', root), patch.object(service, 'STATE_ROOT', root / 'state'), \
                 patch.object(service, 'initialize', return_value=campaign) as initialize, \
                 patch.object(service.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                 patch('efficiency.reliability_service.service_properties', return_value={'ActiveState': 'active'}), \
                 redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(cli.main(['cpu-compare', 'start', '--campaign', str(campaign),
                                           '--source-campaign', '/history', '--validation', '/validation', '--max-seconds', '1234']), 0)
                args = initialize.call_args.args[0]
                self.assertEqual(args.max_seconds, 1234)
                command = run.call_args.args[0]
                for flag in ('--user', '--property=Restart=no', '--property=KillMode=mixed', '--property=TimeoutStopSec=120'):
                    self.assertIn(flag, command)
                for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    self.assertIn('--setenv=' + key + '=1', command)
                self.assertTrue(any(arg.startswith('--property=ExecStopPost=') for arg in command))
                self.assertFalse(set(command) & {'--scope', '--wait', '--pipe', '--pty'})
                service.update(campaign, 'measuring', output=str(output))
                stdout.seek(0); stdout.truncate()
                self.assertEqual(cli.main(['cpu-compare', 'status', '--campaign', str(campaign)]), 0)
                status = json.loads(stdout.getvalue())
                self.assertEqual(status['service']['ActiveState'], 'active')
                self.assertEqual(status['progress']['operation'], 'measurement')
                self.assertEqual(cli.main(['cpu-compare', 'stop', '--campaign', str(campaign)]), 0)
                run.assert_called_with(['systemctl', '--user', 'stop', service.unit_name(campaign)], check=True)
                self.assertTrue((service.state_dir(campaign) / 'stop-request.json').exists())

    def test_stop_hook_charges_interruption_and_preserves_sealed_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'campaigns/unit'
            campaign.mkdir(parents=True)
            (root / 'runs').mkdir()
            governor = root / 'governor'
            governor.write_text('ondemand')
            atomic_json(campaign / 'campaign.json', dict(original_governor='ondemand', attempts=[
                dict(state='running', reserved_seconds=14400, charged_seconds=0)]))
            with patch.object(service, 'ROOT', root), patch.object(service, 'STATE_ROOT', root / 'state'), \
                 patch.object(service, 'GOVERNOR', governor):
                service.stopped(campaign)
                row = service.read(campaign / 'campaign.json')['attempts'][0]
                self.assertEqual((row['state'], row['charged_seconds'], row['audit_passed']), ('interrupted', 14400, False))
                atomic_json(campaign / 'checksums.json', {})
                original = (campaign / 'campaign.json').read_bytes()
                service.update(campaign, 'complete')
                service.stopped(campaign)
                self.assertEqual((campaign / 'campaign.json').read_bytes(), original)
                self.assertEqual(service.read(service.state_dir(campaign) / 'status.json')['phase'], 'complete')


if __name__ == '__main__':
    unittest.main()
