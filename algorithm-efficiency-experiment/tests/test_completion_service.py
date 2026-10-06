"""No-device lifecycle and budget-evidence integration checks."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficiency import completion_budget as budget
from efficiency import completion_service as service
from efficiency.common import atomic_json, digest_value
from efficiency.completion_evidence import audit_budget
from efficiency.hybrid import verify_artifacts


class CompletionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'runs').mkdir()
        (self.root / 'campaigns').mkdir()
        self.campaign = self.root / 'campaigns/fresh-npu'
        self.campaign.mkdir()
        self.governor = self.root / 'governor'
        self.governor.write_text('ondemand\n')
        self.ledger = self.root / 'runs/.completion-budget.json'
        budget.initialize(self.ledger)
        self.reservation = budget.reserve(self.ledger, 'npu', 'develop', self.campaign, purpose='diagnostic')
        self.config = dict(profile='completion', track='npu', stage='diagnostic', phase='hat',
                           campaign=str(self.campaign), parents={}, max_seconds=7200,
                           reserved_seconds=7200, budget_reservation_id=self.reservation['reservation_id'],
                           development_charged_seconds=0, confirmation_charged_seconds=0,
                           baseline_seconds=15, cooldown_seconds=15, cleanup_reserve_seconds=120)
        atomic_json(self.campaign / 'campaign.json', dict(original_governor='ondemand', max_seconds=7200,
                    reservation_id=self.reservation['reservation_id'], attempts=[]))
        for name, value in (('planned-config', self.config), ('budget-reservation', self.reservation),
                            ('registry', dict(seeds=[], hashes=[])), ('implementation-validation', dict(passed=True))):
            atomic_json(self.campaign / (name + '.json'), value)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target, value in (('ROOT', self.root), ('STATE_ROOT', self.root / 'service-state'),
                              ('BUDGET_PATH', self.ledger), ('GOVERNOR', self.governor)):
            self.stack.enter_context(patch.object(service, target, value))
        self.stack.enter_context(patch('efficiency.reliability_recovery.verify_source', return_value=True))
        self.stack.enter_context(patch('efficiency.cpu_compare_service.verify_history', return_value=dict(passed=True)))

    def fake_modules(self, passed=True):
        summary = dict(stage='diagnostic', complete=passed, accepted=False, selected=None,
                       decision='diagnostic_complete' if passed else 'incomplete')
        spec = SimpleNamespace(EVENTS='fake-events.jsonl', specification=lambda c: dict(version='fake'),
                               analyze=lambda rows, config: summary)

        def audit(directory, require_seal=True):
            audit_budget(directory, require_final=require_seal)
            return dict(passed=passed, errors=[])

        return spec, SimpleNamespace(), SimpleNamespace(audit=audit)

    def test_successful_diagnostic_seals_runtime_and_releases_shared_reservation(self):
        class FakeSupervisor:
            def __init__(self, directory, config):
                self.directory, self.config = directory, config

            def execute(self):
                atomic_json(self.directory / 'manifest.json', dict(config=self.config))
                atomic_json(self.directory / 'outcome.json', dict(status='complete', elapsed_seconds=0))
                return 0

        with patch.object(service, 'modules', return_value=self.fake_modules()), patch('efficiency.runner.Supervisor', FakeSupervisor):
            self.assertEqual(service.execute(self.campaign), 0)
        current = budget.snapshot(self.ledger)
        row = current['reservations'][0]
        self.assertEqual(row['state'], 'finalized')
        self.assertEqual(row['disposition'], 'diagnostic_complete')
        self.assertGreater(row['charged_seconds'], 0)
        self.assertLess(row['charged_seconds'], 7200)
        verify_artifacts(self.campaign)
        output = Path(service.read(self.campaign / 'final-validation.json')['output'])
        verify_artifacts(output)
        self.assertTrue(audit_budget(output, require_final=True)['passed'])
        budget.reserve(self.ledger, 'npu', 'develop', self.root / 'campaigns/qualification',
                       continuation_from=self.campaign)

    def test_failure_is_sealed_without_claiming_scientific_completion(self):
        class BrokenSupervisor:
            def __init__(self, directory, config):
                pass

            def execute(self):
                raise RuntimeError('Fake device initialization failure')

        with patch.object(service, 'modules', return_value=self.fake_modules(False)), patch('efficiency.runner.Supervisor', BrokenSupervisor):
            self.assertEqual(service.execute(self.campaign), 1)
        result = service.read(self.campaign / 'final-validation.json')
        self.assertFalse(result['passed'])
        self.assertEqual(result['disposition'], 'infrastructure_failed')
        self.assertFalse(result['summary']['complete'])
        self.assertTrue(result['restored'])
        self.assertEqual(budget.snapshot(self.ledger)['reservations'][0]['state'], 'finalized')
        output = Path(result['output'])
        self.assertTrue((output / 'report.html').is_file())
        verify_artifacts(output)

    def test_stop_before_start_does_not_release_unproven_reservation(self):
        atomic_json(service.state_dir(self.campaign) / 'stop-request.json', dict(utc='test'))
        with self.assertRaises(InterruptedError):
            service.execute(self.campaign)
        self.assertEqual(budget.remaining(budget.snapshot(self.ledger), 'npu', 'develop'), 0)

    def test_sealed_campaign_cannot_restart(self):
        atomic_json(self.campaign / 'checksums.json', {})
        with self.assertRaisesRegex(ValueError, 'immutable'):
            service.execute(self.campaign)

    def test_post_stop_seals_crash_and_unblocks_other_tracks_at_full_charge(self):
        service.stopped(self.campaign)
        current = budget.snapshot(self.ledger)
        self.assertEqual(current['reservations'][0]['state'], 'finalized')
        self.assertEqual(current['reservations'][0]['charged_seconds'], 7200)
        verify_artifacts(self.campaign)
        budget.reserve(self.ledger, 'gpu', 'develop', self.root / 'campaigns/next-gpu')

    def test_post_stop_after_seal_keeps_actual_charge_and_original_bytes(self):
        from efficiency.research_report import seal
        (self.campaign / '.lock').touch()
        final = budget.finish(self.ledger, self.reservation['reservation_id'], 51, 'infrastructure_failed')
        atomic_json(self.campaign / 'budget-finalization.json', final)
        atomic_json(self.campaign / 'final-validation.json', dict(passed=False))
        seal(self.campaign)
        before = (self.campaign / 'checksums.json').read_bytes()
        service.stopped(self.campaign)
        current = budget.snapshot(self.ledger)
        self.assertEqual(current['reservations'][0]['charged_seconds'], 51)
        self.assertEqual((self.campaign / 'checksums.json').read_bytes(), before)

    def test_service_is_bounded_and_does_not_automatically_restart(self):
        command = service.build_service_command(self.campaign)
        self.assertIn('--property=Restart=no', command)
        self.assertIn('--property=RuntimeMaxSec=9000', command)
        self.assertIn('--property=KillMode=mixed', command)
        self.assertIn('--property=TimeoutStopSec=120', command)

    def test_development_selection_opens_confirmation_without_claiming_confirmation(self):
        with patch('efficiency.hybrid.verify_artifacts'), patch.object(service, 'read', side_effect=[
                dict(passed=True), dict(stage='develop', complete=True, accepted=False, selected={'candidate': 1})]):
            self.assertEqual(service.validate_parent(self.campaign)['selected'], {'candidate': 1})
        with patch('efficiency.hybrid.verify_artifacts'), patch.object(service, 'read', side_effect=[
                dict(passed=True), dict(stage='diagnostic', complete=True, accepted=False, selected=None)]):
            with self.assertRaises(ValueError):
                service.validate_parent(self.campaign)

    def test_diagnostic_and_qualification_have_distinct_dispositions(self):
        self.assertEqual(service.classify(dict(stage='diagnostic', complete=True, selected=None), dict(passed=True), [], True), 'diagnostic_complete')
        self.assertEqual(service.classify(dict(stage='develop', complete=True, selected={'x': 1}), dict(passed=True), [], True), 'qualified')
        self.assertEqual(service.classify(dict(stage='develop', complete=True, selected=None), dict(passed=True), [], True), 'rejected')
        self.assertEqual(service.classify(dict(complete=False), dict(passed=False), [dict(event='pilot_gate', fits=False)], True), 'inconclusive')
        self.assertEqual(service.classify(dict(complete=True, accepted=True), dict(passed=True), [], False), 'inconclusive')
        self.assertEqual(service.classify(dict(complete=True), dict(passed=False), [], True), 'inconclusive')

    def test_incorrect_numeric_evidence_cannot_be_retried_as_infrastructure(self):
        for bad in ({'correct': False}, {'matches_warmup': False}, {'validation_errors': 1}, {'unexpected_output_io': True}):
            for value in (bad, dict(requests=[bad])):
                rows = [dict(event='measurement', response=dict(result=value))]
                self.assertEqual(service.classify(dict(complete=False), dict(passed=False), rows, True), 'inconclusive')

    def test_direct_internal_cli_cannot_launch_hardware_without_service(self):
        with patch.dict('os.environ', {}, clear=True), patch.object(service, 'execute') as execute:
            self.assertEqual(service.main(['_run', '--campaign', str(self.campaign)]), 1)
            execute.assert_not_called()


class ReceiptTamperTests(unittest.TestCase):
    def test_resealed_receipt_cannot_change_charge_or_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / 'ledger.json'
            campaign = root / 'campaign'
            directory = root / 'run'
            directory.mkdir()
            budget.initialize(ledger)
            initial = budget.reserve(ledger, 'npu', 'develop', campaign, purpose='diagnostic')
            atomic_json(directory / 'budget-reservation.json', initial)
            config = dict(profile='completion', track='npu', stage='diagnostic', campaign=str(campaign),
                          budget_reservation_id=initial['reservation_id'], reserved_seconds=7200,
                          development_charged_seconds=0, confirmation_charged_seconds=0)
            atomic_json(directory / 'config.json', config)
            budget.mark_started(ledger, initial['reservation_id'])
            final = budget.finish(ledger, initial['reservation_id'], 150, 'diagnostic_complete')
            atomic_json(directory / 'budget-finalization.json', final)
            atomic_json(directory / 'runtime-finalization.json', dict(started_monotonic=100,
                        finished_monotonic=250, charged_seconds=150, reserved_seconds=7200, restored=True))
            atomic_json(directory / 'summary.json', dict(complete=True, accepted=False, selected=None))
            atomic_json(directory / 'validation.json', dict(passed=True))
            self.assertTrue(audit_budget(directory, require_final=True)['passed'])
            # Recalculate all embedded digests after tampering, retaining genuine raw elapsed evidence.
            final['reservation']['charged_seconds'] = 149
            final['snapshot']['reservations'][-1]['charged_seconds'] = 149
            final['snapshot']['sha256'] = digest_value({k: v for k, v in final['snapshot'].items() if k != 'sha256'})
            final['snapshot_sha256'] = final['snapshot']['sha256']
            atomic_json(directory / 'budget-finalization.json', final)
            with self.assertRaisesRegex(ValueError, 'Raw runtime charge'):
                audit_budget(directory, require_final=True)
            final['reservation']['charged_seconds'] = 150
            final['snapshot']['reservations'][-1]['charged_seconds'] = 150
            final['reservation']['disposition'] = 'inconclusive'
            final['snapshot']['reservations'][-1]['disposition'] = 'inconclusive'
            final['snapshot']['sha256'] = digest_value({k: v for k, v in final['snapshot'].items() if k != 'sha256'})
            final['snapshot_sha256'] = final['snapshot']['sha256']
            atomic_json(directory / 'budget-finalization.json', final)
            with self.assertRaisesRegex(ValueError, 'disposition contradicts'):
                audit_budget(directory, require_final=True)


if __name__ == '__main__':
    unittest.main()
