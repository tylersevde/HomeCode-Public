"""Explicit CPU retries share one immutable original allowance; no hardware calls."""
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from efficiency import completion_budget as budget
from efficiency import cpu_compare_service as service
from efficiency.common import atomic_json, digest_file
from efficiency.research_report import seal
from test_completion_budget import failed_cpu, SPENT, REMAINING
from test_cpu_compare_runtime import SimulatedService


class RetryInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'campaigns').mkdir(); (self.root / 'runs').mkdir()
        self.origin = failed_cpu(self.root)
        self.initial_seal = digest_file(self.origin / 'checksums.json')
        self.governor = self.root / 'governor'; self.governor.write_text('ondemand')
        self.validation = self.root / 'validation.json'; atomic_json(self.validation, {})
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        patches = [patch.object(service, 'ROOT', self.root),
                   patch.object(service, 'STATE_ROOT', self.root / 'state'),
                   patch.object(service, 'GOVERNOR', self.governor),
                   patch.object(service, 'source_provenance', return_value={}),
                   patch('efficiency.reliability_recovery.verify_source', return_value=True),
                   patch('efficiency.reliability_campaign.historical_seals', return_value=[]),
                   patch('efficiency.research_campaign.collect_previous', return_value={}),
                   patch.object(service.subprocess, 'run', return_value=SimpleNamespace(stdout='running', returncode=0))]
        for item in patches:
            self.stack.enter_context(item)

    def args(self, name='retry', max_seconds=None, retry_from=True):
        return SimpleNamespace(campaign=self.root / 'campaigns' / name, source_campaign=Path('/scientific-parent'),
                               validation=self.validation, max_seconds=max_seconds,
                               retry_from=self.origin if retry_from else None)

    def test_fresh_retry_imports_exact_spent_seconds_and_pins_both_seals(self):
        campaign = service.initialize(self.args())
        ledger = service.read(campaign / 'campaign.json')
        self.assertEqual(ledger['max_seconds'], REMAINING)
        receipt = service.read(campaign / 'budget-reservation.json')
        self.assertEqual(receipt['snapshot']['cpu_origin']['charged_seconds'], SPENT)
        self.assertEqual(receipt['reservation']['parent']['checksums_sha256'], self.initial_seal)
        self.assertEqual(receipt['reservation']['relation'], 'retry')
        self.assertEqual(digest_file(self.origin / 'checksums.json'), self.initial_seal)

    def test_omitted_retry_parent_cannot_reset_budget(self):
        with self.assertRaisesRegex(ValueError, 'allowance cannot be reset'):
            service.initialize(self.args(retry_from=False))
        self.assertFalse(self.args().campaign.exists())

    def test_requested_four_hours_exceeds_original_remainder(self):
        with self.assertRaisesRegex(ValueError, 'remaining'):
            service.initialize(self.args(max_seconds=14400))

    def test_parallel_second_retry_cannot_reserve_remaining_again(self):
        service.initialize(self.args(max_seconds=1000))
        with self.assertRaisesRegex(ValueError, 'already active'):
            service.initialize(self.args(name='second', max_seconds=1000))

    def test_no_scientific_or_pilot_feasibility_retries(self):
        ledger = service.read(self.origin / 'campaign.json')
        output = Path(ledger['attempts'][0]['output'])
        atomic_json(output / 'summary.json', dict(complete=False, accepted=False, decision='deferred_by_pilot'))
        seal(output); ledger['attempts'][0]['checksums_sha256'] = digest_file(output / 'checksums.json')
        atomic_json(self.origin / 'campaign.json', ledger); seal(self.origin)
        with self.assertRaisesRegex(ValueError, 'Scientific or feasibility'):
            service.initialize(self.args())

    def test_cli_defaults_to_remainder_and_requires_explicit_retry_source(self):
        with patch.object(service, 'start', return_value=0) as start:
            result = service.main(['start', '--campaign', str(self.root / 'campaigns/retry'),
                                   '--source-campaign', '/scientific-parent', '--validation', str(self.validation),
                                   '--retry-from', str(self.origin)])
            self.assertEqual(result, 0)
            self.assertIsNone(start.call_args.args[0].max_seconds)
            self.assertEqual(start.call_args.args[0].retry_from, self.origin)


class RetryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.hardware = SimulatedService(self.root)
        self.origin = failed_cpu(self.root)
        self.path = self.root / 'runs/.completion-budget.json'
        budget.initialize(self.path, self.origin)
        self.receipt = budget.reserve(self.path, 'cpu', 'compare', self.hardware.campaign,
                                      retry_from=self.origin)
        ledger = service.read(self.hardware.campaign / 'campaign.json')
        ledger.update(max_seconds=REMAINING, budget_reservation=self.receipt,
                      retry_from=budget.sealed_reference(self.origin))
        atomic_json(self.hardware.campaign / 'campaign.json', ledger)
        atomic_json(self.hardware.campaign / 'budget-reservation.json', self.receipt)
        atomic_json(self.hardware.campaign / 'retry-lineage.json', budget.validate_retry_source(self.origin))

    def test_end_to_end_retry_finalizes_shared_budget_after_sealing(self):
        self.assertEqual(self.hardware.run(), 0)
        ledger = budget.snapshot(self.path)
        row = ledger['reservations'][0]
        self.assertEqual(row['state'], 'finalized')
        self.assertEqual(row['disposition'], 'rejected')
        run_config = service.read(self.hardware.output / 'config.json')
        self.assertIn(self.receipt['reservation_id'], run_config['fixture_namespace'])
        self.assertEqual(run_config['reserved_seconds'], REMAINING)
        runtime = service.read(self.hardware.output / 'runtime-finalization.json')
        self.assertEqual(row['charged_seconds'], runtime['charged_seconds'])
        self.assertEqual(service.read(self.hardware.output / 'budget-finalization.json'),
                         service.read(self.hardware.campaign / 'budget-finalization.json'))
        with self.assertRaisesRegex(ValueError, 'Scientific stage is closed'):
            budget.reserve(self.path, 'cpu', 'compare', self.root / 'another', retry_from=self.hardware.campaign)

    def test_worker_infrastructure_failure_allows_only_next_sealed_predecessor(self):
        self.hardware.failure = 'measure'
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='incomplete')
        self.hardware.audit_passed = False
        self.assertNotEqual(self.hardware.run(), 0)
        remaining = budget.remaining(budget.snapshot(self.path), 'cpu', 'compare')
        with self.assertRaisesRegex(ValueError, 'latest sealed'):
            budget.reserve(self.path, 'cpu', 'compare', self.root / 'wrong', retry_from=self.origin)
        retry = budget.reserve(self.path, 'cpu', 'compare', self.root / 'another', retry_from=self.hardware.campaign)
        self.assertEqual(retry['reservation']['reserved_seconds'], remaining)
        self.assertLess(remaining, REMAINING)

    def test_pilot_failure_closes_retry_without_scientific_relabeling(self):
        from efficiency import cpu_compare_protocol
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='deferred_by_pilot')
        self.hardware.audit_passed = False
        with patch.object(cpu_compare_protocol, 'pilot_gate', side_effect=TimeoutError('pilot does not fit')):
            self.assertNotEqual(self.hardware.run(), 0)
        # A mocked gate writes no event, so its explicit summary also closes retry.
        self.assertEqual(budget.snapshot(self.path)['reservations'][0]['disposition'], 'inconclusive')

    def stop_service(self):
        with patch.object(service, 'ROOT', self.root), \
                patch.object(service, 'STATE_ROOT', self.root / 'state'), \
                patch.object(service, 'GOVERNOR', self.hardware.governor):
            service.stopped(self.hardware.campaign)

    def test_service_crash_before_controller_starts_seals_full_reservation_charge(self):
        self.stop_service()
        value = budget.snapshot(self.path)
        self.assertEqual(value['reservations'][0]['state'], 'finalized')
        self.assertEqual(value['reservations'][0]['charged_seconds'], REMAINING)
        self.assertEqual(budget.remaining(value, 'cpu', 'compare'), 0)
        self.assertTrue((self.hardware.campaign / 'checksums.json').exists())
        self.assertFalse(service.read(self.hardware.campaign / 'final-validation.json')['passed'])
        self.assertEqual(budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')['reservation']['reserved_seconds'], 7200)

    def test_crash_after_campaign_seal_attaches_existing_actual_charge(self):
        with patch.object(budget, 'attach_evidence', side_effect=RuntimeError('crash after seal')):
            with self.assertRaisesRegex(RuntimeError, 'crash after seal'):
                self.hardware.run()
        seal_before = digest_file(self.hardware.campaign / 'checksums.json')
        current = budget.snapshot(self.path)['reservations'][0]
        self.assertEqual(current['state'], 'finalizing')
        actual_charge = current['charged_seconds']
        self.stop_service()
        self.assertEqual(budget.snapshot(self.path)['reservations'][0]['charged_seconds'], actual_charge)
        self.assertEqual(digest_file(self.hardware.campaign / 'checksums.json'), seal_before)

    def test_single_request_validation_failure_closes_scientific_retry(self):
        original = self.hardware.make_worker
        self.hardware.summary = dict(self.hardware.summary, complete=False, decision='incomplete')
        self.hardware.audit_passed = False

        def faulty_worker(*args, **kwargs):
            worker = original(*args, **kwargs)
            call = worker.call

            def invalid(operation, **payload):
                response = call(operation, **payload)
                if operation == 'measure':
                    response['result']['validation_errors'] = 1
                return response
            worker.call = invalid
            return worker

        self.hardware.make_worker = faulty_worker
        self.assertNotEqual(self.hardware.run(), 0)
        self.assertEqual(budget.snapshot(self.path)['reservations'][0]['disposition'], 'inconclusive')
        with self.assertRaisesRegex(ValueError, 'Scientific stage is closed'):
            budget.reserve(self.path, 'cpu', 'compare', self.root / 'another', retry_from=self.hardware.campaign)

    def test_crash_after_invalid_batch_output_seals_closed_disposition(self):
        output = self.root / 'runs' / (self.hardware.campaign.name + '-compare')
        output.mkdir()
        event = dict(event='measurement', response=dict(result=dict(correct=True,
                     requests=[dict(correct=True, unexpected_output_io=True)])))
        (output / 'cpu-compare.jsonl').write_text(json.dumps(event) + '\n')
        self.stop_service()
        self.assertEqual(budget.snapshot(self.path)['reservations'][0]['disposition'], 'inconclusive')
        with self.assertRaisesRegex(ValueError, 'not an infrastructure failure'):
            budget.validate_retry_source(self.hardware.campaign)


class ScientificOutcomeTests(unittest.TestCase):
    def test_single_and_batch_failures_close_retry_but_normal_warmup_does_not(self):
        for key, value in [('correct', False), ('matches_warmup', False),
                           ('validation_errors', 1), ('unexpected_output_io', True)]:
            for batch in (False, True):
                with self.subTest(field=key, batch=batch):
                    row = {key: value}
                    result = dict(correct=True, requests=[row]) if batch else row
                    self.assertTrue(service.scientific_outcome([
                        dict(event='measurement', response=dict(result=result))]))
        self.assertFalse(service.scientific_outcome([
            dict(event='warmup', response=dict(result=dict(correct=True, matches_warmup=False,
                                                          validation_errors=0, unexpected_output_io=False)))]))
        self.assertTrue(service.scientific_outcome([dict(event='measurement_complete')]))


if __name__ == '__main__':
    unittest.main()
