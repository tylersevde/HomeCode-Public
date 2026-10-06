"""Cumulative allowance and sealed lineage tests; no device or service operations."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from efficiency import completion_budget as budget
from efficiency.common import atomic_json, digest_file, digest_value
from efficiency.research_report import seal


SPENT = 132.87611606300925
REMAINING = 14400 - SPENT


def failed_cpu(root, name='original', charge=SPENT, reservation=14400):
    campaign = root / name
    output = root / (name + '-run')
    campaign.mkdir(); output.mkdir()
    atomic_json(output / 'summary.json', dict(complete=False, accepted=False, decision='incomplete'))
    atomic_json(output / 'governor-restoration.json', dict(restored=True, original='ondemand', current='ondemand'))
    atomic_json(output / 'runtime-finalization.json', dict(charged_seconds=charge, reserved_seconds=reservation,
                restored=True, started_monotonic=100, finished_monotonic=100 + charge))
    seal(output)
    atomic_json(campaign / 'campaign.json', dict(version='cpu-compare-v1', source_campaign='/scientific-parent',
                max_seconds=reservation, original_governor='ondemand', attempts=[dict(stage='cpu-compare',
                scientific_complete=False, audit_passed=False, restored=True, charged_seconds=charge,
                reserved_seconds=reservation, output=str(output), checksums_sha256=digest_file(output / 'checksums.json'))]))
    seal(campaign)
    return campaign


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = failed_cpu(self.root)
        self.path = self.root / 'state/budget.json'
        budget.initialize(self.path, self.origin)

    def close(self, receipt, seconds, disposition):
        final = budget.finish(self.path, receipt['reservation_id'], seconds, disposition)
        campaign = Path(receipt['reservation']['campaign'])
        campaign.mkdir(exist_ok=True)
        atomic_json(campaign / 'budget-finalization.json', final)
        seal(campaign)
        return budget.attach_evidence(self.path, receipt['reservation_id'], campaign)

    def test_imports_exact_charge_and_defaults_to_exact_remainder(self):
        value = budget.snapshot(self.path)
        self.assertEqual(value['cpu_origin']['charged_seconds'], SPENT)
        receipt = budget.reserve(self.path, 'cpu', 'compare', self.root / 'retry', retry_from=self.origin)
        self.assertEqual(receipt['reservation']['reserved_seconds'], 14267.12388393699)
        self.assertEqual(budget.remaining(budget.snapshot(self.path), 'cpu', 'compare'), 0)

    def test_original_allowance_cannot_reset_by_missing_parent_or_larger_cap(self):
        with self.assertRaisesRegex(ValueError, 'original sealed'):
            budget.reserve(self.path, 'cpu', 'compare', self.root / 'bad')
        with self.assertRaisesRegex(ValueError, 'remaining'):
            budget.reserve(self.path, 'cpu', 'compare', self.root / 'bad', requested=14400, retry_from=self.origin)
        replacement = failed_cpu(self.root, 'different')
        with self.assertRaisesRegex(ValueError, 'replaced or reset'):
            budget.initialize(self.path, replacement)

    def test_one_active_reservation_blocks_parallel_and_other_tracks(self):
        budget.reserve(self.path, 'cpu', 'compare', self.root / 'first', requested=1000, retry_from=self.origin)
        with self.assertRaisesRegex(ValueError, 'already active'):
            budget.reserve(self.path, 'cpu', 'compare', self.root / 'second', requested=1000, retry_from=self.origin)
        with self.assertRaisesRegex(ValueError, 'already active'):
            budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')

    def test_crash_before_or_after_provisional_finish_still_spends_full_reservation(self):
        receipt = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu', requested=1000)
        budget.mark_started(self.path, receipt['reservation_id'])
        self.assertEqual(budget.remaining(budget.snapshot(self.path), 'gpu', 'develop'), 6200)
        budget.finish(self.path, receipt['reservation_id'], 100, 'infrastructure_failed')
        self.assertEqual(budget.remaining(budget.snapshot(self.path), 'gpu', 'develop'), 6200)
        with self.assertRaisesRegex(ValueError, 'already active'):
            budget.reserve(self.path, 'gpu', 'develop', self.root / 'retry')

    def test_complete_seal_releases_unused_reservation_but_requires_explicit_retry(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu', requested=1000)
        self.close(first, 200, 'infrastructure_failed')
        self.assertEqual(budget.remaining(budget.snapshot(self.path), 'gpu', 'develop'), 7000)
        with self.assertRaisesRegex(ValueError, 'Explicit predecessor'):
            budget.reserve(self.path, 'gpu', 'develop', self.root / 'retry')
        second = budget.reserve(self.path, 'gpu', 'develop', self.root / 'retry', retry_from=self.root / 'gpu')
        self.assertEqual(second['reservation']['reserved_seconds'], 7000)

    def test_reused_campaign_and_double_finalization_are_rejected(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        self.close(first, 100, 'infrastructure_failed')
        with self.assertRaisesRegex(ValueError, 'already reserved'):
            budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu', retry_from=self.root / 'gpu')
        with self.assertRaisesRegex(ValueError, 'already finished'):
            budget.finish(self.path, first['reservation_id'], 0, 'infrastructure_failed')

    def test_scientific_negative_and_inconclusive_close_stage(self):
        for track, outcome in [('gpu', 'rejected'), ('npu', 'inconclusive')]:
            first = budget.reserve(self.path, track, 'develop', self.root / track)
            self.close(first, 100, outcome)
            with self.assertRaisesRegex(ValueError, 'Scientific stage is closed'):
                budget.reserve(self.path, track, 'develop', self.root / (track + '-retry'), retry_from=self.root / track)

    def test_confirm_budget_cannot_borrow_unused_development(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        self.close(first, 100, 'qualified')
        with self.assertRaisesRegex(ValueError, 'remaining'):
            budget.reserve(self.path, 'gpu', 'confirm', self.root / 'confirm', requested=7201)
        second = budget.reserve(self.path, 'gpu', 'confirm', self.root / 'confirm')
        self.assertEqual(second['reservation']['reserved_seconds'], 7200)

    def test_confirm_requires_qualified_development(self):
        with self.assertRaisesRegex(ValueError, 'qualified development'):
            budget.reserve(self.path, 'npu', 'confirm', self.root / 'confirm')
        self.assertEqual(budget.snapshot(self.path)['reservations'], [])

    def test_npu_diagnostic_and_qualification_share_one_development_allowance(self):
        diagnostic = budget.reserve(self.path, 'npu', 'develop', self.root / 'diagnostic', requested=1800,
                                    purpose='diagnostic')
        self.close(diagnostic, 1200, 'diagnostic_complete')
        with self.assertRaisesRegex(ValueError, 'qualified development'):
            budget.reserve(self.path, 'npu', 'confirm', self.root / 'confirm')
        qualification = budget.reserve(self.path, 'npu', 'develop', self.root / 'qualify',
                                       continuation_from=self.root / 'diagnostic')
        self.assertEqual(qualification['reservation']['reserved_seconds'], 6000)
        self.close(qualification, 1500, 'qualified')
        confirmation = budget.reserve(self.path, 'npu', 'confirm', self.root / 'confirm')
        self.assertEqual(confirmation['reservation']['reserved_seconds'], 7200)

    def test_diagnostic_cannot_qualify_original_task(self):
        diagnostic = budget.reserve(self.path, 'npu', 'develop', self.root / 'diagnostic', purpose='diagnostic')
        with self.assertRaisesRegex(ValueError, 'Diagnostic work cannot qualify'):
            budget.finish(self.path, diagnostic['reservation_id'], 100, 'qualified')

    def test_origin_and_resealed_predecessor_tampering_are_detected(self):
        origin_ledger = budget.read(self.origin / 'campaign.json')
        origin_ledger['attempts'][0]['charged_seconds'] = 1
        atomic_json(self.origin / 'campaign.json', origin_ledger)
        seal(self.origin)
        with self.assertRaises(ValueError):
            budget.snapshot(self.path)

    def test_snapshot_arithmetic_and_chain_are_independently_reconstructed(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu', requested=1000)
        self.close(first, 100, 'infrastructure_failed')
        second = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu-retry', retry_from=self.root / 'gpu')
        value = deepcopy(second['snapshot'])
        value['reservations'][-1]['remaining_before_seconds'] = 7200
        value['sha256'] = digest_value({k:v for k,v in value.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'Cumulative reservation'):
            budget.audit_snapshot(value)

    def test_sealed_actual_charge_cannot_be_changed_in_live_ledger(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        self.close(first, 200, 'qualified')
        value = budget.snapshot(self.path)
        value['reservations'][0]['charged_seconds'] = 100
        value['sha256'] = digest_value({k:v for k,v in value.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'Sealed charge'):
            budget.audit_snapshot(value)

    def test_wrong_campaign_or_altered_final_receipt_cannot_release_reservation(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        final = budget.finish(self.path, first['reservation_id'], 200, 'qualified')
        campaign = self.root / 'gpu'; campaign.mkdir()
        final['reservation']['charged_seconds'] = 1
        atomic_json(campaign / 'budget-finalization.json', final); seal(campaign)
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            budget.attach_evidence(self.path, first['reservation_id'], campaign)

    def test_invalid_or_overrun_charge_is_refused(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu', requested=1000)
        for charge in (-1, 1001, float('nan'), True):
            with self.subTest(charge=charge), self.assertRaisesRegex(ValueError, 'Invalid final charge'):
                budget.finish(self.path, first['reservation_id'], charge, 'infrastructure_failed')

    def test_abandoned_reservation_is_fully_charged_but_other_tracks_can_continue(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        budget.mark_started(self.path, first['reservation_id'])
        final = budget.abandon(self.path, first['reservation_id'])
        campaign = self.root / 'gpu'; campaign.mkdir()
        atomic_json(campaign / 'budget-finalization.json', final); seal(campaign)
        budget.attach_evidence(self.path, first['reservation_id'], campaign)
        self.assertEqual(budget.remaining(budget.snapshot(self.path), 'gpu', 'develop'), 0)
        with self.assertRaisesRegex(ValueError, 'remaining'):
            budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu-retry', retry_from=campaign)
        self.assertEqual(budget.reserve(self.path, 'npu', 'develop', self.root / 'npu')['reservation']['reserved_seconds'], 7200)

    def test_missing_mutable_ledger_cannot_reinitialize_original_allowance(self):
        budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, 'refusing an allowance reset'):
            budget.initialize(self.path, self.origin)

    def test_rolling_mutable_ledger_back_to_valid_older_snapshot_is_rejected(self):
        old = budget.snapshot(self.path)
        budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        atomic_json(self.path, old)
        with self.assertRaisesRegex(ValueError, 'committed journal'):
            budget.snapshot(self.path)

    def test_offline_resealed_snapshot_cannot_move_spending_to_another_track(self):
        first = budget.reserve(self.path, 'gpu', 'develop', self.root / 'gpu')
        self.close(first, 200, 'qualified')
        value = budget.snapshot(self.path)
        value['reservations'][0]['track'] = 'cache'
        value['sha256'] = digest_value({k:v for k,v in value.items() if k != 'sha256'})
        with self.assertRaisesRegex(ValueError, 'Sealed charge, reservation'):
            budget.audit_snapshot(value)


if __name__ == '__main__':
    unittest.main()
