import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json, digest_file
from efficiency import reliability_service as service
from efficiency.reliability_recovery import capture_lineage, verify_lineage, verify_source
from efficiency.reliability_postvalidate import rewrite_events


class LineageTests(unittest.TestCase):
    def test_interrupted_budget_assessment_preserves_every_original_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, run, new = (root / n for n in ('old', 'run', 'new'))
            for p in (old, run, new):
                p.mkdir()
            atomic_json(run / 'config.json', {'old': True})
            (run / 'reliability.jsonl').write_text('{"event":"failure"}\n')
            original = dict(attempts=[dict(stage='cpu-develop', state='running',
                output=str(run), reserved_seconds=14400, charged_seconds=14400)])
            atomic_json(old / 'campaign.json', original)
            before = (old / 'campaign.json').read_bytes()
            checksum = capture_lineage(new, old)
            atomic_json(new / 'campaign.json', dict(recovery_from=str(old), recovery_lineage_sha256=checksum))
            self.assertTrue(verify_lineage(new)['passed'])
            self.assertEqual((old / 'campaign.json').read_bytes(), before)
            assessment = json.loads((new / 'recovery-lineage.json').read_text())['recovered_ledger_assessment']
            self.assertEqual(assessment['attempts'][0]['state'], 'interrupted')
            self.assertEqual(assessment['attempts'][0]['charged_seconds'], 14400)
            (run / 'added-evidence.txt').write_text('new')
            with self.assertRaisesRegex(ValueError, 'Historical'):
                verify_lineage(new)

    def test_manifest_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            atomic_json(p / 'recovery-lineage.json', dict(files={}))
            atomic_json(p / 'campaign.json', dict(recovery_lineage_sha256='0' * 64))
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                verify_lineage(p)

    def test_source_validation_rejects_edits(self):
        with tempfile.TemporaryDirectory() as tmp, patch('efficiency.common.ROOT', Path(tmp)):
            p = Path(tmp) / 'experiment.py'
            p.write_text('original')
            receipt = dict(passed=True, source_sha256={'experiment.py': digest_file(p)})
            self.assertTrue(verify_source(receipt))
            p.write_text('edited')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verify_source(receipt)

    def test_added_source_file_invalidates_validation(self):
        with tempfile.TemporaryDirectory() as tmp, patch('efficiency.common.ROOT', Path(tmp)):
            root = Path(tmp)
            (root / 'experiment.py').write_text('original')
            receipt = dict(passed=True, source_sha256={'experiment.py': digest_file(root / 'experiment.py')})
            (root / 'efficiency').mkdir()
            (root / 'efficiency/added.py').write_text('unvalidated')
            with self.assertRaisesRegex(ValueError, 'file set changed'):
                verify_source(receipt)


class ServiceTests(unittest.TestCase):
    def test_recovery_profile_cannot_be_downgraded_to_legacy(self):
        from efficiency.reliability_campaign import campaign_policy
        with self.assertRaisesRegex(ValueError, 'profile is missing or changed'):
            campaign_policy(dict(attempts=[], recovery_from='/previous'))

    def test_service_owns_runtime_and_stop_hook(self):
        command = service.build_service_command(service.ROOT / 'campaigns/recovery-test')
        for option in ('--user', '--service-type=exec', '--property=Restart=no',
                       '--property=KillMode=control-group', '--property=TimeoutStopSec=120',
                       '--property=RuntimeMaxSec=9h'):
            self.assertIn(option, command)
        self.assertTrue(any(x.startswith('--property=ExecStopPost=') for x in command))
        self.assertFalse(any(x in command for x in ('--scope', '--pipe', '--pty', '--wait')))

    def test_campaign_name_must_be_safe_for_unit_identity(self):
        for name in ('has spaces', 'dollar$', 'newline\n', 'percent%'):
            with self.assertRaises(ValueError):
                service.unit_name(Path('/tmp') / name)

    def test_direct_tool_owned_hardware_driver_is_rejected(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'managed user service'):
                service.run_service(Path('/tmp/not-a-service'))

    def test_missing_summary_stops_before_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'campaigns/test'
            campaign.mkdir(parents=True)
            (root / 'runs').mkdir()
            governor = root / 'governor'
            governor.write_text('ondemand')
            atomic_json(campaign / 'campaign.json', dict(source_campaign='/source', original_governor='ondemand', attempts=[]))
            atomic_json(campaign / 'execution.json', [])
            atomic_json(campaign / 'implementation-validation.json', {})
            with patch.object(service, 'ROOT', root), patch.object(service, 'STATE_ROOT', root / 'state'), \
                 patch('efficiency.refine_governor.GOVERNOR', governor), \
                 patch('efficiency.reliability_recovery.verify_source'), \
                 patch('efficiency.reliability_campaign.dependency'), \
                 patch('efficiency.reliability_campaign.allowance', return_value=14400), \
                 patch.object(service, 'run_child', return_value=-9) as child:
                with self.assertRaisesRegex(RuntimeError, 'missing_summary'):
                    service.execute_stages(campaign)
                self.assertEqual(child.call_count, 1)
            records = json.loads((campaign / 'execution.json').read_text())
            self.assertEqual(records[0]['returncode'], -9)
            self.assertFalse(records[0]['audit_passed'])

    def test_stop_hook_preserves_sealed_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / 'campaign'
            campaign.mkdir()
            (campaign / 'checksums.json').write_text('{}')
            governor = root / 'governor'
            governor.write_text('ondemand')
            with patch.object(service, 'STATE_ROOT', root / 'state'), patch('efficiency.refine_governor.GOVERNOR', governor):
                service.update(campaign, 'complete')
                service.stopped(campaign)
                self.assertEqual(json.loads((service.state_dir(campaign) / 'status.json').read_text())['phase'], 'complete')
                self.assertTrue(json.loads((service.state_dir(campaign) / 'stop-receipt.json').read_text())['sealed_campaign_unchanged'])
            self.assertEqual([p.name for p in campaign.iterdir()], ['checksums.json'])


class StreamingTamperTests(unittest.TestCase):
    def test_streaming_order_change_preserves_intervening_evidence(self):
        rows = [dict(event='measurement', section='main', n=1), dict(event='warmup', n=2),
                dict(event='measurement', section='main', n=3), dict(event='measurement_complete', n=4)]
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'in', Path(tmp) / 'out'
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            rewrite_events(source, out, 'order')
            self.assertEqual([json.loads(l)['n'] for l in out.read_text().splitlines()], [3, 2, 1, 4])
            self.assertEqual([json.loads(l)['n'] for l in source.read_text().splitlines()], [1, 2, 3, 4])

    def test_no_suitable_evidence_is_not_a_passing_tamper_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'in', Path(tmp) / 'out'
            source.write_text('{"event":"failure"}\n')
            with self.assertRaisesRegex(ValueError, 'No suitable'):
                rewrite_events(source, out, 'order')


if __name__ == '__main__':
    unittest.main()
