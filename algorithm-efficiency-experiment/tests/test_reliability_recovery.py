"""Recovery scope, budget and preservation checks without hardware access."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from efficiency import reliability_campaign as campaign
from efficiency.common import atomic_json, digest_file
from efficiency.reliability_report import finalize
from efficiency.reliability_spec import CAPS, MAX_SECONDS, VERSION
from efficiency.research_report import seal


def recovery_ledger(**values):
    result=dict(profile='cpu-gpu-recovery',max_seconds=28800,
                allowed_stages=['cpu-develop','cpu-confirm','gpu-confirm'],attempts=[])
    result.update(values)
    return result


class RecoveryScopeTests(unittest.TestCase):
    def test_legacy_budget_and_scope_are_unchanged(self):
        ledger=dict(attempts=[])
        self.assertEqual(campaign.campaign_policy(ledger),dict(
            profile='reliability',max_seconds=MAX_SECONDS,allowed_stages=list(CAPS)))
        for stage,cap in CAPS.items():self.assertEqual(campaign.allowance(ledger,stage),cap)

    def test_recovery_stage_caps_and_no_borrowing(self):
        ledger=recovery_ledger()
        for stage,cap in zip(ledger['allowed_stages'],(14400,7200,7200)):
            self.assertEqual(campaign.allowance(ledger,stage),cap)
            with self.assertRaises(ValueError):campaign.allowance(ledger,stage,cap+1)
        ledger['attempts']=[dict(stage='cpu-develop',charged_seconds=14300)]
        with self.assertRaisesRegex(ValueError,'exhausted'):campaign.allowance(ledger,'cpu-develop')
        self.assertEqual(campaign.allowance(ledger,'cpu-confirm'),7200)

    def test_aggregate_budget_enforced(self):
        ledger=recovery_ledger(attempts=[dict(stage=stage,charged_seconds=charged) for stage,charged in
            [('cpu-develop',14200),('cpu-confirm',7200),('gpu-confirm',7200)]])
        self.assertEqual(campaign.allowance(ledger,'cpu-develop'),200)
        ledger['attempts'][0]['charged_seconds']=14280
        with self.assertRaisesRegex(ValueError,'exhausted'):campaign.allowance(ledger,'cpu-develop')

    def test_out_of_scope_stages_rejected(self):
        for stage in ('npu-develop','npu-confirm','combined-develop','combined-confirm'):
            with self.assertRaisesRegex(ValueError,'scope'):campaign.allowance(recovery_ledger(),stage)
            with self.assertRaisesRegex(ValueError,'scope'):campaign.dependency(recovery_ledger(),stage)

    def test_scope_cannot_be_widened_or_reordered(self):
        for changes in (dict(max_seconds=57600),dict(allowed_stages=list(CAPS)),
                        dict(allowed_stages=['gpu-confirm','cpu-confirm','cpu-develop']),dict(profile='unknown')):
            with self.assertRaises(ValueError):campaign.campaign_policy(recovery_ledger(**changes))
        with self.assertRaisesRegex(ValueError,'scope'):
            campaign.campaign_policy(recovery_ledger(attempts=[dict(stage='npu-develop')]))

    def test_stage_rejection_precedes_source_locks_and_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);saved=base/'campaign';saved.mkdir()
            atomic_json(saved/'campaign.json',recovery_ledger())
            args=SimpleNamespace(stage='npu-develop',max_seconds=None,campaign=saved,
                                 source_campaign=base/'missing-source',output=base/'output')
            with patch.object(campaign,'verify_artifacts') as verify,patch.object(campaign,'build') as build:
                with self.assertRaisesRegex(ValueError,'scope'):campaign.run(args)
            verify.assert_not_called();build.assert_not_called()
            self.assertFalse(args.output.exists());self.assertFalse((saved/'.lock').exists())


class RecoveryPersistenceTests(unittest.TestCase):
    def test_interruption_without_summary_charges_full_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=recovery_ledger(attempts=[dict(stage='cpu-develop',output=tmp,state='running',
                                                reserved_seconds=14400,charged_seconds=12)])
            campaign.recover(ledger)
            row=ledger['attempts'][0]
            self.assertEqual(row['state'],'interrupted');self.assertEqual(row['charged_seconds'],14400)
            self.assertFalse(row['scientific_complete']);self.assertFalse(row['audit_passed'])
            with self.assertRaisesRegex(ValueError,'exhausted'):campaign.allowance(ledger,'cpu-develop')

    def test_sealed_completion_is_closed_even_without_completion_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);atomic_json(p/'summary.json',dict(complete=True));atomic_json(p/'validation.json',dict(passed=True));seal(p)
            ledger=recovery_ledger(attempts=[dict(stage='cpu-develop',output=tmp,state='running',reserved_seconds=14400)])
            campaign.recover(ledger)
            self.assertTrue(ledger['attempts'][0]['closed'])
            self.assertEqual(ledger['attempts'][0]['checksums_sha256'],digest_file(p/'checksums.json'))
            with self.assertRaisesRegex(ValueError,'immutable'):campaign.dependency(ledger,'cpu-develop')

    def test_prepared_config_keeps_runner_profile_and_recovery_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'runs').mkdir();saved=base/'campaign';saved.mkdir();source=base/'source';source.mkdir()
            governor=base/'governor';governor.write_text('ondemand')
            ledger=recovery_ledger(version=VERSION,campaign_id='new-id',source_campaign=str(source),
                                   original_governor='ondemand',recovery_from='/old-campaign',recovery_lineage_sha256='a'*64)
            atomic_json(saved/'campaign.json',ledger);atomic_json(saved/'registry.json',{})
            atomic_json(source/'final-validation.json',dict(passed=True))
            args=SimpleNamespace(stage='cpu-develop',max_seconds=None,campaign=saved,source_campaign=source,output=base/'runs/run')
            helper=Mock();helper.poll.return_value=0
            def launch(argv,**kwargs):
                Path(argv[argv.index('--socket')+1]).touch();return helper
            seen={}
            def supervise(directory,config):
                seen.update(config)
                atomic_json(directory/'outcome.json',dict(status='stopped'))
                return SimpleNamespace(execute=lambda:1)
            with patch.object(campaign,'ROOT',base),patch.object(campaign,'GOVERNOR',governor),\
                 patch.object(campaign,'verify_artifacts'),patch.object(campaign,'build'),\
                 patch.object(campaign.subprocess,'Popen',side_effect=launch),\
                 patch.object(campaign,'request',return_value=dict(original='ondemand')),\
                 patch('efficiency.runner.Supervisor',side_effect=supervise),\
                 patch('efficiency.reliability_report.build_report',side_effect=RuntimeError('chart failure')),\
                 patch.dict('os.environ',{},clear=False),patch('builtins.print'):
                self.assertEqual(campaign.run(args),1)
            self.assertEqual(seen['profile'],'reliability');self.assertEqual(seen['campaign_profile'],'cpu-gpu-recovery')
            self.assertEqual(seen['campaign_max_seconds'],28800);self.assertEqual(seen['allowed_stages'],ledger['allowed_stages'])
            self.assertEqual(seen['recovery_lineage_sha256'],'a'*64)
            self.assertTrue((args.output/'checksums.json').exists())
            summary=json.loads((args.output/'summary.json').read_text())
            self.assertEqual(summary['decision'],'incomplete');self.assertIsNone(summary['selected'])
            saved_ledger=json.loads((saved/'campaign.json').read_text())
            self.assertFalse(saved_ledger['attempts'][0]['audit_passed'])


class RecoveryFinalizationTests(unittest.TestCase):
    def make_campaign(self,base):
        saved=base/'campaign';saved.mkdir();governor=base/'governor';governor.write_text('ondemand')
        ledger=recovery_ledger(original_governor='ondemand',recovery_from='/old-campaign',recovery_lineage_sha256='a'*64)
        atomic_json(saved/'campaign.json',ledger);atomic_json(saved/'historical-seals.json',[])
        atomic_json(saved/'execution.json',[])
        for filename in ('implementation-validation.json','negative-gates.json','audit-tamper-tests.json'):
            atomic_json(saved/filename,dict(passed=True))
        return saved,governor,ledger

    def test_recovery_report_uses_eight_hours_and_checks_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved,governor,_=self.make_campaign(Path(tmp));verify=Mock();diagnostic=Mock(return_value=dict(passed=True))
            with patch('efficiency.refine_governor.GOVERNOR',governor),patch.dict(sys.modules,{
                    'efficiency.reliability_recovery':SimpleNamespace(verify_lineage=verify,verify_diagnostic=diagnostic)}):finalize(saved)
            verify.assert_called_once_with(saved)
            diagnostic.assert_called_once_with(saved)
            self.assertTrue(json.loads((saved/'final-validation.json').read_text())['diagnostic']['passed'])
            summary=json.loads((saved/'summary.json').read_text())
            self.assertEqual(summary['max_seconds'],28800);self.assertEqual(summary['profile'],'cpu-gpu-recovery')
            context=(saved/'CONTEXT.txt').read_text()
            self.assertIn('8-hour',context);self.assertNotIn('16-hour',context);self.assertNotIn('full-table',context)

    def test_unresolved_attempt_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved,_,ledger=self.make_campaign(Path(tmp));ledger['attempts']=[dict(stage='cpu-develop',state='running')]
            atomic_json(saved/'campaign.json',ledger)
            with patch.dict(sys.modules,{'efficiency.reliability_recovery':SimpleNamespace(verify_lineage=Mock(),verify_diagnostic=Mock())}):
                with self.assertRaisesRegex(ValueError,'Unresolved'):finalize(saved)
            self.assertFalse((saved/'checksums.json').exists())

    def test_changed_lineage_prevents_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved,_,_=self.make_campaign(Path(tmp))
            with patch.dict(sys.modules,{'efficiency.reliability_recovery':SimpleNamespace(
                    verify_lineage=Mock(side_effect=ValueError('Lineage changed')),verify_diagnostic=Mock())}):
                with self.assertRaisesRegex(ValueError,'Lineage'):finalize(saved)
            self.assertFalse((saved/'summary.json').exists())

    def test_changed_stage_scope_prevents_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);saved,_,ledger=self.make_campaign(base);run=base/'run';run.mkdir()
            atomic_json(run/'config.json',dict(campaign_profile='cpu-gpu-recovery',campaign_max_seconds=57600,
                allowed_stages=ledger['allowed_stages'],recovery_from=ledger['recovery_from'],
                recovery_lineage_sha256=ledger['recovery_lineage_sha256']))
            seal(run)
            ledger['attempts']=[dict(stage='cpu-develop',state='stopped',output=str(run),charged_seconds=10,
                                      checksums_sha256=digest_file(run/'checksums.json'))]
            atomic_json(saved/'campaign.json',ledger)
            with patch.dict(sys.modules,{'efficiency.reliability_recovery':SimpleNamespace(verify_lineage=Mock(),verify_diagnostic=Mock())}):
                with self.assertRaisesRegex(ValueError,'metadata'):finalize(saved)
            self.assertFalse((saved/'checksums.json').exists())

    def test_failed_diagnostic_verification_prevents_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved,_,_=self.make_campaign(Path(tmp))
            with patch.dict(sys.modules,{'efficiency.reliability_recovery':SimpleNamespace(
                    verify_lineage=Mock(),verify_diagnostic=Mock(side_effect=ValueError('Diagnostic seal changed')))}):
                with self.assertRaisesRegex(ValueError,'Diagnostic'):finalize(saved)
            self.assertFalse((saved/'summary.json').exists());self.assertFalse((saved/'checksums.json').exists())


if __name__=='__main__':unittest.main()
