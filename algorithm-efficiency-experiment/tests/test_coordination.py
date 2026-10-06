import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from efficiency.coordination import Budget
from efficiency.coordination_audit import interval, raw_advice
from efficiency.coordination_report import analyze, build_report
from efficiency.coordination_spec import (advice_cases, fixture_specs, pilot_required_seconds,
                                          schedule, specification)
from efficiency.coordination_workers import NumericWorker, Worker, owner, run_condition
from efficiency.feedback_spec import paired_interval, parse_proposal, split_manifest


class FakeWorker:
    def __init__(self, directory, config, stack):
        self.identity=owner();self.environment=dict(test=True)
        stack.callback(lambda: Path(directory,f'closed-{self.identity["pid"]}-{self.identity["tid"]}').write_text(json.dumps(owner())))
    def perform(self, task):
        if task['operation']=='crash':os._exit(7)
        if task['operation']=='block':time.sleep(20)
        if task['operation']=='error':raise ValueError('injected failure')
        if owner()!=self.identity:raise RuntimeError('Owner changed')
        time.sleep(.02)
        return dict(identity=self.identity, operation=task['operation'])


class WorkerTests(unittest.TestCase):
    def test_same_workers_for_serial_and_overlap_both_architectures(self):
        for arch in ('thread','process'):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as d:
                workers=[]
                try:
                    for kind in ('numeric','hat'):
                        workers.append(Worker(arch,kind,d,{},lambda:None,FakeWorker))
                    identities=[w.ready['owner'] for w in workers]
                    self.assertEqual(identities[0]['pid']==identities[1]['pid'],arch=='thread')
                    self.assertNotEqual(identities[0]['tid'],identities[1]['tid'])
                    for c in ('cpu_serial','cpu_overlap','gpu_serial','gpu_overlap'):
                        row=run_condition(*workers,c,['test'],{},time.monotonic()+10)
                        for i,key in enumerate(('numerical','adviser')):
                            self.assertEqual(row[key]['owner'],identities[i])
                            self.assertEqual(row[key]['result']['identity'],identities[i])
                        if c.endswith('serial'):
                            self.assertLessEqual(row['numerical']['delivered'],row['adviser']['submitted'])
                        else:
                            self.assertLess(row['adviser']['submitted'],row['numerical']['delivered'])
                finally:
                    for w in reversed(workers):w.close()
                for identity in identities:
                    path=Path(d,f'closed-{identity["pid"]}-{identity["tid"]}')
                    self.assertEqual(json.loads(path.read_text()),identity)
                self.assertTrue(all(not w.release['alive'] and not w.release['forced'] for w in workers))

    def test_worker_error_propagates_and_releases_context(self):
        with tempfile.TemporaryDirectory() as d:
            w=Worker('thread','numeric',d,{},lambda:None,FakeWorker)
            try:
                with self.assertRaisesRegex(RuntimeError,'injected failure'):w.call('error')
            finally:w.close()
            self.assertFalse(w.release['alive'])

    def test_process_crash_does_not_hang_parent(self):
        with tempfile.TemporaryDirectory() as d:
            w=Worker('process','numeric',d,{},lambda:None,FakeWorker)
            try:
                with self.assertRaisesRegex(RuntimeError,'exited without'):w.call('crash')
            finally:w.close()
            self.assertFalse(w.release['alive'])

    def test_deadline_kills_blocked_process(self):
        with tempfile.TemporaryDirectory() as d:
            w=Worker('process','hat',d,{},lambda:None,FakeWorker)
            deadline=time.monotonic()+.15
            def check():
                if time.monotonic()>=deadline:raise TimeoutError('test deadline')
            w.check=check
            try:
                with self.assertRaises(TimeoutError):w.call('block')
            finally:w.close()
            self.assertTrue(w.release['forced']);self.assertFalse(w.release['alive'])

    def test_busy_worker_rejects_second_submission(self):
        with tempfile.TemporaryDirectory() as d:
            w=Worker('thread','numeric',d,{},lambda:None,FakeWorker)
            try:
                w.submit('numeric')
                with self.assertRaises(RuntimeError):w.submit('numeric')
                w.result()
            finally:w.close()


class SpecificationTests(unittest.TestCase):
    def test_new_independent_seeds_and_full_balanced_schedule(self):
        fixtures=fixture_specs();blocks=schedule()
        self.assertEqual(len(fixtures),21)
        self.assertFalse({r['seed'] for r in fixtures}&{r['seed'] for r in split_manifest()})
        main=[b for b in blocks if b['stage']=='main']
        self.assertEqual(sum(len(b['conditions']) for b in main),48)
        for r in range(6):
            pair=[b for b in main if b['repeat']==r]
            self.assertEqual(pair[0]['fixture_ids'],pair[1]['fixture_ids'])
            self.assertEqual(pair[0]['case_id'],f'holdout-{r:02}')
            self.assertEqual(pair[0]['architecture'],'thread' if r%2==0 else 'process')

    def test_advice_sets_are_distinct_and_never_offer_tested_ids(self):
        cases=advice_cases()
        self.assertEqual(len(cases),41)
        self.assertEqual(len({json.dumps(c['messages']) for c in cases}),41)
        for c in cases:
            payload=json.loads(c['messages'][1]['content'])
            self.assertFalse(set(payload['eligible'])&set(payload['tested']))
            self.assertEqual(set(payload['development_ms']),set(payload['tested']))

    def test_pilot_budget_keeps_all_six_repetitions(self):
        rows=[dict(architecture=a,elapsed_seconds=50) for a in ('thread','process')]
        self.assertEqual(pilot_required_seconds(rows),765)
        with self.assertRaises(ValueError):pilot_required_seconds(rows[:1])

    def test_strict_parser_rejects_malformed_truncated_and_ineligible(self):
        for output,status,ledger in [('C2 extra<END>','LOGICAL_END_OF_GENERATION',True),
                ('C2<END>','MAX_TOKENS_REACHED',True),('C3<END>','LOGICAL_END_OF_GENERATION',True),
                ('C2<END>','LOGICAL_END_OF_GENERATION',False)]:
            self.assertIsNone(parse_proposal(output,['<END>'],status,ledger,['C2'])[0])

    def test_independent_bootstrap_matches_paired_block_interval(self):
        pairs=[(100+i*2,60+i*3) for i in range(6)]
        ratio,ci=interval(pairs,91)
        expected=paired_interval(pairs,91)
        self.assertEqual((ratio,ci),(expected['ratio'],expected['ci95']))

    def test_numerical_failure_is_not_a_timing_result(self):
        native=type('Native',(),dict(shape=(1,1,1),run=lambda *a:dict(validation_errors=0)))()
        worker=NumericWorker.__new__(NumericWorker)
        worker.data={'f':(np.ones((1,1,1),np.float32),np.ones((3,1,1),np.float32),np.ones((1,1,1)))}
        worker.cpu=native
        with patch('efficiency.coordination_workers.errors',return_value={'correct':False}):
            with self.assertRaisesRegex(RuntimeError,'Numerical failure'):worker.request('f','native4')

    def test_deadline_and_stale_telemetry(self):
        with tempfile.TemporaryDirectory() as d:
            now=time.monotonic()
            Path(d,'status.json').write_text(json.dumps(dict(monotonic=now,elapsed_seconds=0)))
            b=Budget(d,{'max_seconds':30})
            with self.assertRaises(TimeoutError):b.check()
            Path(d,'status.json').write_text(json.dumps(dict(monotonic=now-20,elapsed_seconds=0)))
            b=Budget(d,{'max_seconds':300})
            with self.assertRaisesRegex(RuntimeError,'stale'):b.check()

    def test_incomplete_report_is_renderable_without_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('efficiency.coordination_report.release_snapshot'):
                build_report(Path(d),charts=False)
            result=json.loads(Path(d,'summary.json').read_text())
            self.assertFalse(result['complete']);self.assertFalse(result['recommend_process_overlap'])
            self.assertTrue(Path(d,'report.html').is_file())


def synthetic_rows(valid=True):
    cases=advice_cases();by_id={c['case_id']:c for c in cases}
    env={'stop_tokens':['<END>'],'parameters':{}}
    def response(cid):
        c=by_id[cid];choice=c['eligible'][0]
        return dict(result=dict(output=choice+'<END>' if valid else 'explanation<END>',
            completion_status='LOGICAL_END_OF_GENERATION',token_ledger_valid=True,
            context_before=0,context_after=10,expected_context_after=10,
            choice=choice if valid else None,messages=c['messages'],effective_parameters={},total_ms=100))
    rows=[dict(event='advice_check',split=c['split'],case_id=c['case_id'],response=response(c['case_id']))
          for c in cases if c['split']!='pilot']
    for b in schedule():
        if b['stage']!='main':continue
        for condition in b['conditions']:
            rows.append(dict(event='condition',stage='main',repeat=b['repeat'],architecture=b['architecture'],
                condition=condition,case_id=b['case_id'],total_ms=100 if b['architecture']=='thread' else 70,
                adviser=response(b['case_id']),numerical=dict(result=dict(correct=True,requests=[dict(
                    correct=True,validation_errors=0,request_ms=1,gpu_ms=1)]*24))))
    rows.append(dict(event='complete'))
    return rows,cases,env


class AnalysisTests(unittest.TestCase):
    def test_invalid_advice_cannot_turn_timing_gain_into_promotion(self):
        rows,cases,env=synthetic_rows(False)
        summary=analyze(rows,{'status':'complete'},cases,env)
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['comparisons']['primary_process_cpu_overlap']['timing_gate'])
        self.assertFalse(summary['recommend_process_overlap'])

    def test_valid_complete_results_can_recommend_opt_in(self):
        rows,cases,env=synthetic_rows()
        summary=analyze(rows,{'status':'complete'},cases,env)
        self.assertTrue(summary['recommend_process_overlap'])
        self.assertEqual(summary['observed_numerical_requests'],1152)

    def test_all_ten_comparisons_have_distinct_names(self):
        rows,cases,env=synthetic_rows()
        comparisons=analyze(rows,{'status':'complete'},cases,env)['comparisons']
        self.assertEqual(len(comparisons),10)
        self.assertEqual(comparisons['process_vs_thread_gpu_overlap']['left'],'thread_gpu_overlap')
        self.assertEqual(comparisons['process_gpu_overlap']['left'],'process_gpu_serial')

    def test_missing_duplicate_or_wrong_numerical_results_block_recommendation(self):
        rows,cases,env=synthetic_rows()
        for bad in (rows[:-2],rows+[rows[-2]]):
            summary=analyze(bad,{'status':'complete'},cases,env)
            self.assertFalse(summary['recommend_process_overlap'])
        rows[-2]['numerical']['result']['correct']=False
        self.assertFalse(analyze(rows,{'status':'complete'},cases,env)['recommend_process_overlap'])

    def test_independent_advice_audit_detects_changed_settings(self):
        rows,cases,env=synthetic_rows()
        row=rows[0]
        self.assertTrue(raw_advice(row['response'],cases[0],env))
        row['response']['result']['effective_parameters']={'temperature':2}
        with self.assertRaisesRegex(ValueError,'settings'):raw_advice(row['response'],cases[0],env)


if __name__=='__main__':unittest.main()
