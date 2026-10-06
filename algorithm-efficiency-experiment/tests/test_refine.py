from collections import Counter
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from efficiency import refine_spec as spec
from efficiency.refine_campaign import allowance,dependency,recover
from efficiency.refine_governor import Lease
from efficiency.refine_scoring import score
from efficiency.refine_audit import semantic_score,decisions,check_interval,check_dialogue
from efficiency.refine_worker import environment
from efficiency.coordination_workers import Worker
from efficiency.study_worker import renderer
from efficiency.study_spec import PARAMETERS,make_table
from efficiency.diagnostic import score_answer

IMPORTED_BINDING=os.environ.get('OMP_PROC_BIND')

class EnvironmentWorker:
    def __init__(self,directory,config,stack):
        self.environment=dict(imported_binding=IMPORTED_BINDING,current=os.environ.get('OMP_PROC_BIND'))
    def perform(self,task):return {}

def config(stage='cpu-develop',selected=None):
    return dict(refine_stage=stage,fixture_namespace='unseen-test-namespace',selected=selected)

def numeric_events(c):
    events=[]
    for job in spec.schedule(c,spec.fixtures(c)):
        if job['section']!='main':continue
        value={'cpu':10,'cpu_duplicate':10,'gpu':100,'stream64':40}[job['arm']['arm']]
        events.append(dict(event='measurement',**job,total_ms=value,response=dict(result=dict(correct=True,matches_warmup=True))))
    events.append(dict(event='measurement_complete'));return events

def npu_events(c):
    events=[]
    for job in spec.schedule(c,spec.fixtures(c)):
        if job['section']!='main':continue
        rows=[dict(valid=True,answer_correct=job['condition']=='punctuation') for _ in range(4)]
        events.append(dict(event='measurement',**job,response=dict(result=dict(rows=rows,contract_failures=[],reset_verified=True))))
    return events+[dict(event='measurement_complete')]

class DesignTests(unittest.TestCase):
    def test_policy_carryover_balanced(self):
        orders=[spec.policy_order(b) for b in range(8)]
        for position in range(8):self.assertEqual(set(row[position] for row in orders),set(range(8)))
        pairs=Counter((a,b) for row in orders for a,b in zip(row,row[1:]))
        self.assertEqual(len(pairs),56);self.assertEqual(set(pairs.values()),{1})
    def test_all_orders_every_cycle(self):
        c=config('cpu-confirm',2)
        for cycle in range(4):
            orders=[o for b in range(cycle*8,cycle*8+8) for o in spec.permutations(c,'shape',b,'main')]
            self.assertEqual(set(orders),set(itertools.permutations(spec.LABELS)))
            self.assertEqual(set(Counter((i,a) for o in orders for i,a in enumerate(o)).values()),{6})
            self.assertEqual(set(Counter((a,b) for o in orders for a,b in zip(o,o[1:])).values()),{6})
    def test_shared_inputs_and_matrix(self):
        c=config();fs=spec.fixtures(c);jobs=spec.schedule(c,fs)
        self.assertEqual(len(fs),54);self.assertEqual(len(jobs),9*8*6*12)
        self.assertEqual(len({f['seed'] for f in fs}),len(fs))
        for f in fs:self.assertEqual({j['policy'] for j in jobs if j['fixture_id']==f['fixture_id']},set(range(8)))
    def test_fresh_stages(self):
        a={f['seed'] for f in spec.fixtures(config())}
        b={f['seed'] for f in spec.fixtures(config('cpu-confirm',2))}
        self.assertFalse(a&b)
    def test_npu_balanced_same_fixtures(self):
        c=config('npu-develop');fs=spec.fixtures(c);jobs=spec.schedule(c,fs)
        self.assertEqual(len(jobs),54)
        self.assertEqual(Counter(f['first_color'] for f in fs if f['section']=='main'),dict.fromkeys(spec.COLORS,3))
        for f in fs:self.assertEqual({j['condition'] for j in jobs if j['fixture_id']==f['fixture_id']},{'native','punctuation'})
    def test_environment_settings(self):
        for p in spec.POLICIES:
            e=environment(p);self.assertEqual(e['OMP_DYNAMIC'],'FALSE');self.assertIsNone(e['GOMP_CPU_AFFINITY'])
            self.assertEqual(e['OMP_WAIT_POLICY'],'PASSIVE' if p['waiting']=='passive' else None)
    def test_environment_before_child_import_and_restored(self):
        old=os.environ.get('OMP_PROC_BIND')
        with tempfile.TemporaryDirectory() as path:
            w=Worker('process','numeric',path,{},lambda:None,factory=EnvironmentWorker,environment={'OMP_PROC_BIND':'CLOSE'})
            try:self.assertEqual(w.ready['environment'],dict(imported_binding='CLOSE',current='CLOSE'))
            finally:w.close()
        self.assertEqual(os.environ.get('OMP_PROC_BIND'),old);self.assertFalse(w.release['forced'])
    def test_no_scientific_retries(self):
        with self.assertRaisesRegex(ValueError,'immutable'):dependency(dict(attempts=[dict(stage='cpu-develop',scientific_complete=True)]),'cpu-develop')
    def test_missing_parent_closed(self):
        with self.assertRaisesRegex(ValueError,'closed'):dependency(dict(attempts=[]),'gpu-confirm')
    def test_caps_no_borrowing(self):
        self.assertEqual(allowance(dict(attempts=[]),'cpu-confirm'),3600)
        with self.assertRaises(ValueError):allowance(dict(attempts=[]),'cpu-confirm',7200)
        with self.assertRaises(ValueError):allowance(dict(attempts=[dict(stage='cpu-develop',charged_seconds=7100)]),'cpu-develop')
    def test_campaign_cap(self):
        with self.assertRaises(ValueError):allowance(dict(attempts=[dict(stage='cpu-develop',charged_seconds=28700)]),'npu-develop')
    def test_invalid_caps(self):
        for v in (float('nan'),float('inf'),120,-1):
            with self.assertRaises(ValueError):allowance(dict(attempts=[]),'npu-develop',v)
    def test_recovery_charges_full_and_closes_completed_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'refine.jsonl').write_text(json.dumps(dict(event='measurement_complete'))+'\n')
            ledger=dict(attempts=[dict(stage='cpu-develop',state='running',output=tmp,reserved_seconds=7200,charged_seconds=0)])
            recover(ledger);self.assertEqual(ledger['attempts'][0]['charged_seconds'],7200)
            with self.assertRaisesRegex(ValueError,'immutable'):dependency(ledger,'cpu-develop')
    def test_infrastructure_retry_spends_remaining_stage_allowance(self):
        ledger=dict(attempts=[dict(stage='npu-develop',charged_seconds=1000,scientific_complete=False)])
        self.assertEqual(dependency(ledger,'npu-develop'),(None,None))
        self.assertEqual(allowance(ledger,'npu-develop'),6200)
    def test_independent_intervals(self):
        a=[1,2,3,4];b=[1,1,2,2];value=spec.interval(a,b)
        check_interval(value,a,b)
        value['interval'][0]+=.01
        with self.assertRaises(ValueError):check_interval(value,a,b)
    def test_cpu_tie_selection_and_audit(self):
        c=config();events=numeric_events(c);summary=spec.analyze(events,c)
        self.assertEqual(summary['selected'],2)
        decisions(events[:-1],c,summary)
        summary['selected']=0
        with self.assertRaises(ValueError):decisions(events[:-1],c,summary)
    def test_each_shape_must_be_equivalent(self):
        c=config('cpu-confirm',2);events=numeric_events(c)
        for r in events[:-1]:
            if r['cell_id']==spec.CELLS[0]['cell_id'] and r['arm']['arm']=='cpu_duplicate':r['total_ms']=11
        summary=spec.analyze(events,c);self.assertIsNone(summary['selected'])
        self.assertLess(summary['metrics'][0]['cpu_equivalence']['aggregate']['estimate'],1.05)
    def test_gpu_gain_not_enough_without_equivalence(self):
        c=config('gpu-confirm',2);events=numeric_events(c)
        for r in events[:-1]:
            if r['arm']['arm']=='cpu_duplicate':r['total_ms']=12
        self.assertFalse(spec.analyze(events,c)['accepted'])
    def test_incomplete_never_promotes(self):
        c=config('gpu-confirm',2);events=numeric_events(c)
        self.assertFalse(spec.analyze(events[1:],c)['complete'])
    def test_npu_development_and_independent_confirmation(self):
        for stage in ('npu-develop','npu-confirm'):
            c=config(stage);events=npu_events(c);summary=spec.analyze(events,c)
            self.assertEqual(summary['selected'],'punctuation');decisions(events[:-1],c,summary)
    def test_npu_completion_is_absolute(self):
        c=config('npu-develop');events=npu_events(c)
        next(e for e in events if e.get('condition')=='punctuation')['response']['result']['rows'][0]['valid']=False
        self.assertIsNone(spec.analyze(events,c)['selected'])
    def test_npu_gain_required_in_confirmation(self):
        c=config('npu-confirm');events=npu_events(c)
        for e in events[:-1]:
            for r in e['response']['result']['rows']:r['answer_correct']=True
        self.assertIsNone(spec.analyze(events,c)['selected'])
    def test_npu_per_size_threshold(self):
        c=config('npu-develop');events=npu_events(c)
        job=next(e for e in events if e.get('condition')=='punctuation' and e['target_tokens']==128)
        for r in job['response']['result']['rows']:r['answer_correct']=False
        summary=spec.analyze(events,c);self.assertEqual(summary['metrics'][1]['correct'],92)
        self.assertIsNone(summary['selected'])

class GovernorTests(unittest.TestCase):
    def setUp(self):
        self.clock=0;self.value='ondemand';self.alive=True;self.writes=[]
        def write(v):self.value=v;self.writes.append(v)
        self.lease=Lease(lambda:self.value,write,lambda:self.alive,lambda:self.clock,7200)
    def test_transition_restore(self):
        self.lease.request('performance');self.assertEqual(self.value,'performance')
        self.lease.restore();self.assertEqual(self.value,'ondemand')
    def test_invalid_governor_does_not_write(self):
        with self.assertRaises(ValueError):self.lease.request('powersave')
        self.assertEqual(self.writes,[])
    def test_watchdog_process_death(self):
        self.lease.request('performance');self.alive=False;self.assertFalse(self.lease.check());self.assertEqual(self.value,'ondemand')
    def test_watchdog_stale_heartbeat(self):
        self.lease.request('performance');self.clock=61;self.assertFalse(self.lease.check());self.assertEqual(self.value,'ondemand')
    def test_absolute_deadline_not_extended(self):
        for i in range(1,121):
            self.clock=i*59;self.lease.request()
        self.clock=7200;self.assertFalse(self.lease.check())
    def test_readback_failure(self):
        self.lease.write=lambda v:None
        with self.assertRaisesRegex(RuntimeError,'readback'):self.lease.request('performance')
    def test_restoration_failure(self):
        self.lease.request('performance');self.lease.write=lambda v:None
        with self.assertRaisesRegex(RuntimeError,'restoration'):self.lease.restore()
    def test_closed_lease_rejected(self):
        self.lease.restore()
        with self.assertRaises(RuntimeError):self.lease.request('performance')

class ScoringTests(unittest.TestCase):
    native=['<|end_of_text|>','<|eom_id|>','<|eot_id|>']
    def check(self,output,status='LOGICAL_END_OF_GENERATION'):
        args=(output,'blue','item001',self.native,self.native+['.','\n'],status)
        result=score(*args);self.assertEqual(result,semantic_score(*args));return result
    def test_period(self):self.assertTrue(self.check('Blue.')['strict_correct'])
    def test_newline(self):self.assertTrue(self.check('blue\n')['strict_correct'])
    def test_native_after_period(self):self.assertTrue(self.check('Blue.<|eot_id|>')['strict_correct'])
    def test_extra_text_rejected(self):self.assertFalse(self.check('Blue. Because the answer is blue.<|eot_id|>')['strict_correct'])
    def test_empty_rejected(self):self.assertEqual(self.check('\n')['score_reason'],'empty_output')
    def test_wrong_fact(self):self.assertFalse(self.check('Red.')['factual_correct'])
    def test_missing_termination(self):self.assertEqual(self.check('Blue')['score_reason'],'missing_terminal_token')
    def test_max_tokens_with_recovery(self):self.assertEqual(self.check('Blue.<|eot_id|>','MAX_TOKENS_REACHED')['answer_category'],'truncated')
    def test_embedded_marker(self):self.assertEqual(self.check('Blue<|eot_id|>.<|eot_id|>')['score_reason'],'exposed_control_marker')
    def test_wrong_identifier(self):self.assertEqual(self.check('item002 is blue.')['score_reason'],'wrong_identifier')
    def test_whole_answer_non_strict(self):
        r=self.check('The color of item001 is blue.');self.assertTrue(r['factual_correct']);self.assertFalse(r['strict_correct'])
    def test_legacy_native_scores_unchanged(self):
        for body in ('Blue','Blue.','it is blue','item001 = red','item002 is blue','Blue. More text',''):
            old=score_answer(body+'<|eot_id|>','blue','item001',self.native)
            new=score(body+'<|eot_id|>','blue','item001',self.native,self.native,'LOGICAL_END_OF_GENERATION')
            for key in ('strict_correct','factual_correct','format_correct','answer_category'):self.assertEqual(old[key],new[key])
    def test_actual_llama_multiturn_template(self):
        from efficiency.common import ROOT,read_jsonl
        events,_=read_jsonl(ROOT/'runs/study-npu-develop-retry1-20261003/study.jsonl')
        env=next(e['environment'] for e in events if e['event']=='worker_ready' and e['environment']['model_id']=='llama')
        render=renderer(env['prompt_template'],'llama')
        text=render([dict(role='system',content='Test'),dict(role='user',content='First?'),dict(role='assistant',content='Blue.'),dict(role='user',content='Next?')])
        self.assertEqual(text.count('<|begin_of_text|>'),1);self.assertIn('Blue.',text);self.assertIn('Next?',text)

if __name__=='__main__':unittest.main()
