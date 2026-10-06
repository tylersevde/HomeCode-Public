from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from efficiency import reliability_spec as spec
from efficiency.reliability_audit import check_interval, decisions
from efficiency.reliability_campaign import allowance, dependency, recover
from efficiency.reliability_report import build_report
from efficiency.reliability_worker import select_fact, fact_messages, ReliableNumeric
from efficiency.refine_worker import environment

def config(stage='cpu-develop',policy=7,selected=None):
    return dict(reliability_stage=stage,fixture_namespace='test-new-namespace',cpu_policy=policy,selected=selected)

def numeric_events(c):
    times=dict(single_a=10,single_b=10,batch_a=160,batch_b=160,cpu=10,cpu_duplicate=10,gpu=100,stream64=40)
    return [dict(event='measurement',**j,total_ms=times[j['label']],response=dict(result=dict(correct=True)))
            for j in spec.schedule(c,spec.fixtures(c)) if j['section']=='main']+[dict(event='measurement_complete')]

def npu_events(c):
    return [dict(event='measurement',**j,response=dict(result=dict(rows=[dict(valid=True,answer_correct=j['condition']!='native') for _ in range(4)],
                contract_failures=[],reset_verified=True))) for j in spec.schedule(c,spec.fixtures(c)) if j['section']=='main']+[dict(event='measurement_complete')]

def combined_events(c):
    times=dict(cpu_lookup=100,cpu_npu_serial=220,cpu_npu_overlap=150,gpu_1=80,gpu_2=85,gpu_6=120,baseline_a=100,baseline_b=100,candidate=80)
    return [dict(event='measurement',**j,total_ms=times[j['label']],correct=True,valid=True,answer_correct=[True]*4)
            for j in spec.schedule(c,spec.fixtures(c)) if j['section']=='main']+[dict(event='measurement_complete')]

class DesignTests(unittest.TestCase):
    def test_user_budget_allocation(self):
        self.assertEqual(sum(spec.CAPS.values()),16*3600)
        self.assertEqual(spec.CAPS['cpu-develop'],4*3600)
        self.assertEqual((spec.BLOCKS['cpu-develop'],spec.BLOCKS['cpu-confirm']),(128,256))
    def test_cpu_matrix_and_call_counts(self):
        c=config();jobs=spec.schedule(c,spec.fixtures(c));main=[j for j in jobs if j['section']=='main']
        self.assertEqual(len(jobs),129*4*6*12)
        self.assertEqual(sum(16 if j['label'].startswith('batch') else 1 for j in main),313344)
        self.assertEqual({j['policy'] for j in jobs},{1,3,5,7})
    def test_cpu_arm_balance(self):
        c=config();orders=[o for b in range(8) for o in spec.orders(c,spec.CPU_ARMS,'shape',b,'main')]
        self.assertEqual(set(orders),set(itertools.permutations(spec.CPU_ARMS)))
        pairs=Counter((a,b) for o in orders for a,b in zip(o,o[1:]))
        self.assertEqual(set(pairs.values()),{6})
    def test_williams_balance(self):
        for labels in (spec.CPU_POLICIES,spec.CONDITIONS):
            orders=[spec.williams(labels,b) for b in range(len(labels))]
            for position in range(len(labels)):self.assertEqual({o[position] for o in orders},set(labels))
            pairs=Counter((a,b) for o in orders for a,b in zip(o,o[1:]))
            self.assertEqual(len(pairs),len(labels)*(len(labels)-1));self.assertEqual(set(pairs.values()),{1})
    def test_combined_confirmation_balanced(self):
        c=config('combined-confirm',selected=dict(baseline='cpu_lookup',candidate='gpu_1'))
        jobs=spec.schedule(c,spec.fixtures(c));cell=spec.CELLS[0]['cell_id']
        orders=[tuple(j['label'] for j in jobs if j['block']==b and j['cell_id']==cell and j['section']=='main') for b in range(24)]
        self.assertEqual(set(Counter(orders).values()),{4});self.assertEqual(len(set(orders)),6)
    def test_unique_fixture_namespaces(self):
        a={f['seed'] for f in spec.fixtures(config())};b={f['seed'] for f in spec.fixtures(config('cpu-confirm'))}
        self.assertFalse(a&b)
        c=config();c['fixture_namespace']='retry';self.assertFalse(a&{f['seed'] for f in spec.fixtures(c)})
    def test_npu_denominators_and_conditions(self):
        for stage,answers,conditions in [('npu-develop',96,3),('npu-confirm',192,2)]:
            c=config(stage);jobs=spec.schedule(c,spec.fixtures(c));main=[j for j in jobs if j['section']=='main']
            self.assertEqual(len(main)*4,answers*conditions)
    def test_npu_each_size_has_balanced_control_order(self):
        c=config('npu-develop');jobs=spec.schedule(c,spec.fixtures(c))
        for n in (128,512,1024):
            orders=[[j['condition'] for j in jobs if j['section']=='main' and j['block']==b and j['target_tokens']==n] for b in range(8)]
            self.assertEqual(sum(o.index('native')<o.index('punctuation') for o in orders),4)
            self.assertEqual(len({tuple(o) for o in orders[:6]}),6)
    def test_tables_balance_all_queried_colors_and_identifiers(self):
        fs=[f for f in spec.fixtures(config('npu-develop')) if f['section']=='main']
        tables=[spec.table(f,20) for f in fs];counts=Counter(a for f in tables for a in f['answers'])
        self.assertEqual(counts,dict.fromkeys(spec.COLORS,12))
        self.assertGreater(len({f['table'][0][0] for f in tables}),8)
    def test_nonmonotonic_sizing_uses_largest_fit(self):
        f=spec.fixtures(config('npu-develop'))[0];f['target_tokens']=100
        render=lambda m:m[1]['content']
        def tokenize(text):
            count=text.count(' = ')
            return list(range(90 if count in (4,100,299) else 110))
        result=spec.sizing(render,tokenize,f)
        self.assertEqual(result['count'],299);self.assertEqual(len(result['counts']),297)
    def test_selector_never_receives_expected_answer(self):
        table=[['item003','blue'],['item027','green']]
        self.assertEqual(select_fact(table,'What color is item027?'),['item027','green'])
        self.assertIn('item027 = green',fact_messages(select_fact(table,'What color is item027?'),'What color is item027?')[1]['content'])
    def test_selector_rejects_missing_duplicate_and_nonquestion(self):
        for table,q in [([['item001','red']],'What color is item999?'),([['item001','red'],['item001','blue']],'What color is item001?'),([['item001','red']],'Tell me the answer')]:
            with self.assertRaises(ValueError):select_fact(table,q)

class EligibilityTests(unittest.TestCase):
    def setUp(self):
        self.small=patch.dict(spec.BLOCKS,{'cpu-develop':8,'cpu-confirm':8,'gpu-confirm':8});self.small.start();self.addCleanup(self.small.stop)
    def check(self,c,events):
        summary=spec.analyze(events,c);decisions(events[:-1],c,summary);return summary
    def test_cpu_tie_prefers_ondemand_unbound(self):
        c=config();self.assertEqual(self.check(c,numeric_events(c))['selected'],3)
    def test_one_unstable_shape_blocks_cpu(self):
        c=config('cpu-confirm');events=numeric_events(c)
        for r in events[:-1]:
            if r['label']=='single_b' and r['cell_id']==spec.CELLS[0]['cell_id']:r['total_ms']=12
        self.assertIsNone(self.check(c,events)['selected'])
    def test_batch_pass_cannot_replace_single_failure(self):
        c=config('cpu-confirm');events=numeric_events(c)
        for r in events[:-1]:
            if r['label']=='single_b':r['total_ms']=12
        s=self.check(c,events);self.assertIsNone(s['selected']);self.assertTrue(all(spec.equivalent(v) for v in s['metrics'][0]['equivalence']['batch'].values()))
    def test_single_pass_cannot_replace_batch_failure(self):
        c=config('cpu-confirm');events=numeric_events(c)
        for r in events[:-1]:
            if r['label']=='batch_b':r['total_ms']=200
        self.assertIsNone(self.check(c,events)['selected'])
    def test_gpu_requires_stable_cpu(self):
        c=config('gpu-confirm');events=numeric_events(c)
        for r in events[:-1]:
            if r['label']=='cpu_duplicate':r['total_ms']=12
        self.assertFalse(self.check(c,events)['accepted'])
    def test_gpu_requires_every_shape(self):
        c=config('gpu-confirm');events=numeric_events(c)
        for r in events[:-1]:
            if r['label']=='stream64' and r['cell_id']==spec.CELLS[0]['cell_id']:r['total_ms']=106
        self.assertFalse(self.check(c,events)['accepted'])
    def test_complete_gpu_confirmation(self):
        c=config('gpu-confirm');self.assertTrue(self.check(c,numeric_events(c))['accepted'])
    def test_perfect_diagnostic_cannot_unlock_npu(self):
        c=config('npu-develop');events=npu_events(c)
        for e in events[:-1]:
            if e['condition']=='punctuation':
                for r in e['response']['result']['rows']:r['answer_correct']=False
        s=self.check(c,events);self.assertIsNone(s['selected']);self.assertTrue(s['metrics'][2]['meets_numeric_threshold']);self.assertFalse(s['metrics'][2]['qualified'])
    def test_full_table_development_and_confirmation(self):
        for stage in ('npu-develop','npu-confirm'):
            c=config(stage);self.assertEqual(self.check(c,npu_events(c))['selected'],'punctuation')
    def test_one_invalid_npu_reply_rejects(self):
        c=config('npu-develop');events=npu_events(c)
        next(e for e in events if e.get('condition')=='punctuation')['response']['result']['rows'][0]['valid']=False
        self.assertIsNone(self.check(c,events)['selected'])
    def test_npu_per_size_gate(self):
        c=config('npu-develop');events=npu_events(c)
        for r in next(e for e in events if e.get('condition')=='punctuation')['response']['result']['rows']:r['answer_correct']=False
        s=self.check(c,events);self.assertEqual(s['metrics'][1]['correct'],92);self.assertIsNone(s['selected'])
    def test_npu_gain_is_required(self):
        c=config('npu-confirm');events=npu_events(c)
        for e in events[:-1]:
            for r in e['response']['result']['rows']:r['answer_correct']=True
        self.assertIsNone(self.check(c,events)['selected'])
    def test_combined_uses_cpu_lookup_baseline(self):
        c=config('combined-develop');s=self.check(c,combined_events(c));self.assertEqual(s['selected'],dict(baseline='cpu_lookup',candidate='gpu_1'))
    def test_combined_rejects_fast_incorrect_candidate(self):
        c=config('combined-develop');events=combined_events(c)
        for e in events[:-1]:
            if e['label'].startswith('gpu'):e['answer_correct'][0]=False
        self.assertIsNone(self.check(c,events)['selected'])
    def test_combined_rejects_speedup_only_over_slower_npu_route(self):
        c=config('combined-develop');events=combined_events(c)
        for e in events[:-1]:
            if e['label'].startswith('gpu'):e['total_ms']=110
        self.assertIsNone(self.check(c,events)['selected'])
    def test_combined_confirmation_and_control_failure(self):
        c=config('combined-confirm',selected=dict(baseline='cpu_lookup',candidate='gpu_1'));events=combined_events(c)
        self.assertTrue(self.check(c,events)['accepted'])
        for e in events[:-1]:
            if e['label']=='baseline_b':e['total_ms']=120
        self.assertFalse(self.check(c,events)['accepted'])
    def test_partial_matrices_never_select(self):
        for c,fn in [(config(),numeric_events),(config('npu-develop'),npu_events),(config('combined-develop'),combined_events)]:
            events=fn(c);self.assertIsNone(spec.analyze(events[1:],c)['selected']);self.assertFalse(spec.analyze(events[:-1],c)['complete'])
    def test_bootstrap_independent_reconstruction_and_tamper(self):
        a=[10,11,9,12];b=[11,12,10,13]
        for ratio in (True,False):
            value=spec.interval(a,b,ratio);check_interval(value,a,b,ratio);value['interval'][0]+=.01
            with self.assertRaises(ValueError):check_interval(value,a,b,ratio)
    def test_selection_tamper_rejected(self):
        c=config();events=numeric_events(c);s=spec.analyze(events,c);s['selected']=7
        with self.assertRaisesRegex(ValueError,'eligibility'):decisions(events[:-1],c,s)

class PersistenceTests(unittest.TestCase):
    def test_budget_charges_prior_attempts_without_borrowing(self):
        ledger=dict(attempts=[dict(stage='cpu-develop',charged_seconds=1000)])
        self.assertEqual(allowance(ledger,'cpu-develop'),13400)
        with self.assertRaises(ValueError):allowance(ledger,'npu-develop',7201)
    def test_campaign_exhaustion_and_invalid_values(self):
        with self.assertRaises(ValueError):allowance(dict(attempts=[dict(stage='x',charged_seconds=57500)]),'cpu-develop')
        for value in (float('nan'),float('inf'),120,-1):
            with self.assertRaises(ValueError):allowance(dict(attempts=[]),'cpu-develop',value)
    def test_all_dependent_stages_require_parents(self):
        for stage in spec.DEPENDENCIES:
            with self.assertRaisesRegex(ValueError,'closed'):dependency(dict(attempts=[]),stage)
    def test_completed_and_deferred_stages_remain_closed(self):
        for fields in (dict(scientific_complete=True),dict(closed=True)):
            with self.assertRaisesRegex(ValueError,'immutable'):dependency(dict(attempts=[dict(stage='cpu-develop',**fields)]),'cpu-develop')
    def test_recovery_charges_full_and_preserves_completion(self):
        for event in (dict(event='measurement_complete'),dict(event='pilot_gate',fits=False)):
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp,spec.EVENTS).write_text(json.dumps(event)+'\n');ledger=dict(attempts=[dict(output=tmp,state='running',reserved_seconds=14400,stage='cpu-develop')])
                recover(ledger);self.assertEqual(ledger['attempts'][0]['charged_seconds'],14400);self.assertTrue(ledger['attempts'][0]['closed'])
    def test_sealed_report_refuses_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp,'checksums.json');p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'immutable'):build_report(tmp)
            self.assertEqual(list(Path(tmp).iterdir()),[p]);self.assertEqual(p.read_text(),'{}')

class BatchTests(unittest.TestCase):
    def make_worker(self):
        from efficiency.coordination_workers import owner
        w=ReliableNumeric.__new__(ReliableNumeric);w.owner=owner();w.policy=spec.POLICIES[7];return w
    def test_every_call_verified_and_one_outer_observation_pair(self):
        w=self.make_worker();rows=[dict(correct=True,matches_warmup=True,validation_errors=0) for _ in range(16)]
        rows[7]['correct']=False
        with patch('efficiency.reliability_worker.sample',return_value=dict(governor='performance')) as sample,patch('efficiency.reliability_worker.Numeric.measure',side_effect=rows) as measured:
            r=w.perform(dict(operation='batch',request_ids=list(range(16)),deadline=float('inf'),fixture_id='x',arm={}))
        self.assertFalse(r['correct']);self.assertEqual(measured.call_count,16);self.assertEqual(sample.call_count,2);self.assertEqual([r['request_id'] for r in r['requests']],list(range(16)))
    def test_batch_deadline_and_governor_drift(self):
        w=self.make_worker();task=dict(operation='batch',request_ids=[0],deadline=0,fixture_id='x',arm={})
        with patch('efficiency.reliability_worker.sample',return_value=dict(governor='performance')):
            with self.assertRaises(TimeoutError):w.perform(task)
        task['deadline']=float('inf')
        with patch('efficiency.reliability_worker.sample',side_effect=[dict(governor='performance'),dict(governor='ondemand')]),patch('efficiency.reliability_worker.Numeric.measure',return_value=dict(correct=True,matches_warmup=True,validation_errors=0)):
            with self.assertRaisesRegex(RuntimeError,'drift'):w.perform(task)
    def test_duplicate_or_empty_batch_ids_rejected(self):
        for ids in ([],[0,0]):
            with self.assertRaises(ValueError):self.make_worker().perform(dict(operation='batch',request_ids=ids))
    def test_single_request_preserves_native_result_and_observer_cost(self):
        w=self.make_worker()
        with patch('efficiency.reliability_worker.sample',return_value=dict(governor='performance')),patch('efficiency.reliability_worker.Numeric.measure',return_value=dict(correct=True,request_ms=12.5)):
            result=w.measure('fixture',{})
        self.assertEqual(result['request_ms'],12.5);self.assertGreaterEqual(result['observer_ms'],0)
        self.assertEqual(result['policy'],spec.POLICIES[7])

if __name__=='__main__':unittest.main()
