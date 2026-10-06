from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json, digest_file, digest_value, profile, read_jsonl
from efficiency.diagnostic import LOGICAL_END
from efficiency.hybrid import (FactIndex, answer, cpu_answer, generate_checked, index_fixture,
    make_schedule, run_workflow, select_controls, verify_artifacts, WORKFLOWS)
from efficiency.hybrid_report import analyze, build_report
from efficiency.state_isolation import parameter_settings
from efficiency.cache_validation import fresh_fixtures, prior_fixtures
from test_cache_validation import render, STOPS
from test_experiment import FakeLLM, FakeGeneration
from test_state_isolation import DEFAULTS

PARAMS = parameter_settings(DEFAULTS,'penalty_1_0')[0]


class Model(FakeLLM):
    def __init__(self, output='blue.'):
        super().__init__(); self.output=output; self.clears=0; self.parameters=[]
    def clear_context(self):
        self.clears+=1; super().clear_context()
    def generate(self,prompt,**kwargs):
        self.parameters.append(kwargs); self.prompts.append(prompt); self.text+=prompt
        return FakeGeneration(self,self.output)


def call(model, color='blue'):
    return answer(FactIndex([dict(item='item001',color=color)]),'item001',model,render,PARAMS,STOPS,1792)


def fixture():
    table=[['item001','blue'],['item002','red'],['item003','green'],['item004','black']]
    return dict(fixture_id='f0',table_index=0,seed=42,target_tokens=128,table=table,
        initial_messages=[dict(role='user',content='Reference facts: item001 = blue; item002 = red; item003 = green; item004 = black. What color is item001?')],
        questions=['What color is item001?','What color is item003?','What color is item004?','What color is item001?'],
        answers=['blue','green','black','blue'],positions=['beginning','middle','end','beginning'])


def synthetic():
    config=dict(profile('hybrid-validation'),hat_sizes=[128],tables_per_size=2)
    fs=[dict(fixture(),fixture_id=f'f{i}',table_index=i) for i in range(2)]
    schedule=make_schedule(fs,config)
    rows=[]
    for j in schedule['main']:
        for w in WORKFLOWS:
            for t in range(4):
                r=dict(event='response',phase='main',job_id=j['job_id'],fixture_id=j['fixture_id'],
                    target_tokens=128,repeat=j['repeat'],workflow=w,turn=t,table_sha256=j['fixture_id'],
                    queried_item=f'item00{t+1}',expected_answer='blue',answer='blue',answer_source='cpu' if w=='cpu' else 'hat',
                    fallback_reason=None,final_correct=True,status='ok',request_ms={'cpu':1,'hybrid':10,'full_table':20}[w])
                if w!='cpu':r.update(output='blue.<END>',strict_correct=True,factual_correct=True,
                                    token_ledger_valid=True,completion_status=LOGICAL_END)
                rows.append(r)
    return rows,schedule,config


class RetrievalTests(unittest.TestCase):
    def test_records_are_immutable_and_copied(self):
        records=[dict(item='item001',color='blue')]; index=FactIndex(records)
        records[0]['color']='red'
        self.assertEqual(cpu_answer(index,'item001')['answer'],'blue')
        with self.assertRaises(TypeError):index._values['item001']=('red','hash')
        with self.assertRaises(FrozenInstanceError):index.records=()

    def test_rejects_duplicates_even_when_values_match(self):
        for color in ('blue','red'):
            with self.subTest(color=color), self.assertRaisesRegex(ValueError,'Duplicate'):
                FactIndex([dict(item='item001',color='blue'),dict(item='item001',color=color)])

    def test_invalid_records_and_identifiers(self):
        for bad in ({},['bad'],[dict(item='item001',color='purple')],
                    [dict(item='ITEM001',color='blue')],[dict(item='item001',color=['blue'])],
                    [dict(item='item001',color='blue',extra='x')]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):FactIndex(bad)
        index=FactIndex([])
        for item in (None,12,'item001 extra','ITEM001'):
            with self.assertRaises(ValueError):index.lookup(item)

    def test_missing_item_never_accesses_model(self):
        result=answer(FactIndex([]),'item999',None,None,None,None,None)
        self.assertEqual(result['status'],'not_found');self.assertIsNone(result['answer'])

    def test_identifiers_match_exactly(self):
        index=FactIndex([dict(item='item001',color='blue'),dict(item='item01',color='red')])
        self.assertEqual(cpu_answer(index,'item01')['answer'],'red')
        self.assertEqual(cpu_answer(index,'item1')['status'],'not_found')


class AnswerTests(unittest.TestCase):
    def test_correct_answer_and_only_requested_record_are_submitted(self):
        model=Model(' BLUE. ')
        index=FactIndex([dict(item='item001',color='blue'),dict(item='item002',color='red')])
        result=answer(index,'item001',model,render,PARAMS,STOPS,1792)
        self.assertEqual(result['answer'],'blue');self.assertEqual(result['answer_source'],'hat')
        self.assertNotIn('item002',result['effective_prompt'])
        self.assertEqual(result['evidence_sha256'],digest_value(dict(item='item001',color='blue')))
        self.assertEqual(model.parameters,[PARAMS]);self.assertTrue(result['strict_correct'])

    def test_wrong_verbose_unsupported_ambiguous_and_control_output_fall_back(self):
        cases=[('red.','wrong_fact'),('The color of item001 is blue.','format_violation'),
               ('purple.','unsupported_or_ambiguous_answer'),('blue or red','unsupported_or_ambiguous_answer'),
               ('<|im_start|>blue','exposed_control_marker'),('item002 is blue','wrong_identifier')]
        for output,reason in cases:
            with self.subTest(output=output):
                r=call(Model(output))
                self.assertEqual(r['answer'],'blue');self.assertEqual(r['answer_source'],'cpu_fallback')
                self.assertEqual(r['fallback_reason'],reason);self.assertFalse(r['strict_correct'])
                self.assertEqual(r['output'],output+'<END>')

    def test_each_query_resets_context(self):
        model=Model();a=call(model);model.output='green.';b=call(model,'green')
        self.assertEqual(model.clears,2);self.assertEqual(a['context_before'],0)
        self.assertEqual(b['context_before'],0);self.assertNotIn('blue',b['effective_prompt'])

    def test_ledger_mismatch_returns_cpu_and_recovers(self):
        class Broken(Model):
            def get_context_usage_size(self):return len(self.text)+(1 if self.text else 0)
        model=Broken();r=call(model)
        self.assertTrue(r['strict_correct']);self.assertFalse(r['token_ledger_valid'])
        self.assertEqual(r['answer_source'],'cpu_fallback');self.assertEqual(r['fallback_reason'],'invalid_context_ledger')
        self.assertEqual(model.text,'');self.assertEqual(model.clears,2)

    def test_truncation_returns_cpu_and_recovers(self):
        class Truncated(FakeGeneration):
            def read(self,**kwargs):
                value=super().read(**kwargs)
                if self.index==len(self.chunks):self.generation_status='Status.MAX_TOKENS_REACHED'
                return value
        class ModelTruncated(Model):
            def generate(self,prompt,**kwargs):self.text+=prompt;return Truncated(self,'blue.')
        model=ModelTruncated();r=call(model)
        self.assertEqual(r['answer_source'],'cpu_fallback');self.assertFalse(r['strict_correct'])
        self.assertEqual(model.text,'')

    def test_failed_reset_and_sdk_fault_propagate(self):
        class BadReset(Model):
            def get_context_usage_size(self):return 1
        with self.assertRaisesRegex(RuntimeError,'empty conversation'):call(BadReset())
        class BadSDK(Model):
            def generate(self,*args,**kwargs):raise RuntimeError('SDK failed')
        with self.assertRaisesRegex(RuntimeError,'SDK failed'):call(BadSDK())

    def test_request_timer_includes_checks_reset_generation_and_validation(self):
        clock=[0.0]
        class Timed(Model):
            def tokenize(self,text):clock[0]+=2;return super().tokenize(text)
            def clear_context(self):clock[0]+=3;super().clear_context()
            def get_context_usage_size(self):clock[0]+=5;return super().get_context_usage_size()
            def generate(self,prompt,**kwargs):clock[0]+=7;return super().generate(prompt,**kwargs)
        from efficiency.hybrid import score_answer as real_score
        def scored(*args):clock[0]+=11;return real_score(*args)
        with patch('time.perf_counter',side_effect=lambda:clock[0]),patch('efficiency.hybrid.score_answer',side_effect=scored):
            r=call(Timed())
        self.assertEqual(r['request_ms'],clock[0]*1000)
        self.assertGreater(r['request_ms'],r['generation_request_ms']+11000)
        self.assertEqual(r['validation_ms'],11000)

    def test_hybrid_ignores_benchmark_answer_labels(self):
        f=fixture();f['answers']=['red']*4
        job=dict(make_schedule([f],dict(profile('hybrid-validation'),hat_sizes=[128],tables_per_size=1))['main'][0])
        with tempfile.TemporaryDirectory() as tmp:
            rs=run_workflow(Model(),render,f,index_fixture(f),job,'hybrid',Path(tmp),PARAMS,STOPS,1792)
        self.assertEqual([r['answer'] for r in rs],['blue','green','black','blue'])
        self.assertTrue(all(not r['final_correct'] for r in rs))

    def test_full_table_bad_ledger_skips_only_dependent_turns(self):
        class Broken(Model):
            def get_context_usage_size(self):return len(self.text)+(1 if self.text else 0)
        f=fixture();job=make_schedule([f],dict(profile('hybrid-validation'),hat_sizes=[128]))['main'][0]
        with tempfile.TemporaryDirectory() as tmp:
            rs=run_workflow(Broken(),render,f,index_fixture(f),job,'full_table',Path(tmp),PARAMS,STOPS,1792)
            rows,_=read_jsonl(Path(tmp)/'hybrid.jsonl')
        self.assertEqual(len(rs),1);self.assertEqual(sum(r['event']=='turn_skipped' for r in rows),3)


class ProtocolTests(unittest.TestCase):
    def test_fresh_fixture_seeds_schedule_size_and_order_balance(self):
        config=profile('hybrid-validation')
        fs=fresh_fixtures(render,lambda s:s.split(),config,dict(seeds=[],table_sha256=[]))
        self.assertEqual(len(fs),30);self.assertEqual(len({f['table_sha256'] for f in fs}),30)
        self.assertTrue(all(f['seed']==2026100301+f['target_tokens']*1000+f['table_index'] for f in fs))
        schedule=make_schedule(fs,config)
        self.assertEqual(schedule,make_schedule(fs,config));self.assertEqual(len(schedule['main']),60)
        self.assertEqual(schedule['expected_main_answers'],720);self.assertEqual(schedule['expected_main_generations'],480)
        self.assertEqual(schedule['expected_control_generations'],20)
        count=Counter((j['target_tokens'],w,j['workflows'].index(w)) for j in schedule['main'] for w in WORKFLOWS)
        self.assertEqual(set(count.values()),{6,7})

    def test_novelty_inventory_includes_hybrid_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);atomic_json(p/'x/hybrid-fixtures.json',[fixture()])
            prior=prior_fixtures(p)
        self.assertEqual(prior['seeds'],[42]);self.assertEqual(len(prior['table_sha256']),1)

    def test_checksum_tampering_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'record').write_text('original')
            atomic_json(p/'checksums.json',{'record':digest_file(p/'record')})
            self.assertEqual(len(verify_artifacts(p)),1)
            (p/'record').write_text('changed')
            with self.assertRaisesRegex(ValueError,'checksum'):verify_artifacts(p)
            atomic_json(p/'checksums.json',{'../outside':'x'})
            with self.assertRaisesRegex(ValueError,'checksum'):verify_artifacts(p)

    def test_saved_controls_require_two_agreeing_repetitions(self):
        fs=[dict(fixture(),target_tokens=size,fixture_id=str(size)) for size in (128,512,1536)]
        rs=[dict(event='response',phase='main',fixture_id=f['fixture_id'],preset='penalty_1_0',
            arm='retain',turn=t,repeat=r,output='blue.<END>',token_ledger_valid=True,completion_status=LOGICAL_END)
            for f in fs for t in range(4) for r in range(2)]
        self.assertEqual(len(select_controls(fs,rs)),3)
        rs[0]['output']='red.<END>'
        with self.assertRaisesRegex(ValueError,'agreeing'):select_controls(fs,rs)


class ReportTests(unittest.TestCase):
    def test_complete_coverage_and_independent_table_ratios(self):
        rows,schedule,config=synthetic();s,pairs,tables=analyze(rows,schedule,config,True)
        self.assertTrue(s['pipeline_passed']);self.assertEqual(len(pairs),16);self.assertEqual(len(tables),2)
        self.assertEqual(s['groups'][0]['full_over_hybrid_ci95'],[2.0,2.0])
        self.assertEqual(s['groups'][0]['hybrid_over_cpu_ci95'],[10.0,10.0])

    def test_fallback_is_final_success_but_raw_failure(self):
        rows,schedule,config=synthetic()
        for r in rows:
            if r['workflow']=='hybrid':r.update(answer_source='cpu_fallback',strict_correct=False,factual_correct=False,fallback_reason='wrong_fact')
        s,_,_=analyze(rows,schedule,config,True);q=s['groups'][0]['quality']['hybrid']
        self.assertTrue(s['pipeline_passed']);self.assertEqual(q['raw_strict_correct'],0);self.assertEqual(q['fallbacks'],16)

    def test_missing_duplicate_and_wrong_task_do_not_pass(self):
        rows,schedule,config=synthetic()
        s,_,_=analyze(rows[:-1],schedule,config,True)
        self.assertFalse(s['pipeline_passed']);self.assertEqual(s['groups'][0]['complete_tables'],1)
        s,_,_=analyze(rows+[rows[0]],schedule,config,True)
        self.assertFalse(s['pipeline_passed']);self.assertEqual(s['duplicate_responses'],1)
        rows[0]['table_sha256']='other'
        s,p,_=analyze(rows,schedule,config,True)
        self.assertTrue(any(x['exclusion_reason']=='different_task' for x in p))
        self.assertFalse(s['pipeline_passed'])

    def test_report_preflight_failure_and_query_without_hardware(self):
        for name in ('hybrid-validation','hybrid-query'):
            with self.subTest(profile=name),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);atomic_json(p/'config.json',dict(profile(name),profile=name,source_run='/source'))
                build_report(p,charts=False)
                self.assertFalse(json.loads((p/'summary.json').read_text())['complete'])
                self.assertTrue((p/'report.html').is_file());verify_artifacts(p)


if __name__=='__main__':unittest.main()
