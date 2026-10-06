import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from efficiency.research_campaign import allowance,recover,register_fixture,fixture_namespace
from efficiency.research_spec import (accepted,balanced_order,checkpoint_equivalent,interval,
    numeric_arms,seed_for,specification,validate_snapshot)
from efficiency.research_report import paired,cache_matches,combined_matches
from efficiency.research_workers import Context
from efficiency.common import atomic_json


class CampaignTests(unittest.TestCase):
    def test_saved_reference_loading_does_not_modify_sealed_directory(self):
        from efficiency.research_audit import load_reference
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reference.py';path.write_text('def cached_stream(x,w): return x\n')
            before={p.name:p.read_bytes() for p in Path(tmp).iterdir()}
            self.assertEqual(load_reference(path).cached_stream(12,0),12)
            self.assertEqual({p.name:p.read_bytes() for p in Path(tmp).iterdir()},before)

    def test_stage_and_campaign_budget_include_failed_attempts(self):
        ledger={'attempts':[dict(stage='profile',charged_seconds=1800),dict(stage='gpu',charged_seconds=600)]}
        self.assertEqual(allowance(ledger,'profile',7200),5400)
        ledger['attempts'] += [dict(stage='elsewhere',charged_seconds=33000)]
        self.assertEqual(allowance(ledger,'gpu',7200),600)
        ledger['attempts'].append(dict(stage='failed',charged_seconds=500))
        with self.assertRaises(ValueError):allowance(ledger,'gpu',7200)

    def test_nonfinite_or_out_of_range_caps_are_rejected(self):
        for value in (float('nan'),float('inf'),-1,120,7201):
            with self.assertRaises(ValueError):allowance({'attempts':[]},'profile',value)

    def test_interrupted_without_outcome_charges_full_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=dict(state='running',output=tmp,reserved_seconds=7200,charged_seconds=0)
            recover({'attempts':[row]})
            self.assertEqual((row['state'],row['charged_seconds'],row['audit_passed']),('interrupted',7200,False))

    def test_recovery_uses_completed_outcome_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            atomic_json(Path(tmp)/'outcome.json',{'elapsed_seconds':123})
            row=dict(state='running',output=tmp,reserved_seconds=7200,charged_seconds=0)
            recover({'attempts':[row]});self.assertEqual(row['charged_seconds'],123)

    def test_seed_and_content_reuse_both_block_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'registry.json';atomic_json(path,dict(seeds=[],hashes=[],registered=[]))
            register_fixture(tmp,'profile','first',12,'abc')
            for seed,value in ((12,'def'),(13,'abc')):
                with self.assertRaises(ValueError):register_fixture(tmp,'confirm','reused',seed,value)
            register_fixture(tmp,'confirm','fresh',14,'xyz')
            self.assertEqual(len(json.loads(path.read_text())['registered']),2)

    def test_seed_domain_separates_stages_and_campaigns(self):
        values={seed_for(c,s,'fixture') for c in ('one','two') for s in ('profile','gpu','confirm')}
        self.assertEqual(len(values),6)

    def test_retry_keeps_budget_but_uses_fresh_fixture_namespace(self):
        self.assertEqual(fixture_namespace('campaign','profile',0),'campaign')
        values={seed_for(fixture_namespace('campaign','profile',attempt),'profile','same-id') for attempt in range(3)}
        self.assertEqual(len(values),3)
        with self.assertRaises(ValueError):fixture_namespace('campaign','profile',-1)

    def test_balancing_visits_all_start_positions(self):
        arms=['a','b','c','d']
        self.assertEqual({balanced_order(arms,i)[0] for i in range(4)},set(arms))
        self.assertEqual(arms,['a','b','c','d'])

    def test_confidence_corrects_three_primary_comparisons(self):
        self.assertAlmostEqual(specification('confirm')['confidence'],1-.05/3)
        self.assertEqual(specification('profile')['confidence'],.95)


class InferenceTests(unittest.TestCase):
    def test_repeated_requests_do_not_inflate_independent_blocks(self):
        rows=[dict(block=b,arm=a,total_ms=value) for b in range(6) for r in range(30)
              for a,value in (('baseline',12),('candidate',10))]
        result=paired(rows,'baseline','candidate')
        self.assertEqual(result['blocks'],6);self.assertAlmostEqual(result['ratio'],1.2)

    def test_shape_regression_blocks_global_win(self):
        result=interval([(12,10)]*8)
        self.assertTrue(accepted(result,True))
        self.assertFalse(accepted(result,True,[.90]))
        self.assertFalse(accepted(result,False))

    def test_incomplete_or_nonfinite_pairs_never_promote(self):
        self.assertIsNone(interval([(1,1)]))
        self.assertIsNone(interval([(1,1),(2,float('nan'))]))
        self.assertFalse(accepted(None,True))


class CacheTests(unittest.TestCase):
    def dialogue(self):
        return [dict(valid=True,prompt_sha256=str(t),output='red',status='LOGICAL_END_OF_GENERATION',context_after=t+10,answer_correct=False) for t in range(4)]

    def test_equivalent_wrong_answers_are_not_claimed_correct(self):
        a,b=self.dialogue(),self.dialogue()
        self.assertTrue(checkpoint_equivalent({'a':a,'b':b}))
        self.assertFalse(any(r['answer_correct'] for r in a))

    def test_output_prefix_ledger_and_completion_all_matter(self):
        for field,value in (('output','blue'),('prompt_sha256','changed'),('context_after',999),('valid',False)):
            a,b=self.dialogue(),self.dialogue();b[2][field]=value
            self.assertFalse(checkpoint_equivalent({'a':a,'b':b}))
        self.assertFalse(checkpoint_equivalent({'a':self.dialogue()[:3],'b':self.dialogue()}))

    def test_corrupted_wrong_model_and_wrong_token_snapshot(self):
        blob=b'context';meta=dict(sha256=hashlib.sha256(blob).hexdigest(),model_sha256='model',context_tokens=10)
        self.assertTrue(validate_snapshot(meta,blob,'model',10))
        for data,model,count in ((b'corrupt','model',10),(blob,'other',10),(blob,'model',11)):
            with self.assertRaises(ValueError):validate_snapshot(meta,data,model,count)

    def test_missing_pair_is_not_equivalent(self):
        rows=[dict(fixture_id='one',repeat=0,arm='rebuild',response={'result':{'rows':self.dialogue()}})]
        self.assertFalse(cache_matches(rows,'rebuild','checkpoint'))

    def test_combined_requires_equal_work_behavior(self):
        response=dict(valid=True,prompt_sha256='p',output='red',status='LOGICAL_END_OF_GENERATION',context_after=10)
        rows=[dict(fixture_id='one',repeat=0,arm=a,correct=True,npu_valid=True,responses={'hat':{'result':dict(response)}}) for a in ('cpu','gpu')]
        self.assertTrue(combined_matches(rows,'gpu','cpu'))
        rows[1]['responses']['hat']['result']['output']='blue'
        self.assertFalse(combined_matches(rows,'gpu','cpu'))


class FakeLLM:
    def __init__(self):self.text='old';self.calls=0;self.corrupt=False;self.token_calls=[]
    def tokenize(self,text):self.token_calls.append(time.monotonic());return list(text.encode())
    def get_context_usage_size(self):return len(self.text.encode())
    def get_generation_recovery_sequence(self):return ''
    def get_stop_tokens(self):return ['<END>']
    def clear_context(self):self.text=''
    def save_context(self):return self.text.encode()
    def load_context(self,blob):self.text=bytes(blob).decode()+('x' if self.corrupt else '')
    def generate(self,prompt,**kwargs):
        self.calls+=1;self.text+=prompt;llm=self
        class Generation:
            generation_status='Status.GENERATING'
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,timeout_ms):
                llm.text+='blue<END>';self.generation_status='Status.LOGICAL_END_OF_GENERATION';return 'blue<END>'
        return Generation()


class ContextRequestTests(unittest.TestCase):
    def context(self):
        c=Context.__new__(Context);c.llm=FakeLLM();c.parameters={'max_generated_tokens':16};c.stops=['<END>']
        c.render=lambda messages:messages[0]['content'];c.limit=1792;c.model_sha='model'
        return c

    def test_checkpoint_roundtrip_records_and_verifies_actual_state(self):
        c=self.context();r=c.request([{'content':'old suffix'}],'old','checkpoint','blue','What color is item001?')
        self.assertTrue(r['valid']);self.assertTrue(r['answer_correct'])
        self.assertEqual(r['checkpoint']['context_tokens'],3);self.assertEqual(r['submitted_prompt'],' suffix')
        self.assertEqual(r['history_token_ids']+r['submitted_token_ids'],r['prompt_token_ids'])
        self.assertLessEqual(max(c.llm.token_calls),r['ended'])

    def test_bad_restore_stops_before_generation(self):
        c=self.context();c.llm.corrupt=True
        with self.assertRaises(ValueError):c.request([{'content':'old suffix'}],'old','checkpoint','blue','What color is item001?')
        self.assertEqual(c.llm.calls,0)

    def test_overflow_stops_before_generation(self):
        c=self.context();c.limit=20
        with self.assertRaises(ValueError):c.request([{'content':'old suffix'}],'old','retain','blue','What color is item001?')
        self.assertEqual(c.llm.calls,0)

    def test_dialogue_contract_failure_is_quarantined_and_reset(self):
        c=self.context();c.limit=10
        f=dict(initial_messages=[{'content':'This context is too long'}],questions=['What color is item001?']*4,answers=['blue']*4)
        result=c.dialogue(f,'retain',time.monotonic()+10)
        self.assertEqual(result['rows'],[]);self.assertEqual(result['quarantined_turns'],[0,1,2,3])
        self.assertEqual(result['contract_failures'][0]['turn'],0)
        self.assertEqual(c.llm.calls,0);self.assertEqual(c.llm.get_context_usage_size(),0)
        self.assertTrue(result['reset_verified'])

    def test_rebuild_and_checkpoint_use_identical_full_prompt(self):
        values=[self.context().request([{'content':'old suffix'}],'old',a,'blue','What color is item001?') for a in ('rebuild','retain','checkpoint')]
        self.assertEqual(len({r['prompt_sha256'] for r in values}),1)
        self.assertEqual(len({r['output'] for r in values}),1)

    def test_invalid_prelude_clears_prior_continuation_and_skips_generation(self):
        from efficiency.coordination_workers import owner
        c=self.context();c.owner=owner();c.followup=('stale',)*4
        fixture=dict(initial_messages=[{'content':'context'}],answers=['blue']*4,questions=['What color is item001?']*4)
        with patch.object(c,'request',return_value=dict(valid=False,output='truncated')):
            prelude=c.perform(dict(operation='prepare',fixture=fixture))
        self.assertFalse(prelude['valid']);self.assertIsNone(c.followup)
        self.assertEqual(c.llm.get_context_usage_size(),0)
        result=c.perform(dict(operation='followup',arm='retain'))
        self.assertTrue(result['skipped']);self.assertFalse(result['generated'])
        self.assertEqual(c.llm.calls,0);self.assertFalse(result['valid'])
        rows=[dict(fixture_id='one',repeat=0,arm=a,correct=True,npu_valid=False,responses={'hat':{'result':result}}) for a in ('cpu','gpu')]
        self.assertFalse(combined_matches(rows,'gpu','cpu'))

    def test_valid_followup_consumes_prepared_state(self):
        from efficiency.coordination_workers import owner
        c=self.context();c.owner=owner();c.followup=([{'content':'old suffix'}],'old','blue','What color is item001?')
        result=c.perform(dict(operation='followup',arm='retain'))
        self.assertTrue(result['valid']);self.assertEqual(c.llm.calls,1)
        self.assertIsNone(c.followup);self.assertEqual(c.llm.get_context_usage_size(),0)


class WorkerTreeTests(unittest.TestCase):
    def test_descendant_memory_and_cpu_include_live_workers(self):
        import psutil
        from efficiency.runner import worker_tree_sample
        parent,child,gone=Mock(),Mock(),Mock()
        parent.children.return_value=[child,gone]
        parent.memory_info.return_value=Mock(rss=100);child.memory_info.return_value=Mock(rss=250)
        parent.cpu_times.return_value=Mock(user=1,system=2);child.cpu_times.return_value=Mock(user=3,system=4)
        gone.memory_info.side_effect=psutil.NoSuchProcess(123)
        with patch('efficiency.runner.psutil.Process',return_value=parent):
            result=worker_tree_sample(Mock(pid=456))
        self.assertEqual(result,dict(worker_tree_rss_bytes=350,worker_tree_cpu_seconds=10,worker_tree_processes=2))


class CombinedTraceTests(unittest.TestCase):
    def trace(self):
        fixture=dict(initial_messages=[dict(role='user',content='item001?')],questions=['item001?','item002?'],answers=['blue','red'])
        environment=dict(prompt_template='{% for m in messages %}{{m.role}}:{{m.content}}<END>{% endfor %}assistant:',stop_tokens=['<END>'])
        initial=dict(effective_prompt='user:item001?<END>assistant:',submitted_prompt='user:item001?<END>assistant:',
            expected_answer='blue',context_before=0,output='blue<END>',valid=True,context_after=10)
        followup=dict(effective_prompt='user:item001?<END>assistant:blue<END>user:item002?<END>assistant:',
            submitted_prompt='user:item002?<END>assistant:',expected_answer='red',context_before=10)
        return fixture,environment,initial,followup

    def test_frozen_task_and_prelude_match_continuation(self):
        from efficiency.research_audit import combined_trace_issues
        f,e,i,r=self.trace()
        self.assertEqual(combined_trace_issues(f,e,i,r,'retain'),[])
        r['submitted_prompt']=r['effective_prompt']
        self.assertEqual(combined_trace_issues(f,e,i,r,'rebuild'),[])

    def test_changed_target_or_history_is_rejected(self):
        from efficiency.research_audit import combined_trace_issues
        for field,value in (('expected_answer','blue'),('context_before',0),('submitted_prompt','different task')):
            f,e,i,r=self.trace();r[field]=value
            self.assertTrue(combined_trace_issues(f,e,i,r,'retain'))


if __name__=='__main__':unittest.main()
