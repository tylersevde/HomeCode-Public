import hashlib
import json
from copy import deepcopy
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json,digest_file,digest_value
from efficiency.coordination_workers import owner
from efficiency.diagnostic import renderer,score_answer
from efficiency.quality_audit import audit,independent_score
from efficiency.quality_campaign import allowance
from efficiency.quality_protocol import ENVIRONMENT_KEYS
from efficiency.quality_report import analyze
from efficiency.quality_spec import (CONDITIONS,PROMPTS,MODEL_SHA,aggregate,bootstrap_difference,build_fixture,
    conditions,decide,fixture_specs,messages,schedule,seed_for,specification)
from efficiency.quality_worker import QualityContext
from efficiency.research_campaign import recover,register_fixture
from efficiency.research_report import seal
from efficiency.research_workers import Context


TEMPLATE='{% for m in messages %}{{m.role}}:{{m.content}}<END>\n{% endfor %}assistant:'


def tokens(text):
    return [int.from_bytes(hashlib.sha256(t.encode()).digest()[:4],'little')
            for t in re.findall(r'<END>|[a-zA-Z]+|[0-9]+|[^\w\s]',text)]


class FakeModel:
    def __init__(self):self.text='';self.calls=[];self.truncate=False
    def tokenize(self,text):return tokens(text)
    def clear_context(self):self.text=''
    def get_context_usage_size(self):return len(tokens(self.text))
    def get_stop_tokens(self):return ['<END>']
    def get_generation_recovery_sequence(self):return ''
    def generate(self,prompt,**kwargs):
        self.calls.append(dict(kwargs));self.text+=prompt
        table=dict(re.findall(r'(item[0-9]+) = ([a-z]+)',prompt))
        item=re.findall(r'What color is (item[0-9]+)\?',prompt)[-1]
        # One synthetic format failure exercises factual vs strict auditing.
        answer=table[item]+'<END>'
        if self.truncate:answer='too long'
        model=self
        class Generation:
            generation_status='Status.GENERATING'
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,timeout_ms):
                model.text+=answer
                self.generation_status='Status.MAX_TOKENS_REACHED' if model.truncate else 'Status.LOGICAL_END_OF_GENERATION'
                return answer
        return Generation()


def context():
    c=QualityContext.__new__(QualityContext);c.owner=owner();c.llm=FakeModel();c.limit=1792
    c.parameters=dict(temperature=.7,top_p=.8,top_k=20,frequency_penalty=1.0,max_generated_tokens=16,do_sample=False,seed=12345)
    c.render=renderer(TEMPLATE);c.stops=['<END>'];c.model_sha=MODEL_SHA
    c.environment=dict(hailort_version='5.1.1',model_sha256=MODEL_SHA,parameters=dict(c.parameters),
                       model_defaults=dict(c.parameters),stop_tokens=c.stops,prompt_template=TEMPLATE,
                       capacity_tokens=2048,experiment_limit_tokens=1792,recovery_token_ids=[])
    return c


class FixtureTests(unittest.TestCase):
    def test_matrix_common_tables_and_both_prompt_limits(self):
        c=context()
        for spec in fixture_specs('develop','new-campaign'):
            f=build_fixture(c.render,c.llm.tokenize,spec)
            self.assertEqual(f['table'][0][1],spec['first_color'])
            self.assertTrue(all(len(v)<=spec['target_tokens'] for v in f['initial_token_ids'].values()))
            self.assertGreater(max(map(len,f['next_row_token_ids'].values())),spec['target_tokens'])
            for arm in conditions('develop'):
                self.assertEqual(messages(f['table'],arm['prompt_id'])[1],messages(f['table'],'legacy')[1])
        self.assertEqual(PROMPTS['explicit'],' '.join(PROMPTS['explicit'].split()))

    def test_balanced_first_colors_and_fresh_namespaces(self):
        for stage,count in [('develop',8),('confirm',16)]:
            specs=fixture_specs(stage,'campaign')
            for size in (128,512,1024):
                colors=[f['first_color'] for f in specs if f['section']=='main' and f['target_tokens']==size]
                self.assertEqual(len(colors),count)
                self.assertEqual(len(set(colors)),8)
        self.assertNotEqual(seed_for('one','develop','f'),seed_for('one','confirm','f'))
        self.assertNotEqual(seed_for('one','develop','f'),seed_for('two','develop','f'))

    def test_too_small_target_stops_setup(self):
        with self.assertRaises(ValueError):build_fixture(renderer(TEMPLATE),tokens,dict(seed=1,first_color='blue',target_tokens=1))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.c=context();self.f=build_fixture(self.c.render,tokens,fixture_specs('develop','x')[0])

    def test_conditions_isolate_parameters_and_preserve_legacy(self):
        original=dict(self.c.parameters)
        for arm in conditions('develop'):
            r=self.c.perform(dict(operation='quality_dialogue',condition=arm,fixture=self.f,deadline=time.monotonic()+10))
            self.assertEqual(len(r['rows']),4)
            self.assertTrue(all(v['valid'] for v in r['rows']))
            self.assertTrue(all(v['parameters']['max_generated_tokens']==arm['max_generated_tokens'] for v in r['rows']))
            self.assertEqual(self.c.parameters,original)
        self.assertEqual(self.c.llm.get_context_usage_size(),0)
        self.assertIs(QualityContext.dialogue,Context.dialogue)

    def test_truncation_quarantines_followups_and_resets(self):
        self.c.llm.truncate=True
        r=self.c.perform(dict(operation='quality_dialogue',condition=conditions('develop')[0],fixture=self.f,deadline=time.monotonic()+10))
        self.assertEqual(len(r['rows']),1);self.assertEqual(r['quarantined_turns'],[1,2,3])
        self.assertFalse(r['rows'][0]['valid']);self.assertEqual(self.c.llm.get_context_usage_size(),0)

    def test_overflow_uses_selected_allowance_before_generation(self):
        arm=conditions('develop')[-1]
        self.c.limit=len(self.f['initial_token_ids']['explicit'])+63
        r=self.c.perform(dict(operation='quality_dialogue',condition=arm,fixture=self.f,deadline=time.monotonic()+10))
        self.assertEqual(self.c.llm.calls,[]);self.assertEqual(r['quarantined_turns'],[0,1,2,3])
        self.assertTrue(r['contract_failures']);self.assertEqual(self.c.parameters['max_generated_tokens'],16)

    def test_deadline_resets_and_restores_parameters(self):
        with self.assertRaises(TimeoutError):
            self.c.perform(dict(operation='quality_dialogue',condition=conditions('develop')[-1],fixture=self.f,deadline=0))
        self.assertEqual(self.c.llm.get_context_usage_size(),0)
        self.assertEqual(self.c.parameters['max_generated_tokens'],16)


def simple_events(stage,arms,correct=True):
    result=[]
    for f in fixture_specs(stage,'x'):
        if f['section']!='main':continue
        for arm in arms:
            rows=[dict(valid=True,answer_correct=correct,request_ms=1,score=dict(answer_category='correct_fact',format_correct=True)) for _ in range(4)]
            result.append(dict(event='dialogue',section='main',block=f['block'],target_tokens=f['target_tokens'],
                condition=arm,total_ms=10,response=dict(result=dict(rows=rows,contract_failures=[],reset_verified=True))))
    return result


class DecisionTests(unittest.TestCase):
    def test_selection_ties_prefer_low_budget_and_legacy(self):
        arms=conditions('develop');rows=simple_events('develop',arms)
        self.assertEqual(decide(rows,'develop',arms)['selected'],'legacy-16')
        rows[0]['response']['result']['rows'][0]['valid']=False
        self.assertEqual(decide(rows,'develop',arms)['selected'],'explicit-16')

    def test_missing_and_quarantined_answers_stay_in_denominator(self):
        arms=conditions('develop');rows=simple_events('develop',arms)
        rows[0]['response']['result']['rows']=[]
        metric=aggregate(rows,'develop',arms)[0]
        self.assertEqual(metric['planned'],96);self.assertEqual(metric['correct'],92)
        self.assertFalse(metric['qualified'])
        self.assertIsNone(decide(simple_events('develop',arms,False),'develop',arms)['selected'])

    def test_size_floor_and_contract_failure_block_selection(self):
        arms=conditions('develop')[:1];rows=simple_events('develop',arms)
        for r in rows[0]['response']['result']['rows']:r['answer_correct']=False
        self.assertFalse(aggregate(rows,'develop',arms)[0]['qualified'])
        rows=simple_events('develop',arms);rows[0]['response']['result']['contract_failures']=[{}]
        self.assertFalse(aggregate(rows,'develop',arms)[0]['qualified'])

    def test_fresh_confirmation_gain_and_unchanged_control(self):
        arms=conditions('confirm','explicit-32');rows=simple_events('confirm',arms)
        self.assertFalse(decide(rows,'confirm',arms)['accepted'])
        for r in rows:
            if r['condition']['label']=='control':r['response']['result']['rows'][0]['answer_correct']=False
        decision=decide(rows,'confirm',arms)
        self.assertTrue(decision['accepted']);self.assertEqual(decision['interval']['difference'],.25)
        self.assertEqual(decision['interval']['blocks'],16)
        same=conditions('confirm','legacy-16');decision=decide(simple_events('confirm',same),'confirm',same)
        self.assertFalse(decision['accepted']);self.assertTrue(decision['qualified_unchanged_control'])

    def test_independent_score_agrees_on_edge_cases(self):
        for output,status in [('blue<END>','LOGICAL_END_OF_GENERATION'),('Blue.<END>','LOGICAL_END_OF_GENERATION'),
            ('item001 is blue<END>','LOGICAL_END_OF_GENERATION'),('item002 is blue<END>','LOGICAL_END_OF_GENERATION'),
            ('blue red<END>','LOGICAL_END_OF_GENERATION'),('blue','MAX_TOKENS_REACHED'),('blue','LOGICAL_END_OF_GENERATION')]:
            actual=score_answer(output,'blue','item001',['<END>'],status)
            independent=independent_score(output,'blue','item001',['<END>'],status)
            self.assertEqual(independent[:3],(actual['strict_correct'],actual['factual_correct'],actual['format_correct']))


class BudgetTests(unittest.TestCase):
    def test_failed_attempts_consume_stage_and_campaign_budget(self):
        ledger=dict(attempts=[dict(stage='develop',charged_seconds=1800)])
        self.assertEqual(allowance(ledger,'develop',7200),5400)
        ledger['attempts'].append(dict(stage='confirm',charged_seconds=7100))
        with self.assertRaises(ValueError):allowance(ledger,'confirm',7200)
        for cap in (120,7201,float('nan'),float('inf')):
            with self.assertRaises(ValueError):allowance(dict(attempts=[]),'develop',cap)

    def test_interrupted_reservation_charged_and_duplicate_fixture_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);ledger=dict(attempts=[dict(state='running',output=str(p),reserved_seconds=7200)])
            recover(ledger);self.assertEqual(ledger['attempts'][0]['charged_seconds'],7200)
            atomic_json(p/'registry.json',dict(seeds=[],hashes=[],registered=[]))
            register_fixture(p,'develop','f',1,'one')
            with self.assertRaises(ValueError):register_fixture(p,'confirm','g',1,'two')
            with self.assertRaises(ValueError):register_fixture(p,'confirm','g',2,'one')


def make_audit_run(path,stage='develop',selected=None):
    c=context();arms=conditions(stage,selected)
    config=dict(quality_stage=stage,fixture_namespace='audit-test',conditions=arms,campaign_id='test',selected=selected)
    atomic_json(path/'config.json',config);atomic_json(path/'protocol.json',specification(stage))
    atomic_json(path/'manifest.json',dict(config=config,source_sha256={},model_sha256=MODEL_SHA))
    baseline={k:c.environment[k] for k in ENVIRONMENT_KEYS}
    atomic_json(path/'baseline-environment.json',baseline);atomic_json(path/'quality-environment.json',c.environment)
    frozen=dict(config_sha256=digest_file(path/'config.json'),protocol_sha256=digest_file(path/'protocol.json'),
        source_sha256={},baseline_environment_sha256=digest_file(path/'baseline-environment.json'))
    if stage=='confirm':
        atomic_json(path/'development-summary.json',dict(selected=selected))
        frozen['parent']=dict(summary_sha256=digest_file(path/'development-summary.json'))
    atomic_json(path/'freeze.json',frozen)
    atomic_json(path/'prior-fixtures.json',dict(seeds=[],hashes=[]))
    fixtures=[build_fixture(c.render,tokens,s) for s in fixture_specs(stage,'audit-test')]
    jobs=schedule(fixtures,arms);by_id={f['fixture_id']:f for f in fixtures}
    atomic_json(path/'quality-fixtures.json',fixtures);atomic_json(path/'quality-schedule.json',jobs)
    events=[dict(event='worker_ready',owner=c.owner,environment=c.environment),
            dict(event='fixtures_frozen',fixtures_sha256=digest_file(path/'quality-fixtures.json'),schedule_sha256=digest_file(path/'quality-schedule.json')),
            dict(event='pilot_gate',fits=True,multiplier=8 if stage=='develop' else 16,pilot_seconds=1,
                 required_seconds=9.6 if stage=='develop' else 19.2,remaining_work_seconds=7000)]
    for job in jobs:
        result=c.perform(dict(operation='quality_dialogue',condition=job['condition'],fixture=by_id[job['fixture_id']],deadline=time.monotonic()+30))
        events.extend([dict(event='dialogue_start',**job),dict(event='dialogue',**job,total_ms=10,response=dict(result=result,owner=c.owner))])
    events.extend([dict(event='worker_release',alive=False,forced=False),dict(event='measurement_complete')])
    (path/'quality.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    atomic_json(path/'outcome.json',dict(status='complete'))
    atomic_json(path/'summary.json',analyze(path))
    return events


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.path=Path(cls.tmp.name)
        cls.events=make_audit_run(cls.path)
        cls.original=(cls.path/'quality.jsonl').read_text()

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def tearDown(self):
        (self.path/'quality.jsonl').write_text(self.original)
        (self.path/'checksums.json').unlink(missing_ok=True)

    def test_full_synthetic_pipeline_and_sealed_read_only_audit(self):
        result=audit(self.path,False);self.assertTrue(result['passed'],result)
        seal(self.path)
        before={str(p):p.read_bytes() for p in self.path.rglob('*') if p.is_file()}
        self.assertTrue(audit(self.path)['passed'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.path.rglob('*') if p.is_file()})

    def test_altered_prompts_targets_parameters_ledgers_and_scores_rejected(self):
        for field,value in [('effective_prompt','changed'),('expected_answer','violet'),('context_before',123456),
                            ('parameters',dict(max_generated_tokens=128)),('answer_correct',False)]:
            with self.subTest(field=field):
                events=deepcopy(self.events)
                r=next(e for e in events if e['event']=='dialogue')['response']['result']['rows'][0]
                r[field]=value
                (self.path/'quality.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
                # Deliberately do not rely on file checksums to detect semantic tampering.
                result=audit(self.path,False)
                self.assertFalse(result['passed'],result)

    def test_missing_job_cannot_be_hidden_by_rebuilt_summary(self):
        events=deepcopy(self.events)
        index=next(i for i,e in enumerate(events) if e['event']=='dialogue' and e['section']=='main')
        events.pop(index)
        (self.path/'quality.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        self.assertFalse(audit(self.path,False)['passed'])

    def test_confirmation_independent_audit_and_tampered_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);make_audit_run(path,'confirm','explicit-32')
            result=audit(path,False);self.assertTrue(result['passed'],result)
            summary=json.loads((path/'summary.json').read_text())
            self.assertFalse(summary['accepted'])
            summary['interval']['difference']=.75
            atomic_json(path/'summary.json',summary)
            self.assertFalse(audit(path,False)['passed'])


if __name__=='__main__':unittest.main()
