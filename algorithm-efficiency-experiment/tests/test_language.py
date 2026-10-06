from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json,digest_value,profile,read_jsonl
from efficiency.hybrid import FactIndex
from efficiency.language import (ask,argument_error,execute,messages,needs_hat,parse_command,parse_cpu,SYSTEM,WARMUPS)
from efficiency.language_protocol import build_corpus,make_schedule,score_result,run
from efficiency.language_report import analyze,build_report,family_bootstrap
from test_hybrid import Model,PARAMS
from test_cache_validation import render,STOPS
from test_experiment import FakeGeneration
from test_state_isolation import DEFAULTS


def fixtures():
    fs=[]
    for size in (128,512,1536):
        for i in range(10):
            table=[[f'item{j:03d}','blue' if j%2 else 'red'] for j in range(1,14)]
            fs.append(dict(fixture_id=f'f-{size}-{i:02d}',target_tokens=size,table=table,table_sha256=digest_value(table)))
    return fs


def index():return FactIndex([dict(item='item001',color='blue'),dict(item='item010',color='red'),dict(item='item002',color='blue')])


def invoke(question,output,engine='auto',model=None):
    return ask(index(),question,engine,model or Model(output),render,PARAMS,STOPS,4000)


def synthetic():
    corpus=build_corpus(fixtures());config=dict(profile('language-validation'),profile='language-validation',source_run='/source')
    schedule=make_schedule(corpus,config);by_id={c['case_id']:c for c in corpus};rows=[]
    for j in schedule['main']:
        c=by_id[j['case_id']]
        for engine in j['engines']:
            command=c['expected_command']
            if engine=='cpu':command=parse_cpu(c['question'])
            r=dict(c,engine=engine,event='response',phase='main',job_id=j['job_id'],repeat=j['repeat'],
                hat_called=needs_hat(c['question'],engine),table_sha256=c['fixture_id'],
                candidate_command=command or dict(operation='ABSTAIN',argument=None),
                operation=command['operation'] if command else None,argument=command['argument'] if command else None,
                status=c['expected_result']['status'] if command else 'abstain',
                answer=c['expected_result']['answer'] if command else None,abstention_reason=None if command else 'abstain',
                request_ms=10 if engine=='hat' or needs_hat(c['question'],engine) else 1)
            r.update(score_result(r,c));rows.append(r)
    return rows,corpus,schedule,config


class ParserTests(unittest.TestCase):
    def test_six_anchored_forms_and_normalization(self):
        pairs=[(' What COLOR is ITEM001 ?! ','LOOKUP','item001'),('Look up item001.','LOOKUP','item001'),
            ('How many items are BLUE?','COUNT','blue'),('Count blue items.','COUNT','blue'),
            ('Which items are blue?','LIST','blue'),('List blue items.','LIST','blue')]
        for q,op,arg in pairs:
            with self.subTest(q=q):self.assertEqual(parse_cpu(q),dict(operation=op,argument=arg))

    def test_no_partial_match_and_exact_identifier_digits(self):
        for q in ('Which items are not blue?','Count blue or red items.','Count blue items and delete item001.',
                  'What weight is item001?','List purple items.','What color is item001 and item002?'):
            self.assertIsNone(parse_cpu(q))
        self.assertEqual(parse_cpu('Look up item0001')['argument'],'item0001')

    def test_cpu_routes_work_without_model_or_renderer(self):
        for engine in ('auto','cpu'):
            r=ask(index(),'How many items are blue?',engine)
            self.assertFalse(r['hat_called']);self.assertEqual(r['answer'],2)
            self.assertFalse(needs_hat('How many items are blue?',engine))
        r=ask(index(),'Please tell me something.','cpu')
        self.assertEqual(r['status'],'abstain');self.assertFalse(r['hat_called'])
        self.assertTrue(needs_hat('List blue items','hat'))

    def test_empty_missing_and_sorted_results(self):
        self.assertEqual(execute(index(),dict(operation='LIST',argument='blue'))['answer'],['item001','item002'])
        self.assertEqual(execute(index(),dict(operation='COUNT',argument='pink'))['answer'],0)
        self.assertEqual(execute(index(),dict(operation='LIST',argument='pink'))['answer'],[])
        self.assertEqual(execute(index(),dict(operation='LOOKUP',argument='item999'))['status'],'not_found')
        self.assertEqual(ask(FactIndex([]),'Count blue items','auto')['answer'],0)
        with self.assertRaises(ValueError):execute(index(),dict(operation='DELETE',argument='item001'))

    def test_invalid_question_and_engine_fail(self):
        for q in ('',None,7,'  '):
            with self.assertRaises(ValueError):ask(index(),q)
        with self.assertRaises(ValueError):ask(index(),'List blue items','bad')

    def test_command_grammar_is_strict(self):
        self.assertEqual(parse_command(' COUNT blue<END>',STOPS),(dict(operation='COUNT',argument='blue'),None))
        self.assertEqual(parse_command('ABSTAIN<END>',STOPS),(dict(operation='ABSTAIN',argument=None),None))
        for output in ('count blue<END>','COUNT blue.<END>','```COUNT blue```<END>',
                       'COUNT blue\nLIST blue<END>','DELETE item001<END>','COUNT purple<END>',
                       'LOOKUP item001','COUNT blue<END>extra<END>','<|im_start|>COUNT blue<END>'):
            with self.subTest(output=output):self.assertIsNone(parse_command(output,STOPS)[0])

    def test_argument_grounding_and_ambiguity(self):
        cases=[('List blue items','LIST','red'),('Colors of item001 and item002','LOOKUP','item001'),
               ('List blue and red','LIST','blue'),('Lookup item001 in blue','LOOKUP','item001'),
               ('Lookup item0001','LOOKUP','item001'),('Count blueberry items','COUNT','blue')]
        for q,op,arg in cases:self.assertIsNotNone(argument_error(q,dict(operation=op,argument=arg)))
        self.assertIsNone(argument_error('Please describe ITEM001',dict(operation='LOOKUP',argument='item001')))


class InferenceTests(unittest.TestCase):
    def test_hat_interprets_but_cpu_executes(self):
        r=invoke('Give me the identifiers of items colored blue.','LIST blue')
        self.assertEqual(r['answer'],['item001','item002']);self.assertEqual(r['interpretation_source'],'hat')
        self.assertTrue(r['hat_called']);self.assertEqual(r['context_before'],0)
        self.assertEqual(r['messages'],messages(r['question']))
        self.assertNotIn('item010',r['effective_prompt'])

    def test_bad_outputs_and_explicit_abstention(self):
        for output,reason in [('LIST red','argument_mismatch'),('Here is LIST blue','malformed_command'),('ABSTAIN','model_abstained')]:
            r=invoke('Please name the blue group.',output)
            self.assertEqual(r['status'],'abstain');self.assertIsNone(r['answer']);self.assertEqual(r['abstention_reason'],reason)

    def test_wrong_but_valid_intent_is_not_hidden(self):
        r=invoke('Give me the number of items colored blue.','LIST blue')
        c=dict(supported=True,expected_command=dict(operation='COUNT',argument='blue'),expected_result=dict(status='ok',answer=2))
        scores=score_result(r,c)
        self.assertEqual(r['status'],'ok');self.assertTrue(scores['incorrect_accepted'])
        self.assertFalse(scores['intent_correct']);self.assertFalse(scores['final_correct'])

    def test_malformed_negative_abstention_is_not_raw_model_success(self):
        r=invoke('Delete item001.','Sorry, no')
        s=score_result(r,dict(supported=False,expected_command=None,expected_result=dict(status='abstain',answer=None)))
        self.assertTrue(s['final_correct']);self.assertFalse(s['raw_intent_correct'])

    def test_context_is_reset_between_independent_questions(self):
        m=Model('COUNT blue');r=invoke('Number of blue items','COUNT blue',model=m)
        m.output='LIST red';s=invoke('Name red items','LIST red',model=m)
        self.assertEqual(m.clears,2);self.assertEqual(s['context_before'],0)
        self.assertEqual(s['answer'],['item010']);self.assertEqual(m.parameters,[PARAMS,PARAMS])

    def test_bad_ledger_and_truncation_abstain_and_recover(self):
        class BadLedger(Model):
            def get_context_usage_size(self):return len(self.text)+(1 if self.text else 0)
        class Truncated(FakeGeneration):
            def read(self,**kwargs):
                value=super().read(**kwargs)
                if self.index==len(self.chunks):self.generation_status='Status.MAX_TOKENS_REACHED'
                return value
        class BadEnd(Model):
            def generate(self,prompt,**kwargs):self.text+=prompt;return Truncated(self,'COUNT blue')
        for model in (BadLedger('COUNT blue'),BadEnd()):
            r=invoke('Number of blue items','COUNT blue',model=model)
            self.assertEqual(r['status'],'abstain');self.assertEqual(model.text,'');self.assertIsNone(r['operation'])

    def test_budget_prevents_generation_and_runtime_failure_propagates(self):
        m=Model('COUNT blue')
        r=ask(index(),'Count of blue','hat',m,render,PARAMS,STOPS,20)
        self.assertFalse(r['hat_called']);self.assertEqual(r['abstention_reason'],'context_budget');self.assertFalse(m.prompts)
        class Failed(Model):
            def generate(self,*args,**kwargs):raise RuntimeError('SDK fault')
        with self.assertRaisesRegex(RuntimeError,'SDK fault'):invoke('Count of blue','',model=Failed())


class ProtocolTests(unittest.TestCase):
    def test_corpus_counts_labels_and_route_counts(self):
        c=build_corpus(fixtures());s=make_schedule(c,profile('language-validation'))
        self.assertEqual(len(c),160);self.assertEqual(len({x['question'] for x in c}),160)
        self.assertEqual(len({x['family_id'] for x in c}),40);self.assertEqual(sum(x['supported'] for x in c),120)
        self.assertFalse(set(WARMUPS)&{x['question'] for x in c})
        self.assertEqual(s['expected_attempts'],960);self.assertEqual(s['expected_main_generations'],592)
        self.assertEqual(s['calls_by_engine'],dict(cpu=0,hat=320,auto=272))
        self.assertEqual(s,make_schedule(c,profile('language-validation')))
        counts=Counter((e,j['engines'].index(e)) for j in s['main'] for e in j['engines'])
        self.assertEqual(set(counts.values()),{106,107})
        self.assertTrue(any(x['expected_result']['answer']==[] for x in c))
        self.assertEqual(sum(x['expected_result']['status']=='not_found' for x in c),10)

    def test_no_evaluation_labels_enter_prompt(self):
        c=build_corpus(fixtures())[0];original=messages(c['question'])
        c['expected_result']['answer']='secret-oracle-value';c['expected_command']['argument']='secret'
        self.assertEqual(original,messages(c['question']));self.assertNotIn('secret',json.dumps(original))

    def test_worker_records_cpu_hat_and_auto_without_duplicate_metadata(self):
        class ContextModel(Model):
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def prompt_template(self):return 'template'
            def get_stop_tokens(self):return STOPS
            def max_context_capacity(self):return 5000
        class Device:
            def __enter__(self):return self
            def __exit__(self,*args):return False
        m=ContextModel('LOOKUP item001')
        modules={'hailo_platform':SimpleNamespace(VDevice=Device,__version__='5.1.1'),
            'hailo_platform.pyhailort.pyhailort':SimpleNamespace(LLM=lambda *args:m)}
        fs=fixtures();case=build_corpus(fs)[0];config=dict(profile('language-validation'),profile='language-validation',model='unused')
        schedule=make_schedule([case],dict(config,language_repeats=1))
        env=dict(hailort_version='5.1.1',prompt_template='template',stop_tokens=STOPS,
            generation_recovery_sequence='<END>',capacity_tokens=5000,model_defaults=DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for name,value in [('source-run/hybrid-environment.json',env),('language-fixtures.json',fs),('corpus.json',[case]),('schedule.json',schedule)]:atomic_json(p/name,value)
            with patch.dict('sys.modules',modules),patch('efficiency.language_protocol.resolve_defaults',return_value=DEFAULTS),patch('efficiency.language_protocol.renderer',return_value=render),patch('efficiency.language_protocol.wait_cool'):
                run(p,config)
            rows,errors=read_jsonl(p/'language_protocol.jsonl')
        self.assertFalse(errors);self.assertEqual(sum(r['event']=='response' and r['phase']=='main' for r in rows),3)
        self.assertEqual(sum(r['event']=='response' and r['hat_called'] for r in rows),7)
        self.assertEqual(rows[-1]['event'],'complete')


class ReportTests(unittest.TestCase):
    def test_complete_coverage_and_family_bootstrap(self):
        rows,c,s,cfg=synthetic();a,p,f=analyze(rows,c,s,cfg,True)
        self.assertTrue(a['useful_routing']);self.assertEqual(len(f),40);self.assertEqual(len(p),320)
        self.assertEqual(a['groups']['cpu']['supported_resolved'],48)
        self.assertEqual(a['groups']['auto']['supported_resolved'],240)
        self.assertAlmostEqual(a['inference']['hat_over_auto'],3200/2768)
        self.assertAlmostEqual(a['inference']['supported_coverage_gain'],.8)

    def test_wrong_accepted_or_incomplete_cannot_pass(self):
        rows,c,s,cfg=synthetic();r=next(r for r in rows if r['engine']=='auto')
        r['incorrect_accepted']=True
        self.assertFalse(analyze(rows,c,s,cfg,True)[0]['useful_routing'])
        self.assertFalse(analyze(rows[:-1],c,s,cfg,True)[0]['coverage_exact'])
        self.assertEqual(analyze(rows+[rows[0]],c,s,cfg,True)[0]['duplicate_responses'],1)

    def test_report_survives_preflight_failure(self):
        for name in ('language-validation','language-query'):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);atomic_json(p/'config.json',dict(profile(name),profile=name,source_run='/source'))
                build_report(p,charts=False)
                self.assertFalse(json.loads((p/'summary.json').read_text())['complete'])
                self.assertTrue((p/'checksums.json').exists())


if __name__=='__main__':unittest.main()
