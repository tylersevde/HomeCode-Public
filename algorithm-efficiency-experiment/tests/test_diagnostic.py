from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficiency.common import digest_value, profile, read_jsonl
from efficiency.diagnostic import (FAILURES, LOGICAL_END, ORDERS, check_budget,
    compare_position_pair, make_schedule, position_fixture, position_session,
    reconstruct_messages, renderer, repeat_request, reset_context, score_answer, select_inputs)
from efficiency.diagnostic_report import analyze, quality
from test_experiment import FakeLLM

TEMPLATE = "{% for m in messages %}{{m.role}}\n{{m.content}}<END>\n{% endfor %}{{ 'assistant\\n' }}"
RENDER = renderer(TEMPLATE)
STOPS = ['<END>']


def fixture(pair='n128-pair06', target=128):
    table = [['item001','blue'],['item002','green'],['item003','red'],['item004','black']]
    questions = [f'What color is {key}?' for key in ('item001','item003','item004')]
    messages = [dict(role='system', content='One word.'), dict(role='user',
        content='Reference facts: '+ '; '.join(f'{k} = {v}' for k,v in table)+'. '+questions[0])]
    return dict(pair_id=pair,target_tokens=target,table=table,questions=questions,
        answers=['blue','red','black'],initial_messages=messages,seed=1,
        initial_prompt_tokens=len(RENDER(messages)),selection='test')


def source_data():
    fixtures, rows = [], []
    for pair, failure_turn in FAILURES:
        size = int(pair.split('-')[0][1:])
        control = f'n{size}-pair02' if size == 1536 else f'n{size}-pair01'
        for pair_id in (pair,control):
            f = fixture(pair_id,size)
            fixtures.append(f)
            messages = deepcopy(f['initial_messages'])
            for turn in range(3):
                if turn:
                    messages.append(dict(role='user',content=f['questions'][turn]))
                full = RENDER(messages)
                for arm in ('rebuild','retain'):
                    rows.append(dict(event='measurement',pair_id=pair_id,arm=arm,turn=turn,
                        effective_prompt=full,effective_input_sha256=digest_value(list(full)),
                        output=f['answers'][turn]+'.<END>',correct=True))
                messages.append(dict(role='assistant',content=f['answers'][turn]+'.'))
    return fixtures, rows, dict(prompt_template=TEMPLATE,stop_tokens=STOPS)


class ScoringTests(unittest.TestCase):
    def score(self, text, **kwargs):
        return score_answer(text, 'blue', 'item001', STOPS, **kwargs)

    def test_accepted_complete_forms(self):
        for text in ('blue.', ' Blue! ', 'The color of item001 is blue.', 'item001 is blue',
                     'item001 = blue;', 'It is blue.', 'THE COLOR OF ITEM001 IS\nBLUE.'):
            with self.subTest(text=text):
                self.assertTrue(self.score(text+'<END>')['factual_correct'])

    def test_format_and_fact_are_separate(self):
        r = self.score('It is blue.<END>')
        self.assertTrue(r['factual_correct'])
        self.assertFalse(r['strict_correct'])
        self.assertFalse(r['format_correct'])
        r = self.score('green.<END>')
        self.assertFalse(r['factual_correct'])
        self.assertTrue(r['format_correct'])
        self.assertEqual(r['answer_category'],'wrong_fact')

    def test_rejects_ambiguous_prose_wrong_identifiers_and_markers(self):
        for text in ('blue or green.', 'blue blue.', 'Not blue.', 'The answer is blue.',
                     'item002 = blue.', 'item01 = blue.', 'item0010 is blue.',
                     'item001 = blue. item001 = red.', 'blue.<|im_start|>user',
                     'blue.<END>garbage', 'blue.<END>', '', 'I think it is blue.'):
            with self.subTest(text=text):
                r = self.score(text+'<END>')
                self.assertIsNone(r['factual_correct'])
                self.assertFalse(r['strict_correct'])

    def test_terminator_and_completion_are_required(self):
        self.assertEqual(self.score('blue')['score_reason'],'missing_terminal_token')
        r = self.score('blue.<END>',status='MAX_TOKENS_REACHED')
        self.assertIsNone(r['factual_correct'])
        self.assertTrue(r['legacy_strict_correct'])
        self.assertEqual(r['answer_category'],'truncated')


class FrozenInputTests(unittest.TestCase):
    def test_reconstruction_retains_separator_and_assistant_punctuation(self):
        fixtures, rows, env = source_data()
        inputs = select_inputs(fixtures,rows,env)
        self.assertEqual(len(inputs['cases']),6)
        self.assertEqual(len(inputs['fixtures']),6)
        self.assertEqual([f['pair_id'] for f in inputs['fixtures'] if f['selection']=='first_turn_correct_control'],
                         ['n128-pair01','n512-pair01','n1536-pair02'])
        for case in inputs['cases']:
            self.assertEqual(RENDER(case['messages']),case['effective_prompt'])
            if case['turn']:
                self.assertIn('blue.<END>\nuser\n',case['effective_prompt'])

    def test_changed_original_prompt_is_rejected(self):
        fixtures,rows,env = source_data()
        rows[0]['effective_prompt'] += ' '
        with self.assertRaisesRegex(ValueError,'byte-for-byte'):
            select_inputs(fixtures,rows,env)

    def test_unequal_source_histories_are_rejected(self):
        fixtures,rows,env = source_data()
        rows[1]['effective_input_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError,'unequal'):
            select_inputs(fixtures,rows,env)

    def test_all_cyclic_orders_preserve_table_text(self):
        f = fixture()
        for order in ORDERS:
            result = position_fixture(f,order)
            self.assertEqual(result['messages'][-1]['content'].split('What color')[0],
                             f['initial_messages'][-1]['content'].split('What color')[0])
            self.assertEqual(set(k for k,v in result['items']), {'item001','item003','item004'})
        self.assertEqual(f,fixture())


class ScheduleTests(unittest.TestCase):
    def inputs(self):
        return select_inputs(*source_data())

    def test_full_schedule_counts_and_determinism(self):
        config = dict(profile('diagnostic'),profile='diagnostic')
        a = make_schedule(self.inputs(), config)
        self.assertEqual(a,make_schedule(self.inputs(),config))
        self.assertEqual(len(a['repeatability']),120)
        self.assertEqual(len(a['position']),54)
        self.assertEqual(a['expected_responses'],444)
        repeats = Counter((j['case_id'],j['route']) for j in a['repeatability'])
        self.assertEqual(set(repeats.values()),{10})
        self.assertEqual(len({j['pair_id'] for j in a['position']}),54)

    def test_position_turn_arm_balance(self):
        schedule = make_schedule(self.inputs(),dict(profile('diagnostic'),profile='diagnostic'))
        counts = Counter()
        for job in schedule['position']:
            self.assertEqual(set(job['arms']),{'rebuild','retain'})
            for arm in job['arms']:
                for turn,position in enumerate(job['positions']):
                    counts[(job['fixture_id'],arm,turn,position)] += 1
        self.assertEqual(len(counts),108)
        self.assertEqual(set(counts.values()),{3})

    def test_smoke_exercises_routes_and_all_orders(self):
        s = make_schedule(self.inputs(),dict(profile('diagnostic-smoke'),profile='diagnostic-smoke'))
        self.assertEqual(s['expected_responses'],22)
        self.assertEqual(len(s['repeatability']),4)
        self.assertEqual({j['order_index'] for j in s['position']},{0,1,2})

    def test_replay_is_bounded_to_selected_fixture_order(self):
        config = dict(profile('diagnostic'),profile='diagnostic',
                      replay_fixture='n128-pair06',replay_order=2)
        s = make_schedule(self.inputs(),config)
        self.assertEqual(s['expected_responses'],18)
        self.assertEqual(s['repeatability'],[])
        self.assertTrue(all(j['fixture_id']=='n128-pair06' and j['order_index']==1 for j in s['position']))
        config['replay_fixture'] = 'unknown'
        with self.assertRaisesRegex(ValueError,'Replay fixture'):
            make_schedule(self.inputs(),config)


class ChatFakeLLM(FakeLLM):
    def generate(self, prompt, **kwargs):
        if isinstance(prompt,list):
            prompt = RENDER(prompt)
        return super().generate(prompt,**kwargs)


class DiagnosticSessionTests(unittest.TestCase):
    def test_native_and_raw_reset_and_record_scores(self):
        case = select_inputs(*source_data())['cases'][0]
        llm = ChatFakeLLM()
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for route in ('raw','native_chat'):
                llm.text = 'unrelated stale conversation'
                rows.append(repeat_request(llm,case,dict(route=route,repeat=0,case_id=case['case_id']),
                                           Path(tmp),2000,STOPS))
            self.assertTrue(all(r['context_before']==0 and r['token_ledger_valid'] for r in rows))
            self.assertEqual(rows[0]['output'],rows[1]['output'])
            self.assertTrue(rows[0]['factual_correct'])
            self.assertEqual(len(read_jsonl(Path(tmp)/'diagnostic.jsonl')[0]),2)

    def test_position_histories_and_first_divergence(self):
        job = dict(pair_id='test',repeat=0,order_index=0,positions=list(ORDERS[0]))
        with tempfile.TemporaryDirectory() as tmp:
            arms = {arm:position_session(ChatFakeLLM(),RENDER,fixture(),job,arm,Path(tmp),2000,STOPS)
                    for arm in ('rebuild','retain')}
            comparisons = compare_position_pair(arms,'test')
            self.assertTrue(all(c['eligible'] and c['outputs_equal'] for c in comparisons))
            self.assertTrue(arms['retain'][1]['submitted_prompt'].startswith('\n'))
            self.assertLess(arms['retain'][1]['submitted_input_tokens'],arms['rebuild'][1]['submitted_input_tokens'])
            arms['retain'][1]['output'] = 'red.<END>'
            arms['retain'][2]['effective_input_sha256'] = 'diverged'
            comparisons = compare_position_pair(arms,'test')
            self.assertTrue(comparisons[1]['first_divergence'])
            self.assertFalse(comparisons[2]['first_divergence'])
            self.assertEqual(comparisons[2]['exclusion_reason'],'different_recorded_history')

    def test_budget_prevents_request_and_failed_reset_stops(self):
        llm = ChatFakeLLM()
        case = select_inputs(*source_data())['cases'][0]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'context budget'):
                repeat_request(llm,case,dict(route='raw',repeat=0,case_id=case['case_id']),Path(tmp),10,STOPS)
        self.assertEqual(llm.prompts,[])
        llm.text = 'stale'
        with patch.object(llm,'clear_context'):
            with self.assertRaisesRegex(RuntimeError,'reset'):
                reset_context(llm)

    def test_native_context_mismatch_is_preserved_as_route_evidence(self):
        class Mismatch(ChatFakeLLM):
            def get_context_usage_size(self):
                return super().get_context_usage_size() + (1 if self.text else 0)
        case = select_inputs(*source_data())['cases'][0]
        with tempfile.TemporaryDirectory() as tmp:
            result = repeat_request(Mismatch(),case,dict(route='native_chat',repeat=0,case_id=case['case_id']),
                                    Path(tmp),2000,STOPS)
            self.assertFalse(result['token_ledger_valid'])
        with tempfile.TemporaryDirectory() as tmp:
            llm = Mismatch()
            result = repeat_request(llm,case,dict(route='raw',repeat=0,case_id=case['case_id']),Path(tmp),2000,STOPS)
            self.assertFalse(result['token_ledger_valid'])
            self.assertEqual(llm.get_context_usage_size(),0)
            rows = read_jsonl(Path(tmp)/'diagnostic.jsonl')[0]
            self.assertEqual([r['event'] for r in rows],['measurement','request_quarantined'])

    def test_bad_ledger_quarantines_context_and_allows_independent_arm(self):
        class FaultOnce(ChatFakeLLM):
            mismatches = 1
            def get_context_usage_size(self):
                if self.text and self.mismatches:
                    self.mismatches -= 1
                    return super().get_context_usage_size()+2
                return super().get_context_usage_size()
        llm = FaultOnce()
        job = dict(pair_id='test',repeat=0,order_index=0,positions=list(ORDERS[0]))
        with tempfile.TemporaryDirectory() as tmp:
            failed = position_session(llm,RENDER,fixture(),job,'retain',Path(tmp),2000,STOPS)
            self.assertEqual(len(failed),1)
            self.assertEqual(llm.get_context_usage_size(),0)
            healthy = position_session(llm,RENDER,fixture(),job,'rebuild',Path(tmp),2000,STOPS)
            self.assertEqual(len(healthy),3)
            self.assertTrue(all(r['token_ledger_valid'] for r in healthy))
            events = read_jsonl(Path(tmp)/'diagnostic.jsonl')[0]
            quarantine = next(r for r in events if r['event']=='session_quarantined')
            self.assertEqual(quarantine['skipped_turns'],2)
            self.assertTrue(quarantine['reset_verified'])
            self.assertEqual(failed[0]['output'],''.join(failed[0]['stream_chunks']))


class DiagnosticAnalysisTests(unittest.TestCase):
    def test_partial_run_reports_missing_coverage(self):
        inputs = select_inputs(*source_data())
        schedule = make_schedule(inputs,dict(profile('diagnostic'),profile='diagnostic'))
        summary,comparisons,measurements = analyze([],schedule)
        self.assertEqual(summary['expected_responses'],444)
        self.assertEqual(summary['observed_responses'],0)
        self.assertEqual(summary['position_comparisons']['exclusions'],{'missing_arm':162})
        self.assertTrue(all(r['total']==0 and r['expected']==10 for r in summary['repeatability']))

    def test_malformed_and_wrong_answers_stay_in_denominator(self):
        scores = [score_answer(text,'blue','item001',STOPS) for text in
                  ('blue.<END>','red.<END>','blue or red.<END>')]
        q = quality(scores)
        self.assertEqual((q['total'],q['factual_correct'],q['wrong_fact'],q['unscorable']),(3,1,1,1))


if __name__ == '__main__':
    unittest.main()
