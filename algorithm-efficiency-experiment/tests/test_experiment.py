import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json, read_jsonl, digest_value
from efficiency.hat import exact_suffix, normalize_answer, run_session, build_fixture, mentioned_colors
from efficiency.monitor import safety_reason
from efficiency.report import compare_hat, bootstrap_median
from efficiency.runner import Supervisor


class BoundaryTests(unittest.TestCase):
    def test_newline_is_part_of_continuation(self):
        prior = 'user\nWhat color?\nassistant\nblue.<END>'
        full = prior+'\nuser\nAnd maple?\nassistant\n'
        suffix, old, new, ids = exact_suffix(full, prior, list)
        self.assertEqual(suffix, '\nuser\nAnd maple?\nassistant\n')
        self.assertEqual(old+new, ids)

    def test_noncompositional_tokenizer_is_rejected(self):
        def tokenizer(text):
            return ['ab'] if text == 'ab' else list(text)
        with self.assertRaisesRegex(ValueError, 'token IDs'):
            exact_suffix('ab', 'a', tokenizer)

    def test_different_transcript_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exact generated transcript'):
            exact_suffix('blue\nnew', 'green', list)

    def test_quality_normalization_does_not_accept_extra_prose(self):
        self.assertEqual(normalize_answer(' Green.<END>', ['<END>']), 'green')
        self.assertNotEqual(normalize_answer('The answer is green.<END>', ['<END>']), 'green')
        self.assertEqual(mentioned_colors('The answer is green.<END>', ['<END>']), ['green'])
        self.assertEqual(mentioned_colors('Green or blue.', []), ['blue','green'])

    def test_fixtures_are_reproducible_and_fit(self):
        render = lambda messages: '\n'.join(m['content'] for m in messages)
        tokenize = lambda text: text.split()
        a = build_fixture(render, tokenize, 128, 42)
        b = build_fixture(render, tokenize, 128, 42)
        self.assertEqual(a, b)
        self.assertLessEqual(a['initial_prompt_tokens'], 128)
        self.assertEqual(a['answers'][0], a['table'][0][1])
        self.assertEqual(a['answers'][2], a['table'][-1][1])


class FakeGeneration:
    def __init__(self, owner, answer):
        self.owner = owner
        self.chunks = [answer, '<END>']
        self.index = 0
        self.generation_status = 'Status.GENERATING'
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, timeout_ms):
        value = self.chunks[self.index]
        self.owner.text += value
        self.index += 1
        if self.index == len(self.chunks):
            self.generation_status = 'Status.LOGICAL_END_OF_GENERATION'
        return value


class FakeLLM:
    def __init__(self):
        self.text = ''
        self.prompts = []
    def clear_context(self): self.text = ''
    def get_context_usage_size(self): return len(self.text)
    def get_generation_recovery_sequence(self): return '<END>'
    def tokenize(self, text): return list(text)
    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.text += prompt
        return FakeGeneration(self, 'green.' if 'maple?' in prompt else 'blue.')


class SessionTests(unittest.TestCase):
    def test_cached_and_replayed_inputs_really_match(self):
        def render(messages):
            return ''.join(f'{m["role"]}\n{m["content"]}<END>\n' for m in messages)+'assistant\n'
        fixture = dict(initial_messages=[dict(role='user',content='cedar?')],
            questions=['cedar?', 'maple?'], answers=['blue','green'], target_tokens=128,
            seed=1, initial_prompt_tokens=27)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            retained = FakeLLM()
            a = run_session(retained, render, fixture, 'retain', 'test', directory, 1000, ['<END>'], 2)
            replayed = FakeLLM()
            b = run_session(replayed, render, fixture, 'rebuild', 'test', directory, 1000, ['<END>'], 2)
            self.assertTrue(all(r['correct'] and r['token_ledger_valid'] for r in a+b))
            self.assertEqual([r['effective_input_sha256'] for r in a], [r['effective_input_sha256'] for r in b])
            self.assertTrue(retained.prompts[1].startswith('\n'))
            self.assertGreater(len(replayed.prompts[1]), len(retained.prompts[1]))

    def test_context_budget_blocks_generation(self):
        fixture = dict(initial_messages=[dict(role='user',content='cedar?')], questions=['cedar?'],
                       answers=['blue'],target_tokens=128,seed=1,initial_prompt_tokens=6)
        model = FakeLLM()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'context budget'):
                run_session(model,lambda messages:'cedar?',fixture,'retain','test',Path(tmp),10,['<END>'],1)
            self.assertEqual(model.prompts, [])


def healthy_row():
    return dict(monotonic=time.monotonic(), cpu_temp_c=50, throttle_flags=0,
                available_memory_bytes=1024**3, sensor_alive=True, hat_sample_age_s=1,
                hat_ts0_c=40, hat_ts1_c=41, hat_consecutive_errors=0, hat_max_c=41)


class SafetyTests(unittest.TestCase):
    def test_normal_and_historical_flags(self):
        row = healthy_row()
        self.assertIsNone(safety_reason(row))
        row['throttle_flags'] = 0x40000
        self.assertIsNone(safety_reason(row, initial_flags=0x40000))
        self.assertIn('historical', safety_reason(row, initial_flags=0))

    def test_each_stop_condition(self):
        cases = [('cpu_temp_c',80), ('throttle_flags',1), ('hat_ts0_c',85),
                 ('hat_sample_age_s',21), ('hat_sample_age_s',-1),
                 ('hat_ts1_c',float('nan')), ('sensor_alive',False),
                 ('hat_consecutive_errors',2), ('available_memory_bytes',1)]
        for key, value in cases:
            with self.subTest(key=key,value=value):
                row = healthy_row()
                row[key] = value
                self.assertIsNotNone(safety_reason(row))

    def test_deadline_and_interrupt_terminate_worker_and_keep_observations(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                config = dict(phase='cpu',model='unused',max_seconds=1.2,
                              baseline_seconds=0,cooldown_seconds=0)
                class TestSupervisor(Supervisor):
                    def launch(self, args, name):
                        self.test_child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
                        if interrupt:
                            self.request_stop(signal.SIGINT, None)
                        return self.test_child
                runner = TestSupervisor(directory,config)
                with patch('efficiency.runner.inventory',return_value={'source_sha256':{}}), \
                     patch('efficiency.runner.sample',side_effect=lambda *args:healthy_row()):
                    code = runner.execute()
                self.assertEqual(code,1)
                self.assertIsNotNone(runner.test_child.poll())
                self.assertEqual(json.loads((directory/'outcome.json').read_text())['status'],'stopped')
                self.assertTrue(read_jsonl(directory/'telemetry.jsonl')[0])


class AnalysisTests(unittest.TestCase):
    def row(self, arm, turn, correct=True, history='same'):
        return dict(event='measurement',pair_id='p1',turn=turn,arm=arm,target_tokens=128,
                    effective_input_sha256=history,token_ledger_valid=True,
                    completion_status='LOGICAL_END_OF_GENERATION',first_visible_ms=100 if arm=='rebuild' else 50,
                    request_ms=200 if arm=='rebuild' else 150,output='blue.<END>',correct=correct,
                    initial_prompt_tokens=123)

    def test_different_histories_excluded_without_hiding_incorrect_answers(self):
        rows = [self.row(a,t) for a in ('rebuild','retain') for t in (0,1)]
        rows[-1].update(effective_input_sha256='different',correct=False)
        summary, comparisons = compare_hat(rows,dict(turns=2,hat_pairs=1))
        self.assertEqual(summary[0]['excluded_turns'],1)
        self.assertEqual(summary[0]['accuracy']['retain']['total'],2)
        self.assertEqual(summary[0]['accuracy']['retain']['correct'],1)
        self.assertFalse(summary[0]['useful_improvement'])

    def test_missing_arm_is_explicit(self):
        _, comparisons = compare_hat([self.row('rebuild',0)],dict(turns=2,hat_pairs=1))
        self.assertEqual(comparisons[0]['exclusion_reason'],'missing_arm')

    def test_bootstrap_requires_independent_pairs(self):
        self.assertIsNone(bootstrap_median([2]))
        self.assertEqual(bootstrap_median([2,2,2]),[2,2])

    def test_truncated_jsonl_retains_complete_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'records.jsonl'
            path.write_text('{"event":"measurement"}\n{"event":')
            rows, errors=read_jsonl(path)
            self.assertEqual(len(rows),1)
            self.assertEqual(errors[0]['line'],2)


if __name__ == '__main__':
    unittest.main()
