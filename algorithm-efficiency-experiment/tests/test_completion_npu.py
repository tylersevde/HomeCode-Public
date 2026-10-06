from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficiency import completion_npu_spec as spec
from efficiency.completion_npu_audit import check_dialogue, decisions
from efficiency.completion_npu_protocol import pilot_gate
from efficiency.completion_npu_worker import CompletionContext
from efficiency.common import digest_value, read_jsonl
from efficiency.coordination_workers import owner
from efficiency.refine_scoring import score
from efficiency.study_spec import BINDINGS, PARAMETERS
from efficiency.study_worker import renderer


TEMPLATE = '{{ bos_token }}{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}assistant:'


def config(stage='diagnostic'):
    value = dict(profile='completion', track='npu', stage=stage, phase='hat',
                 fixture_namespace='fresh-completion-test', max_seconds=7000, reserved_seconds=7000,
                 development_charged_seconds=0, confirmation_charged_seconds=0)
    if stage != 'diagnostic':
        value['revised_policy'] = dict(version=1, id='reviewed-revision', system='Read every reference fact and answer one color.',
            stop_condition='punctuation', justification='Test-only evidence-based revision.', diagnostic_checksums_sha256='a' * 64)
    return value


def events(config):
    rows = []
    for job in spec.schedule(config, spec.fixtures(config)):
        if job['section'] == 'pilot':
            continue
        answers = [dict(turn=turn, valid=True, answer_correct=job['label'] != 'native') for turn in range(4)]
        rows.append(dict(event='measurement', **job, response=dict(result=dict(rows=answers, contract_failures=[], reset_verified=True))))
    return rows + [dict(event='measurement_complete')]


class DesignTests(unittest.TestCase):
    def test_complete_factor_matrix_has_fixed_punctuation_and_no_candidate(self):
        c = config()
        jobs = spec.schedule(c, spec.fixtures(c))
        main = [j for j in jobs if j['section'] == 'main']
        self.assertEqual(len(main), 8 * 3 * 4)
        self.assertEqual(set(j['condition'] for j in jobs), {'punctuation'})
        self.assertEqual({(j['table_mode'], j['history_mode']) for j in jobs},
            {(a, b) for a in ('full_table', 'selected_fact') for b in ('actual_history', 'independent')})
        for size in (128, 512, 1024):
            for position in range(4):
                labels = [j['label'] for i, j in enumerate(main) if j['target_tokens'] == size and i % 4 == position]
                self.assertEqual(set(Counter(labels).values()), {2})
        self.assertEqual(spec.specification(c)['candidate_count_limit'], 1)

    def test_qualification_control_unchanged_and_paired_order_balanced(self):
        for stage, planned in (('develop', 96), ('confirm', 192)):
            c = config(stage)
            control, candidate = spec.arms(c)
            self.assertEqual((control['table_mode'], control['history_mode'], control['system'], control['condition']),
                             ('full_table', 'actual_history', spec.SYSTEM, 'native'))
            self.assertEqual(candidate['system'], c['revised_policy']['system'])
            jobs = [j for j in spec.schedule(c, spec.fixtures(c)) if j['section'] == 'main']
            self.assertEqual(sum(j['label'] == 'candidate' for j in jobs) * 4, planned)
            for size in (128, 512, 1024):
                orders = [j['label'] for i, j in enumerate(jobs) if j['target_tokens'] == size and i % 2 == 0]
                self.assertEqual(len(set(Counter(orders).values())), 1)

    def test_fresh_stage_and_retry_namespaces(self):
        sets = [{f['seed'] for f in spec.fixtures(config(stage))} for stage in spec.BLOCKS]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        c = config(); c['fixture_namespace'] += '-retry'
        self.assertFalse(sets[0] & {f['seed'] for f in spec.fixtures(c)})

    def test_no_preselected_diagnostic_or_unchanged_policy(self):
        c = config(); c['revised_policy'] = config('develop')['revised_policy']
        with self.assertRaisesRegex(ValueError, 'preselect'):
            spec.specification(c)
        c = config('develop'); c['revised_policy']['system'] = spec.SYSTEM
        with self.assertRaisesRegex(ValueError, 'Unchanged'):
            spec.specification(c)
        c = config('develop'); c.pop('revised_policy')
        with self.assertRaisesRegex(ValueError, 'evidence-informed'):
            spec.specification(c)

    def test_diagnostic_and_develop_share_cap_with_cleanup_reserve(self):
        c = config('develop'); c['development_charged_seconds'] = 201
        with self.assertRaisesRegex(ValueError, 'share'):
            spec.validate_config(c, budget=True)
        c['development_charged_seconds'] = 200
        spec.validate_config(c, budget=True)
        for value in (float('nan'), float('inf'), -1):
            c['development_charged_seconds'] = value
            with self.assertRaises(ValueError):
                spec.validate_config(c, budget=True)
        c = config('confirm'); c['confirmation_charged_seconds'] = 201
        with self.assertRaises(ValueError):
            spec.validate_config(c, budget=True)


class QualificationTests(unittest.TestCase):
    def check(self, c, rows):
        summary = spec.analyze(rows, c)
        decisions(rows[:-1], c, summary)
        return summary

    def test_perfect_simplified_diagnostics_never_qualify(self):
        c = config(); result = self.check(c, events(c))
        self.assertTrue(result['complete'])
        self.assertEqual(result['decision'], 'diagnostic_complete')
        self.assertIsNone(result['selected'])
        self.assertFalse(result['accepted'] or result['scientific_qualification'])
        self.assertTrue(all(not m['qualified'] for m in result['metrics']))

    def test_full_table_development_and_fresh_confirmation(self):
        for stage in ('develop', 'confirm'):
            c = config(stage); result = self.check(c, events(c))
            self.assertEqual(result['selected'], c['revised_policy'])
            self.assertEqual(result['accepted'], stage == 'confirm')

    def test_gain_required_even_for_perfect_candidate(self):
        c = config('confirm'); rows = events(c)
        for event in rows[:-1]:
            for row in event['response']['result']['rows']:
                row['answer_correct'] = True
        self.assertIsNone(self.check(c, rows)['selected'])

    def test_lower_interval_bound_required(self):
        c = config('confirm'); rows = events(c)
        for event in rows[:-1]:
            for row in event['response']['result']['rows']:
                row['answer_correct'] = True
            if event['label'] == 'native' and event['block'] == 0:
                for row in event['response']['result']['rows']:
                    row['answer_correct'] = False
        summary = self.check(c, rows)
        self.assertGreaterEqual(summary['accuracy_gain']['estimate'], .05)
        self.assertEqual(summary['accuracy_gain']['interval'][0], 0)
        self.assertIsNone(summary['selected'])

    def test_one_invalid_contract_or_quarantined_turn_blocks_qualification(self):
        for change in ('invalid', 'contract', 'missing'):
            c = config('develop'); rows = events(c)
            result = next(r for r in rows[:-1] if r['label'] == 'candidate')['response']['result']
            if change == 'invalid': result['rows'][0]['valid'] = False
            elif change == 'contract': result['contract_failures'].append(dict(turn=0))
            else: result['rows'].pop()
            self.assertIsNone(self.check(c, rows)['selected'])

    def test_per_size_threshold_cannot_be_replaced_by_overall(self):
        c = config('develop'); rows = events(c)
        target = next(r for r in rows[:-1] if r['label'] == 'candidate' and r['target_tokens'] == 128)
        for row in target['response']['result']['rows']: row['answer_correct'] = False
        summary = self.check(c, rows)
        self.assertEqual(summary['metrics'][1]['correct'], 92)
        self.assertIsNone(summary['selected'])

    def test_exact_thresholds_are_inclusive(self):
        for stage, errors in (('develop', (3, 1, 0)), ('confirm', (6, 3, 0))):
            c = config(stage); rows = events(c)
            for size, wrong in zip((128, 512, 1024), errors):
                answers = [a for r in rows[:-1] if r['label'] == 'candidate' and r['target_tokens'] == size for a in r['response']['result']['rows']]
                for a in answers[:wrong]: a['answer_correct'] = False
            self.assertIsNotNone(self.check(c, rows)['selected'])

    def test_missing_duplicate_reordered_rows_and_completion_cannot_select(self):
        c = config('develop'); original = events(c)
        samples = [original[1:], original[:-1]]
        rows = deepcopy(original); rows[0] = deepcopy(rows[1]); samples.append(rows)
        rows = deepcopy(original); rows[0], rows[1] = rows[1], rows[0]; samples.append(rows)
        for rows in samples:
            self.assertFalse(spec.analyze(rows, c)['complete'])

    def test_independent_audit_rejects_count_gain_and_selection_tampering(self):
        c = config('confirm'); rows = events(c); original = spec.analyze(rows, c)
        for kind in ('count', 'gain', 'selection', 'qualification'):
            summary = deepcopy(original)
            if kind == 'count': summary['metrics'][1]['correct'] -= 1
            elif kind == 'gain': summary['accuracy_gain']['interval'][0] -= .01
            elif kind == 'selection': summary['selected']['system'] += 'changed'
            else: summary['scientific_qualification'] = False
            with self.assertRaises(ValueError): decisions(rows[:-1], c, summary)


class WorkerTests(unittest.TestCase):
    def make_worker(self):
        worker = CompletionContext.__new__(CompletionContext)
        worker.owner = owner()
        worker.native_stops = list(spec.NATIVE_STOPS)
        worker.stops = list(worker.native_stops)
        worker.reset_calls = 0
        worker.requests = []
        def reset(): worker.reset_calls += 1
        def condition(value): worker.stops = worker.native_stops + (['.', '\n'] if value == 'punctuation' else [])
        def request(messages, accumulated, mode, expected, question):
            worker.requests.append(deepcopy(messages))
            return dict(valid=True, answer_correct=False, assistant_body='wrong-raw-model-answer')
        worker.reset, worker.set_condition, worker.request = reset, condition, request
        return worker

    def task(self, label):
        c = config(); fixture = spec.table(spec.fixtures(c)[0], 4)
        arm = next(a for a in spec.arms(c) if a['label'] == label)
        return dict(operation='completion_dialogue', fixture=fixture, arm=arm, deadline=float('inf'))

    def test_actual_history_preserves_wrong_raw_assistant_answers(self):
        worker = self.make_worker(); result = worker.perform(self.task('full_table_actual_history'))
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(worker.requests[1][2], dict(role='assistant', content='wrong-raw-model-answer'))
        self.assertEqual(len(worker.requests[-1]), 8)

    def test_selected_fact_history_accumulates_real_fact_prompts_and_raw_answers(self):
        worker = self.make_worker(); task = self.task('selected_fact_actual_history')
        result = worker.perform(task)
        self.assertTrue(all(row['selected_fact'] is not None for row in result['rows']))
        self.assertEqual(len(worker.requests[-1]), 8)
        self.assertEqual(worker.requests[1][1], worker.requests[0][1])
        self.assertIn('Reference facts:', worker.requests[1][-1]['content'])

    def test_independent_questions_rebuild_full_table_without_history(self):
        worker = self.make_worker(); task = self.task('full_table_independent')
        worker.perform(task)
        self.assertTrue(all(len(messages) == 2 for messages in worker.requests))
        for messages in worker.requests:
            for item, color in task['fixture']['table']:
                self.assertIn(f'{item} = {color}', messages[-1]['content'])

    def test_invalid_history_quarantines_remainder_but_independent_continues(self):
        for label, observed in (('full_table_actual_history', 1), ('full_table_independent', 4)):
            worker = self.make_worker()
            worker.request = lambda *args: dict(valid=False, answer_correct=False, assistant_body=None)
            result = worker.perform(self.task(label))
            self.assertEqual(len(result['rows']), observed)
            self.assertEqual(result['quarantined_turns'], [1, 2, 3] if observed == 1 else [])
            self.assertEqual(worker.reset_calls, 1)

    def test_contract_failures_keep_four_planned_turns(self):
        for label in ('full_table_actual_history', 'full_table_independent'):
            worker = self.make_worker()
            def request(*args): raise ValueError('context overflow')
            worker.request = request
            result = worker.perform(self.task(label))
            self.assertEqual(result['quarantined_turns'], [0, 1, 2, 3])
            self.assertEqual(len(result['contract_failures']), 1 if label.endswith('actual_history') else 4)

    def test_deadline_and_worker_failure_always_reset(self):
        worker = self.make_worker(); task = self.task('full_table_actual_history'); task['deadline'] = 0
        with self.assertRaises(TimeoutError): worker.perform(task)
        self.assertEqual(worker.reset_calls, 1)
        worker = self.make_worker()
        def request(*args): raise RuntimeError('device failure')
        worker.request = request
        with self.assertRaises(RuntimeError): worker.perform(self.task('full_table_actual_history'))
        self.assertEqual(worker.reset_calls, 1)

    def test_simplified_arm_cannot_claim_qualification(self):
        worker = self.make_worker(); task = self.task('selected_fact_independent')
        task['arm']['diagnostic_only'] = False
        with self.assertRaisesRegex(ValueError, 'cannot qualify'): worker.perform(task)


def raw_worker():
    worker = WorkerTests().make_worker()
    worker.render = renderer(TEMPLATE, 'llama')
    def request(messages, accumulated, mode, expected, question):
        output = expected + ('.' if '.' in worker.stops else worker.native_stops[0])
        full = worker.render(messages)
        token_ids = list(range(len(full) // 4))
        after = list(range(len(full + output) // 4))
        condition = 'punctuation' if '.' in worker.stops else 'native'
        scored = score(output, expected, question.split()[-1][:-1], worker.native_stops, worker.stops, 'LOGICAL_END_OF_GENERATION')
        return dict(valid=True, answer_correct=True, assistant_body=scored['assistant_body'], terminal_suffix=scored['terminal_suffix'],
            native_stops=worker.native_stops, effective_stops=worker.stops, condition=condition, score=scored,
            output=output, stream_chunks=[output], status='LOGICAL_END_OF_GENERATION', messages=deepcopy(messages),
            parameters=deepcopy(PARAMETERS), expected_answer=expected, effective_prompt=full, submitted_prompt=full,
            prompt_sha256=digest_value(full), prompt_tokens=len(token_ids), prompt_token_ids=token_ids,
            submitted_token_ids=token_ids, context_before=0, recovery_token_ids=[], context_after=len(after),
            expected_context_after=len(after), after_token_ids=after, ledger_valid=True,
            output_token_ids=[1], terminal_token_ids=[2], started=1, ended=2, request_ms=1000)
    worker.request = request
    return worker


class DialogueAuditTests(unittest.TestCase):
    def case(self, label):
        worker = raw_worker(); task = WorkerTests().task(label); result = worker.perform(task)
        env = dict(native_stops=spec.NATIVE_STOPS, prompt_template=TEMPLATE, recovery_token_ids=[], experiment_limit_tokens=1792)
        size = dict(token_ids=result['rows'][0]['prompt_token_ids'])
        return result, task['fixture'], task['arm'], env, size

    def test_all_four_factor_transcripts_reconstruct(self):
        for label in spec.DIAGNOSTIC_ARMS:
            check_dialogue(*self.case(label))

    def test_raw_score_history_selected_fact_and_denominator_tamper_rejected(self):
        for kind in ('score', 'history', 'fact', 'quarantine', 'tokens'):
            args = self.case('selected_fact_actual_history'); result = args[0]
            if kind == 'score': result['rows'][0]['answer_correct'] = False
            elif kind == 'history': result['rows'][1]['messages'][2]['content'] = 'CPU corrected answer'
            elif kind == 'fact': result['rows'][0]['selected_fact'][1] = 'invalid'
            elif kind == 'quarantine': result['quarantined_turns'] = [3]
            else: result['rows'][0]['prompt_tokens'] += 1
            with self.assertRaises(ValueError): check_dialogue(*args)

    def test_planned_input_generation_never_uses_expected_answers(self):
        task = WorkerTests().task('selected_fact_actual_history')
        before = spec.planned_messages(task['fixture'], task['arm'], 0, [])
        task['fixture']['answers'] = ['changed'] * 4
        self.assertEqual(before, spec.planned_messages(task['fixture'], task['arm'], 0, []))


class PilotTests(unittest.TestCase):
    def test_projection_uses_full_matrix_margin_and_reserves_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            budget = type('Budget', (), dict(work_deadline=1000))()
            with patch('efficiency.completion_npu_protocol.time.monotonic', return_value=100):
                pilot_gate(directory, config(), budget, 90)
            event = read_jsonl(directory / spec.EVENTS)[0][0]
            self.assertEqual(event['required_seconds'], 96)
            self.assertTrue(event['fits'])
            budget.work_deadline = 195
            with patch('efficiency.completion_npu_protocol.time.monotonic', return_value=100):
                with self.assertRaises(TimeoutError): pilot_gate(directory, config(), budget, 90)


if __name__ == '__main__':
    unittest.main()
