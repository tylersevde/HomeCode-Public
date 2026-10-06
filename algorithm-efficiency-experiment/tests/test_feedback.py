import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from efficiency.attention_native import Native, build, errors, fixture, numpy_attention, oracle
from efficiency.feedback_spec import (CATALOG, CELLS, SEED, advisor_messages, deterministic_choice,
    parse_proposal, promotion, resolve_route, split_manifest)
from efficiency.feedback_protocol import Budget, BudgetExpired, Measurements, choose, resume_budget
from efficiency.feedback_report import analyze, expected_measurements


class FeedbackRules(unittest.TestCase):
    def test_splits_do_not_overlap(self):
        rows = split_manifest()
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({r['seed'] for r in rows}), 240)
        self.assertEqual(rows, split_manifest())

    def test_catalog_coordinate_search_is_bounded(self):
        history = []; eligible = set(CATALOG)
        while eligible:
            choice = deterministic_choice(history, eligible)
            self.assertIn(choice, eligible); eligible.remove(choice)
            history.append(dict(candidate=choice, development_total_ms=100))
        self.assertIsNone(deterministic_choice(history, eligible))
        self.assertEqual([h['candidate'] for h in history[:3]], ['C0', 'C4', 'C2'])

    def test_proposal_exact_grammar_and_ledger(self):
        parse = lambda value, status='LOGICAL_END_OF_GENERATION', ledger=True: parse_proposal(value, ['<END>'], status, ledger, ['C2'])
        self.assertEqual(parse('C2<END>'), ('C2', None))
        for value in ('C1<END>', 'C2', 'Try C2<END>', 'C2 C2<END>', 'C8<END>'):
            self.assertIsNone(parse(value)[0])
        self.assertIsNone(parse('C2<END>', 'MAX_TOKENS_REACHED')[0])
        self.assertIsNone(parse('C2<END>', ledger=False)[0])

    def test_prompt_excludes_validation_test_and_other_strategy(self):
        history = [dict(candidate='C0', development_total_ms=3, development_cells={'x':[3, True]},
                        validation_secret='NEVER_SEE', test_secret='NEVER_SEE')]
        text = json.dumps(advisor_messages(history, {'x':'native1'}, ['C1']))
        self.assertNotIn('NEVER_SEE', text)
        self.assertNotIn('validation_secret', text)
        self.assertIn('development', text)

    def test_incorrect_or_marginal_results_cannot_promote(self):
        self.assertFalse(promotion([(100, 80)]*3, correct=False)['promote'])
        self.assertFalse(promotion([(100, 98)]*3)['promote'])
        self.assertFalse(promotion([(100, 50), (100, 160), (100, 40)])['promote'])
        self.assertTrue(promotion([(100, 80)]*3)['promote'])
        self.assertFalse(promotion([(100, 0)]*3)['promote'])

    def test_unknown_shapes_always_use_cpu(self):
        self.assertEqual(resolve_route({'routes': {'invented':'C7'}}, 'invented'), 'native1')
        self.assertEqual(resolve_route({'routes': {'stream-n128-b1':'C7'}}, 'stream-n128-b1'), 'C7')
        with self.assertRaises(ValueError): resolve_route({'routes': {'stream-n128-b1':'arbitrary'}}, 'stream-n128-b1')

    def test_replay_never_calls_adviser(self):
        adviser = Mock()
        choice, metadata = choose(adviser, 'hat', [], {}, list(CATALOG), replay_choice='C5')
        self.assertEqual(choice, 'C5'); self.assertFalse(metadata['credited_to_hat'])
        adviser.propose.assert_not_called()
        with self.assertRaises(ValueError): choose(adviser, 'hat', [], {}, ['C0'], replay_choice='C5')

    def test_rejected_advice_uses_logged_fallback_once(self):
        adviser = Mock(); adviser.propose.return_value = dict(choice=None, rejection_reason='bad')
        candidate, metadata = choose(adviser, 'hat', [], {}, list(CATALOG))
        self.assertEqual(candidate, 'C0'); self.assertFalse(metadata['credited_to_hat'])
        self.assertEqual(metadata['origin'], 'fallback'); adviser.propose.assert_called_once()

    def test_deadline_is_not_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'status.json').write_text(json.dumps(dict(monotonic=100, elapsed_seconds=0)))
            clock = Mock(return_value=100)
            budget = Budget(path, {'max_seconds':3600}, clock=clock)
            budget.begin('correctness_baseline')
            clock.return_value=700
            with self.assertRaises(BudgetExpired): budget.check()

    def test_incomplete_report_never_passes(self):
        result = analyze([], {}, {}, [], {}, {}, False)
        self.assertFalse(result['complete'])
        self.assertEqual(result['planned_measurements'], 5220)
        self.assertEqual(len(expected_measurements()), 5220)
        self.assertFalse(result['hat_policy_benefit'])
        self.assertTrue(all(not x['gpu_benefit'] for x in result['acceptance'].values()))

    def test_nonfinite_outputs_are_errors(self):
        self.assertFalse(errors(np.array([np.nan]), np.array([1.0]))['correct'])
        self.assertFalse(errors(np.array([np.inf]), np.array([1.0]))['correct'])


class NativeCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.directory = build()

    def test_reference_edge_shapes_both_modes_and_cpu_threads(self):
        from efficiency.cpu import reference_module
        with Native(self.directory) as native:
            for n, d, b in ((1,1,1), (2,8,4), (17,16,2), (19,65,2)):
                x, w = fixture(n,d,b,9); expected = oracle(x,w)
                original = np.stack([reference_module().full_prefix(s.astype(np.float64), w.astype(np.float64)) for s in x])
                np.testing.assert_allclose(expected, original, atol=1e-10, rtol=1e-10)
                native.configure(x.shape)
                for mode in ('stream', 'prefill'):
                    for backend in ('native1','native4'):
                        out=np.empty_like(x); native.run(x,w,out,mode,backend)
                        self.assertTrue(errors(out,expected)['correct'])
                    self.assertTrue(errors(numpy_attention(x,w,mode),expected)['correct'])

    def test_causality_batch_isolation_and_reset(self):
        with Native(self.directory) as native:
            x,w=fixture(17,16,2,12); native.configure(x.shape)
            for mode in ('stream','prefill'):
                for backend in ('native1','native4'):
                    a=np.empty_like(x); b=np.empty_like(x); c=np.empty_like(x)
                    native.run(x,w,a,mode,backend)
                    changed=x.copy(); changed[0,9:]*=-3; changed[1]*=4
                    native.run(changed,w,b,mode,backend); native.run(x,w,c,mode,backend)
                    np.testing.assert_array_equal(a[0,:9],b[0,:9])
                    np.testing.assert_array_equal(a,c)
                    changed=x.copy(); changed[1]*=2; native.run(changed,w,b,mode,backend)
                    np.testing.assert_array_equal(a[0],b[0])

    def test_budget_and_native_boundary_validation(self):
        with Native(self.directory) as native:
            with self.assertRaises(RuntimeError): native.configure((16,4096,128))
            x,w=fixture(2,8,1,3); native.configure(x.shape)
            with self.assertRaises(ValueError): native.run(x,w,np.empty_like(x),'stream','C0')
            with self.assertRaises(ValueError): native.run(x,w,np.empty_like(x),'future','native1')
            with self.assertRaises(ValueError): native.run(x,w,np.empty_like(x),'stream','arbitrary')


class CheckpointTests(unittest.TestCase):
    def test_resuming_replay_preserves_replay_evidence_label(self):
        from efficiency.cli import main
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'config.json').write_text(json.dumps(dict(profile='feedback-attention',max_seconds=3530,replay_run='/frozen-replay')))
            (p/'outcome.json').write_text(json.dumps(dict(elapsed_seconds=100)))
            with patch('efficiency.attention_native.build'), patch('efficiency.runner.run',return_value=0) as run:
                self.assertEqual(main(['improve','--source-run','/prior','--resume-run',temp,'--output',str(p/'new')]),0)
                args=run.call_args.args[0]
                self.assertEqual(args.replay_run,Path('/frozen-replay'))
                self.assertEqual(args.max_seconds,3430)

    def test_continuation_deducts_all_prior_segments(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'config.json').write_text(json.dumps(dict(profile='feedback-attention', max_seconds=3530)))
            (p/'outcome.json').write_text(json.dumps(dict(elapsed_seconds=1594.5)))
            budget=resume_budget(p)
            self.assertEqual(budget['remaining_seconds'],1935.5)
            (p/'config.json').write_text(json.dumps(dict(profile='feedback-attention',max_seconds=1935.5,resume_budget=budget)))
            (p/'outcome.json').write_text(json.dumps(dict(elapsed_seconds=400)))
            self.assertEqual(resume_budget(p)['remaining_seconds'],1535.5)
            (p/'outcome.json').write_text(json.dumps(dict(elapsed_seconds=1935.5)))
            with self.assertRaises(ValueError): resume_budget(p)

    def checkpoint(self, p, missing=False):
        cell=CELLS[0]; rows=[]
        for seed in range(3):
            for repeat in range(3):
                for arm, backend in (('incumbent','native1'),('candidate','C0')):
                    rows.append(dict(event='measurement',stage='development',strategy='hat',round=1,
                        cell_id=cell['cell_id'],seed_index=seed,repeat=repeat,arm=arm,backend=backend,
                        fixture_id=f'development-{cell["cell_id"]}-s{seed}',request_ms=1,correct=True))
        if missing: rows.pop()
        (p/'feedback.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return cell, rows

    def test_complete_blocks_reuse_every_measurement_without_device_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); cell, rows=self.checkpoint(p)
            m=Measurements(p,Mock(),None,None,{})
            m.data=Mock(side_effect=AssertionError('Must not load completed fixtures'))
            m.request=Mock(side_effect=AssertionError('Must not repeat hardware work'))
            result=m.block('development',cell,'development',3,dict(incumbent='native1',candidate='C0'),'hat',1)
            self.assertEqual({m.key(r) for r in result},{m.key(r) for r in rows})
            self.assertEqual(len(result),18); m.request.assert_not_called(); self.assertEqual(m.serial,1)

    def test_partial_block_runs_only_missing_arm_after_explicit_warmups(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); cell, rows=self.checkpoint(p,missing=True)
            fid=f'development-{cell["cell_id"]}-s2'
            m=Measurements(p,Mock(),None,None,{fid:dict(input_sha256='frozen')})
            m.data=Mock(return_value=(None,None,None))
            m.request=Mock(return_value=dict(correct=True,request_ms=1))
            result=m.block('development',cell,'development',3,dict(incumbent='native1',candidate='C0'),'hat',1)
            self.assertEqual(len(result),18); self.assertEqual(len({m.key(r) for r in result}),18)
            self.assertEqual(m.request.call_count,3)  # two explicit warmups, one missing request
            m.data.assert_called_once_with(fid)

    def test_changed_checkpoint_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp); cell, rows=self.checkpoint(p)
            m=Measurements(p,Mock(),None,None,{})
            with self.assertRaises(ValueError):
                m.block('development',cell,'development',3,dict(incumbent='native4',candidate='C0'),'hat',1)


if __name__ == '__main__': unittest.main()
