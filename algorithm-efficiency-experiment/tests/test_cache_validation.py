from copy import deepcopy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json, digest_file, digest_value, profile, read_jsonl
from efficiency.cache_validation import (ARMS, PRESETS, EVENT_FILE, fresh_fixtures, make_schedule,
    prior_fixtures, prepare_source, run_session, timed_request, validate_control)
from efficiency.cache_report import analyze, table_ratios, build_report
from efficiency.diagnostic import LOGICAL_END
from efficiency.state_isolation import parameter_settings
from test_experiment import FakeLLM, FakeGeneration
from test_state_isolation import DEFAULTS

STOPS=['<END>']


def render(messages):
    return ''.join(f'{m["role"]}\n{m["content"]}<END>\n' for m in messages)+'assistant\n'


def fixture():
    return dict(fixture_id='fresh-n128-01',seed=1,target_tokens=128,table_index=0,
        initial_messages=[dict(role='user',content='cedar?')],
        questions=['cedar?','maple?','cedar?','cedar?'],answers=['blue','green','blue','blue'],
        positions=['beginning','middle','end','beginning'])


class RecordingLLM(FakeLLM):
    def __init__(self):
        super().__init__();self.parameters=[];self.clears=0
    def clear_context(self):
        self.clears+=1;super().clear_context()
    def generate(self,prompt,**kwargs):
        self.parameters.append(kwargs)
        # Last question only; earlier content may mention maple in rebuilt history.
        self.prompts.append(prompt);self.text+=prompt
        answer='green.' if prompt.rsplit('user\n',1)[-1].startswith('maple?') else 'blue.'
        return FakeGeneration(self,answer)


def small_config():
    return dict(profile('cache-validation'),hat_sizes=[128],tables_per_size=2,source_run='/saved/source')


def synthetic():
    config=small_config()
    fixtures=[dict(fixture(),fixture_id=f'f{i}',table_index=i) for i in range(2)]
    schedule=make_schedule(fixtures,config)
    rows=[]
    for job in schedule['main']:
        for arm in ARMS:
            for turn in range(4):
                rows.append(dict(event='response',phase='main',job_id=job['job_id'],fixture_id=job['fixture_id'],
                    target_tokens=128,preset=job['preset'],repeat=job['repeat'],arm=arm,
                    arm_order=job['arms'].index(arm),turn=turn,effective_input_sha256=f'{job["fixture_id"]}-{turn}',
                    effective_parameters=parameter_settings(DEFAULTS,job['preset'])[0],token_ledger_valid=True,
                    completion_status=LOGICAL_END,output='blue.<END>',strict_correct=True,factual_correct=True,
                    format_correct=True,answer_category='correct_fact',request_ms=20 if arm=='rebuild' else 10,
                    first_visible_ms=10 if arm=='rebuild' else 5))
    return rows,schedule,config


class InputTests(unittest.TestCase):
    def test_fresh_seeds_and_tables_and_exact_schedule(self):
        config=profile('cache-validation')
        fixtures=fresh_fixtures(render,lambda s:s.split(),config,dict(seeds=[],table_sha256=[]))
        self.assertEqual(len(fixtures),30)
        self.assertEqual(len({f['table_sha256'] for f in fixtures}),30)
        self.assertEqual(fixtures,fresh_fixtures(render,lambda s:s.split(),config,dict(seeds=[],table_sha256=[])))
        for f in fixtures:
            self.assertEqual(f['seed'],2026100201+f['target_tokens']*1000+f['table_index'])
        schedule=make_schedule(fixtures,config)
        self.assertEqual(schedule,make_schedule(fixtures,config))
        self.assertEqual(schedule['expected_main_responses'],960)
        self.assertEqual(schedule['expected_validation_responses'],8)
        self.assertEqual(len(schedule['main']),120)
        counts=Counter((j['target_tokens'],j['preset'],j['arms'][0]) for j in schedule['main'])
        self.assertEqual(set(counts.values()),{10})
        for f in fixtures:
            for preset in PRESETS:
                jobs=[j for j in schedule['main'] if j['fixture_id']==f['fixture_id'] and j['preset']==preset]
                self.assertEqual(len(jobs),2)
                self.assertEqual(jobs[0]['arms'],list(reversed(jobs[1]['arms'])))

    def test_prior_seed_or_table_collision_stops_without_reselection(self):
        config=dict(small_config(),tables_per_size=1)
        fixtures=fresh_fixtures(render,lambda s:s.split(),config,dict(seeds=[],table_sha256=[]))
        for prior in (dict(seeds=[fixtures[0]['seed']],table_sha256=[]),
                      dict(seeds=[],table_sha256=[fixtures[0]['table_sha256']])):
            with self.assertRaisesRegex(ValueError,'collision'):
                fresh_fixtures(render,lambda s:s.split(),config,prior)

    def test_prior_inventory_includes_old_diagnostic_and_new_fixtures(self):
        f=dict(seed=123,table=[['item001','red']])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic_json(root/'a/fixtures.json',[f])
            atomic_json(root/'b/diagnostic-inputs.json',dict(fixtures=[f]))
            atomic_json(root/'c/cache-fixtures.json',[dict(f,seed=456)])
            prior=prior_fixtures(root)
            self.assertEqual(prior['seeds'],[123,456])
            self.assertEqual(len(prior['table_sha256']),1)
            self.assertEqual(len(prior['files']),3)

    def test_source_checksum_verification_and_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();out=root/'out';out.mkdir()
            values={'config.json':dict(profile='state-isolation'),
                'manifest.json':dict(model_sha256='model',hailort_cli='5.1.1'),
                'summary.json':dict(complete=True,protocol_completed=True),'state-environment.json':{},
                'state-inputs.json':[],'outcome.json':dict(status='complete')}
            for name,value in values.items():atomic_json(source/name,value)
            (source/'state_isolation.jsonl').write_text('')
            atomic_json(source/'checksums.json',{p.name:digest_file(p) for p in source.iterdir()})
            config=dict(source_run=str(source))
            with patch('efficiency.cache_validation.prior_fixtures',return_value=dict(seeds=[],table_sha256=[],files=[])):
                prepare_source(out,config,values['manifest.json'])
                self.assertEqual(json.loads((out/'source-provenance.json').read_text())['verified_source_artifacts'],7)
                replay=root/'replay';replay.mkdir()
                replay_config=dict(small_config(),profile='cache-validation',source_run=str(source))
                replay_values={'config.json':replay_config,'manifest.json':values['manifest.json'],
                    'cache-fixtures.json':[],'schedule.json':{},'cache-environment.json':{}}
                for name,value in replay_values.items():atomic_json(replay/name,value)
                atomic_json(replay/'checksums.json',{p.name:digest_file(p) for p in replay.iterdir()})
                replay_out=root/'replay-out';replay_out.mkdir()
                prepare_source(replay_out,dict(replay_config,replay_run=str(replay)),values['manifest.json'])
                self.assertFalse(json.loads((replay_out/'replay-provenance.json').read_text())['fresh_evidence'])
                (replay/'cache-fixtures.json').write_text('["modified"]')
                bad=root/'bad';bad.mkdir()
                with self.assertRaisesRegex(ValueError,'Replay checksum mismatch'):
                    prepare_source(bad,dict(replay_config,replay_run=str(replay)),values['manifest.json'])
                (source/'state-inputs.json').write_text('["modified"]')
                with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                    prepare_source(root/'other',config,values['manifest.json'])


class TimingAndSessionTests(unittest.TestCase):
    def test_timer_includes_preparation_clear_reads_cleanup_and_no_diagnostic_rpcs(self):
        clock=[0.0]
        class Completion(FakeGeneration):
            def read(self,**kwargs):
                clock[0]+=5
                return super().read(**kwargs)
            def __exit__(self,*args):clock[0]+=7
        class TimedLLM(RecordingLLM):
            def clear_context(self):clock[0]+=2;super().clear_context()
            def generate(self,prompt,**kwargs):return Completion(self,'blue.')
            def tokenize(self,text):raise AssertionError('Diagnostic RPC inside request')
            def get_context_usage_size(self):raise AssertionError('Diagnostic RPC inside request')
        def prepare():clock[0]+=3;return 'full','submitted'
        with patch('efficiency.cache_validation.time.perf_counter',side_effect=lambda:clock[0]):
            row=timed_request(TimedLLM(),prepare,'rebuild',{},STOPS)
        self.assertEqual(row['request_ms'],22000)
        self.assertEqual(row['first_visible_ms'],10000)
        self.assertEqual(row['prompt_preparation_ms'],3000)
        self.assertEqual(row['clear_context_ms'],2000)
        self.assertEqual(row['generation_ms'],17000)

    def test_settings_apply_to_every_turn_and_both_arms_reconstruct_same_history(self):
        f=fixture();job=make_schedule([f],small_config())['main'][0]
        params=parameter_settings(DEFAULTS,'penalty_1_0')[0]
        with tempfile.TemporaryDirectory() as tmp:
            by_arm={}
            for arm in ARMS:
                llm=RecordingLLM()
                by_arm[arm]=run_session(llm,render,f,job,arm,Path(tmp),params,STOPS,2000)
                self.assertEqual(llm.parameters,[params]*4)
                self.assertEqual(llm.clears,5 if arm=='rebuild' else 1)
                self.assertEqual(len(by_arm[arm]),4)
                self.assertTrue(all(r['strict_correct'] for r in by_arm[arm]))
            self.assertEqual([r['effective_input_sha256'] for r in by_arm['rebuild']],
                             [r['effective_input_sha256'] for r in by_arm['retain']])
            self.assertLess(len(by_arm['retain'][1]['submitted_prompt']),len(by_arm['rebuild'][1]['submitted_prompt']))

    def test_ledger_failure_records_response_skips_dependents_and_resets(self):
        class Broken(RecordingLLM):
            def get_context_usage_size(self):return len(self.text)+(1 if self.text else 0)
        f=fixture();job=make_schedule([f],small_config())['main'][0];llm=Broken()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            result=run_session(llm,render,f,job,'retain',path,{},STOPS,2000)
            rows,_=read_jsonl(path/EVENT_FILE)
            self.assertEqual(len(result),1)
            self.assertFalse(result[0]['token_ledger_valid'])
            self.assertEqual([r['turn'] for r in rows if r['event']=='turn_skipped'],[1,2,3])
            self.assertEqual(llm.text,'')
            healthy=run_session(RecordingLLM(),render,f,job,'rebuild',path,{},STOPS,2000)
            self.assertEqual(len(healthy),4)

    def test_incomplete_generation_quarantines_even_with_valid_count(self):
        class Truncated(FakeGeneration):
            def read(self,**kwargs):
                value=super().read(**kwargs)
                if self.index==len(self.chunks):self.generation_status='Status.MAX_TOKENS_REACHED'
                return value
        class LLM(RecordingLLM):
            def generate(self,prompt,**kwargs):
                self.text+=prompt;return Truncated(self,'blue.')
        f=fixture();job=make_schedule([f],small_config())['main'][0]
        with tempfile.TemporaryDirectory() as tmp:
            result=run_session(LLM(),render,f,job,'retain',Path(tmp),{},STOPS,2000)
            self.assertEqual(len(result),1)
            self.assertTrue(result[0]['token_ledger_valid'])
            self.assertFalse(result[0]['strict_correct'])
            rows,_=read_jsonl(Path(tmp)/EVENT_FILE)
            self.assertEqual(sum(r['event']=='turn_skipped' for r in rows),3)

    def test_unreconstructible_transcript_is_quarantined_before_next_request(self):
        def changing_renderer(messages):
            value=render(messages)
            return value.replace('blue.','BLUE.') if len(messages)>2 else value
        f=fixture();job=make_schedule([f],small_config())['main'][0]
        with tempfile.TemporaryDirectory() as tmp:
            llm=RecordingLLM()
            result=run_session(llm,changing_renderer,f,job,'retain',Path(tmp),{},STOPS,2000)
            self.assertEqual(len(result),1)
            self.assertEqual(llm.text,'')

    def test_timed_preparation_must_match_prevalidation(self):
        f=fixture();job=make_schedule([f],small_config())['main'][0]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('efficiency.cache_validation.prepare_request',side_effect=[('pre','pre'),('post','post')]):
                with self.assertRaisesRegex(RuntimeError,'Timed preparation'):
                    run_session(RecordingLLM(),render,f,job,'rebuild',Path(tmp),{},STOPS,2000)

    def test_failed_reset_and_budget_are_fatal(self):
        f=fixture();job=make_schedule([f],small_config())['main'][0]
        class Stuck(RecordingLLM):
            def get_context_usage_size(self):return 1
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'reset'):
                run_session(Stuck(),render,f,job,'retain',Path(tmp),{},STOPS,2000)
            with self.assertRaisesRegex(RuntimeError,'budget'):
                run_session(RecordingLLM(),render,f,job,'retain',Path(tmp),{},STOPS,1)

    def test_control_gate_requires_all_eight_exact_outputs(self):
        expected={p:{a:['blue.<END>','green.<END>'] for a in ARMS} for p in PRESETS}
        rows=[dict(event='response',phase='validation',preset=p,arm=a,turn=t,output=expected[p][a][t],
            token_ledger_valid=True,completion_status=LOGICAL_END) for p in PRESETS for a in ARMS for t in range(2)]
        self.assertTrue(validate_control(rows,expected)['passed'])
        self.assertFalse(validate_control(rows[:-1],expected)['passed'])
        self.assertFalse(validate_control(rows+[rows[0]],expected)['passed'])
        rows[0]['output']='wrong.<END>'
        self.assertFalse(validate_control(rows,expected)['passed'])


class AnalysisTests(unittest.TestCase):
    def test_complete_correct_data_can_pass_and_incomplete_protocol_cannot(self):
        rows,schedule,config=synthetic()
        s,pairs,tables=analyze(rows,schedule,config,True)
        self.assertTrue(s['coverage_exact'])
        self.assertTrue(all(g['useful_improvement'] for g in s['groups']))
        self.assertEqual(len(tables),4)
        self.assertEqual({t['repeats'] for t in tables},{2})
        self.assertEqual({g['complete_independent_tables'] for g in s['groups']},{2})
        self.assertTrue(all(g['request_speedup_ci95']==[2,2] for g in s['groups']))
        s,_,_=analyze(rows,schedule,config,False)
        self.assertFalse(any(g['useful_improvement'] for g in s['groups']))

    def test_one_wrong_answer_blocks_quality_gate_without_excluding_its_timing(self):
        rows,schedule,config=synthetic();rows[0].update(strict_correct=False,factual_correct=False,answer_category='wrong_fact')
        s,pairs,_=analyze(rows,schedule,config,True)
        group=next(g for g in s['groups'] if g['preset']==rows[0]['preset'])
        self.assertFalse(group['useful_improvement'])
        self.assertTrue(all(p['eligible'] for p in pairs))

    def test_missing_turn_invalidates_its_table_across_both_repetitions(self):
        rows,schedule,config=synthetic();removed=rows.pop(1)
        s,pairs,tables=analyze(rows,schedule,config,True)
        group=next(g for g in s['groups'] if g['preset']==removed['preset'])
        self.assertEqual(group['complete_independent_tables'],1)
        self.assertFalse(group['useful_improvement'])
        self.assertFalse(s['coverage_exact'])
        other=next(g for g in s['groups'] if g['preset']!=removed['preset'])
        self.assertTrue(other['useful_improvement'])

    def test_cluster_ratios_sum_turns_then_combine_repetitions(self):
        jobs=[dict(job_id='a',fixture_id='f'),dict(job_id='b',fixture_id='f')]
        pairs=[]
        for job,multiplier in [('a',2),('b',4)]:
            for turn,retain in [(1,1),(2,2),(3,7)]:
                pairs.append(dict(job_id=job,turn=turn,eligible=True,rebuild_request_ms=retain*multiplier,
                    retain_request_ms=retain,rebuild_first_visible_ms=retain*multiplier,retain_first_visible_ms=retain))
        tables=table_ratios(pairs,jobs,4)
        self.assertEqual(len(tables),1)
        self.assertEqual(tables[0]['request_ratios_by_repeat'],[2,4])
        self.assertEqual(tables[0]['request_ratio'],3)
        pairs[0]['eligible']=False
        self.assertEqual(table_ratios(pairs,jobs,4),[])

    def test_histories_parameters_ledgers_and_duplicates_do_not_pass(self):
        for field,value in [('effective_input_sha256','different'),('effective_parameters',{}),('token_ledger_valid',False)]:
            rows,schedule,config=synthetic();rows[1][field]=value
            s,pairs,_=analyze(rows,schedule,config,True)
            group=next(g for g in s['groups'] if g['preset']==rows[1]['preset'])
            self.assertFalse(group['useful_improvement'])
            self.assertTrue(any(not r['eligible'] for r in pairs))
        rows,schedule,config=synthetic();rows.append(deepcopy(rows[0]))
        s,_,_=analyze(rows,schedule,config,True)
        self.assertEqual(s['duplicate_responses'],1)
        self.assertFalse(s['coverage_exact'])
        self.assertFalse(next(g for g in s['groups'] if g['preset']==rows[0]['preset'])['useful_improvement'])

    def test_empty_report_preserves_failure_and_has_explicit_replay_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);atomic_json(d/'config.json',small_config())
            atomic_json(d/'outcome.json',dict(status='stopped',stop_reason='test preflight failure'))
            build_report(d,charts=False)
            s=json.loads((d/'summary.json').read_text())
            self.assertFalse(s['complete'])
            self.assertFalse(any(g['useful_improvement'] for g in s['groups']))
            self.assertIn('--replay-run',json.loads((d/'reproduce.json').read_text())['command'])
            self.assertIn('test preflight failure',(d/'report.html').read_text())


if __name__=='__main__':unittest.main()
