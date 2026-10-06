from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficiency.common import digest_value,read_jsonl,atomic_json,stop_process_group
from efficiency.diagnostic import score_answer
from efficiency.state_isolation import (BASELINE,MODES,PRESETS,Quarantine,condition,
    make_schedule,parameter_settings,resolve_defaults,restore_snapshot,save_snapshot,
    select_cases,snapshot_memory_guard,target,validate_capabilities,wait_for_child,run_job)
from efficiency.state_report import analyze,build_report
from test_diagnostic import fixture,RENDER,TEMPLATE,STOPS,ChatFakeLLM
from test_experiment import FakeGeneration

DEFAULTS = dict(temperature=.8,top_p=.9,top_k=40,frequency_penalty=1.1,
                max_generated_tokens=64,do_sample=True,seed=99)


def source_data():
    rows,fixtures=[],[]
    for fid,size in [('n128-pair06',128),('n512-pair03',512),('n1536-pair01',1536)]:
        f=fixture(fid,size); fixtures.append(f)
        messages=deepcopy(f['initial_messages'])
        for turn in range(3):
            if turn:messages.append(dict(role='user',content=f['questions'][turn]))
            full=RENDER(messages)
            for arm in ('rebuild','retain'):
                rows.append(dict(event='measurement',study='position',fixture_id=fid,order_index=0,
                    repeat=0,arm=arm,turn=turn,target_tokens=size,messages=deepcopy(messages),
                    effective_prompt=full,effective_input_sha256=digest_value(list(full)),
                    output=f['answers'][turn]+'.<END>',expected_answer=f['answers'][turn],
                    queried_item=f['questions'][turn].split()[-1].rstrip('?')))
            messages.append(dict(role='assistant',content=f['answers'][turn]+'.'))
    return rows,dict(fixtures=fixtures),dict(prompt_template=TEMPLATE,stop_tokens=STOPS)


def cases():return select_cases(*source_data())


class StateFake(ChatFakeLLM):
    def __init__(self):
        super().__init__()
        self.kwargs=[]
        self._llm=SimpleNamespace(create_generator_params=lambda:SimpleNamespace(**DEFAULTS))
    def generate(self,prompt,**kwargs):
        self.kwargs.append(dict(kwargs))
        self.prompts.append(prompt)
        self.text+=prompt
        answer='red.' if 'What color is item003?' in prompt else 'blue.'
        return FakeGeneration(self,answer)
    def save_context(self):return self.text.encode()
    def load_context(self,blob):self.text=blob.decode()


class InputsAndScheduleTests(unittest.TestCase):
    def test_freezes_actual_prior_answers_and_control_changes_only_query(self):
        c=cases();self.assertEqual(len(c),4)
        self.assertEqual(c[0]['prefix'],c[1]['prefix'])
        self.assertEqual(c[1]['queried_item'],'item002')
        self.assertEqual(c[1]['expected_answer'],'green')
        self.assertEqual(c[0]['messages'][:-1],c[1]['messages'][:-1])
        self.assertEqual(len(c[-1]['conditioning']),2)
        self.assertEqual(c[-1]['conditioning'][-1]['output'],'red.<END>')
        for row in c:self.assertEqual(row['prefix']+row['suffix'],row['effective_prompt'])

    def test_rejects_changed_source_history(self):
        rows,inputs,env=source_data()
        rows[1]['output']='wrong.<END>'
        with self.assertRaisesRegex(ValueError,'histories differ'):select_cases(rows,inputs,env)

    def test_exact_counts_balance_and_dependencies(self):
        s=make_schedule(cases());self.assertEqual(s,make_schedule(cases()))
        self.assertEqual(s['planned'],dict(main=dict(targets=288,conditioning=120),validation=dict(targets=12,conditioning=4)))
        self.assertEqual(len(s['main']),48)
        counts=Counter((j['case_id'],j['preset']) for j in s['main'])
        self.assertEqual(set(counts.values()),{3})
        for job in s['main']+s['validation']:
            modes=(['live'] if job['live_first'] else [])+['save_live']+job['after_snapshot']+job['fresh_order']
            self.assertEqual(Counter(modes),Counter(MODES))
            self.assertLess(modes.index('save_live'),modes.index('restore_same'))
            self.assertLess(modes.index('save_live'),modes.index('restore_fresh'))
        self.assertEqual(sum(j['live_first'] for j in s['main']),24)

    def test_focused_replay_keeps_validation_and_all_settings(self):
        s=make_schedule(cases(),case_id='small_failure')
        self.assertEqual(s['planned']['main'],dict(targets=72,conditioning=24))
        self.assertEqual(len(s['validation']),2)
        self.assertEqual({j['preset'] for j in s['main']},set(PRESETS))


class ParameterTests(unittest.TestCase):
    def test_resolved_values_and_version_guard(self):
        self.assertEqual(resolve_defaults(StateFake(),'5.1.1'),DEFAULTS)
        with self.assertRaisesRegex(RuntimeError,'5.1.1'):resolve_defaults(StateFake(),'other')

    def test_explicit_matches_implicit_and_sensitivity_changes_only_penalty(self):
        args,baseline=parameter_settings(DEFAULTS,'implicit')
        self.assertEqual(args,BASELINE)
        explicit,resolved=parameter_settings(DEFAULTS,'explicit_defaults')
        self.assertEqual(baseline,resolved);self.assertEqual(explicit,resolved)
        for name,value in [('penalty_1_0',1.0),('penalty_1_2',1.2)]:
            _,effective=parameter_settings(DEFAULTS,name)
            self.assertEqual(effective,dict(baseline,frequency_penalty=value))
        self.assertTrue(DEFAULTS['do_sample'])

    def test_target_parameters_never_leak_into_conditioning(self):
        c=cases()[0];job=make_schedule(cases())['main'][0]
        job=dict(job,case_id=c['case_id'],preset='penalty_1_2')
        llm=StateFake()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            condition(llm,d,job,c,'live',DEFAULTS,STOPS,2000)
            self.assertTrue(all(k==BASELINE for k in llm.kwargs))
            result=target(llm,d,job,c,'live',DEFAULTS,STOPS,2000)
            self.assertEqual(llm.kwargs[-1]['frequency_penalty'],1.2)
            self.assertTrue(result['token_ledger_valid'])
            self.assertEqual(result['provided_parameters']['seed'],12345)


class SnapshotTests(unittest.TestCase):
    def test_memory_threshold_is_strict(self):
        required=10*100+1024**3
        with self.assertRaisesRegex(RuntimeError,'available bytes'):snapshot_memory_guard(100,required)
        self.assertEqual(snapshot_memory_guard(100,required+1)['required_available_bytes'],required)

    def test_roundtrip_lineage_and_corruption_rejection(self):
        c=cases()[0];job=make_schedule(cases())['main'][0];job=dict(job,case_id=c['case_id'])
        llm=StateFake();llm.text=c['prefix']
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);snap=save_snapshot(llm,d,job,c,'model')
            llm.text='unrelated state'
            restore_snapshot(llm,d,job,c,snap,'model')
            self.assertEqual(llm.text,c['prefix'])
            with self.assertRaisesRegex(RuntimeError,'lineage'):
                restore_snapshot(llm,d,job,c,snap,'other-model')
            (d/snap['snapshot_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'identity'):
                restore_snapshot(llm,d,job,c,snap,'model')

    def test_save_and_restore_count_mismatches_reset(self):
        c=cases()[0];job=make_schedule(cases())['main'][0];job=dict(job,case_id=c['case_id'])
        llm=StateFake();llm.text=c['prefix']
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);snap=save_snapshot(llm,d,job,c,'model')
            with patch.object(llm,'load_context',side_effect=lambda b:setattr(llm,'text','bad')):
                with self.assertRaisesRegex(Quarantine,'Restoring'):
                    restore_snapshot(llm,d,job,c,snap,'model')
            self.assertEqual(llm.text,'')
            llm.text=c['prefix']
            def mutating_save():
                llm.text+='x';return llm.text.encode()
            with patch.object(llm,'save_context',side_effect=mutating_save):
                with self.assertRaisesRegex(Quarantine,'Saving'):save_snapshot(llm,d,job,c,'model')
            self.assertEqual(llm.text,'')

    def test_conditioning_mismatch_is_recorded_then_quarantined(self):
        c=cases()[0];c['conditioning'][0]['output']='wrong.<END>'
        job=make_schedule(cases())['main'][0];llm=StateFake()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(Quarantine,'frozen history'):
                condition(llm,Path(tmp),job,c,'live',DEFAULTS,STOPS,2000)
            rows,_=read_jsonl(Path(tmp)/'state_isolation.jsonl')
            self.assertFalse(rows[0]['conditioning_matches'])
            self.assertEqual(llm.text,'')


class LifecycleAndReportTests(unittest.TestCase):
    def test_supervisor_group_teardown_reaches_fresh_descendant(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            pidfile=Path(tmp)/'pid'
            script="import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);Path(sys.argv[1]).write_text(str(p.pid));p.wait()"
            parent=subprocess.Popen([sys.executable,'-c',script,str(pidfile)],start_new_session=True)
            try:
                deadline=time.monotonic()+3
                while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(pidfile.exists())
                child_pid=int(pidfile.read_text())
                stop_process_group(parent,.1)
                self.assertIsNotNone(parent.poll())
                try:self.assertEqual(psutil.Process(child_pid).status(),psutil.STATUS_ZOMBIE)
                except psutil.NoSuchProcess:pass
            finally:
                if parent.poll() is None:stop_process_group(parent,.1)

    def test_child_timeout_reaps_child(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        with self.assertRaises(subprocess.TimeoutExpired):wait_for_child(child,.05)
        self.assertIsNotNone(child.poll())

    def test_fresh_conditions_begin_after_model_release(self):
        from contextlib import contextmanager
        active=[]
        @contextmanager
        def fake_model(*args):
            active.append(True)
            yield StateFake(),DEFAULTS,STOPS,2000
            active.pop()
        def fresh(*args):self.assertEqual(active,[])
        c=cases()[0];j=next(j for j in make_schedule(cases())['main'] if j['case_id']==c['case_id'])
        with tempfile.TemporaryDirectory() as tmp, patch('efficiency.state_isolation.open_model',fake_model), \
             patch('efficiency.state_isolation.wait_cool'),patch('efficiency.state_isolation.launch_fresh',side_effect=fresh) as child:
            run_job(Path(tmp),{},j,c,'model')
            self.assertEqual(child.call_count,2)

    def test_incomplete_matrix_is_not_complete_and_capability_requires_coverage(self):
        summary,states,parameters=analyze([],make_schedule(cases()))
        self.assertFalse(summary['target_coverage_exact'])
        self.assertEqual(len(states),288)
        self.assertEqual(len(parameters),216)
        self.assertEqual(summary['state_exclusions'],{'missing_response':288})
        self.assertFalse(validate_capabilities([])['passed'])

    def test_offline_report_handles_preflight_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);atomic_json(d/'config.json',dict(profile='state-isolation'))
            atomic_json(d/'outcome.json',dict(status='stopped',stop_reason='test preflight failure'))
            build_report(d,charts=False)
            summary=json.loads((d/'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertTrue((d/'report.html').exists())

    def test_analysis_separates_parameter_effect_from_serialized_state(self):
        schedule=make_schedule(cases(),case_id='small_failure')
        rows=[]
        for job in schedule['main']:
            for mode in MODES:
                output=('blue.' if job['preset']=='penalty_1_0' or mode in ('rebuild','rebuild_fresh') else 'green.')+'<END>'
                row=dict(event='response',kind='target',phase='main',job_id=job['job_id'],case_id=job['case_id'],
                    preset=job['preset'],repeat=job['repeat'],mode=mode,effective_input_sha256='frozen',
                    effective_parameters=parameter_settings(DEFAULTS,job['preset'])[1],
                    token_ledger_valid=True,completion_status='LOGICAL_END_OF_GENERATION',output=output,request_ms=1)
                row.update(score_answer(output,'blue','item001',STOPS));rows.append(row)
        summary,state,parameters=analyze(rows,schedule)
        self.assertTrue(summary['target_coverage_exact'])
        self.assertFalse(summary['conditioning_coverage_exact'])
        contrasts={(r['left_preset'],r['right_preset']):r for r in summary['parameter_contrasts']}
        self.assertEqual(contrasts[('implicit','explicit_defaults')]['output_differences'],0)
        self.assertEqual(contrasts[('implicit','explicit_defaults')]['equal_parameter_vectors'],18)
        self.assertEqual(contrasts[('explicit_defaults','penalty_1_0')]['output_differences'],12)
        restored=[r for r in summary['state_contrasts'] if r['left_mode']=='restore_same' and r['right_mode']=='restore_fresh']
        self.assertTrue(all(r['output_differences']==0 and r['eligible']==3 for r in restored))


if __name__=='__main__':unittest.main()
