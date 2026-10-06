from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from efficiency.attention_native import Native,build,fixture
from efficiency.common import atomic_json
from efficiency.coordination_workers import owner
from efficiency.study_campaign import allowance,run
from efficiency.study_spec import *
from efficiency.study_worker import ModelContext,renderer
from efficiency.study_protocol import Budget
from efficiency.study_audit import check_decision,check_dialogue,audit
from efficiency.research_campaign import recover
from test_quality import FakeModel,TEMPLATE


def config(track='gpu-stream',stage='develop',selected=None):
    return dict(track=track,study_stage=stage,fixture_namespace='fresh-study',selected=selected)


def fake_context():
    c=ModelContext.__new__(ModelContext);c.owner=owner();c.llm=FakeModel();c.limit=1792;c.model='qwen'
    c.parameters=deepcopy(PARAMETERS);c.render=renderer(TEMPLATE,'qwen');c.stops=['<END>'];c.recovery=[]
    c.environment=dict(prompt_template=TEMPLATE,model_id='qwen',stop_tokens=c.stops,experiment_limit_tokens=1792,recovery_token_ids=[])
    return c


def numeric_events(c,times=None):
    times=times or dict(cpu=1,cpu_duplicate=1,gpu=10,stream32=8,stream64=9)
    return [dict(event='measurement',**j,total_ms=times[j['arm']['arm']],response=dict(result=dict(correct=True,matches_warmup=True)))
            for j in schedule(c,fixtures(c))]+[dict(event='measurement_complete')]


class StudyTests(unittest.TestCase):
    def test_four_stage_cumulative_budget(self):
        rows=[dict(track='gpu-stream',stage='develop',charged_seconds=7000)]
        self.assertEqual(allowance(dict(attempts=rows),'gpu-stream','develop',7200),200)
        self.assertEqual(allowance(dict(attempts=rows),'gpu-stream','confirm',7200),7200)
        rows.append(dict(track='gpu-stream',stage='develop',charged_seconds=200))
        with self.assertRaises(ValueError):allowance(dict(attempts=rows),'gpu-stream','develop',7200)
        for bad in (120,-1,float('nan'),float('inf'),7201):
            with self.assertRaises(ValueError):allowance(dict(attempts=[]),'npu-model','develop',bad)

    def test_recovery_charges_reserved_budget(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=dict(attempts=[dict(state='running',reserved_seconds=7200,output=td)])
            recover(ledger);self.assertEqual(ledger['attempts'][0]['charged_seconds'],7200)

    def test_fresh_balanced_fixtures_and_model_load_order(self):
        c=config('npu-model');fs=fixtures(c);self.assertEqual(len(fs),27)
        for n in (128,512,1024):self.assertEqual({f['first_color'] for f in fs if f['section']=='main' and f['target_tokens']==n},set(COLORS))
        c2=dict(c,fixture_namespace='retry');self.assertFalse({f['seed'] for f in fs}&{f['seed'] for f in fixtures(c2)})
        jobs=schedule(c,fs);self.assertEqual(len(jobs),54)
        for b in range(8):
            block=[j for j in jobs if j['block']==b];self.assertEqual([j['model'] for j in block],(['qwen']*3+['llama']*3) if b%2==0 else (['llama']*3+['qwen']*3))

    def test_common_model_sizing_and_answers(self):
        c=config('npu-model');spec=fixtures(c)[0]
        render=lambda messages:' '.join(m['content'] for m in messages)
        a=sizing(render,lambda text:list(range(len(text.split()))),spec)
        b=sizing(render,lambda text:list(range(len(text.split())*2)),spec)
        f=make_table(spec,min(a['count'],b['count']));self.assertGreaterEqual(len(f['table']),4)
        for multiplier in (1,2):
            evidence=sizing(render,lambda text:list(range(len(text.split())*multiplier)),spec,len(f['table']))
            self.assertLessEqual(len(evidence['token_ids']),128)
        self.assertEqual(f['answers'][0],f['answers'][3])

    def test_gpu_selection_and_independent_tamper_detection(self):
        c=config();events=numeric_events(c);s=analyze(events,c);self.assertEqual(s['selected'],'stream32')
        check_decision(events,c,s)
        s['metrics'][3]['speedup']['interval'][0]=0
        with self.assertRaises(ValueError):check_decision(events,c,s)

    def test_cpu_drift_closes_selection(self):
        c=config();s=analyze(numeric_events(c,dict(cpu=1,cpu_duplicate=1.1,gpu=10,stream32=8,stream64=9)),c)
        self.assertEqual(s['decision'],'unstable_cpu_control');self.assertIsNone(s['selected'])

    def test_per_shape_regression_veto(self):
        c=config();events=numeric_events(c)
        for e in events:
            if e.get('arm',{}).get('arm')=='stream32' and e['cell_id']=='stream-n128-b1':e['total_ms']=10.6
        s=analyze(events,c);self.assertEqual(s['selected'],'stream64')

    def test_equal_candidates_prefer_32_and_incomplete_cannot_promote(self):
        c=config();events=numeric_events(c,dict(cpu=1,cpu_duplicate=1,gpu=10,stream32=8,stream64=8))
        self.assertEqual(analyze(events,c)['selected'],'stream32')
        self.assertIsNone(analyze(events[:-1],c)['selected'])

    def test_bootstrap_confirm(self):
        c=config(stage='confirm',selected='stream32');events=numeric_events(c);s=analyze(events,c)
        self.assertTrue(s['accepted']);check_decision(events,c,s)

    def test_full_rebuild_and_transcript_audit(self):
        c=fake_context();f=make_table(fixtures(config('npu-model'))[0],4)
        result=c.dialogue(f,'rebuild',time.monotonic()+10)
        self.assertEqual(len(result['rows']),4);self.assertTrue(all(r['context_before']==0 for r in result['rows']))
        event=dict(response=dict(result=result),initial_token_ids=result['rows'][0]['prompt_token_ids'])
        check_dialogue(event,f,c.environment)
        event['response']['result']['rows'][1]['context_before']=1
        with self.assertRaises(ValueError):check_dialogue(event,f,c.environment)

    def test_truncation_quarantines_rest(self):
        c=fake_context();c.llm.truncate=True;f=make_table(fixtures(config('npu-model'))[0],4)
        result=c.dialogue(f,'rebuild',time.monotonic()+10)
        self.assertEqual(result['quarantined_turns'],[1,2,3]);self.assertFalse(result['rows'][0]['valid'])
        self.assertEqual(c.llm.get_context_usage_size(),0)

    def test_overflow_quarantines_all_and_resets(self):
        c=fake_context();c.limit=1;f=make_table(fixtures(config('npu-model'))[0],4)
        result=c.dialogue(f,'rebuild',time.monotonic()+10)
        self.assertEqual(result['quarantined_turns'],[0,1,2,3]);self.assertEqual(len(result['contract_failures']),1)

    def test_template_bindings_trim_and_single_bos(self):
        render=renderer('{{bos_token}}{% for m in messages %}{{m.content|trim}}{% endfor %}{{date_string}}','llama')
        text=render([dict(role='assistant',content=' blue ')])
        self.assertEqual(text,'<|begin_of_text|>blue03 Oct 2026')

    def test_parameter_copy_cannot_change_default(self):
        c=fake_context();c.parameters['max_generated_tokens']=1;self.assertEqual(PARAMETERS['max_generated_tokens'],32)

    def test_stop_stale_and_cleanup_deadline(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);now=time.monotonic();atomic_json(p/'status.json',dict(monotonic=now,elapsed_seconds=0))
            budget=Budget(p,dict(max_seconds=200,phase='cpu'));budget.check()
            self.assertEqual(json.loads((p/'progress.json').read_text())['phase'],'cpu')
            budget.work_deadline=now-1
            with self.assertRaises(TimeoutError):budget.check()
            budget.work_deadline=now+80;(p/'STOP').touch()
            with self.assertRaises(RuntimeError):budget.check()
            (p/'STOP').unlink();budget.last_progress=0;atomic_json(p/'status.json',dict(monotonic=now-10))
            with self.assertRaises(RuntimeError):budget.check()

    def test_cpu_and_prefill_new_variants_rejected(self):
        with Native(build()) as native:
            x,w=fixture(3,7,1,1);native.configure(x.shape)
            for variant in (3,4):
                for mode in ('stream','prefill'):
                    with self.assertRaises(ValueError):native.run(x,w,np.empty_like(x),mode,'native1',variant=variant)

    def test_offline_audit_rejects_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as td:self.assertFalse(audit(td)['passed'])

    def test_model_gate_and_all_planned_denominator(self):
        c=config('npu-model');events=[]
        for j in schedule(c,fixtures(c)):
            rows=[dict(valid=True,answer_correct=(j['model']=='llama'),request_ms=1) for _ in range(4)]
            events.append(dict(event='measurement',**j,total_ms=4,response=dict(result=dict(rows=rows,contract_failures=[],reset_verified=True))))
        events.append(dict(event='measurement_complete'));s=analyze(events,c);self.assertEqual(s['selected'],'llama');check_decision(events,c,s)
        for e in events:
            if e.get('model')=='llama' and e['section']=='main':e['response']['result']['rows'].pop();break
        s=analyze(events,c);self.assertIsNone(s['selected']);self.assertEqual(s['metrics'][1]['planned'],96)

    def test_failed_development_blocks_confirmation_without_output(self):
        from types import SimpleNamespace
        from efficiency.research_report import seal
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);campaign=p/'campaign';parent=p/'develop';campaign.mkdir();parent.mkdir()
            atomic_json(parent/'summary.json',dict(selected=None));atomic_json(parent/'validation.json',dict(passed=True));seal(parent)
            atomic_json(campaign/'campaign.json',dict(version=VERSION,attempts=[dict(track='gpu-stream',stage='develop',scientific_complete=True,audit_passed=True,state='complete',charged_seconds=100,output=str(parent))]))
            args=SimpleNamespace(track='gpu-stream',stage='confirm',max_seconds=7200,campaign=campaign,output=p/'confirm')
            with self.assertRaisesRegex(ValueError,'confirmation is closed'):run(args)
            self.assertFalse(args.output.exists())


class NativeTemplateRegressionTests(unittest.TestCase):
    def test_qwen_optional_tool_calls_after_assistant(self):
        template='{% for message in messages %}{% if message.role == "assistant" and message.tool_calls %}tools{% else %}{{ message.content }}{% endif %}{% endfor %}'
        render=renderer(template,'qwen')
        history=[dict(role='user',content='first'),dict(role='assistant',content='blue'),dict(role='user',content='next')]
        self.assertEqual(render(history),'firstbluenext')
        history[1]['tool_calls']=[dict(name='test')]
        self.assertEqual(render(history),'firsttoolsnext')
