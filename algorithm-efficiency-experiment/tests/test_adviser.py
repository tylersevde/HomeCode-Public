import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from efficiency.adviser_audit import classify, reference_cpu
from efficiency.adviser_interface import cpu_choice, decide, generate_controlled, parse_response, validate_case
from efficiency.adviser_protocol import Experiment
from efficiency.adviser_report import build_report, summarize
from efficiency.adviser_spec import (choose_challenger, development_cases, fresh_holdout,
                                     make_schedule, messages, specification)
from efficiency.coordination_spec import advice_cases
from efficiency.feedback_spec import CATALOG


def case(eligible=('C3',)):
    tested=sorted(set(CATALOG)-set(eligible))
    p=dict(eligible=list(eligible),tested=tested,development_ms={c:20+int(c[1:]) for c in tested},
           configurations={c:CATALOG[c] for c in eligible})
    return dict(case_id='case',payload=p,baseline_messages=[dict(role='system',content='baseline'),
                                                         dict(role='user',content=json.dumps(p))])


def raw(output='C3<END>',**overrides):
    r=dict(output=output,completion_status='LOGICAL_END_OF_GENERATION',token_ledger_valid=True,
           context_before=0,context_after=10,expected_context_after=10)
    r.update(overrides);return r


class ParserTests(unittest.TestCase):
    def test_exact_and_only_one_completed_period(self):
        for output,origin in [('C3<END>','hat_exact'),(' \nC3. \n<END>','hat_canonicalized')]:
            d=decide(case(),raw(output),['<END>'])
            self.assertEqual(d['choice'],'C3');self.assertEqual(d['origin'],origin)

    def test_explanations_multiple_ids_unicode_and_missing_end_rejected(self):
        for output in ('C3. explanation<END>','C3 C3<END>','C3,C4<END>','C3..<END>',
                       'C3。<END>','C3','C3.<END> trailing','<END>C3<END>','c3<END>','C30<END>'):
            with self.subTest(output=output):
                self.assertEqual(decide(case(),raw(output),['<END>'])['origin'],'cpu_fallback')

    def test_incomplete_generation_never_salvaged(self):
        for output in ('C3<END>','C3.<END>','C3'):
            self.assertEqual(decide(case(),raw(output,completion_status='MAX_TOKENS_REACHED'),['<END>'],True)['origin'],'cpu_fallback')

    def test_invalid_ledger_context_before_and_after_rejected(self):
        for change in ({'token_ledger_valid':False},{'context_before':2},{'context_after':11}):
            self.assertEqual(decide(case(),raw(**change),['<END>'])['origin'],'cpu_fallback')

    def test_ineligible_choice_never_escapes_or_gets_hat_credit(self):
        for output in ('C2<END>','C2.<END>','C2'):
            d=decide(case(),raw(output),['<END>'],True)
            self.assertEqual(d['choice'],'C3');self.assertEqual(d['origin'],'cpu_fallback')
            self.assertEqual(d['fallback_reason'],'ineligible_or_repeated_proposal')

    def test_bounded_candidate_retained_but_suppressed_answer_not_inferred(self):
        d=decide(case(),raw('C3'),['<END>'],True)
        self.assertEqual((d['origin'],d['completion_boundary']),('hat_exact','candidate_stop'))
        for output in ('','<END>','C3.'):
            self.assertEqual(decide(case(),raw(output),['<END>'],True)['origin'],'cpu_fallback')

    def test_empty_set_abstains(self):
        d=decide(case(()))
        self.assertEqual((d['origin'],d['choice']),('abstain',None))

    def test_cpu_selector_invariants_across_all_subsets(self):
        ids=list(CATALOG)
        for n in range(9):
            for eligible in itertools.combinations(ids,n):
                c=case(eligible);choice=cpu_choice(c)
                self.assertEqual(choice,reference_cpu(c['payload']))
                self.assertTrue(choice in eligible if eligible else choice is None)

    def test_bad_case_data_rejected(self):
        for value in (float('nan'),float('inf'),-1,True):
            c=case();c['payload']['development_ms']['C0']=value
            with self.assertRaises(ValueError):validate_case(c)
        c=case();c['payload']['eligible'].append('C0')
        with self.assertRaises(ValueError):validate_case(c)


class FakeLLM:
    def __init__(self):self.stops=['<END>'];self.context=3;self.reject_custom=False;self.reject_restore=False;self.custom_seen=False
    def clear_context(self):self.context=0
    def get_context_usage_size(self):return self.context
    def set_stop_tokens(self,value):
        if len(value)>1:
            if self.reject_custom:raise ValueError('custom settings unsupported')
            self.custom_seen=True
        if self.reject_restore and self.custom_seen and value==['<END>']:raise ValueError('restoration failure')
        self.stops=list(value)
    def get_stop_tokens(self):return self.stops


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.llm=FakeLLM()
        self.env=dict(stop_tokens=['<END>'],parameters={'max_generated_tokens':16},experiment_limit_tokens=1792)

    def test_custom_stops_include_ineligible_ids_and_restore(self):
        with patch('efficiency.adviser_interface.generate_checked',return_value=raw('C3')) as generation:
            result=generate_controlled(self.llm,Mock(),self.env,[],True)
        self.assertEqual(result['active_stops'],['<END>',*CATALOG])
        self.assertEqual(self.llm.stops,['<END>']);self.assertTrue(result['restored'] and result['reset_verified'])
        self.assertEqual(generation.call_args.args[6],['<END>'])

    def test_generation_exception_retains_error_and_restores(self):
        with patch('efficiency.adviser_interface.generate_checked',side_effect=RuntimeError('bad ledger')):
            result=generate_controlled(self.llm,Mock(),self.env,[],True)
        self.assertIn('bad ledger',result['raw']['error']);self.assertTrue(result['restored'])
        self.assertEqual(decide(case(),result['raw'],['<END>'],True)['origin'],'cpu_fallback')

    def test_unsupported_custom_settings_fail_closed_and_restore(self):
        self.llm.reject_custom=True
        with patch('efficiency.adviser_interface.generate_checked') as generation:
            result=generate_controlled(self.llm,Mock(),self.env,[],True)
        generation.assert_not_called();self.assertTrue(result['restored']);self.assertIn('unsupported',result['raw']['error'])

    def test_restoration_failure_visible_to_controller(self):
        self.llm.reject_restore=True
        with patch('efficiency.adviser_interface.generate_checked',return_value=raw('C3')):
            result=generate_controlled(self.llm,Mock(),self.env,[],True)
        self.assertFalse(result['restored']);self.assertIn('restoration failure',result['restoration_error'])


class CorpusTests(unittest.TestCase):
    def test_previous_holdout_is_development_and_new_cases_are_disjoint(self):
        dev=development_cases(advice_cases());holdout=fresh_holdout(dev)
        self.assertEqual(len(dev),40);self.assertTrue(all(c['split']=='development' for c in dev))
        self.assertEqual(sum(c['former_split']=='holdout' for c in dev),20)
        self.assertEqual(len(holdout),48)
        self.assertEqual(sum(bool(c['payload']['eligible']) for c in holdout),40)
        self.assertEqual(holdout,fresh_holdout(dev))
        self.assertFalse({json.dumps(c['payload'],sort_keys=True) for c in dev}&{json.dumps(c['payload'],sort_keys=True) for c in holdout})
        for c in holdout:validate_case(c)

    def test_native_and_examples_prompts_identical_with_real_assistant_examples(self):
        c=development_cases(advice_cases())[0]
        self.assertEqual(messages(c,'examples'),messages(c,'bounded'))
        self.assertEqual([m['role'] for m in messages(c,'examples')],['system','user','assistant','user','assistant','user'])
        self.assertEqual(messages(c,'baseline'),c['baseline_messages'])

    def test_schedule_has_all_arms_once_per_case(self):
        cases=fresh_holdout(development_cases(advice_cases()))
        schedule=make_schedule(cases,['baseline','examples','cpu'],42)
        self.assertEqual(len(schedule),48);self.assertEqual(len({j['case_id'] for j in schedule}),48)
        self.assertTrue(all(set(j['arms'])=={'baseline','examples','cpu'} for j in schedule))

    def test_challenger_selection_uses_quality_before_time(self):
        rows=[dict(arm=a,case_id=str(i),total_ms=1 if a=='examples' else 100,
                   decision=dict(origin='hat_exact' if a=='bounded' or i<39 else 'cpu_fallback'))
              for a in ('examples','bounded') for i in range(40)]
        self.assertEqual(choose_challenger(rows,['baseline','examples','bounded'])['challenger'],'bounded')
        with self.assertRaises(ValueError):choose_challenger(rows[:-1],['examples','bounded'])

    def test_exact_count_and_stable_tie_preference(self):
        rows=[dict(arm=a,case_id=str(i),total_ms=10,decision=dict(origin='hat_exact'))
              for a in ('examples','bounded') for i in range(40)]
        self.assertEqual(choose_challenger(rows,['examples','bounded'])['challenger'],'examples')
        rows[0]['decision']['origin']='hat_canonicalized'
        self.assertEqual(choose_challenger(rows,['examples','bounded'])['challenger'],'bounded')


class ControllerTests(unittest.TestCase):
    def test_empty_eligible_set_never_submits_to_worker(self):
        with tempfile.TemporaryDirectory() as d:
            experiment=Experiment(Path(d),{},Mock());experiment.worker=Mock()
            row=experiment.request(case(()),'bounded','holdout')
            experiment.worker.call.assert_not_called()
            self.assertEqual(row['decision']['origin'],'abstain');self.assertIsNone(row['response'])

    def test_unrestorable_preflight_disables_arm_and_replaces_worker(self):
        with tempfile.TemporaryDirectory() as d:
            experiment=Experiment(Path(d),{},Mock())
            experiment.request=Mock(return_value=dict(state_valid=False))
            experiment.close=Mock();experiment.start=Mock()
            result=experiment.preflight()
            self.assertFalse(result['bounded_supported']);self.assertTrue(result['worker_replacement_required'])
            experiment.close.assert_called_once();experiment.start.assert_called_once()


def report_fixture(accepted=True):
    dev=development_cases(advice_cases());holdout=fresh_holdout(dev)
    ds=make_schedule(dev,['baseline','examples','cpu'],1);hs=make_schedule(holdout,['baseline','examples','cpu'],2)
    by_id={c['case_id']:c for c in dev+holdout};rows=[]
    for stage,schedule in (('development',ds),('holdout',hs)):
        for job in schedule:
            c=by_id[job['case_id']]
            for arm in job['arms']:
                response=None
                if arm!='cpu' and c['payload']['eligible']:
                    response={}
                    r=raw(c['payload']['eligible'][0]+'<END>' if accepted else 'explanation<END>')
                    d=decide(c,r,['<END>'])
                else:d=decide(c)
                rows.append(dict(event='measurement',stage=stage,case_id=c['case_id'],arm=arm,response=response,
                                 decision=d,state_valid=True,total_ms=2))
    rows.append(dict(event='complete'))
    return rows,holdout,ds,hs


class AuditReportTests(unittest.TestCase):
    def test_fallback_can_pass_interface_but_never_model_reliability(self):
        rows,cases,ds,hs=report_fixture(False)
        s=summarize(rows,{'status':'complete'},{'challenger':'examples'},cases,ds,hs)
        self.assertTrue(s['complete'] and s['interface_passed']);self.assertFalse(s['reliability_passed'])
        self.assertEqual(s['holdout_model_requests'],80)

    def test_complete_model_success_can_pass_gate(self):
        rows,cases,ds,hs=report_fixture()
        self.assertTrue(summarize(rows,{'status':'complete'},{'challenger':'examples'},cases,ds,hs)['reliability_passed'])

    def test_missing_duplicate_or_unsafe_decision_cannot_pass(self):
        rows,cases,ds,hs=report_fixture()
        for modified in (rows[:-2],rows+[rows[-2]]):
            self.assertFalse(summarize(modified,{'status':'complete'},{'challenger':'examples'},cases,ds,hs)['reliability_passed'])
        next(r for r in rows if r['event']=='measurement' and r['stage']=='holdout')['decision']['choice']='C99'
        self.assertFalse(summarize(rows,{'status':'complete'},{'challenger':'examples'},cases,ds,hs)['interface_passed'])

    def test_independent_classifier_agrees_and_detects_no_model_call(self):
        c=case()
        for output in ('C3<END>','C3.<END>','C2<END>','C3 explanation<END>','C3'):
            r=raw(output);row=dict(arm='bounded',state_valid=True,response=dict(result=dict(raw=r)))
            expected=decide(c,r,['<END>'],True)
            self.assertEqual(classify(c,row,['<END>']),tuple(expected[k] for k in ('choice','origin','normalization','completion_boundary','fallback_reason')))
        with self.assertRaises(ValueError):classify(c,dict(arm='examples',response=None),['<END>'])

    def test_incomplete_report_renders_without_hardware(self):
        with tempfile.TemporaryDirectory() as d,patch('efficiency.adviser_report.release_snapshot'):
            build_report(Path(d),charts=False)
            self.assertFalse(json.loads(Path(d,'summary.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
