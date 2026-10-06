"""Bounded adviser validation with development-only selection and fresh holdout."""
import json
from pathlib import Path
import shutil
import time

from .adviser_interface import AdviserWorker, decide, validate_case
from .adviser_spec import (ARMS, EVENTS, SEED, choose_challenger, development_cases,
                           fresh_holdout, make_schedule, messages, specification)
from .common import ROOT, atomic_json, digest_file, emit, wait_cool
from .coordination import Budget
from .coordination_workers import Worker
from .feedback_spec import CATALOG
from .hybrid import verify_artifacts


def prepare_source(directory, config, inventory):
    source=Path(config['source_run']).resolve();verified=verify_artifacts(source)
    names=('manifest.json','config.json','summary.json','outcome.json','validation.json',
           'advice-cases.json','adviser-environment.json','next-experiment.json')
    if not all(n in verified for n in names):raise ValueError('A checksummed coordination source is required')
    read=lambda n:json.loads((source/n).read_text())
    if (read('config.json')['profile']!='coordination' or not read('summary.json')['complete'] or
        read('outcome.json')['status']!='complete' or not read('validation.json')['passed']):
        raise ValueError('Source coordination run did not complete and pass its audit')
    for key in ('model_sha256','hailort_cli','archive_sha256','cpu_model'):
        if read('manifest.json')[key]!=inventory[key]:raise ValueError(f'Source identity differs: {key}')
    folder=directory/'source-run';folder.mkdir()
    for name in (*names,'checksums.json'):shutil.copy2(source/name,folder/name)
    atomic_json(directory/'source-provenance.json',dict(source_run=str(source),verified_artifacts=len(verified),
        source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={n:digest_file(folder/n) for n in (*names,'checksums.json')}))
    cases=development_cases(read('advice-cases.json'))
    for case in cases:validate_case(case)
    atomic_json(directory/'development-cases.json',cases)
    atomic_json(directory/'protocol.json',specification())
    atomic_json(directory/'prompt-variants.json',dict(
        examples=messages(cases[0],'examples')[:-1],baseline_system=cases[0]['baseline_messages'][0],
        note='Final user payload varies by case; examples and bounded use identical messages.'))


def same_environment(environment, previous):
    keys=('hailort_version','prompt_template','stop_tokens','capacity_tokens',
          'model_defaults','parameters','experiment_limit_tokens')
    return all(environment[k]==previous[k] for k in keys)


class Experiment:
    def __init__(self,directory,config,budget):
        self.directory,self.config,self.budget=directory,config,budget
        self.worker=None;self.instance=0;self.environment=None

    def start(self):
        self.instance+=1
        self.worker=Worker('process','hat',self.directory,
            dict(self.config,block_id=f'adviser-{self.instance}'),self.budget.check,AdviserWorker)
        self.environment=self.worker.ready['environment']
        emit(self.directory/EVENTS,'worker_ready',instance=self.instance,**self.worker.ready)
        previous=json.loads((self.directory/'source-run/adviser-environment.json').read_text())
        if not same_environment(self.environment,previous):raise ValueError('Model settings differ from source')
        if not (self.directory/'adviser-environment.json').exists():
            atomic_json(self.directory/'adviser-environment.json',self.environment)

    def close(self):
        if self.worker is not None:
            worker=self.worker;self.worker=None
            try:worker.close()
            finally:emit(self.directory/EVENTS,'worker_release',instance=self.instance,
                **getattr(worker,'release',dict(alive=True,forced=True)))

    def request(self,case,arm,stage,override=None,control=None):
        self.budget.stage=stage;self.budget.check()
        started=time.monotonic();response=None
        if arm=='cpu' or not case['payload']['eligible']:
            decision=decide(case);state_valid=True
        else:
            response=self.worker.call('generate',messages=override or messages(case,arm),bounded=arm=='bounded')
            generated=response['result']
            state_valid=generated['restored'] and generated['reset_verified']
            raw=generated['raw']
            decision=decide(case,raw,self.environment['stop_tokens'],arm=='bounded')
            if not state_valid:
                decision.update(choice=decide(case)['choice'],origin='cpu_fallback',normalization=None,
                                fallback_reason='unverified_restoration_or_reset')
        ended=time.monotonic()
        row=emit(self.directory/EVENTS,'measurement',stage=stage,case_id=case['case_id'],arm=arm,
            control=control,instance=self.instance if response else None,response=response,
            decision=decision,state_valid=bool(state_valid),started=started,ended=ended,total_ms=(ended-started)*1000)
        if not state_valid and stage!='preflight':
            raise RuntimeError('Worker settings/context could not be restored; observation preserved')
        return row

    def preflight(self):
        passed=True;observations=[];unsafe=False
        for candidate in CATALOG:
            tested=sorted(set(CATALOG)-{candidate})
            case=dict(case_id='control-'+candidate,payload=dict(eligible=[candidate],tested=tested,
                development_ms={c:100 for c in tested},configurations={candidate:CATALOG[candidate]}))
            prompt=[dict(role='system',content='Copy the requested ID exactly. Do not add punctuation or explanation.'),
                    dict(role='user',content=f'Reply exactly {candidate}')]
            triple=[]
            for control,arm in (('before','baseline'),('custom','bounded'),('after','baseline')):
                row=self.request(case,arm,'preflight',override=prompt,control=control)
                triple.append(row)
                if not row['state_valid']:unsafe=True;break
            if len(triple)==3:
                a,b,c=[r['response']['result'] for r in triple]
                comparable=('output','completion_status','token_ledger_valid','context_before','context_after','expected_context_after')
                controls_equal=all(a['raw'].get(k)==c['raw'].get(k) for k in comparable)
                valid_ledgers=all(r['raw'].get('token_ledger_valid') is True and r['raw'].get('context_before')==0 and
                    r['raw'].get('context_after')==r['raw'].get('expected_context_after') for r in (a,b,c))
                supported=(controls_equal and valid_ledgers and all(r['restored'] and r['reset_verified'] for r in (a,b,c)) and
                    b['active_stops']==list(dict.fromkeys(self.environment['stop_tokens']+list(CATALOG))) and
                    triple[1]['decision']['origin']=='hat_exact' and triple[1]['decision']['choice']==candidate)
            else:controls_equal=False;supported=False;valid_ledgers=False
            observations.append(dict(candidate=candidate,supported=bool(supported),
                                     controls_equal=controls_equal,valid_ledgers=valid_ledgers))
            passed=passed and supported
            if unsafe:break
        result=dict(bounded_supported=bool(passed and len(observations)==8),controls=observations,
                    worker_replacement_required=unsafe,reason=None if passed else 'At least one control failed; bounded arm disabled')
        atomic_json(self.directory/'native-stop-preflight.json',result)
        emit(self.directory/EVENTS,'preflight_complete',**result)
        if unsafe:
            self.close();self.start()
        return result


def run(directory,config):
    directory=Path(directory);budget=Budget(directory,config)
    experiment=Experiment(directory,config,budget)
    try:
        wait_cool(directory);experiment.start()
        preflight=experiment.preflight()
        available=['baseline','examples']+(['bounded'] if preflight['bounded_supported'] else [])
        cases=json.loads((directory/'development-cases.json').read_text());case_map={c['case_id']:c for c in cases}
        pilot=[]
        for case_id in specification()['pilot_cases']:
            for arm in available:pilot.append(experiment.request(case_map[case_id],arm,'pilot'))
        planned_generations=40*len(available)+80
        required=max(r['total_ms'] for r in pilot)/1000*planned_generations*1.20+45
        gate=dict(required_seconds=required,remaining_seconds=budget.deadline-time.monotonic(),
                  projected_generations=planned_generations,margin=1.20,cleanup_reserve_seconds=45)
        gate['fits']=gate['required_seconds']<=gate['remaining_seconds']
        atomic_json(directory/'pilot-gate.json',gate);emit(directory/EVENTS,'pilot_gate',**gate)
        if not gate['fits']:raise TimeoutError('Full adviser comparison does not fit budget; no cases removed')
        development_schedule=make_schedule(cases,available+['cpu'],SEED+1)
        atomic_json(directory/'development-schedule.json',development_schedule)
        observations=[]
        for job in development_schedule:
            for arm in job['arms']:
                observations.append(experiment.request(case_map[job['case_id']],arm,'development'))
        selection=choose_challenger(observations,available)
        selection.update(available_arms=available,development_sha256=digest_file(directory/'development-cases.json'),
            environment_sha256=digest_file(directory/'adviser-environment.json'),
            implementation_sha256={n:digest_file(ROOT/'efficiency'/n) for n in ('adviser_spec.py','adviser_interface.py')})
        atomic_json(directory/'selection.json',selection)
        emit(directory/EVENTS,'selection_frozen',**selection)
        holdout=fresh_holdout(cases)
        for case in holdout:validate_case(case)
        atomic_json(directory/'holdout-cases.json',holdout)
        holdout_schedule=make_schedule(holdout,['baseline',selection['challenger'],'cpu'],SEED+2)
        atomic_json(directory/'holdout-schedule.json',holdout_schedule)
        emit(directory/EVENTS,'holdout_created',cases_sha256=digest_file(directory/'holdout-cases.json'))
        case_map={c['case_id']:c for c in holdout}
        for job in holdout_schedule:
            for arm in job['arms']:experiment.request(case_map[job['case_id']],arm,'holdout')
        emit(directory/EVENTS,'complete',holdout_requests=144,holdout_model_requests=80,
             challenger=selection['challenger'])
    except BaseException as exc:
        emit(directory/EVENTS,'incomplete',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:experiment.close()
