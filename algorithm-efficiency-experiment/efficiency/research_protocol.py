"""Five supervised research stages with frozen inputs, pilot gates, and owned workers."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .attention_native import Native,build,fixture,oracle,errors
from .common import ROOT,atomic_json,digest_file,digest_value,emit,progress,wait_cool
from .coordination import Budget
from .coordination_workers import Worker
from .feedback_spec import CELLS,input_hash
from .research_campaign import register_fixture
from .research_spec import (EVENTS,CACHE_ARMS,CONDITIONS,balanced_order,numeric_arms,
                            seed_for,checkpoint_equivalent,specification)
from .research_workers import Numeric,Context


def read(path):return json.loads(Path(path).read_text())


class ResearchBudget(Budget):
    def __init__(self,directory,config):
        super().__init__(directory,config);self.work_deadline=self.deadline-120


def prepare_source(directory,config,inventory):
    built=build();shutil.copytree(built,directory/'native-build')
    parents={};links={}
    for stage,path in config['parents'].items():
        source=Path(path);manifest=read(source/'manifest.json')
        for key in ('model_sha256','hailort_cli','archive_sha256','cpu_model'):
            if inventory[key]!=manifest[key]:raise ValueError(f'Campaign environment changed: {key}')
        if read(source/'native-build/build.json')['artifacts']!=read(built/'build.json')['artifacts']:
            raise ValueError('Campaign numerical binary changed')
        parents[stage]=read(source/'summary.json')
        links[stage]=dict(path=path,checksums_sha256=digest_file(source/'checksums.json'),
            summary_sha256=digest_file(source/'summary.json'),audit_sha256=digest_file(source/'validation.json'))
    atomic_json(directory/'parents.json',parents);atomic_json(directory/'parent-links.json',links)
    atomic_json(directory/'freeze.json',dict(protocol_sha256=digest_file(directory/'protocol.json'),
        parents_sha256=digest_file(directory/'parents.json'),manifest_source=inventory['source_sha256'],
        stage=config['research_stage'],created_before_fixtures=True))
    emit(directory/EVENTS,'protocol_frozen',**read(directory/'freeze.json'))


def start_worker(stack,directory,config,budget,kind,name,**extra):
    worker=Worker('process',kind,directory,dict(config,block_id=name,**extra),budget.check,
                  factory=Context if kind=='hat' else Numeric)
    emit(directory/EVENTS,'worker_ready',instance=name,**worker.ready)
    def close():
        try:worker.close()
        finally:emit(directory/EVENTS,'worker_release',instance=name,**getattr(worker,'release',dict(alive=True,forced=True)))
    stack.callback(close)
    return worker


def make_numeric_fixtures(directory,config,budget,cells,blocks,section):
    metadata=read(directory/'fixtures.json') if (directory/'fixtures.json').exists() else []
    selected=[]
    for block in range(-1,blocks):
        for cell in cells:
            budget.check();fid=f'{section}-{cell["cell_id"]}-b{block}'
            seed=seed_for(config.get('fixture_namespace',config['campaign_id']),config['research_stage'],fid)
            x,w=fixture(cell['n'],cell['d'],cell['b'],seed);expected=oracle(x,w)
            register_fixture(config['campaign'],config['research_stage'],fid,seed,input_hash(x))
            path=directory/'fixtures'/(fid+'.npz');np.savez(path,x=x,w=w,expected=expected)
            row=dict(**cell,fixture_id=fid,seed=seed,block=block,section=section,
                phase='pilot' if block<0 else 'main',file_sha256=digest_file(path),
                input_sha256=input_hash(x),weights_sha256=input_hash(w),oracle_sha256=input_hash(expected))
            metadata.append(row);selected.append(row)
    atomic_json(directory/'fixtures.json',metadata)
    emit(directory/EVENTS,'fixtures_frozen',section=section,ids=[r['fixture_id'] for r in selected])
    return selected


def pilot_gate(directory,budget,section,pilot_seconds,multiplier,optional=False):
    remaining=budget.work_deadline-time.monotonic();required=pilot_seconds*multiplier*1.2
    gate=dict(section=section,pilot_seconds=pilot_seconds,multiplier=multiplier,margin=1.2,
        required_seconds=required,remaining_work_seconds=remaining,cleanup_reserve_seconds=120,
        fits=required<=remaining,optional=optional)
    emit(directory/EVENTS,'pilot_gate',**gate)
    if not gate['fits'] and not optional:raise TimeoutError(f'Complete {section} matrix does not fit; no observations removed')
    return gate['fits']


def correctness_controls(directory,budget):
    controls=[]
    with Native(directory/'native-build') as cpu,Native(directory/'native-build',gpu=True,validation=True) as gpu:
        for n,d,b in ((1,1,1),(17,7,2),(31,65,2),(33,64,4)):
            budget.check();x,w=fixture(n,d,b,781+n);expected=oracle(x,w)
            cpu.configure(x.shape);gpu.configure(x.shape)
            path=directory/'controls'/f'n{n}-d{d}-b{b}.npz';np.savez(path,x=x,w=w,expected=expected)
            for mode in ('stream','prefill'):
                for backend in ('native1','native4',*(f'C{i}' for i in range(8))):
                    for variant in ((0,) if backend.startswith('native') else (0,1,2)):
                        budget.check();native=cpu if backend.startswith('native') else gpu;output=np.empty_like(x)
                        metrics=native.run(x,w,output,mode,backend,variant=variant,profiling=native is gpu)
                        result=errors(output,expected);out=directory/'outputs'/(input_hash(output)+'.npy')
                        if not out.exists():np.save(out,output,allow_pickle=False)
                        row=dict(n=n,d=d,b=b,mode=mode,backend=backend,variant=variant,
                            input_file=str(path.relative_to(directory)),output_file=str(out.relative_to(directory)),
                            **result,validation_errors=metrics['validation_errors'])
                        controls.append(row)
                        if not result['correct'] or metrics['validation_errors']:
                            atomic_json(directory/'correctness.json',controls);raise RuntimeError(f'Native control failed: {row}')
            # Sign perturbations preserve magnitude and probe causality, batch isolation and reset.
            for variant in (0,1,2):
                baseline=np.empty_like(x);gpu.run(x,w,baseline,'prefill','C6',variant=variant)
                altered=x.copy();cut=max(1,n//2);altered[0,cut:]*=-1
                perturbed=np.empty_like(x);gpu.run(altered,w,perturbed,'prefill','C6',variant=variant)
                repeated=np.empty_like(x);gpu.run(x,w,repeated,'prefill','C6',variant=variant)
                ok=np.array_equal(baseline[0,:cut],perturbed[0,:cut]) and np.array_equal(baseline[1:],perturbed[1:]) and np.array_equal(baseline,repeated)
                files={}
                for name,value in (('baseline',baseline),('perturbed',perturbed),('repeated',repeated)):
                    relative=f'outputs/{input_hash(value)}.npy'
                    if not (directory/relative).exists():np.save(directory/relative,value,allow_pickle=False)
                    files[name]=relative
                controls.append(dict(kind='semantic',n=n,d=d,b=b,variant=variant,correct=bool(ok),cut=cut,
                    input_file=str(path.relative_to(directory)),output_files=files))
                if not ok:raise RuntimeError('Causality, isolation or reset failed')
    atomic_json(directory/'correctness.json',controls)
    emit(directory/EVENTS,'correctness_complete',checks=len(controls))


def numerical(directory,config,budget,stage,parents,section='numeric'):
    blocks=8 if stage=='confirm' else 6
    budget.stage=f'{section}_fixtures';metadata=make_numeric_fixtures(directory,config,budget,CELLS,blocks,section)
    schedule=[dict(fixture_id=r['fixture_id'],phase=r['phase'],block=r['block'],cell_id=r['cell_id'],
        repeats=1 if r['phase']=='pilot' else 3,arms=numeric_arms(stage,r['cell_id'],parents)) for r in metadata]
    atomic_json(directory/f'{section}-schedule.json',schedule)
    emit(directory/EVENTS,'schedule_frozen',section=section,sha256=digest_file(directory/f'{section}-schedule.json'))
    with ExitStack() as stack:
        worker=start_worker(stack,directory,config,budget,'numeric',section)
        pilot_start=time.monotonic();gated=False
        for job in schedule:
            if job['phase']=='main' and not gated:
                pilot_gate(directory,budget,section,time.monotonic()-pilot_start,blocks*3);gated=True
            budget.stage=f'{section}_{job["fixture_id"]}';budget.check();wait_cool(directory)
            worker.call('load',fixture_id=job['fixture_id'])
            for arm in job['arms']:
                response=worker.call('warm',fixture_id=job['fixture_id'],arm=arm)
                emit(directory/EVENTS,'numeric_warm',section=section,instance=section,**job,response=response)
            for repetition in range(job['repeats']):
                for arm in balanced_order(job['arms'],max(0,job['block'])*3+repetition):
                    budget.check();started=time.monotonic()
                    response=worker.call('measure',fixture_id=job['fixture_id'],arm=arm)
                    ended=time.monotonic()
                    emit(directory/EVENTS,'numeric',section=section,instance=section,phase=job['phase'],
                        fixture_id=job['fixture_id'],cell_id=job['cell_id'],block=job['block'],repeat=repetition,
                        arm=arm['arm'],started=started,ended=ended,total_ms=(ended-started)*1000,response=response)


def context_fixture(directory,config,worker,label,target,index,section):
    fid=f'{section}-{label}-n{target}-f{index}'
    seed=seed_for(config.get('fixture_namespace',config['campaign_id']),config['research_stage'],fid)
    response=worker.call('fixture',target=target,seed=seed)['result']
    value_hash=digest_value(response['table'])
    register_fixture(config['campaign'],config['research_stage'],fid,seed,value_hash)
    return dict(**response,fixture_id=fid,model=label,index=index,section=section,table_sha256=value_hash,
                phase='pilot' if index<0 else 'main')


def context_study(directory,config,budget,parents,confirm=False):
    models=[('primary',config['model'])]
    if not confirm and config.get('secondary_model'):models.append(('secondary',config['secondary_model']))
    if not confirm and not config.get('secondary_model'):
        source=Path(config['campaign'])/'resource-status.json'
        detail=read(source).get('download_status') if source.exists() else None
        emit(directory/EVENTS,'secondary_deferred',reason='Compatible secondary model artifact unavailable'+(f': {detail}' if detail else ''))
    arms={a:a for a in CACHE_ARMS} if not confirm else {'rebuild':'rebuild','candidate':parents['context']['selection']['cache']}
    repeats=3 if confirm else 2
    all_fixtures=[];schedules=[]
    for label,model in models:
        budget.stage=f'context_{label}';budget.check();wait_cool(directory)
        with ExitStack() as stack:
            instance=f'context-{label}'
            worker=start_worker(stack,directory,dict(config,model=model),budget,'hat',instance)
            atomic_json(directory/f'{instance}-environment.json',worker.ready['environment'])
            # A disposable warmup precedes the independent pilot.
            warm=context_fixture(directory,config,worker,label,128,-2,'context-warm')
            emit(directory/EVENTS,'context_warm',instance=instance,fixture=warm,
                response=worker.call('dialogue',fixture=warm,arm='rebuild',deadline=budget.work_deadline))
            fixtures=[context_fixture(directory,config,worker,label,size,index,'context')
                      for index in range(-1,8) for size in (128,512,1024)]
            all_fixtures+=fixtures;atomic_json(directory/'context-fixtures.json',all_fixtures)
            jobs=[dict(fixture_id=f['fixture_id'],model=label,phase=f['phase'],block=f['index'],
                       target=f['target_tokens'],repeat=r,arms=arms)
                  for f in fixtures for r in range(1 if f['phase']=='pilot' else repeats)]
            schedules+=jobs;atomic_json(directory/'context-schedule.json',schedules)
            emit(directory/EVENTS,'schedule_frozen',section=instance,sha256=digest_file(directory/'context-schedule.json'))
            by_id={f['fixture_id']:f for f in fixtures};pilot_start=time.monotonic();gated=False
            for job in jobs:
                if job['phase']=='main' and not gated:
                    fits=pilot_gate(directory,budget,instance,time.monotonic()-pilot_start,8*repeats,optional=label=='secondary')
                    if not fits:
                        emit(directory/EVENTS,'secondary_deferred',reason='Complete secondary matrix exceeds remaining work budget');break
                    gated=True
                budget.check();wait_cool(directory);paired={}
                for arm in balanced_order(arms,max(0,job['block'])*repeats+job['repeat']):
                    started=time.monotonic();response=worker.call('dialogue',fixture=by_id[job['fixture_id']],
                        arm=arms[arm],deadline=budget.work_deadline);ended=time.monotonic()
                    paired[arm]=response['result']['rows']
                    emit(directory/EVENTS,'context',instance=instance,**{k:v for k,v in job.items() if k!='arms'},
                        arm=arm,policy=arms[arm],started=started,ended=ended,total_ms=(ended-started)*1000,response=response)
                equal=checkpoint_equivalent(paired)
                divergent=next((t for t in range(4) if any(len(rows)<=t or not rows[t]['valid'] for rows in paired.values()) or
                    len({(rows[t]['prompt_sha256'],rows[t]['output'],rows[t]['status'],rows[t]['context_after']) for rows in paired.values() if len(rows)>t})!=1),None)
                emit(directory/EVENTS,'cache_pair',fixture_id=job['fixture_id'],model=label,phase=job['phase'],repeat=job['repeat'],
                    equivalent=equal,first_divergent_turn=divergent,quarantined_comparisons=list(range(divergent+1,4)) if divergent is not None else [])


def combined(directory,config,budget,parents,confirm=False):
    blocks=8 if confirm else 6;section='combined'
    cells=[c for c in CELLS if c['n']==1024 and c['b']==4]
    metadata=make_numeric_fixtures(directory,config,budget,cells,blocks,section)
    cache=parents['context']['selection']['cache']
    selected=parents.get('combined',{}).get('selection',{}).get('condition','overlap-0')
    conditions={'incumbent':'overlap-0','candidate':selected} if confirm else {c:c for c in CONDITIONS}
    with ExitStack() as stack:
        cpu=start_worker(stack,directory,config,budget,'numeric','combined-cpu',numeric_device='cpu')
        gpu=start_worker(stack,directory,config,budget,'numeric','combined-gpu',numeric_device='gpu')
        hat=start_worker(stack,directory,config,budget,'hat','combined-hat')
        atomic_json(directory/'combined-environment.json',hat.ready['environment'])
        contexts={r['fixture_id']:context_fixture(directory,config,hat,'primary',512,r['block'],r['cell_id']+'-combined') for r in metadata}
        atomic_json(directory/'combined-context-fixtures.json',list(contexts.values()))
        jobs=[dict(fixture_id=r['fixture_id'],cell_id=r['cell_id'],block=r['block'],phase=r['phase'],
            repeats=1 if r['phase']=='pilot' else 3,conditions=conditions) for r in metadata]
        atomic_json(directory/'combined-schedule.json',jobs)
        emit(directory/EVENTS,'schedule_frozen',section=section,sha256=digest_file(directory/'combined-schedule.json'))
        pilot_start=time.monotonic();gated=False
        for job in jobs:
            if job['phase']=='main' and not gated:
                pilot_gate(directory,budget,section,time.monotonic()-pilot_start,blocks*3);gated=True
            budget.stage=job['fixture_id'];budget.check();wait_cool(directory)
            fid=job['fixture_id'];cell=job['cell_id'];cpu_arm=dict(arm='cpu',backend=parents['profile']['selection']['cpu'][cell],variant=0,profiling=False)
            gpu_arm=dict(arm='gpu',**parents['gpu']['selection']['gpu_best'][cell],profiling=False)
            for worker,arm,name in ((cpu,cpu_arm,'combined-cpu'),(gpu,gpu_arm,'combined-gpu')):
                worker.call('load',fixture_id=fid)
                emit(directory/EVENTS,'combined_warm',instance=name,fixture_id=fid,
                     response=worker.call('warm',fixture_id=fid,arm=arm))
            for repetition in range(job['repeats']):
                for label in balanced_order(conditions,max(0,job['block'])*3+repetition):
                    condition=conditions[label];prelude=hat.call('prepare',fixture=contexts[fid])
                    emit(directory/EVENTS,'combined_prelude',fixture_id=fid,phase=job['phase'],repeat=repetition,arm=label,response=prelude)
                    share=int(condition.split('-')[1]);gpu_count=24*share//100
                    gpu_ids=list(range(gpu_count));cpu_ids=list(range(gpu_count,24));responses={}
                    started=time.monotonic()
                    if cpu_ids:cpu.submit('batch',fixture_id=fid,arm=cpu_arm,request_ids=cpu_ids,deadline=budget.work_deadline)
                    if gpu_ids:gpu.submit('batch',fixture_id=fid,arm=gpu_arm,request_ids=gpu_ids,deadline=budget.work_deadline)
                    if condition.startswith('overlap'):hat.submit('followup',arm=cache)
                    if cpu_ids:responses['cpu']=cpu.result()
                    if gpu_ids:responses['gpu']=gpu.result()
                    if condition.startswith('serial'):hat.submit('followup',arm=cache)
                    responses['hat']=hat.result();ended=time.monotonic()
                    emit(directory/EVENTS,'combined',phase=job['phase'],fixture_id=fid,cell_id=cell,block=job['block'],repeat=repetition,
                        arm=label,condition=condition,cache_policy=cache,started=started,ended=ended,total_ms=(ended-started)*1000,
                        correct=all(r['result']['correct'] for k,r in responses.items() if k!='hat'),
                        npu_valid=responses['hat']['result']['valid'],responses=responses)


def run(directory,config):
    directory=Path(directory);budget=ResearchBudget(directory,config);parents=read(directory/'parents.json')
    for name in ('fixtures','outputs','controls'):(directory/name).mkdir()
    stage=config['research_stage']
    try:
        if stage in ('profile','gpu','confirm'):correctness_controls(directory,budget)
        if stage in ('profile','gpu'):numerical(directory,config,budget,stage,parents)
        elif stage=='context':context_study(directory,config,budget,parents)
        elif stage=='combined':combined(directory,config,budget,parents)
        elif stage=='confirm':
            numerical(directory,config,budget,stage,parents)
            context_study(directory,config,budget,parents,confirm=True)
            combined(directory,config,budget,parents,confirm=True)
        emit(directory/EVENTS,'complete',stage=stage)
    except BaseException as exc:
        emit(directory/EVENTS,'incomplete',stage=stage,error=f'{type(exc).__name__}: {exc}')
        raise
