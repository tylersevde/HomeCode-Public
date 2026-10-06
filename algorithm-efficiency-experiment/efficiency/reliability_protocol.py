"""Execute frozen matrices with bounded persistent owners and explicit timing layers."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .attention_native import fixture, oracle
from .common import ROOT, atomic_json, digest_file, emit, wait_cool
from .coordination_workers import Worker
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .research_campaign import register_fixture
from .study_protocol import Budget, controls
from .study_spec import LLAMA, MODEL_HASHES
from .refine_governor import request
from .refine_worker import environment
from .reliability_spec import EVENTS, BLOCKS, POLICIES, CPU_ARMS, GPU_ARMS, fixtures, schedule, arm, table, seed
from .reliability_worker import ReliableNumeric, ReliableContext, select_fact

def read(p):return json.loads(Path(p).read_text())

def prepare_source(directory,config,inventory):
    source=Path(config['source_campaign']);verify_artifacts(source)
    audit=read(source/'final-validation.json')
    if not audit.get('passed',audit.get('integrity_passed',False)):raise ValueError('Source campaign audit failed')
    campaign=Path(config['campaign']);ledger=read(campaign/'campaign.json')
    if ledger.get('source_sha256',inventory['source_sha256'])!=inventory['source_sha256']:raise ValueError('Campaign source changed')
    ledger['source_sha256']=inventory['source_sha256'];atomic_json(campaign/'campaign.json',ledger)
    parents={}
    for stage,path in config['parents'].items():
        p=Path(path);verify_artifacts(p)
        if not read(p/'validation.json')['passed'] or read(p/'summary.json')['selected'] is None:raise ValueError('Parent did not qualify')
        if read(p/'manifest.json')['source_sha256']!=inventory['source_sha256']:raise ValueError('Source changed since development')
        parents[stage]=dict(path=str(p),checksums_sha256=digest_file(p/'checksums.json'))
    if not config['reliability_stage'].startswith('npu'):shutil.copytree(ROOT/'build/attention',directory/'native-build')
    if config['phase']=='hat' and digest_file(LLAMA)!=MODEL_HASHES['llama']:raise ValueError('Llama package identity differs')
    atomic_json(directory/'freeze.json',dict(source_campaign=str(source),source_campaign_sha256=digest_file(source/'checksums.json'),
        source_sha256=inventory['source_sha256'],config_sha256=digest_file(directory/'config.json'),
        protocol_sha256=digest_file(directory/'protocol.json'),parents=parents))

@contextmanager
def worker(directory,config,budget,name,kind='cpu',policy=None):
    budget.stage=name;budget.check();numeric=kind!='hat'
    cfg=dict(config,block_id=name,policy=policy,model_id='llama',numeric_device=kind if kind in ('cpu','gpu') else 'both')
    value=Worker('process','numeric' if numeric else 'hat',directory,cfg,budget.check,
        factory=ReliableNumeric if numeric else ReliableContext,environment=environment(POLICIES[policy]) if numeric else None)
    emit(directory/EVENTS,'worker_ready',instance=name,device_kind=kind,**value.ready)
    try:yield value
    finally:
        try:
            if not numeric:emit(directory/EVENTS,'native_stops_restored',instance=name,response=value.call('restore'))
        finally:
            try:value.close()
            finally:emit(directory/EVENTS,'worker_release',instance=name,**getattr(value,'release',dict(alive=True,forced=True)))

def pilot_gate(directory,config,budget,started):
    seconds=time.monotonic()-started;multiplier=BLOCKS[config['reliability_stage']]
    remaining=budget.work_deadline-time.monotonic();required=seconds*multiplier*1.2
    emit(directory/EVENTS,'pilot_gate',pilot_seconds=seconds,multiplier=multiplier,required_seconds=required,
         remaining_work_seconds=remaining,fits=required<=remaining)
    if required>remaining:raise TimeoutError('Frozen matrix cannot fit remaining stage allowance; stage deferred')

def freeze(directory,fs,config):
    jobs=schedule(config,fs);atomic_json(directory/'fixtures.json',fs);atomic_json(directory/'schedule.json',jobs)
    emit(directory/EVENTS,'fixtures_frozen',fixtures_sha256=digest_file(directory/'fixtures.json'),schedule_sha256=digest_file(directory/'schedule.json'))
    return jobs

def numeric_fixtures(directory,config,budget):
    fs=[]
    for spec in fixtures(config):
        budget.stage='numerical-fixtures';budget.check();x,w=fixture(spec['n'],spec['d'],spec['b'],spec['seed']);expected=oracle(x,w)
        register_fixture(config['campaign'],config['reliability_stage'],spec['fixture_id'],spec['seed'],input_hash(x))
        path=directory/'fixtures'/(spec['fixture_id']+'.npz');np.savez(path,x=x,w=w,expected=expected)
        fs.append(dict(spec,file_sha256=digest_file(path),input_sha256=input_hash(x),weights_sha256=input_hash(w),oracle_sha256=input_hash(expected)))
    return fs

def numeric(directory,config,budget):
    stage=config['reliability_stage'];gpu=stage=='gpu-confirm'
    # Vulkan controls are outside the isolated CPU experiment, including its warmups.
    if gpu:controls(directory,budget)
    fs=numeric_fixtures(directory,config,budget);jobs=freeze(directory,fs,config);byid={f['fixture_id']:f for f in fs}
    for section in ('pilot','main'):
        started=time.monotonic();selected=[j for j in jobs if j['section']==section]
        for block,policy in dict.fromkeys((j['block'],j['policy']) for j in selected):
            wait_cool(directory);budget.check()
            emit(directory/EVENTS,'governor_transition',block=block,section=section,policy=policy,**request(config['governor_socket'],POLICIES[policy]['governor']))
            name=f'{section}-block{block}-policy{policy}'
            with worker(directory,config,budget,name,'both' if gpu else 'cpu',policy) as device:
                current=None
                for job in (j for j in selected if (j['block'],j['policy'])==(block,policy)):
                    budget.stage=name+'-'+job['fixture_id'];budget.check();fid=job['fixture_id'];cell=byid[fid]
                    if current!=fid:
                        wait_cool(directory);budget.check();current=fid;device.call('load',fixture_id=fid)
                        for label in (GPU_ARMS if gpu else CPU_ARMS):
                            response=device.call('warm',fixture_id=fid,arm=arm(cell,label))
                            emit(directory/EVENTS,'warmup',fixture_id=fid,policy=policy,label=label,instance=name,response=response)
                            if not response['result']['correct']:raise RuntimeError('Warmup numerical failure')
                    if gpu and job['position']==0:
                        response=device.call('measure',fixture_id=fid,arm=arm(cell,'stream64'))
                        emit(directory/EVENTS,'anchor',fixture_id=fid,policy=policy,repeat=job['repeat'],instance=name,response=response)
                        if not response['result']['correct']:raise RuntimeError('GPU anchor numerical failure')
                    tick=time.monotonic()
                    if job['label'].startswith('batch'):
                        response=device.call('batch',fixture_id=fid,arm=arm(cell,job['label']),request_ids=list(range(16)),deadline=budget.work_deadline)
                    else:response=device.call('measure',fixture_id=fid,arm=arm(cell,job['label']))
                    end=time.monotonic();emit(directory/EVENTS,'measurement',**job,instance=name,started=tick,ended=end,total_ms=(end-tick)*1000,response=response)
                    result=response['result'];calls=result.get('requests',[result])
                    if not result['correct'] or any(not r['matches_warmup'] or r['validation_errors'] for r in calls):raise RuntimeError('Measured numerical failure')
        if section=='pilot':pilot_gate(directory,config,budget,started)

def npu_fixtures(directory,config,budget,device,specs):
    fs=[];sizes={}
    for spec in specs:
        budget.stage='sizing-'+spec['fixture_id'];budget.check()
        entry=device.call('size',spec=spec)['result'];sizes[spec['fixture_id']]=entry;f=table(spec,entry['count']);fs.append(f)
        register_fixture(config['campaign'],config['reliability_stage'],f['fixture_id'],f['seed'],f['table_sha256'])
    return fs,sizes

def models(directory,config,budget):
    with worker(directory,config,budget,'llama-reliability','hat') as device:
        fs,sizes=npu_fixtures(directory,config,budget,device,fixtures(config));atomic_json(directory/'sizing.json',sizes)
        jobs=freeze(directory,fs,config);byid={f['fixture_id']:f for f in fs}
        for section in ('pilot','main'):
            started=time.monotonic()
            for job in (j for j in jobs if j['section']==section):
                wait_cool(directory);budget.stage=job['fixture_id']+'-'+job['condition'];budget.check();tick=time.monotonic()
                if job['condition']=='single_fact':response=device.call('single_fact',fixture=byid[job['fixture_id']],deadline=budget.work_deadline)
                else:response=device.call('dialogue',fixture=byid[job['fixture_id']],condition=job['condition'],deadline=budget.work_deadline)
                end=time.monotonic();emit(directory/EVENTS,'measurement',**job,instance='llama-reliability',started=tick,ended=end,total_ms=(end-tick)*1000,response=response)
            if section=='pilot':pilot_gate(directory,config,budget,started)

def combined(directory,config,budget):
    controls(directory,budget);fs=numeric_fixtures(directory,config,budget);jobs=freeze(directory,fs,config);byid={f['fixture_id']:f for f in fs}
    policy=config['cpu_policy'];emit(directory/EVENTS,'governor_transition',policy=policy,**request(config['governor_socket'],POLICIES[policy]['governor']))
    with ExitStack() as stack:
        cpu=stack.enter_context(worker(directory,config,budget,'combined-cpu','cpu',policy))
        gpu=stack.enter_context(worker(directory,config,budget,'combined-gpu','gpu',policy))
        hat=stack.enter_context(worker(directory,config,budget,'combined-hat','hat'))
        specs=[dict(fixture_id=f['fixture_id']+'-context',section=f['section'],block=f['block'],target_tokens=f['n'],
                    cell_id=f['cell_id'],seed=seed(config,f['fixture_id']+'-context')) for f in fs]
        contexts,sizes=npu_fixtures(directory,config,budget,hat,specs);atomic_json(directory/'context-fixtures.json',contexts);atomic_json(directory/'sizing.json',sizes)
        bycontext={f['fixture_id'].removesuffix('-context'):f for f in contexts}
        for section in ('pilot','main'):
            started=time.monotonic();current=None
            for job in (j for j in jobs if j['section']==section):
                budget.stage=job['fixture_id']+'-'+job['label'];budget.check();fid=job['fixture_id'];cell=byid[fid];context=bycontext[fid]
                ca,ga=arm(cell,'cpu'),arm(cell,'stream64')
                if current!=fid:
                    wait_cool(directory);budget.check();current=fid
                    for dev,a,name in ((cpu,ca,'combined-cpu'),(gpu,ga,'combined-gpu')):
                        dev.call('load',fixture_id=fid);response=dev.call('warm',fixture_id=fid,arm=a)
                        emit(directory/EVENTS,'warmup',fixture_id=fid,policy=policy,label=a['arm'],instance=name,response=response)
                        if not response['result']['correct']:raise RuntimeError('Combined warmup failure')
                cond=job['condition'];count=int(cond.split('_')[1]) if cond.startswith('gpu_') else 0
                tick=time.monotonic();responses={};lookup=None
                cpu.submit('batch',fixture_id=fid,arm=ca,request_ids=list(range(count,24)),deadline=budget.work_deadline)
                if count:gpu.submit('batch',fixture_id=fid,arm=ga,request_ids=list(range(count)),deadline=budget.work_deadline)
                if cond=='cpu_lookup':
                    start_lookup=time.monotonic();facts=[select_fact(context['table'],q) for q in context['questions']]
                    lookup=dict(selected_facts=facts,answers=[f[1] for f in facts],started=start_lookup,ended=time.monotonic())
                elif cond!='cpu_npu_serial':hat.submit('dialogue',fixture=context,condition='punctuation',deadline=budget.work_deadline)
                responses['cpu']=cpu.result()
                if count:responses['gpu']=gpu.result()
                if cond=='cpu_npu_serial':hat.submit('dialogue',fixture=context,condition='punctuation',deadline=budget.work_deadline)
                if cond!='cpu_lookup':responses['hat']=hat.result()
                correct=all(r['result']['correct'] for k,r in responses.items() if k!='hat')
                if lookup:
                    answers=[a==b for a,b in zip(lookup['answers'],context['answers'])];valid=True
                else:
                    result=responses['hat']['result'];answers=[bool(r['valid'] and r['answer_correct']) for r in result['rows']]
                    answers += [False]*(4-len(answers));valid=len(result['rows'])==4 and all(r['valid'] for r in result['rows']) and not result['contract_failures'] and result['reset_verified']
                end=time.monotonic();emit(directory/EVENTS,'measurement',**job,started=tick,ended=end,total_ms=(end-tick)*1000,
                    responses=responses,lookup=lookup,correct=correct,valid=valid,answer_correct=answers)
            if section=='pilot':pilot_gate(directory,config,budget,started)

def run(directory,config):
    directory=Path(directory);budget=Budget(directory,config)
    for name in ('fixtures','outputs','controls'):(directory/name).mkdir()
    try:
        stage=config['reliability_stage'];method=models if stage.startswith('npu') else combined if stage.startswith('combined') else numeric
        method(directory,config,budget);emit(directory/EVENTS,'measurement_complete')
    except Exception as exc:
        emit(directory/EVENTS,'failure',error=f'{type(exc).__name__}: {exc}');raise
