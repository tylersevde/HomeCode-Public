"""Supervised execution of the frozen refinement matrix."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .attention_native import fixture,oracle
from .common import ROOT,atomic_json,digest_file,emit,wait_cool
from .coordination_workers import Worker
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .research_campaign import register_fixture
from .study_protocol import Budget,controls
from .study_spec import MODEL_HASHES,LLAMA,make_table
from .refine_spec import EVENTS,BLOCKS,POLICIES,LABELS,fixtures,schedule,arm
from .refine_worker import CalibratedNumeric,TerminationContext,environment
from .refine_governor import request

def read(p):return json.loads(Path(p).read_text())

def prepare_source(directory,config,inventory):
    source=Path(config['source_campaign']);verify_artifacts(source)
    if not read(source/'final-validation.json')['passed']:raise ValueError('Source campaign audit failed')
    parent=None
    if config['parent']:
        p=Path(config['parent']);verify_artifacts(p)
        if not read(p/'validation.json')['passed'] or read(p/'summary.json')['selected']!=config['selected']:raise ValueError('Parent selection invalid')
        if read(p/'manifest.json')['source_sha256']!=inventory['source_sha256']:raise ValueError('Source changed since development')
        parent=dict(path=str(p),checksums_sha256=digest_file(p/'checksums.json'))
    if config['phase']=='cpu':shutil.copytree(ROOT/'build/attention',directory/'native-build')
    elif digest_file(LLAMA)!=MODEL_HASHES['llama']:raise ValueError('Frozen Llama model hash mismatch')
    atomic_json(directory/'freeze.json',dict(source_campaign=str(source),source_campaign_sha256=digest_file(source/'checksums.json'),
        source_sha256=inventory['source_sha256'],config_sha256=digest_file(directory/'config.json'),
        protocol_sha256=digest_file(directory/'protocol.json'),parent=parent))

@contextmanager
def worker(directory,config,budget,name,policy=None):
    budget.stage=name;budget.check();cpu=config['phase']=='cpu'
    value=Worker('process','numeric' if cpu else 'hat',directory,
        dict(config,block_id=name,policy=policy,model_id='llama'),budget.check,
        factory=CalibratedNumeric if cpu else TerminationContext,
        environment=environment(POLICIES[policy]) if cpu else None)
    emit(directory/EVENTS,'worker_ready',instance=name,**value.ready)
    try:yield value
    finally:
        try:
            if not cpu:
                response=value.call('restore')
                emit(directory/EVENTS,'native_stops_restored',response=response)
        finally:
            try:value.close()
            finally:emit(directory/EVENTS,'worker_release',instance=name,**getattr(value,'release',dict(alive=True,forced=True)))

def pilot_gate(directory,config,budget,started):
    seconds=time.monotonic()-started;multiplier=BLOCKS[config['refine_stage']]
    remaining=budget.work_deadline-time.monotonic();required=seconds*multiplier*1.2
    emit(directory/EVENTS,'pilot_gate',pilot_seconds=seconds,multiplier=multiplier,required_seconds=required,
         remaining_work_seconds=remaining,fits=required<=remaining)
    if required>remaining:raise TimeoutError('Frozen matrix cannot fit remaining work budget')

def freeze(directory,fs,jobs):
    atomic_json(directory/'fixtures.json',fs);atomic_json(directory/'schedule.json',jobs)
    emit(directory/EVENTS,'fixtures_frozen',fixtures_sha256=digest_file(directory/'fixtures.json'),schedule_sha256=digest_file(directory/'schedule.json'))

def numeric(directory,config,budget):
    for name in ('fixtures','outputs','controls'):(directory/name).mkdir()
    controls(directory,budget);fs=[]
    for spec in fixtures(config):
        budget.stage='numeric_fixtures';budget.check();x,w=fixture(spec['n'],spec['d'],spec['b'],spec['seed']);expected=oracle(x,w)
        register_fixture(config['campaign'],config['refine_stage'],spec['fixture_id'],spec['seed'],input_hash(x))
        path=directory/'fixtures'/(spec['fixture_id']+'.npz');np.savez(path,x=x,w=w,expected=expected)
        fs.append(dict(spec,file_sha256=digest_file(path),input_sha256=input_hash(x),weights_sha256=input_hash(w),oracle_sha256=input_hash(expected)))
    jobs=schedule(config,fs);freeze(directory,fs,jobs);byid={f['fixture_id']:f for f in fs}
    for section in ('pilot','main'):
        started=time.monotonic();selected=[j for j in jobs if j['section']==section]
        groups=list(dict.fromkeys((j['block'],j['policy']) for j in selected))
        for block,policy in groups:
            wait_cool(directory);budget.check()
            state=request(config['governor_socket'],POLICIES[policy]['governor'])
            emit(directory/EVENTS,'governor_transition',block=block,section=section,policy=policy,**state)
            name=f'{section}-block{block}-policy{policy}'
            with worker(directory,config,budget,name,policy) as device:
                current=None
                for job in (j for j in selected if (j['block'],j['policy'])==(block,policy)):
                    budget.stage=name+'-'+job['fixture_id'];budget.check();fid=job['fixture_id']
                    if current!=fid:
                        wait_cool(directory);budget.check();current=fid;device.call('load',fixture_id=fid)
                        for label in LABELS:
                            a=arm(byid[fid],label);response=device.call('warm',fixture_id=fid,arm=a)
                            emit(directory/EVENTS,'warmup',fixture_id=fid,section=section,block=block,policy=policy,arm=a,response=response)
                            if not response['result']['correct']:raise RuntimeError('Warmup numerical failure')
                    if job['position']==0:
                        response=device.call('measure',fixture_id=fid,arm=arm(byid[fid],'stream64'))
                        emit(directory/EVENTS,'anchor',fixture_id=fid,section=section,block=block,policy=policy,repeat=job['repeat'],response=response)
                        if not response['result']['correct'] or not response['result']['matches_warmup']:raise RuntimeError('Anchor numerical failure')
                    tick=time.monotonic();response=device.call('measure',fixture_id=fid,arm=job['arm']);end=time.monotonic()
                    emit(directory/EVENTS,'measurement',**job,started=tick,ended=end,total_ms=(end-tick)*1000,response=response)
                    if not response['result']['correct'] or not response['result']['matches_warmup']:raise RuntimeError('Measured numerical failure')
        if section=='pilot':pilot_gate(directory,config,budget,started)

def models(directory,config,budget):
    with worker(directory,config,budget,'llama-termination') as device:
        fs=[];sized={}
        for spec in fixtures(config):
            budget.stage='sizing-'+spec['fixture_id'];budget.check()
            entry=device.call('size',spec=spec)['result'];sized[spec['fixture_id']]=entry
            f=make_table(spec,entry['count']);fs.append(f)
            register_fixture(config['campaign'],config['refine_stage'],f['fixture_id'],f['seed'],f['table_sha256'])
        atomic_json(directory/'sizing.json',sized);jobs=schedule(config,fs);freeze(directory,fs,jobs)
        byid={f['fixture_id']:f for f in fs}
        for section in ('pilot','main'):
            started=time.monotonic()
            for job in (j for j in jobs if j['section']==section):
                wait_cool(directory);budget.stage=job['fixture_id']+'-'+job['condition'];budget.check()
                tick=time.monotonic();response=device.call('dialogue',fixture=byid[job['fixture_id']],condition=job['condition'],deadline=budget.work_deadline);end=time.monotonic()
                emit(directory/EVENTS,'measurement',**job,started=tick,ended=end,total_ms=(end-tick)*1000,response=response)
            if section=='pilot':pilot_gate(directory,config,budget,started)

def run(directory,config):
    directory=Path(directory);budget=Budget(directory,config)
    try:
        (numeric if config['phase']=='cpu' else models)(directory,config,budget)
        emit(directory/EVENTS,'measurement_complete')
    except Exception as exc:
        emit(directory/EVENTS,'failure',error=f'{type(exc).__name__}: {exc}')
        raise
