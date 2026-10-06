"""Sequential, guarded execution of frozen streaming and model studies."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .attention_native import Native,fixture,oracle,errors
from .common import ROOT,atomic_json,digest_file,digest_value,emit,progress,wait_cool
from .coordination import Budget as BaseBudget
from .coordination_workers import Worker
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .research_campaign import register_fixture
from .research_workers import Numeric
from .study_worker import ModelContext
from .study_spec import EVENTS,MODEL_HASHES,CELLS,fixtures,schedule,arms,make_table


def read(path):return json.loads(Path(path).read_text())


class Budget(BaseBudget):
    def __init__(self,directory,config):
        super().__init__(directory,config);self.work_deadline=self.deadline-120;self.phase=config['phase']
    def check(self):
        now=time.monotonic()
        if now>=self.work_deadline:raise TimeoutError('Study work deadline reached; cleanup reserved')
        if (self.directory/'STOP').exists():raise RuntimeError('Operator STOP file detected')
        if now-self.last_progress>=1:
            status=read(self.directory/'status.json')
            if status.get('stop_reason'):raise RuntimeError(status['stop_reason'])
            if not 0<=now-status['monotonic']<=8:raise RuntimeError('Supervisor telemetry stale')
            progress(self.directory,self.phase,self.stage);self.last_progress=now


def prepare_source(directory,config,inventory):
    parent=None
    if config['parent']:
        p=Path(config['parent']);verify_artifacts(p)
        if not read(p/'validation.json')['passed'] or read(p/'summary.json')['selected']!=config['selected']:raise ValueError('Parent selection invalid')
        if read(p/'manifest.json')['source_sha256']!=inventory['source_sha256']:raise ValueError('Source changed since development')
        parent=dict(path=str(p),checksums_sha256=digest_file(p/'checksums.json'))
        shutil.copy2(p/'summary.json',directory/'development-summary.json')
    provenance={}
    if config['track']=='gpu-stream':
        shutil.copytree(ROOT/'build/attention',directory/'native-build')
        for stage in ('profile','gpu'):
            basis=ROOT/f'runs/research-{stage}-20261003';verify_artifacts(basis)
            shutil.copy2(basis/'summary.json',directory/f'baseline-{stage}.json')
            provenance[stage]=dict(path=str(basis),checksums_sha256=digest_file(basis/'checksums.json'))
        baseline=read(directory/'baseline-gpu.json')['selection']
        for cell in CELLS:
            if baseline['routes'][cell['cell_id']]!={'backend':cell['cpu'],'variant':0} or baseline['gpu_best'][cell['cell_id']]!={'backend':cell['gpu'],'variant':0}:
                raise ValueError('Frozen per-cell baseline differs from prior study')
    else:
        for model,key in (('qwen','model'),('llama','challenger_model')):
            path=Path(config[key])
            if digest_file(path)!=MODEL_HASHES[model]:raise ValueError(f'{model} dependency hash mismatch')
            provenance[model]=dict(path=str(path),bytes=path.stat().st_size,sha256=MODEL_HASHES[model])
        shutil.copy2(Path(config['challenger_model']).parent/'manifest.json',directory/'llama-manifest.json')
    atomic_json(directory/'freeze.json',dict(config_sha256=digest_file(directory/'config.json'),
        protocol_sha256=digest_file(directory/'protocol.json'),source_sha256=inventory['source_sha256'],parent=parent,provenance=provenance))


@contextmanager
def worker(directory,config,budget,kind,name,**extra):
    budget.stage=name;budget.check()
    value=Worker('process',kind,directory,dict(config,block_id=name,**extra),budget.check,
                 factory=Numeric if kind=='numeric' else ModelContext)
    emit(directory/EVENTS,'worker_ready',instance=name,**value.ready)
    try:yield value
    finally:
        try:value.close()
        finally:emit(directory/EVENTS,'worker_release',instance=name,**getattr(value,'release',dict(alive=True,forced=True)))


def pilot_gate(directory,config,budget,started):
    seconds=time.monotonic()-started;multiplier=8 if config['study_stage']=='develop' else 16
    remaining=budget.work_deadline-time.monotonic();required=seconds*multiplier*1.2
    emit(directory/EVENTS,'pilot_gate',pilot_seconds=seconds,multiplier=multiplier,required_seconds=required,
         remaining_work_seconds=remaining,fits=required<=remaining)
    if required>remaining:raise TimeoutError('Frozen full matrix cannot fit remaining stage budget')


def controls(directory,budget):
    rows=[]
    with Native(directory/'native-build',gpu=True,validation=True) as gpu:
        for n,d,b in ((1,1,1),(17,7,2),(33,17,4),(65,65,2)):
            budget.check();x,w=fixture(n,d,b,17701+n);expected=oracle(x,w);gpu.configure(x.shape)
            path=directory/'controls'/f'n{n}-d{d}-b{b}.npz';np.savez(path,x=x,w=w,expected=expected)
            for mode,variants in (('stream',(0,3,4)),('prefill',(0,1,2))):
                for variant in variants:
                    budget.check();out=np.empty_like(x);metrics=gpu.run(x,w,out,mode,'C6',variant=variant)
                    output=f'outputs/{input_hash(out)}.npy';np.save(directory/output,out,allow_pickle=False)
                    row=dict(kind='oracle',mode=mode,variant=variant,input_file=str(path.relative_to(directory)),output_file=output,
                             **errors(out,expected),validation_errors=metrics['validation_errors'])
                    rows.append(row);atomic_json(directory/'correctness.json',rows)
                    if not row['correct'] or metrics['validation_errors']:raise RuntimeError('Numerical control failed')
                    if variant in (3,4):
                        cut=max(1,n//2);altered=x.copy();altered[0,cut:]*=-1
                        perturbed=np.empty_like(x);gpu.run(altered,w,perturbed,'stream','C6',variant=variant)
                        repeated=np.empty_like(x);gpu.run(x,w,repeated,'stream','C6',variant=variant)
                        ok=np.array_equal(out[0,:cut],perturbed[0,:cut]) and np.array_equal(out[1:],perturbed[1:]) and np.array_equal(out,repeated)
                        saved={}
                        for name,value in (('baseline',out),('perturbed',perturbed),('repeated',repeated)):
                            rel=f'outputs/{input_hash(value)}.npy';np.save(directory/rel,value,allow_pickle=False);saved[name]=rel
                        rows.append(dict(kind='semantic',mode=mode,variant=variant,input_file=str(path.relative_to(directory)),
                                         output_files=saved,cut=cut,correct=bool(ok)))
                        atomic_json(directory/'correctness.json',rows)
                        if not ok:raise RuntimeError('Causality, reset or batch isolation failed')
    emit(directory/EVENTS,'correctness_complete',checks=len(rows))


def numeric(directory,config,budget):
    for name in ('fixtures','outputs','controls'):(directory/name).mkdir()
    controls(directory,budget);fs=[]
    for spec in fixtures(config):
        budget.stage='numeric_fixtures';budget.check();x,w=fixture(spec['n'],spec['d'],spec['b'],spec['seed']);expected=oracle(x,w)
        register_fixture(config['campaign'],config['track']+'-'+config['study_stage'],spec['fixture_id'],spec['seed'],input_hash(x))
        path=directory/'fixtures'/(spec['fixture_id']+'.npz');np.savez(path,x=x,w=w,expected=expected)
        fs.append(dict(spec,file_sha256=digest_file(path),input_sha256=input_hash(x),weights_sha256=input_hash(w),oracle_sha256=input_hash(expected)))
        atomic_json(directory/'fixtures.json',fs)
    jobs=schedule(config,fs);atomic_json(directory/'schedule.json',jobs)
    emit(directory/EVENTS,'fixtures_frozen',fixtures_sha256=digest_file(directory/'fixtures.json'),schedule_sha256=digest_file(directory/'schedule.json'))
    byid={f['fixture_id']:f for f in fs}
    with worker(directory,config,budget,'numeric','streaming') as device:
        for section in ('pilot','main'):
            start=time.monotonic();current=None
            for job in (j for j in jobs if j['section']==section):
                budget.stage=job['fixture_id'];budget.check()
                if current!=job['fixture_id']:
                    wait_cool(directory);budget.check();current=job['fixture_id'];device.call('load',fixture_id=current)
                    for arm in arms(config,byid[current]):
                        response=device.call('warm',fixture_id=current,arm=arm)
                        emit(directory/EVENTS,'warmup',fixture_id=current,arm=arm,response=response)
                        if not response['result']['correct']:raise RuntimeError('Warmup numerical failure')
                started=time.monotonic();response=device.call('measure',fixture_id=current,arm=job['arm']);ended=time.monotonic()
                emit(directory/EVENTS,'measurement',**job,started=started,ended=ended,total_ms=(ended-started)*1000,response=response)
                if not response['result']['correct'] or not response['result']['matches_warmup']:raise RuntimeError('Measured output changed or is incorrect')
            if section=='pilot':pilot_gate(directory,config,budget,start)


def models(directory,config,budget):
    specs=fixtures(config);sized={}
    for model in ('qwen','llama'):
        path=config['model'] if model=='qwen' else config['challenger_model']
        with worker(directory,config,budget,'hat','sizing-'+model,model_id=model,model=path) as device:
            sized[model]={}
            for spec in specs:
                sized[model][spec['fixture_id']]=device.call('size',spec=spec)['result']
            atomic_json(directory/'sizing.json',sized)
    fs=[]
    for spec in specs:
        count=min(sized[m][spec['fixture_id']]['count'] for m in sized)
        f=make_table(spec,count)
        register_fixture(config['campaign'],config['track']+'-'+config['study_stage'],f['fixture_id'],f['seed'],f['table_sha256'])
        fs.append(f)
    atomic_json(directory/'fixtures.json',fs);jobs=schedule(config,fs);atomic_json(directory/'schedule.json',jobs)
    emit(directory/EVENTS,'fixtures_frozen',fixtures_sha256=digest_file(directory/'fixtures.json'),schedule_sha256=digest_file(directory/'schedule.json'))
    byid={f['fixture_id']:f for f in fs};evidence={}
    for section in ('pilot','main'):
        started=time.monotonic()
        selected=[j for j in jobs if j['section']==section]
        for block in sorted({j['block'] for j in selected}):
            order=list(dict.fromkeys(j['model'] for j in selected if j['block']==block))
            for model in order:
                wait_cool(directory);budget.check();path=config['model'] if model=='qwen' else config['challenger_model']
                name=f'{section}-block{block}-{model}'
                with worker(directory,config,budget,'hat',name,model_id=model,model=path) as device:
                    for job in (j for j in selected if j['block']==block and j['model']==model):
                        f=byid[job['fixture_id']];budget.stage=f['fixture_id']+'-'+model;budget.check()
                        entry=device.call('size',spec=f,count=len(f['table']))['result']
                        if len(entry['token_ids'])>f['target_tokens']:raise ValueError('Common table exceeds model target')
                        evidence.setdefault(f['fixture_id'],{})[model]=entry;atomic_json(directory/'token-evidence.json',evidence)
                        tick=time.monotonic();response=device.call('dialogue',fixture=f,deadline=budget.work_deadline);end=time.monotonic()
                        emit(directory/EVENTS,'measurement',**job,started=tick,ended=end,total_ms=(end-tick)*1000,response=response)
        if section=='pilot':pilot_gate(directory,config,budget,started)


def run(directory,config):
    directory=Path(directory);budget=Budget(directory,config)
    try:
        (numeric if config['track']=='gpu-stream' else models)(directory,config,budget)
        emit(directory/EVENTS,'measurement_complete')
    except Exception as exc:
        emit(directory/EVENTS,'failure',error=f'{type(exc).__name__}: {exc}')
        raise
