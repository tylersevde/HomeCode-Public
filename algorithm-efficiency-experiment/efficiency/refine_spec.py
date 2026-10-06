"""Preregistered calibration and termination experiment; no hardware imports."""
import hashlib
import itertools
import random
import statistics as stats
import numpy as np
from .study_spec import CELLS, make_table, table_rows, PARAMETERS, BINDINGS
from .hat import COLORS, SYSTEM

VERSION = 'cpu-stop-refinement-v1'
EVENTS = 'refine.jsonl'
CAPS = {'cpu-develop':7200, 'cpu-confirm':3600, 'gpu-confirm':3600,
        'npu-develop':7200, 'npu-confirm':7200}
BLOCKS = {'cpu-develop':8, 'cpu-confirm':32, 'gpu-confirm':16,
          'npu-develop':8, 'npu-confirm':16}
POLICIES = [dict(id=i, governor=g, binding=b, waiting=w) for i,(g,b,w) in enumerate(
    itertools.product(('ondemand','performance'), ('bound','unbound'), ('default','passive')))]
LABELS = ('cpu','cpu_duplicate','gpu','stream64')

def specification(stage):
    return dict(version=VERSION, stage=stage, blocks=BLOCKS[stage], caps=CAPS,
        max_seconds=28800, cleanup_seconds=120, pilot_margin=1.2, policies=POLICIES,
        cells=CELLS, repeats=3, arm_order='all 24 permutations per 8 blocks; GPU64 anchor before each',
        policy_order_base=[0,1,7,2,6,3,5,4], equivalence_interval=[.95,1.05],
        gpu_min_speedup=1.10, gpu_max_cell_slowdown=1.05, bootstrap_seed=20261005,
        bootstrap_samples=10000, bootstrap_unit='whole blocks, average repetitions, equal shapes',
        native_stops=['<|end_of_text|>','<|eom_id|>','<|eot_id|>'], extra_stops=['.','\n'],
        parameters=PARAMETERS, bindings=BINDINGS, system=SYSTEM, sizes=[128,512,1024],
        development_gate=dict(valid=96,correct=92,per_size=29),
        confirmation_gate=dict(valid=192,correct=183,per_size=58,min_gain=.05),
        retry_policy='infrastructure only; fresh fixtures; cumulative stage cap; no scientific retries',
        defaults_changed=False)

def seed(config, fid):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{config["fixture_namespace"]}|{config["refine_stage"]}|{fid}'.encode()).digest()[:8],'little')

def fixtures(config):
    stage=config['refine_stage']; numeric=not stage.startswith('npu')
    cells=CELLS if numeric else [dict(cell_id=f'n{n}',target_tokens=n) for n in (128,512,1024)]
    result=[]
    for section,blocks in (('pilot',[-1]),('main',range(BLOCKS[stage]))):
        for block in blocks:
            for j,cell in enumerate(cells):
                fid=f'{section}-{cell["cell_id"]}-block{block}'
                result.append(dict(cell,fixture_id=fid,section=section,block=block,seed=seed(config,fid),
                    **({} if numeric else dict(first_color=COLORS[block%8] if block>=0 else ('brown','green','blue')[j]))))
    return result

def policy_order(block):
    return [(i+block)%8 for i in (0,1,7,2,6,3,5,4)]

def arm(cell,label):
    return dict(arm=label,backend=cell['cpu'] if label.startswith('cpu') else cell['gpu'],variant=4 if label=='stream64' else 0)

def permutations(config,cell,block,section):
    orders=list(itertools.permutations(LABELS))
    cycle=max(block,0)//8
    random.Random(seed(config,f'{section}|{cell}|cycle{cycle}')).shuffle(orders)
    offset=(max(block,0)%8)*3
    return orders[offset:offset+3]

def schedule(config,fs):
    jobs=[]; stage=config['refine_stage']
    for section in ('pilot','main'):
        for block in sorted({f['block'] for f in fs if f['section']==section}):
            selected=[f for f in fs if f['section']==section and f['block']==block]
            if stage.startswith('npu'):
                for j,f in enumerate(selected):
                    for condition in (('native','punctuation') if (block+j)%2==0 else ('punctuation','native')):
                        jobs.append(dict(fixture_id=f['fixture_id'],section=section,block=block,target_tokens=f['target_tokens'],condition=condition))
            else:
                policies=policy_order(max(block,0)) if stage=='cpu-develop' else [config['selected']]
                for policy in policies:
                    for f in selected:
                        for repeat,order in enumerate(permutations(config,f['cell_id'],block,section)):
                            for position,label in enumerate(order):
                                jobs.append(dict(fixture_id=f['fixture_id'],section=section,block=block,cell_id=f['cell_id'],
                                    policy=policy,repeat=repeat,position=position,arm=arm(f,label)))
    return jobs

def interval(a,b,ratio=True):
    a,b=np.asarray(a,float),np.asarray(b,float)
    indices=np.random.Generator(np.random.PCG64(20261005)).integers(len(a),size=(10000,len(a)))
    aa,bb=a[indices].mean(1),b[indices].mean(1)
    values=np.sort(aa/bb if ratio else aa-bb)
    return dict(estimate=float(a.mean()/b.mean() if ratio else a.mean()-b.mean()),
        interval=[float(values[250]),float(values[9750])],seed=20261005,samples=10000,blocks=len(a))

def analyze(events,config):
    stage=config['refine_stage']; blocks=BLOCKS[stage]
    rows=[e for e in events if e['event']=='measurement' and e['section']=='main']
    complete=any(e['event']=='measurement_complete' for e in events)
    result=dict(stage=stage,complete=False,selected=None,accepted=False,decision='incomplete',metrics=[],defaults_changed=False)
    expected=blocks*6 if stage.startswith('npu') else blocks*72*(8 if stage=='cpu-develop' else 1)
    if not complete or len(rows)!=expected:return result
    result['complete']=True
    if stage.startswith('npu'):
        grouped={}
        for label in ('native','punctuation'):
            jobs=[e for e in rows if e['condition']==label]; answers=[r for e in jobs for r in e['response']['result']['rows']]
            valid=sum(bool(r['valid']) for r in answers)
            correct=sum(bool(r['valid'] and r['answer_correct']) for r in answers)
            counts={str(n):sum(bool(r['valid'] and r['answer_correct']) for e in jobs if e['target_tokens']==n for r in e['response']['result']['rows']) for n in (128,512,1024)}
            contracts=sum(len(e['response']['result']['contract_failures'])+int(not e['response']['result']['reset_verified']) for e in jobs)
            qualified=valid==blocks*12 and correct>=(92 if blocks==8 else 183) and min(counts.values())>=(29 if blocks==8 else 58) and contracts==0
            result['metrics'].append(dict(label=label,planned=blocks*12,observed=len(answers),valid=valid,correct=correct,
                accuracy=correct/(blocks*12),per_size_correct=counts,contracts=contracts,qualified=qualified))
            grouped[label]=[sum(bool(r['valid'] and r['answer_correct']) for e in jobs if e['block']==b for r in e['response']['result']['rows'])/12 for b in range(blocks)]
        gain=interval(grouped['punctuation'],grouped['native'],False); result['accuracy_gain']=gain
        ok=result['metrics'][1]['qualified'] and (stage=='npu-develop' or (gain['estimate']>=.05 and gain['interval'][0]>0))
        result.update(selected='punctuation' if ok else None,accepted=ok and stage=='npu-confirm',decision='candidate_selected' if ok and stage=='npu-develop' else 'termination_improved' if ok else 'no_qualified_candidate')
        return result
    eligible=[]
    for policy in sorted({e['policy'] for e in rows}):
        mine=[e for e in rows if e['policy']==policy]
        def values(label,cell=None):
            return [stats.mean(e['total_ms'] for e in mine if e['block']==b and e['arm']['arm']==label and (cell is None or e['cell_id']==cell)) for b in range(blocks)]
        checks={c:interval(values('cpu_duplicate',c),values('cpu',c)) for c in [None]+[x['cell_id'] for x in CELLS]}
        checks={k or 'aggregate':v for k,v in checks.items()}
        stable=all(.95<=v['interval'][0]<=v['interval'][1]<=1.05 for v in checks.values())
        correct=all(e['response']['result']['correct'] and e['response']['result']['matches_warmup'] for e in mine)
        means={label:stats.mean(values(label)) for label in LABELS}
        speed=interval(values('gpu'),values('stream64'))
        slow=max(stats.mean(values('stream64',c['cell_id']))/stats.mean(values('gpu',c['cell_id'])) for c in CELLS)
        metric=dict(label=str(policy),policy=POLICIES[policy],cpu_equivalence=checks,stable=stable,correct=correct,
            cpu_mean_ms=(means['cpu']+means['cpu_duplicate'])/2,mean_ms=means,gpu_speedup=speed,
            max_cell_slowdown=slow,gpu_cpu_ratio=means['stream64']/((means['cpu']+means['cpu_duplicate'])/2))
        result['metrics'].append(metric)
        if stable and correct:eligible.append(metric)
    chosen=min(eligible,key=lambda m:(m['cpu_mean_ms'],m['policy']['governor']!='ondemand',m['policy']['binding']!='unbound',m['policy']['waiting']!='default')) if eligible else None
    ok=bool(chosen)
    if stage=='gpu-confirm' and ok:ok=chosen['gpu_speedup']['estimate']>=1.10 and chosen['gpu_speedup']['interval'][0]>1 and chosen['max_cell_slowdown']<=1.05
    result.update(selected=chosen['policy']['id'] if ok else None,accepted=ok and stage!='cpu-develop',
        decision=('candidate_selected' if stage=='cpu-develop' else 'cpu_calibrated' if stage=='cpu-confirm' else 'gpu_improved') if ok else 'no_qualified_candidate')
    return result
