"""Frozen reliability-first campaign design. No hardware access."""
import hashlib
import itertools
import random
import statistics as stats
import numpy as np
from .common import digest_value
from .hat import COLORS, SYSTEM
from .study_spec import CELLS, PARAMETERS, BINDINGS, messages
from .refine_spec import POLICIES

VERSION = 'reliability-v1'
EVENTS = 'reliability.jsonl'
MAX_SECONDS = 57600
CAPS = {'npu-develop':7200, 'npu-confirm':7200, 'cpu-develop':14400,
        'cpu-confirm':7200, 'gpu-confirm':7200, 'combined-develop':7200, 'combined-confirm':7200}
BLOCKS = {'npu-develop':8, 'npu-confirm':16, 'cpu-develop':128,
          'cpu-confirm':256, 'gpu-confirm':64, 'combined-develop':6, 'combined-confirm':24}
DEPENDENCIES = {'npu-confirm':('npu-develop',), 'cpu-confirm':('cpu-develop',),
                'gpu-confirm':('cpu-confirm',), 'combined-develop':('cpu-confirm','gpu-confirm','npu-confirm'),
                'combined-confirm':('combined-develop',)}
CPU_POLICIES = (1,3,5,7)
CPU_ARMS = ('single_a','single_b','batch_a','batch_b')
GPU_ARMS = ('cpu','cpu_duplicate','gpu','stream64')
CONDITIONS = ('cpu_lookup','cpu_npu_serial','cpu_npu_overlap','gpu_1','gpu_2','gpu_6')
BOOTSTRAP_SEED = 20261006

def specification(stage):
    return dict(version=VERSION,stage=stage,blocks=BLOCKS[stage],caps=CAPS,max_seconds=MAX_SECONDS,
        cleanup_seconds=120,pilot_margin=1.2,cells=CELLS,cpu_policies=[POLICIES[p] for p in CPU_POLICIES],
        cpu_arms=CPU_ARMS,batch_calls=16,repeats=3,equivalence_interval=[.95,1.05],
        minimum_speedup=1.10,stretch_speedup=1.20,max_shape_slowdown=1.05,
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_samples=10000,bootstrap_unit='whole paired blocks',
        parameters=PARAMETERS,bindings=BINDINGS,system=SYSTEM,sizes=[128,512,1024],
        native_stops=['<|end_of_text|>','<|eom_id|>','<|eot_id|>'],extra_stops=['.','\n'],
        development_gate=dict(valid=96,correct=92,per_size=29),
        confirmation_gate=dict(valid=192,correct=183,per_size=58,min_gain=.05),
        qualifying_task='full-table four-turn dialogues only',single_fact_role='independent-question diagnostic only',
        combined_conditions=CONDITIONS,combined_requests=24,
        timing='submission through validated results; steady-state loaded devices; initialization reported separately',
        retry_policy='infrastructure only, fresh fixtures, cumulative stage cap; no scientific retries',defaults_changed=False)

def seed(config,fid):
    key=f'{VERSION}|{config["fixture_namespace"]}|{config["reliability_stage"]}|{fid}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')

def fixtures(config):
    stage=config['reliability_stage']
    cells=[dict(cell_id=f'n{n}',target_tokens=n) for n in (128,512,1024)] if stage.startswith('npu') else CELLS
    return [dict(c,fixture_id=f'{section}-{c["cell_id"]}-block{b}',section=section,block=b,
                 seed=seed(config,f'{section}-{c["cell_id"]}-block{b}'))
            for section,bs in (('pilot',[-1]),('main',range(BLOCKS[stage]))) for b in bs for c in cells]

def table_rows(spec):
    rng=random.Random(spec['seed']);ids=rng.sample(range(1,1000),300)
    return [[f'item{i:03d}',rng.choice(COLORS)] for i in ids]

def table(spec,count):
    if not 4<=count<=300:raise ValueError('Invalid table size')
    rows=table_rows(spec)[:count];indices=(0,count//2,count-1,0)
    for offset,index in enumerate(indices[:3]):rows[index][1]=COLORS[(max(0,spec['block'])+offset)%len(COLORS)]
    return dict(spec,table=rows,table_sha256=digest_value(rows),initial_messages=messages(rows),
                questions=[f'What color is {rows[i][0]}?' for i in indices],answers=[rows[i][1] for i in indices])

def sizing(render,tokenize,spec):
    # Exhaustive because color-token lengths at the middle/end change with row count.
    encoded=lambda count:list(tokenize(render(table(spec,count)['initial_messages'])))
    counts=[];chosen=None;chosen_ids=None
    for count in range(4,301):
        ids=encoded(count);counts.append([count,len(ids)])
        if len(ids)<=spec['target_tokens']:chosen,chosen_ids=count,ids
    if chosen is None:raise ValueError('Four rows do not fit context target')
    return dict(count=chosen,token_ids=chosen_ids,counts=counts)

def williams(labels,block):
    n=len(labels)
    base=[0];lo,hi=1,n-1
    while lo<=hi:
        base.append(lo);lo+=1
        if lo<=hi:base.append(hi);hi-=1
    return [labels[(i+max(0,block))%n] for i in base]

def orders(config,labels,cell,block,section):
    values=list(itertools.permutations(labels));cycle=max(0,block)//8
    random.Random(seed(config,f'orders|{section}|{cell}|{cycle}')).shuffle(values)
    offset=max(0,block)%8*3
    return values[offset:offset+3]

def schedule(config,fs):
    stage=config['reliability_stage'];jobs=[]
    for section in ('pilot','main'):
        for block in sorted({f['block'] for f in fs if f['section']==section}):
            selected=[f for f in fs if f['section']==section and f['block']==block]
            if stage.startswith('npu'):
                labels=('native','punctuation','single_fact') if stage=='npu-develop' else ('native','punctuation')
                for j,f in enumerate(selected):
                    if len(labels)==3:
                        offset=(max(0,block)//2+j)%3
                        order=labels[offset:]+labels[:offset]
                        if max(0,block)%2:order=order[::-1]
                    else:order=labels if (max(0,block)+j)%2==0 else labels[::-1]
                    for condition in order:
                        jobs.append(dict(fixture_id=f['fixture_id'],section=section,block=block,target_tokens=f['target_tokens'],condition=condition))
            elif stage.startswith('combined'):
                labels=CONDITIONS if stage=='combined-develop' else ('baseline_a','baseline_b','candidate')
                order=williams(labels,block) if len(labels)==6 else list(itertools.permutations(labels))[max(0,block)%6]
                for f in selected:
                    for label in order:
                        condition=label if stage=='combined-develop' else config['selected']['candidate' if label=='candidate' else 'baseline']
                        jobs.append(dict(fixture_id=f['fixture_id'],section=section,block=block,cell_id=f['cell_id'],label=label,condition=condition))
            else:
                policies=williams(CPU_POLICIES,block) if stage=='cpu-develop' else [config['cpu_policy']]
                labels=GPU_ARMS if stage=='gpu-confirm' else CPU_ARMS
                for policy in policies:
                    for f in selected:
                        for rep,order in enumerate(orders(config,labels,f['cell_id'],block,section)):
                            for position,label in enumerate(order):
                                jobs.append(dict(fixture_id=f['fixture_id'],section=section,block=block,cell_id=f['cell_id'],
                                    policy=policy,repeat=rep,position=position,label=label))
    return jobs

def arm(cell,label):
    gpu=label in ('gpu','stream64')
    return dict(arm=label,backend=cell['gpu'] if gpu else cell['cpu'],variant=4 if label=='stream64' else 0)

def interval(a,b,ratio=True):
    a,b=np.asarray(a,float),np.asarray(b,float)
    ix=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(len(a),size=(10000,len(a)))
    av,bv=a[ix].mean(1),b[ix].mean(1);draws=np.sort(av/bv if ratio else av-bv)
    return dict(estimate=float(a.mean()/b.mean() if ratio else a.mean()-b.mean()),interval=[float(draws[250]),float(draws[9750])],
                seed=BOOTSTRAP_SEED,samples=10000,blocks=len(a))

def equivalent(value):return .95<=value['interval'][0]<=value['interval'][1]<=1.05

def values(rows,label,blocks,cell=None):
    grouped={b:[] for b in range(blocks)}
    for r in rows:
        if r['label']==label and (cell is None or r['cell_id']==cell):grouped[r['block']].append(r['total_ms'])
    return [stats.mean(grouped[b]) for b in range(blocks)]

def numeric_analysis(rows,config):
    stage=config['reliability_stage'];blocks=BLOCKS[stage];metrics=[];eligible=[]
    for policy in sorted({r['policy'] for r in rows}):
        mine=[r for r in rows if r['policy']==policy];gpu=stage=='gpu-confirm'
        pairs=[('single','cpu','cpu_duplicate')] if gpu else [('single','single_a','single_b'),('batch','batch_a','batch_b')]
        checks={kind:{cell or 'aggregate':interval(values(mine,b,blocks,cell),values(mine,a,blocks,cell))
                     for cell in [None]+[c['cell_id'] for c in CELLS]} for kind,a,b in pairs}
        labels=GPU_ARMS if gpu else CPU_ARMS
        means={label:stats.mean(values(mine,label,blocks)) for label in labels}
        stable=all(equivalent(v) for group in checks.values() for v in group.values())
        correct=all(r['response']['result']['correct'] for r in mine)
        metric=dict(label=str(policy),policy=POLICIES[policy],equivalence=checks,stable=stable,correct=correct,mean_ms=means,
                    cpu_mean_ms=(means[pairs[0][1]]+means[pairs[0][2]])/2)
        ok=stable and correct
        if gpu:
            speed=interval(values(mine,'gpu',blocks),values(mine,'stream64',blocks))
            slow=max(stats.mean(values(mine,'stream64',blocks,c['cell_id']))/stats.mean(values(mine,'gpu',blocks,c['cell_id'])) for c in CELLS)
            metric.update(speedup=speed,max_shape_slowdown=slow,gpu_cpu_ratio=means['stream64']/metric['cpu_mean_ms'])
            ok=ok and speed['estimate']>=1.1 and speed['interval'][0]>1 and slow<=1.05
        metric['qualified']=ok;metrics.append(metric)
        if ok:eligible.append(metric)
    chosen=min(eligible,key=lambda m:(m['cpu_mean_ms'],m['policy']['governor']!='ondemand',m['policy']['binding']!='unbound')) if eligible else None
    return metrics,chosen['policy']['id'] if chosen else None,{}

def npu_analysis(rows,config):
    stage=config['reliability_stage'];blocks=BLOCKS[stage];metrics=[];byblock={}
    for label in (('native','punctuation','single_fact') if stage=='npu-develop' else ('native','punctuation')):
        jobs=[r for r in rows if r['condition']==label];answers=[a for r in jobs for a in r['response']['result']['rows']]
        valid=sum(bool(a['valid']) for a in answers);correct=sum(bool(a['valid'] and a['answer_correct']) for a in answers)
        per={str(n):sum(bool(a['valid'] and a['answer_correct']) for r in jobs if r['target_tokens']==n for a in r['response']['result']['rows']) for n in (128,512,1024)}
        contracts=sum(len(r['response']['result']['contract_failures'])+int(not r['response']['result']['reset_verified']) for r in jobs)
        meets=valid==blocks*12 and correct>=(92 if blocks==8 else 183) and min(per.values())>=(29 if blocks==8 else 58) and contracts==0
        metrics.append(dict(label=label,planned=blocks*12,observed=len(answers),valid=valid,correct=correct,accuracy=correct/(blocks*12),
            per_size_correct=per,contracts=contracts,meets_numeric_threshold=meets,qualified=meets and label!='single_fact',diagnostic_only=label=='single_fact'))
        byblock[label]=[sum(bool(a['valid'] and a['answer_correct']) for r in jobs if r['block']==b for a in r['response']['result']['rows'])/12 for b in range(blocks)]
    gain=interval(byblock['punctuation'],byblock['native'],False)
    ok=metrics[1]['qualified'] and (stage=='npu-develop' or gain['estimate']>=.05 and gain['interval'][0]>0)
    return metrics,'punctuation' if ok else None,dict(accuracy_gain=gain)

def combined_analysis(rows,config):
    stage=config['reliability_stage'];blocks=BLOCKS[stage];labels=CONDITIONS if stage=='combined-develop' else ('baseline_a','baseline_b','candidate')
    metrics=[]
    for label in labels:
        mine=[r for r in rows if r['label']==label]
        metrics.append(dict(label=label,mean_ms=stats.mean(values(rows,label,blocks)),correct=all(r['correct'] for r in mine),
                            valid=all(r['valid'] for r in mine),strict_correct=sum(sum(r['answer_correct']) for r in mine)))
    extra={};selected=None
    if stage=='combined-develop':
        baselines=[m for m in metrics[:3] if m['correct'] and m['valid'] and m['strict_correct']==blocks*24]
        if not baselines:return metrics,None,extra
        baseline=min(baselines,key=lambda m:(m['mean_ms'],CONDITIONS.index(m['label'])))['label'];candidates=[]
        for m in metrics[3:]:
            speed=interval(values(rows,baseline,blocks),values(rows,m['label'],blocks))
            slow=max(stats.mean(values(rows,m['label'],blocks,c['cell_id']))/stats.mean(values(rows,baseline,blocks,c['cell_id'])) for c in CELLS)
            ok=m['correct'] and m['valid'] and m['strict_correct']==blocks*24 and speed['estimate']>=1.1 and slow<=1.05
            m.update(speedup=speed,max_shape_slowdown=slow,qualified=ok)
            if ok:candidates.append(m)
        if candidates:selected=dict(baseline=baseline,candidate=min(candidates,key=lambda m:(m['mean_ms'],CONDITIONS.index(m['label'])))['label'])
        extra['baseline']=baseline
    else:
        checks={cell or 'aggregate':interval(values(rows,'baseline_b',blocks,cell),values(rows,'baseline_a',blocks,cell)) for cell in [None]+[c['cell_id'] for c in CELLS]}
        def base(cell=None):return [(a+b)/2 for a,b in zip(values(rows,'baseline_a',blocks,cell),values(rows,'baseline_b',blocks,cell))]
        speed=interval(base(),values(rows,'candidate',blocks));slow=max(stats.mean(values(rows,'candidate',blocks,c['cell_id']))/stats.mean(base(c['cell_id'])) for c in CELLS)
        ok=all(equivalent(v) for v in checks.values()) and all(m['correct'] and m['valid'] and m['strict_correct']==blocks*24 for m in metrics) and speed['estimate']>=1.1 and speed['interval'][0]>1 and slow<=1.05
        if ok:selected=config['selected']
        extra.update(equivalence=checks,speedup=speed,max_shape_slowdown=slow)
    return metrics,selected,extra

def analyze(events,config):
    stage=config['reliability_stage'];rows=[r for r in events if r['event']=='measurement' and r['section']=='main']
    expected=sum(j['section']=='main' for j in schedule(config,fixtures(config)))
    result=dict(stage=stage,complete=False,selected=None,accepted=False,decision='incomplete',metrics=[],defaults_changed=False)
    if not any(r['event']=='measurement_complete' for r in events) or len(rows)!=expected:return result
    method=npu_analysis if stage.startswith('npu') else combined_analysis if stage.startswith('combined') else numeric_analysis
    metrics,selected,extra=method(rows,config)
    result.update(complete=True,metrics=metrics,selected=selected,accepted=selected is not None and stage.endswith('confirm'),
        decision=('candidate_selected' if stage.endswith('develop') else 'confirmed') if selected is not None else 'no_qualified_candidate',**extra)
    return result
