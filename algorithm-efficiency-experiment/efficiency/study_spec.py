"""Frozen two-track design and pure analysis; no hardware or application defaults."""
import hashlib
import random
import statistics
import numpy as np
from .common import MODEL, digest_value
from .hat import COLORS, SYSTEM
from .research_spec import balanced_order

VERSION='stream-model-study-v1'
EVENTS='study.jsonl'
LLAMA=MODEL.parent/'llama3.2-3b-study/Llama-3_2-3B-Instruct.hef'
MODEL_HASHES={'qwen':'5310176848638505fbc28add04ba60c97abe345cdb0ec7e3b8ffaa4b0a8c65dd',
              'llama':'1129f5f8384e4e45c5890104dc4ec1aee77e800ce1484ddc3aa942399aada425'}
CELLS=[dict(cell_id=f'stream-n{n}-b{b}',mode='stream',n=n,b=b,d=64,
            cpu='native4' if b==4 and n>=512 else 'native1',
            gpu='C4' if b==1 and n>=512 else 'C6') for n in (128,512,1024) for b in (1,4)]
PARAMETERS=dict(temperature=0.699999988079071,top_p=0.800000011920929,top_k=20,
                frequency_penalty=1.0,max_generated_tokens=32,do_sample=False,seed=12345)
BINDINGS=dict(bos_token='<|begin_of_text|>',eos_token='<|eot_id|>',date_string='03 Oct 2026',tools=None)


def specification(track,stage):
    if track not in ('gpu-stream','npu-model') or stage not in ('develop','confirm'):raise ValueError('Unknown study')
    return dict(version=VERSION,track=track,stage=stage,blocks=8 if stage=='develop' else 16,
        campaign_seconds=28800,track_seconds=14400,stage_seconds=7200,cleanup_seconds=120,
        pilot_blocks=1,pilot_margin=1.2,repeats=3,gpu_cells=CELLS,
        gpu_variants={'stream32':3,'stream64':4},gpu_min_speedup=1.10,gpu_max_cell_slowdown=1.05,
        cpu_control_max_difference=.05,atol=1e-5,rtol=1e-4,
        models=MODEL_HASHES,system=SYSTEM,parameters=PARAMETERS,template_bindings=BINDINGS,
        sizes=[128,512,1024],turns=4,min_rows=4,max_rows=300,context_policy='full_rebuild_each_turn',
        development_min_correct=92,development_per_size=29,confirmation_min_correct=183,
        confirmation_per_size=58,min_accuracy_gain=.05,bootstrap_seed=20261004,bootstrap_samples=10000,
        bootstrap_unit='block grouping all shapes/sizes; summarize repetitions within block',
        order='balanced GPU arms; alternating model order each block, three sizes per load',
        retry_policy='infrastructure only within cumulative stage cap; no retries after scientific outcome',
        scope='GPU attention implementation and deployed NPU model packages; no claim of LLM GPU acceleration',
        defaults_changed=False)


def seed_for(namespace,track,stage,fid):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{namespace}|{track}|{stage}|{fid}'.encode()).digest()[:8],'little')


def fixtures(config):
    count=8 if config['study_stage']=='develop' else 16
    gpu=config['track']=='gpu-stream'
    return [dict(fixture_id=f'{section}-{cell["cell_id"]}-block{block}',section=section,block=block,
                 seed=seed_for(config['fixture_namespace'],config['track'],config['study_stage'],f'{section}-{cell["cell_id"]}-block{block}'),
                 **cell,**({} if gpu else dict(first_color=COLORS[block%8] if block>=0 else ('brown','green','blue')[j])))
            for section,blocks in (('pilot',[-1]),('main',range(count))) for block in blocks
            for j,cell in enumerate(CELLS if gpu else [dict(cell_id=f'n{n}',target_tokens=n) for n in (128,512,1024)])]


def arms(config,cell):
    labels=['cpu','cpu_duplicate','gpu']+(['stream32','stream64'] if config['study_stage']=='develop' else [config['selected']])
    return [dict(arm=label,backend=cell['cpu'] if label.startswith('cpu') else cell['gpu'],
                 variant={'stream32':3,'stream64':4}.get(label,0)) for label in labels]


def schedule(config,fs):
    if config['track']=='gpu-stream':
        return [dict(fixture_id=f['fixture_id'],section=f['section'],block=f['block'],cell_id=f['cell_id'],
                     repeat=rep,arm=arm) for i,f in enumerate(fs) for rep in range(3)
                for arm in balanced_order(arms(config,f),i*3+rep)]
    result=[]
    for section in ('pilot','main'):
        for block in sorted({f['block'] for f in fs if f['section']==section}):
            for model in (('qwen','llama') if block%2==0 else ('llama','qwen')):
                for f in fs:
                    if f['section']==section and f['block']==block:
                        result.append(dict(fixture_id=f['fixture_id'],section=section,block=block,
                                           target_tokens=f['target_tokens'],model=model))
    return result


def table_rows(spec):
    rng=random.Random(spec['seed']);rows=[[f'item{i:03d}',rng.choice(COLORS)] for i in range(1,301)]
    rows[0][1]=spec['first_color'];return rows


def messages(table):
    facts='; '.join(f'{k} = {v}' for k,v in table)
    return [dict(role='system',content=SYSTEM),dict(role='user',content=f'Reference facts: {facts}. What color is {table[0][0]}?')]


def make_table(spec,count):
    if not 4<=count<=300:raise ValueError('Invalid shared table count')
    table=table_rows(spec)[:count];indices=(0,count//2,count-1,0)
    return dict(spec,table=table,table_sha256=digest_value(table),initial_messages=messages(table),
                questions=[f'What color is {table[i][0]}?' for i in indices],answers=[table[i][1] for i in indices])


def sizing(render,tokenize,spec,count=None):
    rows=table_rows(spec)
    encoded=lambda n:list(tokenize(render(messages(rows[:n]))))
    if count is None:
        low,high,best=4,300,None
        while low<=high:
            mid=(low+high)//2
            if len(encoded(mid))<=spec['target_tokens']:best,low=mid,mid+1
            else:high=mid-1
        if best is None:raise ValueError('Four common rows do not fit target')
        count=best
    return dict(count=count,token_ids=encoded(count),next_row_token_ids=encoded(count+1) if count<300 else None)


def paired_interval(a,b,ratio=True):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if len(a)<2 or len(a)!=len(b):return None
    # Persisted integer seed and PCG64 generator; entire paired blocks resampled.
    indices=np.random.default_rng(20261004).integers(0,len(a),size=(10000,len(a)))
    av,bv=a[indices].mean(axis=1),b[indices].mean(axis=1)
    values=av/bv if ratio else av-bv
    values.sort()
    return dict(estimate=float(a.mean()/b.mean() if ratio else a.mean()-b.mean()),
                interval=[float(values[250]),float(values[9750])],seed=20261004,samples=10000,blocks=len(a),generator='PCG64')


def analyze(events,config):
    track,stage=config['track'],config['study_stage'];blocks=8 if stage=='develop' else 16
    complete=any(e['event']=='measurement_complete' for e in events)
    main=[e for e in events if e['event']=='measurement' and e['section']=='main']
    result=dict(track=track,stage=stage,complete=complete,selected=None,accepted=False,defaults_changed=False,
                decision='incomplete',metrics=[])
    if track=='gpu-stream':
        labels=[a['arm'] for a in arms(config,CELLS[0])];means={};byblock={}
        for label in labels:
            rows=[e for e in main if e['arm']['arm']==label]
            means[label]={c['cell_id']:statistics.mean(e['total_ms'] for e in rows if e['cell_id']==c['cell_id'])
                          for c in CELLS if any(e['cell_id']==c['cell_id'] for e in rows)}
            byblock[label]=[statistics.mean(e['total_ms'] for e in rows if e['block']==b) for b in range(blocks)
                            if any(e['block']==b for e in rows)]
            result['metrics'].append(dict(label=label,observed=len(rows),planned=blocks*18,
                mean_ms=statistics.mean(means[label].values()) if means[label] else None,per_cell_ms=means[label],
                all_correct=bool(rows) and all(e['response']['result']['correct'] and e['response']['result']['matches_warmup'] for e in rows)))
        complete=complete and all(r['observed']==r['planned'] and r['all_correct'] for r in result['metrics'])
        if complete:
            control=paired_interval(byblock['cpu_duplicate'],byblock['cpu']);result['cpu_control']=control
            stable=not(abs(control['estimate']-1)>.05 and (control['interval'][0]>1 or control['interval'][1]<1))
            result['stable']=stable;eligible=[]
            for label in labels[3:]:
                interval=paired_interval(byblock['gpu'],byblock[label]);slow=max(means[label][c]/means['gpu'][c] for c in means['gpu'])
                qualified=stable and interval['estimate']>=1.10 and slow<=1.05 and (stage=='develop' or interval['interval'][0]>1)
                metric=next(r for r in result['metrics'] if r['label']==label)
                metric.update(speedup=interval,max_cell_slowdown=slow,qualified=qualified,
                              cpu_ratio=metric['mean_ms']/next(r['mean_ms'] for r in result['metrics'] if r['label']=='cpu'))
                if qualified:eligible.append(metric)
            chosen=min(eligible,key=lambda r:(r['mean_ms'],r['label']!='stream32')) if eligible else None
            result.update(selected=chosen['label'] if chosen else None,accepted=bool(chosen and stage=='confirm'),
                          decision=('candidate_selected' if stage=='develop' else 'gpu_improved') if chosen else ('unstable_cpu_control' if not stable else 'no_qualified_candidate'))
    else:
        correctblocks={}
        for model in ('qwen','llama'):
            jobs=[e for e in main if e['model']==model];rows=[r for e in jobs for r in e['response']['result']['rows']]
            counts={str(size):sum(bool(r['valid'] and r['answer_correct']) for e in jobs if e['target_tokens']==size for r in e['response']['result']['rows']) for size in (128,512,1024)}
            valid=sum(bool(r['valid']) for r in rows);correct=sum(counts.values())
            contracts=sum(len(e['response']['result']['contract_failures'])+(not e['response']['result']['reset_verified']) for e in jobs)
            qualified=len(jobs)==blocks*3 and valid==blocks*12 and correct>=(92 if stage=='develop' else 183) and min(counts.values())>=(29 if stage=='develop' else 58) and not contracts
            times=[e['total_ms'] for e in jobs if len(e['response']['result']['rows'])==4 and all(r['valid'] for r in e['response']['result']['rows'])]
            result['metrics'].append(dict(label=model,planned=blocks*12,observed=len(rows),valid=valid,correct=correct,
                accuracy=correct/(blocks*12),per_size_correct=counts,contracts=contracts,qualified=qualified,
                mean_complete_dialogue_ms=statistics.mean(times) if times else None))
            correctblocks[model]=[sum(bool(r['valid'] and r['answer_correct']) for e in jobs if e['block']==b for r in e['response']['result']['rows'])/12 for b in range(blocks)]
        complete=complete and len(main)==blocks*6
        if complete:
            challenger=result['metrics'][1];gain=paired_interval(correctblocks['llama'],correctblocks['qwen'],False)
            accepted=challenger['qualified'] and (stage=='develop' or (gain['estimate']>=.05 and gain['interval'][0]>0))
            result.update(selected='llama' if accepted else None,accepted=bool(accepted and stage=='confirm'),accuracy_gain=gain,
                          decision=('candidate_selected' if stage=='develop' else 'model_improved') if accepted else 'no_qualified_candidate')
    result['complete']=complete
    if not complete:result.update(selected=None,accepted=False,decision='incomplete')
    return result
