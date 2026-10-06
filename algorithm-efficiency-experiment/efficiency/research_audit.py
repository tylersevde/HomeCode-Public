"""Independent offline reconstruction of campaign coverage, evidence and decisions."""
from collections import Counter,defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import types

import numpy as np
from jinja2 import Environment

from .common import digest_file,digest_value,read_jsonl


def load_reference(path):
    """Execute verified source without creating bytecode inside a sealed run."""
    module=types.ModuleType('independent_original_attention')
    exec(compile(Path(path).read_text(),str(path),'exec'),module.__dict__)
    return module


def combined_trace_issues(fixture,environment,initial,followup,policy):
    """Bind the timed continuation to its frozen task and actual prelude."""
    issues=[];template=Environment().from_string(environment['prompt_template'])
    render=lambda messages:template.render(messages=messages,tools=None,add_generation_prompt=True)
    messages=deepcopy(fixture['initial_messages']);full=render(messages)
    if initial['effective_prompt']!=full or initial['submitted_prompt']!=full or initial['expected_answer']!=fixture['answers'][0]:
        issues.append('Combined prelude task differs')
    if initial['context_before']!=0:issues.append('Combined prelude did not start empty')
    if followup.get('skipped'):return issues
    ending=next((s for s in sorted(environment['stop_tokens'],key=len,reverse=True) if initial['output'].endswith(s)),None)
    if not initial['valid'] or not ending:return issues+['Combined continuation lacks valid initial output']
    messages.extend([dict(role='assistant',content=initial['output'][:-len(ending)]),dict(role='user',content=fixture['questions'][1])])
    full=render(messages);accumulated=initial['effective_prompt']+initial['output']
    if not full.startswith(accumulated):issues.append('Combined continuation prefix differs')
    submitted=full if policy=='rebuild' else full[len(accumulated):]
    if followup['effective_prompt']!=full or followup['submitted_prompt']!=submitted or followup['expected_answer']!=fixture['answers'][1]:
        issues.append('Combined followup task differs')
    if followup['context_before']!=initial['context_after']:issues.append('Combined prelude context was not retained')
    return issues


def audit(directory):
    directory=Path(directory).resolve();read=lambda n:json.loads((directory/n).read_text());issues=[]
    def require(value,reason):
        if not value and len(issues)<100:issues.append(reason)
    checks=read('checksums.json');actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'checksums.json'}
    require(set(checks)==actual,'Artifact inventory differs')
    for name,checksum in checks.items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and p.is_file() and digest_file(p)==checksum,f'Checksum differs: {name}')
    config=read('config.json');protocol=read('protocol.json');summary=read('summary.json');stage=config['research_stage']
    outcome=read('outcome.json');rows,damaged=read_jsonl(directory/'research.jsonl');main=[r for r in rows if r.get('phase')=='main']
    require(not damaged,'Damaged event lines');require(outcome['status']=='complete' and summary['complete'],'Stage incomplete')
    require(outcome['elapsed_seconds']<=config['max_seconds']<=7200,'Stage budget exceeded')
    require(sum(r['event']=='complete' for r in rows)==1 and not any(r['event']=='incomplete' for r in rows),'Completion events differ')
    require(protocol['cleanup_reserve_seconds']==120 and protocol['campaign_limit_seconds']==36000,'Budget protocol differs')
    require(rows[0]['event']=='protocol_frozen','Protocol was not frozen first')
    freeze=read('freeze.json')
    require(freeze['protocol_sha256']==digest_file(directory/'protocol.json') and freeze['parents_sha256']==digest_file(directory/'parents.json'),'Frozen design changed')
    for name,checksum in read('manifest.json')['source_sha256'].items():require(digest_file(directory/'source'/name)==checksum,f'Measurement source changed: {name}')
    for name,checksum in read('native-build/build.json')['artifacts'].items():require(digest_file(directory/'native-build'/name)==checksum,'Native build changed')
    for parent,link in read('parent-links.json').items():
        p=Path(link['path'])
        for name,key in (('checksums.json','checksums_sha256'),('summary.json','summary_sha256'),('validation.json','audit_sha256')):
            require(digest_file(p/name)==link[key],f'Parent evidence changed: {parent}')
        require(json.loads((p/'validation.json').read_text())['passed'],'Parent audit failed')
    prior=read('prior-fixtures.json');seen=set(prior['seeds']);hashes=set(prior['hashes']);fixtures={};outputs={}
    # Import only the preserved mathematical oracle, never the report or selection code.
    reference=load_reference(directory/'source/reference/algorithm_efficiency_demo.py')
    def independent_oracle(x,w):return np.stack([reference.cached_stream(a.astype(np.float64),w.astype(np.float64)) for a in x])
    def check_output(relative,expected):
        key=(relative,hashlib.sha256(expected.tobytes()).hexdigest())
        if key not in outputs:
            value=np.load(directory/relative,allow_pickle=False);delta=np.abs(value.astype(np.float64)-expected)
            outputs[key]=(bool(np.isfinite(value).all() and np.all(delta<=1e-5+1e-4*np.abs(expected))),hashlib.sha256(value.tobytes()).hexdigest())
        return outputs[key]
    for f in read('fixtures.json') if (directory/'fixtures.json').exists() else []:
        fid=f['fixture_id'];require(f['seed'] not in seen and f['input_sha256'] not in hashes,'Numerical fixture not fresh')
        namespace=config.get('fixture_namespace',config['campaign_id'])
        expected_seed=int.from_bytes(hashlib.sha256(f'attention-research-v1|{namespace}|{stage}|{fid}'.encode()).digest()[:8],'little')
        require(f['seed']==expected_seed,'Numerical seed derivation differs')
        seen.add(f['seed']);hashes.add(f['input_sha256'])
        path=directory/'fixtures'/(fid+'.npz');require(digest_file(path)==f['file_sha256'],'Fixture file changed')
        with np.load(path,allow_pickle=False) as file:x,w,expected=file['x'],file['w'],file['expected']
        rng=np.random.default_rng(f['seed']);rx=rng.normal(size=(f['b'],f['n'],f['d'])).astype(np.float32)
        rw=(rng.normal(size=(3,f['d'],f['d']))/np.sqrt(f['d'])).astype(np.float32)
        require(np.array_equal(x,rx) and np.array_equal(w,rw),'Fixture generator differs')
        independent=independent_oracle(x,w);require(np.array_equal(expected,independent),'FP64 oracle differs')
        fixtures[fid]=(f,expected)
    contexts={}
    for filename in ('context-fixtures.json','combined-context-fixtures.json'):
        if not (directory/filename).exists():continue
        for f in read(filename):
            require(f['seed'] not in seen and f['table_sha256'] not in hashes,'Context fixture not fresh')
            require(digest_value(f['table'])==f['table_sha256'],'Context table hash differs')
            seen.add(f['seed']);hashes.add(f['table_sha256']);contexts[f['fixture_id']]=f
            namespace=config.get('fixture_namespace',config['campaign_id'])
            expected_seed=int.from_bytes(hashlib.sha256(f'attention-research-v1|{namespace}|{stage}|{f["fixture_id"]}'.encode()).digest()[:8],'little')
            require(f['seed']==expected_seed,'Context seed derivation differs')
            rng=random.Random(f['seed']);colors=('blue','green','red','yellow','white','black','pink','brown')
            require(f['table']==[[f'item{i:03d}',rng.choice(colors)] for i in range(1,len(f['table'])+1)],'Context table generator differs')
            indices=(0,len(f['table'])//2,len(f['table'])-1,0)
            require(f['questions']==[f'What color is {f["table"][i][0]}?' for i in indices] and f['answers']==[f['table'][i][1] for i in indices],'Context questions/answers differ')
    if (directory/'correctness.json').exists():
        control_data={}
        for control in read('correctness.json'):
            if control.get('kind')=='semantic':
                a,b,c=(np.load(directory/control['output_files'][k],allow_pickle=False) for k in ('baseline','perturbed','repeated'))
                cut=control['cut'];require(control['correct'] and np.array_equal(a[0,:cut],b[0,:cut]) and
                    np.array_equal(a[1:],b[1:]) and np.array_equal(a,c),'Semantic control failed')
                with np.load(directory/control['input_file'],allow_pickle=False) as f:
                    x=f['x'].copy();x[0,cut:]*=-1;expected=independent_oracle(x,f['w'])
                ok,_=check_output(control['output_files']['perturbed'],expected);require(ok,'Perturbed control differs from FP64 oracle')
                continue
            if control['input_file'] not in control_data:
                with np.load(directory/control['input_file'],allow_pickle=False) as f:control_data[control['input_file']]=independent_oracle(f['x'],f['w'])
            correct,_=check_output(control['output_file'],control_data[control['input_file']])
            require(correct and control['correct'] and control['validation_errors']==0,'Vulkan/numerical control failed')
    ready={r['instance']:r for r in rows if r['event']=='worker_ready'};releases=[r for r in rows if r['event']=='worker_release']
    require(len(ready)==len(releases) and all(not r['forced'] and not r['alive'] for r in releases),'Worker release mismatch')
    def envelope(response,instance=None):
        if instance:require(response['owner']==ready[instance]['owner'],'Worker identity mismatch')
        require(response['submitted']<=response['received']<=response['ended']<=response['reply_sent']<=response['delivered'],'Worker timing order differs')
    def numeric(result):
        meta,expected=fixtures[result['fixture_id']]
        correct,checksum=check_output(result['output_file'],expected)
        require(correct==result['correct'] and checksum==result['output_sha256'],'Numerical classification/hash differs')
        require(result['validation_errors']==0,'Vulkan errors recorded')
        require(result['started']<=result['computed']<=result['ended'],'Numerical worker timing order differs')
        require(math.isclose(result['worker_request_ms'],1000*(result['ended']-result['started']),abs_tol=1e-5),'Numerical timer differs')
        require(result['profile']['enabled']==result['profiling'] and result['profile']['variant']==result['variant'],'Profiling mode differs')
        if result['profiling']:
            require(all(v>=0 for v in result['profile']['stages_ms'].values()),'Negative GPU stage duration')
            require(sum(result['profile']['stages_ms'].values())<=result['gpu_ms']*1.01+.01,'GPU stage accounting exceeds total interval')
        else:require(all(v==0 for v in result['profile']['stages_ms'].values()),'Disabled profiling contains stage samples')
    def context(result,environment):
        full=result['effective_prompt'];submitted=result['submitted_prompt'];output=result['output'];stops=environment['stop_tokens']
        require(result['prompt_sha256']==digest_value(full),'Prompt hash differs')
        require(result['prompt_tokens']==len(result['prompt_token_ids']) and result['submitted_tokens']==len(result['submitted_token_ids']),'Token counts differ')
        require(result['context_before']==len(result['history_token_ids']),'Prior context count differs')
        require(result['expected_context_after']==len(result['after_token_ids']),'Expected context count differs')
        if submitted!=full:require(result['history_token_ids']+result['submitted_token_ids']==result['prompt_token_ids'],'Token composition differs')
        require(result['parameters']==environment['parameters'],'Generation parameters differ')
        ending=next((s for s in sorted(stops,key=len,reverse=True) if output.endswith(s)),None)
        valid=bool(ending and result['status']=='LOGICAL_END_OF_GENERATION' and result['context_after']==result['expected_context_after'])
        require(valid==result['valid'],'NPU validity classification differs')
        body=output[:-len(ending)] if ending else ''
        normalized=' '.join(body.casefold().split()).strip().strip('.,!?;:').strip()
        correct=bool(result['status']=='LOGICAL_END_OF_GENERATION' and ending and '<|' not in body and '|>' not in body and
                     not any(s in body for s in stops) and normalized==result['expected_answer'])
        require(correct==result['answer_correct'],'NPU answer scoring differs')
        require(math.isclose(result['request_ms'],1000*(result['ended']-result['started']),abs_tol=1e-5),'NPU request timer differs')
        if result['checkpoint']:
            c=result['checkpoint'];require(c['model_sha256']==environment['model_sha256'] and c['context_tokens']==c['restored_tokens']==result['context_before'],'Checkpoint identity/count differs')
            require(c['bytes']==c['snapshot_bytes'] and c['available_memory_bytes']>c['required_available_bytes']>=10*c['bytes'],'Checkpoint memory guard differs')
    numbers=[];language=[];combined=[];preludes={}
    for row in rows:
        event=row['event']
        if event in ('numeric','numeric_warm','combined_warm'):
            envelope(row['response'],row['instance']);numeric(row['response']['result'])
        if event=='numeric':numbers.append(row)
        if event=='context':
            language.append(row);envelope(row['response'],row['instance']);rs=row['response']['result']['rows']
            result=row['response']['result'];failures=result.get('contract_failures',[])
            if failures:require(len(failures)==1 and failures[0]['turn']==len(rs) and failures[0]['category']=='context_contract','Contract failure coverage differs')
            if len(rs)<4:require(bool(failures) or (rs and not rs[-1]['valid']),'Unexplained missing dialogue turns')
            require(result.get('quarantined_turns',[])==list(range(len(rs),4)),'Quarantined turns differ')
            require([r['turn'] for r in rs]==list(range(len(rs))),'Dialogue turn order differs')
            environment=ready[row['instance']]['environment'];fixture=contexts[row['fixture_id']]
            messages=deepcopy(fixture['initial_messages']);accumulated=''
            template=Environment().from_string(environment['prompt_template'])
            for r in rs:
                context(r,environment);turn=r['turn']
                if turn:messages.append(dict(role='user',content=fixture['questions'][turn]))
                full=template.render(messages=messages,tools=None,add_generation_prompt=True)
                require(full==r['effective_prompt'] and full.startswith(accumulated),'Conversation rendering/history differs')
                submitted=full if row['policy']=='rebuild' else full[len(accumulated):]
                require(submitted==r['submitted_prompt'] and r['expected_answer']==fixture['answers'][turn],'Submitted context or target differs')
                if r['valid']:
                    ending=next(s for s in sorted(environment['stop_tokens'],key=len,reverse=True) if r['output'].endswith(s))
                    messages.append(dict(role='assistant',content=r['output'][:-len(ending)]));accumulated=full+r['output']
                require(bool(r['checkpoint'])==(row['policy']=='checkpoint' and turn>0),'Checkpoint coverage differs')
            require(row['response']['result']['reset_verified'],'Dialogue reset missing')
        if event=='combined_prelude':
            envelope(row['response'],'combined-hat')
            result=row['response']['result'];context(result['initial'],ready['combined-hat']['environment'])
            require(result['valid']==result['initial']['valid'],'Prelude validity differs')
            preludes[row['fixture_id'],row['repeat'],row['arm']]=result
        if event=='combined':
            combined.append(row);requests=[]
            for kind,response in row['responses'].items():
                envelope(response,'combined-'+kind)
                if kind=='hat':
                    result=response['result'];prelude=preludes.get((row['fixture_id'],row['repeat'],row['arm']))
                    require(prelude is not None,'Combined prelude missing')
                    if result.get('skipped'):
                        require(prelude is not None and not prelude['valid'] and prelude.get('reset_verified'),'Skipped followup lacks invalid reset prelude')
                        require(not result['valid'] and not result['answer_correct'] and result['generated'] is False and
                            result['skip_reason']=='invalid_prelude' and result['reset_verified'],'Invalid skip record')
                        require('output' not in result and 'stream_chunks' not in result,'Skipped followup contains generation')
                        require(result['started']<=result['ended'] and math.isclose(result['request_ms'],1000*(result['ended']-result['started']),abs_tol=1e-5),'Skip timing differs')
                    else:
                        require(prelude is not None and prelude['valid'],'Followup used invalid prelude')
                        context(result,ready['combined-hat']['environment'])
                    require(row['npu_valid']==result['valid'],'Combined NPU validity differs')
                    meta=fixtures[row['fixture_id']][0]
                    fid=f'{meta["cell_id"]}-combined-primary-n512-f{meta["block"]}'
                    if prelude is not None:
                        for issue in combined_trace_issues(contexts[fid],ready['combined-hat']['environment'],prelude['initial'],result,row['cache_policy']):require(False,issue)
                else:
                    for r in response['result']['requests']:
                        numeric(r);requests.append(r)
                        require(r['fixture_id']==row['fixture_id'],'Combined numerical task differs')
            require(sorted(r['request_id'] for r in requests)==list(range(24)),'Combined request partition is not exactly 24')
            expected_gpu=int(row['condition'].split('-')[1])*24//100
            require(sum(r['backend'].startswith('C') for r in requests)==expected_gpu,'GPU work share differs')
            require(row['correct']==all(r['correct'] and r['matches_warmup'] for r in requests),'Combined correctness differs')
            if row['condition'].startswith('serial'):
                require(row['responses']['cpu']['delivered']<=row['responses']['hat']['submitted'],'Serial NPU request started too early')
        if event in ('numeric','context','combined'):
            require(math.isclose(row['total_ms'],(row['ended']-row['started'])*1000,abs_tol=1e-5),'Full request timer differs')
            for response in row.get('responses',{'one':row.get('response')}).values():
                require(row['started']<=response['submitted'] and response['delivered']<=row['ended'],'IPC escaped request timer')
    def order(items,index):
        values=list(items);i=index%len(values);values=values[i:]+values[:i]
        return values[::-1] if (index//len(values))%2 else values
    if (directory/'numeric-schedule.json').exists():
        expected=[(j['fixture_id'],rep,a['arm']) for j in read('numeric-schedule.json') for rep in range(j['repeats']) for a in order(j['arms'],max(0,j['block'])*3+rep)]
        require([(r['fixture_id'],r['repeat'],r['arm']) for r in numbers]==expected,'Numerical schedule coverage/order differs')
    if (directory/'context-schedule.json').exists():
        deferred=bool(summary['secondary_deferred']);repeats=3 if stage=='confirm' else 2
        expected=[(j['fixture_id'],j['repeat'],arm) for j in read('context-schedule.json') if not (deferred and j['model']=='secondary' and j['phase']=='main')
                  for arm in order(j['arms'],max(0,j['block'])*repeats+j['repeat'])]
        require([(r['fixture_id'],r['repeat'],r['arm']) for r in language]==expected,'Context schedule coverage/order differs')
    if (directory/'combined-schedule.json').exists():
        expected=[(j['fixture_id'],rep,arm) for j in read('combined-schedule.json') for rep in range(j['repeats']) for arm in order(j['conditions'],max(0,j['block'])*3+rep)]
        require([(r['fixture_id'],r['repeat'],r['arm']) for r in combined]==expected,'Combined schedule coverage/order differs')
    for gate in (r for r in rows if r['event']=='pilot_gate'):
        required=gate['pilot_seconds']*gate['multiplier']*1.2
        require(math.isclose(required,gate['required_seconds']) and gate['fits']==(required<=gate['remaining_work_seconds']),'Pilot projection differs')
        require(gate['fits'] or gate['optional'],'Primary pilot gate failed')
    # Recompute all reported latencies and interval distributions without importing the report.
    mn=[r for r in numbers if r['phase']=='main'];ml=[r for r in language if r['phase']=='main'];mc=[r for r in combined if r['phase']=='main']
    for kind,data,keys,metric in (('numeric',mn,('cell_id','arm'),'mean_ms'),('context',ml,('model','arm'),'mean_dialogue_ms'),('combined',mc,('cell_id','arm'),'mean_ms')):
        group=defaultdict(list)
        for r in data:group[tuple(r[k] for k in keys)].append(r)
        require(len(group)==len(summary[kind]),'Summary group coverage differs')
        for g in summary[kind]:
            values=group[tuple(g[k] for k in keys)]
            require(math.isclose(g[metric],statistics.mean(r['total_ms'] for r in values),rel_tol=1e-12),'Mean latency differs')
            if kind=='numeric':
                data=[r['response']['result'] for r in values]
                require(g['count']==len(values) and g['correct']==all(r['correct'] and r['matches_warmup'] and not r['validation_errors'] for r in data),'Numerical summary correctness/count differs')
                require(g['median_ms']==statistics.median(r['total_ms'] for r in values),'Numerical median differs')
                for field in ('projection','scores','softmax','apply'):
                    require(math.isclose(g['stages_ms'][field],statistics.mean(r['profile']['stages_ms'][field] for r in data),abs_tol=1e-9),'GPU stage mean differs')
            elif kind=='context':
                answers=[a for r in values for a in r['response']['result']['rows']]
                require(g['dialogues']==len(values) and g['planned_answers']==4*len(values) and g['observed_answers']==len(answers),'Context coverage summary differs')
                require(g['valid_answers']==sum(a['valid'] for a in answers) and g['correct_answers']==sum(a['answer_correct'] for a in answers),'Answer accuracy summary differs')
                require(g['checkpoint_count']==sum(a['checkpoint'] is not None for a in answers),'Checkpoint summary differs')
                require(g['contract_failures']==sum(len(r['response']['result'].get('contract_failures',[])) for r in values) and
                    g['quarantined_turns']==sum(len(r['response']['result'].get('quarantined_turns',[])) for r in values),'Contract/quarantine summary differs')
            else:
                require(g['jobs']==len(values) and g['numerical_correct']==all(r['correct'] for r in values) and
                    g['npu_valid']==sum(r['npu_valid'] for r in values) and
                    g['npu_correct_answers']==sum(r['responses']['hat']['result']['answer_correct'] for r in values),'Combined summary differs')
                if 'npu_skipped' in g:require(g['npu_skipped']==sum(r['responses']['hat']['result'].get('skipped',False) for r in values),'Skipped summary differs')
    require(summary['numeric_requests']==len(mn) and summary['context_dialogues']==len(ml) and summary['combined_jobs']==len(mc),'Summary counts differ')
    generations=sum(len(r['response']['result']['rows']) for r in ml)+sum(not r['responses']['hat']['result'].get('skipped',False) for r in mc)
    require(summary['measured_model_generations']==generations,'Measured generation count differs')
    if 'skipped_combined_followups' in summary:require(summary['skipped_combined_followups']==sum(r['responses']['hat']['result'].get('skipped',False) for r in mc),'Combined skip count differs')
    def bootstrap(data,a,b,confidence=.95):
        group=defaultdict(lambda:defaultdict(list))
        for r in data:group[r['block']][r['arm']].append(r['total_ms'])
        pairs=[(statistics.mean(g[a]),statistics.mean(g[b])) for _,g in sorted(group.items()) if a in g and b in g]
        rng=random.Random(20261009);values=[]
        for _ in range(10000):
            draw=rng.choices(pairs,k=len(pairs));values.append(sum(x for x,y in draw)/sum(y for x,y in draw))
        values.sort();tail=(1-confidence)/2
        return dict(ratio=sum(x for x,y in pairs)/sum(y for x,y in pairs),interval=[values[int(tail*10000)],values[min(9999,int((1-tail)*10000))]],
                    confidence=confidence,blocks=len(pairs),samples=10000,seed=20261009)
    interval_count=0
    def same_interval(value,data,a,b,confidence=.95):
        nonlocal interval_count
        require(value==bootstrap(data,a,b,confidence),'Bootstrap interval differs');interval_count+=1
    parents=read('parents.json')
    if stage=='profile':
        for cell in (c['cell_id'] for c in protocol['cells']):
            options=[g for g in summary['numeric'] if g['cell_id']==cell and g['correct']]
            for kind in ('cpu','gpu'):
                subset=[g for g in options if (g['backend'].startswith('native') if kind=='cpu' else g['backend'].startswith('C') and not g['profiling'])]
                require(summary['selection'][kind][cell]==min(subset,key=lambda g:(g['mean_ms'],g['arm']))['backend'],'Profile selection differs')
        for entry in summary['profiling_overhead']:
            medians={g['arm']:g['median_ms'] for g in summary['numeric'] if g['cell_id']==entry['cell_id']}
            ratio=medians[entry['backend']+'-p1']/medians[entry['backend']+'-p0']
            require(entry['ratio']==ratio and entry['diagnostic_only']==(abs(ratio-1)>.05),'Profiling overhead decision differs')
    if stage=='gpu':
        for c in summary['development_comparisons']:
            cell=c['cell_id'];values=[r for r in mn if r['cell_id']==cell];cpu=parents['profile']['selection']['cpu'][cell]
            gs=[g for g in summary['numeric'] if g['cell_id']==cell and g['correct'] and g['backend'].startswith('C')]
            best=min(gs,key=lambda g:(g['mean_ms'],g['arm']));chosen=dict(backend=best['backend'],variant=best['variant'])
            require(summary['selection']['gpu_best'][cell]==chosen,'GPU development winner differs')
            same_interval(c['against_cpu'],values,cpu,best['arm']);same_interval(c['against_original_gpu'],values,best['backend']+'-v0',best['arm'])
            promote=c['against_cpu']['ratio']>=1.1 and c['against_cpu']['interval'][0]>1
            require(c['development_eligible']==promote and summary['selection']['routes'][cell]==(chosen if promote else dict(backend=cpu,variant=0)),'GPU route decision differs')
    def equivalent(data,a,b,joint=False):
        pairs=defaultdict(dict)
        for r in data:
            if r['arm'] in (a,b):pairs[r['fixture_id'],r['repeat']][r['arm']]=r
        if not pairs:return False
        for group in pairs.values():
            if not {a,b}<=set(group):return False
            if joint:
                values=[group[x] for x in (a,b)]
                if not all(r['correct'] and r['npu_valid'] for r in values):return False
                dialogues=[[r['responses']['hat']['result']] for r in values]
            else:
                dialogues=[group[x]['response']['result']['rows'] for x in (a,b)]
                if any(len(d)!=4 for d in dialogues):return False
            for ra,rb in zip(*dialogues):
                if not ra['valid'] or not rb['valid']:return False
                if any(ra[k]!=rb[k] for k in ('prompt_sha256','output','status','context_after')):return False
        return True
    if stage=='context':
        data=[r for r in ml if r['model']=='primary'];eligible=[]
        for c in summary['development_comparisons']:
            arm=c['arm'];match=equivalent(data,'rebuild',arm);require(c['cache_equivalent']==match,'Cache equivalence differs')
            same_interval(c['inference'],data,'rebuild',arm)
            if match and c['inference']['ratio']>=1.1 and c['inference']['interval'][0]>1:eligible.append(arm)
        chosen=min(eligible,key=lambda a:(statistics.mean(r['total_ms'] for r in data if r['arm']==a),a)) if eligible else 'rebuild'
        require(summary['selection']['cache']==chosen,'Cache policy selection differs')
    for g in summary['context']:
        require(g['equivalent_to_rebuild']==equivalent([r for r in ml if r['model']==g['model']],g['arm'],'rebuild'),'Context group equivalence differs')
    if stage=='combined':
        eligible=[]
        for c in summary['development_comparisons']:
            match=equivalent(mc,'overlap-0',c['arm'],True);require(c['behavior_preserved']==match,'Combined behavior differs')
            same_interval(c['inference'],mc,'overlap-0',c['arm'])
            if match:eligible.append(c['arm'])
        chosen=min(eligible,key=lambda a:(statistics.mean(r['total_ms'] for r in mc if r['arm']==a),a)) if eligible else 'overlap-0'
        require(summary['selection']['condition']==chosen,'Combined selection differs')
    if stage=='confirm':
        for c,data,a,b in zip(summary['confirmation'],(mn,ml,mc),('incumbent','rebuild','incumbent'),('candidate',)*3):
            same_interval(c['inference'],data,a,b,1-.05/3)
            correct=all(r['response']['result']['correct'] and r['response']['result']['matches_warmup'] for r in mn) if c['comparison']=='attention' else equivalent(data,a,b,c['comparison']=='coordination')
            require(c['correct']==correct,'Confirmation correctness differs')
            if c['comparison']=='attention':
                changed=any(v['backend']!=parents['profile']['selection']['cpu'][k] for k,v in parents['gpu']['selection']['routes'].items())
                ratios=[bootstrap([r for r in mn if r['cell_id']==cell['cell_id']],a,b,1-.05/3)['ratio'] for cell in protocol['cells']]
            else:
                changed=(parents['context']['selection']['cache']!='rebuild') if c['comparison']=='context' else (parents['combined']['selection']['condition']!='overlap-0')
                ratios=[]
            require(c['changed']==changed and c['shape_ratios']==ratios,'Confirmation intervention/shape guard differs')
            good=c['changed'] and correct and c['inference']['ratio']>=1.1 and c['inference']['interval'][0]>1 and all(r>=1/1.05 for r in c['shape_ratios'])
            require(c['accepted']==good,'Confirmation promotion differs')
        promotion={c['comparison']:c['accepted'] for c in summary['confirmation']}
        expected_routes=parents['gpu']['selection']['routes'] if promotion['attention'] else {c:dict(backend=b,variant=0) for c,b in parents['profile']['selection']['cpu'].items()}
        expected_cache=parents['context']['selection']['cache'] if promotion['context'] else 'rebuild'
        expected_condition=parents['combined']['selection']['condition'] if promotion['coordination'] else 'overlap-0'
        require(summary['selection']==dict(routes=expected_routes,cache=expected_cache,condition=expected_condition),'Final policy differs from confirmation')
    return dict(passed=not issues,issues=issues,verified_artifacts=len(checks),numerical_fixtures=len(fixtures),
        context_fixtures=len(contexts),verified_output_comparisons=len(outputs),numerical_requests=len(numbers),
        context_dialogues=len(language),combined_jobs=len(combined),intervals=interval_count,worker_releases=len(releases))


if __name__=='__main__':
    try:result=audit(sys.argv[1])
    except Exception as exc:result=dict(passed=False,issues=[f'{type(exc).__name__}: {exc}'])
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
