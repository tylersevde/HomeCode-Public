"""Offline reconstruction of fixtures, ownership, output correctness and acceptance."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import statistics as stats
import numpy as np
from .attention_native import fixture, oracle, errors
from .common import digest_file, digest_value, read_jsonl
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .monitor import safety_reason
from .refine_audit import require, semantic_score
from .refine_worker import environment
from .study_spec import MODEL_HASHES, PARAMETERS
from .study_worker import renderer
from .hat import SYSTEM
from .reliability_spec import (EVENTS, BLOCKS, CAPS, MAX_SECONDS, CELLS, POLICIES, CPU_ARMS, GPU_ARMS,
    CONDITIONS, DEPENDENCIES, fixtures, schedule, specification, table, seed, arm)

def read(p):return json.loads(Path(p).read_text())

def check_interval(saved,a,b,ratio=True):
    first,second=np.array(a,dtype=float),np.array(b,dtype=float)
    draws=np.random.Generator(np.random.PCG64(20261006)).integers(0,len(first),(10000,len(first)))
    aa,bb=np.mean(first[draws],axis=1),np.mean(second[draws],axis=1)
    values=np.sort(aa/bb if ratio else aa-bb)
    expected=dict(estimate=float(np.mean(first)/np.mean(second) if ratio else np.mean(first)-np.mean(second)),
                  interval=[float(values[250]),float(values[9750])],seed=20261006,samples=10000,blocks=len(first))
    require(saved==expected,'Paired bootstrap evidence changed')
    return expected['estimate'],expected['interval']

def check_model_row(row,transcript,expected,question,env,condition):
    native=env['native_stops'];effective=native+(['.','\n'] if condition=='punctuation' else [])
    full=renderer(env['prompt_template'],'llama')(transcript);item=re.fullmatch(r'What color is (item[0-9]+)\?',question)[1]
    score=semantic_score(row['output'],expected,item,native,effective,row['status'])
    require(row['score']==score and row['answer_correct']==score['strict_correct'],'Whole-answer scoring changed')
    require(row['native_stops']==native and row['effective_stops']==effective and row['condition']==condition,'Stop settings changed')
    require(row['messages']==transcript and row['effective_prompt']==full==row['submitted_prompt'],'Prompt reconstruction differs')
    require(row['parameters']==PARAMETERS and row['expected_answer']==expected,'Decoding/answer drift')
    require(row['output']==''.join(row['stream_chunks']) and row['prompt_sha256']==digest_value(full),'Stream/prompt evidence changed')
    require(row['context_before']==0 and row['submitted_token_ids']==row['prompt_token_ids'],'Full rebuild differs')
    require(row['prompt_tokens']==len(row['prompt_token_ids']) and row['recovery_token_ids']==env['recovery_token_ids'],'Tokenizer evidence differs')
    require(row['prompt_tokens']+32+len(row['recovery_token_ids'])<=env['experiment_limit_tokens'],'Context budget overflow')
    require(row['expected_context_after']==len(row['after_token_ids']),'After-token accounting differs')
    ledger=row['context_after']==len(row['after_token_ids']);require(row['ledger_valid']==ledger,'Context validity changed')
    suffix=score['terminal_suffix'];body=score['assistant_body'];valid=bool(ledger and suffix and body.strip() and row['status']=='LOGICAL_END_OF_GENERATION')
    require(row['terminal_suffix']==suffix and row['assistant_body']==body and row['valid']==valid,'Termination evidence changed')
    require(isinstance(row['output_token_ids'],list) and bool(row['terminal_token_ids'])==bool(suffix),'Terminal token evidence missing')
    require(row['ended']>=row['started'] and row['request_ms']==(row['ended']-row['started'])*1000,'Model timing changed')
    return body,valid

def check_dialogue(result,f,env,size,condition):
    native=env['native_stops'];effective=native+(['.','\n'] if condition!='native' else [])
    require(result['condition']==condition and result['native_stops']==native and result['effective_stops']==effective,'Dialogue condition changed')
    require(result['reset_verified'],'NPU context not reset')
    rows=result['rows'];require([r['turn'] for r in rows]==list(range(len(rows))),'Turn order differs')
    transcript=deepcopy(f['initial_messages'])
    for turn,row in enumerate(rows):
        if condition=='single_fact':
            item=re.fullmatch(r'What color is (item[0-9]+)\?',f['questions'][turn])[1]
            matches=[r for r in f['table'] if r[0]==item];require(len(matches)==1,'Ambiguous selector input')
            fact=matches[0];require(row['selected_fact']==fact and row['selector_ms']>=0,'CPU selector changed')
            transcript=[dict(role='system',content=SYSTEM),dict(role='user',content=f'Reference facts: {fact[0]} = {fact[1]}. {f["questions"][turn]}')]
        elif turn:transcript.append(dict(role='user',content=f['questions'][turn]))
        body,valid=check_model_row(row,transcript,f['answers'][turn],f['questions'][turn],env,'native' if condition=='native' else 'punctuation')
        if turn==0 and condition!='single_fact':require(row['prompt_token_ids']==size['token_ids'],'Initial sizing token IDs changed')
        if condition!='single_fact':
            if not valid:require(turn==len(rows)-1,'Invalid dialogue continued')
            else:transcript.append(dict(role='assistant',content=body))
    if condition=='single_fact':require(len(rows)==4 and result['quarantined_turns']==[] and result['diagnostic_only'],'One-fact denominator/role changed')
    else:require(result['quarantined_turns']==list(range(len(rows),4)),'Quarantine denominator changed')
    for failure in result['contract_failures']:
        require(failure['category']=='context_contract' and failure['turn']==len(rows) and bool(failure['error']),'Unaccounted context-contract failure')

def decisions(rows,config,summary):
    stage=config['reliability_stage'];blocks=BLOCKS[stage];selected=None
    def vector(sub,label,cell=None):
        grouped={b:[] for b in range(blocks)}
        for r in sub:
            if r['label']==label and (cell is None or r['cell_id']==cell):grouped[r['block']].append(r['total_ms'])
        return [stats.mean(grouped[b]) for b in range(blocks)]
    if stage.startswith('npu'):
        labels=('native','punctuation','single_fact') if stage=='npu-develop' else ('native','punctuation');vectors={};qualified=False
        require([m['label'] for m in summary['metrics']]==list(labels),'NPU summary coverage differs')
        for label,m in zip(labels,summary['metrics']):
            jobs=[r for r in rows if r['condition']==label];answers=[a for r in jobs for a in r['response']['result']['rows']]
            valid=sum(bool(a['valid']) for a in answers);correct=sum(bool(a['valid'] and a['answer_correct']) for a in answers)
            per={str(n):sum(bool(a['valid'] and a['answer_correct']) for r in jobs if r['target_tokens']==n for a in r['response']['result']['rows']) for n in (128,512,1024)}
            contracts=sum(len(r['response']['result']['contract_failures'])+int(not r['response']['result']['reset_verified']) for r in jobs)
            meets=valid==blocks*12 and correct>=(92 if stage=='npu-develop' else 183) and min(per.values())>=(29 if stage=='npu-develop' else 58) and not contracts
            require(m==dict(label=label,planned=blocks*12,observed=len(answers),valid=valid,correct=correct,accuracy=correct/(blocks*12),per_size_correct=per,
                contracts=contracts,meets_numeric_threshold=bool(meets),qualified=bool(meets and label!='single_fact'),diagnostic_only=label=='single_fact'),'NPU counts/gate differ')
            vectors[label]=[sum(bool(a['valid'] and a['answer_correct']) for r in jobs if r['block']==b for a in r['response']['result']['rows'])/12 for b in range(blocks)]
            if label=='punctuation':qualified=meets
        gain,ci=check_interval(summary['accuracy_gain'],vectors['punctuation'],vectors['native'],False)
        if qualified and (stage=='npu-develop' or gain>=.05 and ci[0]>0):selected='punctuation'
    elif stage.startswith('combined'):
        labels=CONDITIONS if stage=='combined-develop' else ('baseline_a','baseline_b','candidate');metrics=[]
        for label in labels:
            mine=[r for r in rows if r['label']==label]
            m=dict(label=label,mean_ms=stats.mean(vector(rows,label)),correct=all(r['correct'] for r in mine),valid=all(r['valid'] for r in mine),strict_correct=sum(sum(r['answer_correct']) for r in mine))
            saved=next(x for x in summary['metrics'] if x['label']==label);require(all(saved[k]==v for k,v in m.items()),'Combined counts/timing differ');metrics.append(m)
        valid=lambda m:m['correct'] and m['valid'] and m['strict_correct']==blocks*24
        if stage=='combined-develop':
            baselines=[m for m in metrics[:3] if valid(m)];eligible=[]
            if baselines:
                baseline=min(baselines,key=lambda m:(m['mean_ms'],CONDITIONS.index(m['label'])))['label'];require(summary['baseline']==baseline,'CPU baseline selection differs')
                for m in metrics[3:]:
                    saved=next(x for x in summary['metrics'] if x['label']==m['label'])
                    speed,ci=check_interval(saved['speedup'],vector(rows,baseline),vector(rows,m['label']))
                    slow=max(stats.mean(vector(rows,m['label'],c['cell_id']))/stats.mean(vector(rows,baseline,c['cell_id'])) for c in CELLS)
                    ok=valid(m) and speed>=1.1 and slow<=1.05
                    require(saved['max_shape_slowdown']==slow and saved['qualified']==ok,'Combined candidate gate differs')
                    if ok:eligible.append(m)
                if eligible:selected=dict(baseline=baseline,candidate=min(eligible,key=lambda m:(m['mean_ms'],CONDITIONS.index(m['label'])))['label'])
        else:
            stable=True
            for cell in [None]+[c['cell_id'] for c in CELLS]:
                _,ci=check_interval(summary['equivalence'][cell or 'aggregate'],vector(rows,'baseline_b',cell),vector(rows,'baseline_a',cell));stable &= ci[0]>=.95 and ci[1]<=1.05
            base=lambda cell=None:[(a+b)/2 for a,b in zip(vector(rows,'baseline_a',cell),vector(rows,'baseline_b',cell))]
            speed,ci=check_interval(summary['speedup'],base(),vector(rows,'candidate'))
            slow=max(stats.mean(vector(rows,'candidate',c['cell_id']))/stats.mean(base(c['cell_id'])) for c in CELLS)
            require(summary['max_shape_slowdown']==slow,'Combined slowdown changed')
            if stable and all(valid(m) for m in metrics) and speed>=1.1 and ci[0]>1 and slow<=1.05:selected=config['selected']
    else:
        eligible=[];gpu=stage=='gpu-confirm'
        require([int(m['label']) for m in summary['metrics']]==sorted({r['policy'] for r in rows}),'CPU summary coverage differs')
        for m in summary['metrics']:
            policy=int(m['label']);mine=[r for r in rows if r['policy']==policy]
            pairs=(('single','cpu','cpu_duplicate'),) if gpu else (('single','single_a','single_b'),('batch','batch_a','batch_b'))
            stable=True
            for kind,a,b in pairs:
                for cell in [None]+[c['cell_id'] for c in CELLS]:
                    _,ci=check_interval(m['equivalence'][kind][cell or 'aggregate'],vector(mine,b,cell),vector(mine,a,cell));stable &= ci[0]>=.95 and ci[1]<=1.05
            means={label:stats.mean(vector(mine,label)) for label in (GPU_ARMS if gpu else CPU_ARMS)}
            cpu=(means[pairs[0][1]]+means[pairs[0][2]])/2;correct=all(r['response']['result']['correct'] for r in mine)
            require(m['policy']==POLICIES[policy] and m['stable']==stable and m['correct']==correct and m['mean_ms']==means and m['cpu_mean_ms']==cpu,'CPU metrics differ')
            ok=stable and correct
            if gpu:
                speed,ci=check_interval(m['speedup'],vector(mine,'gpu'),vector(mine,'stream64'))
                slow=max(stats.mean(vector(mine,'stream64',c['cell_id']))/stats.mean(vector(mine,'gpu',c['cell_id'])) for c in CELLS)
                require(m['max_shape_slowdown']==slow and m['gpu_cpu_ratio']==means['stream64']/cpu,'GPU comparison changed')
                ok=ok and speed>=1.1 and ci[0]>1 and slow<=1.05
            require(m['qualified']==ok,'CPU/GPU eligibility differs')
            if ok:eligible.append((cpu,POLICIES[policy]['governor']!='ondemand',POLICIES[policy]['binding']!='unbound',policy))
        if eligible:selected=min(eligible)[3]
    decision=('candidate_selected' if stage.endswith('develop') else 'confirmed') if selected is not None else 'no_qualified_candidate'
    require(summary['selected']==selected and summary['decision']==decision and summary['accepted']==(selected is not None and stage.endswith('confirm')),'Final eligibility differs')

def audit(directory,require_seal=True):
    directory=Path(directory);checks=[]
    try:
        if require_seal:verify_artifacts(directory);checks.append('artifact seal')
        config=read(directory/'config.json');stage=config['reliability_stage'];npu=stage.startswith('npu');combined=stage.startswith('combined')
        require(config['phase']==('hat' if npu or combined else 'cpu'),'Stage/device phase differs')
        require(read(directory/'protocol.json')==json.loads(json.dumps(specification(stage))),'Protocol changed')
        manifest=read(directory/'manifest.json');freeze=read(directory/'freeze.json')
        require(freeze['source_sha256']==manifest['source_sha256'],'Source identity differs')
        for p,h in manifest['source_sha256'].items():require(digest_file(directory/'source'/p)==h,'Archived source changed')
        require(freeze['config_sha256']==digest_file(directory/'config.json') and freeze['protocol_sha256']==digest_file(directory/'protocol.json'),'Configuration freeze differs')
        source=Path(freeze['source_campaign']);verify_artifacts(source);require(digest_file(source/'checksums.json')==freeze['source_campaign_sha256'],'Source campaign changed')
        for name,entry in freeze['parents'].items():
            parent=Path(entry['path']);verify_artifacts(parent)
            require(digest_file(parent/'checksums.json')==entry['checksums_sha256'] and read(parent/'manifest.json')['source_sha256']==manifest['source_sha256'],'Parent/source changed')
            require(read(parent/'validation.json')['passed'] and read(parent/'summary.json')['selected'] is not None,'Unqualified parent')
            selection=read(parent/'summary.json')['selected']
            require(config['parents'][name]==str(parent),'Parent path differs')
            if name.startswith('cpu'):require(config['cpu_policy']==selection,'Confirmed CPU policy substituted')
            if stage=='combined-confirm':require(config['selected']==selection and config['cpu_policy']==read(parent/'config.json')['cpu_policy'],'Combined candidate substituted')
            elif stage in ('cpu-confirm','gpu-confirm','npu-confirm'):require(config['selected']==selection,'Confirmation selection differs')
        require(set(freeze['parents'])==set(config['parents'])==set(DEPENDENCIES.get(stage,())),'Parent set differs')
        checks.append('source, configuration and parent freeze')
        fs=read(directory/'fixtures.json');specs=fixtures(config);prior=read(directory/'prior-fixtures.json');seen=set();hashes=set()
        def fresh(f,spec,is_table=False):
            require(all(f[k]==v for k,v in spec.items()),'Fixture specification differs')
            require(f['seed'] not in prior['seeds'] and f['seed'] not in seen,'Seed reused');seen.add(f['seed'])
            h=f['table_sha256'] if is_table else f['input_sha256'];require(h not in prior['hashes'] and h not in hashes,'Input reused');hashes.add(h)
            if is_table:require(f==table(spec,len(f['table'])),'Table reconstruction differs')
            else:
                p=directory/'fixtures'/(f['fixture_id']+'.npz');require(digest_file(p)==f['file_sha256'],'Fixture file changed')
                x,w=fixture(f['n'],f['d'],f['b'],f['seed']);expected=oracle(x,w)
                with np.load(p,allow_pickle=False) as data:require(np.array_equal(data['x'],x) and np.array_equal(data['w'],w) and np.array_equal(data['expected'],expected),'FP64 fixture reconstruction failed')
                require((input_hash(x),input_hash(w),input_hash(expected))==(f['input_sha256'],f['weights_sha256'],f['oracle_sha256']),'Fixture digest differs')
        require(len(fs)==len(specs),'Fixture count differs')
        for f,spec in zip(fs,specs):fresh(f,spec,npu)
        contexts=fs if npu else read(directory/'context-fixtures.json') if combined else []
        if combined:
            require(len(contexts)==len(fs),'Combined context count differs')
            for f,c in zip(fs,contexts):
                spec=dict(fixture_id=f['fixture_id']+'-context',section=f['section'],block=f['block'],target_tokens=f['n'],cell_id=f['cell_id'],seed=seed(config,f['fixture_id']+'-context'))
                fresh(c,spec,True)
        sizes=read(directory/'sizing.json') if contexts else {}
        for f in contexts:
            size=sizes[f['fixture_id']];counts=size['counts']
            require([c[0] for c in counts]==list(range(4,301)),'Sizing candidates incomplete')
            best=max(c for c,n in counts if n<=f['target_tokens'])
            require(size['count']==len(f['table'])==best and len(size['token_ids'])==dict(counts)[best],'Largest-fit tokenizer sizing differs')
        byid={f['fixture_id']:f for f in fs};bycontext={f['fixture_id'].removesuffix('-context'):f for f in contexts}
        jobs=schedule(config,fs);require(read(directory/'schedule.json')==jobs,'Schedule changed')
        events,damaged=read_jsonl(directory/EVENTS);require(not damaged,'Damaged raw events')
        freezes=[e for e in events if e['event']=='fixtures_frozen'];require(len(freezes)==1 and freezes[0]['fixtures_sha256']==digest_file(directory/'fixtures.json') and freezes[0]['schedule_sha256']==digest_file(directory/'schedule.json'),'Fixture/schedule freeze differs')
        observations=[e for e in events if e['event']=='measurement'];require(len(observations)==len(jobs),'Incomplete measurement matrix')
        for event,job in zip(observations,jobs):
            require(all(event[k]==v for k,v in job.items()),'Measurement order/substitution differs')
            require(event['total_ms']==(event['ended']-event['started'])*1000 and event['total_ms']>0,'Request timing changed')
        active={};owners=set();warm={};checked={};restored=set();anchor=None
        def numeric_result(response,fid,name,label,expected_ids=None,is_warm=False):
            require(name in active and response['owner']==active[name]['owner'],'Numerical owner differs')
            env=active[name]['environment'];result=response['result'];p=env['policy']
            require(result['policy']==p,'Numerical policy differs')
            require(result['observer_ms']>=0,'Observer timing missing')
            require(all(result[k]['governor']==p['governor'] for k in ('system_before','system_after')),'Governor changed during work')
            rows=result['requests'] if expected_ids is not None else [result]
            if expected_ids is not None:require([r['request_id'] for r in rows]==expected_ids,'Batch request IDs/count changed')
            for row in rows:
                expected_arm=arm(byid[fid],label);require(all(row[k]==v for k,v in expected_arm.items()),'Numerical arm changed')
                key=(fid,row['output_file'])
                if key not in checked:
                    with np.load(directory/'fixtures'/(fid+'.npz')) as data:expected=data['expected']
                    out=np.load(directory/row['output_file']);checked[key]=(errors(out,expected),input_hash(out))
                err,h=checked[key]
                require(h==row['output_sha256'] and all(row[k]==v for k,v in err.items()) and row['correct'] and not row['validation_errors'],'Numerical output/verification differs')
                require(row['started']<=row['computed']<=row['ended'] and row['worker_request_ms']==(row['ended']-row['started'])*1000 and row['request_ms']>=0,'Native timing invalid')
                if is_warm:warm[name,fid,label]=h
                else:require(row['matches_warmup'] and not row['unexpected_output_io'] and warm[name,fid,label]==h,'Warmup/output differs')
            require(result['correct'],'Incorrect numerical batch')
        def response_clocks(event,response):
            require(event['started']<=response['submitted']<=response['received']<=response['ended']<=response['delivered']<=event['ended'],'Clock/transport ordering differs')
        for e in events:
            kind=e['event']
            if kind=='worker_ready':
                name=e['instance'];require(name not in active,'Worker instance reused');active[name]=e;env=e['environment']
                key=(e['owner']['pid'],e['owner']['tid']);require(key not in owners,'Fresh process owner reused');owners.add(key)
                if e['device_kind']=='hat':
                    require(env['hailort_version']=='5.1.1' and env['model_sha256']==MODEL_HASHES['llama'] and env['parameters']==PARAMETERS and set(env['native_stops'])==set(specification(stage)['native_stops']),'Model/runtime drift')
                else:
                    p=env['policy'];require(p in POLICIES and env['openmp_environment']==environment(p) and env['observed']['governor']==p['governor'],'Policy environment differs')
                    require([x['requested'] for x in env['probes']]==[1,4],'Probe matrix differs')
                    for probe in env['probes']:
                        n=probe['requested'];require(probe['team']==n and [r['mask'] for r in probe['threads']]==([1<<i for i in range(n)] if p['binding']=='bound' else [15]*n),'Actual team/affinity differs')
                    if stage.startswith('cpu'):require(env['numeric_device']=='cpu' and env['gpu_initialization_ms'] is None,'GPU contaminated isolated CPU workers')
            elif kind=='native_stops_restored':
                require(e['instance'] in active,'Restored absent owner');env=active[e['instance']]['environment']
                require(e['response']['owner']==active[e['instance']]['owner'] and e['response']['result']==dict(stops=env['native_stops'],context_tokens=0),'Native stops/context not restored');restored.add(e['instance'])
            elif kind=='worker_release':
                name=e['instance'];require(name in active and e['owner']==active[name]['owner'] and not e['forced'] and not e['alive'],'Unclean worker release')
                if active[name]['device_kind']=='hat':require(name in restored,'NPU restoration missing')
                del active[name]
            elif kind in ('warmup','anchor'):
                label=e['label'] if kind=='warmup' else 'stream64'
                numeric_result(e['response'],e['fixture_id'],e['instance'],label,is_warm=kind=='warmup')
                if kind=='anchor':anchor=(e['fixture_id'],e['policy'],e['repeat'])
            elif kind=='measurement':
                if npu:
                    require(e['instance'] in active and e['response']['owner']==active[e['instance']]['owner'],'NPU request owner differs')
                    check_dialogue(e['response']['result'],byid[e['fixture_id']],active[e['instance']]['environment'],sizes[e['fixture_id']],e['condition']);response_clocks(e,e['response'])
                elif not combined:
                    ids=list(range(16)) if e['label'].startswith('batch') else None
                    require(active[e['instance']]['environment']['policy']==POLICIES[e['policy']],'Scheduled policy differs')
                    numeric_result(e['response'],e['fixture_id'],e['instance'],e['label'],ids);response_clocks(e,e['response'])
                    if stage=='gpu-confirm':require(anchor==(e['fixture_id'],e['policy'],e['repeat']),'GPU sequence anchor missing')
                else:
                    cond=e['condition'];count=int(cond.split('_')[1]) if cond.startswith('gpu_') else 0;responses=e['responses'];context=bycontext[e['fixture_id']]
                    keys={'cpu'}|({'gpu'} if count else set())|({'hat'} if cond!='cpu_lookup' else set());require(set(responses)==keys,'Combined route changed')
                    numeric_result(responses['cpu'],e['fixture_id'],'combined-cpu','cpu',list(range(count,24)))
                    if count:numeric_result(responses['gpu'],e['fixture_id'],'combined-gpu','stream64',list(range(count)))
                    for r in responses.values():response_clocks(e,r)
                    if cond=='cpu_lookup':
                        lookup=e['lookup'];index=dict(context['table']);facts=[[q.split()[-1][:-1],index[q.split()[-1][:-1]]] for q in context['questions']]
                        require(lookup['selected_facts']==facts and lookup['answers']==[f[1] for f in facts],'CPU direct lookup differs')
                        require(e['started']<=lookup['started']<=lookup['ended']<=e['ended'],'Lookup timing excluded');answers=[a==b for a,b in zip(lookup['answers'],context['answers'])];valid=True
                    else:
                        require(e['lookup'] is None and responses['hat']['owner']==active['combined-hat']['owner'],'Combined NPU owner differs')
                        r=responses['hat']['result'];check_dialogue(r,context,active['combined-hat']['environment'],sizes[context['fixture_id']],'punctuation')
                        answers=[bool(a['valid'] and a['answer_correct']) for a in r['rows']]+[False]*(4-len(r['rows']));valid=len(r['rows'])==4 and all(a['valid'] for a in r['rows']) and not r['contract_failures'] and r['reset_verified']
                        if cond=='cpu_npu_serial':require(responses['hat']['submitted']>=responses['cpu']['delivered'],'Serial route overlapped')
                        else:require(responses['hat']['submitted']<=responses['cpu']['delivered'],'Overlap route was submitted after CPU delivery')
                    require(e['answer_correct']==answers and e['valid']==valid and e['correct']==all(r['result']['correct'] for k,r in responses.items() if k!='hat'),'Combined correctness differs')
        require(not active,'Unreleased device owner')
        if not npu:
            build=read(directory/'native-build/build.json')
            for p,h in build['artifacts'].items():require(digest_file(directory/'native-build'/p)==h,'Native binary changed')
            for p,h in build['sources'].items():require(manifest['source_sha256'][p]==h,'Native source changed')
            if stage=='gpu-confirm' or combined:
                controls=read(directory/'correctness.json');require(len(controls)==32,'Vulkan controls incomplete')
                for c in controls:
                    with np.load(directory/c['input_file']) as data:expected=oracle(data['x'],data['w'])
                    if c['kind']=='oracle':require(errors(np.load(directory/c['output_file']),expected)['correct'] and not c['validation_errors'],'FP64 GPU control failed')
                    else:
                        out={k:np.load(directory/v) for k,v in c['output_files'].items()};cut=c['cut']
                        require(np.array_equal(out['baseline'],out['repeated']) and np.array_equal(out['baseline'][0,:cut],out['perturbed'][0,:cut]) and np.array_equal(out['baseline'][1:],out['perturbed'][1:]),'Causality/reset/isolation control failed')
        checks.append('fresh fixtures, complete schedule, owned devices, native outputs and full-answer contracts')
        gates=[e for e in events if e['event']=='pilot_gate'];require(len(gates)==1,'Pilot gate missing');g=gates[0]
        require(g['multiplier']==BLOCKS[stage] and g['required_seconds']==g['pilot_seconds']*BLOCKS[stage]*1.2 and g['fits'] and g['required_seconds']<=g['remaining_work_seconds'],'Pilot projection differs')
        summary=read(directory/'summary.json');require(summary['complete'] and any(e['event']=='measurement_complete' for e in events),'Incomplete scientific stage')
        decisions([e for e in observations if e['section']=='main'],config,summary);checks.append('independent counts, paired bootstrap and selection')
        outcome=read(directory/'outcome.json');require(outcome['status']=='complete' and outcome['elapsed_seconds']<=config['max_seconds']<=config['reserved_seconds']<=CAPS[stage],'Supervised budget/outcome differs')
        restoration=read(directory/'governor-restoration.json');require(restoration['restored'] and restoration['current']==config['original_governor'],'Governor restoration failed')
        telemetry,damaged=read_jsonl(directory/'telemetry.jsonl');require(not damaged and telemetry,'Telemetry missing/damaged')
        initial=manifest['initial_throttle_flags']
        for r in telemetry:require(not r.get('stop_reason') and safety_reason(r,initial,require_hat=config['phase']=='hat' and r['phase'] not in ('preflight','complete','stopped')) is None,'Telemetry guard failed')
        checks.append('supervised budget, telemetry and restored settings')
        return dict(passed=True,checks=checks,measurements=len(observations),fixtures=len(fs),
            offline_limit='Recorded token IDs are checked for consistency; hardware tokenizer replay is separate.')
    except Exception as exc:return dict(passed=False,checks=checks,error=f'{type(exc).__name__}: {exc}')
