"""Offline reconstruction of raw evidence, paired intervals and eligibility."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import statistics as stats
import numpy as np
from .attention_native import fixture,oracle,errors
from .common import digest_file,digest_value,read_jsonl
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .hat import COLORS
from .study_spec import MODEL_HASHES,PARAMETERS,make_table
from .study_worker import renderer
from .refine_spec import EVENTS,CELLS,POLICIES,BLOCKS,LABELS,fixtures,schedule,specification
from .refine_worker import environment

def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())

def bootstrap(a,b,ratio=True):
    a,b=np.array(a),np.array(b)
    indices=np.random.Generator(np.random.PCG64(20261005)).integers(0,len(a),(10000,len(a)))
    first,second=np.mean(a[indices],axis=1),np.mean(b[indices],axis=1)
    values=np.sort(first/second if ratio else first-second)
    return float(np.mean(a)/np.mean(b) if ratio else np.mean(a)-np.mean(b)),[float(values[250]),float(values[9750])]

def check_interval(saved,a,b,ratio=True):
    value,ci=bootstrap(a,b,ratio)
    require(saved==dict(estimate=value,interval=ci,seed=20261005,samples=10000,blocks=len(a)),'Bootstrap evidence differs')
    return value,ci

def semantic_score(output,expected,item,native,effective,status):
    terminal=next((s for s in sorted(effective,key=len,reverse=True) if output.endswith(s)),None)
    body=output[:-len(terminal)] if terminal else None
    result=dict(factual_correct=None,strict_correct=False,format_correct=False,parsed_color=None,
        answer_category='malformed',score_reason=None,terminal_suffix=terminal,assistant_body=body)
    if status!='LOGICAL_END_OF_GENERATION':
        result.update(answer_category='truncated' if status=='MAX_TOKENS_REACHED' else 'incomplete',score_reason=status)
    elif terminal is None:result['score_reason']='missing_terminal_token'
    elif not body.strip():result['score_reason']='empty_output'
    elif '<|' in body or '|>' in body or any(s in body for s in native):result['score_reason']='exposed_control_marker'
    else:
        text=' '.join(body.casefold().split()).strip().strip('.,!?;:').strip()
        color=None;one=text in COLORS
        if one:color=text
        else:
            match=re.fullmatch(r'(item[0-9]+) (?:is|=) ([a-z]+)',text.removeprefix('the color of ') if text.startswith('the color of ') and ' = ' not in text else text)
            if match and match[2] in COLORS:
                if match[1]!=item.lower():result['score_reason']='wrong_identifier';return result
                color=match[2]
            elif text.startswith('it is ') and text[6:] in COLORS:color=text[6:]
        if color is None:result['score_reason']='unsupported_or_ambiguous_answer'
        else:result.update(parsed_color=color,factual_correct=color==expected,format_correct=one,
            strict_correct=one and color==expected,answer_category='correct_fact' if color==expected else 'wrong_fact')
    return result

def check_dialogue(event,f,env,sizing):
    result=event['response']['result'];native=env['native_stops'];condition=event['condition']
    effective=native+(['.','\n'] if condition=='punctuation' else [])
    require(result['condition']==condition and result['effective_stops']==effective and result['native_stops']==native,'Stop condition substitution')
    require(result['reset_verified'],'Dialogue reset failed')
    rows=result['rows'];transcript=deepcopy(f['initial_messages']);render=renderer(env['prompt_template'],'llama')
    require([r['turn'] for r in rows]==list(range(len(rows))),'Turn sequence altered')
    for turn,row in enumerate(rows):
        if turn:transcript.append(dict(role='user',content=f['questions'][turn]))
        full=render(transcript);item=f['questions'][turn].split()[-1].rstrip('?')
        expected=semantic_score(row['output'],f['answers'][turn],item,native,effective,row['status'])
        require(row['score']==expected and row['answer_correct']==expected['strict_correct'],'Whole-answer score altered')
        require(row['native_stops']==native and row['effective_stops']==effective and row['condition']==condition,'Stop settings drift')
        require(row['messages']==transcript and row['effective_prompt']==full==row['submitted_prompt'],'Transcript reconstruction differs')
        require(row['parameters']==PARAMETERS and row['expected_answer']==f['answers'][turn],'Parameters or answer changed')
        require(row['output']==''.join(row['stream_chunks']) and row['prompt_sha256']==digest_value(full),'Raw stream/prompt differs')
        require(row['context_before']==0 and row['submitted_token_ids']==row['prompt_token_ids'],'Full rebuild contract differs')
        require(row['prompt_tokens']==len(row['prompt_token_ids']) and row['recovery_token_ids']==env['recovery_token_ids'],'Input token count differs')
        require(row['prompt_tokens']+32+len(row['recovery_token_ids'])<=env['experiment_limit_tokens'],'Token budget overflow')
        if turn==0:require(row['prompt_token_ids']==sizing['token_ids'],'Initial tokenizer evidence differs')
        require(row['expected_context_after']==len(row['after_token_ids']),'Context accounting changed')
        ledger=row['context_after']==len(row['after_token_ids']);require(row['ledger_valid']==ledger,'Context validity changed')
        suffix=expected['terminal_suffix'];body=expected['assistant_body']
        valid=bool(ledger and suffix and body.strip() and row['status']=='LOGICAL_END_OF_GENERATION')
        require(row['terminal_suffix']==suffix and row['assistant_body']==body and row['valid']==valid,'Termination contract differs')
        require(isinstance(row['output_token_ids'],list) and bool(row['terminal_token_ids'])==bool(suffix),'Missing suffix token evidence')
        if not valid:require(turn==len(rows)-1,'Invalid reply continued dialogue')
        else:transcript.append(dict(role='assistant',content=body))
    require(len(rows)+len(result['quarantined_turns'])==4,'Planned denominator changed')
    require(result['quarantined_turns']==list(range(len(rows),4)),'Quarantine altered')

def decisions(rows,config,s):
    stage=config['refine_stage'];blocks=BLOCKS[stage];selected=None;accepted=False
    if stage.startswith('npu'):
        counts={};qualified=False
        for condition in ('native','punctuation'):
            jobs=[e for e in rows if e['condition']==condition];answers=[r for e in jobs for r in e['response']['result']['rows']]
            valid=sum(bool(r['valid']) for r in answers);correct=sum(bool(r['valid'] and r['answer_correct']) for r in answers)
            per={str(n):sum(bool(r['valid'] and r['answer_correct']) for e in jobs if e['target_tokens']==n for r in e['response']['result']['rows']) for n in (128,512,1024)}
            contracts=sum(len(e['response']['result']['contract_failures'])+int(not e['response']['result']['reset_verified']) for e in jobs)
            ok=valid==blocks*12 and correct>=(92 if blocks==8 else 183) and min(per.values())>=(29 if blocks==8 else 58) and contracts==0
            metric=next(m for m in s['metrics'] if m['label']==condition)
            require(metric==dict(label=condition,planned=blocks*12,observed=len(answers),valid=valid,correct=correct,accuracy=correct/(blocks*12),per_size_correct=per,contracts=contracts,qualified=ok),'NPU summary differs')
            counts[condition]=[sum(bool(r['valid'] and r['answer_correct']) for e in jobs if e['block']==b for r in e['response']['result']['rows'])/12 for b in range(blocks)]
            if condition=='punctuation':qualified=ok
        gain,ci=check_interval(s['accuracy_gain'],counts['punctuation'],counts['native'],False)
        ok=qualified and (stage=='npu-develop' or (gain>=.05 and ci[0]>0))
        selected='punctuation' if ok else None;accepted=ok and stage=='npu-confirm'
        decision='candidate_selected' if ok and stage=='npu-develop' else 'termination_improved' if ok else 'no_qualified_candidate'
    else:
        eligible=[]
        for policy in sorted({r['policy'] for r in rows}):
            mine=[r for r in rows if r['policy']==policy];m=next(x for x in s['metrics'] if x['label']==str(policy))
            def means(label,cell=None):return [stats.mean(r['total_ms'] for r in mine if r['block']==b and r['arm']['arm']==label and (cell is None or r['cell_id']==cell)) for b in range(blocks)]
            stable=True
            for cell in [None]+[c['cell_id'] for c in CELLS]:
                _,ci=check_interval(m['cpu_equivalence'][cell or 'aggregate'],means('cpu_duplicate',cell),means('cpu',cell))
                stable=stable and ci[0]>=.95 and ci[1]<=1.05
            averages={label:stats.mean(means(label)) for label in LABELS};cpu=(averages['cpu']+averages['cpu_duplicate'])/2
            require(m['mean_ms']==averages and m['cpu_mean_ms']==cpu and m['stable']==stable and m['correct'],'CPU means/equivalence altered')
            require(m['policy']==POLICIES[policy] and m['gpu_cpu_ratio']==averages['stream64']/cpu,'Policy or CPU comparison changed')
            speed,ci=check_interval(m['gpu_speedup'],means('gpu'),means('stream64'))
            slow=max(stats.mean(means('stream64',c['cell_id']))/stats.mean(means('gpu',c['cell_id'])) for c in CELLS)
            require(m['max_cell_slowdown']==slow,'Per-shape slowdown differs')
            if stable:eligible.append((cpu,POLICIES[policy]['governor']!='ondemand',POLICIES[policy]['binding']!='unbound',POLICIES[policy]['waiting']!='default',policy,speed,ci,slow))
        if eligible:
            chosen=min(eligible);ok=stage!='gpu-confirm' or (chosen[5]>=1.1 and chosen[6][0]>1 and chosen[7]<=1.05)
            if ok:selected=chosen[4];accepted=stage!='cpu-develop'
        decision=('candidate_selected' if stage=='cpu-develop' else 'cpu_calibrated' if stage=='cpu-confirm' else 'gpu_improved') if selected is not None else 'no_qualified_candidate'
    require(s['selected']==selected and s['accepted']==accepted and s['decision']==decision,'Eligibility decision differs')

def audit(directory,require_seal=True):
    directory=Path(directory);checks=[]
    try:
        if require_seal:verify_artifacts(directory);checks.append('artifact seal')
        config=read(directory/'config.json');stage=config['refine_stage'];cpu=config['phase']=='cpu'
        require(read(directory/'protocol.json')==specification(stage),'Protocol changed')
        manifest=read(directory/'manifest.json');freeze=read(directory/'freeze.json')
        require(freeze['source_sha256']==manifest['source_sha256'],'Source freeze differs')
        for p,h in manifest['source_sha256'].items():require(digest_file(directory/'source'/p)==h,'Archived source changed')
        require(freeze['config_sha256']==digest_file(directory/'config.json') and freeze['protocol_sha256']==digest_file(directory/'protocol.json'),'Config freeze changed')
        source=Path(freeze['source_campaign']);verify_artifacts(source)
        require(digest_file(source/'checksums.json')==freeze['source_campaign_sha256'],'Source campaign changed')
        if freeze['parent']:
            parent=Path(freeze['parent']['path']);verify_artifacts(parent)
            require(digest_file(parent/'checksums.json')==freeze['parent']['checksums_sha256'] and read(parent/'manifest.json')['source_sha256']==manifest['source_sha256'],'Held-out source/parent changed')
            require(read(parent/'validation.json')['passed'] and read(parent/'summary.json')['selected']==config['selected'],'Unqualified parent')
        checks.append('source, configuration and parent freeze')
        fs=read(directory/'fixtures.json');specs=fixtures(config);prior=read(directory/'prior-fixtures.json')
        require(len(fs)==len(specs),'Fixture count differs');seen=set();seen_hash=set()
        for f,spec in zip(fs,specs):
            require(all(f[k]==v for k,v in spec.items()),'Fixture spec differs')
            require(f['seed'] not in prior['seeds'] and f['seed'] not in seen,'Seed reused');seen.add(f['seed'])
            h=f['input_sha256'] if cpu else f['table_sha256']
            require(h not in prior['hashes'] and h not in seen_hash,'Input reused');seen_hash.add(h)
            if cpu:
                p=directory/'fixtures'/(f['fixture_id']+'.npz');require(digest_file(p)==f['file_sha256'],'Fixture file changed')
                x,w=fixture(f['n'],f['d'],f['b'],f['seed']);expected=oracle(x,w)
                with np.load(p,allow_pickle=False) as data:require(np.array_equal(data['x'],x) and np.array_equal(data['w'],w) and np.array_equal(data['expected'],expected),'FP64 reconstruction failed')
                require((input_hash(x),input_hash(w),input_hash(expected))==(f['input_sha256'],f['weights_sha256'],f['oracle_sha256']),'Array digest changed')
            else:require(f==make_table(spec,len(f['table'])),'Table reconstruction differs')
        byid={f['fixture_id']:f for f in fs};jobs=schedule(config,fs);require(read(directory/'schedule.json')==jobs,'Schedule changed')
        events,damaged=read_jsonl(directory/EVENTS);require(not damaged,'Damaged events')
        frozen=[e for e in events if e['event']=='fixtures_frozen'];require(len(frozen)==1 and frozen[0]['fixtures_sha256']==digest_file(directory/'fixtures.json') and frozen[0]['schedule_sha256']==digest_file(directory/'schedule.json'),'Freeze evidence differs')
        observations=[e for e in events if e['event']=='measurement'];require(len(observations)==len(jobs),'Measurement matrix incomplete')
        active=None;env=None;warm={};environments={};anchor=None;checked={}
        sizes=read(directory/'sizing.json') if not cpu else None
        if not cpu:
            for f in fs:
                size=sizes[f['fixture_id']]
                require(size['count']==len(f['table']) and len(size['token_ids'])<=f['target_tokens'] and (size['count']==300 or len(size['next_row_token_ids'])>f['target_tokens']),'Largest-fit sizing differs')
        for e in events:
            kind=e['event']
            if kind=='worker_ready':
                require(active is None,'Overlapping workers');active=e;env=e['environment'];environments[e['instance']]=env
                if cpu:
                    p=env['policy'];require(p in POLICIES and env['openmp_environment']==environment(p),'Policy launch differs')
                    require(env['observed']['governor']==p['governor'],'Governor readback differs')
                    require([r['requested'] for r in env['probes']]==[1,4],'Probe team matrix differs')
                    for probe in env['probes']:
                        n=probe['requested'];require(probe['team']==n and len(probe['threads'])==n,'Measured team differs')
                        require([r['mask'] for r in probe['threads']]==([1<<i for i in range(n)] if p['binding']=='bound' else [15]*n),'Measured affinity differs')
                else:require(env['hailort_version']=='5.1.1' and env['model_sha256']==MODEL_HASHES['llama'] and env['parameters']==PARAMETERS and set(env['native_stops'])==set(specification(stage)['native_stops']),'Model or decoding drift')
            elif kind=='worker_release':
                require(active and e['instance']==active['instance'] and e['owner']==active['owner'] and not e['alive'] and not e['forced'],'Unclean worker release');active=None
            elif kind in ('measurement','warmup','anchor'):
                require(active and e['response']['owner']==active['owner'],'Request owner differs')
                r=e['response']['result']
                if cpu:
                    require(r['policy']==POLICIES[e['policy']]==env['policy'],'Policy substitution')
                    for key in ('system_before','system_after'):require(r[key]['governor']==env['policy']['governor'],'Governor changed')
                    f=byid[e['fixture_id']];key=(e['fixture_id'],r['output_file'])
                    if key not in checked:
                        with np.load(directory/'fixtures'/(f['fixture_id']+'.npz')) as data:expected=data['expected']
                        out=np.load(directory/r['output_file']);checked[key]=(errors(out,expected),input_hash(out))
                    check,h=checked[key];require(h==r['output_sha256'] and all(r[k]==v for k,v in check.items()) and r['correct'] and r['validation_errors']==0,'Numerical output differs')
                    if kind=='warmup':warm[active['instance'],e['fixture_id'],e['arm']['arm']]=h
                    else:
                        require(r['matches_warmup'] and not r['unexpected_output_io'] and warm[active['instance'],e['fixture_id'],r['arm']]==h,'Warmup/output mismatch')
                        if kind=='anchor':
                            require(r['arm']=='stream64' and r['variant']==4,'Anchor substitution')
                            anchor=(e['fixture_id'],e['policy'],e['repeat'])
                        else:
                            require(anchor==(e['fixture_id'],e['policy'],e['repeat']),'Missing sequence anchor')
                            require(all(r[k]==v for k,v in e['arm'].items()),'Numeric arm differs')
                elif kind=='measurement':check_dialogue(e,byid[e['fixture_id']],env,sizes[e['fixture_id']])
        require(active is None,'Unreleased worker')
        if not cpu:
            restored=[e for e in events if e['event']=='native_stops_restored']
            require(len(restored)==1 and restored[0]['response']['result']==dict(stops=env['native_stops'],context_tokens=0),'Native stops/context not restored')
        for e,job in zip(observations,jobs):
            require(all(e[k]==v for k,v in job.items()),'Measured order differs')
            require(e['started']<=e['response']['submitted']<=e['response']['received']<=e['response']['ended']<=e['ended'],'Clock/IPC order differs')
            require(e['total_ms']==(e['ended']-e['started'])*1000,'Request timing changed')
        if cpu:
            require(len([e for e in events if e['event']=='anchor'])==len(jobs)//4,'Anchor count differs')
            require(len([e for e in events if e['event']=='warmup'])==len(jobs)//3,'Warmup count differs')
            build=read(directory/'native-build/build.json')
            for p,h in build['artifacts'].items():require(digest_file(directory/'native-build'/p)==h,'Binary artifact changed')
            for p,h in build['sources'].items():require(manifest['source_sha256'][p]==h,'Binary source differs')
            controls=read(directory/'correctness.json');require(len(controls)==32,'Control matrix incomplete')
            for c in controls:
                with np.load(directory/c['input_file']) as data:expected=oracle(data['x'],data['w'])
                if c['kind']=='oracle':require(errors(np.load(directory/c['output_file']),expected)['correct'] and c['validation_errors']==0,'FP64 control failed')
                else:
                    out={k:np.load(directory/v) for k,v in c['output_files'].items()};cut=c['cut']
                    require(np.array_equal(out['baseline'],out['repeated']) and np.array_equal(out['baseline'][0,:cut],out['perturbed'][0,:cut]) and np.array_equal(out['baseline'][1:],out['perturbed'][1:]),'Causality/isolation/reset failed')
        checks.append('fresh fixtures, schedule, ownership, raw outputs and contracts')
        gates=[e for e in events if e['event']=='pilot_gate'];require(len(gates)==1,'Pilot decision missing')
        g=gates[0];require(g['multiplier']==BLOCKS[stage] and g['required_seconds']==g['pilot_seconds']*BLOCKS[stage]*1.2 and g['fits'] and g['required_seconds']<=g['remaining_work_seconds'],'Pilot projection failed')
        summary=read(directory/'summary.json');require(summary['complete'] and any(e['event']=='measurement_complete' for e in events),'Incomplete stage')
        decisions([r for r in observations if r['section']=='main'],config,summary);checks.append('independent scores, block bootstrap and eligibility')
        outcome=read(directory/'outcome.json');require(outcome['status']=='complete' and outcome['elapsed_seconds']<=config['max_seconds'],'Supervised outcome/budget failed')
        restoration=read(directory/'governor-restoration.json');require(restoration['restored'] and restoration['current']==config['original_governor'],'Governor not restored')
        telemetry,damaged=read_jsonl(directory/'telemetry.jsonl');require(not damaged and telemetry and all(not r.get('stop_reason') for r in telemetry),'Telemetry failure')
        checks.append('budget, telemetry and governor restoration')
        return dict(passed=True,checks=checks,measurements=len(observations),fixtures=len(fs),offline_limit='Tokenizer IDs are checked for recorded consistency; hardware tokenizer replay is separate.')
    except Exception as exc:return dict(passed=False,checks=checks,error=f'{type(exc).__name__}: {exc}')
