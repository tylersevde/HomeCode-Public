"""Read-only reconstruction of frozen fixtures, outputs, context and decisions."""
from copy import deepcopy
import json
from pathlib import Path
import statistics
import numpy as np
from .attention_native import fixture,oracle,errors
from .common import digest_file,digest_value,read_jsonl
from .diagnostic import score_answer
from .feedback_spec import input_hash
from .hat import assistant_content
from .hybrid import verify_artifacts
from .study_spec import EVENTS,MODEL_HASHES,PARAMETERS,fixtures,schedule,make_table,specification
from .study_worker import renderer


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def interval(a,b,ratio):
    # Independently reconstruct the declared block bootstrap, not cached decisions.
    rng=np.random.Generator(np.random.PCG64(20261004));indices=rng.integers(len(a),size=(10000,len(a)))
    a,b=np.array(a),np.array(b);aa=a[indices].mean(1);bb=b[indices].mean(1)
    samples=np.sort(aa/bb if ratio else aa-bb)
    return float(a.mean()/b.mean() if ratio else a.mean()-b.mean()),[float(samples[250]),float(samples[9750])]


def check_decision(events,config,summary):
    rows=[e for e in events if e['event']=='measurement' and e['section']=='main'];blocks=8 if config['study_stage']=='develop' else 16
    confirm=config['study_stage']=='confirm'
    selected=None
    if config['track']=='gpu-stream':
        labels=['cpu','cpu_duplicate','gpu']+([config['selected']] if confirm else ['stream32','stream64'])
        cells=list(dict.fromkeys(e['cell_id'] for e in rows));means={};blocksums={}
        for label in labels:
            mine=[e for e in rows if e['arm']['arm']==label]
            require(len(mine)==blocks*18,'GPU planned denominator mismatch')
            means[label]={c:statistics.mean(e['total_ms'] for e in mine if e['cell_id']==c) for c in cells}
            blocksums[label]=[statistics.mean(e['total_ms'] for e in mine if e['block']==b) for b in range(blocks)]
            saved=next(m for m in summary['metrics'] if m['label']==label)
            require(saved['per_cell_ms']==means[label] and saved['mean_ms']==statistics.mean(means[label].values()),'GPU mean altered')
        ratio,ci=interval(blocksums['cpu_duplicate'],blocksums['cpu'],True)
        stable=not(abs(ratio-1)>.05 and (ci[0]>1 or ci[1]<1));require(summary['stable']==stable,'Control stability altered')
        eligible=[]
        for label in labels[3:]:
            speed,ci=interval(blocksums['gpu'],blocksums[label],True);slow=max(means[label][c]/means['gpu'][c] for c in cells)
            qualifies=stable and speed>=1.10 and slow<=1.05 and (not confirm or ci[0]>1)
            metric=next(m for m in summary['metrics'] if m['label']==label)
            require(metric['speedup']['estimate']==speed and metric['speedup']['interval']==ci and metric['qualified']==qualifies,'GPU gate altered')
            if qualifies:eligible.append((statistics.mean(means[label].values()),label!='stream32',label))
        if eligible:selected=min(eligible)[2]
        decision=('gpu_improved' if confirm else 'candidate_selected') if selected else ('no_qualified_candidate' if stable else 'unstable_cpu_control')
    else:
        grouped={};eligible=False
        for model in ('qwen','llama'):
            mine=[e for e in rows if e['model']==model];require(len(mine)==blocks*3,'NPU planned denominator mismatch')
            answers=[r for e in mine for r in e['response']['result']['rows']]
            valid=sum(bool(r['valid']) for r in answers);correct=sum(bool(r['valid'] and r['answer_correct']) for r in answers)
            persize={str(n):sum(bool(r['valid'] and r['answer_correct']) for e in mine if e['target_tokens']==n for r in e['response']['result']['rows']) for n in (128,512,1024)}
            contracts=sum(len(e['response']['result']['contract_failures'])+not_reset(e) for e in mine)
            qualifies=valid==blocks*12 and correct>=(183 if confirm else 92) and min(persize.values())>=(58 if confirm else 29) and contracts==0
            saved=next(m for m in summary['metrics'] if m['label']==model)
            require((saved['valid'],saved['correct'],saved['per_size_correct'],saved['qualified'])==(valid,correct,persize,qualifies),'Model score or gate altered')
            grouped[model]=[sum(bool(r['valid'] and r['answer_correct']) for e in mine if e['block']==b for r in e['response']['result']['rows'])/12 for b in range(blocks)]
            if model=='llama':eligible=qualifies
        gain,ci=interval(grouped['llama'],grouped['qwen'],False)
        require(summary['accuracy_gain']['estimate']==gain and summary['accuracy_gain']['interval']==ci,'Accuracy interval altered')
        if eligible and (not confirm or (gain>=.05 and ci[0]>0)):selected='llama'
        decision=('model_improved' if confirm else 'candidate_selected') if selected else 'no_qualified_candidate'
    require(summary['selected']==selected and summary['accepted']==bool(confirm and selected) and summary['decision']==decision,'Promotion decision altered')


def not_reset(event):return int(not event['response']['result']['reset_verified'])


def check_dialogue(event,f,env):
    result=event['response']['result'];rows=result['rows'];transcript=deepcopy(f['initial_messages'])
    render=renderer(env['prompt_template'],env['model_id']);stops=env['stop_tokens']
    require(result['reset_verified'],'Final reset not verified')
    require([r['turn'] for r in rows]==list(range(len(rows))),'Turn sequence altered')
    for turn,row in enumerate(rows):
        if turn:transcript.append(dict(role='user',content=f['questions'][turn]))
        full=render(transcript);require(row['messages']==transcript and row['effective_prompt']==full and row['submitted_prompt']==full,'Transcript reconstruction failed')
        require(row['parameters']==PARAMETERS and row['expected_answer']==f['answers'][turn],'Decode parameters or expected answer altered')
        require(row['prompt_sha256']==digest_value(full) and row['output']==''.join(row['stream_chunks']),'Prompt/output digest differs')
        require(row['context_before']==0 and row['prompt_token_ids']==row['submitted_token_ids'],'Full rebuild contract failed')
        require(row['prompt_tokens']==len(row['prompt_token_ids']) and row['recovery_token_ids']==env['recovery_token_ids'],'Token ledger altered')
        require(len(row['prompt_token_ids'])+32+len(row['recovery_token_ids'])<=env['experiment_limit_tokens'],'Context overflow')
        require(row['expected_context_after']==len(row['after_token_ids']),'Post-generation token count altered')
        ledger=row['context_after']==len(row['after_token_ids']);require(row['ledger_valid']==ledger,'Ledger validity altered')
        try:body=assistant_content(row['output'],stops);terminal=True
        except ValueError:body=None;terminal=False
        valid=ledger and terminal and row['status']=='LOGICAL_END_OF_GENERATION'
        queried=f['questions'][turn].split()[-1].rstrip('?')
        score=score_answer(row['output'],f['answers'][turn],queried,stops,row['status'])
        require(row['valid']==valid and row['score']==score and row['answer_correct']==score['strict_correct'] and row['assistant_body']==body,'Independent scoring failed')
        if turn==0:require(row['prompt_token_ids']==event['initial_token_ids'],'First prompt tokenizer evidence changed')
        if not valid:require(turn==len(rows)-1,'Invalid dialogue was continued')
        transcript.append(dict(role='assistant',content=body))
    require(result['quarantined_turns']==list(range(len(rows),4)),'Quarantined denominator differs')


def audit(directory,require_seal=True):
    directory=Path(directory);checks=[]
    try:
        if require_seal:verify_artifacts(directory);checks.append('sealed artifacts')
        config=read(directory/'config.json');manifest=read(directory/'manifest.json');freeze=read(directory/'freeze.json')
        require(read(directory/'protocol.json')==specification(config['track'],config['study_stage']),'Protocol altered')
        require(freeze['config_sha256']==digest_file(directory/'config.json') and freeze['protocol_sha256']==digest_file(directory/'protocol.json'),'Freeze altered')
        require(freeze['source_sha256']==manifest['source_sha256'],'Source manifest altered')
        for p,h in manifest['source_sha256'].items():require(digest_file(directory/'source'/p)==h,'Source snapshot changed: '+p)
        if freeze['parent']:
            parent=Path(freeze['parent']['path']);verify_artifacts(parent)
            require(digest_file(parent/'checksums.json')==freeze['parent']['checksums_sha256'],'Parent changed')
        checks.append('source, protocol and parent freeze')
        events,parse_errors=read_jsonl(directory/EVENTS);require(not parse_errors,'Event parse failure')
        fs=read(directory/'fixtures.json');specs=fixtures(config);require(len(fs)==len(specs),'Fixture count mismatch')
        prior=read(directory/'prior-fixtures.json');seen_seeds=set(prior['seeds']);seen_hashes=set(prior['hashes'])
        gpu=config['track']=='gpu-stream';byid={}
        for f,spec in zip(fs,specs):
            require(all(f.get(k)==v for k,v in spec.items()),'Fixture seed/spec altered')
            value=f['input_sha256'] if gpu else f['table_sha256']
            require(f['seed'] not in seen_seeds and value not in seen_hashes,'Fixture reused')
            seen_seeds.add(f['seed']);seen_hashes.add(value);byid[f['fixture_id']]=f
            if gpu:
                path=directory/'fixtures'/(f['fixture_id']+'.npz');require(digest_file(path)==f['file_sha256'],'Fixture file changed')
                with np.load(path,allow_pickle=False) as data:
                    x,w=fixture(f['n'],f['d'],f['b'],f['seed']);expected=oracle(x,w)
                    require(np.array_equal(data['x'],x) and np.array_equal(data['w'],w) and np.array_equal(data['expected'],expected),'Fixture/oracle reconstruction failed')
                    require(input_hash(x)==f['input_sha256'] and input_hash(w)==f['weights_sha256'] and input_hash(expected)==f['oracle_sha256'],'Array digest differs')
            else:require(f==make_table(spec,len(f['table'])),'Fact table reconstruction failed')
        jobs=schedule(config,fs);require(read(directory/'schedule.json')==jobs,'Schedule altered')
        frozen=[e for e in events if e['event']=='fixtures_frozen'];require(len(frozen)==1,'Missing fixture freeze')
        require(frozen[0]['fixtures_sha256']==digest_file(directory/'fixtures.json') and frozen[0]['schedule_sha256']==digest_file(directory/'schedule.json'),'Frozen fixtures changed')
        checks.append('fresh fixtures and complete frozen schedule')
        observations=[e for e in events if e['event']=='measurement'];require(len(observations)==len(jobs),'Measurement matrix incomplete')
        owners={};active=None;environments={};model_environments={}
        for e in events:
            if e['event']=='worker_ready':
                require(active is None,'Overlapping owned workers');active=e['instance'];owners[active]=e['owner']
                env=e['environment'];environments[active]=env
                if not gpu:
                    require(env['model_sha256']==MODEL_HASHES[env['model_id']] and env['parameters']==PARAMETERS and env['hailort_version']=='5.1.1','Model environment drift')
                    stable={k:env[k] for k in ('model_sha256','parameters','prompt_template','stop_tokens','template_bindings','recovery_token_ids','capacity_tokens','experiment_limit_tokens','template_probe_ids')}
                    require(model_environments.setdefault(env['model_id'],stable)==stable,'Native model environment changed between loads')
            elif e['event']=='worker_release':
                require(active==e['instance'] and not e['alive'] and not e['forced'] and e['owner']==owners[active],'Worker not released cleanly');active=None
            elif e['event']=='measurement':
                require(active is not None and e['response']['owner']==owners[active],'Measurement ownership mismatch')
                e['_environment']=environments[active]
        require(active is None,'Worker still active')
        if gpu:
            built=read(directory/'native-build/build.json')
            for p,h in built['artifacts'].items():require(digest_file(directory/'native-build'/p)==h,'Native artifact altered')
            for p,h in built['sources'].items():require(manifest['source_sha256'][p]==h,'Build source mismatch')
            controls=read(directory/'correctness.json');require(len(controls)==32,'Hardware control matrix incomplete')
            for c in controls:
                with np.load(directory/c['input_file'],allow_pickle=False) as data:expected=oracle(data['x'],data['w'])
                if c['kind']=='oracle':require(errors(np.load(directory/c['output_file']),expected)['correct'] and c['correct'] and c['validation_errors']==0,'Oracle control failed')
                else:
                    out={k:np.load(directory/v) for k,v in c['output_files'].items()};cut=c['cut']
                    require(c['correct'] and np.array_equal(out['baseline'],out['repeated']) and np.array_equal(out['baseline'][0,:cut],out['perturbed'][0,:cut]) and np.array_equal(out['baseline'][1:],out['perturbed'][1:]),'Semantic control failed')
            warmups=[e for e in events if e['event']=='warmup']
            require(len(warmups)==len(fs)*(4 if config['study_stage']=='confirm' else 5),'Warmup matrix incomplete')
            warms={(e['fixture_id'],e['arm']['arm']):e['response']['result']['output_sha256'] for e in warmups}
            checked={}
            for e,job in zip(observations,jobs):
                require(all(e[k]==v for k,v in job.items()),'Measurement schedule/order differs')
                r=e['response']['result'];f=byid[e['fixture_id']];key=(f['fixture_id'],r['output_file'])
                require(r['fixture_id']==f['fixture_id'] and all(r[k]==v for k,v in e['arm'].items()),'Numeric arm substitution')
                if key not in checked:
                    with np.load(directory/'fixtures'/(f['fixture_id']+'.npz')) as data:expected=data['expected']
                    out=np.load(directory/r['output_file']);checked[key]=(errors(out,expected),input_hash(out))
                require(checked[key][1]==r['output_sha256']==warms[f['fixture_id'],e['arm']['arm']],'Output digest or warmup differs')
                require(all(r[k]==v for k,v in checked[key][0].items()) and r['correct'] and r['matches_warmup'] and not r['unexpected_output_io'] and r['validation_errors']==0,'Measured numerical verification differs')
        else:
            sizes=read(directory/'sizing.json');evidence=read(directory/'token-evidence.json')
            for f in fs:
                fid=f['fixture_id'];count=len(f['table'])
                require(count==min(sizes[m][fid]['count'] for m in ('qwen','llama')),'Shared row count differs')
                for model in ('qwen','llama'):
                    pre=sizes[model][fid];entry=evidence[fid][model]
                    require(len(pre['token_ids'])<=f['target_tokens'] and (pre['count']==300 or len(pre['next_row_token_ids'])>f['target_tokens']),'Max-fit sizing contract differs')
                    require(entry['count']==count and len(entry['token_ids'])<=f['target_tokens'],'Common tokens exceed target')
                    if pre['count']==count:require(pre['token_ids']==entry['token_ids'] and pre['next_row_token_ids']==entry['next_row_token_ids'],'Sizing token evidence changed')
            for e,job in zip(observations,jobs):
                require(all(e[k]==v for k,v in job.items()),'Model schedule/order differs')
                e['initial_token_ids']=evidence[e['fixture_id']][e['model']]['token_ids']
                check_dialogue(e,byid[e['fixture_id']],e['_environment'])
        checks.append('owned device lifecycle, raw outputs and numerical/context contracts')
        gates=[e for e in events if e['event']=='pilot_gate'];require(len(gates)==1,'Missing pilot decision')
        gate=gates[0];require(gate['multiplier']==(8 if config['study_stage']=='develop' else 16),'Pilot multiplier altered');require(gate['fits'] and gate['required_seconds']==gate['pilot_seconds']*gate['multiplier']*1.2 and gate['required_seconds']<=gate['remaining_work_seconds'],'Pilot projection failed')
        for e in observations:
            require(e['started']<=e['response']['submitted']<=e['response']['received']<=e['response']['ended']<=e['ended'],'Request clock/IPC order differs')
            require(e['total_ms']==(e['ended']-e['started'])*1000,'Request clock altered')
        summary=read(directory/'summary.json');require(summary['complete'] and any(e['event']=='measurement_complete' for e in events),'Study incomplete')
        check_decision(events,config,summary);checks.append('independent score, paired bootstrap and promotion decision')
        outcome=read(directory/'outcome.json');require(outcome['status']=='complete' and outcome['elapsed_seconds']<=config['max_seconds'],'Supervised completion or budget failed')
        telemetry,errs=read_jsonl(directory/'telemetry.jsonl');require(not errs and telemetry and all(not r.get('stop_reason') for r in telemetry),'Telemetry stop or corruption')
        checks.append('supervisor outcome and bounded hardware time')
        return dict(passed=True,checks=checks,measurements=len(observations),fixtures=len(fs),offline_limit='Recorded tokenizer IDs verified for consistency; tokenizer replay requires the HAT. No hidden-token offsets assumed.')
    except Exception as exc:return dict(passed=False,checks=checks,error=f'{type(exc).__name__}: {exc}')
