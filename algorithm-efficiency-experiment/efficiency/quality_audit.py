"""Read-only reconstruction of quality fixtures, requests, coverage and decisions.

Decision and score reconstruction deliberately do not call the report's helpers.
Token IDs are checked as recorded evidence; no NPU inference is needed to audit.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

from .common import digest_file, digest_value, read_jsonl
from .diagnostic import renderer
from .hybrid import verify_artifacts
from .quality_spec import EVENTS, specification

COLORS=('blue','green','red','yellow','white','black','pink','brown')
PROMPTS={'legacy':'Answer using the reference facts. Keep answers to one word.',
         'explicit':'Use only the reference facts. Reply with exactly one lowercase color: blue, green, red, yellow, white, black, pink, or brown. Output no other words.'}


def independent_score(output, expected, item, stops, status):
    if status!='LOGICAL_END_OF_GENERATION':return False,None,False,None
    endings=[s for s in stops if output.endswith(s)]
    if not endings:return False,None,False,None
    body=output[:-len(endings[0])]
    if '<|' in body or '|>' in body or any(s in body for s in stops):return False,None,False,body
    text=' '.join(body.casefold().split()).strip().strip('.,!?;:').strip()
    if text in COLORS:return text==expected,text==expected,True,body
    patterns=(rf'the color of {item} is (\w+)',rf'{item} is (\w+)',rf'{item} = (\w+)',r'it is (\w+)')
    for pattern in patterns:
        match=re.fullmatch(pattern,text)
        if match and match[1] in COLORS:return False,match[1]==expected,False,body
    return False,None,False,body


def audit(directory, require_seal=True):
    directory=Path(directory);issues=[]
    def require(ok,message):
        if not ok:issues.append(message)
    def read(name):return json.loads((directory/name).read_text())
    try:
        verified=verify_artifacts(directory) if require_seal else {}
        if require_seal:
            files={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name!='checksums.json'}
            require(files==set(verified),'Manifest coverage differs')
        config=read('config.json');stage=config['quality_stage'];nblocks=8 if stage=='develop' else 16
        require(stage in ('develop','confirm'),'Unknown stage')
        require(read('protocol.json')==specification(stage),'Frozen protocol differs')
        frozen=read('freeze.json');manifest=read('manifest.json')
        require(manifest['config']==config,'Manifest configuration differs')
        require(frozen['config_sha256']==digest_file(directory/'config.json'),'Frozen config differs')
        require(frozen['protocol_sha256']==digest_file(directory/'protocol.json'),'Frozen protocol hash differs')
        require(manifest['source_sha256']==frozen['source_sha256'],'Frozen sources differ')
        for path,sha in manifest['source_sha256'].items():
            require(digest_file(directory/'source'/path)==sha,f'Source snapshot differs: {path}')
        require(manifest['model_sha256']=='5310176848638505fbc28add04ba60c97abe345cdb0ec7e3b8ffaa4b0a8c65dd','Model differs')
        environment=read('quality-environment.json');baseline=read('baseline-environment.json')
        require(frozen['baseline_environment_sha256']==digest_file(directory/'baseline-environment.json'),'Baseline hash differs')
        for key in ('hailort_version','model_sha256','parameters','model_defaults','stop_tokens','prompt_template','capacity_tokens','experiment_limit_tokens'):
            require(environment[key]==baseline[key],f'Environment differs: {key}')
        require(environment['hailort_version']=='5.1.1' and environment['experiment_limit_tokens']==1792,'Runtime or context limit differs')
        require(environment['parameters']['max_generated_tokens']==16 and environment['parameters']['do_sample'] is False
                and environment['parameters']['seed']==12345 and environment['parameters']['frequency_penalty']==1.0,'Baseline decoding differs')
        if stage=='confirm':
            require(read('development-summary.json')['selected']==config['selected'],'Development selection differs')
            require(frozen['parent']['summary_sha256']==digest_file(directory/'development-summary.json'),'Frozen parent selection hash differs')
        possible={f'{p}-{t}':dict(id=f'{p}-{t}',prompt_id=p,prompt=PROMPTS[p],max_generated_tokens=t)
                  for p in PROMPTS for t in (16,32,64)}
        arms=([dict(v,label=k) for k,v in possible.items()] if stage=='develop' else
              [dict(possible['legacy-16'],label='control'),dict(possible[config['selected']],label='candidate')])
        require(config['conditions']==arms,'Condition matrix differs')
        specs=[]
        for section,blocks in (('pilot',[-1]),('main',range(nblocks))):
            for block in blocks:
                for j,size in enumerate((128,512,1024)):
                    fid=f'{section}-n{size}-b{block}'
                    seed=int.from_bytes(hashlib.sha256(f"npu-quality-v1|{config['fixture_namespace']}|{stage}|{fid}".encode()).digest()[:8],'little')
                    specs.append(dict(fixture_id=fid,section=section,block=block,target_tokens=size,seed=seed,
                                      first_color=('brown','green','blue')[j] if section=='pilot' else COLORS[block%8]))
        fixtures=read('quality-fixtures.json');require(len(fixtures)==len(specs),'Fixture count differs')
        prior=read('prior-fixtures.json');seeds=set(prior['seeds']);hashes=set(prior['hashes'])
        render=renderer(environment['prompt_template']);by_id={}
        def initial(table,p):
            facts='; '.join(f'{k} = {v}' for k,v in table)
            return [dict(role='system',content=PROMPTS[p]),dict(role='user',content=f'Reference facts: {facts}. What color is {table[0][0]}?')]
        for f,spec in zip(fixtures,specs):
            require(all(f[k]==v for k,v in spec.items()),'Fixture specification differs')
            require(f['seed'] not in seeds and f['table_sha256'] not in hashes,'Fixture is not fresh')
            seeds.add(f['seed']);hashes.add(f['table_sha256'])
            rng=random.Random(f['seed']);rows=[[f'item{i:03d}',rng.choice(COLORS)] for i in range(1,301)]
            rows[0][1]=f['first_color'];n=len(f['table'])
            require(4<=n<=300 and f['table']==rows[:n] and digest_value(f['table'])==f['table_sha256'],'Fixture table differs')
            ids=(0,n//2,n-1,0)
            require(f['questions']==[f'What color is {rows[i][0]}?' for i in ids]
                    and f['answers']==[rows[i][1] for i in ids],'Question or expected answer differs')
            require(set(f['initial_token_ids'])==set(PROMPTS),'Prompt sizing evidence missing')
            require(max(map(len,f['initial_token_ids'].values()))<=f['target_tokens'],'Initial target exceeded')
            require(n==300 or max(map(len,f['next_row_token_ids'].values()))>f['target_tokens'],'Table is not maximal')
            by_id[f['fixture_id']]=f
        jobs=[]
        for i,f in enumerate(fixtures):
            index=i if f['section']=='pilot' else i-3
            offset=index%len(arms);order=arms[offset:]+arms[:offset]
            if (index//len(arms))%2:order=list(reversed(order))
            jobs.extend(dict(fixture_id=f['fixture_id'],section=f['section'],block=f['block'],
                             target_tokens=f['target_tokens'],condition=a) for a in order)
        require(read('quality-schedule.json')==jobs,'Frozen schedule differs')
        events,errors=read_jsonl(directory/EVENTS);require(not errors,'Damaged event lines')
        starts=[e for e in events if e['event']=='dialogue_start'];observed=[e for e in events if e['event']=='dialogue']
        require([{k:r[k] for k in jobs[0]} for r in starts]==jobs,'Started job coverage/order differs')
        require([{k:r[k] for k in jobs[0]} for r in observed]==jobs,'Result coverage/order differs')
        ready=[e for e in events if e['event']=='worker_ready'];released=[e for e in events if e['event']=='worker_release']
        require(len(ready)==1 and ready[0]['environment']==environment,'Worker environment differs')
        require(len(released)==1 and not released[0]['alive'] and not released[0]['forced'],'Worker release failed')
        freezes=[e for e in events if e['event']=='fixtures_frozen']
        require(len(freezes)==1 and freezes[0]['fixtures_sha256']==digest_file(directory/'quality-fixtures.json')
                and freezes[0]['schedule_sha256']==digest_file(directory/'quality-schedule.json'),'Fixture freeze differs')
        gates=[e for e in events if e['event']=='pilot_gate']
        require(len(gates)==1 and gates[0]['fits'] and gates[0]['multiplier']==nblocks
                and math.isclose(gates[0]['required_seconds'],gates[0]['pilot_seconds']*nblocks*1.2)
                and gates[0]['required_seconds']<=gates[0]['remaining_work_seconds'],'Pilot gate differs')
        require(any(e['event']=='measurement_complete' for e in events),'Missing completion')
        require(read('outcome.json')['status']=='complete','Supervisor incomplete')
        independent=[]
        for job in observed:
            f=by_id[job['fixture_id']];arm=job['condition'];response=job['response'];result=response['result']
            require(response['owner']==ready[0]['owner'],'Worker ownership differs')
            require(result['reset_verified'],'Dialogue reset failed')
            messages=initial(f['table'],arm['prompt_id']);old_ids=[];accumulated='';scores=[]
            require([r['turn'] for r in result['rows']]==list(range(len(result['rows']))),'Turn sequence differs')
            for turn,r in enumerate(result['rows']):
                if turn:messages.append(dict(role='user',content=f['questions'][turn]))
                full=render(messages)
                require(full.startswith(accumulated) and r['effective_prompt']==full and r['submitted_prompt']==full
                        and r['prompt_sha256']==digest_value(full),'Prompt reconstruction differs')
                require(r['expected_answer']==f['answers'][turn] and r['condition']==arm,'Response target or condition differs')
                expected_params=dict(environment['parameters'],max_generated_tokens=arm['max_generated_tokens'])
                require(r['parameters']==expected_params,'Effective output-token budget differs')
                require(r['history_token_ids']==old_ids and r['context_before']==len(old_ids),'Before-request token ledger differs')
                require(r['prompt_token_ids'][:len(old_ids)]==old_ids,'History token prefix differs')
                require(r['submitted_token_ids']==r['prompt_token_ids'] and r['submitted_tokens']==r['prompt_tokens']==len(r['prompt_token_ids']),'Submitted tokens differ')
                if turn==0:require(r['prompt_token_ids']==f['initial_token_ids'][arm['prompt_id']],'Initial token evidence differs')
                require(r['recovery_tokens']==len(environment['recovery_token_ids']) and
                        r['prompt_tokens']+arm['max_generated_tokens']+r['recovery_tokens']<=1792,'Reserved context allowance exceeded')
                require(r['expected_context_after']==len(r['after_token_ids']) and
                        r['ledger_valid']==(r['context_after']==r['expected_context_after']),'After-request ledger differs')
                require(r['after_token_ids'][:len(r['prompt_token_ids'])]==r['prompt_token_ids'],'Output token prefix differs')
                strict,factual,formatted,body=independent_score(r['output'],f['answers'][turn],re.search(r'item[0-9]+',f['questions'][turn])[0],environment['stop_tokens'],r['status'])
                terminal=any(r['output'].endswith(s) for s in environment['stop_tokens'])
                valid=r['ledger_valid'] and terminal and r['status']=='LOGICAL_END_OF_GENERATION'
                require(r['valid']==valid and r['answer_correct']==strict,'Validity or strict score differs')
                require(r['score']['factual_correct']==factual and r['score']['format_correct']==formatted,'Factual or format score differs')
                category=('truncated' if r['status']=='MAX_TOKENS_REACHED' else 'incomplete') if r['status']!='LOGICAL_END_OF_GENERATION' else (
                    'malformed' if factual is None else ('correct_fact' if factual else 'wrong_fact'))
                require(r['score']['answer_category']==category,'Failure category differs')
                require(r['assistant_body']==body if valid else True,'Assistant content differs')
                require(r['request_ms']>=0 and math.isclose(r['request_ms'],(r['ended']-r['started'])*1000,abs_tol=1e-6),'Request timing differs')
                require(''.join(r['stream_chunks'])==r['output'],'Stream reconstruction differs')
                scores.append(bool(valid and strict))
                if not valid:require(turn==len(result['rows'])-1,'Generation continued after failure')
                accumulated=full+r['output'];old_ids=r['after_token_ids']
                messages.append(dict(role='assistant',content=body))
            missing=list(range(len(result['rows']),4))
            require(result['quarantined_turns']==missing,'Quarantine accounting differs')
            independent.append(dict(job=job,scores=scores,valid=sum(bool(r['valid']) for r in result['rows']),
                                    contracts=len(result['contract_failures'])+(not result['reset_verified'])))
        summary=read('summary.json');candidates=[];qualifications={}
        require(summary['complete'] and not summary['event_errors'],'Report completion differs')
        for arm in arms:
            values=[v for v in independent if v['job']['section']=='main' and v['job']['condition']['label']==arm['label']]
            correct=sum(sum(v['scores']) for v in values);valid=sum(v['valid'] for v in values)
            sizes=[sum(sum(v['scores']) for v in values if v['job']['target_tokens']==s) for s in (128,512,1024)]
            qualified=(len(values)==nblocks*3 and valid==nblocks*12 and correct >= (92 if stage=='develop' else 183)
                       and min(sizes)>=(29 if stage=='develop' else 58) and not any(v['contracts'] for v in values))
            qualifications[arm['label']]=qualified
            metric=next(m for m in summary['metrics'] if m['condition']['label']==arm['label'])
            require(metric['condition']==arm and metric['planned']==nblocks*12 and metric['valid']==valid
                    and metric['correct']==correct and metric['qualified']==qualified,'Aggregate quality gate differs')
            require(metric['accuracy']==correct/(nblocks*12),'Planned accuracy denominator differs')
            for size in (128,512,1024):
                group=[v for v in values if v['job']['target_tokens']==size]
                expected=dict(planned=nblocks*4,observed=sum(len(v['scores']) for v in group),
                              valid=sum(v['valid'] for v in group),correct=sum(sum(v['scores']) for v in group))
                require(metric['per_size'][str(size)]==expected,'Per-size quality denominator differs')
            complete=[v['job']['total_ms'] for v in values if v['valid']==4]
            mean=statistics.mean(complete) if complete else None
            require(metric['mean_dialogue_ms']==mean,'Complete dialogue timing differs')
            if qualified:candidates.append((-correct,mean,arm['max_generated_tokens'],arm['prompt_id']!='legacy',arm['id']))
        if stage=='develop':
            selected=min(candidates)[-1] if candidates else None
            require(summary['selected']==selected,'Development selection differs')
            require(summary['decision']==('candidate_selected' if selected else 'no_qualified_condition'),'Development decision differs')
        else:
            deltas=[]
            for block in range(16):
                totals={label:sum(sum(v['scores']) for v in independent if v['job']['section']=='main'
                    and v['job']['block']==block and v['job']['condition']['label']==label) for label in ('control','candidate')}
                deltas.append((totals['candidate']-totals['control'])/12)
            rng=random.Random(20261003)
            draws=sorted(statistics.mean(rng.choices(deltas,k=16)) for _ in range(10000))
            interval=summary['interval']
            require(math.isclose(interval['difference'],statistics.mean(deltas),abs_tol=1e-12)
                    and all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(interval['interval'],[draws[250],draws[9750]])),'Bootstrap interval differs')
            qualified=qualifications['candidate']
            changed=config['selected']!='legacy-16'
            accepted=bool(changed and qualified and statistics.mean(deltas)>=.05 and draws[250]>0)
            require(summary['changed']==changed and summary['accepted']==accepted,'Confirmation decision differs')
            require(summary['qualified_unchanged_control']==(qualified and not changed),'Unchanged control gate differs')
        return dict(passed=not issues,issues=issues,stage=stage,verified_artifacts=len(verified),
                    fixtures=len(fixtures),dialogues=len(observed),measured_answers=sum(len(v['scores']) for v in independent),
                    auditor_sha256=digest_file(Path(__file__)))
    except (ValueError,KeyError,TypeError,OSError,IndexError,StopIteration) as exc:
        issues.append(f'{type(exc).__name__}: {exc}')
        return dict(passed=False,issues=issues,auditor_sha256=digest_file(Path(__file__)))


if __name__=='__main__':
    result=audit(sys.argv[1]);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
