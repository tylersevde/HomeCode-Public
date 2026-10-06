"""Independent offline reconstruction from fixtures, stored outputs and raw events."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

import numpy as np

from .common import atomic_json, digest_file, read_jsonl


def interval(pairs, seed, count=10000):
    rng=random.Random(seed)
    ratio=lambda p:sum(x[0] for x in p)/sum(x[1] for x in p)
    samples=sorted(ratio([pairs[int(rng.random()*len(pairs))] for _ in pairs]) for _ in range(count))
    return ratio(pairs), [samples[int(.025*count)],samples[int(.975*count)]]


def raw_advice(response, case, env):
    r=response['result']
    terminals=[s for s in sorted(env['stop_tokens'],key=len,reverse=True) if r['output'].endswith(s)]
    body=r['output'][:-len(terminals[0])].strip() if terminals else ''
    valid=bool(terminals and re.fullmatch(r'C[0-7]',body) and body in case['eligible'] and
        r['completion_status']=='LOGICAL_END_OF_GENERATION' and r['token_ledger_valid'] is True and
        r['context_before']==0 and r['context_after']==r['expected_context_after'])
    if r['messages']!=case['messages'] or r['effective_parameters']!=env['parameters']:
        raise ValueError('Adviser prompt or settings changed')
    if r['choice'] != (body if valid else None):
        raise ValueError('Stored adviser classification differs from raw response')
    return valid


def audit(directory):
    directory=Path(directory).resolve()
    read=lambda n:json.loads((directory/n).read_text())
    issues=[]
    def require(ok, reason):
        if not ok: issues.append(reason)
    checks=read('checksums.json')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'checksums.json'}
    require(actual==set(checks),'Artifact inventory does not match files')
    for name, checksum in checks.items():
        p=(directory/name).resolve()
        require(p.is_relative_to(directory) and p.is_file() and digest_file(p)==checksum,f'Checksum mismatch: {name}')
    rows, damaged=read_jsonl(directory/'coordination.jsonl')
    require(not damaged,'Damaged observations')
    protocol, summary, outcome=read('protocol.json'),read('summary.json'),read('outcome.json')
    require(outcome['status']=='complete' and summary['complete'],'Run is incomplete')
    require(outcome['elapsed_seconds']<=read('config.json')['max_seconds'],'Runtime budget exceeded')
    require(protocol['cpu_backend']=='native4' and protocol['gpu_backend']=='C6','Frozen controls changed')
    require(not any(r['event']=='incomplete' for r in rows),'Incomplete event present')
    provenance=read('source-provenance.json')
    for name, checksum in provenance['frozen_sha256'].items():
        require(digest_file(directory/'source-run'/name)==checksum,f'Frozen provenance changed: {name}')
    build=read('native-build/build.json')
    require(build==read('source-run/native-build/build.json'),'Native build differs from source')
    for name,checksum in build['artifacts'].items():
        require(digest_file(directory/'native-build'/name)==checksum,f'Native binary changed: {name}')
    manifest=read('manifest.json')
    for name,checksum in manifest['source_sha256'].items():
        require(digest_file(directory/'source'/name)==checksum,f'Measurement source snapshot changed: {name}')
    fixtures=read('fixtures.json'); by_fixture={r['fixture_id']:r for r in fixtures}
    require(len(fixtures)==21 and len(by_fixture)==21,'Fixture coverage differs')
    previous={r['seed'] for r in read('source-run/fixtures.json')}
    require(len({r['seed'] for r in fixtures})==21 and not previous.intersection(r['seed'] for r in fixtures),'Seeds reused')
    from .cpu import reference_module
    reference=reference_module()
    expected_outputs={}
    for row in fixtures:
        path=directory/'fixtures'/(row['fixture_id']+'.npz')
        require(digest_file(path)==row['file_sha256'],f'Fixture damaged: {row["fixture_id"]}')
        with np.load(path,allow_pickle=False) as f:
            rng=np.random.default_rng(row['seed'])
            x=rng.normal(size=(4,1024,64)).astype(np.float32)
            w=(rng.normal(size=(3,64,64))/np.sqrt(64)).astype(np.float32)
            require(np.array_equal(f['x'],x) and np.array_equal(f['w'],w),'Fixture generation mismatch')
            expected=np.stack([reference.cached_stream(s.astype(np.float64),w.astype(np.float64)) for s in x])
            require(np.array_equal(f['expected'],expected),'Original FP64 reference differs')
            expected_outputs[row['fixture_id']]=expected
            for key,value in (('input_sha256',x),('weights_sha256',w),('oracle_sha256',expected)):
                require(hashlib.sha256(value.tobytes()).hexdigest()==row[key],f'Array hash mismatch: {key}')
    cases={c['case_id']:c for c in read('advice-cases.json')};env=read('adviser-environment.json')
    freezes=[i for i,r in enumerate(rows) if r['event']=='adviser_frozen']
    holdout_positions=[i for i,r in enumerate(rows) if r['event']=='advice_check' and r['split']=='holdout']
    dev_positions=[i for i,r in enumerate(rows) if r['event']=='advice_check' and r['split']=='development']
    require(len(freezes)==1 and len(holdout_positions)==20 and len(dev_positions)==20 and
            max(dev_positions)<freezes[0]<min(holdout_positions),'Adviser freeze order/coverage differs')
    frozen=read('adviser-freeze.json')
    require(frozen['cases_sha256']==digest_file(directory/'advice-cases.json') and
            frozen['environment_sha256']==digest_file(directory/'adviser-environment.json'),'Frozen adviser changed')
    advice_checks=[r for r in rows if r['event']=='advice_check']
    require(Counter(r['case_id'] for r in advice_checks)==Counter(c for c in cases if not c.startswith('pilot')),'Adviser case coverage differs')
    valid={split:sum(raw_advice(r['response'],cases[r['case_id']],env) for r in advice_checks if r['split']==split)
           for split in ('development','holdout')}
    require(valid==summary['valid_checks'],'Adviser validity totals differ')
    blocks=read('schedule.json');block_map={b['block_id']:b for b in blocks}
    conditions=[r for r in rows if r['event']=='condition']
    expected_keys={(b['block_id'],c) for b in blocks for c in b['conditions']}
    require(Counter((r['block_id'],r['condition']) for r in conditions)==Counter(expected_keys),'Condition coverage differs')
    ends=[r for r in rows if r['event']=='block_end']
    require([r['block_id'] for r in ends]==[b['block_id'] for b in blocks],'Block order differs')
    starts=[r['block_id'] for r in rows if r['event']=='block_start']
    require(starts==[b['block_id'] for b in blocks],'Block starts differ')
    ready={(r['block_id'],r['worker']):r for r in rows if r['event']=='worker_ready'}
    releases=[r for r in rows if r['event']=='worker_release']
    require(len(ready)==29 and len(releases)==29 and all(not r['alive'] and not r['forced'] for r in releases),'Workers missing or required forced cleanup')
    for b in blocks:
        selected=[r for r in conditions if r['block_id']==b['block_id']]
        require([r['condition'] for r in selected]==b['conditions'],'Condition ordering differs')
        a=ready[b['block_id'],'numeric']['owner'];h=ready[b['block_id'],'hat']['owner']
        require(a['tid']!=h['tid'],'Device workers share one thread')
        require((a['pid']==h['pid'])==(b['architecture']=='thread'),'Device worker architecture differs')
        for kind in ('numeric','hat'):
            original=ready[b['block_id'],kind]
            require(original['architecture']==b['architecture'],'Ready architecture differs')
    warmups=[r for r in rows if r['event']=='numeric_warmup']
    require(len(warmups)==14 and all(len(r['response']['result']['requests'])==6 for r in warmups),'Warmup coverage differs')
    for r in (r for r in rows if r['event']=='adviser_warmup'):
        raw_advice(r['response'],cases['pilot-00'],env)
    cached={}; numerical_count=0
    def check_output(v):
        key=(v['output_file'],v['fixture_id'])
        if key not in cached:
            path=(directory/v['output_file']).resolve()
            if not path.is_relative_to(directory): raise ValueError('Output path escapes run')
            out=np.load(path,allow_pickle=False);expected=expected_outputs[v['fixture_id']]
            delta=np.abs(out.astype(np.float64)-expected);bound=1e-5+1e-4*np.abs(expected)
            cached[key]=(hashlib.sha256(out.tobytes()).hexdigest(),float(delta.max()),float((delta/bound).max()))
            require(np.isfinite(out).all() and np.all(delta<=bound),'Saved numerical output is incorrect')
        checksum,absolute,fraction=cached[key]
        require(checksum==v['output_sha256'] and v['correct'] and not v['validation_errors'],'Numerical result/hash differs')
        require(absolute==v['max_absolute_error'] and fraction==v['max_tolerance_fraction'],'Numerical error metrics differ')
        require(v['started']<=v['computed']<=v['validated'],'Numerical timestamps out of order')
    for r in warmups:
        for v in r['response']['result']['requests']:check_output(v)
    for r in conditions:
        b=block_map[r['block_id']]
        require((r['stage'],r['repeat'],r['architecture'],r['case_id'])==
                (b['stage'],b['repeat'],b['architecture'],b['case_id']),'Condition metadata differs')
        require(math.isclose(r['total_ms'],1000*(r['ended']-r['started']),abs_tol=1e-6),'Full-job timer differs')
        for field,kind in (('numerical','numeric'),('adviser','hat')):
            v=r[field]
            require(v['owner']==ready[r['block_id'],kind]['owner'],'Context owner changed')
            require(r['started']<=v['submitted']<=v['received']<=v['ended']<=v['reply_sent']<=v['delivered']<=r['ended'],'Worker timing order differs')
        if r['condition'].endswith('serial'):
            require(r['numerical']['delivered']<=r['adviser']['submitted'],'Serial work overlaps')
        values=r['numerical']['result']['requests']
        require(len(values)==24,'Numerical request count differs')
        require([v['fixture_id'] for v in values]==b['fixture_ids']*8,'Numeric input order differs')
        require(all(v['backend']==('native4' if r['condition'].startswith('cpu') else 'C6') for v in values),'Backend differs')
        for v in values:check_output(v);numerical_count+=1
    main=[r for r in conditions if r['stage']=='main']
    main_valid=sum(raw_advice(r['adviser'],cases[r['case_id']],env) for r in main)
    require(main_valid==summary['valid_main_advice'],'Main advice validity differs')
    lookup={(r['repeat'],r['architecture']+'_'+r['condition']):r['total_ms'] for r in main}
    intervals=0
    for name,c in summary['comparisons'].items():
        ratio,ci=interval([(lookup[i,c['left']],lookup[i,c['right']]) for i in range(6)],protocol['seed'])
        require(math.isclose(ratio,c['ratio'],rel_tol=1e-12) and np.allclose(ci,c['ci95'],atol=1e-12,rtol=1e-12),f'Interval differs: {name}')
        timing=ratio>=1.10 and ci[0]>1
        require(c['timing_gate']==timing and c['application_gate']==(timing and valid['holdout']==20 and main_valid==48 and len(main)==48),f'Acceptance differs: {name}')
        intervals+=1
    require(intervals==10,'Missing comparisons')
    pilot=[b for b in ends if b['stage']=='pilot'];gate=read('pilot-gate.json')
    require(gate['fits'] and math.isclose(gate['required_seconds'],sum(b['elapsed_seconds'] for b in pilot)*6*1.2+45),'Pilot gate differs')
    require(summary['recommend_process_overlap']==summary['comparisons']['primary_process_cpu_overlap']['application_gate'],'Recommendation differs')
    return dict(passed=not issues,issues=issues,verified_artifacts=len(checks),
        fixtures=len(fixtures),stored_output_arrays=len(cached),numerical_requests_including_pilot=numerical_count,
        main_conditions=len(main),valid_holdout=valid['holdout'],valid_main=main_valid,
        recomputed_intervals=intervals,worker_releases=len(releases))


if __name__=='__main__':
    try:
        result=audit(sys.argv[1])
    except Exception as exc:
        result=dict(passed=False,issues=[f'{type(exc).__name__}: {exc}'])
    print(json.dumps(result,indent=2))
    if '--write' in sys.argv:
        atomic_json(Path(sys.argv[1])/'validation.json',result)
        from .coordination_report import seal
        seal(Path(sys.argv[1]))
    raise SystemExit(0 if result['passed'] else 1)
