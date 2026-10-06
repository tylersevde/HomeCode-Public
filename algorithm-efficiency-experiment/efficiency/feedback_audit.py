"""Independent offline artifact audit; never initializes numerical devices."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

from .common import atomic_json, digest_file


def audit(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory/name).read_text())
    checks = read('checksums.json')
    failures = []
    for name, expected in checks.items():
        if not (directory/name).is_file() or digest_file(directory/name)!=expected:
            failures.append('Checksum mismatch: '+name)
    manifest = read('manifest.json')
    for name, expected in manifest['source_sha256'].items():
        if digest_file(directory/'source'/name)!=expected: failures.append('Source snapshot mismatch: '+name)
    for name, expected in read('native-build/build.json')['artifacts'].items():
        if digest_file(directory/'native-build'/name)!=expected: failures.append('Binary mismatch: '+name)
    rows = [json.loads(line) for line in (directory/'feedback.jsonl').read_text().splitlines() if line.strip()]
    measured = [r for r in rows if r['event']=='measurement']
    specification = read('protocol.json'); fixtures = {f['fixture_id']: f for f in read('fixtures.json')}
    if len(fixtures)!=240 or len({f['seed'] for f in fixtures.values()})!=240: failures.append('Fixture overlap or coverage')
    import numpy as np
    for fid, fixture in fixtures.items():
        path = directory/'fixtures'/(fid+'.npz')
        if digest_file(path)!=fixture['file_sha256']: failures.append('Fixture checksum: '+fid)
        with np.load(path, allow_pickle=False) as data:
            for key, sha in (('x','input_sha256'), ('w','weights_sha256'), ('expected','oracle_sha256')):
                if hashlib.sha256(data[key].tobytes()).hexdigest()!=fixture[sha]: failures.append('Fixture array hash: '+fid+'/'+key)
    # Independently construct planned observation identities from the frozen protocol.
    expected = set()
    for cell in specification['cells']:
        cid = cell['cell_id']
        for i in range(3):
            for repeat in range(3):
                for arm in specification['cpu_backends']: expected.add(('baseline',None,None,cid,i,repeat,arm))
        for strategy in specification['strategies']:
            for round_number in range(1,4):
                for stage, repeats in (('development',3),('validation',5)):
                    for i in range(3):
                        for repeat in range(repeats):
                            for arm in ('incumbent','candidate'): expected.add((stage,strategy,round_number,cid,i,repeat,arm))
        for i in range(8):
            for repeat in range(5):
                for arm in ('baseline','coordinate','hat'): expected.add(('final',None,None,cid,i,repeat,arm))
    actual = [tuple(r.get(k) for k in ('stage','strategy','round','cell_id','seed_index','repeat','arm')) for r in measured]
    summary = read('summary.json')
    exact = len(actual)==len(set(actual)) and set(actual)==expected
    if exact!=summary['coverage_exact']: failures.append('Reported coverage differs')
    for r in measured:
        f = fixtures[r['fixture_id']]
        if r['input_sha256']!=f['input_sha256'] or r['split']!=f['split'] or r['cell_id']!=f['cell_id']:
            failures.append('Measurement input/split mismatch')
        if not r['correct'] or r['max_tolerance_fraction'] is None or r['max_tolerance_fraction']>1:
            failures.append('Numerical error')
        if r['request_ms']<=0 or r['ended']<r['started'] or r.get('validation_errors',0): failures.append('Invalid timing/validation')
    freeze = [r for r in rows if r['event']=='policies_frozen']
    finals = [r for r in measured if r['stage']=='final']
    if finals:
        if len(freeze)!=1 or freeze[0]['monotonic']>=min(r['started'] for r in finals): failures.append('Policy freeze ordering')
        if freeze[0]['sha256']!=digest_file(directory/'policies.json'): failures.append('Frozen policy changed')
        policies = read('policies.json')
        baseline = read('baseline-policy.json')['routes']
        for r in finals:
            routes = baseline if r['arm']=='baseline' else policies[r['arm']]['routes']
            if r['backend']!=routes[r['cell_id']]: failures.append('Final route differs from frozen policy')
    def bootstrap(pairs, seed):
        count=len(pairs)
        ratio = sum(p[0] for p in pairs)/sum(p[1] for p in pairs)
        rng=random.Random(seed); values=[]
        for _ in range(10000):
            sample=rng.choices(pairs,k=count)
            values.append(sum(p[0] for p in sample)/sum(p[1] for p in sample))
        values.sort()
        return ratio,[values[250],values[9750]]
    def interval(selected, left, right, count, seed):
        pairs=[]
        for index in range(count):
            pairs.append([statistics.median(r['request_ms'] for r in selected if r['seed_index']==index and r['arm']==arm)
                          for arm in (left,right)])
        return bootstrap(pairs,seed)
    verified_intervals=0
    for result in summary['cells']:
        selected=[r for r in finals if r['cell_id']==result['cell_id']]
        ratio, ci=interval(selected,'baseline',result['strategy'],8,specification['seed'])
        if ratio!=result['ratio'] or ci!=result['ci95']: failures.append('Final interval differs')
        verified_intervals+=1
    if len(finals)==1440:
        aggregate_cases=[]
        for strategy in specification['strategies']:
            for mode in ('stream','prefill','mixture'):
                cells=[c['cell_id'] for c in specification['cells'] if mode=='mixture' or c['mode']==mode]
                aggregate_cases.append(('baseline',strategy,cells,summary['comparisons'][strategy][mode]))
        aggregate_cases.append(('coordinate','hat',[c['cell_id'] for c in specification['cells']],summary['comparisons']['hat_vs_coordinate']))
        for left,right,cells,result in aggregate_cases:
            pairs=[]
            for index in range(8):
                pairs.append([sum(statistics.median(r['request_ms'] for r in finals if r['cell_id']==cid
                    and r['seed_index']==index and r['arm']==arm) for cid in cells) for arm in (left,right)])
            ratio,ci=bootstrap(pairs,specification['seed'])
            if ratio!=result['ratio'] or ci!=result['ci95']: failures.append('Aggregate interval differs')
            verified_intervals+=1
    decisions=[r for r in rows if r['event']=='decision']
    for decision in decisions:
        selected=[r for r in measured if r['stage']=='validation' and r['strategy']==decision['strategy']
                  and r['round']==decision['round'] and r['cell_id']==decision['cell_id']]
        ratio, ci=interval(selected,'incumbent','candidate',3,specification['seed']+decision['round'])
        accepted=all(r['correct'] for r in selected) and ratio>=1.10 and ci[0]>1
        if accepted!=decision['promote'] or ratio!=decision['inference']['ratio'] or ci!=decision['inference']['ci95']:
            failures.append('Promotion decision differs')
        if decision['applied'] and decision['after']!=(decision['candidate'] if accepted else decision['before']):
            failures.append('Applied route differs')
        verified_intervals+=1
    proposals=[r for r in rows if r['event']=='proposal']
    for strategy in specification['strategies']:
        selected=[r for r in proposals if r['strategy']==strategy]
        if len({r['candidate'] for r in selected})!=len(selected): failures.append('Repeated candidate')
    for proposal in proposals:
        if 'adviser' not in proposal: continue
        result=proposal['adviser']; payload=json.loads(result['messages'][1]['content'])
        allowed={'catalog','eligible_untested','current_routes','measured_development_history'}
        if set(payload)!=allowed: failures.append('Unexpected model feedback field')
        prior=[r for r in proposals if r['strategy']=='hat' and r['round']<proposal['round']]
        if {h['candidate'] for h in payload['measured_development_history']}!={r['candidate'] for r in prior}:
            failures.append('HAT history isolation differs')
        if result['context_before']!=0: failures.append('HAT context not fresh')
        if proposal['credited_to_hat'] and (not result['token_ledger_valid'] or result['choice']!=proposal['candidate']):
            failures.append('HAT credit invalid')
    concurrency=[r for r in rows if r['event']=='concurrency']
    for r in concurrency:
        a,b=r['numerical'],r['adviser']
        overlap=max(0,min(a['ended'],b['ended'])-max(a['started'],b['started']))*1000
        if abs(overlap-r['overlap_ms'])>1e-9: failures.append('Overlap arithmetic differs')
        if len(a['requests'])!=24 or not all(v['correct'] for v in a['requests']): failures.append('Concurrency numeric coverage/correctness')
    if len(concurrency)==24:
        by_condition={}
        for r in concurrency: by_condition.setdefault(r['condition'],{})[r['repeat']]=r
        comparisons=[(backend+'_serial',backend+'_overlap',summary['concurrency'][backend]) for backend in ('cpu','gpu')]
        comparisons += [('cpu_'+mode,'gpu_'+mode,summary['concurrency_cpu_over_gpu'][mode]) for mode in ('serial','overlap')]
        for left,right,result in comparisons:
            ratio,ci=bootstrap([(by_condition[left][i]['total_ms'],by_condition[right][i]['total_ms']) for i in range(6)],specification['seed'])
            if ratio!=result['ratio'] or ci!=result['ci95']: failures.append('Concurrency interval differs')
            verified_intervals+=1
        spec=read('concurrency-spec.json')
        if any(r['adviser']['messages']!=spec['messages'] for r in concurrency): failures.append('Concurrency prompt changed')
    correctness=read('correctness.json')
    if len(correctness['cases'])!=132 or not all(c['correct'] for c in correctness['cases']): failures.append('Initial correctness coverage')
    expected_complete=exact and len(concurrency)==24 and read('outcome.json')['status']=='complete' and any(r['event']=='complete' for r in rows)
    if expected_complete!=summary['complete']: failures.append('Reported completion differs')
    config=read('config.json'); cap=config.get('resume_budget',{}).get('total_cap_seconds',config['max_seconds'])
    used=config.get('resume_budget',{}).get('prior_elapsed_seconds',0)+read('outcome.json')['elapsed_seconds']
    external=json.loads((directory/'execution-budget.json').read_text()).get('earlier_stopped_attempt_seconds',0) if (directory/'execution-budget.json').exists() else 0
    if used>cap or used+external>3600: failures.append('Total supervised budget exceeded')
    if abs(summary['total_supervised_seconds']-used-external)>1e-6: failures.append('Reported cumulative time differs')
    for strategy in specification['strategies']:
        result=summary['comparisons'][strategy]['mixture']
        cells=[c for c in summary['cells'] if c['strategy']==strategy]
        gpu_cells=sum(b in specification['catalog'] for b in read('policies.json')[strategy]['routes'].values()) if (directory/'policies.json').exists() else 0
        accepted=bool(expected_complete and not summary['replay'] and gpu_cells and result and result['ratio']>=1.10
            and result['ci95'][0]>1 and len(cells)==12 and all(c['ratio']>=1/1.05 for c in cells)
            and all(r['correct'] for r in measured))
        if accepted!=summary['acceptance'][strategy]['gpu_benefit']: failures.append('GPU acceptance differs')
    result=dict(passed=not failures, failures=failures, verified_artifacts=len(checks),
        source_files=len(manifest['source_sha256']), verified_fixtures=len(fixtures),
        measured_requests=len(measured), complete_measurements=exact, recomputed_intervals=verified_intervals,
        audited_decisions=len(decisions), audited_proposals=len(proposals), audited_concurrency=len(concurrency),
        total_supervised_seconds=used+external,
        scope='Offline integrity, independent schedule, split/route checks, paired bootstrap and overlap arithmetic. Does not rerun hardware outputs.')
    atomic_json(directory/'validation.json',result)
    return result


if __name__=='__main__':
    result=audit(sys.argv[1]); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['passed'] else 1)
