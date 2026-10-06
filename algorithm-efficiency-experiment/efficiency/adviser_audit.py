"""Independent offline verification of adviser decisions, controls and holdout isolation."""
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

from .common import atomic_json, digest_file, digest_value, read_jsonl

IDS=[f'C{i}' for i in range(8)]


def reference_cpu(payload):
    remaining=set(payload['eligible'])-set(payload['development_ms'])
    if not remaining:return None
    history=payload['development_ms']
    if not history:return min(remaining)
    center=int(min(history,key=lambda c:(history[c],c))[1])
    for bit in (4,2,1):
        candidate=f'C{center^bit}'
        if candidate in remaining:return candidate
    return min(remaining)


def classify(case,row,stops):
    p=case['payload'];response=row['response']
    if not p['eligible']:
        if response is not None:raise ValueError('Empty-set request invoked model')
        return None,'abstain',None,None,None
    if row['arm']=='cpu':
        if response is not None:raise ValueError('CPU control invoked model')
        return reference_cpu(p),'cpu',None,None,None
    if response is None:raise ValueError('Nonempty model arm omitted generation')
    result=response['result'];raw=result['raw'];choice=None;normalization=None;boundary=None
    if raw.get('error'):reason='generation_error'
    elif not (raw['completion_status']=='LOGICAL_END_OF_GENERATION' and raw['token_ledger_valid'] and
              raw['context_before']==0 and raw['context_after']==raw['expected_context_after']):
        reason='incomplete_or_invalid_context'
    else:
        output=raw['output'];ending=next((s for s in sorted(stops,key=len,reverse=True) if output.endswith(s)),None)
        body=output[:-len(ending)].strip() if ending else output.strip()
        boundary='model_terminal' if ending else 'candidate_stop' if row['arm']=='bounded' and re.fullmatch(r'C[0-7]',body) else None
        if boundary is None:reason='missing_or_suppressed_terminal'
        elif re.fullmatch(r'C[0-7]',body):choice=body;reason=None
        elif ending and re.fullmatch(r'C[0-7]\.',body):choice=body[:-1];normalization='remove_one_trailing_period';reason=None
        else:reason='malformed_proposal'
        if choice is not None and choice not in p['eligible']:choice=None;reason='ineligible_or_repeated_proposal'
    if not row['state_valid']:
        return reference_cpu(p),'cpu_fallback',None,boundary,'unverified_restoration_or_reset'
    if choice is None:return reference_cpu(p),'cpu_fallback',None,boundary,reason
    return choice,'hat_canonicalized' if normalization else 'hat_exact',normalization,boundary,None


def audit(directory):
    directory=Path(directory).resolve();read=lambda n:json.loads((directory/n).read_text())
    issues=[]
    def require(value,reason):
        if not value:issues.append(reason)
    checks=read('checksums.json')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p!=directory/'checksums.json'}
    require(set(checks)==actual,'Artifact inventory differs')
    for name,checksum in checks.items():
        path=(directory/name).resolve()
        require(path.is_relative_to(directory) and path.is_file() and digest_file(path)==checksum,f'Checksum differs: {name}')
    rows,damaged=read_jsonl(directory/'adviser.jsonl');require(not damaged,'Damaged event records')
    measures=[r for r in rows if r['event']=='measurement']
    summary,selection,env=read('summary.json'),read('selection.json'),read('adviser-environment.json')
    outcome,protocol=read('outcome.json'),read('protocol.json')
    require(outcome['status']=='complete' and summary['complete'],'Run incomplete')
    require(outcome['elapsed_seconds']<=read('config.json')['max_seconds'],'Runtime cap exceeded')
    require(not any(r['event']=='incomplete' for r in rows),'Incomplete event present')
    manifest=read('manifest.json')
    for name,checksum in manifest['source_sha256'].items():
        require(digest_file(directory/'source'/name)==checksum,f'Measurement source differs: {name}')
    for name,checksum in read('source-provenance.json')['frozen_sha256'].items():
        require(digest_file(directory/'source-run'/name)==checksum,f'Frozen provenance differs: {name}')
    dev=read('development-cases.json');holdout=read('holdout-cases.json')
    previous={c['case_id']:c for c in read('source-run/advice-cases.json') if c['split'] in ('development','holdout')}
    require(len(dev)==40 and len(holdout)==48,'Case counts differ')
    for c in dev:
        old=previous[c['source_case_id']]
        require(c['split']=='development' and c['baseline_messages']==old['messages'] and
                c['payload']==json.loads(old['messages'][1]['content']),'Old case not preserved as development')
    dev_hashes={digest_value(c['payload']) for c in dev};holdout_hashes={digest_value(c['payload']) for c in holdout}
    require(len(holdout_hashes)==48 and not dev_hashes&holdout_hashes,'Holdout duplicated known data')
    require(Counter(str(len(c['payload']['eligible'])) for c in holdout)==Counter(protocol['eligibility_size_counts']),'Eligibility size coverage differs')
    for c in dev+holdout:
        p=c['payload'];require(set(p['eligible']).isdisjoint(p['tested']) and set(p['eligible'])|set(p['tested'])==set(IDS),'Eligible/tested partition differs')
        require(set(p['development_ms'])==set(p['tested']),'History identifiers differ')
    freeze=[i for i,r in enumerate(rows) if r['event']=='selection_frozen']
    created=[i for i,r in enumerate(rows) if r['event']=='holdout_created']
    ds=[i for i,r in enumerate(rows) if r['event']=='measurement' and r['stage']=='development']
    hs=[i for i,r in enumerate(rows) if r['event']=='measurement' and r['stage']=='holdout']
    require(len(freeze)==len(created)==1 and max(ds)<freeze[0]<created[0]<min(hs),'Holdout isolation/freeze order differs')
    require(selection['development_sha256']==digest_file(directory/'development-cases.json') and
            selection['environment_sha256']==digest_file(directory/'adviser-environment.json'),'Frozen selection inputs differ')
    for name,checksum in selection['implementation_sha256'].items():
        require(digest_file(directory/'source/efficiency'/name)==checksum,'Parser/prompt changed during measurements')
    case_map={c['case_id']:c for c in dev+holdout};templates=read('prompt-variants.json')
    ready={r['instance']:r for r in rows if r['event']=='worker_ready'}
    releases=[r for r in rows if r['event']=='worker_release']
    require(len(ready)==len(releases) and all(not r['alive'] and not r['forced'] for r in releases),'Workers not cleanly released')
    for row in ready.values():
        for key in ('hailort_version','prompt_template','stop_tokens','capacity_tokens','parameters','model_defaults','experiment_limit_tokens'):
            require(row['environment'][key]==env[key],f'Worker model setting differs: {key}')
    decisions={};origin_counts=Counter();restorations=0
    for r in measures:
        if r['stage']=='preflight':
            candidate=r['case_id'].removeprefix('control-')
            case=dict(payload=dict(eligible=[candidate],tested=sorted(set(IDS)-{candidate}),development_ms={c:100 for c in IDS if c!=candidate}))
            expected_messages=[dict(role='system',content='Copy the requested ID exactly. Do not add punctuation or explanation.'),dict(role='user',content=f'Reply exactly {candidate}')]
        else:
            case=case_map[r['case_id']];p=case['payload']
            expected_messages=case['baseline_messages'] if r['arm']=='baseline' else templates['examples']+[
                dict(role='user',content=json.dumps(dict(eligible_options=p['configurations'],tested_history_ms=p['development_ms']),separators=(',',':')))]
        expected=classify(case,r,env['stop_tokens'])
        actual_decision=tuple(r['decision'][k] for k in ('choice','origin','normalization','completion_boundary','fallback_reason'))
        require(expected==actual_decision,f'Decision differs: {r["stage"]}/{r["case_id"]}/{r["arm"]}')
        require(r['decision']['eligible_verified'] is True,'Eligibility not verified')
        require((expected[0] in case['payload']['eligible']) if case['payload']['eligible'] else expected[0] is None,'Unsafe selection escaped')
        require(math.isclose(r['total_ms'],1000*(r['ended']-r['started']),abs_tol=1e-6),'Request timer differs')
        if r['response']:
            response=r['response'];result=response['result'];raw=result['raw']
            require(response['owner']==ready[r['instance']]['owner'],'Worker identity changed')
            require(r['started']<=response['submitted']<=response['received']<=result['started']<=result['ended']<=response['ended']<=response['reply_sent']<=response['delivered']<=r['ended'],'Timing order differs')
            require(raw['messages']==expected_messages and raw['effective_parameters']==env['parameters'],'Prompt or generation settings differ')
            requested=list(dict.fromkeys(env['stop_tokens']+IDS)) if r['arm']=='bounded' else env['stop_tokens']
            require(result['requested_stops']==requested,'Requested stop settings differ')
            if not raw.get('error'):require(result['active_stops']==requested,'Stop readback differs')
            require(r['state_valid']==bool(result['restored'] and result['reset_verified']),'Restoration classification differs')
            if r['stage']!='preflight':require(r['state_valid'],'Unverified post-preflight restoration')
            restorations+=int(r['state_valid'])
        decisions[r['stage'],r['case_id'],r['arm']]=expected
        if r['stage']=='holdout':origin_counts[r['arm'],expected[1]]+=1
    preflight=read('native-stop-preflight.json');control_results=[]
    pf=[r for r in measures if r['stage']=='preflight']
    for c in preflight['controls']:
        triple=[r for r in pf if r['case_id']=='control-'+c['candidate']]
        require([r['control'] for r in triple]==['before','custom','after'][:len(triple)],'Preflight order differs')
        supported=False
        if len(triple)==3:
            a,b,d=[r['response']['result'] for r in triple]
            keys=('output','completion_status','token_ledger_valid','context_before','context_after','expected_context_after')
            equal=all(a['raw'].get(k)==d['raw'].get(k) for k in keys)
            ledgers=all(v['raw'].get('token_ledger_valid') is True and v['raw'].get('context_before')==0 and
                        v['raw'].get('context_after')==v['raw'].get('expected_context_after') for v in (a,b,d))
            supported=equal and ledgers and all(v['restored'] and v['reset_verified'] for v in (a,b,d)) and b['active_stops']==list(dict.fromkeys(env['stop_tokens']+IDS)) and triple[1]['decision']['origin']=='hat_exact' and triple[1]['decision']['choice']==c['candidate']
            require(c['controls_equal']==equal and c['valid_ledgers']==ledgers,'Preflight diagnostics differ')
        require(c['supported']==bool(supported),'Native stop support differs');control_results.append(bool(supported))
    bounded=len(control_results)==8 and all(control_results)
    require(bounded==preflight['bounded_supported'],'Native stop gate differs')
    available=['baseline','examples']+(['bounded'] if bounded else [])
    require(selection['available_arms']==available,'Available arms differ')
    for stage,name,arms in (('development','development-schedule.json',available+['cpu']),
                            ('holdout','holdout-schedule.json',['baseline',selection['challenger'],'cpu'])):
        schedule=read(name)
        require(all(set(j['arms'])==set(arms) and len(j['arms'])==len(arms) for j in schedule),'Scheduled arm set differs')
        expected_order=[(j['case_id'],a) for j in schedule for a in j['arms']]
        actual_order=[(r['case_id'],r['arm']) for r in measures if r['stage']==stage]
        require(expected_order==actual_order,f'{stage} schedule/order differs')
        require(len(set(expected_order))==len(expected_order),'Duplicate scheduled observations')
    scores=[]
    for arm in available[1:]:
        selected=[r for r in measures if r['stage']=='development' and r['arm']==arm]
        scores.append(dict(arm=arm,accepted=sum(r['decision']['origin'].startswith('hat_') for r in selected),
            exact=sum(r['decision']['origin']=='hat_exact' for r in selected),mean_request_ms=sum(r['total_ms'] for r in selected)/40))
    best=min(scores,key=lambda s:(-s['accepted'],-s['exact'],s['mean_request_ms'],['examples','bounded'].index(s['arm'])))
    require(scores==selection['scores'] and best['arm']==selection['challenger'],'Development-only selection differs')
    pilot=[r for r in measures if r['stage']=='pilot'];gate=read('pilot-gate.json')
    require(Counter((r['case_id'],r['arm']) for r in pilot)==Counter((c,a) for c in protocol['pilot_cases'] for a in available),'Pilot coverage differs')
    required=max(r['total_ms'] for r in pilot)/1000*(40*len(available)+80)*1.2+45
    require(gate['fits'] and math.isclose(required,gate['required_seconds']) and required<=gate['remaining_seconds'],'Pilot gate differs')
    for g in summary['groups']:
        arm=g['arm'];selected=[r for r in measures if r['stage']=='holdout' and r['arm']==arm and case_map[r['case_id']]['payload']['eligible']]
        require(len(selected)==40 and g['nonempty_cases']==40 and g['empty_cases']==8,'Holdout group coverage differs')
        for key,origin in (('exact','hat_exact'),('canonicalized','hat_canonicalized'),('cpu_fallback','cpu_fallback'),('cpu_direct','cpu')):
            require(g[key]==origin_counts[arm,origin],f'Summary origin count differs: {arm}/{key}')
        require(g['model_accepted']==g['exact']+g['canonicalized'],'Accepted count differs')
        require(math.isclose(g['mean_request_ms'],sum(r['total_ms'] for r in selected)/40,rel_tol=1e-12),'Latency summary differs')
    accepted=origin_counts[selection['challenger'],'hat_exact']+origin_counts[selection['challenger'],'hat_canonicalized']
    require(summary['interface_passed'] and summary['reliability_passed']==(accepted==40),'Final reliability gate differs')
    require(summary['holdout_requests']==144 and summary['holdout_model_requests']==80,'Held-out request totals differ')
    return dict(passed=not issues,issues=issues,verified_artifacts=len(checks),development_cases=len(dev),
        holdout_cases=len(holdout),verified_decisions=len(measures),native_stop_supported=bounded,
        challenger=selection['challenger'],holdout_model_accepted=accepted,verified_restorations=restorations,
        worker_releases=len(releases),reliability_passed=accepted==40)


if __name__=='__main__':
    try:result=audit(sys.argv[1])
    except Exception as exc:result=dict(passed=False,issues=[f'{type(exc).__name__}: {exc}'])
    print(json.dumps(result,indent=2))
    if '--write' in sys.argv:
        atomic_json(Path(sys.argv[1])/'validation.json',result)
        from .coordination_report import seal
        seal(Path(sys.argv[1]))
    raise SystemExit(0 if result['passed'] else 1)
