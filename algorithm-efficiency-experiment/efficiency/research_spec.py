"""Frozen campaign design and pure decision rules; no device access."""
import hashlib
import math
import random
import statistics

from .feedback_spec import CELLS

VERSION='attention-research-v1'
STAGES=('profile','gpu','context','combined','confirm')
EVENTS='research.jsonl'
GPU_CONTROLS=('C0','C4','C6','C7')
CACHE_ARMS=('rebuild','retain','checkpoint')
CONDITIONS=('serial-0','overlap-0','overlap-25','overlap-50','overlap-100')


def specification(stage):
    if stage not in STAGES:raise ValueError('Unknown research stage')
    return dict(version=VERSION,stage=stage,session_limit_seconds=7200,campaign_limit_seconds=36000,
        cleanup_reserve_seconds=120,pilot_margin=1.2,cells=CELLS,fixture_blocks=8 if stage=='confirm' else 6,
        numeric_repeats=3,context_sizes=[128,512,1024],context_fixtures=8,context_turns=4,
        context_repeats=3 if stage=='confirm' else 2,gpu_controls=list(GPU_CONTROLS),
        variants={"0":"original","1":"causal_scores","2":"causal_all"},
        cache_arms=list(CACHE_ARMS),combined_conditions=list(CONDITIONS),combined_requests=24,
        atol=1e-5,rtol=1e-4,promotion_gain=1.10,shape_slowdown_limit=1.05,
        confidence=1-.05/3 if stage=='confirm' else .95,bootstrap_samples=10000,
        sampling_unit='Paired fresh fixture block; repeated requests are summarized within blocks.',
        timing='Full request includes validation and IPC. GPU intervals overlap host waits and are not additive to wall time.',
        correctness_domain='Original Gaussian FP32 inputs and scaled-normal weights; sign perturbations retain magnitude.',
        seed_derivation='sha256(version|fixture_namespace|stage|fixture_id), first 8 bytes little endian; first attempt namespace is campaign_id',
        selection='Correct eligible minimum development mean; ties by arm name. Fresh confirmation required.',
        future='No kernel fusion, precision reduction, model advice, or driver changes in this campaign.')


def seed_for(campaign,stage,fid):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{campaign}|{stage}|{fid}'.encode()).digest()[:8],'little')


def numeric_arms(stage,cell,parents):
    if stage=='profile':
        return [dict(arm=c,backend=c,variant=0,profiling=False) for c in ('native1','native4')]+[
            dict(arm=f'{c}-p{p}',backend=c,variant=0,profiling=bool(p)) for c in GPU_CONTROLS for p in (0,1)]
    base=parents['profile']['selection']['gpu'][cell]
    cpu=parents['profile']['selection']['cpu'][cell]
    if stage=='gpu':
        return [dict(arm=cpu,backend=cpu,variant=0,profiling=False)]+[
            dict(arm=f'{base}-v{v}',backend=base,variant=v,profiling=False) for v in (0,1,2)]
    selected=parents['gpu']['selection']['routes'][cell]
    return [dict(arm='incumbent',backend=cpu,variant=0,profiling=False),dict(arm='candidate',**selected,profiling=False)]


def balanced_order(items,index):
    items=list(items);offset=index%len(items)
    result=items[offset:]+items[:offset]
    return list(reversed(result)) if (index//len(items))%2 else result


def interval(pairs,confidence=.95,seed=20261009,samples=10000):
    if len(pairs)<2 or any(a<=0 or b<=0 or not math.isfinite(a+b) for a,b in pairs):return None
    rng=random.Random(seed)
    values=[]
    for _ in range(samples):
        drawn=rng.choices(pairs,k=len(pairs));values.append(sum(a for a,b in drawn)/sum(b for a,b in drawn))
    values.sort();tail=(1-confidence)/2
    return dict(ratio=sum(a for a,b in pairs)/sum(b for a,b in pairs),
        interval=[values[int(tail*samples)],values[min(samples-1,int((1-tail)*samples))]],
        confidence=confidence,blocks=len(pairs),samples=samples,seed=seed)


def accepted(result,correct,shape_ratios=()):
    return bool(correct and result and result['ratio']>=1.10 and result['interval'][0]>1 and
                all(r>=1/1.05 for r in shape_ratios))


def validate_snapshot(metadata,blob,model_sha,context_tokens):
    if metadata['model_sha256']!=model_sha:raise ValueError('Checkpoint model differs')
    if metadata['sha256']!=hashlib.sha256(blob).hexdigest():raise ValueError('Checkpoint checksum differs')
    if metadata['context_tokens']!=context_tokens:raise ValueError('Checkpoint token count differs')
    return True


def checkpoint_equivalent(rows):
    """A complete four-turn paired dialogue; raw accuracy is a separate property."""
    if not rows or any(len(v)!=4 for v in rows.values()):return False
    for turn in range(4):
        group=[v[turn] for v in rows.values()]
        if not all(r.get('valid') for r in group):return False
        if len({(r['prompt_sha256'],r['output'],r['status'],r['context_after']) for r in group})!=1:return False
    return True
