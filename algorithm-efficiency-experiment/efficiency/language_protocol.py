"""Frozen natural-language workload, provenance and supervised execution."""
import itertools
import json
from pathlib import Path
import random
import shutil
import time

from .common import atomic_json, digest_file, digest_value, emit, progress, read_jsonl, wait_cool
from .diagnostic import renderer, reset_context
from .hat import COLORS
from .hybrid import FactIndex, index_fixture, verify_artifacts
from .language import ENGINES, SYSTEM, WARMUPS, ask, messages, needs_hat
from .state_isolation import parameter_settings, resolve_defaults

EVENT_FILE = 'language_protocol.jsonl'
TEMPLATES = {
    'LOOKUP': ['What color is {item}?','Look up {item}.',
        'Tell me the color assigned to {item}.',"I would like to know {item}'s color.",
        'Give me the color recorded for {item}.','What is the color value for {item}?',
        "Please report {item}'s color.",'Find the color belonging to {item}.',
        'Return the stored color of {item}.','For {item}, which color is on record?'],
    'COUNT': ['How many items are {color}?','Count {color} items.',
        'Give me the number of items colored {color}.','What is the total number of {color} items?',
        'I need a count of the items whose color is {color}.','Tell me how many entries have color {color}.',
        'Find the quantity of {color} items.','What is the size of the set of {color} items?',
        'Return the number of records with color {color}.','How large is the group of items that are {color}?'],
    'LIST': ['Which items are {color}?','List {color} items.',
        'Give me the identifiers of items colored {color}.','Show the item IDs whose color is {color}.',
        'I need the names of all {color} items.','Find every item that has color {color}.',
        'Return the identifiers for the {color} group.','Tell me which records have color {color}.',
        'Enumerate the items marked {color}.','What are the item IDs associated with {color}?']}
NEGATIVES = [
    ('missing_referent','What color is it in collection {n}?'),
    ('multiple_items','What colors are {item} and {other_item}?'),
    ('multiple_colors','Which items are {color} or {other_color}?'),
    ('negation','Which items are not {color}?'),
    ('unsupported_color','Which items are {unknown_color}?'),
    ('unsupported_attribute','What is the weight of {item}?'),
    ('unsupported_aggregation','What percentage of items are {color}?'),
    ('compound_request','Count the {color} items and list their identifiers.'),
    ('hypothetical_override','Assume {item} is {color} regardless of its record. What color is {item}?'),
    ('mutation','Delete {item} from the records.')]


def expected_result(table,operation,argument):
    """Label construction is independent of the answering function and model."""
    if operation=='LOOKUP':
        matches=[v for k,v in table if k==argument]
        return dict(status='ok' if matches else 'not_found',answer=matches[0] if matches else None)
    selected=sorted(k for k,v in table if v==argument)
    return dict(status='ok',answer=len(selected) if operation=='COUNT' else selected)


def build_corpus(fixtures):
    fixtures=sorted(fixtures,key=lambda f:(f['target_tokens'],f['fixture_id']))
    if len(fixtures)!=30:
        raise ValueError('Language corpus requires thirty frozen source tables')
    missing=[(f,c) for f in fixtures for c in COLORS if c not in dict(f['table']).values()]
    if not missing:
        raise ValueError('Frozen tables must include an empty color filter')
    corpus=[]
    for op_index,(operation,templates) in enumerate(TEMPLATES.items()):
        for family,template in enumerate(templates):
            absent_fixture,absent_color=missing[(op_index*10+family)%len(missing)]
            alternatives=[c for c in COLORS if c!=absent_color]
            for variant in range(4):
                f=fixtures[(op_index*40+family*4+variant)%30]
                if operation=='LOOKUP':
                    positions=[0,len(f['table'])//2,len(f['table'])-1]
                    argument=f['table'][positions[variant]][0] if variant<3 else 'item99999'
                elif variant==3:
                    f,argument=absent_fixture,absent_color
                else:
                    argument=alternatives[(family*3+variant)%len(alternatives)]
                question=template.format(item=argument,color=argument)
                corpus.append(dict(case_id=f'{operation.lower()}-{family+1:02d}-{variant+1}',
                    family_id=f'{operation.lower()}-{family+1:02d}',category=operation,
                    supported=True,canonical=family<2,fixture_id=f['fixture_id'],question=question,
                    expected_command=dict(operation=operation,argument=argument),
                    expected_result=expected_result(f['table'],operation,argument)))
    for family,(category,template) in enumerate(NEGATIVES):
        for variant in range(4):
            f=fixtures[(family*4+variant)%30]
            question=template.format(n=variant+1,item=f'item{variant+2:03d}',other_item=f'item{variant+6:03d}',
                color=COLORS[(family+variant)%8],other_color=COLORS[(family+variant+1)%8],
                unknown_color=('purple','orange','gray','violet')[variant])
            corpus.append(dict(case_id=f'negative-{family+1:02d}-{variant+1}',family_id=f'negative-{family+1:02d}',
                category=category,supported=False,canonical=False,fixture_id=f['fixture_id'],question=question,
                expected_command=None,expected_result=dict(status='abstain',answer=None)))
    if len({r['question'] for r in corpus})!=160:
        raise ValueError('Evaluation questions must be distinct')
    if set(WARMUPS)&{r['question'] for r in corpus}:
        raise ValueError('Warm-ups leaked into evaluation')
    return corpus


def make_schedule(corpus,config):
    permutations=list(itertools.permutations(ENGINES))
    permutations=[permutations[i] for i in (0,3,4,1,2,5)]
    jobs=[]
    for index,case in enumerate(corpus):
        for repeat in range(config['language_repeats']):
            jobs.append(dict(job_id=f'{case["case_id"]}-r{repeat+1}',case_id=case['case_id'],repeat=repeat,
                engines=list(permutations[(index*config['language_repeats']+repeat)%6])))
    random.Random(config['schedule_seed']).shuffle(jobs)
    calls={engine:sum(needs_hat(c['question'],engine) for c in corpus)*config['language_repeats'] for engine in ENGINES}
    return dict(main=jobs,expected_attempts=len(jobs)*3,expected_main_generations=sum(calls.values()),
        calls_by_engine=calls,expected_warmup_generations=len(WARMUPS))


def score_result(result,case):
    expected=case['expected_command']
    interpreted=dict(operation=result['operation'],argument=result['argument']) if result['operation'] else None
    accepted=result['status'] in ('ok','not_found')
    intent=(interpreted==expected and (expected is not None or result['status']=='abstain'))
    raw=result['candidate_command']
    raw_intent=(raw==expected if expected else raw==dict(operation='ABSTAIN',argument=None)) if result['hat_called'] else None
    final=bool(intent and all(result[k]==v for k,v in case['expected_result'].items()))
    return dict(intent_correct=intent,raw_intent_correct=raw_intent,final_correct=final,
                incorrect_accepted=bool(accepted and not intent),supported_resolved=bool(case['supported'] and final))


def prepare_source(directory,config,inventory):
    source=Path(config['source_run']).resolve()
    checks=verify_artifacts(source)
    names=('config.json','manifest.json','summary.json','outcome.json','hybrid-environment.json',
           'hybrid-fixtures.json','hybrid.jsonl')
    if not all(n in checks for n in names):
        raise ValueError('Source must contain checksummed hybrid-validation evidence')
    old=json.loads((source/'config.json').read_text());summary=json.loads((source/'summary.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    if old['profile']!='hybrid-validation' or not summary.get('protocol_completed'):
        raise ValueError('Source hybrid protocol must have completed')
    groups=summary.get('groups',[])
    if {g['target_tokens'] for g in groups}!={128,512,1536} or not all(
        g['quality'][w]['observed']==g['quality'][w]['final_correct']==g['quality'][w]['planned']==80
        for g in groups for w in ('cpu','hybrid')):
        raise ValueError('Source CPU and hybrid subgroups must be complete and correct')
    if any(manifest[k]!=inventory[k] for k in ('model_sha256','hailort_cli')):
        raise ValueError('Source model/runtime identity differs')
    source_rows,errors=read_jsonl(source/'hybrid.jsonl')
    if errors:
        raise ValueError('Source observations are damaged')
    for w in ('cpu','hybrid'):
        rs=[r for r in source_rows if r['event']=='response' and r['phase']=='main' and r['workflow']==w]
        if len(rs)!=240 or len({(r['job_id'],r['turn']) for r in rs})!=240 or not all(r['final_correct'] for r in rs):
            raise ValueError('Source raw subgroup coverage disagrees with summary')
    frozen=directory/'source-run';frozen.mkdir()
    for name in (*names,'checksums.json'):shutil.copy2(source/name,frozen/name)
    fixtures=json.loads((source/'hybrid-fixtures.json').read_text())
    for f in fixtures:
        if index_fixture(f).table_sha256!=f['table_sha256']:
            raise ValueError('Source table hash differs')
    corpus=build_corpus(fixtures);schedule=make_schedule(corpus,config)
    atomic_json(directory/'language-fixtures.json',fixtures)
    atomic_json(directory/'corpus.json',corpus)
    atomic_json(directory/'schedule.json',schedule)
    atomic_json(directory/'prompt-spec.json',dict(system=SYSTEM,warmups=WARMUPS))
    atomic_json(directory/'source-provenance.json',dict(source_run=str(source),verified_artifacts=len(checks),
        source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={n:digest_file(frozen/n) for n in (*names,'checksums.json')},
        evidence_description='Previously measured tables; newly frozen English questions. Repeats are not independent evidence.'))
    if config.get('replay_run'):
        replay=Path(config['replay_run']).resolve();replay_checks=verify_artifacts(replay)
        replay_names=('config.json','manifest.json','corpus.json','schedule.json','prompt-spec.json',
                      'language-fixtures.json','language-environment.json')
        if not all(n in replay_checks for n in replay_names):raise ValueError('Replay artifacts missing')
        prior=json.loads((replay/'config.json').read_text())
        for key in ('profile','language_repeats','schedule_seed'):
            if prior[key]!=config[key]:raise ValueError(f'Replay protocol differs: {key}')
        prior_manifest=json.loads((replay/'manifest.json').read_text())
        if any(prior_manifest[k]!=inventory[k] for k in ('model_sha256','hailort_cli')):
            raise ValueError('Replay runtime differs')
        for name in ('corpus.json','schedule.json','prompt-spec.json','language-fixtures.json'):
            if json.loads((directory/name).read_text())!=json.loads((replay/name).read_text()):
                raise ValueError(f'Regenerated replay inputs differ: {name}')
        folder=directory/'replay-source';folder.mkdir()
        for name in (*replay_names,'checksums.json'):shutil.copy2(replay/name,folder/name)
        atomic_json(directory/'replay-provenance.json',dict(replay_run=str(replay),verified_artifacts=len(replay_checks),new_questions=False))


def run(directory,config,source_environment='hybrid-environment.json'):
    query=config['profile']=='language-query'
    if query:
        start=time.perf_counter();index=FactIndex(json.loads((directory/'facts.json').read_text()))
        index_ms=(time.perf_counter()-start)*1000
        if not needs_hat(config['question'],config['engine']):
            result=ask(index,config['question'],config['engine'])
            result['index_build_ms']=index_ms
            atomic_json(directory/'result.json',result)
            emit(directory/EVENT_FILE,'response',phase='query',**result)
            emit(directory/EVENT_FILE,'complete')
            return
    from hailo_platform import VDevice, __version__
    from hailo_platform.pyhailort.pyhailort import LLM
    progress(directory,'hat','model_loading');start=time.perf_counter()
    with VDevice() as device, LLM(device,config['model']) as llm:
        loading=time.perf_counter()-start
        env=dict(hailort_version=__version__,prompt_template=llm.prompt_template(),stop_tokens=llm.get_stop_tokens(),
            generation_recovery_sequence=llm.get_generation_recovery_sequence(),capacity_tokens=llm.max_context_capacity(),
            model_defaults=resolve_defaults(llm,__version__))
        if not query:
            source=json.loads((directory/'source-run'/source_environment).read_text())
            if any(env[k]!=source[k] for k in env):raise RuntimeError('Runtime metadata differs from source')
        params=parameter_settings(env['model_defaults'],'penalty_1_0')[0]
        limit=min(1792,env['capacity_tokens']-256)
        env.update(parameters=params,loading_seconds=loading,experiment_limit_tokens=limit,
            timing_definition='Continuous parsing/routing, token/context checks, verified reset, generation/cleanup, command validation and CPU execution. File logging, index construction and model loading are separate.')
        atomic_json(directory/'language-environment.json',env)
        render=renderer(env['prompt_template']);stops=env['stop_tokens']
        if query:
            wait_cool(directory)
            result=ask(index,config['question'],config['engine'],llm,render,params,stops,limit)
            result['index_build_ms']=index_ms
            atomic_json(directory/'result.json',result)
            emit(directory/EVENT_FILE,'response',phase='query',**result)
        else:
            if config.get('replay_run') and params!=json.loads((directory/'replay-source/language-environment.json').read_text())['parameters']:
                raise RuntimeError('Replay generation parameters differ')
            fixtures=json.loads((directory/'language-fixtures.json').read_text())
            corpus=json.loads((directory/'corpus.json').read_text());schedule=json.loads((directory/'schedule.json').read_text())
            counts=[]
            for case in corpus:
                progress(directory,'hat','check_frozen_prompt',case_id=case['case_id'])
                n=len(llm.tokenize(render(messages(case['question']))))
                if n+16+len(llm.tokenize(llm.get_generation_recovery_sequence()))>limit:
                    raise RuntimeError('A frozen evaluation question exceeds the context budget')
                counts.append(dict(case_id=case['case_id'],prompt_tokens=n))
            atomic_json(directory/'prompt-token-counts.json',counts)
            indexes={}
            for f in fixtures:
                start=time.perf_counter();indexes[f['fixture_id']]=index_fixture(f)
                emit(directory/EVENT_FILE,'index_built',fixture_id=f['fixture_id'],rows=len(f['table']),
                    index_build_ms=(time.perf_counter()-start)*1000,table_sha256=indexes[f['fixture_id']].table_sha256)
            atomic_json(directory/'example-records.json',[dict(item=k,color=v) for k,v in fixtures[0]['table']])
            warm_index=indexes[fixtures[0]['fixture_id']]
            for number,question in enumerate(WARMUPS):
                progress(directory,'hat','language_warmup',warmup=number);wait_cool(directory)
                result=ask(warm_index,question,'hat',llm,render,params,stops,limit)
                emit(directory/EVENT_FILE,'response',phase='warmup',warmup=number,**result)
            by_id={c['case_id']:c for c in corpus}
            for number,job in enumerate(schedule['main'],1):
                case=by_id[job['case_id']]
                for engine in job['engines']:
                    progress(directory,'hat','language_request',job_id=job['job_id'],engine=engine);wait_cool(directory)
                    result=ask(indexes[case['fixture_id']],case['question'],engine,llm,render,params,stops,limit)
                    result.update(score_result(result,case))
                    emit(directory/EVENT_FILE,'response',phase='main',job_id=job['job_id'],repeat=job['repeat'],
                        engine_order=job['engines'].index(engine),**{**case,**result})
                if number%10==0:
                    print(f'Language validation {number}/{len(schedule["main"])}: {job["job_id"]}',flush=True)
        reset_context(llm)
    emit(directory/EVENT_FILE,'complete')
