"""Offline interpretation accuracy, routing coverage and family-clustered costs."""
import base64
from collections import Counter
import html
import json
import random
import shlex
import statistics

from .common import ROOT, atomic_json, digest_file, read_jsonl, utc
from .language import ENGINES
from .report import median, table, write_csv


def quality(rows,planned,supported_planned):
    supported=[r for r in rows if r['supported']]
    negative=[r for r in rows if not r['supported']]
    raw=[r for r in rows if r['hat_called']]
    return dict(planned=planned,observed=len(rows),intent_correct=sum(r['intent_correct'] for r in rows),
        final_correct=sum(r['final_correct'] for r in rows),supported_planned=supported_planned,
        supported_resolved=sum(r['supported_resolved'] for r in supported),
        required_abstentions=planned-supported_planned,correct_abstentions=sum(r['status']=='abstain' for r in negative),
        incorrect_accepted=sum(r['incorrect_accepted'] for r in rows),
        supported_abstentions=sum(r['status']=='abstain' for r in supported),
        hat_calls=len(raw),raw_intent_correct=sum(r['raw_intent_correct'] is True for r in raw),
        abstention_reasons=dict(Counter(r['abstention_reason'] for r in rows if r['abstention_reason'])),
        request_total_ms=sum(r['request_ms'] for r in rows),request_median_ms=median([r['request_ms'] for r in rows]))


def family_bootstrap(families,seed,samples=10000):
    if len(families)<2:return None
    def compute(fs):
        auto=sum(f['auto_ms'] for f in fs);hat=sum(f['hat_ms'] for f in fs)
        n=sum(f['supported_planned'] for f in fs)
        gain=(sum(f['auto_resolved']-f['cpu_resolved'] for f in fs)/n) if n else None
        return hat/auto,gain
    rng=random.Random(seed);ratios=[];gains=[]
    for _ in range(samples):
        speed,gain=compute(rng.choices(families,k=len(families)))
        ratios.append(speed)
        if gain is not None:gains.append(gain)
    ratios.sort();gains.sort();speed,gain=compute(families)
    interval=lambda xs:[xs[int(.025*len(xs))],xs[int(.975*len(xs))]] if xs else None
    return dict(hat_over_auto=speed,hat_over_auto_ci95=interval(ratios),
        supported_coverage_gain=gain,supported_coverage_gain_ci95=interval(gains),samples=samples,
        independent_families=len(families))


def analyze(rows,corpus,schedule,config,protocol_ok):
    observed=[r for r in rows if r['event']=='response' and r['phase']=='main']
    indexed={(r['job_id'],r['engine']):r for r in observed}
    expected={(j['job_id'],e) for j in schedule['main'] for e in ENGINES}
    planned=160*config['language_repeats']
    duplicates=len(observed)-len(indexed)
    complete=set(indexed)==expected and len(expected)==planned*3==960 and duplicates==0
    supported=120*config['language_repeats']
    groups={e:quality([r for r in observed if r['engine']==e],planned,supported) for e in ENGINES}
    categories=[]
    for category in sorted({c['category'] for c in corpus}):
        cs=[c for c in corpus if c['category']==category]
        for engine in ENGINES:
            rs=[r for r in observed if r['category']==category and r['engine']==engine]
            categories.append(dict(category=category,engine=engine,
                **quality(rs,len(cs)*config['language_repeats'],sum(c['supported'] for c in cs)*config['language_repeats'])))
    pairs=[]
    by_case={c['case_id']:c for c in corpus}
    for j in schedule['main']:
        rs={e:indexed.get((j['job_id'],e)) for e in ENGINES}
        reason=None
        if not all(rs.values()):reason='missing_engine'
        elif len({(r['question'],r['table_sha256']) for r in rs.values()})!=1:reason='different_task'
        elif any(r['request_ms']<=0 for r in rs.values()):reason='invalid_timing'
        case=by_case[j['case_id']]
        p=dict(job_id=j['job_id'],case_id=j['case_id'],family_id=case['family_id'],repeat=j['repeat'],
            eligible=reason is None,exclusion_reason=reason,supported=case['supported'])
        if reason is None:
            for e,r in rs.items():p.update({e+'_ms':r['request_ms'],e+'_resolved':int(r['supported_resolved'])})
        pairs.append(p)
    families=[]
    for fid in sorted({c['family_id'] for c in corpus}):
        group=[p for p in pairs if p['family_id']==fid]
        n=sum(c['family_id']==fid for c in corpus)*config['language_repeats']
        if len(group)==n and all(p['eligible'] for p in group):
            families.append(dict(family_id=fid,attempts_per_engine=n,
                supported_planned=sum(p['supported'] for p in group),
                **{e+'_ms':sum(p[e+'_ms'] for p in group) for e in ENGINES},
                **{e+'_resolved':sum(p[e+'_resolved'] for p in group) for e in ENGINES}))
    inference=family_bootstrap(families,config['schedule_seed'])
    calls_match=all(groups[e]['hat_calls']==schedule.get('calls_by_engine',{}).get(e) for e in ENGINES)
    gates=dict(complete_coverage=bool(complete and len(families)==40),
        increased_supported_coverage=groups['auto']['supported_resolved']>groups['cpu']['supported_resolved'],
        no_incorrect_accepted_auto=groups['auto']['incorrect_accepted']==0,
        fewer_hat_calls=groups['auto']['hat_calls']<groups['hat']['hat_calls'],
        lower_total_request_time=groups['auto']['request_total_ms']<groups['hat']['request_total_ms'],
        planned_call_counts_match=calls_match)
    return dict(observed_attempts=len(observed),planned_attempts=planned*3,coverage_exact=bool(complete),
        duplicate_responses=duplicates,groups=groups,categories=categories,
        inference=inference,success_criteria=gates,useful_routing=bool(protocol_ok and all(gates.values())),
        exploratory_speedup_established=bool(inference and inference['hat_over_auto_ci95'][0]>1 and complete and len(families)==40),
        context_mismatches=sum(r.get('token_ledger_valid') is False for r in observed),
        incorrect_acceptances=[dict(case_id=r['case_id'],engine=r['engine'],repeat=r['repeat'],question=r['question'],
            expected_command=r['expected_command'],operation=r['operation'],argument=r['argument'],answer=r['answer']) for r in observed if r['incorrect_accepted']]),pairs,families


def build_report(directory,charts=True):
    def read(name,default=None):return json.loads((directory/name).read_text()) if (directory/name).exists() else default
    config=read('config.json');query=config['profile']=='language-query'
    rows,errors=read_jsonl(directory/'language_protocol.jsonl')
    telemetry,telemetry_errors=read_jsonl(directory/'telemetry.jsonl')
    outcome=read('outcome.json',dict(status='incomplete'))
    protocol_ok=outcome['status']=='complete' and not errors and not telemetry_errors and any(r['event']=='complete' for r in rows)
    pairs=[];families=[]
    if query:
        result=read('result.json');summary=dict(result=result,complete=bool(protocol_ok and result is not None))
    else:
        corpus=read('corpus.json',[]);schedule=read('schedule.json',dict(main=[]))
        summary,pairs,families=analyze(rows,corpus,schedule,config,protocol_ok)
        summary['complete']=bool(protocol_ok and summary['coverage_exact'])
    summary.update(generated_utc=utc(),protocol_completed=protocol_ok,outcome=outcome,
        environment=read('language-environment.json',{}),new_questions=not bool(config.get('replay_run')) and not query,
        total_generations=sum(r['event']=='response' and r['hat_called'] for r in rows),
        warmup_generations=sum(r['event']=='response' and r['phase']=='warmup' and r['hat_called'] for r in rows),
        damaged_lines=dict(observations=errors,telemetry=telemetry_errors),
        thermal=dict(max_cpu_c=max((r['cpu_temp_c'] for r in telemetry if r.get('cpu_temp_c') is not None),default=None),
            max_hat_c=max((r['hat_max_c'] for r in telemetry if r.get('hat_max_c') is not None),default=None),
            throttle_flags=sorted({r['throttle_flags'] for r in telemetry if r.get('throttle_flags') is not None})))
    atomic_json(directory/'summary.json',summary)
    write_csv(directory/'responses.csv',[r for r in rows if r['event']=='response'])
    write_csv(directory/'paired-attempts.csv',pairs);write_csv(directory/'family-totals.csv',families)
    write_csv(directory/'quality-by-category.csv',summary.get('categories',[]))
    write_csv(directory/'indexes.csv',[r for r in rows if r['event']=='index_built'])
    write_csv(directory/'telemetry.csv',telemetry)
    if not query:
        atomic_json(directory/'reproduce.json',dict(command=shlex.join(['python3',str(ROOT/'experiment.py'),'run',
            '--phase','hat','--profile','language-validation','--source-run',config['source_run'],'--replay-run',str(directory),
            '--output',str(directory.parent/(directory.name+'-replay')),'--max-seconds','3600']),
            note='Verified replay of frozen tables, question families, prompt and schedule; not new evidence.'))
    plots=[]
    if charts and not query and summary['observed_attempts']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        xs=list(range(3));gs=[summary['groups'][e] for e in ENGINES]
        axes[0].bar(xs,[100*g['supported_resolved']/g['supported_planned'] for g in gs],color=['#3973ac','#e38b30','#27826d'])
        axes[0].set(ylabel='Correct supported queries / planned (%)',ylim=(0,105),title='Language coverage')
        axes[1].bar(xs,[g['incorrect_accepted'] for g in gs],color='#b74343')
        axes[1].set(ylabel='Incorrect accepted interpretations',title='Intent errors that passed validation')
        axes[2].bar(xs,[g['request_total_ms']/1000 for g in gs],color=['#3973ac','#e38b30','#27826d'])
        axes[2].set(ylabel='Total measured request seconds',title='Work across the same question set')
        for ax in axes:ax.set(xticks=xs,xticklabels=ENGINES)
        for ext in ('png','svg'):fig.savefig(directory/f'language-results.{ext}',dpi=160)
        plt.close(fig);plots.append('language-results')
        if telemetry:
            fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
            for key,label in [('cpu_temp_c','Pi CPU'),('hat_max_c','HAT maximum')]:
                valid=[r for r in telemetry if r.get(key) is not None]
                ax.plot([r['elapsed_seconds']/60 for r in valid],[r[key] for r in valid],label=label)
            ax.set(xlabel='Elapsed minutes',ylabel='Temperature (°C)',title='Observed temperatures');ax.legend()
            for ext in ('png','svg'):fig.savefig(directory/f'temperatures.{ext}',dpi=160)
            plt.close(fig);plots.append('temperatures')
    fmt=lambda x:'—' if x is None else f'{x:.4g}'
    sections=['<h1>Natural-language queries with CPU/HAT routing</h1>',
        f'<p>Outcome: {html.escape(outcome["status"])}. Protocol completed: {protocol_ok}. Complete coverage: {summary["complete"]}.</p>']
    if query:sections.append('<pre>'+html.escape(json.dumps(summary['result'],indent=2))+'</pre>')
    else:
        sections.extend([f'<p>Measured {summary["observed_attempts"]}/{summary["planned_attempts"]} attempts. '
            f'{summary["total_generations"]} HAT generations including {summary["warmup_generations"]} warm-ups. '
            f'Useful routing gate: {summary["useful_routing"]}.</p>',
            '<h2>Interpretation and execution</h2>',
            table(['Engine','Correct supported / planned','Correct required abstentions','Incorrect accepted','All results correct / planned','Raw HAT intents correct / calls','HAT calls','Total request seconds','Median request ms'],
                [[e,f'{g["supported_resolved"]}/{g["supported_planned"]}',f'{g["correct_abstentions"]}/{g["required_abstentions"]}',g['incorrect_accepted'],
                  f'{g["final_correct"]}/{g["planned"]}',f'{g["raw_intent_correct"]}/{g["hat_calls"]}',g['hat_calls'],
                  fmt(g['request_total_ms']/1000),fmt(g['request_median_ms'])] for e,g in summary['groups'].items()]),
            '<p>Supported queries include missing-item and empty-filter results. The CPU grammar recognizes six explicit forms; '
            'automatic routing uses that parser first. The HAT receives the question, fixed grammar and examples, never table contents '
            'or evaluation labels. Every accepted command executes against the CPU index. A structurally valid command can still '
            'misinterpret the question; those errors are counted as incorrect accepted interpretations, even if its answer coincides by chance.</p>',
            '<h2>Success criteria and uncertainty</h2><pre>'+html.escape(json.dumps(dict(criteria=summary['success_criteria'],
                family_bootstrap=summary['inference']),indent=2))+'</pre>',
            '<p>The fixed corpus contains 120 supported and 40 negative questions across 40 authored phrasing families, each repeated twice. '
            'The tables were measured previously; new question families are the evidence here. Families, not individual arguments or '
            'repetitions, are the bootstrap units. The paired 10,000-sample bootstrap recomputes ratios of corpus-total request times '
            'and supported-query coverage differences. This constructed distribution does not establish general English understanding. '
            'Success requires complete coverage, increased supported coverage, zero incorrect accepted automatic interpretations, fewer '
            'HAT calls and lower total request time than always-HAT. Accuracy failures do not trigger post-hoc prompt tuning.</p>',
            '<h2>Incorrect accepted interpretations</h2>',table(['Case','Engine','Repeat','Question','Expected','Interpreted'],
                [[r['case_id'],r['engine'],r['repeat'],r['question'],r['expected_command'],dict(operation=r['operation'],argument=r['argument'])]
                 for r in summary['incorrect_acceptances']]),
            '<h2>Timing boundaries</h2><p>Request time includes parsing/routing, token/context checks, verified reset, generation/cleanup, '
            'validation and CPU execution. Model loading, index construction, file logging and CLI startup are separate. A CPU-routed '
            'public query does not start the HAT or its sensor. In the benchmark, a model is already loaded for the interleaved HAT arms.</p>'])
    sections.append('<h2>Environment and supervision</h2><pre>'+html.escape(json.dumps(dict(thermal=summary['thermal'],
        parameters=summary['environment'].get('parameters'),model_loading_seconds=summary['environment'].get('loading_seconds'),
        session_seconds=outcome.get('elapsed_seconds'),stop_reason=outcome.get('stop_reason')),indent=2))+'</pre>')
    sections.append('<p>Raw outputs, abstention reasons, prompt identities, source provenance, executed code and checksums accompany the report. '
        'Context counts do not expose hidden generated token IDs. Power and energy are not measured.</p>')
    for stem in plots:
        data=base64.b64encode((directory/f'{stem}.png').read_bytes()).decode()
        sections.append(f'<figure><img alt="{stem}" src="data:image/png;base64,{data}"></figure>')
    css='body{font:16px/1.5 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172033}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{width:100%}'
    (directory/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Language routing</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
    atomic_json(directory/'checksums.json',{str(p.relative_to(directory)):digest_file(p) for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'checksums.json'})
    print(json.dumps(dict(report=str(directory/'report.html'),complete=summary['complete'],total_generations=summary['total_generations']),indent=2))
