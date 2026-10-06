"""Frozen strict-grammar evaluation, with the existing interpreter as a comparator."""
import itertools
import json
from pathlib import Path
import random
import shutil

from .common import ROOT, atomic_json, digest_file, digest_value, read_jsonl
from .hybrid import index_fixture, verify_artifacts
from .language import ask, needs_hat, SYSTEM, WARMUPS
from .language_protocol import score_result, run as language_run
from .strict_corpus import build_corpus
from .strict_language import specification

ENGINES = ('cpu', 'strict', 'auto')
STRATA = ('contract', 'unfamiliar', 'negative')
PROFILE = 'language-strict-validation'


def make_schedule(corpus, config):
    permutations = list(itertools.permutations(ENGINES))
    permutations = [permutations[i] for i in (0, 3, 4, 1, 2, 5)]
    jobs = []
    for number, case in enumerate(corpus):
        for repeat in range(config['language_repeats']):
            jobs.append(dict(job_id=f'{case["case_id"]}-r{repeat+1}', case_id=case['case_id'], repeat=repeat,
                engines=list(permutations[(number*config['language_repeats']+repeat) % 6])))
    random.Random(config['schedule_seed']).shuffle(jobs)
    calls = {engine: sum(needs_hat(c['question'], engine) for c in corpus)*config['language_repeats']
             for engine in ENGINES}
    return dict(main=jobs, engines=ENGINES, expected_attempts=len(jobs)*3,
        expected_main_generations=sum(calls.values()), calls_by_engine=calls,
        expected_warmup_generations=len(WARMUPS))


def prepare_source(directory, config, inventory):
    source = Path(config['source_run']).resolve()
    checks = verify_artifacts(source)
    names = ('config.json', 'manifest.json', 'summary.json', 'outcome.json', 'language-environment.json',
             'language-fixtures.json', 'corpus.json', 'schedule.json', 'prompt-spec.json', 'language_protocol.jsonl')
    if not all(n in checks for n in names): raise ValueError('Verified language source artifacts required')
    read = lambda name: json.loads((source/name).read_text())
    old, summary, manifest = read('config.json'), read('summary.json'), read('manifest.json')
    if old['profile'] != 'language-validation' or not summary.get('complete') or not summary.get('protocol_completed'):
        raise ValueError('A completed language-validation source is required; its accuracy gate may have failed')
    if any(manifest[k] != inventory[k] for k in ('model_sha256', 'hailort_cli')):
        raise ValueError('Source model/runtime identity differs')
    rows, errors = read_jsonl(source/'language_protocol.jsonl')
    main = [r for r in rows if r['event']=='response' and r['phase']=='main']
    expected = {(j['job_id'], e) for j in read('schedule.json')['main'] for e in j['engines']}
    if errors or len(main) != 960 or len(expected) != 960 or {(r['job_id'], r['engine']) for r in main} != expected:
        raise ValueError('Source raw attempt coverage differs')
    grammar_path, corpus_path = [ROOT/'protocols'/n for n in ('strict-grammar-v1.json', 'strict-corpus-v1.json')]
    grammar, frozen_corpus = json.loads(grammar_path.read_text()), json.loads(corpus_path.read_text())
    if grammar['source_sha256'] != digest_file(ROOT/'efficiency/strict_language.py') or digest_value(grammar['grammar']) != digest_value(specification()):
        raise ValueError('Runtime grammar differs from pre-corpus freeze')
    if (frozen_corpus['grammar_freeze_sha256'] != digest_file(grammar_path)
            or frozen_corpus['source_sha256'] != digest_file(ROOT/'efficiency/strict_corpus.py')
            or frozen_corpus['source_checksums_sha256'] != digest_file(source/'checksums.json')
            or grammar['frozen_utc'] >= frozen_corpus['frozen_utc']):
        raise ValueError('Corpus freeze provenance differs')
    fixtures, previous = read('language-fixtures.json'), read('corpus.json')
    indexes = {f['fixture_id']: index_fixture(f) for f in fixtures}
    if len(indexes) != 30 or any(indexes[f['fixture_id']].table_sha256 != f['table_sha256'] for f in fixtures):
        raise ValueError('Source table integrity differs')
    corpus = build_corpus(fixtures, [c['question'] for c in previous])
    if corpus != frozen_corpus['corpus']: raise ValueError('Regenerated corpus differs from authored freeze')
    if read('prompt-spec.json') != dict(system=SYSTEM, warmups=WARMUPS):
        raise ValueError('Interpreter prompt differs from source')
    frozen = directory/'source-run'; frozen.mkdir()
    for name in (*names, 'checksums.json'): shutil.copy2(source/name, frozen/name)
    shutil.copy2(grammar_path, directory/grammar_path.name)
    shutil.copy2(corpus_path, directory/corpus_path.name)
    schedule = make_schedule(corpus, config)
    for name, value in [('corpus.json', corpus), ('language-fixtures.json', fixtures),
                        ('schedule.json', schedule), ('prompt-spec.json', dict(system=SYSTEM, warmups=WARMUPS))]:
        atomic_json(directory/name, value)
    regression = []
    for case in previous:
        result = ask(indexes[case['fixture_id']], case['question'], 'strict')
        regression.append(dict(case_id=case['case_id'], question=case['question'],
            expected_command=case['expected_command'], expected_result=case['expected_result'],
            result=result, scores=score_result(result, case)))
    atomic_json(directory/'regression.json', dict(observed=len(regression), planned=160,
        correct=sum(r['scores']['final_correct'] for r in regression),
        passed=len(regression)==160 and all(r['scores']['final_correct'] for r in regression), cases=regression,
        evidence='Previously observed questions; regression only, not new language evidence.'))
    atomic_json(directory/'source-provenance.json', dict(source_run=str(source), verified_artifacts=len(checks),
        source_checksums_sha256=digest_file(source/'checksums.json'),
        frozen_sha256={n: digest_file(frozen/n) for n in (*names, 'checksums.json')},
        grammar_freeze_sha256=digest_file(grammar_path), corpus_freeze_sha256=digest_file(corpus_path),
        evidence='Reused tables. Contract combinations and unfamiliar paraphrases are reported separately.'))
    if config.get('replay_run'):
        replay = Path(config['replay_run']).resolve(); replay_checks = verify_artifacts(replay)
        replay_names = ('config.json', 'manifest.json', 'corpus.json', 'schedule.json', 'prompt-spec.json',
                        'language-fixtures.json', 'language-environment.json', 'strict-grammar-v1.json', 'strict-corpus-v1.json')
        if not all(n in replay_checks for n in replay_names): raise ValueError('Replay artifacts missing')
        prior = json.loads((replay/'config.json').read_text())
        if any(prior[k] != config[k] for k in ('profile', 'language_repeats', 'schedule_seed')):
            raise ValueError('Replay protocol differs')
        prior_manifest = json.loads((replay/'manifest.json').read_text())
        if any(prior_manifest[k] != inventory[k] for k in ('model_sha256', 'hailort_cli')):
            raise ValueError('Replay runtime differs')
        for name in replay_names[2:]:
            if name == 'language-environment.json': continue
            if json.loads((directory/name).read_text()) != json.loads((replay/name).read_text()):
                raise ValueError(f'Regenerated replay inputs differ: {name}')
        folder = directory/'replay-source'; folder.mkdir()
        for name in (*replay_names, 'checksums.json'): shutil.copy2(replay/name, folder/name)
        atomic_json(directory/'replay-provenance.json', dict(replay_run=str(replay), verified_artifacts=len(replay_checks), new_questions=False))


def run(directory, config):
    language_run(directory, config, source_environment='language-environment.json')
