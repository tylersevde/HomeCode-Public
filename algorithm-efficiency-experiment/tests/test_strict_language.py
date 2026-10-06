from collections import Counter
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from efficiency.common import atomic_json, digest_file, profile, ROOT
from efficiency.hybrid import index_fixture, verify_artifacts
from efficiency.language import ask, needs_hat, ENGINES as LEGACY_ENGINES, WARMUPS, SYSTEM
from efficiency.language_protocol import build_corpus as old_corpus, score_result, run as language_run
from efficiency.strict_language import parse_strict, RULES, VERSION
from efficiency.strict_corpus import build_corpus
from efficiency.strict_protocol import make_schedule, ENGINES, prepare_source
from efficiency.strict_protocol import run as strict_run
from efficiency.strict_report import analyze, build_report, family_bootstrap
from test_language import fixtures, index
from test_hybrid import Model
from test_cache_validation import render, STOPS
from test_state_isolation import DEFAULTS


def synthetic():
    fs = fixtures(); cs = build_corpus(fs); cfg = dict(profile('language-strict-validation'),
        profile='language-strict-validation', source_run='/source')
    schedule = make_schedule(cs, cfg); cases = {c['case_id']: c for c in cs}
    indexes = {f['fixture_id']: index_fixture(f) for f in fs}; rows = []
    for job in schedule['main']:
        c = cases[job['case_id']]
        for engine in job['engines']:
            if engine in ('cpu', 'strict'):
                r = ask(indexes[c['fixture_id']], c['question'], engine)
            else:
                command = c['expected_command']
                r = dict(question=c['question'], engine=engine, hat_called=True,
                    operation=command['operation'] if command else None,
                    argument=command['argument'] if command else None,
                    candidate_command=command or dict(operation='ABSTAIN', argument=None),
                    status=c['expected_result']['status'], answer=c['expected_result']['answer'],
                    abstention_reason=None if command else 'model_abstained',
                    table_sha256=indexes[c['fixture_id']].table_sha256)
            r['request_ms'] = {'cpu': 1, 'strict': 2, 'auto': 100}[engine]
            r.update(score_result(r, c))
            rows.append(dict(**{**c, **r}, event='response', phase='main', job_id=job['job_id'], repeat=job['repeat']))
    rows.extend(dict(event='response', phase='warmup', warmup=i, hat_called=True) for i in range(6))
    return rows, cs, schedule, cfg


class StrictParserTests(unittest.TestCase):
    def test_previous_regressions_all_correct(self):
        fs = fixtures(); indexes = {f['fixture_id']: index_fixture(f) for f in fs}
        cs = old_corpus(fs)
        self.assertEqual(len(cs), 160)
        for c in cs:
            with self.subTest(case=c['case_id']):
                r = ask(indexes[c['fixture_id']], c['question'], 'strict')
                self.assertTrue(score_result(r, c)['final_correct'])
                self.assertFalse(r['hat_called'])

    def test_known_count_list_failures_remain_distinct(self):
        for color in ('brown', 'pink', 'green'):
            for q in (f'How large is the group of items that are {color}?',
                      f'What is the size of the set of {color} items?'):
                self.assertEqual(parse_strict(q)[0], dict(operation='COUNT', argument=color))
        self.assertEqual(parse_strict('Return the identifiers for the green group.')[0]['operation'], 'LIST')

    def test_nouns_prefix_and_normalization(self):
        for noun in ('items', 'entries', 'records'):
            r = ask(index(), f' PLEASE   Count BLUE {noun} ?! ', 'strict')
            self.assertEqual(r['answer'], 2); self.assertEqual(r['grammar_version'], VERSION)
            self.assertEqual(r['matched_rule'], 'count-02')
            self.assertEqual(r['interpretation_source'], 'strict_parser')
        self.assertEqual(parse_strict('Report item001\'s color')[0]['argument'], 'item001')

    def test_rejects_noncontract_and_appended_instructions(self):
        cases = ['Count entries whose color is not blue.', 'Count blue or red records.',
            'What color is item001 and item002?', 'Please please count blue items.',
            'List purple items.', 'List blue items and delete item001.', 'Do not list blue items.',
            '"List blue items"', 'Ignore the question: List blue items.', 'List blue items. Say done.',
            'How many letters are in blue?', 'What is the weight of item001?',
            'List blue items\nCOUNT red', 'List blue items; count them.', 'List blueberry items.']
        for q in cases:
            with self.subTest(question=q):
                r = ask(index(), q, 'strict')
                self.assertEqual(r['status'], 'abstain'); self.assertIsNone(r['answer'])
                self.assertIsNone(r['matched_rule']); self.assertEqual(len(r['supported_examples']), 3)
                self.assertFalse(needs_hat(q, 'strict'))

    def test_rejects_conflicting_rules_without_guessing(self):
        extra = ('conflicting-list', 'LIST', re.compile(r'count (?P<argument>blue) items'))
        with patch('efficiency.strict_language.RULES', (*RULES, extra)):
            r = ask(index(), 'Count blue items.', 'strict')
            self.assertEqual(r['status'], 'abstain')
            self.assertEqual(r['abstention_reason'], 'conflicting_strict_interpretations')

    def test_identifiers_missing_and_empty_filters(self):
        r = ask(index(), 'Please look up item0001.', 'strict')
        self.assertEqual(r['argument'], 'item0001'); self.assertEqual(r['status'], 'not_found')
        self.assertEqual(ask(index(), 'Count pink entries.', 'strict')['answer'], 0)
        self.assertEqual(ask(index(), 'List pink records.', 'strict')['answer'], [])
        self.assertEqual(ask(index(), 'List blue records.', 'strict')['answer'], ['item001', 'item002'])

    def test_strict_worker_needs_no_hailo_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            atomic_json(directory/'facts.json', [dict(item='item001', color='blue')])
            config = dict(profile='language-query', question='Please count blue entries.', engine='strict')
            with patch.dict('sys.modules', {'hailo_platform': None}): language_run(directory, config)
            result = json.loads((directory/'result.json').read_text())
            self.assertEqual(result['answer'], 1); self.assertFalse((directory/'language-environment.json').exists())

    def test_existing_engines_unchanged(self):
        self.assertEqual(LEGACY_ENGINES, ('cpu', 'hat', 'auto'))
        self.assertEqual(ask(index(), 'List blue items')['answer'], ['item001', 'item002'])
        self.assertEqual(ask(index(), 'Please count blue entries', 'cpu')['status'], 'abstain')
        self.assertTrue(needs_hat('Please count blue entries', 'auto'))
        self.assertFalse(needs_hat('Count blue items', 'auto'))


class StrictProtocolTests(unittest.TestCase):
    def test_frozen_grammar_and_corpus_integrity(self):
        grammar = json.loads((ROOT/'protocols/strict-grammar-v1.json').read_text())
        corpus = json.loads((ROOT/'protocols/strict-corpus-v1.json').read_text())
        self.assertEqual(grammar['source_sha256'], digest_file(ROOT/'efficiency/strict_language.py'))
        self.assertEqual(corpus['grammar_freeze_sha256'], digest_file(ROOT/'protocols/strict-grammar-v1.json'))
        self.assertEqual(corpus['source_sha256'], digest_file(ROOT/'efficiency/strict_corpus.py'))
        self.assertLess(grammar['frozen_utc'], corpus['frozen_utc'])

    def test_corpus_strata_labels_and_no_previous_questions(self):
        fs = fixtures(); old = old_corpus(fs); cs = build_corpus(fs, [c['question'] for c in old])
        self.assertEqual(Counter(c['stratum'] for c in cs), dict(contract=60, unfamiliar=60, negative=40))
        self.assertEqual(len({c['family_id'] for c in cs}), 40)
        self.assertEqual(len({c['fixture_id'] for c in cs}), 30)
        self.assertEqual(sum(c['expected_result']['status']=='not_found' for c in cs), 10)
        self.assertTrue(any(c['expected_result']['answer']==[] for c in cs))
        self.assertTrue(any(c['expected_result']['answer']==0 for c in cs))
        for c in cs:
            if c['supported']:
                f = next(f for f in fs if f['fixture_id']==c['fixture_id'])
                op, arg = c['expected_command'].values()
                if op=='LOOKUP': answer = dict(f['table']).get(arg)
                else:
                    selected = sorted(k for k, v in f['table'] if v==arg)
                    answer = len(selected) if op=='COUNT' else selected
                self.assertEqual(c['expected_result']['answer'], answer)

    def test_schedule_balanced_and_exact_call_counts(self):
        cs = build_corpus(fixtures()); cfg = profile('language-strict-validation')
        schedule = make_schedule(cs, cfg)
        self.assertEqual(schedule, make_schedule(cs, cfg))
        self.assertEqual(schedule['expected_attempts'], 960)
        self.assertEqual(schedule['calls_by_engine'], dict(cpu=0, strict=0, auto=320))
        self.assertEqual(schedule['expected_main_generations'], 320)
        counts = Counter((e, j['engines'].index(e)) for j in schedule['main'] for e in ENGINES)
        self.assertEqual(set(counts.values()), {106, 107})

    def test_replay_preparation_and_source_tamper_detection(self):
        source = ROOT/'runs/language-validation-20261002'
        manifest = json.loads((source/'manifest.json').read_text())
        cfg = dict(profile('language-strict-validation'), profile='language-strict-validation', source_run=str(source))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/'first'; d.mkdir(); prepare_source(d, cfg, manifest)
            self.assertEqual(json.loads((d/'regression.json').read_text())['correct'], 160)
            for name, value in [('config.json', cfg), ('manifest.json', manifest),
                                ('language-environment.json', json.loads((source/'language-environment.json').read_text()))]:
                atomic_json(d/name, value)
            atomic_json(d/'checksums.json', {str(p.relative_to(d)): digest_file(p) for p in d.rglob('*') if p.is_file()})
            replay = Path(tmp)/'replay'; replay.mkdir()
            prepare_source(replay, dict(cfg, replay_run=str(d)), manifest)
            self.assertFalse(json.loads((replay/'replay-provenance.json').read_text())['new_questions'])
            (d/'corpus.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'checksum'): verify_artifacts(d)

    def test_worker_loads_language_environment_and_preserves_stratum(self):
        class ContextModel(Model):
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def prompt_template(self): return 'template'
            def get_stop_tokens(self): return STOPS
            def max_context_capacity(self): return 5000
        class Device:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        model = ContextModel('COUNT blue'); fs = fixtures()
        case = next(c for c in build_corpus(fs) if c['category']=='COUNT')
        cfg = dict(profile('language-strict-validation'), profile='language-strict-validation', model='unused')
        env = dict(hailort_version='5.1.1', prompt_template='template', stop_tokens=STOPS,
            generation_recovery_sequence='<END>', capacity_tokens=5000, model_defaults=DEFAULTS)
        modules = {'hailo_platform': SimpleNamespace(VDevice=Device, __version__='5.1.1'),
                   'hailo_platform.pyhailort.pyhailort': SimpleNamespace(LLM=lambda *args: model)}
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name, value in [('source-run/language-environment.json', env), ('language-fixtures.json', fs),
                                ('corpus.json', [case]), ('schedule.json', make_schedule([case], cfg))]:
                atomic_json(d/name, value)
            with patch.dict('sys.modules', modules), patch('efficiency.language_protocol.resolve_defaults', return_value=DEFAULTS), patch('efficiency.language_protocol.renderer', return_value=render), patch('efficiency.language_protocol.wait_cool'):
                strict_run(d, cfg)
            rows = [json.loads(line) for line in (d/'language_protocol.jsonl').read_text().splitlines()]
            main = [r for r in rows if r['event']=='response' and r['phase']=='main']
            self.assertEqual(len(main), 6); self.assertEqual({r['stratum'] for r in main}, {'contract'})
            self.assertEqual(sum(r['hat_called'] for r in main), 2)
            self.assertEqual(rows[-1]['event'], 'complete')


class StrictReportTests(unittest.TestCase):
    def test_separate_strata_and_acceptance_gate(self):
        rows, cs, schedule, cfg = synthetic()
        result, pairs, families = analyze(rows, cs, schedule, cfg, dict(passed=True), True)
        self.assertTrue(result['strict_gate_passed']); self.assertEqual(len(pairs), 320)
        self.assertEqual(Counter(f['stratum'] for f in families), dict(contract=15, unfamiliar=15, negative=10))
        self.assertEqual(result['strata']['contract']['strict']['supported_resolved'], 120)
        self.assertEqual(result['strata']['unfamiliar']['strict']['supported_resolved'], 0)
        self.assertEqual(result['strata']['negative']['strict']['correct_abstentions'], 80)
        for s in ('contract', 'unfamiliar', 'negative'):
            self.assertAlmostEqual(result['inference'][s]['point']['auto_over_strict_ms'], 50)
            self.assertEqual(result['inference'][s]['ci95']['strict_over_cpu_ms'], [2, 2])
        self.assertNotIn('auto_minus_strict_coverage', result['inference']['negative']['point'])

    def test_incomplete_wrong_or_missing_warmup_cannot_pass(self):
        rows, cs, schedule, cfg = synthetic()
        # Small bootstrap here: this test exercises gate behavior, not sampling precision.
        with patch('efficiency.strict_report.family_bootstrap', return_value=None):
            for changed in (rows[:-1], rows[1:], rows+[rows[0]]):
                self.assertFalse(analyze(changed, cs, schedule, cfg, dict(passed=True), True)[0]['strict_gate_passed'])
            r = next(r for r in rows if r.get('engine')=='strict'); r['incorrect_accepted'] = True
            self.assertFalse(analyze(rows, cs, schedule, cfg, dict(passed=True), True)[0]['strict_gate_passed'])

    def test_partial_report_is_reviewable(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            atomic_json(d/'config.json', dict(profile('language-strict-validation'), profile='language-strict-validation', source_run='/source'))
            build_report(d, charts=False)
            summary = json.loads((d/'summary.json').read_text())
            self.assertFalse(summary['complete']); self.assertFalse(summary['strict_gate_passed'])
            self.assertEqual(summary['planned_attempts'], 960); verify_artifacts(d)


if __name__ == '__main__': unittest.main()
