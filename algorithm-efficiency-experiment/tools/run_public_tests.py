#!/usr/bin/env python3
"""Run unchanged source tests in a disposable copy; no published-tree writes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PRIVATE_TESTS = {
    'test_refine.ScoringTests.test_actual_llama_multiturn_template':
        'Requires original private study-npu-develop-retry1-20261003/study.jsonl environment evidence.',
    'test_strict_language.StrictProtocolTests.test_replay_preparation_and_source_tamper_detection':
        'Requires complete original sealed language-validation-20261002 archive pinned by the historical corpus freeze.'}


def each(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from each(test)
        else:
            yield test


def run_child():
    sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / 'tests')]
    suite = unittest.defaultTestLoader.discover('tests')
    for test in each(suite):
        if test.id() in PRIVATE_TESTS:
            reason = PRIVATE_TESTS[test.id()]
            setattr(test, test._testMethodName, lambda reason=reason: (_ for _ in ()).throw(unittest.SkipTest(reason)))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=[dict(test=t.id(), reason=reason) for t, reason in result.skipped],
        passed=result.wasSuccessful()), indent=2))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, nargs='?', default=Path(__file__).resolve().parents[1])
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return run_child()
    root = args.root.resolve()
    from validate_public import verify_manifest
    manifest = verify_manifest(root)
    with tempfile.TemporaryDirectory(prefix='public-offline-tests-') as name:
        working = Path(name)
        paths = set(manifest['original_source_sha256'])
        paths.update(p for p in manifest['files'] if p.startswith(('protocols/', 'reference/')))
        paths.add('tools/run_public_tests.py')
        for name in sorted(paths):
            dst = working / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, dst)
        (working / 'runs').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='tests:.')
        return subprocess.run([sys.executable, '-B', str(working / 'tools/run_public_tests.py'), '--child'],
                              cwd=working, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
