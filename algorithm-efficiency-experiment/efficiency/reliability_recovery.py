"""Read-only historical evidence checks for separately budgeted recovery work."""
from copy import deepcopy
import json
from pathlib import Path

from .common import atomic_json, digest_file, utc


def read(path):
    return json.loads(Path(path).read_text())


def file_manifest(roots):
    files = {}
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f'Historical evidence directory missing: {root}')
        for path in sorted(root.rglob('*')):
            if path.is_file():
                files[str(path)] = digest_file(path)
    return files


def capture_lineage(campaign, previous):
    from .reliability_campaign import recover
    previous = Path(previous).resolve()
    ledger = read(previous / 'campaign.json')
    roots = [str(previous), *dict.fromkeys(a['output'] for a in ledger['attempts'])]
    assessment = deepcopy(ledger)
    recover(assessment)
    value = dict(utc=utc(), recovery_from=str(previous), roots=roots,
                 files=file_manifest(roots), recovered_ledger_assessment=assessment,
                 note='Historical bytes unchanged. Assessment is a copy, not new scientific evidence.')
    path = Path(campaign) / 'recovery-lineage.json'
    atomic_json(path, value)
    return digest_file(path)


def verify_lineage(campaign):
    campaign = Path(campaign)
    ledger = read(campaign / 'campaign.json')
    path = campaign / 'recovery-lineage.json'
    if digest_file(path) != ledger['recovery_lineage_sha256']:
        raise ValueError('Recovery lineage manifest changed')
    value = read(path)
    if value['recovery_from'] != ledger['recovery_from']:
        raise ValueError('Recovery lineage identity changed')
    if file_manifest(value['roots']) != value['files']:
        raise ValueError('Historical recovery evidence changed')
    return dict(passed=True, files=len(value['files']), manifest_sha256=digest_file(path))


def verify_source(validation):
    from .common import ROOT
    if not validation.get('passed') or not validation.get('source_sha256'):
        raise ValueError('Passing implementation validation and source hashes required')
    if set(validation['source_sha256']) != set(source_hashes()):
        raise ValueError('Validated source file set changed')
    for name, expected in validation['source_sha256'].items():
        if digest_file(ROOT / name) != expected:
            raise ValueError('Validated source changed: ' + name)
    return True


def source_hashes():
    """Match inventory's full source set without hardware observation."""
    from .common import ROOT
    sources = [ROOT / 'experiment.py', *sorted((ROOT / 'efficiency').glob('*.py')),
               *sorted((ROOT / 'tests').glob('*.py')), ROOT / 'vendor/hat_sensor.py',
               ROOT / 'reference/algorithm_efficiency_demo.py',
               *sorted((ROOT / 'native/attention').glob('*.cpp')),
               *sorted((ROOT / 'native/attention').glob('*.comp'))]
    return {str(p.relative_to(ROOT)): digest_file(p) for p in sources if p.is_file()}


def verify_diagnostic(campaign):
    from .hybrid import verify_artifacts
    value = read(Path(campaign) / 'diagnostic-validation.json')
    path = Path(value['output'])
    verify_artifacts(path)
    if digest_file(path / 'checksums.json') != value['checksums_sha256']:
        raise ValueError('Diagnostic seal changed')
    if value['returncode'] or not value['result'].get('passed') or read(path / 'summary.json') != value['result']:
        raise ValueError('Diagnostic qualification evidence differs')
    return dict(passed=True, output=str(path), checksums_sha256=value['checksums_sha256'])
