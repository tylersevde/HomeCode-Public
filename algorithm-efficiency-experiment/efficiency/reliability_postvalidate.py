"""Offline audit and copy-only negative checks for completed recovery campaigns."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from .common import atomic_json, read_jsonl, utc
from .hybrid import verify_artifacts
from .reliability_audit import audit
from .reliability_campaign import dependency
from .reliability_recovery import verify_lineage
from .reliability_spec import EVENTS


def read(path):
    return json.loads(Path(path).read_text())


def altered_events(events, kind):
    values = deepcopy(events)
    rows = [e for e in values if e['event'] == 'measurement' and e['section'] == 'main']
    if kind == 'order':
        indices = [i for i, e in enumerate(values) if e['event'] == 'measurement' and e['section'] == 'main']
        values[indices[0]], values[indices[1]] = values[indices[1]], values[indices[0]]
    elif kind == 'batch_count':
        next(e for e in rows if e['label'].startswith('batch'))['response']['result']['requests'].pop()
    elif kind == 'numerical_hash':
        row = next(e for e in rows if not e['label'].startswith('batch'))
        row['response']['result']['output_sha256'] = '0' * 64
    elif kind == 'governor':
        rows[0]['response']['result']['system_before']['governor'] = 'powersave'
    else:
        raise ValueError(kind)
    return values


def check_clone(source, kind):
    with tempfile.TemporaryDirectory(prefix='reliability-audit-copy-') as tmp:
        clone = Path(tmp)
        replaced = {EVENTS} if kind in ('order', 'batch_count', 'numerical_hash', 'governor') else {'config.json' if kind == 'config' else 'freeze.json'}
        for path in source.iterdir():
            if path.name not in replaced | {'checksums.json'}:
                (clone / path.name).symlink_to(path, target_is_directory=path.is_dir())
        if EVENTS in replaced:
            rewrite_events(source / EVENTS, clone / EVENTS, kind)
        elif kind == 'config':
            value = read(source / 'config.json')
            value['campaign_max_seconds'] = 999999
            atomic_json(clone / 'config.json', value)
        else:
            value = read(source / 'freeze.json')
            key = next(iter(value['source_sha256']))
            value['source_sha256'][key] = '0' * 64
            atomic_json(clone / 'freeze.json', value)
        result = audit(clone, require_seal=False)
        if result['passed']:
            raise RuntimeError('Audit accepted altered ' + kind)
        return dict(case=kind, rejected=True, error=result.get('error'))


def rewrite_events(source, destination, kind):
    """Change one record without doubling the full multi-hour matrix in memory."""
    changed = False
    pending = []
    with Path(source).open() as input_stream, Path(destination).open('w') as output:
        for line in input_stream:
            event = json.loads(line)
            main = event['event'] == 'measurement' and event['section'] == 'main'
            if kind == 'order' and not changed:
                if pending and main:
                    output.write(line)
                    output.writelines(pending[1:])
                    output.write(pending[0])
                    pending.clear()
                    changed = True
                    continue
                if main or pending:
                    pending.append(line)
                    continue
            if not changed and main:
                result = event['response']['result']
                if kind == 'batch_count' and event['label'].startswith('batch'):
                    result['requests'].pop()
                    changed = True
                elif kind == 'numerical_hash' and not event['label'].startswith('batch'):
                    result['output_sha256'] = '0' * 64
                    changed = True
                elif kind == 'governor':
                    result['system_before']['governor'] = 'powersave'
                    changed = True
                if changed:
                    line = json.dumps(event) + '\n'
            output.write(line)
    if not changed or pending:
        raise ValueError('No suitable main measurement for ' + kind)


def validate(campaign):
    campaign = Path(campaign)
    ledger = read(campaign / 'campaign.json')
    baselines, cases, gates = [], [], []
    for attempt in ledger['attempts']:
        source = Path(attempt['output'])
        summary = read(source / 'summary.json')
        if not summary['complete']:
            raise ValueError('Incomplete attempt requires a checkpoint, not successful campaign finalization')
        baseline = audit(source)
        if not baseline['passed']:
            raise ValueError('Completed run failed independent audit: ' + str(baseline))
        baselines.append(dict(output=str(source), passed=True))
        kinds = ['order', 'numerical_hash', 'governor', 'config', 'source']
        if attempt['stage'].startswith('cpu'):
            kinds.append('batch_count')
        for kind in kinds:
            case = dict(stage=attempt['stage'], **check_clone(source, kind))
            cases.append(case)
            print(json.dumps(case), flush=True)
        verify_artifacts(source)
    if not baselines:
        raise ValueError('No completed scientific baseline for final audit')
    for stage in ('npu-develop', 'npu-confirm', 'combined-develop', 'combined-confirm'):
        try:
            dependency(ledger, stage)
        except ValueError as exc:
            gates.append(dict(stage=stage, rejected=True, reason=str(exc)))
        else:
            raise ValueError('Recovery profile accepted out-of-scope stage')
    for attempt in ledger['attempts']:
        try:
            dependency(ledger, attempt['stage'])
        except ValueError as exc:
            gates.append(dict(stage=attempt['stage'], rejected=True, reason=str(exc)))
        else:
            raise ValueError('Completed scientific stage could be repeated')
    lineage = verify_lineage(campaign)
    # Alter only a copied manifest; originals and their live paths are never edited.
    with tempfile.TemporaryDirectory(prefix='reliability-lineage-copy-') as tmp:
        clone = Path(tmp)
        atomic_json(clone / 'campaign.json', ledger)
        value = read(campaign / 'recovery-lineage.json')
        value['recovery_from'] += '-altered'
        atomic_json(clone / 'recovery-lineage.json', value)
        try:
            verify_lineage(clone)
        except ValueError as exc:
            cases.append(dict(case='lineage', rejected=True, error=str(exc)))
        else:
            raise ValueError('Changed recovery lineage accepted')
    atomic_json(campaign / 'negative-gates.json', dict(utc=utc(), passed=True, checks=gates))
    atomic_json(campaign / 'audit-tamper-tests.json', dict(utc=utc(), passed=True,
                baselines=baselines, cases=cases, lineage=lineage))
    return dict(passed=True, baselines=len(baselines), rejected_cases=len(cases))
