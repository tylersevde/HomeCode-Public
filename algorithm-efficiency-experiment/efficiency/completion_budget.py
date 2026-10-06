"""Locked cumulative roadmap allowances; reservations survive crashes at full cost.

No function launches hardware or retries. A successful NPU diagnostic may be
followed by one explicitly linked qualification-development session in the same
bucket. Other scientific outcomes close their bucket. Call finish to prepare a
receipt, seal the campaign containing that receipt, then attach_evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import uuid

from .common import atomic_json, digest_file, digest_value, read_jsonl, utc

VERSION = 'completion-budget-v1'
CAPS = {'cpu': {'compare': 14400}, **{
    track: {'develop': 7200, 'confirm': 7200} for track in ('gpu', 'npu', 'cache', 'combined')}}
DISPOSITIONS = {'qualified', 'rejected', 'inconclusive', 'infrastructure_failed',
                'dependency_failed', 'diagnostic_complete'}
ACTIVE = {'reserved', 'running', 'finalizing'}
CLEANUP_SECONDS = 120


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@contextmanager
def locked(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield path


def sealed_reference(directory):
    from .hybrid import verify_artifacts
    directory = Path(directory).resolve(strict=True)
    checks = verify_artifacts(directory)
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
              if p.is_file() and p != directory / 'checksums.json'}
    require(set(checks) == actual, 'Evidence seal coverage differs')
    return dict(path=str(directory), checksums_sha256=digest_file(directory / 'checksums.json'))


def verify_reference(reference):
    require(sealed_reference(reference['path']) == reference, 'Budget ancestor seal changed')


def validate_retry_source(campaign):
    """Read a sealed CPU infrastructure failure, refusing scientific/feasibility retries."""
    campaign = Path(campaign).resolve()
    reference = sealed_reference(campaign)
    ledger = read(campaign / 'campaign.json')
    require(ledger.get('version') == 'cpu-compare-v1' and ledger.get('attempts'),
            'Retry source must be a sealed CPU comparison attempt')
    require(number(ledger.get('max_seconds')) and 120 < ledger['max_seconds'] <= CAPS['cpu']['compare'],
            'Original CPU allowance differs')
    total = []
    outputs = []
    for attempt in ledger['attempts']:
        require(attempt.get('stage') == 'cpu-compare' and not attempt.get('scientific_complete')
                and attempt.get('restored') is True, 'Retry source is scientific or not restored')
        require(number(attempt.get('charged_seconds')) and 0 <= attempt['charged_seconds'] <= attempt['reserved_seconds'],
                'Retry source charge is invalid')
        output = Path(attempt['output']).resolve()
        source = sealed_reference(output)
        require(source['checksums_sha256'] == attempt['checksums_sha256'], 'Retry run seal changed')
        result = read(output / 'summary.json')
        require(not result.get('complete') and not result.get('accepted')
                and result.get('decision') not in ('deferred_by_pilot', 'no_qualified_candidate'),
                'Scientific or feasibility outcomes cannot be retried')
        events, damaged = read_jsonl(output / 'cpu-compare.jsonl')
        require(not damaged and not any(e.get('event') == 'measurement_complete' or
                (e.get('event') == 'pilot_gate' and not e.get('fits')) for e in events),
                'Completed or infeasible matrix cannot be retried')
        restoration = read(output / 'governor-restoration.json')
        require(restoration.get('restored') is True and restoration['current'] == ledger['original_governor'],
                'Retry governor restoration is unverified')
        finalization = read(output / 'runtime-finalization.json')
        require(finalization['charged_seconds'] == attempt['charged_seconds']
                and finalization['reserved_seconds'] == attempt['reserved_seconds']
                and finalization.get('restored') is True, 'Retry elapsed-time receipt differs')
        disposition = ledger.get('budget_disposition', 'infrastructure_failed')
        require(disposition == 'infrastructure_failed', 'Retry source is not an infrastructure failure')
        total.append(attempt['charged_seconds']); outputs.append(source)
    return dict(campaign=reference, outputs=outputs, charged_seconds=math.fsum(total),
                source_campaign=ledger['source_campaign'], original_governor=ledger['original_governor'],
                budget_reservation=ledger.get('budget_reservation'),
                attempt_count=len(ledger['attempts']))


def _body(value):
    return {key: val for key, val in value.items() if key != 'sha256'}


def _stamp(value):
    value['sha256'] = digest_value(_body(value))
    return value


def _usage(row):
    # Incomplete finalization is still a reservation, even if a provisional
    # elapsed-time field was written before a crash.
    return row['reserved_seconds'] if row['state'] in ACTIVE else row['charged_seconds']


def _rows(value, track, stage):
    return [r for r in value['reservations'] if (r['track'], r['stage']) == (track, stage)]


def remaining(value, track, stage):
    require(track in CAPS and stage in CAPS[track], 'Unknown roadmap budget bucket')
    inherited = value['cpu_origin']['charged_seconds'] if track == 'cpu' and value.get('cpu_origin') else 0
    return CAPS[track][stage] - math.fsum([inherited, *(_usage(r) for r in _rows(value, track, stage))])


def audit_snapshot(value, *, verify_seals=True):
    """Validate immutable limits and reconstruct the complete allowance history."""
    require(value.get('version') == VERSION and value.get('caps') == CAPS, 'Roadmap budget limits changed')
    require(value.get('sha256') == digest_value(_body(value)), 'Budget snapshot digest differs')
    origin = value.get('cpu_origin')
    if origin:
        if verify_seals:
            require(validate_retry_source(origin['campaign']['path']) == origin, 'Imported CPU charge or lineage changed')
        require(number(origin['charged_seconds']) and 0 <= origin['charged_seconds'] <= CAPS['cpu']['compare'],
                'Imported CPU charge invalid')
    seen = set(); campaigns = set(); active = []
    previous = {track: {} for track in CAPS}
    consumed = {(track, stage): [origin['charged_seconds'] if track == 'cpu' and origin else 0]
                for track, stages in CAPS.items() for stage in stages}
    for index, row in enumerate(value['reservations']):
        track, stage = row['track'], row['stage']
        require(track in CAPS and stage in CAPS[track], 'Unknown roadmap budget bucket')
        require(row['id'] not in seen and row['campaign'] not in campaigns, 'Budget attempt duplicated')
        require(Path(row['campaign']).is_absolute(), 'Budget campaign must be absolute')
        seen.add(row['id']); campaigns.add(row['campaign'])
        require(row['state'] in ACTIVE | {'finalized'}, 'Unknown reservation state')
        require(number(row['reserved_seconds']) and CLEANUP_SECONDS < row['reserved_seconds'] <= CAPS[track][stage],
                'Invalid reservation size')
        require(number(row['charged_seconds']) and 0 <= row['charged_seconds'] <= row['reserved_seconds'],
                'Invalid reservation charge')
        available = CAPS[track][stage] - math.fsum(consumed[track, stage])
        require(row['remaining_before_seconds'] == available and row['reserved_seconds'] <= available,
                'Cumulative reservation exceeds or changes remaining allowance')
        predecessor = previous[track].get(stage)
        if predecessor:
            require(predecessor['state'] == 'finalized' and predecessor.get('evidence'), 'Prior attempt is unresolved')
            expected = 'continuation' if predecessor['disposition'] == 'diagnostic_complete' else 'retry'
            require(predecessor['disposition'] in ('infrastructure_failed', 'diagnostic_complete')
                    and row['relation'] == expected and row['parent'] == predecessor['evidence'],
                    'Scientific retry or changed attempt chain')
            if expected == 'retry':
                require(row['purpose'] == predecessor['purpose'], 'Infrastructure retry changed its research purpose')
            if expected == 'continuation':
                require((track, stage) == ('npu', 'develop') and row['purpose'] == 'qualification',
                        'Only NPU diagnostic-to-qualification continuation is allowed')
        elif track == 'cpu':
            require(origin is not None and row['relation'] == 'retry' and row['parent'] == origin['campaign'],
                    'CPU allowance cannot be reset without its sealed original attempt')
        else:
            require(row['relation'] == 'initial' and row['parent'] is None, 'Unexpected initial attempt parent')
        require(row['purpose'] in ('qualification', 'diagnostic') and
                (row['purpose'] != 'diagnostic' or (track, stage) == ('npu', 'develop')),
                'Invalid diagnostic purpose')
        if stage == 'confirm':
            development = previous[track].get('develop')
            require(development is not None and development['state'] == 'finalized'
                    and development['disposition'] == 'qualified' and development['purpose'] == 'qualification',
                    'Confirmation requires qualified development')
        if row['state'] in ('finalizing', 'finalized'):
            require(row.get('disposition') in DISPOSITIONS, 'Unknown final disposition')
            require((row['disposition'] == 'diagnostic_complete') <= (row['purpose'] == 'diagnostic'),
                    'Qualification cannot become diagnostic')
            require(row['purpose'] != 'diagnostic' or row['disposition'] != 'qualified',
                    'Diagnostic work cannot qualify the task')
        else:
            require(row['charged_seconds'] == row['reserved_seconds'], 'Unfinished work must retain full reservation')
        if row['state'] in ACTIVE:
            active.append(row['id'])
        else:
            require(row.get('evidence') is not None, 'Finalized reservation lacks a sealed result')
            if verify_seals:
                verify_reference(row['evidence'])
                receipt = read(Path(row['evidence']['path']) / 'budget-finalization.json')
                recorded = dict(row, state='finalizing', evidence=None)
                require(receipt['reservation_id'] == row['id'] and receipt['reservation'] == recorded,
                        'Sealed charge, reservation or disposition differs')
                ancestor = receipt['snapshot']
                require(receipt['family_id'] == value['family_id'] == ancestor['family_id']
                        and receipt['snapshot_sha256'] == ancestor['sha256'] == digest_value(_body(ancestor))
                        and ancestor['cpu_origin'] == origin and ancestor['caps'] == CAPS
                        and ancestor['reservations'] == value['reservations'][:index] + [recorded],
                        'Sealed budget ancestor chain differs')
        consumed[track, stage].append(_usage(row))
        require(math.fsum(consumed[track, stage]) <= CAPS[track][stage], 'Budget overrun')
        previous[track][stage] = row
    require(len(active) <= 1, 'Concurrent roadmap reservations are forbidden')
    return dict(passed=True, reservations=len(seen), active=active,
                remaining_seconds={track: {stage: remaining(value, track, stage) for stage in stages}
                                   for track, stages in CAPS.items()})


def _load(path):
    value = read(path)
    audit_snapshot(value)
    identity = read(path.with_name(path.name + '.identity.json'))
    require(identity == dict(version=VERSION, family_id=value['family_id'], caps=CAPS,
                            cpu_origin_sha256=digest_value(value.get('cpu_origin'))),
            'Canonical budget family identity changed')
    journal = _journal(path)
    require(journal and journal[-1]['snapshot_sha256'] == value['sha256'],
            'Budget ledger does not match its committed journal; recovery is required')
    return value


def _journal(path):
    entries, damaged = read_jsonl(path.with_name(path.name + '.journal.jsonl'))
    require(not damaged, 'Budget journal is damaged; do not reset the allowance')
    previous = None
    for index, entry in enumerate(entries):
        require(entry['sequence'] == index and entry['previous_sha256'] == previous,
                'Budget journal chain changed')
        require(entry['record_sha256'] == digest_value({k:v for k,v in entry.items() if k != 'record_sha256'})
                and entry['snapshot_sha256'] == entry['snapshot']['sha256']
                == digest_value(_body(entry['snapshot'])), 'Budget journal snapshot changed')
        previous = entry['record_sha256']
    return entries


def _save(path, value):
    _stamp(value)
    audit_snapshot(value, verify_seals=False)
    journal = _journal(path)
    record = dict(sequence=len(journal), previous_sha256=journal[-1]['record_sha256'] if journal else None,
                  snapshot_sha256=value['sha256'], snapshot=value)
    record['record_sha256'] = digest_value(record)
    # Commit the full new state before the convenience JSON. An interrupted
    # write causes a conservative refusal; it never silently refunds time.
    with path.with_name(path.name + '.journal.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    atomic_json(path, value)


def initialize(path, cpu_origin=None):
    """Create one canonical roadmap ledger; CPU needs an immutable imported origin."""
    with locked(path) as path:
        origin = validate_retry_source(cpu_origin) if cpu_origin is not None else None
        if path.exists():
            value = _load(path)
            if origin is not None:
                require(value['cpu_origin'] == origin, 'Roadmap origin cannot be replaced or reset')
            return deepcopy(value)
        require(not path.with_name(path.name + '.identity.json').exists()
                and not path.with_name(path.name + '.journal.jsonl').exists(),
                'Budget ledger is missing but its family already exists; refusing an allowance reset')
        value = dict(version=VERSION, family_id=str(uuid.uuid4()), created_utc=utc(),
                     caps=deepcopy(CAPS), cpu_origin=origin, reservations=[])
        atomic_json(path.with_name(path.name + '.identity.json'), dict(version=VERSION,
                    family_id=value['family_id'], caps=CAPS, cpu_origin_sha256=digest_value(origin)))
        _save(path, value)
        return deepcopy(value)


def snapshot(path):
    with locked(path) as path:
        return deepcopy(_load(path))


def _find(value, identifier):
    rows = [r for r in value['reservations'] if r['id'] == identifier]
    require(len(rows) == 1, 'Unknown budget reservation')
    return rows[0]


def _receipt(path, value, row):
    return dict(version=VERSION, ledger_path=str(path), family_id=value['family_id'],
                reservation_id=row['id'], reservation=deepcopy(row), snapshot=deepcopy(value),
                snapshot_sha256=value['sha256'])


def reserve(path, track, stage, campaign, requested=None, retry_from=None,
            continuation_from=None, purpose='qualification'):
    """Reserve explicitly; no automatic retries and no movement between stage budgets."""
    campaign = str(Path(campaign).resolve())
    with locked(path) as path:
        value = _load(path)
        require(not any(r['state'] in ACTIVE for r in value['reservations']), 'A roadmap reservation is already active')
        require(not any(r['campaign'] == campaign for r in value['reservations']), 'Campaign already reserved')
        require(not (retry_from and continuation_from), 'Retry and planned continuation are distinct')
        rows = _rows(value, track, stage)
        predecessor = rows[-1] if rows else None
        relation, parent = 'initial', None
        if predecessor:
            desired = predecessor['disposition']
            require(desired in ('infrastructure_failed', 'diagnostic_complete'), 'Scientific stage is closed')
            supplied = continuation_from if desired == 'diagnostic_complete' else retry_from
            require(supplied is not None, 'Explicit predecessor is required; automatic retries are disabled')
            parent = sealed_reference(supplied)
            require(parent == predecessor['evidence'], 'Retry must reference the latest sealed attempt')
            if track == 'cpu':
                validate_retry_source(supplied)
            relation = 'continuation' if desired == 'diagnostic_complete' else 'retry'
        elif track == 'cpu':
            require(value['cpu_origin'] is not None and retry_from is not None,
                    'CPU retry must reference its original sealed allowance')
            parent = validate_retry_source(retry_from)['campaign']
            require(parent == value['cpu_origin']['campaign'], 'CPU retry origin differs')
            relation = 'retry'
        else:
            require(retry_from is None and continuation_from is None, 'No previous attempt exists')
        available = remaining(value, track, stage)
        requested = available if requested is None else requested
        require(number(requested) and CLEANUP_SECONDS < requested <= available,
                f'Request exceeds remaining {track}/{stage} allowance ({available!r} seconds)')
        row = dict(id=str(uuid.uuid4()), track=track, stage=stage, campaign=campaign,
                   purpose=purpose, relation=relation, parent=parent, reserved_utc=utc(),
                   remaining_before_seconds=available, reserved_seconds=requested,
                   charged_seconds=requested, state='reserved', disposition=None, evidence=None)
        value['reservations'].append(row)
        _save(path, value)
        return _receipt(path, value, row)


def mark_started(path, reservation_id):
    with locked(path) as path:
        value = _load(path); row = _find(value, reservation_id)
        require(row['state'] == 'reserved', 'Reservation already started or finalized')
        row.update(state='running', started_utc=utc())
        _save(path, value)
        return _receipt(path, value, row)


def finish(path, reservation_id, charged_seconds, disposition, evidence=None):
    """Prepare the final charge; it remains reserved until its receipt is sealed."""
    with locked(path) as path:
        value = _load(path); row = _find(value, reservation_id)
        require(row['state'] in ('reserved', 'running'), 'Reservation already finished')
        require(number(charged_seconds) and 0 <= charged_seconds <= row['reserved_seconds'], 'Invalid final charge')
        require(disposition in DISPOSITIONS, 'Unknown disposition')
        row.update(state='finalizing', charged_seconds=charged_seconds, disposition=disposition, finished_utc=utc())
        _save(path, value)
        receipt = _receipt(path, value, row)
    if evidence is not None:
        return attach_evidence(path, reservation_id, evidence)
    return receipt


def abandon(path, reservation_id, disposition='infrastructure_failed'):
    """Conservatively finalize interrupted work without releasing any reserved time."""
    require(disposition in ('infrastructure_failed', 'inconclusive'), 'Invalid interrupted disposition')
    with locked(path) as path:
        value = _load(path); row = _find(value, reservation_id)
        require(row['state'] in ACTIVE, 'Reservation is already finalized')
        row.update(state='finalizing', charged_seconds=row['reserved_seconds'], disposition=disposition,
                   finished_utc=utc(), recovery='Full reservation charged because complete sealed finalization was unavailable')
        _save(path, value)
        return _receipt(path, value, row)


def attach_evidence(path, reservation_id, campaign):
    """Release a reservation only after its exact finalization receipt is sealed."""
    with locked(path) as path:
        value = _load(path); row = _find(value, reservation_id)
        require(row['state'] == 'finalizing', 'Finalization receipt is required before sealing')
        reference = sealed_reference(campaign)
        require(reference['path'] == row['campaign'], 'Final evidence belongs to another campaign')
        receipt = read(Path(campaign) / 'budget-finalization.json')
        require(receipt == _receipt(path, value, row), 'Final evidence receipt differs from reserved ledger')
        row.update(state='finalized', evidence=reference)
        _save(path, value)
        return _receipt(path, value, row)
