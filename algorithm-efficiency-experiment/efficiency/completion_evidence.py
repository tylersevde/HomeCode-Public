"""Read-only cross-checks connecting raw runtime to cumulative budget receipts."""
import os
from pathlib import Path

from .common import digest_value
from .completion_budget import audit_snapshot, read, require


def require_managed_service(unit):
    require(bool(os.environ.get('INVOCATION_ID')) and unit in Path('/proc/self/cgroup').read_text(),
            'Hardware workflow must run in its managed user service; use start')


def check_receipt(receipt):
    snapshot = receipt['snapshot']
    audit_snapshot(snapshot)
    require(receipt['version'] == snapshot['version'] and receipt['family_id'] == snapshot['family_id']
            and receipt['snapshot_sha256'] == snapshot['sha256'], 'Budget receipt identity differs')
    rows = [r for r in snapshot['reservations'] if r['id'] == receipt['reservation_id']]
    require(len(rows) == 1 and rows[0] == receipt['reservation'], 'Budget receipt reservation differs')
    require(rows[0] == snapshot['reservations'][-1], 'Receipt does not describe the latest reservation')
    return rows[0]


def audit_budget(directory, require_final=False):
    directory = Path(directory)
    config = read(directory / 'config.json')
    initial = read(directory / 'budget-reservation.json')
    row = check_receipt(initial)
    require(row['state'] == 'reserved' and row['charged_seconds'] == row['reserved_seconds'],
            'Launch receipt is not a conservative reservation')
    require(config['budget_reservation_id'] == row['id'] and config['campaign'] == row['campaign']
            and config['reserved_seconds'] == row['reserved_seconds'], 'Run reservation identity differs')
    track = 'cpu' if config['profile'] == 'cpu-compare' else config['track']
    stage = 'compare' if track == 'cpu' else ('develop' if config['stage'] == 'diagnostic' else config['stage'])
    require(row['track'] == track and row['stage'] == stage, 'Run charged to the wrong budget')
    if track != 'cpu':
        expected_purpose = 'diagnostic' if config['stage'] == 'diagnostic' else 'qualification'
        require(row['purpose'] == expected_purpose, 'Diagnostic/qualification charge changed')
        for bucket, key in (('develop', 'development_charged_seconds'), ('confirm', 'confirmation_charged_seconds')):
            charged = sum(r['charged_seconds'] for r in initial['snapshot']['reservations'][:-1]
                          if r['track'] == track and r['stage'] == bucket)
            require(config[key] == charged, 'Prior stage charges differ from sealed budget lineage')
        parents = config.get('parents', {})
        required = []
        if track == 'gpu' and config['stage'] == 'confirm':
            required.append(('gpu-develop', 'develop', 'qualification'))
        if track == 'npu' and config['stage'] != 'diagnostic':
            required.append(('diagnostic', 'develop', 'diagnostic'))
            if config['stage'] == 'confirm':
                required.append(('develop', 'develop', 'qualification'))
        for name, bucket, purpose in required:
            parent = read(Path(parents[name]) / 'config.json')
            ancestors = [r for r in initial['snapshot']['reservations'][:-1]
                         if r['id'] == parent['budget_reservation_id']]
            require(len(ancestors) == 1 and ancestors[0]['track'] == track
                    and ancestors[0]['stage'] == bucket and ancestors[0]['purpose'] == purpose
                    and ancestors[0]['state'] == 'finalized'
                    and ancestors[0]['evidence']['path'] == parent['campaign'],
                    'Scientific parent is not charged in the same cumulative budget family')
    if track == 'cpu':
        lineage = read(directory / 'retry-lineage.json')
        require(lineage['campaign'] == row['parent'], 'CPU retry parent differs from budget lineage')
    final_path = directory / 'budget-finalization.json'
    if not final_path.exists():
        require(not require_final, 'Sealed run is missing its final budget receipt')
        return dict(passed=True, reservation_id=row['id'], finalized=False)
    final = read(final_path)
    end = check_receipt(final)
    require(final['family_id'] == initial['family_id'] and final['ledger_path'] == initial['ledger_path']
            and final['reservation_id'] == initial['reservation_id'], 'Final budget family changed')
    require(final['snapshot']['reservations'][:-1] == initial['snapshot']['reservations'][:-1]
            and final['snapshot']['cpu_origin'] == initial['snapshot']['cpu_origin'], 'Earlier charges changed during run')
    for key in ('id', 'track', 'stage', 'campaign', 'purpose', 'relation', 'parent',
                'remaining_before_seconds', 'reserved_seconds', 'reserved_utc'):
        require(end[key] == row[key], 'Budget reservation changed: ' + key)
    require(end['state'] == 'finalizing', 'Sealed receipt must precede its own evidence seal')
    runtime = read(directory / 'runtime-finalization.json')
    require(end['charged_seconds'] == runtime['charged_seconds']
            == runtime['finished_monotonic'] - runtime['started_monotonic']
            and end['reserved_seconds'] == runtime['reserved_seconds'], 'Raw runtime charge differs from ledger')
    summary = read(directory / 'summary.json')
    validation = read(directory / 'validation.json')
    success = summary.get('complete') and validation.get('passed') and runtime.get('restored')
    if success:
        if track == 'npu' and config['stage'] == 'diagnostic':
            expected = 'diagnostic_complete'
        else:
            expected = 'qualified' if summary.get('accepted') or summary.get('selected') is not None else 'rejected'
        require(end['disposition'] == expected, 'Budget disposition contradicts audited scientific result')
    else:
        require(end['disposition'] not in ('qualified', 'rejected', 'diagnostic_complete'),
                'Incomplete or unaudited evidence cannot close as scientifically qualified or rejected')
    return dict(passed=True, reservation_id=row['id'], finalized=True,
                charged_seconds=end['charged_seconds'], disposition=end['disposition'])
