"""Read-only reconstruction of the CPU comparison, independent of its analyzer."""
import json
import math
from pathlib import Path

import numpy as np

from . import cpu_compare_spec as spec
from .attention_native import fixture, oracle, errors
from .common import digest_file, read_jsonl
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .monitor import safety_reason
from .refine_audit import require
from .reliability_audit import check_interval

NUMERICAL_LIBRARY_THREADS = dict.fromkeys(('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                          'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'), '1')


def read(path):
    return json.loads(Path(path).read_text())


def local_path(directory, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), 'Evidence path escapes run directory')
    return path


def finite(value, minimum=0):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


def decisions(rows, config, summary):
    """Recalculate each endpoint from whole paired blocks; never call analyze()."""
    expected = [j for j in spec.schedule(config, spec.fixtures(config)) if j['section'] == 'main']
    require(len(rows) == len(expected), 'Incomplete primary endpoint matrix')
    for row, job in zip(rows, expected):
        require(all(row[k] == v for k, v in job.items()), 'Primary endpoint order differs')
        require(finite(row['total_ms']) and row['total_ms'] > 0, 'Invalid primary endpoint')

    cells = [c['cell_id'] for c in spec.CELLS]
    grouped = {}
    for row in rows:
        key = row['configuration'], row['label'], row['block'], row['cell_id']
        grouped.setdefault(key, []).append(row['total_ms'])

    def vector(configuration, labels, cell=None):
        selected = cells if cell is None else [cell]
        return [float(np.mean([np.mean([np.mean(grouped[configuration, label, block, shape])
                                      for label in labels]) for shape in selected]))
                for block in range(spec.BLOCKS)]

    require([m['kind'] for m in summary['metrics']] == ['single', 'batch'], 'Endpoint coverage differs')
    qualified = []
    for metric in summary['metrics']:
        kind = metric['kind']; labels = [kind + '_a', kind + '_b']
        require(metric['label'] == kind, 'Endpoint label differs')
        stable = True
        require(set(metric['equivalence']) == set(spec.CONFIGURATIONS), 'Control coverage differs')
        for configuration in spec.CONFIGURATIONS:
            equivalence = metric['equivalence'][configuration]
            require(set(equivalence) == {'aggregate', *cells}, 'Shape control coverage differs')
            for cell in [None, *cells]:
                _, interval = check_interval(equivalence[cell or 'aggregate'],
                    vector(configuration, [labels[1]], cell), vector(configuration, [labels[0]], cell))
                stable &= .95 <= interval[0] <= interval[1] <= 1.05
        means = {configuration: float(np.mean(vector(configuration, labels)))
                 for configuration in spec.CONFIGURATIONS}
        require(metric['mean_ms'] == means, 'Endpoint means differ')
        speed, interval = check_interval(metric['speedup'], vector('baseline', labels), vector('candidate', labels))
        slowdowns = []
        require(set(metric['per_cell_speedup']) == set(cells), 'Speedup shape coverage differs')
        for cell in cells:
            baseline, candidate = vector('baseline', labels, cell), vector('candidate', labels, cell)
            check_interval(metric['per_cell_speedup'][cell], baseline, candidate)
            slowdowns.append(float(np.mean(candidate) / np.mean(baseline)))
        correct = True
        for row in rows:
            if row['label'] not in labels:
                continue
            result = row['response']['result']; requests = result.get('requests', [result])
            correct &= bool(result['correct'] and all(r['correct'] and r['matches_warmup']
                and not r['validation_errors'] and not r['unexpected_output_io'] for r in requests))
        eligible = bool(stable and correct and speed >= 1.10 and interval[0] > 1
                        and max(slowdowns) <= 1.05)
        require(metric['stable'] == stable and metric['correct'] == correct
                and metric['max_shape_slowdown'] == max(slowdowns)
                and metric['qualified'] == eligible, 'Endpoint qualification differs')
        qualified.append(eligible)
    accepted = all(qualified)
    require(summary['complete'] and summary['accepted'] == accepted
            and summary['version'] == spec.VERSION
            and summary['decision'] == ('cpu_improved' if accepted else 'no_qualified_candidate')
            and summary['defaults_changed'] is False, 'Final comparison eligibility differs')


def check_provenance(directory, config, manifest, freeze):
    provenance = read(directory / 'provenance.json')
    require(freeze['provenance_sha256'] == digest_file(directory / 'provenance.json'), 'Provenance freeze differs')
    require(provenance['selected'] == 5, 'Historical CPU selection differs')
    for key in ('source_campaign', 'cpu_confirmation'):
        entry = provenance[key]; source = Path(entry['path']); checks = verify_artifacts(source)
        require(checks and digest_file(source / 'checksums.json') == entry['checksums_sha256'], 'Historical evidence changed')
    source = Path(provenance['source_campaign']['path']); parent = Path(provenance['cpu_confirmation']['path'])
    require(Path(config['source_campaign']).resolve() == source.resolve(), 'Source campaign substituted')
    validation = read(source / 'final-validation.json')
    require(validation.get('passed', validation.get('integrity_passed', False)), 'Source campaign audit failed')
    candidates = [a for a in read(source / 'campaign.json')['attempts']
                  if a['stage'] == 'cpu-confirm' and Path(a['output']).resolve() == parent.resolve()]
    require(len(candidates) == 1 and candidates[0]['audit_passed'] and candidates[0]['scientific_complete']
            and candidates[0]['checksums_sha256'] == provenance['cpu_confirmation']['checksums_sha256'],
            'Selected confirmation is not qualified in source campaign')
    prior = read(parent / 'summary.json')
    require(read(parent / 'config.json')['reliability_stage'] == 'cpu-confirm'
            and read(parent / 'validation.json')['passed'] and prior['complete']
            and prior['accepted'] and prior['selected'] == 5, 'Historical CPU confirmation did not qualify')
    build = read(directory / 'native-build/build.json')
    require(build == read(parent / 'native-build/build.json'), 'Native calculations changed since confirmation')
    for relative, digest in build['artifacts'].items():
        require(digest_file(local_path(directory, 'native-build/' + relative)) == digest, 'Native binary changed')
    for relative, digest in build['sources'].items():
        require(manifest['source_sha256'][relative] == digest, 'Native source changed')
    wrapper = 'efficiency/attention_native.py'
    require(manifest['source_sha256'][wrapper] == read(parent / 'manifest.json')['source_sha256'][wrapper],
            'Numerical wrapper changed since confirmation')
    runtime = read(directory / 'cpu-runtime.json')
    require(runtime['version'] == 'cpu-policy-v1' and runtime['model'].startswith('Raspberry Pi 5')
            and runtime['machine'] == 'aarch64' and runtime['cpu_affinity'] == [0, 1, 2, 3]
            and runtime['model'] == manifest['cpu_model'] and runtime['python'] == manifest['python']
            and runtime['numpy'] == manifest['versions']['numpy']
            and runtime['numerical_library_threads'] == NUMERICAL_LIBRARY_THREADS, 'Recorded CPU runtime differs')
    require(set(runtime['native_artifacts']) == {'libattention.so', 'libaffinity.so', 'build.json'},
            'CPU runtime native artifact coverage differs')
    for name, digest in runtime['native_artifacts'].items():
        require(digest_file(directory / 'native-build' / name) == digest, 'CPU runtime native artifact differs')
    require(set(runtime['libraries']) == {'libgomp.so.1', 'libc.so.6'}, 'CPU runtime library coverage differs')
    for library in runtime['libraries'].values():
        require(Path(library['path']).is_absolute() and len(library['sha256']) == 64
                and all(c in '0123456789abcdef' for c in library['sha256']), 'CPU runtime library identity invalid')


def audit(directory, require_seal=True):
    """Return audit evidence without creating or changing any artifact."""
    directory = Path(directory); checks = []
    try:
        if require_seal:
            sealed = verify_artifacts(directory)
            actual = {str(p.relative_to(directory)) for p in directory.rglob('*')
                      if p.is_file() and p != directory / 'checksums.json'}
            require(set(sealed) == actual, 'Artifact seal coverage differs')
            checks.append('complete artifact seal')
        config = read(directory / 'config.json')
        require(config['profile'] == 'cpu-compare' and config['phase'] == 'cpu', 'CPU comparison scope differs')
        require(read(directory / 'protocol.json') == json.loads(json.dumps(spec.specification())), 'Protocol changed')
        require(config['baseline_seconds'] == 15 and config['cooldown_seconds'] == 15
                and config['cleanup_reserve_seconds'] == 120, 'Frozen time reserves differ')
        manifest = read(directory / 'manifest.json'); freeze = read(directory / 'freeze.json')
        if ('efficiency/completion_budget.py' in manifest['source_sha256']
                or 'budget_reservation_id' in config or (directory / 'budget-reservation.json').exists()):
            from .completion_evidence import audit_budget
            audit_budget(directory, require_final=require_seal)
            checks.append('sealed cumulative retry budget and original attempt lineage')
        require(manifest['source_sha256'] and freeze['source_sha256'] == manifest['source_sha256'], 'Source identity differs')
        if 'config' in manifest:
            require(manifest['config'] == config, 'Manifest configuration differs')
        for relative, digest in manifest['source_sha256'].items():
            require(digest_file(local_path(directory, 'source/' + relative)) == digest, 'Archived source changed')
        require(freeze['config_sha256'] == digest_file(directory / 'config.json')
                and freeze['protocol_sha256'] == digest_file(directory / 'protocol.json'), 'Configuration freeze differs')
        check_provenance(directory, config, manifest, freeze)
        checks.append('independent new source freeze and sealed historical selection')

        fs = read(directory / 'fixtures.json'); planned = spec.fixtures(config)
        prior = read(directory / 'prior-fixtures.json'); seeds = set(prior['seeds']); hashes = set(prior['hashes'])
        require(len(fs) == len(planned), 'Fixture count differs')
        for saved, expected_spec in zip(fs, planned):
            require(all(saved[k] == v for k, v in expected_spec.items()), 'Fixture specification differs')
            require(saved['seed'] not in seeds and saved['input_sha256'] not in hashes, 'Fixture input or seed reused')
            seeds.add(saved['seed']); hashes.add(saved['input_sha256'])
            path = local_path(directory, 'fixtures/' + saved['fixture_id'] + '.npz')
            require(digest_file(path) == saved['file_sha256'], 'Fixture file changed')
            x, w = fixture(saved['n'], saved['d'], saved['b'], saved['seed']); expected = oracle(x, w)
            with np.load(path, allow_pickle=False) as data:
                require(np.array_equal(data['x'], x) and np.array_equal(data['w'], w)
                        and np.array_equal(data['expected'], expected), 'FP64 fixture reconstruction failed')
            require((input_hash(x), input_hash(w), input_hash(expected)) ==
                    (saved['input_sha256'], saved['weights_sha256'], saved['oracle_sha256']), 'Fixture digest differs')
        byid = {f['fixture_id']: f for f in fs}; jobs = spec.schedule(config, fs)
        require(read(directory / 'schedule.json') == jobs, 'Schedule changed')
        events, damaged = read_jsonl(directory / spec.EVENTS)
        require(not damaged and events, 'Raw events missing or damaged')
        freezes = [e for e in events if e['event'] == 'fixtures_frozen']
        require(len(freezes) == 1 and freezes[0]['fixtures_sha256'] == digest_file(directory / 'fixtures.json')
                and freezes[0]['schedule_sha256'] == digest_file(directory / 'schedule.json'), 'Fixture/schedule freeze differs')
        observations = [e for e in events if e['event'] == 'measurement']
        require(len(observations) == len(jobs), 'Incomplete measurement matrix')
        for event, job in zip(observations, jobs):
            require(all(event[k] == v for k, v in job.items()), 'Measurement order/substitution differs')
            require(finite(event['total_ms']) and event['total_ms'] > 0
                    and event['total_ms'] == (event['ended'] - event['started']) * 1000, 'Request timing changed')

        active = {}; instances = set(); owners = set(); warm = {}; verified = {}; transitions = []
        current_transition = None; completed = False; gate_seen = False; last_measurement_end = 0; measured = 0

        def numeric(response, fid, name, label, batch=False, is_warm=False):
            require(name in active and response['owner'] == active[name]['owner'], 'Numerical owner differs')
            env = active[name]['environment']; result = response['result']; policy = env['policy']
            require(result['policy'] == policy and finite(result['observer_ms']), 'Numerical policy/observer differs')
            require(all(result[k]['governor'] == policy['governor'] for k in ('system_before', 'system_after')),
                    'Governor changed during work')
            rows = result['requests'] if batch else [result]
            if batch:
                require([r['request_id'] for r in rows] == list(range(spec.BATCH_CALLS)), 'Batch request IDs/count changed')
                require(finite(result['worker_batch_ms']), 'Batch timing invalid')
            for row in rows:
                expected_arm = spec.arm(byid[fid], label)
                require(row['fixture_id'] == fid and all(row[k] == v for k, v in expected_arm.items()), 'Numerical arm changed')
                key = fid, row['output_file']
                if key not in verified:
                    with np.load(directory / 'fixtures' / (fid + '.npz'), allow_pickle=False) as data:
                        expected = data['expected']
                    out = np.load(local_path(directory, row['output_file']), allow_pickle=False)
                    require(out.shape == expected.shape and out.dtype == np.float32, 'Numerical output shape/type differs')
                    verified[key] = errors(out, expected), input_hash(out)
                error, digest = verified[key]
                require(digest == row['output_sha256'] and all(row[k] == v for k, v in error.items())
                        and row['correct'] and not row['validation_errors'], 'Numerical output/verification differs')
                require(all(finite(row[k]) for k in ('started', 'computed', 'ended', 'worker_request_ms', 'request_ms', 'cpu_seconds'))
                        and response['received'] <= row['started'] <= row['computed'] <= row['ended'] <= response['ended']
                        and row['worker_request_ms'] == (row['ended'] - row['started']) * 1000,
                        'Native timing invalid')
                require(not row['unexpected_output_io'] and row['matches_warmup'], 'Warmup/output identity differs')
                warm_key = name, fid, label
                if is_warm:
                    require(warm_key not in warm, 'Duplicate warmup')
                    warm[warm_key] = digest
                else:
                    require(warm.get(warm_key) == digest, 'Warmup/output identity differs')
            require(result['correct'], 'Incorrect numerical result')

        for event in events:
            kind = event['event']
            require(kind in {'fixtures_frozen', 'governor_transition', 'worker_ready', 'worker_release',
                             'warmup', 'measurement', 'pilot_gate', 'measurement_complete', 'failure'},
                    'Unexpected device/event in isolated CPU comparison')
            if kind == 'governor_transition':
                require(not active and not completed, 'Governor switched while worker active or after completion')
                current_transition = (event['section'], event['block'], event['configuration'])
                transitions.append(current_transition)
                require(event['current'] == spec.policy(event['configuration'])['governor']
                        and event['original'] == config['original_governor'], 'Governor transition differs')
            elif kind == 'worker_ready':
                name = event['instance']; env = event['environment']; configuration = event['configuration']
                require(name not in instances and not active and current_transition is not None
                        and configuration == current_transition[2], 'Worker lifecycle/order differs')
                instances.add(name); active[name] = event
                owner_key = event['owner']['pid'], event['owner']['tid']
                require(owner_key not in owners, 'Fresh process owner reused'); owners.add(owner_key)
                require(event['device_kind'] == 'cpu' and env['numeric_device'] == 'cpu' and env['device'] == 'CPU'
                        and env['gpu_initialization_ms'] is None and finite(env['cpu_initialization_ms'])
                        and not any(k.startswith(('npu_', 'hailort_', 'model_')) for k in env), 'Non-CPU initialization')
                policy = spec.policy(configuration)
                require(env['configuration'] == configuration and env['policy'] == policy
                        and env['observed']['governor'] == policy['governor'], 'Policy environment differs')
                expected_env = spec.specification()['openmp'][configuration]
                require(env['openmp_actual'] == expected_env
                        and {k: v for k, v in env['openmp_environment'].items() if v is not None} == expected_env,
                        'OpenMP environment contamination')
                require(env['numerical_library_threads'] == NUMERICAL_LIBRARY_THREADS,
                        'Numerical library thread environment differs')
                require([p['requested'] for p in env['probes']] == [1, 4], 'Probe matrix differs')
                for probe in env['probes']:
                    count = probe['requested']; expected_masks = [1 << i for i in range(count)] if configuration == 'candidate' else [15] * count
                    require(probe['team'] == count and [r['mask'] for r in probe['threads']] == expected_masks,
                            'Actual OpenMP team/affinity differs')
            elif kind == 'worker_release':
                name = event['instance']
                require(name in active and event['owner'] == active[name]['owner']
                        and not event['forced'] and not event['alive'], 'Unclean worker release')
                del active[name]
            elif kind in ('warmup', 'measurement'):
                require(not completed and event['instance'] in active, 'Work after completion or without owner')
                require(event['configuration'] == active[event['instance']]['configuration'], 'Scheduled configuration differs')
                if kind == 'measurement':
                    measured += 1
                    require(current_transition == (event['section'], event['block'], event['configuration']), 'Measurement governor order differs')
                    require(event['section'] != 'main' or gate_seen, 'Main measurement before pilot gate')
                    response = event['response']
                    require(last_measurement_end <= event['started'] <= response['submitted'] <= response['received']
                            <= response['ended'] <= response['delivered'] <= event['ended'], 'Clock/transport ordering differs')
                    last_measurement_end = event['ended']
                numeric(event['response'], event['fixture_id'], event['instance'], event['label'],
                        batch=kind == 'measurement' and event['label'].startswith('batch'), is_warm=kind == 'warmup')
            elif kind == 'pilot_gate':
                require(not active and not gate_seen and not completed, 'Pilot gate position differs'); gate_seen = True
                require(measured == sum(j['section'] == 'pilot' for j in jobs), 'Pilot gate precedes complete pilot')
                require(event['multiplier'] == spec.BLOCKS and finite(event['pilot_seconds']) and event['pilot_seconds'] > 0
                        and event['pilot_seconds'] >= sum(e['total_ms'] for e in observations if e['section'] == 'pilot') / 1000
                        and event['required_seconds'] == event['pilot_seconds'] * spec.BLOCKS * 1.2
                        and event['fits'] and event['required_seconds'] <= event['remaining_work_seconds'], 'Pilot projection differs')
            elif kind == 'measurement_complete':
                require(not active and not completed and gate_seen and measured == len(jobs), 'Completion lifecycle differs'); completed = True
            elif kind == 'failure':
                raise ValueError('Recorded worker failure')
        require(completed and not active, 'Incomplete scientific stage or unreleased owner')
        require(transitions == list(dict.fromkeys((j['section'], j['block'], j['configuration']) for j in jobs)),
                'Configuration transition coverage/order differs')
        checks.append('fresh FP64 fixtures, complete ordering, CPU ownership, every output and warmup identity')
        summary = read(directory / 'summary.json')
        decisions([e for e in observations if e['section'] == 'main'], config, summary)
        checks.append('independent endpoints, paired bootstrap, duplicate controls and qualification')

        outcome = read(directory / 'outcome.json'); runtime = read(directory / 'runtime-finalization.json')
        require(outcome['status'] == 'complete' and not outcome.get('stop_reason')
                and finite(outcome['elapsed_seconds']) and outcome['elapsed_seconds'] <= config['max_seconds']
                <= config['reserved_seconds'] <= spec.MAX_SECONDS, 'Supervised budget/outcome differs')
        require(runtime['reserved_seconds'] == config['reserved_seconds'] and finite(runtime['charged_seconds'])
                and runtime['charged_seconds'] == runtime['finished_monotonic'] - runtime['started_monotonic']
                and outcome['elapsed_seconds'] <= runtime['charged_seconds'] <= config['reserved_seconds']
                and runtime['original_governor'] == config['original_governor'] and runtime['restored'],
                'Full attempt budget/restoration differs')
        require(runtime['started_monotonic'] <= observations[0]['started']
                <= observations[-1]['ended'] <= runtime['finished_monotonic'], 'Measurement outside charged attempt')
        restoration = read(directory / 'governor-restoration.json')
        require(restoration['restored'] and restoration['current'] == config['original_governor'], 'Governor restoration failed')
        telemetry, damaged = read_jsonl(directory / 'telemetry.jsonl')
        require(not damaged and telemetry, 'Telemetry missing/damaged')
        for row in telemetry:
            require(not row.get('stop_reason') and safety_reason(row, manifest['initial_throttle_flags'], require_hat=False) is None,
                    'Telemetry guard failed')
        require(all(finite(row['monotonic']) for row in telemetry)
                and [row['monotonic'] for row in telemetry] == sorted(row['monotonic'] for row in telemetry)
                and runtime['started_monotonic'] <= telemetry[0]['monotonic'] <= observations[0]['started']
                and observations[-1]['ended'] <= telemetry[-1]['monotonic'] <= runtime['finished_monotonic'],
                'Telemetry does not cover measurement interval')
        checks.append('supervised cumulative-time evidence, safe telemetry and restored settings')
        return dict(passed=True, errors=[], checks=checks, measurements=len(observations), fixtures=len(fs),
                    accepted=summary['accepted'], offline_limit='Recorded process, affinity and clock evidence is checked; hardware execution is not replayed.')
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        return dict(passed=False, errors=[message], error=message, checks=checks)
