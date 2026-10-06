"""Independent reconstruction of GPU crossover fixtures, routes and qualification."""
import json
from pathlib import Path

import numpy as np

from . import completion_gpu_spec as design
from .attention_native import fixture, oracle, errors
from .common import digest_file, read_jsonl
from .completion_evidence import audit_budget
from .cpu_compare_audit import finite, local_path
from .cpu_compare_spec import environment
from .cpu_policy import NUMERICAL_LIBRARY_THREADS
from .feedback_spec import input_hash
from .hybrid import verify_artifacts
from .monitor import safety_reason
from .refine_audit import require
from .refine_spec import POLICIES
from .refine_worker import environment as policy_environment


def read(path):
    return json.loads(Path(path).read_text())


def independent_interval(a, b):
    left, right = np.asarray(a, float), np.asarray(b, float)
    indices = np.random.Generator(np.random.PCG64(20261007)).integers(len(left), size=(10000, len(left)))
    ratios = np.sort(left[indices].mean(1)/right[indices].mean(1))
    return dict(estimate=float(left.mean()/right.mean()), interval=[float(ratios[250]), float(ratios[9750])],
                seed=20261007, samples=10000, blocks=len(left))


def check_decisions(events, config, summary):
    """Reconstruct scalar groups and routing independently; never call analyze()."""
    rows = [event for event in events if event['event'] == 'measurement' and event['section'] == 'main']
    count = design.BLOCKS[config['stage']]
    cells = [cell['cell_id'] for cell in design.CELLS]
    values = {}
    for row in rows:
        values.setdefault((row['cell_id'], row['label'], row['block']), []).append(row['total_ms'])
    require(len(values) == len(cells)*len(design.LABELS)*count
            and all(len(group) == design.REPEATS for group in values.values()), 'Statistical grouping differs')
    def vector(cell, label):
        return np.array([np.mean(values[cell, label, block]) for block in range(count)])
    grouped = {(cell, label): vector(cell, label) for cell in cells for label in design.LABELS}
    baseline = {cell: (grouped[cell, 'cpu']+grouped[cell, 'cpu_duplicate'])/2 for cell in cells}
    aggregate = lambda collection: np.stack([collection[cell] for cell in cells], axis=1).mean(1)
    checks = {'aggregate': independent_interval(aggregate({c: grouped[c, 'cpu_duplicate'] for c in cells}),
                                               aggregate({c: grouped[c, 'cpu'] for c in cells}))}
    checks.update({cell: independent_interval(grouped[cell, 'cpu_duplicate'], grouped[cell, 'cpu']) for cell in cells})
    stable = all(.95 <= check['interval'][0] <= check['interval'][1] <= 1.05 for check in checks.values())
    correct = all(event['response']['result']['correct'] and event['response']['result']['matches_warmup']
                  and not event['response']['result']['validation_errors']
                  and not event['response']['result']['unexpected_output_io']
                  for event in events if event['event'] == 'measurement')
    cell_speeds = {cell: independent_interval(baseline[cell], grouped[cell, 'stream64']) for cell in cells}
    if config['stage'] == 'develop':
        routes = {cell: ('stream64' if correct and stable and cell_speeds[cell]['estimate'] >= 1.10
                         and cell_speeds[cell]['interval'][0] > 1 else 'cpu') for cell in cells}
    else:
        routes = config['selected']['routes']
    selected = {cell: grouped[cell, 'stream64'] if routes[cell] == 'stream64' else baseline[cell] for cell in cells}
    speedup = independent_interval(aggregate(baseline), aggregate(selected))
    slowdowns = {cell: float(selected[cell].mean()/baseline[cell].mean()) for cell in cells}
    qualified = bool(stable and correct and 'stream64' in routes.values() and speedup['estimate'] >= 1.10
                     and speedup['interval'][0] > 1 and max(slowdowns.values()) <= 1.05)
    metrics = [dict(cell_id=cell, route=routes[cell],
                    mean_ms={label: float(grouped[cell, label].mean()) for label in design.LABELS},
                    cpu_mean_ms=float(baseline[cell].mean()), stream64_cpu_speedup=cell_speeds[cell],
                    old_gpu_speedup=independent_interval(grouped[cell, 'gpu'], grouped[cell, 'stream64']),
                    routed_slowdown=slowdowns[cell]) for cell in cells]
    expected = dict(version=design.VERSION, track='gpu', stage=config['stage'], complete=True,
                    accepted=qualified and config['stage'] == 'confirm', selected=dict(routes=routes) if qualified else None,
                    decision=('confirmed' if config['stage'] == 'confirm' else 'candidate_selected') if qualified else 'no_qualified_candidate',
                    correct=bool(correct), stable=bool(stable), equivalence=checks, routes=routes,
                    speedup=speedup, max_shape_slowdown=max(slowdowns.values()), metrics=metrics,
                    old_gpu_speedup=independent_interval(aggregate({c: grouped[c, 'gpu'] for c in cells}),
                                                       aggregate({c: grouped[c, 'stream64'] for c in cells})),
                    defaults_changed=False, routing_estimand=design.ROUTING_ESTIMAND)
    require(all(summary.get(key) == value for key, value in expected.items()), 'Independent GPU routing/statistical decision differs')


def check_source(directory, config, manifest, freeze):
    require(config['profile'] == 'completion' and config['track'] == 'gpu'
            and 120 < config['max_seconds'] <= 7200, 'GPU stage identity or budget differs')
    require(read(directory/'protocol.json') == design.spec(config), 'Frozen GPU specification changed')
    require(manifest['config'] == config, 'Manifest configuration differs')
    require(freeze['source_sha256'] == manifest['source_sha256'], 'Source freeze differs')
    for name, checksum in manifest['source_sha256'].items():
        require(digest_file(local_path(directory, 'source/'+name)) == checksum, 'Source snapshot changed: '+name)
    for key, relative in (('config_sha256', 'config.json'), ('protocol_sha256', 'protocol.json'),
                          ('prior_fixtures_sha256', 'prior-fixtures.json'), ('provenance_sha256', 'gpu-provenance.json'),
                          ('native_build_sha256', 'native-build/build.json')):
        require(freeze[key] == digest_file(directory/relative), 'Freeze hash differs: '+relative)
    parents = read(directory/'gpu-provenance.json')
    expected = {'gpu-confirmed'} | ({'gpu-develop'} if config['stage'] == 'confirm' else set())
    require(set(parents) == expected, 'GPU parent coverage differs')
    for name, entry in parents.items():
        parent = Path(entry['path'])
        verify_artifacts(parent)
        require(parent.resolve() == Path(config['parents'][name]).resolve()
                and digest_file(parent/'checksums.json') == entry['checksums_sha256'], 'GPU parent identity changed')
    historical = Path(parents['gpu-confirmed']['path'])
    prior = read(historical/'summary.json')
    require(read(historical/'validation.json')['passed'] and prior['complete'] and prior['accepted']
            and prior['selected'] == 5 and read(historical/'config.json')['reliability_stage'] == 'gpu-confirm',
            'Historical stream64 confirmation did not qualify')
    build = read(directory/'native-build/build.json')
    require(build == read(historical/'native-build/build.json'), 'Historical native binary identity differs')
    require({'libattention.so', 'libaffinity.so', 'project-stream64.spv', 'scores-stream64.spv', 'apply-stream64.spv'}
            <= set(build['artifacts']), 'Required native artifacts missing')
    for name, checksum in build['artifacts'].items():
        require(digest_file(local_path(directory, 'native-build/'+name)) == checksum, 'Native artifact changed')
    for name, checksum in build['sources'].items():
        require(manifest['source_sha256'][name] == checksum, 'Native source/build mismatch')
    wrapper = 'efficiency/attention_native.py'
    require(manifest['source_sha256'][wrapper] == read(historical/'manifest.json')['source_sha256'][wrapper],
            'Native Python wrapper changed')
    if config['stage'] == 'confirm':
        development = Path(parents['gpu-develop']['path'])
        require(read(development/'config.json')['stage'] == 'develop', 'Confirmation parent is not development')
        require(audit(development)['passed'], 'Development evidence failed independent audit')
        selection = read(development/'summary.json')['selected']
        require(selection is not None and selection == config['selected'], 'Confirmation selected different GPU routes')
        require(read(development/'manifest.json')['source_sha256'] == manifest['source_sha256'], 'Source changed after selection')
        require(read(development/'config.json')['fixture_namespace'] != config['fixture_namespace'], 'Confirmation namespace reused')
        require(set(selection['routes']) == {cell['cell_id'] for cell in design.CELLS}
                and all(value in ('cpu', 'stream64') for value in selection['routes'].values()), 'Frozen route coverage differs')


def check_controls(directory):
    rows = read(directory/'correctness.json')
    expected = []
    for n, d, b in ((1, 1, 1), (17, 7, 2), (33, 17, 4), (65, 65, 2)):
        name = f'controls/n{n}-d{d}-b{b}.npz'
        x, weights = fixture(n, d, b, 17701+n)
        with np.load(directory/name, allow_pickle=False) as data:
            require(np.array_equal(data['x'], x) and np.array_equal(data['w'], weights)
                    and np.array_equal(data['expected'], oracle(x, weights)), 'GPU control fixture changed')
        for mode, variants in (('stream', (0, 3, 4)), ('prefill', (0, 1, 2))):
            for variant in variants:
                expected.append(('oracle', mode, variant, name))
                if variant in (3, 4):
                    expected.append(('semantic', mode, variant, name))
    require([(r['kind'], r['mode'], r['variant'], r['input_file']) for r in rows] == expected,
            'GPU control matrix coverage/order differs')
    for row in rows:
        with np.load(local_path(directory, row['input_file']), allow_pickle=False) as data:
            x, weights, expected = data['x'], data['w'], data['expected']
        if row['kind'] == 'oracle':
            output = np.load(local_path(directory, row['output_file']), allow_pickle=False)
            err = errors(output, expected)
            require(all(row[key] == value for key, value in err.items()) and err['correct']
                    and row['validation_errors'] == 0, 'GPU FP64 oracle control failed')
        else:
            output = {key: np.load(local_path(directory, path), allow_pickle=False) for key, path in row['output_files'].items()}
            cut = max(1, x.shape[1]//2)
            require(row['cut'] == cut and row['correct'] and set(output) == {'baseline', 'perturbed', 'repeated'},
                    'GPU semantic control fields differ')
            altered = x.copy(); altered[0, cut:] *= -1
            require(errors(output['baseline'], expected)['correct']
                    and errors(output['perturbed'], oracle(altered, weights))['correct']
                    and np.array_equal(output['baseline'], output['repeated'])
                    and np.array_equal(output['baseline'][0, :cut], output['perturbed'][0, :cut])
                    and np.array_equal(output['baseline'][1:], output['perturbed'][1:]), 'GPU causality/reset/batch isolation failed')


def audit(directory, require_seal=True):
    directory = Path(directory); checks = []
    try:
        if require_seal:
            verify_artifacts(directory); checks.append('artifact seal')
        config = read(directory/'config.json'); manifest = read(directory/'manifest.json')
        budget = audit_budget(directory, require_final=require_seal)
        require(budget['passed'], 'Cumulative completion budget audit failed')
        checks.append('cumulative completion budget receipt and lineage')
        check_source(directory, config, manifest, read(directory/'freeze.json'))
        checks.append('protocol, native implementation, source and parent selection')
        check_controls(directory); checks.append('independently reconstructed FP64 and semantic GPU controls')
        fs = read(directory/'fixtures.json'); planned = design.fixtures(config)
        prior = read(directory/'prior-fixtures.json'); seeds = set(prior['seeds']); hashes = set(prior['hashes'])
        require(len(fs) == len(planned), 'Fixture matrix incomplete')
        expected_arrays = {}
        for actual, wanted in zip(fs, planned):
            require(all(actual.get(key) == value for key, value in wanted.items()), 'Fixture specification/seed changed')
            require(actual['seed'] not in seeds and actual['input_sha256'] not in hashes, 'Fixture reused')
            seeds.add(actual['seed']); hashes.add(actual['input_sha256'])
            path = directory/'fixtures'/(wanted['fixture_id']+'.npz')
            require(digest_file(path) == actual['file_sha256'], 'Fixture file digest changed')
            x, weights = fixture(wanted['n'], wanted['d'], wanted['b'], wanted['seed']); expected = oracle(x, weights)
            with np.load(path, allow_pickle=False) as saved:
                require(np.array_equal(saved['x'], x) and np.array_equal(saved['w'], weights)
                        and np.array_equal(saved['expected'], expected), 'Fixture/FP64 oracle changed')
            require((actual['input_sha256'], actual['weights_sha256'], actual['oracle_sha256'])
                    == (input_hash(x), input_hash(weights), input_hash(expected)), 'Fixture array digest differs')
            # Keep only metadata; large FP64 tensors are reloaded one fixture at a time below.
            expected_arrays[wanted['fixture_id']] = path
        jobs = design.schedule(config, fs)
        require(read(directory/'schedule.json') == jobs, 'Frozen schedule changed')
        events, damaged = read_jsonl(directory/design.EVENTS)
        require(not damaged, 'Raw event stream damaged')
        rows = [event for event in events if event['event'] == 'measurement']
        require(len(rows) == len(jobs), 'Measurement coverage incomplete')
        for event, job in zip(rows, jobs):
            require(all(event.get(key) == value for key, value in job.items()), 'Measurement order/substitution differs')
            require(finite(event['total_ms']) and event['total_ms'] > 0
                    and event['total_ms'] == (event['ended']-event['started'])*1000, 'Request timing altered')
        frozen = [event for event in events if event['event'] == 'fixtures_frozen']
        require(len(frozen) == 1 and frozen[0]['fixtures_sha256'] == digest_file(directory/'fixtures.json')
                and frozen[0]['schedule_sha256'] == digest_file(directory/'schedule.json'), 'Fixture freeze differs')
        byid = {f['fixture_id']: f for f in fs}; active = None; warmed = {}; verified = {}
        complete = gate_seen = controls_done = frozen_seen = False
        transitions = readiness = releases = measured = 0; last_end = 0
        for event in events:
            kind = event['event']
            require(kind in {'governor_transition', 'correctness_complete', 'fixtures_frozen', 'worker_ready',
                             'warmup', 'measurement', 'pilot_gate', 'worker_release', 'measurement_complete'},
                    'Unexpected device activity or recorded failure')
            if kind == 'governor_transition':
                transitions += 1
                require(active is None and readiness == 0 and event['current'] == 'performance'
                        and event['original'] == config['original_governor'], 'Governor transition differs')
            elif kind == 'correctness_complete':
                require(active is None and transitions == 1 and not controls_done and not frozen_seen
                        and event['checks'] == 32, 'GPU control completion differs')
                controls_done = True
            elif kind == 'fixtures_frozen':
                require(controls_done and not frozen_seen and readiness == 0, 'Fixture freeze lifecycle differs')
                frozen_seen = True
            elif kind == 'worker_ready':
                readiness += 1
                require(active is None and not complete and controls_done and frozen_seen, 'Owner lifecycle differs')
                active = event
                env = event['environment']
                require(env['numeric_device'] == 'both' and finite(env['cpu_initialization_ms'])
                        and finite(env['gpu_initialization_ms']) and not any(k.startswith(('hailort_', 'model_')) for k in env),
                        'Unexpected device context or missing initialization')
                require(env['policy'] == POLICIES[5] and env['observed']['governor'] == 'performance'
                        and env['openmp_environment'] == policy_environment(POLICIES[5])
                        and env['openmp_actual'] == {k: v for k, v in environment('candidate').items() if v is not None}
                        and env['numerical_library_threads'] == NUMERICAL_LIBRARY_THREADS, 'Numerical environment differs')
                require([p['requested'] for p in env['probes']] == [1, 4], 'Actual OpenMP probe coverage differs')
                for probe in env['probes']:
                    require(probe['team'] == probe['requested']
                            and [r['mask'] for r in probe['threads']] == [1 << i for i in range(probe['requested'])],
                            'Actual OpenMP affinity/team differs')
            elif kind in ('warmup', 'measurement'):
                require(active is not None and not complete and event['instance'] == active['instance'], 'Work without expected owner')
                response = event['response']; result = response['result']; fid = event['fixture_id']; label = event['label']
                require(response['owner'] == active['owner'], 'Response owner differs')
                expected_arm = design.arm(byid[fid], label)
                require(result['fixture_id'] == fid and all(result.get(k) == v for k, v in expected_arm.items()), 'Numerical route substituted')
                require(result['policy'] == POLICIES[5] and finite(result['observer_ms'])
                        and all(result[key]['governor'] == 'performance' for key in ('system_before', 'system_after')), 'Governor/observer drift')
                key = fid, result['output_file']
                if key not in verified:
                    with np.load(expected_arrays[fid], allow_pickle=False) as data:
                        output = np.load(local_path(directory, result['output_file']), allow_pickle=False)
                        require(output.shape == data['expected'].shape and output.dtype == np.float32, 'Output shape/dtype differs')
                        verified[key] = errors(output, data['expected']), input_hash(output)
                err, checksum = verified[key]
                require(checksum == result['output_sha256'] and all(result.get(k) == v for k, v in err.items())
                        and result['correct'] and result['validation_errors'] == 0 and result['matches_warmup']
                        and not result['unexpected_output_io'], 'Output/oracle/warmup verification differs')
                require(all(finite(result[k]) for k in ('started', 'computed', 'ended', 'worker_request_ms', 'request_ms', 'cpu_seconds'))
                        and response['received'] <= result['started'] <= result['computed'] <= result['ended'] <= response['ended']
                        and result['worker_request_ms'] == (result['ended']-result['started'])*1000,
                        'Native validation timing differs')
                if kind == 'warmup':
                    require((fid, label) not in warmed, 'Duplicate warmup')
                    warmed[fid, label] = checksum
                else:
                    require(warmed.get((fid, label)) == checksum, 'Missing or changed warmup')
                    require(last_end <= event['started'] <= response['submitted'] <= response['received']
                            <= response['ended'] <= response['delivered'] <= event['ended'], 'Response clocks differ')
                    last_end = event['ended']; measured += 1
                    require(event['section'] != 'main' or gate_seen, 'Main work before pilot gate')
            elif kind == 'pilot_gate':
                require(active is not None and not gate_seen and not complete, 'Pilot lifecycle differs')
                require(measured == sum(job['section'] == 'pilot' for job in jobs), 'Pilot incomplete')
                count = design.BLOCKS[config['stage']]
                require(finite(event['pilot_seconds']) and event['pilot_seconds'] > 0 and event['multiplier'] == count
                        and event['required_seconds'] == event['pilot_seconds']*count*1.2 and event['fits']
                        and event['required_seconds'] <= event['remaining_work_seconds']
                        and event['pilot_seconds'] >= sum(row['total_ms'] for row in rows if row['section'] == 'pilot')/1000,
                        'Pilot feasibility changed')
                gate_seen = True
            elif kind == 'worker_release':
                require(active is not None and event['instance'] == active['instance'] and event['owner'] == active['owner']
                        and not event['alive'] and not event['forced'], 'Worker did not release cleanly')
                releases += 1; active = None
            elif kind == 'measurement_complete':
                require(not complete and active is None and gate_seen and measured == len(jobs), 'Completion lifecycle differs')
                complete = True
        require(complete and active is None and transitions == readiness == releases == 1
                and len(warmed) == len(fs)*len(design.LABELS), 'Incomplete numerical lifecycle')
        checks.append('fresh fixture reconstruction, full schedule, process ownership and every numerical result')
        check_decisions(events, config, read(directory/'summary.json'))
        checks.append('independent paired bootstrap and frozen routing decisions')
        outcome = read(directory/'outcome.json'); restoration = read(directory/'governor-restoration.json')
        runtime = read(directory/'runtime-finalization.json')
        require(outcome['status'] == 'complete' and not outcome.get('stop_reason') and finite(outcome['elapsed_seconds'])
                and outcome['elapsed_seconds'] <= config['max_seconds'] <= config['reserved_seconds'] <= 7200,
                'Supervised completion/budget failed')
        require(restoration['restored'] and restoration['current'] == restoration['original'] == config['original_governor'],
                'Governor restoration failed')
        require(all(finite(runtime[key]) for key in ('started_monotonic', 'finished_monotonic', 'charged_seconds', 'reserved_seconds'))
                and runtime['restored'] and runtime['reserved_seconds'] == config['reserved_seconds']
                and runtime['charged_seconds'] == runtime['finished_monotonic']-runtime['started_monotonic']
                and outcome['elapsed_seconds'] <= runtime['charged_seconds'] <= config['reserved_seconds']
                and runtime['original_governor'] == config['original_governor'], 'Full GPU attempt budget/restoration differs')
        require(runtime['started_monotonic'] <= rows[0]['started'] <= rows[-1]['ended'] <= runtime['finished_monotonic'],
                'GPU measurement outside charged attempt')
        telemetry, damaged = read_jsonl(directory/'telemetry.jsonl')
        require(telemetry and not damaged, 'Missing or damaged safety telemetry')
        require(all(not row.get('stop_reason') and safety_reason(row, manifest['initial_throttle_flags'], False) is None for row in telemetry),
                'Safety stop, throttling or memory/thermal limits failed')
        require(all(finite(row['monotonic']) for row in telemetry)
                and [row['monotonic'] for row in telemetry] == sorted(row['monotonic'] for row in telemetry)
                and runtime['started_monotonic'] <= telemetry[0]['monotonic'] <= rows[0]['started']
                and rows[-1]['ended'] <= telemetry[-1]['monotonic'] <= runtime['finished_monotonic'],
                'GPU telemetry does not cover measurement interval')
        checks.append('hardware allowance, thermal/memory protections and restoration')
        return dict(passed=True, checks=checks, measurements=len(rows), fixtures=len(fs),
                    scope='Frozen shape-specific route estimate on twelve tested streaming tensors; no measured dynamic-dispatcher overhead, model acceleration or energy claim')
    except Exception as exc:
        return dict(passed=False, checks=checks, error=f'{type(exc).__name__}: {exc}')
