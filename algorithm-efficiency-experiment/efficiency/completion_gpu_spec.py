"""Preregistered bounded streaming-attention GPU crossover study; no hardware IO."""
from copy import deepcopy
import hashlib
import itertools
import math
import random

import numpy as np

VERSION = 'completion-gpu-v1'
EVENTS = 'completion-gpu.jsonl'
MAX_SECONDS = 7200
BLOCKS = {'develop': 8, 'confirm': 16}
REPEATS = 3
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_SAMPLES = 10000
LABELS = ('cpu', 'cpu_duplicate', 'gpu', 'stream64')
ROUTING_ESTIMAND = ('Frozen shape-specific route estimate assembled from measured complete native calls; '
                    'unchanged CPU shapes use their paired CPU baseline; dynamic dispatcher overhead is unmeasured')
CELLS = [dict(cell_id=f'stream-n{n}-d{d}-b{b}', n=n, d=d, b=b, mode='stream',
              cpu='native1' if b == 1 else 'native4', gpu='C6')
         for n in (128, 512, 1024) for d in (64, 128) for b in (1, 16)]


def spec(config):
    stage = config['stage']
    if stage not in BLOCKS or config.get('track') != 'gpu':
        raise ValueError('Unknown completion GPU track/stage')
    return dict(version=VERSION, track='gpu', stage=stage, blocks=BLOCKS[stage],
        stage_max_seconds=MAX_SECONDS, track_max_seconds=14400, cleanup_seconds=120,
        pilot_blocks=1, pilot_margin=1.2, repeats=REPEATS, cells=deepcopy(CELLS),
        labels=list(LABELS), cpu_policy=5, gpu_backend='C6', stream64_variant=4,
        original_gpu_variant=0, atol=1e-5, rtol=1e-4, minimum_speedup=1.10,
        equivalence_interval=[.95, 1.05], maximum_shape_slowdown=1.05,
        bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_unit='whole paired fixture blocks; average three repetitions per shape; equally weighted shapes',
        ordering='all 24 arm permutations per eight blocks, independently shuffled per shape/cycle',
        selection='GPU only where development per-shape speedup >=1.10 and lower95% >1; all other routes CPU; aggregate policy gate required',
        confirmation='same complete matrix with independently generated fixtures and frozen development routing',
        timing='submission through validated result including transport, transfers and synchronization; initialization separate',
        routing_estimand=ROUTING_ESTIMAND,
        batching='tensor B consists of independent sequences in one native call; no sequential-call batching or submission redesign',
        basis=dict(historical_n1024_d64_b4_ms=dict(cpu=47.18, old_gpu=3740.01, stream64=1178.20),
                   reason='increase independent tensor sequences and feature dimension at the three established sequence lengths; CPU reference is the declared route, not a globally optimal thread count',
                   native_bounds=dict(n=4096, d=128, b=16, workspace_bytes=512*1024**2),
                   largest_matrix_workspace_bytes=workspace_bytes(1024, 128, 16),
                   conservative_largest_old_gpu_seconds=3.74001*4*2,
                   excluded='N2048/B16/D128 extrapolates to about 120 seconds per old-GPU call and cannot fit the full confirmation matrix; N4096/B16 additionally exceeds the native workspace guard',
                   forecast='saved-timing extrapolation is a planning bound, not a substitute for the frozen pilot'),
        defaults_changed=False)


def workspace_bytes(n, d, b):
    return (5*n*d*b + 3*d*d + n*n*b) * 4


def seed(config, name):
    key = f'{VERSION}|{config["fixture_namespace"]}|{config["stage"]}|{name}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def fixtures(config):
    spec(config)
    return [dict(cell, fixture_id=f'{section}-{cell["cell_id"]}-block{block}',
                 section=section, block=block,
                 seed=seed(config, f'{section}-{cell["cell_id"]}-block{block}'))
            for section, blocks in (('pilot', [-1]), ('main', range(BLOCKS[config['stage']])))
            for block in blocks for cell in CELLS]


def arm(cell, label):
    if label not in LABELS:
        raise ValueError('Unknown GPU comparison arm')
    return dict(arm=label, backend=cell['cpu'] if label.startswith('cpu') else cell['gpu'],
                variant=4 if label == 'stream64' else 0)


def orders(config, cell, block, section):
    values = list(itertools.permutations(LABELS))
    random.Random(seed(config, f'orders|{section}|{cell}|{max(0, block)//8}')).shuffle(values)
    offset = max(0, block) % 8 * REPEATS
    return values[offset:offset+REPEATS]


def schedule(config, fs):
    return [dict(fixture_id=f['fixture_id'], section=f['section'], block=f['block'], cell_id=f['cell_id'],
                 repeat=repeat, position=position, label=label)
            for f in fs for repeat, order in enumerate(orders(config, f['cell_id'], f['block'], f['section']))
            for position, label in enumerate(order)]


def interval(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not (
            np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError('Invalid paired timing blocks')
    ix = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(len(a), size=(BOOTSTRAP_SAMPLES, len(a)))
    draws = np.sort(a[ix].mean(1) / b[ix].mean(1))
    return dict(estimate=float(a.mean()/b.mean()), interval=[float(draws[250]), float(draws[9750])],
                seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES, blocks=len(a))


def equivalent(value):
    return .95 <= value['interval'][0] <= value['interval'][1] <= 1.05


def valid_result(event, cell):
    result = event['response']['result']
    return (result.get('fixture_id') == event['fixture_id']
            and all(result.get(key) == value for key, value in arm(cell, event['label']).items())
            and result.get('correct') is True and result.get('matches_warmup') is True
            and result.get('unexpected_output_io') is False and result.get('validation_errors') == 0)


def analyze(events, config):
    specification = spec(config)
    result = dict(version=VERSION, track='gpu', stage=config['stage'], complete=False, accepted=False,
                  selected=None, decision='incomplete', metrics=[], defaults_changed=False,
                  routing_estimand=ROUTING_ESTIMAND)
    rows = [event for event in events if event.get('event') == 'measurement']
    jobs = schedule(config, fixtures(config))
    markers = [i for i, event in enumerate(events) if event.get('event') == 'measurement_complete']
    if len(markers) != 1 or len(rows) != len(jobs):
        return result
    if (any(event.get('event') == 'measurement' for event in events[markers[0]+1:])
            or any(any(row.get(k) != v for k, v in job.items()) for row, job in zip(rows, jobs))):
        result['decision'] = 'invalid_schedule'
        return result
    cells = {cell['cell_id']: cell for cell in CELLS}
    try:
        if not all(isinstance(row['total_ms'], (int, float)) and not isinstance(row['total_ms'], bool)
                   and math.isfinite(row['total_ms']) and row['total_ms'] > 0 for row in rows):
            raise ValueError('Invalid timing')
        correct = all(valid_result(row, cells[row['cell_id']]) for row in rows)
    except (KeyError, TypeError, ValueError):
        result['decision'] = 'invalid_results'
        return result
    count = specification['blocks']
    samples = np.empty((count, len(CELLS), REPEATS, len(LABELS)))
    indices = {cell['cell_id']: i for i, cell in enumerate(CELLS)}
    for row in rows:
        if row['section'] == 'main':
            samples[row['block'], indices[row['cell_id']], row['repeat'], LABELS.index(row['label'])] = row['total_ms']
    means = samples.mean(2)
    baseline = means[:, :, :2].mean(2)
    checks = {'aggregate': interval(means[:, :, 1].mean(1), means[:, :, 0].mean(1))}
    checks.update({cell['cell_id']: interval(means[:, i, 1], means[:, i, 0]) for i, cell in enumerate(CELLS)})
    stable = all(equivalent(check) for check in checks.values())
    per_cell = {cell['cell_id']: interval(baseline[:, i], means[:, i, 3]) for i, cell in enumerate(CELLS)}
    if config['stage'] == 'develop':
        routes = {cell['cell_id']: ('stream64' if correct and stable and per_cell[cell['cell_id']]['estimate'] >= 1.10
                                   and per_cell[cell['cell_id']]['interval'][0] > 1 else 'cpu') for cell in CELLS}
    else:
        routes = deepcopy(config.get('selected', {}).get('routes', {}))
        if set(routes) != set(cells) or not all(route in ('cpu', 'stream64') for route in routes.values()):
            result['decision'] = 'invalid_selected_routes'
            return result
    routed = np.stack([means[:, i, 3] if routes[cell['cell_id']] == 'stream64' else baseline[:, i]
                       for i, cell in enumerate(CELLS)], axis=1)
    speedup = interval(baseline.mean(1), routed.mean(1))
    slowdown = max(float(routed[:, i].mean()/baseline[:, i].mean()) for i in range(len(CELLS)))
    qualified = bool(correct and stable and 'stream64' in routes.values() and speedup['estimate'] >= 1.10
                     and speedup['interval'][0] > 1 and slowdown <= 1.05)
    result.update(complete=True, accepted=qualified and config['stage'] == 'confirm',
                  selected=dict(routes=routes) if qualified else None,
                  decision=('confirmed' if config['stage'] == 'confirm' else 'candidate_selected') if qualified else 'no_qualified_candidate',
                  correct=correct, stable=stable, equivalence=checks, routes=routes,
                  speedup=speedup, max_shape_slowdown=slowdown,
                  old_gpu_speedup=interval(means[:, :, 2].mean(1), means[:, :, 3].mean(1)))
    for i, cell in enumerate(CELLS):
        result['metrics'].append(dict(cell_id=cell['cell_id'], route=routes[cell['cell_id']],
            mean_ms={label: float(means[:, i, li].mean()) for li, label in enumerate(LABELS)},
            cpu_mean_ms=float(baseline[:, i].mean()), stream64_cpu_speedup=per_cell[cell['cell_id']],
            old_gpu_speedup=interval(means[:, i, 2], means[:, i, 3]),
            routed_slowdown=float(routed[:, i].mean()/baseline[:, i].mean())))
    return result
