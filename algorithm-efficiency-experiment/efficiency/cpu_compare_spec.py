"""Frozen CPU baseline comparison, fixture schedule and paired-block analysis.

This module opens no devices. Configuration files cannot override scientific
thresholds, block counts or the bootstrap procedure.
"""
from copy import deepcopy
import hashlib
import itertools
import math
import os
import random

import numpy as np

from .refine_spec import POLICIES
from .reliability_spec import CPU_ARMS
from .study_spec import CELLS

VERSION = 'cpu-compare-v1'
EVENTS = 'cpu-compare.jsonl'
MAX_SECONDS = 14400
BLOCKS = 256
REPEATS = 3
BATCH_CALLS = 16
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_SAMPLES = 10000
CONFIGURATIONS = ('baseline', 'candidate')

# Include standard/libgomp controls even when they were absent in the parent.
# Prefix matching additionally removes vendor extensions and indexed ICVs.
OPENMP_KEYS = (
    'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_NESTED', 'OMP_SCHEDULE',
    'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_WAIT_POLICY', 'OMP_STACKSIZE',
    'OMP_THREAD_LIMIT', 'OMP_MAX_ACTIVE_LEVELS', 'OMP_CANCELLATION',
    'OMP_DEFAULT_DEVICE', 'OMP_TARGET_OFFLOAD', 'OMP_MAX_TASK_PRIORITY',
    'OMP_DISPLAY_ENV', 'OMP_DISPLAY_AFFINITY', 'OMP_AFFINITY_FORMAT',
    'OMP_TOOL', 'OMP_TOOL_LIBRARIES', 'OMP_TOOL_VERBOSE_INIT',
    'OMP_ALLOCATOR', 'OMP_NUM_TEAMS', 'OMP_TEAMS_THREAD_LIMIT',
    'GOMP_CPU_AFFINITY', 'GOMP_STACKSIZE', 'GOMP_SPINCOUNT', 'GOMP_DEBUG',
    'GOMP_RTEMS_THREAD_POOLS',
)


def policy(configuration):
    if configuration == 'baseline':
        return dict(id='baseline', governor='ondemand', binding='default', waiting='default')
    if configuration == 'candidate':
        return deepcopy(POLICIES[5])
    raise ValueError('Unknown CPU comparison configuration')


def environment(configuration):
    """Child-environment overlay; ``None`` means remove an inherited key."""
    policy(configuration)
    keys = set(OPENMP_KEYS) | {key for key in os.environ if key.startswith(('OMP_', 'GOMP_'))}
    desired = dict.fromkeys(sorted(keys))
    if configuration == 'candidate':
        from .refine_worker import environment as policy_environment
        desired.update(policy_environment(POLICIES[5]))
    return desired


def specification():
    # Do not freeze keys discovered in the launching shell: prefix clearing is
    # the contract and each worker records the actual environment it observed.
    candidate = {key: value for key, value in environment('candidate').items() if value is not None}
    return dict(
        version=VERSION, blocks=BLOCKS, max_seconds=MAX_SECONDS,
        cleanup_seconds=120, pilot_blocks=1, pilot_margin=1.2,
        configurations=list(CONFIGURATIONS),
        configuration_policies={name: policy(name) for name in CONFIGURATIONS},
        openmp=dict(clear_prefixes=['OMP_', 'GOMP_'], baseline={}, candidate=candidate),
        cells=deepcopy(CELLS), cpu_arms=list(CPU_ARMS), repeats=REPEATS,
        batch_calls=BATCH_CALLS, configuration_order='alternate each paired block',
        arm_order='all 24 permutations per 8 blocks; identical between configurations',
        equivalence_interval=[.95, 1.05], minimum_speedup=1.10,
        max_shape_slowdown=1.05, atol=1e-5, rtol=1e-4,
        bootstrap_seed=BOOTSTRAP_SEED, bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_unit='whole paired blocks; average repeats and duplicate controls per shape; equal shapes',
        timing='submission through validated result; initialization reported separately',
        retry_policy='no automatic retries; infrastructure only with fresh fixtures within cumulative allowance',
        scope='six streaming attention shapes; individual requests and 16-call batches',
        defaults_changed=False,
    )


def seed(config, fixture_id):
    namespace = config['fixture_namespace']
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('A fresh nonempty fixture namespace is required')
    key = f'{VERSION}|{namespace}|{fixture_id}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def fixtures(config):
    return [dict(cell, fixture_id=f'{section}-{cell["cell_id"]}-block{block}',
                 section=section, block=block,
                 seed=seed(config, f'{section}-{cell["cell_id"]}-block{block}'))
            for section, blocks in (('pilot', [-1]), ('main', range(BLOCKS)))
            for block in blocks for cell in CELLS]


def orders(config, cell, block, section):
    values = list(itertools.permutations(CPU_ARMS))
    cycle = max(0, block) // 8
    random.Random(seed(config, f'orders|{section}|{cell}|{cycle}')).shuffle(values)
    offset = max(0, block) % 8 * REPEATS
    return values[offset:offset + REPEATS]


def schedule(config, fs):
    jobs = []
    for section in ('pilot', 'main'):
        for block in sorted({f['block'] for f in fs if f['section'] == section}):
            selected = [f for f in fs if f['section'] == section and f['block'] == block]
            configurations = CONFIGURATIONS if max(0, block) % 2 == 0 else CONFIGURATIONS[::-1]
            for configuration in configurations:
                for f in selected:
                    for repeat, order in enumerate(orders(config, f['cell_id'], block, section)):
                        for position, label in enumerate(order):
                            jobs.append(dict(fixture_id=f['fixture_id'], section=section, block=block,
                                             cell_id=f['cell_id'], configuration=configuration,
                                             repeat=repeat, position=position, label=label))
    return jobs


def arm(cell, label):
    if label not in CPU_ARMS:
        raise ValueError('Unknown CPU comparison arm')
    return dict(arm=label, backend=cell['cpu'], variant=0)


def interval(a, b):
    """Paired bootstrap over complete blocks, with the preregistered draw count."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError('Bootstrap requires matched complete blocks')
    if not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b > 0).all()):
        raise ValueError('Bootstrap timing must be finite and positive')
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        len(a), size=(BOOTSTRAP_SAMPLES, len(a)))
    draws = np.sort(a[indices].mean(1) / b[indices].mean(1))
    return dict(estimate=float(a.mean() / b.mean()),
                interval=[float(draws[int(BOOTSTRAP_SAMPLES * .025)]),
                          float(draws[int(BOOTSTRAP_SAMPLES * .975)])],
                seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES, blocks=len(a))


def equivalent(value):
    low, high = value['interval']
    return .95 <= low <= high <= 1.05


def _correct(row, cell):
    """Validate every call, including the full contents of a 16-call batch."""
    result = row['response']['result']
    if result.get('correct') is not True:
        return False
    if row['label'].startswith('batch'):
        calls = result.get('requests')
        if not isinstance(calls, list) or len(calls) != BATCH_CALLS:
            return False
        if [call.get('request_id') for call in calls] != list(range(BATCH_CALLS)):
            return False
    else:
        if 'requests' in result:
            return False
        calls = [result]
    expected = arm(cell, row['label'])
    return all(call.get('fixture_id') == row['fixture_id']
               and all(call.get(key) == value for key, value in expected.items())
               and call.get('correct') is True and call.get('matches_warmup') is True
               and call.get('validation_errors') == 0
               and call.get('unexpected_output_io') is False for call in calls)


def analyze(events, config):
    result = dict(version=VERSION, complete=False, accepted=False,
                  decision='incomplete', metrics=[], defaults_changed=False)
    measurements = [row for row in events if row.get('event') == 'measurement']
    jobs = schedule(config, fixtures(config))
    markers = [index for index, row in enumerate(events) if row.get('event') == 'measurement_complete']
    if len(markers) != 1 or len(measurements) != len(jobs):
        return result
    if any(row.get('event') == 'measurement' for row in events[markers[0] + 1:]):
        result['decision'] = 'invalid_schedule'
        return result
    if any(any(row.get(key) != value for key, value in job.items())
           for row, job in zip(measurements, jobs)):
        result['decision'] = 'invalid_schedule'
        return result
    bycell = {cell['cell_id']: cell for cell in CELLS}
    try:
        valid_times = all(isinstance(row['total_ms'], (int, float))
                          and not isinstance(row['total_ms'], bool)
                          and math.isfinite(row['total_ms']) and row['total_ms'] > 0
                          for row in measurements)
        correct = {kind: all(_correct(row, bycell[row['cell_id']]) for row in measurements
                             if row['label'].startswith(kind)) for kind in ('single', 'batch')}
    except (KeyError, TypeError, ValueError, AttributeError):
        result['decision'] = 'invalid_results'
        return result
    if not valid_times:
        result['decision'] = 'invalid_results'
        return result
    result['complete'] = True
    # Exact schedule coverage above makes each slot unique and fixes all group
    # denominators before any floating-point analysis takes place.
    samples = np.empty((2, BLOCKS, len(CELLS), REPEATS, len(CPU_ARMS)), dtype=float)
    cell_indices = {cell['cell_id']: index for index, cell in enumerate(CELLS)}
    for row in measurements:
        if row['section'] == 'main':
            samples[CONFIGURATIONS.index(row['configuration']), row['block'],
                    cell_indices[row['cell_id']], row['repeat'], CPU_ARMS.index(row['label'])] = row['total_ms']
    repeated = samples.mean(axis=3)
    for kind, pair in (('single', (0, 1)), ('batch', (2, 3))):
        duplicate_checks = {}
        for index, configuration in enumerate(CONFIGURATIONS):
            duplicates = repeated[index]
            checks = {'aggregate': interval(duplicates[:, :, pair[1]].mean(1),
                                             duplicates[:, :, pair[0]].mean(1))}
            checks.update({cell['cell_id']: interval(duplicates[:, ci, pair[1]], duplicates[:, ci, pair[0]])
                           for ci, cell in enumerate(CELLS)})
            duplicate_checks[configuration] = checks
        # Mean duplicate controls within each shape, then weight shapes equally.
        byshape = repeated[:, :, :, list(pair)].mean(axis=3)
        byblock = byshape.mean(axis=2)
        speed = interval(byblock[0], byblock[1])
        per_cell = {cell['cell_id']: interval(byshape[0, :, ci], byshape[1, :, ci])
                    for ci, cell in enumerate(CELLS)}
        slowdown = max(float(byshape[1, :, ci].mean() / byshape[0, :, ci].mean())
                       for ci in range(len(CELLS)))
        stable = all(equivalent(check) for checks in duplicate_checks.values() for check in checks.values())
        qualified = (correct[kind] and stable and speed['estimate'] >= 1.10
                     and speed['interval'][0] > 1 and slowdown <= 1.05)
        result['metrics'].append(dict(label=kind, kind=kind, speedup=speed,
            mean_ms={name: float(byblock[i].mean()) for i, name in enumerate(CONFIGURATIONS)},
            per_cell_speedup=per_cell, max_shape_slowdown=slowdown,
            equivalence=duplicate_checks, stable=stable, correct=correct[kind], qualified=qualified))
    result['accepted'] = all(metric['qualified'] for metric in result['metrics'])
    result['decision'] = 'cpu_improved' if result['accepted'] else 'no_qualified_candidate'
    return result
