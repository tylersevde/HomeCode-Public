import contextlib
import importlib.util
import io
import json
import platform
import random
import resource
import time

import numpy as np

from .common import ROOT, atomic_json, emit, progress, wait_cool


def reference_module():
    spec = importlib.util.spec_from_file_location('archive_demo', ROOT / 'reference/algorithm_efficiency_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(directory, config):
    module = reference_module()
    progress(directory, 'cpu', 'correctness_checks')
    checks = module.correctness_checks()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        np.show_config()
    atomic_json(directory / 'cpu-environment.json', dict(
        python=platform.python_version(), numpy=np.__version__,
        numpy_build_configuration=captured.getvalue(), correctness_checks=checks,
        correctness_tolerances=dict(rtol=1e-10, atol=1e-10), blas_threads_requested=1))
    emit(directory / 'cpu.jsonl', 'correctness', passed=len(checks),
         max_absolute_error=max(row['max_absolute_error'] for row in checks))
    for n in config['cpu_lengths']:
        wait_cool(directory)
        progress(directory, 'cpu', 'warmup', sequence_length=n)
        inputs, weights = module.fixture(n, 64, 20261001 + n)
        expected = module.full_prefix(inputs, weights)
        errors = {}
        for name, function in module.VARIANTS.items():
            actual = function(inputs, weights)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            errors[name] = float(np.max(np.abs(actual - expected)))
        # Match the archive's NumPy order generator and original timing seeds.
        rng = np.random.default_rng(20261001 + n + 10000)
        for repetition in range(config['cpu_repeats']):
            wait_cool(directory)
            order = list(module.VARIANTS)
            rng.shuffle(order)
            for position, name in enumerate(order):
                progress(directory, 'cpu', 'measurement', sequence_length=n,
                         repetition=repetition, variant=name)
                cpu_start = time.process_time()
                start = time.perf_counter_ns()
                actual = module.VARIANTS[name](inputs, weights)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                cpu_seconds = time.process_time() - cpu_start
                # Validation is outside the measured interval for every sample.
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                emit(directory / 'cpu.jsonl', 'measurement', sequence_length=n,
                     hidden_dimension=64, seed=20261001+n, repetition=repetition,
                     order_position=position, variant=name, milliseconds=elapsed_ms,
                     cpu_seconds=cpu_seconds, max_absolute_error=float(np.max(np.abs(actual-expected))),
                     mac_count=module.multiply_add_counts(n, 64)[name],
                     persistent_kv_cache_bytes=2*n*64*8 if name == 'cached_stream' else 0,
                     process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        print(f'CPU n={n}: {config["cpu_repeats"]} repetitions per variant complete', flush=True)
    emit(directory / 'cpu.jsonl', 'complete')
