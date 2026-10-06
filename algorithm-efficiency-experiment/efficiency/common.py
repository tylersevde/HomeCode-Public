import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get('AE_MODEL_PATH', str(ROOT / 'models/hailo/v5.1.1/Qwen2.5-1.5B-Instruct.hef')))
PLOT_PYTHON = Path(os.environ.get('AE_PLOT_PYTHON', sys.executable))
ARCHIVE_SHA = '18bebaa5904a2c685198e5ba5688025d3156c284243abc3c414458a8a3f88cec'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, delete=False) as f:
            name = f.name
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def emit(path, event, **fields):
    row = dict(schema_version=1, utc=utc(), monotonic=time.monotonic(), event=event)
    row.update(fields)
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
        f.flush()
    return row


def read_jsonl(path):
    """Return readable observations and explicitly count damaged/partial lines."""
    rows, errors = [], []
    if not Path(path).exists():
        return rows, errors
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError('record is not an object')
            rows.append(value)
        except (ValueError, TypeError) as exc:
            errors.append(dict(line=number, error=str(exc)))
    return rows, errors


def progress(directory, phase, operation, **fields):
    atomic_json(Path(directory) / 'progress.json',
                dict(utc=utc(), monotonic=time.monotonic(), phase=phase, operation=operation, **fields))


def wait_cool(directory):
    """Workers cannot begin a block without a fresh supervisor observation."""
    while True:
        state = json.loads((Path(directory) / 'status.json').read_text())
        age = time.monotonic() - state['monotonic']
        if state.get('stop_reason'):
            raise RuntimeError(state['stop_reason'])
        if not 0 <= age <= 8:
            raise RuntimeError('Supervisor telemetry is stale')
        if state['cpu_temp_c'] < 65 and (state.get('hat_max_c') is None or state['hat_max_c'] < 60):
            return
        progress(directory, state['phase'], 'cooling')
        time.sleep(.5)


def stop_process_group(process, grace=5):
    """Only signal the child session created by this experiment."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(.05)
    # Also terminate surviving descendants if the original parent already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def profile(name):
    if name == 'adviser-validation':
        return dict(baseline_seconds=15, cooldown_seconds=15, schedule_seed=20261008)
    if name == 'coordination':
        return dict(baseline_seconds=15, cooldown_seconds=15, schedule_seed=20261007)
    if name == 'feedback-attention':
        return dict(baseline_seconds=30, cooldown_seconds=30, schedule_seed=20261006)
    if name == 'language-strict-validation':
        return dict(language_repeats=2, schedule_seed=20261005, baseline_seconds=30, cooldown_seconds=30)
    if name == 'language-validation':
        return dict(language_repeats=2, schedule_seed=20261004, baseline_seconds=30, cooldown_seconds=30)
    if name == 'language-query':
        return dict(baseline_seconds=3, cooldown_seconds=3)
    if name == 'hybrid-validation':
        return dict(hat_sizes=[128, 512, 1536], tables_per_size=10, cache_repeats=2,
                    turns=4, fixture_seed_base=2026100301, schedule_seed=20261003,
                    baseline_seconds=30, cooldown_seconds=30)
    if name == 'hybrid-query':
        return dict(baseline_seconds=3, cooldown_seconds=3)
    if name == 'cache-validation':
        return dict(hat_sizes=[128, 512, 1536], tables_per_size=10, cache_repeats=2,
                    turns=4, fixture_seed_base=2026100201, schedule_seed=20261002,
                    baseline_seconds=30, cooldown_seconds=30)
    if name == 'state-isolation':
        return dict(state_repeats=3, schedule_seed=20261002,
                    baseline_seconds=30, cooldown_seconds=30)
    if name in ('diagnostic', 'diagnostic-smoke'):
        smoke = name == 'diagnostic-smoke'
        return dict(hat_sizes=[128] if smoke else [128, 512, 1536], turns=3,
                    repeatability_repeats=1 if smoke else 10,
                    position_repeats=1 if smoke else 3, schedule_seed=20261002,
                    baseline_seconds=3 if smoke else 30,
                    cooldown_seconds=3 if smoke else 30)
    if name == 'smoke':
        return dict(cpu_lengths=[64], cpu_repeats=1, hat_sizes=[128], hat_pairs=1,
                    turns=2, baseline_seconds=3, cooldown_seconds=3)
    return dict(cpu_lengths=[128, 256, 512], cpu_repeats=7, hat_sizes=[128, 512, 1536],
                hat_pairs=7, turns=4, baseline_seconds=30, cooldown_seconds=30)
