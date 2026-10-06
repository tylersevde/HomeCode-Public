"""Independent bounded telemetry; no Hailo calls in the supervisor process."""
import json
import math
from pathlib import Path
import subprocess
import time

import psutil


def command(args):
    return subprocess.check_output(args, text=True, timeout=2).strip()


def safety_reason(row, initial_flags=0, require_hat=True):
    cpu = row.get('cpu_temp_c')
    flags = row.get('throttle_flags')
    if not isinstance(cpu, (int, float)) or not math.isfinite(cpu):
        return 'Essential CPU temperature telemetry is unavailable'
    if not isinstance(flags, int):
        return 'Essential throttle telemetry is unavailable'
    if cpu >= 80:
        return 'CPU reached the 80 C experiment stop limit'
    if flags & 0xf:
        return 'Active undervoltage, frequency capping, or throttling reported'
    if (flags & 0xf0000) & ~(initial_flags & 0xf0000):
        return 'New historical power/throttle flag appeared between samples'
    if row.get('available_memory_bytes', 0) < 300*1024**2:
        return 'Available host memory fell below 300 MiB'
    if require_hat:
        if not row.get('sensor_alive', False):
            return 'Independent HAT temperature sampler exited'
        age = row.get('hat_sample_age_s')
        if not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 20:
            return 'HAT temperature telemetry is stale or missing'
        temps = (row.get('hat_ts0_c'), row.get('hat_ts1_c'))
        if any(not isinstance(t, (int, float)) or isinstance(t, bool) or
               not math.isfinite(t) or not -40 < t < 150 for t in temps):
            return 'HAT temperature telemetry is invalid'
        if row.get('hat_consecutive_errors', 0) >= 2:
            return 'HAT temperature sampler reported two consecutive errors'
        if max(temps) >= 85:
            return 'HAT reached the conservative 85 C experiment stop limit'
    return None


def sample(directory, sensor=None, worker=None):
    row = dict(monotonic=time.monotonic(), cpu_temp_c=None, throttle_flags=None,
               cpu_clock_mhz=None, hat_max_c=None, sensor_alive=sensor is not None and sensor.poll() is None,
               available_memory_bytes=psutil.virtual_memory().available,
               host_cpu_percent=psutil.cpu_percent(), worker_rss_bytes=None, errors=[])
    try:
        row['cpu_temp_c'] = float(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
        row['throttle_flags'] = int(command(['vcgencmd', 'get_throttled']).split('=')[1], 16)
        row['cpu_clock_mhz'] = int(command(['vcgencmd', 'measure_clock', 'arm']).split('=')[1])/1e6
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        row['errors'].append(str(exc))
    if sensor is not None:
        try:
            state = json.loads((directory / 'hat-sensor-state.json').read_text())
            row.update({k: state.get(k) for k in ['hat_ts0_c', 'hat_ts1_c', 'hat_consecutive_errors', 'hat_error_count']})
            stamp = state.get('hat_sample_time')
            row['hat_sample_age_s'] = time.monotonic()-stamp if stamp is not None else None
            temps = (state.get('hat_ts0_c'), state.get('hat_ts1_c'))
            if all(isinstance(t, (int, float)) and math.isfinite(t) for t in temps):
                row['hat_max_c'] = max(temps)
        except (OSError, ValueError, TypeError) as exc:
            row['errors'].append(str(exc))
    if worker is not None and worker.poll() is None:
        try:
            process = psutil.Process(worker.pid)
            row['worker_rss_bytes'] = process.memory_info().rss
            row['worker_cpu_seconds'] = sum(process.cpu_times()[:2])
            row['worker_threads'] = process.num_threads()
        except psutil.Error:
            pass
    return row
