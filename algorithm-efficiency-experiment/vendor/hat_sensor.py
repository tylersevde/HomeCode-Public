#!/usr/bin/env python3
"""Read Hailo chip temperatures in an isolated process, never run inference.

Only successful reads advance hat_sample_time/hat_sample_seq. The parent owns
the >20-second last-success limit, consecutive-error policy, and TERM/KILL
deadline because an extension call can delay Python signal handling.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


def timestamp():
    return datetime.now().astimezone().isoformat(timespec='milliseconds')


def atomic_json(path, value):
    """Publish a complete JSON document or leave the prior document intact."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix='.'+path.name+'.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, allow_nan=False, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Sensor:
    def __init__(self, state_path, events_path, interval=3.0, monotonic=time.monotonic):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('interval must be finite and positive')
        self.state_path = Path(state_path)
        self.events_path = Path(events_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.monotonic = monotonic
        self.stop_requested = False
        self.stop_signal = None
        self.state = dict(schema_version=1, sensor_pid=os.getpid(),
                          sensor_status='starting', hat_ts0_c=None, hat_ts1_c=None,
                          hat_sample_time=None, hat_sample_seq=0,
                          hat_attempt_time=None, hat_attempt_finished_time=None,
                          hat_attempt_seq=0, hat_error='',
                          hat_consecutive_errors=0, hat_error_count=0)

    def publish(self):
        # This timestamp describes publication, NEVER the freshness of a reading.
        self.state['state_written_at'] = timestamp()
        atomic_json(self.state_path, self.state)

    def event(self, kind, **fields):
        row = dict(timestamp=timestamp(), event=kind, sensor_pid=os.getpid(), **fields)
        with self.events_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True)+'\n')

    def request_stop(self, sig=None, frame=None):
        # Do not do file I/O, acquire locks, or call Hailo from a signal handler.
        self.stop_requested = True
        self.stop_signal = sig

    def fail(self, exc, status='query_error'):
        self.state['hat_error'] = type(exc).__name__+': '+str(exc)[:2000]
        self.state['hat_error_count'] += 1
        self.state['hat_consecutive_errors'] += 1
        self.state['hat_attempt_finished_time'] = self.monotonic()
        self.state['sensor_status'] = status
        # Retain both last-good temperatures and their original success timestamp.
        self.publish()
        self.event(status, error=self.state['hat_error'],
                   consecutive_errors=self.state['hat_consecutive_errors'],
                   total_errors=self.state['hat_error_count'],
                   last_success_monotonic=self.state['hat_sample_time'],
                   attempt_seq=self.state['hat_attempt_seq'])

    def poll_once(self, device):
        if self.stop_requested:
            return False
        self.state['hat_attempt_seq'] += 1
        self.state['hat_attempt_time'] = self.monotonic()
        self.state['hat_attempt_finished_time'] = None
        self.state['sensor_status'] = 'reading'
        self.publish()
        try:
            value = device.control.get_chip_temperature()
            raw = (value.ts0_temperature, value.ts1_temperature)
            if any(isinstance(x, bool) for x in raw):
                raise ValueError('temperature sensor returned a boolean')
            temperatures = tuple(float(x) for x in raw)
            if any(not math.isfinite(x) or not -40 < x < 150 for x in temperatures):
                raise ValueError('temperature sensor returned a nonfinite or implausible value')
        except Exception as exc:
            self.fail(exc)
            return False
        finished = self.monotonic()
        self.state.update(hat_ts0_c=temperatures[0], hat_ts1_c=temperatures[1],
                          hat_sample_time=finished,
                          hat_sample_seq=self.state['hat_sample_seq']+1,
                          hat_attempt_finished_time=finished,
                          hat_error='', hat_consecutive_errors=0,
                          sensor_status='ready')
        self.publish()
        self.event('read_success', hat_ts0_c=temperatures[0], hat_ts1_c=temperatures[1],
                   sample_monotonic=finished, sample_seq=self.state['hat_sample_seq'],
                   attempt_seq=self.state['hat_attempt_seq'])
        return True

    def run(self, device_factory):
        device = None
        code = 0
        self.publish()
        self.event('sensor_start', interval_seconds=self.interval)
        try:
            if self.stop_requested:
                return 0
            try:
                device = device_factory()
            except Exception as exc:
                self.fail(exc, 'initialization_error')
                return 2
            while not self.stop_requested:
                self.poll_once(device)
                # Wait after completion: a timeout never causes catch-up polling.
                next_attempt = self.monotonic()+self.interval
                while not self.stop_requested and self.monotonic() < next_attempt:
                    time.sleep(min(.25, max(0.0, next_attempt-self.monotonic())))
        finally:
            self.state['sensor_status'] = 'stopping'
            self.publish()
            self.event('sensor_stopping', signal=self.stop_signal)
            if device is not None:
                try:
                    device.release()
                except Exception as exc:
                    code = 1
                    self.event('release_error', error=type(exc).__name__+': '+str(exc)[:2000])
            self.state['sensor_status'] = 'stopped'
            self.publish()
            self.event('sensor_stopped', signal=self.stop_signal)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--interval', type=float, default=3.0)
    args = parser.parse_args()
    sensor = Sensor(args.state, args.events, args.interval)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, sensor.request_stop)

    def device_factory():
        # Import inside the child and only when run; importing this module is safe
        # for tests and the parent must not construct a Device itself.
        from hailo_platform import Device
        return Device()

    try:
        return sensor.run(device_factory)
    except Exception as exc:
        print('HAT sensor process failed: '+type(exc).__name__+': '+str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
