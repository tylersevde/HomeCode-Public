import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile
import hashlib
import psutil

from .common import (ROOT, MODEL, ARCHIVE_SHA, atomic_json, digest_file, emit,
                     profile, stop_process_group, utc)
from .monitor import sample, safety_reason, command


def worker_tree_sample(worker):
    """Sample coordinator and living descendants; RSS may count shared pages twice."""
    if worker is None:return dict(worker_tree_rss_bytes=None,worker_tree_cpu_seconds=None,worker_tree_processes=0)
    try:
        parent=psutil.Process(worker.pid);processes=[parent,*parent.children(recursive=True)]
    except (psutil.NoSuchProcess,psutil.AccessDenied):processes=[]
    rss=0;cpu=0;count=0
    for process in processes:
        try:
            memory=process.memory_info();times=process.cpu_times()
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        rss+=memory.rss;cpu+=times.user+times.system;count+=1
    return dict(worker_tree_rss_bytes=rss if count else None,
                worker_tree_cpu_seconds=cpu if count else None,worker_tree_processes=count)


def inventory(phase='all', model=MODEL):
    sources = [ROOT / 'experiment.py', *sorted((ROOT / 'efficiency').glob('*.py')),
               *sorted((ROOT / 'tests').glob('*.py')), ROOT / 'vendor/hat_sensor.py',
               ROOT / 'reference/algorithm_efficiency_demo.py',
               *sorted((ROOT / 'native/attention').glob('*.cpp')),
               *sorted((ROOT / 'native/attention').glob('*.comp'))]
    archive = ROOT / 'reference/algorithm_efficiency_experiment.zip'
    if digest_file(archive) != ARCHIVE_SHA:
        raise RuntimeError('Reference ZIP checksum differs from the studied archive')
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if digest_file(ROOT / 'reference' / name) != hashlib.sha256(z.read(name)).hexdigest():
                raise RuntimeError(f'Extracted reference differs from original ZIP: {name}')
    versions = {}
    for name in ('numpy', 'Jinja2', 'psutil'):
        versions[name] = importlib.metadata.version(name)
    value = dict(schema_version=1, created_utc=utc(), platform=platform.platform(),
        python=platform.python_version(), cpu_model=Path('/proc/device-tree/model').read_bytes().rstrip(b'\x00').decode(),
        versions=versions, archive_sha256=ARCHIVE_SHA,
        blas_library=str(Path('/usr/lib/aarch64-linux-gnu/libblas.so.3').resolve()),
        cpu_governor=Path('/sys/devices/system/cpu/cpufreq/policy0/scaling_governor').read_text().strip(),
        initial_throttle_flags=int(command(['vcgencmd', 'get_throttled']).split('=')[1], 16),
        source_sha256={str(p.relative_to(ROOT)): digest_file(p) for p in sources},
        free_ssd_bytes=shutil.disk_usage(ROOT).free)
    if phase in ('hat', 'all'):
        if not Path('/dev/hailo0').exists():
            raise RuntimeError('/dev/hailo0 is unavailable; HAT measurements cannot run')
        if not Path(model).is_file():
            raise RuntimeError(f'Existing model is missing: {model}')
        if shutil.disk_usage(ROOT).free < 1024**3:
            raise RuntimeError('At least 1 GiB free SSD space is required')
        value.update(model_path=str(model), model_bytes=Path(model).stat().st_size,
                     model_sha256=digest_file(model), hailort_cli=command(['hailortcli', '--version']),
                     pcie_link_speed=Path('/sys/bus/pci/devices/0001:01:00.0/current_link_speed').read_text().strip(),
                     pcie_link_width=Path('/sys/bus/pci/devices/0001:01:00.0/current_link_width').read_text().strip())
    return value


class Supervisor:
    def __init__(self, directory, config):
        self.directory = directory
        self.config = config
        self.started = time.monotonic()
        self.deadline = self.started + config['max_seconds']
        self.sensor = None
        self.worker = None
        self.stop_reason = None
        self.phase = 'preflight'
        self.initial_flags = None
        self.last_sample = 0
        self.logs = []
        self.phase_results = []
        self.sample_count = 0

    def request_stop(self, signum, frame):
        self.stop_reason = f'Interrupted by signal {signum}'

    def record(self, require_hat=True):
        if self.config.get('governor_socket'):
            from .refine_governor import request
            try:request(self.config['governor_socket'])
            except Exception as exc:self.stop_reason=self.stop_reason or f'Governor lease failed: {exc}'
        row = sample(self.directory, self.sensor, self.worker)
        if self.config.get('profile') in ('research','quality','study','refine','reliability','cpu-compare','completion'):row.update(worker_tree_sample(self.worker))
        if self.config.get('profile') in ('feedback-attention', 'coordination', 'research', 'study', 'refine', 'reliability', 'completion'):
            try:
                row['gpu_clock_mhz'] = int(command(['vcgencmd', 'measure_clock', 'v3d']).split('=')[1])/1e6
            except (OSError, ValueError, subprocess.SubprocessError):
                row['gpu_clock_mhz'] = None
        if self.initial_flags is None:
            self.initial_flags = row['throttle_flags'] or 0
        reason = safety_reason(row, self.initial_flags, require_hat and self.sensor is not None)
        if reason and self.stop_reason is None:
            self.stop_reason = reason
        if time.monotonic() >= self.deadline:
            self.stop_reason = self.stop_reason or 'Experiment execution deadline reached'
        if (self.directory / 'STOP').exists():
            self.stop_reason = self.stop_reason or 'Operator STOP file detected'
        row.update(phase=self.phase, stop_reason=self.stop_reason, elapsed_seconds=time.monotonic()-self.started)
        emit(self.directory / 'telemetry.jsonl', 'sample', **row)
        atomic_json(self.directory / 'status.json', row)
        self.last_sample = time.monotonic()
        self.sample_count += 1
        return row

    def launch(self, args, name):
        log = (self.directory / f'{name}.log').open('w')
        self.logs.append(log)
        env = os.environ.copy()
        env.update(HAILORT_LOGGER_PATH=str(self.directory), PYTHONDONTWRITEBYTECODE='1')
        return subprocess.Popen(args, cwd=self.directory, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def pause(self, seconds):
        until = min(time.monotonic()+seconds, self.deadline)
        while time.monotonic() < until and not self.stop_reason:
            self.record()
            time.sleep(min(1, max(0, until-time.monotonic())))

    def execute(self):
        old_handlers = {s: signal.signal(s, self.request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            facts = inventory(self.config['phase'], self.config['model'])
            atomic_json(self.directory / 'manifest.json', dict(**facts, config=self.config))
            if self.config.get('profile', '').startswith('diagnostic'):
                from .diagnostic import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'state-isolation':
                from .state_isolation import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'cache-validation':
                from .cache_validation import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'hybrid-validation':
                from .hybrid import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'language-validation':
                from .language_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'language-strict-validation':
                from .strict_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'feedback-attention':
                from .feedback_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'coordination':
                from .coordination import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'adviser-validation':
                from .adviser_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'research':
                from .research_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'reliability':
                from .reliability_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'cpu-compare':
                from .cpu_compare_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'completion':
                import importlib
                from .completion_service import protocol_module
                module = importlib.import_module('efficiency.' + protocol_module(self.config['track']))
                module.prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'refine':
                from .refine_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'study':
                from .study_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            elif self.config.get('profile') == 'quality':
                from .quality_protocol import prepare_source
                prepare_source(self.directory, self.config, facts)
            for relative in facts['source_sha256']:
                target = self.directory / 'source' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            self.record(require_hat=False)
            if self.stop_reason:
                raise RuntimeError(self.stop_reason)
            if self.config['phase'] in ('hat', 'all'):
                self.sensor = self.launch([sys.executable, str(ROOT / 'vendor/hat_sensor.py'),
                    '--state', str(self.directory / 'hat-sensor-state.json'),
                    '--events', str(self.directory / 'hat-sensor-events.jsonl'), '--interval', '3'], 'hat-sensor')
                startup_deadline = time.monotonic()+15
                while time.monotonic() < startup_deadline and not self.stop_reason:
                    row = self.record(require_hat=False)
                    if row.get('hat_sample_age_s') is not None:
                        break
                    if self.sensor.poll() is not None:
                        raise RuntimeError('HAT sampler failed during startup')
                    time.sleep(.25)
                self.record()
                if self.stop_reason:
                    raise RuntimeError(self.stop_reason)
            self.phase = 'baseline'
            self.pause(self.config['baseline_seconds'])
            phases = ['cpu', 'hat'] if self.config['phase'] == 'all' else [self.config['phase']]
            for phase in phases:
                if self.stop_reason:
                    break
                self.phase = phase
                self.record()
                if self.stop_reason:
                    break
                launch_time = time.monotonic()
                emit(self.directory / 'events.jsonl', 'phase_start', phase=phase)
                self.worker = self.launch([sys.executable, str(ROOT / 'experiment.py'), '_worker',
                    '--phase', phase, '--output', str(self.directory)], phase)
                while self.worker.poll() is None:
                    now = time.monotonic()
                    if now-self.last_sample >= 1:
                        self.record()
                    try:
                        state = json.loads((self.directory / 'progress.json').read_text())
                        activity = state['monotonic'] if state['phase'] == phase else launch_time
                    except (OSError, ValueError, KeyError):
                        activity = launch_time
                    if now-activity > 180:
                        self.stop_reason = self.stop_reason or f'{phase} worker made no progress for 180 seconds'
                    if self.stop_reason:
                        stop_process_group(self.worker)
                        break
                    time.sleep(.1)
                code = self.worker.wait(timeout=5)
                self.phase_results.append(dict(phase=phase, returncode=code,
                                               elapsed_seconds=time.monotonic()-launch_time))
                emit(self.directory / 'events.jsonl', 'phase_end', **self.phase_results[-1])
                if code != 0:
                    self.stop_reason = self.stop_reason or f'{phase} worker failed with exit code {code}; inspect {phase}.log'
                self.worker = None
            if not self.stop_reason:
                self.phase = 'cooldown'
                self.pause(self.config['cooldown_seconds'])
        except Exception as exc:
            self.stop_reason = self.stop_reason or f'{type(exc).__name__}: {exc}'
        finally:
            stop_process_group(self.worker)
            self.worker = None
            stop_process_group(self.sensor)
            self.sensor = None
            self.phase = 'stopped' if self.stop_reason else 'complete'
            # Preserve final Pi state; the sensor has intentionally been stopped.
            self.record(require_hat=False)
            outcome = dict(schema_version=1, finished_utc=utc(), status=self.phase,
                stop_reason=self.stop_reason, elapsed_seconds=time.monotonic()-self.started,
                phases=self.phase_results, telemetry_samples=self.sample_count)
            atomic_json(self.directory / 'outcome.json', outcome)
            for handle in self.logs:
                handle.close()
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        print(json.dumps(outcome, indent=2), flush=True)
        return 0 if not self.stop_reason else 1


def run(args):
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(profile(args.profile), phase=args.phase, profile=args.profile,
                  max_seconds=args.max_seconds, model=str(Path(args.model).resolve()))
    if getattr(args, 'source_run', None):
        config['source_run'] = str(args.source_run.resolve())
    if getattr(args, 'replay_fixture', None):
        config.update(replay_fixture=args.replay_fixture, replay_order=args.replay_order)
    if getattr(args, 'state_case', None):
        config['state_case'] = args.state_case
    if getattr(args, 'replay_run', None):
        config['replay_run'] = str(args.replay_run.resolve())
    if getattr(args, 'resume_run', None):
        config['resume_run'] = str(args.resume_run.resolve())
        config['resume_budget'] = args.resume_budget
    if getattr(args, 'query_records', None) is not None:
        config.update(item=args.item, facts_path=str(args.facts.resolve()))
        atomic_json(directory / 'facts.json', args.query_records)
    if getattr(args, 'ask_records', None) is not None:
        config.update(question=args.question, engine=args.engine, facts_path=str(args.facts.resolve()))
        atomic_json(directory / 'facts.json', args.ask_records)
    atomic_json(directory / 'config.json', config)
    os.environ['HAILORT_LOGGER_PATH'] = str(directory)
    # Advisory lock coordinates this harness's own runs; never reset another owner's device.
    with (ROOT / 'runs/.experiment.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another experiment run holds the local experiment lock')
        return Supervisor(directory, config).execute()
