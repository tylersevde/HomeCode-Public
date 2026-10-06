"""Evidence-gated, opt-in CPU policy with a bounded privileged governor lease."""
import argparse
import ctypes as ct
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from .common import ROOT, atomic_json, digest_file, emit, stop_process_group, utc
from .refine_governor import GOVERNOR, request
from .study_spec import CELLS

VERSION = 'cpu-policy-v1'
ENVIRONMENT = dict(OMP_DYNAMIC='FALSE', OMP_PROC_BIND='CLOSE',
                   OMP_PLACES='{0},{1},{2},{3}', OMP_WAIT_POLICY='PASSIVE',
                   GOMP_SPINCOUNT='0')
NUMERICAL_LIBRARY_THREADS = dict(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                 VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
CLEANUP_SECONDS = 120
REQUIRED_EVIDENCE = {'summary.json', 'validation.json', 'manifest.json', 'cpu-runtime.json'}


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def policy_environment(inherited=None):
    """Never alter the invoking process or inherit unmeasured OpenMP overrides."""
    result = {key: value for key, value in (os.environ if inherited is None else inherited).items()
              if not key.startswith(('OMP_', 'GOMP_'))}
    result.update(ENVIRONMENT)
    result.update(NUMERICAL_LIBRARY_THREADS)
    return result


def runtime_identity():
    """Read the native/runtime identity without loading an accelerator or binding threads."""
    native = ROOT / 'build/attention'
    artifacts = {name: digest_file(native / name)
                 for name in ('libattention.so', 'libaffinity.so', 'build.json')}
    libraries = {}
    for name in ('libgomp.so.1', 'libc.so.6'):
        path = (Path('/usr/lib/aarch64-linux-gnu') / name).resolve(strict=True)
        libraries[name] = dict(path=str(path), sha256=digest_file(path))
    return dict(version=VERSION, model=Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode(),
                machine=platform.machine(), kernel=platform.release(), python=platform.python_version(),
                python_executable=str(Path(sys.executable).resolve()),
                numpy=importlib.metadata.version('numpy'),
                numerical_library_threads=dict(NUMERICAL_LIBRARY_THREADS),
                cpu_affinity=sorted(os.sched_getaffinity(0)), libraries=libraries,
                native_artifacts=artifacts)


def _runtime_supported(runtime):
    require(runtime.get('version') == VERSION and
            runtime.get('model', '').startswith('Raspberry Pi 5') and
            runtime.get('machine') == 'aarch64' and runtime.get('cpu_affinity') == [0, 1, 2, 3],
            'Preset requires the measured Raspberry Pi 5 and all four CPU cores')
    require(set(runtime.get('native_artifacts', {})) == {'libattention.so', 'libaffinity.so', 'build.json'},
            'Native runtime identity is incomplete')
    require(set(runtime.get('libraries', {})) == {'libgomp.so.1', 'libc.so.6'},
            'System runtime identity is incomplete')
    require(runtime.get('numerical_library_threads') == NUMERICAL_LIBRARY_THREADS,
            'Numerical library thread limits differ from the comparison')


def _audit(directory):
    # Import lazily: reading a preset or command help never opens a device context.
    from .cpu_compare_audit import audit
    return audit(directory, require_seal=True)


def _qualified_evidence(directory, expected_seal=None):
    directory = Path(directory).resolve(strict=True)
    seal = directory / 'checksums.json'
    sealed_hash = digest_file(seal)
    if expected_seal is not None:
        require(sealed_hash == expected_seal, 'Comparison evidence seal changed')
    checks = read(seal)
    require(isinstance(checks, dict) and REQUIRED_EVIDENCE <= checks.keys(),
            'Sealed comparison evidence is incomplete')
    for relative, expected in checks.items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory) and path.is_file() and digest_file(path) == expected,
                f'Comparison evidence checksum differs: {relative}')
    summary, validation = read(directory / 'summary.json'), read(directory / 'validation.json')
    require(summary.get('complete') is True and summary.get('accepted') is True and
            validation.get('passed') is True, 'Comparison did not qualify a complete accepted preset')
    audited = _audit(directory)
    require(audited.get('passed') is True, 'Independent sealed comparison audit failed')
    require(digest_file(seal) == sealed_hash, 'Comparison evidence changed during audit')
    runtime = read(directory / 'cpu-runtime.json')
    _runtime_supported(runtime)
    sources = read(directory / 'manifest.json').get('source_sha256')
    require(isinstance(sources, dict) and sources and 'efficiency/cpu_policy.py' in sources,
            'Comparison source identity is incomplete')
    return directory, sealed_hash, sources, runtime


def export_preset(run_directory, output):
    """Export only an independently audited, sealed, accepted comparison; return its JSON value."""
    directory, seal, sources, runtime = _qualified_evidence(run_directory)
    output = Path(output).resolve()
    require(not output.is_relative_to(directory), 'Preset output must be outside sealed comparison evidence')
    require(not output.exists(), 'Preset output already exists')
    preset = dict(version=VERSION, policy_id=5, governor='performance', environment=dict(ENVIRONMENT),
                  supported_cells=CELLS, source_sha256=sources, runtime=runtime,
                  evidence=dict(path=str(directory), checksums_sha256=seal),
                  scope='Measured native streaming attention only; no speedup claim for other commands.')
    atomic_json(output, preset)
    return preset


def validate_preset(path, *, check_runtime=True):
    preset = read(path)
    require(preset.get('version') == VERSION and preset.get('policy_id') == 5 and
            preset.get('governor') == 'performance' and preset.get('environment') == ENVIRONMENT and
            preset.get('supported_cells') == CELLS, 'Preset settings differ from measured policy 5')
    evidence = preset.get('evidence', {})
    require(isinstance(evidence.get('path'), str) and Path(evidence['path']).is_absolute() and
            isinstance(evidence.get('checksums_sha256'), str), 'Preset evidence reference is missing')
    _, _, sources, runtime = _qualified_evidence(evidence['path'], evidence['checksums_sha256'])
    require(preset.get('source_sha256') == sources and preset.get('runtime') == runtime,
            'Preset identity differs from sealed comparison evidence')
    if check_runtime:
        for relative, expected in sources.items():
            current = (ROOT / relative).resolve()
            require(current.is_relative_to(ROOT) and current.is_file() and digest_file(current) == expected,
                    f'Current source differs from qualified comparison: {relative}')
        require(not any(os.environ.get(key) for key in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT')),
                'Runtime loader overrides are outside the measured configuration')
        require(runtime_identity() == runtime, 'Current native runtime differs from qualified comparison')
    return preset


def probe_affinity():
    """Invoked only in an unprivileged child, so probing cannot bind the controller."""
    require({key: value for key, value in os.environ.items() if key.startswith(('OMP_', 'GOMP_'))}
            == ENVIRONMENT, 'Probe OpenMP environment differs')
    require(all(os.environ.get(key) == value for key, value in NUMERICAL_LIBRARY_THREADS.items()),
            'Probe numerical library thread limits differ')
    library = ct.CDLL(str(ROOT / 'build/attention/libaffinity.so'))
    library.refine_probe.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
    library.refine_probe.restype = ct.c_int
    probes = []
    for requested in (1, 4):
        values = (ct.c_int * 16)()
        team = library.refine_probe(requested, values)
        require(team == requested, 'Actual OpenMP team differs')
        threads = [dict(tid=values[i * 4], cpu=values[i * 4 + 1],
                        mask=values[i * 4 + 2], place=values[i * 4 + 3]) for i in range(team)]
        require([row['mask'] for row in threads] == [1 << i for i in range(requested)] and
                [row['tid'] for row in threads] == list(range(requested)), 'Actual CPU affinity differs')
        probes.append(dict(requested=requested, team=team, threads=threads))
    expected = Path('/usr/lib/aarch64-linux-gnu/libgomp.so.1').resolve()
    loaded = {str(Path(line.split()[-1]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines()
              if '/libgomp.so' in line}
    require(loaded == {str(expected)}, 'Loaded OpenMP library differs from recorded runtime')
    return probes


def _probe(environment, timeout):
    completed = subprocess.run([sys.executable, '-B', '-m', 'efficiency.cpu_policy', '_probe'],
                               cwd=ROOT, env=environment, text=True, capture_output=True,
                               timeout=timeout, check=True)
    return json.loads(completed.stdout)


def _exit_code(returncode):
    return returncode if returncode >= 0 else 128 - returncode


def run(preset_path, max_seconds, command):
    """Run an explicit unprivileged argv, returning its status (124 for the work deadline)."""
    require(isinstance(max_seconds, (float, int)) and not isinstance(max_seconds, bool) and
            math.isfinite(max_seconds) and CLEANUP_SECONDS < max_seconds <= 14400,
            'max-seconds must be greater than 120 and at most 14400, including cleanup')
    require(bool(command) and all(isinstance(arg, str) and '\0' not in arg for arg in command),
            'An explicit command is required after --')
    require(os.geteuid() != 0, 'Run the launcher as an unprivileged user; only the governor helper is elevated')
    started = time.monotonic()
    deadline, work_deadline = started + max_seconds, started + max_seconds - CLEANUP_SECONDS
    preset = validate_preset(preset_path)
    environment = policy_environment()
    logs = ROOT / 'runs' / ('cpu-policy-' + uuid.uuid4().hex)
    logs.mkdir(parents=True)
    events = logs / 'policy.jsonl'
    print(f'CPU policy log: {logs}', file=sys.stderr, flush=True)
    emit(events, 'start', preset=str(Path(preset_path).resolve()), command=command,
         deadline=deadline, work_deadline=work_deadline, evidence=preset['evidence'])
    stopped = []
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def stop(signum, _frame):
        if not stopped:
            stopped.append(signum)

    for sig in previous_handlers:
        signal.signal(sig, stop)
    try:
        with (ROOT / 'runs/.experiment.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            original = GOVERNOR.read_text().strip()
            helper = child = None
            socket_path = None
            returncode = 1
            restored = False
            # Keep helper logs after temporary socket cleanup for restoration diagnostics.
            with tempfile.TemporaryDirectory(prefix='cpu-policy-governor-') as temporary, \
                    (logs / 'governor-helper.log').open('w') as helper_log:
                try:
                    remaining = work_deadline - time.monotonic()
                    if remaining <= 0:
                        returncode = 124
                    elif stopped:
                        returncode = 128 + stopped[0]
                    else:
                        probes = _probe(environment, min(15, remaining))
                        emit(events, 'affinity_verified', probes=probes)
                        if stopped or time.monotonic() >= work_deadline:
                            returncode = 128 + stopped[0] if stopped else 124
                        else:
                            socket_path = Path(temporary) / 'lease.sock'
                            helper = subprocess.Popen(['pkexec', sys.executable, str(ROOT / 'efficiency/refine_governor.py'),
                                                       '--socket', str(socket_path), '--pid', str(os.getpid()),
                                                       '--seconds', str(deadline - time.monotonic())],
                                                      stdout=helper_log, stderr=subprocess.STDOUT,
                                                      start_new_session=True)
                            authentication_deadline = min(work_deadline, time.monotonic() + 120)
                            while not socket_path.exists() and helper.poll() is None and not stopped and \
                                    time.monotonic() < authentication_deadline:
                                time.sleep(.1)
                            if stopped:
                                returncode = 128 + stopped[0]
                            elif time.monotonic() >= work_deadline:
                                returncode = 124
                            else:
                                require(socket_path.exists(), 'Governor administrator authentication unavailable')
                                lease = request(socket_path)
                                require(lease['original'] == original, 'Governor changed before lease acquisition')
                                active = request(socket_path, 'performance')
                                require(active['current'] == 'performance', 'Requested governor is not active')
                                emit(events, 'governor_applied', **active)
                                child = subprocess.Popen(command, env=environment, shell=False, start_new_session=True)
                                heartbeat = time.monotonic()
                                while child.poll() is None and not stopped and time.monotonic() < work_deadline:
                                    if time.monotonic() - heartbeat >= 5:
                                        state = request(socket_path)
                                        require(state['current'] == 'performance', 'Governor changed during command')
                                        heartbeat = time.monotonic()
                                    time.sleep(min(.1, max(0, work_deadline - time.monotonic())))
                                returncode = (128 + stopped[0] if stopped else
                                              124 if child.poll() is None else _exit_code(child.returncode))
                finally:
                    cleanup_errors = []
                    if child is not None:
                        try:
                            stop_process_group(child, grace=min(5, max(0, deadline - time.monotonic() - 10)))
                        except Exception as exc:
                            cleanup_errors.append(f'Command cleanup: {exc}')
                    if socket_path is not None and socket_path.exists():
                        try:
                            request(socket_path, close=True)
                        except Exception as exc:
                            cleanup_errors.append(f'Governor close: {exc}')
                    if helper is not None:
                        # No SIGKILL of a privileged governor owner: its watchdog must restore first.
                        if socket_path is None or not socket_path.exists():
                            try:
                                helper.terminate()
                            except (PermissionError, ProcessLookupError):
                                pass
                        while helper.poll() is None and time.monotonic() < deadline:
                            time.sleep(min(.1, max(0, deadline - time.monotonic())))
                    current = GOVERNOR.read_text().strip()
                    restored = current == original and (helper is None or helper.poll() is not None)
                    atomic_json(logs / 'result.json', dict(returncode=returncode, command=command,
                                original_governor=original, current_governor=current, restored=restored,
                                cleanup_errors=cleanup_errors, elapsed_seconds=time.monotonic() - started,
                                utc=utc(), evidence=preset['evidence']))
                    emit(events, 'cleanup', restored=restored, errors=cleanup_errors, returncode=returncode)
                    if not restored or cleanup_errors:
                        raise RuntimeError(f'CPU policy cleanup could not be verified; inspect {logs}')
            return returncode
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ['_probe']:
        print(json.dumps(probe_affinity()))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    execute = commands.add_parser('run', help='Temporarily apply an audited CPU preset')
    execute.add_argument('--preset', type=Path, required=True)
    execute.add_argument('--max-seconds', type=float, required=True,
                         help='Total launch allowance (120 < N <= 14400), including 120 seconds for cleanup')
    execute.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command or args.command[0] != '--' or len(args.command) < 2:
        parser.error('Provide an explicit command after --')
    try:
        return run(args.preset, args.max_seconds, args.command[1:])
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'CPU policy failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
