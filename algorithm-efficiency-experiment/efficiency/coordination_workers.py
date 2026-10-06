"""Identical persistent worker protocol over thread queues or spawn-process queues."""
from contextlib import ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np

from .attention_native import Native, errors
from .common import digest_file
from .feedback_spec import input_hash


def owner():
    return dict(pid=os.getpid(), tid=threading.get_native_id())


class NumericWorker:
    def __init__(self, directory, config, stack):
        self.directory = Path(directory)
        self.owner = owner()
        self.cpu = stack.enter_context(Native(self.directory/'native-build'))
        self.gpu = stack.enter_context(Native(self.directory/'native-build', gpu=True))
        self.metadata = {r['fixture_id']:r for r in json.loads((self.directory/'fixtures.json').read_text())}
        self.data, self.reference_outputs = {}, {}
        self.environment = dict(cpu_initialization_ms=self.cpu.initialization_ms,
            gpu_initialization_ms=self.gpu.initialization_ms, device=self.gpu.identity)

    def load(self, ids):
        self.data.clear(); self.reference_outputs.clear()
        for fid in ids:
            path = self.directory/'fixtures'/(fid+'.npz')
            if digest_file(path) != self.metadata[fid]['file_sha256']:
                raise ValueError('Fixture checksum differs')
            with np.load(path, allow_pickle=False) as f:
                self.data[fid] = (f['x'], f['w'], f['expected'])
        return dict(fixture_ids=ids)

    def request(self, fid, backend):
        x, w, expected = self.data[fid]
        native = self.cpu if backend == 'native4' else self.gpu
        setup = time.monotonic()
        if native.shape != x.shape:
            native.configure(x.shape)
        output = np.empty_like(x)
        started = time.monotonic()
        metrics = native.run(x, w, output, 'prefill', backend)
        computed = time.monotonic()
        check = errors(output, expected)
        row = dict(fixture_id=fid, backend=backend, started=started, computed=computed,
            setup_ms=(started-setup)*1000, **metrics, **check,
            output_sha256=input_hash(output))
        row['validated'] = time.monotonic()
        if not check['correct'] or metrics['validation_errors']:
            raise RuntimeError(f'Numerical failure: {row}')
        return row, output

    def perform(self, task):
        if owner() != self.owner:
            raise RuntimeError('Numerical context used by a different owner')
        if task['operation']=='load':
            return self.load(task['fixture_ids'])
        if task['operation']=='warm':
            results = []
            for backend in ('native4', 'C6'):
                for fid in task['fixture_ids']:
                    row, output = self.request(fid, backend)
                    relative = f'outputs/{task["block_id"]}-{backend}-{fid}.npy'
                    np.save(self.directory/relative, output, allow_pickle=False)
                    self.reference_outputs[(fid, backend)] = (row['output_sha256'], relative)
                    results.append(dict(**row, output_file=relative))
            return dict(requests=results)
        if task['operation']!='numeric':
            raise ValueError('Unknown numerical operation')
        results = []
        for i in range(task['batches']):
            if time.monotonic() >= task['deadline']:
                raise TimeoutError('Numerical job deadline reached')
            fid = task['fixture_ids'][i % len(task['fixture_ids'])]
            row, output = self.request(fid, task['backend'])
            expected_hash, relative = self.reference_outputs[(fid, task['backend'])]
            row['matches_warmup'] = row['output_sha256']==expected_hash
            if not row['matches_warmup']:
                relative = f'outputs/{task["job_id"]}-{i}.npy'
                np.save(self.directory/relative, output, allow_pickle=False)
            row['output_file'] = relative
            results.append(row)
        return dict(requests=results, correct=all(r['correct'] for r in results))


class HatWorker:
    def __init__(self, directory, config, stack):
        from hailo_platform import VDevice, __version__
        from hailo_platform.pyhailort.pyhailort import LLM
        from .feedback_protocol import Adviser
        from .state_isolation import parameter_settings, resolve_defaults
        self.owner = owner()
        start = time.monotonic()
        device = stack.enter_context(VDevice())
        llm = stack.enter_context(LLM(device, config['model']))
        self.environment = dict(hailort_version=__version__, prompt_template=llm.prompt_template(),
            stop_tokens=llm.get_stop_tokens(), capacity_tokens=llm.max_context_capacity(),
            model_defaults=resolve_defaults(llm, __version__), loading_seconds=time.monotonic()-start)
        self.environment['parameters'] = parameter_settings(self.environment['model_defaults'], 'penalty_1_0')[0]
        self.environment['experiment_limit_tokens'] = min(1792, self.environment['capacity_tokens']-256)
        self.adviser = Adviser(llm, self.environment)

    def perform(self, task):
        if owner() != self.owner:
            raise RuntimeError('HAT context used by a different owner')
        return self.adviser.propose(task['case']['messages'], task['case']['eligible'])


def worker_loop(kind, incoming, outgoing, directory, config, factory=None):
    """Create, use and destroy every context on the same native thread."""
    identity = owner()
    try:
        with ExitStack() as stack:
            started = time.monotonic()
            worker = (factory or (NumericWorker if kind=='numeric' else HatWorker))(directory, config, stack)
            outgoing.put(dict(kind='ready', owner=identity, started=started,
                              ended=time.monotonic(), environment=worker.environment))
            while True:
                task = incoming.get()
                if task is None:
                    break
                received = time.monotonic()
                result = worker.perform(task)
                ended = time.monotonic()
                outgoing.put(dict(kind='result', job_id=task['job_id'], owner=identity,
                    submitted=task['submitted'], received=received, ended=ended,
                    reply_sent=time.monotonic(), result=result))
        outgoing.put(dict(kind='closed', owner=identity, ended=time.monotonic()))
    except BaseException as exc:
        outgoing.put(dict(kind='error', owner=identity, error=f'{type(exc).__name__}: {exc}',
                          traceback=traceback.format_exc()))


class Worker:
    def __init__(self, architecture, kind, directory, config, check, factory=None, environment=None):
        self.check, self.serial, self.closed, self.pending = check, 0, False, False
        self.prefix = config.get('block_id', 'test')
        self.kind, self.architecture = kind, architecture
        if architecture=='thread':
            self.incoming, self.outgoing = queue.Queue(), queue.Queue()
            self.handle = threading.Thread(target=worker_loop,
                args=(kind,self.incoming,self.outgoing,str(directory),config,factory), daemon=True)
        elif architecture=='process':
            context = mp.get_context('spawn')
            self.incoming, self.outgoing = context.Queue(), context.Queue()
            self.handle = context.Process(target=worker_loop,
                args=(kind,self.incoming,self.outgoing,str(directory),config,factory))
        else:
            raise ValueError('Unknown worker architecture')
        if environment is not None and architecture!='process':
            raise ValueError('Environment isolation requires a fresh process')
        previous={k:os.environ.get(k) for k in (environment or {})}
        try:
            for k,v in (environment or {}).items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
            # spawn snapshots this environment before the child imports numerical libraries.
            self.handle.start()
        finally:
            for k,v in previous.items():
                if v is None:os.environ.pop(k,None)
                else:os.environ[k]=v
        try:
            self.ready = self.receive('ready')
        except BaseException:
            self.close()
            raise

    def receive(self, expected):
        while True:
            self.check()
            try:
                row = self.outgoing.get(timeout=.1)
            except queue.Empty:
                if not self.handle.is_alive():
                    raise RuntimeError(f'{self.kind} worker exited without a result')
                continue
            row['delivered'] = time.monotonic()
            if row['kind']=='error':
                raise RuntimeError(row['error']+'\n'+row['traceback'])
            if row['kind'] != expected:
                raise RuntimeError(f'Unexpected worker message: {row["kind"]}')
            return row

    def submit(self, operation, **payload):
        if self.pending or self.closed:
            raise RuntimeError('Worker already busy or closed')
        self.check(); self.serial += 1
        job_id = f'{self.prefix}-{self.kind}-{self.serial}'
        task = dict(operation=operation, job_id=job_id, submitted=time.monotonic(), **payload)
        self.incoming.put(task); self.pending = job_id
        return task

    def result(self):
        if not self.pending:
            raise RuntimeError('No pending job')
        row = self.receive('result')
        if row['job_id'] != self.pending or row['owner'] != self.ready['owner']:
            raise RuntimeError('Worker response identity mismatch')
        self.pending = False
        return row

    def call(self, operation, **payload):
        self.submit(operation, **payload)
        return self.result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.incoming.put(None)
        self.handle.join(timeout=5)
        forced = self.handle.is_alive()
        if forced and self.architecture=='process':
            self.handle.terminate(); self.handle.join(timeout=2)
            if self.handle.is_alive():
                self.handle.kill(); self.handle.join(timeout=2)
        if self.architecture=='process':
            for channel in (self.incoming, self.outgoing):
                channel.close(); channel.cancel_join_thread()
        self.release = dict(owner=self.ready['owner'] if hasattr(self,'ready') else None,
            architecture=self.architecture, worker=self.kind, forced=forced,
            alive=self.handle.is_alive(), ended=time.monotonic())
        if self.handle.is_alive():
            raise RuntimeError(f'{self.kind} thread failed to stop; supervisor must release worker process')


def run_condition(numeric, hat, condition, fixture_ids, case, deadline, batches=24):
    started = time.monotonic()
    numeric_task = numeric.submit('numeric', backend='native4' if condition.startswith('cpu') else 'C6',
        fixture_ids=fixture_ids, batches=batches, deadline=deadline)
    if condition.endswith('serial'):
        numerical = numeric.result()
        advice_task = hat.submit('advice', case=case)
        advice = hat.result()
    else:
        advice_task = hat.submit('advice', case=case)
        numerical = numeric.result()
        advice = hat.result()
    ended = time.monotonic()
    return dict(started=started, ended=ended, total_ms=(ended-started)*1000,
        numeric_submitted=numeric_task['submitted'], advice_submitted=advice_task['submitted'],
        numerical=numerical, adviser=advice)
