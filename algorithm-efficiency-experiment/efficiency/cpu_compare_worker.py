"""CPU-only owners for default-environment versus frozen policy 5."""
import ctypes as ct
import os
import time

from .research_workers import Numeric
from .coordination_workers import owner
from .refine_worker import sample
from .cpu_compare_spec import environment, policy, BATCH_CALLS
from .cpu_policy import NUMERICAL_LIBRARY_THREADS


class CompareNumeric(Numeric):
    def __init__(self, directory, config, stack):
        self.configuration = config['configuration']
        self.policy = policy(self.configuration)
        desired = environment(self.configuration)
        actual = {k:v for k,v in os.environ.items() if k.startswith(('OMP_', 'GOMP_'))}
        expected = {k:v for k,v in desired.items() if v is not None}
        if actual != expected:
            raise RuntimeError('OpenMP launch environment differs from frozen configuration')
        numerical_threads = {k:os.environ.get(k) for k in NUMERICAL_LIBRARY_THREADS}
        if numerical_threads != NUMERICAL_LIBRARY_THREADS:
            raise RuntimeError('Numerical-library thread limits differ')
        super().__init__(directory, dict(config, numeric_device='cpu'), stack)
        lib = ct.CDLL(str(self.directory / 'native-build/libaffinity.so'))
        lib.refine_probe.argtypes = [ct.c_int, ct.POINTER(ct.c_int)]
        lib.refine_probe.restype = ct.c_int
        probes = []
        for size in (1, 4):
            values = (ct.c_int * 16)()
            team = lib.refine_probe(size, values)
            if team != size:
                raise RuntimeError('Actual OpenMP team differs')
            rows = [dict(tid=values[i*4], cpu=values[i*4+1], mask=values[i*4+2],
                         place=values[i*4+3]) for i in range(team)]
            masks = [1 << i for i in range(size)] if self.configuration == 'candidate' else [15] * size
            if [r['mask'] for r in rows] != masks:
                raise RuntimeError('Actual OpenMP affinity differs')
            probes.append(dict(requested=size, team=team, threads=rows))
        observed = sample()
        if observed['governor'] != self.policy['governor']:
            raise RuntimeError('Requested governor is not active')
        self.environment.update(configuration=self.configuration, policy=self.policy,
                                openmp_environment=desired, openmp_actual=actual,
                                numerical_library_threads=numerical_threads,
                                probes=probes, observed=observed)

    def check_governor(self, before, after):
        if any(s['governor'] != self.policy['governor'] for s in (before, after)):
            raise RuntimeError('Governor drift')

    def measure(self, *args, **kwargs):
        t0 = time.monotonic(); before = sample(); t1 = time.monotonic()
        result = Numeric.measure(self, *args, **kwargs)
        t2 = time.monotonic(); after = sample(); t3 = time.monotonic()
        self.check_governor(before, after)
        result.update(policy=self.policy, configuration=self.configuration,
                      system_before=before, system_after=after,
                      observer_ms=((t1-t0)+(t3-t2))*1000)
        return result

    def perform(self, task):
        if owner() != self.owner:
            raise RuntimeError('CPU worker ownership changed')
        if task['operation'] != 'batch':
            return super().perform(task)
        if task['request_ids'] != list(range(BATCH_CALLS)):
            raise ValueError('Batch must contain exactly sixteen ordered calls')
        observation_start = time.monotonic(); before = sample(); started = time.monotonic()
        rows = []
        for request_id in task['request_ids']:
            if time.monotonic() >= task['deadline']:
                raise TimeoutError('CPU batch deadline')
            rows.append(dict(request_id=request_id,
                             **Numeric.measure(self, task['fixture_id'], task['arm'])))
        observation_end = time.monotonic(); after = sample(); ended = time.monotonic()
        self.check_governor(before, after)
        return dict(requests=rows, policy=self.policy, configuration=self.configuration,
                    system_before=before, system_after=after,
                    worker_batch_ms=(ended-started)*1000,
                    observer_ms=((started-observation_start)+(ended-observation_end))*1000,
                    correct=all(r['correct'] and r['matches_warmup'] and
                                not r['validation_errors'] for r in rows))
