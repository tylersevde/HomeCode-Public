"""Build and call the installed-toolchain CPU/Vulkan attention library."""
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .common import ROOT, atomic_json, digest_file

FLAGS = ['-std=c++17', '-O3', '-mcpu=cortex-a76', '-fopenmp', '-fPIC', '-shared', '-ffp-contract=off']


def build():
    source = ROOT/'native/attention'; output = ROOT/'build/attention'; output.mkdir(parents=True, exist_ok=True)
    inputs = {str(p.relative_to(ROOT)): digest_file(p) for p in sorted(source.iterdir()) if p.suffix in ('.cpp', '.comp')}
    identity = dict(sources=inputs, flags=FLAGS, compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
        shader_compiler=subprocess.check_output(['glslc', '--version'], text=True))
    manifest_path = output/'build.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if all(previous.get(k)==v for k, v in identity.items()) and all(
            (output/p).is_file() and digest_file(output/p)==h for p, h in previous.get('artifacts', {}).items()) and previous.get('artifacts'):
            return output
    commands = [['g++', *FLAGS, str(source/'attention.cpp'), '-o', str(output/'libattention.so'), '-lvulkan']]
    for stem in ('project', 'scores', 'apply', 'softmax'):
        for size in ((64, 128) if stem=='softmax' else (8, 16)):
            for variant in ((0,) if stem=='project' else (0,1,2)):
                suffix=f'-v{variant}' if variant else ''
                commands.append(['glslc', '--target-env=vulkan1.1', '-O', f'-DCAUSAL={variant}',
                    '-D'+('GROUP' if stem=='softmax' else 'TILE')+f'={size}',
                    str(source/f'{stem}.comp'), '-o', str(output/f'{stem}{size}{suffix}.spv')])
    for stem in ('project', 'scores', 'apply'):
        for size in (32,64):
            commands.append(['glslc','--target-env=vulkan1.1','-O',f'-DGROUP={size}',
                str(source/f'{stem}-stream.comp'),'-o',str(output/f'{stem}-stream{size}.spv')])
    with (output/'build.log').open('w') as log:
        for argv in commands:
            completed = subprocess.run(argv, text=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            if completed.returncode: raise RuntimeError(f'Native build failed; inspect {output/"build.log"}')
    identity.update(commands=commands, artifacts={p.name: digest_file(p) for p in sorted(output.iterdir()) if p.suffix in ('.so', '.spv')})
    atomic_json(manifest_path, identity)
    return output


class Metrics(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in ('staging_ms', 'gpu_ms', 'wait_ms', 'command_ms')]+[
        (name, ct.c_uint64) for name in ('submissions', 'bytes', 'validation_errors')]


class ProfileMetrics(ct.Structure):
    _fields_ = [('version',ct.c_uint32),('size',ct.c_uint32),('stages_ms',ct.c_double*4)]+[
        (name,ct.c_double) for name in ('projection_ms','input_copy_ms','output_copy_ms','submit_ms','fence_ms','query_ms')]


class Native:
    def __init__(self, directory, gpu=False, validation=False):
        self.directory = Path(directory); self.gpu = gpu; self.handle = None
        self.lib = lib = ct.CDLL(str(self.directory/'libattention.so'))
        array = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        lib.fb_create.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_char_p, ct.c_size_t]; lib.fb_create.restype = ct.c_void_p
        lib.fb_configure.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]; lib.fb_configure.restype = ct.c_int
        lib.fb_run.argtypes = [ct.c_void_p, array, array, array, ct.c_int, ct.c_int]; lib.fb_run.restype = ct.c_int
        for name in ('fb_error', 'fb_identity'):
            getattr(lib, name).argtypes = [ct.c_void_p]; getattr(lib, name).restype = ct.c_char_p
        lib.fb_metrics.argtypes = [ct.c_void_p, ct.POINTER(Metrics)]; lib.fb_metrics.restype = None
        lib.fb_destroy.argtypes = [ct.c_void_p]; lib.fb_destroy.restype = None
        self.extended = hasattr(lib,'fb_set_options') and hasattr(lib,'fb_profile_metrics')
        if self.extended:
            lib.fb_set_options.argtypes=[ct.c_void_p,ct.c_int,ct.c_int];lib.fb_set_options.restype=ct.c_int
            lib.fb_profile_metrics.argtypes=[ct.c_void_p,ct.POINTER(ProfileMetrics),ct.c_size_t];lib.fb_profile_metrics.restype=ct.c_int
        message = ct.create_string_buffer(4096)
        start = time.perf_counter()
        self.handle = lib.fb_create(os.fsencode(self.directory), gpu, validation, message, len(message))
        if not self.handle: raise RuntimeError(message.value.decode())
        self.initialization_ms = (time.perf_counter()-start)*1000
        self.identity = lib.fb_identity(self.handle).decode()
        self.shape = None

    def configure(self, shape):
        b, n, d = shape
        if self.lib.fb_configure(self.handle, n, d, b): raise RuntimeError(self.lib.fb_error(self.handle).decode())
        self.shape = tuple(shape)

    def run(self, inputs, weights, output, mode, backend, *, variant=0, profiling=False):
        if tuple(inputs.shape)!=self.shape or output.shape!=inputs.shape or weights.shape!=(3, inputs.shape[2], inputs.shape[2]):
            raise ValueError('Configured, input, weight and output shapes must agree')
        if mode not in ('stream', 'prefill'): raise ValueError('Unknown attention mode')
        if backend not in ('native1', 'native4', *(f'C{i}' for i in range(8))): raise ValueError('Unknown native backend')
        code = -1 if backend=='native1' else -4 if backend=='native4' else int(backend[1:])
        if code>=0 and not self.gpu: raise ValueError('GPU backend requires a GPU context')
        if variant not in (0,1,2,3,4) or (variant>=3 and mode!='stream') or (code<0 and (variant or profiling)):
            raise ValueError('Invalid numerical variant/profiling combination')
        if self.extended:
            if self.lib.fb_set_options(self.handle,int(profiling),variant):raise ValueError('Invalid native options')
        elif variant or profiling:raise ValueError('Archived library does not support research options')
        start = time.perf_counter(); cpu_start = time.process_time()
        if self.lib.fb_run(self.handle, inputs, weights, output, mode=='stream', code):
            raise RuntimeError(self.lib.fb_error(self.handle).decode())
        elapsed = (time.perf_counter()-start)*1000; cpu_seconds = time.process_time()-cpu_start
        metrics = Metrics(); self.lib.fb_metrics(self.handle, ct.byref(metrics))
        result=dict(request_ms=elapsed, cpu_seconds=cpu_seconds, **{k: getattr(metrics, k) for k, _ in metrics._fields_})
        if self.extended:
            extended=ProfileMetrics()
            if self.lib.fb_profile_metrics(self.handle,ct.byref(extended),ct.sizeof(extended)) or extended.version!=1 or extended.size!=ct.sizeof(extended):
                raise RuntimeError('Profiling ABI mismatch')
            result['profile']=dict(version=extended.version,enabled=bool(profiling),variant=variant,
                stages_ms=dict(zip(('projection','scores','softmax','apply'),extended.stages_ms)),
                **{k:getattr(extended,k) for k,_ in extended._fields_[3:]})
        return result

    def close(self):
        if self.handle: self.lib.fb_destroy(self.handle); self.handle = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def fixture(n, d, b, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(b, n, d)).astype(np.float32)
    w = (rng.normal(size=(3, d, d))/np.sqrt(d)).astype(np.float32)
    return x, w


def oracle(x, w):
    """FP64 oracle on exactly the quantized FP32 inputs, using the original cached layer."""
    from .cpu import reference_module
    module = reference_module()
    return np.stack([module.cached_stream(sequence.astype(np.float64), w.astype(np.float64)) for sequence in x])


def numpy_attention(x, w, mode):
    if mode=='prefill':
        q, k, v = (x@matrix for matrix in w)
        scores = (q@k.transpose(0, 2, 1))*np.float32(x.shape[-1]**-.5)
        scores[:, np.triu_indices(x.shape[1], 1)[0], np.triu_indices(x.shape[1], 1)[1]] = -np.inf
        scores -= scores.max(axis=-1, keepdims=True); np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        return scores@v
    b, n, d = x.shape; keys = np.empty_like(x); values = np.empty_like(x); result = np.empty_like(x)
    for t in range(n):
        for batch in range(b):
            q = x[batch,t]@w[0]; keys[batch,t] = x[batch,t]@w[1]; values[batch,t] = x[batch,t]@w[2]
            scores = (q@keys[batch,:t+1].T)*np.float32(d**-.5)
            scores -= scores.max(); np.exp(scores, out=scores); scores /= scores.sum()
            result[batch,t] = scores@values[batch,:t+1]
    return result


def errors(actual, expected):
    delta = np.abs(actual.astype(np.float64)-expected); bound = 1e-5+1e-4*np.abs(expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return dict(correct=finite and bool(np.all(delta<=bound)), max_absolute_error=float(delta.max()) if finite else None,
                max_tolerance_fraction=float((delta/bound).max()) if finite else None)
