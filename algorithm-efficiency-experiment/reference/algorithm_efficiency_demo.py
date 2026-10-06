#!/usr/bin/env python3
"""Bounded, reproducible algorithm-efficiency demonstration.

Measures three existing ways to evaluate one causal attention layer on a
stream of fixed synthetic inputs. This is not a trained language model,
autonomous algorithm discovery, recursive self-improvement, or a benchmark
of ChatGPT. All variants use identical float64 inputs and fixed weights.

Run: python algorithm_efficiency_demo.py --output algorithm_efficiency_results.json
Dependency: NumPy. No network, API calls, training, or model downloads.
"""

import os

# Set before importing NumPy: the same one-thread setting applies to every run.
for thread_variable in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import argparse
import contextlib
import datetime
import io
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def softmax(scores):
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def full_prefix(inputs, weights):
    """Reference: recompute all causal outputs, retain the newest each step."""
    wq, wk, wv = weights
    n, d = inputs.shape
    outputs = np.empty_like(inputs)
    future_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scale = d ** -0.5
    for t in range(1, n + 1):
        prefix = inputs[:t]
        q, k, v = prefix @ wq, prefix @ wk, prefix @ wv
        scores = (q @ k.T) * scale
        scores[future_mask[:t, :t]] = -np.inf
        outputs[t - 1] = (softmax(scores) @ v)[-1]
    return outputs


def latest_query_no_cache(inputs, weights):
    """Compute just the needed query, but still recompute earlier keys/values."""
    wq, wk, wv = weights
    n, d = inputs.shape
    outputs = np.empty_like(inputs)
    scale = d ** -0.5
    for t in range(1, n + 1):
        q = inputs[t - 1] @ wq
        k, v = inputs[:t] @ wk, inputs[:t] @ wv
        outputs[t - 1] = softmax((q @ k.T) * scale) @ v
    return outputs


def cached_stream(inputs, weights):
    """Reuse previous keys/values; only the next input is processed each step."""
    wq, wk, wv = weights
    n, d = inputs.shape
    outputs = np.empty_like(inputs)
    keys, values = np.empty_like(inputs), np.empty_like(inputs)
    scale = d ** -0.5
    for index in range(n):
        q = inputs[index] @ wq
        keys[index] = inputs[index] @ wk
        values[index] = inputs[index] @ wv
        scores = (q @ keys[:index + 1].T) * scale
        outputs[index] = softmax(scores) @ values[:index + 1]
    return outputs


VARIANTS = {
    "full_prefix_reference": full_prefix,
    "latest_query_no_cache": latest_query_no_cache,
    "cached_stream": cached_stream,
}


def fixture(n, d, seed):
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(n, d))
    weights = tuple(rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
    return inputs, weights


def correctness_checks():
    """Check all positions on independent seeds and several edge-case sizes."""
    cases = []
    for seed in (7, 29, 101):
        for n, d in ((1, 1), (2, 8), (17, 16), (64, 32)):
            inputs, weights = fixture(n, d, seed)
            expected = full_prefix(inputs, weights)
            for name, function in VARIANTS.items():
                actual = function(inputs, weights)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                cases.append({
                    "seed": seed, "n": n, "d": d, "variant": name,
                    "max_absolute_error": float(np.max(np.abs(actual - expected))),
                })
    return cases


def multiply_add_counts(n, d):
    """Analytical dense-matrix multiply-accumulates; not total FLOPs or energy.

    Excludes softmax, masks, allocation, memory movement and Python overhead.
    Both multiplies and adds together count as one MAC here.
    """
    s1 = n * (n + 1) // 2
    s2 = n * (n + 1) * (2 * n + 1) // 6
    return {
        "full_prefix_reference": 3 * d * d * s1 + 2 * d * s2,
        "latest_query_no_cache": n * d * d + 2 * d * d * s1 + 2 * d * s1,
        "cached_stream": 3 * n * d * d + 2 * d * s1,
    }


def benchmark(n, d, seed, repeats):
    inputs, weights = fixture(n, d, seed)
    reference = full_prefix(inputs, weights)
    errors, samples = {}, {name: [] for name in VARIANTS}
    for name, function in VARIANTS.items():
        actual = function(inputs, weights)  # warm up, validate every position
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
        errors[name] = float(np.max(np.abs(actual - reference)))
    rng = np.random.default_rng(seed + 10000)
    for _ in range(repeats):
        order = list(VARIANTS)
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter()
            VARIANTS[name](inputs, weights)
            samples[name].append((time.perf_counter() - start) * 1000)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    counts = multiply_add_counts(n, d)
    return {
        "sequence_length": n, "hidden_dimension": d, "seed": seed,
        "repeats": repeats, "times_ms": samples, "median_ms": medians,
        "max_absolute_error": errors,
        "speedup_cached_vs_full_prefix": medians["full_prefix_reference"] / medians["cached_stream"],
        "speedup_cached_vs_latest_query_no_cache": medians["latest_query_no_cache"] / medians["cached_stream"],
        "matrix_multiply_accumulate_counts": counts,
        "mac_reduction_cached_vs_latest_query_no_cache": counts["latest_query_no_cache"] / counts["cached_stream"],
        "persistent_kv_cache_bytes": 2 * n * d * inputs.dtype.itemsize,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("algorithm_efficiency_results.json"))
    args = parser.parse_args()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        np.show_config()
    result = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "One synthetic, single-layer causal attention computation; fixed inputs and weights; float64; CPU; one configured BLAS thread.",
        "limitations": [
            "Established optimizations against explicit uncached references, not a novel discovery.",
            "Not a trained LLM, end-to-end language generation, or a measurement of model intelligence.",
            "No recursive improvement loop, self-modification, or universal 5x-per-cycle claim.",
            "All weights and inputs are identical across variants; no changes in precision or added hardware.",
            "KV caching trades persistent storage for reduced recomputation.",
            "Timing depends on machine, numerical library and workload; no power measurement.",
            "Research, code generation and validation costs are not amortized into inference timings.",
        ],
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "blas_threads_requested": 1,
            "numpy_build_configuration": captured.getvalue(),
        },
        "correctness_tolerances": {"rtol": 1e-10, "atol": 1e-10},
        "correctness_checks": correctness_checks(),
        "benchmarks": [benchmark(n, 64, 20261001 + n, 7) for n in (128, 256, 512)],
        "sources": [
            "https://huggingface.co/docs/transformers/cache_explanation",
            "https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/",
            "https://arxiv.org/abs/2505.22954",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correctness_checks_passed": len(result["correctness_checks"]),
                      "benchmarks": result["benchmarks"]}, indent=2))


if __name__ == "__main__":
    main()
