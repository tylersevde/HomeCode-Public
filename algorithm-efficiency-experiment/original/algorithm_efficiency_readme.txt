Algorithm efficiency experiment — October 1, 2026

Outcome
A bounded software optimization was implemented and measured. No deployed AI model was changed, no new algorithm was discovered, and no recursive self-improvement or singularity was achieved.

Workload and comparison
One synthetic causal attention layer, 64-dimensional float64 inputs, fixed random weights, CPU only, one requested numerical-library thread. Every variant computes the attention output at every position. The primary baseline computes only the newest query but recomputes keys and values for the entire prefix. The optimized version caches past keys and values. A less efficient full-prefix reference is also included in the raw results.

Sequence length | Uncached median ms | Cached median ms | Speedup
128 | 4.392 | 1.786 | 2.459x
256 | 14.886 | 4.391 | 3.390x
512 | 60.185 | 9.431 | 6.382x

Method and validation
Seven timed repetitions per variant and workload, randomized execution order, after warm-up. All output positions compared with the full-prefix reference. Additional checks cover three independent seeds and four shapes, including single-position and single-dimension edge cases. Relative and absolute tolerances: 1e-10.
Maximum absolute error in the timed workloads: 1.8873791418627661e-15.
The 512-position cached variant retains 524,288 bytes (0.5 MiB) of keys and values in addition to inputs, weights and outputs. This persistent memory is the tradeoff for avoiding recomputation.

Interpretation
This is a standard key-value caching demonstration with synthetic inputs, not a trained language model or an end-to-end inference benchmark. It does not establish a speedup over already optimized AI systems, improved reasoning accuracy, reduced measured energy, a discovery method, or a 5x gain at every iteration. Runtime and arithmetic-count reduction are different quantities. Research, code generation, and validation costs are not included in inference timings.

The 5^X target
With fixed tasks, quality requirements and hardware, 5^X efficiency would require each successive accepted version to use one-fifth as much compute as the preceding version: C_X = C_0 / 5^X. Four such improvements imply 625x; ten imply 9,765,625x. This is a hypothetical requirement, not a measured growth law. A one-time 5x improvement does not establish it. X as an iteration counter also says nothing about elapsed development time. The exponential 5^X stays finite for finite X; the formula does not establish a singularity.

Research found
AlphaEvolve (Google DeepMind, May 2025) combines code proposals and automated evaluation. Its announcement reports a 23% speedup in one Gemini kernel, translating into a 1% training-time reduction. This illustrates why component gains do not equal whole-model gains.
AIDE^2 (September 22, 2026 preprint) reports seven accepted agent-code improvements during an eight-day run and transfer to held-out benchmarks. It optimizes the harness around fixed foundation models. Its test of the improved agent driving further self-improvement was not decisive against the strong baseline. This is reported bounded empirical progress, not evidence of universal fivefold recursion.

Reproduce
Install or use NumPy, then run:
python algorithm_efficiency_demo.py --output algorithm_efficiency_results.json
The script uses no network, paid API, training job or downloaded model.

Sources
https://huggingface.co/docs/transformers/cache_explanation
https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
https://arxiv.org/html/2609.26457v1
https://arxiv.org/abs/2505.22954

Included files
algorithm_efficiency_demo.py — executable source
algorithm_efficiency_results.json — raw timings, correctness checks, operation counts, environment and limitations
algorithm_efficiency_readme.txt — this explanation
