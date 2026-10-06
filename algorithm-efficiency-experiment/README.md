# Algorithm efficiency experiment — public shareable edition

By Tyler Sevde. This repository preserves a bounded attention experiment and its measured failures as well as improvements. No overall CPU, GPU or NPU candidate qualified; no inference defaults were changed.

- [README.txt](README.txt)
- [INSTRUCTIONS.txt](INSTRUCTIONS.txt)
- [CREDITS.txt](CREDITS.txt)
- [RESEARCH-PAPER.txt](RESEARCH-PAPER.txt)
- [The Codex of this AI Experiment.txt](The%20Codex%20of%20this%20AI%20Experiment.txt)
- [LICENSE-CODE.txt](LICENSE-CODE.txt)
- [LICENSE-DOCS-DATA.txt](LICENSE-DOCS-DATA.txt)
- [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt)
- [LICENSING.txt](LICENSING.txt)

## Reproduce the retained results

With Python 3 and NumPy installed, run `python3 tools/validate_public.py .`. This verifies public hashes and independently recomputes final paired-block intervals, duplicate-control stability, planned NPU denominators, raw answer scoring and actual-history mappings. Numerical fixture arrays, opaque device contexts and telemetry remain in the private archive; this is not a replacement for the original full hardware audit.

- [CPU completion-cpu-20261006T120615Z-compare](evidence/completion-cpu-20261006T120615Z-compare/summary.json)
- [GPU completion-gpu-develop-20261006T145314Z-develop](evidence/completion-gpu-develop-20261006T145314Z-develop/summary.json)
- [NPU completion-npu-diagnostic-20261006T112430Z-diagnostic](evidence/completion-npu-diagnostic-20261006T112430Z-diagnostic/summary.json)
- [NPU completion-npu-develop-20261006T155952Z-develop](evidence/completion-npu-develop-20261006T155952Z-develop/summary.json)

Read the [public manifest](PUBLICATION-MANIFEST.json) and [evidence provenance](evidence/PUBLIC-EVIDENCE-MANIFEST.json). Original seals certify original bytes, not these derivatives.

## Original CPU demonstration

The [unchanged original ZIP](original/algorithm_efficiency_experiment.zip) and its three extracted members are included. Run `demo_dir=$(mktemp -d)` followed by `python3 -B original/algorithm_efficiency_demo.py --output "$demo_dir/algorithm-efficiency-fresh.json"` for fresh CPU measurements. The preserved original results identify their original x86 environment; the author's iPad interface origin does not imply local iPad execution.

## Hardware source

The full harness was tested on Raspberry Pi 5/ARM64 with Hailo runtime 5.1.1 and the recorded native dependencies. It is not presented as a portable turnkey harness. Public adaptation `public-adaptation-v1` changes only two path defaults in `efficiency/common.py` (plus its `sys` import): `AE_MODEL_PATH` defaults to `models/hailo/v5.1.1/Qwen2.5-1.5B-Instruct.hef` below the repository; `AE_PLOT_PYTHON` defaults to the current Python executable. Llama remains derived as `MODEL.parent / 'llama3.2-3b-study/Llama-3_2-3B-Instruct.hef'`. Acquire licensed models and runtime separately; weights and third-party model licenses are not replaced by this repository's licenses. Other original hardware/runtime assumptions remain in the source.

Run `python3 -B tools/run_public_tests.py .` to discover all 589 unchanged source tests in a disposable copy. Two tests explicitly require omitted original historical archives and are skipped by this public runner: `test_refine.ScoringTests.test_actual_llama_multiturn_template` and `test_strict_language.StrictProtocolTests.test_replay_preparation_and_source_tamper_detection`. Five fake user-service lifecycle tests also skip unless `RUN_SYSTEMD_TESTS=1` is set on a Linux machine with an active user manager. The runner prints each skip and its reason. Direct discovery inside the published tree is not the public test workflow. Hardware experiments are separate actions and consume their own budgets.
