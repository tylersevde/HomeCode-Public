ALGORITHM EFFICIENCY EXPERIMENT
Publication guide — 6 October 2026
Owner: Tyler Sevde

This project follows a small, established key-value caching demonstration into
bounded experiments on a Raspberry Pi 5, its Vulkan GPU, and a Hailo AI HAT+ 2.
It contains executable code, experimental protocols, measurements, failed
attempts, audits and a record of why each proposed change was accepted or closed.

The completed roadmap qualified no new policy for promotion. No CPU preset was
exported, no GPU route was promoted, and application defaults were unchanged.
Some component improvements were measured; the required whole-task and quality
conditions still mattered. This is a completed investigation with negative and
conditional results, not a demonstration of recursively improving intelligence.

START HERE

  INSTRUCTIONS.txt
    Start with the portable CPU-only demonstration; then read the separate
    instructions for public evidence reproduction and the advanced Pi harness.

  RESEARCH-PAPER.txt
    Research-style account of the question, experimental sequence, methods,
    results and limitations. The recorded evidence remains the authority.

  COMPLETION-RESULTS.txt
    Final scientific outcomes, exact cumulative accounting and the verified
    pre-publication USB checkpoint. Its historical private paths may refer to
    material that is deliberately absent from the public edition.

  CREDITS.txt
    Human ownership, the owner's AI-tool attributions, upstream components,
    original archive identity and limits of those attributions.

  The Codex of this AI Experiment.txt
    A practical reflection on building an experiment that can reject its own
    proposed improvements and leave useful evidence for the next reader.

  LICENSING.txt, LICENSE-CODE.txt, LICENSE-DOCS-DATA.txt,
  THIRD-PARTY-NOTICES.txt
    Scope of the project's MIT code and CC BY 4.0 documentation/data licensing,
    with separate terms for third-party models, libraries and other materials.

  PUBLICATION-MANIFEST.json and PUBLICATION-STATUS.json
    Edition contents, provenance and the applicable publication/verification
    state. A target repository or planned copy is not a verified publication.
    The public edition may carry a public-scoped status record rather than
    private operational details.

  The private archive's historical README.md and EXPERIMENT-NOTES.md
    Detailed historical documentation. Older future-tense commands and plans
    describe their original stage; they do not reopen a closed campaign.

WHAT THE FINAL EXPERIMENTS FOUND

CPU comparison
  The 16-call batch endpoint achieved 1.19045x speedup, with a paired 95% interval
  of [1.18380, 1.19734]. Individual requests measured 1.21223x, but duplicate
  baseline controls failed the required stability interval at two shapes.
  Both endpoints were required. The overall policy therefore did not qualify.

GPU development
  The stream64 implementation was about 3.44753x faster than the older GPU
  implementation. It nevertheless took 8.4–21.2 times the CPU time over the
  twelve tested shapes. Numerical checks passed; CPU control stability also
  failed. No GPU route was selected and no new GPU confirmation was run.

NPU task quality
  The one revised full-table, actual-history policy completed 96/96 planned
  answers validly, but only 64/96 were strictly correct. The three size groups
  scored 22/32, 19/32 and 23/32; qualification required 92/96 overall and at
  least 29/32 in every group. It failed that accuracy gate. A simpler diagnostic
  arm's 94/96 result could not qualify the harder original task.

Dependent stages
  NPU confirmation, cache development/confirmation and combined development/
  confirmation were not run because their prerequisite gate failed. These are
  unmeasured stages, not measured failures. Earlier exploratory cache and
  combined results do not substitute for fresh qualification.

The final budget family charged 17399.547867786954 of its 72000 seconds across
CPU, GPU development, and NPU diagnosis/development. Unused allowance stayed in
its original buckets; it did not authorize additional scientific retries.
Startup failures, initialization and cleanup were charged where applicable.
This is supervised hardware allowance, not total human/AI development time.

The implementation used for the completion roadmap passed 589 regression tests.
The separate checkpoint helpers passed 47 disposable checks. Neither count is
a hardware performance result. The independent USB checkpoint certified on
6 October 2026 restored 19924 inventory entries and verified 74 canonical seals.
That checkpoint predates this publication packaging work. New HDD or repository
copies are established only by their publication manifest/status and receipts.

THE ORIGINAL DEMONSTRATION

The original three-file ZIP is preserved in the private project's reference/
directory and the public edition's original/ directory. Its SHA-256 is:
  18bebaa5904a2c685198e5ba5688025d3156c284243abc3c414458a8a3f88cec

It compares three ways to compute the same synthetic, single-layer causal
attention outputs: full-prefix recomputation, newest-query recomputation without
retaining keys/values, and cached streaming. It uses fixed random weights,
float64 arrays, identical inputs, one requested numerical-library thread and
all-position correctness checks. NumPy is its only third-party dependency.
There is no trained model, downloaded model, HAT, paid API or network call.

The supplied original timings were recorded on x86_64 Linux with Python 3.12.14
and NumPy 2.3.5. They are not Raspberry Pi timings. The owner's statement that
the ZIP originated with "ChatGPT Astra Max" on iPadOS describes its creation
context, not the machine that executed the saved benchmark. See CREDITS.txt.

The original example illustrates the memory/time tradeoff of an established
optimization. It does not establish a novel attention algorithm, general LLM
acceleration, measured energy savings, improved intelligence, or repeated 5x
gains. The later native FP32/FP64-oracle and NPU tasks have their own protocols;
their ratios cannot be substituted for this original comparison.

PRIVATE AND PUBLIC EDITIONS

The publication targets are the owner's existing repositories:
  Private: https://github.com/TylerSevde/HomeCode
  Public:  https://github.com/TylerSevde/HomeCode-Public

Use PUBLICATION-MANIFEST.json for the actual exported paths and provenance, and
the applicable PUBLICATION-STATUS.json for verified delivery state. This guide
does not certify a push or copy merely by naming those destinations.

The private project and independent archival package preserve the full research
record and its restoration context. The public edition contains a curated set
of source, documentation and evidence selected for public review. It is not a
substitute for the full private backup and does not include proprietary model
weights, the whole service environment, or every raw numerical artifact.

In the public edition, run the documented verifier from its root:
  python3 -B tools/validate_public.py .
It needs NumPy. Released outcomes are under evidence/<original-run-id>/, with
summary.json, protocol.json and measurements.jsonl; the NPU evidence also
includes fixtures.json. See INSTRUCTIONS.txt for the scope of those checks.

Public statistical reproduction means recomputing the released summaries and
decisions from the released observations with the stated method. It does not
mean independently rechecking every stored tensor against every original oracle,
replaying every hardware tokenizer, verifying an omitted parent artifact, or
reproducing the original wall-clock performance. Full numerical auditing needs
the complete raw record and its frozen source/path dependencies. A fresh hardware
replication additionally needs compatible hardware, runtime and model bytes.

The public source adds AE_MODEL_PATH and AE_PLOT_PYTHON for configurable local
paths. Those publication conveniences belong to the public edition. The original
private harness and sealed source snapshots retain their recorded defaults.
No adaptation is retroactively presented as the source that made the measurements.

READING AND REUSING THE WORK

Open the published reports before running commands. A new result needs a new
output directory and an explicitly designed protocol. Existing runs, campaign
seals, budget sidecars and source snapshots are historical records: do not rerun
into them, regenerate their reports in place, or reset their budgets to create
more attempts. No closed campaign automatically resumes from this publication.

The portable CPU demonstration is the shortest route to understanding the
algorithm. The more valuable research lesson is how the later harness kept
correctness, complete planned coverage, paired controls, cumulative budgets and
rejection rules connected to its performance claims.
