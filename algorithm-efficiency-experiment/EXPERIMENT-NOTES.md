PUBLIC DERIVATIVE — Historical run/campaign/checkpoint paths below refer to the private archive. Released final evidence is under evidence/; consult PUBLICATION-MANIFEST.json for exact public coverage.

# Preparation notes

Latest status, 6 October 2026 UTC: the bounded scientific work is complete with
no newly qualified policy. See [COMPLETION-RESULTS.txt](COMPLETION-RESULTS.txt)
and the final dated entries below. The independent USB checkpoint passed its
actual restore and was certified at 2026-10-06T17:25:42.471260+00:00. Earlier entries
retain their historical meaning and do not indicate a running experiment.

The planning-phase API probe established a 2048-token compiled context capacity.
A missing newline between turns initially produced a one-token history mismatch.
Including that newline yielded identical token sequences, outputs and context
counts for retained and rebuilt histories, and context reset returned zero.

Implementation validation retained two hardware smoke runs:

- `runs/smoke-20261002`: newline-separated item/color table and repeated answer
  instruction. The model gave a color then invented a follow-up question, reaching
  the 16-token limit. The original harness ended the session after that event.
- `runs/smoke-fact-format-20261002`: semicolon-separated facts and the shorter
  instruction used during planning. Both arms completed both turns with identical
  token histories and outputs. The model answered in sentences, violating the
  strict one-word rubric while mentioning the expected color.

Before the pilot, the semicolon format was frozen. Smoke fixtures use a separate
seed base from pilot fixtures. Recovery after a token-limit completion now allows
the next turn only after the token ledger and terminal sequence validate; the
truncated observation remains explicitly excluded from completed-response ratios.
Supplementary color-mention counts were added to distinguish formatting failures
from wrong fact values without altering primary correctness or acceptance rules.

The pilot retains seven paired conversations at each of three initial prompt
sizes and four turns per conversation. No success or speedup is assumed. Diagnostic
tokenization and transcript checks precede timing; their overhead is reported
separately and included in conversation wall time, while the primary request
timer includes cache clearing and model generation. Completed runs shorter than
the 30–60 minute budget are not padded with artificial workload or idle time.

## Diagnostic implementation and first attempt

The first diagnostic attempt, `runs/diagnostic-20261002`, stopped safely after
207/444 responses. All 120 frozen requests were identical within each of the
six prompts across ten repetitions and both raw/native API routes. The two
routes agreed exactly, including a consistently wrong answer on the longest
third-turn prompt.

During the balanced study, `n128-pair01-order2-repeat2`, retained turn 3,
produced `YELLOW.<|im_end|>` with runtime occupancy 173 versus 171 tokens after
re-tokenizing the complete decoded transcript. Its recorded input was 168
tokens. The SDK's decoded stream does not expose generated token IDs, so this
mismatch does not establish cache corruption or identify a numerical cause.
The stopped run, its original source snapshot, report and checksums are retained.

The subsequent diagnostic implementation quarantines an unverifiable context,
records skipped dependent turns, and verifies an empty reset before independent
work continues. Exact-history checks remain mandatory; invalid observations are
excluded from paired comparisons and retained in the diagnostic evidence.
Streaming chunks are now saved to help investigate text/token reconstruction.
This protocol amendment does not change tables, prompts, scoring or decoding.
The validation suite exercises quarantine, reset and independent-arm recovery.

## Completed diagnostic results

`runs/diagnostic-v2-20261002` completed all 444 planned measured responses in
1298.76 seconds (21.6 minutes), with no skipped turns. The 33-test suite passed;
all saved prompts and stream concatenations, schedule keys, scores, executed
source snapshots and original artifact checksums were independently verified.
The frozen-request outputs and context counts also matched the stopped first
attempt for all 120 requests.

The six frozen prompts produced one exact output each across ten repetitions
per route. Raw text and native structured chat agreed in every case. Each route
answered 50/60 correctly; the longest third-turn prompt consistently answered
black when red was expected. This sample found no intermittent variation or
observable raw/native route discrepancy.

In the balanced 324-response study, factual correctness across both arms was:

| Queried position | Correct / total | Accuracy |
| --- | ---: | ---: |
| Beginning | 102 / 108 | 94.4% |
| Middle | 57 / 108 | 52.8% |
| End | 42 / 108 | 38.9% |

The beginning/middle/end counts at conversation turns 1, 2 and 3 respectively
were 36/18/12, 36/18/18, and 30/21/12 correct out of 36 for each position.
The lower end-position accuracy persists across turns, supporting a position
association in these selected tables. This does not establish the cause of the
retrieval failure or general accuracy on arbitrary inputs.

Rebuilt context produced 105/162 factually correct answers; retained context
produced 96/162. There were 27 different outputs among 144 eligible paired turns.
Six comparisons favored rebuilding (correct rebuilt fact, wrong retained fact),
nine favored retention, and the remaining differences included malformed output
or different formatting. Twelve comparisons had unequal earlier histories,
three had incomplete generation, and three failed the context-count ledger;
all were excluded from eligible paired comparisons but retained as observations.
Each directional factual difference recurred in all three repetitions of its
fixture/order/turn combination. Caching therefore changes observed behavior in
these cases, without identifying a firmware, precision, kernel or model cause.

All three runtime-count mismatches occurred on retained turn 3 of
`n128-pair01`, middle → end → beginning. Each emitted
`['Y', 'EL', 'LOW', '.', '<|im_end|>']`, decoding to the correct answer YELLOW,
while the runtime counted 173 context tokens and re-tokenizing the complete
transcript counted 171. This is consistent with different generated and
re-encoded text segmentation. The binding exposes decoded strings, so the
actual generated token IDs are not established. Each context was cleared and
verified empty; no dependent turn remained to skip. A separate 18-response
hardware replay reproduced all three mismatches and verified recovery.

Maximum observed temperatures were CPU 57.3°C and HAT 46.07°C. Every recorded
throttle flag was zero, the independent sampler reported no errors, and maximum
context occupancy was 1610 tokens against the 1792-token experiment budget.
Power and energy were not measured.

The main HTML report, raw JSONL, CSV, PNG/SVG plots, validation evidence, frozen
inputs, schedule, and minimal failing case are in `runs/diagnostic-v2-20261002`.
The original pilot and stopped diagnostic attempt remain preserved separately.

The smallest matched-history retained-only wrong fact was independently replayed
in `runs/diagnostic-v2-20261002-replay` (18/18 responses, 92.75 seconds). In all
three repetitions of `n128-pair06`, beginning → middle → end, turn 2 had the
same recorded 147-token prompt hash and valid ledgers in both arms. Rebuilding
answered `black.<|im_end|>` correctly; retention answered `green.<|im_end|>`.
This reproducer demonstrates an output difference under context reuse with the
checked recorded inputs; it does not identify its internal implementation cause.

## State-isolation protocol

The next experiment freezes four histories from the completed diagnostic and
crosses six context/process conditions with four target-generation presets,
three repetitions each. The original conditioning settings remain fixed. The
long case uses the diagnostic's actual black-then-green history. The control
changes only the small case's final query to item002, whose expected answer is
green. The model, runtime, tokenizer, prompt template and stop tokens must match
the checksummed diagnostic evidence.

The main matrix contains 288 target and 120 conditioning responses. Two small
hardware validation cases run first (12 targets, four conditioning responses).
Snapshot restoration must preserve verified context counts and permit complete
generation before the main matrix starts; identical answers across conditions
are not a capability requirement. Snapshot contents are not interpreted as
specific cache or token-ID structures. Fresh conditions recreate the host
process, model and device client after releasing the original resources; they
do not reset accelerator firmware.

The 50-test suite passed before hardware execution, including source-history
reconstruction, parameter isolation, snapshot integrity and memory guards,
conditioning quarantine, fresh-child lifetime and process-group teardown,
schedule coverage, report contrasts and the existing regression suite. The
supervised hardware budget is 3600 seconds including baseline and cooldown.

## Completed state-isolation results

`runs/state-isolation-20261002` completed in 2695.55 seconds (44.9 minutes):
288 main target responses, 120 main conditioning responses, and 16 initial
hardware-validation responses. All 96 case/preset/condition cells had three
identical outputs across their repetitions. There were no skipped targets,
conditioning failures, context-count mismatches or excluded paired comparisons.

The resolved model frequency penalty was 1.100000023841858. Supplying all
resolved settings explicitly produced identical output in all 72 comparisons
with the original implicit settings. Changing only the target penalty to 1.0
changed 36/72 outputs and removed every rebuilt-versus-retained difference
(0/12 differences, compared with 9/12 under the original settings).

Colors below omit the saved punctuation and terminal markers. Each entry
repeated three times; live, saved-live, same-instance restoration and
fresh-process restoration agreed exactly. Both rebuilding conditions agreed.

| Case | Expected | Original settings: rebuilt | Original settings: retained/restored | Penalty 1.0: all six conditions |
| --- | --- | --- | --- | --- |
| Small failure | black | black | green | black |
| Same-table control | green | green | green | green |
| Medium failure | red | red | blue | red |
| Long reverse case | red | black | red | black |

Penalty 1.0 therefore improved consistency in these selected cases, but did not
make the long case correct. Penalty 1.2 changed 48/72 outputs relative to the
explicit original settings. All 48 retained/restored targets at 1.2 contained
extra control markers and were classified as malformed; rebuilding remained
unchanged. The raw streams preserve those failures. Across all settings, 156
of 288 targets were strictly correct, 84 contained a wrong fact, and 48 were
unscorable because of malformed output. These selected-case counts are not an
estimate of general model accuracy.

There were zero output differences in each of the 48 live-versus-saved-live,
saved-live-versus-restored, same-instance-versus-fresh-instance restoration,
and existing-versus-fresh rebuilding comparisons. Fifty snapshots totaling
2,216,845,048 bytes were saved; all 100 loads restored the expected context
count. All 150 model lifetimes completed, with fresh processes starting after
the original resources were released.

This intervention narrows the observed behavior to an interaction between the
frequency-penalty setting and preserved context/state. It does not identify
the internal penalty formula, actual generated token IDs, numerical mechanism,
or serialized state fields. A fresh client process does not reset accelerator
firmware. No snapshot-save observation effect was detected in this matrix.

Peak temperatures were CPU 59.5°C and HAT 44.35°C, with all throttle flags zero
and no sensor API errors. One preflight telemetry sample occurred before the
sensor's state file existed; it was within the startup window and preceded
measurement. Baseline, measurement and cooldown telemetry had no read errors.
The original source runs and SSD archive remain checksummed separately.

The report includes CSV comparisons, the response matrix in PNG/SVG, raw JSONL,
opaque snapshots with lineage, frozen inputs, executed-source copies and a
focused reproducer. The reproducer replays the smallest selected differing
case across all settings and state conditions. The complete matrix already
contains its three repetitions; an additional hardware replay was not run.

## Fresh-workload cache-validation protocol

The next run returns to the efficiency question with 30 newly seeded tables,
four turns, two repetitions, two context methods and two fully explicit
generation settings (original resolved penalty and 1.0): 960 main responses.
Eight saved small-case responses validate the original output pattern before
the main matrix. Settings apply throughout each conversation. The fixed seed
formula is deliberately disjoint from the pilot, and both seeds and complete
tables are checked against preserved inputs before any new generation.

The primary timer covers a continuous application request, including actual
rendering, required clear, generation and cleanup. Untimed diagnostic checks
establish prompt/token identity; timed preparation must reproduce that input.
Initial verified session setup is separate. The primary statistic uses the
ratio of summed follow-up request times per repetition, the median of the two
ratios within each table, and the median across ten independent tables. The
bootstrap resamples tables, with 10,000 samples and seed 20261002.

Quality requires every one of the 80 planned answers per arm to be strictly
correct for that size/setting. Complete matched coverage and a 95% exploratory
speedup interval above 1 are also required. Incomplete or quarantined coverage
cannot become a success through denominator reduction. The supervised budget
remains 3600 seconds, including setup, baseline and cooldown. Replays require
an explicit verified source and are labeled as repeated evidence.

## Completed fresh-workload validation

`runs/cache-validation-20261002` finished its supervised protocol in 2435.12
seconds (40.6 minutes). It recorded 956/960 main responses and four explicit
dependent skips, plus all eight initial control responses: 964 generations in
total. All 120 comparison blocks completed. The report distinguishes completed
protocol execution from incomplete overall measurement coverage. Every planned
main turn is accounted for by a response or a recorded skip.

The 68-test suite passed. All 30 tables were new relative to the saved fixture
inventory, and the frozen schedule counterbalanced arm order between repeats.
All 478 observed table/setting/arm/turn combinations had two identical outputs.

With penalty 1.0, rebuilt and retained context had identical recorded inputs
and outputs for all 240 paired turns. Both arms completed every planned answer.
The per-size results were:

| Initial token target | Strict correct, each arm | Factually correct, each arm | Follow-up request speedup | Exploratory 95% interval |
| --- | ---: | ---: | ---: | --- |
| 128 | 48/80 (60%) | 60/80 (75%) | 1.562× | 1.416–1.563× |
| 512 | 68/80 (85%) | 68/80 (85%) | 3.760× | 3.758–3.761× |
| 1536 | 50/80 (62.5%) | 50/80 (62.5%) | 9.793× | 9.785–9.802× |

Each interval uses all ten independent tables for that size, with the two
repetitions combined within each table. These are follow-up application-request
speedups, including rendering, required clear, generation and cleanup. They
exclude separately recorded diagnostic RPCs. First-turn control ratios were
approximately 1.007, 1.003 and 1.001, respectively.

The speed gain is clear in this workload, and penalty 1.0 preserved rebuilding's
observed answers while reusing context. Accuracy and formatting remain limiting:
no size/setting met the preregistered requirement that every planned answer be
strictly correct in both arms. The small-case factual/strict gap reflects extra
prose in otherwise factually correct answers. Wrong answers and malformed
outputs remain in the denominators; this is not a general accuracy guarantee.

Across settings, all 240 rebuilt outputs were unchanged. Retained output
changed in 42/236 comparisons where both settings produced a response. Under
the original default, retained strict correctness was 48/80, 50/80 and 48/80
at the three sizes. At penalty 1.0 it was 48/80, 68/80 and 50/80. The baseline
had 74, 58 and 72 eligible paired turns out of 80 and only eight, two and six
independent tables with complete matched follow-ups. Its timing estimates
therefore describe selected subsets; the particularly small medium-size subset
does not support a broad performance conclusion.

The four skipped turns were the fourth retained turn of `fresh-n512-04` and
`fresh-n512-09` under the original default, in both repetitions. Earlier outputs
contained control markers and ultimately ended with `<|endoftext|>`; the chat
template's next rendering could not reproduce that exact prefix. The harness
recorded the failure, verified an empty reset and continued independent work.
All attempted generations completed, and every observed runtime context count
matched the reconstructed transcript. There were no context-count mismatches.

Peak temperatures were CPU 56.75°C and HAT 45.26°C. All throttle flags were zero,
the sensor reported no API errors, and experiment workers and the sampler
released the device after cooldown. Raw responses, skips, settings, timing
components, statistical units, plots, source snapshots and validation evidence
are preserved with checksums. `reproduce.json` supplies an explicit verified
replay command; a second full hardware replay was not run.
# Verified hybrid lookup: preregistered implementation protocol

The next experiment uses the existing exact item/color task and returns the
verified CPU value when the HAT answer is wrong, malformed, truncated or has an
invalid context ledger. Source: `runs/cache-validation-20261002`; require its
complete penalty-1.0 matched subgroup and verify all saved source checksums.

Run `hybrid-validation` with the installed Qwen2.5-1.5B HEF and HailoRT 5.1.1.
Use all seven explicit generation parameters with penalty 1.0, greedy decoding,
seed 12345 and maximum 16 output tokens. Freeze thirty fresh tables before
generation: ten at each original target, seed `2026100301 + size*1000 + index`,
where index is 0–9. Reject seed or table-hash reuse. Query beginning, middle,
end and beginning again, with two repetitions. Shuffle blocks with seed
20261003 and cycle the six workflow permutations, balanced to six/seven
occurrences per position per size. There are 720 main answers: 240 CPU,
240 full-table retained HAT, 240 retrieved-record HAT with verification.

Before the main schedule, reproduce four retained turns on the lexicographically
first source table at each size; require agreement with both source repetitions
(12 generations). Follow with one-record controls for all eight colors, requiring
correct final verified values while retaining raw failures. Total planned HAT
generations: 500. No tuning or selection based on these new outputs.

The immutable dictionary index rejects duplicate identifiers and invalid
records. Retrieval and answer validation receive records, not fixture answer
labels. A hybrid request starts with a verified empty context and includes only
the queried row. Accept only strict, logically complete model answers with a
valid context count; save all raw text/chunks and explicit fallback provenance.
An invalid ledger or truncated hybrid request is reset before returning the
CPU value. A full-table conversation with invalid continuation skips dependent
turns explicitly. Failed resets, SDK exceptions, context-budget violations and
hardware limits stop the run and preserve partial coverage.

The continuous request timer includes retrieval, rendering, token/context RPCs,
required reset, generation/cleanup, validation and fallback recovery. Logging,
index construction and model loading are separately reported. Compare sums of
all four request times per repetition, combine two repetitions by their median,
and bootstrap medians across independent tables with 10,000 samples. Pair by
record/table/query identity, not equal token histories: the prompts differ by
design. Never count a corrected answer as raw HAT success. CPU timings include
Python timer/result-construction overhead and are descriptive of this code.

Acceptance requires all 240 CPU and 240 hybrid returned answers correct,
complete paired coverage and correct negative-case behavior. Report raw HAT
strict/factual accuracy, acceptance/fallback rates, latency versus both CPU and
full-table HAT, load/index costs, coverage and temperatures. No HAT advantage
is assumed. The existing supervisor enforces a maximum 3600-second session.
The original ZIP, previous run directories and their checksum inventories stay
unchanged. Verified replay must reproduce the frozen fixtures, schedule and
parameters and be labelled repeated evidence.

## Hybrid validation completed — 2026-10-02

Run: `runs/hybrid-validation-20261002`; supervised elapsed 641.587517 seconds
(10.7 minutes), worker phase 578.294994 seconds. All 90 tests passed, including
22 new tests. The supervisor completed normally with no safety stop. Coverage:
718/720 main answers, 478/480 main HAT generations, all 20 control generations,
and all 240 direct CPU answers. All twelve frozen baseline controls reproduced;
the eight color controls returned correct verified values, with six raw model
acceptances and two fallbacks. Controls are excluded from the main results.

| Original token target | CPU correct | Hybrid returned correct | Hybrid raw strict | Hybrid raw factual | CPU fallbacks | Full-table strict / planned |
|---|---|---|---|---|---|---|
|128|80/80|80/80|68/80|76/80|12|64/80|
|512|80/80|80/80|60/80|64/80|20|44/80 (78 observed)|
|1536|80/80|80/80|66/80|74/80|14|54/80|

The hybrid accepted 194/240 raw HAT answers (80.83%) and returned 46 explicitly
labelled CPU fallbacks. Twenty were recognized correct facts in verbose answers;
26 were unsupported by the frozen scorer. Unsupported examples included
`The color is brown.` and `Brow.`; they must not all be described as wrong facts
or counted as raw model successes. No HAT response produced a scorable wrong
color in the retrieved-record workflow. The original scoring rubric was not
changed. All 240 final hybrid answers and all 240 CPU answers matched records.

A full-table context mismatch occurred at the third query on `fresh-n512-05`
in both repetitions. `YELLOW.<|im_end|>` had runtime context count 559 versus
556 after re-tokenizing the saved text. The model reset was verified; the fourth
query was skipped in each repetition. All other workflows continued independently.
This does not expose the hidden generated token IDs or establish cache corruption.
The saved report has `protocol_completed: true`, `complete: false` and
`pipeline_passed: false`: the latter gate also requires complete three-workflow
paired coverage. Each size has complete, correct CPU and hybrid answer coverage.

Workflow ratios use whole four-question totals, medians over two repetitions
and then independent tables, with 10,000 table-bootstrap samples:

| Target | Eligible tables | Full-table / hybrid request time | Exploratory 95% interval | Hybrid / CPU request time |
|---|---|---|---|---|
|128|10/10|1.1320|[0.9315, 1.1334]|54,705.3|
|512|9/10|1.6938|[1.1869, 2.0920]|51,520.1|
|1536|10/10|3.0226|[2.1706, 3.1005]|49,964.0|

The medium comparison is restricted to nine complete tables and fails the
predeclared complete-coverage gate. The small interval includes 1. Only the
large group meets the declared per-size faster-than-full-table gate. These
comparisons use intentionally different prompts; baseline raw quality is shown
above. They are not measurements of caching alone. The direct CPU request
median is 0.014185/0.015879/0.015120 ms by size; the hybrid median is
677.373/677.914/677.410 ms. Very short CPU timings include Python timer and
result-construction overhead. Index construction and model loading are separate.
For this exact lookup task, the HAT adds substantial latency and no capability
needed to obtain the answer; the CPU baseline is already sufficient.

Peak CPU temperature was 55.1°C, peak HAT temperature 43.082°C, and all throttle
flags were zero across 629 telemetry observations. The HAT was released at the
end of the worker and the sensor exited after cooldown. No power/energy claim
is made. All 359 observed table/workflow/turn cells had identical outputs across
their two repetitions; the missing full-table cell has no measurements.

The independent audit verified every response/chunk reconstruction, prompt,
parameter set, score, fallback decision, table hash, source label, planned key,
session total and CSV row count. It independently recomputed six bootstrap
intervals from 29 eligible tables, checked all 24 executed source files and
16 prior-fixture files, verified original cache/state/diagnostic/pilot inventories
(67/457/55/43 artifacts), and confirmed the SSD ZIP hash. Offline replay source
preparation passed; a second hardware replay was not performed. The audit was
adjusted to account explicitly for the permitted baseline skips rather than
assuming complete observations; measured runtime code and protocol were unchanged.

Public CLI verification also passed. `runs/hybrid-query-known-20261002`
returned yellow for item001 through one strictly correct HAT generation.
`runs/hybrid-query-missing-20261002` returned `not_found` for item99999 with
zero generations and no HAT sensor/model startup. Separate CLI checks rejected
duplicate identifiers and unsupported colors before creating an output directory.
Both query runs have independently verified checksums; `cli-checks.json` records
their identities. No process held `/dev/hailo0` after these checks.

## Natural-language routing — frozen protocol before inference

Implement `ask` and `language-validation` using the unchanged Qwen2.5-1.5B HEF,
HailoRT 5.1.1 and explicit penalty-1.0 parameters (greedy, seed 12345, max output
16). Every model request starts from a verified empty context and obeys the
existing 1792-token experiment limit. No record contents or evaluation labels
enter the model prompt. It interprets one question into LOOKUP/COUNT/LIST with
one literal argument, or ABSTAIN; CPU execution supplies all factual results.

Verify all artifacts in `runs/hybrid-validation-20261002` and require complete,
correct CPU and hybrid subgroups. Preserve its incomplete full-table baseline
without treating it as incomplete source evidence for the two valid subgroups.
Reuse all thirty frozen tables. Before the first generation, freeze the prompt,
corpus, expected commands/results, table mappings and schedule. Six fixed prompt
examples double as disposable warm-ups, excluded from evaluation; none of their
exact questions appears in the evaluation corpus. Do not tune after their output.

The corpus contains ten phrasing families per operation, four arguments per
family: 120 supported questions. Two families per operation use the canonical
CPU forms; eight use other wording. Ten negative families each have four cases:
missing referents, multiple items, multiple colors, negation, unsupported colors,
unsupported attributes, unsupported aggregations, compound requests, hypothetical
overrides and mutation requests. Total 160 distinct questions across forty
families, including ten missing lookups and 23 empty filters on the source tables.
Questions require explicit item IDs or one of the eight allowed color names.

Run each question twice through cpu, hat and auto engines: 960 attempts. The
schedule uses seed 20261004, with six permutations balanced to 106/107 occurrences
of each engine at each order position. Planned main HAT calls: cpu 0, hat 320,
auto 272; six warm-ups bring total generations to 598. CPU routing must avoid
model and sensor startup in public CLI calls. The benchmark interleaves engines
with a model already loaded. The old `query` command remains compatible.

Each request timer includes parsing/routing, token and context RPCs, required
verified reset, generation/cleanup, command validation and CPU execution.
Logging, index building, model loading and complete CLI/session time are separate.
Malformed commands, ungrounded arguments, explicit abstentions, truncated output
and context mismatches return abstention. Bad/incomplete contexts reset before
continuing independent requests. SDK exceptions, failed resets, hardware limits
and deadline terminate the run and retain partial evidence. Maximum session: 3600 s.

Metrics separate raw model intent, final accepted intent, final result accuracy,
supported-query coverage, required abstentions and incorrectly accepted intents.
A valid wrong operation counts as an error even if its answer happens to match.
Malformed negative responses that lead to abstention do not count as correct raw
model interpretations. None of these evaluation labels can affect routing,
validation or execution. Structural argument checks cannot prove intended meaning.

For each complete phrasing family, retain all four argument variants and both
repetitions in paired engine totals. The 10,000-sample paired bootstrap with seed
20261004 resamples forty family clusters and recomputes hat/auto ratios of total
request time and auto-minus-cpu supported-coverage differences. Do not bootstrap
individual answers or claim that this authored corpus represents arbitrary English.

The automatic-route success gate requires complete planned coverage, increased
supported coverage over cpu, zero incorrectly accepted automatic interpretations,
fewer HAT calls and lower total request time than hat. Report exploratory timing
confidence separately. Failed quality criteria are valid experiment outcomes;
do not change the prompt, grammar, corpus or scoring to improve the recorded run.
Save JSONL/CSV evidence, charts, standalone HTML, replay instructions, snapshots,
checksums, test output and an independent audit. Preserve all earlier runs.

## Natural-language routing completed — 2026-10-02

Run: `runs/language-validation-20261002`. The supervised session finished normally
in 1392.817398 seconds (23.2 minutes); worker phase 1329.612084 seconds. All 960
planned attempts completed: 320 per engine. There were 592 main HAT generations
and all six warm-up generations, with no missing measurements or context-count
mismatches. All 110 tests passed, including twenty new language tests. Runtime
code, prompt, corpus, parameters and scoring stayed fixed throughout measurement.

| Engine | Supported queries resolved / planned | Required abstentions correct | Incorrect accepted intents | All final outcomes correct / planned | HAT calls | Total request seconds | Median request ms |
|---|---|---|---|---|---|---|---|
|cpu|48/240 (20%)|80/80|0|128/320|0|0.016430|0.037611|
|hat|94/240 (39.17%)|80/80|6|174/320|320|696.693888|1707.454727|
|auto|124/240 (51.67%)|80/80|6|204/320|272|602.585842|1565.356733|

Automatic routing avoided 48 HAT calls (15%) and reduced total request time by
13.5078% compared with always-HAT. Paired whole-family bootstrap speedup was
1.156174 with exploratory 95% interval [1.042294, 1.335328]. The supported-coverage
gain over the fixed CPU grammar was 31.6667 percentage points, with exploratory
95% interval [19.1667, 44.8529] points. These intervals resample forty authored
phrasing families, preserving their argument variants and repetitions. They do
not describe performance on arbitrary English or an optimal general CPU parser.

The useful-routing gate is false solely because of accepted semantic errors;
complete coverage, improved supported coverage, reduced HAT calls and reduced
total time all passed. Three unique supported count questions were interpreted
as LIST by both HAT-based engines in both repetitions:

- `How large is the group of items that are brown?` (count-10-1)
- `How large is the group of items that are pink?` (count-10-4)
- `What is the size of the set of green items?` (count-08-4)

These commands had correct, explicit arguments and valid grammar. CPU execution
returned the correct lists for those commands, but the questions requested counts.
Thus command validity and factual execution cannot certify natural-language intent.
Each engine's six incorrect accepted attempts are three repeated cases, not six
independent language failures. All negative attempts abstained in every engine.

Supported resolution by operation (out of eighty planned per engine):
LOOKUP cpu 16, hat 30, auto 44; COUNT cpu 16, hat 8, auto 22; LIST cpu 16,
hat 56, auto 58. The HAT-only interpreter did worse than the simple CPU grammar
on count questions in this corpus. Auto retained the exact grammar's successes.

HAT-only abstention reasons were 92 token-limit terminations, 64 explicit model
abstentions, 62 malformed commands and two ambiguous/missing arguments. Automatic
routing had 82 token-limit terminations, 52 explicit model abstentions, 54 malformed
commands and two ambiguous/missing arguments. The parser did not salvage a command
prefix from extra prose or multiple commands. Supported abstentions were 192/140/110
for cpu/hat/auto respectively; rejecting everything is not counted as language
coverage. Raw model intent was strictly correct on 106/320 HAT-only calls and
88/272 auto HAT calls. Negative cases that abstained after malformed output remain
raw interpretation failures despite their correct final abstention.

All 480 observed question/engine cells produced identical outputs and results
across repetitions. Fixed prompt sizes were 193–210 tokens; maximum output stayed
16. Model loading took 8.648957 seconds, separately from request timings. Peak CPU
temperature was 55.1°C, peak HAT temperature 43.032°C, all throttle flags were zero,
and there were 1363 telemetry observations. No hardware or context safety stop
occurred. Power and energy were not measured.

The independent audit passed all 960 attempts and 598 generations, checking raw
chunks, grammar, argument grounding, command execution, intent/final scores,
coverage, routing counts, all CSV exports and both bootstrap intervals recomputed
from raw records. It verified 28 executed source files, all 76 original source-run
artifacts, the original SSD ZIP and replay preparation. There were forty complete
family clusters. A second hardware replay was not performed. The two plotted
figures and embedded standalone HTML were visually inspected.

Six public CLI checks passed in separate supervised run directories: exact lookup,
zero-count filter, sorted item list, missing identifier, CPU abstention and an
automatic HAT paraphrase. The five CPU-routed checks had no model environment or
HAT sensor startup; the HAT paraphrase used one generation. Its case was selected
from an already successful measured example solely to exercise the public CLI,
not as new accuracy evidence. An empty question was rejected before creating a
run. Each CLI directory's checksums were verified, and no process held /dev/hailo0
afterward. Full CLI wall time (about 6.5 seconds on CPU paths and 22.3 seconds for
the HAT example) includes supervision baseline/cooldown and startup, unlike the
inner request measurements. `cli-checks.json` preserves exact times and identities.

## Strict query validation — protocol frozen before measurement

The next profile, `language-strict-validation`, compares the old `cpu` and `auto`
engines with an opt-in `strict` engine; `ask` still defaults to `auto`. The strict
engine executes only a full match to thirty documented forms derived from the
old supported question templates. It normalizes case/whitespace/trailing `.?!`,
accepts one leading `please`, and substitutes the plural nouns items/entries/
records. It accepts one distinct operation/argument pair, preserving identifier
digits, and otherwise abstains with three static syntax examples. It never calls
the model. This is a bounded language contract, not a semantic guarantee over
arbitrary English. Rules live in runtime code, separately from evaluation labels.

`protocols/strict-grammar-v1.json` freezes the grammar specification and runtime
module hash before new-corpus authoring. `protocols/strict-corpus-v1.json` freezes
the authored corpus, generator hash, grammar-freeze hash, and previous source-run
checksums before evaluating the new questions. Preflight verifies all identities
and regenerates the same corpus; a changed grammar or corpus fails preflight.
The prompt and seven explicit generation parameters remain identical to the
previous language experiment, with greedy decoding, penalty 1.0, seed 12345,
sixteen generated tokens and the existing context/reset checks.

All 160 old questions are CPU-only regression evidence: 120 supported answers and
40 required abstentions. The three count/list mistakes are included. The new
corpus has sixty contract combinations (five templates per operation, four
argument variants), sixty unfamiliar paraphrases (five templates per operation,
four variants), and forty negative requests (ten categories, four variants).
Contract variants intentionally exercise known grammar and are not independent
language-generalization evidence. Unfamiliar forms change sentence structure,
not merely color, punctuation, politeness or plural noun. Negatives cover missing
referents, multiple IDs/colors, negation, unknown colors, unsupported attributes,
unsupported aggregation, compound requests, hypothetical overrides and mutation.

Thirty verified fact tables are reused, with ten missing-identifier questions and
empty color filters in both supported strata. Expected commands are authored;
expected answers come from direct table operations independent of the answering
function. No labels or tables enter model prompts. New questions do not exactly
overlap old questions or the six fixed prompt examples/warm-ups. No questions,
rules, prompts or scoring are selected or revised in response to new results.

Each engine handles all 160 questions twice, giving 960 paired main attempts.
The schedule uses seed 20261005 and balanced cyclic engine order (106/107 jobs in
each position). Planned calls are cpu=0, strict=0, auto=320, plus six HAT warm-ups.
The supervised hardware session is bounded to 3600 seconds, with the existing
throttle, temperature, stale-telemetry, progress and context checks; baseline and
cooldown are thirty seconds each. Failed runtime/reset checks stop the session.

Report contract, unfamiliar and negative strata separately. Measure correct
supported coverage, erroneous accepted intents, raw HAT correctness, abstention
reasons, request time and HAT calls. Log model/index startup and total session/CLI
duration separately. A 10,000-sample paired bootstrap resamples complete template
families separately within each stratum (15/15/10 families), retaining four
arguments and both repetitions; seeds are 20261005, 20261006 and 20261007.
Report auto/strict and strict/cpu total request-time ratios, and supported
coverage differences. Ratios can compare answers with abstentions and therefore
are not speedups at equal language coverage. Do not pool strata for inference.

The strict gate requires complete measurements, all 160 regressions correct,
all 120 repeated contract attempts resolved, all eighty repeated negatives
rejected, no incorrect accepted strict intent, exact planned main/warm-up call
counts and valid context ledgers. Unfamiliar-language coverage is a separate
measurement, not a reason to broaden the grammar after seeing outputs. Complete
the experiment and report failed gates without outcome-based tuning. Preserve
raw observations, hashes, executed sources, independent audit, standalone report,
plots, CSV/JSON and verified replay preparation. No energy claim is made.

## Strict query validation completed — 2026-10-02

Run: `runs/language-strict-validation-20261002`. The supervised session completed
in 876.095165 seconds (14.6 minutes), with 812.862201 seconds in the worker phase.
All 960 main attempts and 326 HAT generations (320 main plus six warm-ups)
completed. The 160 previous-question regressions all passed. All 126 unit tests
passed, including sixteen new tests. Runtime sources, grammar, authored corpus,
prompt, parameters, schedule and scoring stayed fixed during measurement.

| Stratum | Engine | Supported resolved / planned | Required abstentions correct / planned | Incorrect accepted intents |
|---|---|---|---|---|
|Documented grammar|cpu|0/120|0/0|0|
|Documented grammar|strict|120/120|0/0|0|
|Documented grammar|auto|38/120|0/0|4|
|Unfamiliar paraphrases|cpu|0/120|0/0|0|
|Unfamiliar paraphrases|strict|0/120|0/0|0|
|Unfamiliar paraphrases|auto|22/120|0/0|0|
|Negative requests|cpu|0/0|80/80|0|
|Negative requests|strict|0/0|80/80|0|
|Negative requests|auto|0/0|80/80|0|

The strict acceptance gate passed every predeclared criterion: complete
measurements, correct regressions, all documented-grammar requests resolved,
all negative requests rejected, zero incorrect strict acceptances, exact call
counts, and valid context ledgers. Strict mode answered 120/240 supported
attempts overall (50%), auto 60/240 (25%), and the original six-form parser zero.
All new questions use forms outside that original parser's grammar. The new
contract stratum is intentionally derived from the expanded grammar; its 100%
coverage is contract validation, not independent general-language evidence.

Strict abstained on every unfamiliar paraphrase. The model correctly interpreted
eleven of sixty distinct unfamiliar questions, giving 22/120 repeated attempts
(18.33%). Its exploratory paired family-bootstrap coverage advantage over strict
on this stratum was 18.33 percentage points, with 95% interval [6.67, 31.67].
This is measured coverage on authored templates and reused tables, not a random
sample of user language. Some authored operation families share sentence
constructions; the reported bootstrap reflects the predefined template-family
grouping and does not establish universal accuracy.

Automatic routing accepted LIST instead of COUNT for two contract questions:
`How large is the group of records that are brown?` and the corresponding pink
question. Each failed in both repetitions, giving four errors rather than four
independent language failures. The explicit color arguments and output command
syntax were valid; CPU execution correctly produced lists for the wrong intent.
Strict returned the expected counts. All 480 question/engine cells produced
identical output/result tuples across repetitions.

Automatic routing used 320 HAT calls, with 70/320 strictly correct raw model
intents. It had 134 token-limit terminations, 68 malformed commands, 52 explicit
model abstentions, and two ambiguous/missing-argument rejections. Eighty negative
attempts abstained correctly, but malformed model output is not credited as a
correct raw interpretation. Strict and cpu made no model calls. The HAT remained
an experimental comparator; its prompt and sixteen-token output limit were not
retuned after observing results.

| Engine | Total request milliseconds | Median request milliseconds | Correct final outcomes / planned |
|---|---|---|---|
|cpu|12.855490|0.036110|80/320|
|strict|24.744250|0.060054|200/320|
|auto|782726.373570|2360.513926|140/320|

The per-stratum report contains all ten exploratory bootstrap intervals.
Request-cost ratios compare differing coverage and abstention behavior, so they
are not equal-coverage speedups. Model loading took 8.653799 seconds separately.
The session recorded 859 telemetry rows, peak CPU 54.55°C, peak HAT 42.172001°C,
zero throttle flags, zero context mismatches and no safety stops. There were ten
missing-identifier questions and 22 empty-filter questions in the new corpus.
No power or energy measurement was made.

The independent audit verified all raw chunks, prompt identities, strict matches
against a separate literal expansion of the frozen contract, accepted argument
grounding, CPU execution, labels, scores, timing boundaries, repeat coverage,
all ten bootstrap intervals recomputed from raw records, and CSV exports. It
verified 33 executed source files, all 86 previous source-run artifacts, and the
original SSD ZIP. Offline replay preparation regenerated the exact frozen corpus
and schedule and passed; a second hardware replay was not performed. Both plots
and their embedding in the standalone HTML report were inspected.

Nine public CLI checks passed: two count formulations from the failure family,
normalized lookup, leading-zero missing identifier, sorted list, empty list,
unfamiliar-language abstention, compound-request abstention, and the unchanged
default automatic canonical lookup. The eight strict checks deliberately used
an absent model path and created neither a model environment nor HAT sensor
artifacts. A blank question was rejected before creating a run directory.
Each CLI run's artifacts were independently checksummed and verified.

Strict CLI inner request times were 4.169–6.404 ms, including first import and
compilation of its grammar. This startup occurred before timed requests in the
benchmark worker. Full CLI durations were 6.495–6.524 seconds, including the
three-second baseline and cooldown. `cli-checks.json` preserves those boundaries
and exact measurements. No process held `/dev/hailo0` after verification. Strict
remains opt-in; the existing default, older profiles and prior artifacts remain
unchanged.

## Bounded attention feedback with VideoCore GPU — 2026-10-02

Implemented `improve`, native vectorized CPU one/four-thread attention, FP32
NumPy, eight Vulkan configurations, and a three-round deterministic/HAT-guided
search. Vulkan must select the hardware V3D device; software rendering is
rejected. The HAT proposes fixed catalog IDs and cannot edit code or weights.
The original archive remains unchanged. Existing query defaults remain intact.

The completed artifacts are in `runs/feedback-attention-continuation-20261002`.
All 5,220 numerical measurements completed: 324 CPU baselines, 1,296 development,
2,160 fresh-round validation and 1,440 untouched final-test observations.
Twenty-four concurrency conditions each ran 24 additional numerical requests.
All measured numerical outputs passed atol=1e-5, rtol=1e-4 against the original
FP64 cached implementation on exactly quantized FP32 inputs. All eight GPU
configurations passed the 132 initial edge-shape and semantic checks with Vulkan
synchronization validation. The final software suite has 145 passing tests.

The initial run stopped after 66.127066 seconds, before benchmarking, because
FP32 NumPy exceeded tolerance when the semantic perturbation multiplied inputs
by up to fourteen. Native CPU passed that stress case. The evidence is preserved
in `runs/feedback-attention-20261002/failure-analysis.json`. The retry changed
the semantic perturbation to sign changes, preserving the frozen Gaussian
input magnitudes/distribution. Tolerances and benchmark fixtures were not changed.
No accuracy guarantee for arbitrary-magnitude inputs is inferred.

The retry stopped at its 1,500-second search allowance after 1,594.508309 seconds
of supervised time and 3,530 measurements. That preserved run is
`runs/feedback-attention-v2-20261002`. An explicit execution amendment continued
only missing arms in a new directory, deducting prior time from the original
3,530-second cap. It verified the same numerical sources, binaries, protocol,
fixtures, model, runtime and device identity; preserved all six recorded choices;
and repeated no completed measurement. The last trial's validation spans the
two sessions. The continuation took 473.338563 seconds. Total supervised time,
including the initial stopped attempt, was 2,133.973938 seconds (35.57 minutes).

Both strategies tried C0, C4 and C6. All 72 promotion decisions retained CPU.
The three HAT proposals were rejected: a truncated list, the already-tested C0,
and a response with extra catalog text. Deterministic fallback choices receive
no model credit. The frozen coordinate and HAT policies equal the CPU baseline
in every cell. Differences between their final timings are repeated measurements
of identical implementations, not evidence of a new algorithm or model benefit.
No amortization claim is accepted. CPU uses native1 for streaming batch-one
cells and streaming N512/B4; other measured cells use native4. Unknown shapes
resolve to native1. Policy artifacts remain opt-in.

The coordinate development measurements put the three GPU candidates about
3.9–279 times slower than their contemporaneous CPU incumbents across cells.
These descriptive ratios vary with CPU timing noise and are not held-out GPU
speed claims. Larger tiles reduced the largest prefill batch from about 1,603 ms
(C0) to 514 ms (C4), but did not beat CPU. Only those three catalog entries were
performance-searched; all eight received the initial correctness checks.

Concurrency used the fastest verified development GPU configuration, even
though neither final policy selected GPU. Mean completion times were:

| Condition | Mean complete job time |
|---|---:|
| CPU then HAT | 5,185.55 ms |
| CPU overlapping HAT | 5,124.40 ms |
| GPU then HAT | 16,106.19 ms |
| GPU overlapping HAT | 15,592.27 ms |

CPU overlap/serial speed ratio was 1.01193 (95% interval 0.99666–1.02537);
GPU overlap/serial ratio was 1.03296 (1.02821–1.03852). These intervals resample
six paired repetitions of fixed jobs, not independent input datasets. All 24
diagnostic adviser outputs failed the single-ID contract, so these are timing
diagnostics rather than validated application gains. The GPU-overlap job was
about 3.04 times slower than CPU overlap. Host intervals overlapped for about
four seconds, but that does not establish continuous device overlap.

One first overlapped GPU request took 1,845 ms at the Python boundary while its
GPU timestamp interval was about 495 ms. A host/API scheduling bottleneck is a
hypothesis, not a confirmed Python GIL diagnosis. `next-experiment.json` freezes
the proposed follow-up: validate a shorter adviser prompt on separate cases,
then compare thread and separately spawned-process orchestration across the same
CPU/GPU and serial/overlap conditions. Each process owns its device context;
live Vulkan/Hailo contexts must not be forked. Preserve token-ledger and numerical
checks, include IPC costs, use fresh inputs, and require a genuine end-to-end
gain against the best CPU configuration before accepting GPU assistance. This
follow-up is planned and has not run.

Peak temperatures across the measured segments were CPU 67.2°C and HAT
40.755°C, with no throttle flags. The live desktop continued to share the GPU;
GPU utilization and energy were not measured. Native allocation/setup and
fixture loading, oracle checks and logging are outside primary request times;
input/weight staging, computation, synchronization and output retrieval are
inside. HAT setup, advice cost and total tuning cost are separately recorded.

The independent audit verified 240 fixtures, exact observation identities,
frozen policy ordering, all 72 decisions, six proposals and 24 concurrency
conditions. It recomputed 107 intervals from raw observations and verified the
cumulative one-hour bound. `source/` preserves measurement-time code;
`analysis-source/` and `final-source/` preserve the delivered analysis and final
implementation. Standalone HTML, CSV/JSON, PNG/SVG, compiler/shader artifacts,
raw ledgers, replay instructions and checksums are retained. The HAT and all
experiment-owned workers were released; unrelated desktop GPU clients remain.

## Completed coordination follow-up — 2026-10-02

The host-orchestration follow-up described above is now implemented and complete
in `runs/coordination-20261002`. The user selected device coordination and a
30-minute supervised cap. One session took 832.415 seconds (13.87 minutes),
including setup, adviser checks, excluded pilot, measurements and cooldown.
The existing model, generation settings, attention code and compiled numerical
artifacts remained fixed. No clocks, packages, model weights or query defaults
were changed.

The new `coordinate` command uses the same persistent numerical and HAT owners
for serial and overlapping work. Thread owners live in one process; process
owners are separately spawned. Ownership, submission, worker execution, result
delivery, GPU timestamps and numerical checks are recorded. Each architecture
block creates its own contexts, warms all three fixtures on both numerical
backends, and performs one excluded HAT warmup. Architecture order alternates
and condition order rotates across the six paired repetitions. Initialization,
warmups and cooldown count against the session budget but are outside measured
full-job latency; communication, numerical validation and adviser validation
are inside full-job latency.

The pilot projected 819.254 seconds including a 20% margin and cleanup reserve,
against 1563.752 seconds remaining. The frozen main matrix therefore ran in
full: 48 conditions and 1,152 attention requests. The pilot added 192 numerical
requests; 84 numerical warmups supplied independently auditable output arrays.
All numerical results passed `atol=1e-5, rtol=1e-4` against the original FP64
reference, using 21 fresh numerical fixtures including pilot fixtures.

| Arrangement | Serial CPU + NPU | Overlap CPU + NPU | Serial GPU + NPU | Overlap GPU + NPU |
| --- | ---: | ---: | ---: | ---: |
| Persistent threads | 2.901 s | 2.842 s | 13.845 s | 13.351 s |
| Spawned processes | 2.922 s | 1.798 s | 13.848 s | 12.028 s |

These are mean full-job times for 24 attention requests plus one adviser
request. The primary thread/process CPU-overlap ratio was 1.5813, with paired
95% interval [1.4289, 1.7271]. Process GPU overlap improved over thread GPU
overlap by 1.1100× [1.0810, 1.1629], but remained 6.691× slower than process
CPU overlap. The six repetition blocks are the statistical units; the 1,152
numerical requests are not treated as independent experimental replicates.

Mean first CPU request wall time during overlap fell from 1132.4 ms with threads
to 45.2 ms with processes. Mean first GPU request wall time fell from 1116.0 ms
to 494.8 ms, while corresponding GPU timestamps stayed near 491–493 ms. This
supports a host-orchestration bottleneck; it does not prove a particular GIL or
SDK mechanism, or continuous simultaneous device execution.

The one shortened adviser prompt passed 7/20 development cases and 9/20
separate held-out cases. It was frozen before held-out execution and was not
revised in response to failures. Main advice passed in 16/48 responses; these
repeat six preassigned held-out cases across eight conditions, not 48 independent
language tasks. Invalid outputs remained in all denominators. Consequently no
application-benefit recommendation was accepted despite timing improvements.
The CPU remains the attention choice. Valid IDs would establish an interface
contract, not evidence that model-guided tuning beats deterministic selection.

All 163 final software tests pass. The independent offline audit regenerated
the 21 fixtures and FP64 references, checked the 84 saved output arrays against
all warmup and measured hashes, reconstructed exact coverage and worker/timing
order, and recomputed all ten confidence intervals. Its first pass caught a
report-label collision that omitted the cross-architecture GPU comparison;
`initial-audit.json` preserves that failed audit. The corrected report uses
distinct names, and a regression test protects all ten comparisons. No
measurement was rerun. Measurement-time code remains in `source/`, while
`analysis-source/` and `final-source/` preserve the corrected implementation.

Peak temperatures were CPU 61.7°C and HAT 40.806°C; every recorded throttle flag
was zero. All 29 worker instances shut down cleanly. The release record lists
unrelated desktop GPU owners and any processes whose descriptors could not be
inspected. Energy and GPU utilization were not measured.

`next-experiment.json` preserves failed adviser cases for future development,
requires new independent holdouts, and proposes GPU kernel profiling before
further tuning. Process separation is a promising timing control for that work;
reliable advice and a comparison against deterministic selection remain needed
before claiming useful recursive improvement. Re-running the frozen command is
a replication, not fresh independent evidence.

## Completed adviser interface validation — 2026-10-02

The user selected reliable NPU advice with a CPU-validated interface. The new
`validate-adviser` command completed `runs/adviser-validation-20261002` within
one 30-minute supervised allocation: 517.775 seconds (8.63 minutes), including
setup, controls, pilot, development, holdout and cooldown. The installed Qwen
model and generation parameters remained fixed. A persistent spawned worker
owned the HAT context; this experiment did not run GPU or attention kernels.

The interface accepts a complete eligible two-character ID, or canonicalizes
exactly one trailing ASCII period on an otherwise complete eligible ID. It does
not extract IDs from explanations, salvage incomplete generations, infer
suppressed output, or accept previously tested candidates. Invalid advice uses
the existing deterministic CPU selector and is explicitly labeled
`cpu_fallback`. An empty eligible set abstains without invoking the model.
Rendering, model IPC, verification, stop-setting/context restoration and any
fallback are included in the application request timer.

All eight native-stop capability controls passed. Each ID was tested with a
default/custom/default sequence. Custom stops were the original stop strings
plus all eight IDs, rather than only the currently eligible IDs. Each candidate
was returned intact with `LOGICAL_END_OF_GENERATION`, correct token accounting,
matching before/after default controls, and verified restoration. This supports
the measured native boundary behavior without assuming general grammar support.

The excluded pilot used three previously observed cases per model arm. It
projected 932.030 seconds for the remaining 200 generations, a 20% margin and
45-second cleanup reserve, against 1734.712 seconds remaining. All planned cases
were therefore retained. Development reused all 40 old cases, including the
former holdout. The prior prompt accepted 20/40 proposals (16 exact, four
canonicalized); structured chat examples accepted 6/40; the same examples with
native stops accepted 19/40 exact proposals. The bounded arm was selected over
the examples arm using the frozen development-only criteria. The old prompt
remained an independent comparator, not a selectable challenger.

After selection and parser/settings hashes were frozen, the run created 40
fresh nonempty holdout cases spanning eligible-set sizes 1–8 and eight empty-set
cases. Baseline, challenger and direct CPU selection each received all 48 cases
in a frozen balanced permutation schedule: 144 interface requests and 80 model
generations. Former holdout cases are never counted as held-out evidence in this
new experiment.

| Engine | Accepted NPU proposals | CPU fallbacks | Mean nonempty request |
| --- | ---: | ---: | ---: |
| Original prompt | 22/40: 18 exact, four canonicalized | 18 | 2256.889 ms |
| Examples with native stops | 18/40: all exact | 22 | 1811.883 ms |
| Deterministic CPU | Not a model arm; 40/40 direct eligible selections | 0 | 0.062584 ms |

All 24 empty-set interface requests abstained without inference. All 144 held-out
interface requests satisfied the selection/abstention contract. However, the
bounded challenger failed the 40/40 NPU reliability gate and did not improve
acceptance over the original prompt on this holdout. Its 40 raw responses were
all complete single IDs; 22 were ineligible. Native stopping solved formatting
in these cases but did not establish correct candidate selection. The baseline
fallbacks comprised 17 incomplete/context-invalid classifications and one
ineligible ID. No superior tuning decisions or application speedup is claimed.

All 188 software tests pass. The independent offline audit passed on its first
run, checking checksums, source snapshots, the development-only selection,
holdout freshness and freeze order, exact schedule coverage, response parsing,
CPU fallback decisions, worker ownership, settings, request timings and summary
counts. It independently classified all 337 decisions and verified all 233 model
request restorations. One HAT worker shut down cleanly. Peak temperatures were
CPU 54.0°C and HAT 43.942°C, and every recorded throttle flag was zero. The release
snapshot found no visible HAT holder; unrelated desktop GPU clients remained,
with inaccessible process descriptors explicitly recorded.

The report, CSV/JSON, PNG/SVG, raw responses, source snapshots and checksums are
retained. `next-experiment.json` records failures for development only and requires
new holdouts before another reliability claim. CPU selection remains the default;
native termination support is available as an experimental capability. The next
adviser study, if pursued, must address eligibility rather than merely format.

## Five-stage attention research campaign — 3 October 2026 UTC

The implemented campaign ran profile, GPU development, NPU context development,
combined scheduling development, and fresh confirmation. The machine-readable
[ledger](campaigns/attention-research-20261003/campaign.json) charges all attempts
against the two-hour stage cap and ten-hour campaign cap. Completed runs are
checksum-linked and immutable. No clocks, drivers, precision thresholds, model
weights, or application defaults were changed.

| Stage | Supervised minutes | Independent audit |
| --- | ---: | --- |
| profile | 36.275 | True |
| gpu | 12.932 | True |
| context | 25.327 | True |
| combined | 72.504 | True |
| confirm | 31.026 | True |

Total supervised time: 2.967727 hours.

The native extension preserves the original Metrics ABI and C0–C7 controls.
Versioned profiling separates projection, score, softmax and apply GPU intervals
from host copies, submission, waits and query retrieval. Causal variants skip
wholly masked score tiles and restrict useful softmax/apply work without changing
FP32 tolerances. The largest prefill GPU case improved from 508.364 to 383.577 ms
(1.325×; 95% fixture-block bootstrap interval 1.314–1.336), while its CPU control
was 45.335 ms. All twelve attention routes retained CPU execution.

Context development compared rebuild, retained context and checkpoint save/clear/load.
All 576 main answers had valid terminal/context records and exact cache parity.
Each mode scored 118/192 strict task answers. Mean four-turn latency was 11.474 s
for rebuild, 5.306 s for retain and 12.126 s for checkpoint. Checkpoint costs
include SDK serialization, hashing, memory guards, clearing and restoration.

The combined development matrix completed 180 main jobs, each with 24 attention
requests and one NPU follow-up. CPU/NPU overlap with retained context averaged
2.994 s for streaming versus 4.047 s serially, and 1.139 s for prefill versus
1.838 s serially. All tested GPU shares were slower. All NPU outputs were
equivalent across conditions, but strict answer accuracy was only 16.7%. These
are secondary development comparisons, not an independent promotion claim.

Fresh confirmation used eight new fixture blocks and 98.333% intervals across
three primary comparisons. All 576 standalone numerical requests and all 2,304
numerical requests inside combined jobs passed. Across the full campaign, 10,224
main numerical requests and 660 correctness/semantic control records passed.

The context candidate had an observed latency ratio of 2.189 (interval 2.073–2.296),
but failed the frozen complete-dialogue gate. One 512-token fixture reached the
16-token output limit on its first answer, identically in both modes across
three repetitions. Six invalid answers and 18 quarantined dependent turns remained
in the 576-answer denominator. All 69 fully completed pairs matched exactly;
each policy had 276 valid and 189 strictly correct answers out of 288 planned.
The failure occurred before context reuse and is not evidence of cache divergence.
Incomplete dialogue latency must not be presented as a valid complete-workload gain.

A separate fresh combined streaming fixture had invalid preludes. The new guard
cleared stale continuation state, skipped six dependent NPU generations and still
executed every numerical request. The raw failures stayed in the job denominator.
The final audit verified these skip/prelude links, actual generation counts, and
fixture-to-request transcript bindings. Unexpected SDK or safety failures still
stop the stage. No such stop occurred.

No new policy was promoted. The unchanged CPU comparison had a ratio of 0.927
(interval 0.818–1.059), illustrating baseline timing variability. Coordination
also compared unchanged policies and failed completion coverage on its prelude
fixture. Recorded fallbacks are CPU attention, rebuild context and overlap-0.
Coordination measurements used retained context; they do not independently validate
the performance of a new composition using the rebuild fallback.

Process improvements include durable budget accounting, fresh retry namespaces,
full tokenization costs inside request timers, read-only oracle loading, failure
quarantine, source/binary snapshots, independent transcript and selection audits,
and descendant-process memory/CPU telemetry. The last telemetry improvement was
introduced before combined development: sampled summed RSS peaked at 368.0 MB
there and 416.1 MB in confirmation. Shared pages can be counted more than once;
these figures do not establish the earlier checkpoint stage's worker-tree peak.
Campaign temperature peaks were CPU 68.3 C and HAT 46.32 C; recorded throttle
flags remained zero. No energy measurements were made.

The final software suite passed 216 tests. The optional compatible secondary
Qwen2 model download returned HTTP 403, so cross-model replication is deferred.
The [overview](campaigns/attention-research-20261003/report.html),
[final validation](campaigns/attention-research-20261003/final-validation.json), and
[next experiment](campaigns/attention-research-20261003/next-experiment.json)
contain the current evidence and proposed follow-up work. The next priority is
completion and task quality, followed by single-query GPU specialization, finer
GPU shares, native checkpoint serialization and cached CPU projection scratch.
These are proposed experiments, not executed improvements.

## CPU-GPU reliability recovery — 4 October 2026 UTC

The [CPU/GPU recovery campaign](campaigns/cpu-gpu-recovery-20261004T130703Z/report.html)
completed CPU development, fresh CPU confirmation and fresh GPU confirmation.
All three scientific audits passed, and the campaign and run artifacts are
sealed. The [final validation](campaigns/cpu-gpu-recovery-20261004T130703Z/final-validation.json)
also records 18 rejected deliberate tamper cases and preservation of 54 earlier
sealed directories. The two authentication failures described below remain
incomplete infrastructure attempts with `audit_passed: false`; their preservation
does not turn them into successful scientific runs.

All numerical comparisons used the same six streaming shapes: `N=128/512/1024`,
`D=64`, and batch size `1/4`. CPU routing was `native4` for batch four at
`N=512/1024`, and `native1` for the other shapes. The original FP64 oracle on the
FP32 inputs and fixed `atol=1e-5`, `rtol=1e-4` requirements remained in effect.
The paired bootstrap resampled complete fixture blocks; repetitions and the
sixteen calls within a worker batch did not become independent samples.

CPU development compared four passive-wait policies on 128 fresh blocks per
policy and selected policy 5: `performance` governor, bound threads and passive
waiting. [CPU confirmation](runs/cpu-gpu-recovery-20261004T130703Z-cpu-confirm-retry1/summary.json)
used 256 new blocks. Numerical correctness passed, and the paired 95% intervals
for duplicate controls fell wholly within 0.95–1.05, both overall and for every
shape, for individual requests and sixteen-request batches. Aggregate intervals
were [0.997155, 1.001499] for individual requests and [0.999627, 1.001362] for
batches. This establishes correctness and repeatability of the selected policy
on the measured workload. Fresh confirmation did not compare its speed against
the old application default or every other development policy.

[GPU confirmation](runs/cpu-gpu-recovery-20261004T130703Z-gpu-confirm-retry1/summary.json)
used 64 new blocks with the confirmed CPU policy. Stream64 improved mean request
time by 2.857821× relative to the baseline GPU implementation, with paired 95%
bootstrap interval [2.850774, 2.863870]. It passed correctness, the CPU
duplicate-control gate, the minimum-gain gate and the per-shape slowdown guard.
Aggregate mean request times were 1,085.629 ms for the baseline GPU, 379.880 ms
for stream64, and 17.090 ms for the duplicate-averaged CPU comparator. Thus
stream64 still required 22.227635× the CPU time. GPU improvement alone did not
justify moving this workload away from the faster CPU route.

The [campaign ledger summary](campaigns/cpu-gpu-recovery-20261004T130703Z/summary.json)
charges 15,175.369814 seconds against the 28,800-second scientific allowance:

| Attempt | Charged seconds | Result |
| --- | ---: | --- |
| CPU development | 7,740.188031 | Candidate selected; scientific audit passed |
| Initial CPU confirmation | 125.193965 | Desktop authentication unavailable; preserved |
| CPU confirmation retry | 4,260.146811 | Confirmed; scientific audit passed |
| Initial GPU confirmation | 125.206140 | Desktop authentication unavailable; preserved |
| GPU confirmation retry | 2,924.634866 | Confirmed; scientific audit passed |

Both startup failures occurred before the supervised numerical matrix began.
Explicit continuations used fresh output directories and fixture namespaces,
retained those charges, and used the remaining stage allowances. Completed CPU
development was not rerun. A separate preliminary diagnostic passed within its
own twenty-minute allowance and contributed no scientific qualification. The
durable user service owned the work independently of the launching assistant
turn; authentication was still required for each governor lease.

The governor was restored to `ondemand`, and application defaults remain
unchanged. The scope is synthetic streaming attention on this Pi and its recorded
runtime. No end-to-end LLM inference improvement, `ask`/`query` application
improvement, or energy saving was established. No NPU or combined-system stage
ran in this recovery campaign; the prior full-table NPU qualification failure
remains closed. Historical source snapshots, failed attempts and sealed evidence
retain their original contents and interpretation.


## Completion implementation — 6 October 2026 UTC

The user authorized the saved completion roadmap and selected a concentrated
execution schedule. EXECUTION-PLAN.txt records engineering estimates separately
from the fixed hardware allowances. New CPU retry accounting preserves the sealed
5 October authentication failure and exact remaining original allowance.

New GPU and NPU protocols implement the bounded tensor-batch study and full
four-way diagnostic. Their independent auditors reconstruct numerical or raw
answer decisions; simplified NPU arms do not qualify the full-table task. Shared
stage reservations, source freezes, conservative interruption accounting and
explicit parent links prevent silent resets and scientific retries.

This entry records implementation, not new measured results. Read
COMPLETION-STATUS.json for execution progress and subsequent sealed reports for
research dispositions. Existing defaults and historical evidence remain unchanged.

## Scientific closure — 6 October 2026 UTC

The original roadmap gates were retained. All executed scientific campaigns
completed and passed their independent evidence audits, while their candidates
failed promotion. Scientific completion records these negative outcomes and
explicitly closes dependent work; it does not imply all devices were beneficial.

The [CPU comparison](runs/completion-cpu-20261006T120615Z-compare/report.html)
completed its excluded pilot and all 256 paired main blocks. The 16-call batch
endpoint passed at 1.1904530968147335x speedup, paired 95% interval
[1.183801798749645, 1.1973401880227976], with valid numerical outputs and stable
duplicate controls. Individual calls measured 1.2122292669161894x, interval
[1.1877927808542437, 1.2375647701555528], but the baseline duplicate-control
intervals for N512/B4 [0.9917682581391661, 1.1748498919490753] and N1024/B4
[0.897798828908681, 0.9933082727252065] were not contained in [0.95, 1.05].
Both endpoints had to qualify. No CPU preset was exported; the single-call
speedup is an observed estimate with a failed stability requirement.

The [GPU development study](runs/completion-gpu-develop-20261006T145314Z-develop/report.html)
completed all twelve N128/512/1024, D64/128, tensor-batch-1/16 shapes. Stream64
improved on the original GPU by 3.44753179879827x, interval
[3.4400043599178804, 3.4544842021959563], but remained 8.402516429558752 to
21.154191621307405 times the CPU duration across tested shapes. Numerical
correctness passed, CPU duplicate controls failed the stability gate, and all
twelve declared routes remained CPU. No GPU route qualified or advanced to
confirmation. Dynamic dispatcher overhead was not measured; this was a bounded
shape-specific route comparison, not a universal GPU performance conclusion.

The [NPU diagnostic](runs/completion-npu-diagnostic-20261006T112430Z-diagnostic/report.html)
observed strictly correct/planned answers of 44/96 for full-table actual history,
80/96 for full-table independent questions, 55/96 for selected-fact actual
history, and 94/96 for selected-fact independent questions. The last three arms
were diagnostic; they could not qualify the original task or dependent work.
The observations informed exactly one frozen full-table revision.

That [revised policy](runs/completion-npu-develop-20261006T155952Z-develop/report.html)
produced 96/96 valid completions with zero contract failures, but only 64/96
strictly correct answers: 22/32, 19/32 and 23/32 for the original-control-sized
128/512/1024 groups. Development required at least 92/96 and 29/32 in every group.
The unchanged native-stop full-table control scored 3/96 strictly correct with
24/96 valid completions; invalid and quarantined turns stayed in the denominator.
Improvement over this control did not override the absolute accuracy gate.
No second revised candidate or fresh confirmation was attempted.

GPU and NPU confirmation, both cache stages and both combined stages were not
run because prerequisite gates failed. Their allocations remained unspent.
The conditional cache/combined API outline is planning evidence only. It records
no cache or combined hardware measurement and establishes no useful policy.

The ledger includes the sealed 5 October authentication charge
132.87611606300925 seconds, the additional 6 October authentication failure
131.85488570400048 seconds, and the successful measurement attempt
9318.376277084986 seconds. CPU total is 9583.107278851996 of 14400 seconds.
GPU development used 3644.4632362799894 of 7200 seconds. NPU diagnosis used
2393.713573957997 and qualification-development used 1778.2637786969717 seconds,
totalling 4171.977352654969 of the shared 7200-second development bucket.
Every reservation is finalized; no time moved between buckets. Full exact
remaining amounts are in [COMPLETION-RESULTS.txt](COMPLETION-RESULTS.txt) and the
canonical ledger. The final governor is restored to `ondemand`; defaults remain
unchanged. Authentication failures remain infrastructure records, not negative
scientific measurements.

The implementation-validation receipt records 589 passing regression tests.
Separate checkpoint validation records 47 passing disposable checks: 21 verifier
cases and 26 orchestration tests. At source-snapshot time the new independent USB package was still pending.
The [closure index](campaigns/completion-closure-20261006T163315Z/) was subsequently
sealed and reviewed. The dated verification entry below records the completed
backup using its external receipt; archived source-snapshot bytes are unchanged.

## Independent backup verified — 6 October 2026 UTC

Checkpoint `20261006T164951Z` was packaged at 2026-10-06T17:06:57.749127+00:00, actually
restored from USB into a new empty SSD directory at 2026-10-06T17:22:02.816935+00:00, and
certified at 2026-10-06T17:25:42.471260+00:00. Its exact 19924-entry inventory and
all 74 canonical seals passed. [Checkpoint details](CHECKPOINT-20261006T164951Z.txt)
identify the independent destination and authoritative `BACKUP-RECEIPT.json`.

The archive preserves the pending status that existed before restoration. Only
the live handoff documents were updated afterward; their hashes and the verified
receipt are recorded in `[HOME]/Documents/completion-checkpoint-work-20261006T164951Z/LIVE-HANDOFF-UPDATE.json`. Scientific code, budgets, sealed
evidence and all archives are unchanged. This closes the requested handoff.
