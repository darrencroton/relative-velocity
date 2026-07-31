# Merger-Rate Plan (Revised): Eight-Run Model & Harness Comparison

**Report question:** Across eight PM Mode-B runs of `docs/MERGER_RATE_PLAN-REVISED.md`, which developer model/harness combinations produced the best code, and how should frontier (paid, time-limited) and local (free, unlimited) models be combined for moderate-to-complex coding tasks?

**Method note:** Section A was completed in full — including all harness runs, cross-testing and probes — *before* any `.pm/runs` report was opened, so the code-quality assessment is independent of the PM's own conclusions. Section B then compares the two.

---

## Executive Summary

All eight branches implement Slices 1–3 of the plan, and all eight pass their own test suites (191–246 tests, 0 failures). Against an independent 100-test conformance harness written from the plan's acceptance criteria alone, six branches scored 100/100 and two scored 99/100. **The plan format worked: every branch covered every named exact-value criterion.** Differentiation is therefore almost entirely in code quality, not feature completeness.

**Top of the ranking is genuinely close; the bottom is not.**

| Rank | Branch | Developer model | Weighted score /35 | One-line verdict |
|---|---|---|---|---|
| 1 | `claude-sonnet-1` | claude-sonnet-5 (high) | **32** | Cleanest validation, fewest confirmed defects, fastest run |
| 2 | `claude-sonnet-2` | claude-sonnet-5 (high) | **31** | Best-documented and most economical; one real coercion defect; **Slice 3 never accepted** |
| 3 | `opencode-qwen27b16-1` | qwen3.6-27b-bf16 | **30** | Only branch with zero confirmed correctness defects; ugly but right |
| 4 | `opencode-qwen27b16-2` | qwen3.6-27b-bf16 | 27 | Solid; scope creep and internal duplication |
| 5 | `opencode-qwen27b8-1` | qwen3.6-27b-q8 | 26 | Correct science; heaviest house-style drift; live warning in own suite |
| 6 | `opencode-qwen27b8-2` | qwen3.6-27b-q8 | 25 | Worst DRY violation; 100 % message-pinned tests |
| 7 | `opencode-qwen35b16-2` | qwen3.6-35b-a3b-bf16 | 24 | Reintroduces the cancellation-prone form the plan warned against |
| 8 | `opencode-qwen35b16-1` | qwen3.6-35b-a3b-bf16 | **15** | Multiple confirmed defects, dead code, leaks 37 temp dirs per test run |

**Four conclusions that matter most:**

1. **Local 27B models reached frontier-adjacent quality — at 3–5× the wall-clock and 3× the supervision.** `qwen27b16-1` has the *best* confirmed-correctness record of all eight. But it took 6h09m and 8 PM steers versus claude-sonnet-5's 54 min and 2 steers.
2. **The 35B-a3b MoE is the trap.** It is ~2× *faster* than the dense 27B (it activates only ~3B params) and ranked 7th and 8th. Speed on a local model is not a proxy for capability, and here it inversely tracked quality.
3. **The frontier model's real advantage showed up in the PM seat, not the diff.** The `claude-sonnet-2` run is the only one that *stopped for a human* — it detected a genuine defect in the frozen plan (numerically degenerate predictors) that all six local runs shipped without noticing. That detection, not the code, is the highest-value output of the eight runs.
4. **The mechanical floor and lint found essentially nothing.** Across the four runs with model-performance notes, floor was 8/8 at every finalize (including on a tree carrying a P1 sign error) and lint was clean on every attempt. Every defect that mattered came from a human-equivalent read of contract against code.

---

## Section A — Independent Code-Quality Analysis

### A.1 Rubric

Six dimensions, 1–5 (5 = excellent). `Correctness` is double-weighted in the total, because for scientific code a wrong number is categorically worse than an ugly one.

| Dimension | What it scores |
|---|---|
| **Correctness** | Does it satisfy the frozen contract, including binding validation conventions and edge behaviour? |
| **Readability** | Naming, structure, comment quality, house-style consistency, accurate documentation |
| **Maintainability** | DRY, dead code, scope discipline, cost of the next change |
| **Testing & Reliability** | Coverage of the acceptance criteria, test rigour, isolation, brittleness |
| **Efficiency** | Idiom, wasted work, latent performance traps |
| **Security** | Input handling, failure modes, no unsafe coercion or silent acceptance |

### A.2 Evidence gathered

| Instrument | What it produced |
|---|---|
| Each branch's own suite | 191–246 tests, **0 failures on all 8** |
| **Conformance harness A** (100 tests, written from the plan's ACs only) | 6 branches 100/100; `claude-sonnet-2` and `qwen35b16-1` 99/100 |
| **Conformance harness B** (19 valid edge tests) | 2 branches 19/19; 6 branches 18/19 |
| **Cross-test matrix** (56 cells: every suite × every other implementation) | Isolated 2 low-coupling "referee" suites |
| **Direct behavioural probes** | Confirmed/refuted 4 candidate defects per branch |
| `ruff` (E,F,W,B,C90,SIM,PIE,RET,ARG,PL) | src near-identical (7–9); tests 16–65 |
| Micro-benchmarks | Settled that alleged efficiency defects are *not* measurable at real scale |

One harness test (`D11`, weight-normalisation overflow) was **discarded as invalid** — the plan explicitly declares extreme-magnitude overflow unspecified, so failing it is not a defect. Reporting it would have been unfair to all eight.

### A.3 Scores

| Branch | Corr. | Read. | Maint. | Test | Effic. | Sec. | Raw /30 | **Weighted /35** |
|---|---|---|---|---|---|---|---|---|
| `claude-sonnet-1` | 5 | 5 | 4 | 4 | 5 | 4 | 27 | **32** |
| `claude-sonnet-2` | 4 | 5 | 5 | 4 | 5 | 4 | 27 | **31** |
| `opencode-qwen27b16-1` | 5 | 3 | 3 | 5 | 4 | 5 | 25 | **30** |
| `opencode-qwen27b16-2` | 4 | 3 | 3 | 5 | 4 | 4 | 23 | **27** |
| `opencode-qwen27b8-1` | 4 | 3 | 2 | 5 | 4 | 4 | 22 | **26** |
| `opencode-qwen27b8-2` | 4 | 3 | 2 | 4 | 4 | 4 | 21 | **25** |
| `opencode-qwen35b16-2` | 4 | 3 | 2 | 4 | 3 | 4 | 20 | **24** |
| `opencode-qwen35b16-1` | 2 | 2 | 2 | 2 | 2 | 3 | 13 | **15** |

### A.4 Confirmed defects, by branch

Each row was reproduced by direct execution, not inferred from reading.

| Defect | Severity | Branches affected | Evidence |
|---|---|---|---|
| `_load_pair_counts` coerces `float(attr)` before form check → **TypeError**, violating both "validate form before coercion" and "no new TypeError" | **P2** | `claude-sonnet-2`, `qwen35b16-1` | Harness A `C1_16`; confirmed independently by *both* referee suites |
| `compute_merger_rate` **silently accepts the string `"500.0"`** as `box_size_mpc` and returns a plausible rate | **P2** | `claude-sonnet-1`, `qwen35b16-1` | Direct probe. This is the exact failure mode the plan names by example. Mitigating: strings are not in this function's *listed* rejection set, so this is out-of-domain, not a criteria breach |
| Degenerate predictor (collapsed `log10(1+z)`) **raises AssertionError** where the plan says it must return `(nan, nan, nan, n_excluded)` | **P1** | `qwen35b16-1` | Direct probe; `assert s_xx_c > 0` on a frozen must-not-raise boundary |
| Degenerate predictor **fabricates a finite fit** (slope ≈ 3.5e16) | **P2** | `claude-sonnet-2`, `qwen27b16-2`, `qwen27b8-1` | Direct probe. This is the same defect class the `claude-sonnet-2` PM stopped the run over |
| `merger_timescale_gyr` leaks `OverflowError` instead of asserting its own result is finite | **P3** | all except `qwen27b16-1`, `qwen27b8-1` | Harness B `D14`. Extreme magnitudes, so largely out-of-domain |
| **Test isolation violated**: 30× `tempfile.mkdtemp()`, zero `tmp_path`, no cleanup → **37 leaked temp dirs per run** | **P2** | `qwen35b16-1` | Measured empirically with a controlled `TMPDIR` |
| `fit_log_rate_vs_redshift` asserts `isinstance(rates, (list, tuple, ndarray))` — forbidden extra validation that rejects declared-valid array-likes | **P3** | `qwen27b8-2` | Code read; the plan says "do not reject any declared-valid value" |
| Live `RuntimeWarning: divide by zero` emitted by the branch's **own passing test suite** | **P3** | `qwen27b8-1` | Captured in the suite run at `merger_rate.py:250` |

**Only `qwen27b16-1` carries no confirmed defect from this table** — and it is also the only branch that returns the correct `nan` tuple on the degenerate-predictor case.

### A.5 Non-defects worth recording

Honest negatives, because they change the scoring:

- **`claude-sonnet-1` did not regress the `datetime.utcnow()` timestamp.** It branched from `6d1ecad`, *before* main's lint-cleanup commit `d567671`, and inherited the deprecated call. This is a **baseline confound**: it is the only branch not compared like-for-like, and the DeprecationWarning in its test run is not its fault.
- **The "Python loop instead of `np.bincount`" criticism does not survive measurement.** At 6 bins on 2 M galaxies: `bincount` 4.0 ms, `np.add.at` 3.3 ms, Python loop 4.9 ms — all dwarfed by `np.digitize` at ~31 ms. It is an idiom point, not a performance defect, and I scored it that way.
- **`check_slope_consistency` returns a builtin `bool` in all eight branches.** I suspected `np.bool_` leakage in the 35B branches; the probe refuted it.

### A.6 Quality signals not captured by tests

| Signal | Best | Worst |
|---|---|---|
| **Module docstring accuracy** | `claude-sonnet-2` — fully rewritten to describe all three slices | `qwen35b16-1`: *"Slice 1 only covers the denominator-and-fraction layer… added by later slices"* — actively false. `qwen35b16-2` and `claude-sonnet-1` similarly stale |
| **Dead code** | `claude-sonnet-*`, `qwen27b16-2` — none | `qwen35b16-1`: `np.asarray(x, dtype=object)` ×3 immediately overwritten (a 17,000× slower conversion, latent if ever reused at scale), a no-op `both_zero` block, and unused `_cov_00`/`_cov_01`. `qwen35b16-2` also carries unused covariance terms **named with leading underscores to dodge the linter** |
| **DRY** | `claude-sonnet-2` — helper-factored | `qwen27b8-2`: three verbatim 12-line scalar-validation blocks (~60 lines). `qwen27b16-2` recomputes `n_mass_bins` inline 3× while its own `_mass_bin_edges` sits unused |
| **House style (`assert`, per plan)** | `claude-sonnet-*`, `qwen35b16-2` | `qwen27b8-1` — `merger_timescale_gyr` is entirely `if/raise AssertionError`; drift is pervasive |
| **API sanity** | index-valued `mass_bin` key | `qwen27b16-2` and `qwen27b8-2` return `mass_bin` as a **string label** (`"[8.0, 8.5)"`), not the bin index |
| **Diff hygiene** | `claude-sonnet-*`, `qwen27b8-*` | `qwen35b16-1/-2`, `qwen27b16-2` gratuitously realigned pre-existing `calc.py` lines the plan declared unchanged |
| **Broken error message** | — | `qwen35b16-1` emits the literal text `{n_expected - 1}` (missing `f` prefix) |

### A.7 Test-suite character

The cross-test matrix's headline numbers are dominated by error-message pinning, so the useful metric is **coupling**, not raw failures.

| Branch | Tests | Test LOC | `pytest.raises` with `match=` | Coupling | Failures induced elsewhere (mean) | Reading |
|---|---|---|---|---|---|---|
| `qwen27b8-1` | 166 | 2072 | 0 / 77 | **0 %** | 9.9 | Most portable; used as referee |
| `qwen27b16-2` | 125 | 1495 | 1 / 66 | **2 %** | 4.6 | Most portable; used as referee |
| `qwen27b16-1` | 156 | 1889 | 20 / 78 | 26 % | 16.6 | Balanced |
| `qwen35b16-2` | 113 | 1611 | 29 / 54 | 54 % | 21.7 | Moderate |
| `qwen35b16-1` | 126 | 1935 | 35 / 41 | 85 % | 20.6 | Brittle **and** leaks temp dirs |
| `claude-sonnet-1` | 112 | 1224 | 66 / 73 | 90 % | 48.0 | Strictest, most brittle |
| `claude-sonnet-2` | 74 | 1161 | 37 / 39 | 95 % | 7.4 | Brittle but most economical (12 `parametrize`) |
| `qwen27b8-2` | 138 | 1733 | 73 / 73 | **100 %** | 37.1 | Any message reword breaks 73 tests |

**Interpretation.** Heavy `match=` use is a genuine trade: it satisfies the plan's "assertion naming the reason" criteria, but it welds the suite to exact strings. `claude-sonnet-2` gets the best of both — 74 tests covering the same ground as `qwen27b8-1`'s 166, via `parametrize` — the highest coverage-per-line in the set.

**All eight branches covered every named exact-value acceptance criterion** (`0.22360679774997896`, `1.090909090909091e-08`, `0.2040278893193579`, the SHA-256 sentinel matrix, the `[1,1,0,0,0,1]` count vector, the unweighted-`polyfit` comparison, the `250.0` box-size provenance fixture). That uniformity across a 20× spread in model capability is the strongest evidence in this report that **frozen numeric criteria are what carried the weak models**.

---

## Section B — PM Run-Report Meta-Analysis

### B.1 Run facts

| Branch | Developer model | Reviewer | Wall clock | Attempts (S1/S2/S3) | Events | Status |
|---|---|---|---|---|---|---|
| `claude-sonnet-1` | claude-sonnet-5, effort=high | codex gpt-5.6-sol | **53m 53s** | 1 / 0 / 1 = **2** | 33 | complete |
| `claude-sonnet-2` | claude-sonnet-5, effort=high | codex gpt-5.6-sol | 1h 46m 09s (~35 m lost to a reviewer timeout) | 1 / 1 / 1 = **3** | 39 | **STOPPED — human decision** |
| `qwen27b16-1` | qwen3.6-27b-bf16 | codex gpt-5.6-sol | **6h 08m 49s** | 4 / 3 / 1 = **8** | 86 | complete |
| `qwen27b16-2` | qwen3.6-27b-bf16 | codex gpt-5.6-sol | 3h 49m 02s | 4 / 2 / 2 = **8** | 65 | complete |
| `qwen27b8-1` | qwen3.6-27b-q8 (~14–15 TPS) | codex gpt-5.6-sol | 3h 45m 36s | 1 / 2 / 2 = **5** | 62 | complete |
| `qwen27b8-2` | qwen3.6-27b-q8 | codex gpt-5.6-sol | 3h 59m 30s | 3 / 2 / 1 = **6** | 61 | complete |
| `qwen35b16-1` | qwen3.6-35b-a3b-bf16 | codex gpt-5.6-sol | 1h 46m 43s | 2 / 2 / 2 = **6** | 89 | complete |
| `qwen35b16-2` | qwen3.6-35b-a3b-bf16 | codex gpt-5.6-sol | 2h 47m 25s | 2 / 2 / 3 = **7** | 62 | complete |

Reviewers were always `codex gpt-5.6-sol`, commissioned at **low effort for drift-audit** and **medium effort for code-review**. No run exceeded 8/10 attempts on any slice; the mechanical floor passed 8/8 at every finalize in every run.

### B.2 The PM's own scores (four runs carry `model-performance.md`)

| Developer | Process | Reporting | Output | PM's summary |
|---|---|---|---|---|
| `qwen27b8-1` | 3/5 | **5/5** | 3/5 | "science is right… but under-tests the specific thing an acceptance criterion is protecting" |
| `qwen27b8-2` | 4/5 | 4/5 | 3/5 | "recurring weakness is test rigor rather than implementation… repeatedly wrote tests that would pass under the very regression they were meant to catch" |
| `qwen35b16-1` | **2/5** | **2/5** | **2/5** | "narration was always plausible and twice materially wrong… required a supervisor who independently recomputed the science" |
| `qwen35b16-2` | 3/5 | **2/5** | 3/5 | "reports confidently on work it has not actually proven. Every number it produced that I checked independently, I had to check" |

The PM's ordering (`27b8-*` ≈ `35b16-2` > `35b16-1`) **matches mine**, arrived at independently.

### B.3 Where the PM and I agree

- **`qwen35b16-1` is the worst run.** PM: 2/5/2/5/2/5. Mine: 15/35. Both flagged the same class — plausible narration over unproven work.
- **The `_load_pair_counts` form-before-coercion defect is real.** The `claude-sonnet-1` PM steered it explicitly in Slice 1; my harness independently flagged it in the two branches where it *wasn't* steered.
- **The degenerate-predictor problem is real and is a plan defect.** The `claude-sonnet-2` PM stopped the run for it; my probes found the same class in 6 of 8 branches.
- **Reviewers trend toward defensive-coding maximalism.** PM rejected ~⅓ of code-review findings on cited plan text. My discarded `D11` test made the identical error, which is a useful check on the reviewer's calibration rather than a criticism of it.

### B.4 Where I found things the PM did not

| Finding | Why the PM missed it |
|---|---|
| **`qwen35b16-1` leaks 37 temp dirs per test run** | The PM *did* see `mkdtemp` and ruled it "a benign mechanism deviation that satisfies the binding requirement", verifying only that the repo's `data/`/`results/` were untouched. It checked the clause's *intent* but not its *hygiene consequence*. The plan named `tmp_path`/`tmp_path_factory` specifically; those are bounded and auto-reaped, `mkdtemp` is neither |
| **Stale/false `src` module docstrings** in 5 branches | The `qwen35b16-1` PM noticed the *test* module docstring being edited but never checked whether the *production* docstring still described the module |
| **`np.asarray(..., dtype=object)` dead code ×3** | Passes `ruff` (the value *is* assigned and later reassigned). No PM mentions it in any run |
| **`mass_bin` returned as a string label** in two branches | Key-name checks passed; nobody checked the value's type |
| **`qwen27b8-1`'s live `RuntimeWarning` in its own green suite** | Warnings are not failures; nothing in the floor or lint surfaces them |

### B.5 The most important cross-run finding: a PM adjudication that flipped

The **same defect** received **opposite rulings in two different runs by the same PM contract**:

- **Run `20260730T021716Z` (`claude-sonnet-1`), Slice 1 steer:** *"`_load_pair_counts` must reject a non-scalar `box_size_mpc` attr with an assertion, not a TypeError… This violates two binding conventions."* → **Steered and fixed.**
- **Run `20260731T072213Z` (`qwen35b16-1`), Slice 1 review round 2:** the reviewer raised it; PM replied *"factually wrong on the pinned numpy 2.5.1 — I checked empirically that `float()` raises TypeError on every non-0-d array, so the value is rejected loudly."* → **Rejected.**

Both statements about `float()` are true. The disagreement is whether "raises TypeError" counts as rejection when the plan says *"Do not introduce TypeError / ValueError raises for new input validation."* The first PM read it as a violation; the second read it as sufficient.

**This is the single highest-leverage process finding in the eight runs.** It is not a model-capability difference — the defect shipped in `qwen35b16-1` and in `claude-sonnet-2` purely because of which way the adjudication fell. Supervision variance, not developer variance, decided the outcome.

### B.6 What the assurance layers actually caught

Aggregated from the four runs with model-performance notes, and consistent with the other four:

| Layer | Findings contributed | Verdict |
|---|---|---|
| Mechanical floor (8 facts) | **~0** — 8/8 at every finalize in all 8 runs, *including on a tree carrying a P1 intercept sign error* | Necessary hygiene; zero discovery value |
| Differential lint (`ruff`) | **~0** — clean on every attempt in every run | The `qwen35b16-2` PM: *"zero of the eleven findings across this run were mechanical"* |
| drift-audit (codex, **low** effort) | High, cheap | Caught the P1 intercept sign error, unauthorized components, three "could-not-fail" test classes. Best value-per-dollar in the system |
| code-review (codex, **medium** effort) | High, with noise | Caught the `np.bytes_` preflight bypass that would have destroyed a prior results file, and the vacuous-pass hole in the headline scientific test. But ~⅓ of findings were rejectable on cited plan text |
| **PM's own verification** | **Highest** | Every run's decisive numerical check was the PM's, because *neither reviewer can execute pytest* (Reviewer Mode forbids state-changing commands) |

**The load-bearing element is the PM seat.** Reviewers cannot run code; the floor and lint find nothing; the developer's self-report is unreliable on the weaker models. Everything rests on a supervisor who independently recomputes the science.

### B.7 Trends, correlations and variance

**Learning within a run is real and large.** Steers fall monotonically across slices in most runs, driven by the PM's curated `notes.md` carry-forward:

| Run | S1 → S2 → S3 attempts |
|---|---|
| `qwen27b16-1` | 4 → 3 → 1 |
| `qwen27b8-2` | 3 → 2 → 1 |
| `qwen27b16-2` | 4 → 2 → 2 |
| `claude-sonnet-1` | 1 → 0 → 1 |

The `qwen27b8-2` PM states it directly: *"first-attempt quality improved sharply once the curated notes carried the conventions forward."* **The notes file is doing measurable work** — a weak model with good carry-forward memory converges toward a strong model's first-attempt quality by slice 3.

**Quantisation cost nothing measurable here.** q8 vs bf16 on the same 27B model: q8 was *faster* (3h45m/3h59m vs 6h09m/3h49m) and needed *fewer* attempts (5, 6 vs 8, 8), while scoring within 4 points on my rubric (26, 25 vs 30, 27). With n=2 per arm this is not significant, but there is **no evidence that bf16 bought quality** on this task.

**Active-parameter count predicts quality better than total parameter count.** The 35B-a3b MoE (~3B active) is the fastest local model and the worst-scoring, ranking 7th and 8th. Total-parameter headline numbers are misleading for code work.

**Run-to-run variance within a model is substantial.** `qwen35b16-1` (15) vs `qwen35b16-2` (24) is a 9-point spread on identical model and plan — larger than the gap between the best local model and the frontier model. **A single run is not a measurement.**

**Wall clock does not track quality.** Correlation across the eight runs is weak and, within the local models, negative: the slowest run (`qwen27b16-1`, 6h09m) is the best local result, and the fastest local runs (`qwen35b16-1`, 1h47m) is the worst.

---

## Statistics

| Metric | claude-sonnet-5 (n=2) | qwen27b-bf16 (n=2) | qwen27b-q8 (n=2) | qwen35b-a3b-bf16 (n=2) |
|---|---|---|---|---|
| Mean wall clock | **80.1 min** | 298.9 min | 232.6 min | 137.1 min |
| Mean attempts | **2.5** | 8.0 | 5.5 | 6.5 |
| Mean weighted score /35 | **31.5** | 28.5 | 25.5 | 19.5 |
| Mean `src/merger_rate.py` LOC | **568** | 627 | 583 | 779 |
| Mean test LOC | **1193** | 1692 | 1903 | 1773 |
| Mean final test count | 193 | 226 | 232 | 200 |
| Mean ruff findings in tests | **23** | 45 | 61 | 44 |
| Slowdown vs frontier | 1.0× | 3.7× | 2.9× | 1.7× |

**Cost-of-supervision proxy:** attempts per accepted slice — claude-sonnet-5 **0.83**, qwen27b-q8 1.83, qwen35b-a3b 2.17, qwen27b-bf16 2.67.

**Verbosity penalty:** the local models wrote 10–37 % more production code and 42–60 % more test code for identical functionality. `qwen35b16-1` wrote **45 % more production code than `claude-sonnet-2`** and scored half as well.

---

## Recommendations

### Pure code quality, time no object

1. **`claude-sonnet-5` at high effort as Developer** — best score, fewest defects, and (decisively) the only configuration that *detected a defect in the plan itself*.
2. **`qwen3.6-27b-bf16` is a credible substitute** where cost dominates. It produced the cleanest correctness record of the eight. Budget ~4–6 hours and ~8 steers per three-slice plan, and expect to pay for it in readability and naming, not in wrong answers.
3. **Never `qwen3.6-35b-a3b` for contract-bound work.** Both runs ranked bottom-two; its reporting reliability was scored 2/5 twice.

### Efficiency view

| Scenario | Recommendation |
|---|---|
| Interactive / on the clock | claude-sonnet-5 Developer, codex reviewers. ~1 h per three-slice plan |
| Overnight / unattended | **qwen3.6-27b-q8** Developer. 2.9× slower, 25/35, and cheaper per attempt than bf16 |
| Maximum local quality, cost irrelevant | qwen3.6-27b-bf16. Accept 3.7× wall clock |
| **Hybrid (recommended default)** | **Local 27B Developer + frontier PM + codex reviewers.** The PM seat is where frontier capability pays: it is the only layer that runs code, and it is where the two shipped defects were decided |

### The specific hybrid to adopt

The evidence points at one allocation:

- **Developer:** local `qwen3.6-27b` (q8 for throughput, bf16 for quality). Free, unlimited, and — given frozen numeric acceptance criteria — reaches within 2–5 rubric points of frontier.
- **PM:** frontier model, high effort. **Do not economise here.** §B.5 shows an adjudication coin-flip decided whether a defect shipped, and §B.6 shows the PM is the only layer that executes code.
- **Reviewers:** codex at low effort for drift-audit (best value-per-dollar in the system) and medium for code-review (accept that ~⅓ of findings will be rejectable).

---

## Recommended Next Tests

Ordered by expected information gain.

1. **Resolve the plan defect first.** `claude-sonnet-2`'s Slice 3 is still unaccepted pending a human decision on the degenerate-predictor contract (`plan:549–553`). My probes show 6 of 8 branches mishandle it. **Amend the plan before running anything else** — every future run will rediscover it. The PM's recommended amendment (declare a minimum resolvable redshift separation and return the `nan` tuple below it, and require centring *both* variables) is sound.
2. **Close the adjudication inconsistency.** Add an explicit line to the plan's Validation Conventions: *"a TypeError leaking from an unguarded coercion is not a valid rejection, even though it fails loudly."* This one sentence would have prevented both shipped instances.
3. **Cross-model PM test (highest value).** Run the *same* Developer model (`qwen27b-q8`) twice with a frontier PM and twice with a local PM. §B.5 predicts the PM seat dominates outcome variance; this isolates it. Nothing else in this dataset tests it, because the PM was frontier in all eight runs.
4. **n=4 per model arm.** The 9-point within-model spread (`qwen35b16-1` vs `-2`) exceeds the between-model gap. Current n=2 cannot distinguish model quality from run luck.
5. **Add a mechanical hygiene gate for what lint missed.** Three of my findings — leaked temp dirs, `RuntimeWarning`s in a green suite, `dtype=object` dead code — are cheaply automatable: run pytest with `-W error::RuntimeWarning`, assert `tmp_path` usage, and add `ruff --select PD,NPY,PERF`. This moves discovery below the expensive reviewer layer.
6. **Test the notes.md mechanism directly.** Run one local model with carry-forward notes disabled. §B.7 suggests it is worth 2–3 attempts per run; that is worth confirming, because it is free.

---

## Risks & Unknowns

- **`claude-sonnet-1` is not a like-for-like comparison.** It branched from `6d1ecad`, before main's lint-cleanup, so it never faced the same starting tree. Treat its #1 rank as provisional.
- **`claude-sonnet-2`'s Slice 3 was never accepted.** Its branch head contains Slice 3 code that failed PM acceptance. I scored the code as written; a strict "delivered work only" accounting would drop it to 2 of 3 slices.
- **n=2 per model arm.** Every per-model mean here has a confidence interval wider than most of the gaps between models.
- **Reviewer effort was not varied.** All eight runs used `codex gpt-5.6-sol` at low/medium. Reviewer contribution cannot be separated from reviewer *choice*.
- **Not checked:** runtime behaviour on real SAGE catalogs (none available); memory profiles; whether any branch's numerics degrade at realistic catalog sizes beyond the 2 M-galaxy micro-benchmark.
- **Severity labels are mine.** The plan does not define P1/P2/P3; I applied the usual convention (P1 = wrong result or contract breach on a frozen boundary, P2 = defined-behaviour breach, P3 = out-of-domain or stylistic).
