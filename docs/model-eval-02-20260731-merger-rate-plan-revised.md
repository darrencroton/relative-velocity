# Model Evaluation 02 — Revised Merger-Rate Plan, Eight Runs

**Series:** Report 2 of 3 · [Index](model-eval-00-index.md) · Prev: [Report 01](model-eval-01-20260729-merger-rate-plan.md) · Next: [Report 03](model-eval-03-20260804-merger-rate-plan-revised.md)
**Date:** 2026-07-31
**Plan under test:** `docs/MERGER_RATE_PLAN-REVISED.md` — 3-slice, unamended
**Runs compared:** 8

**Question:** Which Developer model/harness combinations produced the best code, and how should frontier (paid, time-limited) and local (free, unlimited) models be combined for moderate-to-complex coding tasks?

**Verdict:** All eight branches implement Slices 1–3 and pass their own suites (191–246 tests, 0 failures). Against an independent 100-test conformance harness written from the plan's acceptance criteria alone, **six branches scored 100/100 and two scored 99/100**. The plan format worked: every branch covered every named exact-value criterion. Differentiation is therefore almost entirely code quality, not feature completeness. **Top of the ranking is genuinely close; the bottom is not.**

> **Method note:** the code-quality analysis was completed in full — including all harness runs, cross-testing and probes — *before* any `.pm/runs` report was opened.

---

## 1. Run Inventory

| Branch | Developer model | Reviewer | Wall clock | Attempts (S1/S2/S3) | Events | Status |
|---|---|---|---|---|---|---|
| `claude-sonnet-1` | claude-sonnet-5 @ high | codex gpt-5.6-sol | **53m 53s** | 1 / 0 / 1 = **2** | 33 | ✅ complete |
| `claude-sonnet-2` | claude-sonnet-5 @ high | codex gpt-5.6-sol | 1h 46m (~35m reviewer timeout) | 1 / 1 / 1 = **3** | 39 | ⚠️ **STOPPED — human decision** |
| `opencode-qwen27b16-1` | qwen3.6-27b-bf16 | codex gpt-5.6-sol | **6h 08m** | 4 / 3 / 1 = **8** | 86 | ✅ complete |
| `opencode-qwen27b16-2` | qwen3.6-27b-bf16 | codex gpt-5.6-sol | 3h 49m | 4 / 2 / 2 = **8** | 65 | ✅ complete |
| `opencode-qwen27b8-1` | qwen3.6-27b-q8 (~14–15 TPS) | codex gpt-5.6-sol | 3h 45m | 1 / 2 / 2 = **5** | 62 | ✅ complete |
| `opencode-qwen27b8-2` | qwen3.6-27b-q8 | codex gpt-5.6-sol | 3h 59m | 3 / 2 / 1 = **6** | 61 | ✅ complete |
| `opencode-qwen35b16-1` | qwen3.6-35b-a3b-bf16 | codex gpt-5.6-sol | 1h 46m | 2 / 2 / 2 = **6** | 89 | ✅ complete |
| `opencode-qwen35b16-2` | qwen3.6-35b-a3b-bf16 | codex gpt-5.6-sol | 2h 47m | 2 / 2 / 3 = **7** | 62 | ✅ complete |

Reviewers were always `codex gpt-5.6-sol`, at **low effort for drift-audit** and **medium for code-review**. No run exceeded 8/10 attempts on any slice; the mechanical floor passed 8/8 at every finalize in every run.

---

## 2. Method & Evidence

| Instrument | What it produced |
|---|---|
| Each branch's own suite | 191–246 tests, **0 failures on all 8** |
| **Conformance harness A** (100 tests, from the plan's ACs only) | 6 branches 100/100; `claude-sonnet-2` and `qwen35b16-1` 99/100 |
| **Conformance harness B** (19 valid edge tests) | 2 branches 19/19; 6 branches 18/19 |
| **Cross-test matrix** (56 cells: every suite × every other implementation) | Isolated 2 low-coupling "referee" suites |
| **Direct behavioural probes** | Confirmed/refuted 4 candidate defects per branch |
| `ruff` (E,F,W,B,C90,SIM,PIE,RET,ARG,PL) | src near-identical (7–9); tests 16–65 |
| Micro-benchmarks | Settled that alleged efficiency defects are not measurable at real scale |

One harness test (`D11`, weight-normalisation overflow) was **discarded as invalid** — the plan explicitly declares extreme-magnitude overflow unspecified, so failing it is not a defect. Reporting it would have been unfair to all eight.

---

## 3. Rubric

Six dimensions, 1–5 (5 = excellent). **`Correctness` is double-weighted**, because for scientific code a wrong number is categorically worse than an ugly one. Max 35.

| Dimension | What it scores |
|---|---|
| **Correctness** | Satisfies the frozen contract, including binding validation conventions and edge behaviour |
| **Readability** | Naming, structure, comment quality, house-style consistency, accurate documentation |
| **Maintainability** | DRY, dead code, scope discipline, cost of the next change |
| **Testing & Reliability** | Coverage of the acceptance criteria, test rigour, isolation, brittleness |
| **Efficiency** | Idiom, wasted work, latent performance traps |
| **Security** | Input handling, failure modes, no unsafe coercion or silent acceptance |

---

## 4. Results & Ranking

| Rank | Branch | Developer model | Corr. | Read. | Maint. | Test | Effic. | Sec. | Raw /30 | **Weighted /35** | Norm. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `claude-sonnet-1` | claude-sonnet-5 @ high | 5 | 5 | 4 | 4 | 5 | 4 | 27 | **32** | 91% |
| 2 | `claude-sonnet-2` | claude-sonnet-5 @ high | 4 | 5 | 5 | 4 | 5 | 4 | 27 | **31** | 89% |
| 3 | `opencode-qwen27b16-1` | qwen3.6-27b-bf16 | 5 | 3 | 3 | 5 | 4 | 5 | 25 | **30** | 86% |
| 4 | `opencode-qwen27b16-2` | qwen3.6-27b-bf16 | 4 | 3 | 3 | 5 | 4 | 4 | 23 | **27** | 77% |
| 5 | `opencode-qwen27b8-1` | qwen3.6-27b-q8 | 4 | 3 | 2 | 5 | 4 | 4 | 22 | **26** | 74% |
| 6 | `opencode-qwen27b8-2` | qwen3.6-27b-q8 | 4 | 3 | 2 | 4 | 4 | 4 | 21 | **25** | 71% |
| 7 | `opencode-qwen35b16-2` | qwen3.6-35b-a3b-bf16 | 4 | 3 | 2 | 4 | 3 | 4 | 20 | **24** | 69% |
| 8 | `opencode-qwen35b16-1` | qwen3.6-35b-a3b-bf16 | 2 | 2 | 2 | 2 | 2 | 3 | 13 | **15** | 43% |

One-line verdicts: **1** cleanest validation, fewest confirmed defects, fastest run · **2** best-documented and most economical; one real coercion defect; Slice 3 never accepted · **3** only branch with zero confirmed correctness defects; ugly but right · **4** solid; scope creep and internal duplication · **5** correct science; heaviest house-style drift; live warning in own suite · **6** worst DRY violation; 100% message-pinned tests · **7** reintroduces the cancellation-prone form the plan warned against · **8** multiple confirmed defects, dead code, leaks 37 temp dirs per test run.

### 4.1 Confirmed defects

Each row was reproduced by direct execution, not inferred from reading.

| Defect | Sev. | Branches | Evidence |
|---|---|---|---|
| `_load_pair_counts` coerces `float(attr)` before form check → **TypeError**, violating both "validate form before coercion" and "no new TypeError" | **P2** | `claude-sonnet-2`, `qwen35b16-1` | Harness A `C1_16`; confirmed by *both* referee suites |
| `compute_merger_rate` **silently accepts the string `"500.0"`** as `box_size_mpc` and returns a plausible rate | **P2** | `claude-sonnet-1`, `qwen35b16-1` | Direct probe — the exact failure the plan names by example. Mitigating: strings are not in this function's *listed* rejection set |
| Degenerate predictor **raises AssertionError** where the plan says it must return `(nan, nan, nan, n_excluded)` | **P1** | `qwen35b16-1` | Direct probe; `assert s_xx_c > 0` on a frozen must-not-raise boundary |
| Degenerate predictor **fabricates a finite fit** (slope ≈ 3.5e16) | **P2** | `claude-sonnet-2`, `qwen27b16-2`, `qwen27b8-1` | Direct probe. Same class the `claude-sonnet-2` PM stopped the run over |
| `merger_timescale_gyr` leaks `OverflowError` instead of asserting its result is finite | **P3** | all except `qwen27b16-1`, `qwen27b8-1` | Harness B `D14`. Largely out-of-domain |
| **Test isolation violated**: 30× `mkdtemp()`, zero `tmp_path`, no cleanup → **37 leaked temp dirs per run** | **P2** | `qwen35b16-1` | Measured with a controlled `TMPDIR` |
| `fit_log_rate_vs_redshift` asserts `isinstance(rates, (list, tuple, ndarray))` — forbidden extra validation rejecting declared-valid array-likes | **P3** | `qwen27b8-2` | Code read |
| Live `RuntimeWarning: divide by zero` emitted by the branch's **own passing suite** | **P3** | `qwen27b8-1` | Captured at `merger_rate.py:250` |

**Only `qwen27b16-1` carries no confirmed defect** — and it is also the only branch returning the correct `nan` tuple on the degenerate-predictor case.

### 4.2 Honest negatives

- **`claude-sonnet-1` did not regress the `datetime.utcnow()` timestamp.** It branched from `6d1ecad`, *before* main's lint-cleanup `d567671`, and inherited the deprecated call. A **baseline confound** — it is the only branch not compared like-for-like.
- **The "Python loop instead of `np.bincount`" criticism does not survive measurement.** At 6 bins on 2M galaxies: `bincount` 4.0ms, `np.add.at` 3.3ms, Python loop 4.9ms — all dwarfed by `np.digitize` at ~31ms. An idiom point, not a performance defect.
- **`check_slope_consistency` returns a builtin `bool` in all eight branches.** Suspected `np.bool_` leakage in the 35B branches; the probe refuted it.

### 4.3 Quality signals not captured by tests

| Signal | Best | Worst |
|---|---|---|
| Module docstring accuracy | `claude-sonnet-2` — rewritten to describe all three slices | `qwen35b16-1`: *"Slice 1 only covers…"* — actively false |
| Dead code | `claude-sonnet-*`, `qwen27b16-2` — none | `qwen35b16-1`: `np.asarray(x, dtype=object)` ×3 immediately overwritten (17,000× slower conversion), no-op `both_zero`, unused `_cov_00`/`_cov_01`. `qwen35b16-2` names unused terms with leading underscores **to dodge the linter** |
| DRY | `claude-sonnet-2` — helper-factored | `qwen27b8-2`: three verbatim 12-line validation blocks (~60 lines). `qwen27b16-2` recomputes `n_mass_bins` inline 3× while its own `_mass_bin_edges` sits unused |
| House style (`assert`) | `claude-sonnet-*`, `qwen35b16-2` | `qwen27b8-1` — `merger_timescale_gyr` is entirely `if/raise AssertionError` |
| API sanity | index-valued `mass_bin` key | `qwen27b16-2`, `qwen27b8-2` return `mass_bin` as a **string label** (`"[8.0, 8.5)"`) |
| Diff hygiene | `claude-sonnet-*`, `qwen27b8-*` | `qwen35b16-1/-2`, `qwen27b16-2` realigned pre-existing `calc.py` lines the plan declared unchanged |
| Broken error message | — | `qwen35b16-1` emits literal `{n_expected - 1}` (missing `f` prefix) |

### 4.4 Test-suite character

Cross-test failures are dominated by error-message pinning, so the useful metric is **coupling**, not raw failures.

| Branch | Tests | Test LOC | `raises` with `match=` | Coupling | Failures induced elsewhere | Reading |
|---|---|---|---|---|---|---|
| `qwen27b8-1` | 166 | 2072 | 0 / 77 | **0%** | 9.9 | Most portable; referee |
| `qwen27b16-2` | 125 | 1495 | 1 / 66 | **2%** | 4.6 | Most portable; referee |
| `qwen27b16-1` | 156 | 1889 | 20 / 78 | 26% | 16.6 | Balanced |
| `qwen35b16-2` | 113 | 1611 | 29 / 54 | 54% | 21.7 | Moderate |
| `qwen35b16-1` | 126 | 1935 | 35 / 41 | 85% | 20.6 | Brittle **and** leaks temp dirs |
| `claude-sonnet-1` | 112 | 1224 | 66 / 73 | 90% | 48.0 | Strictest, most brittle |
| `claude-sonnet-2` | 74 | 1161 | 37 / 39 | 95% | 7.4 | Brittle but most economical |
| `qwen27b8-2` | 138 | 1733 | 73 / 73 | **100%** | 37.1 | Any reword breaks 73 tests |

Heavy `match=` use is a genuine trade: it satisfies the plan's "assertion naming the reason" criteria but welds the suite to exact strings. `claude-sonnet-2` gets the best of both — 74 tests covering the same ground as `qwen27b8-1`'s 166, via 12 `parametrize` blocks.

> **All eight branches covered every named exact-value criterion.** That uniformity across a 20× spread in model capability is the strongest evidence here that **frozen numeric criteria are what carried the weak models**.

---

## 5. Role Performance

### 5.1 Developer

| Model | Runs | Scores | Key strength | Key weakness |
|---|---|---|---|---|
| **claude-sonnet-5** @ high | 2 | 32, 31 | Best scores, fewest defects, fastest. **The only configuration that detected a defect in the plan itself.** | `claude-sonnet-2` carries the form-before-coercion defect and its Slice 3 was never accepted. |
| **qwen3.6-27b-bf16** | 2 | 30, 27 | `qwen27b16-1` has the **best confirmed-correctness record of all eight** — the only branch with zero confirmed defects. | 6h09m and 8 steers. Readability and naming, not wrong answers. |
| **qwen3.6-27b-q8** | 2 | 26, 25 | Faster and fewer attempts than bf16 at near-equal quality. | Heaviest house-style drift; `qwen27b8-2`'s 100%-pinned suite; a live `RuntimeWarning` in a green suite. |
| **qwen3.6-35b-a3b-bf16** | 2 | 24, **15** | Fastest local model. | **Ranked 7th and 8th.** Reporting reliability scored 2/5 twice. Dead code, leaked temp dirs, a P1 must-not-raise breach, a broken f-string. |

**The PM's own Developer scores**, from the four runs carrying `model-performance.md`:

| Developer | Process | Reporting | Output | PM's summary |
|---|---|---|---|---|
| `qwen27b8-1` | 3/5 | **5/5** | 3/5 | "science is right… but under-tests the specific thing an acceptance criterion is protecting" |
| `qwen27b8-2` | 4/5 | 4/5 | 3/5 | "repeatedly wrote tests that would pass under the very regression they were meant to catch" |
| `qwen35b16-1` | **2/5** | **2/5** | **2/5** | "narration was always plausible and twice materially wrong" |
| `qwen35b16-2` | 3/5 | **2/5** | 3/5 | "reports confidently on work it has not actually proven" |

The PM's ordering (`27b8-*` ≈ `35b16-2` > `35b16-1`) **matches mine**, arrived at independently.

### 5.2 PM

| Model | Effort | Runs | Verdict |
|---|---|---|---|
| **Opus 5** | — | 8 | **The load-bearing element of the whole system.** Every run's decisive numerical check was the PM's, because *neither Reviewer can execute pytest* (Reviewer Mode forbids state-changing commands). |

But see §6 — the PM seat is also where the **highest-leverage failure** in this dataset occurred.

### 5.3 Reviewer

| Seat | Model / effort | Findings contributed | Verdict |
|---|---|---|---|
| **drift-audit** | codex gpt-5.6-sol, **low** | High, cheap | **Best value-per-dollar in the system.** Caught the P1 intercept sign error, unauthorized components, three "could-not-fail" test classes. |
| **code-review** | codex gpt-5.6-sol, **medium** | High, with noise | Caught the `np.bytes_` preflight bypass that would have destroyed a prior results file, and the vacuous-pass hole in the headline scientific test. But **~⅓ of findings were rejectable on cited plan text**. |

### 5.4 What the assurance layers actually caught

| Layer | Findings | Verdict |
|---|---|---|
| Mechanical floor (8 facts) | **~0** — 8/8 at every finalize in all 8 runs, *including on a tree carrying a P1 sign error* | Necessary hygiene; **zero discovery value** |
| Differential lint (`ruff`) | **~0** — clean on every attempt in every run | The `qwen35b16-2` PM: *"zero of the eleven findings across this run were mechanical"* |
| drift-audit | High | Cheapest real signal |
| code-review | High, noisy | Real but needs adjudication |
| **PM's own verification** | **Highest** | The only layer that runs code |

> This partially **retires Report 01's headline recommendation.** The linter was worth building — it just does not find what actually escapes on this plan. Report 01's 11-of-14 figure was measured on a plan with looser numeric criteria; once the criteria were frozen, the escaping defects moved above the mechanical layer entirely.

---

## 6. Cross-Run Findings

### The single highest-leverage finding: a PM adjudication that flipped

The **same defect** received **opposite rulings in two runs under the same PM contract**:

- **`claude-sonnet-1`, Slice 1 steer:** *"`_load_pair_counts` must reject a non-scalar `box_size_mpc` with an assertion, not a TypeError… This violates two binding conventions."* → **Steered and fixed.**
- **`qwen35b16-1`, Slice 1 review round 2:** reviewer raised it; PM replied *"factually wrong on the pinned numpy 2.5.1 — I checked empirically that `float()` raises TypeError on every non-0-d array, so the value is rejected loudly."* → **Rejected.**

Both statements about `float()` are true. The disagreement is whether "raises TypeError" counts as rejection when the plan says *"Do not introduce TypeError / ValueError raises for new input validation."*

> **This is not a model-capability difference.** The defect shipped in `qwen35b16-1` and `claude-sonnet-2` purely because of which way the adjudication fell. **Supervision variance, not developer variance, decided the outcome.**

### Learning within a run is real and large

Steers fall monotonically across slices in most runs, driven by the PM's curated `notes.md` carry-forward:

| Run | S1 → S2 → S3 attempts |
|---|---|
| `qwen27b16-1` | 4 → 3 → 1 |
| `qwen27b8-2` | 3 → 2 → 1 |
| `qwen27b16-2` | 4 → 2 → 2 |
| `claude-sonnet-1` | 1 → 0 → 1 |

The `qwen27b8-2` PM states it directly: *"first-attempt quality improved sharply once the curated notes carried the conventions forward."* **A weak model with good carry-forward memory converges toward a strong model's first-attempt quality by slice 3.**

### Other trends

- **Quantisation cost nothing measurable.** q8 was *faster* than bf16 on the same 27B model (3h45m/3h59m vs 6h09m/3h49m), needed *fewer* attempts (5,6 vs 8,8), and scored within 4 points. **No evidence bf16 bought quality.**
- **Active-parameter count predicts quality better than total.** The 35B-a3b MoE (~3B active) is the fastest local model and the worst-scoring. **Total-parameter headline numbers are misleading for code work.**
- **Run-to-run variance within a model is substantial.** `qwen35b16-1` (15) vs `-2` (24) is a 9-point spread on identical model and plan — **larger than the gap between the best local model and the frontier model.** A single run is not a measurement.
- **Wall clock does not track quality**, and within the local models it is *negative*: the slowest run is the best local result; the fastest is the worst.

### Where I found things the PM did not

| Finding | Why the PM missed it |
|---|---|
| `qwen35b16-1` leaks 37 temp dirs per run | The PM saw `mkdtemp` and ruled it "a benign mechanism deviation", verifying only that the repo's dirs were untouched. It checked the clause's *intent* but not its *hygiene consequence*. |
| Stale/false `src` module docstrings in 5 branches | The PM noticed the *test* docstring being edited but never checked the *production* one |
| `np.asarray(..., dtype=object)` dead code ×3 | Passes `ruff` — the value *is* assigned and later reassigned |
| `mass_bin` returned as a string label in two branches | Key-name checks passed; nobody checked the value's type |
| A live `RuntimeWarning` in a green suite | Warnings are not failures; nothing in the floor or lint surfaces them |

---

## 7. Recommendations

### Statistics behind the recommendations

| Metric | claude-sonnet-5 (n=2) | qwen27b-bf16 (n=2) | qwen27b-q8 (n=2) | qwen35b-a3b (n=2) |
|---|---|---|---|---|
| Mean wall clock | **80.1 min** | 298.9 min | 232.6 min | 137.1 min |
| Mean attempts | **2.5** | 8.0 | 5.5 | 6.5 |
| Mean weighted score /35 | **31.5** | 28.5 | 25.5 | 19.5 |
| Mean `merger_rate.py` LOC | **568** | 627 | 583 | 779 |
| Mean test LOC | **1193** | 1692 | 1903 | 1773 |
| Mean ruff findings in tests | **23** | 45 | 61 | 44 |
| Slowdown vs frontier | 1.0× | 3.7× | 2.9× | 1.7× |

**Cost-of-supervision proxy** (attempts per accepted slice): claude-sonnet-5 **0.83**, qwen27b-q8 1.83, qwen35b-a3b 2.17, qwen27b-bf16 2.67.

**Verbosity penalty:** local models wrote 10–37% more production code and 42–60% more test code for identical functionality. `qwen35b16-1` wrote **45% more production code than `claude-sonnet-2` and scored half as well.**

### Model selection by role

| Seat | Recommendation | Rationale |
|---|---|---|
| **Developer** | Local `qwen3.6-27b` — **q8 for throughput, bf16 for quality**. Free, unlimited, and given frozen numeric criteria reaches within 2–5 rubric points of frontier. | 30/35 with zero confirmed defects. |
| **PM** | **Frontier, high effort. Do not economise.** | §6 shows an adjudication coin-flip decided whether a defect shipped; §5.4 shows the PM is the only layer that executes code. |
| **drift-audit** | codex at **low** effort | Best value-per-dollar in the system. |
| **code-review** | codex at **medium** effort | Accept that ~⅓ of findings will be rejectable on cited plan text. |
| **Never** | `qwen3.6-35b-a3b` for contract-bound work | Both runs ranked bottom-two; reporting reliability 2/5 twice. |

### By scenario

| Scenario | Recommendation |
|---|---|
| Interactive / on the clock | claude-sonnet-5 Developer, codex Reviewers. ~1h per three-slice plan |
| Overnight / unattended | **qwen3.6-27b-q8** Developer. 2.9× slower, 25/35, cheaper per attempt than bf16 |
| Maximum local quality, cost irrelevant | qwen3.6-27b-bf16. Accept 3.7× wall clock |
| **Hybrid (recommended default)** | **Local 27B Developer + frontier PM + codex Reviewers** |

### Next tests proposed

1. **Resolve the plan defect first** — `claude-sonnet-2`'s Slice 3 is unaccepted pending a human decision on the degenerate-predictor contract. 6 of 8 branches mishandle it. **Amend before running anything else.**
2. **Close the adjudication inconsistency** — add to the Validation Conventions: *"a TypeError leaking from an unguarded coercion is not a valid rejection, even though it fails loudly."*
3. **Cross-model PM test (highest value).** Run the same Developer twice with a frontier PM and twice with a local PM. Nothing in this dataset tests it — the PM was frontier in all eight.
4. **n=4 per model arm.** The 9-point within-model spread exceeds the between-model gap.
5. **Mechanical gate for what lint missed** — `pytest -W error::RuntimeWarning`, assert `tmp_path` usage, `ruff --select PD,NPY,PERF`.
6. **Test the `notes.md` mechanism directly** — run one local model with carry-forward disabled. Worth 2–3 attempts per run if confirmed, and free.

---

## 8. Changes Made As A Result

Both plan amendments below landed in `docs/MERGER_RATE_PLAN-REVISED.md` on **2026-08-03** (commit `5d21ff9`, "Amend the revised merger-rate plan to close two contract defects"), changing the plan digest from `e03093ed` to `9ce92e7c`. This is the causal link to [Report 03](model-eval-03-20260804-merger-rate-plan-revised.md), whose runs split across both digests.

| # | Change | Origin |
|---|---|---|
| **1** | **Collapsed-predictor rule.** "Distinct redshifts" now means distinct in the **formed predictor** `log10(1+z)`, tested *before any centring*, returning `(nan, nan, nan, n_excluded)` rather than raising or fabricating a slope. No minimum-separation threshold is specified, and none is to be invented. | §4.1 — 6 of 8 branches mishandled this; the `claude-sonnet-2` PM stopped its run over it |
| **2** | **Centre-both-variables rule.** The cross term must be accumulated as `sum(w · x_centred · (y − y_weighted_mean))`. Centring only `x` biases the slope by a term proportional to `y`'s offset. This is what "numerically stable form" means. | Same PM stop; the recommended amendment in §7 item 1 |
| **3** | **TypeError-leak clause.** A `TypeError` or `ValueError` *leaking* from an unguarded coercion is **not a valid rejection**, even though it fails loudly — where the value is one the Acceptance Criteria require the slice to reject. | §6 — the adjudication that flipped |

**Outcome, measured in Report 03:** only two of five subsequent runs were given the amended plan. Of those, **the one that completed implemented both numerical amendments correctly**, while all three runs on the unamended digest used an uncentred `y`. The amendments worked — and Report 03 concludes that **plan amendment is a higher-leverage intervention than model upgrade**, fixing a defect class across all future runs at the cost of 46 lines of prose.

---

## 9. Risks & Unknowns

- **`claude-sonnet-1` is not a like-for-like comparison.** It branched from `6d1ecad`, before main's lint-cleanup, so it never faced the same starting tree. **Treat its #1 rank as provisional.**
- **`claude-sonnet-2`'s Slice 3 was never accepted.** Its branch head contains Slice 3 code that failed PM acceptance. I scored the code as written; a strict "delivered work only" accounting drops it to 2 of 3 slices.
- **n=2 per model arm.** Every per-model mean has a confidence interval wider than most of the gaps between models.
- **Reviewer effort was not varied.** All eight used `codex gpt-5.6-sol` at low/medium. Reviewer contribution cannot be separated from reviewer *choice*.
- **The PM was frontier in all eight runs**, so §6's central finding about supervision variance is observational, not controlled.
- **Not checked:** runtime on real SAGE catalogs (none available); memory profiles; numerical degradation beyond the 2M-galaxy micro-benchmark.
- **Severity labels are mine.** The plan does not define P1/P2/P3; I applied the usual convention (P1 = wrong result or contract breach on a frozen boundary, P2 = defined-behaviour breach, P3 = out-of-domain or stylistic).
