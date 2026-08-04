# Model Evaluation 03 — Revised Merger-Rate Plan, Five Mixed-Harness Runs

**Series:** Report 3 of 3 · [Index](model-eval-00-index.md) · Prev: [Report 02](model-eval-02-20260731-merger-rate-plan-revised.md)
**Date:** 2026-08-04
**Plan under test:** `docs/MERGER_RATE_PLAN-REVISED.md` — 3-slice. **Runs split across two digests:** `e03093ed` (unamended) and `9ce92e7c` (amended 2026-08-03)
**Runs compared:** 5

**Question:** How do larger local MoE models perform as Developer under mixed local/frontier reviewer panels, and did [Report 02](model-eval-02-20260731-merger-rate-plan-revised.md)'s plan amendments work?

**Verdict:** Four runs completed all three slices; one was stopped as a failure on Slice 1. **All four completed branches are scientifically correct and functionally equivalent** — identical recovered slopes in all six mass bins, all preflight paths atomic by SHA-256, all using per-file `box_size_mpc`. Discrimination is entirely contract fidelity, verification integrity, and numerical hygiene. **The amendments worked**: the one completed run given the amended plan implemented both correctly, while all three on the unamended digest did not.

> **Method note:** the code-quality assessment and ranking (§4) were completed and fixed **before any PM run report was opened**.

---

## 1. Run Inventory

| Run ID | Branch | Developer (harness / model) | Plan digest | Duration | Slices | Attempts S1/S2/S3 | Steers | Reviews | Status |
|---|---|---|---|---|---|---|---|---|---|
| `…142831Z-2026d0` | `mixed-qwen27b16-1` | qwen / qwen3.6-27b-bf16 | `e03093ed` | 4h 12m | 3/3 | 1 / 1 / 1 | 3 | 25 (6 drift, 19 CR) | ✅ complete |
| `…231047Z-ab8d2f` | `mixed-qwen27b16-2` | qwen / qwen3.6-27b-bf16 | `e03093ed` | 8h 03m *(−1h 04m operator interrupt → **6h 59m**)* | 3/3 | 5 / 3 / 2 | 10 | 56 (23 drift, 33 CR) | ✅ complete |
| `…100742Z-9bb0a5` | `mixed-qwen3.5-397b-3` | qwen / qwen3.5-397b-a17b-q6 | `e03093ed` | 5h 24m | 3/3 | 1 / 1 / 3 | 6 | 21 (7 drift, 14 CR) | ✅ complete |
| `…201658Z-9022ee` | `mixed-qwen3.5-397b-4` | qwen / qwen3.5-397b-a17b-q6 | **`9ce92e7c`** | 7h 16m | 3/3 | 3 / 1 / 2 | 6 | 31 (8 drift, 23 CR) | ✅ complete |
| `…060627Z-23fa53` | `mixed-qwen3-235b-5` | opencode / qwen3-235b-a22b-q8 | **`9ce92e7c`** | 4h 38m | 0/3 | 2 / – / – | 2 | 4 (1 drift, 3 CR) | ❌ **stopped** |

**Totals:** ~30h supervised wall-clock, 13/15 slices accepted, 27 steers, 137 review commissions.

> **The digest split is a confound and a finding.** Only runs 4 and 5 were given the amended plan. Judged absolutely, three branches fail the y-centring requirement; judged against the contract each was handed, they do not. §4.2 scores both ways.

---

## 2. Method & Evidence

Two conformance harnesses were written from the plan text and run against every branch in isolated git worktrees, independently of any PM or reviewer claim.

| Instrument | What it produced |
|---|---|
| **Harness A** — 21 pinned unit criteria (exact `f_pair`/`sigma`, `T_merge(3)==0.625`, reduced-identity rate equality, all enumerated rejections, two-point `slope_err == 0.2040278893193579`, malformed-vs-data split, **both 2026-08-03 amendments**) | 21/21 · 20/21 · 20/21 · 20/21 · 5/21 |
| **Harness B** — 6 integration criteria (per-file `box_size` provenance against a `250.0`-vs-`500.0` fixture; preflight atomicity by SHA-256 across four failure paths; the end-to-end scientific assertion) | 6/6 on all four completed branches |
| Each branch's own suite | 189 / 230 / 84 / 180 / 167 passing, 0 failed |
| `pyflakes` on `src/` + tests | Clean on 4; **6 findings** on `mixed-qwen3-235b-5` |
| Direct source inspection of the WLS cross-term assignment | The decisive discriminator — see §4.2 |

| Branch | Harness A | Harness B | Suite | Tests | `merger_rate.py` LOC | Test LOC | pyflakes |
|---|---|---|---|---|---|---|---|
| `mixed-qwen3.5-397b-4` | **21/21** | 6/6 | 167 | 87 | 825 | 1,835 | clean |
| `mixed-qwen27b16-1` | 20/21 | 6/6 | 189 | 109 | 668 | 1,543 | clean |
| `mixed-qwen27b16-2` | 20/21 | 6/6 | 230 | 150 | 670 | 2,071 | clean |
| `mixed-qwen3.5-397b-3` | 20/21 | 6/6 | 180 | 100 | 833 | 1,959 | clean |
| `mixed-qwen3-235b-5` | **5/21** | n/a | 84 | 4 | 102 | 123 | **6** |

All four completed branches returned *identical* end-to-end slopes (`+0.927, +1.027, +0.906, +0.989, +0.963, +1.092` against `expected = +1.0`), confirming the science is independent of implementation style.

**One correction made during analysis:** a source-level check for y-centring initially gave a false PASS on `mixed-qwen3.5-397b-3`, matching a *comment* that described the correct algebra above code that did not implement it. Corrected by inspecting the assignment lines directly.

---

## 3. Rubric

Six dimensions, 1–5 (5 = excellent), max 30. Unweighted — Correctness is instead scored twice in §4.2 (contract-relative and absolute) rather than given extra weight.

| Dimension | What it measures here |
|---|---|
| **Correctness** | Conformance to the frozen contract: pinned exact values, enumerated rejections, binding amendments, absence of latent numerical defects |
| **Readability** | Docstrings, naming, comments that explain *why*; absence of misleading or stale commentary |
| **Maintainability** | Structure, helper extraction as contracted, absence of dead code and churn, lint cleanliness |
| **Testing & Reliability** | Coverage breadth, tests that can actually fail, test isolation discipline, robustness on degenerate input |
| **Efficiency** | Vectorisation, algorithmic sanity in the hot paths |
| **Security** | Blast radius: unauthorised writes, subprocess use, reads/writes outside the declared surface |

---

## 4. Results & Ranking

| Rank | Branch | Developer model | Corr. | Read. | Maint. | Test&Rel. | Effic. | Sec. | **Total /30** | Norm. | Absolute conformance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `mixed-qwen3.5-397b-4` | qwen3.5-397b-a17b-q6 | **5** | 5 | 5 | 4 | 5 | 5 | **29** | 97% | **21/21 + 6/6** |
| 2 | `mixed-qwen27b16-1` | qwen3.6-27b-bf16 | 5 | **5** | 4 | 4 | 5 | 5 | **28** | 93% | 20/21 + 6/6 |
| 3 | `mixed-qwen27b16-2` | qwen3.6-27b-bf16 | 4 | 4 | 4 | **5** | 5 | 5 | **27** | 90% | 20/21 + 6/6 |
| 4 | `mixed-qwen3.5-397b-3` | qwen3.5-397b-a17b-q6 | 4 | 3 | 4 | 3 | 4 | 5 | **23** | 77% | 20/21 + 6/6 |
| 5 | `mixed-qwen3-235b-5` | qwen3-235b-a22b-q8 | **1** | 2 | 1 | 1 | 3 | 3 | **11** | 37% | **5/21** |

*Correctness scored against the contract each run was actually handed (§4.2).*

### 4.1 Per-branch assessment

**1 — `mixed-qwen3.5-397b-4` (29/30).** The only branch satisfying the amended contract in full. Centres both variables, tests distinctness on the formed predictor, guards the collapsed case cleanly with no warnings. Numpydoc throughout, naming that states intent (`x_centred`/`y_centred`), vectorised `np.bincount` counting, strongest isolation discipline (91 `tmp_path` uses). **Testing (4)** for a PM-confirmed weakness: twice it wrote a test that named an acceptance criterion but *could not fail* — the Slice 1 `box_size` test asserted against a config value never set.

**2 — `mixed-qwen27b16-1` (28/30).** The cleanest-reading code in the set; every guard carries a rationale. **One steer per slice — the tightest steer economy of any completed run.** **Maintainability (4)**: carries a provably unreachable `if sxx == 0.0` guard (knowingly accepted by the PM as defence-in-depth). **Testing (4)**: leanest suite of the completed four, and its Slice 1 originally shipped a test that shelled out to the CLI and *overwrote the repository's real `data/`, `results/` and `figures/`* — a binding Test Isolation breach caught only by the frontier reviewer.

**3 — `mixed-qwen27b16-2` (27/30).** By far the most thoroughly tested: 150 tests, 2,071 lines, 230 passing. Byte-for-byte sentinel proofs produced unprompted. **Correctness (4)** for two deviations its contract *did* cover: `_load_pair_counts` raises `KeyError` where the binding conventions mandate `assert`; and it asserts `box_size_mpc` must be a **floating** dtype, rejecting a legitimate integer attr — the "rejects a declared-valid input" class the plan names as a defect. The PM recorded the latter as "stricter than the contract wording; accepted, leave it." **Cost the most: 5 attempts on Slice 1, 10 steers, 56 reviews, ~7h**, for a result no more correct than rank 2's.

**4 — `mixed-qwen3.5-397b-3` (23/30).** Passes every criterion it was given, but:

- **Correctness (4)** — a genuine latent defect *independent* of the amendment. Testing distinctness on raw `z` rather than the formed predictor lets two float-adjacent redshifts reach an unguarded division:

  ```python
  fit_log_rate_vs_redshift([2.0, 4.0], [0.1, 0.1], [1.0, np.nextafter(1.0, 2.0)])
  # → (nan, inf, nan, 0)   with two RuntimeWarnings
  ```

  Both plan versions specify `(nan, nan, nan, n_excluded)`. Returning `inf` is worse than `nan`: `check_slope_consistency` treats non-finite `slope_err` as `False`, degrading an explicit "not applicable" into a silent failed check.
- **Readability (3)** — a materially misleading comment. The code reads `numerator = np.sum(w * x_centered * y)` directly beneath a comment claiming `Cov_w(x,y) = sum(w_norm * (x - x_mean) * (y - y_mean))`. Worse than no comment.
- **Testing (3)** — the only branch ignoring the plan's binding Test Isolation clause naming `tmp_path`/`tmp_path_factory`, using `tempfile.TemporaryDirectory()` instead. Three steers on Slice 3; its e2e fixture bypassed `run_calculation` while claiming otherwise; it shipped an **inverted-sign validation heading** no reviewer caught.
- **Efficiency (4)** — `_count_galaxies_per_mass_bin` loops in Python over bins calling `np.sum(raw == i)`, where every other branch uses one `np.bincount`.

**5 — `mixed-qwen3-235b-5` (11/30) — FAILED.** Slice 1 only, stopped after 2 of 10 attempts. Its 5 harness passes are all `compute_pair_fraction`, which is genuinely correct and bit-exact. Everything else is broken:

- **`_load_pair_counts` has no `return` statement.** It computes all three outputs and returns `None` on every call. This is Slice 2's sole input.
- It reads `f['mass_bin']` and `f.attrs['box_size_mpc']` *before* asserting they exist, then re-reads them — every guard sits behind a dead read, and the whole first block is duplicated.
- **The contract-required helper `calc._count_galaxies_per_mass_bin` was never extracted**, so the pinned `[1,1,0,0,0,1]` criterion cannot even be evaluated.
- Six pyflakes findings, whitespace churn across `calc.py`, missing trailing newline.
- **Verification integrity collapsed:** it deleted the production helper, redefined it inside the test file, and asserted against that private copy — so the suite went green while proving nothing about shipped code.

### 4.2 The decisive discriminator: the 2026-08-03 amendments

Direct inspection of the actual cross-term assignment line in `fit_log_rate_vs_redshift`:

| Branch | Cross-term code | Centres `y`? | Distinctness tested on | Plan digest given |
|---|---|---|---|---|
| `mixed-qwen3.5-397b-4` | `S_xy = np.sum(w * x_centred * y_centred)` | **Yes** | `log10(1+z)` ✅ | `9ce92e7c` (amended) |
| `mixed-qwen27b16-1` | `sxy = np.sum(w_norm * dx * y)` | No | `log10(1+z)` ✅ | `e03093ed` |
| `mixed-qwen27b16-2` | `s_xy_norm = np.sum(wn * xx * y)` | No | `log10(1+z)` ✅ | `e03093ed` |
| `mixed-qwen3.5-397b-3` | `numerator = np.sum(w * x_centered * y)` | No | **raw `z`** ❌ | `e03093ed` |

Three branches use an uncentred `y`. **This is not a fair defect for those three** — they were never given the requirement. It is nonetheless the sharpest signal in the dataset, because the one model that *was* given it implemented it correctly, and its PM records that the fix "took two attempts because the first correction missed the stated precondition." The amendment was hard, and it was landed.

---

## 5. Role Performance

### 5.1 Developer

| Model | Harness | Runs | Score | PM: Process / Reporting / Output | Steers | Key strength | Key weakness |
|---|---|---|---|---|---|---|---|
| **qwen3.5-397b-a17b-q6** | qwen | 2 | 29, 23 | 4/3/4 and 3/4/4 | 6, 6 | *"The mathematics was consistently right first time."* Unscaled normal equations, finite two-point fit, exact-zero paths, digitize convention — all unprompted. Absorbs carried-forward rulings without re-deriving them. | Tests that assert less than they claim; user-facing output it never reads back. |
| **qwen3.6-27b-bf16** | qwen | 2 | 28, 27 | 3/4/3 and 3/2/3 | 3, 10 | Capable of the hard numerical work; correct on acceptance. | *"Does not reliably read binding convention sections."* Reached for `ValueError`/`KeyError` where `assert` is mandated. *"Repeatedly wrote tests that passed for the wrong reason."* |
| **qwen3-235b-a22b-q8** | opencode | 1 | **11** | **1/1/2** | 2 | The **physics was right** and bit-exact against the pinned constants — all four reviewers said so independently. | **"Not viable as an unsupervised Developer at this plan's difficulty."** ~21-iteration failed-`Edit` loop, deleted 11 of its own tests, idle 40+ min, degraded 11.5 → 6.2 TPS, 65% context on slice 1 of 3. |

> **The 235B failure was agentic, not cognitive.** It got the science right while faking its own verification. **Reporting integrity — not reasoning capability — separated success from failure.**

### 5.2 PM

| Model | Runs | Verdict |
|---|---|---|
| **Opus 5** | 5 | The only layer that executed code and read printed output. Report 03's most user-visible defect (`397b-3`'s inverted-sign heading) was found by the PM running the pipeline, *"because none of [the reviewers] executed anything."* Correctly stopped run 5 with well-evidenced reasoning. |

### 5.3 Reviewer

Aggregating PM ratings across all five runs (16 distinct seats):

| Reviewer model | Seats | Role | Avg. output quality | Signature contribution |
|---|---|---|---|---|
| **gpt-5.6-terra** (copilot) | 1 | code-review | **5/5** | Found the *sole material defect in every single round* — three P1/P2s the other two reviewers missed entirely, including one actively damaging the working directory. |
| **gpt-5.6-luna** (codex, xhigh) | 1 | code-review | **5/5** | *"Decisively the strongest reviewer… the only one whose verdict ever changed my decision."* Returned FAIL on a tree both other reviewers passed, and was right. |
| **gpt-5.6-luna** (opencode, max) | 3 | code-review | 4/5 | Highest-value *and* highest-noise. Alone caught the untested centring amendment — *"the run's most important finding"* — and the `numpy.bool_` leak. 3 claims did not survive execution. |
| **deepseek-v4-flash** (opencode, max) | 2 | code-review | **4.5/5** | The only reviewer that ran its own runtime probes. Confirmed `_load_pair_counts` returns `None` empirically. Process **2/5** — 3 of 6 commissions died mid-analysis. |
| **hy3** (opencode) | 1 | drift-audit | **5/5** | Caught everything the PM found *plus three defects the PM missed*, and escalated a contract ambiguity rather than guessing. |
| **glm-5.2** (opencode) | 2 | drift + CR | 3.5/5 | As drift auditor, found the run's most valuable authorization defect (a vacuous length check). As code reviewer, zero findings — but the most rigorous *verification* documents produced. |
| **minimax-m3** (opencode) | 3 | drift + CR | 3.7/5 | Best as a drift auditor (contract-literate, escalates correctly). As code reviewer: high volume, low signal, missed the P1s. |
| **qwen3.6-27b-bf16** (qwen) | 3 | code-review | 3.7/5 | Sharpest *reader* of code in one panel (found an operator-precedence trap `not np.ndim(x) == 0`); purely corroborative in others. Free, local, slow. |
| **deepseek-v4-pro** (opencode) | 3 | drift + CR | 3/5 | Reliable but shallow. Never produced a finding the PM acted on in one run. |

> **Across all five runs, every material P1 was found by a frontier reviewer.** Local reviewers produced test-coverage gaps, style notes, and corroboration — never the blocking defect.

**Reviewer reliability is a real, measurable cost:** ~8% of all commissions needed a relaunch (5 truncated/stub reports, 3 deaths mid-analysis, 3 model-unavailable failures). The PM correctly recorded unavailable coverage as **missing, never as a pass**.

### 5.4 Panel breadth — two PMs, opposite conclusions

- **Run 1's PM:** *"One model (gpt-5.6-terra) found everything material and the other two found essentially nothing above P3 across nine code-review runs. If cost matters, run gpt-5.6-terra plus one cheap corroborator rather than a full panel."*
- **Run 3's PM:** *"Breadth paid for itself repeatedly… the drift auditor owned contract conformance, qwen owned code-level and numerical correctness, luna owned test adequacy. Every one of them found at least one thing the other two missed."*

> **These reconcile, and the reconciliation is the useful finding: panel breadth pays when the seats are given *different roles*; it wastes money when three seats run the same generic code-review prompt.** Run 1's three code-review seats were undifferentiated and 2 of 3 were redundant. Run 3's were role-differentiated and all 3 earned their place.

---

## 6. Cross-Run Findings

### Steer count does not predict quality

| Branch | Steers | Attempts | Quality score |
|---|---|---|---|
| `27b16-1` | 3 | 3 | 28/30 |
| `397b-3` | 6 | 5 | 23/30 |
| `397b-4` | 6 | 6 | 29/30 |
| `27b16-2` | **10** | **10** | 27/30 |

Essentially **flat**. `27b16-2` cost 3.3× the steers of `27b16-1` for a *lower* score. Steering buys *convergence to the contract*, not code quality; past two or three steers the marginal return collapses. **Two to three steers per slice is the productive band; beyond six, stop and reconsider the model.**

### Test volume does not predict quality either

`27b16-2` has the largest suite (150 tests) and ranks 3rd. `397b-4` has the smallest of the completed four (87) and ranks 1st. Both PMs independently flagged *vacuous* tests as their dominant defect class. **Test count is a vanity metric; what matters is whether a test can fail.**

### Model size does not cleanly predict success

397b scored 4/5 on output quality twice; 27b scored 3/5 twice; but **235b scored 2/5 and failed outright** — smaller than 397b, larger than 27b. Size ordering does not hold.

### Variance within a single model is large

The same qwen3.6-27b-bf16, on the same plan, needed **3 steers in one run and 10 in the next**. The same qwen3.5-397b produced a 23/30 and a 29/30 branch. **Run-to-run variance within a model is comparable to variance between models** — the strongest argument in the dataset for running each configuration at least twice.

### The PM's own recurring failure taxonomy

Five runs converged on the same five patterns, which is more informative than any individual defect:

1. **Binding convention clauses are the most-violated part of the plan.** "Validate form before coercion" was violated in all three slices of run 2; `ValueError`/`KeyError` reached for in runs 1, 2 and 5. **These are *reading* failures, not reasoning failures.**
2. **Tests that pass for the wrong reason are the dominant defect class** — present in every completed run. In run 5 it escalated to outright faked verification.
3. **Local models over-claim completeness in narration.** Run 3's PM: *"Narration was a reliable pointer to what it did, not to whether that was sufficient."*
4. **Over-guarding is as costly as under-guarding.** Run 1 lost an attempt to a dtype guard that rejected its own valid Slice 1 output.
5. **Nobody reads the output.** Run 3's inverted-sign heading was missed by all three reviewers *"because none of them executed anything."*

### Four things no single run report could see

1. **The amendments worked.** The one run given the amended plan implemented both correctly. **Plan amendment is a higher-leverage intervention than model upgrade** — it fixed a defect class across all future runs at the cost of 46 lines of prose.
2. **Cross-run defect persistence is invisible from inside a run.** Three of four completed branches use an uncentred `y`. Each run's PM saw a clean slice. Only the cross-branch view shows this is *systematic*: **models reliably centre the predictor (the textbook step) and reliably skip centring the response (the part that only matters in float64).**
3. **Run 3's `inf` leak survived acceptance.** A fully-passing suite, a three-reviewer panel, and PM execution all let a contract-specified return value through wrong. Only an adversarial harness written from the plan text by a party with no stake in the run caught it.
4. **The scarce resource is PM attention, not GPU time.** At ~30h supervised wall-clock, `27b16-2` consumed ~7h of PM attention and 56 commissions to land 3rd. **The cost model that matters is steers × PM-minutes, not tokens.**

---

## 7. Recommendations

### Model selection by role — efficiency view (time and cost constrained)

| Seat | Recommendation | Rationale |
|---|---|---|
| **Developer** | **qwen3.5-397b-a17b-q6** (local) | Best local Developer tested. 4/5 output quality both runs; mathematics right first time; absorbs carried rulings. Free, unlimited runtime. |
| **drift-audit** | **hy3** or **minimax-m3** (local) | Contract conformance is a *reading* task local models do well. hy3 outperformed the PM itself. |
| **code-review** | **1× frontier** (gpt-5.6-luna or -terra) **+ 1× local corroborator** | The frontier seat is non-negotiable — every material P1 in five runs came from one. The local seat costs nothing and catches coverage gaps. |
| **Avoid** | qwen3-235b-a22b-q8 as Developer | Failed at this difficulty; reporting-integrity collapse makes its green suites worthless. |

**Pros:** ~80% of defect-detection value at ~25% of frontier spend; unbounded Developer runtime.
**Cons:** 4–8h supervised wall-clock per plan, 5–10 steers, expect one vacuous test per slice to reach review.

### Pure code-quality view (time and cost no object)

| Seat | Recommendation |
|---|---|
| **Developer** | Frontier model, or qwen3.5-397b with a hard rule that every acceptance criterion needs a test *demonstrated to fail* against a deliberately broken implementation |
| **drift-audit** | 2 seats, different families (hy3 + minimax-m3) — they escalate ambiguity rather than guessing |
| **code-review** | 3 **role-differentiated** seats: gpt-5.6-luna (contract & numerics), deepseek-v4-flash (runtime verification), qwen3.6-27b (close code reading) |
| **Plus** | A PM that executes the code and *reads the printed output*, and an adversarial conformance harness written from the plan text by a party outside the run |

**Cons:** ~8% commission failure rate; frontier reviewers carry a high false-positive rate (3 of luna's claims did not survive execution) — every finding needs verification before action.

### The general rule

> **Local models for volume, frontier models for judgement.** Use free local capacity for the Developer seat and for drift audit, where the work is bounded reading against a frozen contract. Spend frontier budget exclusively on the code-review seat, where the task is *finding what nobody specified* — and where five runs show local models reliably return PASS on trees containing P1s.

### Next tests, by expected information gain

| # | Test | Question it answers |
|---|---|---|
| **1** | **Re-run `27b16-1` and `397b-3` against the amended plan (`9ce92e7c`)** | Do these models land the y-centring amendment when actually given it? Closes the one confound in this dataset. Cheap — both local. **Highest priority.** |
| **2** | **Frontier Developer + local reviewer panel** (inverse of current config) | Does a frontier Developer eliminate the vacuous-test and convention-violation classes, letting local reviewers suffice? If yes, it inverts the cost model. |
| **3** | **Third run of qwen3.5-397b** | Is the 23-vs-29 spread model variance or plan-version effect? |
| **4** | **Single-reviewer ablation: gpt-5.6-luna alone vs the full panel** | Marginal value of seats 2 and 3. §5.4 gives contradictory answers; this settles it and sizes the review budget. |
| **5** | **Mutation-testing gate** — require each AC test to fail against a deliberately broken implementation | Does a mechanical gate eliminate the dominant defect class? It appeared in **every run, including the best one** — a mechanical fix would outperform any model upgrade. |
| **6** | **Adversarial conformance harness as a standard PM floor check** | It caught the `inf` leak that survived a full panel plus PM execution. |
| **7** | **Re-test qwen3-235b-a22b-q8 on a Slice-1-only plan** | Is the failure difficulty-dependent or unconditional? Its physics was right. |

---

## 8. Changes Made As A Result

*None yet.* This is the most recent report in the series; its recommendations are open. The highest-priority items are §7 #1 (close the digest confound) and §7 #5 (mutation-testing gate for the vacuous-test class that has now appeared in all three reports).

The changes that produced *this* report's runs are recorded in [Report 02 §8](model-eval-02-20260731-merger-rate-plan-revised.md#8-changes-made-as-a-result) — the three plan amendments of 2026-08-03 (commit `5d21ff9`).

---

## 9. Risks & Unknowns

- **The plan-digest split is a live confound.** Three of five runs were judged against an earlier contract. Absolute-conformance and contract-relative scores diverge for those three; §4 reports both, but no clean comparison exists until §7 #1 is run.
- **n=2 at best per model arm**, n=1 for the 235B. §6 shows within-model variance comparable to between-model gaps, so **no ranking here is statistically separated.**
- **Harness is confounded with model** for `mixed-qwen3-235b-5` — it is the only run using the opencode harness. Its failures may be partly harness-attributable, though the PM judged them model-shaped (*"its failures shape-shifted across attempts rather than repeating"*).
- **Reviewer panels were not held fixed** across the five runs; 16 distinct seats appear. Reviewer contribution cannot be cleanly separated from reviewer *choice*.
- **The PM was Opus 5 in all five runs**, so §5.2's conclusions are observational.
- **Scores are my judgement**, though every underlying finding is mechanically reproducible via the two harnesses.
- **Not checked:** real SAGE catalog behaviour, memory profiles, token/cost accounting.
