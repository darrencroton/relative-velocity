# Merger-Rate Plan: Developer Harness/Model Evaluation Across Five `project-manager` Runs

**Date:** 2026-07-29
**Question:** How effectively does the `project-manager` skill execute `docs/MERGER_RATE_PLAN.md`, and how does developer harness/model choice change the outcome?
**Verdict:** All four completed runs produce **numerically identical science**. Ranking is decided entirely by contract discipline, code hygiene, and output usability — not by numerical correctness. **baseline-2 (96) > baseline-3 (92) > opencode-2 (88) > opencode-1 (76)**. Three of four clear the acceptable bar; **opencode-1 does not**.
**Two control violations invalidate part of the stated test design** — see §1.1. Runs 2–3 ran the developer at **high** effort, not medium; run 1's reviewer was **Copilot**, not Codex.
**Runtime is explicitly excluded from scoring** (§1.3). Local inference is free and overnight/24 h runs are acceptable, so wall-clock is reported for planning only and carries zero rubric weight. The cost that *does* still count is **attempts-to-convergence**, because the attempt budget is finite and each steer draws operator attention.

---

## 0. Actions Before the Next Test

Everything actionable, consolidated. Ordered so that each item either unblocks a later one or removes a confound from the next run. Deep rationale is linked; nothing here needs the rest of the report to act on.

### 0.0 Status — what is already done

**A1, A2, A3 are implemented in `ai-agent-coder` (2026-07-29).** Details in §0.5.

| # | Status | What landed |
|---|---|---|
| **A1** | ✅ **done** | Per-slice attempt default raised **3 → 10** in `pm_lib/slice_ops.py`, `pm_lib/cli.py`, `references/run-state.md`, PM `README.md`. The README now frames >10 as *itself the finding* rather than a budget to raise again |
| **A2** | ✅ **done** | New `skills/lint/` — differential linter, 56 tests, self-lints clean |
| **A3** | ✅ **done** | Wired into Mode A (`scoped-implementation` step 4, `commit` step 3) and Mode B (Developer prompt validation step, PM assess step, PM judgement guidance). `code-review` now defers mechanical hygiene to it; `implementation-plan`'s Validation Plan template asks for it |
| A4–A8 | ⬜ open | Unchanged — see §0.1 |
| P1–P4 | ⬜ open | Plan edits to `docs/MERGER_RATE_PLAN.md`, unchanged |
| R1–R4 | ⬜ open | Next runs, unchanged — §0.3 |

**Not done, deliberately:** lint is *not* a ninth mechanical floor fact. The floor's eight facts are repository-integrity properties never legitimately violated; lint is a quality signal this system places in recorded PM judgement *above* the floor, where a tolerance can be granted with a reason. Making it non-waivable would also let a linter release adding one rule hard-block an unrelated run. It is **mandatory to run and recorded in the assessment** instead. Revisit only if that proves too weak in practice.

### 0.1 Toolkit / skill changes — do first

| # | Action | Where | Why | §  |
|---|---|---|---|---|
| **A1** ✅ | **Raise the per-slice attempt cap: frontier 5, local 10, hard ceiling 10.** Exceeding 10 must *stop and surface*, not be raised again — a slice needing >10 rounds is itself the finding | `pm.py` default (currently 3) + launcher | The binding constraint on local quality. opencode-2 spent 4/5 on Slice 5; run 1 died on exhaustion. Its 88 is a **lower bound**, not a ceiling | §5, §7.3 |
| **A2** ✅ | **Build the `lint` skill as a *differential* check** — "no finding absent at `before_head`", **not** absolute cleanliness. Missing linter ⇒ records **N/A**, never a silent pass | new `skills/lint/` | `main` already carries 4 pyflakes findings; an absolute gate fails on arrival and stops every run on unauthorised debt | §7.7 |
| **A3** ✅ | **Wire lint into both modes.** Mode A: `scoped-implementation` step 4 + `commit` step 3. Mode B: the developer-prompt validation step + PM's own rerun at assess time (**not** a floor fact — see §0.0). Run it *before* commissioning any LLM review | `commit` is the universal chokepoint — one edit covers every path | Would have caught **11 of ~14** non-numerical findings across all five runs. Both reviewers missed every one | §7.7 |
| **A4** | **Add the reviewer reachability rule** — verbatim: *"No finding may be rated P0 or P1 unless the report names a concrete caller in this repository that can produce the offending input. Otherwise it is P3 or it is dropped."* | `references/reviewer-prompt.md` | Free, model-independent, and would have eliminated nearly every over-escalation observed. Sol's problem is calibration, not capability | §7.8 |
| **A5** | **Emit a machine-readable run manifest** (`manifest.json`): harness, model, **effort**, plan sha256, reviewer, attempt cap, tool versions — read from the transcript, not operator input. Refuse to start when the declared config disagrees | `pm.py init` | Two of three stated invariants in this round silently did not hold. This is what makes future comparisons trustworthy | §7.1 |
| **A6** | **Capture the end-to-end stdout as a run artefact**, and require the PM to state it read it | `pm.py` + PM SKILL step 3 | The only artefact that reveals presentation defects. opencode-1's redshift-column bug passed every DoD checkbox | §7.6 |
| **A7** | **Count adjudications per plan section within a run**; past ~3 rulings on one section, stop with a **plan defect** reason | PM SKILL judgement guidance | Bounds the escalation loop that killed run 1, without giving the PM plan-editing authority | §7.9 |
| **A8** | **Track carry-forward items as first-class state**, with an explicit closed/deferred decision at the inheriting slice | `pm notes` / `finalize` | `stored_edges` shipped dead despite being named in the PM's own Slice-1 notes | §4.1 |

**Lint tool set** (A2): `ruff check` + `ruff format --check` (replaces pyflakes/black/isort in one dependency), `markdownlint-cli2` (would have caught opencode-1's malformed table), `clang-format --dry-run -Werror` and `-Wall -Wextra -Wpedantic` for C, `gfortran -fsyntax-only -Wall -Wextra` for Fortran (no good standalone linter exists), plus `codespell` and `git diff --check`. Opt-in tier: `clang-tidy`, `cppcheck`, `lizard`. **Keep rules to *defects*, not style-with-judgement** — this plan mandates duplicated `_mass_bin_edges` by convention, and a DRY rule must not fight the plan.

### 0.2 Plan changes — `docs/MERGER_RATE_PLAN.md`

| # | Action | Why |
|---|---|---|
| **P1** | **Disambiguate the Numerical Domain Contract.** State that *dtype / rank / shape / sign / finiteness / complex* are **always in scope and validated before any coercion**, `AssertionError` required; only *magnitude* extremes are out of domain. Add: "a guard that rejects in-domain input is a defect of equal severity to a missing guard" | Two reviewers read the current text to **opposite** conclusions in run 3. Direct cause of opencode-1's undetected non-compliance, ≥3 steers in run 5, and opencode-2's over-strict float-only guard |
| **P2** | Add to Slice 4 DoD: *"every printed data row identifies its own redshift and mass bin; capture stdout and assert it"* | The one defect class no linter catches — and exactly what sank opencode-1 |
| **P3** | Add a global DoD item: *"no new lint findings on changed files; no new warnings on any supported path; diff touches only the authorized surface"* | Five branches shipped 11 such items between them |
| **P4** | Specify the `AGENTS.md` architecture explicitly: `merger_rate.py` is a **parallel branch off `calc.py`'s output**, not downstream of `plot.py` | 3 of 4 branches drew the diagram wrong |

*Deliberately **not** doing:* a persistent cross-run plan-defect ledger. Rediscovering plan defects each run is accepted as realistic and is itself worth testing (§7.9).

### 0.3 Next runs — confirmed order

| # | Run | Isolates | n |
|---|---|---|---|
| **R1** | **`macstudio/qwen/qwen3.6-27b-q8`** as developer | **Quantisation vs capability.** Must come first — until it exists, every other local result is contaminated by this ambiguity | 3 |
| **R2** | **Same local model via Qwen Code CLI** instead of OpenCode | **Harness vs model** — the largest unresolved confound. Four stalls ("tool call printed as literal text", "idling with uncommitted work") are currently unattributable. Free, since `orchestrator` already supports Qwen Code | 2 |
| **R3** | `qwen3.6-27b-bf16` re-run **with A1–A3 in place**, otherwise unchanged | Whether the local gap is *process*-closable. If this lands ~93, no model change is needed at all | 3 |
| **R4** | `macstudio/qwen/qwen3.6-35b-a3b-bf16` | Local capability, same family/harness. MoE (~3 B active) ⇒ more attempts per night | 3 |

**Controls for every run:** freeze and record the plan sha256; hold the reviewer fixed; verify effort from the transcript (not the statusline); pin resolved dependency versions; fresh worktree with `data/`, `results/`, `figures/` cleared; one attempt cap per comparison. **n ≥ 3 for local/unproven models** — the two local runs differ by 12 points, which two samples cannot separate from noise. Report median and range; never rank configurations whose ranges overlap.

### 0.4 Deferred — worth doing, not blocking

- **Seeded-defect corpus for reviewer evaluation** (§7.8). Inject ~20 known defects into one accepted slice diff; measure precision/recall per reviewer config. Answers "Opus 5 high vs Sol vs local 2-of-3" in an afternoon instead of ten runs. Currently the largest measurement gap — nothing establishes reviewer ground truth.
- **If moving the reviewer to Opus 5 high:** consider moving the **PM off Opus 5**, or keep one Codex pass. Independence is the point of commissioning review; shared lineage means shared blind spots (§7.8).
- **Local reviewer ensemble:** only with **2-of-3 voting**, never a union rule — a union amplifies the false positives that are already the problem. Use the small/MoE models in §6.3; three bf16 30 B models will not co-reside in unified memory.
- Sonnet 5 developer at **medium** effort, to restore the missing effort-matched baseline (§7.2 #8).

### 0.5 Implementation record — the `lint` skill (A2/A3) and attempt cap (A1)

Shipped to `ai-agent-coder` on 2026-07-29. Independently reviewed by **codex `gpt-5.6-sol` at high effort** via the `orchestrator` skill (read-only delegate, run `delegates-20260729-204103-4518`).

**What it is.** `skills/lint/` — `SKILL.md`, `README.md`, `scripts/lint.py`, `config/markdownlint.jsonc`, `tests/test_lint.py` (56 tests, no linter binary required). Three subcommands: `detect`, `check`, `install`. Default question is *"does this change introduce a finding that was not there before?"* — head is linted, the same files are linted at the base ref in a throwaway detached worktree, and only the difference is reported. Pre-existing debt cannot block; `main` in this repo already carries 4 pyflakes findings, so an absolute gate would have failed on arrival.

**Tools:** `ruff check` + `ruff format` (Python), `markdownlint-cli2` (Markdown, incl. **MD056 table-column-count** — empirically confirmed to catch opencode-1's malformed-table defect class), `clang-format` + `cppcheck` (C/C++), `codespell` (any), built-in `git diff --check`. Opt-in: `clang-tidy`, `gfortran -fsyntax-only`. No complexity metric — that is judgement, not a defect.

**Load-bearing invariants:** `check`/`detect` never install (installing is a separate explicit human command, dry-run unless `--yes`, so an unattended Developer cannot use lint to route around its no-dependency-changes rule); a missing linter is `unavailable` coverage, never a pass; a linter that exits unexpectedly with nothing parseable is an **error**, not a clean run; a project's own linter config always wins.

**The review found four reachable P1s. Two I confirmed empirically before fixing:**

| # | Defect | Consequence | Fix |
|---|---|---|---|
| 1 | A linter exiting nonzero with no parseable output was recorded as `ran, 0 findings` | **Verified:** an invalid `ruff` config exits 2 with empty stdout → `verdict: pass`. A broken config silently disabled the gate | Per-tool `ok_codes`; any other code with no findings is an error |
| 2 | `whitespace_check` used `base...HEAD`, but the prescribed pre-commit call is `--base HEAD` | **Verified:** `HEAD...HEAD` is empty, so the check ran on nothing; untracked files were never covered at all | Union of committed range + worktree-vs-HEAD + untracked via `--no-index` |
| 3 | `--staged` listed staged paths but linted worktree content, and had no base ref | A staged defect could pass behind an unstaged fix; also blocked on pre-existing debt | **Mode removed.** `check --base HEAD` before staging covers the same ground correctly |
| 4 | A renamed file's untouched findings all read as newly introduced | The base worktree has nothing at the new path | `git diff --name-status -M` builds an old→new map; base findings are remapped before comparison |

Also fixed: digit-normalisation in the signature was destroying semantic numbers (markdownlint's `Expected: 2; Actual: 3`) — removed, since positional digits already live in the excluded `line` field; `clang-format` silently imposed LLVM style with no `.clang-format` present (now skipped without project config); `--require-coverage` rarely fired because findings outranked it (precedence is now **error > coverage gap > findings**); base-side errors were computed but never printed. Four simplifications accepted: `@contextmanager` for the worktree, `rc` dropped from every parser once classification centralised, install recipes keyed by *binary* (ruff ships two tools in one executable), and command-level tests replacing a source-string assertion.

**A hole I found in my own invariant while the review ran:** a change of only unrecognised extensions (`.txt`, `.rst`) with `codespell` missing reported *empty* coverage and passed — nothing checked, the exact silent pass the invariant forbids. Extension-less tools now always count toward coverage.

**Verification.** 56/56 lint tests; 79/79 orchestrator; full PM suite green apart from one **pre-existing** environment-dependent tmux failure (`test_sessions.TestPaneTextRejoinsHardWraps`) that fails identically on a stashed clean tree. CI no-baggage greps and doc-link reachability pass locally; a `Lint skill tests` step was added to CI. The skill lints its own change clean in both differential and absolute mode.

**Dogfooding caught real defects in my own work** — a dead `platform` import, ten `re.M` aliases, and `typing.List` where this codebase consistently uses builtin generics with `from __future__ import annotations`. It also exposed a design flaw: MD013 line-length fired on every paragraph, contradicting the deliberate no-hard-wrap convention, which is why the shipped Markdown config disables taste rules and project config takes precedence.

**Method note for the next round.** I edited `lint.py` *while* the review was running, so `change.diff` was stale for one function by the time the reviewer read it — it correctly flagged the discrepancy. Freeze the artifact before commissioning review.

---

## 1. Results Summary

| # | Branch | Developer harness / model | Dev effort *(verified)* | Reviewer | Attempts *(scored context)* | Tests | **Grade** | Meets bar? | Wall-clock *(**not scored**)* |
|---|---|---|---|---|---|---|---|---|---|
| 2 | `merger-rate/baseline-2` | claude / claude-sonnet-5 | **high** | codex gpt-5.6-sol (med) | 4 | 232 | **A (96)** | ✅ | 1 h 58 m |
| 3 | `merger-rate/baseline-3` | claude / claude-sonnet-5 | **high** | codex gpt-5.6-sol (med) | **1** | 194 | **A− (92)** | ✅ | 1 h 30 m |
| 5 | `merger-rate/opencode-2` | opencode / `macstudio/qwen/qwen3.6-27b-bf16` | n/a (local) | codex gpt-5.6-sol (med) | 12 | **265** | **B+ (88)** | ✅ marginal | 9 h 21 m |
| 4 | `merger-rate/opencode-1` | opencode / `macstudio/qwen/qwen3.6-27b-bf16` | n/a (local) | codex gpt-5.6-sol (med) | 5 | 212 | **C+ (76)** | ❌ | 5 h 59 m |
| 1 | `merger-rate/baseline-1` | claude / claude-sonnet-5 | medium | **copilot** (1a) → codex (1b) | 8 | 212 | *reference only* | n/a — run **failed to complete** | 8 h 59 m + 47 m calendar; ≈1 h 53 m active |

Wall-clock is the rightmost column deliberately: it carries **no rubric weight**. Attempts are shown because they are a *quality* signal (first-pass contract compliance) and a *risk* signal (the budget is 5 per slice; opencode-2 spent 4/5 on Slice 5), not because they cost time.

**Rank (best → worst): baseline-2 → baseline-3 → opencode-2 → opencode-1.** Run 1 is excluded from ranking: it executed a *different plan* (three-slice, `sha256 f4b91a9d…`) versus runs 2–5 (five-slice, `sha256 4c2408ea…`), and stopped twice for human decisions.

### 1.1 Control violations found (read this first)

| Stated in test design | Actually verified | Impact |
|---|---|---|
| Runs 2–3 developer at **medium** effort | **high** effort | Runs 2–3 are *not* effort-matched to run 1. The Sonnet-vs-local comparison is frontier-at-high vs local, not frontier-at-medium. |
| Reviewer fixed as codex gpt-5.6-sol across all runs | Run 1a used **`copilot`** (`review-*-{drift-audit,code-review}-copilot.md`, 19 files) | Run 1's review behaviour is not comparable to runs 2–5. |
| — | Account tier changed **Pro → Team** between run 1 and runs 2–3 | Probably immaterial to model behaviour; may affect rate-limit-induced stalls. |

Effort was verified from **two independent signals**, per the standing rule that the statusline is not evidence:

1. Developer pane splash banner — `Sonnet 5 with medium effort · Claude Pro` (run 1) vs `Sonnet 5 with high effort · Claude Team` (runs 2–3).
2. Session transcript `effort` field — `claude-sonnet-5/medium` at `2026-07-27T13:51Z` and `T23:38Z`; `claude-sonnet-5/high` at `2026-07-28T09:32Z` and `T10:29Z`; `claude-opus-5/low` at `T10:28Z` (the PM session, matching the stated PM config).

The PM run reports recorded `effort=high` correctly — the discrepancy is in the test log, not the tooling.

### 1.2 Wall-clock method

Times are computed from file-mtime spans across each `.pm/runs/<id>/` tree, then cross-checked against git commit timestamps. The two signals agree everywhere:

| Run | mtime span | First → last dev commit | Agreement |
|---|---|---|---|
| 2 | 118 min | 105 min | ✅ (13 min PM setup + final assessment) |
| 3 | 90 min | 75 min | ✅ (15 min) |
| 4 | 359 min | 311 min | ✅ (48 min) |
| 5 | 561 min | 499 min | ✅ (62 min) |

For runs 2–5 the long internal gaps fall immediately after each `prompt.md` is written — that is the developer working, so span = genuine cost. Only **run 1a** contains a true human pause (6 h 40 m overnight, 00:02 → 06:42), which is why its calendar span (8 h 59 m) is discounted to ≈1 h 16 m active. *Caveat:* in run 5, ~1.3 h of gaps follow review artefacts rather than prompts and cannot be cleanly separated from operator attention.

### 1.3 Grading rubric and acceptable bar *(defined before application)*

| Axis | Weight | What earns full marks |
|---|---:|---|
| Correctness & numerical fidelity | 30 | Correct rate/uncertainty algebra; exact-zero convention holds; injected power law recovered in every bin |
| Plan adherence & contract compliance | 25 | Every DoD item met; authorized surface respected; Numerical Domain Contract obligations discharged |
| Test coverage & test quality | 15 | Independently hand-computed expectations, isolated fixtures, no self-referential assertions |
| Code quality & maintainability | 15 | No dead code, no gratuitous churn, no warnings on supported paths, vectorised where the repo is |
| Scientific robustness & honest claims | 10 | Mock-vs-production labelling; unambiguous, readable numerical output |
| Documentation | 5 | Scoped, accurate, well-formed |

**Runtime carries zero weight.** No axis above references wall-clock, tokens, or cost, so the four grades are runtime-independent as computed. Local inference is free and a 24-hour run is acceptable, which makes elapsed time a scheduling fact, not a quality fact. Two runtime-adjacent quantities *are* legitimately in scope, and are scored only through the axes above:

- **Attempts-to-convergence** — a proxy for first-pass contract compliance (plan adherence), and a live risk because the per-slice attempt budget is finite. It is never scored for the time it consumes.
- **Operator interventions** — steers and stall-nudges that require a human. Scored only where they reveal a defect class.

**Acceptable bar** — mergeable into a research pipeline without rework by a domain scientist:

1. Score **≥ 85/100**, **and**
2. **zero** always-reachable user-visible defects, **and**
3. **zero** outright Numerical Domain Contract failures, **and**
4. green `pytest tests/` and green `src/pipeline.py --validate --merger-rate`, **and**
5. the run **completed without exhausting its attempt budget** (run 1 fails this; all others pass).

Nothing in the bar is time-based. A configuration that reaches ≥ 85 in 24 hours passes exactly as well as one that reaches it in 90 minutes.

---

## 2. Evidence Base

All five branches were checked out into isolated git worktrees and executed against the repo venv (Python 3.12.9, numpy 2.5.1, scipy 1.18.0, h5py 3.16.0, matplotlib 3.11.1). Nothing below is taken from the PM reports' own claims.

| Instrument | What it established |
|---|---|
| `pytest tests/` per branch | 212 / 232 / 194 / 212 / 265 passed, exit 0, all five |
| `pipeline.py --validate --merger-rate` per branch | exit 0, figure + results file produced, all five |
| HDF5 cross-comparison of `results/merger_rate.hdf5` | Identical datasets/attrs; **max relative deviation 2.2 × 10⁻¹⁶** across all five |
| 65-probe contract matrix (`probe.py`) | Plan-derived behaviour checks on all public functions |
| Plot-contract probes hooking `Figure.savefig` and `Axes.errorbar` | Log-scale on every path, masking, display floor, annotation, malformed-`z` raise |
| CLI probes | All 5 frozen composition rows, exclusive-group membership, deferred matplotlib import, help text |
| Isolation test (`rm -rf data results figures` → pytest) | No branch leaks into the repo's output dirs |
| `pyflakes` on `src/` + tests, baselined against `main` | Newly introduced dead code only |
| Integer-`box_size_mpc` fixture probe | Latent dtype brittleness |

Pre-existing lint on `main` (`calc.py: numpy unused`, `pipeline.py: sys unused`, two test items) is **excluded** from all findings — no run introduced it.

### 2.1 The strongest single result

All five implementations, written independently, agree to floating-point round-off on every scientific number — pair fractions, `N_pairs`, `N_gal`, rates, uncertainties, per-redshift timescales — and all six mass bins recover the injected power law with identical fitted slopes:

| Mass bin | slope ± err | expected | consistent |
|---|---|---|---|
| [8.0,8.5) | 0.9268 ± 0.1395 | 1.0000 | ✅ |
| [8.5,9.0) | 1.0274 ± 0.0961 | 1.0000 | ✅ |
| [9.0,9.5) | 0.9063 ± 0.0891 | 1.0000 | ✅ |
| [9.5,10.0) | 0.9891 ± 0.0933 | 1.0000 | ✅ |
| [10.0,10.5) | 0.9632 ± 0.0930 | 1.0000 | ✅ |
| [10.5,11.0) | 1.0922 ± 0.0913 | 1.0000 | ✅ |

This is a genuine five-way cross-validation: the plan's core science — pair fraction → rate density → Poisson propagation → weighted log-log fit with *unscaled* covariance — was implemented correctly by every model tested, including the 27B local one. **The hard part was not the discriminator.**

### 2.2 Contract probe matrix — where branches diverge

65 probes; identical results on all branches except opencode-1.

| Probe | b-1 | b-2 | b-3 | oc-1 | oc-2 |
|---|---|---|---|---|---|
| `compute_pair_fraction` rejects string dtype | ✅ | ✅ | ✅ | ❌ `TypeError` | ✅ |
| `compute_pair_fraction` rejects complex | ✅ | ✅ | ✅ | ❌ `TypeError` | ✅ |
| `fit_log_rate_vs_redshift` rejects string dtype | ✅ | ✅ | ✅ | ❌ **no exception** | ✅ |
| `fit_log_rate_vs_redshift` rejects complex | ✅ | ✅ | ✅ | ❌ **no exception** | ✅ |
| Other 61 probes (exactness, domain gates, `σ_f=0 ⇒ σ_R=0`, box³ scaling, nan-vs-raise split, two-point finite fit, unscaled covariance, `n_sigma` guards, `mass_bin_by` gate) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Total** | **65/65** | **65/65** | **65/65** | **61/65** | **65/65** |

Plot and CLI contracts: **all five pass every probe**, including the two items the plan's appendix cites as prior failures — log-log axes on the all-unusable empty-figure path, and the clipped-lower-bar display floor with its annotation.

---

## 3. Per-Run Assessment

### Run 2 — `merger-rate/baseline-2` · claude / sonnet-5 @ high · **A (96)**

| Axis | Score | Evidence |
|---|---:|---|
| Correctness | 30/30 | 65/65 probes; numbers identical; 232 tests |
| Plan adherence | 24/25 | Every DoD item met; `calc.py` diff is strictly additive (+38/−2, the −2 being the `_save_pairs` signature line) |
| Tests | 14/15 | Largest Sonnet suite; `test_end_to_end_matches_independent_recomputation` is a genuine independent recomputation, not a self-referential restatement |
| Code quality | 14/15 | **Zero** newly introduced dead code or unused imports in `src/` or tests — the only branch achieving this |
| Sci robustness | 10/10 | Table prints redshift on every row; mock-data labelling explicit and prominent |
| Docs | 5/5 | Only branch whose `AGENTS.md` diagram shows the fork correctly (`┌───┴───┐`), i.e. `merger_rate.py` parallel to `plot.py` off `calc.py`; adds proper Units rows |

Weaknesses: none material. Propagates the repo's pre-existing `datetime.utcnow()` deprecation into `merger_rate.py` — house-consistent, and the plan did not authorise fixing it. Needed 4 PM steers (one per slice except Slice 3), each a legitimate catch: a `float()`-before-rank-check on a one-element HDF5 attr (silently accepted under numpy 2.3.2, which `requirements.txt` permits), Slice-2 gate presence checks, a table redshift-column fix, and a deferred-import regression test.

**Why it ranks first:** it is the only branch with no finding at all beyond taste.

### Run 3 — `merger-rate/baseline-3` · claude / sonnet-5 @ high · **A− (92)**

| Axis | Score | Evidence |
|---|---:|---|
| Correctness | 30/30 | 65/65 probes; numbers identical |
| Plan adherence | 23/25 | −2: Slice 5 required adding "the timescale/rate units … to the Configuration Reference"; instead the unit was folded into the `merger_fraction` row's prose. One test (`test_raises_cleanly_when_in_domain_result_overflows`) probes overflow the Numerical Domain Contract declares out of scope — mild test-side scope creep, and it emits a `RuntimeWarning: overflow encountered in power` |
| Tests | 12/15 | **194 tests — the thinnest suite.** Every DoD item is covered, but with the least redundancy of the four |
| Code quality | 14/15 | No dead code in `src/`; one unused test import. Defensively adds `np.asarray(..., dtype=float)` + `ndim` assert to `_count_galaxies_per_mass_bin` (good) |
| Sci robustness | 10/10 | Groups the table by redshift with mass bin varying — coherent and unambiguous |
| Docs | 4/5 | `AGENTS.md` diagram appends `merger_rate.py` *below* `plot.py` in the linear chain, implying it consumes plotting output. Architecturally misleading |

**Most autonomous run of all five: 1 steer across 5 slices**, and the fastest at 90 minutes. Notably, in Slice 3 *both* reviewers returned FAIL and the PM rejected both findings on measured evidence (the "dominant weight" P1 requires a weight ratio of ~10¹⁷; measured relative error was 2.6 × 10⁻¹³ at ratio 10⁴). That is the workflow behaving exactly as designed.

**Why it ranks below baseline-2:** thinnest test margin plus two small documentation/scope blemishes. Nothing here would block a merge.

### Run 5 — `merger-rate/opencode-2` · opencode / qwen3.6-27b-bf16 · **B+ (88)**

| Axis | Score | Evidence |
|---|---:|---|
| Correctness | 28/30 | 65/65 probes and identical numbers, **but** `_load_pair_counts` asserts `np.issubdtype(bs_arr.dtype, np.floating)` and so **rejects an integer `box_size_mpc`** — the only branch that does. The error message reads *"box_size_mpc must be numeric, got dtype int64"*, which is self-contradictory |
| Plan adherence | 23/25 | The plan requires `box_size_mpc` be "a finite positive scalar" and declares its domain as `1e-3`–`1e6` Mpc — float-only dtype is a guard *beyond* the contract that rejects valid input |
| Tests | 14/15 | **265 tests — the largest suite**, and genuinely broad (integer-provenance pass-through, sentinel byte-for-byte gates) |
| Code quality | **10/15** | **3 dead locals in `src/merger_rate.py`** (`stored_edges` L70, `n_mass_bins` L387, `fp_err_2d` L935) and 7 more in tests. Most verbose implementation (1009 lines vs 823–889); the fit function repeats six near-identical dtype asserts where siblings loop |
| Sci robustness | 10/10 | Clearest table of all five; explicit boxed mock-data disclaimer |
| Docs | 4/5 | Correct and well-formed, but the same linear-append diagram inaccuracy as baseline-3 |

**Reachability of the int-`box_size` defect:** unreachable today — `generate_test_data.py` writes a float. But `data_reader.py` is, per `AGENTS.md`, "the only module that changes when switching data sources", and real SAGE support is the named next step. A reader returning `box_size = 500` breaks the entire Slice-2 read path with a misleading message. Latent, but pointed directly at the planned future work.

**`stored_edges` is a workflow failure, not just a code one.** The PM's own Slice-1 carry-forward note reads: *"`_load_pair_counts` assigns a local `stored_edges` … and never uses it — dead after the Slice 1 fix. Slice 2's gate is where that attr acquires a real job … Wiring the gate through it resolves the dead local naturally."* Slice 2 was accepted; the local is still dead at HEAD. Nothing in the workflow verifies that a noted carry-forward item was closed.

**Convergence cost (not a time penalty):** 12 steers, 4 of them on Slice 5 alone (brittle test count, help text, matplotlib import test, README output paths, mutual-exclusivity test), plus two free nudges past session stalls. The concern is not the 9 h 21 m — that is free — it is that **Slice 5 consumed 4 of its 5 attempts**. One more finding and the run would have stopped one slice from done. That is a budget-calibration problem (§7), not a reason to reject the model.

### Run 4 — `merger-rate/opencode-1` · opencode / qwen3.6-27b-bf16 · **C+ (76) — below bar**

| Axis | Score | Evidence |
|---|---:|---|
| Correctness | 25/30 | Numbers identical, but **4 contract-probe failures**. `fit_log_rate_vs_redshift` calls `np.asarray(rates, dtype=float)` *before* any dtype check, so a complex array is **silently accepted with its imaginary part discarded** (only a `ComplexWarning`), and a numeric-looking string array is silently coerced. `compute_pair_fraction` raises bare `TypeError` where the house style requires `AssertionError` |
| Plan adherence | 19/25 | The Numerical Domain Contract explicitly lists "wrong dtype … complex arrays" among malformed inputs every function must reject — and run 5's PM applied exactly that reading three times. Separately, `calc.py` (−9 lines) **rewrote the whole pre-existing provenance-attr block** purely to re-align `=` signs, **deleting the `# Provenance metadata.` comment**. The authorized surface was `_save_pairs`'s signature plus three additive writes |
| Tests | 12/15 | 212 tests, but 4 dead locals/unused imports in the test file |
| Code quality | **10/15** | Dead local `fp_err_all` (L780); emits `UserWarning: No artists with labels found to put in legend` on the *supported* all-unusable path (a warning baseline-2 and opencode-2 were steered to fix); `_count_galaxies_per_mass_bin` uses a Python loop over bins instead of `np.bincount`, i.e. O(n_bins·N) in a repo whose stated core is vectorised |
| Sci robustness | **8/10** | **The most consequential defect found in any run** — see below |
| Docs | **3/5** | `AGENTS.md` Units table gains two **3-column rows in a 2-column table**; GitHub-flavored Markdown silently drops the third cell, so the `T_merge(z)` formula it documents is invisible to readers |

**The results-table defect (always reachable, user-visible).** At `print_merger_rate_table` L801–803:

```python
z_str = f"{z:.1f}" if first_z else ""     # suppresses z
print(col.format(z_str, mlabel, ...))     # ...but repeats mlabel every row
```

The loop nests redshift *inside* mass bin, so redshift is the **varying** column — and it is the one blanked, while the constant mass-bin label repeats. Every reader of `--merger-rate` output sees:

```
 2.0     [8.0,8.5)         199      0.1474    1350    1.30e-06    9.23e-08
         [8.0,8.5)         195      0.1501    1299    1.70e-06    1.22e-07
         [8.0,8.5)         215      0.1631    1318    2.35e-06    1.60e-07
         [8.0,8.5)         178      0.1393    1278    2.33e-06    1.75e-07
```

Three of every four rows in the primary scientific results table carry no redshift label. baseline-2 (`mlabel if first else ""`, z printed every row) is correct — and its PM caught precisely this bug and fixed it in commit `d1605ff`. opencode-1's PM did not. For a numerical results table this is worse than cosmetic: the output is ambiguous as published.

Also: the slope-summary block recomputes the usability mask itself (L826–828), then uses `len(redshifts) - n_usable` in one branch and the fit's own `n_excl` in another — duplicated logic with two different excluded-counts, and the `np.isnan(slope)` branch is partly unreachable behind the earlier `n_usable < 2` continue.

**Why it fails the bar:** score 76 (< 85), an always-reachable user-visible defect, and 4 outright contract failures.

### Run 1 — `merger-rate/baseline-1` · reference only

Executed the superseded three-slice plan and **stopped twice for human decisions** — Slice 2 exhausted its three correction attempts on `box_size**3 * timescale_gyr` overflow, then Slice 3 stopped on a fit-contract domain question. Both stops were unbounded escalation through unreachable float64 boundaries, which is exactly what the Numerical Domain Contract was subsequently written to prevent. The code that eventually landed is good (65/65 probes, 212 tests, identical numbers, 2 unused test imports), but the **run** failed: 8 PM steers, 19 review rounds across two slices, and two human interventions.

### Cross-branch comparison

| Dimension | baseline-2 | baseline-3 | opencode-1 | opencode-2 |
|---|---|---|---|---|
| `merger_rate.py` lines (code) | 889 (716) | 823 (653) | 862 (669) | **1009 (791)** |
| Largest function | 110 | 175 | 143 | **207** |
| Docstring coverage | 16/19 | 13/14 | 15/17 | 13/13 |
| Tests | 232 | 194 | 212 | **265** |
| New dead code in `src/` | **0** | **0** | 1 | **3** |
| Dead code in tests | 0 | 1 | 4 | 7 |
| Contract probes | 65/65 | 65/65 | **61/65** | 65/65 |
| Warnings on supported paths | 0 | 1 (own OOD test) | **1 (legend)** | 0 |
| `calc.py` churn beyond additive | none | none | **reformat + comment deleted** | cosmetic realign |
| Table readability | ✅ | ✅ | **❌** | ✅ (best) |
| Docs well-formed | ✅ | diagram wrong | **malformed table** | diagram wrong |
| Steers | 4 | **1** | 5 | 12 |
| Dev-only time (sum) | 80 min | **52 min** | 323 min | 524 min |

**Convergences worth noting.** All four independently chose: locally duplicated `_mass_bin_edges` (the repo convention the plan mandated over DRY); normalised-weight, centred-predictor normal equations with restored absolute covariance scale; `matplotlib.use("Agg")` before pyplot; a local `MASS_COLORS`; deferred imports in `main()`; and tmp_path-only test fixtures. The plan's prescriptiveness, not model capability, is doing that work.

**Divergences are all discipline, never mathematics.** Every failure mode found in the local-model runs is of one kind: coerce-before-validate, dead locals, gratuitous churn, presentation defects, malformed markup. None is a wrong number.

---

## 4. Attribution by Layer

Attribution is labelled **[confident]** where the evidence isolates one layer and **[uncertain]** where layers are confounded.

### 4.1 The `project-manager` skill and its planning/review workflow

**Worked** — the mechanical floor passed 8/8 on every accepted slice, and authorized-surface enforcement caught real drift. Atomic slice boundaries plus per-slice fresh sessions produced 4/4 completed runs on the five-slice plan versus 0/2 on the three-slice version **[confident]**. The "standing PM ruling" pattern in `notes.md` is the mechanism that stopped the escalation spiral that killed run 1: the PM rejected the same out-of-domain finding class ~10 times with cited measurements rather than re-litigating it per slice **[confident]**.

**Gaps.**

1. **Detection is non-deterministic across identical defects.** The redshift-column bug was caught and fixed in baseline-2 (`d1605ff`) and missed entirely in opencode-1. The PM's independent validation re-runs `pytest` and the pipeline but never *reads the printed table for ambiguity* **[confident]**.
2. **Carry-forward items are never verified closed.** opencode-2's `stored_edges` was explicitly assigned to Slice 2 in the PM's own notes and shipped dead **[confident]**.
3. **No deterministic hygiene gate.** A single `pyflakes` invocation would have caught 4 of the 5 dead-code findings; a Markdown table-shape check would have caught opencode-1's malformed rows. The workflow delegates to LLM reviewers work that cheap static tooling does better, and both reviewers missed all of it **[confident]**.
4. **No machine-readable run manifest.** The report header records harness/model/effort correctly, but nothing forces the operator's declared configuration to be reconciled against it — which is how runs 2–3 came to be logged as medium effort **[confident]**.

### 4.2 Harness

The Claude harness produced **zero stalls** across runs 1–3, with 4–25 min developer latency per slice. The OpenCode harness stalled at least **four times** — the PM notes record *"once printing a tool call as literal text, once idling with uncommitted work"* in run 4 and two more nudges in run 5 — each requiring a free `send` nudge to clear **[confident that the stalls occurred]**.

**[uncertain]** Whether the stalls originate in the local inference server, OpenCode's tool-call parser, or the model's output formatting cannot be separated from this evidence. A literal-text tool call is the classic signature of a *parser/format* mismatch rather than a reasoning failure, which weakly favours harness/integration over model — but that is inference, not measurement.

**[uncertain]** Run 1a's reviewer (Copilot) consumed 19 review rounds over two slices versus Codex's ~2 per slice. This is confounded with the three-slice plan's ~970-line diffs. Do not conclude Copilot is a worse reviewer from this.

### 4.3 Models by role

**PM — Opus 5 at low effort: sufficient, no evidence it is the bottleneck [confident].** It held a written contract against ~10 reviewer escalations, adjudicated two reviewers that *directly contradicted each other*, and wrote carry-forward notes precise enough that later slices avoided repeat mistakes. The role's work is judgement-against-a-document plus bookkeeping, and low effort covered it. Its one systematic miss — never verifying its own noted follow-ups — is a workflow gap, not a capability limit.

**Reviewer — Codex GPT-5.6-Sol at medium: high recall, poor calibration [confident].** Genuine catches include the `float()`-on-singleton-array issue under numpy 2.3.2 (which `requirements.txt` permits but the venv's 2.5.1 masks), `np.allclose` defeating an exact provenance gate, and coercion-before-validation. But it *systematically* escalates unreachable float64/dtype boundaries to P1 — overruled three times in run 5's Slice 2 alone — and in run 3 `drift-audit` demanded removal of the very dtype guards `code-review` faulted the absence of. Both reviewers also missed every dead local, the malformed Markdown, and the table defect. **Net: valuable for semantic contract violations, unreliable for severity ranking, and blind to mechanical hygiene.**

**Developer — Sonnet 5 at high effort [confident].** 1–4 attempts, zero dead code in `src/`, 65/65 contract probes, all DoD items evidenced. baseline-3 completing five elevated-risk slices on **one** steer is the strongest autonomy result in the set.

**Developer — qwen3.6-27b-bf16 local [confident].** Got the genuinely hard parts right: the unscaled-covariance subtlety, the finite two-point fit, the frozen malformed-vs-data raise/return split, the exact-zero uncertainty convention, the full plot masking and display-floor contract. Its failures are uniformly *discipline*: validate-after-coerce, dead locals, cosmetic churn inside a surface it was told to touch additively, ambiguous output, malformed markup. It needed 2.4–3× the attempts — which matters only because the budget is finite, not because of the clock.

**The capability/discipline split is the central finding about models.** Every axis where the local model matched the frontier one is *reasoning under a precise specification*. Every axis where it fell short is *mechanical self-discipline* — and mechanical properties are exactly what deterministic tooling checks better than any model. That asymmetry is why §7's first recommendation is a linter, not a bigger model.

**[uncertain]** Run-to-run variance is large for the local model (76 vs 88; 5 vs 12 steers; 6 h vs 9 h) and n=2. Some of the opencode-1/opencode-2 gap may be sampling noise rather than a reproducible difference.

### 4.4 The plan itself

**The single largest causal factor in run success is plan decomposition [confident].** Three slices → 2/2 runs stopped for human decisions. Five slices, same total scope → 4/4 completed. The plan's own appendix predicted this, and the prediction held.

**What the plan got right.** The Numerical Domain Contract is the mechanism that converted run 1's failure mode into ~10 cheap PM rejections. Per-slice DoD checklists closed the prose-only gap the appendix describes: the log-log-plus-empty-figure requirement, which previously produced a linear-axis empty figure, now passes on **all five** branches.

**Where the plan is defective.**

1. **The Domain Contract is genuinely ambiguous on dtype [confident].** It lists "wrong dtype" and "complex arrays" as malformed inputs every function must reject, while also instructing reviewers not to report out-of-domain behaviour as P0/P1. Two reviewers read the same text to opposite conclusions in run 3. That ambiguity is the direct cause of opencode-1's undetected non-compliance and cost run 5 at least three steers. **Fix:** state explicitly that dtype/rank/complex validation is *always in scope and must precede any coercion*, and that only *magnitude* extremes are out of domain.
2. **No DoD item requires readable output [confident].** opencode-1's table defect satisfies every checkbox in Slice 4's Definition of Done. The plan freezes the display floor and its annotation in detail but never requires that each printed row identify its own redshift.
3. **No mechanical-hygiene DoD item.** Nothing requires the diff to be free of dead code, unused imports, or warnings on supported paths — so five branches shipped 11 such items between them.
4. **Slice 5's `AGENTS.md` instruction says "update the … diagram" without specifying the architecture** — three of four branches drew `merger_rate.py` as downstream of `plot.py` rather than parallel to it.

---

## 5. Practical Model-Selection Guidance

### When frontier paid models are worth it

With runtime free, speed is no longer an argument for paying. What survives is narrower — and worth stating precisely, because it is a *shorter* list than the raw wall-clock gap suggests:

- **When each steer costs scarce human attention rather than machine time.** A local run that converges in 12 steers is free in compute but not free in *your* attention, and the attempt budget is finite: opencode-2 spent 4 of 5 attempts on Slice 5. Overnight autonomy is exactly the case where budget exhaustion turns into a stopped run you discover in the morning. **This is a budget-tuning problem before it is a model problem** — see §7.
- **When the deliverable is *published* output** — figures, tables, docs, anything a reader consumes without re-deriving it. Every presentation defect found (ambiguous table, malformed Markdown, deleted comment, legend warning) came from the local model, and this is the one class that neither more attempts nor a linter reliably catches.
- **When the surface is frozen and churn is itself a contract violation.** Both local runs reformatted lines they were not authorized to touch; neither Sonnet run did. Mechanically checkable, so also fixable without paying — see §7.7.
- **When a slice's risk means a mistake is expensive to discover late** — schema changes, CLI contracts, anything downstream work will build on.

### When local free models are good enough

- **When the specification is prescriptive enough to remove design latitude.** This plan pins formulas, signatures, schemas, raise-vs-return semantics, and validation obligations. Given that, a 27B local model reproduced the science *exactly* — agreement to 2 × 10⁻¹⁶ with a frontier model. That is the headline positive result, and with runtime discounted it is close to decisive: **the local model already clears the quality bar (88) on the axis that matters most.**
- **When the work is mechanical breadth** — adding test cases, extending fixtures, widening coverage. opencode-2 produced the largest suite of any run (265 tests).
- **When a competent reviewer and a PM with a generous attempt budget are in the loop.** The local model *converges*; it converges slowly and it needs to be told the same class of thing more than once.
- **Where local is now clearly the right default:** any run where you are willing to trade attempts for money and the output is code-plus-tests rather than prose-and-figures.

### Which role benefits most from a stronger model

**Ranked by marginal return: Developer > Reviewer > PM.**

1. **Developer — highest.** This is where every discipline defect originates; every quality gap in the ranking traces here. But note the qualifier: a *linter* captures most of that value for free, so "spend on the developer" means *after* the deterministic gates are in place, not instead of them.
2. **Reviewer — high, and currently the weakest link on *calibration*.** GPT-5.6-Sol at medium had good recall but ranked severity badly and contradicted its sibling reviewer. Note the cheaper fix first: a linter closes the entire class of defect both reviewers missed, at effectively zero cost.
3. **PM — lowest marginal return.** Opus 5 at **low** effort was demonstrably adequate. Do not spend here; spend on the Developer.

### Where weaker models introduce unacceptable risk

- **Any output a human will read and trust without re-deriving it.** The redshift-column defect is the archetype: green tests, green pipeline, correct numbers, unusable table.
- **Silent-coercion paths.** opencode-1 accepts a complex array and discards the imaginary part with only a warning. Harmless here because no internal caller does it — but this is exactly how unit errors and precision loss enter numerical code undetected.
- **Guards written beyond the specification.** opencode-2's float-only `box_size_mpc` *rejects valid input* with a self-contradictory message. An over-strict guard is as much a defect as a missing one, and is harder to notice because tests pass.
- **Anything touching a frozen on-disk schema or public CLI** without a mechanical diff check — both local runs churned files outside their authorization.

### Application to scientific computing, modelling, numerical methods, and data analysis

- **The numerical core is the *least* model-sensitive part, given a good plan.** Five models, one algebra, agreement to 2 × 10⁻¹⁶. Invest planning effort in pinning formulas, error propagation conventions, and the malformed-vs-data boundary; the model will then implement it.
- **Scientific risk concentrates at the edges, not the centre** — I/O dtype assumptions, provenance gates, units, plot masking, table labelling. That is precisely where the local model failed and where a domain scientist's trust is actually established.
- **Latent dtype brittleness is the characteristic scientific-code failure.** opencode-2's int-`box_size` rejection is invisible today and breaks exactly when `data_reader.py` is swapped for real SAGE data — the repo's stated next step. Always probe reader paths with plausible alternative dtypes; a green suite will not surface this.
- **For large codebases**, the binding constraint shifts from generation to *context discipline*: knowing which of seven modules may be touched. Both local runs failed the authorized-surface test in a 7-module repo; that failure mode scales badly. Prefer stronger developer models as module count and cross-file coupling grow, and enforce the surface mechanically (`git diff --name-only` against the allow-list) rather than by review.
- **Duplication-versus-DRY conventions must be stated explicitly.** All five correctly duplicated `_mass_bin_edges` *because the plan said to*. Left implicit, this is exactly where models "helpfully" refactor across a frozen boundary.

### The governing case: runtime free, 24 h acceptable, quality non-negotiable

This is the operating regime, so it deserves the sharpest answer.

**Verdict: parity is reachable, but *not* by giving the local model more clock.** The distinction matters. opencode-2's residual 4-point gap to baseline-3 is made of three dead locals, an over-strict dtype guard, and a diagram error — **none of which is compute-limited.** More wall-clock, more sampling, and a bigger context window do not remove a dead local; something has to *notice* it. So "run it for 24 hours" is not itself a strategy. Converting free compute into quality requires spending it on **more verification rounds**, not longer thinking.

The five levers, in strict order of value per unit of effort:

1. **Deterministic gates the model cannot argue with** — `pyflakes`/linter clean on changed files (differential vs. base), `git diff --name-only` ⊆ authorized surface, Markdown table-shape check, "no new warnings on any supported path". These close **11 of the ~14 non-numerical findings** across all five runs, cost approximately nothing, are model-independent, and are unaffected by how long the run takes. This is the single highest-value change available.
2. **Raise the per-slice attempt cap to 10 for local models.** This is the one lever that *directly* converts free runtime into quality, and it is currently the binding constraint: opencode-2 spent 4/5 on Slice 5 and run 1 died on budget exhaustion. The toolkit defaults to 3 and this round ran at 5 — both calibrated for a frontier developer. 10 is the right ceiling, and not merely because it is generous: **a slice needing more than 10 rounds is itself the finding.** Past that, the problem is the plan, the model, or the harness, and the human should be told rather than have budget quietly consumed.
3. **A "reader" DoD item** — capture the final `--merger-rate` stdout and assert every data row is self-identifying. This is the one finding class no linter catches, and it is exactly what sank opencode-1.
4. **Keep a strong reviewer.** Do not economise on developer *and* reviewer simultaneously. Every local-developer defect that was caught was caught by Codex or by the PM's own reading — never by the developer itself.
5. **Re-measure before believing any of this.** With (1)–(3) in place I would expect opencode-2-class output at ~93 versus 88 — genuine parity with baseline-3. **This is informed speculation, not evidence.** It is the experiment, not the result.

The honest summary of the regime: **with runtime discounted, the local 27B model is already an acceptable developer (88 ≥ 85) and the remaining gap is process, not capability.** Close the process gap before concluding you need a bigger model.

---

## 6. Local Models Worth Testing

Scoped to **free local inference** (`macstudio/*`, `macbookpro/*`) — the models actually of interest. Sonnet 5 already arrives free via Claude Code and Copilot is a paid subscription, so hosted options are noted only where they serve as a control. `opencode models` reports **55** entries; **evidence** = grounded in this study, **speculation** = reasoned expectation.

### 6.1 Developer candidates (`macstudio/*`)

| Model | Why | Basis |
|---|---|---|
| **`macstudio/qwen/qwen3.6-35b-a3b-bf16`** | **Top pick.** Same family, generation, and harness as the tested `qwen3.6-27b-bf16`, so it isolates capability cleanly. MoE with ~3 B active parameters, so it should also be quicker than the 27 B dense model — irrelevant to scoring now, but it means more attempts per night, which *is* the lever that converts free compute into quality | speculation (architecture); A/B cleanliness is evidence-based |
| **`macstudio/qwen/qwen3.6-27b-q8`** | Cheapest high-value **control**: identical model, quantised. Until you run this you cannot tell whether any local deficit is capability or quantisation — and that ambiguity contaminates every other local result | evidence-based design |
| `macstudio/zai/glm-5.2-iq4-xs` | Newest-generation local model available, and GLM code models are strong on instruction-following — precisely the observed failure axis (coerce-before-validate, unauthorized churn). Strongest candidate on paper. Caveat: `iq4-xs` is aggressive quantisation, confounding capability with quant — interpret only alongside a q8 control | speculation |
| `macstudio/qwen/qwen3.6-35b-a3b-q8` | Fallback if the bf16 MoE exceeds unified memory alongside anything else | — |
| `macstudio/minimax/minimax-m2.7-q8` | Different architecture family. Tests whether coerce-before-validate and dead-local habits are qwen-specific or general to ~30 B local models — a genuinely interesting question, since a *general* answer means process controls, not model swaps | speculation |
| `macstudio/gemma/gemma-4-31b-it-bf16` | Instruction-tuned rather than code-specialised. Low expectation as a developer; more interesting as a *reviewer* voice (§6.3) | speculation |
| *Avoid for scored developer runs:* `macstudio/moonshotai/kimi-k2.7-code-q2` | q2 is too degraded to attribute any result to the model rather than the quantisation | — |
| *Avoid for developer runs:* `macbookpro/*` (q4, 27–35 B) | Aggressive quant on a smaller machine; keep these for parallel reviewer fan-out (§6.3) | — |

### 6.2 The harness-vs-model control, without paying

`github-copilot/claude-sonnet-5` was the cleanest way to separate harness from model — but it is a paid subscription, so here is the **free substitute**: run **the same local model through a different CLI**. The `orchestrator` skill already supports Qwen Code alongside OpenCode, so pointing Qwen Code at `macstudio/qwen/qwen3.6-27b-bf16` holds the model fixed and changes only the harness. That directly tests whether the four observed stalls ("printing a tool call as literal text", "idling with uncommitted work") are OpenCode's tool-call parsing or the model's output formatting — currently the largest unresolved confound in this study, and resolvable at zero cost.

### 6.3 Reviewer candidates for a local ensemble (§7.8)

For a fan-out of 2–3 concurrent reviewers, **memory footprint matters more than peak capability** — three bf16 30 B models will not co-reside in unified memory and will serialise:

- `macstudio/zai/glm-4.7-flash-bf16` / `-q8` — small and fast, the natural fan-out workhorse
- `macstudio/qwen/qwen3.6-35b-a3b-q8` — MoE, low active-parameter count, so cheap to run concurrently
- `macstudio/gemma/gemma-4-31b-it-bf16` — a deliberately *different* lineage, which is what buys error independence in an ensemble
- `macbookpro/qwen/qwen3.6-35b-a3b-q4` — runs on the second machine, giving genuine hardware parallelism rather than contention

---

## 7. Recommendations

### 7.1 Fix the controls first (blocking — do before any new run)

*Consolidated as §0.1 — this is the rationale.* Item 2 below is **not** blocking; it was deferred to §0.4 on the operator's call, since the effort-matched baseline answers a different question than the next runs do.

1. **Emit a machine-readable run manifest** (`.pm/runs/<id>/manifest.json`) capturing harness, model, effort, plan sha256, reviewer, and account tier for **every** role, read from the transcript rather than from operator input. Have PM refuse to start when the operator's declared config disagrees.
2. **Re-run the Sonnet baseline at medium effort** (2 runs). The current baselines are high-effort, so no effort-matched frontier comparison exists.
3. **Fix the plan's dtype ambiguity** (§4.4) before it costs more steers.

### 7.2 Highest-value next experiments, in order

**§0.3 holds the confirmed run order (R1–R4); this is the fuller menu behind it.** Ordered by information gained per unit of *your* attention — runtime is free, so run length is not a tiebreak. The operator's chosen sequence promotes the quantisation control (#3 here) and the harness control (#5 here) ahead of the process-gate re-run, on the reasoning that both remove confounds that would otherwise contaminate every later local result.

| # | Experiment | Isolates | Runs |
|---|---|---|---|
| 1 | **Deterministic gates added (§7.7), then re-run `qwen3.6-27b-bf16` unchanged** | Whether the local gap is *process*-closable. Cheapest and most decisive: if this lands ~93, no model change is needed at all | 3 |
| 2 | **Attempt cap raised to 10**, local developer | Whether the local model simply needs more rounds — the one lever that converts free runtime into quality | 3 |
| 3 | `macstudio/qwen/qwen3.6-27b-q8` as developer | Quantisation vs capability — must precede interpreting any other local result | 2 |
| 4 | `macstudio/qwen/qwen3.6-35b-a3b-bf16` as developer | Local capability, same family/harness | 3 |
| 5 | **Same local model via Qwen Code CLI** (§6.2) | **Harness vs model** — the biggest confound in this study, resolvable free | 2 |
| 6 | `macstudio/zai/glm-5.2-iq4-xs` as developer | Newest local generation; strongest on-paper instruction-following | 3 |
| 7 | Reviewer: Opus 5 high vs Sol vs local 2-of-3 ensemble, on a **seeded-defect corpus** (§7.8) | Reviewer precision/recall without running full slices | n/a |
| 8 | Sonnet 5 developer at **medium** effort | Effort sensitivity at fixed model; restores the missing effort-matched baseline | 2 |
| 9 | PM at Opus 5 **medium** | Whether PM effort is genuinely saturated at low | 1 |

### 7.3 Controls required for a fair comparison

- **Freeze the plan sha256** and record it (already done; keep it) — run 1's incomparability was a plan change.
- **Hold the reviewer fixed** — violated in run 1a.
- **Hold developer effort fixed**, verified from the transcript, not the statusline or the operator's notes.
- **Pin `requirements.txt` resolution** — the numpy 2.3.2-vs-2.5.1 divergence in `float(np.array([x]))` materially changed one finding's severity. Record the resolved versions in the manifest.
- **Fresh worktree per run**, and clear `data/`, `results/`, `figures/` — stale outputs from a prior baseline run were present during run 2 and the PM had to warn about them explicitly in its notes.
- **Hold max-attempts fixed *within* a comparison, but set it per model class.** 5 throughout this round (the toolkit default is 3). Comparing a local model at 5 against a frontier model at 5 is a fair *autonomy* test but an unfair *quality* test, because it caps the weaker model below its convergence point. Frontier developers need fewer rounds and local ones more, so: **frontier 5, local 10, hard ceiling 10 either way.** Exceeding 10 is not a budget problem to be raised again — it is a signal to stop and surface it. Never mix two different caps in one comparison table.
- **Do not use wall-clock as a control or a tiebreak.** It is a scheduling fact. The comparable resources are attempts, operator interventions, and review rounds.

### 7.4 Repeated runs needed for variance

The local model varied by **12 points and 7 steers** across n=2. Two runs cannot distinguish that from noise.

- **Local / unproven models: n = 3 minimum, n = 5 preferred.**
- **Frontier models: n = 2** sufficient for a mean; n = 3 if the two disagree by more than ~5 points (baseline-2 vs baseline-3 differ by 4 — currently acceptable).
- Report **median and range**, never a single run, and never rank two configurations whose ranges overlap.

### 7.5 Metrics to capture automatically

Per slice and per run, into a machine-readable file:

| Metric | Why the current data was hard to trust |
|---|---|
| **Attempts used / budget, per slice** | The primary convergence metric. opencode-2's 4-of-5 on Slice 5 is the single most important number in this study and appears nowhere machine-readable |
| **Operator interventions** — steers, nudges, manual restarts | The human-attention cost, which *is* scarce. Nudges appear only in prose notes |
| Wall-clock per slice, split developer / reviewer / PM | Informational only (**not** a quality metric) — but needed for scheduling overnight runs. I had to reconstruct it from mtimes |
| Idle vs active time, with gap attribution | Run 1a's 6 h 40 m overnight pause is invisible in any recorded total; without this, "wall-clock" silently conflates model time with human sleep |
| Stall events with the observed symptom | Currently narrative only |
| `pytest` count, exit code, **warning count** | Warning counts (8–22) discriminated real quality and were never recorded |
| `pyflakes` delta vs base branch | Would have caught 11 findings mechanically |
| Diff stats vs authorized surface, flagging any out-of-surface line | Would have caught both `calc.py` churn cases |
| Token/cost per role | Absent entirely; required for any value-per-dollar claim |
| Captured stdout of the final `--merger-rate` run | The one artefact that reveals presentation defects |

### 7.6 Plan and workflow improvements

**Plan** (`docs/MERGER_RATE_PLAN.md`):

1. Rewrite the Numerical Domain Contract to separate the two axes explicitly: *dtype, rank, shape, sign, finiteness, complex — always in scope, validated **before any coercion**, `AssertionError` required*; *magnitude extremes within the declared table — out of scope, absence of a guard is not a defect*. Add: "a guard that rejects in-domain input is a defect of equal severity to a missing guard."
2. Add a DoD item to Slice 4: *"Every printed data row identifies its own redshift and mass bin; capture stdout and assert it."*
3. Add a global DoD item: *"`pyflakes` reports no new findings on changed files; no new warnings on any supported path; the diff touches only the authorized surface."*
4. Specify the `AGENTS.md` architecture explicitly: `merger_rate.py` is a **parallel branch off `calc.py`'s output**, not downstream of `plot.py`.

**Workflow** (`project-manager`):

5. **Run deterministic gates before commissioning any LLM review.** Cheaper, faster, and catches what both reviewers demonstrably missed.
6. **Track carry-forward items as first-class state** with an explicit closed/deferred decision at the slice that inherits them. `stored_edges` shipped dead despite being named in the PM's own notes.
7. **Include the captured end-to-end stdout in the PM's independent validation**, and require the PM to state that it read it.
8. **Record reviewer disagreement as a signal.** In run 3, `drift-audit` and `code-review` reached opposite conclusions on the same code — that is strong evidence of plan ambiguity and should escalate to a plan-clarity flag, not merely be adjudicated and forgotten.

### 7.7 A `lint` skill and where it belongs in the workflow

**Design verdict: add it, but as a *differential* check, and do not make it a floor fact in the naive form.**

The PM floor is *mechanical and non-waivable* — "any failure: steer a fix within budget or stop — never accept." Absolute lint cleanliness cannot meet that standard: `main` in the target repo already has 4 pre-existing pyflakes findings, so an absolute gate fails on arrival and every run stops on debt nobody authorised fixing. But the **differential** form *is* crisply decidable and belongs in the floor:

> **Fact 9.** No linter finding present in the changed files at the slice's head that was absent at `before_head`.

That is mechanical, immune to pre-existing debt, and unarguable — exactly the floor's character. It needs one escape hatch: when no linter is configured for the changed file types, fact 9 records **N/A**, not failure. Never let a missing tool read as a pass.

**Insertion points.** Both modes converge on `commit`, which makes it the highest-leverage single edit:

| Mode | Where | Why there |
|---|---|---|
| **A** | `scoped-implementation` step 4 (*Validate*), before the receipt | The developer fixes its own mess while context is hot and the diff is small |
| **A** | `commit` step 3.5, before staging | The universal chokepoint — one edit covers every path that produces a commit |
| **B** | `pm.py finalize` as floor fact 9 | Mechanical, pre-review, and refuses acceptance rather than relying on a reviewer noticing |
| **B** | Developer prompt DoD | So the developer self-checks before reporting, saving an attempt |

Run it **before** commissioning any LLM review. Deterministic findings are cheaper to produce and unarguable; sending a reviewer at code with dead locals wastes reviewer attention on what a linter already knows, and this study shows the reviewers miss it anyway.

**Tool list — general across C, Python, Markdown, Fortran.** Two tiers, because a "lint skill" that requires per-repo configuration will not get adopted:

*Tier 1 — universal, zero-config, fast (run always):*

| Language | Tool | Catches |
|---|---|---|
| Python | **`ruff check`** | Superset of pyflakes/pycodestyle/isort/pyupgrade, ~100× faster, zero-config. Would have caught all 11 dead-code findings here |
| Python | **`ruff format --check`** | Black-compatible, so it replaces `black` outright — one tool, one dependency |
| C | **`clang-format --dry-run -Werror`** | Formatting drift; the analogue of the `calc.py` realignment churn |
| C | **`gcc`/`clang -Wall -Wextra -Wpedantic`** | Highest value per effort in C — uninitialised reads, sign-compare, unused results |
| Markdown | **`markdownlint-cli2`** | **Would have caught opencode-1's 3-column-row-in-a-2-column-table defect** — the one doc defect both reviewers missed |
| Fortran | **`gfortran -fsyntax-only -Wall -Wextra -std=f2018`** | No good standalone Fortran linter exists; the compiler's own warnings are the pragmatic answer |
| Any | **`codespell`** | Typos in identifiers, comments, and docs — cheap and language-agnostic |
| Any | **`git diff --check`** | Whitespace errors and conflict markers, built in |

*Tier 2 — deeper, opt-in per repo (slower, needs config):*

| Language | Tool | Notes |
|---|---|---|
| C | `clang-tidy` | Real static analysis (null deref, resource leaks). Needs `compile_commands.json` — gate on its presence |
| C | `cppcheck --enable=warning,portability` | No compile DB needed; good complement |
| Python | `mypy` / `pyright` | Only where annotations exist; this repo has none, so it would be noise |
| Python | `bandit` | Security-focused; low value for numerical code |
| Fortran | `fortitude` (ruff-family, emerging) | Worth watching; not yet mature enough to gate on |
| Any | `lizard` | Cyclomatic complexity — would have flagged opencode-2's 207-line function |

**Two design cautions.** First, **never let the linter's opinion outrank the plan.** If a plan mandates duplicated `_mass_bin_edges` per repo convention, a DRY-flavoured lint rule must not override it — keep the rule set to *defects* (unused, unreachable, uninitialised, malformed) and out of *style-with-judgement*. Second, **record the tool version in the manifest.** A linter upgrade that adds a rule would otherwise look like a developer regression; the differential form mostly handles this, but only if both sides run the same binary.

### 7.8 Reviewer configuration: is an ensemble multiplying weak by weak?

The concern is well-founded but resolves on *which* error the ensemble is aimed at.

**Diagnosis first: Sol's problem is calibration, not capability.** It found genuinely subtle defects — `float(np.array([x]))` being numpy-version-dependent, `np.allclose` silently defeating an exact provenance gate, coercion-before-validation. A weaker model would miss those. What it did badly was *rank* severity, escalating unreachable float64 boundaries to P1 until the PM overruled the same class ~10 times. That distinction matters, because a bigger reviewer fixes capability, and this is not a capability problem.

**The cheapest fix is free and comes first.** Put a reachability test in the reviewer prompt:

> No finding may be rated P0 or P1 unless the report names a concrete caller in this repository that can produce the offending input. If no such caller exists, it is P3 or it is dropped.

That single sentence would have eliminated nearly every over-escalation observed, at zero cost, for any reviewer model.

**On Opus 5 at high effort as reviewer.** Likely better calibrated, and it would probably respect "the Domain Contract binds reviewers too" more reliably. But there is a structural cost worth naming: **the PM is already Opus 5.** Independence is the entire point of commissioning review, and a reviewer sharing the PM's lineage shares its blind spots — the PM would be checking its own reasoning with the same instrument. Fresh context, a different prompt, and a different effort level preserve *some* independence, but less than a different vendor does. If you move the reviewer to Opus 5, consider moving the PM off it, or keep one Codex pass for lineage diversity.

**On the local ensemble — the answer depends on the failure mode you are targeting:**

- Ensembles improve **recall** when errors are independent (different models miss different things) — that is the union of findings.
- Ensembles degrade **precision**, because false positives also union. **Precision is exactly what is already broken.** Three local reviewers under a union rule would produce *more* spurious P1s, and the PM pays an attempt for each one it chases. That configuration makes the current problem worse — this is the "multiplying weak by weak" concern, and for a union rule it is correct.

**The fix is the aggregation rule, not the models.** Require **k-of-n agreement** (2-of-3) to raise anything above P2. That inverts the ensemble from a recall amplifier into a *precision filter*: idiosyncratic false positives are suppressed, while any defect two independent models both see survives. Disagreement then becomes usable signal in its own right — a 1-of-3 finding is a plan-ambiguity flag, which is precisely what the run-3 drift-audit-vs-code-review contradiction should have been recorded as.

**Two practical cautions.**

1. **Parallel local reviewers are probably not free in wall-clock.** Three bf16 ~30 B models will not co-reside in unified memory on one Mac Studio and will serialise on memory bandwidth. Use the small/MoE models in §6.3, or split across both machines, if genuine parallelism is the goal. (With runtime discounted this matters less — but it does mean "same elapsed time" should not be assumed.)
2. **No LLM reviewer configuration addresses the defects that actually escaped here.** Every finding both reviewers missed was mechanical. Adding reviewers of any strength does not fix that; §7.7 does.

**Recommended order:** (1) linter gate, (2) reachability rule in the reviewer prompt, (3) keep a single strong semantic reviewer, (4) *then* test the local 2-of-3 ensemble — and test it on a **seeded-defect corpus**, not on live runs. Take one accepted slice diff, inject ~20 defects of known class and severity (a dead local, a silent coercion, an off-by-one in a bin edge, a units error, an ambiguous table row), and measure precision and recall per configuration. That answers "is Opus 5 high better than Sol, and is 2-of-3 local better than either" in an afternoon, rather than needing another ten full runs — and it is currently the largest measurement gap in this whole programme, since nothing here establishes reviewer ground truth.

### 7.9 Plan defects that only surface at implementation time

**Is the system already working? Largely yes — and the evidence says so plainly.** Run 1, on a plan without the Numerical Domain Contract, correctly *refused to ship*: it stopped twice rather than accept code it could not justify. Runs 2–5, on a plan with the Contract, all completed, with the PM overruling roughly ten reviewer escalations it judged out-of-domain. That is the intended behaviour of a system whose whole design premise is that the PM is accountable and the plan is frozen. **The strict-slice architecture is not the problem here, and it should not be loosened.**

**The hypothesis that repetition manufactured this finding is half right, and worth conceding.** Running one plan five times does inflate the visibility of its ambiguities: I could see "same defect, five times" precisely because there were five runs. In genuine single-shot use, a PM overriding a miscalibrated reviewer finding *is* the system working, and the marginal cost — a few extra reviewer round-trips — is acceptable. So this is not evidence that the workflow is broken.

**Cross-run memory of plan defects is deliberately *not* wanted.** An earlier draft of this report recommended a persistent `plan-findings.md` ledger so later runs would not rediscover the same ambiguity. That recommendation is **withdrawn on the operator's decision**, and the reasoning is sound: an imperfect plan is the *realistic* condition, plans will not normally be executed five times, and re-encountering a defect fresh each run is itself a useful test of how the system copes. Optimising the rediscovery away would remove the very behaviour under test. Within-run curation via `notes.md` already prevents a *single* run re-litigating a settled ruling — that is the level where it matters, and it already works.

**One gap does remain, and it is about bounding escalation inside a run, not remembering across runs.** There is no loop detector on adjudication. Run 1 died from unbounded escalation; the Domain Contract fixed *that specific* ambiguity, but the general mechanism was never built, so the next plan with a different ambiguity fails the same way. The SKILL has the right diagnostic instinct already ("same-shape failures across a clean relaunch point at the plan or task; shape-shifting failures point at the model") — but it is stated per-slice, and the signal here was only legible *across* slices within one run: run 5's own notes read "I rejected the same out-of-domain finding three times in Slice 2" and "applied three times in Slice 3 alone."

*Fix:* count rulings per plan section within the run. Past roughly the third ruling on one section, the PM should stop and flag a **plan defect** rather than keep grinding — a materially different stop reason from "developer failed", and one that hands the human something they can act on immediately. This composes naturally with the attempt cap (§7.3): both are ways of saying *this run has learned something the human needs to decide*, rather than burning budget silently. It also bounds run 1's failure mode, where the PM kept steering toward a contract that could not be satisfied as written.

**What not to do.** Do not give the PM authority to amend a plan mid-run. The plan digest is floor fact 1, and a PM that can edit its own acceptance criteria has no accountability story left. The flexibility belongs in *escalation quality* — a better-informed, earlier stop — not in loosening the frozen contract. The current strictness is what made this whole comparison measurable in the first place.

---

## 8. Threats to These Conclusions

- **n = 2 per developer configuration.** Every quantitative comparison is a 2-sample estimate. The baseline-2/baseline-3 spread (4 points) and the opencode-1/opencode-2 spread (12 points) are both within plausible sampling noise for a single configuration.
- **Effort mismatch** means "frontier vs local" here is *frontier-at-high* vs local. A medium-effort Sonnet baseline may narrow the gap.
- **The local runs were capped below their convergence point.** Both ran at max-attempts 5, and opencode-2 used 4 of 5 on its last slice. Its 88 is therefore a *lower bound* on what that model reaches at a cap of 10 — which is exactly why experiment 7.2#2 exists. Do not read 88 as the model's ceiling.
- **Harness and model are confounded** for the local runs. Nothing in this evidence separates OpenCode-the-harness from qwen-the-model. Experiment 7.2#5 resolves it free.
- **Grading weights are my judgement**, though the underlying findings are all mechanically reproducible. A reader who weights test-suite breadth above code hygiene would rank opencode-2 above baseline-3.
- **Not attempted:** token/cost accounting (no data), review-quality scoring against a seeded-defect corpus, and any assessment of whether the plan's *science* is correct beyond internal consistency — the mock-data slope check validates the conversion chain, not the astrophysics.

---

## 9. Bottom Line

The `project-manager` skill executed this plan **successfully in 4 of 4 attempts on the five-slice version and 0 of 2 on the three-slice version**. Plan decomposition, not model choice, was the difference between finishing and stopping.

Given a plan this prescriptive, **model choice did not affect the science at all** — five independent implementations agree to 2 × 10⁻¹⁶ and all six mass bins recover the injected power law identically. What model choice bought was **discipline**, not correctness: Sonnet 5 delivered clean, correctly-labelled output in 1–4 attempts; the 27 B local model delivered correct numbers wrapped in dead code, an ambiguous results table, malformed docs, and unauthorized churn, in 5–12.

**With runtime excluded from scoring, the conclusion shifts in the local model's favour.** It already clears the acceptable bar at 88, its 88 is a lower bound imposed by an attempt budget calibrated for a frontier model, and its entire residual deficit is *mechanical* — the class of defect that deterministic tooling catches better than any model, at any price. Eleven of roughly fourteen non-numerical findings across all five runs would have fallen to `ruff`, a diff-surface check, and a Markdown table linter. The current workflow omits all three in favour of LLM review that missed every one of them.

So the priority order is: **linter gate first, attempt budget second, bigger model last.** Spending on the Developer is still the highest-value *model* change, but it is the third lever, not the first. Do not spend on the PM — Opus 5 at low effort held a written contract against ten reviewer escalations and adjudicated two reviewers that contradicted each other. On the Reviewer, fix calibration with a free prompt rule (name a reachable caller or it is not a P1) before buying capability; and if you fan out to local reviewers, use 2-of-3 voting, which makes the ensemble a precision filter rather than the recall amplifier that would make the current problem worse.

Two closing notes. **Plan defects surfacing at implementation time are handled adequately today** — run 1 correctly refused to ship, runs 2–5 shipped with the PM overruling ~10 escalations on cited evidence — and cross-run memory of those defects is deliberately not wanted (§7.9); the one worthwhile addition is an in-run adjudication counter so a repeatedly-contested plan section produces an early, well-informed stop rather than silent budget burn. And **fix the controls before the next comparison**: two of the three stated invariants in this round did not actually hold.
