# Model Evaluation 01 — Baseline Merger-Rate Plan, Five Runs

**Series:** Report 1 of 3 · [Index](model-eval-00-index.md) · Next: [Report 02](model-eval-02-20260731-merger-rate-plan-revised.md)
**Date:** 2026-07-29
**Plan under test:** `docs/MERGER_RATE_PLAN.md` — 5-slice version (`sha256 4c2408ea…`) for runs 2–5; superseded 3-slice version (`sha256 f4b91a9d…`) for run 1
**Runs compared:** 5

**Question:** How effectively does the `project-manager` skill execute this plan, and how does Developer harness/model choice change the outcome?

**Verdict:** All four completed runs produce **numerically identical science** — five independent implementations agree to 2×10⁻¹⁶ and all six mass bins recover the injected power law identically. Ranking is decided entirely by contract discipline, code hygiene, and output usability. **baseline-2 (96) > baseline-3 (92) > opencode-2 (88) > opencode-1 (76).** Three of four clear the bar; opencode-1 does not. Plan *decomposition* — not model choice — was the difference between finishing and stopping: the 5-slice plan completed 4/4, the 3-slice version 0/2.

> **Two control violations invalidate part of the stated test design.** Runs 2–3 ran the Developer at **high** effort, not medium; run 1's Reviewer was **Copilot**, not Codex. See §2.3. Wall-clock carries **zero rubric weight** (§3) — local inference is free and overnight runs are acceptable.

---

## 1. Run Inventory

| # | Branch | Developer (harness / model) | Dev effort *(verified)* | Reviewer | Attempts | Tests | Wall-clock *(not scored)* | Status |
|---|---|---|---|---|---|---|---|---|
| 2 | `merger-rate/baseline-2` | claude / claude-sonnet-5 | **high** | codex gpt-5.6-sol (med) | 4 | 232 | 1h 58m | ✅ complete |
| 3 | `merger-rate/baseline-3` | claude / claude-sonnet-5 | **high** | codex gpt-5.6-sol (med) | **1** | 194 | 1h 30m | ✅ complete |
| 5 | `merger-rate/opencode-2` | opencode / qwen3.6-27b-bf16 | n/a (local) | codex gpt-5.6-sol (med) | 12 | **265** | 9h 21m | ✅ complete |
| 4 | `merger-rate/opencode-1` | opencode / qwen3.6-27b-bf16 | n/a (local) | codex gpt-5.6-sol (med) | 5 | 212 | 5h 59m | ✅ complete |
| 1 | `merger-rate/baseline-1` | claude / claude-sonnet-5 | medium | **copilot** → codex | 8 | 212 | ≈1h 53m active | ❌ **failed to complete** |

Run 1 is excluded from ranking: it executed a *different plan* (3-slice) and stopped twice for human decisions.

---

## 2. Method & Evidence

All five branches were checked out into isolated git worktrees and executed against the repo venv (Python 3.12.9, numpy 2.5.1, scipy 1.18.0, h5py 3.16.0, matplotlib 3.11.1). **Nothing below is taken from the PM reports' own claims.**

| Instrument | What it established |
|---|---|
| `pytest tests/` per branch | 212 / 232 / 194 / 212 / 265 passed, exit 0, all five |
| `pipeline.py --validate --merger-rate` per branch | exit 0, figure + results file produced, all five |
| HDF5 cross-comparison of `merger_rate.hdf5` | Identical datasets/attrs; **max relative deviation 2.2×10⁻¹⁶** |
| 65-probe contract matrix (`probe.py`) | Plan-derived behaviour checks on all public functions |
| Plot-contract probes hooking `Figure.savefig` / `Axes.errorbar` | Log-scale on every path, masking, display floor, annotation |
| CLI probes | All 5 frozen composition rows, exclusive-group membership, deferred import |
| Isolation test (`rm -rf data results figures` → pytest) | No branch leaks into the repo's output dirs |
| `pyflakes` baselined against `main` | Newly introduced dead code only |

Pre-existing lint on `main` is **excluded** from all findings — no run introduced it.

### 2.1 The strongest single result

All five implementations agree to floating-point round-off on every scientific number, and all six mass bins recover the injected power law with identical fitted slopes (`0.9268, 1.0274, 0.9063, 0.9891, 0.9632, 1.0922` against `expected = 1.0`, all consistent).

> The plan's core science — pair fraction → rate density → Poisson propagation → weighted log-log fit with *unscaled* covariance — was implemented correctly by every model tested, including the 27B local one. **The hard part was not the discriminator.**

### 2.2 Contract probe matrix — where branches diverge

65 probes; identical results on all branches except opencode-1.

| Probe | b-1 | b-2 | b-3 | oc-1 | oc-2 |
|---|---|---|---|---|---|
| `compute_pair_fraction` rejects string dtype | ✅ | ✅ | ✅ | ❌ `TypeError` | ✅ |
| `compute_pair_fraction` rejects complex | ✅ | ✅ | ✅ | ❌ `TypeError` | ✅ |
| `fit_log_rate_vs_redshift` rejects string dtype | ✅ | ✅ | ✅ | ❌ **no exception** | ✅ |
| `fit_log_rate_vs_redshift` rejects complex | ✅ | ✅ | ✅ | ❌ **no exception** | ✅ |
| Other 61 probes | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Total** | 65/65 | 65/65 | 65/65 | **61/65** | 65/65 |

### 2.3 Control violations (read before trusting any comparison)

| Stated in test design | Actually verified | Impact |
|---|---|---|
| Runs 2–3 Developer at **medium** effort | **high** effort | Runs 2–3 are *not* effort-matched to run 1. The comparison is frontier-at-high vs local. |
| Reviewer fixed as codex gpt-5.6-sol | Run 1a used **copilot** (19 files) | Run 1's review behaviour is not comparable. |
| — | Account tier changed Pro → Team between runs | Probably immaterial; may affect rate-limit stalls. |

Effort was verified from two independent signals (pane splash banner and session transcript `effort` field), per the standing rule that the statusline is not evidence. The PM run reports recorded effort correctly — **the discrepancy was in the test log, not the tooling.**

---

## 3. Rubric *(defined before application)*

| Axis | Weight | What earns full marks |
|---|---:|---|
| Correctness & numerical fidelity | 30 | Correct rate/uncertainty algebra; exact-zero convention holds; injected power law recovered in every bin |
| Plan adherence & contract compliance | 25 | Every DoD item met; authorized surface respected; Numerical Domain Contract discharged |
| Test coverage & test quality | 15 | Hand-computed expectations, isolated fixtures, no self-referential assertions |
| Code quality & maintainability | 15 | No dead code, no gratuitous churn, no warnings on supported paths, vectorised where the repo is |
| Scientific robustness & honest claims | 10 | Mock-vs-production labelling; unambiguous, readable numerical output |
| Documentation | 5 | Scoped, accurate, well-formed |

**Runtime carries zero weight.** Local inference is free and a 24-hour run is acceptable, so elapsed time is a scheduling fact, not a quality fact. Two runtime-adjacent quantities *are* in scope, scored only through the axes above: **attempts-to-convergence** (a proxy for first-pass contract compliance, and a live risk because the budget is finite) and **operator interventions**.

**Acceptable bar** — mergeable into a research pipeline without rework: score ≥85, **and** zero always-reachable user-visible defects, **and** zero outright Domain Contract failures, **and** green `pytest` + green `--validate --merger-rate`, **and** the run completed without exhausting its attempt budget.

---

## 4. Results & Ranking

| Rank | Branch | Developer | Corr /30 | Plan /25 | Test /15 | Code /15 | Sci /10 | Doc /5 | **Total /100** | Norm. | Bar |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `baseline-2` | sonnet-5 @ high | 30 | 24 | 14 | 14 | 10 | 5 | **96** | 96% | ✅ |
| 2 | `baseline-3` | sonnet-5 @ high | 30 | 23 | 12 | 14 | 10 | 4 | **92** | 92% | ✅ |
| 3 | `opencode-2` | qwen3.6-27b-bf16 | 28 | 23 | 14 | 10 | 10 | 4 | **88** | 88% | ✅ marginal |
| 4 | `opencode-1` | qwen3.6-27b-bf16 | 25 | 19 | 12 | 10 | 8 | 3 | **76** | 76% | ❌ |
| — | `baseline-1` | sonnet-5 @ medium | — | — | — | — | — | — | *reference only* | — | n/a |

**`baseline-2` (96).** The only branch with no finding at all beyond taste. Zero newly-introduced dead code in `src/` or tests. Only branch whose `AGENTS.md` diagram shows the architecture correctly. Its 4 PM steers were each legitimate catches.

**`baseline-3` (92).** **The most autonomous run of all five: 1 steer across 5 slices**, and the fastest at 90 minutes. Thinnest test suite (194) and two small documentation/scope blemishes. Notably, in Slice 3 *both* reviewers returned FAIL and the PM rejected both findings on measured evidence — the workflow behaving exactly as designed.

**`opencode-2` (88).** Largest suite of any run (265 tests), clearest results table, explicit mock-data disclaimer. Held back by **3 dead locals in `src/`** and an over-strict guard: `_load_pair_counts` asserts a floating dtype and so **rejects an integer `box_size_mpc`** with the self-contradictory message *"must be numeric, got dtype int64"*. Unreachable today, but pointed directly at the planned SAGE-reader work. Consumed **4 of 5 attempts on Slice 5** — one more finding and the run would have stopped one slice from done.

**`opencode-1` (76) — below bar.** Four contract-probe failures: `fit_log_rate_vs_redshift` coerces before any dtype check, so a **complex array is silently accepted with its imaginary part discarded**. Rewrote a pre-existing provenance block purely to realign `=` signs, deleting a comment, outside its authorized surface. Its `AGENTS.md` Units table gained 3-column rows in a 2-column table, so the documented formula is invisible in rendered Markdown.

Its decisive defect is **always-reachable and user-visible**: the results-table loop nests redshift *inside* mass bin, so redshift is the varying column — and it is the one blanked, while the constant mass-bin label repeats. Three of every four rows in the primary scientific results table carry no redshift label. `baseline-2` had the identical bug and **its PM caught and fixed it**; opencode-1's PM did not.

### 4.1 Cross-branch comparison

| Dimension | baseline-2 | baseline-3 | opencode-1 | opencode-2 |
|---|---|---|---|---|
| `merger_rate.py` lines (code) | 889 (716) | 823 (653) | 862 (669) | **1009 (791)** |
| Largest function | 110 | 175 | 143 | **207** |
| Tests | 232 | 194 | 212 | **265** |
| New dead code in `src/` | **0** | **0** | 1 | **3** |
| Dead code in tests | 0 | 1 | 4 | 7 |
| Contract probes | 65/65 | 65/65 | **61/65** | 65/65 |
| `calc.py` churn beyond additive | none | none | **reformat + comment deleted** | cosmetic realign |
| Table readability | ✅ | ✅ | **❌** | ✅ (best) |
| Docs well-formed | ✅ | diagram wrong | **malformed table** | diagram wrong |
| Steers | 4 | **1** | 5 | 12 |

**Convergences worth noting.** All four independently chose locally-duplicated `_mass_bin_edges` (the repo convention the plan mandated over DRY), normalised-weight centred-predictor normal equations, `matplotlib.use("Agg")`, deferred imports in `main()`, and tmp_path-only fixtures. **The plan's prescriptiveness, not model capability, is doing that work.**

**Divergences are all discipline, never mathematics.** Every local-model failure is coerce-before-validate, dead locals, gratuitous churn, presentation defects, or malformed markup. **None is a wrong number.**

---

## 5. Role Performance

### 5.1 Developer

| Model | Harness | Runs | Result | Key strength | Key weakness |
|---|---|---|---|---|---|
| **claude-sonnet-5** @ high | claude | 2 | 96, 92 | 1–4 attempts, zero dead code in `src/`, 65/65 probes, all DoD evidenced. `baseline-3` cleared five elevated-risk slices on **one** steer — the strongest autonomy result in the set. | None material at this plan difficulty. |
| **qwen3.6-27b-bf16** | opencode | 2 | 88, 76 | Got the genuinely hard parts right unprompted: unscaled covariance, finite two-point fit, frozen malformed-vs-data split, exact-zero uncertainty, full plot masking and display-floor contract. | Uniformly *discipline*: validate-after-coerce, dead locals, cosmetic churn inside a surface it was told to touch additively, ambiguous output, malformed markup. Needed 2.4–3× the attempts. |

> **The capability/discipline split is the central finding about models.** Every axis where the local model matched the frontier one is *reasoning under a precise specification*. Every axis where it fell short is *mechanical self-discipline* — exactly what deterministic tooling checks better than any model at any price.

**[uncertain]** Run-to-run variance is large for the local model (76 vs 88; 5 vs 12 steers) at n=2. Some of the gap may be sampling noise.

### 5.2 PM

| Model | Effort | Runs | Verdict |
|---|---|---|---|
| **Opus 5** | low | 5 | **Sufficient; no evidence it is the bottleneck.** Held a written contract against ~10 reviewer escalations, adjudicated two reviewers that *directly contradicted each other*, and wrote carry-forward notes precise enough that later slices avoided repeat mistakes. Low effort covered it. |

Its one systematic miss — never verifying its own noted follow-ups — is a workflow gap, not a capability limit. **Lowest marginal return of the three seats; do not spend here.**

### 5.3 Reviewer

| Model | Effort | Role | Verdict |
|---|---|---|---|
| **codex gpt-5.6-sol** | medium | code-review + drift-audit | **High recall, poor calibration.** Genuine catches: `float()`-on-singleton-array being numpy-version-dependent, `np.allclose` defeating an exact provenance gate, coercion-before-validation. But it *systematically* escalates unreachable float64/dtype boundaries to P1 — overruled three times in run 5's Slice 2 alone. In run 3, `drift-audit` demanded removal of the very dtype guards `code-review` faulted the absence of. |
| **copilot** | — | run 1a only | 19 review rounds over two slices vs Codex's ~2 per slice. **[uncertain]** Confounded with the 3-slice plan's ~970-line diffs. Do not conclude Copilot is worse from this. |

> **Both reviewers missed every dead local, the malformed Markdown, and the results-table defect.** Net: valuable for semantic contract violations, unreliable for severity ranking, **blind to mechanical hygiene**.

### 5.4 The plan as a participant

**The single largest causal factor in run success is plan decomposition [confident].** Three slices → 2/2 runs stopped for human decisions. Five slices, same total scope → 4/4 completed.

Plan defects found:

1. **The Numerical Domain Contract is genuinely ambiguous on dtype.** It lists "wrong dtype" and "complex arrays" as malformed inputs every function must reject, while also instructing reviewers not to report out-of-domain behaviour as P0/P1. Two reviewers read it to opposite conclusions. Direct cause of opencode-1's undetected non-compliance and ≥3 steers in run 5.
2. **No DoD item requires readable output.** opencode-1's table defect satisfies every Slice 4 checkbox.
3. **No mechanical-hygiene DoD item** — five branches shipped 11 such items between them.
4. **Slice 5's `AGENTS.md` instruction under-specifies the architecture** — 3 of 4 branches drew the diagram wrong.

---

## 6. Cross-Run Findings

**Attempts, not clock, are the binding constraint.** `opencode-2` spent 4 of 5 attempts on Slice 5; run 1 died on budget exhaustion. The toolkit default (3) and this round's setting (5) were both calibrated for a frontier Developer.

**Deterministic tooling would have caught what LLM review missed.** A single `pyflakes` invocation catches 4 of 5 dead-code findings; a Markdown table-shape check catches opencode-1's malformed rows. **11 of ~14 non-numerical findings across all five runs** fall to `ruff` + a diff-surface check + a Markdown linter. The workflow delegated to LLM reviewers work that cheap static tooling does better.

**Detection is non-deterministic across identical defects.** The redshift-column bug was caught in `baseline-2` and missed entirely in `opencode-1`. The PM's validation re-runs `pytest` and the pipeline but never *reads the printed table*.

**Carry-forward items are never verified closed.** `opencode-2`'s `stored_edges` was explicitly assigned to Slice 2 in the PM's own notes and shipped dead.

**The escalation loop is unbounded.** Run 1 died from it. The Domain Contract fixed *that specific* ambiguity, but the general mechanism was never built — so the next plan with a different ambiguity fails the same way. The signal was legible only *across* slices within one run: run 5's own notes read *"I rejected the same out-of-domain finding three times in Slice 2."*

---

## 7. Recommendations

### Model selection by role

**Ranked by marginal return: Developer > Reviewer > PM.**

| Seat | Recommendation | Rationale |
|---|---|---|
| **Developer** | Highest-value model change — **but third lever overall.** A linter captures most of that value for free. | Every discipline defect originates here. |
| **Reviewer** | Fix *calibration* with a free prompt rule before buying capability: *"no P0/P1 unless the report names a concrete caller in this repository that can produce the offending input."* | Sol's problem is calibration, not capability. |
| **PM** | **Do not spend here.** Opus 5 at low effort was demonstrably adequate. | Held a contract against ten escalations. |

### When frontier is worth it

- When each steer costs scarce **human attention** rather than machine time, and the attempt budget can be exhausted overnight.
- When the deliverable is **published output** — figures, tables, docs. Every presentation defect found came from the local model, and this is the one class neither more attempts nor a linter reliably catches.
- When the surface is frozen and **churn is itself a contract violation**.

### When local free models are good enough

- When the specification is **prescriptive enough to remove design latitude**. Given that, a 27B local model reproduced the science exactly.
- When the work is **mechanical breadth** — `opencode-2` produced the largest suite of any run.
- When a competent Reviewer and a **generous attempt budget** are in the loop.

### The governing case: runtime free, 24h acceptable, quality non-negotiable

> **Parity is reachable, but *not* by giving the local model more clock.** `opencode-2`'s residual 4-point gap is three dead locals, an over-strict guard, and a diagram error — **none of which is compute-limited.** More wall-clock does not remove a dead local; something has to *notice* it.

Levers in strict order of value:

1. **Deterministic gates the model cannot argue with** — differential lint, `git diff --name-only` ⊆ authorized surface, Markdown table-shape check, no-new-warnings. Closes 11 of ~14 non-numerical findings at ~zero cost.
2. **Raise the per-slice attempt cap to 10 for local models.** The one lever that directly converts free runtime into quality. **A slice needing >10 rounds is itself the finding.**
3. **A "reader" DoD item** — capture final stdout and assert every data row is self-identifying. The one class no linter catches.
4. **Keep a strong Reviewer.** Do not economise on Developer *and* Reviewer simultaneously.

**Priority order: linter gate first, attempt budget second, bigger model last.**

---

## 8. Changes Made As A Result

Everything below landed in `ai-agent-coder` on 2026-07-29, before the runs in [Report 02](model-eval-02-20260731-merger-rate-plan-revised.md). This section is the causal link between the two reports.

| Change | Status | What landed |
|---|---|---|
| **Attempt cap 3 → 10** | ✅ done | Per-slice default raised in `pm_lib/slice_ops.py`, `pm_lib/cli.py`, run-state docs. README now frames >10 as *itself the finding* rather than a budget to raise again. |
| **`lint` skill** | ✅ done | New `skills/lint/` — a **differential** linter (*"does this change introduce a finding absent at `before_head`?"*), not an absolute gate, because `main` already carried 4 findings. 59 tests; self-lints clean. Tools: `ruff check`/`format`, `markdownlint-cli2`, `clang-format`, `cppcheck`, `codespell`, `git diff --check`. A missing linter records **unavailable coverage, never a pass**. |
| **Lint wired into both modes** | ✅ done | Mode A: `scoped-implementation` step 4 + `commit` step 3. Mode B: Developer prompt validation + PM assess. Run *before* commissioning any LLM review. |
| **"Read the output a human will read"** | ✅ done | One clause in PM `SKILL.md` step 3. Not mechanisable — the toolkit cannot know a repo's entry command. |
| **Close or re-carry inherited notes** | ✅ done | One clause in PM `SKILL.md` step 3, before accepting. |
| **Plan-defect cost test** | ✅ done, re-cut | Originally proposed as a ruling *counter*; re-cut as a **cost** test after checking it would have stopped `baseline-3`, the best-behaved run. A section that keeps costing attempts, or two reviews contradicting each other on it, is evidence of a defective plan. |
| **Domain Contract dtype ambiguity (P1)** | ✅ done | Split into **Axis 1 — form** (dtype/rank/shape/sign/finiteness/complex; always in scope, validated *before any coercion*) and **Axis 2 — magnitude** (unspecified outside declared ranges). Added: *"a guard that rejects in-domain input is a defect of equal severity to a missing guard."* **Plan sha256 changed.** |

**Deliberately declined**, with reasoning worth preserving:

- **Lint as a ninth mechanical floor fact** — the floor's eight facts are repository-integrity properties never legitimately violated; lint is a quality signal that belongs in recorded PM judgement *above* the floor, where a tolerance can be granted with a reason. Making it non-waivable would also let a linter release adding one rule hard-block an unrelated run.
- **A reviewer reachability rule in `references/reviewer-prompt.md`** — the rule already existed in `code-review/SKILL.md:92`, in a stronger form, and landed before runs 2–5. Verified in force: across run 5's 29 review reports the split was 4 P1 / 16 P2 / 1 P3 and every P1 named a concrete producer path.
- **A machine-readable manifest** — `run.json` already records harness, model, effort, plan sha256, reviewer and attempt cap, HMAC-signed. Replaced by four lines adding reviewer and attempt budget to the run-report header.
- **Three of four plan edits (P2–P4)** — left in deliberately. Writing "every printed row identifies its own redshift" into the plan would remove the only live test of whether the system catches a presentation defect on its own. Only P1 was fixed, because **only P1 was self-contradictory** and therefore corrupted grading rather than testing adaptability.
- **A cross-run plan-defect ledger** — rediscovering plan defects each run is accepted as realistic and is itself worth testing. Within-run curation via `notes.md` already prevents a single run re-litigating a settled ruling.

**Blocking issue found during implementation:** `ruff`, `markdownlint-cli2` and `codespell` were **not installed on the machine**. The lint skill failed closed correctly, but until the binaries existed every run would record a coverage gap and catch none of the 11 findings the skill was built for.

---

## 9. Risks & Unknowns

- **n=2 per Developer configuration.** The `baseline-2`/`baseline-3` spread (4 points) and the `opencode-1`/`opencode-2` spread (12 points) are both within plausible sampling noise.
- **Effort mismatch** means "frontier vs local" here is *frontier-at-high* vs local. A medium-effort baseline may narrow the gap.
- **The local runs were capped below their convergence point.** Both ran at max-attempts 5, and `opencode-2` used 4 of 5 on its last slice. **Its 88 is a lower bound, not a ceiling.**
- **Harness and model are confounded** for the local runs. Nothing separates OpenCode-the-harness from qwen-the-model. Four stalls ("tool call printed as literal text", "idling with uncommitted work") are unattributable.
- **Grading weights are my judgement**, though the underlying findings are mechanically reproducible. A reader weighting test breadth above code hygiene would rank `opencode-2` above `baseline-3`.
- **Not attempted:** token/cost accounting, reviewer scoring against a seeded-defect corpus, and any assessment of the plan's *science* beyond internal consistency.
