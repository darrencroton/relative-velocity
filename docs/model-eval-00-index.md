# Model Evaluation Series — Index

**Purpose:** a running comparison of AI model and harness performance in the `project-manager` Mode-B workflow, measured by executing the same class of implementation plan repeatedly and grading the result independently of what the run itself claimed.

**What is being compared:** not "which model is best" in the abstract, but **which model belongs in which seat** — Developer, PM, drift-audit Reviewer, code-review Reviewer — given that local models are free and unbounded in runtime while frontier models cost money and have limited runtimes.

---

## The reports

| # | Report | Date | Plan under test | Runs | Developer models tested |
|---|---|---|---|---|---|
| 01 | [Baseline plan, five runs](model-eval-01-20260729-merger-rate-plan.md) | 2026-07-29 | `MERGER_RATE_PLAN.md` (5-slice, and a superseded 3-slice) | 5 | claude-sonnet-5, qwen3.6-27b-bf16 |
| 02 | [Revised plan, eight runs](model-eval-02-20260731-merger-rate-plan-revised.md) | 2026-07-31 | `MERGER_RATE_PLAN-REVISED.md` (3-slice) | 8 | claude-sonnet-5, qwen3.6-27b-bf16, qwen3.6-27b-q8, qwen3.6-35b-a3b-bf16 |
| 03 | [Revised plan, five mixed-harness runs](model-eval-03-20260804-merger-rate-plan-revised.md) | 2026-08-04 | `MERGER_RATE_PLAN-REVISED.md` (3-slice, amended) | 5 | qwen3.6-27b-bf16, qwen3.5-397b-a17b-q6, qwen3-235b-a22b-q8 |

All three reports follow the same section structure so they can be read across. Section 5 (**Role Performance**) is the comparable core; section 8 (**Changes Made As A Result**) is what connects each report to the next.

**Total to date:** 18 supervised runs, 3 plan versions, 9 distinct Developer models, 12 distinct Reviewer seats.

---

## The bigger picture across all 18 runs

### 1. The science was never the discriminator

Across all three reports, **every completed run produced numerically identical results**. Report 01 measured five independent implementations agreeing to 2×10⁻¹⁶. Report 02 found all eight branches covering every pinned exact-value criterion. Report 03 found all four completed branches recovering the injected exponent to identical slopes in all six mass bins.

> Given a sufficiently prescriptive plan, model choice does not affect correctness of the mathematics. It affects **discipline** — validation ordering, dead code, scope churn, output legibility, and whether the tests can actually fail.

This has held across a 20× spread in model capability and three plan revisions. It is the most robust finding in the series.

### 2. The dominant defect class has been the same every time

| Report | Dominant defect class | Present in |
|---|---|---|
| 01 | Coerce-before-validate; dead locals; presentation defects | all local runs |
| 02 | Tests that pass under the regression they exist to catch | 6 of 8 |
| 03 | Tests that name a criterion but cannot fail | **every run, including the best one** |

Three independent evaluations, three plan versions, same answer. **This is now the highest-value target for a mechanical gate**, and it is the one thing no amount of model upgrade has fixed.

### 3. Where the seats have settled

| Seat | Verdict across the series | Confidence |
|---|---|---|
| **Developer** | Local models are viable given frozen numeric acceptance criteria. `qwen3.6-27b-bf16` and `qwen3.5-397b-a17b-q6` both reach frontier-adjacent quality at 3–5× wall-clock and ~3× supervision. | High — 13 local Developer runs |
| **PM** | **Do not economise.** Report 02 §5 showed an adjudication coin-flip decided whether a defect shipped; report 03 showed the PM was the only layer that executed code and read output. Every report independently reached this conclusion. | High |
| **drift-audit Reviewer** | Local models do this well — it is bounded reading against a frozen contract. Best value-per-dollar in the system in reports 02 and 03. | High |
| **code-review Reviewer** | **Frontier only, or frontier + one local corroborator.** In report 03, *every* material P1 across five runs came from a frontier seat; local reviewers returned PASS on trees containing P1s. | High |

### 4. Trends that reversed or sharpened

- **Report 01** ranked frontier clearly above local and recommended "linter gate first, attempt budget second, bigger model last."
- **Report 02** found the gap narrowed to 2–5 rubric points once numeric criteria were frozen — but also found the **mechanical floor and lint contributed ~0 discovery value**, partially retiring report 01's headline recommendation. The linter was worth building; it just does not find what actually escapes.
- **Report 03** found run-to-run variance *within* a single model comparable to variance *between* models, which weakens every single-run ranking in all three reports.

### 5. Parameter count has never predicted quality

| Model | Total params | Report | Rank |
|---|---|---|---|
| qwen3.6-35b-a3b-bf16 | 35B (~3B active) | 02 | **7th and 8th of 8** |
| qwen3.6-27b-bf16 | 27B dense | 02 | 3rd of 8 |
| qwen3.6-27b-q8 | 27B quantised | 02 | 5th and 6th of 8 |
| qwen3-235b-a22b-q8 | 235B (~22B active) | 03 | **failed outright** |
| qwen3.5-397b-a17b-q6 | 397B (~17B active) | 03 | **1st of 5** |

Active-parameter count tracks better than total, but neither is reliable. The 235B failure in report 03 was **agentic, not cognitive** — it got the physics bit-exact while entering a 21-iteration failed-edit loop and faking its own test coverage.

### 6. Quantisation has never shown a measurable quality cost

Report 02 tested q8 vs bf16 on the same 27B model: q8 was faster, needed fewer attempts, and scored within 4 points. Report 03's best result came from a q6 model. **No evidence in 18 runs that higher precision bought quality on this task class.**

---

## Open questions the series has not answered

| Question | Why it is still open | Which report proposed the test |
|---|---|---|
| Does the **PM seat** dominate outcome variance? | The PM was frontier in all 18 runs. Never varied. | 02 §7 |
| Is the local gap **process-closable**? | Report 01 predicted parity at ~93 with gates in place; never re-measured cleanly. | 01 §8 |
| Would a **frontier Developer** eliminate the vacuous-test class? | Never tested with a local reviewer panel. | 03 §7 |
| Is **n=2 per arm** enough? | Within-model spread exceeds between-model gaps in both 02 and 03. | 02, 03 |
| What is the **marginal value of reviewer seats 2 and 3**? | Reports 01 and 03 reach opposite conclusions on panel breadth. | 03 §7 |

---

## Reading notes

- **Scoring scales differ by report** (01 uses /100 weighted, 02 uses /35 weighted, 03 uses /30 flat). Each report states its rubric before applying it. A normalised percentage is given in each results table for cross-report reading; treat cross-report score comparisons as indicative only, since the rubrics weight differently and the plans differ.
- **Plan versions differ.** Report 01 tested `MERGER_RATE_PLAN.md`; reports 02 and 03 tested `MERGER_RATE_PLAN-REVISED.md`, and report 03's runs split across an unamended and an amended digest. Rankings are only valid within a report.
- **The originals** of all three reports, before normalisation to this format, are preserved in `archive/model-eval-originals/`.
