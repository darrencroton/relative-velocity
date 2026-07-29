# Implementation Plan: Close-Pair Merger Rate Estimation

## Purpose

Add `src/merger_rate.py`, converting the close-pair counts already produced by
`pair_finder.py` / `calc.py` into a **merger rate density** via the standard
observational close-pair method (Kitzbichler & White 2008; see
`docs/BACKGROUND.md`). The module computes pair fractions per stellar-mass bin
and redshift, converts them to merger rates through a parameterized merger
timescale, propagates Poisson uncertainty, and produces a redshift-evolution
figure plus a results table.

Executed under `project-manager` Mode B. Five slices, run atomically in plan
order; each slice's committed output is the next slice's input.

## Scientific Background

See `docs/BACKGROUND.md` §1 and its Kitzbichler & White 2008 / Jiang et al. 2014
/ Lagos et al. 2021 references.

**1. Close pairs** are already defined by the pipeline: within
`config["max_sep"]` kpc 3D separation and mass ratio `>= config["mass_ratio_min"]`.
Every row in `results/pairs_z{z}.hdf5` satisfies this — no new pair-finding is
needed. `N_pairs(b, z)` counts stored rows with `mass_bin == b`; each unordered
pair is counted once (matching `cKDTree.query_pairs` de-duplication).

**2. Pair fraction** per mass bin `b` and redshift `z`:

```
f_pair(b, z) = N_pairs(b, z) / N_gal(b, z)
```

`N_gal(b, z)` is the **total** number of mass-selected galaxies in that bin —
paired *and* unpaired. This denominator does not exist in the pipeline's output
today and must be added (Slice 1).

`f_pair` is a **pairs-per-galaxy incidence ratio, not a probability bounded by
1** — one galaxy can appear in several stored pairs, so `f_pair > 1` is legal in
a crowded bin. This convention is what makes the algebra in point 4 reduce
exactly to `f_pair * n_gal = N_pairs / V`, independent of `N_gal`.

This module requires `config["mass_bin_by"] == "primary"`. `"mean"` and
`"total"` bin on a joint quantity of both galaxies' masses, which is not a
property of any single galaxy, so no single-galaxy `N_gal(b, z)` is definable
for them at all. `"secondary"` bins on the less-massive member's own mass and
*could* support an analogous denominator — excluding it is a **scope decision,
not a correctness requirement**. `run_merger_rate_calculation` must assert
`"primary"` and raise a clear error naming any other value, including
`"secondary"`.

**3. Merger timescale** — the average time a pair remains observably close
before merging — as a power law in redshift:

```
T_merge(z) [Gyr] = merger_timescale_gyr0 * (1 + z) ** merger_timescale_alpha
```

Both are new `config.py` parameters (Slice 2). This is a deliberately simplified
model, not a fit to any simulation suite; the plan claims architectural and
dimensional correctness only. **This redshift dependence is an injected model
input, not something measured from the data** — point 6 depends on that fact.

**4. Merger rate density** per mass bin and redshift, in `Gyr^-1 Mpc^-3`:

```
R(b, z)     = merger_fraction * f_pair(b, z) * n_gal(b, z) / T_merge(z)
n_gal(b, z) = N_gal(b, z) / box_size_mpc(z)**3
```

`merger_fraction` (new config parameter, `C_merge` in the literature) is the
fraction of close pairs that actually merge within `T_merge` rather than being
chance projections or unbound flybys.

Because `f_pair * n_gal = N_pairs / box_size_mpc**3` exactly, this is
algebraically identical to:

```
R(b, z) = merger_fraction * N_pairs(b, z) / (box_size_mpc(z)**3 * T_merge(z))
```

Compute it via the `f_pair`/`n_gal` route — that matches how the literature
presents the method, and `f_pair`, `N_pairs`, and `N_gal` are reported as
diagnostics in their own right. The reduced form is the key fact behind point 6.

`box_size_mpc(z)` is the per-redshift box size `pair_finder.find_pairs` actually
used for that snapshot (`catalog["box_size"]`) — **not** `config["box_size"]`.
These are equal for the current mock data, but nothing enforces that, and a real
catalog could differ per snapshot. Slice 1 persists the value actually used so
Slice 2 reads the authoritative one.

**5. Poisson uncertainty.** `N_pairs` is treated as Poisson-distributed;
`N_gal` as an exact count (it is the full mass-selected sample, not a subsample).
`T_merge` and `merger_fraction` are fixed model inputs with no propagated
uncertainty — an explicit scope boundary.

```
sigma_f_pair(b, z) = f_pair(b, z) / sqrt(N_pairs(b, z))   if N_pairs > 0, else 0
sigma_R(b, z)      = R(b, z) / sqrt(N_pairs(b, z))        if N_pairs > 0, else 0
```

The `N_pairs == 0` case is **frozen**: exactly zero, not `NaN`, not a
divide-by-zero. This is a **plug-in point-estimate simplification for downstream
plotting and fitting, not a rigorous Poisson confidence interval** — a rigorous
treatment would give a nonzero upper limit even at zero counts (e.g. Gehrels).
Small-`N` Poisson intervals are out of scope; code and docstrings must describe
these as *this plan's convention*, never as "the Poisson uncertainty" unqualified.

**6. The frozen redshift-evolution validation logic.** `generate_test_data.py`
draws a fixed `N_PAIRS = 3000` and `N_FIELD = 2000` at every redshift from the
same distributions, reseeded per redshift — so `N_pairs(b, z)` and `N_gal(b, z)`
are statistically flat across the four mock redshifts.

**This does NOT mean `R(b, z)` should come out flat.** By the reduced formula,
`R ∝ N_pairs / T_merge` with `T_merge = T0 * (1+z)^alpha`, so:

```
R(b, z) ∝ (1 + z) ** (-alpha)
```

`log10(R)` against `log10(1+z)` is therefore expected to be a straight line of
slope **`-merger_timescale_alpha`** (i.e. `+1.0` at the default `alpha = -1.0`),
**not slope zero**. Asserting flatness would be wrong and would correctly fail.

The validation is: fit `log10(R)` vs `log10(1+z)` per mass bin by weighted least
squares (weights from `sigma_R` propagated to log space), and confirm the fitted
slope is statistically consistent with the *config-derived*
`-merger_timescale_alpha` — not with a literature exponent, and not with zero.
This is meaningful because the only source of the slope is the injected
`T_merge(z)`; the pair-count input has no systematic trend by construction. It
verifies the conversion and its error propagation without fabricating a claim
that the mock data recovers real merger-rate evolution.

**This is a mock-data diagnostic, not a production check, and must be labeled as
such everywhere it appears** (printed table, docstrings). The derivation depends
on `N_pairs / box_size_mpc**3` being flat across redshift — a property of *this
mock catalog's construction*, not something guaranteed for future data. The
pipeline has exactly one data source today (real SAGE support is future work per
`docs/PLAN.md`), so this is not a live risk, but the check must not be presented
as remaining valid once a real reader exists: genuine merger-rate evolution or a
redshift-varying box size would legitimately give a different slope, and that
must not be reported as an implementation defect. Stating the limitation clearly
is sufficient; build no runtime mock-vs-real detection.

## Architecture Fit

- New file `src/merger_rate.py`, structured like `plot.py`: reads already-written
  results files, writes new outputs; does not re-run pair finding.
- Per existing repo convention (`_mass_bin_edges` is already duplicated
  independently in `pair_finder.py` and `plot.py`), `merger_rate.py` and the
  addition to `calc.py` each define their own local `_mass_bin_edges(config)`
  rather than importing a private helper. `docs/PLAN.md`'s DRY principle would
  suggest one shared helper; that tension is pre-existing and not resolved here.
- All tunable scientific parameters live in `config.py` and are passed
  explicitly. Frozen filenames, plotting colors, the display-only log-errorbar
  floor factor, and `check_slope_consistency(..., n_sigma=3.0)` may be local
  constants — presentation/API choices, not hidden scientific inputs.
- Fail loud: `n_pairs[b] > 0` with `n_galaxies_per_mass_bin[b] == 0` is
  inconsistent (a pair cannot exist in a bin with zero galaxies) and raises. Both
  zero is valid (an empty bin) and yields `f_pair = 0, sigma_f_pair = 0`.
- Units follow the existing convention: positions/box size in Mpc, no new
  conversions beyond what `pair_finder.py` already does. New units (Gyr,
  `Mpc^-3`, `Gyr^-1 Mpc^-3`) are documented inline. The pipeline does not
  establish whether coordinates are comoving or proper (`docs/PLAN.md` takes all
  values at face value); outputs are labeled `Gyr^-1 Mpc^-3` without an
  unqualified "comoving" claim.

## Numerical Domain Contract

**Binding on implementers and reviewers alike.**

Every function in this plan validates its inputs and fails loud with clear
`AssertionError`s, per the repo's house style. The obligation has two axes, kept
separate here deliberately: conflating them is what let two reviewers read this
section to opposite conclusions about the same code.

**Axis 1 — form. Always in scope, and validated before any coercion.** dtype
(including complex and string), rank, shape, sign where a sign is required, and
finiteness where finiteness is required. A malformed input must raise
`AssertionError` *before* the value is coerced: `np.asarray(x, dtype=float)`
placed ahead of the dtype check silently discards an imaginary part or parses a
numeric-looking string, and that is a defect of the same severity as no check at
all. "Numeric" here means a real integer or floating dtype — booleans and complex
are not numeric, and a function that accepts float64 but rejects an
integer-valued input is rejecting valid data.

**Axis 2 — magnitude.** Behaviour is specified only inside the declared ranges
below; outside them it is unspecified. This exemption covers magnitude only and
never excuses a missing Axis 1 check.

| Quantity | Declared domain |
|---|---|
| `box_size_mpc` | finite, `1e-3` to `1e6` Mpc |
| `timescale_gyr`, `merger_timescale_gyr0` | finite, `1e-6` to `1e6` Gyr |
| `merger_fraction` | finite, `0 < x <= 1` |
| `merger_timescale_alpha` | finite, `abs(alpha) <= 100` |
| `z`, `redshifts` | finite, `> -1` |
| `n_pairs`, `n_galaxies` | finite, non-negative, integer-valued |
| `f_pair`, `sigma_f_pair` | finite, non-negative |
| `rate`, `rate_err` | finite; usability additionally requires `> 0` |
| `rate_err / rate` | bounded below by `1 / sqrt(N_pairs)` by construction |

Implementers need not guard overflow or underflow of intermediate products for
out-of-domain **magnitudes**, and reviewers must not report out-of-domain
magnitude behaviour as P0 or P1. Functions may still reject such magnitudes, and
where a guard is cheap and clear it is welcome — but its absence is not a defect.

**A guard that rejects in-domain input is a defect of equal severity to a missing
guard**, and every rejection message must name the actual reason for the
rejection. Rejecting an integer where the domain says "finite positive scalar",
or reporting "must be numeric" for a value that *is* numeric, are both defects.

**The one in-domain exception, which must hold:** for any in-domain input, a bin
with `sigma_f_pair == 0` must yield `sigma_rate == 0` exactly (point 5).

## Implementation Profile and Execution Order

- Execute Slices 1-5 atomically in order; do not combine commits or reviews
  across slice boundaries.
- Every slice has `Independent audit required: yes`. After each commit the PM
  commissions `drift-audit` then `code-review` sequentially against that exact
  final commit, and reruns the slice's validation independently.
- Each slice may assume all earlier accepted slices. **No slice may silently
  repair an earlier one:** if a prerequisite defect is found, stop and return it
  for an explicit plan decision rather than broadening the current slice.
- Each slice's **Definition of Done** is a checklist. The implementer must
  reproduce it in `validation.md` with each item marked and evidence cited. An
  unticked or unevidenced item is an incomplete slice.

## Test Isolation and Shared Fixtures

- Tests in `tests/test_merger_rate.py` use a copied config whose `data_dir`,
  `results_dir`, and `figures_dir` point beneath pytest's
  `tmp_path`/`tmp_path_factory`. Tests must never read, overwrite, or depend on
  the repository's gitignored `data/`, `results/`, or `figures/`.
- The generated-mock integration fixture calls `generate_all_snapshots(config)`
  and `run_calculation(config)` directly with those temporary directories. It may
  be module-scoped so the four-snapshot setup is shared. "Generated mock data"
  below always means this isolated fixture, never shelling out to the CLI.
- CLI behaviour is validated separately: unit tests patch `sys.argv` / module
  call sites; the end-to-end command in Slice 5 is a post-test validation run.

---

## Slice 1: Galaxy-count denominator + pair-fraction calculation

### Intended Change

Extend `calc.py` to compute, per redshift, the total number of mass-selected
galaxies per mass bin, and write it as a new dataset `n_galaxies_per_mass_bin`
(1D int array, length `n_mass_bins`) in each `results/pairs_z{z}.hdf5` written by
`_save_pairs`. Add two attrs: `box_size_mpc` (float, set to `catalog["box_size"]`
— the value `find_pairs` actually used) and `mass_bin_edges` (1D float array from
the local `_mass_bin_edges(config)`). These let Slice 2 verify the live config
still matches the persisted artifact. **All three additions are purely additive**;
existing datasets (`mass_primary`, `mass_secondary`, `mass_ratio`,
`separation_kpc`, `delta_v`, `mass_bin`, `sep_bin`) and attrs are unchanged.

The count must come from the **full mass-selected catalog** (`load_galaxy_catalog()`
output, already filtered to `[log_mass_min, log_mass_max]`), not just galaxies in
pairs. `run_calculation()` already loads it per redshift — reuse it. Extract the
counting into a directly testable helper
`_count_galaxies_per_mass_bin(log_stellar_mass, config)` returning a 1D int array
of length `n_mass_bins`.

Add a local `calc._mass_bin_edges(config)` with the same formula as the existing
private helpers; both `_count_galaxies_per_mass_bin` and `_save_pairs` use it.

Change `_save_pairs`'s signature to
`_save_pairs(pairs, n_galaxies_per_mass_bin, box_size_mpc, filepath, z, config)`.
`run_calculation` passes the count array and the same `catalog["box_size"]` given
to `find_pairs`. `_save_pairs` writes those values and recomputes neither from the
pair catalog nor from `config["box_size"]`.

**Bin-edge convention must match `pair_finder._assign_mass_bins` exactly**: the
same `np.digitize`-based right-open logic (reimplemented locally per the
duplication convention), so a galaxy with `log_stellar_mass` exactly equal to
`log_mass_max` falls outside all bins and is excluded from every count. This holds
even though `data_reader`'s selection mask is inclusive of `log_mass_max` — a
pre-existing inconsistency this plan does not fix. What matters is that the new
denominator uses the *identical* rule the existing numerator uses.

**This slice introduces no `mass_bin_by` assertion at all.** The `"primary"`-only
restriction applies to Slice 2's entry point, which does not exist yet.
`_count_galaxies_per_mass_bin` and `compute_pair_fraction` are strategy-agnostic
and must remain independently testable without any `mass_bin_by` dependency.

Create `src/merger_rate.py` with:

- `_mass_bin_edges(config)` — local copy of the standard formula.
- `_results_path(z, config)` — `results/pairs_z{z:.1f}.hdf5`, matching
  `calc.py`/`plot.py`.
- `_load_pair_counts(z, config)` — reads one results file, returns
  `(n_pairs_per_bin, n_galaxies_per_bin, box_size_mpc)`. `n_pairs_per_bin` counts
  `mass_bin == b` per bin; `n_galaxies_per_bin` is read from the new dataset;
  `box_size_mpc` from the new attr. Both count arrays are 1D int, length
  `n_mass_bins`, in bin-index order. Fail loudly if: the file is missing; any
  required dataset/attr is absent; `mass_bin` is not a 1D integer-valued array;
  `n_galaxies_per_mass_bin` is not a 1D non-negative integer-valued array of
  exactly `n_mass_bins` entries; or `box_size_mpc` is not a finite positive
  scalar. `mass_bin == -1` is the existing out-of-range sentinel and is not
  counted; any other index outside `[-1, n_mass_bins - 1]` is malformed and raises.
- `compute_pair_fraction(n_pairs, n_galaxies)` — vectorized over the mass-bin
  array, returns `(f_pair, sigma_f_pair)` per points 2 and 5. Both inputs must be
  1D, identically shaped, and contain only finite non-negative integer-valued
  counts; malformed input raises rather than broadcasting, truncating, or
  producing `NaN`. Assert `n_galaxies[b] > 0` for every bin with `n_pairs[b] > 0`.
  Both zero is valid and yields `0, 0`. Returns float arrays of the input shape.

### Definition of Done

- [ ] `n_galaxies_per_mass_bin`, `box_size_mpc`, `mass_bin_edges` present in every
      regenerated results file; every pre-existing dataset and attr unchanged in
      meaning and value.
- [ ] `box_size_mpc` provably comes from `catalog["box_size"]`, not
      `config["box_size"]` — proven by a test where the two differ.
- [ ] The denominator is the full selected catalog: its sum equals the selected
      galaxy count, not the number of galaxies appearing in pairs.
- [ ] A galaxy at exactly `log_mass_max` is excluded from every bin count.
- [ ] `_load_pair_counts` excludes `mass_bin == -1` and raises on any other
      out-of-range index.
- [ ] No `mass_bin_by` read or assertion anywhere in this slice.
- [ ] `venv/bin/python -m pytest tests/` passes, 0 failed.
- [ ] `venv/bin/python src/pipeline.py --validate` exits 0 with unchanged figures.

### Acceptance Criteria

- **Inputs:** `config` (existing keys only; this slice is `mass_bin_by`-agnostic);
  on-disk `results/pairs_z{z}.hdf5` as written by the modified `calc.py`.
- **Outputs:** results files with the new dataset and two attrs; `merger_rate.py`
  exposing `_mass_bin_edges`, `_results_path`, `_load_pair_counts`,
  `compute_pair_fraction`; `calc.py` exposing `_mass_bin_edges` and
  `_count_galaxies_per_mass_bin` as importable, directly testable functions.
- **User-visible:** `--calc-only` / `--validate` / `--generate-test` produce
  results files with one extra dataset and two extra attrs. No change to console
  output, plots, or existing data.
- **Must not change:** `plot.py`'s figures and stats table; all existing tests in
  `tests/test_geometric.py`, `tests/test_pair_finder.py`,
  `tests/test_statistical.py` pass unmodified.

### Authorized Surface

- Files allowed to change:
  - `src/calc.py`
  - `src/merger_rate.py` (new file)
  - `tests/test_merger_rate.py` (new file)
- `calc.py`: `_save_pairs` (frozen expanded signature + three additive writes),
  new `_mass_bin_edges`, new `_count_galaxies_per_mass_bin`, `run_calculation`
  (call the helper, pass its output through).
- `merger_rate.py`: the four functions listed above.
- Tests: `tests/test_merger_rate.py` only. **No existing test file should need
  changes; if one does, stop and report rather than editing it.**

### Explicit Non-Goals

- No merger timescale, rate conversion, uncertainty-on-rate, plotting, or CLI.
- No changes to `pair_finder.py`, `data_reader.py`, `plot.py`,
  `generate_test_data.py`, or `config.py`.
- No change to the meaning or value of any existing dataset or attr.
- No `mass_bin_by` logic of any kind.
- No attempt to reconcile `data_reader.py`'s inclusive-upper-edge selection with
  `pair_finder.py`'s exclusive-upper-edge binning.

### Risk Flags

- Risky surfaces: on-disk data schema (`results/pairs_z{z}.hdf5`) — additive only;
  confirmed no test in `tests/` opens these files directly.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

New file `tests/test_merger_rate.py`, in the hand-crafted-exact-recovery style of
`tests/test_pair_finder.py`:

- `compute_pair_fraction` on hand-crafted arrays (e.g. `n_pairs=[0, 5, 20]`,
  `n_galaxies=[10, 10, 10]`) recovers exact hand-computed `f_pair` and
  `sigma_f_pair`, including the `n_pairs == 0` case (`0`, `0`, no `NaN`/`inf`).
- Raises when `n_galaxies[b] == 0` and `n_pairs[b] > 0` (construct directly).
- Rejects mismatched shapes, non-1D input, negative counts, non-finite counts,
  and non-integer-valued counts, rather than broadcasting or coercing.
- `_count_galaxies_per_mass_bin` direct unit test on a small synthetic
  `log_stellar_mass` array including one value exactly at `config["log_mass_max"]`
  (excluded) and one exactly on an interior edge; assert exact per-bin counts.
- `_load_pair_counts` against a small hand-written HDF5 fixture (or `calc.py` run
  on a tiny synthetic catalog) recovers known `n_pairs_per_bin`,
  `n_galaxies_per_mass_bin`, and `box_size_mpc`. The fixture also carries the
  `mass_bin_edges` and existing `mass_bin_by` provenance attrs Slice 2 validates.
- `_load_pair_counts` rejects a wrong-length `n_galaxies_per_mass_bin`, a
  `mass_bin` outside `[-1, n_mass_bins - 1]`, and a non-finite/non-positive
  `box_size_mpc`. Include `mass_bin == -1` in the valid fixture and prove it is
  excluded from `n_pairs_per_bin`.
- Real-run coverage proving `box_size_mpc` persists `catalog["box_size"]` when it
  differs from `config["box_size"]`.

**Commands:** `venv/bin/python -m pytest tests/` (0 failed);
`venv/bin/python src/pipeline.py --validate` (exit 0, unchanged figures).

**Manual:** open a regenerated `results/pairs_z2.0.hdf5` with `h5py`; confirm
`n_galaxies_per_mass_bin` has length `n_mass_bins` with non-negative integers, and
that `box_size_mpc` / `mass_bin_edges` match the catalog and config.

### Rollback Path

Revert `src/calc.py`'s new writes; move `src/merger_rate.py` and
`tests/test_merger_rate.py` to `archive/merger-rate-slice1/` rather than deleting
them. Rollback is a separately approved change whose authorized surface also
includes those archive paths and `.gitignore` (add `archive/` if absent, per the
global archive convention). Do not rewrite history or use a revert that
transiently deletes the new files.

---

## Slice 2: Merger timescale parameterization + rate conversion

### Intended Change

Add to `config.py` under a `# Merger rate` section:

- `merger_timescale_gyr0 = 2.2` — `T_merge` normalization at `z = 0`, Gyr
  (order-of-magnitude consistent with Kitzbichler & White 2008; not asserted as
  an exact reproduction).
- `merger_timescale_alpha = -1.0` — power-law index (point 3). Slice 3's
  validation uses `expected_slope = -merger_timescale_alpha`; that dependency is
  config-driven, so changing this value needs no code change — just be aware.
- `merger_fraction = 0.6` — fraction of close pairs that merge within `T_merge`.

Add to `src/merger_rate.py`:

- `_merger_rate_results_path(config)` —
  `os.path.join(config["results_dir"], "merger_rate.hdf5")`. Slice 2 writes and
  Slice 4 reads through this one helper.
- `merger_timescale_gyr(z, config)` — implements point 3. Asserts per call, not
  just for the configured redshift list: `merger_timescale_gyr0` finite and `> 0`;
  `merger_timescale_alpha` finite; `z` finite and `> -1`; and the resulting
  `T_merge(z)` finite and `> 0`. All inputs must be numeric scalars — reject
  strings and arrays before any domain check. Independently testable; validates
  its own inputs rather than trusting a caller.
- `compute_merger_rate(f_pair, sigma_f_pair, n_galaxies, box_size_mpc, timescale_gyr, merger_fraction)`
  — vectorized over the mass-bin array for one redshift; implements points 4 and 5
  exactly; returns `(rate, sigma_rate)` in `Gyr^-1 Mpc^-3` as float arrays of the
  input shape. `f_pair`, `sigma_f_pair`, `n_galaxies` must be 1D and identically
  shaped, all finite and non-negative, with `n_galaxies` integer-valued. A bin
  with `n_galaxies == 0` must also have `f_pair == sigma_f_pair == 0`; any other
  combination raises. Compute
  `sigma_rate = merger_fraction * sigma_f_pair * n_galaxies / (box_size_mpc**3 * timescale_gyr)`
  — this equals `rate / sqrt(N_pairs)` for non-empty bins without needing
  `N_pairs` in the signature, and preserves exact-zero uncertainty when
  `sigma_f_pair == 0`. Assert `merger_fraction`, `box_size_mpc`, and
  `timescale_gyr` are finite scalars within the Numerical Domain Contract ranges.
  Documented as independently callable; assumes no caller-side validation.
- `run_merger_rate_calculation(config)` — asserts
  `config["mass_bin_by"] == "primary"` first, raising a clear error naming the
  actual value.

  Before loading any snapshot or opening the output for writing, **preflight**
  every `_results_path(z, config)` and raise a clear missing-file assertion naming
  the first absent path. Then, as a second pre-write gate, open every pair-results
  file and require: its recorded `redshift` equals the configured `z` for that
  path; its `mass_bin_by` equals both `"primary"` and the live config value; its
  `mass_ratio_min` and `max_sep_kpc` equal the live config values; and its
  `mass_bin_edges` matches `_mass_bin_edges(config)` in shape and value. Validate
  dtype and scalar/array shape before coercion — a malformed string or vector attr
  must raise, not crash on comparison. Numeric provenance values must be finite.
  Any mismatch raises a clear assertion naming the file and both values.
  **A failure at either gate must leave any pre-existing `merger_rate.hdf5`
  byte-for-byte untouched.**

  Then for each `z` in `config["redshifts"]`: `_load_pair_counts`,
  `compute_pair_fraction`, `merger_timescale_gyr`, `compute_merger_rate` (passing
  the per-redshift `box_size_mpc` read from that file, **not**
  `config["box_size"]`). Assemble arrays over `(redshift, mass_bin)` and write
  `results/merger_rate.hdf5` containing datasets shaped
  `(n_redshifts, n_mass_bins)`: `pair_fraction`, `pair_fraction_err`, `n_pairs`,
  `n_galaxies`, `merger_rate`, `merger_rate_err`; plus 1D per-redshift datasets
  `merger_timescale_gyr` and `box_size_mpc` (persisted so the combined file is
  independently auditable); plus attrs `redshifts`, `mass_bin_edges`,
  `mass_bin_by`, `mass_ratio_min`, `max_sep_kpc`, `merger_fraction`,
  `merger_timescale_gyr0`, `merger_timescale_alpha`, `timestamp` — mirroring
  `_save_pairs`'s provenance pattern. Use `_merger_rate_results_path(config)` and
  create `config["results_dir"]` if needed.

### Definition of Done

- [ ] The three config keys added; no existing key changed in value or meaning.
- [ ] `merger_timescale_gyr(0, config) == merger_timescale_gyr0` exactly.
- [ ] `compute_merger_rate` reproduces hand-computed `rate` and `sigma_rate`.
- [ ] `sigma_f_pair == 0` yields `sigma_rate == 0` exactly (the in-domain
      exception in the Numerical Domain Contract).
- [ ] Per-file `box_size_mpc` is provably used, not `config["box_size"]` — proven
      by a fixture where they differ and the rate scales as `box_size_mpc**-3`.
- [ ] `mass_bin_by != "primary"` raises a clear error naming the value.
- [ ] Both pre-write gates leave a sentinel `merger_rate.hdf5` byte-for-byte
      unchanged on failure.
- [ ] `results/merger_rate.hdf5` contains every dataset and attr in the frozen
      schema, with the correct shapes.
- [ ] `venv/bin/python -m pytest tests/` passes, 0 failed.

### Acceptance Criteria

- **Inputs:** `config` with the three new keys and `mass_bin_by == "primary"`;
  on-disk `results/pairs_z{z}.hdf5` (with Slice 1's additions) for every
  configured redshift.
- **Outputs:** `results/merger_rate.hdf5` with the schema above;
  `merger_timescale_gyr` and `compute_merger_rate` importable and independently
  callable with no file I/O.
- **User-visible:** none yet — no CLI flag reaches this until Slice 5.
- **Must not change:** everything validated in Slice 1.

### Authorized Surface

- Files allowed to change:
  - `src/config.py`
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- `config.py`: add the three keys only.
- `merger_rate.py`: `_merger_rate_results_path`, `merger_timescale_gyr`,
  `compute_merger_rate`, `run_merger_rate_calculation`. **Slice 1's functions may
  be called but not modified**; if a genuine defect is found, report it rather
  than patching it in this diff.
- Tests: append only; do not remove or weaken Slice 1's tests.

### Explicit Non-Goals

- No plotting, CLI, or redshift-evolution fit — Slices 3-5.
- No uncertainty on `T_merge` or `merger_fraction` (point 5).
- No mass- or redshift-dependent `merger_fraction` — one global config value.
- No `mass_bin_by` strategy other than `"primary"`.
- Documentation of the new config keys is deliberately deferred to Slice 5; its
  absence from `README.md`/`AGENTS.md` here is not a defect.

### Risk Flags

- Risky surfaces: introduces a new persistent output file with its own frozen
  schema — a new persistence contract.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

Append to `tests/test_merger_rate.py`:

- `merger_timescale_gyr` at `z = 0` returns exactly `merger_timescale_gyr0`; at
  `z = 3` returns `merger_timescale_gyr0 * 4 ** merger_timescale_alpha` to
  floating-point precision — using hand-picked config values **distinct from the
  defaults**, so the test cannot pass by coincidence.
- `merger_timescale_gyr` raises on `z <= -1`, `z = nan`, `z = inf`,
  `merger_timescale_gyr0 <= 0`, non-finite `merger_timescale_gyr0`, non-finite
  `merger_timescale_alpha`, and on string/array inputs for all three arguments.
- `compute_merger_rate` on hand-crafted one-element arrays recovers exact `rate`
  and `sigma_rate`.
- `compute_merger_rate` raises on out-of-contract `merger_fraction` (`<= 0`,
  `> 1`, non-finite), `box_size_mpc` (`<= 0`, non-finite), and `timescale_gyr`
  (`<= 0`, non-finite).
- `compute_merger_rate` rejects mismatched/non-1D arrays, negative or non-finite
  values, non-integer `n_galaxies`, and `n_galaxies == 0` with nonzero `f_pair`
  or `sigma_f_pair`.
- `run_merger_rate_calculation` raises when `mass_bin_by != "primary"`.
- **Provenance gate:** parameterized fixtures with a mismatched recorded
  `redshift`, `mass_bin_by`, `mass_ratio_min`, `max_sep_kpc`, or `mass_bin_edges`
  (same bin count, different values), plus malformed string/vector attrs. Each
  raises clearly and leaves a sentinel output byte-for-byte unchanged.
- **Missing-file preflight:** create all but one configured pair file plus a
  sentinel `merger_rate.hdf5`; assert the raise happens before any read/write and
  the sentinel is unchanged.
- **Box-size provenance:** a fixture whose `box_size_mpc` attr deliberately
  differs from `config["box_size"]` (e.g. `250.0` vs the default `500.0`), with a
  config copy whose `redshifts` covers only the fixtured snapshots. Assert the
  resulting `rate` matches the hand-computed value using the *fixture's*
  `box_size_mpc`. An implementation reading `config["box_size"]` must fail this.
- **End-to-end** on the isolated generated-mock fixture: `merger_rate.hdf5`
  exists, dataset shapes are `(len(redshifts), n_mass_bins)`, every frozen
  dataset/attr is present, `n_pairs`/`n_galaxies` are integers, redshift and
  mass-bin ordering and pair-selection provenance are preserved, and
  `merger_rate`/`merger_rate_err` are finite and non-negative.

**Commands:** `venv/bin/python -m pytest tests/` (0 failed).

**Manual:** inspect `results/merger_rate.hdf5` attrs with `h5py`; confirm
`merger_fraction`, `merger_timescale_gyr0`, `merger_timescale_alpha`,
`mass_bin_by`, `mass_ratio_min`, `max_sep_kpc` match `config.py`.

### Rollback Path

Revert the three config keys and the Slice 2 additions to `src/merger_rate.py` /
`tests/test_merger_rate.py`. Slice 1 is unaffected and remains functional.

---

## Slice 3: Weighted redshift-evolution fit

Pure functions only. No I/O, no plotting, no CLI, no Matplotlib.

### Intended Change

Add to `src/merger_rate.py`:

- `fit_log_rate_vs_redshift(rates, rate_errs, redshifts)` — operates on a
  **single mass bin's** 1D array-likes (one value per configured redshift, all
  three the same shape). Convert to NumPy arrays and raise unless every input is
  1D with identical shapes — do not flatten, broadcast, or accept higher rank.
  Fits `log10(rate)` vs `log10(1 + z)` by weighted least squares, weights
  `1 / sigma_log_rate**2`, with `sigma_log_rate = sigma_rate / (rate * ln(10))`.

  **Malformed vs. data conditions — the distinction is frozen:**

  - Any `redshift` that is `<= -1` or non-finite is **malformed** (`log10(1+z)`
    undefined or infinite) and **raises immediately**.
  - A point is **usable** only if `rate` is finite and `> 0` *and* `rate_err` is
    finite and `> 0`. Points failing either are **excluded, not an error** — the
    normal empty-bin case from point 5 — and counted in `n_excluded`.
  - After filtering, if fewer than 2 usable points remain, **or** the usable
    points do not span at least 2 distinct redshifts (rank-deficient), **return
    `(nan, nan, nan, n_excluded)`** rather than raising or fabricating a fit. This
    is a data condition, not malformed input, so it must not raise. Either way the
    bin is reported downstream as "insufficient data", never silently omitted.

  `slope_err` must be the **unscaled** weighted-least-squares parameter error
  computed from the supplied `sigma_log_rate` as true measurement errors. Do not
  use residual-based rescaling (`scipy.optimize.curve_fit` without
  `absolute_sigma=True`, or `numpy.polyfit(..., cov=True)`'s default), which
  silently substitutes a data-driven estimate for the one this plan propagated.
  Use explicit weighted normal equations or `cov="unscaled"`. **The
  exactly-two-usable-points case must return a finite fit** — `numpy.polyfit(...,
  cov=True)` is specifically unsuitable there, as it attempts residual scaling
  with zero residual degrees of freedom.

  Implement the normal equations in a numerically stable form **for the declared
  domain**: normalize the weights and center the predictor about its weighted mean
  before accumulating, then restore the absolute covariance scale. Returns
  `(slope, slope_err, intercept, n_excluded)`. **This function has no knowledge of
  mass bins**; the caller loops over them.

- `check_slope_consistency(slope, slope_err, expected_slope, n_sigma=3.0)` —
  returns `True` if `abs(slope - expected_slope) < n_sigma * slope_err`, else
  `False`. Returns `False` (not an error) if `slope`, `slope_err`, or
  `expected_slope` is non-finite, or if `slope_err <= 0`; the insufficient-data
  `nan` case is "check not applicable" and the caller must report it as such,
  never as a pass or fail. Assert `n_sigma` is finite and `> 0` so an invalid
  threshold cannot fabricate a pass. `expected_slope` is always
  `-config["merger_timescale_alpha"]` in use, but the function takes it as a
  parameter and hardcodes nothing. Pure, no I/O.

### Definition of Done

- [ ] An exact power law is recovered to small numerical tolerance.
- [ ] The fit is **provably weighted** — demonstrated by a heteroscedastic case,
      not merely by fitting a straight line.
- [ ] `slope_err` matches an **independently hand-computed** normal-equations
      variance, not a value from the same library call the implementation uses.
- [ ] Exactly two usable points at distinct redshifts return a **finite**
      unscaled-covariance fit.
- [ ] `< 2` usable points, and `>= 2` usable points at a single redshift, both
      return `(nan, nan, nan, n_excluded)` **without raising**.
- [ ] Malformed `redshifts` (`<= -1`, `nan`, `inf`) and shape/rank violations
      **raise**.
- [ ] `n_excluded` is correct in every case above.
- [ ] `check_slope_consistency` handles the true / false / non-finite /
      non-positive cases per contract, and raises on invalid `n_sigma`.
- [ ] `venv/bin/python -m pytest tests/` passes, 0 failed.

### Acceptance Criteria

- **Inputs:** in-memory 1D array-likes only. No file I/O.
- **Outputs:** `fit_log_rate_vs_redshift` and `check_slope_consistency`
  importable and independently callable.
- **User-visible:** none — nothing calls these until Slice 4.
- **Must not change:** everything validated in Slices 1-2.

### Authorized Surface

- Files allowed to change:
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- `merger_rate.py`: the two new functions only. Earlier slices' functions may be
  called but not modified.
- Tests: append only.

### Explicit Non-Goals

- No plotting, table, CLI, orchestration, or Matplotlib import in this slice.
- No hardcoded literature exponent anywhere as ground truth — `expected_slope` is
  always derived from `config["merger_timescale_alpha"]`.
- No guarding of out-of-domain magnitudes beyond the Numerical Domain Contract.
  (Form validation — dtype, rank, shape, sign, finiteness — is always required;
  see Axis 1.)

### Risk Flags

- Risky surfaces: none — pure functions, no persistence or CLI.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

Append to `tests/test_merger_rate.py`:

- Hand-crafted exact power law `rate = A * (1+z)**m` with tiny *equal* errors
  recovers `slope ≈ m`. (Equal errors are weighting-invariant, so this alone
  cannot prove the fit is weighted — the next test is the one that matters.)
- **Weighting/covariance enforcement.** Build 4+ points on the exact power law
  *except one*, perturbed far off the line but given a very large `rate_err`. A
  correctly weighted fit down-weights it to near-negligible and recovers
  `slope ≈ m`; an unweighted fit is measurably pulled away. Compute the unweighted
  comparison in the test itself via `numpy.polyfit` with no weights and assert the
  weighted result is closer to `m`. Separately, assert `slope_err` matches a
  hand-computed weighted normal-equations variance to tight tolerance — **this is
  what catches a residual-rescaled `slope_err` masquerading as the correct one.**
- Fewer than 2 usable points (mixing `rate <= 0`, non-finite `rate`, non-finite
  or non-positive `rate_err`) returns `(nan, nan, nan, n_excluded)` with the
  correct count, and does not raise.
- Exactly 2 usable points plus at least one excluded returns finite `slope`,
  `slope_err`, `intercept` with the correct `n_excluded`; independently compute
  the expected unscaled two-point covariance so a residual-scaled implementation
  fails.
- 2+ usable points all sharing one redshift returns `(nan, nan, nan, n_excluded)`
  without raising — distinguishing it from the malformed cases below.
- Raises when `redshifts` contains `<= -1`, `nan`, or `inf`; and on mismatched
  shapes or non-1D input.
- `check_slope_consistency`: within range `True`; far outside `False`; non-finite
  slope / slope_err / expected_slope and non-positive `slope_err` all `False`;
  invalid `n_sigma` raises.

**Commands:** `venv/bin/python -m pytest tests/` (0 failed).

### Rollback Path

Revert the Slice 3 additions to `src/merger_rate.py` and
`tests/test_merger_rate.py`. Slices 1-2 remain functional.

---

## Slice 4: Evolution figure, results table, and analysis entry point

### Intended Change

Add to `src/merger_rate.py`:

**Module setup.** Call `matplotlib.use("Agg")` before importing `pyplot`,
matching `plot.py`'s non-interactive convention. Define a local `MASS_COLORS`
constant **duplicated from `plot.py`'s palette** rather than imported — frozen,
per the duplication convention in Architecture Fit, keeping `merger_rate.py` free
of `src/` imports at module level.

- `plot_merger_rate_evolution(config)` — reads the combined file through
  `_merger_rate_results_path(config)`, writes
  `figures/merger_rate_evolution.png`: one panel, **log-log axes on every code
  path**, one line + errorbars per mass bin (`rate` vs `1 + z`).

  Validate the stored `redshifts` up front: any non-finite or `<= -1` value
  raises before any Matplotlib call, matching `fit_log_rate_vs_redshift`'s frozen
  convention.

  `rate == 0` is a valid value (an empty bin, point 5) with no representation on
  a log axis. Apply the same usability mask as the fit (`rate` finite and `> 0`,
  `rate_err` finite and `> 0`) **before** the plotting call. Unusable points must
  never be passed to Matplotlib; a mass bin with no usable points has its
  line/errorbar call skipped without error.

  Because the symmetric plug-in uncertainty gives `rate_err == rate` at
  `N_pairs == 1`, define a display-only positive floor at
  `0.1 * min(all usable rates)` when at least one usable point exists. Build
  asymmetric plotting errors with the upper error unchanged and each lower
  endpoint at `max(rate - rate_err, display_floor)`. **Annotate the figure
  whenever any lower bar is clipped**, so it is not mistaken for the raw symmetric
  interval. Stored and printed scientific values are unchanged.

  If no usable point exists anywhere, write the labeled empty figure without
  computing a floor — **still log-log**. Do not let Matplotlib silently drop or
  warn about invalid log-axis values, and do not crash on `log10(0)`. Create
  `config["figures_dir"]` if necessary.

- `print_merger_rate_table(config)` — prints a table in the visual style of
  `plot.py`'s `print_stats_table`: `f_pair`, `N_pairs`, `N_gal`, `rate`,
  `sigma_rate` per `(mass_bin, redshift)`. Follow it with one line per mass bin
  reporting the fitted slope, its uncertainty, `expected_slope`, `n_excluded`, and
  whether `check_slope_consistency` passed — or **"insufficient data"** when
  `slope` is `nan`. The printed label must explicitly state this verifies
  **mock-data recovery of the merger-timescale model's known injected power law**,
  not a real merger-rate evolution claim about the universe.

- `run_merger_rate_analysis(config)` — calls `run_merger_rate_calculation`, then
  `plot_merger_rate_evolution`, then `print_merger_rate_table`, in that order.
  This **recomputes `results/merger_rate.hdf5` fresh on every call** — it is not a
  read-only viewer of a pre-existing rate file, mirroring how `--validate` already
  re-runs `make_plots` rather than assuming figures are current.

### Definition of Done

- [ ] `matplotlib.use("Agg")` precedes the `pyplot` import; `MASS_COLORS` is
      local, not imported from `plot.py`.
- [ ] Both axes are log-scaled on **every** path, including the all-unusable
      empty figure — asserted by inspecting the Axes, not by eye.
- [ ] Unusable points are provably never passed to `errorbar`.
- [ ] A `rate_err == rate` point has its lower endpoint clipped to the floor, its
      upper error unchanged, and the clipping annotation present.
- [ ] A malformed stored redshift raises **before** any Matplotlib call.
- [ ] The table prints "insufficient data" for a `nan`-slope bin, reports
      `n_excluded`, and carries the mock-data-recovery labeling.
- [ ] `run_merger_rate_analysis` calls calculation → plot → table in that order.
- [ ] **End-to-end on generated mock data:** `check_slope_consistency` is `True`
      for every mass bin with at least 2 usable points, against
      `expected_slope = -config["merger_timescale_alpha"]`.
- [ ] `venv/bin/python -m pytest tests/` passes, 0 failed.

### Acceptance Criteria

- **Inputs:** `config`; on-disk `results/pairs_z{z}.hdf5` for every configured
  redshift. Does **not** require a pre-existing `merger_rate.hdf5`.
- **Outputs:** `results/merger_rate.hdf5` (recomputed);
  `figures/merger_rate_evolution.png`; the console table.
- **User-visible:** none through the CLI yet — wired in Slice 5. Functions are
  called directly in tests.
- **Must not change:** everything validated in Slices 1-3.

### Authorized Surface

- Files allowed to change:
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- `merger_rate.py`: `plot_merger_rate_evolution`, `print_merger_rate_table`,
  `run_merger_rate_analysis`, the local `MASS_COLORS`, and the non-interactive
  backend setup. Earlier functions may be called, not modified.
- Tests: append only.

### Explicit Non-Goals

- No CLI wiring and no documentation updates — Slice 5.
- No change to `plot.py`.
- No hardcoded literature exponent as ground truth.
- No modification to `generate_test_data.py` to manufacture a trend (point 6 —
  a hard boundary).

### Risk Flags

- Risky surfaces: first Matplotlib dependency in this module; writes a figure.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

Append to `tests/test_merger_rate.py`:

- `plot_merger_rate_evolution` runs without error on generated mock data and
  writes `figures/merger_rate_evolution.png`.
- **Plot masking:** against a hand-written combined fixture containing zero,
  negative, and non-finite rates/errors plus one wholly unusable mass bin,
  monkeypatch or inspect Matplotlib calls to prove unusable points never reach
  `errorbar`, then confirm the PNG is still written. Include a valid
  `N_pairs == 1`-equivalent point with `rate_err == rate` and assert its lower
  endpoint is clipped to the frozen floor, its upper error is unchanged, and the
  annotation is present.
- **All-unusable fixture:** a labeled empty PNG is written without error, and
  **both axes are still log-scaled** (assert `get_xscale()` / `get_yscale()`).
- **Malformed stored redshift** (non-finite, and `<= -1`) raises before any
  `errorbar` call.
- **Table labeling:** capture `print_merger_rate_table` output for a fixture with
  one insufficient-data bin; assert it prints "insufficient data" for that bin,
  reports `n_excluded`, and explicitly labels the slope check as mock-data
  recovery of the injected model rather than a production validation.
- **Orchestration:** a direct test that `run_merger_rate_analysis` calls
  `run_merger_rate_calculation`, then `plot_merger_rate_evolution`, then
  `print_merger_rate_table`, in that order (call sites patched).
- **End-to-end on the isolated generated-mock fixture:**
  `run_merger_rate_calculation` + `fit_log_rate_vs_redshift` +
  `check_slope_consistency` with
  `expected_slope = -config["merger_timescale_alpha"]`; assert `True` for every
  mass bin with at least 2 usable redshift points. **This is the real scientific
  assertion of the plan** — verifying the injected timescale power law is
  recovered through the full pair-fraction → rate → log-log-fit chain — and it
  must be allowed to fail loudly if the rate calculation or error propagation is
  wrong.

**Commands:** `venv/bin/python -m pytest tests/` (0 failed).

**Manual:** inspect `figures/merger_rate_evolution.png` — lines should follow the
expected `(1+z)^(-merger_timescale_alpha)` trend (**not flat**) within errorbars
across the four mock redshifts, per point 6.

### Rollback Path

Revert the Slice 4 additions to `src/merger_rate.py` and
`tests/test_merger_rate.py`. Slices 1-3 remain functional.

---

## Slice 5: CLI wiring and documentation

### Intended Change

Add a `--merger-rate` flag to `src/pipeline.py`. It is **not** part of the
existing mutually exclusive group — it is additive and orthogonal to
`--calc-only` / `--plot-only` / `--generate-test` / `--validate`, analogous to how
`--validate` augments the plotting step. When set, `main()` calls
`run_merger_rate_analysis(config)` after the existing calc/plot logic, importing
it **inside `main()`** per the existing deferred-import pattern, so unrelated
invocations and `--help` do not pay for a Matplotlib import.

**Frozen flag-composition table** — do not reinterpret "additive":

| Invocation | Behaviour |
|---|---|
| `--merger-rate` | default calculation + plotting path, then merger-rate analysis (needs input catalogs, not pre-existing pair results) |
| `--calc-only --merger-rate` | recalculate pairs, skip velocity plots, then merger-rate analysis |
| `--plot-only --merger-rate` | use pre-existing pair results for both plotting paths |
| `--generate-test --merger-rate` | generate fresh catalogs and pair results, then analysis |
| `--validate --merger-rate` | full existing pipeline plus the analysis, one invocation |

The merger-rate calculation already asserts (fail loud, `calc.py` style) that
`results/pairs_z{z}.hdf5` exists for every configured redshift before proceeding,
matching the `--plot-only` precedent.

Documentation, scoped tightly:

- `README.md` — add a `--merger-rate` bullet to the Quick start command block;
  extend "What it does" with the optional merger-rate product; document the three
  new config keys and the `results/merger_rate.hdf5` /
  `figures/merger_rate_evolution.png` outputs; **replace the brittle exact test
  count** in the Tests section with a qualitative coverage statement including the
  merger-rate tests. Do not otherwise rewrite existing sections.
- `AGENTS.md` — add `--merger-rate` to the Running the Pipeline block; update the
  Architecture module count, diagram, and descriptions for `merger_rate.py`; add
  the timescale/rate units and the three new keys to the Configuration Reference.
  Do not otherwise rewrite existing sections.
- `docs/PLAN.md` — append `merger_rate.py` to the File Structure diagram **only**.
  Do not edit any other section of this historical document.

### Definition of Done

- [ ] `--merger-rate` parses alone and in combination with **each** mutually
      exclusive mode.
- [ ] `--merger-rate` is provably outside the mutually exclusive group.
- [ ] All five rows of the frozen composition table hold, verified by
      orchestration tests with call sites patched and no file I/O.
- [ ] The import of `run_merger_rate_analysis` occurs inside `main()`; `--help`
      does not import Matplotlib.
- [ ] The flag's help text states that it writes its own result file and figure
      even when composed with `--calc-only` / `--plot-only`.
- [ ] Every existing mode behaves exactly as before when `--merger-rate` is absent.
- [ ] The three docs are updated only within their scoped sections; the brittle
      test count is gone from `README.md`.
- [ ] `venv/bin/python -m pytest tests/` passes, 0 failed.
- [ ] `venv/bin/python src/pipeline.py --validate --merger-rate` exits 0 and
      produces `figures/merger_rate_evolution.png`.

### Acceptance Criteria

- **Inputs:** `config`; on-disk input catalogs in `data/`.
- **Outputs:** `results/merger_rate.hdf5`, `figures/merger_rate_evolution.png`,
  the console table, and the three updated docs.
- **User-visible:** `python src/pipeline.py --merger-rate` follows the normal
  calculation + plotting path then produces the new result, figure, and table,
  ending with the same `"Done."` pattern as the rest of `pipeline.py`.
  `--plot-only --merger-rate` is the mode requiring pre-existing pair results.
- **Must not change:** `--calc-only`, `--plot-only`, `--generate-test`,
  `--validate` behave exactly as before without `--merger-rate`.

### Authorized Surface

- Files allowed to change:
  - `src/pipeline.py`
  - `tests/test_merger_rate.py`
  - `README.md`
  - `AGENTS.md`
  - `docs/PLAN.md`
- `pipeline.py`: `parse_args` (add the flag), `main` (call the entry point), the
  module usage docstring, and affected flag help strings.
- **`src/merger_rate.py` is NOT in this slice's surface.** If a defect is found
  there, stop and report rather than fixing it here.
- Tests: append only.

### Explicit Non-Goals

- No changes to `merger_rate.py`, `plot.py`, `calc.py`, or `config.py`.
- No rewriting of `README.md` / `AGENTS.md` sections beyond those scoped above.
- No edits to `docs/PLAN.md` beyond the one File Structure line.

### Risk Flags

- Risky surfaces: public CLI flags — additive, existing behaviour unchanged; low
  risk but flagged as a CLI contract change.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

Append to `tests/test_merger_rate.py`:

- Patch `sys.argv`; assert `parse_args` accepts `--merger-rate` alone and with
  each existing mutually exclusive mode, that it defaults to `False` when absent,
  and that it is not in the mutually exclusive group.
- `main` orchestration tests with the calculation/plot/analysis call sites patched
  and no file I/O, covering **every** row of the frozen composition table plus the
  default (no `--merger-rate`) case, asserting call order.

**Commands:** `venv/bin/python -m pytest tests/` (0 failed);
`venv/bin/python src/pipeline.py --validate --merger-rate` (exit 0,
`figures/merger_rate_evolution.png` produced).

**Manual:** read the printed table's labeling and confirm it describes recovery of
the injected timescale model, not a literature merger-rate claim.

### Rollback Path

Revert `src/pipeline.py`'s flag and `main()` branch, the Slice 5 test additions,
and the doc updates. Slices 1-4 remain functional and testable; their functions
are simply unreachable from the CLI until this slice lands again.

---

## Appendix: Provenance

An earlier three-slice version of this plan was executed to completion on branch
`merger-rate/baseline` (implementation commits `250b386`..`711d4c1`; PM run
records under `.pm/runs/`). All six science points above were verified against the
written `results/merger_rate.hdf5`: the pair fraction and timescale reproduce to
0.0 absolute error; the `R = C * N_pairs / (V * T_merge)` identity and the Poisson
propagation to ~3e-16 relative error; and the injected power law was recovered in
all six mass bins (slopes 0.906-1.092 against the config-derived expected 1.000,
all within 1.1σ, residuals scattering about zero).

This version restructures that work for re-execution. Three changes, each from an
observed failure:

1. **Five slices instead of three.** The original Slice 3 bundled five functions,
   CLI wiring, and three documentation files into one ~970-line diff — too large
   for one reviewable unit and one developer turn.
2. **The Numerical Domain Contract** is new and binding. Its absence let review
   escalate indefinitely through unreachable float64 boundaries, which is what
   stopped both prior runs.
3. **Per-slice Definition-of-Done checklists.** Requirements stated only in prose
   were satisfied individually but not jointly — a log-log requirement and an
   empty-figure requirement in the same paragraph produced a linear-axis empty
   figure.

The full rationale and evidence behind these three changes were written up
separately as a set of proposed `project-manager` / `code-review` /
`implementation-plan` skill improvements, maintained outside this repository.
Nothing in this plan depends on that write-up; it is background only.

## Next Chat Prompt

Plan file: `/Users/dcroton/Local/git-repos/relative-velocity/docs/MERGER_RATE_PLAN.md`

Use the Mode B launcher in the `project-manager` skill's `SKILL.md` ("Launcher"),
with the repository set to `/Users/dcroton/Local/git-repos/relative-velocity` and
a user-selected harness/model. Run Slices 1-5 atomically in plan order; all five
are elevated-risk and require fresh PM-commissioned `drift-audit` and
`code-review` passes against each slice's exact final commit before acceptance.
