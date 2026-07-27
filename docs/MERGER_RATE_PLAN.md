# Implementation Plan: Close-Pair Merger Rate Estimation

## Purpose

Add a new analysis module, `src/merger_rate.py`, that converts close-pair
counts already produced by the existing pipeline (`pair_finder.py` /
`calc.py`) into a **merger rate density** using the standard observational
close-pair method (Kitzbichler & White 2008; see `docs/BACKGROUND.md`
references). The module computes pair fractions per stellar-mass bin and
redshift, converts them to merger rates via a parameterized merger
timescale, propagates Poisson uncertainty throughout, and produces a
redshift-evolution figure plus a table of results.

This plan targets a **frontier/senior implementer profile**: the three
slices are substantial, internally coupled (each slice's output is the next
slice's input), and each is reviewable as one clear diff. Batching is
offered for a strong implementer but not required.

This plan was reviewed across four rounds by an independent read-only
delegate (Codex, `gpt-5.6-sol`, medium effort) before implementation.
Round 1 found a scientific error in the original Slice 3 validation design
(see point 6 below), corrected in round 2 along with several
ambiguity/scope fixes; round 3 closed out most remaining fit-contract edge
cases (non-finite/duplicate redshift handling in `fit_log_rate_vs_redshift`)
and a few minor wording issues; round 4 closed the same class of gap in
`merger_timescale_gyr`'s input domain and a `compute_merger_rate` wording
mismatch. The version below reflects all four rounds. See the plan's Git
history for prior versions if needed.

## Scientific Background

See `docs/BACKGROUND.md` (§1, and the Kitzbichler & White 2008 / Jiang et
al. 2014 / Lagos et al. 2021 references already listed there) for full
context. Summary of the method this plan implements:

1. **Close pairs** are already defined by the existing pipeline: pairs
   within `config["max_sep"]` kpc 3D separation and mass ratio
   `>= config["mass_ratio_min"]`. Every row already stored in
   `results/pairs_z{z}.hdf5` by `calc.py` satisfies this definition — no
   new pair-finding is needed. `N_pairs(b, z)` denotes the count of such
   stored rows with `mass_bin == b`; each unordered pair is counted exactly
   once (matching `scipy.spatial.cKDTree.query_pairs`'s de-duplication,
   already relied on by `pair_finder.py`).

2. **Pair fraction** per stellar-mass bin `b` and redshift `z`:

   ```
   f_pair(b, z) = N_pairs(b, z) / N_gal(b, z)
   ```

   where `N_gal(b, z)` is the **total** number of mass-selected galaxies in
   that bin at that redshift (paired *and* unpaired) — this denominator
   does not currently exist anywhere in the pipeline's output and must be
   added (Slice 1). `f_pair` here is a **pairs-per-galaxy incidence
   ratio**, not a probability bounded by 1 — a single galaxy can appear in
   more than one stored close pair, so `f_pair` can exceed 1 in a crowded
   bin. This convention (rather than, e.g., "fraction of galaxies that are
   in at least one pair", which would need different bookkeeping) is what
   makes the algebra in point 4 below reduce exactly to
   `f_pair(b,z) * n_gal(b,z) = N_pairs(b,z) / V`, independent of `N_gal`.
   This module requires `config["mass_bin_by"] == "primary"` (see
   Architecture Fit). `"mean"` and `"total"` binning strategies assign a
   pair to a bin based on a joint quantity of *both* galaxies' masses (the
   mean of the two log-masses, or the log of their summed linear masses)
   that is not a property of any single galaxy, so no single-galaxy
   population is well-defined as `N_gal(b, z)` for those strategies at
   all — they genuinely cannot be supported without a fundamentally
   different denominator definition. `"secondary"` binning is different:
   it bins by the less-massive member's own mass, which in principle
   supports an analogous single-galaxy denominator the same way
   `"primary"` does (count all galaxies by their own mass). Restricting
   this module to `"primary"` only is therefore a **scope decision**, not
   a correctness requirement — `"secondary"` is excluded because
   implementing and correctly testing a second binning mode is out of
   scope for this plan, not because it is incoherent. `run_merger_rate_calculation`
   must still assert `config["mass_bin_by"] == "primary"` and raise a
   clear error for any other value, including `"secondary"`.

3. **Merger timescale** — the average time a close pair remains
   observably close before merging — modeled as a simple power law in
   redshift, a common simplified form used in close-pair studies:

   ```
   T_merge(z) [Gyr] = merger_timescale_gyr0 * (1 + z) ** merger_timescale_alpha
   ```

   `merger_timescale_gyr0` and `merger_timescale_alpha` are new `config.py`
   parameters (see Slice 2). This is a deliberately simplified model, not a
   fit to a specific simulation suite — the plan does not claim numerical
   agreement with any single published calibration, only architectural and
   dimensional correctness (an explicit non-goal below). **This redshift
   dependence is a deliberately injected model input, not something
   measured from the mock data — see point 6, which this fact directly
   drives.**

4. **Merger rate density** per stellar-mass bin and redshift, in
   `Gyr^-1 Mpc^-3` (comoving if the input positions/box size are comoving;
   see the box-size provenance note below — the pipeline does not
   currently guarantee this either way):

   ```
   R(b, z) = merger_fraction * f_pair(b, z) * n_gal(b, z) / T_merge(z)
   n_gal(b, z) = N_gal(b, z) / box_size_mpc(z)**3
   ```

   `merger_fraction` (a new `config.py` parameter, `C_merge` in the
   literature) is the fraction of close pairs that actually merge within
   `T_merge` rather than being chance projections or unbound flybys.
   Because `f_pair(b,z) * n_gal(b,z) = N_pairs(b,z) / box_size_mpc(z)**3`
   exactly (point 2), this is algebraically identical to:

   ```
   R(b, z) = merger_fraction * N_pairs(b, z) / (box_size_mpc(z)**3 * T_merge(z))
   ```

   The plan computes it via the `f_pair`/`n_gal` route (matching how the
   literature presents the method, and because `f_pair` and the raw counts
   `N_pairs`/`N_gal` are useful diagnostics reported in their own right —
   see Slice 3's table, which reports `f_pair`, `N_pairs`, and `N_gal`, not
   the derived number density `n_gal` itself), but the reduced form above
   is the key fact behind point 6's corrected validation logic.
   `box_size_mpc(z)` is the per-redshift box size actually used by
   `pair_finder.find_pairs` for that snapshot (`catalog["box_size"]`,
   loaded from the data file) — **not** `config["box_size"]` directly.
   These are expected to be equal for the existing mock test data, but the
   pipeline does not currently enforce that anywhere, and a real catalog
   could in principle have a per-snapshot box size that differs from a
   single global config value. Slice 1 must persist the per-redshift
   `box_size_mpc` value actually used into the results file so Slice 2
   reads the authoritative value for that snapshot rather than assuming
   agreement with `config["box_size"]`.

5. **Poisson uncertainty propagation.** `N_pairs(b, z)` is treated as
   Poisson-distributed; `N_gal(b, z)` is treated as an exact count (it is
   the full mass-selected sample, not a subsample draw), consistent with
   how close-pair studies typically quote errors (Poisson counting error on
   the numerator dominates). `T_merge` and `merger_fraction` are treated as
   fixed model inputs with no propagated uncertainty of their own — this is
   an explicit scope boundary, not an oversight (see Non-Goals). Under
   these assumptions:

   ```
   sigma_f_pair(b, z) = f_pair(b, z) / sqrt(N_pairs(b, z))     if N_pairs > 0
   sigma_f_pair(b, z) = 0                                       if N_pairs == 0
   sigma_R(b, z) = R(b, z) / sqrt(N_pairs(b, z))                if N_pairs > 0
   sigma_R(b, z) = 0                                             if N_pairs == 0
   ```

   The `N_pairs == 0` case is a frozen special case (not `NaN`, not a
   divide-by-zero) — an empty bin has zero rate and zero uncertainty in
   this plan's convention. **This is a deliberate plug-in point-estimate
   simplification for downstream plotting and fitting, not a statistically
   rigorous Poisson confidence interval** — a rigorous treatment would give
   a nonzero upper limit even for zero observed counts (e.g. a Gehrels
   upper limit). Implementing asymptotically-correct small-`N` Poisson
   confidence intervals is explicitly out of scope for this plan; code and
   docstrings must describe `sigma_R`/`sigma_f_pair` as this plan's
   specific convention, not as "the Poisson uncertainty" unqualified.

6. **The frozen redshift-evolution validation logic (Slice 3) — corrected
   after review.** `generate_test_data.py` draws a fixed `N_PAIRS = 3000`
   and `N_FIELD = 2000` at every redshift from the same mass and
   mass-ratio distributions, just reseeded per redshift (see
   `src/generate_test_data.py`) — so `N_pairs(b, z)` and `N_gal(b, z)` are
   expected to be statistically flat across the four mock redshifts (equal
   in expectation, scattering only via Poisson/sampling noise). **This
   does NOT mean the computed merger rate `R(b, z)` should come out flat.**
   By the reduced formula in point 4,
   `R(b,z) = merger_fraction * N_pairs(b,z) / (V * T_merge(z))`, and
   `T_merge(z)` is a deliberately injected power law,
   `T_merge(z) = T0 * (1+z)^alpha`. Substituting:

   ```
   R(b, z) ∝ (1 + z) ** (-alpha)
   ```

   With the flat `N_pairs(b,z)` input, `log10(R(b,z))` plotted against
   `log10(1+z)` is therefore expected to follow a straight line with slope
   **`-merger_timescale_alpha`** (i.e. `+1.0` under this plan's default
   `merger_timescale_alpha = -1.0`), not slope zero. Asserting flatness
   would be wrong and would correctly fail once implemented — an earlier
   version of this plan made exactly that mistake; do not repeat it.

   The correct, exactly-computable, and still-honest validation is: fit
   `log10(R(b,z))` vs `log10(1+z)` per mass bin (weighted least squares,
   weights from `sigma_R` propagated to log space), and confirm the fitted
   slope is statistically consistent with the *known, config-derived*
   expected value `-merger_timescale_alpha` — not with an independently
   asserted literature exponent, and not with zero. This is a meaningful
   check because the only source of the slope is the deliberately injected
   `T_merge(z)` factor (the pair-count input has no systematic trend by
   construction) — so this test verifies the merger-rate conversion and
   its error propagation are implemented correctly, without fabricating a
   false "the mock data recovers a real merger-rate evolution law" claim.
   `generate_test_data.py` is explicitly out of scope for this plan (see
   Non-Goals) — do not modify it.

   **This expected-slope comparison is a mock-data validation diagnostic,
   not a general-purpose production check, and must be labeled as such
   everywhere it appears (printed table, docstrings).** The derivation
   above depends on `N_pairs(b,z)/box_size_mpc(z)**3` being statistically
   flat across redshift, which is a property of *this specific mock
   catalog's construction* (fixed counts, same distributions, same box
   size, every redshift), not a property this plan can guarantee for any
   future real data source. This pipeline currently has exactly one data
   source — `generate_test_data.py`'s mock catalogs (see
   `docs/PLAN.md`'s implementation-order item 6: real SAGE catalog support
   is explicit future work, not implemented) — so this is not a live risk
   today, but the check must not be presented as if it would remain valid
   once a real catalog reader exists: a real merger-rate evolution or a
   redshift-varying box size would legitimately produce a different slope,
   and that must not be reported as an implementation defect. It is
   sufficient for this plan to state this limitation clearly rather than
   build any runtime mock-vs-real detection, since no real-data path
   exists yet to distinguish.

## Architecture Fit

- New file `src/merger_rate.py`, structured like `plot.py`: it reads
  already-written results files and writes new output files/figures; it
  does not re-run pair finding.
- Per repo convention (`_mass_bin_edges` is already duplicated
  independently in `pair_finder.py` and `plot.py`), `merger_rate.py` and
  the small addition to `calc.py` each define their own local
  `_mass_bin_edges(config)` helper rather than importing a private helper
  from another module. This follows the as-built pattern already present
  in the codebase; `docs/PLAN.md`'s DRY principle would suggest a single
  shared helper instead, and that tension is a pre-existing state of the
  repository this plan does not attempt to resolve.
- This module requires `config["mass_bin_by"] == "primary"` (see point 2
  above for why `"mean"`/`"total"` are fundamentally incoherent for this
  denominator while `"secondary"` is merely out of scope).
  `run_merger_rate_calculation` must assert this at the start and raise a
  clear error naming the unsupported strategy otherwise —
  `"secondary"`/`"mean"`/`"total"` are all explicitly out of scope for
  this plan.
- All new parameters live in `config.py`, passed explicitly — no
  hardcoded constants in `merger_rate.py` or `calc.py`.
- Fail loud: an inconsistent state — a mass bin with `n_pairs[b] > 0` but
  `n_galaxies_per_mass_bin[b] == 0` (a pair can't exist in a bin with zero
  galaxies) — raises a clear assertion error. A bin with both `n_pairs[b]
  == 0` and `n_galaxies_per_mass_bin[b] == 0` is valid (an empty bin, not
  an error) and yields `f_pair = 0, sigma_f_pair = 0`.
- Units follow the existing convention exactly: positions/box size in Mpc,
  no unit conversions introduced beyond what `pair_finder.py` already does.
  New units introduced (Gyr for timescales, `Mpc^-3` for number density,
  `Gyr^-1 Mpc^-3` for rate density) are documented inline and in this plan.
  The pipeline does not currently establish whether positions/box size are
  comoving or proper (`docs/PLAN.md` takes all values "at face value");
  this plan does not resolve that either — outputs are labeled
  `Gyr^-1 Mpc^-3` without an unqualified "comoving" claim.

## Implementation Profiles

- Recommended for frontier/senior implementer: slices may be run
  individually (default, checkpointed) or as `Batch A` (Slices 1-2) then
  Slice 3 separately, since Slice 3's plotting/validation is naturally
  reviewed on its own once the numeric core is settled.
- Recommended for standard implementer: run slices individually.
- Recommended for weaker implementer: run atomic slices one at a time,
  do not batch.

## Slice Batches

- Batch A: Slices 1-2 — both are pure numeric/data-schema work (no
  plotting, no CLI), share the same test file, and a strong implementer
  can reasonably complete and review them as one coherent diff. Slice 3
  (plotting + CLI wiring + the redshift-evolution sanity check) should
  always be reviewed separately since it is the slice with the sharpest
  scientific-judgment risk (see point 6 above).

---

## Slice 1: Galaxy-count denominator + pair-fraction calculation

### Intended Change
- Extend `calc.py` to compute, per redshift, the total number of
  mass-selected galaxies in each of the existing mass bins and write it as
  a new dataset `n_galaxies_per_mass_bin` (1D int array, length = number
  of mass bins) in each `results/pairs_z{z}.hdf5` file written by
  `_save_pairs`. Also add a new attr `box_size_mpc` (float) to each results
  file, set to `catalog["box_size"]` for that redshift — the same value
  `pair_finder.find_pairs` actually used for periodicity — so downstream
  volume calculations in Slice 2 do not have to assume it matches
  `config["box_size"]`. Both additions are purely additive: existing
  datasets (`mass_primary`, `mass_secondary`, `mass_ratio`,
  `separation_kpc`, `delta_v`, `mass_bin`, `sep_bin`) and existing attrs
  are unchanged.
- The galaxy count must come from the **full mass-selected catalog**
  (`load_galaxy_catalog()` output, already filtered to
  `[log_mass_min, log_mass_max]` by `data_reader.py`), not just galaxies
  that ended up in a pair. `calc.py` already loads this catalog per
  redshift in `run_calculation()` — reuse it directly. Extract the
  counting logic into a small, directly testable helper,
  `_count_galaxies_per_mass_bin(log_stellar_mass, config)` in `calc.py`,
  returning a 1D int array of length `n_mass_bins`.
- **Bin-edge convention must match `pair_finder._assign_mass_bins`
  exactly**: use the same `np.digitize`-based, right-open bin logic
  (locally reimplemented per the existing duplication convention), where a
  galaxy with `log_stellar_mass` exactly equal to `log_mass_max` falls
  outside all bins and is excluded from every count. This is true even
  though `data_reader.load_galaxy_catalog`'s selection mask is inclusive
  of `log_mass_max` — a pre-existing minor edge-case inconsistency in the
  codebase (between what `data_reader` selects and what `pair_finder`
  bins) that this plan does not attempt to fix. What matters here is that
  the new galaxy-count denominator uses the *identical* binning rule the
  existing pair numerator already uses, so the two are consistent with
  each other; test the exact boundary value explicitly (see Validation
  Plan).
- **This slice introduces no `mass_bin_by` assertion at all.** The
  `"primary"`-only restriction (Architecture Fit) applies to Slice 2's
  `run_merger_rate_calculation` entry point, which does not exist yet in
  this slice. `_count_galaxies_per_mass_bin` and `compute_pair_fraction`
  are strategy-agnostic (they only count galaxies by their own mass /
  operate on already-binned count arrays) and must remain independently
  testable in this slice without any `mass_bin_by` dependency or check.
- Create `src/merger_rate.py` with:
  - `_mass_bin_edges(config)` — same formula as `pair_finder.py` /
    `plot.py` (local copy, per convention).
  - `_results_path(z, config)` — same path convention as
    `calc.py`/`plot.py` (`results/pairs_z{z:.1f}.hdf5`).
  - `_load_pair_counts(z, config)` — reads one results file and returns
    `(n_pairs_per_bin, n_galaxies_per_bin, box_size_mpc)`, where
    `n_pairs_per_bin` is computed by counting `mass_bin == b` in the
    file's `mass_bin` dataset for each bin `b`, `n_galaxies_per_bin` is
    read directly from the `n_galaxies_per_mass_bin` dataset, and
    `box_size_mpc` is read from the new attr. The two count arrays are
    returned as 1D int arrays of length `n_mass_bins`, in bin-index order.
  - `compute_pair_fraction(n_pairs, n_galaxies)` — vectorized over the
    mass-bin array; returns `(f_pair, sigma_f_pair)` per the formulas in
    points 2/5 above. Must assert `n_galaxies[b] > 0` for every bin with
    `n_pairs[b] > 0` (a pair can't exist in a bin with zero galaxies) with
    a clear message; bins where both are zero are valid and yield
    `f_pair = 0, sigma_f_pair = 0`.

### Acceptance Criteria
- Inputs: `config` dict (existing keys; this slice's functions are
  `mass_bin_by`-agnostic and introduce no assertion on it — that begins in
  Slice 2); on-disk `results/pairs_z{z}.hdf5` files as written by the
  modified `calc.py`.
- Outputs: modified `results/pairs_z{z}.hdf5` files containing the new
  `n_galaxies_per_mass_bin` dataset and `box_size_mpc` attr;
  `merger_rate.py` exposing `_mass_bin_edges`, `_results_path`,
  `_load_pair_counts`, `compute_pair_fraction` as importable functions;
  `calc.py` exposing `_count_galaxies_per_mass_bin` as an importable,
  directly testable function.
- User-visible behaviour: running `python src/pipeline.py --calc-only`
  (or `--validate`/`--generate-test`) produces results files with two
  additional pieces of data (one dataset, one attr); no change to console
  output, plots, or existing datasets/attrs.
- Behaviour that must not change: `plot.py`'s existing figures and stats
  table; all existing tests in `tests/test_geometric.py`,
  `tests/test_pair_finder.py`, `tests/test_statistical.py` must still pass
  unmodified.

### Authorized Surface
- Files allowed to change:
  - `src/calc.py`
  - `src/merger_rate.py` (new file)
  - `tests/test_merger_rate.py` (new file)
- Functions/classes/components allowed to change:
  - `calc.py`: `_save_pairs` (add the two new writes), new
    `_count_galaxies_per_mass_bin` helper, `run_calculation` (call the new
    helper and pass its output through to `_save_pairs`)
  - `merger_rate.py`: new module — `_mass_bin_edges`, `_results_path`,
    `_load_pair_counts`, `compute_pair_fraction`
- Tests allowed or expected to change:
  - `tests/test_merger_rate.py` (new file, all tests in it)
  - No existing test file should need changes in this slice; if one does,
    stop and report rather than edit it silently.

### Explicit Non-Goals
- No merger timescale, merger rate conversion, uncertainty-on-rate,
  plotting, or CLI wiring in this slice — those are Slices 2 and 3.
- No changes to `pair_finder.py`, `data_reader.py`, `plot.py`,
  `generate_test_data.py`, or `config.py`.
- No change to the meaning or values of any existing dataset/attr in the
  results files.
- No `mass_bin_by` assertion or restriction logic in this slice at all —
  that begins in Slice 2's `run_merger_rate_calculation` (see Intended
  Change above); this slice's functions must work correctly regardless of
  `mass_bin_by`'s value, since they don't read it.
- No attempt to reconcile `data_reader.py`'s inclusive-upper-edge mass
  selection with `pair_finder.py`'s exclusive-upper-edge bin assignment —
  out of scope; this slice only requires its own new counting logic to be
  internally consistent with the existing pair-binning convention.

### Risk Flags
- Risky surfaces touched: on-disk data schema (`results/pairs_z{z}.hdf5`)
  — additive only, no existing reader depends on the file containing
  *only* the current dataset set (confirmed: no test in `tests/` opens
  these HDF5 files directly).
- Approval needed before implementation: no
- Independent audit required: no

### Validation Plan
- Tests to add/update (new file `tests/test_merger_rate.py`, following the
  hand-crafted-exact-recovery style of `tests/test_pair_finder.py`):
  - `compute_pair_fraction` on hand-crafted `n_pairs`/`n_galaxies` arrays
    (e.g. `n_pairs=[0, 5, 20]`, `n_galaxies=[10, 10, 10]`) recovers exact
    `f_pair` and `sigma_f_pair` values by hand-computed formula, including
    the `n_pairs == 0` special case (`f_pair == 0`, `sigma_f_pair == 0`,
    no `NaN`/`inf` anywhere in the output).
  - Assertion failure is raised when `n_galaxies[b] == 0` and
    `n_pairs[b] > 0` for some `b` (construct this inconsistent input
    directly; do not rely on real data producing it).
  - `_count_galaxies_per_mass_bin` (direct unit test, not just via a
    fixture): build a small synthetic array of `log_stellar_mass` values
    analogous to `tests/test_pair_finder.py`'s hand-crafted-catalog style,
    including at least one value exactly equal to `config["log_mass_max"]`
    (must be excluded from every bin count, per the frozen edge
    convention) and one exactly on an interior bin edge, and assert the
    exact expected per-bin counts.
  - `_load_pair_counts` against a small hand-written HDF5 fixture (or by
    running `calc.py` on a tiny synthetic catalog analogous to
    `tests/test_pair_finder.py`'s `_catalog()` helper) recovers known
    `n_pairs_per_bin`, `n_galaxies_per_mass_bin`, and `box_size_mpc`
    values.
  - `run_merger_rate_calculation`'s (Slice 2) or an equivalent entry
    point's assertion on `mass_bin_by != "primary"` is deferred to Slice 2
    since that's where the entry point is introduced; this slice only
    needs `_count_galaxies_per_mass_bin` and `compute_pair_fraction` to be
    correct in isolation, so no `mass_bin_by` test is required here.
- Commands to run: `pytest tests/` (must remain 80+ passed, 0 failed);
  `python src/pipeline.py --validate` (must complete without error and
  produce unchanged figures).
- Manual checks: open one regenerated `results/pairs_z2.0.hdf5` with
  `h5py` and confirm `n_galaxies_per_mass_bin` is present with length equal
  to the number of mass bins (every value a non-negative integer), and
  `box_size_mpc` matches `config["box_size"]` for the current test data.

### Rollback Path
- Revert `src/calc.py` to drop the new dataset/attr writes; move
  `src/merger_rate.py` and `tests/test_merger_rate.py` to `archive/`
  (per repository convention: never delete outright, archive instead — do
  not `git revert` the introducing commit, since that would remove the
  files outright rather than archive them). No other files touched, so
  this is a clean single-slice rollback.

---

## Slice 2: Merger timescale parameterization + rate conversion with uncertainty

### Intended Change
- Add new parameters to `config.py` under a clearly labeled `# Merger
  rate` section:
  - `merger_timescale_gyr0 = 2.2` — `T_merge` normalization at `z = 0`,
    Gyr (order-of-magnitude consistent with Kitzbichler & White 2008;
    see `docs/BACKGROUND.md` references — not asserted as an exact
    reproduction of their fit).
  - `merger_timescale_alpha = -1.0` — power-law index for the redshift
    dependence of `T_merge` (point 3 above). Slice 3's validation depends
    on this value's sign and magnitude via `expected_slope =
    -merger_timescale_alpha`; if this default changes later, that
    dependency changes with it (it's config-driven, not hardcoded, so no
    code change is needed, just be aware when tuning this value).
  - `merger_fraction = 0.6` — fraction of close pairs that merge within
    `T_merge` (`C_merge` in the literature).
- Add to `src/merger_rate.py`:
  - `merger_timescale_gyr(z, config)` — implements
    `T_merge(z) = merger_timescale_gyr0 * (1 + z) ** merger_timescale_alpha`.
    Must assert, on every call: `merger_timescale_gyr0` is finite and `> 0`;
    `merger_timescale_alpha` is finite; `z` is finite and `> -1` (the same
    physical domain `fit_log_rate_vs_redshift` enforces elsewhere in this
    plan — `1 + z <= 0` makes the power law undefined/sign-flipping, and a
    non-finite `z` or `alpha` can silently produce a finite-looking but
    meaningless result depending on the exponent); and that the resulting
    `T_merge(z)` is finite and `> 0`. All five checks apply per call, not
    just to the configured redshift list — `merger_timescale_gyr` is a
    small independently-testable function and must validate its own inputs
    rather than trust a caller.
  - `compute_merger_rate(f_pair, sigma_f_pair, n_galaxies, box_size_mpc,
    timescale_gyr, merger_fraction)` — vectorized over the mass-bin array
    for one redshift; implements points 4/5 above exactly, returning
    `(rate, sigma_rate)` in `Gyr^-1 Mpc^-3`. Must assert
    `0 < merger_fraction <= 1`, that `box_size_mpc` is finite and `> 0`
    (explicitly finite, not just positive — `box_size_mpc = inf` must be
    rejected the same way `timescale_gyr = inf` already is, per the
    Validation Plan below), and that `timescale_gyr` is finite and `> 0` —
    this function is documented as independently callable and must not
    assume its caller already validated either input.
  - `run_merger_rate_calculation(config)` — asserts
    `config["mass_bin_by"] == "primary"` first (see Architecture Fit),
    raising a clear error naming the actual value otherwise. Then, for
    each `z` in `config["redshifts"]`, calls `_load_pair_counts`,
    `compute_pair_fraction`, `merger_timescale_gyr`, `compute_merger_rate`
    (passing the per-redshift `box_size_mpc` read from that results file,
    not `config["box_size"]`); assembles arrays over
    `(redshift, mass_bin)`; writes a single combined file
    `results/merger_rate.hdf5` containing datasets shaped
    `(n_redshifts, n_mass_bins)`: `pair_fraction`, `pair_fraction_err`,
    `n_pairs`, `n_galaxies`, `merger_rate`, `merger_rate_err`, plus two 1D
    datasets each with one value per redshift: `merger_timescale_gyr` and
    `box_size_mpc` (the per-redshift box size actually used for that
    row's rate calculation, persisted here so the combined file is
    independently auditable without re-opening every `pairs_z{z}.hdf5`),
    and attrs: `redshifts`, `mass_bin_edges`, `merger_fraction`,
    `merger_timescale_gyr0`, `merger_timescale_alpha`, `timestamp` —
    mirroring the provenance-attrs pattern already used in `calc.py`'s
    `_save_pairs`.
  - Fail loud: `run_merger_rate_calculation` asserts each required
    `results/pairs_z{z}.hdf5` file exists before starting, with the same
    style of message `calc.py` already uses for its own missing-file case.

### Acceptance Criteria
- Inputs: `config` dict including the three new parameters
  (`mass_bin_by` must be `"primary"`); on-disk `results/pairs_z{z}.hdf5`
  files (with `n_galaxies_per_mass_bin`/`box_size_mpc` from Slice 1) for
  every redshift in `config["redshifts"]`.
- Outputs: `results/merger_rate.hdf5` with the datasets/attrs listed
  above; `merger_timescale_gyr` and `compute_merger_rate` importable and
  independently callable (no file I/O) for unit testing.
- User-visible behaviour: none yet — no CLI flag calls this in Slice 2
  (wired up in Slice 3). The functions are tested directly.
- Behaviour that must not change: everything validated in Slice 1 remains
  true; `config.py`'s existing keys are unchanged in value or meaning.

### Authorized Surface
- Files allowed to change:
  - `src/config.py`
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- Functions/classes/components allowed to change:
  - `config.py`: add the three new keys only; do not alter any existing
    key's value or meaning
  - `merger_rate.py`: new `merger_timescale_gyr`, `compute_merger_rate`,
    `run_merger_rate_calculation`; Slice 1's functions may be called but
    not modified in this slice unless a genuine defect is found (report it
    rather than silently patching it in this slice's diff)
- Tests allowed or expected to change:
  - `tests/test_merger_rate.py` (append new tests only; do not remove or
    weaken Slice 1's tests)

### Explicit Non-Goals
- No plotting, no CLI wiring, no redshift-evolution fit/sanity check —
  those are Slice 3.
- No uncertainty modeling on `T_merge` or `merger_fraction` themselves —
  frozen as fixed model inputs per point 5.
- No change to `merger_fraction`'s applicability across mass bins — it is
  a single global config value in this plan, not mass- or
  redshift-dependent (a real extension, but out of scope here).
- No support for `mass_bin_by` strategies other than `"primary"`.

### Risk Flags
- Risky surfaces touched: introduces a new persistent output file
  (`results/merger_rate.hdf5`) with its own frozen dataset/attribute
  schema — a new persistence contract, though lower risk than modifying an
  existing one (Slice 1's additive change to `results/pairs_z{z}.hdf5` is
  the only existing-file schema touched, and was already reviewed there).
- Approval needed before implementation: no
- Independent audit required: no

### Validation Plan
- Tests to add/update (append to `tests/test_merger_rate.py`):
  - `merger_timescale_gyr` at `z = 0` returns exactly
    `merger_timescale_gyr0`; at `z = 3` returns
    `merger_timescale_gyr0 * 4 ** merger_timescale_alpha` to floating-point
    precision, using hand-picked `config` values distinct from the
    defaults (so the test can't pass by coincidence).
  - `merger_timescale_gyr` raises on `z <= -1`, `z = nan`, `z = inf`,
    `merger_timescale_gyr0 <= 0`, `merger_timescale_gyr0 = nan`/`inf`, and
    `merger_timescale_alpha = nan`/`inf` — covering the same
    finite-and-in-domain contract `fit_log_rate_vs_redshift` enforces for
    redshift elsewhere in this plan, not just the non-positive cases.
  - `compute_merger_rate` on hand-crafted scalars (pick `f_pair`,
    `sigma_f_pair`, `n_galaxies`, `box_size_mpc`, `timescale_gyr`,
    `merger_fraction` such that the arithmetic is easy to verify by hand)
    recovers the exact expected `rate` and `sigma_rate`.
  - `compute_merger_rate` raises on `merger_fraction <= 0`,
    `merger_fraction > 1`, `box_size_mpc <= 0`, `box_size_mpc = inf`,
    `box_size_mpc = nan`, `timescale_gyr <= 0`, `timescale_gyr = nan`, and
    `timescale_gyr = inf` — both the non-positive and the non-finite cases
    must be covered for both inputs, not just non-positive.
  - `run_merger_rate_calculation` raises a clear error when
    `config["mass_bin_by"] != "primary"` (construct a config copy with,
    e.g., `mass_bin_by = "mean"`).
  - **Box-size provenance test (proves the per-file value is actually
    used, not `config["box_size"]`):** build a small hand-written
    `results/pairs_z{z}.hdf5`-shaped fixture (reusing the Slice 1 fixture
    approach) whose `box_size_mpc` attr is deliberately set to a value
    that **differs** from `config["box_size"]` (e.g. fixture uses `250.0`
    while `config["box_size"]` is left at its default `500.0`), run
    `compute_merger_rate`/`run_merger_rate_calculation` against it, and
    assert the resulting `rate` matches the value computed by hand using
    the *fixture's* `box_size_mpc` (i.e. scales as
    `box_size_mpc_fixture ** -3`) — an implementation that accidentally
    reads `config["box_size"]` instead of the per-file attr would produce
    a detectably different, wrong rate under this test and must fail it.
  - `run_merger_rate_calculation` end-to-end on the real generated test
    data (after `--generate-test` + `--calc-only`): asserts
    `results/merger_rate.hdf5` exists, has the expected dataset shapes
    `(len(config["redshifts"]), n_mass_bins)`, and that every
    `merger_rate` value is finite and `>= 0`.
- Commands to run: `pytest tests/` (0 failed).
- Manual checks: inspect `results/merger_rate.hdf5` attrs with `h5py` and
  confirm `merger_fraction`, `merger_timescale_gyr0`,
  `merger_timescale_alpha` match the values in `config.py`.

### Rollback Path
- Revert the three new `config.py` keys and the Slice 2 additions to
  `src/merger_rate.py`/`tests/test_merger_rate.py`. Slice 1's
  `n_galaxies_per_mass_bin`/`box_size_mpc` additions and functions are
  unaffected and can remain in place independently.

---

## Slice 3: Redshift-evolution output, plot, and self-consistency sanity check

### Intended Change
- Add to `src/merger_rate.py`:
  - `fit_log_rate_vs_redshift(rates, rate_errs, redshifts)` — operates on
    a **single mass bin's** 1D array-like inputs (`rates`, `rate_errs`, one
    value per redshift in `config["redshifts"]`; all three the same
    length; raise if the three input lengths differ). Higher-dimensional
    input is out of scope/undefined by this contract — the only caller,
    `print_merger_rate_table`, only ever passes genuinely 1D per-redshift
    arrays, so no additional shape validation is required. Fits
    `log10(rate)` vs `log10(1 + z)` by weighted least squares (weights
    `1 / sigma_log_rate^2`, propagating `sigma_rate` to log space via
    `sigma_log_rate = sigma_rate / (rate * ln(10))`).

    **Malformed inputs — raise, do not filter around them:** any
    `redshift` that is `<= -1` or non-finite (`nan`/`inf`) is a malformed
    input for this function (`log10(1+z)` is undefined or infinite), not
    a data condition to exclude gracefully — raise immediately if any
    element of `redshifts` fails `-1 < redshift < inf`.

    **A redshift point is otherwise "usable" only if all of the following
    hold:** `rate` is finite and `> 0`; `rate_err` is finite and `> 0`.
    Points failing either condition are excluded from the fit (not an
    error — this is the normal "empty bin" or "excluded" case from
    point 5), contributing to the single `n_excluded` count.

    **Distinctness requirement:** after applying the usability filter,
    if fewer than 2 usable points remain, *or* the usable points do not
    span at least 2 distinct redshift values (e.g. every usable point
    happens to share the same `redshift`, which would make the fit
    rank-deficient), return `(nan, nan, nan, n_excluded)` for that bin
    rather than raising or fabricating a fit — this is a data condition
    (possible in principle, even if it never occurs for this pipeline's
    four distinct configured redshifts), not a malformed-input error, so
    it must not raise. Either way, the bin must be reported as
    "insufficient data" downstream, not silently omitted.
    `slope_err` must come from the
    weighted least-squares parameter covariance computed directly from the
    supplied `sigma_log_rate` values as the true measurement errors — do
    not use a fitting routine's default residual-based error rescaling
    (e.g. `scipy.optimize.curve_fit` without `absolute_sigma=True`), since
    that would silently substitute a different, data-driven error estimate
    for the one this plan already computed and propagated. A correct
    implementation is, e.g., explicit weighted normal equations
    (`numpy.polyfit(..., w=1/sigma_log_rate, cov=True)` uses an
    unscaled covariance only when `cov="unscaled"` is passed — verify
    whichever routine is used actually returns the *unscaled* covariance,
    not the residual-rescaled one, which is the default for most library
    weighted-fit helpers). Returns `(slope, slope_err, intercept,
    n_excluded)` for this one bin. The caller (`print_merger_rate_table`)
    loops over mass bins and calls this once per bin — this function
    itself has no knowledge of mass bins.
  - `check_slope_consistency(slope, slope_err, expected_slope,
    n_sigma=3.0)` — returns `True` if
    `abs(slope - expected_slope) < n_sigma * slope_err`, `False`
    otherwise, and `False` (not an error) if `slope` is `nan` (the
    insufficient-data case must be treated as "check not applicable", and
    the caller must report it as such rather than as a pass or fail).
    `expected_slope` is always `-config["merger_timescale_alpha"]` for
    this plan's use (see point 6 above) — the function itself is generic
    and takes it as a parameter rather than hardcoding it, so it stays
    directly unit-testable against arbitrary hand-picked values. Pure
    function, no I/O.
  - `plot_merger_rate_evolution(config)` — reads `results/merger_rate.hdf5`,
    produces `figures/merger_rate_evolution.png`: one panel, log-log axes,
    one line + errorbars per mass bin (`rate` vs `1 + z`). `rate == 0` is a
    valid value (an empty bin, per point 5) but has no representation on a
    log axis — for a given mass bin, redshift points with `rate <= 0` must
    simply be masked out of that bin's plotted line/errorbars (not passed
    to the plotting call at all), the same exclusion `fit_log_rate_vs_redshift`
    already applies; do not let matplotlib silently drop or warn about
    them, and do not crash on `log10(0)`. Define a local
    `MASS_COLORS` constant in `merger_rate.py` (duplicated from `plot.py`'s
    palette, per the existing duplication convention noted in Architecture
    Fit) rather than importing it from `plot.py` — this keeps
    `merger_rate.py`'s only dependency on other `src/` modules limited to
    what Slice 1 already established (none at import time; it reads HDF5
    files written by `calc.py`, it does not import `calc.py`/`plot.py`
    directly). This is a frozen choice, not left to implementer discretion.
  - `print_merger_rate_table(config)` — prints a table (same visual style
    as `plot.py`'s `print_stats_table`) of `f_pair`, `N_pairs`, `N_gal`,
    `rate`, `sigma_rate` per `(mass_bin, redshift)`, followed by one line
    per mass bin reporting the fitted slope, its uncertainty, the
    `expected_slope` value, and whether `check_slope_consistency` passed
    (or "insufficient data" if `slope` was `nan`) — explicitly labeled in
    the printed output as verifying **recovery of the merger-timescale
    model's known injected power law**, not a real merger-rate evolution
    claim about the universe (the printed label text itself must make
    this distinction).
  - `run_merger_rate_analysis(config)` — calls
    `run_merger_rate_calculation`, then `plot_merger_rate_evolution`, then
    `print_merger_rate_table`; the single entry point Slice 3 wires into
    the CLI. This recomputes `results/merger_rate.hdf5` fresh on every
    call (via `run_merger_rate_calculation`) — it is not a read-only
    viewer of a pre-existing rate file; this mirrors how `pipeline.py`'s
    existing `--validate` flag already re-runs `make_plots` rather than
    assuming figures are current.
- Wire into `src/pipeline.py`: add a new `--merger-rate` flag (not part of
  the existing mutually exclusive group — it is additive/orthogonal to
  `--calc-only`/`--plot-only`/`--generate-test`/`--validate`, analogous to
  how `--validate` augments the plotting step). When set, `main()` calls
  `run_merger_rate_analysis(config)` after the existing calc/plot logic,
  which itself asserts (fail loud, same style as `calc.py`) that
  `results/pairs_z{z}.hdf5` exists for every configured redshift before
  proceeding — matching the existing `--plot-only` precedent of requiring
  prior results.
- Update `README.md` and `AGENTS.md`/`docs/PLAN.md` (docs only) to mention
  the new `--merger-rate` flag and `merger_rate.py` module, following the
  existing documentation style for each file.

### Acceptance Criteria
- Inputs: on-disk `results/pairs_z{z}.hdf5` files for every configured
  redshift (the Slice 1/2 prerequisite); `config` dict.
  `run_merger_rate_analysis` recomputes `results/merger_rate.hdf5` fresh
  via `run_merger_rate_calculation` on every invocation — it does not
  require a pre-existing `results/merger_rate.hdf5`.
- Outputs: `results/merger_rate.hdf5` (recomputed);
  `figures/merger_rate_evolution.png`; console table via
  `print_merger_rate_table`; updated `README.md`/`AGENTS.md`/`docs/PLAN.md`.
- User-visible behaviour: `python src/pipeline.py --merger-rate` (after a
  prior `--calc-only` or `--validate` run has produced
  `results/pairs_z{z}.hdf5` files) produces the new figure and prints the
  table, ending with the same `"Done."` message pattern as the rest of
  `pipeline.py`. `python src/pipeline.py --validate --merger-rate` runs the
  full existing pipeline plus this analysis in one invocation.
- Behaviour that must not change: `--calc-only`, `--plot-only`,
  `--generate-test`, `--validate` behave exactly as before when
  `--merger-rate` is not passed.

### Authorized Surface
- Files allowed to change:
  - `src/merger_rate.py`
  - `src/pipeline.py`
  - `tests/test_merger_rate.py`
  - `README.md`
  - `AGENTS.md`
  - `docs/PLAN.md`
- Functions/classes/components allowed to change:
  - `merger_rate.py`: new `fit_log_rate_vs_redshift`,
    `check_slope_consistency`, `plot_merger_rate_evolution`,
    `print_merger_rate_table`, `run_merger_rate_analysis`, local
    `MASS_COLORS` constant
  - `pipeline.py`: `parse_args` (add `--merger-rate`), `main` (call
    `run_merger_rate_analysis` when set)
  - `README.md`: add a `--merger-rate` bullet to the "Quick start" command
    block, and a short module description near "Configuration" or a new
    small subsection; do not otherwise rewrite existing sections
  - `AGENTS.md`: add `--merger-rate` to the "Running the Pipeline" command
    block, and mention `merger_rate.py` in the "Architecture" module list;
    do not otherwise rewrite existing sections
  - `docs/PLAN.md`: append `merger_rate.py` to the "File Structure" diagram
    only; do not edit any other section of this historical planning
    document
- Tests allowed or expected to change:
  - `tests/test_merger_rate.py` (append new tests only)

### Explicit Non-Goals
- No modification to `generate_test_data.py` to manufacture a redshift
  trend (see point 6 — this is a hard boundary, not a suggestion).
- No hardcoded "expected" literature exponent asserted anywhere in code
  or tests as ground truth for the mock data — `expected_slope` must
  always be derived from `config["merger_timescale_alpha"]`, never a
  separately hardcoded literal.
- No change to `plot.py` itself — `merger_rate.py` defines its own
  `MASS_COLORS` rather than importing `plot.py`'s.

### Risk Flags
- Risky surfaces touched: public CLI flags (`pipeline.py` — additive new
  flag, existing flags' behaviour unchanged; low risk but flagged per the
  "risky surfaces" checklist since it is a CLI contract change).
- Approval needed before implementation: no
- Independent audit required: no

### Validation Plan
- Tests to add/update (append to `tests/test_merger_rate.py`):
  - `fit_log_rate_vs_redshift` on a hand-crafted array following an exact
    power law (`rate = A * (1+z)**m` for a hand-picked `m`, with tiny,
    *equal* hand-picked errors) recovers `slope ≈ m` to within a small
    numerical tolerance. (An exact power law with equal errors is
    invariant to weighting and cannot by itself prove the fit is actually
    weighted — see the heteroscedastic test below, which is the one that
    matters.)
  - **Weighting/covariance enforcement test (the one that actually proves
    the implementation is weighted and uses unscaled covariance, not just
    that it can fit a straight line):** construct a hand-crafted array of
    4+ points that follows the exact power law `rate = A * (1+z)**m` at
    every point *except one*, where that one point is deliberately
    perturbed far off the line but given a very large `rate_err` (so a
    correctly weighted fit should down-weight it near to negligible,
    recovering `slope ≈ m` almost exactly; an unweighted or incorrectly
    weighted fit would be measurably pulled away from `m` by the outlier).
    Assert the recovered `slope` is close to `m` (much closer than an
    unweighted fit of the same points would be — compute the unweighted
    comparison value in the test itself via `numpy.polyfit` with no
    weights and assert the weighted result is closer to `m`). Separately,
    assert the returned `slope_err` matches an independently, explicitly
    computed weighted-least-squares parameter variance (compute this by
    hand in the test via the normal-equations formula, not by calling the
    same library routine the implementation uses) to a tight numerical
    tolerance — this is what actually catches a residual-rescaled
    (`curve_fit` default) `slope_err` masquerading as the correct one.
  - `fit_log_rate_vs_redshift` with fewer than 2 usable points (mix of
    `rate <= 0`, non-finite `rate`, non-finite or non-positive `rate_err`)
    returns `(nan, nan, nan, n_excluded)` with the correct `n_excluded`
    count, and does not raise.
  - `fit_log_rate_vs_redshift` with exactly 2+ usable points that all
    share the same `redshift` value (a rank-deficient/non-distinct-x
    case) returns `(nan, nan, nan, n_excluded)` and does not raise —
    distinguishing this from the malformed-input cases below, which must
    raise.
  - `fit_log_rate_vs_redshift` raises when `redshifts` contains a value
    `<= -1`, when `redshifts` contains a `nan` or `inf` value, and when the
    three input arrays have mismatched lengths.
  - `check_slope_consistency`: a slope within `n_sigma * slope_err` of
    `expected_slope` returns `True`; a slope far outside that range
    returns `False`; a `nan` slope returns `False`.
  - End-to-end: after `--generate-test` + `--calc-only`,
    `run_merger_rate_calculation` + `fit_log_rate_vs_redshift` +
    `check_slope_consistency` on the real generated test data, using
    `expected_slope = -config["merger_timescale_alpha"]`: assert
    `check_slope_consistency` is `True` for every mass bin with at least 2
    usable redshift points — this is the real, meaningful assertion this
    slice makes about the actual pipeline (verifying the injected
    timescale power law is correctly recovered through the full
    pair-fraction → rate → log-log-fit chain), and it must be allowed to
    fail loudly if the error propagation or rate calculation is wrong.
  - `plot_merger_rate_evolution` runs without error on the real generated
    test data and writes `figures/merger_rate_evolution.png`.
- Commands to run: `pytest tests/` (0 failed); `python src/pipeline.py
  --validate --merger-rate` end to end, confirm exit code 0 and that
  `figures/merger_rate_evolution.png` is produced.
- Manual checks: visually inspect `figures/merger_rate_evolution.png` —
  lines should follow the expected `(1+z)^(-merger_timescale_alpha)` trend
  (not flat) within errorbars across the four mock redshifts, consistent
  with point 6's prediction; read the printed table's labeling to confirm
  it correctly describes this as recovering the injected timescale model,
  not a literature merger-rate claim.

### Rollback Path
- Revert `src/pipeline.py`'s `--merger-rate` flag and `main()` branch;
  revert the Slice 3 additions to `src/merger_rate.py` and
  `tests/test_merger_rate.py`; revert the doc updates. Slices 1-2 remain
  functional and testable independently (their functions are simply
  unused by the CLI until Slice 3 lands again).

---

## Next Chat Prompt

Plan file: `docs/MERGER_RATE_PLAN.md`
Slices or batch this session: Batch A (Slices 1-2)

Read the full plan file first. If a selected slice or batch receipt is incomplete or the plan state is unclear, stop and tell me before coding.

Work on the current feature branch for this plan; if none exists, create one and tell me the name.

Use orchestrator as the controlling skill. Act as the Developer: keep implementation, validation, Git operations, and commits local. Use a read-only Reviewer only for investigation, evidence gathering, the hostile drift-audit skill, and an independent code-review skill pass. If no Reviewer is configured or available, perform Developer self-audit and record that provenance explicitly.

For each selected slice or batch, in plan order:
1. Restate the frozen contract (authorized surface + non-goals) from the plan.
2. If any included slice's Risk Flags mark approval-needed, stop and get my approval before coding.
3. apply the scoped-implementation skill against the selected contract.
4. apply the drift-audit skill using a read-only Reviewer when available; otherwise perform Developer self-audit. Report the authorization gate result and who performed it before any quality review.
5. If the gate passes, apply the code-review skill using a read-only Reviewer when available; otherwise perform Developer self-audit through the code-review skill. Record who performed it. If the drift gate fails, fix the drift and re-audit.
6. Surface drift and review findings to me, fix them, then re-run the relevant gate. If consecutive reviews return only minor findings and have clearly converged record residuals in the slice summary and proceed.
7. Ask me before committing. On my approval, commit the selected slice or batch with the commit skill.

After the selected slice(s) or batch are committed, use the handoff skill to record state, audit provenance (Reviewer tool/label or Developer self-audit and fallback context), and the next slice or batch to resume from. Do not continue past the selected scope.

Confirm before starting: plan file read, selected slice(s) or batch, branch, and the first slice. Then begin.
