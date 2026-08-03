# Implementation Plan: Close-Pair Merger Rate Estimation

## Purpose

Add `src/merger_rate.py`, converting the close-pair counts already produced by
`pair_finder.py` / `calc.py` into a **merger rate density** via the standard
observational close-pair method (Kitzbichler & White 2008; see
`docs/BACKGROUND.md`). The module computes pair fractions per stellar-mass bin
and redshift, converts them to merger rates through a parameterized merger
timescale, propagates Poisson uncertainty, and verifies that the redshift
evolution of the result matches the injected timescale model.

This delivers the computational core only. **Plotting, CLI flags, and
documentation updates are out of scope for every slice in this plan** and are not
restated per slice: the outputs are a results file and a per-mass-bin console
summary, consumed directly or by a later presentation layer.

## Scientific Background

See `docs/BACKGROUND.md` §1 and its Kitzbichler & White 2008 / Jiang et al. 2014
/ Lagos et al. 2021 references.

**1. Close pairs** are already defined by the pipeline: within
`config["max_sep"]` kpc 3D separation and mass ratio `>= config["mass_ratio_min"]`.
Every row in `results/pairs_z{z}.hdf5` satisfies this — no new pair-finding is
needed. `N_pairs(b, z)` counts stored rows with `mass_bin == b`; each unordered
pair is counted once, matching `cKDTree.query_pairs` de-duplication.

**2. Pair fraction** per mass bin `b` and redshift `z`:

```text
f_pair(b, z) = N_pairs(b, z) / N_gal(b, z)
```

`N_gal(b, z)` is the **total** number of mass-selected galaxies in that bin —
paired *and* unpaired. This denominator does not exist in the pipeline's output
today and must be added (Slice 1).

`f_pair` is a **pairs-per-galaxy incidence ratio, not a probability bounded by
1**: one galaxy can appear in several stored pairs, so `f_pair > 1` is legal in a
crowded bin. This convention is what makes the algebra in point 4 reduce exactly
to `f_pair * n_gal = N_pairs / V`, independent of `N_gal`.

The module requires `config["mass_bin_by"] == "primary"`. `"mean"` and `"total"`
bin on a joint quantity of both galaxies' masses, which is not a property of any
single galaxy, so no single-galaxy `N_gal(b, z)` is definable for them.
`"secondary"` bins on the less-massive member's own mass and could support an
analogous denominator; excluding it is a scope decision, not a correctness
requirement. `run_merger_rate_calculation` asserts `"primary"` and reports any
other value.

**3. Merger timescale** — the average time a pair remains observably close before
merging — as a power law in redshift:

```text
T_merge(z) [Gyr] = merger_timescale_gyr0 * (1 + z) ** merger_timescale_alpha
```

Both are new `config.py` parameters (Slice 2). This is a deliberately simplified
model, not a fit to any simulation suite; the plan claims architectural and
dimensional correctness only. **This redshift dependence is an injected model
input, not something measured from the data** — point 6 depends on that fact.

**4. Merger rate density** per mass bin and redshift, in `Gyr^-1 Mpc^-3`:

```text
R(b, z)     = merger_fraction * f_pair(b, z) * n_gal(b, z) / T_merge(z)
n_gal(b, z) = N_gal(b, z) / box_size_mpc(z)**3
```

`merger_fraction` (new config parameter, `C_merge` in the literature) is the
fraction of close pairs that actually merge within `T_merge` rather than being
chance projections or unbound flybys.

Because `f_pair * n_gal = N_pairs / box_size_mpc**3` exactly, this is
algebraically identical to:

```text
R(b, z) = merger_fraction * N_pairs(b, z) / (box_size_mpc(z)**3 * T_merge(z))
```

Compute it via the `f_pair` / `n_gal` route — that matches how the literature
presents the method, and `f_pair` and `N_pairs` are reported as diagnostics in
their own right. The reduced form is the key fact behind point 6.

`box_size_mpc(z)` is the per-redshift box size `pair_finder.find_pairs` actually
used for that snapshot (`catalog["box_size"]`) — **not** `config["box_size"]`.
These are equal for the current mock data, but nothing enforces that, and a real
catalog could differ per snapshot. Slice 1 persists the value actually used so
Slice 2 reads the authoritative one.

The volume normalization inherits the pipeline's unit philosophy: coordinates are
taken at face value, and the pipeline does not establish whether they are
comoving or proper. Outputs are therefore labelled `Gyr^-1 Mpc^-3` with no
unqualified "comoving" claim, and the *absolute* normalization is not directly
comparable to an observational rate until a reader pins that convention. The
*shape* of `R` across mass bins at a fixed redshift is unaffected by a uniform
volume normalization; its shape across redshift is not guaranteed until the
coordinate convention is fixed, because a comoving-to-proper conversion is itself
redshift-dependent through `a = 1 / (1 + z)`.

**5. Poisson uncertainty.** `N_pairs` is treated as Poisson-distributed; `N_gal`
as an exact count (it is the full mass-selected sample, not a subsample).
`T_merge` and `merger_fraction` are fixed model inputs with no propagated
uncertainty — an explicit scope boundary.

```text
sigma_f_pair(b, z) = f_pair(b, z) / sqrt(N_pairs(b, z))   if N_pairs > 0, else 0
sigma_R(b, z)      = R(b, z) / sqrt(N_pairs(b, z))        if N_pairs > 0, else 0
```

The `N_pairs == 0` case is exactly zero — not `nan`, not a divide-by-zero. This
is a **plug-in point-estimate simplification for downstream fitting, not a
rigorous Poisson confidence interval**: a rigorous treatment would give a nonzero
upper limit even at zero counts (e.g. Gehrels). Small-`N` Poisson intervals are
out of scope. Code and docstrings must describe these as *this plan's
convention*, never as "the Poisson uncertainty" unqualified.

**6. Redshift-evolution validation.** `generate_test_data.py` draws a fixed
`N_PAIRS = 3000` and `N_FIELD = 2000` at every redshift from the same
distributions, reseeded per redshift, so `N_pairs(b, z)` and `N_gal(b, z)` are
statistically flat across the four snapshots.

**This does not mean `R(b, z)` should come out flat.** By the reduced formula,
`R ∝ N_pairs / T_merge` with `T_merge = T0 * (1+z)^alpha`, so:

```text
R(b, z) ∝ (1 + z) ** (-alpha)
```

`log10(R)` against `log10(1+z)` is therefore expected to be a straight line of
slope **`-merger_timescale_alpha`** (i.e. `+1.0` at the default `alpha = -1.0`),
**not slope zero**. Asserting flatness would be wrong and would correctly fail.

The validation is: fit `log10(R)` vs `log10(1+z)` per mass bin by weighted least
squares, weights from `sigma_R` propagated to log space, and confirm the fitted
slope is consistent with the *config-derived* `-merger_timescale_alpha` — not with
a literature exponent, and not with zero. This is meaningful because the only
source of the slope is the injected `T_merge(z)`; the pair-count input has no
systematic trend by construction. It verifies the conversion and its error
propagation without claiming the mock data reproduces real merger-rate evolution.

The derivation depends on `N_pairs / box_size_mpc**3` being flat across redshift,
which is a property of the current mock catalog's construction. Genuine
merger-rate evolution or a redshift-varying box size would legitimately give a
different slope. The check must be labelled as recovery of the injected model,
and no runtime mock-versus-real detection should be built.

## Architecture Fit

- New file `src/merger_rate.py`, structured like `plot.py`: reads already-written
  results files and writes its own outputs; it does not re-run pair finding.
- Per existing repo convention (`_mass_bin_edges` is already duplicated
  independently in `pair_finder.py` and `plot.py`), `merger_rate.py` and the
  addition to `calc.py` each define their own local `_mass_bin_edges(config)`
  rather than importing a private helper.
- All tunable scientific parameters live in `config.py` and are passed
  explicitly. Frozen filenames and `check_slope_consistency(..., n_sigma=3.0)`
  may be local constants — API choices, not hidden scientific inputs.
- Units follow the existing convention: positions and box size in Mpc, no new
  conversions beyond what `pair_finder.py` already does. New units (Gyr,
  `Mpc^-3`, `Gyr^-1 Mpc^-3`) are documented inline.

## Validation and Failure Conventions

Binding on every slice.

**Fail loud with `assert` and a clear message naming the offending value.** This
matches the repo's house style (`AGENTS.md`: "use assertions with clear messages
for invalid inputs"). The rule governs **validation newly introduced by this
plan**; it is not a licence to change existing behaviour. In particular
`pair_finder._assign_mass_bins` raises `ValueError` for an unknown `mass_bin_by`
and `tests/test_pair_finder.py` freezes that, so it stays as it is. Do not
introduce `TypeError` / `ValueError` raises for new input validation.

> **Binding note added 2026-08-03 (amendment).** A `TypeError` or `ValueError`
> that *leaks* from an unguarded coercion — `float(attr)` on a vector or a
> `None` — is **not** a valid rejection, even though it fails loudly. It is the
> exact failure this clause and *Validate form before coercion* below exist to
> prevent, so it is a defect wherever the value is one the slice's Acceptance
> Criteria require it to reject. Where the Criteria do **not** list the input
> class, its behaviour stays unspecified and no guard is to be added — the two
> rules do not conflict: this note governs *how* a listed rejection must happen,
> not *which* inputs are rejected.

Missing-file behaviour is slice-specific: Slice 1's `_load_pair_counts` may raise
either `FileNotFoundError` or `AssertionError`; Slice 2's preflight must assert.

**Validate form before coercion.** Where a slice requires a dtype or shape check,
it must run *before* the value is converted: `np.asarray(x, dtype=float)` placed
ahead of a dtype check silently parses a numeric-looking string or discards an
imaginary part, which defeats the check entirely.

**Validate exactly the rejection conditions explicitly listed in each slice's
Acceptance Criteria.** Those rejections are the only specified behaviour outside
the valid input domain; all other out-of-domain behaviour, including intermediate
overflow at extreme magnitudes, is unspecified. Do not add further input
validation, and do not reject any declared-valid value — a guard that rejects an
integer-valued input where the criteria say "non-negative integer counts" is a
defect.

## Execution

Implement Slices 1-3 in plan order, one slice per session. Each slice may assume
earlier slices are accepted but **may not repair them**: stop and report any
prerequisite defect rather than broadening the current slice.

## Test Isolation

- Tests in `tests/test_merger_rate.py` use a copied config whose `data_dir` and
  `results_dir` point beneath pytest's `tmp_path` / `tmp_path_factory`. Tests must
  never read or overwrite the repository's gitignored `data/` or `results/`.
- The generated-mock integration fixture calls `generate_all_snapshots(config)`
  and `run_calculation(config)` directly with those temporary directories. It may
  be module-scoped so the four-snapshot setup is shared. "Generated mock data"
  below always means this isolated fixture, never shelling out to the CLI.

---

## Slice 1: Galaxy-count denominator and pair fraction

### Intended Change

Extend `calc.py` to compute, per redshift, the total number of mass-selected
galaxies per mass bin, and write it as a new dataset `n_galaxies_per_mass_bin`
(1D int array, length `n_mass_bins`) in each `results/pairs_z{z}.hdf5` written by
`_save_pairs`. Add one attr, `box_size_mpc` (float, set to `catalog["box_size"]`
— the value `find_pairs` actually used). **Both additions are purely additive**;
existing datasets (`mass_primary`, `mass_secondary`, `mass_ratio`,
`separation_kpc`, `delta_v`, `mass_bin`, `sep_bin`) and attrs are unchanged.

The count must come from the **full mass-selected catalog**
(`load_galaxy_catalog()` output, already filtered to
`[log_mass_min, log_mass_max]`), not just galaxies appearing in pairs.
`run_calculation()` already loads it per redshift — reuse it. Extract the counting
into a directly testable helper
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
same `np.digitize`-based right-open logic, reimplemented locally per the
duplication convention, so a galaxy with `log_stellar_mass` exactly equal to
`log_mass_max` falls outside all bins and is excluded from every count. This holds
even though `data_reader`'s selection mask is inclusive of `log_mass_max` — a
pre-existing inconsistency this plan does not fix. What matters is that the new
denominator uses the *identical* rule the existing numerator uses.

This slice adds no new `mass_bin_by` branching or assertion. Preserve the existing
provenance read in `_save_pairs` and the existing `find_pairs` behaviour, both of
which legitimately read the key. The `"primary"`-only restriction belongs to
Slice 2's entry point. The new counting, loading, and fraction functions must not
read `mass_bin_by`, and must remain independently testable without it.

Create `src/merger_rate.py` with:

- `_mass_bin_edges(config)` — local copy of the standard formula.
- `_results_path(z, config)` — `results/pairs_z{z:.1f}.hdf5`, matching
  `calc.py` / `plot.py`.
- `_load_pair_counts(z, config)` — reads one results file, returns
  `(n_pairs_per_bin, n_galaxies_per_bin, box_size_mpc)`. `n_pairs_per_bin` counts
  `mass_bin == b` per bin; `n_galaxies_per_bin` is read from the new dataset;
  `box_size_mpc` from the new attr. Both count arrays are 1D int, length
  `n_mass_bins`, in bin-index order.
- `compute_pair_fraction(n_pairs, n_galaxies)` — vectorized over the mass-bin
  array, returns `(f_pair, sigma_f_pair)` per points 2 and 5.

### Acceptance Criteria

- **Inputs:** `config` (existing keys only; this slice is `mass_bin_by`-agnostic);
  on-disk `results/pairs_z{z}.hdf5` as written by the modified `calc.py`.
  Declared input domain for `compute_pair_fraction`: `n_pairs` and `n_galaxies`
  are 1D, identically shaped, finite, non-negative and integer-valued, with
  counts below `2**53`. Behaviour outside that domain is unspecified, and
  implementers need not guard overflow or underflow of intermediate products for
  out-of-domain magnitudes.
- **Outputs:** results files with the new dataset and attr; `merger_rate.py`
  exposing `_mass_bin_edges`, `_results_path`, `_load_pair_counts`,
  `compute_pair_fraction`; `calc.py` exposing `_mass_bin_edges` and
  `_count_galaxies_per_mass_bin` as importable, directly testable functions.
- **User-visible behaviour:** the existing calculation modes produce results
  files with one extra dataset and one extra attr. No change to console output or
  plots.
- **Behaviour that must not change:** `plot.py` is untouched; all
  existing tests in `tests/test_geometric.py`, `tests/test_pair_finder.py`,
  `tests/test_statistical.py` pass unmodified.

- [ ] `n_galaxies_per_mass_bin` and `box_size_mpc` are present in every
      regenerated results file, and every pre-existing dataset and attr is
      unchanged in meaning and value.
- [ ] `box_size_mpc` provably comes from `catalog["box_size"]`, not
      `config["box_size"]` — proven by a test where the two differ.
- [ ] The denominator is drawn from the full selected catalog, not from galaxies
      appearing in pairs: the count sum equals the number of selected catalog
      galaxies satisfying `log_mass_min <= mass < log_mass_max`. A selected galaxy
      sitting exactly at `log_mass_max` is intentionally excluded, so the sum is
      below `data_reader`'s inclusive selected count whenever such a galaxy
      exists.
- [ ] With the default config, `_count_galaxies_per_mass_bin` on
      `[7.9, 8.0, 8.5, 10.999, 11.0, 11.1]` returns exactly `[1, 1, 0, 0, 0, 1]`
      — the exact upper edge is excluded, while an interior-edge galaxy is
      assigned to the upper adjacent bin.
- [ ] `compute_pair_fraction([0, 5, 20], [10, 10, 10])` returns
      `f_pair == [0.0, 0.5, 2.0]` and
      `sigma_f_pair == [0.0, 0.22360679774997896, 0.4472135954999579]`.
- [ ] A bin with `n_pairs == 0` and `n_galaxies == 0` yields `f_pair == 0` and
      `sigma_f_pair == 0` exactly — not `nan`, not `inf`. This must hold on every
      path.
- [ ] `compute_pair_fraction` asserts that every bin with `n_pairs > 0` has
      `n_galaxies > 0`, since a pair cannot exist in a bin with no galaxies.
- [ ] `compute_pair_fraction` rejects, with an assertion naming the reason:
      mismatched shapes, non-1D input, negative counts, non-finite counts, and
      non-integer-valued counts. Integer dtypes and integer-valued float64 are
      both valid input and must be accepted.
- [ ] Against a hand-written HDF5 fixture, `_load_pair_counts` returns the known
      `(n_pairs_per_bin, n_galaxies_per_bin, box_size_mpc)` values in bin-index
      order, excludes `mass_bin == -1` (the existing out-of-range sentinel) from
      `n_pairs_per_bin`, and rejects any other index outside
      `[-1, n_mass_bins - 1]`.
- [ ] `_load_pair_counts` rejects a missing file, a missing required dataset or
      attr, a `n_galaxies_per_mass_bin` whose length is not `n_mass_bins`, and a
      `box_size_mpc` that is not a finite positive scalar.
- [ ] `venv/bin/python -m pytest tests/` passes with 0 failed.

### Authorized Surface

- Files allowed to change:
  - `src/calc.py`
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- Functions/classes/components allowed to change: in `calc.py`, `_save_pairs`
  (frozen expanded signature plus the two additive writes), new `_mass_bin_edges`,
  new `_count_galaxies_per_mass_bin`, and `run_calculation` (call the helper and
  pass its output through). In `merger_rate.py`, the four functions listed above.
- Tests allowed or expected to change: `tests/test_merger_rate.py` only. No
  existing test file should need changes; if one does, stop and report rather
  than editing it.

### Explicit Non-Goals

- No merger timescale, rate conversion, uncertainty-on-rate, or fitting.
- No changes to `pair_finder.py`, `data_reader.py`, `plot.py`,
  `generate_test_data.py`, or `config.py`.
- No change to the meaning or value of any existing dataset or attr.
- No attempt to reconcile `data_reader.py`'s inclusive-upper-edge selection with
  `pair_finder.py`'s exclusive-upper-edge binning.

### Risk Flags

- Risky surfaces touched: the on-disk results schema
  (`results/pairs_z{z}.hdf5`) — additive only; no test in `tests/` opens these
  files directly.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

- Tests to add/update: in `tests/test_merger_rate.py`, give **every** Acceptance
  Criteria item above automated coverage — hand-written arrays and HDF5 fixtures
  for the exact-value and rejection cases, and the isolated generated-mock fixture
  for the integration cases. The checklist is the single source of truth; this
  section deliberately does not restate it.
- Commands to run: `venv/bin/python -m pytest tests/` (0 failed).
- Lint (differential, via the `lint` skill): required.
- Manual checks: none.

### Rollback Path

Revert `src/calc.py`'s new writes and move `src/merger_rate.py` and
`tests/test_merger_rate.py` into `archive/merger-rate-slice1/` rather than
deleting them. Nothing else consumes them.

---

## Slice 2: Merger timescale, rate conversion, and persistence

### Intended Change

Add to `config.py` under a `# Merger rate` section:

- `merger_timescale_gyr0 = 2.2` — `T_merge` normalization at `z = 0`, Gyr
  (order-of-magnitude consistent with Kitzbichler & White 2008; not asserted as
  an exact reproduction).
- `merger_timescale_alpha = -1.0` — power-law index (point 3).
- `merger_fraction = 0.6` — fraction of close pairs that merge within `T_merge`.

Add to `src/merger_rate.py`:

- `merger_timescale_gyr(z, config)` — implements point 3. Validates its own
  inputs rather than trusting a caller: `z` finite and `> -1`,
  `merger_timescale_gyr0` finite and `> 0`, `merger_timescale_alpha` finite, all
  three numeric scalars, and the resulting `T_merge(z)` finite and `> 0`.
- `compute_merger_rate(f_pair, sigma_f_pair, n_galaxies, box_size_mpc, timescale_gyr, merger_fraction)`
  — vectorized over the mass-bin array for one redshift; implements points 4 and 5
  exactly; returns `(rate, sigma_rate)` in `Gyr^-1 Mpc^-3` as float arrays of the
  input shape. Compute
  `sigma_rate = merger_fraction * sigma_f_pair * n_galaxies / (box_size_mpc**3 * timescale_gyr)`,
  which equals `rate / sqrt(N_pairs)` for non-empty bins without needing
  `N_pairs` in the signature, and preserves exact-zero uncertainty when
  `sigma_f_pair == 0`.
- `run_merger_rate_calculation(config)` — asserts
  `config["mass_bin_by"] == "primary"` first, reporting the actual value.

  Then, **before opening the output for writing**, preflight every
  `_results_path(z, config)`: assert each exists, naming the first absent path,
  and assert each file's recorded `redshift` attr equals the configured `z` for
  that path. Validate that attr's dtype and scalar shape before coercion, so a
  malformed string or vector attr is reported rather than crashing on comparison.
  **A failure at this gate must leave any pre-existing `merger_rate.hdf5`
  byte-for-byte untouched.**

  Then for each `z` in `config["redshifts"]`: `_load_pair_counts`,
  `compute_pair_fraction`, `merger_timescale_gyr`, `compute_merger_rate` — passing
  the per-redshift `box_size_mpc` read from that file, **not**
  `config["box_size"]`. Assemble arrays over `(redshift, mass_bin)` and write
  `os.path.join(config["results_dir"], "merger_rate.hdf5")` containing datasets shaped
  `(n_redshifts, n_mass_bins)`: `pair_fraction`, `n_pairs`, `merger_rate`,
  `merger_rate_err`; plus attrs `redshifts`, `mass_bin_by`, `merger_fraction`,
  `merger_timescale_gyr0`, `merger_timescale_alpha`, `timestamp` — mirroring
  `_save_pairs`'s provenance pattern. Create `config["results_dir"]` if needed.

### Acceptance Criteria

- **Inputs:** `config` with the three new keys and `mass_bin_by == "primary"`;
  on-disk `results/pairs_z{z}.hdf5` with Slice 1's additions for every configured
  redshift. Valid inputs have finite positive `box_size_mpc`, `timescale_gyr`, and
  `merger_timescale_gyr0` — matching exactly what Slice 1 guarantees for the
  persisted `box_size_mpc`, so any accepted Slice 1 output is in domain here;
  finite `merger_timescale_alpha`; finite `z > -1`; finite non-negative `f_pair`
  and `sigma_f_pair`; and finite non-negative integer-valued `n_galaxies` below
  `2**53`; and finite `merger_fraction` with `0 < x <= 1`. The rejections listed
  below are the only specified out-of-domain behaviour.
- **Outputs:** `os.path.join(config["results_dir"], "merger_rate.hdf5")` with the schema above;
  `merger_timescale_gyr` and `compute_merger_rate` importable and independently
  callable with no file I/O.
- **User-visible behaviour:** none through the CLI — nothing calls
  `run_merger_rate_calculation` automatically. It is invoked directly.
- **Behaviour that must not change:** everything validated in Slice 1.

- [ ] The three config keys are added and no existing key changes in value or
      meaning.
- [ ] `merger_timescale_gyr(0, config) == config["merger_timescale_gyr0"]`
      exactly.
- [ ] With `merger_timescale_gyr0 = 2.5` and `merger_timescale_alpha = -1.0`
      (values distinct from the defaults, so the test cannot pass by
      coincidence), `merger_timescale_gyr(3, config) == 0.625` exactly.
- [ ] `compute_merger_rate([0.5], [0.1], [10], 500.0, 2.2, 0.6)` returns
      `rate == 1.090909090909091e-08` and
      `sigma_rate == 2.181818181818182e-09`.
- [ ] The same call reproduces the reduced identity of point 4:
      for `N_pairs == 5`: assert `rate[0] == 0.6 * 5 / (500.0**3 * 2.2)` exactly.
- [ ] A bin with `sigma_f_pair == 0` yields `sigma_rate == 0` exactly, on every
      path. This must hold for any in-domain input.
- [ ] A bin with `n_galaxies == 0` and non-zero `f_pair` or `sigma_f_pair` is
      rejected by assertion; `n_galaxies == 0` with both zero is valid.
- [ ] Per-file `box_size_mpc` is provably used rather than `config["box_size"]` —
      proven by a fixture whose `box_size_mpc` attr is `250.0` against
      `config["box_size"] == 500.0`, with `redshifts` covering only that fixtured
      snapshot, where the resulting rate matches the value hand-computed from the
      fixture's `250.0`. An implementation reading `config["box_size"]` must fail
      this.
- [ ] `mass_bin_by != "primary"` fails with an assertion naming the actual value.
- [ ] The preflight gate leaves a sentinel output file byte-for-byte unchanged on
      every failure path — a missing pair file, a mismatched recorded `redshift`,
      and a malformed string or vector `redshift` attr — compared by SHA-256 of
      the file contents, not by its size.
- [ ] The written output contains every dataset and attr in the schema above; the
      2D datasets have shape `(len(redshifts), n_mass_bins)`; `n_pairs` has integer
      dtype; redshift and mass-bin ordering are preserved; and on generated mock
      data `merger_rate` and `merger_rate_err` are finite and non-negative.
- [ ] Docstrings describe the uncertainty as this plan's plug-in Poisson error
      convention and never as "the Poisson uncertainty" unqualified.
- [ ] `merger_timescale_gyr` rejects `z <= -1`, non-finite `z`, non-positive or
      non-finite `merger_timescale_gyr0`, non-finite `merger_timescale_alpha`, and
      string or array input for `z`.
- [ ] `compute_merger_rate` rejects non-finite or non-positive `box_size_mpc` and
      `timescale_gyr`, `merger_fraction` outside `(0, 1]`, mismatched or non-1D
      arrays, negative or non-finite array values, non-integer-valued
      `n_galaxies`, and zero `n_galaxies` paired with nonzero `f_pair` or
      `sigma_f_pair`.
- [ ] `venv/bin/python -m pytest tests/` passes with 0 failed.

### Authorized Surface

- Files allowed to change:
  - `src/config.py`
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- Functions/classes/components allowed to change: in `config.py`, add the three
  keys only. In `merger_rate.py`, add `merger_timescale_gyr`,
  `compute_merger_rate`, and `run_merger_rate_calculation`. Slice 1's functions
  may be called but not modified; if a genuine defect is found there, report it
  rather than patching it in this diff.
- Tests allowed or expected to change: `tests/test_merger_rate.py`, append only.
  Do not remove or weaken Slice 1's tests.

### Explicit Non-Goals

- No fitting and no console summary; those are Slice 3.
- No uncertainty on `T_merge` or `merger_fraction` (point 5).
- No mass- or redshift-dependent `merger_fraction` — one global config value.
- No `mass_bin_by` strategy other than `"primary"`.
- No provenance checks beyond the recorded `redshift`; `mass_ratio_min`,
  `max_sep_kpc`, and bin-edge cross-validation are deliberately out of scope.

### Risk Flags

- Risky surfaces touched: introduces a new persistent output file with its own
  schema — a new persistence contract.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

- Tests to add/update: in `tests/test_merger_rate.py`, give **every** Acceptance
  Criteria item above automated coverage — hand-written arrays and HDF5 fixtures
  for the exact-value and rejection cases, and the isolated generated-mock fixture
  for the integration cases. The checklist is the single source of truth; this
  section deliberately does not restate it.
- Commands to run: `venv/bin/python -m pytest tests/` (0 failed).
- Lint (differential, via the `lint` skill): required.
- Manual checks: none.

### Rollback Path

Revert the three config keys and the Slice 2 additions to `src/merger_rate.py`
and `tests/test_merger_rate.py`. Slice 1 is unaffected and remains functional.

---

## Slice 3: Weighted redshift-evolution fit and validation

### Intended Change

Add to `src/merger_rate.py`:

- `fit_log_rate_vs_redshift(rates, rate_errs, redshifts)` — operates on a
  **single mass bin's** 1D array-likes, one value per configured redshift, all
  three the same shape. Assert that every input is 1D with identical shapes; do
  not flatten, broadcast, or accept higher rank. Fits `log10(rate)` vs
  `log10(1 + z)` by weighted least squares, weights `1 / sigma_log_rate**2`, with
  `sigma_log_rate = sigma_rate / (rate * ln(10))`.

  **The malformed-versus-data distinction is frozen:**

  - Any `redshift` that is `<= -1` or non-finite is **malformed**, because
    `log10(1+z)` is undefined or infinite, and fails immediately.
  - A point is **usable** only if `rate` is finite and `> 0` *and* `rate_err` is
    finite and `> 0`. Points failing either are **excluded, not an error** — this
    is the normal empty-bin case from point 5 — and counted in `n_excluded`.
  - After filtering, if fewer than 2 usable points remain, **or** the usable
    points do not span at least 2 distinct redshifts, **return
    `(nan, nan, nan, n_excluded)`** rather than failing or fabricating a fit. This
    is a data condition, not malformed input, so it must not raise.

  > **Binding note added 2026-08-03 (amendment).** "Distinct redshifts" above
  > means **distinct in the formed predictor**, not merely distinct as float64
  > `z`. Two redshifts can differ and still produce the same `log10(1 + z)` — for
  > example `z = 1.0` and `np.nextafter(1.0, 2.0)`, whose `1 + z` both round to
  > `2.0`. Apply the distinctness test to the usable points' computed
  > `log10(1 + z)` values: if fewer than 2 of them are distinct, **return
  > `(nan, nan, nan, n_excluded)`**. It must not raise, and it must not fabricate
  > a finite slope from rounding residue.
  >
  > Test distinctness on the formed predictor values themselves, before any
  > centring — comparing them is exact, whereas a centred residual is not
  > reliably zero for identical inputs, since a weighted mean of a constant array
  > need not round back to that constant. No minimum separation threshold is
  > specified and none is to be invented: two distinct predictor values, however
  > close, are a fit; one value is this branch.

  `slope_err` must be the **unscaled** weighted-least-squares parameter error,
  computed from the supplied `sigma_log_rate` as true measurement errors. Do not
  use residual-based rescaling (`scipy.optimize.curve_fit` without
  `absolute_sigma=True`, or `numpy.polyfit(..., cov=True)`'s default), which
  silently substitutes a data-driven estimate for the one this plan propagated.
  Use explicit weighted normal equations or `cov="unscaled"`. **The
  exactly-two-usable-points case must return a finite fit** —
  `numpy.polyfit(..., cov=True)` raises there, as it attempts residual scaling
  with zero residual degrees of freedom.

  Implement the normal equations in a numerically stable form for the declared
  domain: normalize the weights and center the predictor about its weighted mean
  before accumulating, then restore the absolute covariance scale. Returns
  `(slope, slope_err, intercept, n_excluded)`. **This function has no knowledge
  of mass bins**; the caller loops over them.

  > **Binding note added 2026-08-03 (amendment).** Centre **both** variables, not
  > just the predictor: accumulate the cross term as
  > `sum(w * x_centred * (y - y_weighted_mean))`. This is algebraically identical
  > to using an uncentred `y`, but the two differ in float64 by a term
  > proportional to `y`'s absolute offset, so an uncentred `y` amplifies the
  > predictor's rounding residue by that offset and biases the slope of a
  > barely-resolved predictor. The collapsed case is already handled by the
  > branch above; this is the adjacent numerical-hygiene requirement, and it is
  > what "numerically stable form" means here.

- `check_slope_consistency(slope, slope_err, expected_slope, n_sigma=3.0)` —
  returns `True` if `abs(slope - expected_slope) < n_sigma * slope_err`, else
  `False`. Returns `False`, not an error, if `slope`, `slope_err`, or
  `expected_slope` is non-finite or if `slope_err <= 0`; the insufficient-data
  `nan` case is "check not applicable" and the caller must report it as such,
  never as a pass or fail. Assert `n_sigma` is finite and `> 0` so an invalid
  threshold cannot fabricate a pass. `expected_slope` is always
  `-config["merger_timescale_alpha"]` in use, but the function takes it as a
  parameter and hardcodes nothing. Pure, no I/O.

- `run_merger_rate_validation(config)` — reads
  `os.path.join(config["results_dir"], "merger_rate.hdf5")`, and for each mass bin calls
  `fit_log_rate_vs_redshift` on that bin's `merger_rate` and `merger_rate_err`
  across the stored `redshifts`, then `check_slope_consistency` against
  `expected_slope = -config["merger_timescale_alpha"]`. Validate the stored
  `redshifts` up front: any non-finite or `<= -1` value fails before any fitting.
  Prints one line per mass bin reporting the bin's mass range, fitted slope, its
  uncertainty, `expected_slope`, `n_excluded`, and whether the check passed — or
  `insufficient data` when `slope` is `nan`. The printed heading must state that
  this verifies recovery of the injected merger-timescale model on mock data, not
  a measurement of real merger-rate evolution. Returns a list of per-bin result
  dicts so callers can assert on it without parsing stdout.

### Acceptance Criteria

- **Inputs:** for the two pure functions, in-memory 1D array-likes only, with no
  file I/O. `redshifts` must be finite and `> -1`, and `n_sigma` finite and `> 0`.
  `rates` and `rate_errs` **may** contain non-finite or non-positive values: those
  points are in-domain excluded data, not malformed input, and a point is usable
  only when both its `rate` and its `rate_err` are finite and strictly positive.
  Other out-of-domain behaviour, including intermediate overflow at extreme
  magnitudes, is unspecified. `run_merger_rate_validation` additionally reads
  `os.path.join(config["results_dir"], "merger_rate.hdf5")` as written by Slice 2.

  > **Binding note added 2026-08-03 (amendment).** Two finite redshifts `> -1`
  > whose separation is too small to survive `log10(1 + z)` are **in** the
  > declared domain, and the collapsed-predictor rule in the Intended Change
  > governs them: return `(nan, nan, nan, n_excluded)`. No exact slope is
  > specified for such inputs and none is required — a fit through a collapsed
  > predictor is not meaningful, so there is nothing here for a reviewer to mine
  > as an unmet exactness criterion. The "finite fit" requirement for the
  > exactly-two-usable-points case applies whenever the two formed predictor
  > values are distinct, which covers every pinned fixture in this slice.
- **Outputs:** `fit_log_rate_vs_redshift`, `check_slope_consistency`, and
  `run_merger_rate_validation` importable and independently callable; the console
  summary; and a returned list of per-bin result dicts, each having exactly the
  keys `mass_bin`, `slope`, `slope_err`, `intercept`, `expected_slope`,
  `n_excluded`, and `consistent`, where `consistent` is `None` for the
  insufficient-data case and otherwise a bool.
- **User-visible behaviour:** `run_merger_rate_validation(config)` prints the
  labelled per-mass-bin summary. Nothing is reached through the CLI.
- **Behaviour that must not change:** everything validated in Slices 1-2.

- [ ] An exact power law is recovered to small numerical tolerance: with
      `redshifts = [1, 3, 7]`, `rates = [2, 4, 8]` and tiny equal `rate_errs`,
      the fitted slope is `1.0` to within `1e-9`.
- [ ] The fit is **provably weighted**, demonstrated by a heteroscedastic case
      rather than merely by fitting a straight line: with 4 or more points on an
      exact power law except one perturbed far off the line but given a very large
      `rate_err`, the weighted slope is measurably closer to the true exponent
      than an unweighted `numpy.polyfit` computed in the test itself.
- [ ] `slope_err` matches an independently hand-computed weighted
      normal-equations variance, not a value from the same library call the
      implementation uses. For `redshifts = [1, 3]`, `rates = [2, 4]`,
      `rate_errs = [0.2, 0.4]`, the result is `slope == 1.0` and
      `slope_err == 0.2040278893193579` to within `1e-12`.
- [ ] That exactly-two-usable-points case returns a **finite** `slope`,
      `slope_err`, and `intercept`, with the correct `n_excluded`. A
      residual-rescaled implementation fails this.
- [ ] Fewer than 2 usable points returns `(nan, nan, nan, n_excluded)` with the
      correct count and does not raise, mixing `rate <= 0`, non-finite `rate`,
      and non-finite or non-positive `rate_err`.
- [ ] Two or more usable points all sharing a single redshift returns
      `(nan, nan, nan, n_excluded)` without raising, distinguishing it from the
      malformed cases below.
- [ ] Malformed `redshifts` (`<= -1`, `nan`, `inf`) and shape or rank violations
      fail with an assertion.
- [ ] `n_excluded` is correct in every case above.
- [ ] `check_slope_consistency` returns `True` within range, `False` far outside,
      and `False` for non-finite `slope`, `slope_err`, or `expected_slope` and for
      `slope_err <= 0`; and asserts on non-finite or non-positive `n_sigma`.
- [ ] `run_merger_rate_validation` fails on a malformed stored redshift before
      performing any fit.
- [ ] `run_merger_rate_validation` prints `insufficient data` for a `nan`-slope
      bin, reports `n_excluded` per bin, and its heading states that the check
      verifies recovery of the injected timescale model on mock data rather than
      real merger-rate evolution.
- [ ] **End-to-end on generated mock data:** after `run_calculation` and
      `run_merger_rate_calculation`, `check_slope_consistency` is `True` for every
      mass bin with at least 2 usable redshift points, against
      `expected_slope = -config["merger_timescale_alpha"]`. This is the
      scientific assertion of the plan and must be allowed to fail loudly if the
      rate calculation or its error propagation is wrong.
- [ ] `venv/bin/python -m pytest tests/` passes with 0 failed.

### Authorized Surface

- Files allowed to change:
  - `src/merger_rate.py`
  - `tests/test_merger_rate.py`
- Functions/classes/components allowed to change: in `merger_rate.py`, add
  `fit_log_rate_vs_redshift`, `check_slope_consistency`, and
  `run_merger_rate_validation` only. Earlier slices' functions may be called but
  not modified.
- Tests allowed or expected to change: `tests/test_merger_rate.py`, append only.

### Explicit Non-Goals

- No Matplotlib import anywhere in this slice.
- No hardcoded literature exponent as ground truth — `expected_slope` is always
  derived from `config["merger_timescale_alpha"]`.
- No modification to `generate_test_data.py` to manufacture a redshift trend
  (point 6 — a hard boundary).
- No changes to `src/config.py`.

### Risk Flags

- Risky surfaces touched: none — two pure functions plus one read-only consumer
  of an existing results file.
- Approval needed before implementation: no
- Independent audit required: yes

### Validation Plan

- Tests to add/update: in `tests/test_merger_rate.py`, give **every** Acceptance
  Criteria item above automated coverage — hand-written arrays and HDF5 fixtures
  for the exact-value and rejection cases, and the isolated generated-mock fixture
  for the integration cases. The checklist is the single source of truth; this
  section deliberately does not restate it.
- Commands to run: `venv/bin/python -m pytest tests/` (0 failed).
- Lint (differential, via the `lint` skill): required.
- Manual checks: none.

### Rollback Path

Revert the Slice 3 additions to `src/merger_rate.py` and
`tests/test_merger_rate.py`. Slices 1-2 remain functional; their functions are
simply no longer exercised by a fit.

---

## Next Chat Prompt

Plan file: `/Users/dcroton/Local/git-repos/relative-velocity/docs/MERGER_RATE_PLAN-REVISED.md`

Implement Slices 1-3 in plan order, one slice per session, against the frozen
contract in each slice's own sections. Each slice's Acceptance Criteria checklist
is the completeness gate: reproduce it with each item marked and the evidence
that settles it. Run the slice's validation commands and the differential `lint`
skill before committing, and commit only that slice's work. All three slices
require an independent `drift-audit` and `code-review` of the final diff before
acceptance.
