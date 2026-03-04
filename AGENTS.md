# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python pipeline for computing relative velocity statistics of close galaxy pairs from semi-analytic model (SAM) outputs. **Currently in planning phase** — documentation is complete but no source code exists yet.

## Running the Pipeline

```bash
# Generate test data and run full pipeline with Maxwell-distribution validation plots
python pipeline.py --validate

# Run calculation only (writes pair catalogs to results/)
python pipeline.py --calc-only

# Remake plots from existing results (no recalculation)
python pipeline.py --plot-only

# Generate test data, then run full pipeline
python pipeline.py --generate-test
```

Requirements: Python 3.8+, numpy, scipy, h5py, matplotlib.

## Architecture

Six modules with a strictly linear data flow; no module imports from another except through the pipeline driver.

```
generate_test_data.py  →  data/ HDF5 files
                              ↓
data_reader.py  →  load_galaxy_catalog()  →  dict of arrays
                              ↓
pair_finder.py  →  find_pairs()  →  pair catalog dict
                              ↓
calc.py  →  run_calculation()  →  results/ HDF5 files
                              ↓
plot.py  →  make_plots()  →  figures/ PNG files
```

**`config.py`** — single dict containing all tunable parameters; passed explicitly to every function. No hardcoded values in any module.

**`data_reader.py`** — the only module that changes when switching data sources. Absorbs all unit conversions (h-factors, comoving→proper, etc.) so everything downstream sees clean Mpc / km/s / log10(M_sun). Interface: `load_galaxy_catalog(filepath, config)` returns `{x, y, z, vx, vy, vz, log_stellar_mass, redshift, box_size}`.

**`pair_finder.py`** — computational core. Uses `scipy.spatial.cKDTree` with `boxsize` set for periodic boundary conditions. Converts positions Mpc→kpc (×1000) internally; this is the only unit conversion in the pipeline code itself. Applies mass ratio cut (`mass_ratio_min`), assigns each pair to a mass bin and separation bin.

**`calc.py`** — loops over redshifts, calls reader + pair finder, writes raw pair arrays (not pre-binned histograms) to HDF5. Storing raw pairs means `plot.py` can re-bin without re-running the calculation.

**`plot.py`** — reads results HDF5, produces 2×3 (mass bins) and 2×2 (separation bins) histogram grids. In `validation_mode=True`, overlays analytical Maxwell PDFs using the `sigma_v_per_bin` attributes stored in the test data files.

## Key Design Decisions

- **Mass bin assignment strategy** (`mass_bin_by = "primary"`): pairs are binned by the more massive galaxy's mass. Other options in config: `"secondary"`, `"mean"`, `"total"`.
- **Raw pair storage**: results files store one row per pair, not pre-binned counts, for downstream flexibility.
- **No global state**: every function receives config as an argument; no module-level configuration.
- **Fail loud**: use assertions with clear messages for invalid inputs (unknown `mass_bin_by`, empty pair catalog, etc.).
- **Comments explain why, not what**: physics context, unit assumptions, non-obvious choices — not restatements of the code.

## Units

All values taken at face value throughout the pipeline. The **only** unit conversion in pipeline code is positions Mpc→kpc at the top of `pair_finder.py`.

| Quantity | Unit |
|----------|------|
| Positions (input) | Mpc |
| Positions (internal, pair_finder) | kpc (×1000) |
| Velocities | km/s |
| Stellar mass | log10(M_star / M_sun) |
| Separations | kpc |
| Relative velocity | km/s — magnitude `|v_i − v_j|` |
| Box size (config) | Mpc; converted to kpc inside pair_finder |

If input data uses Mpc/h, comoving coordinates, or other conventions, handle conversions inside `data_reader.py` before returning.

## Test Data Strategy

`generate_test_data.py` constructs pairs explicitly with **analytically known** relative velocity distributions. Each component of the velocity difference is drawn from `N(0, sigma_v(M))` making the 3D speed `|dv|` follow a Maxwell distribution. The `sigma_v` values are stored as HDF5 attributes so `plot.py` can overlay the exact predicted distribution for validation.

Expected Maxwell mean speed: `1.596 × sigma_v` per mass bin. Validation target: pipeline recovers this to within ~2–5% for ~500 pairs per bin.

## Implementation Order

1. `config.py` + directory scaffold + `pipeline.py` argparse skeleton
2. `generate_test_data.py`
3. `data_reader.py` + `pair_finder.py` + `calc.py`
4. `plot.py` with Maxwell overlay in validation mode
5. End-to-end `--validate` run; quantitative checks against analytical predictions
6. (Future) `load_sage_catalog()` in `data_reader.py` for real SAGE data

## Configuration Reference

All parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `box_size` | 500.0 | Simulation box size, Mpc |
| `redshifts` | [2, 3, 4, 5] | Snapshots to process |
| `log_mass_min/max` | 8.0 / 11.0 | Stellar mass range, log10 M_sun |
| `mass_bin_width` | 0.5 | Mass bin width, dex → 6 bins |
| `sep_bins` | [0, 10, 15, 20, 25] | Separation bin edges, kpc → 4 bins |
| `mass_ratio_min` | 0.1 | Min M_secondary/M_primary (1:10 cut) |
| `mass_bin_by` | `"primary"` | Which galaxy's mass defines the bin |
| `max_sep` | 25.0 | Max 3D pair separation, kpc |
| `vel_bin_width` | 20.0 | Histogram bin width, km/s |
| `vel_max` | 1000.0 | Upper limit for velocity histograms, km/s |
| `data_dir` | `"data/"` | Input catalog directory |
| `results_dir` | `"results/"` | Output pair catalog directory |
| `figures_dir` | `"figures/"` | Output plots directory |
