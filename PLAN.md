# Implementation Plan: Relative Velocity Statistics Pipeline

## Design Principles

### Architecture
1. **Modular**: each piece (data I/O, pair finding, calculation, plotting) is a separate module that can be swapped or extended independently.
2. **Separation of concerns**: calculation writes results to disk; plotting reads from disk. No coupling between them except the file format.
3. **Configurable**: all tuneable parameters (mass bins, separation bins, mass ratio cuts, mass-bin assignment strategy, etc.) live in a single config and are passed explicitly. Changing a parameter should never require editing core logic.
4. **Staged development**: start with the simplest version that works end-to-end, then layer in complexity.

### Coding Standards
5. **KISS (Keep It Simple, Stupid)**: prefer the simplest solution that works. No premature abstraction, no class hierarchies where a function will do, no cleverness for its own sake. If a loop is clearer than a one-liner, use the loop.
6. **DRY (Don't Repeat Yourself)**: any logic that appears in more than one place gets extracted into a shared function. Bin-edge computation, file path construction, mass-bin assignment -- each defined once, used everywhere.
7. **Flat is better than nested**: avoid deep nesting. Use early returns, guard clauses, and helper functions to keep indentation shallow.
8. **Explicit over implicit**: no global state. Every function receives what it needs as arguments and returns its results. No side effects beyond file I/O in the designated places.
9. **Fail loud**: use assertions and clear error messages for invalid inputs (e.g., unknown mass_bin_by strategy, empty pair catalog). Don't silently produce wrong results.
10. **Comments explain *why*, not *what***: the code should be readable on its own. Comments are for physics context, non-obvious choices, and unit assumptions -- not restating what the code does.

### Units: Keep It Simple
11. **Take values at face value**: positions are in Mpc, velocities in km/s, stellar masses in log10(M_sun). No cosmological unit conversions (h-factors, scale factors, comoving-to-proper) in the code. The user will handle any necessary conversions before the data enters the pipeline, or adjust the config (e.g., separation thresholds) to match the coordinate system of the input data. This keeps the code clean, general, and free of cosmology-specific bugs.


## File Structure

```
relative-velocity/
├── BACKGROUND.md                # physics and methodology reference
├── PLAN.md                      # this document
│
├── config.py                    # all parameters and settings
├── generate_test_data.py        # standalone: creates mock galaxy catalogs
├── data_reader.py               # modular data I/O (test data now, SAGE later)
├── pair_finder.py               # KD-tree pair finding + relative velocity calc
├── calc.py                      # orchestrates calculation, writes results
├── plot.py                      # reads results, makes histograms
├── pipeline.py                  # runs calc then plot (or either alone)
│
├── data/                        # input galaxy catalogs (generated or real)
│   ├── test_z2.0.hdf5
│   ├── test_z3.0.hdf5
│   ├── test_z4.0.hdf5
│   └── test_z5.0.hdf5
│
├── results/                     # calculation output (pair catalogs)
│   ├── pairs_z2.0.hdf5
│   ├── pairs_z3.0.hdf5
│   ├── pairs_z4.0.hdf5
│   └── pairs_z5.0.hdf5
│
└── figures/                     # output plots
    ├── vrel_histograms_by_mass.png
    ├── vrel_histograms_by_sep.png
    └── ...
```


## Configuration (`config.py`)

A single Python dict (or dataclass) containing all tuneable parameters:

```python
config = dict(
    # Simulation
    box_size       = 500.0,       # Mpc (same units as input positions)
    redshifts      = [2.0, 3.0, 4.0, 5.0],

    # Galaxy selection
    log_mass_min     = 8.0,       # log10(M_star / M_sun)
    log_mass_max     = 11.0,      # log10(M_star / M_sun)
    mass_ratio_min   = 0.1,       # minimum M_secondary/M_primary (1:10 cut)

    # Binning
    mass_bin_width   = 0.5,       # dex
    sep_bins         = [0, 10, 15, 20, 25],  # kpc, separation bin edges
    vel_bin_width    = 20.0,      # km/s, histogram bin width
    vel_max          = 1000.0,    # km/s, upper limit for histograms

    # Pair finding
    max_sep          = 25.0,      # kpc, max 3D separation for pairs

    # Mass bin assignment: which galaxy's mass defines the bin?
    # Options: "primary", "secondary", "mean", "total"
    mass_bin_by      = "primary",

    # File paths
    data_dir    = "data/",
    results_dir = "results/",
    figures_dir = "figures/",
)
```

All downstream code receives this config dict as an argument. No hardcoded magic numbers in any module. Stellar masses are stored and compared in log10(M_sun) throughout.


## Stage 1: Test Data Generator (`generate_test_data.py`)

### Purpose
Create mock galaxy catalogs at z = 2, 3, 4, 5 with **known, analytically predictable** relative velocity distributions, so we can validate every step of the pipeline.

### Strategy: Explicit Pair Construction

Rather than scattering galaxies randomly and hoping pairs form, we **construct pairs directly** with controlled properties.

For each redshift snapshot, generate:
- **~3000 galaxy pairs** (6000 paired galaxies) spread across mass and separation bins
- **~2000 isolated field galaxies** placed far from all pairs (to test that pair-finder ignores them)
- Total: ~8000 galaxies per snapshot

#### Per-pair generation procedure:

1. **Draw primary stellar mass** from a log-uniform distribution over [10^8, 10^11] M_sun, with weighting toward lower masses to mimic a realistic mass function. Target: ~500+ pairs per mass bin at minimum.

2. **Draw mass ratio** uniformly from [0.1, 1.0]. Secondary mass = primary mass x ratio.

3. **Place primary** at a random position in the box.

4. **Draw 3D separation** uniformly from [3, 28] kpc (slightly wider than our bins to test edge handling). Note: since 28 kpc = 0.028 Mpc, convert to Mpc to match position units. Place secondary at this offset from primary (random direction on the unit sphere).

5. **Assign primary a random velocity**: each component drawn from N(0, sigma_bulk) where sigma_bulk = 200 km/s.

6. **Draw relative velocity difference** from a **known, mass-dependent distribution**:
   - Each component (dvx, dvy, dvz) drawn independently from N(0, sigma_v(M))
   - sigma_v depends on the primary's mass bin:

   | Mass bin (log M) | sigma_v per component (km/s) | Expected mean speed (km/s) |
   |------------------|------------------------------|---------------------------|
   | 8.0 - 8.5       | 40                           | 64                        |
   | 8.5 - 9.0       | 60                           | 96                        |
   | 9.0 - 9.5       | 80                           | 128                       |
   | 9.5 - 10.0      | 110                          | 175                       |
   | 10.0 - 10.5     | 150                          | 239                       |
   | 10.5 - 11.0     | 200                          | 319                       |

   The 3D speed |dv| = sqrt(dvx^2 + dvy^2 + dvz^2) follows a **Maxwell distribution** with scale parameter sigma_v:
   ```
   f(v) = sqrt(2/pi) * v^2 / sigma^3 * exp(-v^2 / (2*sigma^2))
   ```
   - Mean: sigma * sqrt(8/pi) ~ 1.596 * sigma
   - Mode: sigma * sqrt(2) ~ 1.414 * sigma
   - Std dev: sigma * sqrt(3 - 8/pi) ~ 0.655 * sigma

   These are the **analytical predictions** we validate against.

7. **Secondary velocity** = primary velocity + (dvx, dvy, dvz).

#### Field galaxies:
- Random positions, guaranteed > 100 kpc (0.1 Mpc) from any paired galaxy.
- Random velocities from N(0, 200) km/s per component.
- Random masses in log10(M) = [8, 11].
- These should produce **zero** contamination in the close-pair results.

### Output format

HDF5 file per redshift with datasets:
```
/x, /y, /z          # positions, Mpc
/vx, /vy, /vz       # velocities, km/s
/log_stellar_mass    # log10(M_star / M_sun)
/is_paired           # bool flag (for validation only; not used by pipeline)
/pair_id             # int pair identifier (-1 for field galaxies)
```

Plus attributes: `redshift`, `box_size`, `sigma_v_per_bin` (the input sigmas, for validation).

### Validation predictions

After running the pipeline on this test data, we expect:
- Histogram shapes match Maxwell distributions with the input sigma_v values
- Mean speeds match 1.596 * sigma_v per mass bin
- No field galaxies appear in any pair (pair count = generated pair count exactly, minus any that fall outside separation bins)
- Pair counts per separation bin match the input generation fractions
- Mass bin assignment matches expectations


## Stage 2: Data Reader (`data_reader.py`)

A module with a simple interface:

```python
def load_galaxy_catalog(filepath, config):
    """
    Returns a dict with keys:
        'x', 'y', 'z'         : positions, Mpc        (1D arrays)
        'vx', 'vy', 'vz'      : velocities, km/s      (1D arrays)
        'log_stellar_mass'     : log10(M_star/M_sun)   (1D array)
        'redshift'             : float
        'box_size'             : float, Mpc
    """
```

**Version 1 (test data)**: reads the HDF5 files from `generate_test_data.py`.

**Version 2 (SAGE)**: a second function `load_sage_catalog(filepath, config)` that reads SAGE HDF5 output, extracts the same properties, and applies any unit conversions needed (h-factors, comoving-to-proper, etc.) **inside the reader** so the rest of the pipeline sees clean Mpc / km/s / log10(M_sun). Same return signature. Swapped by changing one line in config or calc.py.

This is the **only** module that needs to change when switching data sources. All unit/convention differences are absorbed here.


## Stage 3: Pair Finder (`pair_finder.py`)

The computational core. Given a galaxy catalog dict, returns all pairs with their properties.

### Interface

```python
def find_pairs(catalog, config):
    """
    Find all galaxy pairs within max_sep (kpc) and compute
    their relative velocities.

    Returns a dict of arrays with one entry per pair:
        'mass_primary'          : log10(M/M_sun), more massive galaxy
        'mass_secondary'        : log10(M/M_sun)
        'mass_ratio'            : 10^(log_secondary - log_primary), always <= 1
        'separation_kpc'        : 3D separation in kpc
        'delta_v'               : 3D relative speed in km/s
        'mass_bin'              : int, which mass bin (based on config["mass_bin_by"])
        'sep_bin'               : int, which separation bin
    """
```

### Implementation

1. Convert positions from Mpc to kpc (* 1e3) so separations and the search radius are in kpc directly.
2. Build `scipy.spatial.cKDTree` with `boxsize = box_size * 1e3` (kpc, periodic).
3. `tree.query_pairs(r=max_sep)` — returns all pairs within `max_sep` kpc. Simple: positions in kpc, radius in kpc.
4. For each pair, compute separation and relative velocity, apply mass ratio cut.
5. Assign mass bin and separation bin.


## Stage 4: Calculation Driver (`calc.py`)

Orchestrates the full calculation for all redshifts.

```python
def run_calculation(config):
    for z in config["redshifts"]:
        catalog = load_galaxy_catalog(data_path_for(z), config)
        pairs = find_pairs(catalog, config)
        save_pair_results(pairs, results_path_for(z), config)
```

### Results file format

HDF5 per redshift. Stores the **raw pair catalog** (not pre-binned histograms) for maximum flexibility:

```
/delta_v          # relative speed, km/s
/separation_kpc   # 3D separation, kpc
/mass_primary     # log10(M/M_sun), more massive galaxy
/mass_secondary   # log10(M/M_sun)
/mass_ratio       # M_secondary / M_primary (linear, <= 1)
/mass_bin         # assigned mass bin index
/sep_bin          # assigned separation bin index
```

Attributes: redshift, config parameters used, pair count, timestamp.

Storing raw pair data means the plotting code can re-bin velocities, change histogram bin widths, add new statistics, etc. without re-running the calculation.


## Stage 5: Plotting (`plot.py`)

Reads the results files and produces publication-quality histograms.

### Primary figure: velocity distributions by mass (one figure per separation bin)

Layout: 2x3 grid of panels (6 mass bins). Each panel shows 4 overlaid histograms (one per redshift, different colors). X-axis: |Delta_v| (km/s). Y-axis: probability density (normalized histograms).

For the test data, **overplot the analytical Maxwell distribution** with the known input sigma_v to visually validate.

### Secondary figure: velocity distributions by separation

Layout: 2x2 grid (4 separation bins). Each panel shows overlaid histograms for different mass bins at a single redshift. One figure per redshift, or all on one large grid.

### Summary statistics table

Print/save a table of: mass_bin, sep_bin, redshift, N_pairs, mean_v, median_v, std_v.

### Interface

```python
def make_plots(config, validation_mode=False):
    """
    validation_mode=True overlays analytical Maxwell predictions (for test data).
    """
```


## Stage 6: Pipeline (`pipeline.py`)

Entry point that ties everything together.

```python
# Usage:
#   python pipeline.py                  # run full pipeline (calc + plot)
#   python pipeline.py --plot-only      # skip calculation, just remake figures
#   python pipeline.py --calc-only      # run calculation, skip plotting
#   python pipeline.py --generate-test  # generate test data first, then run all
#   python pipeline.py --validate       # generate test data + run + plot with Maxwell overlays
```

Implemented with argparse. Simple, linear flow:

```
[generate test data] -> [calculate pair statistics] -> [plot results]
     (optional)              (writes to disk)         (reads from disk)
```


## Implementation Order

### Phase 1: Scaffolding
- `config.py` with all default parameters
- Directory structure (data/, results/, figures/)
- `pipeline.py` with argparse skeleton

### Phase 2: Test Data
- `generate_test_data.py` — full implementation
- Run it, inspect outputs, confirm pair properties are as intended

### Phase 3: Core Calculation
- `data_reader.py` — test data reader
- `pair_finder.py` — KD-tree pair finding and velocity calculation
- `calc.py` — driver that calls reader + pair finder + writes results
- Run on test data, inspect results files

### Phase 4: Plotting
- `plot.py` — histograms with Maxwell overlay in validation mode
- Visually confirm recovery of input distributions

### Phase 5: Integration & Validation
- `pipeline.py` — wire everything together
- Run `--validate` end-to-end
- Quantitative checks: mean speed within ~2% of analytical prediction; pair counts match; no field galaxy contamination

### Phase 6: SAGE Integration (future)
- Add `load_sage_catalog()` to `data_reader.py`
- Update config with SAGE file paths and any unit conversions
- Run on real data


## Unit Summary

All values are taken at face value. No cosmological conversions (h-factors, scale factors, comoving-to-proper) are applied in the pipeline code. If the input data needs such conversions, they are done **once** in the data reader before anything enters the pipeline.

| Quantity          | Unit              | Notes                                     |
|-------------------|-------------------|-------------------------------------------|
| Positions         | Mpc (input), kpc (internal) | Multiplied by 1e3 at start of pair finder |
| Velocities        | km/s              | Direct from input                         |
| Stellar mass      | log10(M/M_sun)    | Direct from input                         |
| Separations       | kpc               | From position differences                 |
| Relative velocity | km/s              | \|v_i - v_j\|                             |
| Box size          | Mpc (config), kpc (internal) | Same 1e3 conversion as positions   |

The only unit conversion in the code is **Mpc -> kpc** (multiply by 1000) so that separations and search radii are in kpc directly. Everything else passes through unchanged.


## Validation Checklist

After running the full pipeline on test data:

- [ ] Pair counts per (mass_bin, sep_bin) match expectations from input generation
- [ ] Zero field galaxies appear in pair catalog
- [ ] Mean relative speed per mass bin matches analytical Maxwell prediction (1.596 * sigma_v) to within statistical noise (~2-5% for ~500 pairs)
- [ ] Histogram shapes visually match Maxwell PDFs
- [ ] Higher mass bins show higher velocities (monotonic trend)
- [ ] Separation bins show roughly similar velocity distributions (since test data velocities are independent of separation)
- [ ] All 4 redshifts produce consistent results (since test data uses the same sigma_v at all z)
- [ ] `--plot-only` reproduces identical figures without re-running calculation
- [ ] No pairs are double-counted
- [ ] Periodic boundary pairs are handled correctly (test data places some pairs near box edges)
