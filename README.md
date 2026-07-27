# relative-velocity

Compute relative velocity statistics for close galaxy pairs from semi-analytic model outputs.

## What it does

Given a galaxy catalog with positions, velocities, and stellar masses at one or more redshift snapshots, this pipeline:

1. Finds all galaxy pairs within a configurable 3D separation threshold (default 25 kpc)
2. Computes 3D relative velocities for each pair
3. Bins pairs by stellar mass, separation, and redshift
4. Writes pair catalogs to disk
5. Produces histograms of the relative velocity distribution in each bin

## Setup

```bash
./setup.sh
source venv/bin/activate
```

`setup.sh` checks the Python version, creates a `venv/` virtual environment, and installs `requirements.txt` into it. Re-running it is safe — it reuses an existing `venv/`.

Run all commands below from the repo root (paths like `data/`, `results/`, `figures/` are relative to the working directory, not to `src/`).

## Quick start

```bash
# Generate test data, run calculation, plot with Maxwell-distribution validation overlays
python src/pipeline.py --validate

# Generate test data then run the full pipeline
python src/pipeline.py --generate-test

# Run calculation only (data/ must exist)
python src/pipeline.py --calc-only

# Remake plots from existing results (no recalculation)
python src/pipeline.py --plot-only
```

## Configuration

All parameters live in `src/config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `box_size` | 500.0 | Simulation box size (Mpc) |
| `redshifts` | [2, 3, 4, 5] | Redshift snapshots to process |
| `log_mass_min/max` | 8.0 / 11.0 | Stellar mass range (log10 M_sun) |
| `mass_bin_width` | 0.5 | Mass bin width (dex) |
| `sep_bins` | [0, 10, 15, 20, 25] | Separation bin edges (kpc) |
| `mass_ratio_min` | 0.1 | Minimum mass ratio for pairs (1:10) |
| `mass_bin_by` | "primary" | Which galaxy's mass defines the bin |

## Units

The pipeline takes all input values at face value:

- **Positions**: Mpc
- **Velocities**: km/s
- **Stellar mass**: log10(M_star / M_sun)

No cosmological unit conversions are applied. If your data uses different conventions (e.g. Mpc/h, comoving coordinates), handle the conversion in the data reader.

## Tests

```bash
pytest tests/
```

80 tests covering pair-count geometry (Poisson formula, r³/N² scaling, periodic boundary), exact unit recovery (separation, velocity, mass/separation bin assignment), and statistical validation against analytical Maxwell-distribution predictions.

## Requirements

- Python 3.8+
- numpy
- scipy
- h5py
- matplotlib
- pytest

See `requirements.txt` for pinned minimum versions.

## Project layout

- `src/` — pipeline modules (see `docs/PLAN.md` for the architecture and data flow)
- `tests/` — pytest suite
- `docs/` — planning and scientific background documents
- `data/`, `results/`, `figures/` — generated inputs/outputs (gitignored)
