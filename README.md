# relative-velocity

Compute relative velocity statistics for close galaxy pairs from semi-analytic model outputs.

> **Work in progress** — core pipeline is under active development. Results should be validated before use in publications.

## What it does

Given a galaxy catalog with positions, velocities, and stellar masses at one or more redshift snapshots, this pipeline:

1. Finds all galaxy pairs within a configurable 3D separation threshold (default 25 kpc)
2. Computes 3D relative velocities for each pair
3. Bins pairs by stellar mass, separation, and redshift
4. Writes pair catalogs to disk
5. Produces histograms of the relative velocity distribution in each bin

## Quick start

```bash
# Generate test data and run the full pipeline with validation plots
python pipeline.py --validate

# Run calculation only (writes results to results/)
python pipeline.py --calc-only

# Remake plots from existing results (no recalculation)
python pipeline.py --plot-only
```

## Configuration

All parameters live in `config.py`. Key settings:

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

## Requirements

- Python 3.8+
- numpy
- scipy
- h5py
- matplotlib
