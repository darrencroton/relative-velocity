"""
Plotting module for relative velocity statistics.

Reads results HDF5 files written by calc.py and produces:
  1. vrel_by_mass.png      — 2×3 grid (6 mass bins), one histogram per redshift per panel
  2. vrel_by_sep_z{z}.png  — 2×2 grid (4 sep bins), one histogram per mass bin, one per redshift
  3. Summary statistics table printed to stdout.

In validation_mode=True, overlays analytical Maxwell PDFs using sigma_v_per_bin attributes
stored in the test data HDF5 files.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive; call before importing pyplot
import matplotlib.pyplot as plt
import h5py

# --- Fixed color palettes ---

# One color per redshift (z=2,3,4,5): blue, orange, green, red
Z_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# One color per mass bin (6 bins), light → dark along viridis
MASS_COLORS = [
    "#440154", "#3b528b", "#21908d", "#5dc963", "#fde725", "#fdae61"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mass_bin_edges(config):
    n = round((config["log_mass_max"] - config["log_mass_min"]) / config["mass_bin_width"])
    return np.linspace(config["log_mass_min"], config["log_mass_max"], n + 1)


def _vel_edges(config):
    return np.arange(0, config["vel_max"] + config["vel_bin_width"], config["vel_bin_width"])


def _results_path(z, config):
    return os.path.join(config["results_dir"], f"pairs_z{z:.1f}.hdf5")


def _data_path(z, config):
    return os.path.join(config["data_dir"], f"test_z{z:.1f}.hdf5")


def _load_all_results(config):
    """Return dict: z -> {delta_v, mass_bin, sep_bin} as numpy arrays."""
    results = {}
    for z in config["redshifts"]:
        path = _results_path(z, config)
        assert os.path.isfile(path), (
            f"Results file not found: {path}\nRun calculation first (--calc-only or --generate-test)."
        )
        with h5py.File(path, "r") as f:
            results[z] = dict(
                delta_v  = f["delta_v"][:],
                mass_bin = f["mass_bin"][:],
                sep_bin  = f["sep_bin"][:],
            )
    return results


def _load_sigma_v(config):
    """Load sigma_v_per_bin from test data attributes. Returns None if file absent."""
    path = _data_path(config["redshifts"][0], config)
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as f:
        return f.attrs["sigma_v_per_bin"][:]


def _maxwell_pdf(v, sigma):
    """Maxwell speed distribution: f(v) = sqrt(2/pi) * v² / σ³ * exp(-v²/(2σ²))."""
    return np.sqrt(2.0 / np.pi) * v**2 / sigma**3 * np.exp(-v**2 / (2.0 * sigma**2))


def _draw_hist(ax, dv, vel_edges, color, label, alpha=0.85, linewidth=1.5):
    """Plot a normalized probability density step histogram on ax."""
    if len(dv) == 0:
        return
    counts, _ = np.histogram(dv, bins=vel_edges, density=True)
    ax.stairs(counts, vel_edges, color=color, label=label, alpha=alpha, linewidth=linewidth)


# ---------------------------------------------------------------------------
# Figure 1: distributions by mass bin (all sep bins combined)
# ---------------------------------------------------------------------------

def plot_by_mass(all_results, config, validation_mode=False):
    """
    2×3 grid — one panel per mass bin.
    Each panel overlays histograms for all redshifts (different colors).
    validation_mode: overlays Maxwell PDF curve.

    Returns the Figure object.
    """
    mass_edges  = _mass_bin_edges(config)
    n_mass_bins = len(mass_edges) - 1
    vel_edges   = _vel_edges(config)
    v_curve     = np.linspace(0, config["vel_max"], 600)

    sigma_v = _load_sigma_v(config) if validation_mode else None

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    axes_flat = axes.flatten()

    for b in range(n_mass_bins):
        ax = axes_flat[b]

        for z, color in zip(config["redshifts"], Z_COLORS):
            dv = all_results[z]["delta_v"][all_results[z]["mass_bin"] == b]
            _draw_hist(ax, dv, vel_edges, color, label=f"z = {z:.0f}")

        if validation_mode and sigma_v is not None:
            sigma = sigma_v[b]
            ax.plot(
                v_curve, _maxwell_pdf(v_curve, sigma),
                "k--", linewidth=1.8, label=f"Maxwell (σ = {sigma:.0f} km/s)",
            )

        mass_label = f"log M = [{mass_edges[b]:.1f}, {mass_edges[b+1]:.1f})"
        ax.set_title(mass_label, fontsize=9)
        ax.set_xlabel("|Δv| (km/s)", fontsize=8)
        ax.set_ylabel("Prob. density", fontsize=8)
        ax.set_xlim(0, config["vel_max"])
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        if b == 0:
            ax.legend(fontsize=7, framealpha=0.8)

    title = "Relative velocity distributions by stellar mass bin"
    if validation_mode:
        title += " — Maxwell validation"
    fig.suptitle(title, fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: distributions by separation bin (one figure per redshift)
# ---------------------------------------------------------------------------

def plot_by_sep(all_results, config, z):
    """
    2×2 grid — one panel per separation bin, for a single redshift.
    Each panel overlays histograms for all mass bins (different colors).

    Returns the Figure object.
    """
    sep_edges   = config["sep_bins"]
    n_sep_bins  = len(sep_edges) - 1
    mass_edges  = _mass_bin_edges(config)
    n_mass_bins = len(mass_edges) - 1
    vel_edges   = _vel_edges(config)
    data        = all_results[z]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes_flat = axes.flatten()

    for s in range(n_sep_bins):
        ax = axes_flat[s]
        sep_label = f"[{sep_edges[s]}, {sep_edges[s+1]}) kpc"

        for b in range(n_mass_bins):
            mask = (data["mass_bin"] == b) & (data["sep_bin"] == s)
            dv   = data["delta_v"][mask]
            mlabel = f"[{mass_edges[b]:.1f}, {mass_edges[b+1]:.1f})"
            _draw_hist(ax, dv, vel_edges, MASS_COLORS[b], label=mlabel)

        ax.set_title(f"Separation {sep_label}", fontsize=9)
        ax.set_xlabel("|Δv| (km/s)", fontsize=8)
        ax.set_ylabel("Prob. density", fontsize=8)
        ax.set_xlim(0, config["vel_max"])
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        if s == 0:
            ax.legend(fontsize=6.5, ncol=2, framealpha=0.8)

    fig.suptitle(f"Relative velocity by separation bin — z = {z:.1f}", fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

def print_stats_table(all_results, config):
    """Print per-(mass_bin, sep_bin, redshift) statistics to stdout."""
    mass_edges = _mass_bin_edges(config)
    sep_edges  = config["sep_bins"]
    n_mass     = len(mass_edges) - 1
    n_sep      = len(sep_edges) - 1

    col = "{:>4}  {:^16}  {:^14}  {:>6}  {:>8}  {:>8}  {:>8}"
    header = col.format("z", "log M range", "Sep (kpc)", "N", "Mean", "Median", "Std")
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for z in config["redshifts"]:
        data = all_results[z]
        first_z = True
        for b in range(n_mass):
            for s in range(n_sep):
                mask = (data["mass_bin"] == b) & (data["sep_bin"] == s)
                dv   = data["delta_v"][mask]
                if len(dv) == 0:
                    continue
                z_str  = f"{z:.1f}" if first_z else ""
                mlabel = f"[{mass_edges[b]:.1f},{mass_edges[b+1]:.1f})"
                slabel = f"[{sep_edges[s]},{sep_edges[s+1]})"
                print(col.format(
                    z_str, mlabel, slabel, len(dv),
                    f"{dv.mean():.1f}", f"{np.median(dv):.1f}", f"{dv.std():.1f}",
                ))
                first_z = False
        print("-" * len(header))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def make_plots(config, validation_mode=False):
    """
    Generate all figures and print the summary stats table.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    validation_mode : bool
        If True, overlay Maxwell PDF curves on the mass-bin figure.
    """
    os.makedirs(config["figures_dir"], exist_ok=True)

    print("  Loading results...")
    all_results = _load_all_results(config)

    # Figure 1: by mass bin
    print("  Plotting by mass bin...")
    fig  = plot_by_mass(all_results, config, validation_mode=validation_mode)
    path = os.path.join(config["figures_dir"], "vrel_by_mass.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Wrote {path}")

    # Figure 2: by separation bin (one per redshift)
    print("  Plotting by separation bin...")
    for z in config["redshifts"]:
        fig  = plot_by_sep(all_results, config, z)
        path = os.path.join(config["figures_dir"], f"vrel_by_sep_z{z:.1f}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Wrote {path}")

    print_stats_table(all_results, config)
