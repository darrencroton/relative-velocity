"""
KD-tree pair finding and relative velocity computation.

Given a galaxy catalog (Mpc, km/s, log10 M_sun), finds all pairs within
max_sep kpc and returns their properties as arrays.

The only unit conversion in the pipeline: positions Mpc → kpc (×1000)
at the top of find_pairs(), so separations and the KD-tree radius are in kpc.
"""

import numpy as np
from scipy.spatial import cKDTree


def _mass_bin_edges(config):
    """Return array of mass bin edges in log10(M_sun)."""
    n_bins = round((config["log_mass_max"] - config["log_mass_min"]) / config["mass_bin_width"])
    return np.linspace(config["log_mass_min"], config["log_mass_max"], n_bins + 1)


def _assign_mass_bins(log_mass_primary, log_mass_secondary, config):
    """
    Assign each pair to an integer mass bin based on config['mass_bin_by'].

    Returns an int array of bin indices (0-based). Pairs falling outside the
    mass range [log_mass_min, log_mass_max] are assigned index -1.
    """
    strategy = config["mass_bin_by"]
    if strategy == "primary":
        ref_mass = log_mass_primary
    elif strategy == "secondary":
        ref_mass = log_mass_secondary
    elif strategy == "mean":
        ref_mass = 0.5 * (log_mass_primary + log_mass_secondary)
    elif strategy == "total":
        # Total stellar mass in log space: log10(10^m1 + 10^m2)
        ref_mass = np.log10(10**log_mass_primary + 10**log_mass_secondary)
    else:
        raise ValueError(
            f"Unknown mass_bin_by strategy: '{strategy}'. "
            "Valid options: 'primary', 'secondary', 'mean', 'total'."
        )

    edges = _mass_bin_edges(config)
    # np.digitize returns 1-based; subtract 1 for 0-based, set out-of-range to -1.
    raw = np.digitize(ref_mass, edges) - 1
    n_bins = len(edges) - 1
    out_of_range = (raw < 0) | (raw >= n_bins)
    raw[out_of_range] = -1
    return raw


def _assign_sep_bins(separations_kpc, config):
    """
    Assign each pair to an integer separation bin.

    Returns int array; pairs outside the bin range get index -1.
    """
    edges = config["sep_bins"]
    raw = np.digitize(separations_kpc, edges) - 1
    n_bins = len(edges) - 1
    out_of_range = (raw < 0) | (raw >= n_bins)
    raw[out_of_range] = -1
    return raw


def find_pairs(catalog, config):
    """
    Find all galaxy pairs within max_sep kpc and compute their properties.

    Parameters
    ----------
    catalog : dict
        Output of data_reader.load_galaxy_catalog().
    config : dict
        Pipeline configuration.

    Returns
    -------
    dict of 1D arrays, one entry per pair:
        'mass_primary'    : log10(M/M_sun), more massive galaxy
        'mass_secondary'  : log10(M/M_sun)
        'mass_ratio'      : M_secondary / M_primary (linear, always <= 1)
        'separation_kpc'  : 3D separation in kpc
        'delta_v'         : 3D relative speed in km/s
        'mass_bin'        : int, mass bin index (-1 if outside range)
        'sep_bin'         : int, separation bin index (-1 if outside range)
    """
    # Convert positions Mpc → kpc so the KD-tree radius is in kpc directly.
    pos_kpc = np.column_stack([
        catalog["x"] * 1e3,
        catalog["y"] * 1e3,
        catalog["z"] * 1e3,
    ])
    box_kpc   = catalog["box_size"] * 1e3
    log_mass  = catalog["log_stellar_mass"]
    vel       = np.column_stack([catalog["vx"], catalog["vy"], catalog["vz"]])

    max_sep   = config["max_sep"]     # kpc

    # Build KD-tree with periodic boundary conditions.
    tree = cKDTree(pos_kpc, boxsize=box_kpc)

    # query_pairs returns a set of (i, j) with i < j — no double-counting.
    raw_pairs = tree.query_pairs(r=max_sep)

    if len(raw_pairs) == 0:
        # Return empty arrays with correct structure so downstream code still runs.
        empty = np.array([], dtype=float)
        return dict(
            mass_primary=empty,
            mass_secondary=empty,
            mass_ratio=empty,
            separation_kpc=empty,
            delta_v=empty,
            mass_bin=np.array([], dtype=int),
            sep_bin=np.array([], dtype=int),
        )

    idx_i, idx_j = zip(*raw_pairs)
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)

    # Identify primary (more massive) and secondary.
    m_i = log_mass[idx_i]
    m_j = log_mass[idx_j]
    is_i_primary = m_i >= m_j

    log_mass_primary   = np.where(is_i_primary, m_i, m_j)
    log_mass_secondary = np.where(is_i_primary, m_j, m_i)

    # Mass ratio in linear space: always <= 1.
    mass_ratio = 10 ** (log_mass_secondary - log_mass_primary)

    # Apply mass ratio cut.
    mass_ratio_min = config["mass_ratio_min"]
    keep = mass_ratio >= mass_ratio_min
    idx_i              = idx_i[keep]
    idx_j              = idx_j[keep]
    log_mass_primary   = log_mass_primary[keep]
    log_mass_secondary = log_mass_secondary[keep]
    mass_ratio         = mass_ratio[keep]

    if len(idx_i) == 0:
        empty = np.array([], dtype=float)
        return dict(
            mass_primary=empty,
            mass_secondary=empty,
            mass_ratio=empty,
            separation_kpc=empty,
            delta_v=empty,
            mass_bin=np.array([], dtype=int),
            sep_bin=np.array([], dtype=int),
        )

    # Compute separations with minimum image convention (same as cKDTree uses internally).
    diff_kpc = pos_kpc[idx_i] - pos_kpc[idx_j]
    diff_kpc -= box_kpc * np.round(diff_kpc / box_kpc)
    separations_kpc = np.sqrt((diff_kpc**2).sum(axis=1))

    # Relative velocity magnitude.
    dv  = vel[idx_i] - vel[idx_j]
    delta_v = np.sqrt((dv**2).sum(axis=1))

    # Bin assignments.
    mass_bin = _assign_mass_bins(log_mass_primary, log_mass_secondary, config)
    sep_bin  = _assign_sep_bins(separations_kpc, config)

    return dict(
        mass_primary   = log_mass_primary,
        mass_secondary = log_mass_secondary,
        mass_ratio     = mass_ratio,
        separation_kpc = separations_kpc,
        delta_v        = delta_v,
        mass_bin       = mass_bin,
        sep_bin        = sep_bin,
    )
