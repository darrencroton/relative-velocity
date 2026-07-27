"""
Data I/O for the relative velocity pipeline.

Version 1 reads the test HDF5 catalogs produced by generate_test_data.py.
Version 2 (future) will add load_sage_catalog() for real SAGE output.

This is the only module that changes when switching data sources. All unit
conversions (h-factors, comoving→proper, etc.) must happen here so downstream
code always sees positions in Mpc, velocities in km/s, masses in log10(M_sun).
"""

import os
import h5py
import numpy as np


def load_galaxy_catalog(filepath, config):
    """
    Load a galaxy catalog from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    config : dict
        Pipeline configuration (used for any filtering/validation).

    Returns
    -------
    dict with keys:
        'x', 'y', 'z'          : positions, Mpc           (1D float arrays)
        'vx', 'vy', 'vz'       : velocities, km/s         (1D float arrays)
        'log_stellar_mass'      : log10(M_star / M_sun)    (1D float array)
        'redshift'              : float
        'box_size'              : float, Mpc
    """
    assert os.path.isfile(filepath), f"Catalog file not found: {filepath}"

    with h5py.File(filepath, "r") as f:
        catalog = dict(
            x               = f["x"][:].astype(float),
            y               = f["y"][:].astype(float),
            z               = f["z"][:].astype(float),
            vx              = f["vx"][:].astype(float),
            vy              = f["vy"][:].astype(float),
            vz              = f["vz"][:].astype(float),
            log_stellar_mass = f["log_stellar_mass"][:].astype(float),
            redshift        = float(f.attrs["redshift"]),
            box_size        = float(f.attrs["box_size"]),
        )

    n = len(catalog["x"])
    assert n > 0, f"Empty catalog: {filepath}"

    # Basic sanity checks on units / ranges.
    assert catalog["box_size"] > 0, "box_size must be positive"
    assert np.all(catalog["log_stellar_mass"] >= 0), (
        "log_stellar_mass contains non-positive values; check units (expected log10 M_sun)"
    )

    # Apply stellar mass selection.
    log_m = catalog["log_stellar_mass"]
    mask  = (log_m >= config["log_mass_min"]) & (log_m <= config["log_mass_max"])
    n_selected = mask.sum()
    assert n_selected > 0, (
        f"No galaxies in mass range [{config['log_mass_min']}, {config['log_mass_max']}] "
        f"log10 M_sun in {filepath}"
    )

    for key in ("x", "y", "z", "vx", "vy", "vz", "log_stellar_mass"):
        catalog[key] = catalog[key][mask]

    return catalog
