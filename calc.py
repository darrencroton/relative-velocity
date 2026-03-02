"""
Calculation driver: loops over redshifts, finds pairs, writes results to disk.

Results are stored as raw pair catalogs (not pre-binned) so plot.py can
re-bin or compute new statistics without re-running the calculation.
"""

import os
import datetime
import h5py
import numpy as np

from data_reader import load_galaxy_catalog
from pair_finder import find_pairs


def _data_path(z, config):
    return os.path.join(config["data_dir"], f"test_z{z:.1f}.hdf5")


def _results_path(z, config):
    return os.path.join(config["results_dir"], f"pairs_z{z:.1f}.hdf5")


def _save_pairs(pairs, filepath, z, config):
    """Write pair catalog arrays and metadata to HDF5."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, "w") as f:
        for key, arr in pairs.items():
            f.create_dataset(key, data=arr)

        # Provenance metadata.
        f.attrs["redshift"]     = z
        f.attrs["n_pairs"]      = len(pairs["delta_v"])
        f.attrs["timestamp"]    = datetime.datetime.utcnow().isoformat()
        f.attrs["mass_bin_by"]  = config["mass_bin_by"]
        f.attrs["mass_ratio_min"] = config["mass_ratio_min"]
        f.attrs["max_sep_kpc"]  = config["max_sep"]


def run_calculation(config):
    """
    Run pair-finding for all redshift snapshots and write results files.

    Asserts that data files exist before starting so the failure is clear.
    """
    os.makedirs(config["results_dir"], exist_ok=True)

    for z in config["redshifts"]:
        data_path    = _data_path(z, config)
        results_path = _results_path(z, config)

        assert os.path.isfile(data_path), (
            f"Data file not found: {data_path}\n"
            "Run with --generate-test first to create test data."
        )

        print(f"  z={z:.1f}: loading catalog...")
        catalog = load_galaxy_catalog(data_path, config)
        n_gal   = len(catalog["x"])

        print(f"  z={z:.1f}: {n_gal} galaxies selected; finding pairs...")
        pairs   = find_pairs(catalog, config)
        n_pairs = len(pairs["delta_v"])

        print(f"  z={z:.1f}: {n_pairs} pairs found. Writing {results_path}...")
        _save_pairs(pairs, results_path, z, config)

    print("Calculation complete.")
