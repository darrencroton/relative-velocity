"""
Generate mock galaxy catalogs with analytically known relative velocity distributions.

Strategy: construct pairs explicitly so we know exactly what the pipeline should recover.
For each pair, each component of the velocity difference is drawn from N(0, sigma_v(M)),
making the 3D speed |dv| follow a Maxwell distribution with scale sigma_v.

Analytical predictions stored as HDF5 attributes for use by plot.py in validation mode.
"""

import os
import numpy as np
import h5py

# Per-mass-bin velocity dispersion (one component).
# Index 0 = lowest mass bin [8.0, 8.5), ..., index 5 = [10.5, 11.0).
# The 3D speed follows Maxwell(sigma_v), with mean = sigma_v * sqrt(8/pi) ~ 1.596 * sigma_v.
SIGMA_V_PER_BIN = [40.0, 60.0, 80.0, 110.0, 150.0, 200.0]  # km/s per component

N_PAIRS        = 3000   # generated pairs per snapshot
N_FIELD        = 2000   # isolated field galaxies per snapshot
MIN_SEP_KPC    = 3.0    # minimum pair separation to generate (kpc)
MAX_SEP_KPC    = 28.0   # maximum pair separation (slightly wider than analysis range)
SIGMA_BULK     = 200.0  # km/s per component, bulk velocity of each galaxy
FIELD_EXCL_MPC = 0.1    # field galaxies must be > this distance (Mpc) from all pairs


def _mass_bin_index(log_mass, log_mass_min, mass_bin_width):
    """Return integer bin index for a log10 stellar mass value."""
    return int((log_mass - log_mass_min) / mass_bin_width)


def _random_unit_vectors(n, rng):
    """Draw n random unit vectors uniformly on the sphere."""
    phi   = rng.uniform(0, 2 * np.pi, n)
    costh = rng.uniform(-1, 1, n)
    sinth = np.sqrt(1 - costh**2)
    return np.column_stack([sinth * np.cos(phi), sinth * np.sin(phi), costh])


def generate_snapshot(redshift, config, rng):
    """
    Build a mock galaxy catalog for a single redshift snapshot.

    Returns a dict with arrays ready to write to HDF5.
    """
    box_size      = config["box_size"]        # Mpc
    log_mass_min  = config["log_mass_min"]
    log_mass_max  = config["log_mass_max"]
    mass_bin_width = config["mass_bin_width"]
    n_mass_bins   = round((log_mass_max - log_mass_min) / mass_bin_width)

    # --- Build pairs ---

    # Primary masses: log-uniform so we get roughly equal numbers across dex bins.
    # We draw uniformly in log space which naturally gives ~equal counts per dex bin.
    log_mass_primary = rng.uniform(log_mass_min, log_mass_max, N_PAIRS)

    # Mass ratios uniform in [mass_ratio_min, 1.0]; secondary always <= primary.
    mass_ratio_min  = config["mass_ratio_min"]
    mass_ratios     = rng.uniform(mass_ratio_min, 1.0, N_PAIRS)
    log_mass_secondary = log_mass_primary + np.log10(mass_ratios)

    # Primary positions: uniform in box.
    pos_primary = rng.uniform(0, box_size, (N_PAIRS, 3))  # Mpc

    # Pair separations: uniform in [MIN_SEP_KPC, MAX_SEP_KPC], random direction.
    sep_kpc   = rng.uniform(MIN_SEP_KPC, MAX_SEP_KPC, N_PAIRS)
    sep_mpc   = sep_kpc / 1000.0
    directions = _random_unit_vectors(N_PAIRS, rng)
    pos_secondary = pos_primary + sep_mpc[:, None] * directions
    # Apply periodic wrap so secondaries stay inside box.
    pos_secondary = pos_secondary % box_size

    # Bulk velocities for primaries.
    vel_primary = rng.normal(0, SIGMA_BULK, (N_PAIRS, 3))  # km/s

    # Relative velocity per pair, drawn from N(0, sigma_v(M)) per component.
    # sigma_v depends on the primary mass bin.
    bin_indices = np.clip(
        [_mass_bin_index(m, log_mass_min, mass_bin_width) for m in log_mass_primary],
        0, n_mass_bins - 1,
    )
    sigma_v_per_pair = np.array([SIGMA_V_PER_BIN[b] for b in bin_indices])
    dv = rng.normal(0, 1, (N_PAIRS, 3)) * sigma_v_per_pair[:, None]  # km/s
    vel_secondary = vel_primary + dv

    # --- Build field galaxies ---

    # Place field galaxies avoiding proximity to any pair galaxy.
    # Simple rejection loop: draw positions, discard any within FIELD_EXCL_MPC of a pair.
    all_pair_pos = np.vstack([pos_primary, pos_secondary])  # (2*N_PAIRS, 3)

    field_positions = []
    field_attempts  = 0
    while len(field_positions) < N_FIELD:
        field_attempts += 1
        if field_attempts > N_FIELD * 50:
            # Safety valve: shouldn't happen with a 500 Mpc box and 6000 pairs.
            raise RuntimeError(
                f"Could not place {N_FIELD} field galaxies after many attempts. "
                "Box may be too crowded."
            )
        candidate = rng.uniform(0, box_size, 3)
        # Check minimum image distance to all pair galaxies.
        diff = all_pair_pos - candidate[None, :]
        diff -= box_size * np.round(diff / box_size)
        min_dist = np.sqrt((diff**2).sum(axis=1)).min()
        if min_dist > FIELD_EXCL_MPC:
            field_positions.append(candidate)

    field_positions = np.array(field_positions)  # (N_FIELD, 3)
    field_vel       = rng.normal(0, SIGMA_BULK, (N_FIELD, 3))
    field_log_mass  = rng.uniform(log_mass_min, log_mass_max, N_FIELD)

    # --- Assemble full catalog ---

    # Order: primaries, secondaries, field galaxies.
    x   = np.concatenate([pos_primary[:, 0], pos_secondary[:, 0], field_positions[:, 0]])
    y   = np.concatenate([pos_primary[:, 1], pos_secondary[:, 1], field_positions[:, 1]])
    z   = np.concatenate([pos_primary[:, 2], pos_secondary[:, 2], field_positions[:, 2]])
    vx  = np.concatenate([vel_primary[:, 0], vel_secondary[:, 0], field_vel[:, 0]])
    vy  = np.concatenate([vel_primary[:, 1], vel_secondary[:, 1], field_vel[:, 1]])
    vz  = np.concatenate([vel_primary[:, 2], vel_secondary[:, 2], field_vel[:, 2]])

    n_paired = 2 * N_PAIRS
    log_stellar_mass = np.concatenate([
        log_mass_primary, log_mass_secondary, field_log_mass
    ])
    is_paired = np.concatenate([
        np.ones(n_paired, dtype=bool),
        np.zeros(N_FIELD, dtype=bool),
    ])
    # pair_id: primaries and their secondaries share the same id; field = -1.
    pair_id = np.concatenate([
        np.arange(N_PAIRS),            # primaries: ids 0..N_PAIRS-1
        np.arange(N_PAIRS),            # secondaries: same ids
        np.full(N_FIELD, -1, dtype=int),
    ])

    return dict(
        x=x, y=y, z=z,
        vx=vx, vy=vy, vz=vz,
        log_stellar_mass=log_stellar_mass,
        is_paired=is_paired,
        pair_id=pair_id,
    )


def generate_all_snapshots(config):
    """Write one HDF5 file per redshift to config['data_dir']."""
    os.makedirs(config["data_dir"], exist_ok=True)

    for z in config["redshifts"]:
        filepath = os.path.join(config["data_dir"], f"test_z{z:.1f}.hdf5")
        rng = np.random.default_rng(seed=int(z * 100))  # reproducible, different per z

        print(f"  z={z:.1f}: generating {N_PAIRS} pairs + {N_FIELD} field galaxies...")
        catalog = generate_snapshot(z, config, rng)

        with h5py.File(filepath, "w") as f:
            for key, arr in catalog.items():
                f.create_dataset(key, data=arr)

            # Attributes for validation
            f.attrs["redshift"]       = z
            f.attrs["box_size"]       = config["box_size"]
            f.attrs["n_pairs"]        = N_PAIRS
            f.attrs["n_field"]        = N_FIELD
            # sigma_v values ordered by mass bin index; used by plot.py for Maxwell overlay.
            f.attrs["sigma_v_per_bin"] = np.array(SIGMA_V_PER_BIN)

        print(f"  Wrote {filepath}")


if __name__ == "__main__":
    from config import config
    generate_all_snapshots(config)
