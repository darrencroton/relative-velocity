"""
Exact-recovery unit tests for pair_finder.py.

Each test constructs a minimal hand-crafted catalog where the correct answer is
known analytically to machine precision, then asserts the pipeline output matches.
These tests catch logical bugs before any statistical analysis.

Tests are independent of random seeds, file I/O, and external data.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pair_finder import find_pairs

# Minimal config used throughout; keep only the fields pair_finder reads.
BASE_CONFIG = dict(
    box_size       = 1.0,          # Mpc  (= 1000 kpc)
    log_mass_min   = 8.0,
    log_mass_max   = 11.0,
    mass_bin_width = 1.0,          # 3 bins: [8,9), [9,10), [10,11)
    sep_bins       = [0, 500, 1000],  # kpc — intentionally wide so nothing falls outside
    mass_ratio_min = 0.1,
    mass_bin_by    = "primary",
    max_sep        = 100.0,        # kpc
)


def _catalog(positions_mpc, velocities_kms, log_masses, box_size_mpc=1.0):
    """Build a minimal galaxy catalog dict from arrays."""
    pos = np.array(positions_mpc, dtype=float)
    vel = np.array(velocities_kms, dtype=float)
    m   = np.array(log_masses, dtype=float)
    return dict(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2],
        log_stellar_mass=m,
        box_size=float(box_size_mpc),
    )


# ---------------------------------------------------------------------------
# 1. Simple pair recovery
# ---------------------------------------------------------------------------

class TestBasicPairRecovery:
    def test_single_pair_found(self):
        """Two galaxies 10 kpc apart — exactly one pair should be found."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],  # 10 kpc separation
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert len(pairs["delta_v"]) == 1

    def test_single_pair_separation(self):
        """Separation must be recovered to <0.001 kpc precision."""
        sep_kpc = 17.3
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [sep_kpc / 1000.0, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert len(pairs["delta_v"]) == 1
        assert abs(pairs["separation_kpc"][0] - sep_kpc) < 1e-9

    def test_pair_beyond_max_sep_not_found(self):
        """Galaxy just beyond max_sep must not appear."""
        cfg = {**BASE_CONFIG, "max_sep": 25.0}
        sep_just_outside = 25.001  # kpc
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [sep_just_outside / 1000.0, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 0

    def test_pair_at_max_sep_boundary(self):
        """Galaxy just inside max_sep must be found."""
        cfg = {**BASE_CONFIG, "max_sep": 25.0}
        sep_just_inside = 24.999  # kpc
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [sep_just_inside / 1000.0, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 1

    def test_empty_catalog_returns_empty(self):
        """Single galaxy — no pairs possible."""
        cat = _catalog([[0, 0, 0]], [[0, 0, 0]], [10.0])
        pairs = find_pairs(cat, BASE_CONFIG)
        assert len(pairs["delta_v"]) == 0

    def test_no_double_counting(self):
        """Three galaxies all within max_sep: should find exactly 3 pairs, not 6."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0], [0.0, 0.010, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            log_masses=[10.5, 10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert len(pairs["delta_v"]) == 3


# ---------------------------------------------------------------------------
# 2. Periodic boundary condition recovery
# ---------------------------------------------------------------------------

class TestPeriodicBoundary:
    def test_pair_across_x_boundary(self):
        """
        Galaxy A at x=0.001 Mpc, Galaxy B at x=0.999 Mpc in a 1 Mpc box.
        Through the periodic boundary the true separation is 2 kpc, not 998 kpc.
        """
        cfg = {**BASE_CONFIG, "max_sep": 10.0}  # 10 kpc — would miss without PBC
        cat = _catalog(
            positions_mpc=[[0.001, 0.0, 0.0], [0.999, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 1, (
            "Pair straddling the periodic boundary was not found"
        )
        # True separation: (0.001 + (1 - 0.999)) * 1000 = 2 kpc
        assert abs(pairs["separation_kpc"][0] - 2.0) < 1e-9

    def test_pair_across_corner(self):
        """
        Pair near the (x,y,z) = (0,0,0) corner; all three coordinates wrap.
        Galaxy A at (0.001, 0.001, 0.001) Mpc, B at (0.999, 0.999, 0.999) Mpc.
        True separation: sqrt(3) * 2 kpc ≈ 3.464 kpc.
        """
        sep_true = np.sqrt(3.0) * 2.0  # kpc
        cfg = {**BASE_CONFIG, "max_sep": sep_true + 1.0}
        cat = _catalog(
            positions_mpc=[[0.001, 0.001, 0.001], [0.999, 0.999, 0.999]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 1
        assert abs(pairs["separation_kpc"][0] - sep_true) < 1e-9

    def test_pair_across_boundary_velocity_unaffected(self):
        """
        Relative velocity must be computed from the raw velocity vectors,
        not from positions. The boundary wrap must NOT alter the velocity subtraction.
        """
        cfg = {**BASE_CONFIG, "max_sep": 10.0}
        # Known velocities: delta_v should be exactly sqrt(3^2 + 4^2) = 5 km/s
        cat = _catalog(
            positions_mpc=[[0.001, 0.0, 0.0], [0.999, 0.0, 0.0]],
            velocities_kms=[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 1
        assert abs(pairs["delta_v"][0] - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# 3. Velocity recovery
# ---------------------------------------------------------------------------

class TestVelocityRecovery:
    def test_1d_velocity(self):
        """delta_v = 100 km/s along x-axis."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert abs(pairs["delta_v"][0] - 100.0) < 1e-9

    def test_3d_velocity_pythagorean(self):
        """3-4-5 Pythagorean triple: |dv| = 5 km/s exactly."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert abs(pairs["delta_v"][0] - 5.0) < 1e-9

    def test_3d_velocity_all_components(self):
        """|dv| = sqrt(1^2 + 2^2 + 2^2) = 3 km/s."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[1.0, 2.0, 2.0], [0.0, 0.0, 0.0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert abs(pairs["delta_v"][0] - 3.0) < 1e-9

    def test_velocity_is_symmetric(self):
        """
        Swapping which galaxy is 'i' and which is 'j' must give the same |dv|.
        The magnitude |v_i - v_j| == |v_j - v_i|.
        """
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[300.0, 100.0, 50.0], [100.0, 200.0, 150.0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        expected = np.sqrt((300 - 100)**2 + (100 - 200)**2 + (50 - 150)**2)
        assert abs(pairs["delta_v"][0] - expected) < 1e-9

    def test_zero_relative_velocity(self):
        """Galaxy pair with identical velocities: delta_v must be exactly 0."""
        v = [123.4, -567.8, 910.1]
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[v, v],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert pairs["delta_v"][0] == 0.0


# ---------------------------------------------------------------------------
# 4. Mass assignment logic
# ---------------------------------------------------------------------------

class TestMassAssignment:
    def test_primary_is_more_massive(self):
        """mass_primary must always be >= mass_secondary."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[9.5, 10.0],   # j is more massive than i
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert pairs["mass_primary"][0] == 10.0
        assert pairs["mass_secondary"][0] == 9.5

    def test_mass_ratio_always_leq_1(self):
        """mass_ratio = M_secondary / M_primary must always be <= 1."""
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[9.0, 10.0],
        )
        pairs = find_pairs(cat, BASE_CONFIG)
        assert pairs["mass_ratio"][0] <= 1.0
        # mass_ratio = 10^(9 - 10) = 0.1
        assert abs(pairs["mass_ratio"][0] - 0.1) < 1e-9

    def test_mass_ratio_cut_excludes_pair(self):
        """Pair with mass_ratio < mass_ratio_min must be excluded."""
        cfg = {**BASE_CONFIG, "mass_ratio_min": 0.1}
        # log_mass difference of >1 dex → ratio < 0.1
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.5, 9.3],   # ratio = 10^(9.3 - 10.5) = 10^-1.2 ≈ 0.063 < 0.1
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 0

    def test_mass_ratio_cut_keeps_pair(self):
        """Pair with mass_ratio == mass_ratio_min should be kept (>= cut)."""
        cfg = {**BASE_CONFIG, "mass_ratio_min": 0.1}
        # 10^(9.0 - 10.0) = 0.1 exactly
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.0],
        )
        pairs = find_pairs(cat, cfg)
        assert len(pairs["delta_v"]) == 1

    def test_mass_bin_assignment(self):
        """
        With 3 bins [8,9), [9,10), [10,11) and mass_bin_by='primary':
        primary log_mass=9.7 → bin index 1.
        """
        cfg = {**BASE_CONFIG, "mass_bin_width": 1.0}  # bins: [8,9)=0, [9,10)=1, [10,11)=2
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[9.7, 9.5],  # primary=9.7 → bin 1
        )
        pairs = find_pairs(cat, cfg)
        assert pairs["mass_bin"][0] == 1

    def test_mass_bin_by_secondary(self):
        """mass_bin_by='secondary': bin is determined by the less massive galaxy."""
        cfg = {**BASE_CONFIG, "mass_bin_width": 1.0, "mass_bin_by": "secondary"}
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.5, 9.7],  # secondary=9.7 → bin 1; primary=10.5 → bin 2
        )
        pairs = find_pairs(cat, cfg)
        assert pairs["mass_bin"][0] == 1

    def test_mass_bin_by_mean(self):
        """mass_bin_by='mean': bin uses average of log masses."""
        cfg = {**BASE_CONFIG, "mass_bin_width": 1.0, "mass_bin_by": "mean"}
        # mean = (8.6 + 9.6) / 2 = 9.1 → bin 1
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[9.6, 8.6],
        )
        pairs = find_pairs(cat, cfg)
        assert pairs["mass_bin"][0] == 1

    def test_mass_bin_by_total(self):
        """mass_bin_by='total': bin uses log10(10^m1 + 10^m2)."""
        cfg = {**BASE_CONFIG, "mass_bin_width": 1.0, "mass_bin_by": "total"}
        # 10^9.0 + 10^9.0 = 2e9 → log10 = 9.301 → bin 1
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[9.0, 9.0],
        )
        pairs = find_pairs(cat, cfg)
        expected_total = np.log10(2 * 10**9.0)  # ≈ 9.301 → bin 1
        assert pairs["mass_bin"][0] == 1

    def test_invalid_mass_bin_by_raises(self):
        """Unknown mass_bin_by strategy must raise ValueError."""
        cfg = {**BASE_CONFIG, "mass_bin_by": "nonsense"}
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0]],
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        with pytest.raises(ValueError, match="mass_bin_by"):
            find_pairs(cat, cfg)


# ---------------------------------------------------------------------------
# 5. Separation bin assignment
# ---------------------------------------------------------------------------

class TestSepBinAssignment:
    def test_sep_bin_correct(self):
        """
        sep_bins = [0, 10, 15, 20, 25].
        Separation of 12 kpc → bin 1 ([10, 15)).
        """
        cfg = {**BASE_CONFIG, "sep_bins": [0, 10, 15, 20, 25], "max_sep": 25.0}
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.012, 0.0, 0.0]],  # 12 kpc
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert pairs["sep_bin"][0] == 1

    def test_sep_bin_first_bin(self):
        """Separation of 5 kpc → bin 0 ([0, 10))."""
        cfg = {**BASE_CONFIG, "sep_bins": [0, 10, 15, 20, 25], "max_sep": 25.0}
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]],  # 5 kpc
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert pairs["sep_bin"][0] == 0

    def test_sep_bin_last_bin(self):
        """Separation of 22 kpc → bin 3 ([20, 25))."""
        cfg = {**BASE_CONFIG, "sep_bins": [0, 10, 15, 20, 25], "max_sep": 25.0}
        cat = _catalog(
            positions_mpc=[[0.0, 0.0, 0.0], [0.022, 0.0, 0.0]],  # 22 kpc
            velocities_kms=[[0, 0, 0], [0, 0, 0]],
            log_masses=[10.0, 9.5],
        )
        pairs = find_pairs(cat, cfg)
        assert pairs["sep_bin"][0] == 3
