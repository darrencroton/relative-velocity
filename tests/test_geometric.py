"""
Geometric tests for pair counting completeness.

For N galaxies placed uniformly at random in a periodic box of side L (volume V = L^3),
the expected number of pairs within a sphere of radius r is:

    N_pairs_expected = N*(N-1)/2 * (4/3 * pi * r^3) / V

This is the Poisson point process result. It is exact in expectation; the actual
count is Poisson-distributed with this mean for large V (i.e., when the sphere
radius is small compared to the box, r << L/2).

For r << L/2 the finite-box correction (edge effects) is negligible: the
sphere fits comfortably inside any image, so the pair count per unit volume
is just the bulk number density.

References:
  Peebles, P.J.E. (1980) The Large-Scale Structure of the Universe, eq. (36.4)
  Martinez & Saar (2002) Statistics of the Galaxy Distribution, ch. 2

These tests validate the cKDTree pair-finding step in isolation, independent
of mass selection, velocity computation, or binning logic.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scipy.spatial import cKDTree


def _n_pairs_expected(n_gal, box_kpc, r_kpc):
    """Analytical expected pair count for a uniform Poisson field."""
    V = box_kpc**3
    return n_gal * (n_gal - 1) / 2.0 * (4.0 / 3.0 * np.pi * r_kpc**3) / V


def _count_pairs(n_gal, box_kpc, r_kpc, seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, box_kpc, (n_gal, 3))
    tree = cKDTree(pos, boxsize=box_kpc)
    return len(tree.query_pairs(r=r_kpc))


# ---------------------------------------------------------------------------
# 1. Pair count matches analytical expectation
# ---------------------------------------------------------------------------

class TestPairCountFormula:
    """
    We want r << L/2 so boundary / aliasing effects are negligible.
    Using L = 1000 kpc and r_max = 25 kpc gives r/L = 0.025 << 1.
    N = 20,000 → expected ~6500 pairs, Poisson std ~80, so 5-sigma is ±400 (6%).
    We assert within ±10% to give headroom for Poisson fluctuations across seeds.
    """

    N_GAL   = 20_000
    BOX_KPC = 1_000.0
    R_KPC   = 25.0

    def test_pair_count_matches_formula(self):
        expected = _n_pairs_expected(self.N_GAL, self.BOX_KPC, self.R_KPC)
        found    = _count_pairs(self.N_GAL, self.BOX_KPC, self.R_KPC, seed=7)
        rtol     = 0.10
        assert abs(found - expected) / expected < rtol, (
            f"Pair count {found} deviates from expected {expected:.1f} by "
            f"{100*(found-expected)/expected:.1f}% (tolerance {100*rtol:.0f}%)"
        )

    def test_pair_count_reproducible(self):
        """Same seed must give the same count."""
        c1 = _count_pairs(5000, 1000.0, 25.0, seed=42)
        c2 = _count_pairs(5000, 1000.0, 25.0, seed=42)
        assert c1 == c2

    def test_pair_count_r_cubed_scaling(self):
        """
        Doubling the search radius should increase pair count by ~8x (r^3 scaling).
        We use multiple seeds and average to reduce Poisson noise.
        """
        seeds    = [1, 2, 3, 4, 5]
        box_kpc  = 1_000.0
        n_gal    = 10_000
        r_small  = 15.0
        r_big    = 30.0  # exactly 2x

        counts_small = [_count_pairs(n_gal, box_kpc, r_small, s) for s in seeds]
        counts_big   = [_count_pairs(n_gal, box_kpc, r_big,   s) for s in seeds]

        ratio_observed  = np.mean(counts_big) / np.mean(counts_small)
        ratio_predicted = (r_big / r_small)**3   # = 8.0

        assert abs(ratio_observed - ratio_predicted) / ratio_predicted < 0.05, (
            f"r^3 scaling: observed ratio = {ratio_observed:.3f}, "
            f"predicted = {ratio_predicted:.3f}"
        )

    def test_pair_count_n_squared_scaling(self):
        """
        Doubling N (at fixed box size) should quadruple pair count: N*(N-1)/2 ~ N^2/2.
        Average over multiple seeds.
        """
        seeds   = [10, 11, 12, 13, 14]
        box_kpc = 1_000.0
        r_kpc   = 25.0
        n1      = 5_000
        n2      = 10_000   # exactly 2x

        counts_n1 = [_count_pairs(n1, box_kpc, r_kpc, s)      for s in seeds]
        counts_n2 = [_count_pairs(n2, box_kpc, r_kpc, s + 100) for s in seeds]

        ratio_observed  = np.mean(counts_n2) / np.mean(counts_n1)
        # Exact expected ratio: N2*(N2-1) / (N1*(N1-1))
        ratio_predicted = n2 * (n2 - 1) / (n1 * (n1 - 1))

        assert abs(ratio_observed - ratio_predicted) / ratio_predicted < 0.05, (
            f"N^2 scaling: observed ratio = {ratio_observed:.3f}, "
            f"predicted = {ratio_predicted:.3f}"
        )


# ---------------------------------------------------------------------------
# 2. Periodic boundary correctness (geometric, not velocity)
# ---------------------------------------------------------------------------

class TestPeriodicBoundaryGeometry:
    """
    Verify that the cKDTree periodic pair count is the same regardless of
    where galaxies are shifted within the box (translation invariance).

    For a Poisson field, shifting all positions by a constant vector (mod L)
    must not change the pair count.
    """

    def test_translation_invariance(self):
        n_gal   = 5_000
        box_kpc = 1_000.0
        r_kpc   = 25.0
        rng     = np.random.default_rng(55)

        pos = rng.uniform(0, box_kpc, (n_gal, 3))
        tree1 = cKDTree(pos, boxsize=box_kpc)
        n1 = len(tree1.query_pairs(r=r_kpc))

        # Shift by (300, 400, 500) kpc and wrap.
        shift = np.array([300.0, 400.0, 500.0])
        pos2  = (pos + shift) % box_kpc
        tree2 = cKDTree(pos2, boxsize=box_kpc)
        n2 = len(tree2.query_pairs(r=r_kpc))

        assert n1 == n2, (
            f"Pair count changed after box translation: {n1} → {n2}. "
            "Periodic boundary conditions are not translation-invariant."
        )

    def test_no_self_pairs(self):
        """query_pairs must never return (i, i) — no galaxy paired with itself."""
        rng = np.random.default_rng(77)
        pos = rng.uniform(0, 1000.0, (1000, 3))
        tree = cKDTree(pos, boxsize=1000.0)
        pairs = tree.query_pairs(r=50.0)
        for i, j in pairs:
            assert i != j, "Self-pair found in query_pairs output"

    def test_no_duplicate_pairs(self):
        """query_pairs must return each pair exactly once (i < j always)."""
        rng = np.random.default_rng(88)
        pos = rng.uniform(0, 1000.0, (500, 3))
        tree = cKDTree(pos, boxsize=1000.0)
        pairs = list(tree.query_pairs(r=50.0))

        # Check i < j for all pairs (cKDTree contract).
        for i, j in pairs:
            assert i < j, f"Pair ({i},{j}) violates i < j ordering"

        # Check no duplicates.
        assert len(pairs) == len(set(pairs)), "Duplicate pairs in query_pairs output"

    def test_pair_count_independent_of_box_replication(self):
        """
        Pair count in a box of side L must equal the count in the equivalent
        periodic-boundary query on the same set of points.
        We verify this by placing all galaxies in one octant and checking that
        pairs near octant boundaries are found (i.e., PBC is actually active).
        """
        # Place galaxy A at (1, 0, 0) kpc and B at (999, 0, 0) kpc in a 1000 kpc box.
        # Without PBC, separation = 998 kpc (not found for r=5).
        # With PBC, separation = 2 kpc (should be found).
        pos = np.array([[1.0, 0.0, 0.0], [999.0, 0.0, 0.0]])
        tree = cKDTree(pos, boxsize=1000.0)
        pairs = list(tree.query_pairs(r=5.0))
        assert len(pairs) == 1, (
            "Periodic boundary pair not found — cKDTree may not be using PBC."
        )
