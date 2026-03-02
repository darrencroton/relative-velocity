"""
Statistical tests for the full pipeline.

These tests run the pipeline on generated test data and verify that the
recovered distributions match analytically known predictions.

Scientific basis for each test is cited in the docstring.

Maxwell distribution facts used throughout:
  If each component of dv ~ N(0, sigma), then |dv| ~ Maxwell(sigma)
  where Maxwell is the chi distribution with df=3.
  scipy.stats.maxwell(scale=sigma) == scipy.stats.chi(df=3, scale=sigma).

  Exact moments:
    Mean    = sigma * sqrt(8/pi)         ≈ 1.59577 * sigma
    E[X^2]  = 3 * sigma^2               (exact, no irrational prefactor)
    E[X^4]  = 15 * sigma^4              (exact)
    Var     = sigma^2 * (3 - 8/pi)      ≈ 0.45338 * sigma^2
    Skew    = 2*sqrt(2)*(16 - 5*pi) / (3*pi - 8)^(3/2)  ≈ 0.48573  (scale-invariant)

  Kinetic theory relative velocity:
    If v_i ~ N(0, sigma_i)^3 and v_j ~ N(0, sigma_j)^3 independently,
    then v_i - v_j ~ N(0, sqrt(sigma_i^2 + sigma_j^2))^3,
    so |v_i - v_j| ~ Maxwell(sigma_eff) with sigma_eff = sqrt(sigma_i^2 + sigma_j^2).
    Reference: Chapman & Cowling (1970), "The Mathematical Theory of Non-Uniform Gases"
"""

import os
import sys
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config as DEFAULT_CONFIG
from generate_test_data import generate_snapshot, SIGMA_V_PER_BIN
from pair_finder import find_pairs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_results():
    """
    Generate test data and run the pair finder for all redshifts.
    Cached at module scope so the slow generation runs once.
    """
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["mass_bin_width"] = 0.5   # 6 bins matching SIGMA_V_PER_BIN

    results = {}
    for z in cfg["redshifts"]:
        rng     = np.random.default_rng(seed=int(z * 100))
        catalog = generate_snapshot(z, cfg, rng)
        # generate_snapshot returns the full raw catalog dict; strip non-pipeline keys.
        pipeline_catalog = {
            k: catalog[k]
            for k in ("x", "y", "z", "vx", "vy", "vz", "log_stellar_mass")
        }
        pipeline_catalog["box_size"] = cfg["box_size"]
        pipeline_catalog["redshift"] = z

        # Apply mass selection as data_reader would.
        m    = pipeline_catalog["log_stellar_mass"]
        mask = (m >= cfg["log_mass_min"]) & (m <= cfg["log_mass_max"])
        for key in ("x", "y", "z", "vx", "vy", "vz", "log_stellar_mass"):
            pipeline_catalog[key] = pipeline_catalog[key][mask]

        pairs = find_pairs(pipeline_catalog, cfg)
        results[z] = pairs

    return cfg, results


# ---------------------------------------------------------------------------
# 1. KS tests: each mass bin's |dv| distribution must fit Maxwell(sigma_v)
# ---------------------------------------------------------------------------

class TestMaxwellDistribution:
    """
    Test that recovered |dv| distributions are consistent with the
    Maxwell distributions used to generate the test data.

    The KS statistic tests the full CDF shape, not just the mean.
    Reject at p < 0.001 to give ample margin for sampling noise
    while still catching systematic errors.
    """

    @pytest.mark.parametrize("z", [2.0, 3.0, 4.0, 5.0])
    @pytest.mark.parametrize("bin_idx", range(6))
    def test_ks_against_maxwell(self, pipeline_results, z, bin_idx):
        """
        KS test: p-value > 0.001 means the data are consistent with
        Maxwell(SIGMA_V_PER_BIN[bin_idx]) at the 0.1% significance level.

        Scientific note: cdf of Maxwell(scale=sigma) is
            F(v) = erf(v / (sqrt(2)*sigma)) - v*sqrt(2/pi)/sigma * exp(-v^2/(2*sigma^2))
        which is exact in scipy.stats.maxwell.
        """
        cfg, results = pipeline_results
        dv  = results[z]["delta_v"]
        mb  = results[z]["mass_bin"]
        mask = mb == bin_idx
        dv_bin = dv[mask]

        if len(dv_bin) < 30:
            pytest.skip(f"Too few pairs in bin {bin_idx} at z={z} ({len(dv_bin)} pairs)")

        sigma = SIGMA_V_PER_BIN[bin_idx]
        stat, pval = stats.ks_1samp(dv_bin, stats.maxwell(scale=sigma).cdf)
        assert pval > 0.001, (
            f"KS test rejected Maxwell(sigma={sigma}) for mass bin {bin_idx} at z={z}: "
            f"stat={stat:.4f}, p={pval:.4e}. "
            f"N={len(dv_bin)}, mean={dv_bin.mean():.1f} km/s, "
            f"predicted mean={sigma * np.sqrt(8/np.pi):.1f} km/s."
        )


# ---------------------------------------------------------------------------
# 2. Exact moment tests
# ---------------------------------------------------------------------------

class TestMaxwellMoments:
    """
    Test exact analytical moments of the Maxwell distribution.

    E[X^2] = 3*sigma^2 is the cleanest assertion: it involves no irrational
    constants and is exact by the moment formula for chi(df=3, scale=sigma):
        E[X^n] = sigma^n * 2^(n/2+1)/sqrt(pi) * Gamma((n+3)/2)
    For n=2: sigma^2 * 2^2/sqrt(pi) * Gamma(5/2) = sigma^2 * 4/sqrt(pi) * 3*sqrt(pi)/4 = 3*sigma^2. QED.
    """

    @pytest.mark.parametrize("bin_idx", range(6))
    def test_second_moment(self, pipeline_results, bin_idx):
        """
        E[|dv|^2] = 3 * sigma^2  (exact).

        We use a z-score test rather than a fixed percentage tolerance because
        sampling noise grows as sigma^2 / sqrt(N) and can legitimately exceed 5%
        in low-count bins.

        Derivation of variance of the sample estimator:
          Var[X^2] = E[X^4] - (E[X^2])^2 = 15*sigma^4 - 9*sigma^4 = 6*sigma^4
          Var[mean(X^2)] = 6*sigma^4 / N
          std[mean(X^2)] = sqrt(6) * sigma^2 / sqrt(N)

        By the CLT, the z-score is approximately N(0,1), so |z| < 4 fails at
        the ~1-in-16,000 level — strict enough to catch real bias while allowing
        for normal sampling fluctuations.
        """
        cfg, results = pipeline_results
        z = cfg["redshifts"][0]   # z=2 has largest sample; use it for this moment test
        dv   = results[z]["delta_v"][results[z]["mass_bin"] == bin_idx]
        if len(dv) < 50:
            pytest.skip(f"Too few pairs in bin {bin_idx}")

        N        = len(dv)
        sigma    = SIGMA_V_PER_BIN[bin_idx]
        expected = 3.0 * sigma**2
        measured = np.mean(dv**2)

        # Standard error of the sample mean of X^2.
        se = np.sqrt(6.0) * sigma**2 / np.sqrt(N)
        z_score = (measured - expected) / se

        assert abs(z_score) < 4.0, (
            f"E[|dv|^2] z-score = {z_score:.2f} (|z| must be < 4). "
            f"Measured = {measured:.1f}, expected 3*sigma^2 = {expected:.1f}, "
            f"SE = {se:.1f}  (sigma={sigma}, N={N}, bin={bin_idx})"
        )

    @pytest.mark.parametrize("bin_idx", range(6))
    def test_mean(self, pipeline_results, bin_idx):
        """
        E[|dv|] = sigma * sqrt(8/pi)  ≈ 1.5958 * sigma.
        Tolerance 5%.
        """
        cfg, results = pipeline_results
        z = cfg["redshifts"][0]
        dv = results[z]["delta_v"][results[z]["mass_bin"] == bin_idx]
        if len(dv) < 50:
            pytest.skip(f"Too few pairs in bin {bin_idx}")

        sigma    = SIGMA_V_PER_BIN[bin_idx]
        expected = sigma * np.sqrt(8.0 / np.pi)
        measured = dv.mean()
        rtol     = 0.05
        assert abs(measured - expected) / expected < rtol, (
            f"Mean |dv| = {measured:.1f}, expected {expected:.1f} (sigma={sigma}, bin={bin_idx})"
        )

    def test_skewness_scale_invariant(self, pipeline_results):
        """
        Maxwell skewness ≈ 0.4857 regardless of sigma.
        Scale-invariant, so we combine all bins for better statistics.
        Allow 20% relative tolerance for skewness (needs ~1000 points to estimate well).
        """
        cfg, results = pipeline_results
        z    = cfg["redshifts"][0]
        data = results[z]

        # Normalise each bin by its Maxwell sigma before combining, so all bins
        # contribute equally to the pooled skewness estimate.
        normalised_dv = []
        for b, sigma in enumerate(SIGMA_V_PER_BIN):
            dv = data["delta_v"][data["mass_bin"] == b]
            if len(dv) > 0:
                normalised_dv.append(dv / sigma)

        all_dv = np.concatenate(normalised_dv)
        skew   = stats.skew(all_dv)

        # Maxwell(sigma=1) skewness is the same for all sigma.
        maxwell_skew = stats.maxwell.stats(scale=1, moments="s")
        assert abs(skew - float(maxwell_skew)) < 0.20 * float(maxwell_skew), (
            f"Pooled normalised skewness = {skew:.4f}, Maxwell = {float(maxwell_skew):.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Redshift independence test
# ---------------------------------------------------------------------------

class TestRedshiftIndependence:
    """
    In the test data, SIGMA_V_PER_BIN is the same at every redshift.
    Therefore the |dv| distribution in each mass bin should be statistically
    identical across redshifts (up to sampling noise).
    A two-sample KS test between any pair of redshifts should not reject.
    """

    @pytest.mark.parametrize("bin_idx", range(6))
    def test_distributions_consistent_across_redshifts(self, pipeline_results, bin_idx):
        cfg, results = pipeline_results
        zs      = cfg["redshifts"]
        samples = []
        for z in zs:
            dv = results[z]["delta_v"][results[z]["mass_bin"] == bin_idx]
            samples.append(dv)

        # Test every redshift pair.
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                if len(samples[i]) < 20 or len(samples[j]) < 20:
                    continue
                stat, pval = stats.ks_2samp(samples[i], samples[j])
                assert pval > 0.001, (
                    f"KS two-sample test rejected z={zs[i]} vs z={zs[j]} for bin {bin_idx}: "
                    f"p={pval:.4e}. Distributions should be the same."
                )


# ---------------------------------------------------------------------------
# 4. Bulk velocity cancellation
# ---------------------------------------------------------------------------

class TestBulkVelocityCancellation:
    """
    The pair's bulk velocity cancels exactly when computing delta_v = v_i - v_j.
    Since in generate_test_data, v_secondary = v_primary + dv_intrinsic,
    the bulk motion of the pair centre-of-mass drops out:

        delta_v = v_secondary - v_primary = dv_intrinsic

    This must be independent of the bulk velocity dispersion SIGMA_BULK.
    Test by generating catalogs with SIGMA_BULK = 0 and SIGMA_BULK = 2000 km/s,
    then verifying the two resulting |dv| samples are from the same distribution.
    """

    def _make_pairs(self, sigma_bulk, n=2000, seed=99):
        """Generate N pairs with the given bulk sigma and return |dv| array."""
        import copy
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["mass_bin_width"] = 0.5

        rng = np.random.default_rng(seed)
        z   = 2.0

        # Temporarily patch SIGMA_BULK in generate_test_data.
        import generate_test_data as gtd
        orig_bulk = gtd.SIGMA_BULK
        orig_npairs = gtd.N_PAIRS
        orig_nfield = gtd.N_FIELD
        try:
            gtd.SIGMA_BULK = sigma_bulk
            gtd.N_PAIRS    = n
            gtd.N_FIELD    = 100
            cat = generate_snapshot(z, cfg, rng)
        finally:
            gtd.SIGMA_BULK = orig_bulk
            gtd.N_PAIRS    = orig_npairs
            gtd.N_FIELD    = orig_nfield

        pipeline_catalog = {
            k: cat[k] for k in ("x", "y", "z", "vx", "vy", "vz", "log_stellar_mass")
        }
        pipeline_catalog["box_size"] = cfg["box_size"]

        m    = pipeline_catalog["log_stellar_mass"]
        mask = (m >= cfg["log_mass_min"]) & (m <= cfg["log_mass_max"])
        for key in ("x", "y", "z", "vx", "vy", "vz", "log_stellar_mass"):
            pipeline_catalog[key] = pipeline_catalog[key][mask]

        pairs = find_pairs(pipeline_catalog, cfg)
        return pairs["delta_v"]

    def test_delta_v_independent_of_bulk_sigma(self):
        """
        Two-sample KS test between bulk=0 and bulk=2000 km/s must not reject.
        If the velocity subtraction has a sign error or indexing bug, the
        bulk motion will contaminate |dv| and produce different distributions.
        """
        dv_zero = self._make_pairs(sigma_bulk=0.0,    seed=42)
        dv_high = self._make_pairs(sigma_bulk=2000.0, seed=42)

        stat, pval = stats.ks_2samp(dv_zero, dv_high)
        assert pval > 0.01, (
            f"KS test rejected bulk=0 vs bulk=2000: stat={stat:.4f}, p={pval:.4e}. "
            f"Bulk velocity is bleeding into delta_v — check v_i - v_j sign convention."
        )


# ---------------------------------------------------------------------------
# 5. Cross-bin relative velocity test (kinetic theory)
# ---------------------------------------------------------------------------

class TestCrossBinKineticTheory:
    """
    Kinetic theory: if v_i ~ N(0, sigma_1)^3 and v_j ~ N(0, sigma_2)^3
    independently, then |v_i - v_j| ~ Maxwell(sigma_eff) where
        sigma_eff = sqrt(sigma_1^2 + sigma_2^2).

    We construct pairs where the primary has velocity dispersion sigma_1 and
    the secondary has an independent velocity dispersion sigma_2.
    (In our standard test data, v_secondary = v_primary + dv_intrinsic, so
    sigma_secondary is actually sigma_1^2 + sigma_intrinsic^2. Here we build
    a standalone catalog that isolates the two-population case.)

    Reference: Maxwell-Boltzmann relative speed distribution,
    Chapman & Cowling (1970) Chapter 1.
    """

    def test_cross_bin_sigma_eff(self):
        rng = np.random.default_rng(12345)
        N   = 5000

        sigma_1 = 80.0   # km/s per component, population 1
        sigma_2 = 150.0  # km/s per component, population 2
        sigma_eff = np.sqrt(sigma_1**2 + sigma_2**2)

        # Geometry design: ensure exactly N one-to-one cross-population pairs.
        #
        # Intra-population spacing (10 kpc) must exceed max_sep (5 kpc), so same-
        # population neighbors are never paired.  Cross-population partners are
        # offset by 0.5 kpc in y, well within max_sep.  Non-partner cross-pop
        # distance: sqrt(10^2 + 0.5^2) ≈ 10.01 kpc > max_sep → not found.
        # N=5000 galaxies at 10 kpc spacing span 50 Mpc; fits in 500 Mpc box.
        max_sep_kpc    = 5.0      # kpc
        spacing_mpc    = 0.010    # 10 kpc between intra-population neighbors
        cross_off_mpc  = 0.0005   # 0.5 kpc cross-population y-offset

        cfg = {
            **DEFAULT_CONFIG,
            "box_size":       500.0,
            "max_sep":        max_sep_kpc,
            "mass_ratio_min": 0.0,
            "mass_bin_by":    "primary",
            "mass_bin_width": 3.0,   # one bin [8,11) so everything is in bin 0
            "log_mass_min":   8.0,
            "log_mass_max":   11.0,
            "sep_bins":       [0, max_sep_kpc],
        }

        pos_1 = np.zeros((N, 3))
        pos_1[:, 0] = np.arange(1, N + 1) * spacing_mpc  # x: 0.01 … 50 Mpc

        pos_2 = pos_1.copy()
        pos_2[:, 1] += cross_off_mpc   # y offset: 0.5 kpc

        vel_1 = rng.normal(0, sigma_1, (N, 3))
        vel_2 = rng.normal(0, sigma_2, (N, 3))

        # log_mass: all pop-1 at 10.0, all pop-2 at 9.5 (ratio ≈ 0.316 > 0)
        lm_1 = np.full(N, 10.0)
        lm_2 = np.full(N, 9.5)

        all_pos = np.vstack([pos_1, pos_2])
        all_vel = np.vstack([vel_1, vel_2])
        all_lm  = np.concatenate([lm_1, lm_2])

        catalog = dict(
            x=all_pos[:, 0], y=all_pos[:, 1], z=all_pos[:, 2],
            vx=all_vel[:, 0], vy=all_vel[:, 1], vz=all_vel[:, 2],
            log_stellar_mass=all_lm,
            box_size=cfg["box_size"],
        )

        pairs = find_pairs(catalog, cfg)
        dv = pairs["delta_v"]
        # The geometry guarantees exactly N one-to-one cross-pop pairs.
        assert len(dv) == N, (
            f"Expected exactly {N} cross-population pairs, got {len(dv)}"
        )

        # KS test against Maxwell(sigma_eff).
        stat, pval = stats.ks_1samp(dv, stats.maxwell(scale=sigma_eff).cdf)
        assert pval > 0.001, (
            f"KS rejected Maxwell(sigma_eff={sigma_eff:.1f}) for cross-bin pairs: "
            f"stat={stat:.4f}, p={pval:.4e}. "
            f"Kinetic theory prediction: sigma_eff = sqrt({sigma_1}^2 + {sigma_2}^2) = {sigma_eff:.1f} km/s. "
            f"Measured mean = {dv.mean():.1f}, predicted = {sigma_eff * np.sqrt(8/np.pi):.1f} km/s."
        )

        # E[X^2] = 3 * sigma_eff^2 (exact; z-score test as in TestMaxwellMoments).
        N_pairs  = len(dv)
        expected = 3.0 * sigma_eff**2
        measured = np.mean(dv**2)
        se       = np.sqrt(6.0) * sigma_eff**2 / np.sqrt(N_pairs)
        z_score  = (measured - expected) / se
        assert abs(z_score) < 4.0, (
            f"E[|dv|^2] z-score = {z_score:.2f}. "
            f"Measured = {measured:.1f}, expected 3*sigma_eff^2 = {expected:.1f}"
        )


# ---------------------------------------------------------------------------
# 6. Mass ratio distribution (pipeline doesn't distort the input)
# ---------------------------------------------------------------------------

class TestMassRatioDistribution:
    """
    In the generated test data, mass ratios are drawn uniformly from
    [mass_ratio_min, 1.0]. The pipeline's mass ratio cut at mass_ratio_min
    must not distort the shape of the distribution above the threshold.
    A KS test against Uniform(mass_ratio_min, 1.0) verifies this.
    """

    def test_mass_ratio_is_uniform(self, pipeline_results):
        cfg, results = pipeline_results
        z     = cfg["redshifts"][0]
        ratio = results[z]["mass_ratio"]
        ratio_min = cfg["mass_ratio_min"]

        # All recovered ratios must be >= minimum.
        assert (ratio >= ratio_min - 1e-9).all(), "Mass ratio below cut found in results"

        # KS test against Uniform(ratio_min, 1.0).
        stat, pval = stats.ks_1samp(
            ratio,
            stats.uniform(loc=ratio_min, scale=1.0 - ratio_min).cdf,
        )
        assert pval > 0.001, (
            f"KS test rejected Uniform({ratio_min}, 1.0) for mass ratios: "
            f"stat={stat:.4f}, p={pval:.4e}. "
            f"The pipeline may be distorting the mass ratio distribution."
        )
