"""
All tunable parameters for the relative velocity pipeline.
Passed explicitly to every function -- no global state, no hardcoded values elsewhere.
"""

config = dict(
    # Simulation
    box_size        = 500.0,          # Mpc (same units as input positions)
    redshifts       = [2.0, 3.0, 4.0, 5.0],

    # Galaxy selection
    log_mass_min    = 8.0,            # log10(M_star / M_sun)
    log_mass_max    = 11.0,
    mass_ratio_min  = 0.1,            # min M_secondary / M_primary (1:10 cut)

    # Binning
    mass_bin_width  = 0.5,            # dex  →  6 bins: [8.0,8.5), [8.5,9.0), ..., [10.5,11.0)
    sep_bins        = [0, 10, 15, 20, 25],  # kpc, separation bin edges  →  4 bins
    vel_bin_width   = 20.0,           # km/s, histogram bin width
    vel_max         = 1000.0,         # km/s, upper limit for velocity histograms

    # Pair finding
    max_sep         = 25.0,           # kpc, max 3D separation

    # Which galaxy's mass defines the bin for each pair?
    # Options: "primary" (more massive), "secondary", "mean", "total"
    mass_bin_by     = "primary",

    # File paths
    data_dir        = "data/",
    results_dir     = "results/",
    figures_dir     = "figures/",
)
