# Relative Velocity Statistics of Galaxy Pairs from Semi-Analytic Models

## 1. Scientific Context & Motivation

Galaxy mergers are a fundamental driver of galaxy evolution, particularly at high redshift (z ~ 2-5) when the cosmic merger rate peaks. Before two galaxies coalesce, they exist as close pairs whose relative velocities encode information about the gravitational dynamics of the system: the depth of the combined potential well, the orbital configuration, and whether the pair is gravitationally bound and destined to merge.

The **relative velocity** (or intervelocity) of galaxy pairs is a key observable and diagnostic:

- It constrains **merger timescales**: pairs with low relative velocities at small separations are more likely bound and will merge sooner.
- It probes **gravitational dynamics**: the velocity distribution at small separations (< 20 kpc) is shaped by the potential wells of the individual galaxies and their host halos.
- It is mass-dependent: more massive galaxies sit in deeper potential wells, leading to higher pairwise velocity dispersions.
- It evolves with redshift: structure growth and the expansion rate set the large-scale velocity field, while the merger rate and halo mass function evolve with cosmic time.

Observationally, close pair studies typically select galaxies with projected separations of 5-50 kpc and line-of-sight velocity differences of Delta_v < 300-500 km/s. The **3D intervelocity distribution** of galaxy pairs has been shown to exhibit a characteristic peak around ~130 km/s for isolated pairs at low redshift, a feature present in both LCDM simulations and MOND predictions (Pawlowski et al. 2022; Scarpa et al. 2022).

This project focuses on the **3D relative velocity distribution** of galaxy pairs in a mock catalog from a semi-analytic model (SAM), specifically examining how this distribution depends on:
- **Stellar mass** (in bins of 0.5 dex from 10^8 to 10^11 M_sun)
- **3D pair separation** (in bins of ~5 kpc width, from ~5 to 25 kpc)
- **Redshift** (z = 2, 3, 4, 5)


## 2. Key Definitions

### 2.1 Coordinate Systems in Cosmological Simulations

Semi-analytic models run on top of N-body dark matter simulations. Galaxy positions and velocities are typically stored in one of the following conventions:

- **Comoving coordinates** (x_com): Positions that factor out the expansion of the universe. A pair of galaxies at fixed comoving separation are being carried apart by Hubble expansion. Units are typically cMpc/h or ckpc/h.

- **Proper (physical) coordinates** (x_phys): The actual physical distance at a given epoch. Related to comoving coordinates by:
  ```
  x_phys = a * x_com
  ```
  where `a = 1/(1+z)` is the scale factor. Units are pMpc/h or pkpc/h.

- **Peculiar velocity** (v_pec): The velocity of a galaxy relative to the local Hubble flow. This is the "interesting" velocity component, driven by gravitational interactions. In simulations, velocities are often stored as peculiar velocities, sometimes scaled by sqrt(a) or a.

### 2.2 Relative Velocity of a Galaxy Pair

For a pair of galaxies i and j with position vectors **r_i**, **r_j** and peculiar velocity vectors **v_i**, **v_j**, the **3D relative velocity** is:

```
Delta_v = |v_i - v_j|
```

This is the magnitude of the vector difference of the peculiar velocities. Since both galaxies are at approximately the same redshift (and very close in space), the Hubble flow contribution to their relative motion is negligible at separations of < 25 kpc, and the relative velocity is dominated by the peculiar velocity difference.

**Important**: At separations of 10-25 kpc proper, the Hubble flow contribution is:
```
v_Hubble = H(z) * Delta_r_proper
```
At z = 3, H(z) ~ 300 km/s/Mpc, so for Delta_r = 25 kpc = 0.025 Mpc:
```
v_Hubble ~ 300 * 0.025 = 7.5 km/s
```
This is negligible compared to typical peculiar velocity differences of order 100+ km/s, confirming that **peculiar velocity differences alone** are sufficient for this analysis.

### 2.3 Pair Separation

The **3D physical (proper) separation** between two galaxies is:
```
Delta_r = a * |x_com_i - x_com_j|
```
where the comoving separation must account for periodic boundary conditions if the simulation uses a periodic box.

### 2.4 Decomposing Relative Velocity

The relative velocity vector can be decomposed into:
- **Radial component** (v_r): along the line connecting the two galaxies. Negative values indicate infall (approach); positive values indicate recession.
  ```
  v_r = (Delta_v . r_hat)
  ```
  where r_hat = (r_j - r_i) / |r_j - r_i|

- **Tangential component** (v_t): perpendicular to the connecting line.
  ```
  v_t = |Delta_v - v_r * r_hat|
  ```

The **total relative speed** is:
```
|Delta_v| = sqrt(v_r^2 + v_t^2)
```

For this project, the primary quantity of interest is the **total 3D relative speed** |Delta_v|, though the radial and tangential decomposition may also be informative (e.g., infalling pairs vs. flyby encounters).


## 3. The Calculation: Step by Step

### 3.1 Input Data

From the semi-analytic model output at each redshift snapshot (z = 2, 3, 4, 5), we need for each galaxy:
- **3D position** (x, y, z) -- typically in comoving coordinates (cMpc/h or ckpc/h)
- **3D velocity** (vx, vy, vz) -- typically peculiar velocities (km/s)
- **Stellar mass** (M_star) -- in solar masses (M_sun) or log10(M_sun)

Additional metadata needed:
- **Simulation box size** (L_box) -- for periodic boundary conditions
- **Cosmological parameters** (h, Omega_m, Omega_Lambda) -- for unit conversions
- **Scale factor** (a) or **redshift** (z) of the snapshot

### 3.2 Pair Finding Algorithm

For each redshift snapshot:

1. **Select galaxies** in the desired stellar mass range (10^8 < M_star/M_sun < 10^11).

2. **Find all pairs** with 3D proper separation below the maximum threshold (e.g., 25 kpc):
   - Convert comoving positions to proper: `r_phys = a * r_com`
   - For each pair (i, j), compute separation with **periodic boundary conditions**:
     ```
     dx = x_i - x_j
     dx = dx - L_box * round(dx / L_box)   # minimum image convention
     ```
     (repeat for dy, dz, using the comoving box size)
   - Compute proper separation: `Delta_r = a * sqrt(dx^2 + dy^2 + dz^2)`

3. **Avoid double-counting**: Each pair (i,j) should be counted once. Use i < j indexing or divide pair counts by 2.

4. **Compute relative velocity** for each pair:
   ```
   Delta_v = sqrt((vx_i - vx_j)^2 + (vy_i - vy_j)^2 + (vz_i - vz_j)^2)
   ```
   (Ensure velocities are in consistent units, typically km/s peculiar.)

5. **Bin pairs** by:
   - **Stellar mass bin**: assign each pair to a mass bin based on the mass of one or both galaxies (see Section 4)
   - **Separation bin**: 0-10, 10-15, 15-20, 20-25 kpc (proper)
   - The **relative velocity** |Delta_v| is histogrammed within each (mass, separation) bin

### 3.3 Computational Considerations

- **Pair finding efficiency**: A naive O(N^2) search over all galaxy pairs is expensive. For N ~ 10^5-10^7 galaxies, use a **KD-tree** or **ball tree** spatial index (e.g., `scipy.spatial.cKDTree`) to efficiently find pairs within a given radius. The `query_pairs()` or `query_ball_tree()` methods are well-suited for this.

- **Periodic boundary conditions**: `scipy.spatial.cKDTree` supports periodic boxes via the `boxsize` parameter, which handles the minimum image convention automatically.

- **Memory**: For very dense fields, the number of pairs within 25 kpc can be large. Consider processing in chunks or using pair iterators.


## 4. Mass Binning Strategy

### 4.1 Stellar Mass Bins

The requested mass bins are 0.5 dex wide, spanning 10^8 to 10^11 M_sun:

| Bin | log10(M_star/M_sun) range | M_star range |
|-----|---------------------------|--------------|
| 1   | 8.0 - 8.5                | 10^8.0 - 10^8.5 |
| 2   | 8.5 - 9.0                | 10^8.5 - 10^9.0 |
| 3   | 9.0 - 9.5                | 10^9.0 - 10^9.5 |
| 4   | 9.5 - 10.0               | 10^9.5 - 10^10.0 |
| 5   | 10.0 - 10.5              | 10^10.0 - 10^10.5 |
| 6   | 10.5 - 11.0              | 10^10.5 - 10^11.0 |

### 4.2 Which Mass Defines the Bin?

An important choice: how do we assign a pair to a mass bin? Common options:

1. **Primary (more massive) galaxy mass**: bin by the stellar mass of the more massive member. This is the most common convention in close-pair literature.
2. **Secondary (less massive) galaxy mass**: less common, but relevant if studying accretion onto a fixed-mass primary.
3. **Pair mass (M1 + M2)**: the total stellar mass of the pair. Useful if the combined potential is what matters.
4. **Both members must be in the bin**: only count pairs where both galaxies fall in the same mass bin. This restricts to roughly equal-mass (major) pairs within each bin.
5. **Either member**: count a pair in a bin if either galaxy falls in that bin (pairs may appear in multiple bins).

**Recommendation**: Option 1 (primary galaxy mass) is the most standard and interpretable. However, since you are interested in the potential well affecting velocities, Option 3 (total pair mass) or Option 4 (both in bin) may also be physically motivated. This is a choice we should discuss.

### 4.3 Mass Ratio Considerations

At separations of 10-25 kpc, most pairs will be in the process of merging or are at least interacting. The mass ratio of the pair matters:
- **Major mergers**: mass ratio > 1:4 (i.e., M2/M1 > 0.25)
- **Minor mergers**: mass ratio 1:4 to 1:10
- **Very minor mergers**: mass ratio < 1:10

Do we want to restrict to major mergers, or include all mass ratios? This affects both the physics and the pair statistics.


## 5. Separation Bins

The project focuses on **proper (physical) separations** at the given redshift:

| Bin | Separation range (pkpc) | Notes |
|-----|------------------------|-------|
| 1   | 0 - 10                 | Very close; likely merging or in final inspiral |
| 2   | 10 - 15                | Peak of observed pair separations |
| 3   | 15 - 20                | Still within mutual potential influence |
| 4   | 20 - 25                | Outskirts of interaction zone |

At these small separations (< 25 pkpc), the galaxy pairs are deep within a shared dark matter halo or are at least strongly interacting. The velocities will be dominated by the local gravitational potential rather than large-scale flows.

**Context**: The observed pairs have separations of ~5-25 kpc, with most at 10-15 kpc. This motivates the bin choices. Pairs at < 10 kpc are extremely close -- potentially already in the coalescence phase, where the semi-analytic model's treatment of merging (which often uses analytic prescriptions like dynamical friction timescales rather than resolving the orbital dynamics) may introduce model-dependent artifacts.

**Caution**: SAMs typically merge satellite galaxies onto centrals using a dynamical friction clock rather than tracking their actual orbits. This means that very close pairs (< 10 kpc) in the SAM may not have physically realistic relative velocities -- the velocity information comes from the underlying N-body halo/subhalo dynamics, not from the SAM's merger prescription. The reliability of the velocity information at these separations depends on the resolution of the N-body simulation and how subhalo tracking is handled.


## 6. Redshift Snapshots

The analysis spans z = 2, 3, 4, 5, corresponding to lookback times of roughly 10.3, 11.5, 12.1, and 12.5 Gyr (for standard Planck cosmology). Key considerations at each redshift:

| z | a = 1/(1+z) | H(z) approx (km/s/Mpc) | Key physical context |
|---|-------------|------------------------|---------------------|
| 2 | 0.333       | ~230                   | Near peak of cosmic star formation; abundant massive galaxies |
| 3 | 0.250       | ~300                   | High merger rate epoch; significant structure formation |
| 4 | 0.200       | ~380                   | Early massive galaxy assembly; fewer high-mass objects |
| 5 | 0.167       | ~460                   | Very early epoch; low-mass galaxies dominate; sparse statistics for high-mass pairs |

At higher redshifts, the galaxy stellar mass function shifts to lower masses. The highest mass bins (10^10.5 - 10^11) may have very few or no pairs at z = 4-5, depending on the simulation volume and SAM. Statistics will be best at z = 2 and progressively sparser toward z = 5.


## 7. Expected Physical Behavior

### 7.1 Velocity Distribution Shape

The relative velocity distribution for close pairs is expected to be:
- **Peaked** at some characteristic velocity set by the typical orbital speed within the potential well
- **Asymmetric / positively skewed**: a hard lower bound near zero, with a tail to high velocities from unbound flyby encounters or high-energy orbits
- Roughly consistent with a **Maxwell-Boltzmann-like** distribution for the speed (magnitude of 3D velocity difference), since it's the magnitude of a 3-component vector

### 7.2 Mass Dependence

- **Higher mass bins** should show broader distributions shifted to higher velocities, reflecting deeper potential wells
- The characteristic velocity scales roughly as the circular velocity: v_c ~ sqrt(G*M/r), so more massive systems (including DM halo) have higher v_c
- Typical values: v_c ~ 50-300 km/s for halos hosting galaxies in the 10^8 - 10^11 M_star range

### 7.3 Separation Dependence

- **Closer pairs** (< 10 kpc) may show higher relative velocities as they are deeper in the potential well (orbital speed increases inward, ~1/sqrt(r) in a point-mass regime)
- **Wider pairs** (20-25 kpc) may have a broader distribution including both bound pairs (lower velocities) and chance superpositions or flyby encounters (higher velocities)

### 7.4 Redshift Evolution

- At fixed stellar mass, the velocity distribution may shift with redshift due to:
  - Evolution of the halo mass -- stellar mass relation
  - Denser environments at higher z (smaller physical separations for the same comoving separation)
  - Higher merger rates and infall velocities at higher z
  - Differences in the dynamical state of halos (more unrelaxed at high z)


## 8. Output: Histograms

The primary deliverable is a set of histograms of **|Delta_v|** (3D relative speed in km/s) for each combination of:
- 6 mass bins x 4 separation bins x 4 redshifts = **96 histograms**

These can be organized as:
- A grid of panels: rows = mass bins, columns = redshifts, with separation bins as different colored lines/histograms within each panel
- Or: rows = separation bins, columns = redshifts, with mass bins as colors
- Summary statistics per bin: median, mean, standard deviation, and pair count N_pairs

### 8.1 Histogram Binning

For the velocity axis, reasonable choices:
- Linear bins of 10-25 km/s width, spanning 0 to ~1000 km/s (or adaptive based on the data)
- Or logarithmic bins if the distribution spans many orders of magnitude

### 8.2 Normalization

- **Raw counts**: useful for understanding statistics (how many pairs per bin)
- **Probability density**: normalized so the area under the histogram = 1, enabling comparison across bins with different pair counts


## 9. Practical Considerations

### 9.1 Unit Philosophy

The pipeline code takes all values at face value: positions in Mpc, velocities in km/s, stellar masses in log10(M_sun). Any cosmological unit conversions (h-factors, scale factors, comoving-to-proper) are the responsibility of the data reader, applied once at ingestion time so the rest of the code stays clean and general.

### 9.2 Periodic Boundary Conditions

The simulation box is periodic. When computing separations, use the minimum image convention:
```python
dx = pos_i - pos_j
dx = dx - box_size * np.round(dx / box_size)
```
This ensures we find the shortest distance between two galaxies, wrapping around the box edges. `scipy.spatial.cKDTree` handles this automatically via its `boxsize` parameter.

### 9.3 Subhalo vs. Central Galaxies

In SAMs, galaxies are classified as **central** (at the center of a host halo) or **satellite** (orbiting within a host halo). Close pairs at < 25 kpc are most likely:
- A central + satellite within the same halo
- Two satellites within the same massive halo
- Two centrals of separate (but nearby) halos (less common at such small separations)

The physical interpretation differs: central-satellite pairs reflect infall dynamics within a halo, while central-central pairs reflect halo-halo approach velocities.


## 10. Resolved Design Decisions

| Question | Decision |
|----------|----------|
| SAM / simulation | SAGE on Millennium (data reader to be added later) |
| Velocity convention | km/s, taken at face value |
| Position convention | Mpc, taken at face value (user handles any h/a conversions) |
| Mass bin assignment | By primary (most massive) galaxy; configurable |
| Mass ratio cut | 1:10 (mass_ratio_min = 0.1); configurable |
| Central/satellite | Not distinguished initially; may add later |
| Velocity decomposition | 3D total speed only initially; radial/tangential may be added later |
| Stellar mass format | log10(M_star/M_sun) throughout |


## References

- Ventou et al. (2019) - [New criteria for close pair selection from simulations](https://www.aanda.org/articles/aa/full_html/2019/11/aa35597-19/aa35597-19.html)
- Kitzbichler & White (2008) - [Calibration of close pair abundance and merger rate](https://arxiv.org/abs/0804.1965)
- Jiang et al. (2014) - [Scaling relation between merger rate and pair count](https://iopscience.iop.org/article/10.1088/0004-637X/790/1/7)
- Lagos et al. (2021) - [Statistics of galaxy mergers: bridging theory and observation](https://arxiv.org/abs/2107.05601)
- Pawlowski et al. (2022) - [Intervelocity of galaxy pairs in LCDM](https://arxiv.org/abs/2207.09468)
- Scarpa et al. (2022) - [Orbital velocity of isolated galaxy pairs](https://academic.oup.com/mnras/article/512/1/544/6542452)
- Croton et al. (2016) - [SAGE: Model Calibration and Basic Results](https://arxiv.org/abs/1601.04709)
- Davis & Peebles (1983) - Pairwise velocity dispersion estimator (ApJ, 267, 465)
