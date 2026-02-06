# DELVE Ubercalibration Implementation Plan

## Goal

Relative photometric calibration (ubercal) of the full DELVE footprint (~17,000 deg² in griz) using **only DECam internal overlap data** — no external catalogs (PS1, Gaia, SkyMapper, Refcat2) as calibration anchors. Absolute calibration is inherited from the DES footprint, which is already FGCM-calibrated to <3 mmag. Target: **5–10 mmag relative uniformity** across the full southern sky.

## Architecture Overview

The algorithm follows Padmanabhan et al. (2008) with adaptations for DECam's focal plane geometry and DELVE's heterogeneous multi-program dataset. The core idea: if the same star is observed in two different exposures, the difference in measured instrumental magnitudes constrains the difference in zero-points between those exposures. With enough overlapping observations, we can solve a large sparse linear system for all per-exposure (or per-CCD) zero-point corrections simultaneously.

### Calibration model

For each detection of star `s` in exposure `e` on CCD `c`, the calibrated magnitude is:

```
m_cal = m_inst + ZP_{e,c}
```

Where:
- `m_inst` = instrumental magnitude (from SExtractor PSF or aperture photometry)
- `ZP_{e,c}` = per-CCD-per-exposure zero-point (one scalar per CCD per exposure, ~62 × 161,000 ≈ 10M parameters)

This single parameter absorbs **everything** — atmosphere, airmass, extinction, clouds, flat-field error, mirror reflectivity, CCD gain drift — into one number. No atmospheric model, no airmass term, no color term. The overlapping stars tell you the zero-point difference between any two CCD-exposures; you don't need to know *why* they differ. This is the key philosophical difference from FGCM, which forward-models the physics. With ~10M free parameters and billions of constraints, the system is massively over-determined and no parameter reduction is needed.

Flat-field refinement (Phase 4) is applied as a second pass after the initial solve.

---

## Step-by-Step Implementation

### Phase 0: Data Ingestion and Star Catalog Construction

**Data source: NOIRLab Astro Data Lab (remote queries)**

No local data files are available. All data is queried from the NOIRLab Astro Data Lab using the `astro-datalab` Python client. The key tables are:

| Table | Content | Key columns |
|-------|---------|-------------|
| `nsc_dr2.meas` | Single-epoch detections (~billions) | `objectid`, `exposure`, `ccdnum`, `mag_auto`, `magerr_auto`, `mjd`, `ra`, `dec`, `x`, `y`, `flags`, `class_star`, `filter` |
| `nsc_dr2.chip` | Per-CCD-per-exposure metadata + ZP | `exposure`, `ccdnum`, `zpterm`, `zptermerr`, `fwhm` |
| `nsc_dr2.exposure` | Per-exposure metadata | `exposure`, `expnum`, `instrument`, `airmass`, `filter`, `mjd` |
| `des_dr2.y6_gold_zeropoint` | DES FGCM zero-points (anchor) | `expnum`, `ccdnum`, `band`, `mag_zero`, `sigma_mag_zero`, `source` |
| `delve_dr2.objects` | Coadded objects (for star selection) | `quick_object_id`, `spread_model_*`, `wavg_mag_psf_*`, `flags_*` |
| `nsc_dr2.x1p5__object__delve_dr2__objects` | NSC↔DELVE cross-match | `id1` (NSC objectid), `id2` (DELVE object_id) |

**Why NSC DR2 instead of DELVE DR2?** DELVE DR2 on Data Lab only publishes coadded objects — no single-epoch detections. NSC DR2 has single-epoch measurements for all DECam data (~341K exposures), including all DES and DELVE exposures. NSC provides `objectid` grouping (detections already linked to unique objects), so no cross-matching is needed.

**CCD exclusions:**
- **CCDNUM 61 (N30):** Dead since November 2012 (over-illumination event during commissioning). Exclude entirely.
- **CCDNUM 31 (S7):** Amplifier A has unpredictable, time-variable gain at the ~0.3% level throughout DECam's lifetime. The per-CCD-per-exposure parameterization absorbs this, but flag this CCD in diagnostics — it will show higher ZP scatter than other CCDs.

**Tasks:**
1. Query NSC DR2 from NOIRLab Astro Data Lab using the `dl.queryClient` Python module. Query by HEALPix pixel (nside=32) for chunking.
2. Select high-quality stellar detections from `nsc_dr2.meas`:
   - `flags = 0` (clean detections)
   - `class_star > 0.8` (point sources — NSC does not have `spread_model`)
   - `magerr_auto < 0.05` mag (S/N > 20)
   - Magnitude range: `17 < mag_auto < 20` (see notes below)
   - `ccdnum != 61`
   - Filter on `instrument = 'c4d'` via join with `nsc_dr2.exposure`

**Magnitude cut rationale:**
- **Bright limit (17 mag):** Avoids saturation, nonlinearity, and brighter-fatter effects in DECam.
- **Faint limit (20 mag):** Avoids Malmquist bias — near the detection limit, stars that scatter bright are preferentially detected, biasing solved zero-points. With DELVE's 5σ depth of ~23.5 mag, a cut at 20 mag provides a ~3.5 mag buffer. **Sensitivity test:** Re-run with faint limits of 19.5, 20.0, and 20.5 to verify that solved zero-points are stable.

3. Strip the existing zero-point to recover instrumental magnitudes:
   ```
   m_inst = mag_auto - chip.zpterm
   ```
   where `chip.zpterm` is from `nsc_dr2.chip` (joined on `exposure` + `ccdnum`). Note: NSC applies the same `zpterm` to all CCDs in an exposure. This is fine — our ubercal will recover per-CCD variations.

4. **Object grouping:** NSC DR2 already links detections to unique objects via `objectid`. No cross-matching needed. Each unique `objectid` is a unique star.

5. **Detection cap:** For stars with more than 25 detections in a given band, randomly subsample to 25 detections. This caps the per-star contribution to the normal equations at a 25×25 block (300 pairs) without meaningfully degrading the solution — a star observed 25 times already provides excellent constraints.

6. Cache downloaded query results locally as parquet files at `/Volumes/External5TB/DELVE_UBERCAL/cache/` to avoid re-querying. Check cache before issuing new queries.

7. Store final output as parquet: `(objectid, expnum, ccdnum, band, m_inst, m_err, mjd, ra, dec, x, y)`

**Output:** A matched star catalog with ~hundreds of millions of individual detections linked to ~hundreds of millions of unique stars.

**Memory management (16 GB RAM constraint):** Process in chunks by HEALPix region (nside=32, ~3 deg² pixels). Query one pixel at a time from Data Lab, apply cuts, write to parquet cache. Never load all detections into memory at once.

### Phase 1: Build the Overlap Graph

For each pair of exposures that share at least one star, we have a calibration constraint. The "overlap graph" has exposures (or CCD-exposures) as nodes and shared stars as edges.

**Tasks:**
1. For each unique star observed N times, record which (expnum, ccdnum) pairs observed it. Each star contributes edges between all pairs of CCD-exposures that observed it.
2. Do NOT materialize all N*(N-1)/2 pairs per star. Instead, store the list of CCD-exposures per star. The normal equations matrix will be accumulated star-by-star in Phase 2.
3. Check graph connectivity:
   - Use union-find (disjoint set) algorithm to identify connected components
   - The DES footprint must be in the same connected component as the rest of DELVE — this is what transfers the absolute calibration
   - Flag any disconnected components (isolated archival pointings)
   - Report statistics: number of components, size of largest component, list of disconnected exposures

**Critical diagnostic:** If large regions of the sky are disconnected from DES, the ubercal solution will have unconstrained global offsets in those regions.

**Policy on disconnected components:** Drop them. Only calibrate CCD-exposures that belong to the connected component containing DES. Disconnected islands (typically isolated deep PI pointings) cannot inherit the DES absolute scale and are excluded from the final catalog. Report the dropped exposures and their sky coverage for documentation.

**Expected outcome:** Given DELVE's 161,000+ exposures from 278 programs, the vast majority should form a single connected component through the DES footprint + DECaLS + DELVE-WIDE tiling overlaps. Isolated PI programs may create small disconnected islands.

### Phase 2: Solve the Linear System

The ubercal problem reduces to a weighted least-squares minimization:

```
minimize: sum_pairs w_ij * (m_inst_i - m_inst_j - ZP_i + ZP_j)^2
```

Where `w_ij = 1/(sigma_i^2 + sigma_j^2)` is the inverse-variance weight from photometric errors.

This is equivalent to solving the normal equations:

```
(A^T W A) * zp = A^T W * dm
```

Where:
- `A` is the (implicit) sparse design matrix (N_pairs × N_ccd_exposures), never materialized
- `W` is the diagonal weight matrix
- `dm` is the vector of magnitude differences
- `zp` is the vector of zero-points we're solving for

**Building the normal equations star-by-star:**

The design matrix A has billions of rows (one per detection pair) and cannot be materialized. Instead, form the normal equations matrix `A^T W A` (sparse, N×N where N ≈ 10M) and the RHS vector `A^T W dm` (dense, length N) by accumulating contributions star-by-star:

For a star with detections in CCD-exposures [i1, i2, ..., in] with instrumental mags [m1, m2, ..., mn] and errors [e1, e2, ..., en] (where n ≤ 25 after the detection cap):
- For each pair (ia, ib): weight `w = 1/(ea^2 + eb^2)`
- Add `w` to diagonal entries `(ia,ia)` and `(ib,ib)`
- Subtract `w` from off-diagonal entries `(ia,ib)` and `(ib,ia)`
- Add `w*(ma-mb)` to `RHS[ia]` and `w*(mb-ma)` to `RHS[ib]`

This produces a weighted graph Laplacian — symmetric positive semi-definite, extremely sparse (~100M nonzero entries for ~10M parameters).

**Solver: Conjugate Gradient (CG)**

Use `scipy.sparse.linalg.cg` on the normal equations. CG is the natural choice for symmetric positive definite sparse systems and is how Padmanabhan et al. (2008) solved the SDSS ubercal. Do NOT use LSMR/LSQR — those operate on the original design matrix A, which we never materialize.

**Two solve modes:**

The solver supports two modes, controlled by config:

1. **Unanchored solve (`mode: unanchored`)** — for validation:
   - Add tiny Tikhonov regularization to the diagonal (`1e-10 * I`) to make the graph Laplacian non-singular
   - Solve with CG
   - After convergence, shift the entire solution so that `mean(ZP_solved[DES]) = mean(ZP_FGCM[DES])` — this sets the absolute scale using the broadband DES mean without constraining any individual DES CCD-exposure
   - All DES zero-points are determined purely by the overlap network; only the global offset is pinned
   - This solve is used for Validation Test 0 (FGCM comparison)

2. **Anchored solve (`mode: anchored`)** — for production:
   - For each DES CCD-exposure `i`, add `des_anchor_weight` (1e6) to diagonal entry `(i,i)` and `des_anchor_weight * ZP_FGCM[i]` to `RHS[i]`
   - This strongly pins each DES CCD-exposure to its FGCM value while allowing non-DES exposures to float
   - This is the production calibration product

**Convergence criteria:** Relative residual < 1e-5 (sufficient for 5–10 mmag target; tighter tolerances may stall without improving the calibration).

**Output:** A table of solved zero-points `ZP_solved` for every CCD-exposure in both modes. The correction applied to the existing calibration is:
```
delta_ZP = ZP_solved - ZP_current
m_calibrated = m_inst + ZP_solved = m_inst + ZP_current + delta_ZP
```

### Phase 3: Outlier Rejection and Iterative Refinement

The initial solution will be contaminated by:
- Variable stars (RR Lyrae, eclipsing binaries, etc.)
- Cosmic rays / artifacts misidentified as detections
- CCD-edge effects and bad columns
- Non-photometric exposures with rapid transparency variations (clouds)

**Iterative sigma-clipping procedure:**
1. Compute residuals for each detection: `r_i = m_inst_i + ZP_i - <m_star>` where `<m_star>` is the weighted mean magnitude of the star across all its detections
2. For each star, compute the chi-squared of its residuals: `chi^2 = sum(r_i^2 / sigma_i^2)`, `dof = N_detections - 1`
3. Flag stars with chi^2 / dof > 3 (likely variables or artifacts)
4. Flag individual detections with |residual| > 5*sigma (catastrophic outliers)
5. Remove flagged detections from the input catalog
6. Re-solve (Phase 2)
7. Repeat until convergence (typically 3–5 iterations)

**Additional quality cuts after first solve:**
- Flag exposures where the solved ZP deviates by > 0.3 mag from the nightly median (likely cloudy)
- Flag CCDs where the ZP scatter across stars is anomalously large (bad flat-field or readout issues)

### Phase 4: Flat-Field Refinement (Star Flat)

After solving for per-CCD-per-exposure zero-points, systematic residuals as a function of position within each CCD reveal flat-field errors.

**Approach:**
1. For each CCD (1–62, excluding CCDNUM 61), bin residuals by (x, y) pixel position across all exposures
2. Fit a low-order 2D Chebyshev polynomial (order 3, configurable) to the binned median residuals — this is the "star flat" or "illumination correction"
3. Apply the correction and re-solve for zero-points
4. The DECam star flat is known to have ~5 mmag structure, so this step matters at your target precision

**Instrumental epoch boundaries for flat-field corrections:**

The illumination correction may change when the instrument configuration changes. Fit separate corrections per epoch. Known DECam epoch boundaries from the literature:

| MJD | Date | Event | Affected CCDs | Source |
|-----|------|-------|---------------|--------|
| ~56232 | 2012 Nov | CCDNUM 61 (N30) dead — over-illumination | 61 (permanent) | NOIRLab DECam status |
| ~56404 | 2013 Apr 22 | g-band baffling installed | All (g-band) | Morganson et al. 2018 |
| ~56516 | 2013 Aug 12 | r,i,z,Y baffling installed | All (rizY) | Morganson et al. 2018 |
| 56626 | 2013 Nov 30 | CCDNUM 2 (S30) failure | 2 | NOIRLab DECam status |
| ~56730 | 2014 Mar 14 | Shutter/filter aerosolization | All | Morganson et al. 2018 |
| 57751 | 2016 Dec 29 | CCDNUM 2 (S30) recovery | 2 | NOIRLab DECam status |
| ~58350 | 2018 Sep | CCDNUM 41 (N10) hardware intervention | 41 | CTIO config log |

**DES processing epoch boundaries** (from Morganson et al. 2018, Table 2) provide a finer grid for the DES-era data: SVE1 (56240–56345), Y1E1 (56519–56624), Y1E2 (56626–56700), Y2E1 (56875–56991), Y2E2 (56995–57160), Y3E1 (57234–57441), Y4E1 (57613–57753), Y4E2 (57754–57802), Y5E1 (57980–58000).

**Practical approach:** Start with the hardware-driven boundaries above. If residuals show epoch-dependent structure at finer timescales, subdivide using the DES processing epochs. For post-DES community-era data (2019+), epoch boundaries are not published — treat as a single epoch unless residuals demand otherwise.

**Alternative:** Use the existing DECam star flats from DES if available. These were derived from heavily dithered observations and are well-characterized.

### Phase 5: Construct the Calibrated Catalog

1. Apply the final zero-point corrections to all detections
2. For each unique star, compute the weighted-mean calibrated magnitude across all detections
3. Compute the RMS scatter per star as a function of magnitude — this is the empirical photometric repeatability and should approach the photon-noise limit for bright stars
4. Write out the final calibrated object catalog with:
   - `MAG_UBERCAL_{G,R,I,Z}` — the ubercal-calibrated magnitudes
   - `MAGERR_UBERCAL_{G,R,I,Z}` — error on the weighted mean
   - `NOBS_{G,R,I,Z}` — number of detections used
   - `CHI2_{G,R,I,Z}` — chi-squared of the detections around the mean (variability indicator)

---

## Validation

### Test 0: FGCM vs Ubercal comparison (unanchored solve)

**This is the most fundamental consistency check.** Using the unanchored solve (where DES CCD-exposure ZPs are determined purely by the overlap network, with only the global mean pinned to FGCM):
- For every DES CCD-exposure, compute `ZP_FGCM - ZP_ubercal_unanchored`
- Map this difference spatially across the DES footprint (HEALPix nside=64)
- Plot as a function of airmass, MJD, and CCDNUM
- If the overlap network and FGCM are consistent, this map should be spatially flat with scatter ≤ 5 mmag and no coherent structure
- Coherent spatial patterns would indicate either FGCM systematics or limitations of the per-CCD model
- This comparison is genuine because the unanchored solve has NO per-exposure FGCM information — the two methods are independent except for the global mean

### Test 1: Internal photometric repeatability
Plot RMS of stellar magnitudes vs. magnitude. For each star with N ≥ 3 detections, compute the weighted RMS of individual detections around the weighted mean. Should reach ~5 mmag floor for bright stars (17–18 mag), rising as photon noise toward fainter magnitudes. This is the primary metric of ubercal success.

### Test 2: Before/after comparison (DR2 vs ubercal)
For every star in the catalog, compute `MAG_DR2 - MAG_UBERCAL` (i.e., the original DELVE DR2 calibrated magnitude minus the new ubercal magnitude). Map this difference in HEALPix pixels across the sky. Look for:
- The **dec = −30° discontinuity**: In DR2, a ~10 mmag step exists at this declination because ATLAS Refcat2 switches from PS1-based to SkyMapper-based photometry. In the ubercal solution, this step should appear as a ~10 mmag correction in the `MAG_DR2 - MAG_UBERCAL` difference map — the ubercal removes the discontinuity, so the difference map reveals it.
- **Program boundaries**: Sharp edges in the difference map at the boundaries of different DECam programs indicate that ubercal is correcting program-to-program zero-point offsets.
- **DES interior**: Inside the DES footprint, `MAG_DR2 - MAG_UBERCAL` should be near zero (since DES FGCM is the anchor for both), confirming that ubercal preserves the DES calibration.

### Test 3: Gaia XP synthetic photometry comparison (external, not used for calibration)
Transform ubercal magnitudes to the Gaia G band using color transformations (following the procedure in Drlica-Wagner et al. 2022, Figure 4), and compare to Gaia DR3 photometry for bright stars. Map the median offset `G_predicted - G_Gaia` in HEALPix pixels. This replicates the DELVE DR1/DR2 validation diagnostic:
- In DR2, this map shows a ~10 mmag step at dec = −30°.
- After ubercal, the step should vanish — the map should be spatially smooth with scatter ≤ 5–7 mmag.
- This is the most visually compelling validation: a direct apples-to-apples comparison with the published DELVE diagnostic figure, showing the discontinuity removed.

### Test 4: Stellar locus width vs. position
In color-color space (g−r vs r−i), the main-sequence stellar locus should be tight and position-independent. Measure the locus width (perpendicular scatter) in HEALPix pixels across the sky. Spatial variations in locus width indicate calibration non-uniformity. After ubercal, the locus width should be uniform to within ~5 mmag.

### Test 5: DES boundary continuity
Select stars in a narrow strip spanning the DES footprint boundary. Compare the ubercal magnitudes of stars just inside DES (calibrated by FGCM) to stars just outside (calibrated by the ubercal overlap network). There should be no discontinuity — the overlap-based solution should seamlessly connect to the FGCM-anchored region.

---

## Computational Considerations

### Scale estimates
- ~161,000 exposures × 62 CCDs = ~10M CCD-exposure zero-points to solve for
- ~2.5 billion detections, but after quality cuts + 25-detection cap probably ~200M–500M stellar detections
- The design matrix A is implicit (never materialized); the normal equations matrix A^T W A has ~100M nonzero entries

### Memory budget (16 GB RAM)

The system must run on a 16 GB RAM machine. Memory-critical components:

| Component | Estimate | Notes |
|-----------|----------|-------|
| Normal equations matrix (CSR) | ~2 GB | 100M nonzeros × (8 bytes value + 4 bytes col_idx) + 10M row ptrs |
| RHS vector | 80 MB | 10M × 8 bytes |
| CG working vectors (~5) | 400 MB | 5 × 10M × 8 bytes |
| Index mapping tables | ~200 MB | (expnum,ccdnum) ↔ integer index |
| One HEALPix chunk of detections | ~100 MB | Streaming, not all at once |
| **Total peak (Phase 2 solve)** | **~3 GB** | Fits comfortably in 16 GB |

**Critical constraint:** Never load all detections into memory. Phase 0 ingestion and Phase 2 normal equations accumulation must stream through HEALPix chunks. Phase 3 residual computation must also stream.

### Parallelization strategy
- Phase 0: Embarrassingly parallel by HEALPix pixel (use multiprocessing, but limit workers to ~4 to stay within RAM)
- Phase 1: Parallel connectivity check per star (streaming)
- Phase 2: Single solve (CG with multi-threaded BLAS). Normal equations accumulation is serial but streams through chunks.
- Phase 3: Parallel residual computation (streaming), serial re-solve
- Phase 4: Parallel per CCD
- Phase 5: Embarrassingly parallel by star

### Recommended infrastructure
- 16 GB RAM machine (current hardware)
- ~5 TB external disk (available)
- Python with numpy, scipy, healpy, astropy, fitsio, pyarrow (for parquet)
- multiprocessing for Phase 0 (limit concurrency to respect RAM)

---

## Code Structure

```
delve_ubercal/
├── config.yaml                  # Survey parameters, paths, quality cuts
├── phase0_ingest.py             # Data ingestion, quality cuts, cross-matching
├── phase1_overlap_graph.py      # Build overlap graph, check connectivity
├── phase2_solve.py              # CG solve (unanchored + anchored modes)
├── phase3_outlier_rejection.py  # Iterative sigma-clipping and re-solving
├── phase4_starflat.py           # Per-CCD flat-field refinement with epoch support
├── phase5_catalog.py            # Apply corrections, build final catalog
├── validation/
│   ├── fgcm_comparison.py       # Test 0: FGCM vs ubercal (unanchored solve)
│   ├── repeatability.py         # Test 1: Photometric repeatability vs magnitude
│   ├── dr2_comparison.py        # Test 2: Before/after MAG_DR2 - MAG_UBERCAL maps
│   ├── gaia_comparison.py       # Test 3: Gaia XP synthetic photometry comparison
│   ├── stellar_locus.py         # Test 4: Color-color locus width vs position
│   ├── des_boundary.py          # Test 5: DES footprint boundary continuity
│   └── run_all.py               # Run all validation tests
└── utils/
    ├── healpix_utils.py         # HEALPix chunking and streaming I/O
    ├── sparse_utils.py          # Sparse matrix construction helpers
    └── plotting.py              # Diagnostic plots
```

---

## Config File Template

```yaml
# delve_ubercal config
survey:
  bands: [g, r, i, z]
  nside_chunk: 32          # HEALPix nside for data chunking (~3 deg² pixels)

data:
  source: astro_data_lab                         # All data from NOIRLab Astro Data Lab
  meas_table: nsc_dr2.meas                       # Single-epoch detections
  chip_table: nsc_dr2.chip                       # Per-CCD metadata + zpterm
  exposure_table: nsc_dr2.exposure               # Per-exposure metadata (instrument, expnum)
  des_fgcm_table: des_dr2.y6_gold_zeropoint      # DES FGCM zero-points (anchor)
  delve_objects_table: delve_dr2.objects          # DELVE DR2 coadded objects (optional: for spread_model star selection)
  nsc_delve_xmatch: nsc_dr2.x1p5__object__delve_dr2__objects  # NSC↔DELVE cross-match
  output_path: /Volumes/External5TB/DELVE_UBERCAL/output/
  cache_path: /Volumes/External5TB/DELVE_UBERCAL/cache/       # Local parquet cache of query results

quality_cuts:
  mag_min: 17.0                # mag_auto from nsc_dr2.meas
  mag_max: 20.0
  magerr_max: 0.05             # magerr_auto from nsc_dr2.meas
  flags_max: 0                 # NSC flags (0 = clean)
  class_star_min: 0.8          # SExtractor CLASS_STAR from nsc_dr2.meas (no spread_model available)
  min_detections_per_star: 2
  max_detections_per_star: 25  # Random subsample if exceeded
  exclude_ccdnums: [61]        # N30: dead since Nov 2012
  instrument: c4d              # DECam only (filter out Mosaic, NEWFIRM, etc.)

solve:
  method: cg                # Conjugate gradient on normal equations
  max_iterations: 5000
  tolerance: 1.0e-5         # Relative residual
  des_anchor_weight: 1.0e6  # Penalty weight for DES ZP anchoring (anchored mode)
  tikhonov_reg: 1.0e-10     # Diagonal regularization (unanchored mode)

outlier_rejection:
  n_iterations: 5
  star_chi2_cut: 3.0        # chi2/dof threshold for star rejection
  detection_sigma_cut: 5.0  # Individual detection outlier threshold
  exposure_zp_cut: 0.3      # Max deviation from nightly median ZP (mag)

starflat:
  enabled: true
  polynomial_order: 3       # Per-CCD 2D Chebyshev order
  min_stars_per_bin: 50
  # Hardware-driven epoch boundaries (MJD)
  epoch_boundaries:
    global:                  # Apply to all CCDs
      - 56404              # 2013 Apr 22: g-band baffling
      - 56516              # 2013 Aug 12: rizY baffling
      - 56730              # 2014 Mar 14: shutter/filter aerosolization
    per_ccd:
      2:                   # S30
        - 56626            # 2013 Nov 30: failure
        - 57751            # 2016 Dec 29: recovery
      41:                  # N10
        - 58350            # 2018 Sep: hardware intervention
```

---

## Key Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Disconnected calibration graph south of DES | HIGH | Map connectivity first. Drop disconnected components — only calibrate CCD-exposures in the connected component containing DES. Report dropped coverage. |
| Non-photometric exposures introducing noise | MEDIUM | Aggressive outlier rejection + exposure-level quality flags. ZTF Zubercal work shows transparency variations on <1 hour timescales are the dominant error source — iterative clipping handles this |
| DECam CCD state changes (2012–2025) | MEDIUM | Per-CCD-per-exposure parameterization absorbs CCD changes implicitly. Phase 4 star-flat uses epoch boundaries at known hardware events. CCDNUM 61 excluded; CCDNUM 31 flagged for diagnostics. |
| Memory (16 GB RAM for 10M parameter system) | MEDIUM | Normal equations matrix ~2 GB in CSR. Stream all detection data through HEALPix chunks. Never load full detection catalog. Peak memory ~3 GB for solve. |
| Computational scale of pair generation | MEDIUM | Don't materialize all pairs. Form normal equations star-by-star. Cap at 25 detections per star (300 pairs max per star). |
| Chromatic effects (color-dependent ZP) | LOW at 5-10 mmag | Ignored in first pass. Can add a linear color term per exposure if residuals show color dependence |

---

## Minimum Viable Product

For a quick proof-of-concept before running on the full dataset:

1. Select a ~500 deg² test region that spans the DES boundary (e.g., RA=50–70°, Dec=−40° to −25°)
2. Run Phases 0–3 on just this region in BOTH solve modes (unanchored + anchored)
3. Verify that:
   - The dec = −30° discontinuity is removed (anchored solve)
   - Photometric repeatability reaches < 10 mmag for bright stars
   - The DES-interior calibration is preserved (anchored solve)
   - FGCM vs ubercal comparison shows no coherent spatial structure (unanchored solve)
4. If successful, scale to full footprint

This test region should contain ~5,000 exposures and ~50M detections — manageable on a 16 GB machine.
