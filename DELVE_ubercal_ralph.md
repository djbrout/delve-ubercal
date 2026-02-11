# DELVE Ubercalibration — Ralph Wiggum Loop Commands

## Overview

This document contains the `/ralph-loop` commands to implement DELVE ubercalibration phase-by-phase using Claude Code with the Ralph Wiggum plugin. Each phase is a separate loop with its own completion promise. Run them sequentially.

**Prerequisites:**
- Claude Code with ralph-wiggum plugin installed
- Python 3.10+ with numpy, scipy, healpy, astropy, fitsio, astro-datalab, pyarrow, matplotlib, pyyaml
- Network access to NOIRLab Astro Data Lab (data is queried remotely, not stored locally)
- GitHub CLI (`gh`) installed and authenticated
- 16 GB RAM, ~5 TB external disk

---

## Phase 0: Project Setup, GitHub Repo, and Data Ingestion

```
/ralph-loop "
You are building a photometric ubercalibration pipeline for the DELVE survey (DECam Local Volume Exploration). The full plan is in DELVE_ubercal_plan.md in this repo — read it first.

PHASE 0: Create the GitHub repository, project scaffold, and data ingestion module.

PART A — Repository and scaffold:

1. Initialize a git repo and create a GitHub repository called 'delve-ubercal' with a description: 'Photometric ubercalibration of the DELVE survey using DECam internal overlaps (Padmanabhan et al. 2008 method)'. Make it public.

2. Create a clear README.md with:
   - Project overview (what ubercal is, why DELVE needs it)
   - Quick-start instructions
   - Description of each phase
   - Hardware requirements (16 GB RAM, 5 TB disk)
   - Link to the plan document

3. Create requirements.txt:
   numpy>=1.24
   scipy>=1.10
   healpy>=1.16
   astropy>=5.3
   fitsio>=1.2
   astro-datalab>=2.20
   pyarrow>=12.0
   matplotlib>=3.7
   pyyaml>=6.0

4. Create the directory structure:
   delve_ubercal/
   ├── __init__.py
   ├── config.yaml
   ├── phase0_ingest.py
   ├── phase1_overlap_graph.py  (stub)
   ├── phase2_solve.py          (stub)
   ├── phase3_outlier_rejection.py (stub)
   ├── phase4_starflat.py       (stub)
   ├── phase5_catalog.py        (stub)
   ├── validation/
   │   ├── __init__.py
   │   ├── fgcm_comparison.py   (stub)
   │   ├── repeatability.py     (stub)
   │   ├── dr2_comparison.py    (stub)
   │   ├── gaia_comparison.py   (stub)
   │   ├── stellar_locus.py     (stub)
   │   ├── des_boundary.py      (stub)
   │   └── run_all.py           (stub)
   ├── utils/
   │   ├── __init__.py
   │   ├── healpix_utils.py
   │   ├── sparse_utils.py
   │   └── plotting.py
   └── tests/
       ├── __init__.py
       └── test_phase0.py

5. Add a .gitignore for Python, data files (*.fits, *.parquet, *.hdf5), and output directories.

6. Create a Makefile or run.sh that documents the full pipeline execution order.

PART B — Config and data ingestion:

7. Implement config.yaml exactly as specified in DELVE_ubercal_plan.md (copy the config template from the plan).

8. Implement phase0_ingest.py:

   DATA SOURCE: NOIRLab Astro Data Lab. Use the `dl.queryClient` module (from `astro-datalab` package) for ADQL queries. The key tables are:
   - nsc_dr2.meas: single-epoch detections (objectid, exposure, ccdnum, mag_auto, magerr_auto, mjd, ra, dec, x, y, flags, class_star, filter)
   - nsc_dr2.chip: per-CCD metadata + zero-point (exposure, ccdnum, zpterm)
   - nsc_dr2.exposure: per-exposure metadata (exposure, expnum, instrument, filter, mjd, airmass)
   - des_dr2.y6_gold_zeropoint: DES FGCM zero-points for anchoring (expnum, ccdnum, band, mag_zero)

   IMPORTANT: DELVE DR2 on Data Lab only has coadded objects — NO single-epoch detections. Use NSC DR2 (nsc_dr2.meas) which has single-epoch measurements for ALL DECam data including DES and DELVE.

   PROCEDURE:
   - Query by HEALPix pixel (nside=32) to chunk the data. For each pixel, issue an ADQL query joining nsc_dr2.meas with nsc_dr2.chip (on exposure+ccdnum) and nsc_dr2.exposure (on exposure).
   - Apply quality cuts in ADQL: flags = 0, class_star > 0.8, magerr_auto < 0.05, mag_auto BETWEEN 17 AND 20, ccdnum != 61, instrument = 'c4d' (DECam only)
   - Strip existing zero-points: m_inst = mag_auto - chip.zpterm (NSC applies the same zpterm per exposure; ubercal will recover per-CCD variations)
   - Object grouping: NSC already links detections to unique objects via objectid. No cross-matching needed.
   - Cap at 25 detections per star per band (random subsample if N > 25)
   - Cache downloaded query results locally as parquet at /Volumes/External5TB/DELVE_UBERCAL/cache/ (check cache before querying)
   - Final output as parquet: (objectid, expnum, ccdnum, band, m_inst, m_err, mjd, ra, dec, x, y)
   - Also download and cache the DES FGCM zero-point table (des_dr2.y6_gold_zeropoint) for Phase 2 anchoring.
   - MEMORY CONSTRAINT: 16 GB RAM. Process one HEALPix pixel at a time.
   - Include a --test-region flag that limits to RA=50-70, Dec=-40 to -25 (straddling DES boundary and dec=-30)
   - Include a --band flag to process one band at a time

9. Write unit tests for the quality cuts, matching logic, and detection cap.

10. Run the pipeline in test mode to verify it works.

PART C — Phase 0 validation gate (must pass before moving to Phase 1):
   - Config loads without errors
   - phase0_ingest.py runs on test region for at least one band
   - Output parquet files exist and contain all expected columns: objectid, expnum, ccdnum, band, m_inst, m_err, mjd, ra, dec, x, y
   - DES FGCM zero-point table is cached locally
   - No star has more than 25 detections per band in the output
   - CCDNUM 61 does not appear in the output
   - Magnitude range in output is within [17, 20]
   - Number of unique stars and detections are printed and plausible (test region: ~1-10M stars, ~5-50M detections)
   - **ALL PIXELS DOWNLOADED**: Every expected HEALPix pixel must have a corresponding parquet file in phase0_{band}/. Check for missing pixels by comparing expected pixel list to actual files. If ANY pixels are missing (timeout victims), re-query them until they succeed. DO NOT proceed to Phase 1 with missing pixels — this creates holes in sky coverage that propagate to all downstream phases.
   - All unit tests pass
   - Print a summary table: band, n_stars, n_detections, n_exposures, n_ccd_exposures
   REPORT ALL METRICS. If any gate fails, diagnose and fix before declaring done.

11. Git commit with message 'Phase 0: project scaffold and data ingestion'. Push to GitHub.

When complete and all gates pass:
Output <promise>PHASE0_DONE</promise>

After 15 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 20 --completion-promise "PHASE0_DONE"
```

---

## Phase 1: Overlap Graph and Connectivity

```
/ralph-loop "
Read DELVE_ubercal_plan.md and the existing code in delve_ubercal/.

PHASE 1: Implement the overlap graph builder and connectivity checker.

Implement phase1_overlap_graph.py:

1. Read the Phase 0 output catalog (star detections with star_id, expnum, ccdnum, band).

2. Build a calibration graph where nodes are (expnum, ccdnum) pairs — each node is one CCD in one exposure. Two nodes are connected if they share at least one star.

3. For each band independently:
   a. For each star with N >= 2 detections, record which (expnum, ccdnum) pairs observed it.
   b. Use union-find (disjoint set) algorithm to identify connected components.
   c. Identify the component containing the DES footprint (DES exposures are identifiable from the exposure table — PROPID for DES programs).
   d. DROP all CCD-exposures not in the DES-connected component.

4. MEMORY CONSTRAINT: 16 GB RAM. Stream through detections by HEALPix chunk. The union-find data structure itself is lightweight (one integer per node).

5. Report statistics:
   - Total CCD-exposures per band
   - Number in DES-connected component
   - Number of disconnected components and their sizes
   - Number of dropped CCD-exposures and their approximate sky coverage (deg²)
   - Distribution of number of shared stars per CCD-exposure pair (min, median, max, histogram)

6. Save the connectivity mask: a set of (expnum, ccdnum) pairs in the connected component per band. Also save per-star detection lists (needed for Phase 2 accumulation).

7. Important: do NOT materialize all N*(N-1)/2 pairs per star. Only store the list of CCD-exposures per star. Phase 2 will accumulate normal equations star-by-star.

8. Write tests including:
   - Synthetic graph with known components
   - Verify DES exposures are correctly identified
   - Verify disconnected nodes are dropped

9. Run on test region.

PHASE 1 VALIDATION GATE (must all pass):
   - Connectivity report shows one dominant component containing >95% of CCD-exposures
   - DES CCD-exposures are correctly identified in the dominant component
   - Disconnected CCD-exposures are identified, counted, and saved to a drop list
   - The number of connected CCD-exposures is consistent with Phase 0 output (no large unexplained losses)
   - Distribution of shared stars per edge: median should be >= 5 stars per pair
   - All unit tests pass
   - Print summary: band, n_nodes_total, n_nodes_connected, n_nodes_dropped, n_components, pct_connected
   REPORT ALL METRICS. If any gate fails, diagnose and fix.

10. Git commit with message 'Phase 1: overlap graph and connectivity'. Push to GitHub.

When complete and all gates pass:
Output <promise>PHASE1_DONE</promise>

After 15 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 20 --completion-promise "PHASE1_DONE"
```

---

## Phase 2: Sparse Linear Solve

```
/ralph-loop "
Read DELVE_ubercal_plan.md and the existing code in delve_ubercal/.

PHASE 2: Implement the core ubercalibration sparse least-squares solve.

Implement phase2_solve.py:

THE MODEL: For each detection of star s on CCD c in exposure e, the calibrated magnitude is:
   m_cal = m_inst + ZP_{e,c}
One free parameter per (expnum, ccdnum) pair. No atmospheric model, no airmass, no color term.

THE SOLVE: Weighted least-squares minimizing:
   sum over all star pairs: w_ij * (m_inst_i - m_inst_j - ZP_i + ZP_j)^2
where w_ij = 1/(sigma_i^2 + sigma_j^2).

SOLVER: scipy.sparse.linalg.cg (conjugate gradient). Do NOT use LSMR or LSQR — those require the full design matrix A, which has billions of rows and cannot be materialized. We form the normal equations A^T W A directly and solve with CG.

IMPLEMENTATION:
1. Assign each connected (expnum, ccdnum) pair a unique integer index 0..N-1.

2. Build the normal equations matrix A^T W A (sparse, N x N) and vector A^T W dm (dense, length N) by streaming through stars:
   - Load detections one HEALPix chunk at a time
   - For each star with detections in CCD-exposures [i1, i2, ..., in] (n <= 25 after cap) with instrumental mags [m1, ..., mn] and errors [e1, ..., en]:
   - For each pair (ia, ib): weight w = 1/(ea^2 + eb^2)
     - Add w to diagonal (ia,ia) and (ib,ib)
     - Subtract w from off-diagonal (ia,ib) and (ib,ia)
     - Add w*(ma-mb) to RHS[ia] and w*(mb-ma) to RHS[ib]
   - Accumulate into a scipy.sparse DOK or COO matrix, converting to CSR before solving.
   - MEMORY: The final CSR matrix is ~2 GB. Accumulation must also stay within 16 GB.

3. TWO SOLVE MODES (implement both, controlled by a --mode flag):

   a. UNANCHORED MODE (--mode unanchored): For validation.
      - Add Tikhonov regularization: add 1e-10 to every diagonal entry. This breaks the graph Laplacian's null space (constant offset degeneracy) so CG converges. The effect on zero-points is ~1e-10 mag = 0.0001 micromag — completely negligible.
      - Solve with CG, tolerance 1e-5.
      - After convergence, compute: offset = mean(ZP_solved[DES]) - mean(ZP_FGCM[DES])
      - Shift entire solution: ZP_solved -= offset
      - This pins the absolute scale to the DES mean but leaves every individual DES CCD-exposure free — purely overlap-determined.

   b. ANCHORED MODE (--mode anchored): For production.
      - For each DES CCD-exposure i: add des_anchor_weight (1e6) to diagonal A^T_W_A[i,i] and des_anchor_weight * ZP_FGCM[i] to RHS[i].
      - Solve with CG, tolerance 1e-5.
      - This pins each DES CCD-exposure to its FGCM value.

4. Output for each mode: a table mapping (expnum, ccdnum, band) -> ZP_solved and delta_ZP = ZP_solved - ZP_current.

5. Diagnostics (print and save):
   - Number of iterations to convergence
   - Final relative residual
   - Histogram of delta_ZP values
   - Residual RMS before and after solve
   - For unanchored mode: map of ZP_FGCM - ZP_unanchored inside DES

6. Write tests including a SYNTHETIC TEST CASE:
   - Generate fake star observations: 1000 CCD-exposures, 5000 stars, known true zero-points
   - Add Gaussian noise to instrumental magnitudes
   - Run solver in both modes
   - Verify recovered ZPs match input to < 1 mmag RMS
   - This is the critical correctness test.

7. Run on test region in both modes.

PHASE 2 VALIDATION GATE (must all pass):
   - Synthetic test passes: recovered ZPs match input to < 1 mmag RMS
   - CG converges in both modes (relative residual < 1e-5)
   - Anchored mode: delta_ZP for DES CCD-exposures is near zero (|delta_ZP| < 1 mmag for DES)
   - Unanchored mode: ZP_FGCM - ZP_unanchored scatter is < 20 mmag RMS inside DES (will improve after outlier rejection)
   - Histogram of delta_ZP shows sensible distribution (centered near 0, tails < 0.5 mag)
   - Number of CG iterations is < 5000
   - Peak memory usage during solve is < 8 GB
   - All unit tests pass
   - Print summary: band, mode, n_params, n_iterations, residual_rms, delta_zp_median, delta_zp_rms
   REPORT ALL METRICS. If any gate fails, diagnose and fix.

8. Git commit with message 'Phase 2: CG sparse solver with anchored and unanchored modes'. Push to GitHub.

When complete and all gates pass:
Output <promise>PHASE2_DONE</promise>

After 15 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 25 --completion-promise "PHASE2_DONE"
```

---

## Phase 3: Outlier Rejection and Iterative Refinement

```
/ralph-loop "
Read DELVE_ubercal_plan.md and the existing code in delve_ubercal/.

PHASE 3: Implement iterative outlier rejection.

Implement phase3_outlier_rejection.py:

The initial Phase 2 solve is contaminated by variable stars, artifacts, cosmic rays, and non-photometric exposures. This phase iteratively removes them and re-solves.

PROCEDURE (detection-level flagging, repeat for n_iterations=5 or until convergence):
1. Using the current ZP solution, compute residuals for every detection (streaming by HEALPix chunk — 16 GB RAM constraint):
   r_i = m_inst_i + ZP_i - <m_star>
   where <m_star> is the weighted mean magnitude of the star across all its detections.
2. Flag individual outlier detections with |r_i| > 5 * sigma_i (catastrophic outliers — cosmic rays, artifacts).
   Track as (objectid, expnum, ccdnum) tuples in `flagged_det_keys`.
3. For each star, compute chi^2 = sum(r_i^2 / sigma_i^2) and dof = N_detections - 1.
   ONLY flag entire stars if chi^2/dof > 3.0 AND < 2 clean detections remain after step 2.
   This prevents DES deep field stars from being wrongly flagged as "variable".
4. Flag entire exposures where the solved ZP deviates by > 0.3 mag from the nightly median ZP (likely cloudy).
5. Flag CCDs where the intra-CCD ZP scatter across stars is anomalously large (> 3 sigma from median across all CCDs).
6. Remove flagged individual detections (not whole stars) from the input catalog.
7. Re-run Phase 2 solve (both modes) on the cleaned catalog.
8. Report per iteration: number of stars flagged, outlier detections flagged, exposures flagged, new residual RMS.

Output: final cleaned ZP solution (both modes) + lists of flagged stars, detections, and exposures saved to disk.

Write tests:
- Synthetic data with injected variable stars and outliers
- Verify they are correctly identified and removed
- Verify the cleaned solve has lower RMS

Run on test region.

PHASE 3 VALIDATION GATE (must all pass):
   - Iterative loop converges: number of newly flagged objects decreases each iteration
   - Final residual RMS is lower than initial Phase 2 RMS (report both)
   - Fraction of flagged stars is reasonable (expect ~5-15% from variables + artifacts)
   - Fraction of flagged exposures is reasonable (expect < 10% non-photometric)
   - No entire bands or large sky regions are accidentally wiped out
   - Flagged objects/exposures are saved to disk
   - Synthetic outlier test passes
   - All unit tests pass
   - Print per-iteration table: iteration, n_stars_flagged, n_detections_flagged, n_exposures_flagged, residual_rms
   - Print final summary: band, mode, initial_rms, final_rms, pct_stars_flagged, pct_detections_flagged, pct_exposures_flagged
   REPORT ALL METRICS. If any gate fails, diagnose and fix.

Git commit with message 'Phase 3: iterative outlier rejection'. Push to GitHub.

When complete and all gates pass:
Output <promise>PHASE3_DONE</promise>

After 15 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 20 --completion-promise "PHASE3_DONE"
```

---

## Phase 3b: Gradient Detrend (DELVE g-r + Gaia)

This step runs AFTER Phase 3 completes for BOTH g and r bands. It uses
DELVE g-r color for the color term (homogeneous across all bands) and
Gaia DR3 for the RA/Dec reference positions. No fallback — requires both
g and r bands.

```
# Run Phase 3 for both bands first (no gradient detrend):
PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL python -m delve_ubercal.phase3_outlier_rejection --band g --test-patch --mode unanchored
PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL python -m delve_ubercal.phase3_outlier_rejection --band r --test-patch --mode unanchored

# Then detrend both bands (order doesn't matter):
PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL python -m delve_ubercal.phase3b_gradient_detrend --band g --test-patch
PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL python -m delve_ubercal.phase3b_gradient_detrend --band r --test-patch
```

**What it does:**
- Loads Phase 3 ZPs for the target band + other band
- Computes per-star DELVE calibrated magnitudes in both bands → DELVE g-r
- Cross-matches with Gaia DR3 (provides ra, dec, reference mag)
- Fits: `resid = poly3(DELVE g-r) + a*RA + b*Dec + c` (joint color + gradient)
- 3-sigma clips and refits
- Applies RA/Dec linear correction to all CCD-exposure ZPs
- Re-shifts to DES FGCM median for absolute scale (1 parameter)
- Saves detrended ZPs back to `phase3_{band}/zeropoints_unanchored.parquet`

**Why DELVE g-r (not Gaia BP-RP):**
- Gaia G_RP is a poor match for DECam r (62 mmag RMS vs 29 mmag for g/G_BP)
- DELVE g-r gives tighter residuals (32.8 vs 44.3 mmag for g-band)
- Same color axis for ALL bands → homogeneous method, no fallback needed

**Expected gradients (test patch):**
- g-band: RA ~ -5.3, Dec ~ 0.1 mmag/deg
- r-band: RA ~ -4.1, Dec ~ 0.2 mmag/deg

---

## Phase 4: Flat-Field Refinement

```
/ralph-loop "
Read DELVE_ubercal_plan.md and the existing code in delve_ubercal/.

PHASE 4: Implement per-CCD flat-field (illumination correction / star flat) refinement.

Implement phase4_starflat.py:

After solving for per-CCD-per-exposure zero-points, systematic residuals as a function of pixel position within each CCD reveal flat-field errors.

PROCEDURE:
1. For each CCD (1-62, excluding CCDNUM 61 which is dead), collect all detection residuals (m_inst + ZP - <m_star>) from the Phase 3 cleaned solution, along with their (x, y) pixel positions on the CCD.
2. Split data by instrumental epoch. Use the epoch boundaries from config.yaml:
   - Global boundaries (all CCDs): MJD 56404 (g-band baffling), 56516 (rizY baffling), 56730 (aerosolization)
   - Per-CCD boundaries: CCDNUM 2 at MJD 56626 and 57751 (S30 failure/recovery), CCDNUM 41 at MJD 58350 (N10 hardware)
3. Within each (CCD, epoch) group, bin residuals by (x, y) position. Require min_stars_per_bin=50.
4. Fit a 2D Chebyshev polynomial of order 3 to the binned median residuals. This is the illumination correction delta_c(x,y,epoch).
5. Apply: m_corrected = m_inst - delta_c(x,y,epoch).
6. Re-run Phase 2+3 on the corrected instrumental magnitudes.
7. Diagnostic plots: for each CCD, show the 2D residual map before and after correction. Save to output/starflat_diagnostics/.

MEMORY: Stream residuals by CCD. Each CCD's data fits easily in memory.

Write tests:
- Synthetic data with known illumination pattern
- Verify the pattern is recovered
- Verify re-solve RMS improves

Run on test region.

PHASE 4 VALIDATION GATE (must all pass):
   - Per-CCD illumination corrections are computed and saved for all active CCDs
   - Illumination correction amplitude is plausible (expect ~1-10 mmag RMS, literature says ~5 mmag for DECam)
   - If any CCD shows correction > 50 mmag, flag as suspicious and report
   - Re-solve residual RMS is lower than Phase 3 final RMS (report both)
   - Diagnostic plots are saved and show reduced spatial structure
   - Epoch boundaries are respected (separate corrections per epoch)
   - Synthetic test passes
   - All unit tests pass
   - Print summary: ccd, epoch, correction_rms_mmag, n_stars_used
   REPORT ALL METRICS. If any gate fails, diagnose and fix.

Git commit with message 'Phase 4: per-CCD star flat with epoch boundaries'. Push to GitHub.

When complete and all gates pass:
Output <promise>PHASE4_DONE</promise>

After 15 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 20 --completion-promise "PHASE4_DONE"
```

---

## Phase 5: Catalog Construction

```
/ralph-loop "
Read DELVE_ubercal_plan.md and the existing code in delve_ubercal/.

PHASE 5: Build the final ubercalibrated catalog.

Implement phase5_catalog.py:

1. Apply the final zero-point corrections (anchored mode, from Phase 3/4) and illumination corrections (from Phase 4) to all detections:
   m_ubercal = m_inst + ZP_{e,c} + delta_c(x,y,epoch)

2. For each unique star, compute (per band):
   - Weighted mean magnitude across all detections
   - Error on the weighted mean: 1/sqrt(sum(1/sigma_i^2))
   - Number of detections used
   - Chi-squared of detections around the mean (variability indicator)

3. Write the final catalog with columns:
   - OBJECT_ID, RA, DEC
   - MAG_UBERCAL_{G,R,I,Z}
   - MAGERR_UBERCAL_{G,R,I,Z}
   - NOBS_{G,R,I,Z}
   - CHI2_{G,R,I,Z}

4. Also write a zero-point table: (EXPNUM, CCDNUM, BAND, ZP_SOLVED, DELTA_ZP, N_STARS, ZP_RMS) for every calibrated CCD-exposure. Include both anchored and unanchored solutions.

5. Output as FITS files, organized by HEALPix pixel (nside=32) for easy access.

6. MEMORY: Stream by HEALPix pixel. Never load full catalog.

7. Write a catalog README describing all columns, units, and provenance.

8. Run on test region.

PHASE 5 VALIDATION GATE (must all pass):
   - Output FITS files exist and are valid (can be read by astropy.io.fits)
   - All expected columns are present with correct dtypes
   - No NaN or inf values in magnitude columns (stars with valid detections)
   - Magnitude range is physical (15 < MAG_UBERCAL < 25)
   - NOBS >= 1 for all entries (by construction)
   - Zero-point table is complete: every connected CCD-exposure has an entry
   - Total number of objects matches expectations from Phase 0 (minus flagged)
   - Catalog README exists
   - All unit tests pass
   - Print summary: band, n_objects, mag_median, mag_std, nobs_median, nobs_max
   REPORT ALL METRICS. If any gate fails, diagnose and fix.

Git commit with message 'Phase 5: final ubercalibrated catalog'. Push to GitHub.

When complete and all gates pass:
Output <promise>PHASE5_DONE</promise>

After 10 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 15 --completion-promise "PHASE5_DONE"
```

---

## Phase 6: Validation Suite

```
/ralph-loop "
Read DELVE_ubercal_plan.md and the existing code in delve_ubercal/.

PHASE 6: Implement the full validation suite. Tests 0-7 (some numbers skipped for historical reasons).

All tests are in delve_ubercal/validation/run_all.py:

TEST 0 — FGCM vs Ubercal comparison (UNANCHORED + ANCHORED with sky maps).
- For every DES CCD-exposure, compute ZP_FGCM - ZP_ubercal for both solve modes.
- Plot histogram + residual vs ZP scatter.
- Generate FGCM sky map showing spatial residual pattern for both anchored and unanchored side-by-side.
- Sky map uses mean RA/Dec per CCD-exposure computed from Phase 0 detections.
- Save as validation_fgcm_comparison_{band}.png and fgcm_skymap_{band}.png.

TEST 1 — Photometric repeatability vs magnitude.
- For each star with N >= 3 detections, compute weighted RMS of individual detections around the weighted mean.
- Plot RMS vs magnitude. Should reach ~5 mmag floor for bright stars (17-18 mag).
- Save as validation_repeatability_{band}.png.

TEST 2 — DR2 comparison (anchored vs FGCM).
- For DES CCD-exposures, compute delta_ZP (ubercal - FGCM).
- Inside DES footprint, difference should be near zero.
- Save as validation_dr2_comparison_{band}.png.

TEST 4 — Gaia DR3 comparison using G_BP (g-band) or G_RP (riz) with DELVE g-r color term.
- Cross-match with Gaia via pre-built NSC-Gaia table on Data Lab.
- Use G_BP for g-band (closer match than G), G_RP for r/i/z bands.
- Polynomial color term (degree 5) — linear is insufficient for Gaia filter mismatch.
- DELVE g-r color axis (NOT Gaia BP-RP) — falls back to BP-RP if g-r not available.
- 2x2 plot: histogram (after CT only, "RMS = X.X mmag"), color term fit, sky map (DELVE - Gaia after CT), magnitude dependence.
- Save as validation_gaia_{band}.png.

TEST 5 — Stellar locus width (placeholder — requires multi-band).

TEST 6 — DES boundary continuity.
- Compare ZP medians inside vs outside DES footprint.
- Save as validation_des_boundary_{band}.png.

TEST 7 — PS1 DR2 direct comparison (dec > -30).
- Query PS1 DR2 MeanPSFMag via MAST TAP: https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/
  - NOT the REST API at catalogs.mast.stsci.edu (that doesn't support TAP async)
- Table: dbo.MeanObjectView. Key columns: {band}MeanPSFMag, QfPerfect > 0.85, nDetections > 1.
- MUST chunk by RA (2-degree strips) — full region queries get ABORTED by MAST after ~10 min.
- MUST set maxrec=500000 in submit_job() — default 100K row limit truncates dense chunks.
- Cross-match with ubercal catalog by position (1 arcsec) using astropy SkyCoord.
- Fit linear color term using PS1 g-i color.
- 2x2 plot: histogram (after CT only, "RMS = X.X mmag"), color term, sky map, magnitude residual.
- PS1 calibration is from Schlafly et al. (2012) / Magnier et al. (2020), ~7-12 mmag floor.
- Covers dec > -30 ONLY — validates northern half of DELVE footprint that overlaps PS1.
- Save as validation_ps1_{band}.png.

Before/after ubercal comparison is integrated into Tests 4 and 7:
- Both overplot "before ubercal" (blue) and "after ubercal" (orange) histograms.
- Same color term applied to both; PASS requires ubercal RMS <= before RMS.
- Phase 5 catalog includes mag_before_{band} column (original NSC DR2 calibration).

run_all.py:
- Runs Tests 0, 1, 2, 7, 4, 5, 6
- Collects all metrics into a single summary table
- Prints a PASS/FAIL verdict for each test based on thresholds:
  - Test 0: FGCM-ubercal RMS < 15 mmag
  - Test 1: Repeatability floor < 10 mmag
  - Test 2: DES interior offset < 15 mmag
  - Test 4: Gaia comparison RMS < 50 mmag
  - Test 5: Stellar locus spatial variation < 10 mmag
  - Test 6: DES boundary offset < 10 mmag
  - Test 7: PS1 comparison RMS < 30 mmag after color term
  - Tests 4 & 7: ubercal RMS <= before-ubercal RMS (integrated comparison)
- Saves the summary as validation_summary.txt

Run all tests on the test region.

PHASE 6 VALIDATION GATE (must all pass):
   - All 6 validation scripts produce plots (PNG files saved)
   - run_all.py generates validation_summary.txt
   - All metrics are reported with numerical values
   - At least 4 of 6 tests pass their thresholds on the test region (test region may not have enough coverage for all tests to reach optimal performance)
   - No crashes or errors in any validation script
   - All unit tests pass
   REPORT ALL METRICS AND PASS/FAIL VERDICTS.

Git commit with message 'Phase 6: full validation suite'. Push to GitHub.

When complete and all gates pass:
Output <promise>VALIDATION_DONE</promise>

After 15 iterations, if not complete:
- Document what is blocking progress
- List what was attempted
- Suggest alternative approaches
" --max-iterations 25 --completion-promise "VALIDATION_DONE"
```

---

## Running the Full Pipeline

After all phases are built and validated on the test region, run the full pipeline:

```
/ralph-loop "
The DELVE ubercal pipeline is built and validated on the test region. Now run it on the full DELVE DR2 footprint.

1. Run phase0_ingest.py WITHOUT the --test-region flag (full sky). Process band-by-band if needed for memory.
2. Run phase1_overlap_graph.py on full data. Report connectivity stats.
3. Run phase2_solve.py on full data — BOTH modes (unanchored and anchored).
4. Run phase3_outlier_rejection.py on full data (g and r bands first, no gradient detrend).
5. Run phase3b_gradient_detrend.py for g and r (requires both Phase 3 complete — uses DELVE g-r).
6. Run phase4_starflat.py on full data.
7. Run phase5_catalog.py on full data.
8. Run validation/run_all.py on full data.

MEMORY: 16 GB RAM. All phases must stream data by HEALPix chunk. Monitor memory with periodic checks. If any phase exceeds 12 GB RSS, stop and optimize.

DISK: Write all output to /Volumes/External5TB/DELVE_UBERCAL/output/. Expect ~500 GB - 2 TB total output.

Check validation results against thresholds:
- Test 0: FGCM-ubercal RMS < 10 mmag (tighter for full footprint)
- Test 1: Repeatability floor < 7 mmag
- Test 2: dec=-30 discontinuity visible in correction map, DES interior < 3 mmag
- Test 3: Gaia map spatially smooth, no dec=-30 step
- Test 4: Stellar locus uniform to < 7 mmag
- Test 5: DES boundary continuous to < 3 mmag

If any validation test shows problems, diagnose the issue and iterate. Common issues:
- High RMS: check outlier rejection aggressiveness
- Spatial structure: check star flat convergence
- DES boundary step: check DES anchoring

Output <promise>FULL_PIPELINE_DONE</promise> when the full pipeline has run and validation looks good.

After 20 iterations, if not complete:
- Document what is blocking progress and current state
- Save all intermediate outputs
- List remaining issues
" --max-iterations 30 --completion-promise "FULL_PIPELINE_DONE"
```

---

## Phase 7: Write the Paper

After the full pipeline runs and validation passes:

```
/ralph-loop "
The DELVE ubercal pipeline has run successfully on the full footprint and all validation tests pass. Now write a paper describing the method, implementation, and results.

Read DELVE_ubercal_plan.md, all code in delve_ubercal/, and all validation outputs.

Write a LaTeX paper in a new directory paper/ with the following structure:

1. Create paper/delve_ubercal.tex using aastex631 (AAS Journals) format.

2. TITLE: 'Ubercalibration of the DELVE Survey: Uniform Photometry Across the Southern Sky'

3. SECTIONS:

   ABSTRACT (~200 words):
   - DELVE covers 17,000 deg² in griz from DECam
   - Current calibration has ~10 mmag discontinuity at dec=-30 from Refcat2
   - We apply ubercalibration (Padmanabhan et al. 2008) using only internal DECam overlaps
   - Anchored to DES FGCM calibration
   - Achieve X mmag uniformity (fill in from validation results)
   - Catalog of N stars available

   1. INTRODUCTION:
   - DELVE survey overview (cite Drlica-Wagner et al. 2021, 2022)
   - The calibration challenge: 278 programs, heterogeneous observing conditions
   - The dec=-30 Refcat2 discontinuity problem
   - Ubercalibration concept (cite Padmanabhan et al. 2008, Schlafly et al. 2012)
   - This work: first ubercal of the full southern sky from a single instrument

   2. DATA:
   - DELVE DR2 detection catalogs
   - Quality cuts and star selection
   - Detection cap at 25 per star (best-error selection, not random)
   - Scale: N exposures, N CCD-exposures, N stars, N detections

   3. METHOD:
   - 3.1 Calibration model (per-CCD-per-exposure ZP)
   - 3.2 Overlap graph and connectivity
   - 3.3 Normal equations construction (star-by-star accumulation)
   - 3.4 Conjugate gradient solver (two modes: unanchored for validation, anchored for production)
   - 3.5 Iterative outlier rejection
   - 3.6 Star-flat refinement with epoch boundaries
   - Include the key equations

   4. RESULTS:
   - 4.1 Connectivity: N CCD-exposures in main component, N disconnected
   - 4.2 Solved zero-points: distribution, convergence
   - 4.3 FGCM comparison (Test 0): independent validation inside DES
   - 4.4 Photometric repeatability (Test 1)
   - 4.5 Removal of dec=-30 discontinuity (Tests 2 and 3)
   - 4.6 Stellar locus uniformity (Test 4)
   - 4.7 DES boundary continuity (Test 5)
   - Include all validation plots as figures

   5. DISCUSSION:
   - Comparison to FGCM approach
   - Limitations: color terms, very non-photometric exposures
   - Disconnected regions
   - Implications for SN Ia cosmology, stellar streams, dwarf galaxy searches

   6. SUMMARY

   ACKNOWLEDGMENTS, REFERENCES

4. Include all validation plots as figures (copy PNGs to paper/figures/, reference them in the tex).

5. Create paper/Makefile for building the PDF with pdflatex + bibtex.

6. Create paper/references.bib with all cited references.

7. Ensure the paper compiles (run make in paper/).

8. The paper should be publication-ready in terms of structure and completeness. Placeholder values (marked with XXX) are acceptable for numbers that come from the actual run — the structure and narrative should be complete.

Git commit with message 'Phase 7: ubercal paper draft'. Push to GitHub.

Output <promise>PAPER_DONE</promise> when the paper compiles and is structurally complete.

After 15 iterations, if not complete:
- Document what is blocking progress
- Save current state of the paper
" --max-iterations 25 --completion-promise "PAPER_DONE"
```

---

## Notes

- **Run phases sequentially.** Each depends on the output of the previous.
- **The test region (RA=50-70, Dec=-40 to -25) is critical.** It spans the DES boundary and the dec=-30 Refcat2 discontinuity — the two most important diagnostics.
- **Keep DELVE_ubercal_plan.md in the repo root.** Each ralph-loop prompt references it for the full technical context.
- **Git commit after each phase.** Each phase prompt includes a commit instruction.
- **16 GB RAM constraint.** Every phase must stream data by HEALPix chunk. Never load the full detection catalog.
- **Both solve modes.** The unanchored solve is just as important as the anchored solve — it enables the FGCM comparison (Test 0) which is the most fundamental validation.
- **Validation gates are mandatory.** Do not proceed to the next phase if the current phase's gate fails.

## Lessons Learned (Updated 2026-02-08)

### Magnitude Convention (CRITICAL)
- NSC DR2 `mag_aper4` has MAGZERO (~31 mag) baked in from FITS headers. It is NOT truly instrumental.
- Phase 0 computes `m_inst = mag_aper4 - zpterm_chip` (~18.8 mag, approximately calibrated).
- Phase 2 solver finds `ZP_solved ≈ ZP_FGCM ≈ 31.5` (total ZP, same scale as MAGZERO).
- Phase 5 MUST use `mag_ubercal = mag_aper4 + delta_zp` where `delta_zp = ZP_solved - ZP_FGCM`.
- DO NOT compute `m_inst + ZP_solved` — this double-counts MAGZERO, giving magnitudes ~50 instead of ~19.
- For non-DES CCD-exposures: `delta_zp = 0` (no FGCM reference, use NSC calibration).

### DES FGCM Sentinel Values (CRITICAL)
- The DES FGCM table (`des_dr2.y6_gold_zeropoint`) contains garbage entries: ~28 at -9999.0, ~3 at 129.9.
- MUST filter with `25.0 < zp_fgcm < 35.0` in ALL code that uses FGCM values:
  - Phase 2: unanchored DES shift, anchored penalty, diagnostics
  - Phase 3: final diagnostics
  - Phase 6: Tests 0, 2, 6
- Use MEDIAN (not mean) for the DES shift in unanchored mode — more robust to residual outliers.

### DES Deep Field Exposure Time (CRITICAL — FIXED 2026-02-09)
- DES FGCM `mag_zero` includes `2.5*log10(exptime)` — it is a TOTAL ZP, NOT per-second.
- NSC DR2 images are in ADU/s (Community Pipeline divides by exptime), so our `m_inst` and `ZP_solved` are per-second.
- Standard DES survey: exptime=90s, mag_zero ≈ 31.44.
- DES SN deep fields: exptime=175-200s, mag_zero ≈ 32.24 (0.8 mag higher!).
- Also many other exptimes: 45s (11%), 150s, 330s, 360s, 400s — cannot threshold on mag_zero.
- In test patch: 44.5% of DES CCD-exposures are deep fields (SN C-fields at RA~52.5, Dec~-28).
- **Without fix**: anchoring pulls deep field ZPs to wrong values, violating overlap constraints by ~0.8 mag; unanchored DES shift contaminated. Test 0 RMS was 657 mmag (g) / 880 mmag (r).
- **FIX**: `load_des_fgcm_zps()` normalizes all mag_zero to 90s equivalent: `mag_zero - 2.5*log10(exptime/90)`.
  - Exptime from NSC exposure table, cached in `cache/des_exptime.parquet` (86K/99K matched, rest assume 90s).
  - Single function change propagates to ALL consumers: Phase 2/3/4/5 + validation.
- **After fix**: g-band anchored DES RMS = 10.1 mmag, unanchored = 63.5 mmag. r-band: 13.7 / 78.3 mmag.

### NSC zpterm
- `nsc_dr2.chip.zpterm` is a small calibration correction (median ~0.002, range -0.3 to +0.5).
- `nsc_dr2.exposure.zpterm` exists but is per-exposure (same for all CCDs of same exposure).
- The chip table must be joined with the exposure table to get `expnum` (chip table only has `exposure` string).

### Data Lab Query Reliability
- Max 4 parallel workers. 8 overwhelms Data Lab and causes cascade timeouts.
- Use batched submissions: `batch_size = n_workers * 4`.
- Each query takes 1-10 min for data pixels, 30-60s for empty pixels.
- 3x retry with exponential backoff; do NOT cache failed queries as empty.
- Empty output files without raw cache = timeout victims; must detect and re-query.
- **DO NOT run multiple query jobs simultaneously** (e.g., re-querying missing pixels while Phase 0 for another band is running). They compete for Data Lab connections and both time out.

### Missing Pixel Completeness (CRITICAL)
- After Phase 0 completes, ALWAYS verify all expected pixels have parquet files.
- Timeout victims leave NO parquet file (not even an empty one) — check with: `expected_pixels - {pixels with files}`.
- Known problematic pixels in test patch: 8787, 9043, 9298, 9299 (dense stellar fields near galactic plane, queries take >20 min).
- Missing pixels create visible holes in sky coverage maps and corrupt boundary statistics.
- Re-query strategy: wait until no other Data Lab queries are running, then retry with `--parallel 1` and only the missing pixel list.
- DO NOT proceed to Phase 1 until all pixels are present for the current band.
- **For persistent timeout victims**: Use TAP async via pyvo (`scripts/query_missing_pixels_tap.py`).
  The `dl.queryClient.query()` goes through nginx which has a ~600-1800s gateway timeout — impossible for dense stellar fields.
  TAP async (`pyvo.dal.TAPService.submit_job()`) bypasses nginx entirely and has no server-side timeout.
  Each dense pixel may take 30-60 minutes but WILL complete.

### NSC DR2 Aperture Sizes
- mag_aper1: 1" diameter (0.5" radius)
- mag_aper2: 2" diameter (1" radius)
- mag_aper4: 4" diameter (2" radius) — PRIMARY for ubercal
- mag_aper8: 8" diameter (4" radius)
- These are DIAMETERS not radii, not pixel-based.
