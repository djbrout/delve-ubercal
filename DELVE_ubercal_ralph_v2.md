# DELVE Ubercalibration — Pipeline Replication Guide

## What This Is

This document contains everything needed to reproduce the DELVE ubercal
pipeline from scratch. It is written as a series of `/ralph-loop` commands
for Claude Code, but also serves as a human-readable specification.

**What the pipeline does:** Determines one photometric zero-point per CCD
per exposure (~100K parameters) for the DELVE survey using internal
overlaps between DECam exposures (Padmanabhan et al. 2008 method). No
external calibration is used except: (1) Gaia DR3 for a 2-parameter
linear spatial gradient, and (2) DES FGCM for 1 parameter (absolute
zero point via median offset).

**Current results (g-band test patch, 10x10 deg):**

| Metric | Value | Description |
|--------|-------|-------------|
| FGCM comparison | 12.8 mmag | Star-level RMS vs DES FGCM (after detrend) |
| Repeatability | 8.4 mmag | Bright-star photometric floor |
| Gaia DR3 | 38.5 mmag | After color term + 3σ clip |
| PS1 DR2 | 30.8 mmag | After color term + 3σ clip |
| DES boundary | 39.1 mmag | Dec=-30 discontinuity (real systematic) |

## Prerequisites

- Python 3.10+ with: numpy, scipy, healpy, astropy, pyarrow, matplotlib,
  pyyaml, astro-datalab, pyvo, dustmaps
- Network access to NOIRLab Astro Data Lab and MAST (for PS1)
- 16 GB RAM, 5 TB external disk
- PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL

## Pipeline Architecture

```
Phase 0  ──→  Phase 1  ──→  Phase 2  ──→  Phase 3  ──┐
(query)      (graph)       (solve)       (outliers)    │
                                                       │ both g,r done
Phase 0  ──→  Phase 1  ──→  Phase 2  ──→  Phase 3  ──┤
(other band)                                           │
                                                       ▼
                                              Phase 3b (gradient detrend)
                                                       │
                                              Phase 5  (catalog)
                                                       │
                                              Phase 6  (validation)
```

Phases 0-3 run independently per band. Phase 3b requires BOTH g and r
to be through Phase 3 (it uses DELVE g-r color). Then Phase 5 and 6
run per band again.

## Quick Run (Test Patch)

```bash
export PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL

# Stage 1: Per-band phases (g and r independently)
for BAND in g r; do
  python -m delve_ubercal.phase0_ingest --band $BAND --test-patch
  python -m delve_ubercal.phase1_overlap_graph --band $BAND --test-patch
  python -m delve_ubercal.phase2_solve --band $BAND --mode unanchored --test-patch
  python -m delve_ubercal.phase3_outlier_rejection --band $BAND --mode unanchored --test-patch
done

# Stage 2: Cross-band gradient detrend (needs both g,r Phase 3 done)
for BAND in g r; do
  python -m delve_ubercal.phase3b_gradient_detrend --band $BAND --test-patch
done

# Stage 3: Catalog and validation
for BAND in g r; do
  python -m delve_ubercal.phase5_catalog --band $BAND --test-patch
  python -m delve_ubercal.validation.run_all --band $BAND --test-patch
done
```

Or use the orchestrator: `python run_full_pipeline.py --bands g r --test-patch`

---

## Phase 0: Data Ingestion

**What it does:** Queries NOIRLab Astro Data Lab for single-epoch DECam
detections from NSC DR2, applies quality cuts, strips existing calibration,
and saves per-HEALPix-pixel parquet files.

**Key details:**
- Data source: `nsc_dr2.meas` joined with `nsc_dr2.chip` and `nsc_dr2.exposure`
- Aperture: `mag_aper4` (4-arcsec diameter = 2-arcsec radius fixed aperture)
- Instrumental mag: `m_inst = mag_aper4 - zpterm_chip` (~18.8 mag)
- Quality cuts: `flags=0, class_star>0.8, magerr<0.05, 17<mag<20, ccdnum≠61`
- Detection cap: 25 per star per band, selecting the 25 lowest-error detections
- Also caches DES FGCM zero-points (`des_dr2.y6_gold_zeropoint`) with
  exposure-time normalization to 90s: `mag_zero - 2.5*log10(exptime/90)`
- Test patch: RA=[50,60], Dec=[-35,-25], 40 HEALPix pixels (nside=32)
- Chunked by HEALPix pixel to stay under 16 GB RAM

**Critical rules:**
- Max 4 parallel Data Lab workers (more causes cascade timeouts)
- NEVER run concurrent query jobs — they compete and both fail
- After Phase 0, verify ALL expected pixels have output files
- For stubborn timeout pixels: use TAP async via pyvo (bypasses nginx timeout)
- NEVER delete cached query results — filter bad data at analysis time

**Output:** `output/phase0_{band}/detections_nside32_pixel{N}.parquet`
Columns: objectid, expnum, ccdnum, band, m_inst, m_err, mjd, ra, dec, x, y

---

## Phase 1: Overlap Graph

**What it does:** Identifies which CCD-exposures share stars and finds the
largest connected component (containing DES). Drops disconnected islands.

**Key details:**
- Union-find algorithm on (expnum, ccdnum) pairs, streamed by HEALPix chunk
- A star with detections on CCD-exposures [A, B, C] creates edges A-B, A-C, B-C
- Keeps only the main connected component (should contain >95% of nodes)
- Saves per-star detection lists (needed for Phase 2 accumulation)

**Output:** `output/phase1_{band}/star_lists_nside32_pixel{N}.parquet`,
`output/phase1_{band}/connected_nodes.parquet`

---

## Phase 2: CG Sparse Solve

**What it does:** Solves for one zero-point per CCD-exposure by minimizing
weighted pairwise magnitude differences between detections of the same star.

**The model:**
```
For star s observed on CCD-exposures i,j with instrumental mags m_i, m_j:
  Minimize: Σ w_ij * (m_i + ZP_i - m_j - ZP_j)²
  where w_ij = 1/(σ_i² + σ_j²)
```

**Key details:**
- Normal equations (A^TWA) built by streaming stars one HEALPix chunk at a time
- Matrix is ~2 GB in CSR format, peak memory ~3 GB — fits in 16 GB
- Solver: `scipy.sparse.linalg.cg` with rtol=1e-5
- **Unanchored mode only**: Tikhonov regularization (1e-10 on diagonal),
  then shift solution so median(ZP_solved[DES]) = median(ZP_FGCM[DES])
- DES FGCM sentinel values: MUST filter with 25 < zp_fgcm < 35
- Use MEDIAN (not mean) for the DES shift — robust to outliers

**Output:** `output/phase2_{band}/zeropoints_unanchored.parquet`
Columns: expnum, ccdnum, zp_solved, zp_fgcm (DES only), delta_zp

---

## Phase 3: Outlier Rejection + Star Flat

**What it does:** Iteratively flags bad detections, bad stars, and bad
exposures, then re-solves. Also decomposes the solution into a static
"star flat" (per-CCD per-epoch) plus a "gray" component (per-exposure).

**Outlier rejection (5 iterations):**
1. Compute per-detection residual: r_i = m_inst_i + ZP_i - ⟨m_star⟩
2. Flag detections with |r_i| > 4σ_i (detection_sigma_cut=4.0)
3. Flag stars only if chi²/dof > 3 AND < 2 clean detections remain
4. Flag exposures with |ZP - nightly_median| > 0.3 mag
5. Flag CCDs with anomalous scatter (> median + 3σ, with floor of 50 mmag)
6. Re-solve on cleaned data

**Star flat decomposition:**
- star_flat(ccd, epoch) = median(ZP_solved - per_exposure_median) per (ccdnum, epoch)
- Epoch boundaries from config (4 epochs based on DECam hardware changes)
- Gray solve: ~2K exposure parameters instead of ~100K CCD-exposures
- Final ZP = gray(exposure) + star_flat(ccd, epoch)

**Stability fixes baked in:**
- Absolute floor on CCD scatter threshold (min_scatter_mag=0.05)
- Early stopping if RMS worsens by >50%
- Flagged nodes excluded from all diagnostics (phantom node fix)

**Output:** `output/phase3_{band}/zeropoints_unanchored.parquet`,
`output/phase3_{band}/flagged_nodes.parquet`,
`output/phase3_{band}/star_flat.parquet`

---

## Phase 3b: Gradient Detrend

**What it does:** Removes a linear RA/Dec gradient from the solved ZPs
using Gaia DR3 as a spatial reference and DELVE g-r as the color axis.

**Why this is needed:** The overlap graph has a chain-like topology that
leaves large-scale modes poorly constrained. This manifests as a ~5 mmag/deg
RA gradient that the overlap constraints alone cannot fix.

**How it works:**
1. Load Phase 3 ZPs for target band AND the other band (g needs r, r needs g)
2. Compute per-star weighted mean magnitude in both bands → DELVE g-r color
3. Cross-match with Gaia DR3 (provides ra, dec, reference magnitude)
4. Fit jointly: `residual = poly3(DELVE g-r) + a*RA + b*Dec + c`
5. 3σ clip and refit
6. Apply only the linear RA/Dec correction to all CCD-exposure ZPs
7. Re-shift to DES FGCM median for absolute scale

**Critical rules:**
- ALWAYS uses DELVE g-r color — no fallback to Gaia BP-RP
- Requires both g and r Phase 3 complete before detrending EITHER band
- MUST use raw (pre-detrend) ZPs for computing g-r color in the other band
  to avoid cross-band contamination (the `_raw.parquet` backup mechanism)
- Saves `zeropoints_unanchored_raw.parquet` before overwriting with detrended version

**Expected gradients (test patch):**
- g-band: RA ≈ -5.1, Dec ≈ 0.4 mmag/deg
- r-band: RA ≈ -3.5, Dec ≈ 0.2 mmag/deg

**Output:** Overwrites `phase3_{band}/zeropoints_unanchored.parquet`,
saves `phase3_{band}/zeropoints_unanchored_raw.parquet` (backup),
saves `phase3_{band}/gradient_detrend_info.json`

---

## Phase 5: Catalog Construction

**What it does:** Applies the final ZPs to build a per-star calibrated
magnitude catalog.

**Magnitude formula:**
```
m_cal = m_inst + ZP_solved - MAGZERO_offset
```
where MAGZERO_offset = median(ZP_FGCM - zpterm) over all DES CCD-exposures
(~31.44, a single global constant). This works for ALL CCD-exposures
(DES and non-DES) because the solver guarantees m_inst + ZP_solved is
internally consistent.

**Per-star aggregation:**
- Weighted mean magnitude (weight = 1/σ²)
- Error on weighted mean
- Number of detections, chi²
- Also stores `mag_before_{band}` (original NSC DR2 calibration) for comparison

**Output:** `output/phase5_{band}/star_catalog_{band}.parquet`

---

## Phase 6: Validation Suite

**What it does:** Runs 6 diagnostic tests to assess calibration quality.

| Test | What | Pass threshold |
|------|------|----------------|
| 0 | Star-level ubercal vs FGCM (DES only) | RMS < 15 mmag |
| 1 | Photometric repeatability floor | < 10 mmag |
| 4 | Gaia DR3 comparison (poly5 CT, DELVE g-r) | RMS < 50 mmag |
| 6 | DES boundary discontinuity at dec=-30 | < 50 mmag |
| 7 | PS1 DR2 comparison (dec > -30 only, MAST TAP) | RMS < 50 mmag |
| — | Tests 4,7: ubercal RMS ≤ before-ubercal RMS | Improvement |

**Key details:**
- Test 0: Weighted mean per star, mean-subtracted, excludes flagged nodes
- Tests 4,7: Polynomial color term fit, SFD dereddening, before/after histograms
- PS1 queries chunked by RA (2-deg strips), maxrec=500000
- PS1 sentinel values (-999): filter with mag > 10 at analysis time
- All plots saved to `output/validation_{band}/`

---

## Lessons Learned (Baked Into The Code)

These are already handled correctly in the current codebase, but are
documented here so you know WHY certain design choices were made:

1. **mag_aper4 not mag_auto**: Fixed 4" aperture gives consistent
   photometry regardless of seeing. mag_auto varies with source morphology.

2. **DES FGCM exptime normalization**: DES mag_zero includes 2.5*log10(exptime).
   NSC images are in ADU/s. `load_des_fgcm_zps()` normalizes to 90s equivalent.
   Without this, deep field ZPs are wrong by ~0.8 mag.

3. **Detection cap by error, not random**: For heavily-observed stars (DES deep
   fields have 1000+ detections), selecting the 25 lowest-error detections is
   far better than random subsampling.

4. **Cross-band contamination in Phase 3b**: If you detrend g-band first, then
   use detrended g ZPs to compute g-r for r-band detrend, the g spatial
   correction leaks into the color and biases the r gradient measurement.
   Fix: always load raw (pre-detrend) ZPs for the other band.

5. **Phantom nodes**: Flagged CCD-exposures that stay in the solve get
   Tikhonov-default ZPs (~0), which look like 31-mag outliers in diagnostics.
   Fix: exclude flagged nodes from all metrics and validation.

6. **Star flat adds its own gradient**: The per-CCD star flat captures real
   CCD throughput variations, but also introduces epoch-dependent gradients.
   This is why the gradient detrend must happen AFTER the star flat (Phase 3b
   after Phase 3), not during it.

7. **Unanchored mode is sufficient**: With only 3 external parameters
   (1 median offset + 2 gradient slopes), the ~100K individual ZPs are
   fully overlap-determined. No per-CCD-exposure FGCM anchoring needed.

---

## File Layout

```
/Volumes/External5TB/DELVE_UBERCAL/
├── delve_ubercal/           # Python package
│   ├── config.yaml          # All pipeline settings
│   ├── phase0_ingest.py     # Data Lab queries
│   ├── phase1_overlap_graph.py
│   ├── phase2_solve.py      # CG solver + star flat + gradient detrend
│   ├── phase3_outlier_rejection.py
│   ├── phase3b_gradient_detrend.py
│   ├── phase4_starflat.py   # Intra-CCD spatial (not yet essential)
│   ├── phase5_catalog.py
│   └── validation/run_all.py
├── run_full_pipeline.py     # Orchestrator
├── output/                  # All pipeline output by phase and band
├── cache/                   # Query results, Gaia xmatch, PS1 xmatch, dust maps
└── scripts/                 # Ad-hoc analysis scripts
```

---

## Ralph Loop Commands

### Run Everything (Test Patch, g+r)

```
/ralph-loop "
Read DELVE_ubercal_ralph_v2.md for the full pipeline specification.
The codebase is in delve_ubercal/. Config is in delve_ubercal/config.yaml.

Run the complete pipeline on the test patch for g and r bands:

STAGE 1 — Per-band phases (run g then r):
  For each band in [g, r]:
    1. Phase 0: python -m delve_ubercal.phase0_ingest --band {band} --test-patch
       - Verify all 40 pixels have output files before continuing
    2. Phase 1: python -m delve_ubercal.phase1_overlap_graph --band {band} --test-patch
    3. Phase 2: python -m delve_ubercal.phase2_solve --band {band} --mode unanchored --test-patch
    4. Phase 3: python -m delve_ubercal.phase3_outlier_rejection --band {band} --mode unanchored --test-patch

STAGE 2 — Gradient detrend (needs both bands through Phase 3):
  For each band in [g, r]:
    5. Phase 3b: python -m delve_ubercal.phase3b_gradient_detrend --band {band} --test-patch

STAGE 3 — Catalog and validation:
  For each band in [g, r]:
    6. Phase 5: python -m delve_ubercal.phase5_catalog --band {band} --test-patch
    7. Phase 6: python -m delve_ubercal.validation.run_all --band {band} --test-patch

VALIDATION GATES (must all pass for g-band):
  - Test 0 FGCM:        RMS < 15 mmag (expect ~13)
  - Test 1 Repeatability: floor < 10 mmag (expect ~8)
  - Test 4 Gaia:         RMS < 50 mmag (expect ~28-39)
  - Test 7 PS1:          RMS < 50 mmag (expect ~29-31)
  - Test 6 Boundary:     RMS < 50 mmag (expect ~39)
  - Tests 4,7:           after ubercal ≤ before ubercal (improvement)

REPORT all validation metrics. If any gate fails, diagnose and fix.

ENVIRONMENT:
  export PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL

Output <promise>PIPELINE_DONE</promise> when all validation gates pass.
" --max-iterations 30 --completion-promise "PIPELINE_DONE"
```

### Phase 0 Only (for adding new bands)

```
/ralph-loop "
Read DELVE_ubercal_ralph_v2.md. Run Phase 0 data ingestion for a single band.

1. Run: python -m delve_ubercal.phase0_ingest --band {BAND} --test-patch
2. Verify all 40 expected pixels have parquet files in output/phase0_{BAND}/
3. If any pixels are missing, retry with --pixels {list} --parallel 1
4. For persistent timeout pixels, use TAP async (scripts/query_missing_pixels_tap.py)
5. Print: n_pixels, n_detections, n_unique_stars, n_ccd_exposures

Do NOT proceed until all pixels are present.

Output <promise>PHASE0_DONE</promise> when complete.
" --max-iterations 15 --completion-promise "PHASE0_DONE"
```

### Rebuild From Phase 3 (after code changes)

If you change the solver or outlier rejection and need to re-run from Phase 3:

```
/ralph-loop "
Read DELVE_ubercal_ralph_v2.md. Re-run Phase 3 onward for g and r bands.

Phase 0 and 1 data are already cached. Re-run from Phase 3:

1. Phase 3 for g: python -m delve_ubercal.phase3_outlier_rejection --band g --mode unanchored --test-patch
2. Phase 3 for r: python -m delve_ubercal.phase3_outlier_rejection --band r --mode unanchored --test-patch
3. Delete stale raw backups: rm output/phase3_g/zeropoints_unanchored_raw.parquet output/phase3_r/zeropoints_unanchored_raw.parquet
4. Phase 3b for g: python -m delve_ubercal.phase3b_gradient_detrend --band g --test-patch
5. Phase 3b for r: python -m delve_ubercal.phase3b_gradient_detrend --band r --test-patch
6. Phase 5 for g: python -m delve_ubercal.phase5_catalog --band g --test-patch
7. Phase 5 for r: python -m delve_ubercal.phase5_catalog --band r --test-patch
8. Validation for g: python -m delve_ubercal.validation.run_all --band g --test-patch
9. Validation for r: python -m delve_ubercal.validation.run_all --band r --test-patch

IMPORTANT: Step 3 deletes stale raw backups so Phase 3b creates fresh ones
from the new Phase 3 output. Without this, Phase 3b would restore old ZPs.

Output <promise>REBUILD_DONE</promise> when validation passes.
" --max-iterations 20 --completion-promise "REBUILD_DONE"
```
