# DELVE Ubercalibration

Photometric ubercalibration of the DELVE survey (~17,000 deg² in griz) using DECam internal overlaps, following the Padmanabhan et al. (2008) method.

## Overview

The DELVE survey (DECam Local Volume Exploration) combines data from 278 DECam programs to cover the southern sky. The current photometric calibration (DELVE DR2) uses ATLAS Refcat2 as the external reference, which introduces a ~10 mmag discontinuity at dec = -30° where Refcat2 switches from PS1-based to SkyMapper-based photometry.

This pipeline removes that discontinuity by solving for per-CCD-per-exposure zero-points using only internal overlap constraints — if the same star is observed in two different exposures, the magnitude difference constrains the zero-point difference. With ~341K DECam exposures and billions of overlapping measurements, the system is massively over-determined. Absolute calibration is inherited from the DES footprint (FGCM-calibrated to <3 mmag).

**Target: 5–10 mmag relative photometric uniformity across the full southern sky.**

## Quick Start

```bash
pip install -r requirements.txt

# Run on test region (RA=50-70, Dec=-40 to -25) for one band
python -m delve_ubercal.phase0_ingest --test-region --band g

# Full pipeline (see run.sh for details)
bash run.sh
```

## Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `phase0_ingest.py` | Query NSC DR2 from Astro Data Lab, apply quality cuts, cache locally |
| 1 | `phase1_overlap_graph.py` | Build calibration graph, check connectivity to DES |
| 2 | `phase2_solve.py` | CG sparse solver (unanchored + anchored modes) |
| 3 | `phase3_outlier_rejection.py` | Iterative sigma-clipping of variables and artifacts |
| 4 | `phase4_starflat.py` | Per-CCD illumination correction with epoch boundaries |
| 5 | `phase5_catalog.py` | Build final ubercalibrated catalog |
| 6 | `validation/run_all.py` | Full validation suite (6 tests) |

## Data Source

All data is queried from [NOIRLab Astro Data Lab](https://datalab.noirlab.edu/):
- **Single-epoch detections:** `nsc_dr2.meas` (NOIRLab Source Catalog DR2)
- **Zero-points:** `nsc_dr2.chip` (per-CCD zpterm)
- **DES FGCM anchor:** `des_dr2.y6_gold_zeropoint`

## Hardware Requirements

- **RAM:** 16 GB (all phases stream data by HEALPix chunk)
- **Disk:** ~500 GB – 2 TB for cached data and output
- **Network:** Broadband internet for Astro Data Lab queries

## Technical Details

See [DELVE_ubercal_plan.md](DELVE_ubercal_plan.md) for the full technical plan.

## References

- Padmanabhan, N. et al. 2008, ApJ, 674, 1217 — "An Improved Photometric Calibration of the SDSS"
- Drlica-Wagner, A. et al. 2022, ApJS, 261, 38 — "The DELVE Data Release 2"
- Burke, D. L. et al. 2018, AJ, 155, 41 — "Forward Global Photometric Calibration of DES"
