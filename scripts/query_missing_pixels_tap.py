#!/usr/bin/env python3
"""Query missing pixels using TAP async to bypass nginx gateway timeout.

Usage:
    python3 scripts/query_missing_pixels_tap.py --band g
    python3 scripts/query_missing_pixels_tap.py --band g --band r
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyvo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from delve_ubercal.phase0_ingest import (
    _pixel_coord_box,
    apply_local_joins,
    download_lookup_tables,
    get_test_patch_pixels,
    load_config,
)

TAP_URL = "https://datalab.noirlab.edu/tap"
MISSING_PIXELS = [8787, 9043, 9298, 9299]
NSIDE = 32
MAX_WAIT = 7200  # 2 hours per query


def query_pixel_tap(service, pixel, band, config):
    """Query a single pixel using TAP async."""
    cuts = config["quality_cuts"]
    ra_min, ra_max, dec_min, dec_max = _pixel_coord_box(pixel, NSIDE)

    query = f"""
    SELECT
        m.objectid, m.exposure, m.ccdnum,
        m.mag_aper4, m.magerr_aper4,
        m.class_star, m.flags,
        m.mjd, m.ra, m.dec, m.x, m.y
    FROM nsc_dr2.meas m
    WHERE m.filter = '{band}'
      AND m.flags <= {cuts['flags_max']}
      AND m.class_star > {cuts['class_star_min']}
      AND m.magerr_aper4 < {cuts['magerr_max']}
      AND m.mag_aper4 > {cuts['mag_min']}
      AND m.mag_aper4 < {cuts['mag_max']}
      AND m.ccdnum NOT IN ({','.join(str(c) for c in cuts['exclude_ccdnums'])})
      AND m.ra BETWEEN {ra_min:.6f} AND {ra_max:.6f}
      AND m.dec BETWEEN {dec_min:.6f} AND {dec_max:.6f}
    """

    print(f"  Submitting TAP async job for pixel {pixel} ({band})...", flush=True)
    t0 = time.time()

    job = service.submit_job(query)
    job.run()
    print(f"  Job URL: {job.url}", flush=True)

    # Poll for completion
    last_print = 0
    while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
        elapsed = time.time() - t0
        if elapsed - last_print >= 120:
            print(f"    Phase: {job.phase} ({elapsed:.0f}s)", flush=True)
            last_print = elapsed
        if elapsed > MAX_WAIT:
            print(f"    Timeout after {elapsed:.0f}s", flush=True)
            try:
                job.abort()
            except Exception:
                pass
            return None
        time.sleep(30)

    elapsed = time.time() - t0

    if job.phase == "COMPLETED":
        print(f"  Fetching results ({elapsed:.0f}s)...", flush=True)
        votable_result = job.fetch_result()
        df = votable_result.to_table().to_pandas()
        print(f"  Pixel {pixel}: {len(df)} rows in {elapsed:.0f}s", flush=True)
        return df
    else:
        print(f"  FAILED: phase={job.phase} after {elapsed:.0f}s", flush=True)
        return None


def process_pixel(df, pixel, band, config, chip_df, exposure_df, output_dir, cache_dir):
    """Process raw query results into the standard Phase 0 output format."""
    if df is None or len(df) == 0:
        return False

    # Save raw cache (with NaN columns for compatibility)
    raw_cache = cache_dir / f"raw_{band}_nside{NSIDE}_pixel{pixel}.parquet"
    for col in [
        "mag_auto", "magerr_auto", "mag_aper1", "magerr_aper1",
        "mag_aper2", "magerr_aper2", "mag_aper8", "magerr_aper8",
        "kron_radius", "fwhm", "asemi", "bsemi", "theta", "raerr", "decerr",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    df.to_parquet(raw_cache, index=False)
    print(f"  Raw cache saved: {raw_cache}", flush=True)

    # Apply local joins (chip zpterm + exposure table)
    processed = apply_local_joins(df, chip_df, exposure_df, band, config)
    if processed is None or len(processed) == 0:
        print(f"  No data after local joins for pixel {pixel}", flush=True)
        return False

    # Cap detections per star
    max_det = config["quality_cuts"]["max_detections_per_star"]
    star_counts = processed.groupby("objectid").cumcount()
    processed = processed[star_counts < max_det]

    # Rename for output consistency
    output_cols = {
        "objectid": "objectid",
        "expnum": "expnum",
        "ccdnum": "ccdnum",
        "band": "band",
        "m_inst": "m_inst",
        "m_err": "m_err",
        "mjd": "mjd",
        "ra": "ra",
        "dec": "dec",
        "x": "x",
        "y": "y",
    }
    # Keep only columns that exist
    keep = [c for c in output_cols if c in processed.columns]
    out_df = processed[keep]

    # Add star_id column (same as objectid for NSC)
    if "star_id" not in out_df.columns and "objectid" in out_df.columns:
        out_df = out_df.copy()
        out_df["star_id"] = out_df["objectid"]

    phase0_dir = output_dir / f"phase0_{band}"
    phase0_dir.mkdir(parents=True, exist_ok=True)
    out_path = phase0_dir / f"detections_nside{NSIDE}_pixel{pixel}.parquet"
    out_df.to_parquet(out_path, index=False)

    n_stars = out_df["objectid"].nunique() if "objectid" in out_df.columns else 0
    print(f"  Output saved: {out_path} ({n_stars} stars, {len(out_df)} dets)", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Query missing pixels via TAP async")
    parser.add_argument("--band", action="append", default=[], help="Band(s) to query")
    parser.add_argument("--pixels", type=str, default=None,
                        help="Comma-separated pixel IDs (default: 8787,9043,9298,9299)")
    args = parser.parse_args()

    if not args.band:
        args.band = ["g", "r"]

    pixels = [int(p) for p in args.pixels.split(",")] if args.pixels else MISSING_PIXELS

    config = load_config()
    output_dir = Path(config["data"]["output_path"])
    cache_dir = Path(config["data"]["cache_path"])

    service = pyvo.dal.TAPService(TAP_URL)
    print(f"Connected to TAP: {TAP_URL}")
    print(f"Pixels: {pixels}")
    print(f"Bands: {args.band}")
    print()

    for band in args.band:
        print(f"=== Band: {band} ===", flush=True)

        # Download lookup tables
        chip_df, exposure_df = download_lookup_tables(band, config, cache_dir)

        # Check which pixels are already done
        phase0_dir = output_dir / f"phase0_{band}"
        todo = []
        for pix in pixels:
            out_path = phase0_dir / f"detections_nside{NSIDE}_pixel{pix}.parquet"
            if out_path.exists():
                print(f"  Pixel {pix}: already exists, skipping", flush=True)
            else:
                todo.append(pix)

        if not todo:
            print(f"  All pixels already downloaded for {band}-band!", flush=True)
            continue

        print(f"  {len(todo)} pixels to query: {todo}", flush=True)

        for pix in todo:
            df = query_pixel_tap(service, pix, band, config)
            if df is not None and len(df) > 0:
                process_pixel(df, pix, band, config, chip_df, exposure_df, output_dir, cache_dir)
            else:
                print(f"  FAILED: pixel {pix} ({band})", flush=True)

        # Verify completeness
        still_missing = []
        for pix in pixels:
            out_path = phase0_dir / f"detections_nside{NSIDE}_pixel{pix}.parquet"
            if not out_path.exists():
                still_missing.append(pix)

        if still_missing:
            print(f"\n  WARNING: {len(still_missing)} pixels still missing for {band}: {still_missing}", flush=True)
        else:
            print(f"\n  All {len(pixels)} pixels complete for {band}!", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
