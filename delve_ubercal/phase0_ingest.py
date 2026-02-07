"""Phase 0: Data ingestion from NOIRLab Astro Data Lab.

Queries NSC DR2 single-epoch detections, applies quality cuts,
strips zero-points, caps detections per star, and caches locally as parquet.

Optimization: downloads chip and exposure lookup tables once, then queries
meas table without joins (much faster). ZP correction and instrument
filtering are done locally.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dl import queryClient as qc

from delve_ubercal.utils.healpix_utils import (
    get_all_healpix_pixels,
    get_healpix_pixels_in_region,
)


def load_config(config_path=None):
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_test_region_pixels(nside):
    """Get HEALPix pixels for the test region: RA=50-70, Dec=-40 to -25."""
    return get_healpix_pixels_in_region(nside, 50.0, 70.0, -40.0, -25.0)


def _pixel_coord_box(pixel, nside):
    """Get RA/Dec coordinate box for a HEALPix pixel with margin."""
    import healpy as hp

    theta, phi = hp.pix2ang(nside, pixel, nest=False)
    ra_center = np.degrees(phi)
    dec_center = 90.0 - np.degrees(theta)

    pixel_area = hp.nside2pixarea(nside, degrees=True)
    radius_deg = np.sqrt(pixel_area / np.pi) * 1.5  # 50% margin

    cos_dec = np.cos(np.radians(dec_center))
    ra_half = radius_deg / max(cos_dec, 0.1)
    return (
        ra_center - ra_half,
        ra_center + ra_half,
        dec_center - radius_deg,
        dec_center + radius_deg,
    )


def build_pixel_query(pixel, nside, band, config):
    """Build ADQL query for a single HEALPix pixel.

    Queries nsc_dr2.meas only (no joins). The chip and exposure joins
    are done locally for speed.

    Parameters
    ----------
    pixel : int
        HEALPix pixel index (RING ordering at nside).
    nside : int
        HEALPix nside.
    band : str
        Filter band (g, r, i, z).
    config : dict
        Pipeline configuration.

    Returns
    -------
    query : str
        ADQL query string.
    """
    cuts = config["quality_cuts"]
    tables = config["data"]
    ra_min, ra_max, dec_min, dec_max = _pixel_coord_box(pixel, nside)

    query = f"""
    SELECT
        m.objectid,
        m.exposure,
        m.ccdnum,
        m.mag_auto,
        m.magerr_auto,
        m.mjd,
        m.ra,
        m.dec,
        m.x,
        m.y
    FROM {tables['meas_table']} m
    WHERE m.filter = '{band}'
      AND m.flags <= {cuts['flags_max']}
      AND m.class_star > {cuts['class_star_min']}
      AND m.magerr_auto < {cuts['magerr_max']}
      AND m.mag_auto > {cuts['mag_min']}
      AND m.mag_auto < {cuts['mag_max']}
      AND m.ccdnum NOT IN ({','.join(str(c) for c in cuts['exclude_ccdnums'])})
      AND m.ra BETWEEN {ra_min:.6f} AND {ra_max:.6f}
      AND m.dec BETWEEN {dec_min:.6f} AND {dec_max:.6f}
    """
    return query.strip()


def download_lookup_tables(band, config, cache_dir):
    """Download chip zpterm and exposure lookup tables (one-time per band).

    Returns
    -------
    chip_df : pd.DataFrame
        Chip table with (exposure, ccdnum, zpterm).
    exposure_df : pd.DataFrame
        Exposure table with (exposure, expnum, instrument).
    """
    chip_cache = cache_dir / f"nsc_chip_{band}.parquet"
    exp_cache = cache_dir / f"nsc_exposure_{band}.parquet"

    # Download chip table
    if chip_cache.exists():
        print(f"  Chip zpterm table ({band}) already cached.", flush=True)
        chip_df = pd.read_parquet(chip_cache)
    else:
        print(f"  Downloading chip zpterm table ({band})...", flush=True)
        t0 = time.time()
        tables = config["data"]
        cuts = config["quality_cuts"]
        query = f"""
        SELECT exposure, ccdnum, zpterm
        FROM {tables['chip_table']}
        WHERE filter = '{band}'
          AND ccdnum NOT IN ({','.join(str(c) for c in cuts['exclude_ccdnums'])})
        """
        chip_df = qc.query(sql=query, fmt="pandas", timeout=600)
        chip_df.to_parquet(chip_cache, index=False)
        print(f"  Downloaded {len(chip_df):,} chip entries in {time.time()-t0:.0f}s.", flush=True)

    # Download exposure table
    if exp_cache.exists():
        print(f"  Exposure table ({band}) already cached.", flush=True)
        exposure_df = pd.read_parquet(exp_cache)
    else:
        print(f"  Downloading exposure table ({band})...", flush=True)
        t0 = time.time()
        tables = config["data"]
        query = f"""
        SELECT exposure, expnum, instrument
        FROM {tables['exposure_table']}
        WHERE filter = '{band}'
        """
        exposure_df = qc.query(sql=query, fmt="pandas", timeout=600)
        exposure_df.to_parquet(exp_cache, index=False)
        print(f"  Downloaded {len(exposure_df):,} exposure entries in {time.time()-t0:.0f}s.", flush=True)

    return chip_df, exposure_df


def query_pixel(pixel, nside, band, config, cache_dir):
    """Query a single HEALPix pixel from Astro Data Lab, with caching.

    Parameters
    ----------
    pixel : int
        HEALPix pixel index.
    nside : int
        HEALPix nside.
    band : str
        Filter band.
    config : dict
        Pipeline configuration.
    cache_dir : Path
        Directory for cached parquet files.

    Returns
    -------
    df : pd.DataFrame or None
        DataFrame with raw detections (before local joins), or None if no data.
    """
    cache_file = cache_dir / f"raw_{band}_nside{nside}_pixel{pixel}.parquet"

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            return df if len(df) > 0 else None
        except Exception:
            pass  # Re-query if cache is corrupt

    query = build_pixel_query(pixel, nside, band, config)

    for attempt in range(3):
        try:
            result = qc.query(sql=query, fmt="pandas", timeout=600)
            break
        except Exception as exc:
            print(f"  Query failed for pixel {pixel} (attempt {attempt+1}/3): {exc}",
                  flush=True)
            if attempt < 2:
                import time as _time
                _time.sleep(10 * (attempt + 1))  # 10s, 20s backoff
            else:
                return "FAILED"  # sentinel: do NOT cache as empty

    if result is None or len(result) == 0:
        # Write empty parquet so we don't re-query
        empty = pd.DataFrame(
            columns=[
                "objectid", "exposure", "ccdnum",
                "mag_auto", "magerr_auto", "mjd", "ra", "dec", "x", "y",
            ]
        )
        empty.to_parquet(cache_file)
        return None

    # Cache raw result
    result.to_parquet(cache_file, index=False)
    return result


def apply_local_joins(df, chip_df, exposure_df, band, config):
    """Apply chip zpterm correction and exposure instrument filter locally.

    Parameters
    ----------
    df : pd.DataFrame
        Raw meas data with 'exposure', 'ccdnum', 'mag_auto' columns.
    chip_df : pd.DataFrame
        Chip lookup with (exposure, ccdnum, zpterm).
    exposure_df : pd.DataFrame
        Exposure lookup with (exposure, expnum, instrument).
    band : str
        Filter band.
    config : dict
        Pipeline configuration.

    Returns
    -------
    df : pd.DataFrame
        Processed DataFrame with m_inst, expnum, band columns.
    """
    cuts = config["quality_cuts"]

    # Join with exposure table to get expnum and instrument
    df = df.merge(exposure_df[["exposure", "expnum", "instrument"]],
                  on="exposure", how="inner")

    # Filter to DECam only
    df = df[df["instrument"] == cuts["instrument"]]
    df = df.drop(columns=["instrument"])

    # Join with chip table to get zpterm
    df = df.merge(chip_df[["exposure", "ccdnum", "zpterm"]],
                  on=["exposure", "ccdnum"], how="inner")

    # Compute instrumental magnitude
    df["m_inst"] = df["mag_auto"] - df["zpterm"]
    df["m_err"] = df["magerr_auto"]
    df["band"] = band

    # Select final columns
    df = df[["objectid", "expnum", "ccdnum", "band",
             "m_inst", "m_err", "mjd", "ra", "dec", "x", "y"]].copy()

    # Ensure correct dtypes
    df["expnum"] = df["expnum"].astype(np.int64)
    df["ccdnum"] = df["ccdnum"].astype(np.int32)
    df["m_inst"] = df["m_inst"].astype(np.float64)
    df["m_err"] = df["m_err"].astype(np.float64)
    df["mjd"] = df["mjd"].astype(np.float64)
    df["ra"] = df["ra"].astype(np.float64)
    df["dec"] = df["dec"].astype(np.float64)
    df["x"] = df["x"].astype(np.float32)
    df["y"] = df["y"].astype(np.float32)

    return df


def apply_detection_cap(df, max_per_star, rng=None):
    """Cap detections per star to max_per_star via random subsampling.

    Parameters
    ----------
    df : pd.DataFrame
        Detection table with 'objectid' column.
    max_per_star : int
        Maximum detections per star.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    df_capped : pd.DataFrame
        Detection table with at most max_per_star detections per objectid.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    totals = df.groupby("objectid")["objectid"].transform("count")

    # For stars with <= max_per_star detections, keep all
    mask_keep = totals <= max_per_star

    # For stars with > max_per_star, randomly subsample
    needs_cap = df[~mask_keep]
    if len(needs_cap) > 0:
        seed = rng.integers(2**31)
        sampled_idx = (
            needs_cap.groupby("objectid", group_keys=False)
            .apply(lambda g: g.sample(n=max_per_star, random_state=seed))
            .index
        )
        return pd.concat([df[mask_keep], df.loc[sampled_idx]]).reset_index(drop=True)

    return df


def filter_min_detections(df, min_detections):
    """Remove stars with fewer than min_detections."""
    counts = df.groupby("objectid")["objectid"].transform("count")
    return df[counts >= min_detections].copy()


def assign_to_healpix_pixel(df, nside):
    """Assign each detection to its HEALPix pixel based on RA/Dec.

    This re-assigns detections to the correct output pixel
    (the query may return detections from neighboring pixels
    due to the coordinate box margin).
    """
    import healpy as hp

    theta = np.radians(90.0 - df["dec"].values)
    phi = np.radians(df["ra"].values)
    df["healpix_pixel"] = hp.ang2pix(nside, theta, phi, nest=False)
    return df


def download_des_fgcm(config, cache_dir):
    """Download DES FGCM zero-point table and cache locally."""
    cache_file = cache_dir / "des_y6_fgcm_zeropoints.parquet"

    if cache_file.exists():
        print("  DES FGCM zero-points already cached.", flush=True)
        return pd.read_parquet(cache_file)

    print("  Downloading DES FGCM zero-points from Astro Data Lab...", flush=True)
    query = f"""
    SELECT expnum, ccdnum, band, mag_zero, sigma_mag_zero, source, flag
    FROM {config['data']['des_fgcm_table']}
    WHERE source = 'FGCM'
    """

    result = qc.query(sql=query, fmt="pandas", timeout=600)
    print(f"  Downloaded {len(result)} DES FGCM zero-point entries.", flush=True)

    result.to_parquet(cache_file, index=False)
    return result


EMPTY_COLS = [
    "objectid", "expnum", "ccdnum", "band",
    "m_inst", "m_err", "mjd", "ra", "dec", "x", "y",
]


def _process_single_pixel(pixel, nside, band, config, chip_df, exposure_df,
                           band_output_dir, band_cache_dir, cuts):
    """Process a single pixel: query, join, cap, filter, save.

    Returns (pixel, n_stars, n_dets, expnums, ccd_exposures, elapsed).
    Thread-safe: each pixel writes to its own file.
    """
    t0 = time.time()
    out_file = band_output_dir / f"detections_nside{nside}_pixel{pixel}.parquet"

    # Check if output already exists
    if out_file.exists():
        df_out = pd.read_parquet(out_file)
        if len(df_out) > 0:
            uniq = df_out[["expnum", "ccdnum"]].drop_duplicates()
            return (
                pixel, df_out["objectid"].nunique(), len(df_out),
                set(uniq["expnum"].values),
                set(zip(uniq["expnum"].values, uniq["ccdnum"].values)),
                time.time() - t0, "cached",
            )
        # Empty output file: check if raw cache exists to distinguish
        # genuinely empty pixels from previous timeout failures
        raw_cache = band_cache_dir / f"raw_{band}_nside{nside}_pixel{pixel}.parquet"
        if raw_cache.exists():
            return (pixel, 0, 0, set(), set(), time.time() - t0, "cached_empty")
        # No raw cache = previous timeout failure. Delete empty output and re-query.
        out_file.unlink()
        print(f"  Pixel {pixel}: re-querying (previous timeout)", flush=True)

    # Query Data Lab
    df = query_pixel(pixel, nside, band, config, band_cache_dir)

    # "FAILED" sentinel means query timed out — do NOT write empty output
    if isinstance(df, str) and df == "FAILED":
        return (pixel, 0, 0, set(), set(), time.time() - t0, "failed")

    if df is None or len(df) == 0:
        pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
        return (pixel, 0, 0, set(), set(), time.time() - t0, "empty")

    # Apply local joins
    df = apply_local_joins(df, chip_df, exposure_df, band, config)
    if len(df) == 0:
        pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
        return (pixel, 0, 0, set(), set(), time.time() - t0, "empty")

    # Re-assign to correct HEALPix pixel
    df = assign_to_healpix_pixel(df, nside)
    df = df[df["healpix_pixel"] == pixel].drop(columns=["healpix_pixel"])
    if len(df) == 0:
        pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
        return (pixel, 0, 0, set(), set(), time.time() - t0, "empty")

    # Apply detection cap (use pixel as seed for reproducibility)
    rng = np.random.default_rng(42 + pixel)
    df = apply_detection_cap(df, cuts["max_detections_per_star"], rng=rng)

    # Filter minimum detections
    df = filter_min_detections(df, cuts["min_detections_per_star"])
    if len(df) == 0:
        pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
        return (pixel, 0, 0, set(), set(), time.time() - t0, "empty")

    df.to_parquet(out_file, index=False)
    n_stars = df["objectid"].nunique()
    n_dets = len(df)
    uniq = df[["expnum", "ccdnum"]].drop_duplicates()
    return (
        pixel, n_stars, n_dets,
        set(uniq["expnum"].values),
        set(zip(uniq["expnum"].values, uniq["ccdnum"].values)),
        time.time() - t0, "queried",
    )


def ingest_band(band, pixels, config, output_dir, cache_dir, n_workers=1):
    """Run Phase 0 ingestion for one band across the given HEALPix pixels.

    Parameters
    ----------
    band : str
        Filter band (g, r, i, z).
    pixels : np.ndarray
        Array of HEALPix pixel indices to process.
    config : dict
        Pipeline configuration.
    output_dir : Path
        Directory for final output parquet files.
    cache_dir : Path
        Directory for cached query results.
    n_workers : int
        Number of parallel query workers (default 1 = sequential).

    Returns
    -------
    stats : dict
        Summary statistics.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    nside = config["survey"]["nside_chunk"]
    cuts = config["quality_cuts"]

    band_output_dir = output_dir / f"phase0_{band}"
    band_output_dir.mkdir(parents=True, exist_ok=True)

    band_cache_dir = cache_dir / f"phase0_{band}"
    band_cache_dir.mkdir(parents=True, exist_ok=True)

    # Download lookup tables once
    chip_df, exposure_df = download_lookup_tables(band, config, cache_dir)

    total_stars = 0
    total_detections = 0
    total_exposures = set()
    total_ccd_exposures = set()

    n_pixels = len(pixels)
    t0 = time.time()
    completed = 0

    if n_workers <= 1:
        # Sequential mode (original behavior)
        for i, pixel in enumerate(pixels):
            result = _process_single_pixel(
                pixel, nside, band, config, chip_df, exposure_df,
                band_output_dir, band_cache_dir, cuts,
            )
            pix, n_s, n_d, exps, ccd_exps, dt, status = result
            total_stars += n_s
            total_detections += n_d
            total_exposures.update(exps)
            total_ccd_exposures.update(ccd_exps)
            completed += 1

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (n_pixels - completed) / rate if rate > 0 else 0

            if status == "queried":
                print(
                    f"  [{completed}/{n_pixels}] Pixel {pix}: "
                    f"{n_s} stars, {n_d} dets, {dt:.1f}s, ETA {eta:.0f}s",
                    flush=True,
                )
            elif status in ("cached", "cached_empty"):
                if completed % 100 == 0 or completed == n_pixels:
                    print(
                        f"  [{completed}/{n_pixels}] Pixel {pix}: {status} "
                        f"({n_d} dets), ETA {eta:.0f}s",
                        flush=True,
                    )
            else:
                if completed % 50 == 0:
                    print(
                        f"  [{completed}/{n_pixels}] Pixel {pix}: no data, "
                        f"ETA {eta:.0f}s",
                        flush=True,
                    )
    else:
        # Parallel mode — process in batches to avoid overwhelming Data Lab
        batch_size = n_workers * 4  # 4 batches worth of work per round
        print(f"  Using {n_workers} parallel workers, batch size {batch_size}",
              flush=True)
        failed_pixels = []

        for batch_start in range(0, n_pixels, batch_size):
            batch = pixels[batch_start:batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {}
                for pixel in batch:
                    fut = executor.submit(
                        _process_single_pixel,
                        pixel, nside, band, config, chip_df, exposure_df,
                        band_output_dir, band_cache_dir, cuts,
                    )
                    futures[fut] = pixel

                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                    except Exception as exc:
                        pixel = futures[fut]
                        print(f"  Pixel {pixel} failed: {exc}", flush=True)
                        failed_pixels.append(pixel)
                        completed += 1
                        continue

                    pix, n_s, n_d, exps, ccd_exps, dt, status = result
                    total_stars += n_s
                    total_detections += n_d
                    total_exposures.update(exps)
                    total_ccd_exposures.update(ccd_exps)
                    completed += 1

                    if status == "failed":
                        failed_pixels.append(pix)

                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (n_pixels - completed) / rate if rate > 0 else 0

                    if status == "queried" and n_d > 0:
                        print(
                            f"  [{completed}/{n_pixels}] Pixel {pix}: "
                            f"{n_s} stars, {n_d} dets, {dt:.1f}s, ETA {eta:.0f}s",
                            flush=True,
                        )
                    elif status == "failed":
                        print(
                            f"  [{completed}/{n_pixels}] Pixel {pix}: FAILED "
                            f"(will retry), ETA {eta:.0f}s",
                            flush=True,
                        )
                    elif completed % 200 == 0 or completed == n_pixels:
                        print(
                            f"  [{completed}/{n_pixels}] Progress: {status}, "
                            f"ETA {eta:.0f}s",
                            flush=True,
                        )

        # Retry failed pixels sequentially
        if failed_pixels:
            print(f"\n  Retrying {len(failed_pixels)} failed pixels sequentially...",
                  flush=True)
            for pixel in failed_pixels:
                result = _process_single_pixel(
                    pixel, nside, band, config, chip_df, exposure_df,
                    band_output_dir, band_cache_dir, cuts,
                )
                pix, n_s, n_d, exps, ccd_exps, dt, status = result
                total_stars += n_s
                total_detections += n_d
                total_exposures.update(exps)
                total_ccd_exposures.update(ccd_exps)
                if status == "failed":
                    print(f"  Pixel {pix}: still failing after retry", flush=True)
                elif n_d > 0:
                    print(f"  Pixel {pix}: recovered {n_s} stars, {n_d} dets",
                          flush=True)
            print(f"  Retry complete. {len(failed_pixels)} pixels attempted.",
                  flush=True)

    total_time = time.time() - t0

    stats = {
        "band": band,
        "n_pixels": n_pixels,
        "n_stars": total_stars,
        "n_detections": total_detections,
        "n_exposures": len(total_exposures),
        "n_ccd_exposures": len(total_ccd_exposures),
        "total_time_s": total_time,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Data Ingestion")
    parser.add_argument("--band", required=True, help="Filter band (g, r, i, z)")
    parser.add_argument(
        "--test-region", action="store_true",
        help="Limit to test region RA=50-70, Dec=-40 to -25"
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--max-pixels", type=int, default=None,
        help="Limit to first N pixels (for quick testing)"
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel query workers (default 1 = sequential)"
    )
    parser.add_argument(
        "--dec-max", type=float, default=35.0,
        help="Maximum declination for DELVE footprint (default 35 deg)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    nside = config["survey"]["nside_chunk"]

    output_dir = Path(config["data"]["output_path"])
    cache_dir = Path(config["data"]["cache_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which pixels to process
    if args.test_region:
        pixels = get_test_region_pixels(nside)
        print(f"Test region: {len(pixels)} HEALPix pixels (nside={nside})", flush=True)
    else:
        pixels = get_all_healpix_pixels(nside)
        n_all = len(pixels)

        # Filter to DELVE footprint (dec < dec_max)
        import healpy as hp
        theta, phi = hp.pix2ang(nside, pixels, nest=False)
        dec = 90.0 - np.degrees(theta)
        mask = dec < args.dec_max
        pixels = pixels[mask]
        print(f"Full sky: {n_all} -> {len(pixels)} pixels after dec < {args.dec_max} "
              f"(nside={nside})", flush=True)

    if args.max_pixels is not None:
        pixels = pixels[:args.max_pixels]
        print(f"Limited to first {args.max_pixels} pixels", flush=True)

    print(f"Band: {args.band}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    if args.parallel > 1:
        print(f"Parallel workers: {args.parallel}", flush=True)
    print(flush=True)

    # Download DES FGCM zero-points
    download_des_fgcm(config, cache_dir)
    print(flush=True)

    # Run ingestion
    stats = ingest_band(args.band, pixels, config, output_dir, cache_dir,
                        n_workers=args.parallel)

    # Print summary
    print(flush=True)
    print("=" * 60, flush=True)
    print("Phase 0 Summary", flush=True)
    print("=" * 60, flush=True)
    print(f"  Band:            {stats['band']}", flush=True)
    print(f"  Pixels:          {stats['n_pixels']}", flush=True)
    print(f"  Stars:           {stats['n_stars']:,}", flush=True)
    print(f"  Detections:      {stats['n_detections']:,}", flush=True)
    print(f"  Exposures:       {stats['n_exposures']:,}", flush=True)
    print(f"  CCD-exposures:   {stats['n_ccd_exposures']:,}", flush=True)
    print(f"  Time:            {stats['total_time_s']:.1f}s", flush=True)
    print("=" * 60, flush=True)

    return stats


if __name__ == "__main__":
    main()
