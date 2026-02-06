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

    try:
        result = qc.query(sql=query, fmt="pandas", timeout=600)
    except Exception as exc:
        print(f"  Query failed for pixel {pixel}: {exc}", flush=True)
        return None

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


def ingest_band(band, pixels, config, output_dir, cache_dir):
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

    Returns
    -------
    stats : dict
        Summary statistics.
    """
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
    rng = np.random.default_rng(42)

    n_pixels = len(pixels)
    t0 = time.time()

    for i, pixel in enumerate(pixels):
        t_pixel = time.time()

        # Check if output already exists
        out_file = band_output_dir / f"detections_nside{nside}_pixel{pixel}.parquet"
        if out_file.exists():
            # Read cached output to accumulate stats
            df_out = pd.read_parquet(out_file)
            if len(df_out) > 0:
                total_stars += df_out["objectid"].nunique()
                total_detections += len(df_out)
                uniq = df_out[["expnum", "ccdnum"]].drop_duplicates()
                total_exposures.update(uniq["expnum"].values)
                total_ccd_exposures.update(
                    zip(uniq["expnum"].values, uniq["ccdnum"].values)
                )
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_pixels - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{n_pixels}] Pixel {pixel}: cached ({len(df_out)} dets), "
                f"ETA {eta:.0f}s",
                flush=True,
            )
            continue

        # Query Data Lab (meas table only, no joins)
        df = query_pixel(pixel, nside, band, config, band_cache_dir)

        if df is None or len(df) == 0:
            pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_pixels - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{n_pixels}] Pixel {pixel}: no data, "
                f"ETA {eta:.0f}s",
                flush=True,
            )
            continue

        # Apply local joins (chip zpterm, exposure instrument filter)
        df = apply_local_joins(df, chip_df, exposure_df, band, config)

        if len(df) == 0:
            pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
            continue

        # Re-assign to correct HEALPix pixel (filter out margin detections)
        df = assign_to_healpix_pixel(df, nside)
        df = df[df["healpix_pixel"] == pixel].drop(columns=["healpix_pixel"])

        if len(df) == 0:
            pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
            continue

        # Apply detection cap
        df = apply_detection_cap(df, cuts["max_detections_per_star"], rng=rng)

        # Filter minimum detections
        df = filter_min_detections(df, cuts["min_detections_per_star"])

        if len(df) == 0:
            pd.DataFrame(columns=EMPTY_COLS).to_parquet(out_file, index=False)
            continue

        # Write output
        df.to_parquet(out_file, index=False)

        n_stars = df["objectid"].nunique()
        n_dets = len(df)
        total_stars += n_stars
        total_detections += n_dets
        uniq = df[["expnum", "ccdnum"]].drop_duplicates()
        total_exposures.update(uniq["expnum"].values)
        total_ccd_exposures.update(
            zip(uniq["expnum"].values, uniq["ccdnum"].values)
        )

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (n_pixels - i - 1) / rate if rate > 0 else 0
        dt = time.time() - t_pixel

        print(
            f"  [{i+1}/{n_pixels}] Pixel {pixel}: "
            f"{n_stars} stars, {n_dets} dets, {dt:.1f}s, "
            f"ETA {eta:.0f}s",
            flush=True,
        )

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
        print(f"Full sky: {len(pixels)} HEALPix pixels (nside={nside})", flush=True)

    if args.max_pixels is not None:
        pixels = pixels[:args.max_pixels]
        print(f"Limited to first {args.max_pixels} pixels", flush=True)

    print(f"Band: {args.band}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    print(flush=True)

    # Download DES FGCM zero-points
    download_des_fgcm(config, cache_dir)
    print(flush=True)

    # Run ingestion
    stats = ingest_band(args.band, pixels, config, output_dir, cache_dir)

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
