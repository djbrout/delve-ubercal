"""Phase 5: Build final ubercalibrated catalog.

Applies zero-point corrections and star flat corrections to all detections,
computes weighted mean magnitudes per star, and writes the output catalog.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from delve_ubercal.phase0_ingest import get_test_region_pixels, load_config
from delve_ubercal.phase2_solve import build_node_index, load_des_fgcm_zps
from delve_ubercal.phase3_outlier_rejection import load_exposure_mjds
from delve_ubercal.phase4_starflat import evaluate_chebyshev_2d, get_epoch
from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels


def load_zp_solution(phase3_dir, phase2_dir, mode="anchored"):
    """Load the best available ZP solution.

    Returns dict: (expnum, ccdnum) -> zp_solved.
    """
    for d in [phase3_dir, phase2_dir]:
        f = d / f"zeropoints_{mode}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            # Filter out zero-valued ZPs
            df = df[df["zp_solved"] > 1.0]
            return dict(zip(
                zip(df["expnum"].values, df["ccdnum"].values),
                df["zp_solved"].values,
            ))
    return {}


def load_starflat_corrections(phase4_dir):
    """Load star flat corrections if available.

    Returns dict: (ccdnum, epoch) -> Chebyshev coefficients.
    """
    pkl = phase4_dir / "starflat_corrections.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return {}


def load_flagged_stars(phase3_dir):
    """Load set of flagged star IDs from Phase 3."""
    f = phase3_dir / "flagged_stars.parquet"
    if f.exists():
        df = pd.read_parquet(f)
        return set(df["objectid"].values)
    return set()


def build_star_catalog(phase0_files, zp_dict, starflat_corrections,
                       flagged_stars, exposure_mjds, node_to_idx,
                       config, band):
    """Build the final ubercalibrated star catalog.

    Parameters
    ----------
    phase0_files : list of Path
        Phase 0 detection files (with x, y, ra, dec, mjd).
    zp_dict : dict
        (expnum, ccdnum) -> zp_solved.
    starflat_corrections : dict
        (ccdnum, epoch) -> Chebyshev coefficients.
    flagged_stars : set
        Set of objectids to exclude.
    exposure_mjds : dict
        expnum -> MJD.
    node_to_idx : dict
        Connected node index.
    config : dict
        Pipeline config.
    band : str
        Filter band.

    Returns
    -------
    star_cat : pd.DataFrame
        Per-star catalog with weighted mean magnitudes.
    zp_table : pd.DataFrame
        Per-CCD-exposure zero-point table.
    """
    # Accumulate per-star statistics
    star_accum = {}  # objectid -> {sum_w, sum_wm, sum_wr2, n_det, ra, dec}

    median_zp = np.median([zp for zp in zp_dict.values() if zp > 1.0])
    min_valid_zp = median_zp - 5.0

    n_files = 0
    n_dets_total = 0
    n_dets_used = 0

    for f in phase0_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        n_files += 1
        n_dets_total += len(df)

        # Filter to connected nodes with valid ZPs
        nodes = list(zip(df["expnum"].values, df["ccdnum"].values))
        zps = np.array([zp_dict.get(n, np.nan) for n in nodes])
        valid = (~np.isnan(zps)) & (zps > min_valid_zp)

        # Filter flagged stars
        if flagged_stars:
            valid &= ~df["objectid"].isin(flagged_stars).values

        df = df[valid].copy()
        zps = zps[valid]
        n_dets_used += len(df)

        if len(df) == 0:
            continue

        # Apply star flat correction if available
        delta_sf = np.zeros(len(df))
        if starflat_corrections and "x" in df.columns and "y" in df.columns:
            if "mjd" not in df.columns:
                df["mjd"] = df["expnum"].map(exposure_mjds)
            for i, (_, row) in enumerate(df.iterrows()):
                mjd = row.get("mjd", np.nan)
                if np.isnan(mjd):
                    continue
                epoch = get_epoch(mjd, row["ccdnum"], config)
                key = (int(row["ccdnum"]), epoch)
                if key in starflat_corrections:
                    delta_sf[i] = evaluate_chebyshev_2d(
                        [row["x"]], [row["y"]], starflat_corrections[key]
                    )[0]

        # Calibrated magnitude
        m_cal = df["m_inst"].values + zps - delta_sf
        w = 1.0 / (df["m_err"].values ** 2)

        # Accumulate per star
        for j, (_, row) in enumerate(df.iterrows()):
            oid = row["objectid"]
            if oid not in star_accum:
                star_accum[oid] = {
                    "sum_w": 0.0, "sum_wm": 0.0, "n_det": 0,
                    "ra": row.get("ra", np.nan),
                    "dec": row.get("dec", np.nan),
                    "m_vals": [],
                    "w_vals": [],
                }
            sa = star_accum[oid]
            sa["sum_w"] += w[j]
            sa["sum_wm"] += w[j] * m_cal[j]
            sa["n_det"] += 1
            sa["m_vals"].append(m_cal[j])
            sa["w_vals"].append(w[j])

    print(f"    Files processed: {n_files}", flush=True)
    print(f"    Total detections: {n_dets_total:,}", flush=True)
    print(f"    Used detections: {n_dets_used:,}", flush=True)
    print(f"    Unique stars: {len(star_accum):,}", flush=True)

    # Build star catalog
    records = []
    for oid, sa in star_accum.items():
        if sa["n_det"] < 1:
            continue
        mag_mean = sa["sum_wm"] / sa["sum_w"]
        mag_err = 1.0 / np.sqrt(sa["sum_w"])

        # Chi2 for variability
        chi2 = 0.0
        for m, ww in zip(sa["m_vals"], sa["w_vals"]):
            chi2 += ww * (m - mag_mean) ** 2
        dof = max(sa["n_det"] - 1, 1)

        records.append({
            "objectid": oid,
            "ra": sa["ra"],
            "dec": sa["dec"],
            f"mag_ubercal_{band}": mag_mean,
            f"magerr_ubercal_{band}": mag_err,
            f"nobs_{band}": sa["n_det"],
            f"chi2_{band}": chi2 / dof,
        })

    star_cat = pd.DataFrame(records)

    # Build ZP table
    zp_records = []
    for node, zp in zp_dict.items():
        if zp < min_valid_zp:
            continue
        if node not in node_to_idx:
            continue
        zp_records.append({
            "expnum": node[0],
            "ccdnum": node[1],
            "band": band,
            "zp_solved": zp,
        })
    zp_table = pd.DataFrame(zp_records)

    return star_cat, zp_table


def run_catalog(band, pixels, config, output_dir, cache_dir):
    """Run the Phase 5 catalog construction.

    Parameters
    ----------
    band : str
        Filter band.
    pixels : np.ndarray
        HEALPix pixel indices.
    config : dict
        Pipeline configuration.
    output_dir : Path
        Output directory.
    cache_dir : Path
        Cache directory.

    Returns
    -------
    star_cat : pd.DataFrame
    zp_table : pd.DataFrame
    """
    nside = config["survey"]["nside_chunk"]
    phase0_dir = output_dir / f"phase0_{band}"
    phase1_dir = output_dir / f"phase1_{band}"
    phase2_dir = output_dir / f"phase2_{band}"
    phase3_dir = output_dir / f"phase3_{band}"
    phase4_dir = output_dir / f"phase4_{band}"
    phase5_dir = output_dir / f"phase5_{band}"
    phase5_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"  Loading reference data...", flush=True)
    node_to_idx, idx_to_node = build_node_index(
        phase1_dir / "connected_nodes.parquet"
    )
    zp_dict = load_zp_solution(phase3_dir, phase2_dir, mode="anchored")
    print(f"  ZP solutions: {len(zp_dict):,}", flush=True)

    starflat_corrections = load_starflat_corrections(phase4_dir)
    print(f"  Star flat groups: {len(starflat_corrections)}", flush=True)

    flagged_stars = load_flagged_stars(phase3_dir)
    print(f"  Flagged stars: {len(flagged_stars):,}", flush=True)

    exposure_mjds = load_exposure_mjds(cache_dir, band)

    # Phase 0 files (have x, y, ra, dec)
    phase0_files = sorted(phase0_dir.glob(f"detections_nside{nside}_pixel*.parquet"))
    if not phase0_files:
        phase0_files = sorted(phase0_dir.glob("*.parquet"))
    print(f"  Phase 0 files: {len(phase0_files)}", flush=True)

    # Build catalog
    print(f"\n  Building star catalog...", flush=True)
    t0 = time.time()
    star_cat, zp_table = build_star_catalog(
        phase0_files, zp_dict, starflat_corrections, flagged_stars,
        exposure_mjds, node_to_idx, config, band
    )
    cat_time = time.time() - t0
    print(f"  Catalog built in {cat_time:.1f}s", flush=True)

    # Save
    star_file = phase5_dir / f"star_catalog_{band}.parquet"
    star_cat.to_parquet(star_file, index=False)
    print(f"  Star catalog saved: {star_file}", flush=True)

    zp_file = phase5_dir / f"zeropoint_table_{band}.parquet"
    zp_table.to_parquet(zp_file, index=False)
    print(f"  ZP table saved: {zp_file}", flush=True)

    # Diagnostics
    mag_col = f"mag_ubercal_{band}"
    nobs_col = f"nobs_{band}"

    print(f"\n  {'=' * 55}", flush=True)
    print(f"  Phase 5 Catalog Summary — {band}-band", flush=True)
    print(f"  {'=' * 55}", flush=True)
    print(f"    Stars:             {len(star_cat):,}", flush=True)
    print(f"    Mag median:        {star_cat[mag_col].median():.3f}", flush=True)
    print(f"    Mag std:           {star_cat[mag_col].std():.3f}", flush=True)
    print(f"    Mag range:         [{star_cat[mag_col].min():.2f}, "
          f"{star_cat[mag_col].max():.2f}]", flush=True)
    print(f"    Nobs median:       {star_cat[nobs_col].median():.0f}", flush=True)
    print(f"    Nobs max:          {star_cat[nobs_col].max()}", flush=True)
    print(f"    ZP table entries:  {len(zp_table):,}", flush=True)

    # Validate
    n_nan = star_cat[mag_col].isna().sum()
    n_inf = np.isinf(star_cat[mag_col].values).sum()
    mag_min = star_cat[mag_col].min()
    mag_max = star_cat[mag_col].max()
    print(f"    NaN magnitudes:    {n_nan}", flush=True)
    print(f"    Inf magnitudes:    {n_inf}", flush=True)
    print(f"    Mag physical:      {15 < mag_min and mag_max < 55}", flush=True)
    print(f"  {'=' * 55}", flush=True)

    return star_cat, zp_table


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Build final ubercalibrated catalog"
    )
    parser.add_argument("--band", required=True, help="Filter band")
    parser.add_argument(
        "--test-region", action="store_true",
        help="Limit to test region",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    nside = config["survey"]["nside_chunk"]

    output_dir = Path(config["data"]["output_path"])
    cache_dir = Path(config["data"]["cache_path"])

    if args.test_region:
        pixels = get_test_region_pixels(nside)
        print(f"Test region: {len(pixels)} pixels (nside={nside})", flush=True)
    else:
        pixels = get_all_healpix_pixels(nside)
        print(f"Full sky: {len(pixels)} pixels (nside={nside})", flush=True)

    print(f"Band: {args.band}", flush=True)
    print(flush=True)

    run_catalog(args.band, pixels, config, output_dir, cache_dir)


if __name__ == "__main__":
    main()
