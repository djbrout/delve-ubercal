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

# Extinction coefficients: A_band / E(B-V)
# DECam: Schlafly & Finkbeiner (2011) Table 6, R_V=3.1
EXTINCTION_COEFFS = {
    "g": 3.237, "r": 2.176, "i": 1.595, "z": 1.217,
}

from delve_ubercal.phase0_ingest import get_test_patch_pixels, get_test_region_pixels, load_config
from delve_ubercal.phase2_solve import build_node_index, load_des_fgcm_zps, load_nsc_zpterms
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
                       des_fgcm_dict, nsc_zpterm_dict,
                       config, band, phase3_starflat=None):
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
    des_fgcm_dict : dict
        (expnum, ccdnum) -> ZP_FGCM for DES CCD-exposures.
    nsc_zpterm_dict : dict
        (expnum, ccdnum) -> zpterm from NSC chip table.
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
    # sum_wm_before: weighted sum of mag_aper4 (original NSC DR2 calibration)
    # sum_wm: weighted sum of m_cal (ubercal-corrected magnitude)
    star_accum = {}  # objectid -> {sum_w, sum_wm, sum_wm_before, n_det, ra, dec}

    median_zp = np.median([zp for zp in zp_dict.values() if zp > 1.0])
    min_valid_zp = median_zp - 5.0

    # Estimate MAGZERO from DES data: median(ZP_FGCM - zpterm) over DES CCD-exposures.
    # m_inst has MAGZERO baked in, so ZP_solved ≈ ZP_true - MAGZERO + MAGZERO = ZP_FGCM.
    # For non-DES, the "current" NSC total ZP is (zpterm + MAGZERO).
    # delta_zp_nonDES = ZP_solved - (zpterm + MAGZERO) captures the ubercal correction.
    _magzero_vals = []
    for (e, c), fgcm in des_fgcm_dict.items():
        if 25.0 < fgcm < 35.0:
            zpt = nsc_zpterm_dict.get((e, c))
            if zpt is not None and abs(zpt) < 2.0:
                _magzero_vals.append(fgcm - zpt)
    magzero_offset = float(np.median(_magzero_vals)) if _magzero_vals else 31.45
    print(f"    MAGZERO offset (from DES): {magzero_offset:.4f}", flush=True)

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

        # Calibrated magnitude: m_inst + ZP_solved - MAGZERO_offset
        # The solver ensures m_inst + ZP_solved is consistent across all
        # detections of the same star (this is the overlap constraint).
        # MAGZERO_offset removes the MAGZERO that's baked into m_inst,
        # putting magnitudes on the correct absolute scale.
        # This formula is used for ALL CCD-exposures (DES and non-DES),
        # ensuring internal consistency.
        expnums = df["expnum"].values
        ccdnums = df["ccdnum"].values
        zpterms = np.array([
            nsc_zpterm_dict.get((e, c), 0.0)
            for e, c in zip(expnums, ccdnums)
        ])
        mag_aper4 = df["m_inst"].values + zpterms
        m_cal = df["m_inst"].values + zps - magzero_offset - delta_sf
        w = 1.0 / (df["m_err"].values ** 2)

        # Star-flat-only calibration: zpterm + Phase 3 constant star flat
        # This keeps the original zpterm for the gray/large-scale component
        # and applies only the per-CCD-per-epoch correction from overlaps.
        m_starflat = mag_aper4.copy()
        if phase3_starflat is not None:
            if "mjd" not in df.columns:
                df["mjd"] = df["expnum"].map(exposure_mjds)
            for i, (_, row) in enumerate(df.iterrows()):
                mjd = row.get("mjd", np.nan)
                if np.isnan(mjd):
                    continue
                epoch = get_epoch(mjd, int(row["ccdnum"]), config)
                sf_val = phase3_starflat.get((int(row["ccdnum"]), epoch), 0.0)
                m_starflat[i] += sf_val

        # Accumulate per star
        for j, (_, row) in enumerate(df.iterrows()):
            oid = row["objectid"]
            if oid not in star_accum:
                star_accum[oid] = {
                    "sum_w": 0.0, "sum_wm": 0.0, "sum_wm_before": 0.0,
                    "sum_wm_sf": 0.0,
                    "n_det": 0,
                    "ra": row.get("ra", np.nan),
                    "dec": row.get("dec", np.nan),
                    "m_vals": [],
                    "w_vals": [],
                }
            sa = star_accum[oid]
            sa["sum_w"] += w[j]
            sa["sum_wm"] += w[j] * m_cal[j]
            sa["sum_wm_before"] += w[j] * mag_aper4[j]
            sa["sum_wm_sf"] += w[j] * m_starflat[j]
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
        mag_before = sa["sum_wm_before"] / sa["sum_w"]
        mag_starflat = sa["sum_wm_sf"] / sa["sum_w"]
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
            f"mag_starflat_{band}": mag_starflat,
            f"mag_before_{band}": mag_before,
            f"magerr_ubercal_{band}": mag_err,
            f"nobs_{band}": sa["n_det"],
            f"chi2_{band}": chi2 / dof,
        })

    star_cat = pd.DataFrame(records)

    # Add SFD E(B-V) and dereddened magnitudes
    if len(star_cat) > 0 and band in EXTINCTION_COEFFS:
        try:
            from dustmaps.config import config as dm_config
            dm_config["data_dir"] = str(Path(config["data"]["cache_path"]) / "dustmaps")
            from dustmaps.sfd import SFDQuery
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            sfd = SFDQuery()
            coords = SkyCoord(
                ra=star_cat["ra"].values * u.deg,
                dec=star_cat["dec"].values * u.deg,
            )
            ebv = np.asarray(sfd(coords), dtype=float)
            R = EXTINCTION_COEFFS[band]
            star_cat["ebv_sfd"] = ebv
            star_cat[f"mag_dered_{band}"] = star_cat[f"mag_ubercal_{band}"] - R * ebv
            print(f"    SFD E(B-V): median={np.median(ebv):.4f}, "
                  f"A_{band} median={np.median(R * ebv):.3f} mag", flush=True)
        except Exception as exc:
            print(f"    WARNING: SFD dereddening failed: {exc}", flush=True)

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
    zp_dict = load_zp_solution(phase3_dir, phase2_dir, mode="unanchored")
    print(f"  ZP solutions: {len(zp_dict):,}", flush=True)

    starflat_corrections = load_starflat_corrections(phase4_dir)
    print(f"  Star flat groups: {len(starflat_corrections)}", flush=True)

    # Load Phase 3 constant star flat (per-CCD-per-epoch)
    phase3_sf_file = phase3_dir / "star_flat.parquet"
    phase3_starflat = None
    if phase3_sf_file.exists():
        sf_df = pd.read_parquet(phase3_sf_file)
        phase3_starflat = {
            (int(r.ccdnum), int(r.epoch_idx)): r.flat_correction
            for _, r in sf_df.iterrows()
        }
        print(f"  Phase 3 star flat: {len(phase3_starflat)} (CCD, epoch) entries",
              flush=True)

    flagged_stars = load_flagged_stars(phase3_dir)
    print(f"  Flagged stars: {len(flagged_stars):,}", flush=True)

    exposure_mjds = load_exposure_mjds(cache_dir, band)

    # Load DES FGCM and NSC zpterms for magnitude convention fix
    des_fgcm_dict = load_des_fgcm_zps(cache_dir, band)
    print(f"  DES FGCM ZPs: {len(des_fgcm_dict):,}", flush=True)
    nsc_zpterm_dict = load_nsc_zpterms(cache_dir, band)
    print(f"  NSC zpterms: {len(nsc_zpterm_dict):,}", flush=True)

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
        exposure_mjds, node_to_idx, des_fgcm_dict, nsc_zpterm_dict,
        config, band, phase3_starflat=phase3_starflat
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
    mag_before_col = f"mag_before_{band}"
    nobs_col = f"nobs_{band}"

    # Median correction: ubercal - before
    delta_mag = star_cat[mag_col] - star_cat[mag_before_col]
    print(f"\n  {'=' * 55}", flush=True)
    print(f"  Phase 5 Catalog Summary — {band}-band", flush=True)
    print(f"  {'=' * 55}", flush=True)
    print(f"    Stars:             {len(star_cat):,}", flush=True)
    print(f"    Mag median:        {star_cat[mag_col].median():.3f}", flush=True)
    print(f"    Mag std:           {star_cat[mag_col].std():.3f}", flush=True)
    print(f"    Mag range:         [{star_cat[mag_col].min():.2f}, "
          f"{star_cat[mag_col].max():.2f}]", flush=True)
    print(f"    Ubercal correction median: {delta_mag.median()*1000:.1f} mmag", flush=True)
    print(f"    Ubercal correction RMS:    {(delta_mag**2).mean()**0.5*1000:.1f} mmag", flush=True)
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
    print(f"    Mag physical:      {10 < mag_min and mag_max < 25}", flush=True)
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
    parser.add_argument(
        "--test-patch", action="store_true",
        help="Limit to 10x10 deg test patch RA=50-60, Dec=-35 to -25",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    nside = config["survey"]["nside_chunk"]

    output_dir = Path(config["data"]["output_path"])
    cache_dir = Path(config["data"]["cache_path"])

    if args.test_patch:
        pixels = get_test_patch_pixels(nside)
        print(f"Test patch (10x10 deg): {len(pixels)} pixels (nside={nside})", flush=True)
    elif args.test_region:
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
