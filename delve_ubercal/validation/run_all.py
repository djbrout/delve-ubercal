"""Phase 6: Run all validation tests and generate plots.

Tests:
- Test 0: FGCM vs ubercal (unanchored)
- Test 1: Photometric repeatability vs magnitude
- Test 2: DR2 vs ubercal difference (anchored vs FGCM)
- Test 4: Gaia DR3 comparison with color term
- Test 5: Stellar locus width (placeholder - requires multi-band)
- Test 6: DES boundary continuity
- Test 7: PS1 DR2 direct comparison (dec > -30, Schlafly/Magnier calibration)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delve_ubercal.phase0_ingest import get_test_patch_pixels, get_test_region_pixels, load_config
from delve_ubercal.phase2_solve import build_node_index, load_des_fgcm_zps, load_nsc_zpterms
from delve_ubercal.phase3_outlier_rejection import load_exposure_mjds
from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels

# Extinction coefficients: A_band / E(B-V)
# DECam: Schlafly & Finkbeiner (2011) Table 6, R_V=3.1
# Gaia: Casagrande & VandenBerg (2018)
# PS1: Schlafly & Finkbeiner (2011) Table 6
EXTINCTION_COEFFS = {
    "decam_g": 3.237, "decam_r": 2.176, "decam_i": 1.595, "decam_z": 1.217,
    "gaia_g": 2.740, "gaia_bp": 3.374, "gaia_rp": 1.941,
    "ps1_g": 3.172, "ps1_r": 2.271, "ps1_i": 1.682, "ps1_z": 1.322,
}


def _get_sfd_ebv(ra, dec, cache_dir):
    """Query SFD dust map for E(B-V) at given coordinates.

    Parameters
    ----------
    ra, dec : array-like
        Coordinates in degrees.
    cache_dir : Path
        Cache directory (dustmaps data at cache_dir / "dustmaps").

    Returns
    -------
    ebv : np.ndarray
        E(B-V) values from SFD (Schlegel, Finkbeiner & Davis 1998).
    """
    from dustmaps.config import config as dm_config
    dm_config["data_dir"] = str(Path(cache_dir) / "dustmaps")
    from dustmaps.sfd import SFDQuery
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    sfd = SFDQuery()
    coords = SkyCoord(ra=np.asarray(ra, dtype=float) * u.deg,
                      dec=np.asarray(dec, dtype=float) * u.deg)
    return np.asarray(sfd(coords), dtype=float)


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def _get_ccdexp_positions(output_dir, band):
    """Compute mean RA, Dec per (expnum, ccdnum) from Phase 0 detections.

    Returns DataFrame with expnum, ccdnum, ra_mean, dec_mean.
    Cached in output directory.
    """
    cache = output_dir / f"ccdexp_positions_{band}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    phase0_dir = output_dir / f"phase0_{band}"
    positions = []
    for f in sorted(phase0_dir.glob("detections_nside*_pixel*.parquet")):
        df = pd.read_parquet(f, columns=["expnum", "ccdnum", "ra", "dec"])
        pos = df.groupby(["expnum", "ccdnum"]).agg(
            ra_mean=("ra", "mean"), dec_mean=("dec", "mean")
        ).reset_index()
        positions.append(pos)

    if not positions:
        return pd.DataFrame(columns=["expnum", "ccdnum", "ra_mean", "dec_mean"])

    result = pd.concat(positions, ignore_index=True)
    # Average across pixels for CCD-exposures spanning multiple pixels
    result = result.groupby(["expnum", "ccdnum"]).agg(
        ra_mean=("ra_mean", "mean"), dec_mean=("dec_mean", "mean")
    ).reset_index()

    result.to_parquet(cache, index=False)
    return result


def test0_fgcm_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 0: Star-level FGCM vs ubercal comparison.

    For each star with DES detections, computes:
    - m_fgcm:    weighted mean of (m_inst + ZP_FGCM) across DES detections
    - m_ubercal: weighted mean of (m_inst + ZP_ubercal) across DES detections
    - m_before:  weighted mean of (m_inst + zpterm + MAGZERO_offset) across DES detections

    Compares m_ubercal vs m_fgcm and m_before vs m_fgcm (after subtracting
    the global mean offset). This is a star-level, mean-subtracted comparison.

    Returns
    -------
    metrics : dict
    """
    print("\n  === Test 0: Star-Level FGCM Comparison ===", flush=True)

    nside = 32  # from config
    phase0_dir = output_dir / f"phase0_{band}"
    phase3_dir = output_dir / f"phase3_{band}"
    phase2_dir = output_dir / f"phase2_{band}"

    # Load unanchored ZP solution
    zp_dict = {}
    source = None
    for d in [phase3_dir, phase2_dir]:
        f = d / "zeropoints_unanchored.parquet"
        if f.exists():
            df_zp = pd.read_parquet(f)
            df_zp = df_zp[df_zp["zp_solved"] > 1.0]
            zp_dict = dict(zip(
                zip(df_zp["expnum"].values, df_zp["ccdnum"].values),
                df_zp["zp_solved"].values,
            ))
            source = d.name
            break

    if not zp_dict:
        print("    No unanchored solution found.", flush=True)
        return {"status": "SKIP"}

    # Load FGCM ZPs and NSC zpterms
    des_fgcm_dict = load_des_fgcm_zps(cache_dir, band)
    nsc_zpterm_dict = load_nsc_zpterms(cache_dir, band)

    # Filter FGCM sentinels
    des_fgcm_dict = {k: v for k, v in des_fgcm_dict.items() if 25.0 < v < 35.0}

    # MAGZERO offset
    magzero_vals = []
    for (e, c), fgcm in des_fgcm_dict.items():
        zpt = nsc_zpterm_dict.get((e, c))
        if zpt is not None and abs(zpt) < 2.0:
            magzero_vals.append(fgcm - zpt)
    magzero_offset = float(np.median(magzero_vals)) if magzero_vals else 31.45
    print(f"    MAGZERO offset: {magzero_offset:.4f}", flush=True)

    # Load flagged detections (objectid, expnum, ccdnum tuples) and flagged stars
    flagged_stars = set()
    flagged_file = phase3_dir / "flagged_stars.parquet"
    if flagged_file.exists():
        flagged_stars = set(pd.read_parquet(flagged_file)["objectid"].values)

    # Load Phase 3 star flat for star-flat-only comparison
    from delve_ubercal.phase4_starflat import get_epoch
    from delve_ubercal.phase0_ingest import load_config as _load_config
    _config = _load_config(None)
    sf_file = phase3_dir / "star_flat.parquet"
    phase3_sf = {}
    if sf_file.exists():
        sf_df = pd.read_parquet(sf_file)
        phase3_sf = {
            (int(r.ccdnum), int(r.epoch_idx)): r.flat_correction
            for _, r in sf_df.iterrows()
        }
    # Load exposure MJDs for epoch lookup
    from delve_ubercal.phase3_outlier_rejection import load_exposure_mjds as _load_mjds
    _exp_mjds = _load_mjds(cache_dir, band)

    # Build lookup DataFrames for vectorized join
    fgcm_df = pd.DataFrame([
        {"expnum": e, "ccdnum": c, "zp_fgcm": v}
        for (e, c), v in des_fgcm_dict.items()
    ])
    zpterm_df = pd.DataFrame([
        {"expnum": e, "ccdnum": c, "zpterm": v}
        for (e, c), v in nsc_zpterm_dict.items()
    ])
    zp_df = pd.DataFrame([
        {"expnum": e, "ccdnum": c, "zp_ubercal": v}
        for (e, c), v in zp_dict.items()
    ])

    # Accumulate per-star magnitudes from Phase 0 detections (vectorized)
    phase0_files = sorted(phase0_dir.glob(f"detections_nside{nside}_pixel*.parquet"))
    if not phase0_files:
        phase0_files = sorted(phase0_dir.glob("*.parquet"))

    chunks = []
    n_det_total = 0
    for f in phase0_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        n_det_total += len(df)
        # Filter flagged stars
        if flagged_stars:
            df = df[~df["objectid"].isin(flagged_stars)]
        # Join FGCM (only DES detections survive)
        df = df.merge(fgcm_df, on=["expnum", "ccdnum"], how="inner")
        # Join ubercal ZPs
        df = df.merge(zp_df, on=["expnum", "ccdnum"], how="inner")
        # Join NSC zpterms
        df = df.merge(zpterm_df, on=["expnum", "ccdnum"], how="inner")
        if len(df) > 0:
            chunks.append(df)

    print(f"    Phase 0 files: {len(phase0_files)}", flush=True)
    print(f"    Total detections: {n_det_total:,}", flush=True)

    if not chunks:
        print("    No DES detections found.", flush=True)
        return {"status": "SKIP"}

    det = pd.concat(chunks, ignore_index=True)
    print(f"    DES detections used: {len(det):,}", flush=True)

    # Compute weighted magnitudes
    det["w"] = 1.0 / (det["m_err"] ** 2)
    det["wm_fgcm"] = det["w"] * (det["m_inst"] + det["zp_fgcm"])
    det["wm_ubercal"] = det["w"] * (det["m_inst"] + det["zp_ubercal"])
    det["wm_before"] = det["w"] * (det["m_inst"] + det["zpterm"] + magzero_offset)

    # Star-flat-only: zpterm + MAGZERO_offset + Phase 3 star flat
    if phase3_sf:
        sf_corrections = np.zeros(len(det))
        for i, (_, row) in enumerate(det.iterrows()):
            mjd = _exp_mjds.get(int(row["expnum"]))
            if mjd is not None:
                epoch = get_epoch(mjd, int(row["ccdnum"]), _config)
                sf_corrections[i] = phase3_sf.get((int(row["ccdnum"]), epoch), 0.0)
        det["wm_starflat"] = det["w"] * (det["m_inst"] + det["zpterm"]
                                          + magzero_offset + sf_corrections)
    else:
        det["wm_starflat"] = det["wm_before"]

    # Group by star
    grouped = det.groupby("objectid").agg(
        sum_w=("w", "sum"),
        sum_wm_fgcm=("wm_fgcm", "sum"),
        sum_wm_ubercal=("wm_ubercal", "sum"),
        sum_wm_before=("wm_before", "sum"),
        sum_wm_starflat=("wm_starflat", "sum"),
        n_des=("w", "count"),
        ra=("ra", "first"),
        dec=("dec", "first"),
    )
    print(f"    Stars with DES data: {len(grouped):,}", flush=True)

    # Only stars with >= 2 DES detections
    stars = grouped[grouped["n_des"] >= 2].copy()
    stars["mag_fgcm"] = stars["sum_wm_fgcm"] / stars["sum_w"]
    stars["mag_ubercal"] = stars["sum_wm_ubercal"] / stars["sum_w"]
    stars["mag_before"] = stars["sum_wm_before"] / stars["sum_w"]
    stars["mag_starflat"] = stars["sum_wm_starflat"] / stars["sum_w"]

    if len(stars) < 50:
        print("    Too few stars with DES data.", flush=True)
        return {"status": "SKIP"}

    stars = stars.reset_index()
    print(f"    Stars with >= 2 DES detections: {len(stars):,}", flush=True)

    # Compute differences (mmag), subtract the mean offset
    delta_ubercal = (stars["mag_ubercal"] - stars["mag_fgcm"]).values * 1000
    delta_before = (stars["mag_before"] - stars["mag_fgcm"]).values * 1000
    delta_starflat = (stars["mag_starflat"] - stars["mag_fgcm"]).values * 1000

    # Subtract mean offset (we only care about scatter, not absolute calibration)
    delta_ubercal -= np.mean(delta_ubercal)
    delta_before -= np.mean(delta_before)
    delta_starflat -= np.mean(delta_starflat)

    rms_ubercal = np.sqrt(np.mean(delta_ubercal ** 2))
    rms_before = np.sqrt(np.mean(delta_before ** 2))
    rms_starflat = np.sqrt(np.mean(delta_starflat ** 2))

    print(f"    Ubercal vs FGCM (mean-subtracted):    {rms_ubercal:.1f} mmag RMS", flush=True)
    print(f"    Star-flat only vs FGCM (mean-sub):    {rms_starflat:.1f} mmag RMS", flush=True)
    print(f"    Before vs FGCM (mean-subtracted):     {rms_before:.1f} mmag RMS", flush=True)
    improvement = rms_before - rms_ubercal
    print(f"    Ubercal improvement: {improvement:.1f} mmag ({improvement/rms_before*100:.0f}%)",
          flush=True)
    sf_improvement = rms_before - rms_starflat
    print(f"    Star-flat improvement: {sf_improvement:.1f} mmag ({sf_improvement/rms_before*100:.0f}%)",
          flush=True)

    # === Plot: histogram + sky maps (before, star-flat-only, full ubercal) ===
    fig, axes = plt.subplots(1, 4, figsize=(24, 4.5))

    # Histogram
    ax = axes[0]
    ax.hist(delta_before, bins=80, color="steelblue", edgecolor="none", alpha=0.4,
            label=f"Before: {rms_before:.1f} mmag")
    ax.hist(delta_starflat, bins=80, color="green", edgecolor="none", alpha=0.5,
            label=f"Star-flat only: {rms_starflat:.1f} mmag")
    ax.hist(delta_ubercal, bins=80, color="darkorange", edgecolor="none", alpha=0.5,
            label=f"Full ubercal: {rms_ubercal:.1f} mmag")
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"$\\Delta${band} vs FGCM (mmag, mean-subtracted)")
    ax.set_ylabel("Stars")
    ax.set_title(f"Test 0: Star-Level FGCM — {band}-band ({len(stars):,} stars)")
    ax.legend(fontsize=8)

    # Use same color scale for all sky maps
    vmax = max(
        np.percentile(np.abs(delta_before), 95),
        np.percentile(np.abs(delta_ubercal), 95),
    )
    vmax = min(50, vmax)

    # Sky map: BEFORE ubercal
    ax = axes[1]
    sc = ax.scatter(stars["ra"].values, stars["dec"].values, c=delta_before,
                     s=1, alpha=0.5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="mmag")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"BEFORE ubercal ({rms_before:.1f} mmag)")
    ax.invert_xaxis()

    # Sky map: STAR-FLAT ONLY
    ax = axes[2]
    sc = ax.scatter(stars["ra"].values, stars["dec"].values, c=delta_starflat,
                     s=1, alpha=0.5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="mmag")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"STAR-FLAT only ({rms_starflat:.1f} mmag)")
    ax.invert_xaxis()

    # Sky map: FULL ubercal
    ax = axes[3]
    sc = ax.scatter(stars["ra"].values, stars["dec"].values, c=delta_ubercal,
                     s=1, alpha=0.5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="mmag")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"FULL ubercal ({rms_ubercal:.1f} mmag)")
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_fgcm_comparison_{band}.pdf",
                bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_fgcm_comparison_{band}.png",
                bbox_inches="tight")
    plt.close(fig)

    # === Plot 2: FGCM sky map (CCD-exposure level) ===
    positions = _get_ccdexp_positions(output_dir, band)
    # Load ZP dataframe for sky map
    for d in [phase3_dir, phase2_dir]:
        fp = d / "zeropoints_unanchored.parquet"
        if fp.exists():
            df_unanch = pd.read_parquet(fp)
            break
    df_anch = None
    for d in [phase3_dir, phase2_dir]:
        fp = d / "zeropoints_anchored.parquet"
        if fp.exists():
            df_anch = pd.read_parquet(fp)
            break
    if len(positions) > 0:
        _plot_fgcm_skymap(df_unanch, df_anch, positions, band, fig_dir)

    passed = rms_ubercal < rms_before
    best_rms = min(rms_ubercal, rms_starflat)
    best_label = "ubercal" if rms_ubercal <= rms_starflat else "starflat"
    result = {
        "test": "Test 0: FGCM comparison",
        "rms_ubercal_mmag": rms_ubercal,
        "rms_starflat_mmag": rms_starflat,
        "rms_before_mmag": rms_before,
        "improvement_mmag": improvement,
        "n_stars": len(stars),
        "threshold": "ubercal < before",
        "passed": passed,
        "status": (f"{'PASS' if passed else 'FAIL'} "
                   f"(ubercal={rms_ubercal:.1f}, starflat={rms_starflat:.1f}, "
                   f"before={rms_before:.1f} mmag)"),
    }
    return result


def _plot_fgcm_skymap(df_unanch, df_anch, positions, band, fig_dir):
    """Plot FGCM residual sky maps for unanchored and anchored solutions."""
    ncols = 2 if df_anch is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for i, (df, label) in enumerate([(df_unanch, "Unanchored"),
                                      (df_anch, "Anchored")]):
        if df is None:
            continue
        ax = axes[i]
        merged = df.merge(positions, on=["expnum", "ccdnum"], how="left")
        valid = (merged["zp_fgcm"].notna() & (merged["zp_solved"] > 1.0)
                 & (merged["zp_fgcm"] > 25.0) & (merged["zp_fgcm"] < 35.0)
                 & merged["ra_mean"].notna())
        if valid.sum() == 0:
            continue
        diff = (merged.loc[valid, "zp_solved"] - merged.loc[valid, "zp_fgcm"]).values * 1000
        ra = merged.loc[valid, "ra_mean"].values
        dec = merged.loc[valid, "dec_mean"].values
        rms = np.sqrt(np.mean(diff**2))
        vmax = min(50, np.percentile(np.abs(diff), 95))
        sc = ax.scatter(ra, dec, c=diff, s=1, alpha=0.5,
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="mmag")
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(f"{label}: FGCM residual ({rms:.1f} mmag RMS)")
        ax.invert_xaxis()

    fig.suptitle(f"FGCM Sky Map — {band}-band", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fgcm_skymap_{band}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"fgcm_skymap_{band}.png", bbox_inches="tight")
    plt.close(fig)


def test1_repeatability(output_dir, cache_dir, band, fig_dir):
    """Test 1: Photometric repeatability vs magnitude.

    Returns
    -------
    metrics : dict
    """
    print("\n  === Test 1: Photometric Repeatability ===", flush=True)

    phase5_dir = output_dir / f"phase5_{band}"
    cat_file = phase5_dir / f"star_catalog_{band}.parquet"
    if not cat_file.exists():
        print("    No catalog found.", flush=True)
        return {"status": "SKIP"}

    df = pd.read_parquet(cat_file)
    mag_col = f"mag_ubercal_{band}"
    err_col = f"magerr_ubercal_{band}"
    nobs_col = f"nobs_{band}"
    chi2_col = f"chi2_{band}"

    # Filter to stars with >= 3 detections
    multi = df[df[nobs_col] >= 3].copy()
    if len(multi) == 0:
        return {"status": "SKIP"}

    # Repeatability: use chi2 to estimate per-detection scatter
    # RMS ~ sqrt(chi2 * sigma_mean^2 * (n-1))
    # Actually: err_col is error on mean, scatter = err * sqrt(n)
    multi["scatter_mmag"] = multi[err_col].values * np.sqrt(
        multi[nobs_col].values
    ) * 1000

    # Bright star floor: use brightest 20% of stars
    # (magnitudes may be ~50 for internal m_inst + ZP representation)
    mag_20pct = multi[mag_col].quantile(0.20)
    mag_40pct = multi[mag_col].quantile(0.40)
    bright = multi[(multi[mag_col] >= mag_20pct) & (multi[mag_col] < mag_40pct)]
    if len(bright) > 0:
        floor = np.median(bright["scatter_mmag"])
    else:
        floor = np.nan

    print(f"    Stars with >= 3 obs: {len(multi):,}", flush=True)
    print(f"    Bright star floor: {floor:.1f} mmag", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(multi[mag_col].values, multi["scatter_mmag"].values,
               s=1, alpha=0.1, color="steelblue")
    ax.axhline(floor, color="red", ls="--", lw=1.5,
               label=f"floor = {floor:.1f} mmag")
    ax.set_xlabel(f"mag$_{{\\rm ubercal,{band}}}$")
    ax.set_ylabel("Per-detection scatter (mmag)")
    ax.set_title("Photometric Repeatability")
    ax.set_ylim(0, min(100, multi["scatter_mmag"].quantile(0.99)))
    ax.legend()

    ax = axes[1]
    mag_lo = multi[mag_col].quantile(0.05)
    mag_hi = multi[mag_col].quantile(0.95)
    mag_bins = np.arange(mag_lo, mag_hi, (mag_hi - mag_lo) / 15)
    bin_medians = []
    bin_centers = []
    for i in range(len(mag_bins) - 1):
        mask = (multi[mag_col] >= mag_bins[i]) & (multi[mag_col] < mag_bins[i+1])
        if mask.sum() > 5:
            bin_medians.append(np.median(multi.loc[mask, "scatter_mmag"]))
            bin_centers.append((mag_bins[i] + mag_bins[i+1]) / 2)
    ax.plot(bin_centers, bin_medians, "o-", color="steelblue", lw=2)
    ax.axhline(floor, color="red", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel(f"mag$_{{\\rm ubercal,{band}}}$")
    ax.set_ylabel("Median scatter (mmag)")
    ax.set_title("Repeatability vs Magnitude")

    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_repeatability_{band}.pdf",
                bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_repeatability_{band}.png",
                bbox_inches="tight")
    plt.close(fig)

    passed = floor < 10.0
    return {
        "test": "Test 1: Repeatability",
        "floor_mmag": float(floor),
        "n_stars": len(multi),
        "threshold": "< 10 mmag",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }


def test2_dr2_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 2: DR2 vs ubercal difference.

    For the test region we compare against the FGCM values as a proxy,
    since we don't have the full DR2 catalog downloaded.
    """
    print("\n  === Test 2: DR2 Comparison (via FGCM) ===", flush=True)

    phase3_dir = output_dir / f"phase3_{band}"
    phase2_dir = output_dir / f"phase2_{band}"

    # Load anchored solution
    for d in [phase3_dir, phase2_dir]:
        f = d / "zeropoints_anchored.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            break
    else:
        return {"status": "SKIP"}

    des_mask = (df["zp_fgcm"].notna() & (df["zp_solved"] > 1.0)
                & (df["zp_fgcm"] > 25.0) & (df["zp_fgcm"] < 35.0))
    if des_mask.sum() == 0:
        return {"status": "SKIP"}

    diff = (df.loc[des_mask, "delta_zp"]).values * 1000
    rms = np.sqrt(np.mean(diff ** 2))
    median = np.median(diff)

    print(f"    N DES CCD-exp: {des_mask.sum():,}", flush=True)
    print(f"    Delta ZP RMS: {rms:.1f} mmag", flush=True)
    print(f"    Delta ZP median: {median:.1f} mmag", flush=True)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diff, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.axvline(median, color="red", ls="--", lw=1.5,
               label=f"median = {median:.1f} mmag")
    ax.set_xlabel("$\\Delta$ZP (ubercal $-$ FGCM) (mmag)")
    ax.set_ylabel("DES CCD-exposures")
    ax.set_title(f"Test 2: Anchored Solution vs FGCM ({rms:.1f} mmag RMS)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_dr2_comparison_{band}.pdf",
                bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_dr2_comparison_{band}.png",
                bbox_inches="tight")
    plt.close(fig)

    passed = rms < 15.0
    return {
        "test": "Test 2: DR2 comparison",
        "rms_mmag": rms,
        "median_mmag": median,
        "threshold": "< 15 mmag (DES interior)",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }


def _query_ls_dr10_crossmatch(cat, band, cache_dir):
    """Query LS DR10 tractor photometry crossmatched to our catalog stars.

    Uses the pre-built NSC-LS crossmatch table on Data Lab.
    Returns DataFrame with ubercal mag, LS mag, colors, positions.
    """
    from dl import queryClient as qc

    cache_file = cache_dir / f"ls_dr10_xmatch_{band}.parquet"
    if cache_file.exists():
        print("    LS DR10 crossmatch cached.", flush=True)
        return pd.read_parquet(cache_file)

    ra_min, ra_max = cat["ra"].min() - 0.1, cat["ra"].max() + 0.1
    dec_min, dec_max = cat["dec"].min() - 0.1, cat["dec"].max() + 0.1

    print(f"    Querying LS DR10 tractor (PSF, type=PSF)...", flush=True)
    query = f"""
    SELECT t.ls_id, t.ra, t.dec, t.type,
           t.mag_g, t.mag_r, t.mag_i, t.mag_z,
           t.flux_g, t.flux_r, t.flux_i, t.flux_z,
           t.flux_ivar_g, t.flux_ivar_r, t.flux_ivar_i, t.flux_ivar_z,
           t.dered_mag_g, t.dered_mag_r, t.dered_mag_i, t.dered_mag_z,
           t.gaia_phot_g_mean_mag, t.gaia_phot_bp_mean_mag, t.gaia_phot_rp_mean_mag,
           x.id2 as nsc_objectid, x.distance as xmatch_dist
    FROM ls_dr10.tractor t
    JOIN ls_dr10.x1p5__tractor__nsc_dr2__object x ON t.ls_id = x.id1
    WHERE t.type = 'PSF'
      AND t.ra BETWEEN {ra_min:.4f} AND {ra_max:.4f}
      AND t.dec BETWEEN {dec_min:.4f} AND {dec_max:.4f}
      AND x.distance < 1.0
      AND t.flux_ivar_{band} > 0
    """
    result = qc.query(sql=query, fmt="pandas", timeout=600)
    print(f"    Got {len(result):,} LS DR10 matches.", flush=True)

    if len(result) > 0:
        result.to_parquet(cache_file, index=False)
    return result


def _query_gaia_crossmatch(cat, cache_dir):
    """Query Gaia DR3 crossmatched to our catalog stars.

    Uses the pre-built Gaia-NSC crossmatch table on Data Lab.
    """
    from dl import queryClient as qc

    cache_file = cache_dir / "gaia_dr3_xmatch.parquet"
    if cache_file.exists():
        print("    Gaia DR3 crossmatch cached.", flush=True)
        return pd.read_parquet(cache_file)

    ra_min, ra_max = cat["ra"].min() - 0.1, cat["ra"].max() + 0.1
    dec_min, dec_max = cat["dec"].min() - 0.1, cat["dec"].max() + 0.1

    print(f"    Querying Gaia DR3...", flush=True)
    query = f"""
    SELECT g.source_id, g.ra, g.dec,
           g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag,
           g.bp_rp,
           g.phot_g_mean_flux_over_error, g.phot_bp_mean_flux_over_error,
           g.phot_rp_mean_flux_over_error,
           g.ag_gspphot, g.ebpminrp_gspphot,
           x.id2 as nsc_objectid, x.distance as xmatch_dist
    FROM gaia_dr3.gaia_source g
    JOIN gaia_dr3.x1p5__gaia_source__nsc_dr2__object x ON g.source_id = x.id1
    WHERE g.ra BETWEEN {ra_min:.4f} AND {ra_max:.4f}
      AND g.dec BETWEEN {dec_min:.4f} AND {dec_max:.4f}
      AND x.distance < 1.0
      AND g.phot_g_mean_flux_over_error > 50
      AND g.phot_bp_mean_flux_over_error > 10
      AND g.phot_rp_mean_flux_over_error > 10
    """
    result = qc.query(sql=query, fmt="pandas", timeout=600)
    print(f"    Got {len(result):,} Gaia DR3 matches.", flush=True)

    if len(result) > 0:
        result.to_parquet(cache_file, index=False)
    return result


def _fit_color_term(color, delta_mag, sigma_clip=3.0, n_iter=3):
    """Fit a linear color term: delta_mag = a + b * color.

    Returns (a, b, a_err, b_err, mask) after iterative sigma clipping.
    """
    mask = np.isfinite(color) & np.isfinite(delta_mag)
    for _ in range(n_iter):
        if mask.sum() < 10:
            return np.nan, np.nan, np.nan, np.nan, mask
        c = color[mask]
        d = delta_mag[mask]
        # Least squares: d = a + b*c
        A = np.vstack([np.ones(len(c)), c]).T
        result = np.linalg.lstsq(A, d, rcond=None)
        coeffs = result[0]
        a, b = coeffs[0], coeffs[1]
        resid = d - (a + b * c)
        rms = np.sqrt(np.mean(resid**2))
        # Sigma clip
        full_resid = delta_mag - (a + b * color)
        mask = np.isfinite(full_resid) & (np.abs(full_resid) < sigma_clip * rms)

    # Final fit
    c = color[mask]
    d = delta_mag[mask]
    A = np.vstack([np.ones(len(c)), c]).T
    coeffs = np.linalg.lstsq(A, d, rcond=None)[0]
    a, b = coeffs[0], coeffs[1]
    resid = d - (a + b * c)
    rms = np.sqrt(np.mean(resid**2))
    # Parameter uncertainties
    if len(c) > 2:
        cov = np.linalg.inv(A.T @ A) * rms**2
        a_err, b_err = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
    else:
        a_err, b_err = np.nan, np.nan
    return a, b, a_err, b_err, mask


def _fit_color_term_poly(color, delta_mag, degree=3, sigma_clip=3.0, n_iter=3):
    """Fit a polynomial color term: delta_mag = sum(c_i * color^i).

    Better than linear for Gaia comparisons where the filter mismatch
    produces a non-linear color dependence.

    Returns (coeffs, mask, predict_func) where predict_func(color) gives
    the model prediction.
    """
    mask = np.isfinite(color) & np.isfinite(delta_mag)
    coeffs = None
    for _ in range(n_iter):
        if mask.sum() < degree + 5:
            return None, mask, lambda x: np.zeros_like(x)
        c = color[mask]
        d = delta_mag[mask]
        coeffs = np.polyfit(c, d, degree)
        model = np.polyval(coeffs, color)
        resid = delta_mag - model
        rms = np.sqrt(np.nanmean(resid[mask]**2))
        mask = np.isfinite(resid) & (np.abs(resid) < sigma_clip * rms)

    # Final fit
    if mask.sum() < degree + 5:
        return None, mask, lambda x: np.zeros_like(x)
    c = color[mask]
    d = delta_mag[mask]
    coeffs = np.polyfit(c, d, degree)

    def predict_func(x):
        return np.polyval(coeffs, x)

    return coeffs, mask, predict_func


def test3_ls_dr10_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 3: LS DR10 (PS1-calibrated) comparison with color term.

    Compares ubercal magnitudes to LS DR10 PSF magnitudes for point sources.
    LS DR10 is calibrated to PS1 in the north (dec > -30) and provides
    model PSF photometry in griz.
    """
    print("\n  === Test 3: LS DR10 Comparison (color term) ===", flush=True)

    phase5_dir = output_dir / f"phase5_{band}"
    cat_file = phase5_dir / f"star_catalog_{band}.parquet"
    if not cat_file.exists():
        print("    No catalog found.", flush=True)
        return {"status": "SKIP", "test": "Test 3: LS DR10 comparison"}

    cat = pd.read_parquet(cat_file)
    mag_col = f"mag_ubercal_{band}"
    nobs_col = f"nobs_{band}"

    # Filter to well-measured stars
    cat = cat[(cat[nobs_col] >= 3) & cat[mag_col].notna()].copy()

    try:
        ls = _query_ls_dr10_crossmatch(cat, band, cache_dir)
    except Exception as exc:
        print(f"    LS DR10 query failed: {exc}", flush=True)
        return {"status": "SKIP (query failed)", "test": "Test 3: LS DR10 comparison"}

    if len(ls) == 0:
        return {"status": "SKIP (no matches)", "test": "Test 3: LS DR10 comparison"}

    # Merge
    merged = cat.merge(ls, left_on="objectid", right_on="nsc_objectid", how="inner",
                        suffixes=("_ubercal", "_ls"))
    print(f"    Matched stars: {len(merged):,}", flush=True)
    if len(merged) < 50:
        return {"status": "SKIP (too few matches)", "test": "Test 3: LS DR10 comparison"}

    ls_mag_col = f"mag_{band}"
    if ls_mag_col not in merged.columns:
        return {"status": "SKIP (no LS mag)", "test": "Test 3: LS DR10 comparison"}

    delta = (merged[mag_col] - merged[ls_mag_col]).values * 1000  # mmag
    # Color: use LS g-i or g-r depending on what's available
    if "mag_g" in merged.columns and "mag_i" in merged.columns:
        color = (merged["mag_g"] - merged["mag_i"]).values
        color_label = "(g-i)$_{\\rm LS}$"
    elif "mag_g" in merged.columns and "mag_r" in merged.columns:
        color = (merged["mag_g"] - merged["mag_r"]).values
        color_label = "(g-r)$_{\\rm LS}$"
    else:
        color = np.zeros(len(merged))
        color_label = "color"

    # Fit color term
    a, b, a_err, b_err, fit_mask = _fit_color_term(color, delta)
    resid_after = delta - (a + b * color)
    rms_before = np.sqrt(np.nanmean(delta[fit_mask]**2))
    rms_after = np.sqrt(np.nanmean(resid_after[fit_mask]**2))

    print(f"    Color term: {a:.1f} + {b:.1f} * {color_label} mmag", flush=True)
    print(f"    RMS before color term: {rms_before:.1f} mmag", flush=True)
    print(f"    RMS after color term: {rms_after:.1f} mmag", flush=True)

    # Plot: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0): Histogram of residuals
    ax = axes[0, 0]
    ax.hist(delta[fit_mask], bins=80, color="steelblue", edgecolor="none", alpha=0.7,
            label=f"before: {rms_before:.1f} mmag")
    ax.hist(resid_after[fit_mask], bins=80, color="darkorange", edgecolor="none", alpha=0.7,
            label=f"after: {rms_after:.1f} mmag")
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"$\\Delta${band} (ubercal $-$ LS DR10) (mmag)")
    ax.set_ylabel("Stars")
    ax.set_title(f"LS DR10 Comparison — {band}-band")
    ax.legend(fontsize=9)

    # (0,1): Delta mag vs color (color term)
    ax = axes[0, 1]
    ax.scatter(color[fit_mask], delta[fit_mask], s=1, alpha=0.2, color="steelblue")
    color_grid = np.linspace(np.nanpercentile(color[fit_mask], 2),
                              np.nanpercentile(color[fit_mask], 98), 100)
    ax.plot(color_grid, a + b * color_grid, "r-", lw=2,
            label=f"CT = {a:.1f} + {b:.1f} * color")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(color_label)
    ax.set_ylabel(f"$\\Delta${band} (mmag)")
    ax.set_title("Color Term Fit")
    ax.legend(fontsize=9)

    # (1,0): Sky map of residuals (after color term)
    ax = axes[1, 0]
    ra_plot = merged.loc[fit_mask, "ra_ubercal" if "ra_ubercal" in merged.columns else "ra"].values
    dec_plot = merged.loc[fit_mask, "dec_ubercal" if "dec_ubercal" in merged.columns else "dec"].values
    vmax = min(30, np.percentile(np.abs(resid_after[fit_mask]), 95))
    sc = ax.scatter(ra_plot, dec_plot, c=resid_after[fit_mask],
                     s=1, alpha=0.5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="mmag")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Spatial Residual (after CT)")
    ax.invert_xaxis()

    # (1,1): Residual vs magnitude
    ax = axes[1, 1]
    mag_plot = merged.loc[fit_mask, mag_col].values
    ax.scatter(mag_plot, resid_after[fit_mask], s=1, alpha=0.2, color="steelblue")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"mag$_{{\\rm ubercal,{band}}}$")
    ax.set_ylabel(f"$\\Delta${band} after CT (mmag)")
    ax.set_title("Magnitude Dependence")

    fig.suptitle(f"Test 3: DELVE Ubercal vs LS DR10 — {band}-band", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_ls_dr10_{band}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_ls_dr10_{band}.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "test": "Test 3: LS DR10 comparison",
        "rms_before_ct_mmag": float(rms_before),
        "rms_after_ct_mmag": float(rms_after),
        "color_term_offset": float(a),
        "color_term_slope": float(b),
        "n_stars": int(fit_mask.sum()),
        "status": f"RMS={rms_after:.1f} mmag after CT",
        "passed": rms_after < 30.0,
    }


def test4_gaia_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 4: Gaia DR3 comparison using G_BP (g-band) or G_RP (riz) with
    DELVE g-r color term.

    Uses DELVE g-r color (not Gaia BP-RP) for the color term fit, since the
    color axis should be based on the survey being validated, not the reference.
    Falls back to Gaia BP-RP if DELVE g-r is not available.
    """
    # Choose Gaia band: G_BP for g-band, G_RP for riz
    if band in ("g",):
        gaia_mag_col = "phot_bp_mean_mag"
        gaia_label = "G$_{\\rm BP}$"
    else:
        gaia_mag_col = "phot_rp_mean_mag"
        gaia_label = "G$_{\\rm RP}$"

    print(f"\n  === Test 4: Gaia DR3 Comparison ({gaia_label}, DELVE g$-$r color term) ===",
          flush=True)

    phase5_dir = output_dir / f"phase5_{band}"
    cat_file = phase5_dir / f"star_catalog_{band}.parquet"
    if not cat_file.exists():
        print("    No catalog found.", flush=True)
        return {"status": "SKIP", "test": "Test 4: Gaia comparison"}

    cat = pd.read_parquet(cat_file)
    mag_col = f"mag_ubercal_{band}"
    nobs_col = f"nobs_{band}"
    cat = cat[(cat[nobs_col] >= 3) & cat[mag_col].notna()].copy()
    # Filter unphysical magnitudes
    cat = cat[(cat[mag_col] > 15) & (cat[mag_col] < 22)].copy()

    # Try to load DELVE g and r catalogs for g-r color
    color_label = "(BP$-$RP)$_{\\rm Gaia}$"  # fallback
    use_delve_color = False
    g_cat_file = output_dir / "phase5_g" / "star_catalog_g.parquet"
    r_cat_file = output_dir / "phase5_r" / "star_catalog_r.parquet"
    if g_cat_file.exists() and r_cat_file.exists():
        g_cat = pd.read_parquet(g_cat_file, columns=["objectid", "mag_ubercal_g"])
        r_cat = pd.read_parquet(r_cat_file, columns=["objectid", "mag_ubercal_r"])
        gr_merged = g_cat.merge(r_cat, on="objectid", how="inner")
        gr_merged["delve_g_r"] = gr_merged["mag_ubercal_g"] - gr_merged["mag_ubercal_r"]
        # Filter physical colors
        gr_merged = gr_merged[
            (gr_merged["delve_g_r"] > -0.5) & (gr_merged["delve_g_r"] < 2.5) &
            (gr_merged["mag_ubercal_g"] > 15) & (gr_merged["mag_ubercal_g"] < 22) &
            (gr_merged["mag_ubercal_r"] > 15) & (gr_merged["mag_ubercal_r"] < 22)
        ].copy()
        cat = cat.merge(gr_merged[["objectid", "delve_g_r"]], on="objectid", how="left")
        n_with_color = cat["delve_g_r"].notna().sum()
        print(f"    DELVE g-r available for {n_with_color:,} / {len(cat):,} stars", flush=True)
        if n_with_color > 1000:
            use_delve_color = True
            color_label = "$(g-r)_{\\rm DELVE}$"

    try:
        gaia = _query_gaia_crossmatch(cat, cache_dir)
    except Exception as exc:
        print(f"    Gaia query failed: {exc}", flush=True)
        return {"status": "SKIP (query failed)", "test": "Test 4: Gaia comparison"}

    if len(gaia) == 0:
        return {"status": "SKIP (no matches)", "test": "Test 4: Gaia comparison"}

    # Filter Gaia for valid BP/RP mags
    gaia = gaia[gaia[gaia_mag_col].notna() & (gaia[gaia_mag_col] > 10) &
                (gaia[gaia_mag_col] < 25)].copy()

    merged = cat.merge(gaia, left_on="objectid", right_on="nsc_objectid", how="inner",
                        suffixes=("_ubercal", "_gaia"))
    print(f"    Matched stars: {len(merged):,}", flush=True)
    if len(merged) < 50:
        return {"status": "SKIP (too few matches)", "test": "Test 4: Gaia comparison"}

    # Deredden both DELVE and Gaia magnitudes using SFD E(B-V)
    ra_col = "ra_ubercal" if "ra_ubercal" in merged.columns else "ra"
    dec_col = "dec_ubercal" if "dec_ubercal" in merged.columns else "dec"
    ebv = _get_sfd_ebv(merged[ra_col].values, merged[dec_col].values, cache_dir)
    R_decam = EXTINCTION_COEFFS[f"decam_{band}"]
    R_gaia = EXTINCTION_COEFFS["gaia_bp"] if band == "g" else EXTINCTION_COEFFS["gaia_rp"]
    print(f"    Dereddening: R_DECam={R_decam:.3f}, R_Gaia={R_gaia:.3f}, "
          f"E(B-V) median={np.median(ebv):.4f}", flush=True)

    mag_delve_dered = merged[mag_col].values - R_decam * ebv
    mag_gaia_dered = merged[gaia_mag_col].values - R_gaia * ebv
    delta = (mag_delve_dered - mag_gaia_dered) * 1000  # mmag

    # Also compute "before ubercal" residuals if mag_before column exists
    mag_before_col = f"mag_before_{band}"
    has_before = mag_before_col in merged.columns
    if has_before:
        mag_before_dered = merged[mag_before_col].values - R_decam * ebv
        delta_before_ubercal = (mag_before_dered - mag_gaia_dered) * 1000

    if use_delve_color and "delve_g_r" in merged.columns:
        # Deredden DELVE g-r color
        color_raw = merged["delve_g_r"].values
        R_g = EXTINCTION_COEFFS["decam_g"]
        R_r = EXTINCTION_COEFFS["decam_r"]
        color = color_raw - (R_g - R_r) * ebv
        print(f"    Using dereddened DELVE g-r color for color term fit", flush=True)
    else:
        # Deredden Gaia BP-RP
        color_raw = merged["bp_rp"].values
        R_bp = EXTINCTION_COEFFS["gaia_bp"]
        R_rp = EXTINCTION_COEFFS["gaia_rp"]
        color = color_raw - (R_bp - R_rp) * ebv
        color_label = "(BP$-$RP)$_{\\rm Gaia,0}$"
        print(f"    Falling back to dereddened Gaia BP-RP (no DELVE g-r)", flush=True)

    # Fit polynomial color term (degree 5 — non-linear for Gaia filter mismatch)
    ct_degree = 5
    coeffs, fit_mask, predict_func = _fit_color_term_poly(
        color, delta, degree=ct_degree)
    model = predict_func(color)
    resid_after = delta - model
    # Build common mask: stars valid in BOTH before and after (apples-to-apples)
    common_mask = fit_mask.copy()
    if has_before:
        common_mask &= np.isfinite(delta_before_ubercal)

    rms_before_ct = np.sqrt(np.nanmean(delta[common_mask]**2))
    rms_after_ct = np.sqrt(np.nanmean(resid_after[common_mask]**2))

    # Apply same color term to "before ubercal" data (same stars)
    rms_before_ubercal = None
    if has_before:
        resid_before_ubercal = delta_before_ubercal - model
        rms_before_ubercal = np.sqrt(np.nanmean(resid_before_ubercal[common_mask]**2))
        print(f"    Before ubercal RMS (vs Gaia, after CT): {rms_before_ubercal:.1f} mmag",
              flush=True)

    print(f"    Stars in comparison: {common_mask.sum():,}", flush=True)
    if coeffs is not None:
        coeff_str = " + ".join(
            f"{c:.1f}x^{ct_degree-i}" if ct_degree - i > 0 else f"{c:.1f}"
            for i, c in enumerate(coeffs))
        print(f"    Color term (degree {ct_degree}): {coeff_str}", flush=True)
    print(f"    RMS before color term: {rms_before_ct:.1f} mmag", flush=True)
    print(f"    RMS after color term: {rms_after_ct:.1f} mmag", flush=True)

    # Plot: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    # Overplot "before ubercal" histogram if available — same stars
    if has_before:
        ax.hist(resid_before_ubercal[common_mask], bins=80, color="steelblue",
                edgecolor="none", alpha=0.5,
                label=f"Before ubercal: RMS = {rms_before_ubercal:.1f} mmag")
    ax.hist(resid_after[common_mask], bins=80, color="darkorange", edgecolor="none", alpha=0.7,
            label=f"After ubercal: RMS = {rms_after_ct:.1f} mmag")
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"$\\Delta${band} after CT (DELVE $-$ Gaia {gaia_label}) (mmag)")
    ax.set_ylabel("Stars")
    ax.set_title(f"Gaia DR3 Comparison — {band}-band ({common_mask.sum():,} stars)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.scatter(color[common_mask], delta[common_mask], s=1, alpha=0.2, color="steelblue")
    color_grid = np.linspace(np.nanpercentile(color[common_mask], 2),
                              np.nanpercentile(color[common_mask], 98), 100)
    ax.plot(color_grid, predict_func(color_grid), "r-", lw=2,
            label=f"poly deg {ct_degree}")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(color_label)
    ax.set_ylabel(f"$\\Delta${band} (mmag)")
    ax.set_title(f"Color Term Fit (degree {ct_degree})")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ra_plot = merged.loc[common_mask, "ra_ubercal" if "ra_ubercal" in merged.columns else "ra"].values
    dec_plot = merged.loc[common_mask, "dec_ubercal" if "dec_ubercal" in merged.columns else "dec"].values
    vmax = min(30, np.percentile(np.abs(resid_after[common_mask]), 95))
    sc = ax.scatter(ra_plot, dec_plot, c=resid_after[common_mask],
                     s=1, alpha=0.5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="mmag")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Sky Map: DELVE $-$ Gaia {gaia_label} (after CT)")
    ax.invert_xaxis()

    ax = axes[1, 1]
    mag_plot = merged.loc[common_mask, mag_col].values
    ax.scatter(mag_plot, resid_after[common_mask], s=1, alpha=0.2, color="steelblue")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"mag$_{{\\rm ubercal,{band}}}$")
    ax.set_ylabel(f"$\\Delta${band} after CT (mmag)")
    ax.set_title("Magnitude Dependence")

    fig.suptitle(f"Test 4: DELVE vs Gaia DR3 {gaia_label} (dereddened) — {band}-band",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_gaia_{band}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_gaia_{band}.png", bbox_inches="tight")
    plt.close(fig)

    # Pass if ubercal improves over NSC DR2 (or if no before data, just check RMS)
    if rms_before_ubercal is not None:
        passed = rms_after_ct <= rms_before_ubercal and rms_after_ct < 50.0
        status_str = (f"RMS={rms_after_ct:.1f} mmag after CT ({gaia_label}), "
                      f"before={rms_before_ubercal:.1f}")
    else:
        passed = rms_after_ct < 50.0
        status_str = f"RMS={rms_after_ct:.1f} mmag after CT ({gaia_label})"

    return {
        "test": "Test 4: Gaia comparison",
        "gaia_band": gaia_label,
        "rms_before_ct_mmag": float(rms_before_ct),
        "rms_after_ct_mmag": float(rms_after_ct),
        "rms_before_ubercal_mmag": float(rms_before_ubercal) if rms_before_ubercal else None,
        "color_term_degree": ct_degree,
        "color_term_coeffs": [float(c) for c in coeffs] if coeffs is not None else [],
        "n_stars": int(fit_mask.sum()),
        "status": status_str,
        "passed": passed,
    }


def test5_stellar_locus(output_dir, cache_dir, band, fig_dir):
    """Test 5: Stellar locus width (placeholder).

    Full implementation requires multi-band (g, r, i) photometry.
    """
    print("\n  === Test 5: Stellar Locus ===", flush=True)
    print("    PLACEHOLDER: requires multi-band photometry",
          flush=True)
    return {
        "test": "Test 5: Stellar locus",
        "status": "SKIP (requires multi-band)",
        "passed": None,
    }


def test6_des_boundary(output_dir, cache_dir, band, fig_dir):
    """Test 6: DES boundary continuity.

    Compare ZPs inside vs outside DES footprint.
    """
    print("\n  === Test 6: DES Boundary Continuity ===", flush=True)

    phase3_dir = output_dir / f"phase3_{band}"
    phase2_dir = output_dir / f"phase2_{band}"

    for d in [phase3_dir, phase2_dir]:
        f = d / "zeropoints_anchored.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            break
    else:
        return {"status": "SKIP"}

    # Exclude flagged nodes (no data in solve → Tikhonov-default ZPs)
    flagged_exps = set()
    flagged_node_set = set()
    flagged_exp_file = phase3_dir / "flagged_exposures.parquet"
    flagged_nodes_file = phase3_dir / "flagged_nodes.parquet"
    if flagged_exp_file.exists():
        flagged_exps = set(pd.read_parquet(flagged_exp_file)["expnum"].values)
    if flagged_nodes_file.exists():
        fn = pd.read_parquet(flagged_nodes_file)
        if len(fn) > 0:
            flagged_node_set = set(zip(fn["expnum"].values, fn["ccdnum"].values))

    flagged_mask = np.array([
        e in flagged_exps or (e, c) in flagged_node_set
        for e, c in zip(df["expnum"].values, df["ccdnum"].values)
    ])

    # DES vs non-DES comparison (filter sentinel FGCM values AND flagged nodes)
    des_mask = (df["zp_fgcm"].notna() & (df["zp_solved"] > 1.0)
                & (df["zp_fgcm"] > 25.0) & (df["zp_fgcm"] < 35.0)
                & ~flagged_mask)
    non_des_mask = df["zp_fgcm"].isna() & (df["zp_solved"] > 1.0) & ~flagged_mask

    n_des = des_mask.sum()
    n_non_des = non_des_mask.sum()

    if n_des == 0 or n_non_des == 0:
        print("    Not enough DES/non-DES data for boundary test.", flush=True)
        return {"status": "SKIP (insufficient coverage)"}

    zp_des = df.loc[des_mask, "zp_solved"].values
    zp_non_des = df.loc[non_des_mask, "zp_solved"].values

    des_median = np.median(zp_des)
    non_des_median = np.median(zp_non_des)
    boundary_offset = abs(des_median - non_des_median) * 1000

    print(f"    DES CCD-exp: {n_des:,}", flush=True)
    print(f"    Non-DES CCD-exp: {n_non_des:,}", flush=True)
    print(f"    DES median ZP: {des_median:.4f}", flush=True)
    print(f"    Non-DES median ZP: {non_des_median:.4f}", flush=True)
    print(f"    Boundary offset: {boundary_offset:.1f} mmag", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(
        min(zp_des.min(), zp_non_des.min()),
        max(zp_des.max(), zp_non_des.max()),
        60,
    )
    ax.hist(zp_des, bins=bins, alpha=0.6, color="steelblue",
            edgecolor="none", label=f"DES ({n_des})")
    ax.hist(zp_non_des, bins=bins, alpha=0.6, color="darkorange",
            edgecolor="none", label=f"Non-DES ({n_non_des})")
    ax.axvline(des_median, color="steelblue", ls="--", lw=1.5)
    ax.axvline(non_des_median, color="darkorange", ls="--", lw=1.5)
    ax.set_xlabel("ZP$_{\\rm solved}$ (mag)")
    ax.set_ylabel("CCD-exposures")
    ax.set_title(f"Test 6: DES Boundary ({boundary_offset:.1f} mmag offset)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_des_boundary_{band}.pdf",
                bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_des_boundary_{band}.png",
                bbox_inches="tight")
    plt.close(fig)

    passed = boundary_offset < 10.0
    return {
        "test": "Test 6: DES boundary",
        "boundary_offset_mmag": boundary_offset,
        "n_des": int(n_des),
        "n_non_des": int(n_non_des),
        "threshold": "< 10 mmag",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }


def _query_ps1_dr2(cat, band, cache_dir):
    """Query PS1 DR2 mean photometry via MAST TAP for stars in our catalog.

    Uses pyvo TAP async to query the PS1 DR2 MeanObjectView table at MAST.
    Cross-matches by position (1 arcsec radius) with our catalog stars.
    PS1 only covers dec > -30.

    Returns DataFrame with PS1 PSF and aperture magnitudes.
    """
    import pyvo

    cache_file = cache_dir / f"ps1_dr2_xmatch_{band}.parquet"
    if cache_file.exists():
        print("    PS1 DR2 crossmatch cached.", flush=True)
        return pd.read_parquet(cache_file)

    # Only query dec > -30 (PS1 coverage)
    cat_north = cat[cat["dec"] > -30.0].copy()
    if len(cat_north) == 0:
        print("    No catalog stars with dec > -30 for PS1 comparison.", flush=True)
        return pd.DataFrame()

    ra_min, ra_max = cat_north["ra"].min() - 0.1, cat_north["ra"].max() + 0.1
    dec_min, dec_max = max(cat_north["dec"].min() - 0.1, -30.0), cat_north["dec"].max() + 0.1

    print(f"    Querying PS1 DR2 MeanObject (dec > -30)...", flush=True)
    print(f"    Region: RA=[{ra_min:.1f}, {ra_max:.1f}], Dec=[{dec_min:.1f}, {dec_max:.1f}]",
          flush=True)

    tap_url = "https://mast.stsci.edu/vo-tap/api/v0.1/ps1dr2/"
    service = pyvo.dal.TAPService(tap_url)

    import time

    # Split into 2-degree RA strips to avoid MAST TAP timeout/abort
    ra_step = 2.0
    ra_edges = np.arange(ra_min, ra_max + ra_step, ra_step)
    all_chunks = []

    for j in range(len(ra_edges) - 1):
        ra_lo = ra_edges[j]
        ra_hi = min(ra_edges[j + 1], ra_max)

        query = f"""
        SELECT m.objID, m.raMean, m.decMean, m.nDetections,
               m.gMeanPSFMag, m.gMeanPSFMagErr,
               m.rMeanPSFMag, m.rMeanPSFMagErr,
               m.iMeanPSFMag, m.iMeanPSFMagErr,
               m.zMeanPSFMag, m.zMeanPSFMagErr,
               m.gMeanApMag, m.gMeanApMagErr,
               m.rMeanApMag, m.rMeanApMagErr,
               m.iMeanApMag, m.iMeanApMagErr,
               m.zMeanApMag, m.zMeanApMagErr,
               m.gQfPerfect, m.rQfPerfect, m.iQfPerfect, m.zQfPerfect
        FROM dbo.MeanObjectView m
        WHERE m.raMean BETWEEN {ra_lo:.6f} AND {ra_hi:.6f}
          AND m.decMean BETWEEN {dec_min:.6f} AND {dec_max:.6f}
          AND m.nDetections > 1
          AND m.{band}MeanPSFMag > 0
          AND m.{band}MeanPSFMag < 22
          AND m.{band}QfPerfect > 0.85
        """

        print(f"    Chunk RA [{ra_lo:.1f}, {ra_hi:.1f}]...", flush=True)
        t0 = time.time()
        job = service.submit_job(query, maxrec=500000)
        job.run()

        # Poll for completion
        last_print = 0
        while job.phase not in ("COMPLETED", "ERROR", "ABORTED"):
            elapsed = time.time() - t0
            if elapsed - last_print >= 60:
                print(f"      Phase: {job.phase} ({elapsed:.0f}s)", flush=True)
                last_print = elapsed
            if elapsed > 1800:
                print(f"      Timeout after {elapsed:.0f}s", flush=True)
                try:
                    job.abort()
                except Exception:
                    pass
                break
            time.sleep(15)

        elapsed = time.time() - t0

        if job.phase == "COMPLETED":
            votable_result = job.fetch_result()
            chunk = votable_result.to_table().to_pandas()
            print(f"      Got {len(chunk):,} sources ({elapsed:.0f}s)", flush=True)
            if len(chunk) > 0:
                all_chunks.append(chunk)
        else:
            print(f"      FAILED: phase={job.phase} ({elapsed:.0f}s)", flush=True)

    if not all_chunks:
        print("    No PS1 sources retrieved.", flush=True)
        return pd.DataFrame()

    ps1 = pd.concat(all_chunks, ignore_index=True)
    print(f"    Total PS1 DR2 sources: {len(ps1):,}", flush=True)

    if len(ps1) == 0:
        return pd.DataFrame()

    # Positional cross-match with our catalog (1 arcsec)
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    cat_coords = SkyCoord(ra=cat_north["ra"].values * u.deg,
                          dec=cat_north["dec"].values * u.deg)
    ps1_coords = SkyCoord(ra=ps1["raMean"].values * u.deg,
                          dec=ps1["decMean"].values * u.deg)

    idx, sep, _ = cat_coords.match_to_catalog_sky(ps1_coords)
    match_mask = sep < 1.0 * u.arcsec

    if match_mask.sum() == 0:
        print("    No PS1 matches within 1 arcsec.", flush=True)
        return pd.DataFrame()

    # Build merged table
    matched_cat = cat_north.iloc[match_mask.nonzero()[0]].reset_index(drop=True)
    matched_ps1 = ps1.iloc[idx[match_mask]].reset_index(drop=True)
    matched_ps1["xmatch_dist_arcsec"] = sep[match_mask].arcsec

    result = pd.concat([
        matched_cat[["objectid", "ra", "dec",
                      *[c for c in matched_cat.columns if "mag_ubercal" in c or "mag_before" in c or "nobs" in c]]].reset_index(drop=True),
        matched_ps1.add_prefix("ps1_").reset_index(drop=True),
    ], axis=1)

    print(f"    Matched {len(result):,} stars within 1 arcsec.", flush=True)
    result.to_parquet(cache_file, index=False)
    return result


def test7_ps1_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 7: PS1 DR2 direct comparison with color term.

    Compares ubercal magnitudes to PS1 DR2 MeanPSFMag for dec > -30 stars.
    PS1 calibration is from Schlafly et al. (2012) / Magnier et al. (2020),
    accurate to ~7-12 mmag systematic floor depending on band.
    """
    print(f"\n  === Test 7: PS1 DR2 Comparison (dec > -30) ===", flush=True)

    phase5_dir = output_dir / f"phase5_{band}"
    cat_file = phase5_dir / f"star_catalog_{band}.parquet"
    if not cat_file.exists():
        print("    No catalog found.", flush=True)
        return {"status": "SKIP", "test": "Test 7: PS1 DR2 comparison"}

    cat = pd.read_parquet(cat_file)
    mag_col = f"mag_ubercal_{band}"
    nobs_col = f"nobs_{band}"

    # Filter to well-measured stars
    cat = cat[(cat[nobs_col] >= 3) & cat[mag_col].notna()].copy()

    try:
        ps1 = _query_ps1_dr2(cat, band, cache_dir)
    except Exception as exc:
        print(f"    PS1 query failed: {exc}", flush=True)
        return {"status": f"SKIP (query failed: {exc})", "test": "Test 7: PS1 DR2 comparison"}

    if len(ps1) == 0:
        return {"status": "SKIP (no matches)", "test": "Test 7: PS1 DR2 comparison"}

    ps1_mag_col = f"ps1_{band}MeanPSFMag"
    if ps1_mag_col not in ps1.columns:
        return {"status": f"SKIP (no {ps1_mag_col})", "test": "Test 7: PS1 DR2 comparison"}

    print(f"    Matched stars (raw): {len(ps1):,}", flush=True)
    if len(ps1) < 50:
        return {"status": "SKIP (too few matches)", "test": "Test 7: PS1 DR2 comparison"}

    # Filter PS1 sentinel values (-999) from ALL magnitude columns used
    ps1_g_col = "ps1_gMeanPSFMag"
    ps1_i_col = "ps1_iMeanPSFMag"
    quality = (ps1[ps1_mag_col] > 10) & (ps1[ps1_mag_col] < 25)
    if ps1_g_col in ps1.columns:
        quality &= (ps1[ps1_g_col] > 10) & (ps1[ps1_g_col] < 25)
    if ps1_i_col in ps1.columns:
        quality &= (ps1[ps1_i_col] > 10) & (ps1[ps1_i_col] < 25)
    # Also require valid DELVE magnitudes
    quality &= (ps1[mag_col] > 15) & (ps1[mag_col] < 22)
    ps1 = ps1[quality].copy().reset_index(drop=True)
    print(f"    After PS1 quality cuts: {len(ps1):,}", flush=True)
    if len(ps1) < 50:
        return {"status": "SKIP (too few after quality)", "test": "Test 7: PS1 DR2 comparison"}

    # Load "before ubercal" magnitudes from Phase 5 catalog and join
    mag_before_col = f"mag_before_{band}"
    has_before = mag_before_col in ps1.columns
    if not has_before and cat_file.exists():
        try:
            cat_full = pd.read_parquet(cat_file, columns=["objectid", mag_before_col])
            ps1 = ps1.merge(cat_full, on="objectid", how="left")
            has_before = mag_before_col in ps1.columns
        except Exception:
            pass

    # Deredden both DELVE and PS1 magnitudes using SFD E(B-V)
    ebv = _get_sfd_ebv(ps1["ra"].values, ps1["dec"].values, cache_dir)
    R_decam = EXTINCTION_COEFFS[f"decam_{band}"]
    R_ps1 = EXTINCTION_COEFFS[f"ps1_{band}"]
    print(f"    Dereddening: R_DECam={R_decam:.3f}, R_PS1={R_ps1:.3f}, "
          f"E(B-V) median={np.median(ebv):.4f}", flush=True)

    mag_delve_dered = ps1[mag_col].values - R_decam * ebv
    mag_ps1_dered = ps1[ps1_mag_col].values - R_ps1 * ebv
    delta = (mag_delve_dered - mag_ps1_dered) * 1000  # mmag

    if has_before:
        mag_before_dered = ps1[mag_before_col].values - R_decam * ebv
        delta_before_ubercal = (mag_before_dered - mag_ps1_dered) * 1000

    # Color: use dereddened PS1 g-i
    if ps1_g_col in ps1.columns and ps1_i_col in ps1.columns:
        R_g_ps1 = EXTINCTION_COEFFS["ps1_g"]
        R_i_ps1 = EXTINCTION_COEFFS["ps1_i"]
        color = (ps1[ps1_g_col] - ps1[ps1_i_col]).values - (R_g_ps1 - R_i_ps1) * ebv
        color_label = "(g-i)$_{\\rm PS1,0}$"
    elif "ps1_gMeanPSFMag" in ps1.columns and "ps1_rMeanPSFMag" in ps1.columns:
        R_g_ps1 = EXTINCTION_COEFFS["ps1_g"]
        R_r_ps1 = EXTINCTION_COEFFS["ps1_r"]
        color = (ps1["ps1_gMeanPSFMag"] - ps1["ps1_rMeanPSFMag"]).values - (R_g_ps1 - R_r_ps1) * ebv
        color_label = "(g-r)$_{\\rm PS1,0}$"
    else:
        color = np.zeros(len(ps1))
        color_label = "color"

    # Fit color term on "after ubercal" data
    a, b, a_err, b_err, fit_mask = _fit_color_term(color, delta)
    resid_after = delta - (a + b * color)

    # Build common mask: stars valid in BOTH before and after (apples-to-apples)
    common_mask = fit_mask.copy()
    if has_before:
        common_mask &= np.isfinite(delta_before_ubercal)

    rms_before_ct = np.sqrt(np.nanmean(delta[common_mask]**2))
    rms_after_ct = np.sqrt(np.nanmean(resid_after[common_mask]**2))

    # Apply same color term to "before ubercal" data (same stars)
    rms_before_ubercal = None
    if has_before:
        resid_before_ubercal = delta_before_ubercal - (a + b * color)
        rms_before_ubercal = np.sqrt(np.nanmean(resid_before_ubercal[common_mask]**2))
        print(f"    Before ubercal RMS (vs PS1, after CT): {rms_before_ubercal:.1f} mmag",
              flush=True)

    print(f"    Stars in comparison: {common_mask.sum():,}", flush=True)
    print(f"    Color term: {a:.1f} + {b:.1f} * {color_label} mmag", flush=True)
    print(f"    RMS before color term: {rms_before_ct:.1f} mmag", flush=True)
    print(f"    RMS after color term: {rms_after_ct:.1f} mmag", flush=True)

    # Plot: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0): Histogram of residuals — before and after on same stars
    ax = axes[0, 0]
    if has_before:
        ax.hist(resid_before_ubercal[common_mask], bins=80, color="steelblue",
                edgecolor="none", alpha=0.5,
                label=f"Before ubercal: RMS = {rms_before_ubercal:.1f} mmag")
    ax.hist(resid_after[common_mask], bins=80, color="darkorange", edgecolor="none", alpha=0.7,
            label=f"After ubercal: RMS = {rms_after_ct:.1f} mmag")
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"$\\Delta${band} after CT (DELVE $-$ PS1) (mmag)")
    ax.set_ylabel("Stars")
    ax.set_title(f"PS1 DR2 Comparison — {band}-band ({common_mask.sum():,} stars)")
    ax.legend(fontsize=9)

    # (0,1): Delta mag vs color (color term)
    ax = axes[0, 1]
    ax.scatter(color[common_mask], delta[common_mask], s=1, alpha=0.2, color="steelblue")
    color_grid = np.linspace(np.nanpercentile(color[common_mask], 2),
                              np.nanpercentile(color[common_mask], 98), 100)
    ax.plot(color_grid, a + b * color_grid, "r-", lw=2,
            label=f"CT = {a:.1f} + {b:.1f} * color")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(color_label)
    ax.set_ylabel(f"$\\Delta${band} (mmag)")
    ax.set_title("Color Term Fit")
    ax.legend(fontsize=9)

    # (1,0): Sky map of residuals (after color term)
    ax = axes[1, 0]
    ra_plot = ps1.loc[common_mask, "ra"].values
    dec_plot = ps1.loc[common_mask, "dec"].values
    vmax = min(30, np.percentile(np.abs(resid_after[common_mask]), 95))
    sc = ax.scatter(ra_plot, dec_plot, c=resid_after[common_mask],
                     s=1, alpha=0.5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="mmag")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Spatial Residual (after CT)")
    ax.invert_xaxis()

    # (1,1): Residual vs magnitude
    ax = axes[1, 1]
    mag_plot = ps1.loc[common_mask, mag_col].values
    ax.scatter(mag_plot, resid_after[common_mask], s=1, alpha=0.2, color="steelblue")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel(f"mag$_{{\\rm ubercal,{band}}}$")
    ax.set_ylabel(f"$\\Delta${band} after CT (mmag)")
    ax.set_title("Magnitude Dependence")

    fig.suptitle(f"Test 7: DELVE vs PS1 DR2 (dereddened) — {band}-band (dec > $-$30)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_ps1_{band}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_ps1_{band}.png", bbox_inches="tight")
    plt.close(fig)

    if rms_before_ubercal is not None:
        passed = rms_after_ct <= rms_before_ubercal and rms_after_ct < 30.0
        status_str = (f"RMS={rms_after_ct:.1f} mmag after CT, "
                      f"before={rms_before_ubercal:.1f}")
    else:
        passed = rms_after_ct < 30.0
        status_str = f"RMS={rms_after_ct:.1f} mmag after CT"

    return {
        "test": "Test 7: PS1 DR2 comparison",
        "rms_before_ct_mmag": float(rms_before_ct),
        "rms_after_ct_mmag": float(rms_after_ct),
        "rms_before_ubercal_mmag": float(rms_before_ubercal) if rms_before_ubercal else None,
        "color_term_offset": float(a),
        "color_term_slope": float(b),
        "n_stars": int(fit_mask.sum()),
        "status": status_str,
        "passed": passed,
    }


def run_all_tests(band, pixels, config, output_dir, cache_dir):
    """Run all validation tests.

    Returns
    -------
    results : list of dict
    """
    fig_dir = output_dir / f"validation_{band}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(test0_fgcm_comparison(output_dir, cache_dir, band, fig_dir))
    results.append(test1_repeatability(output_dir, cache_dir, band, fig_dir))
    results.append(test2_dr2_comparison(output_dir, cache_dir, band, fig_dir))
    results.append(test7_ps1_comparison(output_dir, cache_dir, band, fig_dir))
    results.append(test4_gaia_comparison(output_dir, cache_dir, band, fig_dir))
    results.append(test5_stellar_locus(output_dir, cache_dir, band, fig_dir))
    results.append(test6_des_boundary(output_dir, cache_dir, band, fig_dir))

    # Print summary
    print(f"\n  {'=' * 60}", flush=True)
    print(f"  Validation Summary — {band}-band", flush=True)
    print(f"  {'=' * 60}", flush=True)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for r in results:
        test_name = r.get("test", "Unknown")
        status = r.get("status", "UNKNOWN")
        p = r.get("passed")
        if p is not None and bool(p):
            n_pass += 1
            icon = "PASS"
        elif p is not None and not bool(p):
            n_fail += 1
            icon = "FAIL"
        else:
            n_skip += 1
            icon = "SKIP"
        print(f"    [{icon:4s}] {test_name}: {status}", flush=True)

    print(f"\n    Passed: {n_pass}, Failed: {n_fail}, Skipped: {n_skip}", flush=True)
    print(f"  {'=' * 60}", flush=True)

    # Save summary
    summary_file = fig_dir / "validation_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"DELVE Ubercalibration Validation Summary — {band}-band\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(f"{r.get('test', 'Unknown')}\n")
            for k, v in r.items():
                if k != "test":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")
        f.write(f"\nPassed: {n_pass}, Failed: {n_fail}, Skipped: {n_skip}\n")

    print(f"  Summary saved: {summary_file}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Run all validation tests"
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

    run_all_tests(args.band, pixels, config, output_dir, cache_dir)


if __name__ == "__main__":
    main()
