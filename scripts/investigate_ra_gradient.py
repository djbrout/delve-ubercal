#!/usr/bin/env python3
"""
Deep investigation of the 65 mmag RA gradient in the unanchored ubercal solver.

Analyses:
0. Existing RA gradient characterization (split by exptime class)
1. Exposure time distribution vs RA
2. m_inst systematics: deep vs normal exposures for same stars
3. Seeing/FWHM vs RA
4. Aperture correction check: m_inst scatter vs FWHM
5. Survey program breakdown by RA
6. Overlap graph connectivity analysis (gray solve)
7. CRITICAL TEST: solver on normal-only data (exptime=90s)

Uses Phase 0 data, Phase 3 zeropoints, DES FGCM ZPs, and NSC exposure tables.
"""

import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import sparse
from scipy.sparse.linalg import cg

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Volumes/External5TB/DELVE_UBERCAL")
CACHE = BASE / "cache"
PHASE0_OUT = BASE / "output" / "phase0_g"
PHASE0_CACHE = CACHE / "phase0_g"
PHASE1_OUT = BASE / "output" / "phase1_g"
PHASE3_OUT = BASE / "output" / "phase3_g"
OUTDIR = BASE / "output" / "validation_g"
OUTDIR.mkdir(parents=True, exist_ok=True)

BAND = "g"

sys.path.insert(0, str(BASE))
from delve_ubercal.phase2_solve import load_des_fgcm_zps


# ── Helper: load all Phase 0 detections ────────────────────────────────────
def load_all_phase0():
    """Load all Phase 0 detection files and concatenate."""
    files = sorted(glob.glob(str(PHASE0_OUT / "detections_nside32_pixel*.parquet")))
    print(f"Loading {len(files)} Phase 0 detection files...")
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total detections: {len(df):,}")
    print(f"  Unique stars: {df['objectid'].nunique():,}")
    print(f"  Unique expnums: {df['expnum'].nunique():,}")
    return df


def load_phase0_with_fwhm():
    """Load Phase 0 cache files (which have FWHM) and merge via NSC exposure table.

    Phase 0 cache has 'exposure' string, not expnum. We join via NSC exposure table.
    """
    nsc_exp = pd.read_parquet(CACHE / f"nsc_exposure_{BAND}.parquet",
                              columns=["exposure", "expnum"])

    cache_files = sorted(glob.glob(str(PHASE0_CACHE / "raw_g_nside32_pixel*.parquet")))
    print(f"Loading {len(cache_files)} Phase 0 cache files for FWHM...")
    dfs = []
    for f in cache_files:
        df = pd.read_parquet(f, columns=["objectid", "exposure", "ccdnum", "fwhm"])
        dfs.append(df)
    if not dfs:
        print("  WARNING: No Phase 0 cache files found!")
        return None
    cache_df = pd.concat(dfs, ignore_index=True)
    cache_df = cache_df.merge(nsc_exp, on="exposure", how="inner")
    # Remove cross-pixel duplicates (stars near HEALPix boundaries appear in multiple cache files)
    cache_df = cache_df.drop_duplicates(subset=["objectid", "expnum", "ccdnum"], keep="first")
    print(f"  Cache detections with expnum (deduplicated): {len(cache_df):,}")
    return cache_df[["objectid", "expnum", "ccdnum", "fwhm"]]


def load_exptime():
    """Load DES exposure time table."""
    return pd.read_parquet(CACHE / "des_exptime.parquet")


def load_zeropoints():
    """Load Phase 3 unanchored zeropoints."""
    return pd.read_parquet(PHASE3_OUT / "zeropoints_unanchored.parquet")


def load_flagged_nodes():
    """Load flagged CCD-exposures from Phase 3."""
    return pd.read_parquet(PHASE3_OUT / "flagged_nodes.parquet")


def load_nsc_exposure():
    """Load NSC exposure table (exposure string -> expnum + instrument)."""
    return pd.read_parquet(CACHE / f"nsc_exposure_{BAND}.parquet")


def compute_exposure_ra_dec(det):
    """Compute median RA and Dec for each exposure from detections."""
    return det.groupby("expnum").agg(
        exp_ra=("ra", "median"),
        exp_dec=("dec", "median")
    )


# ============================================================================
# Analysis 0: Existing RA gradient characterization
# ============================================================================
def analysis_0_existing_gradient(det, zp_df, exptime_df, flagged_nodes):
    print("\n" + "=" * 72)
    print("ANALYSIS 0: Existing RA gradient in Phase 3 unanchored solution")
    print("=" * 72)

    exp_pos = compute_exposure_ra_dec(det)
    flagged_set = set(zip(flagged_nodes["expnum"].values, flagged_nodes["ccdnum"].values))

    # Merge ZPs with exposure position
    zp = zp_df.copy()
    zp = zp.merge(exp_pos, on="expnum", how="left")

    # Filter valid DES ZPs, exclude flagged
    zp["is_flagged"] = [
        (e, c) in flagged_set
        for e, c in zip(zp["expnum"].values, zp["ccdnum"].values)
    ]
    valid = zp[
        (zp["zp_fgcm"].notna()) & (zp["zp_fgcm"] > 25) & (zp["zp_fgcm"] < 35) &
        (~zp["is_flagged"]) &
        (zp["zp_solved"] > 25) & (zp["zp_solved"] < 35)  # exclude phantom nodes
    ].copy()

    diff = valid["zp_solved"] - valid["zp_fgcm"]
    print(f"  Valid DES CCD-exposures (unflagged, non-phantom): {len(valid):,}")
    print(f"  RMS(ZP_solved - FGCM): {diff.std()*1000:.1f} mmag")

    # Merge with exptime
    valid = valid.merge(exptime_df, on="expnum", how="left")
    valid["exptime"] = valid["exptime"].fillna(-1)
    valid["diff_mmag"] = (valid["zp_solved"] - valid["zp_fgcm"]) * 1000

    # RA gradient by exptime class
    ra_bins = np.arange(49, 62, 1.0)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])

    def binned_median(sub, col="diff_mmag", ra_col="exp_ra"):
        result = []
        for i in range(len(ra_bins) - 1):
            mask = (sub[ra_col] >= ra_bins[i]) & (sub[ra_col] < ra_bins[i + 1])
            result.append(sub.loc[mask, col].median() if mask.sum() > 10 else np.nan)
        return np.array(result)

    grad_all = binned_median(valid)
    grad_90 = binned_median(valid[valid["exptime"] == 90.0])
    grad_deep = binned_median(valid[valid["exptime"] > 100.0])

    print(f"\n  RA gradient (median ZP - FGCM per bin, mmag):")
    print(f"  {'RA':>6s}  {'All DES':>10s}  {'DES 90s':>10s}  {'DES deep':>10s}")
    for i in range(len(ra_centers)):
        d_str = f"{grad_deep[i]:10.1f}" if not np.isnan(grad_deep[i]) else "       N/A"
        print(f"  {ra_centers[i]:6.1f}  {grad_all[i]:10.1f}  {grad_90[i]:10.1f}  {d_str}")

    for label, arr in [("all DES", grad_all), ("DES 90s", grad_90)]:
        rng = np.nanmax(arr) - np.nanmin(arr)
        print(f"  RA gradient range ({label}): {rng:.1f} mmag")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 0a: RA gradient
    ax = axes[0, 0]
    ax.plot(ra_centers, grad_all, "ko-", lw=2, ms=6, label="All DES")
    ax.plot(ra_centers, grad_90, "bs-", lw=2, ms=6, label="DES 90s only")
    mask_deep = ~np.isnan(grad_deep)
    if mask_deep.any():
        ax.plot(ra_centers[mask_deep], grad_deep[mask_deep], "r^-", lw=2, ms=6, label="DES deep only")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Median (ZP_solved - ZP_FGCM) [mmag]")
    ax.set_title("RA gradient in existing Phase 3 solution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 0b: 2D map
    ax = axes[0, 1]
    sc = ax.scatter(valid["exp_ra"], valid["exp_dec"],
                    c=valid["diff_mmag"], s=1, alpha=0.3,
                    cmap="RdBu_r", vmin=-100, vmax=100)
    plt.colorbar(sc, ax=ax, label="ZP - FGCM [mmag]")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("ZP residual map (RA, Dec)")
    ax.invert_xaxis()

    # 0c: Histograms by exptime class
    ax = axes[1, 0]
    mask_90 = valid["exptime"] == 90.0
    mask_d = valid["exptime"] > 100.0
    ax.hist(valid.loc[mask_90, "diff_mmag"], bins=100, range=(-200, 200),
            alpha=0.5, color="blue", label=f"DES 90s (N={mask_90.sum():,})")
    if mask_d.sum() > 0:
        ax.hist(valid.loc[mask_d, "diff_mmag"], bins=100, range=(-200, 200),
                alpha=0.5, color="red", label=f"DES deep (N={mask_d.sum():,})")
    ax.set_xlabel("ZP_solved - ZP_FGCM [mmag]")
    ax.set_ylabel("N")
    ax.set_title("ZP residual by exposure type")
    ax.legend(fontsize=8)

    # 0d: ZP residual vs exptime
    ax = axes[1, 1]
    des_only = valid[valid["exptime"] > 0]
    ax.scatter(des_only["exptime"], des_only["diff_mmag"], s=1, alpha=0.1, c="steelblue")
    et_bins = [30, 50, 80, 95, 120, 160, 190, 250, 350, 500]
    et_centers = [0.5 * (et_bins[i] + et_bins[i + 1]) for i in range(len(et_bins) - 1)]
    et_medians = []
    for i in range(len(et_bins) - 1):
        mask = (des_only["exptime"] >= et_bins[i]) & (des_only["exptime"] < et_bins[i + 1])
        et_medians.append(des_only.loc[mask, "diff_mmag"].median() if mask.sum() > 10 else np.nan)
    ax.plot(et_centers, et_medians, "ro-", lw=2, ms=6, label="Binned median")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel("ZP_solved - ZP_FGCM [mmag]")
    ax.set_title("ZP residual vs exposure time")
    ax.legend()

    plt.suptitle("Analysis 0: Existing RA gradient in Phase 3 unanchored solution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_0_existing_gradient.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_0_existing_gradient.png")


# ============================================================================
# Analysis 1: Exposure time distribution vs RA
# ============================================================================
def analysis_1_exptime_vs_ra(det, exptime_df):
    print("\n" + "=" * 72)
    print("ANALYSIS 1: Exposure time distribution vs RA")
    print("=" * 72)

    exp_pos = compute_exposure_ra_dec(det)
    merged = exptime_df.merge(exp_pos, on="expnum", how="inner")
    print(f"  DES exposures with RA: {len(merged):,}")

    unique_et = sorted(merged["exptime"].unique())
    print(f"  Unique exposure times: {len(unique_et)}")
    print(f"  Top exptimes: {merged['exptime'].value_counts().head(10).to_dict()}")

    merged["exp_class"] = "other"
    merged.loc[merged["exptime"] == 90.0, "exp_class"] = "normal_90s"
    merged.loc[merged["exptime"] > 100.0, "exp_class"] = "deep_>100s"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(merged["exptime"], bins=50, edgecolor="k", alpha=0.7)
    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel("Number of exposures")
    ax.set_title("Exposure time distribution (DES)")
    ax.set_yscale("log")

    ax = axes[0, 1]
    for cls, color, marker in [("normal_90s", "blue", "."), ("deep_>100s", "red", "x"), ("other", "green", "^")]:
        sub = merged[merged["exp_class"] == cls]
        if len(sub) > 0:
            ax.scatter(sub["exp_ra"], sub["exptime"], c=color, s=3, alpha=0.3, label=f"{cls} ({len(sub)})")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Exposure time (s)")
    ax.set_title("Exposure time vs RA")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ra_bins = np.arange(49, 62, 0.5)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])
    n_total = np.histogram(merged["exp_ra"], bins=ra_bins)[0]
    n_deep = np.histogram(merged.loc[merged["exp_class"] == "deep_>100s", "exp_ra"], bins=ra_bins)[0]
    frac = np.where(n_total > 0, n_deep / n_total, 0)
    ax.bar(ra_centers, frac, width=0.45, color="tomato", edgecolor="k", alpha=0.7)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Fraction deep (exptime > 100s)")
    ax.set_title("Deep field exposure fraction vs RA")

    ax = axes[1, 1]
    for cls, color in [("normal_90s", "blue"), ("deep_>100s", "red"), ("other", "green")]:
        sub = merged[merged["exp_class"] == cls]
        n_cls = np.histogram(sub["exp_ra"], bins=ra_bins)[0]
        ax.step(ra_centers, n_cls, where="mid", color=color, label=cls, lw=2)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Number of DES exposures")
    ax.set_title("Exposure count by type vs RA")
    ax.legend(fontsize=8)

    plt.suptitle("Analysis 1: Exposure time distribution vs RA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_1_exptime_vs_ra.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_1_exptime_vs_ra.png")


# ============================================================================
# Analysis 2: m_inst systematics - deep vs normal for same stars
# ============================================================================
def analysis_2_minst_systematics(det, exptime_df):
    print("\n" + "=" * 72)
    print("ANALYSIS 2: m_inst systematics (deep vs normal for same stars)")
    print("=" * 72)

    det_et = det.merge(exptime_df, on="expnum", how="inner")
    print(f"  Detections with exptime info (DES only): {len(det_et):,}")

    det_et["is_deep"] = det_et["exptime"] > 100.0
    det_et["is_normal"] = det_et["exptime"] == 90.0

    deep_mean = det_et[det_et["is_deep"]].groupby("objectid")["m_inst"].agg(
        ["mean", "count"]).rename(columns={"mean": "m_deep", "count": "n_deep"})
    norm_mean = det_et[det_et["is_normal"]].groupby("objectid")["m_inst"].agg(
        ["mean", "count"]).rename(columns={"mean": "m_normal", "count": "n_normal"})

    both = deep_mean.join(norm_mean, how="inner")
    print(f"  Stars with both deep and normal detections: {len(both):,}")

    if len(both) == 0:
        print("  WARNING: No stars observed on both deep and normal exposures!")
        return

    both["dm"] = both["m_deep"] - both["m_normal"]
    print(f"  Mean m_inst(deep) - m_inst(normal): {both['dm'].mean()*1000:.1f} mmag")
    print(f"  Median: {both['dm'].median()*1000:.1f} mmag")
    print(f"  Std: {both['dm'].std()*1000:.1f} mmag")

    well = both[(both["n_deep"] >= 2) & (both["n_normal"] >= 2)]
    print(f"  Stars with >= 2 deep AND >= 2 normal: {len(well):,}")
    if len(well) > 0:
        print(f"    Median dm: {well['dm'].median()*1000:.1f} mmag, Std: {well['dm'].std()*1000:.1f} mmag")

    # Get star positions and magnitudes
    star_mag = det.groupby("objectid")["m_inst"].mean()
    star_ra = det.groupby("objectid")["ra"].first()
    both = both.join(star_mag.rename("mag_mean"))
    both = both.join(star_ra.rename("star_ra"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2a: Histogram of dm
    ax = axes[0, 0]
    dm_mmag = both["dm"] * 1000
    ax.hist(dm_mmag, bins=100, range=(-200, 200), edgecolor="k", alpha=0.7, color="steelblue")
    ax.axvline(both["dm"].median() * 1000, color="red", ls="--", lw=2,
               label=f"Median: {both['dm'].median()*1000:.1f} mmag")
    ax.set_xlabel("m_inst(deep) - m_inst(normal) [mmag]")
    ax.set_ylabel("Number of stars")
    ax.set_title(f"m_inst offset (deep - normal), N={len(both):,}")
    ax.legend()

    # 2b: dm vs magnitude
    ax = axes[0, 1]
    ax.scatter(both["mag_mean"], dm_mmag, s=1, alpha=0.1, c="steelblue")
    mag_bins = np.arange(14, 22, 0.5)
    mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    binned = []
    for i in range(len(mag_bins) - 1):
        mask = (both["mag_mean"] >= mag_bins[i]) & (both["mag_mean"] < mag_bins[i + 1])
        binned.append(both.loc[mask, "dm"].median() * 1000 if mask.sum() > 10 else np.nan)
    ax.plot(mag_centers, binned, "ro-", lw=2, ms=5, label="Binned median")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("Mean m_inst (mag)")
    ax.set_ylabel("m_inst(deep) - m_inst(normal) [mmag]")
    ax.set_title("Offset vs brightness")
    ax.set_ylim(-100, 100)
    ax.legend()

    # 2c: dm vs RA
    ax = axes[1, 0]
    well_plot = both[(both["n_deep"] >= 2) & (both["n_normal"] >= 2)]
    ax.scatter(well_plot["star_ra"], well_plot["dm"] * 1000, s=1, alpha=0.1, c="steelblue")
    ra_bins = np.arange(49, 62, 0.5)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])
    binned_ra = []
    for i in range(len(ra_bins) - 1):
        mask = (well_plot["star_ra"] >= ra_bins[i]) & (well_plot["star_ra"] < ra_bins[i + 1])
        binned_ra.append(well_plot.loc[mask, "dm"].median() * 1000 if mask.sum() > 10 else np.nan)
    ax.plot(ra_centers, binned_ra, "ro-", lw=2, ms=5, label="Binned median")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("m_inst(deep) - m_inst(normal) [mmag]")
    ax.set_title("Offset vs RA (stars with >= 2 each)")
    ax.set_ylim(-100, 100)
    ax.legend()

    # 2d: Distribution of detection counts
    ax = axes[1, 1]
    ax.hist(both["n_deep"], bins=range(0, 26), alpha=0.5, label="Deep detections", color="red")
    ax.hist(both["n_normal"], bins=range(0, 26), alpha=0.5, label="Normal detections", color="blue")
    ax.set_xlabel("Number of detections per star")
    ax.set_ylabel("Number of stars")
    ax.set_title("Detection count distribution")
    ax.legend()

    plt.suptitle("Analysis 2: m_inst systematics (deep vs normal for same stars)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_2_minst_systematics.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_2_minst_systematics.png")


# ============================================================================
# Analysis 3: Seeing/FWHM vs RA
# ============================================================================
def analysis_3_fwhm_vs_ra(det, fwhm_df, exptime_df):
    print("\n" + "=" * 72)
    print("ANALYSIS 3: Seeing/FWHM vs RA")
    print("=" * 72)

    if fwhm_df is None:
        print("  SKIPPED: No FWHM data available")
        return

    # Merge FWHM into detections
    det_fwhm = det.merge(fwhm_df[["objectid", "expnum", "ccdnum", "fwhm"]],
                         on=["objectid", "expnum", "ccdnum"], how="inner")
    print(f"  Detections with FWHM: {len(det_fwhm):,} / {len(det):,}")

    # Per-exposure FWHM
    exp_fwhm = det_fwhm.groupby("expnum").agg(
        fwhm_median=("fwhm", "median"),
        ra_median=("ra", "median"),
        n_det=("fwhm", "count")
    ).reset_index()

    exp_fwhm = exp_fwhm.merge(exptime_df, on="expnum", how="left")
    exp_fwhm["exptime"] = exp_fwhm["exptime"].fillna(-1)

    exp_fwhm["exp_class"] = "non-DES"
    exp_fwhm.loc[exp_fwhm["exptime"] == 90.0, "exp_class"] = "DES_90s"
    exp_fwhm.loc[exp_fwhm["exptime"] > 100.0, "exp_class"] = "DES_deep"
    exp_fwhm.loc[(exp_fwhm["exptime"] > 0) & (exp_fwhm["exptime"] != 90.0) &
                 (exp_fwhm["exptime"] <= 100.0), "exp_class"] = "DES_other"

    for cls in ["DES_90s", "DES_deep", "non-DES"]:
        sub = exp_fwhm[exp_fwhm["exp_class"] == cls]
        if len(sub) > 0:
            print(f"  FWHM ({cls}): median={sub['fwhm_median'].median():.2f}\", "
                  f"mean={sub['fwhm_median'].mean():.2f}\", N={len(sub)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for cls, color in [("DES_90s", "blue"), ("DES_deep", "red"), ("non-DES", "green")]:
        sub = exp_fwhm[exp_fwhm["exp_class"] == cls]
        if len(sub) > 0:
            ax.scatter(sub["ra_median"], sub["fwhm_median"], c=color, s=3, alpha=0.3, label=cls)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("Per-exposure seeing vs RA")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ra_bins = np.arange(49, 62, 0.5)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])
    for cls, color in [("DES_90s", "blue"), ("DES_deep", "red"), ("non-DES", "green")]:
        sub = exp_fwhm[exp_fwhm["exp_class"] == cls]
        binned = []
        for i in range(len(ra_bins) - 1):
            mask = (sub["ra_median"] >= ra_bins[i]) & (sub["ra_median"] < ra_bins[i + 1])
            binned.append(sub.loc[mask, "fwhm_median"].median() if mask.sum() > 5 else np.nan)
        ax.plot(ra_centers, binned, "o-", color=color, lw=2, ms=5, label=cls)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Median FWHM (arcsec)")
    ax.set_title("Binned seeing vs RA")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for cls, color in [("DES_90s", "blue"), ("DES_deep", "red"), ("non-DES", "green")]:
        sub = exp_fwhm[exp_fwhm["exp_class"] == cls]
        if len(sub) > 0:
            ax.hist(sub["fwhm_median"], bins=50, range=(0.5, 3.0), alpha=0.5,
                    color=color, label=f"{cls} (N={len(sub)})", density=True)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("Density")
    ax.set_title("Seeing distribution by exposure class")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    des_only = exp_fwhm[exp_fwhm["exptime"] > 0]
    ax.scatter(des_only["exptime"], des_only["fwhm_median"], s=3, alpha=0.3, c="steelblue")
    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("Seeing vs exposure time (DES only)")

    plt.suptitle("Analysis 3: Seeing/FWHM vs RA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_3_fwhm_vs_ra.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_3_fwhm_vs_ra.png")


# ============================================================================
# Analysis 4: Aperture correction check - m_inst scatter vs FWHM
# ============================================================================
def analysis_4_aperture_correction(det, fwhm_df, exptime_df):
    print("\n" + "=" * 72)
    print("ANALYSIS 4: Aperture correction check")
    print("=" * 72)

    if fwhm_df is None:
        print("  SKIPPED: No FWHM data available")
        return

    det_fwhm = det.merge(fwhm_df[["objectid", "expnum", "ccdnum", "fwhm"]],
                         on=["objectid", "expnum", "ccdnum"], how="inner")
    det_fwhm = det_fwhm.merge(exptime_df, on="expnum", how="left")
    det_fwhm["exptime"] = det_fwhm["exptime"].fillna(-1)

    # Compute star mean and residual
    star_mean = det_fwhm.groupby("objectid")["m_inst"].mean().rename("m_mean")
    det_fwhm = det_fwhm.merge(star_mean, on="objectid")
    det_fwhm["resid"] = det_fwhm["m_inst"] - det_fwhm["m_mean"]

    # Keep stars with >= 3 detections
    star_ndet = det_fwhm.groupby("objectid")["m_inst"].count()
    multi_stars = star_ndet[star_ndet >= 3].index
    det_multi = det_fwhm[det_fwhm["objectid"].isin(multi_stars)].copy()
    print(f"  Detections for stars with >= 3 obs: {len(det_multi):,}")

    fwhm_bins = np.arange(0.5, 3.5, 0.1)
    fwhm_centers = 0.5 * (fwhm_bins[:-1] + fwhm_bins[1:])
    resid_median = []
    resid_std = []
    counts = []
    for i in range(len(fwhm_bins) - 1):
        mask = (det_multi["fwhm"] >= fwhm_bins[i]) & (det_multi["fwhm"] < fwhm_bins[i + 1])
        if mask.sum() > 50:
            resid_median.append(det_multi.loc[mask, "resid"].median() * 1000)
            resid_std.append(det_multi.loc[mask, "resid"].std() * 1000)
            counts.append(mask.sum())
        else:
            resid_median.append(np.nan)
            resid_std.append(np.nan)
            counts.append(0)
    resid_median = np.array(resid_median)
    resid_std = np.array(resid_std)

    print(f"  Residual vs FWHM trend (selected bins):")
    for i in range(0, len(fwhm_centers), 5):
        if not np.isnan(resid_median[i]):
            print(f"    FWHM={fwhm_centers[i]:.1f}\": median resid={resid_median[i]:.1f} mmag, "
                  f"std={resid_std[i]:.1f} mmag, N={counts[i]}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(fwhm_centers, resid_median, "bo-", lw=2, ms=4)
    ax.fill_between(fwhm_centers,
                    resid_median - resid_std / np.sqrt(np.maximum(np.array(counts, dtype=float), 1)),
                    resid_median + resid_std / np.sqrt(np.maximum(np.array(counts, dtype=float), 1)),
                    alpha=0.3, color="blue")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("Median m_inst residual [mmag]")
    ax.set_title("Aperture correction: residual vs seeing")

    ax = axes[0, 1]
    ax.plot(fwhm_centers, resid_std, "ro-", lw=2, ms=4)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("RMS of m_inst residual [mmag]")
    ax.set_title("Photometric scatter vs seeing")

    ax = axes[1, 0]
    for label, mask_cond, color in [
        ("DES 90s", det_multi["exptime"] == 90.0, "blue"),
        ("DES deep", det_multi["exptime"] > 100.0, "red"),
        ("non-DES", det_multi["exptime"] < 0, "green"),
    ]:
        sub = det_multi[mask_cond]
        if len(sub) > 100:
            binned = []
            for i in range(len(fwhm_bins) - 1):
                fmask = (sub["fwhm"] >= fwhm_bins[i]) & (sub["fwhm"] < fwhm_bins[i + 1])
                binned.append(sub.loc[fmask, "resid"].median() * 1000 if fmask.sum() > 20 else np.nan)
            ax.plot(fwhm_centers, binned, "o-", color=color, lw=2, ms=4, label=label)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("Median m_inst residual [mmag]")
    ax.set_title("Aperture correction by exposure class")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    valid = det_multi[(det_multi["fwhm"] > 0.5) & (det_multi["fwhm"] < 3.0) &
                       (np.abs(det_multi["resid"]) < 0.2)]
    h = ax.hist2d(valid["fwhm"], valid["resid"] * 1000, bins=[50, 50],
              range=[[0.5, 3.0], [-100, 100]], cmap="viridis")
    ax.axhline(0, color="r", ls="--", lw=1)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("m_inst residual [mmag]")
    ax.set_title("2D histogram: seeing vs residual")
    plt.colorbar(h[3], ax=ax, label="N")

    plt.suptitle("Analysis 4: Aperture correction check (m_inst scatter vs FWHM)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_4_aperture_correction.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_4_aperture_correction.png")


# ============================================================================
# Analysis 5: Survey program breakdown by RA
# ============================================================================
def analysis_5_program_breakdown(det, exptime_df, nsc_exp):
    print("\n" + "=" * 72)
    print("ANALYSIS 5: Survey program breakdown by RA")
    print("=" * 72)

    exp_pos = compute_exposure_ra_dec(det)
    exp_info = exp_pos.reset_index()
    exp_info = exp_info.merge(nsc_exp[["expnum", "instrument"]], on="expnum", how="left")
    exp_info["instrument"] = exp_info["instrument"].fillna("unknown")
    exp_info = exp_info.merge(exptime_df, on="expnum", how="left")

    # Classify by program
    exp_info["program"] = "unknown"
    is_c4d = exp_info["instrument"].str.startswith("c4d")
    has_et = exp_info["exptime"].notna()
    exp_info.loc[is_c4d & has_et & (exp_info["exptime"] == 90.0), "program"] = "DES_survey_90s"
    exp_info.loc[is_c4d & has_et & (exp_info["exptime"] == 45.0), "program"] = "DES_45s"
    exp_info.loc[is_c4d & has_et & (exp_info["exptime"].isin([175.0, 200.0])), "program"] = "DES_SN_deep"
    exp_info.loc[is_c4d & has_et & (exp_info["exptime"].isin([330.0, 360.0, 400.0])), "program"] = "DES_long"
    still_des = is_c4d & has_et & (~exp_info["program"].str.startswith("DES"))
    exp_info.loc[still_des, "program"] = "DES_other"

    non_c4d = ~is_c4d & (exp_info["instrument"] != "unknown")
    exp_info.loc[non_c4d, "program"] = "non-DES"

    c4d_no_et = is_c4d & ~has_et
    exp_info.loc[c4d_no_et, "program"] = "DECam_no_DES_ZP"

    print("  Program breakdown (exposures):")
    print(exp_info["program"].value_counts().to_string())

    # Count detections per program
    det_prog = det.merge(exp_info[["expnum", "program"]], on="expnum", how="left")
    det_prog["program"] = det_prog["program"].fillna("unknown")
    print("\n  Detections per program:")
    print(det_prog["program"].value_counts().to_string())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ra_bins = np.arange(49, 62, 0.5)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])
    programs = ["DES_survey_90s", "DES_SN_deep", "DES_45s", "DES_long", "DES_other",
                "DECam_no_DES_ZP", "non-DES", "unknown"]
    colors_prog = ["royalblue", "red", "orange", "purple", "gray", "cyan", "green", "pink"]

    # 5a: Stacked histogram
    ax = axes[0, 0]
    bottom = np.zeros(len(ra_centers))
    for prog, col in zip(programs, colors_prog):
        sub = exp_info[exp_info["program"] == prog]
        if len(sub) > 0:
            n_prog = np.histogram(sub["exp_ra"], bins=ra_bins)[0]
            ax.bar(ra_centers, n_prog, width=0.45, bottom=bottom, color=col,
                   edgecolor="k", linewidth=0.3, label=prog, alpha=0.8)
            bottom += n_prog
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Number of exposures")
    ax.set_title("Exposure count by program vs RA")
    ax.legend(fontsize=6, loc="upper right")

    # 5b: Detection density by program
    ax = axes[0, 1]
    for prog, col in zip(["DES_survey_90s", "DES_SN_deep", "non-DES", "DECam_no_DES_ZP"],
                          ["blue", "red", "green", "cyan"]):
        sub = det_prog[det_prog["program"] == prog]
        n_det = np.histogram(sub["ra"], bins=ra_bins)[0]
        ax.step(ra_centers, n_det, where="mid", color=col, lw=2, label=prog)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Number of detections")
    ax.set_title("Detection count by program vs RA")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # 5c: Fraction of each program
    ax = axes[1, 0]
    n_total = np.histogram(exp_info["exp_ra"], bins=ra_bins)[0].astype(float)
    n_total[n_total == 0] = 1
    for prog, col in zip(programs[:5], colors_prog[:5]):
        sub = exp_info[exp_info["program"] == prog]
        if len(sub) > 0:
            n_prog = np.histogram(sub["exp_ra"], bins=ra_bins)[0]
            ax.plot(ra_centers, n_prog / n_total, "o-", color=col, lw=2, ms=4, label=prog)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Fraction of exposures")
    ax.set_title("Program fraction vs RA")
    ax.legend(fontsize=7)

    # 5d: Text summary
    ax = axes[1, 1]
    ax.axis("off")
    text = "Program Summary\n" + "=" * 40 + "\n"
    for prog in programs:
        sub = exp_info[exp_info["program"] == prog]
        n_exp = len(sub)
        n_det_p = len(det_prog[det_prog["program"] == prog])
        if n_exp > 0:
            text += f"{prog:>20s}: {n_exp:>5d} exp, {n_det_p:>8,d} det\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace")

    plt.suptitle("Analysis 5: Survey program breakdown by RA", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_5_program_breakdown.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_5_program_breakdown.png")


# ============================================================================
# Analysis 6: Overlap graph connectivity (gray solve)
# ============================================================================
def analysis_6_graph_connectivity(det, exptime_df):
    print("\n" + "=" * 72)
    print("ANALYSIS 6: Overlap graph analysis (gray solve connectivity)")
    print("=" * 72)

    exp_pos = compute_exposure_ra_dec(det)
    star_files = sorted(glob.glob(str(PHASE1_OUT / "star_lists_nside32_pixel*.parquet")))
    print(f"  Loading {len(star_files)} star list files for overlap analysis...")

    # Build exposure-to-exposure overlaps
    exp_pairs = defaultdict(int)
    exp_nstars = defaultdict(int)

    for si, sf in enumerate(star_files):
        sl = pd.read_parquet(sf, columns=["objectid", "expnum"])
        # Group by star, get unique exposures
        for star_id, group in sl.groupby("objectid"):
            exps = sorted(group["expnum"].unique())
            for e in exps:
                exp_nstars[e] += 1
            for i in range(len(exps)):
                for j in range(i + 1, len(exps)):
                    exp_pairs[(exps[i], exps[j])] += 1
        if (si + 1) % 10 == 0:
            print(f"    Processed {si+1}/{len(star_files)} files")

    print(f"  Unique exposure pairs: {len(exp_pairs):,}")
    print(f"  Total shared-star links: {sum(exp_pairs.values()):,}")

    pair_df = pd.DataFrame([
        {"exp1": k[0], "exp2": k[1], "n_shared": v}
        for k, v in exp_pairs.items()
    ])

    pair_df = pair_df.merge(exp_pos["exp_ra"].rename("ra1"), left_on="exp1", right_index=True, how="left")
    pair_df = pair_df.merge(exp_pos["exp_ra"].rename("ra2"), left_on="exp2", right_index=True, how="left")
    pair_df["dra"] = np.abs(pair_df["ra1"] - pair_df["ra2"])

    print(f"\n  RA separation stats for exposure pairs:")
    print(f"    Median dRA: {pair_df['dra'].median():.2f} deg")
    print(f"    Mean dRA: {pair_df['dra'].mean():.2f} deg")
    print(f"    Max dRA: {pair_df['dra'].max():.2f} deg")
    for thresh in [2, 5, 8]:
        n = (pair_df["dra"] > thresh).sum()
        print(f"    Pairs with dRA > {thresh} deg: {n:,} / {len(pair_df):,} ({100*n/len(pair_df):.1f}%)")

    # Per-exposure connectivity
    exp_conn = defaultdict(int)
    for (e1, e2) in exp_pairs:
        exp_conn[e1] += 1
        exp_conn[e2] += 1
    conn_df = pd.DataFrame([
        {"expnum": k, "n_overlaps": v}
        for k, v in exp_conn.items()
    ])
    conn_df = conn_df.merge(exp_pos, on="expnum", how="left")
    conn_df = conn_df.merge(exptime_df, on="expnum", how="left")
    conn_df["exptime"] = conn_df["exptime"].fillna(-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 6a: dRA histogram
    ax = axes[0, 0]
    ax.hist(pair_df["dra"], bins=100, edgecolor="k", alpha=0.7, color="steelblue")
    ax.set_xlabel("RA separation (deg)")
    ax.set_ylabel("Number of exposure pairs")
    ax.set_title(f"RA separation of overlapping exposures (N={len(pair_df):,})")
    ax.set_yscale("log")
    ax.axvline(2, color="red", ls="--", label="2 deg")
    ax.axvline(5, color="orange", ls="--", label="5 deg")
    ax.legend()

    # 6b: Connectivity vs RA
    ax = axes[0, 1]
    for label, mask, color in [
        ("DES 90s", conn_df["exptime"] == 90.0, "blue"),
        ("DES deep", conn_df["exptime"] > 100.0, "red"),
        ("non-DES", conn_df["exptime"] < 0, "green"),
    ]:
        sub = conn_df[mask]
        if len(sub) > 0:
            ax.scatter(sub["exp_ra"], sub["n_overlaps"], s=3, alpha=0.3, c=color, label=label)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Number of overlapping exposures")
    ax.set_title("Per-exposure connectivity vs RA")
    ax.legend(fontsize=8)

    # 6c: Long-range links
    ax = axes[1, 0]
    long_range = pair_df[pair_df["dra"] > 2]
    if len(long_range) > 0:
        ax.scatter(long_range["ra1"], long_range["ra2"], s=2, alpha=0.2, c="steelblue")
        ax.plot([49, 62], [49, 62], "k--", lw=0.5)
        ax.set_xlabel("RA of exposure 1 (deg)")
        ax.set_ylabel("RA of exposure 2 (deg)")
        ax.set_title(f"Long-range links (dRA > 2 deg, N={len(long_range):,})")
    else:
        ax.text(0.5, 0.5, "No long-range links", transform=ax.transAxes, ha="center")
        ax.set_title("No long-range links found")

    # 6d: Shared stars vs dRA
    ax = axes[1, 1]
    dra_bins = np.arange(0, 12, 0.5)
    dra_centers = 0.5 * (dra_bins[:-1] + dra_bins[1:])
    binned = []
    for i in range(len(dra_bins) - 1):
        mask = (pair_df["dra"] >= dra_bins[i]) & (pair_df["dra"] < dra_bins[i + 1])
        binned.append(pair_df.loc[mask, "n_shared"].median() if mask.sum() > 0 else np.nan)
    ax.plot(dra_centers, binned, "bo-", lw=2, ms=4)
    ax.set_xlabel("RA separation (deg)")
    ax.set_ylabel("Median shared stars per pair")
    ax.set_title("Overlap strength vs RA separation")
    ax.set_yscale("log")

    plt.suptitle("Analysis 6: Overlap graph connectivity (gray solve)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_6_graph_connectivity.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_6_graph_connectivity.png")

    return pair_df


# ============================================================================
# Analysis 7: CRITICAL TEST - solver on normal-only data (exclude deep fields)
#
# Uses two-stage star flat + gray solve (same architecture as Phase 3)
# to get a fair comparison with the full solve.
# ============================================================================
def analysis_7_normal_only_solve(det, exptime_df, zp_df, flagged_nodes):
    print("\n" + "=" * 72)
    print("ANALYSIS 7: CRITICAL TEST - solver on normal DES (90s) + non-DES only")
    print("           Using two-stage star flat + gray solve architecture")
    print("=" * 72)

    flagged_set = set(zip(flagged_nodes["expnum"].values, flagged_nodes["ccdnum"].values))
    deep_expnums = set(exptime_df.loc[exptime_df["exptime"] > 100.0, "expnum"].values)
    print(f"  Deep field expnums to exclude (exptime > 100s): {len(deep_expnums):,}")

    # Load star lists, exclude deep fields and flagged nodes
    star_files = sorted(glob.glob(str(PHASE1_OUT / "star_lists_nside32_pixel*.parquet")))
    all_dets = []

    print(f"  Loading {len(star_files)} star list files, excluding deep fields...")
    for sf in star_files:
        sl = pd.read_parquet(sf)
        sl = sl[~sl["expnum"].isin(deep_expnums)].copy()
        node_keys = list(zip(sl["expnum"].values, sl["ccdnum"].values))
        keep = [nk not in flagged_set for nk in node_keys]
        sl = sl[keep]
        all_dets.append(sl)

    det_filtered = pd.concat(all_dets, ignore_index=True)
    star_counts = det_filtered.groupby("objectid").size()
    good_stars = star_counts[star_counts >= 2].index
    det_filtered = det_filtered[det_filtered["objectid"].isin(good_stars)]

    all_nodes = set(zip(det_filtered["expnum"].values, det_filtered["ccdnum"].values))
    nodes = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = list(nodes)
    n_params = len(nodes)
    print(f"  After filtering: {len(det_filtered):,} det, {det_filtered['objectid'].nunique():,} stars, "
          f"{n_params:,} CCD-exposures")

    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Initial per-CCD-exposure solve (to extract star flat)
    # ──────────────────────────────────────────────────────────────────
    print("\n  --- Stage 1: Initial per-CCD-exposure solve ---")
    AtWA_diag = np.zeros(n_params)
    AtWA_off = defaultdict(float)
    rhs = np.zeros(n_params)
    n_pairs = 0

    for star_id, group in det_filtered.groupby("objectid"):
        n_det = len(group)
        if n_det < 2:
            continue
        expnums = group["expnum"].values
        ccdnums = group["ccdnum"].values
        m_insts = group["m_inst"].values
        m_errs = group["m_err"].values

        for i in range(n_det):
            for j in range(i + 1, n_det):
                key_i = (expnums[i], ccdnums[i])
                key_j = (expnums[j], ccdnums[j])
                idx_i = node_to_idx.get(key_i)
                idx_j = node_to_idx.get(key_j)
                if idx_i is None or idx_j is None:
                    continue
                w = 1.0 / (m_errs[i] ** 2 + m_errs[j] ** 2)
                dm = m_insts[i] - m_insts[j]
                AtWA_diag[idx_i] += w
                AtWA_diag[idx_j] += w
                pair_key = (min(idx_i, idx_j), max(idx_i, idx_j))
                AtWA_off[pair_key] -= w
                rhs[idx_i] += w * dm
                rhs[idx_j] -= w * dm
                n_pairs += 1

    print(f"  Overlap pairs: {n_pairs:,}")

    rows_s, cols_s, vals_s = [], [], []
    for i in range(n_params):
        rows_s.append(i); cols_s.append(i); vals_s.append(AtWA_diag[i])
    for (i, j), v in AtWA_off.items():
        rows_s.append(i); cols_s.append(j); vals_s.append(v)
        rows_s.append(j); cols_s.append(i); vals_s.append(v)
    AtWA = sparse.csr_matrix((vals_s, (rows_s, cols_s)), shape=(n_params, n_params))
    AtWA = AtWA + 1e-6 * sparse.eye(n_params)

    print("  Solving CG (stage 1)...")
    x_initial, info1 = cg(AtWA, rhs, rtol=1e-5, maxiter=5000)
    print(f"  CG info={info1}")

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Compute star flat from initial solution
    # ──────────────────────────────────────────────────────────────────
    print("\n  --- Stage 2: Computing star flat ---")

    # Load MJD for epoch assignment
    mjd_df = pd.read_parquet(CACHE / "nsc_exposure_mjd_g.parquet")
    exposure_mjds = dict(zip(mjd_df["expnum"].values, mjd_df["mjd"].values))

    # Epoch boundaries from config (g-band: MJD 56404 is the baffling boundary)
    epoch_boundaries = [56404.0]

    # Per-exposure gray = median ZP across CCDs
    exp_zps = defaultdict(list)
    for idx, (expnum, ccdnum) in enumerate(idx_to_node):
        exp_zps[expnum].append((int(ccdnum), x_initial[idx]))

    gray = {}
    for expnum, ccd_zps in exp_zps.items():
        gray[expnum] = np.median([zp for _, zp in ccd_zps])

    # Helper for epoch index
    def get_epoch_index(mjd, boundaries):
        for i, b in enumerate(boundaries):
            if mjd < b:
                return i
        return len(boundaries)

    # Star flat = median(ZP - gray) per (ccdnum, epoch)
    epoch_deltas = defaultdict(list)
    for expnum, ccd_zps in exp_zps.items():
        mjd = exposure_mjds.get(expnum)
        if mjd is None:
            continue
        epoch_idx = get_epoch_index(mjd, epoch_boundaries)
        g = gray[expnum]
        for ccdnum, zp in ccd_zps:
            epoch_deltas[(ccdnum, epoch_idx)].append(zp - g)

    star_flat = {}
    for key, deltas in epoch_deltas.items():
        star_flat[key] = float(np.median(deltas)) if len(deltas) >= 3 else 0.0

    flat_vals = [v for v in star_flat.values() if v != 0]
    if flat_vals:
        print(f"  Star flat: {len(star_flat)} params, range [{min(flat_vals)*1000:.1f}, "
              f"{max(flat_vals)*1000:.1f}] mmag, RMS {np.sqrt(np.mean(np.array(flat_vals)**2))*1000:.1f} mmag")

    # ──────────────────────────────────────────────────────────────────
    # Stage 3: Gray solve (per-exposure parameters, star flat applied)
    # ──────────────────────────────────────────────────────────────────
    print("\n  --- Stage 3: Gray solve (per-exposure) ---")

    # Build exposure index
    all_expnums = sorted(set(e for e, c in nodes))
    exp_to_idx = {e: i for i, e in enumerate(all_expnums)}
    n_exp = len(all_expnums)
    print(f"  Exposure parameters: {n_exp:,}")

    # Build gray normal equations with star flat correction
    gray_rhs = np.zeros(n_exp)
    gray_rows, gray_cols, gray_vals = [], [], []
    n_gray_pairs = 0
    n_intra_skip = 0

    for star_id, group in det_filtered.groupby("objectid"):
        n_det = len(group)
        if n_det < 2:
            continue
        expnums = group["expnum"].values
        ccdnums = group["ccdnum"].values
        m_insts = group["m_inst"].values
        m_errs = group["m_err"].values

        corr_mags = []
        corr_errs = []
        corr_exp_idx = []
        for j in range(n_det):
            eidx = exp_to_idx.get(int(expnums[j]))
            if eidx is None:
                continue
            mjd = exposure_mjds.get(int(expnums[j]))
            if mjd is None:
                continue
            epoch = get_epoch_index(mjd, epoch_boundaries)
            flat = star_flat.get((int(ccdnums[j]), epoch), 0.0)
            corr_mags.append(m_insts[j] + flat)
            corr_errs.append(m_errs[j])
            corr_exp_idx.append(eidx)

        nc = len(corr_exp_idx)
        if nc < 2:
            continue

        for a in range(nc):
            for b in range(a + 1, nc):
                ia, ib = corr_exp_idx[a], corr_exp_idx[b]
                if ia == ib:
                    n_intra_skip += 1
                    continue
                w = 1.0 / (corr_errs[a] ** 2 + corr_errs[b] ** 2)
                dm = corr_mags[a] - corr_mags[b]
                gray_rows.append(ia); gray_cols.append(ia); gray_vals.append(w)
                gray_rows.append(ib); gray_cols.append(ib); gray_vals.append(w)
                gray_rows.append(ia); gray_cols.append(ib); gray_vals.append(-w)
                gray_rows.append(ib); gray_cols.append(ia); gray_vals.append(-w)
                gray_rhs[ia] -= w * dm
                gray_rhs[ib] += w * dm
                n_gray_pairs += 1

    print(f"  Gray pairs: {n_gray_pairs:,}, intra-exposure skipped: {n_intra_skip:,}")

    gray_AtWA = sparse.csr_matrix(
        (gray_vals, (gray_rows, gray_cols)), shape=(n_exp, n_exp)
    ) + 1e-6 * sparse.eye(n_exp)

    print("  Solving CG (gray)...")
    gray_solved, info_g = cg(gray_AtWA, gray_rhs, rtol=1e-5, maxiter=5000)
    print(f"  CG info={info_g}")

    # Reconstruct full ZP = gray + star_flat
    zp_full = np.zeros(n_params)
    for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
        eidx = exp_to_idx.get(expnum)
        if eidx is None:
            continue
        mjd = exposure_mjds.get(expnum)
        epoch = get_epoch_index(mjd, epoch_boundaries) if mjd else 0
        flat = star_flat.get((int(ccdnum), epoch), 0.0)
        zp_full[node_idx] = gray_solved[eidx] + flat

    # DES median shift (zp_full is relative/near-zero before shift,
    # so only filter on FGCM validity, not on zp_full range)
    des_fgcm = load_des_fgcm_zps(CACHE, BAND)
    des_diffs = []
    for i in range(n_params):
        node = idx_to_node[i]
        if node in des_fgcm:
            fgcm_val = des_fgcm[node]
            if 25.0 < fgcm_val < 35.0:
                des_diffs.append(fgcm_val - zp_full[i])
    if des_diffs:
        shift = np.median(des_diffs)
        zp_full += shift
        print(f"  DES median shift: {shift:.4f} mag")

    # ──────────────────────────────────────────────────────────────────
    # Compare with full Phase 3 solve
    # ──────────────────────────────────────────────────────────────────
    res_df = pd.DataFrame({
        "expnum": [n[0] for n in idx_to_node],
        "ccdnum": [n[1] for n in idx_to_node],
        "zp_normal_only": zp_full,
    })
    res_df = res_df.merge(zp_df[["expnum", "ccdnum", "zp_solved", "zp_fgcm"]],
                          on=["expnum", "ccdnum"], how="left")

    exp_pos = compute_exposure_ra_dec(det)
    res_df = res_df.merge(exp_pos, on="expnum", how="left")

    valid = res_df[
        (res_df["zp_fgcm"].notna()) & (res_df["zp_fgcm"] > 25) & (res_df["zp_fgcm"] < 35) &
        (res_df["zp_normal_only"] > 25) & (res_df["zp_normal_only"] < 35) &
        (res_df["zp_solved"] > 25) & (res_df["zp_solved"] < 35)
    ].dropna(subset=["zp_normal_only", "zp_solved"])

    diff_normal = valid["zp_normal_only"] - valid["zp_fgcm"]
    diff_full = valid["zp_solved"] - valid["zp_fgcm"]

    print(f"\n  DES comparison:")
    print(f"    N DES CCD-exposures: {len(valid):,}")
    print(f"    RMS(ZP_normal_only - FGCM): {diff_normal.std()*1000:.1f} mmag")
    print(f"    RMS(ZP_full_solve - FGCM):  {diff_full.std()*1000:.1f} mmag")

    ra_bins = np.arange(49, 62, 1.0)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])
    grad_normal = []
    grad_full = []
    for i in range(len(ra_bins) - 1):
        mask = (valid["exp_ra"] >= ra_bins[i]) & (valid["exp_ra"] < ra_bins[i + 1])
        grad_normal.append(diff_normal[mask].median() * 1000 if mask.sum() > 10 else np.nan)
        grad_full.append(diff_full[mask].median() * 1000 if mask.sum() > 10 else np.nan)

    grad_normal = np.array(grad_normal)
    grad_full = np.array(grad_full)

    print(f"\n  RA gradient (median ZP - FGCM per bin, mmag):")
    print(f"  {'RA':>6s}  {'Normal-only':>12s}  {'Full solve':>12s}  {'Difference':>12s}")
    for i in range(len(ra_centers)):
        if not np.isnan(grad_normal[i]):
            print(f"  {ra_centers[i]:6.1f}  {grad_normal[i]:12.1f}  {grad_full[i]:12.1f}  "
                  f"{grad_normal[i] - grad_full[i]:12.1f}")

    range_normal = np.nanmax(grad_normal) - np.nanmin(grad_normal)
    range_full = np.nanmax(grad_full) - np.nanmin(grad_full)
    print(f"\n  *** RA gradient range (normal-only, star flat + gray): {range_normal:.1f} mmag ***")
    print(f"  *** RA gradient range (full solve):                    {range_full:.1f} mmag ***")

    if range_normal < range_full * 0.5:
        print(f"\n  --> CONCLUSION: Gradient REDUCED by >50% when deep fields removed!")
        print(f"  --> Deep fields ARE the primary cause of the RA gradient.")
    elif range_normal > range_full * 0.8:
        print(f"\n  --> CONCLUSION: Gradient PERSISTS even without deep fields.")
        print(f"  --> Deep fields are NOT the primary cause. Look at other systematics.")
    else:
        print(f"\n  --> CONCLUSION: Gradient partially reduced ({range_full:.0f} -> {range_normal:.0f} mmag).")
        print(f"  --> Deep fields contribute but are not the sole cause.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(ra_centers, grad_normal, "bo-", lw=2, ms=6,
            label=f"Normal-only (range={range_normal:.0f} mmag)")
    ax.plot(ra_centers, grad_full, "rs-", lw=2, ms=6,
            label=f"Full solve (range={range_full:.0f} mmag)")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Median (ZP_solved - ZP_FGCM) [mmag]")
    ax.set_title("RA gradient: normal-only vs full solve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(diff_normal * 1000, bins=100, range=(-200, 200), alpha=0.5, color="blue",
            label=f"Normal-only (RMS={diff_normal.std()*1000:.1f})")
    ax.hist(diff_full * 1000, bins=100, range=(-200, 200), alpha=0.5, color="red",
            label=f"Full solve (RMS={diff_full.std()*1000:.1f})")
    ax.set_xlabel("ZP_solved - ZP_FGCM [mmag]")
    ax.set_ylabel("N")
    ax.set_title("ZP residual distributions")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(valid["exp_ra"], valid["exp_dec"],
                    c=diff_normal * 1000, s=1, alpha=0.3,
                    cmap="RdBu_r", vmin=-100, vmax=100)
    plt.colorbar(sc, ax=ax, label="ZP - FGCM [mmag]")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Normal-only solve: ZP residual map")
    ax.invert_xaxis()

    ax = axes[1, 1]
    diff_solves = valid["zp_normal_only"] - valid["zp_solved"]
    binned_diff = []
    for i in range(len(ra_bins) - 1):
        mask = (valid["exp_ra"] >= ra_bins[i]) & (valid["exp_ra"] < ra_bins[i + 1])
        binned_diff.append(diff_solves[mask].median() * 1000 if mask.sum() > 10 else np.nan)
    ax.plot(ra_centers, binned_diff, "go-", lw=2, ms=6)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Median (ZP_normal - ZP_full) [mmag]")
    ax.set_title("Difference between two solves vs RA")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Analysis 7: CRITICAL TEST - Normal-only (star flat + gray) vs Full solve",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTDIR / "ra_investigation_7_normal_only_solve.png", dpi=150)
    plt.close()
    print(f"  Saved: ra_investigation_7_normal_only_solve.png")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 72)
    print("RA GRADIENT INVESTIGATION - g-band test patch")
    print("=" * 72)

    # Load data
    det = load_all_phase0()
    exptime_df = load_exptime()
    zp_df = load_zeropoints()
    flagged_nodes = load_flagged_nodes()
    nsc_exp = load_nsc_exposure()
    fwhm_df = load_phase0_with_fwhm()

    # Run all analyses
    analysis_0_existing_gradient(det, zp_df, exptime_df, flagged_nodes)
    analysis_1_exptime_vs_ra(det, exptime_df)
    analysis_2_minst_systematics(det, exptime_df)
    analysis_3_fwhm_vs_ra(det, fwhm_df, exptime_df)
    analysis_4_aperture_correction(det, fwhm_df, exptime_df)
    analysis_5_program_breakdown(det, exptime_df, nsc_exp)
    analysis_6_graph_connectivity(det, exptime_df)
    analysis_7_normal_only_solve(det, exptime_df, zp_df, flagged_nodes)

    print("\n" + "=" * 72)
    print("ALL ANALYSES COMPLETE")
    print("Plots saved to: " + str(OUTDIR))
    print("=" * 72)


if __name__ == "__main__":
    main()
