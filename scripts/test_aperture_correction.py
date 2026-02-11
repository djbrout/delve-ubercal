#!/usr/bin/env python3
"""
Test whether FWHM-dependent aperture correction reduces the RA gradient
in the ubercal solution.

Hypothesis: Seeing-dependent aperture losses create systematic biases
in the overlap constraints. Exposures at different RA/Dec have different
seeing distributions, which creates a false gradient in the solved ZPs.

Steps:
1. Load Phase 0 g-band detections + FWHM from raw cache
2. Compute per-detection residuals (m_inst - weighted_star_mean)
3. Fit aperture correction curve AC(FWHM) from binned medians
4. Apply correction: m_inst_corr = m_inst - AC(FWHM)
5. Build normal equations with corrected magnitudes
6. Solve and compare RA gradient to original solution
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.sparse as sp
from scipy.sparse.linalg import cg

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
BASE = Path("/Volumes/External5TB/DELVE_UBERCAL")
OUTPUT_DIR = BASE / "output"
CACHE_DIR = BASE / "cache"
PHASE0_DIR = OUTPUT_DIR / "phase0_g"
PHASE1_DIR = OUTPUT_DIR / "phase1_g"
PHASE3_DIR = OUTPUT_DIR / "phase3_g"
VAL_DIR = OUTPUT_DIR / "validation_g"
RAW_CACHE_DIR = CACHE_DIR / "phase0_g"

BAND = "g"
FWHM_MIN = 0.5   # arcsec, quality floor
FWHM_MAX = 3.0   # arcsec, quality ceiling
FWHM_FIT_MIN = 0.8   # fit range
FWHM_FIT_MAX = 2.0   # fit range
POLY_DEGREE = 3   # for aperture correction polynomial

TIKHONOV_REG = 1.0e-10
CG_TOL = 1.0e-5
CG_MAXITER = 5000

VAL_DIR.mkdir(parents=True, exist_ok=True)


def load_exposure_map():
    """Load exposure string -> expnum mapping."""
    exp = pd.read_parquet(CACHE_DIR / f"nsc_exposure_{BAND}.parquet",
                          columns=["exposure", "expnum"])
    return dict(zip(exp["exposure"].values, exp["expnum"].values))


def load_detections_with_fwhm():
    """Load Phase 0 detections joined with FWHM from raw cache.

    Returns DataFrame with columns:
        objectid, expnum, ccdnum, m_inst, m_err, fwhm, ra, dec
    """
    print("Loading detections with FWHM...", flush=True)
    exp_map = load_exposure_map()

    # Load Phase 0 output (has expnum)
    p0_files = sorted(PHASE0_DIR.glob("detections_nside32_pixel*.parquet"))
    print(f"  Phase 0 files: {len(p0_files)}", flush=True)

    # Load raw cache (has fwhm + exposure string)
    raw_files = sorted(RAW_CACHE_DIR.glob("raw_g_nside32_pixel*.parquet"))
    print(f"  Raw cache files: {len(raw_files)}", flush=True)

    # Strategy: load raw cache, add expnum, then merge with Phase 0 on
    # (objectid, expnum, ccdnum) to get the FWHM for each Phase 0 detection
    all_p0 = []
    for f in p0_files:
        df = pd.read_parquet(f)
        if len(df) > 0:
            all_p0.append(df)
    p0_df = pd.concat(all_p0, ignore_index=True)
    print(f"  Total Phase 0 detections: {len(p0_df):,}", flush=True)

    # Load FWHM from raw cache pixel by pixel and build lookup
    all_raw = []
    for f in raw_files:
        # Extract pixel number from filename
        pixel_str = f.stem.replace("raw_g_nside32_pixel", "")
        pixel = int(pixel_str)
        raw = pd.read_parquet(f, columns=["objectid", "exposure", "ccdnum", "fwhm"])
        if len(raw) == 0:
            continue
        # Map exposure string -> expnum
        raw["expnum"] = raw["exposure"].map(exp_map)
        raw = raw.dropna(subset=["expnum"])
        raw["expnum"] = raw["expnum"].astype(np.int64)
        raw = raw[["objectid", "expnum", "ccdnum", "fwhm"]]
        all_raw.append(raw)

    raw_df = pd.concat(all_raw, ignore_index=True)
    # Drop duplicate (objectid, expnum, ccdnum) if any
    raw_df = raw_df.drop_duplicates(subset=["objectid", "expnum", "ccdnum"])
    print(f"  Total raw detections with FWHM: {len(raw_df):,}", flush=True)

    # Merge to get FWHM on Phase 0 detections
    merged = p0_df.merge(raw_df, on=["objectid", "expnum", "ccdnum"], how="left")
    n_with_fwhm = merged["fwhm"].notna().sum()
    print(f"  Matched FWHM: {n_with_fwhm:,} / {len(merged):,} "
          f"({100*n_with_fwhm/len(merged):.1f}%)", flush=True)

    # Drop detections without FWHM
    merged = merged.dropna(subset=["fwhm"]).copy()

    # Quality cuts on FWHM
    before = len(merged)
    merged = merged[(merged["fwhm"] > FWHM_MIN) & (merged["fwhm"] < FWHM_MAX)]
    print(f"  After FWHM quality cuts ({FWHM_MIN}-{FWHM_MAX} arcsec): "
          f"{len(merged):,} ({before - len(merged):,} removed)", flush=True)

    return merged


def compute_residuals(det_df):
    """Compute per-detection residuals = m_inst - weighted_star_mean.

    Uses inverse-variance weighting.
    """
    print("Computing per-detection residuals...", flush=True)

    # Weighted mean per star
    det_df = det_df.copy()
    det_df["w"] = 1.0 / (det_df["m_err"] ** 2)
    star_stats = det_df.groupby("objectid").agg(
        wm_inst=("m_inst", lambda x: np.average(x, weights=det_df.loc[x.index, "w"])),
        n_det=("m_inst", "count"),
    ).reset_index()

    # Keep only stars with >= 2 detections
    star_stats = star_stats[star_stats["n_det"] >= 2]
    print(f"  Stars with >= 2 detections: {len(star_stats):,}", flush=True)

    # Merge star mean back
    det_df = det_df.merge(star_stats[["objectid", "wm_inst"]], on="objectid", how="inner")
    det_df["residual"] = det_df["m_inst"] - det_df["wm_inst"]

    print(f"  Detections with residuals: {len(det_df):,}", flush=True)
    print(f"  Residual median: {det_df['residual'].median()*1000:.2f} mmag", flush=True)
    print(f"  Residual RMS: {det_df['residual'].std()*1000:.2f} mmag", flush=True)

    return det_df


def fit_aperture_correction(det_df):
    """Fit AC(FWHM) polynomial from binned medians of residual vs FWHM.

    Returns polynomial coefficients and bin data for plotting.
    """
    print("Fitting aperture correction curve...", flush=True)

    fwhm = det_df["fwhm"].values
    resid = det_df["residual"].values

    # Bin by FWHM
    fwhm_bins = np.arange(FWHM_FIT_MIN, FWHM_FIT_MAX + 0.05, 0.05)
    bin_centers = []
    bin_medians = []
    bin_stds = []
    bin_counts = []

    for i in range(len(fwhm_bins) - 1):
        mask = (fwhm >= fwhm_bins[i]) & (fwhm < fwhm_bins[i + 1])
        n = mask.sum()
        if n < 50:
            continue
        bin_centers.append((fwhm_bins[i] + fwhm_bins[i + 1]) / 2)
        bin_medians.append(np.median(resid[mask]))
        bin_stds.append(np.std(resid[mask]) / np.sqrt(n))
        bin_counts.append(n)

    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)

    print(f"  Bins used: {len(bin_centers)}", flush=True)

    # Fit polynomial to binned medians (weighted by 1/std^2)
    weights = 1.0 / (bin_stds ** 2)
    # Normalize weights
    weights /= weights.sum()
    poly_coeffs = np.polyfit(bin_centers, bin_medians, POLY_DEGREE, w=np.sqrt(weights))
    poly = np.poly1d(poly_coeffs)

    # Evaluate range of correction
    fwhm_eval = np.linspace(FWHM_FIT_MIN, FWHM_FIT_MAX, 200)
    ac_eval = poly(fwhm_eval)
    ac_range = ac_eval.max() - ac_eval.min()
    print(f"  AC polynomial degree: {POLY_DEGREE}", flush=True)
    print(f"  AC range over {FWHM_FIT_MIN}-{FWHM_FIT_MAX} arcsec: "
          f"{ac_range*1000:.1f} mmag", flush=True)
    print(f"  AC at FWHM=0.9: {poly(0.9)*1000:.2f} mmag", flush=True)
    print(f"  AC at FWHM=1.0: {poly(1.0)*1000:.2f} mmag", flush=True)
    print(f"  AC at FWHM=1.2: {poly(1.2)*1000:.2f} mmag", flush=True)
    print(f"  AC at FWHM=1.5: {poly(1.5)*1000:.2f} mmag", flush=True)
    print(f"  AC at FWHM=2.0: {poly(2.0)*1000:.2f} mmag", flush=True)

    return poly, poly_coeffs, bin_centers, bin_medians, bin_stds, bin_counts


def plot_aperture_correction(det_df, poly, bin_centers, bin_medians, bin_stds):
    """Plot residual vs FWHM with aperture correction curve."""
    print("Plotting aperture correction...", flush=True)

    fwhm = det_df["fwhm"].values
    resid = det_df["residual"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: 2D histogram
    ax = axes[0]
    mask_plot = (fwhm >= 0.5) & (fwhm <= 2.5) & (np.abs(resid) < 0.15)
    h = ax.hexbin(fwhm[mask_plot], resid[mask_plot] * 1000,
                   gridsize=80, mincnt=5, norm=LogNorm(), cmap="viridis")
    plt.colorbar(h, ax=ax, label="count")

    # Overlay binned medians
    ax.errorbar(bin_centers, bin_medians * 1000, yerr=bin_stds * 1000,
                fmt="o", color="red", ms=4, capsize=2, label="binned median")

    # Overlay polynomial fit
    fwhm_fine = np.linspace(FWHM_FIT_MIN, FWHM_FIT_MAX, 200)
    ax.plot(fwhm_fine, poly(fwhm_fine) * 1000, "r-", lw=2, label=f"poly deg {POLY_DEGREE}")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("Residual (mmag)")
    ax.set_title("mag_aper4 residual vs FWHM (g-band)")
    ac_range = (poly(fwhm_fine).max() - poly(fwhm_fine).min()) * 1000
    ax.legend(title=f"AC range = {ac_range:.1f} mmag")
    ax.set_ylim(-50, 50)

    # Right: just the binned medians + fit, zoomed
    ax = axes[1]
    ax.errorbar(bin_centers, bin_medians * 1000, yerr=bin_stds * 1000,
                fmt="o", color="red", ms=5, capsize=2, label="binned median")
    ax.plot(fwhm_fine, poly(fwhm_fine) * 1000, "b-", lw=2, label=f"poly deg {POLY_DEGREE}")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("FWHM (arcsec)")
    ax.set_ylabel("Residual (mmag)")
    ax.set_title("Aperture correction curve AC(FWHM)")
    ax.legend()
    ax.set_ylim(-15, 15)

    fig.tight_layout()
    outf = VAL_DIR / "aperture_correction_curve.png"
    fig.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outf}", flush=True)


def build_corrected_star_lists(det_df, poly):
    """Apply aperture correction and return corrected star list files.

    Instead of files, returns a dict: pixel -> DataFrame with corrected m_inst.
    """
    print("Applying aperture correction to detections...", flush=True)

    # Apply correction: m_inst_corr = m_inst - AC(FWHM)
    # AC(FWHM) is the systematic bias, so subtracting it removes the bias
    det_df = det_df.copy()
    det_df["ac"] = poly(det_df["fwhm"].values)
    det_df["m_inst_corr"] = det_df["m_inst"] - det_df["ac"]

    print(f"  AC applied to {len(det_df):,} detections", flush=True)
    print(f"  AC median: {det_df['ac'].median()*1000:.2f} mmag", flush=True)
    print(f"  AC std: {det_df['ac'].std()*1000:.2f} mmag", flush=True)

    return det_df


def accumulate_normal_equations_from_df(det_df, node_to_idx, n_params,
                                         m_col="m_inst"):
    """Build normal equations from a DataFrame (not files).

    Parameters
    ----------
    det_df : DataFrame
        Must have columns: objectid, expnum, ccdnum, <m_col>, m_err
    node_to_idx : dict
    n_params : int
    m_col : str
        Column name for magnitudes to use.

    Returns
    -------
    AtWA, rhs, n_stars_used, n_pairs_total
    """
    rows_list = []
    cols_list = []
    vals_list = []
    rhs = np.zeros(n_params, dtype=np.float64)
    n_stars_used = 0
    n_pairs_total = 0

    # Process star by star
    for star_id, group in det_df.groupby("objectid"):
        if len(group) < 2:
            continue

        mags = group[m_col].values
        errs = group["m_err"].values
        expnums = group["expnum"].values
        ccdnums = group["ccdnum"].values

        indices = []
        star_mags = []
        star_errs = []
        for j in range(len(expnums)):
            node = (int(expnums[j]), int(ccdnums[j]))
            idx = node_to_idx.get(node)
            if idx is not None:
                indices.append(idx)
                star_mags.append(mags[j])
                star_errs.append(errs[j])

        n_det = len(indices)
        if n_det < 2:
            continue

        n_stars_used += 1
        indices = np.array(indices, dtype=np.int64)
        star_mags = np.array(star_mags)
        star_errs = np.array(star_errs)

        for a in range(n_det):
            for b in range(a + 1, n_det):
                ia, ib = indices[a], indices[b]
                ma, mb = star_mags[a], star_mags[b]
                ea, eb = star_errs[a], star_errs[b]

                w = 1.0 / (ea * ea + eb * eb)
                dm = ma - mb

                rows_list.append(ia); cols_list.append(ia); vals_list.append(w)
                rows_list.append(ib); cols_list.append(ib); vals_list.append(w)
                rows_list.append(ia); cols_list.append(ib); vals_list.append(-w)
                rows_list.append(ib); cols_list.append(ia); vals_list.append(-w)

                rhs[ia] -= w * dm
                rhs[ib] += w * dm
                n_pairs_total += 1

    rows_arr = np.array(rows_list, dtype=np.int64)
    cols_arr = np.array(cols_list, dtype=np.int64)
    vals_arr = np.array(vals_list, dtype=np.float64)
    AtWA = sp.coo_matrix((vals_arr, (rows_arr, cols_arr)),
                          shape=(n_params, n_params)).tocsr()

    return AtWA, rhs, n_stars_used, n_pairs_total


def solve_and_shift(AtWA, rhs, node_to_idx, des_fgcm_zps, label=""):
    """Solve with Tikhonov + CG, then shift to DES median."""
    n = AtWA.shape[0]
    AtWA_reg = AtWA + sp.eye(n, format="csr") * TIKHONOV_REG

    t0 = time.time()
    n_iter = [0]
    def callback(xk):
        n_iter[0] += 1

    zp_solved, cg_info = cg(
        AtWA_reg, rhs,
        rtol=CG_TOL,
        maxiter=CG_MAXITER,
        callback=callback,
    )
    solve_time = time.time() - t0

    residual = AtWA_reg @ zp_solved - rhs
    rel_residual = (np.linalg.norm(residual) / np.linalg.norm(rhs)
                    if np.linalg.norm(rhs) > 0 else 0)

    # Shift to DES median
    des_diffs = []
    for node, idx in node_to_idx.items():
        if node in des_fgcm_zps:
            fgcm_val = des_fgcm_zps[node]
            if 25.0 < fgcm_val < 35.0:
                des_diffs.append(zp_solved[idx] - fgcm_val)

    if des_diffs:
        offset = np.median(des_diffs)
        zp_solved -= offset
    else:
        offset = 0.0

    print(f"  [{label}] CG converged={cg_info==0}, iters={n_iter[0]}, "
          f"rel_resid={rel_residual:.2e}, time={solve_time:.1f}s, "
          f"DES offset={offset*1000:.1f} mmag", flush=True)

    return zp_solved


def compute_ra_gradient(zp_solved, node_to_idx, idx_to_node, des_fgcm_zps,
                         node_positions, flagged_nodes=None):
    """Compute ZP_solved - ZP_FGCM vs RA for DES nodes.

    Returns RA bin data and linear gradient fit.
    """
    # Build arrays
    ra_vals = []
    diff_vals = []

    for idx, node in enumerate(idx_to_node):
        if flagged_nodes and node in flagged_nodes:
            continue
        if node not in des_fgcm_zps:
            continue
        fgcm_val = des_fgcm_zps[node]
        if not (25.0 < fgcm_val < 35.0):
            continue
        pos = node_positions.get(node)
        if pos is None:
            continue

        ra_vals.append(pos[0])
        diff_vals.append(zp_solved[idx] - fgcm_val)

    ra_vals = np.array(ra_vals)
    diff_vals = np.array(diff_vals)

    # Bin by RA
    ra_bins = np.arange(48, 62, 1.0)
    bin_centers = []
    bin_medians = []
    bin_stds = []
    bin_counts = []
    for i in range(len(ra_bins) - 1):
        mask = (ra_vals >= ra_bins[i]) & (ra_vals < ra_bins[i + 1])
        n = mask.sum()
        if n < 10:
            continue
        bin_centers.append((ra_bins[i] + ra_bins[i + 1]) / 2)
        bin_medians.append(np.median(diff_vals[mask]))
        bin_stds.append(np.std(diff_vals[mask]) / np.sqrt(n))
        bin_counts.append(n)

    bin_centers = np.array(bin_centers)
    bin_medians = np.array(bin_medians)
    bin_stds = np.array(bin_stds)

    # Fit linear gradient
    if len(bin_centers) >= 3:
        coeffs = np.polyfit(bin_centers, bin_medians, 1)
        slope = coeffs[0]  # mmag/deg
    else:
        slope = 0.0
        coeffs = [0, 0]

    # Total gradient across RA range
    ra_range = ra_vals.max() - ra_vals.min()
    total_gradient = slope * ra_range

    # RMS of DES diff
    rms = np.sqrt(np.mean(diff_vals ** 2))

    return {
        "bin_centers": bin_centers,
        "bin_medians": bin_medians,
        "bin_stds": bin_stds,
        "slope_mmag_per_deg": slope * 1000,
        "total_gradient_mmag": total_gradient * 1000,
        "rms_mmag": rms * 1000,
        "ra_range": ra_range,
        "n_des_nodes": len(diff_vals),
        "linear_coeffs": coeffs,
    }


def plot_comparison(grad_orig, grad_corr, label_orig="Original", label_corr="AC-corrected"):
    """Plot RA gradient comparison."""
    print("Plotting RA gradient comparison...", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: RA gradient for both
    ax = axes[0]
    ax.errorbar(grad_orig["bin_centers"], grad_orig["bin_medians"] * 1000,
                yerr=grad_orig["bin_stds"] * 1000,
                fmt="o-", color="blue", ms=5, capsize=3, label=label_orig)
    ax.errorbar(grad_corr["bin_centers"], grad_corr["bin_medians"] * 1000,
                yerr=grad_corr["bin_stds"] * 1000,
                fmt="s-", color="red", ms=5, capsize=3, label=label_corr)

    # Linear fits
    ra_fine = np.linspace(48, 62, 100)
    ax.plot(ra_fine, np.polyval(grad_orig["linear_coeffs"], ra_fine) * 1000,
            "b--", alpha=0.5, label=f"slope={grad_orig['slope_mmag_per_deg']:.2f} mmag/deg")
    ax.plot(ra_fine, np.polyval(grad_corr["linear_coeffs"], ra_fine) * 1000,
            "r--", alpha=0.5, label=f"slope={grad_corr['slope_mmag_per_deg']:.2f} mmag/deg")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("ZP_solved - ZP_FGCM (mmag)")
    ax.set_title("RA gradient: Original vs AC-corrected")
    ax.legend(fontsize=8)

    # Right: summary text
    ax = axes[1]
    ax.axis("off")
    summary = (
        f"APERTURE CORRECTION TEST (g-band)\n"
        f"{'='*50}\n\n"
        f"{'Metric':<35} {'Original':>12} {'AC-corr':>12}\n"
        f"{'-'*59}\n"
        f"{'RA gradient (mmag/deg)':<35} {grad_orig['slope_mmag_per_deg']:>12.2f} {grad_corr['slope_mmag_per_deg']:>12.2f}\n"
        f"{'Total gradient (mmag)':<35} {grad_orig['total_gradient_mmag']:>12.1f} {grad_corr['total_gradient_mmag']:>12.1f}\n"
        f"{'DES diff RMS (mmag)':<35} {grad_orig['rms_mmag']:>12.1f} {grad_corr['rms_mmag']:>12.1f}\n"
        f"{'DES nodes used':<35} {grad_orig['n_des_nodes']:>12,} {grad_corr['n_des_nodes']:>12,}\n"
        f"{'RA range (deg)':<35} {grad_orig['ra_range']:>12.1f} {grad_corr['ra_range']:>12.1f}\n"
        f"\n"
        f"Gradient improvement: {abs(grad_orig['total_gradient_mmag']) - abs(grad_corr['total_gradient_mmag']):.1f} mmag\n"
        f"RMS improvement: {grad_orig['rms_mmag'] - grad_corr['rms_mmag']:.1f} mmag\n"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", verticalalignment="top")

    fig.tight_layout()
    outf = VAL_DIR / "aperture_correction_ra_gradient.png"
    fig.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outf}", flush=True)


def plot_fwhm_vs_ra(det_df):
    """Plot mean FWHM vs RA to check for systematic seeing variations."""
    print("Plotting FWHM vs RA...", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-detection FWHM vs RA (hexbin)
    ax = axes[0]
    mask = det_df["fwhm"].between(0.5, 2.5)
    h = ax.hexbin(det_df.loc[mask, "ra"], det_df.loc[mask, "fwhm"],
                   gridsize=60, mincnt=5, norm=LogNorm(), cmap="viridis")
    plt.colorbar(h, ax=ax, label="count")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("FWHM (arcsec)")
    ax.set_title("Detection FWHM vs RA")

    # Right: binned median FWHM vs RA
    ax = axes[1]
    ra_bins = np.arange(48, 62, 0.5)
    bin_centers = []
    bin_medians = []
    bin_stds = []
    for i in range(len(ra_bins) - 1):
        m = (det_df["ra"] >= ra_bins[i]) & (det_df["ra"] < ra_bins[i + 1])
        n = m.sum()
        if n < 50:
            continue
        bin_centers.append((ra_bins[i] + ra_bins[i + 1]) / 2)
        bin_medians.append(det_df.loc[m, "fwhm"].median())
        bin_stds.append(det_df.loc[m, "fwhm"].std() / np.sqrt(n))

    ax.errorbar(bin_centers, bin_medians, yerr=bin_stds,
                fmt="o-", color="blue", ms=4, capsize=2)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Median FWHM (arcsec)")
    ax.set_title("Median FWHM vs RA")

    fwhm_range = max(bin_medians) - min(bin_medians) if bin_medians else 0
    ax.text(0.05, 0.95, f"FWHM range across RA: {fwhm_range:.3f} arcsec",
            transform=ax.transAxes, fontsize=10, verticalalignment="top")

    fig.tight_layout()
    outf = VAL_DIR / "aperture_correction_fwhm_vs_ra.png"
    fig.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outf}", flush=True)


def plot_ac_impact_per_exposure(det_df, poly, node_to_idx, idx_to_node, node_positions):
    """Plot the mean AC(FWHM) per CCD-exposure vs RA.

    This shows how much the aperture correction shifts each node's
    effective magnitude, and whether this correlates with RA.
    """
    print("Plotting per-exposure AC impact...", flush=True)

    det_df = det_df.copy()
    det_df["ac"] = poly(det_df["fwhm"].values)

    # Compute mean AC per (expnum, ccdnum)
    exp_ac = det_df.groupby(["expnum", "ccdnum"]).agg(
        mean_ac=("ac", "mean"),
        median_fwhm=("fwhm", "median"),
        n_det=("ac", "count"),
    ).reset_index()

    # Add RA from node_positions
    ras = []
    for _, row in exp_ac.iterrows():
        node = (int(row["expnum"]), int(row["ccdnum"]))
        pos = node_positions.get(node)
        if pos is not None:
            ras.append(pos[0])
        else:
            ras.append(np.nan)
    exp_ac["ra"] = ras
    exp_ac = exp_ac.dropna(subset=["ra"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean AC per node vs RA
    ax = axes[0]
    h = ax.hexbin(exp_ac["ra"], exp_ac["mean_ac"] * 1000,
                   gridsize=50, mincnt=3, norm=LogNorm(), cmap="viridis")
    plt.colorbar(h, ax=ax, label="count")

    # Binned median
    ra_bins = np.arange(48, 62, 1.0)
    bc, bm = [], []
    for i in range(len(ra_bins) - 1):
        m = (exp_ac["ra"] >= ra_bins[i]) & (exp_ac["ra"] < ra_bins[i + 1])
        if m.sum() > 10:
            bc.append((ra_bins[i] + ra_bins[i + 1]) / 2)
            bm.append(exp_ac.loc[m, "mean_ac"].median())
    ax.plot(bc, np.array(bm) * 1000, "r-o", ms=5, lw=2, label="binned median")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Mean AC per node (mmag)")
    ax.set_title("Per-node aperture correction vs RA")
    ax.legend()

    # Right: median FWHM per node vs RA
    ax = axes[1]
    h = ax.hexbin(exp_ac["ra"], exp_ac["median_fwhm"],
                   gridsize=50, mincnt=3, norm=LogNorm(), cmap="viridis")
    plt.colorbar(h, ax=ax, label="count")
    bc2, bm2 = [], []
    for i in range(len(ra_bins) - 1):
        m = (exp_ac["ra"] >= ra_bins[i]) & (exp_ac["ra"] < ra_bins[i + 1])
        if m.sum() > 10:
            bc2.append((ra_bins[i] + ra_bins[i + 1]) / 2)
            bm2.append(exp_ac.loc[m, "median_fwhm"].median())
    ax.plot(bc2, bm2, "r-o", ms=5, lw=2, label="binned median")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Median FWHM per node (arcsec)")
    ax.set_title("Per-node median FWHM vs RA")
    ax.legend()

    fig.tight_layout()
    outf = VAL_DIR / "aperture_correction_per_exposure.png"
    fig.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outf}", flush=True)


def main():
    print("=" * 70)
    print("APERTURE CORRECTION TEST")
    print("Hypothesis: FWHM-dependent aperture losses create RA gradient")
    print("=" * 70)
    print()

    t_start = time.time()

    # ===================================================================
    # Step 1: Load detections with FWHM
    # ===================================================================
    det_df = load_detections_with_fwhm()
    print()

    # ===================================================================
    # Step 2: Compute per-detection residuals
    # ===================================================================
    det_df = compute_residuals(det_df)
    print()

    # ===================================================================
    # Step 3: Fit aperture correction curve
    # ===================================================================
    poly, poly_coeffs, bin_centers, bin_medians, bin_stds, bin_counts = (
        fit_aperture_correction(det_df)
    )
    print()

    # Plot AC curve
    plot_aperture_correction(det_df, poly, bin_centers, bin_medians, bin_stds)
    print()

    # Plot FWHM vs RA
    plot_fwhm_vs_ra(det_df)
    print()

    # ===================================================================
    # Step 4: Load solve infrastructure
    # ===================================================================
    print("Loading solve infrastructure...", flush=True)

    # Load node index from Phase 3 (post-outlier-rejection)
    sys.path.insert(0, str(BASE))
    from delve_ubercal.phase2_solve import build_node_index, load_des_fgcm_zps

    node_to_idx, idx_to_node = build_node_index(
        PHASE1_DIR / "connected_nodes.parquet"
    )
    n_params = len(node_to_idx)
    print(f"  Nodes: {n_params:,}", flush=True)

    # Load DES FGCM
    des_fgcm_zps = load_des_fgcm_zps(CACHE_DIR, BAND)
    print(f"  DES FGCM ZPs: {len(des_fgcm_zps):,}", flush=True)

    # Load node positions
    pos_df = pd.read_parquet(PHASE0_DIR / "node_positions.parquet")
    node_positions = dict(zip(
        zip(pos_df["expnum"].values, pos_df["ccdnum"].values),
        zip(pos_df["ra_mean"].values, pos_df["dec_mean"].values),
    ))
    print(f"  Node positions: {len(node_positions):,}", flush=True)

    # Load flagged nodes from Phase 3
    flagged_nodes_df = pd.read_parquet(PHASE3_DIR / "flagged_nodes.parquet")
    flagged_nodes = set(zip(flagged_nodes_df["expnum"].values,
                            flagged_nodes_df["ccdnum"].values))
    print(f"  Flagged nodes: {len(flagged_nodes):,}", flush=True)
    print()

    # Plot per-exposure AC impact
    plot_ac_impact_per_exposure(det_df, poly, node_to_idx, idx_to_node, node_positions)
    print()

    # ===================================================================
    # Step 5: Apply aperture correction
    # ===================================================================
    det_corr = build_corrected_star_lists(det_df, poly)
    print()

    # ===================================================================
    # Step 6: Build normal equations with ORIGINAL magnitudes
    # ===================================================================
    print("Building normal equations (ORIGINAL m_inst)...", flush=True)
    t0 = time.time()
    AtWA_orig, rhs_orig, n_stars_orig, n_pairs_orig = (
        accumulate_normal_equations_from_df(det_df, node_to_idx, n_params, m_col="m_inst")
    )
    t1 = time.time()
    print(f"  Stars: {n_stars_orig:,}, pairs: {n_pairs_orig:,}, "
          f"time: {t1-t0:.1f}s", flush=True)
    print()

    # ===================================================================
    # Step 7: Build normal equations with CORRECTED magnitudes
    # ===================================================================
    print("Building normal equations (CORRECTED m_inst_corr)...", flush=True)
    t0 = time.time()
    AtWA_corr, rhs_corr, n_stars_corr, n_pairs_corr = (
        accumulate_normal_equations_from_df(det_corr, node_to_idx, n_params, m_col="m_inst_corr")
    )
    t1 = time.time()
    print(f"  Stars: {n_stars_corr:,}, pairs: {n_pairs_corr:,}, "
          f"time: {t1-t0:.1f}s", flush=True)
    print()

    # ===================================================================
    # Step 8: Solve ORIGINAL
    # ===================================================================
    print("Solving ORIGINAL...", flush=True)
    zp_orig = solve_and_shift(AtWA_orig, rhs_orig, node_to_idx, des_fgcm_zps,
                               label="ORIGINAL")
    print()

    # ===================================================================
    # Step 9: Solve CORRECTED
    # ===================================================================
    print("Solving AC-CORRECTED...", flush=True)
    zp_corr = solve_and_shift(AtWA_corr, rhs_corr, node_to_idx, des_fgcm_zps,
                               label="AC-CORRECTED")
    print()

    # ===================================================================
    # Step 10: Compare RA gradients
    # ===================================================================
    print("Computing RA gradients...", flush=True)

    grad_orig = compute_ra_gradient(zp_orig, node_to_idx, idx_to_node,
                                     des_fgcm_zps, node_positions, flagged_nodes)
    grad_corr = compute_ra_gradient(zp_corr, node_to_idx, idx_to_node,
                                     des_fgcm_zps, node_positions, flagged_nodes)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Metric':<35} {'Original':>12} {'AC-corr':>12}")
    print(f"  {'-'*59}")
    print(f"  {'RA gradient (mmag/deg)':<35} {grad_orig['slope_mmag_per_deg']:>12.2f} {grad_corr['slope_mmag_per_deg']:>12.2f}")
    print(f"  {'Total gradient (mmag)':<35} {grad_orig['total_gradient_mmag']:>12.1f} {grad_corr['total_gradient_mmag']:>12.1f}")
    print(f"  {'DES diff RMS (mmag)':<35} {grad_orig['rms_mmag']:>12.1f} {grad_corr['rms_mmag']:>12.1f}")
    print(f"  {'DES nodes used':<35} {grad_orig['n_des_nodes']:>12,} {grad_corr['n_des_nodes']:>12,}")
    print()

    gradient_improvement = abs(grad_orig['total_gradient_mmag']) - abs(grad_corr['total_gradient_mmag'])
    rms_improvement = grad_orig['rms_mmag'] - grad_corr['rms_mmag']

    print(f"  Gradient improvement: {gradient_improvement:+.1f} mmag "
          f"({'BETTER' if gradient_improvement > 0 else 'WORSE'})")
    print(f"  RMS improvement: {rms_improvement:+.1f} mmag "
          f"({'BETTER' if rms_improvement > 0 else 'WORSE'})")
    print()

    # Also compare to Phase 3 existing solution
    # Must filter phantom nodes (zp_solved=0 from Tikhonov default)
    zp3 = pd.read_parquet(PHASE3_DIR / "zeropoints_unanchored.parquet")
    zp3_valid = zp3[zp3["zp_solved"] > 25.0]  # filter phantoms
    zp3_dict = dict(zip(
        zip(zp3_valid["expnum"].values, zp3_valid["ccdnum"].values),
        zp3_valid["zp_solved"].values,
    ))
    zp3_arr = np.full(n_params, np.nan)
    for node, idx in node_to_idx.items():
        if node in zp3_dict:
            zp3_arr[idx] = zp3_dict[node]

    # For Phase 3, add phantom nodes to flagged set
    p3_flagged = flagged_nodes.copy()
    for idx, node in enumerate(idx_to_node):
        if np.isnan(zp3_arr[idx]) or zp3_arr[idx] < 25.0:
            p3_flagged.add(node)
    # Replace NaN with 0 for array indexing (they'll be skipped via flagged_nodes)
    zp3_arr = np.nan_to_num(zp3_arr, nan=0.0)

    grad_p3 = compute_ra_gradient(zp3_arr, node_to_idx, idx_to_node,
                                   des_fgcm_zps, node_positions, p3_flagged)
    print(f"  Phase 3 reference: slope={grad_p3['slope_mmag_per_deg']:.2f} mmag/deg, "
          f"total={grad_p3['total_gradient_mmag']:.1f} mmag, "
          f"RMS={grad_p3['rms_mmag']:.1f} mmag")
    print(f"{'='*60}")
    print()

    # ===================================================================
    # Step 11: Plot comparison
    # ===================================================================
    plot_comparison(grad_orig, grad_corr)
    print()

    # Also plot 3-way comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(grad_p3["bin_centers"], grad_p3["bin_medians"] * 1000,
                yerr=grad_p3["bin_stds"] * 1000,
                fmt="^-", color="green", ms=5, capsize=3, label="Phase 3 (production)")
    ax.errorbar(grad_orig["bin_centers"], grad_orig["bin_medians"] * 1000,
                yerr=grad_orig["bin_stds"] * 1000,
                fmt="o-", color="blue", ms=5, capsize=3, label="This solve (original)")
    ax.errorbar(grad_corr["bin_centers"], grad_corr["bin_medians"] * 1000,
                yerr=grad_corr["bin_stds"] * 1000,
                fmt="s-", color="red", ms=5, capsize=3, label="This solve (AC-corrected)")

    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("RA (deg)", fontsize=12)
    ax.set_ylabel("ZP_solved - ZP_FGCM (mmag)", fontsize=12)
    ax.set_title("RA gradient comparison: Aperture correction test (g-band)", fontsize=13)
    ax.legend(fontsize=10)

    # Add text box
    txt = (f"Phase 3:    slope={grad_p3['slope_mmag_per_deg']:.2f} mmag/deg, "
           f"RMS={grad_p3['rms_mmag']:.1f} mmag\n"
           f"Original:   slope={grad_orig['slope_mmag_per_deg']:.2f} mmag/deg, "
           f"RMS={grad_orig['rms_mmag']:.1f} mmag\n"
           f"AC-corr:    slope={grad_corr['slope_mmag_per_deg']:.2f} mmag/deg, "
           f"RMS={grad_corr['rms_mmag']:.1f} mmag")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    outf = VAL_DIR / "aperture_correction_3way_comparison.png"
    fig.savefig(outf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outf}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("Done.")


if __name__ == "__main__":
    main()
