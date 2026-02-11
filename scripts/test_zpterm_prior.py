#!/usr/bin/env python3
"""Test whether regularizing toward zpterm (initial calibration) suppresses
the RA gradient in the unanchored ubercal solution.

For each lambda_prior value, adds a diagonal prior:
    AtWA[i,i] += lambda_prior
    rhs[i]    += lambda_prior * ZP_prior[i]

where ZP_prior = zpterm + MAGZERO_offset, pulling ZP_solved toward the
initial NSC DR2 calibration.

Metrics:
  - RA gradient (mmag/deg) from linear fit to ZP_solved - ZP_FGCM vs RA
  - DES diff RMS (mmag)
  - Bright-star repeatability floor (mmag)
"""

import sys
import time
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE = Path("/Volumes/External5TB/DELVE_UBERCAL")
OUTPUT = BASE / "output"
CACHE = BASE / "cache"
PHASE0_DIR = OUTPUT / "phase0_g"
PHASE1_DIR = OUTPUT / "phase1_g"
PHASE3_DIR = OUTPUT / "phase3_g"
VAL_DIR = OUTPUT / "validation_g"
BAND = "g"
MAGZERO_OFFSET = 31.4389

sys.path.insert(0, str(BASE))
from delve_ubercal.phase2_solve import (
    build_node_index,
    load_des_fgcm_zps,
    load_nsc_zpterms,
    accumulate_normal_equations,
)
from delve_ubercal.phase0_ingest import get_test_patch_pixels, load_config


def load_flagged_data():
    """Load Phase 3 flagging info."""
    flagged_nodes_df = pd.read_parquet(PHASE3_DIR / "flagged_nodes.parquet")
    flagged_nodes = set(zip(
        flagged_nodes_df["expnum"].values,
        flagged_nodes_df["ccdnum"].values
    ))

    flagged_stars_df = pd.read_parquet(PHASE3_DIR / "flagged_stars.parquet")
    flagged_stars = set(flagged_stars_df["objectid"].values)

    flagged_exposures_df = pd.read_parquet(PHASE3_DIR / "flagged_exposures.parquet")
    flagged_exposures = set(flagged_exposures_df["expnum"].values)

    # flagged_det_keys doesn't exist as a file -- skip
    return flagged_nodes, flagged_stars, flagged_exposures


def filter_star_list(df, flagged_nodes, flagged_stars, flagged_exposures):
    """Remove flagged stars, nodes, and exposures from a star list."""
    if len(df) == 0:
        return df

    # Remove flagged stars
    if flagged_stars:
        mask = ~df["objectid"].isin(flagged_stars)
        df = df[mask]

    # Remove flagged exposures
    if flagged_exposures:
        mask = ~df["expnum"].isin(flagged_exposures)
        df = df[mask]

    # Remove flagged nodes
    if flagged_nodes:
        node_keys = set(zip(df["expnum"].values, df["ccdnum"].values))
        bad = node_keys & flagged_nodes
        if bad:
            mask = ~pd.Series(
                list(zip(df["expnum"].values, df["ccdnum"].values)),
                index=df.index
            ).isin(bad)
            df = df[mask]

    return df


def accumulate_normal_equations_filtered(star_list_files, node_to_idx, n_params,
                                          flagged_nodes, flagged_stars,
                                          flagged_exposures):
    """Build normal equations, excluding flagged data."""
    rows = []
    cols = []
    vals = []
    rhs = np.zeros(n_params, dtype=np.float64)
    n_stars_used = 0
    n_pairs_total = 0

    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue

        df = filter_star_list(df, flagged_nodes, flagged_stars, flagged_exposures)
        if len(df) == 0:
            continue

        for star_id, group in df.groupby("objectid"):
            if len(group) < 2:
                continue

            mags = group["m_inst"].values
            errs = group["m_err"].values
            nodes = list(zip(group["expnum"].values, group["ccdnum"].values))

            indices = []
            star_mags = []
            star_errs = []
            for j, node in enumerate(nodes):
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

                    rows.append(ia); cols.append(ia); vals.append(w)
                    rows.append(ib); cols.append(ib); vals.append(w)
                    rows.append(ia); cols.append(ib); vals.append(-w)
                    rows.append(ib); cols.append(ia); vals.append(-w)
                    rhs[ia] -= w * dm
                    rhs[ib] += w * dm
                    n_pairs_total += 1

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = np.array(vals, dtype=np.float64)
    AtWA = sp.coo_matrix((vals, (rows, cols)), shape=(n_params, n_params))
    AtWA = AtWA.tocsr()

    return AtWA, rhs, n_stars_used, n_pairs_total


def compute_repeatability(star_list_files, zp_solved, node_to_idx,
                           flagged_nodes, flagged_stars, flagged_exposures):
    """Compute bright-star repeatability floor.

    For each star with >= 3 detections, compute scatter(m_inst + ZP_solved).
    Return median scatter for bright stars (mag < 18).
    """
    star_scatters = []
    star_mean_mags = []

    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        df = filter_star_list(df, flagged_nodes, flagged_stars, flagged_exposures)
        if len(df) == 0:
            continue

        for star_id, group in df.groupby("objectid"):
            if len(group) < 3:
                continue

            mags = group["m_inst"].values
            errs = group["m_err"].values
            nodes = list(zip(group["expnum"].values, group["ccdnum"].values))

            cal_mags = []
            for j, node in enumerate(nodes):
                idx = node_to_idx.get(node)
                if idx is not None:
                    cal_mags.append(mags[j] + zp_solved[idx])

            if len(cal_mags) < 3:
                continue

            cal_mags = np.array(cal_mags)
            scatter = np.std(cal_mags, ddof=1)
            mean_mag = np.mean(cal_mags)

            star_scatters.append(scatter)
            star_mean_mags.append(mean_mag)

    star_scatters = np.array(star_scatters)
    star_mean_mags = np.array(star_mean_mags)

    # Bright star floor: median scatter for stars with mean_mag < 18
    # (approximate instrumental + ZP ~ 18 + 31.4 ~ 49.4, but m_inst is ~18-19
    # for bright stars because zpterm was subtracted in Phase 0)
    # Actually m_inst + ZP_solved ~ 31.4 + 18 = ~49 mag? No.
    # m_inst ~ 18 (mag_aper4 - zpterm ~ 21 - 3.7 = 17.3-20)
    # m_inst + ZP_solved ~ 18 + 31.4 ~ 49.4? That's huge.
    # Actually let's just use magnitude bins on m_inst + ZP_solved
    # and find the floor (minimum scatter bin).

    # Use bins of calibrated mag and find the scatter floor
    if len(star_scatters) < 100:
        return np.nan, star_scatters, star_mean_mags

    # The calibrated magnitudes are m_inst + ZP_solved
    # m_inst ~ 17-20 (approximately calibrated), ZP_solved ~ 31.4
    # So cal_mag ~ 48-51? That seems wrong.
    # Actually: m_inst = mag_aper4 - zpterm. mag_aper4 ~ 17-20. zpterm ~ -3.7.
    # So m_inst ~ 17 - (-3.7) = 20.7, or m_inst = 17 + 3.7 = 20.7? No.
    # m_inst = mag_aper4 - zpterm_chip, where zpterm is negative (~-3.7).
    # So m_inst = 17 - (-3.7) = 20.7 for a mag 17 star.
    # ZP_solved ~ 31.4. So m_cal = m_inst + ZP_solved - MAGZERO_offset
    # = 20.7 + 31.4 - 31.4 = 20.7. Wait, that gives back ~mag_aper4.
    # OK, the scatter is what matters, not absolute mag.

    # Bin by mean_mag, find floor
    mag_bins = np.arange(np.percentile(star_mean_mags, 2),
                         np.percentile(star_mean_mags, 98), 0.5)
    bin_medians = []
    for i in range(len(mag_bins) - 1):
        mask = (star_mean_mags >= mag_bins[i]) & (star_mean_mags < mag_bins[i+1])
        if mask.sum() > 20:
            bin_medians.append(np.median(star_scatters[mask]))

    if bin_medians:
        floor = min(bin_medians)
    else:
        floor = np.median(star_scatters)

    return floor, star_scatters, star_mean_mags


def solve_with_prior(AtWA, rhs_base, n_params, zp_prior, lambda_prior,
                      node_to_idx, des_fgcm_zps, flagged_nodes,
                      tikhonov=1.0e-10, tol=1e-5, maxiter=5000):
    """Solve normal equations with zpterm prior.

    AtWA * zp + lambda_prior * (zp - zp_prior) = rhs
    => (AtWA + lambda_prior * I) * zp = rhs + lambda_prior * zp_prior
    """
    rhs = rhs_base.copy()

    # Add prior
    if lambda_prior > 0:
        # Add lambda to diagonal
        prior_diag = sp.eye(n_params, format="csr") * lambda_prior
        AtWA_reg = AtWA + prior_diag
        # Add prior target to RHS
        rhs += lambda_prior * zp_prior
    else:
        AtWA_reg = AtWA.copy()

    # Add Tikhonov
    AtWA_reg = AtWA_reg + sp.eye(n_params, format="csr") * tikhonov

    # Solve
    t0 = time.time()
    n_iter = [0]
    def callback(xk):
        n_iter[0] += 1

    zp_solved, cg_info = cg(
        AtWA_reg, rhs,
        rtol=tol,
        maxiter=maxiter,
        callback=callback,
    )
    solve_time = time.time() - t0

    # Shift to DES median (exclude flagged nodes and sentinel FGCM values)
    des_indices = []
    des_fgcm_vals = []
    for node, idx in node_to_idx.items():
        if node in flagged_nodes:
            continue
        if node in des_fgcm_zps:
            fgcm_val = des_fgcm_zps[node]
            if 25.0 < fgcm_val < 35.0:
                des_indices.append(idx)
                des_fgcm_vals.append(fgcm_val)

    if des_indices:
        des_indices = np.array(des_indices)
        des_fgcm_vals = np.array(des_fgcm_vals)
        offset = np.median(zp_solved[des_indices] - des_fgcm_vals)
        zp_solved -= offset

    return zp_solved, cg_info, n_iter[0], solve_time


def compute_ra_gradient(zp_solved, idx_to_node, des_fgcm_zps, node_positions,
                         flagged_nodes):
    """Compute RA gradient in mmag/deg from linear fit to DES residuals vs RA."""
    ras = []
    resids = []

    for idx, node in enumerate(idx_to_node):
        if node in flagged_nodes:
            continue
        if node not in des_fgcm_zps:
            continue
        fgcm_val = des_fgcm_zps[node]
        if not (25.0 < fgcm_val < 35.0):
            continue
        pos = node_positions.get(node)
        if pos is None:
            continue

        ra, dec = pos
        diff = zp_solved[idx] - fgcm_val
        ras.append(ra)
        resids.append(diff)

    ras = np.array(ras)
    resids = np.array(resids)

    if len(ras) < 10:
        return np.nan, np.nan, ras, resids

    # Linear fit: resid = a * ra + b
    # Center RA for numerical stability
    ra_mean = np.mean(ras)
    A = np.column_stack([ras - ra_mean, np.ones(len(ras))])

    # Sigma-clip
    mask = np.ones(len(ras), dtype=bool)
    for _ in range(3):
        if mask.sum() < 10:
            break
        coeffs = np.linalg.lstsq(A[mask], resids[mask], rcond=None)[0]
        model = A @ coeffs
        res = resids - model
        sigma = np.std(res[mask])
        mask = np.abs(res) < 3 * sigma

    coeffs = np.linalg.lstsq(A[mask], resids[mask], rcond=None)[0]
    slope_mmag_per_deg = coeffs[0] * 1000  # Convert to mmag/deg

    # RMS after removing gradient
    model = A @ coeffs
    rms_mmag = np.std((resids - model)[mask]) * 1000

    return slope_mmag_per_deg, rms_mmag, ras, resids


def compute_des_rms(zp_solved, idx_to_node, des_fgcm_zps, flagged_nodes):
    """Compute RMS of ZP_solved - ZP_FGCM for DES nodes."""
    diffs = []
    for idx, node in enumerate(idx_to_node):
        if node in flagged_nodes:
            continue
        if node not in des_fgcm_zps:
            continue
        fgcm_val = des_fgcm_zps[node]
        if not (25.0 < fgcm_val < 35.0):
            continue
        diffs.append(zp_solved[idx] - fgcm_val)

    diffs = np.array(diffs)
    if len(diffs) == 0:
        return np.nan
    return np.sqrt(np.mean(diffs**2)) * 1000  # mmag


def main():
    print("=" * 70)
    print("TEST: zpterm prior regularization to suppress RA gradient")
    print("=" * 70)
    print()

    config = load_config(None)

    # 1. Build node index
    print("Building node index...", flush=True)
    node_to_idx, idx_to_node = build_node_index(
        PHASE1_DIR / "connected_nodes.parquet"
    )
    n_params = len(node_to_idx)
    print(f"  Parameters: {n_params:,}", flush=True)

    # 2. Load reference data
    print("Loading DES FGCM zero-points...", flush=True)
    des_fgcm_zps = load_des_fgcm_zps(CACHE, BAND)

    print("Loading NSC zpterms...", flush=True)
    nsc_zpterms = load_nsc_zpterms(CACHE, BAND)

    print("Loading node positions...", flush=True)
    pos_df = pd.read_parquet(PHASE0_DIR / "node_positions.parquet")
    node_positions = dict(zip(
        zip(pos_df["expnum"].values, pos_df["ccdnum"].values),
        zip(pos_df["ra_mean"].values, pos_df["dec_mean"].values),
    ))

    # 3. Load flagging info
    print("Loading Phase 3 flagging info...", flush=True)
    flagged_nodes, flagged_stars, flagged_exposures = load_flagged_data()
    print(f"  Flagged nodes: {len(flagged_nodes):,}", flush=True)
    print(f"  Flagged stars: {len(flagged_stars):,}", flush=True)
    print(f"  Flagged exposures: {len(flagged_exposures):,}", flush=True)

    # 4. Build ZP_prior for each node: zpterm + MAGZERO_offset
    print("Building ZP_prior = zpterm + MAGZERO_offset...", flush=True)
    zp_prior = np.zeros(n_params, dtype=np.float64)
    n_with_prior = 0
    n_missing_prior = 0
    for node, idx in node_to_idx.items():
        zpterm = nsc_zpterms.get(node)
        if zpterm is not None:
            zp_prior[idx] = zpterm + MAGZERO_OFFSET
            n_with_prior += 1
        else:
            # For nodes without zpterm, use a reasonable default
            # Use the median ZP_prior from other nodes
            n_missing_prior += 1

    # Fill missing priors with median of available priors
    if n_missing_prior > 0:
        valid_priors = zp_prior[zp_prior != 0]
        if len(valid_priors) > 0:
            median_prior = np.median(valid_priors)
            zp_prior[zp_prior == 0] = median_prior
            print(f"  Filled {n_missing_prior} missing priors with median={median_prior:.4f}")

    print(f"  Nodes with zpterm: {n_with_prior:,} / {n_params:,}", flush=True)
    print(f"  ZP_prior median: {np.median(zp_prior):.4f} mag", flush=True)
    print(f"  ZP_prior std: {np.std(zp_prior):.4f} mag", flush=True)

    # 5. Collect star list files
    nside = config["survey"]["nside_chunk"]
    pixels = get_test_patch_pixels(nside)
    star_list_files = []
    for pixel in pixels:
        f = PHASE1_DIR / f"star_lists_nside{nside}_pixel{pixel}.parquet"
        if f.exists():
            star_list_files.append(f)
    print(f"  Star list files: {len(star_list_files)}", flush=True)

    # 6. Accumulate normal equations (with flagging)
    print("\nAccumulating normal equations (with flagging)...", flush=True)
    t0 = time.time()
    AtWA, rhs_base, n_stars, n_pairs = accumulate_normal_equations_filtered(
        star_list_files, node_to_idx, n_params,
        flagged_nodes, flagged_stars, flagged_exposures
    )
    accum_time = time.time() - t0
    print(f"  Stars: {n_stars:,}, Pairs: {n_pairs:,}, Time: {accum_time:.1f}s", flush=True)
    print(f"  Matrix NNZ: {AtWA.nnz:,}", flush=True)

    # 7. Solve for each lambda_prior value
    lambda_values = [0, 1, 10, 100, 300, 1000, 3000, 10000, 30000, 100000]
    results = {}

    print("\n" + "=" * 70)
    print(f"{'lambda':>10s} | {'RA grad':>10s} | {'DES RMS':>10s} | "
          f"{'Repeat':>10s} | {'Iters':>6s} | {'Time':>6s}")
    print(f"{'':>10s} | {'mmag/deg':>10s} | {'mmag':>10s} | "
          f"{'mmag':>10s} | {'':>6s} | {'s':>6s}")
    print("-" * 70)

    for lam in lambda_values:
        print(f"\nSolving with lambda_prior = {lam}...", flush=True)

        zp_solved, cg_info, n_iter, solve_time = solve_with_prior(
            AtWA, rhs_base, n_params, zp_prior, lam,
            node_to_idx, des_fgcm_zps, flagged_nodes
        )

        if cg_info != 0:
            print(f"  WARNING: CG did not converge (info={cg_info})", flush=True)

        # RA gradient
        ra_grad, ra_scatter, ras, resids = compute_ra_gradient(
            zp_solved, idx_to_node, des_fgcm_zps, node_positions, flagged_nodes
        )

        # DES RMS
        des_rms = compute_des_rms(zp_solved, idx_to_node, des_fgcm_zps, flagged_nodes)

        # Repeatability
        repeat_floor, star_scatters, star_mean_mags = compute_repeatability(
            star_list_files, zp_solved, node_to_idx,
            flagged_nodes, flagged_stars, flagged_exposures
        )

        results[lam] = {
            "zp_solved": zp_solved.copy(),
            "ra_gradient": ra_grad,
            "ra_scatter": ra_scatter,
            "des_rms": des_rms,
            "repeatability": repeat_floor * 1000,  # mmag
            "n_iter": n_iter,
            "solve_time": solve_time,
            "cg_info": cg_info,
            "ras": ras,
            "resids": resids,
        }

        print(f"  lambda={lam:>7d} | "
              f"RA grad={ra_grad:+.2f} mmag/deg | "
              f"DES RMS={des_rms:.1f} mmag | "
              f"Repeat={repeat_floor*1000:.1f} mmag | "
              f"Iters={n_iter} | "
              f"Time={solve_time:.1f}s", flush=True)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'lambda':>10s} | {'RA grad':>10s} | {'DES RMS':>10s} | "
          f"{'Repeat':>10s} | {'Iters':>6s}")
    print(f"{'':>10s} | {'mmag/deg':>10s} | {'mmag':>10s} | "
          f"{'mmag':>10s} | {'':>6s}")
    print("-" * 70)
    for lam in lambda_values:
        r = results[lam]
        print(f"{lam:>10d} | {r['ra_gradient']:>+10.2f} | "
              f"{r['des_rms']:>10.1f} | {r['repeatability']:>10.1f} | "
              f"{r['n_iter']:>6d}")
    print("=" * 70)

    # 8. Find BEST lambda: smallest gradient without degrading repeatability
    baseline_repeat = results[0]["repeatability"]
    repeat_threshold = max(baseline_repeat, 8.4)  # At most 8.4 mmag
    print(f"\nBaseline repeatability (lambda=0): {baseline_repeat:.1f} mmag")
    print(f"Repeatability threshold: {repeat_threshold:.1f} mmag")

    best_lam = 0
    best_grad = abs(results[0]["ra_gradient"])
    for lam in lambda_values:
        r = results[lam]
        if r["repeatability"] <= repeat_threshold * 1.05:  # Allow 5% margin
            if abs(r["ra_gradient"]) < best_grad:
                best_grad = abs(r["ra_gradient"])
                best_lam = lam

    print(f"Best lambda: {best_lam} (gradient={results[best_lam]['ra_gradient']:.2f} mmag/deg, "
          f"repeat={results[best_lam]['repeatability']:.1f} mmag)")

    # 9. Plot
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lam_arr = np.array(lambda_values, dtype=float)
    lam_labels = [str(l) for l in lambda_values]

    # Panel 1: RA gradient vs lambda
    ax = axes[0, 0]
    grads = [results[l]["ra_gradient"] for l in lambda_values]
    ax.plot(range(len(lambda_values)), grads, "o-", color="C0", markersize=8)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axhline(0.5, color="red", ls=":", alpha=0.7, label="+0.5 mmag/deg target")
    ax.axhline(-0.5, color="red", ls=":", alpha=0.7)
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels(lam_labels)
    ax.set_xlabel("lambda_prior")
    ax.set_ylabel("RA gradient (mmag/deg)")
    ax.set_title("RA Gradient vs lambda_prior")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: DES diff RMS vs lambda
    ax = axes[0, 1]
    rms_vals = [results[l]["des_rms"] for l in lambda_values]
    ax.plot(range(len(lambda_values)), rms_vals, "o-", color="C1", markersize=8)
    ax.axhline(results[0]["des_rms"], color="gray", ls="--", alpha=0.5,
               label=f"lambda=0: {results[0]['des_rms']:.1f} mmag")
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels(lam_labels)
    ax.set_xlabel("lambda_prior")
    ax.set_ylabel("DES diff RMS (mmag)")
    ax.set_title("DES Diff RMS vs lambda_prior")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Repeatability floor vs lambda
    ax = axes[1, 0]
    rep_vals = [results[l]["repeatability"] for l in lambda_values]
    ax.plot(range(len(lambda_values)), rep_vals, "o-", color="C2", markersize=8)
    ax.axhline(8.4, color="red", ls=":", alpha=0.7, label="8.4 mmag threshold")
    ax.axhline(results[0]["repeatability"], color="gray", ls="--", alpha=0.5,
               label=f"lambda=0: {results[0]['repeatability']:.1f} mmag")
    ax.set_xticks(range(len(lambda_values)))
    ax.set_xticklabels(lam_labels)
    ax.set_xlabel("lambda_prior")
    ax.set_ylabel("Repeatability floor (mmag)")
    ax.set_title("Repeatability Floor vs lambda_prior")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: ZP_solved - ZP_FGCM vs RA for best lambda
    ax = axes[1, 1]
    r_best = results[best_lam]
    r_base = results[0]

    # Plot baseline (lambda=0) in gray
    ax.scatter(r_base["ras"], r_base["resids"] * 1000, s=0.5, alpha=0.15,
               color="gray", label=f"lambda=0 (grad={r_base['ra_gradient']:.2f})")
    # Plot best in color
    ax.scatter(r_best["ras"], r_best["resids"] * 1000, s=0.5, alpha=0.3,
               color="C0", label=f"lambda={best_lam} (grad={r_best['ra_gradient']:.2f})")
    ax.axhline(0, color="black", ls="-", alpha=0.3)

    # Add linear fits
    for lam, color, ls in [(0, "gray", "--"), (best_lam, "C0", "-")]:
        r = results[lam]
        if len(r["ras"]) > 10:
            ra_mean = np.mean(r["ras"])
            A_fit = np.column_stack([r["ras"] - ra_mean, np.ones(len(r["ras"]))])
            coeffs = np.linalg.lstsq(A_fit, r["resids"], rcond=None)[0]
            ra_range = np.array([r["ras"].min(), r["ras"].max()])
            model = coeffs[0] * (ra_range - ra_mean) + coeffs[1]
            ax.plot(ra_range, model * 1000, color=color, ls=ls, lw=2)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("ZP_solved - ZP_FGCM (mmag)")
    ax.set_title(f"DES Residuals vs RA (best: lambda={best_lam})")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-200, 200)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"g-band: zpterm prior regularization test\n"
                 f"Best lambda={best_lam}: gradient={r_best['ra_gradient']:.2f} mmag/deg, "
                 f"repeat={r_best['repeatability']:.1f} mmag",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    outfile = VAL_DIR / "zpterm_prior_test.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {outfile}")

    return results


if __name__ == "__main__":
    results = main()
