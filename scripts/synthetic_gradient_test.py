#!/usr/bin/env python3
"""Synthetic gradient test using REAL overlap graph from g-band test patch.

Assigns known flat zero-points (ZP_true = 31.5 for all nodes) and synthetic
instrumental magnitudes, then runs the same CG solver to check whether
the solver/graph introduces a spurious gradient.

If gradient < 1 mmag/deg -> issue is in the DATA (systematics)
If gradient > 10 mmag/deg -> issue is in the SOLVER or GRAPH
"""

import sys
import time
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg

sys.path.insert(0, "/Volumes/External5TB/DELVE_UBERCAL")
from delve_ubercal.phase2_solve import (
    accumulate_normal_equations,
    compute_star_flat,
    accumulate_gray_normal_equations,
    solve_gray_unanchored,
    load_des_fgcm_zps,
    get_epoch_index,
    load_exposure_mjds,
)
from delve_ubercal.phase0_ingest import load_config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================
BASE = Path("/Volumes/External5TB/DELVE_UBERCAL")
OUTPUT = BASE / "output"
CACHE = BASE / "cache"
BAND = "g"
ZP_TRUE = 31.5  # Flat zero-point for ALL nodes

PHASE1_DIR = OUTPUT / f"phase1_{BAND}"
PHASE3_DIR = OUTPUT / f"phase3_{BAND}"
PHASE0_DIR = OUTPUT / f"phase0_{BAND}"
VAL_DIR = OUTPUT / f"validation_{BAND}"
VAL_DIR.mkdir(parents=True, exist_ok=True)

SYNTH_DIR = OUTPUT / f"synthetic_test_{BAND}"
SYNTH_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
    config = load_config(str(BASE / "delve_ubercal" / "config.yaml"))

    # =========================================================================
    # Step 1: Load the real overlap graph
    # =========================================================================
    print("=" * 70)
    print("SYNTHETIC GRADIENT TEST — g-band test patch")
    print("=" * 70)

    # Load Phase 3 zeropoints to get the node list (same nodes used in real solve)
    zp_df = pd.read_parquet(PHASE3_DIR / "zeropoints_unanchored.parquet")
    idx_to_node = list(zip(zp_df["expnum"].astype(int).values,
                           zp_df["ccdnum"].astype(int).values))
    node_to_idx = {node: i for i, node in enumerate(idx_to_node)}
    n_params = len(node_to_idx)
    print(f"\nNodes (CCD-exposures): {n_params:,}")

    # Load flagged nodes and exposures from Phase 3
    flagged_nodes_df = pd.read_parquet(PHASE3_DIR / "flagged_nodes.parquet")
    flagged_nodes = set(zip(flagged_nodes_df["expnum"].astype(int).values,
                            flagged_nodes_df["ccdnum"].astype(int).values))
    flagged_exp_df = pd.read_parquet(PHASE3_DIR / "flagged_exposures.parquet")
    flagged_exposures = set(flagged_exp_df["expnum"].astype(int).values)
    print(f"Flagged nodes: {len(flagged_nodes):,}")
    print(f"Flagged exposures: {len(flagged_exposures):,}")

    # Build set of valid nodes (not flagged)
    valid_nodes = set()
    for node in idx_to_node:
        expnum, ccdnum = node
        if node not in flagged_nodes and expnum not in flagged_exposures:
            valid_nodes.add(node)
    print(f"Valid (unflagged) nodes: {len(valid_nodes):,}")

    # Load DES FGCM for DES node identification
    des_fgcm_zps = load_des_fgcm_zps(CACHE, BAND)
    des_nodes_in_graph = set()
    for node in idx_to_node:
        if node in des_fgcm_zps:
            val = des_fgcm_zps[node]
            if 25.0 < val < 35.0:
                des_nodes_in_graph.add(node)
    print(f"DES nodes in graph: {len(des_nodes_in_graph):,}")

    # Load node positions
    pos_df = pd.read_parquet(PHASE0_DIR / "node_positions.parquet")
    node_positions = dict(zip(
        zip(pos_df["expnum"].astype(int).values,
            pos_df["ccdnum"].astype(int).values),
        zip(pos_df["ra_mean"].values, pos_df["dec_mean"].values),
    ))
    print(f"Node positions loaded: {len(node_positions):,}")

    # Load star list files (Phase 1, same as real pipeline)
    star_list_files = sorted(PHASE1_DIR.glob("star_lists_nside32_pixel*.parquet"))
    print(f"Star list files: {len(star_list_files)}")

    # =========================================================================
    # Step 2: Compute true stellar magnitudes and generate synthetic data
    # =========================================================================
    print(f"\nGenerating synthetic data (ZP_true = {ZP_TRUE} for all nodes)...")

    # First pass: compute m_true for each star
    # m_true = median(m_inst + ZP_true) for each star = median(m_inst) + ZP_TRUE
    # But we need per-star true magnitudes. For each star, set
    #   m_true = median of (m_inst_detection + ZP_current_for_that_detection)
    # Actually, simpler: just pick m_true = median(m_inst) + ZP_TRUE
    # Then synth m_inst = m_true - ZP_TRUE + noise = median(m_inst) + noise

    # We'll create synthetic star list files with m_synth replacing m_inst
    np.random.seed(42)  # Reproducibility
    n_stars_total = 0
    n_dets_total = 0
    synth_files = []

    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            synth_files.append(f)  # Keep empty files as-is
            continue

        # Filter to valid nodes
        node_keys = list(zip(df["expnum"].astype(int).values,
                             df["ccdnum"].astype(int).values))
        valid_mask = np.array([n in valid_nodes for n in node_keys])
        df = df[valid_mask].copy()

        if len(df) == 0:
            synth_f = SYNTH_DIR / f.name
            df.to_parquet(synth_f, index=False)
            synth_files.append(synth_f)
            continue

        # Compute m_true per star = median(m_inst) + ZP_TRUE
        star_medians = df.groupby("objectid")["m_inst"].transform("median")
        m_true = star_medians + ZP_TRUE

        # Synthetic m_inst = m_true - ZP_TRUE + noise
        # = star_medians + noise
        noise = np.random.normal(0, df["m_err"].values)
        df["m_inst"] = m_true - ZP_TRUE + noise

        synth_f = SYNTH_DIR / f.name
        df.to_parquet(synth_f, index=False)
        synth_files.append(synth_f)

        n_stars_this = df["objectid"].nunique()
        n_stars_total += n_stars_this
        n_dets_total += len(df)

    print(f"Synthetic stars: {n_stars_total:,}, detections: {n_dets_total:,}")
    print(f"Synthetic star list files: {len(synth_files)}")

    # =========================================================================
    # Step 3: Build normal equations using synthetic data
    # =========================================================================
    print("\nAccumulating normal equations from synthetic data...")
    t0 = time.time()
    AtWA, rhs, n_stars_used, n_pairs = accumulate_normal_equations(
        synth_files, node_to_idx, n_params
    )
    t_accum = time.time() - t0
    print(f"Stars: {n_stars_used:,}, pairs: {n_pairs:,}, time: {t_accum:.1f}s")
    print(f"Matrix NNZ: {AtWA.nnz:,}, size: {AtWA.data.nbytes / 1e6:.1f} MB")

    # =========================================================================
    # Step 4: Solve (full CCD-exposure solve, same as real pipeline)
    # =========================================================================
    print("\n--- Full CCD-exposure solve (unanchored) ---")
    tikhonov = config["solve"]["tikhonov_reg"]
    n = AtWA.shape[0]
    AtWA_reg = AtWA + sp.eye(n, format="csr") * tikhonov

    t0 = time.time()
    n_iter = [0]
    def callback_full(xk):
        n_iter[0] += 1

    zp_solved_full, cg_info = cg(
        AtWA_reg, rhs,
        rtol=float(config["solve"]["tolerance"]),
        maxiter=config["solve"]["max_iterations"],
        callback=callback_full,
    )
    t_solve = time.time() - t0
    residual = AtWA_reg @ zp_solved_full - rhs
    rel_res = np.linalg.norm(residual) / np.linalg.norm(rhs) if np.linalg.norm(rhs) > 0 else 0

    print(f"Converged: {cg_info == 0} (info={cg_info})")
    print(f"Iterations: {n_iter[0]}, time: {t_solve:.1f}s")
    print(f"Relative residual: {rel_res:.2e}")

    # Shift to DES median (match ZP_TRUE for DES nodes)
    des_indices = []
    des_true_vals = []
    for node, idx in node_to_idx.items():
        if node in des_nodes_in_graph and node in valid_nodes:
            des_indices.append(idx)
            des_true_vals.append(ZP_TRUE)
    des_indices = np.array(des_indices)
    des_true_vals = np.array(des_true_vals)
    offset = np.median(zp_solved_full[des_indices] - des_true_vals)
    zp_solved_full -= offset
    print(f"DES median offset applied: {offset*1000:.3f} mmag")

    # =========================================================================
    # Step 5: Analyze full solve results
    # =========================================================================
    print("\n--- Full solve: ZP_solved - ZP_true analysis ---")
    zp_err_full = zp_solved_full - ZP_TRUE  # Should be ~0 everywhere

    # Only analyze valid (unflagged) nodes with positions
    ra_arr = np.full(n_params, np.nan)
    dec_arr = np.full(n_params, np.nan)
    valid_mask = np.zeros(n_params, dtype=bool)
    for idx, node in enumerate(idx_to_node):
        if node in valid_nodes and node in node_positions:
            ra_arr[idx], dec_arr[idx] = node_positions[node]
            valid_mask[idx] = True

    valid_err = zp_err_full[valid_mask]
    valid_ra = ra_arr[valid_mask]
    valid_dec = dec_arr[valid_mask]
    print(f"Valid nodes with positions: {valid_mask.sum():,}")
    print(f"ZP error (solved - true): median = {np.median(valid_err)*1000:.3f} mmag")
    print(f"ZP error RMS: {np.sqrt(np.mean(valid_err**2))*1000:.3f} mmag")
    print(f"ZP error std: {np.std(valid_err)*1000:.3f} mmag")
    print(f"ZP error range: [{np.min(valid_err)*1000:.2f}, {np.max(valid_err)*1000:.2f}] mmag")

    # Fit RA gradient
    A_ra = np.column_stack([np.ones(len(valid_ra)), valid_ra])
    coeffs_ra, _, _, _ = np.linalg.lstsq(A_ra, valid_err, rcond=None)
    ra_slope_full = coeffs_ra[1] * 1000  # mmag/deg
    print(f"RA gradient: {ra_slope_full:.4f} mmag/deg")

    # Fit Dec gradient
    A_dec = np.column_stack([np.ones(len(valid_dec)), valid_dec])
    coeffs_dec, _, _, _ = np.linalg.lstsq(A_dec, valid_err, rcond=None)
    dec_slope_full = coeffs_dec[1] * 1000
    print(f"Dec gradient: {dec_slope_full:.4f} mmag/deg")

    # Fit 2D plane
    A_2d = np.column_stack([np.ones(len(valid_ra)), valid_ra, valid_dec])
    coeffs_2d, _, _, _ = np.linalg.lstsq(A_2d, valid_err, rcond=None)
    ra_slope_2d = coeffs_2d[1] * 1000
    dec_slope_2d = coeffs_2d[2] * 1000
    print(f"2D plane: RA slope = {ra_slope_2d:.4f}, Dec slope = {dec_slope_2d:.4f} mmag/deg")

    # DES-only analysis
    des_mask_valid = np.zeros(n_params, dtype=bool)
    for idx, node in enumerate(idx_to_node):
        if node in des_nodes_in_graph and node in valid_nodes and node in node_positions:
            des_mask_valid[idx] = True

    if des_mask_valid.sum() > 0:
        des_err = zp_err_full[des_mask_valid]
        des_ra = ra_arr[des_mask_valid]
        des_dec = dec_arr[des_mask_valid]
        print(f"\nDES nodes only ({des_mask_valid.sum():,}):")
        print(f"  ZP error RMS: {np.sqrt(np.mean(des_err**2))*1000:.3f} mmag")
        A_des = np.column_stack([np.ones(len(des_ra)), des_ra, des_dec])
        c_des, _, _, _ = np.linalg.lstsq(A_des, des_err, rcond=None)
        print(f"  RA slope: {c_des[1]*1000:.4f} mmag/deg")
        print(f"  Dec slope: {c_des[2]*1000:.4f} mmag/deg")

    # Non-DES analysis
    nondes_mask = np.zeros(n_params, dtype=bool)
    for idx, node in enumerate(idx_to_node):
        if node not in des_nodes_in_graph and node in valid_nodes and node in node_positions:
            nondes_mask[idx] = True

    if nondes_mask.sum() > 0:
        nondes_err = zp_err_full[nondes_mask]
        nondes_ra = ra_arr[nondes_mask]
        nondes_dec = dec_arr[nondes_mask]
        print(f"\nNon-DES nodes only ({nondes_mask.sum():,}):")
        print(f"  ZP error RMS: {np.sqrt(np.mean(nondes_err**2))*1000:.3f} mmag")
        A_nd = np.column_stack([np.ones(len(nondes_ra)), nondes_ra, nondes_dec])
        c_nd, _, _, _ = np.linalg.lstsq(A_nd, nondes_err, rcond=None)
        print(f"  RA slope: {c_nd[1]*1000:.4f} mmag/deg")
        print(f"  Dec slope: {c_nd[2]*1000:.4f} mmag/deg")

    # =========================================================================
    # Step 6: Gray solve (star flat decomposition)
    # =========================================================================
    print("\n" + "=" * 70)
    print("GRAY SOLVE (star flat decomposition)")
    print("=" * 70)

    # Load exposure MJDs
    exposure_mjds = load_exposure_mjds(CACHE, BAND)
    print(f"Exposure MJDs loaded: {len(exposure_mjds):,}")

    # Compute star flat from full synthetic solve
    print("\nComputing star flat from synthetic full solve...")
    star_flat, epoch_boundaries = compute_star_flat(
        zp_solved_full, idx_to_node, exposure_mjds, config,
        flagged_nodes=flagged_nodes, flagged_exposures=flagged_exposures,
    )

    # Build exposure index for gray solve
    exp_set = set()
    for expnum, ccdnum in idx_to_node:
        if (expnum, ccdnum) in valid_nodes:
            exp_set.add(expnum)
    exp_list = sorted(exp_set)
    exp_to_idx = {exp: i for i, exp in enumerate(exp_list)}
    n_exp_params = len(exp_to_idx)
    print(f"\nExposure parameters (gray): {n_exp_params:,}")

    # Accumulate gray normal equations
    print("Accumulating gray normal equations...")
    t0 = time.time()
    AtWA_gray, rhs_gray, n_stars_gray, n_pairs_gray, n_intra = \
        accumulate_gray_normal_equations(
            synth_files, exp_to_idx, n_exp_params,
            star_flat, exposure_mjds, epoch_boundaries,
        )
    t_gray_accum = time.time() - t0
    print(f"Gray: {n_stars_gray:,} stars, {n_pairs_gray:,} pairs, "
          f"{n_intra:,} intra-exp skipped, time: {t_gray_accum:.1f}s")

    # Solve gray
    print("\nSolving gray (unanchored)...")
    zp_gray_full, gray_solved, gray_info = solve_gray_unanchored(
        AtWA_gray, rhs_gray, config, exp_to_idx, des_fgcm_zps,
        star_flat, exposure_mjds, epoch_boundaries,
        node_to_idx, idx_to_node,
    )
    print(f"Gray converged: {gray_info['converged']} (info={gray_info['cg_info']})")
    print(f"Gray iterations: {gray_info['n_iterations']}")
    print(f"Gray solve time: {gray_info['solve_time_s']:.1f}s")

    # Shift gray ZP to match ZP_TRUE for DES nodes
    gray_des_diffs = []
    for idx, node in enumerate(idx_to_node):
        if node in des_nodes_in_graph and node in valid_nodes:
            gray_des_diffs.append(zp_gray_full[idx] - ZP_TRUE)
    if gray_des_diffs:
        gray_offset = np.median(gray_des_diffs)
        zp_gray_full -= gray_offset
        print(f"Gray DES offset: {gray_offset*1000:.3f} mmag")

    # Analyze gray solve
    gray_err = zp_gray_full - ZP_TRUE

    gray_valid_err = gray_err[valid_mask]
    print(f"\nGray ZP error (solved - true):")
    print(f"  Median: {np.median(gray_valid_err)*1000:.3f} mmag")
    print(f"  RMS: {np.sqrt(np.mean(gray_valid_err**2))*1000:.3f} mmag")
    print(f"  Std: {np.std(gray_valid_err)*1000:.3f} mmag")

    # Gray RA gradient
    A_ra_g = np.column_stack([np.ones(valid_mask.sum()), valid_ra])
    c_ra_g, _, _, _ = np.linalg.lstsq(A_ra_g, gray_valid_err, rcond=None)
    gray_ra_slope = c_ra_g[1] * 1000
    print(f"  RA gradient: {gray_ra_slope:.4f} mmag/deg")

    # Gray Dec gradient
    A_dec_g = np.column_stack([np.ones(valid_mask.sum()), valid_dec])
    c_dec_g, _, _, _ = np.linalg.lstsq(A_dec_g, gray_valid_err, rcond=None)
    gray_dec_slope = c_dec_g[1] * 1000
    print(f"  Dec gradient: {gray_dec_slope:.4f} mmag/deg")

    # Gray 2D plane
    A_2d_g = np.column_stack([np.ones(valid_mask.sum()), valid_ra, valid_dec])
    c_2d_g, _, _, _ = np.linalg.lstsq(A_2d_g, gray_valid_err, rcond=None)
    print(f"  2D plane: RA = {c_2d_g[1]*1000:.4f}, Dec = {c_2d_g[2]*1000:.4f} mmag/deg")

    # =========================================================================
    # Step 7: Binned analysis (RA and Dec bins)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BINNED ANALYSIS")
    print("=" * 70)

    n_ra_bins = 10
    ra_edges = np.linspace(valid_ra.min(), valid_ra.max(), n_ra_bins + 1)
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    full_binned_ra = np.zeros(n_ra_bins)
    gray_binned_ra = np.zeros(n_ra_bins)
    counts_ra = np.zeros(n_ra_bins, dtype=int)

    for i in range(n_ra_bins):
        mask = (valid_ra >= ra_edges[i]) & (valid_ra < ra_edges[i + 1])
        if mask.sum() > 0:
            full_binned_ra[i] = np.median(valid_err[mask]) * 1000
            gray_binned_ra[i] = np.median(gray_valid_err[mask]) * 1000
            counts_ra[i] = mask.sum()

    print("\nRA bins (median ZP error in mmag):")
    print(f"{'RA center':>10} {'N nodes':>8} {'Full (mmag)':>12} {'Gray (mmag)':>12}")
    for i in range(n_ra_bins):
        print(f"{ra_centers[i]:10.2f} {counts_ra[i]:8d} {full_binned_ra[i]:12.4f} {gray_binned_ra[i]:12.4f}")

    n_dec_bins = 10
    dec_edges = np.linspace(valid_dec.min(), valid_dec.max(), n_dec_bins + 1)
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    full_binned_dec = np.zeros(n_dec_bins)
    gray_binned_dec = np.zeros(n_dec_bins)
    counts_dec = np.zeros(n_dec_bins, dtype=int)

    for i in range(n_dec_bins):
        mask = (valid_dec >= dec_edges[i]) & (valid_dec < dec_edges[i + 1])
        if mask.sum() > 0:
            full_binned_dec[i] = np.median(valid_err[mask]) * 1000
            gray_binned_dec[i] = np.median(gray_valid_err[mask]) * 1000
            counts_dec[i] = mask.sum()

    print("\nDec bins (median ZP error in mmag):")
    print(f"{'Dec center':>10} {'N nodes':>8} {'Full (mmag)':>12} {'Gray (mmag)':>12}")
    for i in range(n_dec_bins):
        print(f"{dec_centers[i]:10.2f} {counts_dec[i]:8d} {full_binned_dec[i]:12.4f} {gray_binned_dec[i]:12.4f}")

    # =========================================================================
    # Step 8: Diagnostic plot
    # =========================================================================
    print("\nGenerating diagnostic plot...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Synthetic Gradient Test: g-band test patch\n"
                 f"ZP_true = {ZP_TRUE} (flat, no gradient) for all {n_params:,} nodes",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Full solve ZP error vs RA ---
    ax = axes[0, 0]
    # Subsample for scatter plot
    if len(valid_ra) > 10000:
        sub_idx = np.random.choice(len(valid_ra), 10000, replace=False)
    else:
        sub_idx = np.arange(len(valid_ra))
    ax.scatter(valid_ra[sub_idx], valid_err[sub_idx] * 1000, s=1, alpha=0.3, c="gray")
    ax.plot(ra_centers, full_binned_ra, "ro-", linewidth=2, markersize=6, label="Binned median")
    # Fit line
    ra_fit_line = (coeffs_ra[0] + coeffs_ra[1] * ra_centers) * 1000
    ax.plot(ra_centers, ra_fit_line, "b--", linewidth=2,
            label=f"Fit: {ra_slope_full:.3f} mmag/deg")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("ZP_solved - ZP_true (mmag)")
    ax.set_title("Full solve: ZP error vs RA")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(-5, 5)

    # --- Panel 2: Full solve ZP error vs Dec ---
    ax = axes[0, 1]
    ax.scatter(valid_dec[sub_idx], valid_err[sub_idx] * 1000, s=1, alpha=0.3, c="gray")
    ax.plot(dec_centers, full_binned_dec, "ro-", linewidth=2, markersize=6, label="Binned median")
    dec_fit_line = (coeffs_dec[0] + coeffs_dec[1] * dec_centers) * 1000
    ax.plot(dec_centers, dec_fit_line, "b--", linewidth=2,
            label=f"Fit: {dec_slope_full:.3f} mmag/deg")
    ax.set_xlabel("Dec (deg)")
    ax.set_ylabel("ZP_solved - ZP_true (mmag)")
    ax.set_title("Full solve: ZP error vs Dec")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(-5, 5)

    # --- Panel 3: Gray solve ZP error vs RA ---
    ax = axes[1, 0]
    ax.scatter(valid_ra[sub_idx], gray_valid_err[sub_idx] * 1000, s=1, alpha=0.3, c="gray")
    ax.plot(ra_centers, gray_binned_ra, "go-", linewidth=2, markersize=6, label="Binned median")
    ra_fit_gray = (c_ra_g[0] + c_ra_g[1] * ra_centers) * 1000
    ax.plot(ra_centers, ra_fit_gray, "b--", linewidth=2,
            label=f"Fit: {gray_ra_slope:.3f} mmag/deg")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("ZP_solved - ZP_true (mmag)")
    ax.set_title("Gray solve: ZP error vs RA")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(-5, 5)

    # --- Panel 4: Gray solve ZP error vs Dec ---
    ax = axes[1, 1]
    ax.scatter(valid_dec[sub_idx], gray_valid_err[sub_idx] * 1000, s=1, alpha=0.3, c="gray")
    ax.plot(dec_centers, gray_binned_dec, "go-", linewidth=2, markersize=6, label="Binned median")
    dec_fit_gray = (c_dec_g[0] + c_dec_g[1] * dec_centers) * 1000
    ax.plot(dec_centers, dec_fit_gray, "b--", linewidth=2,
            label=f"Fit: {gray_dec_slope:.3f} mmag/deg")
    ax.set_xlabel("Dec (deg)")
    ax.set_ylabel("ZP_solved - ZP_true (mmag)")
    ax.set_title("Gray solve: ZP error vs Dec")
    ax.legend(loc="upper right")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_ylim(-5, 5)

    # --- Panel 5: Histogram of ZP errors ---
    ax = axes[2, 0]
    bins_hist = np.linspace(-3, 3, 101)
    ax.hist(valid_err * 1000, bins=bins_hist, alpha=0.7, color="steelblue",
            label=f"Full solve (RMS={np.sqrt(np.mean(valid_err**2))*1000:.3f})")
    ax.hist(gray_valid_err * 1000, bins=bins_hist, alpha=0.5, color="forestgreen",
            label=f"Gray solve (RMS={np.sqrt(np.mean(gray_valid_err**2))*1000:.3f})")
    ax.set_xlabel("ZP_solved - ZP_true (mmag)")
    ax.set_ylabel("N nodes")
    ax.set_title("Distribution of ZP errors")
    ax.legend()
    ax.axvline(0, color="k", linewidth=0.5)

    # --- Panel 6: 2D sky map of ZP error (full solve) ---
    ax = axes[2, 1]
    vmin, vmax = -2, 2
    sc = ax.scatter(valid_ra[sub_idx], valid_dec[sub_idx],
                    c=valid_err[sub_idx] * 1000, s=2, alpha=0.5,
                    cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="ZP error (mmag)")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("Full solve: ZP error sky map")
    ax.set_aspect("equal")

    plt.tight_layout()
    plot_path = VAL_DIR / "synthetic_gradient_test.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {plot_path}")

    # =========================================================================
    # Step 9: Summary verdict
    # =========================================================================
    t_total = time.time() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFull solve:")
    print(f"  RMS(ZP_solved - ZP_true) = {np.sqrt(np.mean(valid_err**2))*1000:.3f} mmag")
    print(f"  RA gradient  = {ra_slope_full:.4f} mmag/deg")
    print(f"  Dec gradient = {dec_slope_full:.4f} mmag/deg")
    print(f"\nGray solve:")
    print(f"  RMS(ZP_solved - ZP_true) = {np.sqrt(np.mean(gray_valid_err**2))*1000:.3f} mmag")
    print(f"  RA gradient  = {gray_ra_slope:.4f} mmag/deg")
    print(f"  Dec gradient = {gray_dec_slope:.4f} mmag/deg")

    max_gradient = max(abs(ra_slope_full), abs(dec_slope_full),
                       abs(gray_ra_slope), abs(gray_dec_slope))
    print(f"\nMax gradient across all tests: {max_gradient:.4f} mmag/deg")

    if max_gradient < 1.0:
        print("\n>>> VERDICT: Gradient < 1 mmag/deg")
        print(">>> The solver and graph are CLEAN.")
        print(">>> Any gradient in the real data comes from DATA SYSTEMATICS")
        print(">>> (aperture corrections, star flat residuals, Refcat2, etc.)")
    elif max_gradient < 10.0:
        print(f"\n>>> VERDICT: Gradient = {max_gradient:.2f} mmag/deg (between 1-10)")
        print(">>> Marginal — some solver/graph contribution, but small")
    else:
        print(f"\n>>> VERDICT: Gradient = {max_gradient:.2f} mmag/deg (> 10)")
        print(">>> The SOLVER or GRAPH is introducing a spurious gradient!")

    print(f"\nTotal runtime: {t_total:.1f}s")

    # Clean up synthetic files
    print("\nCleaning up synthetic star list files...")
    for f in SYNTH_DIR.glob("*.parquet"):
        f.unlink()
    SYNTH_DIR.rmdir()
    print("Done.")


if __name__ == "__main__":
    main()
