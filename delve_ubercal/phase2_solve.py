"""Phase 2: CG sparse solver (unanchored + anchored modes).

Builds the normal equations matrix A^T W A by streaming through per-star
detection lists, then solves with conjugate gradient for per-CCD-per-exposure
zero-points.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from delve_ubercal.phase0_ingest import get_test_region_pixels, load_config
from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels


def build_node_index(connected_file):
    """Assign each connected (expnum, ccdnum) a unique integer index.

    Returns
    -------
    node_to_idx : dict
        Maps (expnum, ccdnum) -> integer index.
    idx_to_node : list
        Maps index -> (expnum, ccdnum).
    """
    connected = pd.read_parquet(connected_file)
    nodes = list(zip(connected["expnum"].values, connected["ccdnum"].values))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    return node_to_idx, nodes


def load_des_fgcm_zps(cache_dir, band):
    """Load DES FGCM zero-points as a dict: (expnum, ccdnum) -> mag_zero."""
    fgcm = pd.read_parquet(
        cache_dir / "des_y6_fgcm_zeropoints.parquet",
        columns=["expnum", "ccdnum", "band", "mag_zero"],
    )
    fgcm = fgcm[fgcm["band"] == band]
    return dict(zip(
        zip(fgcm["expnum"].values, fgcm["ccdnum"].values),
        fgcm["mag_zero"].values,
    ))


def load_nsc_zpterms(cache_dir, band):
    """Load NSC zpterm values as a dict: (expnum, ccdnum) -> zpterm.

    Joins the chip table (keyed by exposure string) with the exposure table
    (mapping exposure string to expnum integer).
    """
    chip = pd.read_parquet(cache_dir / f"nsc_chip_{band}.parquet")
    exp = pd.read_parquet(cache_dir / f"nsc_exposure_{band}.parquet",
                          columns=["exposure", "expnum"])
    merged = chip.merge(exp, on="exposure", how="inner")
    return dict(zip(
        zip(merged["expnum"].values, merged["ccdnum"].values),
        merged["zpterm"].values,
    ))


def accumulate_normal_equations(star_list_files, node_to_idx, n_params):
    """Build normal equations A^T W A and A^T W dm by streaming stars.

    For each star with detections on CCD-exposures i1..in with mags m1..mn
    and errors e1..en, for each pair (ia, ib):
        w = 1 / (ea^2 + eb^2)
        A^T_W_A[ia,ia] += w;  A^T_W_A[ib,ib] += w
        A^T_W_A[ia,ib] -= w;  A^T_W_A[ib,ia] -= w
        rhs[ia] += w * (ma - mb);  rhs[ib] += w * (mb - ma)

    Parameters
    ----------
    star_list_files : list of Path
        Per-pixel star detection list parquet files.
    node_to_idx : dict
        Maps (expnum, ccdnum) -> integer index.
    n_params : int
        Number of zero-point parameters.

    Returns
    -------
    AtWA : scipy.sparse.csr_matrix
        Normal equations matrix (N x N).
    rhs : np.ndarray
        Right-hand side vector (length N).
    n_stars_used : int
        Number of stars that contributed to the system.
    n_pairs_total : int
        Total number of pairs accumulated.
    """
    # Use dictionaries for accumulation then build COO
    # For memory efficiency, accumulate into arrays
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

        for star_id, group in df.groupby("objectid"):
            if len(group) < 2:
                continue

            # Map to indices, filtering to connected nodes
            mags = group["m_inst"].values
            errs = group["m_err"].values
            nodes = list(zip(group["expnum"].values, group["ccdnum"].values))

            # Filter to connected nodes and get indices
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

            # Accumulate pairs
            for a in range(n_det):
                for b in range(a + 1, n_det):
                    ia, ib = indices[a], indices[b]
                    ma, mb = star_mags[a], star_mags[b]
                    ea, eb = star_errs[a], star_errs[b]

                    w = 1.0 / (ea * ea + eb * eb)
                    dm = ma - mb

                    # Diagonal
                    rows.append(ia); cols.append(ia); vals.append(w)
                    rows.append(ib); cols.append(ib); vals.append(w)
                    # Off-diagonal
                    rows.append(ia); cols.append(ib); vals.append(-w)
                    rows.append(ib); cols.append(ia); vals.append(-w)

                    # RHS: normal equations give A^T W A * ZP = -A^T W * dm
                    # where dm = ma - mb and A_row = [+1, -1]
                    rhs[ia] -= w * dm
                    rhs[ib] += w * dm

                    n_pairs_total += 1

    # Build sparse matrix
    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = np.array(vals, dtype=np.float64)
    AtWA = sp.coo_matrix((vals, (rows, cols)), shape=(n_params, n_params))
    AtWA = AtWA.tocsr()

    return AtWA, rhs, n_stars_used, n_pairs_total


def solve_unanchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps):
    """Solve in unanchored mode with Tikhonov regularization.

    After solving, shifts the solution so that
    mean(ZP_solved[DES]) = mean(ZP_FGCM[DES]).

    Returns
    -------
    zp_solved : np.ndarray
        Solved zero-points.
    info : dict
        Solver diagnostics.
    """
    solve_cfg = config["solve"]
    n = AtWA.shape[0]

    # Add Tikhonov regularization
    tikhonov = solve_cfg["tikhonov_reg"]
    AtWA_reg = AtWA + sp.eye(n, format="csr") * tikhonov

    # Solve
    t0 = time.time()
    n_iter = [0]

    def callback(xk):
        n_iter[0] += 1

    zp_solved, cg_info = cg(
        AtWA_reg, rhs,
        rtol=float(solve_cfg["tolerance"]),
        maxiter=solve_cfg["max_iterations"],
        callback=callback,
    )
    solve_time = time.time() - t0

    # Compute residual
    residual = AtWA_reg @ zp_solved - rhs
    rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs) if np.linalg.norm(rhs) > 0 else 0

    # Shift to DES mean
    des_indices = []
    des_fgcm_vals = []
    for node, idx in node_to_idx.items():
        if node in des_fgcm_zps:
            des_indices.append(idx)
            des_fgcm_vals.append(des_fgcm_zps[node])

    if des_indices:
        des_indices = np.array(des_indices)
        des_fgcm_vals = np.array(des_fgcm_vals)
        offset = np.mean(zp_solved[des_indices]) - np.mean(des_fgcm_vals)
        zp_solved -= offset
    else:
        offset = 0.0

    info = {
        "mode": "unanchored",
        "n_iterations": n_iter[0],
        "cg_info": cg_info,  # 0 = converged
        "relative_residual": rel_residual,
        "solve_time_s": solve_time,
        "des_offset_applied": offset,
        "converged": cg_info == 0,
    }
    return zp_solved, info


def solve_anchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps):
    """Solve in anchored mode with DES FGCM penalty terms.

    Returns
    -------
    zp_solved : np.ndarray
        Solved zero-points.
    info : dict
        Solver diagnostics.
    """
    solve_cfg = config["solve"]
    n = AtWA.shape[0]
    anchor_weight = float(solve_cfg["des_anchor_weight"])

    # Copy matrix and RHS to add anchoring terms
    AtWA_anch = AtWA.copy()
    rhs_anch = rhs.copy()

    # Add DES anchoring: for each DES CCD-exposure, add penalty
    n_anchored = 0
    for node, idx in node_to_idx.items():
        if node in des_fgcm_zps:
            zp_fgcm = des_fgcm_zps[node]
            # Add to diagonal
            AtWA_anch[idx, idx] += anchor_weight
            # Add to RHS
            rhs_anch[idx] += anchor_weight * zp_fgcm
            n_anchored += 1

    AtWA_anch = AtWA_anch.tocsr()

    # Solve
    t0 = time.time()
    n_iter = [0]

    def callback(xk):
        n_iter[0] += 1

    zp_solved, cg_info = cg(
        AtWA_anch, rhs_anch,
        rtol=float(solve_cfg["tolerance"]),
        maxiter=solve_cfg["max_iterations"],
        callback=callback,
    )
    solve_time = time.time() - t0

    # Compute residual
    residual = AtWA_anch @ zp_solved - rhs_anch
    rel_residual = np.linalg.norm(residual) / np.linalg.norm(rhs_anch) if np.linalg.norm(rhs_anch) > 0 else 0

    info = {
        "mode": "anchored",
        "n_iterations": n_iter[0],
        "cg_info": cg_info,
        "relative_residual": rel_residual,
        "solve_time_s": solve_time,
        "n_anchored": n_anchored,
        "anchor_weight": anchor_weight,
        "converged": cg_info == 0,
    }
    return zp_solved, info


def compute_diagnostics(zp_solved, node_to_idx, idx_to_node, des_fgcm_zps,
                        nsc_zpterms, band, mode):
    """Compute and print solve diagnostics.

    Returns
    -------
    result_df : pd.DataFrame
        Table with (expnum, ccdnum, band, zp_solved, delta_zp).
    diag : dict
        Diagnostic statistics.
    """
    n = len(zp_solved)

    # Build result table
    # delta_zp = ZP_solved - ZP_FGCM for DES nodes (meaningful comparison)
    # delta_zp = NaN for non-DES nodes (no existing per-CCD ZP to compare)
    expnums = [node[0] for node in idx_to_node]
    ccdnums = [node[1] for node in idx_to_node]
    zp_ref = np.array([
        des_fgcm_zps.get(node, np.nan) for node in idx_to_node
    ])
    delta_zp = zp_solved - zp_ref  # NaN for non-DES

    result_df = pd.DataFrame({
        "expnum": expnums,
        "ccdnum": ccdnums,
        "band": band,
        "zp_solved": zp_solved,
        "zp_fgcm": zp_ref,
        "delta_zp": delta_zp,
    })

    # DES-specific diagnostics
    des_mask = np.array([node in des_fgcm_zps for node in idx_to_node])
    if des_mask.any():
        des_diff = zp_solved[des_mask] - zp_ref[des_mask]
        des_diff_rms = np.sqrt(np.mean(des_diff ** 2))
        des_diff_median = np.median(des_diff)
        des_diff_max = np.max(np.abs(des_diff))
    else:
        des_diff_rms = np.nan
        des_diff_median = np.nan
        des_diff_max = np.nan

    # ZP_solved statistics (overall spread, not delta)
    zp_median = float(np.median(zp_solved))
    zp_std = float(np.std(zp_solved))

    diag = {
        "n_params": n,
        "n_des": int(des_mask.sum()),
        "zp_median": zp_median,
        "zp_std": zp_std,
        "des_diff_rms": float(des_diff_rms),
        "des_diff_median": float(des_diff_median),
        "des_diff_max": float(des_diff_max),
    }
    return result_df, diag


def run_solve(band, pixels, config, output_dir, cache_dir, mode="both"):
    """Run the full Phase 2 solve pipeline.

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
    mode : str
        'unanchored', 'anchored', or 'both'.

    Returns
    -------
    results : dict
        Results for each mode.
    """
    nside = config["survey"]["nside_chunk"]
    phase1_dir = output_dir / f"phase1_{band}"
    phase2_dir = output_dir / f"phase2_{band}"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build node index
    print(f"  Building node index...", flush=True)
    node_to_idx, idx_to_node = build_node_index(
        phase1_dir / "connected_nodes.parquet"
    )
    n_params = len(node_to_idx)
    print(f"  Parameters: {n_params:,}", flush=True)

    # 2. Load reference data
    print(f"  Loading DES FGCM zero-points...", flush=True)
    des_fgcm_zps = load_des_fgcm_zps(cache_dir, band)

    print(f"  Loading NSC zpterms...", flush=True)
    nsc_zpterms = load_nsc_zpterms(cache_dir, band)

    # 3. Collect star list files for the given pixels
    star_list_files = []
    for pixel in pixels:
        f = phase1_dir / f"star_lists_nside{nside}_pixel{pixel}.parquet"
        if f.exists():
            star_list_files.append(f)
    print(f"  Star list files: {len(star_list_files)}", flush=True)

    # 4. Accumulate normal equations
    print(f"  Accumulating normal equations...", flush=True)
    t0 = time.time()
    AtWA, rhs, n_stars, n_pairs = accumulate_normal_equations(
        star_list_files, node_to_idx, n_params
    )
    accum_time = time.time() - t0
    print(f"  Stars used: {n_stars:,}, pairs: {n_pairs:,}, "
          f"time: {accum_time:.1f}s", flush=True)
    print(f"  Matrix NNZ: {AtWA.nnz:,}, "
          f"size: {AtWA.data.nbytes / 1e6:.1f} MB", flush=True)

    results = {}

    # 5. Solve
    modes_to_run = []
    if mode in ("unanchored", "both"):
        modes_to_run.append("unanchored")
    if mode in ("anchored", "both"):
        modes_to_run.append("anchored")

    for solve_mode in modes_to_run:
        print(f"\n  --- Solving ({solve_mode}) ---", flush=True)

        if solve_mode == "unanchored":
            zp_solved, info = solve_unanchored(
                AtWA, rhs, config, node_to_idx, des_fgcm_zps
            )
        else:
            zp_solved, info = solve_anchored(
                AtWA, rhs, config, node_to_idx, des_fgcm_zps
            )

        print(f"  Converged: {info['converged']} "
              f"(info={info['cg_info']})", flush=True)
        print(f"  Iterations: {info['n_iterations']}", flush=True)
        print(f"  Relative residual: {info['relative_residual']:.2e}", flush=True)
        print(f"  Solve time: {info['solve_time_s']:.1f}s", flush=True)

        # Diagnostics
        result_df, diag = compute_diagnostics(
            zp_solved, node_to_idx, idx_to_node,
            des_fgcm_zps, nsc_zpterms, band, solve_mode
        )

        info.update(diag)

        # Save output
        out_file = phase2_dir / f"zeropoints_{solve_mode}.parquet"
        result_df.to_parquet(out_file, index=False)
        print(f"  Saved: {out_file}", flush=True)

        # Print summary
        print(f"\n  {'=' * 55}", flush=True)
        print(f"  Phase 2 Summary — {band}-band ({solve_mode})", flush=True)
        print(f"  {'=' * 55}", flush=True)
        print(f"    Parameters:        {info['n_params']:,}", flush=True)
        print(f"    DES nodes:         {info['n_des']:,}", flush=True)
        print(f"    Iterations:        {info['n_iterations']}", flush=True)
        print(f"    Rel. residual:     {info['relative_residual']:.2e}", flush=True)
        print(f"    ZP median:         {info['zp_median']:.4f} mag", flush=True)
        print(f"    ZP std:            {info['zp_std']:.4f} mag", flush=True)
        print(f"    DES diff RMS:      {info['des_diff_rms']*1000:.1f} mmag", flush=True)
        print(f"    DES diff median:   {info['des_diff_median']*1000:.1f} mmag", flush=True)
        print(f"    DES diff max:      {info['des_diff_max']*1000:.1f} mmag", flush=True)
        print(f"  {'=' * 55}", flush=True)

        results[solve_mode] = {
            "zp_solved": zp_solved,
            "result_df": result_df,
            "info": info,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: CG sparse solver"
    )
    parser.add_argument("--band", required=True, help="Filter band")
    parser.add_argument(
        "--mode", default="both", choices=["unanchored", "anchored", "both"],
        help="Solve mode",
    )
    parser.add_argument(
        "--test-region", action="store_true",
        help="Limit to test region",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--max-pixels", type=int, default=None,
        help="Limit to first N pixels",
    )
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

    if args.max_pixels is not None:
        pixels = pixels[: args.max_pixels]
        print(f"Limited to first {args.max_pixels} pixels", flush=True)

    print(f"Band: {args.band}, Mode: {args.mode}", flush=True)
    print(flush=True)

    results = run_solve(
        args.band, pixels, config, output_dir, cache_dir, mode=args.mode
    )

    return results


if __name__ == "__main__":
    main()
