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

from collections import defaultdict

from delve_ubercal.phase0_ingest import get_test_patch_pixels, get_test_region_pixels, load_config
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
    """Load DES FGCM zero-points as a dict: (expnum, ccdnum) -> mag_zero.

    FGCM mag_zero includes 2.5*log10(exptime). NSC images are in ADU/s,
    so our ZP_solved is per-second. We normalize mag_zero to a 90s
    equivalent: mag_zero_norm = mag_zero - 2.5*log10(exptime/90).
    For 90s survey exposures this is a no-op; for 200s deep fields
    it subtracts ~0.87 mag, removing the exposure-time mismatch.
    """
    fgcm = pd.read_parquet(
        cache_dir / "des_y6_fgcm_zeropoints.parquet",
        columns=["expnum", "ccdnum", "band", "mag_zero"],
    )
    fgcm = fgcm[fgcm["band"] == band]

    # Normalize mag_zero by exposure time
    exptime_file = cache_dir / "des_exptime.parquet"
    if exptime_file.exists():
        exptime_df = pd.read_parquet(exptime_file)
        fgcm = fgcm.merge(exptime_df[["expnum", "exptime"]], on="expnum", how="left")
        # For exposures without exptime data, assume 90s (no correction)
        fgcm["exptime"] = fgcm["exptime"].fillna(90.0)
        # Normalize to 90s equivalent
        fgcm["mag_zero"] = fgcm["mag_zero"] - 2.5 * np.log10(fgcm["exptime"] / 90.0)

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


def _build_exposure_regularization(node_to_idx, n_params, weight):
    """Build exposure-level regularization matrix.

    Adds a Laplacian penalty within each exposure that penalizes
    CCD-to-CCD ZP variation while leaving the exposure mean free.
    Prior std on each CCD deviation from exposure mean = 1/sqrt(weight).
    """
    from collections import defaultdict

    exp_to_indices = defaultdict(list)
    for (expnum, ccdnum), idx in node_to_idx.items():
        exp_to_indices[expnum].append(idx)

    rows, cols, vals = [], [], []
    n_exposures = 0
    for expnum, indices in exp_to_indices.items():
        n = len(indices)
        if n < 2:
            continue
        n_exposures += 1
        inv_n = 1.0 / n
        for i in range(n):
            # Diagonal: weight * (1 - 1/n)
            rows.append(indices[i])
            cols.append(indices[i])
            vals.append(weight * (1.0 - inv_n))
            # Off-diagonal: -weight / n
            for j in range(n):
                if j != i:
                    rows.append(indices[i])
                    cols.append(indices[j])
                    vals.append(-weight * inv_n)

    L = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n_params, n_params)
    ).tocsr()
    return L, n_exposures


def get_epoch_index(mjd, boundaries):
    """Return epoch index for a given MJD based on sorted boundaries."""
    for i, b in enumerate(boundaries):
        if mjd < b:
            return i
    return len(boundaries)


def load_exposure_mjds(cache_dir, band):
    """Load exposure MJDs as dict: expnum -> mjd."""
    mjd_file = cache_dir / f"nsc_exposure_mjd_{band}.parquet"
    if not mjd_file.exists():
        return {}
    df = pd.read_parquet(mjd_file)
    return dict(zip(df["expnum"].values, df["mjd"].values))


def compute_star_flat(zp_solved, idx_to_node, exposure_mjds, config,
                      flagged_nodes=None, flagged_exposures=None):
    """Compute per-CCD per-epoch star flat from initial ZP solution.

    The star flat is the median ZP residual (after removing the per-exposure
    gray term), grouped by (ccdnum, epoch).

    Parameters
    ----------
    flagged_nodes : set, optional
        (expnum, ccdnum) tuples to exclude from star flat computation.
    flagged_exposures : set, optional
        Expnums to exclude entirely.

    Returns
    -------
    star_flat : dict
        Maps (ccdnum, epoch_idx) -> flat correction (mag).
    epoch_boundaries : list
        Sorted MJD boundaries used for epoch assignment.
    """
    sf_cfg = config.get("starflat", {})
    epoch_bounds = sf_cfg.get("epoch_boundaries", {})
    global_bounds = sorted(epoch_bounds.get("global", []))

    # Group ZPs by exposure (skip flagged)
    exp_zps = defaultdict(list)
    for idx, (expnum, ccdnum) in enumerate(idx_to_node):
        if flagged_nodes and (expnum, ccdnum) in flagged_nodes:
            continue
        if flagged_exposures and expnum in flagged_exposures:
            continue
        exp_zps[expnum].append((int(ccdnum), zp_solved[idx]))

    # Per-exposure gray = median ZP across CCDs
    gray = {}
    for expnum, ccd_zps in exp_zps.items():
        zps = [zp for _, zp in ccd_zps]
        gray[expnum] = np.median(zps)

    # Compute delta = ZP - gray, group by (ccdnum, epoch)
    epoch_deltas = defaultdict(list)
    for expnum, ccd_zps in exp_zps.items():
        mjd = exposure_mjds.get(expnum)
        if mjd is None:
            continue
        epoch_idx = get_epoch_index(mjd, global_bounds)
        g = gray[expnum]
        for ccdnum, zp in ccd_zps:
            epoch_deltas[(ccdnum, epoch_idx)].append(zp - g)

    # Star flat = median delta per CCD per epoch
    star_flat = {}
    for key, deltas in epoch_deltas.items():
        if len(deltas) >= 3:
            star_flat[key] = float(np.median(deltas))
        else:
            star_flat[key] = 0.0

    n_epochs = len(global_bounds) + 1
    n_ccds = len(set(k[0] for k in star_flat))
    flat_vals = [v for v in star_flat.values() if v != 0]
    print(f"  Star flat: {n_ccds} CCDs x {n_epochs} epochs "
          f"= {len(star_flat)} parameters", flush=True)
    if flat_vals:
        print(f"  Star flat range: [{min(flat_vals)*1000:.1f}, "
              f"{max(flat_vals)*1000:.1f}] mmag", flush=True)
        print(f"  Star flat RMS: "
              f"{np.sqrt(np.mean(np.array(flat_vals)**2))*1000:.1f} mmag",
              flush=True)

    return star_flat, global_bounds


def accumulate_gray_normal_equations(star_list_files, exp_to_idx, n_exp_params,
                                     star_flat, exposure_mjds, epoch_boundaries):
    """Build normal equations with per-exposure (gray) parameters.

    Applies star flat correction to m_inst, then accumulates overlap
    constraints between different exposures. Intra-exposure pairs
    (same exposure, different CCDs) are skipped since the gray parameter
    is the same for both.

    Returns
    -------
    AtWA, rhs, n_stars_used, n_pairs, n_intra_skipped
    """
    rows = []
    cols = []
    vals = []
    rhs = np.zeros(n_exp_params, dtype=np.float64)
    n_stars_used = 0
    n_pairs = 0
    n_intra_skipped = 0

    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue

        for star_id, group in df.groupby("objectid"):
            if len(group) < 2:
                continue

            expnums = group["expnum"].values
            ccdnums = group["ccdnum"].values
            mags = group["m_inst"].values
            errs = group["m_err"].values

            # Apply star flat correction and map to exposure indices
            corr_mags = []
            corr_errs = []
            corr_exp_idx = []
            for j in range(len(expnums)):
                exp_idx = exp_to_idx.get(int(expnums[j]))
                if exp_idx is None:
                    continue
                mjd = exposure_mjds.get(int(expnums[j]))
                if mjd is None:
                    continue
                epoch = get_epoch_index(mjd, epoch_boundaries)
                flat = star_flat.get((int(ccdnums[j]), epoch), 0.0)
                corr_mags.append(mags[j] + flat)
                corr_errs.append(errs[j])
                corr_exp_idx.append(exp_idx)

            n_det = len(corr_exp_idx)
            if n_det < 2:
                continue

            n_stars_used += 1

            # Accumulate inter-exposure pairs
            for a in range(n_det):
                for b in range(a + 1, n_det):
                    ia, ib = corr_exp_idx[a], corr_exp_idx[b]
                    if ia == ib:
                        n_intra_skipped += 1
                        continue

                    ma, mb = corr_mags[a], corr_mags[b]
                    ea, eb = corr_errs[a], corr_errs[b]
                    w = 1.0 / (ea * ea + eb * eb)
                    dm = ma - mb

                    rows.append(ia); cols.append(ia); vals.append(w)
                    rows.append(ib); cols.append(ib); vals.append(w)
                    rows.append(ia); cols.append(ib); vals.append(-w)
                    rows.append(ib); cols.append(ia); vals.append(-w)
                    rhs[ia] -= w * dm
                    rhs[ib] += w * dm
                    n_pairs += 1

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    vals = np.array(vals, dtype=np.float64)
    AtWA = sp.coo_matrix(
        (vals, (rows, cols)), shape=(n_exp_params, n_exp_params)
    ).tocsr()

    return AtWA, rhs, n_stars_used, n_pairs, n_intra_skipped


def solve_gray_unanchored(AtWA, rhs, config, exp_to_idx, des_fgcm_zps,
                          star_flat, exposure_mjds, epoch_boundaries,
                          node_to_idx, idx_to_node, node_positions=None):
    """Solve for per-exposure gray terms (unanchored), then reconstruct
    full ZP = gray + star_flat for each CCD-exposure.

    Returns
    -------
    zp_full : np.ndarray
        Full ZP for each CCD-exposure (same indexing as idx_to_node).
    gray_solved : np.ndarray
        Per-exposure gray terms.
    info : dict
    """
    solve_cfg = config["solve"]
    n = AtWA.shape[0]

    # Tikhonov regularization
    tikhonov = solve_cfg["tikhonov_reg"]
    AtWA_reg = AtWA + sp.eye(n, format="csr") * tikhonov

    # Temporal smoothness regularization: consecutive exposures within the
    # same night should have similar gray ZPs (atmosphere varies slowly).
    # This is purely internal — no external reference, just a physical prior.
    smooth_sigma = config.get("starflat", {}).get("gray_smooth_sigma", 0.0)
    n_smooth_pairs = 0
    if smooth_sigma > 0:
        smooth_weight = 1.0 / (smooth_sigma ** 2)
        idx_to_exp = {idx: exp for exp, idx in exp_to_idx.items()}
        # Sort exposures by MJD
        exp_mjd_pairs = []
        for exp_idx in range(n):
            expnum = idx_to_exp.get(exp_idx)
            if expnum and expnum in exposure_mjds:
                exp_mjd_pairs.append((exposure_mjds[expnum], exp_idx))
        exp_mjd_pairs.sort()

        # Add smoothness constraint between consecutive exposures
        # within same night (MJD gap < 0.02 day = ~30 min)
        max_gap = 0.02  # days
        for k in range(len(exp_mjd_pairs) - 1):
            mjd_a, idx_a = exp_mjd_pairs[k]
            mjd_b, idx_b = exp_mjd_pairs[k + 1]
            if (mjd_b - mjd_a) < max_gap:
                AtWA_reg[idx_a, idx_a] += smooth_weight
                AtWA_reg[idx_b, idx_b] += smooth_weight
                AtWA_reg[idx_a, idx_b] -= smooth_weight
                AtWA_reg[idx_b, idx_a] -= smooth_weight
                n_smooth_pairs += 1
        print(f"    Temporal smoothness: σ={smooth_sigma*1000:.0f} mmag, "
              f"{n_smooth_pairs:,} pairs (weight={smooth_weight:.0f})",
              flush=True)

    # Solve
    t0 = time.time()
    n_iter = [0]
    def callback(xk):
        n_iter[0] += 1

    gray_solved, cg_info = cg(
        AtWA_reg, rhs,
        rtol=float(solve_cfg["tolerance"]),
        maxiter=solve_cfg["max_iterations"],
        callback=callback,
    )
    solve_time = time.time() - t0

    residual = AtWA_reg @ gray_solved - rhs
    rel_residual = (np.linalg.norm(residual) / np.linalg.norm(rhs)
                    if np.linalg.norm(rhs) > 0 else 0)

    # Shift to DES median (using per-exposure FGCM medians)
    # For each exposure, compute median FGCM across its CCDs
    idx_to_exp = {idx: exp for exp, idx in exp_to_idx.items()}
    exp_fgcm_medians = {}
    exp_fgcm_nodes = defaultdict(list)
    for (expnum, ccdnum), _ in node_to_idx.items():
        if (expnum, ccdnum) in des_fgcm_zps:
            val = des_fgcm_zps[(expnum, ccdnum)]
            if 25.0 < val < 35.0:
                exp_fgcm_nodes[expnum].append(val)
    for expnum, vals in exp_fgcm_nodes.items():
        exp_fgcm_medians[expnum] = np.median(vals)

    des_gray_diffs = []
    for exp_idx in range(n):
        expnum = idx_to_exp.get(exp_idx)
        if expnum and expnum in exp_fgcm_medians:
            des_gray_diffs.append(gray_solved[exp_idx] - exp_fgcm_medians[expnum])

    if des_gray_diffs:
        offset = np.median(des_gray_diffs)
        gray_solved -= offset
    else:
        offset = 0.0

    # Reconstruct full ZP = gray + star_flat for each CCD-exposure
    n_nodes = len(idx_to_node)
    zp_full = np.zeros(n_nodes, dtype=np.float64)
    for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
        exp_idx = exp_to_idx.get(expnum)
        if exp_idx is None:
            continue
        mjd = exposure_mjds.get(expnum)
        epoch = get_epoch_index(mjd, epoch_boundaries) if mjd else 0
        flat = star_flat.get((int(ccdnum), epoch), 0.0)
        zp_full[node_idx] = gray_solved[exp_idx] + flat

    # Gradient detrend: remove linear RA/Dec gradient.
    # The chain-like overlap graph has poorly constrained large-scale modes;
    # this correction removes the linear component of the solver-introduced drift.
    # Options: "fgcm" (uses DES FGCM), "gaia" (done externally after this function), false
    gradient_detrend = config.get("solve", {}).get("gradient_detrend", False)
    ra_slope_applied = 0.0
    dec_slope_applied = 0.0
    if gradient_detrend == "fgcm" and node_positions is not None:
        ra_vals, dec_vals, diff_vals = [], [], []
        for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
            key = (int(expnum), int(ccdnum))
            if key in des_fgcm_zps and key in node_positions:
                fgcm_val = des_fgcm_zps[key]
                if 25.0 < fgcm_val < 35.0:
                    diff_mmag = (zp_full[node_idx] - fgcm_val) * 1000
                    if abs(diff_mmag) < 200:  # exclude flagged/outlier nodes
                        ra_vals.append(node_positions[key][0])
                        dec_vals.append(node_positions[key][1])
                        diff_vals.append(diff_mmag)

        if len(ra_vals) > 100:
            ra_arr = np.array(ra_vals)
            dec_arr = np.array(dec_vals)
            diff_arr = np.array(diff_vals)
            ra_center = ra_arr.mean()
            dec_center = dec_arr.mean()

            # Fit: diff = a*(RA-RA_center) + b*(Dec-Dec_center) + c
            A_fit = np.column_stack([
                ra_arr - ra_center,
                dec_arr - dec_center,
                np.ones(len(ra_arr)),
            ])
            coeffs, _, _, _ = np.linalg.lstsq(A_fit, diff_arr, rcond=None)
            ra_slope_applied = coeffs[0]  # mmag/deg
            dec_slope_applied = coeffs[1]

            # Apply correction to ALL nodes (DES and non-DES)
            for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
                key = (int(expnum), int(ccdnum))
                if key in node_positions:
                    ra, dec = node_positions[key]
                    correction = (coeffs[0] * (ra - ra_center) +
                                  coeffs[1] * (dec - dec_center)) / 1000  # mag
                    zp_full[node_idx] -= correction

            # Re-shift to DES median after gradient removal
            des_diffs_post = []
            for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
                key = (int(expnum), int(ccdnum))
                if key in des_fgcm_zps:
                    val = des_fgcm_zps[key]
                    if 25.0 < val < 35.0:
                        des_diffs_post.append(zp_full[node_idx] - val)
            if des_diffs_post:
                offset_post = np.median(des_diffs_post)
                zp_full -= offset_post

            print(f"    Gradient detrend: RA={ra_slope_applied:.2f}, "
                  f"Dec={dec_slope_applied:.2f} mmag/deg "
                  f"({len(ra_vals)} DES nodes)", flush=True)

    info = {
        "mode": "unanchored_gray",
        "n_exp_params": n,
        "n_iterations": n_iter[0],
        "cg_info": cg_info,
        "relative_residual": rel_residual,
        "solve_time_s": solve_time,
        "des_offset_applied": offset,
        "converged": cg_info == 0,
        "gradient_ra_slope": ra_slope_applied,
        "gradient_dec_slope": dec_slope_applied,
    }
    return zp_full, gray_solved, info


def apply_gaia_gradient_detrend(zp_full, idx_to_node, node_to_idx, node_positions,
                                star_list_files, gaia_cache_path, band,
                                des_fgcm_zps=None, output_dir=None):
    """Measure and remove linear RA/Dec gradient using Gaia DR3 as reference.

    All-sky method — no FGCM dependency for gradient measurement.
    Always uses DELVE g-r color for the color term (requires both g and r
    bands to have completed Phase 3). No fallback — raises error if the
    other band is not available.

    Parameters
    ----------
    zp_full : ndarray
        Per-CCD-exposure ZP array (modified in place).
    idx_to_node : list of (expnum, ccdnum) tuples
    node_to_idx : dict mapping (expnum, ccdnum) -> index
    node_positions : dict mapping (expnum, ccdnum) -> (ra, dec)
    star_list_files : list of Path
        Star list parquet files for this band.
    gaia_cache_path : str or Path
        Path to Gaia DR3 crossmatch parquet.
    band : str
    des_fgcm_zps : dict, optional
        If provided, re-shifts to DES median after gradient correction.
    output_dir : Path
        Pipeline output directory. Required — loads other band's Phase 3
        ZPs to compute DELVE g-r color.

    Returns
    -------
    dict with ra_slope, dec_slope (mmag/deg), n_stars, etc.
    """
    from pathlib import Path as _Path

    if output_dir is None:
        raise ValueError("output_dir is required for gradient detrend "
                         "(need both g and r bands)")

    gaia_mag_col = 'phot_bp_mean_mag' if band == 'g' else 'phot_rp_mean_mag'

    # --- Compute per-star calibrated magnitude in this band ---
    chunks = []
    for f in star_list_files:
        cols = ['objectid', 'm_inst', 'm_err', 'expnum', 'ccdnum']
        df = pd.read_parquet(f, columns=cols)
        node_keys = list(zip(df['expnum'].astype(int).values,
                             df['ccdnum'].astype(int).values))
        node_indices = np.array([node_to_idx.get(k, -1) for k in node_keys])
        valid = node_indices >= 0
        df = df[valid].copy()
        node_indices = node_indices[valid]
        df['m_cal'] = df['m_inst'].values + zp_full[node_indices]
        df['weight'] = 1.0 / (df['m_err'].values**2 + 1e-6)
        df['wm'] = df['m_cal'] * df['weight']
        chunks.append(df[['objectid', 'wm', 'weight']])

    all_det = pd.concat(chunks, ignore_index=True)
    stars = all_det.groupby('objectid').agg(
        sum_wm=('wm', 'sum'), sum_w=('weight', 'sum'),
        n_det=('weight', 'count'),
    )
    stars['m_cal'] = stars['sum_wm'] / stars['sum_w']
    stars = stars[stars['n_det'] >= 2]

    # --- Load other band for DELVE g-r color (required) ---
    # Prefer raw (pre-detrend) ZPs to avoid contaminating the color with
    # the other band's gradient correction
    other_band = 'r' if band == 'g' else 'g'
    other_raw_file = _Path(output_dir) / f'phase3_{other_band}' / 'zeropoints_unanchored_raw.parquet'
    other_zp_file = other_raw_file if other_raw_file.exists() else \
        _Path(output_dir) / f'phase3_{other_band}' / 'zeropoints_unanchored.parquet'
    other_star_dir = _Path(output_dir) / f'phase1_{other_band}'

    if not other_zp_file.exists():
        raise FileNotFoundError(
            f"Other band ({other_band}) Phase 3 ZPs not found at {other_zp_file}. "
            f"Run Phase 3 for {other_band} first, then detrend both bands.")
    if not other_star_dir.exists():
        raise FileNotFoundError(
            f"Other band ({other_band}) star lists not found at {other_star_dir}.")

    other_zp_df = pd.read_parquet(other_zp_file)
    other_zp_dict = dict(zip(
        zip(other_zp_df['expnum'].astype(int), other_zp_df['ccdnum'].astype(int)),
        other_zp_df['zp_solved']))
    other_star_files = sorted(other_star_dir.glob('star_lists_nside*_pixel*.parquet'))
    if not other_star_files:
        raise FileNotFoundError(
            f"No star list files found in {other_star_dir}.")

    ochunks = []
    for f in other_star_files:
        df2 = pd.read_parquet(f, columns=['objectid', 'm_inst', 'm_err',
                                           'expnum', 'ccdnum'])
        keys2 = list(zip(df2['expnum'].astype(int).values,
                         df2['ccdnum'].astype(int).values))
        zps2 = np.array([other_zp_dict.get(k, np.nan) for k in keys2])
        v2 = np.isfinite(zps2)
        df2 = df2[v2].copy()
        df2['m_cal'] = df2['m_inst'].values + zps2[v2]
        df2['weight'] = 1.0 / (df2['m_err'].values**2 + 1e-6)
        df2['wm'] = df2['m_cal'] * df2['weight']
        ochunks.append(df2[['objectid', 'wm', 'weight']])
    odet = pd.concat(ochunks, ignore_index=True)
    ostars = odet.groupby('objectid').agg(
        sum_wm_o=('wm', 'sum'), sum_w_o=('weight', 'sum'),
    )
    ostars['m_cal_other'] = ostars['sum_wm_o'] / ostars['sum_w_o']
    stars = stars.join(ostars[['m_cal_other']], how='inner')
    if band == 'g':
        stars['delve_gr'] = stars['m_cal'] - stars['m_cal_other']
    else:
        stars['delve_gr'] = stars['m_cal_other'] - stars['m_cal']

    print(f"    Gradient detrend: {len(stars):,} stars with DELVE g-r color", flush=True)

    # --- Cross-match with Gaia (provides ra, dec, reference mag) ---
    gaia_full = pd.read_parquet(gaia_cache_path,
                                columns=['nsc_objectid', 'ra', 'dec',
                                         gaia_mag_col, 'bp_rp',
                                         'phot_g_mean_flux_over_error'])
    gaia_full = gaia_full[(gaia_full[gaia_mag_col].between(10, 25)) &
                          (gaia_full['bp_rp'].between(0.3, 3.5)) &
                          (gaia_full['phot_g_mean_flux_over_error'] > 50)]
    gaia_full = gaia_full.set_index('nsc_objectid')

    stars = stars.join(gaia_full[['ra', 'dec', gaia_mag_col]], how='inner')
    stars['resid_mmag'] = (stars['m_cal'] - stars[gaia_mag_col]) * 1000

    # Clip extreme outliers
    med = stars['resid_mmag'].median()
    stars = stars[stars['resid_mmag'].between(med - 500, med + 500)]
    stars = stars[stars['delve_gr'].between(-0.5, 2.5)]

    print(f"    Gradient detrend: {len(stars):,} matched stars after cuts", flush=True)

    if len(stars) < 100:
        print("    Gradient detrend: too few stars, skipping", flush=True)
        return {"ra_slope": 0.0, "dec_slope": 0.0, "n_stars": len(stars)}

    # --- Joint fit: resid = poly3(DELVE g-r) + a*RA + b*Dec + c ---
    ra_arr = stars['ra'].values
    dec_arr = stars['dec'].values
    resid_arr = stars['resid_mmag'].values
    color_arr = stars['delve_gr'].values

    ra_center = ra_arr.mean()
    dec_center = dec_arr.mean()
    color_median = np.median(color_arr)
    dc = color_arr - color_median

    A_fit = np.column_stack([
        dc, dc**2, dc**3,
        ra_arr - ra_center,
        dec_arr - dec_center,
        np.ones(len(ra_arr)),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A_fit, resid_arr, rcond=None)

    # 3-sigma clip and refit
    resid_model = A_fit @ coeffs
    scatter = resid_arr - resid_model
    std = np.std(scatter)
    clip = np.abs(scatter) < 3 * std
    if clip.sum() > 100:
        coeffs, _, _, _ = np.linalg.lstsq(A_fit[clip], resid_arr[clip], rcond=None)

    ra_slope = coeffs[3]   # mmag/deg
    dec_slope = coeffs[4]  # mmag/deg

    # Apply gradient correction to ALL nodes
    for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
        key = (int(expnum), int(ccdnum))
        if key in node_positions:
            ra, dec = node_positions[key]
            correction = (ra_slope * (ra - ra_center) +
                          dec_slope * (dec - dec_center)) / 1000  # mag
            zp_full[node_idx] -= correction

    # Re-shift to DES FGCM median (absolute scale only — 1 parameter)
    if des_fgcm_zps:
        des_diffs = []
        for node_idx, (expnum, ccdnum) in enumerate(idx_to_node):
            key = (int(expnum), int(ccdnum))
            if key in des_fgcm_zps:
                val = des_fgcm_zps[key]
                if 25.0 < val < 35.0:
                    des_diffs.append(zp_full[node_idx] - val)
        if des_diffs:
            zp_full -= np.median(des_diffs)

    n_final = int(clip.sum()) if clip.sum() > 100 else len(stars)
    print(f"    Gradient detrend (DELVE g-r): RA={ra_slope:.2f}, "
          f"Dec={dec_slope:.2f} mmag/deg ({n_final:,} stars)", flush=True)

    return {
        "ra_slope": ra_slope,
        "dec_slope": dec_slope,
        "ra_center": ra_center,
        "dec_center": dec_center,
        "n_stars": n_final,
        "color_source": "DELVE g-r",
        "source": "gaia",
    }


def solve_unanchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps):
    """Solve in unanchored mode with Tikhonov + exposure regularization.

    After solving, shifts the solution so that
    median(ZP_solved[DES] - ZP_FGCM[DES]) = 0.

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

    # Add exposure-level regularization if configured
    exp_reg_weight = float(solve_cfg.get("exposure_reg_weight", 0))
    n_reg_exp = 0
    if exp_reg_weight > 0:
        L_exp, n_reg_exp = _build_exposure_regularization(
            node_to_idx, n, exp_reg_weight
        )
        AtWA_reg = AtWA_reg + L_exp

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

    # Shift to DES mean (use median, filter sentinel FGCM values)
    des_indices = []
    des_fgcm_vals = []
    for node, idx in node_to_idx.items():
        if node in des_fgcm_zps:
            fgcm_val = des_fgcm_zps[node]
            # Filter sentinel values (-9999, >35, etc.)
            if 25.0 < fgcm_val < 35.0:
                des_indices.append(idx)
                des_fgcm_vals.append(fgcm_val)

    if des_indices:
        des_indices = np.array(des_indices)
        des_fgcm_vals = np.array(des_fgcm_vals)
        offset = np.median(zp_solved[des_indices] - des_fgcm_vals)
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
        "n_reg_exposures": n_reg_exp,
        "exp_reg_weight": exp_reg_weight,
    }
    return zp_solved, info


def detrend_spatial_gradient(zp_solved, idx_to_node, node_positions, des_fgcm_zps,
                             poly_degree=3):
    """Remove large-scale spatial modes from unanchored ZP solution.

    The unanchored solve has poorly constrained large-scale modes because
    all overlaps are local (< 2-3 deg).  This fits a polynomial surface to
    the DES ZP residuals (ZP_solved - ZP_FGCM) as a function of sky position
    and subtracts the smooth correction from ALL nodes, then re-shifts to
    the DES FGCM median.

    Using DES FGCM as reference for the fit effectively "soft-anchors" the
    large-scale modes while leaving small-scale calibration to the solver.

    Parameters
    ----------
    zp_solved : np.ndarray
        Solved ZP values (modified in-place).
    idx_to_node : list of (expnum, ccdnum)
    node_positions : dict
        (expnum, ccdnum) -> (ra_mean, dec_mean).
    des_fgcm_zps : dict
        (expnum, ccdnum) -> FGCM ZP.
    poly_degree : int
        Polynomial degree for the spatial fit (default 3).
        Degree 1 = linear plane, 3 = cubic (captures Dec=-30 step).

    Returns
    -------
    gradient_info : dict
        Fit details and number of nodes used.
    """
    n = len(idx_to_node)
    ra = np.full(n, np.nan)
    dec = np.full(n, np.nan)
    for idx, node in enumerate(idx_to_node):
        pos = node_positions.get(node)
        if pos is not None:
            ra[idx], dec[idx] = pos

    # Identify DES nodes with valid FGCM ZPs
    des_mask = np.zeros(n, dtype=bool)
    fgcm_arr = np.full(n, np.nan)
    for idx, node in enumerate(idx_to_node):
        if node in des_fgcm_zps:
            fgcm_val = des_fgcm_zps[node]
            if 25.0 < fgcm_val < 35.0:
                des_mask[idx] = True
                fgcm_arr[idx] = fgcm_val

    valid = np.isfinite(ra) & np.isfinite(dec) & (zp_solved > 1.0) & des_mask
    if valid.sum() < 10:
        return {"n_nodes_fit": 0}

    # Center coordinates for numerical stability
    ra_mean = np.mean(ra[valid])
    dec_mean = np.mean(dec[valid])
    ra_c = ra - ra_mean
    dec_c = dec - dec_mean

    # Compute DES residuals (what we want to make smooth / zero)
    resid = zp_solved[valid] - fgcm_arr[valid]

    # Build polynomial design matrix: 1, ra, dec, ra*dec, dec^2, dec^3, ...
    # Use full polynomial in Dec (up to poly_degree) but only linear in RA
    # because Dec=-30 step is the main structure to remove
    cols = [np.ones(valid.sum())]
    col_names = ["const"]
    ra_v = ra_c[valid]
    dec_v = dec_c[valid]
    # RA terms (linear only)
    cols.append(ra_v)
    col_names.append("ra")
    # Dec terms up to poly_degree
    for d in range(1, poly_degree + 1):
        cols.append(dec_v ** d)
        col_names.append(f"dec^{d}")
    # Cross term: RA * Dec
    cols.append(ra_v * dec_v)
    col_names.append("ra*dec")

    A_fit = np.column_stack(cols)

    # Robust fit: use iterative sigma-clipping to avoid outlier influence
    mask_fit = np.ones(valid.sum(), dtype=bool)
    for iteration in range(3):
        if mask_fit.sum() < 10:
            break
        coeffs = np.linalg.lstsq(A_fit[mask_fit], resid[mask_fit], rcond=None)[0]
        model = A_fit @ coeffs
        fit_resid = resid - model
        sigma = np.std(fit_resid[mask_fit])
        mask_fit = np.abs(fit_resid) < 3 * sigma

    # Final fit on clipped data
    coeffs = np.linalg.lstsq(A_fit[mask_fit], resid[mask_fit], rcond=None)[0]

    # Evaluate the smooth correction for ALL nodes with positions
    has_pos = np.isfinite(ra)
    cols_all = [np.ones(has_pos.sum())]
    ra_all = ra_c[has_pos]
    dec_all = dec_c[has_pos]
    cols_all.append(ra_all)
    for d in range(1, poly_degree + 1):
        cols_all.append(dec_all ** d)
    cols_all.append(ra_all * dec_all)
    A_all = np.column_stack(cols_all)

    correction = A_all @ coeffs
    # Subtract the smooth correction (excluding constant term)
    # We keep the constant because the re-shift handles absolute level
    zp_solved[has_pos] -= (correction - coeffs[0])

    # Re-shift to DES median
    des_indices = []
    des_fgcm_vals = []
    for idx, node in enumerate(idx_to_node):
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

    # Print summary
    ra_slope = coeffs[1] * 1000  # mmag/deg
    dec_slope = coeffs[2] * 1000  # mmag/deg for linear Dec term
    rms_before = np.std(resid[mask_fit]) * 1000
    resid_after = zp_solved[valid][mask_fit] - fgcm_arr[valid][mask_fit]
    resid_after -= np.median(resid_after)
    rms_after = np.std(resid_after) * 1000
    print(f"    Spatial detrend (degree {poly_degree}): "
          f"RA slope = {ra_slope:.1f} mmag/deg, "
          f"Dec slope = {dec_slope:.1f} mmag/deg", flush=True)
    print(f"    DES residual RMS: {rms_before:.1f} -> {rms_after:.1f} mmag "
          f"({valid.sum():,} DES nodes, {mask_fit.sum():,} after clipping)",
          flush=True)

    coeff_info = {col_names[i]: float(coeffs[i] * 1000)
                  for i in range(len(col_names))}

    return {
        "poly_degree": poly_degree,
        "coefficients_mmag": coeff_info,
        "n_nodes_fit": int(valid.sum()),
        "n_nodes_clipped": int(valid.sum() - mask_fit.sum()),
        "rms_before_mmag": float(rms_before),
        "rms_after_mmag": float(rms_after),
    }


def compute_node_positions(phase0_dir, nside=32):
    """Compute mean RA, Dec per (expnum, ccdnum) from Phase 0 detections.

    Returns dict mapping (expnum, ccdnum) -> (ra_mean, dec_mean).
    Cached to phase0_dir / 'node_positions.parquet'.
    """
    cache = phase0_dir / "node_positions.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return dict(zip(
            zip(df["expnum"].values, df["ccdnum"].values),
            zip(df["ra_mean"].values, df["dec_mean"].values),
        ))

    files = sorted(phase0_dir.glob(f"detections_nside{nside}_pixel*.parquet"))
    if not files:
        files = sorted(phase0_dir.glob("*.parquet"))
    chunks = []
    for f in files:
        df = pd.read_parquet(f, columns=["expnum", "ccdnum", "ra", "dec"])
        if len(df) == 0:
            continue
        pos = df.groupby(["expnum", "ccdnum"]).agg(
            ra_mean=("ra", "mean"), dec_mean=("dec", "mean"),
        ).reset_index()
        chunks.append(pos)

    if not chunks:
        return {}

    result = pd.concat(chunks, ignore_index=True).groupby(
        ["expnum", "ccdnum"]
    ).agg(ra_mean=("ra_mean", "mean"), dec_mean=("dec_mean", "mean")).reset_index()
    result.to_parquet(cache, index=False)
    return dict(zip(
        zip(result["expnum"].values, result["ccdnum"].values),
        zip(result["ra_mean"].values, result["dec_mean"].values),
    ))


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

    # Add exposure-level regularization if configured
    exp_reg_weight = float(solve_cfg.get("exposure_reg_weight", 0))
    n_reg_exp = 0
    if exp_reg_weight > 0:
        L_exp, n_reg_exp = _build_exposure_regularization(
            node_to_idx, n, exp_reg_weight
        )
        AtWA_anch = AtWA_anch + L_exp

    # Add DES anchoring: for each DES CCD-exposure, add penalty
    # Filter sentinel FGCM values (-9999, >35, etc.)
    n_anchored = 0
    for node, idx in node_to_idx.items():
        if node in des_fgcm_zps:
            zp_fgcm = des_fgcm_zps[node]
            if not (25.0 < zp_fgcm < 35.0):
                continue
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
        "n_reg_exposures": n_reg_exp,
        "exp_reg_weight": exp_reg_weight,
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

    # DES-specific diagnostics (filter sentinel FGCM values)
    des_mask = np.array([
        node in des_fgcm_zps and 25.0 < des_fgcm_zps[node] < 35.0
        for node in idx_to_node
    ])
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
    parser.add_argument(
        "--test-patch", action="store_true",
        help="Limit to 10x10 deg test patch RA=50-60, Dec=-35 to -25",
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

    if args.test_patch:
        pixels = get_test_patch_pixels(nside)
        print(f"Test patch (10x10 deg): {len(pixels)} pixels (nside={nside})", flush=True)
    elif args.test_region:
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
