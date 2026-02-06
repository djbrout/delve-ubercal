"""Phase 3: Iterative outlier rejection and re-solving.

Removes variable stars, catastrophic outliers, non-photometric exposures,
and bad CCDs, then re-runs the Phase 2 CG solve on the cleaned catalog.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from delve_ubercal.phase0_ingest import get_test_region_pixels, load_config
from delve_ubercal.phase2_solve import (
    accumulate_normal_equations,
    build_node_index,
    load_des_fgcm_zps,
    load_nsc_zpterms,
    solve_anchored,
    solve_unanchored,
)
from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels


def load_exposure_mjds(cache_dir, band):
    """Load or download MJDs for exposures.

    Returns dict: expnum -> mjd.
    """
    mjd_cache = cache_dir / f"nsc_exposure_mjd_{band}.parquet"
    if mjd_cache.exists():
        df = pd.read_parquet(mjd_cache)
    else:
        from dl import queryClient as qc
        print(f"  Downloading exposure MJDs ({band})...", flush=True)
        query = f"""
        SELECT expnum, mjd
        FROM nsc_dr2.exposure
        WHERE filter = '{band}'
          AND instrument = 'c4d'
        """
        df = qc.query(sql=query, fmt="pandas", timeout=600)
        df.to_parquet(mjd_cache, index=False)
        print(f"  Downloaded {len(df):,} exposure MJDs.", flush=True)
    return dict(zip(df["expnum"].values, df["mjd"].values))


def compute_residuals(star_list_files, zp_dict, node_to_idx):
    """Compute per-detection residuals given a ZP solution.

    Parameters
    ----------
    star_list_files : list of Path
        Per-pixel star detection parquet files.
    zp_dict : dict
        Maps (expnum, ccdnum) -> zp_solved.
    node_to_idx : dict
        Maps (expnum, ccdnum) -> index (used for filtering to connected nodes).

    Returns
    -------
    residuals_df : pd.DataFrame
        Detections with added columns: zp_solved, m_cal, m_star_mean, residual.
    """
    all_dfs = []
    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue

        # Add ZP for each detection
        nodes = list(zip(df["expnum"].values, df["ccdnum"].values))
        zps = np.array([zp_dict.get(n, np.nan) for n in nodes])
        connected = np.array([n in node_to_idx for n in nodes])

        df = df[connected].copy()
        zps = zps[connected]
        df["zp_solved"] = zps

        # Calibrated magnitude
        df["m_cal"] = df["m_inst"].values + zps

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute weighted mean magnitude per star
    w = 1.0 / (combined["m_err"].values ** 2)
    combined["_w"] = w
    combined["_wm"] = w * combined["m_cal"].values

    star_stats = combined.groupby("objectid").agg(
        sum_w=("_w", "sum"),
        sum_wm=("_wm", "sum"),
        n_det=("_w", "count"),
    )
    star_stats["m_star_mean"] = star_stats["sum_wm"] / star_stats["sum_w"]

    combined = combined.merge(
        star_stats[["m_star_mean", "n_det"]],
        left_on="objectid", right_index=True, how="left",
    )

    # Residual
    combined["residual"] = combined["m_cal"] - combined["m_star_mean"]
    combined.drop(columns=["_w", "_wm"], inplace=True)

    return combined


def flag_variable_stars(residuals_df, chi2_cut):
    """Flag stars with chi2/dof > threshold (likely variables).

    Returns set of objectids.
    """
    grouped = residuals_df.groupby("objectid").apply(
        lambda g: pd.Series({
            "chi2": np.sum((g["residual"].values / g["m_err"].values) ** 2),
            "dof": len(g) - 1,
        }),
        include_groups=False,
    )
    # Require at least 2 detections (dof >= 1)
    grouped = grouped[grouped["dof"] >= 1]
    grouped["chi2_dof"] = grouped["chi2"] / grouped["dof"]
    flagged = set(grouped[grouped["chi2_dof"] > chi2_cut].index)
    return flagged


def flag_outlier_detections(residuals_df, sigma_cut):
    """Flag individual detections with |residual| > sigma_cut * error.

    Returns boolean mask (True = flagged).
    """
    return np.abs(residuals_df["residual"].values) > (
        sigma_cut * residuals_df["m_err"].values
    )


def flag_bad_exposures(zp_solved_df, exposure_mjds, zp_cut):
    """Flag exposures where ZP deviates > zp_cut from nightly median.

    Parameters
    ----------
    zp_solved_df : pd.DataFrame
        Phase 2 output with (expnum, ccdnum, zp_solved).
    exposure_mjds : dict
        expnum -> MJD.
    zp_cut : float
        Max deviation in mag.

    Returns set of expnums.
    """
    # Get per-exposure median ZP
    exp_medians = zp_solved_df.groupby("expnum")["zp_solved"].median().reset_index()
    exp_medians.columns = ["expnum", "zp_exp_median"]

    # Add night (floor of MJD)
    exp_medians["mjd"] = exp_medians["expnum"].map(exposure_mjds)
    exp_medians = exp_medians.dropna(subset=["mjd"])
    exp_medians["night"] = np.floor(exp_medians["mjd"].values).astype(int)

    # Nightly median
    nightly = exp_medians.groupby("night")["zp_exp_median"].median()
    exp_medians = exp_medians.merge(
        nightly.rename("zp_nightly_median"),
        left_on="night", right_index=True, how="left",
    )
    exp_medians["zp_deviation"] = np.abs(
        exp_medians["zp_exp_median"] - exp_medians["zp_nightly_median"]
    )

    flagged = set(
        exp_medians[exp_medians["zp_deviation"] > zp_cut]["expnum"].values
    )
    return flagged


def flag_bad_ccds(residuals_df, sigma_factor=3.0):
    """Flag CCDs with anomalously large intra-CCD scatter.

    Returns set of (expnum, ccdnum) nodes.
    """
    # Compute RMS residual per (expnum, ccdnum)
    node_stats = residuals_df.groupby(["expnum", "ccdnum"])["residual"].agg(
        ["std", "count"]
    ).reset_index()
    node_stats = node_stats[node_stats["count"] >= 3]

    if len(node_stats) == 0:
        return set()

    median_scatter = node_stats["std"].median()
    mad = np.median(np.abs(node_stats["std"] - median_scatter))
    # Robust sigma estimate
    sigma_est = 1.4826 * mad if mad > 0 else node_stats["std"].std()

    threshold = median_scatter + sigma_factor * sigma_est
    bad = node_stats[node_stats["std"] > threshold]
    return set(zip(bad["expnum"].values, bad["ccdnum"].values))


def apply_flags_to_star_lists(star_list_files, flagged_stars, flagged_detection_mask,
                               flagged_exposures, flagged_nodes, output_dir):
    """Remove flagged detections and write cleaned star lists.

    Returns list of cleaned file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files = []

    # If we have a detection-level mask, we need to process differently
    # Build a set of flagged (objectid, expnum, ccdnum) for detection-level flags
    # For simplicity, we process file-by-file and re-apply the criteria

    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            out_f = output_dir / f.name
            df.to_parquet(out_f, index=False)
            cleaned_files.append(out_f)
            continue

        n_before = len(df)

        # Remove flagged stars
        if flagged_stars:
            df = df[~df["objectid"].isin(flagged_stars)]

        # Remove flagged exposures
        if flagged_exposures:
            df = df[~df["expnum"].isin(flagged_exposures)]

        # Remove flagged nodes (bad CCDs)
        if flagged_nodes:
            node_keys = set(zip(df["expnum"].values, df["ccdnum"].values))
            bad_mask = np.array([
                (e, c) in flagged_nodes
                for e, c in zip(df["expnum"].values, df["ccdnum"].values)
            ])
            df = df[~bad_mask]

        # Remove stars with < 2 detections after flagging
        counts = df.groupby("objectid")["objectid"].transform("count")
        df = df[counts >= 2]

        out_f = output_dir / f.name
        df.to_parquet(out_f, index=False)
        cleaned_files.append(out_f)

    return cleaned_files


def run_outlier_rejection(band, pixels, config, output_dir, cache_dir, mode="both"):
    """Run iterative outlier rejection.

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
        Final results for each solve mode.
    """
    nside = config["survey"]["nside_chunk"]
    phase1_dir = output_dir / f"phase1_{band}"
    phase2_dir = output_dir / f"phase2_{band}"
    phase3_dir = output_dir / f"phase3_{band}"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    outlier_cfg = config["outlier_rejection"]
    n_iterations = outlier_cfg["n_iterations"]
    chi2_cut = outlier_cfg["star_chi2_cut"]
    sigma_cut = outlier_cfg["detection_sigma_cut"]
    zp_cut = outlier_cfg["exposure_zp_cut"]

    # Load reference data
    print(f"  Loading reference data...", flush=True)
    node_to_idx, idx_to_node = build_node_index(
        phase1_dir / "connected_nodes.parquet"
    )
    n_params = len(node_to_idx)
    des_fgcm_zps = load_des_fgcm_zps(cache_dir, band)
    nsc_zpterms = load_nsc_zpterms(cache_dir, band)

    # Load exposure MJDs for nightly median check
    exposure_mjds = load_exposure_mjds(cache_dir, band)

    # Start with Phase 1 star lists (original, pre-rejection)
    star_list_files = sorted(
        phase1_dir.glob(f"star_lists_nside{nside}_pixel*.parquet")
    )
    print(f"  Star list files: {len(star_list_files)}", flush=True)

    # Use anchored mode for outlier detection (more stable)
    # Read initial Phase 2 solution
    initial_zp_file = phase2_dir / "zeropoints_anchored.parquet"
    if initial_zp_file.exists():
        zp_df = pd.read_parquet(initial_zp_file)
        zp_dict = dict(zip(
            zip(zp_df["expnum"].values, zp_df["ccdnum"].values),
            zp_df["zp_solved"].values,
        ))
    else:
        # If no Phase 2 output, run initial solve
        print("  No Phase 2 output found, running initial solve...", flush=True)
        AtWA, rhs, ns, np_ = accumulate_normal_equations(
            star_list_files, node_to_idx, n_params
        )
        zp_arr, info = solve_anchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps)
        zp_dict = {node: zp_arr[idx] for node, idx in node_to_idx.items()}

    # Track cumulative flagged objects
    all_flagged_stars = set()
    all_flagged_exposures = set()
    all_flagged_nodes = set()
    total_detections_flagged = 0

    iteration_stats = []
    current_star_files = star_list_files  # Start with original files

    for iteration in range(1, n_iterations + 1):
        print(f"\n  === Iteration {iteration}/{n_iterations} ===", flush=True)
        t0 = time.time()

        # 1. Compute residuals
        residuals_df = compute_residuals(current_star_files, zp_dict, node_to_idx)
        if len(residuals_df) == 0:
            print("  No residuals to compute.", flush=True)
            break

        n_stars_total = residuals_df["objectid"].nunique()
        n_det_total = len(residuals_df)
        rms_before = np.sqrt(np.mean(residuals_df["residual"].values ** 2)) * 1000

        # 2. Flag variable stars
        new_flagged_stars = flag_variable_stars(residuals_df, chi2_cut)
        new_flagged_stars -= all_flagged_stars  # Only new ones

        # 3. Flag outlier detections (counted but applied via star-level removal)
        det_mask = flag_outlier_detections(residuals_df, sigma_cut)
        n_outlier_dets = int(det_mask.sum())

        # For detection-level flagging, flag stars that have > 50% outlier detections
        outlier_df = residuals_df[det_mask]
        if len(outlier_df) > 0:
            outlier_star_counts = outlier_df.groupby("objectid").size()
            total_star_counts = residuals_df.groupby("objectid").size()
            for star, n_outlier in outlier_star_counts.items():
                if n_outlier / total_star_counts.get(star, 1) > 0.5:
                    new_flagged_stars.add(star)

        # 4. Flag bad exposures
        # Build ZP dataframe for flagging
        zp_for_flagging = pd.DataFrame([
            {"expnum": node[0], "ccdnum": node[1], "zp_solved": zp}
            for node, zp in zp_dict.items() if node in node_to_idx
        ])
        new_flagged_exp = flag_bad_exposures(
            zp_for_flagging, exposure_mjds, zp_cut
        )
        new_flagged_exp -= all_flagged_exposures

        # 5. Flag bad CCDs
        new_flagged_nodes = flag_bad_ccds(residuals_df)
        new_flagged_nodes -= all_flagged_nodes

        # Update cumulative flags
        all_flagged_stars |= new_flagged_stars
        all_flagged_exposures |= new_flagged_exp
        all_flagged_nodes |= new_flagged_nodes

        n_new_flags = len(new_flagged_stars) + len(new_flagged_exp) + len(new_flagged_nodes)

        print(f"    New flagged stars:     {len(new_flagged_stars):,}", flush=True)
        print(f"    New flagged exposures: {len(new_flagged_exp):,}", flush=True)
        print(f"    New flagged CCD-exp:   {len(new_flagged_nodes):,}", flush=True)
        print(f"    Outlier detections:    {n_outlier_dets:,}", flush=True)
        print(f"    Residual RMS:          {rms_before:.1f} mmag", flush=True)

        # 6. Apply flags and write cleaned star lists
        clean_dir = phase3_dir / f"iter{iteration}"
        cleaned_files = apply_flags_to_star_lists(
            current_star_files, all_flagged_stars, None,
            all_flagged_exposures, all_flagged_nodes, clean_dir
        )

        # 7. Re-solve on cleaned data
        # Need to rebuild node index for cleaned data (some nodes may have been removed)
        # But keep the original node index — nodes without data will just have zero weight
        print(f"    Accumulating cleaned normal equations...", flush=True)
        AtWA, rhs, n_stars_clean, n_pairs_clean = accumulate_normal_equations(
            cleaned_files, node_to_idx, n_params
        )

        print(f"    Stars: {n_stars_clean:,}, Pairs: {n_pairs_clean:,}", flush=True)

        # Solve anchored (used for next iteration's residuals)
        zp_arr, info = solve_anchored(
            AtWA, rhs, config, node_to_idx, des_fgcm_zps
        )
        zp_dict = {node: zp_arr[idx] for node, idx in node_to_idx.items()}

        # Compute post-solve residual RMS
        residuals_after = compute_residuals(cleaned_files, zp_dict, node_to_idx)
        rms_after = np.sqrt(np.mean(residuals_after["residual"].values ** 2)) * 1000 if len(residuals_after) > 0 else np.nan

        iter_time = time.time() - t0
        print(f"    Post-solve RMS:        {rms_after:.1f} mmag", flush=True)
        print(f"    CG converged:          {info['converged']}", flush=True)
        print(f"    Iteration time:        {iter_time:.1f}s", flush=True)

        iteration_stats.append({
            "iteration": iteration,
            "n_stars_flagged": len(new_flagged_stars),
            "n_exposures_flagged": len(new_flagged_exp),
            "n_nodes_flagged": len(new_flagged_nodes),
            "n_outlier_dets": n_outlier_dets,
            "rms_before_mmag": rms_before,
            "rms_after_mmag": rms_after,
            "n_stars_clean": n_stars_clean,
            "n_pairs_clean": n_pairs_clean,
            "converged": info["converged"],
        })

        # Use cleaned files for next iteration
        current_star_files = cleaned_files

        # Check convergence: no new flags
        if n_new_flags == 0:
            print(f"\n  Converged after {iteration} iterations (no new flags).",
                  flush=True)
            break

    # Save flagged objects
    pd.DataFrame({"objectid": list(all_flagged_stars)}).to_parquet(
        phase3_dir / "flagged_stars.parquet", index=False
    )
    pd.DataFrame({"expnum": list(all_flagged_exposures)}).to_parquet(
        phase3_dir / "flagged_exposures.parquet", index=False
    )
    if all_flagged_nodes:
        flagged_nodes_df = pd.DataFrame(
            list(all_flagged_nodes), columns=["expnum", "ccdnum"]
        )
    else:
        flagged_nodes_df = pd.DataFrame(columns=["expnum", "ccdnum"])
    flagged_nodes_df.to_parquet(
        phase3_dir / "flagged_nodes.parquet", index=False
    )

    # Final solve in both modes on the cleaned data
    print(f"\n  === Final solve on cleaned data ===", flush=True)
    final_star_files = current_star_files

    AtWA, rhs, n_stars_final, n_pairs_final = accumulate_normal_equations(
        final_star_files, node_to_idx, n_params
    )
    print(f"  Final stars: {n_stars_final:,}, pairs: {n_pairs_final:,}", flush=True)

    results = {}
    modes_to_run = []
    if mode in ("unanchored", "both"):
        modes_to_run.append("unanchored")
    if mode in ("anchored", "both"):
        modes_to_run.append("anchored")

    for solve_mode in modes_to_run:
        if solve_mode == "unanchored":
            zp_solved, info = solve_unanchored(
                AtWA, rhs, config, node_to_idx, des_fgcm_zps
            )
        else:
            zp_solved, info = solve_anchored(
                AtWA, rhs, config, node_to_idx, des_fgcm_zps
            )

        # Save final ZPs
        result_df = pd.DataFrame({
            "expnum": [node[0] for node in idx_to_node],
            "ccdnum": [node[1] for node in idx_to_node],
            "band": band,
            "zp_solved": zp_solved,
            "zp_fgcm": [des_fgcm_zps.get(node, np.nan) for node in idx_to_node],
        })
        result_df["delta_zp"] = result_df["zp_solved"] - result_df["zp_fgcm"]
        out_file = phase3_dir / f"zeropoints_{solve_mode}.parquet"
        result_df.to_parquet(out_file, index=False)

        # DES comparison
        des_mask = result_df["zp_fgcm"].notna()
        if des_mask.any():
            des_diff = result_df.loc[des_mask, "delta_zp"].values
            des_diff_rms = np.sqrt(np.mean(des_diff ** 2)) * 1000
            des_diff_median = np.median(des_diff) * 1000
        else:
            des_diff_rms = np.nan
            des_diff_median = np.nan

        print(f"\n  {'=' * 55}", flush=True)
        print(f"  Phase 3 Final — {band}-band ({solve_mode})", flush=True)
        print(f"  {'=' * 55}", flush=True)
        print(f"    Parameters:        {n_params:,}", flush=True)
        print(f"    Stars (cleaned):   {n_stars_final:,}", flush=True)
        print(f"    Pairs (cleaned):   {n_pairs_final:,}", flush=True)
        print(f"    Iterations:        {info['n_iterations']}", flush=True)
        print(f"    Converged:         {info['converged']}", flush=True)
        print(f"    Rel. residual:     {info['relative_residual']:.2e}", flush=True)
        print(f"    DES diff RMS:      {des_diff_rms:.1f} mmag", flush=True)
        print(f"    DES diff median:   {des_diff_median:.1f} mmag", flush=True)
        print(f"  {'=' * 55}", flush=True)

        results[solve_mode] = {
            "zp_solved": zp_solved,
            "result_df": result_df,
            "info": info,
            "des_diff_rms_mmag": des_diff_rms,
        }

    # Print iteration summary table
    print(f"\n  === Iteration Summary ===", flush=True)
    print(f"  {'Iter':>4} {'Stars':>7} {'Exps':>6} {'Nodes':>6} "
          f"{'OutlierDet':>10} {'RMS_before':>10} {'RMS_after':>10}", flush=True)
    for s in iteration_stats:
        print(f"  {s['iteration']:4d} {s['n_stars_flagged']:7d} "
              f"{s['n_exposures_flagged']:6d} {s['n_nodes_flagged']:6d} "
              f"{s['n_outlier_dets']:10d} {s['rms_before_mmag']:10.1f} "
              f"{s['rms_after_mmag']:10.1f}", flush=True)

    # Totals
    initial_rms = iteration_stats[0]["rms_before_mmag"] if iteration_stats else np.nan
    final_rms = iteration_stats[-1]["rms_after_mmag"] if iteration_stats else np.nan
    n_total_stars = iteration_stats[0].get("n_stars_clean", 0)

    print(f"\n  === Final Summary ===", flush=True)
    print(f"    Band:                     {band}", flush=True)
    print(f"    Initial residual RMS:     {initial_rms:.1f} mmag", flush=True)
    print(f"    Final residual RMS:       {final_rms:.1f} mmag", flush=True)
    print(f"    Total stars flagged:      {len(all_flagged_stars):,}", flush=True)
    print(f"    Total exposures flagged:  {len(all_flagged_exposures):,}", flush=True)
    print(f"    Total CCD-exp flagged:    {len(all_flagged_nodes):,}", flush=True)

    # Save iteration stats
    pd.DataFrame(iteration_stats).to_parquet(
        phase3_dir / "iteration_stats.parquet", index=False
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Iterative outlier rejection"
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

    print(f"Band: {args.band}, Mode: {args.mode}", flush=True)
    print(flush=True)

    results = run_outlier_rejection(
        args.band, pixels, config, output_dir, cache_dir, mode=args.mode
    )

    return results


if __name__ == "__main__":
    main()
