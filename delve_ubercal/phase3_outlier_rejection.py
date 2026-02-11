"""Phase 3: Iterative outlier rejection and re-solving.

Removes variable stars, catastrophic outliers, non-photometric exposures,
and bad CCDs, then re-runs the Phase 2 CG solve on the cleaned catalog.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from delve_ubercal.phase0_ingest import get_test_patch_pixels, get_test_region_pixels, load_config
from delve_ubercal.phase2_solve import (
    accumulate_gray_normal_equations,
    accumulate_normal_equations,
    build_node_index,
    compute_node_positions,
    compute_star_flat,
    load_des_fgcm_zps,
    load_nsc_zpterms,
    solve_anchored,
    solve_gray_unanchored,
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


def flag_bad_ccds(residuals_df, sigma_factor=3.0, min_scatter_mag=0.05):
    """Flag CCDs with anomalously large intra-CCD scatter.

    Uses max(relative_threshold, min_scatter_mag) to prevent
    over-flagging when the solution is tight.

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

    relative_threshold = median_scatter + sigma_factor * sigma_est
    threshold = max(relative_threshold, min_scatter_mag)
    bad = node_stats[node_stats["std"] > threshold]
    return set(zip(bad["expnum"].values, bad["ccdnum"].values))


def apply_flags_to_star_lists(star_list_files, flagged_stars, flagged_det_keys,
                               flagged_exposures, flagged_nodes, output_dir):
    """Remove flagged detections and write cleaned star lists.

    Parameters
    ----------
    flagged_stars : set
        Objectids to remove entirely.
    flagged_det_keys : set or None
        (objectid, expnum, ccdnum) tuples for individual detection removal.
    flagged_exposures : set
        Expnums to remove entirely.
    flagged_nodes : set
        (expnum, ccdnum) tuples to remove entirely.

    Returns list of cleaned file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files = []

    for f in star_list_files:
        df = pd.read_parquet(f)
        if len(df) == 0:
            out_f = output_dir / f.name
            df.to_parquet(out_f, index=False)
            cleaned_files.append(out_f)
            continue

        # Remove flagged stars
        if flagged_stars:
            df = df[~df["objectid"].isin(flagged_stars)]

        # Remove individual flagged detections
        if flagged_det_keys:
            det_keys = set(zip(
                df["objectid"].values, df["expnum"].values, df["ccdnum"].values
            ))
            bad_det_mask = np.array([
                (o, e, c) in flagged_det_keys
                for o, e, c in zip(df["objectid"].values, df["expnum"].values, df["ccdnum"].values)
            ])
            df = df[~bad_det_mask]

        # Remove flagged exposures
        if flagged_exposures:
            df = df[~df["expnum"].isin(flagged_exposures)]

        # Remove flagged nodes (bad CCDs)
        if flagged_nodes:
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
    all_flagged_det_keys = set()  # (objectid, expnum, ccdnum) tuples
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

        # 2. Flag individual outlier detections (not whole stars)
        det_mask = flag_outlier_detections(residuals_df, sigma_cut)
        n_outlier_dets = int(det_mask.sum())

        # Collect flagged (objectid, expnum, ccdnum) tuples for detection-level removal
        outlier_dets = residuals_df[det_mask]
        new_flagged_det_keys = set(zip(
            outlier_dets["objectid"].values,
            outlier_dets["expnum"].values,
            outlier_dets["ccdnum"].values,
        ))
        all_flagged_det_keys |= new_flagged_det_keys

        # Flag whole stars ONLY if they have chi2/dof > threshold AND <= 2 clean
        # detections remaining (i.e., useless after outlier removal)
        new_flagged_stars = set()
        high_chi2_stars = flag_variable_stars(residuals_df, chi2_cut)
        high_chi2_stars -= all_flagged_stars
        if high_chi2_stars:
            total_star_counts = residuals_df.groupby("objectid").size()
            outlier_star_counts = outlier_dets.groupby("objectid").size() if len(outlier_dets) > 0 else pd.Series(dtype=int)
            for star in high_chi2_stars:
                n_total = total_star_counts.get(star, 0)
                n_bad = outlier_star_counts.get(star, 0)
                n_clean = n_total - n_bad
                if n_clean < 2:
                    # Star has < 2 clean detections — flag it entirely
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
            current_star_files, all_flagged_stars, all_flagged_det_keys,
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

        # Check convergence: no new flags or RMS worsened
        if n_new_flags == 0:
            print(f"\n  Converged after {iteration} iterations (no new flags).",
                  flush=True)
            break

        # Early stop if RMS is getting worse (instability)
        if len(iteration_stats) >= 2:
            prev_rms = iteration_stats[-2]["rms_after_mmag"]
            curr_rms = iteration_stats[-1]["rms_after_mmag"]
            if curr_rms > prev_rms * 1.5:
                print(f"\n  Early stop: RMS worsened ({prev_rms:.1f} -> {curr_rms:.1f} mmag).",
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

            # Star flat refinement (decompose ZP into gray + flat)
            sf_enabled = config.get("starflat", {}).get("enabled", False)
            if sf_enabled:
                print(f"\n    --- Star flat refinement ---", flush=True)

                # Compute star flat from per-CCD ZPs (excluding flagged)
                star_flat, epoch_bounds = compute_star_flat(
                    zp_solved, idx_to_node, exposure_mjds, config,
                    flagged_nodes=all_flagged_nodes,
                    flagged_exposures=all_flagged_exposures,
                )

                # Build exposure index (exclude flagged)
                exp_to_idx = {}
                for expnum, ccdnum in idx_to_node:
                    if (expnum, ccdnum) in all_flagged_nodes:
                        continue
                    if expnum in all_flagged_exposures:
                        continue
                    if expnum not in exp_to_idx:
                        exp_to_idx[expnum] = len(exp_to_idx)
                n_exp_params = len(exp_to_idx)
                print(f"    Exposure params: {n_exp_params:,} "
                      f"(vs {n_params:,} CCD-exposures)", flush=True)

                # Build gray normal equations
                AtWA_gray, rhs_gray, n_stars_gray, n_pairs_gray, n_intra = \
                    accumulate_gray_normal_equations(
                        final_star_files, exp_to_idx, n_exp_params,
                        star_flat, exposure_mjds, epoch_bounds,
                    )
                print(f"    Gray: {n_stars_gray:,} stars, {n_pairs_gray:,} pairs, "
                      f"{n_intra:,} intra-exp skipped", flush=True)

                # Load node positions for gradient detrend
                phase0_dir = output_dir / f"phase0_{band}"
                node_pos_dict = None
                if config.get("solve", {}).get("gradient_detrend", False):
                    node_pos_dict = {}
                    pos_file = phase0_dir / "node_positions.parquet"
                    if pos_file.exists():
                        pos_df = pd.read_parquet(pos_file)
                        for _, row in pos_df.iterrows():
                            node_pos_dict[(int(row["expnum"]), int(row["ccdnum"]))] = (
                                row["ra_mean"], row["dec_mean"]
                            )

                # Solve for gray terms
                zp_sf, gray_solved, gray_info = solve_gray_unanchored(
                    AtWA_gray, rhs_gray, config, exp_to_idx, des_fgcm_zps,
                    star_flat, exposure_mjds, epoch_bounds,
                    node_to_idx, idx_to_node,
                    node_positions=node_pos_dict,
                )

                print(f"    Gray CG: converged={gray_info['converged']}, "
                      f"iters={gray_info['n_iterations']}, "
                      f"residual={gray_info['relative_residual']:.2e}", flush=True)

                if gray_info["converged"]:
                    # NOTE: Gradient detrend is applied as a separate post-Phase 3 step
                    # (after all bands complete) to ensure consistent DELVE g-r color terms.
                    # See apply_gradient_detrend() in phase2_solve.py.

                    # Save star flat for diagnostics
                    sf_records = [
                        {"ccdnum": k[0], "epoch_idx": k[1], "flat_correction": v}
                        for k, v in star_flat.items()
                    ]
                    pd.DataFrame(sf_records).to_parquet(
                        phase3_dir / "star_flat.parquet", index=False,
                    )

                    # Replace per-CCD solution with star flat version
                    zp_solved = zp_sf
                    info["mode"] = "unanchored_starflat"
                    info["n_exp_params"] = gray_info["n_exp_params"]
                    info["gray_iterations"] = gray_info["n_iterations"]
                    info["gray_converged"] = gray_info["converged"]
                else:
                    print(f"    Star flat CG FAILED, keeping per-CCD result",
                          flush=True)

            # Spatial detrending disabled — it soft-anchors to FGCM which
            # won't work outside the DES footprint.  The unanchored solution
            # with ~20 mmag large-scale modes is the honest representation of
            # overlap-only calibration quality.
            # To re-enable: uncomment the block below.
            # phase0_dir = output_dir / f"phase0_{band}"
            # node_positions = compute_node_positions(phase0_dir, nside)
            # grad_info = detrend_spatial_gradient(
            #     zp_solved, idx_to_node, node_positions, des_fgcm_zps,
            # )
            # info["detrend"] = grad_info
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

        # DES comparison (filter sentinel FGCM values AND flagged nodes)
        # Flagged nodes have no data in the solve and get Tikhonov-default ZPs
        flagged_node_mask = np.array([
            (e, c) in all_flagged_nodes or e in all_flagged_exposures
            for e, c in zip(result_df["expnum"].values, result_df["ccdnum"].values)
        ])
        des_mask = (result_df["zp_fgcm"].notna()
                    & (result_df["zp_fgcm"] > 25.0) & (result_df["zp_fgcm"] < 35.0)
                    & ~flagged_node_mask)
        n_des_clean = int(des_mask.sum())
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
        print(f"    DES nodes (clean): {n_des_clean:,}", flush=True)
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

    print(f"Band: {args.band}, Mode: {args.mode}", flush=True)
    print(flush=True)

    results = run_outlier_rejection(
        args.band, pixels, config, output_dir, cache_dir, mode=args.mode
    )

    return results


if __name__ == "__main__":
    main()
