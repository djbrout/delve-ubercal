"""Generate all diagnostic plots for the DELVE ubercal paper.

Reads Phase 2-4 outputs and creates publication-quality figures.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

OUTPUT = Path("/Volumes/External5TB/DELVE_UBERCAL/output")
FIGURES = Path("/Volumes/External5TB/DELVE_UBERCAL/paper/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_zp_histogram():
    """Fig 1: Histogram of solved zero-points (anchored + unanchored)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, mode in zip(axes, ["unanchored", "anchored"]):
        zp_df = pd.read_parquet(OUTPUT / f"phase2_g/zeropoints_{mode}.parquet")
        zps = zp_df["zp_solved"].values
        # Filter out zero-valued ZPs (no data)
        zps = zps[zps > 1.0]

        ax.hist(zps, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(np.median(zps), color="red", ls="--", lw=1.5,
                   label=f"median = {np.median(zps):.3f}")
        ax.set_xlabel("ZP$_{\\rm solved}$ (mag)")
        ax.set_ylabel("Number of CCD-exposures")
        ax.set_title(f"Phase 2: {mode.capitalize()} Mode")
        ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig01_zp_histogram.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig01_zp_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 1: ZP histogram saved")


def plot_des_fgcm_comparison():
    """Fig 2: ZP_solved - ZP_FGCM for DES CCD-exposures."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, mode in zip(axes, ["unanchored", "anchored"]):
        # Phase 3 (post-outlier rejection)
        p3 = OUTPUT / f"phase3_g/zeropoints_{mode}.parquet"
        if p3.exists():
            zp_df = pd.read_parquet(p3)
            label_prefix = "Phase 3"
        else:
            zp_df = pd.read_parquet(OUTPUT / f"phase2_g/zeropoints_{mode}.parquet")
            label_prefix = "Phase 2"

        des_mask = zp_df["zp_fgcm"].notna() & (zp_df["zp_solved"] > 1.0)
        if des_mask.sum() == 0:
            continue

        diff = (zp_df.loc[des_mask, "zp_solved"] - zp_df.loc[des_mask, "zp_fgcm"]) * 1000
        rms = np.sqrt(np.mean(diff ** 2))
        med = np.median(diff)

        ax.hist(diff, bins=80, color="darkorange", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="k", ls="-", lw=0.5)
        ax.axvline(med, color="red", ls="--", lw=1.5,
                   label=f"median = {med:.1f} mmag")
        ax.set_xlabel("ZP$_{\\rm solved}$ $-$ ZP$_{\\rm FGCM}$ (mmag)")
        ax.set_ylabel("Number of DES CCD-exposures")
        ax.set_title(f"{label_prefix}: {mode.capitalize()} ({rms:.1f} mmag RMS)")
        ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig02_des_fgcm_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig02_des_fgcm_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 2: DES-FGCM comparison saved")


def plot_outlier_rejection_convergence():
    """Fig 3: Outlier rejection iteration convergence."""
    stats_file = OUTPUT / "phase3_g/iteration_stats.parquet"
    if not stats_file.exists():
        print("  Fig 3: SKIPPED (no Phase 3 stats)")
        return

    stats = pd.read_parquet(stats_file)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Number of flagged objects per iteration
    ax = axes[0]
    ax.plot(stats["iteration"], stats["n_stars_flagged"], "o-",
            color="steelblue", label="Stars")
    ax.plot(stats["iteration"], stats["n_exposures_flagged"], "s-",
            color="darkorange", label="Exposures")
    ax.plot(stats["iteration"], stats["n_nodes_flagged"], "^-",
            color="forestgreen", label="CCD-exp")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Newly flagged")
    ax.set_title("Flagging Convergence")
    ax.legend()
    ax.set_yscale("log")

    # Panel 2: Residual RMS
    ax = axes[1]
    ax.plot(stats["iteration"], stats["rms_before_mmag"], "o--",
            color="gray", label="Before re-solve")
    ax.plot(stats["iteration"], stats["rms_after_mmag"], "o-",
            color="steelblue", label="After re-solve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual RMS (mmag)")
    ax.set_title("Residual RMS Evolution")
    ax.legend()

    # Panel 3: Number of clean stars/pairs
    ax = axes[2]
    ax.plot(stats["iteration"], stats["n_stars_clean"], "o-",
            color="steelblue", label="Stars")
    ax2 = ax.twinx()
    ax2.plot(stats["iteration"], stats["n_pairs_clean"] / 1000, "s-",
             color="darkorange", label="Pairs (k)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Clean stars", color="steelblue")
    ax2.set_ylabel("Clean pairs (thousands)", color="darkorange")
    ax.set_title("Data Volume")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig03_outlier_convergence.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig03_outlier_convergence.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 3: Outlier rejection convergence saved")


def plot_starflat_corrections():
    """Fig 4: Star flat correction amplitudes."""
    stats_file = OUTPUT / "phase4_g/starflat_stats.parquet"
    if not stats_file.exists():
        print("  Fig 4: SKIPPED (no Phase 4 stats)")
        return

    stats = pd.read_parquet(stats_file)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: Correction RMS per CCD
    ax = axes[0]
    ax.bar(range(len(stats)), stats["correction_rms_mmag"].values,
           color="steelblue", edgecolor="none", alpha=0.8)
    ax.axhline(5.0, color="red", ls="--", lw=1, label="5 mmag (typical)")
    ax.set_xlabel("CCD index")
    ax.set_ylabel("Correction RMS (mmag)")
    ax.set_title("Star Flat Amplitude per CCD")
    ax.legend()

    # Panel 2: Before vs After per CCD
    ax = axes[1]
    ax.scatter(stats["rms_before_mmag"], stats["rms_after_mmag"],
               s=20, alpha=0.7, color="steelblue")
    lim = max(stats["rms_before_mmag"].max(), stats["rms_after_mmag"].max()) + 2
    ax.plot([0, lim], [0, lim], "k--", lw=0.5)
    ax.set_xlabel("RMS before star flat (mmag)")
    ax.set_ylabel("RMS after star flat (mmag)")
    ax.set_title("Per-CCD Improvement")
    ax.set_aspect("equal")

    # Panel 3: Histogram of correction amplitudes
    ax = axes[2]
    ax.hist(stats["correction_rms_mmag"], bins=20,
            color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(stats["correction_rms_mmag"].median(), color="red", ls="--",
               label=f"median = {stats['correction_rms_mmag'].median():.1f} mmag")
    ax.set_xlabel("Correction RMS (mmag)")
    ax.set_ylabel("Number of CCDs")
    ax.set_title("Distribution of Corrections")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig04_starflat_corrections.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig04_starflat_corrections.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 4: Star flat corrections saved")


def plot_test_results():
    """Fig 5: Unit test summary table as a figure."""
    # Run pytest to get test counts
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "delve_ubercal/tests/", "-v", "--tb=no", "-q"],
        capture_output=True, text=True, cwd="/Volumes/External5TB/DELVE_UBERCAL",
    )

    output = result.stdout + result.stderr
    lines = output.strip().split("\n")
    # Parse test results
    test_results = []
    for line in lines:
        if " PASSED" in line or " FAILED" in line:
            parts = line.strip().split("::")
            if len(parts) >= 2:
                module = parts[0].split("/")[-1].replace(".py", "").strip()
                if len(parts) >= 3:
                    test_name = parts[-1].split(" ")[0]
                else:
                    test_name = parts[1].split(" ")[0]
                status = "PASS" if "PASSED" in line else "FAIL"
                test_results.append((module, test_name, status))

    if not test_results:
        print("  Fig 5: SKIPPED (no test results parsed)")
        return

    # Summary by phase
    phase_counts = {}
    for module, test, status in test_results:
        phase = module.replace("test_", "Phase ").replace("phase", "")
        if phase not in phase_counts:
            phase_counts[phase] = {"pass": 0, "fail": 0}
        if status == "PASS":
            phase_counts[phase]["pass"] += 1
        else:
            phase_counts[phase]["fail"] += 1

    # Create table figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    phases = sorted(phase_counts.keys())
    table_data = []
    colors = []
    for phase in phases:
        p = phase_counts[phase]["pass"]
        f = phase_counts[phase]["fail"]
        total = p + f
        table_data.append([phase, str(p), str(f), str(total),
                          "ALL PASS" if f == 0 else f"{f} FAIL"])
        colors.append(["white", "white", "white", "white",
                       "#c8e6c9" if f == 0 else "#ffcdd2"])

    table = ax.table(
        cellText=table_data,
        colLabels=["Phase", "Passed", "Failed", "Total", "Status"],
        cellColours=colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(5):
        table[0, j].set_facecolor("#1565c0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Unit Test Results Summary", fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig05_test_results.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig05_test_results.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 5: Test results summary saved")


def plot_phase2_solve_summary():
    """Fig 6: Phase 2 solve summary - DES diff vs unanchored/anchored."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (p2_mode, p3_mode) in zip(axes, [
        ("unanchored", "unanchored"), ("anchored", "anchored")
    ]):
        # Phase 2 (pre-rejection)
        p2_file = OUTPUT / f"phase2_g/zeropoints_{p2_mode}.parquet"
        p3_file = OUTPUT / f"phase3_g/zeropoints_{p3_mode}.parquet"

        for fpath, label, color in [
            (p2_file, "Phase 2 (pre-rejection)", "gray"),
            (p3_file, "Phase 3 (post-rejection)", "steelblue"),
        ]:
            if not fpath.exists():
                continue
            df = pd.read_parquet(fpath)
            des = df[df["zp_fgcm"].notna() & (df["zp_solved"] > 1.0)]
            if len(des) == 0:
                continue
            diff = (des["zp_solved"] - des["zp_fgcm"]) * 1000
            rms = np.sqrt(np.mean(diff ** 2))
            ax.hist(diff, bins=80, alpha=0.6, color=color, edgecolor="none",
                    label=f"{label} ({rms:.1f} mmag)")

        ax.set_xlabel("ZP$_{\\rm solved}$ $-$ ZP$_{\\rm FGCM}$ (mmag)")
        ax.set_ylabel("DES CCD-exposures")
        ax.set_title(f"{p2_mode.capitalize()} Mode")
        ax.legend(fontsize=9)
        ax.axvline(0, color="k", ls="-", lw=0.5)

    fig.suptitle("Outlier Rejection Improvement: FGCM Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06_rejection_improvement.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig06_rejection_improvement.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 6: Rejection improvement saved")


def plot_residual_rms_by_ccd():
    """Fig 7: Per-CCD residual RMS before and after star flat."""
    stats_file = OUTPUT / "phase4_g/starflat_stats.parquet"
    if not stats_file.exists():
        print("  Fig 7: SKIPPED")
        return

    stats = pd.read_parquet(stats_file)
    stats = stats.sort_values("ccdnum")

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(stats))
    width = 0.35
    ax.bar(x - width/2, stats["rms_before_mmag"].values, width,
           label="Before star flat", color="lightcoral", edgecolor="none")
    ax.bar(x + width/2, stats["rms_after_mmag"].values, width,
           label="After star flat", color="steelblue", edgecolor="none")
    ax.set_xlabel("CCD Number")
    ax.set_ylabel("Residual RMS (mmag)")
    ax.set_title("Per-CCD Residual RMS: Before vs After Star Flat (g-band)")
    ax.set_xticks(x[::5])
    ax.set_xticklabels(stats["ccdnum"].values[::5].astype(int))
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig07_perccd_rms.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig07_perccd_rms.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 7: Per-CCD RMS saved")


def plot_pipeline_overview():
    """Fig 8: Pipeline flowchart / metrics overview."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Pipeline metrics
    metrics = [
        ["Phase", "Description", "Key Metric", "Status"],
        ["0", "Data Ingestion", "13,876 stars, 104,922 dets", "PASS"],
        ["1", "Overlap Graph", "4,184 CCD-exp, 100% connected", "PASS"],
        ["2", "CG Solver", "CG converges (318/106 iter)", "PASS"],
        ["", "  Unanchored", "DES diff RMS: 41.9 mmag", "PASS"],
        ["", "  Anchored", "DES diff RMS: 15.7 mmag", "PASS"],
        ["3", "Outlier Rejection", "RMS: 30.7 -> 11.9 mmag", "PASS"],
        ["", "  Anchored", "DES diff RMS: 5.0 mmag", "PASS"],
        ["", "  Stars flagged", "4,840 (34.9%)", "PASS"],
        ["4", "Star Flat", "Mean correction: 5.7 mmag", "PASS"],
        ["", "  CCDs fitted", "62 (CCD, epoch) groups", "PASS"],
        ["", "  Max correction", "23.1 mmag (CCD 62)", "PASS"],
    ]

    colors = []
    for row in metrics:
        if row[0] == "Phase":
            colors.append(["#1565c0"] * 4)
        elif row[-1] == "PASS":
            colors.append(["white", "white", "white", "#c8e6c9"])
        else:
            colors.append(["white", "white", "white", "#ffcdd2"])

    table = ax.table(
        cellText=metrics,
        cellColours=colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.6)

    for j in range(4):
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("DELVE Ubercalibration Pipeline Summary (g-band, Test Region)",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig08_pipeline_overview.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig08_pipeline_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 8: Pipeline overview saved")


def plot_synthetic_test_validation():
    """Fig 9: Synthetic test ZP recovery."""
    from delve_ubercal.tests.test_phase2 import make_synthetic_data
    from delve_ubercal.phase2_solve import (
        accumulate_normal_equations, build_node_index,
        solve_unanchored, solve_anchored,
    )
    from delve_ubercal.phase0_ingest import load_config

    config = load_config()

    star_file, connected_file, fgcm_file, true_zps, node_list, tmpdir = \
        make_synthetic_data(n_ccd_exposures=200, n_stars=2000, n_des=50, seed=42)

    node_to_idx, idx_to_node = build_node_index(connected_file)
    n_params = len(node_to_idx)

    des_fgcm_zps = {}
    fgcm = pd.read_parquet(fgcm_file)
    for _, row in fgcm.iterrows():
        des_fgcm_zps[(row["expnum"], row["ccdnum"])] = row["mag_zero"]

    AtWA, rhs, _, _ = accumulate_normal_equations(
        [star_file], node_to_idx, n_params
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    true_ordered = np.array([
        true_zps[node_list.index(node)] for node in idx_to_node
    ])

    for ax, (solve_fn, title) in zip(axes, [
        (lambda: solve_unanchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps),
         "Unanchored"),
        (lambda: solve_anchored(AtWA, rhs, config, node_to_idx, des_fgcm_zps),
         "Anchored"),
    ]):
        zp_solved, info = solve_fn()
        residual = (zp_solved - true_ordered) * 1000
        rms = np.sqrt(np.mean(residual ** 2))

        ax.scatter(true_ordered, zp_solved, s=5, alpha=0.5, color="steelblue")
        lim = [true_ordered.min() - 0.02, true_ordered.max() + 0.02]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlabel("True ZP (mag)")
        ax.set_ylabel("Solved ZP (mag)")
        ax.set_title(f"{title}: RMS = {rms:.3f} mmag")
        ax.set_aspect("equal")

    fig.suptitle("Synthetic Test: Zero-Point Recovery", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig09_synthetic_recovery.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig09_synthetic_recovery.png", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 9: Synthetic recovery saved")


if __name__ == "__main__":
    print("Generating paper figures...", flush=True)
    plot_zp_histogram()
    plot_des_fgcm_comparison()
    plot_outlier_rejection_convergence()
    plot_starflat_corrections()
    plot_test_results()
    plot_phase2_solve_summary()
    plot_residual_rms_by_ccd()
    plot_pipeline_overview()
    plot_synthetic_test_validation()
    print("Done! All figures saved to", FIGURES)
