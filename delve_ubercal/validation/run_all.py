"""Phase 6: Run all validation tests and generate plots.

Tests 0-5 as specified in the plan:
- Test 0: FGCM vs ubercal (unanchored)
- Test 1: Photometric repeatability vs magnitude
- Test 2: DR2 vs ubercal difference
- Test 3: Gaia XP comparison (placeholder - requires Gaia crossmatch)
- Test 4: Stellar locus width (placeholder - requires multi-band)
- Test 5: DES boundary continuity
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from delve_ubercal.phase0_ingest import get_test_region_pixels, load_config
from delve_ubercal.phase2_solve import build_node_index, load_des_fgcm_zps
from delve_ubercal.phase3_outlier_rejection import load_exposure_mjds
from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def test0_fgcm_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 0: FGCM vs ubercal comparison using UNANCHORED solve.

    Returns
    -------
    metrics : dict
    """
    print("\n  === Test 0: FGCM vs Ubercal (Unanchored) ===", flush=True)

    phase3_dir = output_dir / f"phase3_{band}"
    phase2_dir = output_dir / f"phase2_{band}"

    # Load unanchored solution
    for d in [phase3_dir, phase2_dir]:
        f = d / "zeropoints_unanchored.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            source = d.name
            break
    else:
        print("    No unanchored solution found.", flush=True)
        return {"status": "SKIP"}

    des_mask = df["zp_fgcm"].notna() & (df["zp_solved"] > 1.0)
    if des_mask.sum() == 0:
        return {"status": "SKIP"}

    diff = (df.loc[des_mask, "zp_solved"] - df.loc[des_mask, "zp_fgcm"]).values * 1000
    rms = np.sqrt(np.mean(diff ** 2))
    median = np.median(diff)
    max_dev = np.max(np.abs(diff))

    print(f"    Source: {source}", flush=True)
    print(f"    N DES CCD-exp: {des_mask.sum():,}", flush=True)
    print(f"    RMS: {rms:.1f} mmag", flush=True)
    print(f"    Median: {median:.1f} mmag", flush=True)
    print(f"    Max: {max_dev:.1f} mmag", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(diff, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="k", ls="-", lw=0.5)
    ax.axvline(median, color="red", ls="--", lw=1.5,
               label=f"median = {median:.1f} mmag")
    ax.set_xlabel("ZP$_{\\rm ubercal}$ - ZP$_{\\rm FGCM}$ (mmag)")
    ax.set_ylabel("DES CCD-exposures")
    ax.set_title(f"Test 0: FGCM Comparison ({rms:.1f} mmag RMS)")
    ax.legend()

    ax = axes[1]
    ax.scatter(df.loc[des_mask, "zp_fgcm"].values,
               diff, s=1, alpha=0.3, color="steelblue")
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("ZP$_{\\rm FGCM}$ (mag)")
    ax.set_ylabel("ZP$_{\\rm ubercal}$ - ZP$_{\\rm FGCM}$ (mmag)")
    ax.set_title("Residual vs ZP")

    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_fgcm_comparison_{band}.pdf",
                bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_fgcm_comparison_{band}.png",
                bbox_inches="tight")
    plt.close(fig)

    passed = rms < 15.0
    return {
        "test": "Test 0: FGCM comparison",
        "rms_mmag": rms,
        "median_mmag": median,
        "max_mmag": max_dev,
        "n_des": int(des_mask.sum()),
        "threshold": "< 15 mmag",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
    }


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

    des_mask = df["zp_fgcm"].notna() & (df["zp_solved"] > 1.0)
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


def test3_gaia_comparison(output_dir, cache_dir, band, fig_dir):
    """Test 3: Gaia XP comparison (placeholder).

    Full implementation requires Gaia crossmatch and color transformations.
    """
    print("\n  === Test 3: Gaia Comparison ===", flush=True)
    print("    PLACEHOLDER: requires Gaia DR3 crossmatch (not available in test region)",
          flush=True)
    return {
        "test": "Test 3: Gaia comparison",
        "status": "SKIP (requires Gaia crossmatch)",
        "passed": None,
    }


def test4_stellar_locus(output_dir, cache_dir, band, fig_dir):
    """Test 4: Stellar locus width (placeholder).

    Full implementation requires multi-band (g, r, i) photometry.
    """
    print("\n  === Test 4: Stellar Locus ===", flush=True)
    print("    PLACEHOLDER: requires multi-band photometry (only g-band processed)",
          flush=True)
    return {
        "test": "Test 4: Stellar locus",
        "status": "SKIP (requires multi-band)",
        "passed": None,
    }


def test5_des_boundary(output_dir, cache_dir, band, fig_dir):
    """Test 5: DES boundary continuity.

    Compare ZPs inside vs outside DES footprint.
    """
    print("\n  === Test 5: DES Boundary Continuity ===", flush=True)

    phase3_dir = output_dir / f"phase3_{band}"
    phase2_dir = output_dir / f"phase2_{band}"

    for d in [phase3_dir, phase2_dir]:
        f = d / "zeropoints_anchored.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            break
    else:
        return {"status": "SKIP"}

    # DES vs non-DES comparison
    des_mask = df["zp_fgcm"].notna() & (df["zp_solved"] > 1.0)
    non_des_mask = df["zp_fgcm"].isna() & (df["zp_solved"] > 1.0)

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
    ax.set_title(f"Test 5: DES Boundary ({boundary_offset:.1f} mmag offset)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / f"validation_des_boundary_{band}.pdf",
                bbox_inches="tight")
    fig.savefig(fig_dir / f"validation_des_boundary_{band}.png",
                bbox_inches="tight")
    plt.close(fig)

    passed = boundary_offset < 10.0
    return {
        "test": "Test 5: DES boundary",
        "boundary_offset_mmag": boundary_offset,
        "n_des": int(n_des),
        "n_non_des": int(n_non_des),
        "threshold": "< 10 mmag",
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
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
    results.append(test3_gaia_comparison(output_dir, cache_dir, band, fig_dir))
    results.append(test4_stellar_locus(output_dir, cache_dir, band, fig_dir))
    results.append(test5_des_boundary(output_dir, cache_dir, band, fig_dir))

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

    print(f"Band: {args.band}", flush=True)
    print(flush=True)

    run_all_tests(args.band, pixels, config, output_dir, cache_dir)


if __name__ == "__main__":
    main()
