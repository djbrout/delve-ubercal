#!/usr/bin/env python3
"""Diagnostic: Check whether the RA gradient is in the ubercal solution or in the DES FGCM reference.

Compares ZP_ubercal, ZP_before (NSC zpterm), and ZP_FGCM as functions of RA
for DES CCD-exposures in the g-band test patch.
"""

import sys
sys.path.insert(0, "/Volumes/External5TB/DELVE_UBERCAL")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from delve_ubercal.phase2_solve import load_des_fgcm_zps, load_nsc_zpterms

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE = Path("/Volumes/External5TB/DELVE_UBERCAL")
CACHE = BASE / "cache"
OUTPUT = BASE / "output"
BAND = "g"

# -------------------------------------------------------------------
# 1. Load Phase 3 unanchored ZPs
# -------------------------------------------------------------------
print("Loading Phase 3 unanchored ZPs...")
zp_df = pd.read_parquet(OUTPUT / f"phase3_{BAND}" / "zeropoints_unanchored.parquet")
print(f"  {len(zp_df)} CCD-exposures total")

# Load flagged nodes to exclude them
flagged_nodes = pd.read_parquet(OUTPUT / f"phase3_{BAND}" / "flagged_nodes.parquet")
flagged_set = set(zip(flagged_nodes["expnum"].values, flagged_nodes["ccdnum"].values))
print(f"  {len(flagged_set)} flagged nodes to exclude")

# Build ZP_ubercal dict (excluding flagged)
zp_ubercal = {}
for _, row in zp_df.iterrows():
    key = (row["expnum"], row["ccdnum"])
    if key not in flagged_set:
        zp_ubercal[key] = row["zp_solved"]
print(f"  {len(zp_ubercal)} unflagged ZP_ubercal values")

# -------------------------------------------------------------------
# 2. Load DES FGCM ZPs
# -------------------------------------------------------------------
print("Loading DES FGCM ZPs...")
fgcm_dict = load_des_fgcm_zps(CACHE, BAND)
print(f"  {len(fgcm_dict)} DES FGCM entries")

# -------------------------------------------------------------------
# 3. Load NSC zpterm values
# -------------------------------------------------------------------
print("Loading NSC zpterms...")
zpterm_dict = load_nsc_zpterms(CACHE, BAND)
print(f"  {len(zpterm_dict)} NSC zpterm entries")

# -------------------------------------------------------------------
# 4. Load node positions (RA, Dec)
# -------------------------------------------------------------------
print("Loading node positions...")
pos_df = pd.read_parquet(OUTPUT / f"phase0_{BAND}" / "node_positions.parquet")
pos_dict = {}
for _, row in pos_df.iterrows():
    pos_dict[(row["expnum"], row["ccdnum"])] = (row["ra_mean"], row["dec_mean"])
print(f"  {len(pos_dict)} node positions")

# -------------------------------------------------------------------
# 5. Compute MAGZERO_offset = median(ZP_FGCM - zpterm) for DES CCD-exps
# -------------------------------------------------------------------
print("\nComputing MAGZERO_offset...")
diffs_for_offset = []
for key, zp_fgcm in fgcm_dict.items():
    if not (25.0 < zp_fgcm < 35.0):
        continue
    if key in zpterm_dict:
        diffs_for_offset.append(zp_fgcm - zpterm_dict[key])

MAGZERO_offset = np.median(diffs_for_offset)
print(f"  MAGZERO_offset = {MAGZERO_offset:.4f} (from {len(diffs_for_offset)} DES CCD-exps)")
print(f"  Mean: {np.mean(diffs_for_offset):.4f}, Std: {np.std(diffs_for_offset):.4f}")

# -------------------------------------------------------------------
# 6. Build merged table for DES CCD-exposures with valid FGCM
# -------------------------------------------------------------------
print("\nBuilding merged table...")
rows = []
for key, zp_fgcm in fgcm_dict.items():
    if not (25.0 < zp_fgcm < 35.0):
        continue
    if key not in zp_ubercal:
        continue
    if key not in zpterm_dict:
        continue
    if key not in pos_dict:
        continue

    ra, dec = pos_dict[key]
    zp_ub = zp_ubercal[key]
    zp_before = zpterm_dict[key] + MAGZERO_offset
    rows.append({
        "expnum": key[0],
        "ccdnum": key[1],
        "ra": ra,
        "dec": dec,
        "zp_ubercal": zp_ub,
        "zp_fgcm": zp_fgcm,
        "zp_before": zp_before,
        "zpterm": zpterm_dict[key],
    })

df = pd.DataFrame(rows)
print(f"  {len(df)} DES CCD-exposures with all data available (before quality cut)")
print(f"  RA range: {df['ra'].min():.1f} -- {df['ra'].max():.1f}")
print(f"  Dec range: {df['dec'].min():.1f} -- {df['dec'].max():.1f}")

# -------------------------------------------------------------------
# 6b. Quality cut: remove ZP_ubercal outliers (solver failures at 0.0, etc.)
# -------------------------------------------------------------------
# ZP_ubercal should be ~31.4; anything below 25 is a solver failure
bad_mask = (df["zp_ubercal"] < 25.0) | (df["zp_ubercal"] > 35.0)
n_bad = bad_mask.sum()
print(f"  Removing {n_bad} CCD-exps with ZP_ubercal outside [25, 35]")
df = df[~bad_mask].copy()
print(f"  {len(df)} DES CCD-exposures after quality cut")

# -------------------------------------------------------------------
# 7. Compute residuals
# -------------------------------------------------------------------
df["ubercal_minus_fgcm"] = df["zp_ubercal"] - df["zp_fgcm"]
df["before_minus_fgcm"] = df["zp_before"] - df["zp_fgcm"]
df["ubercal_minus_before"] = df["zp_ubercal"] - df["zp_before"]
df["fgcm_minus_zpterm"] = df["zp_fgcm"] - df["zpterm"]  # should be ~MAGZERO_offset

# Convert to mmag for plotting
for col in ["ubercal_minus_fgcm", "before_minus_fgcm", "ubercal_minus_before"]:
    df[col + "_mmag"] = df[col] * 1000.0

# -------------------------------------------------------------------
# 8. Compute binned medians and linear fits
# -------------------------------------------------------------------
def binned_median(ra, val, bin_width=1.0):
    """Compute binned median of val vs ra."""
    ra_min, ra_max = ra.min(), ra.max()
    edges = np.arange(ra_min, ra_max + bin_width, bin_width)
    bin_centers = []
    bin_medians = []
    bin_counts = []
    for i in range(len(edges) - 1):
        mask = (ra >= edges[i]) & (ra < edges[i + 1])
        if mask.sum() >= 10:
            bin_centers.append((edges[i] + edges[i + 1]) / 2)
            bin_medians.append(np.median(val[mask]))
            bin_counts.append(mask.sum())
    return np.array(bin_centers), np.array(bin_medians), np.array(bin_counts)


def linear_fit(ra, val):
    """Return slope, intercept of linear fit."""
    coeffs = np.polyfit(ra, val, 1)
    return coeffs[0], coeffs[1]  # slope, intercept


# -------------------------------------------------------------------
# 9. Report statistics
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("GRADIENT ANALYSIS (g-band, DES CCD-exposures, unanchored)")
print("=" * 70)

quantities = [
    ("ZP_ubercal - ZP_FGCM", "ubercal_minus_fgcm_mmag"),
    ("ZP_before - ZP_FGCM", "before_minus_fgcm_mmag"),
    ("ZP_ubercal - ZP_before", "ubercal_minus_before_mmag"),
    ("ZP_FGCM", "zp_fgcm"),
]

for label, col in quantities:
    vals = df[col].values
    ra_vals = df["ra"].values
    slope, intercept = linear_fit(ra_vals, vals)
    ra_range = ra_vals.max() - ra_vals.min()
    gradient_total = slope * ra_range

    bc, bm, _ = binned_median(ra_vals, vals)
    bin_range = bm.max() - bm.min() if len(bm) > 0 else 0

    unit = "mmag" if "mmag" in col else "mag"
    print(f"\n{label}:")
    print(f"  Median: {np.median(vals):.2f} {unit}")
    print(f"  RMS: {np.std(vals):.2f} {unit}")
    print(f"  Linear slope: {slope:.4f} {unit}/deg")
    print(f"  Gradient over RA range ({ra_range:.1f} deg): {gradient_total:.2f} {unit}")
    print(f"  Binned median range: {bin_range:.2f} {unit}")

# Also check FGCM - zpterm vs RA
print(f"\nZP_FGCM - zpterm (raw, should be ~MAGZERO_offset everywhere):")
vals = df["fgcm_minus_zpterm"].values * 1000.0  # mmag
ra_vals = df["ra"].values
slope, intercept = linear_fit(ra_vals, vals)
ra_range = ra_vals.max() - ra_vals.min()
gradient_total = slope * ra_range
bc, bm, _ = binned_median(ra_vals, vals)
bin_range = bm.max() - bm.min() if len(bm) > 0 else 0
print(f"  Median: {np.median(vals):.2f} mmag (= MAGZERO_offset * 1000)")
print(f"  RMS: {np.std(vals):.2f} mmag")
print(f"  Linear slope: {slope:.4f} mmag/deg")
print(f"  Gradient over RA range ({ra_range:.1f} deg): {gradient_total:.2f} mmag")
print(f"  Binned median range: {bin_range:.2f} mmag")

# -------------------------------------------------------------------
# 10. Make 4-panel plot
# -------------------------------------------------------------------
print("\nGenerating plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("FGCM Gradient Check (g-band, DES CCD-exposures, unanchored)", fontsize=14, y=0.98)

plot_specs = [
    ("ubercal_minus_fgcm_mmag", "ZP_ubercal - ZP_FGCM [mmag]", "mmag"),
    ("before_minus_fgcm_mmag", "ZP_before - ZP_FGCM [mmag]", "mmag"),
    ("ubercal_minus_before_mmag", "ZP_ubercal - ZP_before [mmag]", "mmag"),
    ("zp_fgcm", "ZP_FGCM [mag]", "mag"),
]

for ax, (col, ylabel, unit) in zip(axes.flat, plot_specs):
    ra_vals = df["ra"].values
    vals = df[col].values

    # Scatter (subsample if too many points)
    n = len(df)
    if n > 20000:
        idx = np.random.RandomState(42).choice(n, 20000, replace=False)
        ax.scatter(ra_vals[idx], vals[idx], s=1, alpha=0.1, color="gray", rasterized=True)
    else:
        ax.scatter(ra_vals, vals, s=1, alpha=0.15, color="gray", rasterized=True)

    # Binned median
    bc, bm, counts = binned_median(ra_vals, vals)
    ax.plot(bc, bm, "ro-", markersize=4, linewidth=1.5, label="binned median (1 deg)")

    # Linear fit
    slope, intercept = linear_fit(ra_vals, vals)
    ra_range = ra_vals.max() - ra_vals.min()
    gradient_total = slope * ra_range
    ra_fit = np.linspace(ra_vals.min(), ra_vals.max(), 100)
    ax.plot(ra_fit, slope * ra_fit + intercept, "b--", linewidth=1.0,
            label=f"slope={slope:.3f} {unit}/deg\n(total={gradient_total:.1f} {unit})")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set y-limits for mmag plots to show detail
    if "mmag" in col:
        med = np.median(vals)
        # Use 3*MAD for limits
        mad = np.median(np.abs(vals - med))
        ylim = max(5 * mad, 50)  # at least 50 mmag range
        ax.set_ylim(med - ylim, med + ylim)

plt.tight_layout(rect=[0, 0, 1, 0.96])

outdir = OUTPUT / f"validation_{BAND}"
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / "fgcm_gradient_check.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath}")

# -------------------------------------------------------------------
# 11. Extra plot: FGCM - zpterm vs RA
# -------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
vals_extra = df["fgcm_minus_zpterm"].values * 1000.0
ra_vals = df["ra"].values

n = len(df)
if n > 20000:
    idx = np.random.RandomState(42).choice(n, 20000, replace=False)
    ax2.scatter(ra_vals[idx], vals_extra[idx], s=1, alpha=0.1, color="gray", rasterized=True)
else:
    ax2.scatter(ra_vals, vals_extra, s=1, alpha=0.15, color="gray", rasterized=True)

bc, bm, _ = binned_median(ra_vals, vals_extra)
ax2.plot(bc, bm, "ro-", markersize=4, linewidth=1.5, label="binned median (1 deg)")

slope, intercept = linear_fit(ra_vals, vals_extra)
ra_fit = np.linspace(ra_vals.min(), ra_vals.max(), 100)
ax2.plot(ra_fit, slope * ra_fit + intercept, "b--", linewidth=1.0,
         label=f"slope={slope:.3f} mmag/deg")

ax2.set_xlabel("RA [deg]")
ax2.set_ylabel("ZP_FGCM - zpterm [mmag]")
ax2.set_title("MAGZERO variation vs RA (should be flat if MAGZERO is constant)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Set ylimits to show detail
med = np.median(vals_extra)
mad = np.median(np.abs(vals_extra - med))
ylim = max(5 * mad, 50)
ax2.set_ylim(med - ylim, med + ylim)

outpath2 = outdir / "fgcm_zpterm_vs_ra.png"
plt.savefig(outpath2, dpi=150, bbox_inches="tight")
print(f"Saved: {outpath2}")

# -------------------------------------------------------------------
# 12. Additional: Per-exposure analysis (average over CCDs per exposure)
# -------------------------------------------------------------------
print("\n" + "=" * 70)
print("PER-EXPOSURE ANALYSIS (median over CCDs per exposure)")
print("=" * 70)

exp_df = df.groupby("expnum").agg({
    "ra": "mean",
    "dec": "mean",
    "ubercal_minus_fgcm_mmag": "median",
    "before_minus_fgcm_mmag": "median",
    "ubercal_minus_before_mmag": "median",
    "zp_fgcm": "median",
}).reset_index()

print(f"\n{len(exp_df)} unique DES exposures")

for label, col in [
    ("ZP_ubercal - ZP_FGCM", "ubercal_minus_fgcm_mmag"),
    ("ZP_before - ZP_FGCM", "before_minus_fgcm_mmag"),
    ("ZP_ubercal - ZP_before", "ubercal_minus_before_mmag"),
]:
    vals = exp_df[col].values
    ra_vals = exp_df["ra"].values
    slope, intercept = linear_fit(ra_vals, vals)
    ra_range = ra_vals.max() - ra_vals.min()
    gradient_total = slope * ra_range
    bc, bm, _ = binned_median(ra_vals, vals)
    bin_range = bm.max() - bm.min() if len(bm) > 0 else 0
    print(f"\n{label} (per-exposure):")
    print(f"  Median: {np.median(vals):.2f} mmag")
    print(f"  RMS: {np.std(vals):.2f} mmag")
    print(f"  Linear slope: {slope:.4f} mmag/deg")
    print(f"  Gradient over RA range ({ra_range:.1f} deg): {gradient_total:.2f} mmag")
    print(f"  Binned median range: {bin_range:.2f} mmag")

print("\nDone.")
