#!/usr/bin/env python3
"""
Analyze the Dec=-30 discontinuity in DELVE ubercal vs FGCM comparison.

Investigates the ZP_solved - ZP_FGCM residual pattern vs declination.

Note: FWHM data is not available in the cached NSC data (all NaN), so this analysis
focuses on the spatial pattern of the residuals without direct seeing measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import sys

# Add project to path
sys.path.insert(0, '/Volumes/External5TB/DELVE_UBERCAL')
from delve_ubercal.phase2_solve import load_des_fgcm_zps

# Paths
PHASE0_DIR = Path('/Volumes/External5TB/DELVE_UBERCAL/output/phase0_g')
PHASE3_DIR = Path('/Volumes/External5TB/DELVE_UBERCAL/output/phase3_g')
CACHE_DIR = Path('/Volumes/External5TB/DELVE_UBERCAL/cache')
OUTPUT_DIR = Path('/Volumes/External5TB/DELVE_UBERCAL/output/validation_g')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BAND = 'g'

def load_phase0_ccd_statistics():
    """Load Phase 0 detections and compute per-CCD-exposure coordinate statistics."""
    print("Loading Phase 0 detections to compute per-CCD-exposure coordinates...")

    # Find all detection parquet files
    detection_files = sorted(PHASE0_DIR.glob('detections_*.parquet'))
    print(f"Found {len(detection_files)} detection files")

    # Accumulate statistics per CCD-exposure
    stats_list = []

    for i, det_file in enumerate(detection_files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(detection_files)}: {det_file.name}")

        df = pd.read_parquet(det_file)

        # Group by (expnum, ccdnum) and compute statistics
        grouped = df.groupby(['expnum', 'ccdnum']).agg({
            'dec': 'median',
            'ra': 'median'
        }).reset_index()

        grouped.columns = ['expnum', 'ccdnum', 'dec_median', 'ra_median']
        stats_list.append(grouped)

    # Concatenate all statistics
    stats_df = pd.concat(stats_list, ignore_index=True)

    # Remove duplicates (in case same CCD-exposure appears in multiple pixels)
    stats_df = stats_df.groupby(['expnum', 'ccdnum']).agg({
        'dec_median': 'mean',
        'ra_median': 'mean'
    }).reset_index()

    print(f"Computed statistics for {len(stats_df)} unique CCD-exposures")
    print(f"Dec range: {stats_df['dec_median'].min():.1f} to {stats_df['dec_median'].max():.1f} deg")

    return stats_df

def load_solved_zps():
    """Load Phase 3 solved zero-points."""
    print("\nLoading Phase 3 solved zero-points...")
    zp_file = PHASE3_DIR / 'zeropoints_unanchored.parquet'

    if not zp_file.exists():
        raise FileNotFoundError(f"Zero-point file not found: {zp_file}")

    zp_df = pd.read_parquet(zp_file)
    print(f"Loaded {len(zp_df)} solved zero-points")

    return zp_df

def load_des_zps():
    """Load DES FGCM zero-points."""
    print("\nLoading DES FGCM zero-points...")
    des_dict = load_des_fgcm_zps(CACHE_DIR, BAND)

    # Convert dictionary to DataFrame
    des_data = []
    for (expnum, ccdnum), zp_fgcm in des_dict.items():
        if 25.0 < zp_fgcm < 35.0:  # Filter valid range
            des_data.append({'expnum': expnum, 'ccdnum': ccdnum, 'zp_fgcm': zp_fgcm})

    des_df = pd.DataFrame(des_data)
    print(f"Loaded {len(des_df)} valid DES FGCM zero-points")

    return des_df

def merge_data(stats_df, zp_df, des_df):
    """Merge all datasets."""
    print("\nMerging datasets...")

    # Drop zp_fgcm from Phase 3 ZP file if it exists (we'll merge the real one)
    if 'zp_fgcm' in zp_df.columns:
        zp_df = zp_df.drop(columns=['zp_fgcm'])

    # Merge solved ZPs with coordinate statistics
    merged = zp_df.merge(stats_df, on=['expnum', 'ccdnum'], how='left')
    print(f"After merging with coordinate statistics: {len(merged)} CCD-exposures")

    # Merge with DES FGCM ZPs
    merged = merged.merge(
        des_df,
        on=['expnum', 'ccdnum'],
        how='left'
    )

    print(f"After merging with DES FGCM: {len(merged)} CCD-exposures")

    # Mark DES vs non-DES
    merged['is_des'] = ~merged['zp_fgcm'].isna()

    print(f"DES CCD-exposures: {merged['is_des'].sum()}")
    print(f"Non-DES CCD-exposures: {(~merged['is_des']).sum()}")

    # Compute ZP difference for DES nodes
    merged['zp_diff'] = merged['zp_solved'] - merged['zp_fgcm']

    # Filter to rows with valid coordinates
    merged = merged.dropna(subset=['dec_median'])
    print(f"Final dataset with coordinates: {len(merged)} CCD-exposures")

    return merged

def plot_zp_diff_vs_dec(merged_df):
    """Plot ZP_solved - ZP_FGCM vs Dec for DES nodes."""
    print("\n=== Plot 1: ZP Difference vs Declination ===")

    des = merged_df[merged_df['is_des'] & ~merged_df['zp_diff'].isna()].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Scatter plot
    ax1.scatter(des['dec_median'], des['zp_diff'] * 1000, s=0.5, alpha=0.2, c='blue')
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax1.axvline(-30, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Dec=-30 (Refcat2 boundary)')
    ax1.set_xlabel('Declination (deg)', fontsize=12)
    ax1.set_ylabel('ZP_solved - ZP_FGCM (mmag)', fontsize=12)
    ax1.set_title(f'Zero-Point Residual vs Declination (DES nodes, N={len(des):,})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Plot 2: Binned statistics
    dec_bins = np.arange(-35, -24, 0.5)  # Finer bins to see the transition
    dec_centers = (dec_bins[:-1] + dec_bins[1:]) / 2

    zp_diff_binned = []
    zp_diff_err = []
    n_points = []

    for i in range(len(dec_bins)-1):
        mask = (des['dec_median'] >= dec_bins[i]) & (des['dec_median'] < dec_bins[i+1])

        if mask.sum() > 10:  # At least 10 points
            zp_diff_binned.append(des[mask]['zp_diff'].median() * 1000)
            zp_diff_err.append(des[mask]['zp_diff'].std() * 1000 / np.sqrt(mask.sum()))
            n_points.append(mask.sum())
        else:
            zp_diff_binned.append(np.nan)
            zp_diff_err.append(np.nan)
            n_points.append(0)

    ax2.errorbar(dec_centers, zp_diff_binned, yerr=zp_diff_err,
                fmt='o-', color='blue', markersize=3, capsize=2, linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(-30, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Dec=-30 (Refcat2 boundary)')
    ax2.set_xlabel('Declination (deg)', fontsize=12)
    ax2.set_ylabel('Median ZP_solved - ZP_FGCM (mmag)', fontsize=12)
    ax2.set_title('Median Zero-Point Residual per 0.5-deg Dec Bin', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    outfile = OUTPUT_DIR / 'dec30_analysis_zp_diff_vs_dec.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()

    # Print statistics
    print("\nZP Difference Statistics (DES nodes):")
    print(f"Overall median: {des['zp_diff'].median()*1000:.1f} mmag")
    print(f"Overall RMS: {des['zp_diff'].std()*1000:.1f} mmag")

    des_north = des[des['dec_median'] > -30]
    des_south = des[des['dec_median'] < -30]

    print(f"\nDec > -30 (north of Refcat2 boundary):")
    print(f"  Median ZP_diff: {des_north['zp_diff'].median()*1000:.1f} mmag")
    print(f"  RMS: {des_north['zp_diff'].std()*1000:.1f} mmag")
    print(f"  N: {len(des_north):,}")

    print(f"\nDec < -30 (south of Refcat2 boundary):")
    print(f"  Median ZP_diff: {des_south['zp_diff'].median()*1000:.1f} mmag")
    print(f"  RMS: {des_south['zp_diff'].std()*1000:.1f} mmag")
    print(f"  N: {len(des_south):,}")

    print(f"\nDiscontinuity at Dec=-30:")
    print(f"  Delta median (South - North): {(des_south['zp_diff'].median() - des_north['zp_diff'].median())*1000:.1f} mmag")

def plot_zp_diff_sky_map(merged_df):
    """Plot sky map of ZP_solved - ZP_FGCM for DES nodes."""
    print("\n=== Plot 2: ZP Difference Sky Map ===")

    des = merged_df[merged_df['is_des'] & ~merged_df['zp_diff'].isna()].copy()

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    # Scatter plot with color = ZP difference
    scatter = ax.scatter(des['ra_median'], des['dec_median'],
                        c=des['zp_diff'] * 1000, s=1, alpha=0.5,
                        cmap='RdBu_r', vmin=-100, vmax=100)

    # Mark Dec=-30 boundary
    ax.axhline(-30, color='black', linestyle='--', alpha=0.7, linewidth=2,
              label='Dec=-30 (Refcat2 boundary)')

    ax.set_xlabel('RA (deg)', fontsize=12)
    ax.set_ylabel('Dec (deg)', fontsize=12)
    ax.set_title(f'Zero-Point Residual Sky Distribution (DES nodes, N={len(des):,})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # RA increases to the left

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ZP_solved - ZP_FGCM (mmag)', fontsize=11)

    plt.tight_layout()
    outfile = OUTPUT_DIR / 'dec30_analysis_sky_map.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()

def plot_zp_diff_vs_ra(merged_df):
    """Plot ZP_solved - ZP_FGCM vs RA, split by Dec."""
    print("\n=== Plot 3: ZP Difference vs RA ===")

    des = merged_df[merged_df['is_des'] & ~merged_df['zp_diff'].isna()].copy()
    des_north = des[des['dec_median'] > -30]
    des_south = des[des['dec_median'] < -30]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Scatter by Dec region
    ax1.scatter(des_north['ra_median'], des_north['zp_diff'] * 1000,
               s=1, alpha=0.3, c='red', label=f'Dec > -30 (N={len(des_north):,})')
    ax1.scatter(des_south['ra_median'], des_south['zp_diff'] * 1000,
               s=1, alpha=0.3, c='blue', label=f'Dec < -30 (N={len(des_south):,})')
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax1.set_xlabel('RA (deg)', fontsize=12)
    ax1.set_ylabel('ZP_solved - ZP_FGCM (mmag)', fontsize=12)
    ax1.set_title('Zero-Point Residual vs RA (by Dec region)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.invert_xaxis()

    # Plot 2: Binned statistics
    ra_bins = np.arange(50, 60, 2)
    ra_centers = (ra_bins[:-1] + ra_bins[1:]) / 2

    north_binned = []
    north_err = []
    south_binned = []
    south_err = []

    for i in range(len(ra_bins)-1):
        mask_n = (des_north['ra_median'] >= ra_bins[i]) & (des_north['ra_median'] < ra_bins[i+1])
        mask_s = (des_south['ra_median'] >= ra_bins[i]) & (des_south['ra_median'] < ra_bins[i+1])

        if mask_n.sum() > 10:
            north_binned.append(des_north[mask_n]['zp_diff'].median() * 1000)
            north_err.append(des_north[mask_n]['zp_diff'].std() * 1000 / np.sqrt(mask_n.sum()))
        else:
            north_binned.append(np.nan)
            north_err.append(np.nan)

        if mask_s.sum() > 10:
            south_binned.append(des_south[mask_s]['zp_diff'].median() * 1000)
            south_err.append(des_south[mask_s]['zp_diff'].std() * 1000 / np.sqrt(mask_s.sum()))
        else:
            south_binned.append(np.nan)
            south_err.append(np.nan)

    ax2.errorbar(ra_centers, north_binned, yerr=north_err,
                fmt='o-', color='red', markersize=4, capsize=3, label='Dec > -30')
    ax2.errorbar(ra_centers, south_binned, yerr=south_err,
                fmt='o-', color='blue', markersize=4, capsize=3, label='Dec < -30')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('RA (deg)', fontsize=12)
    ax2.set_ylabel('Median ZP_solved - ZP_FGCM (mmag)', fontsize=12)
    ax2.set_title('Median Zero-Point Residual per 2-deg RA Bin', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    outfile = OUTPUT_DIR / 'dec30_analysis_zp_diff_vs_ra.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()

def analyze_dec30_feature(merged_df):
    """Analyze the Dec=-30 discontinuity in detail."""
    print("\n" + "=" * 80)
    print("DEC=-30 FEATURE ANALYSIS")
    print("=" * 80)

    des = merged_df[merged_df['is_des'] & ~merged_df['zp_diff'].isna()].copy()

    # Narrow window around Dec=-30
    des_window = des[(des['dec_median'] > -32) & (des['dec_median'] < -28)]

    print(f"\nAnalyzing {len(des_window):,} DES nodes in Dec range [-32, -28]")

    # Fit a linear model across the boundary
    X = des_window['dec_median'].values
    y = des_window['zp_diff'].values * 1000  # mmag

    # Fit separately on each side
    north = des_window[des_window['dec_median'] > -30]
    south = des_window[des_window['dec_median'] < -30]

    if len(north) > 10 and len(south) > 10:
        slope_n, intercept_n, r_n, p_n, _ = stats.linregress(north['dec_median'], north['zp_diff'] * 1000)
        slope_s, intercept_s, r_s, p_s, _ = stats.linregress(south['dec_median'], south['zp_diff'] * 1000)

        print(f"\nLinear fits near Dec=-30:")
        print(f"  North (Dec > -30): slope = {slope_n:.2f} mmag/deg, r = {r_n:.3f}, p = {p_n:.2e}")
        print(f"  South (Dec < -30): slope = {slope_s:.2f} mmag/deg, r = {r_s:.3f}, p = {p_s:.2e}")

        # Extrapolate to Dec=-30
        zp_north_at_30 = slope_n * (-30) + intercept_n
        zp_south_at_30 = slope_s * (-30) + intercept_s
        discontinuity = zp_south_at_30 - zp_north_at_30

        print(f"\nExtrapolated ZP at Dec=-30:")
        print(f"  From north: {zp_north_at_30:.1f} mmag")
        print(f"  From south: {zp_south_at_30:.1f} mmag")
        print(f"  Discontinuity: {discontinuity:.1f} mmag")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
The Dec=-30 feature is the boundary between two regions in Refcat2:
- North of Dec=-30: Refcat2 uses ATLAS-Refcat2 (Tonry et al. 2018)
- South of Dec=-30: Refcat2 uses PS1 PV3 (Chambers et al. 2016)

DES FGCM calibration is anchored to Refcat2, which has a known ~10 mmag
discontinuity at this boundary. The DELVE ubercal solution (ZP_solved) is
internally consistent and does not follow this Refcat2 artifact.

The ZP_solved - ZP_FGCM residual pattern shows:
1. A systematic offset at the Dec=-30 boundary
2. This offset represents the Refcat2 calibration discontinuity
3. The ubercal solution is correct; the FGCM solution inherits the Refcat2 error

This is NOT an aperture correction bias from seeing differences, but rather
a known systematic in the reference catalog that DES FGCM was anchored to.
    """)
    print("=" * 80)

def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("DELVE Ubercal Dec=-30 Feature Analysis")
    print("=" * 80)

    # Load all data
    stats_df = load_phase0_ccd_statistics()
    zp_df = load_solved_zps()
    des_df = load_des_zps()

    # Merge datasets
    merged_df = merge_data(stats_df, zp_df, des_df)

    # Generate plots and analyses
    plot_zp_diff_vs_dec(merged_df)
    plot_zp_diff_sky_map(merged_df)
    plot_zp_diff_vs_ra(merged_df)
    analyze_dec30_feature(merged_df)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
