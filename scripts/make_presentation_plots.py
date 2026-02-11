"""Generate presentation-quality plots for DELVE UBERCAL slides."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT = Path("/Volumes/External5TB/DELVE_UBERCAL/output")
CACHE = Path("/Volumes/External5TB/DELVE_UBERCAL/cache")
PRES = OUTPUT / "presentation"
PRES.mkdir(exist_ok=True)

band = "g"


# ============================================================
# SLIDE 2: Test patch sky coverage
# ============================================================
def slide2_test_patch():
    """Show the test patch location on sky."""
    import healpy as hp

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Load Phase 0 detections to get coverage
    phase0_dir = OUTPUT / f"phase0_{band}"
    files = sorted(phase0_dir.glob("detections_nside32_pixel*.parquet"))

    all_ra, all_dec = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["ra", "dec"])
        # Subsample for speed
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)
        all_ra.append(df["ra"].values)
        all_dec.append(df["dec"].values)
    all_ra = np.concatenate(all_ra)
    all_dec = np.concatenate(all_dec)

    ax.scatter(all_ra, all_dec, s=0.1, alpha=0.3, c='steelblue', rasterized=True)

    # Mark the test patch boundary
    ra_min, ra_max = 50, 60
    dec_min, dec_max = -35, -25
    rect = plt.Rectangle((ra_min, dec_min), ra_max - ra_min, dec_max - dec_min,
                          linewidth=2, edgecolor='red', facecolor='none',
                          linestyle='--', label='Test patch')
    ax.add_patch(rect)

    # Dec=-30 line
    ax.axhline(-30, color='orange', linewidth=1.5, linestyle=':', alpha=0.8,
               label='Dec = $-30^\\circ$ (DES boundary)')

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Test Patch: {band}-band detections\n"
                 f"RA=[50,60], Dec=[$-35$,$-25$], 40 HEALPix pixels")
    ax.legend(loc='upper right', fontsize=12)
    ax.invert_xaxis()
    ax.set_xlim(62, 48)
    ax.set_ylim(-37, -23)

    fig.savefig(PRES / "slide2_test_patch.png")
    plt.close(fig)
    print("  Slide 2: test patch map saved")


# ============================================================
# SLIDE 3: FGCM comparison WITHOUT tilt correction
# ============================================================
def _star_level_fgcm(zp_file):
    """Compute star-level FGCM comparison (like Test 0).
    Returns DataFrame with ra, dec, diff_mmag per star.
    """
    from delve_ubercal.phase2_solve import load_des_fgcm_zps, load_nsc_zpterms

    zp_df = pd.read_parquet(zp_file)
    des_fgcm = load_des_fgcm_zps(CACHE, band)
    nsc_zpterms = load_nsc_zpterms(CACHE, band)

    # Build ZP dicts
    ubercal_dict = dict(zip(
        zip(zp_df['expnum'].astype(int), zp_df['ccdnum'].astype(int)),
        zp_df['zp_solved']))

    # MAGZERO offset
    diffs = []
    for key, fgcm_val in des_fgcm.items():
        if 25.0 < fgcm_val < 35.0 and key in nsc_zpterms:
            diffs.append(fgcm_val - nsc_zpterms[key])
    magzero_offset = np.median(diffs)

    # Flagged nodes
    flagged_file = OUTPUT / f"phase3_{band}" / "flagged_nodes.parquet"
    flagged_nodes = set()
    if flagged_file.exists():
        fn = pd.read_parquet(flagged_file)
        flagged_nodes = set(zip(fn["expnum"].astype(int), fn["ccdnum"].astype(int)))

    # Stream through Phase 0 detections, compute per-star weighted mean
    phase0_dir = OUTPUT / f"phase0_{band}"
    files = sorted(phase0_dir.glob("detections_nside32_pixel*.parquet"))

    chunks = []
    for f in files:
        df = pd.read_parquet(f, columns=['objectid', 'ra', 'dec', 'm_inst', 'm_err',
                                          'expnum', 'ccdnum'])
        keys = list(zip(df['expnum'].astype(int).values, df['ccdnum'].astype(int).values))

        # Get ZPs for each detection
        zp_ub = np.array([ubercal_dict.get(k, np.nan) for k in keys])
        zp_fg = np.array([des_fgcm.get(k, np.nan) for k in keys])
        flagged = np.array([k in flagged_nodes for k in keys])

        # Only DES detections with valid FGCM and valid ubercal ZPs
        valid = np.isfinite(zp_ub) & np.isfinite(zp_fg) & ~flagged
        valid &= (zp_fg > 25.0) & (zp_fg < 35.0)
        valid &= (zp_ub > 25.0) & (zp_ub < 35.0)
        df = df[valid].copy()
        zp_ub = zp_ub[valid]
        zp_fg = zp_fg[valid]

        if len(df) == 0:
            continue

        df['m_ubercal'] = df['m_inst'].values + zp_ub - magzero_offset
        df['m_fgcm'] = df['m_inst'].values + zp_fg - magzero_offset
        df['w'] = 1.0 / (df['m_err'].values**2 + 1e-6)
        df['wm_ub'] = df['m_ubercal'] * df['w']
        df['wm_fg'] = df['m_fgcm'] * df['w']
        chunks.append(df[['objectid', 'ra', 'dec', 'wm_ub', 'wm_fg', 'w']])

    all_det = pd.concat(chunks, ignore_index=True)
    stars = all_det.groupby('objectid').agg(
        ra=('ra', 'mean'), dec=('dec', 'mean'),
        swm_ub=('wm_ub', 'sum'), swm_fg=('wm_fg', 'sum'),
        sw=('w', 'sum'), n=('w', 'count'),
    )
    stars = stars[stars['n'] >= 2]
    stars['m_ub'] = stars['swm_ub'] / stars['sw']
    stars['m_fg'] = stars['swm_fg'] / stars['sw']
    stars['diff_mmag'] = (stars['m_ub'] - stars['m_fg']) * 1000
    stars['diff_mmag'] -= stars['diff_mmag'].mean()
    return stars.reset_index()


def slide3_fgcm_no_detrend():
    """Show FGCM comparison using raw (pre-detrend) ZPs — star level."""
    raw_file = OUTPUT / f"phase3_{band}" / "zeropoints_unanchored_raw.parquet"
    if not raw_file.exists():
        raw_file = OUTPUT / f"phase3_{band}" / "zeropoints_unanchored.parquet"

    stars = _star_level_fgcm(raw_file)
    rms = np.sqrt(np.mean(stars['diff_mmag']**2))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Skymap
    ax = axes[0]
    vmax = 50
    sc = ax.scatter(stars["ra"], stars["dec"], c=stars["diff_mmag"], s=0.5, alpha=0.4,
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("DELVE $-$ FGCM (mmag)")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"{band}-band: Star-level ubercal $-$ FGCM (no tilt correction)\n"
                 f"RMS = {rms:.1f} mmag, {len(stars):,} stars")
    ax.invert_xaxis()

    ra_c = stars["ra"].mean()
    slope, _ = np.polyfit(stars["ra"] - ra_c, stars["diff_mmag"], 1)
    ax.text(0.05, 0.05, f"RA gradient: {slope:.1f} mmag/deg",
            transform=ax.transAxes, fontsize=12, color='black',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Histogram
    ax = axes[1]
    ax.hist(stars["diff_mmag"], bins=100, range=(-100, 100), color="steelblue",
            alpha=0.7, edgecolor='none', label=f"Ubercal (no tilt)\nRMS = {rms:.1f} mmag")
    ax.set_xlabel("DELVE $-$ FGCM (mmag)")
    ax.set_ylabel("Stars")
    ax.set_title(f"{band}-band: Star-level ubercal vs DES FGCM")
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(PRES / "slide3_fgcm_no_detrend.png")
    plt.close(fig)
    print(f"  Slide 3: FGCM no-detrend saved (RMS={rms:.1f} mmag, slope={slope:.1f})")


# ============================================================
# SLIDE 4: Gaia detrend process
# ============================================================
def slide4_gaia_detrend():
    """Show the detrending process: residuals vs RA before/after."""
    from delve_ubercal.phase2_solve import compute_node_positions, load_des_fgcm_zps

    # Load raw and detrended ZPs
    raw_file = OUTPUT / f"phase3_{band}" / "zeropoints_unanchored_raw.parquet"
    det_file = OUTPUT / f"phase3_{band}" / "zeropoints_unanchored.parquet"
    zp_raw = pd.read_parquet(raw_file)
    zp_det = pd.read_parquet(det_file)

    # Load Gaia crossmatch
    gaia = pd.read_parquet(CACHE / "gaia_dr3_xmatch.parquet")
    gaia_mag_col = 'phot_bp_mean_mag' if band == 'g' else 'phot_rp_mean_mag'
    gaia = gaia[(gaia[gaia_mag_col].between(10, 25)) &
                (gaia['bp_rp'].between(0.3, 3.5)) &
                (gaia['phot_g_mean_flux_over_error'] > 50)]

    # Load star lists and compute per-star calibrated mags (raw)
    phase1_dir = OUTPUT / f"phase1_{band}"
    star_files = sorted(phase1_dir.glob("star_lists_nside*_pixel*.parquet"))

    raw_zp_dict = dict(zip(
        zip(zp_raw['expnum'].astype(int), zp_raw['ccdnum'].astype(int)),
        zp_raw['zp_solved']))
    det_zp_dict = dict(zip(
        zip(zp_det['expnum'].astype(int), zp_det['ccdnum'].astype(int)),
        zp_det['zp_solved']))

    chunks_raw, chunks_det = [], []
    for f in star_files:
        df = pd.read_parquet(f, columns=['objectid', 'm_inst', 'm_err', 'expnum', 'ccdnum'])
        keys = list(zip(df['expnum'].astype(int).values, df['ccdnum'].astype(int).values))
        zps_r = np.array([raw_zp_dict.get(k, np.nan) for k in keys])
        zps_d = np.array([det_zp_dict.get(k, np.nan) for k in keys])
        v = np.isfinite(zps_r) & np.isfinite(zps_d)
        dfs = df[v].copy()
        dfs['m_raw'] = dfs['m_inst'].values + zps_r[v]
        dfs['m_det'] = dfs['m_inst'].values + zps_d[v]
        dfs['w'] = 1.0 / (dfs['m_err'].values**2 + 1e-6)
        chunks_raw.append(dfs[['objectid', 'm_raw', 'w']].rename(columns={'m_raw': 'mc', 'w': 'wt'}))
        chunks_det.append(dfs[['objectid', 'm_det', 'w']].rename(columns={'m_det': 'mc', 'w': 'wt'}))

    # Weighted mean per star (raw)
    all_r = pd.concat(chunks_raw, ignore_index=True)
    all_r['wm'] = all_r['mc'] * all_r['wt']
    sr = all_r.groupby('objectid').agg(swm=('wm', 'sum'), sw=('wt', 'sum'))
    sr['m_cal_raw'] = sr['swm'] / sr['sw']

    # Weighted mean per star (detrended)
    all_d = pd.concat(chunks_det, ignore_index=True)
    all_d['wm'] = all_d['mc'] * all_d['wt']
    sd = all_d.groupby('objectid').agg(swm=('wm', 'sum'), sw=('wt', 'sum'))
    sd['m_cal_det'] = sd['swm'] / sd['sw']

    # Join with Gaia
    gaia_idx = gaia.set_index('nsc_objectid')
    stars = sr[['m_cal_raw']].join(sd[['m_cal_det']], how='inner')
    stars = stars.join(gaia_idx[['ra', 'dec', gaia_mag_col, 'bp_rp']], how='inner')

    stars['resid_raw'] = (stars['m_cal_raw'] - stars[gaia_mag_col]) * 1000
    stars['resid_det'] = (stars['m_cal_det'] - stars[gaia_mag_col]) * 1000

    # Clip outliers
    med = stars['resid_raw'].median()
    clip_mask = stars['resid_raw'].between(med - 500, med + 500)
    clip_mask &= stars['resid_det'].between(med - 500, med + 500)
    stars = stars[clip_mask]

    # Remove color term (poly5 in Gaia BP-RP) — same CT for both raw and detrended
    color = stars['bp_rp'].values
    color_mask = np.isfinite(color) & (color > 0.3) & (color < 3.5)
    coeffs = np.polyfit(color[color_mask], stars['resid_raw'].values[color_mask], 5)
    ct = np.polyval(coeffs, color)
    stars['resid_raw'] -= ct
    stars['resid_det'] -= ct
    # Mean-subtract
    stars['resid_raw'] -= stars['resid_raw'].median()
    stars['resid_det'] -= stars['resid_det'].median()

    # Keep only stars with valid color term removal
    stars = stars[color_mask]

    # Bin by RA
    ra_bins = np.linspace(stars['ra'].min(), stars['ra'].max(), 30)
    ra_centers = 0.5 * (ra_bins[:-1] + ra_bins[1:])
    bin_idx = np.digitize(stars['ra'].values, ra_bins) - 1

    med_raw, med_det = [], []
    for i in range(len(ra_centers)):
        mask = bin_idx == i
        if mask.sum() > 10:
            med_raw.append(np.median(stars['resid_raw'].values[mask]))
            med_det.append(np.median(stars['resid_det'].values[mask]))
        else:
            med_raw.append(np.nan)
            med_det.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Before detrend
    ax = axes[0]
    ax.scatter(stars['ra'], stars['resid_raw'], s=0.3, alpha=0.05,
               c='gray', rasterized=True)
    ax.plot(ra_centers, med_raw, 'ro-', markersize=6, linewidth=2, label='Binned median')
    # Fit line
    valid_bins = np.array([not np.isnan(x) for x in med_raw])
    if valid_bins.sum() > 2:
        slope, intercept = np.polyfit(ra_centers[valid_bins], np.array(med_raw)[valid_bins], 1)
        ax.plot(ra_centers, slope * ra_centers + intercept, 'r--', linewidth=2,
                label=f'Slope: {slope:.1f} mmag/deg')
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel(f"DELVE $-$ Gaia $G_{{\\rm BP}}$ (mmag, after CT)")
    ax.set_title("Before tilt correction")
    ax.set_ylim(-80, 80)
    ax.legend(fontsize=12)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

    # After detrend
    ax = axes[1]
    ax.scatter(stars['ra'], stars['resid_det'], s=0.3, alpha=0.05,
               c='gray', rasterized=True)
    ax.plot(ra_centers, med_det, 'bo-', markersize=6, linewidth=2, label='Binned median')
    valid_bins = np.array([not np.isnan(x) for x in med_det])
    if valid_bins.sum() > 2:
        slope2, intercept2 = np.polyfit(ra_centers[valid_bins], np.array(med_det)[valid_bins], 1)
        ax.plot(ra_centers, slope2 * ra_centers + intercept2, 'b--', linewidth=2,
                label=f'Slope: {slope2:.1f} mmag/deg')
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel(f"DELVE $-$ Gaia $G_{{\\rm BP}}$ (mmag, after CT)")
    ax.set_title("After tilt correction")
    ax.set_ylim(-80, 80)
    ax.legend(fontsize=12)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

    fig.suptitle(f"Slide 4: Gaia gradient detrend ({band}-band, DELVE g$-$r color term)",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(PRES / "slide4_gaia_detrend.png")
    plt.close(fig)
    print(f"  Slide 4: Gaia detrend process saved")


# ============================================================
# SLIDE 5: FGCM comparison WITH tilt fix
# ============================================================
def slide5_fgcm_with_detrend():
    """Show FGCM comparison using detrended ZPs — star level."""
    # Star-level comparison: detrended (after tilt correction)
    det_file = OUTPUT / f"phase3_{band}" / "zeropoints_unanchored.parquet"
    stars_det = _star_level_fgcm(det_file)
    rms = np.sqrt(np.mean(stars_det['diff_mmag']**2))

    # Star-level comparison: raw (before tilt correction) for histogram overlay
    raw_file = OUTPUT / f"phase3_{band}" / "zeropoints_unanchored_raw.parquet"
    if not raw_file.exists():
        raw_file = det_file
    stars_raw = _star_level_fgcm(raw_file)
    rms_raw = np.sqrt(np.mean(stars_raw['diff_mmag']**2))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Skymap (detrended)
    ax = axes[0]
    vmax = 50
    sc = ax.scatter(stars_det["ra"], stars_det["dec"], c=stars_det["diff_mmag"],
                    s=0.5, alpha=0.4, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("DELVE $-$ FGCM (mmag)")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"{band}-band: Star-level ubercal $-$ FGCM (with tilt correction)\n"
                 f"RMS = {rms:.1f} mmag, {len(stars_det):,} stars")
    ax.invert_xaxis()

    # Histogram: before/after comparison
    ax = axes[1]
    ax.hist(stars_raw["diff_mmag"], bins=100, range=(-100, 100), color="lightcoral",
            alpha=0.6, edgecolor='none',
            label=f"No tilt corr: {rms_raw:.1f} mmag")
    ax.hist(stars_det["diff_mmag"], bins=100, range=(-100, 100), color="steelblue",
            alpha=0.7, edgecolor='none',
            label=f"With tilt corr: {rms:.1f} mmag")
    ax.set_xlabel("DELVE $-$ FGCM (mmag)")
    ax.set_ylabel("Stars")
    ax.set_title(f"{band}-band: Star-level ubercal vs DES FGCM")
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(PRES / "slide5_fgcm_with_detrend.png")
    plt.close(fig)
    print(f"  Slide 5: FGCM with-detrend saved (RMS: {rms_raw:.1f} -> {rms:.1f} mmag)")


# ============================================================
# SLIDE 6: Gaia + PS1 external comparisons
# ============================================================
def slide6_gaia_ps1():
    """Show Gaia and PS1 external comparison histograms before/after."""
    cat = pd.read_parquet(OUTPUT / f"phase5_{band}" / f"star_catalog_{band}.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Gaia ---
    gaia_mag_col = 'phot_bp_mean_mag' if band == 'g' else 'phot_rp_mean_mag'
    gaia_label = '$G_{\\rm BP}$' if band == 'g' else '$G_{\\rm RP}$'

    gaia = pd.read_parquet(CACHE / "gaia_dr3_xmatch.parquet")
    gaia = gaia[(gaia[gaia_mag_col].between(10, 25)) &
                (gaia['bp_rp'].between(0.3, 3.5)) &
                (gaia['phot_g_mean_flux_over_error'] > 50)]
    gaia = gaia.set_index('nsc_objectid')

    merged = cat.set_index('objectid').join(gaia[[gaia_mag_col, 'bp_rp']], how='inner')

    # Use DELVE g-r if available
    if f'mag_ubercal_{band}' in merged.columns and f'mag_before_{band}' in merged.columns:
        mag_after = merged[f'mag_ubercal_{band}']
        mag_before = merged[f'mag_before_{band}']
    else:
        mag_after = merged['mag_ubercal']
        mag_before = merged['mag_before']

    # Color term fit (degree 5) on after, then mean-subtract
    color = merged['bp_rp'].values
    resid_after = (mag_after.values - merged[gaia_mag_col].values) * 1000
    resid_before = (mag_before.values - merged[gaia_mag_col].values) * 1000

    # Clip
    med = np.median(resid_after[np.isfinite(resid_after)])
    mask = np.isfinite(resid_after) & np.isfinite(resid_before) & np.isfinite(color)
    mask &= np.abs(resid_after - med) < 500
    mask &= color > 0.3
    mask &= color < 3.5

    coeffs = np.polyfit(color[mask], resid_after[mask], 5)
    ct_after = np.polyval(coeffs, color)
    ct_before = np.polyval(coeffs, color)  # Same color term for fair comparison

    resid_after_ct = resid_after - ct_after
    resid_before_ct = resid_before - ct_before
    # Mean-subtract (we only care about scatter, not absolute offset)
    resid_after_ct -= np.nanmean(resid_after_ct[mask])
    resid_before_ct -= np.nanmean(resid_before_ct[mask])

    # 3-sigma clip
    std_a = np.std(resid_after_ct[mask])
    clip = mask & (np.abs(resid_after_ct) < 3 * std_a)
    resid_after_ct -= np.nanmean(resid_after_ct[clip])
    resid_before_ct -= np.nanmean(resid_before_ct[clip])
    mask = clip

    rms_after = np.sqrt(np.nanmean(resid_after_ct[mask]**2))
    rms_before = np.sqrt(np.nanmean(resid_before_ct[mask]**2))

    ax = axes[0]
    ax.hist(resid_before_ct[mask], bins=80, range=(-200, 200),
            color='lightcoral', alpha=0.6, edgecolor='none',
            label=f'Before ubercal: {rms_before:.1f} mmag')
    ax.hist(resid_after_ct[mask], bins=80, range=(-200, 200),
            color='steelblue', alpha=0.7, edgecolor='none',
            label=f'After ubercal: {rms_after:.1f} mmag')
    ax.set_xlabel(f"DELVE {band} $-$ Gaia {gaia_label} (mmag, after CT)")
    ax.set_ylabel("Stars")
    ax.set_title(f"Gaia DR3 comparison ({merged[mask].shape[0]:,} stars)")
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=11)

    # --- PS1 ---
    ps1_file = CACHE / f"ps1_dr2_xmatch_{band}.parquet"
    if ps1_file.exists():
        ps1 = pd.read_parquet(ps1_file)

        # PS1 columns have ps1_ prefix in our cache
        ps1_mag_col = f'ps1_{band}MeanPSFMag'
        ps1_g_col = 'ps1_gMeanPSFMag'
        ps1_i_col = 'ps1_iMeanPSFMag'

        if ps1_mag_col in ps1.columns:
            # Build unique column list (ps1_mag_col may overlap with ps1_g_col for g-band)
            ps1_cols = list(dict.fromkeys([ps1_mag_col, ps1_g_col, ps1_i_col]))
            ps1_idx = ps1.set_index('objectid')
            cat_idx = cat.set_index('objectid')
            m = cat_idx.join(ps1_idx[ps1_cols], how='inner')

            # Filter PS1 sentinels (-999)
            ps1_valid = (m[ps1_mag_col] > 10) & (m[ps1_mag_col] < 25)
            gi_valid = (m[ps1_g_col].values > 10) & (m[ps1_i_col].values > 10)
            mask_ps1 = ps1_valid & gi_valid

            if f'mag_ubercal_{band}' in m.columns:
                mag_a = m[f'mag_ubercal_{band}']
                mag_b = m[f'mag_before_{band}']
            else:
                mag_a = m['mag_ubercal']
                mag_b = m['mag_before']

            resid_a = (mag_a.values - m[ps1_mag_col].values) * 1000
            resid_b = (mag_b.values - m[ps1_mag_col].values) * 1000
            ps1_gi = m[ps1_g_col].values - m[ps1_i_col].values

            mask_ps1 = mask_ps1.values & np.isfinite(resid_a) & np.isfinite(ps1_gi)
            mask_ps1 &= np.abs(ps1_gi) < 4

            # Color term (degree 3 for PS1)
            coeffs_ps1 = np.polyfit(ps1_gi[mask_ps1], resid_a[mask_ps1], 3)
            ct_a = np.polyval(coeffs_ps1, ps1_gi)
            ct_b = np.polyval(coeffs_ps1, ps1_gi)

            ra_ps1 = resid_a - ct_a
            rb_ps1 = resid_b - ct_b
            # Mean-subtract
            ra_ps1 -= np.nanmean(ra_ps1[mask_ps1])
            rb_ps1 -= np.nanmean(rb_ps1[mask_ps1])

            # 3-sigma clip on after-ubercal residuals (removes ~2% outliers)
            std_a = np.std(ra_ps1[mask_ps1])
            clip = mask_ps1 & (np.abs(ra_ps1) < 3 * std_a)
            ra_ps1 -= np.nanmean(ra_ps1[clip])
            rb_ps1 -= np.nanmean(rb_ps1[clip])
            mask_ps1 = clip

            rms_a = np.sqrt(np.nanmean(ra_ps1[mask_ps1]**2))
            rms_b = np.sqrt(np.nanmean(rb_ps1[mask_ps1]**2))

            ax = axes[1]
            ax.hist(rb_ps1[mask_ps1], bins=80, range=(-200, 200),
                    color='lightcoral', alpha=0.6, edgecolor='none',
                    label=f'Before ubercal: {rms_b:.1f} mmag')
            ax.hist(ra_ps1[mask_ps1], bins=80, range=(-200, 200),
                    color='steelblue', alpha=0.7, edgecolor='none',
                    label=f'After ubercal: {rms_a:.1f} mmag')
            ax.set_xlabel(f"DELVE {band} $-$ PS1 {band} (mmag, after CT)")
            ax.set_ylabel("Stars")
            ax.set_title(f"PS1 DR2 comparison ({mask_ps1.sum():,} stars, dec > $-30^\\circ$)")
            ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
            ax.legend(fontsize=11)
        else:
            axes[1].text(0.5, 0.5, f"PS1 mag column {ps1_mag_col} not found\n{list(ps1.columns)[:10]}",
                         transform=axes[1].transAxes, ha='center', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "PS1 cache not found",
                     transform=axes[1].transAxes, ha='center')

    fig.suptitle(f"Slide 6: External validation ({band}-band)",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(PRES / "slide6_gaia_ps1.png")
    plt.close(fig)
    print(f"  Slide 6: Gaia + PS1 comparison saved")


# ============================================================
# SLIDE 7: PS1 skymap + histogram (before/after detrend)
# ============================================================
def slide7_ps1_skymap():
    """PS1 comparison: skymap of residuals + before/after histogram."""
    cat = pd.read_parquet(OUTPUT / f"phase5_{band}" / f"star_catalog_{band}.parquet")
    ps1 = pd.read_parquet(CACHE / f"ps1_dr2_xmatch_{band}.parquet")

    ps1_mag_col = f'ps1_{band}MeanPSFMag'
    ps1_g_col = 'ps1_gMeanPSFMag'
    ps1_i_col = 'ps1_iMeanPSFMag'

    ps1_cols = list(dict.fromkeys([ps1_mag_col, ps1_g_col, ps1_i_col]))
    ps1_idx = ps1.set_index('objectid')
    cat_idx = cat.set_index('objectid')
    m = cat_idx.join(ps1_idx[ps1_cols], how='inner')

    # Filter PS1 sentinels
    ps1_valid = (m[ps1_mag_col] > 10) & (m[ps1_mag_col] < 25)
    gi_valid = (m[ps1_g_col].values > 10) & (m[ps1_i_col].values > 10)
    mask = ps1_valid.values & gi_valid

    mag_after = m[f'mag_ubercal_{band}'].values
    mag_before = m[f'mag_before_{band}'].values
    ps1_mag = m[ps1_mag_col].values
    ps1_gi = m[ps1_g_col].values - m[ps1_i_col].values
    ra = m['ra'].values
    dec = m['dec'].values

    resid_after = (mag_after - ps1_mag) * 1000
    resid_before = (mag_before - ps1_mag) * 1000

    mask &= np.isfinite(resid_after) & np.isfinite(resid_before) & np.isfinite(ps1_gi)
    mask &= np.abs(ps1_gi) < 4

    # Color term (degree 3, fit on after-ubercal)
    coeffs = np.polyfit(ps1_gi[mask], resid_after[mask], 3)
    ct = np.polyval(coeffs, ps1_gi)
    resid_after_ct = resid_after - ct
    resid_before_ct = resid_before - ct

    # Mean-subtract
    resid_after_ct -= np.nanmean(resid_after_ct[mask])
    resid_before_ct -= np.nanmean(resid_before_ct[mask])

    # 3-sigma clip
    std_a = np.std(resid_after_ct[mask])
    clip = mask & (np.abs(resid_after_ct) < 3 * std_a)
    resid_after_ct -= np.nanmean(resid_after_ct[clip])
    resid_before_ct -= np.nanmean(resid_before_ct[clip])

    rms_after = np.sqrt(np.nanmean(resid_after_ct[clip]**2))
    rms_before = np.sqrt(np.nanmean(resid_before_ct[clip]**2))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Skymap of after-detrend residuals
    ax = axes[0]
    vmax = 60
    sc = ax.scatter(ra[clip], dec[clip], c=resid_after_ct[clip], s=0.5, alpha=0.4,
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("DELVE $-$ PS1 (mmag, after CT)")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"{band}-band: DELVE $-$ PS1 (after detrend)\n"
                 f"RMS = {rms_after:.1f} mmag, {clip.sum():,} stars, dec > $-30^\\circ$")
    ax.invert_xaxis()

    # Histogram before/after
    ax = axes[1]
    ax.hist(resid_before_ct[clip], bins=80, range=(-150, 150),
            color='lightcoral', alpha=0.6, edgecolor='none',
            label=f'Before ubercal: {rms_before:.1f} mmag')
    ax.hist(resid_after_ct[clip], bins=80, range=(-150, 150),
            color='steelblue', alpha=0.7, edgecolor='none',
            label=f'After ubercal: {rms_after:.1f} mmag')
    ax.set_xlabel(f"DELVE {band} $-$ PS1 {band} (mmag, after CT)")
    ax.set_ylabel("Stars")
    ax.set_title(f"{band}-band: PS1 DR2 comparison (3$\\sigma$-clipped)")
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(PRES / "slide7_ps1_skymap.png")
    plt.close(fig)
    print(f"  Slide 7: PS1 skymap saved (RMS: {rms_before:.1f} -> {rms_after:.1f} mmag)")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("Generating presentation plots...")
    slide2_test_patch()
    slide3_fgcm_no_detrend()
    slide4_gaia_detrend()
    slide5_fgcm_with_detrend()
    slide6_gaia_ps1()
    slide7_ps1_skymap()
    print(f"\nAll plots saved to {PRES}/")
