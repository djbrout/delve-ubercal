"""Phase 3b: Gradient detrend using Gaia + DELVE g-r color.

Runs AFTER Phase 3 completes for both g and r bands.
Applies a linear RA/Dec gradient correction measured against Gaia,
using DELVE g-r as the color axis (consistent for all bands).

Usage:
    python -m delve_ubercal.phase3b_gradient_detrend --band g --test-patch
    python -m delve_ubercal.phase3b_gradient_detrend --band r --test-patch
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from delve_ubercal.phase0_ingest import get_test_patch_pixels, load_config
from delve_ubercal.phase2_solve import (
    apply_gaia_gradient_detrend,
    compute_node_positions,
    load_des_fgcm_zps,
)


def run_gradient_detrend(band, config, output_dir, cache_dir):
    """Apply gradient detrend to Phase 3 ZPs using Gaia + DELVE g-r.

    Requires both g and r bands to have completed Phase 3.
    Modifies zeropoints_unanchored.parquet in place (overwrites).
    """
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    phase3_dir = output_dir / f"phase3_{band}"
    phase0_dir = output_dir / f"phase0_{band}"
    phase1_dir = output_dir / f"phase1_{band}"

    # Check prerequisites
    zp_file = phase3_dir / "zeropoints_unanchored.parquet"
    if not zp_file.exists():
        raise FileNotFoundError(f"Phase 3 ZPs not found: {zp_file}")

    # Save raw (pre-detrend) ZPs as backup — other bands will use these
    # for computing DELVE g-r so they don't see detrend artifacts
    raw_zp_file = phase3_dir / "zeropoints_unanchored_raw.parquet"
    if not raw_zp_file.exists():
        import shutil
        shutil.copy2(zp_file, raw_zp_file)
        print(f"  Saved raw ZPs backup to {raw_zp_file}", flush=True)
    else:
        # Restore from raw backup (in case we're re-running detrend)
        import shutil
        shutil.copy2(raw_zp_file, zp_file)
        print(f"  Restored raw ZPs from backup", flush=True)

    other_band = 'r' if band == 'g' else 'g'
    other_zp_file = output_dir / f"phase3_{other_band}" / "zeropoints_unanchored.parquet"
    if not other_zp_file.exists():
        raise FileNotFoundError(
            f"Other band ({other_band}) Phase 3 ZPs not found: {other_zp_file}. "
            f"Run Phase 3 for both g and r before detrending.")

    gaia_cache = cache_dir / "gaia_dr3_xmatch.parquet"
    if not gaia_cache.exists():
        raise FileNotFoundError(f"Gaia DR3 cache not found: {gaia_cache}")

    print(f"\n=== Phase 3b: Gradient Detrend ({band}-band) ===", flush=True)

    # Load Phase 3 ZPs
    zp_df = pd.read_parquet(zp_file)
    print(f"  Loaded {len(zp_df):,} ZPs from Phase 3", flush=True)

    # Build node index (same order as ZP file)
    idx_to_node = list(zip(zp_df['expnum'].astype(int).values,
                           zp_df['ccdnum'].astype(int).values))
    node_to_idx = {node: i for i, node in enumerate(idx_to_node)}
    zp_array = zp_df['zp_solved'].values.copy()

    # Load node positions
    node_positions = compute_node_positions(phase0_dir)
    print(f"  Node positions: {len(node_positions):,}", flush=True)

    # Load DES FGCM ZPs (for absolute zero point only)
    des_fgcm_zps = load_des_fgcm_zps(cache_dir, band)
    print(f"  DES FGCM ZPs: {len(des_fgcm_zps):,}", flush=True)

    # Star list files
    star_list_files = sorted(phase1_dir.glob('star_lists_nside*_pixel*.parquet'))
    print(f"  Star list files: {len(star_list_files)}", flush=True)

    # Apply gradient detrend (always DELVE g-r, no fallback)
    grad_info = apply_gaia_gradient_detrend(
        zp_array, idx_to_node, node_to_idx, node_positions,
        star_list_files, gaia_cache, band,
        des_fgcm_zps=des_fgcm_zps,
        output_dir=output_dir,
    )

    # Update the ZP dataframe and save
    zp_df['zp_solved'] = zp_array
    # Recompute delta_zp
    if 'zp_fgcm' in zp_df.columns:
        zp_df['delta_zp'] = zp_df['zp_solved'] - zp_df['zp_fgcm']

    zp_df.to_parquet(zp_file, index=False)
    print(f"  Saved detrended ZPs to {zp_file}", flush=True)

    # Save gradient info
    import json
    info_file = phase3_dir / "gradient_detrend_info.json"
    with open(info_file, 'w') as f:
        json.dump(grad_info, f, indent=2, default=str)
    print(f"  Saved gradient info to {info_file}", flush=True)

    print(f"\n  Summary: RA={grad_info['ra_slope']:.2f}, "
          f"Dec={grad_info['dec_slope']:.2f} mmag/deg "
          f"({grad_info['n_stars']:,} stars, DELVE g-r)", flush=True)

    return grad_info


def main():
    parser = argparse.ArgumentParser(description="Phase 3b: Gradient detrend")
    parser.add_argument("--band", required=True, choices=["g", "r", "i", "z"])
    parser.add_argument("--test-patch", action="store_true",
                        help="Use 10x10 deg test patch")
    args = parser.parse_args()

    config = load_config()
    output_dir = config["data"]["output_path"]
    cache_dir = config["data"]["cache_path"]

    run_gradient_detrend(args.band, config, output_dir, cache_dir)


if __name__ == "__main__":
    main()
