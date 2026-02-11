#!/usr/bin/env python3
"""Run the full DELVE ubercalibration pipeline on all bands.

Usage:
    python3 run_full_pipeline.py                  # Full sky, all bands
    python3 run_full_pipeline.py --bands g r      # Specific bands
    python3 run_full_pipeline.py --start-phase 2  # Resume from Phase 2
    python3 run_full_pipeline.py --test-region     # Test region only
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

BANDS = ["g", "r", "i", "z"]
BASE_DIR = Path("/Volumes/External5TB/DELVE_UBERCAL")
PYTHON = sys.executable
ENV = {
    "PYTHONPATH": str(BASE_DIR),
    "PYTHONUNBUFFERED": "1",
}

LOG_DIR = BASE_DIR / "logs"


def run_command(cmd, log_file, description):
    """Run a command, logging to file and printing status."""
    import os

    full_env = os.environ.copy()
    full_env.update(ENV)

    print(f"\n{'='*70}", flush=True)
    print(f"  {description}", flush=True)
    print(f"  Command: {' '.join(cmd)}", flush=True)
    print(f"  Log: {log_file}", flush=True)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()

    with open(log_file, "w") as lf:
        lf.write(f"# {description}\n")
        lf.write(f"# Command: {' '.join(cmd)}\n")
        lf.write(f"# Started: {datetime.now().isoformat()}\n\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=full_env, text=True, bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

        proc.wait()

    elapsed = time.time() - t0
    elapsed_str = str(timedelta(seconds=int(elapsed)))

    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"\n  {status} in {elapsed_str}", flush=True)

    with open(log_file, "a") as lf:
        lf.write(f"\n# {status} in {elapsed_str}\n")

    if proc.returncode != 0:
        print(f"  !!! PIPELINE HALTED: {description} failed !!!", flush=True)
        return False
    return True


def check_pixel_completeness(band, test_region=False, test_patch=False):
    """Verify all expected HEALPix pixels have Phase 0 output files.

    Returns (ok, missing_pixels) tuple.
    """
    from delve_ubercal.phase0_ingest import (
        get_test_patch_pixels,
        get_test_region_pixels,
        load_config,
    )
    from delve_ubercal.utils.healpix_utils import get_all_healpix_pixels

    config = load_config()
    nside = config["healpix"]["nside"]

    if test_patch:
        expected = set(int(p) for p in get_test_patch_pixels(nside))
    elif test_region:
        expected = set(int(p) for p in get_test_region_pixels(nside))
    else:
        expected = set(int(p) for p in get_all_healpix_pixels(nside))

    phase0_dir = BASE_DIR / "output" / f"phase0_{band}"
    found = set()
    for pf in phase0_dir.glob("detections_nside*_pixel*.parquet"):
        pix = int(pf.stem.split("pixel")[1])
        found.add(pix)

    missing = sorted(expected - found)
    return len(missing) == 0, missing


def run_phases_0_to_3(band, start_phase, region_flag, region_label, parallel, band_log_dir, timestamp):
    """Run Phases 0-3 for a single band (before cross-band detrend)."""
    parallel_flag = ["--parallel", str(parallel)] if region_label == "full-sky" else []

    phases = [
        (0, f"Phase 0: Data ingestion ({band}-band, {region_label})",
         [PYTHON, "-m", "delve_ubercal.phase0_ingest",
          "--band", band] + region_flag + parallel_flag),

        (1, f"Phase 1: Overlap graph ({band}-band)",
         [PYTHON, "-m", "delve_ubercal.phase1_overlap_graph",
          "--band", band] + region_flag),

        (2, f"Phase 2: CG sparse solve ({band}-band, both modes)",
         [PYTHON, "-m", "delve_ubercal.phase2_solve",
          "--band", band, "--mode", "both"] + region_flag),

        (3, f"Phase 3: Outlier rejection ({band}-band)",
         [PYTHON, "-m", "delve_ubercal.phase3_outlier_rejection",
          "--band", band, "--mode", "both"] + region_flag),
    ]

    for phase_num, description, cmd in phases:
        if phase_num < start_phase:
            print(f"  Skipping Phase {phase_num} ({band}-band)", flush=True)
            continue

        log_file = band_log_dir / f"phase{phase_num}_{band}_{timestamp}.log"
        ok = run_command(cmd, log_file, description)
        if not ok:
            return False

        # After Phase 0: verify all pixels were downloaded
        if phase_num == 0:
            test_region = "test-region" in " ".join(region_flag)
            test_patch = "test-patch" in " ".join(region_flag)
            complete, missing = check_pixel_completeness(
                band, test_region=test_region, test_patch=test_patch
            )
            if not complete:
                print(f"\n  !!! PIXEL COMPLETENESS CHECK FAILED !!!", flush=True)
                print(f"  Missing {len(missing)} pixels: {missing}", flush=True)
                print(f"  Re-querying missing pixels...", flush=True)

                for attempt in range(1, 4):
                    print(f"\n  Re-query attempt {attempt}/3...", flush=True)
                    retry_log = band_log_dir / f"phase0_retry{attempt}_{band}_{timestamp}.log"
                    pixel_str = ",".join(str(p) for p in missing)
                    retry_cmd = [
                        PYTHON, "-m", "delve_ubercal.phase0_ingest",
                        "--band", band, "--pixels", pixel_str,
                        "--parallel", "1",
                    ]
                    run_command(retry_cmd, retry_log, f"Phase 0 retry: {len(missing)} missing pixels ({band})")

                    complete, missing = check_pixel_completeness(
                        band, test_region=test_region, test_patch=test_patch
                    )
                    if complete:
                        print(f"  All pixels now present!", flush=True)
                        break
                    else:
                        print(f"  Still missing {len(missing)} pixels: {missing}", flush=True)

                if not complete:
                    print(f"\n  !!! PIPELINE HALTED: {len(missing)} pixels still missing after 3 retries !!!", flush=True)
                    print(f"  Missing: {missing}", flush=True)
                    print(f"  Fix manually and restart with --start-phase 1", flush=True)
                    return False

    return True


def run_phases_3b_to_6(band, start_phase, region_flag, region_label, band_log_dir, timestamp):
    """Run Phase 3b (gradient detrend) through Phase 6 for a single band.

    Phase 3b requires both g and r to have completed Phase 3.
    """
    # Phase numbering: 3.5=3b detrend, 4=starflat, 5=catalog, 6=validation
    phases = [
        (3.5, f"Phase 3b: Gradient detrend ({band}-band, DELVE g-r)",
         [PYTHON, "-m", "delve_ubercal.phase3b_gradient_detrend",
          "--band", band] + region_flag),

        (4, f"Phase 4: Star flat ({band}-band)",
         [PYTHON, "-m", "delve_ubercal.phase4_starflat",
          "--band", band, "--mode", "both"] + region_flag),

        (5, f"Phase 5: Catalog construction ({band}-band)",
         [PYTHON, "-m", "delve_ubercal.phase5_catalog",
          "--band", band] + region_flag),

        (6, f"Phase 6: Validation ({band}-band)",
         [PYTHON, "-m", "delve_ubercal.validation.run_all",
          "--band", band] + region_flag),
    ]

    for phase_num, description, cmd in phases:
        if phase_num < start_phase:
            print(f"  Skipping Phase {phase_num} ({band}-band)", flush=True)
            continue

        log_file = band_log_dir / f"phase{str(phase_num).replace('.', 'p')}_{band}_{timestamp}.log"
        ok = run_command(cmd, log_file, description)
        if not ok:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full DELVE ubercalibration pipeline"
    )
    parser.add_argument(
        "--bands", nargs="+", default=BANDS,
        help=f"Bands to process (default: {BANDS})",
    )
    parser.add_argument(
        "--start-phase", type=int, default=0,
        help="Start from this phase number (0-6)",
    )
    parser.add_argument(
        "--test-region", action="store_true",
        help="Run on test region only",
    )
    parser.add_argument(
        "--test-patch", action="store_true",
        help="Run on 10x10 deg test patch (RA=50-60, Dec=-35 to -25)",
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Parallel query workers for Phase 0 (default 4)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    region = "test patch (10x10 deg)" if args.test_patch else ("test region" if args.test_region else "FULL SKY")

    print(f"\n{'#'*70}", flush=True)
    print(f"#  DELVE Ubercalibration — Full Pipeline Run", flush=True)
    print(f"#  Region: {region}", flush=True)
    print(f"#  Bands: {args.bands}", flush=True)
    print(f"#  Start phase: {args.start_phase}", flush=True)
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    t0_total = time.time()

    if args.test_patch:
        region_flag = ["--test-patch"]
        region_label = "test-patch"
    elif args.test_region:
        region_flag = ["--test-region"]
        region_label = "test-region"
    else:
        region_flag = []
        region_label = "full-sky"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Stage 1: Phases 0-3 for ALL bands (independent per band) ──
    for band in args.bands:
        print(f"\n{'*'*70}", flush=True)
        print(f"*  Phases 0-3: {band}-band", flush=True)
        print(f"*  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'*'*70}", flush=True)

        band_log_dir = LOG_DIR / band
        band_log_dir.mkdir(parents=True, exist_ok=True)

        ok = run_phases_0_to_3(
            band, args.start_phase, region_flag, region_label,
            args.parallel, band_log_dir, timestamp,
        )
        if not ok:
            print(f"\n!!! Pipeline failed on {band}-band Phase 0-3. Stopping. !!!", flush=True)
            sys.exit(1)

    # ── Stage 2: Phase 3b gradient detrend + Phase 4-6 for ALL bands ──
    # Phase 3b requires both g and r bands to have completed Phase 3
    for band in args.bands:
        print(f"\n{'*'*70}", flush=True)
        print(f"*  Phases 3b-6: {band}-band", flush=True)
        print(f"*  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'*'*70}", flush=True)

        band_log_dir = LOG_DIR / band
        ok = run_phases_3b_to_6(
            band, args.start_phase, region_flag, region_label,
            band_log_dir, timestamp,
        )
        if not ok:
            print(f"\n!!! Pipeline failed on {band}-band Phase 3b-6. Stopping. !!!", flush=True)
            sys.exit(1)

        print(f"\n  {band}-band complete!", flush=True)

    total_time = time.time() - t0_total
    total_str = str(timedelta(seconds=int(total_time)))

    print(f"\n{'#'*70}", flush=True)
    print(f"#  PIPELINE COMPLETE", flush=True)
    print(f"#  Total time: {total_str}", flush=True)
    print(f"#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#'*70}\n", flush=True)


if __name__ == "__main__":
    main()
