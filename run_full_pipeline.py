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


def run_pipeline_for_band(band, start_phase=0, test_region=False, parallel=8):
    """Run all phases for a single band."""
    region_flag = ["--test-region"] if test_region else []
    region_label = "test-region" if test_region else "full-sky"
    parallel_flag = ["--parallel", str(parallel)] if not test_region else []

    band_log_dir = LOG_DIR / band
    band_log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

        log_file = band_log_dir / f"phase{phase_num}_{band}_{timestamp}.log"
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
        "--parallel", type=int, default=4,
        help="Parallel query workers for Phase 0 (default 4)",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    region = "test region" if args.test_region else "FULL SKY"

    print(f"\n{'#'*70}", flush=True)
    print(f"#  DELVE Ubercalibration — Full Pipeline Run", flush=True)
    print(f"#  Region: {region}", flush=True)
    print(f"#  Bands: {args.bands}", flush=True)
    print(f"#  Start phase: {args.start_phase}", flush=True)
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'#'*70}\n", flush=True)

    t0_total = time.time()

    for band in args.bands:
        print(f"\n{'*'*70}", flush=True)
        print(f"*  Starting band: {band}", flush=True)
        print(f"*  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'*'*70}", flush=True)

        ok = run_pipeline_for_band(
            band,
            start_phase=args.start_phase,
            test_region=args.test_region,
            parallel=args.parallel,
        )
        if not ok:
            print(f"\n!!! Pipeline failed on {band}-band. Stopping. !!!", flush=True)
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
