#!/usr/bin/env python3
"""Quick progress check for the DELVE ubercal pipeline."""
import sys
from pathlib import Path
from datetime import datetime

BASE = Path("/Volumes/External5TB/DELVE_UBERCAL")
BANDS = ["g", "r", "i", "z"]
TOTAL_PIXELS = 9664  # after dec < 35 filter

def check_phase0(band):
    out_dir = BASE / f"output/phase0_{band}"
    if not out_dir.exists():
        return 0, 0, 0
    files = list(out_dir.glob("detections_nside32_pixel*.parquet"))
    n_data = 0
    n_empty = 0
    for f in files:
        if f.stat().st_size > 500:
            n_data += 1
        else:
            n_empty += 1
    return len(files), n_data, n_empty

def check_later_phases(band):
    results = {}
    phase_outputs = {
        1: f"output/phase1_{band}/overlap_graph.parquet",
        2: f"output/phase2_{band}/zeropoints_anchored.parquet",
        3: f"output/phase3_{band}/zeropoints_anchored.parquet",
        4: f"output/phase4_{band}/starflat_corrections.parquet",
        5: f"output/phase5_{band}/catalog_{band}.parquet",
        6: f"output/validation_{band}/summary.json",
    }
    for phase, path in phase_outputs.items():
        p = BASE / path
        results[phase] = p.exists()
    return results

print(f"\n{'='*60}")
print(f"  DELVE Ubercal Pipeline Progress")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}\n")

# Check running processes
import subprocess
result = subprocess.run(
    ["ps", "aux"], capture_output=True, text=True
)
running = [l for l in result.stdout.split("\n")
           if "phase" in l.lower() and "grep" not in l]
if running:
    for r in running:
        parts = r.split()
        cmd = " ".join(parts[10:])
        print(f"  Running: {cmd[:70]}")
    print()

# Check log tail
for band in BANDS:
    log = BASE / f"logs/pipeline_full_20260206.log"
    if log.exists():
        lines = log.read_text().strip().split("\n")
        last = [l for l in lines[-5:] if l.strip()]
        if last:
            print(f"  Last log: {last[-1].strip()}")
        break

print()
for band in BANDS:
    total, n_data, n_empty = check_phase0(band)
    phases = check_later_phases(band)

    pct = 100 * total / TOTAL_PIXELS if TOTAL_PIXELS > 0 else 0
    status = "not started"
    if total > 0 and total < TOTAL_PIXELS:
        status = f"Phase 0: {total}/{TOTAL_PIXELS} pixels ({pct:.1f}%) [{n_data} data, {n_empty} empty]"
    elif total >= TOTAL_PIXELS:
        # Check which phase we're on
        last_done = 0
        for p in range(1, 7):
            if phases.get(p, False):
                last_done = p
        if last_done == 6:
            status = "COMPLETE (all phases)"
        elif last_done > 0:
            status = f"Phase {last_done} done, Phase {last_done+1} in progress"
        else:
            status = f"Phase 0 done ({n_data} data, {n_empty} empty), Phase 1 pending"

    print(f"  {band}-band: {status}")

print(f"\n{'='*60}\n")
