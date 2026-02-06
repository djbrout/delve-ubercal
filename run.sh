#!/bin/bash
# DELVE Ubercalibration Pipeline
# Run phases sequentially. Each depends on the previous.
# Add --test-region to limit to the test region (RA=50-70, Dec=-40 to -25).

set -e

BANDS="g r i z"
TEST_FLAGS=""  # Add "--test-region" for test mode

echo "=== DELVE Ubercalibration Pipeline ==="

# Phase 0: Data Ingestion
echo "--- Phase 0: Data Ingestion ---"
for band in $BANDS; do
    echo "  Processing band $band..."
    python -m delve_ubercal.phase0_ingest --band "$band" $TEST_FLAGS
done

# Phase 1: Overlap Graph
echo "--- Phase 1: Overlap Graph ---"
for band in $BANDS; do
    python -m delve_ubercal.phase1_overlap_graph --band "$band" $TEST_FLAGS
done

# Phase 2: Solve (both modes)
echo "--- Phase 2: Solve ---"
for band in $BANDS; do
    python -m delve_ubercal.phase2_solve --band "$band" --mode unanchored $TEST_FLAGS
    python -m delve_ubercal.phase2_solve --band "$band" --mode anchored $TEST_FLAGS
done

# Phase 3: Outlier Rejection
echo "--- Phase 3: Outlier Rejection ---"
for band in $BANDS; do
    python -m delve_ubercal.phase3_outlier_rejection --band "$band" $TEST_FLAGS
done

# Phase 4: Star Flat
echo "--- Phase 4: Star Flat ---"
for band in $BANDS; do
    python -m delve_ubercal.phase4_starflat --band "$band" $TEST_FLAGS
done

# Phase 5: Catalog
echo "--- Phase 5: Catalog ---"
python -m delve_ubercal.phase5_catalog $TEST_FLAGS

# Phase 6: Validation
echo "--- Phase 6: Validation ---"
python -m delve_ubercal.validation.run_all $TEST_FLAGS

echo "=== Pipeline Complete ==="
