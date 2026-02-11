#!/bin/bash
# Wait for Data Lab to come back, then restart the pipeline
export PYTHONPATH=/Volumes/External5TB/DELVE_UBERCAL
export PYTHONUNBUFFERED=1

echo "Waiting for Data Lab to come back..."
while true; do
    result=$(python3 -c "
from dl import queryClient as qc
try:
    r = qc.query(sql='SELECT 1 as test', fmt='pandas', timeout=30)
    print('UP')
except:
    print('DOWN')
" 2>&1)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Data Lab: $result"
    if [ "$result" = "UP" ]; then
        echo "Data Lab is back! Starting pipeline..."
        break
    fi
    sleep 300  # check every 5 minutes
done

# Download lookup tables for r, i, z
echo "Pre-downloading lookup tables..."
python3 -c "
from delve_ubercal.phase0_ingest import download_lookup_tables, load_config
from pathlib import Path
config = load_config()
cache_dir = Path(config['data']['cache_path'])
for band in ['r', 'i', 'z']:
    print(f'Downloading {band}-band...', flush=True)
    try:
        chip_df, exp_df = download_lookup_tables(band, config, cache_dir)
        print(f'  OK: {len(chip_df):,} chip, {len(exp_df):,} exposure', flush=True)
    except Exception as e:
        print(f'  Failed: {e}', flush=True)
"

# Start the full pipeline
echo "Starting full pipeline..."
python3 /Volumes/External5TB/DELVE_UBERCAL/run_full_pipeline.py --bands g r i z --parallel 4
