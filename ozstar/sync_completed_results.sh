#!/usr/bin/env bash
# Pull completed job results + logs from OzStar down to a local destination.
#
# Covers the jobs that finished cleanly so far:
#   15420117  m0_A_free_15420117
#   15420118  m0_E_free_15420118
#   15420115  m0_A_ref_15420115
#   15420116  m0_E_ref_15420116
#   15420114  m0_x2_corrected_15420114
#
# Run from your LOCAL machine (not on OzStar):
#   ./sync_completed_results.sh [local_dest_dir]
#
# Requires an OzStar SSH alias/host reachable as `ozstar` (or pass
# REMOTE_HOST=user@host.swin.edu.au as an env var override).
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ozstar}"
REMOTE_BASE="/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo"
LOCAL_DEST="${1:-./ozstar_results}"

JOBS=(
    "m0_A_free_15420117"
    "m0_E_free_15420118"
    "m0_A_ref_15420115"
    "m0_E_ref_15420116"
    "m0_x2_corrected_15420114"
)

mkdir -p "$LOCAL_DEST/results" "$LOCAL_DEST/logs"

for job_dir in "${JOBS[@]}"; do
    echo "== syncing results/${job_dir} =="
    rsync -avh --progress \
        "${REMOTE_HOST}:${REMOTE_BASE}/results/${job_dir}/" \
        "${LOCAL_DEST}/results/${job_dir}/"
done

echo "== syncing logs =="
rsync -avh --progress \
    --include="m0_15420115.log" \
    --include="m0_15420116.log" \
    --include="m0_15420117.log" \
    --include="m0_15420118.log" \
    --include="m0_x2_15420114.log" \
    --exclude="*" \
    "${REMOTE_HOST}:${REMOTE_BASE}/ozstar/logs/" \
    "${LOCAL_DEST}/logs/"

echo "Done. Synced to ${LOCAL_DEST}"
