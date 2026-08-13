#!/usr/bin/env bash
# Incrementally pull all OzSTAR result artifacts and scheduler logs.
#
# Run from the local machine:
#   ./ozstar/sync_completed_results.sh [local_results_directory]
#
# With no argument, results land directly in lisa_data_generation/results/
# (not results/results/). Re-running is safe: rsync transfers only files that
# are new or changed. This script deliberately has no --delete flag, so local
# artifacts are never removed when the remote tree changes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-avajpeyi@nt.swin.edu.au}"
REMOTE_BASE="${REMOTE_BASE:-/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo}"
LOCAL_RESULTS="${1:-$STUDY_DIR/results}"
LOCAL_LOGS="$LOCAL_RESULTS/logs"

mkdir -p "$LOCAL_RESULTS" "$LOCAL_LOGS"

# Keep temporary local-transfer files out of the visible result tree. A file is
# atomically moved into place at the end of its transfer; the next invocation
# resumes any interrupted copy.
RSYNC_OPTIONS=(
    -avh
    --progress
    --partial
    --partial-dir=.rsync-partial
    --delay-updates
)

echo "== syncing OzSTAR result artifacts =="
rsync "${RSYNC_OPTIONS[@]}" \
    "${REMOTE_HOST}:${REMOTE_BASE}/results/" \
    "$LOCAL_RESULTS/"

echo "== syncing OzSTAR scheduler logs =="
rsync "${RSYNC_OPTIONS[@]}" \
    "${REMOTE_HOST}:${REMOTE_BASE}/ozstar/logs/" \
    "$LOCAL_LOGS/"

echo "Done. Results: $LOCAL_RESULTS"
echo "Logs:    $LOCAL_LOGS"
