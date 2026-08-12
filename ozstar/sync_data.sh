#!/usr/bin/env bash
# Run this LOCALLY (not on OzSTAR) to copy the ~1.6 GB of inputs that git
# deliberately does not track (see ../.gitignore): the archive, instrument
# noise, orbits, and galactic-binary inputs.
#
# Usage: bash ozstar/sync_data.sh [ozstar-login-node]
# Default login node is ozstar; override if your ~/.ssh/config uses a
# different host alias.
set -euo pipefail

HOST="${1:-ozstar}"
REMOTE_DIR="/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "syncing from $LOCAL_DIR to $HOST:$REMOTE_DIR"
rsync -avh --progress "$LOCAL_DIR/combined_esa_xyz.h5" "$HOST:$REMOTE_DIR/combined_esa_xyz.h5"
rsync -avh --progress "$LOCAL_DIR/noise2a/" "$HOST:$REMOTE_DIR/noise2a/"
rsync -avh --progress "$LOCAL_DIR/gb1/" "$HOST:$REMOTE_DIR/gb1/"

echo "done. ~1.6 GB total (1.0G archive + 483M noise2a + 121M gb1)."
