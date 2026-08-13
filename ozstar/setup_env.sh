#!/usr/bin/env bash
# Create or refresh the two OzSTAR checkouts and the shared CPU environment.
set -euo pipefail

BASE=/fred/oz200/avajpeyi/projects/WDM_PSD
PACKAGE_DIR="$BASE/wdm_psd"
LISA_DIR="$BASE/tvpsd_lisa_demo"

mkdir -p "$BASE"
if [[ ! -d "$PACKAGE_DIR/.git" ]]; then
    git clone git@github.com:nz-gravity/TVPsplinePSD.git "$PACKAGE_DIR"
else
    git -C "$PACKAGE_DIR" pull --ff-only
fi
if [[ ! -d "$LISA_DIR/.git" ]]; then
    git clone git@github.com:nz-gravity/tvpsd_lisa_demo.git "$LISA_DIR"
else
    git -C "$LISA_DIR" pull --ff-only
fi

module load gcc/13.3.0 python/3.12.3
if [[ ! -d "$BASE/venv" ]]; then
    python -m venv "$BASE/venv"
fi
source "$BASE/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --editable "$PACKAGE_DIR[lisa]"
mkdir -p "$LISA_DIR/ozstar/logs" "$LISA_DIR/results"

echo "environment ready: $BASE/venv"
