#!/usr/bin/env bash
#SBATCH --job-name=m0-x2-gap7
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/m0_x2_gap7_%j.log
#
# Corrected M0 rung 2, X2 channel, one seven-day gap and one WDM-pixel
# exclusion at each edge. All scientific settings are explicit.
set -euo pipefail

BASE=/fred/oz200/avajpeyi/projects/WDM_PSD
LISA_DIR="$BASE/tvpsd_lisa_demo"
mkdir -p "$LISA_DIR/ozstar/logs" "$LISA_DIR/results"

module load gcc/13.3.0 python/3.12.3
source "$BASE/venv/bin/activate"
cd "$LISA_DIR"

export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$BASE/wdm_psd${PYTHONPATH:+:$PYTHONPATH}"

python ozstar/preflight_m0.py --base "$BASE"

python esa_m0_study.py \
    --channel X2 \
    --mode gapped --gap-scenario single \
    --single-gap-days 7 --single-gap-center-year 0.5 \
    --gap-buffer-pixels 1 --taper-hours 1 \
    --archive "$LISA_DIR/combined_esa_xyz.h5" \
    --orbits "$LISA_DIR/noise2a/orbits.h5" \
    --output-dir "results/m0_x2_gap7_corrected_${SLURM_JOB_ID}" \
    --fmin 1e-4 --fmax 1e-1 --nt 2048 \
    --time-knots 8 --frequency-knots 36 --frequency-knot-break 1e-2 \
    --time-bin 4 --split-cycle 7 --validation-fold 5 --test-fold 6 \
    --pilot-time-profiles 32 --pilot-frequency-width 31 \
    --max-log-range 0.25 --max-frequency-bin 24 \
    --null-min-frequency 0.02 --null-ratio-threshold 0.35 \
    --null-continuum-width 81 --null-dilation-bins 6 \
    --reference-psd-offset \
    --residual-structure stationary_plus_interaction \
    --interaction-time-knots 5 --interaction-scale-prior 0.5 \
    --n-warmup 300 --n-samples 300 --chains 4 \
    --max-tree-depth 12 --target-accept 0.95 --no-progress
