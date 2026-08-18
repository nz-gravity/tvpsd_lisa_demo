#!/usr/bin/env bash
#SBATCH --job-name=surface-x2-gap7
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/surface_x2_gap7_%j.log
#
# Corrected H_orb, X2 channel, one seven-day gap and one WDM-pixel
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
export PYTHONPATH="$BASE/TVPsplinePSD${PYTHONPATH:+:$PYTHONPATH}"

python ozstar/preflight_surface.py --base "$BASE"

python run_surface_study.py \
    --channel X2 \
    --mode gapped --gap-scenario single \
    --single-gap-days 7 --single-gap-center-year 0.5 \
    --gap-buffer-pixels 1 --taper-hours 1 \
    --archive "$LISA_DIR/combined_esa_xyz.h5" \
    --orbits "$LISA_DIR/noise2a/orbits.h5" \
    --output-dir "results/surface_x2_gap7_projected_${SLURM_JOB_ID}" \
    --fmin 1e-4 --fmax 1e-1 --nt 2048 \
    --time-knots 8 --frequency-knots 36 --frequency-knot-break 1e-2 \
    --time-bin 4 --split-cycle 7 --validation-fold 5 --test-fold 6 \
    --pilot-time-profiles 32 --pilot-frequency-width 31 \
    --max-log-range 0.25 --max-frequency-bin 24 \
    --wdm-projection-nodes 16 --truth-frequency-chunk 384 \
    --predictive-frequency-chunk 256 --score-bootstrap-replicates 4000 \
    --null-min-frequency 0.02 --null-ratio-threshold 0.35 \
    --null-continuum-width 81 --null-dilation-bins 6 \
    --hypothesis orb \
    --residual-structure stationary_plus_interaction \
    --interaction-time-knots 5 --interaction-scale-prior 0.5 \
    --n-warmup 500 --n-samples 500 --chains 4 \
    --max-tree-depth 12 --target-accept 0.95 --no-progress
