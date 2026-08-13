#!/usr/bin/env bash
#SBATCH --job-name=m0
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/m0_%j.log
#
# M0 for the paper's ladder rungs 1 and 2, on one channel.
#
#   sbatch ozstar/run_m0.sbatch <channel> <rung>
#     channel : A | E | T | X2 | Y2 | Z2
#     rung    : ref  -- rung 2, response-informed. Fits R+g and R+g+h and
#                       scores both against the fixed reference R.
#               free -- rung 1, reference-agnostic free total TV surface.
#
# Every scientific setting is spelled out here and shared by both rungs except
# the three lines that define the rung itself, so the two fits differ only in
# what they assume. run_m0_x2.sbatch is the frozen corrected X2 anchor and is
# deliberately NOT replaced by this script.
#
# NOTE ON TIME FLEXIBILITY. In the nested (ref) rung the total surface's time
# dependence comes solely from the five-knot zero-time-mean interaction basis;
# --time-knots governs the unrestricted tensor path used by the free rung. The
# free rung is therefore the more flexible fit in time, which is what a
# reference-agnostic robustness check should be. Set --time-knots 5 below if
# you would rather match the two exactly.
set -euo pipefail

CHANNEL="${1:?usage: sbatch run_m0.sbatch <channel> <ref|free>}"
RUNG="${2:?usage: sbatch run_m0.sbatch <channel> <ref|free>}"

case "$RUNG" in
    ref)
        RUNG_ARGS=(
            --reference-psd-offset
            --residual-structure stationary_plus_interaction
            --interaction-time-knots 5
            --interaction-scale-prior 0.5
        )
        ;;
    free)
        # No reference offset: log S is a single unrestricted tensor surface.
        RUNG_ARGS=(--residual-structure tensor)
        ;;
    *)
        echo "unknown rung '$RUNG' (expected 'ref' or 'free')" >&2
        exit 2
        ;;
esac

BASE=/fred/oz200/avajpeyi/projects/WDM_PSD
LISA_DIR="$BASE/tvpsd_lisa_demo"
mkdir -p "$LISA_DIR/ozstar/logs" "$LISA_DIR/results"

module load gcc/13.3.0 python/3.12.3
source "$BASE/venv/bin/activate"
cd "$LISA_DIR"

export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$BASE/TVPsplinePSD${PYTHONPATH:+:$PYTHONPATH}"

python ozstar/preflight_m0.py --base "$BASE"

python esa_m0_study.py \
    --channel "$CHANNEL" \
    --mode continuous \
    --archive "$LISA_DIR/combined_esa_xyz.h5" \
    --orbits "$LISA_DIR/noise2a/orbits.h5" \
    --output-dir "results/m0_${CHANNEL}_${RUNG}_${SLURM_JOB_ID}" \
    --fmin 1e-4 --fmax 1e-1 --nt 2048 \
    --time-knots 8 --frequency-knots 36 --frequency-knot-break 1e-2 \
    --time-bin 4 --split-cycle 7 --validation-fold 5 --test-fold 6 \
    --pilot-time-profiles 32 --pilot-frequency-width 31 \
    --max-log-range 0.25 --max-frequency-bin 24 \
    --null-min-frequency 0.02 --null-ratio-threshold 0.35 \
    --null-continuum-width 81 --null-dilation-bins 6 \
    "${RUNG_ARGS[@]}" \
    --n-warmup 300 --n-samples 300 --chains 4 \
    --max-tree-depth 12 --target-accept 0.95 --no-progress
