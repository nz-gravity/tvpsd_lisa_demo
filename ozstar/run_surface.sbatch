#!/usr/bin/env bash
#SBATCH --job-name=surface
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/surface_%j.log
#
# H_agn/H_orb for the paper's ladder rungs 1 and 2, on one channel.
#
#   sbatch ozstar/run_surface.sbatch <channel> <hypothesis> [mode]
#     channel    : A | E | T | X2 | Y2 | Z2
#     hypothesis : orb -- H_orb, rung 2, response-informed. Fits R+g and
#                         R+g+h and scores both against the fixed reference R.
#                  agn -- H_agn, rung 1, reference-agnostic free total surface.
#                  (`ref`/`free` remain accepted as the old spellings.)
#     mode    : continuous (default) | gapped
#     centre  : gap centre in years (default 0.5); 0.8 puts it on the
#               Galactic modulation peak rather than a quiet stretch
#
# The gapped mode injects one seven-day gap at mid-year and excludes one WDM
# pixel at each edge, matching run_surface_x2_gap7.sbatch so the gapped A/E runs are
# comparable to the frozen X2 gap anchor.
#
# Every scientific setting is spelled out here and shared by both rungs except
# the three lines that define the rung itself, so the two fits differ only in
# what they assume. run_surface_x2.sbatch remains the explicit X2 methods anchor.
#
# NOTE ON TIME FLEXIBILITY. In the nested (ref) rung the total surface's time
# dependence comes solely from the five-knot zero-time-mean interaction basis;
# --time-knots governs the unrestricted tensor path used by the free rung. The
# free rung is therefore the more flexible fit in time, which is what a
# reference-agnostic robustness check should be. Set --time-knots 5 below if
# you would rather match the two exactly.
set -euo pipefail

CHANNEL="${1:?usage: sbatch run_surface.sbatch <channel> <orb|agn> [continuous|gapped]}"
RUNG="${2:?usage: sbatch run_surface.sbatch <channel> <orb|agn> [continuous|gapped]}"
MODE="${3:-continuous}"
GAP_CENTRE="${4:-0.5}"   # gapped: centre in years
GAP_SEED="${4:-1}"       # duty:   gap-schedule seed
BUFFER="${5:-1}"       # WDM pixels excluded at each gap edge

case "$RUNG" in
    orb|ref)
        RUNG_ARGS=(
            --hypothesis orb
            --residual-structure stationary_plus_interaction
            --interaction-time-knots 5
            --interaction-scale-prior 0.5
        )
        ;;
    agn|free)
        # H_agn: no reference offset, log S is a single unrestricted tensor surface.
        RUNG_ARGS=(--hypothesis agn --residual-structure tensor)
        ;;
    *)
        echo "unknown rung '$RUNG' (expected 'ref' or 'free')" >&2
        exit 2
        ;;
esac

case "$MODE" in
    continuous) MODE_ARGS=(--mode continuous) ;;
    gapped)
        MODE_ARGS=(
            --mode gapped
            --gap-scenario single
            --single-gap-days 7
            --single-gap-center-year "$GAP_CENTRE"
            --gap-buffer-pixels "$BUFFER"
            --taper-hours 1
        )
        ;;
    duty)
        # Scheduled 3.5 h repointing every 14 d plus Poisson unscheduled
        # outages: ~60 gaps and ~5% duty-cycle loss, which costs ~13% of WDM
        # rows because every gap pays a taper and a buffer at each edge.
        MODE_ARGS=(
            --mode gapped
            --gap-scenario lisa_like
            --gap-seed "$GAP_SEED"
            --gap-buffer-pixels "$BUFFER"
            --taper-hours 1
        )
        ;;
    *)
        echo "unknown mode '$MODE' (expected continuous, gapped or duty)" >&2
        exit 2
        ;;
esac

if [ "$MODE" = "gapped" ]; then
    TAG="gapped_c${GAP_CENTRE}_b${BUFFER}"
elif [ "$MODE" = "duty" ]; then
    TAG="duty_s${GAP_SEED}_b${BUFFER}"
else
    TAG="$MODE"
fi

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
    --channel "$CHANNEL" \
    "${MODE_ARGS[@]}" \
    --archive "$LISA_DIR/combined_esa_xyz.h5" \
    --orbits "$LISA_DIR/noise2a/orbits.h5" \
    --output-dir "results/surface_${CHANNEL}_${RUNG}_${TAG}_${SLURM_JOB_ID}" \
    --fmin 1e-4 --fmax 1e-1 --nt 2048 \
    --time-knots 8 --frequency-knots 36 --frequency-knot-break 1e-2 \
    --time-bin 4 --split-cycle 7 --validation-fold 5 --test-fold 6 \
    --pilot-time-profiles 32 --pilot-frequency-width 31 \
    --max-log-range 0.25 --max-frequency-bin 24 \
    --wdm-projection-nodes 16 --truth-frequency-chunk 384 \
    --predictive-frequency-chunk 256 --score-bootstrap-replicates 4000 \
    --null-min-frequency 0.02 --null-ratio-threshold 0.35 \
    --null-continuum-width 81 --null-dilation-bins 6 \
    "${RUNG_ARGS[@]}" \
    --n-warmup 500 --n-samples 500 --chains 4 \
    --max-tree-depth 12 --target-accept 0.95 --no-progress
