#!/usr/bin/env bash
#SBATCH --job-name=m1-aet
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/m1_%j.log
#
# M1 (rung 3, component separation), joint over A/E/T, on M0's grid/protocol.
# ~203s locally at nt=2048/sub-null-band with the likelihood preconditioner;
# full 0.1 Hz band and 500/500 warmup/samples will take longer -- the
# ~2850s full-band run seen locally undershot the cluster: job 15420119
# ran the full 2h ceiling without finishing, so this ceiling is now 6h.
# Job 15434685 (and again 15501101, after --mem regressed back to 16G) was
# OOM-killed at 16G partway through the second chain; 32G is required.
#
#   sbatch ozstar/run_m1.sbatch [continuous|gapped] [gap-centre-year]
#
# The gap centre defaults to mid-year (0.5). That is a quiet stretch of the
# Galactic modulation -- 1.7x the annual minimum, log-slope 0.33/yr -- so a
# gap there is the easy case. Pass 0.8 to place it on the modulation peak
# (3.0x minimum), which is the harder and more persuasive test.
#
# The gapped mode matches run_m0.sbatch's -- one seven-day gap, one-hour cosine
# taper, one-pixel edge buffer -- so with the same centre the M0 and M1 gap
# results describe the same outage.
set -euo pipefail

MODE="${1:-continuous}"
GAP_CENTRE="${2:-0.5}"
case "$MODE" in
    continuous) MODE_ARGS=(--mode continuous) ;;
    gapped)
        MODE_ARGS=(
            --mode gapped
            --gap-scenario single
            --single-gap-days 7
            --single-gap-center-year "$GAP_CENTRE"
            --gap-buffer-pixels 1
            --taper-hours 1
        )
        ;;
    *)
        echo "unknown mode '$MODE' (expected 'continuous' or 'gapped')" >&2
        exit 2
        ;;
esac

# Tag encodes the gap centre so two gapped runs cannot overwrite each other.
if [ "$MODE" = "gapped" ]; then
    TAG="gapped_c${GAP_CENTRE}"
else
    TAG="$MODE"
fi

BASE=/fred/oz200/avajpeyi/projects/WDM_PSD
LISA_DIR="$BASE/tvpsd_lisa_demo"
mkdir -p "$LISA_DIR/ozstar/logs"

module load gcc/13.3.0 python/3.12.3
source "$BASE/venv/bin/activate"
cd "$LISA_DIR"

# JAX respects this for CPU intra-op parallelism; match --cpus-per-task.
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# Pin the shared estimator to the sibling checkout, exactly as the M0 jobs do:
# a rung compared against M0 must not import a different tv_pspline_psd.
export PYTHONPATH="$BASE/TVPsplinePSD${PYTHONPATH:+:$PYTHONPATH}"

# M0-named, but the checks that matter here are shared: package provenance,
# archive/orbit presence, and inference/evaluation mask separation.
python -u ozstar/preflight_m0.py --base "$BASE"

mkdir -p "$LISA_DIR/results"
# -u and the progress bar (default on; not passing --no-progress) mean NUTS
# warmup/sampling progress streams into the log live instead of only
# appearing once the run finishes.
# --wdm-projection-nodes MUST match the M0 runs this fit is tabulated beside:
# the projected estimand is what makes M0 and M1 coverage comparable at all.
python -u run_aet_diagonal_pilot.py \
    "${MODE_ARGS[@]}" \
    --output "results/m1_aet_${TAG}_${SLURM_JOB_ID}.npz" \
    --surface-plot "results/m1_aet_${TAG}_${SLURM_JOB_ID}.png" \
    --parameter-plot "results/m1_aet_${TAG}_${SLURM_JOB_ID}_params.png" \
    --wdm-projection-nodes 16 \
    --warmup 500 --samples 500
