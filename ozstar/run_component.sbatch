#!/usr/bin/env bash
#SBATCH --job-name=h_para-aet
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/h_para_%j.log
#
# H_para (rung 3, component separation), joint over A/E/T, on the surface study's grid/protocol.
# ~203s locally at nt=2048/sub-null-band with the likelihood preconditioner;
# full 0.1 Hz band and 500/500 warmup/samples will take longer -- the
# ~2850s full-band run seen locally undershot the cluster: job 15420119
# ran the full 2h ceiling without finishing, so this ceiling is now 6h.
# Job 15434685 (and again 15501101, after --mem regressed back to 16G) was
# OOM-killed at 16G partway through the second chain; 32G is required.
#
#   sbatch ozstar/run_component.sbatch [continuous|gapped] [gap-centre-year]
#
# The gap centre defaults to mid-year (0.5). That is a quiet stretch of the
# Galactic modulation -- 1.7x the annual minimum, log-slope 0.33/yr -- so a
# gap there is the easy case. Pass 0.8 to place it on the modulation peak
# (3.0x minimum), which is the harder and more persuasive test.
#
# --nt is passed explicitly, not left to the runner default: the lisa_like gap
# schedule is a function of the record length, which depends on nt, so the three
# models only see the same outages while this matches run_surface.sbatch's --nt.
# The preflight asserts the two runners agree before any fit starts.
#
# The gapped mode matches run_surface.sbatch's -- one seven-day gap, one-hour cosine
# taper, one-pixel edge buffer -- so with the same centre the the surface study and H_para gap
# results describe the same outage.
set -euo pipefail

MODE="${1:-continuous}"
GAP_CENTRE="${2:-0.5}"   # gapped: centre in years
GAP_SEED="${2:-1}"       # duty:   gap-schedule seed
BUFFER="${3:-1}"       # WDM pixels excluded at each gap edge
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

# Tag encodes the gap centre so two gapped runs cannot overwrite each other.
if [ "$MODE" = "gapped" ]; then
    TAG="gapped_c${GAP_CENTRE}_b${BUFFER}"
elif [ "$MODE" = "duty" ]; then
    TAG="duty_s${GAP_SEED}_b${BUFFER}"
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
# Pin the shared estimator to the sibling checkout, exactly as the surface-study jobs do:
# a rung compared against the surface study must not import a different tv_pspline_psd.
export PYTHONPATH="$BASE/TVPsplinePSD${PYTHONPATH:+:$PYTHONPATH}"

# surface-named, but the checks that matter here are shared: package provenance,
# archive/orbit presence, and inference/evaluation mask separation.
python -u ozstar/preflight_surface.py --base "$BASE"

mkdir -p "$LISA_DIR/results"
# -u and the progress bar (default on; not passing --no-progress) mean NUTS
# warmup/sampling progress streams into the log live instead of only
# appearing once the run finishes.
# --wdm-projection-nodes MUST match the surface runs this fit is tabulated beside:
# the projected estimand is what makes the surface study and H_para coverage comparable at all.
python -u run_component_study.py \
    "${MODE_ARGS[@]}" \
    --output "results/h_para_aet_${TAG}_${SLURM_JOB_ID}.npz" \
    --surface-plot "results/h_para_aet_${TAG}_${SLURM_JOB_ID}.png" \
    --parameter-plot "results/h_para_aet_${TAG}_${SLURM_JOB_ID}_params.png" \
    --nt 2048 \
    --wdm-projection-nodes 16 \
    --warmup 500 --samples 500
