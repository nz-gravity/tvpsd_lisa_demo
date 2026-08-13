#!/usr/bin/env bash
#SBATCH --job-name=m1-aet
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/m1_%j.log
#
# M1 (rung 3, component separation), joint over A/E/T, on M0's grid/protocol.
# ~203s locally at nt=2048/sub-null-band with the likelihood preconditioner;
# full 0.1 Hz band and 500/500 warmup/samples will take longer -- 2h is a
# generous ceiling based on the ~2850s full-band run seen locally.
#
#   sbatch ozstar/run_m1.sbatch
set -euo pipefail

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
export PYTHONPATH="$BASE/wdm_psd${PYTHONPATH:+:$PYTHONPATH}"

# M0-named, but the checks that matter here are shared: package provenance,
# archive/orbit presence, and inference/evaluation mask separation.
python ozstar/preflight_m0.py --base "$BASE"

mkdir -p "$LISA_DIR/results"
python run_aet_diagonal_pilot.py \
    --output "results/m1_aet_${SLURM_JOB_ID}.npz" \
    --surface-plot "results/m1_aet_${SLURM_JOB_ID}.png" \
    --parameter-plot "results/m1_aet_${SLURM_JOB_ID}_params.png" \
    --warmup 500 --samples 500 --no-progress
