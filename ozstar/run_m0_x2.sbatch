#!/usr/bin/env bash
#SBATCH --job-name=m0-x2
#SBATCH --account=oz200
#SBATCH --partition=milan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo/ozstar/logs/m0_x2_%j.log
#
# M0 (rungs 1-2), X2 channel, continuous. ~1185s / 854s NUTS locally
# (esa_m0_study.py's own production numbers). This is the ONLY channel
# esa_m0_study.py currently supports -- it has no --channel argument and
# reads tdi/total index 0 (X2) directly. A/E/T jobs for M0 do not exist yet;
# see ../README.md "Known issues".
#
#   sbatch ozstar/run_m0_x2.sbatch
set -euo pipefail

BASE=/fred/oz200/avajpeyi/projects/WDM_PSD
LISA_DIR="$BASE/tvpsd_lisa_demo"
mkdir -p "$LISA_DIR/ozstar/logs" "$LISA_DIR/results"

module load gcc/13.3.0 python/3.12.3
source "$BASE/venv/bin/activate"
cd "$LISA_DIR"

export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python esa_m0_study.py \
    --mode continuous \
    --output-dir "results/m0_x2_${SLURM_JOB_ID}"
