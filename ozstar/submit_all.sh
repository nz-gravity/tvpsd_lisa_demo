#!/usr/bin/env bash
# Submit the paper's hypothesis ladder (H_agn, H_orb, H_para).
#
#   ozstar/submit_all.sh                     # all three scenarios (17 jobs)
#   ozstar/submit_all.sh continuous          # one or more of: continuous gapped duty
#   ozstar/submit_all.sh gapped duty
#
# Each scenario is one full ladder: the X2 anchor (where one exists), A and E
# under H_orb and H_agn, and H_para jointly over A/E/T. Re-running submits
# duplicates -- check `squeue -u $USER` first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The fits read truth/*_csd. An archive predating the XYZ cross-spectrum fix
# lacks it, and every job would queue, start, then die on the missing dataset.
python - <<'PY' || { echo "ERROR: combined_esa_xyz.h5 has no truth/galactic_csd -- run ozstar/sync_data.sh" >&2; exit 1; }
import sys, h5py
with h5py.File("combined_esa_xyz.h5", "r") as hdf:
    sys.exit(0 if "truth/galactic_csd" in hdf else 1)
PY

submit() {
    if command -v sbatch >/dev/null; then
        sbatch "$@"
    else
        echo "  [dry run, no sbatch] sbatch $*"
    fi
}

scenarios=("$@")
[[ ${#scenarios[@]} -eq 0 ]] && scenarios=(continuous gapped duty)

for scenario in "${scenarios[@]}"; do
    echo "== ${scenario} =="
    case "${scenario}" in
        continuous) mode=() ;;
        gapped)     mode=(gapped) ;;
        duty)       mode=(duty) ;;
        *) echo "unknown scenario: ${scenario} (expected continuous|gapped|duty)" >&2; exit 1 ;;
    esac

    # X2 methods anchors exist for the continuous and single-gap geometries only.
    case "${scenario}" in
        continuous) submit ozstar/run_surface_x2.sbatch ;;
        gapped)     submit ozstar/run_surface_x2_gap7.sbatch ;;
    esac

    for hypothesis in orb agn; do
        for channel in A E; do
            submit ozstar/run_surface.sbatch "${channel}" "${hypothesis}" ${mode[@]+"${mode[@]}"}
        done
    done
    submit ozstar/run_component.sbatch ${mode[@]+"${mode[@]}"}
done
