# OzSTAR

```
ssh ozstar
mkdir -p /fred/oz200/avajpeyi/projects/WDM_PSD
cd /fred/oz200/avajpeyi/projects/WDM_PSD
git clone git@github.com:nz-gravity/tvpsd_lisa_demo.git
bash tvpsd_lisa_demo/ozstar/setup_env.sh
exit

# from your LOCAL machine:
bash ozstar/sync_data.sh

# back on ozstar:
cd /fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo
sbatch ozstar/run_m1.sbatch
sbatch ozstar/run_m0_x2.sbatch
```

CPU-only (`milan` partition), matching everything validated locally this
session -- no `jax[cuda]` extra, no GPU-specific code path has been tried.

## What's here

- `setup_env.sh` -- clones/pulls both repos, symlinks the package repo to the
  name `run_aet_diagonal_pilot.py` expects (`wdm_psd`, sibling to this repo --
  see the comment in the script), creates/updates the venv with the `[lisa]`
  extra.
- `sync_data.sh` -- run **locally**, not on the cluster. Copies the ~1.6 GB
  `combined_esa_xyz.h5` + `noise2a/` + `gb1/` that `.gitignore` deliberately
  excludes from the repo.
- `run_m1.sbatch` -- M1 (rung 3), joint A/E/T component separation.
- `run_m0_x2.sbatch` -- M0 (rungs 1-2), X2 only.

## What's NOT here yet

`esa_m0_study.py` has no `--channel` argument -- it reads `tdi/total` index 0
(X2) directly (`esa_m0_study.py:635`) and its reference PSD is X2-specific
(`analytic_x2_noise_components_psd`). There is no A/E/T job for M0. Adding
`--channel {X2,A,E,T}` (default `X2`, so the frozen X2 path stays untouched)
is the next piece needed before the full three-rung, three-channel ladder can
run as a single batch. Until then, the ladder is:

| rung | channel | job |
|---|---|---|
| 1-2 | X2 | `run_m0_x2.sbatch` |
| 3 | A, E, T jointly | `run_m1.sbatch` |

## Sizing

Both jobs request 6 cores / 16 GB, matching the ~500% CPU observed locally.
M0's own production numbers: 1185s wall (854s NUTS), 4 chains x 300+300.
M1 full-band 500/500: ~2850s locally after the likelihood preconditioner fix.
Both request generous ceilings (1h / 2h) rather than tight ones for a first
run; tighten once you have real OzSTAR timings.
