# OzSTAR

```
ssh ozstar
mkdir -p /fred/oz200/avajpeyi/projects/WDM_PSD
cd /fred/oz200/avajpeyi/projects/WDM_PSD
# Clone the study once so its setup script is available. The setup script then
# creates/updates both the study and shared-package checkouts and the venv.
git clone git@github.com:nz-gravity/tvpsd_lisa_demo.git
bash tvpsd_lisa_demo/ozstar/setup_env.sh
exit

# from your LOCAL machine:
bash ozstar/sync_data.sh

# back on ozstar:
cd /fred/oz200/avajpeyi/projects/WDM_PSD/tvpsd_lisa_demo
sbatch ozstar/run_m1.sbatch
sbatch ozstar/run_m0_x2.sbatch
sbatch ozstar/run_m0_x2_gap7.sbatch
```

CPU-only (`milan` partition), matching everything validated locally this
session -- no `jax[cuda]` extra, no GPU-specific code path has been tried.

## What's here

- `run_m0.sbatch` -- parameterized M0 for the paper's rungs 1 and 2 on any
  channel: `sbatch ozstar/run_m0.sbatch <channel> <ref|free>`. All scientific
  settings live here once, so the two rungs differ only in what they assume.
- `setup_env.sh` -- clones/pulls both repositories and creates/updates the venv
  with the shared package's `[lisa]` extra.
- `sync_data.sh` -- run **locally**, not on the cluster. Copies the ~1.6 GB
  `combined_esa_xyz.h5` + `noise2a/` + `gb1/` that `.gitignore` deliberately
  excludes from the repo.
- `run_m1.sbatch` -- M1 (rung 3), joint A/E/T component separation.
- `preflight_m0.py` -- verifies data, import provenance, and the corrected
  cellwise reference-scaled coarse likelihood before an M0 job starts.
- `run_m0_x2.sbatch` -- corrected response-informed continuous X2 M0 anchor.
- `run_m0_x2_gap7.sbatch` -- corrected response-informed seven-day-gap X2 M0
  anchor with a one-WDM-pixel boundary buffer.

## M0 settings

The M0 scripts deliberately spell out all publication settings; they do not
rely on runner defaults. Both use the full realized WDM band, 36 frequency
knots, the five-time-knot `stationary_plus_interaction` residual around the
analytic response reference, four 300+300 chains, and the locked split/null
evaluation/binning settings. `PYTHONPATH` is pinned to the sibling `wdm_psd` checkout
so an older package installed elsewhere cannot silently run.

All frequencies in retained training rows enter M0 inference and the adaptive
bin pilot, including response-null cells. The response-null mask is used only
for notched evaluation summaries; each run also records all-cell held-out
whitening. The preflight enforces this separation.

`esa_m0_study.py` now accepts X2/Y2/Z2/A/E/T, but the two production scripts in
this directory intentionally rerun the documented X2 anchors first. A/E/T M0
batch scripts should be added only after the corrected X2 comparison is
reviewed.

The current ladder is:

| rung | channel | job |
|---|---|---|
| 1-2 | X2 continuous | `run_m0_x2.sbatch` |
| 1-2 | X2 seven-day gap | `run_m0_x2_gap7.sbatch` |
| 1-2 | any channel | `run_m0.sbatch <channel> <ref\|free>` |
| 3 | A, E, T jointly | `run_m1.sbatch` |

## The no-gap paper run

Six jobs, all submittable together; under an hour of wall clock.

```bash
sbatch ozstar/run_m0_x2.sbatch        # corrected X2 anchor (methods receipt)
sbatch ozstar/run_m0.sbatch A ref     # rung 2
sbatch ozstar/run_m0.sbatch E ref
sbatch ozstar/run_m0.sbatch A free    # rung 1
sbatch ozstar/run_m0.sbatch E free
sbatch ozstar/run_m1.sbatch           # rung 3
```

Rungs 1-2 use A and E only: T appears in the paper solely inside M1, where the
null masking and the sub-3 mHz cut already have a stated treatment. Add
`sbatch ozstar/run_m0.sbatch T {ref,free}` if the ladder table needs the row.

Deliberately not run for this phase: all gapped jobs (phase 2; note M1 has no
gap support yet), the frozen sensitivity checks, and the multi-realization
coverage study. Without the last one, reported coverage stays a
single-realization descriptive statistic, as
`ESA_M0_PUBLICATION_PROTOCOL.md` already states.

Before the rung-3 row of any ladder table is meaningful, M1 needs a held-out
Whittle score matching M0's (`blind_whitening_diagnostics`); it currently
excludes the validation/test rows from the likelihood without scoring them.

## Sizing

Both jobs request 6 cores / 16 GB, matching the ~500% CPU observed locally.
The superseded M0 run took 1185s wall (854s NUTS), 4 chains x 300+300.
M1 full-band 500/500: ~2850s locally after the likelihood preconditioner fix.
Both request generous ceilings (1h / 2h) rather than tight ones for a first
run; tighten once you have real OzSTAR timings.

## Before submission

Pull both repositories and resync data if needed:

```bash
bash ozstar/setup_env.sh
python ozstar/preflight_m0.py --base /fred/oz200/avajpeyi/projects/WDM_PSD
```

The batch scripts repeat the preflight automatically. A job stops before the
full fit if it imports the wrong checkout, lacks an input dataset, or does not
exercise the corrected coarse-reference branch.
