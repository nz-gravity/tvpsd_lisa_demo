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
  channel and either mode:
  `sbatch ozstar/run_m0.sbatch <channel> <ref|free> [continuous|gapped]`. All
  scientific settings live here once, so the rungs differ only in what they
  assume and the modes only in whether a gap is injected.
- `setup_env.sh` -- clones/pulls both repositories and creates/updates the venv
  with the shared package's `[lisa]` extra.
- `sync_data.sh` -- run **locally**, not on the cluster. Copies the ~1.6 GB
  `combined_esa_xyz.h5` + `noise2a/` + `gb1/` that `.gitignore` deliberately
  excludes from the repo.
- `run_m1.sbatch` -- M1 (rung 3), joint A/E/T component separation.
- `preflight_m0.py` -- verifies data, import provenance, the cellwise
  reference-scaled coarse likelihood, and 16-node WDM response-projection
  convergence on actual production-grid X2 nulls before an M0 job starts.
- `run_m0_x2.sbatch` -- projected-reference continuous X2 M0 anchor.
- `run_m0_x2_gap7.sbatch` -- projected-reference seven-day-gap X2 M0
  anchor with a one-WDM-pixel boundary buffer.

## M0 settings

The M0 scripts deliberately spell out all publication settings; they do not
rely on runner defaults. Both use the full realized WDM band, 36 frequency
knots, the five-time-knot `stationary_plus_interaction` residual around the
WDM-projected analytic response reference, four 500+500 chains, and the locked
split/null evaluation/binning settings. `PYTHONPATH` is pinned to the sibling `wdm_psd` checkout
so an older package installed elsewhere cannot silently run.

All frequencies in retained training rows enter M0 inference and the adaptive
bin pilot, including response-null cells. The response-null mask is used only
for notched evaluation summaries; each run also records all-cell held-out
whitening. The preflight enforces this separation.

Every publication run also transforms the archived noise and Galactic time
series independently and checks them against their projected component
expectations. Held-out TV-versus-stationary comparison integrates over
posterior draws and reports a paired block-bootstrap interval. The surface
archive is accompanied by a chain-preserving archive with sampler fields and
spline reconstruction bases.

`esa_m0_study.py` accepts X2/Y2/Z2/A/E/T. The X2 anchors remain as the
documented methods receipt; the paper's ladder runs on A and E through
`run_m0.sbatch`.

The current ladder is:

| rung | channel | job |
|---|---|---|
| 1-2 | X2 continuous | `run_m0_x2.sbatch` |
| 1-2 | X2 seven-day gap | `run_m0_x2_gap7.sbatch` |
| 1-2 | any channel, either mode | `run_m0.sbatch <channel> <ref\|free> [continuous\|gapped]` |
| 3 | A, E, T jointly, either mode | `run_m1.sbatch [continuous\|gapped]` |

## The paper run

Ten jobs, all submittable together; allow the requested two-hour M0 ceilings
until the new posterior-predictive post-processing is timed on OzSTAR.

```bash
# no-gap set
sbatch ozstar/run_m0_x2.sbatch          # corrected X2 anchor (methods receipt)
sbatch ozstar/run_m0.sbatch A ref       # rung 2
sbatch ozstar/run_m0.sbatch E ref
sbatch ozstar/run_m0.sbatch A free      # rung 1
sbatch ozstar/run_m0.sbatch E free
sbatch ozstar/run_m1.sbatch             # rung 3

# gapped set (one seven-day gap at mid-year, one-pixel edge buffer)
sbatch ozstar/run_m0.sbatch A ref gapped
sbatch ozstar/run_m0.sbatch E ref gapped
sbatch ozstar/run_m0.sbatch A free gapped
sbatch ozstar/run_m0.sbatch E free gapped
sbatch ozstar/run_m1.sbatch gapped
```

Rungs 1-2 use A and E only: T appears in the paper solely inside M1, where the
null masking and the sub-3 mHz cut already have a stated treatment. Add
`sbatch ozstar/run_m0.sbatch T {ref,free}` if the ladder table needs the row.

All gapped jobs share one geometry -- a single seven-day gap at mid-year, a
one-hour cosine taper, and a one-WDM-pixel edge buffer -- so M0 and M1 describe
the same outage and both match the frozen `run_m0_x2_gap7` anchor. M1 reuses
M0's `gate_gaps`/`good_time_bins` rather than reimplementing them.

Deliberately not run: the frozen sensitivity checks and the multi-realization
coverage study. Without the last one, reported coverage stays a
single-realization descriptive statistic, as
`ESA_M0_PUBLICATION_PROTOCOL.md` already states.

M1 now reports held-out scores on its own bins (`heldout_binned_diagnostics`),
in M0's three bands, with the gain taken against the analytic OMS+TM reference,
so the rung-3 row of the ladder table measures the same thing as rungs 1-2.

## Sizing

M0 jobs request 6 cores / 32 GB. The larger memory ceiling covers the
chain-preserving archive, component transforms, and posterior-predictive
post-processing in addition to the sampler. The superseded M0 run took 1185s
wall (854s NUTS), 4 chains x 300+300; the publication scripts now use 500+500.
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
full fit if it imports the wrong checkout, lacks an input dataset, does not
exercise the corrected coarse-reference branch, or fails the production-grid
WDM projection convergence check.
