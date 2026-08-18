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
sbatch ozstar/run_component.sbatch
sbatch ozstar/run_surface_x2.sbatch
sbatch ozstar/run_surface_x2_gap7.sbatch
```

CPU-only (`milan` partition), matching everything validated locally this
session -- no `jax[cuda]` extra, no GPU-specific code path has been tried.

## What's here

- `run_surface.sbatch` -- parameterized H_agn/H_orb for the paper's rungs 1 and 2 on any
  channel and either mode:
  `sbatch ozstar/run_surface.sbatch <channel> <orb|agn> [continuous|gapped] [centre]`. All
  scientific settings live here once, so the rungs differ only in what they
  assume and the modes only in whether a gap is injected.
- `setup_env.sh` -- clones/pulls both repositories and creates/updates the venv
  with the shared package's `[lisa]` extra.
- `sync_data.sh` -- run **locally**, not on the cluster. Copies the ~1.6 GB
  `combined_esa_xyz.h5` + `noise2a/` + `gb1/` that `.gitignore` deliberately
  excludes from the repo.
- `run_component.sbatch` -- H_para (rung 3), joint A/E/T component separation.
- `preflight_surface.py` -- verifies data, import provenance, the cellwise
  reference-scaled coarse likelihood, and 16-node WDM response-projection
  convergence on actual production-grid X2 nulls before a surface-study job starts.
- `run_surface_x2.sbatch` -- projected-reference continuous X2 H_orb anchor.
- `run_surface_x2_gap7.sbatch` -- projected-reference seven-day-gap X2 H_orb
  anchor with a one-WDM-pixel boundary buffer.

## Surface-study settings

The surface-study scripts deliberately spell out all publication settings; they do not
rely on runner defaults. Both use the full realized WDM band, 36 frequency
knots, the five-time-knot `stationary_plus_interaction` residual around the
WDM-projected analytic response reference, four 500+500 chains, and the locked
split/null evaluation/binning settings. `PYTHONPATH` is pinned to the sibling `wdm_psd` checkout
so an older package installed elsewhere cannot silently run.

All frequencies in retained training rows enter surface-study inference and the adaptive
bin pilot, including response-null cells. The response-null mask is used only
for notched evaluation summaries; each run also records all-cell held-out
whitening. The preflight enforces this separation.

Every publication run also transforms the archived noise and Galactic time
series independently and checks them against their projected component
expectations. Held-out TV-versus-stationary comparison integrates over
posterior draws and reports a paired block-bootstrap interval. The surface
archive is accompanied by a chain-preserving archive with sampler fields and
spline reconstruction bases.

`run_surface_study.py` accepts X2/Y2/Z2/A/E/T. The X2 anchors remain as the
documented methods receipt; the paper's ladder runs on A and E through
`run_surface.sbatch`.

The current ladder is:

| rung | channel | job |
|---|---|---|
| 1-2 | X2 continuous | `run_surface_x2.sbatch` |
| 1-2 | X2 seven-day gap | `run_surface_x2_gap7.sbatch` |
| 1-2 | any channel, either mode | `run_surface.sbatch <channel> <orb\|agn> [continuous\|gapped]` |
| 3 | A, E, T jointly, either mode | `run_component.sbatch [continuous\|gapped] [centre]` |

## The paper run

Ten jobs, all submittable together; allow the requested two-hour surface-study ceilings
until the new posterior-predictive post-processing is timed on OzSTAR.

```bash
# no-gap set
sbatch ozstar/run_surface_x2.sbatch          # corrected X2 anchor (methods receipt)
sbatch ozstar/run_surface.sbatch A orb       # H_orb (rung 2)
sbatch ozstar/run_surface.sbatch E orb
sbatch ozstar/run_surface.sbatch A agn       # H_agn (rung 1)
sbatch ozstar/run_surface.sbatch E agn
sbatch ozstar/run_component.sbatch             # rung 3

# gapped set (one seven-day gap at mid-year, one-pixel edge buffer)
sbatch ozstar/run_surface.sbatch A orb gapped
sbatch ozstar/run_surface.sbatch E orb gapped
sbatch ozstar/run_surface.sbatch A agn gapped
sbatch ozstar/run_surface.sbatch E agn gapped
sbatch ozstar/run_component.sbatch gapped
```

### Realistic duty cycle

`gapped` is a single seven-day outage: a stress test, not LISA's expected duty
cycle. `duty` injects the scheduled 3.5 h repointing every 14 days plus Poisson
unscheduled outages -- about 60 gaps losing ~5% of the record in duration, but
**~13% of WDM rows**, because every gap pays a one-hour taper and a one-pixel
buffer at each edge whatever its length.

```bash
sbatch ozstar/run_surface.sbatch A agn duty     # H_agn
sbatch ozstar/run_surface.sbatch A orb duty      # H_orb
sbatch ozstar/run_component.sbatch duty            # H_para
```

The trailing argument is the gap-schedule seed for `duty` (the unscheduled
outages are random) and the gap centre for `gapped`; a further argument sets
the edge buffer in WDM pixels. Under one gap the buffer is nearly free
(2.4% of rows at 0 pixels, 2.6% at 2); under `duty` it dominates the loss:

| buffer [px] | rows lost, one 7-day gap | rows lost, duty cycle |
|---|---|---|
| 0 | 2.4% | 7.1% |
| 1 | 2.5% | 12.8% |
| 2 | 2.6% | 18.3% |

The one-pixel buffer was selected on a single-gap configuration where it cost
nothing. It is worth re-deciding under `duty`, where it more than doubles the
data removed:

```bash
sbatch ozstar/run_surface.sbatch A orb duty 1 0   # same schedule, no buffer
```

The gap centre defaults to mid-year. That is a quiet stretch of the Galactic
modulation (1.7x the annual minimum, log-slope 0.33/yr), so it is the easy
case. The modulation peaks at t = 0.81 yr at 3.0x the minimum; placing the gap
there is the harder test and the one a referee will ask about:

```bash
sbatch ozstar/run_component.sbatch gapped 0.8
sbatch ozstar/run_surface.sbatch A orb gapped 0.8   # matched surface comparison
```

Output names carry the centre (`..._gapped_c0.8_<jobid>`), so runs at different
placements cannot overwrite each other.

Rungs 1-2 use A and E only: T appears in the paper solely inside H_para, where the
null masking and the sub-3 mHz cut already have a stated treatment. Add
`sbatch ozstar/run_surface.sbatch T {orb,agn}` if the ladder table needs the row.

All gapped jobs share one geometry at a given centre -- a single seven-day gap,
a one-hour cosine taper, and a one-WDM-pixel edge buffer -- so the surface study and H_para run at
the same centre describe the same outage, and the default centre matches the
frozen `run_surface_x2_gap7` anchor. H_para reuses
the surface study's `gate_gaps`/`good_time_bins` rather than reimplementing them.

Deliberately not run: the frozen sensitivity checks and the multi-realization
coverage study. Without the last one, reported coverage stays a
single-realization descriptive statistic, as
`PUBLICATION_PROTOCOL.md` already states.

H_para now reports held-out scores on its own bins (`heldout_binned_diagnostics`),
in the surface study's three bands, with the gain taken against the analytic OMS+TM reference,
so the rung-3 row of the ladder table measures the same thing as rungs 1-2.

## Sizing

Surface-study jobs request 6 cores / 32 GB. The larger memory ceiling covers the
chain-preserving archive, component transforms, and posterior-predictive
post-processing in addition to the sampler. The superseded surface run took 1185s
wall (854s NUTS), 4 chains x 300+300; the publication scripts now use 500+500.
H_para full-band 500/500: ~2850s locally after the likelihood preconditioner fix.
Both request generous ceilings (1h / 2h) rather than tight ones for a first
run; tighten once you have real OzSTAR timings.

## Before submission

Pull both repositories and resync data if needed:

```bash
bash ozstar/setup_env.sh
python ozstar/preflight_surface.py --base /fred/oz200/avajpeyi/projects/WDM_PSD
```

The batch scripts repeat the preflight automatically. A job stops before the
full fit if it imports the wrong checkout, lacks an input dataset, does not
exercise the corrected coarse-reference branch, or fails the production-grid
WDM projection convergence check.
