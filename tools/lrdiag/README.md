# S-LRDIAG-1 reproduction

Analysis only. No simulator, training, or GPU inference. The shared registration
was appended before analysis. `CONTRACT.md` gives estimands and caveats.

1. `python tools/lrdiag/stage.py cpuP` (and cpuQ) copies only safe runtime code,
   WM2, and the three-arm E3 **validation** episodes. The source hosts are
   read-only. `STAGING.json` binds every transferred file by SHA256. The two
   fixed manifests retain exactly 36 validation worlds / 108 episodes.
2. `python tools/lrdiag/stage.py cpuP --pairs` (and cpuQ) stages the exact sorted
   48-group pairs300 sample, seed 20261026, from S-WMCOMBINED's scoring view.
   `PAIRS_STAGING.json` binds the sample and source bytes. It is non-verdict.
3. Upload the Python files in this directory into ~/lrdiag.
   Run `python launch.py --parity 0 --workers 24` on cpuP and parity 1 on cpuQ.
   cpuQ's Python is ~/miniconda3/bin/python. Environment and low priority are
   set by the coordinator. It writes OWNER/PROGRESS/DONE receipts. Completed
   units are resumable; failures remain visible. A YIELD file, or observed SUMO
   collection, pauses inference without signalling any other process.
   Each host first checks the existing `WMS_SPEED_COMPACT_SLOTS` inference path
   against dense attention on identical 10 s draws. Final runs consistently use
   that supported gather/scatter path. Original dense full-horizon samples remain
   in `dense_reference/`; they are numerical references, not mixed into final K8.
4. `python continue.py --parity 0` / parity 1 wait for E3 completion, then run
   the recorded qualification pulses (cpuP only) and the pairs300 K8 ablations.
   They do not overlap 24-worker phases. Pulse sources are the existing
   ~/c4/runs/pulse/{jobs,src/all} on cpuP, 969824–969835, 30 s tapes.
   Before pairs300 jobs, the same 24-worker pool completes baseline full-road
   class-output upgrades. Same WM2 and K8 seeds; old scalar J and all-class
   speed/count bins must reproduce. Original JSONs and hashes are retained;
   original trace NPZs are not rewritten. This supplies a full-traffic timeline
   alongside the gate's fixed-t0-cohort timeline, without conflating cohort exits
   with perturbation damping.
5. Gather only task-owned results/traces and receipts from both CPUs into a
   task-owned root; the parity halves have disjoint result filenames. Run:
   `python summarize.py --root ROOT --out EVIDENCE`, then `pairs_summary.py`
   and `pulse_summary.py` with the same flags. `figures.py EVIDENCE` renders the
   CSV figures. `verify.py --evidence EVIDENCE` checks complete E3 K8 coverage,
   change-point invariants and regional accounting against the gate objective.
   `gather.py` merges CPUQ's disjoint parity into CPUP with SHA checks and atomic
   writes, including explicitly hash-bound metadata upgrades. `analyze_owned.py`
   runs reducers while pausing/resuming one verified owned model job, keeping
   at most 24 active CPU workers and yielding to collection.

All reported verdict statistics come from E3 validation. K8 means are formed
before statistics; CIs resample worlds. E3 arm-specific pooled slopes use OLS
with intercept. A single within-world arm contrast supplies a ratio, not a
regression; the two-contrast through-origin gain is saved separately. Paired
delay is the registered t0-HDV speed-deficit proxy, not full-fleet source wait.
Timeline regions follow each trajectory dynamically within that fixed cohort.

The BIC detector compares a downward step plus shared linear trend against
constant and linear nulls, includes change-point search in the parameter count,
and uses the 60 s pre-t0 baseline. It detects discrete speed-loss events, can miss
gradual/multiple breakdowns, and reports pre-existing congestion separately.
Pulse gains use initial following order; lane residence and exact-zero amplitude
censoring are retained. They are finite-time nonlinear gains, not asymptotic
string-stability eigenvalues. Qualification worlds are outside E3 validation.
Breakdown is recomputed from **full recorded traffic**, not the cohort-only
training loader. Its 3-bin post-change requirement right-censors the last 30 s.
`observables.py` additionally quantifies how much real CAV traffic arrives after
t0; the production paired-replay entry timeline assigns those synthetic CAVs
native commands. This is a structural confound, not a measured causal attribution.
