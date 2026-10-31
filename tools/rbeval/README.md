# S-RB-MERGE FIT diagnostics

Run with existing Python + numpy/torch; one CPU thread. No SUMO, training,
driver-truth files, assignment files, or directories outside the exact FIT range.

```
python tools/rbeval/validate_merge.py --root /home/ubuntu/wmdata20261031/views --seeds 969400 969401 --horizon 300 --output receipts/rb_merge_fit.json
```

Reads `world_<FIT seed>.json` then only episode metadata and observable
`time_s.npy`, `veh_kin.npy`, `veh_mask.npy`, `vid.npy`. Refuses prohibited paths,
including symlink destinations, and any seed outside 969400–969431.

- **Recorded-state mechanics:** two-sided merge acceptance after causal
  constant-acceleration motion, versus next recorded lane. Reports confusion
  counts and first accepted times. Entrance admission recall uses the source-IDM
  check at observed births; first visibility only (junction reappearance excluded).
- **Autonomous cohort:** all t0 vehicles run native population physics with
  SUMO merge, physics MOBIL mainline; exits are autonomous. No future positions,
  lanes, presences or service are fed back. Reports merge time/position errors,
  censored/missed/extra event slots, normalized ramp speed-loss vehicle-time and
  per-vehicle delay quantiles. Omits entrants; does not certify fleet/source delay.
- **Full population:** optional `--prediction-dir DIR` consumes integrator output
  `<seed>_<arm>.npz`, keys `s,v,lane,mask` [time,slot], `vid` [time,slot] (or fixed
  [slot]), initial identities exactly matching the view, `time_s` exact recorded
  clock including t0. Slots may recycle/reorder; scoring aligns persistent IDs.
  This is the contract for the overnight fully integrated WM-PHYS candidate.

`--stride 1` is the complete one-step diagnostic; larger strides are explicitly
subsampled and MUST NOT be interpreted as complete insertion/event counts.
`--inspect` reports only metadata keys and observable array shapes.

The historical 562.4 real / 1265.4 WM2 ramp delay was a different 48-world
300s fleet contract. No cohort-only score is presented as a matched improvement.

## Committed diagnostic receipts

`docs/rb_merge/fit_native32_post_helper_summary.json` aggregates all 32 native
FIT worlds for source version 2977be6d. Its runtime source hashes were checked
against this checkout. Both task-owned cpuQ jobs have exited; do not relaunch
the retained dispatch scripts to reproduce already completed receipts.

Re-aggregate without running another rollout:
```
python -m tools.rbeval.summarize_merge docs/rb_merge/post_helper_fit_300.json --version 2977be6d-source-corrected-population-MOBIL-mainline --dispatch docs/rb_merge/post_helper_fit_dispatch.json --output docs/rb_merge/fit_native32_post_helper_summary.json
```

Earlier source-version receipts and the adverse first FIT32 screen are retained
separately. Different versions are not pooled. The ramp-lane speed-loss
diagnostic uses reference speed 22.22m/s and excludes a ramp vehicle after its
merge; use full ramp-origin/group clocks for the owner's fleet-delay target.
## Entrant-inclusive LC2013 continuation

`full_fleet.py --runtime <immutable integrated root> --root <observable views>
--targets <shared target CSV directory> --output <new result directory>
--seeds 969400 ... 969431 --horizon 300` runs population-default physics with
LC2013 mainline, SUMO merge/insertion and the existing native CAV executor.
Uses identity-hashed observable `start.pt` pre-t0 insertions and entrance backlog
to infer demand (the existing WMCALIB source contract), preserving `conserve_queue` /
`SourceLedger`. The recorded per-world ramp-entrance-group clock is compared with
ramp-origin road delay **plus source waiting**, including travel after merging.
Cold LC memory and the actual CF selector are explicit in every receipt. No future
observation supplies a prediction or driver setting. Exact future identity timing
is meaningful only for the t0 cohort; synthetic entrants are scored distributionally.

`remote_full_fleet.py status|stage|launch|poll|collect --version v4` owns one
CPU-only dispatch on cpuQ: INT `b0f66943` plus MERGE `0090c168` kernel/two
lookahead hooks, `--cf-model sumo_idm --warm-lc`. Stage/launch/result paths refuse
replacement. Earlier v3 (INT `ff8a637f`, legacy CF, cold LC) is retained separately.
Default version is now `v5`: same INT base plus MERGE `3df73a27` network/insertion
rules and `--trace-merge`, retaining the same arrival stream and population.
Use an explicit `--version v3`/`v4` to inspect or collect those completed runs.
Current default `v6` adds source-EPS5870b556 and the opt-in physics-native CAV
callback d7be510a (`--native-model sumo`). Prior v5 retains its frozen head path.
The validator's direct CLI default stays `--native-model head`; selecting sumo
requires `--cf-model sumo_idm`. Every row reports its actual native model.
`v7` now stages INT4015b641 (latest finite-route LC2013), source minNextSpeed,
and the reused strict frozen motion encoder b192e0c8, with H600 /512slots.
`--encoder-checkpoint <registered wm_s.pt>` infers initial HDV settings once,
keeps CAV defaults, and couples zero-history-centre entrant settings to exact
admission ordinals. The learned motion/lane/merge modules remain bypassed.
The600s run captures120/300/600s clocks in one pass; it does not replay future
observations or refit any model. Summaries accept one uniform horizon per panel.
`--entry-phase after` opts into SUMO's move/LC/then-depart order through the new
`step_after_movement` helper. Boundary scoring uses its explicit source_before
cohort, birth and exit masks. Newborns stay at basePos/departSpeed on birth tick.
No-arrival dynamics are bit-exact to the original physics step. This operator
experiment has a separate source identity from the completed `v7fast` H600 run.
Current runner default is `v8end600 --entry-phase after`; earlier version
directories remain explicit and immutable. `verify_fleet_prefix` checks the saved
H120 clocks/events/front traces against the600s extension without treating those
prefixes as independent evidence or claiming unavailable full-trajectory parity.
`python -m tools.rbeval.summarize_full_fleet --panel <collected directory>
--targets <shared targets> --output <new score.json>` checks complete FIT32,
collection hashes and conservation before reporting per-world and marginal clocks.
