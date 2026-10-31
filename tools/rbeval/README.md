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
`python -m tools.rbeval.summarize_full_fleet --panel <collected directory>
--targets <shared targets> --output <new score.json>` checks complete FIT32,
collection hashes and conservation before reporting per-world and marginal clocks.
