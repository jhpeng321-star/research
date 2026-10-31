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
