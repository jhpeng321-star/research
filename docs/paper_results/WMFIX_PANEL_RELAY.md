# gpuR panel-eval staging — MAIN ADDENDUM 2026-10-31 11:52 (S-WMFIX-PILOT)

gpuR was verified actually FREE before use (1 MiB / 24,564 MiB, zero compute
apps at 11:53; the 07:20 candB WMFAST2 squatter had exited). Pilot PIDs
147313/147315 were killed 11:57:21 after PANEL_step0_r1 == r0 exactly
(noise 0 by determinism); continuation PID 160005 (gpuA, fresh 21,570 s cap).

## Staging (all sha256-verified gpuA -> local -> gpuR, byte-identical)
- TRAIN_PREPARED.json 7bbd41f3... (was absent on gpuR; relayed)
- tools/wmfixmerge/runtime.py 54900efd... (gpuR copy was stale 8b3e6e9b from
  pre-pilot staging; replaced by the gpuA-registered version)
- tools/wmfixmerge/pilot.py 07da9e20... / panel_eval.py b7360320... /
  pilot_continue.py 1d8be175... (new; panel_eval reuses pilot.panel so the
  measurement is the identical code path)
- relay_steps/step_000000.pt 771b128c... (the pilot's exported start)
- implementation_hashes() on gpuR now equals gpuA for all 30 bound files
  (runtime.py included); e3data corpus present.

## Watcher (own PIDs)
`panel_watcher.sh` PID 490799 (bash), evaluating step0_gpuR from 12:10:21
gpuR clock via `panel_eval.py` PID 490803, --memory-fraction 0.85 (matches the
successful shard relaunch; 24 GB free). Sequence: PANEL_step0_gpuR.json
(same-hardware baseline guard, reported in PILOT.json, not in the decision
rule), then PANEL_step10/20/30.json as relay_steps/step_0000{10,20,30}.pt
appear. Out: panels/; log panel_eval_gpuR.log; WATCHER_PID.txt. ~28.5 min per
panel -> all four done ~14:10 local if relays are prompt; panels never touch
gpuA (training unblocked; steps measured ~20 s each, 5 updates by 12:08).

Decision afterwards (in pilot_continue on gpuA, unchanged pre-registered
decide()): baseline = identical r0/r1 mean, SD 0; PASS requires strict
improvement at BOTH horizons + finite grads + frozen-bitwise. gpuR step-0
baseline cross-checks hardware bias; disagreement => re-run the affected
panel on gpuA before declaring PASS (recorded, threshold-free).
