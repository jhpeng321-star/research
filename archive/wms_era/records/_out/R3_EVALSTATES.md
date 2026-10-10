# R3 eval states, H=60 s (wms/r3-evalstates3)

CPU only, no SUMO/training. Output: `_out/wallc1_states_h60/` (1 file set, 0.51 GB, stays on E:).
Command: `_out/run_h60.sh` (lpwo.materialize_wall --history-s 60; read-only --source-root packs
pack_v3 (D:), pack_dspeed, pack_damping + identities; inputs copied to `_out/inputs`; guard 0 refusals,
13 excluded worlds). sha256: `_out/wallc1_states_h60/SHA256SUMS.json`.

| set | states | rejected | bytes |
|---|---:|---:|---:|
| wall_same | 67 | 68 | 110,045,917 |
| wall_pilot | 53 | 77 | 80,942,961 |
| seen_controls | 217 | 62 | 318,952,490 |

Selection, keys, t0 and futures match v2 exactly (verify_state_futures.py: 0 mismatches of
key/world/arm/pack/t0/counter/initial_state/recorded; history tails of all t0 vehicles identical to v2).

## Key finding: 60 s history is NOT available for any state
All 337 states have available_span_s = 5.0 (run_start_limited=True; history_steps=11), i.e. 0 of 337 have 60 s.
Cause: pack_v3 runs (also dspeed/damping selection) record only from t_origin = t0_s of the window
(meta.json t_origin=300 for w8660m2700p10t300; n_ticks=1201 covers 300-900 s); the state t0 is origin + 4.5 s
(first non-empty tick + shift 9). Pre-origin ticks do not exist in the packs/shards, raw runs were deleted
(MIG-04). Earlier attempts (ws-r3-evalstates, -evalstates2) were verified to have the same 5 s span.
Options for a real 60 s history: (a) later t0 (>= origin+60 s) inside the stored run -- new states, futures not
identical, needs a new registration; (b) accept front-padded 5 s history (state files carry masks/available_span_s).
Code: fixed long-history validation in lpwo/materialize_wall.py, lpwo/verify_state_futures.py (new), tests (70 pass).

---
# Part 2: wall_h60 (real 60 s history, NEW set; owner option (a))
Output `_out/wallc1_states_h60_real/` (0.70 GB, E:), REGISTRATION.json there (rule, counts, rejection
reasons, sha256 of inputs/outputs/code, v2 overlap). The v2-identical `wallc1_states_h60/` stays the C1 comparison set;
the two are separate and never merged. `wallc1_states_h60_real_dev1/` is an earlier dev run of the same rule
(seen_controls crashed on a guard path; superseded, left in place, not deleted).

**Fixed outcome-blind rule.** t0 tick = manifest item first tick + 120 native ticks (sim clock origin + 60.5 s;
row tick_start is the empty origin row, so all 121 history rows are real), 240-tick future inside the stored run.
Roster = every vehicle present at t0, named per tick via the pack shard slot_index/slot_vid (cross-checked vs
pack veh_mask and the identity sidecar at the manifest tick; slots are reused, the sidecar alone cannot name later
entrants). Wall arms: members must be CAVs in the roster; an owned wall must be in assembly/hold at t0
(request_s <= t0 < owner_end_s) from timeline metadata only. Same requested windows/worlds/guard as v2.
Base/native_base arms excluded (origin-0 featurised). Code: lpwo/real_t0.py, `--t0-after-origin-s 60`.

| set | requested | states | rejected | paired | states w/ 60 s span | roster vehicles with full real 60 s |
|---|---:|---:|---:|---:|---:|---:|
| wall_same | 135 | 120 | 15 | 8 | 120 | 7110 / 12802 (55.5%) |
| wall_pilot | 130 | 116 | 14 | 0 | 116 | 6847 / 12331 (55.5%) |
| seen_controls | 342 | 38 | 304 | 0 | 38 | 2240 / 4107 (54.5%) |

All 274 wall + 38 control states carry full 60 s history windows (available_span_s=60, 121 rows, not run-start
limited). The other ~45% of t0 vehicles entered the road within the last 60 s (masked rows before entry; per-vehicle
real span in history.roster_span_s, median 60 s, p10 13 s); no state has every roster vehicle at 60 s.
Rejections: wall_same 13 member missing from roster + 1 handed_back + 1 not-yet-requested; wall_pilot 13 + 1;
seen_controls 152 base arms (rule) + 152 dspeed runs (pack_dspeed shards were stripped to meta, no slot_index, so
vehicles that entered after the manifest tick cannot be named -> excluded, not approximated).
Pairing: walls diverge from native within 60 s, so only 8 wall_same states keep identical native physics at t0.
Overlap with v2 keys: wall_same 61/67 common (+59 new), wall_pilot 53/53 (+63), seen_controls 31/217 (v2 controls were mostly pack_dspeed).
Futures are NOT identical to v2 (different t0). Verification: lpwo/verify_real_t0.py, 0 issues (history rows/span,
t0 rule, 241 frames, t0 row equals initial state, finite). Tests: 74 pass (4 new real_t0 tests).
