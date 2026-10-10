# SMOKE_V22_R1_DIAGNOSIS.md — why declared smoke run 1 failed (mechanics only)

2026-10-01. Source: the failed run's `wall_events.jsonl`, `branch_record.json`
and `RUN_RECEIPT.json` (`wall_summary` only) at
`E:/research-dense/runs/campaign_walls_v22_smoke/w8694m2600p40t240__w-lo-early/`,
the verdict `E:/research-dense/output/campaign_walls_v1/SMOKE_V22_VERDICT.json`,
and the declared J4 geometry (`j4_merge_analysis.json`, TRAIN row).
No `evaluate_summary`/`suffix_summary` was opened; the run is a mechanics
fixture and is NEVER read as an outcome. Nothing here tunes the law.

## 1. Run identity and integrity

| field | value |
| --- | --- |
| run (world / arm) | `w8694m2600p40t240__w-lo-early` (t0 = 240 s, horizon [240, 540] = 300 sim-s) |
| receipt | `exit_status complete`, `load_state_calls 1`, `horizon_mode smoke_cap`, 600/600 physics rows, teleports 0, speed-mode violations 0 |
| law version | `wall-v2.2-merge-release` (request/assembly/abort rows all carry it) |
| outcome | `aborted_in_hold`, `aborted_at_s 328.0`, `abort_reason non_member_rear_end_risk:ramp.29` |
| members | `mainline.132` (lane 0) + `mainline.131` (lane 1), span 3.577 m |

## 2. Timeline (from `wall_events.jsonl`)

| t (s) | event |
| ---: | --- |
| 240.0 | `wall_request`: lead s_road 1138.899718 m, lead distance-to-merge 1505.300282 m, D_min 1110.6998 m (v_wall 15.86714 m/s = 0.6 x mean 26.445233), hold draw 240 s — feasible, margin +394.6 m |
| 240.0 | `assembly_start` (assembly-phase member-lane rows 240–243: both members on entry edge 1191554636, no lane change) |
| 244.0 | `assembled` (4 s assembly) |
| 245–327 | HOLD: per-tick member-lane rows, both members on 1191554636, zero lane changes |
| 328.0 | `safety_abort` `non_member_rear_end_risk:ramp.29`, handback `native` |
| 328.0 | `hold_cap`: requested 240.0, effective 84.0, `capped_at_merge false`, `a_priori_effective_hold_s 5.798299` (see §4), `lead_s_road_at_release_m 2552.197583` |
| 328.0 | `wall_handback` (`how safety_abort`, phase hold, members back to native) |

## 3. Where the wall stopped, and what ramp.29 was

- Lead `s_road` at the last hold tick: **2552.1976 m = 92.0024 m BEFORE the
  merge start (2644.2)** — at v_wall 15.867 m/s, ≈ 5.80 s of travel short of
  the release point. The wall never touched the merge zone (all member-lane
  rows stay on the entry edge).
- `ramp.29` is a NON-MEMBER vehicle (id prefix `ramp.` = inserted on the
  merge ramp) that approached the wall FROM BEHIND. The frozen v2.2 defence
  (`wall_group.py` `_safety_abort`) fires `non_member_rear_end_risk:<id>`
  when any non-member's `s_road` is inside `[rear − 10 m, rear]`
  (`REAR_GAP_M`) of the rearmost member AND its speed exceeds
  `v_wall + 2.0 m/s` (`REAR_SPEED_EXCESS_MPS`). The wall crawled at
  15.87 m/s in a 0.4-penetration world whose ambient traffic runs ≈ 26 m/s;
  ramp.29 closed inside the 10 m rear gap above 17.87 m/s → the defence
  fired and handed the members back to native. This is the safety design
  operating on a legal traffic state — no collision, no teleport, no
  member escaped control (`member_left_network` count: 0).

## 4. The two independent defects this exposes

1. **Fixture rule (the reason the smoke "failed").** MC3 of `SMOKE_V22.json`
   says "a v2.2 smoke run with ANY safety_abort FAILS", and the fixture is
   stop-on-fail: run 1 aborted at t=328 → verdict `passed: false` → **runs 2
   and 3 were never launched** (the runs root contains only this one run).
   The abort was a registered v2.2 safety abort in a tolerated reason class,
   not a law or mechanics defect — an abort-prohibitive smoke rule cannot be
   satisfied on worlds where ambient traffic naturally rear-ends a slow
   wall.
2. **Logging bug (behaviour-neutral fix shipped with the re-smoke).** The
   `hold_cap` row's `a_priori_effective_hold_s = 5.798299` was computed from
   the RELEASE-tick lead position (`(2644.2 − 2552.1976)/15.86714`), i.e. it
   just re-measured "how far past the last tick the release point is" —
   meaningless for an abort (the hold had already run 84 s). The a-priori
   cap must be computed AT ASSEMBLY: with the declared J4 assembly position
   (lead 1219.4 m, `j4_merge_analysis.json` TRAIN row) it is
   `(2644.2 − 1219.4)/15.86714 ≈ 89.8 s`, i.e. the cap would have fired at
   ≈ t = 333.8 — the abort pre-empted the merge-start cap by ≈ 5.8 s.
   Fixed by `patches/wall_driver_v22_logging.diff` (assembly-position cap +
   new `hold_cap.lead_s_road_at_assembly_m` evidence field).

## 5. Consequence (draft, pending owner decision)

`SMOKE_V22_R2.json/.md` re-declares the same 3 x 300 sim-s budget on a NEW
runs root (`runs/campaign_walls_v22_smoke_r2`, write-once) with the
WM-recommended rule: a registered safety abort (any frozen v2.2 reason
except `member_left_network`) is RECORDED, not fatal; FAIL only on
`member_left_network`, a missing `hold_cap` row or receipt, a law version
other than v2.2, or any release/abort past the merge start; all 3 runs run
(no stop-on-fail) and >= 1 observed `merge_release` with
`capped_at_merge` logged is required.
