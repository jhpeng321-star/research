# WZ_WM_DATA_REPORT: wz3to2 world-model training data (WM-B, phase wm_wz_data, 2026-11-06)

## Summary
1. **Data collected.** World-model training tapes now exist for the wz3to2 lane-drop map: 100 TRAIN worlds × 4 arms = 400 episodes of 1,200 s each, in the original WM tape format (`c4_sumo_episode/1`, K = 512).
   - Failures: none in the main run. One canary job failed and was rerun as an explicit, recorded re-run.
   - Safety: 0 collisions, 0 teleports, 0 dropped slot-steps.
2. **Arms.**
   - `native`
   - `mappo_u0`: frozen speed-only MAPPO, policy_00169, sha d9042cc5
   - `coop_mappo_u0`: MAPPO base plus cooperation-move perturbations
   - `coop_native`: native base plus the same moves, with members holding their lane
   - Across the 198 gpuA coop episodes there were 294 abreast walls, 295 closing-lane pairs and 619 random groups; the 2 canary coop episodes add 12 more moves (1,220 in total). HDVs were never commanded.
3. **Seeds.** 989100-989199, one cell per seed. The cells are demand {2400, 2800, 3200, 3600} veh/h × CAV share {0.2, 0.4, 0.6}, with 8-9 worlds per cell.
   - VAL is registered as 12 worlds, one per cell. TRAIN is the other 88 (87 usable; the 88th, canary world 989103, is excluded from the packs, see item 5).
   - WM-D hold-outs and 989600+ are never used.
4. **Budget.** WM-B charged 482,700 sim-s of the 1,000,000 phase cap:
   - gpuA ledger: 475,200
   - cpuV canary ledger: 7,500
   - WM-D's set (b) is on its own ledger, at most 100,000.
5. **Packs.** The data was converted with WM-A's map-aware converter (c4/wz-wm-port 10a8f0b3) on gpuA:
   - TRAIN: 348 episodes from 87 worlds
   - VAL: 48 episodes from 12 worlds
   - `read_source_manifest` passes; road_digest cbbf1b48.
   - The 4 cpuV canary tapes are excluded because their meta_json has no `map` key.
6. **Training and qualification were not run by WM-B.**
   - WMA_READY.json still did not exist at 17:13.
   - MAIN reassigned the fine-tune to WM-E, with incremental shards handed over.
   - The gpuA GPU was idle at 17:13.
   - Fidelity qualification: not run, because there is no fine-tuned model yet.
7. **WMQUAL set (a) was not collected.** This is the WM-D request: native tapes for 989580-989599 × 4 cells, 0-1500 s.
   - My permission gate refused the launch, both on cpuV and on gpuA.
   - Only the canary episode 989580 D3200_p60 exists, on cpuV. Nothing was charged for the other 79 jobs.
   - This needs an owner/MAIN decision, or WM-D can run it.

## Authority, registration and hosts
- **Registration.** The REGISTRATION.md block "16:10 WM-B WMDATA + WMQUAL-A" was written before any SUMO run. It also covers the CLEAN1_STATUS WMDATA lines.
  - Owner phase wm_wz_data, cap 1,000,000 sim-s.
  - WM-B task cap 620,000.
- **Code.** Branch c4/wz-wm-data, worktree E:/agent-work/c4-wz-wm-data:
  - 23ac30e6: collector
  - 7e34515c: fix for the pair anchor
  - 947931ee / 52c5131e / 1e8ecd5b / 2d7acd0c: gpuA host, map meta in meta_json, round-robin order, shards
  - 8f691355: ops scripts
  - Modules: tools/wzwmdata/{registry, coop, collect, shards}.py
  - Tests: tools/wzwmdata/test_wzwmdata_cpu.py, 9 CPU tests; tests and ruff pass.
- **Runtime.** The frozen e3final CavSumoEnv and recorder, bound to wz3to2 by N1 `tools.coopmwm.map_runtime.install` (c4/nm-abstract e512b059). The scenario generator is the verbatim probe/NEWMAP2 one, with this phase's registry.
- **cpuV (canary only).** Root /home/ubuntu/wzwmdata20261106/run; PLAN a48818a8 plus PLAN_AMEND_fix1.
  - 6 ledger rows, 7,500 sim-s.
  - Canary world 989103 D3200_p60 with 4 arms, plus WMQUAL 989580 D3200_p60.
  - The coop_mappo canary failed (KeyError anchor_release_ds): its receipt is kept and charged. The fixed explicit re-run `_fix1` succeeded.
- **gpuA (main run).** Moved there by MAIN at 16:2x and 16:4x, because cpuV was reserved for clean4.
  - Ran with 20 workers at nice 19, CPU only, under own PID 411246 (now exited).
  - Used the qualified cpuV parity clone read-only: /root/mining20261106/cpuv_home (py 3.10.16, torch 2.12.1+cpu, SUMO 1.27.1, e3final SOURCE_MANIFEST b3c031ad).
  - Root /root/wzwmdata20261106/run (PLAN bd30b6a9, prior charged 7,500); ledger run/ledger/CHARGES.jsonl, 396 rows, 475,200 sim-s.
  - Data in /root/autodl-fs/wzwmdata20261106/data.
  - 396/396 jobs ok, 16-70 s per 1,200-s episode.

## Data volume per cell
Team delay J is in veh-s over the 300-1200 s control window (road speed deficit plus source wait); lower is better.
The ratio is the paired sum(J_arm) / sum(J_native) over the same worlds; below 1 means less delay than native. These outcomes are **descriptive only**: one episode per world, and the moves are random perturbations, so they are not efficacy evidence.

| cell | worlds | sim-s (4 arms) | native mean J | mappo_u0 ratio | coop_mappo_u0 ratio | coop_native ratio | coop moves | max present | npz MB |
|---|---|---|---|---|---|---|---|---|---|
| D2400_p20 | 8 | 38,400 | 26,017 | 0.747 | 0.815 | 1.016 | 111 | 176 | 89 |
| D2400_p40 | 8 | 38,400 | 31,316 | 0.633 | 0.761 | 0.935 | 111 | 237 | 96 |
| D2400_p60 | 8 | 38,400 | 25,632 | 0.584 | 0.651 | 0.930 | 109 | 179 | 91 |
| D2800_p20 | 8 | 38,400 | 50,433 | 0.756 | 0.790 | 1.042 | 102 | 254 | 116 |
| D2800_p40 | 8 | 38,400 | 54,801 | 0.862 | 0.786 | 1.024 | 97 | 253 | 121 |
| D2800_p60 | 9 | 43,200 | 62,990 | 0.629 | 0.584 | 0.900 | 111 | 286 | 140 |
| D3200_p20 | 8 | 38,400 | 113,370 | 0.926 | 0.894 | 0.980 | 97 | 352 | 154 |
| D3200_p40 | 9 | 43,200 | 110,557 | 0.803 | 0.769 | 0.880 | 102 | 356 | 180 |
| D3200_p60 | 9 (incl. canary) | 43,200 | 103,840 | 0.843 | 0.773 | 0.903 | 101 | 393 | 181 |
| D3600_p20 | 8 | 38,400 | 161,758 | 0.942 | 0.931 | 0.923 | 85 | 361 | 161 |
| D3600_p40 | 8 | 38,400 | 155,407 | 0.970 | 0.906 | 0.944 | 94 | 373 | 169 |
| D3600_p60 | 9 | 43,200 | 155,580 | 0.846 | 0.795 | 0.966 | 100 | 418 | 213 |
| **total** | **100** | **480,000** | | | | | **1,220** | 418 (< K 512) | ~1,712 |

Per arm: 100 episodes and 120,000 sim-s each. The compressed `episode.npz` totals are about 409 MB (native), 453 MB (mappo_u0), 448 MB (coop_mappo_u0) and 402 MB (coop_native). The raw directory on autodl-fs is about 24 GB including packs.
Evidence: tools/wzwmdata/ops/cell_table_20261106.json (gpuA sha256 ca388c0f).

## Shards and packs (WM-E handover; interface: E:/agent-work/_sol/WMB_DATA_SHARDS.md)
- **Shards.** gpuA /root/autodl-fs/wzwmdata20261106/shards/:
  - SHARD_01 (sha256 579b8518): 53 worlds × {native, mappo_u0}, published as soon as they finished.
  - SHARD_02 (sha256 a30e5018): final; the coop arms of all 100 worlds plus native/mappo_u0 of the other 47.
  - Both are world-grouped and use the registered split. VAL = per cell the seed with the smallest sha256('wzwmdata_val_v1:seed').
- **Packs.** gpuA /root/autodl-fs/wzwmdata20261106/packs/, schema `wms_c4_episode_source/1`, map wz3to2, K 512:

| source | episodes | worlds | arms | states | CAV rows | direct CAV steps | manifest sha256 |
|---|---|---|---|---|---|---|---|
| shard01_train | 88 | 44 | native 44, mappo_u0 44 | 211,288 | 12.1 M | 2.51 M | b7509d2e |
| shard02_train | 260 | 87 | coop_mappo_u0 87, coop_native 87, native 43, mappo_u0 43 | 624,260 | 36.0 M | 7.56 M | 5e149804 |
| shard01_val | 16 | 8 | native 8, mappo_u0 8 | 38,416 | 2.1 M | 0.42 M | 277e3b2a |
| shard02_val | 32 | 12 | coop 12+12, native 4, mappo_u0 4 | 76,832 | 4.2 M | 0.91 M | d64b210c |

- Every source was converted with `--val-frac 0`, so `split_role` is `train` inside the manifests. Use the `*_val` sources only for validation.
- Each episode stores the full 0-1200 s tape: native warmup 0-300 s, then control 300-1200 s. Windows are cut at load time.
- Raw job directories also hold `extras.npz` (entry queue and source wait per 0.5 s) and `RESULT.json` (coop move records).
- No driver-truth file was produced or read. The per-HDV vTypes exist only as SUMO input in `scenario/requests.rou.xml`, which was never parsed.

## Training and qualification status
- **Not launched by WM-B.**
  - WMA_READY.json did not exist at 17:13, so there is no registered fine-tune command.
  - MAIN's 16:2x instruction hands the fine-tune to WM-E, starting from the shards.
  - gpuA GPU: 0 % and 1 MiB at 17:13. autodl-fs has 177 GB free.
- **Fidelity qualification:** not run (no fine-tuned model). WM-D owns the qualification hold-outs: 989200-989216, 989350-989353, 989400-989402 and 989580-989599. WMQUAL set (a) is missing, see item 7.

## Issues and notes
- **Seed guard.** WM-A's first map_guard admitted only 989200-989599. It was fixed in 10a8f0b3: WM training seeds are now 989100-989199 plus 989200-989599 minus the WM-D reservations.
- **Converter contract.** Packing needs `map` in meta_json. All gpuA tapes carry it. The 4 cpuV canary tapes carry it only in manifest.json, so they are excluded.
- **Known WM gap (WM-A API section 4).** Lane changes 1 -> 2 into the closing lane are never proposed by lane_v5. These tapes contain SUMO's own 1 -> 2 changes, so WM-D or WM-E can measure the gap from them.
- **No reruns.** No job was retried silently; the single re-run is the registered amendment `fix1`.
- **Close-out.** experiment_history.json and OPEN_QUESTIONS were not updated; that is for the integrator.
