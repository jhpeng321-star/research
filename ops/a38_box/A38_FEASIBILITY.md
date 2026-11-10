# A38 feasibility: second road (written 2026-11-10 evening; zero outcome read)

## Verdict
**NO-GO for A38 as registered** (WM search -> SUMO verification -> WM bank vs hand-written hold vs U0 on a second road)
on any candidate road before Nov 13. There are two independent blockers:
1. **The WM checkpoint 6da9d333 (WM-E ft_r1) is not reachable.** It is not on E: and not on Drive (details in section 4). That blocks
   every new WM search on any road, unless a box that holds /root/autodl-fs/wzwm/ft_r1/... comes back.
2. **The WM-S model frame is 3-lane only.** wz2to1 cannot be bound without model-side changes (section 1).

GO is possible only for a REDUCED, no-WM variant: wz2to1, U0 vs the hand-written hold port (HOLD_R2), with calibration
first. The build for it is done and launch-ready, and it takes about 2-3 h on 48 cores. It does not test the paper's WM
pipeline. Its hold is degenerate (single CAV). Prior native-base evidence on wz2to1 is unfavourable. Value: low.

## 1. wz2to1 (2->1 drop)
- **SUMO side works.** Smoke used 3,400 sim-s (Windows, native base, seed 996199). It is not evidence.
  - The registered generator plus SUMO 1.27.1 run cleanly: 0 collisions and 0 teleports.
  - The hold executor's wall trigger fired on lanes (0, 1), with the member in SUMO lane 1.
  - Plant code was ported on c4/a38-road2:
    - map_descriptor: 2-lane projection up_1 -> WM1, up_0 -> WM2. The wz3to2 digest is unchanged.
    - map_scenarios: wz2to1 profile.
    - road2: oracle wall lanes.
    - nm_finaleval: a38 settings.
    - runner: HOLD_R2 arms.
    - driver: ops/a38_road2.py.
- **U0 (MAPPO d9042cc5).** Not run in SUMO here, because the leverprobe plant is Linux-staged. The observation shapes are fixed
  by the 3-lane frame, so the inputs load, but WM lane 0 (RSU field lanes 0/1) is always empty. That is out of distribution, not a crash.
- **WM cannot bind.** wms.core.road.road_from_descriptor(wz2to1) raises `ValueError: min() iterable argument is empty`.
  Making it bind would need these changes in the c4/wz-wm-port model code:
  - `RoadGeometry.__post_init__` (n_lanes 3, through lanes (0,1));
  - `road_from_descriptor` extents/entries;
  - the entry process `entrances=(main=(0,1), ...)`, which Poisson-inserts into lane 0;
  - `physics/mobil.PHYSICAL_TARGET_OF_LANE=(1,0,1)`;
  - `model/lane_v5.HEAD_OF_LANE=(1,0,2)`. With these, lane-1 vehicles change into a nonexistent lane 0 (a phantom extra lane at
    the bottleneck).
  - Masking lane 0 is a model change, and the learned lane heads were never trained without lane 0. Forbidden in A38.
  - strategy_search / wzprop geometry and roles are also wz3to2-only.
- **Hand-written hold.** HANDWRITTEN_HOLD_ROAD2.md, sha256 4a77c461. It is a mechanical port of duo12_v20 and is
  **DEGENERATE**: only one lane continues, so it is a single-CAV lane-1 hold at 20 m/s, 120 s, dmax 1.0, cooldown 120. The
  documented 2-CAV alternative (HOLD_R2_DUO01, lanes 0+1) is a different mechanism (a full-width wall).
- **Prior evidence (NEWMAP_RESEARCH.md §5.1, native base).**
  - Sign convention: S = min J(solo) - J(group) and T = J(base) - J(group); positive = better.
  - wz2to1 wall W1 (15 m/s, 90 s), 9 situations / 4 worlds, 0 admitted:
    - S = -1,444 [-2,655, -233];
    - T = -516 [-2,838, +1,805].
  - The probe also found a severe lane-end capacity drop (about 770 veh/h discharge at 1,400).
- **No-WM fallbacks.**
  - (i) "Search by SUMO only": not the registered pipeline, and only defensible as a separate, labelled study.
  - (ii) "WM searches on wz3to2, plans ported": the plans are SUMO-lane-1/2 roles. Lane 2 does not exist on wz2to1, so the
    port is again degenerate and untested by the WM. It is not a test of the pipeline.
  - Only (a) U0 vs HOLD_R2 is cleanly runnable.

## 2. weave
Refused. It is not a lane drop: map_descriptor.load requires `bottleneck.dropped_lane`, and the WM frame needs a closing lane.
In the probe, weave breakdown was gridlock-like in 3 of 4 worlds. Wall W1 had 0 admitted, S = -114 [-369, +140].

## 3. Earlier merge scene (highway_merge_shepherd_long)
Source: a read-only subagent; key numbers spot-checked in NEWMAP_RESEARCH.md.
- **Why the project left it.** RUN/NEWMAP_RESEARCH.md:4-7, 171-188 (S as above, positive = better):
  - merge: about 1 admitted 4-CAV wall situation (3/6 seeds, S +250 to +490); mining 0 admitted moves; clean2 B mean synergy -0.034;
  - wz3to2: 8/34 admitted.
  - So the merge was dropped because group cooperation was rare and fragile there.
- **Registered evidence on holds and walls there.**
  - COOPPLAN H300 wall panel (6 worlds; group credit C_G, positive = better): **C_G -681 [-1,296, -144], 0/4 positive**.
  - HICAV verdict NO.
  - COOP_PROBE: walls are "a per-seed bet".
  - The clean2 gain was almost entirely single-actor (one ramp CAV).
  - OE-62 "HARMFUL".
  - Benefits that do exist at the merge come from single-CAV gap and yield moves (COOPPLAN J900 +1,584 [+588, +2,765] saving vs MAPPO,
    mostly gap/yield; walls executed 0 times at H120).
  - **Prior registered evidence says multi-CAV holds do not robustly help at the merge.**
- **Tools.** oracle_plan (X_B 2696, edge 'up' lanes 0/1/2), strategy_search geometry and nm_finaleval/nm_eval_runner are
  wz3to2-only. Only the old clean3 runtime runs the merge map (map_descriptor.ORIGINAL).
  - A mechanical duo12 port meters the mainline, while the bottleneck is ramp merging over 75 m. That would be unfair, or would need
    retuning, which counts as outcome exposure.
- **WM and MAPPO.**
  - The only local merge WM is WM2 (E:/agent-work/c4-execparity3/_parity3_data/wm2/wm_s.pt, sha256 b8797242c6b736b5...).
    Its ranking skill is weak (pooled Spearman 0.10).
  - Drive (gdrive:research-dense-cold/archive_gpuR_from_laptop_20261026/runs/) holds merge-era WMs c4_wm_v7, v7ft,
    v8ft2_full, pairdiff and others. Their shas are not verified.
  - MAPPO d9042cc5 was trained there and is available (section 4).
- **Seeds.** Merge integers are heavily used:
  - 962/964/977xxx; 963000-963959; 965140-969999; 970xxx; 973xxx; 976400-976447; 980400-980411;
  - 981000-981091; 982000-982047; 983100-983323; 984000-984023; 986000-988071+; 989600-989635; 990600-990659;
    995000-995199 (HICAV).
  - Candidates to audit: TRAIN 974000-974999 and panel 978000-978095 (alternates 985000-985095, 994000-994095).
- **Estimated chance of a clean full run by Nov 13: about 10% or less.** The stack would need a full port (NEWMAP estimated 9-15 days the
  other way), and the hold port would be unfair.

## 4. Where the checkpoints are
- **WM-E ft_r1 6da9d333: NOT REACHABLE.**
  - E: search: `find E:/ -name 'wm_s*.pt'` found 3,567 files. All are tests/golden or pytest archives except 14; I hashed those 14 and none
    is 6da9d333. No file is named *6da9d333*.
  - Drive search (filtered, restricted archive excluded): `**wm_s*.pt`, `**ft_r1**` found 82 merge-era / golden files and no ft_r1.
  - The registered location /root/autodl-fs/wzwm/ft_r1/phases/B/a1/run/wm_s.pt is on a remote box (gpuA-family).
  - This also matters for any future search, not just A38.
- **MAPPO d9042cc5: AVAILABLE**, verified. It is inside E:/agent-work/a32/relay/stage.tar, member
  root/mining20261106/cpuv_home/mappospeed/production2/policy_00169.pt, with sha256 d9042cc5a05e...cffa9ca (matches BOUND_U0).
  - The same tarball holds the frozen plant source (src_p20c_b5268408_5ecc4358), the e3final runtime and the p20c bank-root
    bindings. This is the package gpuB ran amendments 32/33/35 from.
  - The bundle script reuses it for any new Linux box. A37 can use it too.

## 5. Ranking and recommendation
| road | (a) clean run by Nov 13 | (b) hold-port fairness | (c) prior evidence holds help |
|---|---|---|---|
| wz2to1 | full pipeline: ~0% (WM blocked twice). Reduced U0 vs HOLD_R2: ~80%, about 2-3 h on 48 cores | degenerate (single CAV) | unfavourable (W1 S -1,444 [-2,655, -233], T -516 n.s.) |
| merge | full: <=10%; reduced: needs a runner port, several days | poor (mainline meter vs ramp merge) | **negative** for walls (COOPPLAN C_G -681 [-1,296, -144]) |
| weave | refused | n/a | none (0 admitted) |

**Recommendation.** Do NOT run A38 as registered. Do not spend the CPU on a second-road WM pipeline before Nov 15; instead, use the box
tonight for A37 on wz3to2. In the paper, state the single-road limitation, and cite the NEWMAP probe numbers above as
the reason the pipeline was only exercised where a multi-lane drop exists. If MAIN still wants second-road
evidence, the only clean option is the reduced wz2to1 transfer of the hand-written hold (U0 vs HOLD_R2, optionally
HOLD_R2_DUO01). It is launch-ready, but it would be reported as a weak, exploratory, degenerate-hold check, not as a
replication of the cooperation result.

## Launch-ready reduced variant (only if MAIN registers it; draft in PROGRESS.md)
- Branch c4/a38-road2 (commit in the final report).
- Box scripts are in E:/agent-work/a38/box:
  - bundle_laptop.sh -> a38_bundle.tar;
  - bootstrap_box.sh;
  - run_a38.sh: tests -> identity gate (wz3to2) -> calibration (U0, 3 demands x 8 TRAIN worlds) -> demand pick
    -> eval -> reports -> analyse;
  - launch_a38.sh (setsid nohup, single instance);
  - offload_laptop.sh (sha-verified, never deletes);
  - laptop_analyse.py.
- Budgets:
  - calibration 28,800 sim-s (cap 31,680);
  - eval 2 arms x 96 x 1,200 + identity 4,800 = 235,200 (cap 258,720); with DUO01, 350,400.
  - Wall time at 44 workers: about 2-3 h total.
- Seeds: TRAIN 996000-996199, panel 997000-997095 (audit UNUSED, SEED_AUDIT.json).
