# MAPPO_WZ_PLAN: a fair speed-only MAPPO comparator on wz3to2 (task N7, 2026-11-06)

Branch `c4/nm-mappo` (base `3e5bc9f9`), worktree `E:/agent-work/c4-nm-mappo`. Prepared only:
**no training, no SUMO, no ssh were run.** All sim-s below are a quote, not spend.

## 0. Summary
- **Full fair run (deliverable 1):** the original speed-only recipe on wz3to2. It warm-starts from
  MAPPO u0 and runs 169 iterations x 46 episodes x (300 s warmup + 600 s control). Cost:
  **6,996,600 sim-s** (4,664,400 controlled s). Wall time: **6.5-8.4 h on cpuV** (48 cores,
  dedicated) or **7.8-10.2 h on cpuQ** (32 cores), which matches the NEWMAP_RESEARCH estimate of
  "about 7.0M sim-s, about 10 h on cpuQ".
- **Reduced option (deliverable 2):** the same run stopped at a pre-registered snapshot.
  **FT-42** costs **1,738,800 sim-s**: 1.6-2.1 h on cpuV, 1.9-2.5 h on cpuQ, 2.8-3.7 h on a 12-core
  cpuV share. **FT-24** costs 993,600 sim-s: 0.9-1.2 h / 1.1-1.5 h / 1.6-2.1 h.
- **Recommendation:** launch ONE full run, preferably on a second machine so N4 keeps cpuV.
  Snapshots 24/42/84 are nested inside it as reduced-cost fallbacks, and an outcome-blind time
  rule binds the comparator (section 5).

## 1. Sources and what "fair" means here
- **Original comparator.** S-MAPPOSPEED (REGISTRATION.md 2026-10-30, branch `c4/mappospeed`):
  - speed-only MAPPO: `[policy] lane_control="native"`, no lane head, `lanes=None`, so lane
    changes are SUMO-native;
  - init from lane-on v5 `mappo_v5_final.pt` (`aecd7d70...`, iteration 170, itself 7,038,000
    sim-s of MAPPO);
  - then 169 iterations x 46 episodes on cpuQ with 30 workers and 16 update threads;
  - command: `--iterations 169 --episodes-per-iteration 46 --max-sumo-seconds 4664400`;
  - result: `production2/policy_00169.pt`, SHA256
    `d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca`. This is the frozen base
    actor of clean3 ("MAPPO u0").
  - Its parsed config (`speed.toml` sha `a7fc8e09...`) is frozen in
    `tools/nm_mappo/ORIGINAL_SPEED_CONFIG.json`, taken from `c4/mappospeed:tools/mappospeed/PREFLIGHT.json`.
- **Same algorithm.** The MAPPO trainer source is identical:
  - `src/wms/control/mappo/*` is unchanged between merge-base `571c1b8b` and `3e5bc9f9`;
  - the speed-only patch `cf97e6fd` is cherry-picked verbatim (`464b69cd`).
- **Same hyperparameters.** `tests/unit/test_nm_mappo.py` asserts that
  `configs/c4_mappo_speed_wz3to2.toml` equals the original EXCEPT these declared deltas:
  - `[map]` and `[env] name="map"`: the env and its map kwargs come from the N1 descriptor;
  - seeds `989200-989599` (the registered wz3to2 TRAIN block);
  - demand and penetration cells `"map"` (N1's registered TRAIN cells);
  - `iterations=169` and `max_sumo_seconds=4664400` written into the file (the original passed
    them on the CLI);
  - `accel_max_mps2=2.6` made explicit (it equals the u0 checkpoint's stored bound);
  - `cleanup_scenarios=true` (infra only).
- **Same starting point as our method.** The full run starts from u0, the same actor as the
  clean3-stack base on wz3to2. So the comparison isolates the learning method on the new map:
  MAPPO's on-policy SUMO PPO versus our cooperation layer.
- **Same CPU budget order.**
  - The full run gives MAPPO about 7.0M on-map sim-s, about 15x the per-arm training experience
    of the clean3-style method (57,600 sim-s per update for A+B+R, x 24 updates, / 3 arms ≈ 0.46M).
  - Even FT-24 gives MAPPO about 2x that experience.
- **From scratch** (`full169_scratch`) is quoted for completeness only. It is weaker: the
  original actor had about 14M sim-s of merge-map MAPPO history before its 169 speed-only iterations.

## 2. Code delivered
| File | Purpose |
|---|---|
| `src/wms/control/mappo/{agent,config_helpers,eval_bridge,rollout,trainer}.py`, `tests/unit/test_c4_mappospeed.py` | Speed-only mode (cherry-pick `cf97e6fd`) |
| `src/wms/control/policy.py` | Speed-only deploy support (`PolicyExecutor.native_lanes`, loader `lane_control`), ported from candA-int V7POL. Without it, the current deploy path sends an all-−1 lane tensor and enables laneChangeMode 513. Default behaviour is unchanged. |
| `src/wms/control/mappo/mapbind.py` (+ 6-line hook in `config_helpers.build_trainer`) | `[map]` binding: N1 descriptor or a refusing **stub**. It resolves the env and cells and enforces TRAIN seeds only: no HELD-OUT 989600-989699, no 964000-964199, 962xxx or 977xxx. A stub binding with a SUMO env raises, so no SUMO episode can run on a placeholder map. Without `[map]`, configs behave byte-identically. |
| `configs/c4_mappo_speed_wz3to2.toml` | The run config (sha256-LF `55f3aee8...` at commit time; set `[map] api` when N1 lands, which changes the sha) |
| `tools/nm_mappo/quote.py` | Cost model, calibrated on S-MAPPOSPEED cpuQ timings |
| `tools/nm_mappo/launch.py` | plan / check / ack / smoke / run / resume / status / stop / bind (mechanical gates) |
| `tools/nm_mappo/supervise.py` | Linux supervisor: nice 19, FD limit ≥ 65536, wall cap, STOP forwarding, own PIDs via pidfd. Adapted from S-MAPPOSPEED. |
| `tools/nm_mappo/sanity_eval.py` | Sanity panel: native / u0-as-is / MAPPO-wz, deterministic deploy executor |
| `tests/unit/test_nm_mappo.py` | 13 CPU tests: fake env and fake N1 descriptor |

**N1 interface expected.** `[map] api = "module:function"`; `function("wz3to2")` returns a
descriptor with attributes or keys (aliases are accepted, see `mapbind.py`):
- `name`;
- `env_name` (default `traffic_shepherd.c4.env.CavSumoEnv`);
- `env_kwargs` (map-selecting kwargs, default `{"map_name": "wz3to2"}`);
- `train_seeds`, `heldout_seeds`, `demands_vph`, `penetrations`;
- optional `spec_sha256`.

N1 must also:
- admit 989200-989599 in the env seed guard (`traffic_shepherd/c4/seeds.check_case` refuses them today);
- keep the actor observation layout. The strict u0 load fails otherwise; `launch.py check` detects that.

## 3. Cost options
The model: iteration wall = waves x 31.86 s x contention + 103.1 s update. It is calibrated on
the original run:
- iterations 1-2: collect 64.1 / 63.3 s, update 103.9 / 117.0 s;
- 169 iterations in ≤ 7.83 h.

wz3to2 slowdown is assumed at x1.0-1.3. The first 2 production iterations (82,800 sim-s) re-quote the ETA.

| Option | Init | Iterations | Planned sim-s (excl. 660 smoke) | Hard cap (1-retry reserve) | cpuV 46 workers (h) | cpuQ 30 workers (h) | cpuV 12-core share (h) |
|---|---|---:|---:|---:|---:|---:|---:|
| **full169_u0** | u0 | 169 | **6,996,600** | 13,994,520 | **6.48-8.43** | **7.83-10.18** | 11.3-14.7 |
| full169_scratch | none | 169 | 6,996,600 | 13,994,520 | 6.48-8.43 | 7.83-10.18 | 11.3-14.7 |
| ft84_u0 | u0 | 84 | 3,477,600 | 6,956,520 | 3.22-4.19 | 3.89-5.06 | 5.62-7.30 |
| **ft42_u0** | u0 | 42 | **1,738,800** | 3,478,920 | **1.61-2.09** | **1.95-2.53** | 2.81-3.65 |
| ft24_u0 | u0 | 24 | 993,600 | 1,988,520 | 0.92-1.20 | 1.11-1.45 | 1.61-2.09 |

- Controlled seconds (the trainer's `--max-sumo-seconds`) = iterations x 46 x 600.
- The smoke costs 660 sim-s (+660 reserve).
- The sanity panel costs 32,400 sim-s (12 worlds x 3 arms x 900 s) and takes a few minutes.
- cpuV gains little over cpuQ: the PPO update, about 103 s per iteration on 16 threads, dominates
  and does not scale beyond 16 threads.
- An extra MAPPO-wz arm in N4's final paired evaluation is N4's budget: about 115,200 sim-s at
  the clean3 shape of 96 x 1200 s.

## 4. Exact commands (run ON the box, from the staged code root; never by an agent without GO)
```bash
# 0. stage: integration commit = N1 + c4/nm-mappo (with [map] api set), as an LF code bundle
git archive --format=tar.gz -o nm_mappo_code.tgz <integration-commit>   # copy + extract to ~/nm_mappo/code
#    u0: copy any verified copy of policy_00169.pt (cpuQ ~/mappospeed/production2/policy_00169.pt,
#    gpuA /root/autodl-tmp/c4/speedhead20261030/mappospeed_final169.pt); sha256 must be d9042cc5...
# 1. plan (zero SUMO): PLAN.json with exact argv, config/init sha, binding, budget, wall cap
python tools/nm_mappo/launch.py plan --option full169_u0 --host cpuQ \
    --run-dir ~/nm_mappo/wz_full169_u0 --init ~/nm_mappo/inputs/policy_00169.pt \
    --python ~/miniconda3/bin/python --sumo-home ~/miniconda3/lib/python3.10/site-packages/sumo
python tools/nm_mappo/quote.py                                # cost table
# 2. CPU check (zero SUMO): the exact config on the fake env + strict u0 warm start on this tree
python tools/nm_mappo/launch.py check --run-dir ~/nm_mappo/wz_full169_u0
python -m pytest -q tests/unit/test_nm_mappo.py tests/unit/test_c4_mappospeed.py tests/unit/test_c4_mappo.py
# 3. MAIN appends the registration block (section 7) containing the PLAN sha256, then:
python tools/nm_mappo/launch.py ack --run-dir ~/nm_mappo/wz_full169_u0 --registration <REGISTRATION.md copy>
# 4. SUMO smoke (660 sim-s): 1 it x 2 ep x (300+30) s, 2 workers; gate = completed, 0 failures/
#    collisions/teleports/lane commands/policy lane changes/dropped vehicle steps (K=192 overflow)
python tools/nm_mappo/launch.py smoke --run-dir ~/nm_mappo/wz_full169_u0
# 5. production (detached supervisor: setsid nohup nice -n 19; wall cap 1.5 x quote-high)
python tools/nm_mappo/launch.py run --run-dir ~/nm_mappo/wz_full169_u0
#    trainer argv (identical to the original apart from paths):
#    python -u -m wms.control.mappo.train --config configs/c4_mappo_speed_wz3to2.toml --out RUN/production \
#      --init-from U0 --update-threads 16 --workers 30 --iterations 169 --episodes-per-iteration 46 \
#      --max-sumo-seconds 4664400            (cpuV plan: --workers 46)
# 6. monitor / stop (clean, resumable) / resume after a box restart
python tools/nm_mappo/launch.py status --run-dir ~/nm_mappo/wz_full169_u0
python tools/nm_mappo/launch.py stop   --run-dir ~/nm_mappo/wz_full169_u0
python tools/nm_mappo/launch.py resume --run-dir ~/nm_mappo/wz_full169_u0
# 7. bind the comparator actor (section 5 rule) -> BOUND_<it>.json with sha256 for N4 / final eval
python tools/nm_mappo/launch.py bind --run-dir ~/nm_mappo/wz_full169_u0 --iteration 169
```
- **Standalone reduced run:** use `--option ft42_u0` (or `ft24_u0`, `ft84_u0`) in step 1; the
  rest is unchanged. It is equivalent to the nested snapshot: no LR or entropy schedule depends
  on the total iteration count.
- **Workers:** `--host cpuV` sets 46 workers (one wave). For a shared box use `--workers 12`,
  which matches the cpuV_shared12 row.

## 5. Sanity-evaluation plan and comparator rule (pre-registered here)
1. **Zero-SUMO checks** (before registration):
   - `launch.py check` passes: strict u0 load, no lane head, bounds [-3, 2.6], map recorded;
   - the binding is non-stub and N1's descriptor sha is recorded;
   - the seeds lie in TRAIN and none in HELD-OUT;
   - the CPU tests pass.
2. **SUMO smoke gate:** mechanical, enforced by `launch.py run` (section 4, step 4).
3. **First two production iterations:**
   - 46 + 46 episodes, finite returns;
   - 0 failures, collisions, teleports and lane commands;
   - record collect/update wall and re-quote the ETA with `launch.py status`;
   - stop if non-finite, or if a collision or teleport appears. A pause needs a MAIN decision.
4. **Learning sanity** (descriptive, from `log.jsonl`): mean episode return and team-delay
   components over iterations; mean of the last 5 iterations vs the first 5.
5. **Post-training sanity panel** (`tools/nm_mappo/sanity_eval.py`, registered separately,
   32,400 sim-s):
   - Panel: one world per registered TRAIN cell on the last TRAIN seeds (989588-989599 for 12
     cells). Never a HELD-OUT seed.
   - Arms: native, u0-as-is, MAPPO-wz (the bound actor). Each arm gets 300 s native warmup and a
     600 s deterministic window.
   - Paired arm − native and wz − u0 differences (negative = better), with a world bootstrap 95% CI
     (B = 2000, rng 0).
   - Flags:
     - S1: 0 collisions and teleports, 0 lane commands for the speed-only arms, 0 dropped vehicle steps;
     - S2: all metrics finite;
     - S3: MAPPO-wz is not significantly worse than native.
   - A failed S1 or S2 means the run is not a valid comparator; investigate, with no silent retry.
   - A failed S3 is reported, not "fixed" by picking another checkpoint.
6. **Comparator binding rule (outcome-blind).**
   - At the moment N4's final paired evaluation is dispatched, bind iteration 169 if it exists.
   - Otherwise bind the latest completed pre-registered snapshot, in the order 84, 42, 24.
   - No evaluation outcome is looked at before binding.
   - `launch.py bind` writes the sha256 receipt.
7. **Claim rule.**
   - The headline evaluation is N4's final paired evaluation on HELD-OUT 989600-989699, with
     arms ours, MAPPO-wz and u0-as-is (and native) on identical worlds.
   - "Beats MAPPO on wz3to2" is stated against MAPPO-wz.
   - If u0-as-is is better than MAPPO-wz on that panel, the claim must also beat u0-as-is. This
     is conservative: it can only make the claim harder.
   - nm-finaleval must replace the hard-coded `MAPPO_SHA = d9042cc5...` comparator binding with
     the bound MAPPO-wz sha, and keep u0 as a separate arm.

## 6. Recommendation (Nov 15 deadline, 9 days)
- **Do it: full169_u0 on a second machine (cpuQ-class, 32 cores).** Launch it as soon as N1 lands
  and the check, registration and smoke pass. It needs about 8-10 h and runs overnight, beside
  N4 on cpuV.
  - The two runs do not compete for cores. The finished comparator (or snapshot 84 or 42) exists
    long before N4's u24 final evaluation.
  - Register the full 6,996,600 sim-s once, with a 1-retry cap of 13,994,520. Actual spend is
    whatever runs before completion or a deliberate stop.
- **If only cpuV is available:**
  - run the same plan with `--workers 12` after N4's smoke/gate (11.3-14.7 h);
  - or, if N4 needs those cores too, run `ft42_u0` standalone: 1,738,800 sim-s, about 2.8-3.7 h
    on 12 cores or 1.6-2.1 h with all 48 cores;
  - extend to 169 iterations later only if time remains (new plan, same registration pattern).
- **Do not run `full169_scratch`.** It costs the same and is a weaker, less comparable baseline.
- **Critical-path risks:**
  - N1 must land (map API, env seed guard for 989xxx, obs layout kept for the strict u0 load);
  - K = 192 slot overflow at 2800 veh/h on 4 km is caught by the smoke gate;
  - cpuV and cpuQ disk: scenario cleanup is on, checkpoints are about 2.3 MB x keep_every 1 x 169
    ≈ 0.4 GB, plus exports.

## 7. Registration block draft (MAIN appends before any SUMO; fill the bracketed values)
```
## 2026-11-0x S-MAPPOWZ (N7) ALLOCATION - registered BEFORE ANY SUMO
- Question: speed-only MAPPO trained on wz3to2 with the original S-MAPPOSPEED recipe, as the fair
  "beats MAPPO" comparator for the wz3to2 coop method. Closest prior: S-MAPPOSPEED (10-30, merge map,
  init aecd7d70, 169 x 46 x 900 s, cpuQ). Exact difference: map (wz3to2 lane drop via N1 descriptor
  [api, spec sha]), seeds 989200-989599 (TRAIN), N1 TRAIN cells; init = its own result u0 d9042cc5.
- Code: <integration commit> (c4/nm-mappo <commit> + N1 <commit>); config sha256-LF <...>;
  PLAN.json sha256 <...> (launch.py plan, option full169_u0, host <cpuQ|cpuV>, workers <30|46|12>).
- Budget: planned 6,996,600 sim-s (4,664,400 controlled) + smoke 660; cap 13,994,520 incl. one retry
  per episode; wall cap <PLAN.wall_cap_s> s from smoke start; nice 19; own PIDs only; no deletion of
  checkpoints/logs. Snapshots 24/42/84 pre-registered; comparator binding rule MAPPO_WZ_PLAN.md s5.
- Never: HELD-OUT 989600-989699, 964000-964199, 962xxx, 977xxx; sealed/holdout paths; truth files.
- Sanity panel (separate block when used): 989588-989599 x native/u0/wz, 32,400 sim-s, panel sha <...>.
```

## 8. Not done / open
- No SUMO, smoke, training or sanity evaluation was executed.
- No registration was appended: REGISTRATION.md lives outside this worktree.
- `protocols/experiment_history.json` and OPEN_QUESTIONS were not updated; that is for the
  integrator at registration and close-out.
- The `[map] api` is unset, so the stub refuses SUMO.
- Two code paths cannot be checked on this laptop:
  - the supervisor's Linux-only parts (pidfd, /proc, setsid);
  - the code ported from the S-MAPPOSPEED supervisor, which was exercised on cpuQ.
- **Fairness caveat.** MAPPO-wz sees the base actor observations (as u0 does), and N1 redefines
  them for the lane drop. If N2's lane-drop features or V2V reach only our cooperation layer,
  state that in the paper. A MAPPO+N2-features variant would change the actor input and is NOT
  the "same algorithm" run quoted here.
