# ENTRY_FIX: open-road A11 rollout (entries + exits), TRAIN-only, no GPU
Branch exp/rollout-entry (merged exp/rollout-exit 4873ede). Code df7c518 (+ docs). Model member_1 sha 41221852...f21e8.
## Implemented
- Existing engine arrival path (rollout2 arrivals_enabled + bundle.arrival + ArrivalSampler) NOT usable: A11's arrival head is degenerate
  (count_mu=0, speed_mu=1 on the smoke counter: inserts nothing), HDV-only, lane 1/2 only, "behind rearmost vehicle" not at the entrance.
- So `closedloop/rollout_entries.py`: arrivals from OBSERVED entrance counters of the t0 token (rate = mean of 12 most recent 10 s bins),
  Poisson per 0.5 s step, seeded (identical draws for all plans of a batch). Mainline split over lanes 0/1 by observed lane share, ramp -> lane 2;
  CAV share = observed share in snapshot (spawned CAVs run native, code 0); speed = mean of 3 rearmost observed vehicles on the lane;
  inserted at the lane start (s_down = -offset) only with safe gap (min 2 m + 1 s*v + braking), else pending (<=6) and retried next step.
- Flags (default OFF, byte-identical): batched_rollout(open_entries=, physical_exits=), A11Predictor(open_entries=, physical_exits=, open_road=True = both),
  runner/launcher `--open-road`.  Tests: closedloop/tests/test_rollout_entries.py (8 synthetic) + existing; 76 passed (closedloop + rollout2).
- Finding: recorded cav_plan_ch is alive-since-t0, so entrants are never counted in the 120 s plan summary (recorded or rollout). Entries only act
  through added HDV traffic; the n_cav gap was exits only (exit fix: s_down vs physical lane end).
## Diagnostic (8 TRAIN chunk-4 states, PI at tau=300 s, head dV; `_out/entry_diag.py|txt|json`)
state            TRUE  scorer(rec)  OLD    EXIT   OPEN(exit+entries)   native-head OLD/OPEN
w8676m2700p20t360 1123   1355       3732    61     80                  3732/80
w8626m2700p02t300   39   -519        501   102    103                   501/103
w8629m2000p02t420  -55   -155        180   -41    -41                   180/-41
w8678m2700p20t300 1324   1308      10577   452    517                 10544/486
w8677m2700p20t360 2382    680       5186   211    204                  5307/178
w8680m2600p20t300  -91    995       6947   -31     -3                  6911/-5
w8679m2400p20t540 1622   1284       4026   118    190                  4026/190
w8683m2400p20t540-1785    270       1922   159    167                  1922/167
MAE vs TRUE: OLD 3564, EXIT 961, OPEN 947, scorer(recorded plan) 761. corr vs TRUE: OLD .45, EXIT .35, OPEN .37, scorer .52.
n_cav bin11 (w8676): recorded 8.6, OLD 21.9, OPEN 2.2 (over-drops). Entries (114 inserted in 120 s, 18 CAVs) change dV by only +20..+70 vs exit-only.
## Smoke (TRAIN w8676m2700p20t360, pl01_a11 --open-road, 180 s, 8 threads, 445 s wall; _out/smoke_open_*, SMOKES.jsonl)
Complete, 360 rows, 0 collisions, 0 teleports, speed-mode-27 guard 4612/0 violations. All 3 decisions picked native for every CAV (25/28/27 CAVs).
Non-native candidate dV (min/med/max): d0 78/80/96, d1 477/485/543, d2 531/534/537 (before: ~3700 constant). Native is still defined as 0, so native wins
by the head(native) offset (~+80..+530 now); PI == native in the rollout. Use --native-reference head to remove the offset.
## Remaining gaps
- Speeds: rollout mean_v 22-27 vs recorded 16-23, min_v 7-20 vs ~0-5 (recorded has stopped CAVs); no congestion/queue reproduced -> exits too fast
  (late n_cav over-drop) and dV near zero for every plan; ordering vs TRUE still weak (corr ~.4). Magnitude is fixed, discrimination is not.
- Spawned CAVs are native only; no fleet policy for newcomers; slots recycled only after exits (K=192 headroom).
## Enable
`python .../closedloop/runner.py ... --open-road` (or launcher `--open-road`); A11Predictor(bundle, open_road=True).
