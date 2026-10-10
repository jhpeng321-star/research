# LPO_PLAN — WM-S leave-PI-out generalisation run (P5)
Status: READY-TO-LAUNCH (prep complete; run NOT started). Metadata only; zero SUMO;
sealed sources untouched. Branch wms/p5; builder `output/wd01/wm_n1/lpo/` (18/18 tests
with p5splits pass). Comparator: **WM-S-TRAIN-20261005** (full-data run, 3060 TRAIN
windows, packs pack_v3+pack_dspeed+pack_damping, code 2ffeac6, batch 8, truncation 20,
start-shift 0.75, sumo-fidelity, curriculum 10-120 s).
## Windows (this dir; schema wm_s_lpo_windows/1; receipt = BUILD_RECEIPT.json)
- windows_train_noPI.json — 1307 windows / 184 worlds, NO CAV ever on PI (damping 358,
  native 254, trajectory 560, wall 135); sha256 f2493e14dac4a5be…
- windows_val.json — 189 windows / 26 worlds, same rule; deterministic world-grouped
  val (sha256(world)[:16] mod 10 == 0; worlds whole, no leakage); sha256 e6924db7…
- pi_eval_states.json — 1494 held-out PI TRAIN states for scoring: fleet-PI 161
  (truearms+chunk4), WM-72 __pi arms 146, fixed/switching with a PI schedule event
  1187; sha256 eaa5dfcaca7c1cfa…
- pi_excluded_runs.json — 1564 pack keys = 1494 PI runs + 70 chunk1_fix27 runs whose
  schedules are unrecoverable (raw deleted; fail-closed, not claimed PI); sha256 ab6144…
- Rule is schedule-aware (per-run switch_schedule.json events; only actions {native,
  damping, pi-external-u} occur); pack_dspeed (arm0-3) + pack_damping (base/native_base)
  asserted PI-free by construction. Train+val+PI+unverif = 3060 = comparator corpus
  exactly (changed variable: PI removal; plus the val monitoring holdout).
## Trainer changes (minimal patch on a NEW branch; wms/p2 NOT edited)
1. wms/cli.py `--exclude-window-list PATH` (repeatable): filter corpus._v1 items by pack
   run key in TrajectoryDataset.from_real_packs AFTER IdeaACorpus split checks, BEFORE
   identity/position validation; refuse on absent keys; log kept counts (1496/2153, 644, 263).
2. `--window-list PATH`: second TRAIN dataset from windows_val.json, monitoring only
   (per-stage val loss; NO early stopping, NO checkpoint selection).
3. Chain: wms_chain.sh verbatim + both flags in run(); smoke400 gate + plan_steps
   unchanged. GPU size ≈ comparator (step budget fixed; 43% windows cycle more often).
## After the run
Score on pi_eval_states (primary: 161 fleet-PI arms; mixed arms secondary), then the
registered CAL-tune PI eval (split ii) as its own sealed step; compare vs
WM-S-TRAIN-20261005 and keep-speed / pure-physics baselines via wms_eval.
