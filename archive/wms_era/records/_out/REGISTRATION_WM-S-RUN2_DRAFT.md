# REGISTRY ENTRY — WM-S-RUN2 (DRAFT, pending owner freeze)

Template: `E:/research-dense/docs/EXPERIMENT_RECORD_TEMPLATE.md` (schema
`experiment_history/2`). This is a **draft registration** prepared by the RUN 2
integrator; it is NOT yet appended to the master ledger and does not by itself
authorise a GPU launch.

- **Date registered / family / name:** 2026-10-06 / WM-S structured world model /
  Run 2 "merge-cal + entries-v2 + traffic losses + 300 s curriculum + stability".
- **Scientific question / hypothesis:** Run 1 (WM-S-TRAIN-20261005) trained a pure
  car-level matcher with a 10/30/60/120 s curriculum and no vehicle entries, no
  calibrated merge gap acceptance and no traffic-level objective. Adding
  (A1) junction carry-through, (A2/A3) observed-history entries, (A4) calibrated
  `merge_cal`, (B1) differentiable traffic-level losses, (B2) a 300 s stage, and
  (B3) the wms/stab stability features (feasibility projection, capped/contractive
  residual, conservation metric)
  closes the run-1 merge-queue / delay gap vs recorded traffic at long horizons
  (≥120 s) **without** degrading car-level valid-target accuracy at ≤60 s.
- **Status:** proposed (registered-not-yet-executed), draft.
- **Record kind:** experiment (training run; GPU wall-clock charged).
- **Controller/actuator + source identity:** WM-S torch physics (structure +
  residual), code commit **6ee1d8a096f6** on branch `wms/run2b` (= `wms/run2`
  e61d653 + merged `wms/stab` 6f9baaf); code tar
  `_out/code_run2_6ee1d8a096f6.tar` sha256
  `af65104e15503894aed8c33ba61a5072191b7d091b1d7d57fb447057de7c5a2a`.
  Calibration params = `E:/rd-wt/ws-merge/_out/merge_params.json` (byte-identical
  copy committed at `_out/merge_params.json`, sha256 `c23385f9...`).
- **Damping baseline and car-following model:** SUMO-fidelity IDM (`sumo_idm`,
  `sumo_fidelity=true`) + MOBIL lane change inside the WM-S rollout; HDVs IDM.
- **Network/scenario; exposed seed/split metadata:** preserved highway-merge
  network; TRAIN packs `pack_v3` + `pack_dspeed` + `pack_damping` with their
  identities/overlays; window list `windows_train_noPI.json` (1307 windows /
  184 worlds; sha256 `f2493e14...`), exclusion `pi_excluded_runs.json`; val
  monitoring `windows_val.json` (189 windows). Split = **TRAIN only**; no CAL,
  CAL_SPLIT_A4, HOLDOUT, chunk-5, J4B, INC1 or sealed bytes read.
- **Runtime observation contract; privileged info:** identical to run 1 — CAV
  self-state + locally detected-neighbour reports with receipt age/missingness/
  coverage; no true penetration, hidden driver type or global metric online.
  Entries are generated only from RSU-observable **pre-t0** history.
- **Decision frame:** continuation_keep_vs_adjust (training objective).
- **Changed variables:** `merge_cal` ON; `--entries-v2 --entry-pad 8`;
  `--traffic-loss`; 5-stage curriculum 10/30/60/120/300 s (ratios 1/1/0.75/0.5/0.25);
  junction carry-through active; **long-horizon stability** (wms/stab):
  feasibility projection ON (`--projection --projection-min-gap-m 2.0`),
  capped residual ON (`--residual-cap-mps2 1.5 --lane-logit-cap 3.0`), contractive
  regulariser ON (`--stability-weight 0.01 --stability-margin 1.0
  --stability-probes 1`), conservation **metric** ON (`--conservation-metric`,
  loss term OFF). **Fixed:** batch 8, truncation 20, start-shift 0.75,
  sumo-fidelity, packs/window lists, seed 101, optimizer defaults,
  `--num-workers 10`. **Noise injection OFF** (`--state-noise-std-m/-mps` absent,
  stability features `projection_enabled`/caps installed as trainer attributes).
  Known confounder: the mechanisms are bundled in one run, so a closure of the
  long-horizon gap cannot be attributed to one mechanism by this run alone.
- **Comparators; matching/reuse evidence:** primary comparator
  **WM-S-TRAIN-20261005** (run-1 full-data, code 2ffeac6, curriculum 10–120 s,
  no entries). Reused, not re-run: the same TRAIN window split and packs
  (identical receipts: corpus 3060, kept 1307) and the same merged junction/entry
  code. Matching is by identical TRAIN windows and identical inference-time
  scoring in a later `wms_eval` step (this registration covers training only).
- **Requested assembly/HOLD/exit and target speeds (nominal, pre-delivery):**
  n/a (offline training; no SUMO, no walls, no HOLD).
- **Registered budget:** `plan_steps` target `BUDGET_S=18000` s (~5 h) of full-run
  compute chosen per-stage from the smoke400 log; hard cap `HARD_CAP_S=28800` s
  (8 h) — the chain refuses to start the full run above the cap. One GPU (gpuA),
  one process. Charging: wall-clock of the full run, recorded actual-vs-estimated.
  Stopping rules: smoke400 gate (400 records, all numeric terms finite,
  traffic-loss term present, entrant total > 0, **projection and conservation
  metrics present/finite**) must pass; plan estimate ≤ 8 h.
  Retry policy: none automatic; a failed/stopped job is preserved and re-registered.
- **Original criteria:** to be quoted verbatim from the frozen run-1 acceptance
  record (RUN1_EVAL / RUN1_EVAL_FAIR) at owner freeze. Draft intent: no car-level
  valid-target regression at ≤60 s and improved long-horizon (120/300 s)
  traffic-level agreement vs recorded. **Not weakened later.**
- **Closest prior experiment(s) and exact unanswered difference:** closest =
  WM-S-TRAIN-20261005 (and the LPO generalisation prep, `_out/lpo/LPO_PLAN.md`).
  Exact unanswered difference: run 1 deliberately excluded entries from training
  (`_ref/C1_READY.md` "Entries in TRAINING: NO") and had no `merge_cal`,
  `--traffic-loss`, or a >120 s stage; no prior run isolates the calibrated-merge
  + observed-history-entry + traffic-objective + 300 s mechanism bundle.
  Novelty-tool note: `tools/experiment_history.py` coverage stops at the 2026-09-07
  wall/entry ledger and does not index WM-S training runs.
- **Immutable registration references:** code commit 6ee1d8a096f6; code tar sha256
  `af65104e15503894aed8c33ba61a5072191b7d091b1d7d57fb447057de7c5a2a`;
  merge_params sha256 `c23385f9e62b970aa21cd3b23b9ac17d0a4e1e5355fff57182b9a4bb11fc7bcf`;
  registration-file sha256 recorded in `_out/RUN2_READY.md` (computed after this
  file is frozen). Ledger append commit SHA = TBD (owner freeze).
- **Paths:** protocol/registration file `_out/REGISTRATION_WM-S-RUN2_DRAFT.md`;
  code `_out/code_run2_6ee1d8a096f6.tar`; chain `_out/run2_chain.sh`; config
  `_out/run2_train_config.json`; planned output
  `/root/autodl-tmp/wms_run2_20261006/work`.
