# A99 follow-on: matched 20 % panel, plan arms on MAPPO20C (D2800_p20). Prepared, NOT launched

Prepared 2026-11-14 about 06:00-06:20 Box 4 time. Zero SUMO (charged 0; no CHARGES.jsonl in /root/a99panel/root).

- Base `/root/a99panel` on Box 4; ship `/root/a99panel_ship` (laptop: `ship/`); queue `/root/a99panelq.sh` installed, NOT started.
- Read-only use: `/root/a99adm_s4` (tree + S4 bank), `/root/a99adm_hand` (hand bank), `/root/a96ref/root` (MAPPO20C binding),
  Box 3 `/root/a96pc/root/A96C_PANEL_C.json` (panel file). Nothing else touched. REGISTRATION.md not edited.

## Arms (worlds 984400-984591, paired)

| arm | construction | executor loaded (dry, runner check) |
|---|---|---|
| BANK_S4_20C (SECONDARY, disclosed: S4 not the registered pick) | A75_BANK on A99 S4 bank 682c01ae (2 plans: mut_4d472232, mut_a60055829), a37 specs -> `a76_d2p20.with_exe(comp rebind)` | 606e59e0 x 192 |
| BANK_HAND_20C | same builder, A99 hand bank 59acb747 (2 plans: wall_g_duo12_stagger40, wall_g_duo12; 3/10 admitted) | 606e59e0 x 192 |
| DUO_HOLD_20C | `a32_specs('DUO_HOLD')` + A67-style six-field replace | 606e59e0 x 192 |
| MAPPO20C (reference) | U0 specs -> `A52.with_executor(A77.read_bound(/root/a96ref/root,'MAPPO20C'))` | 606e59e0 x 192 |
| BANK_S5_20C | omitted: S5 admitted 0/12 on MAPPO20C, no bank file -> "empty" | - |

Executor verification: for every one of the 768 specs, `nm_eval_runner.bound_executor(ck)` (the runner's own pre-physics
check: binding seal, spec = binding, file sha of executor_path) returned 606e59e0; spec diff vs the same arm on U0 =
exactly the six executor fields; six fields = the A99 admission comparator (binding 466bfcbf). See `box_outputs/dry.out`.

## Mechanics gate (no extra worlds)
First 2 panel worlds (984400/984401) of each arm = 8 episodes, 9,600 sim-s, INSIDE the panel budget (same job ids,
the confirm stage runs the remaining 760). PASS iff executor 606e59e0 on every record; bank arms pass the A96 per-plan
rule checks (plan_checks, zone +-1 m, a75 d1af7e0f, no CAV in two sessions, no refusal, 0 collisions/teleports);
DUO carries its a32 receipt. Deviation from A96(c): no U0 identity on stored A76/A32 records (not on Box 4).

## Budget
4 x 192 x 1,200 = 921,600 sim-s (230,400 per arm), cap enforced in code; expected about 1 h at 46 workers.

## Fixes carried
- nm_finaleval cache fix fa98b552 (tree = A99 admission tree, file-by-file verified copy).
- int-key seal bug: overlap_stats / engaged_hist re-wrapped to string keys; own analyse_core with string keys;
  unit test seals a 13-CAV histogram, JSON round trip, verify OK.

## Shas
a99_panel.py 84afaa26, a96_p20.py b29c45d4 (unchanged), test_a99_panel.py b8b82505, patch_runner_a99p.py f738c7e2,
run_a99p.sh b0d696a4, setup_a99p.sh a6bb838c, a99panelq.sh 9bb020bb, runner 47afb409 (= 0c5c2a4b + A96C_RANGES row),
env_a99p.sh 5d436bdf, A99P_BANKS.json aa509561 (receipt 78bfdbb5), panel file 0bb84f3c / panel 6c161a56.

## Launch (coordinator only)
`ssh -i ~/.ssh/rented_box ubuntu@49.235.140.169 'sudo -n bash -c "setsid -f nohup bash /root/a99panelq.sh < /dev/null > /root/a99panelq.nohup 2>&1"' < /dev/null`
Log `/root/a99panelq.log` (ALLOCATION START line; last line A99PANEL_DONE). Refuses a second launch (`/root/a99panelq.launched`).
Results `/root/a99panel/root/A99P_RESULTS_D2800_p20.json`, mech `/root/a99panel/root/A99P_MECH.json`.
