# S-LAUNCHA — candidate-A qualification + production launch (2026-10-29; fresh-session RESUME of the frozen 10:52 session, its work kept)

Runtime: branch `c4/candA-int`, staged gpuA `/root/autodl-tmp/c4/candA_f176d7ac9a28`
(SOURCE.json commit `f176d7ac89dc13886e7c1e0ba4a7ef2814e9eb14`; mix commit `5fddc515`; base
`041cee34` S-INTEGRATE2 tree). WM = incumbent WM2 `b8797242…` (config `f95f7c4c…`,
ACCEPT c02, measured gate `be0297ab…`, CCT-1 informational). Actor = C1 `64f70af4…`
resumed `--new-phase` with the 18 registered `--resume-allow` keys (receipt in the run log).

## STEP 1 — CCT-2 for WM2 + learned newcomer mix (registered K1 contract) — PASS, mix ON

- Panels scored on gpuA (cap 0.30, W6CAP PID 597833 untouched): `cct2mix20261029/`
  = W5_COMBINED/code `score.evaluate` → `e3_score.rollout_dj` (gpb 4, steps 600, seed
  140014, samples 1, entrant_control true, DESIGN controllers) + the ERRSRC `LearnedMix`
  overlay via the `entrant_driver_sampler` rollout hook (`rt_mix` = W5 rt + c4/errsrc hook,
  fail-closed patch; mix key rule `errsrc-mix:{seed}:{group}:{entrance}:{ordinal}`,
  seed = 140014+7919*group). Deviation note: run 1 (11:40–13:52) passed explicit
  `indexes=range(100)` for pairs300, overriding the registered 48-group `_macro_indexes`
  (source has 100 groups; the panel is the registered 48); the assertion fired AFTER
  scoring and the output was discarded; run 2 (14:07–14:59) rescored pairs300 with the
  registered indexes; the validation panel (36 groups = the full split) from run 1 kept
  byte-identically. No-mix parity receipt `PARITY_NOMIX.json`: WM3 panel bit-exact
  168/168 (max diff 0.0); WM2 1 value at 1.2e-4 (cross-run float noise).
- CCT-2 (registered `tools/gated300/cct2.py`, pooled pairs300+validation, 02:25 rule):
  h=180 |G|=26 agreement 0.9615 PASS; h=300 |G|=31 agreement 0.9677 PASS; 90/120 FAIL
  (|G|<20). **H_G = 300 — the mix version PASSES.**
- 08:15 rule branch **(a)**: mix ON + gain-only long credit to H_long = H_G = 300.
  Mix-bound authority `authority/CCT2_b8797242…_mix.json` (sha256 `f42cce0f…`,
  `learned_mix_qualified=true`, panel hashes + provenance inside; raw doc retained).

## STEP 2 — authority placeholders filled; verify PASS

- WM2 sha `b8797242…` + config `f95f7c4c…` (registered gpuA copy; the staged-tree copy
  is identical modulo CRLF); mix state ON with hash-bound errsrc module
  `tools/errsrc/inference.py` (sha `e996d18e…`, byte-identical to c4/errsrc `b1569746`);
  CCT-2 file = the mix-qualified `CCT2_b8797242…_mix.json` (H_G 300);
  `configs/c4_policy14_candA.toml` `long_credit.cct2_file` → `authority/CCT2_{wm_sha256}_mix.json`
  so the trainer's `gated300._install_gain_only` consumes exactly that file; long credit
  `gain_only`, H_long = 300; parity receipt `alwayscmd3_execution_parity.json`
  (pooled 0.146538 ≤ 0.160312, leader rule ON) validated through `execution.verify_parity`;
  leader-rule slot bound (commit a6a1894d + 4 source hashes).
- `CANDA_AUTHORITY.json` (gpuA `arm/loop/c4prep/`) build + `authority.py verify` PASS:
  source `f176d7ac…`, WM2 `b8797242…`, H_G 300, leader_rule `bound-with-receipt`.
- auditfix BOUNDARY rebuilt in the new staged dir (`BOUNDARY_WM2.json`,
  CCT-1 `1ae6b101…` identical to the 041cee34 derivation).

## STEP 3 — exact 2-update GPU qualification (COMMANDS.md Step B) — <RESULT>

- Attempts 1–3 (fast failures, all receipts kept on gpuA): pre-created output dir
  (fresh-dir refusal); `--new-phase` resume guard requiring the 18 registered
  `--resume-allow` keys; no-op `--real-credit` refused by the auditfix guard
  (removed — registered credit unchanged, as ASTRA #5 advised).
- <MEASURED: updates, losses, module grads, gain-only rows, s/update, memory, projection>

## STEP 4 — GO conditions + launch — <RESULT>

ASTRA4 BLOCKER closure notes:
1. Execution parity: closed — mode-2 always-commanded receipt
   `tools/execparity/alwayscmd3_execution_parity.json` PASS (pooled executed-accel RMSE
   0.146538 ≤ native-head band 0.160312; residual-0 == CTRL0 bit-identical; leader rule
   ON), bound + revalidated by `authority.py verify`.
2. Integrated launch runtime: closed — `c4/candA-int` 041cee34 (S-INTEGRATE/S-INTEGRATE2
   merge of auditfix/polspeed/polspeed2/gated300/cct2/exploit/execparity2-3) + S-LAUNCHA
   commits 5fddc515/f176d7ac; immutable staged tree with SOURCE.json; hash-bound
   manifest/config-chain/actor/WM/CCT-2 in the verified authority.
3. WM/horizon authority: closed for WM2 — auditfix `BOUNDARY_WM2.json` + measured gate
   `be0297ab…` + CCT-1 informational + mix-qualified CCT-2 `f42cce0f…` H_G 300; the
   learned mix is qualified ON WM2 only (recorded as such; ERRSRC panels/keys bound).
4. Speed/memory qualification: closed by MAIN 06:00 (i) nine-row exact N4 path (no
   six-row tolerance) + this task's two-update qualification at the launch horizon under
   the final recipe (STEP 3), memory within cap 0.45, scoring-inclusive ≤ 24 h projection.

- Integration tests: 181 passed on the staged runtime (authority_gen/INTEGRATION_TESTS.log;
  candA_integration 11, execparity 69, gated300/cct1/cct2/exploit/auditfix×8/polspeed2_report/
  antiexploit/polspeed2_parallel 101).
- Production launch: <PID/start token/config hash/updates verified>

## Deviations / notes

- The `--resume-allow` key list and removal of `--real-credit` are the only departures
  from the literal COMMANDS.md Step B command; both are forced by the tool's own
  fail-closed guards and were anticipated by ASTRA #5 / POLAUDIT-T #1.
- W6CAP training (PID 597833) ended on schedule (train-until 15:55) before STEP 3; never
  touched. The WM2+mix CCT-2 result landed 14:59, inside the 13:50 amendment time-box.
- GPU: STEP 3/4 run at the registered cap 0.45 on the now-idle card.

## S-ZEROSTART — combined config READY (2026-10-30 ~01:50; qualification/production NOT yet run)

Runtime: `c4/candA-int` 17028d5c, staged gpuA `/root/autodl-tmp/c4/candA_184542ab_zs`
(SOURCE.json commit 17028d5c; = candA_184542ab_p1 [c31d6d70 + nanfix 184542ab] + zero-start +
amendments). Authority `arm/loop/c4prep/CANDA_AUTHORITY_ZS_17028d5c.json`
(sha256 7e3ef92d...; build + verify PASS; `dense_90s_phase1`; binds zerostart.py and the
`policy14.zero_residual_output` flag; config `configs/c4_policy14_candA.toml` sha256 6dd3048e...).
Everything of the stopped run (WM2 b8797242 + learned mix, always-commanded mode-2 law, leader rule
ON, nanfix, monitors, nine-row-N4 memory recipe, segment 2, cap 0.45, seed 965140, 150 updates,
`--resume-allow` list) is unchanged. Changes (all in `configs/c4_policy14_candA.toml` + 3 small code files):

1. zero-start: after the C1 64f70af4 load, `residual_head` OUTPUT layer weight AND bias = 0 and the
   head's Adam moments cleared (`src/wms/control/zerostart.py`, `[policy14] zero_residual_output`).
   Initial residual is exactly 0 for every CAV (also under multiplicative exploration noise) and the
   executed accel is the shared CTRL0 law `max(min(base, cap), -v/dt)`.
2. exploration: rho 0.10 -> 0.50 over the phase, `radius_m = 100` (was the 150 m group radius).
   NOTE: the curriculum position is the GLOBAL update_count (cf_curriculum_position 180 at the first
   update), so the registered phase-relative knots [0,150] are written `[180, 330]`; literal [0,150]
   would have been rho 0.5 flat (= the stopped run, whose logged exploration_rho was already 0.5).
   Persistence curriculum unchanged. Exploration is multiplicative on the actor latent
   (`latent*(1+eps)`), so it is inert while the head output is exactly 0.
3. cooperation group = random size s ~ U{2..6} per start (`counterfactual.group_size_sampling =
   "uniform"`, `group_max_size = 6`; window of min(component, s)); n_group stays 1.
4. credit fix: `n_flip = 0`, `w_team = 0` (no flip rows; no negative sub-team coefficients).
   Objective: `1.0*D_all + 9.9*D_G + 8.9*C_G + 8.9*sum_i D_i` with `C_G = D_G - sum_{i in G} D_i`
   (rows: all 1.0; group 9.9+8.9 = 18.8; member individual rows 8.9-8.9 = 0; non-member
   individual rows 8.9; base row 0) - every coefficient >= 0; ~3 fewer rows per update.
5. long-credit slot: see "Long-credit slot" below.
No new `--resume-allow` key is needed: counterfactual / exploration / policy14 sections are not part
of the checkpoint-vs-TOML trainer diff (`phase_resume.resume_diff` covers DreamerConfig + reward only).

Scorer sanity (gpuA, scan tool, 4 gate-panel starts, no training): the rescored J is bitwise equal for
C1 / update-190 / const +0.5 / zero residual (mean 486.5207) = the dead zone; const -0.5 -> 497.216,
const -0.05 -> 487.103, const -0.01 -> 486.588 (the scorer resolves a -0.01 m/s^2 braking residual).

### Exact commands (gpuA)
```sh
# 2-update qualification (output candA_dry_184542ab_zs; ~12 min at ~350 s/update; needs ~39 GB: stop W8 first,
# ONE SIGTERM to the W8 PID after `ps -o cmd= -p PID` shows .../W8_JAM/code/long_train.py, wait for exit)
cd /root/autodl-tmp/c4/candA_184542ab_zs
nohup setsid ./stepZ.sh dry > /root/autodl-tmp/c4/candA_dry_184542ab_zs.log 2>&1 < /dev/null &
# PASS = 2 finite updates; summary.zero_start receipt; rescored J (rescores.jsonl update 181 vs 182 and
# independent_score_before) differs; peak mem < cap; ~300 s/update. Then production (150 updates):
nohup setsid ./stepZ.sh prod > /root/autodl-tmp/c4/candA_prod_184542ab_zs.log 2>&1 < /dev/null &
```
`stepZ.sh` = the stepB/prod command of candA_prod_184542ab_p1 (same env, fractions .10/.04/.022 x4
workers, FGA flags, `C4_SUMO_LEADER_RULE=1`) with output dir switched by the `dry|prod` argument.
Re-run `tools/candA/authority.py build ... --dense-phase1` + `verify` if any bound file changes.

### Exact shipped cooperation-credit objective (answer to MAIN 02:0x check; test `test_candA_objective_identity_synergy_decomposition_and_registered_comparison`)

Per start, D = (row return) - (base-row return); G = the sampled group (size ~ U{2..6}), C_G the synergy:

    OBJ = 1.0*D_all + 9.9*D_G + 8.9*C_G + 8.9*sum_{all CAV i with an individual row} D_i ,   C_G = D_G - sum_{i in G} D_i
        = 1.0*D_all + 18.8*D_G + 0*sum_{i in G} D_i + 8.9*sum_{i not in G} D_i          (flip rows: none)

Coefficient table (row return coefficient, per start): all-CAV row 1.0; group row 9.9 + 8.9 = 18.8;
each group member's individual row 8.9 - 8.9 = 0; each non-member individual row 8.9; base row 0 (baseline);
flip rows none. All >= 0. Registered (retired) objective for the same weights added the TEAM term
`10.3 * sum_i M_i`, `M_i = D_G - D_{G\i}` (|G| >= 3): it contributed +30.9 on D_G (3 flips x 10.3, so the
registered D_G coefficient was 18.8 + 30.9 = 49.7) AND -10.3 on each flip row (the negative sub-team rows).
The shipped config removes the whole team term (both parts), because they are one term: M_i cannot keep its
D_G part without its D_{G\i} part. Consequence/honest note: with the registered anti-gaming guard
(w_coop <= w_ind) the member solo effects cancel exactly (8.9 - 8.9 = 0), so in row-coefficient form the
synergy credit IS "18.8 x group effect, 0 x member solo effects" - the subtraction of member effects is
already present in `8.9*C_G` and is cancelled by the registered `w_ind = 8.9` solo reward, not dropped.
A reward on `D_G - sum D_i` alone (w_ind < w_coop) would put NEGATIVE coefficients on member individual rows
(rewarding worse solo behaviour) and is refused by the guard / `row_coefficients`. If a larger group credit
is wanted instead (D_G 49.7), or `w_group` 9.9 -> 0 so only the synergy and the individual solo rewards
remain (D_G 8.9), say so: both are one-line config changes (`w_group`, `w_team`).

### Update 02:2x — long-credit variant staged (owner: combined run includes gain-only 300 s; 24 h condition dropped)

Runtime SOURCE commit now `3ea624ff` (tools/config only since 17028d5c; no bound source file changed).
Authorities (gpuA `arm/loop/c4prep/`, both build + verify PASS): dense `CANDA_AUTHORITY_ZS_3ea624ff.json`
(sha256 769e883a...), long-credit `CANDA_AUTHORITY_ZS_LONG_3ea624ff.json` (sha256 14810bb5...; built with
`--entry-config configs/c4_policy14_candA_long300.toml`, no `--dense-phase1`; mode gain_only, H_G 300).
`configs/c4_policy14_candA_long300.toml` (sha256 82865823...) = candA.toml + `[long_credit] enabled = true`
(mix-qualified CCT-2, w_long 1.0). `authority.py` now binds the whole merged chain of the entry config.
The 300 s MEMORY recipe is the only open slot: S-LONGMEM300 must supply `tools/candA/longmem300.env`
(exported WMS_SPEED_* fractions / budget cap / any trainer override) - `stepZ.sh drylong|prodlong` refuses
to run without it. If LONGMEM300 changes a bound file (e.g. parallel_futures.py), merge it, restage the file,
update SOURCE.json and re-run both `authority.py build` + `verify` (the commands are in this section's history).
```sh
cd /root/autodl-tmp/c4/candA_184542ab_zs     # after: tools/candA/longmem300.env present, W8 SIGTERMed and exited
nohup setsid ./stepZ.sh drylong  > /root/autodl-tmp/c4/candA_dry_184542ab_zs_long.log  2>&1 < /dev/null &   # 2-update qualification
nohup setsid ./stepZ.sh prodlong > /root/autodl-tmp/c4/candA_prod_184542ab_zs_long.log 2>&1 < /dev/null &   # production, 150 updates
```
Dense-90 s fallback (only if no 300 s recipe fits): `./stepZ.sh dry` / `./stepZ.sh prod` as above (authority `CANDA_AUTHORITY_ZS_3ea624ff.json`).
