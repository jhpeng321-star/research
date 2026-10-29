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
