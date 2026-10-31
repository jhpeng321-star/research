# S-SUMOES-RUN log (variant (b), design 05b84910, cap 650,000 sim-s)

cpuQ root `~/sumoes20261031` (accounting.jsonl = durable precharge ledger; RUN_LOG.jsonl; state.json;
theta_after_uXX.json; screens/; jobs/<id>/result.json). Local copies: `tools/sumoes/evidence/`.
J600 = E3 team delay (cumulative_600 team_delay_vs, veh-s; lower better). delta = J(plus) - J(minus).

| update | screen kept (WM 60 s medians) | pair deltas (mean over 6 worlds) | accepted | panel mean before -> after | sigma | charged / actual | wall |
|---|---|---|---|---|---|---|---|
| baseline | - | canary s969536 CTRL0 == ours@0 (J600 9555.4909 both, hash 2dae9838) | - | 46,958.76 | sens 1.312 | 8,100 / 8,100 | ~60 s |
| 0 | [0,2,4] (-5.7, -21.5, -6.3; 75 s on 24 CPUs) | +1,246 / +4,748 / +3,182 | NO (all 8 panel worlds +1e-4..5e-4 worse) | 46,958.7624 -> 46,958.7626 | 0.0762 | 47,700 / 47,700 | screen 75 s + SUMO 161 s |

Observations after update 0 (for MAIN):
- The Adam step (lr 1e-3, first step = sign(g)*1e-3 per scalar, norm 0.012) is ~1/76 of the probe
  (sigma*||eps|| ~ 0.89): the candidate changes panel J600 by < 0.001 veh-s per world. With the frozen
  lr the 16 updates cannot move theta materially (max |dtheta_i| <= 0.016 vs sigma 0.076).
- Minus sides of all three kept directions give IDENTICAL J600 on 5/6 worlds (e.g. 11024.7, 95934.8,
  19403.9, 82657.7, 62731.1): one side of each pair is inert (positive residual clipped by the E3
  envelope/SUMO safe speed = the S-ZEROSTART dead zone), the other side brakes and costs +1.2k..+4.7k.
  The ES gradient therefore points into the inert region; WM screen sign agreed with SUMO (all
  kept medians < 0 = plus worse).
- Run PAUSED after update 0 (no mechanical failure); 602,300 sim-s remain under the cap.

## MAIN14:40 amendment implementation / fresh-session resume

- Read git log/status, HANDOFF, this log, and REGISTRATION14:25/14:40 first. cpuQ process
  inventory confirms no SUMOES jobs; state update_count=1, theta=0; ledger=47,700 actual/charged.
- A: replace Adam with sigma times the mean sign-weighted unit directions; panel unchanged.
- B audit correction: both old runtime paths were base-relative; SUMO helper's caprel name
  was misleading. Both now use the same capped-CTRL0-relative composition through
  tools/sumoes/caprel.py (SUMO live cap, WM frozen inferred cap). Native zero/no-effect
  writes are preserved. C1 SHA64f70af4 asserted; frozen runtime and wrappers unchanged.
- C: evaluate braking-side gains vs theta with paired futures on eight TRAIN starts;
  select braking orientation by negative-residual mass, positive median and q10 only.
- Screen no-step counts toward the four-consecutive-rejection stop (zero SUMO charge).
  Historical update0 counts. One registered rejection-based sigma halving after u9 only.
- CPU checks on cpuQ existing environment: 32 passed (1 existing torch warning), no SUMO.
  Windows default Python lacks torch; no environment was created.
- No additional SUMO dispatched at this checkpoint; all original evidence retained.

### Update 1 — screen rejection, no step
- MAIN1440 / commit beb08d64; screen136 WM rollouts, 24 nice19 CPU processes, 102.882 s.
- Kept []; braking-side gain medians by direction0..7 (veh-s, positive better):
  [-18.6241,-13.0176,-16.9719,-7.0724,-26.6924,-9.1613,-3.9606,-7.7089].
- Real SUMO pair differences: none (all screen-rejected). Accept/rollback: no-step,
  theta unchanged bitwise. Panel mean before=after=46,958.76242850634 (cached, not rescored).
- sigma=0.07621979740797888; update-driver wall=0.005689 s; sim-s increment0,
  cumulative charged=actual=47,700 / 650,000. Consecutive rejected opportunities=2 (u0,u1).
- Evidence: request_u01.json, screen_u01.json, u01_state.json; cpuQ driver3042666 exited.

### Update 2 — screen rejection, no step
- Screen kept[]; braking-side gain medians direction0..7, veh-s:
  [-10.77,-0.07,-20.23,-24.40,-1.24,-1.54,-2.29,-6.36] (rounded; exact evidence retained).
- 136 WM rollouts/24 nice19 CPU processes; screenwall102.079500s, driverwall0.004121s.
- No real-SUMO pairs; no-step, theta bitwise unchanged. Panel before=after46,958.76242850634
  (cached, not rescored); sigma0.07621979740797888. Increment0sim-s;
  cumulative47,700charged=actual/650,000. Rejection streak3(u0,u1,u2).
- Evidence: request_u02.json, screen_u02.json, u02_state.json; driver3050671 exited.
