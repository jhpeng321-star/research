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
