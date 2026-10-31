# S-SUMOES — TRACK 3 registered design and MAIN14:40 amendment
Worktree `E:/agent-work/c4-sumoes` (`c4/sumoes` from `c4/pilotdiag@0aefe6e0`); commits per step; no push.
Implementation: `tools/sumoes/`; shipped TOML remains default OFF. Runtime design JSON
`05b84910` is preserved as the historical allocation binding; this amendment supersedes
its Adam step and the original screen rule, without changing worlds, panel, or cap.

## MAIN14:40 implementation (before update 1; 47,700 sim-s already charged)
- A: `theta_candidate = theta + sigma * mean(-sign(Jplus-Jminus) * d/||d||)` over
  WM-kept directions, with real SUMO differences averaged over six worlds. Equal
  fitness casts a zero vote. Do not normalize the mean again. Adam moments are historical
  resume data only. The eight-world acceptance/bitwise rollback remains unchanged.
- B: E3 C1 checkpoint SHA `64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c`
  is asserted in both paths. Audit correction: update-0 SUMO `caprel_commands` actually
  used `native_base + residual`, just like the WM, despite its cap-relative name.
  Updates 1+ use a shared adapter: `CTRL0=max(-v/dt,min(native_base,mode27_cap))`;
  `issued=clip(CTRL0+r,min(-3,CTRL0),max(min(1.2,getAccel),CTRL0))`; executed command
  is capped at mode27 and nonnegative speed. SUMO obtains the cap from live API queries;
  WM uses the frozen geometry-based inferred cap/base. Exactly zero (or no effect)
  preserves the original SUMO native-base write, hence CTRL0 bitwise identity.
- C: On the eight existing TRAIN starts, roll out theta and both perturbation sides
  with common futures. Orient each direction by the larger total negative-residual
  mass, without looking at cost; gain is `J60(theta)-J60(braking side)`. Keep at most
  three with median>0 and q10>0, ranked by median. All costs and braking masses are
  retained. Accelerating-side gains cannot qualify a direction. 136 WM rollouts/update.
- Count screen-all-reject no-step updates as rejected opportunities for the four-consecutive
  rejection stop, including historical update 0. Such an update consumes zero SUMO.
  If reached, the >2/3 rejection check after updates 0–9 permits exactly one sigma-target
  halving. No retry or additional canary is dispatched; the zero-residual write is unchanged.
- Allocation and original guard/exposure receipts remain valid: all worlds are in the same
  registered 104-seed block; no extra worlds or SUMO seconds are added by this correction.

## AMENDMENT 2026-10-31 14:4x — owner-approved variant (b) (REGISTRATION 14:25), supersedes sections 2/6 numbers
- W = **6** fresh gradient worlds per update, panel = **8** worlds; P=3 antithetic pairs, D=8 screened.
- Episodes/update = 2*3*6 + 8 = **44**; planned 8 (CTRL0 panel baseline; canary = s969536 CTRL0,
  reused as baseline[0]) + 16 x 44 = **712 episodes = 640,800 sim-s; HARD CAP 650,000** (9,200 reserve).
- Seeds: fresh C4_LOOP_TRAIN sub-block **969440-969543** (104). Update u gradient worlds
  969440+6u..+5, world j -> cell CELL_CYCLE[(2j + u%2) % 12] (every update spans all 4 penetrations
  and 3 demands); panel 969536-969543 = (p, d) for p in .05/.1/.2/.4 for d in 1900/2700.
- Frozen design JSON `tools/sumoes/design.json` sha256
  **05b84910472db97619147b12df3bc2a28f82125ec2080ae333deb252d80105c7** (cpuQ ~/sumoes20261031/design.json).
- Guards PASS on the frozen cpuQ runtime (require_development_case 'exposed' + c4 check_case, 104/104;
  seeds.py 0c052561, c4/seeds.py 86b396cc); exposure audit: REGISTRATION grep only the 14:25 block
  bounds, no ledger/doc hits; cpuQ/gpuR/gpuA filename scans clean.
- Policy on the E3 runtime: candidate-B cap-relative residual realised on the frozen E3 C1 executor
  (policy 64f70af4): r = 1.2*tanh(head(f)), head = theta (137), applied every physics tick without a
  zero-release branch: clip(native + r, [min(native,-3), max(native, min(1.2, getAccel))]); theta=0 is
  bitwise the CTRL0 forced-native path (zero.py 24095a10).

## 1. Parameterisation
- ES perturbs EXACTLY the S-ESLONG head-only scope: candidate-B `residual_head`
  weight+bias = **137 scalars** (Linear(136,1); hidden_dim 64). Trunk/encoders frozen.
- Zero start (S-ZEROSTART): theta_0 = 0 executes bitwise CTRL0 under the CAPREL law.
- sigma rule (ESLONG command-scale calibration): sigma = 0.10 m/s^2 target residual RMS /
  deployed-t0 head sensitivity RMS (radius*sech^2(z)*sqrt(||f||^2+1), measured per update);
  halve every 4 ACCEPTED updates; floor on perturbed command RMS 0.02 m/s^2.
- Step: MAIN14:40 sign-weighted mean of unit directions, as specified above.

## 2. Per-update SUMO workload (real SUMO on TRAIN worlds)
- Fresh registered C4_LOOP_TRAIN sub-block **969440-969543** (104 seeds; 12-cell
  p.05/.1/.2/.4 x d1900/2400/2700 cycle, position mod 12). Registration + exposure audit
  + `traffic_shepherd.c4.seeds` guard re-check BEFORE any allocation; never 969400-969439
  (WMDATA), never 964xxx/962xxx/977xxx (config refuses).
- Per update u: draw D=8 private-generator directions; screen -> keep P=3; evaluate 3
  antithetic pairs on W=6 fresh gradient worlds (969440+6u..+5), 300 s native warmup +
  600 s control, charged 900/episode. CRN = identical world seed within a pair (and across
  pairs); `initial_state_sha256` equality asserted per pair; fitness J600 = E3 team delay
  (road deficit + source waiting, veh-s, deploy observer cumulative_600; lower better).
- Held-in panel: 8 fixed worlds 969536-969543, rescored after every proposed step.
- Episodes/update at three kept directions = 2*3*6 gradient + 8 panel = **44** (= 39,600 sim-s).

## 3. WM role (guidance, never the fitness)
- Screen all D directions on the frozen WM (WM2 `b8797242`+learned mix) BEFORE SUMO:
  paired 60 s rollouts from 8 frozen TRAIN starts (WMDATA views); keep directions with
  confident predicted gain (median > 0 AND q10 > 0; 90 s descriptive); rank by median,
  keep top 3. Screen rejects cost zero SUMO; an all-reject screen = no-step update (0 SUMO).
- Optional warm start (registered variant only): theta_0 = candB_pilot_rel2 head at a frozen
  SHA if rel2 shows acceptance-surviving updates; default zero start. The 600 s fitness and
  all accept/reject decisions are REAL SUMO; WM never scores an update.

## 4. Acceptance / rollback
- Paired CRN rescore of the held-in panel before/after each step (S-PILOTDIAG option (b)):
  accept iff mean panel J600 does not worsen (no behavioural thresholds); on reject restore
  actor+Adam bitwise; the update is consumed (budget charged) and logged.
- Pre-registered adaptation: if >2/3 of updates 0-9 reject, HALT, halve sigma target once,
  resume (single allowed adaptation). Safety gate (collisions/teleports/dropped=0, lane
  audit 0) on every episode: mechanical failure stops dispatch; no retry authority.

## 5. Evaluation — unchanged (S-EVAL-A, already approved)
- Frozen S-EVAL-A: 24 frozen fresh worlds x {native, CTRL0, speed-only MAPPO, ours-B-ES}
  x 900 s; 86,400 sim-s separate hard cap; 2000-draw world-cluster percentile bootstrap,
  95% CIs, paired primary estimands ours-MAPPO / ours-CTRL0 / ours-native. No TRAIN-world
  reuse, no optional stopping, no new seeds (9773xx untouched by this task).

## 6. EXACT budget
- Planned: 9 (CTRL0/ours-zero canary and remaining seven panel baseline episodes) + 16 x 44
  = **713 episodes = 641,700 sim-s**. Hard cap **650,000 sim-s** (8,300 reserve: registered
  amendments only; charge-before-dispatch; no refunds/retries; actual-0 infra failures need
  a registered canary rule). 16 updates max.
- Wall (WMDATA-measured rates): cpuQ 32 cores, 20 workers nice 19 -> ~36-60 s/900 s
  episode/worker: 22-episode wave ~1-2 min; total SUMO ~20-35 min. WM screen 128 x 60 s
  rollouts ~10-25 min serial on gpuA-host CPUs (~2-5 min across cpuQ's ~16 spare cores or a
  gpuA slice) -> per update ~5-25 min, **end-to-end ~1.5-6 h**; fits before Nov 6 with a
  full-restart slack. Driver/receipts: `SUMOES_RECEIPTS.json` per run + JSONL per update.

## 7. What the paper can claim
- If S-EVAL-A confirms: "a 137-parameter per-CAV residual policy, started at zero (=CTRL0)
  and trained by antithetic ES whose fitness is real-SUMO 600 s team delay on fresh TRAIN
  worlds, with WM 60 s direction screening and a held-in paired acceptance panel, beats
  speed-only MAPPO (and CTRL0) on frozen fresh evaluation worlds" (CIs as frozen in 5).
- Honest limits: screened ES is intentionally biased toward WM-confident directions (no
  unbiasedness claim); no WM >90 s fidelity claim (PILOTDIAG); per-update estimates are
  noisy (no per-update significance claims); accel-only (no lane control); safety reported
  descriptively. A null result is reportable as a bounded real-fitness search outcome.

## Prerequisites before ANY SUMO second (owner approval of this design + registration)
1. Register block 969440-969543 (completed: exposure audit and 104/104 seed guards PASS).
2. Stage byte-identical S-WMDATA/S-SPEEDHEAD wrappers on the frozen cpuQ E3 runtime
   (`b3c031ad`; zero `24095a10`, observer `b22af462`, baselines `c63b14bf`); ACK canary.
3. Flip `[sumoes] enabled = true` in a fresh registered config only; implement the real
   `SumoRuntimeEvaluator` (stub interface) on that runtime; interactive ACKs per job.
