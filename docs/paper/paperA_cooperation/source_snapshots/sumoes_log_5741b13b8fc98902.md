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

### Update 3 — screen rejection; four-rejection early stop
- Screen kept[]; braking-side gain medians direction0..7, veh-s:
  [-21.82,-5.69,-6.34,-4.63,-4.12,-7.00,-4.42,-26.03] (rounded; exact evidence retained).
- 136 WM rollouts/24 nice19 CPU processes; screenwall101.419477s, driverwall0.004179s.
- No real-SUMO pairs; no-step, theta bitwise unchanged. Panel before=after46,958.76242850634
  (cached, not rescored); sigma0.07621979740797888. Increment0sim-s;
  cumulative47,700charged=actual/650,000. Rejection streak4(u0,u1,u2,u3).
- Registered scientific stop confirmed by run.py before update4: four_consecutive_rejected_updates_including_screen_no_step.
  No update4 request, WM screen, or SUMO dispatch. No rejection-based sigma halving (u9 not reached).
- Evidence: request_u03.json, screen_u03.json, u03_state.json, MAIN1440_RUN_LOG.jsonl;
  update3 driver3056777 exited. No mechanical failure or owned processes to signal.

## Final reconciliation / S-EVAL-A handoff
- FINAL_RECONCILIATION.json verifies53unique charged jobs,53completed results,
  14distinct SUMO worlds, actual=charged47,700<=650,000, all safety/lane counters0,
  all repeated-world initial hashes identical; no live task Python jobs.
- Final head is137zeros (CTRL0-equivalent), canonical headSHA
  226f2fad46e9e695b1fc277750a1bfdf5c403f17b38984efde3a16773e5fc255.
  Artifact FINAL_HEAD.json; fileSHA99982b566a7214e6a70e732052e4b9e70acb7523abddc24cf792e6756269502a.
- Report docs/paper_results/SUMOES_RUN.md; no accepted updates or learned gain;
   corrected-law nonzero SUMO path was not dispatched because all screens rejected.

## Track 3B — MAIN15:45 MAPPO-base switch

- Historical CTRL0 results above are retained. Remote preflight:53 completed jobs,
 47,700charged=actual, no SUMOES processes, no STOP. MAPPO export d9042cc5 verified.
- Amendment design_3b.json SHA9851cc62086086f8fa1796f5233fb4321425a63b385717d599126a722c740ead.
 Seven new CPU tests PASS; WM zero-head direct-MAPPO bridge bitwise4ticks PASS;
 signed residual WM outputs finite;32 original regressions PASS; Ruff PASS.
- Implementation receipt MAIN1545_IMPLEMENTATION.json preserves exact source hashes.
 No new SUMO at this checkpoint. Original remote source retained in PRE3B_SOURCE.tar.
- Zero head uses byte-identical original MAPPO arm wrapper/CLI; C1 features unchanged.
 WM top2 ranking never vetoes. Normalized sigma step; strict mean improvement;
 four rejected REAL updates stops this changed-base run. Original shared650k ledger.
- Residual sign statistics refer to bounded composed action minus MAPPO action before
 unchanged physics speedMode27 cap, not a counterfactual physical-acceleration effect.

### Track3B canary — PASS, followed by operational stop

- Source commit a49d980f; cpuQ driver3107770 exited. Two900s episodes on969440:
 plain MAPPO J600=10157.678294306275; zero-head3B J600=10157.678294306275,
 bitwise equal; initial SHA both bb3ce9aa2eb64143a5d6e599abd2651e984f578eb18bf83e64e2b9045572698a.
- Collisions/teleports/dropped/lane counters0; plain wall25.792354s,3B34.550413s.
 3B acting1604/1617=0.9919604205318491; executed-command residual signs
 negative0/positive0/zero1604; absolute residual sum0. Canary head sensitivity1.2749305568.
- Canary was saved durably to state_3b.json before driver failed at
 `log('canary_done', **receipt)`: receipt includes `kind`, colliding with log's
 positional `kind` argument (TypeError). Both SUMO workers completed successfully;
 this is a driver logging failure, not a scientific rejection or failed episode.
- Mechanical-failure stop honored: no baseline, WM screen, or update dispatched.
 Total55completed/precharged episodes; charged=actual49,500/650,000;
 increment1,800; remaining600,500. Evidence3B_CANARY_STATE.json retains full receipt.
- Additional CPU tie-panel test passes (8new tests total), verifying a nonzero
 proposed step on an equal panel is rejected with bitwise theta rollback.

### Track3B post-stop repair — CPU only

- Separate log event name from receipt.kind; scientific receipt forwarding now
 succeeds. Exact regression reproduces the failing canary log call and verifies
 both fields survive. Nine new CPU tests PASS on cpuQ; Ruff PASS.
- Repaired run_3b.py staged on cpuQ. Original faulty source retained in3b_step1.tar;
 original failure log unchanged. Shared STOP marker installed, no live task workers
 or drivers at reconciliation, no process signals, no further SUMO dispatch.

### Track3B final reconciliation

- FINAL_RECONCILIATION_3B.json verifies55unique charges/results,49,500charged=actual,
 cap650,000/remaining600,500; all safety/lane counters0 and repeated-world hashes equal.
- Final head137zeros, canonicalSHA226f2fad46e9e695b1fc277750a1bfdf5c403f17b38984efde3a16773e5fc255.
 FINAL_HEAD_3B.json binds MAPPOd9042cc5/C1trunk64f70af4; MAPPO-equivalent,
 not an eligible learned candidate. FileSHA0e54f112cbacb6b558879c51c1605df5ca0065a26728993b839b6c722122c442.
- No3Bpanelbaseline/learningcurve; updates0,accepted0. Report SUMOES_RUN.md now
 includes latest3B stopped result followed by retained CTRL0 historical report.
- Mechanical STOP retained despite tested repair. Owner continuation required;
 future authorized next step is baseline, not a repeated canary. No evaluation launched.

### Track3B MAIN15:54 authorized continuation — pre-dispatch checkpoint

- Fresh session read clean git status/log, HANDOFF/RUN, registration14:25/14:40,
  Track3B15:45 and explicit15:54 mechanical-stop lift. No SUMOES processes on cpuQ.
- Remote durable ledger55jobs/49,500sim-s matches saved canary state; updates0,
  panel empty. Repaired run_3b.py SHA8946bbe5d5982b9905f243c7ded98089d708447ffc840e3c8227a2428490097a
  matches tested repair. Next step eight-world baseline, without repeating canary.
- STOP and old stopped-result final artifacts will be renamed into retained history;
  all original charges remain. Budget600,500remaining, hard cap650,000.
- Two initial SSH inspection commands had quoting-only Python syntax errors before
  any task file access or dispatch; corrected stdin transport succeeded.
- Scope disclosure: exact-file Grep again returned parent-tree matches (same tool
  behavior disclosed historically). Only named registration entries used; subsequent
  registration reads/edits use exact-path Read/apply_patch. No new seed allocation.

### Track3B MAPPO acceptance baseline — complete

- MAIN1554 continuation; driver3155472 exited. Own STOP and old remote FINAL_HEAD_3B
  renamed RETAINED_PRE1554_*; old local reconciliation retained. First preservation
  preflight stopped harmlessly because reconciliation existed locally only; no dispatch.
- Seeds969536..969543 J600 in order:
  [15076.878557223372,62725.25051304699,28024.81257199762,73863.5866986485,
   10286.284310893401,69219.34456078897,10501.730779864547,102265.46942332505].
- Mean46,495.41967697356veh-s;8x900=7,200increment; cumulative56,700charged=actual,
  63completed jobs;593,300remaining. All initial hashes match prior same-world hashes,
  all safety/lane counters0. Acting71,540/72,126=0.9918753293; residual signs
  negative0/positive0/zero71,540. Head remains137zeros, MAPPO-equivalent.
- Sensitivity1.3119926923847935; sigma0.07621993672711026. Wall41.157s
  (launch to durable state save); max episode38.095s.
- Evidence3B_BASELINE_MAIN1554_STATE.json. Operational continue_3b.py runs exactly
  one request/screen/update per exclusive launch, writes STOP on subprocess failure,
  checks ledger and registered stop before dispatch. Source RuffPASS locally;
  remote Python import/status PASS (remote ruff module unavailable). Scientific
  run_3b.py and wm_screen.py unchanged. Operator commits each completed update.

### Track3B update0 — in flight / resource yield (not a final result)

- Baseline checkpointdc537ba6; supervisor3162394, nice19; no duplicate dispatch.
- WM136rollouts137.474140716s, kept[3,5], predicted gains[.2671127319,.2130894661].
- All24real perturbation episodes completed and passed driver safety/CRN gates.
  Eight candidate-panel jobs precharged, awaiting original load<=30 admission;
  cpuQ32CPUs, observed peer-contended load~31. No priority/resource rule changed.
- Durable charge85,500=56,700+24x900+8x900; saved state remains56,700 until
  update completion. No failure, retry, rejection or panel outcome yet.
- checkpoint_3b.py (RuffPASS) reconciles only ledger-named result.json receipts,
  verifies every seed/hash/safety/lane/actual charge, and writes exclusive checkpoint
  evidence after all task processes exit. No SUMO or scientific-code changes.
- Mirrored MAIN1554_u00_INFLIGHT.json preserves the24completed raw job receipts,
  pending8IDs, screen, launch PID and prior state during the resource wait.
  Cumulative actual78,300 (87completed), charge85,500 (95jobs); pending7,200 retained.
  Pair mean Jplus-Jminus: direction3=-1516.8667999108632,
  direction5=+486.1790572007497veh-s. These are gradient observations, not panel gains.

### Track3B update0 — REJECTED, bitwise rollback

- WM kept[3,5],136rollouts137.474140716s. Six-world mean Jplus-Jminus
  [-1516.8667999108632,+486.1790572007497]veh-s; full pair-world J600 in
  MAIN1554_u00.json. Sigma.07621993672711026; normalized step norm.055460042574.
- Panel before/MAPPO-base mean46,495.41967697356; candidate48,097.82292018744,
  delta+1,602.40324321388veh-s -> reject, zero head bitwise retained; real rejection streak1.
- Candidate panel969536..969543 J600:
  [15077.384513873152,62725.649217575796,28042.438198561646,80148.17698394971,
   12452.236635283813,69091.88613145243,14061.116554238555,103183.69512656449].
- Collisions/teleports/dropped/lane0; all CRN hashes equal. Acting301615/304223
  =.9914273411280541 over32episodes; residual signs negative185892/positive115550/
  zero173. Candidate-panel-only acting73636/74298=.9910899351261138,
  residual signs negative73552/positive0/zero84 (command differences before physics cap).
- Increment28,800; cumulative85,500charged=actual/650,000,95completed/precharged
  jobs,564,500remaining. Driverwall1236.274119473s (includes load-gate wait),
  plus screen137.474s. Supervisor3162394 exited; no live task processes.
- Reconciliation/receipt/screen/head snapshot MAIN1554_u00.json, headSHA226f2fad...
  unchanged. No sigma halving. Resource-yield entry above is historical, now resolved.
