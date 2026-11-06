# S-C0DIAG-1 — cycle-0 failure diagnosis

**Decision: reject deployment of `fa242777…`; keep WAIT_OWNER. The dominant failure is an insufficiently trained, consistently braking deterministic actor admitted by a progress-only gate—not a failure visible only in SUMO.** Registered analysis only; zero new SUMO, seeds, training updates or operational changes.

## Ranked findings

### 1. Confirmed: persistent negative commands stop the fleet

Across all 26 POLICY branches, **100% of recorded finite direct acceleration requests are negative**, mean **−0.412 m/s²**. CAV speed averages **2.44 versus native 20.72 m/s**; stopped-time (<0.1 m/s) is **82.45% versus 1.77%**, crawling (<2 m/s) **83.96% versus 11.41%**. Policy CAV speed falls from **19.01 m/s in seconds0–10 to 0.49 in seconds180–300**. Repeated braking blocks following traffic and insertion, consistent with the +415,385 entrance/+203,592 road veh-s and lost arrivals.

This is not evidence of entropy collapse or newly learned extreme metering. On the **same 372 observed initial CAV states**, deterministic raw means are **−0.434 at u100 → −0.367 at u150**, every action still negative; latent exploration sigma is **1.003 → 1.017**. Seeded initialization reconstruction gives −0.750; it is not a saved u0 checkpoint. The affine-tanh range **[−3,1.2]** has midpoint −0.9; `init_zero_accel=false`. Training improves an initially braking actor without reaching acceleration/discharge behavior. Stochastic training commands remain broad (u150 executed-action mean/std −0.759/1.153); deterministic collection removes those positive samples. Earlier v8/v9 results used600 updates and native policy lanes; they do not establish readiness after150 updates of this different recipe.

### 2. Confirmed: gate/collection wiring admits a policy already worse in imagination

`policy14_cli.py:351–365` checks **u10 J < u0 J**, not J < native. Thus16/16 improvements, mean−1.294, establish optimization progress only. Final independent source-inclusive10s rescore is **742.396**, versus native **721.075**: policy still **+21.321 veh-s worse**. All150 updates used10s imagination; the30s stage starts afterward.

Fresh offline replays from all26 recorded initial states also find POLICY worse in **26/26 at10s and60s**: mean road-cost differences **WM +13.57/+607.18**, **SUMO +16.12/+608.49 veh-s**. These explicitly match the **t0 cohort**, entries OFF, physical exits, no recorded future-mask censoring—not full-network/source-inclusive validation. At `968132@360`,300s gives WM **80.76% stopped-time**, road difference **+11,304.68**, versus SUMO **+9,476.19 veh-s**. The WM predicts the catastrophe’s direction and stopping mechanism; metering exploitation is not needed to explain it.

Collection configuration explicitly has **`gated_arm=false`**, `gate_candidates=8`; every world records `gate_table_used=false`, zero GATED branches. K=8 calibration is not a runtime acceptance check. `run_policy` adopts any successfully completed trainer checkpoint without a final native-relative veto.

### 3. Confirmed secondary symptoms; causal shares unresolved

Observed CAV lane changes are **10,288 versus517 native**, ramp merges **47 versus152**. Learned lanes and near-uniform categorical entropy permit substantial oscillation; its independent contribution to delay is unmeasured. `native_lane_change=true` applies to **non-acting reference CAVs**, not policy CAVs; it does not mean SUMO-native policy lanes.

There is a formal ramp-ownership mismatch: `bridge.py:217–226`, `policy_ramp_merge=false` plus `native_ramp_merge="wm_law"`, permits automatic WM policy-CAV merges; SUMO learned-lane ownership requires head requests. Setting **`policy14.policy_ramp_merge=true`** aligns ownership. However, this ablation leaves the representative300s replay identical, with zero automatic ramp merges originally: **not a demonstrated cause here**.

Critic support is weak: cycle0 fitted12 native returns, not its current policy. On26 actual current-policy states, online/target RMSEs are **16,615/16,630 veh-s**; prediction SD≈2,500 versus actual18,485. This supports long-tail miscalibration, not a proven wrong-sign critic gradient.

## Execution/objective checks

Both CPU checkpoint hashes match `fa242777…`;13 allowlisted controller/input source hashes match across all three hosts. Reconstructed first executor commands match tapes within **9.54e−7 m/s²**, with **zero lane-request mismatches**. Targets satisfy `v_target=v+a×1s` within1.88e−6; mean realised-minus-issued one-second acceleration is−0.00973. Clipping at zero speed explains near-zero issued commands after stopping, not a hidden hold. Ownership uses `slowDown(...,1s)`, speedMode27 and explicit native release. Physics steps are0.5s; decisions1s. SAFE SPEED2 only disables diagnostics/thins audits; no input normalization/order change was found.

`objective14.authoritative_reward` is **−(road+source waiting)**; gamma=lambda=1. Existing26 exported returns equal negative authoritative SUMO J. No sign reversal or omitted entrance term is evidenced.

## Minimal fix and proposed smoke

Keep the progress gate, but add a **separate final native-relative deployment veto** in `policy14_cli` and enforce its SHA-bound receipt in `loop14.run_policy` before adoption/collection. Require paired critic-free source-inclusive J against native, including a300s discharge check; this checkpoint must fail. Do not infer success from more updates, positive synergy, or an uncalibrated K-check. Align ramp ownership as above before retraining; validate deterministic outputs explicitly.

**Proposed S-C0DIAG-SMOKE-1; NOT RUN/allocated:** next two eligible TRAIN seeds in968100–968227 (currently968142–968143; recheck ledger),1900vph,5%/40%; warmup300, one saved start360, native/policy300s, extras OFF: **2×(361+600)=1,922 sim-s**. Register exact source/checkpoint/gate hashes and charge before launch. Verify rejected checkpoints cannot enter collection; for an admitted replacement compare commands, lane ownership, realised speeds, source-inclusive J, arrivals and incidents. This is a mechanics/rejection smoke, not efficacy certification.

Evidence/reproduction: `tools/c0diag/{SUMMARY,audit,tapes,replay,replay_enriched,ownership,actor,critic,provenance_*}.json`, adjacent offline scripts; local validation `python tools/c0diag/verify.py`. GPU allocation stayed below12GB; CPU work used two threads, no simulation workers.
