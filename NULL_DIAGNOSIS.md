# NULL_DIAGNOSIS — why ~half of the world-model plans have EXACTLY zero effect in SUMO

Read-only diagnosis of the `coopmwm20261102_lscollect1` SUMO-verification labels
(branch `c4/null-diag`, 2026-11-03). No SUMO was run, nothing was written or
signalled on `cpuQ`; no `sealed/`, `holdout`, `heldout`, `cal_split`, `chunk5`,
`j4b` or `inc1` path was opened.

---

## 1. Executive summary

Out of **27 graded** verifications (8 more failed before grading, no branches):
- **6 are exact nulls**: all 8 candidate branches produce delay **bitwise equal**
  to the `policy continues` branch (`identical_windows = 8/8`, every window
  component — `road_delay_vs`, `source_wait_vs`, `hard_brake`, `ttc_below_3_vs`,
  `collisions`, `teleports` — equal to the last float bit):
  `178198ee`, `17a09ce8`, `84ca0533`, `abf17661`, `b75425da`, `e63dcf70`.
- **2 are partial nulls** (margin exactly 0 because the best candidate's window is
  bitwise identical to policy while others differ): `0e23b698` (1/8 identical),
  `4924d43f` (4/8 identical).
- **8 more are near-nulls** (`|margin| <= 0.27 veh-s`): `19f0607b`, `1e75884e`,
  `3abc1aef`, `5420b879`, `79bb022c`, `9f753b50`, `cb1025ea`, `fda242d7`.
- 11 situations have real effects (−3244 … +1369 veh-s).

**Answer: the nulls are physical, not a wiring bug.** For every exact null we
verified that (i) the whole simulated world is bitwise identical across all 9
branches, while (ii) the continuation fingerprints **diverge one second after
t0** — i.e. the candidate's commands *did* reach the plant and *did* change the
pre-ownership safe-command tensor, and the difference was then erased between
`safe_accel` and realized motion. The erasure happens because the plans propose
accelerations for members that **cannot convert acceleration into progress**:
they are at the lane speed limit (the actuator's own target-speed ceiling makes
every positive command produce the *same* target float), or are pinned by a
leader/queue so SUMO's car-following/swarming constraint overrides the command.
On top of that, in 4 of the 6 exact nulls the WM's top-8 degenerated to the
**pure policy-centre family** (`i0s*` candidates with `saving_vs_policy ≡ 0.0`
recorded at proposal time) — the search returned the policy itself and tiny
library variants around a saturated centre.

Hypothesis verdicts (details in §7): **(a) confirmed (primary)**, **(b) rejected**,
**(c) rejected**, **(d) confirmed as eraser (registered actuator semantics, not a
bug)**, **(e) confirmed (contributor)**.

---

## 2. Data inspected (read-only, via `ssh cpuQ`)

| Source | Used for |
|---|---|
| `labels/*_SUMOENV1.json` (35 files; 27 graded + 8 failed) | per-branch `J_veh_s`, full per-branch `window` metric dicts, `continuation_prefix_sha256`, candidates, t0/window/episode |
| `branches/sv_lscollect1_B3_986004_t0380_c4cab5f8/` (exact null `abf17661`) | 8 `candN.spec.json` plan profiles, `SNAPSHOT.pt` (t0 sample, `accel_bounds`, noise memory) |
| `requests/<search_id>.json` (27 files) | t0 group `rows` (speed/position/lane), policy `commands`, `policy_residuals` |
| `proposer_receipts/<search_id>.json` (27 files) | all candidates with effective `perturbation_canonical`, `saving_vs_policy`, `topk` |
| `snapshots/lscollect1_B3_*.pt.gz` | existence check only |
| Net speeds | `/home/ubuntu/e3final/runtime/assets/research/highway_merge_shepherd_long/osm.net.xml.gz`: mainline lanes 31.29 m/s, ramp lane 22.22 m/s |

The three preserved branch dirs (`…986004_t0380…`, `…986018_t0300…`,
`…986016_t0300…`) have their `candN/{observable.jsonl.gz,RESULT.json}` pruned and
their root `*.result.json` overwritten by a later failed re-run
(`CavEnvError('SUMO binary not found on PATH')`), so per-second branch command
audits no longer exist; the receipts + specs + snapshot proved sufficient.

---

## 3. How a plan command is applied (the evidence chain)

1. **Plan installation at t0** — `tools/coopmwm/sumo_verify.py:588-593`
   (`VerifyController.apply`): candidate appended to `active_plans` with
   `start_s = t0`; committed plans restored; one shared CRN draw consumed
   (line 600).
2. **Profile consumption** — `tools/coopmwm/acceleration.py:127-135`:
   `sample['absolute_accels'].update(acceleration_at(c['plan'], now-c['start_s']))`;
   profile values are **absolute accelerations** (`tools/coopmwm/mppi_search.py:305-310`),
   built as `profile = clip(policy_traj + perturbation, bounds)`
   (`mppi_search.py:269-302`); member action is overridden by
   `raw - commands[vid]` (`acceleration.py:133-135`).
3. **Request emission** — `tools/coopmwm/acceleration.py:48-59`
   (`requests()`): plan members are emitted **only if `vid in commands`**
   (line 58) — `commands` is the pre-safety MAPPO command vector over *present*
   CAVs (`tools/coopfree/execution.py:47-50`).
4. **Pre-safety clip** — `tools/coopfree/execution.py:51-55` (`inject`):
   `raw[0,i] = min(accel_max, max(accel_min, requested_raw))` with executor
   bounds `accel_bounds = [-3.0, 2.6]` (from the branch `SNAPSHOT.pt`,
   `snap['sumo']['accel_bounds']`).
5. **Safety filter** — `execution.py:55` calls the unchanged
   `wms.control.safety.safe_accel` (`E:/agent-work/c4-leverprobe/src/wms/control/safety.py:52-129`):
   a nearest-same-lane-leader stopping-distance bound soft-min'd with the
   proposal (temperature 0.05; when the bound dominates, the filtered value is
   **bitwise the bound**, independent of the proposal).
6. **Command digest** — `tools/coopmwm/coopfree/backend.py:41`: the branch
   fingerprint hashes this **safe tensor** (`self.command_digest.update(...)`),
   i.e. *before* ownership.
7. **Ownership / actuator** — `backend.py:59` → `env.step` →
   `traffic_shepherd/ownership.py:114-147`: CAV speedMode set to **27**
   (`c4/env.py:84`, `ownership.py:102-110`), then the *only* longitudinal
   writer is `slowDown(v, target, dt)` (`ownership.py:137`) with the target from
   `direct_target` (`traffic_shepherd/damping.py:97-109`):
   ```python
   lo = max(accel_min, -speed/dt)
   hi = min(accel_cap, (allowed_speed - speed)/dt)   # <-- the speed-ceiling clip
   command = clip(acceleration, lo, hi)
   target  = max(0, speed + dt*command)
   ```
   Once `speed > allowed - 2.6`, `hi < 2.6` and **every raw ≥ hi yields exactly
   `target = allowed`** — the same float regardless of the plan. A stopped
   vehicle (`speed = 0`) has `lo = 0` (no reverse).
8. **Physics / metric** — SUMO (speedMode 27 still enforces the vType accel 2.6
   cap and leader/junction constraints, `c4/env.py:10-12`); per-0.5 s
   `speed_deficit += 0.5 * sum(max(0, 1 - s/31.29))` (`c4/env.py:601-602`) plus
   source-queue wait (`c4/env.py:616-619`) accumulate into the fingerprint's
   `metrics` (`tools/leverprobe/backend.py:128-133`, hex floats).
   Bitwise-equal totals ⇒ realized speeds identical at every substep.

---

## 4. Verified facts about the exact nulls

For each of `178198ee`, `17a09ce8`, `84ca0533`, `abf17661`, `b75425da`, `e63dcf70`:

- **F1 — total physical identity.** All 8 candidate `window` dicts equal the
  policy's dict bitwise (verified field-by-field from the label receipts), and
  `J_candidates_veh_s[i] == J_policy_veh_s` for all i. Over 360–420 simulated
  seconds this is only possible if every delay-relevant vehicle speed is
  identical at every 0.5 s substep.
- **F2 — commands reached the plant.** `continuation_prefix_sha256` differs from
  the policy's from t0+1 onward (178198ee: from t0+8; abf17661: cand1/2/3/4/7/8
  at 381, cand5/6 at 384). Fingerprint fields other than `commands` (tape,
  observable, metrics, memory, dwell, `tools/leverprobe/backend.py:128-133`) are
  functions of realized motion, which F1 fixes; hence the divergent component is
  the **safe-command tensor** — the plan changed the request, the request was
  erased downstream of `safe_accel`.
- **F3 — candidates are what was asked.** `candidate_ids` == proposer receipt
  `topk` in all 27 graded cases (checked; no id mismatch, no substitution).
- **F4 — members exist and are controllable at t0.** Every member of every
  graded situation appears in `requests/<sid>.json` `rows` with `is_cav: true`
  and a finite policy `commands[vid]` — no member was missing, departed, or
  un-keyed at t0. (E.g. the exact null `abf17661` at t0=380: `ramp.29`
  v=0.000 m/s s=75.4 lane 2 — *stopped in the ramp queue at the merge-end
  geometry* (`MERGE_S1_M = 75.44`, `traffic_shepherd/c4/geometry.py:35`);
  `mainline.141` v=3.399 crawling in the mainline queue.)
- **F5 — the policy itself is saturated in the nulls.** `raw = clip(cmd+residual)`
  hits the +2.6 pre-safety clip for at least one member in every exact null, and
  the ceiling `hi = (allowed − v)/1` is below the policy raw for at least one
  member in `84ca0533`, `4924d43f`, `5420b879`, `8ad19fe8`, `f175ffa9`,
  `b75425da`, `0e23b698`.
- **F6 — four exact nulls verified the pure centre family.** The verified
  candidate IDs are the iteration-0 seed block `i0s0, i0s1, i0s10…i0s15` with
  `saving_vs_policy = 0.0` recorded for **every** candidate
  (`17a09ce8`, `84ca0533`, `b75425da`, `e63dcf70`; also the near-nulls
  `19f0607b`, `79bb022c`). `i0s0` is the zero-perturbation centre plan
  (`mppi_search.build_plan` with `delta = 0`, origin `'centre'`).

---

## 5. Per-situation inventory

Graded verifications (margin = `best_margin_veh_s`; "ident" = candidate branches
whose whole window is bitwise identical to policy; raw = `clip(cmd+residual,
[-3, 2.6])`; hi = `(allowed − v)/1 s`; CB = ceiling binds (raw ≥ hi); PC =
policy raw at the 2.6 pre-safety clip).

| sid | episode (cell/demand/pen) | t0→end | members (v [m/s], raw, hi) | ident | margin [veh-s] | WM pred[0] | verdict |
|---|---|---|---|---|---|---|---|
| 178198ee | 986020 (3/1900/.1) | 300→660 | mainline.101 (16.0, 2.60 PC, 15.3); ramp.33 (14.8, 2.60 PC, 16.5) | 8/8 | **0.0** | 0.0 | **EXACT NULL** — policy at accel clip, all perts +0.17…+0.35 land inside clip; divergence at t0+8 erased by platoon CF |
| 17a09ce8 | 986035 (5/2700/.1) | 300→720 | mainline.113 (29.3, 1.93, 2.01); mainline.114 (29.2, 1.36, 2.08) | 8/8 | **0.0** | 0.0 | **EXACT NULL** — centre-family candidates (`i0s*`), speeds within 2.1 m/s of the 31.29 limit; erased at target ceiling/CF |
| 84ca0533 | 986041 (6/1900/.2) | 300→720 | mainline.98 (30.6, 2.11 **CB** 0.73); mainline.100 (29.9, 2.34 **CB** 1.34) | 8/8 | **0.0** | 0.0 | **EXACT NULL** — both members already at the limit; target ≡ 31.29 for policy and every candidate |
| abf17661 | 986004 (0/1900/.05) | 380→800 | ramp.29 (**v=0.000**, s=75.4 merge-end queue, raw 2.60 PC); mainline.141 (3.4, raw 2.60 PC) | 8/8 | **0.0** | 0.30 | **EXACT NULL** — one member jam-pinned at v=0, other's profile ≡ clip(2.6); perts ≤ 0.15 inside saturated region |
| b75425da | 986043 (7/2400/.2) | 300→720 | mainline.109 (30.0, 2.50 **CB** 1.34); ramp.17 (27.8, 2.57, 3.46); mainline.111 (29.2, 2.60 PC **CB** 2.10); mainline.113 (28.9, 2.33, 2.38) | 8/8 | **0.0** | 0.0 | **EXACT NULL** — 3/4 members ceiling-bound; centre-family candidates |
| e63dcf70 | 986039 (6/1900/.2) | 300→720 | mainline.87 (25.4, 2.36, 5.87); mainline.104 (27.1, 2.60 PC, 4.14); ramp.30 (24.9, 2.58, 6.39) | 8/8 | **0.0** | 0.0 | **EXACT NULL** — centre-family candidates (max positive pert +0.045); differences erased by leader/CF capping in the queue wave |
| 4924d43f | 986018 (3/1900/.1) | 300→720 | mainline.134 (30.9, 2.60 PC **CB** 0.40); mainline.135 (30.9, 2.60 PC **CB** 0.40) | 4/8 | **0.0** | 0.026 | **PARTIAL** — perts ≤ |0.03| (vs hi=0.40!): cand1-4 all-positive perts ⇒ target ≡ 31.29 identical; cand5-8 dip below ceiling later ⇒ ±0.1-0.35 veh-s |
| 0e23b698 | 986047 (7/2400/.2) | 300→720 | ramp.34 (13.8, 2.60 PC); mainline.117 (29.8, 2.33 **CB** 1.47) | 1/8 | **0.0** | 1.49 | **PARTIAL** — cand4 (index 3) window identical to policy = the "best" plan IS the policy; other candidates harmful (+800…+1400 veh-s) |
| 19f0607b | 986036 (6/1900/.2) | 300→720 | mainline.113 (25.3, 2.60 PC); ramp.14 (18.2, 2.53) | 0/8 | −0.003 | 0.0 | near-null, centre-family |
| 1e75884e | 986028 (4/2400/.1) | 300→720 | mainline.99 (5.7, 2.35); ramp.32 (0.5, 2.27) | 0/8 | +0.065 | 0.26 | near-null; members starting from near-stop |
| 3abc1aef | 986040 (6/1900/.2) | 300→600 | ramp.22 (15.6, 2.05); mainline.100 (29.3, 1.45, hi 1.96) | 0/8 | +9.6e-6 | −0.0 | near-null; all perts negative ⇒ no ceiling help |
| 5420b879 | 986038 (6/1900/.2) | 300→450 | mainline.111 (29.5, 2.60 PC **CB** 1.82); ramp.25 (19.2, 2.04); mainline.113 (29.5, 2.60 PC **CB** 1.82); mainline.117 (**31.13**, 2.49 **CB** 0.16) | 0/8 | −0.0028 | 0.038 | near-null; one member 0.16 m/s below the limit |
| 79bb022c | 986037 (6/1900/.2) | 300→450 | mainline.121/122/123 (27.2-27.4, raw 2.44/2.60 PC/2.60 PC) | 0/8 | −0.013 | 0.0 | near-null, centre-family |
| 9f753b50 | 986024 (4/2400/.1) | 300→600 | ramp.29 (0.62, 2.60 PC); ramp.32 (13.6, 2.23) | 0/8 | +0.0012 | **41.0** | near-null; WM predicted 41 veh-s; ramps pinned by mainline gap |
| cb1025ea | 986019 (3/1900/.1) | 300→660 | mainline.91/93/95/94 (24.7-26.0, raw 2.60 PC/2.38/2.07/2.21) | 0/8 | +0.252 | **255.6** | near-null; WM predicted 256, plans decelerate a platoon that SUMO holds together |
| fda242d7 | 986042 (7/2400/.2) | 300→570 | mainline.116 (28.2, 2.60 PC); mainline.117 (28.1, 2.60 PC) | 0/8 | −0.0018 | 0.27 | near-null; policy saturated, perts −0.83…−1.75 mostly swallowed |
| 14f02178 | 986030 (5/2700/.1) | 300→720 | mainline.130/131 (27.1/25.7, 2.60 PC/2.45) | 0/8 | −541.8 | 1.5 | effect (all candidates worse) |
| 16471b1d | 986025 (4/2400/.1) | 300→720 | mainline.134 (28.9, 1.70); mainline.91 (16.9, 2.60 PC) | 0/8 | +7.17 | 0.68 | effect (small but real: pert −1.23 changed motion) |
| 2ab50521 | 986032 (5/2700/.1) | 300→540 | mainline.122/125 (27.2/27.1, 1.71/1.51) | 0/8 | −86.4 | 226 | effect (harmful) |
| 4c56168f | 986033 (5/2700/.1) | 300→720 | 4 ramp members (0.8-6.6, raw 0.21-2.55) | 0/8 | −259.8 | 260 | effect (harmful); WM predicted +260, realized −260 |
| 520c4087 | 986026 (4/2400/.1) | 300→720 | ramp.24 (0.0), mainline.108/111 (26.5/26.7) | 0/8 | −3244 | −16.9 | effect (harmful) |
| 590f8883 | 986045 (7/2400/.2) | 300→600 | mainline.107/109 (26.8/27.2, **raw −1.35/−2.41 braking**) | 0/8 | **+1368.7** | 40.0 | effect — policy NOT saturated; plans (+0.10/−0.72 perts) have real room |
| 8ad19fe8 | 986034 (5/2700/.1) | 300→600 | mainline.107/109 (30.8/30.2, 2.60 PC **CB** 0.46/1.09) | 0/8 | −11.9 | 0.006 | effect; ceiling members again |
| b7104de5 | 986044 (7/2400/.2) | 300→600 | 5 members mixed (0.5-26.3) | 0/8 | +85.5 | 20.4 | effect |
| bce3b90b | 986023 (3/1900/.1) | 300→720 | mainline.109/103 (30.1/29.0, 2.60 PC **CB** 1.16/2.29) | 0/8 | −3242 | 262 | effect (harmful); +0.07/+0.08 perts above a ceiling policy ⇒ small headroom changes cascade |
| d5f54da8 | 986016 (2/2700/.05) | 300→480 | mainline.100 (28.7, raw **0.75**); ramp.24/26 (1.1/1.4, raw 1.71) | 0/8 | **+522.0** | **19.2** | effect — the reference case: policy commands modest, ramps have 30 m/s headroom; plans push +0.37…+0.63 ⇒ real motion change |
| f175ffa9 | 986046 (7/2400/.2) | 300→660 | mainline.108 (30.2, 1.23 **CB** 1.06); ramp.18 (**v=0.0**); mainline.109 (30.0, 1.31) | 0/8 | +434.1 | 11.6 | effect |

8 further labels failed before grading (`1ded134e`, `2dee450b`, `39bdbbe7`,
`449338a1`, `7566a51c`, `8312a1bc`, `bb847452` — mostly `SUMO binary not found
on PATH` / prepare failures); the 3 preserved `branches/` dirs belong to
`abf17661` (exact null), `4924d43f` (partial), `d5f54da8` (effect).

---

## 6. Mechanism: where exactly the plan's effect dies

The exact nulls satisfy F1+F2 simultaneously, so for each member-second the
command pipeline maps *different* plan/policy requests to the *same* realized
motion. The eraser layers, all confirmed in code:

1. **Pre-safety clip** (`execution.py:54`): plan profiles are clipped to
   `[-3.0, +2.6]` at build time (`mppi_search.py:294`) and again at injection.
   When the MAPPO policy itself floors the throttle (`cmd+residual ≥ 2.6`, F5),
   the plan's `clip(centre + perturbation)` equals the policy's clipped raw for
   every perturbation ≥ `2.6 − centre` — i.e. all upward perturbations are
   invisible *before* safety. (178198ee, abf17661 member `mainline.141`,
   4924d43f positive perts.)
2. **Speed-ceiling target clip** (`damping.py:104-108`): with `allowed = 31.29`
   (mainline; net evidence) and `dt = 1`, any member within 2.6 m/s of the limit
   has `hi = allowed − v`; every request `raw ≥ hi` — plan *or* policy — issues
   exactly `slowDown(allowed)`, the same float. (84ca0533 both members;
   4924d43f both; b75425da 3 of 4; bce3b90b, 8ad19fe8, f175ffa9, 5420b879
   partially.)
3. **Leader-bound soft-min saturation** (`safety.py:123-129`): for a member with
   a same-lane leader and a small stopping budget the filtered value equals the
   bound bitwise regardless of a ±0.1-1.0 proposal difference.
4. **SUMO car-following / junction pinning under speedMode 27**
   (`c4/env.py:84`, `ownership.py:102-110`): the remaining differences (safe
   tensors differ ⇒ `commands` fingerprint diverges at t0+1, F2) are swallowed
   by SUMO's leader/merge constraints — the vehicle's realized speed is set by
   its leader, not by the commanded acceleration. Verified endpoints: `ramp.29`
   pinned at v=0.000 for the whole window (abf17661); platoon members at
   15-30 m/s in 1900-2700 vph waves (178198ee, 17a09ce8, e63dcf70, b75425da's
   free member ramp.17).

**Why the WM walks into this.** The search is policy-centred
(`mppi_search.build_plan`): candidates are `clip(policy_traj + perturbation)`.
When the policy is saturated (F5), the centre sits on the clip boundary and the
whole positive perturbation half-space collapses onto the policy command. Four
exact nulls additionally show the topk degenerated to the iteration-0 seed
block with `saving_vs_policy ≡ 0.0` (F6) — the search spent its decision
verifying the policy against itself. In contrast, every effectful situation
(d5f54da8, 590f8883, f175ffa9, b7104de5, 16471b1d) has at least one member with
real headroom (low speed, unsaturated policy command) — e.g. d5f54da8's ramps at
v ≈ 1 m/s with policy raw 0.75-1.71 and plans pushing +0.5-0.6.

**Corroborating prior evidence.** `tools/coopmwm/zero_grade_probe.py:117-147`
already established the same class on the earlier campaign: "raw and issued
requests differ but sensed motion/rosters/rewards identical" — `observed
physical-null` plans whose pre-ownership commands differ while the tape does
not. This diagnosis extends it: the lscollect1 receipts prove the erasure
bitwise for whole 9-branch worlds, and identify the erasing layers above.

---

## 7. Hypothesis verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| a | Members at the speed limit or leader-constrained so "accelerate more" cannot change anything | **CONFIRMED — primary cause** | F5 (policy saturation, ceiling binds); §6 layers 2-4; exact nulls have members at 29.2-31.1 m/s vs allowed 31.29 or pinned at v=0 (abf17661 `ramp.29`); contrast: all effectful cases have low-speed, unsaturated members |
| b | Members not yet inserted / not controllable at t0 | **REJECTED** (for this dataset) | F4: every member present in `rows`, `is_cav`, in `commands`; continuation fingerprints diverge at t0+1 ⇒ commands were accepted. (The earlier warm-campaign zeros *did* involve members departing 1-4 s after t0 — `zero_grade_probe.py:120` — a different regime.) |
| c | Commands never applied (id mismatch / vehicle left) | **REJECTED** | F3 (`candidate_ids == topk` in 27/27); F4 (members keyed); F2 (plan branch command tensors diverged ⇒ they were applied); near-nulls/partial nulls show graded effects flowing through the same chain |
| d | SUMO speedMode / safety overriding | **CONFIRMED as the registered eraser** (not a defect) | speedMode 27 (`c4/env.py:84`), vType accel cap 2.6 (`c4/env.py:10-12`), `safe_accel` leader bound (`safety.py`), and crucially the **actuator's target-speed ceiling** `hi=(allowed−v)/dt` (`damping.py:104-108`) which makes distinct requests produce identical `slowDown` targets. The "post-safety command mirror" conclusion was right about outcomes but the commands themselves are NOT mirrored pre-clip — they diverge and are erased afterwards |
| e | Commands identical to policy because the perturbation is too small after quantisation | **CONFIRMED as contributor** | Not float quantisation but *saturation quantisation*: perts ≤ 0.03 (4924d43f, hi=0.40), ≤ 0.15 (abf17661), and the centre-family topk (F6: `i0s0` has perturbation exactly 0) all land inside the clipped/saturated region |

---

## 8. Recommended fix (code-change proposal; NOT deployed)

Goal: stop spending a 9-branch SUMO charge (≈ 9 × t_end sim-s) and stop
imitating no-op plans on **physically infeasible** decisions. Two layers,
both cheap, deterministic, CPU-only, and additive (no behaviour change when the
group is feasible).

### 8.1 Collector-side gate before verification (primary)

In `tools/coopmwm/sumo_verify.py`, inside `verified_guide.call` (lines 454-492),
after candidates are selected and before `capture()`/`verifier(...)`, compute
for each candidate whether it can issue a **different actuator target** than the
policy-centre plan on at least one member-second, using exactly the registered
chain semantics (`inject` clip + `direct_target` ceiling):

```python
def _issued_target(v_mps, raw, allowed, *, accel_min=-3.0, accel_max=2.6, dt=1.0):
    lo, hi = max(accel_min, -v_mps / dt), min(accel_max, (allowed - v_mps) / dt)
    return None if lo > hi else max(0.0, v_mps + dt * min(max(raw, lo), hi))

def actionable(cand, centre_plan, rows, commands, allowed_by_lane, *, eps=1e-9):
    """False when every member-second of `cand` maps to the same slowDown target
    as the centre plan (the policy continuation) under the registered actuator
    clip — i.e. executing `cand` cannot change the world."""
    from .mppi_search import acceleration_at
    for k, vid in enumerate(cand['members']):
        if vid not in commands:            # not controllable at t0 -> inactionable
            return False
        v0 = rows[vid]['v_mps']
        allowed = allowed_by_lane[rows[vid]['lane']]
        for t in range(cand['duration_s']):
            a = acceleration_at(cand, t).get(vid)
            b = acceleration_at(centre_plan, t).get(vid)
            if a is None or b is None:
                continue
            # crude single-step propagation of member speed under the centre
            v = min(allowed, max(0.0, v0 + t * min(accel_max, (allowed - v0))))
            if abs(_issued_target(v, a, allowed) - _issued_target(v, b, allowed)) > eps:
                return True
    return False
```

Then, in `call`:

- drop candidates failing `actionable(...)` (record them in
  `b['verify_k']['inactionable']` for the label);
- if **no** candidate remains actionable (this is exactly the F6 centre-family
  null and the saturated-ceiling nulls), skip the SUMO verification entirely:
  `proposal.update(accepted=False, reason='sumo_verify skipped: no actionable '
  'candidate (ceiling/jam-infeasible group)')` — no snapshot, no charge, group
  follows policy. The one consumed pad draw that keeps the CRN stream aligned is
  unchanged.

`allowed_by_lane` comes from the model lane table (mainline 31.29, ramp 22.22;
`traffic_shepherd/c4/geometry.py:22-31` + net evidence) or, better, from the
CAV's subscribed `allowed_speed_mps` already present in the environment state
(`traffic_shepherd/c4/sumo_io.py:82`).

### 8.2 Search-side feasibility clamp (root cause)

In `tools/coopmwm/mppi_search.py`, restrict sampling/scoring to the actionable
region so the WM never *proposes* infeasible accelerations:

- propagate the centre rollout speeds `v_i(t)` (integrating the centre profile
  under the same ceiling — pure arithmetic, no SUMO);
- clip each candidate's profile to
  `profile_i(t) <= min(bounds[1], (allowed_i − v_i(t))/dt)` during
  `build_plan` (and to `≥ max(bounds[0], −v_i(t)/dt)` for the stopped-floor),
  exactly mirroring `damping.direct_target`;
- treat candidates whose clipped profile equals the centre on every member-second
  as duplicates of `policy continues` and exclude them from `topk` (they can
  never beat policy in SUMO; today they consume up to 8 of 8 verification slots,
  F6).

This makes the WM's `saving_vs_policy` reflect *issuable* plans, so the
`+255/+261/+41` hallucinated savings (cb1025ea, bce3b90b, 9f753b50) disappear
from the search's Thompson weights as well.

### 8.3 Label side (cheap, improves the next WM iteration)

When writing verification receipts (labels), add per member at t0:
`allowed_mps`, `headroom_mps2 = (allowed − v)/dt`, `policy_raw_at_clip` (bool),
`leader_gap_m/leader_v` if available — so imitation/library builders can mask
infeasible seconds instead of learning that "accelerate at 2.6 does nothing"
(`plans.py:99-115` currently weights candidates purely by predicted saving).

Expected effect on this dataset: all 6 exact nulls and both partial nulls would
have been skipped before any SUMO branch (their members are ceiling-bound,
centre-duplicates, or jam-pinned per §5), saving ≈ 8 × 6-9 × t_end ≈ 30-40k
sim-s per 8 graded nulls, with zero change for the 11 effectful situations
(each has an actionable candidate under §8.1's predicate).

---

## 9. Method notes

- Everything above was derived from existing artifacts; the only remote
  executions were `python3 -` one-shot read-only scripts over `labels/`,
  `requests/`, `proposer_receipts/`, `branches/*/{*.spec.json,SNAPSHOT.pt}`,
  and the frozen net file. Nothing was written remotely; local code was read at
  `E:/agent-work/c4-nulldiag` (branch `c4/null-diag`) and the frozen plant
  sources at `E:/agent-work/c4-leverprobe` (`tools/leverprobe`,
  `src/wms/control`, `traffic_shepherd`).
- Caveat: per-second branch command audits (`observable.jsonl.gz`) were pruned
  from the three preserved verification dirs, so the per-second attribution of
  *which* eraser layer acted at each second is inferred from (i) bitwise window
  identity, (ii) fingerprint divergence timing, (iii) the reconstructed
  t0 state and profile-vs-clip arithmetic. The existence of erasure (F1+F2) is
  measured, not inferred.
