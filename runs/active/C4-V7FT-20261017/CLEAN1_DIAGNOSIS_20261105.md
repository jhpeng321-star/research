# CLEAN1 diagnosis — why B has not demonstrated cooperation yet
**Plain-language summary (2026-11-05; resumed 2026-11-06 through final completed u9):**
- B is learning and acting; it is not mostly choosing `none`: training `none` stays near 11%, while `ease_off` falls from 83% to 77% and wall rises from 3% to 8.4%.
- The quoted +0.043 veh-s measures the group’s extra benefit over its best single member, not B’s total benefit over the common base; the receipt’s `B-common_base` label is misleading.
- Exact-zero scores do not mean no move or zero residual: both zero-score B states had accepted moves and nonzero residuals.
- The clearest bottleneck is weak, reversed move actuation: at u6, adding the recorded braking offset raises predicted acceleration by only 0.0085 m/s² on average in 1,263 sampled active contexts.
- B accepted 1,285 moves at u1, but all 16 matched total-delay outcomes equaled A; move selection initially had no physical effect.
- Training opportunities are also skewed: roughly seven of eight proposed moves are usually individual easing, and many selected joint moves cannot acquire their members.
- Bank learning is substantial, but concentrated: eight admitted seed examples, seven plans, only three usable trigger-supervision contexts, and the same three practice contexts repeated twice per update.
- Wall selections have negative mean PPO advantage at u1–u6 and u8–u9; gap/ramp selections are usually positive after u1 but rare. These are associations, not matched move-versus-none causal effects.
- The continuous PPO step is not hitting its KL cap; bank imitation is. Raising the common KL budget indiscriminately would mainly strengthen already-dominant imitation.
- Merge inputs reach the layer and affect its output; missing=0 only means own packets were present, not that all useful merge information was available.
- Comparable physical B–A outcomes remain mixed: u8 is 1.3% better, u9 is 1.7% worse on exact paired training mixtures; neither establishes group cooperation.
- Clean1 was owner-stopped at final A9/B9/R9. Preserve that evidence; prioritize actuation/credit alignment, feasible joint proposals and diverse practice in a follow-up. No run changes were made.

## 0. Scope, provenance, and what was actually inspected

**Analysis ID: S-DIAG-CLEAN1-20261105.** Zero simulation, zero training, zero new charges. Remote work consisted of SSH receipt/source reads, CPU arithmetic, checkpoint tensor inspection and forward-only residual sensitivity calculations. No process discovery, inspection, signaling, restart, or run modification was performed. No remote output files were intentionally created; Python bytecode writing was disabled for environment-backed analysis. Local scratch products are under `E:/Temp/opencode/sdiag_*`; this document is the requested deliverable.

All paths below are relative to `/home/ubuntu/coopmwm20261105_clean1` unless explicitly qualified. Source references are relative to `/home/ubuntu/coopmwm20261105_clean1_src_d81d4fae/pytools/tools/coopmwm/`, the newest source tree found on resume and the source named by `RESUMED_1434.json`. SHA comparisons establish that `track2_backend.py`, `launch_production.py`, `coop_layer.py`, `trigger_policy.py`, `track2_reward.py`, and `moves.py` are byte-identical in original `7cc6edaa`, previously inspected `94d90e84`, and final `d81d4fae` trees. Thus the original source-based diagnosis and line references still apply. The environment `/home/ubuntu/cpus_env/env.sh` was used when reading tensors. Only permitted receipt, checkpoint, observable-tape and implementation paths were used; excluded evaluation partitions and population/driver truth were not consulted.

**Final cutoff, verified on resume:** the root contains completed `A_UPDATE_009.json`, `B_UPDATE_009.json`, and `R_UPDATE_009.json`, each with a checkpoint identity. The remote `CLEAN1_STATUS.md` lags R at u8; completed update receipts take precedence. `STOPPED_BY_OWNER_CLEAN2_RESTART.json` and `STOP.json` record the owner's durable stop and restart decision; local `CLEAN1_STATUS.md` at 16:21 confirms final A9/B9/R9. No u10 update receipt was found in the root receipt inventory. The original draft's u8-only cutoff and recommendation to continue clean1 are superseded by this paragraph. No process health was inspected or inferred. The last captured original-session ledger total was **1,189,440 / 5,000,000 sim-s**; the local stop-era status reports **1,218,240**, including parity960 and plan-fix2400. These are dated accounting snapshots, not a fresh total for subsequently owned work. The approximately0.5M in the question is older still.

Primary evidence:

1. `CLEAN1_STATUS.md`, captured `LEARNING_MONITOR_u001.json` / `LEARNING_MONITOR_u006.json`, the per-arm monitor inventory, and full `A_UPDATE_001..009.json`, `B_UPDATE_001..009.json`, `R_UPDATE_001..009.json` receipts.
2. All **144 B training episode receipts**, u1–u9, under `jobs/t2c_warm_uUUU_B_<world>/RESULT.json.gz` and `jobs/t2c_warm_uUUU_pNN_B_<world>/RESULT.json.gz`. These contain actual trigger tables/selections, accepted flags, move offsets, residuals, post-action sensor packets and recorded trigger values. The resumed session reused all completed u1–u8 arithmetic and added only the missing u9 analysis.
3. `track2_diagnostics/replicated_u006_live.json`, its branch receipts, and all seven B all-group continuations and their reference continuations. Post-snapshot samples were inspected, rather than treating the replayed first sample as current-policy telemetry.
4. `track2_diagnostics/probe_u002_live.json`, `probe_u004_live.json`; `replay_bank/B/BANK_QUALITY.json`, `PRACTICE_PLAN_B_006.json`, update-bound bank summaries; matching legacy plan receipts; `B_track2_000..009.pt`.
5. Exact same-world A training outcomes and **54 matched A practice-reference episode receipts**, six per update, for physical B–A comparisons.

Selected identities:

- Canonical configuration SHA: `cc009d490c6876ec9e6ec98072617ac6f76b20fc7501fcc83bb7905f19a7ac4a`.
- u6 replicated diagnostic receipt SHA: `d6b47e6a908f8d6d264098d4ff74c5ae9817f0e5a612e0b4d776d18d3eaa6b76`.
- B u6 update receipt SHA: `783d8457a38b2c0ec85448e96c0677c06e50692002c83a14d7e79f414e63f4ec`.
- B u8 update receipt SHA: `37193a0b1d908d0dbcd64f968ea5cc4541b75d64acdde4e65426e6bc8a092caa`.
- B u9 update receipt file SHA256: `7e015cf284d63e0bd34adef1d25597d890eb7f03b7974d934a98de8c806412f1`; A u9: `dd50f415f604e637948b9d7fa62d77c86b298616613aa775118952a94e6d85c8`; R u9: `b72150a8bfdb649359e8d266bb13d8c584716b478c8645cb856968f8f70ee6fd`.
- Durable owner stop receipt file SHA256: `0f02b9ef9f84135be8dfc14b75ddaf8b133f8072db0b0fe0e697ca81df5b95b2`.
- B u6 base-checkpoint SHA: `a7cf1772203be36a1ee5bbb5f64f8317db02903355dc94a6998c4453fa7f7e87`. B’s learned cooperation parameters live in the separate `B_track2_006.pt` sidecar; inspecting the base checkpoint alone would miss learning.

### Recalculation checks

Trigger counts and accepted starts recomputed from episode samples match all nine update receipts. Reward arithmetic was reconstructed from actual post-action own/authorized-neighbour packets. Raw practice RESULTs retain replayed prefix seconds that the trainer strips: **126 seconds/update**, own reward sum **−31.472830327994**, team sum **−59.2794787101584**. Removing these reproduces all nine reported own/team sums to floating-point rounding. Those prefix seconds contain **zero current trigger decisions**, so the family and decision-advantage results below are unaffected. See `practice_starts.strip_steps`, lines 293–298, and `track2_backend._temporal_targets`, lines 792–844.

## 1. Trigger history, actual execution, and the exact-zero test entries

### Training: mostly individual easing, not mostly none

Rates below are percentages of **per-CAV trigger selections**, not per-second actuation. Counts are in parentheses for `none`; all other family counts can be recovered from the cited receipts.

| Update | Decisions | none | ease_off | wall | gap_open | ramp_assist | Accepted move starts |
|---|---:|---:|---:|---:|---:|---:|---:|
| u1 | 1,998 | 11.06% (221) | 83.18% | 3.00% | 1.50% | 1.25% | 1,285 |
| u2 | 2,005 | 10.12% (203) | 82.69% | 3.84% | 2.39% | 0.95% | 1,295 |
| u3 | 1,947 | 11.56% (225) | 81.72% | 4.98% | 1.08% | 0.67% | 1,254 |
| u4 | 2,008 | 10.66% (214) | 81.27% | 4.73% | 1.99% | 1.34% | 1,307 |
| u5 | 1,950 | 11.03% (215) | 78.67% | 6.21% | 2.62% | 1.49% | 1,230 |
| u6 | 1,927 | 11.31% (218) | 80.54% | 5.66% | 1.92% | 0.57% | 1,225 |
| u7 | 2,041 | 11.07% (226) | 79.03% | 6.81% | 2.35% | 0.73% | 1,298 |
| u8 | 1,927 | 10.74% (207) | 77.94% | 7.84% | 2.02% | 1.45% | 1,245 |
| u9 | 1,912 | 11.35% (217) | 77.04% | 8.37% | 2.04% | 1.20% | 1,215 |

**Important denominator:** `trigger_counts` counts selected proposals, including ones refused because of overlapping commitments. For example, u6 selected 109 walls but accepted only **37**; gap opening selected 37/accepted **14**, ramp assistance selected 11/accepted **2**, easing selected 1,552/accepted **1,172**. u8 walls selected151/accepted56; u9 walls160/accepted61, gaps39/accepted14, ramps23/accepted5 and easing1,473/accepted1,135. Several simultaneous members can propose the same group; these are not independent wall executions.

**The proposal generator heavily sets the prior.** Across u6’s actual tables there were 14,374 easing candidates, 712 wall, 214 gap, 116 ramp, and 1,927 none entries: easing was **82.9% of all 17,343 candidate slots**. u8 easing occupied 14,291/17,343 = **82.4%** of slots. Uniform selection over the nine slots already predicts approximately 11.1% none and approximately 82–83% easing. The observed u8 wall share, 7.84%, exceeds its approximately 4.01% slot share. Thus the history shows some increased preference for walls, not a collapsed no-op policy. Candidate feasibility and duplicated easing options materially constrain exploration. Sources: raw `track2_trigger.tables`; `moves.candidates`, lines 331–370; `trigger_policy.decide`, lines 214–255.

### Replicated u6: the seven registered B states

Only selections belonging to the **intervention group** are counted below, after the captured snapshot action (`time_s > t0`). Counting all background CAVs would answer a different question. These diagnostic continuations sample stochastically under restored CRN; fixed paired evaluation uses argmax. The job’s outer deterministic flag does not override the VerifyController’s continuation sampling. Sources: `sumo_verify.py`, lines 684–724; `launch_evaluation.py`, lines 25–42.

| Replicate / state | World | Group selections after t0 | Accepted starts | Group-vs-reference improvement, veh-s | Extra benefit over best solo, veh-s |
|---|---:|---|---|---:|---:|
| r0/s0 | 986040 | ease 2 | ease 2 | +0.000001309 | +0.000001188 |
| r0/s1 | 986040 | ease 2 | ease 2 | **0.0** | **0.0** |
| r0/s2 | 986040 | ease 3, wall 3 | ease 3, wall 1 | −0.383673 | −0.631782 |
| r1/s0 | 986048 | ease 3 | ease 2 | −0.095355 | −0.095379 |
| r2/s0 | 986042 | ease 2, none 1 | ease 2 | **+0.037034** | **0.0** |
| r3/s0 | 986046 | ease 3 | ease 3 | **+708.776230** | +0.519787 |
| r4/s0 | 986068 | ease 4 | ease 4 | −0.000149 | −0.000149 |

Totals: **20 ease, 3 wall, 1 none** selections; 18 accepted easing starts and one accepted wall start. B plainly does not mostly select none in these states either. This is a small, selected state panel, not a population trigger-rate estimate.

### What the reported statistic actually means

The implementation computes:

`delta_all = J_reference − J_all`

`cooperation_gain = delta_all − max_i(J_reference − J_solo_i) = min_i J_solo_i − J_all`.

There is **no zero-floor** on solo improvement. `track2_monitor` calls a bootstrap comparison of this synergy score against zero **`B-common_base`**. That name does not turn it into total reference-relative improvement. Sources: `policy_synergy.metrics`, lines 141–145; `track2_monitor._analyse`, lines 483–485.

The quoted B mean **+0.0427329205**, CI **[−0.1414648011, +0.2927961548]**, is correctly reproduced as this **extra-over-best-solo** score. Replicate 0 averages its three states before the five-replicate average. The corresponding *descriptive* mean `delta_all`, using that same replicate weighting, is **+141.717974 veh-s**, dominated by the single +708.776 state; no CI for this alternative metric is claimed here. In that large positive state the best solo already saves **708.256444 veh-s**, leaving only **0.519787** of group-specific benefit.

R’s synergy mean **−110.823165**, CI **[−332.434344, −0.000265]**, is dominated by r3’s **−554.028416**. B−R’s **+110.865898**, CI **[−0.093439, +332.709978]**, is not convincing evidence of B cooperation; avoiding one R failure can explain much of the contrast.

### Exact zero does not identify a no-op

- **r0/s1:** both members selected and started easing. There were **42 group-CAV action ticks**, every recorded learned residual nonzero, range **[−0.282988, +0.221672] m/s²**. Five ticks had a nonzero move offset. Commands differed from reference on **27/42** matched-clock ticks, yet aggregate delay and synergy were exactly equal.
- **r2/s0:** two easing starts were accepted and the third selection was none. All **95** residual values were nonzero; **55** group-CAV ticks had nonzero offsets. `delta_all = +0.03703395302`, exactly equal to the best solo improvement. **Synergy is zero because one member achieves the same aggregate benefit.**
- **r4/s0:** four easing starts were accepted but all **30** group-CAV move offsets were zero during their observed lifetimes; learned residuals were nevertheless nonzero. Commands differed only around numerical precision. Delayed onset/short opportunity is a plausible explanation for this ineffective move exposure, not proven individually from onset-only summaries.

The post-t0 receipts contain `coop_layer`, `track2`, and `policy_synergy_diagnostic_only`; the learned wrapper is actually present. The first t0 sample lacks these fields because it is the replayed source action, which should not be mistaken for a sidecar-loading failure.

## 2. Learning signal: is moving rewarded or penalized?

### What reward B actually optimizes

Per acting CAV, each physical control second:

`own = −max(0, 1 − observed_speed / 31.29)`

`team = own + mean(authorized_observed_neighbour_rewards)`, with neighbour weight **1.0** and empty-neighbour contribution zero.

This is real measured local speed-deficit cost. It is **not the network’s source-inclusive total delay**, and “team” here means own plus local neighbour mean, not a sum over all requested vehicles. Credit uses full rollout discounted return-to-go, **gamma = exp(−1/600)**, divided by **10,000**; credit stops at episode end or the acting CAV’s departure. Residual PPO uses that nonpositive return with a **zero baseline**. Trigger PPO uses `return − recorded trigger value`. Dense teacher and WM group credit are disabled. Sources: `track2_reward.neighbourhood_reward`, lines 80–131; `track2_backend`, lines 691–748 and 792–844.

### Actual decision-tick reward: accepted move versus none

These are per-CAV means of real **t+1** reward at a trigger decision. A move accepted at t can have an onset delay, and old commitments can remain active after a none selection. “Accepted” therefore means a move start was admitted, not that its offset acted immediately. Higher/less-negative is better.

| Update | Own: accepted | Own: none | Team: accepted | Team: none |
|---|---:|---:|---:|---:|
| u1 | −0.3002 | −0.3454 | −0.5224 | −0.6224 |
| u2 | −0.3292 | −0.3541 | −0.5796 | −0.6594 |
| u3 | −0.2944 | −0.3651 | −0.5125 | −0.6828 |
| u4 | −0.3077 | −0.2595 | −0.5415 | −0.4962 |
| u5 | −0.3062 | −0.3615 | −0.5354 | −0.6280 |
| u6 | −0.2903 | −0.3148 | −0.5018 | −0.5909 |
| u7 | −0.3141 | −0.3685 | −0.5606 | −0.6616 |
| u8 | −0.2967 | −0.3042 | −0.5138 | −0.5616 |
| u9 | −0.2807 | −0.3234 | −0.4839 | −0.5757 |

**There is no evidence of a universal penalty for any move firing.** Accepted starts have less-negative immediate rewards in eight of nine updates, but this is confounded by which states offer/accept which moves. It is not a same-state estimate of the move’s causal value. Whole episodes usually contain both move and none decisions, so “move episode versus none episode” would not be a valid clean split.

### Actual trigger advantage by selected family

Means below use the trainer’s discounted, scaled target and **recorded per-CAV trigger value**. Positive means trigger PPO tends to reinforce the sampled choice; negative means it tends to suppress it. These are sample-count-weighted descriptive means, including rejected non-none selections because the current PPO code includes them.

| Update | none | ease_off | wall | gap_open | ramp_assist |
|---|---:|---:|---:|---:|---:|
| u1 | −0.060791 | −0.054833 | **−0.101402** | −0.008006 | −0.008927 |
| u2 | +0.004027 | +0.007411 | **−0.011983** | +0.021670 | +0.038049 |
| u3 | −0.000120 | +0.001532 | **−0.006087** | +0.014876 | +0.020317 |
| u4 | +0.001824 | +0.002574 | **−0.002088** | +0.007033 | +0.010048 |
| u5 | −0.001534 | −0.001121 | **−0.004024** | +0.000392 | −0.000025 |
| u6 | −0.002504 | −0.003248 | **−0.004011** | +0.001203 | +0.001023 |
| u7 | +0.002877 | +0.002550 | +0.002516 | +0.000269 | +0.007993 |
| u8 | +0.000290 | +0.000038 | **−0.001697** | +0.002620 | +0.003590 |
| u9 | −0.002797 | −0.002932 | **−0.004007** | −0.004059 | +0.000277 |

At u6, wall advantage is positive for only **29.4%** of 109 selections, versus gap **56.8%** of 37 and ramp **45.5%** of 11. At u8 those shares are wall **47.7%**, gap **71.8%**, ramp **67.9%**. Wall is the consistently disadvantaged family by mean, but the gap/ramp samples are very small and arise in slower states: u6 own/team reward for gap is **−0.9503/−1.7484**, compared with easing **−0.2653/−0.4750**. Raw cross-family reward would conflate congestion with action quality.

Residual PPO’s zero-baseline return is nonpositive for every family; that is an objective/variance issue, **not evidence that every sampled action is causally harmful**. Trigger values swing from initially too positive to later negative values, changing all family advantage signs together. Rejected proposals also receive policy-gradient credit despite not starting their proposed commitment. Together with weak move actuation, this makes useful joint-action credit noisy.

### Same-state evidence and its limit

The strongest existing matched observation is u1: **all 16 B versus exact A-reference total-delay outcomes agree**, while B accepted 1,285 moves and its residual/gain were zero. Thus many categorical “moves” initially did nothing to acceleration. The replicated all/solo/reference branches share snapshot and CRN and identify the policy bundle’s incremental physical outcome, as shown in §1; they do **not** isolate a single move versus none while holding the learned residual and later choices fixed. No such causal move-versus-none reward estimate is manufactured from these receipts.

## 3. Bank imitation: learning, concentration, and state overlap

### CE genuinely decreases, but on three usable contexts

`bank_move_ce`, u1→u9:

**2.063005 → 1.849368 → 1.633644 → 1.404474 → 1.170792 → 0.940977 → 0.727508 → 0.546412 → 0.401653.**

Every update reports **three** trigger-supervision examples and five accepted CE passes. At u6 those passes move CE **1.010137 → 0.940977**; at u8 **0.601897 → 0.546412**; at u9 **0.445908 → 0.401653**, final CE KL0.004526. The decreasing trend is real fitting of a tiny fixed training set, not an independent generalization score. The exact projected label is inserted into the bank candidate slate at slot1; online slates must generate their own feasible approximations. Source: `launch_production.bank_move_supervision`, lines63–151.

The update-bound effective bank has **eight admitted/cooperative seed examples from seven plans**, **226 continuous imitation pairs**, **750 target member-actions per full pass**, and no augmented examples. Five seed examples are tagged source policy **u3**; their sum of example-level AWR weights is **92.47%** of all eight example weights. This is an example-level concentration measure, not a claim about exact gradient share or pair-weighted exposure. Two examples hit weight20, one is weight16.3663; the three practice examples have weights only **1.8256, 1.2972, 1.7724**. Source policy-version tags describe the legacy producers, not clean1’s update number.

### Which families dominate?

The seven distinct source plans project onto **six wall-shaped labels and one none label** under the current shared vocabulary. The none-labelled plan is still certified cooperative in its original native-plan comparison: projecting a richer recorded profile onto a small move library can erase its shape. Its seed-1 advantage is only **0.013823 veh-s**, synergy **0.000496**.

Verified projections include:

- World986066/t300, six-member native plan → three-member wall, onset9, hold90, depth0.55067; two seed examples, weights20 and2.7865.
- World986068/t600, four-member native plan → two-member wall, onset2, hold90, depth1.0; weight20.
- World986003/t390 → two-member wall, onset3, hold22, depth1.0; weight16.3663.
- World986052/t300 → none; weight1.000013.
- Practice worlds986026/t300, 986036/t300, 986047/t360 → three-/four-/four-member wall projections, holds29/28/24, depths1.0.

These are projections of the verified native plan profiles, not proof that executing the projected wall retains the native plan’s verified benefit. The stored legacy plans include richer offsets and, for some plans, more members/longer durations than this online vocabulary can represent. `moves.from_plan`, lines 447–496, explicitly minimizes profile fit, caps membership at four and hold at90 seconds, and tie-breaks generic multi-actor shapes toward wall.

**Remaining instrumentation gap:** the receipts report only the count of three surviving trigger contexts, not their identities/family mix after the local-visibility filter. The source-plan family dominance above is established; the exact post-filter three-example family histogram was not successfully captured within this read-only session. Do not equate eight admitted bank rows with eight CE contexts, or claim all three CE targets are walls without that missing join.

### Practice capacity and overlap with the test panel

The effective practice target is **50%** of 16 episodes, but actual B practice is **6/16 = 37.5%** every update. `PRACTICE_PLAN_B_006.json` requests8, has only **three eligible examples**, and caps repetitions at2: capacity6. The same three short-plan contexts are repeated twice. Raising only `practice_weighting.fraction` cannot increase practice under these unchanged eligibility/repeat constraints. Augmentation is OFF and contributes zero additional contexts.

The seven registered u6 states comprise worlds **986040 (three states), 986048, 986042, 986046, 986068**. At world/t0 granularity the admitted B bank overlaps **986068/t600**; the three usable practice contexts overlap **none** of these test world/time pairs. A world/time match is not a claim of identical physical state, snapshot hash, or CRN. The two repeatedly zero-ish 986040 states are not among B’s seven admitted plan contexts. Thus successful CE on three bank contexts need not transfer to these diagnostic opportunities. This overlap calculation is descriptive; it is not a proposal to train on the diagnostic panel.

### Imitation weight is meaningful relative to PPO

This is **sequential optimization**, not `PPO_loss + lambda * CE` with a tiny lambda:

1. Four continuous residual-PPO epochs.
2. Trigger PPO over approximately236–238 decision batches.
3. Separate full-bank continuous AWR passes, coefficient/beta1, clipped positive weights up to20.
4. Five separate trigger-CE passes, weighted mean over the surviving contexts.

Continuous imitation spends almost its full **independent 0.045 KL budget every update**, with large gradients and backtracking (§4). Trigger CE also changes parameters after PPO and has an independent trust region. At u6, average trigger PPO policy loss is **+0.002702**, versus entropy-term magnitude **0.021732**; at u8 these are **−0.000285** and **0.021488**. CE is approximately0.94/0.55 on its examples. Loss numbers on different batches and with cancellation are not gradient ratios, but the accepted CE steps and near-cap continuous imitation prove imitation is not negligible. No evidence supports “just turn imitation up” as the first remedy.

## 4. Layer capacity, step size, and residual distribution

Architecture: **59 inputs → 64 tanh units → one linear residual**, plus a learned scalar gain multiplying the active move offset. Inputs are29 local sensor features,29 merge features, and one active-offset feature. The residual output itself is **not hard-bounded**; the gain is `tanh`-bounded. The MAPPO base is frozen in B/R. LR is **3e−4**, gradient clipping0.5, four PPO epochs; joint Gaussian KL cap0.045. Source: `coop_layer`, lines 70–99, 128–159; `track2_backend`, lines 135–163, 184–238.

| Update | Accepted PPO KL after epoch4 | Final PPO gradient norm | Separate bank KL | Bank passes completed / requested |
|---|---:|---:|---:|---:|
| u1 | 0.025361 | 0.000310 | 0.044501 | 5/5 |
| u2 | 0.005405 | 0.000957 | 0.044222 | 5/5 |
| u3 | 0.007501 | 0.000490 | 0.040565 | 4/5 |
| u4 | 0.011782 | 0.000816 | 0.043452 | 3/5 |
| u5 | 0.011763 | 0.000624 | 0.044403 | 4/5 |
| u6 | 0.013933 | 0.000461 | 0.042246 | 3/5 |
| u7 | 0.014651 | 0.000485 | 0.043730 | 2/5 |
| u8 | 0.008633 | 0.000660 | 0.042143 | 2/5 |
| u9 | 0.008320 | 0.000373 | 0.043582 | 2/5 |

**All 36 residual PPO epochs are accepted.** Their gradient norms are far below0.5, and u6’s KL rises **0.000964 → 0.003712 → 0.008082 → 0.013933**. The PPO leg is limited by its LR/four-step schedule and weak/scaled objective, not KL rejection or gradient clipping. Raising only the KL cap would change none of these accepted PPO steps.

**Bank imitation is trust-region constrained.** Its last recorded raw gradient norms span approximately23–69, its KL stays near0.045, and u6 reports14 backtracks and refuses pass4 even at scale1/64. u7/u8 refuse pass3; u8’s bank AWR loss still falls **12.453961 → 11.885571**. Full-pass AWR scales steps upward, up to registered multipliers, then rolls back over-budget attempts. A larger bank cap would plausibly increase imitation movement, but may intensify the concentrated supervision. Its KL reference is the **post-PPO policy**, not the rollout policy; stage-specific KLs must not be treated as one end-to-end bound.

### Residuals are growing; the named move pathway remains weak

Training rollout residuals below are the **combined learned mean corrections**, before action/plant constraints, measured on actual member actions. Rollout uN uses checkpoint uN−1.

| Training update | Mean absolute residual, m/s² | Signed mean, m/s² | Observed min…max, m/s² |
|---|---:|---:|---:|
| u1 | 0.0000 | 0.0000 | 0…0 |
| u2 | 0.0327 | +0.0209 | −0.0460…+0.0943 |
| u3 | 0.0558 | +0.0358 | −0.0806…+0.1647 |
| u4 | 0.0781 | +0.0504 | −0.1340…+0.2296 |
| u5 | 0.1057 | +0.0701 | −0.1876…+0.2935 |
| u6 | 0.1285 | +0.0859 | −0.2479…+0.3501 |
| u7 | 0.1571 | +0.1094 | −0.3146…+0.4266 |
| u8 | 0.1728 | +0.1099 | −0.3733…+0.4939 |
| u9 | 0.1737 | +0.0981 | −0.4450…+0.5015 |

At u6, **80.35%** of these residuals are positive. This is a small, changing longitudinal controller, not a dead layer. These are distribution summaries of policy output, not measured acceleration changes or beneficial traffic effects.

The u9 absolute-residual distribution is p05 **0.01206**, median **0.13970**, p95 **0.39191 m/s²**, with **zero exact zeros among114,879** member-actions. Signed residual p05/median/p95 is **−0.36400/+0.09873/+0.36775 m/s²**.

Checkpoint `tanh(trigger_gain)`, u0→u9:

**0, −0.001193, −0.002374, −0.003519, −0.004602, −0.005715, −0.006875, −0.007676, −0.007765, −0.007867.**

At u6 a nominal move offset of −0.3…−1.0 m/s² contributes only **+0.00206…+0.00687 m/s²** through this explicit term: tiny and opposite to intended easing. The residual network also receives the active-offset feature, so the scalar gain alone does **not** bound the move’s total influence.

**New forward-only check resolves that second pathway.** Using checkpointu6 on the same3,060 recorded feature rows used in §5, **1,263 rows have a nonzero active offset**, mean−0.67986 m/s². Holding all58 other inputs fixed and replacing only the offset by zero gives:

- Network-only active-offset effect: mean **+0.003782 m/s²**.
- Explicit gain contribution: mean **+0.004674 m/s²**.
- Combined effect: mean **+0.008456**, median **+0.007363**, p95 **+0.015942**, range **+0.000867…+0.027118 m/s²**.
- **All1,263 combined changes are positive; none adds braking in this sample.**

This is direct evidence that the named braking offset has a weak, reversed *instantaneous mean-action pathway* at u6, even accounting for its network input. It is not a traffic ablation, does not hold future trigger history fixed, and does not establish universal behavior outside these rows. It strengthens, rather than merely assumes, the actuation diagnosis.

There is an important training-interface mismatch: production sampling adds `gain * active_offset`, while bank `forward()` omits this explicit gain term and ordinary bank contexts default the offset feature to zero. Continuous bank imitation therefore teaches a generic residual but does **not directly train the named-move gain**. Source: `track2_backend._distribution`, lines 161–163; `_features`, lines 139–143; inherited `coop_layer.forward`, lines 152–159. This is stronger evidence for an actuation/credit bottleneck than the single scalar KL0.014.

## 5. Merge features: present, used, but usefulness is not established

`merge_missing_share = 0.0` means only the extractor’s **own_missing** flag is zero. It is not complete partner/lane/traffic coverage. At u8, the verified figures are **114,199 CAV outcomes, 38,630 empty neighbourhoods, 7,704 unavailable partner reports**. Packet incompleteness must be judged from those fields rather than the own-missing scalar.

**Direct input-dependence check:** without changing weights or simulating, checkpoint u6 was evaluated on **3,060 recorded feature rows**, every fifth tick from the existing normal B/u6 worlds986004,986016,986028.

- Keeping all other inputs fixed and zeroing the29 merge columns changes the residual by mean absolute **0.256659 m/s²**, median **0.201689**, p95 **0.529302**.
- Permuting those columns jointly across recorded rows, seed61105, changes it by mean absolute **0.167753 m/s²**, median **0.139582**, p95 **0.452424**.
- 25 of29 merge columns were nonzero somewhere in this sample. All3,060 zero-block comparisons, and3,046/3,060 permutation comparisons, changed output.

Thus **the residual uses merge inputs**. Zeroing/permuting them can create inconsistent observations, so this is sensitivity evidence, not a valid traffic-performance ablation. It does not show that the layer uses the features intelligently. The categorical trigger consumes its separate local/V2V trunk and descriptors; it does **not directly consume this29-feature merge block**. Sources: `track2_backend._features`, lines135–147; `trigger_policy._hidden`, lines155–186; `launch_production.update_metrics`, lines198–207.

## 6. B versus A on comparable physical outcomes

The relevant metric is actual **source-inclusive team delay in veh-s per episode**, lower is better. It includes road delay plus waiting outside the road. Neither the printed shaped/global training reward nor B’s local cooperation reward is used for this comparison.

Two views are supplied:

- **All16:** each B normal episode paired with the same-world A episode, each B practice episode paired with its specifically recorded A continuation from the same practice snapshot/CRN. This preserves B’s mixture instead of comparing its six practice episodes with unrelated A normal worlds.
- **Normal10:** only the same-world normal episodes, to show whether practice weighting changes the story.

| Collection update | Matched A mean, all16 | B mean, all16 | B−A, all16 | B−A, normal10 | B−A, practice6 |
|---|---:|---:|---:|---:|---:|
| u1 | 83,544.45 | 83,544.45 | 0.00 | 0.00 | 0.00 |
| u2 | 82,940.72 | 84,677.86 | +1,737.14 | +90.31 | +4,481.85 |
| u3 | 79,054.14 | 79,445.99 | +391.85 | −110.87 | +1,229.72 |
| u4 | 82,300.11 | 83,444.74 | +1,144.63 | +2,543.69 | −1,187.15 |
| u5 | 81,787.72 | 81,532.27 | −255.45 | −491.02 | +137.18 |
| u6 | 77,343.16 | 78,999.18 | +1,656.03 | +1,613.21 | +1,727.39 |
| u7 | 83,776.08 | 83,719.55 | −56.53 | −860.84 | +1,283.99 |
| u8 | 81,983.93 | 80,913.31 | **−1,070.62** | **−868.57** | **−1,407.38** |
| u9 | 77,122.12 | 78,456.34 | **+1,334.22** | **+1,316.86** | **+1,363.15** |

Positive differences are worse for B. u8 all16 is **1.31% lower** delay than its matched A mixture; normal10 is **0.99% lower**. At finalu9, all16 delay is **1.73% higher**, with both normal and practice subsets worse. Results are mixed and not a monotonic learning curve. Worlds rotate across updates, A also learns, and uN collection precedes optimization of checkpointuN. Practice uses three duplicated contexts, not six independent draws. No significance claim is made from these training means.

The early isolated probes likewise do not show emerging synergy: A/B mean synergy at u2 is **−0.522354/−0.307193**, at u4 **−0.335052/−0.349375**. These are small diagnostic synergy scores, not overall delays. Existing safety telemetry also matters: u6 B’s critical-event excess against A is **+29.1875/episode**, CI **[−1.875,+61.3445]**. At u9 the paired composite excess is **+19.46875/episode**, CI **[+0.71875,+38.65625]**, despite zero collisions/teleports. That comparison originally paused B; `B_SAFETY_REPORT_ONLY_1434.json` records an explicit owner amendment admitting optimization with composite reporting only. Later durable owner STOP supersedes that recovery. The monitor's `CONTINUE` label is not evidence of passing the original composite test or authorization to restart.

## 7. Ranked causes and concrete remaining levers

The ranking separates established mechanisms from hypotheses. Expected directions are mechanistic expectations, not promised effect sizes. **No lever below was applied.** Clean1 is owner-stopped at u9; none is an instruction to resume it. Hyperparameter/action-space changes belong in an explicitly authorized continuation or separately labelled follow-up. Other workers' later clean2/plan-fix work is recorded in the shared status log and was not dispatched or audited by this analysis; check its existing receipts before undertaking any overlapping work.

### 1 — Weak and partly misaligned move-to-acceleration coupling / credit (highest confidence)

**Evidence:** zero physical move effect at u1; tiny negative explicit gain through u9; the complete u6 active-offset pathway raises rather than lowers acceleration in all1,263 sampled active contexts; bank sampling/forward interface mismatch; credit for rejected proposals; nonzero generic residual learning without demonstrated extra-over-solo benefit.

**Concrete lever:** in a short follow-up, give named moves a calibrated, positive, bounded execution pathway, or train imitation through the exact production distribution with recorded active offsets. The no-simulation separation of `residual(z,offset)`, `residual(z,0)` and the explicit gain term is now complete above. Compare a positive-gain/production-consistent version against the current coupling using the same ordinary training recipe. Record selected, accepted, active-offset and effective-action changes separately. Do not simply force full-strength walls without such separation.

**Expected effect:** make move-family choices affect the intended longitudinal behavior and give trigger PPO a more learnable consequence. **Risk:** a formerly weak intervention becomes strong; adverse braking/TTC and reversal of an adapted residual are plausible. Fixing production-consistent imitation can expose that the projected move is a poor substitute for the original native plan.

### 2 — Too few feasible joint opportunities; candidate/commitment bottleneck (high confidence)

**Evidence:** approximately83% easing slots before learning; u6 only37/109 wall proposals accepted; exact-test r4 accepted four moves but never produced a nonzero move offset during group exposure. Test groups often remain eligible only briefly. The policy selects parameterized candidates every60 seconds, with onset up to30 seconds.

**Concrete lever:** a follow-up with family-balanced **feasible** slates, reduced duplicate easing options, commitment-aware masking/arbitration, and onset/hold choices matched to observable remaining merge opportunity. Consider a targeted exploration bonus for accepted, actually actuated, feasible joint choices rather than a bonus for “any non-none.” Preserve correction for the changed sampling distribution.

**Expected effect:** more usable wall/gap/ramp experience per episode and less credit assigned to refused actions. **Risk:** forced family balance can create bad interventions where only individual easing is appropriate; a flat move bonus rewards useless activity. There is already entropy exploration, so simply increasing global entropy is poorly targeted.

### 3 — Bank/practice coverage and action-representation mismatch (high confidence)

**Evidence:** seven plans/eight seed examples, three CE contexts, three repeated practice starts; 92.5% example-weight mass from legacy u3; native rich profiles projected to restricted wall/none vocabulary; only one test world/time overlaps the admitted bank and none overlap the practice set.

**Concrete lever:** first widen the *usable* training-context pool using permitted ordinary training data and distinct verified opportunities, including realistic wall/gap/ramp parameters and negative/none examples with state coverage. Expand observable/representable contexts before raising practice share. If retaining only three contexts, raising the repeat cap can increase practice, but should be explicitly treated as repetition rather than diversity. Do not add the diagnostic panel as a targeted training set.

**Expected effect:** better transfer and less memorization; more chances to experience delayed benefits. **Risk:** new labels require verification cost, and a larger unbalanced bank can dilute good examples. More repetitions may overfit the same three worlds. Augmentation is currently OFF; enabling it is a substantive follow-up, not an existing free capacity increase.

### 4 — Local reward / departure-limited credit does not directly reward network synergy (moderate-to-high confidence)

**Evidence:** objective is own plus local neighbour mean, while the claimed endpoint is improvement over best solo in source-inclusive traffic delay; CAV departure terminates its return; wall mean advantages are mostly negative; explicit move effect is weak. Raw negative reward alone is not proof of a defect.

**Concrete lever:** a short follow-up using a well-scaled state baseline for continuous PPO and improved delayed observable team credit; evaluate whether useful cooperative consequences persist after the acting CAV leaves. Where existing paired ordinary-training evidence permits, compare incremental outcomes rather than only absolute congestion costs. Log advantage by selected/accepted/actually-actuated family and by opportunity regime. Merely increasing the neighbour weight is not a demonstrated fix.

**Expected effect:** lower-variance, better-aligned credit for sacrificing briefly to help others. **Risk:** global/delayed labels can be noisy or hard to attribute; reward redesign changes the estimand and must not be confused with the current registered recipe. WM group credit/teacher are currently disabled, so their absence cannot be interpreted as failed WM credit learning in this run.

### 5 — Continuous PPO takes only four small, unnormalized-return steps (moderate confidence)

**Evidence:** all PPO steps accepted below cap; small gradients; u6 KL0.0139 versus0.045. The layer has nonzero outputs and growing residual norms, so lack of capacity is not the leading explanation.

**Concrete lever:** after correcting signal/actuation, test **2× layer LR (3e−4→6e−4)** or more PPO epochs under the **existing0.045 PPO cap**, with independent PPO/bank monitoring. A trust-region-targeted PPO schedule is more direct than raising a cap that never rejects PPO steps.

**Expected effect:** quicker on-policy adaptation if the direction is informative. **Risk:** faster optimization of the wrong/noisy signal, oscillation with the subsequent bank step, greater critical-event exposure. Increasing the shared KL cap first mostly loosens already-binding bank imitation and is low priority.

### 6 — Too early / too few distinct experiences, rather than exhausted budget (moderate confidence)

**Evidence:** only nine completed updates before owner STOP; CE and residuals still change; the original primary criterion was u24; the u6 estimate has five replicates and is dominated by one opportunity. The stop-era accounting snapshot is1.218M, not0.5M; reservations and subsequent work prevent treating its apparent remainder as freely available.

**Concrete lever:** preserve clean1's finalu9 evidence. More updates could help an explicitly authorized continuation or a new follow-up, but the durable owner STOP means there is **no current instruction to continue this run to u24**. Do not select a “best” checkpoint on this descriptive u6 score. Consider >24 updates only as a predeclared extension with reserved evaluation funding and distinct training opportunities; first account for already-owned clean2/plan-fix allocations.

**Expected effect:** more opportunity for the slowly changing wall preference/residual to improve if the signal and exposure are useful. **Risk:** repeated weak-credit training and three-context practice may simply converge to a generic acceleration tweak. Additional A/B/R training at16×1200s each plus six matched A references for B and R is approximately **72,000 sim-s per extra full update**; this is an estimate, not a quote or dispatch. The original canonical4.31022M plan and5M ceiling do not authorize arbitrary extra updates or make the dated accounting remainder unallocated.

### 7 — Imitation strength, layer width, and lane-changing (lower-priority or larger follow-up)

- **Imitation weight:** increasing it alone is low priority because continuous imitation already nearly exhausts its own KL cap and CE fits three contexts. Better first test rebalanced per-context/per-family weights or moderate reduction of concentrated legacy influence while retaining the identical targets. Expected benefit is less bank dominance; risk is forgetting genuinely useful plans. Loss magnitudes alone cannot choose the coefficient.
- **Layer width:** no capacity-limit evidence was found; merge inputs change outputs and residual magnitude grows. A wider layer adds fitting power but does not fix poor actuation, feasibility, or labels. Risk is more overfitting.
- **Variety/wall plans:** more *feasible and representable* wall/gap/ramp plans is a strong coverage lever under cause3, not merely “more wall labels.” Validate the projected move’s effective physical action, not just its native-plan provenance.
- **Lane-change option:** the current added cooperation pathway is longitudinal; a learned lateral coordination option could access genuinely different merge opportunities. This is a separate short research follow-up with new action semantics, mechanical validation and safety comparison. It is not a small scalar tweak to rescue the ongoing run, and current receipts do not establish that lack of lane control is the main cause.

## Bottom line for the owner

**B is not stuck at none and is not inert. It is learning a small, merge-dependent residual and a slowly more wall-preferring trigger, but the named cooperative moves have weak, reversed instantaneous actuation, sparse feasible joint exposure, concentrated supervision, and noisy/misaligned credit.** The diagnostic additionally subtracts the best solo benefit, so large useful individual improvements can coexist with almost zero cooperation score. Preserve clean1's owner-stopped finalu9 evidence; prioritize actuation/credit consistency and feasible diverse practice in a clearly labelled follow-up. A blanket increase in KL or imitation weight is not the evidence-led first move.
