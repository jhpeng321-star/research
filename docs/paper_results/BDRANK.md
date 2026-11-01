# S-BDRANK — rank actions, not individual traffic futures
**Answer: predicted mean delay ranks useful actions better than predicted breakdown percentage on development; probability saturation largely removes the latter’s ranking information.**
**Do not switch to breakdown-first on this evidence:** all development-selected rules lose to MAPPO on the separate reserved panel; the caches do not identify a large causal cost of avoiding this detector’s event.

## Contract and scope
- Recorded cost: WMDATA969400–969431 development32, separate969432–969439 evaluation8; WMDATA2 replication969550–969599 (50 worlds, six controllers). 500 episodes total.
- Controller forecasts: complete FULL-v4/22cc32f6, 20 worlds ×5 controllers ×K8; development12 (969420–969431) and reserved8 separate. WMTEACH: fixed12 diagnostic states, v4/v5 each96 cached futures, H120/H300.
- Exact registered LRDIAG BIC detector SHA `6de4df03…`, 10s full-traffic region speeds/60s prehistory; upstream primary, merge/ramp CSV diagnostics. No speed thresholds. Pre-existing steps separate; horizon-specific dominant steps are not nested first-breakdown events.
- J includes road delay **and source waiting**, in veh-s; positive saving means MAPPO minus chosen J. Each horizon uses the same-horizon recorded outcome. A detected speed-loss step is not proof of a phase transition.
- 95% CIs use10,000 fixed-cell whole-world bootstrap draws, seed20261101. A singleton contributing demand×share stratum gives **NA**, including pooled controller/option claims here; replicated-cell CSV intervals remain scoped diagnostics.

## Q1 — how expensive is recorded breakdown?
- Upstream development event/no-new-event mean J: H120 7,591/7,608; H300 23,000/17,605; H600 54,010/51,878; pre-existing cases excluded from these two statuses.
- Replication event/no-new-event mean J: H120 9,127/9,118; H300 28,451/24,572; H600 72,433/64,261.
- Evaluation8 separately: H120 4,997/6,471; H300 20,191/39,600; H600 55,857/104,824; at600s only ONE no-event outcome, versus38 event/one pre-existing.
- These pooled differences are confounded: cell×controller-standardized event-minus-no-event J300/J600 is −3,373/−5,457 in development, +2,959/+15,414 in replication. Only52/20 and193/96 episodes respectively support those contrasts; no causal avoidance-cost claim.
- Within-world ALL-controller-pair regression, excluding pre-existing pairs: occurrence / timing / depth / unexplained shares of squared J600 differences are development **2.8%/4.1%/14.8%/78.3%**.
- Replication same decomposition: **1.6%/4.6%/13.8%/80.1%**; evaluation8 separately: **4.8%/1.2%/22.4%/71.6%**. These are order-independent Shapley allocations of in-sample R², not causal mediation.
- Development combined R² at120/300/600 is7.5/15.1/21.7%; leave-world-out R² −2.9/7.6/16.9%. Replication600s:19.9% fitted/17.9% leave-world-out. Most controller variation remains unexplained.
- Conditional paired600s depth slopes: +1,264 development/+1,191 replication veh-s per m/s deeper fitted step. Timing and severity matter more than occurrence in this linear association; depth is not queue length.
- Confirm WMBREAKDOWN narrowly: development600s upstream MAPPO/PI/CIRCLES shifts vs native or CTRL0 are +3.125/−3.125/+3.125pp, despite MAPPO/PI J gains486/1,498 vs CTRL0.
- The ±3pp statement is not universal: replication300s CIRCLES is +16pp vs native/+18pp vs CTRL0; evaluation600s all three are +12.5pp. All cells, references and horizons are exported.

## Q2 — which ranking signal is right?
- Below: tie-adjusted concordance against **real delay ordering**, %, H120/300/600; predicted ties earn½, factual ties are excluded. Untied concordance and all three tie classes are exported separately.
- Development: probability 57.1/48.3/50.8; later onset 46.7/50.0/59.2; lower depth 50.8/44.2/54.2; **mean delay 62.5/57.5/68.3**.
- Reserved8 separately: probability 43.2/58.1/50.0; onset 59.5/65.0/50.0; depth 43.2/53.8/60.0; mean delay 60.8/60.0/65.0.
- Matching-signal untied fidelity, development: event 60.0/72.7/0.0; onset 46.7/50.0/52.6; depth 45.1/50.6/54.6%. These score the matching factual signal, not delay.
- Matching-signal reserved fidelity: event 14.3/12.5/0.0; onset 55.6/50.0/56.7; depth 54.3/52.6/57.5%. Event600s has only1 development/3 reserved doubly-untied pairs: 0% is extremely sparse.
- Saturation: probability ties on112/120 development and68/80 reserved600s controller pairs; against real delay its untied score62.5%/50% hides93.3%/85% tied predictions. Mean delay has no predicted ties and gets82/120 and52/80 right.
- WMTEACH available O0–O4 delay-pair concordance: v4 **65.5/75.9%**, v5 **63.8/51.7%** at120/300s (58 pairs/horizon/version, zero prediction ties). Unavailable options are excluded by the inherited availability mask.
- Exact-choice agreement v4 50.0/66.7%, v5 16.7/33.3%. These are roster-only/source-prior/cold-memory diagnostics, one state per cell; neither a live-policy test nor student fitting data.

## Q3 — fixed choice rules, development first
- Rule order (a–e): lowest P(event); latest restricted onset; lowest expected detected depth; lowest mean delay; lexicographic P(event)-then-delay. Numbers below are **saving/regret** per world, veh-s.
- Onset score integrates detector-onset survival to H: new onset t, no-event censored at H, pre-existing0; it does not impute an exact unseen onset. Depth=E[event × fitted depth], zero without a new event; conditional onset/depth are separately exported.
- Development120: +4/95; -91/190; -6/104; -43/141; -52/150.
- Development300: +50/522; -745/1,317; -781/1,354; +230/342; -81/653.
- Development600: +278/2,286; +391/2,173; -3/2,567; +1,695/870; +1,250/1,314.
- Reserved120: -14/187; -28/201; -67/240; -2/175; -17/190.
- Reserved300: +21/793; +464/350; +147/667; -393/1,207; -222/1,037.
- Reserved600: +0/2,053; -1,395/3,448; -1,198/3,251; -480/2,533; -108/2,161.
- Development selects probability at120, delay at300/600; frozen before constructing reserved choices. Confirmation savings **−14/−393/−480**, regrets187/1,207/2,533. No reserved rule is selected.
- Exact prediction ties prefer MAPPO, then native/CTRL0/PI/CIRCLES. Probability ties at the minimum in9/10/10 of12 development worlds; its modest savings depend heavily on fallback. CSVs include full tie sets and best/worst tie-choice ranges.
- This is hypothetical selection among the five recorded controllers, not a validated online policy; no P(win) gate is added. Delay ranks best on development at every horizon, but ranking accuracy alone does not guarantee positive selected-action saving.
- Option delay-only chosen saving/regret: v4 H120 +19.5/40.2, H300 +680.3/593.0; v5 −28.2/87.9 and +172.7/1,100.6. O0=MAPPO; diagnostic exact labels never select controller rules.

## Stopped items and evidence
- **Duration and queue extent STOP:** absent from the registered definition/output; no threshold or time-to-horizon substitute. Real reductions/defining a duration metric need0 GPU-min. If no separate spatial archive exists, controller queue extent needs160 five-controller H600 forecasts saving position/speed/lane/presence: approximately688.1 occupied GPU-min plus instrumentation, after a definition is registered; not run.
- **Option breakdown/onset/severity and rules(a,b,c,e) STOP:** read-only audit of all192 cached v4/v5 futures finds scalar delay and `breakdown=None`, no regional speed/prehistory; supplied exact option labels likewise contain delay effects only.
- Needed, not run:96 paired H300 option futures/version saving registered10s speed bins + valid60s observable prehistory. Observed task totals estimate **71.5 GPU-min v4 +91.8 v5 =163.3**, excluding instrumentation overhead. Exact detector labels additionally require archived exact trajectories; if absent, new SUMO would be required and is outside this task.
- Round2 controller-v5 panel remains pending in frozen16:16:43 receipt:7/216 complete world jobs across models. Only completion metadata read; no partial-panel efficacy, redispatch or peer-job interaction.
- [CSV/evidence index](bdrank/README.md) · [analysis definitions/reproduction](../../tools/bdrank/README.md) · [input/task SHA manifests](bdrank/input_manifest.json) · [verification](bdrank/verification.json).
- Analysis rechecks11,700 registered detector outputs and all source-inclusive clock sums; independent verification checks source hashes, pair/tie arithmetic, decision/regret identities and paired-regression equivalence to within-world fixed effects. No SUMO, training, new inference or process signals.
- Retrospective, sparse, historically exposed panels: association and ranking diagnostics only. A lower predicted breakdown percentage is not presently a demonstrated better decision rule.
