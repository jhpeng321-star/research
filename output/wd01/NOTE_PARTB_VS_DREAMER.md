# Note: current Part B design vs a DreamerV3-style RSSM + critic (for the owner's Part B decision)

WD-01, 2026-09-26. Design text only; no experiment behind it.

## What each approach estimates

| | Current Part B (DESIGN §3) | DreamerV3-style |
| --- | --- | --- |
| Target | Paired `ΔR = R_plan − R_native` and `R_native` over the rest of the 10-min horizon. Supervised from branch outcomes; every branch point has a native sibling with the same arrival seed. | Critic `V(z)` = expected discounted return from latent state z **under the current learned policy**. Trained by TD(λ) on imagined rollouts of an RSSM. |
| State | Explicit end-of-Part-A state: lane-bin field + CAV/wall tokens + demand summary | Learned latent belief (RSSM: deterministic GRU state + stochastic categorical latent) |
| Action handling | A plan is an input (per-CAV actions + wall tokens), held to the horizon | A fixed-size action vector from an actor; imagination trains the actor |
| Uncertainty | Ensemble × reconstructor samples × arrival draws, **calibrated over M×K** as the contract requires | Mostly aleatoric (twohot distribution); epistemic uncertainty is not native to Dreamer |

## Fit to our constraints

1. **The planner needs a plan-vs-native difference, not a policy value.** The contract compares each candidate plan with native on the same samples. That pairing (common random numbers) is what makes the ≥ 80% agreement rule and the margin work. A Dreamer critic values "whatever the actor does next". We would need a plan-conditioned Q(s, plan) plus a native critic, and we would lose the shared-noise pairing unless we rebuild it. **Advantage: current design.**
2. **Hard safety limits.** Safety must be checked on explicit per-vehicle trajectories (collision, min TTC, exposure), and it must fail closed. An RSSM latent is not auditable, and decoding it back to identified vehicles is the hard part Dreamer avoids. Part A must stay explicit either way. Dreamer could at most replace Part B, and never the safety gate.
3. **Per-CAV, variable-count, group actions.** Dreamer assumes a fixed action space. Our actions are a set: per-CAV choices plus jointly selected wall groups, with a varying number of CAVs. That needs permutation-invariant action tokens. It is feasible but new, and untested in any Dreamer work.
4. **Offline campaign data.** Dreamer is online RL: its actor exploits model errors, which the environment corrects. Our data is a fixed campaign, with planner-driven branches later. Offline, imagination-trained policies drift out of distribution unless we add conservative penalties (MOPO/COMBO-style). Candidate *evaluation* by the planner has a milder version of this problem (the optimizer's curse), which the uncertainty margin already targets.
5. **Compute.** An RSSM with an actor-critic is heavy on CPU. It is fine on a 3060 at modest size. The current Part B (1–3 M parameters, supervised) trains on CPU.
6. **Data efficiency.** Dreamer's TD bootstrapping reuses states along a trajectory, which is a real advantage when labels are expensive. Every Part B label costs a 600 s branch pair.

## Parts worth adopting now (low risk, no change of estimand)

- **symlog targets + twohot distributional heads** (or quantile heads) for `R_native` and `ΔR`. Delay is heavy-tailed and its scale varies strongly with demand (2000 vs 2700 veh/h). A distributional head gives the per-cell quantiles the 80% rule and the margin need.
- **Return normalisation per demand regime** (percentile scaling, as in DreamerV3) for stable training across loadings.
- **TD-bootstrapped auxiliary for `R_native`.** The native policy is fixed, so a horizon-conditioned value `V_native(s, τ_left)` can be trained with multi-step bootstrapping along native trajectories. That makes the cheapest branches (native) much more label-efficient. `ΔR` stays supervised and paired.
- **A learned state encoder (belief)** in place of hand features, trained supervised on the Part B targets. It could later be shared with the reconstructor's sampled sets. This takes Dreamer's representation idea without the RL loop.

## Worth a later research branch (not the decision path)

- **Imagination-trained per-CAV proposal policies**, used only to *propose* candidates. Every proposal still passes the explicit Part A safety gate and a Part B paired evaluation against native. This can cut the planner's search cost. It cannot relax safety, because it never decides.
- **A full RSSM world model**, compared with Part A+B on owner priority (1): extra delay of planner decisions vs the perfect simulator.

## Recommendation

Keep the supervised, paired, plan-conditioned Part B as the decision estimator. Adopt in its v1:
- symlog/twohot (or quantile) heads;
- per-regime normalisation;
- a TD-bootstrapped native-value auxiliary.

Treat Dreamer-style imagination policies as an optional proposal generator. Start it after Part A is action-conditioned on campaign data, and evaluate it by decision regret against the perfect simulator. None of this changes the predict contract.
