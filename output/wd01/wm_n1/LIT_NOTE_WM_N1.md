# Literature note for WM-N1 (neural traffic / trajectory world models)

WD-01, wm_n1 piece (d), 2026-09-30. **Document only.**

Purpose: inform `WM_N1_DESIGN.md`. This note reads the neural
trajectory-forecasting, world-model, causal-inference and uncertainty
literature against the WM-N1 design (sections 2–6) and flags places where the
literature would argue for a design change. It does **not** change the frozen
`REGISTRATION_WM_N1.md`; any proposed change is a flag for the owner, not an
edit.

**Citation honesty.** There is no paper-search tool in this session. Every
reference below is given from memory as author(s) + year + title; a reference is
marked **verified-in-repo** *only* if a file inside this workspace states it in
full, and otherwise carries **(to verify)**. No DOI, arXiv id, page number or
venue is asserted unless a repository file already states it. Where I am unsure
of a title or year I say so rather than guess. Because almost all of the
neural-forecasting literature is *not* in this repo, the citation table in §8 is
dominated by **(to verify)** entries; that is expected and is the honest state
of evidence, not an omission.

---

## 1. Attention / GNN multi-agent trajectory predictors

### What the literature does

- Early social reasoning used **social pooling**: nearby agents' LSTM hidden
  states are pooled into a grid and fed to each agent's decoder (Alahi et al.
  2016, *Social LSTM* **(to verify)**). The key idea is *permutation-invariant
  neighbour aggregation* rather than a fixed input slot per neighbour.
- **Social GAN** (Gupta et al. 2018 **(to verify)**) replaced pooling with an
  attention/relation module plus a variety-reward generator, showing that
  multi-modal outputs and social interaction are separable concerns.
- **Graph and attention encoders** became standard: graph attention over a
  radius graph (Veličković et al. 2018, *Graph Attention Networks* **(to
  verify)**), agent–agent and agent–map relation modules (e.g. Trajectron++,
  Salzmann et al. 2020 **(to verify)**), and vectorized scene encoders that
  treat lanes and agents as graph nodes with relative-geometry features
  (**VectorNet**, Gao et al. 2020 **(to verify)**; **LaneGCN**, Liang et al.
  2020 **(to verify)**).
- **Transformer motion forecasters** (e.g. the Waymo Open Motion Dataset
  baseline, Ettinger et al. 2021 **(to verify)**; the original transformer in
  Vaswani et al. 2017 **(to verify)**) use relative-position/geometry terms as
  *attention biases*, and output either a set of trajectory anchors or
  *per-timestep query* trajectories rather than a free autoregressive rollout.

### What transfers to a 1-D multilane highway with ~100 vehicles

- **Permutation-invariant token + attention over a radius graph is the right
  shape for us.** Our design §2 already does this: per-vehicle MLP token, L=4
  lane-aware relative-attention layers over a ±150 m neighbourhood with
  relative-geometry attention biases (Δs, Δlane, Δv). This is directly
  consistent with the vectorized/relative-bias line above, and it scales
  linearly in the number of vehicles because each car only attends to a local
  window. With ~100 vehicles on a 1-D corridor, local attention is not a
  compromise; the interaction range is genuinely local, and the merge zone
  (~100 m) is inside the 150 m window.
- **Explicit relative features matter more than raw positions** at this scale;
  the design's (Δs, Δlane, Δv) biases are exactly the features that GNN and
  lane-graph work found useful **(to verify)**.
- **Lane structure should be a graph, not an image.** We have 3 physical lanes
  plus a merge; the design's one-hot lane + adjacency in the attention mask is
  the cheap analogue of LaneGCN's lane graph **(to verify)**. The merge edge
  (main_right → accel, and back) is the only "topology" that carries real
  signal, so it deserves an explicit adjacency entry rather than being inferred.
- **Plan and wall tokens as queries over CAV/member tokens** (design §2) is
  consistent with the "query attends to relevant agents" pattern; the
  wall members form a small set that a wall token must pool.
- **Risk / caution.** Trajectory-prediction encoders are usually evaluated on
  <8 s horizons; our step rollout is 0.5 s × many steps (up to 570 s for
  outcomes). The encoder design transfers; the *evaluation regime* does not.
  The literature does not support trusting a step rollout for 570 s (see §2).

**Mapping:** supports design §2 encoder as-is. No change proposed, except the
note that 4 layers over ≤128 tokens is well within the "small model" regime
(2–5 M params) and is unlikely to be the bottleneck; the decoder/recurrence is.

---

## 2. Stochastic step decoders vs direct multi-horizon outcome heads

### What the literature does

- **Mixture / multi-modal step decoders.** Mixture density networks (Bishop
  1994, *Mixture Density Networks* **(to verify)**) parameterize a per-step
  distribution (e.g. a Gaussian mixture). Trajectory forecasters refine this
  into mixture-of-2D-Gaussians per timestep (Social GAN **(to verify)**) or
  categorical latent modes over whole trajectories.
- **Quantile / proper-scoring regression.** Quantile heads trained with the
  pinball (quantile) loss go back to Koenker & Bassett 1978 **(to verify)**,
  and the reason quantiles are the right target — proper scoring, CRPS — is
  Gneiting & Raftery 2007 **(to verify)**.
- **Direct multi-horizon ("one-shot") heads** predict several horizons at once
  instead of rolling out; this is standard in forecasters that report
  minADE/minFDE at multiple horizons and is often the more accurate head for
  long horizons because it does not compound one-step error **(to verify)**.
- **Exposure bias / drift.** Autoregressive decoders trained with teacher
  forcing suffer train–inference mismatch ("exposure bias"); scheduled
  sampling (Bengio et al. 2015 **(to verify)**) anneals from ground-truth to
  model inputs during training, and multi-step / rollout losses optimize the
  quantity that is actually used at inference. Even so, error accumulates
  super-linearly with horizon in stochastic simulators **(to verify)**.

### What transfers

- **The design's split — a stochastic step decoder for rollout, plus direct
  multi-horizon outcome heads evaluated in one forward pass — is exactly the
  right hedge.** The literature's lesson is that one-shot multi-horizon heads
  are more reliable for the long horizon, while a step decoder is needed for
  short-horizon physical checks and for the validity layer. WM-N1 does both.
  Keep them in one trunk but do **not** route the τ-outcome heads through the
  rollout; the design already avoids this (§2.4).
- **K = 3 Gaussian mixture on acceleration** (design §2.2) is a mild
  multimodality assumption. At a merge, "brake / hold / accelerate" plus a
  lane decision is roughly 3 longitudinal modes, but the lane head and the
  acceleration mixture should be jointly conditioned (an acceleration mode
  that only exists in the merge lane must not be emitted on the mainline). The
  design's factorized 3-way categorical + K=3 mixture can leak; consider a
  small joint categorical over (lane, accel-mode) **as a flag**, not an edit.
- **19 quantiles for outcomes** (design §2.4/§3) is standard and compatible
  with a pinball loss. For a 10–90 % coverage target you need well-calibrated
  tail quantiles; 19 levels are plenty in the interior but the extreme tails
  (for the 10–90 rule, the 0.10/0.90 levels) are the ones that matter, and the
  design's conformal step (§3/§5) is what should carry their calibration.
- **Drift remedies are already in §3:** scheduled sampling on 20-step (10 s)
  rollout windows, and multi-step pathwise loss on positions/speeds. The
  literature supports both. The remaining risk is that the *outcome* head is
  trained per-τ with no dependence on the rollout, so the reported short-
  horizon s_RMSE (rollout) and the τ-outcome CRPS (one-shot) come from two
  differently-behaving heads; that is fine but should be stated so the
  evaluation is not read as "one model is good at both".

**Mapping:** supports design §2.2/§2.4 and §3 data units (b); flags the
factorized lane/accel head and the dual-head attribution.

---

## 3. Learned traffic simulators / world models for planning and control

### What the literature does

- **Latent world models.** Ha & Schmidhuber 2018, *World Models* **(to
  verify)**, and PlaNet / DreamerV3 (Hafner et al. 2019, 2023 **(to verify)**)
  learn a latent dynamics model (RSSM: deterministic recurrent state + stochastic
  discrete latent) and train a policy/critic in imagination. DreamerV3 adds
  symlog transforms, two-hot distributional heads and per-domain return
  normalization for stability across reward scales **(to verify)**.
- **Offline model-based RL.** MBPO (Janner et al. 2019 **(to verify)**)
  branches short rollouts from real states; MOPO (Yu et al. 2020 **(to
  verify)**) and COMBO (Yu et al. 2021 **(to verify)**) penalize
  out-of-distribution model exploitation with uncertainty / conservative value
  estimates. The recurring finding: *a learned model is only as good as its
  coverage, and policy optimization against the model exploits model error*.
- **Learned microsimulators.** TrafficSim (Suo et al. 2021 **(to verify)**)
  learns realistic multi-agent driving behavior from data. The lesson is that
  learned simulators need rollout-stability tricks and are validated on
  distributional/behavioral statistics, not just one-step error.
- **Physics-informed hybrids.** IDM (Treiber, Hennecke & Helbing 2000,
  *Congested traffic states in empirical observations and microscopic
  simulations*, arXiv:cond-mat/0002177 — **verified-in-repo**, cited in
  `archive/pre_circles/docs/TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md`)
  and MOBIL (Kesting, Treiber & Helbing 2007 **(to verify)**; "MOBIL" named in
  `output/wd01/PARTA_V2_GENERAL_DESIGN.md` but without a full bibliographic
  entry) give a strong deterministic prior. Physics-informed neural networks
  (Raissi et al. 2019 **(to verify)**) show that a physics operator should be
  embedded as a prior, with the network learning the **residual**. Hybrid
  IDM/MOBIL + learned residual is the natural, data-efficient choice when the
  label budget is expensive (each branch pair costs 600 s of simulation).

### What transfers

- **WM-N1 is a *plan-conditioned*, supervised world model, not an
  imagination-trained RL agent.** The design trains on paired branches from a
  fixed campaign. That avoids the MOPO/COMBO offline-exploitation failure mode
  (no actor is trained against the model; the planner only *evaluates*
  candidates), which the existing repo note
  `output/wd01/NOTE_PARTB_VS_DREAMER.md` already argues (that note gives no
  full references, so it is context, not a verified citation source).
- **The design's step-dynamics prior ablation (IDM/MOBIL acceleration as an
  input, network learns a residual)** is the physics-informed hybrid, and the
  literature supports it as a data-efficiency move. Because our parameters are
  one nominal Treiber set (design/Part A), the residual is the only learned
  part; this is the safest "hybrid" and should be the default *if* the pure
  model's short-horizon s_RMSE underperforms Part A v2g — the design's §6
  decision rule already measures exactly this.
- **Dreamer-style latent imagination is not the decision path.** It values
  "what the current actor does next", loses the paired same-seed contrast the
  planner's ΔV needs, and its latent is not auditable for the explicit safety
  gate. The repo note reaches the same conclusion. Keep it as an optional
  proposal generator only.
- **Learned-simulator validity is a distribution/behavior question.**
  WM-N1's validity layer (interventions per veh-step, collisions per veh-step,
  arrival-count error) is the right kind of check, but "count of projections"
  must not be read as "model is physically valid" — a low intervention count
  can also mean the model never approaches a constrained state.

**Mapping:** supports design §5 residual/physics-prior ablation and §6
validity metrics; endorses the supervised (non-RL) choice.

---

## 4. Causal / treatment-effect and contrast learning

### What the literature does

- **Potential outcomes / individual treatment effect (ITE)** (Rubin 1974 **(to
  verify)**) frames the target as a *difference* between outcomes under two
  treatments on the same unit. Estimating heterogeneous effects with learned
  models is done via meta-learners (S-, T-, X-learner; Künzel et al. 2019
  **(to verify)**) or causal forests (Wager & Athey 2018 **(to verify)**).
- **Paired / common-random-numbers (CRN) design.** Variance reduction by
  coupling the same random draws across treatments is classical Monte Carlo
  (common random numbers; e.g. Glasserman 2003 **(to verify)**). For our
  simulator, branch runs that share the arrival seed are exactly a CRN design,
  and training the *difference* cancels shared noise, so the contrast label is
  far lower-variance than either marginal.
- **Why a contrast loss helps.** A loss on the marginal outcome must fit the
  large, demand-driven mean; the ΔV signal is the small residual after that. A
  loss applied directly to the paired difference (a) matches the planner's
  decision quantity, and (b) inherits the CRN variance reduction, so the same
  label budget yields a sharper ΔV than post-hoc differencing of marginals
  **(to verify)**.

### What transfers

- **The design's PAIRED data unit and contrast loss (design §3, λ_c = 1) are
  well motivated.** The comonotone quantile pairing (same quantile level from
  both arms, then difference) is a reasonable way to make a distributional
  contrast; note that "comonotone coupling" gives the *difference of
  quantiles*, not the quantiles of the difference — those coincide only under
  strict comonotonicity, which real branches need not satisfy. This is a subtle
  point the design should state: the paired pinball loss is being applied to a
  coupled quantile *difference*, and its calibration should be verified by the
  same per-τ conformal step and the sign-accuracy metric (§6).
- **Alternative (flag):** a two-model / S- or T-learner baseline is what the
  hybrid effectively is; the contrast loss is the X-learner-ish improvement.
  Keeping all three on CAL is consistent with the design's N/T/C comparison.
- **Confounding.** Because arms are planner-chosen *from the same world and
  t0* with a shared arrival seed, the pairing removes world-level confounding.
  This is a genuine strength relative to observational traffic data, where arm
  assignment correlates with traffic state **(to verify)**.

**Mapping:** supports design §3 paired samples + contrast loss and §6
contrast metrics. Flags: state explicitly that the contrast is a coupled
quantile difference; make sure the sign-accuracy CI is world-bootstrapped (it
is).

---

## 5. Uncertainty

### What the literature does

- **Epistemic uncertainty** is commonly approximated by deep ensembles
  (Lakshminarayanan et al. 2017 **(to verify)**), MC dropout (Gal & Ghahramani
  2016 **(to verify)**), or their combination; ensembling is generally the
  stronger and more robust choice **(to verify)**. Kendall & Gal 2017 **(to
  verify)** separates aleatoric (data noise) from epistemic (model) terms.
- **Aleatoric** uncertainty from a distributional head is captured by the
  sampling / quantile / mixture outputs.
- **Calibration.** Split conformal prediction (Vovk et al. 2005; Shafer & Vovk
  2008 **(to verify)**; Lei et al. 2018 **(to verify)**) gives finite-sample
  marginal coverage under **exchangeability**. CRPS and the pinball loss are
  strictly proper scoring rules (Gneiting & Raftery 2007 **(to verify)**);
  PIT histograms and coverage are the standard calibration diagnostics.
- **Distribution shift.** Conformal guarantees degrade under covariate shift
  unless reweighted or adaptive (Ovadia et al. 2019, *Can You Trust Your
  Model's Uncertainty?* **(to verify)**); dataset-shift framing is Quiñonero-
  Candela et al. 2009 **(to verify)**.

### What transfers

- **The design's 5-member deep ensemble matches the recon choice and is
  adequate** for epistemic uncertainty at this scale; more members help mostly
  the tail. Deep ensembles are known to be strong but *not* perfectly
  calibrated, which is why the per-τ conformal step is needed.
- **Conformal on CAL is only valid if TEST is exchangeable with CAL.** WM-N1's
  TEST is a **fresh chunk 5** generated after the fact (design §4), i.e.
  different worlds and possibly a shifted seed/parameter distribution; split
  conformal's coverage guarantee technically does not transfer across such a
  shift **(to verify)**. **Flag:** report empirical coverage on TEST anyway,
  and treat any coverage shortfall as evidence of shift rather than as a bug.
  Do not silently relabel TEST as if it were CAL. (This is a reporting flag;
  the frozen registration already commits to reporting coverage and PIT and
  already forecloses tuning on TEST, so no registration change is needed.)
- **Heavy-tailed delay.** Delay is heavy-tailed and scale varies with demand
  (2000 vs 2700 veh/h); the design's per-τ quantiles + conformal + symlog-style
  normalization concern are the right tools **(to verify)**. CRPS is scale-
  sensitive, so blending weights and CRPS comparisons should be read per
  demand regime or normalized, or the high-demand worlds will dominate.
- **Rare events.** Collisions are near-zero; a logit + Brier/collision-rate
  check is a diagnostic only. Proper scoring for rare events is unstable; the
  decision rule's collision term (≤ T + 0.1/1000) is the safer aggregate.

**Mapping:** supports design §3 uncertainty and §6 coverage/PIT; flags the
exchangeability assumption behind CAL→TEST conformal.

---

## 6. Stacking / model averaging of predictive distributions, and disagreement

### What the literature does

- **Linear pools / forecast combination** (Bates & Granger 1969 **(to
  verify)**) average predictive distributions with weights. Stacked
  generalization (Wolpert 1992 **(to verify)**) fits combination weights on
  held-out data to avoid overfitting. For probabilistic forecasts, **CRPS is a
  proper score and the linear pool's CRPS-optimal weights can be fitted by
  minimizing CRPS** (Gneiting & Raftery 2007 **(to verify)**); when weights are
  fitted on the same data used to choose them, cross-fitting (like the design's
  world-grouped 2-fold) is required to avoid optimistic weights **(to
  verify)**.
- **Disagreement / abstention.** Ensemble or inter-model disagreement as an
  uncertainty signal and a reject option dates to Chow's rule (Chow 1970 **(to
  verify)**) and is used as an OOD/abstain cue in modern ensembles **(to
  verify)**. It is useful only if the members' errors are sufficiently
  decorrelated.

### What transfers

- **The design's stacking (F_C = w_τ F_N + (1−w_τ) F_T, CRPS-minimizing w_τ
  with world-grouped 2-fold cross-fitting, then conformal) is exactly the
  literature-recommended procedure.** Linear pooling of CDFs (the "linear
  pool") is the standard combination operator; CRPS is a proper score, so
  minimizing it on held-out CAL is defensible.
- **Disagreement D_τ = CRPS-distance(F_N, F_T)/scale** as a planner
  abstain signal is reasonable, but note two failure modes: (i) if N and T
  share a systematic bias (e.g. both understate merge delay), disagreement
  stays low exactly when both are wrong; (ii) the CRPS-distance is not a
  calibrated probability, so exposing it as "uncertainty" should be paired with
  the calibrated ensembles. The design already reports the Spearman correlation
  of D with realised |error| and an abstain curve — keep that as the
  empirical justification, not an assertion.
- **Weight degeneracy.** With only N and T, w_τ ∈ [0,1] can collapse to 0 or 1
  (choosing one model), which is fine and is itself a result; the 2 % tie rule
  in §5 is a pragmatic guard. No change proposed.

**Mapping:** supports design §5 combination layer fully; flags that
disagreement is not calibrated and should be validated by the reported
correlation/abstain curve.

---

## 7. Concrete recommendations and risks for WM-N1, mapped to design §2–6

Flags only; **nothing here changes the frozen registration**. Each item names
where it lands in `WM_N1_DESIGN.md`; each is tied to a §1–6 citation.

| # | Recommendation / risk | Design section | Basis | Change? |
| --- | --- | --- | --- | --- |
| R1 | Encoder shape (per-agent token, lane-aware relative attention over ±150 m, relative-geometry biases) is the standard and is right for ~100 cars; no larger model is needed. | §2.1 | Alahi 2016; Veličković 2018; Gao 2020; Liang 2020 (all **to verify**) | keep |
| R2 | Factorized (lane 3-way) × (accel K=3) can emit an accel mode that is invalid in the lane; prefer a joint (lane, accel-mode) categorical conditioned on the lane graph. | §2.2 | mixture/structured decoders (Bishop 1994; Social GAN) **to verify** | **flag** |
| R3 | Keep direct multi-horizon τ-outcome heads independent of the rollout; do not report short-horizon rollout accuracy and τ-outcome CRPS as if one head produced both. | §2.3–2.4, §6 | one-shot vs autoregressive forecasters; exposure bias (Bengio 2015) **to verify** | keep + state |
| R4 | Scheduled sampling + 20-step multi-step pathwise loss is the correct drift remedy; no stronger claim about 570 s rollout accuracy than the data supports. | §3(b), §2.2 | Bengio 2015 **to verify** | keep |
| R5 | Contrast loss on paired, same-seed branches is the right decision-target loss (CRN variance reduction; ITE framing); state that the comonotone pairing yields a coupled quantile *difference*, whose calibration is checked by conformal + sign accuracy. | §3 (paired), §0, §6 | Rubin 1974; Künzel 2019; CRN (Glasserman) **to verify** | **flag** (wording) |
| R6 | Supervised plan-conditioned model (no imagination-trained actor) avoids offline model-exploitation; keep Dreamer-style latent imagination as proposal-only. | §3, §5 | MBPO/MOPO/COMBO; repo note `NOTE_PARTB_VS_DREAMER.md` | keep |
| R7 | IDM/MOBIL + learned residual is the data-efficient hybrid; IDM itself is repo-verified. Make it the fallback if short-horizon s_RMSE loses to Part A v2g. | §5(b) | Treiber 2000 (**verified-in-repo**); Kesting 2007 **to verify**; Raissi 2019 **to verify** | keep |
| R8 | Conformal per-τ is valid only under CAL↔TEST exchangeability; the fresh chunk-5 TEST may be shifted, so report empirical coverage/PIT and interpret shortfall as shift, not bug. | §3, §4, §6 | Vovk 2005; Lei 2018; Ovadia 2019 **to verify** | **flag** (reporting) |
| R9 | Read CRPS/blend weights per demand regime or normalized, because CRPS is scale-sensitive and heavy-tailed delay is regime-dependent. | §3, §5(a), §6 | Gneiting & Raftery 2007 **to verify** | **flag** (reporting) |
| R10 | Stacking + cross-fit + conformal is the literature-standard combination; disagreement is *not* calibrated, so keep the correlation/abstain curve as its justification. | §5(a),(c) | Bates & Granger 1969; Wolpert 1992; Chow 1970 **to verify** | keep |
| R11 | Deep ensemble of 5 is adequate; more helps tails only. Ensembles are not perfectly calibrated, so conformal is load-bearing. | §3 uncertainty | Lakshminarayanan 2017; Kendall & Gal 2017 **to verify** | keep |
| R12 | Validity-layer intervention counts are a diagnostic, not proof of physical validity (low counts can mean the model avoids constrained states). | §2.5, §6 | learned-simulator behavior validation (TrafficSim) **to verify** | keep |

None of R2/R5/R8/R9 require an edit to the frozen registration; they are
sentences to add or a head to make joint. If adopted before training they do
not change the estimand, the splits or the decision rule.

---

## 8. Citation table (with verification status)

`verified-in-repo` means a file in this workspace states the reference in full
(or gives a resolvable identifier). Everything else is `(to verify)`: recalled
from memory, no identifier asserted.

| Citation | Informs | Status |
| --- | --- | --- |
| Alahi et al. 2016, *Social LSTM: Human Trajectory Prediction in Crowded Spaces* | §1 | (to verify) |
| Gupta et al. 2018, *Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks* | §1, R2 | (to verify) |
| Salzmann et al. 2020, *Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data* | §1 | (to verify) |
| Veličković et al. 2018, *Graph Attention Networks* | §1, R1 | (to verify) |
| Gao et al. 2020, *VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation* | §1, R1 | (to verify) |
| Liang et al. 2020, *Learning Lane Graph Representations for Motion Forecasting* (LaneGCN) | §1, R1 | (to verify) |
| Ettinger et al. 2021, *Large Scale Interactive Motion Forecasting ... Waymo Open Motion Dataset* | §1 | (to verify) |
| Vaswani et al. 2017, *Attention Is All You Need* | §1 | (to verify) |
| Bishop 1994, *Mixture Density Networks* | §2, R2 | (to verify) |
| Koenker & Bassett 1978, *Regression Quantiles* | §2 | (to verify) |
| Bengio et al. 2015, *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* | §2, R3, R4 | (to verify) |
| Ha & Schmidhuber 2018, *World Models* | §3 | (to verify) |
| Hafner et al. 2019, *Learning Latent Dynamics for Planning from Pixels* (PlaNet) | §3 | (to verify) |
| Hafner et al. 2023, *Mastering Diverse Domains through World Models* (DreamerV3) | §3, R6 | (to verify) |
| Janner et al. 2019, *When to Trust Your Model: Model-Based Policy Optimization* (MBPO) | §3, R6 | (to verify) |
| Yu et al. 2020, *MOPO: Model-based Offline Policy Optimization* | §3, R6 | (to verify) |
| Yu et al. 2021, *COMBO: Conservative Offline Model-Based Policy Optimization* | §3, R6 | (to verify) |
| Suo et al. 2021, *TrafficSim: Learning to Simulate Realistic Multi-Agent Behaviors* | §3, R12 | (to verify) |
| Treiber, Hennecke & Helbing 2000, *Congested traffic states in empirical observations and microscopic simulations*, arXiv:cond-mat/0002177 | §3, R7 | **verified-in-repo** (`archive/pre_circles/docs/TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md`) |
| Kesting, Treiber & Helbing 2007, *General lane-changing model MOBIL for car-following models* | §3, R7 | (to verify) |
| Raissi, Perdikaris & Karniadakis 2019, *Physics-Informed Neural Networks* | §3, R7 | (to verify) |
| Rubin 1974, *Estimating causal effects of treatments in randomized and nonrandomized studies* | §4, R5 | (to verify) |
| Künzel et al. 2019, *Metalearners for estimating heterogeneous treatment effects using machine learning* | §4, R5 | (to verify) |
| Wager & Athey 2018, *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests* | §4 | (to verify) |
| Glasserman 2003, *Monte Carlo Methods in Financial Engineering* (common random numbers) | §4, R5 | (to verify) |
| Lakshminarayanan et al. 2017, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* | §5, R11 | (to verify) |
| Gal & Ghahramani 2016, *Dropout as a Bayesian Approximation* | §5 | (to verify) |
| Kendall & Gal 2017, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* | §5, R11 | (to verify) |
| Vovk, Gammerman & Shafer 2005, *Algorithmic Learning in a Random World* | §5, R8 | (to verify) |
| Shafer & Vovk 2008, *A Tutorial on Conformal Prediction* | §5 | (to verify) |
| Lei et al. 2018, *Distribution-Free Predictive Inference for Regression* | §5, R8 | (to verify) |
| Gneiting & Raftery 2007, *Strictly Proper Scoring Rules, Prediction, and Estimation* | §2, §5, §6, R9 | (to verify) |
| Ovadia et al. 2019, *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift* | §5, R8 | (to verify) |
| Quiñonero-Candela et al. 2009, *Dataset Shift in Machine Learning* | §5 | (to verify) |
| Bates & Granger 1969, *The Combination of Forecasts* | §6, R10 | (to verify) |
| Wolpert 1992, *Stacked Generalization* | §6, R10 | (to verify) |
| Chow 1970, *On optimum recognition error and reject tradeoff* | §6, R10 | (to verify) |
| Stern et al. 2017, *Dissipation of stop-and-go waves via control of autonomous vehicles: Field experiments*, arXiv:1705.01693 | context (damping baseline) | **verified-in-repo** (`archive/pre_circles/docs/superpowers/reports/2026-08-06-reference-list.md`) |
| Vinitsky et al. 2018, *Benchmarks for reinforcement learning in mixed-autonomy traffic* (Flow) | context (RL baseline) | **verified-in-repo** (`archive/pre_circles/docs/superpowers/reports/2026-08-06-reference-list.md`) |

### Verification-status summary

- **2 verified-in-repo** entries (Treiber et al. 2000 IDM; Stern et al. 2017),
  each confirmed by an existing repository file, plus Vinitsky et al. 2018 for
  context.
- **All neural-forecasting / world-model / causal / uncertainty references
  are `(to verify)`**: this repository does not contain them, and no search
  tool is available in this session. They must be confirmed against real
  sources (venue, year, exact title) before any external write-up.

### Internal (non-citation) repo evidence used

- `output/wd01/wm_n1/WM_N1_DESIGN.md` (the design this note informs).
- `output/wd01/wm_n1/REGISTRATION_WM_N1.md` (frozen; not modified).
- `output/wd01/NOTE_PARTB_VS_DREAMER.md` (existing context on Dreamer-style
  world models; it gives **no full references**, so it is used as reasoning
  context, not as a verified citation source).
- `output/wd01/PARTA_V2_GENERAL_DESIGN.md` (names IDM/MOBIL; no full MOBIL
  bibliography).
- `archive/pre_circles/docs/TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md`
  (IDM arXiv id).
- `archive/pre_circles/docs/superpowers/reports/2026-08-06-reference-list.md`
  (Stern 2017, Flow, and other traffic references).
