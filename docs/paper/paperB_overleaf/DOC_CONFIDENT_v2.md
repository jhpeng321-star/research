# Zero-Shot Prediction of Human Responses to Unseen CAV Controllers with a Physics-Informed Traffic World Model

(Exported from the owner-review Doc, rev 11, 2026-11-01 — CONFIDENT rewrite. This is the text main.tex must match. Graphs stay empty; red [PENDING] items await results.)

## Abstract

Choosing a controller for connected and automated vehicles (CAVs) requires predicting how surrounding human drivers will react to behavior the predictor has never seen. We present WM-S, a physics-informed traffic world model that executes the candidate controller's own code and advances human drivers with microscopic car-following, lane-change and merge physics whose per-driver settings and traffic arrivals are learned from observed motion. Because human response is modeled through physical interaction rather than learned controller identity, WM-S transfers zero-shot to controllers excluded from training. On an unseen cooperative wall controller, WM-S reduces the 60 s median human-vehicle position error to 8.71 m, versus 25.41 m and 40.07 m for trajectory-based and controller-identity black-box predictors (paired difference −9.98 m, 95% CI [−16.78, −5.09]), and it remains more accurate at 120 s. With genuine driving history, the current model is 45 m more accurate than the trajectory black box on an unseen PI controller. A physics-faithful rebuild that follows SUMO's IDM, LC2013 lane-change and merge logic lowers the error of predicted delay distributions by 26–47% across 2–10 min horizons and recovers the correct direction of controller effects. We also map how far such predictions can be trusted: controller-ranking direction is reliable up to about 90 s, and even simulation from the observable state loses accuracy beyond a few minutes because driver settings and future arrivals are hidden. Finally, WM-S selects controllers that outperform always using MAPPO and a black-box selector at short horizons. [PENDING: comparison with learned controller pickers]

## Index Terms

Traffic world models, mixed traffic, physics-informed prediction, controller transfer, uncertainty, microscopic simulation.

## I. Introduction

Connected and automated vehicles (CAVs) can smooth traffic, meter merges and coordinate with each other, but every new CAV controller changes how the surrounding human drivers behave. Deciding which controller to deploy, and when, therefore requires a model that predicts how human-driven vehicles (HDVs) will respond to a controller before it is tried. Field trials are expensive and simulators of the real road are rarely available, so this prediction must come from a learned model of traffic.

Data-driven traffic predictors learn the response of human drivers to the controllers present in their training data. When a genuinely new controller appears, the situations it creates fall outside that experience and black-box predictions degrade. Our key observation is that human drivers do not react to a controller's identity; they react to gaps, speeds and lane opportunities. A world model that executes the candidate controller's own code and advances human drivers with physical interaction rules can therefore transfer to unseen controllers without retraining.

We build this idea into WM-S, a physics-informed traffic world model. CAVs are driven by the executable controller; human drivers follow microscopic car-following, lane-change and merge physics; and what cannot be observed, each driver's settings and the traffic still to arrive, is learned from observed motion and represented by sampled futures. This design makes the model controller-agnostic, keeps its behavior physically interpretable, and lets it express uncertainty about hidden traffic rather than about arbitrary noise.

Our contributions are:

- **WM-S**, a controller-executing, physics-informed traffic world model with learned per-driver settings, learned arrivals and hidden-variable futures, together with a physics-faithful variant that follows SUMO's IDM, LC2013 and merge logic.
- **Zero-shot prediction of unseen controllers**: on an unseen cooperative wall, WM-S predicts human responses 3–5× more accurately than black-box predictors at 60 s and remains more accurate at 120 s; with genuine history it also outperforms them on an unseen PI controller.
- **Distributional fidelity**: the physics-faithful rebuild lowers the error of predicted delay distributions by 26–47% and recovers the correct direction of controller effects.
- **Trust and decisions**: we map the horizon over which model verdicts are reliable, show that long-horizon limits stem from hidden information, and demonstrate that model-based controller selection outperforms fixed and black-box selection at short horizons.

## II. Related Work

**Microscopic traffic models.** Analytical car-following and lane-changing models provide structured descriptions of human driving. IDM captures longitudinal interaction [3], MOBIL formalizes lane-change incentive and safety [4], and SUMO implements widely used variants of both, including the LC2013 lane-change model [10]. WM-S builds on these mechanisms and learns their per-driver parameters from observed motion.

**Physics-informed learning for mixed traffic.** Sheng et al. [2] combine IDM dynamics with neural residuals for CAV trajectory control, and Liu et al. [1] use physics-informed state inference and a learned transition ensemble for offline cooperative mixed-traffic control. These works use physics to regularize learning for a fixed control task. WM-S instead targets transfer to new controllers: it executes the controller's own code inside the model and keeps human dynamics controller-agnostic, which is what enables zero-shot prediction.

**Learned driving world models and forecasting.** Generative and interactive world models such as GAIA-1 [5] and Waymax [6], and dynamics-aware forecasters such as Trajectron++ [7], learn rich scene evolution from data. They are not designed to predict the response to a controller absent from training, which is the setting we study; our black-box baselines represent this learning-only approach on identical data.

**Trusting model verdicts.** Recent work argues that a world model's verdicts on policies must be validated before use [8], and surveys of mixed-traffic modeling highlight the gap between learned and analytical agents [9]. We contribute a traffic-specific answer: we measure the horizon over which WM-S's controller comparisons are reliable and evaluate its uncertainty by the calibration of its predicted distributions.

## III. Problem Formulation and Method

### A. Controller-Conditioned Traffic Prediction

Let h_t be the observable history available to the roadside unit and CAVs, x̂_k the predicted traffic state at step k, and π a candidate CAV controller with internal memory m_k. WM-S rolls the controller and the traffic forward together: (u_k, m_{k+1}) = π(o(x̂_k), m_k) and x̂_{k+1} = F_θ(x̂_k, u_k; z), where o builds the controller's observations from the predicted state and z collects hidden variables (driver settings and future arrivals). The controller therefore reacts to the traffic its own earlier actions produced, exactly as it would on the road.

We evaluate predictions in two modes. In teacher-forced mode, CAVs replay their recorded motion and WM-S predicts only the human response, isolating the quality of the human-dynamics model. In closed-loop mode, the controller acts on the model's own predictions, which tests the complete system.

For a cost J_H(π) over horizon H, the effect of a controller relative to a reference is ΔJ_H = J_H(π) − J_H(π_ref); negative values favor the candidate. A model-based selector picks the controller with the lowest predicted cost, and its regret is the realized cost of that choice minus the best realized cost among the candidates.

### B. Physics-Informed Human Dynamics

WM-S advances each human driver with microscopic physics. Longitudinal motion follows IDM, â_{i,k} = a_IDM(x̂_k; η_i), where η_i are the driver's settings (desired speed, time headway, minimum gap, acceleration and braking). A history encoder infers η_i from the driver's own observed motion and its interactions with neighbors; drivers with little history are shrunk toward a learned population distribution. No driver labels or types are ever used.

Lane changes and merges are discrete events that strongly shape congestion at a bottleneck. The physics-faithful variant of WM-S implements SUMO's LC2013 lane-change logic (strategic, cooperative, speed-gain and keep-right motivations with SUMO's safety gaps) and SUMO-style merge and insertion ordering, so that simulated drivers decide when to change lanes and merge as SUMO drivers do. All learned acceleration corrections are switched off in this variant: behavior comes from the physics and the learned settings alone.

Traffic is an open system: vehicles request entry, wait outside the road, are admitted, drive and exit. WM-S learns the arrival process from entrance counters with a Gamma–Poisson model, assigns new drivers settings drawn from the learned population, and tracks both on-road delay and waiting at the source, so that predicted trajectories connect directly to network delay.

[FIGURE 1 — WM-S architecture] Observed history infers driver settings and arrivals; the executable CAV controller acts on predicted observations; car-following, lane-change and merge physics advance human drivers; sampled hidden variables produce a distribution of futures. (Graph to be added.)

### C. Zero-Shot Evaluation Protocol

A prediction is zero-shot when the evaluated controller's behavior was never used to fit the model. We audit every training corpus to confirm that the evaluated controllers (the cooperative wall, PI and MAPPO) are absent, while their executable code is available to WM-S at test time. The black-box baselines are trained on the same corpora, so all models face the same unseen controller.

Comparisons are paired: all models are evaluated on the same worlds, start states, histories and human-driver identities, and confidence intervals resample whole worlds. We also report a genuine-history protocol, in which every model receives the drivers' actual preceding 60 s of motion, reflecting what a deployed system observes.

### D. Hidden-Variable Futures and Credibility

Given the full state, traffic simulation is deterministic: we verified that SUMO restarted from the same physical state produces identical outcomes for every random seed. The uncertainty an observer faces therefore comes from what is hidden, namely each driver's settings and the vehicles still to arrive. WM-S represents exactly this: each imagined future samples driver settings from their inferred posteriors and arrivals from the learned process, then plays out deterministic physics. Candidate controllers are compared on the same sampled futures, so differences reflect the controller rather than luck.

This yields a predictive distribution of delay and congestion for each controller. We judge the model by how well these distributions match reality (CRPS and interval coverage), and we choose controllers by expected outcome, optionally requiring a calibrated probability that the candidate beats the incumbent.

## IV. Experimental Setup

### A. Network, Worlds and Controllers

We study a highway on-ramp merge simulated in SUMO [10], where mainline traffic, ramp arrivals and an upstream bottleneck interact. Scenarios span demands of 1900–2700 veh/h and CAV penetrations of 5–40%, with heterogeneous human drivers. The model observes only what CAVs and a roadside unit at the merge can sense: vehicle positions and speeds within sensing range, merge- and ramp-zone traffic, and entrance counts.

The controller library contains native SUMO driving, a simple speed-command baseline (CTRL0), a speed-only MAPPO policy [11], a PI speed controller, the CIRCLES wave-damping controller, and a cooperative wall controller that meters inflow into the merge by coordinating CAVs across lanes. The wall and PI controllers serve as unseen controllers.

The unseen-wall study uses 67 start states from 29 worlds. The closed-loop study uses 40 recorded worlds with genuine 60 s histories. The physics-faithful rebuild is evaluated on 20 worlds (8 held out for the final comparison) under all five library controllers with 8 sampled futures each. The controller-selection study uses 90 recorded worlds.

### B. Baselines and Comparability

We compare WM-S with two black-box predictors trained on the same data. BB-traj is a physics-free neural trajectory model that uses the same inputs and the same recorded CAV motion; comparing against it isolates the value of physical structure. BB-id predicts CAV motion from a learned controller-identity embedding instead of executing the controller; comparing against it isolates the value of executing controller code.

As additional references we use SUMO itself started from the observable state (the best a physics-faithful model could do with the same information) and a climatology baseline that predicts the typical outcome for each traffic level, CAV share and controller.

### C. Metrics and Statistical Units

Trajectory accuracy is the absolute position error of human-driven vehicles at a given horizon, compared on the same vehicles across models. Network performance is total delay in vehicle-seconds, combining on-road speed deficit and waiting at the source.

For distributions we use the continuous ranked probability score, CRPS = E|X − Y| − ½ E|X − X′|, where Y is the real outcome and X, X′ are independent model futures; lower is better and it rewards both accuracy and calibrated spread. For decisions we report the direction accuracy of predicted controller effects and the regret of model-based controller selection.

All intervals are 95% confidence intervals from paired bootstrap resampling of whole worlds.

## V. Results

### A. Zero-Shot Response Prediction

**Unseen wall.** The cooperative wall creates traffic situations none of the models saw in training: two or more CAVs slowing together across lanes to meter the merge. WM-S predicts the human response to it with a median 60 s position error of 8.71 m, compared with 25.41 m for BB-traj and 40.07 m for BB-id, a 3–5× reduction. The paired difference to BB-traj is −9.98 m [−16.78, −5.09] and to BB-id −31.78 m [−45.37, −18.61]. The advantage persists at 120 s, where WM-S is 49.21 m [7.31, 110.05] closer than BB-traj. Because the physical interaction rules do not depend on which controller is driving, WM-S generalizes to the new behavior while the learned predictors extrapolate poorly.

**Unseen PI controller and the value of history.** WM-S infers each driver's settings from their recent motion, so it benefits from the history a deployed system has. With genuine 60 s histories, the current model predicts the human response to the unseen PI controller 45.05 m more accurately than BB-traj at 60 s [25.25, 52.73]. Within the same starts, giving the model full history instead of a snapshot alone improves its accuracy by 13.47 m [1.49, 26.92], showing that the learned driver inference does real work.

**Closed loop.** When the controller acts on the model's own predictions, WM-S remains competitive with BB-traj on human positions at 60 s (−5.17 m for PI and −3.45 m for MAPPO, both within noise) and predicts 60 s network delay more accurately under MAPPO (absolute-error difference −116 veh-s [−375, −11]).

### B. Where the Trust Horizon Ends

A model used to choose controllers must predict which controller does better. WM-S predicts the direction of controller effects correctly with accuracy 0.92–0.95 at 10–60 s and 0.89 [0.84, 0.93] at 90 s, passing the registered reliability test at every horizon up to 90 s; at 120 s accuracy drops to 0.75 [0.68, 0.82]. This defines a trust horizon of about 90 s for controller comparisons, which directly informs how often decisions should be re-planned.

The longer-horizon limit is largely informational rather than a defect of WM-S alone. Starting SUMO itself from the observable state, with driver settings inferred from motion, gives excellent short-range forecasts (delay CRPS 647 veh-s at 120 s versus 1433 for climatology) but loses to climatology at 5–10 min (16728 versus 9131 veh-s at 600 s). Even exact physics cannot recover outcomes that depend on drivers' hidden settings and on vehicles that have not yet arrived.

Consistent with this, restarting SUMO from an identical physical state with eight different random seeds yields identical outcomes (zero spread at 600 s): given the full state, traffic is deterministic. Long-horizon uncertainty is therefore uncertainty about hidden variables, which is exactly what WM-S samples.

[FIGURE 2 — Zero-shot accuracy and trust horizon] (a) Position error vs. horizon for WM-S, BB-traj and BB-id on the unseen wall; (b) direction accuracy vs. horizon with the 90 s trust horizon; (c) delay CRPS of WM-S, observable-start SUMO and climatology vs. horizon. (Graph to be added.)

### C. Distributional Fidelity of the Physics Rebuild

Replacing learned corrections with SUMO-faithful physics and learned hidden variables makes WM-S's predicted traffic substantially more realistic. On the held-out worlds, across all five controllers, the CRPS of predicted total delay falls from 1499 to 1105 veh-s at 2 min (−26%), from 6817 to 3777 at 5 min (−45%) and from 24752 to 13105 at 10 min (−47%) relative to the previous learned-correction model. Human-vehicle position error at 60 s improves by 15.26 m.

The rebuilt model also recovers the correct direction of controller effects over 10 minutes, which the previous model got wrong: it predicts MAPPO to reduce delay relative to native driving by 1246 veh-s (real: 2014), PI by 1674 (real: 1131), and CIRCLES to increase it by 4583 (real: 3746). These are exactly the quantities a controller-selection system needs.

The remaining gaps point to concrete next steps: the rebuilt model still produces somewhat more congestion than reality (merge breakdown 5.6 percentage points further from the recorded rate than the previous model at 5 min), its controller comparisons are over-confident, and its lane-change model proposes too many keep-right changes. [PENDING: v5 results with calibrated uncertainty, refined lane changes and learned physics parameters]

### D. Controller Selection and Partial Switching

No single controller is best everywhere. Across 90 recorded worlds, each library controller wins in some traffic conditions at 10 min (native 22, CTRL0 17, MAPPO 19, PI 29, CIRCLES 3), and the winner changes with demand and CAV share. This creates room for a model that picks the right controller for the situation.

Using its predictions within the trust horizon, WM-S selects controllers that outperform simply always using MAPPO, the strongest fixed controller, by 8.24 veh-s per 60 s decision [0.08, 18.99] on 32 states from 21 worlds, and that incur 8.41 veh-s [0.31, 19.23] less regret than selections made with BB-traj. Physical structure therefore improves not only predictions but the decisions built on them.

[PENDING: receding-horizon switching every 20 s among six controllers including the wall, comparing WM-S selection with BB-traj selection, a fleet-level PPO controller picker, a per-CAV MAPPO controller picker, each fixed controller and random switching (Table I).]

TABLE I — Controller switching on fresh worlds (to be filled)

| Selector | Fleet + source cost | Branch regret |
| --- | --- | --- |
| WM selector | [--] | [--] |
| BB-traj selector | [--] | [--] |
| Fleet PPO picker | [--] | [--] |
| Per-CAV MAPPO picker | [--] | [--] |

## VI. Discussion and Limitations

Executing the controller and modeling humans through physics is what lets a traffic world model handle controllers it has never seen: the learned components are confined to quantities that do not depend on the controller (driver settings and arrivals), so new controller behavior is handled by the physics rather than extrapolated by a network. The rebuild shows that the more faithfully the rules follow how drivers actually decide, the more realistic the predicted distributions become, and the trust-horizon analysis shows where predictions should hand over to re-planning.

**Limitations.** The unseen-wall comparison was run in teacher-forced mode, and its test worlds appear in training under other controllers, so it demonstrates transfer to an unseen controller rather than to unseen worlds. Results are in simulation; transfer to real drivers requires field data. Our black-box baselines are in-house models trained on identical data rather than published forecasters. The physics-faithful rebuild still over-predicts congestion and is over-confident in controller comparisons; a reference switching run on a small subset of worlds did not yet show an advantage over the best fixed controller. Calibrating the predicted distributions and completing the switching study are our immediate next steps.

## VII. Conclusion

We presented WM-S, a physics-informed traffic world model that executes CAV controller code and advances human drivers with microscopic physics whose settings and arrivals are learned from observed motion. This design predicts human responses to unseen controllers far more accurately than black-box predictors, with 3–5× lower error on an unseen cooperative wall, and its physics-faithful rebuild cuts distributional delay error by up to 47% while recovering the direction of controller effects. By mapping the horizon over which its verdicts can be trusted and grounding uncertainty in hidden traffic variables, WM-S provides a principled basis for choosing CAV controllers before they are deployed.

## Acknowledgment

Author, affiliation and funding information will be supplied before submission.

## References

(unchanged from refs.bib: [1] Liu et al. 2026 arXiv:2608.17739; [2] Sheng et al. 2024 CTR; [3] Treiber et al. 2000 PRE; [4] Kesting et al. 2007 TRR; [5] GAIA-1; [6] Waymax; [7] Trajectron++; [8] Oefinger et al. 2026 arXiv:2607.07196; [9] Rahmani et al. 2026 arXiv:2604.12857; [10] Lopez et al. 2018 ITSC (SUMO); [11] Yu et al. 2021 MAPPO.)
