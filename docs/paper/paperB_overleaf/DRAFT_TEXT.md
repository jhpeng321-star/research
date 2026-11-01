# WM-S: Physics-Informed Prediction under Unseen Traffic Controllers and the Limits of Its Verdicts

Author names to be supplied

Affiliations and contact details to be supplied

## Abstract

Evaluating a new connected and automated vehicle controller requires predicting how surrounding human drivers respond to actions outside the predictor's training experience. We study a physics-informed traffic world model that separates executable controller code from human-driver dynamics. The model family combines microscopic car-following, lane-change and merge rules with history-conditioned driver settings, learned corrections or arrivals, and hidden-variable future sampling. A physics-faithful rebuild targets SUMO's IDM and LC2013 behavior explicitly. In recovered historical unseen-wall experiments, median human-vehicle position error at $60$ s is $8.71$ m, compared with $25.41$ and $40.07$ m for in-house black-box baselines; the paired structured-minus-trajectory contrast is $-9.98\,[-16.78,-5.09]$ m.
<!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:8,27. -->
This result is teacher-forced controller-response prediction, not autonomous feedback superiority or unseen-world transfer. Genuine-history proportional–integral prediction provides a favorable conditional extension, whereas the completed common-plant feedback comparison has inconclusive principal-horizon endpoint advantages. Direction tests support a bounded short-horizon envelope; improved distributional point estimates in the rebuild coexist with breakdown-rate error and overconfident controller comparisons. Exploratory offline selection is favorable, but reference switching remains partial and the learned-picker comparison is pending. These results support controller/environment factorization while showing why trajectory accuracy, distributional fidelity and reliable controller choice require separate validation.

## Index Terms

Traffic world models, mixed traffic, physics-informed prediction, controller transfer, uncertainty, microscopic simulation.

## I. Introduction
Connected and automated vehicles (CAVs) can influence surrounding human-driven vehicles (HDVs) through speed control, coordinated motion and interactions at bottlenecks. Before choosing a controller, however, one must estimate the response of the traffic system to its actions. A predictor that reproduces typical trajectories may still misjudge a new intervention. The controller changes the observations on which its subsequent actions depend, while human reactions, lane changes and arrivals alter both local motion and network cost. The relevant question is therefore not only whether a forecast looks plausible, but whether it preserves the consequences of an unfamiliar feedback law.

An explicit separation between known controller execution and uncertain human dynamics offers a useful modeling contract. Rather than learning a mapping from controller identity to future traffic, a world model advances the surrounding vehicles and lets the candidate controller act on predicted observations. Microscopic physics constrains this evolution, while learned behavior representations can account for driver heterogeneity and incomplete observations. Physics-informed traffic world models and residual dynamics already provide important foundations [1, 2]; our emphasis is the transfer and decision-validation problem created when the controller itself changes.

This distinction also makes failure interpretable. A model may predict relative HDV responses well under recorded CAV motion yet accumulate error when CAVs react to its own forecasts. It may reduce traffic-cost error relative to a weaker baseline while reversing controller rankings. Source waiting, admission and lane-change timing can further change the relationship between endpoint error and fleet delay. We consequently treat state prediction, paired intervention effects, uncertainty and controller selection as related but non-equivalent tests.

Our contributions are:

- An executable-controller/environment factorization for a versioned physics-informed traffic world-model family, with explicit treatment of human dynamics, driver settings and open-road accounting.
- A controller-exclusion study that separates historical teacher-forced response prediction, genuine-history diagnosis and autonomous common-plant feedback, preserving their different exposure and observation contracts.
- A control-oriented reliability analysis linking prediction error, direction horizons, distributional scores and controller regret, including failed qualification and incomplete switching evidence.

The supported headline is historical response prediction under an excluded controller. We do not infer universal black-box superiority, real-road validity or a qualified long-horizon controller judge from that result.

## II. Related Work
Analytical car-following and lane-changing models provide structured priors for microscopic traffic dynamics. IDM represents longitudinal interaction [3], while MOBIL formalizes lane-change incentive and safety [4]. These priors do not make lane-event timing or merge handling interchangeable across simulators. In particular, MOBIL and SUMO lane logic are different mechanisms. Our historical models use version-dependent MOBIL-style or learned lane rules; the separate physics-faithful rebuild targets an LC2013 highway subset and SUMO-style merge and insertion behavior.

The closest learned traffic models already combine physics with data. Sheng et al. [2] combine IDM-based dynamics with neural residuals and residual reinforcement learning for CAV trajectory control. Liu et al. [1] use physics-informed state inference and a learned transition/reward ensemble for offline cooperative mixed-traffic control. Their physics regularizes inference rather than serving as the explicit microscopic transition evaluated here. Our distinction is the combination of controller-code factorization, controller-exclusion evaluation and paired decision diagnostics; additive residual modeling and physics-informed world modeling are not new in themselves.

Learned driving worlds and interactive simulators address broader action-conditioned scene evolution. GAIA-1 [5] models driving generation, and Waymax [6] supports interactive multi-agent simulation. Dynamics-aware forecasting is also established, as illustrated by Trajectron++ [7]. Our in-house trajectory and controller-identity baselines are not implementations of those systems. Their results therefore support comparisons within the declared experimental family, not a ranking against published forecasting methods.

The need to validate a simulator before trusting its policy verdict has also been articulated directly [8]. The broader mixed-traffic modeling literature spans analytical, learned and hybrid agent descriptions [9]. We operationalize this concern through traffic-specific controller effects, regret, road/source accounting and horizon-dependent diagnostics. We neither propose a universal certification standard nor claim that passing a direction test establishes simulator admissibility.

## III. Problem Formulation and Method
### Controller-Conditioned Traffic Prediction
Let $h_t$ denote the permitted observation history, $x_t$ the represented traffic state and $\pi$ a candidate CAV controller with internal memory $m_t$. A feedback forecast evolves as
$$
\begin{aligned}
 (u_k,m_{k+1}) &= \pi(o(\hat{x}_k),m_k),\\
 \hat{x}_{k+1} &= F_\theta(\hat{x}_k,u_k;z),
\end{aligned}
$$
where $o$ constructs controller observations and $z$ denotes hidden driver and arrival variables. After initialization, future observations do not re-anchor the forecast. The controller sees predicted traffic and can therefore respond to consequences of its earlier predicted actions. A valid continuation must bind its causal memory as well as its current observation.

We distinguish this contract from teacher forcing. In the historical wall experiment, recorded future CAV motion is supplied while HDVs evolve under the predictor. This isolates conditional human response but provides information unavailable to autonomous prediction. Recorded-command replay is another distinct contract: issuing an archived action sequence does not imply that the controller receives predicted observations. These protocols are labeled separately throughout the results.

For a supported cost $J_H(\pi)$ over horizon $H$, we define the intervention effect relative to a reference controller as $\Delta J_H=J_H(\pi)-J_H(\pi_{\mathrm{ref}})$. Negative values favor the candidate. An offline selector minimizes predicted cost over the evaluated library, and oracle regret is its realized cost minus the minimum realized library cost at that decision. The reference, controller set and cost population are part of this definition; changing them changes the estimand.

### Physics-Informed Human Dynamics
The WM-S family uses microscopic interaction rules to advance each human driver. In later hybrid configurations, a history encoder maps observed vehicle motion and neighbor interactions to settings such as desired speed, time headway, spacing and acceleration/braking behavior. Limited observations lead to shrinkage toward population defaults. A compact longitudinal description is
$$

 \hat{a}_{i,k}=a_{\mathrm{IDM}}(\hat{x}_k;\hat{\eta}_i)
                  +g(k)r_\theta(h_t,\hat{x}_k),

$$
where $\hat{\eta}_i$ is a driver representation, $r_\theta$ is a learned correction and $g$ is a version-specific rollout gate. Training uses recorded multi-step motion; settings are predictive latent variables, not verified recovery of true driver types. Historical residual-only configurations did not wire inferred settings into the physics. Their successes cannot be attributed retrospectively to learned physical-parameter identification.

Lane changes and merges require discrete event logic in addition to acceleration. The historical family combines checkpoint-dependent analytical or learned lane decisions with mandatory-merge and gap handling. The separate WM-PHYS rebuild implements SUMO-faithful IDM, an LC2013 highway subset, and merge/insertion ordering. Its frozen FULL-v4 variant uses causal own-start motion, a fitted population/driver posterior and learned arrivals, with learned corrections and fitted lane-change settings disabled. It is a new forecast system, not a controlled settings-only ablation of the historical network.

Open-road evolution includes request generation, waiting outside the road, admission, vehicle motion and exit. The rebuild estimates arrivals from causal entrance counters through a Gamma–Poisson model and assigns entrants population-conditioned settings. These operations connect individual trajectories to aggregate delay. A vehicle awaiting admission contributes source waiting even when absent from an on-road position tensor. Conversely, an exited original vehicle cannot supply a later identity-matched endpoint. Neither omission should be silently replaced by zero error.

[FIGURE: Controller/environment factorization]

**Figure 1.** Planned architecture diagram: observed history conditions driver settings and arrivals; executable CAV controllers act on predicted observations. Human dynamics, lane/merge events and road/source accounting advance the forecast. Historical residual-only WM-S, later hybrid WM2 and the separate LC2013-based rebuild will be labeled explicitly. Graph intentionally empty in this owner-review draft.

### Zero-Shot Evaluation Protocol
Controller exclusion is checkpoint-specific: the evaluated controller's outcomes must not have been used to fit the tested predictor. Executable controller code may be available without supplying its response trajectories during fitting. Audited corpora support exclusion of the evaluated algorithms, but copied initialization ancestry is not fully linked. We therefore describe the later algorithm-novelty evidence as provisional rather than treating an architecture name as proof of strict exclusion.

Controller novelty and world novelty are separate. Historical wall evaluation worlds appear in training under other controller arms. The diagnosis also changes training-world coverage between older and newer checkpoints, so it cannot isolate an architectural cause for differences. Fair comparisons must additionally bind the observed history, future CAV inputs, actuator and lane ownership, source support and survivor cohort. A later start with genuine history is a separate evaluation; the original forecast may not receive observations from its own future.

### Hidden-Variable Futures and Credibility
Observable traffic does not determine every driver's settings, unobserved internal state or future requests. Sampling these variables produces candidate futures even when the configured physical simulator is deterministic conditional on its complete state and population. Paired controller evaluations share latent draws where supported, helping distinguish intervention effects from changes in the sampled environment. This construction represents uncertainty conditional on the model; it does not automatically cover model misspecification or calibrate action probabilities.

Credibility therefore requires more than ensemble spread. We evaluate endpoint accuracy, aggregate delay, signed controller effects and empirical forecast distributions, and inspect whether apparently confident controller comparisons succeed. A useful model must preserve relevant population accounting and event behavior as well as a marginal mean. The reported horizon envelope is an empirical panel-specific finding. It is not a formal abstention policy, safety certificate or authorization to extrapolate an untested controller or horizon.

## IV. Experimental Setup
### Network, Worlds and Controllers
The study uses SUMO highway-merge traffic [10], where mainline flow interacts with ramp arrivals and an upstream bottleneck. Observable inputs comprise vehicle motion/history, local traffic context, geometry and entrance counters where available. The reports define their own road regions and accounting clocks; we retain those bindings rather than imposing one pooled dataset. These are simulator-based development studies, not measurements of deployment on a real road.

The controller library includes native behavior, CTRL0, a MAPPO-based speed controller [11], proportional–integral (PI) control and CIRCLES; the historical transfer study additionally evaluates an unseen wall law. Controller labels denote frozen implementations, not interchangeable algorithm classes. Full feedback requires the appropriate executor memory and plant. In the completed current-model comparison, WM2 and BB-traj share an external longitudinal/gap-based merge plant without ordinary CAV lane changes; production-native dynamics are a separate diagnostic.

The historical wall panel contains $67$ starts from $29$ worlds at the headline clock.
<!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:8. -->
The completed autonomous study uses $40$ recorded development worlds and genuine $60$ s observation histories.
<!-- Source: E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:2-6. -->
The rebuild campaign covers $20$ worlds, with the quoted paired publication comparison restricted to its $8$ reserved worlds; it evaluates $5$ controller arms with $8$ futures per arm.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:18-25. -->
These panels have different roles and must not be pooled into one effective sample. Scenario-winner and switching studies are identified separately below.

### Baselines and Comparability
BB-traj is an in-house physics-free trajectory predictor. BB-id conditions learned CAV motion on controller identity. In the historical wall comparison, BB-traj shares the recorded-CAV teacher, whereas BB-id ignores that teacher and predicts CAV motion itself. BB-id is therefore not teacher matched and cannot provide an executable feedback-MAPPO comparison. Frozen black-box models retain their original training contracts; published forecasting systems cited for context were not run.

Additional references include uncorrected physics, observable-start SUMO and cell-conditioned climatology where the corresponding reports provide them. These answer different questions: a conditioned climatology tests whether a distributional forecast adds value over scenario averages, while observable-start SUMO tests a physics-based continuation with imperfect initialization. Neither reference is a complete-state oracle. Historical checkpoint, corpus and runtime changes also prevent attributing all model differences to physics alone.

### Metrics and Statistical Units
HDV endpoint error is absolute longitudinal position error on the report's specified surviving identities. Paired contrasts are constructed on shared identities before aggregation and need not equal differences of marginal medians. Aggregate delay is measured in vehicle-seconds. In the autonomous comparison, $J$ combines source waiting and road speed-deficit integrals; the historical selection export's team cost retains its original name because its exact implementation binding is incomplete. Road-only predicted cost is not relabeled fleet-plus-source cost.

We report controller-effect direction, library oracle regret, breakdown-rate fidelity and the continuous ranked probability score (CRPS). For a scalar outcome $Y$ and model future $X$, $\mathrm{CRPS}=\mathbb{E}|X-Y|-\tfrac12\mathbb{E}|X-X'|$, with $X'$ an independent model draw; lower is better. Event-rate error and action reliability complement this proper distributional score. A low CRPS alone does not guarantee correct controller ranking or lane-event precision.

Reported brackets are descriptive source confidence intervals, using whole-world resampling for the cited paired comparisons; they are not multiplicity-adjusted. We preserve each report's within-world weighting, starts and endpoint attrition. Full-world intervals for the rebuild are unavailable under its registered scheme because some contributing strata contain singleton worlds. Conditional mixed-cell intervals and retracted interim intervals are not substituted. Incomplete switching coverage is labeled PARTIAL rather than interpreted as the planned full-population estimate.

## V. Results
### Zero-Shot Response Prediction
The recovered historical unseen-wall experiment supports the structured response model under its declared teacher contract. At $60$ s, median HDV position errors are $8.71$, $25.41$ and $40.07$ m for structured WM-S, BB-traj and BB-id, respectively.
<!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:8,27. -->
The paired WM-S-minus-BB-traj contrast is $-9.98\,[-16.78,-5.09]$ m.
<!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:27. -->
At $120$ s the corresponding contrast is $-49.21\,[-110.05,-7.31]$ m, although absolute errors have grown substantially.
<!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:38. -->
The controller is excluded from the audited historical training arms, but its evaluation worlds were seen under other arms. Entries are disabled. Thus the result is evidence for conditional unseen-controller HDV response prediction, not unseen-world generalization, autonomous wall execution or superior wall selection.

The PI diagnosis shows why history and coverage must be explicit. At original snapshot starts, WM2-minus-BB-traj error at $60$ s is adverse, $+17.85\,[3.24,31.27]$ m.
<!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:14. -->
At separately registered shifted starts with genuine $60$ s history, the conditional contrast becomes $-45.05\,[-52.73,-25.25]$ m at the same prediction horizon.
<!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:25-32. -->
Within identical shifted starts, full history improves WM2 over its snapshot input by $-13.47\,[-26.92,-1.49]$ m.
<!-- Source: E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md:37; horizon 60 s. -->
Later-start selection also changes difficulty and survivors, and newer checkpoints have different training-world coverage. The entire between-panel improvement is therefore not a pure history effect. These experiments retain recorded CAV motion and conditional presence/exits; reset-controller feedback remains adverse and cannot establish an autonomous rescue.

The completed common-plant study directly addresses feedback under autonomous HDV, entry, exit and source evolution. Its paired $60$ s endpoint differences are $-5.17\,[-18,7.09]$ m for PI and $-3.45\,[-13.4,4.58]$ m for MAPPO.
<!-- Source: E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:19. -->
Both intervals include zero, and the earliest endpoint clock favors BB-traj. Later relative aggregate-delay error gains coexist with weak controller direction and regret. At $300$ s no common endpoint cohort remains.
<!-- Source: E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md:17,23,27-38. -->
This is completed, mixed feedback evidence, not confirmation of a general closed-loop endpoint advantage. Causal wall-feedback restoration remains unavailable.

### Where the Trust Horizon Ends
The frozen E2 direction export reports PASS at tested horizons of $10$, $30$, $60$ and $90$ s, and FAIL at $120$ and $180$ s.
<!-- Source: E:/research-dense/docs/paper_results/E2_direction_horizon.md:5-10. -->
For example, direction accuracy is $0.8895\,[0.8424,0.9341]$ at $90$ s and $0.7526\,[0.6824,0.8187]$ at $120$ s.
<!-- Source: E:/research-dense/docs/paper_results/E2_direction_horizon.md:8-9. -->
We reproduce the exported verdicts without inventing a confidence-bound decision rule. Exact checkpoint and sign-denominator bindings are incomplete. The result does not establish failure at every untested later time, transfer a trusted horizon to WM2 or override failed overall fidelity qualification.

Observable-start SUMO exposes a related limit even when the transition physics is native. Its motion-inferred delay CRPS at $120/300/600$ s is $647/4635/16728$ veh-s, against conditioned climatology $1433/4375/9131$ veh-s.
<!-- Source: E:/agent-work/c4-sumocore/docs/paper_results/SUMOCORE.md:12. -->
The short-range gain does not extend to long-range distributional superiority, and the inferred model gets the reported long-horizon controller-effect signs wrong. Initialization collisions occur in $18/144$ futures and remain in the scores.
<!-- Source: E:/agent-work/c4-sumocore/docs/paper_results/SUMOCORE.md:19,23. -->
This is an informative observable-start reference, not a clean best-achievable information ceiling.

The ENSPROBE continuation experiment further separates hidden-variable uncertainty from simulator RNG. With requests and driver population fixed, all tested reseeded physical tapes agree; maximum within-state/arm delay standard deviation at $600$ s is $0$ veh-s.
<!-- Source: E:/agent-work/c4-ensprobe/docs/paper_results/ENSPROBE.md:7,17. -->
The configured ensemble is deterministic conditional on those restored starts. This does not imply zero uncertainty from an observer's history or determinism of real traffic. Nor does it turn earlier restore discrepancies into an irreducible noise floor. Unknown settings and arrivals must be varied explicitly, and remaining forecast bias cannot simply be excused as random traffic futures.

[FIGURE: Prediction, horizons and scenario winners]

**Figure 2.** Planned evidence panels: historical teacher-forced wall errors and paired differences; the separate current-model feedback comparison; frozen direction-horizon verdicts; and controller winners by scenario. Panels will retain their own model, population, metric and execution labels. Graphs intentionally empty; the text reports the existing evidence.

### Distributional Fidelity of the Physics Rebuild
On the same reserved-world comparison, FULL-v4 reduces absolute-delay CRPS relative to WM2. At $120/300/600$ s, WM2 scores $1499/6817/24752$ veh-s and FULL-v4 scores $1105/3777/13105$ veh-s.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:18-25. -->
The independently exported paired differences are $-393/-3040/-11647$ veh-s; rounding of the displayed means explains their apparent subtraction discrepancy.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:24,41. -->
The $60$ s common-HDV mean-world-median endpoint contrast is $-15.26$ m.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:43. -->
These are favorable point estimates, not interval-supported full-panel superiority: the registered full-world confidence intervals are NA because of singleton strata, and earlier numerical intervals were retracted after a bootstrap bug.

The reported controller-effect point means also have the correct signs. At $600$ s, FULL-v4 predicts MAPPO-minus-native $-1246$ veh-s versus recorded $-2014$ veh-s, PI-minus-native $-1674$ versus $-1131$ veh-s, and CIRCLES-minus-native $+4583$ versus $+3746$ veh-s.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:30-32. -->
Correct signs do not imply correct magnitudes or calibrated statewise decisions. Mean delay bias remains $+14648$ veh-s at that horizon.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:26. -->
At $300$ s the breakdown-rate distance from recorded traffic worsens by $+5.63$ percentage points relative to WM2.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:42. -->
The conjunctive adoption rule therefore FAILS; switching retains WM2 despite the rebuild's better CRPS and endpoint point estimates.

Action confidence and lane events expose additional weaknesses. For selected comparisons with forecast win probability at least $0.8$, observed win rates at $120/300/600$ s are $52.9/55.6/70.0\%$, against forecast $92.6/92.4/93.1\%$.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:50-52. -->
These are retrospective comparisons with hypothetical incumbents, not live switching outcomes. A separate conditional lane-change diagnostic reports default precision $3.006\%$ and recall $37.942\%$, with keep-right false positives dominant.
<!-- Source: E:/agent-work/c4-rb-int/docs/paper_results/WMPHYS_MORNING.md:34. -->
It is not the reserved rebuild adoption cohort. Physics fidelity improves important aspects of prediction while leaving breakdown, lane-event precision and overconfidence unresolved. [PENDING: independently frozen successor v5 results and qualification].

### Controller Selection and Partial Switching
Controller diversity provides a reason to study selection, but does not prove that a predictor can exploit it. Across $90$ recorded worlds at $600$ s, unique fixed-episode wins are native $22$, CTRL0 $17$, MAPPO $19$, PI $29$ and CIRCLES $3$.
<!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md:5-8. -->
Every library controller wins somewhere. A separate full-rerun same-state study also finds controller-dependent opportunities, but some best-to-second margins are small. Fixed-world winners, same-state oracle headroom and predictable switching benefit remain distinct findings; the planned scenario-winner panel will preserve that distinction.

The strict-history E1 export supplies exploratory short-horizon choice evidence. At $60$ s on $32$ starts from $21$ worlds, WM-13E-selected minus always-MAPPO cost is $-8.241\,[-18.99,-0.08363]$ veh-s.
<!-- Source: E:/research-dense/docs/paper_results/E1_C1_fair.md:197. -->
The independently reconstructed paired WM-minus-BB-traj oracle-regret contrast is $-8.41\,[-19.23,-0.31]$ veh-s.
<!-- Source: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md:29. -->
This supports a narrow offline-selection advantage under the exported team objective. It is not inferred from differences of marginal confidence intervals and does not establish deployed receding-horizon benefit. Exact historical cost/plant bindings and full initialization lineage still limit stronger interpretation.

The reference switching study is explicitly PARTIAL after its budget stop. The all-arm-complete intersection contains $5$ low-penetration worlds and $15$ measured decisions per model.
<!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH.md:41-43. -->
Its $60$ s WM-minus-BB regret is $+5.102\,[-9.389,+16.746]$ veh-s; WM-minus-best-fixed is $+8.232\,[+1.692,+18.477]$ veh-s, with best-fixed identity reselected inside each bootstrap draw.
<!-- Source: E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH.md:45,48-51. -->
There is no demonstrated WM regret advantage on this conditional subset. Scheduling-dependent coverage excludes other traffic cells, so it is neither a full-panel benefit nor a full-panel null. Predictions optimize on-road delay with entries disabled, while evaluation includes fleet and pending-source delay.

The wall-inclusive comparison against learned pickers remains [PENDING: S-C1SWITCH2 complete paired WM/BB/fleet-PPO/per-CAV-MAPPO outcomes]. Its primary learned selector is fleet PPO; the per-CAV MAPPO comparator has a different joint action space. A delivered training checkpoint does not establish selection efficacy, and incomplete model episodes cannot be treated as a complete paired campaign. Table I reserves the comparison without substituting training progress or partial episodes for outcomes.

**Table I.** Pending wall-inclusive switching comparison. All outcome cells are deliberately unfilled; no performance estimate is available here.

Columns: selector; fleet/source cost; branch regret.

- WM selector: [--]; [--].
- BB-traj selector: [--]; [--].
- Fleet PPO picker: [--]; [--].
- Per-CAV MAPPO picker: [--]; [--].

## VI. Discussion and Limitations
The evidence favors separating prediction from verdict qualification. Historical wall response gains, conditional full-history PI gains and exploratory offline regret gains answer narrower questions than autonomous controller ranking. The rebuilt physics further shows that lower distributional error and correct sample-mean effect signs can coexist with an unacceptable breakdown-rate distance and overconfident calls. Failed calibration, queue/value and merge-repair routes in the source package reinforce this point: better local fit or bookkeeping is not sufficient long-run decision evidence.

Several threats constrain causal interpretation. Synthetic traffic shares structure with the physics prior, so this is not real-driver transfer. Checkpoint comparisons change corpora, world coverage, histories and physics together. Learned settings are not identified ground-truth driver types, and historical residual-only results do not validate their later wiring. Frozen in-house baselines also have unequal training and CAV-input contracts; their comparison cannot support superiority to all black-box predictors. Unbroken initialization exclusion, matched component ablations and full data/compute comparability remain unresolved.

Execution and population details matter equally. Controller resets can lose command memory, and common controller code does not imply identical realized actuation. The current feedback study retains a simplified common plant and failed exact cross-host parity despite passing its prospective numerical acceptance gate. Surviving initial vehicles omit synthetic entrants from endpoint matching; source waiting and road-only costs must remain distinct. Teacher-forced, shifted-start, recorded-command and autonomous panels cannot be combined as one replication. Likewise, selective switching completion and singleton-stratum intervals limit uncertainty claims rather than supplying evidence of a null effect.

For future use, validation should bind controller, plant, horizon, observation history and cost population, then test effect magnitude, direction, distributional calibration and decision regret jointly. Hidden-variable samples should represent uncertain drivers and arrivals rather than merely changing an inactive simulator seed. This is a recommended validation discipline, not a tested refusal policy or safety guarantee. Independently frozen successor physics and complete learned-picker outcomes may strengthen the evidence, but cannot retroactively convert the existing failed or partial gates into successes.

## VII. Conclusion
WM-S makes controller response prediction explicit by factoring executable CAV control from physics-informed human dynamics and open-road evolution. Recovered historical evidence supports unseen-controller response prediction under recorded CAV motion, and genuine history improves a conditional PI comparison. Autonomous endpoint superiority remains inconclusive, while short-horizon selection evidence does not establish long-horizon controller trust. The physics-faithful rebuild improves distributional and endpoint point estimates yet fails its breakdown-distance adoption condition and remains overconfident in action comparisons. The resulting lesson is constructive and bounded: structured traffic prediction can transfer across controller behavior, but reliable controller verdicts require their own protocol-specific evidence.

## Acknowledgment
Author, affiliation and funding information will be supplied by the owner before submission.

## References

[1] L. Liu, C. Xie, and X. Xiong, "Offline multi-agent reinforcement learning with a physics-informed world model for cooperative mixed traffic control," 2026, arXiv preprint, version 2. [Online]. Available: https://arxiv.org/abs/2608.17739

[2] Z. Sheng, Z. Huang, and S. Chen, "Traffic expertise meets residual RL: Knowledge-informed model-based residual reinforcement learning for CAV trajectory control," Communications in Transportation Research, vol. 4, p. 100142, 2024. [Online]. Available: https://arxiv.org/abs/2408.17380

[3] M. Treiber, A. Hennecke, and D. Helbing, "Congested traffic states in empirical observations and microscopic simulations," Physical Review E, vol. 62, pp. 1805–1824, 2000. [Online]. Available: https://arxiv.org/abs/cond-mat/0002177

[4] A. Kesting, M. Treiber, and D. Helbing, "General lane-changing model MOBIL for car-following models," Transportation Research Record, vol. 1999, no. 1, pp. 86–94, 2007. [Online]. Available: https://doi.org/10.3141/1999-10

[5] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado, "GAIA-1: A generative world model for autonomous driving," 2023, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2309.17080

[6] C. Gulino, J. Fu, W. Luo, G. Tucker, E. Bronstein, Y. Lu, J. Harb, X. Pan, Y. Wang, X. Chen, J. D. Co-Reyes, R. Agarwal, R. Roelofs, Y. Lu, N. Montali, P. Mougin, Z. Yang, B. White, A. Faust, R. McAllister, D. Anguelov, and B. Sapp, "Waymax: An accelerated, data-driven simulator for large-scale autonomous driving research," 2023, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2310.08710

[7] T. Salzmann, B. Ivanovic, P. Chakravarty, and M. Pavone, "Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data," 2020, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2001.03093

[8] C. Oefinger, F. R. Schäfer, K. Moller, M. Piccinini, and J. Betz, "Validate the dream before you trust its verdict: Admissibility for world-model simulators," 2026, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2607.07196

[9] S. Rahmani, S. Rasouli, D. Cornelisse, E. Vinitsky, B. van Arem, and S. C. Calvert, "Artificial intelligence for modeling and simulation of mixed automated and human traffic," 2026, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2604.12857

[10] P. A. Lopez, E. Wiessner, M. Behrisch, L. Bieker-Walz, J. Erdmann, Y.-P. Flotterod, R. Hilbrich, L. Lucken, J. Rummel, and P. Wagner, "Microscopic traffic simulation using SUMO," in 2018 21st International Conference on Intelligent Transportation Systems (ITSC), 2018, pp. 2575–2582. [Online]. Available: https://doi.org/10.1109/ITSC.2018.8569938

[11] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, "The surprising effectiveness of PPO in cooperative, multi-agent games," 2021, arXiv preprint. [Online]. Available: https://arxiv.org/abs/2103.01955
