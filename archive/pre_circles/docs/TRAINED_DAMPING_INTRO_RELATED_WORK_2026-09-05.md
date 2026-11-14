# Manuscript introduction and related-work draft

Editorial status: source-grounded draft for the restored trained-damping project,
2026-09-05. This is a targeted literature review, not an exhaustive novelty
search. The running controller has not passed acceptance, and the proposed RSU
escalation contribution has not been validated. Use with
`TRAINED_DAMPING_METHODS_DRAFT_2026-09-05.md`. No historical wall result is imported.

## Introduction draft

Automated vehicles can influence traffic beyond their own trajectories. In
controlled ring-road experiments, Stern and colleagues demonstrated that
controlling a single vehicle could dissipate stop-and-go waves among human
drivers. These experiments established a physical basis for traffic smoothing
with sparse mobile actuators. They also distinguished wave formation through
car following from congestion triggered by lane changes or geometric bottlenecks.
[Stern et al., field experiments](https://arxiv.org/abs/1705.01693)

Learning-based traffic control has an established foundation. Flow studies
mixed autonomy under partial control and observation, including a small neural
controller using local observations in a single-lane setting. Its results show
that local information can support learned traffic smoothing in a specified
environment; they do not determine the performance of a different controller
at an open highway merge.
[Wu et al., Flow](https://arxiv.org/abs/1710.05465)

The field has also progressed beyond ring-road demonstrations. Jang and
colleagues describe reinforcement-learning controllers developed for a 100-AV
highway field operational test, including simulation design, reward shaping,
deployment and safety considerations. Local learned damping itself is therefore
prior art, and its implementation alone is insufficient to establish this
project's intended contribution.
[Jang et al., RL oscillation dampening](https://arxiv.org/abs/2402.17050)

This project studies a specific control decision at a merge: when should an
infrastructure supervisor add coordinated intervention to an already useful
local damping controller? The comparison requires a credible standalone control
arm. If the nominal learned controller is harmful, an apparent gain from
escalation may simply compensate for that harm. Conversely, observing congestion
does not establish that the available CAVs can execute a useful coordinated
intervention. This motivates separating baseline validation, intervention
feasibility and incremental outcome value in the experimental design.

We formulate local damping as a shared per-CAV feedback policy with bounded
acceleration corrections and native safety handback. Its actor receives only
own-state, map-relative and range-limited front/rear observations over a short
history. Team costs during training include all requested trips, including
vehicles waiting at sources, and continue after individual CAVs leave. The
training critic and physical outcome logger may use privileged simulator state;
the deployed actor does not require them. The local information contract and
full-horizon accounting are explicit implementation choices, not claims of a
new reinforcement-learning algorithm.

After prospective validation, the accepted damping checkpoint D* will be frozen
as the counterfactual control arm. The proposed RSU will use timestamped CAV
reports to decide whether coordinated moving-wall control offers incremental
value. Sensing coverage, communication delay, command ownership and the complete
assembly/hold/release outcome must be represented in that evaluation. A global
simulator health score cannot stand in for information actually acquired by the
CAVs. The wall decision will require evidence of executable control and useful
outcomes relative to continuing D*.

The intended experimental questions are: (i) whether local learned damping
provides useful standalone stabilization while preserving service within a
declared operating range; (ii) whether a coordinated intervention can add useful
control without undermining that baseline; and (iii) whether CAV-reported
observations support a prospectively valid decision to escalate. At present,
only the first controller's implementation and training are available. Final
results, contribution claims and the abstract must await the corresponding
evaluation evidence.

## Related work draft

### Analytical and learned local smoothing

Explicit local feedback remains a relevant comparator. Hayat and colleagues
report highway experiments using local controllers to smooth traffic on I-24
as part of MegaVanderTest. Their study describes control design, deployment and
measurements of the surrounding traffic, providing a relevant reference for
assessing the practical value of an analytical local law.
[Hayat et al., explicit local controllers](https://arxiv.org/abs/2310.18151)

The Flow benchmark work releases mixed-autonomy tasks, implementations and
tutorials to support reproducible reinforcement-learning comparisons. This
supports evaluating a controller across fixed scenarios and reference arms,
rather than treating a positive training reward as evidence of general control
performance.
[Vinitsky et al., mixed-autonomy benchmarks](https://proceedings.mlr.press/v87/vinitsky18a.html)

For the current campaign, the analytical comparator is a frozen five-second
visible-leader velocity-matching law using the same sensing and command adapter
as the learned actor. That provides a controlled engineering comparison. It is
not a reproduction of FollowerStopper, PI-with-saturation, or the highway
controllers above, and it must not be described as the best available analytical
method. The final paper's baseline requirements depend on its eventual claims.
An assertion that learning is necessary would require evidence beyond merely
being noninferior to this one fixed comparator.

### CAV communication and mixed-traffic representations

Chen and colleagues propose a CAV–aggregated-HDV–CAV representation that groups
intervening human-driven vehicles and combines information for learned
car-following control. The study illustrates that communication and state
representation are substantive parts of a controller design.
[Chen et al., CAV-AHDV-CAV](https://arxiv.org/abs/2407.02517)

Our current damping actor uses onboard local measurements without inter-CAV
fusion. Fusion belongs to the subsequent RSU decision layer. Consequently,
success or failure of the damping actor cannot by itself establish that a
particular RSU observation set is sufficient. Those are different information
interfaces and prediction/control tasks. The final comparison must describe
which measurements are acquired locally, which are communicated, and which
variables exist only in simulator instrumentation.

### Coordinated moving bottlenecks and state reconstruction

Čičić and Johansson study accumulating controlled vehicles and using the
resulting moving bottleneck to dissipate waves in a multi-class cell-transmission
model. Thus, collecting CAVs into a traffic actuator is also established prior
work. Their macroscopic formulation offers a relevant mechanism comparison for
the proposed microscopic moving-wall implementation.
[Čičić and Johansson, accumulated moving bottlenecks](https://people.kth.se/~kallej/papers/transport_cdc19cicic.pdf)

Their later front-tracking framework combines local CAV measurements, traffic
state reconstruction, online model learning and prediction-based wave control.
It explicitly uses CAVs as sensors and actuators, with randomly arriving
vehicles. Its motivating model considers a homogeneous road without ramps,
and its control formulation differs from the proposed microscopic merge
experiment. This is nevertheless close prior art for a supervisor acting from
CAV reports; collecting those reports is not itself a novel contribution.
[Čičić and Johansson, reconstruction and control](https://people.kth.se/~kallej/papers/transport_transp22cicic.pdf)

The proposed contribution to investigate is the incremental decision over a
frozen, validated learned damping baseline, with feasibility, service and
harmful outcomes checked at the microscopic control interface. That is a
research target. Its novelty relative to reconstruction-based predictive
control and existing hierarchical traffic-control systems requires a fuller
method comparison, and its value requires prospective integrated results.

## Editorial consequences for the current project

The research supports investigating useful CAV-acquirable observations; a failed
feature screen in this repository does not establish that such observations
cannot exist. It also raises the standard for the paper: established work already
covers local learned damping, field deployment, CAV-based state reconstruction
and moving-bottleneck control. A generic claim that “we learn damping and use an
RSU” would be too broad to distinguish this project.

Keep the running campaign unchanged. This review does not authorize a new
baseline sweep, observation change, wall run or predictor dataset. The accepted
controller and later treatment gate remain prerequisites. Before making final
novelty claims, compare the nearest reconstruction-based and hierarchical
methods in detail, select claim-appropriate baselines under a separate fixed
protocol, and show incremental benefit against D* with honest uncertainty.

Current paper framing: **conditional coordination over validated local learned
damping, evaluated using CAV-acquirable information and full-trip outcomes**.
This preserves the owner's intended integrated paper. It neither establishes
publication readiness nor substitutes an analytical-only or wall-only result
for the required trained damping controller.

Source scope: foundational and selected recent primary papers were checked on
2026-09-05. The arXiv links identify the reviewed versions; publication metadata
should be normalized when preparing the final bibliography. Reported results
from other papers are their findings, not validation of the present code or
directly comparable effect sizes. No acceptance outcomes were accessed for
this drafting work.
