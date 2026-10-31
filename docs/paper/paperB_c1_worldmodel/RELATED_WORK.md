# Paper B — verified related-work and citation notes

## Verification rule

Every entry in `refs.bib` was checked against the arXiv abstract page or DOI resolution below during this task. Existence, author order, exact title and available publication fields were checked; the selected close-paper HTML sections were also read where indicated. No inaccessible venue metadata is filled from memory. Preprint entries deliberately remain preprints when only arXiv metadata was verified. Dates/identifiers in this file are bibliographic metadata, not experimental outcomes.

Verification date: October 31, 2026 (session date). Live pages may change; explicit versions below identify the reviewed content. Network fetching was read-only and required no browser login.

## Positioning that can survive review

**Safe distinction:** WM-S studies explicit executable-controller/environment factorization, microscopic traffic response under controller exclusion, and the limits of using predictions to rank/select controllers. Its evidence distinguishes teacher forcing, recorded longitudinal replay with feedback entrants, and full feedback, and retains road/source accounting failures.

**Do not claim:** first physics-informed traffic WM; first IDM-plus-neural residual; first traffic policy learned in imagination; first recognition that good-looking trajectories can judge actions incorrectly; universal admissibility certification; superiority to published forecasting or video WMs that were not run.

The intended full-feedback unseen-controller advantage remains `[[PENDING: eligible result / S-C1CLOSED]]`. Literature positioning does not fill that empirical gap.

## Closest work — physics-informed traffic WMs

### `liu_physics_wm` — Liu, Xie and Xiong

- **Verified title:** *Offline Multi-Agent Reinforcement Learning with a Physics-Informed World Model for Cooperative Mixed Traffic Control*.
- **Verified authors:** Lu Liu, Chi Xie, Xi Xiong.
- **Page:** https://arxiv.org/abs/2608.17739 ; reviewed HTML https://arxiv.org/html/2608.17739v2 .
- **Metadata:** preprint; reviewed revision October 9, 2026; no journal/conference acceptance inferred.
- **What the paper does:** reconstructs cell density/speed and microscopic CAV state from local histories; coupled macro/micro traffic physics regularizes state inference. The physics is explicitly described as a soft supervision prior, not the transition function. An ensemble learns next state and reward; pessimism and uncertainty truncation support offline cooperative policy learning. Review basis: abstract and Methodology sections on inference, transition/reward model and imagination; numerical setup/evaluation inspected.
- **Overlap:** partially observed mixed-traffic bottleneck; physics, learned WM and multi-CAV control; action-conditioned imagined trajectories.
- **Defensible distinction:** WM-S embeds explicit microscopic physics in its transition and executes controller code separately, then tests controller exclusion, regret and verdict limits. Their inspected experimental sections emphasize reconstruction/next-state and learned-policy performance; they do not supply our declared excluded-controller comparison or road/source paired-error account. Say “our evaluation emphasizes ...”, not “they cannot generalize”.
- **Must cite:** introduction motivation and related-work paragraph on closest traffic WMs. This is close prior art, not a peripheral survey citation.

### `sheng_residual` — Sheng, Huang and Chen

- **Verified title:** *Traffic expertise meets residual RL: Knowledge-informed model-based residual reinforcement learning for CAV trajectory control*.
- **Verified authors:** Zihao Sheng, Zilin Huang, Sikai Chen.
- **Page:** https://arxiv.org/abs/2408.17380 ; reviewed HTML https://arxiv.org/html/2408.17380v2 .
- **Publication fields verified on arXiv:** *Communications in Transportation Research*, volume 4, article 100142 (2024), DOI https://doi.org/10.1016/j.commtr.2024.100142 . Later arXiv revision is not a different publication year.
- **What it does:** IDM base dynamics plus neural residuals, and residual RL over a physics-based initial policy for stop-and-go-wave mitigation. Both model residual and policy residual are explicit. Review basis: abstract, introduction, Methodology on virtual dynamics and residual policy.
- **Overlap:** direct, substantive prior for IDM-plus-neural residual modeling and model-based traffic control.
- **Defensible distinction:** WM-S's controller-exclusion/feedback contracts, learned-settings/gating/merge bookkeeping family and paired controller-verdict evaluation; no novelty claim for additive residual dynamics. Do not say the prior has no generalization study: it includes a transferability-analysis section; that alone does not equal our controller-exclusion protocol.

## Verdict trust and traffic-simulation context

### `oefinger_admissibility`

- **Verified title/authors:** *Validate the Dream Before You Trust Its Verdict: Admissibility for World-Model Simulators*; Christian Oefinger, Finn Rasmus Schäfer, Korbinian Moller, Mattia Piccinini, Johannes Betz.
- **Page:** https://arxiv.org/abs/2607.07196 ; reviewed HTML https://arxiv.org/html/2607.07196v1 .
- **Status:** arXiv preprint; page says accepted at the RSS Robot World Models workshop. BibTeX uses the verified preprint, without implying a main-conference paper.
- **What it supports:** world models used as policy test oracles require action-conditioned validation and declared limits; generation quality and action-following need not agree. The proposed ladder is not an externally adopted certification standard. Review basis: abstract, introduction, admissibility ladder, discussion and experiment-scope caveats.
- **Distinction:** we operationalize traffic-specific effects/regret, source accounting and regional error on simulator evidence. We neither originate the trust question nor claim to satisfy their ladder. Their paper targets generative video WMs; our structured state-space model and simulator-to-simulator evidence have different validation objects.

### `rahmani_survey`

- **Verified title:** *Artificial Intelligence for Modeling and Simulation of Mixed Automated and Human Traffic*.
- **Authors:** Saeed Rahmani, Shiva Rasouli, Daphne Cornelisse, Eugene Vinitsky, Bart van Arem, Simeon C. Calvert.
- **Page:** https://arxiv.org/abs/2604.12857 (reviewed abstract page).
- **Use:** taxonomy of agent-level behavior, environment simulation, cognitive/physics-informed methods; motivates evaluating surrounding traffic rather than only ego policy. Page says submitted to IEEE; no accepted venue is asserted.
- **Do not use for:** empirical efficacy, proof of novelty, or claims that all existing simulators fail our tasks.

## Analytical originals

### `treiber_idm`

- Martin Treiber, Ansgar Hennecke, Dirk Helbing, *Congested Traffic States in Empirical Observations and Microscopic Simulations*.
- Verified https://arxiv.org/abs/cond-mat/0002177 ; journal fields there: *Physical Review E* 62, 1805–1824 (2000), DOI https://doi.org/10.1103/PhysRevE.62.1805 .
- Use for IDM equations/prior, not for WM-S's particular residual/merge implementation or safety guarantees.

### `kesting_mobil`

- A. Kesting, M. Treiber, D. Helbing, *General Lane-Changing Model MOBIL for Car-Following Models* (initials as returned by the DOI metadata).
- Verified DOI page https://doi.org/10.3141/1999-10 : *Transportation Research Record* 1999(1), 86–94 (2007).
- Use for the incentive/safety analytical lane-changing prior. Historical and learned-discretionary WM-S variants must be distinguished. MOBIL is not synonymous with SUMO's lane logic.

## Learned driving WMs and interactive simulation

### `hu_gaia1`

- Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado, *GAIA-1: A Generative World Model for Autonomous Driving*.
- Verified https://arxiv.org/abs/2309.17080 (abstract page).
- Use: learned video/text/action-conditioned driving generation as context. Distinction: our explicit traffic-state responses and complete-population cost can be evaluated directly. No claim that GAIA-1 lacks action conditioning or that it was beaten experimentally.

### `gulino_waymax`

- Cole Gulino and coauthors, *Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research*.
- Verified https://arxiv.org/abs/2310.08710 (abstract page; full author order in `refs.bib`).
- Use: interactive multi-agent simulation with learned and hard-coded behavior, and overfitting to simulated agents. Distinction: a general simulation platform versus our controller-exclusion study and physics/residual population accounting. Do not present the platform itself as a single learned WM baseline.

## Forecasting context versus actual in-house black-box baselines

### `salzmann_trajectronpp`

- Tim Salzmann, Boris Ivanovic, Punarjay Chakravarty, Marco Pavone, *Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data*.
- Verified https://arxiv.org/abs/2001.03093 (abstract page; ECCV is named on the page; BibTeX uses the original-year preprint).
- Graph-structured recurrent, dynamics-aware prediction, optionally conditioned on ego motion plans. Do not claim all black-box forecasters ignore dynamics, map information or ego actions.

### `nayakanti_wayformer`

- Nigamaa Nayakanti, Rami Al-Rfou, Aurick Zhou, Kratarth Goel, Khaled S. Refaat, Benjamin Sapp, *Wayformer: Motion Forecasting via Simple & Efficient Attention Networks*.
- Verified https://arxiv.org/abs/2207.05844 (abstract page).
- Attention-based fusion for heterogeneous scene information and trajectory forecasting. Use as forecasting context only; no evaluated WM-S versus Wayformer claim.

### Actual baselines in this project

- `E:/agent-work/c4-paperb/src/wms/model/factory.py`: `blackbox_traj` is the in-house physics-free trajectory model; `blackbox_id` is the controller-ID-conditioned learned-CAV variant. `src/wms/model/spec.py` documents the latter's learned movers.
- `E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md`, caveats: BB-id ignores the recorded teacher, whereas BB-traj is teacher matched. An unavailable feedback BB-id comparison is unavailable, not evidence of failure.
- These models are not implementations of the published forecasters cited above. Their provenance is the project code and training/score receipts; adding an external citation cannot complete missing baseline comparability.
- `[[PENDING: final C1 baseline architecture/training/corpus/exposure table / S-C1GAP + S-C1CLOSED]]`.

## Experimental platform and comparator provenance

### `lopez_sumo`

- P. A. Lopez and coauthors, *Microscopic Traffic Simulation using SUMO* (initials as returned by the DOI metadata).
- Verified DOI https://doi.org/10.1109/ITSC.2018.8569938 : ITSC (2018), 2575–2582; author list in BibTeX follows the returned DOI metadata.
- Cite simulator platform. The paper does not verify our installed simulator version, network, request ledger or actuation adapter.

### `yu_mappo`

- Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu, *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*.
- Verified https://arxiv.org/abs/2103.01955 (reviewed abstract page, latest revision November 4, 2022; accepted NeurIPS Datasets and Benchmarks per page).
- Cite algorithmic comparator, not the identity/budget/execution of a specific frozen traffic checkpoint. BibTeX retains the verified original-year preprint.

## Citation policy for the owner

- Cite the mandatory closest physics/residual and admissibility works even if space is tight. Use the survey to compress broad context.
- Keep the optional forecasting/platform context concise; cite only entries actually used in prose. `refs.bib` is a verified pool, not an instruction to print uncited references.
- No first-of-kind claim and no borrowed empirical numbers from literature are needed for this paper's argument.
- No unrelated arXiv candidates encountered during lookup are included in the bibliography.
