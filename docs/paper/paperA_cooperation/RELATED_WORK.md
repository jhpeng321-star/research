# Related work — verified identities and positioning

Verification date: 2026-10-31. Every entry actually included in `refs.bib` was checked through its arXiv abstract page or DOI endpoint. Discovery searches are not substitutes for identity checks. `reference_verification.csv` is the compact local verification record; each entry below cites that source file and supplies the live page. Content comparisons use abstracts unless explicitly marked full-text/section inspected. Preprints are cited as preprints, not invented venue acceptances.

## Positioning spine for the owner

- **Credit assignment precedent:** Counterfactual subtraction and counterfactual policy learning are established. C_G asks whether a whole group's longitudinal intervention saves more full-population delay than its component solo interventions. M_i instead measures a member's marginal contribution in a team. The audit's contribution is the specified traffic intervention/replay/measurement protocol and its empirical findings, not subtraction itself.
- **Traffic precedent:** Cooperative merging, AV coalition reward reallocation, mixed-autonomy RL and real-road coordinated control already exist. Fleet return, cooperation incentives and observed coordination are related but distinct from the audit's simulator-exact excess-over-solos estimand.
- **Closest model-based precedent:** MACD already learns from counterfactual imagined trajectories; Liu–Xie–Xiong already combine physics-informed traffic state inference, a learned model and offline MARL. Novelty must be stated around measured intervention interaction, oracle-versus-surrogate fidelity and the tested learning result.
- **Evaluation precedent:** Counterfactual planning already formalizes contextual cooperation; process-aware evaluation already rejects return as a complete coordination description; world-model admissibility already emphasizes action-response validity. Our proposed contribution is a traffic-specific exact-replay operationalization and quantified reliability boundary.

## Counterfactual credit and coalition values

### `wolpert_tumer2002` — Wolpert and Tumer, difference-reward lineage
- **Verified identity:** “Collective Intelligence, Data Routing and Braess' Paradox,” *Journal of Artificial Intelligence Research*, **16**, **359–387 (2002)**; DOI [10.1613/jair.995](https://doi.org/10.1613/jair.995). <!-- Source: reference_verification.csv, key=wolpert_tumer2002; DOI page verified 2026-10-31. -->
- **Use:** credit-assignment/collective-intelligence ancestry. Direct difference-reward equations and their historical attribution are also visible in Dr.Reinforce's inspected background section.
- **Distinction:** C_G compares coalition value with summed solo values; it is not merely an individual difference utility. Do not claim first use of counterfactual subtraction.
- **Scope of check:** DOI identity; detailed historical-priority comparison remains [[PENDING: inspect foundational full text if making priority statements / owner literature pass]].

### `agogino_tumer2008` / `tumer_agogino2009` — air-traffic precedent and venue correction
- **Verified identities:** Agogino and Tumer, “Regulating Air Traffic Flow with Coupled Agents,” *AAMAS*, **535–542 (2008)**, DOI [10.65109/uoia7063](https://doi.org/10.65109/uoia7063); Tumer and Agogino, “Improving Air Traffic Management with a Learning Multiagent System,” *IEEE Intelligent Systems*, **24(1), 18–21 (2009)**, DOI [10.1109/MIS.2009.10](https://doi.org/10.1109/MIS.2009.10). <!-- Source: reference_verification.csv, keys=agogino_tumer2008,tumer_agogino2009; DOI pages and official AAMAS contents verified 2026-10-31. -->
- **Use:** multi-agent congestion control and agent/global objective alignment predate the present traffic application.
- **Correction:** the brief's **“AAAI 2008”** identity was not established. The official AAMAS contents confirm the coupled-agents paper; Crossref sometimes gives a generic/incorrect proceedings-volume container, so `refs.bib` uses the official AAMAS venue rather than copying that field. **[[PENDING: identify any distinct intended AAAI air-traffic paper, with title and DOI/arXiv page / owner literature pass]].** No fabricated AAAI entry is included. <!-- Source: reference_verification.csv, key=agogino_tumer2008; owner related-work pointer. -->
- **Distinction:** transfer of credit-assignment ideas to mixed highway merging, with full branch-defined interaction rather than a broad “first congestion MARL” claim.

### `foerster_coma2018` — COMA
- **Verified identity:** Foerster, Farquhar, Afouras, Nardelli and Whiteson, “Counterfactual Multi-Agent Policy Gradients,” *AAAI*, **2018**; [arXiv:1705.08926](https://arxiv.org/abs/1705.08926), DOI [10.1609/aaai.v32i1.11794](https://doi.org/10.1609/aaai.v32i1.11794). <!-- Source: reference_verification.csv, key=foerster_coma2018. -->
- **Known mechanism:** a centralized Q critic marginalizes a single agent's action while keeping other actions fixed, to train decentralized actors.
- **Distinction:** our audit measures finite-horizon outcomes from physical simulator interventions; the estimand is group excess over native-background solo effects. This does not invalidate COMA's credit objective or establish superior learning without a comparison.

### `castellini_drreinforce` — Dr.Reinforce
- **Verified identity:** Castellini, Devlin, Oliehoek and Savani, “Difference Rewards Policy Gradients,” [arXiv:2012.11258](https://arxiv.org/abs/2012.11258). The arXiv page also links DOI [10.1007/s00521-022-07960-5](https://doi.org/10.1007/s00521-022-07960-5) and records an AAMAS extended-abstract version. Cite the verified arXiv version to avoid mixing journal online/issue metadata. <!-- Source: reference_verification.csv, key=castellini_drreinforce. -->
- **Inspected:** HTML background and method sections. It differences the known immediate reward, or learns a reward network, instead of learning a Q critic for the subtraction.
- **Distinction:** complete longitudinal traffic branches capture downstream state/queue effects; instantaneous reward differencing on a fixed trajectory and trajectory-level intervention replay are not interchangeable. Do not transfer its policy-gradient guarantees to our frozen-continuation local surrogate.

### `chai_macd2024` — MACD
- **Verified identity:** Chai, Fu, Zhao and Zhu, “Aligning Credit for Multi-Agent Cooperation via Model-based Counterfactual Imagination,” *AAMAS*, **281–289 (2024)**; DOI [10.65109/awzc1597](https://doi.org/10.65109/awzc1597). <!-- Source: reference_verification.csv, key=chai_macd2024. -->
- **Inspected:** official proceedings first-page PDF text in memory. MACD expands to **Multi-Agent Counterfactual Dreamer**; it uses centralized imagination with decentralized execution and an additional learned-model counterfactual trajectory.
- **Distinction:** this is direct prior art for model-based counterfactual learning. Our intended novelty is exact traffic-interaction measurement plus explicit oracle/surrogate disagreement, not “first counterfactual Dreamer.” Do not invent an arXiv identifier; DOI and official PDF are verified.

### `han_shapley2022` — AV Shapley reward reallocation
- **Verified identity:** Han, Wang, Su, Shi and Miao, “Stable and Efficient Shapley Value-Based Reward Reallocation for Multi-Agent Reinforcement Learning of Autonomous Vehicles,” [arXiv:2203.06333](https://arxiv.org/abs/2203.06333); abstract page states ICRA acceptance. <!-- Source: reference_verification.csv, key=han_shapley2022. -->
- **Known mechanism:** transferable-utility coalition reward reallocation; stability result depends on the convex-game assumption.
- **Distinction:** C_G is not a fair allocation or a stable-coalition certificate; it evaluates total nonadditive delay saving under a particular baseline. Cite this work when saying cooperation incentives for AVs are established.

## Mixed-autonomy and merging control

### `chen_merge2021` — on-ramp MARL
- **Verified identity:** Chen, Hajidavalloo, Li, Chen, Wang, Jiang and Wang, “Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic,” [arXiv:2105.05701](https://arxiv.org/abs/2105.05701). <!-- Source: reference_verification.csv, key=chen_merge2021. -->
- **Known mechanism:** shared parameters, local rewards, action masks, priority safety supervisor and curriculum under dynamic communications.
- **Distinction:** shared rewards/local shaping can foster coordinated behavior; our audit asks a separately defined excess-over-solos outcome question. Its controller architecture, scenario and safety supervisor differ; do not present a local frozen MAPPO result as a direct reproduction or defeat of Chen et al.

### `wu_flow2022` — Flow
- **Verified identity:** Wu, Kreidieh, Parvate, Vinitsky and Bayen, “Flow: A Modular Learning Framework for Mixed Autonomy Traffic,” [arXiv:1710.05465](https://arxiv.org/abs/1710.05465); journal DOI [10.1109/TRO.2021.3087314](https://doi.org/10.1109/TRO.2021.3087314) verifies *IEEE Transactions on Robotics*, **38(2), 1270–1286 (2022)**. <!-- Source: reference_verification.csv, key=wu_flow2022. -->
- **Use:** mixed-autonomy traffic-control RL foundation and modular simulator-based experimentation.
- **Distinction:** no claim to invent low-penetration CAV traffic regulation. Performance in another network/plant is not a benchmark result for the present adapter.

### `lee_circles2024` — CIRCLES open-road control
- **Verified identity:** Lee et al., “Traffic Control via Connected and Automated Vehicles: An Open-Road Field Experiment with 100 CAVs,” [arXiv:2402.17043](https://arxiv.org/abs/2402.17043). <!-- Source: reference_verification.csv, key=lee_circles2024. -->
- **Known mechanism:** hierarchical MegaController with centralized speed planning and local vehicle control, using third-party/live vehicle feeds.
- **Critical scope:** the frozen CIRCLES ONNX adapter in this repository is a released damping-controller artifact, not the full field-test architecture. Its provenance explicitly says the exact bridge to a publication's evaluated checkpoint is unverified. No claim that this audit tests the entire consortium system or reproduces field performance.
- **Evidence:** `E:/agent-work/c4-papera/baselines/circles/provenance.json`, publication identity and transfer limitations. [[PENDING: exact publication-to-evaluated-checkpoint lineage if required / baseline owner]].

## Closest modeling and evaluation papers

### `liu_physics_wm2026` — physics-informed offline traffic MARL
- **Verified identity:** **Lu Liu, Chi Xie and Xi Xiong**, “Offline Multi-Agent Reinforcement Learning with a Physics-Informed World Model for Cooperative Mixed Traffic Control,” [arXiv:2608.17739v2](https://arxiv.org/abs/2608.17739v2). <!-- Source: reference_verification.csv, key=liu_physics_wm2026. -->
- **Inspected:** HTML introduction and methods, including state inference, ensemble dynamics/rewards and policy learning. Physics supplies macroscopic/microscopic consistency supervision for reconstructed cell states; the learned ensemble supplies transitions and rewards. Offline imagined MAPPO uses pessimism and uncertainty truncation. This is much closer prior art than generic trajectory prediction.
- **Positioning:** Paper A's lead contribution should be exact longitudinal coalition-interaction measurement and its tested consequences. WM-S's executable-controller/per-vehicle structure and measured failure boundaries can be contrasted with their cell-state physics-supervised ensemble. Both use limited CAV histories in their stated deployment contract; do not claim partial observability or physics-informed cooperative traffic control as new.
- **Evidence boundary:** their published performance is not a matched baseline here. Our surrogate's unqualified status prevents claiming superior reliability or deployment effectiveness.

### `barnett_counterfactual2026` — counterfactual cooperation measurement
- **Verified identity:** Barnett, Wantlin and Adams, “Measuring Cooperation with Counterfactual Planning,” *Game Theory and AI for Security*, LNCS **16223**, **23–41**, DOI [10.1007/978-3-032-08064-6_2](https://doi.org/10.1007/978-3-032-08064-6_2). Publisher “Cite this paper” and BibTeX give **2026**; first online is **2025-10-12**. DOI content negotiation gives the online year. Use publisher BibTeX year and retain the distinction. <!-- Source: reference_verification.csv, key=barnett_counterfactual2026; Springer page and citation export. -->
- **Inspected:** publisher abstract and accessible context. Their measure contrasts behavior with a self-interested actor in context and a specified horizon, via counterfactual planning.
- **Distinction:** our native-release excess-over-solos is a different operational definition; native driving is not generally the self-interested optimal response. We measure beneficial interaction, not all meanings of cooperation. This paper is essential prior art against broad measurement novelty.

### `cardei_coordination2026` — process-aware evaluation
- **Verified identity:** Cardei, Landers and Doryab, “Coordination Matters: Process-Aware Evaluation of Cooperative Multi-Agent Reinforcement Learning,” [arXiv:2605.06557v2](https://arxiv.org/abs/2605.06557). Abstract page says under review. <!-- Source: reference_verification.csv, key=cardei_coordination2026. -->
- **Known scope:** interference, participation, coordinated task completion and coordination efficiency complement aggregate outcomes.
- **Distinction:** observable gap/yield/wall signatures supply process description; simulator member-release outcomes supply intervention-specific evidence. Neither replaces the other. Do not assert their diagnostics are equivalent to C_G or that return-only critique is new.

### `oefinger_validate2026` — world-model admissibility
- **Verified identity:** Oefinger, Schäfer, Moller, Piccinini and Betz, “Validate the Dream Before You Trust Its Verdict: Admissibility for World-Model Simulators,” [arXiv:2607.07196](https://arxiv.org/abs/2607.07196); abstract page states RSS Workshop on Robot World Models acceptance. <!-- Source: reference_verification.csv, key=oefinger_validate2026. -->
- **Known scope:** action-following and admissibility for using a learned simulator as a test oracle, distinct from visual realism.
- **Distinction:** our quantitative traffic-cost/interaction fidelity and queue-horizon diagnosis instantiate a related concern. Do not frame the need to validate action-conditioned WM verdicts as an original general principle.

## Implementation citations

- **`yu_mappo2022`:** verified MAPPO method paper, “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games,” [arXiv:2103.01955](https://arxiv.org/abs/2103.01955), accepted NeurIPS **2022**. The paper is method provenance; each local checkpoint still needs its own receipt. <!-- Source: reference_verification.csv, key=yu_mappo2022. -->
- **`lopez_sumo2018`:** verified “Microscopic Traffic Simulation using SUMO,” DOI [10.1109/ITSC.2018.8569938](https://doi.org/10.1109/ITSC.2018.8569938), ITSC **2018**, **2575–2582**. This cites the simulator; replay/version exactness is task evidence, not established by the citation. <!-- Source: reference_verification.csv, key=lopez_sumo2018. -->

## Suggested citation economy

Use the three outline paragraphs: credit ancestry/COMA/Dr.Reinforce/MACD; traffic/AV reward allocation/Flow/CIRCLES; contextual/process-aware measurement and WM reliability/Liu. Place MAPPO and SUMO method citations in experiments. Full identities are verified; a completed exhaustive novelty search and all full-text comparisons are not claimed. Keep the unresolved AAAI pointer in this package, not in the manuscript bibliography.
