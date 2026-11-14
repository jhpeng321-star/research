# Literature check: cooperative multi-CAV control trained inside a learned world model (2026-10-09)

Coordinator subagent (web search). [V] = verified this session, [V*] = exists but details unconfirmed, [U] = unverified.

## Closest prior work: cite and differentiate
1. **Liu, Xie & Xiong (2026), arXiv 2608.17739 [V].**
   - What they do: offline MARL with a physics-informed (macro+micro supervised) probabilistic ensemble world model; MAPPO/CTDE trained on imagined rollouts with pessimism and uncertainty truncation; SUMO on-ramp bottleneck.
   - VERY CLOSE - the main paper to differentiate from.
   - Ours differs: online-iterative; executes CAV controller code; shown to generalise to unseen controllers.
2. **Sheng, Huang & Chen (2024), Communications in Transportation Research 4:100142 [V].**
   - What they do: knowledge-informed model-based residual RL; physics + NN residual virtual environment (Dyna-style).
   - Differs: single-agent, no out-of-distribution validation.
3. **Du et al., DMPO, IROS 2022 [V].**
   - What they do: decentralised model-based MARL with communicated predictions, on Flow / CACC.
4. **Chen et al., IEEE T-ITS 24(11) 2023 [V].**
   - What they do: model-free shared-parameter MARL for on-ramp merging in mixed traffic.
   - Role: the merging baseline.
5. **Zheng, Son & Lin, ICRA 2023 [V*].**
   - What they do: analytic policy gradients through IDM in a differentiable traffic sim.
   - Also cite: Son et al. 2022 (differentiable hybrid sim).

## Background
- **Flow and mixed-autonomy benchmarks:** Flow (Wu et al.); Vinitsky et al. CoRL 2018 (lane-blocking reward hacking); Vinitsky, Lichtle, Parvate & Bayen, ACM TCPS 2023 (decentralised AVs as moving bottlenecks).
- **Deployed RL controllers:** CIRCLES (Jang et al. 2024; Lichtle et al. 2024).
- **MAPPO/CTDE V2X merging:** Ji et al. 2025 and variants.
- **Graph MARL:** MARL-CAVG.
- **Model-based MARL:** MAMBA (AAMAS 2022), MAG, MARIE, CALL (V2V latent sharing).
- **Dyna-style macro platooning:** Zou, Jin & Xiong 2024.
- **Imagination / Dreamer:** Dreamer-SAC (short horizons work best); DreamerV3 for signal control.
- **Moving-bottleneck theory:** Ramadan & Seibold.

## Novelty assessment
- **Already done:**
  - model-free MARL/CTDE for merging;
  - physics+residual virtual environments (single-agent);
  - decentralised model-based MARL on Flow;
  - IDM analytic gradients;
  - imagined-rollout offline MARL with a physics-informed WM on a SUMO on-ramp (Liu et al. 2026).
- **Appears new:**
  - a WM that executes arbitrary CAV controller code and is SHOWN to generalise to unseen controllers (C1), hence a valid training ground for off-distribution policies;
  - differentiable gated physics residual for analytic multi-agent gradients plus a value function bridging the ~5 min benefit horizon (EFFH-02);
  - a shared decentralised V2V/RSU policy over 2-40% penetration;
  - emergent wall/metering validated in SUMO.
- **Expected reviewer objections:**
  - model exploitation (report SUMO-measured vs WM-predicted gains; uncertainty penalties; DAgger-style data refresh);
  - compounding error over 300 s (our 120 s blow-ups);
  - analytic gradients through discrete MOBIL/gates (compare with PPO in-model);
  - baselines: model-free MAPPO at an equal SUMO budget, DMPO/MAMBA-style, PI and the wall law;
  - circularity of SUMO IDM vs an IDM-structured WM;
  - reward hacking via ramp/source waiting (report all service);
  - V2V/RSU assumptions at 2% penetration.
