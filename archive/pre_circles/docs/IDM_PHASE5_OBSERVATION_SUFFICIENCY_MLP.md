# IDM Phase-5 nonlinear observation-sufficiency diagnostic

Frozen after the corrected 128-feature linear diagnostic produced AUROC 0.614 and balanced accuracy 0.653. The linear result remains a failed linear-probe artifact; it does not establish that a nonlinear PPO actor lacks information.

The existing `idm_phase5_observation_sufficiency_v2` samples are reused without new SUMO data.

- Inputs: actor-visible indices 0–127 only.
- Forbidden: centralized-critic tail, demand, seed, arm, filename, and future outcome.
- Split: leave one SUMO seed out; all samples from that seed are held out together.
- Preprocessing: training-fold mean/std only.
- Model: MLP 128 -> 64 ReLU -> 16 ReLU -> 1 logit.
- Optimizer: Adam, learning rate 0.001, weight decay 0.001, 300 full-batch epochs.
- Weighting: equal total weight per demand/seed cell, then equal total class weight.
- Determinism: fixed initialization seed 20260827 plus held-out seed.
- Primary gates: pooled AUROC >=0.70 and balanced accuracy >=0.65.
- Negative control: the same pipeline with cell labels deterministically permuted; real-label AUROC must exceed shuffled AUROC by >=0.10.

If this diagnostic fails, the observation contract must be expanded and recollected before RL training. Thresholds will not be relaxed based on its outcome.