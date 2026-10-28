# S-GATED300

Worktree `E:/agent-work/c4-gated300`, branch `c4/gated300`, based on
`c4/c4prep` `f5bbeec0`. Owner authority: 2026-10-28 22:05; CCT-1 registration
at 22:10 in the active C4 registration. Default OFF; zero SUMO.

## Frozen contract

Dense trusted credit remains at 90 s, with candidate A's existing 10→90 s
curriculum. Long credit is bounded by a WM-specific CCT-1 artifact and 300 s.
Missing authority or H_C ≤90 means exact existing execution. Noise sigma in
veh-s: 90=25.38, 120=41.22, 180=42.11, 300=43.91 (chaotic-sensitivity proxy).
Confident means strictly |predicted dJ| >2 sigma. CCT-1 requires sign ≥0.80,
coverage ≥0.25, and mean absolute real dJ of wrong confident calls ≤2 sigma.
H_C is the longest contiguous passing prefix of 90/120/180/300 s.

Reward is −J, so paired reward is improvement = −(treatment−reference cost).
Each start and logical all/individual/group/coop/team credit row gates on its
K8 median and two-sided pessimistic sign: improvement q10>0; harm q90<0.
Total credit is dense + w_long × gated long, w_long=1.0 fixed prospectively,
never fitted/tuned to outcomes. No critic or value function for long credit.

Implementation and measured verification are recorded in subsequent commits.
