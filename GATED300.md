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

## CCT-1 calibration receipt (INFORMATIONAL, not the verdict panel)

`tools/gated300/cct1.py INPUT --panel calibration|validation|pairs300 --out FILE`
reads the existing gate's `rows[].dJ[h]` (flat or grouped arms), preserves every
arm/start, excludes real-zero pairs only from sign counts, and uses all pairs
for confidence coverage. No wrong confident signs means mean wrong magnitude 0.
World-cluster 95% percentile intervals use 2000 draws, seed 0. `H_C=0` means
even 90 s failed. The legacy threshold and horizon results are retained verbatim.
Calibration/validation artifacts have `policy_authorized=false`; only the
explicit `pairs300` verdict role can supply trainer authority. Output creation
is exclusive (no input/output overwrite).

One read-only gpuR transaction fetched exactly the three requested selection
files (no WM inference, simulator, or remote writes). Evidence and source-byte
SHA256s: `tools/gated300/evidence/*_source.json` and `*_CCT1.json`.
Each panel has 108 arm pairs / 36 world clusters, sampled K1 as recorded in
the original files. These are calibration diagnostics, NOT a WM adoption verdict.

Numbers below list horizons in order **90 / 120 / 180 / 300 s**:

- **MACRO selection 0000**, WM `b8797242`: confident sign agreement
  **.6500 / .6522 / .8333 / .8235**; coverage
  **.3704 / .4259 / .5000 / .4722**; wrong-call mean |real dJ| veh-s
  **29.00 / 26.02 / 165.40 / 414.47**. All horizons FAIL; informational H_C=0.
- **MACRO selection 0030**, WM `030c5870`: confident sign agreement
  **.6829 / .6667 / .7966 / .8596**; coverage
  **.3796 / .4444 / .5463 / .5278**; wrong-call mean |real dJ| veh-s
  **24.20 / 29.62 / 96.97 / 254.85**. All horizons FAIL; informational H_C=0.
- **W1_PAIRED selection 0010**, WM `8a04b7d7`: confident sign agreement
  **.7143 / .7083 / .7778 / .8182**; coverage
  **.3889 / .4444 / .5000 / .5093**; wrong-call mean |real dJ| veh-s
  **28.44 / 24.66 / 97.09 / 362.26**. All horizons FAIL; informational H_C=0.

All coverages clear .25; the 90/120 s accuracies fail .80. At 300 s the
accuracies clear .80 but wrong-call mean magnitudes exceed 87.82 veh-s.
Full unrounded values, intervals and unchanged all-pairs numbers are in JSON.
Six CCT-1 CPU tests passed on Python 3.13 / torch CPU before this step's commit.
