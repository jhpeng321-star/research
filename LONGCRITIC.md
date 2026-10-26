# S-LONGCRITIC-1 (2026-10-26)

Branch `c4/longcritic`, base `d6a426c6` (committed mainstart tip).
Registered in the shared REGISTRATION.md before offline execution. **Not activated.**

## Implementation

Three independent default-OFF gates:

1. `[longcritic].enabled`: retain authoritative real-return regression; require
   `real_return_mix=1`, gamma=1. A separate native critic is warmed exclusively
   on real native returns, then frozen. Auxiliary regression fits
   `V_policy(s) - stop(V_native_real(s))` to a shrunk WM policy-minus-native
   **reward difference**, never a WM return level. The policy critic remains the
   short-imagination terminal bootstrap, so its learned state derivatives carry
   delayed benefit into actor gradients. One original TRAIN start per update,
   rotated deterministically; two CRN rows, all-policy/native; horizons rotate
   among supported 60/120/180/240/300 s measurements. Histories, driver futures,
   arrivals, lane draws and policy noise stay paired. Long rollouts are no-grad;
   they skip target-critic evaluation. Entrant capacity scales with horizon.
   Native critic/optimizer/warm status are checkpointed. After the selected
   difference horizon, residual policy-native advantage is assumed zero; this is
   a finite-horizon approximation, not evidence for benefits beyond 300 s.
2. `[exploration].long_persistence`: existing group AR(1) e-folding time ramps
   30→60 s at curriculum positions 0→150, training imagination only. Each
   rollout resets noise; deployed actor/executor and MAPPO are unchanged.
3. `collect.long_group.enabled`: one additional GROUP arm at 300 s per world;
   point index `(world_index + cycle) % branch_points`, stable across retries.
   Reuse existing 300 s B/ALL prefixes; save a separate long credit record with
   full source-plus-road ledger. No short ONLY/FLIP arm is extended. Long-arm
   incidents enter the loop stop accounting.

## Measured reliability and accounting

Weight = `max(0,2*sign_lower95-1) * max(0,corr_lower95)^2`, using existing
group-bootstrap intervals, further multiplied per start by
`mean_delta²/(mean_delta² + variance_delta/n_futures)`. Both target and auxiliary
loss are shrunk. Missing/stale horizons yield zero; no extrapolation or manual
weight. gpuR's 13E panel has 96 groups/192 pairs: at 60 s lower bounds
0.898909/0.715742 imply **0.40871** before future shrinkage. This panel measures
t0-HDV differences (a proxy, not full-system validation). The repaired round-16
panel now supplies 60 s lower bounds 0.920213/0.678008, weight **0.386339**;
panel SHA256 `9ddb32143a16bea2df5dafd157bcbd86a806fc316824717db9f3d6c12e1c55aa`.
Its 120–300 s weights remain **zero** without evidence. Aggregate +897/+839 is insufficient
to estimate uncertainty or nonzero per-horizon weights.

13 worlds reserve **3,900 extra sim-s**, primary cycle **33,033** versus 29,133.
Two full attempts add 7,800; further attempts fail before dispatch if cumulative
long reservations exceed 8,000. Failed starts stay charged; actual seconds come
from executed arms, never inferred from reservations. Total cap stays 210,000;
prior 58,266 charged/0 actual is retained without refund.

## Verification / blockers

**340 CPU tests pass, two existing missing-tool skips** on gpuA: delayed payoff at t=200
with 10 s actor imagination reverses actor direction only with long assistance;
uncertainty/level cancellation, CRNs, real-WM integration, checkpoint round trip,
OFF update tensors, deterministic budgets and charged-zero-actual retry tests.
Cross-parent identity against d6a426c6 passes **348 exact tensors**, two updates
each on plain/cooperative/real-WM trainers; RNG and metrics also match. Ruff passes.
GPU timing remains pending. GPU measurement
has not run: required gpuB RSYNC_RC=0 receipt is unavailable within authorized
SSH scope. The 17:08 transfer-verification register entry lacks that per-log
receipt. `tools/longcritic/benchmark.py` enforces it and the 0.3/4-thread caps.
E: exhausted free space; the interrupted test edit was recovered and retested.
S-DELAYDIAG also reports unmatched fleets in the cited level ratios; these are
not calibration evidence. No SUMO, evaluation seeds, live activation or peer PIDs.

## Exact proposed registration amendment (NOT installed)

“At the next unstarted cycle boundary, install c4/longcritic and set policy.config
to configs/c4_policy14_longcritic.toml; retain frozen gate-panel/horizon arguments.
Add --longcritic-native-returns pointing to policy.initial_real_returns and
--longcritic-reliability pointing to the accepted-WM-SHA-bound measured TRAIN
panel; absent evidence gives zero assistance. Enable collect.long_group.enabled.
Reserve 33,033 primary sim-s and at most 66,066 including one full retry per
productive cycle, subordinate to 210,000 total and 8,000 additional long-group
sim-s/cycle. Preserve all previous charges, seeds, failures and gates. Record
source/config/registration hashes before dispatch. Activation requires completed
RSYNC-gated gpuA timing; S-LONGCRITIC-1 does not activate.”
