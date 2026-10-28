# S-CCT2

Worktree `E:/agent-work/c4-cct2`, branch `c4/cct2`, based on `c4/gated300`
`a5953fb9`. Owner authority: 2026-10-29 02:25 main registration (before any
W2/W4/W5/ERRSRC final verdict). Default OFF; zero SUMO; no GPU.

## Registered CCT-2 (02:25, followed exactly)

- sigma_h as CCT-1: 90=25.38, 120=41.22, 180=42.11, 300=43.91 veh-s. Paired
  dJ = arm - native (negative = improvement); real dJ = 0 pairs excluded from
  sign counts only.
- Confident GAIN set G_h = predicted dJ_h < -2*sigma_h, **pooled over the two
  held-out verdict panels (pairs300 + validation; never selection/calibration)**.
- PASS at h iff |G_h| >= 20 AND sign agreement on G_h >= 0.80.
- H_G = largest h in {120, 180, 300} that PASSES; **no contiguous-prefix
  requirement**.
- Informational: per-panel splits, world-cluster bootstrap CI (2000 draws, seed
  0), mean |real dJ| of wrong gain calls, harm-call stats.
- Use: with H_G, candidate A may receive gated long credit up to H_long = H_G
  ONLY for rows whose K8 median predicts improvement beyond 2*sigma_{H_long}
  AND q10 of improvement > 0 (gain-only mode); harm/below-margin rows get none
  (dense 90 s credit only). CCT-1's all-direction H_C keeps its meaning.

## Tool and authority file

`tools/gated300/cct2.py PAIRS300_JSON VALIDATION_JSON --out FILE` reads the
existing horizon-gate `rows[].dJ[h] = [predicted, real]` of both verdict
panels, refuses calibration/selection roles, mismatched WM hashes and
entrant/CRN-inconsistent panels, and writes an exclusive `gated300_cct2/1`
document whose filename must contain the full 64-hex WM sha256. The pooled
world bootstrap resamples merged world clusters (a world present in both
panels forms one cluster). |G_h| counts every confident-gain pair including
real-zero ones; agreement divides only over real-nonzero members.

## Gain-only long-credit mode

`src/wms/control/gated300.py` grows `[long_credit]` keys `mode` (default
`"cct1"`) and `cct2_file` (default empty). Everything else is unchanged:

```toml
[long_credit]
enabled = true
mode = "gain_only"
cct2_file = "/absolute/authority/CCT2_{wm_sha256}.json"
w_long = 1.0
score_training_horizon = true
```

Binding mirrors CCT-1: exact WM sha256, pooled pairs300+validation verdict
panels, `policy_authorized`, H_G recheck against the registered largest
passing horizon (n_gain >= 20, agreement in [.80, 1], matching sigma/threshold
and verdict), entrant/CRN and K8/q10 requirements. Missing `cct2_file`,
absent file or H_G = 0 leaves the run bitwise OFF (no rollout, no loss).
`gate(..., gain_only=True)` keeps ON only median improvement > 2*sigma with
q10 > 0: confident-harm and below-margin rows receive exactly zero long
credit; the CCT-1 two-sided mode and all existing defaults/tests are
unchanged. Matched scoring (`policy14.independent_score`) applies the same
direction-aware gate. Checkpoint payloads carry the new config keys and
resume re-installs the CCT-2 binding exactly as CCT-1's.

## W1 informational CCT-2 (NOT adopted; no authority consumed)

Read-only run on W1's verdict panels, WM `3f6b4129` (local evidence
`E:/agent-work/c4-overnight-w4/tools/overnight_w4/evidence/W1/LONG_{pairs300,
validation}.json`; one read-only gpuR `ls+sha256sum` transaction verified the
copies byte-identical to `/root/autodl-tmp/c4/overnight20261028/W1_PAIRED/LONG/`).
204 pooled arm pairs / 56 world clusters; horizons **90 / 120 / 180 / 300 s**:

- pooled |G_h| **10 / 10 / 26 / 28**; gain sign agreement **.90 / 1.00 /
  1.00 / 1.00**; verdicts **FAIL / FAIL / PASS / PASS** (90/120 fail only on
  |G_h| < 20); **H_G = 300**.
- wrong-gain-call mean |real dJ| **78.14 / 0 / 0 / 0** veh-s; gain CI95
  **[.50,1.00] / [1,1] / [1,1] / [1,1]**.
- harm calls **112 / 107 / 108 / 99**, harm agreement **.857 / .654 / .620 /
  .677**, wrong-harm mean |real dJ| **61.94 / 109.19 / 231.96 / 417.41**.
- per-panel |G_h| splits — pairs300 **0 / 1 / 8 / 10**, validation
  **10 / 9 / 18 / 18**; every nonempty split agrees 1.00 except validation 90
  at .90.

So W1's gains direction is perfectly signed where confident at 120-300 s while
CCT-1 (all-direction) stays H_C=0; W1 remains not adopted and this confers no
trainer authority. Full unrounded values: `tools/gated300/evidence/
w1_final_CCT2.json` with source SHAs in `w1_final_source.json`.

## CPU verification

Python 3.13.13 / torch 2.14.1+cpu (CUDA hidden, OMP/MKL 1, UTF-8), repo root
plus `src` on PYTHONPATH, `C4_ALLOWED_ROOTS=E:/Temp/opencode`, temporary
outputs confined to `E:/Temp/opencode/cct2_*`. Ruff clean on all changed files.

- **35/35 gated300 tests PASS** (`test_gated300.py` + `test_gated300_cct1.py`
  + `test_gated300_cct2.py`, 55.45 s): 7 new CCT-2 tool tests (sign convention
  negative = improvement, zero-real exclusion, n<20 FAIL, pooling with
  informational per-panel splits, no-contiguous-prefix H_G, cross-panel world
  clusters + bootstrap reproducibility, fail-closed roles/WM/entrant/NaN,
  read-only inputs, WM-hash-bound exclusive output) and 7 new gain-only tests
  (gain-only gate keeps only confident gains with harm/margin/bad-q10 OFF and
  zero gradient, bitwise-identical OFF updates for off/missing/H_G=0, active
  binding H_G=H_long=120 with artifact sha256 and nonzero long credit,
  refusals for wrong WM/panels/authorization/schema/tampered H_G/entrant/unknown
  mode, WM-change refusal).
- Regressions: looplong + antiexploit + policy14_integration + coopw
  **98 PASS / 2 SKIP** (31.86 s); LONGMEM **25 PASS / 2 CUDA-only SKIP**
  (156.61 s); loopresume + policy14_recovery + gate_panel **23 PASS**.

```text
python -B -m pytest -q -o addopts= -p no:cacheprovider --basetemp E:/Temp/opencode/cct2_new tests/unit/test_gated300.py tests/unit/test_gated300_cct1.py tests/unit/test_gated300_cct2.py
```

No GPU smoke, no SUMO, no shared process/lock/budget touched; one read-only
gpuR SSH transaction (ls + sha256sum) only.
