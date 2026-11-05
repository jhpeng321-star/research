VERDICT: PASS

## Blockers

None (0 blockers). No required P2 fixes identified.

## Test counts

- Independently executed in this fresh-session review: **118 passed, 0 failed, 0 skipped**, in **19.89s**; exit 0.
- Command, from this worktree:
  `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tests/unit/test_track2_reward.py tests/unit/test_merge_features.py tools/coopmwm/test_coop_layer_cpu.py`
- Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTEST_ADDOPTS=-p no:cacheprovider`.
- P2 Ruff check: exit 0 (`python -m ruff check --no-cache tools/coopmwm/track2_reward.py tests/unit/test_track2_reward.py`, using the same interpreter).
- `git diff --check c4/track-base`: passed.
- Counts cover the explicitly named, inspected CPU suites, not a repository-wide or live-simulator test run.

## Reviewed scope

- Independent verifier: astra; fresh-session re-verification on 2026-11-06.
- Branch: `c4/track2-reward`.
- Reviewed HEAD: `ea9760ecb559e3a066b8a998e19b97a43e61591b`.
- Base: `c4/track-base`, `13f184b5a7e12be04aebf246ef8e4dbb81555545`.
- Read the complete `E:/agent-work/_sol/tracks_spec.md`, `P2_READY.json`, and the changes against the base, including the added cooperation-layer and merge-feature dependencies and their tests/readiness receipts. Existing observation, policy, control-constant and advantage code was checked where needed to validate contracts.
- Applied the P2 MAIN 02:30 redesign: P2 supplies `track2_reward.py`; dense WM teaching and WM-counterfactual group credit are superseded for this launch. Production wiring belongs to P1/T2C. The later F/G tasks in the current shared specification target separate branches based on `c4/rerun-launch`; they do not redefine this P2 deliverable based on `c4/track-base`.

## Correctness and rule checks

1. **Real observable reward:** `tools/coopmwm/track2_reward.py:80-130` implements negative measured speed deficit and `r_i + w * mean(r_j)` over unique observable neighbours, including both CAVs and HDVs and excluding self. Empty neighbourhoods contribute zero. The result uses measured vehicle speeds, not WM predictions, hidden types, global delay, or source-queue outcomes. The explicitly documented endpoint-speed measurement is not represented as substep-integrated simulator delay.
2. **Locality and timestamp discipline:** `tools/coopmwm/track2_reward.py:98-115` validates current own packets, indexes only selected remote publishers, requires exact one-tick-old remote evidence, applies existing radio range/partner count, and resolves duplicates deterministically. `tools/coopmwm/merge_features.py:112-136` enforces atomic packet validity, publisher-local range and declared lane coverage. The receipt-row adapter clips direct observations to the ego's sensing range before reading neighbour speeds (`track2_reward.py:133-155`). No truth-data reads or simulator/WM calls are introduced.
3. **Outcome alignment and fail-closed behaviour:** `tools/coopmwm/track2_reward.py:183-214` uses explicit post-action observations or the following control-tick frame, refuses missing terminal/ego evidence, and rejects misaligned timestamps. `:228-230` preserves raw episodes, samples and base rewards through new shallow copies; repeated shaping is refused at `:185-186`.
4. **B/R fairness, A and default-off identity:** `tools/coopmwm/track2_reward.py:49-77` uses one shared weight/provenance and restricts any arm declaration to exactly B and R. A bypasses even B/R configuration validation. `:172-173` returns the exact original episodes object when inactive, without inspecting it. Tests exercise unreadable inputs, B/R-equal transformations and unaffected raw objects. The merged layer also has byte/RNG/noise-memory identity tests.
5. **Registration and horizon:** Weight has no invented production default and must be positive, finite and accompanied by provenance. Speed limit, sensing/radio ranges, partner cap and control interval match existing contracts. No planner/look-ahead horizon is changed. `tools/coopmwm/track2_reward.py:250-258` computes advantages over the complete shaped episode with existing UNIT/GAMMA/LAMBDA before filtering PPO-ineligible records; noneligible steps are not deleted from the return calculation.
6. **Actual layer-only learning tests:** The P2 suite checks analytical reward/sign expectations, temporal alignment, global-reward independence, locality, deduplication, malformed/missing evidence, fairness, identity and GAE eligibility. `tests/unit/test_track2_reward.py:294-319` instantiates the real Policy and CooperationLayer, performs an accepted update from the produced records, verifies that layer parameters change, and checks that every base state tensor remains identical with no gradients. The dependency suite independently checks joint KL against an exhaustive reference, rollback, optimizer isolation, V2V-base freezing, and exact off/zero-init action behaviour. These are substantive assertions, not readiness-receipt or mock-success checks.

## Integration handoffs

`P2_READY.json:103-115` accurately records work owned by launch registration/P1/T2C: supply the registered numerical weight and provenance identically for B/R; capture terminal and departing-ego post-action evidence and timestamped V2V packets; call the API only for cooperation-layer learning while keeping the superseded reward/WM paths off. Missing terminal evidence is explicitly refused by this module. This PASS is for the P2 deliverable and inspected dependencies, not evidence that the separate production integration or a live SUMO smoke has completed.

## Resume provenance

Git status/log and the existing readiness, review and acceptance receipts were read first. The previous review at `22d902d68d1c20628d265d5eae96150ca94a6791` recorded PASS with 118 passed, 0 failed, 0 skipped in 18.43s on 2026-11-05. Its implementation, tests, dependency merges and acceptance receipt were preserved. The current source diff and substantive test assertions were inspected again, and the requested CPU verification was run once after checking that no matching P2 pytest process was active. No implementation work, simulator jobs or dispatches were repeated.

## Review execution

CPU only; no SSH, remote jobs, SUMO, or GPU execution. No prohibited data paths or truth files were opened. The worktree was clean on arrival. Bytecode generation, pytest caching and external pytest plugin autoload were disabled for verification; Ruff used `--no-cache`. No source/test/configuration edits or commits were made; this existing review file is the sole modified deliverable.
