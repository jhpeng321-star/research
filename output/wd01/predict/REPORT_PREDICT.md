# REPORT_PREDICT.md — WD-01 part 2B, predict() contract v1 implementation

Task: implement the frozen `predict()` contract v1
(`output/wd01/PREDICT_CONTRACT_v1.md`, `contract_version = "wd01_predict/1"`)
as a typed Python module with validators, §5 native-reference resolution, a
Predictor protocol and a conformance kit. **Zero simulation** — no SUMO/TraCI
imports anywhere; pure numpy + stdlib; importable without torch.

Branch: `zcode/wd01-predict` (not pushed, not merged).
Python: `C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe` (3.10.20, numpy 2.2.6).

## Module map (`output/wd01/predict/`)

| module | contents |
| --- | --- |
| `__init__.py` | public surface; eager: contract/validate/native_ref/protocol; lazy (PEP 562): conformance + reference predictor, so `python -m predict.conformance` is warning-free |
| `contract.py` | constants: `CONTRACT_VERSION="wd01_predict/1"`, `PLAN_VERSION=1`, actions (`native`/`damping`/`pi-external-u`/`wall`), phases (`assemble`/`hold`/`release`), abort codes `0/1/2` (`ABORT_NONE/ABORT_SAFETY/ABORT_ASSEMBLY_TIMEOUT`), lane ids `0=main_left, 1=main_right, 2=accel`, entrances. Frozen dataclasses: `Arrival`, `ArrivalDraw`, `SampleMeta`, `StateSample`, `ControllerStateEntry`, `WallSpec`, `PlanCandidate`, `WallOutcome`, `HiddenTrajectorySet`, `Meta`, `Prediction`; strict `from_raw` converters (unknown keys rejected; `schedule` reserved — rejected in v1) |
| `validate.py` | `validate_plan(plan, controller_state)` — every §3.3 rule (see below); `validate_state_samples(state_samples)` — §3.1 (K>=1, shared `time_s`, typed meta, arrivals); `validate_prediction(pred, M, K, n_plans, trajectory_s=None)` — §4 shapes/dtypes incl. `native_*`, bool collision flags, `[M,K,T,4]` trajectories with `T=trajectory_s+1` (or `part_a_horizon_s+1`) and suffix whole-row NaN policy, walls dict with `int8` abort codes in {0,1,2} plus assembled/abort consistency, `native_unlocked_*` present iff `meta.native_unlocked_available`, full §4.6 meta incl. `contract_version`, per-member `weights_sha256`, `safety_window_s == part_a_horizon_s`, ids lengths, `wall_clock_s`. Violations raise `PlanViolation` / `StateSamplesViolation` / `PredictionViolation` (all `ValueError` subclasses); nothing returns a silent guess |
| `native_ref.py` | §5 `native_reference(controller_state, cav_ids) -> (native PlanCandidate, native_kept_walls)` — ASSEMBLE walls kept (locked members, `phase="assemble"`, `hold_s=math.inf` = "only a safety abort ends it"); HOLD and RELEASE walls dissolved at t0 (`phase="release"`, `release_at_s=0.0`); every other CAV `native` in `per_cav`. §4.5 `native_unlocked_reference(...)` — every wall dissolved, including walls still in assembly. Both emit v1-valid plans (pass `validate_plan`) |
| `protocol.py` | `Predictor` (runtime-checkable Protocol): `predict_many(state_samples, plans, controller_state, *, horizon_s=600.0, trajectory_s=None) -> list[Prediction]`, `predict(...) -> Prediction`; `PredictorMixin` supplies `predict() == predict_many(...)[0]` |
| `conformance.py` | `run_conformance(predictor)` kit: synthetic call (6 CAVs, walls W1 assemble / W2 hold / W3 release with locked members, 2 free CAVs, K>=3 samples, 4 plans: §5 native, §4.5 unlocked, keep-W1+damping, new wall W4 with empty `per_cav`). Checks: fixture validity, output order (single-plan **anchors** + swap test), `[M,K]` alignment across plans, shared native identical within a call, `validate_prediction` on every output, determinism on repeat call (`wall_clock_s` excluded — informational), `predict()` identity, `meta.native_kept_walls` == §5 resolution. Raises `ConformanceError` listing all failures; returns `ConformanceReport` on PASS. `python -m predict.conformance` runs it on the reference predictor |
| `reference_const_speed.py` | `ConstSpeedPredictor` — **NOT A MODEL, conformance reference only**. Deterministic constant-speed motion (exit → whole-row NaN tail), stub delay/safety fields, M=1, `part_a_horizon_s=60`, `weights_sha256` = sha256 of its own source. Rejects `trajectory_s > part_a_horizon_s` (a §2-permitted strictness) |

### §3.3 rules enforced by `validate_plan` (each with a dedicated test)

`plan_version == 1`; `per_cav` is the FULL map of every non-member CAV
(missing and extra keys both raise); wall members never in `per_cav`; actions
restricted to `native|damping|pi-external-u`; locked CAVs appear only as
members of their existing wall under the same `wall_id` (and therefore must
appear as members in every valid v1 plan); wall ids unique/non-empty;
membership sets disjoint, non-empty, duplicate-free, members known CAVs;
`speed_mps` finite > 0; phase in {assemble, hold, release}; exactly one of
`release_at_s`/`hold_s` (finite >= 0; `hold_s=math.inf` reserved for the
native indefinite hold); controller-state hygiene (§3.2): action enum,
`action=="wall"` ⟺ `wall_id` set ⟺ `wall_phase` set, `time_in_phase_s >= 0`,
`locked` bool, locked ⟹ `wall_id`.

## How implementations run the conformance kit

The kit is implementation-agnostic: `run_conformance(predictor)` takes any
object satisfying the `Predictor` protocol. The three implementations run the
same call with their own instance:

```powershell
# from the worktree root, any implementation:
cd D:\rd-wt\wd01b\output\wd01
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe -c "import sys; sys.path.insert(0, '.'); from predict import run_conformance; from my_impl import MyPredictor; print(run_conformance(MyPredictor()))"
```

Exact command for the shipped reference (no extra code needed):

```powershell
cd D:\rd-wt\wd01b\output\wd01
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe -m predict.conformance
```

Output (2026-09-26): `status=PASS, predictor=reference_const_speed/1,
n_plans=4, n_samples=3, M=1, K=3, horizon_s=600.0, trajectory_s=null`, checks:
`inputs_pass_validators, output_order_matches_plans, mk_alignment_across_plans,
native_shared_within_call, validators_pass, deterministic_on_repeat_call,
predict_matches_predict_many, native_kept_walls_match_reference`.

- **Planner (perfect simulator, PL)**: instantiate the SUMO-branching
  predictor (M=1, K=arrival draws, common random numbers) and call
  `run_conformance(planner_predictor)` in its CI; add
  `full_horizon_*` extras freely — gating uses the window fields either way.
- **Learned (world model, WD)**: wrap the Part-A rollout + Part-B estimator
  (M=ensemble, K=hidden×arrival) and call `run_conformance(learned_predictor)`
  before and after any weight change; `weights_sha256` must list one sha per
  member so the determinism check binds the weights.
- Both implementations should also add
  `pytest output/wd01/predict/tests -q` to their suite — it pins the contract
  behaviour the kit and validators share.

## Spec questions (stricter reading implemented; contract text unchanged)

1. **Indefinite hold (§5).** v1 `WallSpec` cannot say "stay in assemble until
   a safety abort ends it". We encode native kept walls as
   `phase="assemble", hold_s=math.inf`; `validate_plan` accepts `+inf` only in
   `hold_s`. Recommend v2 add an explicit `indefinite_hold` flag or `null`
   semantics.
2. **Dissolved walls stay in the plan (§5 × §3.3).** "HOLD dissolved → members
   hand back to native" is model semantics; as a *plan encoding* it cannot put
   locked members into `per_cav` (§3.3 forbids it), so dissolved walls are
   emitted as `phase="release", release_at_s=0.0` ("release now" = dissolve at
   t0) with members retained. `native_kept_walls` carries the keep/dissolve
   distinction the model acts on.
3. **Member order (§3.3).** "Ordered by lane index" is uncheckable:
   `controller_state` carries no lane ids. `native_reference` orders members by
   `sorted(cav_id)` (deterministic); planners must lane-order their own walls.
   Recommend v2 add per-member lane ids to `controller_state`.
4. **`phase="release"` walls still need exactly one of `release_at_s`/`hold_s`.**
   Stricter reading: the XOR applies to all phases; "release now" is
   `release_at_s=0.0`.
5. **Plan wall `speed_mps` must be finite > 0.** A stopped "wall" (0 m/s) is
   rejected as degenerate for a moving-wall contract.
6. **Trajectory NaN policy (§4.3).** "NaN from the exit step on" is enforced
   as: whole rows (all 4 columns), trailing suffix per (member, sample), no
   holes, step 0 always real. The contract does not say whether the `t_s`
   column goes NaN or whether re-entry is possible; we took the strict
   no-re-entry reading (a vehicle that left cannot return).
7. **`min_ttc_s` may be NaN** (no follower–leader pair exists); all other
   `[M,K]` floats must be finite. The contract is silent on NaN there.
8. **§4.1 identities enforced with tolerance.** `delay_vs_native_s ==
   plan_delay_s − native_delay_s` and `delay_part_a_s + delay_part_b_s ==
   plan_delay_s` via `allclose(rtol=1e-6, atol=1e-9)`; the contract defines
   the relations but no tolerance.
9. **`n_plans` in `validate_prediction`** is call context only (no
   per-Prediction field depends on it); kept so call sites pass the dimension
   explicitly.
10. **Arrival entrance labels** are not fixed by the contract; we ship
    constants `mainline`/`ramp` but validators only require non-empty strings.
    `coverage` is enforced to `[0,1]` (contract says float only).
11. **`trajectory_s > part_a_horizon_s`**: the contract says implementations
    *may* reject; the reference predictor rejects, and `validate_prediction`
    only checks the resulting `T` consistency either way.
12. **`wall_clock_s`** is excluded from conformance equality (informational
    per §4.6); everything else must be bit-identical across repeat calls.

## v1.1 update (WD-01 part 2D, 2026-09-26)

The authoritative contract text (with §8 "v1 clarifications" and §9 "v1.1") is
`D:/rd-wt/wd01/output/wd01/PREDICT_CONTRACT_v1.md`, sha256
`74a1de8edb52437babfda611336e46fb9653f7d1cbd43da878ac0295d5fe0280` (verified
before copying; the copy at `output/wd01/PREDICT_CONTRACT_v1.md` is byte
identical). §8 answers the twelve spec questions below — every stricter
reading from p2B was adopted verbatim (native indefinite hold = `hold_s=+inf`,
dissolved walls as release-now members, member order = cav_id in v1,
`release_at_s` offset from t0 / `hold_s` from assembly completion, finite > 0
wall speed, suffix whole-row NaN, NaN policy, §4.1 tolerances, entrance labels
`mainline`/`ramp`, coverage in [0,1], `wall_clock_s` excluded from
determinism, explicit `n_plans`). §8.7/§8.9 enforcement is now in the
validator (entrance labels restricted to `mainline`/`ramp`).

Implemented for §9 (v1.1, backward compatible; the 120 p2B tests still pass
unchanged):

- `contract.py`: `SUPPORTED_CONTRACT_VERSIONS = ("wd01_predict/1",
  "wd01_predict/1.1")`; `ControllerStateEntry.lane_id: int|None` in
  {0,1,2}; `Prediction` gains optional `cell_valid`, `native_cell_valid`,
  `has_closing_pair`, `native_has_closing_pair` (bool [M,K]; None on a 1.0
  output, REQUIRED in 1.1 — never defaulted True).
- `validate.py`: `validate_prediction` dispatches on `meta.contract_version`;
  1.1 outputs without the cell arrays are rejected; in a VALID cell
  `min_ttc_s` is NaN iff `has_closing_pair` is False (same for native), every
  other float is finite, and invalid cells may carry NaN anywhere; the §4.1
  identities are checked on cells valid on both sides; a 1.0 output carrying
  §9 fields is a version mismatch; `validate_plan` enforces ascending
  `lane_id` ordering of wall members when every member has a `lane_id`
  (check skipped when any is missing); `lane_id` domain (0, 1, 2) checked.
- `native_ref.py`: `native_reference` orders wall members by `lane_id` when
  all members carry one, else by `cav_id` (§8.3 fallback).
- `safety.py` (new): `safety_veto(pred) -> bool[M,K]` implementing the §9
  FAIL CLOSED consumer rule (invalid plan/native cell, NaN in any delay
  field, collision, wall `safety_abort`), documented as evaluated BEFORE
  `min_ttc_s`.
- `reference_const_speed.py`: `ConstSpeedPredictor(contract_version=...)`
  implements and reports either version; the 1.1 stub marks all cells valid
  with closing pairs (finite 30 s TTC), consistent with the NaN-iff rule.
- `conformance.py`: `run_conformance_both_versions(predictor_factory)` runs
  the full kit per version (the kit `__main__` now reports both); `_NATIVE_*
` fields extended; `broken_predictors()` provides the three §9 fixtures —
  NaN `min_ttc_s` with `has_closing_pair=True`, failure signalled only by NaN
  while `cell_valid=True`, and a 1.1 output missing `cell_valid` — all of
  which the kit REJECTS.

pytest (rl_driving env, repo root):

```
158 passed, 1 warning in 0.72s
```

`python -m predict.conformance` → `{"status": "PASS", "versions":
["wd01_predict/1", "wd01_predict/1.1"]}` (8 checks each).

New tests: `tests/test_v1_1.py` (§8.1–§8.13 one test per item; §9 lane-id
domain, wall-member lane ordering incl. partially-missing, native lane
ordering, required cell_valid / never-defaulted, NaN-iff rules plan+native,
invalid-cell NaN allowance, v1 must not carry §9 fields, version handling,
kit rejections, both-versions run) and `tests/test_safety.py` (§9 veto rules,
incl. NaN-min-ttc-with-no-pair NOT vetoing and pre-min_ttc ordering). Total
158 passed = 120 p2B + 38 new.

sha256 of files (this update):

```
13a61d85ea6a2b99e1cbe52084ffee16313eea1d9e8a21dae798175946b1334a  predict/__init__.py
5295e98d99ea0fba014a51f953ecc9e9f6249cfe9fef788c14780245814a90a0  predict/conformance.py
7794ed98b723bcf826030ffcd1dcc457b3e73648b9f561897b6bb34b3f702d9b  predict/contract.py
6e24b15ad0b20bbefd0b5c0e9ab713a9f38d1a601fd30c5b486f1cb617c62086  predict/native_ref.py
e94100f1cd3431959cb52f43ce869c2bef9c516e25528849205ba2973b7d96ab  predict/protocol.py
d8d907c76a85452a537f00e993e92c14a5abf1b3eab8712247d15c0ed34243f4  predict/reference_const_speed.py
59e9889f68a8e2ac8f49dcc9c043ea749525fe74e41a83dd2589ab0dd4fb8941  predict/safety.py
7f83ce0bf9d8ec6e81a57c3f325d7a2978c7ae2b744dc9a1bb5c84162e3f5e78  predict/validate.py
c8250a66c249d303ed56df00a7bad419ae384a15cfb5c2c4d60fa190fe90ad45  predict/tests/conftest.py
142a601eef22fbb8f562d09aacdc0dae3df667064e384d953c10bba5d038678d  predict/tests/test_conformance.py
9c3023c19e7b67c3ef8fa0cc53d75e66cc0b7d9b25e1b740c371e6deb16badaa  predict/tests/test_contract.py
cb77469d96173ee6e8ccbee86fae08db6f52e363ae7bed2b63471d6813ba82a3  predict/tests/test_native_ref.py
bfcd932bd8f709456fbc3806c812935d0d370148b3bb7d53e70ee7e0baf9607d  predict/tests/test_no_torch.py
e28e9ccb4d4cebe5acebc6cb7e36d133f26dacbfc95dfe194774d59bf2d1ffff  predict/tests/test_safety.py
916a8c61f70f5d582fca5f5389563c6a8a68582e4dcf460eb242a4c95c26817c  predict/tests/test_validate_plan.py
45a2aeeddf5883e28bcc18b47f497beb3a3a737296f82833d479b3cc7e2514cc  predict/tests/test_validate_prediction.py
9966f2ecef78a51d0a75e777d46e4ee9f543a77c3e082fe50d678b69a04c90a7  predict/tests/test_validate_state_samples.py
f7ca38dd8342d86de569c37ff6f455ebbca1ff7df5366fd1cb3a5fb3c7db5758  predict/tests/test_v1_1.py
```

Git log (branch zcode/wd01-predict, local only):

```
6c32dae wd01 part2b: REPORT_PREDICT.md (module map, conformance usage, spec questions, evidence)
3b531c2 wd01 part2b: predict() contract v1 typed module (zero simulation)
589be1a wd01: predict contract v1 + Parts A/B design doc (design-only, owner Q2=yes)
a0a0b24 wd01 part1: world-model identity + rollout drift (code, tests, results, report)
```

---

## Verification (2026-09-26)

```
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe -m pytest output/wd01/predict/tests -q -p no:cacheprovider
=> 120 passed, 1 warning in 0.60s
```
(the single warning is the repo-root pytest config's `cache_dir` notice, not a
test failure; `--co -q` collects 120 tests)

Test coverage by acceptance item: §3.3 — `test_validate_plan.py` (24 tests,
one+ per rule, positive and negative); §3.1 — `test_validate_state_samples.py`
(12); §4 — `test_validate_prediction.py` (26, incl. native_*/bool/int8/
NaN-policy/meta/unlocked); §5 — `test_native_ref.py` (12: ASSEMBLE kept, HOLD
dissolved, RELEASE dissolved, `native_kept_walls`, unlocked variant); kit —
`test_conformance.py` (9, incl. 6 broken predictors caught: reversed order,
nondeterminism, native drift, bad shape, broken `predict()`, wrong ids);
constants/immutability — `test_contract.py` (5); zero-simulation —
`test_no_torch.py` (fresh subprocess: import `predict`, assert torch/sumolib/
traci/ray never enter `sys.modules`).

## sha256 of committed files

```
42320f47135adf41fb0a238aca5c0eb45d0360226329a5f2ede317a8e74cdf8b  predict/__init__.py
418aa78da03a7e87ad166669c31cd7c6e1a8ae591141083f0092a773ca203b9d  predict/conformance.py
fbbf2bf6edd5d9a4986b2b515af249c3a3aa2895c06ae924330bcc391b5fcb93  predict/contract.py
3c2fb2d3faef81cfe6dfe6a827080e59720860c7f91e53e65db00649a8fcd02d  predict/native_ref.py
e94100f1cd3431959cb52f43ce869c2bef9c516e25528849205ba2973b7d96ab  predict/protocol.py
4b1138b9ec3ea73ef9e2e76fb2de8c54fd9c2e7bd41ca489a005e71b114fb8fc  predict/reference_const_speed.py
a0d69a77c05b227c3db266a63a57dfe374bda625efce11e96e9023ad51746a22  predict/validate.py
c8250a66c249d303ed56df00a7bad419ae384a15cfb5c2c4d60fa190fe90ad45  predict/tests/conftest.py
142a601eef22fbb8f562d09aacdc0dae3df667064e384d953c10bba5d038678d  predict/tests/test_conformance.py
9c3023c19e7b67c3ef8fa0cc53d75e66cc0b7d9b25e1b740c371e6deb16badaa  predict/tests/test_contract.py
cb77469d96173ee6e8ccbee86fae08db6f52e363ae7bed2b63471d6813ba82a3  predict/tests/test_native_ref.py
146868b09b92699e223717d9e760e9cbe8fedb9e6cd6ced36469346cb5d6981d  predict/tests/test_no_torch.py
916a8c61f70f5d582fca5f5389563c6a8a68582e4dcf460eb242a4c95c26817c  predict/tests/test_validate_plan.py
45a2aeeddf5883e28bcc18b47f497beb3a3a737296f82833d479b3cc7e2514cc  predict/tests/test_validate_prediction.py
9966f2ecef78a51d0a75e777d46e4ee9f543a77c3e082fe50d678b69a04c90a7  predict/tests/test_validate_state_samples.py
```

## Git log

```
3b531c2 wd01 part2b: predict() contract v1 typed module (zero simulation)
589be1a wd01: predict contract v1 + Parts A/B design doc (design-only, owner Q2=yes)
a0a0b24 wd01 part1: world-model identity + rollout drift (code, tests, results, report)
```

Plus the report commit itself (adds `output/wd01/predict/REPORT_PREDICT.md`);
branch `zcode/wd01-predict`, local only — no push, no merge.

## Notes for the next agent

- Start from `predict/__init__.py`; the doctest-ish snippet at the top shows
  the intended usage of the kit.
- `predict.conformance.build_*` fixtures are importable for your own
  implementation tests — reuse them so fixtures stay comparable.
- The `n_plans` argument and the anchor-based order check are the two places
  where v1 text is thin; see spec questions 3 and 9 before changing them.


---

## v2 update (WD-01 part 2F, 2026-09-26)

The authoritative v2 text is `output/wd01/PREDICT_CONTRACT_v2.md`, sha256
`db475371685b3160da811c9afb9ca0322a6f70aec6d29ab53c3bb0a0aac90775` (verified
before implementation and again after: the contract texts were NOT modified).
v2 (`wd01_predict/2`) is a DELTA on v1/v1.1 per the preamble; the frozen
v1/v1.1 behaviour is unchanged — all 158 pre-v2 tests pass (three of them
amended only where they pinned the version surface itself, listed below).
Zero simulation: the package still imports without torch/SUMO/TraCI (pinned
by `tests/test_no_torch.py`).

Implemented (all inside `output/wd01/predict/`):

- `contract.py`: `CONTRACT_VERSION_V2 = "wd01_predict/2"`;
  `SUPPORTED_CONTRACT_VERSIONS = (v1, v1.1, v2)`; `DEFAULT_COMMIT_S = 30.0`,
  `CONTINUATIONS = ("planner", "native")`. `Arrival.arrival_id` optional
  (§8.14b). `Meta` gains the §4 additions (`commit_s`, `continuation`,
  `continuation_policy_id` — None exactly when continuation == "native",
  `value_model_id`, `value_model_sha256` per member,
  `future_min_ttc_validated`, `future_min_ttc_validation_id` — non-None
  exactly when validated); all default None/False and are REQUIRED on v2
  outputs. `WallOutcome` gains the §3 future-wall heads
  (`p_assembly_completes`, `expected_assembly_time_s`, `p_assembly_abort`),
  REQUIRED together on v2, absent on v1/v1.1. `Prediction` gains the §3
  future-safety outputs — `future_cell_valid` / `native_future_cell_valid`,
  `future_has_closing_pair` / `native_future_has_closing_pair` (REQUIRED,
  never defaulted) and the [M,K] floats `future_ttc_lt3_exposure_s`,
  `future_hard_brake_s`, `p_future_collision` (in [0,1]),
  `future_min_ttc_q05_s` + all four native_ versions.
- `validate.py`: three-way version dispatch. v2 rules: §4 meta (commit_s
  finite > 0 and <= part_a_horizon_s; continuation enum; policy-id iff-rule;
  value-model identity per member; validated iff id);
  `safety_window_s == commit_s` on v2 (`== part_a_horizon_s` on v1/v1.1);
  the four future validity/pair books REQUIRED; in a valid future cell every
  future float is finite except `future_min_ttc_q05_s`, which is NaN iff
  `future_has_closing_pair` is False (same for native_); probabilities finite
  in [0,1] in valid future cells; wall future heads checked against
  `future_cell_valid` with `expected_assembly_time_s` NaN only where
  `p_assembly_completes == 0` (one direction: at p == 0 a finite time is
  still legitimate — the contract text is "NaN only where"); §4.1 identities
  now need window-valid AND future-valid on both sides; v2 outputs must
  carry the §9 window books too; v1/v1.1 outputs must carry NONE of the v2
  arrays/meta (version mismatch). Trajectory default for v2:
  `trajectory_s=None` means `commit_s` (v1/v1.1 unchanged:
  `part_a_horizon_s`). §8.14: `arrival_id` optional non-empty str; arrival
  rows with `t_s < t0` are accepted by construction (`t_s` only has to be a
  finite time >= 0). §8.15: the StateSample builder (`SampleMeta.from_raw`,
  also reached through `StateSample.from_raw`) maps hybrid-state
  `merge_observed=None` to False (fail closed); hand-built non-bool metas
  are still rejected by the validator.
- `safety.py`: `safety_veto_v2(pred, tol_future_collision=0.0, p_max=0.01)`
  implementing §5 rules 1-5 over all MxK cells: (1) fail closed on any of
  the four validity books False / a NaN delay / a window collision /
  wall `safety_abort`; (2) absolute window gate `min_ttc_s >= 1.5` within Δ
  wherever a closing pair exists (NaN with a pair vetoes — fail closed);
  (3) the owner's future-collision vetoes — `p_future_collision > p_max` in
  ANY cell vetoes the whole call, plus the paired per-cell veto against
  `native_p_future_collision + tol`; (4) relative exposure (window + future,
  TTC<3 and hard braking) with the planner's mean and >= 90%-of-cells rules
  at tolerance 0 — a breach vetoes the whole call; (5) G-1: the 1.5 s floor
  extends to `future_min_ttc_q05_s` only when
  `meta.future_min_ttc_validated` is True, and only in cells with a closing
  pair. `live_mode_allowed(meta) -> bool`: False unless
  `future_min_ttc_validated` is True with a non-empty
  `future_min_ttc_validation_id` (§5 refusal rule). A v2 output missing any
  REQUIRED future field raises ValueError — a consumer never defaults it.
- `protocol.py`: `PredictorV2` (runtime-checkable) with the §2 signature
  `predict_many(..., horizon_s=600.0, commit_s=30.0, continuation="planner",
  trajectory_s=None)`; `PredictorV2Mixin` (predict == predict_many[0]).
- `reference_const_speed.py`: `ConstSpeedPredictorV2` (NOT A MODEL,
  conformance reference only; model_id `reference_const_speed/2`) — emits a
  full v2 output: part_a = commit-window share, part_b = continuation value
  (split sums to the total); all future fields all-clear with a conservative
  closing-pair encoding (pair present, finite 30 s q05, p_collision 0);
  future-wall heads always complete at the phase's assembly time; meta
  carries commit_s/continuation/policy-id (None under "native")/value-model
  identity, `future_min_ttc_validated=False` (the stub's future TTC head is
  NOT validated, so `live_mode_allowed` is False and the G-1 floor is off —
  the contract's starting regime). Per-call `commit_s`/`continuation`
  overrides; rejects `commit_s > part_a_horizon_s` and
  `trajectory_s > part_a_horizon_s`. The v1 `ConstSpeedPredictor` refuses to
  emit v2 outputs (NotImplementedError) — its v1/v1.1 behaviour unchanged.
- `conformance.py`: kit core shared by both generations; `run_conformance`
  keeps the v1 API and its 8 checks; `run_conformance_v2(...)` runs them on
  the commit-window API plus two v2 checks (`v2_future_outputs_present`,
  `v2_meta_matches_call`); `run_conformance_both_versions` now pins the
  FROZEN v1 family; `broken_predictors_v2()` ships the six §3/§5 fixtures —
  missing `future_cell_valid`; NaN `future_min_ttc_q05_s` with
  `future_has_closing_pair=True`; NaN `delay_part_b_s` while
  `future_cell_valid=True`; `continuation_policy_id` set with
  `continuation="native"`; `future_min_ttc_validated=True` without id;
  wall `expected_assembly_time_s` NaN with `p_assembly_completes > 0` —
  every one of which the kit REJECTS (each surfaces under
  `validators_pass`/`v2_meta_matches_call`). The kit `__main__` runs v1 +
  v1.1 + v2 (both continuations) and reports all PASS.
- `__init__.py`: v2 surface exported (constants, `PredictorV2`,
  `ConstSpeedPredictorV2`, `run_conformance_v2`, `broken_predictors_v2`,
  `safety_veto_v2`, `live_mode_allowed`).

Three existing tests were amended, and only because they pinned the version
surface itself (the behaviour they protect is preserved): `test_v1_1.py::
test_s9_versions_accepted_and_reported` (now asserts the three-version
tuple and probes `"wd01_predict/9"` as the unknown version);
`test_v1_1.py::test_s9_kit_exercises_both_versions` (asserts the frozen v1
family `wd01_predict/1` + `wd01_predict/1.1` for
`run_conformance_both_versions`); `test_validate_prediction.py::
test_contract_version_is_checked` (probe literal moved off the real
namespace).

pytest (rl_driving env, repo root):

```
C:/Users/jpeng/Miniconda3/envs/rl_driving/python.exe -m pytest output/wd01/predict/tests -q -p no:cacheprovider
=> 246 passed, 1 warning in 3.07s
```

(158 pre-v2 tests + 88 new; the single warning is the repo-root pytest
config's `cache_dir` notice, not a failure.) New tests:
`tests/test_v2.py` (§1/§2 constants and commit-window API; §3 future outputs
required-never-defaulted, NaN-iff rules plan+native, invalid-future-cell NaN
allowance, composition of validity over the delay identities, probability
bounds, future-wall heads incl. the NaN-only-at-p==0 rule; §4 meta rules
positive/negative incl. the validated-iff-id rule and v1-must-not-carry;
§8.14 arrivals before t0 / optional arrival_id; §8.15 bool enforcement and
the None->False builder mapping; v2 kit passes both continuations and
rejects all six broken predictors) and `tests/test_safety_v2.py` (§5 rules
1-5 positive/negative incl. owner absolute/paired vetoes with tolerance,
mean and >= 90%-of-cells rules, G-1 on/off with validation,
live_mode_allowed matrix, v2-only type guard).

`python -m predict.conformance` returns `{"status": "PASS", ...}` with
`wd01_predict/1` and `/1.1` at 8 checks each (predictor
`reference_const_speed/1`) and `v2: {planner, native}` at 10 checks each
(predictor `reference_const_speed/2`, M=1, K=3, n_plans=4).

sha256 of the predict module files (this update; REPORT_PREDICT.md itself
excluded, as in p2D):

```
a9ebe5fb0503322c8c2f8fc97e99e783c71ee8026c61042b46da52277ea60251  predict/__init__.py
c185c8441a8ba9d3b6e89a05533a3608fe6697a04f0ed6c0a92307d117f0327f  predict/conformance.py
0ecf174daccd24f95d4f79e0c222c1237e2b7a3c06dac2ba87e5637db94370b2  predict/contract.py
6e24b15ad0b20bbefd0b5c0e9ab713a9f38d1a601fd30c5b486f1cb617c62086  predict/native_ref.py
ae7d710efe57ddc55cdfa73a9cbc31f80800176c9a41ad372c8c46e7e601b173  predict/protocol.py
6c36d071749930b8e96ea1414aa46122c9ce809a623871bf236369d2d6fad994  predict/reference_const_speed.py
937da8d8737e6cd97f4136fddd3dfe560108d263a7df99a73433f7375b3efe29  predict/safety.py
99e0f0aac91450905531f63cced2099f3d4c836832a64af80636024ffdafd664  predict/validate.py
92c802870030515c555ac6c440e0d73616635753989d5cd51cd4fe96a6afb0d4  predict/tests/conftest.py
ca19976d8271dd22714d65f3fe6df9d3a96c62bdc348009fdd65363b8c789fca  predict/tests/test_conformance.py
a2b93f25950780cd027143113c11a398c4f753c6036dd06b5bc6d78d2c003c84  predict/tests/test_contract.py
5135dd60fa6f46340a9b236c864f358f2906934ddf11ca1af69392fdd2286d8a  predict/tests/test_native_ref.py
394834f7414172dbcd1a9263cab79ec753a811733637f5a33fb21206fb3cf575  predict/tests/test_no_torch.py
c1e9373c5291c0778306f5c402abd35a18a87f736bed61c3d5b1e214b4e45231  predict/tests/test_safety.py
21f70b3b70c6e78183f19010f1261dc2dacb71ecb360d3e97779491c24c24d6f  predict/tests/test_safety_v2.py
940d1649522cf46b29a244cf65ac967c4634ba55f737949cc008b6e410915056  predict/tests/test_v1_1.py
92ec6637c91acb71d4684880d7826f1ff92b54a5a37bf3fca2e8e7864b4d8506  predict/tests/test_v2.py
227875d14fd8e0103a2669dfce76ef7b2da82b4efcb461d6f82117b484865411  predict/tests/test_validate_plan.py
ace77e224675dbe9e1d44d0c7b0650d40b622e102c2e6c90f11c0e28cc20cb65  predict/tests/test_validate_prediction.py
e84d755304453752f82236a65727d5f5f36dbe250984de06d539b988dff246b5  predict/tests/test_validate_state_samples.py
```

Git log (branch `zcode/wd01-predv2`, local only — no push, no merge):

```
f1bd6a3 wd01 predict v2: wd01_predict/2 delta (contract sha db4753.. verified,
        texts untouched): commit_s/continuation API + PredictorV2; delay
        part_a=commit window, part_b=continuation value; future_* + native_
        safety with REQUIRED future_cell_valid/has_closing_pair books; future
        wall heads; §4 meta additions; safety_veto_v2 §5 rules 1-5 incl.
        owner absolute+paired future-collision vetoes and G-1,
        live_mode_allowed; §8.14/§8.15 for all versions; conformance kit v2
        both continuations + 6 broken predictors rejected; 158 old + 88 new
        tests; zero simulation, holdout NOT accessed
a8ffb67 wd01 part3k: SUMO-like merge dynamics ... (pre-existing head)
c618aac wd01 part3j: numeric delay under-prediction decomposition ... (history)
```

Plus this report commit itself (adds the v2 update section above).

## Veto alignment with PL-01 (2026-09-27)

`safety_veto_v2` in `output/wd01/predict/safety.py` was realigned EXACTLY to
the planner's (PL-01) reading of the §5 consumer rules. No contract text and
no shared record was modified; no simulation, no data. Signature is now

```
safety_veto_v2(pred, tol_future_collision=0.0, p_max=0.01,
               rel_safety_tol_s=0.0, rel_safety_quantile=0.90,
               native_collision_vetoes=True,
               window_must_not_increase=False) -> bool[M, K]
```

Changed rules (per clause):

1. Rule 1 (fail closed) — unchanged clauses: the four validity books False,
   a NaN delay field, the PLAN window collision, wall `safety_abort` in the
   cell. CHANGED: the NATIVE window-collision veto is kept while the owner
   decision is pending but is now a keyword flag
   (`native_collision_vetoes=True` default; the planner may pass False).
2. Rule 2 (absolute window gate) — plan side unchanged: `collision` False
   and `min_ttc_s >= 1.5` wherever `has_closing_pair`; a plan NaN TTC with a
   pair vetoes. CHANGED (native side): the 1.5 s floor NO LONGER applies to
   the native window — `native_min_ttc_s < 1.5` does NOT veto; a native NaN
   min TTC with `native_has_closing_pair` DOES veto.
3. Rule 3 (future collision) — unchanged in behaviour: any plan
   `p_future_collision > p_max` (default 0.01) vetoes the whole call; the
   paired cell is vetoed where plan p > native p + `tol_future_collision`
   (default 0); a NaN probability on either side vetoes.
4. Rule 4 (relative safety) — REWRITTEN. Per metric m in
   {`ttc_lt3_exposure_s`, `hard_brake_s`}, per cell, the delta is now
   SUMMED: `(plan_m - native_m) + (plan_future_m - native_future_m)` (was:
   window and future arrays gated as four independent rule sets). The call
   is ok for m iff `mean(delta) <= rel_safety_tol_s` AND
   `mean(delta <= rel_safety_tol_s) >= rel_safety_quantile` (new keywords,
   defaults 0.0 / 0.90). ANY non-finite delta anywhere fails the whole call
   (was: non-finite cells were dropped from the usable set). A breach vetoes
   the whole call. OPTIONAL (owner decision pending, same day follow-up):
   with `window_must_not_increase=True` the WINDOW-only delta
   `(plan_m - native_m)` must ADDITIONALLY pass the same mean + quantile
   gates per metric (non-finite window delta -> veto all); the default
   False keeps the summed-only behaviour.
5. Rule 5 (G-1) — plan side unchanged: if `meta.future_min_ttc_validated`,
   veto cells with `future_has_closing_pair` and
   `future_min_ttc_q05_s < 1.5` (NaN with a pair vetoes). ADDED (native):
   a native `future_min_ttc_q05_s` NaN with `native_future_has_closing_pair`
   vetoes, whether or not the head is validated.

Tests (`tests/test_safety_v2.py`, one per rule clause, 13 added — incl.
native min TTC < 1.5 NOT vetoing, the summed rule-4 delta, the quantile
rule, `native_collision_vetoes` both ways, and `window_must_not_increase`
both ways — a summed-delta pass / window-delta fail case and a
window-quantile-gate fail case). Amended tests, each pinning
the old stricter behaviour, with reasons inline:

- `test_v2_window_ttc_floor_with_closing_pair` — pinned the old native
  1.5 s floor (native 0.4 s TTC vetoed); amended: the native window TTC is
  no longer floor-checked (PL-01 rule 2).
- `test_v2_relative_future_exposure_mean_rule_vetoes_call` — reworded to
  the PL-01 summed-delta mean gate (was four independent array means).
- `test_v2_relative_future_exposure_fraction_rule_vetoes_call` — the
  <= fraction gate now applies to the summed delta (was the separate
  future array).
- `test_v2_relative_window_exposure_rules_fire_too` — the window part
  enters the summed delta (was its own independent rule set).

Verification: `python -m pytest -q` — 259 passed (246 pre-alignment + 13
new; the four amended tests are among the 246).

sha256 of the realigned files:

```
575222a9409cec18e64a9e731f8c65067c6c2a3b68969722d1eb745688c68b19  predict/safety.py
9ec6ac2a50608502efeaa8a8a83ab29454ec974da7fb50bfa1b105ccae2cbf2c  predict/tests/test_safety_v2.py
```

## Owner decisions 2026-09-27 applied to the reference defaults

The owner decisions recorded 2026-09-27 (`PREDICT_CONTRACT_v2.md` §7, commit
6d2a883) are now the DEFAULTS of `safety_veto_v2` in
`output/wd01/predict/safety.py`. The behaviour of each flag is unchanged —
only the reference defaults flipped:

```
safety_veto_v2(pred, tol_future_collision=0.0, p_max=0.01,
               rel_safety_tol_s=0.0, rel_safety_quantile=0.90,
               native_collision_vetoes=False,
               window_must_not_increase=True) -> bool[M, K]
```

- `native_collision_vetoes=False`: a NATIVE window collision no longer
  vetoes a plan cell by default — it is logged (§7, first bullet); the
  planner can re-enable the veto with `native_collision_vetoes=True`.
- `window_must_not_increase=True`: rule 4 now additionally requires the
  commit WINDOW alone to be non-increasing for TTC<3 exposure and hard
  braking (the same mean + >= 90%-of-cells gates on the window-only delta),
  on top of the summed window + future rule (§7, second bullet); `False`
  restores the pre-decision summed-only behaviour.

No contract text and no shared record was modified; no simulation, no data.
Tests (`tests/test_safety_v2.py`): four tests were amended — the rule-1
native-collision clause, the native-collision flag-both-ways test, and both
rule-4 window-clause tests (each pinning the old default, one-line reason
inline); one new test asserts both new signature defaults plus their
behaviour at the defaults.

Verification: `python -m pytest -q output/wd01/predict/tests` — 260 passed
(259 pre-flip + 1 new; the four amended tests are among the 259).

sha256 of the changed files:

```
d1018801b09f20aa4c0a99653c0beac99c3601afcc6f39e5a79ac827939a408f  predict/safety.py
42ea959d4831c32232e2a79b5972311c4e80d646cf0e0fa3b924768d8388cc90  predict/tests/test_safety_v2.py
```
