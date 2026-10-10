# REVIEW — WM-N1 Idea A piece P1 CTRL, SECOND PASS (critical)

Reviewer: execution agent (GLM), 2026-10-01.  Scope: critical second pass of
the P1 CTRL piece at `zcode/revA1g` = `zcode/wmn1p1` `e2348b4`
(`ctrl/**`, `PARITY.md`) against:

- `output/wd01/wm_n1/INTERFACES_V2.md` (frozen) + amendment 3
  `REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md` (`git show zcode/wd01:...`),
- pre-flight gate G2 `PREFLIGHT_GATES_IDEA_A.md`
  (`git show zcode/wd01:output/wd01/wm_n1/PREFLIGHT_GATES_IDEA_A.md`),
- the FROZEN wall law v2.2: `REGISTRATION_WALL_V22.md` +
  `DESIGN_V22_MERGE_RELEASE.md` (LF e7fd3536) and the patched copies
  `output/wd01/walls_design/merge_release/copies/post` on `zcode/wd01`
  (`wall_request.py`, `wall_group.py`, `wall_driver.py`, `common.py`).

Method: zero SUMO, no training, no fitting; only TRAIN physics rows /
command schedules were ever read (by the archived G2 replay; the raw
`runs/campaign_*_chunk4` trees are deleted under MIG-04, so the archived
`ctrl/results/parity_g2.json` is the frozen evidence and was NOT
recomputed).  No HOLDOUT/CAL/chunk-5, `evaluate_summary`,
`dispatcher_rows` or insertion data touched.  `WD_FORBID_D=1` clean.

Closest priors (ledger `closest`): CV-32 / OE-01 / OE-02 (wall controller
verification + zero-simulation analyses); none is a P1 code review — the
direct priors are the WM-N1 piece registrations (`INTERFACES_V2.md`
mismatch 6, amendment 3 A3.1(a)) and the partA_ctrl E1-E5/E4 mirror
validation; WM-59 + `REGISTRATION_WALL_V22.md` register the v2.2 law.
No new simulation is proposed or run here; this is a zero-simulation
review + unit-tested implementation fix.

Legend: **PASS** / **PASS\*** (passes with a recorded, bounded caveat) /
**FAIL — fixed** (fixed in this review) / **FAIL — open**.

## Focus item 1 — BOTH wall laws selectable via `law_version`; v2.2 exactly as frozen

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| 1.1 | Both laws selectable: `wall-v2.1-closure` (J4 training rollouts + G2 on J4) and `wall-v2.2` | **PASS** | `config.py:24-25` defines both tags; `config.py:45` defaults `wall_law="wall-v2.2"` (INTERFACES_V2.md:167); `config.py:53-61` refuses anything else; `config.py:64-70` exposes the `law_version` alias (added this review). `mover.py:371-382` selects the v2.1 mirror path vs the registered-law path by the version, and `tests/test_ctrl_interface.py::test_config_law_version_selects_both_wall_laws` pins it. |
| 1.2 | v2.1 path = the exact J4 mirrors, never released early | **PASS** | `mover.py:371-375` delegates to `partA_ctrl.wall_gap_to_front` / `wall_pace` (`emulators.py:441-479`, `WALL_LAW_VERSION="wall-v2.1-closure"` at `emulators.py:437`); the v2.1 branch leaves `fresh` all-False (`mover.py:370`) so `released` is all-False (`mover.py:403`) and the latch is never written; `test_mover_wall_v21_matches_j4_mirrors` pins bit-equality with the mirrors, and the G2 wall columns reached 100 % command parity on both J4 windows. Note: the mover feeds the mirrors the cumulative `s_road`, which for 2-abreast members on one edge is offset-invariant for `gap_to_front` (differences unchanged) — confirmed by the exact J4 command parity. |
| 1.3 | v2.2 release: LEAD member's edge-aware `s_road` reaches 2644.2 (X=0), HOLD only, ONCE, `merge_release` logged | **FAIL — fixed** | Was a stateless per-tick `release_mask` (`wall_hook.py:173`) with NO once-latch and NO event surface — "fires exactly once" and "logged merge_release" (frozen `wall_group.py:246-259`) were unimplementable by a consumer. Fixed: `CtrlState.merge_released` latch (`mover.py:97-111`), `CavStepOutput.merge_release_events` (`mover.py:115-132`), frozen event fields in `wall_v22.py:211-224` (`release_kind="merge_release"`, `release_point_s=2644.2`, `merge_start_s=2644.2`, `handback="native"`), per-(row, wall-id) once semantics in `mover.py:385-402` + `wall_v22.MergeReleaseSession` (`wall_v22.py:227-283`); pinned by `test_mover_wall_v22_release_fires_once_and_logs_merge_release`, `test_mover_wall_v22_release_is_batched_per_row_and_wall`, `test_merge_release_session_fires_once_per_wall`. |
| 1.4 | Release threshold with the frozen geometry tolerance | **FAIL — fixed** | `release_mask` compared `>= 2644.2` without the frozen `- 1e-9` (post copy `wall_group.py:239`). Fixed at `wall_hook.py:47,197` (`MERGE_RELEASE_TOL_M = 1e-9`); pinned by `test_mover_wall_v22_release_honours_frozen_tolerance` (−5e-10 m releases, −2e-9 m does not). |
| 1.5 | Fixed-arm hold cap `min(H, max(0, 2644.2 - s_lead)/v_wall)`, logged | **FAIL — fixed (implemented)** | Absent from the hook. Implemented as the frozen formulas: `wall_v22.hold_cap_s` (`wall_v22.py:163-176`, = post `wall_request.py` incl. the 6-decimal rounding) and the once-per-wall `hold_cap` log row `wall_v22.hold_cap_record` (`wall_v22.py:179-208`, exactly the `wall_driver.hold_cap_row` field set: `requested_hold_s`, `effective_hold_s`, `capped_at_merge`, `cap_point_s=2644.2`, `a_priori_effective_hold_s`, `lead_s_road_at_release_m`, `wall_speed_mps`, `source="cav_report_frames"`); drawn-vs-effective kept distinct. Pinned by `test_wall_v22_hold_cap_formula`, `test_wall_v22_hold_cap_record_fields`. |
| 1.6 | Proposer `D_min = 70 * v_wall` with `dmin_unreachable` | **FAIL — fixed (implemented)** | Absent from the hook. Implemented: `wall_v22.min_proposal_distance_m` (`wall_v22.py:80-91`, `X + v*(t_assemble + H_min) = 70 v` with `ASSEMBLY_TIME_ASSUMED_S=10`, `MIN_MEANINGFUL_HOLD_S=60`), the refusal record `wall_v22.dmin_refusal` (`wall_v22.py:94-114`, reason `dmin_unreachable` with lead distance / D_min / wall speed, 1e-9 gate tolerance), and the full pair gate `wall_v22.check_pair` (`wall_v22.py:117-152`). Pinned by `test_wall_v22_dmin_is_70_times_wall_speed` (15.3 → 1071.0 m, 17.0 → 1190.0 m), `test_wall_v22_dmin_refusal_record`, `test_wall_v22_pair_gate_2_abreast`. |
| 1.7 | 2-abreast shape (one CAV per mainline lane) | **PASS** (gate) | The pacing/pairing shape lives with the wall session (frozen `wall_group` member/lane coverage), but the proposer-side gate is now in P1: `check_pair` requires two members in DISTINCT lanes and span ≤ 75 m (`wall_v22.py:130-150`); the mover paces members per lane via the mirrors exactly as v2.1 does (G2 wall windows, 2 members/lane, 100 % command parity). |
| 1.8 | v2.2 constants equal the frozen registration (never tuned) | **PASS** | `wall_v22.py:45-68` + `wall_hook.py:44-58` reproduce `REGISTRATION_WALL_V22.md` §2 exactly (2644.2 / 2719.6 / 3197.5, X=0, draws {60,120,240}, span cap 75); `test_wall_v22_frozen_constants` pins them against the registered values. |

**Item 1 verdict: PASS** (after the fixes above; the two "implemented"
items were the brief's "if P1 only has a hook, implement v2.2 in ctrl/
using the frozen patched copies as the reference, with tests").

## Focus item 2 — G2 parity: per-strategy tolerance, verdict, drift explanation

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| 2.1 | ≥ 2 TRAIN windows per strategy (native, damping, PI, wall v2.1) | **PASS** | 9 windows = 2 chunk-4 TRAIN worlds × 4 strategies (+ `native_base` reference); worlds TRAIN by the registered WORLD_PLAN (`PARITY.md` §Windows; `parity.py:90-100`). Split discipline: only physics rows + command schedules read. |
| 2.2 | Clear tolerance PER strategy and explicit PASS/FAIL | **FAIL — fixed** | `PARITY.md` stated global thresholds without a per-strategy verdict table. Fixed: `PARITY.md` §"Stated tolerance (gate G2) — PER STRATEGY, with verdict" now tables per-strategy achieved values against the identical thresholds (the residual is the execution chain, not the controller) and verdicts each strategy: native / native_base / damping / pi / wall v2.1 all **PASS**. |
| 2.3 | Commands match exactly (command error ~0) | **PASS** | 51821/51821 logged command rows (100 %), max |diff| 7.2e-7 m/s = float32 rounding of the logged values; per-strategy worst: damping 7.2e-7, all others 0.0 (`PARITY.md` table; `parity_g2.json` `command_check`). Frame reconstruction exact (8997 `pi_source_evidence` frames, max diff 0.0, 0 presence flips). |
| 2.4 | Trajectory drift explained (mirror vs SUMO speedMode / lane changes), defensible criterion, no moved goalpost | **PASS\*** | `PARITY.md` §"Structural mismatch attribution": the drift is the measured `kin_idm` execution rule vs SUMO's internal mechanics (chunk-1 measured on 1.10e6 TRAIN half-steps: p50 1.7e-3, p95 0.336, max 6.9 m/s; chunk-4 CAVs run speedMode 27 — bit 2 off allows emergency decel below −b), SUMO lane changes (not mirrored; 1159–1594 half-steps/window), junction transits / the accel→mainline station discontinuity (re-anchors counted, 880 across 9 windows) and `slot_unknown_geometry` (352). The acceptance criterion is now anchored: C1 exact commands (primary) + C2 exact frames + C3 per-tick speed/position bounds whose short-horizon form is derived a priori from the measured rule residual (10 s ⇒ ≈ 3.4 m p95 bound); the observed p50 0.145–0.589 m / RMS ≤ 10.36 m sit inside that envelope. Not a goalpost move: the criterion is stated before the numbers and the command contract is exact. |
| 2.5 | P1 supports BOTH wall v2.1 (J4 parity) and v2.2 (G2 gate text) | **PASS** | v2.1 = the two recorded-run wall columns (G2); v2.2 = item 1 implementation + unit tests (no recorded J4 v2.2 run exists yet — correct per gate text "P1 must support", not "must have replayed"). |

**Item 2 verdict: PASS** (after the PARITY.md fix; archived evidence was
not recomputed — the raw chunk-4 runs are deleted, MIG-04).

## Focus item 3 — never learned; no labels into the HDV path; batched; deterministic; CUDA-ready; torch 2.5.1

| # | Item | Verdict | Evidence |
| --- | --- | --- | --- |
| 3.1 | Controllers exact, never learned, no fitting | **PASS** | Every strategy path is a call into `partA_ctrl.emulators` (read-only via `ctrl/_bridge.py:40-43`) or the frozen v2.2 transcription; the ONNX payload is sha-pinned (`config.py:15-16`, re-checked at `emulators.py:236-240`) and the mover refuses to approximate it (`_bridge.py:46-52`). No training/fitting anywhere in `ctrl/**`. |
| 3.2 | No labels / strategy channels into an HDV path | **PASS** | P1 has no HDV input path at all: its output surface is per-slot commands (`CavStepOutput`), consumed by the engine; `is_cav` is read only to gate commands (`mover.py:266`), never exported as a feature; the HDV net contract (`INTERFACES_V2.md` P2, `use_cav_flag=False`) is untouched. CAV plans reach the world only as executed target speeds (amendment 3 A3.1(a) "exactly and never learned"). |
| 3.3 | Batched shapes | **PASS** | All mover tensors `[B, K]`; `test_mover_*` construct `[2, 4]` grids; the new release mask/events are batched per (row, wall id) (`test_mover_wall_v22_release_is_batched_per_row_and_wall`). |
| 3.4 | Determinism | **PASS** | No RNG in `ctrl/**`; stateless release mask + explicit latch; `test_mover_step_is_deterministic` and the new session once-test assert identical repeats. |
| 3.5 | CUDA readiness of the ONNX damping controller (is CPU onnxruntime OK? — documented) | **PASS** (documented) | `PARITY.md` §"Runtime, determinism, CUDA readiness" (added): the damping policy is executed by the vendored `partA_ctrl/onnx2torch.py` torch graph (ops Constant/Div/Gemm/Tanh/Clip/ReduceMax/Mul/Add/Sub/Exp) — **P1 never imports onnxruntime**, so CPU onnxruntime is NOT needed on the GPU box; the same module runs on `cuda` under `torch 2.5.1+cu121` via `CavMover(device=...)`. Device-pinned per run: deterministic on one device (bit-identical repeats tested); bit-level cross-device equality is not guaranteed (float32 GEMM reduction order) — the CPU box is the parity reference. Singleton raw-SI input, embedded normalization and wrapper unchanged. |
| 3.6 | torch 2.5.1-compatible ops only | **PASS** | Only `where/clamp/zeros/full/scatter_reduce/nonzero/item/reshape` (scatter_reduce explicitly whitelisted by INTERFACES_V2.md:259-261); no compile/amp/SDPA/export; no `torch.load` in the piece. Local env torch 2.12.1+cpu passes; nothing 2.6+-only is used. |
| 3.7 | INTERFACES_V2 stub surface preserved | **PASS** | `config.{CAV_ACTIONS, ACTION_CODES, CIRCLES_MODEL_SHA256, WALL_LAW_V21, WALL_LAW_V22}`, `wall_hook.{WallLaw, register_wall_law, resolve_wall_law, wall_law_status, clear_registry, WallLawUnavailable}`, `mover.{CavStepInput, CavStepOutput, CtrlState, execution_speed, CavMover.init_state/step}` all present with the frozen signatures (`__init__.py` re-exports; `test_public_symbols_present`). The new fields have defaults, so existing constructors keep working. |

**Item 3 verdict: PASS.**

## Defects found and fixed (review-fix commits)

1. **v2.2 "once" + event logging missing** (item 1.3) — latch + frozen
   `merge_release` event surface implemented; was unimplementable from the
   stateless hook alone.
2. **Frozen 1e-9 release tolerance missing** (item 1.4).
3. **D_min / hold cap / dmin_unreachable / pair gate missing** (items
   1.5-1.7) — implemented in `ctrl/wall_v22.py` from the frozen post
   copies, with tests.
4. **PARITY.md had no per-strategy tolerance/verdict table** (item 2.2).
5. **Minor robustness:** a registered law without `release_mask` crashed
   the mover; now `getattr`-guarded (`mover.py:382-385`).

## Accepted conventions (documented, not defects)

- The mover maps the batched merge-relative world to the law coordinate as
  `s_road = s_m + 2644.2` (`mover.py:369`), the FROZEN law constant, while
  the SUMO entry lane length is 2644.22 m (G2 `MERGE_OFFSET_M`): the 0.02 m
  difference is ≈ 1-2 ms at wall speed, far below the 1 s control tick that
  quantizes the release. Recorded (first-pass convention, kept).
- Wall-action slots with `wall_group = -1` are paced as a gap-0 singleton
  wall by the mover; member/group assignment is the engine's (P3)
  contract. Unchanged from the reviewed commit (G2 wall parity was
  produced with the recorded member slots and is unaffected).
- `CtrlConfig.wall_law` keeps the INTERFACES_V2.md:167 name;
  `law_version` is the alias required by the task wording (both names
  select the same field; tests pin both).

## Verification (this review)

```
$ WD_FORBID_D=1 python -m pytest ctrl/tests -q   (from output/wd01/wm_n1)
.............................                                            [100%]
29 passed in 5.29s
$ python -m ruff check output/wd01/wm_n1/ctrl
All checks passed!
```

Baseline before the fixes: 17 passed (same command). The 12 added tests
pin: law_version selection, frozen v2.2 constants, D_min = 70·v,
`dmin_unreachable` record + 1e-9 boundary, 2-abreast/span pair gate,
hold-cap formula + `hold_cap` row, `merge_release` event fields, the
once-per-wall mover latch with events, batched per-row release, the
frozen release tolerance, and the session reset semantics.

No SUMO was spent; no sealed data was opened; `WD_FORBID_D=1` enforced
for every pytest invocation.
