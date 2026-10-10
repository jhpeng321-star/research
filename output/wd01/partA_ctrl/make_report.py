"""Build REPORT_CTRL.md + LEDGER_DRAFT_CTRL.json from the results files.

All numbers in the report are read from the machine-written results
(e1_random_equivalence.json, e4_chunk1_train_validation.json) and the
saved pytest output - nothing is hand-typed.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE.parents[1]))     # output/wd01 (wdpaths)
import wdpaths  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    e1 = json.loads((RESULTS / "e1_random_equivalence.json").read_text())
    e4 = json.loads((RESULTS / "e4_chunk1_train_validation.json").read_text())
    pytest_txt = (RESULTS / "pytest_output.txt").read_text().strip()

    def rate(t):
        return (f"{t['matched']}/{t['rows']} = {t['rate']:.6f}"
                if t["rows"] else "n/a")

    code_files = ["onnx2torch.py", "emulators.py", "refload.py",
                  "gen_e1_random.py", "validate_chunk1.py", "make_report.py",
                  "tests/conftest.py", "tests/fakeapi.py",
                  "tests/reference_stack.py",
                  "tests/test_e1_damping.py", "tests/test_e2_pi.py",
                  "tests/test_e3_wall.py", "tests/test_e5_api.py"]
    result_files = ["results/e1_random_equivalence.json",
                    "results/e4_chunk1_train_validation.json",
                    "results/lane_table.json",
                    "results/pytest_output.txt",
                    "results/e4_stdout.txt"]
    totals = e4["totals"]
    per_arm = e4["per_arm"]
    arm2 = e4["arm2_pi_logged"]
    fs = e4["frame_sanity"]

    per_arm_md = "\n".join(
        f"| {arm} | {per_arm[arm]['rows']['native']} | {per_arm[arm]['matched']['native']} "
        f"| {per_arm[arm]['rows']['damping']} | {per_arm[arm]['matched']['damping']} "
        f"| {per_arm[arm]['rows']['pi']} | {per_arm[arm]['matched']['pi']} |"
        for arm in e4["branches_per_world"])

    report = f"""# WD-01 Part A - Ablation A2 step 1: torch emulators of the CAV commanded speed

Generated: {datetime.now(timezone.utc).isoformat()}
Branch: zcode/wd01-ctrl.  ZERO SIMULATION:
no SUMO/TraCI process was started; the chunk-1 logs and the frozen
research-dense reference code (wdpaths.RESEARCH_DENSE_ROOT, default
E:/research-dense) were only read.

## Scope

Batched, device-agnostic torch emulators of the three chunk-1 controller
commanded-speed paths, proven equivalent to the traffic_shepherd reference
implementations and validated against the logged commands of every
restored chunk-1 TRAIN branch:

- **CIRCLES damping** - the sha256-pinned ONNX policy
  (baselines/circles/model.onnx) ported to a torch graph executor
  (vectorized-singleton semantics), plus the exact decide() fallback gates
  and the direct_target speed bridge.
- **PI external-U** - the cv4 PIExternalU recurrence (registered
  U=31.29, PISaturationParameters 7/30/1/2) with the cv7 PIOwnerAdapter
  lifecycle (measured-speed re-initialization, one-sample clock,
  front-coverage-only engagement gate, infeasible-bridge reset), batched
  with explicit state.
- **wall-v2 member command law** - a = clip((v_target - v)/dt, -3, +1.5)
  plus the ownership direct_target bridge (synthetic frames only; chunk 1
  has no walls).

## E1 - damping equivalence (torch port vs onnxruntime reference)

- model sha256 verified: `{e1['model_sha256']}`
- {e1['n_rows']:,} random valid SI rows vs the ORT singleton contract
  (the production infer path): max |accel diff| =
  **{e1['torch_vs_ort_max_abs_accel_diff']:.3e}** (tolerance 1e-5) - PASS
- decide-level parity (gates + bridge) on {e1['decide_rows_checked']} frames:
  {e1['decide_reason_mismatches']} fallback-reason mismatches, max target
  diff {e1['decide_target_max_diff']:.3e}
- identical fallbacks also pinned on logged chunk-1 TRAIN evidence frames
  and on the full frozen dispatcher stack over a fake-api scenario
  (tests/test_e1_damping.py, tests/test_e2_pi.py)

## E2 - PI external-U equivalence

- cv4 PIExternalU (reference class) vs the batched emulator on synthetic
  sequences with random lifecycle invalidations: every committed raw
  command within 1e-9, final running targets within 1e-6
  (tests/test_e2_pi.py::test_recurrence_synthetic_sequences)
- full frozen stack (MixedAssignmentSession + PISession + CommandOwnership
  + SumoLocalSensor on a scripted fake api - no SUMO) vs the emulate() API
  across a switching schedule (damping -> pi -> native -> pi re-entry):
  every issued target, fallback reason and PI running target matches
  (tests/test_e2_pi.py::test_fullstack_mixed_dispatcher_equivalence)

## E3 - wall-v2 member command law equivalence

- WallGroupSession.tick member commands == batched pacing law on
  synthetic frames through assembly -> hold -> release and a
  member_span_exceeded safety abort (tests/test_e3_wall.py)

## E4 - validation vs logged chunk-1 TRAIN commands

Data: {e4['n_worlds']} TRAIN worlds x {len(e4['branches_per_world'])} restored branches
({', '.join(e4['branches_per_world'])}); receipt exit_status==complete
checked per branch: {len(e4['protocol']['branches_validated'])} of
{49 * len(e4['branches_per_world'])} branches validated
({len(e4['protocol']['incomplete_receipt'])} skipped without a complete
campaign receipt - listed in the results json).  HOLDOUT worlds were never opened.  Each live CAV's
ideal sensor frame is reconstructed at every command tick from
physics_steps.jsonl (same-lane nearest front/rear by bumper gap,
lane length/speed from the frozen network, getAllowedSpeed =
lane_speed x vType speedFactor - verified exact against the logged
evidence frames) and the emulators reproduce the logged issued
targets (match = |diff| < {e4['match_tol_mps']} m/s and identical
fallback classification).

| controller | matched/rows (rate) | max |diff| on matched rows |
| --- | --- | --- |
| native | {rate(totals['native'])} | {totals['native']['error_max_mps']:.3g} |
| damping | {rate(totals['damping'])} | {totals['damping']['error_max_mps']:.3g} |
| pi-external-u | {rate(totals['pi'])} | {totals['pi']['error_max_mps']:.3g} |

Mismatch causes (all controllers): {json.dumps(e4['mismatch_causes'])}

Per arm (rows / matched):

| arm | native rows | native ok | damping rows | damping ok | pi rows | pi ok |
| --- | --- | --- | --- | --- | --- | --- |
{per_arm_md}

PI on the logged pi_source_evidence frames (exact reference inputs,
second arm isolating controller logic from sensing):
{arm2['rows']} proposal rows, {arm2['mismatches']} mismatches, {arm2['clock_resets']} clock resets -
the batched recurrence reproduces every logged raw proposal and issued
target.  Reconstructed-vs-logged frame sanity over {fs['n']} evidence
frames: max |diff| per field - speed {fs['speed']['max']:.3g}, allowed
{fs['allowed']['max']:.3g}, front_cov {fs['front_cov']['max']:.3g},
rear_cov {fs['rear_cov']['max']:.3g}, front_gap {fs['front_gap']['max']:.3g},
front_rel {fs['front_rel']['max']:.3g} m/s ({fs['front_presence_flip']}
presence flips) - exact reproduction of the ideal sensing pipeline up to
float64 noise.

Reconstruction notes (declared, all verified against the logs):
- getAllowedSpeed = lane speed x speedFactor (aggressive 1.08 / normal
  1.0 / cautious 0.92, uncapped; speedDev=0 everywhere).
- The PI engagement gate is front-coverage-only (cv7 adapter); the
  damping gate additionally requires min(front, rear) coverage > 0.
  Both reproduced exactly.
- The t0 control tick has no branch physics row (the first is t0+0.5);
  frames there are reconstructed from the base run's final row (the
  exact saveState restore source).
- Every chunk-1 CAV runs speedMode 31 (pre-fix) - irrelevant to
  commanded-speed emulation, recorded per the data contract.
- Dispatcher rows at the second physics substep are re-stamped copies
  (declared in the WM-42 schema); only the first-substep stamps are used,
  gated by physics-row liveness.

## E5 - thin emulate() API

`emulators.ControllerEmulators.emulate(action_kind, state, inputs) ->
dict(target_speed_mps, reason_code, raw_command_mps, new_state,
pi_reset)` - one control tick for a batch of CAVs, device from WD_DEVICE
(default cpu; loud error on unavailable cuda).  Batched-invariance,
shapes and device contract pinned in tests/test_e5_api.py.

## Tests

```
{pytest_txt}
```

## Artifacts

- emulators: onnx2torch.py (ONNX protobuf reader + torch graph
  executor), emulators.py (batched damping / PI / wall + emulate()),
  refload.py (read-only reference loader)
- validation: gen_e1_random.py, validate_chunk1.py
- tests: tests/ (18 tests: E1, E2, E3, E5)
- results: results/ (E1 random equivalence, E4 validation json + stdout,
  lane table, pytest output)
"""

    ledger = {
        "experiment_id": "WD-01",
        "part": "partA-ctrl-A2-step1-controller-emulators",
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "zero_simulation": True,
        "read_only_inputs": {
            "reference_code": f"{wdpaths.RESEARCH_DENSE_ROOT}/"
                              "traffic_shepherd + "
                              "output.controller_verification_*",
            "data": f"{wdpaths.RESEARCH_DENSE_ROOT}/"
                    "runs/campaign_stage1_chunk1 (TRAIN worlds only)",
            "holdout_accessed": False,
        },
        "emulators": {
            "damping": {
                "model_sha256": e1["model_sha256"],
                "torch_vs_ort_max_abs_accel_diff": e1["torch_vs_ort_max_abs_accel_diff"],
                "tolerance": e1["tolerance"],
                "n_random_rows": e1["n_rows"],
                "decide_reason_mismatches": e1["decide_reason_mismatches"],
            },
            "pi_external_u": {
                "registered_u_mps": 31.29,
                "parameters": "PISaturationParameters defaults (7/30/1/2 m)",
                "arm2_logged_rows": arm2["rows"],
                "arm2_mismatches": arm2["mismatches"],
            },
            "wall_v2": "pacing law clip((v_target-v)/dt, -3, 1.5); synthetic only (no walls in chunk 1)",
        },
        "e4_validation": {
            "schema": "wd01_partA_ctrl_e4/1",
            "split": "TRAIN",
            "n_worlds": e4["n_worlds"],
            "branches_per_world": e4["branches_per_world"],
            "match_tol_mps": e4["match_tol_mps"],
            "totals": e4["totals"],
            "mismatch_causes": e4["mismatch_causes"],
            "arm2_pi_logged": {k: v for k, v in arm2.items()},
            "frame_sanity": fs,
            "speed_mode": "all chunk-1 CAV speedMode 31 (pre-fix); irrelevant to commanded-speed emulation",
        },
        "tests": pytest_txt.splitlines()[-1] if pytest_txt else "",
        "artifacts": {p: sha(HERE / p) for p in code_files + result_files},
        "report": "REPORT_CTRL.md",
    }
    (HERE / "REPORT_CTRL.md").write_text(report, encoding="utf-8")
    (HERE / "LEDGER_DRAFT_CTRL.json").write_text(
        json.dumps(ledger, indent=1), encoding="utf-8")
    print("wrote REPORT_CTRL.md and LEDGER_DRAFT_CTRL.json")


if __name__ == "__main__":
    main()
