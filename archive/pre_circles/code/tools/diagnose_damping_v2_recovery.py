"""Audit frozen traces against the recovery equations. Zero SUMO/training work."""

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.idm.train_native_damping import load_rollout, atomic_json
from idm.damping_recovery_analysis import large_gap_acceleration, margin_from_local_observation

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/probe/native_selectable_damping_v2_check"
OUTPUT = ROOT / "runs/probe/native_selectable_damping_v2_recovery_diagnosis"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((INPUT / "manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert digest(ROOT / name) == expected
    arms = {}
    inputs = {}
    for arm in ("expert", "learned"):
        path = INPUT / f"{arm}_sharp_brake.json"
        r = load_rollout(path)
        inputs[str(path)] = digest(path)
        inputs[str(path.with_suffix(".npz"))] = digest(path.with_suffix(".npz"))
        rows = []
        for step in range(40, 71):
            x = r["actor_observations"][step]
            command = r["commands"][step]
            margin = margin_from_local_observation(x)
            frame = x[68:85]
            ego = float(frame[0]) * 30
            desired = 2.5 + 1.5 * ego
            gap = float(frame[7]) * 100 if frame[9] > 0.5 else None
            blend = 0 if gap is None else float(np.clip((gap - desired) / max(desired, 5), 0, 1))
            catchup = 0 if gap is None else float(np.clip((gap - desired) / 5, -2, 2))
            rows.append(
                dict(
                    step=step,
                    ego_speed_mps=ego,
                    leader_speed_mps=None if gap is None else float(frame[0] - frame[8]) * 30,
                    filtered_ego_mps=float(x[85]) * 30,
                    gap_m=gap,
                    blend=blend,
                    catchup_mps=catchup,
                    requested_correction_mps2=command["requested_delta"],
                    expert_acceleration_mps2=command["expert_acceleration"],
                    target_acceleration_mps2=command["target_acceleration"],
                    operation=command["operation"],
                    visibility=margin,
                )
            )
        invalid = [
            row
            for row in rows
            if row["visibility"] is not None and row["visibility"]["margin_m"] < 0
        ]
        release = next(row for row in rows if row["operation"] == "release_to_native")
        stable = [row for row in rows if row["step"] in (60, 61)]
        arms[arm] = dict(
            rows=rows,
            first_negative_visibility_margin=invalid[0],
            first_release=release,
            average_late_recovery_acceleration=float(
                np.mean([row["target_acceleration_mps2"] for row in stable])
            ),
            zero_correction_asymptote_error=float(
                np.mean([row["target_acceleration_mps2"] for row in stable])
                - large_gap_acceleration()
            ),
        )
    result = dict(
        schema="damping_v2_recovery_equation_audit/1",
        input_sha256=inputs,
        source_sha256={
            n: digest(ROOT / n)
            for n in ("idm/damping_recovery_analysis.py", "tools/diagnose_damping_v2_recovery.py")
        },
        native_contract_sources_unchanged=True,
        new_simulator_seconds=0,
        new_training_steps=0,
        branch_asymptotes_mps2=dict(
            zero_correction=large_gap_acceleration(),
            maximum_constant_correction=large_gap_acceleration(correction_mps2=0.3),
        ),
        arms=arms,
        scope="equations_and_recorded_state_diagnosis_not_a_validated_replacement_controller",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT / "audit.json", result)
    print(
        json.dumps(
            dict(
                asymptotes=result["branch_asymptotes_mps2"],
                first_negative_steps={
                    a: r["first_negative_visibility_margin"]["step"] for a, r in arms.items()
                },
                release_steps={a: r["first_release"]["step"] for a, r in arms.items()},
            )
        )
    )


if __name__ == "__main__":
    main()
