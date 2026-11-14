"""Read-only decomposition of the fixed external controller's saved responses.

No SUMO, fitting, parameter selection, or acceptance data. The ideal sampled
linearization is diagnostic and does not validate the actual hybrid controller.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
import torch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/probe/published_damping_reuse_audit_2026-09-06"
FIT = ROOT / "runs/probe/published_idm_fit_check_2026-09-06"
MODEL_HASH = "762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    return dict(zip(["min", "p10", "median", "p90", "max"], np.quantile(values, [0, .1, .5, .9, 1]).tolist()))


class Response:
    def __init__(self):
        sys.path.insert(0, str(AUDIT / "analysis_deps"))
        import onnx

        path = FIT / "candidate.onnx"
        if sha256(path) != MODEL_HASH:
            raise ValueError("wrong model")
        model = onnx.load(path)
        self.weights = {
            t.name: torch.as_tensor(onnx.numpy_helper.to_array(t).copy(), dtype=torch.float64)
            for t in model.graph.initializer
        }

    def components(self, x):
        # Exact first-candidate graph semantics, evaluated at double precision
        # for derivative diagnostics. Compare reconstructed output to saved ORT.
        hidden = x / torch.tensor([40., 40., 100.], dtype=x.dtype)
        for index in [0, 2, 4, 6]:
            prefix = f"mlp_extractor.policy_net.{index}"
            hidden = torch.tanh(hidden @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"])
        raw_core = hidden @ self.weights["action_net.weight"].T + self.weights["action_net.bias"]
        core = raw_core[:, 0].clamp(-3, 1.5)
        vp = (x / torch.tensor([1., 1e6, 1e6], dtype=x.dtype)).amax(dim=1)
        gp = (x / torch.tensor([1e6, 1e6, 1.], dtype=x.dtype)).amax(dim=1)
        h = gp / vp
        p = 1 - torch.sigmoid(5 * (h - 2))
        q = torch.sigmoid(float(np.float32(.1)) * (gp - 120))
        neural_weight = (1 - q) * (1 - p)
        nn_component = neural_weight * core
        headway_component = (1 - q) * p * (.9375 * h - 2.375)
        large_gap_component = q * .5
        output = (nn_component + headway_component + large_gap_component).clamp(-3, 1.5)
        return {
            "core": core, "headway_s": h, "neural_weight": neural_weight,
            "nn_component": nn_component, "headway_component": headway_component,
            "large_gap_component": large_gap_component, "output": output,
        }

    def evaluate(self, x):
        with torch.no_grad():
            return {k: v.numpy() for k, v in self.components(torch.as_tensor(x, dtype=torch.float64)).items()}


def equilibrium_audit(model):
    result = []
    # Fixed engineering anchors, not a search over controller parameters.
    for speed in [5., 15., 25.]:
        grid = np.linspace(2.5, 100, 391)
        acceleration = model.evaluate(np.column_stack([np.full(len(grid), speed), np.full(len(grid), speed), grid]))["output"]
        roots = []
        for left, right, a_left, a_right in zip(grid[:-1], grid[1:], acceleration[:-1], acceleration[1:]):
            if a_left * a_right < 0:
                roots.append(brentq(lambda g: float(model.evaluate([[speed, speed, g]])["output"][0]), left, right))
        for gap in roots:
            state = torch.tensor([[speed, speed, gap]], dtype=torch.float64, requires_grad=True)
            output = model.components(state)["output"]
            gradient = torch.autograd.grad(output.sum(), state)[0].numpy()[0]
            av, al, ag = map(float, gradient)
            sampled = []
            for dt, physics in [(1., .5), (.1, .1), (.05, .05)]:
                # v' = v + dt*a; g' = g + dt*(vl-v) - c*a.
                # c follows semi-implicit Euler position updates with constant a.
                c = .5 * dt * (dt + physics)
                jacobian = np.array([[1 + dt * av, dt * ag], [-dt - c * av, 1 - c * ag]])
                eigenvalues = np.linalg.eigvals(jacobian)
                # Constant leader-speed input over each interval; output is ego
                # speed at decision times. Pole stability does not imply |H|<=1.
                omega = np.r_[0., np.geomspace(1e-5, np.pi / dt, 12000)]
                z = np.exp(1j * omega * dt)
                b1, b2 = dt * al, dt - c * al
                transfer = ((z - jacobian[1, 1]) * b1 + jacobian[0, 1] * b2) / (
                    (z - jacobian[0, 0]) * (z - jacobian[1, 1]) - jacobian[0, 1] * jacobian[1, 0]
                )
                peak = int(np.argmax(np.abs(transfer)))
                sampled.append({
                    "command_dt_s": dt, "physics_dt_s": physics,
                    "eigenvalues": [[float(z.real), float(z.imag)] for z in eigenvalues],
                    "spectral_radius": float(np.max(np.abs(eigenvalues))),
                    "zero_frequency_gain": float(abs(transfer[0])),
                    "peak_speed_gain_on_frequency_grid": float(abs(transfer[peak])),
                    "peak_omega_rad_per_s": float(omega[peak]),
                })
            result.append({
                "speed_mps": speed, "gap_m": gap, "headway_s": gap / speed,
                "absolute_acceleration_at_root": float(output.detach()[0]),
                "acceleration_derivative_ego_speed": av,
                "acceleration_derivative_leader_speed": al,
                "acceleration_derivative_gap": ag,
                "continuous_nonamplification_margin": av * av - al * al - 2 * ag,
                "sampled_ideal_models": sampled,
            })
    return result


def main():
    torch.set_num_threads(1)
    manifest_path = FIT / "fit_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(ROOT / "experiments/idm/check_published_idm_fit.py") != manifest["runner_sha256"]:
        raise ValueError("fit runner changed")
    model = Response()
    records = []
    for seed, loading, rate, mode in manifest["jobs"]:
        if mode == "native":
            continue
        key = f"s{seed}_p{round(rate * 100):02}_{mode}"
        path = FIT / "results" / f"{key}.json"
        record = json.loads(path.read_text())
        arrays_path = path.with_suffix(".npz")
        if sha256(arrays_path) != record["arrays_sha256"]:
            raise ValueError("saved arrays changed")
        arrays = np.load(arrays_path)
        observations, actions = arrays["observations"], arrays["actions"]
        components = model.evaluate(observations[:, 1:])
        output = components["output"]
        if not np.array_equal(np.isfinite(output), np.isfinite(actions[:, 0])):
            raise ValueError("nonfinite reconstruction mismatch")
        finite = np.isfinite(output)
        error = np.max(np.abs(output[finite] - actions[finite, 0]))
        if error > 5e-5:
            raise ValueError(f"reconstruction differs: {key} {error}")
        executed = np.isfinite(actions[:, 1])
        changed = executed & (np.abs(actions[:, 1] - actions[:, 0]) > 1e-6)
        weight = components["neural_weight"][finite]
        record_out = {
            "key": key, "record_sha256": sha256(path), "arrays_sha256": sha256(arrays_path),
            "emitted_outputs": len(observations), "max_ort_reconstruction_error_mps2": float(error),
            "finite_outputs": int(finite.sum()), "executed_commands": int(executed.sum()),
            "modified_commands": int(changed.sum()),
            "modified_fraction_of_executed": float(changed.sum() / executed.sum()),
            "neural_weight_quantiles": quantiles(weight),
            "neural_weight_below_0p1_fraction": float(np.mean(weight < .1)),
            "neural_weight_below_0p5_fraction": float(np.mean(weight < .5)),
            "positive_core_made_negative_by_wrapper_fraction": float(np.mean((components["core"][finite] > 0) & (output[finite] < 0))),
            "raw_outside_workspace_acceleration_envelope_fraction": float(np.mean((output[finite] < -1) | (output[finite] > .5))),
            "headway_s_quantiles": quantiles(components["headway_s"][finite]),
            "mean_absolute_additive_components_mps2": {
                key: float(np.mean(np.abs(components[key][finite])))
                for key in ["nn_component", "headway_component", "large_gap_component"]
            },
            "response_by_measured_headway": [],
        }
        for low, high in [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, float("inf"))]:
            selected = finite & (components["headway_s"] >= low) & (components["headway_s"] < high)
            if not selected.any():
                continue
            record_out["response_by_measured_headway"].append({
                "range_s": [low, high if np.isfinite(high) else "inf"],
                "decisions": int(selected.sum()),
                "mean_core_acceleration_mps2": float(np.mean(components["core"][selected])),
                "mean_wrapped_acceleration_mps2": float(np.mean(output[selected])),
                "mean_neural_weight": float(np.mean(components["neural_weight"][selected])),
                "mean_executed_acceleration_mps2": float(np.mean(actions[selected & executed, 1])) if (selected & executed).any() else None,
            })
        records.append(record_out)
    result = {
        "schema": "external_controller_response_diagnosis_v1",
        "script_sha256": sha256(Path(__file__)), "fit_manifest_sha256": sha256(manifest_path),
        "model_sha256": MODEL_HASH,
        "new_simulation_seconds": 0, "training_updates": 0,
        "records": records,
        "fixed_policy_equilibria": equilibrium_audit(model),
        "limits": [
            "Neural mixing weight is a graph coefficient, not attribution of traffic outcomes.",
            "Rows are sensor-valid emitted decisions, not equally weighted vehicles or all CAV time.",
            "Saved observations lack full rear/acceleration context; this cannot isolate each projection constraint.",
            "Ideal linearization omits SUMO safety, cut-ins, saturation, sensor loss and native switching.",
            "Two exposed corridor cases are not prospective acceptance or representative penetration validation.",
        ],
    }
    out = AUDIT / "response_diagnosis.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out), "outputs_audited": sum(r["emitted_outputs"] for r in records),
        "records": [{k: r[k] for k in ["key", "neural_weight_below_0p1_fraction", "modified_fraction_of_executed"]} for r in records],
        "equilibria": result["fixed_policy_equilibria"],
    }, indent=2))


if __name__ == "__main__":
    main()
