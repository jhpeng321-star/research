"""Read-only graph and CPU inference audit; no SUMO, fitting, or model selection.

The parser can be installed in the audit directory without changing the runtime:
python -m pip install --no-deps --target <root>/analysis_deps onnx==1.17.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def reconstructed_wrapper(name: str, values: np.ndarray, core: float) -> float:
    """Scalar reconstruction of the exported graph, not a proposed controller."""
    v, vl, gap = (float(x) for x in values[0])
    ego_proxy = max(v, vl / 1e6, gap / 1e6)
    gap_proxy = max(v / 1e6, vl / 1e6, gap)

    def logistic(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))

    gap_weight = logistic(float(np.float32(0.1)) * (gap_proxy - 120))
    if "finaldecn" in name:
        if ego_proxy == 0:
            return float("nan")
        headway = gap_proxy / ego_proxy
        brake_weight = logistic(5 * (2 - headway))
        near = (1 - brake_weight) * core + brake_weight * (0.9375 * headway - 2.375)
        output = (1 - gap_weight) * near + gap_weight * 0.5
    else:
        leader_proxy = max(v / 1e6, vl, gap / 1e6)
        ttc = gap_proxy / max(float(np.float32(0.1)), ego_proxy - leader_proxy)
        if "ttc4p0" in name:
            brake_weight = logistic(4 - ttc)
            near = (1 - brake_weight) * core - 3 * brake_weight
            output = (1 - gap_weight) * near + 0.5 * gap_weight
        else:
            threshold, slope, free_accel = (
                (7, float(np.float32(1.8)), 0.5) if "ttc7p0" in name else (6.5, 1.5, 0.75)
            )
            brake_weight = logistic(slope * (threshold - ttc))
            free = (1 - gap_weight) * core + free_accel * gap_weight
            output = (1 - brake_weight) * free - 3 * brake_weight
    return float(np.clip(output, -3, 1.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("runs/probe/published_damping_reuse_audit_2026-09-06"),
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "analysis_deps"))
    import onnx
    import onnxruntime as ort

    probes = [
        ("steady_15_gap30", [15, 15, 30]),
        ("steady_15_gap60", [15, 15, 60]),
        ("catchup", [5, 20, 30]),
        ("closing", [20, 5, 30]),
        ("leader_stopped", [15, 0, 20]),
        ("large_gap", [15, 15, 150]),
        ("ego_stopped", [0, 15, 30]),
        ("both_stopped_gap5", [0, 0, 5]),
        ("all_zero_boundary", [0, 0, 0]),
        ("zero_gap_moving", [15, 15, 0]),
    ]
    report = {
        "schema": "published_rl_graph_audit_v1",
        "script_sha256": sha256(Path(__file__)),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "probe_scope": "software interface only; not physical efficacy or safety evidence",
        "models": {},
    }
    graph_dir = args.root / "graphs"
    graph_dir.mkdir(exist_ok=True)
    for path in sorted((args.root / "source" / "controllers").glob("*/super_resolution.onnx")):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        initializers = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
        constants = {}
        nodes = []
        for node in model.graph.node:
            attrs = {}
            for attr in node.attribute:
                value = onnx.helper.get_attribute_value(attr)
                if isinstance(value, onnx.TensorProto):
                    value = onnx.numpy_helper.to_array(value)
                attrs[attr.name] = json_value(value)
            if node.op_type == "Constant":
                constants[node.output[0]] = attrs["value"]
            nodes.append({
                "op": node.op_type, "input": list(node.input),
                "output": list(node.output), "attributes": attrs,
            })
        weights_digest = hashlib.sha256()
        weights = {}
        for name, value in sorted(initializers.items()):
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            weights_digest.update(name.encode())
            weights_digest.update(str(value.shape).encode())
            weights_digest.update(value.tobytes())
            weights[name] = {"shape": list(value.shape), "sha256": digest}
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), session_options, providers=["CPUExecutionProvider"]
        )
        singleton = []
        for name, values in probes:
            data = np.asarray([values], dtype=np.float32)
            first = session.run(None, {"input": data})[0]
            second = session.run(None, {"input": data})[0]
            normalized = data / np.array([40, 40, 100], dtype=np.float32)
            core = normalized
            for index in [0, 2, 4, 6]:
                prefix = f"mlp_extractor.policy_net.{index}"
                core = np.tanh(core @ initializers[prefix + ".weight"].T + initializers[prefix + ".bias"])
            core = core @ initializers["action_net.weight"].T + initializers["action_net.bias"]
            reconstructed = reconstructed_wrapper(
                path.parent.name, data, float(np.clip(core, -3, 1.5)[0, 0])
            )
            singleton.append({
                "name": name, "input": values, "output": json_value(first),
                "shape": list(first.shape), "finite": bool(np.isfinite(first).all()),
                "repeat_exact": bool(np.array_equal(first, second, equal_nan=True)),
                "raw_core_output": json_value(core),
                "clipped_core_output": json_value(np.clip(core, -3, 1.5)),
                "reconstructed_output": json_value(reconstructed),
                "reconstruction_absolute_error": json_value(abs(float(first[0, 0]) - reconstructed)),
            })
        batch_inputs = np.asarray([values for _, values in probes[:4]], dtype=np.float32)
        batch_output = session.run(None, {"input": batch_inputs})[0]
        single_values = np.asarray([item["output"][0][0] for item in singleton[:4]])
        diagonal_error = float(np.max(np.abs(batch_output.diagonal() - single_values)))
        (graph_dir / f"{path.parent.name}.txt").write_text(
            onnx.helper.printable_graph(model.graph) + "\n\nConstants:\n" + json.dumps(constants, indent=2),
            encoding="utf-8",
        )
        report["models"][path.parent.name] = {
            "path": path.as_posix(), "sha256": sha256(path),
            "producer": model.producer_name,
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "inputs": [{"name": t.name, "shape": t.shape, "type": t.type} for t in session.get_inputs()],
            "outputs": [{"name": t.name, "shape": t.shape, "type": t.type} for t in session.get_outputs()],
            "all_core_weights_sha256": weights_digest.hexdigest(), "weights": weights,
            "constants": constants, "nodes": nodes, "singleton_probes": singleton,
            "max_finite_reconstruction_error": max(
                row["reconstruction_absolute_error"] for row in singleton if row["finite"]
            ),
            "batch_probe": {
                "inputs": batch_inputs.tolist(), "output_shape": list(batch_output.shape),
                "outputs": batch_output.tolist(), "diagonal_max_abs_error_vs_singletons": diagonal_error,
                "valid_one_output_per_row_shape": batch_output.shape == (len(batch_inputs), 1),
                "adapter_requirement": "Use batch size 1. Output broadcasts across different rows for batch > 1.",
            },
        }
    unique = {item["all_core_weights_sha256"] for item in report["models"].values()}
    report["unique_core_weight_sets"] = len(unique)
    destination = args.root / "graph_inference_audit.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "audit": str(destination), "models": len(report["models"]),
        "unique_core_weight_sets": len(unique),
        "outputs": {name: {r["name"]: r["output"] for r in item["singleton_probes"]}
                    for name, item in report["models"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
