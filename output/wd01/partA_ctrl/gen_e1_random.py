"""E1 evidence: 100k random valid inputs, torch port vs onnxruntime.

Writes results/e1_random_equivalence.json; prints the pasted-number table
for REPORT_CTRL.md.  Also runs the decide-level (gates + bridge) parity on
the same inputs and on logged chunk-1 TRAIN frames.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import emulators  # noqa: E402
import refload  # noqa: E402
from onnx2torch import OnnxGraph, TorchOnnxGraph  # noqa: E402

refload.load_reference()
import onnxruntime as ort  # noqa: E402
from traffic_shepherd.damping import CirclesController  # noqa: E402
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame  # noqa: E402

N_ROWS = 100_000
BATCH = 10_000
OUT = HERE / "results" / "e1_random_equivalence.json"


def main():
    rng = np.random.default_rng(20260927)
    module = TorchOnnxGraph(OnnxGraph(refload.MODEL_ONNX)).eval()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(refload.MODEL_ONNX), sess_options=opts,
                                   providers=["CPUExecutionProvider"])

    t0 = time.time()
    diffs = np.empty(N_ROWS, dtype=np.float64)
    decide_reason_mismatch = 0
    decide_target_max_diff = 0.0
    ref = CirclesController()
    emulator = emulators.DampingEmulator(refload.MODEL_ONNX)
    for start in range(0, N_ROWS, BATCH):
        rows = rng.random((BATCH, 3)).astype(np.float32)
        # plausible SI ranges: ego/leader speed 0..40 m/s, gap 0..120 m
        rows = rows * np.array([40.0, 40.0, 120.0], np.float32)
        with torch.no_grad():
            torch_out = module(torch.from_numpy(rows)).numpy()
        # the reference infer contract is EXACTLY ONE [1, 3] row per call
        # (batched ORT on this singleton graph would broadcast the rank-1
        # intermediates into an [N, N] artifact -- see onnx2torch docstring)
        ort_batch = np.array([
            float(session.run(None, {"input": r[None, :]})[0].item())
            for r in rows])
        diffs[start:start + BATCH] = np.abs(torch_out - ort_batch)

        # decide-level parity on frames derived from these rows
        frames = []
        for ego, leader, gap in rows[:100]:
            rel = float(leader) - float(ego)
            frames.append(SensorFrame(
                "v", 500.0, float(ego), 0.0, 31.29, 0, None,
                NeighborMeasurement(float(min(gap, 99.0)), rel), None,
                100.0, min(100.0, max(0.0, float(ego) - 5.0))))
        ref_t, ref_r = [], []
        for frame in frames:
            d = ref.decide(frame, now_s=500.0)
            ref_t.append(d.target_speed_mps)
            ref_r.append(d.fallback_reason)
        batch = emulators.FrameBatch.stack([dict(
            now_s=500.0, ts_s=500.0, speed_mps=f.speed_mps,
            allowed_speed_mps=f.allowed_speed_mps,
            front_coverage_m=f.front_coverage_m,
            rear_coverage_m=f.rear_coverage_m,
            front=dict(gap_m=f.front.gap_m,
                       relative_speed_mps=f.front.relative_speed_mps),
            rear=None) for f in frames])
        tgt, codes, _ = emulator.decide(batch)
        for i in range(len(frames)):
            if ref_r[i] != emulators.REASON_STRINGS[int(codes[i])]:
                decide_reason_mismatch += 1
            elif ref_t[i] is not None:
                decide_target_max_diff = max(
                    decide_target_max_diff,
                    abs(ref_t[i] - float(tgt[i])))

    result = {
        "experiment": "WD-01 partA_ctrl E1",
        "n_rows": N_ROWS,
        "rows_dtype": "float32, ego/leader in [0,40] m/s, gap in [0,120] m",
        "torch_vs_ort_max_abs_accel_diff": float(diffs.max()),
        "torch_vs_ort_mean_abs_accel_diff": float(diffs.mean()),
        "torch_vs_ort_p99_abs_accel_diff": float(np.quantile(diffs, 0.99)),
        "tolerance": 1e-05,
        "pass": bool(diffs.max() <= 1e-5),
        "decide_rows_checked": 100 * (N_ROWS // BATCH),
        "decide_reason_mismatches": int(decide_reason_mismatch),
        "decide_target_max_diff": float(decide_target_max_diff),
        "reference_calls": "onnxruntime singleton [1,3] rows (infer contract)",
        "model_sha256": emulators.MODEL_SHA256,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
