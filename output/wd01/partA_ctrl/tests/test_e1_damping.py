"""E1: torch CIRCLES damping emulator == reference (onnxruntime) impl."""
import numpy as np
import pytest
import torch

import emulators
import refload
from onnx2torch import OnnxGraph, TorchOnnxGraph

refload.load_reference()
from traffic_shepherd.damping import CirclesController  # noqa: E402
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame  # noqa: E402

MODEL = refload.MODEL_ONNX
ADAPTER_SHA = emulators.MODEL_SHA256

import onnxruntime as ort  # noqa: E402

_ORT_OPTS = ort.SessionOptions()
_ORT_OPTS.intra_op_num_threads = _ORT_OPTS.inter_op_num_threads = 1


def ort_session():
    return ort.InferenceSession(str(MODEL), sess_options=_ORT_OPTS,
                                providers=["CPUExecutionProvider"])


def random_frames(rng, n):
    frames = []
    for _ in range(n):
        speed = float(rng.uniform(0, 35))
        allowed = 31.29 if rng.random() < 0.8 else 22.22
        fcov = float(rng.uniform(0, 105))
        rcov = float(rng.uniform(0, 100))
        hl = rng.random() < 0.9
        hr = rng.random() < 0.5
        try:
            frames.append(SensorFrame(
                "v", 100.0, speed, 0.0, allowed, 0, None,
                NeighborMeasurement(float(rng.uniform(0.1, max(fcov, 0.1))),
                                    float(rng.uniform(-5, 5))) if hl else None,
                NeighborMeasurement(float(rng.uniform(0.1, max(rcov, 0.1))),
                                    float(rng.uniform(-5, 5))) if hr else None,
                fcov, rcov))
        except ValueError:
            pass  # negative measured state: outside the frame contract
    return frames


def batch_of(frames, now_s=100.0):
    return emulators.FrameBatch.stack([
        None if f is None else dict(
            now_s=now_s, ts_s=f.timestamp_s, speed_mps=f.speed_mps,
            allowed_speed_mps=f.allowed_speed_mps,
            front_coverage_m=f.front_coverage_m,
            rear_coverage_m=f.rear_coverage_m,
            front=None if f.front is None else dict(
                gap_m=f.front.gap_m, relative_speed_mps=f.front.relative_speed_mps),
            rear=None if f.rear is None else dict(
                gap_m=f.rear.gap_m, relative_speed_mps=f.rear.relative_speed_mps))
        for f in frames])


def test_model_sha256_pinned():
    import hashlib
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    assert digest == ADAPTER_SHA


def test_torch_graph_matches_onnxruntime_singleton():
    """Per-row parity of the torch port vs singleton ORT runs (E1 core)."""
    rng = np.random.default_rng(7)
    rows = (rng.random((2000, 3)) * [40.0, 40.0, 120.0]).astype(np.float32)
    module = TorchOnnxGraph(OnnxGraph(MODEL)).eval()
    with torch.no_grad():
        torch_out = module(torch.from_numpy(rows)).numpy()
    session = ort_session()
    ort_out = np.array([
        float(session.run(None, {"input": row[None, :]})[0].item())
        for row in rows])
    diff = np.abs(torch_out - ort_out)
    assert diff.max() <= 1e-5, f"max |accel diff| = {diff.max()}"


def test_decide_parity_random_frames():
    rng = np.random.default_rng(11)
    frames = random_frames(rng, 500) + [None]
    ref = CirclesController()
    ref_t, ref_r = [], []
    for frame in frames:
        d = ref.decide(frame, now_s=100.0)
        ref_t.append(d.target_speed_mps)
        ref_r.append(d.fallback_reason)
    emulator = emulators.DampingEmulator(MODEL)
    tgt, codes, accel = emulator.decide(batch_of(frames))
    for i, frame in enumerate(frames):
        er = emulators.REASON_STRINGS[int(codes[i])]
        et = float(tgt[i])
        assert ref_r[i] == er, f"row {i}: {ref_r[i]!r} != {er!r}"
        if ref_t[i] is None:
            assert np.isnan(et)
        else:
            assert abs(ref_t[i] - et) <= 1e-5, \
                f"row {i}: {ref_t[i]} vs {et}"


def test_direct_target_parity_including_infeasible():
    from traffic_shepherd.config import load_config
    from traffic_shepherd.damping import direct_target

    class F:
        def __init__(self, speed, allowed):
            self.speed_mps = speed
            self.allowed_speed_mps = allowed

    rng = np.random.default_rng(3)
    speed = rng.uniform(0, 45, 20000)
    allowed = rng.uniform(0, 40, 20000)
    accel = rng.uniform(-6, 4, 20000)
    cmd, tgt = emulators.direct_target(
        torch.tensor(speed, dtype=torch.float64),
        torch.tensor(allowed, dtype=torch.float64),
        torch.tensor(accel, dtype=torch.float64))
    cfg = load_config()
    for i in range(0, 20000, 97):
        rc, rt = direct_target(F(float(speed[i]), float(allowed[i])),
                               float(accel[i]), cfg)
        assert (rc is None) == bool(torch.isnan(cmd[i]))
        if rc is None:
            assert torch.isnan(tgt[i])
        else:
            assert abs(rc - float(cmd[i])) <= 1e-12
            assert abs(rt - float(tgt[i])) <= 1e-12


def test_frame_problem_reason_order():
    """Adversarial frames hitting every gate, in reference priority order."""
    ref = CirclesController()
    emulator = emulators.DampingEmulator(MODEL)
    # missing_observation
    tgt, codes, _ = emulator.decide(batch_of([None]))
    assert int(codes[0]) == emulators.REASON_CODES["missing_observation"]
    # noncurrent_observation
    off_time = SensorFrame("v", 100.5, 20.0, 0.0, 31.29, 0, None, None, None,
                           50.0, 10.0)
    d = ref.decide(off_time, now_s=100.0)
    assert d.fallback_reason == "noncurrent_observation"
    tgt, codes, _ = emulator.decide(batch_of([off_time]))
    assert int(codes[0]) == emulators.REASON_CODES["noncurrent_observation"]
    # invalid_observation (negative speed is unreachable through the
    # SensorFrame constructor; the raw gate is exercised directly)
    batch = emulators.FrameBatch.stack([
        dict(now_s=100.0, ts_s=100.0, speed_mps=-1.0, allowed_speed_mps=31.29,
             front_coverage_m=50.0, rear_coverage_m=10.0, front_present=False,
             front_gap_m=0.0, front_rel_mps=0.0, rear_present=False,
             rear_gap_m=0.0, rear_rel_mps=0.0, frame_present=True)])
    assert int(emulators.frame_problem_codes(batch)[0]) == \
        emulators.REASON_CODES["invalid_observation"]
    # coverage_exceeds_contract
    batch = emulators.FrameBatch.stack([
        dict(now_s=100.0, ts_s=100.0, speed_mps=20.0, allowed_speed_mps=31.29,
             front_coverage_m=100.01, rear_coverage_m=10.0, front_present=False,
             front_gap_m=0.0, front_rel_mps=0.0, rear_present=False,
             rear_gap_m=0.0, rear_rel_mps=0.0, frame_present=True)])
    assert int(emulators.frame_problem_codes(batch)[0]) == \
        emulators.REASON_CODES["coverage_exceeds_contract"]
    # invalid_neighbor_measurement (gap beyond coverage)
    batch = emulators.FrameBatch.stack([
        dict(now_s=100.0, ts_s=100.0, speed_mps=20.0, allowed_speed_mps=31.29,
             front_coverage_m=50.0, rear_coverage_m=10.0, front_present=True,
             front_gap_m=60.0, front_rel_mps=0.0, rear_present=False,
             rear_gap_m=0.0, rear_rel_mps=0.0, frame_present=True)])
    assert int(emulators.frame_problem_codes(batch)[0]) == \
        emulators.REASON_CODES["invalid_neighbor_measurement"]
    # missing_or_uncovered_leader: valid frame, no front
    frame_ok_nofront = SensorFrame("v", 100.0, 20.0, 0.0, 31.29, 0, None,
                                   None, None, 50.0, 10.0)
    d = ref.decide(frame_ok_nofront, now_s=100.0)
    assert d.fallback_reason == "missing_or_uncovered_leader"
    tgt, codes, _ = emulator.decide(batch_of([frame_ok_nofront]))
    assert int(codes[0]) == emulators.REASON_CODES["missing_or_uncovered_leader"]
    assert np.isnan(float(tgt[0]))


def _evidence_frames(branch_run):
    """Logged pi_source_evidence frames from a chunk-1 TRAIN branch."""
    import json
    run_dir = refload.CHUNK1_RUNS / branch_run / "run"
    frames = []
    for line in open(run_dir / "suffix_commands.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row.get("kind") == "pi_source_evidence":
            frames.append(row["frame"])
            if len(frames) >= 300:
                break
    return frames


def _world_split(branch_run):
    import json
    plan = json.load(open(refload.WORLD_PLAN, encoding="utf-8"))
    world = branch_run.split("__")[0]
    return next(w["split"] for w in plan["worlds"] if w["world"] == world)


@pytest.mark.parametrize("branch_run", [
    "w8411m2700p02t240__pi",
    "w8413m2600p02t420__switching1",
    "w8416m2000p02t300__fixed2",
])
def test_decide_parity_on_logged_chunk1_frames(branch_run):
    """E1 'logged frames' leg: identical fallbacks on real captured frames."""
    assert _world_split(branch_run) == "TRAIN"
    frames_logged = _evidence_frames(branch_run)
    assert len(frames_logged) >= 50
    batch_rows = [dict(
        now_s=f["timestamp_s"], ts_s=f["timestamp_s"], speed_mps=f["speed_mps"],
        allowed_speed_mps=f["allowed_speed_mps"],
        front_coverage_m=f["front_coverage_m"], rear_coverage_m=f["rear_coverage_m"],
        front=None if f["front"] is None else dict(
            gap_m=f["front"]["gap_m"], relative_speed_mps=f["front"]["relative_speed_mps"]),
        rear=None if f["rear"] is None else dict(
            gap_m=f["rear"]["gap_m"], relative_speed_mps=f["rear"]["relative_speed_mps"]))
        for f in frames_logged]
    ref = CirclesController()
    ref_t, ref_r = [], []
    for f in frames_logged:
        d = ref.decide(_to_sensor_frame(f), now_s=f["timestamp_s"])
        ref_t.append(d.target_speed_mps)
        ref_r.append(d.fallback_reason)
    emulator = emulators.DampingEmulator(MODEL)
    tgt, codes, _ = emulator.decide(emulators.FrameBatch.stack(batch_rows))
    for i in range(len(frames_logged)):
        assert ref_r[i] == emulators.REASON_STRINGS[int(codes[i])], f"row {i}"
        if ref_t[i] is None:
            assert np.isnan(float(tgt[i]))
        else:
            assert abs(ref_t[i] - float(tgt[i])) <= 1e-5, f"row {i}"


def _to_sensor_frame(f):
    return SensorFrame(
        f["vehicle_id"], f["timestamp_s"], f["speed_mps"], f["acceleration_mps2"],
        f["allowed_speed_mps"], f["lane_index"],
        f.get("distance_to_merge_m"),
        None if f["front"] is None else NeighborMeasurement(
            f["front"]["gap_m"], f["front"]["relative_speed_mps"]),
        None if f["rear"] is None else NeighborMeasurement(
            f["rear"]["gap_m"], f["rear"]["relative_speed_mps"]),
        f["front_coverage_m"], f["rear_coverage_m"],
        f.get("previous_command_delta", 0.0), f.get("lane_id"), f.get("position_m"))
