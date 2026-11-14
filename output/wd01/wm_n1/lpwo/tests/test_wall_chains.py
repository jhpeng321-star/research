"""Static chain contracts and synthetic log gates; no Bash, trainers or models."""
import ast
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHAIN = ROOT / "lpwo" / "wall_bb_chain.sh"
# WM-S era sources were archived unchanged (archive/wms_era/PATH_MAP.json).
LEGACY_WMS = ROOT.parents[2] / "archive" / "wms_era" / "wm_n1" / "wms"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def text():
    return CHAIN.read_text(encoding="utf-8")


def test_local_model_flags_are_exact_trainer_flags():
    t = text()
    assert "black_box_traj) flags=(--blackbox-traj)" in t
    assert "black_box_id) flags=(--black-box)" in t
    tree = ast.parse((LEGACY_WMS / "cli.py").read_text(encoding="utf-8"))
    flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
             for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
    for flag in ("--blackbox-traj", "--black-box", "--window-list", "--val-window-list",
                 "--exclude-window-list", "--stage-steps", "--outcomes", "--unroll-s"):
        assert flag in flags
    assert "pending outside receipt confirmation" in t


def test_gpu_hosts_paths_and_identical_lf_protocol():
    t = text()
    assert "gpuA|gpuB)" in t
    assert 'GPU=${GPU:-gpuA}' in t
    for var in ("PY", "CODE", "OUT", "TRAIN_WINDOWS", "VAL_WINDOWS", "EXCLUDED_WINDOWS",
                "BASE_PACK", "BASE_IDENTITIES", "TRAIN_OUTCOMES", "DSPEED_PACK",
                "DSPEED_IDENTITIES", "DSPEED_OVERLAY", "DAMPING_PACK", "DAMPING_IDENTITIES",
                "DAMPING_OVERLAY"):
        assert "${" + var + ":?" in t
    assert '--window-list "$TRAIN_WINDOWS" --val-window-list "$VAL_WINDOWS"' in t
    assert '--exclude-window-list "$EXCLUDED_WINDOWS"' in t
    assert "FULL_STAGE_STEPS=2500,2500,1875,1250" in t
    assert "SMOKE_STAGE_STEPS=10,10,10,10" in t
    assert "STAGES=10,30,60,120" in t
    assert "--unroll-s 10 --unroll-s 30 --unroll-s 60 --unroll-s 120" in t
    for flag in ("--sumo-fidelity", "--start-shift-prob 0.75", "--truncation-steps 20",
                 '--num-workers "$NUM_WORKERS"', '--threads "$CPU_THREADS"', "--batch-size 8"):
        assert flag in t
    assert "EXTRA_FLAGS" not in t  # Cannot silently override lists or model identity.
    assert '[[ "${!name}" == /* ]]' in t
    assert 'out/"TRAIN_RECEIPT.json"' in t and 'out/"physics.json"' in t


def test_all_smoke_gates_precede_full_and_fail_closed():
    t = text()
    assert "set -euo pipefail" in t
    assert t.index('"$OUT/smoke_$tag"') < t.index("-m wms.gate_stages") < t.index('"$OUT/full_$tag"')
    assert "--require-terms position,speed,lane" in t
    assert 'Counter(r["stage_s"] for r in rows)=={10:10,30:10,60:10,120:10}' in t
    assert 'r["valid_targets"]>0' in t
    assert 'all(math.isfinite(float(r[k]))' in t
    assert "continue" not in t
    assert "refusing existing OUT" in t
    assert "refusing existing stage" in t


def test_real_total_hard_cap_and_failure_receipts():
    t = text()
    assert "MAX_HOURS=${MAX_HOURS:-8}" in t
    assert "math.isfinite(h) and 0<h<=8" in t
    assert "DEADLINE_S=$((START_S + CAP_S))" in t
    assert "DEADLINE_S - $(date +%s)" in t
    assert 'timeout --signal=KILL "${remaining_s}s" "$@"' in t
    assert '--foreground' not in t and '--kill-after' not in t
    assert 'bounded "$PY" -m wms.cli train' in t
    assert 'bounded "$PY" -m wms.gate_stages' in t
    assert "trap 'rc=$?;" in t
    assert 'log "$tag full allocated' in t
    assert 'log "$tag smoke allocated' in t


def test_inline_smoke_gate_python_is_syntactically_valid():
    t = text()
    start = t.rindex("  bounded \"$PY\" -c '\n") + len("  bounded \"$PY\" -c '\n")
    end = t.index("\n' \"$OUT/smoke_$tag/train_log.jsonl\"", start)
    ast.parse(t[start:end])


def test_inline_physics_capture_python_is_syntactically_valid():
    t = text()
    start = t.index("  bounded \"$PY\" -c '\n") + len("  bounded \"$PY\" -c '\n")
    end = t.index("\n' \"$dest\"", start)
    ast.parse(t[start:end])


def test_cpu_threads_and_loader_queue_budget_are_bounded():
    t = text()
    assert "NUM_WORKERS=${NUM_WORKERS:-0}" in t
    assert "CPU_THREADS=${CPU_THREADS:-4}" in t
    assert "0<=w<=4 and 1<=t<=4" in t
    assert 'export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"' in t
    assert 'export OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"' in t
    assert "aggregate host RAM unmeasured" in t


def smoke_gate_source():
    t = text()
    start = t.rindex("  bounded \"$PY\" -c '\n") + len("  bounded \"$PY\" -c '\n")
    end = t.index("\n' \"$OUT/smoke_$tag/train_log.jsonl\"", start)
    return t[start:end]


def synthetic_log():
    # Same field names/units as _forward_backward -> train_batch -> fit. No
    # trainer import, optimizer step, raw receipt or real log is used here.
    return [{"step": i * 10 + j + 1, "stage_s": stage, "epoch": i + 1,
             "loss": 1.11, "position": 1.0, "speed": 0.1, "lane": 0.01,
             "seconds": 0.02, "gate_curve": [0.0, 0.1, 0.2], "gate_g0": 0.0,
             "gate_g_last": 0.2, "gate_monotone": True, "finite": True,
             "unroll_steps": int(2 * stage), "valid_targets": 20}
            for i, stage in enumerate((10.0, 30.0, 60.0, 120.0)) for j in range(10)]


def test_gate_fields_match_local_trainer_log_schema():
    source = (LEGACY_WMS / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    keys = set()
    for name in ("_forward_backward", "train_batch", "fit"):
        for node in ast.walk(methods[name]):
            if isinstance(node, ast.Dict):
                keys.update(k.value for k in node.keys if isinstance(k, ast.Constant))
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                keys.add(node.slice.value)
    assert set(synthetic_log()[0]) <= keys
    assert 'torch.set_num_threads(int(self.config.threads))' in source


def test_multistage_gate_accepts_synthetic_local_log_contract(monkeypatch):
    from wms import gate_stages
    rows = synthetic_log()
    monkeypatch.setattr(gate_stages, "_read_rows", lambda path: rows)
    monkeypatch.setattr(Path, "is_file", lambda path: False)
    result = gate_stages.evaluate(ROOT / "synthetic_train_log.jsonl",
                                  stages=(10, 30, 60, 120), terms=("position", "speed", "lane"))
    assert result["passed"] and result["records"] == 40


@pytest.mark.parametrize("fault", [None, "nonfinite", "missing_loss", "missing_stage", "empty_targets"])
def test_inline_count_finite_gate_on_synthetic_logs_only(monkeypatch, fault):
    rows = synthetic_log()
    if fault == "nonfinite":
        rows[-1]["speed"] = float("nan")
    elif fault == "missing_loss":
        rows[-1].pop("position")
    elif fault == "missing_stage":
        rows = rows[:-10]
    elif fault == "empty_targets":
        rows[-1]["valid_targets"] = 0
    data = "\n".join(json.dumps(r) for r in rows)
    # Interpret only the embedded pure log checker against in-memory rows. This
    # does not execute the shell chain or any training/model command.
    namespace = {"open": lambda path: io.StringIO(data)}
    monkeypatch.setattr(sys, "argv", ["synthetic-gate", "synthetic-memory-log"])
    code = compile(smoke_gate_source(), str(CHAIN) + ":synthetic-gate", "exec")
    if fault is None:
        exec(code, namespace)
    else:
        with pytest.raises((AssertionError, KeyError)):
            exec(code, namespace)
