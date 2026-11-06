"""Short CPU wiring proof for the R5MERGE post-merge consistency loss (NOT efficacy).

Rolls the round-2 checkpoint for a few TRAIN-val starts with gradients, builds the
recorded targets of the same window, evaluates the post-merge terms and takes a few
optimiser steps on the residual head.  A handful of steps on two starts cannot show
that the loss fixes the lane-1 phantom queues; it only proves that the switch is
wired, finite, covers recorded 2->1 transitions and carries gradients.

    python scripts/c4/postmerge_cpu_proof.py --wm .../wm_s_stage240.pt \\
        --states .../delay_states.json --select 2 --horizon 40 --steps 2 \\
        --out postmerge_cpu_proof.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
DT_S = 0.5


def _ensure_src() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _delay_module():
    path = Path(__file__).resolve().parent / "wm_delay_check.py"
    spec = importlib.util.spec_from_file_location("wm_postmerge_proof_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse(path) -> str:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {path}")
    return str(path)


def _recorded_targets(record, ids, t0_s, n_steps, arm):
    """``UnrollTargets`` from the recorded SUMO frames of the same window."""
    from wms.eval.normalize import recorded_for
    from wms.train.losses.base import UnrollTargets

    frames = recorded_for(record, arm)
    index = {vid: k for k, vid in enumerate(ids)}
    k = len(ids)
    states = np.zeros((1, n_steps + 1, k, 7), dtype=np.float32)
    present = np.zeros((1, n_steps + 1, k), dtype=bool)
    is_cav = np.zeros((1, n_steps + 1, k), dtype=bool)
    for frame in frames:
        i = int(round((float(frame.get("time_s", 0.0)) - t0_s) / DT_S))
        if i < 0 or i > n_steps:
            continue
        for row in frame.get("vehicles", []):
            slot = index.get(str(row.get("id")))
            if slot is None or not row.get("present", True):
                continue
            states[0, i, slot, 0] = float(row.get("s", 0.0))
            states[0, i, slot, 1 + int(row.get("lane", 0))] = 1.0
            states[0, i, slot, 4] = float(row.get("v", 0.0))
            states[0, i, slot, 6] = float(row.get("length", 5.0) or 5.0)
            present[0, i, slot] = True
            is_cav[0, i, slot] = bool(row.get("is_cav"))
    states[0, 1:, :, 5] = (states[0, 1:, :, 4] - states[0, :-1, :, 4]) / DT_S
    s = torch.from_numpy(states)
    hdv = torch.from_numpy(present[:, :-1] & ~is_cav[:, :-1])
    valid = torch.from_numpy(present)
    weights = torch.ones(1, n_steps, 1)
    return UnrollTargets(
        states=s, n_steps=n_steps, weights=weights, hdv_mask=hdv, aux_mask=hdv,
        aux_count=int(hdv.sum()), denom=(weights * hdv).sum(),
        lane_target=s[:, 1:, :, 1:4].argmax(-1), loss_valid=valid,
        is_cav=torch.from_numpy(is_cav), dt_s=DT_S, position_delta_m=5.0,
        speed_delta_mps=1.0, history_lane_head=False)


def _initial_state(predictor, record, arm, n_steps):
    """The predictor's initial state + teacher/history (gradients enabled later)."""
    from wms.eval.normalize import DT_S as GRID
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    from wms.physics.batched import road_end_exit_bound

    state = record
    tools = predictor.backend.tools
    t0_s, ids, kin, is_cav = t0_vehicles(state)
    driver_history, driver_history_mask = predictor._history.build(state, ids)
    initial = {
        "s_m": torch.as_tensor(kin[None, :, 0]),
        "v_mps": torch.as_tensor(kin[None, :, 4]),
        "a_mps2": torch.zeros((1, len(ids))),
        "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
        "length_m": torch.as_tensor(kin[None, :, 6]),
        "present": torch.ones((1, len(ids)), dtype=torch.bool),
        "is_cav": torch.as_tensor(is_cav[None]),
    }
    exit_s = road_end_exit_bound(getattr(predictor.model, "physics", None))
    if exit_s is not None:
        initial["exit_s_m"] = torch.full((1, len(ids)), float(exit_s))
    teacher, present_seq, junction_seq = recorded_teacher(
        state, t0_s, ids, kin, n_steps, arm, tools)
    assert GRID == DT_S
    return t0_s, ids, initial, teacher, present_seq, junction_seq, \
        driver_history, driver_history_mask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--select", type=int, default=2)
    ap.add_argument("--horizon", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    _ensure_src()
    for path in (args.wm, args.states, args.out):
        _refuse(path)
    helpers = _delay_module()
    starts = helpers.select_evenly(helpers.load_records(args.states), int(args.select))
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    from wms.train.losses.base import LossContext
    from wms.train.losses.post_merge import (PostMergeAccelConfig, post_merge_terms,
                                             prepare_post_merge)

    predictor = WmsPredictor.from_checkpoint(args.wm, RolloutFlags(), device="cpu")
    model = predictor.model
    model.checkpoint_step = False
    model.emit_physics_accel = True
    model.train()
    n_steps = max(1, int(round(float(args.horizon) / DT_S)))
    config = PostMergeAccelConfig(weight=0.5, residual_weight=0.5, window_s=10.0)
    terms = post_merge_terms(config)
    params = [p for p in model.parameters() if p.requires_grad]
    residual_params = list(model.residual_net.parameters())
    temporal_params = list(model.temporal.parameters())
    optimizer = torch.optim.Adam(params, lr=float(args.lr))
    report = []
    for record in starts:
        arm = str(record.get("source_arm") or record.get("arm") or "native")
        recorded = record.get("recorded") or {}
        if isinstance(recorded, dict) and recorded and arm not in recorded:
            arm = sorted(recorded)[0]
        t0_s, ids, initial, teacher, present_seq, junction_seq, dh, dhm = _initial_state(
            predictor, record, arm, n_steps)
        targets = _recorded_targets(record, ids, t0_s, n_steps, arm)
        plan = prepare_post_merge(config, targets)
        row = {"world": record.get("world"), "key": record.get("key"),
               "post_merge_events": int(plan["count"]), "steps": []}
        for step in range(int(args.steps)):
            optimizer.zero_grad(set_to_none=True)
            roll = model.rollout(
                {key: (value.clone() if torch.is_tensor(value) else value)
                 for key, value in initial.items()},
                n_steps=n_steps, k0=0, cav_teacher=teacher, present_seq=present_seq,
                junction_seq=junction_seq, driver_history=dh,
                driver_history_mask=dhm, controller_id=None)
            ctx = LossContext(pred=roll, current=initial, targets=targets,
                              c0=0, c1=n_steps)
            values = {}
            total = torch.zeros(())
            for term in terms:
                value = term(ctx)
                if value is not None:
                    values[term.name] = float(value.value.detach())
                    total = total + value.value
            total.backward()

            def _norm(group):
                return float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                            for p in group if p.grad is not None)))

            optimizer.step()
            row["steps"].append({"step": step, "terms": values,
                                 "total": float(total.detach()),
                                 "grad_norm_total": _norm(params),
                                 "grad_norm_temporal": _norm(temporal_params),
                                 "grad_norm_residual_head": _norm(residual_params)})
        report.append(row)
    doc = {"schema": "c4_postmerge_cpu_proof/1", "wm": args.wm,
           "select": int(args.select), "horizon_s": float(args.horizon),
           "optimizer_steps": int(args.steps),
           "config": {"weight": config.weight,
                      "residual_weight": config.residual_weight,
                      "window_s": config.window_s,
                      "huber_delta_mps2": config.huber_delta_mps2},
           "note": "wiring/finiteness proof only; a few CPU steps are NOT efficacy",
           "starts": report}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
