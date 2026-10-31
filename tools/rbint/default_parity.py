"""Capture/compare CPU default-OFF fingerprints before and after branch integration.

This is synthetic compatibility evidence, not a recorded-state smoke or efficacy
test. References contain tensor hashes, not model weights or driver assignments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.model.history import LONG_HISTORY_F
from wms.physics.config import PhysicsConfig


def fingerprints(value):
    if torch.is_tensor(value):
        data = value.detach().cpu().contiguous()
        return {"shape": list(data.shape), "dtype": str(data.dtype),
                "sha256": hashlib.sha256(data.numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): fingerprints(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [fingerprints(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value).__name__)


def synthetic_state():
    return dict(
        s_m=torch.tensor([[-130., -90., -50., -10., 30., 65., -120., -70., -20., 25., 5., 50.]]),
        v_mps=torch.tensor([[23., 22., 21., 20., 19., 21., 24., 22., 19., 18., 14., 10.]]),
        a_mps2=torch.zeros(1, 12),
        lane=torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]]),
        length_m=torch.full((1, 12), 5.),
        present=torch.ones(1, 12, dtype=torch.bool),
        is_cav=torch.tensor([[False] * 8 + [True] + [False] * 3]),
    )


def capture():
    torch.set_num_threads(1)
    result = {}
    cases = {
        "legacy_defaults": (ModelSpec(), PhysicsConfig()),
        "wm2_architecture": (
            ModelSpec(rollout_mode="hybrid", history_s=60., lane_model="hazard_v5",
                      driver_param_mode="physical", merge_decision="first_passage",
                      merge_passage_law="hybrid", follower_courtesy=True,
                      follower_courtesy_version=2, follower_courtesy_intent="pivot",
                      mainline_merge_yield_learnable=True, ramp_speed_adapt_learnable=True,
                      temporal_accel_off=True),
            PhysicsConfig(lane_learnable=True, emergency_braking=True,
                          lane_change_secure_gap=True, secure_gap_veto=True,
                          emergency_onset="driver_b", ramp_lane_end_stop=True),
        ),
    }
    for name, (spec, physics) in cases.items():
        torch.manual_seed(20261031)
        model = WmSModel(spec, physics=physics).eval()
        history_kwargs = {}
        if model.long_history:
            # Explicit missing-history mask: this is a synthetic compatibility
            # fixture, never a claim of authentic recorded pre-t0 history.
            history_kwargs = dict(
                driver_history=torch.zeros(1, 12, model.driver_history_len, LONG_HISTORY_F),
                driver_history_mask=torch.zeros(1, 12, model.driver_history_len,
                                                dtype=torch.bool),
            )
        with torch.no_grad():
            forward = model(synthetic_state(), k_steps=17, **history_kwargs)
            torch.manual_seed(20261031)
            roll = model.rollout(synthetic_state(), n_steps=20,
                                 cav_controller=model.hold_teacher, **history_kwargs)
        result[name] = {"weights": fingerprints(model.state_dict()),
                        "forward": fingerprints(forward), "rollout": fingerprints(roll)}
    return {"schema": "rbint_default_parity/1", "torch": str(torch.__version__),
            "cases": result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    result = capture()
    if args.write:
        with args.write.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Captured {len(result['cases'])} synthetic default-OFF cases: {args.write}")
        return 0
    reference = json.loads(args.compare.read_text(encoding="utf-8"))
    if result != reference:
        changed = [case for case in result["cases"]
                   if result["cases"][case] != reference["cases"].get(case)]
        raise AssertionError(f"default-OFF fingerprint changed: {changed}; "
                             f"torch {reference.get('torch')} -> {result['torch']}")
    print(f"PASS: {len(result['cases'])} default-OFF models, all weights/forward/rollout tensors exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
