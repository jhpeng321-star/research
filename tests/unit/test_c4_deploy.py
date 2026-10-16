"""C4 deployment runner (wms.control.deploy): fast CPU tests; SUMO parts skip without sumo."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from wms.control import deploy
from wms.control.contract import RewardWeights

R5B = Path("runs/active/WM-S-R5B-20261012/box/gpu/r5b_main_a1/wm_s_best.pt")
HAS_SUMO = shutil.which("sumo") is not None


def _state(k=6, cav=(1, 4), absent=(2,)):
    s = torch.arange(k, dtype=torch.float32)[None] * 25.0 - 400.0
    present = torch.ones(1, k, dtype=torch.bool)
    present[0, list(absent)] = False
    is_cav = torch.zeros(1, k, dtype=torch.bool)
    is_cav[0, list(cav)] = True
    return {"s_m": s, "v_mps": torch.full((1, k), 20.0), "a_mps2": torch.zeros(1, k),
            "lane": torch.where(present, torch.zeros(1, k, dtype=torch.long), -1),
            "present": present, "is_cav": is_cav & present,
            "length_m": torch.full((1, k), 5.0)}


def test_compact_inputs_gathers_present_slots_consistently():
    st = _state()
    hist = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1).expand(1, 6, 10, 6).clone()
    out, keep = deploy.compact_inputs({"state": st, "history": hist,
                                       "history_mask": torch.ones(1, 6, 10, dtype=torch.bool)})
    assert keep.tolist() == [0, 1, 3, 4, 5]
    assert out["state"]["s_m"].shape == (1, 5) and out["history"].shape == (1, 5, 10, 6)
    assert out["history"][0, :, 0, 0].tolist() == [0.0, 1.0, 3.0, 4.0, 5.0]
    assert out["state"]["is_cav"][0].tolist() == [False, True, False, True, False]


def test_long_history_matches_trainer_shape():
    t, k = 121, 4
    kin = np.zeros((t, k, 7), np.float32)
    kin[:, :, 0] = np.arange(t)[:, None] * 10.0 + np.arange(k)[None] * 30.0
    kin[:, :, 1] = 1.0
    kin[:, :, 4] = 20.0
    raw = {"kin": kin, "mask": np.ones((t, k), bool), "target_mask": np.ones((t - 1, k), bool)}
    feat, mask = deploy.long_history(raw, torch.ones(k, dtype=torch.bool), history_s=60.0,
                                     history_stride_s=1.0, dt_s=0.5)
    assert feat.shape == (1, k, 61, 9) and mask.shape == (1, k, 61) and bool(mask.all())


def test_checked_refuses_restricted_paths(tmp_path):
    with pytest.raises(ValueError):
        deploy.checked(tmp_path / "HoldOut" / "x")
    assert deploy.checked(tmp_path / "ok") == tmp_path / "ok"


def test_parser_accepts_campaign_aliases():
    args = deploy.parser().parse_args([
        "--controller", "c4_policy_rsu_check", "--seed", "4107", "--loading", "2600",
        "--penetration", "0.1", "--role", "exposed", "--output", "x", "--policy-ckpt", "p.pt",
        "--wm-ckpt", "w.pt", "--rsu-state", "belief", "--belief-ckpt", "b.pt"])
    assert (deploy.ALIASES[args.controller], args.demand, args.out, args.policy, args.state) == (
        "policy_check", 2600, "x", "p.pt", "belief")


def test_exec_fn_nan_on_non_cav_and_zero_group_signal():
    from wms.control.policy import CavPolicy, RsuGroupEncoder

    torch.manual_seed(0)
    policy, encoder = CavPolicy().eval(), RsuGroupEncoder().eval()
    act = deploy.make_exec_fn(policy, deploy.zero_signal(encoder))
    out = act(_state())
    assert out.shape == (1, 6)
    assert np.isfinite(out[0, [1, 4]]).all() and np.isnan(out[0, [0, 2, 3, 5]]).all()
    assert (out[0, [1, 4]] >= -3.0).all() and (out[0, [1, 4]] <= 1.5).all()


def test_pi_invalidate_keeps_other_vehicles():
    """Regression: dropping one PI vehicle must not erase the other vehicles' PI state."""
    from wms.train.cav_drive import TimelineControllers

    from wms.train.cav_drive import load_controller_dependencies

    # Frozen controller sources with ONE shared (stateless) CIRCLES ONNX session (the former
    # deploy.classic_dependencies; damping/PI are no longer check candidates).
    deps = list(load_controller_dependencies(np.array([1, 2])))
    shared = deps[1](deps[0])
    deps[1] = lambda _config, _shared=shared: _shared
    deps = tuple(deps)
    ids = torch.full((1, 3, 2), 2, dtype=torch.long)
    zeros = torch.zeros(1, 3, 2)
    tc = TimelineControllers(None, {"controller": ids, "acceleration_mps2": zeros,
                                    "target_speed_mps": zeros}, dependencies=deps)
    for vid in ("0:0", "0:1"):
        tc.pi.reset(vid, measured_ego_speed_mps=20.0)
        tc.pi_active.add(vid)
    tc._invalidate("0:0")
    assert "0:1" in tc.pi._state and "0:0" not in tc.pi._state


def test_candidates_and_checker_on_random_wm():
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.reward import team_reward
    from wms.control.rsu_check import RsuChecker
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.manual_seed(2)
    model = WmSModel(ModelSpec(rollout_mode="hybrid")).eval()
    from wms.control.rsu_check import NoiseStream

    streams = [NoiseStream(0, k, k_full=8) for k in range(2)]
    cands = deploy.build_candidates(model, CavPolicy().eval(), RsuGroupEncoder().eval(),
                                    horizon_s=10, sample_streams=streams)
    assert tuple(cands) == deploy.CANDIDATES
    assert not {"native", "damping", "pi"} & set(cands)     # direct-output candidates only
    log = []
    checker = RsuChecker([model], cands, team_reward, RewardWeights(), horizon_s=10,
                         decision_period_s=5, n_driver_samples=2, driver_noise_scale=1.0,
                         embedding_noise_std=0.05, log_fn=log.append)
    st = _state(8, cav=(1, 3, 6), absent=())
    hist = torch.zeros(1, 8, model.history_len, model.history_features)
    inputs, _ = deploy.compact_inputs({"state": st, "history": hist,
                                       "history_mask": torch.ones(1, 8, model.history_len,
                                                                  dtype=torch.bool)})
    for stream in streams:
        stream.begin(0, slot_ids=torch.arange(8))
    choice, evidence = checker.decide(inputs.pop("state"), now_s=300.0, **inputs)
    assert choice in cands and set(evidence) == set(cands)
    assert all(math.isfinite(e["mean"]) for e in evidence.values())
    json.dumps(deploy._jsonable(log[0]), allow_nan=False)


# ------------------------------------------------------------------ SUMO
def _policy_ckpt(path: Path) -> Path:
    from wms.control.policy import CavPolicy, RsuGroupEncoder

    torch.manual_seed(3)
    torch.save({"format": "c4_dreamer/1", "policy": CavPolicy().state_dict(),
                "group_encoder": RsuGroupEncoder().state_dict()}, path)
    return path


@pytest.mark.skipif(not HAS_SUMO, reason="SUMO not installed")
def test_env_per_tick_fallback_mixing(tmp_path):
    from traffic_shepherd.c4.env import CavSumoEnv

    env = CavSumoEnv(k_slots=192, physics_log=False, scenario_root=tmp_path / "scn")
    env.reset(seed=4107, demand_vph=2700, penetration=0.4, warmup_s=60, horizon_s=8,
              controller="direct")
    labels = []
    try:
        for fb in ("damping", "damping", "pi", "pi", "native", "pi", "damping", "native"):
            _, info = env.step(None, fallback=fb)
            labels.append(set(info["actions"].values()))
    finally:
        env.close()
    assert "damping" in labels[1] and "pi-external-u" in labels[3]
    assert labels[4] <= {"native"} and "pi-external-u" in labels[5]
    assert info["summary"]["collisions"] == 0
    with pytest.raises(ValueError):
        env.step(None, fallback="wall")


@pytest.mark.skipif(not HAS_SUMO, reason="SUMO not installed")
def test_deploy_policy_episode(tmp_path):
    ckpt = _policy_ckpt(tmp_path / "policy.pt")
    out = tmp_path / "ep"
    code = deploy.main(["--controller", "policy", "--seed", "4107", "--demand", "2700",
                        "--penetration", "0.2", "--warmup", "60", "--horizon", "5",
                        "--policy", str(ckpt), "--out", str(out), "--no-physics-log"])
    summary = json.loads((out / "DEPLOY_SUMMARY.json").read_text(encoding="utf-8"))
    assert code == 0 and summary["status"] == "complete"
    assert (out / "episode" / "summary.json").is_file()
    assert summary["executed_control_ticks"] == {"policy": 5}
    assert summary["episode_summary"]["lane_change_mode_policy"] == 513


@pytest.mark.skipif(not HAS_SUMO or not R5B.is_file(), reason="SUMO or R5b WM checkpoint missing")
@pytest.mark.parametrize("source", ["oracle", "belief"])
def test_deploy_policy_check_episode(tmp_path, source):
    ckpt = _policy_ckpt(tmp_path / "policy.pt")
    out = tmp_path / source
    # margin huge -> the policy is always kept, so its SUMO execution path is exercised
    code = deploy.main(["--controller", "policy_check", "--state", source, "--seed", "4107",
                        "--demand", "2700", "--penetration", "0.2", "--warmup", "60",
                        "--horizon", "6", "--decision-period", "5", "--imagine-horizon", "10",
                        "--n-driver-samples", "2", "--driver-noise-scale", "1.0", "--margin", "1e9", "--belief-lead", "30",
                        "--policy", str(ckpt), "--wm-ckpt", str(R5B), "--out", str(out),
                        "--no-physics-log"])
    summary = json.loads((out / "DEPLOY_SUMMARY.json").read_text(encoding="utf-8"))
    assert code == 0 and summary["status"] == "complete", summary["error"]
    assert summary["decisions"] == 2 and summary["choice_counts"] == {"policy": 2}
    records = [json.loads(line) for line in (out / "decisions.jsonl").open(encoding="utf-8")]
    assert [r["state_source"] for r in records] == [source, source]
    assert set(records[0]["candidates"]) == set(deploy.CANDIDATES)
    assert "coverage" in records[0]["source"]
    segments = [json.loads(line) for line in (out / "segments.jsonl").open(encoding="utf-8")]
    assert sum(s["ticks"] for s in segments) == 6
    assert set(segments[0]["sumo_cav_actions"]) <= {"direct", "native"}
    assert summary["collisions"] == 0
