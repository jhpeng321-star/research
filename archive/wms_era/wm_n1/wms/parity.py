"""T6 parity harness: torch ``wms.physics_torch`` vs numpy ``rollout2``.

TRAIN-only, no SUMO, one CPU process.  Provides the 8 FIDELITY chunk-4 state
builders, a numpy reference rollout (residual disabled, lane changes optionally
frozen, ``sumo_fidelity`` model-frame geometry), the equivalent torch rollout
and the comparison metric (max abs position/speed difference at 10/60/120 s and
lane-decision agreement).  Used by ``wms/tests/test_rollout2_parity.py`` and
the ``_out/T6.md`` evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]          # repo root E:/rd-wt/ws-torchphys
WM_N1 = Path(__file__).resolve().parents[1]         # .../output/wd01/wm_n1
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

from closedloop.a11_predictor import NONE_CODE, world_from_live  # noqa: E402
from closedloop.live_state import DepartureHistory, live_state_from_row  # noqa: E402
from rollout2 import baseline  # noqa: E402
from rollout2.engine import IdeaABundle, _world_f64  # noqa: E402
from rollout2.state import Rollout2Config, Rollout2State, StrategyActions  # noqa: E402
from rollout2.step import step_once  # noqa: E402

from wms.physics_torch import IDMParameters, PhysicsConfig, PhysicsState  # noqa: E402
from wms.physics_torch import step as torch_step  # noqa: E402

CHUNK = ROOT / "_ext" / "campaign_chunk4"
FIDELITY = [
    ("w8676m2700p20t360", 360.0), ("w8626m2700p02t300", 300.0),
    ("w8629m2000p02t420", 420.0), ("w8678m2700p20t300", 300.0),
    ("w8677m2700p20t360", 360.0), ("w8680m2600p20t300", 300.0),
    ("w8679m2400p20t540", 540.0), ("w8683m2400p20t540", 540.0),
]
N_STEPS = 240          # 120 s at 0.5 s
CHECK = (20, 120, 240)  # 10 / 60 / 120 s


def load_fidelity_state(world: str, t0: float):
    base = CHUNK / f"{world}__base"
    man = json.loads((base / "saved_state" / "STATE_MANIFEST.json").read_text(encoding="utf-8"))
    dep = DepartureHistory(base, t0)
    return live_state_from_row(man["physics_row_at_t0"], dep)


def bundle_none() -> IdeaABundle:
    return IdeaABundle(ctrl=None, hdv=None, arrival=None, corr=None)


def _kin_arrays(state):
    kin = np.asarray(state.veh_kin, np.float64)
    mask = np.asarray(state.rollout_mask, bool)
    lane_oh = kin[:, 1:4]
    lane = np.where(lane_oh.sum(1) > 0, lane_oh.argmax(1), -1).astype(np.int64)
    cav = np.asarray(state.is_cav, bool) & mask
    return kin[:, 0].copy(), kin[:, 4].copy(), lane, mask, cav, kin[:, 6].copy()


def numpy_rollout(state, *, n_steps: int = N_STEPS, lane_changes_enabled: bool = False,
                  t0: float = 0.0):
    world = world_from_live(state, 1)
    k_n = world.slots
    codes = np.full((1, k_n), NONE_CODE, np.int64)
    cfg = Rollout2Config(params=None, integrator="d1", residual_enabled=False,
                         arrivals_enabled=False, lane_changes_enabled=lane_changes_enabled,
                         t0_s=float(t0), physical_exits=True, sumo_fidelity=True)
    p = baseline.Params(baseline.model_frame_params(cfg.params, sumo_idm=True))
    st = Rollout2State(1, k_n)
    cfg.state = st
    acts = StrategyActions(action_code=torch.as_tensor(codes),
                           wall_speed_mps=torch.full((1, k_n), float("nan"), dtype=torch.float64))
    w = _world_f64(world)
    pos = np.zeros((n_steps + 1, k_n))
    spd = np.zeros((n_steps + 1, k_n))
    lns = np.zeros((n_steps + 1, k_n), np.int64)
    msk = np.zeros((n_steps + 1, k_n), bool)
    b = bundle_none()

    def snap(j):
        pos[j] = w.s_m.numpy()
        spd[j] = w.v_mps.numpy()
        lns[j] = w.lane.numpy()
        msk[j] = w.present.numpy()

    snap(0)
    for i in range(n_steps):
        with torch.no_grad():
            w, _ = step_once(w, acts, bundle=b, config=cfg, t_s=float(t0) + i * p.dt)
        snap(i + 1)
    return SimpleNamespace(pos=pos, spd=spd, lns=lns, msk=msk, params=p)


def torch_config_params():
    p = baseline.Params(baseline.model_frame_params(None, sumo_idm=True))
    idm = p.idm_p
    seg_starts = tuple(torch.tensor(p._seg_starts[li], dtype=torch.float64) for li in range(3))
    seg_limits = tuple(torch.tensor(p._seg_lims[li], dtype=torch.float64) for li in range(3))
    lane_lengths = torch.tensor(p.llen, dtype=torch.float64)
    lane_offsets = torch.tensor(p.off, dtype=torch.float64)
    cfg = PhysicsConfig(
        dt=p.dt, lane_count=3, physical_exits=True,
        seg_starts=seg_starts, seg_limits=seg_limits,
        lane_lengths=lane_lengths, lane_offsets=lane_offsets,
        a_clip_min=float(p.a_clip[0]), a_clip_max=float(p.a_clip[1]),
        # numpy rollout2 `_leaders_vector` uses the bumper gap (centre distance
        # minus leader length); the module default is centre distance for the
        # synthetic single-lane tests.
        bumper_gap=True,
        lane_changes_enabled=False,
        merge_start_m=p.s0z, merge_end_m=p.s1z,
        merge_leader_gap_m=float(p.merge_p["gap_floor_m"]),
        merge_follower_gap_m=float(p.idm_p["s0_m"]),
        merge_b_safe_mps2=float(p.mobil_p["b_safe_mps2"]),
        mobil_threshold_mps2=float(p.mobil_p["accel_threshold_mps2"]),
        keep_right_bias_mps2=float(p.mobil_p["keep_right_bias_mps2"]),
    )
    params = IDMParameters.constant(
        1, 1, dtype=torch.float64,
        v0=31.29, accel=float(idm["a_max_mps2"]), decel=float(p.a_clip[1]),
        b_comf=float(idm["b_comf_mps2"]), tau=float(idm["T_s"]),
        min_gap=float(idm["s0_m"]), length=5.0, politeness=float(p.mobil_p["politeness_p"]),
        safe_decel=float(p.mobil_p["b_safe_mps2"]), delta=int(idm["delta"]))
    return cfg, params, p


def torch_rollout(state, *, n_steps: int = N_STEPS):
    cfg_base, params_base, p = torch_config_params()
    s, v, lane, mask, cav, length = _kin_arrays(state)
    k_n = s.shape[0]
    s_t = torch.tensor(s, dtype=torch.float64).unsqueeze(0)
    v_t = torch.tensor(v, dtype=torch.float64).unsqueeze(0)
    lane_t = torch.tensor(lane, dtype=torch.long).unsqueeze(0)
    active_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    # residual-off native CAVs receive no command, so they follow IDM exactly
    # like an HDV; the CAV flag only gates the (unused here) external input.
    cav_t = torch.zeros_like(active_t)
    params = IDMParameters(
        v0=params_base.v0.expand(1, k_n).clone(), accel=params_base.accel.expand(1, k_n).clone(),
        decel=params_base.decel.expand(1, k_n).clone(), b_comf=params_base.b_comf.expand(1, k_n).clone(),
        tau=params_base.tau.expand(1, k_n).clone(), min_gap=params_base.min_gap.expand(1, k_n).clone(),
        length=torch.tensor(length, dtype=torch.float64).unsqueeze(0), politeness=params_base.politeness.expand(1, k_n).clone(),
        safe_decel=params_base.safe_decel.expand(1, k_n).clone(), delta=params_base.delta.expand(1, k_n).clone())
    cfg = PhysicsConfig(**{**cfg_base.__dict__, "lane_changes_enabled": False})
    st = PhysicsState(s_t, v_t, lane_t, active_t, cav_t)
    pos = np.zeros((n_steps + 1, k_n))
    spd = np.zeros((n_steps + 1, k_n))
    lns = np.zeros((n_steps + 1, k_n), np.int64)
    msk = np.zeros((n_steps + 1, k_n), bool)

    def snap(j):
        pos[j] = st.s[0].numpy()
        spd[j] = st.v[0].numpy()
        lns[j] = st.lane[0].numpy()
        msk[j] = st.active[0].numpy()

    snap(0)
    with torch.no_grad():
        for i in range(n_steps):
            out = torch_step(st, params, config=cfg)
            st = out.state
            pos[i + 1] = st.s[0].numpy()
            spd[i + 1] = st.v[0].numpy()
            lns[i + 1] = st.lane[0].numpy()
            msk[i + 1] = st.active[0].numpy()
    return SimpleNamespace(pos=pos, spd=spd, lns=lns, msk=msk, params=p)


def compare(ref, got, *, check=CHECK):
    """Max abs |dpos|/|dv| over vehicles present in both at each check step."""
    out = {}
    for j in check:
        both = ref.msk[j] & got.msk[j]
        dp = np.max(np.abs(ref.pos[j][both] - got.pos[j][both])) if both.any() else 0.0
        dv = np.max(np.abs(ref.spd[j][both] - got.spd[j][both])) if both.any() else 0.0
        lane_agree = float((ref.lns[j][both] == got.lns[j][both]).mean()) if both.any() else 1.0
        out[int(j)] = {"n": int(both.sum()), "max_dpos": float(dp), "max_dv": float(dv),
                       "lane_agree": lane_agree}
    return out


def main():  # pragma: no cover - diagnostic entry
    report = {}
    for world, t0 in FIDELITY:
        st = load_fidelity_state(world, t0)
        ref = numpy_rollout(st, t0=t0)
        got = torch_rollout(st)
        report[world] = compare(ref, got)
        print(world, report[world])
    (ROOT / "_out" / "T6_parity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
