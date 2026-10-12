"""Imagined cooperation branches: the SUMO branching arms re-run inside WM-S.

Given the SUMO environment at a saved decision point (``traffic_shepherd.c4.branching``),
this computes the same branch improvements ``D_X`` in imagination from the env's true
WM-frame state + rolling history (oracle source, as ``wms.control.deploy --state oracle``),
so the WM's cooperation estimates can be validated against the real-simulator branches.

Arm semantics in imagination: arm members are WM CAV slots driven by the controller under
test (``is_cav`` = member); every non-member CAV is modelled as a native driver (an HDV
slot with encoder-identified driver parameters), matching SUMO, where released CAVs follow
SUMO IDM.  Driver parameters are encoded once with every slot treated as a driver and the
same standard-normal driver draws are reused for every arm (common random numbers).

Approximations (documented in BRANCHING_NOTES.md): no entries (vehicles inserted during the
branch and source waiting are absent), CAV lane changes only via the forced ramp merge,
team metrics = ``wms.control.reward.team_reward`` (no source waiting term).
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Optional

import numpy as np
import torch

from .contract import (ACCEL_MAX_MPS2, ACCEL_MIN_MPS2, PHYSICS_DT_S, STEPS_PER_CONTROL,
                       RewardWeights)

#: WM team_reward key -> SUMO branch metric name (traffic_shepherd.c4.branching)
METRIC_MAP = {"reward": "reward", "delay": "road_delay_vs", "throughput": "throughput_soft",
              "smooth": "smooth", "hard_brake": "hard_brake", "cost_ttc": "cost_ttc"}
#: +1 benefit, -1 cost (same convention as the SUMO side: D positive = better)
SIGN = {"reward": +1, "road_delay_vs": -1, "throughput_soft": +1, "smooth": -1,
        "hard_brake": -1, "cost_ttc": -1}
_MODELS: dict = {}


def _model(ckpt, device="cpu"):
    key = (str(ckpt), str(device))
    if key not in _MODELS:
        from wms.eval.backend import load_backend
        _MODELS[key] = load_backend(ckpt, device=device).model
    return _MODELS[key]


class CallableCavController:
    """Generic ``state -> accel [1,K]`` SUMO callable as a WM-S ``CavController`` [B,K,4].

    Each batch row has its own deep copy of the callable.  The callable sees the full
    ``K_full``-slot state (compacted slots scattered back) with the ORIGINAL CAV mask, as in
    SUMO; its output is applied to the row's member CAV slots only, held for one control
    interval (2 physics steps) and executed with ``direct_command_motion``.
    """

    def __init__(self, fns: list, keep: torch.Tensor, k_full: int, cav_full: torch.Tensor):
        self.fns, self.keep, self.k_full = fns, keep, int(k_full)
        self.cav_full = cav_full          # [B, K_compact] original CAV mask
        self.held = None
        self.index = 0

    def _full(self, state, b):
        k = self.k_full
        out = {}
        for key, dtype in (("s_m", torch.float32), ("v_mps", torch.float32),
                           ("a_mps2", torch.float32), ("lane", torch.long),
                           ("present", torch.bool), ("length_m", torch.float32)):
            value = state.get(key)
            if value is None:
                value = (torch.full_like(state["s_m"], 5.0) if key == "length_m"
                         else torch.zeros_like(state["s_m"]))
            full = torch.zeros(1, k, dtype=dtype) if key != "length_m" else torch.full((1, k), 5.0)
            full[0, self.keep] = value[b].detach().cpu().to(dtype)
            out[key] = full
        cav = torch.zeros(1, k, dtype=torch.bool)
        cav[0, self.keep] = self.cav_full[b].cpu() & out["present"][0, self.keep]
        out["is_cav"] = cav
        out["junction"] = torch.zeros(1, k, dtype=torch.bool)
        return out

    def __call__(self, state):
        from wms.model.cav import direct_command_motion

        from .bridge import _merge_lane
        s, v = state["s_m"], state["v_mps"]
        present = state["present"].bool()
        mask = present & state["is_cav"].bool() & (state["lane"] >= 0)
        if self.index % STEPS_PER_CONTROL == 0 or self.held is None:
            held = torch.zeros_like(s)
            for b, fn in enumerate(self.fns):
                out = fn(self._full(state, b))
                if isinstance(out, dict):
                    out = out["accel"]
                elif isinstance(out, tuple):
                    out = out[0]
                acc = torch.as_tensor(np.asarray(out.detach().cpu() if torch.is_tensor(out)
                                                 else out, np.float64)).reshape(-1)
                row = acc[self.keep.cpu()].to(s.dtype)
                held[b] = torch.nan_to_num(row, nan=0.0)
            self.held = held
        self.index += 1
        held = torch.where(mask, self.held.clamp(ACCEL_MIN_MPS2, ACCEL_MAX_MPS2),
                           torch.zeros_like(self.held))
        lane = _merge_lane(state, mask)
        vv = torch.where(present, v, torch.zeros_like(v))
        ss = torch.where(present, s, torch.zeros_like(s))
        command = torch.stack((held, vv + ACCEL_MAX_MPS2 * PHYSICS_DT_S), -1)
        next_s, next_v, applied, _ = direct_command_motion(ss, vv, command, PHYSICS_DT_S,
                                                           ACCEL_MIN_MPS2, ACCEL_MAX_MPS2)
        return torch.stack((next_s, lane.to(next_s.dtype), next_v, applied), -1)


def _member_slots(env, members, keep) -> torch.Tensor:
    """[K_compact] bool: compact slots whose vehicle is a member CAV (None = every CAV)."""
    f = env.frame
    ids = env.builder.vehicle_ids
    out = []
    for slot in keep.tolist():
        vi = int(f.vid_index[slot])
        cav = bool(f.present[slot] and f.is_cav[slot])
        out.append(cav and (members is None or (vi >= 0 and ids[vi] in members)))
    return torch.as_tensor(out, dtype=torch.bool)


@torch.no_grad()
def imagined_cooperation(ckpt, env, arms: list, controller, fn_snap, horizon_s: int, *,
                         n_samples: int = 4, seed: int = 0, device: str = "cpu",
                         reward_fn: Optional[Callable] = None) -> dict:
    """Imagined ``{"metrics": {arm: {metric: mean}}, "D": {arm: {metric: D}}}`` (paired)."""
    from .deploy import OracleSource, TimelineCandidate, classic_dependencies, compact_inputs
    from .reward import team_reward

    started = time.monotonic()
    reward_fn = reward_fn or team_reward
    model = _model(ckpt, device)
    raw, _ = OracleSource(model).inputs(env)
    inputs, keep = compact_inputs(raw)
    state = inputs.pop("state")
    k_full = int(raw["state"]["s_m"].shape[1])
    arms = [a for a in arms if a["name"] != "B_repeat"]
    names = [a["name"] for a in arms]
    n_arm, n = len(arms), int(n_samples)
    batch = n_arm * n
    cav_orig = state["is_cav"][0].bool()

    # drivers: every slot treated as a driver for the encoding (released CAVs are native)
    enc_state = dict(state, is_cav=torch.zeros_like(state["is_cav"]))
    hist = {key: inputs.get(key) for key in ("history", "history_mask", "driver_history",
                                             "driver_history_mask")}
    for h, m in (("history", "history_mask"), ("driver_history", "driver_history_mask")):
        if hist[h] is not None and hist[m] is None:
            hist[m] = torch.ones_like(hist[h][..., 0], dtype=torch.bool)
    model.eval()
    encoded, embedding, params = model.encode_drivers(enc_state, **hist)
    gen = torch.Generator().manual_seed(int(seed))
    drivers = {}
    if params is not None:
        pn = tuple(params)
        values = torch.stack([params[p] for p in pn], -1)
        draws = torch.randn((n, *values.shape[1:]), generator=gen).to(values)
        scale = encoded["uncertainty"] if encoded is not None else values.abs() * 0.1
        samples = values + draws * scale
        samples = torch.maximum(torch.minimum(samples, values * 1.5), values * 0.5)
        samples = samples.repeat(n_arm, 1, 1)
        drivers["driver_params"] = {p: samples[..., i] for i, p in enumerate(pn)}
    if embedding is not None:
        draws = torch.randn((n, *embedding.shape[1:]), generator=gen).to(embedding)
        drivers["driver_embedding"] = (embedding + 0.05 * draws).repeat(n_arm, 1, 1)

    def rep(x):
        return x.repeat(batch, *([1] * (x.ndim - 1))) if torch.is_tensor(x) and x.ndim else x
    bstate = {key: rep(value) for key, value in state.items()}
    member = torch.stack([_member_slots(env, a["members"], keep) for a in arms])  # [A, Kc]
    bstate["is_cav"] = member.repeat_interleave(n, 0) & bstate["present"].bool()
    bhist = {key: rep(value) for key, value in hist.items()}

    steps = int(round(horizon_s / PHYSICS_DT_S))
    if controller.named is None:
        fns = [copy.deepcopy(fn_snap) for _ in range(batch)]
        ctrl = CallableCavController(fns, keep, k_full, cav_orig[None].repeat(batch, 1))

        def dispatch(predicted):
            return ctrl(predicted)
    elif controller.named == "native":
        def dispatch(predicted):
            return model.hold_teacher(predicted)
    else:
        deps = classic_dependencies()
        name = "damping" if controller.named == "damping" else "pi-external-u"
        cand = TimelineCandidate(model, name, int(np.ceil(horizon_s)) + 2, deps)

        def dispatch(predicted):
            return cand(predicted)
    rollout = model.rollout(dict(bstate), n_steps=steps, cav_teacher=None,
                            cav_controller=dispatch, **bhist, **drivers)
    weights = RewardWeights()
    sums = {key: torch.zeros(batch, dtype=torch.float64) for key in METRIC_MAP}
    current = dict(bstate)
    for step in range(steps):
        accel = rollout["accel"][:, step]
        nxt = {**current, "s_m": rollout["positions"][:, step], "v_mps": rollout["speeds"][:, step],
               "lane": rollout["lanes"][:, step], "present": rollout["masks"][:, step],
               "a_mps2": accel}
        nxt.pop("veh_kin", None)
        terms = reward_fn(current, nxt, accel, weights)
        for key in METRIC_MAP:
            sums[key] += terms[key].double().cpu()
        current = nxt
    per = {METRIC_MAP[k]: v.reshape(n_arm, n) for k, v in sums.items()}
    metrics = {arm: {m: float(per[m][i].mean()) for m in per} for i, arm in enumerate(names)}
    b = names.index("B")
    d = {}
    for i, arm in enumerate(names):
        d[arm] = {m: float((SIGN[m] * (per[m][i] - per[m][b])).mean()) for m in per}
    return {"metrics": metrics, "D": d, "n_samples": n, "slots": int(keep.numel()),
            "wall_s": time.monotonic() - started, "checkpoint": str(ckpt)}
