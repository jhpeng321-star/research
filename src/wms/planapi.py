"""WM-S planning API for WM-C's proposer on wz3to2 (WM-A, 2026-11-06): load / observe / evaluate.

Contract (``WMC_READY.json`` -> ``WMA_READY.json["api"]``)::

    handle = load(checkpoint=None, map_spec=None, device="cpu", config=DEFAULT_CONFIG, ...)
    state  = observe(handle, moment)
    scores = evaluate(handle, state, actions, K=8, H=300.0, seed=0)  # {action_id: [J_1..J_K]}

``J_k`` = team delay in veh-s over ``[t0, t0 + H)``: ``sum_t dt * sum_present max(0, 1 - v/31.29)``
of the predicted road vehicles (plus source waiting of the optional entry process; 0 without one).
Draw ``k`` is a deterministic function of ``(state, seed, k)`` and is SHARED by every action of
the call (common random numbers): the lane-sampling uniforms of the WM rollout are drawn once per
``k`` from ``sha256(seed, k, state digest)`` and reused for every action row.

``observe``: RSU-observable content only.  ``moment["cavs"]`` = CAV self-states
``{id, edge, lane, x | pos, v[, length]}``; optional ``moment["vehicles"]`` = CAV-detected
neighbours (N1 'cav_reports' rows) in the same format with ``is_cav``; optional
``moment["counter"]`` ``[2, 30]`` = RSU entrance counts (10 s bins, newest first; entrance 0 =
through lanes, 1 = closing lane) -> Poisson entries.  Rows are projected into the WM frame with
the bound road (:meth:`wms.core.road.RoadGeometry.project`).  NO world labels (cell, demand,
penetration, seed) are read.

``actions``: WM-C rows ``{id, kind, members: [{id, role:'hold', params:{v_w, hold_s, dmax, amax,
release_x[, start_s, anchor, anchor_release_x]}}], ...}``.  Members follow the ``hold_speed`` law
``a = clip((v_w - v) / 1 s, -dmax, amax)`` from ``start_s`` until ``start_s + hold_s``, until the
member passes ``release_x`` (global x) or the ``anchor`` vehicle passes ``anchor_release_x``; before
and after they move with the WM's own (native) acceleration.  Member lanes are held, except that a
member on the closing lane inside the lane-end zone merges by the WM's first-passage law
(``WM_MERGE_KEY`` delegation).  Non-member CAVs are modelled as native drivers (WM HDV law, as in
``wms.control.coop_imagine``).  A member id absent from the state is refused.

Fresh (``checkpoint=None``) handles are for dry runs only: ``handle.meta["trained"] is False``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from wms.core.road import RoadGeometry, bind_road, current_road, load_road

DEFAULT_CONFIG = "configs/c4_wm_loop14_r16_ft.toml"
SPEED_LIMIT_MPS = 31.29
DT_S = 0.5
NOMINAL_LENGTH_M = 5.0


class WMContractError(ValueError):
    """Input outside the planning contract (fail closed)."""


@dataclass
class Handle:
    model: Any
    road: RoadGeometry
    device: str = "cpu"
    meta: dict = field(default_factory=dict)
    max_rows: int = 64


def _sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(checkpoint: Optional[str] = None, *, map_spec: Optional[str] = None, device: str = "cpu",
         config: str = DEFAULT_CONFIG, max_rows: int = 64, seed: int = 101) -> Handle:
    """WM handle on the process road (``map_spec`` binds it; else WMS_MAP_SPEC/COOPMWM_MAP_SPEC).

    ``checkpoint``: a ``wm_s/1`` checkpoint (with its run's physics.json) -> the trained WM;
    ``None`` -> a fresh model from ``config`` (seeded; DRY RUNS ONLY)."""
    road = bind_road(load_road(map_spec)) if map_spec else current_road()
    meta = {"road": road.name, "road_digest": road.digest, "device": device}
    if checkpoint:
        from wms.eval.backend import load_backend
        model = load_backend(checkpoint, device=device).model
        meta.update(checkpoint=str(checkpoint), checkpoint_sha256=_sha_file(checkpoint),
                    trained=True)
    else:
        from wms.cli.assemble import build_model, build_physics
        from wms.config.load import load_train_config
        root = Path(__file__).resolve().parents[2]
        cfg_path = Path(config) if Path(config).is_absolute() else root / config
        cfg = load_train_config(cfg_path, [f'runtime.device="{device}"'])
        torch.manual_seed(int(seed))
        model = build_model(cfg, build_physics(cfg)).to(device)
        meta.update(config=str(cfg_path), trained=False, init_seed=int(seed))
    model.eval().requires_grad_(False)
    return Handle(model=model, road=road, device=device, meta=meta, max_rows=int(max_rows))


# ---------------------------------------------------------------------------------- observe
def _row(road: RoadGeometry, r: Mapping[str, Any], is_cav: bool) -> dict:
    lane, s = road.project(r["edge"], int(r["lane"]), x=r.get("x"), pos=r.get("pos"))
    v = float(r["v"])
    if not (math.isfinite(s) and math.isfinite(v)):
        raise WMContractError(f"non-finite observation for {r.get('id')}")
    return {"id": str(r["id"]), "s": float(s), "lane": int(lane), "v": v,
            "length": float(r.get("length", NOMINAL_LENGTH_M)), "is_cav": bool(is_cav)}


def observe(handle: Handle, moment: Mapping[str, Any]) -> dict:
    """WM-frame state of a moment from RSU-observable rows only (see the module docstring)."""
    road = handle.road
    if moment.get("map", road.name) != road.name:
        raise WMContractError(f"moment map {moment.get('map')!r} != WM road {road.name!r}")
    rows = {}
    for r in moment.get("cavs", ()):
        rows[str(r["id"])] = _row(road, r, True)
    for r in moment.get("vehicles", ()):
        if str(r["id"]) not in rows:
            rows[str(r["id"])] = _row(road, r, bool(r.get("is_cav", False)))
    if not rows:
        raise WMContractError("moment has no observed vehicle")
    vehicles = sorted(rows.values(), key=lambda x: (-x["s"], x["id"]))
    state = {"t0": float(moment.get("t0", 0.0)), "map": road.name, "vehicles": vehicles,
             "observed": "cav_reports" if moment.get("vehicles") else "cav_self"}
    if moment.get("counter") is not None:
        state["counter"] = [[float(c) for c in row] for row in moment["counter"]]
    return state


def state_digest(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------------- evaluate
def _member_specs(actions: Sequence[Mapping[str, Any]], index: dict) -> list[list[dict]]:
    out = []
    for a in actions:
        members = []
        for m in a.get("members", ()) or ():
            if str(m.get("role", "hold")) == "prog":
                members.append(_prog_spec(a, m, index))
                continue
            if str(m.get("role", "hold")) != "hold":
                raise WMContractError(f"unsupported member role {m.get('role')!r}")
            if str(m["id"]) not in index:
                raise WMContractError(f"action {a.get('id')}: member {m['id']} not in the state")
            p = dict(m.get("params", {}))
            for key in ("v_w", "hold_s", "dmax", "amax"):
                if key not in p or not math.isfinite(float(p[key])):
                    raise WMContractError(f"action {a.get('id')}: member param {key} missing")
            anchor = p.get("anchor")
            if anchor is not None and str(anchor) not in index:
                anchor = None                     # unobserved anchor: hold ends by time / release
            members.append({"slot": index[str(m["id"])], "v_w": float(p["v_w"]),
                            "start_s": float(p.get("start_s", 0.0)), "hold_s": float(p["hold_s"]),
                            "dmax": float(p["dmax"]), "amax": float(p["amax"]),
                            "release_x": (float(p["release_x"]) if p.get("release_x") is not None
                                          else math.inf),
                            "anchor": None if anchor is None else index[str(anchor)],
                            "anchor_release_x": (float(p["anchor_release_x"])
                                                 if p.get("anchor_release_x") is not None
                                                 else math.inf)})
        out.append(members)
    return out


class HoldSpeedController:
    """Rollout callback: hold_speed law on member slots, WM native motion otherwise."""

    def __init__(self, members_by_row: list[list[dict]], n_slots: int, anchor_x: float,
                 dt_s: float, device):
        rows = len(members_by_row)
        self.dt = float(dt_s)
        self.anchor_x = float(anchor_x)
        shape = (rows, n_slots)

        def full(value):
            return torch.full(shape, float(value), device=device)
        self.member = torch.zeros(shape, dtype=torch.bool, device=device)
        self.v_w, self.start, self.end = full(0), full(0), full(-1)
        self.dmax, self.amax = full(0), full(0)
        self.release, self.anchor_release = full(math.inf), full(math.inf)
        self.anchor = torch.full(shape, -1, dtype=torch.long, device=device)
        for b, members in enumerate(members_by_row):
            for m in members:
                j = m["slot"]
                self.member[b, j] = True
                self.v_w[b, j], self.dmax[b, j], self.amax[b, j] = m["v_w"], m["dmax"], m["amax"]
                self.start[b, j], self.end[b, j] = m["start_s"], m["start_s"] + m["hold_s"]
                self.release[b, j], self.anchor_release[b, j] = m["release_x"], m["anchor_release_x"]
                if m["anchor"] is not None:
                    self.anchor[b, j] = m["anchor"]
        self.released = torch.zeros(shape, dtype=torch.bool, device=device)

    def __call__(self, state: dict) -> torch.Tensor:
        from wms.model import cav as cav_motion

        s, v, lane = state["s_m"], state["v_mps"], state["lane"].long()
        t = float(state.get("physics_step_index", 0)) * self.dt
        native = state.get("merge_pivot_native_accel")
        if native is None:
            native = torch.zeros_like(v)
        x = s + self.anchor_x
        ax = torch.gather(x, 1, self.anchor.clamp_min(0))
        self.released = self.released | (self.member & (
            (x >= self.release) | ((self.anchor >= 0) & (ax >= self.anchor_release))))
        active = self.member & (t >= self.start) & (t < self.end) & ~self.released
        hold = torch.maximum(torch.minimum((self.v_w - v) / 1.0, self.amax), -self.dmax)
        a = torch.where(active, hold, native.to(v.dtype))
        v_next = (v + a * self.dt).clamp_min(0.0)
        a = (v_next - v) / self.dt
        s_next = s + 0.5 * (v + v_next) * self.dt
        state[cav_motion.WM_MERGE_KEY] = self.member & (lane == 2)
        return torch.stack((s_next, lane.to(v.dtype), v_next, a), dim=-1)


def _prog_spec(action, m, index):
    """LCMOVES 'prog' member spec (see :mod:`wms.planapi_prog`)."""
    from wms.planapi_prog import prog_spec
    return prog_spec(action, m, index)


def _draw_uniform(seed: int, k: int, digest: str, n_steps: int, n_slots: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(hashlib.sha256(
        f"wzapi:{seed}:{k}:{digest}".encode()).hexdigest()[:15], 16))
    return torch.rand((n_steps, n_slots), generator=g)


def _entries(counter, n_rows, seed, device):
    if counter is None:
        return None
    import numpy as np
    from wms.model.entries import TorchEntryProcess
    return TorchEntryProcess(np.asarray(counter, np.float32), seed=int(seed), batch=n_rows)


def evaluate(handle: Handle, state: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], *,
             K: int = 8, H: float = 300.0, seed: int = 0, trace: Optional[dict] = None) -> dict:
    """``{action_id: [J_1..J_K]}`` team delay (veh-s) per CRN draw (see the module docstring)."""
    if state.get("map") != handle.road.name:
        raise WMContractError(f"state map {state.get('map')!r} != WM road {handle.road.name!r}")
    if int(K) < 1 or not (float(H) > 0):
        raise WMContractError("K >= 1 and H > 0 required")
    ids = [str(r["id"]) for r in state["vehicles"]]
    index = {vid: j for j, vid in enumerate(ids)}
    aids = [str(a["id"]) for a in actions]
    if len(set(aids)) != len(aids):
        raise WMContractError("duplicate action ids")
    members = _member_specs(actions, index)
    model, dev = handle.model, torch.device(handle.device)
    n_steps = max(1, int(round(float(H) / DT_S)))
    counter = state.get("counter")
    n_pad = 0 if counter is None else 64
    n = len(ids) + n_pad
    digest = state_digest(state)
    exit_s = handle.road.road_end_m if getattr(model.physics, "exit_at_road_end", True) else None
    out = {aid: [None] * int(K) for aid in aids}
    rows = [(a, k) for k in range(int(K)) for a in range(len(aids))]
    for lo in range(0, len(rows), max(1, handle.max_rows)):
        chunk = rows[lo:lo + handle.max_rows]
        b = len(chunk)

        def col(key, default=0.0):
            vals = [float(r.get(key, default)) for r in state["vehicles"]] + [0.0] * n_pad
            return torch.tensor(vals, dtype=torch.float32, device=dev).expand(b, n).clone()
        present = torch.zeros(b, n, dtype=torch.bool, device=dev)
        present[:, :len(ids)] = True
        ctrl_members = [members[a] for a, _k in chunk]
        is_cav = torch.zeros(b, n, dtype=torch.bool, device=dev)
        for r, mem in enumerate(ctrl_members):
            for m in mem:
                is_cav[r, m["slot"]] = True
        initial = {"s_m": col("s"), "v_mps": col("v"), "a_mps2": torch.zeros(b, n, device=dev),
                   "lane": col("lane").long(), "length_m": col("length", NOMINAL_LENGTH_M),
                   "present": present, "is_cav": is_cav}
        if exit_s is not None:
            initial["exit_s_m"] = torch.full((b, n), float(exit_s), device=dev)
        if any(m.get("role") == "prog" for mem in ctrl_members for m in mem):
            for mem in ctrl_members:
                for m in mem:
                    if m.get("role") == "prog" and not state["vehicles"][m["slot"]]["is_cav"]:
                        raise WMContractError(f"prog member {m['id']} is not an observed CAV")
            from wms.planapi_prog import ProgramController
            controller = ProgramController(
                [[m for m in mem if m.get("role") != "prog"] for mem in ctrl_members],
                [[m for m in mem if m.get("role") == "prog"] for mem in ctrl_members],
                n, handle.road.anchor_x_m or 0.0, DT_S, dev, physics=model.physics,
                road=handle.road)
        else:
            controller = HoldSpeedController(ctrl_members, n, handle.road.anchor_x_m or 0.0,
                                             DT_S, dev)
        uniform = torch.stack([_draw_uniform(seed, k, digest, n_steps, n) for _a, k in chunk]
                              ).to(dev)
        entries = _entries(counter, b, seed, dev)
        hist = {}
        if getattr(model, "long_history", False):
            # No observed pre-t0 driver history in the contract: masked (population settings).
            from wms.model.history import LONG_HISTORY_F
            hist = {"driver_history": torch.zeros(b, n, model.driver_history_len, LONG_HISTORY_F,
                                                  device=dev),
                    "driver_history_mask": torch.zeros(b, n, model.driver_history_len,
                                                       dtype=torch.bool, device=dev)}
        with torch.no_grad():
            roll = model.rollout(initial, n_steps=n_steps, k0=0, cav_controller=controller,
                                 lane_uniform=uniform, entries=entries, **hist)
        speeds = torch.cat((initial["v_mps"][:, None], roll["speeds"][:, :-1]), 1)
        masks = torch.cat((present[:, None], roll["masks"][:, :-1].bool()), 1)
        deficit = (1.0 - speeds / SPEED_LIMIT_MPS).clamp_min(0.0) * masks
        j = (DT_S * deficit.sum((1, 2))).double().cpu().tolist()
        if trace is not None:                     # diagnostics / tests: per-row predicted motion
            trace.setdefault("rows", []).extend(chunk)
            trace.setdefault("speeds", []).append(speeds.cpu())
            trace.setdefault("lanes", []).append(roll["lanes"].cpu())
            trace["ids"] = ids
        for (a, k), value in zip(chunk, j):
            if not math.isfinite(value):
                raise WMContractError(f"non-finite WM delay for action {aids[a]} draw {k}")
            out[aids[a]][k] = float(value)
    return out


def describe() -> dict:
    """Machine-readable contract (WMA_READY.json ``api``)."""
    return {"load": "wms.planapi:load", "observe": "wms.planapi:observe",
            "evaluate": "wms.planapi:evaluate", "supported_roles": ["hold", "prog"],
            "evaluate_signature": "evaluate(handle, state, actions, K=8, H=300.0, seed=0) -> "
                                  "{action_id: [J_1..J_K]}"}
