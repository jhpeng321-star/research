"""WM-S backend for the qualification suite (wms lineage of c4/wz-wm-port, road bound to wz3to2).

Rolls a ``wm_s`` checkpoint forward from a :class:`~tools.wzqual.traj.StartState` built from a
WMQUAL tape, for named arms in ONE batch (rows arm-major / sample-minor):

* observable inputs only: the t0 frame, the pre-t0 native history (``config.history_s``, by id,
  incl. vehicles that left before t0) and the t0 entrance counter [2, 30] (entries v2: observed
  rate + observed entry history; no demand label, no recorded future, no driver truth);
* every non-member CAV is a native driver (WM HDV slot with encoder-identified settings), as in
  ``wms.control.coop_imagine``; arm members are CAV slots driven CLOSED LOOP by
  :class:`~tools.wzqual.moves.MoveExecutor` on the predicted state (accel commands held for the
  1 s control interval; lane commands applied as a lane target; released / not-yet-started
  members and any entrant CAV follow a textbook-CAV IDM in their lane; closing-lane members hand
  their merge to the WM merge law, ``cav.WM_MERGE_KEY``);
* sample k of every arm shares: driver-setting draws (encoder uncertainty x ``driver_noise``,
  ``wms.control.futures.sample_driver_encodings``), lane uniforms (if the WM samples lanes) and
  the entry stream (``TorchEntryProcessV2`` row seeds + coupled draws) - common random numbers.

``tau`` roles (CACC headway) are not executable here (the WM has no per-slot headway command);
those candidates are excluded and counted by run_qual (C0 coverage).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

from . import frame as F
from .moves import MoveExecutor
from .traj import StartState, Traj

CAV_IDM = dict(a=1.2, b=2.0, T=1.2, s0=2.0, v0=F.SPEED_LIMIT, delta=4.0, length=5.0)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RuleController:
    """``cav_controller`` callback: per batch row an executor; returns [B, K, 4] (s, lane, v, a).

    ``slot_ids[b]`` maps row b's t0 slots to vehicle ids (rows of different starts differ).
    The state is copied to the CPU once per call (one transfer); commands are evaluated in numpy.
    """

    def __init__(self, executors: list, slot_ids: list, dt: float):
        self.ex = executors
        self.slot_ids = slot_ids
        self.slot_of = [{vid: k for k, vid in enumerate(ids)} for ids in slot_ids]
        self.dt = float(dt)
        b = len(executors)
        self.accel = [dict() for _ in range(b)]
        self.lane_target = [dict() for _ in range(b)]
        self.released = [set() for _ in range(b)]
        self.watch = [set(e.ids) | {m["params"].get(k) for m in e.members for k in ("anchor", "partner")}
                      for e in executors]
        self.active = [bool(e.members) for e in executors]
        self.calls = 0

    @staticmethod
    def _idm(s, v, lane, present, k):
        """Textbook-CAV IDM acceleration of slot k (numpy row), leader in the same lane;
        a closing-lane vehicle also sees the lane end as a standing obstacle."""
        same = present & (lane == lane[k]) & (s > s[k])
        gap, dv = 1e4, 0.0
        if same.any():
            j = int(np.argmin(np.where(same, s, 1e9)))
            gap, dv = float(s[j] - s[k] - CAV_IDM["length"]), float(v[k] - v[j])
        if int(lane[k]) == F.CLOSING_LANE and s[k] < F.DROP_S and F.DROP_S - s[k] < gap:
            gap, dv = float(F.DROP_S - s[k]), float(v[k])
        p = CAV_IDM
        vk = float(v[k])
        star = p["s0"] + max(0.0, vk * p["T"] + vk * dv / (2.0 * (p["a"] * p["b"]) ** 0.5))
        a = p["a"] * (1.0 - (vk / p["v0"]) ** p["delta"] - (star / max(gap, 0.1)) ** 2)
        return min(max(a, -9.0), p["a"])

    def __call__(self, state: dict) -> torch.Tensor:
        from wms.model import cav as cav_motion

        dev = state["s_m"].device
        step = state.get("physics_step_index", self.calls)
        step = int(step.flatten()[0]) if torch.is_tensor(step) else int(step)
        self.calls += 1
        s_t, v_t, lane_t = state["s_m"], state["v_mps"], state["lane"]
        out = torch.stack((s_t, lane_t.to(s_t.dtype), v_t, torch.zeros_like(s_t)), -1)
        cav_t = state["is_cav"].bool() & state["present"].bool()
        rows = {b for b in range(len(self.ex)) if self.active[b]}
        rows |= set(torch.nonzero(cav_t.any(1)).flatten().tolist())      # entrant CAVs, if any
        if not rows:
            state[cav_motion.WM_MERGE_KEY] = torch.zeros_like(cav_t)
            return out
        s, v = s_t.detach().cpu().numpy(), v_t.detach().cpu().numpy()
        lane = lane_t.detach().cpu().numpy()
        present = state["present"].bool().cpu().numpy()
        cav = cav_t.cpu().numpy()
        o = out.detach().cpu().numpy().copy()
        mg = np.zeros(present.shape, bool)
        for b in sorted(rows):
            slot_of = self.slot_of[b]
            if step % 2 == 0 and self.active[b]:
                view = {}
                for vid in self.watch[b]:
                    k = slot_of.get(vid)
                    if k is not None and present[b, k]:
                        view[vid] = (float(s[b, k]), int(lane[b, k]), float(v[b, k]))
                for vid, cmds in self.ex[b].control(step // 2, view).items():
                    for kind, val in cmds:
                        if kind == "accel":
                            self.accel[b][vid] = float(val)
                        elif kind == "lane":
                            self.lane_target[b][vid] = int(val)
                        elif kind == "release":
                            self.released[b].add(vid)
                            self.accel[b].pop(vid, None)
                            self.lane_target[b].pop(vid, None)
            ids = self.slot_ids[b]
            for k in np.flatnonzero(cav[b]).tolist():
                vid = ids[k] if k < len(ids) else None
                held = vid is not None and vid in self.accel[b] and vid not in self.released[b]
                a = self.accel[b][vid] if held else self._idm(s[b], v[b], lane[b], present[b], k)
                nv = min(max(float(v[b, k]) + a * self.dt, 0.0), F.SPEED_LIMIT)
                ns = float(s[b, k]) + 0.5 * (float(v[b, k]) + nv) * self.dt
                ln = int(lane[b, k])
                if vid is not None and vid in self.lane_target[b]:
                    ln = self.lane_target[b].pop(vid)
                if ln == F.CLOSING_LANE:
                    mg[b, k] = True
                    if ns > F.DROP_S:                      # SUMO: wait at the lane end
                        ns, nv = F.DROP_S, 0.0
                o[b, k] = (ns, float(ln), nv, (nv - float(v[b, k])) / self.dt)
        state[cav_motion.WM_MERGE_KEY] = torch.as_tensor(mg, device=dev)
        return torch.as_tensor(o, dtype=out.dtype, device=dev)


class WmBackend:
    name = "wm"
    stochastic = True
    models_entries = True
    closed_loop = True
    supported_roles = frozenset({"hold", "pulse", "lane_to", "merge_early", "zip_merge", "zip_yield"})

    def __init__(self, checkpoint: str, *, device: str = "cpu", map_spec: Optional[str] = None,
                 physics_toml: Optional[str] = None, driver_noise: float = 1.0,
                 embedding_noise: float = 0.05):
        if map_spec:
            os.environ["WMS_MAP_SPEC"] = str(map_spec)
        from wms.core.road import current_road
        from wms.eval.backend import load_backend
        from wms.eval.history_inputs import HistoryInputs

        self.road = current_road()
        if self.road.name != "wz3to2":
            raise RuntimeError(f"road bound to {self.road.name!r}; set WMS_MAP_SPEC to the wz3to2 MAP_SPEC")
        physics = model_config = None
        self.physics_source = "checkpoint (embedded / physics.json / TRAIN_RECEIPT.json)"
        if physics_toml:
            # Baseline (WM-E 2026-11-06): physics = build_physics(load_train_config(TOML)) and the
            # TOML's flat config as model_config, so baseline and fine-tune differ only in weights.
            from wms.cli.assemble import build_physics
            from wms.config.load import load_train_config
            cfg = load_train_config(physics_toml)
            physics = asdict(build_physics(cfg))
            model_config = cfg.to_legacy_flat()
            self.physics_source = f"{physics_toml} sha256 {_sha(physics_toml)} (physics + flat model config)"
        self.be = load_backend(checkpoint, device=device, physics=physics, model_config=model_config)
        self.model = self.be.model
        self.device = device
        self.hist = HistoryInputs(self.be)
        self.history_s, self.history_stride_s = self.hist.long_history_config()
        self.driver_noise, self.embedding_noise = float(driver_noise), float(embedding_noise)
        self.checkpoint_sha256 = _sha(checkpoint)
        try:
            self.physics = asdict(self.model.physics)
        except TypeError:
            self.physics = {}
        import json as _json
        self.physics_sha256 = hashlib.sha256(_json.dumps(self.physics, sort_keys=True, default=str)
                                             .encode()).hexdigest()

    # ------------------------------------------------------------ observable state record
    def state_record(self, start: StartState) -> dict:
        fr = start.frame
        vehicles = [dict(id=n, s=float(s), lane=int(ln), v=float(v), present=True, is_cav=bool(c))
                    for n, s, ln, v, c in zip(fr["ids"], fr["s"], fr["lane"], fr["v"], fr["is_cav"])]
        rec = dict(t0_s=start.t0, state=dict(vehicles=vehicles), counter=start.counter().tolist())
        hs = float(self.history_s or 60.0)
        h = start.history.window(max(float(start.history.time_s[0]), start.t0 - hs), start.t0)
        cols = sorted(set(h.vid.tolist()))
        col_of = {v: j for j, v in enumerate(cols)}
        t = h.n_steps
        kin = np.zeros((t, len(cols), 7), np.float32)
        mask = np.zeros((t, len(cols)), bool)
        for i in range(len(h.step)):
            j = col_of[int(h.vid[i])]
            k = int(h.step[i])
            kin[k, j, 0] = h.s[i]
            kin[k, j, 1 + int(np.clip(h.lane[i], 0, 2))] = 1.0
            kin[k, j, 4] = h.v[i]
            kin[k, j, 6] = 5.0
            mask[k, j] = True
        kin[1:, :, 5] = np.where(mask[1:] & mask[:-1], (kin[1:, :, 4] - kin[:-1, :, 4]) / F.DT, 0.0)
        rec["history"] = dict(ids=[h.ids[v] for v in cols], kin=kin, mask=mask,
                              target_mask=(mask[:-1] & mask[1:]), history_s=hs,
                              history_stride_s=float(self.history_stride_s or 1.0), dt_s=F.DT,
                              available_span_s=(t - 1) * F.DT,
                              run_start_limited=bool(h.time_s[0] > start.t0 - hs + 1e-6))
        return rec

    # ------------------------------------------------------------------ rollout
    def rollout(self, start: StartState, arms: Mapping[str, list], *, horizon_s: float,
                n_samples: int = 8, seed: int = 0) -> dict:
        return self.rollout_batch([(start, arms, seed)], horizon_s=horizon_s, n_samples=n_samples)[0]

    @torch.no_grad()
    def rollout_batch(self, jobs: list, *, horizon_s: float, n_samples: int = 8) -> list:
        """Several starts in ONE batched rollout.  ``jobs = [(start, arms, seed), ...]``; rows are
        start-major, then arm, then sample.  Row layout per start: its t0 vehicles in slots
        [0, n_i), never-present slots up to N = max n_i, then the shared entry pool [N, K)."""
        from wms.control.futures import sample_driver_encodings
        from wms.eval import entries_input as EI
        from wms.eval.rollout_inputs import t0_vehicles
        from wms.physics.batched import road_end_exit_bound

        reset = getattr(self.model, "reset_diag_state", None)
        if reset is not None:
            reset()
        dev = torch.device(self.device)
        n = max(1, int(n_samples))
        steps = int(round(float(horizon_s) / F.DT))
        tools = self.be.tools
        mod = EI._entries(tools)
        per = []
        for start, arms, seed in jobs:
            rec = self.state_record(start)
            t0_s, ids, kin, _ = t0_vehicles(rec)
            dh, dhm = self.hist.build(rec, ids)
            history = EI._entry_history(rec, t0_s, kin, tools)
            per.append(dict(start=start, arms=arms, seed=int(seed), rec=rec, ids=ids, kin=kin,
                            dh=dh, dhm=dhm, history=history,
                            pad=EI._pad_size(history.rate.sum(), horizon_s)))
        N = max(len(p["ids"]) for p in per)
        n_pad = max(p["pad"] for p in per)
        K = N + n_pad
        S = len(per)
        st = dict(s_m=torch.zeros(S, K), v_mps=torch.zeros(S, K), a_mps2=torch.zeros(S, K),
                  lane=torch.zeros(S, K, dtype=torch.long), length_m=torch.full((S, K), 5.0),
                  present=torch.zeros(S, K, dtype=torch.bool), is_cav=torch.zeros(S, K, dtype=torch.bool))
        dh_all = dhm_all = None
        for i, p in enumerate(per):
            m = len(p["ids"])
            st["s_m"][i, :m] = torch.as_tensor(p["kin"][:, 0])
            st["v_mps"][i, :m] = torch.as_tensor(p["kin"][:, 4])
            st["lane"][i, :m] = torch.as_tensor(p["kin"][:, 1:4].argmax(-1))
            st["present"][i, :m] = True
            if p["dh"] is not None:
                if dh_all is None:
                    dh_all = torch.zeros(S, K, *p["dh"].shape[2:], dtype=p["dh"].dtype)
                    dhm_all = torch.zeros(S, K, p["dhm"].shape[2], dtype=torch.bool)
                dh_all[i, :m] = p["dh"][0]
                dhm_all[i, :m] = p["dhm"][0]
        exit_s = road_end_exit_bound(getattr(self.model, "physics", None))
        if exit_s is not None:
            st["exit_s_m"] = torch.full((S, K), float(exit_s))
        st = {k: v.to(dev) for k, v in st.items()}
        hist_args = {} if dh_all is None else dict(driver_history=dh_all.to(dev),
                                                   driver_history_mask=dhm_all.to(dev))
        gen = torch.Generator().manual_seed(int(per[0]["seed"]) % (2 ** 63))
        sampled = sample_driver_encodings(
            self.model, st, hist_args, n,
            lambda shape, reference, key: torch.randn(tuple(shape), generator=gen).to(reference),
            driver_noise_scale=self.driver_noise, embedding_noise_std=self.embedding_noise)
        row_start, row_arm, row_k, execs, slot_ids = [], [], [], [], []
        for i, p in enumerate(per):
            for name in p["arms"]:
                for k in range(n):
                    row_start.append(i)
                    row_arm.append(name)
                    row_k.append(k)
                    execs.append(MoveExecutor(p["arms"][name], t0=p["start"].t0))
                    slot_ids.append(p["ids"])
        B = len(row_start)
        rs = torch.as_tensor(row_start, device=dev)
        rk = torch.as_tensor(row_k, device=dev)
        state = {k: v[rs] for k, v in st.items()}
        member = torch.zeros(B, K, dtype=torch.bool)
        for r in range(B):
            mids = {m["id"] for m in per[row_start[r]]["arms"][row_arm[r]]}
            for j, vid in enumerate(per[row_start[r]]["ids"]):
                if vid in mids:
                    member[r, j] = True
        state["is_cav"] = member.to(dev)
        drivers = {}
        if "driver_params" in sampled:              # [n, S, K] -> row (start, sample)
            drivers["driver_params"] = {q: v[rk, rs] for q, v in sampled["driver_params"].items()}
        if "driver_embedding" in sampled:
            drivers["driver_embedding"] = sampled["driver_embedding"][rk, rs]
        lanes = {}
        if getattr(self.model, "_samples_lanes", False):
            draws = torch.stack([torch.rand(n, steps, K, generator=torch.Generator().manual_seed(
                int(hashlib.sha256(f"wzqual-lanes:{p['seed']}".encode()).hexdigest()[:12], 16)))
                for p in per])                              # [S, n, T, K]
            lanes["lane_uniform"] = draws[torch.as_tensor(row_start), torch.as_tensor(row_k)].to(dev)
        cfg_kwargs = {}
        rule = self.be.config.get("entry_insert_rule")
        if rule and "insert_rule" in mod.EntryConfig.__dataclass_fields__:
            cfg_kwargs["insert_rule"] = str(rule)
        row_seeds = [int(hashlib.sha256(f"wzqual-entry:{per[i]['seed']}:{k}".encode()).hexdigest()[:12], 16)
                     for i, k in zip(row_start, row_k)]
        proc = mod.TorchEntryProcessV2(histories=[per[i]["history"] for i in row_start], dt_s=F.DT,
                                       seed=int(per[0]["seed"]), batch=B, cfg=mod.EntryConfig(**cfg_kwargs),
                                       pool=EI._pool(N, n_pad), row_seeds=row_seeds, couple_draws=True)
        ctrl = RuleController(execs, slot_ids, F.DT)
        bh = {} if not hist_args else {k: v[rs] for k, v in hist_args.items()}
        roll = self.model.rollout(state, n_steps=steps, k0=0, cav_controller=ctrl, entries=proc,
                                  projection=self.be.projection, controller_id=None, **bh, **drivers,
                                  **lanes)
        pos, spd = roll["positions"].cpu().numpy(), roll["speeds"].cpu().numpy()
        lan, msk = roll["lanes"].cpu().numpy(), roll["masks"].cpu().numpy().astype(bool)
        s0, v0 = state["s_m"].cpu().numpy(), state["v_mps"].cpu().numpy()
        l0, p0 = state["lane"].cpu().numpy(), state["present"].cpu().numpy()
        results = [{name: [] for name in p["arms"]} for p in per]
        for r in range(B):
            p = per[row_start[r]]
            names = (list(p["ids"]) + [f"__void_{j}" for j in range(N - len(p["ids"]))]
                     + [f"__entry_{j}" for j in range(n_pad)])
            cav_slot = np.zeros(K, bool)
            cav_slot[:len(p["ids"])] = rec_cav(p["rec"], p["ids"])
            Sx = np.concatenate([s0[r][None], pos[r]], 0)
            Vx = np.concatenate([v0[r][None], spd[r]], 0)
            Lx = np.concatenate([l0[r][None], lan[r]], 0)
            Px = np.concatenate([p0[r][None], msk[r]], 0)
            ki, vi = np.nonzero(Px)
            times = p["start"].t0 + F.DT * np.arange(steps + 1)
            results[row_start[r]][row_arm[r]].append(
                Traj(times, ki, vi, Sx[ki, vi], Lx[ki, vi], Vx[ki, vi], names, cav_slot, None,
                     dict(backend=self.name, arm=row_arm[r], sample=row_k[r])))
        return results


def rec_cav(rec: dict, ids: list) -> list:
    flags = {v["id"]: v.get("is_cav", False) for v in rec["state"]["vehicles"]}
    return [flags.get(i, False) for i in ids]
