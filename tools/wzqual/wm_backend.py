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
    """``cav_controller`` callback: per batch row an executor; returns [B, K, 4] (s, lane, v, a)."""

    def __init__(self, executors: list, slot_ids: list, dt: float, n_t0: int):
        self.ex = executors
        self.slot_ids = slot_ids                 # slot -> vehicle id (t0 slots only)
        self.slot_of = {vid: k for k, vid in enumerate(slot_ids)}
        self.dt = float(dt)
        self.n_t0 = n_t0
        b = len(executors)
        self.accel = [dict() for _ in range(b)]  # held accel per member id
        self.lane_target = [dict() for _ in range(b)]
        self.released = [set() for _ in range(b)]
        self.watch = [set(e.ids) | {m["params"].get(k) for m in e.members for k in ("anchor", "partner")}
                      for e in executors]
        self.calls = 0

    def _idm(self, s, v, lane, present, k):
        """Textbook-CAV IDM acceleration of slot k (row tensors), leader in the same lane;
        a closing-lane vehicle also sees the lane end as a standing obstacle."""
        same = present & (lane == lane[k]) & (s > s[k])
        gap, dv = torch.tensor(1e4), torch.tensor(0.0)
        if bool(same.any()):
            j = torch.argmin(torch.where(same, s, torch.full_like(s, 1e9)))
            gap = s[j] - s[k] - CAV_IDM["length"]
            dv = v[k] - v[j]
        if int(lane[k]) == F.CLOSING_LANE and float(s[k]) < F.DROP_S:
            end_gap = torch.tensor(F.DROP_S - float(s[k]))
            if end_gap < gap:
                gap, dv = end_gap, v[k]
        p = CAV_IDM
        star = p["s0"] + torch.clamp(v[k] * p["T"] + v[k] * dv / (2.0 * (p["a"] * p["b"]) ** 0.5), min=0.0)
        a = p["a"] * (1.0 - (v[k] / p["v0"]) ** p["delta"] - (star / torch.clamp(gap, min=0.1)) ** 2)
        return float(torch.clamp(a, min=-9.0, max=p["a"]))

    def __call__(self, state: dict) -> torch.Tensor:
        from wms.model import cav as cav_motion

        s, v, lane = state["s_m"], state["v_mps"], state["lane"]
        present = state["present"].bool()
        cav = state["is_cav"].bool() & present
        step = state.get("physics_step_index", self.calls)
        step = int(step.flatten()[0]) if torch.is_tensor(step) else int(step)
        self.calls += 1
        out = torch.stack((s, lane.to(s.dtype), v, torch.zeros_like(s)), -1).clone()
        merge = torch.zeros_like(present)
        for b, ex in enumerate(self.ex):
            if step % 2 == 0:
                view = {}
                for vid in self.watch[b]:
                    k = self.slot_of.get(vid)
                    if k is not None and bool(present[b, k]):
                        view[vid] = (float(s[b, k]), int(lane[b, k]), float(v[b, k]))
                for vid, cmds in ex.control(step // 2, view).items():
                    for kind, val in cmds:
                        if kind == "accel":
                            self.accel[b][vid] = float(val)
                        elif kind == "lane":
                            self.lane_target[b][vid] = int(val)
                        elif kind == "release":
                            self.released[b].add(vid)
                            self.accel[b].pop(vid, None)
                            self.lane_target[b].pop(vid, None)
            for k in torch.nonzero(cav[b]).flatten().tolist():
                vid = self.slot_ids[k] if k < self.n_t0 else None
                held = vid is not None and vid in self.accel[b] and vid not in self.released[b]
                a = self.accel[b][vid] if held else self._idm(s[b], v[b], lane[b], present[b], k)
                nv = min(max(float(v[b, k]) + a * self.dt, 0.0), F.SPEED_LIMIT)
                ns = float(s[b, k]) + 0.5 * (float(v[b, k]) + nv) * self.dt
                ln = int(lane[b, k])
                if vid is not None and vid in self.lane_target[b]:
                    ln = self.lane_target[b].pop(vid)
                if ln == F.CLOSING_LANE:
                    merge[b, k] = True
                    if ns > F.DROP_S:                      # SUMO: wait at the lane end
                        ns, nv = F.DROP_S, 0.0
                out[b, k] = torch.tensor([ns, float(ln), nv, (nv - float(v[b, k])) / self.dt], dtype=out.dtype)
        state[cav_motion.WM_MERGE_KEY] = merge
        return out


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
    @torch.no_grad()
    def rollout(self, start: StartState, arms: Mapping[str, list], *, horizon_s: float,
                n_samples: int = 8, seed: int = 0) -> dict:
        from wms.control.futures import sample_driver_encodings
        from wms.eval import entries_input as EI
        from wms.eval.rollout_inputs import t0_vehicles
        from wms.physics.batched import road_end_exit_bound

        reset = getattr(self.model, "reset_diag_state", None)
        if reset is not None:
            reset()
        rec = self.state_record(start)
        t0_s, ids, kin, _cav_obs = t0_vehicles(rec)
        n_t0 = len(ids)
        names = list(arms)
        n_arm, n = len(names), max(1, int(n_samples))
        B = n_arm * n
        steps = int(round(float(horizon_s) / F.DT))
        dh, dhm = self.hist.build(rec, ids)
        # entries v2 (observable counter + observed entry history), one row stream per sample
        tools = self.be.tools
        mod = EI._entries(tools)
        history = EI._entry_history(rec, t0_s, kin, tools)
        n_pad = EI._pad_size(history.rate.sum(), horizon_s)
        cfg_kwargs = {}
        rule = self.be.config.get("entry_insert_rule")
        if rule and "insert_rule" in mod.EntryConfig.__dataclass_fields__:
            cfg_kwargs["insert_rule"] = str(rule)
        row_seeds = [int(hashlib.sha256(f"wzqual-entry:{seed}:{k}".encode()).hexdigest()[:12], 16)
                     for _a in range(n_arm) for k in range(n)]
        proc = mod.TorchEntryProcessV2(history=history, dt_s=F.DT, seed=int(seed), batch=B,
                                       cfg=mod.EntryConfig(**cfg_kwargs), pool=EI._pool(n_t0, n_pad),
                                       row_seeds=row_seeds, couple_draws=True)
        K = n_t0 + n_pad
        one = dict(s_m=torch.zeros(1, K), v_mps=torch.zeros(1, K), a_mps2=torch.zeros(1, K),
                   lane=torch.zeros(1, K, dtype=torch.long), length_m=torch.full((1, K), 5.0),
                   present=torch.zeros(1, K, dtype=torch.bool), is_cav=torch.zeros(1, K, dtype=torch.bool))
        one["s_m"][0, :n_t0] = torch.as_tensor(kin[:, 0])
        one["v_mps"][0, :n_t0] = torch.as_tensor(kin[:, 4])
        one["lane"][0, :n_t0] = torch.as_tensor(kin[:, 1:4].argmax(-1))
        one["present"][0, :n_t0] = True
        exit_s = road_end_exit_bound(getattr(self.model, "physics", None))
        if exit_s is not None:
            one["exit_s_m"] = torch.full((1, K), float(exit_s))
        if dh is not None:
            dh = torch.cat((dh, torch.zeros(1, n_pad, *dh.shape[2:], dtype=dh.dtype)), dim=1)
            dhm = torch.cat((dhm, torch.zeros(1, n_pad, dhm.shape[2], dtype=torch.bool)), dim=1)
        gen = torch.Generator().manual_seed(int(seed) % (2 ** 63))
        hist_args = {} if dh is None else dict(driver_history=dh, driver_history_mask=dhm)
        drivers = {}
        sampled = sample_driver_encodings(
            self.model, one, hist_args, n,
            lambda shape, reference, key: torch.randn(tuple(shape), generator=gen).to(reference),
            driver_noise_scale=self.driver_noise, embedding_noise_std=self.embedding_noise)
        if "driver_params" in sampled:          # [n, 1, K] -> rows arm-major / sample-minor
            drivers["driver_params"] = {p: v[:, 0].repeat(n_arm, 1) for p, v in sampled["driver_params"].items()}
        if "driver_embedding" in sampled:
            drivers["driver_embedding"] = sampled["driver_embedding"][:, 0].repeat(n_arm, 1, 1)
        lanes = {}
        if getattr(self.model, "_samples_lanes", False):
            draws = torch.rand(n, steps, K, generator=gen)
            lanes["lane_uniform"] = draws.repeat(n_arm, 1, 1)
        state = {key: val.repeat(B, *([1] * (val.ndim - 1))) for key, val in one.items()}
        member = torch.zeros(B, K, dtype=torch.bool)
        execs = []
        for a, name in enumerate(names):
            mids = {m["id"] for m in arms[name]}
            for k in range(n):
                row = a * n + k
                execs.append(MoveExecutor(arms[name], t0=start.t0))
                for j, vid in enumerate(ids):
                    if vid in mids:
                        member[row, j] = True
        state["is_cav"] = member
        ctrl = RuleController(execs, ids, F.DT, n_t0)
        bh = {} if dh is None else dict(driver_history=dh.repeat(B, 1, 1, 1),
                                        driver_history_mask=dhm.repeat(B, 1, 1))
        roll = self.model.rollout(state, n_steps=steps, k0=0, cav_controller=ctrl, entries=proc,
                                  projection=self.be.projection, controller_id=None, **bh, **drivers,
                                  **lanes)
        pos, spd = roll["positions"].cpu().numpy(), roll["speeds"].cpu().numpy()
        lan, msk = roll["lanes"].cpu().numpy(), roll["masks"].cpu().numpy().astype(bool)
        slot_names = list(ids) + [f"__entry_{j}" for j in range(n_pad)]
        is_cav_slot = np.zeros(K, bool)
        is_cav_slot[:n_t0] = np.asarray([bool(x) for x in rec_cav(rec, ids)])
        out: dict = {name: [] for name in names}
        times = start.t0 + F.DT * np.arange(steps + 1)
        for a, name in enumerate(names):
            for k in range(n):
                r = a * n + k
                S = np.concatenate([one["s_m"][0].numpy()[None], pos[r]], 0)
                V = np.concatenate([one["v_mps"][0].numpy()[None], spd[r]], 0)
                L = np.concatenate([one["lane"][0].numpy()[None], lan[r]], 0)
                P = np.concatenate([one["present"][0].numpy()[None], msk[r]], 0)
                ki, vi = np.nonzero(P)
                out[name].append(Traj(times, ki, vi, S[ki, vi], L[ki, vi], V[ki, vi], slot_names,
                                      is_cav_slot, None, dict(backend=self.name, arm=name, sample=k)))
        return out


def rec_cav(rec: dict, ids: list) -> list:
    flags = {v["id"]: v.get("is_cav", False) for v in rec["state"]["vehicles"]}
    return [flags.get(i, False) for i in ids]
