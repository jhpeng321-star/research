"""S-RB-CF teacher-forced tape check: WM2 vs the pure SUMO-IDM physics arm.

For the 32 recorded WMDATA development FIT worlds 969400-969431 (native arm;
cpuQ observable views fetched read-only) this builds, per world, the exact
``wms_eval_states/1`` record the predictor consumes (id-keyed 121-row pre-t0
history with the wallregen ``_align_history`` semantics, t0 = 300 s, the
recorded native future as the CAV teacher) and rolls three arms for 60 s with
entries OFF:

  wm2          - the checkpoint as-is (SHA-verified b8797242...);
  sumo_enc     - ``physics.cf_model='sumo_idm'`` + every learned acceleration
                 correction OFF (``RolloutFlags.pure_physics``), encoder-inferred
                 per-driver settings;
  sumo_pop     - the same pure SUMO arm with population-default driver settings
                 (``diag_encoder_population``).

Reported per arm: HDV speed MAE / position MAE at 10/30/60 s and the same-lane
bumper-gap MAE of the HDV fleet inside each frame (WM frame vs recorded frame),
identity-matched on the t0 roster.  CPU only; no SUMO, no training, no
evaluation seeds (969432-969439 are never opened).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

DT_S = 0.5
HISTORY_ROWS = 121
HISTORY_S = 60.0
HORIZON_S = 60.0
MERGE_END_M = 75.44
WM2_SHA = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
DEV_SEEDS = tuple(range(969400, 969432))          # development panel ONLY
FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1"
                       r"|hetdata_truth|vtype", re.I)
NOMINAL_LENGTH_M = 5.0


class RawTape:
    """One raw collected episode as a wallregen ``Timeline`` (single arm)."""

    def __init__(self, path: Path):
        self.kin = np.asarray(np.load(path / "veh_kin.npy", mmap_mode="r"))
        self.mask = np.asarray(np.load(path / "veh_mask.npy", mmap_mode="r")).astype(bool)
        self.vid = np.asarray(np.load(path / "vid.npy", mmap_mode="r")).astype(np.int64)
        cav = np.asarray(np.load(path / "is_cav.npy", mmap_mode="r")).astype(bool)
        self.t = np.asarray(np.load(path / "time_s.npy", mmap_mode="r"), np.float64)
        self.k = self.kin.shape[1]
        onehot = self.kin[..., 1:4]
        self.lane = np.where(onehot.sum(-1) > 0, onehot.argmax(-1), -1)
        self.registry = {int(v): f"v{int(v)}" for v in np.unique(self.vid) if v >= 0}
        self.is_cav_reg = np.zeros(max(self.registry, default=-1) + 1, bool)
        seen = self.vid >= 0
        for v in np.unique(self.vid[seen]):
            self.is_cav_reg[int(v)] = bool(cav[seen][self.vid[seen] == int(v)].any())

    def index_of(self, t0: float):
        j = int(round((t0 - self.t[0]) / DT_S))
        if 0 <= j < len(self.t) and abs(self.t[j] - t0) < 1e-6:
            return j
        return None

    def kin_rows(self, lo: int, hi: int) -> np.ndarray:
        out = np.zeros((hi - lo, self.k, 7), np.float32)
        m = self.mask[lo:hi]
        out[:, :, 0] = np.where(m, self.kin[lo:hi, :, 0], 0.0)
        for lane in range(3):
            out[:, :, 1 + lane] = ((self.lane[lo:hi] == lane) & m).astype(np.float32)
        out[:, :, 4] = np.where(m, self.kin[lo:hi, :, 4], 0.0)
        out[:, :, 5] = np.where(m, self.kin[lo:hi, :, 5], 0.0)
        out[:, :, 6] = np.where(m, NOMINAL_LENGTH_M, 0.0)
        return out

    def present(self, j: int) -> dict:
        slots = np.flatnonzero(self.mask[j] & (self.vid[j] >= 0))
        return {int(self.vid[j][s]): int(s) for s in slots.tolist()}

    def row(self, j: int, vid: int, slot: int) -> dict:
        return {"id": self.registry[vid], "s": float(self.kin[j, slot, 0]),
                "lane": int(self.lane[j, slot]), "v": float(self.kin[j, slot, 4]),
                "present": True,
                "is_cav": bool(self.is_cav_reg[vid])}

    def frame(self, j: int, roster=None) -> dict:
        pres = self.present(j)
        vids = sorted(pres) if roster is None else [v for v in roster if v in pres]
        return {"time_s": float(self.t[j]),
                "vehicles": [self.row(j, v, pres[v]) for v in vids]}

    def future(self, j0: int, roster: list, steps: int) -> list:
        alive = set(roster)
        frames = []
        for j in range(j0, min(len(self.t), j0 + steps + 1)):
            pres = self.present(j)
            alive &= set(pres)
            frames.append({"time_s": float(self.t[j]),
                           "vehicles": [self.row(j, v, pres[v])
                                        for v in roster if v in alive]})
        return frames

    def _align_history(self, j0, kin, mask, vid):
        """wallregen_states.Timeline._align_history (exact, single-arm axis)."""
        roster = self.present(j0)
        present = [np.flatnonzero(m & (v >= 0)) for m, v in zip(mask, vid)]
        context_counts = [sum(int(v[s]) not in roster for s in slots)
                          for v, slots in zip(vid, present)]
        k = max(self.k, len(roster) + max(context_counts, default=0))
        context_slots = set(range(k)) - set(roster.values())
        out_kin = np.zeros((len(kin), k, 7), np.float32)
        out_mask = np.zeros((len(kin), k), bool)
        out_vid = np.full((len(kin), k), -1, vid.dtype)
        context: dict[int, int] = {}
        for j in range(len(kin) - 1, -1, -1):
            source = {int(vid[j, s]): int(s) for s in present[j]}
            context = {v: s for v, s in context.items() if v in source}
            free = iter(sorted(context_slots - set(context.values())))
            for v in source:
                if v not in roster and v not in context:
                    context[v] = next(free)
            for v, src in source.items():
                dst = roster[v] if v in roster else context[v]
                out_kin[j, dst] = kin[j, src]
                out_mask[j, dst] = True
                out_vid[j, dst] = v
        return out_kin, out_mask, out_vid

    def history(self, j0: int, rows: int = HISTORY_ROWS, declared_s: float = HISTORY_S):
        """wallregen_states.Timeline.history(align_ids=True) on the raw tape."""
        lo = max(0, j0 - rows + 1)
        kin = self.kin_rows(lo, j0 + 1)
        mask = self.mask[lo:j0 + 1]
        vid = self.vid[lo:j0 + 1]
        kin, mask, vid = self._align_history(j0, kin, mask, vid)
        target = (vid[:-1] == vid[1:]) & (vid[:-1] >= 0)
        pres0 = self.mask[j0] & (self.vid[j0] >= 0)
        ids = [self.registry[int(self.vid[j0][k])] if k < self.k and pres0[k]
               else f"__history_context_slot{k}" for k in range(kin.shape[1])]
        t = kin.shape[0]
        return {"ids": ids, "history_steps": int(t),
                "kin": np.round(kin, 4).tolist(), "mask": mask.tolist(),
                "target_mask": target.tolist(),
                "history_s": float(declared_s), "history_stride_s": 1.0,
                "dt_s": DT_S, "available_span_s": (t - 1) * DT_S,
                "run_start_limited": bool((j0 - rows + 1) < 0),
                "history_rows_requested": int(rows)}


def build_record(tape: RawTape, t0: float, arm: str, horizon_s: float, world: str):
    j0 = tape.index_of(t0)
    if j0 is None:
        raise ValueError(f"t0 {t0} not on the tape")
    steps = int(round(horizon_s / DT_S))
    roster = sorted(tape.present(j0))
    future = tape.future(j0, roster, steps)
    return {"key": f"rbcf/{world}__{arm}", "world": world, "arm": arm,
            "policy": arm, "t0_s": float(t0), "dt_s": DT_S,
            "horizon_s": (len(future) - 1) * DT_S, "cav_controllers": {},
            "arms": [arm], "initial_state": tape.frame(j0),
            "history": tape.history(j0),
            "recorded": {arm: future}, "recorded_horizon_steps": {arm: len(future) - 1}}


# ------------------------------------------------------------------- metrics
def lane_gaps(s: np.ndarray, lane: np.ndarray, present: np.ndarray,
              length: np.ndarray) -> np.ndarray:
    """Same-lane bumper gap to the nearest ahead per slot (free = 250)."""
    gap = np.full(s.shape, 250.0)
    for j in range(s.shape[0]):
        idx = np.flatnonzero(present[j] & (lane[j] >= 0))
        if idx.size < 2:
            continue
        ss, ln = s[j, idx], lane[j, idx]
        order = np.argsort(ss)
        idx, ss, ln = idx[order], ss[order], ln[order]
        same = ln[:-1] == ln[1:]
        gap[j, idx[:-1][same]] = np.clip(ss[1:][same] - ss[:-1][same]
                                         - length[j, idx[1:][same]], 0.3, 250.0)
    return gap


def hdv_metrics(out: dict, horizon_s: float, dt: float = DT_S) -> dict:
    n_steps = int(round(horizon_s / dt))
    wm_pos, wm_vel, wm_msk = out["wm_pos"], out["wm_vel"], out["wm_msk"]
    rec_pos, rec_vel, rec_msk = out["rec_pos"], out["rec_vel"], out["rec_msk"]
    hdv = ~out["is_cav"]
    rows = horizon_s // dt
    res = {}
    wm_lane = out["wm_lane"]
    rec_lane = out["rec_lane"]
    wm_gap = lane_gaps(wm_pos[:n_steps + 1], wm_lane[:n_steps + 1],
                       wm_msk[:n_steps + 1], np.full_like(wm_pos, 5.0))
    rec_gap = lane_gaps(rec_pos[:n_steps + 1], rec_lane[:n_steps + 1],
                        rec_msk[:n_steps + 1], np.full_like(rec_pos, 5.0))
    horizons = tuple(h for h in (10, 30, 60, 120) if h <= n_steps * dt)
    for h in horizons:
        t = min(int(h / dt), n_steps)
        m = wm_msk[t] & rec_msk[t] & hdv
        res[f"{h}s"] = {
            "v_mae": float(np.abs(wm_vel[t][m] - rec_vel[t][m]).mean()),
            "s_mae": float(np.abs(wm_pos[t][m] - rec_pos[t][m]).mean()),
            "rows": int(m.sum())}
    m = (wm_msk[1:n_steps + 1] & rec_msk[1:n_steps + 1] & hdv[None, :])
    res["gap_mae"] = float(np.abs(wm_gap[1:n_steps + 1][m]
                                  - rec_gap[1:n_steps + 1][m]).mean())
    res["gap_rows"] = int(m.sum())
    return res


# --------------------------------------------------------------------- rollout
def roll(predictor, capture, rec: dict, arm: str, horizon_s: float) -> dict:
    import torch
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles
    n_steps = int(round(horizon_s / DT_S))
    t0, ids, kin, is_cav = t0_vehicles(rec)
    teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm,
                                               predictor.backend.tools)
    if teacher is None:
        raise RuntimeError(f"no recorded arm {arm}")
    tr = teacher[0].numpy()
    rp = present_seq[0].numpy().astype(bool)
    predictor.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm},
                             horizon_s, 0)
    r = capture["last"]
    n = len(ids)
    pos = np.concatenate([tr[:1, :, 0], r["positions"][0, :, :n].numpy()], 0)
    msk = np.concatenate([rp[:1], r["masks"][0, :, :n].numpy().astype(bool)], 0)
    vel = np.concatenate([tr[:1, :, 2], r["speeds"][0, :, :n].numpy()], 0)
    lane = np.concatenate([tr[:1, :, 1],
                           r["lanes"][0, :, :n].numpy().astype(np.int64)], 0)
    return {"wm_pos": pos, "wm_vel": vel, "wm_msk": msk, "wm_lane": lane,
            "rec_pos": tr[:, :, 0], "rec_vel": tr[:, :, 2], "rec_msk": rp,
            "rec_lane": tr[:, :, 1].astype(np.int64),
            "is_cav": np.asarray(is_cav, bool), "ids": list(ids)}


def load_arm(ckpt: Path, variant: str, run_physics: dict | None = None):
    import torch
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    from wms.physics.idm_sumo import PURE_SUMO_PHYSICS, PURE_SUMO_PHYSICS_EULER
    torch.set_num_threads(1)
    flags = RolloutFlags()
    physics = dict(run_physics or {})
    if variant in ("sumo_enc", "sumo_enc_euler", "sumo_pop"):
        flags = RolloutFlags(pure_physics=True)
        physics.update(PURE_SUMO_PHYSICS_EULER if variant == "sumo_enc_euler"
                       else PURE_SUMO_PHYSICS)
    if variant == "sumo_pop":
        flags = RolloutFlags(pure_physics=True, diag_encoder_population=True)
    if variant == "legacy_pure":
        # attribution arm: WM2's own physics core, every learned acceleration
        # correction OFF (isolates the CF model from the correction removal).
        flags = RolloutFlags(pure_physics=True)
    p = WmsPredictor.from_checkpoint(str(ckpt), flags, physics=physics)
    cap = {"last": None}
    orig = p.model.rollout

    def captured(state, **kw):
        out = orig(state, **kw)
        cap["last"] = out
        return out

    p.model.rollout = captured
    return p, cap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapes", type=Path, default=Path("E:/Temp/opencode/rb_cf_tapes"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("E:/agent-work/c4-execparity3/_parity3_data/wm2/wm_s.pt"))
    ap.add_argument("--horizon-s", type=float, default=HORIZON_S)
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--variants", default="wm2,sumo_enc,sumo_pop")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N worlds")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tools/rb_cf/tape_check_results.json")
    a = ap.parse_args()
    for name in (a.tapes, a.ckpt):
        if FORBIDDEN.search(str(name)):
            raise SystemExit(f"refusing restricted path {name}")
    if hashlib.sha256(a.ckpt.read_bytes()).hexdigest() != WM2_SHA:
        raise SystemExit("WM2 checkpoint SHA mismatch")
    import torch
    from wms.config.load import load_raw
    torch.manual_seed(0)
    # WM2's run physics (the local checkpoint copy has no physics.json next to
    # it; the registered run config resolves the same [physics] table).
    run_config = Path("E:/agent-work/c4-execparity3/_parity3_data"
                      "/runtime/configs/c4_wm_loop14_r16_ft.toml")
    if FORBIDDEN.search(str(run_config)):
        raise SystemExit(f"refusing restricted path {run_config}")
    run_physics = dict(load_raw(run_config).get("physics", {}))

    dirs = []
    for seed in DEV_SEEDS:
        hits = sorted(a.tapes.glob(f"w{seed}m*__native"))
        if len(hits) != 1:
            raise SystemExit(f"seed {seed}: expected 1 native tape, found {len(hits)}")
        dirs.append(hits[0])
    if a.limit:
        dirs = dirs[:a.limit]

    results = {"schema": "rb_cf_tape_check/1", "wm_sha256": WM2_SHA,
               "horizon_s": a.horizon_s, "t0_s": a.t0, "worlds": [],
               "panel": "WMDATA dev 969400-969431 native; entries OFF; "
                        "CAVs teacher-forced from the recording"}
    for d in dirs:
        world = d.name.rsplit("__", 1)[0]
        tape = RawTape(d)
        rec = build_record(tape, a.t0, "native", a.horizon_s, world)
        row = {"world": world,
               "sha": {n: hashlib.sha256((d / (n + ".npy")).read_bytes()).hexdigest()
                       for n in ("veh_kin", "veh_mask", "vid", "is_cav")}}
        for variant in [v.strip() for v in a.variants.split(",")]:
            p, cap = load_arm(a.ckpt, variant, run_physics)
            start = time.monotonic()
            out = roll(p, cap, rec, "native", a.horizon_s)
            row[variant] = hdv_metrics(out, a.horizon_s)
            row[variant]["seconds"] = round(time.monotonic() - start, 1)
        results["worlds"].append(row)
        print(json.dumps({"world": world, "wm2_10s_v": round(row.get(
            "wm2", {}).get("10s", {}).get("v_mae", float("nan")), 4),
            "sumo_enc_10s_v": round(row.get(
                "sumo_enc", {}).get("10s", {}).get("v_mae", float("nan")), 4)}),
            flush=True)

    def agg(variant, key, field=None):
        vals = []
        for w in results["worlds"]:
            value = w.get(variant, {}).get(key)
            if isinstance(value, dict):
                value = value.get(field)
            if value is not None and np.isfinite(value):
                vals.append(value)
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "n": len(vals)} if vals else None

    summary = {}
    keys = [k for k in results["worlds"][0][a.variants.split(",")[0].strip()]
            if k.endswith("s")]
    for variant in [v.strip() for v in a.variants.split(",")]:
        summary[variant] = {
            key: {field: agg(variant, key, field)
                  for field in ("v_mae", "s_mae")}
            for key in keys}
        summary[variant]["gap_mae"] = agg(variant, "gap_mae", "mean")
    results["summary"] = summary
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(summary, indent=1))
    print(json.dumps({"out": str(a.out)}))


if __name__ == "__main__":
    main()
