"""WALLREGEN tapes -> ``wms_eval_states/1`` unseen-wall evaluation sets (the score_c4.py format).

Reads a :mod:`traffic_shepherd.c4.wallregen` run directory (``worlds/<world>/world.json`` +
``base.npz`` + ``arm_<name>.npz``) and writes, per variant, a directory laid out like
``data_cache/wallc1_states_h60{,_real}``:

* ``wall_same.json``   every wall arm of every complete world (all outcomes, as wall_same);
* ``wall_pilot.json``  the wall arms whose wall assembled at some point (the old wall_pilot
  excluded ``aborted_in_assembly`` / ``not_feasible``; here also requests that never happened
  or were still assembling at the branch end);
* ``wall_native.json`` the matched NATIVE arm at the same (world, t0) (initial state from the
  native branch), for wall-benefit scoring (wall minus native) where the wall state has already
  diverged at t0 (variants ``h60`` / ``real``);
* ``BUILD_RECEIPT.json``, ``REJECTIONS.jsonl``, ``SHA256SUMS.json``.

Variants (t0 relative to the branch start t_w, the old sets' rules):

* ``h60``  t0 = t_w + 4.5 s (old shift 9 steps); history = the 11 native rows <= t0 (5 s),
  declared ``history_s`` 60 / stride 1 / ``run_start_limited`` true - shape-identical to the old
  ``wallc1_states_h60`` records (whose pack started at the branch restore);
* ``real`` t0 = t_w + 60.5 s; history = 121 native rows (60 s) - like ``wallc1_states_h60_real``;
* ``req``  t0 = the wall REQUEST tick of each arm (not_feasible arms have none); 60 s history;
  the wall and native arms are bit-identical at t0 (common random numbers, no command yet), so
  every record carries both recorded futures (``paired`` true): the clean wall-benefit set.
* ``h60full`` / ``reqfull`` keep the corresponding h60 / req decision times, but use 121
  native history rows from base times < t_w followed by branch times <= t0. History is
  remapped by vehicle id to the t0 slots; only a short base tape limits the 60 s span.

Record content follows ``wms.eval.states.build_state``: t0 snapshot (all present vehicles,
junction rows with lane -1), recorded futures of the t0 roster only (a vehicle drops out at its
first missing tick), id-keyed history over all K slot columns (``__history_context_slot<k>`` for
slots empty at t0), RSU entrance ``counter [2,30]`` (10 s bins, last 300 s), wall metadata.

    python -m traffic_shepherd.c4.wallregen_states --run RUN_DIR --variant h60 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DT_S = 0.5
HORIZON_STEPS = 240
NOMINAL_LENGTH_M = 5.0
V0_MPS = 27.78
MIN_VEHICLES = 5
COUNTER_BINS, COUNTER_BIN_S = 30, 10.0
FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)
VARIANTS = {
    # name: (t0 rule, history native rows, declared history_s, force run_start_limited)
    "h60": ("tw+4.5", 11, 60.0, True),
    "real": ("tw+60.5", 121, 60.0, False),
    "req": ("request", 121, 60.0, False),
    "h60full": ("tw+4.5", 121, 60.0, False),
    "reqfull": ("request", 121, 60.0, False),
}
FULL_HISTORY_VARIANTS = frozenset(("h60full", "reqfull"))
PILOT_EXCLUDED = ("not_feasible", "aborted_in_assembly", "waiting_at_end", "in_assembly_at_end")
WALL_ARMS = ("w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_of(world: str) -> int:
    return int(hashlib.sha256(str(world).encode("utf-8")).hexdigest()[:8], 16) % 100000


_WORLD_RE = re.compile(r"^w(\d+)m(\d+)p(\d+)t(\d+)$")


def world_meta(world: str) -> dict:
    m = _WORLD_RE.match(world)
    if not m:
        return {"world": world, "world_id": None, "demand": None, "penetration_pct": None,
                "t0_s": None}
    wid, demand, pen, t0 = m.groups()
    return {"world": world, "world_id": int(wid), "demand": int(demand),
            "penetration_pct": int(pen), "t0_s": float(t0)}


# --------------------------------------------------------------------------- timelines
class Timeline:
    """Base pre-t_w history + one arm's branch tape on one time axis (vid = arm registry)."""

    def __init__(self, base: dict, arm: dict, registry: list, is_cav: np.ndarray):
        keys = ("time_s", "s", "lane", "v", "a", "mask", "vid")
        if abs(float(base["time_s"][-1]) - float(arm["time_s"][0])) > 1e-6:
            raise ValueError("base history does not end at the branch start")
        self.cols = {k: np.concatenate([base[k][:-1], arm[k]], axis=0) for k in keys}
        self.n_base = len(base["time_s"]) - 1          # index of t_w on the full axis
        self.registry = registry
        self.is_cav_reg = is_cav
        self.first_seen_s = arm["first_seen_s"]
        self.first_entrance = arm["first_entrance"]
        self.k = self.cols["s"].shape[1]

    @property
    def t(self) -> np.ndarray:
        return self.cols["time_s"]

    def index_of(self, time_s: float) -> int | None:
        j = int(round((float(time_s) - float(self.t[0])) / DT_S))
        if 0 <= j < len(self.t) and abs(float(self.t[j]) - float(time_s)) < 1e-6:
            return j
        return None

    def cav(self, vid: int) -> bool:
        return bool(self.is_cav_reg[vid]) if 0 <= vid < len(self.is_cav_reg) else False

    def kin(self, j: int) -> np.ndarray:
        c = self.cols
        m = c["mask"][j]
        out = np.zeros((self.k, 7), np.float32)
        out[:, 0] = np.where(m, c["s"][j], 0.0)
        for lane in range(3):
            out[:, 1 + lane] = (c["lane"][j] == lane) & m
        out[:, 4] = np.where(m, c["v"][j], 0.0)
        out[:, 5] = np.where(m, c["a"][j], 0.0)
        out[:, 6] = np.where(m, NOMINAL_LENGTH_M, 0.0)
        return out

    def present(self, j: int) -> dict:
        """{vid: slot} of occupied slots at full index ``j``."""
        c = self.cols
        slots = np.flatnonzero(c["mask"][j] & (c["vid"][j] >= 0))
        return {int(c["vid"][j][s]): int(s) for s in slots.tolist()}

    def row(self, j: int, vid: int, slot: int) -> dict:
        c = self.cols
        return {"id": self.registry[vid], "s": float(c["s"][j][slot]),
                "lane": int(c["lane"][j][slot]), "v": float(c["v"][j][slot]),
                "present": True, "is_cav": self.cav(vid)}

    def frame(self, j: int, roster=None) -> dict:
        pres = self.present(j)
        vids = sorted(pres) if roster is None else [v for v in roster if v in pres]
        return {"time_s": float(self.t[j]),
                "vehicles": [self.row(j, v, pres[v]) for v in vids]}

    def future(self, j0: int, roster: list, steps: int = HORIZON_STEPS) -> list:
        alive = set(roster)
        frames = []
        for j in range(j0, min(len(self.t), j0 + steps + 1)):
            pres = self.present(j)
            alive &= set(pres)
            frames.append({"time_s": float(self.t[j]),
                           "vehicles": [self.row(j, v, pres[v]) for v in roster if v in alive]})
        return frames

    def _align_history(self, j0: int, kin: np.ndarray, mask: np.ndarray,
                       vid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pin t0 vehicles to their t0 slots, retaining all other observed traffic as context.

        Registry indices survive the saved-state restore (each arm inherits the base
        registry prefix), but slot columns need not. Work backwards so historical context
        keeps its column over adjacent ticks, with identity changes breaking target_mask.
        Empty t0 columns are anonymous context, never rollout identities. If reserving the
        t0 roster exhausts K, append anonymous columns rather than dropping neighbours.
        """
        roster = self.present(j0)
        present = [np.flatnonzero(m & (v >= 0)) for m, v in zip(mask, vid)]
        context_counts = [sum(int(v[s]) not in roster for s in slots)
                          for v, slots in zip(vid, present)]
        k = max(self.k, len(roster) + max(context_counts, default=0))
        context_slots = set(range(k)) - set(roster.values())
        out_kin = np.zeros((len(kin), k, 7), np.float32)
        out_mask = np.zeros((len(kin), k), bool)
        out_vid = np.full((len(kin), k), -1, vid.dtype)
        context = {}
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

    def history(self, j0: int, rows: int, declared_s: float, force_limited: bool, *,
                align_ids: bool = False) -> dict:
        lo = max(0, j0 - rows + 1)
        c = self.cols
        kin = np.stack([self.kin(j) for j in range(lo, j0 + 1)])
        mask = c["mask"][lo:j0 + 1].astype(bool)
        vid = c["vid"][lo:j0 + 1]
        if align_ids:
            kin, mask, vid = self._align_history(j0, kin, mask, vid)
        target = (vid[:-1] == vid[1:]) & (vid[:-1] >= 0)
        pres0 = c["mask"][j0] & (c["vid"][j0] >= 0)
        ids = [self.registry[int(c["vid"][j0][k])] if k < self.k and pres0[k]
               else f"__history_context_slot{k}" for k in range(kin.shape[1])]
        t = kin.shape[0]
        return {"ids": ids, "history_steps": int(t),
                "kin": np.round(kin, 4).tolist(), "mask": mask.tolist(),
                "target_mask": target.tolist(),
                "history_s": float(declared_s), "history_stride_s": 1.0, "dt_s": DT_S,
                "available_span_s": (t - 1) * DT_S,
                "run_start_limited": bool(force_limited or (j0 - rows + 1) < 0),
                "history_rows_requested": int(rows)}

    def counter(self, t0_s: float) -> list:
        out = np.zeros((2, COUNTER_BINS), np.float32)
        ts = np.asarray(self.first_seen_s, np.float64)
        ent = np.asarray(self.first_entrance, np.int64)
        ok = np.isfinite(ts) & (ent >= 0)
        age = float(t0_s) - ts[ok]
        e = ent[ok]
        keep = (age >= -1e-9) & (age < COUNTER_BINS * COUNTER_BIN_S)
        np.add.at(out, (e[keep], np.floor(np.maximum(age[keep], 0) / COUNTER_BIN_S).astype(int)),
                  1.0)
        return out.tolist()


def frames_match(a: dict, b: dict, tol: float = 1e-4) -> bool:
    ra = {r["id"]: r for r in a["vehicles"]}
    rb = {r["id"]: r for r in b["vehicles"]}
    if set(ra) != set(rb):
        return False
    return all(ra[i]["lane"] == rb[i]["lane"] and abs(ra[i]["s"] - rb[i]["s"]) <= tol
               and abs(ra[i]["v"] - rb[i]["v"]) <= tol for i in ra)


def phase_at(tl: dict, t0: float) -> str:
    req, asm, end = tl.get("request_s"), tl.get("assembled_at_s"), tl.get("owner_end_s")
    if req is None or t0 < req - 1e-9:
        return "none"
    if end is not None and t0 >= end - 1e-9 and tl.get("handback_s") is not None:
        return "handed_back"
    if asm is None or t0 < asm - 1e-9:
        return "assembly"
    return "hold"


# --------------------------------------------------------------------------- loading
def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_world(wdir: Path) -> tuple[dict, dict, dict]:
    rec = json.loads((wdir / "world.json").read_text(encoding="utf-8"))
    base = _npz(wdir / "base.npz")
    tls = {}
    for arm, info in rec["arms"].items():
        a = _npz(wdir / f"arm_{arm}.npz")
        tls[arm] = Timeline(base, a, info["registry"], a["is_cav"])
    return rec, base, tls


# --------------------------------------------------------------------------- records
def _record(*, rec: dict, arm: str, tl: Timeline, t0: float, variant: str, set_name: str,
            native: Timeline | None, rows: int, declared_s: float, force_limited: bool,
            population: str) -> dict:
    world = rec["world"]
    j0 = tl.index_of(t0)
    if j0 is None:
        raise ValueError(f"t0 {t0} not on the {arm} tape")
    roster = sorted(tl.present(j0))
    initial = tl.frame(j0, roster)
    recorded = {"wall" if arm.startswith("w-") else "native": tl.future(j0, roster)}
    own = next(iter(recorded))
    pair_status, paired = None, False
    if native is not None and arm.startswith("w-"):
        jn = native.index_of(t0)
        if jn is not None and frames_match(initial, native.frame(jn)):
            nat_ids = {native.registry[v]: v for v in native.present(jn)}
            n_roster = [nat_ids[v["id"]] for v in initial["vehicles"]]
            recorded["native"] = native.future(jn, n_roster)
            pair_status, paired = "matched_initial_physical_state", True
        else:
            pair_status = "initial_physical_state_mismatch"
    arm_info = rec["arms"][arm]
    timeline = dict(arm_info.get("wall_timeline") or {})
    if timeline:
        timeline["world"] = world
    t_w = float(rec["plan_world"]["t_w_s"])
    shift = int(round((t0 - t_w) / DT_S))
    out = {
        "key": f"wallregen/{world}__{arm}",
        "pack": f"wallregen_{population}",
        "world": world,
        "run": f"{world}__{arm}",
        "chunk": "wallregen",
        "batch": f"wallregen_{population}",
        "arm": own,
        "policy": own,
        "split": "EVAL_WALLREGEN",
        "t0_s": float(t0),
        "t0_shift_steps": shift,
        "t0_shift_s": shift * DT_S,
        "t0_origin_s": t_w,
        "t0_rel_origin_s": float(t0) - t_w,
        "t0_rule": VARIANTS[variant][0],
        "horizon_s": (len(recorded[own]) - 1) * DT_S,
        "dt_s": DT_S,
        "seed": _seed_of(world),
        "sumo_seed": int(rec["plan_world"]["seed"]),
        "v0_mps": V0_MPS,
        "cav_controllers": {},
        "world_meta": world_meta(world),
        "counter": tl.counter(t0),
        "arms": sorted(recorded),
        "paired": paired,
        "initial_state": initial,
        "history": (tl.history(j0, rows, declared_s, force_limited, align_ids=True)
                    if variant in FULL_HISTORY_VARIANTS
                    else tl.history(j0, rows, declared_s, force_limited)),
        "recorded": recorded,
        "recorded_horizon_steps": {k: len(v) - 1 for k, v in recorded.items()},
        "source_set": set_name,
        "variant": variant,
        "population": population,
        "population_manifest": rec.get("population_manifest"),
        "world_seen_in_training": False,
        "controller_seen_in_training": False,
        "unseen_controller": own == "wall",
        "source_arm": arm,
    }
    if own == "wall":
        out.update({
            "native_pair_status": pair_status,
            "wall_members": list(timeline.get("members") or []),
            "wall_timeline": timeline,
            "wall_engaged": bool(timeline.get("engaged")),
            "wall_owned": bool(timeline.get("owned")),
            "wall_exposure_status": timeline.get("outcome"),
            "wall_phase_at_t0": phase_at(timeline, t0),
        })
    return out


class _Writer:
    def __init__(self, path: Path, set_name: str, population: str, variant: str):
        self.path = path
        self.tmp = Path(str(path) + ".tmp")
        self.fh = self.tmp.open("w", encoding="ascii")
        self.n = 0
        head = {"schema": "wms_eval_states/1", "set": set_name, "horizon_s": 120.0,
                "population": population, "variant": variant}
        self.fh.write(json.dumps(head)[:-1] + ', "windows": [')

    def add(self, record: dict) -> None:
        self.fh.write(("," if self.n else "") + "\n" + json.dumps(record, separators=(",", ":")))
        self.n += 1

    def close(self) -> int:
        self.fh.write(f'\n], "n_states": {self.n}}}')
        self.fh.close()
        self.tmp.replace(self.path)
        return self.n


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(run: Path, out: Path, variant: str, *, worlds=None) -> dict:
    run, out = Path(run), Path(out)
    for p in (run, out):
        if FORBIDDEN.search(str(p).replace("\\", "/")):
            raise SystemExit(f"restricted path token in {p}")
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {sorted(VARIANTS)}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    rule, rows, declared_s, force_limited = VARIANTS[variant]
    plan = json.loads((run / "PLAN.json").read_text(encoding="utf-8"))
    population = plan["population"]
    writers = {name: _Writer(out / f"{name}.json", name, population, variant)
               for name in ("wall_same", "wall_pilot", "wall_native")}
    rejections = []
    counts = {"worlds_complete": 0, "worlds_missing": 0}
    outcomes: dict = {}
    t_start = time.time()
    for w in plan["worlds"]:
        if worlds and w["world"] not in worlds:
            continue
        wdir = run / "worlds" / w["world"]
        try:
            rec, _base, tls = load_world(wdir)
            if not rec.get("complete"):
                raise FileNotFoundError("world incomplete")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            counts["worlds_missing"] += 1
            rejections.append({"world": w["world"], "reason": f"world_unavailable: {exc}"})
            continue
        counts["worlds_complete"] += 1
        t_w = float(w["t_w_s"])
        native = tls.get("native")
        native_t0s = set()
        for arm in [a for a in rec["arms"] if a in WALL_ARMS]:
            tlinfo = rec["arms"][arm].get("wall_timeline") or {}
            outcome = tlinfo.get("outcome")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if rule == "tw+4.5":
                t0 = t_w + 4.5
            elif rule == "tw+60.5":
                t0 = t_w + 60.5
            else:
                t0 = tlinfo.get("request_s")
                if t0 is None:
                    rejections.append({"world": w["world"], "arm": arm,
                                       "reason": f"no_request ({outcome})"})
                    continue
            try:
                r = _record(rec=rec, arm=arm, tl=tls[arm], t0=float(t0), variant=variant,
                            set_name="wall_same", native=native, rows=rows,
                            declared_s=declared_s, force_limited=force_limited,
                            population=population)
            except ValueError as exc:
                rejections.append({"world": w["world"], "arm": arm, "reason": str(exc)})
                continue
            if len(r["initial_state"]["vehicles"]) < MIN_VEHICLES:
                rejections.append({"world": w["world"], "arm": arm, "reason": "degenerate_t0"})
                continue
            if r["recorded_horizon_steps"]["wall"] < HORIZON_STEPS:
                rejections.append({"world": w["world"], "arm": arm,
                                   "reason": "short_future", "steps": r["recorded_horizon_steps"]})
                continue
            if rule == "request" and not r["paired"]:
                rejections.append({"world": w["world"], "arm": arm,
                                   "reason": "req_state_not_matched (determinism defect)"})
            writers["wall_same"].add(r)
            if outcome not in PILOT_EXCLUDED:
                r["source_set"] = "wall_pilot"
                writers["wall_pilot"].add(r)
            if native is not None and not r["paired"] and float(t0) not in native_t0s:
                native_t0s.add(float(t0))
                try:
                    n = _record(rec=rec, arm="native", tl=native, t0=float(t0), variant=variant,
                                set_name="wall_native", native=None, rows=rows,
                                declared_s=declared_s, force_limited=force_limited,
                                population=population)
                    writers["wall_native"].add(n)
                except ValueError as exc:
                    rejections.append({"world": w["world"], "arm": "native", "reason": str(exc)})
    n_out = {name: wr.close() for name, wr in writers.items()}
    with (out / "REJECTIONS.jsonl").open("w", encoding="utf-8") as fh:
        for r in rejections:
            fh.write(json.dumps(r) + "\n")
    sums = {p.name: _sha256(p) for p in sorted(out.glob("*.json"))}
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1), encoding="utf-8")
    receipt = {"schema": "c4_wallregen_states_receipt/1", "created": _now(),
               "source_run": str(run), "plan_sha256": _sha256(run / "PLAN.json"),
               "population": population, "variant": variant, "t0_rule": rule,
               "history_rows": rows, "declared_history_s": declared_s,
               "run_start_limited_forced": force_limited, "horizon_steps": HORIZON_STEPS,
               "pilot_excluded_outcomes": list(PILOT_EXCLUDED), "n_states": n_out,
               "counts": counts, "wall_outcomes": outcomes, "n_rejections": len(rejections),
               "wall_s": round(time.time() - t_start, 1), "sha256": sums}
    (out / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="wallregen run directory (PLAN.json + worlds/)")
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--worlds", default=None, help="comma-separated subset")
    a = ap.parse_args(argv)
    receipt = convert(Path(a.run), Path(a.out), a.variant,
                      worlds=set(a.worlds.split(",")) if a.worlds else None)
    print(json.dumps({k: receipt[k] for k in ("variant", "population", "n_states", "counts",
                                              "wall_outcomes", "n_rejections")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
