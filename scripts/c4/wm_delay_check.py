"""A6 DELAY-ACCURACY CHECK: WM vs SUMO delay level and paired delay difference.

Two measurements on a FIXED, deterministic selection of TRAIN-val records:

1. **Delay LEVEL** - the team on-road delay of a recorded start, per region and horizon,
   computed identically from the recorded SUMO frames and from the WM rollout of the same
   start (``relu(1 - v/31.29) * present`` on a 1 s grid, on-road ``s < 553.33 m``).  Regions
   are the RSU zones (``up_near`` / ``ramp`` / ``merge`` / ``downstream``, see
   :func:`region_of`) plus the ramp/source ``ramp_queue``.  The metric reported for the loop
   is the 120 s total ``|log(WM / SUMO)|``.

2. **Paired delay DIFFERENCE** - on the recorded paired branches of ``data/genrand_branch``
   (same saved SUMO state, an action arm vs the base/native arm, both recorded) the tool
   compares the WM-predicted arm delta against the recorded SUMO arm delta: sign agreement,
   Spearman rank correlation and mean absolute error, averaged over ``--k`` WM samples.  The
   metric reported for the loop is the sign agreement at 60 s.

The tool also reports the per-vehicle ``|ds|`` at 10/30 s for HDVs near a CAV (within
``SENSOR_RANGE_M``) as a **non-gating** diagnostic (the old rollback term, reported only).

    python scripts/c4/wm_delay_check.py \\
        --wm runs/c4_wm_v10_iter000/wm_s_best.pt \\
        --states fixed_train_val_states.json --pairs genrand_branch_states.json \\
        --selection data_cache/delay_states_selection.json \\
        --pairs-selection data_cache/delay_pairs_selection.json \\
        --horizons 60,120,300 --k 8 --device cpu --workers 3 \\
        --entries queue --sumo-pairs check/pairs.json --out check/delay_check.json

Deterministic selection: the same input always yields the same records (evenly spaced across
the sorted ``(world, key, source_arm, t0_s)`` / pair keys), so ``--save-selection`` /
``--save-pairs-selection`` write it once for every later run.  Refuses any path carrying a
restricted token (sealed/holdout/heldout/cal_split/chunk5/j4b/inc1).  torch/wms are imported
lazily: the pure helpers import without torch.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
from pathlib import Path

SCHEMA = "c4_wm_delay_check/1"
SELECTION_SCHEMA = "c4_wm_delay_check/selection/1"
DEFAULT_HORIZONS = (60.0, 120.0, 300.0)
DEFAULT_K = 8
DEFAULT_SELECT = 48
GRID_S = 1.0
DT_S = 0.5
FREE_SPEED_MPS = 31.29
STOPPED_V_MPS = 0.5
SENSOR_RANGE_M = 100.0
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")

#: Region bounds on the s_down frame (upstream negative, merge zone 0..75.44, road ends 553.33).
MERGE_S0_M = 0.0
MERGE_S1_M = 75.44
ON_ROAD_S_MAX_M = 553.33
UP_NEAR_S_MIN_M = -800.0       # RSU mainline zone start (contract.RSU_MAINLINE_S_M)
RAMP_S_MIN_M = -378.66         # RSU ramp zone start (contract.RSU_RAMP_S_M)
RAMP_LANE = 2
REGIONS = ("up_near", "ramp", "merge", "downstream")
#: Horizon of the level metric the loop rollback reads (120 s).
LEVEL_HORIZON_S = 120.0
#: Horizon of the paired-delta metric the loop rollback reads (60 s).
PAIR_HORIZON_S = 60.0
#: Per-vehicle |ds| horizons reported (non-gating).
NEAR_CAV_HORIZONS = (10.0, 30.0)

_WORKER: dict = {}


# --------------------------------------------------------------------------- pure helpers
def refuse_restricted(path) -> str:
    """Return ``str(path)`` unless it carries a restricted token (then raise ``SystemExit``)."""
    text = str(path)
    if any(token in text.replace("\\", "/").lower() for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {text}")
    return text


def parse_horizons(text) -> tuple[float, ...]:
    """"60,120,300" -> ``(60.0, 120.0, 300.0)``."""
    if isinstance(text, (list, tuple)):
        return tuple(float(h) for h in text)
    return tuple(float(part) for part in str(text).replace(";", ",").split(",") if part.strip())


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _median(values):
    vals = sorted(float(v) for v in values if _finite(v))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _h(horizon) -> str:
    return str(int(round(float(horizon))))


def region_of(s, lane) -> str | None:
    """Region of one vehicle, or ``None`` when it is off the modelled road.

    The on-ramp/acceleration lane (``RAMP_LANE``) is the ``ramp`` region; mainline lanes
    (0/1) split into ``up_near`` / ``merge`` / ``downstream`` at the merge-zone bounds.
    """
    if not _finite(s):
        return None
    s = float(s)
    if s >= ON_ROAD_S_MAX_M:
        return None
    if int(lane) == RAMP_LANE:
        return "ramp" if RAMP_S_MIN_M <= s < MERGE_S1_M else None
    if s < MERGE_S0_M:
        return "up_near" if s >= UP_NEAR_S_MIN_M else None
    return "merge" if s < MERGE_S1_M else "downstream"


def _grid_times(t0_s: float, horizon_s: float) -> list[float]:
    """Integer-second grid ``t0, t0+1, ..., t0+h-1`` (1 s accumulation steps)."""
    n = int(round(float(horizon_s) / GRID_S))
    return [float(t0_s) + i * GRID_S for i in range(max(0, n))]


def _frame_at(frames, t: float) -> dict | None:
    frames = list(frames or [])
    if not frames:
        return None
    best = min(frames, key=lambda f: abs(float(f["time_s"]) - t))
    if abs(float(best["time_s"]) - t) > DT_S:
        return None
    return best


#: 10-20: score the WM on the same vehicles the recorded SUMO stream contains (the t0 roster).
#: The exported records carry no post-t0 entrants, so WM entrants (entries on/queue) would add
#: delay SUMO never counts (v5m 300 s ratio 4.9x was this artefact).
ROSTER_ONLY = os.environ.get("WM_DELAY_ROSTER_ONLY", "1") == "1"


def frame_delay(frames, t0_s: float, horizons,
                free_speed_mps: float = FREE_SPEED_MPS, ids=None) -> dict:
    """Normalised on-road delay per region + ramp/source queue for each horizon.

    ``delay = sum over the 1 s grid of relu(1 - v/free_speed)`` for every present vehicle,
    binned by :func:`region_of`.  ``ramp_queue`` is the stopped (``v < 0.5``) vehicle-seconds
    of the entrance backlog (queued before the ramp start or on an entrant slot).  Returns
    ``{horizon_label: {region: value, "total": value, "ramp_queue": value}}``.
    """
    out: dict[str, dict[str, float]] = {}
    frames = list(frames or [])
    for horizon in horizons:
        block = {region: 0.0 for region in REGIONS}
        block["ramp_queue"] = 0.0
        for t in _grid_times(t0_s, horizon):
            frame = _frame_at(frames, t)
            if frame is None:
                continue
            for row in frame.get("vehicles", []):
                if not row.get("present", True):
                    continue
                if ids is not None and str(row.get("id", "")) not in ids:
                    continue      # 10-20: recorded SUMO streams track the t0 roster only
                s, v = float(row.get("s", 0.0)), float(row.get("v", 0.0))
                region = region_of(s, row.get("lane", -1))
                if region is not None:
                    block[region] += max(0.0, 1.0 - v / free_speed_mps)
                elif v < STOPPED_V_MPS and (s < RAMP_S_MIN_M
                                            or str(row.get("id", "")).startswith("__entry_")):
                    block["ramp_queue"] += 1.0
        block["total"] = float(sum(block[region] for region in REGIONS))
        block["total_with_queue"] = float(block["total"] + block["ramp_queue"])
        out[_h(horizon)] = block
    return out


def near_cav_abs_ds(predicted, recorded, t0_frame, horizons,
                    sensor_range_m: float = SENSOR_RANGE_M) -> dict:
    """Median per-vehicle ``|ds|`` for HDVs within ``sensor_range_m`` of a CAV at t0.

    Reported (non-gating) diagnostic.  CAV positions are their recorded t0 positions; the
    matched HDVs are present in BOTH streams at the horizon.
    """
    t0_rows = {str(row["id"]): row for row in t0_frame.get("vehicles", [])
               if row.get("present", True)}
    cavs = [float(row["s"]) for row in t0_rows.values() if row.get("is_cav")]
    t0 = float(t0_frame.get("time_s", 0.0))
    out: dict[str, float | None] = {}
    for horizon in horizons:
        p, r = _present_at(predicted, t0 + float(horizon)), _present_at(recorded, t0 + float(horizon))
        vals = []
        for vid in set(p) & set(r):
            row = t0_rows.get(vid)
            if row is None or row.get("is_cav") or not cavs:
                continue
            if min(abs(float(row["s"]) - c) for c in cavs) <= sensor_range_m:
                vals.append(abs(float(p[vid]["s"]) - float(r[vid]["s"])))
        out[_h(horizon)] = _median(vals)
    return out


def _present_at(frames, t: float) -> dict[str, dict]:
    frame = _frame_at(frames, t)
    if frame is None:
        return {}
    return {str(row["id"]): row for row in frame.get("vehicles", []) if row.get("present", True)}


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def sign_agreement(sumo_deltas, wm_deltas) -> float | None:
    """Fraction of pairs whose WM and SUMO deltas carry the same sign (ties agree on 0)."""
    pairs = [(float(a), float(b)) for a, b in zip(sumo_deltas, wm_deltas)
             if _finite(a) and _finite(b)]
    if not pairs:
        return None
    return sum(_sign(a) == _sign(b) for a, b in pairs) / len(pairs)


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys) -> float | None:
    """Spearman rank correlation (``None`` on <2 pairs or zero variance)."""
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys) if _finite(a) and _finite(b)]
    if len(pairs) < 2:
        return None
    rx, ry = _ranks([a for a, _ in pairs]), _ranks([b for _, b in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def mean_abs_error(sumo_deltas, wm_deltas) -> float | None:
    vals = [abs(float(a) - float(b)) for a, b in zip(sumo_deltas, wm_deltas)
            if _finite(a) and _finite(b)]
    return (sum(vals) / len(vals)) if vals else None


def log_ratio(wm_total, sumo_total) -> float | None:
    """``log(WM / SUMO)`` for two positive totals, else ``None``."""
    if not _finite(wm_total) or not _finite(sumo_total):
        return None
    if float(wm_total) <= 0.0 or float(sumo_total) <= 0.0:
        return None
    return math.log(float(wm_total) / float(sumo_total))


def selection_key(record: dict) -> tuple[str, str, str, float]:
    """Deterministic ordering key of one state record."""
    return (str(record.get("world")), str(record.get("key") or record.get("pair_key") or ""),
            str(record.get("source_arm") or record.get("arm") or ""),
            round(float(record.get("t0_s", 0.0)), 3))


def pair_key(record: dict) -> tuple[str, str, str, float]:
    """Deterministic ordering key of one paired-branch unit."""
    return (str(record.get("pair_key")), str(record.get("ref_arm")),
            str(record.get("action_arm")), round(float(record.get("t0_s", 0.0)), 3))


def select_evenly(items, n: int, key=selection_key) -> list:
    """Pick ``n`` items evenly spaced across the sorted keys (all items when fewer than ``n``)."""
    ordered = sorted(items, key=key)
    if n <= 0 or not ordered:
        return []
    if len(ordered) <= n:
        return ordered
    if n == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]


# --------------------------------------------------------------------------- paired branches
def _is_reference_arm(name: str) -> bool:
    text = str(name).strip().lower()
    return text in ("native", "base", "b") or text.endswith("_native")


def group_pairs(records) -> list[dict]:
    """Group per-arm state records into ``{pair_key, ref_arm, action_arm, record}`` units.

    Accepts either one record per pair (its ``recorded`` dict already carries both arms) or
    one record per arm sharing a ``pair_key`` (their ``recorded`` dicts are merged).  The
    reference arm is the native/base arm, else the lexicographically first arm.
    """
    groups: dict[str, dict] = {}
    for record in records:
        key = record.get("pair_key")
        if not key:
            continue
        key = str(key)
        block = groups.setdefault(key, {"state": record, "arms": {}})
        recorded = record.get("recorded") or {}
        if isinstance(recorded, dict) and recorded:
            for name, frames in recorded.items():
                if frames:
                    block["arms"].setdefault(str(name), list(frames))
    units: list[dict] = []
    for key in sorted(groups):
        block = groups[key]
        usable = sorted(name for name, frames in block["arms"].items() if frames)
        if not usable:
            continue
        ref = next((name for name in usable if _is_reference_arm(name)), usable[0])
        state = dict(block["state"])
        for name in usable:
            if name == ref:
                continue
            record = dict(state)
            record["recorded"] = {ref: block["arms"][ref], name: block["arms"][name]}
            record["pair_key"] = key
            record["ref_arm"] = ref
            record["action_arm"] = name
            units.append(record)
    return units


# --------------------------------------------------------------------------- scoring
class PredictorFactory:
    """Picklable factory of ``WmsPredictor.predict_states`` honouring device and entries."""

    def __init__(self, path, device: str = "cpu", entries: str = "off",
                 diag: dict | None = None) -> None:
        self.path = str(path)
        self.device = str(device)
        self.entries = str(entries)
        # V10BRKPROBE inference-only braking switches (None/default = bit-identical).
        self.diag = dict(diag or {})

    def __call__(self):
        from wms.eval.predictor import RolloutFlags, WmsPredictor
        if self.entries == "queue":
            flags = RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True,
                                 **self.diag)
        elif self.entries == "on":
            flags = RolloutFlags(entries=True, entries_v2=True, **self.diag)
        else:
            flags = RolloutFlags(entries=False, entries_v2=False, **self.diag)
        return WmsPredictor.from_checkpoint(self.path, flags,
                                            device=self.device).predict_states


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_records(path) -> list[dict]:
    """All records of one states/windows JSON file, or of every ``*.json`` in a directory."""
    from wms.eval.normalize import iter_windows
    target = Path(path)
    if target.is_dir():
        records: list[dict] = []
        for name in sorted(target.glob("*.json")):
            records.extend(iter_windows(name))
        return records
    return list(iter_windows(target))


def load_selection(path, key: str = "states") -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"selection file {path} carries no {key}")
    return rows


def write_selection(path, rows, key: str = "states") -> None:
    payload = {"schema": SELECTION_SCHEMA, "n": len(rows), key: list(rows)}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _controllers(record: dict) -> dict:
    ctrl = record.get("cav_controllers") or {}
    return dict(ctrl) if isinstance(ctrl, dict) else {}


def predict_arm(predictor, record, arm, horizons, k, predict_horizon_s, t0_frame) -> dict:
    """K-sample-averaged WM delay block + first-sample near-CAV |ds| for one arm."""
    region_sum = {_h(hz): {r: 0.0 for r in REGIONS} for hz in horizons}
    queue_sum = {_h(hz): 0.0 for hz in horizons}
    first_frames = None
    n = max(1, int(k))
    for sample in range(n):
        frames = predictor(record, {**_controllers(record), "arm": arm}, predict_horizon_s,
                           sample)
        if first_frames is None:
            first_frames = frames
        roster = ({str(v.get("id", "")) for v in t0_frame.get("vehicles", [])}
                  if ROSTER_ONLY and t0_frame else None)
        block = frame_delay(frames, float(record["t0_s"]), horizons, ids=roster)
        for hz in horizons:
            label = _h(hz)
            for region in REGIONS:
                region_sum[label][region] += block[label][region]
            queue_sum[label] += block[label]["ramp_queue"]
    wm: dict[str, dict[str, float]] = {}
    for hz in horizons:
        label = _h(hz)
        wm[label] = {r: region_sum[label][r] / n for r in REGIONS}
        wm[label]["ramp_queue"] = queue_sum[label] / n
        wm[label]["total"] = sum(wm[label][r] for r in REGIONS)
    from wms.eval.normalize import recorded_for
    near = near_cav_abs_ds(first_frames, recorded_for(record, arm), t0_frame,
                           NEAR_CAV_HORIZONS) if first_frames is not None else {}
    return wm, near


def evaluate_start(predictor, record, horizons, k, predict_horizon_s) -> dict:
    """WM vs SUMO delay level + near-CAV |ds| for every recorded arm of one start."""
    from wms.eval.normalize import recorded_for
    arms: dict[str, dict] = {}
    t0_frame = _t0_frame(record)
    for arm in sorted((record.get("recorded") or {}).keys()):
        try:
            rec_frames = recorded_for(record, arm)
            wm, near = predict_arm(predictor, record, arm, horizons, k, predict_horizon_s,
                                   t0_frame)
            arms[arm] = {"wm": wm, "sumo": frame_delay(rec_frames, float(record["t0_s"]),
                                                       horizons),
                         "near_cav_abs_ds": near}
        except Exception as exc:  # noqa: BLE001 - one bad arm never aborts the rest
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"world": str(record.get("world")), "key": record.get("key"),
            "source_arm": record.get("source_arm"), "t0_s": float(record.get("t0_s", 0.0)),
            "arms": arms}


def _t0_frame(record: dict) -> dict:
    for key in ("state", "initial_state"):
        state = record.get(key)
        if isinstance(state, dict) and "vehicles" in state:
            return {"time_s": float(state.get("time_s", record.get("t0_s", 0.0))),
                    "vehicles": list(state["vehicles"])}
    return {"time_s": float(record.get("t0_s", 0.0)), "vehicles": []}


def evaluate_pair(predictor, record, horizons, k, predict_horizon_s) -> dict:
    """WM-predicted and recorded SUMO delay delta (action - base) for one pair."""
    from wms.eval.normalize import recorded_for
    ref, action = record["ref_arm"], record["action_arm"]
    t0_frame = _t0_frame(record)
    wm_ref, _ = predict_arm(predictor, record, ref, horizons, k, predict_horizon_s, t0_frame)
    wm_act, _ = predict_arm(predictor, record, action, horizons, k, predict_horizon_s, t0_frame)
    sumo_ref = frame_delay(recorded_for(record, ref), float(record["t0_s"]), horizons)
    sumo_act = frame_delay(recorded_for(record, action), float(record["t0_s"]), horizons)
    per_horizon = {}
    for hz in horizons:
        label = _h(hz)
        per_horizon[label] = {
            "wm_delta": wm_act[label]["total"] - wm_ref[label]["total"],
            "sumo_delta": sumo_act[label]["total"] - sumo_ref[label]["total"],
        }
    return {"pair_key": record.get("pair_key"), "ref_arm": ref, "action_arm": action,
            "per_horizon": per_horizon}


# --------------------------------------------------------------------------- aggregation
def aggregate_level(start_results, horizons) -> dict:
    """Median WM/SUMO region delay and the total log ratio across starts."""
    out: dict[str, dict] = {}
    for hz in horizons:
        label = _h(hz)
        wm = {r: [] for r in REGIONS}
        sumo = {r: [] for r in REGIONS}
        wmq, sumoq, logr = [], [], []
        for row in start_results:
            for arm in row["arms"].values():
                if "error" in arm:
                    continue
                for r in REGIONS:
                    wm[r].append(arm["wm"][label][r])
                    sumo[r].append(arm["sumo"][label][r])
                wmq.append(arm["wm"][label]["ramp_queue"])
                sumoq.append(arm["sumo"][label]["ramp_queue"])
                logr.append(log_ratio(arm["wm"][label]["total"], arm["sumo"][label]["total"]))
        wm_total = sum(_median(wm[r]) or 0.0 for r in REGIONS)
        sumo_total = sum(_median(sumo[r]) or 0.0 for r in REGIONS)
        out[label] = {
            "wm": {r: _median(wm[r]) for r in REGIONS},
            "sumo": {r: _median(sumo[r]) for r in REGIONS},
            "wm_total": wm_total, "sumo_total": sumo_total,
            "wm_ramp_queue": _median(wmq), "sumo_ramp_queue": _median(sumoq),
            "log_ratio_total": _median([v for v in logr if v is not None]) if logr else None,
            "n": len(logr),
        }
    return out


def aggregate_pairs(pair_results, horizons) -> dict:
    """Sign agreement, Spearman and MAE of WM vs SUMO arm deltas across pairs."""
    out: dict[str, dict] = {}
    for hz in horizons:
        label = _h(hz)
        sumo = [r["per_horizon"][label]["sumo_delta"] for r in pair_results]
        wm = [r["per_horizon"][label]["wm_delta"] for r in pair_results]
        out[label] = {
            "n": len(pair_results),
            "sign_agreement": sign_agreement(sumo, wm),
            "spearman": spearman(sumo, wm),
            "mean_abs_error": mean_abs_error(sumo, wm),
            "sumo_delta_mean": (sum(sumo) / len(sumo)) if sumo else None,
            "wm_delta_mean": (sum(wm) / len(wm)) if wm else None,
        }
    return out


def aggregate_near_cav(start_results) -> dict:
    vals: dict[str, list] = {_h(h): [] for h in NEAR_CAV_HORIZONS}
    for row in start_results:
        for arm in row["arms"].values():
            block = arm.get("near_cav_abs_ds")
            if isinstance(block, dict):
                for label in vals:
                    if _finite(block.get(label)):
                        vals[label].append(block[label])
    return {label: _median(values) for label, values in vals.items()}


def rollback_block(level, pairs, near_cav) -> dict:
    """Flat metrics the loop rollback rule reads (plus the non-gating near-CAV |ds|)."""
    label_l, label_p = _h(LEVEL_HORIZON_S), _h(PAIR_HORIZON_S)
    return {
        "pair_sign_agreement": (pairs.get(label_p) or {}).get("sign_agreement"),
        "pair_sign_agreement_horizon_s": int(PAIR_HORIZON_S),
        "delay_level_log_ratio": (level.get(label_l) or {}).get("log_ratio_total"),
        "delay_level_log_ratio_horizon_s": int(LEVEL_HORIZON_S),
        "near_cav_abs_ds_10": near_cav.get("10"),
        "near_cav_abs_ds_30": near_cav.get("30"),
    }


# --------------------------------------------------------------------------- parallel driver
def _init_worker(wm, device, entries) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    _WORKER["predictor"] = PredictorFactory(wm, device=device, entries=entries)()


def _map(items, worker, wm, args, workers: int) -> list:
    """Map ``worker(predictor_or_None, item)`` (a picklable parallel driver).

    ``workers <= 1`` builds the predictor in-process and passes it to ``worker``; otherwise
    each process builds its own predictor in :func:`_init_worker` and ``worker`` receives
    ``None`` so it reads its process-local predictor.
    """
    items = list(items)
    if workers <= 1:
        predictor = PredictorFactory(wm, device=args.device, entries=args.entries)()
        return [worker(predictor, item) for item in items]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(str(wm), args.device, args.entries)) as pool:
        futures = [pool.submit(worker, None, item) for item in items]   # submit all first (was serial)
        return [f.result() for f in futures]


def _work_start(predictor, payload):
    predictor = predictor if predictor is not None else _WORKER["predictor"]
    record, horizons, k, ph = payload
    return evaluate_start(predictor, record, horizons, k, ph)


def _work_pair(predictor, payload):
    predictor = predictor if predictor is not None else _WORKER["predictor"]
    record, horizons, k, ph = payload
    return evaluate_pair(predictor, record, horizons, k, ph)


# --------------------------------------------------------------------------- CLI
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", action="append", required=True, help="WM checkpoint (repeatable)")
    ap.add_argument("--states", help="states/windows JSON (or dir) for the delay LEVEL")
    ap.add_argument("--pairs", help="paired-branch states JSON for the paired DELTA")
    ap.add_argument("--selection", help="fixed delay-LEVEL selection JSON")
    ap.add_argument("--pairs-selection", help="fixed paired-branch selection JSON")
    ap.add_argument("--save-selection", help="write the deterministic level selection here")
    ap.add_argument("--save-pairs-selection", help="write the deterministic pair selection here")
    ap.add_argument("--select", type=int, default=DEFAULT_SELECT)
    ap.add_argument("--horizons", default="60,120,300")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--entries", choices=("off", "on", "queue"), default="queue")
    ap.add_argument("--sumo-pairs", help="parse-pair JSON providing total_D for the rollback")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--md", help="optional markdown summary path")
    return ap


def _total_d(path) -> float | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for key in ("total_D", "rollback"):
        block = doc.get(key) or {}
        if _finite(block.get("policy_minus_native")):
            return float(block["policy_minus_native"])
        if _finite(block.get("sumo_total_D_delta")):
            return float(block["sumo_total_D_delta"])
    return None


def _resolve_inputs(args):
    if args.selection and Path(args.selection).exists():
        starts = load_selection(args.selection, "states")
    else:
        if not args.states:
            raise SystemExit("--states is required when --selection is absent")
        starts = select_evenly(load_records(args.states), args.select)
        if not starts:
            raise SystemExit("no starts selected")
        if args.save_selection:
            write_selection(args.save_selection, starts, "states")

    if args.pairs_selection and Path(args.pairs_selection).exists():
        pairs = load_selection(args.pairs_selection, "pairs")
    elif args.pairs:
        pairs = select_evenly(group_pairs(load_records(args.pairs)), args.select, key=pair_key)
        if args.save_pairs_selection:
            write_selection(args.save_pairs_selection, pairs, "pairs")
    else:
        pairs = []
    return starts, pairs


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _ensure_src_on_path()
    for token in (args.out, args.md, args.states, args.pairs, args.selection,
                  args.pairs_selection, args.save_selection, args.save_pairs_selection,
                  args.sumo_pairs, *args.wm):
        if token:
            refuse_restricted(token)
    horizons = parse_horizons(args.horizons)
    if not horizons:
        raise SystemExit("--horizons must not be empty")

    starts, pairs = _resolve_inputs(args)
    ph = max(horizons)
    workers = max(1, int(args.workers))
    results: dict = {"schema": SCHEMA, "horizons": [_h(h) for h in horizons], "k": args.k,
                     "entries": args.entries, "n_starts": len(starts), "n_pairs": len(pairs),
                     "rollback_metrics": {"pair_horizon_s": int(PAIR_HORIZON_S),
                                          "level_horizon_s": int(LEVEL_HORIZON_S)},
                     "wms": {}}
    primary = None
    for wm in args.wm:
        start_payload = [(r, horizons, args.k, ph) for r in starts]
        pair_payload = [(r, horizons, args.k, ph) for r in pairs]
        start_results = _map(start_payload, _work_start, wm, args, workers)
        pair_results = _map(pair_payload, _work_pair, wm, args, workers) if pairs else []
        level = aggregate_level(start_results, horizons)
        paired = aggregate_pairs(pair_results, horizons) if pair_results else {}
        near_cav = aggregate_near_cav(start_results)
        results["wms"][Path(wm).name] = {"wm": str(wm), "delay_level": level,
                                         "paired": paired, "near_cav_abs_ds": near_cav}
        if primary is None:
            primary = rollback_block(level, paired, near_cav)
    primary = primary or rollback_block({}, {}, {})
    if args.sumo_pairs:
        total_d = _total_d(args.sumo_pairs)
        if total_d is not None:
            primary["sumo_total_D_delta"] = total_d
    results["rollback"] = primary
    results["primary_wm"] = Path(args.wm[0]).name if args.wm else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(render_markdown(results), encoding="utf-8")
    print(json.dumps({"out": str(out), "n_starts": results["n_starts"],
                      "n_pairs": results["n_pairs"], "rollback": results["rollback"]},
                     indent=2, sort_keys=True))
    return 0


def render_markdown(results: dict) -> str:
    lines = [f"# WM delay check (`{results['schema']}`)", "",
             f"- horizons: {', '.join(results['horizons'])} s; k={results['k']}; "
             f"entries={results['entries']}",
             f"- {results['n_starts']} starts, {results['n_pairs']} pairs", ""]
    for name, block in results.get("wms", {}).items():
        lines += [f"## {name}", "", "| horizon | WM total | SUMO total | log ratio |",
                  "|---|---|---|---|"]
        for label, row in block["delay_level"].items():
            lines.append(f"| {label} | {_fmt(row['wm_total'])} | {_fmt(row['sumo_total'])} | "
                         f"{_fmt(row['log_ratio_total'])} |")
        if block["paired"]:
            lines += ["", "| horizon | sign agreement | Spearman | MAE |", "|---|---|---|---|"]
            for label, row in block["paired"].items():
                lines.append(f"| {label} | {_fmt(row['sign_agreement'])} | "
                             f"{_fmt(row['spearman'])} | {_fmt(row['mean_abs_error'])} |")
        lines.append("")
    lines += ["## Rollback metrics", ""]
    for key, value in sorted(results.get("rollback", {}).items()):
        lines.append(f"- {key}: {_fmt(value)}")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
