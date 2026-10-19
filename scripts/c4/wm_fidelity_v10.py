"""Score a C4 world-model checkpoint on a FIXED set of TRAIN-val states (v10 loop fidelity).

This is a small CPU-or-GPU tool used by ``scripts/c4/loop_v10.py`` (and its config
``configs/c4_dreamer_loop_v10loop.json``, which reads the key path ``hdv.abs_ds.60``).  It
mirrors ``tools/wallregen_score.py``: it streams state records with
``wms.eval.normalize.iter_windows``, scores them with ``wms.eval.score.score_states`` and a
``ScoreSpec`` at 10/30/60/120 s, and reports the median over states of each state's HDV median
``|ds|`` plus the blow-up count (the rule from ``wallregen_score.stats_for``).

    python scripts/c4/wm_fidelity_v10.py --wm runs/.../wm_s_best.pt \\
        --states /path/to/fixed_train_val_states.json \\
        --selection data_cache/v10_fixed_selection.json --save-selection ... \\
        --horizons 10,30,60,120 --arm native --device cpu --workers 1 \\
        --entries off --out runs/.../wm_fidelity.json

The selection is deterministic: the same input always yields the same ``--select`` records,
evenly spaced across the sorted ``(world, key, source_arm, t0_s)`` keys.

Refuses any input/output path containing a restricted token (sealed/holdout/heldout/cal_split/
chunk5/j4b/inc1).  The module is import side-effect free: torch/wms are imported lazily inside
functions, so the pure helpers (``select_states`` / ``summarize``) import without torch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "c4_wm_fidelity_v10/1"
SELECTION_SCHEMA = "c4_wm_fidelity_v10/selection/1"
DEFAULT_HORIZONS = (10.0, 30.0, 60.0, 120.0)
DEFAULT_SELECT = 48
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
BLOWUP_RULE = "HDV median |ds| >50m at 60s AND >100m at 120s"


# --------------------------------------------------------------------------- pure helpers
def refuse_restricted(path) -> str:
    """Return ``str(path)`` unless it contains a restricted token (then raise ``SystemExit``)."""
    text = str(path)
    if any(token in text.replace("\\", "/").lower() for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {text}")
    return text


def selection_key(record: dict) -> tuple[str, str, str, float]:
    """The deterministic ordering key of one state record."""
    return (str(record.get("world")), str(record.get("key")),
            str(record.get("source_arm")), round(float(record.get("t0_s", 0.0)), 3))


def select_states(records, n: int) -> list[dict]:
    """Pick ``n`` records evenly spaced across the sorted selection keys.

    The same input always yields the same records; fewer than ``n`` records are all returned.
    """
    ordered = sorted(records, key=selection_key)
    total = len(ordered)
    if n <= 0 or total == 0:
        return []
    if total <= n:
        return ordered
    if n == 1:
        return [ordered[total // 2]]
    return [ordered[round(i * (total - 1) / (n - 1))] for i in range(n)]


def parse_horizons(text) -> tuple[float, ...]:
    """"10,30,60,120" -> ``(10.0, 30.0, 60.0, 120.0)``."""
    if isinstance(text, (list, tuple)):
        return tuple(float(h) for h in text)
    return tuple(float(part) for part in str(text).replace(";", ",").split(",") if part.strip())


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def _median(values) -> float | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _horizon_labels(horizons) -> list[str]:
    return [str(int(h)) for h in horizons]


def _arm_block(row: dict, arm: str) -> dict:
    return ((row.get("arms") or {}).get(arm)) or {}


def _hdv_ds_med(block: dict, horizon: float):
    car = block.get("car") or {}
    return ((car.get(str(float(horizon))) or {}).get("hdv") or {}).get("ds_med")


def default_arm(rows, prefer: str = "native") -> str:
    """``prefer`` if present in the first row that has arms, else that row's first arm."""
    for row in rows:
        arms = row.get("arms") or {}
        if arms:
            return prefer if prefer in arms else next(iter(sorted(arms)))
    return prefer


def summarize(rows, arm: str, horizons, wm=None) -> dict:
    """Reduce ``score_states`` rows to the fidelity JSON (HDV median |ds| per horizon + blowups)."""
    rows = list(rows)
    labels = _horizon_labels(horizons)
    errors = 0
    medians: dict[str, list[float]] = {label: [] for label in labels}
    for row in rows:
        block = _arm_block(row, arm)
        if not block or block.get("error"):
            errors += 1
            continue
        for horizon, label in zip(horizons, labels):
            value = _hdv_ds_med(block, horizon)
            if _finite(value):
                medians[label].append(float(value))
    abs_ds = {label: _median(vals) for label, vals in medians.items()}
    blows = denominator = 0
    for row in rows:
        block = _arm_block(row, arm)
        if not block or block.get("error"):
            continue
        a = _hdv_ds_med(block, 60.0)
        b = _hdv_ds_med(block, 120.0)
        if _finite(a) and _finite(b):
            denominator += 1
            blows += int(float(a) > 50 and float(b) > 100)
    return {
        "schema": SCHEMA,
        "wm": wm,
        "arm": arm,
        "n_states": len(rows),
        "horizons": {label: {"hdv_abs_ds_med": abs_ds[label]} for label in labels},
        "hdv": {"abs_ds": abs_ds},
        "blowups": {"n": blows, "denominator": denominator, "rule": BLOWUP_RULE},
        "errors": errors,
    }


# --------------------------------------------------------------------------- loading / scoring
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


def load_selection(path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise SystemExit(f"selection file {path} carries no states")
    return states


def write_selection(path, records) -> None:
    payload = {"schema": SELECTION_SCHEMA, "n": len(records),
               "keys": [list(selection_key(r)) for r in records], "states": list(records)}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class PredictorFactory:
    """Picklable factory of ``WmsPredictor.predict_states`` honouring device and entries."""

    def __init__(self, path, device: str = "cpu", entries: str = "off") -> None:
        self.path = str(path)
        self.device = str(device)
        self.entries = str(entries)

    def __call__(self):
        from wms.eval.predictor import RolloutFlags, WmsPredictor
        if self.entries == "queue":
            flags = RolloutFlags(entries=True, entries_v2=True, entry_queue_state=True)
        else:
            flags = RolloutFlags(entries=False, entries_v2=False)
        return WmsPredictor.from_checkpoint(self.path, flags,
                                            device=self.device).predict_states


def score_records(records, wm, horizons, device: str, workers: int, entries: str) -> list[dict]:
    from wms.eval.score import ScoreSpec, score_states
    spec = ScoreSpec(horizons_s=tuple(horizons), predict_horizon_s=max(120.0, max(horizons)),
                     seed=0)
    factory = PredictorFactory(wm, device=device, entries=entries)
    return score_states(records, factory, spec, workers=workers)


# --------------------------------------------------------------------------- CLI
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wm", required=True, help="world-model checkpoint to score")
    ap.add_argument("--states", help="a states/windows JSON file, or a directory of *.json files")
    ap.add_argument("--selection", help="fixed selection JSON; if it exists its states are scored")
    ap.add_argument("--save-selection", help="write the deterministic selection here")
    ap.add_argument("--select", type=int, default=DEFAULT_SELECT, help="number of states to select")
    ap.add_argument("--horizons", default="10,30,60,120")
    ap.add_argument("--arm", default=None, help="arm to report (default: native, else the first)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--entries", choices=("off", "queue"), default="off")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    _ensure_src_on_path()
    for token in (args.wm, args.states, args.selection, args.save_selection, args.out):
        if token:
            refuse_restricted(token)
    horizons = parse_horizons(args.horizons)
    if not horizons:
        raise SystemExit("--horizons must not be empty")

    selection = Path(args.selection) if args.selection else None
    if selection is not None and selection.exists():
        records = load_selection(selection)
    else:
        if not args.states:
            raise SystemExit("--states is required when --selection is absent")
        records = select_states(load_records(args.states), args.select)
        if not records:
            raise SystemExit("no states selected")
    if args.save_selection:
        write_selection(args.save_selection, records)

    rows = score_records(records, args.wm, horizons, args.device, args.workers, args.entries)
    arm = args.arm or default_arm(rows)
    result = summarize(rows, arm, horizons, wm=str(args.wm))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "arm": arm, "n_states": result["n_states"],
                      "hdv_abs_ds": result["hdv"]["abs_ds"], "blowups": result["blowups"],
                      "errors": result["errors"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
