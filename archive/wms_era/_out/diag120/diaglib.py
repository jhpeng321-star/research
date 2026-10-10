"""Shared helpers for the DIAG120 blow-up diagnosis (offline, CPU only).

Memory rule: state files are streamed one window at a time via the existing
``wms_eval.c1v2.iter_windows`` reader; nothing whole is loaded here.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

WS = Path(r"E:/rd-wt/ws-diag120")
WMS = WS / "output/wd01/wm_n1"
if str(WMS) not in sys.path:
    sys.path.insert(0, str(WMS))

#: Set file name -> (model, file) for every stored per-state score sheet.
SHIFT = Path(r"E:/rd-wt/ws-evalshift/_out/shift_scores")
C1EVAL = Path(r"E:/rd-wt/ws-c1eval/runs/active/C1-WMS-C1FAST-20261007")
RUN3_OFF = Path(r"E:/rd-wt/ws-eval2/_out/run3_c1_entries_off")
RUN3_ON = Path(r"E:/rd-wt/ws-eval2/_out/run3_c1_entries_on")
EVALH = Path(r"E:/rd-wt/ws-evalh/_out")

BLOW_60 = 50.0
BLOW_120 = 100.0


def load_states_file(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["states"] if isinstance(data, dict) else data


def blowup(state: dict, arm: str) -> bool:
    block = state.get("arms", {}).get(arm, {})
    if block.get("error"):
        return False
    car = block.get("car", {})
    ds60 = (car.get("60.0", {}).get("hdv", {}) or {}).get("ds_med")
    ds120 = (car.get("120.0", {}).get("hdv", {}) or {}).get("ds_med")
    if ds60 is not None and ds60 > BLOW_60:
        return True
    if ds120 is not None and ds120 > BLOW_120:
        return True
    return False


def car(state: dict, arm: str) -> dict:
    block = state.get("arms", {}).get(arm, {})
    return block.get("car", {}) if not block.get("error") else {}


def hdv_ds(state: dict, arm: str, horizon: float) -> tuple[float | None, float | None]:
    g = (car(state, arm).get(str(float(horizon)), {}).get("hdv", {}) or {})
    return g.get("ds_med"), g.get("ds_p90")


def hdv_dv(state: dict, arm: str, horizon: float) -> tuple[float | None, float | None]:
    g = (car(state, arm).get(str(float(horizon)), {}).get("hdv", {}) or {})
    return g.get("dv_med"), g.get("dv_p90")


def traffic_diff(state: dict, arm: str) -> dict:
    block = state.get("arms", {}).get(arm, {})
    if block.get("error"):
        return {}
    return block.get("traffic", {})


def model_files(model: str) -> dict[str, Path]:
    """Map named set -> stored per-state score file for a model label."""
    out: dict[str, Path] = {}
    if model == "c1_struct":
        for s in ("seen", "u_new", "u_same"):
            out[s] = SHIFT / "c1_struct" / f"states__c1_struct__{s}.json"
    elif model == "oldhist":
        for s in ("seen", "u_new", "u_same"):
            out[s] = SHIFT / "oldhist" / f"states__oldhist__{s}.json"
    else:
        for s in ("seen", "u_new", "u_same"):
            out[s] = SHIFT / model / f"states__{model}__{s}.json"
    return out


def finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)
