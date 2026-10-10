"""Shared IO + statistics helpers for the WD-01-PB-RESCORE1 analyses.

Zero simulation; reads only the label root built by
rescore1.build_truearms_labels (TRAIN/CAL sources by construction) and
frozen read-only roots.  HOLDOUT never appears in any selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (WD01, WD01 / "partB_v0data", WD01 / "partA_data",
           WD01 / "part4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import wdpaths  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
LABEL_ROOT = RD / "output" / "wd01_partB_truearms_labels_v1"
V032_ROOT = RD / "output" / "wd01_partB_v0data_v032"
RESULTS = WD01 / "partB_rescore1"
TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
T_OFFSET_S = 10.0          # branch decision grid starts at t0 + 10 s
SEED = 20260930
N_BOOT = 2000

# canonical source -> policy label of the RESCORE1 label root
CANONICAL = {"base": "damping_fleet", "native": "switching_mix",
             "pi": "switching_pi_arm", "truearms_native": "true_native",
             "truearms_pi": "true_pi"}


def load_label_manifest(root: Path = LABEL_ROOT) -> dict:
    return json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))


def label_frame(root: Path, chunk_id: str, world: str, source: str,
                cache: dict) -> pd.DataFrame:
    key = (str(root), chunk_id, world, source)
    if key not in cache:
        p = root / "worlds" / chunk_id / f"{world}__{source}_labels.parquet"
        cache[key] = pd.read_parquet(p)
    return cache[key]


def y_at(df: pd.DataFrame, t: float, tau: float,
         col: str = "total_delay_ref_s") -> float | None:
    g = df[np.isclose(df["t"], t) & np.isclose(df["tau"], tau)]
    if g.empty:
        return None
    return float(g[col].iloc[0])


def world_rows(man: dict) -> list[dict]:
    """[(chunk_id, world, split, demand, penetration, t0)] TRAIN/CAL only
    (the label root lists no other splits; asserted)."""
    out = []
    for key, e in sorted(man["worlds"].items()):
        assert e["split"] in ("TRAIN", "CALIBRATION"), \
            f"non-TRAIN/CAL source in the label root: {key}"
        row = (e["chunk_id"], e["world"], e["split"],
               int(e["demand_veh_h"]), float(e["penetration"]),
               float(e["t0_s"]))
        if row not in out:
            out.append(row)
    return out


# ------------------------------------------------------------ stats ----

def sign_acc(pred: np.ndarray, real: np.ndarray) -> float:
    ok = np.sign(np.asarray(pred)) == np.sign(np.asarray(real))
    nz = ~(np.asarray(pred) == 0)
    if not nz.any():
        return float("nan")
    return float(ok[nz].mean())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2:
        return float("nan")

    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        sx = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def world_bootstrap_ci(worlds: list, ok: np.ndarray,
                       n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """90% CI of mean(ok) under world-level resampling (pairs move
    together with their world)."""
    worlds = np.asarray(worlds)
    uniq = sorted(set(worlds.tolist()))
    idx_of = {w: np.flatnonzero(worlds == w) for w in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx_of[uniq[i]] for i in pick])
        stats[b] = ok[sel].mean()
    return (float(np.quantile(stats, 0.05)),
            float(np.quantile(stats, 0.95)))


def agg(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    return dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                std=float(x.std()) if len(x) else None,
                q25=float(np.quantile(x, 0.25)) if len(x) else None,
                q75=float(np.quantile(x, 0.75)) if len(x) else None)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def r2(x, nd=1):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None
