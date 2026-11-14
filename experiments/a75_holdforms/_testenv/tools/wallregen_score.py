"""Score world-model checkpoints on the REGENERATED unseen-wall sets (WALLREGEN; CPU, no SUMO).

Same per-state scorer as ``runs/active/WM-S-C4-20261013/score_refresh1/score_c4.py``
(``wms.eval.score.score_states``: CAVs teacher-forced from the recorded arm, HDVs predicted,
identity-matched |ds| / |dv| at 10/30/60/120 s, section delays at 60/120 s), pointed at a
``traffic_shepherd.c4.wallregen_states`` output directory instead of the frozen old sets.  The old
script's step-0 / baseline gates and C1/R4 caches are bound to the OLD sets and do not apply:
here every model (C1, R4, C4, v5 ...) is scored fresh on the new states and compared pairwise.

    python tools/wallregen_score.py --set-dir data_cache/wallc1t_states_h60 \\
        --ckpt C4=runs/.../wm_s_best.pt --ckpt C1=data_cache/ckpt_c1fast/wm_s.pt \\
        --out runs/active/<NEW_DIR> [--sets wall_same wall_pilot] [--benefit] [--workers 8]

Outputs ``<out>/results.json`` + ``<out>/SCORE.md``; per-state rows cached under ``<out>/cache``
(keyed by checkpoint sha and set-file sha) so a rerun only scores missing states.

``--benefit`` also scores ``wall_native.json`` and reports wall-BENEFIT agreement: recorded vs
predicted (native - wall) section delay (positive = the wall reduced delay), per horizon and
section, over (i) records that carry both arms (``paired``, e.g. the ``req`` variant) and (ii)
unpaired wall records matched to the native record of the same (world, t0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

HORIZONS = (10.0, 30.0, 60.0, 120.0)
ARM = "wall"
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")


def sha8(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def state_key(row: dict) -> tuple:
    return (str(row.get("world")), str(row.get("key")), str(row.get("source_arm")),
            round(float(row.get("t0_s", 0)), 3))


def tested_arm(row: dict, prefer: str = ARM) -> str:
    arms = row.get("arms") or {}
    return prefer if prefer in arms else next(iter(arms))


def iter_records(path: Path, limit=None) -> list[dict]:
    from wms.eval.normalize import iter_windows
    return list(iter_windows(path, limit=limit))


def score_cached(tag: str, ckpt: Path, set_path: Path, records: list, workers: int, cache: Path,
                 entries: bool) -> list:
    from wms.eval.run import CheckpointPredictorFactory
    from wms.eval.score import ScoreSpec, score_states
    cache_file = cache / f"{tag}__{set_path.stem}__{sha8(ckpt)}__{sha8(set_path)}" \
                         f"{'__entries' if entries else ''}.json"
    done = {}
    if cache_file.is_file():
        try:
            done = {state_key(r): r for r in json.loads(cache_file.read_text("utf-8"))["states"]}
        except (OSError, json.JSONDecodeError, KeyError):
            done = {}
    todo = [r for r in records if state_key(r) not in done]
    print(f"{tag} on {set_path.stem}: {len(todo)} to score, {len(done)} cached", flush=True)
    if todo:
        factory = CheckpointPredictorFactory(str(ckpt), entries=entries, entries_v2=False)
        spec = ScoreSpec(horizons_s=HORIZONS, predict_horizon_s=120.0, seed=0)
        for row in score_states(todo, factory, spec, workers=workers, inflight=workers * 2):
            done[state_key(row)] = row
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"states": [done[state_key(r)] for r in records
                                              if state_key(r) in done]},
                                  separators=(",", ":")), encoding="utf-8")
        tmp.replace(cache_file)
    return [done[state_key(r)] for r in records if state_key(r) in done]


def stats_for(rows: list) -> dict:
    from wms.eval.stats import summarize
    out = {}
    for h in HORIZONS:
        out[str(int(h))] = {}
        for metric, field in (("ds", "ds_med"), ("dv", "dv_med")):
            vals = []
            for row in rows:
                blk = ((row.get("arms") or {}).get(tested_arm(row)) or {}).get("car", {}) \
                    .get(str(h), {}).get("hdv", {})
                v = blk.get(field)
                if v is not None and math.isfinite(v):
                    vals.append(float(v))
            out[str(int(h))][metric] = summarize(vals)
    blows = n = 0
    errors = 0
    for row in rows:
        arm = (row.get("arms") or {}).get(tested_arm(row)) or {}
        if arm.get("error"):
            errors += 1
            continue
        car = arm.get("car", {})
        a = car.get("60.0", {}).get("hdv", {}).get("ds_med")
        b = car.get("120.0", {}).get("hdv", {}).get("ds_med")
        if a is not None and b is not None:
            n += 1
            blows += int(a > 50 and b > 100)
    out["blowups"] = {"n": blows, "denominator": n,
                      "rule": "HDV median |ds| >50m at 60s AND >100m at 120s"}
    out["errors"] = errors
    return out


def _spearman(x, y):
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


BENEFIT_HORIZONS = (30.0, 60.0, 120.0)
BENEFIT_SECTIONS = ("road", "upstream", "merge", "downstream")
MERGE_S0, MERGE_S1 = 0.0, 75.44          # wms.eval.metrics.SECTIONS
DT = 0.5


def hdv_delay(frames: list, t0: float, v0: float) -> dict:
    """HDV-only normalised delay (vehicle-seconds) of the t0 roster over [t0, t0 + h), by section.

    CAVs are excluded: they are teacher-forced in both arms (identical predicted and recorded),
    so including them would make any model look perfect on the wall's own slow-down."""
    out = {str(h): {sec: 0.0 for sec in BENEFIT_SECTIONS} for h in BENEFIT_HORIZONS}
    for fr in frames:
        dt_rel = float(fr["time_s"]) - t0
        for row in fr.get("vehicles", []):
            if row.get("is_cav") or not row.get("present", True):
                continue
            d = max(0.0, v0 - float(row["v"])) / v0 * DT
            s = float(row["s"])
            sec = "upstream" if s < MERGE_S0 else ("merge" if s < MERGE_S1 else "downstream")
            for h in BENEFIT_HORIZONS:
                if -1e-9 <= dt_rel < h - 1e-9:
                    out[str(h)]["road"] += d
                    out[str(h)][sec] += d
    return out


_BW: dict = {}


def _b_init(ckpt: str, entries: bool) -> None:
    import torch
    torch.set_num_threads(1)
    from wms.eval.predictor import RolloutFlags, WmsPredictor
    _BW["p"] = WmsPredictor.from_checkpoint(ckpt, RolloutFlags(entries=entries, entries_v2=False))


def _b_work(payload: tuple) -> dict:
    wall_rec, nat_rec = payload
    from wms.eval.normalize import recorded_for
    t0 = float(wall_rec["t0_s"])
    v0 = float(wall_rec.get("v0_mps", 27.78))
    kind = "unpaired" if nat_rec is not None else "paired"
    item = {"world": wall_rec["world"], "t0_s": t0, "source_arm": wall_rec.get("source_arm"),
            "key": wall_rec.get("key"), "kind": kind,
            "wall_exposure_status": wall_rec.get("wall_exposure_status"),
            "wall_phase_at_t0": wall_rec.get("wall_phase_at_t0"), "error": None, "b": {}}
    try:
        delays = {}
        for arm, src in (("wall", wall_rec), ("native", nat_rec or wall_rec)):
            pred = _BW["p"].predict_states(src, {**src.get("cav_controllers", {}), "arm": arm},
                                           120.0, 0)
            delays[arm] = {"predicted": hdv_delay(pred, t0, v0),
                           "recorded": hdv_delay(recorded_for(src, arm), t0, v0)}
        for h in map(str, BENEFIT_HORIZONS):
            for sec in BENEFIT_SECTIONS:
                item["b"].setdefault(h, {})[sec] = {
                    src: delays["native"][src][h][sec] - delays["wall"][src][h][sec]
                    for src in ("recorded", "predicted")}
        item["delays"] = delays
    except Exception as exc:  # noqa: BLE001 - recorded per item
        import traceback
        item["error"] = f"{type(exc).__name__}: {exc}: {traceback.format_exc()[-400:]}"
    return item


def benefit_items(ckpt: Path, wall_records: list, native_records: list, workers: int,
                  entries: bool) -> list:
    """Wall benefit = native - wall HDV delay (positive = the wall reduced HDV delay).

    paired: the record carries both arms from one identical t0 state (``req`` variant; late arms
    of ``h60``); unpaired: the native record of the same (world, t0) - different initial
    physical states, so the difference mixes the wall effect with the state difference."""
    from concurrent.futures import ProcessPoolExecutor
    by_wt = {(str(r["world"]), round(float(r["t0_s"]), 3)): r for r in native_records}
    jobs = []
    for r in wall_records:
        if "native" in (r.get("recorded") or {}):
            jobs.append((r, None))
        else:
            n = by_wt.get((str(r["world"]), round(float(r["t0_s"]), 3)))
            if n is not None:
                jobs.append((r, n))
    if not jobs:
        return []
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_b_init,
                             initargs=(str(ckpt), entries)) as pool:
        return list(pool.map(_b_work, jobs, chunksize=1))


def benefit_summary(items: list) -> dict:
    out = {}
    ok = [i for i in items if not i.get("error")]
    for kind in ("paired", "unpaired"):
        sel = [i for i in ok if i["kind"] == kind]
        blk = {}
        for h in map(str, BENEFIT_HORIZONS):
            for sec in BENEFIT_SECTIONS:
                rec = np.array([i["b"][h][sec]["recorded"] for i in sel])
                pred = np.array([i["b"][h][sec]["predicted"] for i in sel])
                if not len(rec):
                    continue
                blk[f"{h}/{sec}"] = {
                    "n": int(len(rec)), "rec_mean": float(rec.mean()),
                    "pred_mean": float(pred.mean()),
                    "sign_agree": float(np.mean(np.sign(np.round(rec, 6))
                                                == np.sign(np.round(pred, 6)))),
                    "spearman": _spearman(rec, pred),
                    "mae": float(np.abs(rec - pred).mean()),
                    "n_wall_better_rec": int((rec > 1e-6).sum()),
                    "n_wall_better_pred": int((pred > 1e-6).sum())}
        out[kind] = blk
    out["n_errors"] = len(items) - len(ok)
    return out


def fmt(v, d=2):
    return "n/a" if v is None else f"{v:.{d}f}"


def write_md(res: dict, path: Path) -> None:
    lines = [f"# WALLREGEN unseen-wall scoring - {res['set_dir']}", "",
             f"population: {res.get('population')}; variant: {res.get('variant')}; "
             f"CPU, teacher-forced CAVs, entries {'ON' if res['entries'] else 'off'}; "
             "median over states of the per-state HDV median |ds| (m).", "",
             "| set | model | n | 10 s | 30 s | 60 s | 120 s | blow-ups | errors |",
             "|---|---|---|---|---|---|---|---|---|"]
    for set_name, models in res["models"].items():
        for tag, blk in models.items():
            s = blk["summary"]
            cells = [fmt(s[str(int(h))]["ds"]["median"]) for h in HORIZONS]
            lines.append(f"| {set_name} | {tag} | {blk['n_states']} | " + " | ".join(cells)
                         + f" | {s['blowups']['n']}/{s['blowups']['denominator']} | {s['errors']} |")
    if res.get("paired"):
        lines += ["", "## Paired median difference (model - reference), 95% CI over worlds",
                  "| set | pair | 10 s | 30 s | 60 s | 120 s |", "|---|---|---|---|---|---|"]
        for set_name, pairs in res["paired"].items():
            for name, blk in pairs.items():
                cells = []
                for h in HORIZONS:
                    b = blk[str(int(h))]
                    cells.append(f"{fmt(b.get('point'))} [{fmt(b.get('lo'))}, {fmt(b.get('hi'))}]"
                                 if b and b.get("point") is not None else "n/a")
                lines.append(f"| {set_name} | {name} | " + " | ".join(cells) + " |")
    if res.get("benefit"):
        lines += ["", "## Wall benefit: native - wall HDV delay of the t0 roster (veh-s; positive = "
                  "wall better); CAVs excluded (teacher-forced)",
                  "| model | pairs | h/section | n | rec mean | pred mean | sign agree | rho | MAE |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tag, blk in res["benefit"].items():
            for kind in ("paired", "unpaired"):
                for key, bb in blk["summary"].get(kind, {}).items():
                    if not (key.endswith("/road") or key.endswith("/merge")):
                        continue
                    lines.append(f"| {tag} | {kind} | {key} | {bb['n']} | {fmt(bb['rec_mean'])} | "
                                 f"{fmt(bb['pred_mean'])} | {fmt(bb['sign_agree'])} | "
                                 f"{fmt(bb['spearman'])} | {fmt(bb['mae'])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set-dir", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="TAG=path (repeatable); the "
                    "first is the reference of the paired differences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["wall_same", "wall_pilot"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N states per set")
    ap.add_argument("--benefit", action="store_true")
    ap.add_argument("--entries", action="store_true", help="entries v1 ON (RSU counter process)")
    a = ap.parse_args(argv)
    set_dir, out = Path(a.set_dir), Path(a.out)
    for p in (set_dir, out):
        if any(t in str(p).replace("\\", "/").lower() for t in FORBIDDEN):
            raise SystemExit(f"restricted path token in {p}")
    models = {}
    for spec in a.ckpt:
        tag, _, path = spec.partition("=")
        if not path or not Path(path).is_file():
            raise SystemExit(f"bad --ckpt {spec!r}")
        models[tag] = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = {}
    rp = set_dir / "BUILD_RECEIPT.json"
    if rp.is_file():
        receipt = json.loads(rp.read_text(encoding="utf-8"))
    res = {"schema": "c4_wallregen_score/1", "set_dir": str(set_dir), "entries": a.entries,
           "population": receipt.get("population"), "variant": receipt.get("variant"),
           "checkpoints": {t: {"path": str(p), "sha8": sha8(p)} for t, p in models.items()},
           "models": {}, "paired": {}, "runtimes_s": {}}
    from wms.eval.report import paired_bootstrap
    scored: dict = {}
    for set_name in a.sets:
        path = set_dir / f"{set_name}.json"
        records = iter_records(path, limit=a.limit)
        res["models"][set_name] = {}
        for tag, ckpt in models.items():
            t0 = time.time()
            rows = score_cached(tag, ckpt, path, records, a.workers, out / "cache", a.entries)
            scored[(set_name, tag)] = rows
            res["models"][set_name][tag] = {"n_states": len(rows), "summary": stats_for(rows),
                                            "n_worlds": len({r["world"] for r in rows})}
            res["runtimes_s"][f"{tag}__{set_name}"] = round(time.time() - t0, 1)
        tags = list(models)
        for tag in tags[1:]:
            res["paired"].setdefault(set_name, {})[f"{tag}_minus_{tags[0]}"] = {
                str(int(h)): paired_bootstrap(scored[(set_name, tag)], scored[(set_name, tags[0])],
                                              ARM, h, n_boot=2000, seed=0) for h in HORIZONS}
    if a.benefit:
        res["benefit"] = {}
        bench_set = "wall_same" if "wall_same" in a.sets else a.sets[0]
        wall_records = iter_records(set_dir / f"{bench_set}.json", limit=a.limit)
        nat_path = set_dir / "wall_native.json"
        nat_records = iter_records(nat_path) if nat_path.is_file() else []
        for tag, ckpt in models.items():
            t0 = time.time()
            items = benefit_items(ckpt, wall_records, nat_records, a.workers, a.entries)
            res["benefit"][tag] = {"set": bench_set, "summary": benefit_summary(items),
                                   "items": items}
            res["runtimes_s"][f"benefit__{tag}"] = round(time.time() - t0, 1)
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_md(res, out / "SCORE.md")
    print((out / "SCORE.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
