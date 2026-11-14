"""Traffic-state split for the shifted C1 v2 baseline report.

Groups every shifted state by its t0 near-merge HDV speed and reports the
per-model HDV |ds| median at 60 s (and 120 s where present) plus the paired
median difference against ``c1_struct`` on the shared states.  Read-only over
the per-state result files and the shifted state sets; never loads a whole set
into RAM (streams the state files, which is all that is needed for the t0 speed
proxy).

Definition used here (reproducible from the state records alone): ``v_hdv0`` is
the mean speed of the non-CAV ("HDV") vehicles in the initial frame; classes are
free ``> 20``, congested ``10..20``, jam ``< 10`` m/s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

MODELS = ("c1_struct", "blackbox_traj", "blackbox_id", "keep_speed",
          "pure_physics", "oldhist")
SETS = ("u_same", "u_new", "seen")
ARMS = {"u_same": "pi", "u_new": "pi", "seen": "native"}
CLASSES = ("free", "congested", "jam")
#: Near-merge band in the ``s_down`` frame (merge section is [0, 75.44] m); the
#: same wide band reproduces the reference congestion mix on this corpus.
NEAR_MERGE_BAND_M = (-300.0, 300.0)
DEFAULT_HORIZON = 60.0


def vnear_of(record: dict) -> float | None:
    """Mean t0 HDV speed in the near-merge band (``s_down`` in [-300, 300] m).

    Falls back to every modelled HDV, then to every present vehicle, when the
    band holds no HDV.
    """
    vehicles = (record.get("initial_state") or {}).get("vehicles") or []
    hdv = [v for v in vehicles if v.get("present", True) and not v.get("is_cav", False)]
    lo, hi = NEAR_MERGE_BAND_M
    near = [v for v in hdv if lo <= float(v.get("s", -1e18)) <= hi]
    for group in (near, hdv, [v for v in vehicles if v.get("present", True)]):
        speeds = [float(v.get("v", 0.0)) for v in group]
        if speeds:
            return sum(speeds) / len(speeds)
    return None


def classify(vnear: float | None) -> str | None:
    if vnear is None:
        return None
    if vnear > 20.0:
        return "free"
    if vnear >= 10.0:
        return "congested"
    return "jam"


def _stream_windows(path: Path):
    from .c1v2 import iter_windows

    yield from iter_windows(path)


def load_vnear(states_path: Path) -> dict[tuple[str, float], float]:
    out = {}
    for record in _stream_windows(states_path):
        value = vnear_of(record)
        if value is not None:
            out[(str(record.get("world")), round(float(record.get("t0_s", 0.0)), 3))] = value
    return out


def load_ds(result_path: Path, arm: str, horizon: float) -> dict[tuple[str, float], float]:
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    out = {}
    for state in data.get("states", []):
        block = (state.get("arms") or {}).get(arm) or {}
        if block.get("error"):
            continue
        value = ((block.get("car") or {}).get(str(float(horizon)), {})
                 .get("hdv", {}) or {}).get("ds_med")
        if value is None:
            continue
        out[(str(state.get("world")), round(float(state.get("t0_s", 0.0)), 3))] = float(value)
    return out


def build(states_dir: Path, scores_dir: Path, *, horizon: float) -> dict:
    vnear = {name: load_vnear(states_dir / f"{name}.json") for name in SETS}
    report = {"schema": "wms_eval_shift_traffic_split/1", "horizon_s": float(horizon),
              "vnear_definition": (f"mean t0 speed of non-CAV vehicles with s_down in "
                                   f"[{NEAR_MERGE_BAND_M[0]:g}, {NEAR_MERGE_BAND_M[1]:g}] m "
                                   "(fallback: all HDVs); free >20, congested 10..20, "
                                   "jam <10 m/s"),
              "sets": {}}
    for set_name in SETS:
        arm = ARMS[set_name]
        ds = {m: load_ds(scores_dir / m / f"states__{m}__{set_name}.json", arm, horizon)
              for m in MODELS}
        ref = ds["c1_struct"]
        block: dict = {"arm": arm, "classes": {}}
        for cls in CLASSES:
            keys = [k for k, v in vnear[set_name].items() if classify(v) == cls]
            entry: dict = {"n": len(keys), "models": {}}
            for m in MODELS:
                values = [ds[m][k] for k in keys if k in ds[m]]
                entry["models"][m] = (median(values) if values else None)
            diffs = [ds[m][k] - ref[k] for k in keys if k in ds[m] and k in ref]
            entry["models_diff_vs_c1_struct"] = {}
            for m in MODELS:
                d = [ds[m][k] - ref[k] for k in keys if k in ds[m] and k in ref]
                entry["models_diff_vs_c1_struct"][m] = median(d) if d else None
            entry["n_paired"] = len(diffs)
            block["classes"][cls] = entry
        report["sets"][set_name] = block
    return report


def markdown(report: dict) -> str:
    lines = ["", f"## Traffic-state split (HDV |ds| median at {report['horizon_s']:g} s)",
             "",
             f"t0 state class from `{report['vnear_definition']}`. The paired "
             "column is the median of `model - c1_struct` over the same states "
             "(negative favours the model).", ""]
    for set_name, block in report["sets"].items():
        lines += [f"### `{set_name}` (arm `{block['arm']}`)", "",
                  "| class | n | c1_struct | oldhist | blackbox_traj | blackbox_id | "
                  "keep_speed | pure_physics | oldhist - c1_struct |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cls in CLASSES:
            entry = block["classes"][cls]
            m = entry["models"]

            def f(x):
                return "n/a" if x is None else f"{x:.2f}"
            lines.append(
                f"| {cls} | {entry['n']} | {f(m['c1_struct'])} | {f(m['oldhist'])} "
                f"| {f(m['blackbox_traj'])} | {f(m['blackbox_id'])} "
                f"| {f(m['keep_speed'])} | {f(m['pure_physics'])} "
                f"| {f(entry['models_diff_vs_c1_struct']['oldhist'])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="shift_traffic_split")
    parser.add_argument("--states-dir", required=True)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--append-md")
    args = parser.parse_args(argv)
    report = build(Path(args.states_dir), Path(args.scores_dir), horizon=args.horizon)
    Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    text = markdown(report)
    if args.append_md:
        with Path(args.append_md).open("a", encoding="utf-8") as fh:
            fh.write(text)
    print(json.dumps({"out_json": args.out_json,
                      "classes": {s: {c: report["sets"][s]["classes"][c]["n"]
                                      for c in CLASSES} for s in report["sets"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
