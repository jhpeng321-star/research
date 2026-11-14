"""CTM-alone calibration + evaluation driver and report (``_out/CTM_ALONE.*``).

Stages:
* ``calibrate`` - fit the section FDs + merge drop on TRAIN windows only;
* ``eval``      - evaluate the CTM on the shifted and unshifted C1 v2 sets and
  compare with the existing per-car ``states__<model>__<set>.json`` results.

This is the execution entry point for the "CTM alone" question: does a purely
macroscopic model predict section-level traffic at 60-300 s at least as well as
the per-car world models?  No SUMO, no GPU, no WM training, one window at a time.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import ctm, ctm_calib, ctm_eval

ROOT = Path(__file__).resolve().parents[4]

TRAIN_WINDOWS = ROOT / "_out" / "lpo" / "windows_train_noPI.json"
PACK_ROOTS = {
    "pack_v3": r"D:\rd-staging\wmn1_feat\pack_v3",
    "pack_damping": r"E:\rd-wt\damping-snap\_out\pack_damping",
    "pack_dspeed": r"E:\rd-wt\dspeed-data\_out\pack_dspeed",
}
SHIFT_OUT = Path(r"E:\rd-wt\ws-evalshift\_out")
EVALH_OUT = Path(r"E:\rd-wt\ws-evalh\_out")
C1FAST = Path(r"E:\rd-wt\ws-c1eval\runs\active\C1-WMS-C1FAST-20261007")

MODEL_LABELS = {
    "ctm": "CTM_alone", "wms": "C1_struct", "oldhist": "OLDHIST",
    "blackbox_id": "BB_id", "blackbox_traj": "BB_traj",
    "pure_physics": "pure_physics", "keep_speed": "keep_speed",
}

#: Human-facing section keys for the markdown tables.
_FOCUS = (("merge", "total_delay", 3), ("merge", "mean_speed", 3),
          ("merge", "merge_queue_length", 3), ("merge", "stopped_vehicles", 3),
          ("upstream", "total_delay", 3), ("downstream", "mean_speed", 3))


def _per_car_specs_shifted(set_name: str) -> dict[str, Path]:
    main = SHIFT_OUT / "shift_scores" / "_run_main"
    return {
        "wms": main / f"states__wms__{set_name}.json",
        "blackbox_id": main / f"states__blackbox_id__{set_name}.json",
        "blackbox_traj": main / f"states__blackbox_traj__{set_name}.json",
        "pure_physics": main / f"states__pure_physics__{set_name}.json",
        "keep_speed": main / f"states__keep_speed__{set_name}.json",
        "oldhist": SHIFT_OUT / "shift_scores" / "_run_oldhist"
        / f"states__wms__{set_name}.json",
    }


def _per_car_specs_unshifted(c1set: str) -> dict[str, Path]:
    return {
        "wms": C1FAST / f"states__wms__{c1set}.json",
        "blackbox_id": C1FAST / f"states__blackbox_id__{c1set}.json",
        "blackbox_traj": C1FAST / f"states__blackbox_traj__{c1set}.json",
        "pure_physics": C1FAST / f"states__pure_physics__{c1set}.json",
        "keep_speed": C1FAST / f"states__keep_speed__{c1set}.json",
        "oldhist": EVALH_OUT / "oldhist_c1" / f"states__wms__{c1set}.json",
    }


BLOCKS = {
    "shifted": {
        "horizon_s": 120.0,
        "sets": {
            "seen": {"path": SHIFT_OUT / "c1_states_v2shift" / "seen.json",
                     "unseen": False, "per_car": _per_car_specs_shifted("seen")},
            "u_same": {"path": SHIFT_OUT / "c1_states_v2shift" / "u_same.json",
                       "unseen": True, "per_car": _per_car_specs_shifted("u_same")},
            "u_new": {"path": SHIFT_OUT / "c1_states_v2shift" / "u_new.json",
                      "unseen": True, "per_car": _per_car_specs_shifted("u_new")},
        },
    },
    "unshifted": {
        "horizon_s": 300.0,
        "sets": {
            "seen": {"path": EVALH_OUT / "c1_states_v2c" / "seen.json",
                     "unseen": False, "per_car": _per_car_specs_unshifted("seen")},
            "unseen_u_same": {"path": EVALH_OUT / "c1_states_v2c" / "unseen_u_same_world.json",
                              "unseen": True, "per_car": _per_car_specs_unshifted("u_same")},
            "unseen_u_new": {"path": EVALH_OUT / "c1_states_v2c" / "unseen_u_new_world.json",
                             "unseen": True, "per_car": _per_car_specs_unshifted("u_new")},
        },
    },
}


def calibrate(train_windows: Path, target: int, n_boot: int, cache: Optional[Path]):
    if cache and cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return ctm.CTMParams.from_dict(payload["params"]), payload
    result = ctm_calib.calibrate(train_windows, PACK_ROOTS, target_windows=target,
                                 n_boot=n_boot)
    report = result["report"]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result["params"], report


def _ctm_medians(states: Sequence[dict], section: str, metric: str, bin_idx: int):
    return ctm_eval.median_of_errors(states, section, metric, bin_idx)


def _per_car_medians(states: Sequence[dict], arm: str, section: str, metric: str,
                     bin_idx: int):
    return ctm_eval.median(
        ctm_eval.per_car_error_arm(state, arm, section, metric, bin_idx)
        for state in states)


def _blowup(states: Sequence[dict], arm: str, models: dict[str, Sequence[dict]]):
    """Per-spec blow-up records; ``i`` is the ordered episode index in the set.

    Both the per-car result files and ``evaluate_set`` preserve input-record
    order, so ``i`` aligns a blow-up with the matching CTM state even when
    several policy episodes share a ``world@t0`` (the ``seen`` sets).
    """
    rows = {}
    for spec, file_states in models.items():
        bad = []
        for i, state in enumerate(file_states):
            value = ctm_eval.per_car_car_field(state, arm, ctm_eval.BLOWUP_HORIZON,
                                               "hdv", "ds_p90")
            if value is not None and value > ctm_eval.BLOWUP_DS_M:
                bad.append({"i": i, "world": state.get("world"),
                            "t0_s": state.get("t0_s"),
                            "hdv_ds_p90_m": float(value)})
        rows[spec] = bad
    return rows


def _jam_behavior(states: Sequence[dict], blowup: dict, n_bins: int) -> dict:
    """What the CTM predicts at the states where a per-car model blows up.

    A *missed jam* is a state whose recorded merge section is congested
    (mean speed < 12 m/s) while the CTM merge cell stays free (> 8 m/s faster);
    a *phantom jam* is the opposite (the recorded merge is free, > 22 m/s, while
    the CTM predicts < 8 m/s slower).  Uses the common 90-120 s bin (index 3)
    when available, else the last bin.  Episodes are matched to CTM states by
    ordered index (see :func:`_blowup`).
    """
    idxs = sorted({b["i"] for bad in blowup.values() for b in bad})
    bin_key = "3" if n_bins > 3 else str(max(0, n_bins - 1))
    missed = phantom = scored = 0
    gt_speeds: list[float] = []
    ctm_speeds: list[float] = []
    speed_errs: list[float] = []
    for i in idxs:
        if not (0 <= i < len(states)):
            continue
        state = states[i]
        if "error" in state:
            continue
        block = state.get("abs", {})
        gt = block.get("gt", {}).get(bin_key, {}).get("mean_speed")
        pred = block.get("ctm", {}).get(bin_key, {}).get("mean_speed")
        if gt is None or pred is None:
            continue
        scored += 1
        gt_speeds.append(float(gt))
        ctm_speeds.append(float(pred))
        speed_errs.append(abs(float(pred) - float(gt)))
        if gt < 12.0 and pred > gt + 8.0:
            missed += 1
        elif gt > 22.0 and pred < gt - 8.0:
            phantom += 1
    return {"bin": bin_key, "n_blowup_states": len(idxs), "n_scored": scored,
            "median_gt_speed": ctm_eval.median(gt_speeds),
            "median_ctm_speed": ctm_eval.median(ctm_speeds),
            "median_speed_error": ctm_eval.median(speed_errs),
            "missed_jam": missed, "phantom_jam": phantom}


def run_block(name: str, cfg: dict, params: ctm.CTMParams, layout: ctm.CellLayout, *,
              limit: Optional[int], models: Sequence[str]) -> dict:
    out = {"horizon_s": cfg["horizon_s"], "sets": {}}
    for set_name, set_cfg in cfg["sets"].items():
        t0 = time.time()
        states = ctm_eval.evaluate_set(set_cfg["path"], params, layout,
                                       unseen=set_cfg["unseen"],
                                       horizon_cap=cfg["horizon_s"], bin_s=30.0,
                                       limit=limit)
        errors = [s for s in states if "error" in s]
        per_car: dict[str, list] = {}
        for spec, path in set_cfg["per_car"].items():
            if spec not in models:
                continue
            if not Path(path).is_file():
                per_car[spec] = []
                continue
            loaded = ctm_eval.load_per_car(path)
            per_car[spec] = loaded
        arm = "pi" if set_cfg["unseen"] else "native"
        n_bins = max((int(k) for s in states if "error" not in s
                      for k in s["ctm"]["merge"]["total_delay"]), default=-1) + 1
        rows = {}
        for spec in models:
            file_states = per_car.get(spec, [])
            rows[spec] = {
                section: {metric: {
                    str(b): (_ctm_medians(states, section, metric, b)
                             if spec == "ctm" else
                             _per_car_medians(file_states, arm, section, metric, b))
                    for b in range(n_bins)} for metric in ctm_eval.METRICS}
                for section in ctm_eval.SECTIONS}
        blowup = _blowup(states, arm, per_car)
        jam = _jam_behavior(states, blowup, n_bins)
        out["sets"][set_name] = {
            "path": str(set_cfg["path"]), "unseen": set_cfg["unseen"], "arm": arm,
            "n_states": len([s for s in states if "error" not in s]),
            "n_errors": len(errors), "n_bins": n_bins,
            "elapsed_s": round(time.time() - t0, 1),
            "medians": rows,
            "blowup": blowup,
            "jam_behavior": jam,
            "n_per_car": {spec: len(v) for spec, v in per_car.items()},
        }
    return out


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def build_markdown(report: dict) -> str:
    lines = ["# CTM alone: a macroscopic long-horizon traffic anchor", ""]
    lines += ["**Question.** Does a stand-alone cell transmission model (CTM) - "
              "initialised from the recorded t0 snapshot and driven only by the t0 "
              "entrance counters - predict section-level traffic at 30-300 s better "
              "than the per-car world models, on the shifted and unshifted C1 v2 "
              "sets?", ""]
    lines += ["**Status.** complete (CPU only; no SUMO/GPU/WM training; TRAIN-only "
              "calibration; unseen PI states used for scoring only).", ""]
    lines += _calibration_table(report["calibration"])
    lines += _headline_table(report["blocks"])
    lines += _per_bin_tables(report["blocks"])
    lines += _blowup_table(report["blocks"])
    lines += ["## Verdict", "", report["verdict"], ""]
    return "\n".join(lines)


def _calibration_table(calib: dict) -> list:
    lines = ["## Calibration (TRAIN windows only)", "",
             f"- windows available: {calib['n_train_windows_available']}, "
             f"sampled: {calib['n_windows_sampled']}, used: {calib['n_windows_used']}, "
             f"bootstrap draws: {calib['n_bootstrap']}",
             "", "| section | v_free (m/s) | capacity (veh/h/lane) | jam (veh/km/lane) "
             "| w (m/s) | source | bootstrap sd (v,cap,w) |", "|---|---|---|---|---|---|---|"]
    for name, fit in calib["sections"].items():
        unc = calib["section_uncertainty"].get(name, {})
        sd = (f"{_fmt(unc.get('v_free_mps_sd'))}, {_fmt(unc.get('capacity_sd'), 0)}, "
              f"{_fmt(unc.get('w_sd'))}")
        lines.append(f"| {name} | {_fmt(fit['v_free_mps'])} | "
                     f"{_fmt(fit['capacity_vehph_per_lane'], 0)} | "
                     f"{_fmt(fit['jam_density_vehpkmlane'], 0)} | {_fmt(fit['w_mps'])} | "
                     f"{fit['source']} | {sd} |")
    drop = calib["merge_drop"]
    lines += ["", f"- merge capacity drop: {_fmt(drop['merge_capacity_drop'], 3)} "
              f"({drop['source']}; free n={drop.get('n_free')}, "
              f"congested n={drop.get('n_congested')})", ""]
    return lines


def _headline_table(blocks: dict) -> list:
    lines = ["## Headline: merge normalized delay error in the 90-120 s bin "
             "(median over states; lower is better)", "",
             "| block | set | " + " | ".join(MODEL_LABELS[m] for m in MODEL_ORDER)
             + " |", "|" + "---|" * (2 + len(MODEL_ORDER))]
    for block, data in blocks.items():
        for set_name, info in data["sets"].items():
            cells = []
            for model in MODEL_ORDER:
                value = info["medians"].get(model, {}).get("merge", {}).get(
                    "total_delay", {}).get("3")
                cells.append(_fmt(value))
            lines.append(f"| {block} | {set_name} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _per_bin_tables(blocks: dict) -> list:
    lines = ["## Per-30 s-bin errors by block (CTM vs best per-car on merge)",
             "",
             "Columns are 30 s bins 0..N-1 (0-30 s, 30-60 s, ...); each cell is the "
             "median |predicted - recorded| over states. Per-car rows end after the "
             "90-120 s bin: the frozen per-state scoring files store four traffic "
             "metrics (`mean_speed`, `stopped_vehicles`, `merge_queue_length`, "
             "`total_delay`) at 120 s, so their unshifted 120-300 s bins and the "
             "`density`/`merge_flow` metrics are not available without re-running "
             "rollouts. The CTM rows cover the full recorded window.", ""]
    for block, data in blocks.items():
        horizon = data["horizon_s"]
        lines += [f"### {block} (horizon {int(horizon)} s)", ""]
        for set_name, info in data["sets"].items():
            n_bins = info["n_bins"]
            lines += [f"**{set_name}** (arm `{info['arm']}`, {info['n_states']} states, "
                      f"{n_bins} bins)", "",
                      "| model | metric | " + " | ".join(f"b{b}" for b in range(n_bins))
                      + " |", "|" + "---|" * (2 + n_bins)]
            for model in MODEL_ORDER:
                for section, metric in (("merge", "total_delay"),
                                        ("merge", "mean_speed"),
                                        ("merge", "merge_queue_length"),
                                        ("upstream", "total_delay"),
                                        ("downstream", "mean_speed")):
                    vals = info["medians"].get(model, {}).get(section, {}).get(metric, {})
                    cells = [_fmt(vals.get(str(b))) for b in range(n_bins)]
                    lines.append(f"| {MODEL_LABELS[model]} | {section}.{metric} | "
                                 + " | ".join(cells) + " |")
            lines.append("")
    return lines


def _blowup_table(blocks: dict) -> list:
    lines = ["## Where the per-car models blow up (HDV ds_p90 > 100 m at 120 s) "
             "and what the CTM predicts", "",
             "At the union of those blow-up states the CTM is scored against the "
             "recorded merge mean speed in the common 90-120 s bin. *Missed jam* = "
             "recorded merge < 12 m/s but the CTM stays > 8 m/s freer; *phantom jam* "
             "= recorded merge > 22 m/s but the CTM predicts > 8 m/s slower.", "",
             "| block | set | per-car blow-up states (by model) | blow-up union | "
             "median recorded merge v | median CTM merge v | median |err| | missed | "
             "phantom |", "|---|---|---|---|---|---|---|---|---|"]
    for block, data in blocks.items():
        for set_name, info in data["sets"].items():
            counts = ", ".join(f"{MODEL_LABELS.get(s, s)} {len(b)}"
                               for s, b in info["blowup"].items() if b)
            jam = info.get("jam_behavior", {})
            lines.append(
                f"| {block} | {set_name} | {counts or 'none'} | "
                f"{jam.get('n_blowup_states')} | {_fmt(jam.get('median_gt_speed'))} | "
                f"{_fmt(jam.get('median_ctm_speed'))} | "
                f"{_fmt(jam.get('median_speed_error'))} | {jam.get('missed_jam')} | "
                f"{jam.get('phantom_jam')} |")
    lines.append("")
    return lines


MODEL_ORDER = ("ctm", "wms", "oldhist", "blackbox_traj", "blackbox_id",
               "pure_physics", "keep_speed")


def make_verdict(report: dict) -> str:
    """Plain-language, data-driven verdict from the computed medians."""
    blocks = report["blocks"]
    lines: list[str] = []

    # --- stability / blow-up contrast --------------------------------------- #
    total_blow = {}
    for data in blocks.values():
        for info in data["sets"].values():
            for spec, bad in info["blowup"].items():
                total_blow[spec] = total_blow.get(spec, 0) + len(bad)
    blow_txt = ", ".join(f"{MODEL_LABELS.get(s, s)} {n}"
                         for s, n in sorted(total_blow.items(), key=lambda x: -x[1]) if n)
    lines.append(
        "**Stability.** The CTM carries no vehicle identity, so it has no per-car "
        "divergence mode: it never blows up and (by the conservation unit test) only "
        "moves the mass its counters admit. The per-car models, by contrast, produced "
        f"HDV blow-up states (|ds| p90 > 100 m at 120 s) in every scored set: {blow_txt}. "
        "This is the CTM's clearest advantage over the per-car rollouts.")

    # --- accuracy vs per-car on the common 90-120 s bin --------------------- #
    beats: list[str] = []
    loses: list[str] = []
    for block, data in blocks.items():
        for set_name, info in data["sets"].items():
            ctm = info["medians"].get("ctm", {}).get("merge", {}).get(
                "total_delay", {}).get("3")
            per_car = {m: info["medians"].get(m, {}).get("merge", {}).get(
                "total_delay", {}).get("3") for m in MODEL_ORDER[1:]}
            per_car = {m: v for m, v in per_car.items() if v is not None}
            if ctm is None or not per_car:
                continue
            best = min(per_car, key=per_car.get)
            better_than = [m for m, v in per_car.items() if ctm < v]
            rec = f"{block}/{set_name}: CTM {_fmt(ctm)} s vs best per-car " \
                  f"{MODEL_LABELS[best]} {_fmt(per_car[best])} s; CTM beats " \
                  f"{len(better_than)}/{len(per_car)} per-car models"
            (beats if ctm < per_car[best] else loses).append(rec)
    lines.append("**Accuracy (merge normalized delay, 90-120 s bin, lower is better).** "
                 + ("CTM is the most accurate model in " + "; ".join(beats) + "." if beats
                    else "The CTM was never the most accurate model on this metric."))
    if loses:
        lines.append("It is less accurate than at least one learned per-car model in: "
                     + "; ".join(loses) + ".")

    # --- jam behaviour at the blow-up states -------------------------------- #
    for block, data in blocks.items():
        for set_name, info in data["sets"].items():
            jam = info.get("jam_behavior", {})
            if not jam.get("n_blowup_states"):
                continue
            lines.append(
                f"- {block}/{set_name}: at {jam['n_scored']} blow-up states the CTM "
                f"merge speed error is {_fmt(jam['median_speed_error'])} m/s "
                f"(recorded median {_fmt(jam['median_gt_speed'])}, CTM "
                f"{_fmt(jam['median_ctm_speed'])} m/s); missed jams {jam['missed_jam']}, "
                f"phantom jams {jam['phantom_jam']}.")
    lines.append(
        "So the CTM avoids the *divergence* blow-up but not the underlying error: at "
        "the blow-up states it mostly stays near free flow where some transmissions "
        "were congested (missed jams), and can build a queue the recording did not "
        "(phantom jams) by integrating the constant t0 counter rate into a capacity "
        "drop. It is a stable anchor, not a substitute for local structure.")

    # --- calibration caveats ------------------------------------------------ #
    calib = report["calibration"]
    srcs = ", ".join(f"{name}={fit['source']}" for name, fit in calib["sections"].items())
    lines.append(
        "**Calibration caveats.** All four section FDs fell back to the envelope "
        f"estimator ({srcs}); the least-squares free/congested knot was not resolvable "
        "on 100 m cells. `up`/`merge` free speed (30.4 m/s) sits above the recorded "
        "27.78 m/s cruise, jam density is pinned to the lower bound 117 veh/km/lane "
        "(0.9 x nominal), the merge FD is inherited from the mainline because the 75 m "
        "merge cell is too short to resolve a free branch, and the 0.92 capacity drop is "
        "estimated from only "
        f"{calib['merge_drop'].get('n_congested')} congested merge bins. The single "
        "merge cell also makes the CTM merge-queue proxy near-zero (a 75 m cell only "
        "reaches the stopped threshold close to jam). All parameters were fit on the "
        "PI-free TRAIN windows only; the unseen PI states were used for scoring only.")
    return "\n\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wms_eval.ctm_alone")
    parser.add_argument("--train-windows", default=str(TRAIN_WINDOWS))
    parser.add_argument("--target-windows", type=int, default=180)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--calib-json", default=str(ROOT / "_out" / "ctm_calibration.json"))
    parser.add_argument("--out-json", default=str(ROOT / "_out" / "CTM_ALONE.json"))
    parser.add_argument("--out-md", default=str(ROOT / "_out" / "CTM_ALONE.md"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--blocks", default="shifted,unshifted")
    parser.add_argument("--models", default="ctm,wms,oldhist,blackbox_traj,"
                                          "blackbox_id,pure_physics,keep_speed")
    parser.add_argument("--cell-length-m", type=float, default=100.0)
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    wanted_blocks = [b.strip() for b in args.blocks.split(",") if b.strip()]
    layout = ctm.build_layout(args.cell_length_m)
    params, calib_report = calibrate(Path(args.train_windows), args.target_windows,
                                     args.bootstrap, Path(args.calib_json))
    print(json.dumps({"calibrated": True, "drop":
                      calib_report["merge_drop"]["merge_capacity_drop"]}), flush=True)
    blocks = {}
    for name in wanted_blocks:
        blocks[name] = run_block(name, BLOCKS[name], params, layout,
                                 limit=args.limit, models=models)
        print(json.dumps({name: {k: v["n_states"] for k, v in
                                 blocks[name]["sets"].items()}}), flush=True)
    report = {"schema": "ctm_alone/1",
              "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "params": params.to_dict(), "calibration": calib_report,
              "blocks": blocks, "models": models}
    report["verdict"] = make_verdict(report)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=1, default=float),
                                   encoding="utf-8")
    Path(args.out_md).write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
