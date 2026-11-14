"""Planner-readiness ranking test: can the world model rank native vs PI?

The C1 v2 unseen sets carry *paired* arms: the same world and snapshot recorded
once under ``native`` and once under ``pi``.  A planner does not need absolute
accuracy; it needs the model to predict the **difference** between the two
candidate actions so it can pick the better one.  This module scores that:

* **per-state values** -- for every arm it keeps the PREDICTED and RECORDED
  per-30 s-bin traffic metrics (mean speed, stopped vehicles, merge queue,
  normalized total delay) per section, plus two horizon aggregates: the
  network-wide normalized total delay and the throughput (vehicles crossing
  the downstream merge boundary);
* **paired deltas** -- predicted ``pi - native`` versus recorded ``pi - native``
  at 60/120/300 s, with sign agreement, Spearman rank correlation and MAE, on
  the subset of states whose recorded delta exceeds the recorded-noise floor;
* **decision regret** -- choose the arm with the lower predicted network delay,
  then charge the recorded regret of that choice against an oracle, with
  ``always native`` and ``always pi`` references and the correct-choice share;
* **absolute accuracy** -- relative errors (%) per section and per bin so the
  long-run traffic prediction quality is answerable in plain terms.

Nothing here trains or simulates: it is a CPU-only offline evaluation of an
already-trained ``wm_s/1`` checkpoint.  The noise floor is a fixed small
threshold on the recorded network-delay delta, motivated by REP-P, which proved
the campaign simulator is deterministic (within-pair replicate noise = 0) so the
only floor left is numerical/tie-breaking (see ``_out/RANKING.md``).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .c1v2 import iter_windows
from .data import DataError, json_default, recorded_for
from .metrics import SECTIONS, cumulative_delay, throughput, traffic_metrics, truncate_frames

HORIZONS_S = (60.0, 120.0, 300.0)
DEFAULT_BIN_S = 30.0
DEFAULT_NOISE_FLOOR = 1.0  # veh-s on the recorded network-delay delta
SECTION_METRICS = ("mean_speed", "stopped_vehicles", "merge_queue_length", "total_delay")
RECORDED_ARMS = ("native", "pi")


class RankingError(ValueError):
    """The ranking inputs or a per-state rollout payload are invalid."""


# --------------------------------------------------------------------------- #
# per-state rollout
# --------------------------------------------------------------------------- #
def traffic_bundle(frames: list[dict], *, t0_s: float, horizon_s: float,
                   free_speed_mps: float, bin_s: float = DEFAULT_BIN_S) -> dict:
    """Store PREDICTED/RECORDED per-bin metrics plus the horizon aggregates."""
    inside = truncate_frames(frames, t0_s=t0_s, horizon_s=horizon_s)
    aggregate = cumulative_delay(inside, t0_s=t0_s, horizon_s=horizon_s, bin_s=bin_s,
                                 free_speed_mps=free_speed_mps)
    aggregate["throughput"] = throughput(inside)
    return {
        "bins": traffic_metrics(inside, t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed_mps),
        "aggregate": aggregate,
    }


def evaluate_state(record: dict, predictor, horizon_s: float, bin_s: float,
                   cav_mode: str = "teacher") -> dict:
    """Roll out both arms of one paired state and keep the traffic values."""
    t0_s = float(record.get("t0_s", 0.0))
    seed = int(record.get("seed", 0))
    free_speed = float(record.get("v0_mps", 13.89))
    controllers = dict(record.get("cav_controllers", {}))
    controllers["cav_mode"] = cav_mode
    item: dict[str, Any] = {
        "world": record.get("world", "unknown"), "t0_s": t0_s, "seed": seed,
        "horizon_s": float(horizon_s), "cav_mode": cav_mode, "arms": {},
    }
    for arm in RECORDED_ARMS:
        try:
            predicted = predictor(dict(record), {**controllers, "arm": arm}, horizon_s, seed)
        except Exception as exc:  # noqa: BLE001 - one arm must not abort the state
            item["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            recorded = recorded_for(record, arm)
        except DataError as exc:
            item["arms"][arm] = {"error": str(exc)}
            continue
        item["arms"][arm] = {
            "error": None,
            "predicted": traffic_bundle(predicted, t0_s=t0_s, horizon_s=horizon_s,
                                        free_speed_mps=free_speed, bin_s=bin_s),
            "recorded": traffic_bundle(recorded, t0_s=t0_s, horizon_s=horizon_s,
                                       free_speed_mps=free_speed, bin_s=bin_s),
        }
    return item


def run_shard(spec: str, checkpoint: str | None, states_path: str | Path, horizon_s: float,
              bin_s: float, shard: int, nshards: int, physics: dict | None,
              cav_mode: str = "teacher", limit: int | None = None,
              label: str | None = None) -> dict:
    """Score one disjoint slice of a state set and return a JSON-safe payload."""
    from .models import load_model

    predictor, flags = load_model(spec, checkpoint=checkpoint, entries=False, physics=physics)
    states = []
    for index, record in enumerate(iter_windows(states_path, limit=limit)):
        if index % nshards != shard:
            continue
        item = evaluate_state(record, predictor, horizon_s, bin_s, cav_mode)
        item["index"] = index
        states.append(item)
    return {
        "schema": "wms_eval_ranking_states/1", "model": spec, "checkpoint": checkpoint,
        "set": label or Path(states_path).stem, "states_path": str(states_path),
        "horizon_s": float(horizon_s), "bin_s": float(bin_s), "shard": shard,
        "nshards": nshards, "cav_mode": cav_mode, "flags": getattr(flags, "__dict__", {}),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "states": states,
    }


# --------------------------------------------------------------------------- #
# metric extraction from stored bundles
# --------------------------------------------------------------------------- #
def _cumulative_network(bins: dict, t0_s: float, horizon_s: float) -> float | None:
    total = 0.0
    seen = False
    for key in sorted(bins, key=lambda k: int(k)):
        start = float(bins[key]["start_s"])
        if start - t0_s >= horizon_s - 1e-9:
            continue
        for name in SECTIONS:
            value = bins[key]["sections"][name].get("total_delay")
            if value is not None:
                total += float(value)
                seen = True
    return total if seen else None


def _cumulative_section(bins: dict, t0_s: float, horizon_s: float, section: str) -> float | None:
    total = 0.0
    seen = False
    for key in sorted(bins, key=lambda k: int(k)):
        start = float(bins[key]["start_s"])
        if start - t0_s >= horizon_s - 1e-9:
            continue
        value = bins[key]["sections"][section].get("total_delay")
        if value is not None:
            total += float(value)
            seen = True
    return total if seen else None


def _snapshot(bins: dict, t0_s: float, horizon_s: float, section: str,
              metric: str, bin_s: float) -> float | None:
    index = max(0, int(round(horizon_s / bin_s)) - 1)
    block = bins.get(str(index))
    if block is None:
        return None
    return block["sections"][section].get(metric)


def paired_metrics(item: dict, horizon_s: float, bin_s: float = DEFAULT_BIN_S) -> dict | None:
    """Predicted/recorded paired deltas (pi - native) for one state."""
    arms = item.get("arms", {})
    native, pi = arms.get("native"), arms.get("pi")
    if not native or not pi or native.get("error") or pi.get("error"):
        return None
    t0 = float(item["t0_s"])
    out: dict[str, Any] = {"world": item.get("world"), "t0_s": t0}
    metrics = {
        "network_total_delay": lambda b: _cumulative_network(b, t0, horizon_s),
        "merge_total_delay": lambda b: _cumulative_section(b, t0, horizon_s, "merge"),
        "upstream_total_delay": lambda b: _cumulative_section(b, t0, horizon_s, "upstream"),
        "downstream_total_delay": lambda b: _cumulative_section(b, t0, horizon_s, "downstream"),
        "merge_mean_speed": lambda b: _snapshot(b, t0, horizon_s, "merge", "mean_speed", bin_s),
        "merge_queue_length": lambda b: _snapshot(b, t0, horizon_s, "merge", "merge_queue_length", bin_s),
        "merge_stopped_vehicles": lambda b: _snapshot(b, t0, horizon_s, "merge", "stopped_vehicles", bin_s),
    }
    absolutes: dict[str, dict[str, dict[str, float | None]]] = {}
    for name, getter in metrics.items():
        pn, pp = getter(native["predicted"]["bins"]), getter(pi["predicted"]["bins"])
        rn, rp = getter(native["recorded"]["bins"]), getter(pi["recorded"]["bins"])
        predicted = None if pn is None or pp is None else float(pp) - float(pn)
        recorded = None if rn is None or rp is None else float(rp) - float(rn)
        out[name] = {"predicted": predicted, "recorded": recorded}
        absolutes[name] = {
            "predicted": {"native": pn, "pi": pp},
            "recorded": {"native": rn, "pi": rp},
        }
    out["absolutes"] = absolutes
    # throughput over the horizon (crossing counts, from the aggregate)
    pt = (pi["predicted"]["aggregate"]["throughput"]
          - native["predicted"]["aggregate"]["throughput"])
    rt = (pi["recorded"]["aggregate"]["throughput"]
          - native["recorded"]["aggregate"]["throughput"])
    out["throughput"] = {"predicted": float(pt), "recorded": float(rt)}
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _sign(value: float) -> int:
    if value > 1e-12:
        return 1
    if value < -1e-12:
        return -1
    return 0


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))


def paired_stats(rows: Sequence[dict], metric: str, *, noise_floor: float) -> dict:
    """Sign agreement / Spearman / MAE of predicted vs recorded deltas."""
    pairs = [(row[metric]["predicted"], row[metric]["recorded"]) for row in rows
             if row.get(metric) and row[metric]["predicted"] is not None
             and row[metric]["recorded"] is not None]
    decisive = [(p, r) for p, r in pairs if abs(r) > noise_floor]
    dec_pred = [p for p, _r in decisive]
    dec_rec = [r for _p, r in decisive]
    agree = sum(1 for p, r in decisive if _sign(p) == _sign(r))
    return {
        "n": len(pairs), "n_decisive": len(decisive),
        "sign_agreement": (agree / len(decisive)) if decisive else None,
        "spearman": spearman(dec_pred, dec_rec) if len(decisive) >= 2 else None,
        "mae": (sum(abs(p - r) for p, r in pairs) / len(pairs)) if pairs else None,
        "bias": (sum(p - r for p, r in pairs) / len(pairs)) if pairs else None,
        "mae_decisive": (sum(abs(p - r) for p, r in decisive) / len(decisive)) if decisive else None,
    }


def _decision_rows(rows: Sequence[dict]) -> list[dict]:
    """Absolute predicted/recorded network delay for both arms, per state."""
    out = []
    for row in rows:
        abs_ = row.get("absolutes", {}).get("network_total_delay")
        if not abs_:
            continue
        pn, pp = abs_["predicted"]["native"], abs_["predicted"]["pi"]
        rn, rp = abs_["recorded"]["native"], abs_["recorded"]["pi"]
        if None in (pn, pp, rn, rp):
            continue
        out.append({"pn": float(pn), "pp": float(pp), "rn": float(rn),
                    "rp": float(rp), "recorded_delta": float(rp) - float(rn)})
    return out


def decision_regret(rows: Sequence[dict], *, noise_floor: float) -> dict:
    """Pick the arm with lower predicted network delay; charge recorded regret.

    Regret = recorded delay of the chosen arm - recorded delay of the better
    (oracle) arm.  Baseline references always pick native / always pick pi; the
    oracle scores 0 by construction.
    """
    scored = _decision_rows(rows)
    if not scored:
        return {"n": 0, "model_mean_regret": None, "model_correct_share": None,
                "always_native_mean_regret": None, "always_pi_mean_regret": None,
                "oracle_mean_regret": 0.0, "model_saving_vs_native": None,
                "decisive": {"n": 0, "model_mean_regret": None,
                             "model_correct_share": None}}

    def evaluate(select):
        regrets, correct = [], 0
        for row in scored:
            rn, rp = row["rn"], row["rp"]
            chosen = select(row)
            chosen_rec = rp if chosen == "pi" else rn
            best = min(rn, rp)
            regrets.append(chosen_rec - best)
            if chosen_rec <= best + 1e-9:
                correct += 1
        return sum(regrets) / len(regrets), correct / len(regrets)

    model_regret, model_correct = evaluate(lambda row: "pi" if row["pp"] < row["pn"] else "native")
    native_regret, _ = evaluate(lambda _row: "native")
    pi_regret, _ = evaluate(lambda _row: "pi")
    saving = sum(row["rn"] - (row["rp"] if row["pp"] < row["pn"] else row["rn"])
                 for row in scored) / len(scored)

    decisive = [row for row in scored if abs(row["recorded_delta"]) > noise_floor]
    if decisive:
        d_regret, d_correct = evaluate_rows(decisive, lambda row: "pi" if row["pp"] < row["pn"] else "native")
    else:
        d_regret, d_correct = None, None
    return {
        "n": len(scored),
        "model_mean_regret": model_regret,
        "model_correct_share": model_correct,
        "always_native_mean_regret": native_regret,
        "always_pi_mean_regret": pi_regret,
        "oracle_mean_regret": 0.0,
        "model_saving_vs_native": saving,
        "decisive": {"n": len(decisive), "model_mean_regret": d_regret,
                     "model_correct_share": d_correct},
    }


def evaluate_rows(rows: Sequence[dict], select) -> tuple[float, float]:
    regrets, correct = [], 0
    for row in rows:
        rn, rp = row["rn"], row["rp"]
        chosen_rec = rp if select(row) == "pi" else rn
        best = min(rn, rp)
        regrets.append(chosen_rec - best)
        if chosen_rec <= best + 1e-9:
            correct += 1
    return sum(regrets) / len(regrets), correct / len(regrets)


def absolute_rel_error(rows: Sequence[dict], section_metric: str, *, section: str,
                       bin_idx: int | None = None) -> dict:
    """Median relative error (%) of predicted vs recorded traffic magnitudes."""
    errors = []
    for item in rows:
        blocks = item.get("arms", {})
        for arm in RECORDED_ARMS:
            block = blocks.get(arm)
            if not block or block.get("error"):
                continue
            pred_bins = block["predicted"]["bins"]
            rec_bins = block["recorded"]["bins"]
            keys = [str(bin_idx)] if bin_idx is not None else sorted(pred_bins, key=lambda k: int(k))
            for key in keys:
                pblock = pred_bins.get(key)
                rblock = rec_bins.get(key)
                if not pblock or not rblock:
                    continue
                p = pblock["sections"][section].get(section_metric)
                r = rblock["sections"][section].get(section_metric)
                if p is None or r is None:
                    continue
                threshold = 0.5 if section_metric in ("total_delay", "stopped_vehicles",
                                                      "merge_queue_length") else 1.0
                if abs(float(r)) <= threshold:
                    continue
                errors.append(abs(float(p) - float(r)) / abs(float(r)))
    if not errors:
        return {"n": 0, "median_pct": None, "mean_pct": None}
    return {"n": len(errors), "median_pct": 100.0 * median(errors),
            "mean_pct": 100.0 * sum(errors) / len(errors)}


# --------------------------------------------------------------------------- #
# analysis + report
# --------------------------------------------------------------------------- #
def load_run(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("states"), list):
        raise RankingError(f"{path} is not a ranking states payload")
    return payload


def merge_shards(paths: Sequence[str | Path]) -> dict:
    """Merge shard payloads of one (model, set, horizon) into one payload."""
    parts = [load_run(p) for p in paths]
    first = parts[0]
    states = sorted((s for part in parts for s in part["states"]),
                    key=lambda s: int(s.get("index", 0)))
    return {**first, "shard": None, "nshards": None, "states": states}


def analyze(runs: Sequence[dict], *, noise_floor: float = DEFAULT_NOISE_FLOOR,
            bin_s: float = DEFAULT_BIN_S) -> dict:
    """Build the full ranking analysis from per-(model,set) payloads."""
    report: dict[str, Any] = {
        "schema": "wms_eval_ranking_result/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "noise_floor_veh_s": noise_floor,
        "horizons_s": list(HORIZONS_S),
        "models": {},
    }
    for run in runs:
        model = str(run.get("model"))
        set_name = str(run.get("set") or Path(str(run.get("states_path", "set"))).stem)
        entry = report["models"].setdefault(model, {"checkpoint": run.get("checkpoint"),
                                                    "sets": {}})
        states = run["states"]
        horizon_run = float(run.get("horizon_s", 300.0))
        set_block: dict[str, Any] = {"n_states": len(states),
                                     "horizon_run_s": horizon_run,
                                     "cav_mode": run.get("cav_mode", "teacher"),
                                     "errors": sum(1 for s in states
                                                   if any((s["arms"].get(a) or {}).get("error")
                                                          for a in RECORDED_ARMS)),
                                     "horizons": {}}
        for horizon in HORIZONS_S:
            if horizon > horizon_run + 1e-9:
                continue
            rows = []
            for item in states:
                paired = paired_metrics(item, horizon, bin_s)
                if paired is not None:
                    rows.append(paired)
            if not rows:
                continue
            set_block["horizons"][f"{horizon:g}"] = {
                "n_paired_states": len(rows),
                "network_total_delay": paired_stats(rows, "network_total_delay",
                                                    noise_floor=noise_floor),
                "merge_total_delay": paired_stats(rows, "merge_total_delay",
                                                  noise_floor=noise_floor),
                "merge_mean_speed": paired_stats(rows, "merge_mean_speed",
                                                 noise_floor=noise_floor),
                "throughput": paired_stats(rows, "throughput", noise_floor=0.0),
                "decision_regret": decision_regret(rows, noise_floor=noise_floor),
            }
        set_block["absolute"] = {
            "by_section": {
                f"{section}.{metric}": absolute_rel_error(states, metric, section=section)
                for section in SECTIONS for metric in SECTION_METRICS
            },
            "network_delay_by_bin": {
                str(idx): absolute_rel_error(states, "total_delay", section="merge", bin_idx=idx)
                for idx in range(int(HORIZONS_S[1] // bin_s))
            },
        }
        entry["sets"][set_name] = set_block
    return report


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def _f(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _pct(value, digits: int = 0) -> str:
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def markdown(report: dict) -> str:
    lines = [
        "# RANKING - can the current world model rank controllers correctly?",
        "",
        "**Planner-readiness test (CPU only; no training, no SUMO).** The C1 v2 unseen "
        "sets contain paired arms: the same world/t0 recorded under `native` and under "
        "`pi`. The model is scored on whether it predicts the *difference* "
        "`pi - native`, not absolute accuracy, and whether that difference is good "
        "enough to choose the better action.",
        "",
        f"- Recorded-noise floor: a recorded network-delay delta is treated as "
        f"decisive only when `|delta| > {report['noise_floor_veh_s']:.1f}` veh-s. "
        "REP-P proved the campaign simulator is deterministic (within-pair replicate "
        "noise = 0), so this fixed numeric threshold only removes exact/near ties.",
        "- Metric: normalized total delay (veh-s) accumulated over the window; "
        "`network` sums upstream+merge+downstream. Throughput = distinct vehicles "
        "crossing the downstream merge boundary.",
        "- Decision rule: choose `pi` when predicted `pi - native` network delay < 0, "
        "else `native` (ties -> native).",
        "",
    ]

    # headline regret table
    lines += ["## Decision regret on network delay (lower is better; oracle = 0)", "",
              "| model | set | horizon | n | correct | model regret | always-native | always-pi |",
              "|---|---|---|---|---|---|---|---|"]
    for model, mentry in report["models"].items():
        for set_name, block in mentry["sets"].items():
            for horizon, hblock in sorted(block["horizons"].items(), key=lambda kv: float(kv[0])):
                reg = hblock["decision_regret"]
                lines.append(
                    f"| {model} | {set_name} | {horizon}s | {reg['n']} "
                    f"| {_pct(reg['model_correct_share'])} "
                    f"| {_f(reg['model_mean_regret'])} "
                    f"| {_f(reg['always_native_mean_regret'])} "
                    f"| {_f(reg['always_pi_mean_regret'])} |")
    lines.append("")

    # paired delta stats
    lines += ["## Paired delta quality (predicted vs recorded `pi - native`)", "",
              "Sign agreement/Spearman/MAE use only states with a decisive recorded "
              "delta; MAE(all) uses every paired state.", "",
              "| model | set | horizon | metric | n | decisive | sign agree | Spearman | MAE(dec) | MAE(all) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for model, mentry in report["models"].items():
        for set_name, block in mentry["sets"].items():
            for horizon, hblock in sorted(block["horizons"].items(), key=lambda kv: float(kv[0])):
                for metric in ("network_total_delay", "merge_total_delay", "merge_mean_speed"):
                    stat = hblock[metric]
                    lines.append(
                        f"| {model} | {set_name} | {horizon}s | {metric} | {stat['n']} "
                        f"| {stat['n_decisive']} | {_f(stat['sign_agreement'])} "
                        f"| {_f(stat['spearman'])} | {_f(stat['mae_decisive'])} "
                        f"| {_f(stat['mae'])} |")
    lines.append("")

    # absolute accuracy
    lines += ["## Absolute traffic accuracy (median relative error %, pooled bins/states)",
              "",
              "| model | set | section.metric | n | median % | mean % |",
              "|---|---|---|---|---|---|"]
    for model, mentry in report["models"].items():
        for set_name, block in mentry["sets"].items():
            for key, stat in block["absolute"]["by_section"].items():
                lines.append(f"| {model} | {set_name} | {key} | {stat['n']} "
                             f"| {_f(stat['median_pct'], 1)} | {_f(stat['mean_pct'], 1)} |")
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_physics(path: str | None) -> dict | None:
    if not path:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("physics"), dict):
        return raw["physics"]
    return raw if isinstance(raw, dict) else None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval ranking")
    sub = p.add_subparsers(dest="command", required=True)

    roll = sub.add_parser("roll", help="score one shard of paired states")
    roll.add_argument("--model", required=True)
    roll.add_argument("--ckpt")
    roll.add_argument("--states", required=True)
    roll.add_argument("--out", required=True)
    roll.add_argument("--horizon", type=float, default=120.0)
    roll.add_argument("--bin-s", type=float, default=DEFAULT_BIN_S)
    roll.add_argument("--physics")
    roll.add_argument("--shard", type=int, default=0)
    roll.add_argument("--nshards", type=int, default=1)
    roll.add_argument("--cav-mode", choices=("teacher", "closed_loop"), default="teacher")
    roll.add_argument("--limit", type=int, default=None)
    roll.add_argument("--label", help="override the set label stored in the payload")

    analyze_p = sub.add_parser("analyze", help="build RANKING.md/.json from merged runs")
    analyze_p.add_argument("--run", action="append", required=True,
                           help="PATH to a merged states payload (repeatable)")
    analyze_p.add_argument("--merge", action="append", default=[],
                           help="OUT=glob-free comma-separated shard paths to merge first")
    analyze_p.add_argument("--out", required=True, help="markdown output path")
    analyze_p.add_argument("--noise-floor", type=float, default=DEFAULT_NOISE_FLOOR)
    analyze_p.add_argument("--bin-s", type=float, default=DEFAULT_BIN_S)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "roll":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = run_shard(args.model, args.ckpt, args.states, args.horizon, args.bin_s,
                            args.shard, args.nshards, _load_physics(args.physics),
                            args.cav_mode, args.limit, args.label)
        out.write_text(json.dumps(payload, default=json_default), encoding="utf-8")
        print(json.dumps({"out": str(out), "model": args.model, "states": len(payload["states"])}))
        return 0
    # analyze
    runs = [load_run(p) for p in args.run]
    report = analyze(runs, noise_floor=args.noise_floor, bin_s=args.bin_s)
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown(report), encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=1, default=json_default), encoding="utf-8")
    print(json.dumps({"out_md": str(out_md), "out_json": str(out_json),
                      "models": sorted(report["models"])}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
