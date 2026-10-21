"""Round-8 merge-core normaliser diagnostic (Z_TF / Z_ON) on fixed TRAIN windows.

Loads a checkpoint and a training preset, forces the SECUREGAP core (and loss version 2),
rolls ``--limit`` fixed recorded windows through the normal chunked unroll without any
optimizer step, and prints the positive mean per-car sequence-TF and on-policy NLL values
used to freeze ``z_tf`` / ``z_on``, plus the terminal/exposure diagnostics of the repaired
supported-prefix survival.

    python tools/merge_core_norm.py --checkpoint PATH --config configs/c4_wm_v10round2.toml \
        --windows PATH_TO_TRAIN_STATES.json --limit 8 --version 2

Restricted split paths (``sealed``, ``holdout``/``heldout``, ``cal_split``, ``chunk5``,
``j4b``, ``inc1``) and evaluation seeds ``964000-964199`` are refused.  No data are written.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wms.config.load import load_raw  # noqa: E402
from wms.core.constants import DEFAULT_LENGTH_M  # noqa: E402
from wms.eval.backend import default_tools  # noqa: E402
from wms.eval.normalize import iter_windows  # noqa: E402
from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles  # noqa: E402
from wms.model.checkpoint import load_checkpoint  # noqa: E402
from wms.train.losses.base import LossContext, UnrollTargets  # noqa: E402
from wms.train.losses.merge_core import (MergeCoreConfig, MergeCoreTerm,  # noqa: E402
                                         merge_core_metrics)

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
FORBIDDEN_SEEDS = range(964000, 964200)


def guard(path: str | Path) -> Path:
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return Path(path)


def _merge_core_config(raw: dict, args: argparse.Namespace) -> MergeCoreConfig:
    section = dict((raw.get("loss") or {}).get("merge_core") or {})
    allowed = {f.name for f in fields(MergeCoreConfig)}
    section = {k: v for k, v in section.items() if k in allowed}
    section["version"] = int(args.version)
    section["tf_weight"] = float(args.tf_weight if args.tf_weight is not None
                                 else section.get("tf_weight", 0.0))
    section["weight"] = float(args.weight if args.weight is not None
                              else section.get("weight", 0.0))
    section["n_quad"] = int(args.n_quad or section.get("n_quad", 8))
    section["metrics"] = True
    if section["tf_weight"] > 0.0:
        try:
            import wms.train.losses.merge_core_seqtf  # noqa: F401
        except ImportError:
            print("[merge_core_norm] merge_core_seqtf is absent; setting tf_weight=0",
                  file=sys.stderr)
            section["tf_weight"] = 0.0
    return MergeCoreConfig(**section)


def _window_objects(record: dict, n_steps: int, tools, device: torch.device):
    t0_s, ids, kin, is_cav_np = t0_vehicles(record)
    teacher, present, _junction = recorded_teacher(
        record, t0_s, ids, kin, n_steps, "native", tools)
    if teacher is None or present is None:
        return None
    teacher = teacher.to(device)
    present = present.to(device).bool()
    b, t1, k, _ = teacher.shape
    states = torch.zeros(b, t1, k, 7, dtype=teacher.dtype, device=device)
    states[..., 0] = teacher[..., 0]
    lane = teacher[..., 1].long().clamp(0, 2)
    states[..., 1:4] = torch.nn.functional.one_hot(lane, 3).to(states.dtype)
    states[..., 4] = teacher[..., 2]
    states[..., 5] = teacher[..., 3]
    states[..., 6] = float(DEFAULT_LENGTH_M)
    is_cav = torch.as_tensor(is_cav_np, dtype=torch.bool, device=device).unsqueeze(0)
    is_cav_seq = is_cav.unsqueeze(1).expand(b, t1, k)
    initial = {
        "veh_kin": states[:, 0], "s_m": states[:, 0, :, 0], "v_mps": states[:, 0, :, 4],
        "a_mps2": states[:, 0, :, 5], "lane": states[:, 0, :, 1:4].argmax(-1),
        "length_m": states[:, 0, :, 6], "present": present[:, 0], "is_cav": is_cav,
        "lane_age": torch.zeros(b, k, dtype=torch.long, device=device),
    }
    weights = torch.ones(b, n_steps, 1, dtype=states.dtype, device=device) / max(n_steps, 1)
    hdv_mask = present[:, 1:n_steps + 1] & ~is_cav_seq[:, 1:n_steps + 1]
    targets = UnrollTargets(
        states=states, n_steps=n_steps, weights=weights, hdv_mask=hdv_mask,
        aux_mask=hdv_mask & present[:, :n_steps], aux_count=int(hdv_mask.sum()),
        denom=(weights * hdv_mask.to(weights.dtype)).sum().clamp_min(1.0),
        lane_target=states[:, 1:n_steps + 1, :, 1:4].argmax(-1),
        loss_valid=present, is_cav=is_cav_seq, dt_s=0.5,
        position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False)
    return initial, targets, teacher, present, is_cav_seq


def _run_window(model, tools, record, cfg: MergeCoreConfig, n_steps: int, chunk: int):
    built = _window_objects(record, n_steps, tools, torch.device("cpu"))
    if built is None:
        return None
    initial, targets, teacher, present, is_cav_seq = built
    term = MergeCoreTerm(cfg)
    term.prepare(targets)
    current = initial
    hist = hist_mask = None
    values = []
    with torch.no_grad():
        for c0 in range(0, n_steps, chunk):
            c1 = min(n_steps, c0 + chunk)
            roll = model.rollout(
                current, n_steps=c1 - c0, k0=c0,
                cav_teacher=teacher[:, c0:c1 + 1],
                present_seq=present[:, c0:c1 + 1],
                is_cav_seq=is_cav_seq[:, c0:c1 + 1],
                history=hist, history_mask=hist_mask)
            value = term(LossContext(pred=roll, current=current, targets=targets,
                                     c0=c0, c1=c1))
            if value is None:
                return None
            values.append(float(value.value.detach()))
            current = {key: (v.detach() if torch.is_tensor(v) else v)
                       for key, v in roll["state"].items()}
            current["is_cav"] = is_cav_seq[:, c1]
            current["length_m"] = targets.states[:, c1, :, 6]
            hist, hist_mask = roll.get("history"), roll.get("history_mask")
    return merge_core_metrics(targets.prepared["merge_core"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", required=True, help="TRAIN state-window JSON (states/windows)")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--unroll-s", type=float, default=None)
    ap.add_argument("--truncation-steps", type=int, default=None)
    ap.add_argument("--version", type=int, default=2, choices=(1, 2))
    ap.add_argument("--tf-weight", type=float, default=None)
    ap.add_argument("--weight", type=float, default=None)
    ap.add_argument("--n-quad", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if int(args.seed) in FORBIDDEN_SEEDS:
        raise SystemExit(f"refusing evaluation seed {args.seed}")
    checkpoint = guard(args.checkpoint)
    config_path = guard(args.config)
    windows_path = guard(args.windows)

    raw = load_raw(config_path)
    cfg = _merge_core_config(raw, args)
    curriculum = raw.get("curriculum") or {}
    unroll_s = float(args.unroll_s if args.unroll_s is not None
                     else (curriculum.get("unroll_s") or [120.0])[0])
    chunk = int(args.truncation_steps if args.truncation_steps is not None
                else (curriculum.get("truncation_steps_per_stage") or [120])[0])
    n_steps = max(1, int(round(unroll_s / 0.5)))

    loaded = load_checkpoint(checkpoint, strict=False,
                             model_config={"merge_decision": "secure_gap"})
    model = loaded.model.eval()
    tools = default_tools()

    rows = []
    for index, record in enumerate(iter_windows(windows_path)):
        if index >= int(args.limit):
            break
        metrics = _run_window(model, tools, record, cfg, n_steps, chunk)
        if metrics is None:
            print(f"[merge_core_norm] window {index}: no recorded native arm; skipped",
                  file=sys.stderr)
            continue
        row = {
            "window": index,
            "z_on": metrics.get("merge_core_on_nll", float("nan")),
            "z_tf": metrics.get("merge_core_tf_nll_per_car", float("nan")),
            "n_cars": metrics.get("merge_core_n_cars"),
            "events": metrics.get("merge_core_terminal_event"),
            "censored_admin": metrics.get("merge_core_terminal_censored_admin"),
            "support_failure": sum(metrics.get(f"merge_core_terminal_support_failure_{r}", 0.0)
                                   for r in ("pos", "speed", "gap", "topology", "identity")),
            "early_nonevent": metrics.get("merge_core_terminal_early_nonevent"),
            "zero_exposure": metrics.get("merge_core_terminal_zero_exposure"),
            "unsupported_events": metrics.get("merge_core_unsupported_events"),
            "scored_seconds": metrics.get("merge_core_scored_seconds"),
            "observed_seconds": metrics.get("merge_core_observed_seconds"),
            "opportunity_intervals": metrics.get("merge_core_opportunity_intervals"),
        }
        rows.append(row)
        print("[merge_core_norm] " + " ".join(
            f"{key}={row[key]:.4f}" if isinstance(row[key], float) else f"{key}={row[key]}"
            for key in row), flush=True)

    def mean_positive(key: str) -> float | None:
        values = [row[key] for row in rows
                  if isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                  and row[key] > 0.0]
        return float(sum(values) / len(values)) if values else None

    z_on = mean_positive("z_on")
    z_tf = mean_positive("z_tf") if cfg.tf_weight > 0.0 else None
    exposure = {
        "windows": len(rows),
        "events": sum(row["events"] or 0.0 for row in rows),
        "censored_admin": sum(row["censored_admin"] or 0.0 for row in rows),
        "support_failure": sum(row["support_failure"] or 0.0 for row in rows),
        "early_nonevent": sum(row["early_nonevent"] or 0.0 for row in rows),
        "zero_exposure": sum(row["zero_exposure"] or 0.0 for row in rows),
        "unsupported_events": sum(row["unsupported_events"] or 0.0 for row in rows),
        "scored_seconds": sum(row["scored_seconds"] or 0.0 for row in rows),
        "observed_seconds": sum(row["observed_seconds"] or 0.0 for row in rows),
    }
    print(f"[merge_core_norm] Z_ON={z_on} Z_TF={z_tf}", flush=True)
    print(f"[merge_core_norm] exposure={json.dumps(exposure, sort_keys=True)}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"z_on": z_on, "z_tf": z_tf, "windows": rows, "exposure": exposure}, indent=2),
            encoding="utf-8")
    if z_on is None or not math.isfinite(z_on) or z_on <= 0.0:
        print("[merge_core_norm] no positive on-policy exposure; refusing to freeze Z_ON",
              file=sys.stderr)
        return 2
    if cfg.tf_weight > 0.0 and (z_tf is None or not math.isfinite(z_tf) or z_tf <= 0.0):
        print("[merge_core_norm] no positive sequence-TF exposure; refusing to freeze Z_TF",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
