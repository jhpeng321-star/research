"""Record per-CAV executed teacher deviations and bounded residual reachability."""
from __future__ import annotations

import argparse
import json
import os

import torch

from .law import caprelative_available, representation
from .runtime import Feedback, check_seed, load_assets, load_training_starts, rollout, safe_path


def summarize(rows):
    result = {}
    for law in ("current", "cap_relative"):
        if law == "cap_relative" and not caprelative_available():
            result[law] = {"status": "pending_S_CAPREL", "expressible_fraction": None}
            continue
        count = exact = changed = switching = 0
        magnitude = reproduced = 0.
        for row in rows:
            mask = row["mask"]
            fit = representation(row["state"], row["teacher"], law)
            count += int(mask.sum())
            exact += int((mask & fit["expressible"]).sum())
            changed += int((mask & (fit["deviation"] != 0.)).sum())
            switching += int((mask & ~row["ready"]).sum())
            magnitude += float(fit["deviation_magnitude"][mask].sum())
            reproduced += float(fit["reproduced_magnitude"][mask].sum())
        result[law] = dict(status="measured", cav_ticks=count, nonzero_deviation_ticks=changed,
            expressible_fraction=exact/count if count else None,
            expressible_deviation_fraction=reproduced/magnitude if magnitude else None,
            absolute_deviation_mps2_sum=magnitude, controller_switch_ticks=switching)
    result["non_residual_components"] = {
        "lane_change_ticks": sum(int((r["mask"] & r["lane_changed"]).sum()) for r in rows),
        "ramp_merge_decisions": sum(int((r["mask"] & (r["state"]["lane"] == 2) &
                                          (r["teacher_lane"] == 1)).sum()) for r in rows),
        "native_lane_target_disagreements": sum(int((r["mask"] &
                         (r["teacher_lane"] != r["native_lane"])).sum()) for r in rows),
        "lane_controller": "MAPPO head vs student native WM lanes; no residual lane authority",
    }
    return result


def ledger_rows(rows, **provenance):
    for row in rows:
        fit = representation(row["state"], row["teacher"])
        cap = representation(row["state"], row["teacher"], "cap_relative") \
              if caprelative_available() else None
        for b, k in row["mask"].nonzero().tolist():
            yield dict(provenance, tick=row["tick"], batch=b, slot=k,
                entrant=bool(row["entrant"][b, k]), lane=int(row["state"]["lane"][b, k]),
                mappo_executed=float(row["teacher"][b, k]),
                ctrl0_executed=float(fit["ctrl0"][b, k]), delta=float(fit["deviation"][b, k]),
                current_expressible=bool(fit["expressible"][b, k]),
                cap_relative_expressible=None if cap is None else bool(cap["expressible"][b, k]),
                lane_changed=bool(row["lane_changed"][b, k]),
                teacher_lane=int(row["teacher_lane"][b, k]),
                native_lane=int(row["native_lane"][b, k]), commanded=bool(row["ready"][b, k]))


def parser(description=__doc__):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--wm", required=True)
    p.add_argument("--wm-config", required=True)
    p.add_argument("--mappo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="configs/c4_policy14_candA.toml")
    p.add_argument("--seed", type=int, default=965160)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p


def main(argv=None):
    p = parser()
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--horizon", type=int, default=60)
    args = p.parse_args(argv)
    check_seed(args.seed)
    if args.rollouts < 1 or args.horizon < 1:
        p.error("positive rollouts and horizon required")
    os.environ.update(C4_EXECUTION_PARITY="2", C4_SUMO_LEADER_RULE="1")
    torch.set_num_threads(1)
    from wms.control.train_cli import _read_config
    model, teacher, receipts = load_assets(args.wm, args.wm_config, args.mappo, args.device)
    train, _ = load_training_starts(_read_config(args.config), model, args.seed)
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    totals = []
    with (out / "actions.jsonl").open("x", encoding="utf-8") as stream:
        for i in range(args.rollouts):
            rows = []
            batch = train.sample(1, device=args.device)
            rollout(model, batch, Feedback(teacher, record=rows.append), horizon_s=args.horizon,
                    seed=args.seed + i)
            totals.extend(rows)
            for row in ledger_rows(rows, rollout=i, starts=batch.metadata):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    report = dict(schema="distill_representability/1", wm=receipts, seed=args.seed,
                  horizon_s=args.horizon, learned_mix=True, **summarize(totals))
    (out / "REPRESENTABILITY.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
