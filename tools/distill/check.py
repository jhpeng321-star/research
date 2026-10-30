"""Frozen-panel K8 paired imagination: distilled vs MAPPO vs CTRL0.

Reuses W-verdict's wmcombined.e3_score panel/group loader, delay_proxy and
PairedRandom streams. Frozen verdict receipts select the panels; outcomes
never select starts. All three controllers are feedback-driven, with entries
and endogenous exits. Both the t0-HDV score and fleet+source J are retained.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from wms.control.start_states import StartBatch
from tools.wmcombined.e3_score import group_rows
from tools.wm_eval_common import delay_proxy

from .represent import parser
from .runtime import (Feedback, check_metadata, check_seed, configure_execution,
                      load_assets, rollout, safe_path, sha256)

HORIZONS = (60, 120, 300)
ARMS = ("distilled", "mappo", "ctrl0")


def frozen_source(root, receipt_path, panel):
    """Reuse the exact W-verdict pair keys; refuse a source/panel mismatch."""
    from wms.data.c4_pairs import PairedBranchSource
    from tools.wmcombined.e3_score import build_source
    root, receipt_path = safe_path(root), safe_path(receipt_path)
    manifest = json.loads(safe_path(root / "C4_EPISODES_MANIFEST.json").read_text())
    check_metadata(manifest["episodes"])
    for entry in manifest["episodes"]:
        safe_path(root / entry["dir"])
    receipt = json.loads(receipt_path.read_text())
    keys = [str(row["pair_key"]) for row in receipt["rows"]]
    expected = 48 if panel == "pairs300" else 36
    if len(keys) != expected or len(set(keys)) != expected:
        raise ValueError(f"{panel}: expected the registered {expected}-group W-verdict receipt")
    if panel == "validation":
        # build_source's split membership assertions are the original scorer's.
        safe_path(root / "SPLIT.json")
        source = build_source(root, "validation")
    else:
        source = PairedBranchSource(root, horizon_steps=600, history_s=60., history_stride_s=1.,
            entries_v2=True, junction_fix="ffill", command_known=True,
            role_of={e["world"]: "train" for e in manifest["episodes"]}, role="train", name="pairs300")
    by_key = {g["pair_key"]: i for i, g in enumerate(source.groups)}
    if not set(keys) <= set(by_key) or source.receipt["dropped_groups_t0_mismatch"]:
        raise ValueError("frozen panel keys/identical t0 contract do not match source")
    return source, [by_key[k] for k in keys], {
        "panel_receipt_sha256": sha256(receipt_path), "manifest_sha256": sha256(root / "C4_EPISODES_MANIFEST.json"),
        "pair_keys": keys, "groups": expected}


def start_from_record(record, device):
    """Only t0 and pre-t0 history enter imagination; no future controls/masks."""
    kin = torch.as_tensor(record["states"][0], device=device).float()
    present = torch.as_tensor(record["present"][0], device=device).bool()
    lane = torch.where(kin[:, 1:4].sum(-1) > 0, kin[:, 1:4].argmax(-1), -1)
    state = dict(s_m=kin[:, 0][None], v_mps=kin[:, 4][None], a_mps2=kin[:, 5][None],
        lane=lane[None], length_m=kin[:, 6][None], present=present[None],
        is_cav=torch.as_tensor(record["is_cav"][0], device=device).bool()[None])
    history = {target: torch.as_tensor(record[key], device=device)[None]
               for key, target in (("hist", "history"), ("hist_mask", "history_mask"),
                                   ("driver_hist", "driver_history"), ("driver_hist_mask", "driver_history_mask"))
               if key in record}
    def feature(key):
        return torch.as_tensor(record[key], device=device).float()[None] if key in record else None
    return StartBatch(state, history, [dict(record.get("meta", {}))], feature("entry_feat"),
                      entry_queue=feature("entry_queue"), entry_feat_control=feature("entry_feat_control"))


def costs(batch, roll, entries, horizons=HORIZONS):
    """Original W-verdict t0-HDV kernel plus the explicit candidate-A objective."""
    from wms.control.contract import RewardWeights
    from wms.control.objective14 import authoritative_reward
    n = batch.state["s_m"].shape[1]
    speed = np.concatenate((batch.state["v_mps"].cpu().numpy()[:, None],
                            roll["speeds"][:, :, :n].cpu().numpy()), 1)
    masks = np.concatenate((batch.state["present"].cpu().numpy()[:, None],
                            roll["masks"][:, :, :n].cpu().numpy()), 1)
    selected = (batch.state["present"] & ~batch.state["is_cav"]).cpu().numpy()
    totals = []
    for t, before in enumerate(entries.states):
        after = {**before, "s_m": roll["positions"][:, t], "v_mps": roll["speeds"][:, t],
                 "lane": roll["lanes"][:, t], "present": roll["masks"][:, t]}
        queue = torch.as_tensor(entries.backlog[t], device=after["v_mps"].device,
                                dtype=after["v_mps"].dtype)
        part = authoritative_reward({**before, "entry_queue": queue}, after,
                                     roll["accel"][:, t], RewardWeights())
        totals.append(part["J"].cpu())
    cumulative = torch.stack(totals, 1).cumsum(1)
    return {str(h): {"t0_hdv": delay_proxy(speed[0], masks[0], selected[0], h),
                     "fleet_source": float(cumulative[0, round(h/.5)-1])} for h in horizons}


def paired_summary(draws):
    """Pair *within* each future before reducing K8; never subtract quantiles."""
    output = {}
    for h in draws[0][ARMS[0]]:
        output[h] = {}
        for metric in ("t0_hdv", "fleet_source"):
            output[h][metric] = {}
            for left, right in itertools.combinations(ARMS, 2):
                delta = np.asarray([d[left][h][metric] - d[right][h][metric] for d in draws])
                output[h][metric][f"{left}_minus_{right}"] = {
                    "draw_dJ": delta.tolist(), "mean": float(delta.mean()),
                    "median": float(np.median(delta)), "q10": float(np.quantile(delta, .1)),
                    "q90": float(np.quantile(delta, .9))}
    return output


def panel_summary(rows):
    from tools.wmcombined.e3_score import _boot_groups, _stable_seed
    result = {}
    for h in map(str, HORIZONS):
        result[h] = {}
        for metric in ("t0_hdv", "fleet_source"):
            result[h][metric] = {}
            for left, right in itertools.combinations(ARMS, 2):
                contrast = f"{left}_minus_{right}"
                worlds = {}
                for row in rows:
                    world = row["pair_key"].split("@")[0]
                    worlds.setdefault(world, []).append(row["paired"][h][metric][contrast]["mean"])
                point, interval = _boot_groups(list(worlds.values()), lambda values: float(np.mean(values)),
                                               seed=_stable_seed(metric + contrast, h))
                result[h][metric][contrast] = {"mean_dJ": point, "world_cluster_ci95": interval,
                                               "worlds": len(worlds), "starts": len(rows)}
    return result


def evaluate_start(model, teacher, student, encoder, batch, *, seed, paired_batch=None,
                   group_index=0, pad_max=288, horizons=HORIZONS, samples=8):
    draws = []
    law = getattr(student, "distillation_law", "current")
    for draw in range(samples):
        arm_costs = {}
        for arm, drive in (("distilled", "student"), ("mappo", "teacher"), ("ctrl0", "ctrl0")):
            feedback = Feedback(teacher, student, encoder, drive=drive, law=law)
            roll, entries, _ = rollout(model, batch, feedback, horizon_s=max(horizons), seed=seed,
                draw=draw, pad_max=pad_max, paired_batch=paired_batch, group_keys=[group_index])
            arm_costs[arm] = costs(batch, roll, entries, horizons)
        draws.append(arm_costs)
    return {"draw_costs": draws, "paired": paired_summary(draws)}


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--student", required=True)
    p.add_argument("--pairs300-root", required=True)
    p.add_argument("--validation-root", required=True)
    p.add_argument("--pairs300-panel", required=True, help="existing W-verdict JSON with its 48 row pair_keys")
    p.add_argument("--validation-panel", required=True, help="existing W-verdict validation JSON, 36 groups")
    p.add_argument("--pad-max", type=int, default=288)
    p.set_defaults(seed=140014)
    args = p.parse_args(argv)
    check_seed(args.seed)
    configure_execution(args.execution_driver)
    torch.set_num_threads(1)
    model, teacher, receipts = load_assets(args.wm, args.wm_config, args.mappo, args.device)
    from wms.control.policy import load_policy_components
    student, encoder, _, payload = load_policy_components(safe_path(args.student), device=args.device)
    if not student.residual_on_native or "distillation" not in payload:
        raise ValueError("expected a distilled residual candidate-A checkpoint")
    student.distillation_law = payload["distillation"]["signature"]["law"]
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    panels = {}
    from wms.data.collate import collate
    for name in ("pairs300", "validation"):
        source, indexes, receipt = frozen_source(getattr(args, name + "_root"),
                                                  getattr(args, name + "_panel"), name)
        rows = []
        for gi in indexes:
            key, reference, _ = group_rows(source, gi)
            # The source already checks identical t0; load just its reference start.
            record = source.group_records(gi, [reference], base_row=0)[0]
            batch = start_from_record(record, args.device)
            result = evaluate_start(model, teacher, student, encoder, batch,
                seed=args.seed + 7919 * gi, paired_batch=collate([record]), group_index=gi,
                pad_max=args.pad_max)
            row = {"pair_key": key, **result}
            rows.append(row)
            (out / f"{name}_{gi:03d}.json").write_text(json.dumps(row, allow_nan=False))
        panels[name] = {"receipt": receipt, "summary": panel_summary(rows), "rows": rows}
    report = {"schema": "distill_imagination_check/1", "wm": receipts,
              "student_sha256": sha256(args.student), "K": 8, "horizons_s": HORIZONS,
              "seed": args.seed, "pad_max": args.pad_max, "learned_mix": True,
              "scorer": "tools.wmcombined.e3_score group_rows + tools.wm_eval_common.delay_proxy",
              "contract": "frozen W-verdict starts; identity-keyed paired feedback; no recorded future inputs",
              "panels": panels, "gate_changed": False, "distilled_only": True}
    (out / "IMAGINATION.json").write_text(json.dumps(report, indent=1, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
