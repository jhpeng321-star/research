"""Workspace-only causal TRAIN preflight; never substitutes for S5 qualification.

Run with the registered round-8 stage300 checkpoint via --checkpoint for useful
motion evidence. Without it this diagnoses wiring at initialization only. No
driver sidecars, simulator state flags, future motion, or evaluation episodes
are loaded. Unsupported events and incomplete histories block qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wms.config.load import load_train_config
from wms.cli.assemble import build_physics
from wms.data.pack_history import prior_history
from wms.model import ModelSpec, WmSModel
from wms.model.checkpoint import load_checkpoint
from wms.train.inputs import build_targets, prepare_inputs
from wms.train.losses.merge_pivot import MergePivotConfig, MergePivotTerm, score_chunk
from wms.train.settings import TrainSettings


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
POOL = (("w963004m1900p40t300__native", 600, 240),
        ("w963004m1900p40t300__native", 660, 600),
        ("w963007m2100p10t300__native", 600, 240),
        ("w963007m2100p10t300__native", 660, 600))


def local(path):
    path = Path(path)
    if any(word in str(path).lower() for word in FORBIDDEN):
        raise ValueError("prohibited path")
    resolved = path.resolve()
    if any(word in str(resolved).lower() for word in FORBIDDEN) or not resolved.is_relative_to(ROOT):
        raise ValueError("inputs must be inside this worktree")
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=local)
    parser.add_argument("--windows", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(713)
    cfg = load_train_config(ROOT / "configs/c4_wm_v10round9.toml")
    settings = TrainSettings.from_config(cfg)
    if args.checkpoint is None:
        model = WmSModel(ModelSpec.from_mapping(cfg.to_legacy_flat()), physics=build_physics(cfg)).eval()
    else:
        loaded = load_checkpoint(args.checkpoint, model_config={
            "merge_decision": "first_passage", "merge_passage_law": "hybrid"})
        if float(loaded.payload_meta.get("stage_s", -1)) != 300.:
            raise ValueError("the registered stage300 checkpoint is required")
        model = loaded.model.eval()
    rows = []
    with torch.no_grad():
        for episode, start, steps in POOL[:args.windows]:
            print(json.dumps(dict(phase="causal_replay", episode=episode,
                                  offset_s=start * .5, horizon_s=steps * .5)), flush=True)
            # Only this fixed TRAIN pool is admitted, before touching any array.
            seed = int(episode[1:7])
            if 964000 <= seed <= 964199:
                raise ValueError("evaluation seeds are refused")
            folder = local(ROOT / "_sample/v5_native/episodes" / episode)
            class HistoryArrays:
                def array(self, name):
                    if name not in ("veh_kin", "veh_mask", "target_mask"):
                        raise ValueError("unexpected history array")
                    return np.load(local(folder / f"{name}.npy"), mmap_mode="r", allow_pickle=False)

                def run_start(self, tick):
                    return 0
            batch = {}
            for key, name in (("states", "veh_kin"), ("present", "veh_mask"), ("is_cav", "is_cav")):
                array = np.load(local(folder / f"{name}.npy"), mmap_mode="r", allow_pickle=False)
                batch[key] = torch.from_numpy(np.array(array[start:start + steps + 1], copy=True)).unsqueeze(0)
            for prefix, kw in (("hist", {}), ("driver_hist", dict(
                    history_s=model.spec.history_s, history_stride_s=model.spec.history_stride_s))):
                hist, mask = prior_history(HistoryArrays(), start, batch["present"][0, 0].numpy(), **kw)
                batch[prefix] = torch.from_numpy(hist).unsqueeze(0)
                batch[prefix + "_mask"] = torch.from_numpy(mask).unsqueeze(0)
            inp = prepare_inputs(batch, model, settings, torch.device("cpu"), steps)
            targets = build_targets(inp, batch, settings, model.physics.dt_s,
                                    torch.device("cpu"), physics=model.physics)
            targets.prepared["merge_pivot_source"] = model, inp
            term = MergePivotTerm(MergePivotConfig(tf_weight=.6, z_tf=395.2, z_on=2.15,
                                                   passage_law="hybrid", quadrature_order=32))
            term.prepare(targets)
            prep = targets.prepared["merge_pivot"]
            print(json.dumps(dict(phase="likelihood", recorded_events=int(prep["event"].sum()),
                                  physical_vetoes=int((prep["event"] & ~prep["geometry"]["eligible"]).sum()))),
                  flush=True)
            c = model.merge_pivot(prep["geometry"])
            params = model.merge_pivot.params()
            counts, nll = {}, 0.
            # Trim non-risk padding BEFORE integration. Each presence lifecycle
            # is scored in full; unknown-age cases remain explicitly enumerated.
            for k in range(c.shape[-1]):
                cuts = [0, *torch.nonzero(prep["reset"][0, :, k]).flatten().tolist(), steps]
                cuts = sorted(set(cuts))
                for left, right in zip(cuts[:-1], cuts[1:]):
                    indices = torch.nonzero(prep["risk"][0, left:right, k]).flatten()
                    if not len(indices):
                        continue
                    a, b = left + int(indices[0]), min(right, left + int(indices[-1]) + 2)
                    def cut(x):
                        return x[:, a:b, k:k + 1]
                    boundary = cut(c)
                    if not bool(prep["known"][0, a, k]):
                        # The score is masked by the declared incomplete history;
                        # avoid spending quadrature work on an unscored posterior.
                        boundary = torch.full_like(boundary, -torch.inf)
                    score, _, stats = score_chunk(
                        boundary, params["tau"], cut(prep["risk"]), cut(prep["event"]),
                        known_entry=cut(prep["known"]), reset=cut(prep["reset"]),
                        eligible=cut(prep["geometry"]["eligible"]), w=params["w"])
                    nll += float(score)
                    for key, value in stats.items():
                        counts[key] = counts.get(key, 0) + value
            rows.append(dict(episode=episode, offset_s=start * .5, horizon_s=steps * .5,
                             recorded_events=int(prep["event"].sum()),
                             recorded_physical_veto_events=int((prep["event"] & ~prep["geometry"]["eligible"]).sum()),
                             competing_exits=int(prep["competing_exit"].sum()),
                             vanished_exits=int(prep["vanished"].sum()),
                             nll=nll if np.isfinite(nll) else None, finite_nll=bool(np.isfinite(nll)),
                             counts=counts))
            print(json.dumps(dict(phase="window_complete", **rows[-1]), allow_nan=False), flush=True)
    blockers = []
    if args.checkpoint is None:
        blockers.append("No workspace round8 stage300 checkpoint: initialization-only replay, not S5 evidence.")
    if any(row["recorded_physical_veto_events"] for row in rows):
        blockers.append("Recorded events have exact physical/conflict vetoes; F3 reconciliation is required.")
    if any(row["counts"].get("unknown_age", 0) for row in rows):
        blockers.append("Incomplete survivor histories remain; F2 initialization is required.")
    if any(not row["finite_nll"] for row in rows):
        blockers.append("Infinite exact likelihood prevents optimizer updates on this pool.")
    if not sum(row["counts"].get("positive_mass_events", 0) for row in rows):
        blockers.append("No supported scored positive event; the S5 event gate is not met.")
    report = dict(status="blocked" if blockers else "preflight_passed", blockers=blockers,
                  optimizer_updates=0, motion="shared frozen WM causal replay",
                  checkpoint=str(args.checkpoint) if args.checkpoint else None,
                  z_tf=395.2, z_on=2.15, tf_on_weights=[.6, .4], rows=rows)
    print(json.dumps(report, allow_nan=False))
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
