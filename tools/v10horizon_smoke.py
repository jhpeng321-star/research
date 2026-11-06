"""V10HORIZON CPU smoke: plateau curriculum + prioritised replay on tiny TRAIN-val windows.

No GPU, no SUMO, no evaluation seeds.  This host has no ``data_cache`` pack, so the smoke
uses deterministic synthetic TRAIN-val windows (the same ``TrajectoryDataset.from_arrays``
fixture the train-layer unit tests use) shaped for the full 10/30/60/120 s curriculum.

It writes ``runs/active/c4_wm_v10horizon_smoke/`` (train_log.jsonl, val_log.jsonl,
SMOKE_EVIDENCE.json) and prints a compact per-horizon loss trace.  The assertions are in
``tests/unit/train/test_v10horizon.py``; this script is the evidence producer.

    python tools/v10horizon_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT, ROOT / "tests" / "unit" / "train"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402

from _compat import FlatConfig, legacy  # noqa: E402
from synth import make_arrays, small_model  # noqa: E402
from wms.data.replay_flags import compute_replay_flags  # noqa: E402
from wms.train.replay import PriorityReplayDataset  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402

train_mod = legacy("train")
OUT = ROOT / "runs" / "active" / "c4_wm_v10horizon_smoke"


def _dataset_windows(n=4, steps=240, k=8):
    arrays = make_arrays(steps=steps, k=k, n=n, seed=5)
    # The synthetic platoon uses lanes 0/1 only; put one recorded slot on the ramp lane so
    # the merge-zone replay flag has a real positive in this smoke (unit tests cover the rest).
    arrays["states"][:, :, k - 2, 1:4] = np.array([0.0, 0.0, 1.0], np.float32)
    return train_mod.TrajectoryDataset.from_arrays(arrays)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = _dataset_windows()
    flags = compute_replay_flags(base)
    train = PriorityReplayDataset(
        base, merge_zone=flags["merge_zone"], cav_follower=flags["cav_follower"],
        merge_oversample=4.0, cav_oversample=3.0, error_max_boost=8.0,
        error_alpha=1.0, error_decay=0.9)
    flat = dict(device="cpu", threads=2, num_workers=0, seed=11,
                batch_size=2, truncation_steps=8,
                unroll_curriculum_s=[10.0, 30.0, 60.0, 120.0],
                stage_steps=[6, 6, 6, 6],
                epochs_per_stage=1, steps_per_epoch=2000,
                lr=5.0e-4, grad_clip=1.0,
                select_by_long_horizon=False,
                val_every=3, val_max_windows=4,
                # V10HORIZON switches
                plateau_advance=True, plateau_patience=1, plateau_min_delta=1.0,
                plateau_min_steps=1, plateau_max_steps=0,
                replay_enabled=True, replay_merge_oversample=4.0,
                replay_cav_follower_oversample=3.0, replay_error_max_boost=8.0,
                replay_error_alpha=1.0, replay_error_decay=0.9,
                replay_probe_windows=4, replay_refresh_every=0,
                out_dir=str(OUT))
    trainer = WmSTrainer(FlatConfig(flat), small_model(seed=11), train,
                         val_dataset=base, out_dir=OUT)
    trainer.fit()

    train_rows = [json.loads(x) for x in (OUT / "train_log.jsonl").read_text().splitlines()]
    val_rows = [json.loads(x) for x in (OUT / "val_log.jsonl").read_text().splitlines()]
    by_stage: dict[int, list[dict]] = {}
    for row in train_rows:
        by_stage.setdefault(int(row["step"]), []).append(row)

    def slope(stage_s: float) -> dict:
        rows = [r for r in train_rows if float(r["stage_s"]) == stage_s]
        if not rows:
            return {}
        first, last = float(rows[0]["loss"]), float(rows[-1]["loss"])
        best = min(float(r["loss"]) for r in rows)
        ys = [float(r["loss"]) for r in rows]
        slope = float(np.polyfit(np.arange(len(ys)), ys, 1)[0]) if len(ys) > 1 else 0.0
        return {"n_steps": len(rows), "first_loss": first, "last_loss": last,
                "min_loss": best, "slope_per_step": slope,
                "decreased": bool(slope < 0.0 or last < first or best < first)}

    plateau_events = [r for r in val_rows if r.get("curriculum_advance_reason")]
    replay_rows = [r for r in val_rows if "replay" in r]
    evidence = {
        "schema": "v10horizon_cpu_smoke/1",
        "device": "cpu", "sumo": False, "gpu": False,
        "n_windows": len(base), "window_steps": int(base[0]["states"].shape[0]) - 1,
        "replay_flags": {k: flags[k] for k in
                         ("n_windows", "n_merge_zone", "n_cav_follower", "n_both")},
        "curriculum_s": [10.0, 30.0, 60.0, 120.0],
        "stage_plan": [{"stage_s": p.stage_s, "unroll_steps": p.unroll_steps,
                        "cap_steps": p.steps} for p in trainer.stages],
        "loss_by_horizon": {f"{h:g}": slope(h) for h in (10.0, 30.0, 60.0, 120.0)},
        "plateau_events": plateau_events,
        "replay_stage_stats": replay_rows,
        "total_optimizer_steps": len(train_rows),
    }
    (OUT / "SMOKE_EVIDENCE.json").write_text(json.dumps(evidence, indent=2) + "\n",
                                             encoding="utf-8")
    print(json.dumps({"loss_by_horizon": evidence["loss_by_horizon"],
                      "plateau_events": [(e.get("stage_s"), e.get("curriculum_advance_reason"))
                                         for e in plateau_events],
                      "out": str(OUT)}, indent=2))
    ok = True
    for h in (30.0, 60.0):
        row = evidence["loss_by_horizon"][f"{h:g}"]
        ok = ok and bool(row.get("decreased"))
    ok = ok and bool(plateau_events)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
