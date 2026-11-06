"""Decision-time compute of the confident gate / K-candidate plan check on CPU (no SUMO).

    python tools/sumocredit/gate_timing.py --episodes <converted c4 episode dir(s)> --policy P.pt \
        --wm-ckpt wm_s.pt --wm-config config.resolved.json --threads 2 --out timing.json

States are real TRAIN WM-frame states read from converted episodes (``veh_kin``/``veh_mask``/
``is_cav`` at a tick, the loop14_returns convention), compacted to the present slots like the
deployment check; the driver history is passed as explicitly unobserved (same tensor shapes and
encoder cost as a real 60 s history).  The reliability tables are synthetic placeholders (timing only: every unit's
evidence is computed regardless of the table).  Reports the wall time of ``ConfidentGate.decide``
per decision for each (K, horizon, unit).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def load_state(episode: Path, t_s: float):
    times = np.load(episode / "time_s.npy")
    t = int(np.searchsorted(times, t_s - 1e-6))
    kin = np.load(episode / "veh_kin.npy", mmap_mode="r")[t].copy()
    present = np.load(episode / "veh_mask.npy", mmap_mode="r")[t].astype(bool)
    cav = np.load(episode / "is_cav.npy", mmap_mode="r")[t].astype(bool)
    lane = np.where(present & (kin[:, 1:4].sum(-1) > 0), kin[:, 1:4].argmax(-1), -1)
    keep = np.flatnonzero(present & (lane >= 0))

    def row(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x)[keep], dtype=dtype)[None]
    return {"s_m": row(kin[:, 0]), "v_mps": row(kin[:, 4]), "a_mps2": row(kin[:, 5]),
            "length_m": row(kin[:, 6]), "lane": row(lane, torch.long),
            "present": torch.ones(1, keep.size, dtype=torch.bool), "is_cav": row(cav, torch.bool)}


def main(argv=None) -> int:
    from wms.control.confident_gate import ArmImagination, ConfidentGate, ReliabilityTable
    from wms.control.policy import load_policy_components
    from wms.control.policy14_cli import load_members
    from wms.control.policy14_probes import missing_history

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--t", type=float, default=420.0)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--wm-ckpt", required=True)
    ap.add_argument("--wm-config", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ks", default="1,4,8")
    ap.add_argument("--horizons", default="10,20,30")
    ap.add_argument("--units", default="fleet,group")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    models, _ = load_members([args.wm_ckpt], [args.wm_config], "cpu")
    policy, encoder, rules, _ = load_policy_components(args.policy, device="cpu")
    states = [load_state(Path(e), args.t) for e in args.episodes]
    rows = []
    for unit in args.units.split(","):
        for h in (float(x) for x in args.horizons.split(",")):
            for k in (int(x) for x in args.ks.split(",")):
                meta = {"split": "TRAIN", "unit": unit, "reference": "native", "horizon_s": h,
                        "state_source": "belief"}
                table = ReliabilityTable.fit([0.0, 1.0], [0, 1], meta)
                mean = ReliabilityTable.fit([0.0, 1.0], [0, 1], {**meta, "comparison": "mean"})
                imag = ArmImagination(models, policy, encoder, lane_rules=rules, reference="native",
                                      horizon_s=h)
                gate = ConfidentGate(imag, table, unit=unit, state_source="belief", n_candidates=k,
                                     mean_table=mean if k > 1 else None)
                for st in states:
                    ids = [f"v{j}" if bool(c) else None for j, c in enumerate(st["is_cav"][0].tolist())]
                    for _ in range(args.repeats):
                        t0 = time.perf_counter()
                        d = gate.decide(st, now_s=0.0, ids=ids, **missing_history(models[0], st))
                        rows.append({"unit": unit, "horizon_s": h, "K": k, "slots": int(st["s_m"].shape[1]),
                                     "cavs": int(st["is_cav"].sum()), "units": len(d.units),
                                     "arms": 1 + len(d.units) * k, "wall_s": time.perf_counter() - t0})
                done = [r for r in rows if (r["unit"], r["horizon_s"], r["K"]) == (unit, h, k)]
                print(json.dumps({"unit": unit, "horizon_s": h, "K": k,
                                  "median_wall_s": float(np.median([r["wall_s"] for r in done])),
                                  "max_wall_s": float(np.max([r["wall_s"] for r in done]))}), flush=True)
    summary = {}
    for r in rows:
        summary.setdefault(f"{r['unit']}|H{int(r['horizon_s'])}|K{r['K']}", []).append(r["wall_s"])
    doc = {"threads": args.threads, "torch": torch.__version__, "episodes": args.episodes, "t_s": args.t,
           "median_wall_s": {k: float(np.median(v)) for k, v in summary.items()},
           "max_wall_s": {k: float(np.max(v)) for k, v in summary.items()}, "rows": rows}
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
