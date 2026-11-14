"""Validation sanity gate for run2 (BEFORE registration v2 / holdout).

On the 4 validation roots x t0 {120..480 step 60} (28 windows):
- model members + ensemble vs B1/B2 speed RMSE at H = 60 s;
- predicted collisions per 1000 vehicle-steps (H=60);
- truth/pred base rates for the S flags (any TTC<3 s, any hard brake).
Gate: ens speed RMSE < min(B1, B2) AND ens collisions <= 1/1000.
"""
import json
import sys
import time

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine as E  # noqa: E402
import p3lib as P  # noqa: E402

T0 = time.time()
WDIR = HERE / "weights" / "run2"


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def main():
    dev = P.device()
    splits = P.roots_by_split()
    members = sorted(f.name for f in WDIR.glob("member*.pt"))
    assert len(members) >= 3, "run2 members missing"
    res = {}
    for variant in (["member" + str(i) for i in range(len(members))]
                    + ["ens_mean", "B1_persistence", "B2_idm_normal"]):
        rows = []
        for root in splits["val"]:
            for t0 in range(120, 481, 60):
                if variant == "ens_mean":
                    snaps_list = []
                    for w in members:
                        r = E.Rollout(root, float(t0), kind="model",
                                      weights=WDIR / w, dev=dev).run()
                        snaps_list.append(r.snapshots)
                    ra = r.ra
                    kk = r.k0
                    ens = {j: E.ens_snapshot(snaps_list, j)
                           for j in range(0, 241)}
                    snaps, rr = ens, ra
                elif variant.startswith("member"):
                    i = int(variant.replace("member", ""))
                    r = E.Rollout(root, float(t0), kind="model",
                                  weights=WDIR / f"member{i}.pt",
                                  dev=dev).run()
                    snaps, ra, kk = r.snapshots, r.ra, r.k0
                else:
                    kind = ("persistence" if variant == "B1_persistence"
                            else "idm_normal")
                    r = E.Rollout(root, float(t0), kind=kind,
                                  dev=dev).run()
                    snaps, ra, kk = r.snapshots, r.ra, r.k0
                rows.append(E.window_metrics(snaps, ra, kk, 60.0))
        n = len(rows)
        res[variant] = dict(
            windows=n,
            speed_rmse_h60=round(float(np.sqrt(np.mean(
                [r["speed_rmse"] ** 2 for r in rows]))), 4),
            collisions_per_1000=round(float(np.mean(
                [r["collisions_per_1000"] for r in rows])), 4),
            ttc3_recall=round(float(np.mean(
                [r["any_ttc3_pred"] for r in rows])), 4),
            ttc3_true_rate=round(float(np.mean(
                [r["any_ttc3_true"] for r in rows])), 4),
            hb_recall=round(float(np.mean(
                [r["any_hb_pred"] for r in rows])), 4),
            hb_true_rate=round(float(np.mean(
                [r["any_hb_true"] for r in rows])), 4),
            pos_rmse_h60=round(float(np.sqrt(np.mean(
                [r["pos_rmse"] ** 2 for r in rows]))), 2),
            delay_bias=round(float(np.mean(
                [r["delay_pred"] - r["delay_true"] for r in rows])), 2))
        log(f"{variant}: {res[variant]}")

    gate = dict(
        ens_speed_rmse_h60=res["ens_mean"]["speed_rmse_h60"],
        B1_speed_rmse_h60=res["B1_persistence"]["speed_rmse_h60"],
        B2_speed_rmse_h60=res["B2_idm_normal"]["speed_rmse_h60"],
        ens_collisions_per_1000=res["ens_mean"]["collisions_per_1000"])
    gate["beat_B1"] = gate["ens_speed_rmse_h60"] < gate["B1_speed_rmse_h60"]
    gate["beat_B2"] = gate["ens_speed_rmse_h60"] < gate["B2_speed_rmse_h60"]
    gate["collisions_ok"] = gate["ens_collisions_per_1000"] <= 1.0
    gate["PASS"] = gate["beat_B1"] and gate["beat_B2"] and \
        gate["collisions_ok"]
    gate["members"] = members
    out = dict(gate=gate, per_variant=res)
    P.dump_json(HERE / "results" / "val_gate_run2.json", out)
    print(json.dumps(gate, indent=1))
    return 0 if gate["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
