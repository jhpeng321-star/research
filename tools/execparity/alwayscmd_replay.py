"""Offline replay (iii) of the 00:20 always-commanded law on its own SUMO logs.

Reads the production jobs' per-tick execution ledgers and episode tapes (already
fetched read-only), replays each logged state + logged residual through the
imagination mirror ``wms.control.alwayscmd.executed`` and reports the executed
acceleration error against SUMO's realised acceleration, the issued-command
shortfall SUMO's own speedMode-27 cap produced, and the WM native-head error on
the same rows. PASS = pooled executed RMSE <= pooled native-head RMSE.

S-EXECPARITY2: the imagination mirror now executes commanded CAVs through the
SAME shared exact law as the deployment executor
(``wms.control.sumo_idm``: SUMO 1.27.1 ``MSCFModel_IDM::_v`` on the designed
``cav_typical`` parameters, legacy gap convention, lane speed limits, exact
safe-speed cap). ``mirror_from_issued`` additionally replays SUMO's own issued
command through the shared cap only, isolating the cap/state residual error
from the base-law error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).resolve().parent)]

from replay import sha, states_from_episode, summary
from wms.control.alwayscmd import (BACKEND_CONTRACT, executed, inferred_base,
                                   inferred_cap)
from wms.control.native_cav import head_native_accel

torch.set_num_threads(1)


def log_rows(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if float(r["t_s"]) >= 300. - 1e-9:
                rows.append(r)
    return rows


def world_rows(job, *, residual_reference="base"):
    """Map every execution-log row to its tape column and replay the law."""
    tape_path = job / "deploy/episode/episode.npz"
    with np.load(tape_path, allow_pickle=False) as z:
        tape = dict(z)
    times = tape["time_s"]
    vids = tape["vehicle_ids"]
    state_at = states_from_episode(tape)
    rows = log_rows(job / "execution_log.jsonl.gz")
    if any(r.get("residual_reference", "base") != residual_reference for r in rows):
        raise ValueError("execution log residual_reference differs from replay law")
    if residual_reference == "ctrl0":
        result = json.loads((job / "result.json").read_text(encoding="utf-8"))
        if result.get("caprel_source_sha256") != caprel_sources():
            raise ValueError("deployment/replay caprel source hashes differ")
    out = []
    for r in rows:
        t = float(r["t_s"])
        i = int(np.searchsorted(times, t))
        if i + 1 >= len(times) or abs(times[i] - t) > 1e-6:
            continue
        slot_ids = vids[tape["vid_index"][i]]
        hits = np.flatnonzero(slot_ids == r["vehicle_id"])
        if len(hits) != 1:
            continue
        k = int(hits[0])
        stable = (tape["present"][i, k] and tape["present"][i + 1, k]
                  and tape["is_cav"][i, k] and tape["lane"][i, k] >= 0
                  and tape["vid_index"][i + 1, k] == tape["vid_index"][i, k])
        if not stable:
            continue
        v0 = float(tape["v_mps"][i, k])
        if abs(v0 - float(r["speed_before_mps"])) > 1e-6:
            continue
        realised = (float(tape["v_mps"][i + 1, k]) - v0) / 0.5
        st = state_at(i)
        keep = torch.nonzero(st["present"][0]).flatten()
        local = int(np.flatnonzero(keep.numpy() == k)[0])
        st = {key: value.index_select(1, keep) for key, value in st.items()}
        residual = torch.zeros_like(st["v_mps"])
        residual[0, local] = float(r["residual_mps2"])
        with torch.no_grad():
            head = float(head_native_accel(st)[0, local])
            base = float(inferred_base(st, None)[0, local])
            cap = float(inferred_cap(st)[0, local])
            imag = float(executed(st, residual, residual_reference=residual_reference)[0, local])
        mirror = max(-v0 / .5, min(float(r["issued_mps2"]), cap))
        out.append(dict(t_s=t, k=k, speed=v0, residual=float(r["residual_mps2"]),
                        sumo_base=float(r["base_law_mps2"]), issued=float(r["issued_mps2"]),
                        realised=realised, imagined=imag, head=head,
                        inferred_base=base, cap=cap, mirror=mirror))
    return out, sha(tape_path)


def caprel_sources():
    return {name: sha(ROOT / "src/wms/control" / name)
            for name in ("residual.py", "alwayscmd.py", "sumo_idm.py")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, required=True, help="dir of fetched s*_alwaysx job outputs")
    ap.add_argument("--driver", type=Path, required=True)
    ap.add_argument("--residual-reference", choices=("base", "ctrl0"), default="base")
    ap.add_argument("--identity", type=Path, required=True,
                    help="fetched canary CTRL0_IDENTITY.json (zero_ctrl0_identity source)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True,
                    help="deployment-gate receipt written only when the registered criteria PASS")
    args = ap.parse_args()
    os.environ["C4_EXECUTION_DRIVER"] = str(args.driver.resolve())
    jobs = sorted(p for p in args.jobs.iterdir()
                  if p.is_dir() and (p / "execution_log.jsonl.gz").is_file())
    assert jobs, "no fetched always-commanded job outputs"
    worlds, hashes = {}, {}
    pooled = {k: [] for k in ("imagined_minus_realised", "issued_minus_realised",
                              "head_minus_realised", "inferred_minus_sumo_base",
                              "mirror_minus_realised", "residual", "realised")}
    for job in jobs:
        rows, tape_sha = world_rows(job, residual_reference=args.residual_reference)
        assert rows, f"no stable rows replayed for {job.name}"
        arr = {k: np.array([r[k] for r in rows]) for k in
               ("imagined", "issued", "realised", "head", "inferred_base",
                "sumo_base", "residual", "cap", "mirror")}
        err = arr["imagined"] - arr["realised"]
        iss = arr["issued"] - arr["realised"]
        head_err = arr["head"] - arr["realised"]
        base_err = arr["inferred_base"] - arr["sumo_base"]
        mirror_err = arr["mirror"] - arr["realised"]
        worlds[job.name] = {
            "n_rows": len(rows),
            "row_cadence_s": float(np.diff(np.unique([r["t_s"] for r in rows])).min()),
            "imagined_minus_realised": summary(err),
            "issued_minus_realised": summary(iss),
            "head_minus_realised": summary(head_err),
            "inferred_base_minus_sumo_base": summary(base_err),
            "mirror_from_issued_minus_realised": summary(mirror_err),
            "base_exact_fraction_1e-9": float((np.abs(base_err) < 1e-9).mean()),
            "residual": summary(arr["residual"]),
            "realised": summary(arr["realised"]),
            "issued_at_or_below_realised": float((iss <= 1e-3).mean()),
            "within_native_head_band": float(np.sqrt((err * err).mean())
                                             <= np.sqrt((head_err * head_err).mean())),
        }
        hashes[job.name] = {"episode_sha256": tape_sha,
                            "execution_log_sha256": sha(job / "execution_log.jsonl.gz"),
                            "result_sha256": sha(job / "result.json")}
        for key, values in (("imagined_minus_realised", err), ("issued_minus_realised", iss),
                            ("head_minus_realised", head_err),
                            ("inferred_minus_sumo_base", base_err),
                            ("mirror_minus_realised", mirror_err),
                            ("residual", arr["residual"]), ("realised", arr["realised"])):
            pooled[key].extend(values.tolist())
        print(job.name, len(rows), flush=True)
    rng = np.random.default_rng(20261028)
    names = list(worlds)
    draws = rng.integers(0, len(names), (5000, len(names)))
    bootstrap = {}
    for key in ("imagined_minus_realised", "head_minus_realised"):
        x = np.array([worlds[n][key]["rmse"] for n in names], dtype=float)
        bootstrap[key] = {"world_mean_rmse": float(x.mean()),
                          "ci95": np.quantile(x[draws].mean(1), [.025, .975]).tolist()}
    pooled_out = {k: summary(v) for k, v in pooled.items()}
    result = {
        "schema": "execparity_alwayscmd_replay/2",
        "backend_contract": BACKEND_CONTRACT,
        "imagination_base_law": "exact_shared_executor_law/sumo_idm/1",
        "driver_sha256": sha(args.driver),
        "decision": "00:20",
        "worlds": worlds, "jobs": hashes, "pooled": pooled_out,
        "world_bootstrap": bootstrap,
        "pass_pooled_executed_rmse_within_head_rmse":
            pooled_out["imagined_minus_realised"]["rmse"]
            <= pooled_out["head_minus_realised"]["rmse"],
        "measured_real_sumo": True,
        "limitations": [
            "log cadence is the composer call grid; realised uses tape first-substep speed difference",
            "allowed speed uses the registered lane limits (mainline 31.29 / ramp 22.22);",
            "junction-internal connector speed (26.75) and foe yielding are not observable in a WM frame",
            "native head is an approximate reference band, not an executor",
        ],
    }
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    if args.residual_reference == "ctrl0":
        if identity.get("residual_reference") != "ctrl0":
            raise ValueError("caprel requires a fresh cap-relative zero-identity canary")
        if identity.get("caprel_source_sha256") != caprel_sources():
            raise ValueError("caprel identity source binding differs from replay")
        result["residual_reference"] = "ctrl0"
    result["zero_ctrl0_identity"] = bool(identity.get("bit_identical") is True)
    result["passed"] = bool(result["pass_pooled_executed_rmse_within_head_rmse"]
                            and result["zero_ctrl0_identity"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if result["passed"]:
        receipt = {
            "schema": "execparity_always_commanded_acceleration/1",
            "backend_contract": BACKEND_CONTRACT,
            "measured_real_sumo": True,
            "zero_ctrl0_identity": True,
            "passed": True,
            "worlds": [name.split("_")[0][1:] for name in worlds],
            "driver_sha256": sha(args.driver),
            "native_arm_head_error": {"rmse": pooled_out["head_minus_realised"]["rmse"]},
            "c1_residual_execution_error": {"rmse": pooled_out["imagined_minus_realised"]["rmse"]},
            "error": {"ours": {"rmse": pooled_out["imagined_minus_realised"]["rmse"]}},
            "imagination_base_law": result["imagination_base_law"],
            "shared_module_sha256": hashlib.sha256(
                (ROOT / "src/wms/control/sumo_idm.py").read_bytes()).hexdigest(),
            "decision": "00:20",
            "report": str(args.out),
        }
        if args.residual_reference == "ctrl0":
            receipt["residual_reference"] = "ctrl0"
            receipt["caprel_source_sha256"] = caprel_sources()
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
        print(f"RECEIPT {args.receipt}")
    else:
        print("REGISTERED CRITERIA FAILED; no deployment receipt written")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("worlds", "jobs")}, indent=2))


if __name__ == "__main__":
    main()
