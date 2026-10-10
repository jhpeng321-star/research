"""T1c: retrain the v3 vehicle-dynamics MLP with identical hyperparameters and
seeds from the SURVIVING train tracks (byte-exact copies), evaluate the new
checkpoint under the same protocol, and compare against the surviving one.

Steps
  1. copy the source train module + its wm dependency byte-exact into
     identity/retrain_src/ (sha256-verified);
  2. copy the 67 surviving train npz byte-exact into
     identity/retrain_src/tracks/train/ (sha256-verified per file);
  3. run the copied trainer (deterministic: torch seed 0, numpy seed 0);
  4. compare state dicts, evaluate with identity_eval protocol, apply the
     registered identity tolerances;
  5. save a modern (weights_only=True) retrained checkpoint + manifest.

Writes: results/retrain_run.json, results/identity_eval_retrained.json,
results/identity_verdict.json, identity/retrained/vehicle_dynamics_v3_retrained.pt
"""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import wd1lib as L

RSRC = L.IDENTITY / "retrain_src"
NEW_PT = L.IDENTITY / "retrained" / "vehicle_dynamics_v3_retrained.pt"


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    a, b = L.sha256_file(src), L.sha256_file(dst)
    assert a == b, f"copy mismatch {src} -> {dst}"
    return a


def main():
    t0 = time.time()
    log = dict(generated_utc=datetime.now(timezone.utc).isoformat())
    hashes_mod = {}
    for p in sorted(L.WM.glob("*.py")):
        hashes_mod[p.name] = copy_file(p, RSRC / p.name)
    log["copied_module_sha256"] = hashes_mod

    # 2. byte-exact train tracks
    train_files = sorted((L.WM / "tracks" / "train").glob("*.npz"))
    hashes = {}
    for f in train_files:
        hashes[f.name] = copy_file(f, RSRC / "tracks" / "train" / f.name)
    log["n_train_npz_copied"] = len(hashes)
    log["train_npz_sha256"] = hashes

    # 3. run the copied trainer (it reads RSRC/tracks/train, writes RSRC ckpt)
    cmd = [sys.executable, str(RSRC / "train_vehicle_dynamics.py")]
    print("RUN:", " ".join(cmd), flush=True)
    t1 = time.time()
    proc = subprocess.run(cmd, cwd=str(RSRC), capture_output=True, text=True)
    log["train_seconds"] = round(time.time() - t1, 1)
    log["train_returncode"] = proc.returncode
    log["train_stdout_tail"] = proc.stdout[-2500:]
    log["train_stderr_tail"] = proc.stderr[-1500:]
    print(proc.stdout[-2500:])
    if proc.returncode != 0:
        L.dump_json(L.RESULTS / "retrain_run.json", log)
        print("TRAINING FAILED", proc.stderr[-1500:])
        return 1
    src_ckpt = RSRC / "vehicle_dynamics.pt"
    log["retrained_raw_ckpt_sha256"] = L.sha256_file(src_ckpt)

    # 4a. state-dict comparison vs surviving
    surv = torch.load(L.SURV_CKPT, map_location="cpu", weights_only=False)
    newc = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    sd_a, sd_b = surv["model_state"], newc["model_state"]
    assert sorted(sd_a) == sorted(sd_b), "state dict keys differ"
    wdiff = {k: float((sd_a[k] - sd_b[k]).abs().max())
             for k in sd_a}
    n_same = sum(1 for v in wdiff.values() if v == 0.0)
    log["weights"] = dict(
        n_tensors=len(wdiff), n_identical_tensors=n_same,
        max_abs_diff=max(wdiff.values()),
        per_tensor_max_abs_diff=wdiff)
    log["x_mean_max_abs_diff"] = float(np.abs(surv["x_mean"] -
                                              newc["x_mean"]).max())
    log["x_std_max_abs_diff"] = float(np.abs(surv["x_std"] - newc["x_std"]).max())

    # 4b. evaluate retrained under the same protocol
    import identity_eval as IE
    IE.main("retrained", str(src_ckpt))
    ev_s = L.load_json(L.RESULTS / "identity_eval_surviving.json")
    ev_r = L.load_json(L.RESULTS / "identity_eval_retrained.json")
    ms, mr = ev_s["metrics"], ev_r["metrics"]
    delta = {k: mr[k] - ms[k] for k in ms if isinstance(ms[k], (int, float))}
    log["metric_deltas_retrained_minus_surviving"] = delta

    # prediction agreement on the identical dv-pair lists
    ps, pr = ev_s["dv_pairs"], ev_r["dv_pairs"]
    assert len(ps) == len(pr)
    a = np.array([p[3] for p in ps])
    b = np.array([p[3] for p in pr])
    agreement = dict(
        n_pairs=len(a),
        mean_abs_diff=float(np.abs(a - b).mean()),
        max_abs_diff=float(np.abs(a - b).max()),
        pearson=float(np.corrcoef(a, b)[0, 1]),
    )
    log["prediction_agreement_surviving_vs_retrained"] = agreement

    # 4c. registered identity verdict
    reg = dict(
        one_step_rmse_tol_mps=0.02,
        dv_mae_tol_mps=0.005,
        one_step_rmse_logged_mps=ms["mean_model_rmse_mps"],
        dv_mae_logged_mps=ms["dv_mae_model"],
        must_beat_persistence=ms["mean_persistence_rmse_mps"])
    d_rmse = mr["mean_model_rmse_mps"] - ms["mean_model_rmse_mps"]
    d_mae = mr["dv_mae_model"] - ms["dv_mae_model"]
    verdict = dict(
        registered_tolerances=reg,
        retrained_one_step_rmse_mps=mr["mean_model_rmse_mps"],
        abs_dev_rmse_vs_logged=abs(d_rmse),
        rmse_within_tol=bool(abs(d_rmse) <= reg["one_step_rmse_tol_mps"]),
        beats_persistence=bool(mr["mean_model_rmse_mps"] <
                               reg["must_beat_persistence"]),
        retrained_dv_mae_mps=mr["dv_mae_model"],
        abs_dev_mae_vs_logged=abs(d_mae),
        mae_within_tol=bool(abs(d_mae) <= reg["dv_mae_tol_mps"]),
        weights_max_abs_diff=log["weights"]["max_abs_diff"],
        n_weights_identical=log["weights"]["n_identical_tensors"],
        n_weights_total=log["weights"]["n_tensors"],
        pred_agreement=agreement,
    )
    verdict["identity_pass"] = bool(
        verdict["rmse_within_tol"] and verdict["beats_persistence"] and
        verdict["mae_within_tol"])
    log["verdict"] = verdict

    # 5. modern weights_only-safe copy of the retrained checkpoint
    tmod = L.train_mod()
    tmod_model = tmod.DynamicsMLP(in_dim=newc["arch"]["in_dim"],
                                  hidden=tuple(newc["arch"]["hidden"]))
    tmod_model.load_state_dict(newc["model_state"])
    meta = dict(
        format="wd01_part1_modern_v1",
        source_format="worldmodel_v3_vehicle train_vehicle_dynamics.py output",
        retrain_utc=log["generated_utc"],
        arch=dict(in_dim=newc["arch"]["in_dim"],
                  hidden=[int(h) for h in newc["arch"]["hidden"]]),
        x_mean=[float(x) for x in newc["x_mean"]],
        x_std=[float(x) for x in newc["x_std"]],
        config={k: (v if isinstance(v, (str, int, float)) else str(v))
                for k, v in newc["config"].items()},
        provenance=dict(
            surviving_ckpt_sha256=L.SURV_CKPT_SHA256,
            train_npz_sha256=hashes,
            copied_module_sha256=log["copied_module_sha256"],
            identity_pass=verdict["identity_pass"]),
        identity_verdict=verdict,
    )
    L.save_modern_ckpt(NEW_PT, tmod_model, meta)
    log["modern_ckpt"] = dict(path=str(NEW_PT),
                              sha256=L.sha256_file(NEW_PT),
                              bytes=NEW_PT.stat().st_size)
    chk = L.load_modern_ckpt(NEW_PT)
    log["modern_ckpt"]["weights_only_reload_ok"] = (
        sorted(chk[0]) == sorted(newc["model_state"]))

    L.dump_json(L.RESULTS / "retrain_run.json", log)
    L.dump_json(L.RESULTS / "identity_verdict.json", verdict)
    print(json.dumps({k: log[k] for k in
                      ("weights", "prediction_agreement_surviving_vs_retrained",
                       "verdict", "modern_ckpt", "train_seconds")}, indent=1)[:3000])
    print(f"TOTAL retrain pipeline {time.time() - t0:.0f}s")
    return 0 if verdict["identity_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
