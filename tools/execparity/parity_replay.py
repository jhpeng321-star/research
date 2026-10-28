"""Compare opt-in imagination execution with recorded first-substep SUMO acceleration.

Every tenth control tick in each of ten worlds, fixed before reading outcomes.
WM3 driver inference uses only the masked pre-command observable history. No fit,
future arrival, type assignment, simulator or evaluation-based calibration.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import ROOT, executor, sha, states_from_episode, summary
from wms.control.execution import inferred_follow_cap, executed_switch_residual as executed_residual
from wms.control.native_cav import head_native_accel
from wms.data.history import long_history_features
from wms.eval.backend import load_backend
from traffic_shepherd.c4.recorder import episode_kin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--driver", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert sha(args.wm).startswith("41e18f09")
    if args.driver:
        os.environ["C4_EXECUTION_DRIVER"] = str(args.driver.resolve())
    else:
        os.environ.pop("C4_EXECUTION_DRIVER",None)
    on, _ = executor(args.policy)
    model = load_backend(args.wm, cav_exec_cap="idm").model
    model.requires_grad_(False)
    result, errors, head_errors = {}, {a: [] for a in ("ours", "mappo")}, []
    residual_errors = []
    for seed in range(977000,977020,2):
        result[str(seed)] = {}
        for arm in ("ours", "mappo", "native"):
            path = args.data / f"e3final/main/e3final/shard0/jobs/s{seed}_{arm}/deploy/episode/episode.npz"
            with np.load(path, allow_pickle=False) as tape:
                z = dict(tape)
            state_at = states_from_episode(z)
            kin = torch.as_tensor(episode_kin(z))
            present = torch.as_tensor(z["present"])
            err, native_err, cap_values, residual_err, imag_safety_delta, follow_delta = [], [], [], [], [], []
            for c, t in enumerate(z["control_time_s"]):
                if t < 300 or int(round(t-300)) % 10:
                    continue
                i = int(np.searchsorted(z["time_s"], t))
                if i+1 >= len(z["time_s"]):
                    continue
                st = state_at(i)
                # Keep all slots seen in the historical window, including departed
                # leaders. Mask reused identities exactly as the deployment history.
                n_hist = int(round(float(model.history_s)/.5))+1
                begin = max(0, i-n_hist+1)
                keep = torch.nonzero(present[begin:i+1].any(0)).flatten()
                st = {key: value.index_select(1, keep) for key,value in st.items()}
                indices = keep.numpy()
                target = torch.as_tensor(z["vid_index"][begin:i,indices] == z["vid_index"][begin+1:i+1,indices])
                with torch.no_grad():
                    hist, mask = long_history_features(kin[begin:i+1].index_select(1,keep),
                        present[begin:i+1].index_select(1,keep), target, st["present"][0],
                        history_s=model.history_s, history_stride_s=model.history_stride_s, dt_s=.5)
                    _, _, params = model.encode_drivers(st, driver_history=hist[None],
                                                        driver_history_mask=mask[None])
                    cap = inferred_follow_cap(st, params)[0].numpy()
                    head = head_native_accel(st)[0].numpy()
                valid = (z["present"][i,indices] & z["is_cav"][i,indices]
                         & (z["lane"][i,indices] >= 0) & z["present"][i+1,indices]
                         & (z["vid_index"][i,indices] == z["vid_index"][i+1,indices]))
                real = (z["v_mps"][i+1,indices].astype(float)-z["v_mps"][i,indices])/.5
                native_err.extend((head[valid]-real[valid]).tolist())
                follow_delta.extend((real[valid]-cap[valid]).tolist())
                if arm != "native":
                    rq = z["issued_acceleration_mps2"][c,indices]
                    valid &= np.isfinite(rq)
                    # Replay the actual recorded absolute request through the same
                    # cap executor. C1 r>0 disappears; MAPPO still has braking authority.
                    predicted = np.maximum(np.minimum(np.minimum(rq, 1.2), cap),
                                           -z["v_mps"][i,indices]/.5)
                    err.extend((predicted[valid]-real[valid]).tolist())
                    cap_values.extend(cap[valid].tolist())
                    if arm == "ours":
                        from wms.control.nativeinit import compose_native
                        from wms.control.safety import safe_accel
                        with torch.no_grad():
                            r = torch.as_tensor(on(st).residual,dtype=torch.float32)
                            executed = executed_residual({**st,"cav_exec_cap_mps2":torch.as_tensor(cap)[None]},
                                r,torch.as_tensor(head)[None])[0].numpy()
                            raw = compose_native(torch.as_tensor(head)[None],r,-3.,2.6)
                            soft = safe_accel(st,raw,accel_max=2.6)
                        residual_err.extend((executed[valid]-real[valid]).tolist())
                        imag_safety_delta.extend((soft-raw)[0].numpy()[valid].tolist())
            result[str(seed)][arm] = {"episode_sha256": sha(path), "native_head_error": summary(native_err),
                                      "realised_minus_inferred_follow": summary(follow_delta)}
            if arm == "native":
                head_errors.extend(native_err)
            else:
                errors[arm].extend(err)
                result[str(seed)][arm].update(error=summary(err), inferred_cap=summary(cap_values))
                if arm == "ours":
                    residual_errors.extend(residual_err)
                    result[str(seed)][arm].update(residual_execution_error=summary(residual_err),
                        imagination_only_safe_accel_delta=summary(imag_safety_delta))
        print(seed, flush=True)
    aggregate = {arm: summary(err) for arm,err in errors.items()}
    native = summary(head_errors)
    result = {"schema": "execparity_recorded_acceleration/1", "wm_sha256": sha(args.wm),
              "driver_sha256": sha(args.driver) if args.driver else None,
              "checkpoint_sha256": sha(args.policy),
              "sampling": "every tenth post-warmup control tick; first .5s substep; all stable CAVs",
              "worlds": result, "error": aggregate, "native_arm_head_error": native,
              "c1_residual_execution_error": summary(residual_errors),
              "target": "each commanded-arm RMSE <= same-panel native-arm head RMSE",
              "passed": all(v["rmse"] <= native["rmse"] for v in aggregate.values())
                        and summary(residual_errors)["rmse"] <= native["rmse"],
              "limitations": ["old tapes omit prospective getFollowSpeed, allowedSpeed, signal and stop state",
                  "WM3 inferred settings are frozen; no fit on E3 evaluation states",
                  "full closed-loop SUMO trajectory parity remains an unrun canary"]}
    assert args.out.parent.is_dir()
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "worlds"}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
