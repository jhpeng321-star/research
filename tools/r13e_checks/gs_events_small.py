"""R13E B3: same-event GAPACC gs equivalence on a SMALL recorded event set (CPU only).

Recorded lane2->1 insertions of the first ``--n-states`` R13E offline TRAIN-role panel
states; S-R12C encoder per-slot settings captured from the exact pre-t0 history are given
to BOTH the actual GAPACC ``install_gate`` and R13E ``secure_gap_margin``. Inference only;
no outputs are written except the JSON report (``--out``, created exclusively).

  nice -n 19 python -B tools/r13e_checks/gs_events_small.py --root /root/autodl-tmp/c4/v5 \
      --n-states 12 --out /root/autodl-tmp/c4/v5/diag_emerg/gs_events_small_<sha8>.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT, ROOT / "tools"):
    sys.path.insert(0, str(_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--n-states", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import torch
    from tools.r13e_checks import gs_equivalence as diff
    from tools.r13e_checks import offline as study
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles

    study.cpu_limits()
    root = diff.checked_path(args.root)
    work = root / "diag_emerg"
    out = Path(args.out)
    diff.checked_path(out.parent)
    docs, worlds = study.train_manifests([root / "data/v5_native", root / "data/genrand_plain"])
    records = study.load_train_states(work / "train_states48.json.gz", worlds)[:args.n_states]
    predictor, _capture, weights = study.load_exact_predictor(
        root / "runs/c4_wm_v10_round12c/wm_s.pt")
    reference = diff.module_at(work / "gapacc_ref/factworker.py", "gapacc_reference")
    extractor = diff.module_at(work / "gapacc_ref/ins.py", "gapacc_events")
    captured, cases = {}, []
    original_forward = predictor.model.forward

    def forward(*a, **kw):
        result = original_forward(*a, **kw)
        captured["dp"] = result["driver_parameters"].detach().cpu().numpy()[0]
        return result

    predictor.model.forward = forward
    for index, record in enumerate(records):
        study.seed_all(study.RNG_SEED + 2 * index)
        predictor.predict_states(record, {"arm": record["arm"]}, .5, study.RNG_SEED + 2 * index)
        dp = captured["dp"]
        t0, ids, kin, cav = t0_vehicles(record)
        teacher, present, _ = recorded_teacher(record, t0, ids, kin, 240, record["arm"],
                                               predictor.backend.tools)
        traj = teacher[0].numpy()
        for event in extractor.events(traj[..., 0], traj[..., 1], present[0].numpy(),
                                      traj[..., 2], cav):
            r, f, leader = (int(event[k]) for k in ("r", "f", "l"))
            c = diff.case(f"state{index}:row{event['tm']}:slot{r}",
                          v=float(event["vR"]), vf=float(event["vF"]), vl=float(event["vL"]),
                          gf=float(event["gf"]), gl=float(event["gl"]),
                          has_f=f >= 0, has_l=leader >= 0, dp=dp[[r, max(f, 0), max(leader, 0)]].copy())
            c["hdv"] = bool(event["hdvR"])
            cases.append(c)
    predictor.model.forward = original_forward
    assert study.state_digest(predictor.model.state_dict()) == weights, "checkpoint bytes changed"
    report = {"code_root": str(ROOT), "n_states": len(records), "recorded_events": len(cases),
              "reference_sha256": study.file_digest(work / "gapacc_ref/factworker.py"),
              "extractor_sha256": study.file_digest(work / "gapacc_ref/ins.py"),
              "definition": "R13E gs = GAPACC gs (T_back*v_back + relu(...), no s0); "
                            "invalid inputs fail closed", "events": {}}
    differences = []
    for name, selected in (("all", cases), ("hdv", [c for c in cases if c["hdv"]])):
        report["events"][name] = {}
        for dtype in (torch.float32, torch.float64):
            result = diff.compare(reference, selected, dtype, predictor.model.merge_pivot)
            report["events"][name][str(dtype)] = result
            differences.append(result["n"] - result["gate_equal"])
    report["verdict"] = "EQUIVALENT" if cases and not any(differences) else "NOT_EQUIVALENT"
    text = json.dumps(study.json_safe(report), indent=2, allow_nan=False) + "\n"
    with out.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"verdict": report["verdict"], "events": len(cases),
                      "differences": differences}))


if __name__ == "__main__":
    main()
