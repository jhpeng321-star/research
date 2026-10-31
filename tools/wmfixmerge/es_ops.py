"""Owned, receipt-first CPU launch helpers for S-WMFIX-ES; never signals a PID."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


BASE = "/root/autodl-tmp/c4"
COMMON = ["--config", BASE + "/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml",
          "--checkpoint", BASE + "/restart20261026/arm/loop/wm/c02/wm_s.pt",
          "--controllers", BASE + "/overnight20261028/W5_COMBINED/DESIGN.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "canary"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out-name")
    parser.add_argument("--arms", default="circles,ctrl0,mappo,pi")
    args = parser.parse_args()
    from tools.wmfixmerge.data import install_guard, safe, digest
    install_guard()
    root = safe(args.root)
    out = root / (args.out_name or ("probe_k8" if args.mode == "probe" else "es_canary_v3"))
    if not out.is_dir():
        raise ValueError("pre-created output directory required")
    # Never duplicate a launch after interruption.
    with (out / "DISPATCH.json").open("x") as stream:
        json.dump(dict(launcher=os.getpid(), mode=args.mode, prepared_sha256=digest(args.prepared)), stream)
    env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    jobs = []
    if args.mode == "probe":
        for future in range(8):
            jobs.append((f"future{future}", [sys.executable, "-m", "tools.wmfixmerge.es_probe",
                *COMMON, "--prepared", args.prepared, "--n-futures", "8", "--future-index", str(future),
                "--out", str(out / f"PROBE_f{future}.json")]))
    else:
        from tools.wmfixmerge.es import DEFAULTS, PARAMS
        for index, arm in enumerate(("circles", "ctrl0", "mappo", "pi")):
            if arm not in args.arms.split(","):
                continue
            tag = f"canary_{arm}"
            manifest = dict(gen=999, sweep_id=tag, incumbents=DEFAULTS,
                cells=[dict(cand=0, corpus="wm", group=0, group_label="wm:0", arm=index,
                    arm_name=arm, horizon=60, pair_key="w969400m1900p05t300@t300",
                    params=[DEFAULTS[p["name"]] for p in PARAMS])])
            path = out / f"MANIFEST_{tag}.json"
            with path.open("x") as stream:
                json.dump(manifest, stream)
            jobs.append((tag, [sys.executable, "-m", "tools.wmfixmerge.es", "--mode", "sweep",
                *COMMON, "--prepared", args.prepared, "--prepared-wmdata",
                str(root / "TRAIN_PREPARED_WMDATA.json"), "--wmdata-controllers",
                "tools/wmfixmerge/es_wmdata_controllers.json", "--manifest", str(path), "--out", str(out)]))
    pids = []
    for tag, cmd in jobs:
        with (out / (tag + ".log")).open("xb") as log:
            proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row = dict(pid=proc.pid, tag=tag, argv=cmd)
        with (out / (tag + "_PID.json")).open("x") as stream:
            json.dump(row, stream)
        pids.append(row)
    print(json.dumps(pids), flush=True)


if __name__ == "__main__":
    main()
