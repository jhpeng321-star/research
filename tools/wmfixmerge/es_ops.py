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
    parser.add_argument("mode", choices=("probe", "canary", "export-smoke", "physics-smoke", "drive", "finish-watch"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out-name")
    parser.add_argument("--arms", default="circles,ctrl0,mappo,pi")
    args = parser.parse_args()
    from tools.wmfixmerge.data import install_guard, safe, digest
    install_guard()
    root = safe(args.root)
    if args.mode == "finish-watch":
        out = root / "es_run_full"
        if (out / "FINISH_WATCH_PID.json").exists():
            raise ValueError("existing finish watcher; reconcile its PID first")
        env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        with (root / "es_finish_watch.log").open("xb") as log:
            proc = subprocess.Popen(["nice", "-n", "19", sys.executable, "-m",
                "tools.wmfixmerge.es_finish", "--watch", "--root", str(out)],
                env=env, cwd=str(Path(__file__).resolve().parents[2]), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid)), flush=True)
        return
    if args.mode == "drive":
        out = root / "es_run_full"
        if not out.is_dir() or any(out.iterdir()):
            raise ValueError("fresh pre-created ES output required")
        cmd = [sys.executable, "-m", "tools.wmfixmerge.es", "--mode", "drive", *COMMON,
            "--prepared", args.prepared, "--prepared-wmdata", "/home/ubuntu/wmfix_es/TRAIN_PREPARED_WMDATA.json",
            "--cpuq-prepared-wmdata", "/home/ubuntu/wmfix_es/TRAIN_PREPARED_WMDATA.json",
            "--registry", str(root / "TRAIN_PREPARED_COMBINED_FULL.json"),
            "--wmdata-controllers", "tools/wmfixmerge/es_wmdata_controllers.json",
            "--resolved-config", BASE + "/wmfixmerge20261031/train/candidate_config.json",
            "--out", str(out), "--cpuq-out", "/home/ubuntu/wmfix_es/es_run_full",
            "--gens", "8", "--popsize", "2", "--sigma", ".10", "--lr", ".01",
            "--fitness-horizon", "120", "--max-seconds", "12600", "--stop-patience", "3",
            "--gpuA-fit-groups", "0-11", "--cpuq-fit-groups", "none",
            "--gpuA-workers", "18", "--cpuq-workers", "30", "--account-for-probes"]
        with (root / "ES_FULL_DISPATCH.json").open("x") as stream:
            json.dump(dict(launcher=os.getpid(), argv=cmd), stream)
        env = dict(os.environ, PYTHONPATH="src:.", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        with (root / "es_full_drive.log").open("xb") as log:
            proc = subprocess.Popen(["nice", "-n", "19", *cmd], cwd=str(Path(__file__).resolve().parents[2]),
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with (root / "ES_FULL_PID.json").open("x") as stream:
            json.dump(dict(pid=proc.pid, argv=cmd), stream)
        print(json.dumps(dict(pid=proc.pid, out=str(out))), flush=True)
        return
    if args.mode in ("export-smoke", "physics-smoke"):
        from types import SimpleNamespace
        from tools.wmfixmerge.es import DEFAULTS, export_physical_checkpoint
        out = root / (args.out_name or "export_smoke")
        if not out.is_dir():
            raise ValueError("pre-created output directory required")
        opts = SimpleNamespace(config=COMMON[1], checkpoint=COMMON[3])
        export_physical_checkpoint(opts, out, 0, dict(gen=0, params=DEFAULTS))
        if args.mode == "physics-smoke":
            # Serialization-only fixture: no rollout, objective or calibration.
            changed = dict(DEFAULTS)
            changed["mainline_courtesy_zone_m"] += 1.
            export_physical_checkpoint(opts, out, 1, dict(gen=1, params=changed))
            from tools.wmcombined.e3_score import load_for_scoring
            shared = out / "generation_001_config.json"
            _, _, candidate = load_for_scoring(shared, out / "generation_001.pt")
            _, _, reference = load_for_scoring(shared, out / "generation_000.pt")
            if candidate.physics.mainline_courtesy_zone_m != changed["mainline_courtesy_zone_m"]:
                raise ValueError("candidate physics lost under shared canonical config")
            if reference.physics.mainline_courtesy_zone_m != DEFAULTS["mainline_courtesy_zone_m"]:
                raise ValueError("shared config erased reference physics")
            print("shared-config candidate/reference physical difference preserved", flush=True)
        print("canonical export/reload PASS", flush=True)
        return
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
