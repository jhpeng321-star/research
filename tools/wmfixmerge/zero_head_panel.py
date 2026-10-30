"""Zero-head candidate path on the corrected WMREQUAL3 validation panel.

WMFIX_REVIEW most-valuable step (addendum 03:55 item 4): conserve_queue +
learned admission ON, merge-response head at zero (plain WM2 weights, no head),
K8, native lanes, mix ON, fixed source clock.  Emits one ``batch_NNN.json`` per
validation group with the same row shape as the WMREQUAL3 batches, so the
upstream residual under the new entry path can be compared against WM2 and the
admission/queue change can be isolated from the head (bundle confound).
Resumable: complete batches are preserved, ``--groups`` selects a subset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

BOUND_FILES = ("tools/wmrequal/score.py", "tools/wmrequal/native_lanes.py",
               "tools/wmrequal/labels.py", "tools/wmfixmerge/runtime.py",
               "tools/errsrc/inference.py", "src/wms/model/entries.py",
               "src/wms/model/learned_admission.py", "src/wms/model/wms.py")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.25,
                        help="of total; 0.25 of 48 GB is the reviewer's 12 GB slice")
    parser.add_argument("--groups", default="0-35")
    parser.add_argument("--code-sha", default="")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in args.groups.split("-"))
    import torch
    from dataclasses import replace
    import importlib.util
    from tools.wmfixmerge.data import WM2_SHA
    if digest(plan["reference"]) != WM2_SHA:
        raise ValueError("reference is not the registered WM2 checkpoint")
    torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), 0)
    torch.cuda.reset_peak_memory_stats()
    from tools.wmcombined import e3_score as e3
    from wms.data.c4_pairs import PairedBranchSource
    from tools.wmrequal.grouped_feedback import feedback
    import wms.control.entrant as entrant
    sys.path.insert(0, str(ROOT / "tools/wmrequal"))
    spec = importlib.util.spec_from_file_location("wmfix_zero_head_score",
                                                  ROOT / "tools/wmrequal/score.py")
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    real = json.loads(Path(plan["validation_real"]).read_text())["panels"]["validation"]
    root = Path(plan["validation_root"])
    manifest = json.loads((root / "C4_EPISODES_MANIFEST.json").read_text())
    split = json.loads((root / "SPLIT.json").read_text())
    entries = [e for e in manifest["episodes"]
               if split["worlds"][str(e["seed"])] == "validation"]
    if {e["pair_key"] for e in entries} != {r["pair_key"] for r in real}:
        raise ValueError("corrected validation real/source group mismatch")
    source = PairedBranchSource(root, horizon_steps=600, history_s=60., history_stride_s=1.,
        entries_v2=True, junction_fix="ffill", command_known=True, arm_kinds=("B", "POLICY"),
        role="train", role_of={e["world"]: "train" if e in entries else "val"
                               for e in manifest["episodes"]})
    if len(source.groups) != 36 or source.horizon != 600:
        raise ValueError("incomplete corrected validation cohort")
    specs = json.loads(Path(plan["controllers"]).read_text())["controllers"]
    entrant.feedback_from_specs = feedback
    _, settings, model = e3.load_for_scoring(Path(plan["base_config"]), Path(plan["reference"]))
    if model.merge_response is not None:
        raise ValueError("zero-head path must run WM2 weights without a head")
    settings = replace(settings, seed=140014, entrant_control=True,
        entrant_controllers={a: specs[a] for a in ("mappo", "ours_c1", "coopoff_c1")},
        entry_queue_state=True, entry_pad_auto=True, entry_pad_max=1024,
        entry_insert_rule="sumo_like_v2")
    model = model.to(torch.device(args.device)).eval().requires_grad_(False)
    model.cav_exec_cap = "idm"
    original_entries = score.Entries

    class ConservingEntries(original_entries):
        def __init__(self, process):
            process.cfg = replace(process.cfg, conserve_queue=True)
            super().__init__(process)

    score.Entries = ConservingEntries
    launch = dict(pid=os.getpid(), argv=sys.argv, time=time.time(), device=args.device,
                  memory_fraction=args.memory_fraction, code_sha=args.code_sha,
                  gpu_before=None, files={name: digest(ROOT / name) for name in BOUND_FILES},
                  inputs=dict(reference=digest(plan["reference"]),
                              base_config=digest(plan["base_config"]),
                              controllers=digest(plan["controllers"]),
                              validation_real=digest(plan["validation_real"])))
    try:
        import subprocess
        launch["gpu_before"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except Exception:
        pass
    with (out / "LAUNCH.json").open("x") as stream:
        json.dump(launch, stream, indent=2)
    print("OWN_PID", os.getpid(), "groups", hi - lo + 1, "steps 600", flush=True)
    done = []
    for i in range(lo, hi + 1):
        path = out / f"batch_{i:03d}.json"
        if path.is_file():
            rows = json.loads(path.read_text())["rows"]
            if len(rows) == 32:
                print("PRESERVED_BATCH", i, flush=True)
                done.append(i)
                continue
        began = time.time()
        rows = score.score_batch(model, settings, e3, source, [i], real,
                                 torch.device(args.device), 600, False,
                                 native_lanes=True, mix_modes=(True,), learned_admission=True)
        if len(rows) != 32:
            raise ValueError(f"group {i} produced {len(rows)} rows, expected 32")
        receipt = dict(seconds=round(time.time() - began, 1),
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                       group=i, pair_key=rows[0]["pair_key"])
        with path.open("x") as stream:
            json.dump(dict(receipt=receipt, rows=rows), stream, allow_nan=False)
        print("BATCH", i, receipt["seconds"], receipt["peak_gpu_bytes"], flush=True)
        done.append(i)
    score.Entries = original_entries
    with (out / "DONE.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), completed=len(done), groups=done,
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                       wall_seconds=round(time.time() - launch["time"], 1),
                       files=launch["files"], inputs=launch["inputs"],
                       code_sha=args.code_sha), stream, indent=2)
    print("DONE", len(done), flush=True)


if __name__ == "__main__":
    main()
