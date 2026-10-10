"""WD-01-PA2G-LC2 pre-change nominal reference (byte-identity anchor).

Run BEFORE any engine/variant edit: records the registered eval path
(eval_a2.window_rows, model 'v2g', replica=None, const off) on the FIRST
TWO worlds of lc1_a2_worlds32.json, all 18 windows, H in {10,30,60,120}.
The output rows are the frozen reference that the post-variant-impl
nominal path (all flags OFF) must reproduce byte-identically.

TRAIN only (split asserted per world before any run file is opened);
zero SUMO.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wdpaths  # noqa: E402
DATA_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT)

from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402
from partA_v2g.engine import load_params, param_set_id  # noqa: E402

OUT = HERE / "results" / "lc2"
N_REF_WORLDS = 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    worlds = json.loads(
        (HERE / "results" / "lc1_a2_worlds32.json").read_text())
    EA.OPENED_RUN_FILES.clear()
    EA.TRAIN_WORLD_NAMES.clear()
    plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
    meta = {w["world"]: w for w in plan["worlds"]}
    entries = []
    for name in worlds[:N_REF_WORLDS]:
        e = dict(meta[name])
        assert e.get("split") == "TRAIN", f"{name} not TRAIN"
        e["chunk"] = 4
        m = re.search(r"w\d+m(\d+)p(\d+)t(\d+)", name)
        e["demand"] = int(m.group(1)) if m else -1
        e["penetration"] = int(m.group(2)) / 100.0 if m else float("nan")
        e["native_run"] = DATA_ROOT / "runs" / "campaign_stage1_chunk4" / \
            f"{name}__native_base" / "run" / "physics_steps.jsonl"
        entries.append(e)
    EA.TRAIN_WORLD_NAMES.update(e["world"] for e in entries)
    print(f"LC2 pre-change nominal reference: {len(entries)} worlds "
          f"(first of lc1_a2_worlds32.json), param_set_id "
          f"{param_set_id()[:12]}", flush=True)
    EA.MAPPER = EA.ModelMapper()
    params = load_params()
    out = OUT / "nominal_ref_prechange.jsonl"
    n_rows = 0
    with open(out, "w", encoding="utf-8") as fh:
        for entry in entries:
            log = EA.NativeLog(entry)
            for k in range(1, 19):
                w = log.t0 + 30.0 * k
                if w > log.t_end + 1e-9:
                    break
                rows, _t = EA2.window_rows(
                    params, log, entry, w, None,
                    hook_of=lambda ww, vehs: EA2.native_control_hook(
                        ww, params, set(vehs) & log.cav_ids),
                    emit_const=False)
                assert all(r["model"] == "v2g" for r in rows)
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                    n_rows += 1
            print(f"  {entry['world']}: cumulative rows {n_rows}", flush=True)
    opened = sorted(EA.OPENED_RUN_FILES)
    EA.assert_train_only()
    (OUT / "nominal_ref_prechange.opened.json").write_text(
        json.dumps(opened, indent=1), encoding="utf-8")
    (OUT / "nominal_ref_prechange.meta.json").write_text(json.dumps(dict(
        purpose="WD-01-PA2G-LC2 byte-identity anchor, generated BEFORE "
                "the variant implementation touched engine.py",
        worlds=[e["world"] for e in entries], rows=n_rows,
        param_set_id=param_set_id(),
        generator="eval_a2.window_rows (model v2g, replica None, "
                  "emit_const False)",
        wall_clock_s=time.perf_counter() - t0,
        opened_run_files=opened), indent=1), encoding="utf-8")
    print(f"DONE {n_rows} rows -> {out} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
