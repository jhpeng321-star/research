"""Report-only 60 s evidence as soon as its fixed cohort finishes; never authorises re-arm."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import drive as D


def main():
    os.nice(19)
    while time.time() < D.DEADLINE:
        files = list((D.ROOT / "on/wm_coop").glob("w*_t*.json"))
        if len(files) == 33 and all("error" not in json.loads(p.read_text()) for p in files):
            break
        time.sleep(15)
    else:
        return
    prior = Path.home() / "mappodiag"
    for tag in ("off", "on"):
        with (D.ROOT / "logs" / f"preview60_{tag}.log").open("ab") as log:
            subprocess.run([sys.executable, str(prior / "analyze_b.py"), "--wm-coop", str(D.ROOT / tag / "wm_coop"),
                "--coopmeas-jobs", str(Path.home() / "coopmeas/run/jobs"), "--src", str(prior / "src_bsrc"),
                "--thresholds", str(Path.home() / "coopmeas/thresholds.json"),
                "--out", str(D.ROOT / tag / "B60_RESULTS.json")], cwd=D.REPO, env=D.ENV,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    import analyze as N
    off, on = [json.loads((D.ROOT / tag / "B60_RESULTS.json").read_text()) for tag in ("off", "on")]
    aa, bb = off["coop_points"], on["coop_points"]
    assert len(aa) == len(bb) == 33 and [r["unit"] for r in aa] == [r["unit"] for r in bb]
    rows = [{"world": a["world"], "off_all": a["wm_all"], "off_b": a["wm_b"],
             "on_all": b["wm_all"], "on_b": b["wm_b"], "sumo_all": a["sumo_all"], "sumo_b": a["sumo_b"]}
            for a, b in zip(aa, bb)]

    def contrast(rr, side):
        return N.rate(rr, side + "_all") - N.rate(rr, side + "_b")
    contrasts = {s: N.boot(rows, lambda rr: contrast(rr, s)) for s in ("off", "on", "sumo")}
    gap = N.boot(rows, lambda rr: abs(contrast(rr, "off") - contrast(rr, "sumo"))
                 - abs(contrast(rr, "on") - contrast(rr, "sumo")))
    result = {"status": "REPORT-ONLY PREVIEW; all 300 s clocks/required evidence must finish before adoption",
              "n": 33, "vacate_contrast_per_cav_h_ci95": contrasts, "absolute_gap_reduction_ci95": gap,
              "native_counts": N.native_counts(aa, bb),
              "off_summary": {k: v for k, v in off.items() if k != "coop_points"},
              "on_summary": {k: v for k, v in on.items() if k != "coop_points"}}
    N.save("PRIMARY60_PREVIEW.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("native_counts", "off_summary", "on_summary")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        D.write(D.ROOT / "PREVIEW_FAILED.json", {"error": repr(error), "time": time.time()})
        raise
