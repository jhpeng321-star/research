"""R12 telemetry reads from the round-12 train log (reads, not reject rules).  Always exits 0.

Usage: telemetry_reads.py RUN_DIR OUT_JSON OUT_MD
From RUN_DIR/train_log.jsonl:
  (a) learned driver centre: key ``driver_centre`` (dict v0, T, s0, a, b in physical units) at steps
      1, 20, 40, 60, 80, 100, 110 and the last step; MISSING if the key is absent;
  (b) contrast courtesy trend: ``follower_courtesy_contrast_control_resid_mean`` over the steps with courtesy rows
      (finite value, and ``follower_courtesy_contrast_n_controls`` > 0 when logged): mean of the first 10 and the last
      10 such steps; READ flag "head contrast-limited" if the last-10 mean < -0.3 (not a base failure);
  (c) driver prior: round 12 launches with the prior OFF, so ``driver_prior`` / ``driver_prior_means`` must be absent
      from every step (status ABSENT-OK); if present the means of the first / last step are reported (PRESENT).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

STEPS = (1, 20, 40, 60, 80, 100, 110)
CONTRAST = "follower_courtesy_contrast_control_resid_mean"
N_CONTROLS = "follower_courtesy_contrast_n_controls"
TH_CONTRAST = -0.3


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _row_at(rows: list[dict], step: int):
    for r in rows:
        if r.get("step") == step:
            return r
    return rows[step - 1] if 0 < step <= len(rows) and all("step" not in r for r in rows) else None


def centre_read(rows: list[dict]) -> dict:
    if not any(isinstance(r.get("driver_centre"), dict) for r in rows):
        return {"status": "MISSING", "at_step": {}}
    at = {}
    for s in STEPS:
        r = _row_at(rows, s)
        at[str(s)] = r.get("driver_centre") if r is not None else None
    last = next((r for r in reversed(rows) if isinstance(r.get("driver_centre"), dict)), None)
    at["last"] = last.get("driver_centre") if last else None
    at["last_step"] = last.get("step") if last else None
    return {"status": "OK", "at_step": at}


def contrast_read(rows: list[dict]) -> dict:
    vals = [r[CONTRAST] for r in rows if _finite(r.get(CONTRAST))
            and (not _finite(r.get(N_CONTROLS)) or r.get(N_CONTROLS) > 0)]
    if not vals:
        return {"status": "MISSING", "n_steps": 0, "head_contrast_limited": None}
    first = sum(vals[:10]) / len(vals[:10])
    last = sum(vals[-10:]) / len(vals[-10:])
    return {"status": "OK", "n_steps": len(vals), "first10_mean": first, "last10_mean": last,
            "head_contrast_limited": bool(last < TH_CONTRAST)}


def prior_read(rows: list[dict]) -> dict:
    with_prior = [r for r in rows if "driver_prior" in r or "driver_prior_means" in r]
    if not with_prior:
        return {"status": "ABSENT-OK", "n_steps_with_prior": 0}
    return {"status": "PRESENT", "n_steps_with_prior": len(with_prior),
            "first_means": with_prior[0].get("driver_prior_means"), "last_means": with_prior[-1].get("driver_prior_means"),
            "first_driver_prior": with_prior[0].get("driver_prior"), "last_driver_prior": with_prior[-1].get("driver_prior")}


def reads(rows: list[dict]) -> dict:
    return {"n_steps": len(rows), "centre": centre_read(rows), "contrast": contrast_read(rows), "prior": prior_read(rows)}


def _fmt_centre(c) -> str:
    if not isinstance(c, dict):
        return "n/a"
    return ", ".join(f"{k} {v:.3g}" if _finite(v) else f"{k} {v}" for k, v in c.items())


def write_md(path: Path, res: dict) -> None:
    c, k, p = res["centre"], res["contrast"], res["prior"]
    L = ["# R12 telemetry reads (train_log.jsonl; reads, not reject rules)\n", f"Steps logged: {res['n_steps']}.\n",
         f"## (a) learned driver centre: {c['status']}\n"]
    if c["status"] == "OK":
        L += ["| step | centre (v0, T, s0, a, b) |", "|---|---|"]
        L += [f"| {s} | {_fmt_centre(v)} |" for s, v in c["at_step"].items() if s != "last_step"]
    L.append(f"\n## (b) contrast control residual trend: {k['status']}\n")
    if k["status"] == "OK":
        L.append(f"first-10 mean {k['first10_mean']:+.3f}, last-10 mean {k['last10_mean']:+.3f} over {k['n_steps']} steps "
                 f"with courtesy rows; head contrast-limited (last-10 < {TH_CONTRAST}): "
                 f"**{'YES' if k['head_contrast_limited'] else 'no'}** (read).")
    L.append(f"\n## (c) driver prior (launch: OFF): **{p['status']}**\n")
    if p["status"] == "PRESENT":
        L.append(f"prior telemetry on {p['n_steps_with_prior']} steps; means first {p['first_means']}, last {p['last_means']}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    run, out_json, out_md = Path(argv[0]), Path(argv[1]), Path(argv[2])
    log = run / "train_log.jsonl"
    try:
        res = reads(read_rows(log)) if log.exists() else {"n_steps": 0, "error": f"{log} missing",
                                                          "centre": {"status": "MISSING", "at_step": {}},
                                                          "contrast": {"status": "MISSING", "n_steps": 0},
                                                          "prior": {"status": "MISSING"}}
    except Exception as exc:  # exit 0 always
        res = {"n_steps": 0, "error": repr(exc), "centre": {"status": "ERROR", "at_step": {}},
               "contrast": {"status": "ERROR", "n_steps": 0}, "prior": {"status": "ERROR"}}
    out_json.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    write_md(out_md, res)
    print(f"telemetry: centre {res['centre']['status']}, contrast {res['contrast']['status']} "
          f"(limited={res['contrast'].get('head_contrast_limited')}), prior {res['prior']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
