"""Explore rollout diagnostics: verify against stored, classify jams, growth shape."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
import diaglib as DL  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def series(state: dict, key: str) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in state[key]], dtype=float)


def jam_class(state: dict) -> dict:
    """Compare predicted vs recorded merge congestion from the 0.5 s series."""
    t = state["t"]
    n = len(t)
    # index 120 == 60 s, 240 == 120 s
    def peak(sec_key, sect, metric, upto):
        vals = []
        for i in range(0, upto + 1):
            row = sec_key.get(str(i))
            if row and row.get(sect, {}).get(metric) is not None:
                vals.append(row[sect][metric])
        return max(vals) if vals else 0.0

    def mean(sec_key, sect, metric, upto):
        vals = []
        for i in range(0, upto + 1):
            row = sec_key.get(str(i))
            if row and row.get(sect, {}).get(metric) is not None:
                vals.append(row[sect][metric])
        return float(np.mean(vals)) if vals else 0.0

    up = min(n - 1, 240)
    pr, rc = state["sec_pred"], state["sec_rec"]
    out = {}
    for label, upto in (("60", min(n - 1, 120)), ("120", up)):
        out[label] = {
            "q_pred": peak(pr, "merge", "queue", upto),
            "q_rec": peak(rc, "merge", "queue", upto),
            "stop_pred": mean(pr, "merge", "stopped", upto),
            "stop_rec": mean(rc, "merge", "stopped", upto),
            "stop_pred_up": mean(pr, "up", "stopped", upto),
            "stop_rec_up": mean(rc, "up", "stopped", upto),
            "sp_pred": mean(pr, "merge", "mean_speed", upto),
            "sp_rec": mean(rc, "merge", "mean_speed", upto),
        }
    return out


def growth_shape(state: dict) -> dict:
    """Classify the HDV median |ds| curve: gradual vs sudden jump."""
    ds = series(state, "ds_med")
    n = len(ds)
    cs = np.where(np.isnan(ds), 0.0, ds)
    # sudden jump: largest single-step increase in first 240 steps
    d = np.diff(cs)
    j = int(np.argmax(d)) if d.size else -1
    jump = float(d[j]) if j >= 0 else 0.0
    # fit ds(t) over 0..240 to linear and quadratic (ignore nan)
    idx = np.flatnonzero(~np.isnan(ds[:min(n, 241)]))
    shape = {}
    if idx.size > 10:
        tt = idx * 0.5
        yy = ds[idx]
        try:
            lin = np.polyfit(tt, yy, 1)
            quad = np.polyfit(tt, yy, 2)
            lin_res = float(np.sqrt(np.mean((np.polyval(lin, tt) - yy) ** 2)))
            quad_res = float(np.sqrt(np.mean((np.polyval(quad, tt) - yy) ** 2)))
            shape = {"lin_slope": float(lin[0]), "lin_res": lin_res,
                     "quad_res": quad_res, "quad_a": float(quad[0]),
                     "r2_lin": float(1 - lin_res ** 2 / max(np.var(yy), 1e-9))}
        except Exception:
            pass
    return {"jump_max": jump, "jump_t_s": float(j * 0.5) if j >= 0 else None,
            "ds60": state.get("ds60"), "ds120": state.get("ds120"), **shape}


def main() -> int:
    stored = json.loads((D / "stored_summary.json").read_text())
    for model, fn in (("c1_struct", "roll_struct_u_same.jsonl"),
                      ("oldhist", "roll_oldhist_u_same.jsonl")):
        rows = load_jsonl(D / fn)
        good = [r for r in rows if "error" not in r]
        nblow = sum(1 for r in good if r["blow"])
        print(f"== {model}: n={len(good)} blow={nblow} "
              f"stored_blow={stored[model]['u_same']['n_blow']}")
        # agreement
        st = {s["world"]: s for s in DL.load_states_file(
            DL.SHIFT / model / f"states__{model}__u_same.json")}
        dis = [r["world"] for r in good if DL.blowup(st.get(r["world"], {}), "pi") != r["blow"]]
        dis = [r["world"] for r in good if DL.blowup(st.get(r["world"], {}), "pi") != r["blow"]]
        print("   disagreements:", len(dis), dis[:5])
        # jam classes split by blow
        def summarize(sel):
            rows2 = [r for r in good if r["blow"] == sel]
            qdiff, sdiff = [], []
            for r in rows2:
                jc = jam_class(r)
                qdiff.append(jc["120"]["q_pred"] - jc["120"]["q_rec"])
                sdiff.append(jc["120"]["stop_pred"] - jc["120"]["stop_rec"])
            return (len(rows2), float(np.mean(qdiff)) if qdiff else None,
                    float(np.mean(sdiff)) if sdiff else None,
                    float(np.mean([x > 20 for x in qdiff])) if qdiff else None)
        for sel in (True, False):
            n, mq, ms, frac_phantom = summarize(sel)
            print(f"   blow={sel}: n={n} mean qdiff={mq:.1f} mean stopdiff={ms:.2f} "
                  f"phantom_frac={frac_phantom}")
        # growth shape of blowups
        jump_t = []
        for r in good:
            if r["blow"]:
                g = growth_shape(r)
                if g["jump_max"] > 5.0 and g["jump_t_s"] is not None:
                    jump_t.append(g["jump_t_s"])
        if jump_t:
            print(f"   sudden-jump states: {len(jump_t)}/{(nblow)} "
                  f"median jump t={np.median(jump_t):.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
