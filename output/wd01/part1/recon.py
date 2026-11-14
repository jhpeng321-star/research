"""Recon (read-only). Resolves the 88 tags to regenerated runs (retry1 first),
samples run structure, dumps logged RESULTS.json summary. Prints compact JSON."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

SRC = Path(wdpaths.RESEARCH_DENSE_ROOT)
OUT = SRC / "output"
WM = OUT / "worldmodel_v3_vehicle"
REGEN = SRC / "runs" / "regenerated_v1"
RETRY = SRC / "runs" / "regenerated_v1_retry1"
HERE = Path(__file__).resolve().parent

def sha256(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

report, errs = {}, []

def guard(name, fn):
    try:
        report[name] = fn()
    except Exception as e:
        errs.append(f"{name}: {type(e).__name__}: {e}")

def find_run(tag):
    m = re.match(r"^cv(\d+)_(.+)$", tag)
    if not m:
        return None, None, "tag-shape"
    cands = []
    for base, pref in ((RETRY, "retry1"), (REGEN, "regen1")):
        for jd in sorted(base.glob(f"cv{m.group(1)}/*")):
            if jd.name == m.group(2) and (jd / "run").is_dir():
                cands.append((pref, jd / "run"))
    if not cands:
        return None, None, "not-found"
    return cands[0][0], cands[0][1], None

man = json.loads((WM / "build_tracks_manifest.json").read_text(encoding="utf-8"))

def do_manifest():
    rs = man["runs"]
    return dict(n_runs=len(rs), top_keys=sorted(man.keys()), run_keys=sorted(rs[0].keys()),
                n_pairs_total=man.get("n_pairs_total"),
                by_split={s: sum(1 for r in rs if r["split"] == s)
                          for s in sorted({r["split"] for r in rs})})
guard("manifest", do_manifest)

def do_map():
    out = []
    for r in man["runs"]:
        pref, rdir, err = find_run(r["tag"])
        if rdir is None:
            out.append(dict(tag=r["tag"], err=err))
            continue
        phys, tele = rdir / "physics_steps.jsonl", rdir / "telemetry.jsonl"
        out.append(dict(tag=r["tag"], split=r["split"], pref=pref, run_dir=str(rdir),
                        phys_sha256=sha256(phys) if phys.exists() else None,
                        tele_ok=tele.exists(), n_pairs=r["n_pairs"]))
    (HERE / "recon_out").mkdir(exist_ok=True)
    (HERE / "recon_out" / "tag_run_map.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    ok = [x for x in out if "err" not in x]
    return dict(n_ok=len(ok), n_bad=len(out) - len(ok),
                bad=[x for x in out if "err" in x],
                n_retry1=sum(1 for x in ok if x["pref"] == "retry1"),
                all_phys_hashed=all(x["phys_sha256"] for x in ok),
                all_tele_ok=all(x["tele_ok"] for x in ok))
guard("map", do_map)

mapped = [x for x in json.loads((HERE / "recon_out" / "tag_run_map.json")
                                .read_text(encoding="utf-8")) if "err" not in x]

def do_sample():
    rdir = Path(wdpaths.remap(mapped[0]["run_dir"]))
    out = dict(tag=mapped[0]["tag"], files=sorted(p.name for p in rdir.iterdir()))
    with rdir.joinpath("physics_steps.jsonl").open(encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    out["phys_keys"] = sorted(row.keys())
    veh = row.get("vehicles") or {}
    if veh:
        out["veh_keys"] = sorted(next(iter(veh.values())).keys())
    with rdir.joinpath("telemetry.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if '"report"' in line:
                tr = json.loads(line)
                out["tele_first"] = dict(kind=tr.get("kind"),
                                         frame_keys=sorted((tr.get("frame") or {}).keys()))
                break
    mj = rdir / "manifest.json"
    if mj.exists():
        mm = json.loads(mj.read_text(encoding="utf-8"))
        out["run_manifest_scalars"] = {k: mm[k] for k in sorted(mm)
                                       if isinstance(mm[k], (str, int, float, bool))}
    return out
guard("sample_run", do_sample)

def do_logged():
    res = json.loads((WM / "RESULTS.json").read_text(encoding="utf-8"))
    o = res["one_step"]
    roll = res.get("rollout") or {}
    return dict(top_keys=sorted(res.keys()), one_step_keys=sorted(o.keys()),
                n_evals=res.get("n_evals"), n_runs=res.get("n_runs"),
                start_ticks=res.get("start_ticks"),
                mean_model=o.get("mean_model_rmse_mps"),
                mean_model_raw=o.get("mean_model_rmse_raw_mps"),
                pooled_model=o.get("pooled_model_rmse_mps"),
                pooled_model_raw=o.get("pooled_model_rmse_raw_mps"),
                mean_pers=o.get("mean_persistence_rmse_mps"),
                pooled_pers=o.get("pooled_persistence_rmse_mps"),
                mean_copy=o.get("mean_field_copy_rmse_mps"),
                dv_error=res.get("dv_error"),
                verdict_pass=res.get("verdict", {}).get("pass"),
                rollout_sans_horizon={k: v for k, v in roll.items() if k != "horizon"})
guard("logged", do_logged)

def do_ckpt():
    ck = torch.load(WM / "vehicle_dynamics.pt", map_location="cpu", weights_only=False)
    ts = dict(ck.get("train_summary") or {})
    ts.pop("history", None)
    return dict(keys=sorted(ck.keys()), arch=ck.get("arch"),
                config=ck.get("config"), train_summary=ts,
                x_mean_head=[round(float(x), 4) for x in (ck.get("x_mean") or [])[:3]])
guard("ckpt", do_ckpt)

def do_sizes():
    z = np.load(next((WM / "tracks" / "holdout").glob("*.npz")), allow_pickle=True)
    return dict(tracks_mb=round(sum(p.stat().st_size
                                    for p in (WM / "tracks").rglob("*.npz")) / 1e6, 1),
                sample_npz_keys=sorted(z.files),
                sample_meta=str(z["meta"])[:220])
guard("npz", do_sizes)

report["errors"] = errs
print(json.dumps(report, indent=1, default=str))
