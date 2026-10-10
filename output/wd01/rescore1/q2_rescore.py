"""WD-01-PB-RESCORE1 Q2: proxy-skill re-scoring of the v0.3.2 models.

The deployable v0.3.2 models (strict = HEADLINE, detector = SENSITIVITY)
predict V(s, tau, policy) from the damping base arm's RSU-observable
state at t0.  Their OLD-interpretation contrast is

  dV_old(tau) = V(s_t0, tau, damping_fleet) - V(s_t0, tau, switching_mix)

(same state, policy-conditioned -- exactly the v0.3.x contrast code's
construction).  Q2 scores dV_old against the REALISED TRUE contrast

  dY(tau) = Y_damping(t0, tau) - Y_true_native(t0 + 10 s, tau)

from the RESCORE1 label root.  Existing models only: NO training, no
refit, no model selection; CAL is read report-only (disclosed as the
second CAL read for v0.3.2) and TRAIN results are labelled
in-sample-trained.

Per variant x split x tau: n, sign accuracy of E[dV_old] vs dY with a
world-bootstrap 90% CI, CRPS of dY under the paired predictive
difference distribution (M x quantile independent pairing: every
damping-quantile level paired with every switching-quantile level, 49
equally-likely differences -- the quantile-grid analogue of the v0.3.x
independent-member pairing), bias, and the zero-contrast and
TRAIN-cell-mean (demand x penetration) baselines.

Raw rows are written BEFORE aggregation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE.parent / "partB_v0data",
           HERE.parent / "partA_data", HERE.parent / "part4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common as C  # noqa: E402
import wdpaths  # noqa: E402  (E:-native roots; no D:)

from partB_v0.data import FEAT_SUFFIX, V0Dataset, V0Manifest  # noqa: E402
from partB_v0.data import sha256_of  # noqa: E402
from partB_v0.gbm import GbmValueModel  # noqa: E402
from partB_v0.model import policy_id  # noqa: E402

VARIANTS = {
    "strict": dict(label="strict (HEADLINE)",
                   feature_set="deployable_strict",
                   model_dir=C.RD / "output" / "partB_results"
                            / "v0.3.2_strict"),
    "detector": dict(label="detector (SENSITIVITY)",
                     feature_set="deployable_detector",
                     model_dir=C.RD / "output" / "partB_results"
                              / "v0.3.2_detector"),
}
CHUNKS = ("chunk2", "chunk3")


def restricted_manifest(variant: dict) -> V0Manifest:
    """v0.3.2 MANIFEST restricted to the chunk-2/3 damping BASE sources of
    the label root's TRAIN/CAL worlds (the only states Q2 predicts from);
    every file Q2 will read is sha256-verified against the frozen
    MANIFEST."""
    man_raw = json.loads(
        (C.V032_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    label_man = C.load_label_manifest()
    want_worlds = {(e["chunk_id"], e["world"], e["split"])
                   for e in label_man["worlds"].values()}
    entries = {}
    for key, e in sorted(man_raw["worlds"].items()):
        if e["chunk_id"] not in CHUNKS or e["source"] != "base":
            continue
        k3 = (e["chunk_id"], e["world"], e["split"])
        assert k3 in want_worlds, \
            f"v0.3.2 world not in the TRAIN/CAL label root: {key}"
        entries[key] = e
        for rel, sha in e["files"].items():
            suffix = rel.rsplit("_", 1)[-1]
            need = (rel.endswith("_labels.parquet")
                    or rel.endswith("_td.parquet")
                    or rel.endswith(FEAT_SUFFIX[variant["feature_set"]]))
            if not need:
                continue
            p = C.V032_ROOT / rel
            assert p.is_file(), f"missing v0.3.2 file: {rel}"
            got = sha256_of(p)
            assert got == sha, f"v0.3.2 hash mismatch: {rel}"
    man = V0Manifest(root=C.V032_ROOT,
                     sha256=sha256_of(C.V032_ROOT / "MANIFEST.json"),
                     raw=man_raw, entries=entries)
    man.files_verified = True
    print(f"q2[{variant['feature_set']}]: {len(entries)} damping base "
          f"sources selected, files verified", flush=True)
    return man


def predict_world(model, ds, wname: str, t0: float) -> dict:
    """{tau: (dV samples[49], dV mean)} for one world's t0 state."""
    r = ds.rows
    m = ((r["world"] == wname) & np.isclose(r["t"], t0)
         & (r["source"] == "base")).to_numpy()
    sel = r[m]
    if sel.empty:
        raise RuntimeError(f"no t0 base rows for {wname}")
    pre = float(sel["pre_fix"].astype(float).iloc[0])
    taus, g_idxs = [], []
    for tau in C.TAUS:
        row = sel[np.isclose(sel["tau"], tau)]
        if row.empty:
            continue
        taus.append(float(tau))
        g_idxs.append(int(row["t_idx"].iloc[0]))
    taus_arr = np.asarray(taus, np.float64)
    pid_d = policy_id("damping_fleet")
    pid_s = policy_id("switching_mix")
    g_arr = np.asarray(g_idxs, np.int64)
    X_d = model._matrix_global(ds, g_arr, np.full(len(g_arr), pid_d),
                               np.full(len(g_arr), pre))
    X_s = model._matrix_global(ds, g_arr, np.full(len(g_arr), pid_s),
                               np.full(len(g_arr), pre))
    A = model.predict_quantiles_matrix(X_d, taus_arr, "delay")
    B = model.predict_quantiles_matrix(X_s, taus_arr, "delay")
    qs = sorted(A)
    out = {}
    for i, tau in enumerate(taus):
        a = np.array([A[q][i] for q in qs], np.float64)
        b = np.array([B[q][i] for q in qs], np.float64)
        diffs = (a[:, None] - b[None, :]).ravel()
        i05 = qs.index(0.5)
        out[tau] = dict(samples=diffs, mean=float(a.mean() - b.mean()),
                        median=float(a[i05] - b[i05]),
                        q05=float(np.quantile(diffs, 0.05)),
                        q95=float(np.quantile(diffs, 0.95)))
    return out


def collect(variant: dict, cache: dict) -> pd.DataFrame:
    man = restricted_manifest(variant)
    ds = V0Dataset(man, feature_set=variant["feature_set"])
    model = GbmValueModel.load(variant["model_dir"], log=print)
    label_man = C.load_label_manifest()
    rows = []
    for chunk_id, world, split, dem, pen, t0 in C.world_rows(label_man):
        pw = predict_world(model, ds, world, t0)
        f_damp = C.label_frame(C.LABEL_ROOT, chunk_id, world, "base",
                               cache)
        f_tn = C.label_frame(C.LABEL_ROOT, chunk_id, world,
                             "truearms_native", cache)
        for tau, pred in pw.items():
            y_d = C.y_at(f_damp, t0, tau)
            y_n = C.y_at(f_tn, t0 + C.T_OFFSET_S, tau)
            if y_d is None or y_n is None:
                continue
            dy = y_d - y_n
            crps = float(_crps_empirical(pred["samples"],
                                         np.asarray([dy]))[0])
            rows.append(dict(
                chunk=chunk_id, world=world, split=split,
                demand_veh_h=dem, penetration=pen, tau=tau,
                dV_mean=pred["mean"], dV_median=pred["median"],
                dV_q05=pred["q05"], dV_q95=pred["q95"], dY=dy, sign_ok=bool(
                    np.sign(pred["mean"]) == np.sign(dy)),
                crps_model=crps, crps_zero=abs(dy)))
    del ds
    return pd.DataFrame(rows)


def _crps_empirical(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    from partB_v0.distributions import crps_empirical
    return crps_empirical(samples, y)


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """TRAIN-cell-mean true contrast per (tau, demand x penetration)."""
    tr = df[df["split"] == "TRAIN"]
    cell = tr.groupby(["tau", "demand_veh_h", "penetration"])["dY"] \
             .mean().rename("cell_mean").reset_index()
    df = df.merge(cell, on=["tau", "demand_veh_h", "penetration"],
                  how="left")
    df["crps_cellmean"] = (df["dY"] - df["cell_mean"]).abs()
    df["sign_ok_cellmean"] = np.sign(df["cell_mean"]) == np.sign(df["dY"])
    df["bias"] = df["dV_mean"] - df["dY"]
    df["cell_baseline_level"] = np.where(df["cell_mean"].notna(),
                                         "demand_x_penetration", "thin")
    return df


def tables(df: pd.DataFrame, variant: dict) -> dict:
    out = dict(variant=variant["label"],
               feature_set=variant["feature_set"],
               model_dir=str(variant["model_dir"]),
               per_split_tau=[])
    for split in ("CALIBRATION", "TRAIN"):
        gs = df[df["split"] == split]
        for tau in C.TAUS:
            g = gs[np.isclose(gs["tau"], tau)]
            if g.empty:
                continue
            ok = g["sign_ok"].to_numpy().astype(float)
            lo, hi = C.world_bootstrap_ci(g["world"].tolist(), ok)
            out["per_split_tau"].append(dict(
                split=split + (" (in-sample-trained)" if split == "TRAIN"
                               else " (headline, report-only 2nd read)"),
                tau=tau, n=int(len(g)),
                sign_accuracy=C.r2(float(ok.mean()), 3),
                ci90_lo=C.r2(lo, 3), ci90_hi=C.r2(hi, 3),
                crps_model=C.r2(g["crps_model"].mean()),
                crps_zero_baseline=C.r2(g["crps_zero"].mean()),
                crps_cellmean_baseline=C.r2(g["crps_cellmean"].mean()),
                sign_acc_cellmean_baseline=C.r2(
                    float(g["sign_ok_cellmean"].astype(float).mean()), 3),
                bias=C.r2(g["bias"].mean()),
                mae=C.r2(g["bias"].abs().mean())))
    return out


def markdown(t: dict) -> str:
    L = [f"## Q2 -- {t['variant']}", "",
         "dV_old = V(s_t0, tau, damping) - V(s_t0, tau, switching_mix) "
         "from the frozen v0.3.2 deployable model (same state, "
         "policy-conditioned), scored against the realised TRUE contrast "
         "dY = Y_damping(t0, tau) - Y_true_native(t0+10s, tau).  CRPS of "
         "dY under the 7x7 independent-pairing predictive difference; "
         "positive dV/dY = damping has more predicted/realised delay "
         "than the contrast arm.", "",
         "| split | tau | n | sign acc | 90% CI | CRPS model | CRPS "
         "zero | CRPS cell-mean | bias | MAE |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for row in t["per_split_tau"]:
        L.append(
            f"| {row['split']} | {row['tau']:.0f} | {row['n']} "
            f"| {row['sign_accuracy']:.3f} "
            f"| [{row['ci90_lo']:.3f}, {row['ci90_hi']:.3f}] "
            f"| {row['crps_model']:.1f} | {row['crps_zero_baseline']:.1f} "
            f"| {row['crps_cellmean_baseline']:.1f} "
            f"| {row['bias']:.1f} | {row['mae']:.1f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="both",
                    choices=("both", "strict", "detector"))
    args = ap.parse_args(argv)
    names = list(VARIANTS) if args.variants == "both" else [args.variants]
    all_tables = []
    for name in names:
        variant = VARIANTS[name]
        cache: dict = {}
        df = collect(variant, cache)
        df = add_baselines(df)
        raw = C.RESULTS / f"q2_raw_rows_{name}.csv"
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)          # raw rows BEFORE aggregation
        t = tables(df, variant)
        t["raw_rows"] = str(raw)
        all_tables.append(t)
        C.write_json(C.RESULTS / f"q2_tables_{name}.json", t)
        (C.RESULTS / f"q2_tables_{name}.md").write_text(
            markdown(t), encoding="utf-8")
        print(f"Q2[{name}]: {len(df)} (world, tau) rows; tables + raw "
              f"rows written")
        for row in t["per_split_tau"]:
            if row["split"].startswith("CAL"):
                print(f"  CAL tau={row['tau']:.0f}: sign acc "
                      f"{row['sign_accuracy']:.3f} "
                      f"[{row['ci90_lo']:.3f}, {row['ci90_hi']:.3f}], "
                      f"CRPS model {row['crps_model']} vs zero "
                      f"{row['crps_zero_baseline']} vs cell-mean "
                      f"{row['crps_cellmean_baseline']}, bias "
                      f"{row['bias']}")
    C.write_json(C.RESULTS / "q2_tables.json", all_tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
