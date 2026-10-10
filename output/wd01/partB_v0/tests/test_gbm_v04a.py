"""WD-01-PB-V04A: v0.4a dataset loading, GBM design (policy one-hot,
schedule summary, per-action CAV counts, no pre_fix), auxiliary handling,
counterfactual fleet design and the final-evaluation plumbing, on a tiny
synthetic v0.4a root.  CPU only, tiny fits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from partB_v0 import acceptance_v04a as AC
from partB_v0 import final_eval_v04a as FE
from partB_v0.data import SCHEMA, V0Dataset, load_manifest
from partB_v0.gbm import (V04A_ACTION_COUNT_NAMES, V04A_PI_NAMES,
                          V04A_POLICY_NAMES, GbmValueModel,
                          counterfactual_design, row_feature_matrix,
                          summary_features)
from partB_v0.model import POLICIES_V04A, policy_id

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)
T0 = 240.0
DATA_END = T0 + 600.0

# source -> (policy class, role, n tokens per action [native, damping, pi])
SRC = {
    "base": ("damping_fleet", "primary", (0, 2, 0)),
    "native": ("native_fleet", "primary", (2, 0, 0)),
    "pi": ("pi_fleet", "primary", (0, 0, 2)),
    "fixed1": ("fixed_mix", "primary", (1, 1, 0)),
    "switching1": ("switching_mix", "primary", (0, 1, 1)),
    "swnative": ("switching_mix", "aux", (1, 0, 1)),
}
POL_EFFECT = {"damping_fleet": 0.0, "native_fleet": 300.0, "pi_fleet": 280.0,
              "fixed_mix": 150.0, "switching_mix": 100.0}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_v04a_root(root: Path, n_train=6, n_cal=3, seed=3) -> str:
    rng = np.random.default_rng(seed)
    worlds = {}
    wdir = root / "worlds" / "chunk4"
    wdir.mkdir(parents=True, exist_ok=True)
    for i in range(n_train + n_cal):
        split = "TRAIN" if i < n_train else "CALIBRATION"
        wname = f"w{i:04d}m2700p02t{int(T0)}"
        load = rng.normal(0, 30)
        for source, (pol, role, cnt) in SRC.items():
            is_base = source in ("base",)
            lo = T0 if is_base or role == "primary" else T0 + 10
            ts = np.arange(lo, T0 + 570.0 + 1e-9, 10.0)[:10]
            n = len(ts)
            tau_avail = DATA_END - ts
            tok = np.zeros((n, 64, 20), np.float32)
            msk = np.zeros((n, 64), np.float32)
            k = 0
            for slot, c in enumerate(cnt):
                for _ in range(c):
                    tok[:, k, 5 + slot] = 1.0
                    msk[:, k] = 1.0
                    k += 1
            state = rng.normal(0, 1, (n,))
            lane_bins = np.zeros((n, 3, 320, 3), np.float32)
            lane_bins[:, :, :10, 0] = (5 + 2 * state)[:, None, None]
            lane_bins[:, :, :10, 1] = 20.0
            feats = dict(
                t=ts.astype(np.float64), tau_available=tau_avail,
                lane_bins=lane_bins,
                lane_mask=np.ones((n, 3, 320), np.float32),
                lane_geom=np.ones((3, 320), np.float32),
                cav_tokens=tok, cav_mask=msk,
                wall_tokens=np.zeros((n, 4, 11), np.float32),
                wall_mask=np.zeros((n, 4), np.float32),
                demand=rng.random((n, 6)).astype(np.float32),
                context=(rng.random((n, 8)) * 0.1).astype(np.float32),
                world=np.str_(wname), source=np.str_(source),
                chunk_id=np.str_("chunk4"))
            files = {}
            for suf, dem in (("_feat_dep_strict.npz", 6),
                             ("_feat_dep_det.npz", 12)):
                f = dict(feats)
                f["demand"] = rng.random((n, dem)).astype(np.float32)
                p = wdir / f"{wname}__{source}{suf}"
                np.savez_compressed(p, **f)
                files[f"worlds/chunk4/{p.name}"] = _sha(p)
            shares = np.array(cnt, float)
            shares = shares / shares.sum()
            rows = []
            for ti, t in enumerate(ts):
                for tau in TAUS:
                    if tau > tau_avail[ti] + 1e-9:
                        continue
                    y = (20.0 * tau + 40.0 * state[ti] * tau / 60
                         + POL_EFFECT[pol] * tau / 300 + load
                         + rng.normal(0, 5))
                    rows.append(dict(
                        world=wname, source=source, chunk_id="chunk4",
                        split=split, demand_veh_h=2700, penetration=0.02,
                        p40=False, pre_fix=False, t0_s=T0, policy=pol,
                        t=float(t), tau=float(tau),
                        tau_available=float(tau_avail[ti]),
                        cav_speed_mode=27, total_delay_ref_s=y,
                        ttc_lt3_veh_s=abs(rng.normal(5, 1)),
                        hard_brake_s=abs(rng.normal(3, 1)),
                        collision_any=0,
                        pi_native=shares[0], pi_damping=shares[1],
                        pi_pi=shares[2], role=role, aux=role == "aux",
                        arm_kind="x"))
            lab = pd.DataFrame(rows)
            f_lab = wdir / f"{wname}__{source}_labels.parquet"
            lab.to_parquet(f_lab, index=False)
            files[f"worlds/chunk4/{f_lab.name}"] = _sha(f_lab)
            f_td = wdir / f"{wname}__{source}_td.parquet"
            pd.DataFrame(columns=["world", "source", "chunk_id", "split",
                                  "t", "t_next", "tau", "d30_s",
                                  "label_mc_complete"]).to_parquet(
                f_td, index=False)
            files[f"worlds/chunk4/{f_td.name}"] = _sha(f_td)
            worlds[f"chunk4/{wname}__{source}"] = dict(
                world=wname, source=source, chunk_id="chunk4", split=split,
                role=role, aux=role == "aux", demand_veh_h=2700,
                penetration=0.02, pre_fix=False, t0_s=T0, policy=pol,
                policy_v04a=pol, cav_speed_mode=27, files=files)
    man = dict(schema=SCHEMA, zero_simulation=True, tau_s=list(TAUS),
               commit_window_s=30.0, chunks={"chunk4": {}},
               v04a=dict(policy_classes=list(POLICIES_V04A)), worlds=worlds)
    (root / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    return _sha(root / "MANIFEST.json")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("v04a_root")
    make_v04a_root(r)
    return r


@pytest.fixture(scope="module")
def ds(root):
    man = load_manifest(root, log=lambda *_: None)
    return V0Dataset(man, feature_set="deployable_strict",
                     log=lambda *_: None)


def test_policy_ids_are_backward_compatible_and_extended():
    assert policy_id("switching_mix") == 0 and policy_id("damping_fleet") == 1
    assert [policy_id(p) for p in POLICIES_V04A] == list(range(5))


def test_v04a_manifest_flags_and_exclude_aux(root):
    man = load_manifest(root, log=lambda *_: None)
    assert man.v04a and man.n_aux_skipped == 0
    man2 = load_manifest(root, exclude_aux=True, log=lambda *_: None)
    assert man2.n_aux_skipped == 9 and len(man2) == len(man) - 9
    assert not any(e["aux"] for e in man2.entries.values())


def test_dataset_rows_carry_policy_pi_and_aux(ds):
    r = ds.rows
    assert set(r["policy"]) == set(POLICIES_V04A)
    assert r["aux"].dtype == bool and r["aux"].any() and (~r["aux"]).any()
    assert (r.loc[r["source"] == "swnative", "aux"]).all()
    for c in V04A_PI_NAMES:
        assert c in r.columns


def test_design_has_policy_pi_counts_and_no_pre_fix(ds):
    X, names = row_feature_matrix(ds)
    assert "pre_fix" not in names and "policy_id" not in names
    for n in V04A_ACTION_COUNT_NAMES + V04A_POLICY_NAMES + V04A_PI_NAMES:
        assert n in names
    assert X.shape[1] == len(names)
    S, sn = summary_features(ds)
    assert sn[-4:] == list(V04A_ACTION_COUNT_NAMES)
    # per-action CAV counts follow the tokens of the source
    r = ds.rows
    i = int(np.flatnonzero(r["source"].to_numpy() == "fixed1")[0])
    cnt = X[i, len(sn) - 4:len(sn)]
    assert cnt.tolist() == [1, 1, 0, 0]
    onehot = X[i, len(sn):len(sn) + 5]
    assert onehot.tolist() == [0, 0, 0, 0, 1]        # fixed_mix
    assert X[i, len(sn) + 5:].tolist() == [0.5, 0.5, 0.0]


def test_counterfactual_design_puts_every_cav_on_the_fleet_action(ds):
    S, sn = summary_features(ds)
    rows = S[:5]
    for pol, k in (("native_fleet", 0), ("damping_fleet", 1),
                   ("pi_fleet", 2)):
        X = counterfactual_design(rows, pol)
        cnt = X[:, len(sn) - 4:len(sn)]
        n = rows[:, -4:].sum(axis=1)
        assert np.allclose(cnt[:, k], n) and cnt.sum() == pytest.approx(
            n.sum())
        assert (X[:, len(sn) + POLICIES_V04A.index(pol)] == 1).all()
        assert (X[:, len(sn) + 5 + k] == 1).all()
    with pytest.raises(ValueError):
        counterfactual_design(rows, "fixed_mix")


def _fit(ds, out):
    m = GbmValueModel(variant="auto", max_iter=20, cal_folds=2,
                      feature_set="deployable_strict")
    m.fit(ds, log=lambda *_: None)
    m.save(out)
    return m


@pytest.fixture(scope="module")
def model(ds, tmp_path_factory):
    out = tmp_path_factory.mktemp("v04a_model")
    m = _fit(ds, out)
    return m, out


def test_fit_marks_design_and_excludes_aux_from_selection(ds, model):
    m, out = model
    assert m.design == "v04a" and m.pre_fix_at_predict is None
    assert m.aux_excluded_from_calibration
    r = ds.rows
    for sel in m.selection:
        tau = sel["tau"]
        tr = (r["tau"] == tau) & (r["split"] == "TRAIN")
        assert sel["n"] == int((tr & ~r["aux"]).sum())
        assert sel["n"] < int(tr.sum())          # aux rows trained only
    # OOF arrays cover ALL TRAIN rows (aux included in the folds)
    for tau, oof in m.oof_delay.items():
        n = int(((r["tau"] == tau) & (r["split"] == "TRAIN")).sum())
        assert all(len(v) == n for v in oof.values())
    side = json.loads((out / "gbm_value.json").read_text())
    assert side["design"] == "v04a"
    assert side["aux_excluded_from_calibration"] is True
    assert "pre_fix" not in side["feature_names"]
    m2 = GbmValueModel.load(out, log=lambda *_: None)
    assert m2.design == "v04a"


def test_policy_effect_is_learned_and_queries_are_monotone(ds, model):
    m, _ = model
    r = ds.rows
    cal = np.flatnonzero((r["split"] == "CALIBRATION").to_numpy()
                         & ~r["aux"].to_numpy())
    pr = m.predict_quantiles_rows(ds, cal, "delay")
    qs = sorted(pr)
    P = np.column_stack([pr[q] for q in qs])
    assert (np.diff(P, axis=1) >= -1e-9).all()
    # native arm systematically higher delay than damping in the data
    med = pr[0.5]
    tau = r["tau"].to_numpy()[cal]
    src = r["source"].to_numpy()[cal]
    m60 = np.isclose(tau, 300.0)
    assert med[m60 & (src == "native")].mean() > \
        med[m60 & (src == "base")].mean()


def test_oof_quantiles_align_with_the_fit_and_shifts(ds, model):
    m, _ = model
    r = ds.rows
    trn = np.flatnonzero((r["split"] == "TRAIN").to_numpy()
                         & ~r["aux"].to_numpy())
    oq = FE.oof_quantiles(m, ds, trn)
    P = np.column_stack([oq[q] for q in sorted(oq)])
    assert np.isfinite(P).all() and (np.diff(P, axis=1) >= -1e-9).all()
    with pytest.raises(RuntimeError, match="non-TRAIN"):
        cal = np.flatnonzero((r["split"] == "CALIBRATION").to_numpy())
        FE.oof_quantiles(m, ds, cal[:3])


def test_pairs_contrasts_and_blocks_run_end_to_end(ds, model):
    m, _ = model
    pairs_tr = AC.build_pairs(ds.rows, ("TRAIN",)).reset_index(drop=True)
    assert set(pairs_tr["contrast"]) == set(AC.REPORT_CONTRASTS)
    assert not pairs_tr["arm_source"].eq("swnative").any()
    refs = AC.train_references(pairs_tr)
    blk = FE.contrast_block(ds, m, "CALIBRATION", refs, pairs_tr,
                            n_boot=50)
    rec = blk["records"]
    assert {r["contrast"] for r in rec} == set(AC.REPORT_CONTRASTS)
    assert all(r["n_pairs"] == 3 for r in rec if r["contrast"] ==
               "damping-native")
    oob = FE.contrast_block(ds, m, "TRAIN", refs, pairs_tr, oof=True,
                            n_boot=50)
    assert len(oob["records"]) == len(rec)
    a3 = AC.a3_verdict(rec)
    assert set(a3) == set(AC.GATING_CONTRASTS)


def test_a1_block_reports_per_policy_and_strata(ds, model):
    m, _ = model
    r = ds.rows
    cal = np.flatnonzero((r["split"] == "CALIBRATION").to_numpy()
                         & ~r["aux"].to_numpy())
    a1 = FE.a1_block(ds, m, cal, oof=False)
    assert set(a1["by_policy"]) == {"damping_fleet", "native_fleet",
                                    "pi_fleet", "fixed_mix",
                                    "switching_mix"}
    assert len(a1["per_tau"]) == 5 and "2700" in a1["by_demand"]
    assert 0.0 <= a1["coverage_overall"] <= 1.0


def test_cli_train_only_never_opens_calibration_files(root, tmp_path):
    """`train --train-only --no-eval` fits from TRAIN alone: no CAL file
    of the root is opened (the single CAL read is the final evaluation)."""
    import sys
    from partB_v0.__main__ import main
    cal_worlds = [f"w{i:04d}m2700p02t240" for i in range(6, 9)]
    opened = []

    def hook(event, args):
        if event == "open":
            opened.append(str(args[0]))
    sys.addaudithook(hook)
    out = tmp_path / "out"
    rc = main(["train", "--value-model", "gbm", "--gbm-variant", "auto",
               "--features", "deployable_strict", "--data", str(root),
               "--out", str(out), "--no-eval", "--train-only",
               "--gbm-max-iter", "10", "--gbm-cal-folds", "2"])
    assert rc == 0
    hits = [p for p in opened if any(w in p.replace("\\", "/")
                                     for w in cal_worlds)]
    assert not hits, hits[:3]
    side = json.loads((out / "gbm_value.json").read_text())
    assert side["design"] == "v04a"
    m = GbmValueModel.load(out, log=lambda *_: None)
    assert m.oof_delay and m.feature_set == "deployable_strict"
    # --exclude-aux ablation via the same CLI
    out2 = tmp_path / "out2"
    rc = main(["train", "--value-model", "gbm", "--gbm-variant", "auto",
               "--features", "deployable_strict", "--data", str(root),
               "--out", str(out2), "--no-eval", "--train-only",
               "--exclude-aux", "--gbm-max-iter", "10",
               "--gbm-cal-folds", "2"])
    assert rc == 0
    m2 = GbmValueModel.load(out2, log=lambda *_: None)
    assert all(sel["n"] > 0 for sel in m2.selection)


def test_final_eval_end_to_end_single_read_and_report(root, tmp_path):
    """Fit the three variants on the mini root, run the final evaluation
    once (lock file, eval_report, metrics, verify, REPORT), and check a
    second run is refused."""
    from partB_v0.__main__ import main
    from partB_v0.outputs import verify_outputs
    res = tmp_path / "results"
    for name, feats, extra in (
            ("v0.4a_strict", "deployable_strict", []),
            ("v0.4a_detector", "deployable_detector", []),
            ("v0.4a_strict_noaux", "deployable_strict", ["--exclude-aux"])):
        rc = main(["train", "--value-model", "gbm", "--gbm-variant", "auto",
                   "--features", feats, "--data", str(root),
                   "--out", str(res / name), "--no-eval", "--train-only",
                   "--gbm-max-iter", "10", "--gbm-cal-folds", "2"] + extra)
        assert rc == 0
    # plumbing DRY RUN first: TRAIN rows in the CAL role, no CAL file
    # opened, no lock, nothing written into the model dirs
    import sys
    cal_worlds = [f"w{i:04d}m2700p02t240" for i in range(6, 9)]
    opened = []

    def hook(event, args):
        if event == "open":
            opened.append(str(args[0]))
    sys.addaudithook(hook)
    before = {p.name: p.stat().st_mtime_ns
              for p in (res / "v0.4a_strict").iterdir() if p.is_file()}
    assert FE.main(["--data", str(root), "--results-root", str(res),
                    "--dry-run", "--dry-out", str(tmp_path / "dry"),
                    "--n-boot", "20"]) == 0
    assert not (res / "CAL_READ_ONCE_v04a.json").exists()
    assert (tmp_path / "dry" / "DRY_RUN_REPORT.md").is_file()
    assert not [p for p in opened if any(w in p.replace("\\", "/")
                                         for w in cal_worlds)]
    after = {p.name: p.stat().st_mtime_ns
             for p in (res / "v0.4a_strict").iterdir() if p.is_file()}
    assert before == after
    rep = tmp_path / "REPORT_V04A.md"
    rc = FE.main(["--data", str(root), "--results-root", str(res),
                  "--report", str(rep), "--n-boot", "40"])
    assert rc == 0
    lock = json.loads((res / "CAL_READ_ONCE_v04a.json").read_text())
    assert set(lock["verdicts"]) == {"strict", "detector", "strict_noaux"}
    assert set(lock["verdicts"].values()) <= {"ACCEPT", "PARTIAL", "FAIL"}
    txt = rep.read_text(encoding="utf-8")
    for needle in ("Verdict table", "A1", "A2", "A3", "damping-native",
                   "pi-native", "TRAIN-OOB", "Ideal-sensing premise",
                   "Per-policy coverage"):
        assert needle in txt, needle
    for name in ("v0.4a_strict", "v0.4a_detector", "v0.4a_strict_noaux"):
        d = res / name
        assert (d / "eval_report.md").is_file()
        met = json.loads((d / "metrics.json").read_text())
        assert met["eval_v04a"]["verdict"]["verdict"] in (
            "ACCEPT", "PARTIAL", "FAIL")
        ok, problems = verify_outputs(d)
        assert ok, problems
    # ONE CAL read: a second invocation is refused, nothing recomputed
    assert FE.main(["--data", str(root), "--results-root", str(res)]) == 2
