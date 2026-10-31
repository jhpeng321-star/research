"""CPU-only tests on synthetic EXAM v1 data. No SUMO: any process spawn fails.

Run from the repo root with the rl_driving interpreter:
  C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest tools/coopbandit_learn/test_cpu.py -q
or via unittest:
  python -m unittest tools.coopbandit_learn.test_cpu
"""
import os

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1")

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.coopbandit_learn import artifact as artifact_mod
from tools.coopbandit_learn import fixture, interpret, loader, policy
from tools.coopbandit_learn.cv import conformal_quantile, radius_from, run as cv_run, _fold_of
from tools.coopbandit_learn.models import MODELS
from tools.coopbandit_learn.schema import DEV_WORLDS, OPTIONS, POLICY_HORIZON, checked, sha

WORLDS = 12
_ROOT = None
_PATCHER = None


def setUpModule():
    global _ROOT, _PATCHER
    _PATCHER = patch("subprocess.Popen", side_effect=AssertionError("SUMO/process spawn disabled in tests"))
    _PATCHER.start()
    _ROOT = Path(tempfile.mkdtemp(prefix="coopbandit_learn_test_"))
    fixture.build(_ROOT / "exam", n_worlds=WORLDS, seed=7)


def tearDownModule():
    _PATCHER.stop()


_CACHE = {}


def _dataset():
    if "ds" not in _CACHE:
        _CACHE["ds"] = loader.load(_ROOT / "exam")
    return _CACHE["ds"]


def _cv(kind="gbt"):
    if kind not in _CACHE:
        _CACHE[kind] = cv_run(_dataset(), kind=kind)
    return _CACHE[kind]


class TestFixtureAndLoader(unittest.TestCase):
    def test_roundtrip_and_shapes(self):
        ds = _dataset()
        self.assertEqual(len(ds.states), WORLDS * 5)
        self.assertEqual(len(ds.labels), WORLDS * 5 * 5 * 3)
        self.assertEqual(sorted(ds.feature_names),
                         sorted(["time_s", "setting", "group_size", "ramp_present"] +
                                [f"l{l}_{k}" for l in range(3) for k in ("count", "cavs", "speed", "queue", "density")] +
                                [f"g{i}_{k}" for i in range(2) for k in ("s", "v", "lane", "gap", "backgap")]))
        self.assertTrue(ds.partial, "small test fixture is an exploratory partial prefix")
        sid = ds.states.state_id.iloc[0]
        self.assertIn("O0", ds.labels_for(sid, 600))
        self.assertTrue(set(ds.available(sid)) <= set(OPTIONS))

    def test_availability_rules(self):
        ds = _dataset()
        for r in ds.states.itertuples():
            self.assertTrue(r.O1 and r.O4)
            self.assertFalse(r.O3 and r.O2, "O2 (span 0/1) and O3 (both lane 1) are mutually exclusive")
            if r.O3:
                self.assertEqual(ds.features.loc[r.state_id, "ramp_present"], 1.0)

    def test_manifest_tamper_detected(self):
        work = _ROOT / "tamper"
        if not work.exists():
            fixture.build(work, n_worlds=6, seed=3)
        labels = work.joinpath("labels.csv").read_text(encoding="utf-8")
        work.joinpath("labels.csv").write_text(labels.replace(",True,", ",False,", 1), encoding="utf-8")
        with self.assertRaises(ValueError):
            loader.load(work)

    def test_dev_world_rejected(self):
        work = _ROOT / "devleak"
        if work.exists():
            work = work.with_name("devleak2")
        fixture.build(work, n_worlds=6, seed=3)
        rows = [json.loads(s) for s in work.joinpath("states.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["world_id"] = DEV_WORLDS[0]
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        work.joinpath("states.jsonl").write_text(body, encoding="utf-8")
        manifest = json.loads(work.joinpath("MANIFEST.json").read_text(encoding="utf-8"))
        manifest["states.jsonl"] = sha(work / "states.jsonl")
        work.joinpath("MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        with self.assertRaises(ValueError):
            loader.load(work)

    def test_restricted_path_guard(self):
        for name in ("holdout", "sealed", "cal_split", "chunk5", "j4b", "inc1"):
            with self.assertRaises(ValueError):
                checked(Path("E:/nowhere") / name / "labels.csv")


class TestCVAndCalibration(unittest.TestCase):
    def test_conformal_quantile_and_radius(self):
        self.assertEqual(conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]), 10.0)
        self.assertEqual(conformal_quantile([5.0]), 5.0)
        values = list(range(1, 21))
        self.assertEqual(conformal_quantile(values), 20.0)
        self.assertEqual(conformal_quantile(values, level=0.5), 11.0)
        self.assertEqual(radius_from([-5.0, -2.0, -9.0]), 0.0, "signed radius clamps at 0")
        self.assertEqual(radius_from([3.0, 7.0]), 7.0)

    def test_fold_assignment_groups(self):
        ds = _dataset()
        worlds = ds.states.world_id.astype(int).to_numpy()
        fold_of = _fold_of(np.unique(worlds))
        self.assertEqual(set(fold_of), set(int(w) for w in np.unique(worlds)))
        self.assertEqual(len(set(fold_of.values())), 5)
        for w in DEV_WORLDS:
            self.assertNotIn(w, fold_of)

    def test_oof_world_disjoint_and_complete(self):
        result = _cv("gbt")
        ds = _dataset()
        oof_worlds = set(result.oof.world)
        expected = set(int(w) for w in ds.states.world_id)
        self.assertEqual(oof_worlds, expected)
        for world, group in result.oof.groupby("world"):
            fold = result.fold_of[int(world)]
            radius = result.fold_radius[fold]
            self.assertTrue((group.fold == fold).all())
            self.assertGreaterEqual(radius, 0.0, "radius is a signed overprediction bound, clamped at 0")
        self.assertLess(result.oof.residual.abs().max(), 1e6)

    def test_models_are_sklearn_gbt_with_frozen_params(self):
        from sklearn.ensemble import GradientBoostingRegressor
        model = MODELS["gbt"]()
        self.assertIsInstance(model, GradientBoostingRegressor)
        self.assertEqual((model.n_estimators, model.max_depth, model.min_samples_leaf,
                          model.loss, model.random_state), (100, 2, 8, "huber", 0))
        result = _cv("gbt")
        self.assertEqual(set(result.models_full), set(OPTIONS))
        self.assertTrue(all(m is not None for m in result.models_full.values()))


class TestPolicyRule(unittest.TestCase):
    def test_documented_gate_no_threshold(self):
        decide = policy.decide
        self.assertEqual(decide({"O1": 5.0}, 10.0, ("O1",)), "O0", "positive but below radius -> O0")
        self.assertEqual(decide({"O1": 10.5}, 10.0, ("O1",)), "O1", "above radius -> cooperate")
        self.assertEqual(decide({"O1": -3.0, "O2": 50.0}, 10.0, ("O1", "O2")), "O2")
        self.assertEqual(decide({"O2": 50.0}, 10.0, ("O1",)), "O0", "unavailable option never chosen")
        self.assertEqual(decide({"O1": -50.0, "O4": -40.0}, 10.0, ("O1", "O4")), "O0")
        self.assertEqual(decide({}, 10.0, ()), "O0")

    def test_evaluation_math_and_bootstrap(self):
        ds = _dataset()
        result = _cv("gbt")
        ev = policy.evaluate(ds, result)
        self.assertEqual(len(ev.chosen), len(ds.states))
        self.assertTrue((ev.realized <= ev.oracle + 1e-9).all(), "realized can never beat oracle")
        self.assertTrue((ev.oracle >= 0).all())
        self.assertGreaterEqual(ev.summary["regret"]["mean"], 0.0)
        self.assertAlmostEqual(ev.baseline_summary["regret"]["mean"], ev.summary["oracle"]["mean"], places=6)
        lo, hi = ev.summary["regret"]["ci95"]
        self.assertLessEqual(lo, ev.summary["regret"]["mean"] + 1e-9)
        self.assertGreaterEqual(hi, ev.summary["regret"]["mean"] - 1e-9)
        cells = policy.per_cell(ds, ev)
        self.assertEqual(sum(c["states"] for c in cells.values()), len(ds.states))

    def test_evaluation_rejects_dev_worlds(self):
        import copy
        result = _cv("gbt")
        tampered = copy.copy(result)
        tampered.fold_of = dict(result.fold_of)
        tampered.fold_of[DEV_WORLDS[0]] = 0
        with self.assertRaises(ValueError):
            policy.evaluate(_dataset(), tampered)


class TestRuntimeAndArtifact(unittest.TestCase):
    def test_parity_and_choice(self):
        ds = _dataset()
        result = _cv("gbt")
        art = artifact_mod.build(result)
        X = ds.feature_matrix(list(ds.states.state_id))
        worst = artifact_mod.verify(art, X, result)
        self.assertLessEqual(worst, artifact_mod.PARITY_TOL)
        runtime = artifact_mod.PolicyRuntime(art)
        feats = ds.features.loc[ds.states.state_id.iloc[0]].to_dict()
        avail = ds.available(ds.states.state_id.iloc[0])
        self.assertEqual(runtime.choose(feats, avail),
                         policy.decide(runtime.predict(feats), result.radius_full, avail))

    def test_missing_feature_and_version_guards(self):
        ds = _dataset()
        runtime = artifact_mod.PolicyRuntime(artifact_mod.build(_cv("gbt")))
        feats = ds.features.loc[ds.states.state_id.iloc[0]].to_dict()
        with self.assertRaises(ValueError):
            runtime.choose({k: v for k, v in list(feats.items())[:-1]}, ("O1",))
        with self.assertRaises(ValueError):
            artifact_mod.PolicyRuntime({"version": 99})

    def test_artifact_roundtrip_text(self):
        art = artifact_mod.build(_cv("gbt"))
        text = json.dumps(art, sort_keys=True)
        again = artifact_mod.PolicyRuntime.from_text(text)
        self.assertEqual(again.feature_names, art["feature_names"])
        self.assertEqual(again.radius, art["radius"])


class TestInterpretation(unittest.TestCase):
    def test_tables_and_figures(self):
        ds = _dataset()
        result = _cv("gbt")
        imp = interpret.importance_table(result)
        self.assertTrue(all(0.0 <= r["importance"] <= 1.0 for r in imp))
        for option in OPTIONS:
            total = sum(r["importance"] for r in imp if r["option"] == option)
            self.assertTrue(total == 0.0 or abs(total - 1.0) < 1e-6,
                            "importances sum to 1 unless the model found no split at all")
        o1 = sum(r["importance"] for r in imp if r["option"] == "O1")
        self.assertGreater(o1, 0.1, "planted lane-0 density signal must be learned for O1")
        X = ds.feature_matrix(list(ds.states.state_id))
        pdp = interpret.partial_dependence_table(result, X, top_k=2, grid=3)
        self.assertGreater(len(pdp), 0)
        out = _ROOT / "figs"
        paths = interpret.plots(result, imp, pdp, out)
        interpret.write_tables(out, imp, pdp)
        for path in paths:
            self.assertGreater(path.stat().st_size, 1000)


class TestCliEndToEnd(unittest.TestCase):
    def test_cli_produces_artifacts(self):
        from tools.coopbandit_learn.cli import analyze
        out = _ROOT / "cli_out"
        report = analyze(_ROOT / "exam", out, kinds=("gbt", "linear"), plots=False)
        for name in ("report.json", "SUMMARY.md", "policy.json", "feature_importance.csv",
                     "partial_dependence.csv"):
            self.assertTrue((out / name).exists(), name)
        self.assertIn("gbt", report["kinds"])
        self.assertIn("linear", report["kinds"])
        self.assertLessEqual(report["kinds"]["gbt"]["summary"]["regret"]["mean"],
                             report["kinds"]["gbt"]["baseline"]["regret"]["mean"] + 1e-9)
        self.assertEqual(report["kinds"]["gbt"]["summary"]["oracle"]["mean"],
                         report["kinds"]["gbt"]["baseline"]["regret"]["mean"])
        text = (out / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("no behavioural threshold", text)
        self.assertIn("world-grouped", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
