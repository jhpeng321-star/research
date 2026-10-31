import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .__main__ import credibility
from .metrics import WorldIntervals, brier, crps, interval_coverage, reliability
from .rbeval_adapter import baseline_worlds, load_harness, observations, safe_path
from .report import reliability_svg, summarize_rows


def episode(seed, arm, delay, event=False):
    return dict(seed=seed, arm=arm, demand=2400, penetration=.2,
                clocks={"120": {"team_delay_vs": delay}},
                bins={"upstream": {"events": {"120": {
                    "event": event, "onset_s": 30 if event else None, "preexisting": False}}}})


def synthetic_observations():
    real = [episode(s, a, d) for s in (969400, 969401, 969420, 969432)
            for a, d in (("a", 10), ("b", 14))]
    pred = {}
    for arm, delay in (("a", 10), ("b", 14)):
        pred[969420, arm] = []
        for k in range(8):
            e = episode(969420, arm, delay+k-3.5, k < 6)
            e["future"] = k
            pred[969420, arm].append(e)
    return real, pred


class ScoreTests(unittest.TestCase):
    def test_crps_independent_cdf_integral_and_fair_form(self):
        x, y = np.array([-2., 0., 1., 4.]), 2.
        knots = sorted(set([*x, y]))
        integral = 0.
        for a, b in zip(knots[:-1], knots[1:]):
            mid = (a+b)/2
            integral += (b-a) * (np.mean(x <= mid) - (y <= mid))**2
        self.assertAlmostEqual(crps(x, y), integral)
        self.assertAlmostEqual(crps(x, y, fair=True),
            np.abs(x-y).mean() - np.abs(x[:, None]-x).sum()/(2*len(x)*(len(x)-1)))
        self.assertEqual(crps([2], 5), 3)
        self.assertIsNone(crps([2], 5, fair=True))
        for bad in ([], [[1]], [float("inf")]):
            with self.assertRaises(ValueError):
                crps(bad, 2)

    def test_brier_calibration_70_percent_and_climatology(self):
        rows = [dict(seed=i, probability=.7, outcome=int(i < 7)) for i in range(10)]
        boot = WorldIntervals({i: ("one",) for i in range(10)}, repetitions=1000)
        bins = reliability(rows, boot)
        self.assertAlmostEqual(bins[7]["mean_probability"], .7)
        self.assertAlmostEqual(bins[7]["observed_rate"], .7)
        self.assertEqual(bins[7]["worlds"], 10)
        self.assertEqual(bins[0]["count"], 0)
        self.assertIsNone(bins[0]["observed_rate"])
        self.assertAlmostEqual(np.mean([brier(.7, r["outcome"]) for r in rows]), .21)
        self.assertGreater(np.mean([brier(1, r["outcome"]) for r in rows]), .21)
        edge = reliability([dict(seed=0, probability=1., outcome=1)], boot)
        self.assertEqual(edge[-1]["count"], 1)
        with self.assertRaises(ValueError):
            brier(.5, .5)

    def test_interval_coverage_exchangeable_enumeration(self):
        outcomes = np.arange(9)
        covered = []
        for i, y in enumerate(outcomes):
            range_row = interval_coverage(np.delete(outcomes, i), y)[-1]
            covered.append(range_row["covered"])
            self.assertEqual(range_row["continuous_exchangeable_reference"], 7/9)
        self.assertAlmostEqual(np.mean(covered), 7/9)
        ties = interval_coverage([3]*8, 3)
        self.assertTrue(all(r["covered"] for r in ties))
        self.assertTrue(all(r["width"] == 0 for r in ties))

    def test_world_clustering_not_future_or_action_replication(self):
        rows = [dict(seed=1, v=0), dict(seed=2, v=2)]
        boot = WorldIntervals({1: ("x",), 2: ("x",)}, repetitions=1000)
        a, b = boot.mean(rows, "v"), boot.mean(rows*20, "v")
        for key in ("mean", "low", "high", "worlds"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["worlds"], 2)
        self.assertIsNone(boot.mean(rows[:1]*20, "v")["low"])

    def test_baselines_exclude_all_target_arms_and_reserved_labels(self):
        real, _ = synthetic_observations()
        target = next(e for e in real if e["seed"] == 969420)
        for cell in (False, True):
            self.assertEqual(set(baseline_worlds(real, target, cell=cell)), {969400, 969401})
        data = observations(synthetic_observations()[1], real, ("a", "b"), (120,), ("upstream",))
        changed = copy.deepcopy(real)
        for e in changed:
            if e["seed"] >= 969432:
                e["clocks"]["120"]["team_delay_vs"] = -1e9
                e["bins"]["upstream"]["events"]["120"]["event"] = True
        self.assertEqual(data, observations(synthetic_observations()[1], changed,
                                           ("a", "b"), (120,), ("upstream",)))

    def test_paired_score_uses_same_world_climate_and_shared_future(self):
        real, pred = synthetic_observations()
        data = observations(pred, real, ("a", "b"), (120,), ("upstream",))
        row = next(r for r in data["distributions"] if r["target"] == "paired_delay")
        self.assertEqual(row["predictive_sd"], 0)
        self.assertEqual(row["crps"], 0)
        self.assertEqual(row["pooled_crps"], 0)
        event = next(r for r in data["events"] if r["target"] == "preference")
        self.assertEqual(event["probability"], 1)
        self.assertEqual(event["pooled_probability"], 1)
        self.assertEqual(event["cell_brier"], 0)
        self.assertEqual(event["cell_n"], 2)
        summaries, diagrams = summarize_rows(data, bootstrap=100)
        self.assertTrue(summaries)
        self.assertIn("<svg", reliability_svg(diagrams, "preference", "synthetic & test"))
        self.assertIn("synthetic &amp; test", reliability_svg(diagrams, "preference", "synthetic & test"))

    def test_unknown_events_never_become_false_or_reduced_K(self):
        real, pred = synthetic_observations()
        pred[969420, "a"][0]["bins"]["upstream"]["events"]["120"]["event"] = None
        data = observations(pred, real, ("a", "b"), (120,), ("upstream",))
        row = next(r for r in data["events"] if r["action"] == "a" and r["target"] == "breakdown")
        self.assertIsNone(row["probability"])
        self.assertIsNone(row["brier"])
        self.assertEqual(row["unscorable_futures"], 1)
        pred[969420, "a"].pop()
        with self.assertRaises(ValueError):
            observations(pred, real, ("a", "b"), (120,), ("upstream",))

    def test_prohibited_paths_rejected_before_access(self):
        for word in ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
                     "hetdata_truth", "vehicle_params", "driver_params", "vType_ASSIGNMENT"):
            with self.assertRaises(ValueError):
                safe_path(Path("never-open") / word / "x.json")


class HarnessIntegrationTests(unittest.TestCase):
    def test_registered_validator_to_credibility_without_simulation(self):
        # Preserved scratch; synthetic copied outcomes are ONLY a plumbing fixture.
        default = Path(__file__).resolve().parents[2].parent / "c4-rb-eval"
        root = Path(os.environ.get("WMS_RBEVAL_ROOT", str(default)))
        if not (root / "tools/rbeval/import_external.py").is_file():
            self.skipTest("Set WMS_RBEVAL_ROOT to the trusted local evaluator for integration")
        core, _ = load_harness(root)
        parent = Path("E:/Temp/opencode") if os.name == "nt" else Path("/tmp")
        self.assertTrue(parent.is_dir())
        scratch = Path(tempfile.mkdtemp(prefix="wmdecide-synthetic-", dir=parent))
        real = core.load(core.HERE / "real_observables.json")["episodes"]
        episodes = []
        for arm in core.ARMS:
            r = next(e for e in real if e["seed"] == 969420 and e["arm"] == arm)
            episodes.append(dict(arm=arm, initial_sha256=r["initial_sha256"],
                bins={reg: dict(speed=r["bins"][reg]["speed"][6:],
                                post_counts=r["bins"][reg]["post_counts"]) for reg in core.REGIONS},
                clocks={h: {key: c[key] for key in
                    ("team_delay_vs", "road_delay_vs", "source_wait_vs")}
                    for h, c in r["clocks"].items()}))
        manifest = dict(model="synthetic-wmdecide-plumbing", model_sha256=core.digest(core.HERE/"core.py"),
            config_sha256=core.digest(core.HERE/"references.json"),
            sources={str(core.HERE/"core.py"): core.digest(core.HERE/"core.py")},
            tasks=[dict(seed=969420, future=k, episodes=copy.deepcopy(episodes)) for k in range(8)])
        core.save_new(scratch/"manifest.json", manifest)
        result = subprocess.run([sys.executable, "-B", str(core.HERE/"import_external.py"),
            "--manifest", str(scratch/"manifest.json"), "--out", str(scratch/"normalized")],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        report = credibility(root, [scratch/"normalized"], scratch/"report", allow_partial=True)
        self.assertEqual(report["audit"]["complete_worlds"], [969420])
        self.assertEqual(report["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(report["qualification"], "NOT_ASSESSED_USE_RBEVAL_GATE")
        self.assertTrue((scratch/"report/reliability_preference.svg").is_file())
        self.assertTrue((scratch/"report/CREDIBILITY.md").is_file())
        with self.assertRaises(ValueError):
            credibility(root, [scratch/"normalized"], scratch/"report", allow_partial=True)
        with self.assertRaises(AssertionError):
            credibility(root, [scratch/"normalized"]*2, scratch/"duplicates", allow_partial=True)
        with self.assertRaises(AssertionError):
            credibility(root, [scratch/"normalized"], scratch/"incomplete_full_panel")


if __name__ == "__main__":
    unittest.main()
