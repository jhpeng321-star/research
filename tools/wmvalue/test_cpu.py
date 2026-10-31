import copy
import unittest

import numpy as np

from .audit import allowed_seed, safe
from .core import (ARMS, TIMES, Value, action_summary, arm_means,
                   assemble_predictions, evaluate, features, metrics, paired_rows,
                   remaining_labels)
from .report import require_fit_ready, summarize


def state(t=300):
    return dict(time_s=t, kin=np.array([[-100., 0., 12., .3], [30., 2., 8., -.2],
                                       [100., 1., 29., 1.]]),
                present=np.array([1, 1, 1]), sensor_mask=np.array([1, 1, 0]),
                known_cav=np.array([1, 0, 0]), coverage=np.ones((3, 5)),
                flow_counts=np.array([[20, 10], [9, 4]]))


def test_cells(mode="perfect"):
    rows = []
    for seed in range(969432, 969440):
        for t in TIMES:
            for j, arm in enumerate(ARMS):
                real = (j-2)*(seed-969430)*(900-t)
                bad = -real
                preds = dict(A=bad, B=real, C=real, D=bad, E=bad)
                if mode == "direct_perfect":
                    preds["D"] = real
                rows.append(dict(seed=seed, t0=t, arm=arm, real=real, **preds))
    return rows


class ObservationTests(unittest.TestCase):
    def test_forbidden_paths_and_roles(self):
        for name in ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
                     "hetdata_truth", "vehicle_params", "driver_params", "vType"):
            with self.assertRaises(ValueError):
                safe("data/"+name+"/record.npy")
        for seed in (964000, 962003, 977002, 969432, 969439):
            with self.assertRaises(ValueError):
                allowed_seed(seed, "fit")
        for seed in (969400, 969431, 969550, 969599):
            self.assertEqual(allowed_seed(seed, "fit"), seed)

    def test_invisible_truth_cannot_change_features(self):
        s = state()
        expected = features(s)
        s["kin"][2] = [np.nan, np.nan, np.nan, np.nan]
        s["known_cav"][2] = 1
        s["penetration"] = .99
        s["hidden_driver_type"] = "unused"
        np.testing.assert_array_equal(expected, features(s))
        del s["flow_counts"]
        with self.assertRaises(KeyError):
            features(s)

    def test_coverage_is_not_empty_traffic(self):
        s = state()
        s["coverage"][0, 1] = 0
        with self.assertRaises(ValueError):
            features(s)

    def test_clock_uses_absolute900_not600(self):
        ts = np.arange(300., 900.5, .5)
        rem = remaining_labels(ts, (ts-300)*2, 1200)
        self.assertEqual(rem[0], 1200)
        self.assertEqual(rem[-1], 0)
        with self.assertRaises(ValueError):
            remaining_labels(ts, (ts-300)*2, 1199)
        with self.assertRaises(ValueError):
            remaining_labels(ts[:601], (ts[:601]-300)*2, 600)

    def test_command_summary_is_masked(self):
        a = action_summary(np.array([[1., 20.], [np.nan, np.nan]]),
                           np.array([[1, 1], [0, 0]]))
        np.testing.assert_array_equal(a, [1, 1, 0, 1, 20, 0])

    def test_fit_rejects_reserved_world_before_learning(self):
        with self.assertRaises(ValueError):
            Value().fit([dict(seed=969432, role="fit")])

    def test_real_value_fit_and_terminal_constraint(self):
        rows = [dict(seed=969400, role="fit", arm="native", state=state(t),
                     remaining_J=float(900-t), action_summary=np.zeros(6))
                for t in range(300, 901, 30)]
        v = Value().fit(rows)
        self.assertTrue(np.isfinite(v.predict(state(540))))
        self.assertEqual(v.predict(state(900)), 0.)
        self.assertEqual(v.fit_seeds, [969400])
        direct = Value(actions=True).fit(rows)
        self.assertEqual(direct.predict(state(900), np.zeros(6)), 0.)


class DecisionTests(unittest.TestCase):
    def test_missing_observations_block_fitting(self):
        docs = []
        for corpus, seeds, arms in (("wmdata", range(969400, 969432), ARMS),
                                    ("wmdata2", range(969550, 969600), (*ARMS, "generic_random"))):
            docs.append(dict(schema="wmvalue_audit/1", role="fit", corpus=corpus,
                             records=[dict(seed=s, arm=a, time_end=900, exists=True)
                                      for s in seeds for a in arms]))
        status = summarize(docs)
        self.assertEqual(status["verdict"], "INCOMPLETE")
        with self.assertRaises(ValueError):
            require_fit_ready(status)
        docs[0]["records"].append(docs[0]["records"][0])
        with self.assertRaises(ValueError):
            summarize(docs)

    def test_arm_baseline_cannot_use_reserved_outcomes(self):
        rows = [dict(seed=969400, role="fit", arm=a, state=state(t),
                     remaining_J=float(900-t)) for a in ARMS for t in TIMES]
        baseline = arm_means(rows)
        self.assertEqual(baseline["native", 300], 600.)
        rows[0]["seed"] = 969432
        with self.assertRaises(ValueError):
            arm_means(rows)

    def test_ties_and_constant_rank(self):
        np.testing.assert_array_equal(metrics([0., 1., -1.], [0., 0., 0.]), [1/3, 0., 2/3])
        self.assertAlmostEqual(metrics([1., 1., 2.], [3., 3., 5.])[1], 1.)

    def test_pairing_and_no_missing_cells(self):
        cells = test_cells()
        pairs = paired_rows(cells)
        self.assertEqual(len(pairs), 400)
        self.assertEqual(pairs[0]["real"], cells[1]["real"]-cells[0]["real"])
        with self.assertRaises(ValueError):
            paired_rows(cells[:-1])
        with self.assertRaises(ValueError):
            paired_rows(cells+[cells[0]])

    def test_gate_requires_superiority_to_direct_too(self):
        result = evaluate(test_cells("direct_perfect"))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["contrasts"]["B-D"]["sign_ci95"], [0., 0.])
        self.assertEqual(result["bootstrap"]["unit"], "whole world")

    def test_decisive_gate_accepts_clear_improvement(self):
        result = evaluate(test_cells())
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["winners"], ["B", "C"])
        self.assertEqual(result["metrics"]["B"]["mae"]["ci95"], [0., 0.])

    def test_bootstrap_value_per_future_before_averaging(self):
        class Fake:
            def predict(self, s, summary=None):
                return float(s["time_s"]**2)
        rolls = {a: [dict(future=k, start_time=300, end_time=900, J=1000.,
                          windows={str(h): dict(J=float(k+h), state=state(300+h))
                                   for h in (60, 90)}) for k in range(8)] for a in ARMS}
        out = assemble_predictions(969432, 300, {a: state() for a in ARMS}, rolls,
                                   Fake(), Fake(), {a: np.zeros(6) for a in ARMS},
                                   {(a, 300): 10. for a in ARMS})
        self.assertEqual(out["native"]["B"], 63.5+360**2)
        broken = copy.deepcopy(rolls)
        broken["native"][0]["end_time"] = 600
        with self.assertRaises(ValueError):
            assemble_predictions(969432, 300, {a: state() for a in ARMS}, broken,
                                 Fake(), Fake(), {}, {})


if __name__ == "__main__":
    unittest.main()
