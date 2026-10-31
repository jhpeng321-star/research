import copy
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .core import features
from .reconstruct import (OBS_KEYS, ZONES, boundary_events, command_summaries, label_clock, observe_prefix,
                          observations, project)


def tape():
    t = np.arange(0., 900.5, .5)
    n = len(t)
    d = {k: np.zeros((n, 3)) for k in ("s_m", "lane", "v_mps", "a_mps2")}
    d.update(time_s=t, present=np.zeros((n, 3), bool), is_cav=np.zeros((n, 3), bool),
             vid_index=np.full((n, 3), -1, int), vehicle_ids=np.array(["mainline.0"]))
    d["present"][1:, 0] = True
    d["vid_index"][1:, 0] = 0
    d["is_cav"][1:, 0] = True
    d["s_m"][1:, 0] = np.linspace(-2640, -100, n-1)
    d["v_mps"][1:, 0] = 10
    return d


class CausalTests(unittest.TestCase):
    def test_suffix_poisoning_and_physical_truncation_are_identical(self):
        d = tape()
        expected = features(observe_prefix(d, 360))
        altered = copy.deepcopy(d)
        for k in ("s_m", "v_mps", "a_mps2"):
            altered[k][721:] = np.nan
        altered["present"][721:] = True
        altered["vid_index"][721:] = 999999
        altered["is_cav"][721:] = False
        # Even future time boundaries and future registry contents are irrelevant.
        altered["time_s"][721:] = -500
        altered["vehicle_ids"] = np.array(["arbitrary future contents"])
        np.testing.assert_array_equal(expected, features(observe_prefix(altered, 360)))
        truncated = {k: d[k][:721] for k in OBS_KEYS}
        np.testing.assert_array_equal(expected, features(observe_prefix(truncated, 360)))
        stream = observations(altered, [300, 360, 420])
        next(stream)
        np.testing.assert_array_equal(expected, features(next(stream)))

    def test_unsensed_vehicle_and_acceleration_do_not_leak(self):
        kin = np.array([[-1000, 0, 10, .5], [-950, 0, 20, -2.],
                        [-930, 0, 12, 8.], [-900, 1, 20, 5.]])
        state = project(kin, np.ones(4), [1, 0, 0, 0], np.zeros((2, 2)), 360)
        self.assertEqual(state["sensor_mask"].tolist(), [True, True, False, False])
        self.assertEqual(state["kin"][1, 3], 0)
        before = features(state)
        kin[2:, 2:] = 5000
        kin[1, 3] = -999
        after = project(kin, np.ones(4), [1, 0, 0, 0], np.zeros((2, 2)), 360)
        np.testing.assert_array_equal(before, features(after))

    def test_projection_matches_existing_cav_rsu_helpers(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
        import torch
        from wms.control.belief import RsuZones, detection_mask

        rng = np.random.default_rng(4)
        kin = np.column_stack((rng.uniform(-2700, 550, 90), rng.integers(0, 3, 90),
                               rng.uniform(0, 31, 90), rng.normal(size=90)))
        present, cav = rng.random(90) > .1, rng.random(90) > .7
        state = dict(s_m=torch.tensor(kin[:, 0])[None], lane=torch.tensor(kin[:, 1])[None],
                     present=torch.tensor(present)[None], is_cav=torch.tensor(cav)[None])
        detected = detection_mask(state)[0].numpy()
        rsu = np.array([RsuZones().contains(int(k[1]), k[0]) for k in kin])
        out = project(kin, present, cav, np.zeros((2, 2)), 300)
        np.testing.assert_array_equal(out["sensor_mask"], present & (cav | detected | rsu))
        self.assertEqual(tuple(x[1] for x in RsuZones().by_lane), ZONES)

    def test_entry_counts_not_current_population_and_reject_interior_birth(self):
        d = tape()
        d["present"][200:, 0] = False
        events = boundary_events({k: d[k][:721] for k in OBS_KEYS})
        self.assertEqual(events.tolist(), [[.5, 0.]])
        out = observe_prefix(d, 360)
        np.testing.assert_array_equal(out["flow_counts"], [[1, 0], [0, 0]])
        d["s_m"][1, 0] = -500
        with self.assertRaisesRegex(ValueError, "entry detector"):
            observe_prefix(d, 360)

    def test_source_features_rejected(self):
        out = observe_prefix(tape(), 360)
        out["queue"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "prohibits"):
            features(out)

    def test_internal_junction_self_has_no_fabricated_lane(self):
        out = project(np.array([[70., -1., 10., .5]]), [1], [1], np.zeros((2, 2)), 360)
        f = features(out)
        self.assertEqual(f[-2], 1.)
        self.assertEqual(sum(f[1:105:7]), 0.)

    def test_direct_baseline_uses_issued_zero_not_null_or_future_motion(self):
        rows = [dict(timestamp_s=300, vehicle_id="cav", issued_command=0.,
                     issued_target_speed_mps=None, measured_speed_mps=999.),
                dict(timestamp_s=300, vehicle_id="unidentified", issued_command=500.,
                     issued_target_speed_mps=500.),
                dict(timestamp_s=301, vehicle_id="cav", issued_command=999.,
                     issued_target_speed_mps=999.)]
        stream = io.StringIO("\n".join(json.dumps(r) for r in rows))
        with patch("tools.wmvalue.reconstruct.gzip.open", return_value=stream):
            summary = command_summaries("recorded_actions.gz", [300, 900], {300: {"cav"}})
        np.testing.assert_array_equal(summary[300], [1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(summary[900], np.zeros(6))


class ClockTests(unittest.TestCase):
    def test_birth_exit_endstep_source_and_objective_reconciliation(self):
        d = tape()
        d["present"][:] = False
        d["vid_index"][:] = -1
        d["present"][601:603, 0] = True
        d["vid_index"][601:603, 0] = 0
        d["v_mps"][601, 0] = 0
        d["v_mps"][602, 0] = 31.29
        clocks = {"120": dict(road=.5, source=0., J=.5),
                  "300": dict(road=.5, source=100., J=100.5),
                  "600": dict(road=.5, source=400., J=400.5)}
        with patch("tools.wmvalue.reconstruct.departures",
                   side_effect=lambda seed, rate, stream: np.array([300.1, 500.1])
                   if stream == "mainline" else np.array([])):
            out = label_clock(d, 969400, 1900, clocks)
            self.assertEqual(out["remaining"][0], 400.5)
            self.assertEqual(out["remaining"][-1], 0.)
            clocks["300"]["source"] += .5
            with self.assertRaisesRegex(ValueError, "objective_300 disagreement"):
                label_clock(d, 969400, 1900, clocks)


if __name__ == "__main__":
    unittest.main()
