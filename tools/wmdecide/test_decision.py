import unittest

import numpy as np

from .decision import Future, decide, summarize


def futures(values, events=None):
    return [Future(str(i), float(v), None if events is None else events[i])
            for i, v in enumerate(values)]


class DecisionTests(unittest.TestCase):
    def test_common_random_numbers_remove_shared_noise(self):
        noise = np.array([0, 1000, -300, 400, 20, 900, -40, 200])
        result = summarize({"a": futures(noise + 10), "b": futures(noise + 12)})
        pair = result["pairwise"][0]
        self.assertEqual(pair["saving"]["sd"], 0)
        self.assertEqual(pair["saving"]["mean"], 2)
        self.assertEqual(pair["probability_beats"], 1)
        self.assertGreater(result["actions"]["a"]["team_delay"]["sd"], 100)

    def test_gate_boundary_and_expected_delay(self):
        base = futures([10] * 10)
        yes = futures([9] * 8 + [11] * 2)
        no = futures([9] * 7 + [11] * 3)
        self.assertEqual(decide({"base": base, "a": yes}, "base")["selected"], "a")
        self.assertEqual(decide({"base": base, "a": no}, "base")["selected"], "base")
        catastrophe = futures([9] * 9 + [100])
        self.assertFalse(decide({"base": base, "a": catastrophe}, "base")["switched"])

    def test_pair_ids_align_and_fail_closed(self):
        a, b = futures([1, 2]), list(reversed(futures([2, 3])))
        self.assertEqual(summarize({"a": a, "b": b})["pairwise"][0]["saving"]["sd"], 0)
        for bad in ([b[0]], [b[0], b[0]], futures([1, float("nan")])):
            with self.assertRaises(ValueError):
                summarize({"a": a, "b": bad})

    def test_ties_unknown_and_single_draw(self):
        a = futures([4], [True])
        result = decide({"a": a, "b": futures([4])}, "a")
        self.assertFalse(result["switched"])
        self.assertEqual(result["summary"]["pairwise"][0]["probability_tie"], 1)
        self.assertEqual(result["summary"]["actions"]["a"]["breakdown_probability"], 1)
        self.assertIsNone(result["summary"]["actions"]["b"]["breakdown_probability"])
        self.assertIsNone(result["summary"]["actions"]["a"]["team_delay"]["sd"])

    def test_lowest_eligible_expected_delay(self):
        result = decide({"base": futures([10] * 8), "b": futures([8] * 8),
                         "a": futures([8] * 8), "c": futures([9] * 8)}, "base")
        self.assertEqual(result["selected"], "a")


if __name__ == "__main__":
    unittest.main()
