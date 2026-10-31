import unittest

import numpy as np

from .audit import ARMS
from .fit_causal import validate_arrays


def complete():
    cells = [(s, a, t) for s in range(969400, 969432) for a in ARMS for t in range(300, 901, 30)]
    return dict(seed=np.array([s for s, _, _ in cells]), arm=np.array([a for _, a, _ in cells]),
                time_s=np.array([t for _, _, t in cells]), x=np.zeros((len(cells), 111)),
                actions=np.zeros((len(cells), 6)), y=np.array([900-t for _, _, t in cells], float))


class PreparedFitTests(unittest.TestCase):
    def test_reserved_and_missing_cells_cannot_enter_fitting(self):
        data = complete()
        validate_arrays(data, "wmdata")
        data["seed"][0] = 969432
        with self.assertRaisesRegex(ValueError, "seed outside"):
            validate_arrays(data, "wmdata")
        data = {k: v[:-1] for k, v in complete().items()}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_arrays(data, "wmdata")

    def test_exact_terminal_constraint(self):
        data = complete()
        data["y"][20] = 1
        with self.assertRaisesRegex(ValueError, "terminal"):
            validate_arrays(data, "wmdata")


if __name__ == "__main__":
    unittest.main()
