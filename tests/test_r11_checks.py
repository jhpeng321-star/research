"""R11 post-run check logic (R11_REVIEW C3): the registered keep-10b rule and the one-step absolute profile."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools" / "r11_checks", ROOT / "tools" / "r10b_checks", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import absolute_bias as AB  # noqa: E402

BASE = {"merges_per_run": 10.0, "stuck_mean_ramp_in_zone": 1.0, "stuck_share_wait_gt60": 0.1,
        "ttc_lt_1p5_share": 0.01, "min_gap_ins_p5": 2.0, "post_ins_hard_brake_share": 0.1}


def test_keep_10b_needs_both_the_bias_excess_and_a_worse_behaviour():
    same = {AB.REF: dict(BASE), AB.R11: dict(BASE)}
    # bias worse by > 0.3 but no behaviour worse -> pass
    assert AB.decide({AB.REF: 0.3, AB.R11: 0.7}, {}, same, -0.3)["verdict"] == "PASS"
    # behaviour worse but bias within 0.3 -> pass
    worse = {AB.REF: dict(BASE), AB.R11: {**BASE, "merges_per_run": 9.0}}
    assert AB.decide({AB.REF: 0.3, AB.R11: 0.55}, {}, worse, -0.3)["verdict"] == "PASS"
    # both -> keep 10b; the one-step bias alone can trigger too
    d = AB.decide({AB.REF: 0.3, AB.R11: 0.7}, {}, worse, -0.3)
    assert d["verdict"].startswith("KEEP-ROUND-10B") and d["worse_than_10b"] == ["merges_per_run"]
    assert AB.decide({AB.REF: 0.3, AB.R11: 0.4}, {AB.REF: 0.2, AB.R11: 0.6}, worse, -0.3)["bias_trigger"]
    for key, sign in AB.WORSE_SIGN.items():
        beh = {AB.REF: dict(BASE), AB.R11: {**BASE, key: BASE[key] + sign * 0.5}}
        assert AB.decide({AB.REF: 0.0, AB.R11: 0.5}, {}, beh, -0.3)["worse_than_10b"] == [key]
    assert AB.decide({}, {}, same, None)["verdict"] == "MISSING"


def test_under_yield_flag():
    same = {AB.REF: dict(BASE), AB.R11: dict(BASE)}
    assert AB.decide({AB.REF: 0.0, AB.R11: 0.0}, {}, same, -0.05)["under_yield_flag"]
    assert not AB.decide({AB.REF: 0.0, AB.R11: 0.0}, {}, same, -0.3)["under_yield_flag"]


def test_onestep_profile_bins_pre_and_post_merge_rows():
    n = 400
    rng = np.random.default_rng(0)
    d = {"fin": np.zeros(n), "rec": np.zeros(n), "has": np.zeros(n), "tm": np.full(n, np.nan),
         "ev": -np.ones(n), "te": np.full(n, np.nan), "win": rng.integers(0, 20, n).astype(float)}
    d["has"][:200] = 1
    d["tm"][:200] = np.repeat([1.0, 2.0, 5.0, 6.0], 50)
    d["fin"][:100] = 1.0                                         # +1 bias at -1 / -2 s, 0 at -5 / -6 s
    d["ev"][200:300] = 0
    d["te"][200:300] = 1.0
    d["fin"][200:300] = -0.5
    prof, last3, sel = AB.onestep_profile(d)
    assert prof[-1][0] == 1.0 and prof[-2][0] == 1.0 and prof[-5][0] == 0.0 and prof[1][0] == -0.5
    assert last3[0] == 1.0 and int(sel.sum()) == 100
