"""The registered scientific gate cannot be weakened by unattended re-arm plumbing."""
import copy

import pytest

from tools.nativelane.adopt import DEADLINE, eligible


def receipt():
    return {"adopt": True, "adopt_scientific": True, "decision_time": DEADLINE - 3600,
            "absolute_gap_reduction_ci95": [1., .2, 1.8],
            "vacate_contrast_per_cav_h_ci95": {"on": [.2, 0., .4], "sumo": [-.5, -.7, -.3]},
            "native_counts": {"counts_move_toward_sumo": True, "n": 33},
            "qualification": {"complete": True, "pass_to_fail": []},
            "parity": {"synthetic_bit_identity": {"legacy": {"bit_identical": True},
                                                   "loop14c": {"bit_identical": True}}}}


def test_ci_excludes_no_change_or_sign_agreement_is_required():
    d = receipt()
    assert eligible(d)
    d["absolute_gap_reduction_ci95"][1] = 0.
    assert not eligible(d)
    d["vacate_contrast_per_cav_h_ci95"]["on"][0] = -.1
    assert eligible(d)


@pytest.mark.parametrize("field", ["count", "cohort", "qualification", "flip", "parity", "deadline", "decision"])
def test_each_mandatory_veto_is_fail_closed(field):
    d = copy.deepcopy(receipt())
    if field == "count":
        d["native_counts"]["counts_move_toward_sumo"] = False
    elif field == "cohort":
        d["native_counts"]["n"] = 32
    elif field == "qualification":
        d["qualification"]["complete"] = False
    elif field == "flip":
        d["qualification"]["pass_to_fail"] = ["set1:pulse_m/follower@5"]
    elif field == "parity":
        d["parity"]["synthetic_bit_identity"]["legacy"]["bit_identical"] = False
    elif field == "deadline":
        d["decision_time"] = DEADLINE
    else:
        d["adopt"] = False
    assert not eligible(d)
