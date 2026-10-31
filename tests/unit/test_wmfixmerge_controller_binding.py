"""Recorded-policy identity must bind entrant feedback before expensive panels."""
import pytest

from tools.wmfixmerge import entrant_binding as runtime


def test_wmdata_ctrl0_alias_and_speed_mappo_are_bound_by_recorded_hash():
    specs = {"ours_c1": {"sha256": "zero", "checkpoint": "zero.pt"},
             "coopoff_c1": {"sha256": "coop", "checkpoint": "coop.pt"},
             "mappo": {"sha256": "speed", "checkpoint": "speed.pt"}}
    episodes = [{"arm": arm, "policy_sha256": sha, "controllers_used": ["native", "direct"]}
                for arm, sha in (("ctrl0", "zero"), ("mappo", "speed"))]
    result = runtime.bind_entrant_specs(specs, episodes)
    assert result["ctrl0"] == specs["ours_c1"]
    assert result["mappo"] == specs["mappo"]
    assert "ctrl0" not in specs
    specs["mappo"] = {"sha256": "old-mappo", "checkpoint": "old.pt"}
    with pytest.raises(ValueError, match="mappo"):
        runtime.bind_entrant_specs(specs, episodes)


def test_direct_arm_never_inferred_and_conflicting_recordings_fail_closed():
    specs = {"ours_c1": {"sha256": "zero"}, "coopoff_c1": {"sha256": "coop"}}
    with pytest.raises(ValueError, match="unknown"):
        runtime.bind_entrant_specs(specs, [dict(arm="unknown", controllers_used=["direct"],
                                              policy_sha256="x")])
    episodes = [dict(arm="ctrl0", controllers_used=["direct"], policy_sha256=sha)
                for sha in ("zero", "different")]
    with pytest.raises(ValueError, match="ctrl0"):
        runtime.bind_entrant_specs(specs, episodes)


def test_original_fit_controller_map_is_identical():
    specs = {"ours_c1": {"sha256": "zero"}, "coopoff_c1": {"sha256": "coop"}}
    episodes = [dict(arm=a, policy_sha256=s["sha256"], controllers_used=["direct"])
                for a, s in specs.items()]
    assert runtime.bind_entrant_specs(specs, episodes) == specs
