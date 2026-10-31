"""Bind direct entrant algorithms to the policies named by recorded TRAIN data."""


def bind_entrant_specs(specs, episodes):
    result = {a: dict(specs[a]) for a in ("ours_c1", "coopoff_c1")}
    for episode in episodes:
        if "direct" not in episode.get("controllers_used", ()):
            continue
        arm = episode["arm"]
        # WMDATA's CTRL0 is exactly the already-bound zero/ours_c1 checkpoint.
        # No direct action is extrapolated from an arbitrary recorded identity.
        spec = specs.get(arm, specs.get("ours_c1") if arm == "ctrl0" else None)
        expected = episode.get("policy_sha256")
        if not expected or not spec or spec.get("sha256") != expected:
            raise ValueError(f"missing/mismatched recorded entrant policy for {arm}")
        if arm in result and result[arm] != spec:
            raise ValueError(f"conflicting recorded entrant policy for {arm}")
        result[arm] = dict(spec)
    return result
