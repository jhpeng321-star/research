"""WD-01-PA2G-LC2 variant tests (registration REGISTRATION_PA2G_LC2.md).

Pins the registered structural LC variants as OFF-by-default flags:

- V-a persistence: a discretionary change fires only after the MOBIL
  criterion held for 4 consecutive qualifying steps (T_persist 2.0 s at
  dt 0.5); a failing step resets the counter.
- V-b manoeuvre time: after any change the vehicle is blocked from
  DISCRETIONARY candidacy for 4.0 s; mandatory merges are unaffected.
- V-c strategic early merge: zero incentive threshold plus b_merge
  1.0 m/s^2 on the merge gain from the zone start; the safe-gap tests
  are unchanged.
- Nominal integrity: params_nominal.json untouched (both sha forms);
  with every flag off the engine path is identical to the raw nominal
  parameters (engine-level trajectory identity, and -- when the local
  truth root and the pre-change reference exist -- byte-identity of the
  registered eval rows on the reference world).
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WD01 = HERE.parents[1]
for p in (str(WD01), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from partA_v2g import variants as VAR  # noqa: E402
from partA_v2g.engine import (PartAV2GEngine, load_params,  # noqa: E402
                              param_set_id)

P = load_params()

NOMINAL_RAW_SHA = "ba1fd93f0a21b1ffd2cdf20693e74fb8e3fbb8e4a1c57e7f88d9a3929e3455ce"
NOMINAL_LF_SHA_PREFIX = "079a55768f803b23c92af188680c1df91dc63f04acbbbac948870b46e75935f7"


def _lf_sha256(path: Path) -> str:
    import hashlib
    b = path.read_bytes()
    lf = bytes(b[i] for i in range(len(b))
               if not (b[i] == 13 and i + 1 < len(b) and b[i + 1] == 10))
    return hashlib.sha256(lf).hexdigest()


def eng(vehicles, params=P, steps=0):
    e = PartAV2GEngine(params, t0=480.0, vehicles=vehicles)
    e.run(steps)
    return e


# ----------------------------------------------------- params integrity

def test_params_nominal_unchanged():
    assert param_set_id() == NOMINAL_RAW_SHA
    assert _lf_sha256(
        WD01 / "partA_v2g" / "params_nominal.json").startswith(
        NOMINAL_LF_SHA_PREFIX)
    assert "lc_variants" not in P


def test_variant_config_registered_values():
    cfg = VAR.load_variant_config()
    assert cfg["registered_values"]["va_persistence"]["T_persist_s"] == 2.0
    assert cfg["registered_values"]["vb_manoeuvre"]["T_lc_s"] == 4.0
    assert cfg["registered_values"]["vc_early_merge"]["b_merge_mps2"] == 1.0
    assert cfg["registered_values"]["vc_early_merge"]["threshold_zero"] \
        is True


def test_variant_params_injection_and_off_default():
    for name, va, vb, vc in (("nominal", False, False, False),
                             ("va", True, False, False),
                             ("vb", False, True, False),
                             ("vc", False, False, True),
                             ("vabc", True, True, True)):
        sec = VAR.variant_params(P, name)["lc_variants"]
        assert sec["va_enabled"] is va
        assert sec["vb_enabled"] is vb
        assert sec["vc_enabled"] is vc
    assert VAR.enabled_flags(P) == {"va_enabled": False,
                                    "vb_enabled": False,
                                    "vc_enabled": False}
    vp = VAR.variant_params(P, "va")
    assert vp["lc_variants"]["T_persist_s"] == 2.0
    with pytest.raises(ValueError):
        VAR.lc_variants_section("nope")
    before = json.dumps(P, sort_keys=True)
    VAR.variant_params(P, "vabc")
    assert json.dumps(P, sort_keys=True) == before  # input not mutated


# ------------------------------------------------------------ V-a scenes

VA_SCENE = {"lead": dict(lane=1, s=2000.0, v=5.0),
            "ego": dict(lane=1, s=1950.0, v=28.0)}


def _first_disc_change(params, scene, cap=30):
    """Step of the first executed discretionary change (any vehicle)."""
    e = PartAV2GEngine(params, t0=480.0, vehicles=dict(scene))
    for j in range(1, cap + 1):
        e.step()
        if e.lc_counts["disc_left"] + e.lc_counts["disc_right"] >= 1:
            return j, e
    return None, e


def test_va_fires_only_after_4_consecutive_qualifying_steps():
    j_nom, _ = _first_disc_change(P, VA_SCENE)
    assert j_nom == 1  # ungated: the criterion qualifies at step 1

    j_va, e = _first_disc_change(VAR.variant_params(P, "va"), VA_SCENE)
    assert j_va == 4  # exactly the 4th consecutive qualifying step
    assert e._persist[("lead", 0)] == 0 and e._persist[("ego", 0)] == 0
    # and no change was executed before the counter reached 4
    e2 = PartAV2GEngine(VAR.variant_params(P, "va"), t0=480.0,
                        vehicles=dict(VA_SCENE))
    for j in range(1, 4):
        e2.step()
        assert e2.lc_counts["disc_left"] == 0
        assert e2._persist[("lead", 0)] == j  # counts once per step


def test_va_counter_resets_on_broken_chain():
    # the ego's incentive must SURVIVE the 2 s persistence itself: a
    # 200 m closing gap keeps the criterion qualified while the counter
    # builds (a 45 m queue gap would defuse during the 4 steps)
    scene = {"slowlead": dict(lane=1, s=2250.0, v=5.0),
             "ego": dict(lane=1, s=2050.0, v=28.0),
             "blocker": dict(lane=0, s=3000.0, v=0.0)}
    e = PartAV2GEngine(VAR.variant_params(P, "va"), t0=480.0,
                       vehicles=scene)
    i_ego = e.names.index("ego")
    i_blk = e.names.index("blocker")
    e.run(2)
    assert e.lc_counts["disc_left"] == 0
    assert e._persist[("ego", 0)] == 2
    # break the chain: a stopped blocker 6 m ahead of the ego's mapped
    # lane-0 position (gap_lead ~ 1 m -> the incentive fails)
    e.s[i_blk] = float(e.s[i_ego]) + (float(e.off[0]) - float(e.off[1])) \
        + 6.0
    e.step()
    assert e.lc_counts["disc_left"] == 0
    assert int(e.lane[i_ego]) == 1
    assert e._persist[("ego", 0)] == 0  # reset by the failing step
    # restore the free target lane: the ego fires on the 4th consecutive
    # qualifying step after the rebuild
    e.s[i_blk] = 3000.0
    fired = None
    for j in range(30):
        cnt = e._persist.get(("ego", 0), 0)
        e.step()
        if int(e.lane[i_ego]) == 0:
            fired = (j, cnt)
            break
    assert fired is not None, "ego never re-fired after the reset"
    assert fired[1] == 3  # fired on the 4th consecutive qualifying step


# ------------------------------------------------------------ V-b scenes

VB_SCENE = {"lead": dict(lane=1, s=2020.0, v=5.0),
            "ego": dict(lane=1, s=1950.0, v=28.0)}


def test_vb_blocks_discretionary_for_4s_after_any_change():
    j_nom, _ = _first_disc_change(P, VB_SCENE)
    assert j_nom == 1  # ungated: the lead's criterion qualifies at step 1

    e = PartAV2GEngine(VAR.variant_params(P, "vb"), t0=480.0,
                       vehicles=dict(VB_SCENE))
    assert e._disc_cooldown == 4.0
    i_lead = e.names.index("lead")
    e._lc_last["lead"] = 480.0   # a (simulated) change at t0
    for j in range(1, 9):        # t_change in [480.0, 483.5]: blocked
        e.step()
        assert int(e.lane[i_lead]) == 1, \
            f"lead changed within the 4.0 s manoeuvre block (step {j})"
    # the block expires at t_change >= 484.0 (8 steps of 0.5 s); the lead
    # is eligible again from step 9 (its incentive may legitimately fail
    # now, so only the block itself is asserted)


MERGE_SCENE = {"mr": dict(lane=1, s=460.0, v=25.0),
               "acc": dict(lane=2, s=385.0, v=28.0)}


def test_vb_mandatory_merges_unaffected():
    e = PartAV2GEngine(VAR.variant_params(P, "vb"), t0=480.0,
                       vehicles=dict(MERGE_SCENE))
    e._lc_last["acc"] = 480.0   # fresh change; nominal cooldown is 0.0
    e.step()
    assert e.lc_counts["merge"] == 1, \
        "the manoeuvre-time block must not gate mandatory merges"


# ------------------------------------------------------------ V-c scenes

VC_SCENE = {"foll": dict(lane=1, s=2638.56, v=1.0),
            "acc": dict(lane=2, s=385.0, v=0.0)}


def _merge_decision_pair(scene):
    en = eng(dict(scene), P, steps=0)
    ev = eng(dict(scene), VAR.variant_params(P, "vc"), steps=0)
    d1n, r1n = en._lane_d_sorted(1)
    d1v, r1v = ev._lane_d_sorted(1)
    r = en.names.index("acc")
    return en._merge_decision(r, d1n, r1n)[0], \
        ev._merge_decision(r, d1v, r1v)[0]


def test_vc_zero_threshold_plus_bias_accepts_incentive_only_rejects():
    nominal_acc, vc_acc = _merge_decision_pair(VC_SCENE)
    assert nominal_acc is False
    assert vc_acc is True


def test_vc_behaves_early_merge_in_rollout():
    e = eng(dict(VC_SCENE), VAR.variant_params(P, "vc"), steps=10)
    assert e.lc_counts["merge"] >= 1
    n = eng(dict(VC_SCENE), P, steps=10)
    assert n.lc_counts["merge"] == 0


def test_vc_safety_test_unchanged():
    scene = {"foll": dict(lane=1, s=2265.56 + 430.0 - 9.0, v=25.0),
             "acc": dict(lane=2, s=430.0, v=3.0)}
    nominal_acc, vc_acc = _merge_decision_pair(scene)
    assert nominal_acc is False
    assert vc_acc is False  # the safe-gap rejection survives the bias


def test_vc_candidacy_from_zone_start_unchanged():
    """Candidacy already spans the whole merge zone in the nominal
    params (zone_s0_m = 378.66); V-c must not widen it."""
    s0 = P["merge"]["zone_s0_m"]
    s1 = P["merge"]["zone_s1_m"]
    assert (s0, s1) == (378.66, 454.1)


# ------------------------------------------------- nominal flag-off path

def test_nominal_flags_off_engine_identity():
    scenes = (VA_SCENE, VB_SCENE, MERGE_SCENE, VC_SCENE)
    for scene in scenes:
        a = eng(dict(scene), P, steps=40)
        b = eng(dict(scene), VAR.variant_params(P, "nominal"), steps=40)
        assert a.trajectory_table() == b.trajectory_table()
        assert a.lc_counts == b.lc_counts
        assert a.collisions == b.collisions
        assert a.exited == b.exited


def test_nominal_byte_identity_vs_prechange_reference():
    """Registered gate: the flags-off eval path must reproduce the
    PRE-change reference rows byte-identically (canonical json) on the
    reference world (first two windows; full gate in run_lc2.py)."""
    root = WD01 / "wdpaths.py"
    if not root.exists():
        pytest.skip("wdpaths unavailable")
    sys.path.insert(0, str(WD01))
    import wdpaths  # noqa: E402
    data_root = Path(wdpaths.RESEARCH_DENSE_ROOT)
    ref = WD01 / "partA_v2g" / "results" / "lc2" / \
        "nominal_ref_prechange.jsonl"
    if not data_root.exists() or not ref.exists():
        pytest.skip("truth root or pre-change reference not present")
    from partA_v2g import eval_a as EA
    from partA_v2g.run_lc2 import lc2_window_rows, load_panel_entries
    EA.MAPPER = EA.ModelMapper()
    entry = load_panel_entries()[0]
    ref_rows = {}
    for line in ref.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ref_rows[(r["world"], r["w"], r["H"])] = json.dumps(r,
                                                            sort_keys=True)
    log = EA.NativeLog(entry)
    n_cmp = 0
    for k in (1, 2):
        w = log.t0 + 30.0 * k
        rows, _mo = lc2_window_rows(VAR.variant_params(P, "nominal"),
                                    log, entry, w, "v2g")
        for r in rows:
            key = (r["world"], r["w"], r["H"])
            assert key in ref_rows
            assert json.dumps(r, sort_keys=True) == ref_rows[key], \
                f"byte-identity failed at {key}"
            n_cmp += 1
    assert n_cmp == 8
