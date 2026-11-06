"""Counted exclusion of TF sequences made impossible by the WM's own one-step motion."""
import torch

from wms.model.merge_pivot import FirstPassageNet
from wms.train.losses.merge_pivot import score_chunk
from wms.train.losses.merge_pivot_support import (
    CLASSES, boundary_support, counted_exclusion, exclusion_counts, lifecycle_rows,
    prefix_survival_zero)

TARGET = 1


def make_geo(n=8, k=4):
    shape = (1, n, k)
    z = lambda dtype=torch.float32: torch.zeros(shape, dtype=dtype)  # noqa: E731
    return dict(eligible=torch.ones(shape, dtype=torch.bool), v=torch.full(shape, 10.),
                has_f=z(torch.bool), has_l=z(torch.bool), gap_f=z(), gap_l=z(),
                v_f=z(), v_l=z(), pre_target=z(torch.bool), label_succ_x=z(),
                label_succ_length=torch.full(shape, 5.), label_succ_lane=z(torch.long),
                label_linked=z(torch.bool), label_pre_junction=z(torch.bool))


def side(geo, t, k, which="f", gap=20., v=8.):
    geo["has_" + which][0, t, k] = True
    geo["gap_" + which][0, t, k], geo["v_" + which][0, t, k] = gap, v


def chains(n, k, spec):
    """spec: slot -> (first_row, last_row, event?)."""
    risk = torch.zeros(1, n, k, dtype=torch.bool)
    event = torch.zeros_like(risk)
    for slot, (a, b, ev) in spec.items():
        risk[0, a:b + 1, slot] = True
        event[0, b, slot] = ev
    return risk, event, torch.zeros_like(risk)


def run(geo, spec, prefix=None, n=8, k=4):
    risk, event, reset = chains(n, k, spec)
    _, live = lifecycle_rows(risk, event, reset)
    res = counted_exclusion(geo, live, event, reset, target_lane=TARGET, prefix_zero=prefix)
    return res, risk, event, reset


def test_boundary_support_is_the_parameter_independent_infinity_pattern_of_the_boundary():
    torch.manual_seed(0)
    n, k = 40, 6
    geo = make_geo(n, k)
    for key in ("gap_f", "gap_l"):
        geo[key] = torch.rand(1, n, k) * 30 - 3          # includes non-positive gaps
    for key in ("has_f", "has_l", "eligible"):
        geo[key] = torch.rand(1, n, k) > 0.35
    geo["v"] = torch.rand(1, n, k) * 20
    geo["v_f"], geo["v_l"] = torch.rand(1, n, k) * 20, torch.rand(1, n, k) * 20
    geo["eligible"][0, 3, 2] = False
    geo["v"][0, 3, 2] = torch.nan            # ineligible rows may carry anything
    geo["eligible"][0, 4, 1] = False
    geo["gap_l"][0, 4, 1] = torch.nan
    for init in ({}, {"s0_m": 7.0, "T_s": 0.4}):
        c = FirstPassageNet(init).boundary(geo)
        dead, atom = boundary_support(geo)
        assert torch.equal(dead, c == -torch.inf)
        assert torch.equal(atom, c == torch.inf)
        assert bool(dead.any()) and bool(atom.any()) and bool(torch.isfinite(c).any())


def test_vetoed_event_chain_is_excluded_whole_and_counted_other_chains_untouched():
    geo = make_geo()
    for t in range(8):
        for slot in (0, 1):
            side(geo, t, slot)                          # finite boundary: supported
    geo["eligible"][0, 4, 0] = False                    # slot 0: event step vetoed
    res, risk, event, reset = run(geo, {0: (1, 4, True), 1: (1, 4, True)})
    assert res["rows"][0, :, 0].tolist() == [False] + [True] * 4 + [False] * 3
    assert not res["rows"][0, :, 1].any()
    assert res["starts"][0, :, 0].nonzero().flatten().tolist() == [1]
    # no recorded successor in this fixture: the veto is the model's own motion
    assert int(res["cls"][0, 1, 0]) == CLASSES.index("event_vetoed_wm_motion")
    assert int(res["event"].sum()) == 1
    assert exclusion_counts(res)["excluded_cars"] == 1


def test_zero_clearance_atom_wait_and_prefix_are_each_counted_once_by_class():
    geo = make_geo()
    for t in range(8):
        for slot in range(4):
            side(geo, t, slot)
    geo["gap_f"][0, 3, 0] = 0.0                         # slot 0: touching at the event step
    geo["has_f"][0, 2, 1] = False                       # slot 1: certain passage at row 2 ...
    geo["has_l"][0, 2, 1] = False                       # ... but it keeps waiting (event row 4)
    prefix = torch.zeros(1, 4, dtype=torch.bool)
    prefix[0, 2] = True                                 # slot 2: certified prefix passed an atom
    spec = {0: (1, 3, True), 1: (1, 4, True), 2: (0, 4, True), 3: (1, 4, True)}
    res, *_ = run(geo, spec, prefix)
    counts = exclusion_counts(res)
    assert counts["excluded_event_zero_clearance"] == 1
    assert counts["excluded_wait_survival_zero"] == 1
    assert counts["excluded_prefix_survival_zero"] == 1
    assert counts["excluded_cars"] == 3
    assert counts["excluded_event_cars"] == 3 and counts["excluded_censor_cars"] == 0

    assert not res["rows"][0, :, 3].any()               # slot 3 is supported


def test_certain_passage_that_ends_in_the_event_is_supported_not_excluded():
    geo = make_geo()
    for t in range(8):
        side(geo, t, 0)
    geo["has_f"][0, 4, 0] = False                       # atom exactly at the event step
    res, *_ = run(geo, {0: (1, 4, True)})
    assert not res["rows"].any()


def test_exclusion_drops_the_identity_tail_so_a_reentry_is_not_born_as_a_new_chain():
    geo = make_geo()
    for t in range(8):
        side(geo, t, 0)
    geo["eligible"][0, 3, 0] = False
    risk, event, reset = chains(8, 4, {0: (1, 3, True)})
    risk[0, 5:7, 0] = True                              # re-entry of the same identity
    _, live = lifecycle_rows(risk, event, reset)
    res = counted_exclusion(geo, live, event, reset, target_lane=TARGET)
    assert res["tail"][0, :, 0].tolist() == [False] + [True] * 7
    reset[0, 6, 0] = True                               # a NEW identity starts at row 6
    _, live = lifecycle_rows(risk, event, reset)
    res = counted_exclusion(geo, live, event, reset, target_lane=TARGET)
    assert res["tail"][0, :, 0].tolist() == [False] + [True] * 5 + [False] * 2


def test_supported_sequence_loss_is_bit_identical_and_impossible_one_is_finite_after_exclusion():
    torch.manual_seed(1)
    n, k = 8, 2
    geo = make_geo(n, k)
    for t in range(n):
        for slot in range(k):
            side(geo, t, slot, gap=12. + 3 * t)
    geo["eligible"][0, 5, 1] = False                    # slot 1: impossible event
    net = FirstPassageNet({}, hybrid=True)
    params = net.params()
    c = net(geo)
    risk, event, reset = chains(n, k, {0: (1, 6, True), 1: (1, 5, True)})
    _, live = lifecycle_rows(risk, event, reset)
    kw = dict(known_entry=torch.ones_like(risk), reset=reset, eligible=geo["eligible"],
              w=params["w"])
    raw, _, stats_raw = score_chunk(c, params["tau"], risk, event, **kw)
    assert stats_raw["zero_mass_events"] == 1 and not torch.isfinite(raw)
    res = counted_exclusion(geo, live, event, reset, target_lane=TARGET)
    after, _, stats = score_chunk(c, params["tau"], risk & ~res["tail"], event & ~res["tail"], **kw)
    assert stats["zero_mass_events"] == 0 and stats["zero_survival"] == 0 and torch.isfinite(after)
    alone_risk, alone_event = risk.clone(), event.clone()
    alone_risk[..., 1], alone_event[..., 1] = False, False
    alone, _, _ = score_chunk(c, params["tau"], alone_risk, alone_event, **kw)
    assert torch.equal(after, alone)                    # supported sequence: bit for bit


def test_prefix_survival_zero_reads_only_certified_history_atoms():
    geo = make_geo(4, 3)
    for t in range(4):
        side(geo, t, 0)
        side(geo, t, 1)
    geo["has_f"][0, 1, 0] = geo["has_l"][0, 1, 0] = False   # slot 0: atom in prefix
    geo["has_f"][0, 1, 1] = geo["has_l"][0, 1, 1] = False   # slot 1: atom, uncertified
    history = dict(geometry=geo, history_mask=torch.ones(1, 4, 3, dtype=torch.bool),
                   certified=torch.tensor([[True, False, False]]))
    out = prefix_survival_zero([history, None], torch.zeros(2, 3))
    assert out.tolist() == [[True, False, False], [False, False, False]]


def test_fitter_flow_drops_exactly_the_chains_the_training_flow_flags():
    from tools import r9_fit_exact as fit

    n, k = 8, 4
    geo = make_geo(n, k)
    for t in range(n):
        for slot in range(k):
            side(geo, t, slot)
    geo["eligible"][0, 4, 0] = False                    # slot 0: vetoed event
    geo["has_f"][0, 2, 1] = geo["has_l"][0, 2, 1] = False   # slot 1: certain passage, keeps waiting
    spec = {0: (1, 4, True), 1: (1, 4, True), 2: (1, 4, True), 3: (1, 5, False)}
    res, *_ = run(geo, spec)
    expected = {slot for slot in range(k) if bool(res["rows"][0, :, slot].any())}
    assert expected == {0, 1}
    sequences = []
    for slot, (a, b, ev) in spec.items():
        sequences.append(dict(world="w", source="s", episode="e", vehicle=slot, slot=slot,
                              entry_row=a, event_row=b + 1 if ev else None, event=ev,
                              times=list(range(b - a + 1)), unknown_age=False,
                              lifecycle_active=[True] * (b - a + 1)))
    cache = dict(geometry=geo, metadata=dict(physics=dict(merge_target_lane=TARGET)))
    kept, dropped = fit.drop_counted_exclusions(sequences, cache)
    assert {s["slot"] for s in dropped} == expected
    assert {s["slot"] for s in kept} == {2, 3}
    classes = {s["slot"]: s["class"] for s in dropped}
    assert classes == {0: "event_vetoed_wm_motion", 1: "wait_survival_zero"}
    assert all(s["scored"] for s in dropped)


def test_non_finite_eligible_geometry_fails_loudly_instead_of_being_excluded():
    import pytest

    for key, value in (("v", torch.nan), ("gap_f", torch.nan), ("v_f", torch.inf)):
        geo = make_geo()
        for t in range(8):
            side(geo, t, 0)
        geo[key][0, 3, 0] = value
        with pytest.raises(ValueError, match="numerical error"):
            boundary_support(geo)
        geo["eligible"][0, 3, 0] = False            # an ineligible row may carry anything
        boundary_support(geo)


def test_censored_cars_are_kept_unless_their_survival_is_exactly_zero():
    geo = make_geo()
    for t in range(8):
        for slot in range(4):
            side(geo, t, slot)
    geo["eligible"][0, 4, 0] = False                    # slot 0: censored while vetoed: kept
    geo["has_f"][0, 3, 1] = geo["has_l"][0, 3, 1] = False   # slot 1: censored survivor of a certain passage
    spec = {0: (1, 4, False), 1: (1, 4, False), 2: (1, 5, False), 3: (1, 4, True)}
    res, *_ = run(geo, spec)
    counts = exclusion_counts(res)
    assert not res["rows"][0, :, 0].any()               # ineligible at the censor step is NOT a veto
    assert counts["excluded_cars"] == 1 and counts["excluded_wait_survival_zero"] == 1
    assert counts["excluded_censor_cars"] == 1 and counts["excluded_event_cars"] == 0
    assert res["rows"][0, :, 1].tolist() == [False] + [True] * 4 + [False] * 3


def test_an_excluded_event_is_never_converted_into_a_censor():
    geo = make_geo()
    for t in range(8):
        side(geo, t, 0)
    geo["eligible"][0, 4, 0] = False
    res, risk, event, reset = run(geo, {0: (1, 4, True)})
    kept_risk, kept_event = risk & ~res["tail"], event & ~res["tail"]
    assert not kept_risk[0, :, 0].any() and not kept_event.any()   # no waiting factors survive


def test_primary_class_is_disjoint_and_overlapping_causes_are_reported():
    geo = make_geo()
    for t in range(8):
        side(geo, t, 0)
    geo["has_f"][0, 2, 0] = geo["has_l"][0, 2, 0] = False   # certain passage at row 2 ...
    geo["gap_f"][0, 4, 0] = 0.0                             # ... and zero clearance at the event
    res, *_ = run(geo, {0: (1, 4, True)})
    counts = exclusion_counts(res)
    assert counts["excluded_cars"] == 1 and counts["excluded_overlap_cars"] == 1
    assert counts["excluded_event_zero_clearance"] == 1 and counts["excluded_wait_survival_zero"] == 0
    assert counts["excluded_bit_event_zero_clearance"] == 1
    assert counts["excluded_bit_wait_survival_zero"] == 1
    assert sum(counts["excluded_" + name] for name in CLASSES) == counts["excluded_cars"]


def test_mask_is_invariant_to_batch_composition_and_scored_counts_are_a_subset():
    geo = make_geo()
    for t in range(8):
        for slot in range(4):
            side(geo, t, slot)
    geo["eligible"][0, 4, 0] = False
    geo["has_f"][0, 2, 2] = geo["has_l"][0, 2, 2] = False
    spec = {0: (1, 4, True), 1: (1, 4, True), 2: (1, 4, True), 3: (1, 5, False)}
    res, risk, event, reset = run(geo, spec)
    for keep in ([0], [1, 2], [3, 2, 0]):
        sub_geo = {key: val[..., keep] for key, val in geo.items()}
        _, live = lifecycle_rows(risk[..., keep], event[..., keep], reset[..., keep])
        part = counted_exclusion(sub_geo, live, event[..., keep], reset[..., keep],
                                 target_lane=TARGET)
        assert torch.equal(part["rows"], res["rows"][..., keep])
        assert torch.equal(part["bits"], res["bits"][..., keep])
    scored = torch.zeros_like(res["starts"])
    scored[..., 0] = True
    counts = exclusion_counts(res, scored)
    assert counts["excluded_cars"] == 2 and counts["scored_excluded_cars"] == 1
