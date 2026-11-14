"""MemmapCorpus tests: guards, pair sampler, outcome mapping, shapes."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model import constants as C
from packsynth import make_runs, write_pack
from train.dataset import (OUTCOME_COLUMN_MAP, CorpusRefusal, MemmapCorpus,
                           batch_has_pair, contrast_batch_indices,
                           contrast_pairs, ensure_pair, is_reference,
                           map_outcome_targets)


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack"
    write_pack(root, make_runs())
    return root


def test_train_selection(pack):
    corpus = MemmapCorpus(pack)
    assert len(corpus) == 5                       # w1 x3 arms + w2 x2 arms
    assert corpus.worlds() == ["w1", "w2"]
    assert all(corpus[i]["split"] == "TRAIN" for i in range(len(corpus)))


def test_cal_requires_explicit_split(pack):
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack, split="HOLDOUT")
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack, split="TEST")
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(pack, split="whatever")
    cal = MemmapCorpus(pack, split="CALIBRATION")
    assert len(cal) == 1 and cal.worlds() == ["c1"]
    alias = MemmapCorpus(pack, split="CAL")
    assert len(alias) == 1


def test_chunk5_pack_is_refused(tmp_path):
    runs = make_runs()
    runs.append(dict(chunk="chunk5", world="w9", arm="native",
                     policy="native", split="TEST", t0_s=100.0))
    root = tmp_path / "pack5"
    write_pack(root, runs)
    with pytest.raises(CorpusRefusal):
        MemmapCorpus(root)


def test_item_tensors_and_targets(pack):
    corpus = MemmapCorpus(pack)
    item = corpus[0]
    k = 16
    assert item["veh_step"].shape == (k, C.VEH_F)
    assert item["veh_mask"].dtype == torch.bool
    assert item["plan"].shape == (k, C.N_SEGMENTS, C.PLAN_F)
    assert item["wall"].shape == (C.MAX_WALLS, C.WALL_F)
    assert item["counter"].shape == (C.N_ENTRANCES, C.COUNTER_BINS)
    assert item["accel"].shape == (k,)
    assert item["arrivals"].shape == (C.N_ENTRANCES,)
    assert item["outcome_quantiles"].shape == (C.N_TAU, C.N_OUTCOME_TARGETS)
    assert item["outcome_mask"].shape == (C.N_TAU,)
    assert item["collision"].shape == (C.N_TAU,)
    assert item["veh_step"].dtype == torch.float32


def test_outcome_column_map_is_exact():
    assert [OUTCOME_COLUMN_MAP[t] for t in C.OUTCOME_TARGETS] == [
        "total_delay_ref_s", "ttc_lt3_veh_s", "hard_brake_s"]


def test_outcome_target_mapping_and_mask():
    rows = [
        {"tau": 30.0, "total_delay_ref_s": 7.0, "ttc_lt3_veh_s": 1.5,
         "hard_brake_s": 0.25, "collision_any": 1.0, "tau_available": True},
        {"tau": 60.0, "total_delay_ref_s": 9.0, "ttc_lt3_veh_s": 2.5,
         "hard_brake_s": 0.5, "collision_any": 0.0, "tau_available": False},
        {"tau": 570.0, "total_delay_ref_s": -1.0, "ttc_lt3_veh_s": float("nan"),
         "hard_brake_s": 0.0, "collision_any": 0.0, "tau_available": True},
    ]
    targets, mask, collision = map_outcome_targets(rows)
    assert targets.shape == (C.N_TAU, C.N_OUTCOME_TARGETS)
    assert targets[0].tolist() == [7.0, 1.5, 0.25]
    assert mask.tolist() == [True, False, False, False, True]
    assert collision[0] == 1.0
    assert targets[4].tolist() == [-1.0, 0.0, 0.0]      # NaN -> 0
    assert np.all(targets[1] == np.array([9.0, 2.5, 0.5], dtype=np.float32))


def _decision_time_rows():
    """Two decision times in one item cell; tau 30 exists at both, tau 60 only
    at t0, tau 120 only at the later decision time."""
    def row(t, tau, delay):
        return {"t": t, "tau": tau, "total_delay_ref_s": delay,
                "ttc_lt3_veh_s": 0.0, "hard_brake_s": 0.0,
                "collision_any": 0.0, "tau_available": True}

    return [
        row(0.0, 30.0, 1.0),
        row(600.0, 30.0, 9.0),
        row(0.0, 60.0, 2.0),
        row(600.0, 120.0, 8.0),
    ]


def test_outcome_target_mapping_uses_item_t0_row():
    targets, mask, _ = map_outcome_targets(_decision_time_rows(), t0_s=0.0)
    assert targets[0, 0] == 1.0                 # t0 row, not the t=600 row
    assert targets[1, 0] == 2.0
    assert mask.tolist() == [True, True, False, False, False]


def test_outcome_target_mapping_missing_t0_row_masks_false():
    rows = [{"t": 600.0, "tau": 120.0, "total_delay_ref_s": 8.0,
             "ttc_lt3_veh_s": 0.0, "hard_brake_s": 0.0, "collision_any": 0.0,
             "tau_available": True}]
    targets, mask, _ = map_outcome_targets(rows, t0_s=0.0)
    assert not mask.any()
    assert np.all(targets == 0.0)


def test_outcome_target_mapping_duplicate_t0_row_fails_closed():
    rows = [
        {"t": 0.0, "tau": 30.0, "total_delay_ref_s": 1.0, "ttc_lt3_veh_s": 0.0,
         "hard_brake_s": 0.0, "collision_any": 0.0, "tau_available": True},
        {"t": 0.0, "tau": 30.0, "total_delay_ref_s": 5.0, "ttc_lt3_veh_s": 0.0,
         "hard_brake_s": 0.0, "collision_any": 0.0, "tau_available": True},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        map_outcome_targets(rows, t0_s=0.0)


def test_outcome_target_mapping_none_t0_keeps_legacy_last_row():
    targets, _, _ = map_outcome_targets(_decision_time_rows())
    assert targets[0, 0] == 9.0                 # legacy last-row-wins
    assert targets[1, 0] == 2.0



def test_sibling_pairs_are_same_world_t0_treatment_then_reference(pack):
    corpus = MemmapCorpus(pack)
    seen = 0
    for pos in range(len(corpus)):
        item = corpus[pos]
        if is_reference(item):
            continue
        a, b = corpus.sibling_pairs(pos)
        ia, ib = corpus[a], corpus[b]
        assert ia["pair_key"] == ib["pair_key"] == item["pair_key"]
        assert ia["arm"] == item["arm"]
        assert ib["policy"] == "native"
        assert ia["arm"] != ib["arm"]
        assert ia["t0_s"] == ib["t0_s"]
        seen += 1
    assert seen == 3                              # base, pi (w1) + base (w2)


def test_sibling_pairs_from_reference_still_treatment_first(pack):
    corpus = MemmapCorpus(pack)
    native_pos = next(i for i in range(len(corpus))
                      if is_reference(corpus[i]))
    a, b = corpus.sibling_pairs(native_pos)
    assert corpus[b]["policy"] == "native"
    assert not is_reference(corpus[a])
    assert corpus[a]["pair_key"] == corpus[b]["pair_key"]


def test_contrast_batch_indices(pack):
    corpus = MemmapCorpus(pack)
    items = [corpus[i] for i in range(len(corpus))]
    idx = contrast_batch_indices(items)
    assert idx["a"].numel() == idx["b"].numel() == 3
    for a, b in zip(idx["a"].tolist(), idx["b"].tolist()):
        assert not is_reference(items[a])
        assert is_reference(items[b])
        assert items[a]["pair_key"] == items[b]["pair_key"]
        assert items[a]["arm"] != items[b]["arm"]


def test_pair_sampling_orders_pairs_contiguously(pack):
    corpus = MemmapCorpus(pack, pair_sampling=True)
    groups = {}
    for pos in range(len(corpus)):
        groups.setdefault(corpus[pos]["pair_key"], []).append(pos)
    for positions in groups.values():
        assert positions == list(range(min(positions), min(positions)
                                       + len(positions)))
    # every treatment is immediately followed by a reference within its group
    for positions in groups.values():
        arms = [corpus[p]["policy"] for p in positions]
        if "native" in arms:
            assert positions[-1] == max(positions)
            assert corpus[positions[-1]]["policy"] == "native"


# ------------------------------------ pair-aware batch sampler (real path) --
def _pair_items():
    """Hand-built item metadata covering true-native, switching-native, and a
    ``native_base`` reference group."""
    def it(world, arm, policy):
        return dict(world=world, arm=arm, policy=policy, chunk="chunk4",
                    split="TRAIN", t0_s=100.0,
                    pair_key=f"{world}@t0=100")

    return [
        # group a: a TRUE native reference plus switching/baseline treatments
        it("a", "native", "native"),
        it("a", "switching1", "switching"),
        it("a", "base", "damping"),
        # group b: arm named 'native' but policy switching -> no reference
        it("b", "native", "switching"),
        it("b", "switching2", "switching"),
        # group c: native_base is a policy-native reference
        it("c", "native_base", "native"),
        it("c", "pi", "pi"),
    ]


def test_is_reference_is_policy_driven_and_excludes_switching_native():
    items = _pair_items()
    assert not is_reference(items[3])            # 'native' arm, switching policy
    assert is_reference(items[0]) and is_reference(items[5])


def test_contrast_pairs_exclude_switching_native_reference():
    items = _pair_items()
    pairs = contrast_pairs(items)
    # group b contributes NO pair; a and c do (2 + 1 treatments)
    assert len(pairs) == 3
    assert not any(items[r]["pair_key"].startswith("b@") for _, r in pairs)
    for t, r in pairs:
        assert is_reference(items[r])
        assert not is_reference(items[t])
        assert items[t]["pair_key"] == items[r]["pair_key"]
    # group a's treatments each pair with the single true native reference
    a_refs = {r for t, r in pairs if items[t]["pair_key"].startswith("a@")}
    assert a_refs == {0}


def test_ensure_pair_makes_every_batch_complete_and_deterministic():
    import random

    items = _pair_items()
    pairs = contrast_pairs(items)
    n = len(items)
    gen = random.Random(7)
    for _ in range(200):
        order = list(range(n))
        gen.shuffle(order)
        for start in range(0, n, 3):
            idx = order[start:start + 3]
            out, _cursor = ensure_pair(idx, items, pairs, 0)
            assert batch_has_pair(out, items), (idx, out)
    # same input + cursor -> identical output (seeded determinism)
    assert ensure_pair([3], items, pairs, 0) == ensure_pair([3], items, pairs, 0)
    # a batch already holding a complete pair is returned unchanged
    assert ensure_pair([0, 1], items, pairs, 0) == ([0, 1], 0)


# --------------------------------------------- replicated-arm pack keys ----
def _replicated_runs():
    """The chunk-1/2/3 (fix27, truearms) replicated-arm shape: one native
    reference plus __fixed1/__fixed2/__switching1..3 treatment replicates;
    the run-dir suffix (not the policy label) discriminates the items."""
    runs = [dict(chunk="chunk1", world="wrep", arm="native", policy="native",
                 split="TRAIN", t0_s=480.0)]
    for arm in ("fixed1", "fixed2", "switching1", "switching2", "switching3"):
        runs.append(dict(
            chunk="chunk1", world="wrep", arm=arm,
            policy="fixed" if arm.startswith("fixed") else "switching",
            split="TRAIN", t0_s=480.0))
    return runs


def test_replicated_arms_are_distinct_items_with_one_reference(tmp_path):
    root = tmp_path / "pack"
    write_pack(root, _replicated_runs())
    corpus = MemmapCorpus(root)
    assert len(corpus) == 6
    identities = [(corpus[i]["world"], corpus[i]["arm"])
                  for i in range(len(corpus))]
    assert len(set(identities)) == 6          # no policy-keyed collapse
    assert corpus.pair_keys() == ["wrep@t0=480"]
    policies = sorted(corpus[i]["policy"] for i in range(len(corpus)))
    assert policies == ["fixed", "fixed", "native",
                        "switching", "switching", "switching"]


def test_replicated_arms_pair_each_treatment_with_the_reference(tmp_path):
    root = tmp_path / "pack"
    write_pack(root, _replicated_runs())
    corpus = MemmapCorpus(root, pair_sampling=True)
    refs = set()
    treatments = 0
    for pos in range(len(corpus)):
        item = corpus[pos]
        if is_reference(item):
            continue
        a, b = corpus.sibling_pairs(pos)
        assert corpus[a]["arm"] == item["arm"]
        assert corpus[b]["policy"] == "native"
        assert corpus[a]["pair_key"] == corpus[b]["pair_key"]
        refs.add(corpus[b]["arm"])
        treatments += 1
    assert treatments == 5                    # fixed1/2 + switching1/2/3
    assert refs == {"native"}                 # a single shared reference

    items = [corpus[i] for i in range(len(corpus))]
    idx = contrast_batch_indices(items)
    assert idx["a"].numel() == idx["b"].numel() == 5
    for a, b in zip(idx["a"].tolist(), idx["b"].tolist()):
        assert not is_reference(items[a]) and is_reference(items[b])
        assert items[a]["pair_key"] == items[b]["pair_key"]
