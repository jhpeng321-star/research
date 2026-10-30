import hashlib
import subprocess
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.errsrc.inference import LearnedMix
from wms.model.entries import EntryConfig, EntranceHistory, TorchEntryProcessV2
from wms.model.learned_admission import LearnedAdmission


def state():
    return dict(s_m=torch.zeros(1, 4), v_mps=torch.zeros(1, 4), a_mps2=torch.zeros(1, 4),
        lane=torch.zeros(1, 4, dtype=torch.long), length_m=torch.ones(1, 4) * 5,
        present=torch.zeros(1, 4, dtype=torch.bool), is_cav=torch.zeros(1, 4, dtype=torch.bool))


def inferred():
    return {n: torch.tensor([[v, v * 1.1, v, v]]) for n, v in
            (("v0", 17.), ("time_headway", 1.7), ("decel", 2.3), ("min_gap", 2.1))}


def cav_pool_row(v0=21.0):
    return torch.tensor([[v0, 2.0, 3.0, 1.5]])  # names order: v0, time_headway, decel, min_gap


def ramp_hist(cav_share=0.0):
    hist = EntranceHistory.defaults()
    hist.cav_share = np.array([0.0, float(cav_share)])
    return hist


def learned_process(mix, seed=17, cav_share=0.0, **cfg):
    full = dict(conserve_queue=True, learned_admission=LearnedAdmission(mix),
                insert_rule="sumo_like_v2")
    full.update(cfg)
    return TorchEntryProcessV2(history=ramp_hist(cav_share), cfg=EntryConfig(**full),
                               row_seeds=[seed], couple_draws=True)


def test_admission_candidate_equals_subsequent_learned_birth_assignment():
    params = inferred()
    mix = LearnedMix(params, torch.tensor([[[True], [True], [False], [False]]]), 17, ["world"],
                     cav_pools=[cav_pool_row()])
    admission = LearnedAdmission(mix)
    candidate = admission.candidate(0, 0, 0)
    s = state()
    s["present"][0, 0] = True
    newborn = s["present"].clone()
    sampled = mix(s, newborn, params)
    for name, value in candidate.items():
        assert sampled[name][0, 0].item() == value
    # Retry of the next identity keeps its own draw, independent of service attempts.
    assert admission.candidate(0, 0, 1) == admission.candidate(0, 0, 1)


def test_actual_admission_uses_inferred_v0_not_class_speed_and_conserves():
    params = inferred()
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 17, ["world"],
                     cav_pools=[cav_pool_row()])
    admission = LearnedAdmission(mix)
    cfg = EntryConfig(conserve_queue=True, learned_admission=admission, insert_rule="sumo_like_v2")
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(), cfg=cfg,
                                  row_seeds=[17], couple_draws=True)
    process._emit_due = lambda b: [0]
    s = state()
    expected = admission.candidate(0, 0, 0)
    born = process.insert(s)
    assert born.sum() == 1
    slot = int(born[0].nonzero()[0])
    assert s["v_mps"][0, slot] == expected["v0"]
    ledger = process._source_ledgers[0, 0]
    assert ledger.due == 1 and ledger.inserted == 1 and ledger.queue == 0


def test_invalid_bundle_or_nonphysical_pool_rejects():
    with pytest.raises(ValueError, match="exact conservation"):
        EntryConfig(learned_admission=object())
    params = inferred()
    params["decel"].fill_(float("nan"))
    admission = LearnedAdmission(LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 17,
                                            ["world"], cav_pools=[cav_pool_row()]))
    with pytest.raises(ValueError, match="nonphysical"):
        admission.candidate(0, 0, 0)
    empty = LearnedAdmission(LearnedMix(inferred(), torch.ones(1, 4, 1, dtype=torch.bool), 17,
                                        ["world"], cav_pools=[torch.zeros(0, 4)]))
    with pytest.raises(ValueError, match="CAV birth without"):
        empty.candidate(0, 0, 0, cav=True)


def test_scorer_rejects_learned_admission_without_matching_mix_before_rollout(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "tools/wmrequal/score.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("wmfix_admission_score_test", path)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    with pytest.raises(ValueError, match="mix ON only"):
        score.score_batch(None, None, None, None, None, None, None, None, False, learned_admission=True)


def test_reviewer_multibirth_cav_then_hdv_recycled_slot_keeps_admission_identity():
    """WMFIX_REVIEW.md claim 3: a CAV candidate takes the lowest *virgin* slot while
    the next HDV candidate takes the recycled slot, so newborn slot order differs
    from admission order.  The entrant admitted on ordinal k's settings must be the
    entrant that later moves with ordinal k's settings."""
    names = ("v0", "time_headway", "decel", "min_gap")
    rows = ((17.0, 1.70, 2.30, 2.10), (18.7, 1.87, 2.53, 2.31), (20.4, 2.04, 2.76, 2.51))
    params = {n: torch.tensor([[r[i] for r in rows] + [rows[0][i]]]) for i, n in enumerate(names)}
    observed = torch.tensor([[[True], [True], [True], [False]]])
    share, mix_seed, key = 0.5, 0, "world"

    def u_cav(seed, ordinal):
        return float(np.random.default_rng([seed, 1, 1, ordinal]).random())

    seed = next(s for s in range(4096) if u_cav(s, 0) < share <= u_cav(s, 1))
    mix = LearnedMix(params, observed, mix_seed, [key], cav_pools=[cav_pool_row()])
    hist = EntranceHistory.defaults()
    hist.cav_share = np.array([0.0, share])
    cfg = EntryConfig(conserve_queue=True, learned_admission=LearnedAdmission(mix),
                      insert_rule="sumo_like_v2")
    process = TorchEntryProcessV2(history=hist, cfg=cfg, row_seeds=[seed], couple_draws=True)
    s = state()
    s["present"][0, 0] = True       # t0 occupant: slot 0 can never be virgin
    process._observe(s)             # freeze virgin slots from the t0 snapshot
    s["present"][0, 0] = False      # occupant left: slot 0 is recycled-free
    process._emit_due = lambda b: [1, 1]
    newborn = process.insert(s)
    assert int(newborn.sum()) == 2
    cav_slot = int((newborn[0] & s["is_cav"][0]).nonzero()[0])
    hdv_slot = int((newborn[0] & ~s["is_cav"][0]).nonzero()[0])
    assert (cav_slot, hdv_slot) == (1, 0)          # slot order != admission order
    admission = LearnedAdmission(mix)
    cav0, hdv1 = admission.candidate(0, 1, 0, cav=True), admission.candidate(0, 1, 1)
    assert cav0["v0"] != hdv1["v0"]
    # admission evidence: the empty-lane CAV departs capped by its own candidate v0
    assert s["v_mps"][0, cav_slot].item() == pytest.approx(min(cav0["v0"], 22.22))
    # the HDV was admitted (safe speed ~3 m/s) on hdv1's settings, not cav0's
    assert 0.0 < s["v_mps"][0, hdv_slot].item() < cav0["v0"]
    sampled = mix(s, newborn, params, getattr(process, "last_birth_ordinals", None))
    for name in names:
        assert sampled[name][0, cav_slot].item() == cav0[name], name
        assert sampled[name][0, hdv_slot].item() == hdv1[name], name
    ledger = process._source_ledgers[0, 1]
    assert (ledger.due, ledger.inserted, ledger.queue) == (2, 2, 0)


def test_multibirth_both_entrances_same_step_keeps_per_entrance_ordinals():
    params = inferred()
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 0, ["world"],
                     cav_pools=[cav_pool_row()])
    admission = LearnedAdmission(mix)
    process = learned_process(mix, seed=23)
    process._emit_due = lambda b: [0, 1]
    s = state()
    newborn = process.insert(s)
    assert int(newborn.sum()) == 2
    ramp = int((newborn[0] & (s["lane"][0] == 2)).nonzero()[0])
    main = int((newborn[0] & (s["lane"][0] != 2)).nonzero()[0])
    main0, ramp0 = admission.candidate(0, 0, 0), admission.candidate(0, 1, 0)
    assert main0["v0"] != ramp0["v0"]  # the test must be able to see a swap
    sampled = mix(s, newborn, params, process.last_birth_ordinals)
    for name in ("v0", "time_headway", "decel", "min_gap"):
        assert sampled[name][0, main].item() == main0[name], name
        assert sampled[name][0, ramp].item() == ramp0[name], name


def test_exhaustion_keeps_backlog_and_identity_ordinal_for_later_birth():
    params = inferred()
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 5, ["world"],
                     cav_pools=[cav_pool_row()])
    admission = LearnedAdmission(mix)
    process = learned_process(mix, seed=29)
    s = state()
    s["present"][:] = True           # no free slot at all: every placement fails
    process._observe(s)
    process._emit_due = lambda b: [1]
    newborn = process.insert(s)
    assert int(newborn.sum()) == 0 and process.last_birth_ordinals == {}
    ledger = process._source_ledgers[0, 1]
    assert (ledger.due, ledger.inserted, ledger.queue) == (1, 0, 1)
    s["present"][0, 0] = False       # a slot frees: the backlog is admitted with ordinal 0
    process._emit_due = lambda b: []
    newborn = process.insert(s)
    assert int(newborn.sum()) == 1
    slot = int(newborn[0].nonzero()[0])
    sampled = mix(s, newborn, params, process.last_birth_ordinals)
    for name in ("v0", "time_headway", "decel", "min_gap"):
        assert sampled[name][0, slot].item() == admission.candidate(0, 1, 0)[name], name


def test_rejected_candidate_retries_on_the_same_identity_draw():
    params = inferred()
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 5, ["world"],
                     cav_pools=[cav_pool_row()])
    admission = LearnedAdmission(mix)
    process = learned_process(mix, seed=31)
    s = state()
    front = float(process.entry_s[2]) + 5.0 + 0.1
    s["present"][0, 3], s["lane"][0, 3] = True, 2   # stopped blocker 1 m past the front
    s["s_m"][0, 3], s["v_mps"][0, 3] = front + 1.0 + 5.0, 0.0
    process._emit_due = lambda b: [1]
    assert int(process.insert(s).sum()) == 0        # gap 1 m < minGap: deferred, not consumed
    assert process._source_ledgers[0, 1].queue == 1
    s["present"][0, 3] = False                      # lane clears; a mainline arrival joins
    process._emit_due = lambda b: [0]
    newborn = process.insert(s)
    assert int(newborn.sum()) == 2
    ramp = int((newborn[0] & (s["lane"][0] == 2)).nonzero()[0])
    main = int((newborn[0] & (s["lane"][0] != 2)).nonzero()[0])
    sampled = mix(s, newborn, params, process.last_birth_ordinals)
    retry, main0 = admission.candidate(0, 1, 0), admission.candidate(0, 0, 0)
    for name in ("v0", "time_headway", "decel", "min_gap"):
        assert sampled[name][0, ramp].item() == retry[name], name
        assert sampled[name][0, main].item() == main0[name], name


def test_min_speed_rejection_restored_for_learned_admission():
    from wms.model.entries import sumo_safe_insert_speed
    params = inferred()
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 5, ["world"],
                     cav_pools=[cav_pool_row()])
    admission = LearnedAdmission(mix)
    process = learned_process(mix, seed=37, sumo_min_speed_mps=5.0)
    s = state()
    front = float(process.entry_s[2]) + 5.0 + 0.1
    gap = 6.0
    s["present"][0, 3], s["lane"][0, 3] = True, 2
    s["s_m"][0, 3], s["v_mps"][0, 3] = front + gap + 5.0, 0.0
    safe = sumo_safe_insert_speed(gap, 0.0, 1.7, 2.3, 2.1)
    assert 0.0 < safe < 5.0                          # the geometry separates the gates
    process._emit_due = lambda b: [1]
    assert int(process.insert(s).sum()) == 0         # gap passes, safe speed is refused
    assert process._source_ledgers[0, 1].queue == 1
    process.cfg.sumo_min_speed_mps = 0.0             # incumbent gate off: same geometry admits
    newborn = process.insert(s)
    assert int(newborn.sum()) == 1
    slot = int(newborn[0].nonzero()[0])
    assert s["v_mps"][0, slot].item() == pytest.approx(safe)


def test_cav_entrants_draw_from_their_own_settings_pool():
    params = inferred()
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 5, ["world"],
                     cav_pools=[cav_pool_row(v0=21.0)])
    admission = LearnedAdmission(mix)
    process = learned_process(mix, seed=41, cav_share=1.0)
    process._emit_due = lambda b: [1]
    s = state()
    newborn = process.insert(s)
    assert int(newborn.sum()) == 1
    slot = int(newborn[0].nonzero()[0])
    assert bool(s["is_cav"][0, slot])
    cav0 = admission.candidate(0, 1, 0, cav=True)
    assert cav0["v0"] == 21.0 and admission.candidate(0, 1, 0)["v0"] != 21.0
    assert s["v_mps"][0, slot].item() == pytest.approx(min(21.0, 22.22))
    sampled = mix(s, newborn, params, process.last_birth_ordinals)
    for name in ("v0", "time_headway", "decel", "min_gap"):
        assert sampled[name][0, slot].item() == cav0[name], name


def test_flag_off_actual_insert_and_rng_match_committed_entries():
    text = subprocess.check_output(["git", "show", "fc69fcfd:src/wms/model/entries.py"],
                                   cwd=Path(__file__).resolve().parents[2], text=True)
    legacy = types.ModuleType("wms.model._wmfix_legacy_entries")
    legacy.__package__ = "wms.model"
    sys.modules[legacy.__name__] = legacy
    exec(compile(text, "<committed WM2 entries>", "exec"), legacy.__dict__)
    old = legacy.TorchEntryProcessV2(history=legacy.EntranceHistory.defaults(), seed=29,
                                   row_seeds=[17], couple_draws=True)
    new = TorchEntryProcessV2(history=EntranceHistory.defaults(), seed=29, row_seeds=[17], couple_draws=True)
    old._emit_due = new._emit_due = lambda b: [0, 1]
    a, b = state(), state()
    for _ in range(3):
        assert torch.equal(old.insert(a), new.insert(b))
        for key in a:
            assert torch.equal(a[key], b[key]), key
        assert np.array_equal(old.pending, new.pending)
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
