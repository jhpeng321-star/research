import subprocess
import sys
import types
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


def test_admission_candidate_equals_subsequent_learned_birth_assignment():
    params = inferred()
    mix = LearnedMix(params, torch.tensor([[[True], [True], [False], [False]]]), 17, ["world"])
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
    mix = LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 17, ["world"])
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
    admission = LearnedAdmission(LearnedMix(params, torch.ones(1, 4, 1, dtype=torch.bool), 17, ["world"]))
    with pytest.raises(ValueError, match="nonphysical"):
        admission.candidate(0, 0, 0)


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
