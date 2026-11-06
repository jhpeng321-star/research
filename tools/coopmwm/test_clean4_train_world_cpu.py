"""N4 (N11 F1/F7): map-aware TRAIN-world predicate on the B/R bank path; no raw merge literals left.

Real NEWMAP_MOVES_B rows (when present in the run dir) must normalise to registered wz3to2 TRAIN
worlds; held-out/forbidden/unregistered worlds are refused; merge-map semantics are unchanged.
"""
import json
from pathlib import Path
import re

import pytest

from tools.coopmwm import map_descriptor as md, map_runtime as mr

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO/'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'
RUN = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
# Modules on the clean4 B/R bank / practice / G1 / monitor path that formerly inlined the literal.
BANK_PATH = ('group_lane_bank.py', 'practice_weighting.py', 'executable_labels.py', 'launch_augmentation.py',
             'wall_negatives.py', 'wall_tune.py', 'track2_monitor.py', 'bank_breadth.py', 'policy_synergy.py',
             'replay_bank.py', 'launch_preparation.py')
LITERAL = re.compile(r'986000\s*<=?\s*[\w\[\]\'\".()]+\s*<=?\s*986071')


@pytest.fixture
def wz(monkeypatch):
    md._reset_for_tests()
    monkeypatch.setenv(md.ENV, str(SPEC))
    yield
    md._reset_for_tests()


@pytest.fixture
def original(monkeypatch):
    md._reset_for_tests()
    monkeypatch.delenv(md.ENV, raising=False)
    yield
    md._reset_for_tests()


def test_no_raw_merge_train_literal_on_bank_path():
    for name in BANK_PATH:
        text = (REPO/'tools/coopmwm'/name).read_text(encoding='utf-8')
        code = '\n'.join(line.split('#', 1)[0] for line in text.splitlines())
        assert not LITERAL.search(code), name


def test_merge_semantics_unchanged(original):
    assert mr.registered_train_world(dict(seed=986000, demand=1900, penetration=0.05))
    assert mr.registered_train_world(dict(seed=986071))
    for seed in (985999, 986072, 989300, 989650, 964100):
        assert not mr.registered_train_world(dict(seed=seed))
    assert mr.primary_train_seed(986040) and not mr.primary_train_seed(989300)
    # a map world is refused while the process map is the original map (fail closed)
    assert not mr.registered_train_world(dict(map='wz3to2', seed=989300, demand=3200, penetration=0.6))


def test_map_rows_normalise_and_guard(wz):
    probe_row = dict(map='wz3to2', cell='D2400_p60', demand=2400, penetration=0.6, seed=989200, repeat=0,
                     split='TRAIN 989200-989599')
    assert mr.registered_train_world(probe_row)
    w = mr.normalised_world(probe_row)
    assert w['cell_key'] == 'D2400_p60' and 'split' not in w
    assert mr.registered_train_world(dict(map='wz3to2', seed=989599, cell='D3200_p60'))
    for bad in (dict(map='wz3to2', seed=989600, demand=3200, penetration=0.6),      # held-out
                dict(map='wz3to2', seed=989699, cell='D3200_p60'),
                dict(map='wz3to2', seed=964100, demand=3200, penetration=0.6),      # forbidden
                dict(map='wz3to2', seed=989300, demand=3000, penetration=0.6),      # unregistered cell
                dict(map='wz2to1', seed=989300, demand=3200, penetration=0.6),
                dict(map='wz3to2', seed='989300', demand=3200, penetration=0.6)):
        assert not mr.registered_train_world(bad), bad
    assert mr.primary_train_seed(989200) and mr.primary_train_seed(989599)
    assert not mr.primary_train_seed(989600) and not mr.primary_train_seed(964000)
    assert mr.primary_train_seed(986001)  # merge seeds keep their meaning


@pytest.mark.skipif(not (RUN/'NEWMAP_MOVES_B.jsonl').is_file(), reason='laptop run dir only')
def test_real_newmap_rows_all_accepted(wz):
    from tools.coopmwm import group_lane_bank
    rows = [json.loads(line) for line in (RUN/'NEWMAP_MOVES_B.jsonl').read_text(encoding='utf-8').splitlines()
            if line.strip()]
    assert len(rows) >= 27
    for row in rows:
        assert mr.registered_train_world(row['world']), row['world']
        # binding() validates every other field too; world must no longer be the refusal reason
        try:
            group_lane_bank.binding(row)
        except ValueError as error:
            assert 'TRAIN world' not in str(error), error
    for row in rows[:3]:
        held = dict(row, world=dict(row['world'], seed=989650))
        with pytest.raises(ValueError):
            group_lane_bank.binding(held)
