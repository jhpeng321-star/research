"""P7 quick-tune: deterministic world-stratified TRAIN subset + epoch fraction.

The CAL-only improvement loop trains member 1 on a fixed, world-stratified
TRAIN subset and a fraction of the epochs with the SAME seed, so the quick
run is reproducible.  All CPU, synthetic windows, no real corpus.
"""
from __future__ import annotations

import pytest

import train.idea_a as ia
from test_idea_a_fakes import make_windows, seeded_bundle


def _items(worlds):
    return [{"world": w, "arm": f"a{i}"} for i, w in enumerate(worlds)]


def test_world_stratified_subset_is_deterministic_and_stratified():
    items = _items(["w1"] * 8 + ["w2"] * 8 + ["w3"] * 4)
    a = ia.world_stratified_subset(items, 0.25, seed=101)
    b = ia.world_stratified_subset(items, 0.25, seed=101)
    assert a == b                                   # same seed -> same subset
    assert a["n_kept"] == 5                         # 2 + 2 + 1
    assert a["per_world"] == {"w1": 2, "w2": 2, "w3": 1}
    assert set(a["per_world"]) == {"w1", "w2", "w3"}  # every world present
    assert len(a["kept"]) == len(set(a["kept"]))
    assert all(0 <= pos < len(items) for pos in a["kept"])


def test_world_stratified_subset_keeps_at_least_one_per_world():
    items = _items(["w1"] * 10 + ["w2"] * 10)
    sel = ia.world_stratified_subset(items, 0.01, seed=7)
    assert sel["per_world"] == {"w1": 1, "w2": 1}
    assert len(sel["kept"]) == 2


def test_world_stratified_subset_different_seed_can_differ():
    items = _items(["w1"] * 20 + ["w2"] * 20)
    a = ia.world_stratified_subset(items, 0.5, seed=1)
    b = ia.world_stratified_subset(items, 0.5, seed=2)
    assert a["per_world"] == b["per_world"]           # same counts
    assert a["kept"] != b["kept"]                     # different items
    full = ia.world_stratified_subset(items, 1.0, seed=1)
    assert full["kept"] == list(range(len(items)))


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
def test_world_stratified_subset_rejects_bad_frac(frac):
    with pytest.raises(ia.IdeaAConfigError):
        ia.world_stratified_subset(_items(["w1"]), frac, seed=0)


def test_quick_tune_config_validation_and_effective_epochs():
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(quick_tune=True, subset_frac=0.0)
    with pytest.raises(ia.IdeaAConfigError):
        ia.IdeaATrainConfig(quick_tune=True, epoch_frac=2.0)
    full = ia.IdeaATrainConfig(epochs=15)
    assert full.effective_epochs() == 15
    quick = ia.IdeaATrainConfig(epochs=15, quick_tune=True, epoch_frac=0.35)
    assert quick.effective_epochs() == 5             # round(0.35 * 15)
    tiny = ia.IdeaATrainConfig(epochs=1, quick_tune=True, epoch_frac=0.1)
    assert tiny.effective_epochs() == 1              # never zero epochs


class _WindowCorpus:
    """A corpus-like stand-in exposing the metadata quick-tune needs."""

    def __init__(self, windows, worlds):
        self._windows = list(windows)
        self._items = _items(worlds)
        self._order = list(range(len(self._windows)))
        self.split = "TRAIN"
        self.pack_dir = None

    def __len__(self):
        return len(self._order)

    def __getitem__(self, i):
        return self._windows[self._order[i]]


def test_quick_tune_fit_uses_world_subset_and_epoch_frac(tmp_path):
    windows = make_windows(4, seed=0)
    corpus = _WindowCorpus(windows, ["w1", "w1", "w2", "w2"])
    cfg = ia.IdeaATrainConfig(epochs=4, steps_per_epoch=1, batch_size=2,
                              quick_tune=True, subset_frac=0.5,
                              epoch_frac=0.25, ensemble_k=1,
                              out_dir=str(tmp_path / "out"))
    tr = ia.IdeaATrainer(cfg, corpus, member=1, seed=101,
                         bundle=seeded_bundle(101))
    receipt = tr.fit()
    assert receipt["quick_tune"] is True
    assert receipt["epochs"] == 1                    # round(0.25 * 4)
    assert receipt["subset"]["n_total"] == 4
    assert receipt["subset"]["n_kept"] == 2          # 1 per world
    assert receipt["subset"]["per_world"] == {"w1": 1, "w2": 1}
    assert (tmp_path / "out" / "member_1.pt").is_file()


def test_quick_tune_refuses_explicit_windows():
    cfg = ia.IdeaATrainConfig(quick_tune=True, out_dir="unused")
    tr = ia.IdeaATrainer(cfg, corpus=None, bundle=seeded_bundle(1))
    with pytest.raises(ia.IdeaAConfigError):
        tr.fit(windows=make_windows(2, seed=0))


def test_quick_tune_cli_dry_run_reports_setting(capsys):
    rc = ia.main(["--member", "1", "--seed", "101", "--corpus", "pack",
                  "--out", "out", "--quick-tune", "--subset-frac", "0.25",
                  "--epoch-frac", "0.35", "--dry-run"])
    assert rc == 0
    import json

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["quick_tune"] is True
    assert payload["subset_frac"] == 0.25
    assert payload["epoch_frac"] == 0.35
