"""Interface tests: the frozen stub signatures are implemented unchanged."""
from __future__ import annotations

import inspect

from train import cli, config, dataset, trainer


def _params(fn):
    sig = inspect.signature(fn)
    return list(sig.parameters)


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def test_dataset_signatures():
    # amendment 5 (FORMAT.md revision 2) adds the keyword-only format
    # parameter k_max (None = take the pack's K_max); it is appended, so the
    # frozen positional contract is unchanged.
    assert _params(dataset.MemmapCorpus.__init__) == [
        "self", "pack_dir", "split", "pair_sampling", "k_max"]
    assert _default(dataset.MemmapCorpus.__init__, "split") == "TRAIN"
    assert _default(dataset.MemmapCorpus.__init__, "pair_sampling") is False
    assert _default(dataset.MemmapCorpus.__init__, "k_max") is None
    assert _params(dataset.MemmapCorpus.__len__) == ["self"]
    assert _params(dataset.MemmapCorpus.__getitem__) == ["self", "i"]
    assert _params(dataset.MemmapCorpus.sibling_pairs) == ["self", "i"]
    assert _params(dataset.MemmapCorpus.worlds) == ["self"]


def test_trainer_signatures():
    assert _params(trainer.Trainer.__init__) == [
        "self", "config", "corpus", "member", "seed", "device"]
    assert _default(trainer.Trainer.__init__, "device") is None
    assert _params(trainer.Trainer.fit) == ["self"]
    assert _params(trainer.Trainer.save_checkpoint) == ["self", "path"]
    assert _params(trainer.Trainer.resume) == ["self", "path"]
    assert _params(trainer.Trainer.evaluate_cal) == ["self"]


def test_cli_and_config_signatures():
    assert _params(cli.main) == ["argv"]
    assert _params(cli.record_seed) == ["out_dir", "member", "seed"]
    assert _params(config.TrainConfig.load) == ["path"]
    assert _params(config.TrainConfig.save) == ["self", "path"]
    assert _params(config.TrainConfig.to_dict) == ["self"]


def test_importable_entry_points():
    # The package import contract from INTERFACES.md: the CLI parser exists and
    # `python -m train` is wired through __main__.
    assert cli.build_parser().prog == "train"
