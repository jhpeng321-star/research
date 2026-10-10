"""Cross-piece contract: a train.Trainer checkpoint loads in the adapter.

The acceptance review asks for one small end-to-end check that the TRAIN
checkpoint format, the ADAPTER loader and the BUNDLE expectations agree: train
tiny ensemble members, load the member directory the way the bundle produces it
(``member_<k>.pt`` + ``member_<k>.json`` sidecars), and predict.  CPU-only,
tiny architecture, no SUMO.
"""
from __future__ import annotations

import numpy as np
import torch

from adapter.checkpoint import load_ensemble
from adapter.predictor import WINDOW_KEYS, WmN1Predictor
from evalh.protocol import OutcomeRow
from packsynth import make_runs, write_pack
from train.config import TrainConfig
from train.dataset import MemmapCorpus
from train.trainer import Trainer


def _tiny_config(tmp_path) -> TrainConfig:
    return TrainConfig(d_model=16, n_heads=2, n_layers=1, ffn=32,
                       bias_hidden=8, counter_channels=8, n_mix=3,
                       epochs=1, steps_per_epoch=2, batch_size=2, lr=1e-2,
                       device="cpu", out_dir=str(tmp_path / "members"))


def test_trainer_member_dir_loads_in_adapter_and_predicts(tmp_path):
    torch.set_num_threads(1)
    pack = tmp_path / "pack"
    write_pack(pack, make_runs())
    corpus = MemmapCorpus(pack, pair_sampling=True)
    cfg = _tiny_config(tmp_path)
    out = tmp_path / "members"
    for member in range(1, 6):
        Trainer(cfg, corpus, member=member, seed=cfg.seed_for_member(member),
                device="cpu").fit()

    # exactly the bundle layout: member_<k>.pt + member_<k>.json, no config.json
    assert sorted(p.name for p in out.glob("member_*.pt")) == [
        f"member_{k}.pt" for k in range(1, 6)]
    assert (out / "member_1.json").is_file()
    assert not (out / "config.json").exists()

    ensemble, conf = load_ensemble(out, device="cpu")
    assert len(ensemble.members) == 5
    assert tuple(ensemble.seeds) == tuple(cfg.seeds)
    assert conf["arch"]["d_model"] == 16

    item = corpus[0]
    window = {key: item[key] for key in WINDOW_KEYS}
    predictor = WmN1Predictor.from_checkpoint(
        out, {item["world"]: window}, device="cpu")
    rows = [OutcomeRow(world=item["world"], t0_s=item["t0_s"],
                       decision_t_s=0.0, tau_s=60.0, policy="damping",
                       y_delay_s=1.0)]
    q = predictor.predict_quantiles(rows)
    assert q.shape == (1, 19)
    assert np.all(np.isfinite(q))
    assert np.all(np.diff(q, axis=1) >= -1e-9)
