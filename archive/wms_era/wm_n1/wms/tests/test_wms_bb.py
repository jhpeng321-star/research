from __future__ import annotations

import numpy as np
import torch

from output.wd01.wm_n1.hdv.config import HdvConfig
from output.wd01.wm_n1.wms.model import (CONTROLLER_VOCAB, PhysicsConfig,
                                         WmSModel, controller_index)
from output.wd01.wm_n1.wms.train import (TrajectoryDataset, WmSTrainConfig,
                                         WmSTrainer, collate)


def small_model(**kwargs):
    return WmSModel(
        physics=PhysicsConfig(max_gap_steps=8),
        hdv=HdvConfig(d_model=8, n_heads=2, n_layers=1, ffn=16),
        **kwargs,
    )


def state(batch=2, slots=4, cav_slot=0):
    lane = torch.zeros(batch, slots, dtype=torch.long)
    lane[:, slots // 2:] = 1
    one_hot = torch.nn.functional.one_hot(lane, 3).float()
    kin = torch.cat((torch.arange(slots).float().view(1, slots, 1).expand(batch, -1, -1),
                     one_hot, torch.full((batch, slots, 1), 12.0),
                     torch.zeros(batch, slots, 2)), dim=-1)
    is_cav = torch.zeros(batch, slots, dtype=torch.bool)
    is_cav[:, cav_slot] = True
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": torch.ones(batch, slots, dtype=torch.bool),
            "is_cav": is_cav}


def test_controller_index_covers_vocab_and_aliases():
    assert CONTROLLER_VOCAB == ("none", "native", "pi", "damping", "wall", "other", "direct")
    assert controller_index("native") == 1
    assert controller_index("pi") == 2
    assert controller_index("fleet-pi") == 2
    assert controller_index("fleet-pi-external-u") == 2
    assert controller_index("damping") == 3
    assert controller_index("direct") == CONTROLLER_VOCAB.index("direct")
    assert controller_index("wall") == 4
    assert controller_index("trajectory") == CONTROLLER_VOCAB.index("other")
    assert controller_index(None) == CONTROLLER_VOCAB.index("other")


def test_structured_model_has_no_controller_embedding_and_excludes_cavs():
    import pytest
    model = small_model()
    assert model.controller_embed is None
    out = model(state(), k_steps=1)
    assert not bool(out["hdv_mask"][:, 0].any())
    assert bool(out["hdv_mask"][:, 1].all())
    with pytest.raises(ValueError):
        model.step(state())


def test_black_box_predicts_cavs_with_controller_id_and_ignores_teacher():
    model = small_model(black_box=True, controller_embed_dim=4)
    assert model.controller_embed is not None
    out = model(state(), k_steps=1, controller_id=torch.full((2,), 2))
    # every present slot is a learned mover, CAV included
    assert bool(out["hdv_mask"].all())
    # a CAV teacher is never consulted in black-box mode
    teacher = torch.full((2, 4, 4), 999.0)
    stepped = model.step(state(), k_steps=1, cav_teacher=teacher,
                         controller_id=torch.full((2,), 2))
    assert not torch.allclose(stepped["next_s_m"][:, 0], teacher[:, 0, 0])


def test_black_box_embedding_receives_gradient():
    model = small_model(black_box=True, controller_embed_dim=4)
    model.train()
    out = model(state(), k_steps=1, controller_id=torch.full((2,), 3))
    (out["accel_mps2"].sum() + out["lane_logits"].sum()).backward()
    assert model.controller_embed.weight.grad is not None
    assert bool(model.controller_embed.weight.grad.abs().sum() > 0)


def test_checkpoint_residual_keeps_gradients_finite():
    model = small_model(black_box=True, checkpoint_residual=True)
    cfg = WmSTrainConfig(device="cpu", threads=4, steps_per_epoch=1,
                         unroll_curriculum_s=(10.0,), batch_size=2,
                         black_box=True, checkpoint_residual=True,
                         controller_embed_dim=4)
    n, t1, slots = 2, 21, 4
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    present = np.ones((n, t1, slots), dtype=bool)
    is_cav = np.zeros((n, t1, slots), dtype=bool)
    is_cav[:, :, 0] = True
    dataset = TrajectoryDataset.from_arrays(
        {"states": states, "present": present, "is_cav": is_cav})
    batch = collate([dataset[0], dataset[1]])
    rec = WmSTrainer(model, cfg)._forward_backward(batch, unroll_steps=20)
    assert np.isfinite(rec["loss"])


def test_black_box_loss_supervises_cavs_structured_does_not():
    n, t1, slots = 2, 21, 4
    base = state(batch=n, slots=slots)["veh_kin"].numpy()
    states = np.repeat(base[:, None], t1, axis=1)
    states[..., 0] += np.arange(t1, dtype=np.float32)[None, :, None] * 6.0
    present = np.ones((n, t1, slots), dtype=bool)
    is_cav = np.zeros((n, t1, slots), dtype=bool)
    is_cav[:, :, 0] = True
    dataset = TrajectoryDataset.from_arrays(
        {"states": states, "present": present, "is_cav": is_cav})
    batch = collate([dataset[0], dataset[1]])
    common = dict(device="cpu", threads=4, unroll_curriculum_s=(10.0,),
                  batch_size=2)
    structured = WmSTrainer(small_model(), WmSTrainConfig(**common))
    black = WmSTrainer(small_model(black_box=True, controller_embed_dim=4),
                       WmSTrainConfig(black_box=True, controller_embed_dim=4,
                                      **common))
    s = structured._forward_backward(batch, unroll_steps=20)
    b = black._forward_backward(batch, unroll_steps=20)
    assert s["valid_targets"] == (n * 20 * (slots - 1))
    assert b["valid_targets"] == (n * 20 * slots)


def test_collate_stacks_controller_id():
    records = [{"states": np.zeros((4, 7), np.float32),
                "present": np.ones(4, bool), "slot_index": np.arange(4),
                "controller_id": 2},
               {"states": np.zeros((4, 7), np.float32),
                "present": np.ones(4, bool), "slot_index": np.arange(4),
                "controller_id": 4}]
    out = collate(records)
    assert out["controller_id"].tolist() == [2, 4]
