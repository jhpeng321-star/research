"""Physical observation suffixes must reach actor and centralized critic."""

import gymnasium as gym
import numpy as np
import torch

from env.features import context_feature_names, raw_obs_dim
from env.mappo import mappo_obs_dim
from models import FeatureEncoder, MAPPOModel


def _all_context_names():
    return context_feature_names(
        include_meter_load=True,
        include_cav_count=True,
        include_wave_signal=True,
        include_health=True,
        include_dv=True,
        include_margin=True,
        include_idm_absorber_gate=True,
        include_idm_controller_context=True,
    )


def test_context_schema_order_and_dimension():
    assert _all_context_names() == (
        "meter_load",
        "cav_count",
        "wave_risk",
        "wave_forming",
        "metastable_health",
        "injection_dv",
        "runway_margin",
        "idm_absorber_gate",
        "idm_nominal_acceleration",
        "idm_filtered_reference",
        "idm_reference_error",
        "idm_projection_delta",
        "idm_severe_level",
        "idm_wave_energy_level",
        "idm_severe_short_ema",
        "idm_wave_short_ema",
        "idm_severe_trend",
        "idm_wave_energy_trend",
    )
    assert raw_obs_dim(20) == 121
    assert (
        raw_obs_dim(
            20,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        )
        == 139
    )


def test_every_context_field_changes_embedding_and_has_gradient():
    names = _all_context_names()
    enc = FeatureEncoder(max_neighbors=20, context_dim=len(names))
    linear = enc.context_embed[0]
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(0.5)

    x = torch.zeros(1, raw_obs_dim(20) + len(names), requires_grad=True)
    base = enc(x).detach()
    start = raw_obs_dim(20)
    for i, _name in enumerate(names):
        changed = x.detach().clone()
        changed[0, start + i] = 1.0
        assert not torch.allclose(enc(changed), base)

    enc(x).sum().backward()
    grad = x.grad[0, start : start + len(names)]
    assert torch.all(grad != 0), grad


def test_mappo_actor_and_critic_consume_context_but_not_mask():
    context_dim = 8
    raw_dim = raw_obs_dim(20) + context_dim
    mask_dim = 3
    local_dim = raw_dim + mask_dim
    max_cavs = 2
    obs_space = gym.spaces.Box(-np.inf, np.inf, (mappo_obs_dim(local_dim, max_cavs),), np.float32)
    model = MAPPOModel(
        obs_space=obs_space,
        action_space=gym.spaces.Discrete(mask_dim),
        num_outputs=mask_dim,
        model_config={
            "custom_model_config": {
                "max_neighbors": 20,
                "max_cavs_critic": max_cavs,
                "local_obs_dim": local_dim,
                "raw_local_dim": raw_dim,
                "action_mask_dim": mask_dim,
                "context_dim": context_dim,
                "hidden": 16,
            }
        },
        name="context_consumption_test",
    )
    with torch.no_grad():
        model.encoder.context_embed[0].weight.fill_(1.0)
        model.encoder.context_embed[0].bias.fill_(0.5)

    local = torch.zeros(local_dim)
    local[-mask_dim:] = 1.0
    peers = torch.zeros(max_cavs, local_dim)
    peers[:, -mask_dim:] = 1.0
    trust = torch.tensor([1.0, 0.0])
    packed = torch.cat((local, peers.reshape(-1), trust)).unsqueeze(0)

    logits0, _ = model({"obs": packed}, [], None)
    value0 = model.value_function().clone()
    changed = packed.clone()
    changed[0, raw_obs_dim(20)] = 1.0
    # Change the same physical suffix in the active critic peer.
    peer0_start = local_dim
    changed[0, peer0_start + raw_obs_dim(20)] = 1.0
    logits1, _ = model({"obs": changed}, [], None)
    value1 = model.value_function().clone()
    assert not torch.allclose(logits0, logits1)
    assert not torch.allclose(value0, value1)

    # Mask changes constrain logits only; they are outside the physical encoder.
    encoded0 = model.encoder(local[:raw_dim].unsqueeze(0))
    local_mask_changed = local.clone()
    local_mask_changed[-1] = 0.0
    encoded1 = model.encoder(local_mask_changed[:raw_dim].unsqueeze(0))
    assert torch.allclose(encoded0, encoded1)
