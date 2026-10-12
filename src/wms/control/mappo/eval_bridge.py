"""Load a trained MAPPO actor as ``policy_fn(state) -> accel [1, K]``.

This is the bridge into the SAME evaluation harness the C4 policy uses: the
checkpoint's ``meta.names`` carry the actor / group-encoder / observation /
safety dotted names, so the saved actor rebuilds and runs exactly like the C4
policy (deterministic action, safety layer applied).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import Tensor

from .rollout import resolve_name
from .trainer import _torch_load


def load_policy_fn(
    ckpt_path: Path | str,
    *,
    obs_builder=None,
    group_encoder=None,
    safety_layer=None,
    device: str = "cpu",
) -> Callable[[Mapping[str, Tensor]], Tensor]:
    """Return ``policy_fn(state) -> accel [1, K]`` (bounds + safety applied)."""
    ckpt = _torch_load(Path(ckpt_path))
    meta = ckpt.get("meta", {})
    names = meta.get("names", {})
    obs_builder = obs_builder or resolve_name(names.get("obs_builder_name", "fake_obs"))
    safety = safety_layer or resolve_name(names.get("safety_name", "fake_safety"))
    actor = resolve_name(names.get("actor_name", "fake_actor"))(**meta.get("actor_kwargs", {}))
    encoder = group_encoder or resolve_name(names.get("group_encoder_name", "fake_encoder"))(
        **meta.get("encoder_kwargs", {})
    )
    actor.load_state_dict(ckpt["actor"])
    encoder.load_state_dict(ckpt["group_encoder"])
    actor.eval()
    encoder.eval()

    @torch.no_grad()
    def policy_fn(state: Mapping[str, Tensor]) -> Tensor:
        obs = obs_builder(state)
        group_signal = encoder(obs)
        accel, _ = actor(obs, group_signal, deterministic=True)
        return safety(state, accel)

    return policy_fn
