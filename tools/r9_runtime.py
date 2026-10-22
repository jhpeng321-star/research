"""Single causal episode contract for the fitter and physical-support audit.

Only the source observation whitelist and current control timeline are read.
The legacy vectorized helper remains an explicitly different parity diagnostic.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import torch

from tools import r9_exact as exact
from wms.data.recorded_prefix import causal_scene
from wms.model import WmSModel
from wms.model.merge_core import core_features
from wms.train.losses.merge_pivot import build_recorded_geometry


def runtime_model(frozen):
    """Migrate only the new pivot scalars; all loaded checkpoint tensors freeze."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = WmSModel(replace(frozen.model.spec, merge_decision="first_passage"),
                         physics=frozen.model.physics)
        model.load_state_dict(frozen.model.state_dict(), strict=True)
    return model.cpu().eval().requires_grad_(False)


def load_scene(path):
    loaded = exact.load_episode(path)
    path = exact.guard(path)
    target_path = exact.guard(path / "target_mask.npy")
    present, vid = loaded["present"].astype(bool), loaded["identity"]
    continuity = (present[:-1] & present[1:] & (vid[:-1] == vid[1:]))
    if target_path.is_file():
        continuity &= np.load(target_path, allow_pickle=False)[:len(continuity)].astype(bool)
    loaded["raw"] = loaded["kin"]
    loaded["kin"] = causal_scene(loaded["kin"], present, continuity)
    loaded["continuity"] = continuity
    names = {"controller": "controller", "acceleration_mps2": "issued_acceleration_mps2",
             "target_speed_mps": "issued_target_speed_mps"}
    paths = {key: exact.guard(path / (name + ".npy")) for key, name in names.items()}
    clock_path = exact.guard(path / "control_time_s.npy")
    timeline = None
    if clock_path.is_file() and all(p.is_file() for p in paths.values()):
        clock = np.load(clock_path, allow_pickle=False)
        if not np.allclose(clock, loaded["time_s"][0] + np.arange(len(clock)), rtol=0, atol=1e-6):
            raise ValueError("runtime replay requires the complete 1 s control clock")
        timeline = {key: torch.from_numpy(np.load(p, allow_pickle=False).copy())[None]
                    for key, p in paths.items()}
        timeline["control_dt_s"] = 1.
    loaded["timeline"] = timeline
    return loaded


def cached_prepare(frozen, loaded, directory, n_quad=8):
    directory = exact.guard(directory)
    digest = hashlib.sha256()
    digest.update(frozen.metadata["sha256"].encode())
    digest.update(str(frozen.model.physics).encode())
    digest.update(str(n_quad).encode())
    for key in ("kin", "present", "is_cav", "identity", "continuity"):
        digest.update(np.ascontiguousarray(loaded[key]).tobytes())
    for key, value in (loaded["timeline"] or {}).items():
        digest.update(key.encode())
        digest.update(value.numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
    for path in (exact.ROOT / "tools/r9_runtime.py", exact.ROOT / "tools/r9_exact.py",
                 exact.ROOT / "src/wms/model/wms.py", exact.ROOT / "src/wms/model/merge_pivot.py",
                 exact.ROOT / "src/wms/train/losses/merge_pivot.py",
                 exact.ROOT / "src/wms/train/cav_drive.py",
                 exact.ROOT / "src/wms/data/recorded_prefix.py",
                 exact.ROOT / "src/wms/data/route_frame.py"):
        digest.update(exact.guard(path).read_bytes())
    from tools.r9_fit_exact import _replay_model, _SparsePairAttention
    digest.update(inspect.getsource(_replay_model).encode())
    digest.update(inspect.getsource(_SparsePairAttention).encode())
    path = exact.guard(directory / (digest.hexdigest() + ".pt"))
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)
    cache = prepare(frozen, loaded, n_quad)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


def compact_forward(model):
    """Skip absent-slot neural work while retaining original sorted slot priority."""
    forward = model.forward

    def compacted(state, **kwargs):
        k = state["present"].shape[-1]
        active = state["present"].any(0).nonzero().flatten()
        if not len(active):
            active = torch.tensor([0])

        def cut(value):
            if isinstance(value, dict):
                return {key: cut(val) for key, val in value.items()}
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == k:
                return value.index_select(1, active)
            return value

        out = forward(cut(state), **cut(kwargs))
        restored = {}
        for key, value in out.items():
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == len(active):
                full = value.new_zeros(value.shape[0], k, *value.shape[2:])
                full[:, active] = value
                value = full
            restored[key] = value
        return restored

    model.forward = compacted


@torch.no_grad()
def prepare(frozen, loaded, n_quad=8, *, compact=True):
    """Run the training/runtime replay once, capturing heads without feeding them back."""
    # Sparse attention changes only masked pair-bias work, never a scene edge.
    from tools.r9_fit_exact import _replay_model
    from wms.data.history import long_history_features

    model = _replay_model(runtime_model(frozen))
    if compact:
        compact_forward(model)
    rec = torch.as_tensor(loaded["kin"], dtype=torch.float32)[None]
    present = torch.as_tensor(loaded["present"], dtype=torch.bool)[None]
    cav = torch.as_tensor(loaded["is_cav"], dtype=torch.bool)[None]
    cont = torch.as_tensor(loaded["continuity"], dtype=torch.bool)[None]
    observed = observed_mask = None
    if model.long_history:
        observed, observed_mask = long_history_features(
            rec[0, :1], present[0, :1], cont[0, :0], present[0, 0],
            history_s=model.history_s, history_stride_s=model.history_stride_s,
            dt_s=model.physics.dt_s)
        observed, observed_mask = observed[None], observed_mask[None]
    rows = []
    birth = has_birth = None

    def capture(i, state, out, moved, geo, context):
        nonlocal birth, has_birth
        embedding = out["driver_embedding"]
        if birth is None:
            birth = torch.zeros_like(embedding)
            has_birth = torch.zeros_like(present[:, 0])
        if i:
            has_birth &= cont[:, i - 1]
        ramp = present[:, i] & (state["lane"] == model.physics.merge_lane)
        need = ramp & ~has_birth
        birth = torch.where(need.unsqueeze(-1), embedding, birth)
        has_birth |= need
        _, lane_base = frozen.model._physics_step(state, context["driver_params"])
        legacy_logits, _, _, legacy_hazard = frozen.model._lane_v5_forward(
            rec[:, i], present[:, i], out["hdv_mask"], lane_base,
            context["lane_age"], context["lane_v5_acc"],
            context["driver_history"], context["driver_history_mask"])
        original = core_features(rec[:, i], present[:, i], model.physics,
                                 intent=legacy_logits.softmax(-1)[..., model.physics.merge_target_lane])
        intent = original["intent"]
        original = {key: rec[:, i, :, 4 if key == "v" else 5] if key in ("v", "a")
                    else original[key] for key in exact.FEATURES}
        corrected = {key: geo[key] for key in exact.FEATURES if key in geo}
        corrected["a"] = moved[..., 5]
        for side in ("f", "l"):
            index = geo["index_" + side].clamp_min(0)
            corrected["a_" + side] = torch.where(geo["has_" + side],
                                                moved[..., 5].gather(-1, index), 0.)
        in_zone = ((rec[:, i, :, 0] >= model.physics.merge_zone_s0_m)
                   & (rec[:, i, :, 0] <= model.physics.merge_zone_s1_m))
        rows.append(dict(original=original, corrected=corrected,
                         original_eligible=ramp & in_zone & legacy_hazard[1].bool() & ~cav[:, i],
                         seq_eligible=ramp & in_zone & ~cav[:, i],
                         embedding=embedding[0].clone(), birth=birth[0].clone(),
                         intent=intent[0].clone(), moved=moved[0].clone()))

    inp = SimpleNamespace(states=rec, present=present, is_cav=cav, continuity=cont,
                          hist=None, hist_mask=None, observed_hist=observed,
                          observed_hist_mask=observed_mask, controller_id=None,
                          action_timeline=loaded["timeline"], replay_observer=capture)
    started = perf_counter()
    geo = build_recorded_geometry(model, inp, SimpleNamespace(n_steps=rec.shape[1] - 1))
    embedding = torch.stack([r["embedding"] for r in rows], 1)
    birth = torch.stack([r["birth"] for r in rows], 1)
    intent = torch.stack([r["intent"] for r in rows], 1)
    cache = dict(kin=rec[0], present=present[0], is_cav=cav[0],
                 identity=torch.as_tensor(loaded["identity"]), geometry=geo,
                 embedding=embedding, birth_embedding=birth, intent=intent,
                 proposals=torch.stack([r["moved"] for r in rows]),
                 geometry_flags={key: value[0].T for key, value in geo.items()
                                 if key in ("overlap", "physical_eligible", "reservation_veto", "ramp")})
    for name, field in (("original_runtime", "original"), ("corrected", "corrected")):
        features = {key: torch.stack([r[field][key][0] for r in rows], -1)
                    for key in exact.FEATURES}
        logits = exact.head_logits(frozen.core, features, embedding,
                                   birth_embedding=birth, intent=intent, n_quad=n_quad)
        eligible = (geo["eligible"][0].T if name == "corrected" else
                    torch.stack([r["original_eligible"][0] for r in rows], -1))
        cache[name] = {**logits, "features": features, "eligible": eligible}
        cache["weights"] = logits["weights"]
    original = cache["original_runtime"]["features"]
    seq = exact.seq_tf_logits({key: value.T[None] for key, value in original.items()},
                              frozen.core, n_quad=n_quad)
    cache["seq_tf"] = dict(logits=seq["logits"][0], features=original,
                            eligible=torch.stack([r["seq_eligible"][0] for r in rows], -1))
    cache["metadata"] = {**frozen.metadata, "replay_engine": "shared_runtime_causal",
                         "replay_seconds": perf_counter() - started,
                         "CAV_controls": "current_timeline" if loaded["timeline"] is not None
                         else "WM_physics_fallback",
                         "driver_protocol": "causal_birth_reencoding_and_continuity_resets",
                         "geometry": "WmSModel.pivot_decision_geometry",
                         "longitudinal_mode": "frozen_first_passage_runtime_core_yield_inactive",
                         "head_scope": "full_round8_head_on_common_runtime_geometry"}
    return cache
