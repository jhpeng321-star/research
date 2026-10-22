"""CPU round-8 checkpoint comparators; no simulator or fitter side effects.

There are TWO different legacy laws in the real implementation:

* ``seq_tf_logits`` / ``seq_tf_score`` reproduce ``merge_core_seqtf.seq_tf_chunk``:
  population location zero, no residual, recorded pre-motion geometry. Nonzero
  checkpoint heads remain loaded but that loss does not use them.
* ``head_logits`` reproduces the deployed secure-gap head: location at the first
  ramp appearance is persistent; the residual is evaluated at EACH step's local
  median alpha, then shared across quadrature nodes (not recomputed per node).

Quick fitter integration::

    frozen = load_frozen()  # strict wm_s/1, all weights, CPU float32
    cache = precompute_episode(frozen, kin=kin, present=mask, is_cav=cav,
                               identity=vid, corrected=False)
    cars = episode_sequences(cache)  # each has original_runtime [Q,T] logits
    law = hazard_distribution(cars[0]['original_runtime']['logits'], cache['weights'],
                              cars[0]['original_runtime']['eligible'])
    # Event at last decision: -law['log_mass'][-1]; censor: -log_survival[-1].
    # Authorizing/output row = decision/input row + 1.

For corrected geometry, supply the run's authoritative PhysicsConfig to
``load_frozen(physics=...)`` and use ``corrected=True``. A copied checkpoint
without a physics block/sidecar is sufficient for head/seq-TF parity, NOT a
verified dynamics replay. ``allow_unverified_physics=True`` explicitly permits a
diagnostic using flat checkpoint switches plus PhysicsConfig defaults.

``precompute_episode`` holds encoded driver parameters/embeddings from row zero
under ``identity_policy='runtime'``; ``'identity'`` re-encodes new identities and
clears their history (an explicit corrected-bookkeeping protocol). It
does not apply observational EMA driver updates. For a different observation
schedule pass ``forward_inputs[row]`` with the exact runtime kwargs, including
driver_embedding/driver_params, histories and recurrent state overrides via
``states[row]``. Birth encodings and the sequence latent still persist. Prefix
rows must be supplied, not invented. ``episode_sequences`` marks unidentified
left truncation and never silently excludes those cars.

``matched_refit_core`` clones the complete learned head for differentiable
``head_logits`` refitting on cached geometry/encodings. Longitudinal dynamics,
driver encoders and intent remain frozen. This is a conditional fixed-path
refit, not a closed-loop refit of the WM or a verified SUMO substep model.
"""
# The standalone tool bootstraps this worktree's src package before importing WMS.
# ruff: noqa: E402
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from wms.data.history import long_history_features
from wms.model.checkpoint import CHECKPOINT_FORMAT
from wms.model.controllers import DIRECT_ID
from wms.model.driver_encoder import PARAMETER_NAMES
from wms.model.history import history_step_features, initial_history, push_history
from wms.model.merge_core import MergeCoreNet, secure_gap_margin, secure_gap_sides
from wms.model.merge_frailty import gauss_hermite
from wms.model.merge_pivot import advance_longitudinal, geometry as corrected_geometry
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig
from wms.physics.mobil import update_merge_wait
from wms.physics.neighbours import lead_neighbour
from wms.physics.state import state_kin
from wms.train.losses.merge_core_seqtf import _car_sigma, seq_tf_chunk

DEFAULT_CHECKPOINT = ROOT / "data/r9_train/wm_s_stage300.pt"
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth")
FEATURES = ("gap_l", "gap_f", "v", "v_l", "v_f", "a", "a_l", "a_f", "has_l", "has_f")


def guard(path) -> Path:
    """Workspace-relative paths; resolve symlinks/traversal BEFORE any data open."""
    raw = str(path)
    resolved = (ROOT / path).resolve()
    text = (raw + "/" + str(resolved)).lower()
    if not resolved.is_relative_to(ROOT) or any(word in text for word in FORBIDDEN):
        raise ValueError(f"restricted/outside workspace path refused: {path}")
    if any(964000 <= int(m[0]) <= 964199
           for m in re.finditer(r"(?<!\d)964\d{3}(?!\d)", text)):
        raise ValueError("evaluation seed path refused")
    return resolved


def _digest(path):
    h = hashlib.sha256()
    with guard(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(path):
    """Read only guarded observation arrays; no truth/LC-state/control labels.

    Returns kwargs for precompute_episode plus time_s and episode metadata.
    The caller supplies current runtime controls separately when available.
    """
    path = guard(path)
    meta = json.loads(guard(path / "episode.json").read_text(encoding="utf-8-sig"))
    if 964000 <= int(meta.get("seed", -1)) <= 964199:
        raise ValueError("evaluation seed refused")
    guard(ROOT / str(meta.get("world", "TRAIN")))
    names = {"kin": "veh_kin", "present": "veh_mask", "identity": "vid",
             "is_cav": "is_cav", "time_s": "time_s"}
    result = {key: np.load(guard(path / (name + ".npy")), allow_pickle=False)
              for key, name in names.items()}
    delta = np.diff(result["time_s"])
    if not len(delta) or not np.allclose(delta, .5, atol=1e-9, rtol=0):
        raise ValueError("round-8 replay requires observed uniform dt=0.5 s")
    return {**result, "metadata": meta}


@dataclass
class FrozenRound8:
    model: WmSModel
    metadata: dict

    @property
    def core(self) -> MergeCoreNet:
        return self.model.merge_core


def load_frozen(path=DEFAULT_CHECKPOINT, *, physics: PhysicsConfig | Mapping | None = None):
    """Strict, deterministic CPU load; no optional/missing tensor initialization.

    Sets CPU threads=1 and deterministic algorithms on. Model construction uses
    a forked CPU RNG (caller RNG is preserved); no data/bootstrap seed is drawn.
    Sibling physics.json / TRAIN_RECEIPT.json are guarded independently. Stored
    config paths (init_from, out_dir, sidecars) are metadata, never opened.
    """
    path = guard(path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("exact comparator requires a wm_s/1 checkpoint with model/config")
    config = dict(payload["config"])
    spec = ModelSpec.from_mapping(config)
    if spec.merge_decision != "secure_gap" or config.get("black_box", False):
        raise ValueError("expected structured round-8 secure_gap checkpoint")
    source = "explicit" if physics is not None else "embedded"
    raw = physics if physics is not None else config.get("physics")
    if raw is None:
        for name in ("physics.json", "TRAIN_RECEIPT.json"):
            sibling = guard(path.with_name(name))
            if sibling.is_file():
                doc = json.loads(sibling.read_text(encoding="utf-8-sig"))
                raw = doc if name == "physics.json" else doc.get("physics")
                if raw is not None:
                    source = name
                    break
    confirmed = raw is not None
    if raw is None:
        source = "UNVERIFIED_flat_checkpoint_switches_plus_PhysicsConfig_defaults"
        raw = {k: v for k, v in config.items() if k in PhysicsConfig.__dataclass_fields__}
    physics = raw if isinstance(raw, PhysicsConfig) else PhysicsConfig(**dict(raw))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = WmSModel(spec, physics=physics)
        model.load_state_dict(payload["model"], strict=True)
    model.cpu().eval().requires_grad_(False)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("nonfinite checkpoint weights")
    metadata = {"path": path.relative_to(ROOT).as_posix(), "sha256": _digest(path),
                "stage_s": payload.get("stage_s"), "step": payload.get("step"),
                "physics_source": source, "physics_confirmed": confirmed,
                "physics": asdict(physics), "config": config,
                "seq_tf_semantics": "zero_location_no_residual_recorded_pre_motion",
                "runtime_semantics": "birth_location_persistent_normal_local_median_residual",
                "dtype": "float32", "device": "cpu"}
    return FrozenRound8(model, metadata)


def matched_refit_core(frozen: FrozenRound8) -> MergeCoreNet:
    """Independent complete head clone; yield is frozen (it affects cached motion).

    Optimize params() scalars, alpha_net/bias/sigma and residual head/scale.
    State dict has the EXACT checkpoint merge_core architecture and key names.
    Use with head_logits; do not detach its result before computing the loss.
    """
    core = copy.deepcopy(frozen.core).requires_grad_(True)
    core.yield_head.requires_grad_(False)
    core.yield_scale_raw.requires_grad_(False)
    return core


def head_logits(core, features, embedding, *, birth_embedding=None, intent=None, n_quad=8,
                require_both=None):
    """Full deployed head, features [...,T], encodings [...,T,D] -> [...,Q,T].

    birth_embedding is [...,T,D] (persistent per identity, including pre-zone
    ramp appearances). If omitted the FIRST supplied encoding is the birth;
    use only for a single identity whose birth is actually observed here.
    Current location is used for residual features, birth location for alpha_q.
    """
    shape = features["v"].shape
    local = core.alpha_location(embedding, shape)
    if birth_embedding is None:
        birth = local[..., :1].expand_as(local)
    else:
        birth = core.alpha_location(birth_embedding, shape)
    p = core.params()
    nodes, weights = gauss_hermite(n_quad, dtype=local.dtype, device=local.device)
    median_alpha = torch.exp(local.clamp(-8, 8))
    sides = secure_gap_sides(*(features[k] for k in FEATURES), p["s0"], p["T"], p["b"],
                             p["tau_a"], median_alpha, require_both)
    residual = core.margin_residual(*sides, features["v"], features["v_l"], features["v_f"],
                                    features["gap_l"], features["gap_f"], intent)
    z = nodes.view(*([1] * (len(shape) - 1)), -1, 1)
    alpha = torch.exp((birth.unsqueeze(-2) + core.alpha_scale() * z).clamp(-8, 8))
    margin = secure_gap_margin(*(features[k].unsqueeze(-2) for k in FEATURES),
                               p["s0"], p["T"], p["b"], p["tau_a"], alpha,
                               residual.unsqueeze(-2), None if require_both is None else
                               require_both.unsqueeze(-2))
    return {"logits": margin / p["temperature"].clamp_min(1e-3), "weights": weights,
            "location": local, "birth_location": birth, "residual": residual}


def seq_tf_logits(scene, core, *, n_quad=8, sigma=None, require_both=None):
    """Exact legacy seq-TF conditional logits [B,K,Q,N], even with nonzero heads.

    Pass the SAME sigma as seq_tf_chunk (its caller supplies birth_sigma).
    Default is the checkpoint's shared learned alpha_scale, not location.
    Eligibility/lifecycle are scene bookkeeping; seq_tf_score delegates those
    semantics, including its legacy unsupported-event omission, to actual code.
    """
    sample = scene["gap_l"]
    bsz, _, k = sample.shape
    nodes, weights = gauss_hermite(n_quad, dtype=sample.dtype, device=sample.device)
    sig = _car_sigma(core.alpha_scale() if sigma is None else sigma, bsz, k)
    if sig.dim() == 0:
        alpha = (sig * nodes).clamp(-8, 8).exp().view(1, n_quad, 1, 1)
    else:
        alpha = (sig * nodes.view(1, n_quad, 1)).clamp(-8, 8).exp().unsqueeze(2)
    p = core.params()
    margin = secure_gap_margin(*(scene[key].unsqueeze(1) for key in FEATURES),
                               p["s0"], p["T"], p["b"], p["tau_a"], alpha, None,
                               None if require_both is None else require_both.unsqueeze(1))
    return {"logits": (margin / p["temperature"].clamp_min(1e-3)).permute(0, 3, 1, 2),
            "weights": weights}


def seq_tf_score(scene, frozen, c0=0, c1=None, *, log_post=None, n_quad=8, sigma=None,
                 require_both=None):
    """Call the real seq_tf_chunk, returning (nll,count,posterior,metrics).

    log_post is [B,Q,K], detached across chunks by runtime itself. scene's
    _post_identity is mutated exactly as in runtime; clone a scene for each law.
    """
    return seq_tf_chunk(scene, c0, scene["n_steps"] if c1 is None else c1,
                        **frozen.core.params(), sigma=frozen.core.alpha_scale() if sigma is None
                        else sigma, n_quad=n_quad, log_post=log_post, require_both=require_both)


def _logsumexp(x, dim):
    supported = torch.isfinite(x).any(dim, keepdim=True)
    value = torch.logsumexp(torch.where(supported, x, torch.zeros_like(x)), dim)
    return torch.where(supported.squeeze(dim), value, torch.full_like(value, -torch.inf))


def hazard_distribution(logits, weights, eligible=None, *, log_post=None):
    """Persistent quadrature law [...,Q,T]; time never independently marginalized.

    Optional NORMALIZED log_post [...,Q] is the survivor posterior at this
    window's start. Returns log masses/survival/CDF [...,T], plus final survivor
    log_post [...,Q] and conditional per-node log survival. Zero event masses
    stay -inf. A zero survival denominator raises when reused as log_post.
    """
    if logits.ndim < 2 or torch.isnan(logits).any():
        raise ValueError("logits need quadrature/time axes and no NaNs")
    weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    if weights.shape != (logits.shape[-2],) or not bool((weights > 0).all()):
        raise ValueError("one positive quadrature weight per node required")
    if not torch.allclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("quadrature weights must sum to one")
    prior = weights.log().expand(logits.shape[:-1]) if log_post is None else log_post
    if prior.shape != logits.shape[:-1] or not bool(torch.isfinite(prior).any(-1).all()):
        raise ValueError("zero/invalid history conditioning denominator or posterior shape")
    if not torch.allclose(torch.logsumexp(prior, -1), torch.zeros_like(prior[..., 0]),
                          atol=1e-5, rtol=0):
        raise ValueError("log_post must be normalized survivor weights")
    if eligible is None:
        eligible = torch.ones_like(logits[..., 0, :], dtype=torch.bool)
    if eligible.shape != logits[..., 0, :].shape:
        raise ValueError("eligibility shape must equal logits without quadrature axis")
    log_h = torch.where(eligible.unsqueeze(-2), F.logsigmoid(logits), -torch.inf)
    log_wait = torch.where(eligible.unsqueeze(-2), F.logsigmoid(-logits), 0.0)
    cumulative = log_wait.cumsum(-1)
    previous = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
    log_mass = _logsumexp(prior.unsqueeze(-1) + previous + log_h, -2)
    log_survival = _logsumexp(prior.unsqueeze(-1) + cumulative, -2)
    final = prior + cumulative[..., -1] if logits.shape[-1] else prior
    norm = _logsumexp(final, -1)
    safe_norm = torch.where(torch.isfinite(norm), norm, torch.zeros_like(norm))
    posterior = final - safe_norm.unsqueeze(-1)  # all -inf explicitly if certain event
    return {"log_mass": log_mass, "log_survival": log_survival,
            "cdf": -torch.expm1(log_survival), "log_post": posterior,
            "node_log_survival": cumulative}


def condition_on_prefix(logits, weights, eligible, stop):
    """Condition on no merge in rows [:stop]; history is conditioning, not scored twice."""
    if not 0 <= stop <= logits.shape[-1]:
        raise ValueError("prefix stop outside recorded path")
    law = hazard_distribution(logits[..., :stop], weights, eligible[..., :stop])
    denominator = (law["log_survival"][..., -1] if stop else
                   torch.zeros_like(logits[..., 0, 0]))
    if not bool(torch.isfinite(denominator).all()):
        raise ValueError("zero history conditioning denominator")
    return law["log_post"], denominator


def _state(kin, present, cav):
    lane = kin[..., 1:4].argmax(-1)
    lane = torch.where(kin[..., 1:4].sum(-1) > 0, lane, -1)
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": present, "is_cav": cav}


def longitudinal_proposal(frozen, state, *, out=None, forward_kwargs=None, action_id=None,
                          action_command=None, cav_controller=None,
                          allow_unverified_physics=False):
    """One runtime WM longitudinal update, with PRE-change lane membership.

    No recorded next state is accepted. With no callback/valid direct command,
    CAVs use WM physics prediction (the corrected pivot's explicit fallback),
    not a fabricated future teacher. Non-direct native/PI/damping execution
    requires the caller's causal controller callback. Returns (moved,out).
    """
    if not frozen.metadata["physics_confirmed"] and not allow_unverified_physics:
        raise ValueError("authoritative run physics required for causal geometry")
    model = frozen.model
    kwargs = {} if forward_kwargs is None else dict(forward_kwargs)
    if out is None:
        out = model.forward(state, **kwargs)
    out = dict(out)
    kin = state_kin(state)
    moved = advance_longitudinal(kin, out["accel_mps2"], model.physics.dt_s,
                                 model.physics.stop_time_integration)
    present, cav = state["present"].bool(), state["is_cav"].bool() & state["present"].bool()
    direct = torch.zeros_like(cav) if action_id is None else cav & (action_id == DIRECT_ID)
    cap = (model._exec_cap(out, state, kwargs.get("driver_params"))
           if cav.any() and model.cav_exec_cap != "off" else None)
    ns, nv = moved[..., 0], moved[..., 4]
    if direct.any():
        direct, ns, nv = model._apply_direct_commands(
            out, direct, kin[..., 0], kin[..., 4], ns, nv, action_command, exec_cap=cap)
    if cav_controller is not None:
        ns, nv, nl = model._apply_cav_overrides(
            out, state, cav, direct, present, kwargs.get("k_steps", 0), action_id,
            action_command, None, cav_controller, ns, nv, state["lane"], exec_cap=cap)
        if bool((cav & (nl != state["lane"])).any()):
            raise ValueError("CAV lane-changing callback needs joint fixed-target arbitration")
    elif action_id is not None and bool((cav & ~direct).any()):
        raise ValueError("non-direct/invalid CAV command needs causal controller callback")
    moved = moved.clone()
    moved[..., 0], moved[..., 4], moved[..., 5] = ns, nv, out["accel_mps2"]
    return moved, out


def _corrected_features(kin, present, cav, physics):
    geo = corrected_geometry(kin, present, physics, decision_mask=~cav)
    s = kin[..., 0]
    ds = s.unsqueeze(-1) - s.unsqueeze(-2)
    k = s.shape[-1]
    target = present & (kin[..., 1:4].argmax(-1) == physics.merge_target_lane)
    others = target.unsqueeze(-2) & ~torch.eye(k, dtype=torch.bool)
    for side, mask, distance in (("f", ds >= 0, ds), ("l", ds <= 0, -ds)):
        index = torch.where(others & mask, distance, torch.inf).argmin(-1)
        geo["a_" + side] = torch.where(geo["has_" + side], kin[..., 5].gather(-1, index), 0)
    geo["a"] = kin[..., 5]
    return geo


@torch.no_grad()
def precompute_episode(frozen, *, kin, present, is_cav, identity, n_quad=8, corrected=True,
                       forward_inputs=None, states=None, actions=None, k_steps=0,
                       allow_unverified_physics=False, identity_policy="runtime"):
    """Replay causal rows and cache original-runtime and corrected head logits.

    Arrays [N+1,K,7]/[N+1,K]; identity is mandatory and never a head input.
    actions optionally contains runtime-vocabulary action_id [N,K],
    action_command [N,K,2]; use a callback via longitudinal_proposal separately
    for non-direct controllers. forward_inputs is a length-N list of runtime
    forward kwargs. states optionally overrides auxiliary state fields per row.
    k_steps is scalar or [N,K], NOT elapsed episode time by default: recorded
    one-step observations default to k=0; an unroll protocol must supply its k.

    Results: original_runtime / corrected / seq_tf dictionaries with features
    [K,N], logits [K,Q,N], eligibility [K,N]; shared embedding/birth_embedding
    [K,N,D], proposals [N,K,7], and original identity/lane/event arrays.
    seq_tf uses shared checkpoint sigma; use seq_tf_logits(sigma=...) when
    reproducing a caller's different per-car birth_sigma. seq_tf_score alone
    also reproduces legacy slot selection and unsupported-event omission.
    No post-event hypothetical exposure is manufactured by episode_sequences.
    identity_policy='runtime' retains fixed t0 encodings, presence-only alpha
    resets and historical buffers, as WMS.rollout does. 'identity' explicitly
    repairs ID replacement/history/encoder resets; report its effects separately.
    """
    kin = torch.as_tensor(kin, dtype=torch.float32, device="cpu")
    present, cav = (torch.as_tensor(x, dtype=torch.bool, device="cpu")
                    for x in (present, is_cav))
    identity = torch.as_tensor(identity, device="cpu")
    if kin.ndim != 3 or kin.shape[-1] != 7 or any(x.shape != kin.shape[:2]
            for x in (present, cav, identity)) or kin.shape[0] < 2 or kin.shape[1] < 1:
        raise ValueError("episode needs N+1 full-scene rows and matching identity/mask arrays")
    if not bool(torch.isfinite(kin[present]).all()):
        raise ValueError("missing present kinematics cannot be silently imputed")
    kin = torch.where(present.unsqueeze(-1), kin, 0)
    n, k = kin.shape[0] - 1, kin.shape[1]
    if identity_policy not in ("runtime", "identity"):
        raise ValueError("identity_policy must be runtime or identity")
    repair_identity = identity_policy == "identity"
    model = frozen.model
    histories = masks = params = embedding = birth = None
    has_birth = torch.zeros(1, k, dtype=torch.bool)
    acc = None
    lane_age = torch.full((1, k), 999.0)
    wait = None
    rows = {name: [] for name in ("original_runtime", "corrected")}
    embeddings, births, intents, proposals = [], [], [], []
    for row in range(n):
        state = _state(kin[row:row+1], present[row:row+1], cav[row:row+1])
        new = present[row] & (torch.ones(k, dtype=torch.bool) if row == 0 else
                              (~present[row-1] | ((identity[row] != identity[row-1])
                                                  if repair_identity else False)))
        gap, _ = lead_neighbour(state["veh_kin"], state["present"], model.physics)
        feat = history_step_features(state["veh_kin"], gap)
        if histories is None:
            histories, masks = initial_history(feat, state["present"], model.history_len)
        else:
            histories, masks = push_history(histories, masks, feat, state["present"])
            if repair_identity:
                histories[:, new, :-1] = 0
                masks[:, new, :-1] = False
            changed = state["lane"] != kin[row-1:row, :, 1:4].argmax(-1)
            lane_age = torch.where(changed, 0.0, lane_age + 1)
            if repair_identity:
                lane_age[:, new] = 999
        # Long history uses ONLY <= current row, with all-native identity continuity.
        dh = dm = None
        if model.long_history and (row == 0 or (repair_identity and bool(new.any()))):
            same = present[:row] & present[1:row+1] & (identity[:row] == identity[1:row+1])
            dh, dm = long_history_features(
                kin[:row+1], present[:row+1], same, present[row], history_s=model.history_s,
                history_stride_s=model.history_stride_s, dt_s=model.physics.dt_s)
            dh, dm = dh.unsqueeze(0), dm.unsqueeze(0)
        if embedding is None or (repair_identity and bool(new.any())):
            _, enc, par = model.encode_drivers(state, driver_history=dh, driver_history_mask=dm,
                                               history=histories, history_mask=masks)
            if embedding is None:
                embedding, params = enc, par
            else:
                embedding = torch.where(new.view(1, k, 1), enc, embedding)
                params = {key: torch.where(new.view(1, k), par[key], params[key])
                          for key in PARAMETER_NAMES}
        if wait is None:
            wait = model.initial_merge_wait(state["present"], state["veh_kin"], dh, dm)
            lane_age = model._initial_lane_age(state, state["present"], state["veh_kin"], dh, dm)
        else:
            wait = update_merge_wait(wait, state["veh_kin"], state["present"], model.physics)
            if repair_identity:
                wait[:, new] = 0
        if acc is not None and repair_identity:
            acc = torch.where(new.view(1, k, 1), 0.0, acc)
        state["merge_wait_s"] = wait
        if states is not None:
            state.update(states[row])
        step_k = k_steps if np.ndim(k_steps) == 0 else torch.as_tensor(k_steps[row])
        if torch.is_tensor(step_k) and step_k.ndim > 0:
            step_k = step_k.unsqueeze(0)
        kwargs = dict(k_steps=step_k, history=histories, history_mask=masks,
                      driver_params=params, driver_embedding=embedding,
                      lane_age=lane_age, lane_v5_acc=acc)
        if row == 0:
            kwargs.update(driver_history=dh, driver_history_mask=dm)
        if forward_inputs is not None:
            kwargs.update(forward_inputs[row])
        out = model.forward(state, **kwargs)
        acc = out.get("lane_v5_acc")
        current_embedding = out.get("driver_embedding")
        if current_embedding is None:
            raise ValueError("episode helper requires a driver-encoding secure-gap WM")
        has_birth &= state["present"] & ~new.unsqueeze(0)
        ramp = state["present"] & (state["lane"] == model.physics.merge_lane)
        need = ramp & ~has_birth
        if birth is None:
            birth = torch.zeros_like(current_embedding)
        birth = torch.where(need.unsqueeze(-1), current_embedding, birth)
        has_birth |= need
        embeddings.append(current_embedding[0].clone())
        births.append(birth[0].clone())
        intents.append(out["merge_core_intent"][0])
        original = {key: (state["veh_kin"][..., 4 if key == "v" else 5] if key in ("v", "a")
                          else out["merge_core_" + key]) for key in FEATURES}
        in_zone = ((state["s_m"] >= model.physics.merge_zone_s0_m) &
                   (state["s_m"] <= model.physics.merge_zone_s1_m))
        original["eligible"] = ramp & in_zone & out["lane_v5_acts"].bool() & ~state["is_cav"]
        rows["original_runtime"].append(original)
        if corrected:
            action = {} if actions is None else {key: torch.as_tensor(value[row]).unsqueeze(0)
                                                 for key, value in actions.items()}
            moved, _ = longitudinal_proposal(frozen, state, out=out, forward_kwargs=kwargs,
                                             allow_unverified_physics=allow_unverified_physics,
                                             **action)
            proposals.append(moved[0])
            rows["corrected"].append(_corrected_features(moved, state["present"], state["is_cav"],
                                                         model.physics))
    embeddings = torch.stack(embeddings, 1)
    births = torch.stack(births, 1)
    intents = torch.stack(intents, 1)
    cache = {"kin": kin, "present": present, "is_cav": cav, "identity": identity,
             "embedding": embeddings, "birth_embedding": births, "intent": intents,
             "proposals": torch.stack(proposals) if proposals else None,
             "metadata": {**frozen.metadata, "driver_protocol": "encode_once_no_EMA",
                          "identity_policy": identity_policy,
                          "k_steps": "caller_supplied" if np.ndim(k_steps) else int(k_steps),
                          "corrected": corrected, "CAV_fallback": "WM_physics_no_future_teacher",
                          "geometry": "one_update_pre_change_lanes_full_scene_reservations",
                          "decision_likelihood_scope": "conditional_proposal_before_commit_arbitration",
                          "lane_conflict_resolve": model.lane_conflict_resolve,
                          "joint_conflict_likelihood": False,
                          "runtime_inputs_overridden": forward_inputs is not None or states is not None}}
    for mode, values in rows.items():
        if not values:
            continue
        features = {key: torch.stack([v[key][0] for v in values], -1) for key in FEATURES}
        logits = head_logits(frozen.core, features, embeddings, birth_embedding=births,
                             intent=intents, n_quad=n_quad)
        cache[mode] = {**logits, "features": features,
                       "eligible": torch.stack([v["eligible"][0] for v in values], -1)}
        cache["weights"] = logits["weights"]
    # Seq-TF intentionally ignores both heads and the deployment envelope. This
    # cache supplies per-identity distributions; legacy SLOT selection/unsupported
    # event omission is available only via seq_tf_score with its actual scene.
    features = cache["original_runtime"]["features"]
    sc = {key: value.T.unsqueeze(0) for key, value in features.items()}
    seqtf = seq_tf_logits(sc, frozen.core, n_quad=n_quad)
    ramp = kin[:-1, :, 1:4].argmax(-1) == model.physics.merge_lane
    feasible = (present[:-1] & ~cav[:-1] & ramp &
                (kin[:-1, :, 0] >= model.physics.merge_zone_s0_m) &
                (kin[:-1, :, 0] <= model.physics.merge_zone_s1_m))
    cache["seq_tf"] = {"logits": seqtf["logits"][0], "weights": seqtf["weights"],
                       "features": features, "eligible": feasible.T}
    return cache


def episode_sequences(cache):
    """All HDV ramp identity segments, cut at first event/exit/end (no deletion).

    Returns metadata + per-mode feature/embedding/logit views suited to a fitter.
    entry_row is the first RECORDED in-zone row; birth_row is first ramp row.
    unknown_history=True needs supplied prefixes/missing-history model before
    calibrated survivor scoring. terminal_kind is explicit: exits are not
    asserted to be independent administrative censoring.
    """
    kin, present, cav, ids = (cache[key] for key in ("kin", "present", "is_cav", "identity"))
    lane = torch.where(kin[..., 1:4].sum(-1) > 0, kin[..., 1:4].argmax(-1), -1)
    physics = SimpleNamespace(**cache["metadata"]["physics"])
    n, k = present.shape[0] - 1, present.shape[1]
    cars = []
    for slot in range(k):
        row = 0
        while row < n:
            if not present[row, slot]:
                row += 1
                continue
            start, end = row, row + 1
            while end <= n and present[end, slot] and ids[end, slot] == ids[start, slot]:
                end += 1
            row = end
            ramp = [r for r in range(start, min(end, n))
                    if lane[r, slot] == physics.merge_lane]
            risk = [r for r in ramp if physics.merge_zone_s0_m <= kin[r, slot, 0]
                    <= physics.merge_zone_s1_m]
            if not risk or cav[risk[0], slot]:
                continue
            entry, stop, event = risk[0], risk[0], False
            for r in range(entry, min(end, n)):
                if lane[r, slot] != physics.merge_lane or not (
                        physics.merge_zone_s0_m <= kin[r, slot, 0] <= physics.merge_zone_s1_m):
                    break
                stop = r + 1
                if r + 1 < end and lane[r+1, slot] == physics.merge_target_lane:
                    event = True
                    break
            kind = ("event" if event else "administrative_end" if stop == n and end > n
                    and lane[n, slot] == physics.merge_lane else "identity_or_absence_exit"
                    if stop + 1 >= end else "route_or_zone_exit")
            car = {"vehicle": int(ids[start, slot]), "slot": slot, "birth_row": ramp[0],
                   "entry_row": entry, "stop_row": stop, "event_row": stop if event else None,
                   "event": event, "terminal_kind": kind,
                   "unknown_history": entry == 0 or (entry == start and kin[entry, slot, 0] >
                                                                     physics.merge_zone_s0_m),
                   "embedding": cache["embedding"][slot, entry:stop],
                   "birth_embedding": cache["birth_embedding"][slot, entry:stop],
                   "intent": cache["intent"][slot, entry:stop]}
            for mode in ("original_runtime", "corrected", "seq_tf"):
                if mode not in cache:
                    continue
                values = cache[mode]
                car[mode] = {"logits": values["logits"][slot, :, entry:stop],
                             "eligible": values["eligible"][slot, entry:stop],
                             "features": {key: value[slot, entry:stop]
                                          for key, value in values["features"].items()}}
            cars.append(car)
    return cars
