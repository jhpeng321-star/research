"""``WmN1Predictor``: the evalh ``Predictor`` for a WM-N1 ensemble.

Wraps ``model.ensemble.WmN1Ensemble`` loaded from a member checkpoint and
implements the ``evalh.protocol.Predictor`` protocol:

* ``predict_quantiles(rows) -> [len(rows), 19]`` for the row's own
  ``(tau_s, policy)`` (design section 2.4).  The world tensors come from the
  packed corpus at the row's ``(world, t0_s)`` (or from
  ``row.features["tensors"]``); the policy is the arm already baked into the
  window, and an optional ``row.features["actions"]`` per-CAV action override
  is honoured when present.
* ``sample_rollouts(rows, horizons_s) -> evalh.protocol.RolloutBatch`` running
  ``model.rollout.rollout`` and collapsing the annealed trajectories
  ``[n_samples, n_steps+1, B, K]`` to ``RolloutBatch.pos/speed [N, H, V]``
  (INTERFACES Mismatch 4).

The predictor is named on the ``evalh.run_eval`` command line, e.g.
``--predictor adapter.predictor:WmN1Predictor``.  Because the rollout uses a
CPU ``torch.Generator`` (``model.rollout``), ``sample_rollouts`` runs on CPU
and temporarily moves the ensemble there if the predictor device is not CPU.

The vehicle-slot count ``K`` is a FORMAT parameter (amendment 5: 128 for the
frozen v2 format, 192 for v3) and is carried by the window tensors; the net
has no K-dependent weights, so one checkpoint serves either format.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from evalh.idea_a import VARIANT_A, IdeaAVariantPredictor
from evalh.predictors import PredictorLoadError
from evalh.protocol import HORIZONS_S, QUANTILE_LEVELS, PredictorBase
from evalh.protocol import RolloutBatch
from model import constants as C
from model.ensemble import WmN1Ensemble
from model.rollout import rollout

from . import checkpoint as CK

#: The six frozen featurizer tensors a window must provide.
WINDOW_KEYS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask", "counter")

#: Expected rank (without the batch axis) of each tensor.
_WINDOW_RANK = {"veh_step": 2, "veh_mask": 1, "plan": 3, "wall": 2,
                "wall_mask": 1, "counter": 2}

_ACTION_INDEX = {name: i for i, name in enumerate(C.ACTIONS)}


class WindowUnavailableError(RuntimeError):
    """No window tensors could be resolved for a row."""


class PredictorNotLoadedError(RuntimeError):
    """The predictor was constructed without a checkpoint."""


# ---------------------------------------------------------- corpus lookup --
def _lookup_corpus(corpus, world: str, t0_s: float) -> dict:
    """Resolve a window dict from a corpus object/callable/mapping."""
    for name in ("window_tensors", "get_window", "window"):
        fn = getattr(corpus, name, None)
        if callable(fn):
            try:
                return fn(world, t0_s)
            except TypeError:
                return fn(world=world, t0_s=t0_s)
    if callable(corpus) and not hasattr(corpus, "keys"):
        return corpus(world, t0_s)
    if hasattr(corpus, "__getitem__"):
        for key in ((str(world), float(t0_s)), f"{world}@{t0_s:g}", str(world)):
            try:
                return corpus[key]
            except (KeyError, TypeError):
                continue
    raise WindowUnavailableError(
        f"corpus {corpus!r} exposes no window for ({world!r}, {t0_s!r})")


def _as_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32)
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _normalize_window(win: dict) -> dict:
    """Strip an optional leading batch axis and cast to float32 tensors."""
    out = {}
    for key in WINDOW_KEYS:
        if key not in win:
            raise WindowUnavailableError(f"window is missing tensor {key!r}")
        t = _as_float_tensor(win[key])
        rank = _WINDOW_RANK[key]
        if t.dim() == rank + 1 and t.shape[0] == 1:
            t = t[0]
        if t.dim() != rank:
            raise WindowUnavailableError(
                f"window tensor {key!r} has shape {tuple(t.shape)}, expected "
                f"rank {rank}")
        out[key] = t
    return out


def _apply_actions(win: dict, actions) -> dict:
    """Override the per-CAV action one-hot from ``row.features['actions']``.

    ``actions`` is a ``{slot: name}`` mapping or a length-``K`` sequence of
    action names (``"native"``, ``"damping"``, ``"pi"``, ``"wall"``,
    ``"unknown"``).  Only slots marked ``is_cav`` are written.
    """
    veh = win["veh_step"].clone()
    plan = win["plan"].clone()
    K = veh.shape[0]
    if isinstance(actions, dict):
        items = actions.items()
    else:
        actions = list(actions)
        items = ((i, a) for i, a in enumerate(actions))
    for slot, action in items:
        slot = int(slot)
        if slot < 0 or slot >= K:
            continue
        if float(veh[slot, C.IS_CAV_IDX]) <= 0.0:
            continue
        idx = _ACTION_INDEX.get(str(action))
        if idx is None:
            continue
        veh[slot, C.ACT_NATIVE:C.ACT_UNKNOWN + 1] = 0.0
        veh[slot, C.ACT_NATIVE + idx] = 1.0
        plan[slot, :, 0:5] = 0.0
        plan[slot, :, idx] = 1.0
    return dict(win, veh_step=veh, plan=plan)


# --------------------------------------------------------------- adapter --
class WmN1Predictor(PredictorBase):
    """Concrete ``evalh.protocol.Predictor`` over a WM-N1 ensemble."""

    name = "wm_n1"
    quantile_levels = QUANTILE_LEVELS

    def __init__(self, checkpoint=None, corpus=None, *, device: str = "cuda"):
        """``checkpoint`` is an ensemble checkpoint (path or pre-built
        :class:`model.ensemble.WmN1Ensemble`); ``corpus`` resolves a row's
        window tensors at its ``(world, t0_s)``."""
        self.device = str(device)
        self.corpus = corpus
        self.checkpoint_path: Optional[str] = None
        self.config: dict = {}
        #: ``{member_index_0based: {world, ...}}``; empty = all members saw
        #: every TRAIN world (no member-level OOF available).
        self.member_train_worlds: dict = {}
        self._ensemble: Optional[WmN1Ensemble] = None
        if checkpoint is not None:
            self._load(checkpoint)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint, corpus=None, *,
                        device: str = "cuda") -> "WmN1Predictor":
        """Load all members; return a mixture predictor."""
        return cls(checkpoint, corpus, device=device)

    def _load(self, checkpoint) -> None:
        if isinstance(checkpoint, WmN1Ensemble):
            self._ensemble = checkpoint.eval()
            self.config = CK.ensemble_config(checkpoint)
            self.checkpoint_path = None
        else:
            ensemble, conf = CK.load_ensemble(checkpoint, device=self.device)
            self._ensemble = ensemble
            self.config = conf
            self.checkpoint_path = str(checkpoint)
        self.member_train_worlds = _parse_member_worlds(self.config)

    def _require_loaded(self) -> WmN1Ensemble:
        if self._ensemble is None:
            raise PredictorNotLoadedError(
                "WmN1Predictor has no checkpoint; construct it with "
                "from_checkpoint(checkpoint, corpus, device=...)")
        return self._ensemble

    # -- window assembly ---------------------------------------------------
    def _window(self, row) -> dict:
        feats = getattr(row, "features", None) or {}
        if "tensors" in feats:
            return _normalize_window(feats["tensors"])
        if self.corpus is not None:
            win = _lookup_corpus(self.corpus, row.world,
                                 float(getattr(row, "t0_s", 0.0)))
            win = _normalize_window(win)
            if "actions" in feats:
                win = _apply_actions(win, feats["actions"])
            return win
        raise WindowUnavailableError(
            f"row {getattr(row, 'world', '?')!r}: no corpus and no "
            "features['tensors'] to resolve the window tensors")

    def _build_batch(self, rows, *, device: Optional[str] = None) -> dict:
        dev = self.device if device is None else device
        windows = [self._window(r) for r in rows]
        batch = {}
        for key in WINDOW_KEYS:
            batch[key] = torch.stack([w[key] for w in windows], dim=0).to(dev)
        return batch

    # -- outcome head ------------------------------------------------------
    def _predict(self, rows, member_idx: Optional[Sequence[int]]) -> np.ndarray:
        ensemble = self._require_loaded()
        rows = list(rows)
        if not rows:
            return np.zeros((0, self.quantile_levels.size), dtype=float)
        batch = self._build_batch(rows)
        members = (list(range(len(ensemble.members)))
                   if member_idx is None else [int(i) for i in member_idx])
        with torch.no_grad():
            outputs = [ensemble.members[i](**batch) for i in members]
            mixture = WmN1Ensemble.mixture_quantiles(outputs)  # [B, T, Tgt, Q]
        q = mixture.detach().to("cpu").numpy()
        taus = np.asarray(C.TAUS, dtype=float)
        tgt = C.OUTCOME_TARGETS.index("delay_s")
        out = np.empty((len(rows), C.N_QUANTILES), dtype=float)
        for i, r in enumerate(rows):
            ti = int(np.argmin(np.abs(taus - float(r.tau_s))))
            out[i] = q[i, ti, tgt, :]
        out = np.maximum.accumulate(out, axis=1)
        return self._validate(out, len(rows))

    def predict_quantiles(self, rows) -> np.ndarray:
        """[len(rows), 19] delay quantiles at each row's own tau/policy."""
        return self._predict(rows, None)

    def predict_quantiles_members(self, rows, member_idx) -> np.ndarray:
        """Same as :meth:`predict_quantiles` but over a member subset.

        Used by the OOF export to score a world fold with only the members
        that did not train on that fold's worlds.
        """
        return self._predict(rows, member_idx)

    # -- rollout head ------------------------------------------------------
    def sample_rollouts(self, rows, horizons_s=None) -> RolloutBatch:
        """Return ``evalh.protocol.RolloutBatch`` for the short-horizon metrics."""
        rows = list(rows)
        if horizons_s is None:
            horizons_s = (tuple(getattr(rows[0], "horizons_s", HORIZONS_S))
                          if rows else HORIZONS_S)
        horizons = tuple(float(h) for h in horizons_s)
        n_rows = len(rows)
        if n_rows == 0:
            return RolloutBatch(horizons, np.zeros((0, len(horizons), 0)),
                                np.zeros((0, len(horizons), 0)))

        ensemble = self._require_loaded()
        n_steps = max(1, int(round(max(horizons) / C.DT_S)))
        batch = self._build_batch(rows, device="cpu")

        moved = self.device != "cpu"
        if moved:
            ensemble.to("cpu")
        try:
            traj_pos = traj_vel = traj_mask = None
            intervention = np.zeros(n_rows, dtype=float)
            for k, member in enumerate(ensemble.members):
                out = rollout(member, batch, n_steps=n_steps, n_samples=1,
                              seed=101 + 7 * k)
                pos = out["positions"][0].detach().to("cpu")
                vel = out["speeds"][0].detach().to("cpu")
                msk = out["masks"][0].detach().to("cpu")
                if traj_pos is None:
                    traj_pos = pos.clone()
                    traj_vel = vel.clone()
                    traj_mask = msk.clone()
                else:
                    traj_pos += pos
                    traj_vel += vel
                    traj_mask |= msk
                for counts in out["validity"]:
                    intervention += float(sum(counts.as_dict().values()))
            n_mem = len(ensemble.members)
            traj_pos = (traj_pos / n_mem).numpy()
            traj_vel = (traj_vel / n_mem).numpy()
            traj_mask = traj_mask.numpy()
        finally:
            if moved:
                ensemble.to(self.device)

        present = _present_slots(batch["veh_mask"].numpy())
        v_cols = max((len(p) for p in present), default=0)
        h = len(horizons)
        pos_out = np.zeros((n_rows, h, v_cols), dtype=float)
        speed_out = np.zeros((n_rows, h, v_cols), dtype=float)
        mask_out = np.zeros((n_rows, h, v_cols), dtype=float)
        step_idx = [min(n_steps, int(round(hi / C.DT_S))) for hi in horizons]
        for b in range(n_rows):
            slots = present[b]
            n_use = min(len(slots), v_cols)
            for hi, si in enumerate(step_idx):
                pos_out[b, hi, :n_use] = traj_pos[si, b, slots[:n_use]]
                speed_out[b, hi, :n_use] = traj_vel[si, b, slots[:n_use]]
                mask_out[b, hi, :n_use] = 1.0
        return RolloutBatch(
            horizons, pos_out, speed_out, mask=mask_out,
            collision=np.zeros(n_rows, dtype=float),
            ttc_lt3_s=np.zeros(n_rows, dtype=float),
            hard_brake_s=np.zeros(n_rows, dtype=float),
            intervention=intervention)


# --------------------------------------------------------------- helpers --
def _present_slots(mask: np.ndarray) -> list:
    """List of occupied slot indices per row (matches the rollout's K axis)."""
    return [np.flatnonzero(mask[b] > 0.5).tolist() for b in range(mask.shape[0])]


def _parse_member_worlds(config: dict) -> dict:
    """Normalise checkpoint OOF bookkeeping to ``{0-based member: set}``.

    ``member_train_worlds`` (per-member list) wins; else a single
    ``train_worlds`` list means every member saw every listed world.
    """
    raw = config.get("member_train_worlds")
    if raw:
        out = {}
        for key, worlds in raw.items():
            try:
                idx = int(key) - 1          # checkpoint keys are 1-based
            except (TypeError, ValueError):
                continue
            out[idx] = {str(w) for w in worlds}
        return out
    train_worlds = config.get("train_worlds")
    if train_worlds:
        all_w = {str(w) for w in train_worlds}
        n = int(config.get("n_members", len(_seeds(config))))
        return {i: set(all_w) for i in range(n)}
    return {}


def _seeds(config: dict) -> list:
    return list(config.get("seeds", ()))


def eligible_members(member_train_worlds: dict, fold_worlds: Iterable,
                     n_members: int) -> list:
    """Member indices whose training worlds do not intersect ``fold_worlds``."""
    fold = {str(w) for w in fold_worlds}
    if not member_train_worlds:
        return []
    return [i for i in range(n_members)
            if not (set(member_train_worlds.get(i, set())) & fold)]


# --------------------------------------------------- composed Idea A bundle --
#: The P1-P5 modules of the composed Idea A bundle (INTERFACES_V2 P8), plus
#: the A6 state-only global-token encoder (``state_enc``) when the checkpoint
#: trained with ``state_global_token`` ON.
IDEA_A_BUNDLE_PARTS = ("ctrl", "hdv", "arrival", "corr", "state_enc")

_SPEC = Union[str, os.PathLike, Mapping, object]


class IdeaABundlePredictor(IdeaAVariantPredictor):
    """The evalh ``Predictor`` for the composed Idea A bundle (variant A).

    Thin named subclass of :class:`evalh.idea_a.IdeaAVariantPredictor` so the
    bundled checkpoint can be named on the ``evalh.run_eval`` command line as
    ``adapter.predictor:IdeaABundlePredictor``.
    """

    name = VARIANT_A


def _resolve_attr(spec: str):
    """Resolve a ``module:attr`` (dotted attr) import path."""
    if ":" not in str(spec):
        raise PredictorLoadError(f"spec {spec!r} must be 'module:attr'")
    mod_name, attr = str(spec).split(":", 1)
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _build_config(cfg_spec):
    """Resolve a spec ``config``: a config object, or ``{class, kwargs}``."""
    if cfg_spec is None:
        return None
    if not isinstance(cfg_spec, Mapping):
        return cfg_spec
    kwargs = dict(cfg_spec.get("kwargs", {}))
    cls_spec = cfg_spec.get("class")
    if cls_spec:
        return _resolve_attr(cls_spec)(**kwargs)
    return kwargs or None


def _build_bundle_part(spec: _SPEC, base_dir: Optional[Path] = None):
    """Build one P1-P5 module from a module, spec, or ``{factory, config, state}``.

    ``config`` is a config object, or ``{"class": spec, "kwargs": {...}}`` to
    build one.  It is passed positionally to ``factory`` when present.
    """
    if spec is None or not isinstance(spec, (str, os.PathLike, Mapping)):
        return spec
    if isinstance(spec, (str, os.PathLike)):
        cls = _resolve_attr(os.fspath(spec))
        return cls() if isinstance(cls, type) else cls
    factory = spec.get("factory", spec.get("class"))
    obj = None
    if factory is not None:
        fn = _resolve_attr(factory)
        if spec.get("kwargs"):
            obj = fn(**dict(spec["kwargs"]))
        else:
            cfg_obj = _build_config(spec.get("config"))
            if cfg_obj is not None:
                obj = fn(cfg_obj)
            else:
                obj = fn() if callable(fn) else fn
    state = spec.get("state")
    if state is not None and obj is not None:
        if isinstance(state, (str, os.PathLike)):
            state_path = Path(os.fspath(state))
            if base_dir is not None and not state_path.is_absolute():
                state_path = Path(base_dir) / state_path
            state = torch.load(str(state_path), map_location="cpu",
                               weights_only=True)
        obj.load_state_dict(state)
    return obj


def load_idea_a_bundle(source, *, device: str = "cpu"):
    """Build the composed Idea A bundle from a mapping or bundle directory.

    ``source`` may be:

    * the P3 ``rollout2.engine.IdeaABundle`` (returned unchanged);
    * a mapping with optional ``ctrl`` / ``hdv`` / ``arrival`` / ``corr`` /
      ``cfg`` entries; each part is a module instance, a ``module:attr`` spec,
      or ``{"factory": spec, "config": {...}, "state": path-or-dict}``;
    * a directory holding ``idea_a.json`` (the mapping) plus any state files
      it references relative to that directory.
    """
    from rollout2.engine import IdeaABundle as _EngineBundle

    if isinstance(source, _EngineBundle):
        return source
    base_dir: Optional[Path] = None
    if isinstance(source, (str, os.PathLike)):
        base_dir = Path(os.fspath(source))
        manifest = base_dir / "idea_a.json"
        if not manifest.is_file():
            raise WindowUnavailableError(
                f"Idea A bundle {base_dir} has no idea_a.json manifest")
        source = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise WindowUnavailableError(
            f"cannot build an Idea A bundle from {type(source)!r}")
    parts = {k: _build_bundle_part(source.get(k), base_dir)
             for k in IDEA_A_BUNDLE_PARTS}
    cfg = source.get("cfg", source.get("config"))
    try:
        return _EngineBundle(ctrl=parts["ctrl"], hdv=parts["hdv"],
                             arrival=parts["arrival"], corr=parts["corr"],
                             cfg=cfg, state_enc=parts["state_enc"])
    except TypeError:                                    # light namespace
        return _EngineBundle(parts["ctrl"], parts["hdv"], parts["arrival"],
                             parts["corr"], cfg, parts["state_enc"])


def load_idea_a_predictor(source, *, device: str = "cpu",
                          **predictor_kwargs) -> IdeaABundlePredictor:
    """Load the composed Idea A bundle and wrap it as a Predictor."""
    bundle = load_idea_a_bundle(source, device=device)
    return IdeaABundlePredictor(bundle, device=device, **predictor_kwargs)
