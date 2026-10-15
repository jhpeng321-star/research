"""TRAIN-only start windows, penetration-balanced sampling and optional re-labelling."""
from __future__ import annotations

import os

import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor


_RESTRICTED = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)
_PENETRATION = re.compile(r"m\d+p(02|05|10|20|40)(?!\d)", re.I)  # e.g. w8411m2700p02t240


def checked_path(path, *, workspace=None) -> Path:
    """Refuse restricted tokens and symlink escapes before any file access."""
    value = Path(path)
    if _RESTRICTED.search(str(value)):
        raise ValueError(f"restricted source/output path: {value}")
    resolved = value.resolve()
    if _RESTRICTED.search(str(resolved)):
        raise ValueError(f"restricted resolved path: {value}")
    root = Path(workspace).resolve() if workspace else Path(__file__).resolve().parents[3]
    # Extra read roots (e.g. the data directory on the GPU box), os.pathsep-separated.
    extra = [Path(p).resolve() for p in os.environ.get("C4_ALLOWED_ROOTS", "").split(os.pathsep) if p]
    if not any(resolved.is_relative_to(r) for r in (root, *extra)):
        raise ValueError(f"path is outside workspace: {value}")
    return resolved


@dataclass
class StartBatch:
    state: dict[str, Tensor]
    rollout_kwargs: dict = field(default_factory=dict)
    metadata: list[dict] = field(default_factory=list)
    #: [B, F] observed pre-t0 entry-history features (``wms.data.pack_history.entry_features``,
    #: RSU-observable t <= t0 only) when the dataset was built with ``entries_v2``; else None.
    entry_feat: Tensor | None = None
    #: Set by ``wms.control.dreamer.with_entry_slots`` (entry histories, seeds, padding).
    entry_spec: dict | None = None


def penetration_of(meta):
    world = str(meta.get("world_id", meta.get("world", "")))
    match = _PENETRATION.search(world)
    if match is None:
        raise ValueError(f"world id needs a p02/p05/p10/p20/p40 suffix: {world}")
    return int(match.group(1)) / 100


def _metadata(dataset, index):
    if hasattr(dataset, "start_meta"):       # POLICYV5 C4 start windows (metadata-only lookup)
        return dict(dataset.start_meta(index))
    loader = getattr(dataset, "_loader", None)
    if loader is not None and hasattr(loader, "corpus"):
        return dict(loader.corpus.meta(loader.kept_orig[index]))
    record = dataset.records[index] if hasattr(dataset, "records") else dataset[index]
    return dict(record.get("meta", {}))


def _eligible(meta):
    if _RESTRICTED.search(str(meta)):
        return False
    # PackCorpus is a TRAIN view; reject any explicit contrary provenance.
    if str(meta.get("split", "TRAIN")).upper() != "TRAIN":
        return False
    return True


class StartStateSampler:
    """Uniform penetration then uniform window. Own CPU RNG is checkpointable.

    VAL is a disjoint development subset of eligible TRAIN windows, never an
    external evaluation split. Filtering is metadata-only before window loading.
    """

    def __init__(self, dataset, *, indices=None, excluded_penetrations=(), relabel=False,
                 target_penetration=0.2, seed=0, source_weights=None):
        self.dataset = dataset
        # C4 Dreamer loop (policy ``[data] start_source_weights``): within the drawn penetration,
        # pick the start source by these weights (renormalised over the sources present at that
        # penetration), then a uniform window of it.  None = the uniform-window default.
        self.source_weights = ({str(k): float(v) for k, v in source_weights.items()}
                               if source_weights else None)
        if self.source_weights and any(not w >= 0.0 for w in self.source_weights.values()):
            raise ValueError("start_source_weights must be nonnegative numbers")
        self.relabel = bool(relabel)
        self.target_penetration = float(target_penetration)
        if self.relabel and self.target_penetration not in (0.05, 0.1, 0.2, 0.4):
            raise ValueError("relabel target must be one of .05/.1/.2/.4")
        self.generator = torch.Generator().manual_seed(seed)
        self.groups = {}
        self.metadata = {}
        if str(getattr(dataset, "receipt", {}).get("split", "TRAIN")).upper() != "TRAIN":
            raise ValueError("only TRAIN source windows are allowed")
        for index in range(len(dataset)) if indices is None else indices:
            meta = _metadata(dataset, index)
            if not _eligible(meta):
                continue
            penetration = penetration_of(meta)
            if penetration in excluded_penetrations:
                continue
            self.groups.setdefault(penetration, []).append(index)
            self.metadata[index] = meta
        self.penetrations = sorted(self.groups)
        if not self.penetrations:
            raise ValueError("no eligible TRAIN start windows remain")
        self.source_groups = None
        if self.source_weights:
            self.source_groups = {}
            for penetration, group in self.groups.items():
                by_source = {}
                for index in group:
                    name = self.metadata[index].get("start_source")
                    if name not in self.source_weights:
                        raise ValueError(f"start window source {name!r} has no start_source_weights entry")
                    by_source.setdefault(name, []).append(index)
                self.source_groups[penetration] = sorted(by_source.items())

    def _draw_index(self):
        penetration = self.penetrations[int(torch.randint(
            len(self.penetrations), (), generator=self.generator))]
        group = self.groups[penetration]
        if self.source_groups is not None:
            parts = [(name, idx) for name, idx in self.source_groups[penetration]
                     if self.source_weights[name] > 0.0]
            if parts:
                weights = torch.tensor([self.source_weights[name] for name, _ in parts],
                                       dtype=torch.double)
                group = parts[int(torch.multinomial(weights, 1, generator=self.generator))][1]
        return group[int(torch.randint(len(group), (), generator=self.generator))]

    def _load(self, index):
        record = self.dataset[index]
        meta = record.get("meta", self.metadata[index])
        if not _eligible(meta):
            raise ValueError("window loader returned non-TRAIN/restricted provenance")
        kin = torch.as_tensor(record["states"][0]).float()
        present = torch.as_tensor(record["present"][0]).bool()
        cav = torch.as_tensor(record["is_cav"][0]).bool().clone() & present
        if self.relabel:
            slots = torch.where(present)[0]
            count = int(round(len(slots) * self.target_penetration))
            shuffled = slots[torch.randperm(len(slots), generator=self.generator)]
            cav.zero_()
            cav[shuffled[:count]] = True
        lane = kin[:, 1:4].argmax(-1)
        lane = torch.where(present & (kin[:, 1:4].sum(-1) > 0), lane, -1)
        state = {"s_m": kin[:, 0], "v_mps": kin[:, 4], "a_mps2": kin[:, 5],
                 "length_m": kin[:, 6], "lane": lane, "present": present, "is_cav": cav}
        hist = {}
        for source, target in (("hist", "history"), ("hist_mask", "history_mask"),
                               ("driver_hist", "driver_history"),
                               ("driver_hist_mask", "driver_history_mask")):
            if source in record:
                hist[target] = torch.as_tensor(record[source])
        meta = dict(meta)
        if "entry_feat" in record:       # kept out of the JSON-able metadata by _batch
            meta["_entry_feat"] = torch.as_tensor(record["entry_feat"]).float()
        return state, hist, meta

    def _batch(self, loaded, device):
        states, histories, metadata = zip(*loaded) if loaded else ((), (), ())
        if any(hist.keys() != histories[0].keys() for hist in histories):
            raise ValueError("all sampled windows must carry the same history fields")
        if "driver_history" not in histories[0] and "history" in histories[0]:
            for hist in histories:
                hist["driver_history"] = hist["history"]
                hist["driver_history_mask"] = hist["history_mask"]
        feats = [m.pop("_entry_feat", None) for m in metadata]
        entry_feat = (torch.stack(feats) if feats and all(f is not None for f in feats)
                      else None)
        return StartBatch({key: torch.stack([s[key] for s in states]).to(device)
                           for key in states[0]},
                          {key: torch.stack([h[key] for h in histories]).to(device)
                           for key in histories[0]}, list(metadata), entry_feat)

    def sample(self, batch_size, device="cpu"):
        return self._batch([self._load(self._draw_index()) for _ in range(batch_size)], device)

    def sample_where(self, batch_size, device="cpu", *, predicate, fraction=0.6, max_tries=20):
        """Like :meth:`sample`, but the first ``round(fraction * batch_size)`` starts satisfy
        ``predicate(state [1,K] dict)`` (up to ``max_tries`` draws each, then the last draw is
        kept).  ``metadata[i]["group_rich"]`` records the outcome.  Predicate results are cached
        per window when CAV labels are not re-drawn (relabel off)."""
        cache = self.__dict__.setdefault("_predicate_cache", {})
        wanted = int(round(float(fraction) * batch_size))
        loaded = []
        for slot in range(batch_size):
            tries = max(1, int(max_tries)) if slot < wanted else 1
            for attempt in range(tries):
                index = self._draw_index()
                if slot < wanted and not self.relabel and cache.get(index) is False                         and attempt < tries - 1:
                    continue
                state, hist, meta = self._load(index)
                ok = bool(predicate({k: v[None] for k, v in state.items()}))
                if not self.relabel:
                    cache[index] = ok
                if ok or slot >= wanted:
                    break
            meta["group_rich"] = ok
            loaded.append((state, hist, meta))
        return self._batch(loaded, device)

    def state_dict(self):
        return {"rng": self.generator.get_state(), "groups": self.groups,
                "relabel": self.relabel, "target_penetration": self.target_penetration}

    def load_state_dict(self, state):
        if (state["groups"] != self.groups or state["relabel"] != self.relabel or
                state["target_penetration"] != self.target_penetration):
            raise ValueError("resume start-state selection differs from saved run")
        self.generator.set_state(state["rng"].cpu())


def split_start_states(dataset, *, val_fraction=0.1, seed=0, **kwargs):
    """Stratified disjoint development VAL windows from the TRAIN-only corpus."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie strictly between zero and one")
    eligible = StartStateSampler(dataset, seed=seed, **kwargs)
    generator = torch.Generator().manual_seed(seed)
    train, val = [], []
    for group in eligible.groups.values():
        if len(group) < 2:
            raise ValueError("each penetration needs at least two windows for TRAIN/VAL")
        order = torch.randperm(len(group), generator=generator).tolist()
        n_val = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        val.extend(group[i] for i in order[:n_val])
        train.extend(group[i] for i in order[n_val:])
    return (StartStateSampler(dataset, indices=train, seed=seed + 1, **kwargs),
            StartStateSampler(dataset, indices=val, seed=seed + 2, **kwargs))
