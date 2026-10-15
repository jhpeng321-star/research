"""PAIRLOSS: paired SUMO branches as paired TRAIN batches, and multi-source C4 mixing.

GENRAND branch episodes (``traffic_shepherd.c4.genrand``, ``GENRAND_NOTES.md`` section 3) share
a ``pair_key = <world>@t<branch time>``: every arm of a group restarts from the SAME saved
SUMO state at the branch time t0 (identical pre-history, HDV roster and slots) and differs only
in the CAV actions after t0 (``SEED:j`` random-command arms, ``B`` = all native).

* :class:`PairedBranchSource` - one window per arm, starting at the branch time, with the
  horizon capped at the recorded branch length; arms are grouped by ``pair_key`` and checked
  for identical t0 slots / kinematics (a group that disagrees is dropped and counted).
* :class:`PairedBatchDataset` - wraps the assembled TRAIN dataset (all its attributes are
  delegated) and implements ``__getitems__``: torch's map-style fetcher hands it the whole
  batch of sampled indices, and with probability ``fraction`` (seeded by the indices, so the
  draw is reproducible and independent of the global RNG) the batch is replaced by a PAIRED
  batch of ``groups`` pair groups x up to ``max_arms`` arms (the reference arm - all-native,
  else the first arm - always included).  Records carry ``pair_group`` / ``pair_ref`` (row of
  the group's reference inside the batch), collated into ``batch["pair_group"]`` /
  ``batch["pair_ref"]`` for :mod:`wms.train.losses.paired_diff`.
* :func:`world_roles` / :func:`add_c4_sources` - one split side per world across every C4
  source (validation wins on disagreement) and the sampling-share mix of several sources.

The model sees the CAV actions only through the recorded CAV teacher (realized motion), as for
every other window; nothing here is a model input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from wms.data.c4_episodes import (C4EpisodeRefusal, C4WindowLoader, _open_episode,
                                  episode_window_ticks, read_source_manifest)
from wms.data.dataset import TrajectoryDataset

DT_S = 0.5
#: Branch kind of the all-native reference arm (``branching``/``genrand``: ``B``).
REFERENCE_KIND = "B"


def world_roles(docs: Sequence[dict]) -> tuple[dict[str, str], list[str]]:
    """``({world: role}, overridden worlds)``: one split side per world over all manifests.

    A world that is validation in ANY source is validation everywhere (conservative: no
    validation world can leak into TRAIN through another source).
    """
    seen: dict[str, set[str]] = {}
    for doc in docs:
        for e in doc["episodes"]:
            seen.setdefault(e["world"], set()).add(str(e.get("split_role")))
    roles = {w: ("val" if "val" in r else "train") for w, r in seen.items()}
    overridden = sorted(w for w, r in seen.items() if len(r) > 1)
    return roles, overridden


def _is_reference(entry: dict) -> bool:
    branch = entry.get("branch") or {}
    if str(branch.get("kind", "")) == REFERENCE_KIND:
        return True
    return str(entry.get("arm", "")).endswith("_native")


class PairedBranchSource:
    """Paired branch windows of one converted source (one window per arm at the branch time)."""

    def __init__(self, root: str | Path, *, horizon_steps: int, history_s=None,
                 history_stride_s: float = 1.0, entries_v2: bool = False,
                 k_slots: Optional[int] = None, heldout=None,
                 role_of: Optional[dict] = None, role: str = "train",
                  junction_fix: str = "off", command_known: bool = False, name: str = "paired",
                  horizon_mode: str = "fixed"):
        if horizon_mode not in ("fixed", "stage"):
            raise ValueError("paired horizon_mode must be fixed or stage")
        self.horizon_mode = horizon_mode
        root = Path(root)
        doc = read_source_manifest(root)
        if k_slots is not None and int(doc["k_slots"]) != int(k_slots):
            raise C4EpisodeRefusal(f"paired source has K={doc['k_slots']} slots, packs K={k_slots}")
        roles = role_of or {}
        entries = [e for e in doc["episodes"] if e.get("pair_key")
                   and roles.get(e["world"], e.get("split_role")) == role]
        if not doc.get("allow_designed", False):
            bad = [e["episode"] for e in entries
                   if set(e.get("controllers_used", ())) - {"native", "direct"}]
            if bad:
                raise C4EpisodeRefusal(f"designed controllers in {bad[:3]}")
        if heldout is not None and entries:
            heldout.refuse_worlds([e["world"] for e in entries], where=f"paired {name}")
        self.name = name
        self.root = root
        # t0 = the branch time (warmup_s of a branch episode); horizon capped by the branch.
        starts, spans = [], []
        for e in entries:
            ep = _open_episode(str(root / e["dir"]))
            times = np.asarray(ep.array("time_s"))
            i0 = int(np.searchsorted(times, float(e["warmup_s"]) - 1e-6))
            starts.append(i0)
            spans.append(len(times) - 1 - i0)
        if not entries:
            raise C4EpisodeRefusal(f"{root}: no {role} paired-branch episodes (pair_key)")
        available = {}
        for e, span in zip(entries, spans):
            key = str(e["pair_key"])
            available[key] = min(available.get(key, int(horizon_steps)), span)
        self.horizon = int(min(int(horizon_steps), min(spans)))
        if horizon_mode == "stage":
            self.horizon = max(available.values())
        if self.horizon < 2:
            raise C4EpisodeRefusal(f"{root}: paired branches shorter than one control step")
        index: list[tuple[int, int]] = []
        keep: list[dict] = []
        tick_of: list[int] = []
        for e, i0 in zip(entries, starts):
            ep = _open_episode(str(root / e["dir"]))
            horizon = (available[str(e["pair_key"])] if horizon_mode == "stage"
                       else self.horizon)
            if horizon < 2:
                continue
            ticks = episode_window_ticks(ep, horizon_steps=horizon, stride_s=1.0e6,
                                         start_s=float(e["warmup_s"]), warmup_s=float(e["warmup_s"]))
            if not ticks or ticks[0] != i0:
                continue
            keep.append(e)
            tick_of.append(i0)
        groups: dict[str, list[int]] = {}
        for i, e in enumerate(keep):
            groups.setdefault(str(e["pair_key"]), []).append(i)
        self.groups: list[dict] = []
        dropped_small, dropped_mismatch = 0, 0
        for key in sorted(groups):
            members = groups[key]
            if len(members) < 2:
                dropped_small += 1
                continue
            ref = next((m for m in members if _is_reference(keep[m])), members[0])
            ep_r = _open_episode(str(root / keep[ref]["dir"]))
            vid_r = np.asarray(ep_r.array("vid")[tick_of[ref]])
            kin_r = np.asarray(ep_r.array("veh_kin")[tick_of[ref]])
            ok = True
            for m in members:
                ep_m = _open_episode(str(root / keep[m]["dir"]))
                if (not np.array_equal(np.asarray(ep_m.array("vid")[tick_of[m]]), vid_r)
                        or not np.array_equal(np.asarray(ep_m.array("veh_kin")[tick_of[m]]),
                                              kin_r)):
                    ok = False
                    break
            if not ok:
                dropped_mismatch += 1
                continue
            rows = [ref] + [m for m in members if m != ref]
            self.groups.append({"pair_key": key, "rows": rows,
                                "has_native_ref": _is_reference(keep[ref])})
            if horizon_mode == "stage":
                self.groups[-1]["horizon_steps"] = available[key]
        if not self.groups:
            raise C4EpisodeRefusal(f"{root}: no usable pair group (>= 2 arms, identical t0)")
        used = sorted({r for g in self.groups for r in g["rows"]})
        remap = {old: new for new, old in enumerate(used)}
        for g in self.groups:
            g["rows"] = [remap[r] for r in g["rows"]]
        entries_used = [keep[i] for i in used]
        index = [(j, tick_of[i]) for j, i in enumerate(used)]
        loader_args = dict(history_s=history_s, history_stride_s=history_stride_s,
                           entries_v2=entries_v2, junction_fix=junction_fix,
                           command_known=command_known)
        if horizon_mode == "stage":
            loader = _HorizonWindowLoader(root, entries_used, index,
                                         [available[str(e["pair_key"])] for e in entries_used],
                                         loader_args)
        else:
            loader = C4WindowLoader(root, entries_used, index, horizon_steps=self.horizon,
                                    **loader_args)
        self.receipt = {
            "mode": "c4-paired-branches", "split": "TRAIN", "role": role, "source": str(root),
            "name": name, "horizon_steps": self.horizon, "n_groups": len(self.groups),
            "n_windows": len(index),
            "arms_per_group": sorted({len(g["rows"]) for g in self.groups}),
            "groups_with_native_reference": int(sum(g["has_native_ref"] for g in self.groups)),
            "dropped_groups_single_arm": dropped_small,
            "dropped_groups_t0_mismatch": dropped_mismatch,
            "worlds": sorted({e["world"] for e in entries_used}),
        }
        self.dataset = TrajectoryDataset([{} for _ in index], loader=loader, receipt=self.receipt)
        self.dataset.positions = list(index)
        if horizon_mode == "stage":
            self.horizon = max(g["horizon_steps"] for g in self.groups)
            self.receipt.update(horizon_mode="stage", horizon_steps=self.horizon,
                                horizon_buckets={str(h): sum(g["horizon_steps"] == h
                                                            for g in self.groups)
                                                 for h in sorted({g["horizon_steps"]
                                                                  for g in self.groups})})

    def __len__(self) -> int:
        return len(self.dataset)

    def group_records(self, group: int, rows: Sequence[int], *, base_row: int,
                      horizon_steps: Optional[int] = None) -> list[dict]:
        """Records of ``rows`` (dataset rows of ``group``; the first is the reference)."""
        out = []
        for r in rows:
            rec = dict(self.dataset[int(r)] if horizon_steps is None else
                       self.dataset._loader.load(int(r), horizon_steps))
            rec["pair_group"] = int(group)
            rec["pair_ref"] = int(base_row)
            rec["mix_source"] = self.name
            out.append(rec)
        return out


class _HorizonWindowLoader:
    """Lazy loaders per requested horizon; no episode arrays are copied or padded."""

    def __init__(self, root, entries, index, horizons, kwargs):
        self.root, self.entries, self.index = root, entries, index
        self.horizons, self.kwargs = horizons, kwargs
        self.loaders = {}

    def __call__(self, index):
        return self.load(index, self.horizons[index])

    def load(self, index, horizon):
        if horizon > self.horizons[index]:
            raise ValueError("paired window exceeds available group horizon")
        if horizon not in self.loaders:
            self.loaders[horizon] = C4WindowLoader(
                self.root, self.entries, self.index, horizon_steps=horizon, **self.kwargs)
        return self.loaders[horizon](index)


class PairedBatchDataset(Dataset):
    """TRAIN dataset wrapper that turns a ``fraction`` of the batches into paired batches."""

    def __init__(self, inner: Dataset, pairs: PairedBranchSource, *, fraction: float,
                 groups: int, max_arms: int, batch_size: int, seed: int = 0):
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("paired batch fraction must be in [0, 1]")
        self.inner = inner
        self.pairs = pairs
        self.fraction = float(fraction)
        self.n_groups = int(groups)
        self.max_arms = int(max_arms)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.paired_horizon = None
        self.eligible_groups = None
        if getattr(pairs, "horizon_mode", "fixed") == "stage":
            self.set_paired_horizon(pairs.horizon)
        receipt = getattr(inner, "receipt", None)
        if isinstance(receipt, dict):
            receipt["paired_batches"] = {
                "fraction": self.fraction, "groups": self.n_groups, "max_arms": self.max_arms,
                "batch_size": self.batch_size, "source": pairs.receipt}

    # ---- delegation (the trainer reads stage_weights / receipt / describe_stages ...)
    def __getattr__(self, name: str):
        if name in ("inner", "pairs") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> dict:
        return self.inner[index]

    def set_paired_horizon(self, horizon_steps: int) -> None:
        """Prefer all groups covering the stage; otherwise use the longest available bucket.

        A batch always uses one common horizon so collate cannot pad shorter targets into
        apparently valid supervision. Call before constructing the stage's worker loader.
        """
        if getattr(self.pairs, "horizon_mode", "fixed") != "stage":
            return
        if int(horizon_steps) < 1:
            raise ValueError("paired stage horizon must be positive")
        horizons = [g["horizon_steps"] for g in self.pairs.groups]
        self.paired_horizon = min(int(horizon_steps), max(horizons))
        self.eligible_groups = [i for i, h in enumerate(horizons) if h >= self.paired_horizon]

    def __getitems__(self, indices: Sequence[int]) -> list[dict]:
        idx = [int(i) for i in indices]
        rng = np.random.default_rng([self.seed, len(idx), *[i % (2 ** 32) for i in idx]])
        if self.fraction <= 0.0 or rng.random() >= self.fraction:
            return [self.inner[i] for i in idx]
        return self.paired_batch(rng, n_rows=len(idx))

    def paired_batch(self, rng: np.random.Generator, n_rows: Optional[int] = None) -> list[dict]:
        """One paired batch: ``groups`` groups, reference arm + random other arms each."""
        n_rows = int(n_rows or self.batch_size)
        groups = self.pairs.groups
        eligible = self.eligible_groups
        n_groups = len(groups) if eligible is None else len(eligible)
        k = max(1, min(self.n_groups, n_groups, max(1, n_rows // 2)))
        arms = max(2, min(self.max_arms, n_rows // k))
        chosen = rng.choice(len(groups) if eligible is None else eligible, size=k, replace=False)
        out: list[dict] = []
        for gi in sorted(int(g) for g in chosen):
            rows = groups[gi]["rows"]
            others = rows[1:]
            pick = rng.choice(len(others), size=min(arms - 1, len(others)), replace=False)
            take = [rows[0]] + [others[int(j)] for j in sorted(pick)]
            horizon_kw = ({} if self.paired_horizon is None else
                          {"horizon_steps": self.paired_horizon})
            out.extend(self.pairs.group_records(gi, take, base_row=len(out), **horizon_kw))
        return out


def add_c4_sources(train, val, sources: Sequence[tuple[str, dict, float]], *,
                   jam_speed_mps: float = 5.0, jam_min_fraction: float = 0.0):
    """Mix several C4 sources ``(name, build_c4_datasets(...), mix_ratio)`` into ``(train, val)``.

    Each source gets its sampling share ``mix_ratio``; the existing sources keep their relative
    shares scaled by ``1 - sum(mix_ratio)`` (as :func:`wms.data.c4_episodes.add_c4_source` for
    one source).  Validation windows are appended as ``<name>_val`` sources.
    """
    from wms.data import jam as jamdata
    from wms.data.mixdata import MixDataRefusal, MixedTrajectoryDataset, MixSource

    sources = [(n, s, float(r)) for n, s, r in sources if s.get("train") is not None and r > 0.0]
    total = sum(r for _, _, r in sources)
    if not sources:
        return train, val
    if not 0.0 < total < 1.0:
        raise MixDataRefusal("C4 source mix ratios must sum to a value in (0, 1)")
    keep = 1.0 - total
    receipt_extra: dict = {}
    if isinstance(train, MixedTrajectoryDataset):
        base = [MixSource(s.name, s.dataset, share=share * keep, events=s.events, jam=s.jam)
                for s, share in zip(train.sources, train.shares)]
        event_oversample, jam_oversample = train.event_oversample, train.jam_oversample
        receipt_extra = {k: v for k, v in train.receipt.items()
                         if k not in ("mode", "n_records", "sources", "event_oversample",
                                      "jam_oversample", "stage_sampling")}
    elif isinstance(train, jamdata.JamOversampledDataset):
        base = [MixSource("original", train.dataset, share=keep,
                          jam=np.asarray(train.flags, bool))]
        event_oversample, jam_oversample = 1.0, train.oversample
        receipt_extra = {"jam": train.receipt.get("jam")}
    else:
        base = [MixSource("original", train, share=keep)]
        event_oversample, jam_oversample = 1.0, 1.0
    added = []
    for name, src, ratio in sources:
        jam = None
        if jam_oversample > 1.0:
            jam = jamdata.compute_jam_flags(src["train"], speed_mps=float(jam_speed_mps),
                                            min_fraction=float(jam_min_fraction))["flags"]
        added.append(MixSource(name, src["train"], share=ratio, events=src["train_events"],
                               jam=jam))
    mixed = MixedTrajectoryDataset(base + added, event_oversample=event_oversample,
                                   jam_oversample=jam_oversample)
    mixed.receipt.update(receipt_extra)
    mixed.receipt["c4_sources"] = {name: src["receipt"] for name, src, _ in sources}
    vals = [(name, src["val"]) for name, src, _ in sources if src.get("val") is not None]
    if vals:
        if isinstance(val, MixedTrajectoryDataset):
            vsources = [MixSource(s.name, s.dataset, share=1.0) for s in val.sources]
        elif val is not None:
            vsources = [MixSource("original_val", val, share=1.0)]
        else:
            vsources = []
        val = MixedTrajectoryDataset(vsources + [MixSource(f"{name}_val", v, share=1.0)
                                                 for name, v in vals])
    return mixed, val


def mix_c4_sources(sources: Sequence[tuple[str, dict, float]], *, jam_oversample: float = 1.0,
                   jam_speed_mps: float = 5.0, jam_min_fraction: float = 0.0):
    """V5DATA (``data.c4_base``): ``(train, val)`` mixed from C4 sources ONLY (no pack base).

    ``sources`` are ``(name, build_c4_datasets(...), mix_ratio)`` with the base source first;
    each source is drawn with share ``mix_ratio`` (the ratios sum to 1, checked by the config)
    and, with ``jam_oversample`` > 1, its jam windows (``data.jam`` rule) are up-weighted inside
    the source exactly as :func:`add_c4_sources` does for the sources it adds.  Validation is
    every source's held-out-world windows as ``<name>_val`` (``None`` when there are none).
    """
    from wms.data import jam as jamdata
    from wms.data.mixdata import MixDataRefusal, MixedTrajectoryDataset, MixSource

    sources = [(n, s, float(r)) for n, s, r in sources if s.get("train") is not None and r > 0.0]
    if not sources:
        raise MixDataRefusal("no C4 source with TRAIN windows")
    added = []
    for name, src, ratio in sources:
        jam = None
        if float(jam_oversample) > 1.0:
            jam = jamdata.compute_jam_flags(src["train"], speed_mps=float(jam_speed_mps),
                                            min_fraction=float(jam_min_fraction))["flags"]
        added.append(MixSource(name, src["train"], share=ratio, events=src["train_events"],
                               jam=jam))
    mixed = MixedTrajectoryDataset(added, event_oversample=1.0,
                                   jam_oversample=float(jam_oversample))
    mixed.receipt["c4_sources"] = {name: src["receipt"] for name, src, _ in sources}
    vals = [MixSource(f"{name}_val", src["val"], share=1.0)
            for name, src, _ in sources if src.get("val") is not None]
    return mixed, (MixedTrajectoryDataset(vals) if vals else None)
