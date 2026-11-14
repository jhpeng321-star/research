"""V0 dataset loader: MANIFEST verification, splits, row tables, batches.

Reads the canonical partB_v0data layout (READ-ONLY; DATA_CARD_V0.md):
per source <world>__<source>{_feat.npz,_labels.parquet,_td.parquet} under
worlds/<chunk>/, indexed by MANIFEST.json ("wd01_partB_v0data/1").

Contract implemented here:
  * MANIFEST.json sha256 is the data version tag; every file listed in
    the MANIFEST is sha256-verified against it unless verify=False
    (--no-verify);
  * split selection (TRAIN / CALIBRATION); HOLDOUT is never requested and
    asserted absent -- the V0 holdout was never built;
  * --exclude-superseded (default): sources whose MANIFEST entry carries
    superseded_by are skipped (V0 has none; the guard is for later
    rebuilds);
  * policy exposure: source base -> "damping_fleet", native/pi ->
    "switching_mix" (v0.3.2 fix #3: the campaign arms are random per-CAV
    switching schedules; the long MANIFEST policy prose stays
    provenance-only);
  * pre_fix, chunk_id, demand_veh_h, penetration, p40 exposed per row;
  * row weights equalise native vs damping rows per tau (native long-tau
    rows are thin: native tau=570 TRAIN = 141 chunk1 + 147 chunk2 rows).

All paths opened from MANIFEST-derived strings go through wdpaths.remap
(no D: anywhere; WD_FORBID_D stays green).  ZERO simulation: this module
only reads the built parquet/npz files.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import wdpaths  # output/wd01 root (E:-native; no D:)
from .model import POLICIES_V04A, policy_id

SCHEMA = "wd01_partB_v0data/1"

# canonical policy ids per source (the model's POLICIES tuple).
# v0.3.2 fix #3: the campaign '__native'/'__pi' arms of chunks 1-3 +
# chunk1_fix27 are random per-CAV switching schedules (per-CAV actions
# from switch_schedule.json / suffix_commands.jsonl, never
# dispatcher_rows.jsonl), NOT a fleet-wide native/PI policy -- they map
# to "switching_mix", never to a 'native' claim.  (Must stay equal to
# partB_v0data.v0lib / build_v032 POLICY_OF_SOURCE -- asserted in tests.)
POLICY_OF_SOURCE = {"base": "damping_fleet", "native": "switching_mix",
                    "pi": "switching_mix"}

TAUS = (30.0, 60.0, 120.0, 300.0, 570.0)

LABEL_VALUE_COLS = ("total_delay_ref_s", "ttc_lt3_veh_s", "hard_brake_s")
TD_COLS_KEEP = ("t", "t_next", "tau", "d30_s", "label_mc_complete")

FEAT_BLOCKS = ("lane_bins", "lane_mask", "cav_tokens", "cav_mask",
               "wall_tokens", "wall_mask", "demand", "context")

# Feature-set switch: "oracle" is the historical V0 feature file
# (<src>_feat.npz; its demand/context blocks carry the exact future-arrival
# schedule -- an offline upper bound, see PARTB_LEAK_AUDIT.md).  The
# deployable sets load sidecars built by partB_v0data/build_v031.py /
# build_v032.py: SAME state blocks, with demand/context replaced by
# RSU-observable inputs only (no future schedule, no future CAV share, no
# time-to-branch, no t0 -- v0.3.2 fix #1 -- no true penetration; every
# input has timestamp <= t).
#   deployable_strict   v0.3.2 HEADLINE (CAV-STRICT, owner decision): CAV
#                       self-report entrance counts ONLY -- no all-vehicle
#                       entrance counts, entrance queues zeroed, no
#                       penetration-revealing ratio.
#   deployable_detector v0.3.2 labelled SENSITIVITY: all-vehicle entrance
#                       counts + CAV counts + entrance queues.
#   deployable          DEPRECATED v0.3.1 alias of deployable_detector
#                       (reads the v0.3.1-era _feat_dep.npz sidecars).
FEATURE_SETS = ("oracle", "deployable", "deployable_detector",
                "deployable_strict")
FEAT_SUFFIX = {"oracle": "_feat.npz",
               "deployable": "_feat_dep.npz",
               "deployable_detector": "_feat_dep_det.npz",
               "deployable_strict": "_feat_dep_strict.npz"}
DEPRECATED_FEATURE_SETS = {"deployable": "deployable_detector"}


def sha256_of(path: str | os.PathLike) -> str:
    """Streaming sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class V0Manifest:
    """Verified MANIFEST + the selected, filtered source entries."""

    root: Path
    sha256: str                              # sha256 of MANIFEST.json itself
    raw: dict
    entries: dict = field(default_factory=dict)   # key -> MANIFEST entry
    n_superseded_skipped: int = 0
    n_aux_skipped: int = 0                   # v0.4a: --exclude-aux
    files_verified: bool = False

    @property
    def v04a(self) -> bool:
        """True for a WD-01-PB-V04A root (true-policy sources, per-CAV
        action tokens, per-row policy conditioning)."""
        return "v04a" in self.raw

    def __len__(self) -> int:
        return len(self.entries)


def default_data_root() -> Path:
    """Canonical V0 dataset root (env WD01_V0_DATA > RESEARCH_DENSE_ROOT)."""
    env = os.environ.get("WD01_V0_DATA", "").strip()
    if env:
        return Path(env)
    return Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / "wd01_partB_v0data"


def load_manifest(root: str | os.PathLike, *,
                  splits: tuple[str, ...] = ("TRAIN", "CALIBRATION"),
                  chunks: tuple[str, ...] | None = None,
                  exclude_superseded: bool = True,
                  verify_files: bool = True,
                  exclude_aux: bool = False,
                  log=print) -> V0Manifest:
    """Load MANIFEST.json, verify it + its files, select source entries.

    Raises on: unknown schema, a requested HOLDOUT split (never built in
    V0 -- requesting it is a contract violation), or any sha mismatch.
    """
    root = Path(root)
    man_path = root / "MANIFEST.json"
    man_path = Path(wdpaths.remap(man_path))
    raw = json.loads(man_path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"unexpected dataset schema {raw.get('schema')!r} "
                         f"(expected {SCHEMA!r}) at {man_path}")
    man = V0Manifest(root=root, sha256=sha256_of(man_path), raw=raw)

    if "HOLDOUT" in splits:
        raise ValueError("HOLDOUT must never be requested for V0 training "
                         "or evaluation (it is evaluated only under a "
                         "registration; it was never built)")
    chunks = tuple(chunks) if chunks else None

    for key, entry in sorted(raw["worlds"].items()):
        if entry["split"] not in splits:
            continue
        if chunks and entry["chunk_id"] not in chunks:
            continue
        if entry["split"] == "HOLDOUT":          # defensive; never selected
            raise AssertionError(f"HOLDOUT source selected: {key}")
        if exclude_superseded and entry.get("superseded_by"):
            man.n_superseded_skipped += 1
            continue
        if exclude_aux and entry.get("aux"):
            man.n_aux_skipped += 1           # v0.4a auxiliary ablation
            continue
        man.entries[key] = entry

    assert not any(e["split"] == "HOLDOUT" for e in man.entries.values()), \
        "HOLDOUT sources present after filtering"

    if verify_files:
        n_ok = 0
        for key, entry in sorted(man.entries.items()):
            for rel, sha in entry["files"].items():
                p = Path(wdpaths.remap(root / rel))
                if not p.is_file():
                    # v0.3: superseded sources' files are omitted from
                    # bundles (entries kept for provenance); with the
                    # default --exclude-superseded they are never
                    # selected, so reaching this line means they were
                    # explicitly included (--include-superseded) or the
                    # dataset is genuinely incomplete.
                    sup = entry.get("superseded_by")
                    raise ValueError(
                        f"data file missing: {rel} (source {key}"
                        + (f", superseded_by={sup!r}; its files are "
                           f"omitted from bundles -- drop "
                           f"--include-superseded" if sup else "")
                        + ")")
                got = sha256_of(p)
                if got != sha:
                    raise ValueError(
                        f"sha256 mismatch for {rel}: MANIFEST {sha} != file "
                        f"{got}")
                n_ok += 1
        man.files_verified = True
        log(f"data: verified {n_ok} files against MANIFEST "
            f"(sha256 {man.sha256[:12]}...)")
    else:
        log(f"data: MANIFEST loaded WITHOUT file verification "
            f"(sha256 {man.sha256[:12]}...)")
    n_holdout = sum(1 for e in raw["worlds"].values()
                    if e["split"] == "HOLDOUT")
    log(f"data: {len(man.entries)} sources selected "
        f"(splits={'+'.join(splits)}, "
        f"chunks={'+'.join(chunks) if chunks else 'all'}), "
        f"{man.n_superseded_skipped} superseded skipped, "
        f"{n_holdout} HOLDOUT entries present in MANIFEST (not selected, "
        f"never built in V0)")
    return man


def world_group_ids(src_names, src_split) -> list[str]:
    """Bootstrap unit ids (v0.3.2 fix #9): "<chunk>/<world>" for every
    TRAIN source -- a world's base and switching arms share ONE id, so a
    world is sampled or left out as a whole."""
    return sorted({n.rsplit("__", 1)[0]
                   for n, sp in zip(src_names, src_split)
                   if sp == "TRAIN"})


def bootstrap_world_ids(src_names, src_split, seed: int,
                        member: int) -> tuple[list, list]:
    """(sampled TRAIN world ids, OOB TRAIN world ids) for one member:
    len(world ids) draws WITH replacement over world_group_ids (the whole
    world -- every source -- moves together; fix #9).  Same rng scheme
    ([seed, member, 7]) as the v0.2 trainer's bootstrap."""
    train_worlds = world_group_ids(src_names, src_split)
    rng = np.random.default_rng([seed, member, 7])
    sampled = sorted(rng.choice(train_worlds,
                                size=len(train_worlds),
                                replace=True).tolist())
    oob = sorted(set(train_worlds) - set(sampled))
    return sampled, oob


class V0Dataset:
    """In-memory concatenation of the selected sources' rows + features.

    Attributes after load:
      rows   DataFrame of label rows (+ src_idx, t_idx, row_weight)
      td     DataFrame of TD tuples  (+ src_idx, t_idx, tnext_idx)
      feats  dict of stacked per-decision-time arrays, indexed by the
             global decision-time position g = (src_idx, t_idx)
      lane_geom  [3, LMAX] static geometry validity (identical all sources)
    """

    def __init__(self, man: V0Manifest, *, load_features: bool = True,
                 feature_set: str = "oracle", log=print):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {feature_set!r} "
                             f"({' | '.join(FEATURE_SETS)})")
        if feature_set in DEPRECATED_FEATURE_SETS:
            log(f"data: feature set {feature_set!r} is DEPRECATED (v0.3.1 "
                f"alias of "
                f"{DEPRECATED_FEATURE_SETS[feature_set]!r} without the "
                f"v0.3.2 fix #1 zeroed context col); use "
                f"deployable_strict (headline) or deployable_detector "
                f"(sensitivity) on a v0.3.2 root")
        self.man = man
        self.root = Path(wdpaths.remap(man.root))
        self.feature_set = feature_set
        self.log = log
        self._load_tables()
        if load_features:
            self._load_features()

    # ------------------------------------------------------------ tables --
    def _load_tables(self):
        labs, tds = [], []
        for si, (key, entry) in enumerate(sorted(self.man.entries.items())):
            world, source = entry["world"], entry["source"]
            lab = pd.read_parquet(Path(wdpaths.remap(
                self.root / f"worlds/{entry['chunk_id']}/"
                            f"{world}__{source}_labels.parquet")))
            td = pd.read_parquet(Path(wdpaths.remap(
                self.root / f"worlds/{entry['chunk_id']}/"
                            f"{world}__{source}_td.parquet")))
            lab = lab.assign(src_idx=si)
            td = td.assign(src_idx=si)
            labs.append(lab)
            tds.append(td)
        rows = pd.concat(labs, ignore_index=True)
        td_all = pd.concat(tds, ignore_index=True)
        self.v04a = bool(self.man.v04a)
        if self.v04a:
            # v0.4a root: the canonical policy class is a per-row label
            # column (the source names are no longer 1:1 with policies),
            # plus the planner-known schedule summary pi_* and the
            # auxiliary flag (training only)
            need = ["policy", "pi_native", "pi_damping", "pi_pi", "aux"]
            miss = [c for c in need if c not in rows.columns]
            if miss:
                raise ValueError(f"v0.4a label tables lack {miss}")
            unknown = sorted(set(rows["policy"]) - set(POLICIES_V04A))
            if unknown:
                raise ValueError(f"unknown v0.4a policy classes: {unknown}")
            rows["aux"] = rows["aux"].astype(bool)
            src_pol = {s: p for s, p in zip(rows["source"], rows["policy"])}
            td_all["policy"] = td_all["source"].map(src_pol)
        else:
            unknown = sorted(set(rows["source"]) - set(POLICY_OF_SOURCE))
            if unknown:
                raise ValueError(f"sources with no policy mapping: "
                                 f"{unknown}")
            rows["policy"] = rows["source"].map(POLICY_OF_SOURCE)
            td_all["policy"] = td_all["source"].map(POLICY_OF_SOURCE)
        rows["policy_id"] = rows["policy"].map(policy_id)
        self.rows = rows
        self.td = td_all

    def attach_indices(self, t_index):
        """Map label/td t values to global decision-time indices.

        t_index: dict (src_idx, t_s) -> global row (rounded to 1e-6 s).
        Called by _load_features; a features-less dataset has no t_idx.
        """
        tkey = self.rows["t"].to_numpy().round(6)
        src = self.rows["src_idx"].to_numpy()
        self.rows = self.rows.assign(
            t_idx=[t_index[(s, float(t))] if (s, float(t)) in t_index else -1
                   for s, t in zip(src, tkey)])
        bad = int((self.rows["t_idx"] < 0).sum())
        if bad:
            raise AssertionError(f"{bad} label rows with no feature row")

        tk = self.td["t"].to_numpy().round(6)
        tn = self.td["t_next"].to_numpy().round(6)
        s = self.td["src_idx"].to_numpy()
        self.td = self.td.assign(
            t_idx=[t_index[(a, float(t))] if (a, float(t)) in t_index else -1
                   for a, t in zip(s, tk)],
            tnext_idx=[t_index[(a, float(t))]
                       if (a, float(t)) in t_index else -1
                       for a, t in zip(s, tn)])

    # ---------------------------------------------------------- features --
    def _feat_path(self, entry) -> Path:
        world, source = entry["world"], entry["source"]
        return self.root / f"worlds/{entry['chunk_id']}/" \
                           f"{world}__{source}{FEAT_SUFFIX[self.feature_set]}"

    def _load_features(self):
        blocks = {k: [] for k in FEAT_BLOCKS}
        t_index: dict = {}
        g = 0
        lane_geom = None
        self.src_names: list[str] = []
        self.src_split: list[str] = []
        self.src_sizes: list[int] = []       # v0.3: decision times per source
        for si, (key, entry) in enumerate(sorted(self.man.entries.items())):
            z = np.load(Path(wdpaths.remap(self._feat_path(entry))),
                        allow_pickle=False)
            n = int(z["t"].shape[0])
            for k in FEAT_BLOCKS:
                blocks[k].append(z[k])
            if lane_geom is None:
                lane_geom = z["lane_geom"]
            elif not np.array_equal(lane_geom, z["lane_geom"]):
                raise AssertionError(f"lane_geom differs across sources at "
                                     f"{key}")
            for ti, t in enumerate(z["t"]):
                t_index[(si, round(float(t), 6))] = g + ti
            self.src_names.append(key)
            self.src_split.append(entry["split"])
            self.src_sizes.append(n)
            g += n
        self.feats = {k: np.concatenate(v, axis=0) for k, v in blocks.items()}
        self.lane_geom = lane_geom
        self.attach_indices(t_index)
        self._build_row_weights()
        self.log(f"data: {g} decision times, {len(self.rows)} label rows, "
                 f"{len(self.td)} TD rows in memory "
                 f"(feature set: {self.feature_set})")

    # ----------------------------------------------------------- weights --
    def _build_row_weights(self):
        """w = 1 / n(tau, policy): sums to 1 per (tau, policy) group, so a
        batch's native and damping rows contribute equally at each tau."""
        n = self.rows.groupby(["tau", "policy"])["t"].transform("size")
        w = 1.0 / n.to_numpy(dtype=np.float64)
        self.rows = self.rows.assign(row_weight=w)

    def td_weights(self) -> np.ndarray:
        """Equalisation weights for the TD table (same rule as rows)."""
        n = self.td.groupby(["tau", "source"])["t"].transform("size")
        return 1.0 / n.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------ batches --
    def gather_global(self, g_idx: np.ndarray, tau: np.ndarray,
                      policy: np.ndarray, pre_fix: np.ndarray) -> dict:
        """Feature batch at global decision-time positions with explicit
        tau / policy / pre_fix conditioning (used by gather, gather_td and
        the evaluation queries)."""
        g_idx = np.asarray(g_idx, dtype=np.int64)
        ctx = np.concatenate(
            [self.feats["context"][g_idx],
             np.asarray(pre_fix, dtype=np.float32)[:, None]], axis=1)
        return dict(
            lane_bins=self.feats["lane_bins"][g_idx],
            lane_mask=self.feats["lane_mask"][g_idx],
            lane_geom=self.lane_geom[None].repeat(len(g_idx), 0),
            cav_tokens=self.feats["cav_tokens"][g_idx],
            cav_mask=self.feats["cav_mask"][g_idx],
            wall_tokens=self.feats["wall_tokens"][g_idx],
            wall_mask=self.feats["wall_mask"][g_idx],
            demand=self.feats["demand"][g_idx],
            context=ctx.astype(np.float32),
            tau=np.asarray(tau, dtype=np.float32),
            policy=np.asarray(policy, dtype=np.int64),
        )

    def gather(self, idx: np.ndarray) -> dict:
        """Feature batch for label-row indices (numpy float32/long)."""
        idx = np.asarray(idx, dtype=np.int64)
        r = self.rows
        return self.gather_global(
            r["t_idx"].to_numpy()[idx],
            r["tau"].to_numpy()[idx],
            r["policy_id"].to_numpy()[idx],
            r["pre_fix"].to_numpy()[idx].astype(np.float32))

    def gather_td(self, idx: np.ndarray) -> tuple[dict, dict, np.ndarray]:
        """(batch at t, batch at t+30, d30_s) for TD-row indices."""
        idx = np.asarray(idx, dtype=np.int64)
        td = self.td
        src = td["src_idx"].to_numpy()[idx]
        pre = np.array([
            1.0 if self.man.entries[self.src_names[s]]["pre_fix"] else 0.0
            for s in src], dtype=np.float32)
        pol = np.array([policy_id(POLICY_OF_SOURCE[
            self.man.entries[self.src_names[s]]["source"]])
            for s in src], dtype=np.int64)
        tau = td["tau"].to_numpy()[idx].astype(np.float32)
        b_t = self.gather_global(td["t_idx"].to_numpy()[idx], tau, pol, pre)
        b_n = self.gather_global(td["tnext_idx"].to_numpy()[idx],
                                 tau - 30.0, pol, pre)
        d30 = td["d30_s"].to_numpy()[idx].astype(np.float32)
        return b_t, b_n, d30


# ===========================================================================
# v0.3 data plumbing (BRIEF_B3_V03.md A1, A4, B8, B11, C13; B3b).
#
#   * summary_features_base: the diag_tau30.py summary features per
#     decision time (per-lane counts & mean speeds, corridor-third counts,
#     context, demand); policy and pre_fix are appended per ROW, matching
#     the diagnosis exactly (40 columns total on the oracle feature set).
#   * fit_v03_stats: TRAIN-only feature standardisation statistics and
#     per-(tau, policy) target medians (A1) + per-tau delta scales (C13);
#     saved with the weights (A4).
#   * MedTable: the per-tau medians as an exact piecewise-linear function
#     of tau, so TD targets at off-grid tau-30 normalise through an exact
#     function (B8: "make the interpolation exact"), never label guesses.
#   * V03TensorData: device-resident standardised tensors (B11), row-level
#     normalised targets, TD table with on-grid n-step chains (B8), CRN
#     delta pairs (C13) and per-member OOB row sets (B6).
# ===========================================================================

STD_BLOCKS = ("lane_bins", "cav_tokens", "wall_tokens", "demand", "context")
SUMMARY_BASE_DIM = 38                     # oracle feature set (deployable
STATS_SCHEMA = "wd01_partB_v03_stats/1"   # differs; inferred from stats)
VALUE_COLS = {"delay": "total_delay_ref_s", "exp": "ttc_lt3_veh_s",
              "brake": "hard_brake_s"}
TD_CHAIN_STEPS = {30.0: 0, 60.0: 1, 120.0: 3, 300.0: 9, 570.0: 18}


def summary_features_base(feats: dict) -> np.ndarray:
    """[G, F] summary features per decision time (diag_tau30.py logic):
    per-lane counts, per-lane mean speeds, corridor-third counts, context,
    demand."""
    lb = feats["lane_bins"]                    # [G, 3, L, 3] cnt, v, std v
    cnt, spd = lb[..., 0], lb[..., 1]
    lane_n = cnt.sum(axis=2)                                   # [G, 3]
    lane_v = (cnt * spd).sum(axis=2) / np.maximum(lane_n, 1e-6)
    L = cnt.shape[2]
    thirds = [cnt[:, :, i * L // 3:(i + 1) * L // 3].sum(axis=(1, 2))
              for i in range(3)]
    return np.column_stack([lane_n, lane_v, *thirds,
                            feats["context"], feats["demand"]]).astype(
                                np.float32)


def fit_feature_stats(feats: dict, train_g: np.ndarray) -> dict:
    """Per-column mean/std over TRAIN decision times, per block.

    lane_bins is standardised per channel (count / mean speed / std speed);
    masks and lane_geom are passed through unscaled.  summary_base is the
    diag summary block; policy/pre_fix (the last 2 summary columns at row
    level) are 0/1 flags and travel unscaled.
    """
    stats: dict = {}
    blocks = {k: feats[k] for k in STD_BLOCKS}
    blocks["summary_base"] = summary_features_base(feats)
    for name, arr in blocks.items():
        sub = arr[train_g].astype(np.float64)
        axes = tuple(range(arr.ndim - 1))
        mean = sub.mean(axis=axes)
        std = sub.std(axis=axes)
        std = np.where(std < 1e-6, 1.0, std)
        stats[name] = dict(mean=mean.reshape(-1).tolist(),
                           std=std.reshape(-1).tolist(),
                           shape=list(arr.shape[1:]),
                           stat_shape=list(mean.shape))
    return stats


def _apply_stats(arr: np.ndarray, st: dict) -> np.ndarray:
    msh = tuple(st.get("stat_shape") or st["shape"])
    mean = np.asarray(st["mean"], dtype=np.float32).reshape(
        (1,) * (arr.ndim - len(msh)) + msh)
    std = np.asarray(st["std"], dtype=np.float32).reshape(
        (1,) * (arr.ndim - len(msh)) + msh)
    return ((arr - mean) / std).astype(np.float32)


def fit_target_medians(train_rows: pd.DataFrame) -> dict:
    """A1: per-(tau, policy) median of each value column on TRAIN."""
    out = {}
    for short, col in VALUE_COLS.items():
        grp = train_rows.groupby(["policy", "tau"])[col].median()
        pooled = train_rows.groupby("tau")[col].median()
        out[short] = {
            pol: {str(float(t)): float(grp.get((pol, t),
                                               pooled.get(t, np.nan)))
                  for t in TAUS}
            for pol in ("switching_mix", "damping_fleet")}
    return out


class MedTable:
    """Per-(tau, policy) medians as an exact piecewise-linear function of
    tau (linear between the label-grid nodes, clamped outside)."""

    def __init__(self, target_medians: dict, short: str):
        self.taus = np.array(TAUS, dtype=np.float64)
        self.by_policy = {}
        for pol in ("switching_mix", "damping_fleet"):
            self.by_policy[pol] = np.array(
                [target_medians[short][pol].get(str(float(t)), np.nan)
                 for t in TAUS], dtype=np.float64)
        self.pooled = np.nanmean(
            np.stack(list(self.by_policy.values())), axis=0)

    def _nodes(self, pol: str) -> np.ndarray:
        v = self.by_policy[pol].copy()
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = self.pooled[bad]
        if not np.isfinite(v).all():
            raise ValueError(f"MedTable has no nodes for policy {pol!r}")
        return v

    def at(self, tau, policy) -> np.ndarray:
        """Exact median at (possibly off-grid) tau per policy name."""
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        policy = np.atleast_1d(np.asarray(policy))
        out = np.empty(tau.shape, dtype=np.float64)
        rest = np.ones(tau.shape, dtype=bool)
        for pol in ("switching_mix", "damping_fleet"):
            m = rest & (policy == pol)
            if m.any():
                out[m] = np.interp(tau[m], self.taus, self._nodes(pol))
                rest &= ~m
        if rest.any():
            out[rest] = np.interp(tau[rest], self.taus, self.pooled)
        return out

    def to_json(self) -> dict:
        return {"taus": self.taus.tolist(),
                **{p: self._nodes(p).tolist() for p in self.by_policy}}


def fit_delta_scales(train_pairs: pd.DataFrame) -> dict:
    """C13: per-tau scale of the damping-fleet vs switching-mix delay
    delta on TRAIN (median |delta|; robust to the heavy tails)."""
    out = {}
    for t in TAUS:
        s = 1.0
        if len(train_pairs):
            sel = train_pairs[np.isclose(train_pairs["tau"], t)]
            if len(sel):
                med = float(np.median(np.abs(sel["delta_raw"].to_numpy())))
                if np.isfinite(med) and med > 1e-9:
                    s = med
        out[str(float(t))] = s
    return out


class DeltaScale:
    """Per-tau delta scale as an exact piecewise-linear function of tau."""

    def __init__(self, scales: dict):
        self.taus = np.array(TAUS, dtype=np.float64)
        self.vals = np.array([scales.get(str(float(t)), 1.0) for t in TAUS],
                             dtype=np.float64)

    def at(self, tau) -> np.ndarray:
        return np.interp(np.atleast_1d(np.asarray(tau, dtype=np.float64)),
                         self.taus, self.vals)

    def to_json(self) -> dict:
        return {"taus": self.taus.tolist(), "scale": self.vals.tolist()}


def build_delta_pairs(ds: V0Dataset, *, split: str,
                      label_fn=None) -> pd.DataFrame:
    """C13 CRN pairs: at a world's t0 the damping-fleet (base) and
    switching-mix (__native; a random per-CAV switching schedule, v0.3.2
    fix #3 -- NOT a fleet-native policy) runs restore the same state.
    Pairs are formed where BOTH sources have a label at the same tau near
    t0; the switching-arm restore grid can start 10 s after t0 (V0
    builder), so its side sits at its smallest available t >= t0 and the
    offset is recorded.  `pair_kind` is "restore" (offset 0) or
    "restore_offset<N>s"; chunk-4 native_base pairs would be "scenario"
    (same scenario, not the same state) and are out of scope here.

    label_fn is the REP-P replicate-mean hook: it receives the merged
    per-tau pair frame (columns y_a / y_b) and returns the delta array;
    the default is y_a - y_b (single-realisation difference).
    """
    r = ds.rows
    r = r[r["split"] == split]
    out = []
    for (world, chunk), g in r.groupby(["world", "chunk_id"]):
        ga = g[g["source"] == "base"]
        gb = g[g["source"] == "native"]
        if ga.empty or gb.empty:
            continue
        t0 = float(ga["t0_s"].iloc[0])
        a0 = ga[np.isclose(ga["t"], t0)]
        if a0.empty:
            continue
        tb = float(gb["t"].min())
        b0 = gb[np.isclose(gb["t"], tb)]
        if b0.empty:
            continue
        offset = tb - t0
        merged = a0.reset_index().merge(b0.reset_index(), on=["tau"],
                                        suffixes=("_a", "_b"))
        if len(merged) == 0:
            continue
        if label_fn is None:
            delta = (merged["total_delay_ref_s_a"].to_numpy()
                     - merged["total_delay_ref_s_b"].to_numpy())
        else:
            delta = np.asarray(label_fn(merged), dtype=np.float64)
        for i, row in enumerate(merged.itertuples()):
            out.append(dict(
                world=str(world), chunk=str(chunk), split=split,
                tau=float(row.tau), g_idx=int(row.t_idx_a),
                y_a=float(row.total_delay_ref_s_a),
                y_b=float(row.total_delay_ref_s_b),
                delta_raw=float(delta[i]),
                offset_s=float(offset),
                pre_fix=1.0 if bool(row.pre_fix_a) else 0.0,
                pair_kind=("restore" if abs(offset) < 1e-9 else
                           f"restore_offset{int(round(offset))}s")))
    cols = ["world", "chunk", "split", "tau", "g_idx", "y_a", "y_b",
            "delta_raw", "offset_s", "pre_fix", "pair_kind"]
    return pd.DataFrame(out, columns=cols)


def fit_v03_stats(ds: V0Dataset, *, log=print) -> dict:
    """Complete v0.3 statistics dict (JSON-serialisable, saved with the
    weights, A4): feature standardisation + per-tau target medians (A1)
    + per-tau delta scales (C13)."""
    sizes = np.asarray(ds.src_sizes, dtype=np.int64)
    g_src = np.repeat(np.arange(len(ds.src_names)), sizes)
    src_split = np.array(ds.src_split)
    train_g = src_split[g_src] == "TRAIN"
    train_rows = ds.rows[ds.rows["split"] == "TRAIN"]
    raw_pairs = build_delta_pairs(ds, split="TRAIN")
    stats = dict(
        schema=STATS_SCHEMA,
        feature=fit_feature_stats(ds.feats, train_g),
        target=fit_target_medians(train_rows),
        delta_scale=fit_delta_scales(raw_pairs),
        taus=[float(t) for t in TAUS],
        feature_set=getattr(ds, "feature_set", "oracle"),
        manifest_sha256=ds.man.sha256)
    log(f"data: v0.3 stats fit on {int(train_g.sum())} TRAIN decision "
        f"times, {len(train_rows)} TRAIN rows, {len(raw_pairs)} TRAIN "
        f"pairs")
    return stats


class V03TensorData:
    """Device-resident v0.3 training/eval data built from a V0Dataset.

    Standardised with `stats` (fit via fit_v03_stats when None).  Row
    targets are per-tau-normalised (A1).  Provides exact-normaliser TD
    targets (one-step bootstrap or on-grid n-step chains, B8), CRN delta
    pairs (C13) and per-member OOB row sets (B6).
    """

    def __init__(self, ds: V0Dataset, device: torch.device, *,
                 stats: dict | None = None, log=print):
        self.ds = ds
        self.device = device
        self.log = log
        self.stats = stats or fit_v03_stats(ds, log=log)
        self.med = {short: MedTable(self.stats["target"], short)
                    for short in VALUE_COLS}
        self._build_tensors()

    # ------------------------------------------------------------- build --
    def _to_dev(self, arr, dtype=np.float32) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(
            np.asarray(arr, dtype=dtype))).to(self.device)

    def model_dims(self) -> dict:
        """Input dimensions implied by the loaded feature set (oracle
        demand has 21 columns, deployable 12 -- the projections size from
        the stats, so --features deployable works for the net too; the
        v0.4 token widths are data-driven the same way: pre-v0.4 roots
        carry cav/wall 17/6, v0.4 roots 20/11)."""
        f = self.stats["feature"]
        return dict(
            demand_raw=int(f["demand"]["shape"][0]),
            ctx_raw=int(f["context"]["shape"][0]),
            cav_raw=int(f["cav_tokens"]["shape"][-1]),
            wall_raw=int(f["wall_tokens"]["shape"][-1]),
            summary_dim=len(f["summary_base"]["mean"]) + 2)

    def _build_tensors(self):
        ds = self.ds
        # ---------------- standardised feature blocks (A4, B11) --------
        blocks = {k: ds.feats[k] for k in STD_BLOCKS}
        blocks["summary_base"] = summary_features_base(ds.feats)
        self.t_feats = {k: self._to_dev(_apply_stats(
            v, self.stats["feature"][k])) for k, v in blocks.items()}
        self.t_feats["lane_mask"] = self._to_dev(ds.feats["lane_mask"])
        self.t_feats["cav_mask"] = self._to_dev(ds.feats["cav_mask"])
        self.t_feats["wall_mask"] = self._to_dev(ds.feats["wall_mask"])
        self.t_lane_geom = self._to_dev(ds.lane_geom)

        # ---------------- label rows ------------------------------------
        rows = ds.rows
        self.row_t_idx = rows["t_idx"].to_numpy().astype(np.int64)
        self.row_tau = rows["tau"].to_numpy().astype(np.float32)
        self.row_policy = rows["policy_id"].to_numpy().astype(np.int64)
        self.row_pre = rows["pre_fix"].to_numpy().astype(np.float32)
        self.row_split = rows["split"].to_numpy().astype(str)
        self.row_chunk = rows["chunk_id"].to_numpy().astype(str)
        self.row_world = rows["world"].to_numpy().astype(str)
        self.row_source = rows["source"].to_numpy().astype(str)
        self.row_w = rows["row_weight"].to_numpy().astype(np.float32)
        pol_names = rows["policy"].to_numpy().astype(str)
        self.t_row_g = self._to_dev(self.row_t_idx, np.int64)
        self.t_row_tau = self._to_dev(self.row_tau)
        self.t_row_pol = self._to_dev(self.row_policy, np.int64)
        self.t_row_pre = self._to_dev(self.row_pre)
        self.t_row_w = self._to_dev(self.row_w)
        self.y_coll = self._to_dev(
            rows["collision_any"].to_numpy().astype(np.float32))
        # A1: y / TRAIN median per (tau, policy)
        self.med_row = {short: self.med[short].at(self.row_tau, pol_names)
                        for short in VALUE_COLS}
        self.y_norm = {}
        for short in VALUE_COLS:
            y = rows[VALUE_COLS[short]].to_numpy().astype(np.float64)
            self.y_norm[short] = self._to_dev(
                y / np.maximum(self.med_row[short], 1e-9))
        self.split_train = self.row_split == "TRAIN"
        self.split_cal = self.row_split == "CALIBRATION"

        # ---------------- TD table + on-grid n-step chains (B8) --------
        self._build_td()
        # ---------------- delta pairs (C13) ------------------------------
        self._build_pairs()
        # ---------------- per-member OOB (B6) ----------------------------
        self.oob_rows: list[np.ndarray] = []

    def _build_td(self):
        td = self.ds.td
        t_real = td["t"].to_numpy().round(6)
        src = td["src_idx"].to_numpy().astype(np.int64)
        pol_names = td["policy"].to_numpy().astype(str)
        tau = td["tau"].to_numpy().astype(np.float32)
        d30 = td["d30_s"].to_numpy().astype(np.float64)
        t_idx = td["t_idx"].to_numpy().astype(np.int64)
        tnext_idx = td["tnext_idx"].to_numpy().astype(np.int64)
        pre = np.array([1.0 if self.ds.man.entries[
            self.ds.src_names[s]]["pre_fix"] else 0.0 for s in src],
            dtype=np.float32)

        # d30 per (src, t) (identical across tau rows at one t) and the
        # tau=30 label per (src, t) for chain terminals
        d30_map: dict = {}
        for si, t, dv in zip(src, t_real, d30):
            d30_map.setdefault((int(si), round(float(t), 6)), float(dv))
        lab30: dict = {}
        rows = self.ds.rows
        m30 = np.isclose(rows["tau"].to_numpy(), 30.0)
        for si, t, y in zip(rows["src_idx"].to_numpy()[m30],
                            rows["t"].round(6).to_numpy()[m30],
                            rows["total_delay_ref_s"].to_numpy()[m30]):
            y = float(y)
            lab30[(int(si), round(float(t), 6))] = (y, np.isfinite(y))

        n = len(t_idx)
        k_arr = np.array([TD_CHAIN_STEPS.get(round(float(t), 3), 0)
                          for t in tau], dtype=np.int64)
        chain_ok = np.zeros(n, dtype=bool)
        chain_sum = np.zeros(n, dtype=np.float64)
        term_y30 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            k = int(k_arr[i])
            if k < 1:
                continue
            si, t0 = int(src[i]), float(t_real[i])
            s = 0.0
            ok = True
            for j in range(k):
                dv = d30_map.get((si, round(t0 + 30.0 * j, 6)))
                if dv is None:
                    ok = False
                    break
                s += dv
            if ok:
                yterm, fin = lab30.get((si, round(t0 + 30.0 * k, 6)),
                                       (np.nan, False))
                ok = bool(fin)
                term_y30[i] = yterm
            chain_ok[i] = ok
            chain_sum[i] = s

        self.td = dict(t_idx=t_idx, tnext_idx=tnext_idx, tau=tau, d30=d30,
                       src=src, pre=pre,
                       split=td["split"].to_numpy().astype(str),
                       policy_id=np.array([policy_id(p) for p in pol_names],
                                          dtype=np.int64))
        self.td_chain = dict(k=k_arr, ok=chain_ok, sum=chain_sum,
                             term_y30=term_y30)
        # exact normalisers (B8): med(tau) for the target at the row's tau,
        # med(tau-30) to de-normalise the one-step bootstrap value
        self.td_med_tau = self.med["delay"].at(tau, pol_names)
        self.td_med_next = self.med["delay"].at(tau.astype(np.float64) - 30.0,
                                                pol_names)
        self.t_td = {k: self._to_dev(v, np.int64 if k in
                                     ("t_idx", "tnext_idx", "src",
                                      "policy_id") else np.float32)
                     for k, v in self.td.items() if k != "split"}
        self.t_td_chain = dict(
            ok=self._to_dev(chain_ok.astype(np.float32)).to(torch.bool),
            sum=self._to_dev(chain_sum),
            term_y30=self._to_dev(term_y30),
            k=self._to_dev(k_arr, np.int64))
        self.t_td_med_tau = self._to_dev(self.td_med_tau)
        self.t_td_med_next = self._to_dev(self.td_med_next)
        self.td_split_train = self.td["split"] == "TRAIN"
        self.td_policy_names = pol_names

    def _build_pairs(self):
        """C13 pairs -> frames + device tensors.  The delta target is
        delta_raw = y_damping - y_native; the REP-P replicate-mean hook
        rebuilds these frames with a label_fn (see build_delta_pairs)."""
        self.delta_scale = DeltaScale(self.stats["delta_scale"])
        self.pairs = {}
        for split in ("TRAIN", "CALIBRATION"):
            pf = build_delta_pairs(self.ds, split=split)
            if len(pf):
                pf["delta_norm"] = pf["delta_raw"].to_numpy() \
                    / self.delta_scale.at(pf["tau"].to_numpy())
            self.pairs[split] = pf

    def pair_tensors(self, split: str):
        """(g_idx, tau, delta_norm, delta_real, pre_fix) device tensors."""
        pf = self.pairs.get(split)
        if pf is None or len(pf) == 0:
            z = torch.zeros(0, device=self.device)
            return z, z.clone(), z.clone(), z.clone(), z.clone()

        def mk(col, dt=np.float32):
            return self._to_dev(pf[col].to_numpy(), dt)
        return (mk("g_idx", np.int64), mk("tau"), mk("delta_norm"),
                mk("delta_raw"), mk("pre_fix"))

    # -------------------------------------------------- member pools (B6) --
    def bootstrap_worlds(self, seed: int, member: int) -> tuple[list, list]:
        """(sampled TRAIN WORLD ids, OOB TRAIN WORLD ids) for one member.

        v0.3.2 fix #9: the bootstrap unit is the WORLD id
        ("chunk/<world>", every source of the world together -- base and
        switching arm always sampled or excluded jointly), not the
        per-source key, so a member never trains on one arm of a world
        while its other arm is OOB.  Same rng scheme as the v0.2
        trainer."""
        return bootstrap_world_ids(self.ds.src_names, self.ds.src_split,
                                   seed, member)

    def set_members(self, seed: int, members: int):
        """Each member's OOB label-row indices (all tau, both policies)
        from its bootstrap worlds (B6); world-grouped (fix #9)."""
        self.oob_rows = []
        for m in range(members):
            _, oob_worlds = self.bootstrap_worlds(seed, m)
            oob_names = {w.split("/")[-1] for w in oob_worlds}
            sel = ((self.ds.rows["split"] == "TRAIN")
                   & self.ds.rows["world"].isin(oob_names)).to_numpy()
            self.oob_rows.append(np.flatnonzero(sel))

    # ------------------------------------------------------------- gather --
    def gather_global(self, g, tau, policy, pre) -> dict:
        """Model batch at global positions with explicit conditioning;
        every block comes from the standardised device tensors (B11)."""
        if not torch.is_tensor(g):
            g = torch.as_tensor(np.ascontiguousarray(g, dtype=np.int64))
        g = g.to(self.device).long()

        def col(v, dt):
            if torch.is_tensor(v):
                return v.to(device=self.device, dtype=dt)
            return torch.as_tensor(np.ascontiguousarray(
                np.asarray(v, dtype=np.float32 if dt is torch.float32
                           else np.int64)), device=self.device).to(dt)
        tau_t = col(tau, torch.float32)
        pol_t = col(policy, torch.int64).long()
        pre_t = col(pre, torch.float32)
        summary = torch.cat(
            [self.t_feats["summary_base"].index_select(0, g),
             pol_t.to(torch.float32).unsqueeze(1),
             pre_t.unsqueeze(1)], dim=1)
        f = self.t_feats
        sel = lambda k: f[k].index_select(0, g)      # noqa: E731
        return dict(
            lane_bins=sel("lane_bins"), lane_mask=sel("lane_mask"),
            lane_geom=self.t_lane_geom,
            cav_tokens=sel("cav_tokens"), cav_mask=sel("cav_mask"),
            wall_tokens=sel("wall_tokens"), wall_mask=sel("wall_mask"),
            demand=sel("demand"), context=sel("context"),
            summary=summary, tau=tau_t, policy=pol_t)

    def gather_rows(self, idx) -> dict:
        """Model batch for label-row indices (row tau/policy/pre)."""
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(np.ascontiguousarray(
                idx, dtype=np.int64))
        idx = idx.to(self.device).long()
        return self.gather_global(self.t_row_g[idx], self.t_row_tau[idx],
                                  self.t_row_pol[idx], self.t_row_pre[idx])

    def row_targets(self, idx):
        """(y_delay_norm, y_exp_norm, y_brake_norm, coll, w) device
        tensors for label-row indices."""
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(np.ascontiguousarray(
                idx, dtype=np.int64))
        idx = idx.to(self.device).long()
        return (self.y_norm["delay"][idx], self.y_norm["exp"][idx],
                self.y_norm["brake"][idx], self.y_coll[idx],
                self.t_row_w[idx])

    def save_stats(self, path) -> None:
        Path(path).write_text(json.dumps(self.stats, indent=1),
                              encoding="utf-8")
