"""SUMO-labelled cooperation moves on wz3to2 (N3 NEWMAP2, N8 COOPKINDS, PLANGEN, LOWCAV, later
WM-C verifications) as candidates for the action-response check.

One :class:`Candidate` per (situation_id, arm, variant): its start (world seed, cell, t0), the
full member specs (role + params, executable by :mod:`tools.wzqual.moves`) and the SUMO labels on
``[t0, t0+300)`` of the UNDITHERED replicate r0:

* ``T`` = J(base) - J(group)  (veh-s, positive = the move helps), every screened candidate;
* ``S`` = min_i J(solo_i) - J(group)  (synergy), candidates that reached the solos stage;
* ``solo_gain`` = J(base) - J(solo_i) per member where recorded; ``admitted`` (all replicates,
  later harm, safety), ``stage``, lower bounds and replicate lists for robustness reporting.

Sources (schemas): ``coopmwm-mined-move-v1`` (*_MOVES_B/R.jsonl, one row per replicate seed),
``newmap2-negative-v1`` (NEWMAP/PLANGEN/LOWCAV negatives; member roles from the probe's
MOMENTS.json, else the registered nm2 reconstruction), ``coopkinds-negative-v1``.
Rows are read as data; nothing here runs SUMO.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import frame as F

LABEL_GLOBS = ("NEWMAP_MOVES_B.jsonl", "NEWMAP_MOVES_R.jsonl", "NEWMAP_NEGATIVES.jsonl",
               "COOPKINDS_MOVES_B.jsonl", "COOPKINDS_MOVES_R.jsonl", "COOPKINDS_NEGATIVES.jsonl",
               "PLANGEN_MOVES_B.jsonl", "PLANGEN_MOVES_R.jsonl", "PLANGEN_NEGATIVES.jsonl",
               "LOWCAV_MOVES_B.jsonl", "LOWCAV_MOVES_R.jsonl", "LOWCAV_NEGATIVES.jsonl",
               "WMPROP_MOVES_B.jsonl", "WMPROP_MOVES_R.jsonl", "WMPROP_NEGATIVES.jsonl")
#: registered nm2 member reconstruction (nm2_plan.REL / VARIANTS) for rows without a move spec
NM2_REL = dict(release_x=F.X_B, amax=1.0)
HOLD_FAMILIES = {"closing_pair", "abreast_wall", "R_random", "wall", "pair"}


@dataclass
class Candidate:
    cid: str
    source: str
    situation_id: str
    seed: int
    cell: str
    t0: float
    arm: str
    family: str
    variant: str
    members: list
    T: Optional[float] = None
    T_lb: Optional[float] = None
    S: Optional[float] = None
    S_lb: Optional[float] = None
    solo_gain: Optional[dict] = None
    best_solo: Optional[str] = None
    stage: str = ""
    admitted: bool = False
    reps_T: list = field(default_factory=list)
    reps_S: list = field(default_factory=list)
    spec_source: str = ""

    @property
    def world(self) -> str:
        return f"{self.seed}:{self.cell}"

    @property
    def moment(self) -> str:
        return f"{self.world}@{int(round(self.t0))}"

    @property
    def member_ids(self) -> list:
        return [m["id"] for m in self.members]


def _f(x):
    return None if x is None else float(x)


def _first(d):
    if isinstance(d, dict):
        return _f(d.get("1", next(iter(d.values()), None)))
    return _f(d)


def load_moments(paths: Iterable[str | Path]) -> dict:
    """``{situation_id: {variant: [member specs]}}`` from probe MOMENTS.json files."""
    out: dict = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for m in data if isinstance(data, list) else data.values():
            sid = m.get("sid") or m.get("situation_id")
            if sid and isinstance(m.get("variants"), dict):
                out[sid] = {vid: v.get("members", []) for vid, v in m["variants"].items()}
    return out


def _nm2_reconstruct(row: dict) -> list:
    """Registered nm2 member spec: role hold, params + REL, pair_t anchor = closing-lane member."""
    params = dict(row.get("params") or {})
    members = list(row["members"])
    lanes = list(row.get("member_lanes") or [None] * len(members))
    anchor = None
    if "anchor_release_x" in params:
        anchor = next((m for m, ln in zip(members, lanes) if ln == 0), members[0])
    out = []
    for vid, ln in zip(members, lanes):
        p = dict(NM2_REL, **params)
        if anchor is not None:
            p["anchor"] = anchor
        out.append(dict(id=vid, role="hold", params=p, lane=ln))
    return out


def _spec_from_move(row: dict) -> Optional[list]:
    move = row.get("move")
    if isinstance(move, str):                 # COOPKINDS_MOVES_B stores the move as a JSON string
        try:
            move = json.loads(move)
        except ValueError:
            move = None
    if not isinstance(move, dict) or not move.get("members"):
        return None
    roles = row.get("member_roles")
    out = []
    for i, m in enumerate(move["members"]):
        role = m.get("role") or (roles[i] if roles else None)
        if role is None and row.get("family") in HOLD_FAMILIES:
            role = "hold"
        out.append(dict(id=m["id"], role=role, params=dict(m.get("params") or {}), lane=m.get("lane")))
    return out


def _members(row: dict, moments: dict) -> tuple[list, str]:
    spec = _spec_from_move(row)
    if spec:
        return spec, "row.move"
    mom = moments.get(row.get("situation_id"), {}).get(row.get("variant"))
    if mom:
        return [dict(id=m["id"], role=m.get("role", "hold"), params=dict(m.get("params") or {}),
                     lane=m.get("lane")) for m in mom], "MOMENTS.json"
    if row.get("family") in HOLD_FAMILIES and row.get("params") is not None:
        return _nm2_reconstruct(row), "nm2_reconstruction"
    return [], "missing"


def _world(row: dict) -> tuple[int, str]:
    w = row["world"]
    cell = w.get("cell") or F.cell_key(w["demand"], w["penetration"])
    return int(w["seed"]), str(cell)


def _base(row: dict, path: str, moments: dict) -> dict:
    seed, cell = _world(row)
    members, how = _members(row, moments)
    return dict(source=os.path.basename(path), situation_id=row["situation_id"], seed=seed, cell=cell,
                t0=float(row["t0"]), arm=str(row.get("arm", "B")), family=str(row["family"]),
                variant=str(row["variant"]), members=members, spec_source=how)


def load_candidates(label_dir: str | Path, *, moments_paths: Iterable[str | Path] = (),
                    extra_files: Iterable[str | Path] = ()) -> list[Candidate]:
    moments = load_moments(moments_paths)
    files = []
    for g in LABEL_GLOBS:
        files += sorted(glob.glob(str(Path(label_dir) / g)))
    files += [str(p) for p in extra_files]
    acc: dict[tuple, dict] = {}
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            schema = row.get("schema")
            key = (row["situation_id"], str(row.get("arm", "B")), str(row["variant"]))
            if schema == "coopmwm-mined-move-v1":
                c = acc.setdefault(key, dict(_base(row, path, moments), admitted=True, stage="admitted",
                                             reps={}))
                rs = row.get("replicate_summary") or {}
                if rs.get("T"):
                    c["reps_T"], c["reps_S"] = list(rs["T"]), list(rs.get("S") or [])
                if int(row.get("seed", 1)) == 1:
                    c.update(T=_first(row.get("advantage_veh_s")), T_lb=_first(row.get("adv_lower95")),
                             S=_first(row.get("synergy_veh_s")), S_lb=_first(row.get("synergy_lower95")),
                             solo_gain=row.get("solo_gain_veh_s"), best_solo=row.get("best_solo"))
            elif schema == "newmap2-negative-v1":
                c = acc.setdefault(key, _base(row, path, moments))
                reps = row.get("reps") or {}
                r1 = reps.get("1") or {}
                c.update(T=_f(row.get("advantage_veh_s")), T_lb=_f(row.get("adv_lower95")),
                         S=_f(row.get("synergy_veh_s")), S_lb=_f(row.get("synergy_lower95")),
                         solo_gain=r1.get("solo_gain"), best_solo=r1.get("best_solo"),
                         stage=str(row.get("stage", "")), admitted=bool(row.get("admitted")),
                         reps_T=[reps[k]["T"] for k in sorted(reps) if reps[k].get("T") is not None],
                         reps_S=[reps[k]["S"] for k in sorted(reps) if reps[k].get("S") is not None])
            elif schema == "coopkinds-negative-v1":
                c = acc.setdefault(key, _base(row, path, moments))
                c.update(T=_f(row.get("T")), T_lb=_f(row.get("T_lb95")), S=_f(row.get("S")),
                         S_lb=_f(row.get("S_lb95")), solo_gain=row.get("solo_gain"),
                         best_solo=row.get("best_solo"), stage=str(row.get("status", "")),
                         admitted=bool(row.get("admitted")),
                         reps_T=[x["T"] for x in (row.get("r1"), row.get("r2")) if x],
                         reps_S=[x["S"] for x in (row.get("r1"), row.get("r2")) if x])
            else:
                continue
    out = []
    for key, c in sorted(acc.items()):
        c.pop("reps", None)
        cid = "|".join(key)
        out.append(Candidate(cid=cid, **{k: c.get(k) for k in Candidate.__dataclass_fields__
                                         if k != "cid" and k in c}))
    return out


def moments_of(cands: list[Candidate]) -> dict:
    out: dict = {}
    for c in cands:
        out.setdefault(c.moment, []).append(c)
    return out
