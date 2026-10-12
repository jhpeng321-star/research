"""Parity of the live SUMO -> WM-frame conversion with the frozen pack featurizer.

``verify_episode(episode_dir)`` re-featurizes the recorded ``run/physics_steps.jsonl.gz``
with the frozen WM-N1 ``features.featurizer.parse_run`` (the code every WM-S pack was
built with, ``output/wd01/wm_n1``) and checks that the episode's live slot assignment,
lane, s_down (with junction carry-forward), speed and acceleration are identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import ROOT
from .recorder import load_episode

WM_N1 = ROOT / "output" / "wd01" / "wm_n1"


def frozen_featurizer():
    if str(WM_N1) not in sys.path:
        sys.path.append(str(WM_N1))      # appended: wm_n1 also holds an old `wms` package
    from features import featurizer, geometry  # noqa: E402
    return featurizer, geometry


def verify_episode(episode_dir, *, tol: float = 5e-4) -> dict:
    """``tol``: float32 storage of |s| <= 2.7 km rounds at <= 1.2e-4 m (packs use float16)."""
    featurizer, _ = frozen_featurizer()
    episode_dir = Path(episode_dir)
    ep = load_episode(episode_dir)
    k = int(ep["present"].shape[1])
    parsed = featurizer.parse_run(episode_dir, 0.0, k_max=k,
                                  physics_path=episode_dir / "run" / "physics_steps.jsonl.gz")
    n = min(parsed["n_ticks"], ep["present"].shape[0])
    slot_vid = parsed["slot_vid"]
    vids = [str(v) for v in ep["vehicle_ids"].tolist()]
    # frozen slot -> vehicle id per tick vs live
    frozen = np.full((n, k), "", dtype=object)
    idx = parsed["slot_index"][:n]
    for t, s in zip(*np.nonzero(idx >= 0)):
        frozen[t, s] = slot_vid[idx[t, s]]
    live = np.full((n, k), "", dtype=object)
    vidx = ep["vid_index"][:n]
    for t, s in zip(*np.nonzero(vidx >= 0)):
        live[t, s] = vids[vidx[t, s]]
    slot_ok = bool((frozen == live).all())
    tick, slot = parsed["rec_tick"], parsed["rec_slot"]
    keep = tick < n
    tick, slot = tick[keep], slot[keep]
    lane = parsed["rec_lane"][keep]
    sdown = parsed["rec_sdown"][keep]
    junction = parsed["rec_junc"][keep].astype(bool)
    v = parsed["rec_v"][keep]
    a = parsed["rec_a"][keep]
    # featurizer leaves junction s at 0 in rec_sdown and carries forward in build_shard;
    # compare modelled rows directly and junction rows via the carried value.
    lane_ok = bool((ep["lane"][tick, slot] == lane).all())
    modelled = ~junction
    s_err = float(np.abs(ep["s_m"][tick, slot][modelled] - sdown[modelled]).max(initial=0.0))
    v_err = float(np.abs(ep["v_mps"][tick, slot] - v).max(initial=0.0))
    a_err = float(np.abs(ep["a_mps2"][tick, slot] - a).max(initial=0.0))
    junction_ok = bool((ep["junction"][tick, slot] == junction).all())
    # build_shard junction carry-forward (per slot, rows in (slot, tick) order)
    order = np.lexsort((tick, slot))
    carried = sdown.astype(np.float64).copy()
    last: dict = {}
    for i in order.tolist():
        if not junction[i] and lane[i] >= 0:
            last[int(slot[i])] = float(sdown[i])
        else:
            carried[i] = last.get(int(slot[i]), 0.0)
    s_carry_err = float(np.abs(ep["s_m"][tick, slot] - carried).max(initial=0.0))
    n_rows = int(len(tick))
    ok = slot_ok and lane_ok and junction_ok and max(s_err, s_carry_err, v_err, a_err) <= tol
    return dict(ok=ok, n_ticks=int(n), n_vehicle_rows=n_rows, slot_assignment_equal=slot_ok,
                lane_equal=lane_ok, junction_equal=junction_ok, max_abs_s_err_m=s_err,
                max_abs_s_err_with_junction_carry_m=s_carry_err,
                max_abs_v_err_mps=v_err, max_abs_a_err_mps2=a_err,
                frozen_dropped_concurrent=int(parsed["n_dropped_concurrent"]),
                n_junction_rows=int(junction.sum()))
