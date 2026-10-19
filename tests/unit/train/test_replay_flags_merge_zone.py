"""V10ROUND1: the merge-zone replay flag needs a lane-2 HDV within 40 m of the lane end.

The synthetic tests pin the definition (HDV only, distance to ``merge_zone_s1_m`` below
``data.replay.merge_zone_end_m``, at some step).  The real-data test reports the share on
the local TRAIN-val windows (``merge_calib/data/v5_native``); it skips when the data are
not mounted.  It only reads the validation split of that source.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from wms.data.replay_flags import (MERGE_ZONE_END_M, MERGE_ZONE_S1_M,
                                   window_replay_flags)

VAL_ENV = ("WMS_DATA_MERGE_CALIB", "WMS_DATA_V5_NATIVE")
#: The assembly builds every source at the longest curriculum horizon (240 s = 481 states);
#: ``_wrap_replay`` scans exactly those window records.
HORIZON_STEPS = 481
STRIDE_S = 10.0
MAX_EPISODES = 12


def _merge_source() -> Path:
    for name in VAL_ENV:
        value = os.environ.get(name)
        if value and (Path(value) / "C4_EPISODES_MANIFEST.json").is_file():
            return Path(value)
    pytest.skip("set WMS_DATA_MERGE_CALIB (or WMS_DATA_V5_NATIVE) to the TRAIN-val source")


def test_merge_zone_flag_default_band():
    assert MERGE_ZONE_END_M == 40.0
    assert MERGE_ZONE_S1_M == pytest.approx(75.44)


def test_merge_zone_share_on_real_train_val_windows():
    from wms.data.c4_episodes import (_open_episode, episode_window_ticks,
                                      read_source_manifest)

    root = _merge_source()
    doc = read_source_manifest(root)
    entries = [e for e in doc["episodes"] if (root / e["dir"]).is_dir()]
    if not entries:
        pytest.skip(f"{root}: no local episode directories")
    entries = entries[:MAX_EPISODES]
    n = old = new = 0
    for entry in entries:
        ep = _open_episode(str(root / entry["dir"]))
        kin = np.asarray(ep.array("veh_kin"))
        mask = np.asarray(ep.array("veh_mask"), bool)
        target = np.asarray(ep.array("target_mask"), bool)
        is_cav = np.asarray(ep.array("is_cav"), bool)
        lane = kin[..., 1:4].argmax(-1)
        s = kin[..., 0]
        ticks = episode_window_ticks(ep, horizon_steps=HORIZON_STEPS, stride_s=STRIDE_S,
                                     start_s=None, warmup_s=float(entry["warmup_s"]))
        for tick in ticks:
            alive = np.cumprod(np.concatenate((mask[tick][None],
                                               target[tick:tick + HORIZON_STEPS]), 0), 0)
            alive = alive.astype(bool)
            sl = slice(tick, tick + HORIZON_STEPS + 1)
            n += 1
            if (alive & (lane[sl] == 2)).any():                     # legacy rule
                old += 1
            record = {
                "states": np.zeros(alive.shape + (7,), np.float32),
                "present": alive,
                "is_cav": is_cav[sl] & alive,
            }
            record["states"][..., 0] = s[sl]
            record["states"][..., 1:4] = kin[sl][..., 1:4]
            if window_replay_flags(record)[0]:
                new += 1
    assert n > 0
    old_share, new_share = old / n, new / n
    print(f"merge-zone share on {len(entries)} TRAIN-val episodes / {n} windows: "
          f"legacy {old_share:.3%} -> new {new_share:.3%}")
    assert new_share < old_share, "the 40 m band must flag fewer windows than 'any lane-2 HDV'"
    assert new_share < 0.98, "the new flag must not mark essentially every window"
