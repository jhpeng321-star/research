# V10CAVFOLLOW: brake-follow emphasis for CAV-led events

**Status:** complete (loss implemented, unit-tested).
**Branch:** `c4/v10cavfollow` from `c4/v10int` `ba56ebf7`.
**Files:** `src/wms/train/losses/{trajectory,compose}.py`,
`src/wms/config/{train,legacy_flat}.py`, `src/wms/train/settings.py`,
`tests/unit/train/test_brakewave.py`.

## Change

`loss.brake_follow_cav_weight` (default `0.0` = OFF, bit-identical) adds an extra
per-step position/speed multiplier on HDV followers whose nearest same-lane recorded
leader is a CAV within 40 m and decelerates below -1.0 m/s^2. On flagged steps the
multiplier is `1 + brake_follow_cav_weight`; unflagged steps stay at 1. It composes
(multiplicatively) with the existing `loss.brake_follow_weight` HDV-led mask, which is
unchanged.

The mask is the existing `brake_follow_mask` with `leader_is_cav=True`; CAV-led and
HDV-led events are cached separately (`brake_follow_cav` / `brake_follow`). Default-off
does not build the mask and leaves the rollout loss byte-identical.

## Motivation

`brake_diag/BRAKE_DIAG.md`: at matched kinematics the WM HDVs brake about 0.24 m/s^2
too hard behind braking CAVs in 11/12 seeds, while being 0.05-0.15 too soft behind HDVs.
The existing brake-follow emphasis only targets HDV-led events.

## Tests

`python -m pytest tests/unit/train/test_brakewave.py -q` (7 passed):
CAV-led events are flagged on HDV followers, default-off identity, positive-weight
emphasis, flat-config roundtrip, and negative-weight rejection.
`python -m pytest tests/unit/config tests/unit/train -q` (214 passed).
`python -m ruff check` on all changed files passed.

No GPU, no SUMO, no evaluation seeds 964000-964199, no sealed/holdout/heldout/cal_split/
chunk5/j4b/inc1 files; TRAIN-val only.
