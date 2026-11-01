# S-DISTILL prospective offline design — 2026-11-01

- Authority: MAIN11:47 S-DISTILL; branch c4/distill-1101.
- No new SUMO for training. Original EXAM bytes/hash manifest remain immutable.
- Fit/calibrate/select ONLY EXAM 300 states / 60 registered TRAIN worlds.
- COOPPLAN original/extension/H300 worlds are registered never-fit DEV;
  extract their exact evaluated-candidate effects for POST-FREEZE audit only.
  No model/threshold selection from these DEV outcomes without MAIN amendment.
- Never fit on previous live DEV981080..981091 or any fresh live panel world.
- Recompute inputs with tools.coopbandit.protocol.observable; verify original
  EXAM feature equality and option availability. IDs/world/cell/demand/share,
  future trajectories, safety outcomes and exact labels are never predictors.
- Primary label D600, fixed before fitting; D120/D300 retained for audit.
- Per-option available-only squared-loss GBT and small scaled MLP regressors;
  direct teacher-choice classifier is a separately reported comparator.
- Outer5/inner4 world-grouped CV; scaling, models, residual calibration and
  candidate/confidence selection fit only within each outer training partition.
- Candidate confidence grid 0/.5/.65/.8/.9; zero is unconstrained argmax mean.
  Confidence = empirical positive-effect probability from inner-OOF signed
  residuals of available options, not a simultaneous lower-bound radius.
- Rank available options by predicted mean D; use O0 if maximum mean <=0 or
  its CV-selected confidence fails. No fallback to a lower-ranked option.
- Select GBT/MLP and confidence by equal-world expected exact-label saving;
  deterministic ties prefer confidence0 then GBT. No minimum-acting pre-gate.
- Report honest outer-CV saving/world-bootstrap CI, teacher agreement including
  active-only agreement, moves, feature importance and observable triggers.
- Full-TRAIN inner OOF selects final frozen policy; audit DEV only afterwards.
- Live dispatch requires exact MAIN line S-DISTILL BUDGET APPROVED, independent
  fresh-seed exposure/dual guards and frozen source/policy/CPU receipts.
- Proposed live24 worlds: two per12cells, high-share first; full paired1200s
  episodes=57600sim-s, remaining142400 reserved for executed-group C_G twins
  including reference/full-prefix costs; no unreserved/partial twin execution.
- Fresh seed block chosen/audited only after approval; never prohibited blocks.
- Same60s cadence/t300..960, option construction/execution/safety as COOPPLAN;
  J600/J900 whole-world bootstrap10000/seed20261101, complete safety/move audit.
- Own PIDs only; no deletion/push/newenv/prohibited paths/truth/assignment reads.
- CPU-only offline work registered before fitting; no live dispatch armed.
