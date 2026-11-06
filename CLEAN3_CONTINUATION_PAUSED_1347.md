# S-LAUNCH recovery — 2026-11-06 13:47 +08

Recovered branch `c4/clean3-launch` at `eff16027`; inspected git status/log, task pointers and continuation receipts, full `tracks_spec.md`, and REGISTRATION lines 7557–7856 before action. Keep pre-existing untracked `CLEAN3_RESUME_W4_READONLY_02.json`, `ops/clean3_inventory_admission.py`, and `ops/clean3_r12_watch.py` intact.

## Governing authority

REGISTRATION 13:00/13:01 supersedes the historical launch/resume prompt: owner requested pausing clean3 and using cpuV for measurement. Root `/home/ubuntu/coopmwm20261106_clean3_r6` remains stopped after the u1 G4 fault; no training restart is authorized by this recovery. Registered stopped state is A1/B0/R0, root charges 120,760 sim-s including 19,200 reserved G4 sim-s with zero branch physics started. These are prior operator facts pending this recovery's remote verification.

The 13:01 entry records `STOPPED_BY_OWNER.json` SHA `dfb53061922f922073bba37ce312cb8a528a5c38e54836f66fe7c61415f1a584`. REGISTRATION's final G4 entry records fix `8c1c47c3` as code only, not deployed, and explicitly says incomplete-reservation completion is not built. Keep the earlier all-example exposure discrepancy open. The lane-on baseline and W1–W3/measurement tasks already have operators; do not duplicate them.

## This recovery's scope

Use one bounded F6-multiplexed, read-only public-receipt audit to verify the pause, recorded owner incarnations, launch/config/source/smoke/gate bindings, sealed updates, and ledger. No model, SUMO, gate, qualification job, process signal, stop clearance, source deployment, budget reallocation, or replacement watcher. Then append timestamped status and commit the evidence. Existing completed qualification and runtime tests remain historical evidence, not rerun work.

## Verified 13:49:38 +08; evidence committed `50ddf329`

`ops/clean3_paused_audit.py` successfully executed its full read-only assertions on cpuV through accepted F6. All **735** deployed source files match the original manifest; immutable launch/config and original smoke/gate seals and absolute zero-event gate bindings pass. Original chain560611/start12554274 and A568096/B568097/R568098/P3monitor568099 are exited. STOP/FAULT and the exact registered owner-stop SHA remain present; AUTHORITY contains the pause entry. A u1 seal and checkpoint SHA pass; B/R have no u1 seal, all three u0 checkpoint hashes retained. No combined monitor or scheduled u2/u4/u6 result exists.

Ledger remains **118 rows / 120,760 charged sim-s**, SHA `fc3e5b51d65c94c3e5992a13f6e5f65fe952f12dff9ab4838bc63e461432773a`, matching the operator's 13:01 stopped-state evidence. G4 has **16 jobs / 19,200 reserved sim-s / no RESULT files**; this is not a measured synergy result or permission to recycle its reservation. Fault is B's frozen-runtime guard, not a newly established collision/teleport finding. The existing G4 fix remains undeployed and its reserved-attempt completion path remains unbuilt. The all-example exposure finding also remains open.

Receipts `CLEAN3_PAUSED_PROGRESS_1349.json` (13:48:08 +08) and `CLEAN3_OWNER_PAUSED_AUDIT_1352.json` (13:49:38 +08; embedded timestamp authoritative) preserve the evidence. No new simulation, remote write, dispatch, signal, runtime amendment, stop clearance, or cpuQ action. Current pointer `CLEAN3_OWNER_PAUSED_CURRENT.json` supersedes the older running W5/W6 snapshots. Owner pause remains governing; measurement operators retain their assignments.
