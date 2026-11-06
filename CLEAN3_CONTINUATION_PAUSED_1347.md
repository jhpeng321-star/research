# S-LAUNCH recovery — 2026-11-06 13:47 +08

Recovered branch `c4/clean3-launch` at `eff16027`; inspected git status/log, task pointers and continuation receipts, full `tracks_spec.md`, and REGISTRATION lines 7557–7856 before action. Keep pre-existing untracked `CLEAN3_RESUME_W4_READONLY_02.json`, `ops/clean3_inventory_admission.py`, and `ops/clean3_r12_watch.py` intact.

## Governing authority

REGISTRATION 13:00/13:01 supersedes the historical launch/resume prompt: owner requested pausing clean3 and using cpuV for measurement. Root `/home/ubuntu/coopmwm20261106_clean3_r6` remains stopped after the u1 G4 fault; no training restart is authorized by this recovery. Registered stopped state is A1/B0/R0, root charges 120,760 sim-s including 19,200 reserved G4 sim-s with zero branch physics started. These are prior operator facts pending this recovery's remote verification.

The 13:01 entry records `STOPPED_BY_OWNER.json` SHA `dfb53061922f922073bba37ce312cb8a528a5c38e54836f66fe7c61415f1a584`. REGISTRATION's final G4 entry records fix `8c1c47c3` as code only, not deployed, and explicitly says incomplete-reservation completion is not built. Keep the earlier all-example exposure discrepancy open. The lane-on baseline and W1–W3/measurement tasks already have operators; do not duplicate them.

## This recovery's scope

Use one bounded F6-multiplexed, read-only public-receipt audit to verify the pause, recorded owner incarnations, launch/config/source/smoke/gate bindings, sealed updates, and ledger. No model, SUMO, gate, qualification job, process signal, stop clearance, source deployment, budget reallocation, or replacement watcher. Then append timestamped status and commit the evidence. Existing completed qualification and runtime tests remain historical evidence, not rerun work.
