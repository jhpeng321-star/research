# Execution receipt (2026-10-31, UTC+08)

- Worktree E:/agent-work/c4-wmvalue, branch c4/wmvalue, clean initial HEAD f07bd01e.
- Read shared OWNER S-WMVALUE direction at REGISTRATION.md:5612 and E2 horizon
  report in E:/research-dense/docs/paper_results/E2_direction_horizon.md.
- Design commit 12685912; prefit line appended to shared ignored registration.
- Code commit 44017297, before status/report finalization.
- Existing local environment: C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe;
  Python3.10.20, numpy2.2.6, sklearn1.7.2, threadpoolctl3.6.0. No installation.
- `python -B -m unittest tools.wmvalue.test_cpu`: 14 PASS, 18.014 s.
- `ruff check tools/wmvalue`: All checks passed. The environment's `python -m
  ruff` emitted no output even for --version, so the standalone executable was
  used for the actual check; its first check caught an unused import, now fixed.
- `git diff --cached --check`: PASS before code commit.

## Remote read-only data audit

Uploaded only audit.py to existing approved corpus parent directories; verified
remote parents with `test -d` before uploads. Sequential SSH invocations used
existing /root/miniconda3/bin/python (gpuR) and
/home/ubuntu/miniconda3/bin/python (cpuQ), nice19, no model imports or subprocesses.
The audit computes archive hashes, inspects NPZ member names and loads only
time_s; trajectory archives are inspected for member names only. It does not
load vehicle length truth, meta_json, scenario files or hidden-parameter arrays.
V1 receipts audited episode schemas. V2 also checked traj_base member names.
Original receipts/code are retained, including uploaded files; nothing deleted.
All 460 audited episodes are FIT; none of seeds969432..969439 was opened.

Remote V2 files:
- cpuQ /home/ubuntu/wmdata20261031/wmvalue_audit_v2_12685912.py
- cpuQ /home/ubuntu/wmdata20261031/wmvalue_schema_audit_v2_12685912.json
- gpuR /root/wmdata2/wmvalue_audit_v2_12685912.py
- gpuR /root/wmdata2/wmvalue_schema_audit_v2_12685912.json

`python -B -m tools.wmvalue.report` => INCOMPLETE; status.json created exclusively.
The remote audit commands completed normally; there is no pending WMVALUE job.

## Source checks and bounded interpretation

- E:/agent-work/c4-wmdata/tools/wmdata/export_views.py:108-128 crops absolute
  [240,600]; 273-277 documents full600s tails in raw jobs.
- E:/agent-work/c4-speedhead/tools/speedhead/observer.py:28-33 records observed
  warmup source queue in start.pt; 90-102 collects it only while after<=warmup.
- Same observer lines37-54 accumulates source objective each step but writes
  clocks only at control elapsed120/300/600; the full running clock is not saved.
- E:/agent-work/c4-wmfixmerge/tools/wmfixmerge/prepare.py:73-94 rebuilds missing
  source quantities using seed/demand departure schedules. No such reconstructed
  quantity was used as an observable FEATURE in this experiment.
- E:/agent-work/c4-wmfixmerge/tools/wmrequal/labels.py:25-33 independently
  reproduces departure RNG (not a recorded sensor). This source was read only.
- E:/agent-work/c4-pilotdiag/src/wms/control/belief.py:724-746,760-812 contains
  CAV neighbour/RSU projection helpers. Their existence means missing raw masks
  alone is not an impossibility proof. No observation adapter was implemented or
  certified here; code requires the caller to supply one with source provenance.
- No exhaustive claim is made about arbitrary alternative source log filenames,
  or about whether an approved causal reconstruction could unblock the corpus.

## Compute

Local nvidia-smi: driver unavailable. gpuR nvidia-smi at17:38:40: 4090,
18750/24564MiB,100% utilized, five peer PIDs. gpuA at17:38:46:4090D,
30973/49140MiB,92% utilized, seven peer PIDs. Only queried metadata; no CUDA
allocation, signals, priority changes or other interactions with peer PIDs.
No SUMO, real model fit, WM inference, policy run, eval/prohibited paths or push.
