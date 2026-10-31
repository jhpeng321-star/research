# S-SUMOES-RUN handoff (2026-10-31 ~14:55 local; Claude -> GLM/MAIN)

Done (branch c4/sumoes, no push): step 1 design amended to variant (b) + registered; step 2 real
evaluator + canary; step 3 WM screen on cpuQ CPUs; step 4 update 0 run. Tests: 25 CPU (ruff clean).
- Seeds 969440-969543 (96 gradient + panel 969536-969543); design.json sha256
  05b84910472db97619147b12df3bc2a28f82125ec2080ae333deb252d80105c7; guards PASS, exposure clean.
- sim-s charged 47,700 = actual 47,700 (baseline 9 episodes + update 0 44 episodes) of cap 650,000.
- Running PIDs: none (driver 2970811 baseline and 2992094 update 0 exited; screen 2988164 exited).
- Results: canary PASS (CTRL0 == ours@0 bitwise J600); update 0 REJECTED (panel +0.0002 veh-s);
  see tools/sumoes/RUN.md. Current theta = 0 (CTRL0). state.json sha 673bf406.

cpuQ layout ~/sumoes20261031: design.json, accounting.jsonl (precharge ledger, fsync), RUN_LOG.jsonl,
state.json, pytools/tools/{sumoes,ctrlarm/zero.py,speedhead/observer.py}, wmscreen/c4 (WM bundle
628ff05d from gpuA), screens/, jobs/, batches/<stamp>/ (worker specs/logs/PIDS.json), logs/.

Exact next steps per update u (all on cpuQ, `source ~/newbox_env.sh`, CUDA_VISIBLE_DEVICES=''):
1. `nice -n 19 python pytools/tools/sumoes/remote/run.py request`  (writes screens/request_uXX.json)
2. `nohup setsid nice -n 19 python -u pytools/tools/sumoes/remote/wm_screen.py <u> 24 > logs/screen_uXX.log 2>&1 &`  (~75 s)
3. `nohup setsid nice -n 19 python -u pytools/tools/sumoes/remote/run.py updates --screener file --max 1 > logs/update_uXX.log 2>&1 &`  (~3 min, 39,600 sim-s)
4. Append the update_done line to RUN.md + REGISTRATION.md; copy evidence; commit.
Stop rules: STOP file or any failure -> stop (no retries); >2/3 rejects in updates 0-9 -> the single
sigma halving (edit sigma_target only via a registered amendment, then continue).

MAIN must decide BEFORE update 1 (run paused deliberately):
(a) Adam lr 1e-3 is ~1/76 of the probe scale -> steps are invisible (panel moved 0.0002 veh-s);
    a registered lr amendment (e.g. sigma-scale step) is needed or the remaining 15 updates (594,000 sim-s) buy ~nothing.
(b) One side of every kept pair is inert (positive residual dead zone on the E3 composition), so ES can
    only learn "brake less than the probe"; consider screening on the braking side only / a negative
    bias prior, or accept a likely null result.
(c) WM screen law mismatch: the frozen WM runtime composes residuals base-relative (no caprel), SUMO
    uses forced cap-relative; trunk is the E3 C1 checkpoint 64f70af4 (not candidate-B policy_000190
    de235a78) in both WM and SUMO.
