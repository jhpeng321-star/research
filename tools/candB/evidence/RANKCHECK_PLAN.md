# S-RANKCHECK measurement plan — 2026-10-30
- Base `4fa7cd62`, owner MAIN 17:30; R2 registration: `E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md:4951-4956`.
- WM-only, no updates/SUMO; WM2 b8797242, candidate-B exact loss, mix ON, CTRL0, paired K8, 90/300 s, 16 consecutive actual TRAIN-sampler starts minimum.
- Arms native/CTRL0/aecd7d70 native lanes/CIRCLES/PI/current/fresh-zero B plus cpuQ speed-only partial export.
- cpuQ latest completed export at first inspection (17:29): `/home/ubuntu/mappospeed/production2/policy_00025.pt`; copied read-only to `E:/Temp/opencode/rankcheck_mappospeed_policy_00025.pt`.
- Export SHA256 `69e414e53e7f520bbd112ff292aa207d3da6388d73c3ac91e894079dfe40a543`, iteration25 (partial; target169), native lanes, no lane-head weights; unchanged WM loader verified locally.
- gpuA first nvidia-smi 17:29:14: 49,140 MiB total, 25,612 MiB used, existing PID938487 untouched. Our allocator fraction <=.40, serial inference only.
- Fresh source staging via `tools/candB/stage.sh`; explicit measurement-only mode cannot be consumed by launch verification. Historical mismatched CCT-2 remains unqualified.
- WORLD-cluster bootstrap 5,000 draws, seed20261030; equal-world means after averaging repeated starts/futures within world; all sampled starts retained.
- Final report: `docs/paper_results/RANKCHECK.md`, <=70 lines, physical cost and actor loss separately, R2 decision and exact published SUMO sources.
