# MAIN01:48 staged inference admission

Both GPUs have the registered >=7GiB spare memory at metadata inspection:
gpuA43,565MiB and gpuR8,124MiB, while peer GPU workloads remain active.
No peer PID inspection, signals or resource changes. Each own service caps its
allocator at2.5GiB and rechecks free memory immediately before loading.
Real frozen WM2 K8/full60s/six-way/continued-WALL CPU probes PASS identically
on BOTH GPUs' existing runtime installations (same sample SHA67e20ef7...).
Local final CPU suite:133PASS/1host-onlyskip in existing conda, no new environment.
Model-specific arm admission requires BOTH real GPU parity receipts and its own
inference binding, plus all common CPU/WALL/exposure/canary gates; no RL dependency.
GPU clocks include loading, validation, idle and inference, with immutable hard caps
28,800s/card WM and5,400s/card BB. No GPU restart/top-up after a failed allocation.
Common prediction objective remains explicitly road-only; evaluation full-team.
Synthetic cross-device parity must include all six K8 rows and incumbent WALL
continuation, numerical rtol/atol1e-6; never test or tune on main outcomes.
Durable content-hash relay resends immutable RPC bytes only; no scientific retries.
Release owned GPU allocations as soon as all model episodes finish, before branch
physics finishes; stop own CPUs on an owned service STOP. No WM-PHYS activation.

WM services now allocated once: gpuA342449 and gpuR761968,28800s/card hard caps.
Real synthetic fresh/continued-WALL six-way/K8 cross-GPU parity PASS, receipt
GPU_PARITY_wm.json SHA2566c64210d77708dddc9fc90fbf0eb733dc24b880f44c96b16fd08ad91de885554.
WM CPU stages292bfffe94b13ed8 pass133/1skip EACH; full physics not yet dispatched here.
BB frozen afaaa32c checkpoint and physics-sidecar hashes copied identically to own
immutable stages on both hosts; CPU six-way/full60s/continued-WALL probes PASS.
gpuR free memory after WM loading is7209MiB, too marginal for a second CUDA context
before its actual7GiB gate. Queue BB only after WM releases BOTH owned services and
free>=8GiB (resource-only context margin), then recheck actual registered admission.
The one-use BB follower will run parity/CPU gates before its own main arm, never
retry WM or BB physics/services. Its prospective allocation remains the original
79800sim-s/3GPUh; all loading/idle charged, no cap increase or replacement worlds.
