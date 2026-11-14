"""T6 CPU benchmark: batched torch physics step, B=4, K=192, one process."""
import sys, os, time
sys.path.insert(0, os.getcwd())
import torch

torch.set_num_threads(4)
from output.wd01.wm_n1.wms.physics_torch import IDMParameters, PhysicsConfig, PhysicsState, step

B, K, N_WARM, N = 4, 192, 10, 200
g = torch.Generator().manual_seed(0)
s = torch.rand((B, K), generator=g, dtype=torch.float64) * 5000.0
v = torch.rand((B, K), generator=g, dtype=torch.float64) * 20.0 + 5.0
lane = torch.randint(0, 3, (B, K), generator=g, dtype=torch.long)
active = torch.ones((B, K), dtype=torch.bool)
cav = torch.zeros((B, K), dtype=torch.bool)
st = PhysicsState(s, v, lane, active, cav)
params = IDMParameters.constant(B, K, dtype=torch.float64)
cfg = PhysicsConfig(lane_count=3)
for _ in range(N_WARM):
    st = step(st, params, config=cfg).state
t0 = time.perf_counter()
for _ in range(N):
    st = step(st, params, config=cfg).state
dt = time.perf_counter() - t0
print(f"B={B} K={K} steps={N} total_s={dt:.3f} ms_per_step={1000*dt/N:.3f} steps_per_s={N/dt:.1f}")
