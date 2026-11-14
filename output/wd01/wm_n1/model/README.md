# WM-N1 model package

Ready-to-train PyTorch implementation of the WM-N1 world model
(`../WM_N1_DESIGN.md` sections 2-3; frozen `../REGISTRATION_WM_N1.md`).  CPU
smoke only here: unit tests on synthetic tensors plus a tiny overfit smoke.
No SUMO, no real training.

## Inputs (frozen featurizer format)

Exactly the six tensors in `../features/FORMAT.md`:

| argument | shape | dtype |
| --- | --- | --- |
| `veh_step` | `[B, K, 16]` | float |
| `veh_mask` | `[B, K]` | bool/int |
| `plan` | `[B, K, 19, 8]` | float |
| `wall` | `[B, W=4, 11]` | float |
| `wall_mask` | `[B, W]` | bool/int |
| `counter` | `[B, 2, 30]` | float |

`WmN1Net.forward` accepts **only** these names.  Forbidden inputs (`type_id`,
`speed_mode`, true penetration, `pre_fix`, world strings, seed, `t0`) are not
parameters and raise `TypeError` if passed.

## API

```python
from model.net import WmN1Net, WmN1Config, parameter_count
from model import losses
from model.rollout import rollout, project_step
from model.ensemble import WmN1Ensemble

net = WmN1Net(WmN1Config())            # ~3.9 M parameters per member
out = net(veh_step, veh_mask, plan, wall, wall_mask, counter)
# out["outcome_quantiles"]  [B, 5 tau, 3 targets, 19 quantiles]
# out["collision_logits"]   [B, 5 tau]
# out["step"]               GMM accel (K=3) + 3-way lane logits
# out["arrival"]            per-entrance NB count + insertion speed

traj = rollout(net, batch, n_steps=20, n_samples=5, seed=0)
# positions/speeds/lanes [n_samples, n_steps+1, B, K], validity counts

ens = WmN1Ensemble()                   # 5 independent-seed members
mix = ens.mixture_predictive(ens(**batch))
```

### Losses (`model/losses.py`)

`gmm_accel_nll`, `lane_change_nll`, `neg_binom_nll`, `arrival_loss`,
`outcome_pinball` / `outcome_crps` (19-quantile pinball), `collision_bce`,
`contrast_loss` (paired sibling-arm comonotone quantile difference, design
section 3 line 115, `lambda_c = 1`) and the `total_loss` aggregator.

### Rollout (`model/rollout.py`)

Autoregressive re-encoding rollout with the physical-validity layer.  Each
0.5 s step integrates position with the clipped acceleration using a
trapezoidal rule on speed (``v_new = max(0, v + a*dt)``,
``s_new = s + 0.5*(v + v_new)*dt``; design section 2.2), then applies the
validity layer: accel clipped to `[-9, +3]`, no negative speed, lane changes
only at in-bounds lanes with target gap >= 0, and a same-lane rear-bumper-gap
projection.  Every projection is counted (`ValidityCounts`).  Deterministic
under a fixed seed.

## Tests

```powershell
$env:WD_FORBID_D="1"
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest output/wd01/wm_n1/model/tests -q
```

Covers masking invariance to padded/permuted slots, validity guarantees,
contrast-loss zero/sign, rollout determinism, the forbidden-input check, the
featurizer-constant contract, and the tiny overfit smoke.
