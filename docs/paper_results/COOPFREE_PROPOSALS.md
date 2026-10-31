# S-COOPFREE proposal interface for S-COOPPLAN

The corrected learned proposal checkpoint appears after a complete optimizer update
(156 windows across24 TRAIN worlds, exact HOLD baselines; exact member-solo credit
for66 whole windows within the residual cap). Legacy MAPPO-credit checkpoints are
retained remotely but must not be represented as the corrected learner.
`HANDOFF.md` lists each completed checkpoint and its SHA256. The read-only publisher
polls the existing run; it does not dispatch SUMO, change sources or restart failures.

## Artifacts and verification
- Corrected local exports: `E:/agent-work/c4-coopfree/tools/coopfree/publications/hold/`.
- Original runtime: cpuQ `/home/ubuntu/coopfree20261101_build5`.
- `policy_uNNN.pt`: format `coopfree/1`, keys `policy`, `optimizer`, `update`,
  `sigma`, `enlargements`. Load with `torch.load(...,map_location='cpu',weights_only=True)`.
- Verify against `UPDATE_NNN.json.checkpoint_sha256` and `MANIFEST.json.files`.
- Corrected remote filenames are `policy_hold_u001.pt` / `HOLD_UPDATE_001.json`;
  the `hold/` publisher aliases them to `policy_u001.pt` / `UPDATE_001.json` with identical bytes.
- The update receipt includes all156 sampled contexts/actions and HOLD-referenced
  D_G at120/300/600s; exact D_i/C_G only where new complete mean-background solos exist.
  `exact_credit_windows`/`exact_credit_worlds` identify diagnostic coverage explicitly.
  Issued/executed differences are remeasured against HOLD; safety audits remain actual.
- `BINDING.json` and `CPU_APPROVED.json` bind the exact scientific source/runtime.
- `HOLD_CPU_APPROVED.json` additionally binds the separately CPU-certified correction;
  `HOLD_AUTHORITY.json` binds MAIN01:48 and the unchanged2M cap. Continuous set-points
  alone receive score gradients; W is uniform/exogenous, avoiding baseline-dependent W bias.
- DEV/discovery files are descriptive evaluation artifacts, not TRAIN proposal inputs.

## Propose a free-form group action
Use `tools/coopfree/policy.py` and `protocol.py` from this worktree, matching the
source hashes in `BINDING.json`. In the existing Torch environment:

```python
from tools.coopfree.policy import SetPointPolicy, encode, decode
import torch

state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
policy = SetPointPolicy()
policy.load_state_dict(state['policy'])
policy.eval()
context = encode(observable_rows, current_mappo_raw_accelerations)
if context['members']:
    generator = torch.Generator().manual_seed(proposal_seed)
    sample = policy.sample(context, generator, sigma=state['sigma'])
    targets = decode(sample)
    duration_seconds = sample['window']
```

`observable_rows` contain `vid`, `is_cav`, `lane`, `s_m`, `v_mps` and optional
observable `length_m`; use the same coordinate frame as the frozen executor.
`current_mappo_raw_accelerations` is a dict keyed by CAV vid, in m/s², taken at
the executor's current **pre-safety** decision boundary (`execution.inject`).
This is the actual policy anchor; using post-safety acceleration would shift the
input distribution. Original safety is applied to the resulting set-point requests.
The policy's observable-proximity selector includes every in-zone CAV in any lane.
An empty group means no proposal. `targets[vid]` contains absolute `speed` (m/s)
and `tau` (seconds), sustained for the sampled10/20/30s window.
For a deterministic candidate, pass `deterministic=True`; several seeded stochastic
proposals provide continuous group candidates without a move library.

Execute via `requests(targets, rows)` and the unchanged safety/CommandOwnership
path (see `execution.py`/`backend.py`), with MAPPO outside the zone and after expiry.
Do not bypass the safety layer or interpret target speed as a raw actuator command.
The planner's own prediction/execution gate remains its responsibility.

## Interpretation
TRAIN exploration credit is not evidence the deterministic deployed policy improves.
Compare paired J600/J900 DEV receipts and learned-group discovery credit before claims.
The zero-update canary has exact first-command parity but99.26% changed later motion
frames from sustained holds; it is neither a learned checkpoint nor rollout parity.
