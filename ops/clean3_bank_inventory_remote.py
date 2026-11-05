"""Read the actual F3 TRAIN readers; no policy load, reservation or physics."""
from datetime import datetime, timezone
import json
from pathlib import Path
from tools.coopmwm import launch_training as training, launch_production as production
from tools.coopmwm import bank_breadth as breadth, bank_v2v, moves
from tools.coopmwm.protocol import digest

root = Path('/home/ubuntu/coopmwm20261105_clean2')
cfg = training.checked_config(root/'CONFIG.json')
assert (root/'STOPPED_BY_OWNER.json').is_file()
native = production.backend_config(cfg)
rows = []
for arm in ('B', 'R'):
    def consume(*, cfg, receipt, example, plan, pairs, trace):
        item = dict(arm=arm, verification_id=receipt['verification_id'],
            example_id=(example or {}).get('example_id'), world=receipt['episode']['world'],
            native_members=plan['members'], t0=receipt['t0'])
        try:
            snap, binding = breadth.verified_restart(cfg['run_root'], arm, receipt, example, plan)
            macro = moves.from_plan(plan)
            frames = bank_v2v.trace_frames(trace)
            frame = frames.get(snap['t0'])
            item.update(macro=macro.to_dict(), actors=list(macro.actors),
                window=receipt['window'], policy=snap['policy'],
                weight_keys=list(snap['weights'])[:8],
                snapshot_episode=snap['episode'], sumo=snap['sumo'],
                observable_t0=frame['vehicles'] if frame else [],
                source_sha256=digest(plan), bound=True)
        except (ValueError, KeyError, TypeError) as error:
            item.update(bound=False, reason=str(error))
        rows.append(item)
    breadth.visit(native, arm, consume)
print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
    status='READ_ONLY_BANK_INVENTORY', rows=rows, new_sim_s=0), allow_nan=False))
