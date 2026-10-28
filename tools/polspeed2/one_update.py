"""Use the CLI's registered two-update dry-run mode, STOP after update 181."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import torch
from wms.control.policy14 import Policy14Trainer
from wms.control import policy14_cli as cli
from wms.control.dreamer import MEMORY_CONFIG_FIELDS

original = Policy14Trainer.update


def update_once(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if self.update_count == 181:
        torch.save({'policy': self.policy.state_dict(),
                    'group_encoder': self.group_encoder.state_dict(),
                    'optimizer': self.actor_optimizer.state_dict()},
                   self.output_dir / 'actor181.pt')
        (self.output_dir / 'STOP').touch()
    return result


Policy14Trainer.update = update_once


def cached_panel(function):
    def call(trainer, val, panel, device):
        # Independent scorers restore all RNG and do not update the trainer.
        # Cache their measured startup-only results for an identical actor,
        # WM, panel and scoring config. Training/update timing is unaffected.
        config = {k: v for k, v in asdict(trainer.config).items()
                  if k not in MEMORY_CONFIG_FIELDS and k != 'checkpoint_imagination'}
        key = dict(function=function.__name__, config=config,
                   counterfactual=trainer.counterfactual.as_dict(),
                   models=trainer.world_model_ids, panel=panel, update=trainer.update_count)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode())
        for module in (trainer.policy, trainer.group_encoder):
            for name, value in module.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        root = Path('/root/autodl-tmp/c4/polspeed2_20261028/score_cache')
        root.mkdir(exist_ok=True)
        path = root / (digest.hexdigest() + '.json')
        if path.exists():
            print('startup score cache hit', function.__name__, path.name, flush=True)
            return json.loads(path.read_text())['result']
        result = function(trainer, val, panel, device)
        path.write_text(json.dumps(dict(key=key, result=result), default=str))
        return result
    return call


if os.environ.get('POLSPEED_CACHE_STARTUP', '1') == '1':
    cli.score_gate_panel = cached_panel(cli.score_gate_panel)
    cli.score_native_panel = cached_panel(cli.score_native_panel)
raise SystemExit(cli.main())
