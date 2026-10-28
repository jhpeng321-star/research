"""Use the CLI's registered two-update dry-run mode, STOP after update 181."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from wms.control.policy14 import Policy14Trainer
from wms.control import policy14_cli as cli
from wms.control.dreamer import MEMORY_CONFIG_FIELDS
from wms.control.counterfactual import CounterfactualConfig
from wms.control.train_cli import _read_config

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
        result = reuse_measured_startup(function.__name__, trainer, panel)
        if result is None:
            result = function(trainer, val, panel, device)
        path.write_text(json.dumps(dict(key=key, result=result), default=str))
        return result
    return call


def reuse_measured_startup(name, trainer, panel):
    """An aborted owned run completed its startup scorers before any update.

    Only reuse those measured values when all scoring inputs match. In
    particular the six-row recipe cannot reuse the nine-row startup scores.
    """
    source = os.environ.get('POLSPEED_STARTUP_SUMMARY')
    if not source:
        return None
    summary = json.loads(Path(source).read_text())
    launch = json.loads(Path(source).with_name('LAUNCH.json').read_text())
    source_config = launch['argv'][launch['argv'].index('--config') + 1]
    cf = CounterfactualConfig(**_read_config(source_config)['counterfactual'])
    if cf.as_dict() != trainer.counterfactual.as_dict():
        return None
    if summary['resumed_from']['update_count'] != trainer.update_count or summary.get('updates'):
        return None
    before = summary['effective_config_before_updates']
    if any(value != before[key] for key, value in asdict(trainer.config).items()
           if key not in MEMORY_CONFIG_FIELDS):
        return None
    def argument(flag):
        return Path(sys.argv[sys.argv.index(flag) + 1])
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(argument('--resume')) != summary['resumed_from']['sha256']:
        return None
    member, = summary['wm_members']
    if (sha(argument('--wm-ckpt')) != member['checkpoint_sha256'] or
            sha(argument('--wm-config')) != member['config_sha256']):
        return None
    if json.loads(argument('--gate-panel').read_text()) != panel:
        return None
    print('reusing measured startup from aborted owned run', name, source, flush=True)
    if name == 'score_native_panel':
        return summary['native_curve'][0]
    if name == 'score_gate_panel':
        return summary['gate_panel']['J_before']
    raise ValueError(name)


if os.environ.get('POLSPEED_CACHE_STARTUP', '1') == '1':
    cli.score_gate_panel = cached_panel(cli.score_gate_panel)
    cli.score_native_panel = cached_panel(cli.score_native_panel)
raise SystemExit(cli.main())
