"""One process per source version: all tensors/RNGs/metrics after two updates."""
import importlib.util
from pathlib import Path
import os
import random

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/c4/paircrit20261026')


def helper(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/unit' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


torch.set_num_threads(2)
plain, real = helper('test_c4_dreamer'), helper('test_longcritic')
result = {}
for name in ('plain', 'real_wm_coop'):
    plain._seed(20261026)
    trainer = plain._trainer() if name == 'plain' else real._real_wm_trainer()[0]
    rows = []
    for _ in range(2):
        row = trainer.update()
        row.pop('wall_time_s')
        rows.append(row)
    result[name] = {key: getattr(trainer, key).state_dict() for key in
        ('policy', 'group_encoder', 'critic', 'target_critic', 'actor_optimizer', 'critic_optimizer')}
    result[name]['metrics'] = rows
    result[name]['torch_rng'] = torch.get_rng_state()
    result[name]['numpy_rng'] = np.random.get_state()
    result[name]['python_rng'] = random.getstate()
torch.save(result, ROOT / ('bitid_' + os.environ['PAIRCRIT_VERSION'] + '.pt'))
