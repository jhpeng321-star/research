"""Use the CLI's registered two-update dry-run mode, STOP after update 181."""
import runpy

import torch
from wms.control.policy14 import Policy14Trainer

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
runpy.run_module('wms.control.policy14_cli', run_name='__main__')
