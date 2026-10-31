"""Frozen fleet inference bridge for the S-C1SWITCH2 episode runner."""
from .observations import build
from .protocol import ITERATIONS, LIBRARY, PROTOCOL, TIMES, digest


class FrozenFleetSelector:
    def __init__(self, checkpoint, expected_sha):
        if digest(checkpoint) != expected_sha:
            raise ValueError('selector checkpoint hash mismatch')
        import torch
        from .learning import Selector
        record = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (record['iteration'] != ITERATIONS or record['protocol'] != PROTOCOL or
                tuple(record['library']) != LIBRARY):
            raise ValueError('only final169 protocol-v2 selector is eligible')
        self.model = Selector()
        self.model.load_state_dict(record['model'])
        self.model.eval()
        self.model.requires_grad_(False)
        self.checkpoint_sha256 = expected_sha
        self.reset()

    def reset(self):
        self.index, self.incumbent = 0, 'native'

    def select(self, payload):
        if self.index >= len(TIMES) or payload['history'][-1]['now_s'] != TIMES[self.index]:
            raise ValueError('frozen fleet picker called off-grid or without episode reset')
        observation = build(payload, variant='fleet', incumbents={'fleet': self.incumbent})
        actions, _ = self.model.act(observation, deterministic=True)
        self.incumbent = LIBRARY[actions['fleet']]
        self.index += 1
        return self.incumbent
