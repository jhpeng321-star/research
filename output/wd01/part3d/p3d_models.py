"""Part 3d models: vType posterior (B2) and lane-change hazard (B3)."""
import numpy as np
import torch
import torch.nn as nn

VTYPES = {
    0: dict(name="aggressive", tau=0.95, a_max=3.0, b=4.5, s0=1.5, sf=1.08),
    1: dict(name="normal", tau=1.1, a_max=2.6, b=4.5, s0=2.5, sf=1.00),
    2: dict(name="cautious", tau=1.5, a_max=2.0, b=3.0, s0=3.5, sf=0.92),
}
PRIOR = np.array([0.1, 0.7, 0.2])


class PosteriorNet(nn.Module):
    """p(vType | summary of last <=10 s (v, a, ds, leader gap/rel))."""

    def __init__(self, in_dim=14):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(),
                                 nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


class HazardNet(nn.Module):
    """p(change direction | features): in_dim -> 32 -> 2 logits
    (left, right)."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(),
                                 nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def save_model(path, model, meta):
    sd = {"model_state": {k: v.detach().cpu()
                          for k, v in model.state_dict().items()},
          "meta": meta}
    torch.save(sd, path)


def load_model(path, model):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd["model_state"])
    model.eval()
    return sd.get("meta", {})
