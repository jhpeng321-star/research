"""Part 3d models: (B2) vType posterior from <=10 s history; (B3)
per-direction lane-change hazard.  Small torch models, class-balanced,
saved as state_dict + plain metadata (weights_only loadable)."""
import numpy as np
import torch
import torch.nn as nn


def hist_summary(hist):
    """hist [n, 20, 3] (v, a, ds) -> [n, 11]."""
    v, a = hist[..., 0], hist[..., 1]
    return torch.stack([v[:, -1], a[:, -1], v.mean(1), v.std(1), a.mean(1),
                        a.std(1), v.min(1).values, v.max(1).values,
                        v[:, -4:].mean(1), a[:, -4:].mean(1),
                        v[:, -1] - v[:, 0]], dim=1)


class Posterior(nn.Module):
    """p(vType | 10 s history): 11 -> 32 -> 3 (logits)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(11, 32), nn.SiLU(),
                                 nn.Linear(32, 3))

    def forward(self, hist):
        return self.net(hist_summary(hist))


class Hazard(nn.Module):
    """Two binary hazards (left, right) sharing a trunk.  in_dim = 19:
    v, dist-to-merge-end, 12 slot gap/rel features, merge-zone one-hot(2),
    posterior probs(3)."""

    def __init__(self, in_dim=19):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.SiLU(),
                                 nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def save_model(path, model, meta):
    torch.save({"model_state": {k: v.detach().cpu()
                                for k, v in model.state_dict().items()},
                "meta": meta}, path)


def load_model(path, model):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd["model_state"])
    model.eval()
    return sd.get("meta", {})
