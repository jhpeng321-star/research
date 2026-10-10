"""Tests for the continuous policy log_std init (near-coast start).

Three 600k campaigns crashed at iteration ~3 with RLlib's default std=1.0
m/s^2 (full-range jerking from step 0). The fix biases the final layer's
log_std half so the untrained policy starts at std ~ e^-2 = 0.14.
"""

import torch
from torch import nn

from models import _init_log_std_bias


def test_log_std_half_biased():
    head = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    mean_bias_before = float(head[-1].bias[0].detach())
    _init_log_std_bias(head, -2.0)
    assert float(head[-1].bias[1].detach()) == -2.0
    assert float(head[-1].bias[0].detach()) == mean_bias_before  # mean untouched


def test_multi_dim_actions():
    head = nn.Sequential(nn.Linear(4, 6))  # 3 mean + 3 log_std
    mean_bias_before = head[-1].bias[:3].detach().clone()
    _init_log_std_bias(head, -1.5)
    assert torch.allclose(head[-1].bias[:3], mean_bias_before)
    assert torch.allclose(head[-1].bias[3:], torch.full((3,), -1.5))


def test_custom_value():
    head = nn.Sequential(nn.Linear(4, 2))
    _init_log_std_bias(head, -3.0)
    assert float(head[-1].bias[1].detach()) == -3.0


def test_no_crash_on_odd_shapes():
    _init_log_std_bias(nn.Sequential(nn.ReLU()), -2.0)  # no Linear at all
    head = nn.Sequential(nn.Linear(4, 1))  # single output -> n=0, skipped
    _init_log_std_bias(head, -2.0)
