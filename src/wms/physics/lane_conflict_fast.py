"""Exact candidate-only execution of the existing lane-conflict arbitration.

The original loop tests present/movable/change for EVERY padded slot, causing
CPU/CUDA scalar synchronizations even when no change is proposed. Those three
conditions do not depend on the evolving resolved assignment. Filtering once
keeps the same stable front-to-back candidate order and all sequential checks.
"""
import torch


def resolve_lane_conflicts_fast(s, length, lane, proposed, present, movable, margin_m):
    resolved = torch.where(movable, lane, proposed).clone()
    order = torch.argsort(s, dim=-1, descending=True, stable=True)
    slots = torch.arange(s.shape[-1], device=s.device)
    candidates = present & movable & (proposed != lane)
    for batch in range(s.shape[0]):
        selected = order[batch][candidates[batch, order[batch]]]
        for index in selected:
            target = proposed[batch, index]
            other = present[batch] & (slots != index) & (resolved[batch] == target)
            hit = ((s[batch]-length[batch] < s[batch, index]+margin_m)
                   & (s[batch] > s[batch, index]-length[batch, index]-margin_m))
            if not (other & hit).any():
                resolved[batch, index] = target
    return resolved
