"""Observable joint-move lifetime shared by live and imagined CPU execution."""
import torch

from .v2v_obs import V2VConfig


def in_contact(commitment, rows, cavs):
    """A missing member or lost ego contact releases the *whole* commitment."""
    ego = commitment['ego_id']
    members = set(commitment['members'])
    if ego not in cavs or not members <= cavs:
        return False
    by_id = {r['vid']: r for r in rows}
    return all(abs(by_id[v]['s_m']-by_id[ego]['s_m']) <= V2VConfig().range_m
               for v in members)


class Lifetimes:
    """Reset at each future's tick zero; cancellation is permanent within a branch.

    Test contact on the same whole-second cadence as OnlinePlanner.apply. The
    intervening half-step retains that decision, as does the live command chain.
    Only observable presence, CAV membership and position are inspected.
    """
    def __init__(self):
        self.cancelled = None

    def mask(self, state, ids, tick, actions):
        if tick == 0:
            self.cancelled = [set() for _ in actions]
        if self.cancelled is None or len(self.cancelled) != len(actions):
            raise ValueError('commitment lifetime requires a tick-zero decision')
        mask = torch.ones((len(actions), len(ids)), dtype=torch.bool,
                          device=state['s_m'].device)
        slots = {v: i for i, v in enumerate(ids)}
        for b, action in enumerate(actions):
            if 'offset_commitments' not in action:
                continue  # Legacy profiles retain their original decoder.
            mask[b] = False
            if tick % 2 == 0:
                rows = [dict(vid=v, s_m=float(state['s_m'][b, i]))
                        for i, v in enumerate(ids) if bool(state['present'][b, i])]
                cavs = {r['vid'] for r in rows if bool(state['is_cav'][b, slots[r['vid']]])}
            for c, commitment in enumerate(action['offset_commitments']):
                if tick % 2 == 0 and (tick//2 >= len(commitment['offset_profile'])
                                      or not in_contact(commitment, rows, cavs)):
                    self.cancelled[b].add(c)
                if c not in self.cancelled[b]:
                    for v in commitment['members']:
                        if v in slots:
                            mask[b, slots[v]] = True
        return mask
