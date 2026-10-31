"""Exact gap6 greedy merge with inactive iterations elided (no numerical change)."""


def gap6(state, cav):
    import torch
    from wms.core.constants import DEFAULT_LENGTH_M
    from wms.control.contract import MERGE_S0_M, MERGE_S1_M
    lane, present = state['lane'], state['present'].bool()
    s = torch.where(present,state['s_m'],torch.zeros_like(state['s_m'])).detach()
    length = state.get('length_m',torch.full_like(s,DEFAULT_LENGTH_M)).detach()
    length = torch.where(present,length,torch.zeros_like(length))
    candidates = cav & (lane == 2) & (s >= MERGE_S0_M) & (s <= MERGE_S1_M)
    count = int(candidates.sum(-1).max())
    if not count:
        return lane
    rank = torch.where(candidates,s,torch.full_like(s,-float('inf')))
    order = rank.argsort(dim=-1,descending=True,stable=True)
    assigned = lane
    # The frozen implementation loops over ALL slots. After the last candidate,
    # every scatter is an identity; removing those steps leaves identical bits.
    for i in range(count):
        ego = order[:,i:i+1]
        ds = s-s.gather(1,ego)
        target = present & (assigned == 1)
        front = torch.where(target & (ds >= 0),ds-length,torch.full_like(ds,float('inf'))).amin(-1,keepdim=True)
        rear = torch.where(target & (ds < 0),-ds-length.gather(1,ego),torch.full_like(ds,float('inf'))).amin(-1,keepdim=True)
        merge = candidates.gather(1,ego) & (front >= 6.) & (rear >= 6.)
        next_lane = torch.where(merge,torch.ones_like(ego),assigned.gather(1,ego))
        assigned = assigned.scatter(1,ego,next_lane)
    return assigned
