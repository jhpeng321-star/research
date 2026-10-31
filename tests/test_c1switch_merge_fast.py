import pytest


def test_exact_frozen_gap6_parity():
    torch = pytest.importorskip('torch')
    from wms.control.bridge import _merge_lane
    from tools.c1switch.merge_fast import gap6
    g = torch.Generator().manual_seed(0)
    for k in (1,2,17,64,192):
        state = dict(s_m=torch.rand(32,k,generator=g)*200-100,
                     length_m=torch.rand(32,k,generator=g)*7+2,
                     lane=torch.randint(0,3,(32,k),generator=g),
                     present=torch.rand(32,k,generator=g)>.2)
        cav = state['present'] & (torch.rand(32,k,generator=g)>.4)
        assert torch.equal(_merge_lane(state,cav),gap6(state,cav))
        state['lane'].zero_()
        assert torch.equal(_merge_lane(state,cav),gap6(state,cav))
