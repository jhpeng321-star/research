"""RSU own-sensor field and its use in the group signal."""
import torch

from wms.control.contract import F_RSU_CELL, GROUP_SIGNAL_DIM
from wms.control.obs import build_observations
from wms.control.policy import RsuGroupEncoder
from wms.control.rsu_field import CELLS, N_RSU_CELLS, build_rsu_field


def _state():
    s = torch.tensor([[-790.0, -760.0, 10.0, 30.0, -300.0, 500.0]])
    return {"s_m": s, "v_mps": torch.tensor([[20.0, 22.0, 2.0, 3.0, 15.0, 30.0]]),
            "a_mps2": torch.zeros_like(s), "lane": torch.tensor([[0, 0, 1, 1, 2, 0]]),
            "present": torch.ones_like(s, dtype=torch.bool),
            "is_cav": torch.tensor([[True, False, False, True, False, True]])}


def test_field_cells_and_features():
    field = build_rsu_field(_state())
    assert field.shape == (1, N_RSU_CELLS, F_RSU_CELL)
    first = CELLS.index((0, -800.0, -750.0))
    assert torch.isclose(field[0, first, 0], torch.tensor(40.0))        # 2 veh / 0.05 km
    assert torch.isclose(field[0, first, 1], torch.tensor(21.0))
    assert field[0, first, 5] == 0.5                                     # one of two is a CAV
    merge = [i for i, c in enumerate(CELLS) if c[0] == 1 and c[1] <= 10.0 < c[2]][0]
    assert field[0, merge, 3] == 1.0                                     # queue: mean v < 5
    assert field[..., 4].sum() == 3   # s=10 and s=30 share a cell; s=500 is outside the zones


def test_group_signal_depends_on_field():
    torch.manual_seed(0)
    state = _state()
    enc = RsuGroupEncoder()
    obs = build_observations(state)
    with_field = enc(obs, state["s_m"], build_rsu_field(state))
    without = enc(obs, state["s_m"], None)
    assert with_field.shape == (1, 6, GROUP_SIGNAL_DIM)
    assert not torch.allclose(with_field, without)
    assert torch.all(with_field[~obs.cav_mask] == 0)
