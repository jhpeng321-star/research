"""Shared constants live in ``wms.core.constants`` and are only re-exported elsewhere."""
from __future__ import annotations

import itertools

import wms.core.constants as core
import wms.data.history as data_history
import wms.model.controllers as controllers
import wms.model.history as model_history
import wms.physics.state as physics_state
from wms.config.data import history_spec as config_history_spec


def test_reexports_are_the_core_objects():
    assert controllers.CONTROLLER_VOCAB is core.CONTROLLER_VOCAB
    assert controllers.N_CONTROLLERS == core.N_CONTROLLERS == 7
    assert controllers.controller_index is core.controller_index
    assert (model_history.HISTORY_H, model_history.HISTORY_F) == (core.HISTORY_H, core.HISTORY_F)
    assert model_history.LONG_HISTORY_F == data_history.LONG_HISTORY_F == core.LONG_HISTORY_F == 9
    assert physics_state.KIN_F == core.KIN_F == 7
    assert physics_state.DEFAULT_LENGTH_M == core.DEFAULT_LENGTH_M == 5.0


def test_timeline_ids_map_into_the_vocabulary():
    assert [core.CONTROLLER_VOCAB[i] for i in controllers.TIMELINE_TO_VOCAB] == \
        ["native", "damping", "pi", "direct"]
    assert core.CONTROLLER_VOCAB[controllers.DIRECT_ID] == "direct"


def test_the_three_history_specs_agree():
    grid = itertools.product((None, 5.0, 30.0, 60.0, 120.0), (0.5, 1.0, 2.0))
    for history_s, stride in grid:
        assert (model_history.history_spec(history_s, stride)
                == data_history.history_spec(history_s, stride)
                == config_history_spec(history_s, stride))
