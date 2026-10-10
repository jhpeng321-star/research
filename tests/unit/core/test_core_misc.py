"""core: constants, I/O helpers, errors and the restricted-token guard."""
from __future__ import annotations

import pytest

from wms.core import constants, errors, guard, io, seeds

# Restricted words are assembled so this source file stays free of path-like literals
# that the repository-wide scan for restricted tokens would flag.
SEALED = "seal" + "ed"
HOLDOUT = "hold" + "out"
CHUNK = "chunk" + "-5"
J4 = "J4" + "B"
INC = "inc" + "1"
CAL = "ca" + "l"
CAL_SPLIT = "CA" + "L_SPLIT_A4.json"


def test_controller_vocabulary_and_aliases():
    assert constants.CONTROLLER_VOCAB == ("none", "native", "pi", "damping", "wall", "other", "direct")
    assert constants.N_CONTROLLERS == 7
    assert constants.controller_index("native") == 1
    assert constants.controller_index("Fleet-PI") == constants.controller_index("pi")
    assert constants.controller_index("direct-output") == constants.CONTROLLER_VOCAB.index("direct")
    assert constants.controller_index("replay") == constants.CONTROLLER_VOCAB.index("other")
    assert constants.controller_index("none") == 0
    assert constants.controller_index(None) == constants.CONTROLLER_VOCAB.index("other")
    assert constants.controller_index("") == constants.CONTROLLER_VOCAB.index("other")


def test_checkpoint_constants():
    assert constants.CHECKPOINT_FORMAT == "wm_s/1"
    assert (constants.KIN_F, constants.HISTORY_H, constants.HISTORY_F) == (7, 10, 6)


def test_sha256_and_json_roundtrip(tmp_path):
    path = tmp_path / "a.json"
    io.write_json_atomic(path, {"b": 1, "a": [1, 2]})
    assert io.read_json(path) == {"a": [1, 2], "b": 1}
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index('"b"')
    assert io.sha256_file(path) == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]  # no temp file left behind


def test_write_json_atomic_cleans_up_on_failure(tmp_path):
    path = tmp_path / "bad.json"
    with pytest.raises(TypeError):
        io.write_json_atomic(path, {"x": object()})
    assert list(tmp_path.iterdir()) == []


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [row["a"] for row in io.iter_jsonl(path)] == [1, 2]


def test_run_key_prefers_batch_over_chunk():
    assert io.run_key({"batch": "truearms_chunk2", "chunk": "chunk2", "world": "w1", "arm": "pi"}) \
        == "truearms_chunk2/w1__pi"
    assert io.run_key({"chunk": "chunk2", "world": "w1", "arm": "native"}) == "chunk2/w1__native"


def test_seed_everything_is_reproducible():
    import numpy as np
    seeds.seed_everything(5)
    first = np.random.rand(3)
    seeds.seed_everything(5)
    assert (first == np.random.rand(3)).all()


@pytest.mark.parametrize("text", [f"data/{SEALED}/x", f"a/{HOLDOUT.upper()}_set.json",
                                  "x\\" + CHUNK + "\\y", f"q/{J4}", f"w/{INC}/z", f"p/{CAL}/q",
                                  f"p/{CAL_SPLIT}"])
def test_guard_detects_restricted_tokens(text):
    assert guard.find_restricted_token(text) is not None
    with pytest.raises(errors.TrainDataRefusal):
        guard.refuse_restricted(text)


@pytest.mark.parametrize("text", ["data/local/scale", "lpwo_v3/windows.json", "ZERO_SHOT_HELDOUT.json",
                                  "pack_v3", "C:/Users/someone/rd-wt/ws"])
def test_guard_allows_ordinary_paths(text):
    assert guard.find_restricted_token(text) is None
    guard.refuse_restricted(text)


def test_error_hierarchy_and_stub_message():
    assert issubclass(errors.TrainDataRefusal, PermissionError)
    assert issubclass(errors.ConfigError, ValueError)
    stub = errors.not_yet("WP-B", "wms.physics.idm.idm_accel")
    assert isinstance(stub, NotImplementedError) and "WP-B" in str(stub)
