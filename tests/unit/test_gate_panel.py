"""S-GPUB-LOOP frozen gate panel (policy14_cli --gate-panel / --gate-panel-dump)."""
import json

import pytest

torch = pytest.importorskip("torch")

from wms.control import policy14_cli as cli  # noqa: E402


class FakeVal:
    def __init__(self):
        self.penetrations = [0.05, 0.1, 0.2, 0.4]
        self.groups = {p: [i + 100 * k for i in range(6)] for k, p in enumerate(self.penetrations)}
        self.metadata = {i: {"world": f"w96{3000 + i}m2400p10t300", "t0": 300.0 + i, "start_source": "native_v5",
                             "_entry_feat": object()}
                         for g in self.groups.values() for i in g}


def test_panel_is_stratified_deterministic_and_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "checked_path", lambda p: __import__("pathlib").Path(p))
    val = FakeVal()
    a, b = cli.build_gate_panel(val), cli.build_gate_panel(val)
    assert a == b and a["n"] == 16 and len({s["index"] for s in a["starts"]}) == 16
    per = {}
    for s in a["starts"]:
        per[s["index"] // 100] = per.get(s["index"] // 100, 0) + 1
    assert sorted(per.values()) == [4, 4, 4, 4]
    assert all("_entry_feat" not in s["meta"] for s in a["starts"])
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(a), encoding="utf-8")
    assert cli.load_gate_panel(path, val)["starts"] == a["starts"]
    val.metadata[a["starts"][0]["index"]]["t0"] = -1.0
    with pytest.raises(ValueError, match="does not match"):
        cli.load_gate_panel(path, val)


def test_panel_refuses_eval_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "checked_path", lambda p: __import__("pathlib").Path(p))
    val = FakeVal()
    panel = cli.build_gate_panel(val)
    index = panel["starts"][0]["index"]
    val.metadata[index]["world"] = "w964010m2400p10t300"
    panel["starts"][0]["meta"] = cli._json_meta(val.metadata[index])
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation seed"):
        cli.load_gate_panel(path, val)
