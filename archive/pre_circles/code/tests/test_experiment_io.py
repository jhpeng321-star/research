import json

from experiments.io import atomic_write_json


def test_atomic_write_json_replaces_existing_file(tmp_path):
    output = tmp_path / "nested" / "result.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"stale": true}', encoding="utf-8")

    atomic_write_json(output, {"value": 3, "nan": float("nan")})

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["value"] == 3
    assert str(payload["nan"]).lower() == "nan"
    assert list(output.parent.glob("*.tmp")) == []
