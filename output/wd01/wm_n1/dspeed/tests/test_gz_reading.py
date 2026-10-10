"""A9 step 5: gzip physics reading is identical to the plain reader."""
import gzip
import json

import v0lib

from dspeed import common as D

ROWS = [
    {"time_s": 480.5, "vehicles": {}},
    {"time_s": 481.0, "vehicles": {}},
    {"time_s": 481.5, "vehicles": {}},
]


def _write_plain(tmp_path):
    p = tmp_path / "physics_steps.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n",
                 encoding="utf-8")
    return p


def _write_gz(tmp_path):
    p = tmp_path / "physics_steps.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in ROWS) + "\n")
    return p


def test_open_physics_gz_matches_plain(tmp_path):
    plain = _write_plain(tmp_path)
    gz = _write_gz(tmp_path)
    with v0lib.open_physics(plain) as fh:
        a = [json.loads(x) for x in fh]
    with v0lib.open_physics(gz) as fh:
        b = [json.loads(x) for x in fh]
    assert a == b == ROWS


def test_first_physics_time_gz(tmp_path):
    plain = _write_plain(tmp_path)
    gz = _write_gz(tmp_path)
    assert D.first_physics_time(plain) == 480.5
    assert D.first_physics_time(gz) == 480.5
