"""The vendored TOML-subset reader: supported syntax, strict failures, parity with tomllib."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from wms.core import toml
from wms.core.toml_subset import TomlError, loads

CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def test_scalars_and_keys():
    text = '''
# comment
a = 1
b = -2.5e1   # trailing comment
c = "x\\ty\\u00e9"
d = 'lit\\eral'
e = true
f = false
g = 1_000
"quoted key" = 3
h.i.j = 4
'''
    assert loads(text) == {"a": 1, "b": -25.0, "c": "x\tyé", "d": "lit\\eral", "e": True,
                           "f": False, "g": 1000, "quoted key": 3, "h": {"i": {"j": 4}}}


def test_special_floats():
    data = loads("a = inf\nb = -inf\nc = nan")
    assert data["a"] == math.inf and data["b"] == -math.inf and math.isnan(data["c"])


def test_tables_arrays_and_inline_tables():
    text = '''
top = [1, 2,
       3,   # comment inside
]
[a]
x = 1
[a.b]
y = ["s", "t"]
inline = { p = 1, q.r = "z" }
empty = []
[c]
z = [[1, 2], [3]]
'''
    assert loads(text) == {
        "top": [1, 2, 3],
        "a": {"x": 1, "b": {"y": ["s", "t"], "inline": {"p": 1, "q": {"r": "z"}}, "empty": []}},
        "c": {"z": [[1, 2], [3]]},
    }


def test_dotted_table_extends_implicit_parent():
    assert loads("[a.b]\nx = 1\n[a]\ny = 2") == {"a": {"b": {"x": 1}, "y": 2}}


@pytest.mark.parametrize("bad", [
    "a = ",                        # missing value
    "a = 1 2",                     # trailing garbage
    "a = 1\na = 2",                # duplicate key
    "[t]\n[t]",                    # duplicate table
    "a = 'unterminated",           # unterminated string
    'a = """multi"""',             # multi-line string unsupported
    "[[arr]]\nx = 1",              # array of tables unsupported
    "a = 1979-05-27",              # dates unsupported
    "a = 0x10",                    # hex unsupported
    "a = 007",                     # leading zeros
    "a = 1__0",                    # bad underscore
    "a = [1, 2",                   # unterminated array
    "= 1",                         # missing key
    'a = "bad \\q escape"',        # invalid escape
    "a = 1\n[a]",                  # key redefined as table
])
def test_rejects_invalid_or_unsupported(bad):
    with pytest.raises(TomlError):
        loads(bad)


def test_error_mentions_line_number():
    with pytest.raises(TomlError, match="line 3"):
        loads("a = 1\nb = 2\nc = ?")


def test_all_shipped_presets_parse_identically_with_stdlib_when_available():
    tomllib = pytest.importorskip("tomllib")
    for path in sorted(CONFIGS.glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        assert loads(text) == tomllib.loads(text), path.name


def test_facade_wraps_decode_errors_and_loads_files(tmp_path):
    file = tmp_path / "x.toml"
    file.write_text("a = 1\n", encoding="utf-8")
    assert toml.load(file) == {"a": 1}
    with pytest.raises(TomlError):
        toml.loads("a = = 1")
