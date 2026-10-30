"""Zero-head comparator: clock correction and threshold-free retarget rule."""
import json
from pathlib import Path

import pytest


def _batch_rows(pair, arms, j_of, queue_of):
    rows = []
    for arm, draw in [(a, d) for a in arms for d in range(8)]:
        rows.append(dict(pair_key=pair, arm=arm, draw=draw,
                         values={str(h): dict(J=j_of(arm, h), source=.25 * j_of(arm, h),
                                              upstream=.5 * j_of(arm, h), downstream=0.0,
                                              discharge=0, queue=queue_of(arm, h))
                                 for h in (60, 90, 120, 180, 300)}))
    return rows


def _write_tree(tmp_path, real_rows, base_rows, entry_rows):
    (tmp_path / "REAL.json").write_text(json.dumps(dict(panels=dict(validation=real_rows))))
    for name, rows in (("base", base_rows), ("entry", entry_rows)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "batch_000.json").write_text(json.dumps(dict(rows=rows)))
    return dict(real=str(tmp_path / "REAL.json"), base=str(tmp_path / "base"),
                entry=str(tmp_path / "entry"))


def test_zero_head_compare_corrects_clock_and_decides_retarget(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    from tools.wmfixmerge import zero_head_compare as zc
    arms = ("native", "ours_c1", "coopoff_c1", "mappo")
    # WM2 (old clock): J has a constant 10 per row; queue grows 4 -> 8 across
    # horizons, Q0 totals are 4, so the correction removes .5*(q-4) = 0..2.
    real_rows = [dict(pair_key="k", world="w", arm=a, native=(a == "native"), queue0_total=4,
                      values={str(h): dict(J=6.0 + h + (2.0 if a != "native" else 0.0),
                                           source=1.0, upstream=3.0,
                                           downstream=1.0, discharge=0, queue=4 + i)
                              for i, h in enumerate((60, 90, 120, 180, 300))})
                 for a in arms]
    base_rows = _batch_rows("k", arms, lambda a, h: 10.0, lambda a, h: 4 + h // 90)
    # entry (fixed clock) reproduces the real contrasts exactly -> residual gone
    real_j = {a: {h: 6.0 + h + (2.0 if a != "native" else 0.0)
                  for h in (60, 90, 120, 180, 300)} for a in arms}
    entry_rows = _batch_rows("k", arms, lambda a, h: real_j[a][h], lambda a, h: 0)
    paths = _write_tree(tmp_path, real_rows, base_rows, entry_rows)
    real = json.loads(Path(paths["real"]).read_text())["panels"]["validation"]
    native = next(r for r in real if r["native"])
    table = zc.index_side(zc.load_batches(paths["base"]), {r["pair_key"]: r for r in real
                                                           if r["native"]}, True)
    for i, h in enumerate((60, 90, 120, 180, 300)):
        q_n = 4 + h // 90
        shift = .5 * (q_n - 4)
        cell = table[("k", "native", 0)][h]
        assert cell["J"] == pytest.approx(10.0 - shift), h
        assert cell["source"] == pytest.approx(2.5 - shift), h
    out = tmp_path / "ZERO_HEAD_COMPARE.json"
    import sys
    sys.argv = ["zero_head_compare", "--base-dir", paths["base"], "--entry-dir",
                paths["entry"], "--real", paths["real"], "--out", str(out)]
    zc.main()
    report = json.loads(out.read_text())
    assert report["schema"] == "wmfix_zero_head_compare/1"
    assert report["retarget"]["residual_remains"] is False
    assert report["retarget"]["gone"] is True
    assert report["horizons"]["60"]["wm2"]["mae"] > report["horizons"]["60"]["entry"]["mae"]
