import json

import pytest

from tools.wmfixmerge import finalize_step10 as finalize
from tools.wmfixmerge.finalize_step10 import OUT, later_worker


def test_worker_binding_rejects_peer_output_and_checkpoint():
    argv = ["python", "-m", "tools.wmfixmerge.panel_eval_v2", "--out", str(OUT),
            "--candidate", str(OUT / "step_000020.pt"), "--name", "step20_full_s0"]
    assert later_worker(argv)
    for index, value in ((4, "/peer"), (6, str(OUT / "step_000010.pt")),
                         (8, "step10_full_s0"), (2, "peer.module")):
        changed = argv.copy()
        changed[index] = value
        assert not later_worker(changed)
    assert not later_worker(["python"])


@pytest.mark.parametrize("damage", [None, "duplicate_cell", "wrong_checkpoint", "altered_publication"])
def test_complete_panel_reassembly_rejects_invalid_receipts(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(finalize, "digest", lambda path: "candidate")
    all_rows = []
    for i in range(4):
        groups = list(range(i * 13, (i + 1) * 13))
        rows = [dict(group=g, horizon=h, arm=a, contrast=float(g + a))
                for g in groups for h in (120, 300) for a in (0, 1)]
        all_rows.append(rows)
        doc = dict(step=10, groups=groups, rows=rows,
                   producer=dict(prepared_sha256=finalize.PREPARED_SHA,
                                 base_sha256=finalize.WM2_SHA,
                                 candidate_sha256="candidate"))
        if i == 0 and damage == "duplicate_cell":
            rows[0] = rows[1].copy()
        if i == 0 and damage == "wrong_checkpoint":
            doc["producer"]["candidate_sha256"] = "other"
        (tmp_path / f"SHARD_step10_full_s{i}.json").write_text(json.dumps(doc))
    panel = finalize.assemble(all_rows, range(52))
    if damage == "altered_publication":
        panel["300"]["paired_J_MAE"] += 1
    (tmp_path / "FULL_PANEL_STEP10.json").write_text(json.dumps(panel))
    if damage is None:
        assert finalize.verify_panel(tmp_path, 10) == {120: 26.0, 300: 26.0}
    else:
        with pytest.raises(ValueError):
            finalize.verify_panel(tmp_path, 10)
