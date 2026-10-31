import hashlib
import json

import pytest

from tools.rbeval.summarize_full_fleet import load_panel


def panel(tmp_path, *, repeated=False, corrupt_source=False):
    rows = [dict(seed=seed, arm='native', horizon_s=300,
        demand=dict(inserted=[1, 2], pending=[0, 0], due=[1, 2],
                    conserved=True, no_slot=0, dropped=0), merges=dict(events=[]))
        for seed in range(969400, 969432)]
    if repeated:
        rows[-1]['seed'] = 969400
    if corrupt_source:
        rows[0]['demand']['pending'][0] = 1
    files = {'COMPLETE.json': json.dumps(dict(rows=rows))}
    files.update({f"{r['seed']}.json": json.dumps(r) for r in rows})
    for name, text in files.items():
        (tmp_path/name).write_text(text)
    (tmp_path/'COLLECTION.json').write_text(json.dumps({n: hashlib.sha256((tmp_path/n).read_bytes()).hexdigest()
                                                     for n in files}))


def test_complete_panel_is_accepted(tmp_path):
    panel(tmp_path)
    assert len(load_panel(tmp_path)) == 32


def test_hash_change_is_not_silently_scored(tmp_path):
    panel(tmp_path)
    (tmp_path/'969400.json').write_text('{}')
    with pytest.raises(ValueError, match='hash differs'):
        load_panel(tmp_path)


def test_repeated_world_cannot_substitute_for_missing_world(tmp_path):
    panel(tmp_path, repeated=True)
    with pytest.raises(ValueError, match='exact FIT32'):
        load_panel(tmp_path)


def test_source_conservation_checked_in_receipts(tmp_path):
    panel(tmp_path, corrupt_source=True)
    with pytest.raises(ValueError, match='source conservation'):
        load_panel(tmp_path)


def test_mixed_horizons_cannot_be_pooled_as_one_panel(tmp_path):
    panel(tmp_path)
    complete = json.loads((tmp_path/'COMPLETE.json').read_text())
    complete['rows'][0]['horizon_s'] = 600
    (tmp_path/'COMPLETE.json').write_text(json.dumps(complete))
    manifest = json.loads((tmp_path/'COLLECTION.json').read_text())
    manifest['COMPLETE.json'] = hashlib.sha256((tmp_path/'COMPLETE.json').read_bytes()).hexdigest()
    (tmp_path/'COLLECTION.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='different arm or horizon'):
        load_panel(tmp_path)
