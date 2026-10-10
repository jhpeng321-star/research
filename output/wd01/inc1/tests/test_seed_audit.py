"""Seed-audit tests (membership, token search, mechanical next block)."""
from __future__ import annotations

from types import SimpleNamespace

from inc1 import constants as C
from inc1 import seeds_audit as A


def _fake_seeds_module():
    return SimpleNamespace(
        RESERVED=tuple(range(9601, 9606)),
        OTHER=tuple(range(7001, 7010)),
        NOT_INT=("x", "y"),
    )


def test_membership_and_next_free_block(tmp_path):
    surfaces = []
    claimed = tmp_path / "REGISTRATION_OTHER.md"
    claimed.write_text("world w9606 is used\n", encoding="utf-8")
    surfaces.append({"label": "other", "path": claimed})
    record = A.audit_block(start=9601, count=20,
                           seeds_module=_fake_seeds_module(), surfaces=surfaces)
    assert not record["free"]
    assert set(record["collisions"]) == {9601, 9602, 9603, 9604, 9605, 9606}
    chosen = A.choose_block(start=9601, count=20,
                            seeds_module=_fake_seeds_module(), surfaces=surfaces)
    assert chosen["seeds"][0] == 9621
    assert chosen["free"]


def test_decimal_float_is_incidental_not_a_collision(tmp_path):
    ledger = tmp_path / "experiment_history.json"
    ledger.write_text('{"note": "service +13.9618/wait"}', encoding="utf-8")
    record = A.audit_block(start=9601, count=20, seeds_module=SimpleNamespace(),
                           surfaces=[{"label": "ledger", "path": ledger}])
    assert record["free"]
    row = next(r for r in record["rows"] if r["seed"] == 9618)
    assert row["incidental_numeric"]
    assert not row["token_hits"]


def test_w_token_counts_as_collision(tmp_path):
    reg = tmp_path / "REGISTRATION_X.md"
    reg.write_text("the run w9601 was trained\n", encoding="utf-8")
    record = A.audit_block(start=9601, count=20, seeds_module=SimpleNamespace(),
                           surfaces=[{"label": "reg", "path": reg}])
    assert record["collisions"] == [9601]


def test_seed_substring_inside_a_hash_is_not_a_collision(tmp_path):
    """A 4-digit run inside a long hex hash is a hash substring, not a
    world/seed identity use; only a standalone token collides."""
    doc = tmp_path / "REGISTRATION_Y.md"
    doc.write_text(
        'state_xml_sha256 '
        '"5f227c51d1b46ac9604a5953553d9926a3efabe9558747d13174bcbd5c4a86f2"\n',
        encoding="utf-8")
    record = A.audit_block(start=9601, count=20, seeds_module=SimpleNamespace(),
                           surfaces=[{"label": "hash", "path": doc}])
    assert record["free"]
    assert record["masked_hex_runs"] >= 1
    assert not record["rows"][3]["token_hits"]      # seed 9604


def test_committed_seed_audit_is_free_and_binds_registration():
    import json

    assert C.SEED_AUDIT_PATH.is_file(), "run the seed audit first"
    payload = json.loads(C.SEED_AUDIT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == C.SEED_AUDIT_SCHEMA
    assert payload["chosen_start"] == C.SEED_BLOCK_START
    assert payload["chosen_free"] is True
    assert payload["collisions_in_first_choice"] == []
    assert payload["registration_matches"] is True
    assert payload["missing_surfaces"] == []
    assert "chunk5_test_seal.json" in payload["surfaces"]
    assert any(s.startswith("experiment_history.json(research-dense")
               for s in payload["surfaces"])
    assert any(s.startswith("EXTERNAL_WORLD_PLAN:")
               for s in payload["surfaces"])
    assert any(s.startswith("EXTERNAL_REGISTRATION:")
               for s in payload["surfaces"])
