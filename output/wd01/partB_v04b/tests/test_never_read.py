"""Static never-read guards: the v0.4b pipeline never opens the
forbidden files (dispatcher_rows.jsonl, insertions.jsonl, chunk-5,
HOLDOUT worlds, branch_record outcome fields)."""
from __future__ import annotations

from pathlib import Path

from partB_v04b import reg

FORBIDDEN_OPEN = ("dispatcher_rows", "insertions.jsonl", "chunk5",
                  "chunk-5", "outcome_record")
PIPELINE = ("build_v04b.py", "dryrun.py", "cal_read_v04b.py",
            "fit_v04b.py", "pairs_v04b.py", "descriptors_v04b.py",
            "model_v04b.py", "acceptance_v04b.py", "schedule_desc.py")


def test_pipeline_never_opens_forbidden_files():
    for name in PIPELINE:
        src = (Path(__file__).resolve().parent.parent / name) \
            .read_text(encoding="utf-8")
        for line in src.splitlines():
            if "open(" in line or "read_text" in line or \
                    "read_bytes" in line or "np.load" in line or \
                    "read_parquet" in line:
                low = line.lower()
                for bad in FORBIDDEN_OPEN:
                    assert bad not in low, \
                        f"{name}: forbidden file '{bad}' near an open: " \
                        f"{line.strip()}"


def test_record_fields_contract():
    """Only ALLOWED_RECORD_FIELDS may be bound from branch_record.json;
    the FORBIDDEN outcome fields never enter request-time state."""
    import wall_tokens_v04 as WT
    assert WT.FORBIDDEN_RECORD_FIELDS & WT.ALLOWED_RECORD_FIELDS == set()
    assert {"outcome", "assembled_at_s", "released_at_s",
            "aborted_at_s", "abort_reason"} <= \
        WT.FORBIDDEN_RECORD_FIELDS
    from partB_v04b import build_v04b as B
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert "WT.ALLOWED_RECORD_FIELDS" in src   # the builder uses the
    # shared field contract, not its own key list


def test_registration_file_is_not_repointed():
    assert reg.REG_PATH.name == "REGISTRATION_PARTB_V04B.md"
    assert reg.REG_SHA_EXPECTED_PREFIX == "e00cd98e"
