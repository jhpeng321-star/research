"""A9 step 5: ``--extra-corpus`` is default-OFF and does not alter --corpus."""
import pytest

from dspeed import common as D


def test_parser_default_off():
    from train.idea_a import build_parser
    a = build_parser().parse_args(
        ["--member", "1", "--corpus", "X", "--out", "Y"])
    assert a.extra_corpus is None
    assert a.extra_overlay is None
    assert a.extra_identities is None


def test_parser_repeatable():
    from train.idea_a import build_parser
    a = build_parser().parse_args(
        ["--member", "1", "--corpus", "X", "--out", "Y",
         "--extra-corpus", "A", "--extra-corpus", "B",
         "--extra-overlay", "O1", "--extra-overlay", "O2",
         "--extra-identities", "I1", "--extra-identities", "I2"])
    assert a.extra_corpus == ["A", "B"]
    assert a.extra_overlay == ["O1", "O2"]
    assert a.extra_identities == ["I1", "I2"]


def test_no_extra_builds_plain_memmap_corpus():
    if not (D.PACK_ROOT / "manifest.json").is_file():
        pytest.skip("dspeed pack not built")
    from train.idea_a import IdeaACorpus
    c = IdeaACorpus(str(D.PACK_ROOT), split="TRAIN", k_max=192)
    assert not getattr(c._v1, "_multi_pack", False)
