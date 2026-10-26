"""Local stdlib verification of offline receipts and the <=600-word amendment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"


def main():
    fetched = json.loads((RESULTS / "FETCH_SHA256.json").read_text())
    for rel, digest in fetched.items():
        assert hashlib.sha256((RESULTS / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, rel
    summary = json.loads((RESULTS / "measure30_final/RUN_SUMMARY.json").read_text())
    assert summary["status"] == "completed" and len(summary["updates"]) == 30
    curve = summary["native_curve"]
    assert [r["update"] for r in curve] == [0, 10, 20, 30]
    assert curve[0]["policy_J"] == curve[0]["native_J"]
    assert curve[0]["delta_per_start"] == [0.] * 16
    for row in curve:
        assert row["passed"] and not row["critic_tail_included"]
        assert row["native_J"] == curve[0]["native_J"]
        assert all(p <= n for p, n in zip(row["policy_J"], row["native_J"]))
    best = min(curve[1:], key=lambda r: (r["mean_delta"], r["update"]))
    assert summary["recommended_updates_per_cycle"] == best["update"] == 30
    delivered = json.loads((RESULTS / "DELIVERED_RESCORE.json").read_text())
    assert delivered["status"] == "PASS" and delivered["result"] == best
    assert delivered["actor_updates"] == delivered["sumo_seconds"] == 0
    normalized = json.loads((RESULTS / "SOURCE_NORMALIZED_SHA256.json").read_text())
    assert normalized.keys() == delivered["delivered_runtime_source_sha256"].keys()
    for rel, digest in normalized.items():
        assert hashlib.sha256((ROOT / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == digest, rel
    tests = json.loads((RESULTS / "TESTS.json").read_text())
    assert tests["passed"] == 352 and tests["failed"] == tests["skipped"] == 0
    words = len((ROOT / "NATIVEINIT.md").read_text(encoding="utf-8").split())
    assert words <= 600, words
    print(json.dumps({"status": "PASS", "verified_receipts": len(fetched), "tests": 352,
                      "doc_words": words, "recommended_updates": 30,
                      "mean_delta_curve": [r["mean_delta"] for r in curve], "sumo_seconds": 0}))


if __name__ == "__main__":
    main()
