"""A77 host identity: new sanity rows vs the stored A65 sanity_pre4 rows (same panel cb64b674, D2800_p60, native + u0).

    python compare_rows.py <stored rows.jsonl> <new rows.jsonl> <out.json>
PASS iff the same 24 (world, arm) rows exist and every field except the host-specific policy path is exactly equal
(numbers compared with ==, max abs diff 0). Exit 0 on PASS, 3 on FAILED.
"""
import json
import sys
from pathlib import Path

IGNORE = {"policy"}


def load(p):
    rows = [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    return {(r["world"], r["arm"]): r for r in rows}


def main():
    a, b, out = load(sys.argv[1]), load(sys.argv[2]), Path(sys.argv[3])
    diffs, maxabs = [], 0.0
    for key in sorted(set(a) | set(b)):
        ra, rb = a.get(key), b.get(key)
        if ra is None or rb is None:
            diffs.append(dict(row=list(key), missing="stored" if ra is None else "new"))
            continue
        for k in sorted((set(ra) | set(rb)) - IGNORE):
            va, vb = ra.get(k), rb.get(k)
            if va != vb:
                d = abs(va - vb) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None
                maxabs = max(maxabs, d) if d is not None else float("inf")
                diffs.append(dict(row=list(key), field=k, stored=va, new=vb))
    ok = not diffs and len(a) == 24
    doc = dict(schema="a77.host_identity.v1", verdict="PASS" if ok else "FAILED", rows=len(b), stored_rows=len(a),
               max_abs_diff=maxabs, differences=diffs[:50],
               rule="all fields except the policy path equal on all 24 rows (12 worlds x native/u0, D2800_p60)")
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: doc[k] for k in ("verdict", "rows", "max_abs_diff")}))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
