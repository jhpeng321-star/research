"""Audit reuse keys, without interpreting unconditional tails as late restarts."""
import argparse
import json

from .audit import ARMS, digest, safe
from .core import TIMES


def audit_predictions(root):
    root = safe(root)
    files = sorted(safe(p) for p in root.glob("*/w*_k*.json"))
    cells, sources = set(), {}
    for path in files:
        d = json.loads(path.read_text())
        seed, t0 = int(d["metadata"]["seed"]), int(d["metadata"]["t0"])
        if not 969432 <= seed <= 969439:
            continue
        future = int(d["future"])
        if t0 not in TIMES or not 0 <= future < 8 or str(900-t0) not in d["costs"]:
            continue
        sources[str(path)] = digest(path)
        for arm in d["arms"]:
            cells.add((seed, t0, arm, future))
    expected = {(s, t, a, f) for s in range(969432, 969440)
                for t in TIMES for a in ARMS for f in range(8)}
    matched = cells & expected
    complete = [(s, t, a) for s in range(969432, 969440) for t in TIMES
                for a in ARMS if all((s, t, a, f) in matched for f in range(8))]
    return dict(schema="wmvalue_A_coverage/1", root=str(root), sources=sources,
                expected_futures=len(expected), matched_futures=len(matched),
                expected_arm_cells=200, complete_arm_cells=len(complete),
                missing_cells=[dict(seed=s, t0=t, arm=a) for s in range(969432, 969440)
                               for t in TIMES for a in ARMS if (s, t, a) not in complete],
                observed_start_times=sorted({t for _, t, _, _ in cells}),
                observed_arms=sorted({a for _, _, a, _ in cells}),
                verdict="READY" if len(complete) == 200 else "INCOMPLETE",
                note="Coverage only; model/CRN/cost parity must also pass before reuse.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    doc = audit_predictions(a.root)
    with safe(a.out).open("x") as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("sources", "missing_cells")}))
