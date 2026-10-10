import json, sys
from collections import Counter
from pathlib import Path
from h60_probe import states


def scan(path):
    c = Counter()
    spans = Counter()
    for r in states(path):
        h = r["history"]
        st = int(h["history_steps"])
        c[st] += 1
        span = h.get("available_span_s")
        spans[round(float(span), 1) if span is not None else None] += 1
        c["n_states"] += 1
        c["t0_missing"] += 0
        if h.get("run_start_limited"):
            c["run_start_limited"] += 1
        n0 = len(r["initial_state"]["vehicles"])
        c["n0"] += n0
        c["n_ids_sum"] += len(h["ids"])
        if len(h["mask"][-1]) and not all(h["mask"][-1]):
            c["t0_row_incomplete"] += 1
    return c, spans


if __name__ == "__main__":
    c, spans = scan(Path(sys.argv[1]))
    print(json.dumps({"path": sys.argv[1], "counts": dict(c),
                      "spans": {str(k): v for k, v in sorted(spans.items(), key=lambda x: (x[0] is None, x[0]))}},
                     indent=1))
