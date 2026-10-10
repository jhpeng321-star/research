import json, sys
from pathlib import Path


def states(path):
    dec = json.JSONDecoder()
    marker = '"windows":['
    with path.open("r", encoding="utf-8") as fh:
        head = fh.read(256)
        if marker not in head:
            raise ValueError(f"{path}: no windows marker; head={head[:80]!r}")
        buf = head.split(marker, 1)[1]
        while True:
            s = buf.lstrip()
            if s.startswith("]"):
                break
            if s[:1] == ",":
                buf = s[1:].lstrip(); s = buf
            if not s:
                chunk = fh.read(1 << 20)
                if not chunk: break
                buf += chunk; continue
            try:
                rec, end = dec.raw_decode(s)
            except json.JSONDecodeError:
                chunk = fh.read(1 << 20)
                if not chunk: raise
                buf += chunk; continue
            buf = s[end:]
            yield rec


def probe(path, n=2):
    out = {"path": str(path), "size": path.stat().st_size}
    recs = []
    for i, r in enumerate(states(path)):
        if i >= n: break
        h = r["history"]
        kin = h["kin"]
        recs.append({
            "index": i,
            "keys": sorted(r.keys()),
            "history_keys": sorted(h.keys()),
            "history_steps": h.get("history_steps"),
            "history_s": h.get("history_s"),
            "history_stride_s": h.get("history_stride_s"),
            "dt_s_hist": h.get("dt_s"),
            "available_span_s": h.get("available_span_s"),
            "run_start_limited": h.get("run_start_limited"),
            "n_ids": len(h.get("ids", [])),
            "kin_r_dim": len(kin),
            "kin_c_dim": len(kin[0]) if kin else 0,
            "mask_r": len(h["mask"]),
            "target_r": len(h["target_mask"]),
            "n_initial_vehicles": len(r["initial_state"]["vehicles"]),
            "n_recorded_arms": len(r["recorded"]),
            "arm_lens": {k: len(v) for k, v in r["recorded"].items()},
        })
    out["first"] = recs
    out["count"] = sum(1 for _ in states(path))
    return out


if __name__ == "__main__":
    print(json.dumps(probe(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2), indent=1))
