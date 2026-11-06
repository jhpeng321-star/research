"""Read only the explicit, unsealed scoring outputs; emit a compact, hashed snapshot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select_label(doc, label):
    selected = doc.get(label, doc)
    if not isinstance(selected, dict) or not selected.get("core") or not selected.get("table"):
        raise ValueError(f"{label}: no complete qualification structure")
    return selected


def reaction_items(doc):
    """Frozen aggregate and per-set HDV reaction sign/magnitude items, including late sign."""
    out = {}
    for section in ("core", "support"):
        for key, value in doc.get(section, {}).items():
            if key.startswith(("sign:", "signlate:", "mag:")):
                out[f"{section}/{key}"] = value
    for name, table in doc.get("table", {}).items():
        for key, value in table.items():
            if key.startswith(("sign:", "signlate:", "mag:")):
                out[f"table/{name}/{key}"] = value
    for name, table in doc.get("per_set", {}).items():
        for unit, row in table.items():
            for key in ("sign_verdict", "mag_verdict"):
                if key in row:
                    out[f"per_set/{name}/{unit}/{key}"] = row[key]
    if not out:
        raise ValueError("no HDV reaction items")
    valid = {"PASS", "FAIL", "INCONCLUSIVE", "REPORT-ONLY"}
    if any(value not in valid for value in out.values()):
        raise ValueError("incomplete reaction verdict")
    return out


def strict_ratio(doc):
    value = float(doc["fid"]["all_pairs"]["ratio"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("strict all-pair TTC ratio missing/nonfinite")
    return value


def choose_base(reference, candidate):
    ref, cand = reaction_items(reference), reaction_items(candidate)
    missing = sorted(set(ref) - set(cand))
    if missing:
        raise ValueError(f"candidate reaction items missing: {missing}")
    new_fails = sorted(key for key, value in cand.items() if value == "FAIL" and ref.get(key) != "FAIL")
    r13, r14 = strict_ratio(reference), strict_ratio(candidate)
    return {"base": "14b" if not new_fails and r14 <= r13 else "13e_b2",
            "new_reaction_fails": new_fails, "strict_ttc_ratio": {"13e": r13, "14b": r14},
             "rule": "no new HDV-reaction sign/magnitude FAIL and strict all-pair TTC ratio <= 13E"}


def choose_three(documents):
    """All three complete verdicts required; eligibility always compares against 13E."""
    order = ("13e_b2", "14b", "16")
    reference = reaction_items(documents["13e_b2"])
    ratios, failures = {}, {}
    for label in order:
        items = reaction_items(documents[label])
        if set(reference) - set(items):
            raise ValueError(f"{label}: missing reference reaction items")
        failures[label] = sorted(k for k, v in items.items() if v == "FAIL" and reference.get(k) != "FAIL")
        ratios[label] = strict_ratio(documents[label])
    eligible = [label for label in order if not failures[label]]
    best = min(eligible, key=lambda label: (ratios[label], -order.index(label)))
    return {"base": best, "strict_ttc_ratio": ratios, "new_reaction_fails": failures,
            "rule": "no new FAIL vs 13E; minimum strict all-pair TTC ratio; tie -> newer round"}


def snapshot_three(spec):
    docs, hashes, checkpoints, mtimes = {}, {}, {}, []
    for label in ("13e_b2", "14b", "16"):
        item = spec[label]
        path = Path(item["verdict"]).expanduser()
        checkpoint = Path(item["checkpoint"]).expanduser()
        for p in (path, checkpoint):
            if any(t in str(p).lower() for t in ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "vtype")):
                raise ValueError("restricted scoring path")
        docs[label] = select_label(json.loads(path.read_text()), item["label"])
        hashes[label] = digest(path)
        checkpoints[label] = digest(checkpoint)
        mtimes.append(path.stat().st_mtime)
    return {"ready": True, **choose_three(docs), "verdict_sha256": hashes, "checkpoint_sha256": checkpoints,
            "reference_checkpoint_sha256": checkpoints["13e_b2"], "verdict_mtime": max(mtimes),
            "qualifications": docs}


def snapshot(home=None):
    home = Path.home() if home is None else Path(home)
    reference = home / "actqual/runs/r13e/QUAL_r13e.json"
    choices = [home / "actqual/runs/r14b/QUAL_r14b.json",
               home / "autopilot_r10/runs/r14b/QUAL_r14b.json",
               home / "autopilot_r10/runs/r14b/score/r14b/QUAL_r14b.json"]
    candidate = next((p for p in choices if p.is_file()), None)
    if candidate is None:
        return {"ready": False, "reason": "14b qualification verdict absent"}
    ref = select_label(json.loads(reference.read_text()), "r13e")
    cand = select_label(json.loads(candidate.read_text()), "r14b")
    decision = choose_base(ref, cand)
    checkpoint = home / "autopilot_r10/runs/r14b/ck/wm_s_best.pt"
    return {"ready": True, **decision,
            "verdict_path": str(candidate), "verdict_mtime": candidate.stat().st_mtime,
            "reference_sha256": digest(reference), "candidate_sha256": digest(candidate),
            "scored_checkpoint_sha256": digest(checkpoint),
            "reference_checkpoint_sha256": digest(home / "autopilot_r10/runs/r13e/ck/wm_s_best.pt"),
            "reference": ref, "candidate": cand}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--three-spec")
    args = parser.parse_args()
    try:
        print(json.dumps(snapshot_three(json.loads(args.three_spec)) if args.three_spec else snapshot()))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"ready": False, "reason": f"incomplete scoring: {error}"}))
