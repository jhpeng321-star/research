"""Post-process identity_eval outputs to committed-size artifacts:
results/full/<name>.json  (full, untracked)
results/identity_eval_<which>_slim.json  (drops dv_pairs, keeps per_eval)
results/dv_pairs_<which>.csv  (14,583 holdout pair predictions)
"""
import csv
import sys

import wd1lib as L

FULL = L.RESULTS / "full"


def slim(which):
    src = L.RESULTS / f"identity_eval_{which}.json"
    d = L.load_json(src)
    pairs = d.pop("dv_pairs", [])
    FULL.mkdir(parents=True, exist_ok=True)
    (FULL / src.name).write_text(src.read_text(encoding="utf-8"),
                                 encoding="utf-8")
    L.dump_json(L.RESULTS / f"identity_eval_{which}_slim.json", d)
    out = L.RESULTS / f"dv_pairs_{which}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["run", "t", "vid", "dv_pred_mps", "dv_true_mps",
                     "v_self_mps", "v_leader_mps", "gap_m"])
        wr.writerows(pairs)
    return len(pairs)


def main():
    n1 = slim("surviving")
    n2 = slim("retrained")
    print(f"dv pairs written: surviving={n1} retrained={n2}")
    return 0 if n1 == n2 else 1


if __name__ == "__main__":
    sys.exit(main())
