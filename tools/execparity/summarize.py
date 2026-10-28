"""Derived report statistics from committed replay summaries; no tape/model evaluation."""
import json
from pathlib import Path
import numpy as np


def pool(rows):
    n = sum(r["n"] for r in rows)
    return {"n":n,**{key:float(sum(r["n"]*r[key]**power for r in rows)/n)**(1./power)
                     for key,power in (("rmse",2),("mae",1))},
            "mean":float(sum(r["n"]*r["mean"] for r in rows)/n)}


def main():
    root = Path(__file__).resolve().parent
    verification = json.loads((root/"verification.json").read_text())
    parity = json.loads((root/"parity_results.json").read_text())
    rows = [w["ours"] for w in verification["worlds"].values()]
    rng = np.random.default_rng(20261028)
    indices = rng.integers(0,len(rows),(5000,len(rows)))
    ci = lambda x: {"mean":float(np.mean(x)),"ci95":np.quantile(np.asarray(x)[indices].mean(1),[.025,.975]).tolist()}
    out = {"residual_on_pooled":pool([r["residual_on"] for r in rows]),
           "residual_off_pooled":pool([r["residual_off"] for r in rows]),
           "residual_on_min":min(r["residual_on"]["q00_05_50_95_100"][0] for r in rows),
           "residual_on_max":max(r["residual_on"]["q00_05_50_95_100"][-1] for r in rows),
           "residual_on_minus_off_max":max(max(abs(x) for x in r["on_minus_off"]["q00_05_50_95_100"]) for r in rows),
           "follow_identifiable_n":sum(r["follow_identifiable_n"] for r in rows),
           "follow_identifiable_world_fraction":ci([r["follow_identifiable_n"]/r["n"] for r in rows]),
           "old_imagination_error_pooled":pool([r["old_imagination_error"] for r in rows]),
           "imagination_only_safe_accel_delta_pooled":pool([
               w["ours"]["imagination_only_safe_accel_delta"] for w in parity["worlds"].values()]),
           "parity_world_rmse":{},"realised_minus_learned_follow_world_mean":{}}
    for arm in ("ours","mappo","native"):
        key = "native_head_error" if arm=="native" else "error"
        out["parity_world_rmse"][arm] = ci([w[arm][key]["rmse"] for w in parity["worlds"].values()])
        out["realised_minus_learned_follow_world_mean"][arm] = ci([
            w[arm]["realised_minus_inferred_follow"]["mean"] for w in parity["worlds"].values()])
    out["parity_world_rmse"]["c1_residual_execution"] = ci([
        w["ours"]["residual_execution_error"]["rmse"] for w in parity["worlds"].values()])
    (root/"derived_results.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))


if __name__=="__main__": main()
