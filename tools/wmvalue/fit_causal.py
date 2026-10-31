"""Fit the two frozen CPU regressors only from complete causal FIT exports."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import sklearn
from threadpoolctl import threadpool_limits

from .audit import ARMS, allowed_seed, digest, safe
from .core import MODEL_PARAMS, TIMES, Value, finite


def validate_arrays(data, corpus):
    seeds = range(969400, 969432) if corpus == "wmdata" else range(969550, 969600)
    arms = ARMS if corpus == "wmdata" else (*ARMS, "generic_random")
    # Reject forbidden split identities before preprocessing, weighting or learning.
    for seed in data["seed"]:
        allowed_seed(seed, "fit")
    cells = list(zip(data["seed"].tolist(), data["arm"].tolist(), data["time_s"].tolist()))
    expected = {(s, a, t) for s in seeds for a in arms for t in range(300, 901, 30)}
    if len(set(cells)) != len(cells) or set(cells) != expected:
        raise ValueError("incomplete/duplicate/unregistered causal FIT cells")
    n = len(cells)
    finite(data["x"], (n, 111))
    finite(data["actions"], (n, 6))
    y = finite(data["y"], (n,))
    if (y < 0).any() or (y[data["time_s"] == 900] != 0).any():
        raise ValueError("invalid real cost-to-go/terminal constraint")


def load_export(array_path, receipt_path):
    array_path, receipt_path = safe(array_path), safe(receipt_path)
    doc = json.loads(receipt_path.read_text())
    if (doc["schema"] != "wmvalue_causal_fit/1" or doc["role"] != "fit"
            or doc["corpus"] not in ("wmdata", "wmdata2") or not doc["complete"]
            or doc["source_features"] or doc["reserved_worlds_opened"]
            or not doc["causal_flow"] or not doc["nominal_length_only"]):
        raise ValueError("causal FIT gate failed")
    for name, expected in doc["code_sha256"].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError("exporter source changed after causal audit")
    if digest(array_path) != doc["arrays_sha256"]:
        raise ValueError("causal FIT array hash mismatch")
    with np.load(array_path, allow_pickle=False) as z:
        data = {k: z[k] for k in ("seed", "arm", "time_s", "x", "y", "actions")}
    validate_arrays(data, doc["corpus"])
    return data, doc


def fit(exports, out, receipt):
    out, receipt = safe(out), safe(receipt)
    if out.exists() or receipt.exists() or not out.parent.is_dir() or not receipt.parent.is_dir():
        raise ValueError("fit outputs must be new; parents must exist")
    begun = time.monotonic()
    datasets, docs = zip(*(load_export(*pair) for pair in exports))
    if {d["corpus"] for d in docs} != {"wmdata", "wmdata2"} or len(docs) != 2:
        raise ValueError("require exactly both complete FIT corpora")
    data = {k: np.concatenate([d[k] for d in datasets]) for k in datasets[0]}
    groups = list(zip(data["seed"].tolist(), data["arm"].tolist()))
    count = Counter(groups)
    arms = {s: len({a for ss, a in groups if ss == s}) for s, _ in groups}
    weights = np.array([1/(arms[s]*count[s, a]) for s, a in groups])
    weights *= len(weights)/weights.sum()
    models = {}
    with threadpool_limits(limits=1):
        for name, with_actions in (("V", False), ("D", True)):
            x = np.c_[data["x"], data["actions"]] if with_actions else data["x"]
            value = Value(actions=with_actions)
            value.model = HistGradientBoostingRegressor(**MODEL_PARAMS).fit(x, data["y"], sample_weight=weights)
            value.fit_seeds = sorted(set(data["seed"].tolist()))
            models[name] = value
    means = {(a, t): float(data["y"][(data["arm"] == a) & (data["time_s"] == t)].mean())
             for a in ARMS for t in TIMES}
    models["E"] = means
    with out.open("xb") as stream:
        joblib.dump(models, stream)
    deltas = [abs(v) for d in docs for r in d["records"]
              for h in r["clock_differences"].values() for v in h.values()]
    source_deltas = [abs(h["source"]) for d in docs for r in d["records"]
                     for h in r["clock_differences"].values()]
    result = dict(schema="wmvalue_fit/1", real_data_fit=True, rows=len(data["y"]),
                  worlds=len(set(data["seed"].tolist())), episodes=len(count), features=111,
                  models=["V", "D", "E"], parameters=MODEL_PARAMS, sklearn_version=sklearn.__version__,
                  weighting="equal worlds, then arms, then 21 decision times", threads=1,
                  fit_seconds=time.monotonic()-begun, model_path=str(out), model_sha256=digest(out),
                  source_sha256=digest(__file__), design_sha256=digest(
                      Path(__file__).parents[2]/"docs/paper_results/WMVALUE_DESIGN.md"),
                  inputs=[dict(arrays=str(p), receipt=str(r), receipt_sha256=digest(r),
                               arrays_sha256=digest(p)) for p, r in exports],
                  max_clock_discrepancy_veh_s=max(deltas), max_source_discrepancy_veh_s=max(source_deltas),
                  fit_seeds=models["V"].fit_seeds, reserved_worlds_opened=False,
                  real_data_evaluation=False, wm_inference=False, sumo_seconds=0,
                  evaluation_status="BLOCKED_ON_MATCHED_A_AND_WM_RESTART_ADAPTER",
                  policy_training_authorized=False)
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wmdata", required=True, nargs=2, metavar=("NPZ", "RECEIPT"))
    p.add_argument("--wmdata2", required=True, nargs=2, metavar=("NPZ", "RECEIPT"))
    p.add_argument("--out", required=True)
    p.add_argument("--receipt", required=True)
    a = p.parse_args()
    fit([a.wmdata, a.wmdata2], a.out, a.receipt)
