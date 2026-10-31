"""Verify the preserved real-fit artifact and report bindings, without scoring."""
import json
from pathlib import Path

import joblib
import numpy as np

from .audit import allowed_seed, digest, safe
from .fit_causal import load_export


def main():
    root = Path(__file__).parents[2]
    evidence = root/"tools/wmvalue/evidence"
    doc = json.loads((evidence/"frozen_fit.json").read_text())
    assert digest(root/"tools/wmvalue/fit_causal.py") == doc["source_sha256"]
    assert digest(root/"docs/paper_results/WMVALUE_DESIGN.md") == doc["design_sha256"]
    path = safe(root/doc["model_path"].replace("\\", "/"))
    assert digest(path) == doc["model_sha256"]
    for source in doc["inputs"]:
        assert digest(root/source["receipt"]) == source["receipt_sha256"]
        load_export(root/source["arrays"], root/source["receipt"])
    models = joblib.load(path)
    assert set(models) == {"V", "D", "E"}
    for name, n in (("V", 111), ("D", 117)):
        value = models[name]
        assert value.model.n_features_in_ == n
        assert value.fit_seeds == doc["fit_seeds"]
        for seed in value.fit_seeds:
            allowed_seed(seed, "fit")
        state = dict(time_s=900, kin=np.zeros((0, 4)), present=[], sensor_mask=[],
                     known_cav=[], coverage=np.zeros((3, 5)), flow_counts=np.zeros((2, 2)))
        assert value.predict(state, np.zeros(6) if name == "D" else None) == 0
    report = (root/"docs/paper_results/WMVALUE.md").read_text(encoding="utf-8")
    assert len(report.splitlines()) <= 60
    status = json.loads((evidence/"amended_status.json").read_text())
    coverage = json.loads((evidence/"a_coverage_final.json").read_text())
    assert status["verdict"] == "INCOMPLETE" and not status["reserved_worlds_opened"]
    assert status["model_sha256"] == doc["model_sha256"]
    assert status["local_A_coverage"]["complete_arm_start_cells"] == coverage["complete_arm_cells"]
    assert status["local_A_coverage"]["future_keys"] == coverage["matched_futures"]
    print(json.dumps(dict(artifact_hash_and_roundtrip="PASS", frozen_source_bindings="PASS",
                          both_complete_FIT_exports="PASS", report_lines=len(report.splitlines()),
                          reserved_scoring=False, model_sha256=doc["model_sha256"])))


if __name__ == "__main__":
    main()
