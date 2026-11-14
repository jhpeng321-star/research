"""Apply the frozen safety-first Gate-B label to existing v5.4 physical branches.

The label is a deterministic postprocessor and never changes a SUMO trajectory,
treatment, feature, dose, or causal-identity diagnostic.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from counterfactual.idm_gate_labels import (
    classify_idm_gate_b_case,
    contract_manifest as gate_b_contract_manifest,
)
from experiments.idm.wall_v5_4_p30_support_contract import SUPPORT_SEEDS
from experiments.io import atomic_write_json


def relabel_seed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("cases") or []
    for row in rows:
        row["safety_service_oracle"] = classify_idm_gate_b_case(row)
    payload["gate_b_label_contract"] = gate_b_contract_manifest()
    return payload


def main() -> int:
    from experiments.idm.run_idm_wall_v5_4_p30_support_pilot import runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    seed_hashes: dict[str, str] = {}
    for seed in SUPPORT_SEEDS:
        path = root / "seeds" / f"seed{seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        relabel_seed_payload(payload)
        atomic_write_json(path, payload)
        seed_hashes[str(seed)] = sha256(path.read_bytes()).hexdigest()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gate_b_label_contract"] = gate_b_contract_manifest()
    manifest["gate_b_relabel_provenance"] = {
        "physical_branches_changed": False,
        "features_changed": False,
        "dose_or_identity_changed": False,
        "seed_artifact_sha256_after_labeling": seed_hashes,
    }
    atomic_write_json(manifest_path, manifest)
    summary = runner.build_summary(root)
    atomic_write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
