"""Summarize fixed schema receipts; refuse to proceed with missing telemetry."""
import json
from pathlib import Path

from .audit import ARMS, allowed_seed, digest, safe


def summarize(receipts):
    if len(receipts) != 2 or {d["corpus"] for d in receipts} != {"wmdata", "wmdata2"}:
        raise ValueError("require both registered FIT corpus receipts")
    result = []
    for doc in receipts:
        if doc["schema"] != "wmvalue_audit/1" or doc["role"] != "fit":
            raise ValueError("invalid audit schema/role")
        records = doc["records"]
        seeds = range(969400, 969432) if doc["corpus"] == "wmdata" else range(969550, 969600)
        arms = ARMS if doc["corpus"] == "wmdata" else (*ARMS, "generic_random")
        keys = [(allowed_seed(r["seed"], "fit"), r["arm"]) for r in records]
        if len(keys) != len(set(keys)) or set(keys) != {(s, a) for s in seeds for a in arms}:
            raise ValueError("audit corpus incomplete/duplicated")
        def count(key):
            return sum(bool(r.get(key, False)) for r in records)
        result.append(dict(corpus=doc["corpus"], episodes=len(records),
                           archive_exists=count("exists"),
                           through900=sum(r.get("time_end", 0) == 900 for r in records),
                           uniform_half_seconds=count("uniform_half_seconds"),
                           recorded_queue=count("recorded_queue"),
                           recorded_sensor_mask=count("recorded_sensor_mask"),
                           recorded_request_counts=count("recorded_request_counts"),
                           real_objective600_exists=count("real_objective600_exists"),
                           queue_sidecars=count("queue_sidecar_exists"),
                           observation_contracts=count("observation_contract_exists"),
                           ready=count("ready")))
    ready = all(r["ready"] == r["episodes"] for r in result)
    return dict(schema="wmvalue_status/1", verdict="READY_FOR_ADAPTER_AUDIT" if ready else "INCOMPLETE",
                reason="Missing observed source queues/request counts and sensor contract" if not ready
                else "Schema availability alone does not prove observation provenance",
                corpora=result, real_data_fit=False, real_data_evaluation=False,
                reserved_worlds_opened=False, sumo_seconds=0,
                policy_training_authorized=False,
                methods={m: dict(sign_accuracy=None, spearman=None, mae=None, ci95=None)
                         for m in ("A", "B", "C", "D", "E")})


def require_fit_ready(status):
    if status["verdict"] != "READY_FOR_ADAPTER_AUDIT":
        raise ValueError("INCOMPLETE: observation data gate failed; real fitting forbidden")


def main():
    root = safe(Path(__file__).parent / "evidence")
    paths = [root / "wmdata_schema_v2.json", root / "wmdata2_schema_v2.json"]
    receipts = [json.loads(safe(p).read_text(encoding="utf-8")) for p in paths]
    result = summarize(receipts)
    result["receipt_sha256"] = {p.name: digest(p) for p in paths}
    result["design_sha256"] = digest(Path(__file__).parents[2] / "docs/paper_results/WMVALUE_DESIGN.md")
    with safe(root / "status.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
