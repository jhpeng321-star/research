from hashlib import sha256
from pathlib import Path

from counterfactual.idm_shadow_observation_v2_contract import (
    CONFIRMATORY_GATES,
    HISTORY_CAPTURE_S,
    PRIMARY_HISTORY_S,
    PROHIBITED_INPUT_TOKENS,
    SHADOW_HORIZON_S,
    SHADOW_REPORT_TIMES_S,
    SOURCE_SHA256,
    contract_manifest,
)


def test_shadow_v2_partition_and_decisions_are_frozen():
    manifest = contract_manifest()
    assert manifest["source_seeds"] == list(range(4401, 4521))
    assert manifest["decision_steps"] == [620, 660, 680, 700]
    assert manifest["sealed_partitions"]["existing_validation"] == list(range(4601, 4661))
    assert manifest["sealed_partitions"]["final_paper"] == list(range(5001, 5061))


def test_shadow_v2_horizon_history_and_twin_method_are_frozen():
    manifest = contract_manifest()
    assert SHADOW_HORIZON_S == 10
    assert SHADOW_REPORT_TIMES_S == (1, 2, 5, 10)
    assert PRIMARY_HISTORY_S == 20
    assert HISTORY_CAPTURE_S == 60
    assert manifest["shadow"]["method"] == "deterministic_time_zero_exact_sumo_twin_replay"
    assert manifest["shadow"]["prefix_identity_required"] is True
    assert manifest["history"]["sixty_second_sequence_is_conditional"] is True


def test_shadow_v2_source_hashes_and_confirmatory_gate_are_frozen():
    assert set(SOURCE_SHA256) == {
        "label_manifest",
        "label_summary",
        "feasibility_analysis",
        "v1_diagnosis",
    }
    assert all(len(value) == 64 for value in SOURCE_SHA256.values())
    assert CONFIRMATORY_GATES["minimum_matched_auc_gain"] == 0.03
    assert CONFIRMATORY_GATES["minimum_selected_safety_rate"] == 0.90
    assert CONFIRMATORY_GATES["minimum_joint_benefit_precision"] == 0.50


def test_shadow_v2_source_files_match_the_frozen_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = {
        "label_manifest": root
        / "runs/eval/idm_wall_v5_4_predictor_development_p30_s4401_4520/manifest.json",
        "label_summary": root
        / "runs/eval/idm_wall_v5_4_predictor_development_p30_s4401_4520/summary.json",
        "feasibility_analysis": root
        / "runs/probe/idm_wall_v5_4_predictor_development_feasibility_p30_s4401_4520/analysis.json",
        "v1_diagnosis": root
        / "runs/eval/idm_wall_v5_4_observation_sufficiency_diagnosis/report.json",
    }
    for name, path in paths.items():
        assert sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256[name]


def test_shadow_v2_prohibits_outcomes_identities_and_future_data():
    required = {
        "seed",
        "vehicle_id",
        "realized_duration",
        "completion_reason",
        "release_reason",
        "outcome",
        "oracle",
        "counterfactual_effect",
        "postdecision",
    }
    assert required.issubset(set(PROHIBITED_INPUT_TOKENS))


def test_shadow_v2_manifest_digest_is_stable():
    assert contract_manifest() == contract_manifest()
    assert len(contract_manifest()["contract_sha256"]) == 64
