from tools.analyze_idm_decentralized_activation_sufficiency import (
    _observation_hash,
    _protocol_identity,
)


def _summary(*, decision_index=1, action_set="activation_speed_ceiling_v1", demand=2200):
    return {
        "protocol": {
            "schema": "idm_decentralized_authority/1",
            "demand": demand,
            "action_set": action_set,
            "option_steps": 20,
            "minimum_dose_steps": 16,
            "safety_horizon_steps": 60,
            "service_horizon_steps": 60,
            "landmark": "risk_onset",
            "landmark_contract": {"persistence_steps": 3},
            "cav_penetration": 0.1,
            "candidate_ranking": "runway",
            "candidate_count": 2,
            "decision_index": decision_index,
            "reward_refs_sha256": "frozen-reference",
        }
    }


def test_sufficiency_pool_allows_distinct_decision_indices():
    assert _protocol_identity([_summary(decision_index=1), _summary(decision_index=2)])


def test_sufficiency_pool_allows_frozen_cross_demand_evidence():
    assert _protocol_identity([_summary(demand=2200), _summary(demand=2700)])


def test_sufficiency_pool_rejects_treatment_change():
    assert not _protocol_identity(
        [_summary(), _summary(action_set="activation_speed_ceiling_p70_v1")]
    )


def test_sufficiency_pool_rejects_reference_scale_change():
    changed = _summary()
    changed["protocol"]["reward_refs_sha256"] = "different-reference"
    assert not _protocol_identity([_summary(), changed])


def test_observation_hash_is_stable_and_value_sensitive():
    assert _observation_hash([1.0, 2.0]) == _observation_hash([1.0, 2.0])
    assert _observation_hash([1.0, 2.0]) != _observation_hash([1.0, 3.0])
