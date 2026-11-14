from tools.build_idm_decentralized_lower_reward_refs import DEMANDS, SEEDS, build_payload


def test_lower_reward_refs_cover_frozen_cells_and_excluded_seeds():
    payload = build_payload()
    assert payload["demands_vph"] == list(DEMANDS)
    assert payload["seeds"] == list(SEEDS)
    assert set(payload["refs"]) == {
        f"osm.dense.idm_m{demand}r400.sumocfg" for demand in DEMANDS
    }
    for row in payload["refs"].values():
        assert row["corridor_severe_braking_count_ref_per_s"] > 0.0
        assert row["corridor_acceleration_energy_sum_ref_m2ps4"] > 0.0
