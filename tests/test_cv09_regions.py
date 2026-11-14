from output.controller_verification_v9.pi_region_fixtures import all_fixtures


def test_two_missing_pi_regions_are_independently_calculated():
    rows = all_fixtures()
    assert [row["fixture"]["name"] for row in rows] == ["PI-UNSAT-MIXED", "PI-BRAKE-ZERO"]
    assert abs(rows[0]["alpha"] - 0.03) < 1e-12
    assert rows[0]["acceleration_clipped"] is False
    assert abs(rows[0]["raw_command_mps"] - 1.4173445) < 1e-10
    assert rows[1]["alpha"] == 0.0
    assert rows[1]["raw_command_mps"] == 0.5
    assert rows[1]["issued_acceleration_mps2"] == -0.5


def test_fixture_geometry_reconstructs_bumper_gap():
    for row in all_fixtures():
        fixture = row["fixture"]
        assert abs(fixture["leader_front_m"] - 100.0 - 5.0 - fixture["gap_m"]) < 1e-12
