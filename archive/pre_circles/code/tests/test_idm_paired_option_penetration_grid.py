from pathlib import Path

from tools.run_idm_paired_option_penetration_grid import cell_command, cell_output


def test_grid_cell_command_freezes_paired_option_contract(tmp_path):
    command = cell_command(
        python=Path("python.exe"),
        repo=tmp_path,
        output=tmp_path / "out",
        demand=2000,
        penetration=0.25,
        decision_index=1,
        seeds=[4121, 4122],
        workers=4,
        reward_refs=tmp_path / "refs.json",
    )
    joined = " ".join(command)
    assert "--action-set activation_speed_ceiling_v1" in joined
    assert "--safety-horizon-steps 60" in joined
    assert "--cav-penetration 0.25" in joined
    assert "--candidate-count 1" in joined
    assert "--landmark risk_onset" in joined


def test_grid_output_names_include_penetration_demand_and_recurrence(tmp_path):
    assert cell_output(tmp_path, 2100, 0.05, 2).name == (
        "idm_paired_option_grid_p05_m2100_l2_v1"
    )
