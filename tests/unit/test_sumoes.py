"""S-SUMOES driver skeleton: CPU tests on the deterministic stub. No SUMO anywhere."""
import json
import math

import pytest
import torch

from tools.sumoes.config import CELL_CYCLE, PANEL_CELLS, ConfigError, SumoEsConfig
from tools.sumoes.driver import (AcceptAllScreener, BudgetExceeded, BudgetLedger,
                                 MechanicalFailure, SumoEsDriver)
from tools.sumoes.sumo_eval import SumoRuntimeEvaluator
from tools.sumoes.es import (EsState, HeadAdam, estimate, head_dimension, head_vector,
                             panel_improved, sigma_for, state_dict_with_head)
from tools.sumoes.stub import RankingScreener, StubSimulator

CONFIG = SumoEsConfig(enabled=True)


def state_dict_for(dim=136, dtype=torch.float64):
    return {"residual_head.weight": torch.zeros(1, dim, dtype=dtype),
            "residual_head.bias": torch.zeros(1, dtype=dtype),
            "other": torch.zeros(3, dtype=dtype)}


def test_config_defaults_off_and_frozen_budget_identity():
    shipped = SumoEsConfig.load("configs/c4_sumoes_design.toml")
    assert shipped.enabled is False
    assert shipped.planned_episodes() == 8 + 16 * (2 * 3 * 6 + 8) == 712
    assert shipped.planned_sim_s() == 712 * 900 == 640_800
    assert shipped.cap_sim_s == 650_000
    assert shipped.cap_sim_s - shipped.planned_sim_s() == 9_200


def test_config_refuses_bad_geometry_and_blocks():
    with pytest.raises(ConfigError):
        SumoEsConfig(enabled=True, episode_charge_sim_s=800)
    with pytest.raises(ConfigError):
        SumoEsConfig(enabled=True, wm_screen_directions=2)
    with pytest.raises(ConfigError):
        SumoEsConfig(enabled=True, sigma_min_mps2=0.5)
    with pytest.raises(ConfigError):
        SumoEsConfig(enabled=True, seed_block_first=969_440, seed_block_last=969_480)
    for lo, hi in ((964_000, 964_199), (962_000, 962_199), (977_000, 977_999)):
        with pytest.raises(ConfigError):
            SumoEsConfig(enabled=True, seed_block_first=lo, seed_block_last=hi)
    with pytest.raises(ConfigError, match="WMDATA"):
        SumoEsConfig(enabled=True, seed_block_first=969_400, seed_block_last=969_503)
    with pytest.raises(ConfigError, match="exceeds cap"):
        SumoEsConfig(enabled=True, cap_sim_s=640_000)


def test_seed_schedule_panel_disjoint_and_cell_cycle():
    cfg = CONFIG
    assert cfg.gradient_seeds(0) == tuple(range(969_440, 969_446))
    assert cfg.gradient_seeds(15) == tuple(range(969_530, 969_536))
    assert cfg.panel_seeds() == tuple(range(969_536, 969_544))
    assert not set(cfg.gradient_seeds(15)) & set(cfg.panel_seeds())
    with pytest.raises(ConfigError):
        cfg.gradient_seeds(16)
    assert cfg.cell_for(969_440) == CELL_CYCLE[0]
    assert cfg.cell_for(969_441) == CELL_CYCLE[2]
    assert cfg.cell_for(969_446) == CELL_CYCLE[1]
    for update in range(16):
        cells = [cfg.cell_for(s) for s in cfg.gradient_seeds(update)]
        assert {p for p, _ in cells} == {0.05, 0.1, 0.2, 0.4}
        assert {d for _, d in cells} == {1900, 2400, 2700}
    assert len({cfg.cell_for(s) for s in cfg.gradient_seeds(0) + cfg.gradient_seeds(1)}) == 12
    assert [cfg.cell_for(s) for s in cfg.panel_seeds()] == list(PANEL_CELLS)


def test_head_vector_roundtrip_137_and_shape_guards():
    state = state_dict_for()
    assert head_dimension(state) == 137
    theta = torch.randn(137, dtype=torch.float64)
    injected = state_dict_with_head(state, theta)
    assert torch.equal(head_vector(injected), theta)
    assert injected["other"] is state["other"]
    assert injected["residual_head.weight"].dtype is torch.float64
    assert injected["residual_head.weight"].shape == (1, 136)
    float32 = state_dict_for(dtype=torch.float32)
    injected32 = state_dict_with_head(float32, theta)
    assert injected32["residual_head.weight"].dtype is torch.float32
    with pytest.raises(ValueError):
        head_vector({"residual_head.weight": torch.zeros(2, 68), "residual_head.bias":
                     torch.zeros(2)})
    with pytest.raises(ValueError):
        state_dict_with_head(state, torch.zeros(10))


def test_sigma_rule_halving_and_floor():
    sigma = sigma_for(0.10, 20.0)
    assert math.isclose(sigma, 0.005)
    with pytest.raises(ValueError):
        sigma_for(0.0, 20.0)
    adam = HeadAdam(lr=1e-3)
    state = EsState(theta=torch.zeros(137, dtype=torch.float64), adam=adam,
                    sigma_target_residual_rms_mps2=0.10, sigma_min_mps2=0.02,
                    sigma_halve_every_accepted=4, direction_seed=7_000_031)
    assert math.isclose(state.sigma(20.0), 0.005)
    state.accepted_count = 4
    assert math.isclose(state.sigma(20.0), 0.0025)
    state.accepted_count = 12
    assert math.isclose(state.sigma(20.0), 0.02 / 20.0)


def test_estimate_formula_and_panel_improved_semantics():
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    deltas = torch.tensor([3.0, -5.0])
    gradient = estimate(directions, deltas, 0.5)
    manual = ((3.0 * torch.tensor([1.0, 0.0]) - 5.0 * torch.tensor([0.0, 1.0])) / 2) / 2 / 0.5
    assert torch.equal(gradient, manual)
    assert panel_improved([10.0, 12.0], [9.0, 11.0])
    assert panel_improved([10.0, 10.0], [10.0, 10.0])
    assert not panel_improved([9.0, 9.0], [9.0, 10.0])
    assert panel_improved([10.0, 10.0], [8.0, 11.0])
    with pytest.raises(ValueError):
        panel_improved([1.0], [1.0, 2.0])


def test_stub_crn_cancels_world_offset_in_pairs():
    stub = StubSimulator(dimension=4)
    theta = torch.randn(4, dtype=torch.float64)
    plus = stub.episode(theta, 969_440, 0)["J600"]
    minus = stub.episode(theta, 969_440, 0)["J600"]
    plus_shifted = stub.episode(theta + 0.1, 969_440, 0)["J600"]
    assert plus == minus
    assert plus_shifted != plus
    offset = stub.offset(969_440)
    replay = stub.episode(theta, 969_440, 0)["J600"]
    assert math.isclose(replay - offset - float(stub.b @ theta)
                        - 0.5 * float((theta - stub.star) @ (theta - stub.star)), 0.0,
                        abs_tol=1e-9)


def driver_for(config=CONFIG, simulator=None, screener=None):
    simulator = simulator or StubSimulator()
    return (SumoEsDriver(config=config, evaluator=simulator,
                         screener=screener or AcceptAllScreener(config.pairs_per_update)),
            simulator)


def test_full_stub_run_budget_and_progress():
    driver, simulator = driver_for()
    summary = driver.run(updates=3)
    receipts = summary["receipts"]
    assert receipts[0]["kind"] == "ctrl0_panel_baseline"
    assert len(receipts) == 4
    episodes = 8 + 3 * (2 * 3 * 6 + 8)
    assert driver.ledger.jobs == episodes
    assert summary["budget"]["charged_sim_s"] == episodes * 900
    assert summary["budget"]["charged_sim_s"] <= CONFIG.cap_sim_s
    updates = receipts[1:]
    for index, receipt in enumerate(updates):
        assert sum(receipt["panel_J_after"]) <= sum(receipt["panel_J_before"])
        if index:
            previous = updates[index - 1]
            expected_before = (previous["panel_J_after"] if previous["accepted"]
                               else previous["panel_J_before"])
        else:
            expected_before = receipts[0]["panel_J"]
        assert receipt["panel_J_before"] == expected_before
    final = torch.tensor(summary["final_state"]["theta"])
    assert float(final.abs().max()) > 0.0


def test_rejected_update_rolls_back_bitwise_and_still_charges():
    config = SumoEsConfig(enabled=True)
    gradient_stub = StubSimulator(dimension=137)

    class PanelAdversarial:
        def __init__(self):
            self.episodes = 0

        def evaluate(self, theta, world_seed, update, tag):
            self.episodes += 1
            j = gradient_stub.episode(theta, world_seed, update)["J600"]
            if world_seed in config.panel_seeds():
                j = j + 1e6 * float(theta @ theta)
            return dict(job=f"s{world_seed}_{tag}", J600=j, actual_sim_s=900,
                        safety_clean=True, initial_state_sha256=f"h{world_seed}")

    evaluator = PanelAdversarial()
    driver = SumoEsDriver(config=config, evaluator=evaluator,
                          screener=AcceptAllScreener(config.pairs_per_update))
    state = driver.initial_state()
    baseline = driver.baseline(state)
    snapshot = state.to_dict()
    receipt = driver.run_update(state)
    assert receipt["accepted"] is False
    assert state.to_dict()["theta"] == snapshot["theta"]
    assert state.adam.step_count == 0 and state.adam.m is None
    assert state.update_count == 1
    assert receipt["budget"]["charged_sim_s"] == (8 + 2 * 3 * 6 + 8) * 900
    assert receipt["budget"]["charged_sim_s"] == (
        baseline["budget"]["charged_sim_s"] + (2 * 3 * 6 + 8) * 900)


def test_budget_ledger_cap_and_actual_clamping():
    ledger = BudgetLedger(cap_sim_s=1_800, episode_charge_sim_s=900)
    assert ledger.authorize("a") == 900
    ledger.record_actual("a", 900)
    assert ledger.authorize("b") == 900
    ledger.record_actual("b", 900)
    with pytest.raises(BudgetExceeded):
        ledger.authorize("c")
    ledger2 = BudgetLedger(cap_sim_s=900, episode_charge_sim_s=900)
    ledger2.authorize("c")
    ledger2.record_actual("c", 0)
    ledger2.record_actual("c", 5_000)
    assert ledger2.actual_sim_s == 900
    assert ledger2.charged_sim_s == 900


def test_resume_equals_single_run():
    driver_a, _ = driver_for()
    full = driver_a.run(updates=4)
    driver_b, _ = driver_for()
    state = driver_b.initial_state()
    driver_b.receipts.append(driver_b.baseline(state))
    driver_b.run_update(state)
    driver_b.run_update(state)
    resumed = EsState.from_dict(state.to_dict())
    for _ in range(2):
        driver_b.run_update(resumed)
    assert resumed.to_dict() == full["final_state"]
    assert [r["sigma"] for r in driver_b.receipts[1:]] == [r["sigma"] for r in
                                                           full["receipts"][1:]]


def test_determinism_two_runs_identical():
    one, _ = driver_for()
    two, _ = driver_for()
    assert one.run(updates=2) == two.run(updates=2)


def test_safety_failure_stops_dispatch():
    class Unsafe:
        def evaluate(self, theta, world_seed, update, tag):
            return dict(job="s1_u0", J600=0.0, actual_sim_s=900, safety_clean=False)

    driver = SumoEsDriver(config=CONFIG, evaluator=Unsafe(),
                          screener=AcceptAllScreener(1))
    state = driver.initial_state()
    with pytest.raises(MechanicalFailure):
        driver.run_update(state)


def test_screening_selects_top_directions_and_zero_screen_is_noop():
    stub = StubSimulator(dimension=8)
    screener = RankingScreener(stub, keep=2)
    driver, _ = driver_for(SumoEsConfig(enabled=True), simulator=stub, screener=screener)
    state = driver.initial_state(8)
    state.theta.normal_()
    driver.baseline(state)
    directions = state.directions(0, CONFIG.wm_screen_directions)
    screen = screener.screen(state.theta, 0.005, directions, 0)
    assert len(screen["kept"]) == 2 and len(screen["rejected"]) == 6
    assert min(screen["predicted_gain"]) >= max(
        screen["all_gains"][i] for i in screen["rejected"])

    class RejectAll:
        def screen(self, theta, sigma, directions, update):
            return dict(kept=[], predicted_gain=[], rejected=list(range(len(directions))))

    driver2 = SumoEsDriver(config=CONFIG, evaluator=stub, screener=RejectAll())
    state2 = driver2.initial_state(8)
    driver2.baseline(state2)
    theta_before = state2.theta.clone()
    receipt = driver2.run_update(state2)
    assert receipt["screen"]["kept"] == []
    assert receipt["no_step"] is True and receipt["accepted"] is None
    assert torch.equal(state2.theta, theta_before)
    assert receipt["budget"]["jobs"] == 8


def test_default_off_and_sumo_evaluator_refuse(tmp_path):
    with pytest.raises(PermissionError):
        driver_for(SumoEsConfig.load("configs/c4_sumoes_design.toml"))[0].run()
    with pytest.raises(TypeError):
        SumoRuntimeEvaluator()
    enabled = tmp_path / "enabled.toml"
    enabled.write_text("[sumoes]\nenabled = true\nmax_updates = 2\n"
                       "seed_block_first = 969440\nseed_block_last = 969459\n",
                       encoding="utf-8")
    from tools.sumoes.driver import main
    out = tmp_path / "out"
    assert main(["--config", str(enabled), "--evaluator", "stub",
                 "--out", str(out)]) == 0
    payload = json.loads((out / "SUMOES_RECEIPTS.json").read_text())
    assert payload["budget"]["charged_sim_s"] == (8 + 2 * (2 * 3 * 6 + 8)) * 900
    assert len(payload["receipts"]) == 3
