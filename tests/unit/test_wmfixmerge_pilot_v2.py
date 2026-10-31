"""Pure decision helpers of the PILOT v2 protocol (no torch, no GPU)."""
import pytest
from types import SimpleNamespace

from tools.wmfixmerge import pilot_v2 as p
from tools.wmfixmerge import data_pilot as d


def v(a, b):
    return {120: a, 300: b}


def test_mini_update_streak_and_best():
    best, streak = p.mini_update([(5, v(10, 20)), (10, v(8, 18)), (15, v(9, 30)), (20, v(9, 40))], None)
    assert best == (10, 13.0) and streak == 2
    best, streak = p.mini_update([(15, v(9, 30)), (5, v(10, 20)), (10, v(8, 18))], None)
    assert best == (10, 13.0) and streak == 1  # chronological order enforced


def test_choose_step_earliest_tie():
    assert p.choose_step([(10, v(1, 3)), (5, v(2, 2)), (15, v(0, 4))]) == 5
    with pytest.raises(ValueError):
        p.choose_step([])


def test_assemble_fail_closed_and_mae():
    rows = [dict(group=g, horizon=h, arm=a, contrast=(-1) ** a * (g + 1))
            for g in range(2) for h in (120, 300) for a in (0, 1)]
    panel = p.assemble([rows[:4], rows[4:]], [0, 1])
    assert panel["120"]["paired_J_MAE"] == pytest.approx(1.5) and panel["300"]["rows"] == 4
    with pytest.raises(ValueError):
        p.assemble([rows[:4]], [0, 1])


def test_decide_v2_strict_both_horizons():
    assert p.decide_v2(v(10, 10), v(9, 9))["passed"]
    assert not p.decide_v2(v(10, 10), v(9, 10))["passed"]
    assert not p.decide_v2(v(10, 10), v(9, 9), frozen_bitwise=False)["passed"]


def test_data_route_requires_immutable_binding_and_smoothing_off(monkeypatch):
    hashes = {"prepared": d.PREPARED_SHA, "contract": d.CONTRACT_SHA, "checkpoint": d.WM2_SHA}
    monkeypatch.setattr(d, "digest", hashes.__getitem__)
    args = SimpleNamespace(prepared="prepared", harness_contract="contract", checkpoint="checkpoint",
                           smooth_events=False)
    assert d.check_binding(args)["smooth_events"] is False
    args.smooth_events = True
    with pytest.raises(ValueError, match="OFF"):
        d.check_binding(args)
    args.smooth_events = False
    hashes["prepared"] = "different"
    with pytest.raises(ValueError, match="52-world"):
        d.check_binding(args)


def test_default_route_still_requires_feasibility_and_smoothing(monkeypatch):
    monkeypatch.setattr(p, "install_guard", lambda: None)
    with pytest.raises(SystemExit) as exc:
        p.main(["--prepared", "p", "--config", "c", "--checkpoint", "w", "--controllers", "d", "--out", "o"])
    assert exc.value.code == 2


def test_explicit_flag_dispatches_registered_route(monkeypatch):
    monkeypatch.setattr(p, "install_guard", lambda: None)
    seen = []
    monkeypatch.setattr(d, "check_binding", lambda args: seen.append(args))
    monkeypatch.setattr(d, "run", lambda args: 23)
    assert p.main(["--prepared", "p", "--config", "c", "--checkpoint", "w", "--controllers", "d",
                   "--out", "o", "--registered-data-pilot", "--harness-contract", "h"]) == 23
    assert len(seen) == 1 and not seen[0].smooth_events


def test_data_gate_is_unchanged_decide_and_requires_all_three_steps():
    from tools.wmfixmerge.pilot import decide
    baseline = v(10, 20)
    candidates = {10: v(9, 21), 20: v(11, 19), 30: v(12, 22)}
    result = d.decide_panels(baseline, candidates, True, True)
    original = decide({h: [value, value] for h, value in baseline.items()}, candidates)
    for key in original:
        assert result[key] == original[key]
    assert not d.decide_panels(baseline, candidates, False, True)["passed"]
    with pytest.raises(ValueError, match="all step"):
        d.decide_panels(baseline, {10: v(9, 19)}, True, True)


def test_selection_worlds_never_enter_optimizer_cycle():
    seeds = list(range(969400, 969440)) + list(range(973000, 973012))
    groups = [dict(pair_key=str(s)) for s in seeds]
    labels = [dict(pair_key=str(s), seed=s) for s in seeds]
    fitting = d.fit_groups(groups, labels)
    assert len(fitting) == 44
    assert {seeds[i] for i in fitting}.isdisjoint(d.SELECTION_SEEDS)


def test_panel_queue_bounds_startup_reservations(monkeypatch):
    class FakePanels:
        device = "cuda"
        running = {}
        def launch(self, name, candidate, groups):
            self.running[name] = (candidate, groups)
        def poll(self, name):
            return None
    panels = FakePanels()
    queue = d.PanelQueue(panels)
    queue.submit("a", "checkpoint", [0])
    queue.submit("b", "checkpoint", [1])
    monkeypatch.setattr(d.subprocess, "check_output", lambda *a, **k: "6000")
    queue.tick()
    queue.tick()
    assert set(panels.running) == {"a"} and len(queue.pending) == 1
