"""P11 CLI tests: dry listing, entry-point listing, report + summary writing."""
from __future__ import annotations

import json
from pathlib import Path

from fakes import FakeFn, FakeResolver, clean_torch_tree
from preflight import cli
from preflight.gates import (g1_zero_residual as g1, g2_ctrl_parity as g2,
                             g3_label_leak as g3, g4_overfit as g4,
                             g5_e2e_smoke as g5, g6_heldout_dryrun as g6)
from preflight.registry import GATE_ORDER
from preflight.report import SUMMARY_FILENAME, write_gate_report, write_summary
from preflight.settings import Settings


def test_dry_listing_lists_six_gates_and_entry_points(capsys):
    text = cli.dry_listing()
    for name in GATE_ORDER:
        assert name in text
    assert "rollout2.equivalence.zero_residual_report" in text
    assert "ctrl.parity.strategy_parity_report" in text
    assert "hdv.input_schema.main_config_schema" in text
    assert "hdv.input_schema.perturbation_check" in text
    assert "train.idea_a.overfit_report" in text
    assert "bundle.e2e.run_e2e_smoke" in text
    assert "train.idea_a.heldout_split_report" in text
    assert f"({len(GATE_ORDER)} gates)" in text


def test_main_dry_returns_zero(capsys):
    rc = cli.main(["all", "--dry"])
    out = capsys.readouterr().out
    assert rc == 0
    for name in GATE_ORDER:
        assert name in out


def _pass_resolver():
    return FakeResolver({
        g1.ENTRY_POINTS[0]: FakeFn(
            {"ok": True, "integrator": "v2g", "windows": ["w0", "w1"],
             "max_position_error_m": 1e-9, "max_speed_error_mps": 1e-9}),
        g2.ENTRY_POINTS[0]: FakeFn(
            {"ok": True, "tolerance": 1e-3,
             "strategies": {s: {"max_error": 1e-5, "n_windows": 2}
                            for s in g2.REQUIRED_STRATEGIES
                            + g2.NEW_RULE_STRATEGIES}}),
        g3.ENTRY_POINTS[0]: FakeFn(
            {"fields": ["s_down_m", "v_mps", "a_mps2", "length_m"]}),
        g3.ENTRY_POINTS[1]: FakeFn(
            {"identical": True, "max_abs_delta": 0.0, "n_perturbations": 3}),
        g4.ENTRY_POINTS[0]: FakeFn(
            {"ok": True, "device": "cpu", "n_windows": 2, "threshold": 1e-2,
             "final_loss": 1e-4, "loss_curve": [1.0, 1e-4]}),
        g5.ENTRY_POINTS[0]: FakeFn(
            {"ok": True, "device": "cpu", "timing_seconds": 30.0,
             "stages": list(g5.STAGES)}),
        g6.ENTRY_POINTS[0]: FakeFn(
            {"splits": {
                "heldout-i-j4b": {"train": ["j4"], "eval": ["j4b"],
                                  "excluded": ["j4b"]},
                "heldout-ii-leave-pi": {"train": ["native"], "eval": ["pi"],
                                        "excluded": ["pi"]},
                "heldout-iii-leave-highest-wall-speed": {
                    "train": ["native"], "eval": ["wall-speed-highest"],
                    "excluded": ["wall-speed-highest"]}}}),
    })


def test_run_gates_all_pass_and_reports_written(tmp_path):
    s = Settings(wm_n1_dir=clean_torch_tree(tmp_path),
                 reports_dir=tmp_path / "reports", device="cpu")
    results = cli.run_gates(list(GATE_ORDER), s, resolve=_pass_resolver())
    assert all(r.passed for r in results), [r.detail for r in results]
    for r in results:
        write_gate_report(s.resolve_report_dir(), r)
    summary = write_summary(s.resolve_report_dir(), results, dry=False)
    body = json.loads(Path(summary).read_text(encoding="utf-8"))
    assert body["all_passed"] and body["n_gates"] == len(GATE_ORDER)
    assert body["gates"]["e2e-smoke"]["passed"] is True


def _missing_g4_resolver():
    """All gates resolve except G4 (the piece-missing FAIL path)."""
    mapping = dict(_pass_resolver().mapping)
    mapping.pop(g4.ENTRY_POINTS[0], None)
    return FakeResolver(mapping)


def test_main_all_with_missing_pieces_writes_summary(tmp_path, capsys,
                                                     monkeypatch):
    # Stage 2: every real piece now exists, so inject one missing entry point
    # through the resolver to keep exercising the CLI's piece-missing FAIL path
    # (stage 1 relied on G2 not having landed yet).
    reports = tmp_path / "reports"
    real_run_gates = cli.run_gates
    monkeypatch.setattr(
        cli, "run_gates",
        lambda names, settings, resolve=None: real_run_gates(
            names, settings, resolve=_missing_g4_resolver()))
    rc = cli.main(["all", "--reports-dir", str(reports)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    body = json.loads((reports / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert body["n_gates"] == len(GATE_ORDER)
    assert body["all_passed"] is False
    for name in GATE_ORDER:
        assert (reports / f"{name}.json").is_file()
