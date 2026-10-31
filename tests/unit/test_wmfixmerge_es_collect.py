"""Receipt-only throughput arithmetic and reporting contract."""
import json

from tools.wmfixmerge import es_collect as collect


def test_k8_aggregation_and_report_line_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "RECEIPTS", tmp_path)
    template = collect.REPORT.read_text(encoding="utf-8")
    for host in ("cpuQ", "gpuA"):
        for i in range(8):
            (tmp_path / f"{host}_PROBE_f{i}.json").write_text(json.dumps(dict(
                future_index=i, n_futures=8, threads=1, group_seconds=100.,
                runtime_load_seconds=2., max_rss_mb=500. + i)))
    winner = dict(fitness=1., paired_J_MAE=5., source_contrast_MAE=2.)
    (tmp_path / "GEN_000.json").write_text(json.dumps(dict(gen=0, candidates={"0": winner},
        best_overall=dict(gen=0, cand=0, **winner))))
    q, n = collect.probe("cpuQ")
    assert n == 8
    assert q["cpu_seconds"] == 800.
    assert q["parallel_wall"] == 102.
    rendered = collect.render(template)
    assert "cpuQ N=30 5.3 min" in rendered
    assert "gpuA N=18 8.9 min" in rendered
    assert "combined 3.3 min" in rendered
    assert "g0=5.00" in rendered and "g0=2.00" in rendered
    assert "NOT ADOPTABLE" in rendered
    assert len(rendered.splitlines()) <= 60


def test_partial_probe_never_claims_measured_k8(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "RECEIPTS", tmp_path)
    rendered = collect.render(collect.REPORT.read_text(encoding="utf-8"))
    assert "cpuQ 0/8; gpuA 0/8" in rendered
    assert "Extrapolated 12-FIT-group" not in rendered
    assert "pending generation 0" in rendered


def test_collector_never_stages_foreign_edits(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def fake(argv):
        calls.append(argv)
        return SimpleNamespace(stdout=" M user_work.py\n")
    monkeypatch.setattr(collect, "command", fake)
    collect.commit_receipts({"docs/paper_results/WMFIX_ES.md": "abc"})
    assert calls == [["git", "status", "--porcelain", "--untracked-files=all"]]
