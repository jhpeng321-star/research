"""HARNESS_CONTRACT: the source rectangle is the labels' left boundary q_0..q_{n-1}.

tools/wmrequal/labels.py sums ``.5 * queue[:n]`` over post-service queue samples
q_0..q_{n-1} (S-GAPCHECK left-boundary rectangle).  Both runtime samplers must
reproduce that series: the tracked queue list is seeded with the pre-rollout Q0
sample and the rectangle drops the final post-service sample.  The pre-fix
convention summed q_1..q_n, a one-tick shift of ``.5 * (q_n - q_0)`` per row
(``.5 * delta q_n`` per paired contrast).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.wmfixmerge.runtime import RecordedDemand, TrackedEntries
from wms.model.entries import EntranceHistory, EntranceQueue, EntryConfig, TorchEntryProcessV2


def blocked_process(n_ticks):
    """Conserve-ON entrance with backlog Q0=[2,1], +[1,0] due per tick and no free
    slot, so the exact recorded queue series is q_k = [2 + k, 1]."""
    cfg = EntryConfig(conserve_queue=True, insert_rule="sumo_like_v2")
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(), cfg=cfg,
                                  row_seeds=[7], couple_draws=True,
                                  queues=[EntranceQueue(backlog=np.array([2., 1.]),
                                                        demand_rate=np.array([.2, .2]))])
    process._emit_due = RecordedDemand([[[1, 0]] * n_ticks])
    return process


def full_state():
    return dict(s_m=torch.zeros(1, 4), v_mps=torch.zeros(1, 4), a_mps2=torch.zeros(1, 4),
        lane=torch.zeros(1, 4, dtype=torch.long), length_m=torch.ones(1, 4) * 5,
        present=torch.ones(1, 4, dtype=torch.bool), is_cav=torch.zeros(1, 4, dtype=torch.bool))


def test_tracked_source_rectangle_is_the_label_left_boundary():
    n = 4
    tracked = TrackedEntries(blocked_process(n))
    s = full_state()
    for _ in range(n):
        tracked.insert(s)
    q = np.stack(tracked.queues, 1)                    # [1, n+1, 2]
    assert q.shape == (1, n + 1, 2)
    assert q[0, 0].tolist() == [2, 1]                  # seeded pre-rollout Q0 sample
    assert q[0, n].tolist() == [2 + n, 1]              # q_n is the last sample
    label = .5 * sum(float(q[0, k].sum()) for k in range(n))
    fixed = .5 * float(q[:, :n].sum())
    legacy = .5 * float(q[:, 1:].sum())                # pre-fix runtime convention
    assert fixed == pytest.approx(label)
    # the measured one-tick shift of the old convention, exactly WMFIX_REVIEW claim 4
    assert legacy == pytest.approx(fixed + .5 * (float(q[0, n].sum()) - float(q[0, 0].sum())))


def test_scorer_entries_seed_and_end_of_horizon_queue_sample(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "tools/wmrequal/score.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("wmfix_contract_score_test", path)
    score = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score)
    n = 3
    entries = score.Entries(blocked_process(n))
    assert entries.queues[0].tolist() == [[2, 1]]
    s = full_state()
    for _ in range(n):
        entries.insert(s)
    q = np.stack(entries.queues, 1)
    assert q.shape == (1, n + 1, 2)
    queue = torch.as_tensor(q)
    source = .5 * queue[:, :-1].sum(-1)                # scorer's fixed rectangle
    label = .5 * sum(float(q[0, k].sum()) for k in range(n))
    assert float(source[0, :n].sum()) == pytest.approx(label)
    # values[h]['queue'] stays the labels' end-of-horizon sample q_n (index n)
    assert float(queue[0, n].sum()) == pytest.approx(float(q[0, n].sum()))


def test_labels_left_boundary_series_matches_runtime_under_exact_service():
    """Exact service: Q0=[0,0] and one admitted arrival per tick keeps the queue
    at zero, so the runtime rectangle and the analytic label rectangle agree at
    every prefix while service demonstrably happens each tick."""
    n = 4
    cfg = EntryConfig(conserve_queue=True, insert_rule="sumo_like_v2")
    process = TorchEntryProcessV2(history=EntranceHistory.defaults(), cfg=cfg,
                                  row_seeds=[7], couple_draws=True,
                                  queues=[EntranceQueue(backlog=np.array([0., 0.]),
                                                        demand_rate=np.array([.2, .2]))])
    process._emit_due = RecordedDemand([[[1, 0]] * n])
    tracked = TrackedEntries(process)
    s = dict(s_m=torch.zeros(1, 8), v_mps=torch.zeros(1, 8), a_mps2=torch.zeros(1, 8),
             lane=torch.zeros(1, 8, dtype=torch.long), length_m=torch.ones(1, 8) * 5,
             present=torch.zeros(1, 8, dtype=torch.bool), is_cav=torch.zeros(1, 8, dtype=torch.bool))
    for k in range(n):
        newborn = tracked.insert(s)
        assert int(newborn.sum()) == 1, k              # the arrival is served, not dropped
        assert tracked.queues[-1].tolist() == [[0, 0]], k
    q = np.stack(tracked.queues, 1)
    label = .5 * sum(float(q[0, k].sum()) for k in range(n))
    assert .5 * float(q[:, :n].sum()) == pytest.approx(label) == pytest.approx(0.0)


def _row(native, queues):
    horizons = ("60", "90", "120", "180", "300")
    return dict(pair_key="k", native=native, source_available=True, queue0_total=4,
                values={h: dict(queue=q, source=10.0 + i)
                        for i, (h, q) in enumerate(zip(horizons, queues))})


def test_contract_measures_the_one_tick_shift_from_recorded_labels(tmp_path):
    from tools.wmfixmerge.harness_contract import build, measure
    stats = measure([_row(True, [4, 2, 8, 10, 12]), _row(False, [6, 2, 8, 10, 16])])
    h60 = stats["60"]
    assert h60["row_shift_veh_s"]["mean"] == pytest.approx(.5)      # shifts 0.0 and 1.0
    assert h60["row_shift_veh_s"]["max"] == pytest.approx(1.0)
    assert h60["contrast_shift_veh_s"]["mean"] == pytest.approx(1.0)  # .5*(6-4)
    assert h60["mean_abs_source_veh_s"] == pytest.approx(10.0)
    assert stats["300"]["contrast_shift_veh_s"]["mean"] == pytest.approx(2.0)
    real = tmp_path / "REAL.json"
    real.write_text(json.dumps([_row(True, [4, 2, 8, 10, 12]), _row(False, [6, 2, 8, 10, 16])]))
    report, prepared = tmp_path / "WMREQUAL3.md", tmp_path / "TRAIN_PREPARED.json"
    report.write_text("report")
    prepared.write_text("{}")
    out = tmp_path / "HARNESS_CONTRACT.json"
    contract = build(report, prepared, real, 1, out)
    assert contract["schema"] == "wmfix_harness_contract/1"
    assert contract["source_clock_match"] is True and contract["residual_remains"] is True
    assert contract["prepared_sha256"] and contract["real_sha256"]
    assert json.loads(out.read_text())["wmrequal3_one_tick_shift"]["60"] == contract[
        "wmrequal3_one_tick_shift"]["60"]
