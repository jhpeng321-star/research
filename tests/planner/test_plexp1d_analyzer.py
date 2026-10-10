"""PL-EXP1d analyzer tests (zero SUMO; synthetic tmp_path fixtures only).

PL-EXP1d = the PL-EXP1c design + walls (wall-v2.1-closure) on the 6 TRAIN
worlds with penetration >= 0.10. The run outputs do NOT exist yet, so every
fixture here is synthetic: closed-loop-format physics rows, synthetic
receipts, and (where a wall section is under test) synthetic
``walls.jsonl`` / ``candidates.jsonl`` / ``walls_summary`` blocks.

What is pinned here:

- the paired contrast PL-EXP1d planner - PL-EXP1c planner
  (``--paired-runs-root``): the AGENTS sign (delta = treatment - reference,
  NEGATIVE = improvement for delay), the delay veh*s / % values, the rule-C
  safety deltas, and the explicit list of worlds missing in either root;
- the per-world WALL section built from ``walls_summary`` + ``walls.jsonl``
  + ``candidates.jsonl`` + the physics ``cav_actions`` (wall-v2 member
  CAV-seconds);
- a run WITHOUT walls leaves the output byte-identical to the pre-PL-EXP1d
  analyzer (a SHA-256 over the path-normalised ``PLEXP1_RESULTS.json`` of a
  PL-EXP1b/1c-style fixture; the golden digest was captured from the
  analyzer BEFORE walls/paired support was added);
- a ``--worlds`` file carrying only the 6-world PL-EXP1d subset does not
  refuse and never pools without the per-penetration rows.

The shared fixture builders / analyzer module are imported from
``tests/planner/test_plexp1_analyzer.py`` so there is a single source of
truth for the synthetic closed-loop rows.
"""

import hashlib
import json
from pathlib import Path

import pytest

from tests.planner import test_plexp1_analyzer as base

az = base.az
WORLD = base.WORLD
VEHICLES = base.VEHICLES

pytestmark = pytest.mark.skipif(
    not base._p4_available(), reason="WD-01 p4 delay library absent"
)


@pytest.fixture(autouse=True)
def _short_window(monkeypatch):
    """The synthetic worlds are 20 s long, not the registered 600 s."""
    monkeypatch.setattr(az, "WINDOW_S", base.HORIZON)


# --------------------------------------------------------------- fixtures
def make_planner_run(root, world=WORLD, *, speed=20.0, coverage=None,
                     hard_brake=False, receipt_overrides=None,
                     walls_summary=None, walls_rows=None, candidates=None,
                     wall_cav_rows=0, decisions=None):
    """A synthetic PL-EXP1b/1c/1d planner run ``<root>/<world>__planner_oracle``.

    Closed-loop-format physics rows (the p4 library can feed them), an
    oracle planner receipt (guards pass), and optionally the PL-EXP1d wall
    artefacts. ``wall_cav_rows`` marks the first N physics rows with a
    ``wall-v2`` cav_action (wall-member CAV-seconds = N * dt)."""
    coverage = base.HORIZON if coverage is None else coverage
    run = Path(root) / f"{world}__planner_oracle"
    run.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "pl04_closed_loop_receipt/1",
        "exit_status": "complete",
        "run_purpose": "planner",
        "world": world,
        "seed": base.SEED,
        "split": "TRAIN",
        "predictor_class": "oracle",
        "decision_period_s": 10.0,
        "horizon_s": coverage,
        "sim_s": {"master_s": 0.0, "lookahead_s": 5.0, "total_s": 5.0, "cap_s": 500.0},
        "sumo_launches": [],
    }
    receipt.update(receipt_overrides or {})
    if walls_summary is not None:
        receipt["walls_summary"] = walls_summary
    (run / "RUN_RECEIPT.json").write_text(json.dumps(receipt), encoding="utf-8")

    rows = base._integrated_rows(
        {v: [(coverage, speed)] for v in VEHICLES}, base.T0 + 0.5, base.T0 + coverage
    )
    if hard_brake:
        for row in rows[:2]:
            row["vehicles"]["mainline.6"]["accel_mps2"] = -4.0
    for row in rows[: max(0, int(wall_cav_rows))]:
        row["cav_actions"] = {"mainline.6": {"action": "wall-v2", "owner": "wall-v2"}}
    base._write_jsonl(run / "physics_steps.jsonl", rows)
    base._write_jsonl(run / "decisions.jsonl", decisions if decisions is not None else [])
    if walls_rows is not None:
        base._write_jsonl(run / "walls.jsonl", walls_rows)
    if candidates is not None:
        base._write_jsonl(run / "candidates.jsonl", candidates)
    scenario = run / "scenario"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "requests.rou.xml").write_text(base.ROU_XML, encoding="utf-8")
    return run


def build_nonwall_world(tmp_path, world=WORLD):
    """(runs root, campaign root, worlds file) for one PL-EXP1b/1c-style world:
    a synthetic planner run + the two reference masters + the damping base."""
    planner_src = make_planner_run(tmp_path / "src", world=world, speed=20.0)
    runs, campaign = base.build_world(tmp_path, planner_src, world=world)
    worlds = base.worlds_file(tmp_path)
    return runs, campaign, worlds


# ------------------------------------------------ byte-identity regression
def _replace_paths(obj, root):
    if isinstance(obj, str):
        return obj.replace(root, "<ROOT>")
    if isinstance(obj, dict):
        return {k: _replace_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_paths(v, root) for v in obj]
    return obj


def normalised_results_json(out_dir, root):
    """The path-normalised ``PLEXP1_RESULTS.json`` text (so the digest is
    independent of the tmp directory the fixture is built under)."""
    payload = json.loads(
        (Path(out_dir) / "PLEXP1_RESULTS.json").read_text(encoding="utf-8")
    )
    return json.dumps(_replace_paths(payload, str(root)), indent=2, sort_keys=True)


#: SHA-256 of ``normalised_results_json`` for the PL-EXP1b/1c-style fixture,
#: captured from the analyzer BEFORE the PL-EXP1d wall/paired support landed.
_GOLDEN_SHA256 = "80c98798117f4a559f252126f9f9535398e02e05033c913063b78ca5752ea9a4"


def test_nonwall_run_leaves_results_json_byte_identical(tmp_path):
    """A run without ``walls_summary`` and no ``--paired-runs-root`` produces
    exactly the pre-PL-EXP1d ``PLEXP1_RESULTS.json`` (a digest match over the
    path-normalised serialisation -> byte-identity)."""
    runs, campaign, worlds = build_nonwall_world(tmp_path)
    out = tmp_path / "out"
    code = az.main([
        "--runs-root", str(runs), "--worlds", str(worlds),
        "--campaign-runs-root", str(campaign), "--out", str(out),
    ])
    assert code == 0
    digest = hashlib.sha256(normalised_results_json(out, tmp_path).encode("utf-8")).hexdigest()
    assert digest == _GOLDEN_SHA256
    # and no PL-EXP1d-only keys leaked into the output
    results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
    assert "paired" not in results["summary"]
    assert "paired_runs_root" not in results["meta"]
    assert all("walls" not in w and "paired" not in w for w in results["worlds"])


# ------------------------------------------------------------ paired contrast
def _pair_fixture(tmp_path):
    """PL-EXP1d runs root (planner 24 m/s + one injected hard brake) and the
    PL-EXP1c paired root (planner 20 m/s), with shared references/campaign."""
    runs_d = tmp_path / "plexp1d"
    runs_c = tmp_path / "plexp1c"
    make_planner_run(runs_d, speed=24.0, hard_brake=True)
    make_planner_run(runs_c, speed=20.0)
    for root in (runs_d,):
        base.make_ref_run(root, WORLD, "native", speed=18.0)
        base.make_ref_run(root, WORLD, "pi", speed=21.0)
    campaign = tmp_path / "campaign"
    base.make_chunk_world(campaign / "campaign_stage1_chunk2", WORLD, damping_speed=22.0)
    return runs_d, runs_c, campaign


def test_paired_contrast_sign_and_values(tmp_path):
    runs_d, runs_c, campaign = _pair_fixture(tmp_path)
    layout, provider = base.az.chunk1_layout(), base.az.p4_delay_provider(
        base.az.P4_SEARCH_PATHS)
    record = az.analyze_world(
        base.WORLD_ENTRY, runs_d, campaign, layout, provider, 1.5,
        runs_d, paired_runs_root=runs_c,
    )
    paired = record["paired"]
    assert paired["present"] is True
    planner_d = record["planner"]["delay_s"]
    planner_c = paired["planner_pleexp1c"]["delay_s"]
    # AGENTS sign: delta = treatment (1d) - reference (1c); the 24 m/s planner
    # is FASTER than the 20 m/s one -> NEGATIVE = improvement
    assert paired["delta_delay_s"] == pytest.approx(planner_d - planner_c)
    assert paired["delta_delay_s"] < 0
    assert paired["delta_delay_pct_of_reference"] == pytest.approx(
        100.0 * (planner_d - planner_c) / planner_c)
    assert paired["delta_delay_pct_of_reference"] < 0
    safety = paired["safety_delta"]
    # cost metric: the injected hard brake makes 1d WORSE -> POSITIVE delta
    assert safety["hard_brake_veh_s"] == pytest.approx(1.0)
    assert safety[az.PRIMARY_EXPOSURE_KEY] == pytest.approx(0.0)
    assert safety["collision_planner"] is False
    assert safety["collision_reference"] is False
    # rule C (1d vs 1c): the hard-brake regression fails the paired verdict
    assert paired["owner_rule_primary"]["evaluable"] is True
    assert paired["owner_rule_primary"]["rel_ok"] is False
    assert paired["owner_rule_primary"]["pass"] is False


def test_paired_worlds_missing_listed_not_dropped(tmp_path):
    runs_d, runs_c, campaign = _pair_fixture(tmp_path)
    world_b = "w_pl04i_b"
    entry_b = {"chunk": "c2", "world": world_b, "seed": base.SEED, "penetration": 0.05,
               "loading_veh_h": 2400, "t0_s": base.T0, "split": "TRAIN"}
    make_planner_run(runs_d, world=world_b, speed=18.0)
    base.make_ref_run(runs_d, world_b, "native", speed=18.0)
    base.make_ref_run(runs_d, world_b, "pi", speed=21.0)
    base.make_chunk_world(campaign / "campaign_stage1_chunk2", world_b)
    # NOTE: world_b has NO PL-EXP1c paired run
    out = tmp_path / "out"
    code = az.main([
        "--runs-root", str(runs_d), "--paired-runs-root", str(runs_c),
        "--worlds", str(base.worlds_file(tmp_path, [base.WORLD_ENTRY, entry_b])),
        "--campaign-runs-root", str(campaign), "--out", str(out),
    ])
    assert code == 0
    results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
    assert results["meta"]["paired_runs_root"] == str(runs_c)
    assert results["summary"]["paired_missing_worlds"] == [world_b]
    by_world = {w["world"]: w for w in results["worlds"]}
    assert by_world[base.WORLD]["paired"]["present"] is True
    assert by_world[world_b]["paired"]["present"] is False
    assert by_world[world_b]["paired"]["planner_pleexp1c"] is None
    # pooled by penetration, never pooled alone
    for key in ("paired_delta_delay_s", "paired_delta_delay_pct"):
        assert "pooled" in results["summary"][key]
        assert "by_penetration" in results["summary"][key]
    report = (out / "PLEXP1_REPORT.md").read_text(encoding="utf-8")
    assert "PL-EXP1d" in report and "PL-EXP1c" in report
    assert world_b in report


# ---------------------------------------------------------------- wall section
def _wall_artifacts():
    walls_summary = {
        "decisions": 10,
        "decisions_with_priced_walls": 4,
        "priced_wall_candidates": 8,
        "priced_truncated": 2,
        "chosen_walls": 3,
        "chosen_truncated": 1,
        "chosen_hold_options": {"keep": 2, "release": 1},
        "events": {"registered": 2, "assembly": 2, "hold": 1, "handback": 1,
                   "dissolved": 2},
        "abort_reasons": {"1": 0},
    }
    walls_rows = [
        {"t_s": 10.0, "event": "registered", "phase": "assembly", "wall_id": "w1"},
        {"t_s": 20.0, "event": "hold", "phase": "hold", "wall_id": "w1"},
        {"t_s": 30.0, "event": "handback", "phase": "handback", "wall_id": "w1"},
        {"t_s": 40.0, "event": "dissolved", "phase": None, "wall_id": "w1"},
        {"t_s": 50.0, "event": "abort", "phase": "handback", "wall_id": "w2",
         "abort_reason": 1},
    ]
    candidates = [
        {"schema": "pl04_candidate_row/1", "stage": "wall", "chosen": True,
         "walls": [{"wall_id": "w1", "phase": "assemble"}],
         "veto": {"vetoed": False, "rules": []}},
        {"schema": "pl04_candidate_row/1", "stage": "wall", "chosen": False,
         "walls": [{"wall_id": "w2", "phase": "assemble"}],
         "veto": {"vetoed": True, "rules": ["assembly_bar"]}},
        {"schema": "pl04_candidate_row/1", "stage": "wall", "chosen": False,
         "walls": [{"wall_id": "w3", "phase": "hold"}],
         "veto": {"vetoed": True, "rules": ["absolute_ttc", "relative"]}},
        {"schema": "pl04_candidate_row/1", "stage": "probe", "chosen": False,
         "veto": {"vetoed": False, "rules": []}},
    ]
    return walls_summary, walls_rows, candidates


def test_wall_section_counts(tmp_path):
    walls_summary, walls_rows, candidates = _wall_artifacts()
    runs = tmp_path / "plexp1d"
    make_planner_run(
        runs, speed=20.0, walls_summary=walls_summary, walls_rows=walls_rows,
        candidates=candidates, wall_cav_rows=4,
    )
    base.make_ref_run(runs, WORLD, "native", speed=18.0)
    base.make_ref_run(runs, WORLD, "pi", speed=21.0)
    campaign = tmp_path / "campaign"
    base.make_chunk_world(campaign / "campaign_stage1_chunk2", WORLD, damping_speed=22.0)
    layout, provider = base.az.chunk1_layout(), base.az.p4_delay_provider(
        base.az.P4_SEARCH_PATHS)
    record = az.analyze_world(base.WORLD_ENTRY, runs, campaign, layout, provider)
    wall = record["walls"]
    assert wall["decisions"] == 10
    assert wall["decisions_with_priced_walls"] == 4
    assert wall["priced_wall_candidates"] == 8
    assert wall["chosen_walls"] == 3
    assert wall["truncated_share_priced"] == pytest.approx(2 / 8)
    assert wall["truncated_share_chosen"] == pytest.approx(1 / 3)
    assert wall["chosen_hold_options"] == {"keep": 2, "release": 1}
    # lifecycle events + abort reasons from walls.jsonl
    assert wall["events"] == {"assemble": 1, "hold": 1, "release": 2, "abort": 1}
    assert wall["abort_reasons"] == {"safety": 1}
    # wall-member CAV-seconds under 'wall-v2' from physics cav_actions
    assert wall["wall_member_cav_seconds"] == pytest.approx(4 * 0.5)
    # candidates.jsonl wall-stage counts
    cand = wall["candidates"]
    assert cand["wall_stage_candidates"] == 3
    assert cand["vetoed"] == 2
    assert cand["vetoed_by_rule"] == {"absolute_ttc": 1, "assembly_bar": 1, "relative": 1}
    assert cand["chosen"] == 1

    # the CLI renders the wall section + wall CSV columns
    out = tmp_path / "out_wall"
    assert az.main([
        "--runs-root", str(runs), "--worlds", str(base.worlds_file(tmp_path)),
        "--campaign-runs-root", str(campaign), "--out", str(out),
    ]) == 0
    report = (out / "PLEXP1_REPORT.md").read_text(encoding="utf-8")
    assert "wall group action" in report
    csv_header = (out / "worlds.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "wall_member_cav_seconds" in csv_header


def test_run_without_walls_has_no_wall_section(tmp_path):
    runs, campaign, _worlds = build_nonwall_world(tmp_path)
    layout, provider = base.az.chunk1_layout(), base.az.p4_delay_provider(
        base.az.P4_SEARCH_PATHS)
    record = az.analyze_world(base.WORLD_ENTRY, runs, campaign, layout, provider)
    assert "walls" not in record


# ------------------------------------------------------------- --worlds filter
def test_six_world_subset_worlds_filter(tmp_path):
    """A ``--worlds`` file with only the 6 registered PL-EXP1d worlds
    (penetration >= 0.10) must not refuse and must stratify, not mis-pool."""
    specs = [
        ("c2", "w8517m2700p10t480", base.SEED, 0.10, 2700, base.T0),
        ("c3", "w8581m2600p10t300", base.SEED, 0.10, 2600, base.T0),
        ("c2", "w8526m2700p20t300", base.SEED, 0.20, 2700, base.T0),
        ("c3", "w8594m2600p20t300", base.SEED, 0.20, 2600, base.T0),
        ("c2", "w8541m2700p40t240", base.SEED, 0.40, 2700, base.T0),
        ("c3", "w8610m2000p40t540", base.SEED, 0.40, 2000, base.T0),
    ]
    runs = tmp_path / "plexp1d"
    campaign = tmp_path / "campaign"
    entries = []
    for chunk, world, seed, pen, veh, t0 in specs:
        make_planner_run(runs, world=world, speed=20.0)
        base.make_ref_run(runs, world, "native", speed=18.0)
        base.make_ref_run(runs, world, "pi", speed=21.0)
        chunk_dir = campaign / base.az.CHUNK_DIRS[chunk]
        base.make_chunk_world(chunk_dir, world, damping_speed=22.0)
        entries.append({"chunk": chunk, "world": world, "seed": seed,
                        "penetration": pen, "loading_veh_h": veh, "t0_s": t0,
                        "split": "TRAIN"})
    worlds = base.worlds_file(tmp_path, entries, name="worlds_six.json")
    out = tmp_path / "out"
    code = az.main([
        "--runs-root", str(runs), "--worlds", str(worlds),
        "--campaign-runs-root", str(campaign), "--out", str(out),
    ])
    assert code == 0
    results = json.loads((out / "PLEXP1_RESULTS.json").read_text(encoding="utf-8"))
    summary = results["summary"]
    assert summary["n_registered_worlds"] == 6
    assert summary["n_included_in_pooled"] == 6
    # pooled rows always travel with the per-penetration breakdown
    assert list(summary["delta_delay_vs_native_s"]["by_penetration"]) == [
        "0.1", "0.2", "0.4"]
    assert summary["delta_delay_vs_native_s"]["pooled"]["n"] == 6
