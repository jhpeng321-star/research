"""OE-46 branch B dataset tests (zero simulation; read-only on frozen inputs).

Checks DATASET_OE46.json against its frozen SPEC (output/oe46_b_dataset_v1/
SPEC_FREEZE.md): population identity from registered metadata, label fidelity
vs flow_effect, provenance flags, request-weighted arithmetic, and
no-outcome-based exclusion.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/oe46_b_dataset_v1"
OE34 = ROOT / "runs/active/oe34_observation_relational"

FEASIBLE_32 = [
    90004, 90005, 90007, 90008, 90011, 90019, 90021, 90024, 90026, 90030,
    90031, 90034, 90035, 90036, 90037, 90038, 90040, 90041, 90042, 90043,
    90045, 90046, 90047, 90048, 90050, 90051, 90053, 90054, 90056, 90057,
    90058, 90060]
BRANCH_22 = {90004, 90005, 90007, 90011, 90019, 90021, 90024, 90030, 90034,
             90037, 90040, 90045, 90046, 90047, 90048, 90051, 90053, 90054,
             90056, 90057, 90058, 90060}
COST_KEYS = ("service", "wait", "braking", "effort", "ttc")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return load(OUT / "DATASET_OE46.json")


@pytest.fixture(scope="module")
def comparators():
    return load(OUT / "COMPARATORS.json")


@pytest.fixture(scope="module")
def results():
    return load(ROOT / "output/flow_effect_v1/RESULTS.json")


@pytest.fixture(scope="module")
def rows_by_seed(dataset):
    return {r["seed"]: r for r in dataset["rows"]}


def test_population_identity_from_registered_metadata(dataset, rows_by_seed):
    """Membership must be reproducible from binding states + registration
    split alone (no outcome fields)."""
    reg = load(OE34 / "registration.json")
    fit = set(reg["split"]["fit"])
    derived = []
    for seed in range(90001, 90061):
        b5 = load(OE34 / f"bindings/s{seed}_h5.json")
        if b5["unavailable"] is False:
            derived.append(seed)
        elif b5["reason"] == "hold not active at HOLD+5":
            derived.append(seed)
        else:
            assert b5["reason"] == "entry_refusal", seed
    assert sorted(derived) == FEASIBLE_32
    assert sorted(r["seed"] for r in dataset["rows"]) == FEASIBLE_32
    assert len(rows_by_seed) == 32 == dataset["population"]["feasible_groups"]
    for row in dataset["rows"]:
        want = "development" if row["seed"] in fit else "holdout"
        assert row["split"] == want
        assert row["h5_provenance"] == (
            "executed_h5" if row["seed"] in BRANCH_22
            else "policy_equivalent_prebranch")
    assert dataset["population"]["executed_h5"] == 22
    assert dataset["population"]["policy_equivalent_prebranch"] == 10
    assert dataset["population"]["development"] == 14
    assert dataset["population"]["holdout"] == 18
    assert dataset["population"]["never_offered_external"] == 28


def test_never_offered_and_out_of_scope_documented_not_used(dataset):
    """The 28 entry_refusal seeds and 5 reused rows appear only as
    documentation; no dataset row or comparator detail may carry them."""
    for seed in range(90001, 90061):
        if seed in FEASIBLE_32:
            continue
        b5 = load(OE34 / f"bindings/s{seed}_h5.json")
        assert b5["unavailable"] is True
        assert b5["reason"] == "entry_refusal"
        assert b5["equivalent_to"] == "damping"
    comp = load(OUT / "COMPARATORS.json")
    for block in comp["comparators"].values():
        for d in block["per_scenario"]:
            assert d["seed"] in FEASIBLE_32
    for source in [4150, 4151, 4154, 4158, 4160]:
        assert source not in FEASIBLE_32
        assert dataset["population"]["out_of_scope_relational_rows"] == [
            4150, 4151, 4154, 4158, 4160]


def test_label_fidelity_vs_flow_effect(dataset, rows_by_seed, results):
    cases_h5 = {c["seed"]: c for c in results["cases"] if c["action"] == "h5"}
    cases_h20 = {c["seed"]: c
                 for c in results["cases"] if c["action"] == "h20"}
    assert len(cases_h5) == 32 and len(cases_h20) == 32
    dist = {"beneficial": 0, "harmful": 0, "nonbeneficial": 0}
    for seed, row in rows_by_seed.items():
        case = cases_h5[seed]
        assert row["label_block"] == case["label"]
        assert row["branch_available"] == case["branch_available"]
        for key in COST_KEYS:
            assert row["physical_deltas"][key] == \
                pytest.approx(case["physical_deltas"][key], abs=1e-12)
        dist[case["label"]["label"]] += 1
        if row["h5_provenance"] == "policy_equivalent_prebranch":
            c20 = cases_h20[seed]
            assert case["physical_deltas"] == c20["physical_deltas"]
            assert case["label"]["label"] == c20["label"]["label"]
    assert dist == {"beneficial": 8, "harmful": 20, "nonbeneficial": 4}
    assert dataset["label_distribution_h5_vs_damping"] == dist


def test_arm_costs_match_raw_summaries(rows_by_seed):
    for seed, row in rows_by_seed.items():
        d_sum = load(OE34 / f"s{seed}_damping/summary.json")
        assert row["arms"]["damping"]["population"] == d_sum["population"]
        d_costs = row["arms"]["damping"]["costs_per_request"]
        assert d_costs["service"] == pytest.approx(
            d_sum["system_seconds_per_requested_vehicle"], abs=1e-12)
        assert d_costs["wait"] == pytest.approx(
            d_sum["source_wait_vehicle_seconds"] / d_sum["population"],
            abs=1e-12)
        for key, field in (("braking", "braking_vehicle_seconds"),
                           ("effort", "acceleration_energy"),
                           ("ttc", "ttc_below_3_vehicle_seconds")):
            assert d_costs[key] == pytest.approx(
                d_sum["totals"][field] / d_sum["population"], abs=1e-12)
        assert row["arms"]["damping"]["collision_involvements"] == \
            d_sum["totals"]["collision_involvements"]
        assert row["arms"]["damping"]["starting_teleports"] == \
            d_sum["totals"]["starting_teleports"]
        if row["h5_provenance"] == "executed_h5":
            h_sum = load(OE34 / f"s{seed}_h5/summary.json")
        else:
            h_sum = load(OE34 / f"s{seed}_h20/summary.json")
        h = row["arms"]["h5"]
        assert h["provenance"] == row["h5_provenance"]
        assert h["population"] == h_sum["population"]
        assert h["costs_per_request"]["service"] == pytest.approx(
            h_sum["system_seconds_per_requested_vehicle"], abs=1e-12)
        # stored delta == h5 arm costs - damping arm costs
        for key in COST_KEYS:
            assert row["physical_deltas"][key] == pytest.approx(
                h["costs_per_request"][key]
                - d_costs[key], abs=1e-9)
        assert row["verification"]["max_abs_delta_mismatch"] <= 1e-9


def test_request_weighted_arithmetic(comparators, rows_by_seed):
    for name, block in comparators["comparators"].items():
        detail = block["per_scenario"]
        assert len(detail) == 32
        pops = [d["population"] for d in detail]
        total_pop = sum(pops)
        agg = block["aggregate"]
        assert agg["groups"] == 32
        assert agg["requested_population_total"] == total_pop
        for key in COST_KEYS:
            vals = [d["costs_per_request"][key] for d in detail]
            assert agg[f"{key}_per_request_sum"] == pytest.approx(
                sum(vals), abs=1e-9)
            assert agg[f"{key}_per_request_mean"] == pytest.approx(
                sum(vals) / 32, abs=1e-9)
            assert agg[f"{key}_per_request_weighted_mean"] == pytest.approx(
                sum(v * p for v, p in zip(vals, pops)) / total_pop, abs=1e-9)
        for key in ("collision_involvements", "starting_teleports"):
            vals = [d[key] for d in detail]
            assert agg[f"{key}_sum"] == sum(vals)
            assert agg[f"{key}_weighted_mean"] == pytest.approx(
                sum(v * p for v, p in zip(vals, pops)) / total_pop, abs=1e-9)
        # per-scenario rows and weighted aggregates stay separate: the
        # weighted mean of a heterogeneous panel differs from the plain mean
        # whenever populations vary (they do), so assert both are present and
        # the assignment outcome provenance is explicit
        for d in detail:
            assert d["assigned_arm"] in ("damping", "h5")
            if d["assigned_arm"] == "h5":
                assert d["outcome_provenance"] == \
                    rows_by_seed[d["seed"]]["h5_provenance"]
            else:
                assert d["outcome_provenance"] == "damping_reference"


def test_comparator_assignments_and_counts(comparators):
    damping = comparators["comparators"]["damping"]
    always = comparators["comparators"]["always_h5"]
    hind = comparators["comparators"][
        "hindsight_h5_if_beneficial_OPPORTUNITY_BOUND"]
    assert damping["selection_counts"] == {
        "beneficial_retained": 0, "beneficial_missed": 8,
        "harmful_executed": 0, "harmful_avoided": 20,
        "nonbeneficial": 4, "unknown_or_refused": 0}
    assert always["selection_counts"] == {
        "beneficial_retained": 8, "beneficial_missed": 0,
        "harmful_executed": 20, "harmful_avoided": 0,
        "nonbeneficial": 4, "unknown_or_refused": 0}
    assert hind["selection_counts"] == {
        "beneficial_retained": 8, "beneficial_missed": 0,
        "harmful_executed": 0, "harmful_avoided": 20,
        "nonbeneficial": 4, "unknown_or_refused": 0}
    assert {d["seed"] for d in hind["per_scenario"]
            if d["assigned_arm"] == "h5"} == {
        d["seed"] for d in always["per_scenario"]
        if d["label"] == "beneficial"}
    assert "oracle" in comparators["nature"].lower()
    assert "ORACLE" in hind["nature"]
    assert hind["selection_counts"]["harmful_executed"] == 0


def test_relational_blocks_verbatim_and_hash_bound(rows_by_seed):
    dev = load(ROOT / "output/observation_relational_v1/"
                    "DATASET_development.json")
    hold = load(ROOT / "output/observation_relational_v1/"
                     "DATASET_holdout.json")
    src = {}
    for ds in (dev, hold):
        for row in ds["rows"]:
            if row.get("entry_relational"):
                src[row["seed"]] = row["entry_relational"]
    for seed, row in rows_by_seed.items():
        er = row["entry_relational"]
        assert er == src[seed]
        assert er["schema"] == "oe34_relational_observation/1"
        assert er["decision_time_s"] == 600.0
        assert len(er["numeric_tensor"]) == 6
        assert len(er["numeric_tensor"][0]) == 3
        assert len(er["numeric_tensor"][0][0]) == 16
        assert len(er["missingness_tensor"]) == 6
        assert len(er["missingness_tensor"][0][0]) == 5
        assert len(er["lags_s"]) == 3
        assert len(er["numeric_columns"]) == 16
        assert len(er["slots"]) == 6
        rel_sha = er["source_sha256"]["telemetry.jsonl"]
        h = hashlib.sha256(
            (OE34 / f"s{seed}_damping/telemetry.jsonl").read_bytes()).hexdigest()
        assert rel_sha == h
        assert row["verification"]["telemetry_sha256_matches_relational_source"]


def test_snapshot_scalars_match_damping_telemetry(rows_by_seed):
    """Recompute the 4 OE-20 scalars independently for 8 groups spanning both
    splits and both provenance classes (SPEC section 3)."""
    check = [90004, 90008, 90034, 90036, 90037, 90041, 90050, 90060]
    for seed in check:
        row = rows_by_seed[seed]
        members = set(row["binding_evidence"]["member_ids"])
        fresh = {}
        with open(OE34 / f"s{seed}_damping/telemetry.jsonl",
                  encoding="utf-8") as fh:
            for line in fh:
                if "rsu_received_report" not in line:
                    continue
                rec = json.loads(line)
                if rec.get("kind") != "rsu_received_report":
                    continue
                if rec["received_at_s"] > 600.0:
                    continue
                ts = rec["frame"]["timestamp_s"]
                if ts < 599.0 or ts > 600.0:
                    continue
                vid = rec["frame"]["vehicle_id"]
                if vid not in fresh or (ts, rec["received_at_s"]) > \
                        (fresh[vid][0], fresh[vid][1]):
                    fresh[vid] = (ts, rec["received_at_s"], rec["frame"])
        frames = [f[2] for f in fresh.values()]
        speeds = [f["speed_mps"] for f in frames]
        snap = row["snapshot"]
        assert snap["reporter_count"] == len(frames)
        assert snap["all_mean_speed_mps"] == pytest.approx(
            sum(speeds) / len(speeds), abs=1e-9)
        c = [f for f in frames if f["vehicle_id"] in members]
        c_speeds = [f["speed_mps"] for f in c]
        assert snap["cohort_mean_speed_mps"] == pytest.approx(
            sum(c_speeds) / len(c_speeds), abs=1e-9)
        near = sum(1 for f in frames
                   if f.get("distance_to_merge_m") is not None
                   and abs(f["distance_to_merge_m"]) <= 100.0)
        assert snap["near_merge_reporter_count"] == near
        mean = sum(c_speeds) / len(c_speeds)
        allowed = [f["allowed_speed_mps"] for f in c
                   if f.get("allowed_speed_mps") is not None]
        target = min(24.0, min(allowed, default=24.0), max(0.0, mean - 2.0))
        assert snap["required_slowdown_mps"] == pytest.approx(
            mean - target, abs=1e-9)


def test_no_outcome_based_exclusion(dataset, rows_by_seed):
    """Population membership must be computable without reading any outcome
    field: rebuild it from registered feasibility/provenance only and compare
    with the dataset; also verify all labels/deltas were carried for ALL 32
    groups (nobody dropped or added for their label)."""
    membership = set()
    for seed in range(90001, 90061):
        b5 = load(OE34 / f"bindings/s{seed}_h5.json")
        if (not b5["unavailable"]) or \
                b5["reason"] == "hold not active at HOLD+5":
            membership.add(seed)
    assert membership == {r["seed"] for r in dataset["rows"]}
    labels = {}
    for row in dataset["rows"]:
        labels[row["seed"]] = row["label_block"]["label"]
        assert row["physical_deltas"] is not None
        assert row["arms"]["damping"]["costs_per_request"] is not None
        assert row["arms"]["h5"]["costs_per_request"] is not None
    assert sorted(list(labels.values()).count(k) for k in
                  ("beneficial", "harmful", "nonbeneficial")) == [4, 8, 20]


def test_missingness_inventory_explicit(dataset):
    assert dataset["missingness_summary"]["snapshot"][
        "groups_all_mean_missing"] >= 0
    for row in dataset["rows"]:
        inv = row["missingness"]
        assert inv["snapshot"]["cohort_size"] == \
            inv["snapshot"]["cohort_members_fresh_count"] + \
            (inv["snapshot"]["cohort_size"]
             - inv["snapshot"]["cohort_members_fresh_count"])
        rel = inv["relational"]
        assert rel["present"] is True
        assert rel["numeric_cells_total"] == 288
        assert 0.0 <= rel["numeric_missing_fraction"] <= 1.0
        assert rel["numeric_missing_fraction"] == pytest.approx(
            rel["numeric_cells_missing"] / 288, abs=1e-12)
        assert inv["h5_counterfactual"]["provenance"] == row["h5_provenance"]
    rel = dataset["missingness_summary"]["relational"]
    assert rel["numeric_cells_total"] == 32 * 288


def test_sidecar_hashes_bind_dataset_and_inputs():
    sidecar = load(OUT / "SHA256SIDECAR.json")
    h = hashlib.sha256(
        (OUT / "DATASET_OE46.json").read_bytes()).hexdigest()
    assert sidecar["dataset_sha256"] == h
    manifest = load(OUT / "SOURCE_MANIFEST.json")
    paths = [e["path"] for e in manifest["inputs"]]
    assert set(paths) == set(sidecar["input_file_sha256"])
    for entry in manifest["inputs"]:
        actual = hashlib.sha256(
            (ROOT / entry["path"]).read_bytes()).hexdigest()
        assert actual == entry["sha256"] == \
            sidecar["input_file_sha256"][entry["path"]], entry["path"]
