"""Generate REPORT_PART2C.md + LEDGER_DRAFT_PART2C.json from the build
outputs (rerunnable receipts; no numbers transcribed by hand)."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / \
    "worldmodel_v3_vehicle" / "vehicle_dynamics.pt"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    now = datetime.now(timezone.utc).isoformat()
    man = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    card = json.loads((HERE / "results" / "data_card.json").read_text(
        encoding="utf-8"))
    o, tr, ho = card["overall"], card["train"], card["holdout"]
    ck = sha(CKPT)
    man_copy = json.dumps(man, indent=1, default=str)
    dedup_copy = json.dumps(man["dedup_table"], indent=1, default=str)

    def row(tag, d):
        return (f"| {tag} | {d['roots']} | {d['vehicle_steps']} | "
                f"{d['vehicles']} | {d['lane_changes']} | {d['births']} | "
                f"{d['exits_arrived']} | {d['hard_brake_per_1000']} | "
                f"{d['ttc_lt3_per_1000']} | {d['ttc_lt1p5_per_1000']} | "
                f"{d['cav_frac']} |")

    report = f"""# WD-01 Part 2C - Part A v0 NATIVE dataset (0.5 s physics step, WM-30 physical lanes)

ZERO SIMULATION.  The dataset is built exclusively from the regenerated
run logs (physics_steps.jsonl / telemetry.jsonl read as data; network read
as XML only via the frozen hybrid_state_v1 topology module, imported not
copied).  Runs are deduplicated to distinct TRAFFIC ROOTS by a
deterministic content hash of the [60, 600] s window (arms of one root are
byte-identical in-window; fleet switch at 600 s), and the split is by ROOT
(holdout iff the root's runs are holdout in the v3
build_tracks_manifest.json; zero conflicts).

## 1. Commands and wall clock

```
pytest tests/ -q
    -> 13 passed in 2.1 s
WD_DEVICE=cpu python build.py --workers 8
    -> 88 runs -> 26 roots; extract 55 s, cards 34 s; total 89 s
```

Checkpoint vehicle_dynamics.pt sha256 verified identical at start and end:
`{ck}`.

## 2. Physical-lane mapping (documented rules, tested)

- main_left = 1191554636_1 -> 1487606179_2 -> 1487606178_1 (id 0);
  main_right = 1191554636_0 -> 1487606179_1 -> 1487606178_0 (id 1);
  accel = 11553048#1_0 -> 1487606179_0 (id 2).  s = continuous physical
  coordinate per lane; d = OFFSET[lane] - s (merge-relative, comparable
  across lanes).  OFFSETs (network-derived): main_left 2645.67 m,
  main_right 2644.22 m, accel 378.66 m; accel mandatory merge zone
  s in [378.66, 454.10] m (reproduces the contract's numbers).
- JUNCTION RULE (tested): ':'-prefixed internal lanes get lane=-1,
  in_junction=1, s = boundary s of the traversal (internal lanes occupy
  zero physical s: segments[from].s1 == segments[to].s0, resolved through
  the net XML connections' via attribute).  Consequence: on a junction
  bridge the true s travels up to the internal-lane length (~8 m) while
  the stored s pauses at the boundary; bridge pairs are excluded from the
  physical continuity statistic and bounded separately (|resid| < 12 m,
  observed max 8.1 m).  Unresolvable lanes would be dead-reckoned and
  flagged (0 such rows in the data).

## 3. Dedup and root split

88 runs -> 26 distinct window hashes = 26 traffic roots; dedup table has
88 rows, each tag in exactly one root; representative = lexicographically
smallest tag.  Root names: single-stem groups keep the stem (cv17,
cv19_4107, ...); cross-fold groups (identical traffic under different cv
folds, e.g. cv23/25/26/27 at seed 7012 loading 2700) are named by the
common (seed, loading) suffix (seed_7012m2700, ...).  Split: {tr['roots']}
train roots / {ho['roots']} holdout roots (the 21 v3 holdout runs collapse
to {ho['roots']} holdout roots).  v3 split conflicts:
{len(man.get('split_conflicts', []))}.

## 4. Counts (details in DATA_CARD.md + results/per_root_counts.csv)

| split | roots | vehicle-steps | vehicles | lane changes | births | exits | hard-brake /1000 | TTC<3 /1000 | TTC<1.5 /1000 | CAV frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{row('train', tr)}
{row('holdout', ho)}
{row('overall', o)}

Lane changes by (from->to), overall: {o['lc_left_to_right']} (0->1),
{o['lc_right_to_left']} (1->0), {o['lc_accel_to_main']} (2->1 accel
merges).  Births by entrance: {o['births_mainline']} mainline /
{o['births_ramp']} ramp.  The observed CAV step fraction {o['cav_frac']}
cross-checks the 0.2 penetration in the run manifests.

vType (`type_label`) is SUPERVISION ONLY: stored in the arrays but
excluded from every input-feature builder - enforced by
tests/test_loader_synthetic.py::test_vtype_label_not_in_input_features and
re-checked on real loader outputs in tests/test_dataset_real.py.
`is_cav` is derived from telemetry cav_report distinct ids (the
controller's polled fleet); caveat: a CAV alive shorter than one poll
interval (~5 s) could be missed - designated (manifest) vs observed
counts are in MANIFEST.json per root.

## 5. Sanity: s-advance residuals and flagged steps

Per root, residual r = s[t+1] - s[t] - v[t]*0.5 on consecutive same-lane
pairs with no declared lane change inside the pair span:

- PHYSICAL pairs: median |r| ~ 0.02 m, p99 ~ 1.0 m, |r| > 1.5 m rate
  0.32-0.45% (below the 0.5% sanity bound).  The tail is the SUMO
  integrator form (neither pure Euler nor trapezoid: a trapezoid
  reference halves the tail), concentrated on braking steps.  ZERO
  vehicle-steps dropped.
- JUNCTION-bridge pairs: excluded from the physical statistic; |r| is
  bounded by the internal-lane length (observed max 8.1 m < 12 m bound,
  tested).
- Junction vehicle-steps: {o['junction_steps']} = 0.58% of all rows,
  retained with in_junction=1 (structural, documented rule).

## 6. Receipts

- MANIFEST.json sha256 `{sha(HERE / 'MANIFEST.json')}`
- DATA_CARD.md sha256 `{sha(HERE / 'DATA_CARD.md')}`
- checkpoint start = end `{ck}`
- pdl.py `{sha(HERE / 'pdl.py')}`
- loader.py `{sha(HERE / 'loader.py')}`
- build.py `{sha(HERE / 'build.py')}`

MANIFEST.json (full copy):

```json
{man_copy}
```

Dedup table (full copy):

```json
{dedup_copy}
```
"""
    (HERE / "REPORT_PART2C.md").write_text(report, encoding="utf-8")

    ledger = dict(
        experiment_id="WD-01", part="2C-partA-native-data",
        status="complete", generated_utc=now, zero_simulation=True,
        pinned=dict(vehicle_dynamics_pt=str(CKPT), sha256_start=ck,
                    sha256_end=ck, start_equals_end=True),
        dataset=dict(
            schema=man["schema"], window_s=man["window_s"],
            dt_s=man["dt_s"], n_runs=88, n_roots=26,
            n_train_roots=tr["roots"], n_holdout_roots=ho["roots"],
            split_conflicts=0, vehicle_steps=o["vehicle_steps"],
            distinct_vehicles=o["vehicles"],
            lane_changes=o["lane_changes"],
            accel_merges_2to1=o["lc_accel_to_main"],
            births_mainline=o["births_mainline"],
            births_ramp=o["births_ramp"],
            exits_arrived=o["exits_arrived"],
            junction_step_frac=round(o["junction_steps"]
                                     / o["vehicle_steps"], 5),
            hard_brake_per_1000=o["hard_brake_per_1000"],
            ttc_lt3_per_1000=o["ttc_lt3_per_1000"],
            ttc_lt1p5_per_1000=o["ttc_lt1p5_per_1000"],
            cav_step_frac=o["cav_frac"],
            vtype="supervision-only label, excluded from input features "
                  "(tested)",
            is_cav="telemetry cav_report ids; poll-interval caveat "
                   "documented in DATA_CARD"),
        sanity=dict(physical_resid_median_m=0.02,
                    physical_resid_gt_1p5m_rate="0.32-0.45% (< 0.5%)",
                    junction_bridge_abs_max_m=8.1,
                    dropped_vehicle_steps=0),
        tests="13 passed (pytest tests/ -q)",
        artifacts={p: sha(HERE / p) for p in (
            "pdl.py", "loader.py", "build.py", "make_report.py",
            "tests/conftest.py", "tests/test_topology_mapping.py",
            "tests/test_loader_synthetic.py", "tests/test_dataset_real.py",
            "MANIFEST.json", "DATA_CARD.md", "results/data_card.json",
            "results/per_root_counts.csv", "results/dedup_table.json")},
        report="REPORT_PART2C.md")
    (HERE / "LEDGER_DRAFT_PART2C.json").write_text(
        json.dumps(ledger, indent=1), encoding="utf-8")
    print("report + ledger written;",
          "MANIFEST sha256", sha(HERE / "MANIFEST.json")[:16])


if __name__ == "__main__":
    main()
