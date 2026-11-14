"""Schema/3 TRAIN outcome OVERLAY sidecar tests (synthetic; no D:, no real
pack/labels/shards).

Builds a tiny mixed-split pack + a /1 sidecar -> real /2 rebind sidecar +
synthetic shard trees (meta.json + raw physics one-liner + v0.4a label
parquets), then proves:

* the overlay builder accepts zero-row TRAIN items whose shard snapshot
  sits at t0 + 0.5 s and refuses (with a receipt reason) any item whose
  first physics row is NOT t0 + 0.5 (alignment refusal; never shifted);
* owner decision (b): base/native_base items are never overlaid;
* the loader re-derives the pack plan, enforces identity and coverage,
  and refuses tampered overlay rows / uncovered items;
* train.dataset consumes WM_N1_TRAIN_OUTCOMES pointing at the schema/3
  sidecar: zero-row items now carry t0-aligned outcome supervision and
  label_t0 telemetry, excluded arms stay zeroed, and the fixed1 treatment
  becomes an outcome-supported pair member.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train import label_isolation as LI
from train import label_overlay as LO
from train.dataset import CorpusRefusal, MemmapCorpus, contrast_pairs

TAUS = (30.0, 60.0)


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# zero-row TRAIN items: w9 fixed1 (aligned), w9 wall arm (aligned),
# w8 base (owner-b), w7 fixed1 (MISALIGNED shard snapshot).
RUNS = [
    dict(batch="chunk4", chunk="chunk4", world="w9", arm="native",
         policy="native", split="TRAIN", t0=300.0, n=2, tick=0, plan=0),
    dict(batch="chunk2", chunk="chunk2", world="w9", arm="fixed1",
         policy="fixed", split="TRAIN", t0=300.0, n=0, tick=1, plan=1),
    dict(batch="walls_pilot_chunk4", chunk="chunk4", world="w9",
         arm="w-lo-early", policy="wall", split="TRAIN", t0=300.0, n=0,
         tick=2, plan=2),
    dict(batch="chunk2", chunk="chunk2", world="w8", arm="base",
         policy="damping", split="TRAIN", t0=300.0, n=0, tick=3, plan=3),
    dict(batch="chunk2", chunk="chunk2", world="w7", arm="fixed1",
         policy="fixed", split="TRAIN", t0=250.0, n=0, tick=4, plan=4),
    dict(batch="chunk4", chunk="chunk4", world="c1", arm="native",
         policy="native", split="CALIBRATION", t0=100.0, n=1, tick=5,
         plan=5),
]


def _layout():
    first = 0
    out = []
    for run in RUNS:
        out.append(dict(run, first=first))
        first += run["n"]
    return out


def _write_pack(pack: Path):
    pack.mkdir(parents=True, exist_ok=True)
    items, runs = [], []
    for run in _layout():
        rel = f"{run['batch']}/{run['world']}/{run['arm']}"
        rows = list(range(run["first"], run["first"] + run["n"]))
        items.append(dict(
            world=run["world"], arm=run["arm"], chunk=run["chunk"],
            batch=run["batch"], split=run["split"], t0_s=run["t0"],
            pair_key=f"{run['world']}@t0={run['t0']:g}",
            policy=run["policy"], tick=run["tick"], plan_row=run["plan"],
            outcome_rows=rows))
        runs.append(dict(key=f"{run['batch']}/{run['world']}__{run['arm']}",
                         shard_dir=rel, split=run["split"],
                         n_outcome_rows=run["n"]))
    manifest = dict(schema="wm_n1_pack/1", k_max=8,
                    arrays={}, outcomes={"file": "outcomes.parquet"},
                    items=items)
    index = dict(schema="wm_n1_corpus_pack/2", k_max=8, runs=runs)
    (pack / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")
    (pack / "INDEX.json").write_text(json.dumps(index, indent=1),
                                     encoding="utf-8")


def _pinned_rows():
    pin = ["world", "source", "chunk_id", "split", "t0_s", "policy", "t",
           "tau", "tau_available", "total_delay_ref_s", "ttc_lt3_veh_s",
           "hard_brake_s", "collision_any", "_global_row_id"]
    rows = []
    rid = 0
    for run in _layout():
        if run["n"] == 0:
            continue
        for tau in TAUS[:run["n"]]:
            rows.append(dict(
                world=run["world"], source=run["arm"],
                chunk_id=run["chunk"], split=run["split"],
                t0_s=run["t0"], policy="native_fleet", t=run["t0"],
                tau=tau, tau_available=100.0, total_delay_ref_s=5.0,
                ttc_lt3_veh_s=1.0, hard_brake_s=0.0, collision_any=0,
                _global_row_id=rid))
            rid += 1
    return pd.DataFrame(rows, columns=pin)


def _write_input_sidecar(sidecar: Path, pack: Path):
    sidecar.mkdir(parents=True, exist_ok=True)
    frame = _pinned_rows()
    frame.to_parquet(sidecar / LI.SIDECAR_PARQUET_NAME, index=False)
    plan = LI.plan_sidecar(pack, None)
    runs, items = [], []
    for run in plan["train_runs"]:
        runs.append(dict(
            key=run["key"], split=run["split"], shard_rel=run["shard_rel"],
            shard_dir_recorded=run["shard_dir"],
            input_sha256=run["input_sha256"],
            outcomes_sha256=("a" * 64) if run["n"] else None,
            first_global_row=run["first"], n_outcome_rows=run["n"]))
        it = run["item"]
        items.append(dict(
            world=it["world"], arm=it["arm"], chunk=it["chunk"],
            batch=it.get("batch"), split=it["split"], t0_s=it["t0_s"],
            pair_key=it["pair_key"], policy=it.get("policy"),
            tick=int(it["tick"]), plan_row=int(it["plan_row"]),
            first_global_row=run["first"], n_rows=run["n"]))
    receipt = dict(
        schema=LI.SIDECAR_SCHEMA, created_utc="2026-10-04T00:00:00Z",
        split="TRAIN", tool="test",
        pack=dict(dir=str(pack), manifest_name="manifest.json",
                  manifest_sha256=_sha(pack / "manifest.json"),
                  index_name="INDEX.json",
                  index_sha256=_sha(pack / "INDEX.json"),
                  k_max=8, format=None, n_items=len(plan["items"]),
                  n_items_train=plan["selection"]["n_items_train"],
                  n_items_other=plan["selection"]["n_items_other"],
                  n_rows_total=plan["n_rows_total"]),
        selection=dict(plan["selection"], shards_root="unused",
                       policy="test"),
        runs=runs, items=items,
        sidecar=dict(file=LI.SIDECAR_PARQUET_NAME,
                     sha256=_sha(sidecar / LI.SIDECAR_PARQUET_NAME),
                     n_rows=int(len(frame)),
                     parquet_index=LI.ROW_ID_COLUMN,
                     columns={c: str(d) for c, d in frame.dtypes.items()}))
    (sidecar / LI.RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")


def _policy_of(source):
    return {"fixed1": "fixed_mix", "w-lo-early": "wall"}.get(source,
                                                             "native_fleet")


def _kind_of(source):
    return {"fixed1": "fixed", "w-lo-early": "wall"}.get(source,
                                                         "native_fleet")


def _v04a_labels(root: Path, chunk: str, world: str, source: str,
                 t0: float, split="TRAIN"):
    rows = []
    for t in (t0, t0 + 30.0):
        for tau in TAUS:
            rows.append(dict(
                world=world, source=source, chunk_id=chunk, split=split,
                demand_veh_h=2700, penetration=0.1, p40=False,
                pre_fix=False, t0_s=t0, policy=_policy_of(source), t=t,
                tau=tau, tau_available=100.0, cav_speed_mode=27,
                time_on_section_s=1.0, free_flow_ref_s=0.5, traversed_m=20.0,
                wait_mainline_s=0.0, wait_ramp_s=0.0, total_delay_ref_s=0.5,
                ttc_lt3_veh_s=0.0, hard_brake_s=0.0, collision_any=0,
                collision_n_veh=0, teleports=0, n_queued_at_horizon=0,
                n_vehicles_present=7, n_unmatched_insertions=0,
                pi_native=1.0, pi_damping=0.0, pi_pi=0.0, role="primary",
                aux=False, arm_kind=_kind_of(source)))
    d = root / "worlds" / chunk
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        d / f"{world}__{source}_labels.parquet", index=False)


def _write_shard(shards: Path, run: dict, *, snapshot_t: float):
    sd = shards / run["batch"] / run["world"] / run["arm"]
    sd.mkdir(parents=True, exist_ok=True)
    physics = sd / "physics.jsonl"
    physics.write_text(json.dumps({"time_s": snapshot_t}) + "\n",
                       encoding="utf-8")
    meta = dict(
        chunk=run["chunk"], batch=run["batch"], world=run["world"],
        arm=run["arm"], split=run["split"], policy=run["policy"],
        t0_s=run["t0"],
        t_origin=(run["t0"] if run["arm"] != "base" else 0.0),
        n_ticks=100, data_end=run["t0"] + 100.0,
        decision_t=[run["t0"], run["t0"] + 30.0],
        input_physics=str(physics))
    (sd / "meta.json").write_text(json.dumps(meta, indent=1),
                                  encoding="utf-8")


def _build_side2(env) -> Path:
    from train import label_rebind as LR

    def writer(path, obj):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, indent=1, sort_keys=True))

    orig = LR._write_json
    LR._write_json = writer
    try:
        out2 = env["tmp"] / "sidecar2_rebound"
        LR.build_rebound_sidecar(
            env["pack"], env["shards"], env["sidecar1"], env["frozen"],
            out2, "test-commit",
            _sha(env["sidecar1"] / LI.RECEIPT_NAME),
            _sha(env["sidecar1"] / LI.SIDECAR_PARQUET_NAME))
    finally:
        LR._write_json = orig
    return out2


@pytest.fixture()
def env(tmp_path):
    pack = tmp_path / "pack"
    shards = tmp_path / "shards"
    frozen = tmp_path / "v04a_frozen"
    cov = tmp_path / "v04a_cov"
    sidecar1 = tmp_path / "sidecar1"
    _write_pack(pack)
    for run in _layout():
        _write_shard(shards, run,
                     snapshot_t=(run["t0"] + 0.5 if run["arm"] != "base"
                                 else 0.5))
    # w7 fixed1 is the MISALIGNED item (snapshot at t0+1.0)
    (shards / "chunk2" / "w7" / "fixed1" / "physics.jsonl").write_text(
        json.dumps({"time_s": 251.0}) + "\n", encoding="utf-8")
    _v04a_labels(frozen, "chunk4", "w9", "native", 300.0)
    _v04a_labels(frozen, "chunk4", "c1", "native", 100.0, split="CALIBRATION")
    _v04a_labels(cov, "chunk2", "w9", "fixed1", 300.0)
    _v04a_labels(cov, "chunk4", "w9", "w-lo-early", 300.0)
    _v04a_labels(cov, "chunk2", "w7", "fixed1", 250.0)
    (frozen / "SHA256SUMS").write_text("", encoding="utf-8")
    (cov / "SHA256SUMS").write_text("", encoding="utf-8")
    _write_input_sidecar(sidecar1, pack)
    e = dict(tmp=tmp_path, pack=pack, shards=shards, frozen=frozen, cov=cov,
             sidecar1=sidecar1, out=tmp_path / "sidecar3")
    e["input_sidecar"] = _build_side2(e)
    e["receipt_sha"] = _sha(e["input_sidecar"] / LI.RECEIPT_NAME)
    e["parquet_sha"] = _sha(e["input_sidecar"] / LI.SIDECAR_PARQUET_NAME)
    return e


def _build_side3(env):
    return LO.build_overlay_sidecar(
        env["pack"], env["shards"], env["input_sidecar"],
        env["receipt_sha"], env["parquet_sha"], env["cov"], env["out"],
        "commit-test", frozen_root=env["frozen"])


def _pos_of_arm(corpus, world, arm):
    for pos in range(len(corpus)):
        it = corpus._items[corpus._order[pos]]
        if it["world"] == world and it["arm"] == arm:
            return pos
    raise AssertionError(f"no item {world}/{arm}")


# ---------------------------------------------------------------- builder
def test_build_accepts_aligned_and_refuses_misaligned(env):
    rep = _build_side3(env)
    assert rep["n_accepted_groups"] == 2          # w9 fixed1 + wall arm
    assert rep["n_excluded_groups"] == 2          # base (b) + misaligned w7
    assert rep["n_zero_row_items"] == 4
    assert rep["alignment"]["n_checked"] == 3
    assert rep["alignment"]["n_refused"] == 1
    receipt = json.loads((env["out"] / LO.RECEIPT_NAME)
                         .read_text(encoding="utf-8"))
    reasons = {e["key"]: e["reason"] for e in receipt["overlay"]["excluded"]}
    assert "alignment" in reasons["chunk2/w7__fixed1"]
    assert "owner decision (b)" in reasons["chunk2/w8__base"]
    counts = receipt["overlay"]["counts_by_batch_arm"]
    assert counts["chunk2|fixed1"]["labelled"] == 1
    assert counts["chunk2|fixed1"]["excluded"] == 1
    assert counts["chunk2|base"] == dict(labelled=0, excluded=1,
                                         excluded_reasons=counts[
                                             "chunk2|base"][
                                             "excluded_reasons"])


def test_build_refuses_existing_out(env):
    _build_side3(env)
    with pytest.raises(LO.LabelOverlayError):
        _build_side3(env)


# ----------------------------------------------------------------- loader
def test_loader_and_dataset_use_overlay(env):
    _build_side3(env)
    report = LI.verify_sidecar(env["pack"], env["out"])
    assert report["ok"]
    assert report["schema"] == LO.SIDECAR_SCHEMA_V3
    assert report["overlay"]["n_accepted_groups"] == 2

    groups, receipt = LI.load_train_overlay(env["pack"], env["out"])
    assert set(groups) == {
        ("chunk2", "chunk2", "w9", "fixed1", "TRAIN", 300.0),
        ("walls_pilot_chunk4", "chunk4", "w9", "w-lo-early", "TRAIN",
         300.0)}
    assert receipt["schema"] == LO.SIDECAR_SCHEMA_V3

    monkey = pytest.MonkeyPatch()
    monkey.setenv("WM_N1_TRAIN_OUTCOMES", str(env["out"]))
    try:
        corpus = MemmapCorpus(env["pack"], split="TRAIN")
        # zero-row fixed1 item now has t0-aligned supervision
        i = _pos_of_arm(corpus, "w9", "fixed1")
        targets, mask, _col = corpus.outcome_target(i)
        assert mask.any() and np.any(np.asarray(targets) != 0)
        # label_t0 telemetry on the delivered rows (the synthetic pack
        # carries no tensors, so __getitem__ is out of scope here)
        from train.dataset import label_t0_rows
        it = corpus._items[corpus._order[i]]
        rows = corpus._select_outcome_rows(
            corpus._outcomes(),
            [int(p) for p in it["outcome_rows"] if int(p) >= 0], item=it)
        assert label_t0_rows(rows, it["t0_s"]) == 2
        # the wall item too
        j = _pos_of_arm(corpus, "w9", "w-lo-early")
        _t2, m2, _c2 = corpus.outcome_target(j)
        assert m2.any()
        # the misaligned item stayed excluded -> all-False mask
        k = _pos_of_arm(corpus, "w7", "fixed1")
        _t3, m3, _c3 = corpus.outcome_target(k)
        assert not m3.any()
        # owner-(b) base stays zeroed
        b = _pos_of_arm(corpus, "w8", "base")
        _t4, m4, _c4 = corpus.outcome_target(b)
        assert not m4.any()
        # the fixed1 treatment AND the wall treatment are now
        # outcome-supported pair members of the w9 group
        pairs = contrast_pairs(corpus._items)
        assert len(pairs) == 2
        supported = 0
        for a, _ref in pairs:
            _t5, m5, _c5 = corpus.outcome_target(a)
            if m5.any():
                supported += 1
        assert supported == 2
    finally:
        monkey.undo()


def test_loader_refuses_tampered_overlay(env):
    _build_side3(env)
    ov = pd.read_parquet(env["out"] / LO.OVERLAY_PARQUET_NAME)
    ov.loc[ov.index[0], "total_delay_ref_s"] = 12345.0
    ov.to_parquet(env["out"] / LO.OVERLAY_PARQUET_NAME, index=False)
    # the receipt still binds the ORIGINAL bytes -> sha check fires
    with pytest.raises(LI.LabelIsolationError):
        LI.load_train_overlay(env["pack"], env["out"])
    # rewrite the receipt + SUMS binding too, then tamper an IDENTITY
    # column (values are only SHA-bound; identity is re-checked per row)
    ov = pd.read_parquet(env["out"] / LO.OVERLAY_PARQUET_NAME)
    ov.loc[ov.index[0], "world"] = "tampered"
    ov.to_parquet(env["out"] / LO.OVERLAY_PARQUET_NAME, index=False)
    receipt = json.loads((env["out"] / LO.RECEIPT_NAME)
                         .read_text(encoding="utf-8"))
    receipt["sidecar"]["sha256"] = _sha(env["out"]
                                        / LO.OVERLAY_PARQUET_NAME)
    (env["out"] / LO.RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
    sums = [f"{receipt['sidecar']['base_sha256']}  "
            f"{LO.BASE_PARQUET_NAME}",
            f"{receipt['sidecar']['sha256']}  {LO.OVERLAY_PARQUET_NAME}",
            f"{_sha(env['out'] / LO.RECEIPT_NAME)}  {LO.RECEIPT_NAME}"]
    (env["out"] / LO.SUMS_NAME).write_text("\n".join(sums) + "\n",
                                           encoding="utf-8")
    with pytest.raises(LI.LabelIsolationError,
                       match="row count differs|identity mismatch"):
        LI.load_train_overlay(env["pack"], env["out"])


def test_loader_refuses_uncovered_item(env):
    _build_side3(env)
    groups, _ = LI.load_train_overlay(env["pack"], env["out"])
    key = ("chunk2", "chunk2", "w9", "fixed1", "TRAIN", 300.0)
    assert len(groups[key]) == 4
    receipt = json.loads((env["out"] / LO.RECEIPT_NAME)
                         .read_text(encoding="utf-8"))
    ov = pd.read_parquet(env["out"] / LO.OVERLAY_PARQUET_NAME)
    keep = ~((ov["batch"] == "chunk2") & (ov["arm"] == "fixed1")
             & (ov["world"] == "w9"))
    ov[keep].to_parquet(env["out"] / LO.OVERLAY_PARQUET_NAME, index=False)
    receipt["overlay"]["groups"] = [
        g for g in receipt["overlay"]["groups"]
        if not (g["batch"] == "chunk2" and g["arm"] == "fixed1"
                and g["world"] == "w9")]
    receipt["overlay"]["n_accepted_groups"] = len(
        receipt["overlay"]["groups"])
    receipt["overlay"]["n_accepted_rows"] = int(len(ov[keep]))
    receipt["sidecar"]["sha256"] = _sha(env["out"]
                                        / LO.OVERLAY_PARQUET_NAME)
    receipt["sidecar"]["n_rows"] = int(len(ov[keep]))
    (env["out"] / LO.RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
    sums = [f"{receipt['sidecar']['base_sha256']}  "
            f"{LO.BASE_PARQUET_NAME}",
            f"{receipt['sidecar']['sha256']}  {LO.OVERLAY_PARQUET_NAME}",
            f"{_sha(env['out'] / LO.RECEIPT_NAME)}  {LO.RECEIPT_NAME}"]
    (env["out"] / LO.SUMS_NAME).write_text("\n".join(sums) + "\n",
                                           encoding="utf-8")
    with pytest.raises(LI.LabelIsolationError, match="coverage gap"):
        LI.load_train_overlay(env["pack"], env["out"])


def test_dataset_refuses_without_sidecar(env, monkeypatch):
    monkeypatch.delenv("WM_N1_TRAIN_OUTCOMES", raising=False)
    corpus = MemmapCorpus(env["pack"], split="TRAIN")
    with pytest.raises(CorpusRefusal):
        corpus.outcome_target(0)
