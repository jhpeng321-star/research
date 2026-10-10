"""WM-N1 label-rebind tests (synthetic only; no D:, no real pack/labels).

Builds a tiny hand-made TRAIN pack (metadata + one /1 sidecar), a synthetic
v0.4a label root and TRAIN shard ``meta.json`` files, then proves:

* a campaign chunk-2 ``native``/``switching`` item is rebound from ``native``
  to ``swnative`` with the ``switching_mix`` policy;
* unchanged runs are copied value-identically;
* a run whose rebind fails is EXCLUDED (rows removed, reason recorded) and
  the dataset sees zero outcome rows for it;
* a zero-row item whose fixed lookup WOULD find labels is listed, not added;
* the /2 loader refuses tampered rebind blocks and row identity, and the /1
  behaviour is untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from train import label_isolation as LI
from train import label_rebind as LR
from train.dataset import CorpusRefusal, MemmapCorpus


@pytest.fixture(autouse=True)
def _no_rename_json_writer(monkeypatch):
    """The strict synthetic guard forbids rename; finalize the /2 receipt and
    report by exclusive creation (identical payload bytes)."""
    def writer(path, obj):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(obj, indent=1, sort_keys=True)
        with open(p, "x", encoding="utf-8") as fh:
            fh.write(body)
        return p
    monkeypatch.setattr(LR, "_write_json", writer)

PIN = ["world", "source", "chunk_id", "split", "t0_s", "policy", "t",
       "tau", "tau_available", "total_delay_ref_s", "_global_row_id"]

RUNS = [
    dict(batch="chunk2", chunk="chunk2", world="c1", arm="native",
         policy="switching", split="TRAIN", t0=300.0, n=2, tick=0, plan=0),
    dict(batch="chunk4", chunk="chunk4", world="w1", arm="base",
         policy="damping", split="TRAIN", t0=540.0, n=1, tick=1, plan=1),
    dict(batch="chunk1", chunk="chunk1", world="w2", arm="native",
         policy="switching", split="TRAIN", t0=100.0, n=0, tick=2, plan=2),
    dict(batch="chunk2", chunk="chunk2", world="w3", arm="pi",
         policy="switching", split="TRAIN", t0=200.0, n=1, tick=3, plan=3),
]


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _layout():
    first = 0
    out = []
    for run in RUNS:
        out.append(dict(run, first=first))
        first += run["n"]
    return out


def _write_pack(pack: Path):
    pack.mkdir(parents=True, exist_ok=True)
    items = []
    runs = []
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
    return manifest, index


def _pinned_rows():
    rows = [
        dict(world="c1", source="native", chunk_id="chunk2", split="TRAIN",
             t0_s=300.0, policy="native_fleet", t=300.0, tau=30.0,
             tau_available=100.0, total_delay_ref_s=5.0, _global_row_id=0),
        dict(world="c1", source="native", chunk_id="chunk2", split="TRAIN",
             t0_s=300.0, policy="native_fleet", t=300.0, tau=60.0,
             tau_available=100.0, total_delay_ref_s=6.0, _global_row_id=1),
        dict(world="w1", source="base", chunk_id="chunk4", split="TRAIN",
             t0_s=540.0, policy="damping_fleet", t=540.0, tau=30.0,
             tau_available=100.0, total_delay_ref_s=4.0, _global_row_id=2),
        dict(world="w3", source="pi", chunk_id="chunk2", split="TRAIN",
             t0_s=200.0, policy="pi_fleet", t=200.0, tau=30.0,
             tau_available=100.0, total_delay_ref_s=7.0, _global_row_id=3),
    ]
    return pd.DataFrame(rows, columns=PIN)


def _write_sidecar(input_sidecar: Path, pack: Path):
    input_sidecar.mkdir(parents=True, exist_ok=True)
    frame = _pinned_rows()
    parquet = input_sidecar / LI.SIDECAR_PARQUET_NAME
    frame.to_parquet(parquet, index=False)
    plan = LI.plan_sidecar(pack, None)
    runs = []
    items = []
    for run in plan["train_runs"]:
        runs.append(dict(
            key=run["key"], split=run["split"], shard_rel=run["shard_rel"],
            shard_dir_recorded=run["shard_dir"],
            input_sha256=run["input_sha256"],
            outcomes_sha256=("e" * 64) if run["n"] else None,
            first_global_row=run["first"], n_outcome_rows=run["n"]))
        it = run["item"]
        items.append(dict(
            world=it["world"], arm=it["arm"], chunk=it["chunk"],
            batch=it.get("batch"), split=it["split"], t0_s=it["t0_s"],
            pair_key=it["pair_key"], policy=it.get("policy"),
            tick=int(it["tick"]), plan_row=int(it["plan_row"]),
            first_global_row=run["first"], n_rows=run["n"]))
    receipt = dict(
        schema=LI.SIDECAR_SCHEMA, created_utc="2026-10-03T00:00:00Z",
        split="TRAIN", tool="test",
        pack=dict(dir=str(pack), manifest_name="manifest.json",
                  manifest_sha256=_sha(pack / "manifest.json"),
                  index_name="INDEX.json",
                  index_sha256=_sha(pack / "INDEX.json"),
                  k_max=8, format=None, n_items=len(items),
                  n_items_train=len(runs), n_items_other=0,
                  n_rows_total=plan["n_rows_total"]),
        selection=dict(plan["selection"], shards_root=str(pack.parent),
                       policy="test"),
        runs=runs, items=items,
        sidecar=dict(file=LI.SIDECAR_PARQUET_NAME, sha256=_sha(parquet),
                     n_rows=int(len(frame)),
                     parquet_index=LI.ROW_ID_COLUMN,
                     columns={c: str(d) for c, d in frame.dtypes.items()}))
    (input_sidecar / LI.RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt


def _write_shards(shards: Path):
    for run in _layout():
        rel = shards / run["batch"] / run["world"] / run["arm"]
        rel.mkdir(parents=True, exist_ok=True)
        (rel / "meta.json").write_text(json.dumps(dict(
            chunk=run["chunk"], batch=run["batch"], world=run["world"],
            arm=run["arm"], split="TRAIN", policy=run["policy"],
            t0_s=run["t0"], decision_t=[run["t0"]])), encoding="utf-8")


def _write_v04a(v04a: Path):
    worlds = v04a / "worlds"
    (worlds / "chunk2").mkdir(parents=True, exist_ok=True)
    (worlds / "chunk1").mkdir(parents=True, exist_ok=True)
    good = pd.DataFrame([
        dict(world="c1", source="swnative", chunk_id="chunk2", split="TRAIN",
             t0_s=300.0, policy="switching_mix", t=300.0, tau=30.0,
             tau_available=100.0, total_delay_ref_s=9.0),
        dict(world="c1", source="swnative", chunk_id="chunk2", split="TRAIN",
             t0_s=300.0, policy="switching_mix", t=300.0, tau=60.0,
             tau_available=100.0, total_delay_ref_s=9.5),
    ], columns=[c for c in PIN if c != "_global_row_id"])
    p_good = worlds / "chunk2" / "c1__swnative_labels.parquet"
    good.to_parquet(p_good, index=False)
    p_would = worlds / "chunk1" / "w2__swnative_labels.parquet"
    good.assign(world="w2", chunk_id="chunk1").to_parquet(p_would, index=False)
    sums = [(p_good, "worlds/chunk2/c1__swnative_labels.parquet"),
            (p_would, "worlds/chunk1/w2__swnative_labels.parquet")]
    (v04a / "SHA256SUMS").write_text(
        "".join(f"{_sha(p)}  {rel}\n" for p, rel in sums), encoding="utf-8")


@pytest.fixture
def env(tmp_path):
    pack = tmp_path / "pack"
    _write_pack(pack)
    input_sidecar = tmp_path / "sidecars_in"
    receipt = _write_sidecar(input_sidecar, pack)
    shards = tmp_path / "shards"
    _write_shards(shards)
    v04a = tmp_path / "v04a"
    _write_v04a(v04a)
    out = tmp_path / "labels_fix1"
    report = LR.build_rebound_sidecar(
        pack, shards, input_sidecar, v04a, out, "deadbeef",
        _sha(input_sidecar / LI.RECEIPT_NAME),
        _sha(input_sidecar / LI.SIDECAR_PARQUET_NAME))
    return dict(pack=pack, shards=shards, input_sidecar=input_sidecar,
                v04a=v04a, out=out, report=report, in_receipt=receipt)


def test_builder_rebinds_switching_item_and_copies_unchanged(env):
    rep = env["report"]
    assert rep["n_rebound_runs"] == 1
    assert rep["n_rebound_rows"] == 2
    assert rep["n_excluded_runs"] == 1
    assert rep["n_excluded_rows"] == 1
    assert rep["n_would_gain_label"] == 1
    assert rep["rebound"][0]["new_source"] == "swnative"
    assert rep["rebound"][0]["new_policy"] == "switching_mix"
    assert rep["excluded"][0]["world"] == "w3"
    assert rep["would_gain_label"][0]["world"] == "w2"

    frame = LI.load_train_outcomes(env["pack"], env["out"])
    assert sorted(int(x) for x in frame.index) == [0, 1, 2]
    assert frame.loc[[0, 1]]["source"].tolist() == ["swnative", "swnative"]
    assert frame.loc[[0, 1]]["policy"].tolist() == \
        ["switching_mix", "switching_mix"]
    assert float(frame.loc[0, "total_delay_ref_s"]) == 9.0
    assert frame.loc[2, "source"] == "base"
    assert 3 not in frame.index


def test_excluded_run_yields_zero_outcome_rows(env):
    frame = LI.load_train_outcomes(env["pack"], env["out"])
    corpus = object.__new__(MemmapCorpus)
    corpus._outcome_mode = "sidecar"
    empty = corpus._select_outcome_rows(frame, [3])
    assert len(empty) == 0
    kept = corpus._select_outcome_rows(frame, [2])
    assert len(kept) == 1


def test_partial_sidecar_item_refuses_but_whole_absent_is_empty(env):
    frame = LI.load_train_outcomes(env["pack"], env["out"])
    corpus = object.__new__(MemmapCorpus)
    corpus._outcome_mode = "sidecar"
    # all present -> the item's rows are returned unchanged
    assert len(corpus._select_outcome_rows(frame, [0, 1])) == 2
    assert len(corpus._select_outcome_rows(frame, [2])) == 1
    # all absent (the excluded run) -> zero rows, never a partial read
    assert len(corpus._select_outcome_rows(frame, [3])) == 0
    # part present, part absent -> the sidecar and pack disagree: fail closed
    with pytest.raises(CorpusRefusal, match="only part"):
        corpus._select_outcome_rows(frame, [2, 3])
    with pytest.raises(CorpusRefusal, match="only part"):
        corpus._select_outcome_rows(frame, [0, 1, 3])


def test_receipt_is_schema_2_with_rebind_block(env):
    receipt = json.loads((env["out"] / LI.RECEIPT_NAME).read_text(
        encoding="utf-8"))
    assert receipt["schema"] == LI.SIDECAR_SCHEMA_V2
    assert receipt["sidecar"]["sha256"] == \
        _sha(env["out"] / LI.SIDECAR_PARQUET_NAME)
    block = receipt["label_rebind"]
    assert block["input_receipt_sha256"] == \
        _sha(env["input_sidecar"] / LI.RECEIPT_NAME)
    assert block["code_commit"] == "deadbeef"
    assert (block["n_rebound_runs"], block["n_rebound_rows"]) == (1, 2)
    rebound = next(r for r in receipt["runs"] if r["rebound"] is True)
    assert rebound["outcomes_sha256"] is None
    assert len(rebound[LR.LABEL_SOURCE_SHAKEY]) == 64


def _rewrite_receipt(out, mutate):
    receipt = json.loads((out / LI.RECEIPT_NAME).read_text(encoding="utf-8"))
    mutate(receipt)
    (out / LI.RECEIPT_NAME).write_text(json.dumps(receipt, indent=1),
                                       encoding="utf-8")


@pytest.mark.parametrize("mutate", [
    lambda r: r["label_rebind"]["rebound"][0].__setitem__("new_source", "native"),
    lambda r: r["label_rebind"]["rebound"][0].__setitem__(
        "new_policy", "native_fleet"),
    lambda r: next(x for x in r["runs"] if x.get("rebound")) .__setitem__(
        LR.LABEL_SOURCE_SHAKEY, "0" * 63),
    lambda r: next(x for x in r["runs"] if x.get("rebound")) .__setitem__(
        "resolved_source", "native"),
    lambda r: r["label_rebind"].__setitem__("n_rebound_runs", 2),
])
def test_rebind_block_tamper_refuses(env, mutate):
    _rewrite_receipt(env["out"], mutate)
    with pytest.raises(LI.LabelIsolationError):
        LI.load_train_outcomes(env["pack"], env["out"])


def test_tampered_rebound_parquet_refuses(env):
    with open(env["out"] / LI.SIDECAR_PARQUET_NAME, "ab") as fh:
        fh.write(b"\x00")
    with pytest.raises(LI.LabelIsolationError, match="sha256"):
        LI.load_train_outcomes(env["pack"], env["out"])


def test_missing_excluded_run_leaves_untouched_row_refuses(env):
    def mutate(r):
        for run in r["runs"]:
            if run.get("excluded"):
                run["excluded"] = False
                run["outcomes_sha256"] = "e" * 64
        r["label_rebind"]["excluded"] = []
        r["label_rebind"]["n_excluded_runs"] = 0

    _rewrite_receipt(env["out"], mutate)
    with pytest.raises(LI.LabelIsolationError):
        LI.load_train_outcomes(env["pack"], env["out"])


def test_builder_refuses_existing_output(env):
    with pytest.raises(LR.LabelRebindError, match="existing"):
        LR.build_rebound_sidecar(
            env["pack"], env["shards"], env["input_sidecar"], env["v04a"],
            env["out"], "deadbeef",
            _sha(env["input_sidecar"] / LI.RECEIPT_NAME),
            _sha(env["input_sidecar"] / LI.SIDECAR_PARQUET_NAME))
